from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import CFG, PROCESSED, load_tables  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HORIZON = CFG["simulation"]["horizon_days"]
REVIEW_CADENCE = CFG["review_policy"]["standard_cadence_days"]
Z_VALUES = {0.99: 2.326, 0.95: 1.645, 0.90: 1.282, 0.85: 1.036}


def load_inputs():
    tables = load_tables()
    as_of = pd.Timestamp(CFG["project"]["as_of_date"])
    horizon_dates = pd.date_range(as_of + pd.Timedelta(days=int(1)), periods=HORIZON)
    actuals = tables["demand_history"]
    actuals = actuals[actuals["date"].isin(horizon_dates)]

    # Live-ops fallback: days without revealed actuals get ensemble p50 as
    # scenario demand (identical for both policies -> fair comparison).
    missing = horizon_dates.difference(pd.DatetimeIndex(actuals["date"].unique()))
    if len(missing):
        sys.path.insert(0, str(ROOT / "db"))
        from inputs import load_forecasts  # noqa: E402
        fcst = load_forecasts()
        if len(fcst):
            scen = fcst[fcst["forecast_date"].isin(missing)][
                ["sku_id", "region", "forecast_date", "p50"]
            ].rename(columns={"forecast_date": "date", "p50": "units"})
            scen["atc_code"] = ""
            keep_cols = ["date", "sku_id", "atc_code", "region", "units"]
            base = actuals[[c for c in keep_cols if c in actuals.columns]].copy()
            actuals = pd.concat(
                [base, scen[[c for c in keep_cols if c in scen.columns]]],
                ignore_index=True,
            )
            print(f"[simulate] {len(missing)}/{HORIZON} horizon days lack actuals "
                  f"-> ensemble p50 scenario demand")

    skus = tables["sku_master"].set_index("sku_id")
    lanes = tables["lanes"]
    return tables, as_of, horizon_dates, actuals, skus, lanes


@dataclass
class Batch:
    qty: float
    expiry: pd.Timestamp
    unit_cost: int
    seq: int = 0


@dataclass
class SiteState:
    batches: dict[str, list[Batch]] = field(default_factory=dict)


def initial_sites(inv: pd.DataFrame) -> tuple[dict[str, dict[str, list[Batch]]], int]:
    sites: dict[str, dict[str, list[Batch]]] = {}
    inv = inv.sort_values("received_date")
    seq = 1
    for _, r in inv.iterrows():
        sites.setdefault(r["location"], {}).setdefault(r["sku_id"], []).append(
            Batch(
                qty=float(r["qty_units"]),
                expiry=r["expiry_date"],
                unit_cost=int(r["unit_cost_inr"]),
                seq=seq,
            )
        )
        seq += 1
    return sites, seq


def drop_expired(sites, today):
    written_off_units = 0
    written_off_value = 0
    for loc in sites:
        for sku, blist in sites[loc].items():
            keep = []
            for b in blist:
                if b.expiry < today:
                    if b.qty > 0:
                        written_off_units += b.qty
                        written_off_value += b.qty * b.unit_cost
                else:
                    keep.append(b)
            sites[loc][sku] = keep
    return written_off_units, written_off_value


def serve_demand(site_sku_batches, demand, expiry_blind):
    site_sku_batches.sort(key=lambda b: b.seq if expiry_blind else b.expiry)
    remaining = demand
    served = 0.0
    for b in site_sku_batches:
        if remaining <= 0:
            break
        take = min(b.qty, remaining)
        b.qty -= take
        served += take
        remaining -= take
    return served, max(remaining, 0)


def total_on_hand(sites, loc, sku):
    return sum(b.qty for b in sites.get(loc, {}).get(sku, []))


def simulate(policy: str):
    tables, as_of, horizon_dates, actuals, skus, lanes = load_inputs()
    inv = tables["inventory_batches"]
    sites, seq_counter = initial_sites(inv)

    if policy == "proposed":
        sys.path.insert(0, str(ROOT / "db"))
        from inputs import load_forecasts, load_transfer_plan  # noqa: E402
        fcst = load_forecasts()
        mu_tbl = fcst.groupby(["sku_id", "region"])["p50"].mean()
        sigma_tbl = (
            fcst.groupby(["sku_id", "region"])
            .apply(lambda g: ((g["p90"] - g["p10"]).mean() / (2 * 1.2816)), include_groups=False)
        )
    else:
        hist = tables["demand_history"]
        hist = hist[(hist["date"] > as_of - pd.Timedelta(days=int(7))) & (hist["date"] <= as_of)]
        mu_tbl = hist.groupby(["sku_id", "region"])["units"].mean()
        h28 = tables["demand_history"][
            (tables["demand_history"]["date"] > as_of - pd.Timedelta(days=int(35)))
            & (tables["demand_history"]["date"] <= as_of)
        ]
        sigma_tbl = h28.groupby(["sku_id", "region"])["units"].std().fillna(0)

    repl_lead = (
        lanes[lanes["mode"] == "replenish"]
        .set_index("to_location")["lead_time_days"]
        .to_dict()
    )
    crit_map = skus["criticality"].to_dict()
    cost_map = skus["unit_cost_inr"].to_dict()

    incoming_orders: dict[tuple, list[tuple[int, float]]] = {}

    def place_order(day_idx, sku, region, mu, sigma):
        z = Z_VALUES[round(CFG["service_levels"].get(crit_map.get(sku, "standard"), 0.9), 2)]
        L = repl_lead.get(region, 14)
        ss = z * sigma * np.sqrt(L)
        target = mu * (L + REVIEW_CADENCE) + ss
        pos = total_on_hand(sites, region, sku)
        pending = sum(q for d, q in incoming_orders.get((sku, region), []) if d > day_idx)
        gap = target - pos - pending
        if gap > 0:
            incoming_orders.setdefault((sku, region), []).append((day_idx + L, gap))

    transfer_in: dict[tuple, list[tuple[int, float]]] = {}
    if policy == "proposed":
        sys.path.insert(0, str(ROOT / "db"))
        from inputs import load_transfer_plan  # noqa: E402
        tp = load_transfer_plan()
        if len(tp):
            for _, t in tp.iterrows():
                src_list = sites.get(t["from_location"], {}).get(t["sku_id"], [])
                moved = 0.0
                need = float(t["qty_units"])
                for b in sorted(src_list, key=lambda x: x.expiry):
                    take = min(b.qty, need - moved)
                    b.qty -= take
                    moved += take
                    if moved >= need:
                        break
                if moved > 0:
                    transfer_in.setdefault((t["sku_id"], t["to_location"]), []).append(
                        (int(t["transfer_lead_days"]), moved, t["expiry_date"])
                    )

    sku_region_pairs = [(s, l) for l in sites for s in sites[l]]
    all_pairs = set(zip(actuals["sku_id"], actuals["region"])) | set(sku_region_pairs)

    review_days = set(range(0, HORIZON, REVIEW_CADENCE))
    log_rows = []

    for day_idx, today in enumerate(horizon_dates):
        wu, wv = drop_expired(sites, today)

        for (sku, region), arr in list(incoming_orders.items()):
            arrived = [(d, q) for d, q in arr if d <= day_idx]
            if arrived:
                qty = sum(q for _, q in arrived)
                seq_counter += 1
                sites.setdefault(region, {}).setdefault(sku, []).append(
                    Batch(
                        qty=qty,
                        expiry=today + pd.Timedelta(days=int(540)),
                        unit_cost=int(cost_map.get(sku, 10)),
                        seq=seq_counter,
                    )
                )
                incoming_orders[(sku, region)] = [
                    (d, q) for d, q in arr if d > day_idx
                ]

        for (sku, region), arr in list(transfer_in.items()):
            due = [(d, q, e) for d, q, e in arr if d <= day_idx]
            if due:
                for d, q, e in due:
                    seq_counter += 1
                    sites.setdefault(region, {}).setdefault(sku, []).append(
                        Batch(
                            qty=q,
                            expiry=pd.Timestamp(e),
                            unit_cost=int(cost_map.get(sku, 10)),
                            seq=seq_counter,
                        )
                    )
                transfer_in[(sku, region)] = [
                    (d, q, e) for d, q, e in arr if d > day_idx
                ]

        day_actuals = actuals[actuals["date"] == today]
        dem_map = {
            (r["sku_id"], r["region"]): float(r["units"]) for _, r in day_actuals.iterrows()
        }

        for pair in all_pairs:
            sku, region = pair
            demand = dem_map.get(pair, 0.0)
            blist = sites.get(region, {}).get(sku, [])
            served, unmet = serve_demand(blist, demand, expiry_blind=(policy == "status_quo"))
            end_inv = sum(b.qty for b in blist)

            if policy == "proposed":
                mu = float(mu_tbl.get(pair, 0.0)) if hasattr(mu_tbl, "get") else 0.0
                sigma = float(sigma_tbl.get(pair, 0.0)) if hasattr(sigma_tbl, "get") else 0.0
            else:
                mu = float(mu_tbl.get(pair, 0.0))
                sigma = float(sigma_tbl.get(pair, 0.0))
            if day_idx in review_days:
                place_order(day_idx, sku, region, mu, sigma)

            log_rows.append(
                {
                    "policy": policy,
                    "date": today,
                    "sku_id": sku,
                    "region": region,
                    "criticality": crit_map.get(sku, "standard"),
                    "demand": demand,
                    "fulfilled": served,
                    "unfulfilled": unmet,
                    "expired_units": wu,
                    "expired_value_inr": wv,
                    "ending_inventory": end_inv,
                }
            )
    df = pd.DataFrame(log_rows)
    exp = df.groupby(["policy", "date"]).agg(
        expired_units=("expired_units", "first"), expired_value_inr=("expired_value_inr", "first")
    ).reset_index()
    df = df.drop(columns=["expired_units", "expired_value_inr"]).merge(exp, on=["policy", "date"], how="left")
    return df


def kpis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pol, g in df.groupby("policy"):
        crit = g[g["criticality"].isin(["critical"])]
        rows.append(
            {
                "policy": pol,
                "fill_rate_pct": round(100 * g["fulfilled"].sum() / max(g["demand"].sum(), 1), 2),
                "critical_fill_rate_pct": round(
                    100 * crit["fulfilled"].sum() / max(crit["demand"].sum(), 1), 2
                ),
                "stockout_units": int(g["unfulfilled"].sum()),
                "critical_stockout_sitedays": int((crit[crit["unfulfilled"] > 0]).shape[0]),
                "writeoff_value_inr": int(
                    g.groupby("date")["expired_value_inr"].first().sum()
                ),
                "avg_ending_inventory": round(g["ending_inventory"].mean(), 0),
            }
        )
    return pd.DataFrame(rows)


def main():
    frames = []
    for policy in ["status_quo", "proposed"]:
        print(f"simulating {policy} ...")
        frames.append(simulate(policy))
    sim = pd.concat(frames, ignore_index=True)

    k = kpis(sim)

    # write simulation + KPIs to PostgreSQL OUTPUT tables
    try:
        sys.path.insert(0, str(ROOT / "db"))
        import outputs as db_out

        db_out.write_simulation(sim, k, db_out.current_as_of())
    except Exception as exc:
        print(f"[simulate] DB write failed: {type(exc).__name__}: {exc}")

    print(k.to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(CFG["project"]["seed"])

ATC_CODES = list(CFG["sku"]["brands"].keys())


def load_base_series() -> pd.DataFrame:
    from ingest import ATC_CODES, load_any_csv

    path = ROOT / "data" / "raw" / "salesdaily.csv"
    data, _ = load_any_csv(path)
    data = data.set_index("date").sort_index()
    missing = [c for c in ATC_CODES if c not in data.columns]
    if missing:
        raise ValueError(f"installed dataset lacks categories: {missing}")
    return data[ATC_CODES].astype(float)


def build_sku_master() -> pd.DataFrame:
    rows = []
    for atc in ATC_CODES:
        brands = CFG["sku"]["brands"][atc]
        shares = RNG.dirichlet(np.ones(len(brands)) * 2.0)
        cost_lo, cost_hi = CFG["sku"]["cost_range_inr"][atc]
        shelf_range = CFG["sku"]["shelf_life_days"].get(
            atc, CFG["sku"]["shelf_life_days"]["default"]
        )
        crit_map = {"critical": 0, "high": 1, "standard": 2, "low": 3}
        order = sorted(range(len(brands)), key=lambda i: -shares[i])
        crit_rank = {}
        base_crit = CFG["sku"]["criticality"][atc]
        for pos, idx in enumerate(order):
            crit_rank[idx] = min(crit_map[base_crit] + (1 if pos >= 2 else 0), 3)
        inv_crit = {v: k for k, v in crit_map.items()}
        for i, brand in enumerate(brands):
            rows.append(
                {
                    "sku_id": f"{atc}-{i+1:02d}",
                    "brand_name": brand,
                    "atc_code": atc,
                    "category_share": round(float(shares[i]), 4),
                    "criticality": inv_crit[crit_rank[i]],
                    "unit_cost_inr": int(RNG.integers(cost_lo, cost_hi)),
                    "shelf_life_days": int(RNG.integers(shelf_range[0], shelf_range[1])),
                }
            )
    return pd.DataFrame(rows)


def build_locations() -> pd.DataFrame:
    rows = []
    for loc in CFG["locations"]:
        rows.append(
            {
                "location_id": loc["id"],
                "name": loc["name"],
                "type": loc["type"],
                "demand_share": loc["demand_share"],
                "capacity_units": 0,
            }
        )
    df = pd.DataFrame(rows)
    total_share = df["demand_share"].sum()
    df["demand_share"] = (df["demand_share"] / total_share).round(4)
    return df


def build_lanes(locations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metro_ids = locations.loc[locations["type"] == "metro_dc", "location_id"]
    tier2_ids = locations.loc[locations["type"] == "tier2_wh", "location_id"]
    lo_m, hi_m = CFG["lanes"]["supplier_to_metro_lead_days"]
    lo_t, hi_t = CFG["lanes"]["supplier_to_tier2_lead_days"]
    for mid in metro_ids:
        rows.append(
            {
                "from_location": "SUPPLIER",
                "to_location": mid,
                "mode": "replenish",
                "lead_time_days": int(RNG.integers(lo_m, hi_m)),
            }
        )
    for tid in tier2_ids:
        rows.append(
            {
                "from_location": "SUPPLIER",
                "to_location": tid,
                "mode": "replenish",
                "lead_time_days": int(RNG.integers(lo_t, hi_t)),
            }
        )
    for t in CFG["lanes"]["transfers"]:
        rows.append(
            {
                "from_location": t["from"],
                "to_location": t["to"],
                "mode": "transfer",
                "lead_time_days": t["lead_days"],
            }
        )
    return pd.DataFrame(rows)


def build_distributors() -> pd.DataFrame:
    lo, hi = CFG["distributors"]["order_cycle_days_range"]
    rows = []
    for loc in CFG["locations"]:
        rows.append(
            {
                "distributor_id": f"DST_{loc['id']}",
                "region": loc["id"],
                "order_cycle_days": int(RNG.integers(lo, hi)),
                "order_size_sigma": CFG["distributors"]["size_noise_sigma"],
            }
        )
    return pd.DataFrame(rows)


def expand_promos(dates: pd.DatetimeIndex) -> pd.DataFrame:
    years = sorted(set(dates.year) | set(dates.year + 1))
    rows = []
    for p in CFG["promos"]:
        sm, sd = map(int, p["start_mmdd"].split("-"))
        em, ed = map(int, p["end_mmdd"].split("-"))
        for y in years:
            start = pd.Timestamp(year=y, month=sm, day=sd)
            end = pd.Timestamp(year=y, month=em, day=ed)
            if em < sm:
                end = pd.Timestamp(year=y + 1, month=em, day=ed)
            if end < dates.min() or start > dates.max():
                continue
            rows.append(
                {
                    "promo_id": p["id"],
                    "name": p["name"],
                    "start_date": max(start, dates.min()),
                    "end_date": min(end, dates.max()),
                    "uplift": p["uplift"],
                    "regions": ",".join(p["regions"]),
                }
            )
    return pd.DataFrame(rows).sort_values("start_date").reset_index(drop=True)


def flu_bump(dates: pd.DatetimeIndex) -> np.ndarray:
    peak_md = CFG["flu_season"]["peak_month_day"]
    sigma = CFG["flu_season"]["sigma_days"]
    vals = []
    for d in dates:
        year = d.year if d.month >= 6 else d.year
        center = pd.Timestamp(f"{year}-{peak_md}")
        if (d - center).days < -120:
            center = pd.Timestamp(f"{year + 1}-{peak_md}")
        elif (d - center).days > 120:
            center = pd.Timestamp(f"{year - 1}-{peak_md}")
        vals.append(np.exp(-(((d - center).days) ** 2) / (2 * sigma**2)))
    return np.array(vals)


def build_flu_index(dates: pd.DatetimeIndex) -> pd.DataFrame:
    shift = CFG["flu_season"]["lead_shift_days"]
    shifted_dates = dates - pd.Timedelta(days=shift)
    base_shape = flu_bump(shifted_dates)
    rows = []
    noise_state = 0.0
    phi = 0.90
    for i, d in enumerate(dates):
        noise_state = phi * noise_state + (1 - phi) * RNG.normal(0, 8)
        for loc in CFG["locations"]:
            region_offset = 15 if loc["type"] == "tier2_wh" else 0
            idx_val = (
                100 * base_shape[i]
                + region_offset * base_shape[i]
                + abs(noise_state)
                + 2
            )
            rows.append({"date": d, "region": loc["id"], "flu_index": round(idx_val, 2)})
    return pd.DataFrame(rows)


def promo_multiplier_lookup(promos: pd.DataFrame):
    active = []
    for _, r in promos.iterrows():
        regions = (
            [loc["id"] for loc in CFG["locations"]]
            if r["regions"] == "ALL"
            else r["regions"].split(",")
        )
        active.append((r["start_date"], r["end_date"], set(regions), r["uplift"]))
    return active


def build_demand_history(base: pd.DataFrame, skus: pd.DataFrame) -> pd.DataFrame:
    dates = base.index
    scale = CFG["project"]["volume_scale"]
    promos = expand_promos(dates)
    promo_active = promo_multiplier_lookup(promos)

    bump = flu_bump(dates)
    flu_cfg = CFG["flu_season"]
    sensitive_mask = {a: (a in flu_cfg["sensitive_atc"]) for a in ATC_CODES}

    sku_by_atc = {a: skus[skus["atc_code"] == a] for a in ATC_CODES}
    frames = []
    for atc in ATC_CODES:
        series = base[atc].to_numpy(dtype=float)
        cat_skus = sku_by_atc[atc]
        for _, sku in cat_skus.iterrows():
            for loc in CFG["locations"]:
                uplift_peak = (
                    flu_cfg["tier2_peak_uplift"]
                    if loc["type"] == "tier2_wh"
                    else flu_cfg["metro_peak_uplift"]
                )
                mult = np.where(
                    sensitive_mask[atc], 1.0 + uplift_peak * bump, 1.0
                )
                units = (
                    series
                    * scale
                    * float(sku["category_share"])
                    * loc["demand_share"]
                    * mult
                )
                noise = RNG.lognormal(mean=0.0, sigma=0.08, size=len(dates))
                units = units * noise

                promo_mult = np.ones(len(dates))
                for start, end, regions, upl in promo_active:
                    if loc["id"] not in regions:
                        continue
                    mask = (dates >= start) & (dates <= end)
                    promo_mult[mask] = np.maximum(promo_mult[mask], 1.0 + upl)
                units = units * promo_mult

                frames.append(
                    pd.DataFrame(
                        {
                            "date": dates,
                            "sku_id": sku["sku_id"],
                            "atc_code": atc,
                            "region": loc["id"],
                            "units": np.maximum(np.round(units), 0).astype(int),
                        }
                    )
                )
    df = pd.concat(frames, ignore_index=True)
    return df


def recent_avg_demand(demand: pd.DataFrame, as_of: pd.Timestamp, window: int = 30):
    hist = demand[demand["date"] < as_of]
    cutoff = as_of - pd.Timedelta(days=int(window))
    recent = hist[hist["date"] >= cutoff]
    avg = recent.groupby(["sku_id", "region"])["units"].mean()
    return avg


def build_inventory_batches(
    demand: pd.DataFrame, skus: pd.DataFrame, as_of: pd.Timestamp
) -> pd.DataFrame:
    inv = CFG["inventory"]
    avg = recent_avg_demand(demand, as_of)
    skus_idx = skus.set_index("sku_id")
    rows = []
    batch_counter = 0
    for (sku_id, region), daily_avg in avg.items():
        atc = skus_idx.loc[sku_id, "atc_code"]
        location_type = next(l["type"] for l in CFG["locations"] if l["id"] == region)
        unit_cost = int(skus_idx.loc[sku_id, "unit_cost_inr"])

        slow_pathology = (
            location_type == "metro_dc" and atc in inv["metro_slow_atc"]
        )
        shortage_pathology = (
            location_type == "tier2_wh" and atc in inv["tier2_critical_atc"]
        )

        if slow_pathology:
            cover_lo, cover_hi = inv["metro_slow_cover_days"]
            exp_lo, exp_hi = inv["metro_slow_expiry_days"]
            n_batches = 3
        elif shortage_pathology:
            cover_lo, cover_hi = inv["tier2_critical_cover_days"]
            exp_lo, exp_hi = inv["expiry_horizon_days"]
            if RNG.random() < 0.30:
                rows.append(
                    {
                        "batch_id": f"B{batch_counter:05d}",
                        "sku_id": sku_id,
                        "location": region,
                        "qty_units": 0,
                        "expiry_date": as_of + pd.Timedelta(days=int(180)),
                        "received_date": as_of - pd.Timedelta(days=int(30)),
                        "unit_cost_inr": unit_cost,
                        "status": "stockout",
                    }
                )
                batch_counter += 1
                continue
            n_batches = int(RNG.integers(1, 3))
        else:
            cover_lo, cover_hi = inv["healthy_cover_days"]
            exp_lo, exp_hi = inv["expiry_horizon_days"]
            n_batches = int(RNG.integers(*inv["batches_per_location"]))

        total_units = daily_avg * RNG.integers(cover_lo, cover_hi)
        if total_units <= 0 or n_batches == 0:
            continue
        weights = RNG.dirichlet(np.ones(n_batches) * 3.0)
        split = np.round(total_units * weights).astype(int)
        for qty in split:
            expiry_days = int(RNG.integers(exp_lo, exp_hi))
            received_days = int(RNG.integers(20, 120))
            status = (
                "near_expiry_risk"
                if slow_pathology
                else ("healthy" if expiry_days > 110 else "watch")
            )
            rows.append(
                {
                    "batch_id": f"B{batch_counter:05d}",
                    "sku_id": sku_id,
                    "location": region,
                    "qty_units": int(qty),
                    "expiry_date": as_of + pd.Timedelta(days=int(expiry_days)),
                    "received_date": as_of - pd.Timedelta(days=int(received_days)),
                    "unit_cost_inr": unit_cost,
                    "status": status,
                }
            )
            batch_counter += 1
    df = pd.DataFrame(rows)
    return df[df["qty_units"] >= 0].reset_index(drop=True)


def finalize_capacity(inventory: pd.DataFrame) -> pd.DataFrame:
    totals = inventory.groupby("location")["qty_units"].sum()
    target = CFG["capacity_headroom_target"]
    rows = []
    for loc in CFG["locations"]:
        lid = loc["id"]
        used = int(totals.get(lid, 0))
        capacity = int(used / target) if used > 0 else 100000
        rows.append(
            {
                "location_id": lid,
                "name": loc["name"],
                "type": loc["type"],
                "demand_share": loc["demand_share"],
                "capacity_units": capacity,
            }
        )
    return pd.DataFrame(rows)


def main():
    as_of = pd.Timestamp(CFG["project"]["as_of_date"])
    base = load_base_series()

    skus = build_sku_master()
    demand = build_demand_history(base, skus)
    inventory = build_inventory_batches(demand, skus, as_of)
    locations = finalize_capacity(inventory)
    lanes = build_lanes(locations)
    distributors = build_distributors()
    promos = expand_promos(base.index)
    flu_index = build_flu_index(base.index)

    meta = {
        "as_of_date": str(as_of.date()),
        "history_start": str(base.index.min().date()),
        "history_end": str(base.index.max().date()),
        "n_skus": len(skus),
        "n_locations": len(locations),
        "n_demand_rows": len(demand),
        "inventory_value_inr": int(
            (inventory["qty_units"] * inventory["unit_cost_inr"]).sum()
        ),
        "near_expiry_batches": int(
            (
                inventory["expiry_date"]
                <= as_of + pd.Timedelta(days=int(110))
            ).sum()
        ),
    }

    skus.to_parquet(OUT / "sku_master.parquet", index=False)
    locations.to_parquet(OUT / "locations.parquet", index=False)
    lanes.to_parquet(OUT / "lanes.parquet", index=False)
    distributors.to_parquet(OUT / "distributors.parquet", index=False)
    promos.to_parquet(OUT / "promo_calendar.parquet", index=False)
    flu_index.to_parquet(OUT / "flu_index.parquet", index=False)
    inventory.to_parquet(OUT / "inventory_batches.parquet", index=False)
    demand.to_parquet(OUT / "demand_history.parquet", index=False)
    (OUT / "meta.json").write_text(yaml.safe_dump(meta))

    print(yaml.safe_dump(meta))
    print(demand.groupby("atc_code")["units"].sum().sort_values(ascending=False))
    print("\nnear-expiry value by location:")
    ne = inventory[inventory["expiry_date"] <= as_of + pd.Timedelta(days=int(110))].copy()
    ne["value_inr"] = ne["qty_units"] * ne["unit_cost_inr"]
    print(ne.groupby("location")["value_inr"].sum())


if __name__ == "__main__":
    main()

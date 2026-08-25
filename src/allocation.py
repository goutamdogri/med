from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import CFG, PROCESSED, load_tables  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def shortage_rescue_transfers(inv, repl, mu):
    need_rows = repl[repl["status"].isin(["stockout_risk", "low"])]
    transfer_lanes = lanes_df = load_tables()["lanes"]
    transfer_lanes = transfer_lanes[transfer_lanes["mode"] == "transfer"]
    lanes_by_dest = {
        dest: (src, lead)
        for src, dest, lead in zip(
            transfer_lanes["from_location"],
            transfer_lanes["to_location"],
            transfer_lanes["lead_time_days"],
        )
        if src.startswith("DC_")
    }
    onhand = inv.groupby(["sku_id", "location"])["qty_units"].sum()
    rescues = []
    for _, r in need_rows.iterrows():
        key = (r["sku_id"], r["region"])
        lane = lanes_by_dest.get(r["region"])
        if not lane:
            continue
        src, lead = lane
        src_key = (r["sku_id"], src)
        oh_src = float(onhand.get(src_key, 0))
        mu_src = float(mu.get(src_key, 0.0))
        mu_dst = float(mu.get(key, 0.0)) or float(r["mu_daily"])
        if mu_src <= 0 or mu_dst <= 0:
            continue
        src_dos = oh_src / mu_src
        min_keep_days = max(15.0, r["lead_time_days"])
        giveable_units = oh_src - mu_src * min_keep_days
        dest_gap = max(0.0, float(r["target_position"]) - float(r["on_hand"]))
        qty = min(giveable_units, dest_gap, mu_dst * 60)
        if qty < 10 or src_dos < min_keep_days + 5:
            continue
        rescues.append(
            {
                "batch_id": None,
                "sku_id": r["sku_id"],
                "from_location": src,
                "to_location": r["region"],
                "qty_units": int(qty),
                "expiry_date": pd.NaT,
                "days_to_expiry": np.nan,
                "transfer_lead_days": int(lead),
                "value_saved_inr": int(qty * 0),
                "reason": "shortage_rescue",
                "src_days_of_supply_before": round(src_dos, 1),
            }
        )
    return pd.DataFrame(rescues)


def build_allocation_plan() -> tuple[pd.DataFrame, pd.DataFrame]:
    as_of = pd.Timestamp(CFG["project"]["as_of_date"])
    tables = load_tables()
    inv = tables["inventory_batches"]
    lanes = tables["lanes"]
    sys.path.insert(0, str(ROOT / "db"))
    from inputs import load_forecasts, load_replenishment  # noqa: E402
    fcst = load_forecasts()
    repl = load_replenishment()

    mu = fcst.groupby(["sku_id", "region"])["p50"].mean()
    inv["days_to_expiry"] = (inv["expiry_date"] - as_of).dt.days
    inv["mu_here"] = [mu.get((s, l), 0.0) for s, l in zip(inv["sku_id"], inv["location"])]
    inv["sellable_before_expiry"] = np.minimum(
        inv["qty_units"], inv["mu_here"] * inv["days_to_expiry"].clip(lower=0)
    )
    inv["leftover"] = (
        inv["qty_units"] - inv["sellable_before_expiry"]
    ).clip(lower=0)
    inv["risk_value_inr"] = inv["leftover"] * inv["unit_cost_inr"]

    at_risk = inv[(inv["leftover"] > 5) & (inv["days_to_expiry"] <= 180)].copy()

    need = repl.set_index(["sku_id", "region"])
    transfer_lanes = lanes[lanes["mode"] == "transfer"]

    transfers = []
    at_risk = at_risk.sort_values("risk_value_inr", ascending=False)

    dest_cache = {
        (f, t): ld
        for f, t, ld in zip(
            transfer_lanes["from_location"],
            transfer_lanes["to_location"],
            transfer_lanes["lead_time_days"],
        )
    }

    for _, b in at_risk.iterrows():
        remaining_leftover = b["leftover"]
        src_dos = (
            b["qty_units"] / b["mu_here"] if b["mu_here"] > 0 else np.inf
        )
        candidates = [
            (dest, lead_d)
            for (src, dest), lead_d in dest_cache.items()
            if src == b["location"]
        ]
        scored = []
        for dest, lead_d in candidates:
            key = (b["sku_id"], dest)
            if key not in need.index:
                continue
            row = need.loc[key]
            gap = max(0.0, row["target_position"] - row["on_hand"])
            mu_dest = float(mu.get(key, 0.0))
            dest_dos = row["on_hand"] / mu_dest if mu_dest > 0 else np.inf
            if dest_dos >= src_dos:
                continue
            consumable_before_expiry = mu_dest * b["days_to_expiry"]
            usable = min(
                remaining_leftover,
                max(gap, consumable_before_expiry * 0.8),
            )
            if usable > 5:
                scored.append(
                    {
                        "dest": dest,
                        "lead": lead_d,
                        "usable": usable,
                        "score": usable * b["unit_cost_inr"],
                    }
                )
        scored.sort(key=lambda x: -x["score"])
        for c in scored:
            if remaining_leftover <= 5:
                break
            qty = int(min(c["usable"], remaining_leftover))
            if qty <= 5:
                continue
            transfers.append(
                {
                    "batch_id": b["batch_id"],
                    "sku_id": b["sku_id"],
                    "from_location": b["location"],
                    "to_location": c["dest"],
                    "qty_units": qty,
                    "expiry_date": b["expiry_date"],
                    "days_to_expiry": int(b["days_to_expiry"]),
                    "transfer_lead_days": c["lead"],
                    "value_saved_inr": int(qty * b["unit_cost_inr"]),
                }
            )
            remaining_leftover -= qty

    transfer_df = pd.DataFrame(transfers)
    if len(transfer_df) == 0:
        transfer_df = pd.DataFrame(
            columns=[
                "batch_id",
                "sku_id",
                "from_location",
                "to_location",
                "qty_units",
                "expiry_date",
                "days_to_expiry",
                "transfer_lead_days",
                "value_saved_inr",
            ]
        )
    transfer_df["reason"] = "expiry_rescue"

    rescues = shortage_rescue_transfers(inv, repl, mu)
    if len(rescues) > 0:
        transfer_df = pd.concat([transfer_df, rescues], ignore_index=True)

    transferred_by_batch = (
        transfer_df.groupby("batch_id")["qty_units"].sum()
        if len(transfer_df)
        else pd.Series(dtype=int)
    )

    def residual(row):
        moved = int(transferred_by_batch.get(row["batch_id"], 0))
        return max(0.0, row["leftover"] - moved)

    at_risk["residual_writeoff_units"] = at_risk.apply(residual, axis=1)
    at_risk["residual_value_inr"] = (
        at_risk["residual_writeoff_units"] * at_risk["unit_cost_inr"]
    )

    writeoffs = at_risk[at_risk["residual_writeoff_units"] > 0][
        [
            "batch_id",
            "sku_id",
            "location",
            "qty_units",
            "leftover",
            "residual_writeoff_units",
            "unit_cost_inr",
            "residual_value_inr",
            "expiry_date",
            "days_to_expiry",
        ]
    ].sort_values("residual_value_inr", ascending=False)

    fefo_order = inv.sort_values(["location", "sku_id", "expiry_date"])[
        ["location", "sku_id", "batch_id", "expiry_date", "qty_units", "status"]
    ]

    return transfer_df, writeoffs, fefo_order


def main():
    transfers, writeoffs, _ = build_allocation_plan()

    # write plans to PostgreSQL OUTPUT tables
    try:
        sys.path.insert(0, str(ROOT / "db"))
        import outputs as db_out

        as_of = db_out.current_as_of()
        db_out.write_transfer_plan(transfers, as_of)
        db_out.write_writeoff_risk(writeoffs, as_of)
    except Exception as exc:
        print(f"[allocation] DB write failed: {type(exc).__name__}: {exc}")

    print(f"transfers planned: {len(transfers)} | units moved: {transfers.qty_units.sum():,.0f}")
    print(f"value rescued: ₹{transfers.value_saved_inr.sum():,.0f}")
    print("\nsample transfers:")
    print(transfers.head(8).to_string(index=False))
    print(f"\npredicted residual write-offs after action: ₹{writeoffs.residual_value_inr.sum():,.0f}")
    print("(baseline write-off exposure was the full 'leftover' pool)")


if __name__ == "__main__":
    main()

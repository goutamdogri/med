"""
simulate_ingest_day.py — DEMO helper that mimics the production backend inserting
a new day of data into medcare.

Each run reveals exactly ONE more day from the pre-generated history
(data/processed/demand_history.parquet extends beyond what was seeded):
  1. demand_history       += that day's sales (all SKUs x regions)
  2. disease_burden_index += that day's ILI readings
  3. inventory_batches    += a fresh as_of_date snapshot, depleted FEFO by the
                             day's sales, with recomputed status flags

In production these inserts come from the real backend instead — this script is
only used so the daily cron can demonstrate live operation offline.

Usage:
    .venv/bin/python db/simulate_ingest_day.py            # reveal next day
    .venv/bin/python db/simulate_ingest_day.py --date 2019-03-02
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from connection import get_engine, insert_df, read_sql, scalar  # noqa: E402
from sqlalchemy import text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def reveal_demand_and_flu(engine, day: pd.Timestamp) -> bool:
    """Returns False if nothing left to reveal."""
    parquet_max = pd.read_parquet(PROCESSED / "demand_history.parquet")["date"].max()
    parquet_max = pd.Timestamp(parquet_max)
    if day > parquet_max:
        print(f"no unrevealed data left (generated history ends {parquet_max.date()})")
        return False

    dh = pd.read_parquet(PROCESSED / "demand_history.parquet")
    day_rows = dh[pd.to_datetime(dh["date"]) == day].copy()
    if day_rows.empty:
        print(f"no demand rows for {day.date()} in generated history")
        return False
    day_rows["ingested_at"] = pd.Timestamp.now()
    n = insert_df(day_rows[["date", "sku_id", "atc_code", "region", "units", "ingested_at"]],
                  "demand_history")

    flu = pd.read_parquet(PROCESSED / "flu_index.parquet")
    frows = flu[pd.to_datetime(flu["date"]) == day].rename(
        columns={"flu_index": "index_value", "date": "record_date"}
    )
    if not frows.empty:
        # idempotent: surveillance rows usually already exist (public feed seeded ahead)
        from fill_derived import upsert_df

        frows["index_type"] = "ili"
        frows["source"] = "IDSP_State_Surveillance"
        frows["source_lag_days"] = 3
        frows["ingested_at"] = pd.Timestamp.now()
        upsert_df(
            frows,
            "disease_burden_index",
            ["record_date", "region", "index_value", "index_type",
             "source", "source_lag_days", "ingested_at"],
        )
    print(f"revealed {day.date()}: {n} demand rows, {len(frows)} flu rows")
    return True


def roll_inventory_snapshot(engine, day: pd.Timestamp) -> None:
    prev_date = scalar(
        "SELECT MAX(as_of_date) FROM inventory_batches WHERE as_of_date < :d",
        {"d": day.date()},
    )
    inv = read_sql(
        "SELECT batch_id, sku_id, location, qty_units, expiry_date, received_date, "
        "unit_cost_inr, grn_number FROM inventory_batches WHERE as_of_date = :d",
        {"d": prev_date},
    )
    if inv.empty:
        return
    inv["expiry_date"] = pd.to_datetime(inv["expiry_date"])

    # today's sold units per (sku, region)
    sold = read_sql(
        "SELECT sku_id, region, SUM(units) AS units FROM demand_history "
        "WHERE date = :d GROUP BY sku_id, region",
        {"d": day.date()},
    ).set_index(["sku_id", "region"])["units"].to_dict()

    # trailing 28-day mean demand for cover/status computation
    mu28 = read_sql("""
        SELECT sku_id, region, AVG(units) AS mu FROM (
            SELECT sku_id, region, units FROM demand_history
            WHERE date > (:d::date - INTERVAL '28 days') AND date <= :d
        ) t GROUP BY sku_id, region""", {"d": day.date()})
    mu_map = mu28.set_index(["sku_id", "region"])["mu"].to_dict()

    inv["sold_today"] = [sold.get((s, l), 0)
                         for s, l in zip(inv["sku_id"], inv["location"])]

    def fefo_deplete(g: pd.DataFrame) -> pd.Series:
        """Sell from soonest-expiring batch first within one (sku, location)."""
        remaining = int(g["sold_today"].iloc[0])
        out = []
        for qty in g["qty_units"]:
            take = min(int(qty), max(remaining, 0))
            out.append(int(qty) - take)
            remaining -= take
        return pd.Series(out, index=g.index)

    inv["new_qty"] = (
        inv.sort_values("expiry_date")
        .groupby(["sku_id", "location"], group_keys=False)
        .apply(fefo_deplete)
    )

    def status_for(sku: str, loc: str, oh: float, min_dte: int) -> str:
        if oh <= 0:
            return "stockout"
        mu = float(mu_map.get((sku, loc), 0.0))
        cover = oh / mu if mu > 0 else 999.0
        if min_dte <= 60:
            return "near_expiry_risk"
        if cover > 120:
            return "watch"
        return "healthy"

    grp = inv.groupby(["sku_id", "location"]).agg(
        on_hand=("new_qty", "sum"), min_expiry=("expiry_date", "min")
    ).reset_index()
    grp["status"] = [
        status_for(r["sku_id"], r["location"], r["on_hand"],
                   int((r["min_expiry"] - day).days))
        for _, r in grp.iterrows()
    ]
    inv = inv.drop(columns=["sold_today"]).merge(
        grp[["sku_id", "location", "status"]], on=["sku_id", "location"]
    )

    out = pd.DataFrame({
        "as_of_date":     day.date(),
        "batch_id":       inv["batch_id"],
        "sku_id":         inv["sku_id"],
        "location":       inv["location"],
        "qty_units":      inv["new_qty"].astype(int),
        "expiry_date":    inv["expiry_date"].dt.date,
        "received_date":  pd.to_datetime(inv["received_date"]).dt.date,
        "unit_cost_inr":  inv["unit_cost_inr"],
        "status":         inv["status"],
        "grn_number":     inv["grn_number"],
    })
    insert_df(out, "inventory_batches")
    print(f"inventory snapshot rolled to {day.date()}: "
          f"{int(out['qty_units'].sum()):,} units on hand")


def main():
    ap = argparse.ArgumentParser(description="Reveal one more simulated day into pharma_sc")
    ap.add_argument("--date", type=str, default=None,
                    help="specific day to reveal (YYYY-MM-DD); default = next after latest ingested")
    args = ap.parse_args()

    engine = get_engine()
    if args.date:
        day = pd.Timestamp(args.date)
    else:
        last = scalar("SELECT MAX(date) FROM demand_history")
        day = pd.Timestamp(last) + pd.Timedelta(days=1) if last else \
            pd.read_parquet(PROCESSED / "demand_history.parquet")["date"].min()

    if not reveal_demand_and_flu(engine, day):
        sys.exit(0)
    roll_inventory_snapshot(engine, day)


if __name__ == "__main__":
    main()

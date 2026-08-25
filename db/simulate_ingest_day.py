"""
simulate_ingest_day.py — Promote one day's data from staging tables → live tables.

Reads pipeline_state.simulated_today to determine which day to promote.
Does NOT advance the date (the backend handles that).

Promotes from:
  demand_history_staging       → demand_history
  disease_burden_staging       → disease_burden_index
  inventory_staging            → inventory_batches (weekly snapshots)
  distributor_orders_staging   → distributor_orders
  warehouse_capacity_staging   → warehouse_capacity_log

Usage:
    .venv/bin/python db/simulate_ingest_day.py             # promote simulated_today
    .venv/bin/python db/simulate_ingest_day.py --date 2019-03-02  # promote specific day
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from connection import get_engine, insert_df, read_sql, scalar  # noqa: E402
from pipeline_state import get_simulated_today  # noqa: E402
from sqlalchemy import text  # noqa: E402


def promote_demand(engine, day: pd.Timestamp) -> int:
    """Promote demand rows for the given day from staging → live."""
    rows = read_sql(
        "SELECT date, sku_id, atc_code, region, units "
        "FROM demand_history_staging WHERE date = :d",
        {"d": day.date()},
    )
    if rows.empty:
        return 0
    rows["ingested_at"] = pd.Timestamp.now()
    n = insert_df(rows, "demand_history")
    # Delete promoted rows from staging
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM demand_history_staging WHERE date = :d"),
            {"d": day.date()},
        )
    return n


def promote_flu(engine, day: pd.Timestamp) -> int:
    """Promote flu/ILI rows for the given day from staging → live."""
    rows = read_sql(
        "SELECT record_date, region, index_value "
        "FROM disease_burden_staging WHERE record_date = :d",
        {"d": day.date()},
    )
    if rows.empty:
        return 0
    rows["index_type"] = "ili"
    rows["source"] = "IDSP_State_Surveillance"
    rows["source_lag_days"] = 3
    rows["ingested_at"] = pd.Timestamp.now()
    # Upsert — surveillance rows may already exist
    from fill_derived import upsert_df
    upsert_df(rows, "disease_burden_index",
              ["record_date", "region", "index_value", "index_type",
               "source", "source_lag_days", "ingested_at"])
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM disease_burden_staging WHERE record_date = :d"),
            {"d": day.date()},
        )
    return len(rows)


def promote_inventory(engine, day: pd.Timestamp) -> int:
    """Promote inventory snapshots where as_of_date matches or is the nearest past weekly snapshot."""
    # Find the most recent staging snapshot on or before the day
    snap = scalar(
        "SELECT MAX(as_of_date) FROM inventory_staging WHERE as_of_date <= :d",
        {"d": day.date()},
    )
    if snap is None:
        return 0
    # Only promote if this snapshot hasn't been promoted yet
    existing = scalar(
        "SELECT COUNT(*) FROM inventory_batches WHERE as_of_date = :d",
        {"d": snap},
    )
    if existing and existing > 0:
        return 0
    rows = read_sql(
        "SELECT as_of_date, batch_id, sku_id, location, qty_units, "
        "expiry_date, received_date, unit_cost_inr, status "
        "FROM inventory_staging WHERE as_of_date = :d",
        {"d": snap},
    )
    if rows.empty:
        return 0
    rows["grn_number"] = rows["batch_id"].apply(
        lambda b: f"GRN-{int(str(b).lstrip('B')) + 10000:05d}-{str(snap).replace('-', '')}"
    )
    n = insert_df(rows, "inventory_batches")
    return n


def promote_orders(engine, day: pd.Timestamp) -> int:
    """Promote distributor orders where order_date = day."""
    rows = read_sql(
        "SELECT order_id, distributor_id, region, sku_id, atc_code, "
        "order_date, expected_delivery, actual_delivery, ordered_qty, "
        "fulfilled_qty, unit_cost_inr, order_value_inr, "
        "days_since_last_order, fulfillment_status, shortfall_qty "
        "FROM distributor_orders_staging WHERE order_date = :d",
        {"d": day.date()},
    )
    if rows.empty:
        return 0
    rows["ingested_at"] = pd.Timestamp.now()
    n = insert_df(rows, "distributor_orders")
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM distributor_orders_staging WHERE order_date = :d"),
            {"d": day.date()},
        )
    return n


def promote_capacity(engine, day: pd.Timestamp) -> int:
    """Promote warehouse capacity snapshot where snapshot_date = day."""
    rows = read_sql(
        "SELECT snapshot_date, location_id, capacity_units, used_units, "
        "available_units, utilization_pct, near_expiry_units, sku_count "
        "FROM warehouse_capacity_staging WHERE snapshot_date = :d",
        {"d": day.date()},
    )
    if rows.empty:
        return 0
    rows["computed_at"] = pd.Timestamp.now()
    n = insert_df(rows, "warehouse_capacity_log")
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM warehouse_capacity_staging WHERE snapshot_date = :d"),
            {"d": day.date()},
        )
    return n


def main():
    ap = argparse.ArgumentParser(
        description="Promote one day's data from staging → live tables"
    )
    ap.add_argument("--date", type=str, default=None,
                    help="Day to promote (YYYY-MM-DD); default = pipeline_state.simulated_today")
    args = ap.parse_args()

    engine = get_engine()

    if args.date:
        day = pd.Timestamp(args.date)
    else:
        day = get_simulated_today()

    print(f"\n=== promote staging → live for {day.date()} ===")

    n_demand = promote_demand(engine, day)
    n_flu = promote_flu(engine, day)
    n_inv = promote_inventory(engine, day)
    n_orders = promote_orders(engine, day)
    n_cap = promote_capacity(engine, day)

    print(f"  demand_history:          {n_demand} rows")
    print(f"  disease_burden_index:    {n_flu} rows")
    print(f"  inventory_batches:       {n_inv} rows")
    print(f"  distributor_orders:      {n_orders} rows")
    print(f"  warehouse_capacity_log:  {n_cap} rows")
    print("done.\n")


if __name__ == "__main__":
    main()

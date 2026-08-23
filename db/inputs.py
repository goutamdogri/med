"""
inputs.py — Loads ML pipeline INPUT tables directly from MySQL (pharma_sc),
mapping schema column names back to the legacy shapes src/features.py expects:

    disease_burden_index.index_value   -> flu_index.flu_index
    promo_calendar.planned_uplift_pct  -> promo_calendar.uplift
    location_demand_summary (latest Q) -> locations.demand_share
    inventory_batches                  -> only the newest as_of_date snapshot
"""

from __future__ import annotations

import pandas as pd

from connection import read_sql, scalar


def _dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


def get_as_of_date() -> pd.Timestamp:
    """Canonical pipeline origin = last ingested day of demand."""
    return pd.Timestamp(scalar("SELECT MAX(date) FROM demand_history"))


def load_tables_from_db() -> dict[str, pd.DataFrame]:
    t: dict[str, pd.DataFrame] = {}

    t["demand_history"] = _dates(read_sql(
        "SELECT date, sku_id, atc_code, region, units FROM demand_history"
    ), ["date"])
    t["flu_index"] = _dates(read_sql(
        "SELECT record_date AS date, region, index_value AS flu_index "
        "FROM disease_burden_index"
    ), ["date"])
    t["sku_master"] = read_sql(
        "SELECT sku_id, brand_name, atc_code, criticality, unit_cost_inr, shelf_life_days "
        "FROM sku_master"
    )
    t["lanes"] = read_sql(
        "SELECT from_location, to_location, mode, lead_time_days FROM lanes"
    )
    t["distributors"] = read_sql(
        "SELECT distributor_id, region, order_cycle_days, order_size_sigma FROM distributors"
    )
    t["promo_calendar"] = _dates(read_sql(
        "SELECT promo_id, name, start_date, end_date, planned_uplift_pct AS uplift, regions "
        "FROM promo_calendar WHERE status <> 'cancelled'"
    ), ["start_date", "end_date"])

    # demand_share feature reconstructed from latest quarterly derived table
    t["locations"] = _dates(read_sql(
        "SELECT location_id, name, type, capacity_units FROM locations"
    ), [])
    share = read_sql("""
        SELECT location_id, national_share AS demand_share
        FROM (
            SELECT location_id, national_share,
                   ROW_NUMBER() OVER (
                       PARTITION BY location_id
                       ORDER BY period_year DESC, period_quarter DESC
                   ) AS rn
            FROM location_demand_summary
        ) ranked
        WHERE rn = 1""")
    t["locations"] = t["locations"].merge(share, on="location_id", how="left")
    n = len(t["locations"])
    t["locations"]["demand_share"] = t["locations"]["demand_share"].fillna(1.0 / n).round(4)

    t["inventory_batches"] = _dates(read_sql(
        "SELECT batch_id, sku_id, location, qty_units, expiry_date, received_date, "
        "unit_cost_inr, status "
        "FROM inventory_batches "
        "WHERE as_of_date = (SELECT MAX(as_of_date) FROM inventory_batches)"
    ), ["expiry_date", "received_date"])

    return t

"""
fill_derived.py — Recomputes the [DERIVED] tables of medcare directly from PostgreSQL.

Safe to run at any interval (daily / weekly / monthly) — every write is an
upsert keyed on the table's UNIQUE constraints, so reruns never duplicate rows.

Tables handled:
  sku_market_share_monthly   monthly brand share within ATC category   (from demand_history)
  location_demand_summary    quarterly regional share + YoY growth      (from demand_history)
  sku_cost_history           seeded once; regenerated only with --full
  warehouse_capacity_log     weekly utilization snapshot                (from inventory_batches)

Usage:
    .venv/bin/python db/fill_derived.py                 # all tables, incremental-safe
    .venv/bin/python db/fill_derived.py --full          # force rebuild incl. cost history
    .venv/bin/python db/fill_derived.py --tables market_share,capacity
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from connection import get_engine, read_sql, scalar  # noqa: E402
from sqlalchemy import text  # noqa: E402

ALL_TABLES = ["market_share", "demand_summary", "cost_history", "capacity"]


def upsert_df(df: pd.DataFrame, table: str, cols: list[str], chunk: int = 1000) -> int:
    """INSERT ... ON CONFLICT DO UPDATE for every column (relies on table UNIQUE keys)."""
    if df.empty:
        print(f"  = {table:<28} no rows")
        return 0
    placeholders = ", ".join(f":{c}" for c in cols)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols)
    stmt = text(
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT DO UPDATE SET {updates}"
    )
    records = df[cols].to_dict(orient="records")
    engine = get_engine()
    with engine.begin() as conn:
        for i in range(0, len(records), chunk):
            conn.execute(stmt, records[i : i + chunk])
    print(f"  + {table:<28} {len(records):>7} rows upserted")
    return len(records)


def _demand_range(engine) -> tuple[pd.Timestamp, pd.Timestamp]:
    row = read_sql("SELECT MIN(date) AS mn, MAX(date) AS mx FROM demand_history")
    return pd.Timestamp(row["mn"].iloc[0]), pd.Timestamp(row["mx"].iloc[0])


# ─── [DERIVED] sku_market_share_monthly ────────────────────────────────────────

def fill_market_share(engine) -> None:
    mn, mx = _demand_range(engine)
    with engine.begin() as conn:
        conn.execute(text(
            "DELETE FROM sku_market_share_monthly "
            "WHERE period_year < :y0 OR period_year > :y1 "
            "   OR (period_year = :y0 AND period_month < :m0) "
            "   OR (period_year = :y1 AND period_month > :m1)"),
            {"y0": mn.year, "m0": mn.month, "y1": mx.year, "m1": mx.month})

    sku_month = read_sql("""
        SELECT sku_id, atc_code,
               EXTRACT(YEAR FROM date)::int  AS period_year,
               EXTRACT(MONTH FROM date)::int AS period_month,
               SUM(units)  AS total_units_sold
        FROM demand_history
        GROUP BY sku_id, atc_code, period_year, period_month""")
    cat_month = read_sql("""
        SELECT atc_code,
               EXTRACT(YEAR FROM date)::int  AS period_year,
               EXTRACT(MONTH FROM date)::int AS period_month,
               SUM(units)  AS category_units_sold
        FROM demand_history
        GROUP BY atc_code, period_year, period_month""")
    df = sku_month.merge(cat_month, on=["atc_code", "period_year", "period_month"])
    df["market_share"] = (
        df["total_units_sold"] / df["category_units_sold"].replace(0, np.nan)
    ).round(4)
    df["computed_on"] = date.today()
    upsert_df(
        df,
        "sku_market_share_monthly",
        ["sku_id", "atc_code", "period_year", "period_month",
         "total_units_sold", "category_units_sold", "market_share", "computed_on"],
    )


# ─── [DERIVED] location_demand_summary ────────────────────────────────────────

def fill_demand_summary(engine) -> None:
    mn, mx = _demand_range(engine)
    with engine.begin() as conn:
        conn.execute(text(
            "DELETE FROM location_demand_summary "
            "WHERE period_year < :y0 OR period_year > :y1 "
            "   OR (period_year = :y0 AND period_quarter < :q0) "
            "   OR (period_year = :y1 AND period_quarter > :q1)"),
            {"y0": mn.year, "q0": mn.quarter, "y1": mx.year, "q1": mx.quarter})

    loc_q = read_sql("""
        SELECT region AS location_id,
               EXTRACT(YEAR FROM date)::int    AS period_year,
               EXTRACT(QUARTER FROM date)::int AS period_quarter,
               SUM(units)   AS total_units
        FROM demand_history
        GROUP BY region, period_year, period_quarter""")
    nat_q = read_sql("""
        SELECT EXTRACT(YEAR FROM date)::int AS period_year,
               EXTRACT(QUARTER FROM date)::int AS period_quarter,
               SUM(units) AS national_total
        FROM demand_history
        GROUP BY period_year, period_quarter""")
    df = loc_q.merge(nat_q, on=["period_year", "period_quarter"])
    df["national_share"] = (
        df["total_units"] / df["national_total"].replace(0, np.nan)
    ).round(4)

    # YoY growth % vs same quarter of previous year
    prev = df.copy()
    prev["period_year"] = prev["period_year"] - 1
    prev = prev.rename(columns={"total_units": "prev_units"})[
        ["location_id", "period_year", "period_quarter", "prev_units"]
    ]
    df = df.merge(prev, on=["location_id", "period_year", "period_quarter"], how="left")
    df["yoy_growth_pct"] = np.where(
        df["prev_units"].fillna(0) > 0,
        ((df["total_units"] - df["prev_units"]) / df["prev_units"] * 100).round(2),
        None,
    )
    df["computed_on"] = date.today()
    upsert_df(
        df,
        "location_demand_summary",
        ["location_id", "period_year", "period_quarter", "total_units",
         "national_share", "yoy_growth_pct", "computed_on"],
    )


# ─── [DERIVED] sku_cost_history ───────────────────────────────────────────────

def fill_cost_history(engine, force: bool) -> None:
    n = scalar("SELECT COUNT(*) FROM sku_cost_history")
    if n and not force:
        print(f"  = sku_cost_history            {n} rows exist; skipping (use --full to regenerate)")
        return
    skus = read_sql("SELECT sku_id, unit_cost_inr FROM sku_master")
    rng = np.random.default_rng(seed=99)
    cost_periods = [
        ("2014-01-01", "2016-06-30", "Initial supplier onboarding price",              1.00),
        ("2016-07-01", "2017-06-30", "Pre-GST price revision",                        1.08),
        ("2017-07-01", "2020-12-31", "Post-GST supplier renegotiation",               1.15),
        ("2021-01-01", None,         "Annual renegotiation — COVID surcharge removed", 1.12),
    ]
    rows = []
    for _, sku in skus.iterrows():
        base = int(sku["unit_cost_inr"])
        for eff_from, eff_to, reason, mult in cost_periods:
            rows.append({
                "sku_id":         sku["sku_id"],
                "unit_cost_inr":  max(1, int(round(base * mult * rng.uniform(0.97, 1.03)))),
                "effective_from": eff_from,
                "effective_to":   eff_to,
                "reason":         reason,
                "approved_by":    "procurement@pharma.in",
            })
    df = pd.DataFrame(rows)
    if force:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM sku_cost_history"))
    upsert_df(
        df,
        "sku_cost_history",
        ["sku_id", "unit_cost_inr", "effective_from", "effective_to", "reason", "approved_by"],
    )


# ─── Weekly warehouse capacity snapshot ───────────────────────────────────────

def fill_capacity(engine, force: bool) -> None:
    """Insert a fresh weekly snapshot whenever inventory_batches has a newer
    snapshot date than the last warehouse_capacity_log entry (>=7 days later,
    or always with --full). Mirrors the seed logic so charts stay continuous."""
    inv_max = scalar("SELECT MAX(as_of_date) FROM inventory_batches")
    snap_max = scalar("SELECT MAX(snapshot_date) FROM warehouse_capacity_log")
    if inv_max is None:
        print("  = warehouse_capacity_log     no inventory snapshots yet")
        return
    due = snap_max is None or force or (
        pd.Timestamp(inv_max) - pd.Timestamp(snap_max)
    ) >= pd.Timedelta(days=7)
    if not due:
        print(f"  = warehouse_capacity_log     latest snapshot {snap_max} still current")
        return

    caps = read_sql("SELECT location_id, capacity_units FROM locations").set_index("location_id")
    inv = read_sql("""
        SELECT location, sku_id, qty_units, expiry_date
        FROM inventory_batches
        WHERE as_of_date = :d""", {"d": inv_max})
    near_cutoff = pd.Timestamp(inv_max) + pd.Timedelta(days=90)
    inv["expiry_date"] = pd.to_datetime(inv["expiry_date"])

    rows = []
    for loc_id, cap in caps["capacity_units"].items():
        g = inv[inv["location"] == loc_id]
        used = int(g["qty_units"].sum())
        ne = int(g.loc[g["expiry_date"] <= near_cutoff, "qty_units"].sum())
        rows.append({
            "snapshot_date":    inv_max,
            "location_id":      loc_id,
            "capacity_units":   int(cap),
            "used_units":       used,
            "available_units":  max(0, int(cap) - used),
            "utilization_pct":  round(used / cap * 100, 2) if cap else 0.0,
            "near_expiry_units": min(ne, used),
            "sku_count":        int(g["sku_id"].nunique()),
        })
    upsert_df(
        pd.DataFrame(rows),
        "warehouse_capacity_log",
        ["snapshot_date", "location_id", "capacity_units", "used_units",
         "available_units", "utilization_pct", "near_expiry_units", "sku_count"],
    )


def main():
    ap = argparse.ArgumentParser(description="Fill pharma_sc [DERIVED] tables from MySQL")
    ap.add_argument("--tables", type=str, default="all",
                    help="comma list: market_share,demand_summary,cost_history,capacity (default all)")
    ap.add_argument("--full", action="store_true",
                    help="force regeneration (incl. sku_cost_history)")
    args = ap.parse_args()

    chosen = ALL_TABLES if args.tables == "all" else [t.strip() for t in args.tables.split(",")]
    unknown = [t for t in chosen if t not in ALL_TABLES]
    if unknown:
        raise SystemExit(f"unknown tables: {unknown}; choose from {ALL_TABLES}")

    engine = get_engine()
    print(f"\n=== fill_derived @ {date.today()} ===")
    if "market_share" in chosen:
        fill_market_share(engine)
    if "demand_summary" in chosen:
        fill_demand_summary(engine)
    if "cost_history" in chosen:
        fill_cost_history(engine, force=args.full)
    if "capacity" in chosen:
        fill_capacity(engine, force=args.full)
    print("done.\n")


if __name__ == "__main__":
    main()

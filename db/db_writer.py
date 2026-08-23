"""
db_writer.py — Seeds all INPUT and INPUT-ROLLING tables from processed parquet files.

Run this once after build_dataset.py to seed the DB with the full historical dataset.
DERIVED tables are filled separately by db/fill_derived.py (computed from MySQL).

Usage:
    python db/db_writer.py --mode seed     # Full seed (run once)
    python db/db_writer.py --mode rolling  # Append latest snapshot rolling data only

Requirements:
    pip install sqlalchemy pymysql pandas pyarrow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sqlalchemy.engine import Engine

sys.path.insert(0, str(Path(__file__).resolve().parent))
from connection import DB_NAME, get_engine, insert_df  # noqa: E402

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


def write(df: pd.DataFrame, table: str, engine: Engine, chunk: int = 500):
    """Append a DataFrame to MySQL (INPUT/rolling tables; no unique-key conflicts on fresh seed)."""
    return insert_df(df, table, chunk=chunk)


# ─── Enrichment helpers ────────────────────────────────────────────────────────

# Realistic manufacturer names per ATC category
MANUFACTURER_MAP = {
    "M01AB": "Sun Pharma Industries Ltd",
    "M01AE": "Cipla Ltd",
    "N02BA": "Abbott India Ltd",
    "N02BE": "GSK Pharmaceuticals India",
    "N05B":  "Torrent Pharmaceuticals Ltd",
    "N05C":  "Alkem Laboratories Ltd",
    "R03":   "AstraZeneca India Pvt Ltd",
    "R06":   "Mankind Pharma Ltd",
}

REGULATORY_CLASS_MAP = {
    "M01AB": "Schedule H",
    "M01AE": "Schedule H",
    "N02BA": "OTC",
    "N02BE": "OTC",
    "N05B":  "Schedule H",
    "N05C":  "Schedule H1",
    "R03":   "Schedule H",
    "R06":   "OTC",
}

HSN_MAP = {
    "M01AB": "30049099",
    "M01AE": "30049099",
    "N02BA": "30049041",
    "N02BE": "30049041",
    "N05B":  "30049039",
    "N05C":  "30049039",
    "R03":   "30049069",
    "R06":   "30049099",
}

NLEM_MAP = {
    "M01AB": False, "M01AE": True, "N02BA": True,
    "N02BE": True,  "N05B":  False, "N05C": False,
    "R03":   True,  "R06":   False,
}

CRITICALITY_RATIONALE_MAP = {
    "critical": "Life-sustaining or no therapeutic substitute in Tier-2 regions; covered under NLEM; stockout triggers patient safety escalation.",
    "high":     "Frequent acute-care use; therapeutic alternatives limited; supply disruption impacts treatment adherence.",
    "standard": "Broad therapeutic class with multiple substitutes; routine replenishment cadence sufficient.",
    "low":      "Elective / lifestyle category; stockout does not pose immediate patient risk; extended cover acceptable.",
}

# Realistic city/state data per location
LOCATION_ENRICH = {
    "DC_MUMBAI":     {"city": "Mumbai",      "state": "Maharashtra",   "pincode": "400093", "gstin": "27AABCS1234A1Z5"},
    "DC_DELHI":      {"city": "New Delhi",   "state": "Delhi",         "pincode": "110020", "gstin": "07AABCS5678A1Z2"},
    "WH_NAGPUR":     {"city": "Nagpur",      "state": "Maharashtra",   "pincode": "440001", "gstin": "27AABCS9012A1Z8"},
    "WH_INDORE":     {"city": "Indore",      "state": "Madhya Pradesh","pincode": "452001", "gstin": "23AABCS3456A1Z4"},
    "WH_COIMBATORE": {"city": "Coimbatore",  "state": "Tamil Nadu",    "pincode": "641001", "gstin": "33AABCS7890A1Z6"},
    "WH_LUCKNOW":    {"city": "Lucknow",     "state": "Uttar Pradesh", "pincode": "226010", "gstin": "09AABCS2345A1Z3"},
}

COMMISSIONED_DATE_MAP = {
    "DC_MUMBAI":     "2013-04-01",
    "DC_DELHI":      "2013-04-01",
    "WH_NAGPUR":     "2014-07-15",
    "WH_INDORE":     "2014-09-01",
    "WH_COIMBATORE": "2015-01-10",
    "WH_LUCKNOW":    "2015-03-20",
}

DISTRIBUTOR_NAMES = {
    "DST_DC_MUMBAI":     "Mumbai MedSupply Pvt Ltd",
    "DST_DC_DELHI":      "Delhi PharmaLink Distributors",
    "DST_WH_NAGPUR":     "Nagpur HealthServe Agencies",
    "DST_WH_INDORE":     "Indore CarePoint Distributors",
    "DST_WH_COIMBATORE": "Coimbatore MedTrade Pvt Ltd",
    "DST_WH_LUCKNOW":    "Lucknow PharmaCentre Ltd",
}

CARRIER_MAP = {
    "replenish": "Blue Dart Express Ltd",
    "transfer":  "TCI Supply Chain Solutions",
}


# ─── Seeding Functions ─────────────────────────────────────────────────────────

def seed_sku_master(engine):
    """[INPUT] Enrich and write sku_master from sku_master.parquet"""
    df = pd.read_parquet(PROCESSED / "sku_master.parquet")

    df["manufacturer"]          = df["atc_code"].map(MANUFACTURER_MAP)
    df["regulatory_class"]      = df["atc_code"].map(REGULATORY_CLASS_MAP)
    df["hsn_code"]              = df["atc_code"].map(HSN_MAP)
    df["nlem_listed"]           = df["atc_code"].map(NLEM_MAP)
    df["criticality_rationale"] = df["criticality"].map(CRITICALITY_RATIONALE_MAP)
    df["criticality_reviewed_on"] = "2023-04-01"
    df["criticality_reviewed_by"] = "supply_chain_committee@pharma.in"

    # Drop category_share — it goes to sku_market_share_monthly instead
    df = df.drop(columns=["category_share"], errors="ignore")

    write(df, "sku_master", engine)


def seed_locations(engine):
    """[INPUT] Enrich and write locations from locations.parquet"""
    df = pd.read_parquet(PROCESSED / "locations.parquet")

    df["city"]           = df["location_id"].map(lambda x: LOCATION_ENRICH.get(x, {}).get("city"))
    df["state"]          = df["location_id"].map(lambda x: LOCATION_ENRICH.get(x, {}).get("state"))
    df["pincode"]        = df["location_id"].map(lambda x: LOCATION_ENRICH.get(x, {}).get("pincode"))
    df["gstin"]          = df["location_id"].map(lambda x: LOCATION_ENRICH.get(x, {}).get("gstin"))
    df["commissioned_on"]= df["location_id"].map(COMMISSIONED_DATE_MAP)

    # Drop demand_share — it goes to location_demand_summary instead
    df = df.drop(columns=["demand_share"], errors="ignore")

    write(df, "locations", engine)


def seed_lanes(engine):
    """[INPUT] Write lanes from lanes.parquet with carrier info"""
    df = pd.read_parquet(PROCESSED / "lanes.parquet")
    df["carrier"] = df["mode"].map(CARRIER_MAP)
    write(df, "lanes", engine)


def seed_distributors(engine):
    """[INPUT] Enrich and write distributors from distributors.parquet"""
    df = pd.read_parquet(PROCESSED / "distributors.parquet")
    df["distributor_name"] = df["distributor_id"].map(DISTRIBUTOR_NAMES)
    df["contact_email"] = df["distributor_id"].str.lower().str.replace("dst_", "") + "@distributor.pharma.in"
    write(df, "distributors", engine)


def seed_promo_calendar(engine):
    """[INPUT] Write promo_calendar from promo_calendar.parquet.
    Past promos get actual_uplift_pct (slightly different from planned — realistic).
    """
    df = pd.read_parquet(PROCESSED / "promo_calendar.parquet")
    today = pd.Timestamp.today().normalize()

    # Rename planned uplift
    df = df.rename(columns={"uplift": "planned_uplift_pct"})

    # Compute actual_uplift_pct for completed promos: planned ± small noise
    rng = np.random.default_rng(seed=42)
    completed = df["end_date"] < today
    df["actual_uplift_pct"] = None
    noise = rng.normal(0, 0.03, size=completed.sum())  # ±3% variance
    df.loc[completed, "actual_uplift_pct"] = (
        df.loc[completed, "planned_uplift_pct"].values + noise
    ).clip(0.01, 0.50).round(3)

    # Set status
    df["status"] = "completed"
    df.loc[df["end_date"] >= today, "status"] = "planned"
    df.loc[(df["start_date"] <= today) & (df["end_date"] >= today), "status"] = "active"

    df["created_by"]  = "marketing_team@pharma.in"
    df["approved_on"] = (pd.to_datetime(df["start_date"]) - pd.Timedelta(days=14)).dt.date

    # target_atc_codes: map promo regions to ATC sensitivity
    df["target_atc_codes"] = "R03,R06,N02BE,M01AE"  # reasonable default for seasonal promos

    write(df, "promo_calendar", engine)


def seed_demand_history(engine, full_history: bool = False):
    """[INPUT-ROLLING] Write demand_history from demand_history.parquet.
    By default only days up to config as_of_date are loaded — later days are
    revealed one at a time by simulate_ingest_day.py (production: real ingest)."""
    df = pd.read_parquet(PROCESSED / "demand_history.parquet")
    if not full_history:
        wm = pd.Timestamp(CFG["project"]["as_of_date"])
        n0 = len(df)
        df = df[pd.to_datetime(df["date"]) <= wm]
        print(f"  · trimmed to watermark {wm.date()} ({n0 - len(df)} future rows withheld)")
    df["ingested_at"] = pd.Timestamp.now()
    write(df, "demand_history", engine)


def seed_disease_burden_index(engine, full_history: bool = True):
    """[INPUT-ROLLING] Write disease_burden_index from flu_index.parquet.
    Renames flu_index → index_value and adds realistic source metadata.
    NOT watermarked: public surveillance data is published ahead of internal
    sales and is required as a future covariate (futr_flu_index) by the model.
    """
    df = pd.read_parquet(PROCESSED / "flu_index.parquet")
    if not full_history:
        wm = pd.Timestamp(CFG["project"]["as_of_date"])
        df = df[pd.to_datetime(df["date"]) <= wm]
    df = df.rename(columns={"flu_index": "index_value", "date": "record_date"})
    df["index_type"]      = "ili"
    df["source"]          = "IDSP_State_Surveillance"
    df["source_lag_days"] = 3
    df["ingested_at"]     = pd.Timestamp.now()
    write(df, "disease_burden_index", engine)


def seed_inventory_batches(engine):
    """[INPUT-ROLLING] Write inventory_batches from inventory_batches.parquet.
    Adds as_of_date from meta.json and synthetic GRN numbers.
    """
    df = pd.read_parquet(PROCESSED / "inventory_batches.parquet")
    meta = yaml.safe_load((PROCESSED / "meta.json").read_text())
    df["as_of_date"] = meta["as_of_date"]

    # Generate realistic GRN numbers
    df["grn_number"] = df["batch_id"].apply(
        lambda b: f"GRN-{int(b[1:]) + 10000:05d}-{meta['as_of_date'].replace('-', '')}"
    )
    write(df, "inventory_batches", engine)


def seed_distributor_orders(engine):
    """[INPUT-ROLLING] Generate synthetic historical distributor orders from demand_history.
    Shows realistic ordering patterns: variable cycle times, occasional shortfalls.
    """
    demand = pd.read_parquet(PROCESSED / "demand_history.parquet")
    demand["date"] = pd.to_datetime(demand["date"])
    dist_df = pd.read_parquet(PROCESSED / "distributors.parquet")
    sku_df  = pd.read_parquet(PROCESSED / "sku_master.parquet").set_index("sku_id")

    rng = np.random.default_rng(seed=77)
    rows = []
    order_counter = 1

    for _, dist in dist_df.iterrows():
        region = dist["region"]
        cycle  = int(dist["order_cycle_days"])
        sigma  = float(dist["order_size_sigma"])

        region_demand = demand[demand["region"] == region].sort_values("date")
        if region_demand.empty:
            continue

        date_min = region_demand["date"].min()
        date_max = region_demand["date"].max()

        # Generate order dates at ~cycle intervals with jitter
        order_date = date_min + pd.Timedelta(days=int(rng.integers(1, cycle)))
        while order_date <= date_max:
            # Aggregate demand over next `cycle` days as the order quantity
            window_end = order_date + pd.Timedelta(days=cycle)
            window_demand = region_demand[
                (region_demand["date"] >= order_date) &
                (region_demand["date"] < window_end)
            ]

            for sku_id, grp in window_demand.groupby("sku_id"):
                base_qty = int(grp["units"].sum())
                if base_qty <= 0:
                    continue
                # Add sizing noise
                ordered_qty = max(1, int(base_qty * rng.lognormal(0, sigma)))
                # 10% chance of partial fulfillment
                fulfilled_qty = ordered_qty if rng.random() > 0.10 else int(ordered_qty * rng.uniform(0.6, 0.95))
                unit_cost = int(sku_df.loc[sku_id, "unit_cost_inr"]) if sku_id in sku_df.index else 10
                lead = rng.integers(5, 14)

                rows.append({
                    "order_id":             f"PO-{order_counter:07d}",
                    "distributor_id":       dist["distributor_id"],
                    "region":               region,
                    "sku_id":               sku_id,
                    "atc_code":             sku_df.loc[sku_id, "atc_code"] if sku_id in sku_df.index else "",
                    "order_date":           order_date.date(),
                    "expected_delivery":    (order_date + pd.Timedelta(days=int(lead))).date(),
                    "actual_delivery":      (order_date + pd.Timedelta(days=int(lead) + rng.integers(0, 3))).date(),
                    "ordered_qty":          ordered_qty,
                    "fulfilled_qty":        fulfilled_qty,
                    "unit_cost_inr":        unit_cost,
                    "order_value_inr":      ordered_qty * unit_cost,
                    "days_since_last_order": cycle + int(rng.integers(-2, 3)),
                    "fulfillment_status":   "fulfilled" if fulfilled_qty == ordered_qty else "partial",
                    "shortfall_qty":        max(0, ordered_qty - fulfilled_qty),
                })
                order_counter += 1

            # Next order date with jitter
            jitter = int(rng.integers(-2, 3))
            order_date += pd.Timedelta(days=cycle + jitter)

    write(pd.DataFrame(rows), "distributor_orders", engine)


def seed_warehouse_capacity_log(engine):
    """[INPUT-ROLLING] Compute weekly warehouse capacity utilization from inventory_batches."""
    inv = pd.read_parquet(PROCESSED / "inventory_batches.parquet")
    loc = pd.read_parquet(PROCESSED / "locations.parquet")
    meta = yaml.safe_load((PROCESSED / "meta.json").read_text())
    as_of = pd.Timestamp(meta["as_of_date"])

    cap_map = loc.set_index("location_id")["capacity_units"].to_dict()

    inv["expiry_date"]  = pd.to_datetime(inv["expiry_date"])
    near_expiry_cutoff  = as_of + pd.Timedelta(days=90)

    rows = []
    # Simulate weekly snapshots for past 12 weeks
    rng = np.random.default_rng(seed=55)
    for week_offset in range(12, 0, -1):
        snap_date = (as_of - pd.Timedelta(weeks=week_offset)).date()
        for loc_id, cap in cap_map.items():
            loc_inv = inv[inv["location"] == loc_id]
            # Simulate historical utilization with slight noise
            used = int(loc_inv["qty_units"].sum() * rng.uniform(0.80, 1.05))
            used = min(used, cap)
            ne_units = int(loc_inv[loc_inv["expiry_date"] <= near_expiry_cutoff]["qty_units"].sum() * rng.uniform(0.9, 1.1))
            rows.append({
                "snapshot_date":    snap_date,
                "location_id":      loc_id,
                "capacity_units":   cap,
                "used_units":       used,
                "available_units":  max(0, cap - used),
                "utilization_pct":  round(used / cap * 100, 2) if cap > 0 else 0.0,
                "near_expiry_units": min(ne_units, used),
                "sku_count":        int(loc_inv["sku_id"].nunique()),
            })

    # Current snapshot
    for loc_id, cap in cap_map.items():
        loc_inv = inv[inv["location"] == loc_id]
        used = int(loc_inv["qty_units"].sum())
        ne_units = int(loc_inv[loc_inv["expiry_date"] <= near_expiry_cutoff]["qty_units"].sum())
        rows.append({
            "snapshot_date":    as_of.date(),
            "location_id":      loc_id,
            "capacity_units":   cap,
            "used_units":       used,
            "available_units":  max(0, cap - used),
            "utilization_pct":  round(used / cap * 100, 2) if cap > 0 else 0.0,
            "near_expiry_units": ne_units,
            "sku_count":        int(loc_inv["sku_id"].nunique()),
        })

    write(pd.DataFrame(rows), "warehouse_capacity_log", engine)


# ─── Main ──────────────────────────────────────────────────────────────────────

SEED_FUNCTIONS = [
    # (table, seeder, truncate_first) — seed mode is a full reload, so each
    # target table is emptied first to keep reruns idempotent.
    # INPUT — static master tables (run once)
    ("sku_master",                 seed_sku_master,            True),
    ("locations",                  seed_locations,             True),
    ("lanes",                      seed_lanes,                 True),
    ("distributors",               seed_distributors,          True),
    ("promo_calendar",             seed_promo_calendar,        True),
    # INPUT-ROLLING — grows over time
    ("demand_history",             seed_demand_history,        True),
    ("disease_burden_index",       seed_disease_burden_index,  True),
    ("inventory_batches",          seed_inventory_batches,     True),
    ("distributor_orders",         seed_distributor_orders,    True),
    # weekly capacity snapshots (historical + latest)
    ("warehouse_capacity_log",     seed_warehouse_capacity_log, True),
]

ROLLING_FUNCTIONS = [
    # Only these are appended when new data arrives
    ("demand_history",       seed_demand_history,       False),
    ("disease_burden_index", seed_disease_burden_index, False),
    ("inventory_batches",    seed_inventory_batches,    False),
]


def main():
    parser = argparse.ArgumentParser(description="Seed pharma_sc DB from processed parquet files")
    parser.add_argument(
        "--mode",
        choices=["seed", "rolling"],
        default="seed",
        help="seed = full initial load | rolling = append latest rolling data only",
    )
    parser.add_argument(
        "--full-history", action="store_true",
        help="load demand/flu history beyond the as-of watermark (disables day-reveal demo)",
    )
    args = parser.parse_args()

    engine = get_engine()
    print(f"\n{'='*55}")
    print(f"  Mode: {args.mode.upper()}")
    print(f"  DB:   {DB_NAME} @ {engine.url.host}:{engine.url.port}")
    print(f"{'='*55}\n")

    functions = SEED_FUNCTIONS if args.mode == "seed" else ROLLING_FUNCTIONS

    import inspect
    from sqlalchemy import text

    for table_name, fn, truncate in functions:
        try:
            if truncate:
                with engine.begin() as conn:
                    conn.execute(text(f"TRUNCATE TABLE {table_name}"))
            params = inspect.signature(fn).parameters
            kw = {"full_history": args.full_history} if "full_history" in params else {}
            fn(engine, **kw)
        except Exception as e:
            print(f"  ✗ {table_name:<35} ERROR: {e}")
            sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  Done. {len(functions)} table(s) written successfully.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()

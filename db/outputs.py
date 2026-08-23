"""
outputs.py — Writes ML pipeline results into the pharma_sc [OUTPUT] tables.

Every writer is idempotent per run: rows for the same as_of_date are replaced
(DELETE + INSERT, or upsert where UNIQUE keys exist), so rerunning any stage
or a full day never duplicates data.

All stages dual-write: they keep producing parquet/csv/json artifacts for the
dashboard AND call these functions.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from connection import get_engine, insert_df, scalar, write_run  # noqa: E402
from sqlalchemy import text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def current_as_of() -> pd.Timestamp:
    """Canonical run date: rolled forward by rolling_forecast/ensemble in config.yaml."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    return pd.Timestamp(cfg["project"]["as_of_date"])


def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
    """inf -> NULL; keep NaN (pandas writes NULL natively)."""
    df = df.copy()
    return df.replace([np.inf, -np.inf], np.nan)


# ─── [OUTPUT] forecasts_final ─────────────────────────────────────────────────

def write_forecasts(fcst: pd.DataFrame, as_of_date,
                    models_used: list[str], weights: dict[str, float]) -> int:
    df = _sanitize(fcst)
    out = pd.DataFrame({
        "as_of_date":      pd.Timestamp(as_of_date).date(),
        "sku_id":          df["sku_id"],
        "atc_code":        df["atc_code"],
        "region":          df["region"],
        "forecast_date":   pd.to_datetime(df["forecast_date"]).dt.date,
        "horizon":         df["horizon"].astype(int),
        "p10":             df["p10"].round(2),
        "p50":             df["p50"].round(2),
        "p90":             df["p90"].round(2),
        "momentum_u":      df.get("momentum_u", np.nan),
        "flu_ratio":       df.get("flu_ratio", np.nan),
        "sense_adjustment": df.get("sense_adjustment", np.nan),
        "models_used":     ",".join(sorted(models_used)),
        "lgbm_weight":     round(float(weights.get("lgbm", 0.0)), 3) or None,
        "chronos_weight":  round(float(weights.get("chronos", 0.0)), 3) or None,
    })
    for c in ["momentum_u", "flu_ratio", "sense_adjustment"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(4)
    return write_run(out, "forecasts_final", as_of_date)


# ─── [OUTPUT] rolling_run_log ─────────────────────────────────────────────────

def write_run_log(as_of_date, previous_as_of_date, models_used: list[str],
                  weights: dict[str, float], wmape: float | None,
                  forecast_rows: int, status: str = "success",
                  error_message: str | None = None, duration_s: int | None = None,
                  triggered_by: str = "cron") -> None:
    d = pd.Timestamp(as_of_date).date()
    if previous_as_of_date is None:
        # auto-resolve from the audit trail
        previous_as_of_date = scalar(
            "SELECT MAX(as_of_date) FROM rolling_run_log WHERE as_of_date < :d",
            {"d": d},
        )
    stmt = text("""
        INSERT INTO rolling_run_log
            (as_of_date, previous_as_of_date, models_used, lgbm_weight, chronos_weight,
             wmape, forecast_rows, status, error_message, run_duration_seconds, triggered_by)
        VALUES (:d, :pd, :mu, :lw, :cw, :wm, :rows, :st, :err, :dur, :trig) AS new
        ON DUPLICATE KEY UPDATE
            previous_as_of_date=new.previous_as_of_date, models_used=new.models_used,
            lgbm_weight=new.lgbm_weight, chronos_weight=new.chronos_weight,
            wmape=new.wmape, forecast_rows=new.forecast_rows, status=new.status,
            error_message=new.error_message, run_duration_seconds=new.run_duration_seconds,
            triggered_by=new.triggered_by""")
    params = {
        "d": d,
        "pd": pd.Timestamp(previous_as_of_date).date() if previous_as_of_date else None,
        "mu": ",".join(sorted(models_used)),
        "lw": round(float(weights.get("lgbm", 0.0)) or 0, 3),
        "cw": round(float(weights.get("chronos", 0.0)) or 0, 3),
        "wm": round(float(wmape), 4) if wmape is not None and np.isfinite(wmape) else None,
        "rows": int(forecast_rows),
        "st": status,
        "err": error_message,
        "dur": int(duration_s) if duration_s is not None else None,
        "trig": triggered_by,
    }
    with get_engine().begin() as conn:
        conn.execute(stmt, params)
    print(f"  + rolling_run_log             as_of={params['d']} status={status}")


# ─── [OUTPUT] replenishment_orders ────────────────────────────────────────────

def write_replenishment(plan: pd.DataFrame, as_of_date) -> int:
    df = _sanitize(plan)
    df.insert(0, "as_of_date", pd.Timestamp(as_of_date).date())
    cols = ["as_of_date", "sku_id", "region", "criticality", "lead_time_days",
            "service_level", "mu_daily", "sigma_daily", "safety_stock",
            "target_position", "on_hand", "order_qty", "order_value_inr",
            "days_of_supply_on_hand", "status"]
    return write_run(df[cols].assign(
        days_of_supply_on_hand=df["days_of_supply_on_hand"].where(np.isfinite(df["days_of_supply_on_hand"]))
    ), "replenishment_orders", as_of_date)


# ─── [OUTPUT] transfer_plan / writeoff_risk ───────────────────────────────────

def write_transfer_plan(tp: pd.DataFrame, as_of_date) -> int:
    df = _sanitize(tp)
    df.insert(0, "as_of_date", pd.Timestamp(as_of_date).date())
    cols = ["as_of_date", "batch_id", "sku_id", "from_location", "to_location",
            "qty_units", "expiry_date", "days_to_expiry", "transfer_lead_days",
            "value_saved_inr", "reason", "src_days_of_supply_before"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce").dt.date
    return write_run(df[cols], "transfer_plan", as_of_date)


def write_writeoff_risk(wo: pd.DataFrame, as_of_date) -> int:
    df = _sanitize(wo)
    df.insert(0, "as_of_date", pd.Timestamp(as_of_date).date())
    cols = ["as_of_date", "batch_id", "sku_id", "location", "qty_units", "leftover",
            "residual_writeoff_units", "unit_cost_inr", "residual_value_inr",
            "expiry_date", "days_to_expiry"]
    df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce").dt.date
    return write_run(df[cols], "writeoff_risk", as_of_date)


# ─── [OUTPUT] simulation_daily / kpi_summary ──────────────────────────────────

def write_simulation(sim: pd.DataFrame, kpi: pd.DataFrame, as_of_date) -> tuple[int, int]:
    sim = _sanitize(sim)
    sim.insert(0, "as_of_date", pd.Timestamp(as_of_date).date())
    sim_cols = ["as_of_date", "policy", "date", "sku_id", "region", "criticality",
                "demand", "fulfilled", "unfulfilled", "expired_units",
                "expired_value_inr", "ending_inventory"]
    n_sim = write_run(sim[sim_cols], "simulation_daily", as_of_date)

    kpi = _sanitize(kpi)
    kpi.insert(0, "as_of_date", pd.Timestamp(as_of_date).date())
    kpi_cols = ["as_of_date", "policy", "fill_rate_pct", "critical_fill_rate_pct",
                "stockout_units", "critical_stockout_sitedays", "writeoff_value_inr",
                "avg_ending_inventory"]
    # upsert on uq_kpi (as_of_date, policy)
    stmt = text("""
        INSERT INTO kpi_summary ({cols}) VALUES ({ph}) AS new
        ON DUPLICATE KEY UPDATE {upd}""".format(
        cols=", ".join(kpi_cols),
        ph=", ".join(f":{c}" for c in kpi_cols),
        upd=", ".join(f"{c}=new.{c}" for c in kpi_cols)))
    with get_engine().begin() as conn:
        conn.execute(stmt, kpi[kpi_cols].to_dict(orient="records"))
    print(f"  + kpi_summary                 {len(kpi):>7} rows upserted")
    return n_sim, len(kpi)


# ─── [OUTPUT] alerts / alert_digest ───────────────────────────────────────────

def write_alerts(alerts: list[dict], digest_text: str, review_mode: str,
                 surge_regions: list[str], red_count: int,
                 model_used: str, as_of_date) -> tuple[int, int]:
    rows = [{
        "as_of_date":  pd.Timestamp(as_of_date).date(),
        "severity":    a["severity"],
        "type":        a["type"],
        "sku_id":      a.get("sku_id", "*"),
        "region":      a["region"],
        "facts":       __import__("json").dumps(a.get("facts", {}), default=str),
        "action":      a.get("action"),
    } for a in alerts]
    n_alerts = write_run(pd.DataFrame(rows), "alerts", as_of_date)

    digest = pd.DataFrame([{
        "as_of_date":      pd.Timestamp(as_of_date).date(),
        "review_mode":     review_mode,
        "surge_regions":   ",".join(surge_regions) if surge_regions else None,
        "red_alert_count": int(red_count),
        "digest_text":     digest_text,
        "model_used":      model_used,
    }])
    cols = list(digest.columns)
    stmt = text("""
        INSERT INTO alert_digest ({cols}) VALUES ({ph}) AS new
        ON DUPLICATE KEY UPDATE {upd}""".format(
        cols=", ".join(cols),
        ph=", ".join(f":{c}" for c in cols),
        upd=", ".join(f"{c}=new.{c}" for c in cols)))
    with get_engine().begin() as conn:
        conn.execute(stmt, digest.to_dict(orient="records"))
    print(f"  + alert_digest                as_of={pd.Timestamp(as_of_date).date()} mode={review_mode}")
    return n_alerts, 1


def table_counts(tables: list[str] | None = None) -> pd.DataFrame:
    names = tables or ["forecasts_final", "replenishment_orders", "transfer_plan",
                       "writeoff_risk", "simulation_daily", "kpi_summary",
                       "alerts", "alert_digest", "rolling_run_log"]
    frames = []
    for t in names:
        n = scalar(f"SELECT COUNT(*) FROM {t}")
        last = scalar(f"SELECT MAX(as_of_date) FROM {t}")
        frames.append({"table": t, "rows": n, "last_as_of": last})
    return pd.DataFrame(frames)

"""
metrics.py — Model performance evaluation and metrics computation.

Computes accuracy/precision metrics for all models (ensemble, lgbm, chronos,
nhits, tft) and writes results to the model_evaluation_metrics table.

Metrics computed:
  - WMAPE (Weighted Mean Absolute Percentage Error)
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - MAPE (Mean Absolute Percentage Error)
  - R² (Coefficient of Determination)

Horizon bands: 1-7, 8-14, 15-21, 22-42 (overall)

Usage:
    .venv/bin/python src/metrics.py                    # evaluate latest run
    .venv/bin/python src/metrics.py --as-of 2019-01-17  # evaluate specific date
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import CFG, PROCESSED, build_panel, load_tables  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

HORIZON_BANDS = [
    ("1-7", 1, 7),
    ("8-14", 8, 14),
    ("15-21", 15, 21),
    ("22-42", 22, 42),
]


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denom) if denom else np.nan


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {m: np.nan for m in ["wmape", "mae", "rmse", "mape", "r2"]}
    return {
        "wmape": wmape(yt, yp),
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mape": float(mean_absolute_percentage_error(yt, yp) * 100),
        "r2": float(r2_score(yt, yp)),
    }


def evaluate_forecasts(as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Evaluate all available forecasts against actual demand at a given as_of_date."""
    sys.path.insert(0, str(ROOT / "db"))
    from connection import scalar
    from inputs import load_forecasts as db_load_forecasts

    tables = load_tables()
    panel = build_panel(tables)

    actual = panel.set_index(["sku_id", "region", "date"])["units"]

    fcst = db_load_forecasts(as_of_date)
    if fcst.empty:
        print(f"[metrics] No forecasts found for {as_of_date.date()}")
        return pd.DataFrame()

    rows = []
    for horizon_label, lo, hi in HORIZON_BANDS:
        band_fcst = fcst[(fcst["horizon"] >= lo) & (fcst["horizon"] <= hi)]
        if band_fcst.empty:
            continue

        keys = list(zip(
            band_fcst["sku_id"],
            band_fcst["region"],
            pd.DatetimeIndex(band_fcst["forecast_date"]),
        ))
        truth = np.array([actual.get(k, np.nan) for k in keys], dtype=float)
        pred = band_fcst["p50"].values

        metrics = compute_all_metrics(truth, pred)
        for metric_name, metric_value in metrics.items():
            rows.append({
                "model_name": "ensemble",
                "metric_name": metric_name,
                "metric_value": round(metric_value, 6) if np.isfinite(metric_value) else None,
                "horizon_band": horizon_label,
            })

    # Overall metrics (all horizons)
    keys = list(zip(fcst["sku_id"], fcst["region"], pd.DatetimeIndex(fcst["forecast_date"])))
    truth = np.array([actual.get(k, np.nan) for k in keys], dtype=float)
    pred = fcst["p50"].values
    metrics = compute_all_metrics(truth, pred)
    for metric_name, metric_value in metrics.items():
        rows.append({
            "model_name": "ensemble",
            "metric_name": metric_name,
            "metric_value": round(metric_value, 6) if np.isfinite(metric_value) else None,
            "horizon_band": None,
        })

    return pd.DataFrame(rows)


def write_metrics_to_db(metrics_df: pd.DataFrame, as_of_date) -> int:
    """Write computed metrics to model_evaluation_metrics table."""
    if metrics_df.empty:
        print("[metrics] No metrics to write")
        return 0

    sys.path.insert(0, str(ROOT / "db"))
    from connection import get_engine, write_run
    from sqlalchemy import text

    engine = get_engine()
    d = pd.Timestamp(as_of_date).date()

    # Delete previous metrics for this date
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM model_evaluation_metrics WHERE as_of_date = :d"),
            {"d": d},
        )

    # Insert new metrics
    records = []
    for _, row in metrics_df.iterrows():
        records.append({
            "as_of_date": d,
            "model_name": row["model_name"],
            "metric_name": row["metric_name"],
            "metric_value": row["metric_value"],
            "horizon_band": row.get("horizon_band"),
        })

    if records:
        from connection import insert_df
        df = pd.DataFrame(records)
        insert_df(df, "model_evaluation_metrics")

    print(f"  + model_evaluation_metrics     {len(records):>3} rows written for {d}")
    return len(records)


def main():
    ap = argparse.ArgumentParser(description="Compute model evaluation metrics")
    ap.add_argument("--as-of", type=str, default=None,
                    help="Evaluate forecasts for this date (YYYY-MM-DD); default = latest")
    args = ap.parse_args()

    if args.as_of:
        as_of = pd.Timestamp(args.as_of)
    else:
        sys.path.insert(0, str(ROOT / "db"))
        from connection import scalar
        as_of_date = scalar("SELECT MAX(as_of_date) FROM forecasts_final")
        as_of = pd.Timestamp(as_of_date) if as_of_date else pd.Timestamp(CFG["project"]["as_of_date"])

    print(f"\n=== model evaluation @ {as_of.date()} ===")
    metrics_df = evaluate_forecasts(as_of)
    if not metrics_df.empty:
        write_metrics_to_db(metrics_df, as_of)
        print("\n=== Metrics Summary ===")
        pivot = metrics_df.pivot_table(
            index="metric_name", columns="horizon_band", values="metric_value"
        )
        print(pivot.round(4).to_string())
    print("done.\n")


if __name__ == "__main__":
    main()

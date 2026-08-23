from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import CFG, PROCESSED, build_panel, load_tables  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.yaml"
SNAPSHOT_DIR = PROCESSED / "demo_snapshot"

SNAPSHOT_FILES = [
    "forecasts_final.parquet",
    "ensemble_meta.yaml",
    "replenishment_orders.parquet",
    "transfer_plan.parquet",
    "writeoff_risk.parquet",
    "simulation_daily.parquet",
    "kpi_summary.csv",
    "alerts.json",
    "alert_digest.md",
]

LIVE_KEYS = [
    "sku_id",
    "region",
    "atc_code",
    "forecast_date",
    "horizon",
    "p10",
    "p50",
    "p90",
    "momentum_u",
    "flu_ratio",
    "sense_adjustment",
]


def take_snapshot(as_of: pd.Timestamp) -> None:
    if SNAPSHOT_DIR.exists():
        return
    SNAPSHOT_DIR.mkdir()
    for name in SNAPSHOT_FILES:
        src = PROCESSED / name
        if src.exists():
            shutil.copy(src, SNAPSHOT_DIR / name)
    (SNAPSHOT_DIR / "as_of.txt").write_text(
        str(yaml.safe_load(CONFIG.read_text())["project"]["as_of_date"])
    )
    print(f"demo snapshot saved to {SNAPSHOT_DIR.name}/")


def restore_demo() -> None:
    if not SNAPSHOT_DIR.exists():
        print("no demo snapshot found")
        return
    for f in SNAPSHOT_DIR.iterdir():
        if f.name != "as_of.txt":
            shutil.copy(f, PROCESSED / f.name)
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["project"]["as_of_date"] = (SNAPSHOT_DIR / "as_of.txt").read_text().strip()
    CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"demo state restored (as-of {cfg['project']['as_of_date']}); rerun dashboard to refresh")


def chronos_zero_shot(panel: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame | None:
    import torch

    from torch_models import CHRONOS_MODEL_ID, HORIZON

    try:
        try:
            from transformers import BaseChronosPipeline

            pipe = BaseChronosPipeline.from_pretrained(
                CHRONOS_MODEL_ID,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.float32,
            )
        except Exception:
            from chronos import BaseChronosPipeline

            pipe = BaseChronosPipeline.from_pretrained(CHRONOS_MODEL_ID)
    except Exception as e:
        print(f"chronos unavailable ({e}); LGBM-only refresh")
        return None

    hist = panel[panel["date"] <= as_of]
    contexts, uids = [], []
    for (sku, region), grp in hist.groupby(["sku_id", "region"]):
        ctx = grp.sort_values("date")["units"].tail(512).to_numpy(dtype=np.float32)
        if len(ctx) < 30:
            continue
        contexts.append(torch.from_numpy(ctx))
        uids.append((sku, region))

    raw = pipe.predict(contexts, prediction_length=HORIZON)
    if isinstance(raw, tuple):
        raw = raw[0]
    q = np.asarray(raw, dtype=float)
    if q.ndim == 3 and q.shape[1] < q.shape[-1]:
        q = np.transpose(q, (0, 2, 1))
    n_levels = q.shape[-1]
    mid_idx = int(np.argmin(np.abs(np.linspace(0.1, 0.9, n_levels) - 0.5)))

    fut_dates = [as_of + pd.Timedelta(days=int(i)) for i in range(1, HORIZON + 1)]
    rows = []
    for i, (sku, region) in enumerate(uids):
        for j, d in enumerate(fut_dates):
            rows.append(
                {
                    "sku_id": sku,
                    "region": region,
                    "forecast_date": d,
                    "horizon": j + 1,
                    "chronos_p10_tmp": max(q[i][j][0], 0),
                    "chronos_p50": max(q[i][j][mid_idx], 0),
                    "chronos_p90_tmp": max(q[i][j][-1], 0),
                }
            )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str, default=None, help="forecast origin YYYY-MM-DD; default = last date in demand history")
    ap.add_argument("--restore-demo", action="store_true", help="restore demo forecasts taken before a rolling run")
    args = ap.parse_args()

    if args.restore_demo:
        restore_demo()
        return

    tables = load_tables()
    demand_max = tables["demand_history"]["date"].max()
    as_of = pd.Timestamp(args.date) if args.date else demand_max
    if as_of > demand_max:
        raise SystemExit(f"origin {as_of.date()} is beyond demand history ({demand_max.date()}); upload newer sales first")

    take_snapshot(as_of)

    from ensemble import apply_sensing, lgbm_forecasts, sensing_factors, wmape

    panel = build_panel(tables)
    lgbm_f = lgbm_forecasts(panel, as_of).rename(
        columns={"p10_lgbm": "lgbm_p10_tmp", "p50_lgbm": "lgbm_p50", "p90_lgbm": "lgbm_p90_tmp"}
    )

    chrono_f = chronos_zero_shot(panel, as_of)
    parts = [lgbm_f] + ([chrono_f] if chrono_f is not None and len(chrono_f) else [])

    merged = parts[0]
    for sub in parts[1:]:
        merged = merged.merge(sub, on=["sku_id", "region", "forecast_date", "horizon"], how="inner")

    meta_p = PROCESSED / "ensemble_meta.yaml"
    w_all = (yaml.safe_load(meta_p.read_text()) or {}).get("ensemble_weights") if meta_p.exists() else None
    w_all = {k: float(v) for k, v in (w_all or {}).items()}
    avail = {"lgbm"} | ({"chronos"} if "chronos_p50" in merged.columns else set())
    weights = {k: v for k, v in (w_all or {"lgbm": 1}).items() if k in avail} or {"lgbm": 1.0}
    tot = sum(weights.values())
    weights = {k: v / tot for k, v in weights.items()}
    print(f"rolling blend @ {as_of.date()}:", {k: round(v, 3) for k, v in weights.items()})

    for q in ["10", "50", "90"]:
        acc = None
        for m, wgt in weights.items():
            src = f"{m}_p{q}" if q == "50" else f"{m}_p{q}_tmp"
            part = merged[src] * wgt
            acc = part if acc is None else acc + part
        merged[f"p{q}"] = acc

    atc_map = panel.drop_duplicates("sku_id").set_index("sku_id")["atc_code"]
    merged["atc_code"] = merged["sku_id"].map(atc_map)
    diag, _ = sensing_factors(panel, as_of)
    merged = apply_sensing(merged, diag)

    merged[LIVE_KEYS].to_parquet(PROCESSED / "forecasts_final.parquet", index=False)

    cfg = yaml.safe_load(CONFIG.read_text())
    prev_asof = cfg["project"]["as_of_date"]
    cfg["project"]["as_of_date"] = str(as_of.date())
    CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False))

    (PROCESSED / "rolling_meta.yaml").write_text(
        yaml.safe_dump(
            {
                "mode": "rolling_daily",
                "as_of": as_of.date().isoformat(),
                "previous_as_of": prev_asof,
                "models_used": sorted(weights),
                "rows": len(merged),
            }
        )
    )

    actual = panel.set_index(["sku_id", "region", "date"])["units"]
    keys = list(zip(merged["sku_id"], merged["region"], pd.DatetimeIndex(merged["forecast_date"])))
    truth = np.array([actual.get(k, np.nan) for k in keys], dtype=float)
    mask = ~np.isnan(truth)
    if mask.any():
        print(f"realized WMAPE vs actuals: {wmape(truth[mask], merged.loc[mask, 'p50']):.4f}")
    else:
        print("no actuals beyond origin yet (normal in live ops)")

    top = (
        merged.groupby("atc_code")["sense_adjustment"].mean().sort_values().tail(4) * 100
    ).round(1)
    print("strongest sensed uplift by category:")
    print(top.astype(str).add("%").to_string())


if __name__ == "__main__":
    main()

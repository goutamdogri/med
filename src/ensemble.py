from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import CFG, FEATURES, HORIZONS, PROCESSED, build_covariates, build_panel, load_tables, make_supervised, melt_horizons  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
LGB_PARAMS = {
    "objective": "l1",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_samples": 60,
    "n_estimators": 700,
    "verbosity": -1,
}
QUANTILE_PARAMS = dict(LGB_PARAMS)
SENSITIVE_ATC = set(CFG["flu_season"]["sensitive_atc"])
MOMENTUM_LAMBDA = 0.5
MOMENTUM_TAU_DAYS = 5.0
MAX_ADJ = 0.20


def wmape(y_true, y_pred):
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denom) if denom else np.nan


def lgbm_forecasts(panel, as_of, tables=None):
    sup7 = make_supervised(panel, stride=7)
    melted_train = melt_horizons(sup7, panel)
    train = melted_train[melted_train["cutoff_date"] <= as_of]

    cov = build_covariates(tables, panel) if tables else None
    sup_all = make_supervised(panel, stride=1)
    inf_sup = sup_all[sup_all["date"] == as_of]
    infer = melt_horizons(inf_sup, panel, require_target=False, cov=cov)

    preds = {}
    for tag, params in [
        ("p50", LGB_PARAMS),
        ("q10", {**QUANTILE_PARAMS, "objective": "quantile", "alpha": 0.10}),
        ("q90", {**QUANTILE_PARAMS, "objective": "quantile", "alpha": 0.90}),
    ]:
        m = lgb.LGBMRegressor(**params)
        m.fit(train[FEATURES], train["target"])
        p = m.predict(infer[FEATURES])
        preds[tag] = np.clip(p, 0, None)

    out = infer[["sku_id", "region", "forecast_date"]].copy()
    out["horizon"] = (
        (
            pd.to_datetime(out["forecast_date"]) - as_of
        ).dt.days
    )
    out["p10_lgbm"] = preds["q10"]
    out["p50_lgbm"] = preds["p50"]
    out["p90_lgbm"] = preds["q90"]
    return out[["sku_id", "region", "forecast_date", "horizon", "p10_lgbm", "p50_lgbm", "p90_lgbm"]]


def ensemble_weights():
    torch_scores = pd.read_csv(MODELS / "backtest_torch.csv")
    lgb_bt = pd.read_csv(MODELS / "backtest_lgbm.csv")
    as_of = str(pd.Timestamp(CFG["project"]["as_of_date"]).date())
    lgb_row = lgb_bt[(lgb_bt["model"] == "lightgbm") & (lgb_bt["origin"] == as_of)]
    scores = dict(zip(torch_scores["model"], torch_scores["wmape_asof_origin"]))
    if len(lgb_row):
        scores["lgbm"] = float(lgb_row["wmape"].iloc[0])
    inv = {k: 1.0 / v for k, v in scores.items()}
    total = sum(inv.values())
    return {k: v / total for k, v in inv.items()}, scores


def sensing_factors(panel, as_of):
    d = panel[["date", "sku_id", "region", "atc_code", "units", "flu_index"]]
    hist = d[d["date"] <= as_of]
    g = hist.groupby(["sku_id", "region"])

    recent = hist[hist["date"] > as_of - pd.Timedelta(days=int(7))]
    baseline = hist[
        (hist["date"] <= as_of - pd.Timedelta(days=int(7)))
        & (hist["date"] > as_of - pd.Timedelta(days=int(35)))
    ]
    recent_mean = recent.groupby(["sku_id", "region"])["units"].mean()
    base_mean = baseline.groupby(["sku_id", "region"])["units"].mean()
    momentum = (recent_mean / base_mean.replace(0, np.nan)).fillna(1.0)

    flu_now = hist[hist["date"] == as_of].groupby(["atc_code"])["flu_index"].first()
    flu_prev = (
        hist[hist["date"] == as_of - pd.Timedelta(days=int(14))]
        .groupby(["atc_code"])[["flu_index"]]
        .first()
    )
    flu_ratio = (flu_now / flu_prev.replace(0, np.nan)["flu_index"]).dropna()
    if len(flu_ratio) < len(flu_now):
        flu_ratio = flu_ratio.reindex(flu_now.index).fillna(1.0)

    diag = momentum.rename("momentum_u").reset_index()
    atc_map = d.drop_duplicates("sku_id").set_index("sku_id")["atc_code"]
    diag["atc_code"] = diag["sku_id"].map(atc_map)
    diag["flu_ratio"] = diag["atc_code"].map(flu_ratio)
    return diag, flu_ratio


def apply_sensing(fcst: pd.DataFrame, diag: pd.DataFrame) -> pd.DataFrame:
    fc = fcst.merge(diag[["sku_id", "region", "momentum_u", "flu_ratio"]],
                    on=["sku_id", "region"], how="left")

    def damp(h):
        return np.exp(-np.maximum(h - 1, 0) / MOMENTUM_TAU_DAYS)

    mom_adj = MOMENTUM_LAMBDA * (fc["momentum_u"].clip(0.5, 2.0) - 1.0)
    flu_sens = fc["atc_code"].isin(SENSITIVE_ATC).astype(float)
    flu_adj = 0.25 * (fc["flu_ratio"].clip(0.5, 2.0) - 1.0) * flu_sens
    total_adj = ((mom_adj + flu_adj) * damp(fc["horizon"])).clip(-MAX_ADJ, MAX_ADJ)
    fc["sense_adjustment"] = total_adj
    for c in ["p10", "p50", "p90"]:
        fc[c] = (fc[c] * (1 + total_adj)).clip(lower=0)
    return fc


def demand_fingerprint() -> str:
    import hashlib
    sys.path.insert(0, str(ROOT / "db"))
    from connection import scalar  # noqa: E402
    n = scalar("SELECT COUNT(*) FROM demand_history")
    mx = scalar("SELECT MAX(date)::text FROM demand_history")
    return hashlib.md5(f"{n}:{mx}".encode()).hexdigest()


def torch_forecasts_fresh(as_of: pd.Timestamp) -> bool:
    sidecar = MODELS / "forecasts_torch.meta.json"
    if not sidecar.exists():
        return False
    import json as _json

    meta = _json.loads(sidecar.read_text())
    if str(meta.get("as_of")) != str(as_of.date()):
        return False
    return meta.get("demand_hash") == demand_fingerprint()


def main():
    as_of = pd.Timestamp(CFG["project"]["as_of_date"])
    tables = load_tables()
    panel = build_panel(tables)

    lgbm_f = lgbm_forecasts(panel, as_of).rename(
        columns={"p10_lgbm": "lgbm_p10_tmp", "p50_lgbm": "lgbm_p50", "p90_lgbm": "lgbm_p90_tmp"}
    )
    parts = [lgbm_f]
    models_used = ["lgbm"]

    if torch_forecasts_fresh(as_of):
        torch_f = pd.read_csv(MODELS / "forecasts_torch.csv")
        for m in ["tft", "chronos", "nhits"]:
            sub = torch_f[torch_f["model"] == m][
                ["sku_id", "region", "forecast_date", "horizon", "p10", "p50", "p90"]
            ].rename(
                columns={
                    "p10": f"{m}_p10_tmp",
                    "p50": f"{m}_p50",
                    "p90": f"{m}_p90_tmp",
                }
            )
            if len(sub):
                parts.append(sub)
                models_used.append(m)
    else:
        print("torch forecasts stale (data or as_of changed) -> LGBM-only ensemble; rerun torch_models.py to include NNs")

    merged = parts[0]
    for sub in parts[1:]:
        merged = merged.merge(sub, on=["sku_id", "region", "forecast_date", "horizon"], how="inner")

    weights, scores = ensemble_weights()
    weights = {k: v for k, v in weights.items() if k in models_used}
    wsum = sum(weights.values())
    weights = {k: v / wsum for k, v in weights.items()}
    print("models used:", models_used)
    print("ensemble weights:", {k: round(v, 3) for k, v in weights.items()})

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

    keep = [
        "sku_id",
        "region",
        "atc_code",
        "forecast_date",
        "horizon",
        "p10",
        "p50",
        "p90",
        *[f"{m}_p50" for m in models_used if f"{m}_p50" in merged.columns],
        "momentum_u",
        "flu_ratio",
        "sense_adjustment",
    ]
    keep = list(dict.fromkeys(keep))
    merged = merged[keep]

    summary = {
        "as_of": as_of.date().isoformat(),
        "demand_hash": demand_fingerprint(),
        "models_used": models_used,
        "model_wmapes_same_origin": scores,
        "ensemble_weights": weights,
        "rows": len(merged),
    }
    import yaml

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "ensemble_meta.yaml").write_text(yaml.safe_dump(summary))
    print(yaml.safe_dump(summary))

    actual = panel.set_index(["sku_id", "region", "date"])["units"]
    keys = list(
        zip(merged["sku_id"], merged["region"], pd.DatetimeIndex(merged["forecast_date"]))
    )
    truth = np.array([actual.get(k, np.nan) for k in keys], dtype=float)
    mask = ~np.isnan(truth)
    realized_wmape = wmape(truth[mask], merged.loc[mask, "p50"])
    print("ENSEMBLE WMAPE @ as-of origin:", round(realized_wmape, 4))

    # write forecasts to PostgreSQL OUTPUT tables
    try:
        sys.path.insert(0, str(ROOT / "db"))
        import outputs as db_out

        prev_cfg = None
        db_out.write_forecasts(merged, as_of, models_used, weights)
        db_out.write_run_log(
            as_of, prev_cfg, models_used, weights, realized_wmape,
            len(merged), triggered_by="retrain",
        )
    except Exception as exc:
        print(f"[ensemble] DB write failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()

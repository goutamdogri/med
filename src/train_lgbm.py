from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (  # noqa: E402
    CFG,
    FEATURES,
    HORIZONS,
    build_panel,
    load_tables,
    make_supervised,
    melt_horizons,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
PROCESSED = ROOT / "data" / "processed"
MODELS.mkdir(exist_ok=True)

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

BACKTEST_ORIGINS = ["2018-04-01", "2018-07-01", "2018-10-01", "2019-01-15"]


def wmape(y_true, y_pred):
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denom) if denom else np.nan


def baseline_predictions(test: pd.DataFrame, panel: pd.DataFrame) -> dict[str, pd.Series]:
    lookup = panel.set_index(["sku_id", "region", "date"])["units"]
    idx = list(
        zip(test["sku_id"], test["region"], pd.DatetimeIndex(test["forecast_date"]))
    )
    sn7_idx = [
        (s, r, d - pd.Timedelta(days=int(7))) for s, r, d in idx
    ]
    sn364_idx = [
        (s, r, d - pd.Timedelta(days=int(364))) for s, r, d in idx
    ]
    naive7 = np.array([lookup.get(k, np.nan) for k in sn7_idx], dtype=float)
    naive364 = np.array([lookup.get(k, np.nan) for k in sn364_idx], dtype=float)

    roll28 = (
        panel.groupby(["sku_id", "region"])["units"].transform(
            lambda s: s.shift(1).rolling(28).mean()
        )
    )
    r28_lookup = panel.assign(roll28=roll28).set_index(["sku_id", "region", "date"])[
        "roll28"
    ]
    ma28 = np.array([r28_lookup.get(k, np.nan) for k in idx], dtype=float)
    return {
        "naive_7": naive7,
        "seasonal_naive_364": naive364,
        "ma_28": ma28,
    }


def evaluate(pred: pd.Series | np.ndarray, test: pd.DataFrame) -> dict:
    out = {"wmape": wmape(test["target"].values, np.asarray(pred))}
    for lo, hi in [(1, 7), (8, 14), (15, 21), (22, 28)]:
        m = test["horizon"].between(lo, hi)
        out[f"wmape_h{lo}_{hi}"] = wmape(
            test.loc[m, "target"].values, np.asarray(pred)[m.values]
        )
    return out


def run_backtest(panel, supervised, melted):
    rows = []
    for origin in BACKTEST_ORIGINS:
        o = pd.Timestamp(origin)
        train = melted[melted["forecast_date"] <= o]
        cutoffs = supervised.loc[supervised["date"] <= o, "date"].unique()
        if len(cutoffs) == 0:
            continue
        last_cutoff = max(cutoffs)
        test_mask = melted["cutoff_date"] == last_cutoff
        test_mask &= melted["forecast_date"] > o
        test = melted[test_mask].copy()

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(train[FEATURES], train["target"])
        pred = model.predict(test[FEATURES])

        row = {"origin": origin, "model": "lightgbm", **evaluate(pred, test)}
        rows.append(row)
        for name, base in baseline_predictions(test, panel).items():
            valid = ~np.isnan(base)
            sub = test[valid]
            row_b = {"origin": origin, "model": name, **evaluate(base[valid], sub)}
            rows.append(row_b)
        print(f"origin {origin}: done")
    return pd.DataFrame(rows)


def fit_production(panel, melted, as_of):
    train = melted[melted["cutoff_date"] <= as_of]
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(train[FEATURES], train["target"])
    model.booster_.save_model(str(MODELS / "lgbm_global.txt"))
    imp = pd.DataFrame(
        {"feature": FEATURES, "gain": model.feature_importances_}
    ).sort_values("gain", ascending=False)
    imp.to_csv(MODELS / "lgbm_feature_importance.csv", index=False)
    print("production model trained on", train["cutoff_date"].max(), "| rows:", len(train))
    print(imp.head(10).to_string(index=False))
    return model


def main():
    tables = load_tables()
    panel = build_panel(tables)
    supervised = make_supervised(panel, stride=7)
    melted = melt_horizons(supervised, panel)

    bt = run_backtest(panel, supervised, melted)
    bt.to_csv(MODELS / "backtest_lgbm.csv", index=False)
    summary = bt.groupby("model")[
        ["wmape", "wmape_h1_7", "wmape_h8_14", "wmape_h15_21", "wmape_h22_28"]
    ].mean()
    print("\n=== Backtest summary (mean across origins) ===")
    print(summary.round(4).to_string())

    as_of = pd.Timestamp(CFG["project"]["as_of_date"])
    fit_production(panel, melted, as_of)


if __name__ == "__main__":
    main()

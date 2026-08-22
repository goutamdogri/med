from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import CFG, PROCESSED, build_panel, load_tables  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HORIZON = CFG["simulation"]["horizon_days"]
CHRONOS_MODEL_ID = "amazon/chronos-bolt-base"



def prepare_series(panel: pd.DataFrame):
    df = panel.copy()
    df["unique_id"] = df["sku_id"] + "|" + df["region"]
    return df


def split_fit_future(df: pd.DataFrame, as_of: pd.Timestamp):
    fit = df[df["date"] <= as_of]
    fut_dates = pd.date_range(as_of + pd.Timedelta(days=int(1)), periods=HORIZON, freq="D")
    cov_region = df.groupby(["region", "date"])[["flu_index", "promo_uplift"]].first()
    pieces = []
    for uid in df["unique_id"].unique():
        region = uid.split("|")[1]
        fut = pd.DataFrame({"ds": fut_dates})
        try:
            cov = cov_region.loc[region].reindex(fut_dates)
        except KeyError:
            cov = pd.DataFrame(index=fut_dates)
        if "flu_index" not in cov.columns or cov["flu_index"].isna().any():
            hist_flu = cov_region.xs(region, level="region")["flu_index"]
            fill = float(hist_flu.iloc[-1])
            fut["flu_index"] = [
                fill if pd.isna(v) else v
                for v in (cov["flu_index"].values if "flu_index" in cov.columns else [])
            ] or fill
            if len(fut) != HORIZON:
                fut["flu_index"] = fill
        else:
            fut["flu_index"] = cov["flu_index"].values
        if (
            "promo_uplift" not in cov.columns
            or len(fut) != HORIZON
            or cov["promo_uplift"].isna().any()
        ):
            fut["promo_uplift"] = 0.0
        else:
            fut["promo_uplift"] = cov["promo_uplift"].values
        if len(fut) != HORIZON:
            continue
        fut["unique_id"] = uid
        pieces.append(fut[["unique_id", "ds", "flu_index", "promo_uplift"]])
    fut_df = pd.concat(pieces, ignore_index=True)

    fit_n = fit.rename(columns={"date": "ds", "units": "y"})[
        ["unique_id", "ds", "y", "flu_index", "promo_uplift"]
    ]
    return fit_n, fut_df


def _fit_one(model_factory, fit_df: pd.DataFrame, fut_df: pd.DataFrame):
    import torch
    from neuralforecast import NeuralForecast

    try:
        nf = NeuralForecast(models=[model_factory("auto")], freq="D")
        nf.fit(df=fit_df, val_size=HORIZON)
        preds = nf.predict(futr_df=fut_df).reset_index()
        del nf
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return preds
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        print(f"gpu fit failed ({type(e).__name__}); retrying on cpu")
        nf = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    nf = NeuralForecast(models=[model_factory("cpu")], freq="D")
    nf.fit(df=fit_df, val_size=HORIZON)
    preds = nf.predict(futr_df=fut_df).reset_index()
    del nf
    return preds


def run_neuralforecast(fit_df: pd.DataFrame, fut_df: pd.DataFrame):
    from neuralforecast.losses.pytorch import MQLoss
    from neuralforecast.models import NHITS, TFT

    NHITS_STEPS = 800
    TFT_STEPS = 600

    def nhits_fac(accel):
        kw = {"devices": 1} if accel == "cpu" else {}
        return NHITS(
            h=HORIZON,
            input_size=168,
            futr_exog_list=["flu_index", "promo_uplift"],
            loss=MQLoss(level=[80]),
            max_steps=NHITS_STEPS,
            val_check_steps=100,
            early_stop_patience_steps=3,
            accelerator=accel,
            scaler_type="robust",
            batch_size=32,
            enable_progress_bar=False,
            **kw,
        )

    def tft_fac(accel):
        kw = {"devices": 1} if accel == "cpu" else {}
        return TFT(
            h=HORIZON,
            input_size=168,
            futr_exog_list=["flu_index", "promo_uplift"],
            loss=MQLoss(level=[80]),
            max_steps=TFT_STEPS,
            accelerator=accel,
            scaler_type="robust",
            hidden_size=48,
            batch_size=8,
            enable_progress_bar=False,
            **kw,
        )

    pieces = []
    for name, fac in [("nhits", nhits_fac), ("tft", tft_fac)]:
        print(f"training {name} ...")
        preds = _fit_one(fac, fit_df, fut_df)
        piece = collect_nf_preds(preds)
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def collect_nf_preds(preds: pd.DataFrame) -> pd.DataFrame:
    frames = []
    model_map = {"NHITS": "nhits", "TFT": "tft"}
    for col_prefix, name in model_map.items():
        if not any(c.startswith(col_prefix) for c in preds.columns):
            continue
        lo_col = next(c for c in preds.columns if c.startswith(col_prefix) and "lo" in c)
        hi_col = next(c for c in preds.columns if c.startswith(col_prefix) and "hi" in c)
        med_col = next(c for c in preds.columns if c.startswith(col_prefix) and "median" in c)
        piece = preds[["unique_id", "ds"]].copy()
        piece["p10"] = preds[lo_col].clip(lower=0)
        piece["p50"] = preds[med_col].clip(lower=0)
        piece["p90"] = preds[hi_col].clip(lower=0)
        piece["model"] = name
        piece.rename(columns={"ds": "forecast_date"}, inplace=True)
        piece["horizon"] = (
            (piece["forecast_date"] - piece["forecast_date"].min()).dt.days + 1
        )
        frames.append(piece)
    return pd.concat(frames, ignore_index=True)


def run_chronos(fit_df: pd.DataFrame) -> pd.DataFrame | None:
    try:
        import torch

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

        contexts = []
        uids = []
        for uid, grp in fit_df.sort_values("ds").groupby("unique_id"):
            ctx = grp["y"].tail(512).to_numpy(dtype=np.float32)
            contexts.append(torch.from_numpy(ctx))
            uids.append(uid)

        raw = pipe.predict(contexts, prediction_length=HORIZON)
        if isinstance(raw, tuple):
            raw = raw[0]
        q = np.asarray(raw, dtype=float)
        if q.ndim == 3 and q.shape[1] < q.shape[-1]:
            q = np.transpose(q, (0, 2, 1))
        n_levels = q.shape[-1]
        mid_idx = int(np.argmin(np.abs(np.linspace(0.1, 0.9, n_levels) - 0.5)))
        q_lo, q_mid, q_hi = q[..., 0], q[..., mid_idx], q[..., -1]

        fut_dates = sorted(
            fit_df["ds"].max() + pd.Timedelta(days=int(i)) for i in range(1, HORIZON + 1)
        )
        rows = []
        for i, uid in enumerate(uids):
            for j, d in enumerate(fut_dates):
                rows.append(
                    {
                        "unique_id": uid,
                        "forecast_date": d,
                        "horizon": j + 1,
                        "p10": max(q_lo[i][j], 0),
                        "p50": max(q_mid[i][j], 0),
                        "p90": max(q_hi[i][j], 0),
                        "model": "chronos",
                    }
                )
        out = pd.DataFrame(rows)
        out[["sku_id", "region"]] = out["unique_id"].str.split("|", expand=True)
        return out
    except Exception as e:
        print(f"chronos unavailable ({e}); skipping")
        return None


def score_forecasts(forecasts: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    actual = panel.set_index(["sku_id", "region", "date"])["units"]
    idx = list(
        zip(
            forecasts["sku_id"],
            forecasts["region"],
            pd.DatetimeIndex(forecasts["forecast_date"]),
        )
    )
    truth = np.array([actual.get(k, np.nan) for k in idx], dtype=float)
    f = forecasts.copy()
    f["actual"] = truth
    f = f.dropna(subset=["actual"])
    rows = []
    for name, grp in f.groupby("model"):
        w = float(np.abs(grp["actual"] - grp["p50"]).sum() / np.abs(grp["actual"]).sum())
        rows.append({"model": name, "wmape_asof_origin": round(w, 4), "n": len(grp)})
    return pd.DataFrame(rows)


def main():
    as_of = pd.Timestamp(CFG["project"]["as_of_date"])
    tables = load_tables()
    panel = build_panel(tables)
    df = prepare_series(panel)
    fit_df, fut_df = split_fit_future(df, as_of)

    all_frames = [run_neuralforecast(fit_df, fut_df)]
    print("neuralforecast done")

    chronos_preds = run_chronos(fit_df)
    if chronos_preds is not None:
        all_frames.append(chronos_preds)
        print("chronos done")

    forecasts = pd.concat(all_frames, ignore_index=True)
    forecasts[["sku_id", "region"]] = forecasts["unique_id"].str.split(
        "|", expand=True
    )
    import hashlib

    demand_hash = hashlib.md5(
        (PROCESSED / "demand_history.parquet").read_bytes()
    ).hexdigest()
    forecasts.to_parquet(PROCESSED / "forecasts_torch.parquet", index=False)
    (PROCESSED / "forecasts_torch.meta.json").write_text(
        json.dumps(
            {
                "as_of": str(pd.Timestamp(CFG["project"]["as_of_date"]).date()),
                "demand_hash": demand_hash,
                "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        )
    )

    scores = score_forecasts(forecasts, panel)
    scores.to_csv(PROCESSED / "backtest_torch.csv", index=False)
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()

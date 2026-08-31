# Deep Learning Forecasting Engine — Deep Dive

**Source file:** `medcare_ml_model/src/torch_models.py`

This document explains the neural network forecasting pipeline: how three deep learning models (N-HiTS, TFT, and Chronos) are trained, how they produce probabilistic forecasts, and how the results are packaged for consumption by the ensemble engine.

---

## Overview

`torch_models.py` is the **GPU-accelerated forecasting sidecar** that runs alongside the LightGBM pipeline. Its outputs feed directly into `ensemble.py`, which blends them with LGBM forecasts using inverse-WMAPE weights.

Three distinct model families are run:

| Model | Framework | Architecture Type | Exogenous Inputs |
|---|---|---|---|
| **N-HiTS** | NeuralForecast | Multi-rate stacked MLP | ✅ flu_index, promo_uplift |
| **TFT** | NeuralForecast | Transformer + LSTM gating | ✅ flu_index, promo_uplift |
| **Chronos** | HuggingFace Transformers | Pre-trained zero-shot LLM | ❌ (univariate only) |

The full execution flow:

```
Historical panel (all SKU × region timeseries)
        │
        ▼
prepare_series()       → add unique_id = "sku_id|region"
        │
        ▼
split_fit_future()     → fit_df (history up to as_of)
                       → fut_df (future covariate scaffold for HORIZON days)
        │
        ├─────────────────────┬──────────────────────┐
        ▼                     ▼                      ▼
run_neuralforecast()    run_neuralforecast()    run_chronos()
  → N-HiTS (P10/P50/P90)  → TFT (P10/P50/P90)  → Chronos (P10/P50/P90)
        │                     │                      │
        └─────────────────────┴──────────────────────┘
                              │
                              ▼
                    Concatenate all forecasts
                              │
                    score_forecasts() → backtest_torch.csv
                              │
                    forecasts_torch.csv
                    forecasts_torch.meta.json   ← for freshness check in ensemble.py
```

---

## Module-Level Constants

```python
HORIZON          = CFG["simulation"]["horizon_days"]
CHRONOS_MODEL_ID = "amazon/chronos-bolt-base"
```

| Constant | Description |
|---|---|
| `HORIZON` | The number of future days to forecast (e.g., 28). Loaded from the project config YAML and shared with all models to ensure consistent output length. |
| `CHRONOS_MODEL_ID` | The HuggingFace Hub identifier for the pre-trained Chronos model. `chronos-bolt-base` is Amazon's compressed, faster variant of the full Chronos-T5 model, optimised for inference speed while retaining accuracy. |

---

## Function Breakdown

### 1. `prepare_series(panel)` — Identifier Construction

```python
def prepare_series(panel: pd.DataFrame):
    df = panel.copy()
    df["unique_id"] = df["sku_id"] + "|" + df["region"]
    return df
```

NeuralForecast and Chronos both require a `unique_id` column to distinguish time series from one another. This function creates a composite key `"sku_id|region"` (e.g., `"MED-001|NORTH"`) using a pipe delimiter. The `|` character is specifically chosen because it is unlikely to appear in either field, making the later `str.split("|")` parse reliable.

---

### 2. `split_fit_future(df, as_of)` — Data Partitioning and Covariate Scaffolding

This function is one of the most complex in the file. It splits the data into a training set and a future covariate dataframe, with careful handling of missing future covariates.

#### Training Split

```python
fit = df[df["date"] <= as_of]
```

All historical data up to and including `as_of` is used for training. No data after `as_of` is seen by the models — this enforces strict temporal integrity.

#### Future Date Grid

```python
fut_dates = pd.date_range(as_of + pd.Timedelta(days=1), periods=HORIZON, freq="D")
```

Generates the exact future dates the models must predict: `as_of + 1` through `as_of + HORIZON`.

#### Per-Series Future Covariate Building

```python
cov_region = df.groupby(["region", "date"])[["flu_index", "promo_uplift"]].first()
```

A region-level covariate lookup is built from the full panel. Covariates are shared across all SKUs within a region (flu activity and promotions are region-wide phenomena, not SKU-specific).

For each `unique_id`:
1. **Extract the region** from `uid.split("|")[1]`.
2. **Reindex** the covariate table to the future date grid.
3. **Flu index fallback:** If flu data is missing for any future date, forward-fill with the last observed value (`hist_flu.iloc[-1]`). This is a sensible default — if we have no scheduled flu data, assume the current level persists.
4. **Promo uplift fallback:** If promotion data is missing, default to `0.0` — assume no unplanned promotion.
5. **Length guard:** If `len(fut) != HORIZON`, skip this series entirely (prevents ragged arrays from breaking the model).

**Output DataFrames:**

| DataFrame | Purpose | Columns |
|---|---|---|
| `fit_n` | Training data | `unique_id`, `ds` (date), `y` (units sold), `flu_index`, `promo_uplift` |
| `fut_df` | Future covariate scaffold | `unique_id`, `ds` (future date), `flu_index`, `promo_uplift` |

> **Note:** `fut_df` contains only covariates — no `y` column. NeuralForecast uses this to inform the models about the future context (e.g., a known upcoming flu season spike or a pre-planned promotion) during inference.

---

### 3. `_fit_one(model_factory, fit_df, fut_df)` — GPU/CPU Failover Training

```python
def _fit_one(model_factory, fit_df, fut_df):
    try:
        nf = NeuralForecast(models=[model_factory("auto")], freq="D")
        nf.fit(df=fit_df, val_size=HORIZON)
        preds = nf.predict(futr_df=fut_df).reset_index()
        del nf
        torch.cuda.empty_cache()
        return preds
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        print(f"gpu fit failed; retrying on cpu")
        ...
    nf = NeuralForecast(models=[model_factory("cpu")], freq="D")
    ...
```

A two-attempt training wrapper with **GPU→CPU automatic failover**:

1. **Attempt 1 (`"auto"`):** Let PyTorch/NeuralForecast choose the accelerator automatically (uses GPU if available).
2. **On `OutOfMemoryError` or `RuntimeError`:** Clear the GPU cache and retry with `accelerator="cpu"`.

Key practices:
- `del nf` + `torch.cuda.empty_cache()` — explicit VRAM cleanup after inference to prevent memory leaks in a long-running sidecar process.
- `val_size=HORIZON` — reserves the last `HORIZON` days of the training data as an internal validation set for early stopping and hyperparameter tuning.

---

### 4. `run_neuralforecast(fit_df, fut_df)` — N-HiTS and TFT Training

This function trains both NeuralForecast models and collects their predictions.

#### Model Configurations

**N-HiTS (Neural Hierarchical Interpolation for Time Series):**
```python
NHITS(
    h=HORIZON,
    input_size=168,
    futr_exog_list=["flu_index", "promo_uplift"],
    loss=MQLoss(level=[80]),
    max_steps=800,
    val_check_steps=100,
    early_stop_patience_steps=3,
    scaler_type="robust",
    batch_size=32,
)
```

| Parameter | Value | Meaning |
|---|---|---|
| `h` | `HORIZON` | Forecast horizon (number of future steps to predict) |
| `input_size` | 168 | Look-back window: 168 days (≈ 6 months) of history as context |
| `futr_exog_list` | `["flu_index", "promo_uplift"]` | Known future covariates to condition the forecast on |
| `loss=MQLoss(level=[80])` | Multi-Quantile Loss at the 80th percentile | Trains the model to output P10, P50, and P90 simultaneously. `level=[80]` instructs it to target the 10th and 90th percentile bounds (the 80% prediction interval) alongside the median |
| `max_steps` | 800 | Maximum gradient update steps |
| `val_check_steps` | 100 | Validate on the held-out set every 100 steps |
| `early_stop_patience_steps` | 3 | Stop training if validation loss doesn't improve for 3 consecutive validation checks (i.e., 300 steps) |
| `scaler_type` | `"robust"` | Normalises each time series using median and IQR instead of mean/std — robust to outlier spikes in demand data |
| `batch_size` | 32 | Sequences per gradient update step |

**Why N-HiTS?** N-HiTS uses hierarchical interpolation across multiple temporal resolutions. It is particularly strong at long-horizon forecasting (14–28 days) because it explicitly decomposes the signal at different time scales (daily noise, weekly patterns, monthly trends).

---

**TFT (Temporal Fusion Transformer):**
```python
TFT(
    h=HORIZON,
    input_size=168,
    futr_exog_list=["flu_index", "promo_uplift"],
    loss=MQLoss(level=[80]),
    max_steps=600,
    scaler_type="robust",
    hidden_size=48,
    batch_size=8,
)
```

| Parameter | Value | Meaning |
|---|---|---|
| `max_steps` | 600 | Fewer steps than N-HiTS because TFT is heavier (Transformer architecture) |
| `hidden_size` | 48 | Internal embedding dimension for the attention mechanism. Kept small (vs the default 64-128) to prevent overfitting on medium-sized pharmaceutical datasets |
| `batch_size` | 8 | Smaller than N-HiTS (8 vs 32) because TFT's self-attention has quadratic memory complexity — fewer sequences per batch prevent VRAM exhaustion |

**Why TFT?** TFT combines variable-selection networks, gated recurrent units (GRUs), and multi-head attention. Its interpretable attention weights can reveal which past time steps and which covariates were most influential for each forecast — making it valuable for business explainability.

---

### 5. `collect_nf_preds(preds)` — Standardising NeuralForecast Output

```python
def collect_nf_preds(preds: pd.DataFrame) -> pd.DataFrame:
    model_map = {"NHITS": "nhits", "TFT": "tft"}
    for col_prefix, name in model_map.items():
        lo_col  = next(c for c in preds.columns if c.startswith(col_prefix) and "lo"  in c)
        hi_col  = next(c for c in preds.columns if c.startswith(col_prefix) and "hi"  in c)
        med_col = next(c for c in preds.columns if c.startswith(col_prefix) and "median" in c)
        piece["p10"] = preds[lo_col].clip(lower=0)
        piece["p50"] = preds[med_col].clip(lower=0)
        piece["p90"] = preds[hi_col].clip(lower=0)
```

NeuralForecast outputs columns with auto-generated names like `NHITS-median`, `NHITS-lo-10`, `NHITS-hi-90`. This function:
1. Detects the relevant columns dynamically using `startswith` + substring matching (future-proofs against minor NeuralForecast version column name changes).
2. Renames them to the canonical `p10`, `p50`, `p90` schema used throughout the pipeline.
3. Clips all values at `0` — negative demand is impossible.
4. Computes `horizon` as `(forecast_date − min_forecast_date).dt.days + 1` (1-indexed: day 1 = tomorrow, day 28 = 4 weeks out).

---

### 6. `run_chronos(fit_df)` — Pre-Trained Zero-Shot Forecasting

```python
pipe = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-base",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.float32,
)
```

Chronos is a **zero-shot foundation model for time series** — it requires no task-specific training. It was pre-trained by Amazon on a massive corpus of diverse time series and can forecast any new series directly from historical context without any fine-tuning.

#### Context Window Extraction

```python
ctx = grp["y"].tail(512).to_numpy(dtype=np.float32)
contexts.append(torch.from_numpy(ctx))
```

Each time series is truncated to its **last 512 observations** (≈ 17 months of daily data). This matches the Chronos model's maximum context window. Longer histories are safely truncated from the left; shorter series are used as-is.

#### Batch Inference

```python
raw = pipe.predict(contexts, prediction_length=HORIZON)
```

All series contexts are passed together in a single batch call. Chronos returns a sample array of shape `(n_series, n_samples, HORIZON)`. The code handles both `(n_series, n_samples, HORIZON)` and the transposed `(n_series, HORIZON, n_samples)` shape via:

```python
if q.ndim == 3 and q.shape[1] < q.shape[-1]:
    q = np.transpose(q, (0, 2, 1))
```

#### Quantile Extraction

```python
n_levels = q.shape[-1]
mid_idx  = int(np.argmin(np.abs(np.linspace(0.1, 0.9, n_levels) - 0.5)))
q_lo, q_mid, q_hi = q[..., 0], q[..., mid_idx], q[..., -1]
```

Chronos samples are treated as an empirical distribution. The code:
- Takes the **first sample** as the lower bound (P10 proxy).
- Finds the **median sample** (closest to 50th percentile) from a linear quantile grid.
- Takes the **last sample** as the upper bound (P90 proxy).

This is a lightweight approximation — proper quantile extraction would sort all samples per step, but this approach works well for an ensemble component.

**Chronos strengths vs limitations:**

| Strength | Limitation |
|---|---|
| No training time required (zero-shot) | Cannot use covariates (flu_index, promo_uplift) — purely univariate |
| Excellent out-of-the-box accuracy on novel series | Truncates context to 512 days — very long history is lost |
| Graceful degradation: `except Exception` catches any import/model failure and returns `None` | Heavier model download (~400MB) required on first run |

If Chronos is unavailable (import error, download failure, or any exception), it returns `None` and is silently excluded from the ensemble. The `main()` function handles this with `if chronos_preds is not None`.

---

### 7. `score_forecasts(forecasts, panel)` — Same-Origin WMAPE Scoring

```python
def score_forecasts(forecasts, panel):
    actual = panel.set_index(["sku_id", "region", "date"])["units"]
    idx    = list(zip(forecasts["sku_id"], forecasts["region"], pd.DatetimeIndex(forecasts["forecast_date"])))
    truth  = np.array([actual.get(k, np.nan) for k in idx], dtype=float)
    f["actual"] = truth
    f = f.dropna(subset=["actual"])
    for name, grp in f.groupby("model"):
        w = float(np.abs(grp["actual"] - grp["p50"]).sum() / np.abs(grp["actual"]).sum())
        rows.append({"model": name, "wmape_asof_origin": round(w, 4), "n": len(grp)})
```

Scores each model against actual observed demand where overlapping data exists. Since the forecast is for the future, only the forecast horizon days for which actuals have already been recorded (e.g., from a historical backfill context) contribute to this score.

The metric is **WMAPE** (Weighted Mean Absolute Percentage Error) — consistent with the LightGBM pipeline's evaluation, ensuring a fair comparison for ensemble weight calculation in `ensemble.py`.

The output is saved to `backtest_torch.csv` with columns `[model, wmape_asof_origin, n]`, which `ensemble_weights()` in `ensemble.py` reads directly to compute inverse-WMAPE blending weights.

---

### 8. `main()` — Orchestration and Artifact Generation

```python
def main():
    as_of  = pd.Timestamp(CFG["project"]["as_of_date"])
    tables = load_tables()
    panel  = build_panel(tables)
    df     = prepare_series(panel)
    fit_df, fut_df = split_fit_future(df, as_of)

    all_frames = [run_neuralforecast(fit_df, fut_df)]   # N-HiTS + TFT
    chronos_preds = run_chronos(fit_df)
    if chronos_preds is not None:
        all_frames.append(chronos_preds)

    forecasts = pd.concat(all_frames, ignore_index=True)
    forecasts[["sku_id", "region"]] = forecasts["unique_id"].str.split("|", expand=True)

    scores = score_forecasts(forecasts, panel)
    scores.to_csv(models_dir / "backtest_torch.csv", index=False)
    forecasts.to_csv(models_dir / "forecasts_torch.csv", index=False)
    ...
    (models_dir / "forecasts_torch.meta.json").write_text(json.dumps(meta))
```

Three artifacts are written to `models/`:

| Artifact | Consumer | Purpose |
|---|---|---|
| `forecasts_torch.csv` | `ensemble.py` | The actual probabilistic forecasts (P10/P50/P90) for all models and series |
| `backtest_torch.csv` | `ensemble.py → ensemble_weights()` | Per-model WMAPE scores used to compute inverse-WMAPE ensemble weights |
| `forecasts_torch.meta.json` | `ensemble.py → torch_forecasts_fresh()` | Staleness guard: stores `as_of` date and a `demand_hash` (MD5 of row count + max date in `demand_history`) so the ensemble can detect if data has changed since the torch models were last run |

#### The Meta JSON (Freshness Guard)

```python
meta = {
    "as_of": str(as_of.date()),
    "demand_hash": hashlib.md5(f"{n}:{mx}".encode()).hexdigest(),
    "models": list(forecasts["model"].unique()),
}
```

This is the contract between `torch_models.py` and `ensemble.py`. If new demand data is ingested or `as_of` changes, the hash will no longer match, and `ensemble.py` will automatically fall back to LGBM-only mode rather than blending stale neural network forecasts with fresh LGBM output.

---

## Output Schema (`forecasts_torch.csv`)

| Column | Type | Description |
|---|---|---|
| `unique_id` | str | Composite key `"sku_id\|region"` |
| `sku_id` | str | Product identifier (parsed from unique_id) |
| `region` | str | Geographic region (parsed from unique_id) |
| `forecast_date` | date | The specific future date being predicted |
| `horizon` | int | Days ahead from `as_of` (1 = tomorrow, HORIZON = furthest day) |
| `p10` | float | 10th percentile demand (pessimistic bound) |
| `p50` | float | Median demand (main forecast signal) |
| `p90` | float | 90th percentile demand (optimistic bound) |
| `model` | str | `"nhits"`, `"tft"`, or `"chronos"` |

---

## Key Design Decisions

| Design Choice | Rationale |
|---|---|
| **MQLoss(level=[80])** | Training with multi-quantile loss simultaneously produces P10, P50, and P90 from a single model training pass — more efficient than training three separate models. The 80% prediction interval (P10–P90) is the standard for pharmaceutical demand planning. |
| **`input_size=168` (6 months)** | Captures weekly and monthly seasonality patterns. Values shorter than 28 days would miss monthly patterns; longer than 6 months adds little additional signal for most pharma SKUs while increasing memory and training time. |
| **`scaler_type="robust"`** | Pharmaceutical demand data has many spikes (e.g., disease outbreaks, bulk orders). Robust scaling (median + IQR) is unaffected by these outliers, unlike mean/std normalisation which would distort the scaled values. |
| **GPU→CPU failover in `_fit_one`** | The sidecar must be able to run on machines without a GPU (development, low-cost cloud instances) and on GPU machines with limited VRAM when running multiple large models sequentially. Silent fallback avoids total pipeline failure. |
| **Chronos as zero-shot only** | Chronos requires no training data — it provides a "free" additional forecast that can meaningfully improve ensemble diversity, especially for new SKUs with very short history where N-HiTS and TFT might underfit. |
| **Metadata freshness contract** | Prevents silent data drift between the torch forecasts and the LGBM forecasts. Without this guard, the ensemble could blend a 2-week-old TFT forecast with a today's LGBM forecast — producing a combined output that misrepresents both. |
| **Pipe delimiter in `unique_id`** | Avoids ambiguity when parsing `sku_id|region` back out. A hyphen would fail if either field itself contains hyphens (common in product codes). |

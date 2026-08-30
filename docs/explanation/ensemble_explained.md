# Ensemble Forecasting Engine — Deep Dive

**Source file:** `medcare_ml_model/src/ensemble.py`

This document explains the complete ensemble forecasting pipeline: how multiple models are trained, combined, sensed, and written to the database to produce the final probabilistic demand forecast.

---

## Overview

The ensemble script is the **brain of the forecast pipeline**. Instead of trusting a single model, it:

1. Generates **probabilistic forecasts** (P10, P50, P90) from LightGBM
2. Optionally ingests pre-computed forecasts from **deep learning models** (TFT, Chronos, N-HiTS)
3. **Blends** model outputs using performance-based weights
4. Applies a **real-time demand sensing** layer that adjusts forecasts based on the latest sales momentum and flu-season signals
5. Writes the final output to **PostgreSQL** and generates a metadata summary

---

## Module-Level Constants

```python
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
QUANTILE_PARAMS = dict(LGB_PARAMS)  # base config, overridden per quantile
SENSITIVE_ATC   = set(CFG["flu_season"]["sensitive_atc"])
MOMENTUM_LAMBDA = 0.5
MOMENTUM_TAU_DAYS = 5.0
MAX_ADJ = 0.20
```

| Constant | Purpose |
|---|---|
| `LGB_PARAMS` | Shared hyperparameter dict for the point-forecast model (L1/MAE objective). See [lgbm_training_process.md](lgbm_training_process.md) for a full parameter breakdown. |
| `QUANTILE_PARAMS` | A copy of `LGB_PARAMS` that is mutated per quantile — the `objective` and `alpha` are overridden inline when calling the q10/q90 models. |
| `SENSITIVE_ATC` | The set of drug/product ATC codes (e.g., antivirals, cold remedies) whose demand is correlated with flu-season activity. Loaded from the project config YAML. |
| `MOMENTUM_LAMBDA` | A scaling factor (λ = 0.5) controlling *how much* of the recent sales momentum is applied as a forecast adjustment. |
| `MOMENTUM_TAU_DAYS` | A time-decay constant (τ = 5 days). The momentum correction decays exponentially as the forecast horizon grows further into the future. |
| `MAX_ADJ` | Clamps the total sensing adjustment to ±20%, preventing wild over-corrections based on noisy short-term signals. |

---

## Function-by-Function Breakdown

### 1. `wmape(y_true, y_pred)` — Evaluation Metric

```python
def wmape(y_true, y_pred):
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denom) if denom else np.nan
```

**Weighted Mean Absolute Percentage Error**. Divides total absolute error by total actual volume. Used throughout the module to score both individual models and the final ensemble.

- If all actuals are zero (denominator is 0), returns `nan` to avoid a division-by-zero.
- Naturally prioritises high-volume SKUs, matching supply chain business priorities.

---

### 2. `lgbm_forecasts(panel, as_of, tables=None)` — The LightGBM Model Block

This is the core LightGBM inference step. It generates three forecasts (P10, P50, P90) in a single pass.

#### Step-by-step:

**Training Data Preparation:**
```python
sup7 = make_supervised(panel, stride=7)
melted_train = melt_horizons(sup7, panel)
train = melted_train[melted_train["cutoff_date"] <= as_of]
```
- Creates the supervised feature table using a weekly stride (`stride=7`).
- "Melts" it into one row per (SKU, region, forecast-date) observation.
- Filters to data with `cutoff_date ≤ as_of` — ensuring no data leakage.

**Inference Data Preparation:**
```python
cov = build_covariates(tables, panel) if tables else None
sup_all = make_supervised(panel, stride=1)
inf_sup = sup_all[sup_all["date"] == as_of]
infer = melt_horizons(inf_sup, panel, require_target=False, cov=cov)
```
- For inference, the stride is `1` (daily) to have a precise row for `as_of`.
- `require_target=False` allows generating a feature row even when the future sales (the label) don't exist yet — since these are *future* forecast dates.
- Optional **covariates** (e.g., calendar events, promotions) are injected here if `tables` is provided.

**Training Three Models:**
```python
for tag, params in [
    ("p50", LGB_PARAMS),
    ("q10", {**QUANTILE_PARAMS, "objective": "quantile", "alpha": 0.10}),
    ("q90", {**QUANTILE_PARAMS, "objective": "quantile", "alpha": 0.90}),
]:
    m = lgb.LGBMRegressor(**params)
    m.fit(train[FEATURES], train["target"])
    p = m.predict(infer[FEATURES])
    preds[tag] = np.clip(p, 0, None)
```

| Tag | Objective | Meaning |
|---|---|---|
| `p50` | `l1` (MAE) | The **best-guess** (median) point forecast. Minimises average absolute error. |
| `q10` | `quantile, α=0.10` | The **pessimistic** lower bound. Minimises quantile loss at the 10th percentile. Model learns to predict a value that actual demand will exceed 90% of the time. |
| `q90` | `quantile, α=0.90` | The **optimistic** upper bound. Actual demand will be below this 90% of the time. |

> `np.clip(p, 0, None)` ensures no negative unit predictions, as demand can never be less than zero.

**Output:** A DataFrame with columns `[sku_id, region, forecast_date, horizon, p10_lgbm, p50_lgbm, p90_lgbm]`.

---

### 3. `ensemble_weights()` — Performance-Based Blending

```python
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
```

The blending strategy is **inverse-WMAPE weighting**: better-performing models get proportionally higher weights.

#### Algorithm:
1. Load the pre-saved backtest results for each model (LGBM and PyTorch models like TFT, N-HiTS, Chronos).
2. Extract the WMAPE at the `as_of` origin for each model.
3. Take the **reciprocal** of each WMAPE: `weight_raw = 1 / wmape`. A lower error → higher reciprocal → higher weight.
4. **Normalise** so all weights sum to exactly 1.0.

**Example:** If LGBM has WMAPE=0.15 and TFT has WMAPE=0.10:
- `inv_lgbm = 1/0.15 ≈ 6.67`, `inv_tft = 1/0.10 = 10.0`
- `total = 16.67`
- `w_lgbm = 6.67/16.67 ≈ 0.40`, `w_tft = 10.0/16.67 ≈ 0.60`

This is fully data-driven — no manual tuning required.

---

### 4. `sensing_factors(panel, as_of)` — Real-Time Demand Signal Detection

This function computes two real-world adjustment signals from the most recent data.

#### Signal 1: Sales Momentum

```python
recent = hist[hist["date"] > as_of - pd.Timedelta(days=7)]
baseline = hist[(hist["date"] <= as_of - pd.Timedelta(days=7)) &
                (hist["date"] > as_of - pd.Timedelta(days=35))]
recent_mean = recent.groupby(["sku_id", "region"])["units"].mean()
base_mean   = baseline.groupby(["sku_id", "region"])["units"].mean()
momentum    = (recent_mean / base_mean.replace(0, np.nan)).fillna(1.0)
```

- **Recent window:** Last 7 days of sales.
- **Baseline window:** The 28-day period before that (days 8–35 before `as_of`).
- `momentum > 1.0` → demand has accelerated recently.
- `momentum < 1.0` → demand has decelerated recently.
- SKUs with no recent/baseline data default to `1.0` (no adjustment).

#### Signal 2: Flu Index Ratio

```python
flu_now  = hist[hist["date"] == as_of].groupby("atc_code")["flu_index"].first()
flu_prev = hist[hist["date"] == as_of - pd.Timedelta(days=14)].groupby("atc_code")["flu_index"].first()
flu_ratio = (flu_now / flu_prev.replace(0, np.nan)["flu_index"]).dropna()
```

- Compares the **current flu index** to the flu index from **14 days ago**, per ATC drug category.
- `flu_ratio > 1.0` → flu activity is rising; demand for sensitive ATC codes (e.g., antivirals) should increase.
- `flu_ratio < 1.0` → flu activity is receding.
- The 14-day comparison window is enough to capture a meaningful trend shift.

---

### 5. `apply_sensing(fcst, diag)` — Adjusting the Ensemble Forecast

```python
def damp(h):
    return np.exp(-np.maximum(h - 1, 0) / MOMENTUM_TAU_DAYS)

mom_adj  = MOMENTUM_LAMBDA * (fc["momentum_u"].clip(0.5, 2.0) - 1.0)
flu_sens = fc["atc_code"].isin(SENSITIVE_ATC).astype(float)
flu_adj  = 0.25 * (fc["flu_ratio"].clip(0.5, 2.0) - 1.0) * flu_sens
total_adj = ((mom_adj + flu_adj) * damp(fc["horizon"])).clip(-MAX_ADJ, MAX_ADJ)

fc["sense_adjustment"] = total_adj
for c in ["p10", "p50", "p90"]:
    fc[c] = (fc[c] * (1 + total_adj)).clip(lower=0)
```

This function translates the raw signals into a bounded percentage adjustment and applies it across all quantiles.

#### Momentum Adjustment (`mom_adj`):

- `momentum` is clipped to `[0.5, 2.0]` — extreme outliers can't dominate.
- Adjustment = λ × (momentum − 1). E.g., momentum=1.3 → `0.5 × 0.3 = +15%` uplift.

#### Flu Adjustment (`flu_adj`):

- A coefficient of `0.25` dampens the flu signal (it is less reliable than direct sales).
- `flu_sens` is a binary mask — only SKUs in `SENSITIVE_ATC` receive a flu adjustment. All others are multiplied by `0`.
- E.g., flu_ratio=1.4 for a sensitive product → `0.25 × 0.4 = +10%` uplift.

#### Time Decay (`damp`):

```
damp(h) = exp(-(h - 1) / τ),  where τ = MOMENTUM_TAU_DAYS = 5
```

| Horizon (h) | Decay Factor |
|---|---|
| 1 (tomorrow) | 1.00 (full signal) |
| 5 | e^(-4/5) ≈ 0.67 |
| 10 | e^(-9/5) ≈ 0.17 |
| 28 | e^(-27/5) ≈ 0.004 |

The sensing adjustment is almost fully applied to the near-term forecast but decays to near-zero for long-horizon predictions. This makes intuitive sense: a current demand surge tells us a lot about tomorrow but almost nothing about what demand will be in 4 weeks.

#### Final Clamp:
`total_adj` is clamped to `[-0.20, +0.20]` — the model can adjust a forecast by at most ±20%. This prevents catastrophic over-corrections from a single noisy signal.

---

### 6. `demand_fingerprint()` — Data Staleness Detection

```python
def demand_fingerprint() -> str:
    n  = scalar("SELECT COUNT(*) FROM demand_history WHERE date <= simulated_today")
    mx = scalar("SELECT MAX(date)::text FROM demand_history WHERE date <= simulated_today")
    return hashlib.md5(f"{n}:{mx}".encode()).hexdigest()
```

Generates a short MD5 hash based on the count and latest date of the `demand_history` table. This "fingerprint" uniquely identifies the current state of the training data.

---

### 7. `torch_forecasts_fresh(as_of)` — Staleness Guard

```python
def torch_forecasts_fresh(as_of) -> bool:
    meta = json.loads((MODELS / "forecasts_torch.meta.json").read_text())
    if str(meta.get("as_of")) != str(as_of.date()):
        return False
    return meta.get("demand_hash") == demand_fingerprint()
```

Reads the `.meta.json` sidecar file written by `torch_models.py`. The pre-computed PyTorch forecasts are only considered **fresh** (and therefore included in the ensemble) if:

1. The `as_of` date in the metadata **matches** the current run's `as_of` date.
2. The `demand_hash` in the metadata **matches** the current data fingerprint.

If either check fails, the script falls back to an **LGBM-only ensemble** and prints a warning: *"torch forecasts stale → LGBM-only ensemble; rerun torch_models.py to include NNs"*. This prevents the ensemble from blending fresh LGBM outputs with stale or mismatched neural network predictions.

---

## The `main()` Function — Full Pipeline Execution

```
main()
  ├── load_tables()
  ├── build_panel()
  │
  ├── lgbm_forecasts()        → [p10_lgbm, p50_lgbm, p90_lgbm] per SKU/region/horizon
  │
  ├── torch_forecasts_fresh() → True/False
  │     ├── True  → read forecasts_torch.csv → add TFT, Chronos, N-HiTS
  │     └── False → LGBM-only mode (print warning)
  │
  ├── merge all model forecast DataFrames on [sku_id, region, forecast_date, horizon]
  │
  ├── ensemble_weights()      → inverse-WMAPE weights per model
  │
  ├── Weighted blend          → p10, p50, p90 (weighted sum across models)
  │
  ├── sensing_factors()       → momentum_u, flu_ratio per SKU
  ├── apply_sensing()         → adjusted p10, p50, p90 (±20% max)
  │
  ├── compute realized WMAPE  → if as-of-date actuals exist, score the ensemble
  │
  ├── write_forecasts()       → PostgreSQL: forecasts_final, forecasts_quantiles
  ├── write_run_log()         → PostgreSQL: model_run_log
  └── ensemble_meta.yaml      → summary: weights, WMAPEs, row count, data hash
```

### Weighted Blending Logic (in detail):

```python
for q in ["10", "50", "90"]:
    acc = None
    for m, wgt in weights.items():
        src  = f"{m}_p{q}" if q == "50" else f"{m}_p{q}_tmp"
        part = merged[src] * wgt
        acc  = part if acc is None else acc + part
    merged[f"p{q}"] = acc
```

For each quantile (P10, P50, P90), the script iterates over all active models and computes a **weighted sum**. The naming convention `_p50` vs `_p{q}_tmp` distinguishes the final ensemble column from the individual model columns (which are kept for diagnostic purposes).

### Output Columns:

| Column | Description |
|---|---|
| `sku_id` | Product identifier |
| `region` | Geographic region |
| `atc_code` | Drug/product ATC classification |
| `forecast_date` | The specific date being predicted |
| `horizon` | Days ahead from `as_of` (1 = tomorrow, 28 = 4 weeks out) |
| `p10` | 10th percentile demand (pessimistic / low-stock risk bound) |
| `p50` | Median demand (main replenishment signal) |
| `p90` | 90th percentile demand (optimistic / safety stock bound) |
| `{model}_p50` | Individual model's median (e.g., `lgbm_p50`, `tft_p50`) |
| `momentum_u` | Raw sales momentum ratio (recent vs baseline) |
| `flu_ratio` | Flu index change ratio vs 14 days ago |
| `sense_adjustment` | Applied adjustment fraction (−0.20 to +0.20) |

---

## Key Design Decisions

| Design Choice | Rationale |
|---|---|
| **Probabilistic output (P10/P50/P90)** | Enables safety-stock and service-level calculations downstream; a single point forecast is insufficient for inventory optimization. |
| **Inverse-WMAPE weighting** | Fully data-driven; no manual tuning. The ensemble automatically shifts weight toward whichever model was most accurate in backtesting. |
| **Demand fingerprinting** | Prevents silent data drift; if new demand data arrives, stale torch forecasts are automatically excluded until rerun. |
| **Exponential time-decay on sensing** | Near-term forecasts are very sensitive to the latest demand signals; long-horizon forecasts should not be moved significantly by a 7-day blip. |
| **ATC-gated flu adjustment** | Prevents flu-index noise from affecting unrelated products (e.g., medical devices, vitamins) — only clinically relevant ATC codes are sensitive. |
| **±20% cap on adjustments** | Acts as a safety rail. No sensing signal, no matter how extreme, can double or halve a forecast overnight. |

# Training Deep Dive: `train_lgbm.py` and `torch_models.py`

---

## Part 1: `train_lgbm.py`

### Core Philosophy: One Global Model

Most forecasting pipelines train one model per SKU or one model per horizon.
This pipeline takes the opposite approach: **a single LightGBM model** is trained on
all 32 SKUs × 6 locations × 42 horizons simultaneously.

The `horizon` value (1 to 42) is just another feature column. The model learns:
"when horizon=1, yesterday's lag matters most; when horizon=35, the annual lag and flu index dominate."

Approximate training set size:
- ~(1400 history days / stride 7) = 200 cutoff dates
- × 32 SKUs × 6 locations = 192 series
- × 42 horizons
- = ~1.6 million rows

---

### LGB_PARAMS — What Each Setting Means

```python
LGB_PARAMS = {
    "objective": "l1",           # MAE loss → predicts the median, robust to outliers
    "learning_rate": 0.05,       # small steps → better generalization
    "num_leaves": 63,            # moderate tree complexity (2^6 - 1)
    "subsample": 0.85,           # use 85% of rows per tree → prevents overfitting
    "colsample_bytree": 0.85,    # use 85% of features per tree → prevents overfitting
    "min_child_samples": 60,     # each leaf must have ≥60 data points
    "n_estimators": 700,         # 700 boosting rounds
    "verbosity": -1,             # silent training
}
```

**Why `l1` (MAE) and not `l2` (MSE)?**

Pharma demand has many spike days: stockout corrections, promo bursts, epidemic surges.
MSE squares the error — one 10× spike dominates the entire loss and distorts the model.
MAE is linear — all errors contribute proportionally. It also makes the model predict
the **median** of demand, which is more conservative and safer for inventory planning
than predicting the mean.

---

### Step 1: Build Training Data

```python
tables = load_tables()
panel = build_panel(tables)
supervised = make_supervised(panel, stride=7)
melted = melt_horizons(supervised, panel)
```

After `melted` is created, one cutoff date (e.g. 2018-04-01) for one SKU-region pair
expands into 42 rows — one per horizon. The lag and rolling features are the SAME
for all 42 rows (they reflect what was known at prediction time). But `futr_flu_index`
and `futr_promo_uplift` differ per row because they reflect conditions on the actual
target date, not the cutoff date.

---

### Step 2: Walk-Forward Backtest — `run_backtest()`

```python
BACKTEST_ORIGINS = ["2018-04-01", "2018-07-01", "2018-10-01", "2019-01-15"]
```

These four dates cover different seasons:
- April: spring, allergy season ending
- July: monsoon onset, flu building
- October: Diwali festival, promotions active
- January 15: peak flu season

For each origin:

```
origin = 2018-07-01
         │
◄────── TRAIN ────────│──── TEST ────────────────────►
all melted rows where  │  the 42-day window whose
forecast_date ≤ origin │  cutoff_date = last available
                       │  date before origin
```

```python
train = melted[melted["forecast_date"] <= o]
model = lgb.LGBMRegressor(**LGB_PARAMS)
model.fit(train[FEATURES], train["target"])
pred = model.predict(test[FEATURES])
```

This is a strict walk-forward evaluation. The model never sees any data from the future
when being evaluated. This mimics real-world conditions: on July 1st you only have
data up to June 30th.

---

### Step 3: Baseline Comparisons

Three statistical baselines compete against LightGBM:

**naive_7:** "Tomorrow's demand equals demand from 7 days ago."
Strength: captures weekly seasonality. Weakness: ignores trend and annual patterns.

**seasonal_naive_364:** "Tomorrow's demand equals demand from exactly 364 days ago."
Strength: captures annual seasonality. Weakness: ignores short-term momentum.

**ma_28:** "Tomorrow's demand equals the 28-day moving average."
Strength: smooth, stable. Weakness: ignores all seasonality.

If LightGBM cannot beat all three baselines, the feature engineering is broken.

---

### Step 4: The wMAPE Metric

```python
def wmape(y_true, y_pred):
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denom)
```

Regular MAPE explodes when y_true is near zero (e.g. a slow-moving SKU on a quiet day).
wMAPE uses the sum of all actuals as denominator, making it immune to near-zero values.
Interpretation: "Total absolute error as a percentage of total demand volume."

The metric is also broken down by horizon bucket:
- `wmape_h1_7`: near-term (days 1–7), should be most accurate
- `wmape_h8_14`: medium near-term
- `wmape_h15_21`: medium far-term
- `wmape_h22_28`: longest horizon evaluated

Near-term should always have lower wMAPE than far-term. If not, the lag features are leaking.

---

### Step 5: Production Training — `fit_production()`

After the backtest confirms the model works, a final model is trained on ALL data:

```python
train = melted[melted["cutoff_date"] <= as_of]   # everything available
model = lgb.LGBMRegressor(**LGB_PARAMS)
model.fit(train[FEATURES], train["target"])
model.booster_.save_model(str(MODELS / "lgbm_global.txt"))
```

`lgbm_global.txt` is a LightGBM text-format model file loadable in milliseconds.
This is what `ensemble.py` uses during inference — it does NOT re-train from scratch.

Feature importances are also saved to `lgbm_feature_importance.csv`, which shows
which of the 19 FEATURES contributed most (by gain) to the model's decisions.

---

### Outputs

- `models/lgbm_global.txt` — the production model
- `data/processed/backtest_lgbm.csv` — wMAPE per origin for LightGBM and baselines
- `data/processed/lgbm_feature_importance.csv` — feature gain rankings

---
---

## Part 2: `torch_models.py`

### Core Philosophy: Sequential Models for Temporal Patterns

Where LightGBM treats each row as independent, neural models see each SKU-region
as a continuous time series and learn temporal dependencies (what happened 3 weeks ago
influences what happens today in ways that raw lag features may not fully capture).

Three models are used:
1. **NHITS** — a hierarchical neural network (trained)
2. **TFT** — Temporal Fusion Transformer (trained)
3. **Chronos** — a pre-trained Amazon foundation model (zero-shot, not trained here)

---

### Step 1: Data Preparation

**`prepare_series(panel)`**

```python
df["unique_id"] = df["sku_id"] + "|" + df["region"]
# result: "N02BE-01|DC_MUMBAI"
```

NeuralForecast needs a `unique_id` column to identify each time series.
There are 32 × 6 = **192 separate series** trained simultaneously.

**`split_fit_future(df, as_of)`**

This splits data into two DataFrames:

`fit_df` — historical training data with columns: `unique_id`, `ds` (date), `y` (units),
`flu_index`, `promo_uplift`. This covers the entire history up to `as_of`.

`fut_df` — future covariate values with columns: `unique_id`, `ds`, `flu_index`,
`promo_uplift`. This covers the 42 forecast days after `as_of`. There is NO `y` column
here — those are the values we want to predict.

The split is clean: `fit_df` has actuals, `fut_df` has only covariates.
The models use `fut_df` at prediction time to understand what promotions and flu levels
will look like during the forecast window.

**Flu index fill logic:** If `flu_index` is missing for some future date, the last known
value is carried forward. Flu outbreaks don't vanish overnight, so this is medically
sound.

---

### Step 2: NHITS Model

```python
NHITS(
    h=42,                                          # forecast 42 days ahead
    input_size=168,                                # look back 168 days (24 weeks)
    futr_exog_list=["flu_index", "promo_uplift"],  # future covariates
    loss=MQLoss(level=[80]),                       # quantile loss
    max_steps=800,
    val_check_steps=100,
    early_stop_patience_steps=3,
    scaler_type="robust",                          # handles outlier-heavy pharma data
    batch_size=32,
)
```

**What NHITS does internally:**
NHITS uses a stack of MLP blocks, each operating at a different time resolution.
The top block captures long-range trends, middle blocks capture seasonal patterns,
and the bottom block captures short-term fluctuations. Each block subtracts its
prediction from the residual and passes the remainder to the next block
(hierarchical interpolation). This is fast and works well for multi-horizon forecasts.

**`input_size=168`**: Each training sample uses 168 days of history as context.
This gives the model 24 weeks of lookback — enough to observe one partial annual cycle
and multiple weekly cycles.

**Early stopping:** Validation is checked every 100 steps. If validation loss doesn't
improve for 3 consecutive checks (300 steps), training stops. The 42-day validation
window (`val_size=HORIZON`) is the last 42 days of each series, held out automatically.

---

### Step 3: TFT Model

```python
TFT(
    h=42,
    input_size=168,
    futr_exog_list=["flu_index", "promo_uplift"],
    loss=MQLoss(level=[80]),
    max_steps=600,
    scaler_type="robust",
    hidden_size=48,     # deliberately small to prevent overfitting
    batch_size=8,       # small batch size → noisy gradients → regularization effect
)
```

**What TFT does internally:**
TFT uses multi-head self-attention to learn which past time steps are most relevant
to predicting each future step. It also has learned variable importance — it can
discover that `flu_index` matters more for R03 (bronchodilators) than for N05C
(hypnotics). The Gated Residual Network components allow the model to selectively
suppress irrelevant inputs.

**Why `hidden_size=48`?** With only 192 time series, a large TFT (hidden_size=256+)
would overfit. Keeping hidden_size small forces the model to generalize across series
rather than memorizing each one.

**Why `batch_size=8`?** Small batches introduce gradient noise, which acts as
implicit regularization. This is a standard trick when dataset size is limited.

---

### Step 4: The `MQLoss` — Getting Probabilistic Forecasts

```python
loss=MQLoss(level=[80])
```

`level=[80]` tells the loss to optimize for the 80% prediction interval,
which means predicting three quantiles simultaneously:
- **p10**: the 10th percentile (lower bound of interval)
- **p50**: the 50th percentile (median forecast)
- **p90**: the 90th percentile (upper bound of interval)

The quantile loss (also called pinball loss) for quantile q is asymmetric:

```
If actual > predicted:  loss = q × (actual - predicted)
If actual < predicted:  loss = (1-q) × (predicted - actual)
```

For p90: under-predicting is penalized 9× more than over-predicting.
For p10: over-predicting is penalized 9× more than under-predicting.

This asymmetry is what forces the model to learn calibrated quantiles.
After training, roughly 80% of actual demand values should fall between p10 and p90.

**Why quantiles matter for inventory:** The replenishment formula uses the spread
between p10 and p90 to estimate demand uncertainty (sigma). Wider spread = more safety
stock needed. Narrow spread = tighter inventory management is safe.

---

### Step 5: GPU Handling — `_fit_one()`

```python
def _fit_one(model_factory, fit_df, fut_df):
    try:
        nf = NeuralForecast(models=[model_factory("auto")], freq="D")
        nf.fit(df=fit_df, val_size=HORIZON)
        preds = nf.predict(futr_df=fut_df)
        torch.cuda.empty_cache()
        return preds
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        nf = NeuralForecast(models=[model_factory("cpu")], freq="D")
        nf.fit(df=fit_df, val_size=HORIZON)
        return nf.predict(futr_df=fut_df)
```

If a GPU is available, training uses it. If it runs out of memory (OOM),
the function catches the error and retries on CPU. `torch.cuda.empty_cache()`
is called after NHITS finishes to free GPU memory before TFT training starts.

---

### Step 6: Chronos — Zero-Shot Foundation Model

```python
pipe = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-base",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
)
```

Chronos is fundamentally different from NHITS and TFT. It is **not trained on this data**.
Amazon pre-trained it on a massive corpus of diverse time series. It performs zero-shot
forecasting: you give it the raw demand history and it predicts the future without any
fine-tuning.

```python
for uid, grp in fit_df.groupby("unique_id"):
    ctx = grp["y"].tail(512).to_numpy(dtype=np.float32)
    contexts.append(torch.from_numpy(ctx))

raw = pipe.predict(contexts, prediction_length=HORIZON)
```

`tail(512)`: Chronos accepts up to 512 context tokens. If history is longer, only
the most recent 512 days are used (most relevant to near-future demand).

Chronos predicts multiple stochastic samples from an implied probability distribution.
These samples form an empirical distribution:
- The 0th sample → p10
- The middle sample → p50
- The last sample → p90

**Limitation:** Chronos cannot use `flu_index` or `promo_uplift` as inputs.
It only sees the raw `y` values. For flu-sensitive ATC codes, this means Chronos
will underperform NHITS and TFT during flu season. The `sensing_factors()` step
in `ensemble.py` partially compensates for this gap.

**Advantage:** Zero training time. Even if you run the pipeline on a machine with no
GPU and limited time, Chronos contributes a forecast immediately.

---

### Step 7: Stale Cache Detection

```python
demand_hash = hashlib.md5(
    (PROCESSED / "demand_history.parquet").read_bytes()
).hexdigest()

metadata = {
    "as_of": "2019-01-17",
    "demand_hash": demand_hash,
    "generated": "2026-08-24 23:40:00",
}
```

Training NHITS and TFT takes 20–60 minutes. To avoid retraining unnecessarily
every day, a sidecar file `forecasts_torch.meta.json` stores the `as_of_date`
and an MD5 hash of the demand data.

When `ensemble.py` runs, it calls `torch_forecasts_fresh()` which checks:
1. Does `forecasts_torch.meta.json` exist?
2. Is the stored `as_of` equal to the current `as_of_date`?
3. Is the stored `demand_hash` equal to the current MD5 of `demand_history.parquet`?

If all three match, the neural forecasts are reused. If the demand data changed
(new sales ingested) or the `as_of_date` advanced, `torch_forecasts_fresh()`
returns False and the ensemble falls back to LGBM-only with a message:
"torch forecasts stale → LGBM-only ensemble; rerun torch_models.py to include NNs"

This is why the **daily rolling pipeline is fast** (LGBM inference only, ~seconds)
and only the **monthly retrain** pays the full neural training cost.

---

## Side-by-Side Model Comparison

```
                LightGBM         NHITS/TFT          Chronos
─────────────────────────────────────────────────────────────────
Training time   ~2 min           20–60 min           0 (none)
Runs daily?     Yes              No                  Yes (zero-shot)
Future covars?  Yes (features)   Yes (futr_exog)     No
Output type     Single value     p10/p50/p90         p10/p50/p90
Horizon method  horizon feature  native multi-step   native multi-step
Retrain freq    Monthly          Monthly             Never
Data format     Tabular (melted) Time series         Time series
```

---

## End-to-End Training Flow

```
features.load_tables() + build_panel()
           │
           ├─────────────────────────────────────────────────────────────
           │                                                             │
    train_lgbm.py                                               torch_models.py
           │                                                             │
    make_supervised(stride=7)                              prepare_series()
    melt_horizons()                                        split_fit_future()
           │                                                       │
    LGBMRegressor(l1 loss)                          NHITS + TFT via MQLoss
    fit(train[FEATURES], train["target"])           + Chronos zero-shot
           │                                                       │
    models/lgbm_global.txt                         forecasts_torch.parquet
    backtest_lgbm.csv                              backtest_torch.csv
           │                                                       │
           └──────────────────┬────────────────────────────────────┘
                              │
                        ensemble.py
                  inverse-wMAPE weighting
                     + demand sensing
                              │
                  forecasts_final.parquet
```

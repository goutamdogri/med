1. LightGBM (train_lgbm.py + features.py)
   Learning setup: one global model across all 192 SKU×region series. Direct multi-horizon: each training row = (cutoff day t, horizon h∈1..42) → predicts units[t+h]. Cutoffs sampled every 7 days → 1.39M rows trained on data ≤ 2019-01-15.
   Feature vector (19):
   Group Features
   Autoregressive lag_1, lag_7, lag_14, lag_28 (units at t−1,7,14,28), lag_364 (year-ago same weekday)
   Rolling stats roll_mean_7, roll_std_7, roll_mean_28 (all shifted −1 → no leakage)
   Future calendar futr_dow, futr_week_of_year, futr_month, futr_is_weekend (of the forecast date)
   Future-known covariates futr_promo_uplift (promo calendar ∩ region), futr_flu_index (surveillance index that leads demand ~8 days)
   Static criticality_code, unit_cost_inr, is_tier2, demand_share, horizon
   Hyperparams: objective=l1 (MAE aligns with WMAPE), lr=0.05, num_leaves=63, n_estimators=700, subsample/colsample=0.85, min_child_samples=60. Two extra clones trained with objective=quantile, alpha=0.10/0.90 produce P10/P90.
2. N-HiTS & TFT (torch_models.py via neuralforecast)
   Input: long-format panel (unique_id=sku|region, ds, y) — same 192 series, no lag engineering (the nets learn their own).
   Param N-HiTS TFT
   Lookback input_size 168 days 168 days
   Horizon h 42 42
   Known-future covariates (futr_exog_list) flu_index, promo_uplift same
   Loss MQLoss(level=[80]) → P10/P50/P90 same
   Steps 800 (early stop patience 3, check@100) 600
   Capacity default MLP widths, batch 32 hidden 48, batch 8 (slimmed for your 3.6GB GPU → CPU fallback)
   Scaling robust scaler same
   TFT additionally runs an LSTM history encoder + interpretable attention over the window; N-HiTS uses multi-rate interpolation stacks.
3. Chronos-Bolt-base (transformers/HF Hub)
   Zero-shot: never trained on our data. Input = raw context only — last 512 daily points per series, no covariates, no calendar. The pretrained tokenizer converts values into tokens, a seq2seq transformer decodes the next 42 steps as 9 quantile levels; we keep Q10/Q50/Q90.
4. Ensemble + sensing overlay (ensemble.py)

- Weights ∝ 1/WMAPE from same-origin backtest → tft .267 · chronos .262 · lgbm .241 · nhits .229
- Sensing adjustment on top: u = mean(last 7d)/mean(prior 28d) momentum + flu_ratio (index today vs t−14, applied only to R03/R06/N02BE/M01AE); combined factor damped exp(−(h−1)/5) and capped ±20%.
  Key contrast worth saying to judges: LightGBM needs hand-built lags; the nets take covariates directly; Chronos takes nothing but history — and still lands within 2 points of the trained TFT.
  ▣ Build · Big Pickle · 1m 25s
  Build·Big PickleOpenCode Zen

# LightGBM Training Process Explanation

This document provides a detailed explanation of the LightGBM (LGBM) model training process implemented in `medcare_ml_model/src/train_lgbm.py`. It covers the data preparation steps, the specific model parameters used, the backtesting evaluation strategy, and the final production training process.

## 1. Data Preparation Pipeline

Before the model can be trained, the raw data must be transformed into a format suitable for supervised machine learning. The `main()` function orchestrates this pipeline:

1. **`load_tables()`**: Loads the raw input datasets (e.g., historical sales, product catalog, calendar features, etc.).
2. **`build_panel()`**: Assembles these tables into a "panel" structure—a time-series cross-sectional dataset indexed by `sku_id`, `region`, and `date`. This establishes a continuous timeline for every product in every location.
3. **`make_supervised(stride=7)`**: Converts the panel data into predictive features. It calculates rolling aggregations, lag features, and other time-based transformations. The `stride=7` implies that features might be calculated with a 7-day step, or it aligns the prediction cadence to weekly intervals to reduce data size or align with business cycles.
4. **`melt_horizons()`**: This is a crucial step for multi-step forecasting. Instead of building 28 separate models for 28 days ahead, the dataset is "melted" (unpivoted). For a given `cutoff_date`, it creates multiple rows—one for each future `forecast_date`. A `horizon` column indicates how many days into the future the prediction is for, and a single `target` column holds the actual value to predict. This allows a single global LightGBM model to predict any horizon by learning the relationship between features and the horizon itself.

## 2. Model Architecture and Parameters (`LGB_PARAMS`)

The core algorithm is a Gradient Boosted Decision Tree (GBDT) using the LightGBM framework (`lgb.LGBMRegressor`). The model's behavior is governed by the following hyperparameters:

*   **`objective: "l1"`**: The model optimizes for Mean Absolute Error (MAE) rather than the default Mean Squared Error (MSE/L2). The L1 objective is less sensitive to extreme outliers, which are common in retail and supply chain demand data (e.g., sudden one-off bulk orders).
*   **`learning_rate: 0.05`**: The step size applied at each boosting iteration. A rate of 0.05 is relatively small, meaning the model learns slowly and carefully. This requires more trees but usually results in better generalization to unseen data.
*   **`num_leaves: 63`**: The maximum number of leaves in a single decision tree. This controls the complexity of the interactions the model can learn. 63 is moderately high, allowing the model to capture complex, non-linear relationships without heavily overfitting.
*   **`subsample: 0.85`**: (Also known as bagging fraction). For every tree built, LightGBM randomly selects 85% of the training rows. This stochastic element reduces variance and prevents the model from memorizing the training data (overfitting).
*   **`colsample_bytree: 0.85`**: For every tree built, LightGBM randomly selects 85% of the available features (columns). This forces the model to learn from a diverse set of features and prevents it from relying entirely on a few dominant ones.
*   **`min_child_samples: 60`**: A regularization parameter. A split in a tree will only be considered if it results in leaf nodes containing at least 60 historical data points. This prevents the model from creating highly specific rules that only apply to a tiny fraction of the data (noise).
*   **`n_estimators: 700`**: The total number of sequential trees to build in the boosting process.
*   **`verbosity: -1`**: Suppresses internal LightGBM logging output during training.

## 3. Evaluation Metric: WMAPE

The model's performance is evaluated using **WMAPE** (Weighted Mean Absolute Percentage Error).

```python
def wmape(y_true, y_pred):
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denom) if denom else np.nan
```

Unlike standard MAPE, which can become infinite if actual sales are zero, WMAPE divides the total absolute error by the total actual volume. This means the metric naturally prioritizes high-volume SKUs over low-volume ones, which aligns perfectly with supply chain priorities (a 10% error on an item selling 10,000 units is more impactful than a 50% error on an item selling 2 units).

## 4. The Training Process

The script performs two distinct training routines: **Backtesting** and **Production Fitting**.

### Phase 1: Backtesting (`run_backtest`)

Backtesting simulates how the model would have performed historically if it were in production. This proves the model's reliability before trusting its future predictions.

1.  **Origins**: The model iterates over specific past dates (`BACKTEST_ORIGINS` = "2018-04-01", "2018-07-01", "2018-10-01", "2019-01-15").
2.  **Train/Test Split**: For each origin:
    *   **Training Set**: Strictly data where the `forecast_date` is on or before the origin date.
    *   **Test Set**: Uses the last available `cutoff_date` before the origin, and tests on `forecast_dates` that occur *after* the origin.
3.  **Evaluation against Baselines**: The model is scored using WMAPE and compared against three naive baseline heuristics to ensure ML is actually adding value:
    *   `naive_7`: Predicts the exact sales from 7 days ago.
    *   `seasonal_naive_364`: Predicts the exact sales from 364 days ago (captures yearly seasonality while preserving day-of-week).
    *   `ma_28`: A simple 28-day moving average.
4.  **Horizon Analysis**: WMAPE is calculated overall and segmented by forecast horizon (Days 1-7, 8-14, 15-21, 22-28) to see how quickly accuracy degrades over time.
5.  **Output**: Results are averaged and saved to `models/backtest_lgbm.csv`.

### Phase 2: Production Training (`fit_production`)

Once backtesting validates the configuration, the script trains the final model meant for active forecasting.

1.  **Data Subset**: It uses all available data up to the `as_of` date specified in the project configuration (`CFG["project"]["as_of_date"]`).
2.  **Training**: A new LightGBM model is initialized with `LGB_PARAMS` and trained on this comprehensive historical dataset.
3.  **Artifact Generation**:
    *   The model weights are saved to `models/lgbm_global.txt` for inference in the production sidecar.
    *   **Feature Importances**: The script calculates how much "gain" (improvement in accuracy) each feature contributed across all trees. This is sorted and saved to `models/lgbm_feature_importance.csv`. This provides interpretability, showing business stakeholders which factors (e.g., rolling averages, day of week, pricing) are driving the forecast.

## 📊 Model Output Tables (Predictions & Decisions Only)

There are **9 output tables** written to MySQL (plus parquet/JSON artifacts on disk). Here they are, organized by which script produces them:

---

### 1. `forecasts_final` — `ensemble.py` / `rolling_forecast.py`

The **core demand forecast** — the primary model output.

| Field | Type | Description |
|---|---|---|
| `as_of_date` | DATE | Forecast origin date |
| `sku_id` | VARCHAR | Product identifier |
| `atc_code` | VARCHAR | Drug therapeutic category |
| `region` | VARCHAR | Store/DC region |
| `forecast_date` | DATE | The day being forecasted |
| `horizon` | INT | Days ahead from `as_of_date` |
| `p10` | DECIMAL | 10th-percentile demand (ensemble) |
| `p50` | DECIMAL | Median demand prediction (ensemble) |
| `p90` | DECIMAL | 90th-percentile demand (ensemble) |
| `momentum_u` | DECIMAL | Real-time momentum multiplier |
| `flu_ratio` | DECIMAL | Flu-index sensing ratio |
| `sense_adjustment` | DECIMAL | Net % adjustment applied by sensing |
| `models_used` | VARCHAR | e.g. `"chronos,lgbm"` |
| `lgbm_weight` | DECIMAL | Ensemble weight given to LightGBM |
| `chronos_weight` | DECIMAL | Ensemble weight given to Chronos |

---

### 2. `replenishment_orders` — `replenishment.py`

The **order quantity recommendation** per SKU/region, derived from `forecasts_final`.

| Field | Type | Description |
|---|---|---|
| `as_of_date` | DATE | Run date |
| `sku_id` | VARCHAR | Product |
| `region` | VARCHAR | Location |
| `criticality` | VARCHAR | `critical` / `high` / `standard` |
| `lead_time_days` | INT | Supplier lead time |
| `service_level` | DECIMAL | Target fill rate (e.g. 0.99) |
| `mu_daily` | DECIMAL | Forecast mean daily demand |
| `sigma_daily` | DECIMAL | Forecast demand std dev |
| `safety_stock` | INT | Computed safety stock units |
| `target_position` | INT | Ideal total inventory position |
| `on_hand` | INT | Current on-hand units |
| `order_qty` | INT | **Recommended order quantity** |
| `order_value_inr` | INT | Order monetary value |
| `days_of_supply_on_hand` | DECIMAL | Current stock coverage in days |
| `status` | VARCHAR | `ok` / `low` / `stockout_risk` |

---

### 3. `transfer_plan` — `allocation.py`

**Inter-location stock transfer recommendations** to rescue near-expiry inventory.

| Field | Type | Description |
|---|---|---|
| `as_of_date` | DATE | Run date |
| `batch_id` | VARCHAR | Source batch being moved |
| `sku_id` | VARCHAR | Product |
| `from_location` | VARCHAR | Source DC/store |
| `to_location` | VARCHAR | Destination DC/store |
| `qty_units` | INT | **Units to transfer** |
| `expiry_date` | DATE | Batch expiry |
| `days_to_expiry` | INT | Days until expiry |
| `transfer_lead_days` | INT | Transit time |
| `value_saved_inr` | INT | Write-off value rescued |
| `reason` | VARCHAR | `expiry_rescue` / `shortage_rescue` |
| `src_days_of_supply_before` | DECIMAL | Source DOS before transfer |

---

### 4. `writeoff_risk` — `allocation.py`

**Residual write-off exposure** after transfers — what the model predicts will still expire unsold.

| Field | Type | Description |
|---|---|---|
| `as_of_date` | DATE | Run date |
| `batch_id` | VARCHAR | Batch at risk |
| `sku_id` | VARCHAR | Product |
| `location` | VARCHAR | Location holding the batch |
| `qty_units` | INT | Total batch quantity |
| `leftover` | DECIMAL | Units that can't be sold before expiry |
| `residual_writeoff_units` | DECIMAL | **Units still written off after transfers** |
| `unit_cost_inr` | INT | Cost per unit |
| `residual_value_inr` | DECIMAL | **Residual write-off value** |
| `expiry_date` | DATE | Batch expiry date |
| `days_to_expiry` | INT | Days until expiry |

---

### 5. `simulation_daily` — `simulate.py`

**Day-by-day simulation log** comparing the ML-proposed policy vs. status quo over a 90-day horizon.

| Field | Type | Description |
|---|---|---|
| `as_of_date` | DATE | Run date |
| `policy` | VARCHAR | `proposed` or `status_quo` |
| `date` | DATE | Simulation day |
| `sku_id` | VARCHAR | Product |
| `region` | VARCHAR | Location |
| `criticality` | VARCHAR | SKU criticality tier |
| `demand` | DECIMAL | Actual/scenario demand for that day |
| `fulfilled` | DECIMAL | Units served |
| `unfulfilled` | DECIMAL | **Unmet demand (stockout)** |
| `expired_units` | DECIMAL | Units expired that day |
| `expired_value_inr` | DECIMAL | Expired stock value |
| `ending_inventory` | DECIMAL | Closing inventory position |

---

### 6. `kpi_summary` — `simulate.py`

**Aggregated KPI comparison** between proposed and status-quo policies.

| Field | Type | Description |
|---|---|---|
| `as_of_date` | DATE | Run date |
| `policy` | VARCHAR | `proposed` or `status_quo` |
| `fill_rate_pct` | DECIMAL | Overall fill rate % |
| `critical_fill_rate_pct` | DECIMAL | Fill rate for critical SKUs only |
| `stockout_units` | INT | Total unmet demand units |
| `critical_stockout_sitedays` | INT | #site-days with critical SKU stockout |
| `writeoff_value_inr` | INT | Total expired stock value |
| `avg_ending_inventory` | DECIMAL | Average ending inventory |

---

### 7. `alerts` — `alerts.py`

**Per-SKU actionable alerts** generated by the model (shortage risk, expiry risk, demand surge).

| Field | Type | Description |
|---|---|---|
| `as_of_date` | DATE | Run date |
| `severity` | VARCHAR | `RED` or `AMBER` |
| `type` | VARCHAR | `shortage_risk` / `expiry_writeoff_risk` / `demand_surge_detected` |
| `sku_id` | VARCHAR | Product (or `*` for region-level) |
| `region` | VARCHAR | Affected region |
| `facts` | JSON | Structured alert data (dos, units_at_risk, etc.) |
| `action` | TEXT | **Recommended action text** |

---

### 8. `alert_digest` — `alerts.py`

**AI-generated executive brief** (Groq cloud LLM / fallback template) summarizing the day's alerts.

| Field | Type | Description |
|---|---|---|
| `as_of_date` | DATE | Run date |
| `review_mode` | VARCHAR | `DAILY_SURGE_MODE` or `WEEKLY_STANDARD` |
| `surge_regions` | VARCHAR | Comma-separated surge regions |
| `red_alert_count` | INT | Number of RED alerts |
| `digest_text` | TEXT | **LLM-generated escalation brief** |
| `model_used` | VARCHAR | `openai/gpt-oss-120b` or `template_fallback` |

---

### 9. `rolling_run_log` — `ensemble.py` / `rolling_forecast.py`

**Audit/run log** — tracks every pipeline execution, not a prediction per se, but a model metadata output.

| Field | Type | Description |
|---|---|---|
| `as_of_date` | DATE | Run date |
| `previous_as_of_date` | DATE | Previous origin date |
| `models_used` | VARCHAR | Models blended |
| `lgbm_weight` | DECIMAL | LightGBM ensemble weight |
| `chronos_weight` | DECIMAL | Chronos ensemble weight |
| `wmape` | DECIMAL | Realized forecast error |
| `forecast_rows` | INT | Number of forecast rows written |
| `status` | VARCHAR | `success` / `error` |
| `error_message` | TEXT | Error if any |
| `run_duration_seconds` | INT | Pipeline runtime |
| `triggered_by` | VARCHAR | `cron` / `manual` / `api` |

---

### What `train_lgbm.py` and `torch_models.py` produce

These are **training** scripts — they don't write output tables to MySQL but produce **intermediate artifacts** on disk:

| Script | Artifact | Purpose |
|---|---|---|
| `train_lgbm.py` | `backtest_lgbm.csv` | LightGBM backtest WMAPE scores (used by ensemble weighting) |
| `train_lgbm.py` | `models/lgbm_global.txt` | Saved LightGBM model file |
| `train_lgbm.py` | `lgbm_feature_importance.csv` | Feature gain scores |
| `torch_models.py` | `forecasts_torch.parquet` | Raw TFT/NHITS/Chronos forecasts (p10/p50/p90 per model) |
| `torch_models.py` | `backtest_torch.csv` | Torch model WMAPE scores |
| `torch_models.py` | `forecasts_torch.meta.json` | Freshness metadata for cache validation |

> These are consumed **as inputs** by `ensemble.py` to produce the final `forecasts_final` output. The torch parquet itself (`forecasts_torch.parquet`) contains the raw per-model forecasts with fields: `unique_id`, `forecast_date`, `horizon`, `p10`, `p50`, `p90`, `model`, `sku_id`, `region`.
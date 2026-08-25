# 📥 Input Data Requirements — Rollover Forecasting & Monthly Retraining

This document answers: **"What data do I need to provide, and when?"** for every stage of the pipeline.

There are two operational cycles:

| Cycle | Trigger | Scripts run |
|---|---|---|
| **Daily Rollover** | Every day (cron / API) | `rolling_forecast.py` → `replenishment.py` → `allocation.py` → `simulate.py` → `alerts.py` |
| **Monthly Retraining** | Once a month | `train_lgbm.py` → `torch_models.py` → `ensemble.py` (full re-ensemble) |

---

## 🔄 Daily Rollover Forecasting — Full Chain

Every day when you do a rollover, the pipeline reads these tables **fresh from MySQL**. You must ensure they are up-to-date **before** running `rolling_forecast.py`.

---

### Step 1 — `rolling_forecast.py` (Demand Forecast)

This is the first script. It reads **all** of the following:

#### 📋 `demand_history` *(most critical — must be updated daily)*

New sales rows must be appended every day. This is the primary signal.

| Field | Type | Required? | Notes |
|---|---|---|---|
| `date` | DATE | ✅ | The sales date — new rows added daily |
| `sku_id` | VARCHAR | ✅ | e.g. `M01AB-01` |
| `atc_code` | VARCHAR | ✅ | e.g. `M01AB` |
| `region` | VARCHAR | ✅ | Location ID e.g. `DC_MUMBAI` |
| `units` | INT | ✅ | Units sold that day |

> **Minimum history required:** 400+ days (≈13+ months). The model needs 364-day lag features and rolling windows. The `rolling_forecast.py` will use `MAX(date)` from this table as the new `as_of_date`.

---

#### 📋 `disease_burden_index` *(update weekly or whenever WHO/surveillance data arrives)*

Maps to `flu_index` inside the pipeline.

| Field | Type | Required? | Notes |
|---|---|---|---|
| `record_date` | DATE | ✅ | Surveillance date |
| `region` | VARCHAR | ✅ | Location ID |
| `index_value` | DECIMAL | ✅ | Flu burden index (0–100+) |

> Flu data can be ahead of sales data — the model uses future flu readings as a covariate. Gaps are forward-filled automatically.

---

#### 📋 `sku_master` *(static — update only when SKUs are added/retired)*

| Field | Type | Required? | Notes |
|---|---|---|---|
| `sku_id` | VARCHAR | ✅ | Unique product ID |
| `brand_name` | VARCHAR | ✅ | Brand name |
| `atc_code` | VARCHAR | ✅ | ATC therapeutic category |
| `criticality` | VARCHAR | ✅ | `critical` / `high` / `standard` / `low` |
| `unit_cost_inr` | INT | ✅ | Cost per unit (used in value calculations) |
| `shelf_life_days` | INT | ✅ | Max shelf life (used in expiry simulation) |

---

#### 📋 `locations` *(static — update when locations are added)*

| Field | Type | Required? | Notes |
|---|---|---|---|
| `location_id` | VARCHAR | ✅ | e.g. `DC_MUMBAI`, `WH_PUNE` |
| `name` | VARCHAR | ✅ | Human-readable name |
| `type` | VARCHAR | ✅ | `metro_dc` or `tier2_wh` |
| `capacity_units` | INT | optional | Warehouse capacity |

> `demand_share` is derived from the `location_demand_summary` derived table (see below), not stored directly in `locations`.

---

#### 📋 `location_demand_summary` *(derived — updated quarterly)*

Used to compute each location's share of national demand. The pipeline always picks the latest quarter.

| Field | Type | Required? | Notes |
|---|---|---|---|
| `location_id` | VARCHAR | ✅ | Location |
| `period_year` | INT | ✅ | Year of the quarter |
| `period_quarter` | INT | ✅ | Quarter (1–4) |
| `national_share` | DECIMAL | ✅ | e.g. `0.23` = 23% of national volume |

---

#### 📋 `lanes` *(static — update when logistics routes change)*

Used for lead-time lookups in replenishment and transfer planning.

| Field | Type | Required? | Notes |
|---|---|---|---|
| `from_location` | VARCHAR | ✅ | Source (e.g. `SUPPLIER`, `DC_MUMBAI`) |
| `to_location` | VARCHAR | ✅ | Destination |
| `mode` | VARCHAR | ✅ | `replenish` (supplier→DC) or `transfer` (DC→DC) |
| `lead_time_days` | INT | ✅ | Transit time in days |

---

#### 📋 `promo_calendar` *(update when promotions are planned/cancelled)*

| Field | Type | Required? | Notes |
|---|---|---|---|
| `promo_id` | VARCHAR | ✅ | Promo identifier |
| `name` | VARCHAR | ✅ | Promo name |
| `start_date` | DATE | ✅ | Promo start |
| `end_date` | DATE | ✅ | Promo end |
| `planned_uplift_pct` | DECIMAL | ✅ | Demand uplift factor (e.g. `0.15` = +15%) |
| `regions` | VARCHAR | ✅ | Comma-separated location IDs, or `ALL` |
| `status` | VARCHAR | ✅ | Rows with `cancelled` are excluded |

---

#### 📋 `distributors` *(static — update when distributor terms change)*

Used to model order noise and cycle behavior in simulation.

| Field | Type | Required? | Notes |
|---|---|---|---|
| `distributor_id` | VARCHAR | ✅ | e.g. `DST_DC_MUMBAI` |
| `region` | VARCHAR | ✅ | Served location |
| `order_cycle_days` | INT | ✅ | Typical order cycle |
| `order_size_sigma` | DECIMAL | ✅ | Demand noise std dev |

---

**Output of Step 1:** `forecasts_final` table + parquet. This is the input to all downstream steps.

---

### Step 2 — `replenishment.py` (Order Recommendations)

Reads the output of Step 1 plus:

#### 📋 `inventory_batches` *(must be updated daily — current stock snapshot)*

This is the most operationally sensitive table. Every day's batch snapshot must be loaded.

| Field | Type | Required? | Notes |
|---|---|---|---|
| `batch_id` | VARCHAR | ✅ | Unique batch identifier |
| `sku_id` | VARCHAR | ✅ | Product |
| `location` | VARCHAR | ✅ | Where the batch is physically stored |
| `qty_units` | INT | ✅ | Current quantity on hand |
| `expiry_date` | DATE | ✅ | Batch expiry date |
| `received_date` | DATE | ✅ | Date received |
| `unit_cost_inr` | INT | ✅ | Cost per unit |
| `status` | VARCHAR | ✅ | `healthy` / `watch` / `near_expiry_risk` / `stockout` |
| `as_of_date` | DATE | ✅ | Snapshot date — pipeline loads only `MAX(as_of_date)` |

Also reads (same as Step 1): `sku_master`, `lanes`, and `forecasts_final` (from Step 1 output).

**Output of Step 2:** `replenishment_orders` table.

---

### Step 3 — `allocation.py` (Transfer Plan + Write-off Risk)

Reads the outputs of Steps 1 and 2 plus:

- `inventory_batches` — to find near-expiry batches available to move
- `lanes` — to find valid transfer routes between locations
- `forecasts_final` (Step 1 output) — to compute destination demand rate
- `replenishment_orders` (Step 2 output) — to identify shortage locations as transfer targets

> **No new external table is required here.** All inputs are already loaded in Steps 1–2.

**Outputs of Step 3:** `transfer_plan` + `writeoff_risk` tables.

---

### Step 4 — `simulate.py` (90-Day Policy Simulation)

Reads:

- `demand_history` — actual sales for the simulation horizon (if available); falls back to `forecasts_final` p50 for future dates
- `inventory_batches` — starting inventory state
- `sku_master` — criticality and cost
- `lanes` — replenishment lead times
- `forecasts_final` (Step 1 output) — for the proposed policy's demand signal
- `transfer_plan` (Step 3 output) — to apply transfers in the proposed simulation

> **No new external table is required here** beyond what was already needed.

**Outputs of Step 4:** `simulation_daily` + `kpi_summary` tables.

---

### Step 5 — `alerts.py` (Alerts + AI Digest)

Reads the outputs of all previous steps:

- `replenishment_orders` (Step 2) — to find critical shortage SKUs
- `writeoff_risk` (Step 3) — to find near-expiry at-risk batches
- `forecasts_final` (Step 1) — to detect regional demand surges via `sense_adjustment`
- `kpi_summary` (Step 4) — to include KPI context in the AI digest

> **No new external table is required here.**

**Outputs of Step 5:** `alerts` + `alert_digest` tables.

---

## 🗓️ Summary: What to Update Before Each Rollover

| Table | Update Frequency | Who Provides It |
|---|---|---|
| `demand_history` | **Daily** — append new rows | ERP / POS system |
| `inventory_batches` | **Daily** — insert new `as_of_date` snapshot | WMS / ERP |
| `disease_burden_index` | **Weekly** (or when data arrives) | Surveillance / WHO feed |
| `promo_calendar` | **As needed** — add/cancel promos | Marketing team |
| `sku_master` | **Rarely** — on product add/retire | Product master data team |
| `locations` | **Rarely** — on DC/warehouse changes | Logistics team |
| `lanes` | **Rarely** — on route changes | Logistics team |
| `distributors` | **Rarely** — on contract changes | Procurement team |
| `location_demand_summary` | **Quarterly** | Analytics / BI team |

---

## 🏋️ Monthly Retraining — What It Needs

Monthly retraining runs `train_lgbm.py` and `torch_models.py`. These need **all the same tables as Step 1** (they call `load_tables()` and `build_panel()` exactly like the rollover does), plus:

- A **longer, complete `demand_history`** — the more history, the better. At minimum 13 months; ideally 3+ years.
- `disease_burden_index` — flu covariates for the entire historical period.
- `promo_calendar` — historical promos for the entire training window.

#### Extra artifacts produced only during retraining:

| Artifact (on disk) | Produced by | Purpose |
|---|---|---|
| `models/lgbm_global.txt` | `train_lgbm.py` | Saved LightGBM model weights |
| `backtest_lgbm.csv` | `train_lgbm.py` | Backtest WMAPE across 4 historical origins |
| `lgbm_feature_importance.csv` | `train_lgbm.py` | Feature gain scores |
| `forecasts_torch.parquet` | `torch_models.py` | Raw TFT / NHITS / Chronos predictions |
| `backtest_torch.csv` | `torch_models.py` | WMAPE scores per NN model |
| `forecasts_torch.meta.json` | `torch_models.py` | Cache freshness fingerprint |

After retraining, you **must run `ensemble.py`** (full re-ensemble) to regenerate `forecasts_final` using the newly trained models and updated ensemble weights. The daily rollover script (`rolling_forecast.py`) will then automatically pick up the latest weights from `ensemble_meta.yaml`.

---

## 🔁 Complete Pipeline Flow (Rollover Day)

```
[MySQL INPUT tables — updated by ops]
        │
        ▼
rolling_forecast.py   ──►  forecasts_final  (MySQL OUTPUT)
        │
        ▼
replenishment.py      ──►  replenishment_orders  (MySQL OUTPUT)
        │
        ▼
allocation.py         ──►  transfer_plan + writeoff_risk  (MySQL OUTPUT)
        │
        ▼
simulate.py           ──►  simulation_daily + kpi_summary  (MySQL OUTPUT)
        │
        ▼
alerts.py             ──►  alerts + alert_digest  (MySQL OUTPUT)
```

```
[Monthly — same INPUT tables + full history]
        │
        ▼
train_lgbm.py         ──►  lgbm_global.txt + backtest_lgbm.csv  (disk)
torch_models.py       ──►  forecasts_torch.parquet + backtest_torch.csv  (disk)
ensemble.py           ──►  forecasts_final  (MySQL OUTPUT) + ensemble_meta.yaml
```

# MedCare Pharma — Demand Sensing & Replenishment Control Tower

End-to-end demand planning solution for a pharma distributor facing **flu-season surges (+60%)**,
**Tier-2 stock-outs of critical SKUs**, and **expiry-driven write-offs stuck in metro DCs**.

Built for the _Demand Sensing & Replenishment_ (P1) hackathon problem statement.

---

## Headline results (42-day forward simulation vs real held-out demand)

| KPI                                 | Status Quo | Proposed  | Improvement        |
| ----------------------------------- | ---------- | --------- | ------------------ |
| Fill rate                           | 81.6%      | **91.8%** | +10.2 pts          |
| Critical SKU fill rate              | 75.5%      | **90.4%** | +14.9 pts          |
| Critical stock-out site-days        | 234        | **87**    | **−63%**           |
| Expiry write-offs                   | ₹21.6L     | **₹7.7L** | **−₹13.9L (−64%)** |
| Avg ending inventory (excess proxy) | 3062       | 2477      | −19%               |

## Forecast leaderboard — same-origin backtest (192 SKU×region series, 42-day horizon)

| Model                                   | WMAPE     | Type                       |
| --------------------------------------- | --------- | -------------------------- |
| **Ensemble + sensing overlay**          | **0.376** | weighted blend             |
| LightGBM (global, direct multi-horizon) | 0.3935    | tabular GBM                |
| Chronos-Bolt-base                       | 0.3989    | zero-shot foundation model |
| TFT (trained, covariates)               | 0.4091    | PyTorch / neuralforecast   |
| N-HiTS                                  | 0.4303    | PyTorch / neuralforecast   |
| Seasonal-Naive-364                      | ~0.55     | baseline                   |

Scores shift a point or two between training runs (GPU/CPU nondeterminism); the ensemble has topped every run.

Top LightGBM feature by gain: `futr_flu_index` — the leading indicator works.

## Architecture

```
Kaggle pharma-sales-data (real, 2014–19)
        │ build_dataset.py
        ▼
Synthetic network layer: 32 SKUs × 6 locations (2 metro DCs + 4 Tier-2),
batch/expiry inventory with injected pathology (metro near-expiry excess,
Tier-2 critical shortages), lanes/lead-times, promo calendar, flu index
        │
        ├── features.py ─► train_lgbm.py ─► backtest leaderboard
        ├── torch_models.py (N-HiTS/TFT trained on GPU w/ CPU fallback,
        │                    Chronos-Bolt zero-shot)
        ├── ensemble.py ─► inverse-WMAPE blend + damped sensing overlay
        │                  (momentum + flu-index leading signal, capped ±20%)
        ├── replenishment.py ─► order-up-to = μ(L+R) + z·σ·√L,
        │                       σ from ensemble P90–P10 band,
        │                       service level 99%/95%/90% by criticality
        ├── allocation.py ─► FEFO + expiry-aware transfers (rescue near-expiry
        │                    stock to regions that can consume it in time)
        │                    + shortage-rescue transfers metro→Tier-2
        ├── simulate.py ─► daily discrete-event sim of both policies
        └── alerts.py ─► RED/AMBER rules + surge-mode cadence switch
                         + Groq cloud LLM (openai/gpt-oss-120b) escalation narrative
```

## Key design decisions

- **Hybrid data**: real Kaggle daily sales give genuine seasonality; deterministic seeded
  expansion builds the network layer the problem demands (batches, expiry, lead times).
- **Probabilistic forecasting feeds inventory math**: safety stock uses the ensemble's
  P10–P90 band instead of fixed historical σ.
- **Expiry-aware allocation**: transfer only where destination consumption-before-expiry
  supports it; residual risk routed to markdown/donate recommendations instead of blind moves.
- **Explainable decisions**: replenishment/allocation are transparent formulas & rules;
  ML is confined to prediction. Judges can audit every number.
- **Escalation workflow**: weekly S&OP → daily surge mode when sensed uplift > 20% or ≥5 red alerts.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # then edit credentials
./run_all.sh                  # full pipeline from scratch (~15 min incl. training)
```

## ML Model Operations

The ML pipeline is a **PostgreSQL-backed** service that produces daily demand forecasts, replenishment plans, transfer allocations, and simulation KPIs. It has three operational modes: **monthly retrain** (full model refresh), **daily rollover** (recompute all outputs from latest data), and **live rollover** (fast forecast-only update). All modes are exposed via a **FastAPI sidecar** (port 8000) that the Express backend calls.

### Hosting

The ML service runs as a standalone Python process. There are no Docker images — you run it directly on the host or VM.

**Requirements:**

- Python 3.12+
- PostgreSQL 14+ (same instance the Express backend uses)
- ~4 GB RAM (LightGBM + Chronos inference; neural models optional on CPU)
- GPU optional (N-HiTS/TFT train faster on CUDA but auto-fallback to CPU)

**Start the FastAPI sidecar:**

```bash
cd medcare_ml_model
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The sidecar is **stateless** — it spawns pipeline steps as subprocesses and tracks run status in memory. If the process restarts, in-flight runs are lost but completed results persist in PostgreSQL.

**Environment variables** (set in `.env`, loaded automatically):

| Variable         | Required | Default                                                   | Description                                 |
| ---------------- | -------- | --------------------------------------------------------- | ------------------------------------------- |
| `PHARMA_DB_URL`  | Yes      | `postgresql://postgres:<password>@localhost:5432/medcare` | PostgreSQL connection string                |
| `GROQ_API_KEY`   | No       | (template fallback)                                       | Groq API key for AI-generated escalation digests |

### FastAPI server — endpoints

| Method | Path                   | Description                                            |
| ------ | ---------------------- | ------------------------------------------------------ |
| `GET`  | `/health`              | Health probe → `{"status": "ok"}`                      |
| `POST` | `/run/daily`           | Trigger daily rollover (returns `run_id` immediately)  |
| `POST` | `/run/retrain`         | Trigger monthly retrain (returns `run_id` immediately) |
| `GET`  | `/run/{run_id}/status` | Poll run status: `running` / `completed` / `failed`    |
| `GET`  | `/runs`                | List last 50 tracked runs                              |

**Example — trigger and poll a daily rollover:**

```bash
# trigger
RUN_ID=$(curl -s -X POST http://localhost:8000/run/daily | jq -r '.run_id')
echo "run_id: $RUN_ID"

# poll until done
while true; do
  STATUS=$(curl -s http://localhost:8000/run/$RUN_ID/status | jq -r '.status')
  echo "status: $STATUS"
  [ "$STATUS" != "running" ] && break
  sleep 3
done
```

**Mutual exclusion:** only one run can execute at a time. If a run is in progress, the second trigger returns `409 Conflict`.

### Pipeline flows

#### Flow 1 — Daily rollover forecast (full chain)

**What it does:** Regenerates every output table from the latest demand data. This is the "nightly batch" that keeps the dashboard current.

**Triggered by:**

- `POST /run/daily` from the Express backend (production)
- `./scripts/daily_roll.sh` from cron (demo/standalone)
- `.venv/bin/python src/rolling_forecast.py --full-chain --triggered-by manual` (CLI)

**Flow diagram:**

```
Backend / Cron / CLI
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  rolling_forecast.py --full-chain                       │
│                                                         │
│  1. Load tables from PostgreSQL                         │
│  2. LightGBM refit on data up to as_of (fast, ~10s)     │
│  3. Chronos-Bolt zero-shot inference (~5s)              │
│  4. Blend p10/p50/p90 using stored ensemble weights     │
│  5. Apply sensing overlay (momentum + flu index)        │
│  6. Write forecasts_final + rolling_run_log to DB       │
│                                                         │
│  If --full-chain:                                       │
│  7. replenishment.py → order-up-to safety stock calc    │
│  8. allocation.py   → FEFO + expiry-aware transfers     │
│  9. simulate.py     → 42-day discrete-event sim         │
│  10. alerts.py      → RED/AMBER rules + AI digest       │
└─────────────────────────────────────────────────────────┘
        │
        ▼
  PostgreSQL [OUTPUT] tables updated:
  forecasts_final, replenishment_orders, transfer_plan,
  writeoff_risk, simulation_daily, kpi_summary, alerts,
  alert_digest, rolling_run_log
```

**Timing:** ~30 seconds (LightGBM refit + Chronos inference + downstream chain).

**Idempotent:** rerunning for the same `as_of_date` overwrites cleanly (DELETE + INSERT).

**What feeds into it:** the Express backend inserts new daily rows into `demand_history`, `disease_burden_index`, and `inventory_batches`. The ML pipeline reads these tables — no file drops needed.

#### Flow 2 — Monthly retraining (full model refresh)

**What it does:** Retrains all models from scratch, recomputes ensemble weights, then runs the full output chain. This is the "monthly model refresh" that incorporates accumulated new data.

**Triggered by:**

- `POST /run/retrain` from the Express backend (production)
- `./scripts/monthly_retrain.sh` from cron (demo/standalone)
- `.venv/bin/python src/retrain.py` (CLI)

**Flow diagram:**

```
Backend / Cron / CLI
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  retrain.py                                             │
│                                                         │
│  1. Set as_of_date to MAX(date) in demand_history       │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Step 1: train_lgbm.py                             │  │
│  │   - Walk-forward backtest on full demand_history  │  │
│  │   - Train global LightGBM booster (19 features)   │  │
│  │   - Save model → models/lgbm_global.txt           │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Step 2: torch_models.py                           │  │
│  │   - N-HiTS: train on GPU/CPU (~3-5 min)           │  │
│  │   - TFT: train on GPU/CPU (~5-10 min)             │  │
│  │   - Chronos-Bolt: zero-shot (no training)         │  │
│  │   - Backtest all three, save scores               │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Step 3: ensemble.py                               │  │
│  │   - Recompute weights via inverse-WMAPE           │  │
│  │   - Save weights → models/ensemble_meta.yaml      │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Steps 4-7: full output chain                      │  │
│  │   replenishment → allocation → simulate → alerts  │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  8. fill_derived.py → refresh market_share, demand_sum  │
│  9. Write retrain_log.json to models/                   │
└─────────────────────────────────────────────────────────┘
        │
        ▼
  All models + ensemble weights refreshed.
  All [OUTPUT] tables recomputed.
```

**Timing:** ~3-5 minutes (GPU) or ~10-15 minutes (CPU-only).

**Variants:**

```bash
.venv/bin/python src/retrain.py --skip-torch    # skip neural models (~2 min, CPU-only)
.venv/bin/python src/retrain.py --rebuild-data  # regenerate synthetic dataset from raw CSV first
```

**What feeds into it:** reads ALL rows from `demand_history` (not just recent). A month of backend-fed rows is picked up automatically.

#### Flow 3 — Live rollover forecast (fast, forecast-only)

**What it does:** Produces a fresh 42-day forecast window from the latest demand data WITHOUT running the full output chain. This is the "on-demand forecast" that the backend can trigger when a user clicks "Refresh Forecast" or when new data arrives mid-day.

**Triggered by:**

- `.venv/bin/python src/rolling_forecast.py` (CLI, no `--full-chain`)
- In production: the backend calls `POST /run/daily` which runs the full chain (forecast is always included)

**Flow diagram:**

```
CLI (or backend via /run/daily)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  rolling_forecast.py  (without --full-chain)            │
│                                                         │
│  1. Load tables from PostgreSQL                         │
│  2. LightGBM refit on data up to as_of (~10s)           │
│  3. Chronos-Bolt zero-shot inference (~5s)              │
│  4. Blend p10/p50/p90 using stored ensemble weights     │
│  5. Apply sensing overlay (momentum + flu index)        │
│  6. Write forecasts_final + rolling_run_log to DB       │
│  7. Print realized WMAPE (backtest check)               │
└─────────────────────────────────────────────────────────┘
        │
        ▼
  Only forecasts_final + rolling_run_log updated.
  Replenishment/allocation/simulate NOT recomputed.
  (Run with --full-chain to include them.)
```

**Timing:** ~15 seconds (forecast only, no downstream chain).

**When to use:** mid-day refresh when new sales data arrives, or when a user wants to see updated forecasts without waiting for the full chain.

### One-time database setup

```bash
cp .env.example .env                          # edit PHARMA_DB_URL
.venv/bin/python db/db_writer.py --mode seed  # load historical data into PostgreSQL
.venv/bin/python db/fill_derived.py --full    # compute derived analytics tables
```

The schema is defined in `db/schema_postgres.sql` (22 tables). Run it against your PostgreSQL instance:

```bash
psql "$PHARMA_DB_URL" -f db/schema_postgres.sql
```

### Derived-table refresh intervals

`db/fill_derived.py` is idempotent — safe at any cadence:

| Table                      | Cadence                                                          | Notes                         |
| -------------------------- | ---------------------------------------------------------------- | ----------------------------- |
| `sku_market_share_monthly` | daily (cheap full recompute)                                     | pruned to ingested date range |
| `location_demand_summary`  | daily                                                            | quarterly share + YoY         |
| `warehouse_capacity_log`   | weekly snapshot auto-inserts when ≥7 days of new inventory exist |                               |
| `sku_cost_history`         | seeded once; `--full` regenerates                                |                               |

### Demo day simulator

`db/simulate_ingest_day.py` reveals one more day from the pre-generated history
(watermarked seed leaves ~8 months unrevealed): appends sales, ILI readings,
and an FEFO-depleted inventory snapshot — standing in for the production
backend so the cron demonstrably advances day by day.

### Bring your own data

No retraining is needed to _present_ — the dashboard reads saved artifacts. To plan on **your own sales history**:

```bash
.venv/bin/python src/ingest.py path/to/sales.csv    # wide or long CSV, any common date format
.venv/bin/python src/retrain.py --skip-torch        # fast rebuild (~5 min)
.venv/bin/python src/retrain.py                     # full rebuild incl. neural models
```

- Accepts **wide** CSVs (date + one column per ATC category — Kaggle format) and **long** CSVs (`date, atc_code, units`); SKU-level ids are mapped to their ATC prefix automatically.
- Validation report flags missing categories, calendar gaps, negative values (<13 months of history degrades year-lag features).
- The original Kaggle dataset is backed up to `data/raw/salesdaily_original_backup.csv` on first install.
- **Staleness guard**: a fingerprint of the demand table is stored with every ensemble run; if neural forecasts don't match the current data, `ensemble.py` silently falls back to a renormalized LightGBM-only blend instead of serving stale predictions.

### Suggested crontab

```cron
# Daily rollover at 06:00 (full chain: forecast + replenishment + allocation + sim + alerts)
0 6 * * *  cd /path/to/medcare_ml_model && ./scripts/daily_roll.sh >> logs/daily.log 2>&1

# Monthly retrain on the 1st at 02:00 (full model refresh)
0 2 1 * *  cd /path/to/medcare_ml_model && ./scripts/monthly_retrain.sh >> logs/monthly.log 2>&1
```

### FastAPI sidecar as a systemd service (production)

```ini
# /etc/systemd/system/medcare-ml.service
[Unit]
Description=MedCare ML Pipeline Sidecar
After=network.target postgresql.service

[Service]
Type=simple
User=medcare
WorkingDirectory=/opt/medcare/medcare_ml_model
EnvironmentFile=/opt/medcare/medcare_ml_model/.env
ExecStart=/opt/medcare/medcare_ml_model/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now medcare-ml
```

## Repo layout

```
config.yaml            all knobs: seed, network, promos, lead times, service levels
data/raw               Kaggle CSVs (kagglehub)
data/processed         generated tables, forecasts, plans, KPIs, alerts
src/                   pipeline modules (see Architecture) + ingest.py / retrain.py
app/main.py            FastAPI sidecar (port 8000) — daily/monthly triggers + run polling
models/                saved LightGBM booster, ensemble weights, backtest CSVs
scripts/               cron wrappers (daily_roll.sh, monthly_retrain.sh)
db/                    PostgreSQL access layer, schema, seed scripts
docs/                  deep-dive explanations, backend spec, data requirements
```

## Dashboard pages

1. **Executive Summary** — KPI cards vs status quo, served-vs-demand curves, cumulative write-offs, model leaderboard
2. **Demand Sensing** — actual vs ensemble with 80% band, per-model overlays, sensing factors, feature importance
3. **Allocation & Transfers** — expiry-rescue + shortage-rescue plans, lane volumes, shelf-life heatmap
4. **Replenishment Plan** — filterable order book with safety-stock logic
5. **Escalation Center** — review mode banner, RED/AMBER alert board, AI escalation brief, policy table
6. **Data & Retrain** — upload your own sales CSV, validation report, one-click guided retrain

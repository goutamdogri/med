# MedCare Pharma — Demand Sensing & Replenishment Control Tower

End-to-end demand planning solution for a pharma distributor facing **flu-season surges (+60%)**,
**Tier-2 stock-outs of critical SKUs**, and **expiry-driven write-offs stuck in metro DCs**.

Built for the *Demand Sensing & Replenishment* (P1) hackathon problem statement.

---

## Headline results (42-day forward simulation vs real held-out demand)

| KPI | Status Quo | Proposed | Improvement |
|---|---|---|---|
| Fill rate | 81.6% | **91.8%** | +10.2 pts |
| Critical SKU fill rate | 75.5% | **90.4%** | +14.9 pts |
| Critical stock-out site-days | 234 | **87** | **−63%** |
| Expiry write-offs | ₹21.6L | **₹7.7L** | **−₹13.9L (−64%)** |
| Avg ending inventory (excess proxy) | 3062 | 2477 | −19% |

## Forecast leaderboard — same-origin backtest (192 SKU×region series, 42-day horizon)

| Model | WMAPE | Type |
|---|---|---|
| **Ensemble + sensing overlay** | **0.376** | weighted blend |
| LightGBM (global, direct multi-horizon) | 0.3935 | tabular GBM |
| Chronos-Bolt-base | 0.3989 | zero-shot foundation model |
| TFT (trained, covariates) | 0.4091 | PyTorch / neuralforecast |
| N-HiTS | 0.4303 | PyTorch / neuralforecast |
| Seasonal-Naive-364 | ~0.55 | baseline |

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
                         + local Gemma-4B (Ollama) escalation narrative
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

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./run_all.sh                      # full pipeline (~15 min incl. training)
.venv/bin/streamlit run app/streamlit_app.py
```

## MySQL-backed operations (production mode)

All pipeline inputs are read from the **`pharma_sc`** MySQL database and every
output is dual-written to parquet (dashboard) **and** dedicated `[OUTPUT]`
tables. The backend inserts new daily rows directly into
`demand_history`, `disease_burden_index`, `inventory_batches` — no file drops needed.

### One-time setup

```bash
.venv/bin/pip install pymysql
MYSQL_ROOT_PWD=<root-password> ./db/setup_db.sh          # creates schema + pharma_user
cp .env.example .env                                     # then edit credentials
.venv/bin/python db/db_writer.py --mode seed             # load history up to as-of watermark
.venv/bin/python db/fill_derived.py --full               # compute [DERIVED] tables from MySQL
```

Connection is configured via `PHARMA_DB_URL` (env var or `.env`, git-ignored).
If MySQL is unreachable the pipeline automatically falls back to `data/processed/` parquet files.

### Daily rollover forecast (cron)

```bash
./scripts/daily_roll.sh
# equivalent manual steps:
.venv/bin/python db/simulate_ingest_day.py               # demo only — backend does this in prod
.venv/bin/python src/rolling_forecast.py --full-chain --triggered-by cron
.venv/bin/python db/fill_derived.py
```

One command regenerates the day's forecast → replenishment → transfers →
simulation → alerts, writing to `forecasts_final`, `replenishment_orders`,
`transfer_plan`, `writeoff_risk`, `simulation_daily`, `kpi_summary`, `alerts`,
`alert_digest` and the `rolling_run_log` audit trail — all stamped with today's
`as_of_date`. Reruns for the same date overwrite cleanly (idempotent).

Suggested crontab:

```cron
0 6 * * *  cd /path/to/med && ./scripts/daily_roll.sh >> logs/daily.log 2>&1
0 2 1 * *  cd /path/to/med && ./scripts/monthly_retrain.sh >> logs/monthly.log 2>&1
```

### Monthly retraining

```bash
./scripts/monthly_retrain.sh            # trains on ALL demand_history in MySQL
# fast variant:  .venv/bin/python src/retrain.py --skip-torch
# demo reset:    .venv/bin/python src/retrain.py --rebuild-data   # regenerate synthetic data from raw CSV
```

The retrain sets the pipeline origin to the newest ingested day, retrains
LightGBM + neural models, recomputes ensemble weights, reruns the full output
chain, and refreshes the derived analytics tables (`sku_market_share_monthly`,
`location_demand_summary`, …). Every stage reads its training data straight
from MySQL, so a month of backend-fed rows is picked up with zero extra work.

### Derived-table refresh intervals

`db/fill_derived.py` is idempotent — safe at any cadence:

| Table | Cadence | Notes |
|---|---|---|
| `sku_market_share_monthly` | daily (cheap full recompute) | pruned to ingested date range |
| `location_demand_summary` | daily | quarterly share + YoY |
| `warehouse_capacity_log` | weekly snapshot auto-inserts when ≥7 days of new inventory exist | |
| `sku_cost_history` | seeded once; `--full` regenerates | |

### Demo day simulator

`db/simulate_ingest_day.py` reveals one more day from the pre-generated history
(watermarked seed leaves ~8 months unrevealed): appends sales, ILI readings,
and an FEFO-depleted inventory snapshot — standing in for the production
backend so the cron demonstrably advances day by day.

### Bring your own data

No retraining is needed to *present* — the dashboard reads saved artifacts. To plan on **your own sales history**, use the **📥 Data & Retrain** dashboard page (or CLI):

```bash
.venv/bin/python src/ingest.py path/to/sales.csv    # wide or long CSV, any common date format
.venv/bin/python src/retrain.py --skip-torch        # fast rebuild (~5 min)
.venv/bin/python src/retrain.py                     # full rebuild incl. neural models
```

- Accepts **wide** CSVs (date + one column per ATC category — Kaggle format) and **long** CSVs (`date, atc_code, units`); SKU-level ids are mapped to their ATC prefix automatically.
- Validation report flags missing categories, calendar gaps, negative values (<13 months of history degrades year-lag features).
- The original Kaggle dataset is backed up to `data/raw/salesdaily_original_backup.csv` on first install.
- **Staleness guard**: a fingerprint of the demand table is stored with every ensemble run; if neural forecasts don't match the current data, `ensemble.py` silently falls back to a renormalized LightGBM-only blend instead of serving stale predictions. The retrain orchestrator backs up `data/processed/` and rolls back automatically if any step fails.

### Daily rolling forecast (operations mode)

The forecast window is **42 days** ahead (`config.yaml → simulation.horizon_days`). To produce today's
forecast from the latest observed data (~2 min — LightGBM refit + Chronos-Bolt zero-shot inference,
no neural retraining):

```bash
.venv/bin/python src/rolling_forecast.py                     # origin = last date in demand history
.venv/bin/python src/rolling_forecast.py --date 2019-02-01   # explicit origin
```

- Blends LightGBM + Chronos using the stored ensemble weights, applies the same sensing overlay, and overwrites `forecasts_final.parquet` so every dashboard page reflects the live view; `config.yaml` as-of rolls forward automatically.
- Prints realized WMAPE when future actuals already exist (backtest sanity check).
- In live ops, first append yesterday's sales (`src/ingest.py updated.csv` or the dashboard upload), then roll.
- The full demo state is snapshotted to `data/processed/demo_snapshot/` on first roll; `--restore-demo` brings it back exactly (forecasts, plans, alerts, config as-of).

Optional: local Gemma via Ollama for the AI brief (`gemma4:e2b`, configurable in `src/alerts.py`);
falls back to templated digest automatically if Ollama is down.

GPU optional: N-HiTS/TFT auto-fall back to CPU (TFT trains on CPU in ~10 min here).

## Repo layout

```
config.yaml            all knobs: seed, network, promos, lead times, service levels
data/raw               Kaggle CSVs (kagglehub)
data/processed         generated tables, forecasts, plans, KPIs, alerts
src/                   pipeline modules (see Architecture) + ingest.py / retrain.py
app/streamlit_app.py   6-page control tower dashboard
models/                saved LightGBM booster
```

## Dashboard pages

1. **Executive Summary** — KPI cards vs status quo, served-vs-demand curves, cumulative write-offs, model leaderboard
2. **Demand Sensing** — actual vs ensemble with 80% band, per-model overlays, sensing factors, feature importance
3. **Allocation & Transfers** — expiry-rescue + shortage-rescue plans, lane volumes, shelf-life heatmap
4. **Replenishment Plan** — filterable order book with safety-stock logic
5. **Escalation Center** — review mode banner, RED/AMBER alert board, AI escalation brief, policy table
6. **Data & Retrain** — upload your own sales CSV, validation report, one-click guided retrain

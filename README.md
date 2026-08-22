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

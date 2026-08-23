#!/usr/bin/env bash
# Daily rollover — run via cron each morning (e.g. 06:00).
#   1. reveal one more simulated day into pharma_sc (demo stand-in for the real backend;
#      in production delete this line — the backend inserts demand/flu/inventory rows)
#   2. full-chain rollover from MySQL: forecasts → replenishment → transfers → simulation
#      → alerts, everything dual-written to parquet + pharma_sc [OUTPUT] tables
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export $(grep -v '^#' .env 2>/dev/null | xargs) 2>/dev/null || true

$PY db/simulate_ingest_day.py          # remove in production (backend feeds data instead)
$PY src/rolling_forecast.py --full-chain --triggered-by cron
$PY db/fill_derived.py                 # keep derived analytics current (idempotent)
echo "[daily_roll] done at $(date)"

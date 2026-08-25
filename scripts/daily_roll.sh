#!/usr/bin/env bash
# Daily rollover — run via cron each morning (e.g. 06:00).
#   1. promote simulated_today's data from staging → live tables
#   2. full-chain rollover: forecasts → replenishment → transfers → simulation → alerts
#   3. refresh derived analytics tables
#
# The backend advances pipeline_state.simulated_today before triggering this.
# This script does NOT advance the date — it reads whatever is in the DB.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export $(grep -v '^#' .env 2>/dev/null | xargs) 2>/dev/null || true

$PY db/simulate_ingest_day.py                          # promote staging → live for simulated_today
$PY src/rolling_forecast.py --full-chain --triggered-by cron
$PY db/fill_derived.py                                 # keep derived analytics current (idempotent)
echo "[daily_roll] done at $(date)"

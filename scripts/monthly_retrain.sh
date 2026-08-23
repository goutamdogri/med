#!/usr/bin/env bash
# Monthly retrain — run via cron on the 1st of each month (e.g. 02:00).
# Trains LightGBM + neural models on ALL demand_history accumulated in MySQL,
# recomputes ensemble weights, and refreshes derived analytics tables.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export $(grep -v '^#' .env 2>/dev/null | xargs) 2>/dev/null || true

$PY src/retrain.py                     # add --skip-torch for a fast CPU-only refresh
echo "[monthly_retrain] done at $(date)"

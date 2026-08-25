#!/usr/bin/env bash
set -e
PY=${PY:-.venv/bin/python}

echo "== 1/8 data =="
$PY src/build_dataset.py
echo "== 2/8 lightgbm backtest =="
$PY src/train_lgbm.py
echo "== 3/8 neural models =="
$PY src/torch_models.py
echo "== 4/8 ensemble + sensing =="
$PY src/ensemble.py
echo "== 5/8 replenishment =="
$PY src/replenishment.py
echo "== 6/8 allocation =="
$PY src/allocation.py
echo "== 7/7 simulation + alerts =="
$PY src/simulate.py
$PY src/alerts.py
echo "== 8/8 metrics =="
$PY src/metrics.py
echo "DONE. ML sidecar runs at http://localhost:8000 (uvicorn app.main:app)"

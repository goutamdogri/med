#!/usr/bin/env bash
set -e
PY=${PY:-.venv/bin/python}

echo "== 1/7 data =="
$PY src/build_dataset.py
echo "== 2/7 lightgbm backtest =="
$PY src/train_lgbm.py
echo "== 3/7 neural models =="
$PY src/torch_models.py
echo "== 4/7 ensemble + sensing =="
$PY src/ensemble.py
echo "== 5/7 replenishment =="
$PY src/replenishment.py
echo "== 6/7 allocation =="
$PY src/allocation.py
echo "== 7/7 simulation + alerts =="
$PY src/simulate.py
$PY src/alerts.py
echo "DONE. Launch: .venv/bin/streamlit run app/streamlit_app.py"

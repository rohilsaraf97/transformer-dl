#!/usr/bin/env bash
# Train all 9 (heads x layers) configurations, then build plots and the summary table.
# Extra arguments are forwarded to train.py, e.g.: ./run_all.sh --epochs 10
set -euo pipefail
PY=${PYTHON:-python}
mkdir -p logs

for heads in 2 4 8; do
  for layers in 2 4 8; do
    $PY train.py --nhead "$heads" --num_layers "$layers" "$@" 2>&1 | tee "logs/h${heads}_l${layers}.log"
  done
done

$PY make_report.py

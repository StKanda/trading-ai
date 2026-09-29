#!/usr/bin/env bash
set -euo pipefail
./trading-ai pipeline --symbol NVDA --source local --data data/alpaca/NVDA/split --timeframe 5 --dl-objective economic --dl-models MLP_12 TCN_12 --seeds 42 123 777 --epochs 30 --patience 5 --train-sessions 504 --fold-sessions 63 --out runs/NVDA_V12_01

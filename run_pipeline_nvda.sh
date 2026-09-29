#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
STAMP="$(date +%Y%m%d_%H%M%S)"
./trading-ai pipeline \
  --symbol NVDA \
  --source yahoo \
  --timeframe 5 \
  --dl-models MLP_12 MLP_24 TCN_12 TCN_24 \
  --seeds 42 123 777 \
  --epochs 20 \
  --patience 4 \
  --out "runs/NVDA_${STAMP}"

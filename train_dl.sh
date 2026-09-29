#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 1 ]; then echo "Usage: $0 DATA_DIR [CONTEXT]" >&2; exit 2; fi
if ! python -c 'import torch' >/dev/null 2>&1; then
  echo "PyTorch absent. Installe la couche Deep Learning avec:" >&2
  echo "  python -m pip install -r requirements_dl.txt" >&2
  exit 1
fi
exec python main.py train-dl --data "$1" --symbol SPY --context "${2:-24}" --out runtime/dl_smoke

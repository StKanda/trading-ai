#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 1 ]; then echo "Usage: $0 DATA_DIR [SYMBOL] [CHAMPION]" >&2; exit 2; fi
DATA="$1"; SYMBOL="${2:-SPY}"; CHAMPION="${3:-RULE_MOMENTUM}"
exec python worker.py --data "$DATA" --symbol "$SYMBOL" --champion "$CHAMPION"

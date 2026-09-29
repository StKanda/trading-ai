#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 2 ]; then echo "Usage: $0 DATA_DIR YYYY-MM-DD [CHAMPION]" >&2; exit 2; fi
DATA="$1"; SESSION="$2"; CHAMPION="${3:-RULE_MOMENTUM}"
exec python replay.py --data "$DATA" --session "$SESSION" --champion "$CHAMPION" --out "runtime/replay_${SESSION}.json"

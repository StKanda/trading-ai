#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 DATA_DIR TEST_START TEST_END_EXCLUSIVE" >&2
  exit 2
fi
DATA="$1"; START="$2"; END="$3"
python main.py audit-v10 --data "$DATA" --symbol SPY --out runtime/data_audit.json
./run_tests.sh
rm -rf results_v10_research results_v10_final
python main.py research --data "$DATA" --symbol SPY --timeframe 5 --test-start "$START" --test-end "$END" --out results_v10_research
python main.py final --data "$DATA" --lock results_v10_research/protocol_lock.json --out results_v10_final

#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 /chemin/snapshot YYYY-MM-DD_TEST_START YYYY-MM-DD_TEST_END_EXCLUSIF" >&2
  exit 2
fi
DATA="$1"; START="$2"; END="$3"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESEARCH="results_v9_research_${STAMP}"
FINAL="results_v9_final_${STAMP}"
python audit_data.py --data "$DATA" --symbol SPY --out "audit_${STAMP}.json"
python -m unittest discover -s tests -v
python main.py research --data "$DATA" --symbol SPY --timeframe 5 --test-start "$START" --test-end "$END" --out "$RESEARCH"
echo "Recherche terminée. Verrou: $RESEARCH/protocol_lock.json"
python main.py final --data "$DATA" --lock "$RESEARCH/protocol_lock.json" --out "$FINAL"
echo "Test final terminé: $FINAL"

#!/usr/bin/env bash
set -euo pipefail
if ! python -c 'import streamlit' >/dev/null 2>&1; then
  echo 'Streamlit absent. Installer: python -m pip install -r requirements.txt' >&2
  exit 1
fi
if [ "$#" -ge 1 ]; then
  export TRADING_AI_DATA_DIR="$1"
fi
exec python -m streamlit run app.py --server.showEmailPrompt=false --browser.gatherUsageStats=false

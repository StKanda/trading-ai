#!/usr/bin/env bash
set -eu
cd -- "$(dirname -- "$0")"
if [ -x ../.venv/bin/python ]; then
  exec ../.venv/bin/python main.py serve --demo
elif [ -n "${VIRTUAL_ENV:-}" ]; then
  exec python main.py serve --demo
else
  exec python3 main.py serve --demo
fi

#!/usr/bin/env bash
set -euo pipefail
./trading-ai alpaca-download --symbols NVDA SPY --start 2016-01-01 --feed sip --adjustments raw split --out data/alpaca

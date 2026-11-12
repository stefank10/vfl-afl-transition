#!/usr/bin/env bash
# Full rebuild: raw pull (R) -> tidy -> A (translate) -> C (readiness) -> charts + site data
set -euo pipefail
cd "$(dirname "$0")"
[ -f data/raw/pull_log.csv ] || Rscript R/pull_data.R
python3 -m src.load
python3 -m src.translate
python3 -m src.readiness
python3 -m src.clearances
python3 -m src.figures

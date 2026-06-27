#!/usr/bin/env bash
# Reproduce the entire ThermoRoute study end to end.
# Usage:  bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export OMP_NUM_THREADS=8

echo "[1/6] data preparation + audit"
python3 scripts/01_prepare_data.py
echo "[2/6] unit tests (leakage / metrics)"
python3 -m pytest tests/ -q
echo "[3/6] full experiment matrix (this is the long step)"
python3 scripts/04_run_experiments.py
echo "[4/6] mechanism analysis (trains one model, extracts router/kappa)"
python3 scripts/05_explain.py
echo "[5/6] figures"
python3 scripts/06_make_figures.py
echo "[6/6] tables"
python3 scripts/07_make_tables.py
echo "DONE — see outputs/{figures,tables,reports}."

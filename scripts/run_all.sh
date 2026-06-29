#!/usr/bin/env bash
# Reproduce the entire ThermoRoute study end to end:
# (a) 3-station case study (b1/s2/p3, ~30 min on CPU);
# (b) USGS large-sample main analysis (40/120 stations, multi-hour on CPU).
# The USGS panel must already be acquired in data_usgs/ (see step [5]); the
# acquisition step is network-bound and shipped as pre-acquired panels.
#
# Usage:
#   bash scripts/run_all.sh                       # full pipeline (both tracks)
#   USGS_PANEL=data_usgs/panel_usgs.parquet \
#       bash scripts/run_all.sh                   # use a different USGS panel
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export OMP_NUM_THREADS=8
export KMP_DUPLICATE_LIB_OK=TRUE
USGS_PANEL="${USGS_PANEL:-data_usgs/panel_usgs_100.parquet}"

echo "================ TRACK A: 3-station case study ================"
echo "[1/11] data preparation + audit (3-station)"
python3 scripts/01_prepare_data.py
echo "[2/11] unit tests (leakage / metrics)"
python3 -m pytest tests/ -q
echo "[3/11] 3-station experiment matrix"
python3 scripts/04_run_experiments.py
echo "[4/11] 3-station mechanism analysis"
python3 scripts/05_explain.py
echo "[5/11] 3-station figures"
python3 scripts/06_make_figures.py
echo "[6/11] 3-station tables"
python3 scripts/07_make_tables.py
echo "[7/11] decision-value (REV) analysis"
python3 scripts/08_decision_value.py

echo ""
echo "================ TRACK B: USGS large-sample main analysis ================"
echo "[8/11] USGS experiment (baselines + air2stream + ThermoRoute × seeds + LGO + ablations)"
echo "      using panel: ${USGS_PANEL}"
python3 scripts/09_usgs_experiment.py --panel "${USGS_PANEL}" --air2stream --seeds 5
echo "[9/11] USGS calibration, REV, mechanism (κ, router drivers)"
python3 scripts/10_usgs_analysis.py
echo "[10/11] per-station Wilcoxon + bootstrap CI (Claims 1, 3)"
python3 scripts/12_claim_stats.py
echo "[11/11] K-fold leave-group-out + 3-seed ablations (Claims 2, 4)"
python3 scripts/13_rigor.py

echo ""
echo "DONE — see outputs/{figures,tables,reports} and paper/ for the manuscript."

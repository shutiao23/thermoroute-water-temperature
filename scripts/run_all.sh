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
echo "(multi-hour on CPU: 5 ThermoRoute seeds + 4 region-transfer folds + 3 LSTM"
echo " seeds + 4 LSTM transfer folds are the heavy stages; every stage is"
echo " checkpointed, so an interrupted run resumes.)"
echo "[8/20] USGS experiment (baselines + air2stream + ThermoRoute × seeds + LGO + ablations)"
echo "      using panel: ${USGS_PANEL}"
# Output names are the *_v2 current-truth files that scripts 10/12/13 and the
# sample-consistency test read first — a rerun updates the paper's tables
# instead of writing to a retired generation's filename.
python3 scripts/09_usgs_experiment.py --panel "${USGS_PANEL}" --air2stream --seeds 5 \
    --out_predictions usgs_predictions_v2.parquet \
    --out_report usgs_experiment_v2.md \
    --out_scores usgs_scores_v2.csv
echo "[9/20] per-station LightGBM (M4 — the stronger-of-two learned-baseline foil)"
python3 scripts/_perstation_lgb.py
echo "[10/20] K-fold leave-group-out + 3-seed ablations (Claims 2, 4)"
python3 scripts/13_rigor.py
echo "[11/20] leave-HUC2-region-out transfer — 4 folds of ThermoRoute (go/no-go)"
for f in 0 1 2 3; do python3 scripts/13c_region_transfer.py --fold "$f"; done
echo "[12/20] region-transfer assemble: global LightGBM per fold + verdict + figure"
python3 scripts/13c_region_transfer.py --assemble
echo "[13/20] deep sequence baseline (global LSTM): in-sample × 3 seeds -> splice v2"
python3 scripts/16_lstm_baseline.py --insample
echo "[14/20] deep sequence baseline: leave-HUC2-region-out transfer (4 folds)"
python3 scripts/16_lstm_baseline.py --transfer
echo "[15/20] 3-way transfer + in-sample LSTM report"
python3 scripts/16_lstm_baseline.py --report
echo "[16/20] Proposition-1 bounded-degradation, verified empirically (Fig 3)"
python3 scripts/17_prop1_binding.py
echo "[17/20] full REV decision-value curve over cost-loss grid (Fig 5)"
python3 scripts/18_rev_curve.py
echo "[18/20] probabilistic (PICP/CRPS/reliability/Brier) + multi-metric (Fig 4)"
python3 scripts/19_probabilistic.py
echo "[19/22] TUURT transfer triad + stratified robustness"
python3 scripts/20_tuurt.py
python3 scripts/15_stratified.py
echo "[20/22] exceedance warnings at fixed EPA ecological thresholds (18/20 °C)"
python3 scripts/21_ecological_thresholds.py
echo "[21/22] adaptive conformal (ACI) + conditional coverage"
python3 scripts/22_adaptive_conformal.py
echo "[22/22] USGS calibration/REV/mechanism, claim stats, artifact manifest"
python3 scripts/10_usgs_analysis.py
python3 scripts/12_claim_stats.py
python3 scripts/14_manifest.py

echo ""
echo "DONE — see outputs/{figures,tables,reports} and paper/ for the manuscript."
echo "Rebuild the PDFs with: (cd paper && ../scripts/... ) — see README."

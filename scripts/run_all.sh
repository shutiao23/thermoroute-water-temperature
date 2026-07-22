#!/usr/bin/env bash
# Reproduce the entire ThermoRoute study end to end:
# (a) legacy three ordinary monitoring sites (b1/s2/p3, ~30 min on CPU);
# (b) USGS large-sample development analysis (120 stations, multi-hour on CPU).
# The USGS panel must already be acquired in data_usgs/ (see step [5]); the
# acquisition step is network-bound and shipped as pre-acquired panels.
#
# Usage:
#   bash scripts/run_all.sh                       # full canonical pipeline
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/processed outputs/{tables,figures,predictions,reports,models,logs}
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0
export WORKER_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
readonly CANONICAL_USGS_PANEL="data_usgs/panel_usgs_120v2.parquet"
if [[ -n "${USGS_PANEL:-}" && "$USGS_PANEL" != "$CANONICAL_USGS_PANEL" ]]; then
  echo "run_all freezes the canonical panel; custom USGS_PANEL is unsupported" >&2
  exit 2
fi
export USGS_PANEL="$CANONICAL_USGS_PANEL"
if [[ ! -f "$USGS_PANEL" ]]; then
  echo "USGS_PANEL does not exist: $USGS_PANEL" >&2
  exit 2
fi

echo "================ TRACK A: legacy ordinary monitoring sites ================"
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
echo "[7/11] descriptive cost-loss (REV) illustration"
python3 scripts/08_decision_value.py

echo ""
echo "================ TRACK B: USGS large-sample development analysis ================"
echo "(multi-hour on CPU: 5 ThermoRoute seeds + 4 region-transfer folds + 5 LSTM"
echo " seeds + 4 LSTM transfer folds are the heavy stages; trained stages are"
echo " checkpointed, so an interrupted run resumes.)"
echo "[8/27] USGS experiment (baselines + air2stream + ThermoRoute × seeds + LGO + ablations)"
echo "      using panel: ${USGS_PANEL}"
# Stage 9 is an immutable parent.  Its command returns successfully only after
# the report, three formal pointers and final content-bound completion receipt
# are durable.  Stage 24 rejects a missing or stale receipt and binds the
# accepted receipt into the frozen suite identity.
python3 scripts/09_usgs_experiment.py --panel "${USGS_PANEL}" --air2stream --seeds 5 \
    --device cpu \
    --out_predictions usgs_predictions_stage9_v2.parquet
echo "[9/27] matched-budget neural controls + exact 31-member feature ladder"
# Stage 09b publishes its content-bound receipt only after all 31 member
# predictions, their sidecars, the common-key audit, budget, combined
# predictions and report validate.  Stage 24 requires and revalidates it.
python3 scripts/09b_development_controls.py --panel "${USGS_PANEL}"
echo "[10/27] per-station LightGBM (M4 — the stronger-of-two learned-baseline foil)"
python3 scripts/_perstation_lgb.py --panel "${USGS_PANEL}"
echo "[11/27] exploratory development holdout + 5-seed ablations"
python3 scripts/13_rigor.py
echo "[12/27] exploratory leave-HUC2-region-out gauged transfer — 4 folds of ThermoRoute"
for f in 0 1 2 3; do python3 scripts/13c_region_transfer.py --fold "$f"; done
echo "[13/27] region-transfer assemble: global LightGBM per fold + descriptive figure"
python3 scripts/13c_region_transfer.py --assemble
echo "[14/27] deep sequence baseline (global LSTM): in-sample × 5 seeds -> derive final v2"
python3 scripts/16_lstm_baseline.py --insample
echo "[15/27] deep sequence baseline: leave-HUC2-region-out transfer (4 folds)"
python3 scripts/16_lstm_baseline.py --transfer
echo "[16/27] 3-way transfer + in-sample LSTM report"
python3 scripts/16_lstm_baseline.py --report
echo "[17/27] Algebraic diagnostic (Fig 3; no safety claim)"
python3 scripts/17_prop1_binding.py
echo "[18/27] descriptive REV curve over the cost-loss grid (Fig 5)"
python3 scripts/18_rev_curve.py
echo "[19/27] probabilistic (PICP/three-quantile score/reliability/Brier) + multi-metric (Fig 4)"
python3 scripts/19_probabilistic.py
echo "[20/27] legacy transfer diagnostics (not ungauged) + regime stratification"
python3 scripts/20_tuurt.py
python3 scripts/15_stratified.py
echo "[21/27] ecological-threshold eligibility audit / strict 7DADM when inputs exist"
python3 scripts/21_ecological_thresholds.py
echo "[22/27] adaptive conformal diagnostics (no conditional-coverage claim)"
python3 scripts/22_adaptive_conformal.py
echo "[23/27] predeclared input-stress/OOD robustness (frozen ensemble; common keys)"
python3 scripts/23_robustness.py --panel "${USGS_PANEL}"
echo "[24/27] USGS calibration/REV/mechanism and claim statistics"
python3 scripts/10_usgs_analysis.py
python3 scripts/12_claim_stats.py
echo "[25/27] station-agnostic pooled external suite (development data only)"
python3 scripts/25_train_external_pooled_suite.py
echo "[26/27] freeze the complete Route-A model suite"
python3 scripts/24_freeze_model_suite.py \
    --stage9-receipt outputs/models/route_a_stage09_completion.json \
    --stage09b-receipt outputs/models/route_a_stage09b_completion.json
echo "[27/27] isolated full-model replay and final artifact manifest"
if [[ -f outputs/model_replay/route_a_development_replay_v1.json ]]; then
  python3 -I -B scripts/27_verify_development_replay.py --check
else
  python3 -I -B scripts/27_verify_development_replay.py
fi
python3 scripts/14_manifest.py

echo ""
echo "DONE — development outputs are under outputs/; see outputs/README.md for authority."
echo "Route A still requires its separate freeze, chronology, authorization, and opening chain."
echo "Rebuild the PDFs with: (cd paper && ../scripts/... ) — see README."

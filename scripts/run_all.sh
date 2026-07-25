#!/usr/bin/env bash
# Reproduce the canonical Route-A USGS development analysis end to end.
# The older b1/s2/p3 ordinary-monitoring-site case is not Route-A evidence and
# runs only when explicitly requested with --include-legacy-monitoring-case.
# The USGS panel must already be acquired in data_usgs/ (see step [5]); the
# acquisition step is network-bound and shipped as pre-acquired panels.
#
# Usage:
#   bash scripts/run_all.sh                       # canonical Route-A pipeline
#   bash scripts/run_all.sh --include-legacy-monitoring-case
set -euo pipefail
cd "$(dirname "$0")/.."

# The later model-freeze chronology deliberately rejects compiled bytecode in
# protected source/control directories. Keep the canonical pipeline from
# creating those untracked files during its own pre-flight tests or stages.
unset PYTHONPYCACHEPREFIX
export PYTHONDONTWRITEBYTECODE=1
# Every formal child starts with an interpreter-selected hash secret.  Identity
# hashes remain reproducible because the code canonicalises unordered values;
# fixing PYTHONHASHSEED before interpreter start would disable the Stage-24
# anti-order-dependence check rather than improve the evidence contract.
unset PYTHONHASHSEED

readonly THERMOROUTE_PYTHON="${THERMOROUTE_PYTHON:-python}"
if ! command -v "$THERMOROUTE_PYTHON" >/dev/null 2>&1; then
  echo "run_all cannot find THERMOROUTE_PYTHON=$THERMOROUTE_PYTHON" >&2
  exit 2
fi
"$THERMOROUTE_PYTHON" -c \
  'import sys; v=sys.version_info[:2]; sys.exit(f"Route A requires Python 3.12, got {v[0]}.{v[1]}") if v != (3, 12) else None'

INCLUDE_LEGACY_MONITORING_CASE=0
if (( $# > 1 )); then
  echo "usage: bash scripts/run_all.sh [--include-legacy-monitoring-case]" >&2
  exit 2
fi
if (( $# == 1 )); then
  if [[ "$1" != "--include-legacy-monitoring-case" ]]; then
    echo "usage: bash scripts/run_all.sh [--include-legacy-monitoring-case]" >&2
    exit 2
  fi
  INCLUDE_LEGACY_MONITORING_CASE=1
fi

mkdir -p data/processed outputs/{tables,figures,predictions,reports,models,logs}
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
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

echo "================ PRE-FLIGHT TESTS ================"
"$THERMOROUTE_PYTHON" -m pytest tests/ -q

if (( INCLUDE_LEGACY_MONITORING_CASE == 1 )); then
  echo "================ OPTIONAL LEGACY MONITORING-SITE CASE ================"
  echo "[legacy 1/6] data preparation + audit"
  "$THERMOROUTE_PYTHON" scripts/01_prepare_data.py
  echo "[legacy 2/6] experiment matrix"
  "$THERMOROUTE_PYTHON" scripts/04_run_experiments.py
  echo "[legacy 3/6] mechanism analysis"
  "$THERMOROUTE_PYTHON" scripts/05_explain.py
  echo "[legacy 4/6] figures"
  "$THERMOROUTE_PYTHON" scripts/06_make_figures.py
  echo "[legacy 5/6] tables"
  "$THERMOROUTE_PYTHON" scripts/07_make_tables.py
  echo "[legacy 6/6] descriptive cost-loss illustration"
  "$THERMOROUTE_PYTHON" scripts/08_decision_value.py
else
  echo "Legacy b1/s2/p3 monitoring-site case skipped (explicit opt-in only)."
fi

echo ""
echo "================ ROUTE A: USGS DEVELOPMENT ANALYSIS ================"
echo "(multi-hour on CPU: 5 ThermoRoute seeds + 4 region-transfer folds + 5 LSTM"
echo " seeds + 4 LSTM transfer folds are the heavy stages; trained stages are"
echo " checkpointed, so an interrupted run resumes.)"
echo "[8/27] USGS experiment (primary baselines + ThermoRoute × seeds + LGO + ablations)"
echo "      using panel: ${USGS_PANEL}"
# Stage 9 is an immutable parent.  Its command returns successfully only after
# the report, three formal pointers and final content-bound completion receipt
# are durable.  Stage 24 rejects a missing or stale receipt and binds the
# accepted receipt into the frozen suite identity.
"$THERMOROUTE_PYTHON" scripts/09_usgs_experiment.py --panel "${USGS_PANEL}" --seeds 5 \
    --device cpu \
    --out_predictions usgs_predictions_stage9_v2.parquet
echo "[9/27] matched-budget neural controls + exact 31-member feature ladder"
# Stage 09b publishes its content-bound receipt only after all 31 member
# predictions, their sidecars, the common-key audit, budget, combined
# predictions and report validate.  Stage 24 requires and revalidates it.
"$THERMOROUTE_PYTHON" scripts/09b_development_controls.py --panel "${USGS_PANEL}"
echo "[10/27] per-station LightGBM (M4 — the stronger-of-two learned-baseline foil)"
"$THERMOROUTE_PYTHON" scripts/_perstation_lgb.py --panel "${USGS_PANEL}"
echo "[11/27] exploratory development holdout + 5-seed ablations"
"$THERMOROUTE_PYTHON" scripts/13_rigor.py
echo "[12/27] exploratory leave-HUC2-region-out gauged transfer — 4 folds of ThermoRoute"
for f in 0 1 2 3; do "$THERMOROUTE_PYTHON" scripts/13c_region_transfer.py --fold "$f"; done
echo "[13/27] region-transfer assemble: global LightGBM per fold + descriptive figure"
"$THERMOROUTE_PYTHON" scripts/13c_region_transfer.py --assemble
echo "[14/27] deep sequence baseline (global LSTM): in-sample × 5 seeds -> derive final v2"
"$THERMOROUTE_PYTHON" scripts/16_lstm_baseline.py --insample
echo "[15/27] deep sequence baseline: leave-HUC2-region-out transfer (4 folds)"
"$THERMOROUTE_PYTHON" scripts/16_lstm_baseline.py --transfer
echo "[16/27] 3-way transfer + in-sample LSTM report"
"$THERMOROUTE_PYTHON" scripts/16_lstm_baseline.py --report
echo "[17/27] Algebraic diagnostic (Fig 3; no safety claim)"
"$THERMOROUTE_PYTHON" scripts/17_prop1_binding.py
echo "[18/27] descriptive REV curve over the cost-loss grid (Fig 5)"
"$THERMOROUTE_PYTHON" scripts/18_rev_curve.py
echo "[19/27] probabilistic (PICP/three-quantile score/reliability/Brier) + multi-metric (Fig 4)"
"$THERMOROUTE_PYTHON" scripts/19_probabilistic.py
"$THERMOROUTE_PYTHON" scripts/19_probabilistic.py --check
echo "[20/27] legacy transfer diagnostics (not ungauged) + regime stratification"
"$THERMOROUTE_PYTHON" scripts/20_tuurt.py
"$THERMOROUTE_PYTHON" scripts/15_stratified.py
echo "[21/27] ecological-threshold eligibility audit / strict 7DADM when inputs exist"
"$THERMOROUTE_PYTHON" scripts/21_ecological_thresholds.py
echo "[22/27] adaptive conformal diagnostics (no conditional-coverage claim)"
"$THERMOROUTE_PYTHON" scripts/22_adaptive_conformal.py
echo "[23/27] predeclared input-stress/OOD robustness (frozen ensemble; common keys)"
"$THERMOROUTE_PYTHON" scripts/23_robustness.py --panel "${USGS_PANEL}"
echo "[24/27] USGS calibration/REV/mechanism and claim statistics"
"$THERMOROUTE_PYTHON" scripts/10_usgs_analysis.py
"$THERMOROUTE_PYTHON" scripts/12_claim_stats.py
echo "[25/27] station-agnostic pooled external suite (development data only)"
"$THERMOROUTE_PYTHON" scripts/25_train_external_pooled_suite.py
"$THERMOROUTE_PYTHON" scripts/25_train_external_pooled_suite.py --check
echo "[26/27] freeze the complete Route-A model suite"
"$THERMOROUTE_PYTHON" scripts/24_freeze_model_suite.py \
    --stage9-receipt outputs/models/route_a_stage09_completion.json \
    --stage09b-receipt outputs/models/route_a_stage09b_completion.json \
    --lstm-receipt outputs/models/route_a_stage16_completion.json \
    --external-receipt outputs/models/route_a_stage25_completion.json
echo "[27/27] isolated full-model replay and final artifact manifest"
if [[ -f outputs/model_replay/route_a_development_replay_v1.json ]]; then
  "$THERMOROUTE_PYTHON" -I -B scripts/27_verify_development_replay.py --check
else
  "$THERMOROUTE_PYTHON" -I -B scripts/27_verify_development_replay.py
fi
"$THERMOROUTE_PYTHON" scripts/14_manifest.py

echo ""
echo "DONE — development outputs are under outputs/; see the root README and frozen claim ledger for authority."
echo "Route A still requires its separate freeze, chronology, authorization, and opening chain."
echo "Rebuild the PDFs with: (cd paper && ../scripts/... ) — see README."

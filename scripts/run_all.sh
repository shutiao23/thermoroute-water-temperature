#!/usr/bin/env bash
# Reproduce the canonical Route-A USGS development analysis end to end.
# The older b1/s2/p3 ordinary monitoring stations are not Route-A evidence;
# their case is deliberately outside this canonical runner.
# The USGS panel must already be acquired in data_usgs/; the
# acquisition step is network-bound and shipped as pre-acquired panels.
#
# Usage:
#   bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if (( $# != 0 )); then
  echo "usage: bash scripts/run_all.sh" >&2
  exit 2
fi

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

mkdir -p outputs/{tables,figures,predictions,reports,models,logs}
export PYTHONPATH=src
# Numerical policy: protocols/route_a_numerical_policy_v2.json freezes the
# role caps (8 for every stage-09/09b/16/25 role).  The ambient value must
# equal the frozen role cap; scripts fail closed via assert_role_thread_cap.
export THERMOROUTE_FORMAL_THREADS="${THERMOROUTE_FORMAL_THREADS:-8}"
export OMP_NUM_THREADS="$THERMOROUTE_FORMAL_THREADS"
export MKL_NUM_THREADS="$THERMOROUTE_FORMAL_THREADS"
export OPENBLAS_NUM_THREADS="$THERMOROUTE_FORMAL_THREADS"
export VECLIB_MAXIMUM_THREADS="$THERMOROUTE_FORMAL_THREADS"
export NUMEXPR_NUM_THREADS="$THERMOROUTE_FORMAL_THREADS"
export WORKER_THREADS="$THERMOROUTE_FORMAL_THREADS"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
readonly CANONICAL_USGS_PANEL="data_usgs/panel_usgs_120v2.parquet"
readonly CANONICAL_USGS_STATION_REGISTRY="data_usgs/station_registry_v1.csv"
if [[ -n "${USGS_PANEL:-}" && "$USGS_PANEL" != "$CANONICAL_USGS_PANEL" ]]; then
  echo "run_all freezes the canonical panel; custom USGS_PANEL is unsupported" >&2
  exit 2
fi
export USGS_PANEL="$CANONICAL_USGS_PANEL"
if [[ -n "${USGS_STATION_REGISTRY:-}" && \
      "$USGS_STATION_REGISTRY" != "$CANONICAL_USGS_STATION_REGISTRY" ]]; then
  echo "run_all freezes the canonical station registry; custom USGS_STATION_REGISTRY is unsupported" >&2
  exit 2
fi
export USGS_STATION_REGISTRY="$CANONICAL_USGS_STATION_REGISTRY"
if [[ ! -f "$USGS_PANEL" ]]; then
  echo "USGS_PANEL does not exist: $USGS_PANEL" >&2
  exit 2
fi
if [[ ! -f "$USGS_STATION_REGISTRY" ]]; then
  echo "USGS_STATION_REGISTRY does not exist: $USGS_STATION_REGISTRY" >&2
  exit 2
fi
"$THERMOROUTE_PYTHON" scripts/14_manifest.py --check-route-a-boundary

echo "================ PRE-FLIGHT TESTS ================"
"$THERMOROUTE_PYTHON" -m pytest tests/ -q

echo ""
echo "================ ROUTE A: USGS DEVELOPMENT ANALYSIS ================"
echo "(multi-hour on CPU: 5 ThermoRoute seeds + 4 region-transfer folds + 5 LSTM"
echo " seeds + 4 LSTM transfer folds are the heavy stages; trained stages are"
echo " checkpointed, so an interrupted run resumes.)"
echo "[1/19] USGS experiment (primary baselines + ThermoRoute × seeds + LGO + ablations)"
echo "      using panel: ${USGS_PANEL}"
# Stage 9 is an immutable parent.  Its command returns successfully only after
# the report, three formal pointers and final content-bound completion receipt
# are durable.  Stage 24 rejects a missing or stale receipt and binds the
# accepted receipt into the frozen suite identity.
"$THERMOROUTE_PYTHON" scripts/09_usgs_experiment.py --panel "${USGS_PANEL}" --seeds 5 \
    --device cpu \
    --out_predictions usgs_predictions_stage9_v2.parquet
echo "[2/19] matched-budget neural controls + complete declared-seed feature ladder"
# Stage 09b publishes its content-bound receipt only after every declared member
# predictions, their sidecars, the common-key audit, budget, combined
# predictions and report validate.  Stage 24 requires and revalidates it.
"$THERMOROUTE_PYTHON" scripts/09b_development_controls.py --panel "${USGS_PANEL}"
echo "[3/19] per-station LightGBM (M4 — the stronger-of-two learned-baseline foil)"
"$THERMOROUTE_PYTHON" scripts/_perstation_lgb.py --panel "${USGS_PANEL}"
echo "[4/19] exploratory development holdout + 5-seed ablations"
"$THERMOROUTE_PYTHON" scripts/13_rigor.py
echo "[5/19] exploratory leave-HUC2-region-out gauged transfer — 4 folds of ThermoRoute"
for f in 0 1 2 3; do "$THERMOROUTE_PYTHON" scripts/13c_region_transfer.py --fold "$f"; done
echo "[6/19] region-transfer assemble: global LightGBM per fold + descriptive figure"
"$THERMOROUTE_PYTHON" scripts/13c_region_transfer.py --assemble
echo "[7/19] deep sequence baseline (global LSTM): in-sample × 5 seeds -> derive final v2"
"$THERMOROUTE_PYTHON" scripts/16_lstm_baseline.py --insample
echo "[8/19] deep sequence baseline: leave-HUC2-region-out transfer (4 folds)"
"$THERMOROUTE_PYTHON" scripts/16_lstm_baseline.py --transfer
echo "[9/19] 3-way transfer + in-sample LSTM report"
"$THERMOROUTE_PYTHON" scripts/16_lstm_baseline.py --report
echo "[10/19] Algebraic diagnostic (Fig 3; no safety claim)"
"$THERMOROUTE_PYTHON" scripts/17_prop1_binding.py
echo "[11/19] fail-closed REV NOT EVALUATED status (no predeclared cost-loss ratios)"
"$THERMOROUTE_PYTHON" scripts/18_rev_curve.py
echo "[12/19] probabilistic (PICP/three-quantile score/reliability/Brier) + multi-metric (Fig 4)"
"$THERMOROUTE_PYTHON" scripts/19_probabilistic.py
"$THERMOROUTE_PYTHON" scripts/19_probabilistic.py --check
echo "[13/19] legacy transfer diagnostics (not ungauged) + regime stratification"
"$THERMOROUTE_PYTHON" scripts/20_tuurt.py
"$THERMOROUTE_PYTHON" scripts/15_stratified.py
echo "[14/19] adaptive conformal diagnostics (no conditional-coverage claim)"
"$THERMOROUTE_PYTHON" scripts/22_adaptive_conformal.py
echo "[15/19] predeclared input-stress/OOD robustness (frozen ensemble; common keys)"
"$THERMOROUTE_PYTHON" scripts/23_robustness.py --panel "${USGS_PANEL}"
echo "[16/19] USGS calibration/latent diagnostics and claim statistics"
"$THERMOROUTE_PYTHON" scripts/10_usgs_analysis.py
"$THERMOROUTE_PYTHON" scripts/12_claim_stats.py
echo "[17/19] station-agnostic pooled external suite (development data only)"
"$THERMOROUTE_PYTHON" scripts/25_train_external_pooled_suite.py
"$THERMOROUTE_PYTHON" scripts/25_train_external_pooled_suite.py --check
echo "[18/19] freeze the complete Route-A model suite"
"$THERMOROUTE_PYTHON" scripts/24_freeze_model_suite.py \
    --stage9-receipt outputs/models/route_a_stage09_completion.json \
    --stage09b-receipt outputs/models/route_a_stage09b_completion.json \
    --lstm-receipt outputs/models/route_a_stage16_completion.json \
    --external-receipt outputs/models/route_a_stage25_completion.json
echo "[19/19] isolated full-model replay and final artifact manifest"
if [[ -f outputs/model_replay/route_a_development_replay_v1.json ]]; then
  "$THERMOROUTE_PYTHON" -I -B scripts/27_verify_development_replay.py --check
else
  "$THERMOROUTE_PYTHON" -I -B scripts/27_verify_development_replay.py
fi
"$THERMOROUTE_PYTHON" scripts/14_manifest.py --development-prelabel
"$THERMOROUTE_PYTHON" scripts/14_manifest.py --check --development-prelabel \
    --strict-environment

echo ""
echo "DONE — development outputs are under outputs/; see the root README and frozen claim ledger for authority."
echo "Route A still requires its separate freeze, chronology, authorization, and opening chain."
echo "Rebuild the PDFs with: (cd paper && ../scripts/... ) — see README."

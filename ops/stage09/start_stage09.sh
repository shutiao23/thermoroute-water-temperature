#!/usr/bin/env bash
# Formal Stage-09 launcher (multicore Route-A amendment v2).
# Numerical policy route_a_numerical_policy_v2.json freezes cap 8 for every
# role; parent and worker processes share the cap so the runtime identity
# check holds.  Throughput is process-level: 5 seed workers + control workers
# (execution-only; not in RunIdentity).  Memory budget for a 125Gi host:
# seed phase ~35Gi, control phase 16 x ~2.8Gi ≈ 45Gi peak.
#
# This starts a NEW content-addressed run under the current source tree.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OPS_DIR="$REPO_ROOT/ops/stage09"
ENV_CANDIDATES=(
  "${PHASE2_WATCH_ENV:-}"
  "$OPS_DIR/phase2_watch.env"
  "$REPO_ROOT/outputs/logs/phase2_watch.env"
)
for envf in "${ENV_CANDIDATES[@]}"; do
  [[ -n "$envf" && -f "$envf" ]] || continue
  set -a
  # shellcheck disable=SC1090
  source "$envf"
  set +a
  break
done

export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src
export THERMOROUTE_FORMAL_THREADS="${THERMOROUTE_FORMAL_THREADS:-8}"
export OMP_NUM_THREADS="$THERMOROUTE_FORMAL_THREADS" \
  MKL_NUM_THREADS="$THERMOROUTE_FORMAL_THREADS" \
  OPENBLAS_NUM_THREADS="$THERMOROUTE_FORMAL_THREADS"
export VECLIB_MAXIMUM_THREADS="$THERMOROUTE_FORMAL_THREADS" \
  NUMEXPR_NUM_THREADS="$THERMOROUTE_FORMAL_THREADS" \
  WORKER_THREADS="$THERMOROUTE_FORMAL_THREADS"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset PYTHONHASHSEED PYTHONPYCACHEPREFIX || true

THERMOROUTE_PYTHON="${THERMOROUTE_PYTHON:-$PWD/.venv-route-a/bin/python}"
CONTROL_WORKERS="${STAGE09_CONTROL_WORKERS:-16}"
if [[ "$CONTROL_WORKERS" -gt 96 ]]; then CONTROL_WORKERS=96; fi
if [[ "$CONTROL_WORKERS" -lt 1 ]]; then CONTROL_WORKERS=1; fi

EXPECTED_SOURCE_SHA256="${EXPECTED_SOURCE_SHA256:-<NEW_SOURCE_SHA256>}"
VOID_SOURCE_SHA256="${VOID_SOURCE_SHA256:-ee99225c55b2ceacdac6fdf596f0b452d5dbdb4d417edb81e234732b81521ab1}"
VOID_STAGE09_RUN_IDS="${VOID_STAGE09_RUN_IDS:-bb02498a8396ea7c6110}"

mkdir -p outputs/logs
LOG=outputs/logs/stage09_formal.log

# Preflight: refuse a live Stage-09 already running.
stage09_already_live=0
for pid in $(pgrep -f 'scripts/09_usgs_experiment\.py' 2>/dev/null || true); do
  [[ -r "/proc/$pid/cmdline" ]] || continue
  cmd="$(tr '\0' ' ' <"/proc/$pid/cmdline")"
  case "$cmd" in
    *python*scripts/09_usgs_experiment.py*)
      case "$cmd" in
        *\ --help*|*\ -h\ *|*/bin/bash\ -O\ extglob*) continue ;;
      esac
      stage09_already_live=1
      break
      ;;
  esac
done
if [[ "$stage09_already_live" -eq 1 ]]; then
  echo "[$(date -Is)] refuse: Stage-09 python already live" | tee -a "$LOG" >&2
  exit 1
fi

SOURCE="$("$THERMOROUTE_PYTHON" - <<'PY'
from pathlib import Path
from thermoroute.repro import source_tree_hash
print(source_tree_hash(Path(".").resolve()))
PY
)"
echo "[$(date -Is)] preflight live source_sha256=${SOURCE:0:12}… expected=${EXPECTED_SOURCE_SHA256:0:12}… (workers=$CONTROL_WORKERS)" | tee -a "$LOG"
if [[ "$SOURCE" == "$VOID_SOURCE_SHA256" ]]; then
  echo "[$(date -Is)] FATAL: source tree still matches voided bb02498a lineage ($VOID_SOURCE_SHA256)" | tee -a "$LOG" >&2
  exit 2
fi
if [[ "$EXPECTED_SOURCE_SHA256" == "$VOID_SOURCE_SHA256" ]]; then
  echo "[$(date -Is)] FATAL: EXPECTED_SOURCE_SHA256 points at voided bb02498a lineage" | tee -a "$LOG" >&2
  exit 2
fi
if [[ "$SOURCE" != "$EXPECTED_SOURCE_SHA256" ]]; then
  echo "[$(date -Is)] FATAL: live source_tree_hash=${SOURCE} != EXPECTED_SOURCE_SHA256=${EXPECTED_SOURCE_SHA256}" | tee -a "$LOG" >&2
  exit 2
fi
# Refuse any accidental pin to voided run_id (bb02498a…).
if [[ -n "${EXPECTED_STAGE09_RUN_ID:-}" ]]; then
  case ",$VOID_STAGE09_RUN_IDS," in
    *",$EXPECTED_STAGE09_RUN_ID,"*)
      echo "[$(date -Is)] FATAL: EXPECTED_STAGE09_RUN_ID=$EXPECTED_STAGE09_RUN_ID is void; do not resume bb02498a" | tee -a "$LOG" >&2
      exit 2
      ;;
  esac
fi

echo "[$(date -Is)] launching NEW Stage-09 under source_sha256=${SOURCE:0:12}… (workers=$CONTROL_WORKERS, threads=$THERMOROUTE_FORMAL_THREADS)" | tee -a "$LOG"

# 125Gi host: parent ~7GB + control member ~2.8GB; 16 workers ≈ 45Gi peak.
exec "$THERMOROUTE_PYTHON" scripts/09_usgs_experiment.py \
  --panel data_usgs/panel_usgs_120v2.parquet \
  --seeds 5 --device cpu --control-workers "$CONTROL_WORKERS" \
  --out_predictions usgs_predictions_stage9_v2.parquet

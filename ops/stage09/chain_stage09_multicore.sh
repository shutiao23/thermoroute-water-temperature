#!/usr/bin/env bash
# chain_stage09_multicore.sh — model-training subchain runner (v2).
#
# IMPORTANT: this runs ONLY the model-training subchain (Stage-09, 09b, 16,
# 25) = 4 of the 19 development-chain steps.  The full development chain
# continues with per-station/global LightGBM fairness, region transfer,
# probabilistic/robustness, claim statistics, Stage-24 model-suite freeze,
# Stage-27 isolated replay, and the strict manifest (see scripts/run_all.sh).
# On success this script prints MODEL_TRAINING_SUBCHAIN_COMPLETE — it never
# claims the development chain is complete.
#
# Numerical policy: route_a_numerical_policy_v2.json freezes cap 8 for every
# role.  Parent and workers of one run share one cap so the runtime identity
# check holds.  The ambient THERMOROUTE_FORMAL_THREADS must equal the role
# cap; the scripts fail closed otherwise.
#
# Memory budget (125Gi host, ceiling 96Gi):
#   - Stage-09 seeds:   5 x ~7Gi ≈ 35Gi (40 threads)
#   - Stage-09 control: 16 members x ~2.8Gi ≈ 45Gi peak (128 threads)
#   - Stage-09b:        16 members x ~2.8Gi ≈ 45Gi peak (128 threads)
#   - Stage-16:         9 tasks x ~2.5Gi ≈ 25Gi (72 threads)
#   - Stage-25:         single process (8 threads)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PY=/home/lzq/anaconda3/envs/route-a/bin/python
LOG=outputs/logs/multicore_chain.log
PEAK_LOG=outputs/logs/multicore_chain.memory.csv
mkdir -p outputs/logs
[ -f "$PEAK_LOG" ] || echo "ts,phase,used_gb,available_gb" > "$PEAK_LOG"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

PEAK_USED=0
memory_sample() {
  local used avail
  read -r _ used avail _ < <(free -g | awk 'NR==2 {print $2,$3,$7}')
  if [ "$used" -gt "$PEAK_USED" ]; then PEAK_USED="$used"; fi
  echo "$(date -u +%H:%M:%S),${1:-$PHASE},${used},${avail}" >> "$PEAK_LOG"
}
PHASE=idle

set_threads() {
  export THERMOROUTE_FORMAL_THREADS="$1"
  export OMP_NUM_THREADS="$1" MKL_NUM_THREADS="$1" OPENBLAS_NUM_THREADS="$1"
  export VECLIB_MAXIMUM_THREADS="$1" NUMEXPR_NUM_THREADS="$1" WORKER_THREADS="$1"
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
}

log "chain start (worktree=$REPO_ROOT, policy=route_a_numerical_policy_v2, cap=8)"

# 1) Wait for any other Stage-09 python (e.g. the serial main-tree run) to exit.
log "waiting for other Stage-09 processes to exit..."
while pgrep -f 'python.*scripts/09_usgs_experiment\.py' >/dev/null 2>&1; do
  sleep 60
done
log "no other Stage-09 live; starting model-training subchain"

# 2) Stage-09 multicore (5 seed processes x 8 threads, 16 control workers).
set_threads 8
PHASE=stage09
log "Stage-09: seeds=5 workers=16 cap=8"
$PY scripts/09_usgs_experiment.py \
  --panel data_usgs/panel_usgs_120v2.parquet \
  --seeds 5 --device cpu --control-workers 16 \
  --out_predictions usgs_predictions_stage9_v2.parquet >>"$LOG" 2>&1 &
STAGE09_PID=$!
while kill -0 "$STAGE09_PID" 2>/dev/null; do memory_sample; sleep 60; done
memory_sample
wait "$STAGE09_PID"
log "Stage-09 DONE (receipt: outputs/models/route_a_stage09_completion.json)"

# 3) Stage-09b development controls (16 member processes x 8 threads).
PHASE=stage09b
log "Stage-09b: precompute-workers=16 cap=8"
$PY scripts/09b_development_controls.py --precompute-workers 16 >>"$LOG" 2>&1 &
STAGE09B_PID=$!
while kill -0 "$STAGE09B_PID" 2>/dev/null; do memory_sample; sleep 60; done
memory_sample
wait "$STAGE09B_PID"
log "Stage-09b DONE"

# 4) Stage-16 LSTM baseline (grid + 5 seeds + 4 folds, 8 threads each).
PHASE=stage16
log "Stage-16: insample"
$PY scripts/16_lstm_baseline.py --insample >>"$LOG" 2>&1 &
S16_PID=$!
while kill -0 "$S16_PID" 2>/dev/null; do memory_sample; sleep 60; done
memory_sample
wait "$S16_PID"
log "Stage-16: transfer"
$PY scripts/16_lstm_baseline.py --transfer >>"$LOG" 2>&1 &
S16B_PID=$!
while kill -0 "$S16B_PID" 2>/dev/null; do memory_sample; sleep 60; done
memory_sample
wait "$S16B_PID"
log "Stage-16: report"
$PY scripts/16_lstm_baseline.py --report >>"$LOG" 2>&1
log "Stage-16 DONE"

# 5) Stage-25 external pooled suite.
PHASE=stage25
log "Stage-25: external pooled suite"
$PY scripts/25_train_external_pooled_suite.py >>"$LOG" 2>&1 &
S25_PID=$!
while kill -0 "$S25_PID" 2>/dev/null; do memory_sample; sleep 60; done
memory_sample
wait "$S25_PID"
log "Stage-25 DONE"
log "MODEL_TRAINING_SUBCHAIN_COMPLETE (4/19; remaining steps: per-station LGB fairness, "
log "region transfer, probabilistic/robustness, claim stats, Stage-24 freeze, "
log "Stage-27 replay, strict manifest; see scripts/run_all.sh)"
log "peak memory observed: ${PEAK_USED} GiB"

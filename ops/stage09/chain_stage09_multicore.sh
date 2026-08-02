#!/usr/bin/env bash
# chain_stage09_multicore.sh — multicore Route-A lineage runner.
# 1) Waits for the serial main-tree Stage-09 to exit (any live
#    scripts/09_usgs_experiment.py python) so the two lineages never share
#    the host at the same time.
# 2) Runs the multicore lineage in THIS worktree: 09 -> 09b -> 16 -> 25.
# Memory budget (125Gi host, ceiling 96Gi):
#   - Stage-09 seeds:  5 x ~6.6Gi ≈ 35Gi (80 threads)
#   - Stage-09 control: 32 members x ~2.6Gi ≈ 90Gi peak (64 threads)
#   - Stage-09b:        32 members x ~2.6Gi ≈ 90Gi peak (64 threads)
#   - Stage-16:         9 tasks x ~2.5Gi ≈ 25Gi (90 threads)
#   - Stage-25:         single process (16 threads)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PY=/home/lzq/anaconda3/envs/route-a/bin/python
LOG=outputs/logs/multicore_chain.log
mkdir -p outputs/logs

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

set_threads() {
  export THERMOROUTE_FORMAL_THREADS="$1"
  export OMP_NUM_THREADS="$1" MKL_NUM_THREADS="$1" OPENBLAS_NUM_THREADS="$1"
  export VECLIB_MAXIMUM_THREADS="$1" NUMEXPR_NUM_THREADS="$1" WORKER_THREADS="$1"
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
}

log "chain start (worktree=$REPO_ROOT)"

# 1) Wait for the serial main-tree run (parent + control workers) to exit.
log "waiting for serial main-tree Stage-09 to exit..."
while pgrep -f 'python.*scripts/09_usgs_experiment\.py' >/dev/null 2>&1; do
  sleep 60
done
log "serial run gone; starting multicore lineage"

# 2) Stage-09 multicore (5 seed processes x 16 threads, 32 control workers).
set_threads 16
log "Stage-09: seeds=5 workers=32 threads=16"
$PY scripts/09_usgs_experiment.py \
  --panel data_usgs/panel_usgs_120v2.parquet \
  --seeds 5 --device cpu --control-workers 32 \
  --out_predictions usgs_predictions_stage9_v2.parquet >>"$LOG" 2>&1
log "Stage-09 DONE (receipt: outputs/models/route_a_stage09_completion.json)"

# 3) Stage-09b development controls (32 member processes x 2 threads).
set_threads 2
log "Stage-09b: precompute-workers=32 threads=2"
$PY scripts/09b_development_controls.py --precompute-workers 32 >>"$LOG" 2>&1
log "Stage-09b DONE"

# 4) Stage-16 LSTM baseline (grid + 5 seeds + 4 folds, 10 threads each).
set_threads 10
log "Stage-16: insample"
$PY scripts/16_lstm_baseline.py --insample >>"$LOG" 2>&1
log "Stage-16: transfer"
$PY scripts/16_lstm_baseline.py --transfer >>"$LOG" 2>&1
log "Stage-16: report"
$PY scripts/16_lstm_baseline.py --report >>"$LOG" 2>&1
log "Stage-16 DONE"

# 5) Stage-25 external pooled suite.
set_threads 16
log "Stage-25: external pooled suite"
$PY scripts/25_train_external_pooled_suite.py >>"$LOG" 2>&1
log "Stage-25 DONE"
log "CHAIN COMPLETE"

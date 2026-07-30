#!/usr/bin/env bash
# Watch Stage-09 completion receipt, then continue run_all steps 2-19.
#
# Hardened for WSL reboot / transient Stage-09 disappearances:
# - flock single-flight + lock.meta.json (pid/boot_id) for stale-lock diagnosis
# - bind promotion to EXPECTED_SOURCE_SHA256 (required); never default to voided bb02498a
# - refuse VOID_STAGE09_RUN_IDS even if a stale receipt is on disk
# - by default keep waiting if Stage-09 exits without receipt (self-heal / relaunch)
# - refuse Phase-2 while Stage-09 python still live; refuse if orphan 09b live
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p outputs/logs

OPS_DIR="$REPO_ROOT/ops/stage09"
ENV_CANDIDATES=(
  "${PHASE2_WATCH_ENV:-}"
  "$OPS_DIR/phase2_watch.env"
  "$REPO_ROOT/outputs/logs/phase2_watch.env"
)
for envf in "${ENV_CANDIDATES[@]}"; do
  [[ -n "$envf" && -f "$envf" ]] || continue
  # shellcheck disable=SC1090
  set -a
  # shellcheck disable=SC1090
  source "$envf"
  set +a
  echo "[$(date -Is)] loaded env $envf" >&2
  break
done

LOCK_FILE=outputs/logs/phase2_watch.lock
LOCK_META=outputs/logs/phase2_watch.lock.meta.json
LOG=outputs/logs/phase2_watch.log
RECEIPT=outputs/models/route_a_stage09_completion.json

export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset PYTHONHASHSEED PYTHONPYCACHEPREFIX || true
export THERMOROUTE_PYTHON="${THERMOROUTE_PYTHON:-$PWD/.venv-route-a/bin/python}"

HOST_CPUS="$(nproc --all 2>/dev/null || nproc)"
PHASE2_09B_WORKERS="${PHASE2_09B_WORKERS:-6}"
if [[ "$PHASE2_09B_WORKERS" -gt 8 ]]; then PHASE2_09B_WORKERS=8; fi
if [[ "$PHASE2_09B_WORKERS" -lt 1 ]]; then PHASE2_09B_WORKERS=1; fi

# Binding: source digest is mandatory. run_id is optional until the new run exists.
EXPECTED_SOURCE_SHA256="${EXPECTED_SOURCE_SHA256:-}"
EXPECTED_STAGE09_RUN_ID="${EXPECTED_STAGE09_RUN_ID:-}"
VOID_STAGE09_RUN_IDS="${VOID_STAGE09_RUN_IDS:-bb02498a8396ea7c6110}"
VOID_SOURCE_SHA256="${VOID_SOURCE_SHA256:-ee99225c55b2ceacdac6fdf596f0b452d5dbdb4d417edb81e234732b81521ab1}"
PHASE2_ABORT_ON_STAGE09_EXIT="${PHASE2_ABORT_ON_STAGE09_EXIT:-0}"

if [[ -z "$EXPECTED_SOURCE_SHA256" ]]; then
  echo "[$(date -Is)] FATAL: EXPECTED_SOURCE_SHA256 unset. Copy ops/stage09/phase2_watch.env.example → phase2_watch.env" | tee -a "$LOG" >&2
  exit 2
fi
hex_only="$(printf '%s' "$EXPECTED_SOURCE_SHA256" | tr -cd '0-9a-fA-F')"
if [[ "$EXPECTED_SOURCE_SHA256" != "$hex_only" || ${#EXPECTED_SOURCE_SHA256} -ne 64 ]]; then
  echo "[$(date -Is)] FATAL: EXPECTED_SOURCE_SHA256 must be 64 hex chars" | tee -a "$LOG" >&2
  exit 2
fi
# Fail closed: refuse voided bb02498a lineage (run_id and its source digest).
if [[ "$EXPECTED_SOURCE_SHA256" == "$VOID_SOURCE_SHA256" ]]; then
  echo "[$(date -Is)] FATAL: EXPECTED_SOURCE_SHA256 is voided bb02498a lineage ($VOID_SOURCE_SHA256)" | tee -a "$LOG" >&2
  exit 2
fi
if [[ -n "$EXPECTED_STAGE09_RUN_ID" ]]; then
  case ",$VOID_STAGE09_RUN_IDS," in
    *",$EXPECTED_STAGE09_RUN_ID,"*)
      echo "[$(date -Is)] FATAL: EXPECTED_STAGE09_RUN_ID=$EXPECTED_STAGE09_RUN_ID is void; use a NEW Stage-09 under source_sha256=$EXPECTED_SOURCE_SHA256" | tee -a "$LOG" >&2
      exit 2
      ;;
  esac
fi

# Preflight: live tree must equal the bound expected digest (no silent drift).
LIVE_SOURCE="$("$THERMOROUTE_PYTHON" - <<'PY'
from pathlib import Path
from thermoroute.repro import source_tree_hash
print(source_tree_hash(Path(".").resolve()))
PY
)"
if [[ "$LIVE_SOURCE" != "$EXPECTED_SOURCE_SHA256" ]]; then
  echo "[$(date -Is)] FATAL: live source_tree_hash=${LIVE_SOURCE} != EXPECTED_SOURCE_SHA256=${EXPECTED_SOURCE_SHA256}" | tee -a "$LOG" >&2
  exit 2
fi
if [[ "$LIVE_SOURCE" == "$VOID_SOURCE_SHA256" ]]; then
  echo "[$(date -Is)] FATAL: live source tree still matches voided bb02498a lineage" | tee -a "$LOG" >&2
  exit 2
fi

# --- stale-lock recovery (flock is authoritative; meta is diagnostic; never blind-rm lock) ---
recycle_stale_lock_meta() {
  if [[ ! -f "$LOCK_META" ]]; then
    return 0
  fi
  local meta_pid boot_now boot_meta
  meta_pid="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("pid",""))' "$LOCK_META" 2>/dev/null || true)"
  boot_now="$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo unknown)"
  boot_meta="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("boot_id",""))' "$LOCK_META" 2>/dev/null || true)"
  if [[ -n "$meta_pid" && -d "/proc/$meta_pid" ]]; then
    return 0
  fi
  # Dead PID and/or boot_id changed → previous WSL session; drop meta only (keep lock inode).
  if [[ -z "$meta_pid" || ! -d "/proc/$meta_pid" || ( -n "$boot_meta" && "$boot_meta" != "$boot_now" ) ]]; then
    echo "[$(date -Is)] recycling stale lock meta (pid=${meta_pid:-none} boot_meta=${boot_meta:-none} boot_now=$boot_now)" | tee -a "$LOG"
    rm -f "$LOCK_META"
  fi
}

recycle_stale_lock_meta

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  # Fail-closed: contention is NOT success (avoids systemd/ensure false-positive active).
  echo "[$(date -Is)] FATAL: another Phase-2 watcher holds $LOCK_FILE (flock contention); refusing exit 0" | tee -a "$LOG" >&2
  exit 75
fi

boot_id="$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo unknown)"
WATCHER_PID=$$
python3 - <<PY
import json, time
from pathlib import Path
meta = {
    "format": "thermoroute.phase2-watch-lock.v1",
    "pid": int("$WATCHER_PID"),
    "boot_id": "$boot_id",
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "repo": str(Path(".").resolve()),
    "expected_source_sha256_prefix": "${EXPECTED_SOURCE_SHA256:0:12}",
    "live_source_sha256_prefix": "${LIVE_SOURCE:0:12}",
}
Path("$LOCK_META").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
PY
trap 'rm -f "$LOCK_META"' EXIT

stage09_python_live() {
  # Ignore --help probes and agent shells that merely mention the path.
  local pid cmd
  for pid in $(pgrep -f 'scripts/09_usgs_experiment\.py' 2>/dev/null || true); do
    [[ -r "/proc/$pid/cmdline" ]] || continue
    cmd="$(tr '\0' ' ' <"/proc/$pid/cmdline")"
    case "$cmd" in
      *python*scripts/09_usgs_experiment.py*)
        case "$cmd" in
          *\ --help*|*\ -h\ *|*/bin/bash\ -O\ extglob*) continue ;;
        esac
        return 0
        ;;
    esac
  done
  return 1
}

apply_formal_threads() {
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 WORKER_THREADS=1
}

apply_hot_threads() {
  export OMP_NUM_THREADS="$HOST_CPUS"
  export MKL_NUM_THREADS="$HOST_CPUS"
  export OPENBLAS_NUM_THREADS="$HOST_CPUS"
  export VECLIB_MAXIMUM_THREADS="$HOST_CPUS"
  export NUMEXPR_NUM_THREADS="$HOST_CPUS"
  export WORKER_THREADS="$HOST_CPUS"
}

apply_formal_threads

echo "[$(date -Is)] watcher start; waiting for $RECEIPT (09b_workers=$PHASE2_09B_WORKERS host_cpus=$HOST_CPUS expected_source=${EXPECTED_SOURCE_SHA256:0:12}… expected_run=${EXPECTED_STAGE09_RUN_ID:-<any-non-void>} abort_on_exit=$PHASE2_ABORT_ON_STAGE09_EXIT)" | tee -a "$LOG"

saw_stage09=0
misses=0
down_notices=0
while true; do
  if stage09_python_live; then
    if [[ $saw_stage09 -eq 0 ]]; then
      echo "[$(date -Is)] observed live Stage-09 python" | tee -a "$LOG"
    fi
    saw_stage09=1
    misses=0
  else
    if [[ $saw_stage09 -eq 0 && ! -f "$RECEIPT" ]]; then
      # Grace: Stage-09 may start after the watcher (or ensure_* will start it).
      misses=$((misses + 1))
      if [[ $misses -eq 1 || $((misses % 30)) -eq 0 ]]; then
        echo "[$(date -Is)] Stage-09 not yet live; still waiting (misses=$misses)" | tee -a "$LOG"
      fi
      # Do not abort for "never appeared" — reboot self-heal relies on waiting.
    elif [[ $saw_stage09 -eq 1 && ! -f "$RECEIPT" ]]; then
      misses=$((misses + 1))
      if [[ "$PHASE2_ABORT_ON_STAGE09_EXIT" == "1" && $misses -ge 2 ]]; then
        echo "[$(date -Is)] Stage-09 process exited without receipt; abort (PHASE2_ABORT_ON_STAGE09_EXIT=1)" | tee -a "$LOG"
        exit 1
      fi
      if [[ $misses -eq 1 || $((misses % 10)) -eq 0 ]]; then
        down_notices=$((down_notices + 1))
        echo "[$(date -Is)] Stage-09 down without receipt; keeping watch for relaunch (misses=$misses)" | tee -a "$LOG"
      fi
      # Allow a later relaunch to count as a fresh observation cycle.
      if [[ $misses -ge 2 ]]; then
        saw_stage09=0
      fi
    fi
  fi
  if [[ -f "$RECEIPT" ]]; then
    break
  fi
  sleep 60
done

echo "[$(date -Is)] receipt present; waiting for Stage-09 python to exit" | tee -a "$LOG"
while stage09_python_live; do
  sleep 30
done

echo "[$(date -Is)] Stage-09 idle; validating receipt (source=${EXPECTED_SOURCE_SHA256:0:12}… run_id=${EXPECTED_STAGE09_RUN_ID:-unset})" | tee -a "$LOG"
apply_formal_threads
"$THERMOROUTE_PYTHON" - <<PY | tee -a "$LOG"
from pathlib import Path
import json
from thermoroute.model_suite import (
    validate_stage09_completion_receipt,
    STAGE9_COMPLETION_RECEIPT_PATH,
    STAGE9_COMPONENT_POINTER_PATH,
)
root = Path(".").resolve()
receipt_path = root / STAGE9_COMPLETION_RECEIPT_PATH
doc = json.loads(receipt_path.read_text(encoding="utf-8"))
run_id = doc.get("run_id") if isinstance(doc.get("run_id"), str) else None
identity = doc.get("run_identity") if isinstance(doc.get("run_identity"), dict) else {}
if run_id is None:
    run_id = identity.get("run_id") if isinstance(identity.get("run_id"), str) else None
source = identity.get("source_sha256") if isinstance(identity.get("source_sha256"), str) else None
expected_source = "$EXPECTED_SOURCE_SHA256"
expected_run = "$EXPECTED_STAGE09_RUN_ID"
void_ids = {x.strip() for x in "$VOID_STAGE09_RUN_IDS".split(",") if x.strip()}
if not source:
    raise SystemExit("receipt lacks run_identity.source_sha256")
if source != expected_source:
    raise SystemExit(
        f"receipt source_sha256={source!r} != expected {expected_source!r} "
        f"(refusing to promote wrong lineage)"
    )
if run_id in void_ids:
    raise SystemExit(
        f"receipt run_id={run_id!r} is void/superseded; "
        f"expected a NEW Stage-09 under source_sha256={expected_source[:12]}…"
    )
if expected_run and run_id != expected_run:
    raise SystemExit(
        f"receipt run_id={run_id!r} != expected {expected_run!r} (stale receipt?)"
    )
validate_stage09_completion_receipt(
    receipt_path,
    root=root,
    stage9_pointer=root / STAGE9_COMPONENT_POINTER_PATH,
)
print("PASS_FORMAL_STAGE09_COMPLETE", "run_id", run_id, "source_sha256", source[:12] + "…")
PY
echo "[$(date -Is)] Stage-09 receipt validated; starting Phase-2 chain" | tee -a "$LOG"

run() {
  echo "[$(date -Is)] RUN $*  [OMP=$OMP_NUM_THREADS WORKER_THREADS=$WORKER_THREADS]" | tee -a "$LOG"
  "$THERMOROUTE_PYTHON" "$@" >>"$LOG" 2>&1
  echo "[$(date -Is)] OK $*" | tee -a "$LOG"
}

PANEL=data_usgs/panel_usgs_120v2.parquet

if pgrep -f '[Pp]ython[^ ]* .*scripts/09b_development_controls\.py' >/dev/null 2>&1; then
  echo "[$(date -Is)] refuse Phase-2: live 09b python still present (reap orphans first)" | tee -a "$LOG"
  exit 1
fi

apply_formal_threads
run scripts/09b_development_controls.py --panel "$PANEL" --precompute-workers "$PHASE2_09B_WORKERS"

apply_hot_threads
run scripts/_perstation_lgb.py --panel "$PANEL"
run scripts/13_rigor.py
for f in 0 1 2 3; do run scripts/13c_region_transfer.py --fold "$f"; done
run scripts/13c_region_transfer.py --assemble

apply_formal_threads
run scripts/16_lstm_baseline.py --insample
run scripts/16_lstm_baseline.py --transfer
run scripts/16_lstm_baseline.py --report

apply_hot_threads
run scripts/17_prop1_binding.py
run scripts/18_rev_curve.py

apply_formal_threads
run scripts/19_probabilistic.py
run scripts/19_probabilistic.py --check

apply_hot_threads
run scripts/20_tuurt.py
run scripts/15_stratified.py
run scripts/22_adaptive_conformal.py
run scripts/23_robustness.py --panel "$PANEL"
run scripts/10_usgs_analysis.py
run scripts/12_claim_stats.py

apply_formal_threads
run scripts/25_train_external_pooled_suite.py
run scripts/25_train_external_pooled_suite.py --check
run scripts/24_freeze_model_suite.py \
  --stage9-receipt outputs/models/route_a_stage09_completion.json \
  --stage09b-receipt outputs/models/route_a_stage09b_completion.json \
  --lstm-receipt outputs/models/route_a_stage16_completion.json \
  --external-receipt outputs/models/route_a_stage25_completion.json
if [[ -f outputs/model_replay/route_a_development_replay_v1.json ]]; then
  "$THERMOROUTE_PYTHON" -I -B scripts/27_verify_development_replay.py --check >>"$LOG" 2>&1
else
  "$THERMOROUTE_PYTHON" -I -B scripts/27_verify_development_replay.py >>"$LOG" 2>&1
fi
run scripts/14_manifest.py --development-prelabel
run scripts/14_manifest.py --check --development-prelabel --strict-environment
echo "[$(date -Is)] Phase-2 chain finished" | tee -a "$LOG"

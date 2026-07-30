#!/usr/bin/env bash
# Ensure Phase-2 watcher is running after WSL reboot / accidental exit.
# Does NOT start long Stage-09 training unless ENABLE_STAGE09_AUTOSTART=1.
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
  set -a
  # shellcheck disable=SC1090
  source "$envf"
  set +a
  break
done

ENABLE_STAGE09_AUTOSTART="${ENABLE_STAGE09_AUTOSTART:-0}"
WATCHER="$OPS_DIR/start_phase2_watch.sh"
STAGE09="$OPS_DIR/start_stage09.sh"
LOG=outputs/logs/ensure_phase2_watch.log
NOHUP_OUT=outputs/logs/phase2_watch.nohup

watcher_live() {
  # Prefer lock meta written by the watcher (avoids false positives from agent
  # shells whose argv merely contains the string "start_phase2_watch").
  local meta=outputs/logs/phase2_watch.lock.meta.json
  local pid cmd
  if [[ -f "$meta" ]]; then
    pid="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("pid",""))' "$meta" 2>/dev/null || true)"
    if [[ -n "$pid" && -d "/proc/$pid" && -r "/proc/$pid/cmdline" ]]; then
      cmd="$(tr '\0' ' ' <"/proc/$pid/cmdline")"
      case "$cmd" in
        *start_phase2_watch.sh*) return 0 ;;
      esac
    fi
  fi
  return 1
}

stage09_live() {
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

echo "[$(date -Is)] ensure_phase2_watch: begin (autostart_stage09=$ENABLE_STAGE09_AUTOSTART)" | tee -a "$LOG"

# Recycle stale lock *meta* only when flock shows no holder. Never blind-rm the lock inode.
LOCK=outputs/logs/phase2_watch.lock
META=outputs/logs/phase2_watch.lock.meta.json
lock_held=0
if [[ -f "$LOCK" || -f "$META" ]]; then
  if ( flock -n 9 && true ) 9>"$LOCK"; then
    # We acquired → no live holder. Drop meta; leave lock for next real watcher.
    rm -f "$META"
    echo "[$(date -Is)] stale lock was free; recycled meta" | tee -a "$LOG"
  else
    lock_held=1
    echo "[$(date -Is)] lock held by live watcher; nothing to clear" | tee -a "$LOG"
  fi
fi

if watcher_live; then
  echo "[$(date -Is)] watcher already live" | tee -a "$LOG"
elif [[ "$lock_held" -eq 1 ]]; then
  # Fail-closed: flock held but meta/pid does not look like our watcher → do not
  # pretend success (systemd false-positive) and do not start a second instance.
  echo "[$(date -Is)] FATAL: $LOCK held but watcher meta/pid not live; refuse blind clear / second start" | tee -a "$LOG" >&2
  exit 75
else
  if [[ ! -x "$WATCHER" ]]; then
    echo "[$(date -Is)] FATAL: missing $WATCHER" | tee -a "$LOG" >&2
    exit 2
  fi
  nohup bash "$WATCHER" >>"$NOHUP_OUT" 2>&1 &
  started_pid=$!
  echo "[$(date -Is)] started watcher pid=$started_pid" | tee -a "$LOG"
  # Verify arming (avoids systemd/ensure reporting success after instant flock/hash fail).
  ok=0
  for _ in 1 2 3 4 5; do
    sleep 1
    if watcher_live; then
      ok=1
      break
    fi
    if ! kill -0 "$started_pid" 2>/dev/null; then
      break
    fi
  done
  if [[ "$ok" -ne 1 ]]; then
    echo "[$(date -Is)] FATAL: watcher did not become live after start (pid=$started_pid); see $NOHUP_OUT" | tee -a "$LOG" >&2
    exit 2
  fi
fi

if [[ "$ENABLE_STAGE09_AUTOSTART" == "1" ]]; then
  if stage09_live; then
    echo "[$(date -Is)] Stage-09 already live" | tee -a "$LOG"
  elif [[ -f outputs/models/route_a_stage09_completion.json ]]; then
    echo "[$(date -Is)] receipt already present; not starting Stage-09" | tee -a "$LOG"
  else
    nohup bash "$STAGE09" >>outputs/logs/stage09_formal.nohup 2>&1 &
    echo "[$(date -Is)] started Stage-09 pid=$! (ENABLE_STAGE09_AUTOSTART=1)" | tee -a "$LOG"
  fi
else
  echo "[$(date -Is)] Stage-09 autostart disabled (set ENABLE_STAGE09_AUTOSTART=1 to relaunch training after reboot)" | tee -a "$LOG"
fi

echo "[$(date -Is)] ensure_phase2_watch: done" | tee -a "$LOG"

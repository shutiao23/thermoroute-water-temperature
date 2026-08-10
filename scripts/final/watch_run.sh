#!/usr/bin/env bash
# Generic watchdog for a long experiment.
#
#   watch_run.sh <name> <pattern> <logfile> <progress-regex> <expected> [stall-seconds]
#
# Writes a one-line status to outputs/final/.watch/<name>.status every poll and
# a verdict when the run ends, so progress survives this shell, the session, and
# any agent teardown.  Several background waiters were silently killed by a
# teardown earlier in this project and their runs looked "stuck" as a result;
# a status file on disk is the thing that does not disappear.
#
# Three outcomes, and silence is never one of them:
#   DONE     process exited and the progress count reached <expected>
#   FAILED   process exited early, with the log tail captured
#   STALLED  process alive but no new progress for <stall-seconds>
#
# It never restarts the run.  Several arms in this project are sealed to exactly
# one execution, so an automatic retry could manufacture a second outcome.

set -uo pipefail

NAME="${1:?usage: watch_run.sh <name> <pattern> <log> <regex> <expected> [stall]}"
PATTERN="${2:?}"
LOG="${3:?}"
REGEX="${4:?}"
EXPECTED="${5:?}"
STALL="${6:-2400}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATUS_DIR="$REPO/outputs/final/.watch"
STATUS="$STATUS_DIR/$NAME.status"
mkdir -p "$STATUS_DIR"

# grep -c already prints 0 when nothing matches *and* exits non-zero, so a
# "|| echo 0" fallback emits two lines and breaks the arithmetic comparison
# that decides DONE versus FAILED.
count_progress() {
    local n
    n=$(grep -cE "$REGEX" "$LOG" 2>/dev/null)
    echo "${n:-0}"
}

# pgrep -f matches whole command lines, and this watchdog carries the pattern as
# an argument -- so a naive `pgrep -f "$PATTERN"` matches the watchdog itself and
# it waits forever on a run that already finished.  That is exactly how a
# completed 24/24 proof got reported as STALLED.  Excluding this script's own
# name is what makes the liveness question about the run.
run_alive() {
    pgrep -af "$PATTERN" 2>/dev/null | grep -v "watch_run.sh" | grep -q .
}
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "$1" | tee "$STATUS"; }

last=-1
changed=$(date +%s)

# Handle the case where the run already finished before the watchdog started.
while run_alive; do
    now=$(date +%s)
    current=$(count_progress)
    if [ "$current" != "$last" ]; then
        last=$current
        changed=$now
        say "$(stamp) RUNNING $NAME $current/$EXPECTED"
    elif [ $((now - changed)) -gt "$STALL" ]; then
        say "$(stamp) STALLED $NAME $current/$EXPECTED no progress for $((now-changed))s"
        tail -25 "$LOG" 2>/dev/null
        exit 3
    fi
    sleep 30
done

final=$(count_progress)
if [ "$final" -ge "$EXPECTED" ]; then
    say "$(stamp) DONE $NAME $final/$EXPECTED"
    exit 0
fi
say "$(stamp) FAILED $NAME $final/$EXPECTED process exited early"
tail -30 "$LOG" 2>/dev/null
exit 1

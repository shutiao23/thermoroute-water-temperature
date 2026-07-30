#!/usr/bin/env bash
# WSL boot hook: re-arm Phase-2 watcher after distro start (no training by default).
#
# Install options (pick one):
#   A) Windows Task Scheduler → wsl.exe -d <Distro> -u <user> -- bash /path/to/wsl-boot-hook.sh
#   B) /etc/wsl.conf:
#        [boot]
#        command = /bin/bash /home/.../ops/stage09/wsl-boot-hook.sh
#      then: wsl --shutdown && reopen
#   C) systemd user unit (see thermoroute-phase2-watch.service) when systemd is enabled
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG="$REPO_ROOT/outputs/logs/wsl_boot_hook.log"
mkdir -p "$REPO_ROOT/outputs/logs"

{
  echo "[$(date -Is)] wsl-boot-hook start"
  # Wait briefly for filesystem / home to be ready after WSL cold start.
  sleep 5
  bash "$REPO_ROOT/ops/stage09/ensure_phase2_watch.sh"
  echo "[$(date -Is)] wsl-boot-hook done"
} >>"$LOG" 2>&1

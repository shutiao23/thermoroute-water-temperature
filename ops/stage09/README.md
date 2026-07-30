# Phase-2 / Stage-09 ops (WSL self-heal)

## Why this exists

Historical failures:

1. `Stage-09 process exited without receipt; abort` — watcher treated a transient Stage-09 disappearance as terminal and never resumed.
2. WSL reboot dropped both Stage-09 and the Phase-2 watcher; leftover `outputs/logs/phase2_watch.lock` (empty) blocked or confused restarts.
3. Hard-coded `EXPECTED_STAGE09_RUN_ID=bb02498a8396ea7c6110` would risk promoting a **void** run after the M-01 source-tree change.

**Current formal rerun** must bind `EXPECTED_SOURCE_SHA256=19289553aa0929bdb5803a8a3eaa96b38a651b3d52da441fb6298f2ac9228b55` and must **never** promote `bb02498a8396ea7c6110`.

## Files

| Path | Role |
|---|---|
| `ops/stage09/start_stage09.sh` | Launch a **new** Stage-09 under the current source tree |
| `ops/stage09/start_phase2_watch.sh` | Wait for receipt → validate source binding → Phase-2 chain |
| `ops/stage09/ensure_phase2_watch.sh` | Recycle stale lock meta; (re)start watcher; optional Stage-09 autostart |
| `ops/stage09/phase2_watch.env.example` | Copy → `phase2_watch.env` |
| `ops/stage09/thermoroute-phase2-watch.service` | systemd --user draft |
| `ops/stage09/wsl-boot-hook.sh` | WSL boot re-arm (watcher only by default) |
| `outputs/logs/start_*.sh` | Thin wrappers → `ops/stage09/` (gitignored; regenerated locally) |

## One-time setup

```bash
cd /home/shutiao/workspace/projects_part_time/project1/thermoroute-water-temperature
cp -n ops/stage09/phase2_watch.env.example ops/stage09/phase2_watch.env
# Confirm source digest matches the new tree:
PYTHONPATH=src .venv-route-a/bin/python -c \
  'from pathlib import Path; from thermoroute.repro import source_tree_hash; print(source_tree_hash(Path(".").resolve()))'
# expect: 19289553aa0929bdb5803a8a3eaa96b38a651b3d52da441fb6298f2ac9228b55
```

Optional reboot self-heal (watcher only — **does not** start training unless you set `ENABLE_STAGE09_AUTOSTART=1` in the env file):

```bash
# systemd user (if enabled in this WSL)
mkdir -p ~/.config/systemd/user
cp ops/stage09/thermoroute-phase2-watch.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now thermoroute-phase2-watch.service

# OR WSL boot.command (root edit /etc/wsl.conf) pointing at wsl-boot-hook.sh
```

## Manual launch (when ready — do not run casually)

```bash
# Terminal A — arm watcher FIRST (binds new source; refuses bb02498a)
bash ops/stage09/ensure_phase2_watch.sh
# or: nohup bash ops/stage09/start_phase2_watch.sh >> outputs/logs/phase2_watch.nohup 2>&1 &

# Terminal B — start NEW Stage-09 (not a bb02498a resume)
nohup bash ops/stage09/start_stage09.sh >> outputs/logs/stage09_formal.nohup 2>&1 &
```

After Stage-09 prints its content-addressed `run_id`, optionally pin it:

```bash
# edit ops/stage09/phase2_watch.env
# EXPECTED_STAGE09_RUN_ID=<new_run_id>
```

## Stale lock policy

- Authoritative lock = `flock` on `outputs/logs/phase2_watch.lock`.
- Diagnostic sidecar = `outputs/logs/phase2_watch.lock.meta.json` (`pid`, `boot_id`).
- Safe to clear when `flock -n` succeeds (no holder) **or** meta PID is dead / `boot_id` changed after WSL restart.
- `ensure_phase2_watch.sh` recycles stale meta automatically.

## Binding / anti-promote rules

Watcher refuses to start Phase-2 unless the receipt’s `run_identity.source_sha256` equals `EXPECTED_SOURCE_SHA256`.
It also refuses any `run_id` listed in `VOID_STAGE09_RUN_IDS` (default includes `bb02498a8396ea7c6110`).

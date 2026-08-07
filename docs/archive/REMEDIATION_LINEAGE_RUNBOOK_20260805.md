# The remediation lineage: runbook

**Date:** 2026-08-05
**Purpose:** execute exactly one full lineage under the corrected source, and never need another before submission.
**Worktree:** `/home/lzq/workspace/parttime/thermoroute-remediation`, branch `feat/route-a-remediation`, based on `feat/multicore` at `f6bc478`.

---

## 1. Why this run exists

Stage-24 cannot complete under the current source: its worker environment hardcodes
one thread and omits `THERMOROUTE_FORMAL_THREADS`, so it replays at one thread the
Stage-09b members trained at eight. See
[`STAGE24_WORKER_THREAD_CAP_BLOCKER_20260805.md`](STAGE24_WORKER_THREAD_CAP_BLOCKER_20260805.md).

Fixing it requires editing `scripts/`, which invalidates all four training receipts.
Since one lineage must be spent, the whole deferred queue is landed at once:
R1–R4, R6–R9. R5 is deferred; see
[`SOURCE_HASH_SCOPE_REMEDIATION_PLAN.md`](SOURCE_HASH_SCOPE_REMEDIATION_PLAN.md) §3.2.

## 2. Isolation

The remediation worktree is separate so the multicore tree's 15 GB of models,
predictions and receipts stay intact as a fallback until the new lineage succeeds.
Nothing in this run writes there.

Preconditions verified 2026-08-05:

| Item | State |
|---|---|
| `data_usgs/panel_usgs_120v2.parquet` | present, sha256 `0427a07e…`, byte-identical to the multicore tree and to the value bound in the existing receipts |
| `station_registry_v1.csv`, `huc_metadata_usgs_v1.csv`, `frozen_panel_v1.json` | present |
| `development_predictor_bridge_v1/` | complete (4 files) |
| `outputs/` | effectively empty — a clean lineage |
| Disk | 403 GB free; the run needs roughly 15 GB |
| Baseline `source_tree_hash` | `bf9500aa…` before the remediation edits |

## 3. Gate before launching

Do not start the lineage until all of the following hold. Each is cheap; the run is
not.

1. **Full test suite passes.** `PYTHONPATH=src python -m pytest tests/ -q`. Report
   exact counts. This is the primary guard against a mid-run failure.
2. **The three hashes are printed and recorded** — `source_tree_hash`,
   `model_source_hash`, `harness_hash`. The last two are the new identity basis.
3. **The two-tier hash actually works**: touching a file under `tests/` must leave
   `model_source_hash` unchanged; touching one under `src/thermoroute/` must change it.
4. **Stage-24's worker cap is right.** Confirm `_formal_worker_environment` exports
   `THERMOROUTE_FORMAL_THREADS` at the policy cap and that no thread variable is
   hardcoded to `"1"` anywhere under `scripts/`.
5. **Stage-19 runs.** Its degeneracy handling is what recovers Stage-19 and Stage-10.
6. **No stage script hardcodes a thread count.** `grep -rn '"1"' scripts/ | grep -i thread`.

## 4. Execution

Use the canonical runner, not a hand-written chain. `scripts/run_all.sh` exports the
full formal environment before every stage — exactly what the ops resume chain failed
to do — and runs the test suite as preflight.

```bash
cd /home/lzq/workspace/parttime/thermoroute-remediation
setsid nohup bash scripts/run_all.sh > outputs/logs/remediation_lineage.log 2>&1 &
```

19 steps: 09 → 09b → per-station LGB → 13 rigor → 13c folds → 13c assemble →
16 in-sample → 16 transfer → 16 report → 17 → 18 → **19 (now expected to pass)** →
20/15 → 22 → 23 → **10 + 12** → 25 → 24 → 27 + 14.

Expected wall clock, from the previous lineage: Stage-09 is the largest term;
09b ≈ 3 h, 16 ≈ 4.6 h, 25 ≈ 3.8 h. Total 1.5–2.5 days.

## 5. Supervision

Supervise; do not auto-restart from the top. The previous watchdog cost 5–7 hours by
re-running completed work, and would have looped forever on a deterministic failure.

- Watch for stage transitions and for `Traceback`, `Error`, `Killed`, `OOM`.
- On failure: **stop and diagnose.** Do not restart blindly. Stage scripts checkpoint
  internally, so a corrected relaunch resumes at the failed stage.
- Memory: Stage-09 previously OOM'd at ~24 GiB RSS. 125 GB total, ~86 GB free at
  launch. Watch RSS rather than assuming.

## 6. After the run

1. Confirm four completion receipts exist and each binds the new
   `model_source_sha256`.
2. Run 24 → 27 → 14 → 26 (`run_all.sh` covers 24/27/14; run 26 separately).
3. Merge the manuscript work per
   [`WORKTREE_MERGE_RUNBOOK_20260805.md`](WORKTREE_MERGE_RUNBOOK_20260805.md), adapted:
   the authoritative branch is now `feat/route-a-remediation`, and the gate is that
   `model_source_hash` — not the whole-tree hash — must be unchanged by the merge.
4. Only then proceed to M → I → G → C and the opening.
5. **Before the opening**, run the read-only degeneracy check against the
   target-period predictions (plan §3.3). It costs minutes and guards a step that
   cannot be repeated.

## 7. Retain the old lineage

Do not delete the multicore worktree's `outputs/` until the new lineage has produced
four validated receipts and a frozen model suite. It is the only fallback.

# Route-A numerical-policy amendment v2 (unified role cap)

Status: frozen pre-label governance record.  No 2021--2023 target temperature
was requested, acquired, inspected, scored, or inferred before this amendment
and its implementation were written.

## Why v2 replaces v1

Amendment v1 (route_a_numerical_policy_amendment_v1.json) declared 16 threads
for the mainline process and 2 threads for control member processes.  That
design cannot work: the Stage-09 `RunIdentity` embeds the numerical runtime
contract, which embeds the process thread cap, and every worker process
recomputes its own runtime contract and requires exact equality with the
parent identity (`stage09_parallel.py`, `assert_unchanged`).  A 2-thread
control member can never match a 16-thread parent, so the control closure
must fail closed.  v1 also contained two factual defects:

1. its rationale contradicted itself (first citing bit-exact cross-run
   reproducibility as the previous policy, then claiming cross-machine
   bit-exactness was never claimed);
2. it declared `prediction_replay_atol_zero`, while the implemented
   verifiers use 1e-12 for LightGBM shard parity and 1e-5 for neural
   sequence replay (ThermoRoute / LSTM / controls).

## What v2 freezes

| knob | value |
|---|---|
| per-process thread cap, every Stage-09/09b/16/25 role | 8 |
| cap source of truth | `protocols/route_a_numerical_policy_v2.json` |
| ambient `THERMOROUTE_FORMAL_THREADS` | must equal the frozen role cap; never defines it |
| parent and child roles of one run | one shared cap (identity consistency) |
| LightGBM shard parity | 1e-12 |
| neural sequence replay (ThermoRoute/LSTM/controls) | 1e-5 |
| probabilistic replay | 1e-6 |
| LightGBM prediction `num_threads` | 1 |
| torch deterministic algorithms / TF32 off / highest precision | unchanged |
| seeds, panel, split, architecture, claims | unchanged |

## Execution contract

Every script asserts its role cap against the policy document at startup
(`assert_role_thread_cap`), the Stage-09/09b authorizations bind
`member_process_thread_cap` and the policy-document digest, and worker
processes inherit the same cap as their parent so the runtime identity check
holds.  Child processes with a nonzero exit code fail the run immediately.

## Governance

- Supersedes the execution-ceiling portion of
  `route_a_native_thread_enforcement_notice_v1.md` and replaces amendment v1.
- v1 remains on the branch as a historical record of a superseded design;
  its defects are listed in v2's `supersedes.v1_defects`.
- Reproducibility level: fixed seed + pinned dependency versions +
  metric-level reproducibility; every within-run exactness check is retained.

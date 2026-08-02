# Route-A numerical-policy amendment v1 (multicore execution)

Status: frozen pre-label governance record.  No 2021--2023 target temperature
was requested, acquired, inspected, scored, or inferred before this amendment
and its implementation were written.

## What changed

The Route-A formal numerical policy previously forced exactly one native
thread per process (`route_a_native_thread_enforcement_notice_v1.md`) so that
a run could be bit-exact across machines and launch configurations.

Amendment v1 relaxes the execution ceiling only:

| knob | previous | amended |
|---|---|---|
| BLAS/OpenMP threads per process | 1 | process-declared cap via `THERMOROUTE_FORMAL_THREADS` (16 mainline, 2 control members) |
| Torch intra-op threads | 1 | same process-declared cap |
| `torch.use_deterministic_algorithms` | true | true (unchanged) |
| `torch.set_float32_matmul_precision("highest")`, TF32 off | required | required (unchanged) |
| `PYTHONHASHSEED` declaration + canonical-sort identity policy | required | required (unchanged) |
| LightGBM training `n_jobs` | 1 | process-declared cap; **bit-identical predictions verified** (n_jobs 1 vs 16 max abs diff = 0.0 on the run host) |
| LightGBM prediction `num_threads` | 1 | 1 (unchanged; replay-exact) |
| per-run internal parity / replay checks (`atol=0.0`, `atol=1e-12`) | required | required (unchanged) |

## What is preserved (scientific inputs are untouched)

Cohort, variables, panel bytes, station registry, time split, targets, model
architecture, loss, hyperparameters, seeds, calibration fits, estimand,
statistical decision rules, forecast keys, and claim wording are all
unchanged.  This amendment changes only the execution-evidence boundary:
reproducibility moves from "bit-exact across machines/configurations" to
"fixed seed + pinned dependency versions + metric-level reproducibility",
while every within-run exactness check (prediction replay, shard parity,
common-key registry) is retained and still fails closed.

## Determinism notes

- Within one run, every process declares its thread cap at import time
  (`THERMOROUTE_FORMAL_THREADS`); training and replay in the same lineage use
  the same cap, so internal bit-exact parity checks keep passing.
- LightGBM with `deterministic=True, force_col_wise=True` was verified
  bit-identical across thread counts on the run host (max abs prediction
  difference 0.0).
- Torch keeps deterministic algorithms on; cross-thread-count Torch results
  may differ in the last ULPs, which is the accepted metric-level boundary.

## Governance

- Supersedes only the execution-ceiling portion of
  `protocols/route_a_native_thread_enforcement_notice_v1.md`; that notice
  remains the historical record of the discovery that motivated the
  fail-closed pool assertion.
- The new lineage is content-addressed under the amended source tree; the
  Stage-09 run manifest records `formal_numerical_policy` with the amended
  thread caps, so the effective policy is machine-verifiable per run.

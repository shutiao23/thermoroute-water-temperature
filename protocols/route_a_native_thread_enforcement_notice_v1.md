# Route-A native-thread enforcement notice v1

Status: pre-label corrective engineering record.  No 2021--2023 target
temperature was requested, acquired, inspected, scored, or inferred before this
notice and its implementation were written.

Evidence boundary: this is a repository-internal honest-owner attestation backed
by the recorded local files and Git history.  It has no independent custodian,
external timestamp, or write-once proof, so it must not be represented as
independent evidence that no person accessed the target outcomes.

## Discovery

On 2026-07-24, after the sealed development rerun had started but before it
produced a canonical prediction artifact or completion receipt, adversarial
review found a gap between the declared and enforced numerical policy.

The controller environment set all five BLAS/OpenMP thread variables to one,
and the run manifest recorded every discovered native pool with
`num_threads=1`.  However, the Stage-09 entrypoint imported the Torch-only
`thermoroute.train.configure_deterministic_runtime` helper.  The formal
assertion checked environment strings and Torch state but did not fail closed
on the live `threadpoolctl.threadpool_info()` values.  A negative control could
therefore leave a BLAS pool at two threads while satisfying the old assertion.

## Disposition of the interrupted diagnostic run

- Git commit: `6d59e4218b84d117fa4391755a1fada05b766d79`
- run id: `aeef01f4ca1360c77438`
- source SHA-256: `5e0e134fc2e7a70ec43f9105ecaed301924e6841c91a1f09dc897955359e5043`
- run-manifest SHA-256:
  `c1219d450e9f4d7d41b1b7c1c5d03a8833d8d437720dc9fd3ab4f026a4420605`
- observed startup state: four discovered native pools, all reporting one
  thread
- completed cache state at interruption: 19 of 75 Stage-09 LightGBM model-head
  shards
- canonical predictions: absent
- Stage-09 completion receipt: absent

The run was interrupted deliberately and moved outside the active worktree.  It
is diagnostic evidence only.  It must not be resumed into, cited by, or frozen
as the formal Route-A model suite.  Its actual single-thread snapshot does not
retroactively supply the missing fail-closed enforcement.

## Corrective contract

The correction does not change the cohort, variables, time split, targets,
model architecture, loss, hyperparameters, seeds, calibration, estimand, or
statistical decision rules.  It changes only the execution-evidence boundary:

1. every formal numerical entrypoint uses
   `thermoroute.repro.configure_deterministic_runtime`, which retains a
   process-lifetime `threadpool_limits(1)` controller;
2. the formal assertion requires a non-empty live native-pool report;
3. every report record must be a BLAS or OpenMP pool and must expose the exact
   built-in integer `num_threads == 1`;
4. missing diagnostics, malformed records, unknown APIs, booleans, converted
   numeric types, or any value other than one fail closed;
5. formal training and replay entrypoints re-run the assertion after long
   numerical work and before canonical artifacts, component pointers, or
   completion receipts are published; and
6. transient thread counts remain excluded from stable native-binary identity.
   The run manifest still records the live snapshot as audit provenance.

All formal artifacts must therefore be regenerated under a new source hash and
content-addressed run id.  No artifact from the interrupted run is eligible for
cache reuse under the corrected source identity.

## Unrelated legacy-site semantics

This engineering correction has no bearing on the legacy `b1`, `s2`, and `p3`
files.  Those identifiers remain ordinary monitoring stations, not reservoirs
or a cascade, and imply no upstream/downstream, regulation, hydraulic
connection, or travel-time relationship.

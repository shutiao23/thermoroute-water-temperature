> Historical record (superseded by the conventional design).

# Stage-24 failure: the replay ran outside the declared numerical runtime

**Date:** 2026-08-05
**Status:** DIAGNOSED and FIXED at zero lineage cost (fix lives in `ops/`, which is not hashed).
**Symptom:** `ModelSuiteError: Stage-09b development-controls gate failed`, caused by
`DevelopmentControlsGateError: Stage-09b prediction differs from checkpoint best_model_state`.

---

## 1. What happened

At 20:03 UTC Stage-24 failed after about 40 minutes, while re-validating the
Stage-09b completion receipt:

```
File "src/thermoroute/development_controls_gate.py", line 963, in _validate_member_predictions
    raise DevelopmentControlsGateError(
DevelopmentControlsGateError: Stage-09b prediction differs from checkpoint best_model_state
```

The check is exact — no tolerance:

```python
digests[member] = prediction_content_digest(normalised)
if digests[member] != prediction_content_digest(replayed):
    raise DevelopmentControlsGateError(
        "Stage-09b prediction differs from checkpoint best_model_state")
```

## 2. Why this was not a scientific failure

The Stage-09b receipt records that the very same replay had already passed:

```json
"status": "PASS_STAGE09B_BEST_MODEL_STATE_PREDICTION_REPLAY",
"best_model_state_prediction_replay_verified": true,
"training_replay_verified": ...,
"run_identity": { "source_sha256": "bf9500aa…", "run_id": "e87f141ad92c33ce1d3f" }
```

and `source_sha256` equals the live tree hash. Same code, same weights, same data —
different answer. That points at process state, not at the science.

## 3. Root cause

Two facts combine.

**(a) The gate does not establish the runtime it depends on.**
`src/thermoroute/development_controls_gate.py` imports from `.repro` only hashing and
schema helpers:

```python
from .repro import (
    RUN_SCHEMA_VERSION, RunIdentity, atomic_write_json, formal_policy_document,
    sha256_file, sha256_json, sidecar_path, source_tree_hash, validate_artifact_sidecar,
)
```

`configure_deterministic_runtime` is **not** among them. That function is what pins
the thread caps, installs the `threadpoolctl` limiter for the process lifetime, and
sets `torch.use_deterministic_algorithms(True)`,
`torch.backends.cudnn.deterministic = True`, `benchmark = False`, tf32 off and
`torch.set_float32_matmul_precision("highest")`.

The gate therefore assumes its caller already configured the process. That assumption
holds when Stage-09b validates its own receipt in the process that just trained the
members. It fails when Stage-24 re-validates the same receipt from a fresh process.

**(b) The launcher supplied no environment at all.**
`scripts/run_all.sh` exports the formal block before every stage:

```bash
export THERMOROUTE_FORMAL_THREADS="${THERMOROUTE_FORMAL_THREADS:-8}"
export OMP_NUM_THREADS="$THERMOROUTE_FORMAL_THREADS"   # + MKL/OPENBLAS/VECLIB/NUMEXPR/WORKER_THREADS
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

The hand-written resume chain `ops/stage09/chain_stage09_remaining.sh` exported
**none** of it — measured: zero matching lines.

This matters more than it appears, because `thermoroute.repro` computes

```python
FORMAL_THREAD_LIMIT = _formal_thread_limit()      # at IMPORT time
```

from `THERMOROUTE_FORMAL_THREADS`, **falling back to 1**. Demonstrated directly:

```
RuntimeError: process thread cap 1 differs from the frozen policy cap 8 for role stage16
```

So Stage-24 replayed, single-threaded and non-deterministic, predictions that had
been produced with eight threads under the deterministic policy — violating the
frozen rule `"train_and_replay_share_process_cap": true` in
`protocols/route_a_numerical_policy_v2.json`. The forward pass differed in its last
bits, and the exact digest comparison rejected it.

## 4. The fix

Three new files under `ops/` — outside `DEFAULT_SOURCE_PATTERNS`, so
`source_tree_hash` is unchanged (verified: still
`bf9500aaf549de02e6abc0fe471147ef66903205ad1450f0a2285579e32539e9`).

### `ops/stage09/run_stage_formal.py`

Successor to `run_fullstack.py`. In order:

1. reads the uniform role cap from `protocols/route_a_numerical_policy_v2.json`
   using the standard library only, and exports the thread environment —
   **before** importing `thermoroute`, because `FORMAL_THREAD_LIMIT` is frozen at
   import time;
2. executes the Stage-16 module graph, reproducing the training processes' native
   library set so the runtime contract hash matches (this is what `run_fullstack`
   already did, and it is still required);
3. calls `configure_deterministic_runtime()` and verifies the resolved cap;
4. runs the requested stage as `__main__`.

Verified:

```
[run_stage_formal] threads=8 torch_threads=8 deterministic=True matmul=highest
Route-A artifact boundary OK
```

### `ops/stage09/chain_route_a_finish.sh`

Runs 24 → 27 → 14 → 26 through that runner, exports the full `run_all.sh`
environment block, records each completed step so a restart resumes instead of
repeating, and emits `ROUTE_A_FINISH_CHAIN_COMPLETE`.

### `ops/stage09/monitor_route_a.sh`

Resumes rather than restarting from the top; determines liveness from process state
and the chain's own completion marker; refuses to start if the chain script is not
executable; and **stops after the same step fails twice**, on the ground that a
deterministic failure is not a transient one.

## 5. This is conformance, not circumvention

The frozen policy requires exactly what the fix establishes: a uniform role cap of 8,
`torch_deterministic_algorithms: true`, `float32_matmul_precision: highest`, tf32 off,
and `train_and_replay_share_process_cap: true`. The launcher was not meeting its own
declared contract. No check was weakened, no tolerance introduced, no receipt edited.

## 6. Proper remediation, deferred

The durable fix belongs in the source: `_validate_member_predictions` should call
`configure_deterministic_runtime()` itself (or assert it is active) rather than
depending on an undocumented caller precondition. That is a `src/` edit and costs a
full lineage, so it is queued as **R7** in
[`SOURCE_HASH_SCOPE_REMEDIATION_PLAN.md`](SOURCE_HASH_SCOPE_REMEDIATION_PLAN.md).

Note also that `24_freeze_model_suite.py` never calls `assert_role_thread_cap`, while
Stage-16 does — which is why the mismatch surfaced as an obscure digest failure
instead of a clear thread-cap error. Add that assertion in the same lineage.

## 7. General lesson

This is the third defect of the same shape:

| # | Symptom | Cause |
|---|---|---|
| 1 | Stage-18 runtime contract failure | import order changed the loaded native-library set |
| 2 | `Stage-16 identity differs from the current numerical runtime` | same |
| 3 | this one | required global torch/thread state never established |

Every one of them is a stage depending on **implicit process state** that its caller
happened to provide. A contract that binds the runtime must also *establish* the
runtime; verifying a precondition that nothing sets is how a correct result gets
rejected.

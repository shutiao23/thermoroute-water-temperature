# Stage-24 worker thread cap: the fifth hardcoded "1", and why the freeze cannot hold

**Date:** 2026-08-05
**Status:** BLOCKER — requires a source edit, therefore a lineage. Decision required.
**Supersedes the ops-only diagnosis in** [`STAGE24_REPLAY_RUNTIME_DEFECT_20260805.md`](STAGE24_REPLAY_RUNTIME_DEFECT_20260805.md) §3, which identified the launcher half of the problem but not the worker half.

---

## 1. The defect

`scripts/24_freeze_model_suite.py` builds a complete, allowlisted environment for its
worker process and passes it with `env=environment`, replacing the parent environment
entirely:

```python
def _formal_worker_environment(cache: Path, nonce: str) -> dict[str, str]:
    """Return the complete allowlisted Stage-24 worker environment."""
    return {
        "PATH": os.defpath, "LANG": "C", "LC_ALL": "C", "TZ": "UTC",
        "TMPDIR": str(cache.resolve()),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
        ...
    }
```

Every thread variable is hardcoded to `"1"`, and **`THERMOROUTE_FORMAL_THREADS` is
absent**. Consequently, in the worker:

```python
# src/thermoroute/repro.py
raw = os.environ.get("THERMOROUTE_FORMAL_THREADS") or "1"   # -> 1
FORMAL_THREAD_LIMIT = _formal_thread_limit()                # frozen at import
```

and `configure_deterministic_runtime()` — which Stage-24 does call, at module level —
faithfully pins the worker to **one thread**.

## 2. Why that breaks the freeze

The Stage-09b members were trained and self-verified at **eight** threads:

```python
# scripts/09b_development_controls.py
STAGE09B_MEMBER_THREADS = int(os.environ.get("THERMOROUTE_FORMAL_THREADS") or "8")
```

The frozen policy requires the two to agree:

```json
"every_process_in_a_role_uses_the_exact_role_cap": true,
"parent_and_child_roles_of_one_run_share_one_cap": true,
"train_and_replay_share_process_cap": true,
"role_thread_caps": {"stage09": 8, "stage09b": 8, "stage16": 8, "stage25": 8}
```

Stage-24 replays at 1 what was produced at 8. The forward pass differs in its last
bits, and the comparison is an exact content digest with no tolerance:

```python
if digests[member] != prediction_content_digest(replayed):
    raise DevelopmentControlsGateError(
        "Stage-09b prediction differs from checkpoint best_model_state")
```

**The code that enforces the thread-cap policy is itself violating it.**

## 3. Three different defaults for one policy value

| Site | Default when `THERMOROUTE_FORMAL_THREADS` is unset |
|---|---|
| `src/thermoroute/repro.py::_formal_thread_limit` | **1** |
| `scripts/09b_development_controls.py` | **8** |
| `scripts/16_lstm_baseline.py` | **8** |
| `scripts/24_freeze_model_suite.py` worker allowlist | **1**, hardcoded, variable omitted |

The frozen policy document is the only authority, and none of these four sites reads
it for its default.

## 4. Why it went undetected

Neither `scripts/24_freeze_model_suite.py` nor
`src/thermoroute/development_controls_gate.py` calls `assert_role_thread_cap` —
measured: **0 occurrences in each**. `src/thermoroute/model_suite.py` calls it once.

Had Stage-24 asserted its role cap the way Stage-16 does, this would have failed in
under a second with

```
process thread cap 1 differs from the frozen policy cap 8 for role stage24
```

Instead it surfaced after roughly forty minutes as an opaque prediction-digest
mismatch that reads like a data-integrity failure.

This is the **fifth** hardcoded `1` of the same family, after the `n_jobs`, 09b gate,
bridge-report and release-verifier instances. It had not been found because Stage-24
had never previously run far enough to reach the Stage-09b re-validation.

## 5. Why the ops-level fix cannot reach it

`ops/stage09/run_stage_formal.py` correctly establishes the runtime in the **parent**
(verified: `threads=8 torch_threads=8 deterministic=True matmul=highest`). The worker
discards the parent environment wholesale via `env=environment`. The dictionary is
constructed inside a hashed source file. There is no environment passthrough, no
policy lookup, and no override hook.

**No change under `ops/` can fix this.**

## 6. Consequence: the freeze cannot hold

[`CODE_FREEZE_DISCIPLINE_20260805.md`](CODE_FREEZE_DISCIPLINE_20260805.md) assumed the
development chain could reach a frozen model suite without further source edits. That
assumption is false. Stage-24 cannot complete, so:

- no frozen model suite,
- therefore no M→I→G→C chain,
- therefore no opening,
- therefore no paper results.

A source edit is unavoidable, and any source edit costs one full lineage.

## 7. Recommendation: spend one lineage, and bundle everything into it

Since a lineage must be spent, spend it exactly once and land the entire deferred
queue from
[`SOURCE_HASH_SCOPE_REMEDIATION_PLAN.md`](SOURCE_HASH_SCOPE_REMEDIATION_PLAN.md)
together:

| ID | Fix | Effect |
|---|---|---|
| **R8** | `_formal_worker_environment` reads the policy cap and exports `THERMOROUTE_FORMAL_THREADS`; BLAS vars set to the cap | unblocks Stage-24 — *required* |
| **R7** | Stage-24 and the 09b gate call `assert_role_thread_cap`; the gate establishes (or asserts) the deterministic runtime instead of assuming it | this class of defect fails in one second, not forty minutes |
| **R1** | Two-tier hash: `model_source_sha256` gates, `harness_sha256` records | future test/CI/ops/manuscript edits cost nothing |
| **R2** | Stage-19 `q05 >= q95` → `q05 > q95`, degeneracy count recorded | **recovers Stage-19 and Stage-10**, i.e. the probabilistic suite, Fig 3 and Fig S5 |
| **R4** | Thread cap out of the runtime identity contract; recorded, not binding | parallelism becomes tunable without a retrain |
| **R6** | Claim registry pins claim blocks, not the whole-manuscript sha256 | the manuscript can be edited without invalidating the registry |
| **R3** | Route-B protocol seal (optional) | preserves the Route-B option at no extra cost |
| **R5** | Partition prediction tables by model/scope | the 70–90 min diagnostics drop to minutes |

Estimated cost: roughly 1.5–2.5 days of wall clock for Stage-09 → 09b → 16 → 25 plus
the diagnostics and the freeze chain, based on the previous lineage
(09b ≈ 3 h, 16 ≈ 4.6 h, 25 ≈ 3.8 h, 09 the largest term).

What it buys, permanently:

- Stage-24 completes and the model suite freezes;
- Stage-19 and Stage-10 come back, so the manuscript keeps its probabilistic
  content and both figures;
- the retrain tax on tests, CI, ops and the manuscript goes to zero;
- the next occurrence of this defect class fails immediately with a clear message.

**Do not apply R8 alone.** Fixing only the blocker spends the same lineage and leaves
every other item still owed — which is how five lineages were spent already.

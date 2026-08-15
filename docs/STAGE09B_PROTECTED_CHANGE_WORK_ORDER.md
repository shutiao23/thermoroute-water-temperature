> Historical record (superseded by the conventional design).

# Stage09b protected change work order

## 1. Purpose, scope, and current state

This is a **pre-authorization, outcome-free change work order**.  It records
the only compliant repair for the Stage09b configuration-shape failure
documented in
[`STAGE09B_CONFIG_SHAPE_FAILURE_20260802.md`](STAGE09B_CONFIG_SHAPE_FAILURE_20260802.md).
It is neither a protocol amendment, a completion receipt, nor permission to
start a run.

| Item | Current value |
| --- | --- |
| Completed Stage09 parent | `7cb2bfb18c1f9aa3dba7` |
| Parent source SHA-256 | `19289553aa0929bdb5803a8a3eaa96b38a651b3d52da441fb6298f2ac9228b55` |
| Parent receipt file SHA-256 | `07a0dd1e54cfcc96179c8adaffe2d987776c210e27972faafca90cec6b12f111` |
| Failed Stage09b attempt | `a930214d93fb7bdca83e` |
| Outcomes | No 2021–2023 outcome was requested or read |
| Post-fix source SHA-256 | `0e932f19975033ef0749d2589b60c6aefe057adbef1ee1d9e2f75bef8180920f` (computed 2026-08-02 after the allowlisted repair and T1–T5 regressions passed) |

The repair fixes an in-memory Python container mismatch only.  It must not
change the declared nine arms, their order, their variables, their five seeds,
the 45-member registry, scientific training behavior, model-matrix amendment,
or the evidence role (`development_only_exploratory`).

## 2. Authorization prerequisites

Work may begin only after the project owner explicitly authorizes **all** of the
following:

1. modification of the protected production script and the minimal tests in
   the allowlist below;
2. creation of a new `source_tree_hash` and therefore a new content-addressed
   Stage09 run identity;
3. a new guarded Stage09 execution under that identity, with reuse limited to
   caches that the guarded entrypoint itself validates as eligible;
4. replacement of the canonical Stage09 pointer/receipt only through the
   guarded Stage09 publication transaction; and
5. re-arming the existing watcher only after the new Stage09 receipt has been
   independently validated.

The authorization must not be inferred from permission to inspect, document,
or retry the prior source identity.  It must also state that this work order
does **not** authorize target acquisition, opening, reading post-2020 water
temperature outcomes, changing the Route A/B inferential design, or making a
performance claim.

Preflight before editing must record that no Stage09, Stage09b, or watcher
process is live; that the formal locks are free; and that the old Stage09
receipt and the failed `a930...` directory are left untouched.  A live process
or a contested lock is a stop condition, not an invitation to kill, remove, or
rewrite it.

## 3. File allowlist and explicit prohibitions

### 3.1 Protected source-change allowlist

Only these files may be changed for this defect:

| File | Permitted change |
| --- | --- |
| `scripts/09b_development_controls.py` | Canonicalize each arm descriptor at the parent `run_config["arms"]` construction, before `resolve_run_identity()` and before Stage09b plan freeze. |
| `tests/test_development_controls.py` | Add the direct production-script regression for the canonical arm descriptor, using the existing importlib loader. |
| `tests/test_stage09b_precompute.py` | Remove the fixture's JSON-round-trip masking behavior and add the freeze/worker-equality and tuple-rejection regressions. |

All three paths are included by `thermoroute.repro.DEFAULT_SOURCE_PATTERNS`.
They are therefore a single source-boundary change, not a test-only change.

### 3.2 Permitted runtime binding action after source acceptance

After the new source digest is known, the existing local watcher environment
may be rebound for execution only:

`ops/stage09/phase2_watch.env`

Set `EXPECTED_SOURCE_SHA256` to the computed new digest.  Leave
`EXPECTED_STAGE09_RUN_ID` empty while the new Stage09 run is being created;
write its actual new run ID only after its receipt passes verification.  This
is a local runtime pin, not a substitute for, or an edit to, a source seal.
It must not be used to change worker semantics, bypass receipt validation, or
promote `7cb2...` under the new source.

### 3.3 Prohibited changes and workarounds

The following are outside this work order:

- `src/thermoroute/stage09b_precompute.py`, including weakening
  `_validate_formal_config()` or allowing tuple-shaped descriptors;
- every protocol, amendment, seal, comparator registry, model-matrix document,
  scientific configuration, source-hash algorithm, or watcher/launcher source
  file;
- outputs, run manifests, receipts, pointers, work orders, checkpoints,
  prediction files, locks, or cache sidecars; none may be hand-created,
  altered, deleted, moved, or relabelled;
- monkeypatches, `sitecustomize`, wrapper scripts, environment injection,
  alternate JSON manifests, hidden-member CLI invocation, or a direct Python
  Stage09/09b launch that is not the guarded sequence below;
- any target-period request, read, normalization, QC, scoring, opening, or
  result/claim editing; and
- unrelated refactoring, formatting sweeps, dependency changes, or adjustments
  to the nine-arm/five-seed contract.

In particular, `a930214d93fb7bdca83e` has no authorization/work-order/member
lineage and is not resumable.  It must remain a non-authoritative failure
record.

## 4. Required implementation semantics

The current production construction is at
`scripts/09b_development_controls.py:1751`:

```python
"arms": [asdict(arm) for arm in arms],
```

`ArmSpec.variables` and `ArmSpec.seeds` are deliberately tuples in
`src/thermoroute/development_controls.py`; `dataclasses.asdict()` preserves
those nested tuples.  In contrast,
`src/thermoroute/stage09b_precompute.py:_validate_formal_config()` requires
`variables` to be a non-empty Python `list` and `seeds` to equal the frozen
five-seed Python list.  The validator is correct: it protects equality among
the run manifest, authorization, and worker work orders.

Replace only the parent descriptor construction with an explicit JSON-native
representation, conceptually:

```python
"arms": [
    {
        **asdict(arm),
        "variables": list(arm.variables),
        "seeds": list(arm.seeds),
    }
    for arm in arms
],
```

A tiny private helper in the same script is acceptable only if it makes this
exact mapping directly testable.  It must return the same five keys as
`ArmSpec`, must have no I/O, and must be used by the live `run_config` path.
Do not use a JSON serialize/deserialize round trip as the production repair:
the canonical shape must exist before identity calculation and plan freeze.

The intended invariant is:

```text
immutable ArmSpec tuples (internal implementation)
    -> explicit JSON-native descriptor (resolved configuration)
    -> RunIdentity/config SHA -> run.json -> authorization -> work order
```

Only container type changes at the resolved-config boundary.  Values and
ordering remain byte-for-byte canonical after JSON serialization.

### 4.1 Applied repair (2026-08-02)

The applied patch implements the descriptor helper exactly as specified above
and also normalizes the nested `time_split` field, which the same defect class
affects: `C.SPLIT.as_dict()` returns tuple-valued ranges, and the JSON-native
run manifest written by `initialise_run_directory` therefore disagrees with the
live tuple-shaped config in `_validate_run_manifest`, failing with
"Stage-09b run manifest differs" even after the arm descriptors are canonical.
This is the same tuple-to-list normalization requirement captured by the
T3 in-memory/JSON parity acceptance row and stays inside the same
`scripts/09b_development_controls.py` construction boundary.  No validator,
protocol, scientific value, or model-matrix document was changed.

## 5. Required regression matrix

The code change is not accepted merely because `run.json` happens to serialize
tuples as arrays.  The focused tests must cover these cases before any guarded
experiment starts.

| ID | Case | Required assertion |
| --- | --- | --- |
| T1 | Production descriptor construction | All nine descriptors built from `declared_arms()` have exactly `arm_id`, `family`, `feature_set`, `variables`, and `seeds`; both sequence fields are lists. |
| T2 | Registry preservation | Arm IDs/order, variable values/order, seed values/order, and the expected nine-by-five member registry equal the frozen declarations exactly. |
| T3 | In-memory/JSON parity | The live resolved configuration already equals its canonical JSON round trip; no test fixture may use that round trip to repair an otherwise invalid input. |
| T4 | Parent-to-worker closure | `freeze_stage09b_precompute_plan()` succeeds from the live-shape configuration, and each generated work order validates against the same identity and expected configuration. |
| T5 | Fail-closed negative control | Replacing a descriptor's `variables` or `seeds` with a tuple causes `_validate_formal_config()`/plan freeze to raise `Stage09bPrecomputeError` with the arm/seed-contract failure. |

At minimum run and retain the output of:

```text
python -m pytest -q tests/test_development_controls.py tests/test_stage09b_precompute.py
```

Then run the repository's stationary full quality/test gates under the normal
formal runtime, recording the exact commands, revisions, and results.  Any
failure requires diagnosis before proceeding; a narrowed test pass cannot
waive a failing full gate.

## 6. New source, receipt, and run-identity consequences

`source_tree_hash()` inventories `scripts/**/*.py` and `tests/**/*.py` as well
as `src`, protocols, and selected project configuration.  The allowed patch
therefore makes the present source digest
`19289553aa0929bdb5803a8a3eaa96b38a651b3d52da441fb6298f2ac9228b55` stale for
new formal promotion.

Consequences that must be accepted before execution:

1. Stage09 run `7cb2bfb18c1f9aa3dba7` and its receipt SHA-256
   `07a0dd1e54cfcc96179c8adaffe2d987776c210e27972faafca90cec6b12f111` remain
   accurate historical evidence for the old source only.  They cannot parent
   Phase 2 under the repaired source.
2. The new Stage09 `RunIdentity` must be recomputed by the guarded entrypoint;
   its `source_sha256`, config binding, and hence content-addressed `run_id`
   must not be assumed to equal `7cb2...`.
3. The canonical `outputs/models/route_a_stage09_completion.json` path is a
   moving, guarded publication pointer.  Do not manually preserve, delete,
   rename, or overwrite it.  A successful new Stage09 may atomically replace
   it, along with the guarded component pointers, only after validating its
   own complete closure.  This work order and the failure record preserve the
   old receipt fingerprint and lineage boundary.
4. Caches can be reused only when the guarded Stage09 code validates their
   source/identity provenance.  No old `7cb2...`, void `bb02498...`, or failed
   `a930...` artifact may be imported, promoted, or claimed by hand.
5. The model-matrix amendment/seal is unchanged by this shape repair.  It must
   nevertheless be revalidated as part of the new guarded lineage; no
   scientific seal is amended by this work order.

## 7. Authorized execution sequence after a valid approval

The order below is intentional.  A watcher bound to the new source would
reject the still-present old-source canonical receipt if it were started first.

1. **Quiescent preflight.** Confirm no relevant process or flock holder is
   live, record the old receipt and failed-09b fingerprints, and record the
   exact starting source identity.  Do not alter outputs.
2. **Patch and tests only.** Apply the narrow allowed patch.  Review the diff
   against the allowlist, run T1–T5 and the full stationary gates, then compute
   and record the new `source_tree_hash` from the final tested tree.
3. **Bind the local launcher environment.** Set only
   `EXPECTED_SOURCE_SHA256=<new digest>` in the existing local watcher env;
   clear its old `EXPECTED_STAGE09_RUN_ID=7cb2...`.  Keep the watcher stopped.
4. **Run new guarded Stage09.** Invoke only
   `ops/stage09/start_stage09.sh` through the existing guarded entrypoint with
   the approved resource settings.  It must preflight the new source and
   create a new content-addressed run.  It alone may decide whether any cache
   is eligible and, if complete, atomically publish the new canonical Stage09
   pointers/receipt.
5. **Independently validate the new parent.** After Stage09 exits, validate the
   new receipt using the repository validator plus its file/self hashes, new
   `run_id`, new source binding, exact component pointers, and the false
   `confirmation_outcomes_requested_or_read` flag.  A mere log line or member
   count is insufficient.
6. **Bind and start the existing watcher.** Put the verified new run ID into
   `EXPECTED_STAGE09_RUN_ID`, retain the new source pin, then start only the
   existing `ops/stage09/start_phase2_watch.sh`/approved watcher entrypoint.
   The watcher must independently validate the new receipt before it launches
   Stage09b.
7. **Let guarded Stage09b create a new plan.** The watcher launches 09b as its
   normal first Phase-2 action.  It must create a new precompute authorization,
   exactly 45 work orders, and a new Stage09b run/receipt.  Do not reuse
   `a930...`, manufacture work orders, or invoke a member worker directly.
8. **Continue only on receipt evidence.** Phase 2 can continue under the
   watcher only after a fresh Stage09b completion receipt validates.  All
   resulting values remain development-only and cannot authorize target
   outcomes or publication claims.

## 8. Stop, rollback, and incident handling

Stop immediately and report the evidence if any of these occurs:

- a diff touches a path outside the allowlist, changes the nine-arm/five-seed
  values, or requires weakening the Stage09b validator;
- a focused or full test fails, the final source hash changes after testing, or
  the source/runtime preflight does not match its pin;
- a cache fails eligibility validation, a source-bound receipt is stale, or an
  old run is offered as a substitute for the new run;
- Stage09 cannot complete and validate a new receipt, its canonical pointers
  do not bind to the new identity, or its outcome-read flag is not false;
- watcher validation fails, Stage09b again raises an arm/seed contract error,
  45 work orders are not created, or a conflicting process/lock appears; or
- any action would require outcomes, manual output edits, or a protocol/model
  matrix change.

Before a new Stage09 is started, an abandoned code patch may be reversed only
by a reviewable inverse patch limited to the same allowlist and then retested.
Do not use `git reset --hard`, broad checkout/restore commands, or deletion of
run/output evidence.  Once a new guarded Stage09 has started, do not modify
the protected source or pins in place: stop, retain the evidence, and obtain a
new work order for the next source identity.

## 9. Acceptance evidence bundle

The repair is accepted only when all items below are present and mutually
consistent:

1. explicit user approval matching Section 10;
2. a reviewed, whitespace-clean diff limited to the allowlist, including no
   unapproved scientific/protocol/output changes;
3. retained outputs for T1–T5 and the full stationary gates, with the final
   tested source inventory/hash recorded after—not before—the tests;
4. the guarded Stage09 launcher log showing the expected new source digest and
   the newly computed run ID;
5. a new `PASS_FORMAL_STAGE09_COMPLETE` receipt that independently validates
   against the live new source, its run manifest, predictions, sidecar,
   components pointer, and receipt self-hash, while declaring no outcome read;
6. watcher evidence that it validated that exact new receipt and then launched
   09b; and
7. a fresh `PASS` Stage09b completion receipt with nine arms, five seeds per
   arm, exactly 45 member receipts/work orders, matching authorization/config
   hashes, and no target-outcome access.

The old Stage09 receipt hash and the failed `a930...` run must remain listed as
historical/non-promotable evidence in the acceptance record.  Passing any
subset does not authorize target opening or change Route A's descriptive-only
inference boundary.

## 10. User authorization statement template

> I authorize the narrowly scoped protected change in
> `scripts/09b_development_controls.py` and the listed minimal regressions in
> `tests/test_development_controls.py` and
> `tests/test_stage09b_precompute.py`, solely to canonicalize Stage09b arm
> `variables` and `seeds` from tuples to lists before run-identity calculation
> and plan freeze.  I accept that this creates a new source hash and requires a
> new guarded Stage09 run and a new Stage09b lineage; the old `7cb2...` receipt
> remains historical only and must not be promoted.  I authorize reuse only of
> caches accepted by the guarded entrypoint, followed by independent receipt
> validation and then the existing watcher.  I do not authorize protocol or
> model-matrix changes, manual output/receipt/work-order edits, direct worker
> launches, or any 2021–2023 outcome request, read, opening, scoring, or claim.

## 11. Authorization record

**Authorization #1 — 2026-08-02 (14:23 UTC+01:00).** The project owner provided the
following explicit authorization, which satisfies the prerequisites in
Section 2 (original wording, for source `a4b174e1...`):

> I authorize running a new Stage-09 under the new source `a4b174e1...`,
> accepting the old `7cb2...` receipt as historical evidence only and not
> promoting it across hashes; reuse only caches validated as eligible by the
> guarded entrypoint; no opening authorization; no protocol changes; no manual
> receipt construction.

**Authorization #2 — 2026-08-02 (16:40 UTC+01:00) — 96-core throughput change.**
The project owner chose to raise the execution-only worker caps to use the full target
host (128-core Intel Xeon Gold 6430). This changes `MAX_CONTROL_WORKERS` in
`src/thermoroute/stage09_parallel.py` (8 → 96) and `MAX_PARALLEL_WORKERS` in
`src/thermoroute/stage09b_precompute.py` (8 → 96), both inside the source-hash
boundary, plus the matching ops launcher clamps. The project owner authorized this
change and the resulting new source identity. Worker counts remain
execution-only parameters that never enter RunIdentity; determinism gates
(single-threaded members, OMP_NUM_THREADS=1) are unchanged.

Authorized scope, bound to this work order and to
`docs/RUN_PACKAGE_20260802/README.md`:

1. The new source identity is `0e932f19975033ef0749d2589b60c6aefe057adbef1ee1d9e2f75bef8180920f`
   (computed 2026-08-02 after the worker-cap change; verified by
   `preflight_env.sh` on the target host).
2. The target host environment is `route-a` (conda, Python 3.12.13) with
   `torch 2.12.0+cpu` and lock-aligned dependencies; full focused regression
   passed (`tests/test_stage09_parallel.py`, `tests/test_stage09b_precompute.py`,
   `tests/test_development_controls.py`, `tests/test_stage09_completion.py`).
3. Execution sequence per Section 7: quiescent preflight → guarded Stage-09
   launcher → independent receipt validation → watcher binding → Phase-2.
4. Not authorized by this record: target-period (2021–2023) acquisition or
   opening; protocol/model-matrix amendment; manual construction or editing of
   receipts, work orders, or manifests; direct member-worker invocation;
   promotion of `7cb2...`, `bb02498a...`, or `a930...` artifacts.


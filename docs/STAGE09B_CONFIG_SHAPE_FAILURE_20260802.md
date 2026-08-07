> Historical record (superseded by the conventional design).

# Stage09b config-shape failure — evidence and recovery boundary

| Field | Value |
| --- | --- |
| Date | 2026-08-02 (Europe/London) |
| Evidence role | Outcome-free engineering failure record; not a completion receipt or protocol amendment |
| Completed parent | Stage09 run `7cb2bfb18c1f9aa3dba7` |
| Parent source identity | `19289553aa0929bdb5803a8a3eaa96b38a651b3d52da441fb6298f2ac9228b55` |
| Parent receipt file SHA-256 | `07a0dd1e54cfcc96179c8adaffe2d987776c210e27972faafca90cec6b12f111` |
| Failed child attempt | Stage09b run `a930214d93fb7bdca83e` |
| Outcome boundary | No 2021–2023 outcome was requested or read |

## 1. What happened

The existing watcher independently validated the Stage09 completion receipt and
then invoked the guarded Phase-2 chain. Stage09b stopped while freezing its
parent precompute plan, before any member process was authorized or trained.
The exception was:

```text
Stage09bPrecomputeError: Stage-09b arm/seed contract changed
ControlExperimentError: Stage-09b member precompute failed
```

The failed run directory contains `run.json` only. It has no authorization,
work order, member receipt, prediction/checkpoint cache or Stage09b completion
receipt. The watcher and formal locks released normally, and no Stage09,
Stage09b or watcher process remained after the failure.

## 2. Exact root cause

`scripts/09b_development_controls.py` constructs `run_config["arms"]` with
`dataclasses.asdict(arm)`. `ArmSpec.variables` and `ArmSpec.seeds` are tuples,
and `asdict()` preserves those nested tuple types in the live Python object.

`src/thermoroute/stage09b_precompute.py::_validate_formal_config()` requires:

- `variables` to be a JSON-list-shaped Python `list`; and
- `seeds` to equal the frozen five-seed list.

The arm identifiers, order, variables, seed values and nine-by-five model matrix
did not change. The failure is a container-shape mismatch. `run.json` looks
canonical because JSON serialization converts tuples to arrays after run
identity construction; the freeze validator receives the unnormalized in-memory
object and rejects it first.

The unit fixture in `tests/test_stage09b_precompute.py` performs a JSON
round-trip before freeze validation, unintentionally normalizing the tuples and
masking the production-path defect.

## 3. Why retrying the current source is not legitimate

Changing worker count, retrying the watcher or invoking the hidden member CLI
does not normalize the live parent config. No work orders exist to resume.
Manually manufacturing JSON authorization/work-order files would also fail the
worker's equality check against the tuple-shaped live config and would violate
the guarded provenance chain.

Monkeypatching, `sitecustomize`, an external shim or a hand-edited run manifest
would not be bound by the current source/runtime identity. Those are not
compliant recovery paths. The failed `a930...` attempt must remain a
non-authoritative failure record.

## 4. Minimal source repair (not yet authorized)

The production config constructor must emit canonical list shapes before run
identity and freeze validation, conceptually:

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

Do not merely loosen the validator: a later run-manifest/authorization equality
check would still compare JSON lists with live tuples.

Minimum regression evidence must prove:

1. the real production config built from `declared_arms()` passes formal
   validation without a JSON round-trip;
2. every arm has list-shaped `variables` and `seeds`, and the frozen arm/seed
   registry is unchanged;
3. the in-memory config is identical to its canonical JSON round-trip;
4. parent freeze and worker authorization equality both pass; and
5. deliberately injected tuple-shaped noncanonical configs still fail closed.

## 5. Source-identity consequence

The repair touches `scripts/**/*.py`, which is part of `source_tree_hash`.
Therefore it necessarily creates a new source identity. The completed Stage09
receipt for `7cb2...` remains valid historical evidence for
`19289553...`, but cannot be promoted or relabelled as the parent of Phase 2
under the repaired source.

The amendment/seal scientific matrix need not change merely because the
serialization shape is corrected. Nevertheless, the new source identity and
content-addressed run IDs must be derived and verified normally.

## 6. Authorized recovery sequence

No step below may begin until the user explicitly authorizes the protected
source edit and the resulting new lineage.

1. Apply only the canonical list-shape repair and its regression tests.
2. Run focused tests, then the stationary full quality/test gates.
3. compute and record the new source identity; preserve all previous receipts
   and failure directories without cross-hash promotion;
4. update the guarded Stage09/watcher pinning for that exact identity;
5. rerun Stage09 through the guarded entrypoint, reusing only caches the entrypoint
   validates as eligible;
6. independently verify the new Stage09 completion receipt;
7. restart the bound watcher and let it create a new Stage09b plan/run; and
8. continue Phase 2 only after the new Stage09b completion receipt validates.

This record does not authorize target acquisition, opening, protocol changes,
manual artifact construction or any claim from development scores.

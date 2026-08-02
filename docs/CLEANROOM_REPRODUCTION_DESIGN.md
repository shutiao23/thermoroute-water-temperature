# P0-R0 clean-room reproduction design

| Field | Value |
| --- | --- |
| Date | 2026-08-01 |
| Status | **DESIGN ONLY / NO HASHED-PATH CHANGE / NO TEST EXECUTION** |
| Stage-09 constraint | Implementation is deferred until run `7cb2bfb18c1f9aa3dba7` has a validated completion receipt, or the user explicitly abandons it and releases the boundary through the governed watcher/lock procedure; no related formal process may be running. |
| Goal | A fresh, fixed Linux environment can validate the authorized evidence chain and reproduce the paper without reading or writing a live experiment namespace. |

## 1. Current capability and gap

The repository already has useful pieces:

- a Python 3.12 fully hashed requirements file;
- a GitHub Actions `locked` / `latest` matrix;
- full pytest, ruff, mypy, provenance, release, wheel-import and synthetic-smoke
  steps;
- a PRE Markdown-to-AGU-TeX freshness check;
- run identity that binds the source inventory and dependency artifacts.

Those pieces do not yet prove P0-R0:

1. The full test suite has not passed in a stationary artifact namespace after
   the current source change.
2. The `--help` entrypoint test snapshots the real repository `outputs` tree,
   so concurrent experiment writes create false failures.
3. The snapshot omits some output namespaces, so its “zero artifact output” name
   is stronger than its coverage.
4. The wheel smoke venv uses `--system-site-packages`, which can hide a missing
   wheel dependency.
5. CI does not run `pip check` and does not prove that the locked graph installs
   as wheels on the target Linux image.
6. There is no container or equivalent fixed image receipt, no clean-room
   archive E2E, no PDF compilation check, no locked DOCX runtime, no POST paper
   renderer and no `make reproduce-paper` entrypoint.

## 2. Artifact-namespace isolation patch

The first post-Stage-09 patch should be deliberately narrow and limited to the
formal-entrypoint tests.

### Proposed test behavior

1. Create `tmp_path / "cleanroom"`.
2. Copy only `src/` and `scripts/` into it, excluding `__pycache__` and `*.pyc`.
3. Invoke every registered `--help` entrypoint from the temporary root with the
   temporary script path and `PYTHONPATH=<cleanroom>/src`.
4. Snapshot the complete `<cleanroom>/outputs/**` namespace before and after,
   including paths, file type, size and high-resolution mtime.
5. Assert that the before and after snapshots are identical.
6. Add a unit test that writes a sentinel under `outputs/reports/` in a fixture
   and proves that the snapshot helper detects it. This prevents another partial
   namespace from being labelled complete.

### Non-goals

- Do not make a live experiment tree “stable” by ignoring its writes.
- Do not monkeypatch the production output root.
- Do not copy real `outputs/`, data, model bundles or receipts into the unit-test
  clean room.
- Do not claim the old concurrent failures were product defects; they tested a
  mutable shared namespace.

### Verification order after the source boundary is released

1. Run only the modified formal-entrypoint tests.
2. Run the full suite with no Stage 09/09b/16/25 process and archive the test
   receipt.
3. Run ruff and mypy.
4. Recompute the source hash and record that all subsequent model work uses the
   new identity; never relabel the old Stage-09 receipt as belonging to it.

## 3. Fixed Linux environment

Choose one supported image first; cross-platform claims are optional and must be
separate.

### Required properties

- immutable base-image digest, not a floating tag;
- Python 3.12 patch version recorded;
- CPU architecture, libc, BLAS/OpenMP and Torch build recorded;
- dependency installation with `--require-hashes` and a declared wheel/source
  policy;
- `pip check` after installation;
- package wheel built and installed into a venv without
  `--system-site-packages`;
- no network access during validation/replay after the image and legal input
  bundle have been staged;
- numeric tolerances declared for cross-host replay; no unsupported bitwise
  cross-hardware claim.

The current hashed lock must be regenerated and tested on the selected Linux
image if it cannot install there exactly. Regeneration creates a new source
identity and cannot occur during the protected Stage-09 run.

## 4. Reproducer layers

One command should orchestrate layers whose authority remains distinct:

```text
environment → source/tests → receipt validation → POST rendering → PDF/SI QA
```

### Layer A — public synthetic smoke

- uses only synthetic fixtures and redistributable code;
- trains a tiny model and validates schemas/invariants;
- is safe for fork CI;
- never claims to reproduce Route-A or Route-B empirical numbers.

### Layer B — local evidence validation

- requires an explicitly supplied local evidence bundle;
- performs no network acquisition and no opening;
- validates receipt self-hashes, byte bindings, chronology and authorized input
  identities;
- refuses incomplete, void, mixed-run or rights-unknown public bundles.

### Layer C — paper reproduction

- consumes only validated receipts;
- renders all tables, figures, Results, Discussion and SI;
- fails if any manuscript number is not receipt-derived;
- compiles TeX to PDF in the fixed image and runs page/asset checks;
- optionally builds DOCX only when `python-docx` and its runtime are locked.

## 5. `make reproduce-paper` contract

The future entrypoint should accept explicit inputs and never infer authorization:

```text
make reproduce-paper \
  EVIDENCE_BUNDLE=/absolute/path/to/local-evidence \
  BUILD_ROOT=/absolute/path/to/empty-build-root
```

It must:

1. reject a non-empty or symlinked build root unless an explicit safe reuse
   contract validates it;
2. never write the repository's live `outputs/**` tree;
3. validate the environment and all receipts before rendering;
4. select PRE or POST rendering from validated evidence state, not a user-provided
   narrative flag;
5. write a self-hashed reproduction receipt containing source/image/dependency
   identities, commands, exit codes, input/output hashes and declared tolerances;
6. fail closed if POST evidence is absent; it may build the PRE manuscript but
   must not create placeholder performance values.

## 6. CI acceptance matrix

| Job | Data | Required result |
| --- | --- | --- |
| `synthetic-cleanroom` | synthetic only | install, lint, type, full tests, smoke, wheel import, `pip check` |
| `pre-paper` | frozen PRE documents | Markdown→TeX current and PDF compiles |
| `local-evidence-replay` | protected/manual local bundle | receipts and renderer reproduce within declared tolerances |
| `public-release-gate` | repository only | PUBLIC bundle continues to fail closed until rights approval |
| `second-host-replay` | approved bundle on independent Linux host | numerical-tolerance receipt; needed for P1-05 |

## 7. Acceptance criteria

P0-R0 is complete only when:

- a fresh clone on the fixed Linux image installs from the exact locked graph;
- tests are 0-fail in an isolated namespace;
- ruff, mypy, claim validation and `pip check` pass;
- the public synthetic path is independent of unresolved data rights;
- a validated local evidence bundle reproduces all authorized paper outputs with
  a self-hashed receipt;
- PDF/SI build and QA pass;
- background runs cannot mutate or invalidate the clean-room test/build root.

# Post-opening remediation: two-tier source hash, and the deferred-fix queue

**Date:** 2026-08-05
**Status:** PLANNED — execute only after the one-time opening completes.
**Purpose:** pay down, with a single lineage, the technical debt that has cost five full retrains.

---

## 1. The defect

`source_tree_hash` treats the entire engineering surface as scientifically
load-bearing:

```python
DEFAULT_SOURCE_PATTERNS = (
    "src/**/*.py", "scripts/**/*.py", "scripts/**/*.sh",
    "tests/**/*.py",                                  # <-- cannot change any number
    "protocols/**/*.json", "protocols/**/*.md",
    ".github/workflows/*.yml", ".github/workflows/*.yaml",   # <-- cannot change any number
    "pyproject.toml", "requirements.txt", "requirements-lock*.txt",
)
```

Because every training receipt binds `source_sha256` and every consumer re-derives it
from the live tree, **editing a test invalidates the science.**

### 1.1 Measured cost

Five lineages already spent, none of which could alter a predicted value:

| Commit | Change | Could it change a number? |
|---|---|---|
| `a8dbcee` | worker caps → 96 | no |
| `61eb21b` | unified role cap v2 | no |
| `2e7e6c0` | LightGBM `n_jobs` from policy | no (`deterministic=True, force_col_wise=True`) |
| `64839c8` | 09b completion gate follows policy cap | no |
| `1e517b7` | 09b gate float slack on bridge report | no |

At roughly two days per lineage this is the dominant cost item in the project — far
larger than the modelling itself.

### 1.2 The self-defeating property

Improving the test suite invalidates the results the test suite exists to protect.
The system penalises the exact practice it was built to enforce.

## 2. The design: split the hash in two

Replace one identity hash with two, only one of which gates.

### 2.1 `model_source_sha256` — GATES identity

Covers only what can change a predicted number:

```python
MODEL_SOURCE_PATTERNS = (
    "src/thermoroute/**/*.py",
    "scripts/0*.py", "scripts/1*.py", "scripts/2*.py",   # stage entrypoints
    "protocols/**/*.json", "protocols/**/*.md",
    "requirements-lock*.txt",
    "pyproject.toml",
)
```

Combined, as now, with `panel_sha256`, `registry_sha256`, `config_sha256`,
`input_closure_sha256` and the declared seed set.

### 2.2 `harness_sha256` — RECORDED, does not gate

Covers the engineering surface. Written into every receipt for provenance; **never**
compared against the live tree:

```python
HARNESS_PATTERNS = (
    "tests/**/*.py",
    "scripts/**/*.sh",
    ".github/workflows/*.yml", ".github/workflows/*.yaml",
    "ops/**/*.sh", "ops/**/*.py",
)
```

### 2.3 Why this is not a weakening of the evidence

A reviewer's question is *"does this code, on this data, with these seeds, produce
these numbers?"* — answered entirely by `model_source_sha256`. No reviewer's question
is answered by binding the CI workflow file into the run identity. The harness hash
is still recorded, so the full engineering state remains auditable; it simply stops
being able to retroactively void completed science.

### 2.4 Compatibility

Receipts written under the current scheme keep `source_sha256`. The migration writes
all three fields (`source_sha256`, `model_source_sha256`, `harness_sha256`) and
switches every `!=` guard to compare `model_source_sha256` only. Old receipts remain
verifiable under a documented legacy path.

## 3. The deferred-fix queue

Everything below lands in **one** lineage. Nothing here may be applied during the
freeze ([`CODE_FREEZE_DISCIPLINE_20260805.md`](CODE_FREEZE_DISCIPLINE_20260805.md)).

| # | Fix | File | Why deferred |
|---|---|---|---|
| R1 | Two-tier hash (§2) | `src/thermoroute/repro.py` + all guard sites | hashed path |
| R2 | Stage-19 `q05 >= q95` → `q05 > q95`, plus a recorded `degenerate_zero_width_rows` count in the receipt | `scripts/19_probabilistic.py` | hashed path; see [disposition](STAGE19_DEGENERATE_INTERVAL_DISPOSITION_20260805.md) |
| R3 | Route-B protocol seal | `protocols/route_b_*.json` | hashed path; see [suspension](ROUTE_B_SUSPENSION_20260805.md) |
| R4 | Remove the thread cap from the runtime identity contract; keep it recorded | `protocols/route_a_numerical_policy_v*.json`, runtime contract | makes parallelism tunable without a retrain |
| R5 | Partition the prediction tables (§4) | `scripts/09_*.py`, readers | hashed path |
| R6 | Claim registry should pin the **claim blocks**, not the whole-manuscript sha256 | `protocols/route_a_claim_registry_v1.json` | hashed path; see §3.1 |
| R7 | `_validate_member_predictions` must establish (or assert) the deterministic runtime itself; add `assert_role_thread_cap` to Stage-24 | `src/thermoroute/development_controls_gate.py`, `scripts/24_freeze_model_suite.py` | hashed path; see [Stage-24 defect](STAGE24_REPLAY_RUNTIME_DEFECT_20260805.md) |

Executing R1 first means R2–R7 are cheap forever after; but they must still be
applied together in the same lineage to avoid paying twice.

### 3.1 Why R6 matters

`protocols/route_a_claim_registry_v1.json` pins `preopen_document_sha256` for
`paper/ThermoRoute_paper.md`. The manuscript rewrite of 2026-08-05 changed the file
from `4843656b…` to `c59af7a7…`, so `scripts/26_validate_claims.py` will report a
document-hash mismatch until the registry is re-sealed — and re-sealing edits a
hashed path.

This is the same pathology as `tests/**/*.py` being in the identity hash, applied to
the manuscript: **every future edit to the paper invalidates the claim registry, and
each re-seal costs a lineage.** A manuscript is edited dozens of times before
submission; this cannot stand.

The substance the registry protects is unharmed. Verification after the rewrite:

- all 8 claim blocks byte-exact (5,982 B, `63698d1e…`);
- both allowlisted legacy sentences present verbatim;
- all 21 free-text lints pass with whitespace normalisation, 0 unallowed hits;
- B-02 forbidden-verb scan clean.

R6 therefore changes the registry to bind the **claim-block digest** plus the lint
results, and to record the whole-document hash as provenance only. Until R6 lands,
the mismatch is expected and documented; do not "fix" it by editing `protocols/`.

**Sequencing note:** the finishing chain (24 → 27 → 14 → 26) runs in the multicore
worktree, which still holds the pre-rewrite manuscript bytes, so Stage-26 passes
there. The mismatch appears only after the manuscript work is merged in. Do not
re-run Stage-26 after the merge and read its failure as a regression.

## 4. Performance remediation (same lineage, R5)

### 4.1 The problem

`outputs/predictions/usgs_predictions_with_perstation_v2.parquet` is 922 MB /
27,740,891 rows in long format
(`model × scope × feature_set × seed × site × horizon × date`), derived from a 13 MB,
657,480-row input panel — a 42× row and 240× byte amplification.

Every diagnostic stage reads the whole table. Measured wall clock from
`outputs/logs/multicore_remaining.log` on 2026-08-05:

| Stage | Wall clock | Actual output |
|---|---|---|
| `18_rev_curve` | 70 min | one status table |
| `20_tuurt` | 71 min | a 3 × 3 table |
| `15_stratified` | 72 min | one stratified table |
| `22_adaptive_conformal` | 86 min | a 25-row table |

The arithmetic in each is seconds. The cost is I/O and revalidation.

### 4.2 The fix

Write predictions as a hive-partitioned dataset partitioned by `model` and `scope`,
so each diagnostic reads only the arms it consumes:

```
outputs/predictions/stage9_v2/model=LightGBM/scope=temporal/part-0.parquet
outputs/predictions/stage9_v2/model=ThermoRoute/scope=region_transfer/part-0.parquet
...
```

Expected effect: the four stages above drop from ~5 hours combined to minutes.
Content hashing stays exact — hash the sorted per-partition digests.

## 5. Operational remediation (no lineage cost — `ops/` is not hashed)

Can be done as soon as no chain is running.

### 5.1 The monitor restarts the whole chain instead of resuming

`ops/stage09/monitor_chains.sh` restarts from stage `[3/19]` on any failure.
Evidence from `outputs/logs/monitor_chains.log`, six restarts on 2026-08-04/05:

```
08-04 12:51 -> [3/19]   08-05 05:01 -> [3/19]
08-04 17:04 -> [3/19]   08-05 08:10 -> [3/19]
08-04 22:57 -> [3/19]   08-05 12:19 -> [3/19]
```

Each restart re-ran per-station LightGBM, the rigor folds, four region-transfer folds
and the assemble step — about 30 minutes of pure rework per restart. Stage-25 was
fully retrained twice (13,625 s and 3,555 s ≈ 4.8 h wasted).

**Fix:** record the last completed stage in a state file and resume from `last + 1`.

### 5.2 The monitor kills healthy chains

It treats any line matching `ERROR` in the log as death:

```
[2026-08-05T12:18:34Z] ERRORS detected in remaining log after line 901
[2026-08-05T17:48:52Z] ERRORS detected in remaining log after line 1117
```

Both killed a running chain over a non-fatal stderr line.

**Fix:** liveness must be determined by process state and exit code, never by
grepping the log for a substring.

### 5.3 The monitor itself failed silently

```
nohup: failed to run command 'ops/stage09/chain_stage09_remaining.sh': Permission denied
```
repeated four times over eight minutes.

**Fix:** verify the target is executable at monitor start and fail loudly if not.

## 6. Execution order

1. Opening completes; opening receipts validated. **Freeze lifts.**
2. Apply R1–R5 together on a branch. Run the full test suite.
3. Run one — and only one — full lineage: Stage-09 → 09b → 16 → 25 → 24 → 27 → 14 → 26.
4. Confirm every receipt validates under `model_source_sha256`.
5. Apply §5 operational fixes at any convenient point (no lineage cost).
6. Record the new lineage identity in the manuscript's reproducibility statement.

## 7. What this is worth

Once R1 lands, the marginal cost of a test fix, a lint pass, a CI change, an ops
change or a thread-count experiment falls from **two days to zero**. Every subsequent
project on this codebase inherits that.

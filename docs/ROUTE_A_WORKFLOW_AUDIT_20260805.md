> Historical record (superseded by the conventional design).

# Route-A workflow audit: why the development chain cost days instead of hours

**Date:** 2026-08-05
**Scope:** the Route-A development pipeline, its governance apparatus, and its operational harness.
**Method:** direct inspection of both worktrees, the chain and monitor logs, the identity contracts, and the produced artifacts. Every number below was measured, not estimated.

---

## 1. Headline

The cost driver was never the science. It was that the verification system's identity
contract made **every engineering fix cost a full scientific rerun**, while the system
generated its own defects — thread caps, config shapes, import order, gate thresholds
— faster than two-day retrains could absorb them. That is a positive feedback loop,
and it is an architectural property, not a discipline failure.

## 2. The scale mismatch

| Quantity | Measured |
|---|---|
| Input panel | `data_usgs/panel_usgs_120v2.parquet`, **13 MB**, 657,480 rows × 10 columns |
| Cohort | 120 stable USGS site numbers, 2006–2020, 34 states, 15 HUC2 groups |
| Common evaluation keys | 249,072 |
| Scientific core code | **7,610 lines** (model, training, features, baselines, metrics, conformal, …) |
| Governance code in `src/` | **56,834 lines — 88 % of `src/`** |
| `scripts/` | 43,584 lines |
| `tests/` | 50,316 lines, 60 files |
| Protocol documents | 25 |
| `raise` statements in `src/` | 3,977 |
| `sha256` references | 3,684 |
| `receipt` references | 1,866 |
| `src/thermoroute/opening.py` | **12,087 lines**, for a one-time "download labels and score" step that has not yet run |
| `src/thermoroute/chronology.py` | 4,174 lines, to establish a commit ordering that `git log` already establishes |

A 13 MB tabular regression problem carries roughly 150,000 lines of verification
infrastructure. LightGBM on this panel trains in seconds; five CPU LSTM seeds take
one to two hours. The entire scientific computation is about one machine-day.

## 3. Root causes, in order of cost

### 3.1 `tests/**/*.py` is inside the identity hash — the single most expensive design decision

`DEFAULT_SOURCE_PATTERNS` in `src/thermoroute/repro.py` includes `tests/**/*.py`,
`scripts/**/*.sh` and `.github/workflows/*`. Every training receipt binds
`source_sha256`, and every consumer re-derives it from the live tree and fails closed.

**Fixing a typo in a test invalidates all four training receipts and forces a full
retrain.**

Five lineages have already been spent this way, none capable of changing a number:

| Commit | Change |
|---|---|
| `a8dbcee` | raise worker caps to 96 |
| `61eb21b` | unified numerical-policy role cap v2 |
| `2e7e6c0` | LightGBM training `n_jobs` follows policy |
| `64839c8` | Stage-09b completion gate follows policy cap |
| `1e517b7` | 09b gate float slack on bridge report |

Five full retrains, all about thread counts and gate thresholds.

The sharpest illustration is in the branch diff itself. Part of the payload of one of
those lineages is a **single word in a docstring** in `src/thermoroute/repro.py`:

```diff
-    single-thread policy is attested separately by ``formal_numerical_policy``.
+    capped-thread policy is attested separately by ``formal_numerical_policy``.
```

That word is inside `source_tree_hash`. Changing it changes the run identity, and the
run identity is what four training receipts bind. A comment edit is, mechanically,
a scientific invalidation.

The property is also self-defeating: improving the test suite invalidates the results
the test suite exists to protect.

Remediation: [`SOURCE_HASH_SCOPE_REMEDIATION_PLAN.md`](SOURCE_HASH_SCOPE_REMEDIATION_PLAN.md) §2.

### 3.2 Performance tuning was made scientifically load-bearing

`protocols/route_a_numerical_policy_v2.json` states that a parent and its workers
share one thread cap and that the runtime contract embeds the process thread cap in
the run identity. Consequently **parallelism cannot be changed without changing the
scientific identity** — which is backwards, and is the direct cause of three of the
five lineages above.

The same policy pins `lightgbm_prediction_num_threads: 1`,
`torch_deterministic_algorithms: true`, `tf32: false`,
`float32_matmul_precision: highest`. These produce the single-threaded replay tails
(over an hour in Stage-16) and the 1–3 hour Stage-27 budget, whose purpose is to
confirm to 1e-12 that predictions already on disk equal themselves.

### 3.3 The monitor restarts the chain from the top instead of resuming

From `outputs/logs/monitor_chains.log`, six restarts on 2026-08-04/05, each re-entering
at stage `[3/19]`:

```
08-04 12:51 -> [3/19]    08-05 05:01 -> [3/19]
08-04 17:04 -> [3/19]    08-05 08:10 -> [3/19]
08-04 22:57 -> [3/19]    08-05 12:19 -> [3/19]
```

Each restart re-ran per-station LightGBM, the rigor folds, four region-transfer folds
and the assemble step — roughly 30 minutes of pure rework each. Stage-25 was fully
retrained twice: 13,625 s (3.8 h) and 3,555 s (1.0 h).

Three further defects compounded it.

**The restart command was malformed.** `restart_chain()` contained

```bash
nohup bash "$CHAIN_CMD" >>"$MONLOG" 2>nohup "$CHAIN_CMD" >>"$MONLOG" 2>&1 &1 &
```

— the residue of a botched edit. This is the source of the four
`nohup: failed to run command '…': Permission denied` lines over eight minutes on
2026-08-05.

**The completion marker never matched.** The monitor searched for
`CORE_RESULTS_CHAIN_COMPLETE`; the chain emitted `DEVELOPMENT_CHAIN_CORE_COMPLETE`.
A fully successful chain would therefore never have been recognised as finished, and
the monitor would have restarted it indefinitely. This defect had not yet been
observed only because the chain had not yet succeeded.

**Restart was unconditional.** A deterministic failure became an infinite retry loop:
each restart re-ran the same stage and hit the same error. This materialised at
20:03 UTC on 2026-08-05, when Stage-24 failed reproducibly and the monitor was
positioned to re-run a 40-minute stage forever.

*Correction to an earlier reading of this log:* the `ERRORS detected in remaining log`
lines do **not** cause a restart — that branch only writes to the monitor log. The
restarts at 12:18 and 17:48 were triggered by genuine process death, detected
correctly. The monitor's fault is what it did next, not how it decided.

**For an idempotent-but-expensive chain, a restart-from-top monitor is worse than no
monitor.** Conservative waste: 5–7 hours.

### 3.4 Every diagnostic re-reads a 922 MB prediction table to emit a three-row table

`outputs/predictions/usgs_predictions_with_perstation_v2.parquet` is 27,740,891 rows
in long format (`model × scope × feature_set × seed × site × horizon × date`) — a 42×
row and 240× byte amplification of a 657,480-row input.

Measured wall clock, 2026-08-05:

| Stage | Wall clock | Output |
|---|---|---|
| `18_rev_curve` | 70 min | one status table |
| `20_tuurt` | 71 min | a 3 × 3 skill table |
| `15_stratified` | 72 min | one stratified table |
| `22_adaptive_conformal` | 86 min | a 25-row table |

About five hours in one day for arithmetic that takes seconds. The cost is I/O and
revalidation, with no scientific content. Remediation: partition by `model`/`scope`
(plan §4).

### 3.5 Fail-closed with no severity tiers

Stage-19 halts the chain, and cascades into Stage-10, on a condition affecting
**135 of 26,993,675 member-level rows (0.0005 %)**. Measurement showed the condition
is not what it was believed to be: there are **zero** strict quantile-ordering
violations; the 135 rows are zero-width intervals (`q05 == q50 == q95`) at
ice-affected sites where water temperature is pinned near the freezing point and the
three percentiles genuinely coincide. Full analysis:
[`STAGE19_DEGENERATE_INTERVAL_DISPOSITION_20260805.md`](STAGE19_DEGENERATE_INTERVAL_DISPOSITION_20260805.md).

A contract that cannot distinguish "the model is wrong" from "the model is right and
the interval is degenerate" will keep halting on correct results.

### 3.6 Two worktrees diverged, with incompatible identities

`feat/route-a-completion` (2.6 GB, manuscript and figures) and `feat/multicore`
(15 GB, the live chain and all receipts) are git worktrees of one repository that had
drifted apart by four commits, all of them under hashed paths. Because
`source_tree_hash` covers `src/`, `scripts/`, `tests/` and `protocols/`, **the two
trees necessarily have different identities**, and the manuscript tree could never
validate the compute tree's receipts. This was a live hazard scheduled to surface at
Stage-14 / Stage-26.

### 3.7 The governance apparatus exceeds what it protects

`opening.py` is 12,087 lines for an operation that downloads labels once and scores
them. `chronology.py` is 4,174 lines to prove commit ordering. The protocol directory
contains seals of amendments of seals — for example
`route_a_numerical_policy_amendment_seal_v2.json` seals
`route_a_numerical_policy_amendment_v2.json`, which amends `…_v1.json`, which is
about **thread counts**.

### 3.8 The pre-registration's central verdict was arithmetically determined from day one

The outcome-free inference gate requires `n_clusters ≥ 30`. The frozen cohort has at
most 15 HUC2 groups (inverse-Herfindahl effective cluster count ≈ 9.54). The verdict
`DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED` was therefore knowable at the outset from
`df.huc2.nunique()`. A large confirmatory apparatus was built to formalise a
conclusion available from one line of pandas.

This is not an argument against pre-registration. It is an argument for checking a
gate's feasibility against the cohort *before* building the machinery that enforces it.

### 3.9 Estimation error was structural

Estimates were formed from training time, but roughly 80 % of wall clock is
validation, reload and rehash — and that cost scales with the accumulated artifact
tree, which grows monotonically. Estimates therefore degrade systematically over the
life of the project. The cost model, not the estimator, was wrong.

## 4. What is genuinely good and must be preserved

The audit found the *scientific* design to be stronger than the field norm:

- **Baseline suite** — persistence, damped persistence, air2stream, global LightGBM,
  per-station LightGBM, global LSTM. Most river-temperature papers compare against
  persistence alone.
- **Leakage control** — explicit issue-time separation and a strictly left-looking
  encoder.
- **Leave-HUC2-region-out transfer**, honestly labelled as *gauged* transfer rather
  than ungauged prediction.
- **Clustered inference** — whole-HUC2 bootstrap and sign-flip rather than treating
  daily rows as independent.
- **The one-time opening** — holding out 2021–2023 and acquiring labels exactly once
  is real methodological value and is defensible and valuable in a WRR submission.
  This is the part of the governance apparatus that earns its keep.

## 5. Calibration for future work

The apparatus was imported from clinical-trial pre-registration and high-energy-physics
blind analysis. Those fields justify it because a single experiment costs a fortune,
cannot be repeated, and sits in a strong incentive field for analytic flexibility.

Retrospective river-temperature regression on public USGS data satisfies none of
those conditions: the data can be re-downloaded by anyone at any time, a full rerun
costs about one machine-day, and no one profits from a favourable RMSE.

A usable test before adding any verification mechanism:

> **Would a WRR reviewer reject this paper for its absence?**

- One-time opening of a held-out period → yes, it is methodology. Keep it.
- SHA-256 receipt chains → no.
- A seal of an amendment of a seal → no.

AGU's actual requirements are a Data Availability Statement, data in a repository
with a DOI, and software archived with a DOI and a licence.

## 6. Decisions taken as a result

| Decision | Record |
|---|---|
| Freeze all hashed paths until the opening completes | [`CODE_FREEZE_DISCIPLINE_20260805.md`](CODE_FREEZE_DISCIPLINE_20260805.md) |
| Do not fix Stage-19; report it honestly | [`STAGE19_DEGENERATE_INTERVAL_DISPOSITION_20260805.md`](STAGE19_DEGENERATE_INTERVAL_DISPOSITION_20260805.md) |
| Suspend Route B; preserve the design record | [`ROUTE_B_SUSPENSION_20260805.md`](ROUTE_B_SUSPENSION_20260805.md) |
| Pay all deferred fixes with one post-opening lineage | [`SOURCE_HASH_SCOPE_REMEDIATION_PLAN.md`](SOURCE_HASH_SCOPE_REMEDIATION_PLAN.md) |
| Restructure the manuscript around the strong-baseline / leakage / transfer / conformal / one-time-holdout spine; compress governance to two Methods paragraphs | [`PAPER_RESTRUCTURE_20260805.md`](PAPER_RESTRUCTURE_20260805.md) |
| Rebind Fig 3 / Fig S5 onto Stage-22 evidence | [`FIGURE_PLAN_STAGE19_INDEPENDENT_20260805.md`](FIGURE_PLAN_STAGE19_INDEPENDENT_20260805.md) |

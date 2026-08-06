# Option A — drop the confirmatory apparatus, report a fixed-cohort descriptive benchmark

| Field | Value |
| --- | --- |
| Date | 2026-08-06 |
| Question | What does it cost to move from "pre-specified confirmatory design whose gate failed" to "fixed-cohort descriptive benchmark"? |
| Method | Read-only audit of `protocols/`, `src/`, `scripts/`, `tests/` in `/home/lzq/workspace/parttime/thermoroute-remediation`; independent recomputation of the cluster geometry with the gate's own code |
| Verdict | **Free in substance, but not for the reason assumed.** No protocol edit, no registry edit, no code edit is required *by Option A*. The one code change that makes manuscript editing safe (R6) is already written and already inside the single lineage being spent for the Stage-24 blocker. Marginal cost of Option A: **zero**. |
| Status | Analysis only. Nothing under `protocols/`, `src/`, `scripts/`, `tests/` was modified. Nothing committed. |

---

## 0. Executive answer

**Yes, it is free — but two things in the framing are wrong, one in your favour and one against.**

1. **In your favour.** The `n_clusters >= 30` threshold is *not* what makes this study
   descriptive. `claim_eligible` is a **hardcoded literal `False`**
   (`src/thermoroute/inference_gate.py:501`), never computed from the cluster gate.
   Three independent components fail permanently, and the cluster gate is only one of
   them. Switching the clustering unit to HUC8 would pass the cluster gate and change
   *nothing*: the verdict would still be `DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED`.
   The reviewer's "ritual designed to fail" reading is invited by the manuscript's own
   §3.6, which presents the cluster gate as *the* reason. It is the weakest of the three.

2. **Against you.** "Entirely on the manuscript side at zero lineage cost" is not true
   of the branch you are on. On `feat/route-a-completion` the committed
   `scripts/26_validate_claims.py` compares the **whole manuscript file** against the
   SHA-256 pinned inside the sealed claim registry. That pin is **already broken** —
   the manuscript is at `1507f20c…`, the registry pins `4843656b…`, and
   `tests/test_claim_registry.py::test_every_required_preopen_document_matches_its_frozen_sha256`
   **fails today**. Re-pinning is a `protocols/` edit, which is in the gating source
   tier and therefore costs a lineage. The escape (R6) is a `scripts/*.py` edit, which
   is *also* in the gating tier.

3. **Why it is nevertheless free.** R6 is already written (uncommitted in the
   remediation worktree) and already scheduled inside the one lineage being spent for
   the Stage-24 worker-thread-cap blocker
   (`docs/REMEDIATION_LINEAGE_RUNBOOK_20260805.md` §1: "Since one lineage must be
   spent, the whole deferred queue is landed at once: R1–R4, R6–R9"). Option A adds no
   file to that batch. **The debt exists, it is already owed, and Option A does not
   increase it.**

---

## 1. Independent verification of the cluster numbers

Recomputed with the gate's own function, `thermoroute.inference_gate.cluster_geometry`,
against **both** the file you used and the file the gate actually reads
(`STATION_REGISTRY_RELATIVE = "data_usgs/station_registry_v1.csv"`,
`src/thermoroute/inference_gate.py:50`).

| unit | n_clusters | effective (1/Σs²) | effective fraction | largest share | gate |
| --- | ---: | ---: | ---: | ---: | --- |
| HUC2 (current) | 15 | 9.536 | 0.6358 | 0.2167 | FAIL (count, fraction) |
| HUC4 | 64 | 32.432 | 0.5068 | 0.1000 | FAIL (fraction) |
| HUC6 | 75 | 36.364 | 0.4848 | 0.1000 | FAIL (fraction) |
| HUC8 | 95 | 72.000 | 0.7579 | 0.0417 | **PASS** |

**Your table is exactly right**, to every digit, from both sources. Cluster sizes under
HUC2 are `[2,2,3,3,4,5,5,7,8,8,10,10,13,14,26]`, `cluster_size_cv = 0.7569`,
`n_stations = 120`.

Three notes:

- **A trap for anyone repeating this.** `data_usgs/station_registry_v1.csv` stores
  `huc_cd` and `huc2` **without leading zeros** (`1060003`, not `01060003`; `1`, not
  `01`), while `data_usgs/huc_metadata_usgs_v1.csv` preserves them. Slicing
  `huc_cd[:2]` on the registry yields **18 clusters**, not 15 — 64 of 120 rows are
  affected. Zero-pad to 8 (or 12) first. The gate itself is unaffected because it reads
  the `huc2` column directly (`inference_gate.py:289`), and the unpadded `'1'` and the
  padded `'01'` are the same partition.
- The gate computes over the *frozen registry*, i.e. **before** reportability
  attrition. Post-opening `n_clusters` per test can only be ≤ 15.
- Your "≈21 HUC2 regions nationally" is right and the constraint is structural: the WBD
  has 21 (18 CONUS + 19 Alaska + 20 Hawaii + 21 Caribbean; 22 Pacific Islands if
  counted). **`>= 30` is unreachable for any U.S. cohort whatsoever**, not merely for
  this one. That is the reviewer's point and it stands.

---

## 2. The hypothesis, tested

> *"Achievable entirely on the manuscript side, at zero lineage cost, because the gate
> already resolves permanently to `DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED` — the five
> formal comparisons are already restricted to fixed-cohort descriptive effects."*

### 2.1 The premise is CONFIRMED, and is stronger than stated

`claim_eligible` is not a computed value. It is a literal in the returned document:

```python
# src/thermoroute/inference_gate.py:499-503
    "null_simulation_gate": null_simulation,
    "claim_eligible": False,
    "analysis_mode": "FIXED_COHORT_DESCRIPTIVE_ONLY",
    "blocking_reasons": blocking,
```

and the validator re-asserts it (`inference_gate.py:531-532`):

```python
    if actual.get("claim_eligible") is not False:
        raise InferenceGateError("current Route-A inference gate did not fail closed")
```

and the **opening refuses to run unless the gate has failed** (`opening.py:5255`):

```python
    if gate["claim_eligible"] is not False:
        raise OpeningContractError("current Route-A gate did not fail closed")
```

There is no code path anywhere in `src/` or `scripts/` that can set it to `True`.

**Three independent permanent failures**, of which the cluster gate is only one:

| # | Component | Why it can never pass | Locus |
| --- | --- | --- | --- |
| 1 | `structural_assumption_gate` | Both assumptions are hardcoded `"status": "NOT_ESTABLISHED"` — `INDEPENDENT_EXCHANGEABLE_HUC2_SAMPLING` ("not probability sampled") and `JOINT_CLUSTER_VECTOR_SIGN_SYMMETRY` ("no randomized sign assignment") | `inference_gate.py:59-77` |
| 2 | `cluster_gate` | `n_clusters >= 30` with HUC2 fixed as the unit; 21 exist nationally | `inference_gate.py:55-57, 429-434` |
| 3 | `null_simulation_gate` | `"pass": False` unconditionally; status is `NOT_IMPLEMENTED_FAIL_CLOSED`, and `NULL_SIMULATION_NOT_PASSING` is appended to `blocking_reasons` on every run | `inference_gate.py:440-470` |

**This is the single most useful finding for the reviewer response.** Component 1 is the
honest, defensible reason — it is a statement about the *sampling design*, not about a
threshold. Component 3 is candidly "we never built it". Component 2 is the one that
reads as a ritual. The manuscript currently leads with component 2.

### 2.2 The consequence is CONFIRMED — the five rows are already descriptive

`scripts/26_validate_claims.py` selects the required claim wording as a pure function of
the gate booleans, **before** looking at any p-value or interval:

```python
# scripts/26_validate_claims.py:1557-1570
def _result_verdict(row, *, inference_claim_eligible, outcome_qc_claim_eligible):
    if row["status"] != "ESTIMABLE":
        return "NOT_ESTIMABLE"
    if inference_claim_eligible is not True and outcome_qc_claim_eligible is not True:
        return "DESCRIPTIVE_ONLY_BOTH_GATES_FAILED"
    if inference_claim_eligible is not True:
        return "DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED"
```

and then **overwrites** the registry's declared polarity/template with that verdict
(`26_validate_claims.py:1666-1683`: `entry["polarity"] = verdict; entry["template_id"] = verdict`).

The five registry specs carry the sentinel `"template_id": "AUTO_CONFIRMATORY_DECISION"`
and `"polarity": "AUTO_FROM_VERIFIED_STATISTICS"`
(`protocols/route_a_claim_registry_v1.json:418-504`) — the sentinel is never looked up.
The confirmatory templates `SUPERIORITY_SUPPORTED` / `NONINFERIORITY_SUPPORTED` are
**unreachable**, because `support_requires` includes
`authorization.inference_gate.claim_eligible == true` (registry `:552`).

### 2.3 Answer to your Question 2 — **NO, the registry does not have to change**

> *"After the opening, will `26_validate_claims.py` demand result claims phrased as
> formal five-comparison outcomes?"*

**No. It will demand the opposite, byte-exactly.** The mandated post-opening sentence
per test is already the fixed-cohort descriptive one
(`protocols/route_a_claim_registry_v1.json:413`):

> `Route A {test_id}: DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED for {candidate} versus {reference} at horizon {horizon} with margin {margin_c} °C; in the fixed availability-enriched cohort, station-median RMSE effect={effect_c} °C, n={n_stations} reportable stations in {n_clusters} HUC2 clusters. The whole-HUC2 bootstrap interval [{ci_low_c}, {ci_high_c}] °C, exact sign-flip raw p={p_raw}, and Holm p={p_holm} are assumption-conditional sensitivities only because the frozen inference gate failed; they do not support superiority, non-inferiority, equivalence, parity, or a U.S.-river superpopulation claim.`

That *is* a fixed-cohort descriptive benchmark statement. Additionally:

- There is **no check that requires inferential language** and **no check that forbids
  descriptive phrasing**. The only body check is exact equality against the
  deterministically rendered template (`26_validate_claims.py:2434-2440`).
- The only directional guard is **anti-overclaim**: `LINT_UNSTRUCTURED_ROUTE_A_RESULT`
  bans `confirms|establishes|demonstrates|supports|achieves|outperforms|superiority|non-inferiority`
  within 100 characters of `Route-A|confirmatory …|one-time opening|2021…2023|H1-…|H2-…`
  in free text outside claim blocks. **Dropping the word "confirmatory" from the
  narrative reduces this lint's surface area.** Option A moves *with* the enforcement,
  not against it.
- `required_postopen_coverage` (registry `:505-515`) requires each of the five
  `RESULT_*` claim IDs to appear **exactly once** post-opening, and **zero times**
  pre-opening (`26_validate_claims.py:2448-2459`). This is a *rendering* obligation, not
  a *framing* obligation: all five rows must still be shown. That is desirable for a
  descriptive benchmark anyway — it is what stops selective reporting.

**Therefore the registry byte-pin question is moot.** For completeness, had an edit been
needed, the cost is documented below (§5.3): four pins, one of them a commit-immutable
sealed attestation that **cannot be repaired**, forcing a new amendment document plus a
new seal — and, because `protocols/**/*.json` is in the gating tier, a full retrain.
This is precisely the wall recorded in
`docs/M01_M14_M15_MANUSCRIPT_BATCH_DEFERRED.md`. **Option A does not hit it.**

### 2.4 Answer to your Question 4 — **NO, the computation never differs**

`produce_trusted_opening_products` (`src/thermoroute/opening.py:10039-10245`) is
straight-line. There is no `if claim_eligible` branch anywhere in it. Order:

outcome-quality audit → approved-target sensitivity → spatial sensitivity →
outcome-QC gate → probabilistic evaluation → write five JSONs → `compute_confirmatory_statistics`
→ write statistics/CSV/parquet → temporal-coverage audit → render `report_v1.md`.

The gate affects **wording only**. `_render_confirmatory_report`
(`opening.py:9559-9563`) computes a `combined_directional_gate` flag and prints it,
while explicitly stating that "The five unfiltered fixed-cohort effects remain
reportable." Every table — the five-row family, per-HUC/equal-HUC/leave-one-HUC,
exact-A sensitivity, year/season sensitivity — is rendered unconditionally.

**"Run the formal family and report it descriptively" and "run a descriptive benchmark"
are byte-identical computations.** Confirmed.

### 2.5 The part of the hypothesis that is wrong

The manuscript is pinned by SHA-256 inside the sealed claim registry:

```
protocols/route_a_claim_registry_v1.json → preopen_document_sha256
    "paper/ThermoRoute_paper.md": "4843656b5858c8d0f997d0bc5dcc55aa79e77fc8c974b1b7a1ee17ce519104e7"
    "paper/agu_submission/ThermoRoute_WRR.tex": "34a34af4f727f57831bb48bb3520b39a5c4cd4e7d1d4999675961e908824da92"
```

Actual on `feat/route-a-completion` today: `1507f20c3e83a648…` and `dbbf2b14d3af5799…`.
Both already diverge, from the WRR restructure (`dba47c0`, `ed0029d`, `dee33fb`).
Verified by running the test:

```
FAILED tests/test_claim_registry.py::test_every_required_preopen_document_matches_its_frozen_sha256
AssertionError: paper/ThermoRoute_paper.md
assert '4843656b5858...' == '1507f20c3e83...'
```

So manuscript editing is **not currently free on this branch**. It is made free by R6.

### 2.6 R6 — the fix, already written, already queued

`docs/R6_CLAIM_REGISTRY_BLOCK_BINDING.md` (untracked, remediation worktree) changes
`26_validate_claims.py` so the registry binds **claim blocks**, not whole files:

```python
# scripts/26_validate_claims.py:107-110  (uncommitted, remediation worktree)
BYTE_FROZEN_DOCUMENT_PATTERNS: tuple[str, ...] = (
    "protocols/*.md",
    "protocols/**/*.md",
)
```

with the rationale in the docstring at `26_validate_claims.py:2031-2037`:

> "`preopen_document_sha256` is still read for the documents that are inside the
> run-identity source set anyway; for narrative documents it is recorded provenance of
> the sealed PRE bytes and is **deliberately never compared**, so ordinary manuscript
> revision no longer forces a protocol re-seal."

The guarantee is preserved: the eight `LIMIT_*` claim blocks (and post-opening the five
`RESULT_*` blocks) are still rendered from the registry and compared byte-for-byte, and
the free-text lints still run. Only the surrounding prose is unfrozen.

**Status check performed:**

| Item | State |
| --- | --- |
| `scripts/26_validate_claims.py` (R6) | modified, **uncommitted**, remediation worktree only |
| `tests/test_claim_registry.py` | modified, **uncommitted**, remediation worktree only |
| `docs/R6_CLAIM_REGISTRY_BLOCK_BINDING.md` | **untracked** |
| Present on `feat/route-a-completion`? | **No** — `grep -c BYTE_FROZEN_DOCUMENT_PATTERNS` returns 0 |
| Scheduled? | Yes — `docs/REMEDIATION_LINEAGE_RUNBOOK_20260805.md` §1, batch R1–R4, R6–R9 |
| Why a lineage is being spent anyway | Stage-24 hardcodes `OMP_NUM_THREADS="1"` and omits `THERMOROUTE_FORMAL_THREADS`, replaying at 1 thread members trained at 8 (`docs/STAGE24_WORKER_THREAD_CAP_BLOCKER_20260805.md`) |

---

## 3. The two source-hash tiers — what "costs a lineage" actually means

From `src/thermoroute/repro.py:861-907`. This is the distinction the whole cost analysis
turns on.

| Tier | Patterns | Effect of an edit |
| --- | --- | --- |
| **`MODEL_SOURCE_PATTERNS`** (gating) | `src/thermoroute/**/*.py`, **`scripts/*.py`**, `protocols/**/*.json`, `protocols/**/*.md`, `requirements-lock*.txt`, `pyproject.toml` | Changes `model_source_sha256`. **Every training receipt becomes "stale for the current model source"** — compared live in `model_suite.py:3155,4711,6513,7380`, `stage09_parallel.py:245,2533`, `development_controls_gate.py:602`, `opening.py:3286`, and 6 stage scripts. **This is the lineage cost.** |
| `HARNESS_PATTERNS` (engineering) | `tests/**/*.py`, `scripts/**/*.sh`, `scripts/data_usgs/**/*.py`, `ops/**`, `.github/workflows/*` | Recorded, **never compared**. Free. |
| Not in any pattern | **`paper/**`**, **`docs/**`**, `outputs/**` | Outside the identity hash entirely. Free. |

Two consequences that matter:

- **`paper/` is in no pattern.** The manuscript has never been part of the identity
  hash. It was only ever bound by the registry's `preopen_document_sha256` pin — which
  R6 stops comparing. After R6, manuscript edits are free **permanently**.
- **`scripts/*.py` is top-level-only, deliberately** (`repro.py:879-884`), so
  `scripts/26_validate_claims.py` **is** gating. R6 costs a lineage. `tests/` does not.

---

## 4. The change table

Every change required to move from a pre-specified confirmatory design to a fixed-cohort
descriptive benchmark, split by cost.

### 4.1 MANUSCRIPT-ONLY — free

Free after R6 lands; blocked before it only by the already-broken whole-file pin.

| # | Change | File / location |
| --- | --- | --- |
| M1 | **Retitle.** Drop "and a pre-specified inference gate" from the title | `paper/ThermoRoute_paper.md:1`; `paper/agu_submission/ThermoRoute_WRR.tex` `\title` |
| M2 | **Reframe §3.6.** Retitle "Estimand, comparison family, and the pre-specified inference gate" → e.g. "Estimand, comparison set, and why the analysis is descriptive". Present the five rows as a **frozen comparison set** reported as fixed-cohort descriptive effects, not a "confirmatory family" | `paper/ThermoRoute_paper.md:606-690`; tex `:660-770` |
| M3 | **Fix the reviewer's actual objection.** Replace "The frozen cohort cannot pass it" (`:674`) with the honest three-reason account of §2.1 — lead with `INDEPENDENT_EXCHANGEABLE_HUC2_SAMPLING` being not established (a design fact), state the null simulation was never implemented, and demote the cluster thresholds to a corroborating diagnostic. **Say explicitly that no U.S. cohort could reach 30 HUC2 regions**, and that the design is therefore descriptive by construction rather than by a threshold that happened to be missed | `paper/ThermoRoute_paper.md:669-681` |
| M4 | **Abstract / §1 / §5.1 / §7.** Same reframe; §5.1 "A pre-specified gate that failed is a result, not an excuse" (`:1000`) needs rewriting — under Option A the gate is not the story, the descriptive benchmark is | `:22, :59, :176, :1000, :1291`; tex mirrors |
| M5 | **Add the descriptive results layer.** Per-HUC2 effects, equal-HUC summary, leave-one-HUC2 influence, station-count distribution, year/season stability. All source fields exist (§6) | `paper/ThermoRoute_paper.md` §4 |
| M6 | **Define "skill".** One sentence — see §7 | near `paper/ThermoRoute_paper.md:828` |
| M7 | **Lint pass.** Re-check `LINT_UNSTRUCTURED_ROUTE_A_RESULT` after the rewrite. Removing "confirmatory" shrinks the trigger set, but "Route-A … descriptive benchmark **supports** …" would still trip on `supports` | whole manuscript |
| M8 | Mirror M1–M7 into `paper/highlights.md`, `paper/cover_letter.md`, `paper/si/SI05_comparison_family.md`, `paper/FIGURE_REDRAW_SPEC.md` | `paper/**` |

### 4.2 PROTOCOL — costs a lineage

| # | Change | Required by Option A? |
| --- | --- | --- |
| P1 | Edit `protocols/route_a_claim_registry_v1.json` (`result_claim_specs`, templates, `required_postopen_coverage`) | **NO.** §2.3 — the validator already demands descriptive phrasing. |
| P2 | Re-pin `preopen_document_sha256` for the manuscript | **NO**, if R6 lands. **YES**, if R6 is dropped — and then it is a `protocols/` edit, i.e. a full retrain plus the four-pin re-seal of §5.3. R6 is strictly cheaper and permanent. |
| P3 | Amend `protocols/route_a_inference_amendment_v2.json` to change the clustering unit or the `>= 30` threshold (`:87`) | **NO — and actively harmful.** This is the "subdivide until it passes" move. HUC8 passes only because adjacent HUC8 units are not independent. Also pointless: `claim_eligible` is hardcoded `False`. |
| P4 | Extend `primary_inference_contract.exploratory_spatial_inference_sensitivity` to add per-station or quantile fields | **NO** — see §6.3 for the cheap alternative. Would break `route_a_protocol_seal_v1.json` and `PROTOCOL_OBJECT_SHA256`. |

**Nothing under `protocols/` must change for Option A.**

### 4.3 CODE — costs a lineage

| # | Change | Required by Option A? | Deferrable past the opening? |
| --- | --- | --- | --- |
| C1 | **R6** — `scripts/26_validate_claims.py`, block binding instead of whole-file binding | **Not by Option A**, but required for *any* manuscript edit on this branch. Already written, already in the remediation batch. | **No — must land before the lineage runs.** But it already is scheduled to. |
| C2 | `tests/test_claim_registry.py` — drop the whole-file assertion for narrative documents | Yes, paired with C1 | **`tests/` is HARNESS-only — free.** Can land at any time. |
| C3 | `src/thermoroute/inference_gate.py` — change `MIN_CLUSTERS`, the clustering unit, or the hardcoded `claim_eligible` | **NO.** Do not touch. | n/a |
| C4 | `src/thermoroute/opening.py` — emit per-station effects / effect quantiles / win counts | **NO** — recoverable post hoc from `trusted/temporal_predictions_v1.parquet` (§6.3) | The *artifact* is not deferrable (the opening is one-shot), but the *derived quantity* is. |
| C5 | A renderer that turns `trusted/statistics_v1.json` + `trusted/spatial_sensitivity_v1.json` into manuscript tables | Optional. **No such stage exists today.** `scripts/12_claim_stats.py:363-476` does this for the *development* period only. | **Yes — fully deferrable.** Writing it as `scripts/31_*.py` before the lineage costs nothing extra; writing it after costs a lineage. **Write it now, inside the same batch.** |

**Summary: Option A requires zero PROTOCOL changes and zero CODE changes of its own.**
The only code item on the critical path (C1/R6) is already in the batch for an unrelated
blocker. C5 is the one thing worth *adding* to that batch opportunistically.

---

## 5. Detail on the costly items

### 5.1 R6 (C1) — cannot be deferred, but is already paid for

`scripts/26_validate_claims.py` matches `scripts/*.py` in `MODEL_SOURCE_PATTERNS`, so
landing R6 changes `model_source_sha256` and invalidates all four existing training
receipts. `docs/REMEDIATION_LINEAGE_RUNBOOK_20260805.md` §1 is explicit that this cost
is being paid regardless, for the Stage-24 thread cap, and that R6 rides along. The
remediation worktree's `outputs/` is empty by design so the multicore tree's 15 GB stays
intact as a fallback.

**Consequence for sequencing: land every gating-tier change you will ever want in this
one batch.** After the lineage completes, the next `src/`, `scripts/*.py` or
`protocols/` edit costs another full retrain.

### 5.2 C5 (the descriptive renderer) — deferrable in principle, not in practice

No stage renders manuscript tables from the confirmatory artifacts. Post-opening you
would either hand-transcribe from `trusted/report_v1.md` or add a script — and adding a
top-level script post-lineage costs another lineage. **Add it to the current batch.**
`scripts/12_claim_stats.py:363-476` is the working template; it already emits
`development_route_a_estimand_mirror{,_cluster_sensitivity,_cluster_loco}.csv`.

### 5.3 The registry byte-pin — established, but not on Option A's path

Recorded for completeness, since you asked what an edit would force. Current registry
digest `bc3d489b6f2cfe945789a57990a15291b332ae63be6c8df0211aa2b6ff8b70b6`. Four
executable pins:

| # | Pin | Kind |
| --- | --- | --- |
| 1 | `scripts/26_validate_claims.py:72` `MODEL_MATRIX_CLAIM_REGISTRY_SHA256` | code constant; raises at `:211-214` |
| 2 | `src/thermoroute/model_matrix_amendment.py:130` inside `GOVERNANCE_SHA256` | code constant; live re-hash at `:430-445` |
| 3 | `protocols/route_a_model_matrix_amendment_v1.json:59-62` `governance_inputs.route_a_claim_registry_v1` | **sealed prelabel attestation**, created exactly once at commit `e6e369a0` |
| 4 | `tests/test_model_matrix_amendment.py:132-136` | test literal (free tier) |

Pin 3 **cannot be repaired**: `model_matrix_amendment.py:1020-1126` requires exactly one
creation commit and byte-immutability to tip, so rewriting it in place is structurally
impossible. Repair requires a **new amendment document plus a new seal in a strictly
later commit** — the established pattern used for
`route_a_numerical_policy_amendment_v2` (`61eb21b` document, `150325f` seal),
`route_a_inference_amendment_v2` (`3104b88`), and
`route_a_probability_metric_erratum_v1` (`0b04736`). Blast radius of a registry edit:
`26_validate_claims.py`, the Stage-09 model-matrix gate
(`stage09_parallel.py:880-895`), the Stage-24 model-suite freeze
(`model_suite.py:7191-7231`), Stage-24 opening authorization, Stage-28 chronology
freeze, and five test modules. **Option A avoids all of it.**

Useful precedent if a governance change is ever needed: new protocol documents are **not
required to join `GOVERNANCE_SHA256`** — that map is a closed set frozen at `e6e369a0`.
`protocols/route_a_numerical_policy_v2.json` (added 2026-08-03) is read at runtime by
format and content with **no byte pin at all**. A superseding amendment document is the
sanctioned route; editing a sealed file is not.

---

## 6. What a descriptive benchmark can draw on, post-opening

### 6.1 Available, computed unconditionally

All of these are produced regardless of the gate verdict (§2.4). Run directory is
`outputs/confirmatory/route_a_<namespace>/`.

| Quantity | Artifact | Field path |
| --- | --- | --- |
| Five-row effects, CI, p, Holm | `trusted/statistics_v1.json` | `tests[].{test_id, candidate, reference, horizon, margin_c, status, median_effect_c, ci_low_c, ci_high_c, n_stations, n_clusters, win_rate, p_one_sided_raw, p_holm, reject_at_0_05, confidence_bound_supports_margin, effect_convention, sign_flip_configurations}` |
| **Per-HUC2 effects** | `trusted/spatial_sensitivity_v1.json` | `comparisons[].per_huc[] = {huc2, n_stations, median_station_effect_c}` (`opening.py:8535-8542`) |
| **Equal-HUC summary** | `trusted/spatial_sensitivity_v1.json` | `comparisons[].equal_huc_median_effect_c` — median of the per-HUC medians (`opening.py:8543-8547`) |
| **Leave-one-HUC2 influence** | `trusted/spatial_sensitivity_v1.json` | `comparisons[].leave_one_huc[] = {held_out_huc2, n_remaining_stations, n_remaining_clusters, station_weighted_median_effect_c, effect_minus_margin_c}` (`opening.py:8564-8571`); range in `influence_min_c` / `influence_max_c` (`:8586-8587`) |
| LOHO direction stability (pass/fail) | `trusted/outcome_qc_gate_v1.json` | `leave_one_huc_direction[].{full_margin_direction, all_huc_deletions_match_full_margin_direction, pass}` (`outcome_qc.py:465-509`) |
| **Station distribution — cluster sizes** | `trusted/spatial_sensitivity_v1.json` | `comparisons[].per_huc[].n_stations` |
| **Station distribution — win fraction** | `trusted/statistics_v1.json` | `tests[].win_rate` = `mean(effect < 0)` (`opening.py:7893`) |
| **Year stability** | `trusted/temporal_coverage_audit_v1.json` | `comparison_sensitivities[].leave_one_year_equal_cell_descriptive[] = {omitted_year, descriptive_median_effect_c}` (`coverage_audit.py:1238-1298`) |
| **Season stability** | `trusted/temporal_coverage_audit_v1.json` | `comparison_sensitivities[].leave_one_season_equal_cell_descriptive[] = {omitted_season, descriptive_median_effect_c}`, DJF/MAM/JJA/SON (`coverage_audit.py:1255-1280`) |
| Qualifier-restricted target sensitivity | `trusted/approved_target_sensitivity_v1.json` | `comparisons[] = {test_id, …, effect_c, ci_low_c, ci_high_c, n_exact_a_keys}` (`opening.py:8477-8496`) |
| Cluster-size min/median/max/CV/effective-G | `trusted/report_v1.md` **(prose only)** | rendered by `_spatial_cluster_diagnostics` (`opening.py:8605-8703`), **not stored as JSON fields** |
| Raw predictions | `trusted/temporal_predictions_v1.parquet` | `model, scope, feature_set, seed, site_id, horizon, split, issue_date, target_date, y_true, y_pred, q05, q50, q95, p_exceed` (`results.py:30-36`) |

Full run-directory file set: `opening.py:1252-1305`.

### 6.2 Available today (development period only)

`/home/lzq/workspace/parttime/thermoroute-water-temperature-multicore/outputs/tables/`:
`development_route_a_estimand_mirror.csv` (6 rows, 31 columns),
`..._cluster_sensitivity.csv`, `..._cluster_loco.csv` (90 rows = 6 comparisons × 15
HUC2), plus `.md` and `outputs/figures/development_route_a_estimand_mirror.png`.

`_cluster_loco.csv` columns: `held_out_cluster` (`HUC2:01`), `held_out_station_count`,
`remaining_station_count`, `remaining_cluster_count`, `station_weighted_effect`,
`effect_minus_null_margin`.

`_cluster_sensitivity.csv` adds the full record from `significance.py:345-388`, including
`cluster_size_{min,median,max,cv}`, `largest_cluster_share`,
`effective_cluster_count_inverse_herfindahl`, `effective_cluster_fraction`,
`loco_effect_{min,max}`, `loco_max_abs_shift_from_full`, `loco_direction`,
`loco_direction_stable`, `inference_strength`, `warning_codes`.

**Gap to note:** the development mirror has **no per-HUC2 effect column and no
equal-HUC summary** — only LOCO. The post-opening `spatial_sensitivity_v1.json` has
both. So the development analogue is *narrower* than what the opening will give you.
The manuscript already quotes an equal-HUC number for the development period
(`:857`, "Region-weighted skill — the mean of the 15 per-HUC2 medians"), so that
computation exists somewhere; it is just not in the mirror CSV.

### 6.3 Not emitted — and how to get it without a code change

| Missing | Recovery |
| --- | --- |
| Per-station effect vectors | Recompute from `trusted/temporal_predictions_v1.parquet` (has `site_id`, `horizon`, `y_true`, `y_pred`, `model`). **No opening change needed.** |
| Effect quantiles / IQR across stations | Same. |
| Explicit counts of stations favouring each side | Same; or `round(win_rate × n_stations)`. |
| Cluster-size summary stats as machine-readable fields | Parse `trusted/report_v1.md`, or recompute from `per_huc[].n_stations`. |

**Governance caveat:** anything recomputed from the parquet is *outside* the trusted
scorer. `26_validate_claims.py` only byte-checks the five `RESULT_*` claim blocks; free
text is lint-checked only. So such numbers are permitted, but they are not
receipt-bound, and a reviewer asking "which of these came from the sealed scorer?"
deserves a straight answer. Recommend a table footnote distinguishing scorer-derived
from post-hoc-derived quantities.

---

## 7. The sign convention — your premise is refuted

> *"The formal estimand is `RMSE(ThermoRoute) − RMSE(reference)` … while the results text
> uses `RMSE(reference) − RMSE(ThermoRoute)`."*

**There is no sign conflict.** Code, protocol, manuscript, SI and figure spec all use
`candidate − reference`, negative favours ThermoRoute. An exhaustive search for
`RMSE(reference) −`, `reference-minus-candidate`, `positive favours`, `higher is better`
across `paper/`, `protocols/`, `src/`, `scripts/`, `docs/`, `tests/` in **both** repos
returns **zero** occurrences of the opposite convention.

| Layer | Statement | Locus |
| --- | --- | --- |
| Code | `effects = np.asarray([candidate[site] - baseline[site] for site in sites])` | `opening.py:7833` |
| Code (label) | `"effect_convention": "station_RMSE_ThermoRoute-minus-reference"` | `opening.py:7843, 7886` |
| Code (win) | `float(np.mean(effects < 0.0))` | `opening.py:7893` |
| Code | `"Effects use the convention ``candidate - reference``; negative is better."` | `significance.py:76` |
| Protocol | `"paired_effect_convention": "station_RMSE_ThermoRoute minus station_RMSE_reference; negative is better"` | `route_a_confirmatory_v1.json:350` |
| Protocol | `"alternative": "candidate_minus_reference_below_margin"` ×5 | `:396, 407, 418, 429, 440` |
| Manuscript | "the paired difference `RMSE(ThermoRoute) − RMSE(reference)`" | `paper/ThermoRoute_paper.md:611` |
| Table caption | "ΔRMSE = RMSE(ThermoRoute) − RMSE(damped). Negative ⇒ ThermoRoute better." | `scripts/07_make_tables.py:79` |
| POST figure spec | "candidate-minus-reference RMSE. Negative values favour ThermoRoute." | `paper/FIGURE_REDRAW_SPEC.md:153` |

The `+0.05 °C` margin is a **ceiling on allowable degradation** with a one-sided `less`
test and support rule `ci_high_c < margin_c` (registry `:543, 557`;
`opening.py:7905-7909`). A positive margin under a `<` test is only coherent when
positive means worse. The convention is self-consistently load-bearing.

### 7.1 What you actually saw — an undefined second metric

`src/thermoroute/metrics.py:59-60` defines a **different quantity**:

```python
def skill_score(y, yhat, ref) -> float:
    """RMSE skill vs a reference forecast (e.g. persistence): 1 - RMSE/RMSE_ref."""
```

This is legitimately **positive-is-better**, and the manuscript reports it as bare
numbers — `+0.203`, `+0.187`, `+0.251`, `+0.168`, `+0.076`, `+0.038` — at
`paper/ThermoRoute_paper.md:828-831, 857-859, 870-877, 1062, 1283` (tex `:902, 931, 958,
1154, 1351`) and in `outputs/reports/usgs_experiment.md:7-11`, **without ever defining
it**. Grep for a definition of "skill" in the manuscript returns only usages.

§4.1 places `+0.168` (skill, better) within two paragraphs of `−0.130 °C` (ΔRMSE,
better), with no formula and only the `°C` unit distinguishing them. **That is the
direction error waiting to happen** — not a sign inconsistency, a definitional gap.

### 7.2 Recommended unification — one sentence, free

Do **not** flip anything. Flipping the code would break the sealed protocol bytes, four
test assertions (`tests/test_significance_route_a.py:39-43, 63-68, 167-169`;
`tests/test_chronology.py:189`), the `opening.py:1696` and `chronology.py:367-368` string
guards, the `ci_high_c < margin_c` rule, and the `+0.05` margin semantics — a
preregistration re-seal, for zero benefit.

Instead, add near `paper/ThermoRoute_paper.md:828` (and its tex twin):

> Two differently-signed quantities are reported. The paired effect is
> `ΔRMSE = RMSE(ThermoRoute) − RMSE(reference)` in °C, where **negative** favours
> ThermoRoute. The skill score is `1 − RMSE(ThermoRoute)/RMSE(reference)`, dimensionless,
> where **positive** favours ThermoRoute. They are not interchangeable and are never
> combined.

Then audit every figure axis and table header for which of the two it carries. **This is
MANUSCRIPT-ONLY and free.** One minor hazard: `paper/WRR_FIGURE_STYLE_GUIDE.md:362` lists
`"Negative == Worse Performance"` — a quoted example caption from a *different* sample
paper, not a statement about ΔRMSE, but a figure author skimming it could invert an axis.

---

## 8. What is wrong in the framing — consolidated

| Claim | Verdict |
| --- | --- |
| HUC2/HUC4/HUC6/HUC8 numbers | **Correct**, verified to every digit via the gate's own `cluster_geometry`. |
| `>= 30` is unreachable by construction | **Correct** — 21 HUC2 regions nationally. Unreachable for *any* U.S. cohort. |
| Only HUC8 passes, and passing that way is the "subdivide" move | **Correct.** Do not do it. |
| The five comparisons are already fixed-cohort descriptive | **Correct**, and `26_validate_claims.py` will *enforce* descriptive phrasing post-opening, not fight it. |
| The change is narrative and presentational, not computational | **Correct.** `produce_trusted_opening_products` has no gate branch; the computation is byte-identical either way. |
| The gate fails *because* the cluster threshold is unreachable | **Incomplete and misleading.** `claim_eligible` is a hardcoded literal `False`. Three components fail permanently and independently. HUC8 would pass the cluster gate and change nothing. The manuscript's emphasis on the cluster gate is what invites the "ritual" reading — **fix that, and the reviewer's objection dissolves without touching the design.** |
| Zero lineage cost | **True marginally, false absolutely.** The manuscript is byte-pinned by the sealed registry and the pin is already broken (test fails today). The escape is R6, a gating-tier `scripts/*.py` edit. But R6 is already written and already inside the lineage being spent for the Stage-24 blocker, so Option A adds nothing. |
| Registry must change if the validator demands confirmatory phrasing | **Moot** — it demands the opposite. No registry edit needed. The four-pin wall is real but Option A does not hit it. |
| The sign convention is inconsistent between estimand and results text | **Refuted.** Zero occurrences of `RMSE(reference) − RMSE(ThermoRoute)` in either repo. The real defect is an **undefined skill score** (`1 − RMSE/RMSE_ref`, positive-is-better) printed next to ΔRMSE without a definition. Fix with one sentence, free. |

---

## 9. Recommended sequence

1. **Now, free:** draft the M1–M8 manuscript rewrite; add the §7.2 skill definition; fix
   `tests/test_claim_registry.py` (HARNESS tier, free).
2. **Before the lineage launches:** confirm R6 (C1) is in the remediation batch — it is.
   **Add C5** (a `scripts/31_*.py` descriptive-benchmark renderer reading
   `statistics_v1.json` + `spatial_sensitivity_v1.json`), modelled on
   `scripts/12_claim_stats.py:363-476`. After the lineage, adding it costs another
   retrain.
3. **Run the one lineage** per `docs/REMEDIATION_LINEAGE_RUNBOOK_20260805.md`.
4. **Open**, then let `26_validate_claims.py --write-generated-results` append the five
   `DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED` blocks. Fill §4 from §6.1.
5. **Do not** touch `inference_gate.py`, the claim registry, or the confirmatory
   protocol at any point.

---

*Read-only audit. No file under `protocols/`, `src/`, `scripts/`, or `tests/` was
modified. Nothing committed. All line references are to
`/home/lzq/workspace/parttime/thermoroute-remediation` at `f6bc478` plus its uncommitted
working tree, which is noted wherever it matters.*

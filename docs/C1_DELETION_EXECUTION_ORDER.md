# C1 — Apparatus deletion execution order (architecture review)

| Field | Value |
|---|---|
| Date | 2026-08-06 |
| Branch | `feat/conventional` |
| Status | READ-ONLY REVIEW DELIVERABLE — no file was modified or deleted to produce this; nothing here executes a deletion |
| Reviewed plan | `docs/APPARATUS_DELETION_PLAN.md` (384 lines) |
| Review mode | Fork A (recommended) is assumed throughout, unless a step is explicitly marked "Fork B only" |
| Verification method | `grep` import-graph sweep over `src/`, `scripts/`, `tests/`, `ops/`, `.github/`; line-range reads of every EDIT-file gate site; KEEP-closure transitive check |

This document records the architecture-review findings against the live tree,
gives a dependency-ordered execution sequence that fixes the gaps found, lists
every surgical-edit site at file:line precision, and specifies the post-deletion
verification regime.
**It does not perform any deletion or edit.**

---

## 1. Verification summary

### 1.1 What the plan got right (confirmed against the live tree)

1. **KEEP closure is apparatus-free (§0).** `conventional_score.py` imports only
   `config, checkpoint, features, frozen_inference, metrics, registry, results,
   quantiles, repro`. `repro.py` imports **only stdlib** (no `from .` at all);
   `results.py` imports **only `repro`**. The only `src/` importers of
   `model_suite` / `stage09_parallel` / `stage09b_precompute` / `predictor_bridge`
   are `opening` (DEL), `input_closure` (KEEP, not in closure),
   `development_controls_gate` (KEEP, not in closure) and `development_replay`
   (DEL) — **none inside the conventional-scorer closure.** Deleting the
   apparatus cannot break the scorer by import. ✓

2. **All six EDIT-file gate sites are real and the plan's line references are exact.**
   Every `!= source_tree_hash(root)` live-tree gate and every cross-receipt check
   the plan names was located at the cited line (see §3). In particular the four
   `model_suite.py` live-tree gates (3141 / 4696 / 6495 / 7356) all have RHS
   `source_tree_hash(root)`; the seven cross-receipt checks (4193 / 5950 / 6614 /
   7521 / 7622 / 7697-7698 / 7779-7780) all have RHS a stored receipt field. ✓

3. **KEEP-module blockers are real.** `chronology` ← `model_suite.py:56`
   (`STAGE09_ARTIFACT_PATHS`) + `09_usgs_experiment.py:142`;
   `development_controls_gate` ← `model_suite.py:57` + `09b_development_controls.py:165`;
   `model_matrix_amendment` ← `model_suite.py:67` + `stage09_parallel.py:50` +
   `stage09b_precompute.py:44`; `input_closure` ← `model_suite.py:61` +
   `stage09_parallel.py:49` + `stage09b_precompute.py:43` +
   `development_controls_gate.py:65` (plus scripts 09/09b/16/19/23/25/`_perstation_lgb`);
   `historical_inputs` ← `predictor_bridge.py:23`; `frozen_inference` ←
   `conventional_score`. ✓

4. **DEL source-module importers are in-scope** for: `development_replay`,
   `inference_gate`, `outcome_acquisition`, `outcome_qc`, `opening_contract`
   (src side), `coverage_bridge`, `coverage_audit`, `probability_metric_erratum`,
   `release_acceptance`. Each is imported only by `opening` (DEL), other DEL
   modules, DEL scripts, and DEL tests. ✓

5. **Fork-A protocol-KEEP rationale holds.** `protocols/route_a_numerical_policy_v2.json`
   is referenced by `repro.py:79`; `protocols/route_a_model_matrix_amendment_v1.json`
   is referenced by `model_matrix_amendment.py` and `chronology.py` (both KEEP
   under Fork A). Keeping both under Fork A is correct. ✓

6. **`ops/stage09/` = 8 files**, matching §1b exactly; no Python imports them. ✓

7. **`conftest.py` is already clean** — 24 lines, only a session-umask fixture,
   zero apparatus references (see Issue 6).

### 1.2 Issues found (ordered by severity)

#### ISSUE 1 — HIGH: CI workflow is omitted entirely
`.github/workflows/ci.yml` invokes DEL apparatus artefacts in **four** steps and
the plan never mentions `.github/`. Deleting the scripts without editing CI
breaks every push/PR build:

| ci.yml lines | Step | Invokes | Verdict in plan |
|---|---|---|---|
| 84-91 | "Provenance manifest round trip" | `scripts/14_manifest.py` (DEL §1c) | unaddressed |
| 92-100 | "Canonical Markdown to AGU TeX is current" | `scripts/26_validate_claims.py` (DEL §1c) | unaddressed |
| 132-150 | "Public distribution remains fail-closed" | `scripts/make_release_archive.sh` (unclassified) | unaddressed |
| 154-165 | (same step) | `scripts/verify_release.py` (DEL §1c) | unaddressed |
| 70-83 | "Repro… release contracts" | `tests/test_manifest_release.py` (DEL) + `tests/test_repro.py` (EDIT) | partially addressed |

**Fix:** add a CI-edit phase that removes/replaces these steps (see Step 1.5
below) **before** the scripts are deleted, and replace the "Public distribution
fail-closed" governance assertion with the conventional-scorer regression check.

#### ISSUE 2 — HIGH: gray-zone tests LOAD/INVOKE DEL scripts, not just source-hash gates
The plan's gray-zone instruction (§1d: "trim the source-hash assertions, keep the
scientific assertions") is **insufficient**. Several gray-zone tests load or run
DEL apparatus scripts at collection/run time. After Phase 2 deletes those
scripts, `pytest --collect-only` (Phase 3) fails before any assertion runs:

| Gray test | Sites | What it does with the DEL script |
|---|---|---|
| `test_stage09b_completion.py` | 59, 60 | `_load_script("scripts/24_freeze_model_suite.py", …)` and `_load_script("scripts/verify_release.py", …)` **at module scope** → collection break |
| `test_stage09_completion.py` | 62, 1489 | `_load_script("scripts/24_freeze_model_suite.py", …)` + subprocess `["24_freeze_model_suite.py", …]` |
| `test_formal_training_entrypoints.py` | 19,41,59,61,68,70,448,535,749,762,765 | entrypoint manifest + `.read_text()` + path checks for `24_freeze_model_suite.py` & `27_verify_development_replay.py` |
| `test_model_suite.py` | 653, 2107 | path / `.read_text()` on `24_freeze_model_suite.py` |
| `test_repro.py` | 703, 799, 800 | `_load_script_module("scripts/14_manifest.py", …)` + `_load_script_module("scripts/verify_release.py", …)` (inside the source-hash-gating region slated for deletion — OK **only if** the split removes them) |

**This is not line-trimming.** The loaded scripts produce **fixtures**
(`stage24_controls_fixture`, `stage09b_release_fixture`, etc.) that feed the
scientific assertions the plan wants to keep. Repair therefore requires
**replacing those fixtures with inline/synthetic equivalents** (or dropping the
dependent cases). The plan's effort estimate ("trim assertions, ~2-4 h") omits
this fixture-replacement engineering. It must be completed in Step 1 (before any
script is deleted) or collection breaks in Step 3.

#### ISSUE 3 — HIGH: `test_quantile_scoring_paths.py` is misclassified as gray (KEEP/EDIT)
The file is 84 lines and is **wholly coupled to `opening.py` (DEL)**: it imports
`opening` (L15) and exercises `opening._score_lightgbm_bundle`,
`OPENING._verify_file_binding`, `OPENING.load_lightgbm_bundle`,
`OPENING._confirmation_tabular_design` (L50-80). It asserts **no** source-hash
gate — it tests LightGBM quantile-scoring math through `opening` internals. It
cannot be made to survive `opening.py` deletion by "trimming source-hash
assertions." **Reclassify to DEL** (Step 3), unless the scoring path it exercises
is first ported to a kept module (e.g. `conventional_score`).

#### ISSUE 4 — MEDIUM: unclassified files reference DEL scripts
| File | References | Problem |
|---|---|---|
| `scripts/run_all.sh` | `14_manifest.py`, `24_freeze_model_suite.py`, `27_verify_development_replay.py` | Canonical training runner. Not in §1b/§1c. Under Fork A it is presumably **KEEP** (reviewer reproducibility) → must be edited to drop apparatus steps; if DEL the canonical reproduction path is lost. |
| `scripts/make_release_archive.sh` | `verify_release`, `14_manifest`, `24_freeze_model_suite`, `30_verify_release_fresh_process`, `26_validate_claims` | Governance release-archive builder. Not classified. Referenced by CI (Issue 1). Should be **DEL** (governance) and its CI step removed. |
| `tests/test_legacy_site_semantics.py` | `26_validate_claims.py` (L397), `14_manifest.py` (L600-601) | Not in §1d DEL list **and** not in the gray EDIT list — fully unclassified. Must be classified (DEL or EDIT) before Step 3. |
| `scripts/deterministic_zip.py` | — | Utility, unclassified; confirm no DEL dependency before assuming KEEP. |

#### ISSUE 5 — MEDIUM: `chronology.py` (KEEP) embeds a frozen source-file manifest listing DEL files
`chronology.py:104-125+` holds a constant listing DEL modules/scripts/tests
(`opening.py`, `opening_contract.py`, `outcome_acquisition.py`, `outcome_qc.py`,
`probability_metric_erratum.py`, `release_acceptance.py`, `24_freeze_model_suite.py`,
`26_validate_claims.py`, `28_freeze_prelabel_chronology.py`,
`30_verify_release_fresh_process.py`, `make_release_archive.sh`,
`route_a_opening_orchestrator.py`, `verify_release.py`, plus DEL tests). Under
Fork A `chronology` is KEPT, so this constant becomes **stale**. Risk: if a kept
training-pipeline code path **reads** these files (existence check / hashing)
rather than merely recording the list, deletion raises `FileNotFoundError`. (Under
Fork A the live-tree hash gate is gone, so a *changed* hash will not fail — but a
*missing file* will.) **Action during Step 1:** inspect how the manifest is
consumed; if it is read, prune the DEL entries or make the read tolerant. The plan
does not mention this.

#### ISSUE 6 — LOW: `conftest.py` trim is a no-op
Plan §3 / Phase 3 says "trim `tests/conftest.py` fixtures referencing deleted
modules." `conftest.py` (24 lines) has **zero** apparatus references — only a
session umask fixture. No trim is needed; the instruction is inaccurate (harmless).

#### ISSUE 7 — LOW: `confirmatory.py` blocker mis-cited
Plan §1a says `confirmatory` is "blocked by `09b` + `data_usgs/*` scripts."
`09b_development_controls.py` does **not** import `thermoroute.confirmatory`
(only mentions "confirmatory" in a docstring and `blind_or_confirmatory` dict
keys). The real kept blockers are `scripts/data_usgs/confirmatory_holdout.py:84`
and `scripts/data_usgs/discover_confirmatory_candidates.py:51`. KEEP verdict is
still correct; only the cited blocker is wrong.

#### ISSUE 8 — LOW: `test_repro.py` split must explicitly drop the `opening_contract` top-level import
`test_repro.py:19` does `from thermoroute import opening_contract as
opening_contract_module`, used **only** at L803 and L847 (inside the
source-hash-gating region the plan says to delete). After the split, L19 is dead
**and** `opening_contract` is DEL → `ImportError` at collection. The plan's split
description does not name this import. (L18 `chronology` import is harmless under
Fork A since `chronology` is KEEP, but is dead after the split and should go too.)

#### ISSUE 9 — LOW: §1d "gray" tag is ambiguous
The §1d block tags 7 files `**gray**` (`test_stage19_probabilistic`,
`test_quantile_scoring_paths`, `test_rev_boundary`, `test_claim_stats`,
`test_usgs_analysis_receipt`, `test_stage16_v2_consumer_gates`,
`test_information_matched_controls`) but they are **not** in the explicit
"Gray zone — EDIT, not delete" enumeration (which lists 12 *different* files).
I confirmed that, except `test_quantile_scoring_paths` (Issue 3), **none of these
7 import any DEL module**, so they will not break on collection — but their
disposition (EDIT vs DEL) is unresolved and the "21 DELETE (the 15+)" head-count
does not match the ~29 entries listed. Each must be given an explicit verdict.

---

## 2. Dependency-ordered execution sequence

Principle: **Edit-A and gray-test/CI repair first** (so deletions are
zero-identity-cost and collection stays green), then **leaves → apex** deletions.
Every step ends with the smoke import + a targeted test run; commit per step so a
break is isolated. Phase numbers track the plan; new steps are inserted where the
plan had gaps.

> Hard preconditions for the whole sequence: Phase 0 of the plan must already be
> GREEN — `scripts/conventional_holdout_2021_2023.py` produces
> `outputs/conventional/validation_2019_2020.json` (frozen-bundle predictions
> within tolerance) and `holdout_metrics_2021_2023.csv`. Do not start otherwise.

### Step 0 — GATE (prerequisite, no deletion)
| | |
|---|---|
| Action | Confirm conventional scorer end-to-end on frozen bundles |
| Files | `scripts/conventional_holdout_2021_2023.py`, `outputs/conventional/` |
| Preconditions | Phase 0 of the plan GREEN |
| Verify | `python scripts/conventional_holdout_2021_2023.py` exits 0; both artefacts exist; recorded dev-period predictions match within tolerance |

### Step 1 — Edit-A (destructive edits on 4 files) + full gray-test/CI repair
This is the load-bearing step. It must complete **before any deletion** so that
(a) the source-hash live-tree gates no longer fail, and (b) gray tests and CI no
longer reference soon-to-be-deleted scripts. Of the "6 EDIT files", only **4**
need destructive edits (`results`, `model_suite`, `stage09_parallel`,
`stage09b_precompute`); `repro` and `predictor_bridge` need **no destructive
edit** (§3.5, §3.6).

| | |
|---|---|
| Action | Neutralise the 7 live-tree `!= source_tree_hash(root)` gates (see §3); keep all 9 cross-receipt checks |
| Files (destructive) | `src/thermoroute/results.py`, `model_suite.py`, `stage09_parallel.py`, `stage09b_precompute.py` |
| Files (no destructive edit) | `src/thermoroute/repro.py`, `predictor_bridge.py` |
| Preconditions | Step 0 GREEN |
| Gray-test repair (Issue 2) | In `test_stage09b_completion.py`, `test_stage09_completion.py`, `test_formal_training_entrypoints.py`, `test_model_suite.py`, `test_repro.py`: remove source-hash assertions **and** replace every `_load_script`/`_load_script_module`/subprocess/`read_text`/path-manifest reference to a DEL script with an inline/synthetic fixture (or drop the dependent case). Drop the dead `opening_contract` (and `chronology`) top-level imports from `test_repro.py` (Issue 8) |
| Reclassify (Issue 3) | Mark `tests/test_quantile_scoring_paths.py` for deletion in Step 3 (do not edit-and-keep) |
| chronology manifest (Issue 5) | Inspect `chronology.py:104-125+` consumption; if read at runtime, prune DEL entries / make read tolerant |
| Verify | `python -c "import thermoroute.conventional_score"`; `python -c "import thermoroute.model_suite, thermoroute.stage09_parallel, thermoroute.stage09b_precompute, thermoroute.predictor_bridge, thermoroute.results, thermoroute.repro"`; `python -m py_compile src/thermoroute/results.py src/thermoroute/model_suite.py src/thermoroute/stage09_parallel.py src/thermoroute/stage09b_precompute.py`; `pytest -q tests/test_repro.py tests/test_model_suite.py tests/test_stage09_completion.py tests/test_stage09_parallel.py tests/test_stage09b_completion.py tests/test_stage09b_precompute.py tests/test_predictor_bridge.py tests/test_results.py 2>/dev/null || pytest -q` (the gray suite must collect and pass with gates neutralised) |

### Step 1.5 — CI repair (Issue 1) — NEW, not in plan
| | |
|---|---|
| Action | Remove/replace the four apparatus-dependent CI steps so CI stays green once scripts are gone |
| Files | `.github/workflows/ci.yml` |
| Edits | (a) L70-83: drop `tests/test_manifest_release.py` from the pytest list (DEL), keep `test_repro.py`/`test_checkpoint.py`/`test_prediction_schema.py`; (b) L84-91 "Provenance manifest round trip": **remove** (`14_manifest.py` DEL); (c) L92-100: remove the `26_validate_claims.py` invocation (L95), keep the `build_agu.py --check` pandoc line; (d) L132-165 "Public distribution remains fail-closed": **remove entirely** (`make_release_archive.sh` + `verify_release.py` DEL) and optionally replace with the conventional-scorer regression command |
| Preconditions | Step 1 done |
| Verify | `python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml'))"` (YAML still valid); locally dry-run the surviving steps (`ruff check src tests`, `ruff check --select F scripts`, `mypy src/thermoroute --ignore-missing-imports`, `pytest tests/test_repro.py tests/test_checkpoint.py tests/test_prediction_schema.py -q`) |

### Step 2 — Classify + delete gate scripts (leaves)
| | |
|---|---|
| Action | Resolve Issue 4 classifications, then delete the §1c apparatus scripts **and** `scripts/make_release_archive.sh` (governance) |
| Files (DEL) | `scripts/24_confirmatory_opening.py`, `route_a_opening_orchestrator.py`, `route_a_outcome_acquisition.py`, `route_a_trusted_scorer.py`, `26_validate_claims.py`, `27_verify_development_replay.py`, `28_freeze_prelabel_chronology.py`, `14_manifest.py`, `freeze_route_a_inference_gate.py`, `freeze_route_a_inference_amendment_seal.py`, `verify_release.py`, `24_freeze_model_suite.py`, `30_verify_release_fresh_process.py`, `_preopen_manuscript_guard.py`, **`make_release_archive.sh`** |
| Classify first (Issue 4) | Decide `run_all.sh` (recommended: KEEP, edit out apparatus steps now), `deterministic_zip.py` (confirm KEEP), `tests/test_legacy_site_semantics.py` (DEL or EDIT — it references `26_validate_claims.py`/`14_manifest.py`) |
| `run_all.sh` edit (if KEEP) | Remove invocations of `14_manifest.py`, `24_freeze_model_suite.py`, `27_verify_development_replay.py` |
| Preconditions | Step 1 + 1.5 done (CI no longer calls these) |
| Verify | `python -c "import thermoroute.conventional_score"`; `grep -rn "verify_release\|14_manifest\|26_validate_claims\|24_freeze_model_suite\|27_verify_development_replay\|make_release_archive" scripts/ src/ tests/ .github/` returns only expected/kept hits; `pytest --collect-only -q` still succeeds |

### Step 3 — Delete gate tests + the reclassified quantile test
| | |
|---|---|
| Action | Delete the §1d DEL test set **plus** `test_quantile_scoring_paths.py` (Issue 3); confirm `test_legacy_site_semantics.py` disposition from Step 2 |
| Files (DEL) | `test_confirmatory_opening.py`, `test_manifest_release.py`, `test_chronology.py`, `test_claim_registry.py`, `test_confirmatory_inputs.py`, `test_development_controls.py`, `test_trusted_publication.py`, `test_historical_inputs.py`, `test_coverage_audit.py`, `test_probability_metric_erratum.py`, `test_model_matrix_amendment.py`, `test_release_acceptance.py`, `test_development_replay.py`, `test_coverage_bridge.py`, `test_inference_gate.py`, `test_input_closure.py`, `test_outcome_qc.py`, `test_preopen_manuscript_guard.py`, `test_numerical_policy_v2.py`, `test_manifest_inventory.py`, `test_environmental_audit.py`, **`test_quantile_scoring_paths.py`** |
| conftest (Issue 6) | **No edit needed** — `conftest.py` is already apparatus-free |
| Preconditions | Step 2 done; gray tests already repaired in Step 1 |
| Verify | `pytest --collect-only -q` exits 0 with no `ImportError`/`ModuleNotFoundError`; `pytest -q` (surviving suite) passes |

### Step 4 — Delete enforcement protocols (respect Fork A)
| | |
|---|---|
| Action | Delete the §1e DEL list; **keep** `route_a_numerical_policy_v2.json` (consumed by `repro.py:79`) and `route_a_model_matrix_amendment_v1.json` (consumed by `model_matrix_amendment.py` + `chronology.py`) under Fork A |
| Files (DEL) | `route_a_claim_registry_v1.json` (recommend DEL per R6), `route_a_protocol_seal_v1.json`, `route_a_inference_amendment_v1.json`/`_v2.json`, `route_a_inference_amendment_seal_v1.json`/`_seal_v2.json`, `route_a_model_matrix_amendment_seal_v1.json`, `route_a_numerical_policy_amendment_v1.json`/`_v1.md`/`_v2.json`/`_v2.md`, `route_a_numerical_policy_amendment_seal_v2.json`, `route_a_outcome_qc_policy_v1.json`, `route_a_probability_metric_erratum_v1.json`/`_seal_v1.json`, `route_a_calibration_inclusion_erratum_v1.json`, `route_a_native_artifact_publication_notice_v1.md`, `route_a_native_thread_enforcement_notice_v1.md` |
| Files (KEEP, Fork A) | `route_a_confirmatory_protocol.md`, `route_a_confirmatory_v1.json`, `route_a_temporal_coverage_policy_v1.json`, `legacy_three_site_semantics_notice_v1.md`, `route_a_numerical_policy_v2.json`, `route_a_model_matrix_amendment_v1.json` |
| Preconditions | Step 3 done; live-tree gate already gone (Step 1) so protocol edits have zero identity cost |
| Verify | `python -c "import thermoroute.conventional_score"`; `grep -rn "route_a_outcome_qc_policy\|route_a_inference_amendment\|route_a_protocol_seal" src/ scripts/` returns no kept consumer |

### Step 5 — Delete `ops/stage09/*` (8 files)
| | |
|---|---|
| Action | Remove the shell harness; no Python imports it |
| Files (DEL) | `chain_stage09_multicore.sh`, `start_stage09.sh`, `start_phase2_watch.sh`, `ensure_phase2_watch.sh`, `wsl-boot-hook.sh`, `thermoroute-phase2-watch.service`, `phase2_watch.env.example`, `README.md` |
| Preconditions | Step 4 done |
| Verify | `python -c "import thermoroute.conventional_score"`; `grep -rn "stage09_multicore\|phase2_watch\|wsl-boot-hook" .github/ scripts/` returns no kept reference |

### Step 6 — Delete opening-subtree source modules (leaf-first, one commit each)
Order is reverse-topological on the **verified** import graph. `opening.py` is the
apex importer of the DEL subgraph and is imported by **no KEEP module**, so it
goes first and orphans the rest. (Confirmed: the only non-DEL importer of
`opening` was `test_quantile_scoring_paths.py`, deleted in Step 3.)

| # | Module | Can delete because its importers are already gone | Verify after |
|---:|---|---|---|
| 6.1 | `opening.py` | importers were gate scripts/tests (Step 2/3) + `test_quantile_scoring_paths` (Step 3); no KEEP module imports it | `import thermoroute.conventional_score` + `pytest -q` |
| 6.2 | `development_replay.py` | importers: `opening` (6.1), `27_verify` (Step 2), `test_development_replay` (Step 3) | same |
| 6.3 | `outcome_acquisition.py` | importers: `opening` (6.1), `route_a_outcome_acquisition` (Step 2), tests (Step 3) | same |
| 6.4 | `opening_contract.py` | importers: `opening` (6.1), `outcome_acquisition` (6.3), `test_repro` (split in Step 1 dropped the import) | same |
| 6.5 | `inference_gate.py` | importers: `opening` (6.1), freeze scripts (Step 2), `test_inference_gate` (Step 3) | same |
| 6.6 | `outcome_qc.py` | importers: `opening` (6.1), `inference_gate` (6.5), tests (Step 3) | same |
| 6.7 | `coverage_bridge.py` | importers: `opening` (6.1), `test_coverage_bridge` (Step 3) | same |
| 6.8 | `coverage_audit.py` | importers: `opening` (6.1), `coverage_bridge` (6.7), `inference_gate` (6.5), tests (Step 3) | same |
| 6.9 | `probability_metric_erratum.py` | importers: `opening` (6.1), `test_probability_metric_erratum` (Step 3) | same |
| 6.10 | `release_acceptance.py` | importers: `test_release_acceptance` (Step 3) | same |
| 6.11 | `environmental_audit.py` | importers: `test_environmental_audit` (Step 3) + `scripts/data_usgs/audit_development_environment.py` (delete in Step 2 alongside environmental_audit's other consumers; see Issue 4 note) | same |

> Issue 4 note: `scripts/data_usgs/audit_development_environment.py` imports
> `environmental_audit` and is **not** in the §1c list. It must be deleted in
> Step 2 (it is a governance leaf) **or** before 6.11, else 6.11 leaves a dangling
> importer. Recommended: add it to Step 2.

### Step 7 — DEFERRED (Fork B only)
De-entangle (Edit-B) and delete the training pipeline (`model_suite`,
`stage09_parallel`, `stage09b_precompute`, `predictor_bridge`, `09`/`09b`/`16`/`25`)
and the now-orphaned apparatus (`chronology`, `development_controls_gate`,
`model_matrix_amendment`, `input_closure`, `historical_inputs`, `confirmatory`).
**Out of scope for Fork A.** Edit-B sites (for reference): `model_suite.py:56`
(`STAGE09_ARTIFACT_PATHS`), `model_suite.py:57-59,7600`
(`validate_stage09b_completion_receipt`), `stage09_parallel.py:49-62`,
`stage09b_precompute.py:43-58`, `predictor_bridge.py:23`. Do not undertake unless
retraining is explicitly abandoned.

---

## 3. Surgical-edit precise location list (Edit-A, Fork A)

Every site below was read on the live tree. **Drop** = remove the live-tree
`!= source_tree_hash(root)` comparison. **KEEP** = cross-receipt or structural
check; do not touch.

### 3.1 `src/thermoroute/results.py` — 1 live-tree gate
- `:25` — `source_tree_hash` import. After dropping the gate, `source_tree_hash`
  is used **only** at `:172`, so the import becomes unused → remove it.
- `:168` KEEP — `panel_sha256` check.
- `:170-171` KEEP — `registry_sha256` check.
- `:172-175` **DROP** — `if require_current_source and run.get("source_sha256") != source_tree_hash(root): raise ValueError(...)`.

### 3.2 `src/thermoroute/model_suite.py` — 4 live-tree gates, 7 cross-receipt KEEP
- `:101` — `source_tree_hash` import (leave; still used by the 4 gates until dropped, then re-audit for other uses — grep shows only the 4 gate lines use it, so it can be removed after).
- **DROP (RHS = `source_tree_hash(root)`):** `:3141`, `:4696`, `:6495`, and `:7356-7359`.
  - `:7353` sets `source_digest = str(development.get("source_sha256", ""))`.
  - `:7354-7355` KEEP — 64-len structural guard `if len(source_digest) != 64: raise …`.
  - `:7356-7359` **DROP** — `if source_digest != source_tree_hash(root): raise …`.
- **KEEP (cross-receipt; RHS is a stored receipt field):** `:4193`
  (`development.get("source_sha256") != identity.get("source_sha256")`), `:5950`
  (`metadata.get("source_sha256") != identity.get("source_sha256")`), `:6614`
  (same form), `:7521` (`metadata.get("source_sha256") != source_digest`), `:7622`
  (`controls_identity.get("source_sha256") != development["source_sha256"]`),
  `:7697-7698` (`stage16_identity.get("source_sha256") != development["source_sha256"]`),
  `:7779-7780` (`stage25_identity.get("source_sha256") != development["source_sha256"]`).
- Risk control: classify every `!=` by its RHS — `source_tree_hash(root)` → drop;
  `identity[...]`/`development[...]`/`metadata[...]`/`source_digest` → keep.

### 3.3 `src/thermoroute/stage09_parallel.py` — 1 live-tree gate
- `:74` — `source_tree_hash` import.
- `:237` `def assert_unchanged` (first of two; the second at `:267` just calls `self.guard()` and has no gate).
- `:239` KEEP — `assert_formal_numerical_policy()`.
- `:240` KEEP — `self.input_closure.assert_unchanged()`.
- `:241-244` **DROP** — `if source_tree_hash(self.root) != self.identity.source_sha256: raise Stage09ParallelError("Stage-09 source tree changed after member authorization")`.
- `:245-248` KEEP — runtime-contract check `sha256_json(numerical_runtime_contract()) != self.identity.runtime_sha256`.
- `:2545` KEEP — cross-receipt `metadata.get("source_sha256") != identity.source_sha256`.

### 3.4 `src/thermoroute/stage09b_precompute.py` — 3 live-tree gates
- `:69` — `source_tree_hash` import.
- `:234-235` **DROP** — `if source_tree_hash(self.root) != self.identity.source_sha256: raise Stage09bPrecomputeError("Stage-09b source tree changed")`. (Keep `:232-233` closure check and `:236-237` runtime check.)
- `:1756-1757` **DROP** — `if source_tree_hash(repository) != identity.source_sha256: raise Stage09bPrecomputeError("work order binds another source tree")`. (Keep `:1754-1755` closure check and `:1758-1759` runtime check.)
- `:2184-2188` compound **DROP only the `:2186` operand** — remove `or source_tree_hash(repository) != identity.source_sha256`; keep the `:2185` `closure.binding_digest != identity.input_closure_sha256` and `:2187` `sha256_json(numerical_runtime_contract()) != identity.runtime_sha256` operands.

### 3.5 `src/thermoroute/predictor_bridge.py` — NO destructive edit
- `:40` import; `:1043` **records** `"source_tree_sha256": source_tree_hash(root)` (provenance, keep); `:308`/`:334`/`:366` are `source_fields` schema set-equality, **not** live-tree gates. No `!= source_tree_hash(root)` exists. Edit-A: none. (`:23` imports `historical_inputs`, KEEP under Fork A — no Edit-B.)

### 3.6 `src/thermoroute/repro.py` — NO destructive edit
- Primitives (locatable, all KEEP): `DEFAULT_SOURCE_PATTERNS :726-738`, `source_inventory :741`, `source_tree_hash :753`, `RunIdentity.source_sha256 :786`, `build_run_identity :1023-1031` (records `source_sha256` at `:1031`), schema validator `:1327-1355`. **No `!= source_tree_hash` gate lives here** — repro only provides/records the hash. Edit-A: none. (Optional R1 two-tier split of `DEFAULT_SOURCE_PATTERNS` is deferred.)

### 3.7 Gray-test repair sites (Issue 2 — not in plan's §2)
- `tests/test_stage09b_completion.py:59-60` — replace `_load_script` of `24_freeze_model_suite.py` + `verify_release.py` with inline fixtures.
- `tests/test_stage09_completion.py:62,1489` — replace `_load_script` + subprocess of `24_freeze_model_suite.py`.
- `tests/test_formal_training_entrypoints.py:19,41,59,61,68,70,448,535,749,762,765` — drop `24_freeze_model_suite.py`/`27_verify_development_replay.py` from the entrypoint manifest and remove `.read_text()`/path checks.
- `tests/test_model_suite.py:653,2107` — remove `24_freeze_model_suite.py` path/`read_text`.
- `tests/test_repro.py:703,799,800` — removed with the source-hash-gating cases; **also drop top-level imports `:18` (`chronology`) and `:19` (`opening_contract`)** (Issue 8).

---

## 4. Post-deletion verification plan

### 4.1 Compile + import smoke (after every step, mandatory after Step 6)
```
python -m py_compile $(git ls-files 'src/thermoroute/*.py')
python -c "import thermoroute.conventional_score"
python -c "import thermoroute.model_suite, thermoroute.stage09_parallel, thermoroute.stage09b_precompute, thermoroute.predictor_bridge, thermoroute.results, thermoroute.repro"
```
The first line must succeed for every surviving source file. The second is the
core safety guarantee (KEEP closure). The third confirms the Fork-A training
pipeline still imports.

### 4.2 Surviving tests that must collect and pass
KEEP (apparatus-independent, unchanged): `test_adaptive`, `test_air2stream`,
`test_checkpoint`, `test_composite_loss_units`, `test_ecology`,
`test_frozen_inference`, `test_leakage`, `test_metrics`, `test_model_contracts`,
`test_neural_baselines`, `test_probability`, `test_registry`, `test_robustness`,
`test_sample_consistency`, `test_significance_route_a`, `test_spatial_registry`,
`test_training_resume`, `test_data_evidence` (gray-EDIT, source-hash trim only).

Gray-EDIT (repaired in Step 1; must still pass with gates neutralised):
`test_model_suite`, `test_stage09_completion`, `test_stage09_parallel`,
`test_stage09b_completion`, `test_stage09b_precompute`, `test_predictor_bridge`,
`test_lgb_shards`, `test_preprocessing_estimators`, `test_formal_training_entrypoints`,
`test_prediction_schema`, `test_run_lock`, `test_repro` (split).

Pending explicit disposition (Issue 9, no DEL-module import so no collection
break, but verdict needed): `test_stage19_probabilistic`, `test_rev_boundary`,
`test_claim_stats`, `test_usgs_analysis_receipt`, `test_stage16_v2_consumer_gates`,
`test_information_matched_controls`, `test_legacy_site_semantics` (Issue 4).

Run: `pytest -q` (full surviving suite) and `pytest --collect-only -q` (must exit
0 with no `ImportError`/`ModuleNotFoundError`).

### 4.3 Scorer regression confirmation
```
python scripts/conventional_holdout_2021_2023.py
```
Must reproduce `outputs/conventional/validation_2019_2020.json` (frozen-bundle
predictions matching stored development predictions within tolerance) and
`holdout_metrics_2021_2023.csv`. Compare metrics byte-for-byte (or within
documented tolerance) against the Step-0 baseline. This is the definitive proof
that the apparatus deletion left the science intact.

### 4.4 No-dangling-reference sweep
```
grep -rn "verify_release\|14_manifest\|26_validate_claims\|24_freeze_model_suite\|27_verify_development_replay\|28_freeze_prelabel_chronology\|30_verify_release_fresh_process\|make_release_archive\|route_a_opening_orchestrator\|route_a_outcome_acquisition\|route_a_trusted_scorer\|freeze_route_a_inference\|_preopen_manuscript_guard\|opening_contract\|outcome_acquisition\|outcome_qc\|coverage_bridge\|coverage_audit\|probability_metric_erratum\|release_acceptance\|environmental_audit\|development_replay\|inference_gate" src/ scripts/ tests/ .github/ ops/
```
After Steps 2-6 the only remaining hits must be inside KEPT modules that
legitimately still reference a KEEP name (e.g. `chronology`'s stale manifest if
not pruned — flagged Issue 5) — every other hit is a dangling reference to a
deleted file and must be resolved.

### 4.5 Lint + type contracts (matches CI)
```
ruff check src tests
ruff check --select F scripts
mypy src/thermoroute --ignore-missing-imports
```

---

## 5. Bottom line

The plan's **scientific spine is sound and verified**: the KEEP closure is
apparatus-free, every EDIT-file gate site is real and correctly classified, and
the Fork-A KEEP blockers are genuine. The plan's **deletion ordering is
topologically correct**. However the plan **under-scopes the surrounding
machinery** in four HIGH/MEDIUM ways that would break the build if executed
verbatim:

1. **CI (`.github/workflows/ci.yml`) is never mentioned** yet calls 4 DEL scripts
   (Issue 1) — add Step 1.5.
2. **Gray-zone tests load/run DEL scripts as fixtures**, not merely assert
   source-hash gates (Issue 2) — Step 1 must include fixture-replacement
   engineering, not line-trimming.
3. **`test_quantile_scoring_paths.py` is coupled to `opening.py`** and must be
   reclassified DEL (Issue 3).
4. **`run_all.sh`, `make_release_archive.sh`, `audit_development_environment.py`,
   `test_legacy_site_semantics.py` are unclassified** yet reference DEL scripts
   (Issue 4); `chronology.py`'s frozen manifest goes stale (Issue 5).

With Steps 1, 1.5 and the Issue-4 classifications added, the sequence in §2 is
safe to execute under Fork A. Fork B (Step 7) remains deferred and high-risk.

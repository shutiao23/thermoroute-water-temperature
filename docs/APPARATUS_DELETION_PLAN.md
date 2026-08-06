# Apparatus deletion + build-fix plan

| Field | Value |
|---|---|
| Date | 2026-08-06 |
| Branch | `feat/conventional` |
| Status | PLAN — execute **only after** the conventional scorer is confirmed working (Phase 0) |
| Companion docs | `ROUTE_A_WORKFLOW_AUDIT_20260805.md`, `SOURCE_HASH_SCOPE_REMEDIATION_PLAN.md`, `PAPER_RESTRUCTURE_20260805.md` |
| Replacement path | `src/thermoroute/conventional_score.py` + `scripts/conventional_holdout_2021_2023.py` (already standalone) |

## 0. The single safety guarantee

`conventional_score.py` (and its entrypoint `conventional_holdout_2021_2023.py`) was
written deliberately standalone. Its transitive import closure was verified by AST
walk against the live tree:

```
KEEP closure = {conventional_score, config, checkpoint, features, frozen_inference,
  metrics, registry, results, quantiles, repro, data, datasets, thermoroute, train,
  weighting, usgs, provenance}
apparatus leaked into the closure = NONE
```

None of these modules import `opening`, `chronology`, `inference_gate`,
`development_controls_gate`, `development_replay`, `outcome_*`, `coverage_*`,
`opening_contract`, `probability_metric_erratum`, `release_acceptance`,
`environmental_audit`, `model_suite`, `stage09_parallel`, `stage09b_precompute`,
`predictor_bridge`, `input_closure`, `model_matrix_amendment`, `historical_inputs`,
or `confirmatory`. **Deleting the apparatus therefore cannot break the conventional
scorer by import.** The only shared code is `repro` and `results`, which are
surgically edited (§2), not deleted.

The audit measured the apparatus at **56,834 lines — 88 % of `src/`** (`opening.py`
12,087; `chronology.py` 4,174) wrapping a 7,610-line scientific core. This plan
deletes the governance shell and keeps the science.

### Decision fork

Two end-states are possible. **Fork A is the recommended plan** and is what §2
("edit all six") assumes.

| Fork | Training pipeline (`model_suite`, `stage09*`, `predictor_bridge`, `09`/`09b`/`16`/`25`) | `chronology`, `development_controls_gate`, `model_matrix_amendment`, `input_closure`, `historical_inputs`, `confirmatory` | Of the 6 surgical-edit files |
|---|---|---|---|
| **A (recommended)** | KEPT (reviewer reproducibility of the frozen bundles; AGU software-archive requirement) | KEPT — load-bearing for the training pipeline | all 6 EDIT (Edit-A) |
| B (aggressive, deferred §7) | DELETED (frozen bundles become the sole deliverable; retraining forfeited) | DELETED | only `repro` + `results` survive |

`chronology.py` and `development_controls_gate.py` are governance *in spirit* (the
audit names both as prime deletion targets) but are **blocked** by training-pipeline
imports (`model_suite.py:56` imports `STAGE09_ARTIFACT_PATHS`; `model_suite.py:57,7600`
imports `validate_stage09b_completion_receipt`; `09_usgs_experiment.py` and
`09b_development_controls.py` use them substantively). Under Fork A they are KEEP;
they become DELETE-eligible only under Fork B (§7).

---

## 1. DELETE-ENTIRELY vs EDIT vs KEEP classification

Legend: **DEL** = delete entirely · **EDIT** = keep, modify · **KEEP** = keep as-is.
"Blocker" = a kept module that imports this one and must be de-entangled first.

### 1a. Source modules — governance apparatus

| File | Lines | Verdict (Fork A) | Blocker / note |
|---|---:|---|---|
| `opening.py` | 12,087 | **DEL** | importers are gate scripts + gate tests (all deleted in §4.2–4.3); no KEEP module imports it |
| `development_replay.py` | 1,015 | **DEL** | importers: `opening` (DEL), `27_verify_development_replay` (DEL), `test_development_replay` (DEL) |
| `inference_gate.py` | 1,213 | **DEL** | importers: `opening` (DEL), `freeze_route_a_inference_*` (DEL), `test_inference_gate` (DEL) |
| `outcome_acquisition.py` | — | **DEL** | importers: `opening` (DEL), `route_a_outcome_acquisition` (DEL), `test_manifest_release` (DEL) |
| `outcome_qc.py` | — | **DEL** | importers: `opening` (DEL), `inference_gate` (DEL), tests (DEL) → orphaned |
| `opening_contract.py` | — | **DEL** | importers: `outcome_acquisition` (DEL), `opening` (DEL), tests (DEL) |
| `coverage_bridge.py` | — | **DEL** | importers: `opening` (DEL), tests (DEL); delete before `coverage_audit` |
| `coverage_audit.py` | — | **DEL** | importers: `opening` (DEL), `coverage_bridge` (DEL), `inference_gate` (DEL), tests (DEL) |
| `probability_metric_erratum.py` | — | **DEL** | importers: `opening` (DEL), tests (DEL) |
| `release_acceptance.py` | — | **DEL** | importers: `test_release_acceptance` (DEL) only |
| `environmental_audit.py` | — | **DEL** | importers: `audit_development_environment` (DEL), tests (DEL) |
| `chronology.py` | 4,174 | **KEEP** (Fork A) / DEL (Fork B) | blocked by `model_suite.py:56` + `09_usgs_experiment.py` |
| `development_controls_gate.py` | 1,464 | **KEEP** (Fork A) / DEL (Fork B) | blocked by `model_suite.py:57,7600` + `09b_development_controls.py` |
| `model_matrix_amendment.py` | — | **KEEP** | blocked by `model_suite`/`stage09_parallel`/`stage09b_precompute` |
| `input_closure.py` | — | **KEEP** | blocked by the whole training pipeline |
| `historical_inputs.py` | — | **KEEP** | blocked by `predictor_bridge.py:23` |
| `confirmatory.py` | — | **KEEP** | blocked by `09b` + `data_usgs/*` scripts |
| `frozen_inference.py` | — | **KEEP** | used by the conventional scorer |
| `model_suite.py` | 8,086 | **EDIT** | §2 (training-pipeline entrypoint; Fork A) |
| `stage09_parallel.py` | 2,855 | **EDIT** | §2 |
| `stage09b_precompute.py` | 2,331 | **EDIT** | §2 |
| `predictor_bridge.py` | 1,063 | **EDIT** | §2 |
| `results.py` | 273 | **EDIT** | §2 (conventional scorer depends on it) |
| `repro.py` | 1,492 | **EDIT** | §2 (conventional scorer + `checkpoint` depend on it) |

### 1b. `ops/stage09/*` — the chain/monitor harness (8 files)

| File | Verdict | Note |
|---|---|---|
| `chain_stage09_multicore.sh`, `start_stage09.sh`, `start_phase2_watch.sh`, `ensure_phase2_watch.sh`, `wsl-boot-hook.sh`, `thermoroute-phase2-watch.service`, `phase2_watch.env.example`, `README.md` | **DEL** | Pure operational harness for the multicore retrain chain. Not imported by any Python; the audit (§3.3, §5.1–5.3) documents three defects (restart-from-top, log-grep liveness, silent `nohup` failure). `ops/` is outside `DEFAULT_SOURCE_PATTERNS`, so deletion has zero identity cost. |

### 1c. Apparatus scripts

| Script | Lines | Verdict | Note |
|---|---:|---|---|
| `24_confirmatory_opening.py` | 486 | **DEL** | one-time opening launcher (governance) |
| `route_a_opening_orchestrator.py` | 61 | **DEL** | fixed `python -I` child of the opening |
| `route_a_outcome_acquisition.py` | 37 | **DEL** | raw-only NWIS acquisition child |
| `route_a_trusted_scorer.py` | 40 | **DEL** | fresh-process scorer authority |
| `26_validate_claims.py` | 2,246 | **DEL** | claim-ledger + manuscript-block validator; binds `preopen_document_sha256` (the R6 pathology) |
| `27_verify_development_replay.py` | 197 | **DEL** | full pre-confirmation replay in fresh children |
| `28_freeze_prelabel_chronology.py` | 212 | **DEL** | freezes `chronology.py` receipt |
| `14_manifest.py` | 1,061 | **DEL** | provenance manifest builder; only `test_manifest_inventory` (DEL) + `verify_release` (DEL) consume it |
| `freeze_route_a_inference_gate.py` | 62 | **DEL** | creates/verifies the inference gate |
| `freeze_route_a_inference_amendment_seal.py` | 56 | **DEL** | seal of an amendment (audit §3.7) |
| `verify_release.py` | 19,306 | **DEL** | largest single file; evidence-ZIP verifier. Confirm no KEEP reference (none) before deleting |
| `24_freeze_model_suite.py` | 545 | **DEL** | assembles the sealed model suite |
| `30_verify_release_fresh_process.py` | 255 | **DEL** | fresh-process PREOPEN release-mechanics receipt |
| `_preopen_manuscript_guard.py` | — | **DEL** | manuscript guard |

All are leaves: their only importers are the gate tests (deleted in §4.3). The
scientific/training scripts (`01`–`13c`, `15`–`25` minus `24_freeze_model_suite`,
`29`) are **KEPT**.

### 1d. Gate tests — 21 DELETE (the "15+")

Pure governance-gate tests; each imports only apparatus modules/slated scripts. All **DEL**.

`test_confirmatory_opening.py` (4,017) · `test_manifest_release.py` (12,555) ·
`test_chronology.py` (2,041) · `test_claim_registry.py` (1,579) ·
`test_confirmatory_inputs.py` (1,740) · `test_development_controls.py` (1,041) ·
`test_trusted_publication.py` (1,447) · `test_historical_inputs.py` (956) ·
`test_repro.py` (962, **partial — see gray zone**) · `test_coverage_audit.py` (829) ·
`test_probability_metric_erratum.py` (765) · `test_model_matrix_amendment.py` (723) ·
`test_release_acceptance.py` (722) · `test_development_replay.py` (684) ·
`test_stage19_probabilistic.py` (653, **gray**) · `test_coverage_bridge.py` (557) ·
`test_inference_gate.py` (521) · `test_input_closure.py` (469) · `test_outcome_qc.py` (233) ·
`test_preopen_manuscript_guard.py` (209) · `test_numerical_policy_v2.py` (120) ·
`test_manifest_inventory.py` (54) · `test_environmental_audit.py` (46) ·
`test_quantile_scoring_paths.py` (84, **gray**) · `test_rev_boundary.py` (321, **gray**) ·
`test_claim_stats.py` (210, **gray**) · `test_usgs_analysis_receipt.py` (137, **gray**) ·
`test_stage16_v2_consumer_gates.py` (122, **gray**) · `test_information_matched_controls.py` (290, **gray**)

**Gray zone — EDIT, not delete.** These test the *kept* training pipeline and
assert source-hash invariants. After Edit-A (§2) they will fail on the removed
gates; trim the source-hash assertions, keep the scientific assertions:
`test_model_suite.py` (2,121) · `test_stage09_completion.py` (2,337) ·
`test_stage09_parallel.py` (1,563) · `test_stage09b_completion.py` (1,049) ·
`test_stage09b_precompute.py` (827) · `test_predictor_bridge.py` (307) ·
`test_lgb_shards.py` (448) · `test_preprocessing_estimators.py` (434) ·
`test_formal_training_entrypoints.py` (1,138) · `test_prediction_schema.py` (118) ·
`test_run_lock.py` (328) · `test_data_evidence.py` (335).

`test_repro.py` is split: delete the `chronology`/source-hash-gating cases, keep the
`sha256_file`/`canonical_json`/atomic-write cases (those test `repro`, which is KEPT).

### 1e. `protocols/` — 24 files: enforcement vs reusable design

| File | Verdict | Rationale |
|---|---|---|
| `route_a_confirmatory_protocol.md` | **KEEP** | design record: cohort, evaluation period, estimand, gate thresholds, five-row family. Manuscript §3.6/§3.7 restates it; SI may reference it |
| `route_a_confirmatory_v1.json` | **KEEP** | frozen cohort spec (120 stations, splits) — reusable |
| `route_a_temporal_coverage_policy_v1.json` | **KEEP** | the 8 predeclared temporal-coverage candidates (manuscript §4.6 Table 4.3) |
| `legacy_three_site_semantics_notice_v1.md` | **KEEP** | the `ALLOW_LEGACY_THREE_SITE_*` allowlist referenced verbatim in manuscript §1/§2.5 |
| `route_a_claim_registry_v1.json` | **DEL** (or EDIT per R6) | binds `preopen_document_sha256` of the whole manuscript → every paper edit invalidates it (R6 pathology). Claim blocks already live byte-exact in manuscript §6.1, so the registry has no surviving consumer once `26_validate_claims.py` is deleted. Recommend DEL; keep an EDIT (bind claim-block digest + lint only) only if SI provenance requires it |
| `route_a_protocol_seal_v1.json` | **DEL** | seal of the protocol |
| `route_a_inference_amendment_v1.json`, `_v2.json` | **DEL** | amendments |
| `route_a_inference_amendment_seal_v1.json`, `_seal_v2.json` | **DEL** | seals of amendments (audit §3.7) |
| `route_a_model_matrix_amendment_v1.json` | **DEL** (Fork B) / KEEP (Fork A) | imported by kept `model_matrix_amendment` module under Fork A |
| `route_a_model_matrix_amendment_seal_v1.json` | **DEL** | seal |
| `route_a_numerical_policy_v2.json` | **DEL** (Fork B) / KEEP (Fork A) | thread-count policy; `assert_formal_numerical_policy` is called by the kept training pipeline under Fork A |
| `route_a_numerical_policy_amendment_v1.json`, `_v1.md`, `_v2.json`, `_v2.md` | **DEL** | amendments + prose |
| `route_a_numerical_policy_amendment_seal_v2.json` | **DEL** | seal of an amendment |
| `route_a_outcome_qc_policy_v1.json` | **DEL** | QC policy; consumer (`outcome_qc`) deleted |
| `route_a_probability_metric_erratum_v1.json`, `_seal_v1.json` | **DEL** | erratum + seal |
| `route_a_calibration_inclusion_erratum_v1.json` | **DEL** | erratum |
| `route_a_native_artifact_publication_notice_v1.md` | **DEL** | publication notice |
| `route_a_native_thread_enforcement_notice_v1.md` | **DEL** | thread-enforcement notice |

**KEEP = 4 design/legacy files. DEL = ~18 enforcement files** (two of those are
KEEP-under-Fork-A because the training pipeline still calls the numerical-policy /
model-matrix validators). `protocols/` is inside `DEFAULT_SOURCE_PATTERNS`, but after
Edit-A the source hash no longer gates, so protocol edits have zero identity cost.

---

## 2. Surgical edits — remove `source_sha256 != source_tree_hash` enforcement, keep logic

Principle: neutralize the **live-tree** gates (re-derive `source_tree_hash(root)`
and compare to a stored receipt — the self-defeating check). **Keep** the
**cross-receipt consistency** checks (receipt A's `source_sha256` == receipt B's
`source_sha256`), which prove two receipts belong to the same run and are legitimate
provenance. Keep `source_sha256` **recorded** in every receipt; just stop failing on
a live-tree mismatch.

### 2a. `repro.py` (KEEP) — the hash primitives
- `DEFAULT_SOURCE_PATTERNS` `repro.py:726-738`, `source_tree_hash` `:753`,
  `source_inventory` `:741`, `RunIdentity.source_sha256` `:786`,
  `build_run_identity` `:1023-1031`, schema validator `:1327-1355`.
- **No `!= source_tree_hash` gate lives here** — repro only *provides* the hash and
  *records* the field. Edit: none required for correctness. Optional (R1 two-tier):
  split `DEFAULT_SOURCE_PATTERNS` into `MODEL_SOURCE_PATTERNS` (gating, never used
  again after Edit-A) + `HARNESS_PATTERNS` (recorded). Leave `source_tree_hash`
  available as a recorded-only provenance helper.

### 2b. `results.py` (KEEP) — one live-tree gate
- import `source_tree_hash` `results.py:25`; gate `:172-175`
  (`if require_current_source and run.get("source_sha256") != source_tree_hash(root)`).
- **Edit-A:** drop the `:172-175` clause. **Keep** the scientific checks
  `panel_sha256` `:168` and `registry_sha256` `:170-171` (those prove the predictions
  came from the declared panel/registry — real provenance). Remove the now-unused
  `source_tree_hash` import if no other reference remains (only `:172` uses it).

### 2c. `model_suite.py` (EDIT) — 4 live-tree gates + 7 cross-receipt checks to keep
- import `source_tree_hash` `model_suite.py:101`.
- **Live-tree gates to neutralize:** `:3141` (Stage-9 receipt), `:4696` (Stage-16),
  `:6495` (Stage-25/external), `:7356` (development contract, with the 64-len guard
  at `:7353-7355` — keep the structural guard, drop the `!= source_tree_hash(root)`
  comparison at `:7356-7359`).
- **Cross-receipt checks to KEEP (do not touch):** `:4193`, `:5950`, `:6614`,
  `:7521`, `:7622`, `:7697-7698`, `:7779-7780` — these compare two stored
  `source_sha256` fields, proving receipts share a run.
- Risk: over-neutralization. Each `!=` must be inspected: `!= source_tree_hash(root)`
  → drop; `!= identity["source_sha256"]` / `!= development["source_sha256"]` → keep.

### 2d. `stage09_parallel.py` (EDIT) — one live-tree gate
- import `:74`; gate `:241`
  (`if source_tree_hash(self.root) != self.identity.source_sha256`).
- **Edit-A:** drop `:241-244`. Keep `input_closure.assert_unchanged()` `:240` and
  the runtime-contract check `:245-248` (real provenance). Cross-receipt check
  `:2545` stays.

### 2e. `stage09b_precompute.py` (EDIT) — three live-tree gates
- import `:69`; gates `:234`, `:1756`, and the compound clause at `:2186`
  (`... or source_tree_hash(repository) != identity.source_sha256 or ...`).
- **Edit-A:** drop `:234-235`, drop `:1756-1757`, and remove **only the**
  `source_tree_hash(repository) != identity.source_sha256` operand from the `:2184-2188`
  compound (keep the `closure.binding_digest` and `runtime_sha256` operands).

### 2f. `predictor_bridge.py` (EDIT) — no gate, recording only
- import `:40`; `:1043` records `"source_tree_sha256": source_tree_hash(root)`
  into the bridge manifest. `:334` is a `source_fields` set-equality (schema), not a
  live-tree gate.
- **Edit-A:** none — there is no `!=` enforcement to remove. Leave `:1043` as
  recorded provenance. (De-entanglement note: `:23` imports `historical_inputs`,
  which is KEEP, so no Edit-B needed.)

### 2g. Edit-B (de-entanglement) — ONLY if Fork B (§7)
Required only to delete `chronology`/`development_controls_gate`/`model_matrix_amendment`/
`input_closure`/`historical_inputs`. Concrete sites: `model_suite.py:56`
(`STAGE09_ARTIFACT_PATHS` — inline the constant), `model_suite.py:57-59,7600`
(`validate_stage09b_completion_receipt` — stub or inline), `stage09_parallel.py:49-62`,
`stage09b_precompute.py:43-58`, `predictor_bridge.py:23`. Moderate-to-high effort;
out of scope for Fork A.

---

## 3. Import graph — what breaks per deletion + build-fix

The KEEP closure (§0) imports no apparatus module, so **deletions in §4.5 cannot
break the conventional scorer**. Breaks are confined to the apparatus subgraph and
the gray-zone tests.

| Deletion | Breaks | Build-fix |
|---|---|---|
| `opening.py` | nothing in KEEP; only gate scripts/tests (already deleted) | none — verify `python -c "import thermoroute.conventional_score"` |
| `development_replay.py` | only `opening` (DEL), `27_verify` (DEL) | none |
| `inference_gate.py` | only `opening` (DEL), freeze scripts (DEL) | none |
| `outcome_acquisition.py` | only `opening` (DEL) | none |
| `outcome_qc.py` | `opening` (DEL), `inference_gate` (DEL) — delete after both | none |
| `opening_contract.py` | `outcome_acquisition` (DEL), `opening` (DEL) | none |
| `coverage_bridge.py` | `opening` (DEL) | none; delete before `coverage_audit` |
| `coverage_audit.py` | `coverage_bridge` (DEL), `inference_gate` (DEL) | none |
| `probability_metric_erratum.py` | `opening` (DEL) | none |
| `release_acceptance.py` / `environmental_audit.py` | tests (DEL) | none |
| gate scripts (§4.2) | their tests (§4.3) | none (leaves) |
| gate tests (§4.3) | `conftest.py` fixtures referencing deleted modules | trim `tests/conftest.py` |
| enforcement protocols (§4.4) | nothing (data files read only by deleted scripts) | none |
| `chronology.py` (Fork B only) | `model_suite.py:56`, `09_usgs_experiment.py` | Edit-B: inline constant + edit 09 |
| `development_controls_gate.py` (Fork B only) | `model_suite.py:57,7600`, `09b` | Edit-B: stub `validate_stage09b_completion_receipt` |
| Edit-A on the 6 files | gray-zone tests asserting source-hash gates | trim those assertions (§1d) |

Build-fix verification after each phase:
1. `python -c "import thermoroute.conventional_score"` (smoke import).
2. `python -c "import thermoroute.model_suite, thermoroute.stage09_parallel, thermoroute.stage09b_precompute, thermoroute.predictor_bridge, thermoroute.results, thermoroute.repro"` (Fork A pipeline still imports).
3. `pytest -q` on the surviving (non-deleted) tests.

---

## 4. Safe leaf-first deletion order (execute after Phase 0)

### Phase 0 — GATE (prerequisite, not deletion)
Run `scripts/conventional_holdout_2021_2023.py`. Confirm
`outputs/conventional/validation_2019_2020.json` reports frozen-bundle predictions
matching the stored development predictions within tolerance, and
`holdout_metrics_2021_2023.csv` is produced. **Do not proceed until this passes.**

### Phase 1 — Edit-A on the 6 files (§2)
Neutralize the live-tree source-hash gates in `results.py`, `model_suite.py`,
`stage09_parallel.py`, `stage09b_precompute.py` (`repro.py`/`predictor_bridge.py`
need no destructive edit). Run the surviving test suite; fix gray-zone test
assertions (§1d). Commit. This makes every later deletion zero-identity-cost.

### Phase 2 — Delete gate scripts (leaves)
`24_confirmatory_opening.py`, `route_a_opening_orchestrator.py`,
`route_a_outcome_acquisition.py`, `route_a_trusted_scorer.py`,
`26_validate_claims.py`, `27_verify_development_replay.py`,
`28_freeze_prelabel_chronology.py`, `14_manifest.py`,
`freeze_route_a_inference_gate.py`, `freeze_route_a_inference_amendment_seal.py`,
`verify_release.py`, `24_freeze_model_suite.py`,
`30_verify_release_fresh_process.py`, `_preopen_manuscript_guard.py`.
Verify no surviving test imports them; commit.

### Phase 3 — Delete gate tests + trim `conftest.py`
Delete the §1d DELETE list; trim `tests/conftest.py` fixtures referencing deleted
modules. `pytest --collect-only` must succeed. Commit.

### Phase 4 — Delete enforcement protocols
Delete the §1e DEL list (respect Fork A: keep `numerical_policy_v2.json` +
`model_matrix_amendment_v1.json` while the training pipeline calls their
validators). Commit.

### Phase 5 — Delete `ops/stage09/*`
Shell harness only; no Python imports. Commit.

### Phase 6 — Delete opening-subtree source modules (leaf-first)
Reverse-topological order, each committed separately so a breakage is isolated:
1. `opening.py` (12,087 lines — the big one)
2. `development_replay.py`
3. `outcome_acquisition.py` → `opening_contract.py`
4. `inference_gate.py`
5. `outcome_qc.py`
6. `coverage_bridge.py` → `coverage_audit.py`
7. `probability_metric_erratum.py`
8. `release_acceptance.py`, `environmental_audit.py`

After each: `python -c "import thermoroute.conventional_score"` + `pytest -q`.

### Phase 7 (DEFERRED — Fork B, optional, higher risk)
De-entangle (Edit-B) and delete the training pipeline (`model_suite`,
`stage09_parallel`, `stage09b_precompute`, `predictor_bridge`, `09`/`09b`/`16`/`25`)
and the now-orphaned apparatus (`chronology`, `development_controls_gate`,
`model_matrix_amendment`, `input_closure`, `historical_inputs`, `confirmatory`).
Only if the frozen bundles are the sole deliverable and retraining is forfeited.

---

## 5. Effort, risk, riskiest deletions

### Effort
| Phase | Effort | Risk |
|---|---|---|
| 0 (conventional scorer confirm) | already written; verification only | low |
| 1 (Edit-A, 6 files, ~10 gate sites) | 2–4 h + test run | medium (over-neutralization) |
| 2–3 (scripts + tests) | 0.5 day, mechanical | low (leaves; KEEP closure clean) |
| 4 (protocols) | 0.25 day | low (data files) |
| 5 (ops) | minutes | negligible |
| 6 (opening subtree, ~12 modules) | 0.5 day | low (verified no KEEP importer) |
| **A total** | **≈ 2 days** | **low–medium** |
| 7 (Fork B) | 2–4 days | **high** |

### Riskiest deletions / edits
1. **`model_suite.py` Edit-A** — 4 live-tree gates sit among 7 cross-receipt checks
   that look syntactically identical (`!= ... source_sha256`). Over-neutralizing a
   cross-receipt check silently lets mismatched receipts pass. Mitigation: classify
   every `!=` by its RHS (`source_tree_hash(root)` → drop; `identity[...]`/
   `development[...]` → keep) and review each diff.
2. **`chronology.py` / `development_controls_gate.py` under Fork B** — blocked by
   `model_suite` + `09`/`09b`. Edit-B de-entanglement is the highest-effort,
   highest-regret work; a stubbed `validate_stage09b_completion_receipt` can hide a
   real regression. Mitigation: stay on Fork A; defer §7 indefinitely unless
   retraining is explicitly abandoned.
3. **`protocols/route_a_claim_registry_v1.json`** — if KEPT, it re-introduces the R6
   pathology (whole-manuscript sha256 binding; every paper edit invalidates it).
   Recommend DEL (claim blocks live in manuscript §6.1) rather than EDIT.
4. **`verify_release.py` (19,306 lines)** — largest single deletion; confirm by grep
   that no KEEP script/module references it (verified: none) before removing.
5. **`results.py:172`** — must preserve the `panel_sha256`/`registry_sha256` checks
   (`:168-171`); only the `source_sha256` live-tree clause is dropped.
6. **Gray-zone tests** — `test_repro.py` is dual-purpose (core hash utils + apparatus
   gating). Split carefully: keep `sha256_file`/`canonical_json`/atomic-write cases,
   drop chronology/source-hash-gating cases.

### What this is worth
Fork A deletes ≈ 20,000 lines of governance source (`opening.py` alone is 12,087),
all 14 gate scripts (incl. the 19,306-line `verify_release.py`), ~18 enforcement
protocols, the `ops/stage09` harness, and ~20 gate tests — while leaving the
scientific core, the training pipeline, and the conventional scorer intact and
unblocked by the self-defeating source hash. The audit's measured cost driver (88 %
of `src/` being governance) is removed without forfeiting reviewer reproducibility.

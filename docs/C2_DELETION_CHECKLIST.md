# C2 — Apparatus deletion dry-run checklist (auditable, per-file)

| Field | Value |
|---|---|
| Source plan | `docs/APPARATUS_DELETION_PLAN.md` (384 lines, 2026-08-06) |
| Branch | `feat/conventional` (verified) |
| Status | **DRY-RUN CHECKLIST ONLY — nothing deleted, nothing written outside this doc** |
| Basis | Verified against the live tree at repo root on 2026-08-06 (paths exist, line counts = `wc -l`, importers = grep over `src/ scripts/ tests/ paper/ data_usgs/`) |
| Delivered | This file: ordered deletion table (a), blockers/omissions (b), per-step verification (c), KEEP list (d) |

## 0. Scope summary (verified counts)

| Category | Files | Lines | Notes |
|---|---:|---:|---|
| §6 source modules (governance apparatus) | 11 | 24,592 | plan §1a (plan left 8 of these as "—", actual counts recorded below) |
| §2 scripts (gate scripts) | 14 | 24,736 | plan §1c / §4.2 |
| §3 tests (§1d list) | 29 | 34,087 | 22 definitive (incl. `test_repro.py` partial) + 7 plan-flagged "(gray)" |
| §4 protocols (Fork A DEL) | 18 | 1,849 | plan §1e (of 24; 4 KEEP + 2 KEEP-under-Fork-A) |
| §5 `ops/stage09/*` | 8 | 827 | plan §1b |
| **Total (plan scope)** | **80** | **86,091** | `test_repro.py` counted at full 962 (partial delete) |
| + omitted from plan (B7) | 1 | 62 | `scripts/data_usgs/audit_development_environment.py` |
| **Total (incl. B7)** | **81** | **86,153** | |

Import-graph audit result: **no KEEP module imports any DEL module**. All DEL-module importers sit inside the DEL set, the gray tests, or the block list in (b). `tests/conftest.py` (24 lines) holds only a session umask fixture — **no fixture references deleted modules** (plan Phase 3 "trim conftest.py" is a verify-only no-op).

---

## (a) Ordered deletion table

Legend: `[ ]` = not done · `[x]` = done. Numbers in "引用待清点" refer to block-list items in (b). Phase order must be preserved; each phase commits separately.

### Phase 0 — GATE (prerequisite, NOT a deletion; do not proceed until it passes)

- [ ] P0-a: `PYTHONPATH=src python scripts/conventional_holdout_2021_2023.py` → confirm `outputs/conventional/validation_2019_2020.json` matches stored development predictions within tolerance and `holdout_metrics_2021_2023.csv` is produced.

### Phase 1 — Edit-A on the 6 files (prerequisite for every deletion; not deletions themselves)

| [ ] | File | Lines | Edit (from plan §2) | Precondition |
|---|---|---|---|---|
| [ ] | 1 | `src/thermoroute/results.py` | 273 | drop live-tree gate `:172-175`; keep `panel_sha256` `:168` + `registry_sha256` `:170-171`; drop now-unused `source_tree_hash` import `:25` | — |
| [ ] | 2 | `src/thermoroute/model_suite.py` | 8,086 | neutralize `:3141`, `:4696`, `:6495`, `:7356-7359` (keep the 64-len structural guard `:7353-7355`); **keep** cross-receipt checks `:4193 :5950 :6614 :7521 :7622 :7697-7698 :7779-7780` — classify every `!=` by RHS | riskiest edit (§5 risk 1) |
| [ ] | 3 | `src/thermoroute/stage09_parallel.py` | 2,855 | drop `:241-244`; keep `input_closure.assert_unchanged()` `:240`, runtime contract `:245-248`, cross-receipt `:2545` | — |
| [ ] | 4 | `src/thermoroute/stage09b_precompute.py` | 2,331 | drop `:234-235` and `:1756-1757`; in compound `:2184-2188` remove only the `source_tree_hash(repository) != identity.source_sha256` operand (keep `closure.binding_digest`, `runtime_sha256`) | — |
| [ ] | 5 | `src/thermoroute/repro.py` | 1,492 | no destructive edit required (records, never gates); optional R1 two-tier pattern split | — |
| [ ] | 6 | `src/thermoroute/predictor_bridge.py` | 1,063 | no edit required — `:1043` records only; `:334` is schema set-equality, not a gate | — |

### Phase 2 — Delete gate scripts (14 plan-listed + 1 flagged, B7). Verify no surviving test imports them; commit.

| [ ] | # | Path | Lines | Reason (plan) | 引用待清点 (precondition) |
|---|---|---|---|---|---|
| [ ] | S1 | `scripts/24_confirmatory_opening.py` | 486 | one-time opening launcher | importers: `tests/test_confirmatory_opening.py` (DEL), `tests/test_formal_training_entrypoints.py` (EDIT — B5 trim), `verify_release.py` (DEL, string), `chronology.py` (KEEP, dead constant — B8) |
| [ ] | S2 | `scripts/route_a_opening_orchestrator.py` | 61 | fixed `python -I` child of opening | importers: `test_confirmatory_opening.py` (DEL), `test_manifest_release.py` (DEL), `opening.py` (DEL, string), `chronology.py` (KEEP, dead — B8) |
| [ ] | S3 | `scripts/route_a_outcome_acquisition.py` | 37 | raw-only NWIS acquisition child | importers: `test_confirmatory_opening.py` (DEL), `test_manifest_release.py` (DEL), `opening.py` (DEL, string), `verify_release.py` (DEL, string) |
| [ ] | S4 | `scripts/route_a_trusted_scorer.py` | 40 | fresh-process scorer authority | importers: `test_manifest_release.py` (DEL), `test_confirmatory_opening.py` (DEL), `opening.py` (DEL, string), `verify_release.py` (DEL, string) |
| [ ] | S5 | `scripts/26_validate_claims.py` | 2,246 | claim-ledger + manuscript-block validator (R6 pathology) | importers: `test_claim_registry.py` (DEL), `test_manifest_release.py` (DEL), **`test_legacy_site_semantics.py` (KEEP — B4)**, **`make_release_archive.sh` (KEEP — B3)**, `verify_release.py` (DEL, string), `chronology.py` (KEEP, dead — B8) |
| [ ] | S6 | `scripts/27_verify_development_replay.py` | 197 | full pre-confirmation replay | importers: `test_development_replay.py` (DEL), `test_formal_training_entrypoints.py` (EDIT — B5), **`run_all.sh` (KEEP — B2)**, `verify_release.py` (DEL, string), `development_replay.py`/`opening.py` (DEL, string) |
| [ ] | S7 | `scripts/28_freeze_prelabel_chronology.py` | 212 | freezes `chronology.py` receipt | importers: `test_manifest_release.py` (DEL), `test_formal_training_entrypoints.py` (EDIT — B5), **`make_release_archive.sh` (KEEP — B3)**, `verify_release.py` (DEL, string), `chronology.py` (KEEP, dead — B8) |
| [ ] | S8 | `scripts/14_manifest.py` | 1,061 | provenance manifest builder | importers: `test_manifest_inventory.py` (DEL), `test_manifest_release.py` (DEL), `test_repro.py` (partial — **B6**), **`test_legacy_site_semantics.py` (KEEP — B4)**, **`run_all.sh` (KEEP — B2)**, **`make_release_archive.sh` (KEEP — B3)**, `verify_release.py` (DEL, string) |
| [ ] | S9 | `scripts/freeze_route_a_inference_gate.py` | 62 | creates/verifies the inference gate | importers: none outside DEL set — leaf |
| [ ] | S10 | `scripts/freeze_route_a_inference_amendment_seal.py` | 56 | seal of an amendment | importers: none outside DEL set — leaf |
| [ ] | S11 | `scripts/verify_release.py` | 19,306 | evidence-ZIP verifier (largest file) | importers: `test_manifest_release.py` (DEL), `test_stage09b_completion.py` (EDIT — B5), `test_repro.py` (partial — **B6**), **`make_release_archive.sh` (KEEP — B3)**, `release_acceptance.py` (DEL, string), `chronology.py` (KEEP, dead — B8); docs/paper prose references = staleness only |
| [ ] | S12 | `scripts/24_freeze_model_suite.py` | 545 | assembles the sealed model suite | importers: `test_model_suite.py` (EDIT — B5), `test_stage09_completion.py` (EDIT — B5), `test_stage09b_completion.py` (EDIT — B5), `test_formal_training_entrypoints.py` (EDIT — B5), **`run_all.sh` (KEEP — B2)**, **`make_release_archive.sh` (KEEP — B3)**, `verify_release.py` (DEL, string), `chronology.py` (KEEP, dead — B8) |
| [ ] | S13 | `scripts/30_verify_release_fresh_process.py` | 255 | fresh-process PREOPEN receipt | importers: **`make_release_archive.sh` (KEEP — B3)**, `release_acceptance.py` (DEL, string), `verify_release.py` (DEL, string), `chronology.py` (KEEP, dead — B8) |
| [ ] | S14 | `scripts/_preopen_manuscript_guard.py` | 172 | manuscript guard | **BLOCKER: `scripts/29_render_preopen_manuscripts.py` (KEEP) imports it at module level (`:28`) and calls `assert_preopen_manuscript_render_allowed` (`:52-55`)** — B1; `test_preopen_manuscript_guard.py` (DEL) |
| [ ] | S15 | `scripts/data_usgs/audit_development_environment.py` | 62 | CLI wrapper over `thermoroute.environmental_audit` — **omitted from plan lists; required for E-module deletion (B7)** | importers: none (referenced by nobody; `test_environmental_audit.py` (DEL) imports the module function, not this script) |

### Phase 3 — Delete gate tests (plan §1d list; all 29, 7 plan-flagged gray). Then trim gray-zone EDIT tests (B5) and `test_repro.py` (B6); `conftest.py` needs no trim (verified — B10). Commit.

| [ ] | # | Path | Lines | Notes |
|---|---|---|---|---|
| [ ] | T1 | `tests/test_confirmatory_opening.py` | 4,017 | imports `opening`/`outcome_acquisition`/`opening_contract` (DEL) |
| [ ] | T2 | `tests/test_manifest_release.py` | 12,555 | consumes 7 DEL scripts + `opening` |
| [ ] | T3 | `tests/test_chronology.py` | 2,041 | imports KEEP `chronology`/`checkpoint`/`repro` — plan-DEL; coverage note: verify assertions are gate/source-hash-only before deleting (EDIT candidate per §1d principle) |
| [ ] | T4 | `tests/test_claim_registry.py` | 1,579 | consumes `26_validate_claims.py` + `opening`/`outcome_qc` |
| [ ] | T5 | `tests/test_confirmatory_inputs.py` | 1,740 | imports `opening` |
| [ ] | T6 | `tests/test_development_controls.py` | 1,041 | imports KEEP `results`/`development_controls`/`checkpoint`/`repro`/`train` — same coverage note as T3 |
| [ ] | T7 | `tests/test_trusted_publication.py` | 1,447 | imports `opening`/`outcome_acquisition` |
| [ ] | T8 | `tests/test_historical_inputs.py` | 956 | imports `opening` (`:25`) |
| [ ] | T9 | `tests/test_repro.py` | 962 | **PARTIAL DELETE** — keep `sha256_file`/`canonical_json`/atomic-write cases; drop chronology/source-hash-gating cases **and** the `14_manifest.py` (`:703`) + `verify_release.py` (`:800`) loaders (B6) |
| [ ] | T10 | `tests/test_coverage_audit.py` | 829 | imports `coverage_audit` (DEL) |
| [ ] | T11 | `tests/test_probability_metric_erratum.py` | 765 | imports `probability_metric_erratum` (DEL) |
| [ ] | T12 | `tests/test_model_matrix_amendment.py` | 723 | imports KEEP `model_matrix_amendment` — coverage note (T3) |
| [ ] | T13 | `tests/test_release_acceptance.py` | 722 | imports `release_acceptance` (DEL) |
| [ ] | T14 | `tests/test_development_replay.py` | 684 | imports `development_replay` (DEL) + `27_verify` (DEL) |
| [ ] | T15 | `tests/test_stage19_probabilistic.py` | 653 | **GRAY** — imports only KEEP modules (`conformal`/`results`/`model_suite`/`probability`/`repro`), loads kept `19_probabilistic.py`; no DEL importer ⇒ safe to delete, but scientific-value check before deletion (EDIT/KEEP candidate) |
| [ ] | T16 | `tests/test_coverage_bridge.py` | 557 | imports `coverage_bridge`/`coverage_audit` (DEL) |
| [ ] | T17 | `tests/test_inference_gate.py` | 521 | imports `inference_gate`/`outcome_qc`/`coverage_audit` (DEL) |
| [ ] | T18 | `tests/test_input_closure.py` | 469 | imports KEEP `input_closure`/`provenance` — coverage note (T3) |
| [ ] | T19 | `tests/test_outcome_qc.py` | 233 | imports `outcome_qc` (DEL) |
| [ ] | T20 | `tests/test_preopen_manuscript_guard.py` | 209 | consumes `_preopen_manuscript_guard.py` (DEL) |
| [ ] | T21 | `tests/test_numerical_policy_v2.py` | 120 | numerical-policy gate test |
| [ ] | T22 | `tests/test_manifest_inventory.py` | 54 | consumes `14_manifest.py` (DEL) |
| [ ] | T23 | `tests/test_environmental_audit.py` | 46 | imports `environmental_audit` (DEL) |
| [ ] | T24 | `tests/test_quantile_scoring_paths.py` | 84 | **GRAY** — imports `opening` (DEL) directly ⇒ **cannot survive**; delete |
| [ ] | T25 | `tests/test_rev_boundary.py` | 321 | **GRAY** — imports KEEP `decision`/`repro`, loads kept `18_rev_curve.py`; no DEL importer ⇒ safe to delete, scientific-value check first |
| [ ] | T26 | `tests/test_claim_stats.py` | 210 | **GRAY** — loads kept `12_claim_stats.py`; no DEL importer ⇒ safe to delete, scientific-value check first |
| [ ] | T27 | `tests/test_usgs_analysis_receipt.py` | 137 | **GRAY** — loads kept `10_usgs_analysis.py`; no DEL importer ⇒ safe to delete, scientific-value check first |
| [ ] | T28 | `tests/test_stage16_v2_consumer_gates.py` | 122 | **GRAY** — pure AST test of kept `15/18/20/22` scripts; no DEL importer ⇒ safe to delete, scientific-value check first |
| [ ] | T29 | `tests/test_information_matched_controls.py` | 290 | **GRAY** — imports KEEP `development_controls`/`neural_baselines`; no DEL importer ⇒ safe to delete, scientific-value check first |

### Phase 4 — Delete enforcement protocols (Fork A: keep `numerical_policy_v2.json` + `model_matrix_amendment_v1.json`). Commit.

| [ ] | # | Path | Lines | Reason / consumer |
|---|---|---|---|---|
| [ ] | P1 | `protocols/route_a_claim_registry_v1.json` | 677 | R6 pathology; consumer `26_validate_claims.py` (DEL) |
| [ ] | P2 | `protocols/route_a_protocol_seal_v1.json` | 38 | seal of the protocol; consumers all DEL (`opening.py:1710,1922`, `probability_metric_erratum.py:379`, `inference_gate.py:49`) + **KEEP `chronology.py:41,3842,3973` dead constant — B8** |
| [ ] | P3 | `protocols/route_a_inference_amendment_v1.json` | 142 | amendment; consumer `opening.py` (DEL) |
| [ ] | P4 | `protocols/route_a_inference_amendment_v2.json` | 183 | amendment |
| [ ] | P5 | `protocols/route_a_inference_amendment_seal_v1.json` | 1 | seal |
| [ ] | P6 | `protocols/route_a_inference_amendment_seal_v2.json` | 1 | seal |
| [ ] | P7 | `protocols/route_a_model_matrix_amendment_seal_v1.json` | 1 | seal; validator in KEEP `model_matrix_amendment.py:1083` — verify no live call after Edit-A (Fork A: seal not consumed by kept pipeline) |
| [ ] | P8 | `protocols/route_a_numerical_policy_amendment_v1.json` | 61 | amendment |
| [ ] | P9 | `protocols/route_a_numerical_policy_amendment_v1.md` | 56 | prose |
| [ ] | P10 | `protocols/route_a_numerical_policy_amendment_v2.json` | 83 | amendment |
| [ ] | P11 | `protocols/route_a_numerical_policy_amendment_v2.md` | 55 | prose |
| [ ] | P12 | `protocols/route_a_numerical_policy_amendment_seal_v2.json` | 36 | seal |
| [ ] | P13 | `protocols/route_a_outcome_qc_policy_v1.json` | 84 | QC policy; consumer `outcome_qc.py:256` (DEL) |
| [ ] | P14 | `protocols/route_a_probability_metric_erratum_v1.json` | 154 | erratum; consumer `probability_metric_erratum.py` (DEL) |
| [ ] | P15 | `protocols/route_a_probability_metric_erratum_seal_v1.json` | 1 | seal |
| [ ] | P16 | `protocols/route_a_calibration_inclusion_erratum_v1.json` | 81 | erratum |
| [ ] | P17 | `protocols/route_a_native_artifact_publication_notice_v1.md` | 121 | publication notice |
| [ ] | P18 | `protocols/route_a_native_thread_enforcement_notice_v1.md` | 74 | thread-enforcement notice |

### Phase 5 — Delete `ops/stage09/*` (shell harness; zero Python importers verified). Commit.

| [ ] | # | Path | Lines |
|---|---|---|---|
| [ ] | O1 | `ops/stage09/chain_stage09_multicore.sh` | 114 |
| [ ] | O2 | `ops/stage09/start_stage09.sh` | 108 |
| [ ] | O3 | `ops/stage09/start_phase2_watch.sh` | 330 |
| [ ] | O4 | `ops/stage09/ensure_phase2_watch.sh` | 128 |
| [ ] | O5 | `ops/stage09/wsl-boot-hook.sh` | 23 |
| [ ] | O6 | `ops/stage09/thermoroute-phase2-watch.service` | 23 |
| [ ] | O7 | `ops/stage09/phase2_watch.env.example` | 25 |
| [ ] | O8 | `ops/stage09/README.md` | 76 |

### Phase 6 — Delete opening-subtree source modules (reverse-topological, one commit each)

| [ ] | # | Path | Lines | Reason (plan) | 引用待清点 (precondition) |
|---|---|---|---|---|---|
| [ ] | M1 | `src/thermoroute/opening.py` | 12,087 | the big one; importers are all DEL (gate scripts/tests) | verified: no KEEP importer anywhere (src/scripts/tests/paper/data_usgs) |
| [ ] | M2 | `src/thermoroute/development_replay.py` | 1,015 | importers: `opening` (DEL, `:69`), `27_verify` (DEL), `test_development_replay` (DEL) | delete after M1 |
| [ ] | M3 | `src/thermoroute/outcome_acquisition.py` | 2,629 | importers: `opening` (DEL, `:10210 :11488 :11763 :5625 :7933`), `route_a_outcome_acquisition` (DEL), tests (DEL) | delete after M1 |
| [ ] | M4 | `src/thermoroute/opening_contract.py` | 1,108 | importers: `outcome_acquisition` (DEL, `:30`), `opening` (DEL, `:112`), `test_repro` (partial — B6 must drop `:19` import), `test_confirmatory_opening` (DEL) | delete after M3 |
| [ ] | M5 | `src/thermoroute/inference_gate.py` | 1,213 | importers: `opening` (DEL, `:83`), freeze scripts (DEL), `test_inference_gate` (DEL) | delete after M1 |
| [ ] | M6 | `src/thermoroute/outcome_qc.py` | 1,098 | importers: `opening` (DEL, `:126`), `inference_gate` (DEL, `:30`), tests (DEL) | delete after M1+M5 |
| [ ] | M7 | `src/thermoroute/coverage_bridge.py` | 982 | importers: `opening` (DEL, `:65`), `coverage_audit` (DEL, `:29`), tests (DEL) | delete BEFORE M8 |
| [ ] | M8 | `src/thermoroute/coverage_audit.py` | 1,609 | importers: `opening` (DEL, `:58`), `coverage_bridge` (DEL, `:29`), `inference_gate` (DEL, `:36`), tests (DEL) | delete after M7 |
| [ ] | M9 | `src/thermoroute/probability_metric_erratum.py` | 1,056 | importers: `opening` (DEL, `:140`), tests (DEL) | delete after M1 |
| [ ] | M10 | `src/thermoroute/release_acceptance.py` | 1,462 | importers: `test_release_acceptance` (DEL) only | — |
| [ ] | M11 | `src/thermoroute/environmental_audit.py` | 333 | importers: `audit_development_environment.py` (**B7 — must delete first**), `test_environmental_audit` (DEL) | delete after S15 |

---

## (b) 阻断项与遗漏清单

1. **[阻断] `_preopen_manuscript_guard.py` vs `29_render_preopen_manuscripts.py`** — plan lists the guard as DEL (S14) and `29` as KEPT, but `scripts/29_render_preopen_manuscripts.py:28` imports it at module level and calls `assert_preopen_manuscript_render_allowed` before any python-docx import (`:52-55`, `:1263`). Deleting the guard breaks 29 at import time. **Resolution required before S14**: surgical edit of 29 to drop the guard import/call (guard is PRE-state enforcement — governance), or keep the guard. Plan silent on either.
2. **[阻断] `scripts/run_all.sh` (KEPT) invokes 3 DEL scripts** — `14_manifest.py` (`:71 :141-142`), `24_freeze_model_suite.py` (`:130`), `27_verify_development_replay.py` (`:137 :139`). Not in plan. **Resolution**: remove those invocation sites as part of Phase 2 (canonical training pipeline otherwise fails mid-run).
3. **[阻断/遗漏] `scripts/make_release_archive.sh` (KEPT, unlisted) invokes 6 DEL scripts** — `verify_release.py` (`:116 :252 :255 :258 :287`), `24_freeze_model_suite.py` (`:154`), `28_freeze_prelabel_chronology.py` (`:155`), `30_verify_release_fresh_process.py` (`:156`), `26_validate_claims.py` (`:157`), `14_manifest.py` (`:276 :278`). It builds/verifies the governance release archive — plan never mentions it. **Resolution**: decide DEL (release-archive machinery is governance) or full surgical edit; must be settled before Phase 2.
4. **[阻断] `tests/test_legacy_site_semantics.py` (KEPT, unlisted in any plan list) breaks at Phase 2** — `:397` reads `scripts/26_validate_claims.py` as part of a generator-lint assertion; `:600-601` asserts `run_all.sh` contains `14_manifest.py --check-route-a-boundary` / `--development-prelabel`. **Resolution**: edit both tests (and run_all.sh per B2) — not covered by plan.
5. **[阻断 (coordination)] gray-zone EDIT tests reference DEL scripts and must be trimmed in Phase 1/3** — `test_formal_training_entrypoints.py` (`:18-21 :37-70 :448 :535 :672 :749-765`: `24_confirmatory_opening`, `24_freeze_model_suite`, `27_verify_development_replay`, `28_freeze_prelabel_chronology`), `test_model_suite.py` (`:653 :2107`), `test_stage09_completion.py` (`:62 :1489`), `test_stage09b_completion.py` (`:59-60`). Plan §3 mentions "trim gray-zone assertions" but does not enumerate these script references.
6. **[阻断 (coordination)] `test_repro.py` partial-delete scope** — plan §1d says keep `sha256_file`/`canonical_json`/atomic-write cases; verified it ALSO loads `scripts/14_manifest.py` (`:703`) and `scripts/verify_release.py` (`:800`) and imports `opening_contract` (`:19`) — all DEL. Those cases/imports must be dropped in the same partial edit.
7. **[遗漏] `scripts/data_usgs/audit_development_environment.py` (62 lines)** — plan §1a names it as `environmental_audit`'s importer "(DEL)" but it appears in neither §1c nor §4.2. Verified: only importer of `thermoroute.environmental_audit` outside tests. **Add to Phase 2 (S15)**, otherwise M11 breaks.
8. **[待清点] `chronology.py` (KEEP) retains dead gate constants** — `REQUIRED_GATE_PATHS` (`:100-120`) lists 16+ DEL paths (incl. `opening.py`, `opening_contract.py`, `outcome_acquisition.py`, `outcome_qc.py`, `probability_metric_erratum.py`, `release_acceptance.py`, 8 DEL scripts incl. `make_release_archive.sh`, 8 DEL tests) and `DEFAULT_PROTOCOL_SEAL` (`:41`) = DEL protocol P2. Verified live callers of these validators are only `opening.py`/`verify_release.py`/`24_confirmatory_opening.py`/`28_freeze_prelabel_chronology.py` — all DEL; the kept pipeline uses only `chronology.STAGE09_ARTIFACT_PATHS` (`model_suite.py:56`, `09_usgs_experiment.py:142`). ⇒ constants become dead code; **no runtime blocker, but verify post-Phase 4/6** and consider cleanup commit.
9. **[数量核对] plan §1d says "21 DELETE" but lists 29 files** — 22 definitive (incl. partial `test_repro.py`) + 7 flagged "(gray)" (`T15 T24 T25 T26 T27 T28 T29`). Only `T24 test_quantile_scoring_paths.py` imports a DEL module (`opening`); the other 6 gray tests import only KEEP modules and test kept scripts (`19/18/12/10` + `15/18/20/22` consumer gates + `development_controls`/`neural_baselines`) — safe to delete, but each warrants a scientific-value check (EDIT/KEEP per §1d gray-zone principle) before deletion.
10. **[核实] plan Phase 3 "trim `tests/conftest.py`"** — verified `conftest.py` (24 lines) contains only the session umask fixture; **no fixture references deleted modules**. No trim needed; `pytest --collect-only` remains the gate.
11. **[覆盖提示, non-blocking]** plan-DEL tests `T3/T6/T12/T18` import KEEP modules (`chronology`, `results`/`development_controls`/`checkpoint`/`repro`/`train`, `model_matrix_amendment`, `input_closure`/`provenance`) and assert gate/source-hash invariants. Deleting them removes coverage of kept science — verify each contains no surviving scientific assertion before delete; EDIT candidates per §1d principle.
12. **[非代码]** `paper/agu_submission/README.md` (`:12 :179`), `docs/*` prose reference `verify_release.py`, `26_validate_claims.py`, `route_a_claim_registry_v1.json` etc. — documentation staleness only, no import impact; schedule doc follow-up separately.

---

## (c) Verification commands per step (read-only safe; run after each committed step)

Phase 0 (gate):
- `PYTHONPATH=src python scripts/conventional_holdout_2021_2023.py` → inspect `outputs/conventional/validation_2019_2020.json` + `outputs/conventional/holdout_metrics_2021_2023.csv`

Phase 1 (after each Edit-A file):
- `python -m py_compile src/thermoroute/results.py src/thermoroute/model_suite.py src/thermoroute/stage09_parallel.py src/thermoroute/stage09b_precompute.py src/thermoroute/repro.py src/thermoroute/predictor_bridge.py`
- `python -c "import thermoroute.conventional_score"`
- `python -c "import thermoroute.model_suite, thermoroute.stage09_parallel, thermoroute.stage09b_precompute, thermoroute.predictor_bridge, thermoroute.results, thermoroute.repro"`
- `pytest -q tests/test_model_suite.py tests/test_stage09_completion.py tests/test_stage09_parallel.py tests/test_stage09b_completion.py tests/test_stage09b_precompute.py tests/test_predictor_bridge.py tests/test_lgb_shards.py tests/test_preprocessing_estimators.py tests/test_formal_training_entrypoints.py tests/test_prediction_schema.py tests/test_run_lock.py tests/test_data_evidence.py tests/test_repro.py` (gray-zone EDIT suite, post-trim)

Phase 2 (after each script, then batch):
- smoke: `python -c "import thermoroute.conventional_score"` and the 6-module import line above
- after B2/B3 edits: `bash -n scripts/run_all.sh scripts/make_release_archive.sh`
- after B4 edit: `pytest -q tests/test_legacy_site_semantics.py`
- after B5 trim: `pytest -q tests/test_formal_training_entrypoints.py`
- `grep -rn '26_validate_claims\|14_manifest\|verify_release\|24_freeze_model_suite\|27_verify_development_replay\|28_freeze_prelabel_chronology\|30_verify_release_fresh_process\|24_confirmatory_opening\|route_a_opening_orchestrator\|route_a_outcome_acquisition\|route_a_trusted_scorer\|freeze_route_a_inference\|_preopen_manuscript_guard' src/ scripts/ tests/` → expect only B8 dead constants + docs

Phase 3:
- `pytest --collect-only -q` (must succeed; conftest untouched — B10)
- `pytest -q` (full surviving suite)
- `grep -rn 'audit_development_environment' scripts/ tests/` → expect only the deleted script itself

Phase 4:
- `grep -rn -E 'route_a_claim_registry|route_a_protocol_seal|inference_amendment|numerical_policy_amendment|outcome_qc_policy|probability_metric_erratum|calibration_inclusion|native_artifact|native_thread' src/ scripts/ tests/` → expect only `chronology.py` dead constants (B8) + KEEP `numerical_policy`/`model_matrix_amendment` references
- smoke imports (both lines above)

Phase 5:
- `ls ops/stage09/` → empty; `bash -n` remaining shell scripts in repo

Phase 6 (after EACH module deletion):
- `python -c "import thermoroute.conventional_score"`
- `python -c "import thermoroute.model_suite, thermoroute.stage09_parallel, thermoroute.stage09b_precompute, thermoroute.predictor_bridge, thermoroute.results, thermoroute.repro"`
- `pytest -q` (surviving suite)
- after M1: `grep -rn 'thermoroute.opening\|from .opening' src/ scripts/ tests/` → no matches

---

## (d) KEEP files (Fork A)

**Source modules (all `src/thermoroute/`)** — science core + scorer:
- `conventional_score.py` (495), `frozen_inference.py` (728), `config.py`, `checkpoint.py`, `features.py`, `metrics.py`, `registry.py`, `quantiles.py`, `data.py`, `datasets.py`, `thermoroute.py`, `train.py`, `weighting.py`, `usgs.py`, `provenance.py`, `adaptive.py`, `air2stream.py`, `baselines.py`, `conformal.py`, `decision.py`, `development_controls.py`, `ecology.py`, `evidence.py`, `lgb_shards.py`, `neural_baselines.py`, `nwp.py`, `probability.py`, `robustness.py`, `significance.py`, `spatial.py`
- Training-pipeline entrypoints (**EDIT** per §2): `model_suite.py` (8,086), `stage09_parallel.py` (2,855), `stage09b_precompute.py` (2,331), `predictor_bridge.py` (1,063), `results.py` (273), `repro.py` (1,492)
- Fork-A load-bearing governance-in-spirit (KEEP; DELETE-eligible only under Fork B §7): `chronology.py` (4,174), `development_controls_gate.py` (1,464), `model_matrix_amendment.py` (1,355), `input_closure.py` (862), `historical_inputs.py` (1,572), `confirmatory.py` (641)

**Scripts (`scripts/`)**:
- `01_prepare_data.py` … `13c_region_transfer.py`, `15_stratified.py` … `25_train_external_pooled_suite.py` (minus `24_freeze_model_suite.py`), `29_render_preopen_manuscripts.py` (subject to B1), `conventional_holdout_2021_2023.py` (407), `ci_smoke.py`, `deterministic_zip.py`, `_legacy_site_semantics.py`, `run_all.sh` (subject to B2), `make_release_archive.sh` (subject to B3), `data_usgs/*` (except `audit_development_environment.py` — subject to B7)

**Tests (`tests/`)**: everything not in the §1d list, incl. gray-zone EDIT 12 (`test_model_suite`, `test_stage09_completion`, `test_stage09_parallel`, `test_stage09b_completion`, `test_stage09b_precompute`, `test_predictor_bridge`, `test_lgb_shards`, `test_preprocessing_estimators`, `test_formal_training_entrypoints`, `test_prediction_schema`, `test_run_lock`, `test_data_evidence`), the 7 gray tests pending decision (T15/T24/T25/T26/T27/T28/T29), `conftest.py` (untouched)

**Protocols (`protocols/`)**: `route_a_confirmatory_protocol.md`, `route_a_confirmatory_v1.json`, `route_a_temporal_coverage_policy_v1.json`, `legacy_three_site_semantics_notice_v1.md`; Fork-A KEEP: `route_a_numerical_policy_v2.json`, `route_a_model_matrix_amendment_v1.json`

**`ops/`**: none — `ops/stage09/*` (8 files) all DEL (Phase 5)

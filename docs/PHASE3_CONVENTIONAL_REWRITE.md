# Phase 3 — Text cleanup record (conventional design rewrite)

**Date:** 2026-08-07
**Scope:** README.md and docs/ only. No code, tests, scripts, paper/, or .github/
files were touched. Nothing was committed.

This record documents the Phase 3 text cleanup: removing the Route-A sealed
pre-registration narrative (seals, one-time opening, receipts, claim registry,
inference gates) from the repository's user-facing text and retiring the
apparatus-era design documents.

## 1. README.md — rewritten

The old README described the repository as a "byte-frozen pre-opening evidence
snapshot" whose 2021–2023 labels "remain sealed until all gates pass" — a
statement already contradicted by the scored artifacts in
`outputs/conventional/`. The new README is a single conventional-design
narrative following data → model → prediction → statistics → figures:

- **Project intro** — retrospective daily river-water-temperature hindcasting
  benchmark; conventional independent temporal holdout (development
  2006–2020, holdout 2021–2023).
- **Held-out results table** — RMSE and skill vs persistence /
  damped-persistence at 1/3/7 d, traced to
  `outputs/conventional/holdout_metrics_2021_2023.csv`.
- **Study design** — development panel, station registry, splits table,
  ≥100-target reportability rule, feature order, pooled-training arms
  described as same-site sensitivity (not site-disjoint).
- **Model design** — four-piece forecast, CQR + Platt calibration details,
  five-seed ensembles, ablations, 7DADM descriptive analysis.
- **Scoring workflow** — `conventional_holdout_2021_2023.py` (2019–2020
  reproduction tie-back → 2021–2023 acquisition from the snapshot cache →
  validation checks → persistence), then `conventional_derive_statistics.py`
  (pure-statistics no-re-run guarantee), plus `build_results_table.py` and
  `verify_holdout_metrics.py`.
- **Output artifacts** — the `outputs/conventional/` file contract.
- **Bundle provenance** — `data_usgs/model_bundle_manifest_v1.json`,
  `scripts/verify_model_bundles.py`, `--bundle-root` /
  `THERMOROUTE_BUNDLE_ROOT`.
- **Repository layout + module index** — the KEEP modules of
  `src/thermoroute/` with one-line purposes.
- **Tests/verification commands, reproducibility boundary, license.**

Removed from the old README (all governance or apparatus-era content):
byte-frozen pre-opening snapshot framing; evidence workflow (seal/receipt/
opening ordering); temporal-coverage-audit section tied to the claim machinery;
the public-remote governance stop and release-archive distribution rules
(still recorded in the kept `docs/RIGHTS_*` and `docs/FAIR_*` files); the
`protocols/` registry links; "fail-closed" apparatus wording. Kept scientific
uses: conformal prediction, calibration, frozen bundles, station registry.

## 2. docs/ — 33 files retired

### 2a. Moved to `docs/archive/` (21 files, `git mv`)

Apparatus mechanics or abandoned designs with no reader value in the
conventional design; content preserved unchanged:

`B02_PERMANENT_DESCRIPTIVE_CLAIM_WORDING.md`, `BLUEPRINT_EXECUTION_LEDGER.md`,
`CLEANROOM_REPRODUCTION_DESIGN.md`, `M01_REGISTRY_STATUS_RENAME_DEFERRED.md`,
`OPTION_A_DESCRIPTIVE_BENCHMARK_SCOPE.md`, `P0_A0_GOVERNANCE_CLOSURE_DESIGN.md`,
`P1_PRELABEL_STRATA_REGISTRY_DRAFT.md`, `POST_PAPER_PROJECTION_DESIGN.md`,
`REMEDIATION_LINEAGE_RUNBOOK_20260805.md`, `REPORTING_POLICY_DESCRIPTIVE_ONLY.md`,
`ROUTE_A_H1_H3_CLARIFICATION_DRAFT.md`, `ROUTE_B_COMPARATOR_PROVENANCE_20260801.md`,
`ROUTE_B_GE30_CLUSTER_SAMPLING_FRAME_DRAFT.md`,
`ROUTE_B_MEASUREMENT_MISSINGNESS_UQ_DRAFT.md`,
`ROUTE_B_MODEL_BUDGET_IDENTIFIABILITY_DRAFT.md`,
`ROUTE_B_OWNER_CUSTODIAN_DECISION_FORM.md`, `ROUTE_B_POWER_MDE_DRAFT.md`,
`ROUTE_B_PRELABEL_DECISION_PACKAGE.md`,
`SOURCE_HASH_SCOPE_REMEDIATION_PLAN.md`,
`STAGE09_CROSS_SOURCE_PROVENANCE_BRIDGE_DESIGN.md`,
and the directory `RUN_PACKAGE_20260802/` (README.md + preflight_env.sh).

### 2b. Marked as historical records (12 files)

Added the line `> Historical record (superseded by the conventional design).`
at the top of each; historical facts unchanged:

`AIR2STREAM_SOURCE_BUILD_AUDIT_20260801.md`, `BB02498A_VOID_EVENT.md`,
`MULTICORE_V2_LINEAGE_STATUS_20260803.md`, `PRE_POST_STATIC_CLAIM_AUDIT_20260801.md`,
`ROUTE_A_WORKFLOW_AUDIT_20260805.md`, `ROUTE_B_SUSPENSION_20260805.md`,
`SI_STATIC_INTEGRITY_AUDIT_20260802.md`, `STAGE09B_CONFIG_SHAPE_FAILURE_20260802.md`,
`STAGE09B_PROTECTED_CHANGE_WORK_ORDER.md`,
`STAGE19_DEGENERATE_INTERVAL_DISPOSITION_20260805.md`,
`STAGE24_REPLAY_RUNTIME_DEFECT_20260805.md`,
`STAGE24_WORKER_THREAD_CAP_BLOCKER_20260805.md`.

### 2c. Kept in place (26 files) — no marking

Active working documents for the current effort. These intentionally still
contain governance-era vocabulary (they describe the transition or the old
state) and are listed here as *intentionally retained*:

- Deletion-process docs: `APPARATUS_DELETION_PLAN.md`, `C1_DELETION_EXECUTION_ORDER.md`,
  `C2_DELETION_CHECKLIST.md`, `CODE_FREEZE_DISCIPLINE_20260805.md` (its
  "in force until the one-time opening completes" header line is stale; the
  git-restore/safe-path discipline remains in use per the plan's Trap 1).
- Manuscript/figure working docs: `B3_NUMBER_PLACEHOLDER_MAP.md`,
  `PAPER_BENCHMARK_RESTRUCTURE.md`, `PAPER_RESTRUCTURE_20260805.md`,
  `PAPER_POLISH_20260805.md`, `PAPER_FIGURE_SI_RECONCILIATION.md`,
  `FIGURE_PLAN_STAGE19_INDEPENDENT_20260805.md`,
  `FIG01_RESTRUCTURE_RERENDER_20260806.md`, `WRR_SUBMISSION_CHECKLIST.md`
  (its §3.7 "one-time acquisition" row belongs to another work stream),
  `WRR_FIGURE_STYLE_TEARDOWN.md`, `M01_M14_M15_MANUSCRIPT_BATCH_DEFERRED.md`,
  `AGU2025_TEMPLATE_MIGRATION.md`, `LITERATURE_EVIDENCE_MAP_2021_2026.md`
  (Route A/B there are arm labels, not governance).
- Rights/FAIR docs: `FAIR_EXTERNAL_CLOSEOUT_REQUEST_PACK.md`,
  `FAIR_RIGHTS_ACCEPTANCE_MATRIX.md`, `FAIR_SUBMISSION_READINESS_AND_TEMPLATES.md`,
  `R10_AGU2025_RELEASE_ASSETS.md`, `RIGHTS_BYTE_HISTORY_AUDIT_20260801.md`,
  `RIGHTS_CRITICAL_PATH.md`, `RIGHTS_INVENTORY_DRAFT.md`,
  `RIGHTS_PROVIDER_EVIDENCE_20260801.md`.
- Process docs: `WORKTREE_MERGE_RUNBOOK_20260805.md`.

## 3. Remaining governance vocabulary (intentional)

| Location | Term | Why retained |
|---|---|---|
| `docs/C1_*`, `docs/C2_*`, `docs/APPARATUS_DELETION_PLAN.md` | seal/opening/receipt/claim registry | They are the live deletion work orders for the code deletion; they describe what is being removed. They become historical once the deletion completes. |
| `docs/CODE_FREEZE_DISCIPLINE_20260805.md` | opening | Stale end-condition; kept for the safe-path discipline. |
| `docs/WRR_SUBMISSION_CHECKLIST.md` | one-time acquisition/opening receipt | Active submission tracker owned by the manuscript work stream; rows pending re-mapping. |
| `docs/B3_NUMBER_PLACEHOLDER_MAP.md` | seal/opening/receipt | It is itself the SI-vocabulary audit of the paper. |
| `docs/FIGURE_PLAN_STAGE19_*`, `docs/PAPER_*` | opening/gate | Transition-era paper working docs; superseded rows belong to the paper work stream. |
| `src/thermoroute/*` docstrings (e.g. `conventional_score.py`) | "Route-A models", "pre-registration apparatus" | Code comments/strings belong to the cleanup work stream; none of these modules were renamed here. |

Stale cross-references to archived files (e.g. `PAPER_BENCHMARK_RESTRUCTURE.md`
citing `OPTION_A_*` §1, `PAPER_RESTRUCTURE_20260805.md` citing
`B02_PERMANENT_DESCRIPTIVE_CLAIM_WORDING.md`, `FAIR_EXTERNAL_CLOSEOUT_REQUEST_PACK.md`
citing `CLEANROOM_REPRODUCTION_DESIGN.md`, `CODE_FREEZE_DISCIPLINE` citing
`SOURCE_HASH_SCOPE_REMEDIATION_PLAN.md`) now point at paths under
`docs/archive/`. Content is preserved; rewriting those prose references is
deferred to the owning work streams.

## 4. Verification

- `README.md` contains no seal / opening / receipt / claim-registry /
  pre-registration / one-time / inference-gate vocabulary (checked by grep).
- 21 files moved via `git mv`; 12 files header-marked; 0 deletions, 0 content
  rewrites of historical facts.
- `git status` shows only the intended docs/ moves, the header additions, the
  README rewrite, and the new record file on top of the pre-existing staged
  work from other work streams.

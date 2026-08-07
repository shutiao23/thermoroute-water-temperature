# ThermoRoute submission blueprint — execution ledger

| Field | Value |
| --- | --- |
| Snapshot | 2026-08-02 — post-Stage-09 receipt / pre-09b source-boundary decision |
| Branch / HEAD | `feat/route-a-completion` / `bc6d4eac` |
| Purpose | Track the supplied P0/P1/P2 blueprint as verifiable work, not as narrative progress |
| Authority | **Status ledger only.** It is not a completion receipt, protocol seal, opening authorization, rights decision, or submission approval. |
| Source identity | This file is under `docs/` and is outside `DEFAULT_SOURCE_PATTERNS`; it must not be used to relabel or promote a Stage-09 cache. |
| Persistence | The current docs/paper worktree contains modified and untracked drafts. They are reviewable local artifacts, not committed/sealed protocol or release evidence. |
| 2026-08-02 update | Stage09b config-shape repair applied and tested (new source `0e932f19…`); run package prepared; manuscript wording batch deferred to P0-A0 closure (see `docs/M01_M14_M15_MANUSCRIPT_BATCH_DEFERRED.md`). |

## 1. Hard boundaries

1. The authoritative completed Stage-09 formal run is `7cb2bfb18c1f9aa3dba7`, bound to
   `source_sha256=19289553aa0929bdb5803a8a3eaa96b38a651b3d52da441fb6298f2ac9228b55`.
   `outputs/models/route_a_stage09_completion.json` is `PASS_FORMAL_STAGE09_COMPLETE`;
   its file SHA-256 is `07a0dd1e54cfcc96179c8adaffe2d987776c210e27972faafca90cec6b12f111`.
2. There is no live Stage-09, 09b, or watcher process. The watcher first validated
   the Stage-09 receipt and then launched 09b; 09b exited fail-closed before any
   member training.
3. 09b run `a930214d93fb7bdca83e` rejected the live arm/seed configuration because
   tuple values did not satisfy its JSON-list type contract. It created no
   authorization, work order, member cache, or completion receipt. No 2021–2023
   outcome has been requested or read. The Stage-09 completion receipt does not
   itself constitute target-period performance evidence or an opening authorization.
4. The previous run `bb02498a8396ea7c6110` is void and must not be resumed,
   promoted, mixed with the live run, or deleted before its retained audit role
   is discharged.
5. A source edit to correct 09b's config-shape contract would produce a new source
   identity. It must not occur without explicit user authorization; it would require
   a newly sealed Stage-09→09b lineage and cannot relabel or promote the completed
   `7cb2...` receipt across hashes. Until that decision, do not modify `src/**`,
   `scripts/**`, `tests/**`, `protocols/**`, workflow files, `pyproject.toml`, or
   requirement locks. These paths enter the run source hash.
6. Do not create a new-source Stage-09 run, manually resume `a930...`, acquire
   confirmation labels, create an opening intent, or perform the one-time opening
   without the separately required user authorization.

## 2. Status vocabulary

| Status | Meaning |
| --- | --- |
| `DONE_RECEIPTED` | Required immutable receipt exists and independently validates |
| `PARTIAL_RECEIPTED` | At least one required immutable receipt validates, but the work package requires additional receipts/gates |
| `DONE_CODE_ONLY` | Implementation exists, but the required new run/receipt is absent |
| `DRAFT_ONLY` | Design/document exists but is not a frozen executable protocol |
| `BLOCKED_STAGE09` | Requires source-hash changes or a stationary artifact namespace |
| `BLOCKED_PROTECTED_SOURCE` | Legitimate continuation requires an authorized edit inside the source-hash boundary and a new identity |
| `BLOCKED_AUTHORIZATION` | A human approval gate is intentionally unsatisfied |
| `BLOCKED_OWNER_CUSTODIAN` | Required scientific-owner, custodian or statistical-reviewer choices/signatures are intentionally blank |
| `BLOCKED_EXTERNAL` | Requires new data, rights holder, DOI service, stakeholder, or independent machine |
| `NOT_STARTED` | No artifact meeting the blueprint acceptance criterion was found |
| `EVIDENCE_COLLECTED` | Authoritative facts/provenance were gathered, but the executable artifact, human decision, seal or receipt required for completion is absent |
| `BUILD_PENDING` | Exact source/build inputs or blockers are recorded, but no passing build/reproduction receipt exists |
| `PRE_STATIC_AUDITED` | Frozen pre-opening bytes and claim coverage passed a static audit; no post-opening result is implied |
| `PRE_CORE_BYTES_AUDITED` | The exact core PRE bytes named by the 2026-08-01 audit passed; later SI/paper changes are outside that audit |
| `SI_STATIC_AUDITED` | SI shell indexing, placeholders and evidence routing passed a separate internal static audit; no SI result or receipt is implied |

Rows may combine qualifiers with `/` or `+` when different acceptance dimensions
have different states. Any blocker or `DRAFT_ONLY` component dominates the
overall completion decision; a compound status is never equivalent to
`DONE_RECEIPTED`.

## 3. P0 ledger

| ID | Current status | Evidence present | Missing acceptance evidence / next safe action |
| --- | --- | --- | --- |
| P0-A0 | `DONE_CODE_ONLY + DRAFT_ONLY + BLOCKED_PROTECTED_SOURCE + BLOCKED_AUTHORIZATION` | Commit `008e1f34` aligns 2018 calibration inclusion per horizon and makes the composite loss unit-covariant; the H1/H3 clarification, integrated closure and dual-hash adoption alternative are designed in `docs/ROUTE_A_H1_H3_CLARIFICATION_DRAFT.md`, `docs/P0_A0_GOVERNANCE_CLOSURE_DESIGN.md` and `docs/STAGE09_CROSS_SOURCE_PROVENANCE_BRIDGE_DESIGN.md`. | The calibration erratum is still a draft, has no seal/validator/opening binding, and no full 2018 per-horizon key-equivalence receipt exists. Current code requires one source hash. The 09b repair and any A0 closure implementation need an explicitly authorized new boundary; cross-source adoption may not relabel old artifacts. |
| P0-R0 | `DRAFT_ONLY + BLOCKED_PROTECTED_SOURCE + BLOCKED_AUTHORIZATION` | Python 3.12 fully hashed lock and a locked/latest GitHub CI matrix exist; the implementation and acceptance design is recorded in `docs/CLEANROOM_REPRODUCTION_DESIGN.md`. The old condition “wait for a valid 7cb2 receipt” is satisfied. | No container or `make reproduce-paper`; full stationary-namespace pytest receipt is absent; current CI is not a one-command receipt-to-paper reproducer. Implementation changes hashed paths and must be coordinated with the authorized new source boundary rather than described as blocked by a live Stage09 process. |
| P0-A1 | **`PARTIAL_RECEIPTED + BLOCKED_PROTECTED_SOURCE + BLOCKED_AUTHORIZATION`** | Formal Stage-09 run `7cb2...` under `19289553...` has `PASS_FORMAL_STAGE09_COMPLETE`; the completion receipt file SHA-256 is `07a0dd1e54cfcc96179c8adaffe2d987776c210e27972faafca90cec6b12f111`. The watcher independently accepted it before attempting 09b. The exact failure/recovery boundary is recorded in `docs/STAGE09B_CONFIG_SHAPE_FAILURE_20260802.md`; `docs/STAGE09B_PROTECTED_CHANGE_WORK_ORDER.md` defines the narrow allowlist, regression matrix, authorization language and guarded replacement-lineage sequence. **2026-08-02:** the allowlisted config-shape repair is applied and tested (`scripts/09b_development_controls.py` canonical arm descriptors + JSON-native `time_split`; T1–T5 regressions added; full suite 1694 passed / 1 deferred claim-registry check on `.venv-route-a`). New tested source identity: `0e932f19975033ef0749d2589b60c6aefe057adbef1ee1d9e2f75bef8180920f`; `ops/stage09/phase2_watch.env` pin updated. The repair is **not yet run**: new guarded Stage-09 execution, receipt, 09b lineage and watcher relaunch all remain `BLOCKED_AUTHORIZATION` per work order §2. | P0-A1 requires the full development closure, not Stage-09 alone. Stage09b/16/25 receipts, model-suite freeze and independent replay remain absent. The 09b config-shape fix and the resulting new Stage-09 lineage are not authorized; this Stage-09 receipt cannot cross that source boundary. |
| P0-A2 | `BLOCKED_PROTECTED_SOURCE + BLOCKED_AUTHORIZATION` | One Stage-09 receipt exists for the old exact source. The subsequent 09b attempt `a930...` failed before member training because `ArmSpec` tuples were not normalized to the list form required by the formal config contract. | A1 is incomplete. No model-suite freeze, independent replay, input/gate/chronology chain, M→I→G→C commits, or external timestamp exists. The only legitimate recovery changes protected source, needs user authorization, then requires a new sealed Stage-09→09b lineage. |
| P0-A3 | **`BLOCKED_AUTHORIZATION`** | Opening code and fail-closed contracts exist. | A2 is incomplete and explicit approval for the unique opening has not been given. No 2021–2023 label acquisition or opening attempt is allowed. Even after completion, Route A remains descriptive-only. |
| P0-B1 | `DRAFT_ONLY + BLOCKED_OWNER_CUSTODIAN` | `docs/ROUTE_B_GE30_CLUSTER_SAMPLING_FRAME_DRAFT.md` corrects the impossible HUC2≥30 design and specifies a graph-disconnected-network PSU, two-stage sample, schemas, input-completeness limits and gates; `docs/ROUTE_B_POWER_MDE_DRAFT.md` specifies the power/assurance grid and K rule. `docs/ROUTE_B_PRELABEL_DECISION_PACKAGE.md` consolidates the unresolved choices, ≥30/effective-balance gates and four-stage freeze. `docs/ROUTE_B_OWNER_CUSTODIAN_DECISION_FORM.md` makes D01–D13 and G01–G16 signable without preselecting any answer and explicitly does not authorize target acquisition. | D01–D13 are unsigned/unselected and G01–G16 remain open. No topology/candidate download, frame, graph, random selection, simulated power grid, receipt or external timestamp exists; graph disconnection is not statistical independence; neither package nor form is sealed or proven feasible. |
| P0-B2 | `DRAFT_ONLY + EVIDENCE_COLLECTED + BUILD_PENDING` | `docs/ROUTE_B_MODEL_BUDGET_IDENTIFIABILITY_DRAFT.md` specifies information-set parity, model/trial schemas, draft compute budgets, C/F end-to-end equivalence and a fail-closed identifiability gate. `docs/ROUTE_B_COMPARATOR_PROVENANCE_20260801.md` binds the current author Air2stream commit/license and exact PyPI/source identities for N-HiTS and TFT candidates, with arm-specific eligibility. `docs/ROUTE_B_PRELABEL_DECISION_PACKAGE.md` integrates arm choices, fair-budget requirements and falsifiable headline kill conditions. The isolated Air2stream attempt in `docs/AIR2STREAM_SOURCE_BUILD_AUDIT_20260801.md` verified source/input bytes but failed closed because no compiler/golden-output contract exists. | No frozen protocol, user-selected probabilistic comparator, passing Air2stream source-build/reference receipt, common-budget pilot, trial ledger, unit-equivalence receipt or identifiability receipt exists. |
| P0-B3 | `DRAFT_ONLY` | `docs/ROUTE_B_MEASUREMENT_MISSINGNESS_UQ_DRAFT.md` specifies raw measurement lineage, flow/day semantics, opportunity/attrition, per-horizon calibration, missingness sensitivity and crossed space×time UQ. `docs/ROUTE_B_PRELABEL_DECISION_PACKAGE.md` consolidates the corresponding user/statistical-reviewer choices and fail-closed gates. | No target/raw transaction, registry, frozen completeness threshold, fill-invariance test, sensitivity output or UQ receipt exists. |
| P0-B4 | `NOT_STARTED` | None eligible. | Depends on sealed B1–B3 and untouched outcomes. The result must be allowed to falsify the headline and cannot use subgroup rescue. |
| P0-P | `DRAFT_ONLY + PRE_CORE_BYTES_AUDITED + SI_STATIC_AUDITED` | The eight claim-registry-bound core PRE document hashes and permanent-claim blocks pass the exact 2026-08-01 scope in `docs/PRE_POST_STATIC_CLAIM_AUDIT_20260801.md`; that audit does not cover later edits. A corrected, XML-valid conceptual PRE SVG exists. `paper/si/README.md` plus SI00–SI16 and the FigS1–FigS8 inventory materialize static no-result shells; their current indexing, equation, placeholders, cell-level value routing and scope boundaries pass `docs/SI_STATIC_INTEGRITY_AUDIT_20260802.md`. | “Shells present” is not “SI complete”: all 17 SI documents remain DRAFT/PRE and ten `_RECEIPT` files are empty projections. The H1/H3 clarification is unsealed; no raster/page QA, receipt-derived Results/Discussion, filled Figures 1–4/SI values, POST renderer, PDF or page-QA receipt exists. |
| P0-F | `EVIDENCE_COLLECTED + BLOCKED_EXTERNAL` | Rights inventory/risk-path drafts, fail-closed PUBLIC gate, dynamic local/remote-ref and history exposure audits, official-source USGS/Daymet/gridMET/AGU evidence matrix and administration/license/reproducer intake templates exist. `docs/FAIR_RIGHTS_ACCEPTANCE_MATRIX.md` maps every P0-F acceptance row to current evidence, qualified/external inputs, future receipts and conservative PUBLIC disposition branches. `docs/FAIR_EXTERNAL_CLOSEOUT_REQUEST_PACK.md` supplies narrow, hash-bound, unsent request/intake drafts for provider scope, TeX files, author metadata, qualified byte-rights review, DOI deposit and independent Linux reproduction. | The request pack is `DRAFT_ONLY / NOT_SENT / NO_EXTERNAL_ACTION_AUTHORIZED`. Per-object SHA-256 inventory, exact-byte policy mapping, written ORNL/Daymet redistribution scope, AGU template file-level permission or exclusion implementation, qualified provider decisions, unreachable-object classification, history remediation, DOI, verified author/ORCID/CRediT/COI/funding, public clean-room archive and independent reproduction are absent. |

## 3.1 General P0-01–P0-12 crosswalk

The supplied review uses both a general submission-blocker numbering and a
Route-A/Route-B implementation numbering. Neither replaces the other.

| General ID | Current decision | Implementation rows / strongest evidence | Why it is not complete |
| --- | --- | --- | --- |
| P0-01 final source/protocol boundary | `BLOCKED` | P0-A0, P0-R0, P0-A1; one exact-source Stage09 receipt exists | 09b repair creates a new hash; errata/clarification/final boundary are not implemented or sealed |
| P0-02 outcome-free Route B cohort | `PARTIAL / BLOCKED_OWNER_CUSTODIAN` | P0-B1; sampling/power drafts, `ROUTE_B_PRELABEL_DECISION_PACKAGE.md` and the blank `ROUTE_B_OWNER_CUSTODIAN_DECISION_FORM.md` | D01–D13 unsigned/unselected; G01–G16 open; no frame, graph, sample, power receipt or seal |
| P0-03 fair comparator matrix | `PARTIAL` | P0-B2; Air2stream/N-HiTS/TFT provenance | no selected matrix, passing official build/reference, common-budget pilot, trial ledger or frozen protocol |
| P0-04 Stage09/09b/16/25 rerun | `BLOCKED` | P0-A1; Stage09 `7cb2...` is the one completed atomic receipt | 09b failed before members; 09b/16/25 and frozen-suite receipts are absent |
| P0-05 independent clean-room suite replay | `BLOCKED` | P0-R0/P0-A2 designs | no fixed image, one-command build, stationary full-test receipt or independent replay |
| P0-06 predictors/QC/opening freeze | `BLOCKED` | P0-A2/P0-B3 designs | suite/replay incomplete; predictor snapshot, chronology and authorization absent |
| P0-07 one-time 2021–2023 acquisition and QA | `BLOCKED` | P0-A3; parent receipt confirms outcomes not requested/read | no acquisition authorization, raw transactions, normalized outcomes or QA receipt |
| P0-08 primary analysis/sensitivities | `BLOCKED` | P0-A3/P0-B4 | no opening receipt, exact target common-key results or sensitivity receipts |
| P0-09 identifiability/bound falsification | `PARTIAL` | P0-B2 design and falsifiable headline kill conditions | no known-truth simulator, matched unbounded execution, latent recovery/stability or utility result |
| P0-10 manuscript, figures and SI | `PARTIAL` | P0-P static PRE/SI shells | no receipt-derived Results/Discussion, filled figures/SI, POST renderer, PDF or page QA |
| P0-11 FAIR/rights/DOI | `BLOCKED` | P0-F and `FAIR_RIGHTS_ACCEPTANCE_MATRIX.md` | byte decisions, author declarations, public archive, clean-host receipt and DOI absent |
| P0-12 independent review/submission audit | `BLOCKED` | no eligible final packet | no final PDF/SI/DOI, three-reviewer audit, final clean-room replay or claim-to-evidence audit |

No row in this crosswalk is `PROVEN COMPLETE`. A completed sub-receipt may be
reported atomically without upgrading its containing P0 work package.

## 4. Route-A/Route-B P1 ledger

| ID | Status | Required artifact |
| --- | --- | --- |
| P1-01 | `DRAFT_ONLY` | `docs/P1_PRELABEL_STRATA_REGISTRY_DRAFT.md` now gives exact outcome-free event/flow/season/year/qualifier/history assignment contracts, unknown/support rules, schemas, receipts, freeze order and acceptance tests; unresolved scientific cutpoints remain explicit. No sealed executable registry or result exists. |
| P1-02 | `DRAFT_ONLY` | `docs/ROUTE_B_GE30_CLUSTER_SAMPLING_FRAME_DRAFT.md` separates known-gauge and strict-ungauged arms and complete-component exclusion; no sampled graph, leave-year/component execution or result exists. |
| P1-03 | `EVIDENCE_COLLECTED + BUILD_PENDING` | Official Air2stream and N-HiTS/TFT candidate identities, licences and arm eligibility are recorded; the Air2stream source-build attempt is fail-closed; no comparator is selected/frozen and no common-budget run exists. |
| P1-04 | `DRAFT_ONLY` | `docs/ROUTE_B_MEASUREMENT_MISSINGNESS_UQ_DRAFT.md` specifies station/component-balanced interval score, pinball, reliability, Brier and calibration slope and forbids three-quantile CRPS; no executable metric receipt exists. |
| P1-05 | `BLOCKED_EXTERNAL` | Second independent Linux host reproduction receipt with declared numeric tolerances |
| P1-06 | `DRAFT_ONLY` | `paper/si/README.md` and SI00–SI16 now materialize the full static no-result scaffold: design, formal rows, scores, probability metrics, controls, temporal/spatial/QC/external/missingness appendices, reproduction, rights and data-dictionary routing. Final receipt-filled values, search spaces, seeds, model sizes, attrition, failures and exact command closure remain absent. |

## 4.1 General P1 crosswalk

| General ID | Status | Required remaining evidence |
| --- | --- | --- |
| P1-01 additional-year replication | `NOT_STARTED / BLOCKED_EXTERNAL` | outcome-free new-year protocol and untouched 2024–2025 (or later) replication receipt |
| P1-02 multiscale spatial dependence | `DRAFT_ONLY` | HUC2/HUC8/network/distance-buffered execution and LOCO results |
| P1-03 temporal-dependence UQ | `DRAFT_ONLY` | block/rolling/year-season calibration and coverage/width receipts |
| P1-04 SOTA supplementary comparison | `EVIDENCE_COLLECTED + BUILD_PENDING` | official process comparator and strong temporal model under equal information/budget/keys |
| P1-05 missingness/quality stress | `DRAFT_ONLY` | observed-history, qualifier, flatline/drift and fill-policy sensitivities |

## 5. Route-A/Route-B P2 ledger

| ID | Status | Dependency / scope boundary |
| --- | --- | --- |
| P2-01 | `BLOCKED_EXTERNAL` | Requires a real water-resource user, action, lead time and cost-loss definition; otherwise hypothetical only |
| P2-02 | `BLOCKED_EXTERNAL` | Requires an independently governed non-U.S. basin dataset and preregistered transfer/tuning rule |
| P2-03 | `BLOCKED_EXTERNAL` | Requires rights closure and DOI/versioned hosting; metadata/synthetic demo is the fallback |
| P2-04 | `NOT_STARTED` | Policy mapping may proceed only after a quantified endpoint exists; daily-mean forecasts are not regulatory determinations |

## 5.1 General P2 crosswalk

| General ID | Status | Scope boundary |
| --- | --- | --- |
| P2-01 operational NWP replay | `NOT_STARTED / BLOCKED_EXTERNAL` | requires archived as-issued forecast vintages and latency/state contracts |
| P2-02 process coupling | `NOT_STARTED / BLOCKED_EXTERNAL` | requires catchment/network/upstream data and identifiable energy/process constraints |
| P2-03 ecological/regulatory endpoint | `NOT_STARTED / BLOCKED_EXTERNAL` | requires independently governed daily-max/7DADM and standards/species registries |
| P2-04 companion dataset/dashboard | `BLOCKED_EXTERNAL` | requires rights closure, DOI/versioned hosting and a reviewed public member set |
| P2-05 cross-country transfer | `NOT_STARTED / BLOCKED_EXTERNAL` | requires independently governed non-U.S. data and preregistered no-target-region tuning |
| P2-06 policy/communication | `NOT_STARTED / BLOCKED_EXTERNAL` | requires real action thresholds, lead times and utility before policy or press claims |

## 6. Immediate safe queue after the Stage-09 receipt and before the 09b source-boundary decision

1. Correct and expand the Route-B design under `docs/`, including candidate
   artifact schemas and fail-closed acceptance tests.
2. Produce the H1/H3 clarification draft under `docs/` without changing the
   frozen five-test JSON family.
3. Produce a clean-room architecture and artifact-namespace isolation design;
   defer implementation in hashed paths.
4. Execute read-only rights inventory commands and record facts separately from
   legal conclusions; never publish or rewrite history from this ledger.
5. Build the 2021–2026 literature and SI evidence map without inserting any
   pending performance value.
6. Preserve the completed `7cb2...` receipt and the failed `a930...` pre-member
   record. If the user authorizes the protected config fix, record the new source
   identity and use only the guarded entrypoint for the resulting new Stage-09
   lifecycle; do not resume or fabricate 09b artifacts.

## 7. Release gates

The project is not submission-ready until every selected route has its required
receipt chain. Route A alone can at most yield an audit-ready fixed-cohort
descriptive benchmark. Strong superpopulation, transfer, operational, national,
ecological, or policy claims require a separate Route B and cannot be recovered
by rewriting Route-A results.

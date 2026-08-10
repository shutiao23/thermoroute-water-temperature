# WRR information-regime completion tracker

Snapshot: 2026-08-09 UTC
Scope: all work requested by the external review in
`pasted-text.txt`, reconciled with the current repository evidence state.

This tracker is operational. The detailed scientific rationale remains in
`docs/WRR_REFEREE_REPORT_AND_COMPLETION_PLAN_20260809.md`; evidence eligibility
is governed only by `docs/SCIENTIFIC_EVIDENCE_STATUS.md` and the decision log.

## Non-negotiable evidence boundary

- The historical 432 information-ladder shards and 12 forcing shards are
  withdrawn by DLOG-025. They are forensic records, not manuscript evidence.
- The historical 0.622 C, +0.48 C, 30-fold and 27-to-1 statements remain
  withdrawn until independently frozen replacement results exist.
- No new model score may be computed before a reviewed, exact-byte v4 execution
  seal binds the corrected runner, inputs, registries, environment and clean
  design commit.
- The already-open 2021--2023 analysis is a post-outcome-normalized reanalysis,
  never an untouched or prospective confirmation.

## Execution order and acceptance gates

| ID | Priority | Deliverable | Current state | Definition of done |
|---|---:|---|---|---|
| T00 | P0 | Preserve withdrawn evidence boundary | IN PROGRESS | Old F/L outputs cannot enter the claim ledger, paper macros or unified manuscript authority |
| T01 | P0 | Correct score-free semantic data registries | COMPLETE: DLOG-028 | Raw observedness is captured before imputation; train and validation examples are disjoint and exact-byte bound |
| T02 | P0 | Correct cell/fold/model/input/contrast/environment registries | COMPLETE: DLOG-028 | Temporal partitions are not CV folds; geometry seed rules are explicit; all 48 degraded-matrix cells are represented; status inventory is generated |
| T03 | P0 | Production-path lineage regression | COMPLETE | Five real-input tests cover raw masks, train-only imputation, disjoint periods and exact formal keys/targets without reading scores |
| T04 | P0 | F2b go/no-go | COMPLETE: NO-GO / `PLANNED_DEGRADED` | DLOG-027 closes the current 72-cell claim; use the 48-cell F0/F3 scope unless a separately frozen archive pilot later passes every upgrade gate |
| T05 | P0 | v4 execution protocol and seal | COMPLETE: DLOG-028, `3eba8d6` + `cd69b0c` | Immutable sealed copy and seal bind exact protocol bytes, registries, sources, runtime, clean design commit and authorized scope |
| T06 | P0 | Corrected 12-cell F0/F3 tree run | COMPLETE: `outputs/final/forcing_regime_v5_observed` and `3eba8d6` | Twelve create-only observed-lineage shards exist with disjoint 2006--2015 train / 2016--2017 validation and exact common keys |
| T07 | P0 | Independent 12-cell authority | COMPLETE: `outputs/final/forcing_regime_v5_observed_point_authority_v1` | A second reconstruction verifies lineage, one raw cell, station metrics and paired forcing effects; point authority is frozen |
| T08 | P0 | Corrected L0/L1/L2 tree ladder | BLOCKED BY T07 | Versioned 432-cell replacement plus masks, exact keys and a new independent authority; old bytes remain untouched |
| T09 | P0 | Ten-seed geometry and LightGBM crossed cells | BLOCKED BY T08 | Random seeds 0--9 x four folds complete; whole-region is placed in the random distribution; 24 LightGBM F0/F3 spatial cells complete |
| T10 | P0 | Plain-TCN primary cells and ThermoRoute hard cells | BLOCKED BY EXECUTABLE NEURAL RUNNER | Point-only matched loaders/trainers/scorers/writers exist; bounded/unbounded hard-regime variants pass exact-key and budget gates |
| T11 | P0 | Degraded 48-cell or full 72-cell matrix | BLOCKED BY T04/T09/T10 | Every non-degraded logical cell has a counterfactual pair and station-first effects; no separate-design arithmetic is presented as a budget |
| T12 | P0 | F2a matched temperature diagnostic | BLOCKED BY T05 | F0/F2a/F3-temperature-only share the frozen 329,628-key registry and denominator guard; never labelled operational/as-issued |
| T13 | P0 | F3 placebos and forcing components | BLOCKED BY T05 | True, within-month shuffled and +/-7-day shifted arms plus predeclared components run on matched keys |
| T14 | P0 | L2-U2 | BLOCKED BY EXECUTABLE RUNNER | LightGBM and plain TCN, F0/whole-region/7-day, five fit seeds; WTEMP/FLOW/site-state invariance proven |
| T15 | P1 | L3 and novelty analysis | BLOCKED BY BASIN ATTRIBUTES | Predeclared 8--12 hydrologic attributes, coverage gate, training-only geographic/hydroclimatic distance and transfer curves |
| T16 | P1 | Validated forced hybrid | IMPLEMENTED; AUTHORITY/CALIBRATION RUN PENDING | Official air2stream validation or transparent equilibrium-response model passes sign/unit/reference/calibration tests |
| T17 | P1 | Event and decision metrics | BLOCKED BY VALID FORCING RESULTS | Warm q90, signed rapid warming/cooling, POD/FAR/CSI/Brier/onset use training-only thresholds and common forcing keys |
| T18 | P1 | Larger no-flow cohort and nested flow subset | EXTERNAL DATA WORK | Headline cells replicate on a temperature+meteorology cohort, or fixed 120-site availability-selected scope is activated everywhere |
| T19 | P1 | 2024--2025 frozen audit | EXTERNAL DATA WORK | Audit protocol is sealed before outcome access, then run once without retuning; otherwise post-outcome wording remains active |
| T20 | P1 | Inference and interactions | BLOCKED BY T11 | Station-first effects, two-stage seed aggregation, 10,000-draw HUC2 bootstrap, sign flips, LOCO, Holm family and F-by-L/G/A interactions |
| T21 | P1 | Anchor sensitivity, decomposition intervals and MDE | READY AFTER GOVERNANCE FREEZE | Predeclared anchor variants, cluster intervals and resolvable-effect calculation are frozen and reported |
| T22 | P0 | Information-regime manuscript rewrite | BLOCKED BY PROMOTED RESULTS | Title/abstract/key points/RQs/results/discussion use only frozen evidence; oracle, retrospective forecast, gauged and cold-start tasks remain distinct |
| T23 | P0 | Four figures, three tables, SI and claim ledger | BLOCKED BY T20/T22 | All visuals regenerate from the unified authority; no stale pooled/station-median figure; every number has an exact ledger source and `used_in` span |
| T24 | P0 | AGU build and length closure | BLOCKED BY T22/T23 | Markdown and TeX are synchronized, abstract <=250 words, figures/tables/equations/sections are continuous, PDF <25 PU |
| T25 | P0 | Submission metadata and FAIR release | NEEDS AUTHOR/RIGHTS INPUT | Author/affiliation/ORCID/funding/CRediT/conflict fields, data/software DOI, licence, tag, URL and independent reproduction are closed |
| T26 | P0 | Final verification | BLOCKED BY ALL ABOVE | Full tests collect and pass in declared environments; Ruff/format, manuscript consistency, injection tests, manifests and hashes pass |

## Model allocation for future work

- Scientific architecture, protocol, experiment design, core model code and
  evidence review: `gpt-5.6-sol` with `max` reasoning.
- Paper drafting and substantive scientific revision: `gpt-5.6-sol` with
  `xhigh` reasoning.
- Ordinary code/control changes: `gpt-5.6-sol` with `high` reasoning.
- The requested `gpt-5.6-luna` model is not available in this runtime. Small
  automation work uses the lowest suitable available model/effort without
  lowering the review level of scientific artifacts.

## User- or institution-controlled closeout items

The research implementation can proceed without further scientific choices,
but T25 cannot be honestly completed from repository context alone. It needs
the actual author roster, affiliations, ORCIDs, funding and contribution
declarations, redistribution-rights decisions, DOI minting authority, public
repository/release destinations and an independent reproduction operator.

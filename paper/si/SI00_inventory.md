# SI00 — inventory and evidence map

**Status:** scaffold finalized; empirical values are filled from the results
authority (`outputs/final/`, protocol v1) where the corresponding experiment
has run; cells without a bound authority value remain `[pending computation]`.

This is an inventory of static SI design items, not an evidence receipt. Any
unbound result cell is `[pending computation]`.

## Redesign inventory deltas (2026-08-08)

| SI | Change |
|---|---|
| SI04 | redesign chronology table added (protocol v1, decision log) |
| SI07 | station-median grid sourced from `outputs/final/station_metrics.parquet`; air2stream row on the common 116-station set; plain controls admitted |
| SI09 | information-matched controls re-scored on the 2021–2023 window (no calibration); status no longer development-only |
| SI11 | independent-window 2×2 spatial factorial (geometry × adaptation), repeated random splits, distance/hydroclimatic novelty |
| SI14 | key-registry history-completeness strata (Major Comment 12 sensitivity) |
| SI15 | results-authority commands and manifest added |

## Inventory

| Item | Present PRE material | What it may state now | What it must not state now |
|---|---|---|---|
| SI01 | frozen cohort, registry binding and HUC2 cluster geometry | named cohort facts and projection fields | realized target-period availability or performance |
| SI02 | issue-time boundary, product bridge, **replay isolation, acquisition transport/durability, adversary model, and body-hash semantics** | allowed/forbidden information, the provenance contract, the four denied replay capabilities, the four transaction properties, and what a body hash does and does not bind | as-issued operational availability; any security property against a malicious owner or same-privilege adversary; exactly-once HTTP delivery |
| SI03 | model-description text and Figure-1 equation | identities, units and limitations | fitted coefficients or component efficacy |
| SI04 | fixed protocol and amendments named in PRE text | chronology, scope and erratum role | that opening or release closure occurred |
| SI05 | five registered rows | pairs, horizons, margins, estimand and required receipt fields | effects, CIs, p-values, Holm values or decisions |
| SI06 | formal five-row output shell | fixed display geometry and evidence-field requirements | any result before opening-receipt verification |
| SI07 | all-model score shell, **plus the air2stream `NOT_RUN` status/provenance and the unscored per-station LightGBM variant** | metric definitions, exact-common-key routing, the thirteen-row temporal registry (six primary + seven controls), and what a defensible air2stream comparison would require | development-cache scores or selective model rows; any air2stream number; a global-model score reported under the per-station label |
| SI08 | probability and reliability shell, **plus the Stage-19 non-reporting disposition** | target-period metric/bin schemas, calibration limitations, and the measured development-period degeneracy facts | unbound coverage, Brier, pinball or reliability values; any development-period probability metric; the phrase "quantile crossing" |
| SI09 | Stage09/09b controls shell; the **development-period** companion table to Figure S10 | frozen arm/seed provenance routing; the Stage-09b information-matched controls under their own evidence role | control efficacy before the full development suite is frozen; any Stage-09b value inside a target-period figure |
| SI10 | temporal-coverage shell | availability, year/season and block-sensitivity schemas | realized target-period counts or scores |
| SI11 | spatial-sensitivity shell, **plus the recomputed HUC2/HUC4/HUC6/HUC8 cluster ladder** | HUC/leave-cluster display geometry; the outcome-free cluster geometry, the cluster thresholds it does not meet, why HUC8 was not adopted, and the leading-zero recomputation trap | national or independent-cluster claims; HUC8 presented as a route to comparison eligibility |
| SI12 | QC/qualifier shell | raw-response, series, qualifier and gate routing | a passed QC claim without exact-A evidence |
| SI13 | external history-dependent shell | explicit known-gauge external scope | ungauged, operational or network-transfer wording |
| SI14 | missingness/failure shell | attrition and failure-case schemas | post hoc threshold selection or omitted failures |
| SI15 | reproduction shell | required hashes, commands, environment and parity fields | one-click or independent-replay claims without receipts |
| SI16 | rights/data-dictionary shell, **plus the full redistribution class enumeration relocated from the main text** | decision fields, units, release routing, the four classes defaulting to exclusion and the two excluded outright, and the exact scope of the MIT licence | redistribution approval or FAIR completion |

## Evidence routing

| Consumer | Required future authority | Required fields or bindings |
|---|---|---|
| cohort/table/registry projection | registry and panel binding, then receipt-bound projection | source path, SHA-256, cohort identifier, projection rule |
| information-boundary diagram | acquisition and bridge bindings | issue date, target date, horizon, predictor date, source/vintage, admissibility flag |
| equation/model description | frozen model-suite binding | model identifier, source hash, parameter/configuration binding, units |
| protocol/claim statement | authorization, comparison registry and inference-cluster-geometry binding | protocol/amendment identity, chronology binding, gate status, comparison eligibility |
| five-row effect table | test-window receipt `formal_tests[*]` | test identity, model pair, horizon, margin, status, effect, CI, station/cluster counts, win rate, raw/Holm p, bound checks |
| all-model and probabilistic tables | receipt-bound predictions/evaluation rows | exact key-set binding, formula, filters, units, calibration role and source pointer |
| controls and model budgets | Stage09/09b/16/25 receipts plus frozen suite | arm/model identity, seed, information set, parameter/search budget and receipt lineage |
| temporal/spatial/QC/failure appendices | opening, coverage, spatial and QC receipts | declared strata, denominators, missingness reasons, cluster roles and fail-closed statuses |
| reproduction and public release | replay/render receipts plus rights manifest | source/runtime/input hashes, commands, parity tolerance, byte inventory, rights decision and licence scope |

The field names in the last column are a POST projection contract from
`docs/POST_PAPER_PROJECTION_DESIGN.md`, not a claim that a current receipt
exists.

## Figure inventory

**Fourteen** figures accompany the submission: four main-text figures and ten
supporting figures. The authoritative manifest is
`paper/FIGURE_REDRAW_SPEC.md` §4–§5, with the reassignment record in its §6.4 and
in `docs/PAPER_FIGURE_SI_RECONCILIATION.md`.
`docs/FIGURE_PLAN_STAGE19_INDEPENDENT_20260805.md` §5 remains the authority for
the Stage-19 determination and the render order, but its nine-figure table is
**superseded** by the 2026-08-06 benchmark restructure. This table exists so that
the SI file set and the figure set cannot drift apart unnoticed.

| Figure | Job | State | Evidence period | SI files it must agree with |
|---|---|---|---|---|
| Fig. 1 | station map, cohort geometry, persistence challenge | `PRE_MATERIALIZED` | draft structural | SI01, SI11 |
| Fig. 2 | how baseline choice changes reported skill | `POST_TEMPLATE_ONLY` | target | SI06, SI07 |
| Fig. 3 | how the spatial partition changes the transfer conclusion | `POST_TEMPLATE_ONLY_DEVELOPMENT` | **development 2019–2020** | SI01, SI13 |
| Fig. 4 | regional/seasonal heterogeneity and what coverage costs | `POST_TEMPLATE_ONLY` | target | SI08, SI10, SI11 |
| Fig. S1 | cohort selection and registry geometry | `PRE_MATERIALIZED` | draft cohort geometry | SI01 |
| Fig. S2 | chronology and issue-time/product boundary | `PRE_MATERIALIZED` | draft boundary | SI02 |
| Fig. S3 | full model, bounded correction, and calibration dataflow | `PRE_MATERIALIZED_DESIGN` | draft model design | SI03 |
| Fig. S4 | point-performance heterogeneity | `POST_TEMPLATE_ONLY` | target | SI06, SI07 |
| Fig. S5 | event score, reliability, expanded probability diagnostics | `POST_TEMPLATE_ONLY` | target | SI08 |
| Fig. S6 | temporal opportunity, missingness, attrition | `POST_TEMPLATE_ONLY` | target | SI10, SI14 |
| Fig. S7 | spatial and leave-HUC2 influence | `POST_TEMPLATE_ONLY` | target | SI11 |
| Fig. S8 | outcome QC, external-history arm, failures | `POST_TEMPLATE_ONLY` | target | SI12, SI13, SI14 |
| Fig. S9 *(optional)* | development-period conformal calibration sensitivity | `POST_TEMPLATE_ONLY_DEVELOPMENT` | **development 2019–2020** | SI08 |
| Fig. S10 | registered architecture interventions and the bounded-deviation audit | `POST_TEMPLATE_ONLY` | target | SI07, SI09 |

Eleven of these — Figures 2–4 and S4–S10 — are POST-gated and are blocked on the
test-window evaluation receipt. Figures S1–S3 are already materialized and carry no
evaluation-period coordinate. **Figure 1's committed bytes are stale**: the
2026-08-06 restructure replaced its bounded-correction panel with the station
map, so the artifact must be re-rendered before submission.

Two figures are development-period: Figure 3 and Figure S9. Both carry a
mandatory in-panel scope band, both are still gated on the test-window receipt, and
no value in either may be compared numerically with any target-period figure.
One figure never mixes two evidence periods, and the POST skeletons now enforce
that at panel granularity and across `shares_value_ids_with`.

Figure cross-references are absent from the manuscript by design.
`paper/FIGURE_REDRAW_SPEC.md` §7 fixes each figure's first-citation position and
states that the citations must be inserted by the PRE/POST renderer rather than
by hand. The manuscript carries fourteen inert `FIGURE_ANCHOR` comments, one per
figure above, whose ids and `state` tokens must equal this table's.

## Static source map

| Source path | Role in this scaffold | Readout allowed here |
|---|---|---|
| `paper/ThermoRoute_paper.md` | canonical frozen PRE narrative | design facts and permanent limitations only |
| `paper/FIGURE1_AND_SI_SKELETON.md` | superseded four-panel geometry retained for audit provenance | historical placeholder and receipt-gate rules only |
| `paper/FIGURE_REDRAW_SPEC.md` | **authoritative** Figure 1--4 / Figure S1--S10 visual and evidence contract | redraw layout, captions, source bindings, gates, QA, and the §6.4 reassignment record |
| `docs/PAPER_FIGURE_SI_RECONCILIATION.md` | the 2026-08-06 figure delta, the SI relocation map, the compile result and the validator outcome | where each former panel went, what was refused and why |
| `docs/FIGURE_PLAN_STAGE19_INDEPENDENT_20260805.md` | Stage-19 determination, dependency paths and render order; **its nine-figure §5 table is superseded** | why Stage-19 is not a blocker; the exact POST artifact names; the render ordering rationale |
| `docs/STAGE19_DEGENERATE_INTERVAL_DISPOSITION_20260805.md` | measured degeneracy facts and the decision not to amend the contract | the §2.1 table reproduced in SI08; the wording ban |
| `docs/POST_PAPER_PROJECTION_DESIGN.md` | deterministic POST projection design | manifest schema and future authority routing |
| `protocols/route_a_primary_v1.json` | fixed registered-family source | pointer only; no rewriting from SI |
| `protocols/route_a_inference_amendment_v2.json` | inference-scope overlay source | pointer only; no eligibility override |
| `protocols/route_a_claim_registry_v1.json` | claim rendering source | pointer only; no handwritten claim substitution |

## Required POST checks before any fill

1. Each declared input path, format, self-hash and byte binding validates.
2. Every visible value resolves to exactly one manifest `value_id` and source
   pointer.
3. The five study rows occur exactly once and retain the descriptive verdict.
4. Missing, non-finite, undeclared or extra evidence rejects the build.
5. A PRE marker, including `[pending computation]`, is an error in
   a purported POST render; removing it never authorizes invented values.
6. SI06–SI16 and FigS1–FigS10 implement the README cell-level binder contract:
   every visible subvalue resolves through
   `rows[binder_row_id].cells[field].value_id`; a single row-level source ID is
   invalid for a multi-value row.
7. SI08 carries no development-period probability metric. The development-period
   probabilistic stage is not produced for this submission; its disposition and
   the measured facts are recorded in SI08 §Stage-19 and in
   `docs/STAGE19_DEGENERATE_INTERVAL_DISPOSITION_20260805.md`. The target-period
   metrics that SI08 projects come from the trusted scorer inside the one-time
   opening, not from that stage.
8. The phrase "quantile crossing" must not appear in any SI file, figure caption,
   figure spec, or renderer in reference to that disposition. Zero strict
   ordering violations were measured; the correct term is "zero-width
   (degenerate) nominal interval".

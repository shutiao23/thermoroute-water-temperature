# SI00 — inventory and evidence map

**Status:** DRAFT / PRE-OPENING / NOT RECEIPT.

This is an inventory of static SI design items, not an evidence receipt. Any
unbound result cell is `[pending — 探索期数据，不可写入结论]`.

## Inventory

| Item | Present PRE material | What it may state now | What it must not state now |
|---|---|---|---|
| SI01 | frozen cohort, registry binding and HUC2 gate geometry | named cohort facts and projection fields | realized target-period availability or performance |
| SI02 | issue-time boundary and product bridge | allowed/forbidden information and provenance contract | as-issued operational availability |
| SI03 | model-description text and Figure-1 equation | identities, units and limitations | fitted coefficients or component efficacy |
| SI04 | sealed protocol and amendments named in PRE text | chronology, scope and erratum role | that opening or release closure occurred |
| SI05 | five registered rows | pairs, horizons, margins, estimand and required receipt fields | effects, CIs, p-values, Holm values or decisions |
| SI06 | formal five-row output shell | fixed display geometry and evidence-field requirements | any result before opening-receipt verification |
| SI07 | all-model score shell | metric definitions and exact-common-key routing | development-cache scores or selective model rows |
| SI08 | probability and reliability shell, **plus the Stage-19 non-reporting disposition** | target-period metric/bin schemas, calibration limitations, and the measured development-period degeneracy facts | unbound coverage, Brier, pinball or reliability values; any development-period probability metric; the phrase "quantile crossing" |
| SI09 | Stage09/09b controls shell | frozen arm/seed provenance routing | control efficacy before the full development suite is frozen |
| SI10 | temporal-coverage shell | availability, year/season and block-sensitivity schemas | realized target-period counts or scores |
| SI11 | spatial-sensitivity shell | HUC/leave-cluster display geometry | national or independent-cluster claims |
| SI12 | QC/qualifier shell | raw-response, series, qualifier and gate routing | a passed QC claim without exact-A evidence |
| SI13 | external history-dependent shell | explicit known-gauge external scope | ungauged, operational or network-transfer wording |
| SI14 | missingness/failure shell | attrition and failure-case schemas | post hoc threshold selection or omitted failures |
| SI15 | reproduction shell | required hashes, commands, environment and parity fields | one-click or independent-replay claims without receipts |
| SI16 | rights/data-dictionary shell | decision fields, units and release routing | redistribution approval or FAIR completion |

## Evidence routing

| Consumer | Required future authority | Required fields or bindings |
|---|---|---|
| cohort/table/registry projection | registry and panel binding, then receipt-bound projection | source path, SHA-256, cohort identifier, projection rule |
| information-boundary diagram | acquisition and bridge bindings | issue date, target date, horizon, predictor date, source/vintage, admissibility flag |
| equation/model description | frozen model-suite binding | model identifier, source hash, parameter/configuration binding, units |
| protocol/claim statement | authorization, claim registry and inference-gate binding | protocol/amendment identity, chronology binding, gate status, claim eligibility |
| five-row effect table | opening receipt `formal_tests[*]` | test identity, model pair, horizon, margin, status, effect, CI, station/cluster counts, win rate, raw/Holm p, bound checks |
| all-model and probabilistic tables | receipt-bound predictions/evaluation rows | exact key-set binding, formula, filters, units, calibration role and source pointer |
| controls and model budgets | Stage09/09b/16/25 receipts plus frozen suite | arm/model identity, seed, information set, parameter/search budget and receipt lineage |
| temporal/spatial/QC/failure appendices | opening, coverage, spatial and QC receipts | declared strata, denominators, missingness reasons, cluster roles and fail-closed statuses |
| reproduction and public release | replay/render receipts plus rights manifest | source/runtime/input hashes, commands, parity tolerance, byte inventory, rights decision and licence scope |

The field names in the last column are a POST projection contract from
`docs/POST_PAPER_PROJECTION_DESIGN.md`, not a claim that a current receipt
exists.

## Figure inventory

Thirteen figures accompany the submission: four main-text figures and nine
supporting figures. The authoritative manifest is
`docs/FIGURE_PLAN_STAGE19_INDEPENDENT_20260805.md` §5; the panel and binder
contracts are in `paper/FIGURE_REDRAW_SPEC.md` §4–§5. This table exists so that
the SI file set and the figure set cannot drift apart unnoticed.

| Figure | State | Evidence period | SI files it must agree with |
|---|---|---|---|
| Fig. 1 | `PRE_MATERIALIZED_REDRAW` | pre-opening design | SI01, SI03, SI05 |
| Fig. 2 | `POST_TEMPLATE_ONLY` | target | SI06, SI07 |
| Fig. 3 | `POST_TEMPLATE_ONLY` | target | SI08 |
| Fig. 4 | `POST_TEMPLATE_ONLY` | target | SI09, SI10, SI14 |
| Fig. S1 | `PRE_MATERIALIZED` | pre-opening cohort geometry | SI01 |
| Fig. S2 | `PRE_MATERIALIZED` | pre-opening boundary | SI02 |
| Fig. S3 | `PRE_MATERIALIZED_DESIGN` | pre-opening model design | SI03 |
| Fig. S4 | `POST_TEMPLATE_ONLY` | target | SI06, SI07 |
| Fig. S5 | `POST_TEMPLATE_ONLY` | target | SI08 |
| Fig. S6 | `POST_TEMPLATE_ONLY` | target | SI10, SI14 |
| Fig. S7 | `POST_TEMPLATE_ONLY` | target | SI11 |
| Fig. S8 | `POST_TEMPLATE_ONLY` | target | SI12, SI13, SI14 |
| Fig. S9 *(optional)* | `POST_TEMPLATE_ONLY_DEVELOPMENT` | **development 2019–2020** | SI08 |

Nine of these — Figures 2–4 and S4–S9 — are POST-gated and are blocked on the
one-time opening receipt. Figures 1 and S1–S3 are already materialized and carry
no evaluation-period coordinate.

Figure S9 is the only development-period figure. It carries a mandatory in-panel
scope band, and no value in it may be compared numerically with Figure 3 or
Figure S5. One figure never mixes two evidence periods.

Figure cross-references are absent from the manuscript by design.
`paper/FIGURE_REDRAW_SPEC.md` §7 fixes each figure's first-citation position and
states that the citations must be inserted by the PRE/POST renderer rather than
by hand.

## Static source map

| Source path | Role in this scaffold | Readout allowed here |
|---|---|---|
| `paper/ThermoRoute_paper.md` | canonical frozen PRE narrative | design facts and permanent limitations only |
| `paper/FIGURE1_AND_SI_SKELETON.md` | superseded four-panel geometry retained for audit provenance | historical placeholder and receipt-gate rules only |
| `paper/FIGURE_REDRAW_SPEC.md` | current Figure 1--4 / Figure S1--S9 visual and evidence contract | redraw layout, captions, source bindings, gates and QA |
| `docs/FIGURE_PLAN_STAGE19_INDEPENDENT_20260805.md` | current POST figure manifest, dependencies and render order | which figure binds which artifact; why Stage-19 is not a blocker |
| `docs/STAGE19_DEGENERATE_INTERVAL_DISPOSITION_20260805.md` | measured degeneracy facts and the decision not to amend the contract | the §2.1 table reproduced in SI08; the wording ban |
| `docs/POST_PAPER_PROJECTION_DESIGN.md` | deterministic POST projection design | manifest schema and future authority routing |
| `protocols/route_a_confirmatory_v1.json` | sealed registered-family source | pointer only; no rewriting from SI |
| `protocols/route_a_inference_amendment_v2.json` | inference-scope overlay source | pointer only; no eligibility override |
| `protocols/route_a_claim_registry_v1.json` | claim rendering source | pointer only; no handwritten claim substitution |

## Required POST checks before any fill

1. Each declared input path, format, self-hash and byte binding validates.
2. Every visible value resolves to exactly one manifest `value_id` and source
   pointer.
3. The five Route-A rows occur exactly once and retain the bound gate verdict.
4. Missing, non-finite, undeclared or extra evidence rejects the build.
5. A PRE marker, including `[pending — 探索期数据，不可写入结论]`, is an error in
   a purported POST render; removing it never authorizes invented values.
6. SI06–SI16 and FigS1–FigS9 implement the README cell-level binder contract:
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

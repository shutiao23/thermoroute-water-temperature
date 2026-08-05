# ThermoRoute figure-redraw specification

**Document role:** authoritative redraw and evidence-binding specification for
four main-text figures and eight Supporting Information (SI) figures.  This file
does not authorize model fitting, target-label access, result filling, opening,
or submission.

**Current phase:** PRE / no verified opening POST receipt.  Figure 1 and
Figures S1--S3 are materialized as PRE artifacts under the source bindings and
visual QA below.  Figures 2--4 and Figures S4--S8 are schema templates only.
They must not be rendered as submission SVG/PDF/PNG files, even with blank axes
or placeholder values, before their POST gates pass.

**Canonical prose source:** `paper/ThermoRoute_paper.md`
**Current PRE TeX target:** `paper/agu_submission/ThermoRoute_WRR.tex`
**Evidence principle:** every visible empirical coordinate is a deterministic
projection of bound evidence; a familiar filename, development cache, or
completion receipt alone is never publication authority.

---

## 1. Non-negotiable figure-state matrix

The states in this table are exhaustive.

| Figure | Canonical responsibility | Current state | Current artifact allowance |
|---|---|---|---|
| Figure 1 | quantitative mismatch, bounded-correction insight, and evidence boundary | `PRE_MATERIALIZED_REDRAW` | **PRE artifact only; no POST coordinates and no forest placeholder** |
| Figure 2 | point-performance evidence | `POST_TEMPLATE_ONLY` | **No** |
| Figure 3 | interval and event-probability evidence | `POST_TEMPLATE_ONLY` | **No** |
| Figure 4 | mechanism sensitivities and applicability boundaries | `POST_TEMPLATE_ONLY` | **No** |
| Figure S1 | cohort selection and registry geometry | `PRE_MATERIALIZED` | **PRE artifact only** |
| Figure S2 | chronology and issue-time/product information boundary | `PRE_MATERIALIZED` | **PRE artifact only** |
| Figure S3 | full model and calibration mechanism | `PRE_MATERIALIZED_DESIGN`; final version remains suite-gated | **PRE design artifact only** |
| Figure S4 | point-performance heterogeneity | `POST_TEMPLATE_ONLY` | **No** |
| Figure S5 | expanded probabilistic diagnostics | `POST_TEMPLATE_ONLY` | **No** |
| Figure S6 | temporal opportunity, missingness, and attrition | `POST_TEMPLATE_ONLY` | **No** |
| Figure S7 | spatial influence and cluster sensitivity | `POST_TEMPLATE_ONLY` | **No** |
| Figure S8 | outcome QC, external-history arm, and failure disposition | `POST_TEMPLATE_ONLY` | **No** |
| Figure S9 | development-period conformal calibration sensitivity (optional) | `POST_TEMPLATE_ONLY_DEVELOPMENT` | **No** |

For any state containing `TEMPLATE_ONLY`, “template” means this written panel,
caption, and binder contract only.  It does **not** mean a rendered empty figure.
A blank forest, a `PENDING` box, a dummy coordinate, or a zero substituted for a
missing value is a prohibited result rendering.  The POST skeletons refuse on the
substring `TEMPLATE_ONLY`, so a rebound figure keeps its fail-closed behaviour
while recording the rebinding in its state token.

**Stage-19 disposition (2026-08-05, final).** The Stage-19 **development**
script will not be produced for this submission, and Stage-10 cascades from it.
Figures 3 and S5 are **not** affected: their target-period metrics are computed
independently by the trusted scorer inside the one-time opening
(`src/thermoroute/opening.py:8932-8948`), which emits the full probabilistic
family per cohort × model × horizon plus station-balanced reliability bins.  Both
figures keep their original designs and no panel is dropped.

Stage-19 is downgraded from a hard blocker to a **recorded provenance
qualifier**: its absence is reported in the render receipt and never refuses a
render.  The corrected failure numbers — 0 strict quantile crossings, 135
zero-width nominal intervals in 26,993,675 member-level rows — are in
`docs/STAGE19_DEGENERATE_INTERVAL_DISPOSITION_20260805.md`; the figure
determination is in `docs/FIGURE_PLAN_STAGE19_INDEPENDENT_20260805.md`.

The Stage-22 adaptive-conformal evidence is **development-period**
(2019-01-01 to 2020-12-24) and is governed by §2.1 layer 2.  It is confined to
Figure S9, which declares its evidence period and renders a mandatory in-panel
scope band.  It may never be read as a target-period result, and no value in it
may be compared numerically with Figure 3 or S5; the confirmatory target period
starts 2021-01-01.

**One figure never mixes two evidence periods.**  A figure binds either
target-period or development-period evidence, declares which, and — when
development — renders an in-panel scope band.  The POST skeletons enforce this.

---

## 2. Global evidence and binder contract

### 2.1 Authority layers

1. **PRE structural authority.** Frozen panel/registry facts, protocol geometry,
   implementation identities, and training-period descriptive quantities may be
   projected only from explicitly bound files.  They are not target-period
   results.
2. **Development-only authority.** Stage-09 or Stage-09b completion receipts may
   support clearly labelled development/exploratory diagnostics.  They cannot
   fill a target-period point, interval, probability, external-cohort, formal
   comparison, or verdict coordinate.
3. **POST authority.** A result coordinate requires all of the following:
   verified opening receipt, frozen model suite, bound prediction/evaluation
   source, claim registry, inference and outcome-QC gates, POST evidence manifest,
   and cell/mark-level binder mapping.  Missing or failed evidence is displayed as
   a bound `NA`/failure state when the registered display requires it; it is never
   silently dropped.
4. **Render authority.** A scientific value is still not a submission artifact
   until the render receipt binds the figure-data projection, vector/raster
   outputs, fonts, environment, page placement, and visual-QA result.

### 2.2 Mark-level binding

Every visible number, status, date, interval endpoint, axis annotation derived
from evidence, and plotted coordinate has one declared `value_id`.  The generic
figure binder shape is:

```text
figures.<figure_id>.panels.<panel_id>.marks[<mark_id>].cells.<field>.value_id
```

A coordinate with two empirical dimensions binds them separately:

```text
...cells.x.value_id
...cells.y.value_id
```

Every referenced value object contains at least:

```text
values.<value_id>.value
values.<value_id>.unit
values.<value_id>.evidence_role
values.<value_id>.source_pointer
values.<value_id>.derivation
values.<value_id>.rounding
```

Every figure also binds:

```text
figure_id
figure_schema_version
source_bindings[]          # path, SHA-256, format, semantic role
panel_order[]
mark_registry[]
caption_value_ids[]
scope_status_value_id
render_profile
```

One row-level source ID is insufficient for a multi-value mark.  Reusing one ID
for visibly different values, independently recomputing a value already used by
a main table, or using different rounding for the same visible quantity rejects
the render.  Figure 2's formal rows and main Table T2 must use the same value IDs.

### 2.3 Common numerical rules

- Point comparisons use identical station/date/horizon target keys for each
  registered model pair.
- A station/horizon RMSE enters a registered effect only with at least 100 valid
  paired targets for both models.
- The formal effect is the unweighted median of station-level
  candidate-minus-reference RMSE.  Negative values favour ThermoRoute.
- All-model displays identify whether they use one all-model common key set or
  pairwise common keys; the two denominator systems are never mixed in one mark.
- Formal HUC2 intervals and p-values remain assumption-conditional sensitivities.
  The permanent fixed-cohort descriptive status is rendered regardless of their
  numerical direction.
- Probability summaries give every retained station equal total weight.  Raw row
  counts and station-balanced rates are labelled as different quantities.
- Three-quantile pinball uses the nominal member-averaged pre-CQR heads; coverage
  and width use the deployed CQR interval; event metrics use the post-Platt
  probability.  All three clauses hold at target period: the trusted scorer
  inside the one-time opening computes them, so the withheld Stage-19
  development script does not affect them.
- A coverage or width number is always reported next to the interval score that
  buys it.  An adaptive conformal variant is never described as cost-free, and a
  non-finite width is bound as an explicit NA with its row count, never clipped,
  dropped, or imputed.
- The +0.05 degrees C comparison ceiling is a frozen numerical threshold, not an
  ecological, regulatory, measurement-error, or stakeholder-importance margin.

---

## 3. Unified visual tokens

### 3.1 Semantic palette

Colors encode roles, not decoration.  The same role keeps the same color in every
main and SI figure.

| Token | Hex | Meaning |
|---|---|---|
| `TR_BLUE` | `#0072B2` | ThermoRoute point/result series |
| `TR_BLUE_LIGHT` | `#DCEAF4` | ThermoRoute band or module fill |
| `DAMPED_ORANGE` | `#E69F00` | damped-persistence anchor/reference |
| `PERSIST_GRAY` | `#777777` | ordinary persistence |
| `CLIM_GRAY` | `#B8B8B8` | climatology |
| `LGBM_PURPLE` | `#CC79A7` | LightGBM |
| `LSTM_GREEN` | `#009E73` | global LSTM |
| `ALLOWED_TEAL` | `#008C7A` | allowed information or verified path |
| `ALLOWED_TEAL_LIGHT` | `#DCEFEA` | allowed-region fill |
| `WARNING_VERMILION` | `#D55E00` | limitation or prohibited input; never “bad model” |
| `WARNING_LIGHT` | `#F9E3D6` | limitation box fill |
| `GATE_RED` | `#B2182B` | failed/blocked claim gate only; never effect direction |
| `NEUTRAL_INK` | `#202020` | text and axes |
| `NEUTRAL_GRID` | `#D0D0D0` | grid/rules |
| `NA_FILL` | `#F2F2F2` | explicit bound NA/not-estimable status |

No result is encoded by red/green alone.  Series also use stable marker shapes,
line styles, or hatching.  Warning red is reserved for scope/gate state and must
not imply that a numerically positive or negative effect is statistically
decisive.

### 3.2 Typography and geometry

- Final full-width target: 140 mm, matching the 5.5-inch text width in the
  committed AGU class; single-column target: 85 mm only when every label remains
  legible.  A 180-mm export is a publisher-requested variant, not the governing
  placed-size assumption.
  Rationale for 85 mm (revised from 88 mm on 2026-08-05): AGU's published
  single-column range is 50--85 mm.  A figure authored at 88 mm is scaled to
  85/88 = 96.6% during production, which pulls 7.5 pt ticks down to 7.24 pt and
  8 pt axis text to 7.73 pt -- below the floor declared in the next bullet.
  Authoring at 85 mm removes the scaling step so the type-size floor holds as
  stated.
- Body and axis text target 8 pt or larger; the absolute final-size floor is
  7.5 pt for ticks, legends, or compact annotations.  Panel labels target
  9--10 pt bold.
- Preferred font: journal-compatible sans serif with embedded glyphs.  Do not
  rely on an unembedded CJK font in the submission artifact.
- Panel labels are `(a)`, `(b)`, etc., at a consistent upper-left offset.
- Minimum line width: 0.6 pt; primary series 1.1--1.4 pt; reference lines at least
  0.8 pt.
- Background is white.  No gradients, shadows, 3-D marks, rounded-card dashboard
  aesthetic, pictorial icons, or decorative river imagery.
- Count bars start at zero.  A non-zero or log axis is permitted only when clearly
  marked in the panel and caption.
- Effect plots show the zero line and every registered margin/ceiling.  Axis
  limits are fixed by a declared rule, not hand-selected after viewing results.
- Distribution displays prefer raw station dots, ECDFs, or box summaries over a
  smoothed violin whose bandwidth can imply unsupported density structure.
- Legends are shared when panels use the same series; direct labels are preferred
  when they do not overlap.

### 3.3 Caption grammar

Every caption contains, in this order:

1. what is shown;
2. cohort/time period and information set;
3. aggregation, denominator, and uncertainty construction;
4. one receipt-rendered or design-level takeaway;
5. the claim boundary needed to prevent over-interpretation.

Captions do not merely say “performance comparison.”  They also do not contain a
directional adjective before the bound POST value exists.

### 3.4 Export and page QA

- Preferred scientific output: SVG plus PDF with embedded fonts.  Raster panels,
  if unavoidable, require at least 300 dpi at placed size; line art requires at
  least 600 dpi.
- SVG XML, PDF bounding boxes, font embedding, color-vision simulation, grayscale
  differentiation, clipping, overprint, panel order, legend completeness, and
  caption references are checked before acceptance.
- A second render from identical evidence must be byte-identical or satisfy a
  stricter declared normalization/tolerance rule.
- The final TeX/PDF must have no missing figure, undefined reference, overflow,
  clipped label, or orphan caption.

---

## 4. Main-text figure specifications

## Figure 1 — Why constrained predictions require constrained claims

**State:** `PRE_MATERIALIZED_REDRAW`
**Research question:** What structural mismatch motivates ThermoRoute, and what
single design principle resolves it without overstating the evidence?
**Narrative role:** quantified mismatch -> bounded mechanism -> evidence boundary.

### Panel structure and plot types

**(a) Persistence challenge.** Plot one station-level point per retained site for
each horizon $h\in\{1,3,7\}$, where the point is a frozen training-period
descriptive statistic such as the median observed
$\lvert T_{t+h}-T_t\rvert$.  Use three aligned dot/box summaries.  The calculation is
restricted to 2006--2015, exact calendar-day pairs, finite observed WTEMP at both
ends, no target imputation, and equal station representation.  It is a motivation
diagnostic, not a model score or confirmation result.

**(b) Bounded correction.** Show the damped anchor $A_{t+h}$, unrestricted
learned displacement $z_{t+h}=P_{t+h}-A_{t+h}+r_{\theta,t+h}$, and

\[
\widehat y_{t+h}=A_{t+h}+\delta\tanh(z_{t+h}/\delta),
\qquad \delta=1.0\ ^\circ\mathrm C.
\]

Use an anchor line and a shaded $A\pm\delta$ envelope.  The warning
“Deviation from anchor; not an error or safety bound” is part of the panel, not a
footnote hidden in the caption.

**(c) Dependence-aware evidence and claim boundary.** Plot the 15
registry-derived HUC2 station counts and show the dimensional collapse from
657,480 site-days to 120 sites and 15 pre-attrition HUC2 groups.  Beside the
bars, render the three frozen gate checks without a shared false numeric axis:
15 versus at least 30 groups, $9.54/15=0.636$ versus at least 0.75 effective
fraction, and 21.7% versus less than 25% largest-group share.  Continue with a
short left-to-right evidence spine:

```text
date-indexed allowed history
  -> exact model-pair keys
  -> station-level RMSE
  -> whole-HUC2 sensitivity
  -> claim/QC gate
  -> fixed-cohort descriptive statement
```

The target outcome and horizon-specific future weather are shown outside the
predictor input boundary.  The panel says that the frozen date-index contract
excludes those inputs; it does not claim archived as-issued availability or proof
of local-day alignment.

### Caption takeaway

> Strong short-horizon persistence and the collapse from daily rows to a small,
> unbalanced set of spatial groups create two coupled risks: unconstrained learned
> corrections can depart from a strong reference, and row-level evaluation can
> overstate evidential certainty. ThermoRoute bounds the point correction around
> damped persistence and separately bounds the eligible claim through exact-key,
> station-balanced, HUC-aware, fail-closed evaluation. The algebraic envelope is
> not a truth-error, ecological, regulatory, or deployment-safety guarantee.

### Data fields and value IDs

| Panel | Required fields/values | Required source role |
|---|---|---|
| (a) | `site_no`, `date_t`, `date_th`, `horizon_days`, `wtemp_t`, `wtemp_th`, `observed_pair`, `site_statistic`, `n_pairs` | frozen 2006--2015 panel and registry; training-only descriptive derivation |
| (b) | anchor identity, `delta_scale`, equation/config identity, source hash | implementation/configuration binding |
| (c) | panel rows, site count, pre-attrition HUC2 count, HUC2 counts, effective cluster count/fraction, largest-group share, allowed-variable/date roles, aggregation roles, inference/QC gate identities, fixed scope status | frozen panel/registry/environmental audit plus protocol, claim registry, and information-boundary contract |

Every station marker in panel (a) binds its site identity, horizon, statistic, and
pair count.  Every displayed summary line has a separate value ID; it is not
recomputed by the plotting layer.

### Gate

- **PRE:** permitted after source-hash validation, declared panel-(a) derivation,
  no access to post-2020 outcomes, and visual QA.
- **POST:** Figure 1 remains the same design/motivation role; it is not refilled
  with confirmation effects.  POST may update only bound provenance/status text,
  not replace panel (a) with target performance.

### Prohibited semantics

“No leakage” without qualification; “15 reportable HUC2 groups”; independent
river-network components; national representation; safe prediction; bounded
error; causal thermal transport; superiority; a target-period forest; any
`PENDING` result cell.

### Acceptance

- A reader can state the mismatch, insight, and evidence boundary from the figure
  and caption alone.
- All panel-(a) values are training-only and station-balanced.
- The panel-(c) gate quantities are not visually compared on a false common scale.
- The bound warning remains legible at final placed size.
- No target-period score, interval, p-value, or verdict is present.

---

## Figure 2 — Point performance on frozen common keys

**State:** `POST_TEMPLATE_ONLY`
**Research question:** On the fixed observable cohort, how does ThermoRoute's point
performance compare with all six primary models and with the five registered
references?
**Narrative role:** primary end-to-end evidence.

### Panel structure and plot types

**(a)--(c) All-model station distributions by horizon.** One small multiple for
each of 1, 3, and 7 days.  Plot station-level RMSE points or ECDFs for persistence,
damped persistence, climatology, LightGBM, global LSTM, and ThermoRoute on one
declared all-model exact-common-key set.  Show retained station counts.

**(d) Registered five-row forest.** Plot the unweighted median station-level
ThermoRoute-minus-reference RMSE and whole-HUC2 bootstrap interval for all five
registered rows.  Draw the 0.00 degrees C line for damped-persistence rows and
the +0.05 degrees C ceiling for LightGBM rows.  Put status, station count, and
cluster count beside each row.  Raw p, Holm p, win rate, and bound checks belong
in main Table T2 and share its value IDs; do not encode them as significance
stars.

### Caption takeaway

The caption's directional result sentence is generated only from the receipt.  Its
fixed ending is:

> All five comparisons describe the frozen observable cohort. HUC2 intervals,
> exact sign-flip p-values, and Holm values are assumption-conditional
> sensitivities under `DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED`; they do not
> establish superiority, non-inferiority, equivalence, parity, or a U.S.-river
> superpopulation result.

### Data fields and value IDs

| Display | Required values |
|---|---|
| all-model mark | model ID, horizon, site ID, common-key digest, paired-key count, station RMSE, reportability status |
| forest mark | test ID, candidate/reference IDs, horizon, margin, status, effect, CI low/high, station count, cluster count, gate verdict |
| T2 companion | win rate, raw p, Holm p, margin checks and their exact shared value IDs |

Source pointers resolve to trusted POST predictions, `formal_tests[*]`, the
opening receipt, and gate artifacts.  Selective omission of a model, site, horizon,
or formal row rejects the figure.

### Gate

- **PRE:** schema text only.  Do not generate an SVG/PDF/PNG or placeholder
  forest.
- **POST:** complete opening/evidence/render receipts; exact key registry; all
  five formal rows exactly once; all six primary models; bound NA handling;
  permanently descriptive gate text.

### Prohibited semantics

Significance coloring or stars; “wins,” “beats,” “non-inferior,” “equivalent,” or
“parity”; a national claim; row-wise pooled uncertainty presented as station
uncertainty; hidden non-estimable rows; development-cache substitution; a y-axis
or x-axis chosen after inspecting favourable values.

### Acceptance

- Negative effect direction and margin interpretation are explicit.
- All-model and formal-pair denominators are separately labelled.
- Figure 2 and T2 share value IDs and rounding.
- Counts and NA rows remain visible.
- Caption contains a receipt-rendered reading, mechanism-neutral explanation, and
  permanent scope boundary.

---

## Figure 3 — Marginal intervals and event-probability behavior

**State:** `POST_TEMPLATE_ONLY`
**Restored:** 2026-08-05, superseding the same-day interim rebinding onto
Stage-22.  See `docs/FIGURE_PLAN_STAGE19_INDEPENDENT_20260805.md` §3.
**Research question:** How sharp and empirically calibrated are the frozen
uncertainty outputs, and what probability discrimination/calibration trade-offs
remain?
**Narrative role:** benefit-plus-cost evidence for probabilistic outputs.
**Evidence period:** target, 2021-01-01 through 2023-12-31.

### Stage-19 does not affect this figure

The withheld `scripts/19_probabilistic.py` is a **development** tool.  The
target-period probabilistic family is computed independently by the trusted
scorer inside the one-time opening.  `src/thermoroute/opening.py:8932-8948`
emits, per cohort × model × horizon:

`coverage_90`, `mean_interval_width_c`, `pinball_q05_c`, `pinball_q50_c`,
`pinball_q95_c`, `equal_weight_three_quantile_pinball_mean_c`, `brier_score`,
`frozen_reference_brier_score`, `brier_skill_frozen_seasonal`, `log_loss`,
`auroc`, `auprc`, `ece_10_equal_width`, `calibration_intercept`,
`calibration_slope`, `event_rate`

plus station-balanced reliability bins whose weights must sum to one
(`opening.py:8920-8926`), against a frozen seasonal event reference validated by
`validate_frozen_seasonal_event_reference`.  That is a superset of what this
figure requires, at exactly the granularity it requires.  The original design is
therefore retained in full and no panel is dropped.

The development-period Stage-22 conformal evidence is **not** used here.  It is
confined to Figure S9.

### Panel structure and plot types

**(a) Coverage--width plane.** Plot station-balanced empirical 90% marginal
coverage against mean interval width for every eligible learned model and
horizon.  Use horizon markers and model colors; draw the 0.90 nominal reference
without implying a formal coverage test.  Point-only models bind a
`NOT_AVAILABLE` status and never receive invented heads.

**(b) Event score.** Plot Brier skill against the frozen seasonal reference by
model and horizon.  Show the zero-skill line and the bound reference identity.
The reference is the frozen seasonal climatology; confirmation-period event
prevalence is never used as the Brier reference.  Log score and three-quantile
pinball remain in SI/Table unless a predeclared layout requires them.

**(c)--(e) Reliability.** One panel per horizon.  Plot observed station-balanced
event frequency against mean forecast probability with the identity line.  Point
area represents the bound bin denominator or station-balanced effective weight;
empty bins are retained as explicit empty-bin annotations.

### Caption takeaway

> The figure jointly reports empirical marginal coverage and width, evaluates
> event probabilities against a frozen seasonal reference, and shows the support
> behind each reliability bin. These are station-balanced descriptive diagnostics;
> they do not establish conditional coverage, operational calibration, or decision
> value, and the equal-weight three-quantile pinball summary is not CRPS.

### Data fields and value IDs

`cohort`, `model`, `horizon`, pre/post-reportability forecast counts, site count,
minimum targets, station-weight audit, `coverage_90`,
`mean_interval_width_c`, nominal q05/q50/q95 pinball components,
`brier_score`, frozen-reference Brier score, Brier skill, log loss, AUROC, AUPRC,
ECE, calibration intercept/slope, event/non-event counts, event rate, reliability
bin ID/bounds/denominator/mean probability/observed frequency, undefined reason,
and probability-pipeline source identities.

Each reliability x coordinate, y coordinate, and point-size value has a separate
value ID.  Point-only models bind a `NOT_AVAILABLE` status and do not receive
invented heads.

### Gate

- **PRE:** schema text only; no axes or dummy reliability points may be rendered.
- **POST:** opening receipt; `trusted/probabilistic_evaluation_v2.json`;
  `trusted/temporal_predictions_v1.parquet`;
  `trusted/availability_registry_v1.csv`; erratum binding; confirmatory protocol;
  SI08; render receipt.

### Pre-opening guard (blocking risk, not a figure risk)

`opening.py:7093-7094` applies a **strict** `q05 < q95` to the member-averaged
nominal heads and raises `OpeningContractError` — aborting the entire one-time
opening — on violation.  This is the Stage-19 degeneracy trap one layer up.
Measured on the development panel: member averaging clears every affected
LightGBM key (5 members); the only 12 survivors are single-member
`LightGBM-perstation` keys, and that model appears in neither `PRIMARY_MODELS`
nor the confirmatory protocol.  **Re-run this check on the target-period
predictions before executing the one-time opening.**

### Prohibited semantics

Conditional coverage; distribution-free target-period guarantee; CRPS; operational
forecast reliability; economic value; merged or silently removed empty bins;
unreported single-class/fit-failure NA; coverage without width; using confirmation
event prevalence as the Brier reference; "quantile crossing" as the Stage-19
cause; development-period conformal numbers presented as target-period results.

### Acceptance

- Coverage and sharpness are read together.
- The quantile/CQR/Platt source for every metric is correct.
- Every bin exposes support and weighting.
- Undefined metrics remain visible with their reason.
- Identity, nominal, and zero-skill references are visually distinct and named.
- No value in this figure is development-period.

## Figure 4 — Mechanism sensitivities and applicability boundaries

**State:** `POST_TEMPLATE_ONLY`
**Research question:** Which registered interventions change the prediction, and
over what observable temporal and external-site scope do the reported effects
remain descriptive?
**Narrative role:** mechanism evidence -> sensitivity -> boundary.

### Panel structure and plot types

**(a) Architecture interventions.** Use a horizon-by-control dot matrix for all
seven registered one-factor controls: prior-only, no dynamic prior, fixed
relaxation, no router, no mixture, no TCN, and unbounded residual.  Plot paired
station-level RMSE difference from full ThermoRoute and show seed/member
completeness.  The bounded/unbounded pair also reports the algebraic-deviation
violation audit.  Do not select controls after seeing their values.

**(b) Availability/attrition.** Use a horizon-specific waterfall that keeps
calendar opportunities, observed issue WTEMP, observed target WTEMP, exact paired
keys, retained stations, and reportable clusters as distinct stages.

**(c) Temporal sensitivity.** Plot all eight frozen descriptive candidates:
equal weighting of the 12 year-by-season cells, three leave-one-year values, and
four leave-one-season values.  Mark the deterministic unfavourable value; retain
the formal effect as a distinct reference and never replace it.

**(d) External history-dependent arm.** Show the six primary models by horizon on
the exact external-arm key set.  The panel title includes “site-ID disjoint,
history-dependent; not ungauged.”  It may show paired effects or station-RMSE
distributions, but not a river-network transfer map.

### Caption takeaway

> The registered controls diagnose sensitivity to individual architecture
> interventions, while the attrition, temporal, and external panels delimit the
> observable cohort on which performance is described. The controls are
> exploratory and noncausal; favourable temporal sensitivity cannot replace the
> formal effect, and the external arm uses target-site water-temperature history
> through each issue date.

### Data fields and value IDs

| Panel | Required values |
|---|---|
| (a) | control/model ID, exact intervention, seed/member registry, key digest, site/horizon effect, bound-violation count/rate, suite/receipt lineage |
| (b) | horizon, stage ID, eligible-before, retained-after, exclusion reason, site/cluster counts, denominator role |
| (c) | test ID, sensitivity ID/order, candidate definition, effect, support, formal-effect reference, deterministic-worst flag |
| (d) | external cohort binding, site-disjoint audit, history requirement, model/horizon, exact keys, site count, score/effect, status |

Stage-09b PlainMLP/PlainCausalTCN and feature-ladder results remain explicitly
development-only.  They may be tabulated in SI09 under their own evidence role;
they may not be substituted for panel (a)'s POST target sensitivities.

### Gate

- **PRE:** schema text only.  No rendered panel may contain Stage-09 or Stage-09b
  development numbers.
- **POST:** all seven target control rows under the final model/seed contract,
  verified temporal-coverage and external receipts, exact denominator bindings,
  and render receipt.  If any panel lacks authority, Figure 4 is not generated;
  a development panel is not substituted.

### Prohibited semantics

Component necessity; causal attribution; capacity-matched claim unless the exact
comparison proves it; post hoc control selection; “all calendar days”; stability
across years/seasons; ungauged prediction; hydrologic independence; network
transfer; rescue of a failed or unfavourable formal row.

### Acceptance

- All seven interventions and all eight temporal candidates are present.
- Every waterfall stage has a named denominator and exclusion reason.
- The external scope warning is visible inside the panel and caption.
- Development-only and POST values are never mixed.
- A boundary result is discussed even when point performance is favourable.

---

## 5. Supporting-figure specifications

## Figure S1 — Cohort selection and registry geometry

**State:** `PRE_MATERIALIZED`
**Question:** What fixed, availability-enriched sample was assembled, and how is
its spatial grouping distributed?
**Panels/plot types:** `(a)` candidate-selection waterfall
(1,465 candidates, rejection categories, 120 retained); `(b)` station-coordinate
map colored by HUC2 with no coverage polygon; `(c)` zero-based HUC2 station-count
bars; `(d)` compact registry diagnostics for repeated HUC codes and nearest-site
distance/support, without implying hydraulic connectivity.

**Caption takeaway:** The fixed sample is geographically broad but selected for
data availability, spatially clustered, and not a probability sample or a
nationally representative set of U.S. rivers. The committed ledgers audit the
selection outcome but do not reconstruct the original discovery execution.

**Fields/value IDs:** candidate and rejection IDs/reasons; retained site ID;
latitude/longitude/state/HUC2; pre-attrition HUC2 count; repeated-HUC and proximity
diagnostics; all numerators/denominators; registry/panel/rejection-ledger hashes.
Any basemap or HUC boundary source has its own citation, version, license, and
source binding.

**Gate:** PRE registry, panel, rejection ledger, environmental audit, projection
reconciliation, map-source rights, and visual QA.  No POST outcome is required.

**Forbidden:** national coverage; independent reach or river-network component;
upstream/downstream arrows; regulation inference from station names; target RMSE
or skill on the map.

**Acceptance:** waterfall counts reconcile exactly; HUC2 bars sum to 120; all map
sites resolve to one registry row; missing coordinates are explicit; map legend
does not imply representativeness.

---

## Figure S2 — Temporal chronology and information/product boundary

**State:** `PRE_MATERIALIZED`
**Question:** Which information is available at each stage, and what can the
predictor-product bridge establish?
**Panels/plot types:** `(a)` 2006--2015 train, 2016--2017 validation, 2018
calibration, 2019--2020 inspected exploratory development, and later one-time
target chronology; `(b)` issue-time allowed/forbidden timeline at horizons 1, 3,
and 7 days; `(c)` variable-by-provider/source-date matrix; `(d)` two-column bridge
capability/limitation matrix.

**Caption takeaway:** Route A is a date-indexed retrospective hindcast with a
frozen outcome-access sequence. It excludes target outcomes and
horizon-specific future forecasts from predictor inputs, but it does not replay
archived as-issued predictor vintages or prove NWIS/provider local-day alignment.

**Fields/value IDs:** split dates/roles; issue date; target date; horizon;
variable ID; provider/product; source date; retrieval/vintage reference;
admissibility; missing/fail-closed state; bridge path/hash/status; declared bridge
capabilities and non-capabilities.

**Gate:** frozen input/product contract, protocol chronology, and
`PASS_EXACT_PRODUCT_BRIDGE` binding.  No realized target availability is drawn.

**Forbidden:** operational replay; exactly-as-issued forecast; ungauged input set;
causal transport; “no leakage” as an unqualified empirical guarantee.

**Acceptance:** every variable has one source/date rule; forbidden inputs are
outside the model arrow; 2019--2020 is visibly exploratory; bridge limitations
are as prominent as its PASS status.

---

## Figure S3 — Full model, bound, and calibration dataflow

**State:** `PRE_MATERIALIZED_DESIGN`; final authoritative render requires the
frozen model suite.
**Question:** How do allowed histories produce point, interval, and event outputs,
and which invariants apply?
**Panels/plot types:** `(a)` seven-variable/missing-mask input tensor and derived
context; `(b)` learned relaxation proposal plus sparse variable/lag router,
strictly left-looking TCN, and mixture; `(c)` separate MSE point, q05/q50/q95,
and event heads plus the anchor-bound identity; `(d)` member averaging, 2018 CQR,
2018 Platt, and deployment-output dataflow.

**Caption takeaway:** ThermoRoute uses a learned proposal and temporal allocation
modules inside a statistical predictor, while the final point and q50 corrections
are numerically constrained about the frozen damped anchor. CQR may retain or
widen the nominal interval and leaves q50 unchanged. No latent variable identifies
physical travel time, residence time, heat transfer, regulation, or a causal
driver.

**Fields/value IDs:** ordered input registry; mask roles; 32-day construction
buffer; maximum router lag 14; TCN blocks/kernel/receptive field 7; mixture and
head identities; WLEVEL-disabled status; delta; anchor/proposal identities;
member count; CQR/Platt fit periods and scopes; raw/deployed qhat rules; source
and bundle hashes.

**Gate:** a PRE design may use bound implementation/configuration facts.  The
final paper version requires exact parity with the frozen suite and replayed
bundle metadata; architecture drift invalidates the art.

**Forbidden:** 32-day effective memory; network routing; causal attention;
physical κ; point=q50 alias; mixture-distribution quantiles; interval/safety bound
conflation; fitted coefficient or performance value before receipt authority.

**Acceptance:** 32-day buffer, lag-14 router, and seven-step TCN receptive field
are visibly distinct; q05/q50/q95 and point heads are not conflated; CQR and Platt
operate on the correct artifacts/period; the non-safety statement is legible.

---

## Figure S4 — Point-performance heterogeneity

**State:** `POST_TEMPLATE_ONLY`
**Question:** Which stations and HUC2 groups underlie the aggregate point effects?

**Panels/plot types:** HUC2-sorted site-by-horizon heatmap of paired
ThermoRoute-minus-reference RMSE; ECDFs of station effects; optional HUC2 small
multiples using one shared declared scale.  This figure expands Figure 2 and does
not repeat its aggregate forest.

**Caption takeaway:** Aggregate medians can mask spatially heterogeneous fixed-
cohort effects; the display preserves every reportable station and explicit NA
without converting HUC2 patterns into independent river-network or national
evidence.

**Fields/value IDs:** site/HUC2/horizon/model pair; exact key digest; station RMSE
for both models; paired difference; paired target count; reportability/NA reason;
sorting registry; shared scale limits.

**Gate:** verified POST predictions, exact keys, opening receipt, and QC/gate
bindings.  PRE rendering is prohibited.

**Forbidden:** selective sites; per-panel adaptive color scales; national map
interpolation; hydrologic connectivity; causal spatial explanation; omitted NA.

**Acceptance:** every retained/NA station reconciles with Figure 2/T2; shared
scale and zero are clear; HUC sorting is deterministic; small cells remain
legible at final size.

---

## Figure S5 — Expanded probabilistic diagnostics

**State:** `POST_TEMPLATE_ONLY`
**Restored:** 2026-08-05, superseding the same-day interim rebinding onto
Stage-22.  See `docs/FIGURE_PLAN_STAGE19_INDEPENDENT_20260805.md` §3.
**Question:** Do the aggregate probabilistic summaries in Figure 3 conceal model,
horizon, station, or bin-level failure?
**Evidence period:** target, 2021-01-01 through 2023-12-31.

**Stage-19 does not affect this figure.** The SI08 metric family is produced at
target period by the trusted scorer inside the one-time opening
(`src/thermoroute/opening.py:8932-8948`), per cohort × model × horizon, with
station-balanced reliability bins.  The original expanded design is retained in
full and no panel is dropped.  The development-period Stage-22 conformal evidence
is confined to Figure S9 and is never mixed into this figure.

**Panels/plot types:** `(a)` full model-by-horizon coverage/width dot matrix,
every cell bound or explicit NA; `(b)` pinball, interval, Brier, and log-score
matrix with an explicit scoring-stage legend distinguishing nominal pre-CQR
heads, the deployed CQR interval, and the post-Platt probability; `(c)`
reliability panels with every registered bin and count, empty bins explicit, and
station-balanced bin weights reconciling to one; `(d)` calibration
slope/intercept and AUROC/AUPRC/ECE discrimination diagnostics with bound NA
reasons for single-class or fit-failure cases.  This figure expands Figure 3 and
does not repeat its aggregate plane.

**Caption takeaway:** Probability diagnostics are reported with their scoring
stage, station-balanced weighting, support, and non-estimability state; no single
coverage number is treated as a conditional or distribution-free guarantee.

**Fields/value IDs:** all SI08 metric fields, probability source stage, model and
horizon counts, every bin boundary/statistic/denominator/station-balanced weight,
undefined reasons, reference identity, event-reference binding, threshold scope,
and calibration-fit status.

**Gate:** opening receipt; `trusted/probabilistic_evaluation_v2.json`;
`trusted/temporal_predictions_v1.parquet`;
`trusted/external_predictions_v1.parquet`;
`trusted/availability_registry_v1.csv`; erratum binding; SI08; confirmatory
protocol.  PRE rendering is prohibited.

**Forbidden:** three-quantile score labelled CRPS; silent metric substitution;
empty-bin merging; conditional-coverage language; model rows with invented heads;
distribution-free target-period guarantee; confirmation event prevalence as the
Brier reference; "quantile crossing" as the Stage-19 cause; development-period
conformal numbers presented as target-period results.

**Acceptance:** all registered metrics appear or bind explicit NA; reliability
support reconciles to parent counts; the scoring-stage legend prevents pre-/post-
calibration conflation; no value in this figure is development-period.

---


## Figure S6 — Temporal opportunity, missingness, and attrition

**State:** `POST_TEMPLATE_ONLY`
**Question:** Which calendar opportunities become observable and reportable
forecast keys, and how do frozen time sensitivities change the descriptive
effect?
**Panels/plot types:** denominator-preserving Sankey/waterfall; horizon-by-year
and season opportunity heatmap; complete eight-candidate sensitivity dot plot;
retained-row block/feedback sensitivity with the exact block semantics.

**Caption takeaway:** Route A evaluates observable issue/target keys rather than
all calendar days. Temporal sensitivities diagnose this fixed observable cohort
but cannot establish missing-at-random, year stability, season stability, real
feedback latency, or conditional coverage.

**Fields/value IDs:** calendar opportunity, eligible issue, observed issue WTEMP,
observed target WTEMP, exact paired key, retained station, completeness,
missingness reason, year/season/block ID, effect/score, support, formal-effect
reference, and feedback-date proxy role.

**Gate:** SI10/SI14 and temporal-coverage receipts physically replayed and bound
to opening evidence.  PRE target counts or placeholder marks are prohibited.

**Forbidden:** seven retained rows called seven calendar days; all-calendar
performance; real publication-latency replay; favourable sensitivity used to
replace/rescue the formal effect; imputed unavailable outcomes.

**Acceptance:** every stage retains a named denominator; flows reconcile exactly;
all eight candidates appear in frozen order; calendar and retained-row blocks are
visually distinguished.

---

## Figure S7 — Spatial and leave-HUC2 influence

**State:** `POST_TEMPLATE_ONLY`
**Question:** How sensitive are fixed-cohort effects to HUC2 composition and
single-cluster omission?
**Panels/plot types:** per-HUC2 effect/count dot plot; leave-one-HUC2 effect plot;
cluster-share/effective-count diagnostics; permanent claim-gate status box.

**Caption takeaway:** Spatial sensitivities expose influence from the unbalanced,
coarse HUC2 grouping. Numerical stability under leave-one-HUC2 omission does not
make clusters independent and cannot upgrade Route A beyond fixed-cohort
description.

**Fields/value IDs:** cluster definition/version; HUC2/omitted unit; station and
cluster counts; largest share; effective count/fraction; per-HUC and leave-one
effect; interval/status; gate inputs/verdict; registry/UQ source binding.

**Gate:** verified SI11 spatial-sensitivity receipt, registry binding, opening
receipt, and fixed gate status.  PRE may supply geometry to S1 but must not render
S7 result axes.

**Forbidden:** HUC2 as independent network component; national inference;
superiority/non-inferiority; stable leave-one-HUC as proof of robustness;
post-outcome alternative clustering.

**Acceptance:** all HUC2 groups and omissions appear or bind NA; counts reconcile
with reportability; gate thresholds and actual inputs are distinct; scope warning
is visible without relying on caption-only text.

---

## Figure S8 — Outcome QC, external-history scope, and failure disposition

**State:** `POST_TEMPLATE_ONLY`
**Question:** What target evidence survived frozen QC, what exactly is external
about the external arm, and which failures remain in the record?
**Panels/plot types:** `(a)` raw-response -> normalized-series -> exact-A retained
QC waterfall with qualifier/conflict categories; `(b)` development/external
site-ID disjointness and history-dependence diagram plus external results; `(c)`
failure/attrition matrix by declared reason and disposition.  If these panels are
not legible as one figure, keep their exact tables in SI12--SI14 rather than
compressing them into decorative graphics.

**Caption takeaway:** Target values remain traceable to immutable raw evidence,
and exclusions/conflicts/failures stay visible. The external cohort is disjoint
only by exact site ID and still uses its own observed WTEMP history; neither QC
survival nor metadata disjointness establishes ungauged, network-independent, or
operational performance.

**Fields/value IDs:** raw request/response and series IDs; statistic/unit;
qualifier/method subset; raw/retained/excluded/conflict counts; QC status and
exact-A binding; external site registry/disjointness/history fields; exact keys,
scores/effects/status; failure reason, before/after counts, disposition, and
source pointers.

**Gate:** SI12/SI13/SI14 receipts, immutable raw bytes, QC gate, external suite,
opening receipt, and failure-case bindings.  PRE rendering is prohibited because
the external cohort and target QC/result evidence are not complete.

**Forbidden:** site replacement; post hoc threshold tuning; suppressed conflicts
or failed cases; qualifier cherry-picking; ungauged or river-network transfer;
national inference; regulatory compliance; external metadata treated as outcome
authority.

**Acceptance:** every waterfall and failure count reconciles; exact-A is defined
as reproduction under frozen QC rather than permission to alter the cohort;
external history warning is in-panel; adverse and non-estimable rows remain.

---

## Figure S9 — Development-period conformal calibration sensitivity

**State:** `POST_TEMPLATE_ONLY_DEVELOPMENT`
**Added:** 2026-08-05.  See `docs/FIGURE_PLAN_STAGE19_INDEPENDENT_20260805.md` §4.
**Question:** Over the development period, how sensitive is interval validity to
the choice of conformal calibration method, and what does adaptivity cost?
**Evidence period:** **development, 2019-01-01 through 2020-12-24** — not a
target-period result.

This figure carries the Stage-22 adaptive-conformal evidence
(`scripts/22_adaptive_conformal.py`; 249,072 exact keys, 120 sites, 15 HUC2
groups, leads {1, 3, 7}).  It is a **separate** figure rather than a panel of S5
because S5 is target-period and this is development-period, and one figure never
mixes two evidence periods.  It is **optional**: dropping it weakens no
registered claim, because nothing in the confirmatory five-test family depends on
it.

**Panels/plot types:** `(a)` method-by-slice coverage matrix over the five frozen
methods (`split-CQR`, `7-retained-row block-max CQR`, idealized delayed ACI at
γ ∈ {0.005, 0.02, 0.05}) × five slices (overall, warm train-q90 tail, leads
1/3/7 d), every cell bound or explicit NA with key counts; `(b)` width and
interval-score matrix on the same grid, with non-finite cells rendered as bound
NA carrying their row counts (44 / 782 / 4,068 at γ = 0.005 / 0.02 / 0.05) rather
than clipped or imputed numbers; `(c)` per-HUC2 dispersion across all 15 groups
under equal-station weighting against unweighted row rates, labelled as two
different quantities; `(d)` ACI feedback-update counts by γ with the in-panel
statement that `target_date` is an idealized feedback proxy and that no verified
publication timestamp, source revision, or reporting-latency record exists.

**Caption takeaway:** Over the 2019--2020 development period, interval validity
is reported for five frozen conformal calibration systems together with the width
and interval score that buy it. These are station-balanced descriptive
development diagnostics on a fixed cohort. They are not target-period results,
they cannot be compared numerically with Figures 3 or S5, and no adaptive variant
is cost-free.

**Fields/value IDs:** `conformal.*` (method ID/definition, slice, lead, key and
site counts, marginal coverage, mean width, interval score, nominal target,
non-finite width count and reason, ACI feedback count, feedback proxy role, block
semantics note); `weighting.*`; `huc2.*`; `provenance.*` (evidence period, target
start, source digest).

**Gate:** opening receipt; Stage-22 row table and report; confirmatory protocol;
bound evidence-period declaration; in-panel scope band
(`figS9.scope.development_period_not_confirmation`).  PRE rendering is prohibited.

**Forbidden:** target-period coverage or any confirmatory interval claim;
conditional coverage; distribution-free finite-sample guarantee; adaptive
conformal presented as cost-free; silently dropped or clipped non-finite widths;
"seven calendar-day blocks" (the block method uses 7 **retained rows**); real
feedback latency or publication-vintage replay; station-balanced and row-count
rates merged into one mark; numerical comparison against Figure 3 or S5 as if the
two were the same cohort.

**Acceptance:** every method × slice cell appears or binds explicit NA;
non-finite widths appear with their counts; the two weighting denominators are
separately labelled; the evidence-period scope band is legible inside the figure.

---

## 6. Explicit retirements and removals

### 6.1 Current pre-opening SVG

The former four-panel bytes at
`paper/agu_submission/figures/fig01_preopening_concept.svg` are retired as a
submission Figure 1.  The path may be replaced only by the deterministic
three-panel renderer specified here; the former hand-authored dashboard is not
patched, renamed, or promoted.  Its information rule, model equation, cluster
gate, and nonexistent result did not form one argument.

The following changes are mandatory in the redraw:

- **Remove old Figure 1d entirely.** No pending five-row forest appears in Figure
  1.  The receipt-derived forest becomes Figure 2d after the POST gate.
- Reuse the old non-result information-boundary idea in the **new** Figure 1c at
  thesis level and in Figure S2 at full detail; the old Figure 1d forest itself is
  not reused.
- Move the bounded equation to Figure 1b at thesis level and to Figure S3 in full
  detail.
- Move expanded cohort/registry geometry to Figure S1; Figure 1c retains only
  the HUC2 counts and gate checks needed for the thesis.
- Replace `15 reportable HUC2 groups` with `15 pre-attrition HUC2 groups` and,
  where relevant, `at most 15 reportable groups`.
- Replace the absolute phrase `No leakage across the issue-time boundary` with
  the narrower contract statement that target outcomes and horizon-specific
  future inputs are excluded by the frozen date-index rule.

### 6.2 Legacy Stage-06 figures

All figures generated by `scripts/06_make_figures.py` are retired from Route-A
paper evidence and must not be restored, copied, redrawn, or used as a numerical
source.  This includes:

- legacy b1/s2/p3 monitoring-site identifier/distribution and climatology plots;
- development result heatmaps and skill-versus-horizon curves;
- the cherry-picked p3/2020 warm-season trajectory;
- old coverage/width and reliability plots;
- router allocation, learned-kappa, and flow-stratified latent diagnostics;
- LOSO/warm-start plots presented as transfer evidence.

The three identifiers b1, s2, and p3 are outside Route A and have unresolved
provenance, station metadata, measurement dictionary, and redistribution rights.
No figure may turn their ordering into a reservoir cascade, river network,
upstream/downstream relationship, regulation status, or travel time.

### 6.3 Other prohibited figures

- A duplicate SI five-row forest.  The former planned FigS3 forest is removed;
  Figure S3 is reassigned to the full model/calibration mechanism.
- National performance choropleths or interpolated skill surfaces.
- River-network arrows or learned-router graphics labelled as physical routing.
- Official-Air2stream competitive plots without the official implementation and
  documented fair calibration search.
- Regulatory/ecological exceedance graphics presented as model results.
- P-value stars, green PASS badges, or traffic-light verdicts on permanently
  descriptive effects.
- Hash, rights, or data-dictionary pie/radar diagrams.  SI15 and SI16 remain
  precise tables and machine-readable inventories.

---

## 7. Manuscript placement and cross-reference contract

The current Markdown and PRE TeX contain no figure cross-reference or embedded
figure.  A complete paper projection must add references only through the
appropriate PRE/POST renderer:

| Figure | Required first citation |
|---|---|
| Figure 1 | after the narrow research question and before detailed related work/design |
| Figure 2 | POST Results question on point performance |
| Figure 3 | POST Results question on probabilistic behavior |
| Figure 4 | POST Results/Discussion question on mechanism and boundary |
| Figure S1 | Data/cohort description |
| Figure S2 | Target-period inputs and issue-time boundary |
| Figure S3 | Model-design overview before component prose |
| Figures S4--S8 | the matching POST Results, sensitivity, QC, or limitation paragraph |
| Figure S9 | the Limitations paragraph on interval calibration, cited only as development-period sensitivity |

The PRE-only `paper/agu_submission/build_agu.py` is not a POST figure renderer.
No manual `\includegraphics`, caption, or result transcription is authorized by
this document.  The future POST builder must verify that every cited figure
exists, every generated figure is cited, numbering is unique, and caption/body
claims resolve to the same value IDs.

---

## 8. Final acceptance checklist

### Evidence

- [ ] Current materialization is limited to Figure 1 and Figures S1--S3.
- [ ] Figures 2--4 and S4--S8 have no rendered submission artifact before POST.
- [ ] Every empirical mark and caption number resolves to one value ID.
- [ ] Every value ID binds value, unit, evidence role, source, derivation, and rounding.
- [ ] PRE training diagnostics contain no post-2020 outcome and no development score substitution.
- [ ] All five registered rows, all six primary models, all seven registered controls, and all eight temporal candidates appear where required.
- [ ] Missing, adverse, conflicting, failed, and non-estimable states remain visible.

### Argument

- [ ] Figure 1 alone communicates mismatch -> insight -> evidence boundary.
- [ ] Figure 2 supplies end-to-end point evidence without confirmatory wording.
- [ ] Figure 3 reports probability benefit together with width/support/cost.
- [ ] No figure or caption attributes the Stage-19 withdrawal to "quantile
      crossing"; the measured cause is a zero-width nominal interval and strict
      crossings were 0.
- [ ] Every figure declares one evidence period, and any development-period
      figure renders an in-panel scope band.
- [ ] No development-period value is compared numerically with a target-period
      value.
- [ ] Figure 4 tests the mechanism and exposes temporal/external limits.
- [ ] SI figures expand rather than duplicate main figures.
- [ ] Every claimed mechanism has a registered sensitivity or is described only as design.

### Semantics

- [ ] `causal` is used only for left-looking time order, never causal inference.
- [ ] The bound is relative to the anchor, never truth error or safety.
- [ ] External means site-ID disjoint and history-dependent, never ungauged.
- [ ] HUC2 is a coarse sensitivity group, never an independent river-network component.
- [ ] Route A remains fixed-cohort descriptive regardless of numerical direction.
- [ ] No figure implies national, regulatory, ecological, operational, or decision-value validity.

### Visual and publication QA

- [ ] Model colors, markers, units, effect direction, and horizon symbols are consistent.
- [ ] Final text is at least 7.5 pt (8 pt body target) and remains legible in
      grayscale/color-vision checks.
- [ ] No clipping, overlap, missing glyph, truncated label, or unexplained axis truncation remains.
- [ ] Fonts are embedded and vector bounding boxes are correct.
- [ ] Captions state what/configuration/takeaway/boundary and are cited in order.
- [ ] The final TeX/PDF has no missing figure, undefined reference, or PRE placeholder.

Failure of any applicable item blocks submission rendering; it does not authorize
manual repair with an unbound value.

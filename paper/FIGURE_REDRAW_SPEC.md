# ThermoRoute figure-redraw specification

**Document role:** authoritative redraw and evidence-binding specification for
four main-text figures (Figures 1--4) and ten Supporting Information (SI)
figures (Figures S1--S10, of which S9 is optional; see §5).  This file
does not authorize model fitting, target-label access, result filling, opening,
or submission.

**Benchmark restructure (2026-08-06).** The manuscript was restructured from an
architecture paper into an evaluation-benchmark paper
(`docs/PAPER_BENCHMARK_RESTRUCTURE.md`).  That restructure required four main
figures whose jobs did not match Figures 1--4 as previously specified.  The
reassignment is applied in §4 and §5 and recorded, panel by panel, in §6.4 and
in `docs/PAPER_FIGURE_SI_RECONCILIATION.md`.  Two proposals in the restructure
brief -- a development-period reference-ladder panel inside target-period
Figure 2, and a development-period conformal panel inside target-period
Figure 4 -- are **refused** here, because one figure never mixes two evidence
periods (§1).  Their content is routed to Figure 3 and Figure S9 respectively.

**Current phase:** PRE / no verified opening POST receipt.  Figures S1--S3 are
materialized as PRE artifacts under the source bindings and visual QA below.
**Figure 1's materialized bytes are stale against this revision** -- panel (b)
changed from the bounded-correction schematic to the station map -- so its
artifact must be re-rendered by the PRE renderer before submission and must not
be shipped as it stands.  Figures 2--4 and Figures S4--S10 are schema templates
only.  They must not be rendered as submission SVG/PDF/PNG files, even with
blank axes or placeholder values, before their POST gates pass.

**Canonical prose source:** `paper/ThermoRoute_paper.md`
**Current PRE TeX target:** `paper/agu_submission/ThermoRoute_WRR.tex`
**Evidence principle:** every visible empirical coordinate is a deterministic
projection of bound evidence; a familiar filename, development cache, or
completion receipt alone is never publication authority.

---

## 1. Non-negotiable figure-state matrix

The states in this table are exhaustive.

| Figure | Canonical responsibility | Evidence period | Current state | Current artifact allowance |
|---|---|---|---|---|
| Figure 1 | station map, cohort geometry against the claim gate, and the persistence challenge | PRE structural | `PRE_MATERIALIZED_REDRAW_STALE` | **PRE artifact only, and the committed bytes predate this revision: re-render required** |
| Figure 2 | how baseline choice changes reported skill | target | `POST_TEMPLATE_ONLY` | **No** |
| Figure 3 | how the spatial partition changes the transfer conclusion | **development 2019--2020** | `POST_TEMPLATE_ONLY_DEVELOPMENT` | **No** |
| Figure 4 | regional and seasonal heterogeneity, and what interval coverage costs | target | `POST_TEMPLATE_ONLY` | **No** |
| Figure S1 | cohort selection and registry geometry | PRE structural | `PRE_MATERIALIZED` | **PRE artifact only** |
| Figure S2 | chronology and issue-time/product information boundary | PRE structural | `PRE_MATERIALIZED` | **PRE artifact only** |
| Figure S3 | full model, bounded correction, and calibration mechanism | PRE structural | `PRE_MATERIALIZED_DESIGN`; final version remains suite-gated | **PRE design artifact only** |
| Figure S4 | point-performance heterogeneity | target | `POST_TEMPLATE_ONLY` | **No** |
| Figure S5 | expanded probabilistic diagnostics, event score, and reliability | target | `POST_TEMPLATE_ONLY` | **No** |
| Figure S6 | temporal opportunity, missingness, and attrition | target | `POST_TEMPLATE_ONLY` | **No** |
| Figure S7 | spatial influence and cluster sensitivity | target | `POST_TEMPLATE_ONLY` | **No** |
| Figure S8 | outcome QC, external-history arm, and failure disposition | target | `POST_TEMPLATE_ONLY` | **No** |
| Figure S9 | development-period conformal calibration sensitivity (optional) | **development 2019--2020** | `POST_TEMPLATE_ONLY_DEVELOPMENT` | **No** |
| Figure S10 | registered architecture interventions and the bounded-deviation audit | target | `POST_TEMPLATE_ONLY` | **No** |

For any state containing `TEMPLATE_ONLY`, “template” means this written panel,
caption, and binder contract only.  It does **not** mean a rendered empty figure.
A blank forest, a `PENDING` box, a dummy coordinate, or a zero substituted for a
missing value is a prohibited result rendering.  The POST skeletons refuse on the
substring `TEMPLATE_ONLY`, so a rebound figure keeps its fail-closed behaviour
while recording the rebinding in its state token.

**Stage-19 disposition (2026-08-05, final).** The Stage-19 **development**
script will not be produced for this submission, and Stage-10 cascades from it.
Figure 4(c) and Figure S5 are **not** affected: their target-period metrics are computed
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
may be compared numerically with Figure 4(c) or S5; the confirmatory target period
starts 2021-01-01.

**One figure never mixes two evidence periods.**  A figure binds either
target-period or development-period evidence, declares which, and — when
development — renders an in-panel scope band.  The POST skeletons enforce this
at two levels: `FigureSpec.evidence_period` declares the figure's period, and
`PanelSpec.evidence_period` — when a panel declares one at all — must equal it.
`validate_manifest()` raises `ManifestError` otherwise, so a mixed-period figure
cannot be expressed in the manifest, let alone rendered.

That rule is what refuses two proposals in the benchmark-restructure brief:

| Refused proposal | Why | Where the content went |
|---|---|---|
| Figure 2 panel (a), "the reference ladder", specified as *development-period with a scope band* inside an otherwise target-period figure | mixes 2019--2020 with 2021--2023 in one figure; a reader comparing two rungs would be comparing two cohorts | Figure 2(a) is rebuilt as the **target-period** ladder over the five reference models the trusted scorer emits; the development-period ladder (+0.251 against persistence versus +0.038 against damped persistence at 7 d) is Figure 3(a) row 1 |
| Figure 4 panel (d), "what calibration costs", specified as *development-period with a scope band* inside an otherwise target-period figure | same hazard, and the Stage-22 evidence already has a home | Figure S9 panels (a)--(b), which carry exactly those numbers with a mandatory scope band |

The complementary rule is that a **development-period figure is still a POST
figure**: Figure 3 and Figure S9 both require the verified opening receipt
before they render.  The receipt is not what supplies their numbers; it is the
gate that proves the submission is past the one-shot boundary and that no
development display is being published as a substitute for a target-period
result that was never produced.

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

## Figure 1 — The cohort, and what its geometry can carry

**State:** `PRE_MATERIALIZED_REDRAW_STALE`
**Evidence period:** PRE structural — the frozen registry and the 2006--2015
training partition only.  No post-2020 outcome, no development score.
**Research question:** what panel is this, where are its gauges, and what
inferential weight can its spatial geometry support?
**Narrative role:** cohort -> geometry against the claim gate -> the motivation
quantity the whole benchmark is measured against.

**Reassignment note (2026-08-06).** Panel (b) was the bounded-correction
schematic.  Under the benchmark restructure the architecture is the object under
test rather than the contribution, so a thesis-level mechanism panel no longer
belongs in the opening figure; the station map, which the restructure needs and
which previously existed only inside Figure S1, takes its place.  The
bounded-correction schematic moves to Figure S3 (§5), where the full
model/bound/calibration dataflow already lives.  **The committed Figure 1 bytes
were rendered against the previous panel set and are therefore stale.**

### Panel structure and plot types

**(a) Station map.** Plot the 120 retained sites on a CONUS base, coloured by
HUC2 region and sized by retained observed-`WTEMP` day count, with redundant
non-colour encoding so the panel survives a grayscale check.  Inset: a
zero-based histogram of nearest-neighbour distance between retained stations,
annotating the 10 km mark (19 stations) and the 289 km mean
nearest-training-gauge distance of the whole-region holdout on the same axis, so
the reader sees the two spatial scales the benchmark contrasts.  Any basemap or
HUC boundary source carries its own citation, version, licence, and source
binding; without such a source the panel renders a coordinate scatter and binds
`NO_BASEMAP`.  *Sources:* `data_usgs/station_registry_v1.csv`; the frozen
environmental audit for proximity; the Stage-13c region-transfer table for the
289 km value, bound as a **structural geometry** quantity and never as a score.

**(b) Cluster geometry against the gate.** Plot the 15 registry-derived HUC2
station counts as zero-based bars (2 … 26) and show the dimensional collapse from
657,480 site-days to 120 sites and 15 pre-attrition HUC2 groups.  Beside the
bars, render the three frozen gate checks as three separate small gauges, never
on a shared false numeric axis: 15 versus at least 30 groups,
$9.54/15=0.636$ versus at least 0.75 effective fraction, and 21.7% versus less
than 25% largest-group share.  A fourth strip shows the HUC2/HUC4/HUC6/HUC8
ladder (15 / 64 / 75 / 95 clusters at effective fractions
0.636 / 0.507 / 0.485 / 0.758) with HUC8 marked *passes the arithmetic; adjacent
units on one river are not independent*.  Continue with a short left-to-right
evidence spine:

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
of local-day alignment.  *Sources:*
`docs/OPTION_A_DESCRIPTIVE_BENCHMARK_SCOPE.md` §1 for the ladder; the frozen
registry; the claim registry and inference-gate identities.

**(c) The persistence challenge.** Plot one station-level point per retained site
for each horizon $h\in\{1,3,7\}$, where the point is a frozen training-period
descriptive statistic such as the median observed
$\lvert T_{t+h}-T_t\rvert$.  Use three aligned dot/box summaries.  The calculation
is restricted to 2006--2015, exact calendar-day pairs, finite observed WTEMP at
both ends, no target imputation, and equal station representation.  It is the
motivation quantity, not a model score and not a confirmation result.

### Caption takeaway

> The evaluated cohort is 120 availability-selected gauges whose spatial grouping
> collapses to 15 unbalanced HUC2 units with an effective count under ten, and
> whose day-to-day water temperature is strongly persistent at every lead
> reported here. Those two facts set the benchmark's terms: a reference model
> that does not already exploit persistence and seasonality concedes most of the
> error budget before fitting, and a partition of this geometry cannot carry
> region-clustered inference. Neither statement is a model result.

### Data fields and value IDs

| Panel | Required fields/values | Required source role |
|---|---|---|
| (a) | `site_no`, latitude/longitude, `huc2`, retained observed-`WTEMP` day count, nearest-neighbour distance, 10 km count, 289 km whole-region mean, basemap/source-rights status | frozen registry, environmental audit, and region-transfer fold geometry as a **structural** quantity |
| (b) | panel rows, site count, pre-attrition HUC2 count, HUC2 counts, effective cluster count/fraction, largest-group share, the HUC2/4/6/8 ladder, allowed-variable/date roles, aggregation roles, inference/QC gate identities, fixed scope status | frozen panel/registry/environmental audit plus protocol, claim registry, and information-boundary contract |
| (c) | `site_no`, `date_t`, `date_th`, `horizon_days`, `wtemp_t`, `wtemp_th`, `observed_pair`, `site_statistic`, `n_pairs` | frozen 2006--2015 panel and registry; training-only descriptive derivation |

Every station marker in panels (a) and (c) binds its site identity, horizon,
statistic, and pair count.  Every displayed summary line has a separate value ID;
it is not recomputed by the plotting layer.

### Gate

- **PRE:** permitted after source-hash validation, declared panel-(c)
  derivation, no access to post-2020 outcomes, and visual QA.  The current
  committed artifact does **not** satisfy this gate for the panel set above and
  must be re-rendered.
- **POST:** Figure 1 keeps the same cohort/geometry role; it is not refilled with
  confirmation effects.  POST may update only bound provenance/status text, not
  replace a panel with target performance.

### Prohibited semantics

“No leakage” without qualification; “15 reportable HUC2 groups”; independent
river-network components; national representation or a national coverage claim;
upstream/downstream arrows; interpolated skill surface on the map; safe
prediction; bounded error; causal thermal transport; superiority; a target-period
forest; any `PENDING` result cell.

### Acceptance

- A reader can state where the gauges are, how coarse and unbalanced their
  grouping is, and how strongly persistent the target is, from the figure and
  caption alone.
- All panel-(c) values are training-only and station-balanced.
- The panel-(b) gate quantities are not visually compared on a false common
  scale, and the HUC8 row is never presentable as a route to eligibility.
- No target-period score, interval, p-value, or verdict is present.
- The 289 km annotation in panel (a) is bound as fold geometry, never as skill.

---

## Figure 2 — Baseline choice, not architecture, sets the reported gain

**State:** `POST_TEMPLATE_ONLY`
**Evidence period:** target, 2021-01-01 through 2023-12-31.  Every rung, point,
and interval in this figure is scored by the trusted scorer inside the one-time
opening.
**Research question:** how much of a reported gain survives a strong reference,
and how does ThermoRoute's point performance compare with all six primary models
and with the five registered comparison rows?
**Narrative role:** primary end-to-end evidence, and the first of the
benchmark's three design levers.
**First citation:** close of the Results subsection on the reference model.

### Panel structure and plot types

**(a) The reference ladder.** For each lead, score the *same* ThermoRoute
predictions against every reference the trusted scorer emits — persistence,
damped persistence, seasonal climatology, global LightGBM, and the global LSTM —
and plot the five resulting skill values on one dimensionless axis, connected, so
the spread between the weakest and the strongest reference is the panel's
subject.  Label the persistence and damped-persistence rungs by name.  One panel
per lead, or one panel with lead as a marker shape.  The axis is labelled
*dimensionless; positive favours the candidate* (equation 10 of the manuscript),
and no ΔRMSE quantity appears on it.
*Artifact:* `trusted/temporal_predictions_v1.parquet` and
`trusted/availability_registry_v1.csv` on the declared all-model
exact-common-key set, reduced with the scorer's own station-RMSE recipe.

The information-matched plain causal TCN is **not** a rung here.  It is a
development-only control (Stage-09b), it is absent from the protocol's model
registry, and putting it on a target-period axis would be exactly the
development-cache substitution this figure's gate forbids.  Its comparison is a
development-period quantity and belongs to Figure S10's SI09 companion table.

**(b)--(d) All-model station distributions by horizon.** One small multiple for
each of 1, 3, and 7 days.  Plot station-level RMSE points or ECDFs for
persistence, damped persistence, climatology, LightGBM, global LSTM, and
ThermoRoute on one declared all-model exact-common-key set.  Show retained
station counts.

**(e) Registered five-row forest.** Plot the unweighted median station-level
ThermoRoute-minus-reference RMSE and whole-HUC2 bootstrap interval for all five
registered rows.  Draw the 0.00 degrees C line for damped-persistence rows and
the +0.05 degrees C ceiling for LightGBM rows.  Put status, station count, and
cluster count beside each row.  Raw p, Holm p, win rate, and bound checks belong
in main Table T2 and share its value IDs; do not encode them as significance
stars.

Panels (b)--(e) carry ΔRMSE or RMSE in degrees C and are labelled *negative
favours the candidate*; panel (a) carries a dimensionless skill score labelled
*positive favours the candidate*.  The two are never plotted on one axis and
never share a colourbar, because a positive value means opposite things in the
two conventions.

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
| ladder mark | reference model ID, horizon, skill value, its two component station-median RMSEs, common-key digest, station count, sign convention |
| all-model mark | model ID, horizon, site ID, common-key digest, paired-key count, station RMSE, reportability status |
| forest mark | test ID, candidate/reference IDs, horizon, margin, status, effect, CI low/high, station count, cluster count, gate verdict |
| T2 companion | win rate, raw p, Holm p, margin checks and their exact shared value IDs |

Source pointers resolve to trusted POST predictions, `formal_tests[*]`, the
opening receipt, and gate artifacts.  Selective omission of a model, site, horizon,
reference rung, or formal row rejects the figure.

### Gate

- **PRE:** schema text only.  Do not generate an SVG/PDF/PNG or placeholder
  forest.
- **POST:** complete opening/evidence/render receipts; exact key registry; all
  five formal rows exactly once; all six primary models; every reference rung the
  scorer emits; bound NA handling; permanently descriptive gate text.

### Prohibited semantics

Significance coloring or stars; “wins,” “beats,” “non-inferior,” “equivalent,” or
“parity”; a national claim; row-wise pooled uncertainty presented as station
uncertainty; hidden non-estimable rows; development-cache substitution; a
development-period rung on the target-period ladder; a skill value and a ΔRMSE
value on one axis; a y-axis or x-axis chosen after inspecting favourable values.

### Acceptance

- Negative effect direction and margin interpretation are explicit.
- The ladder's dimensionless axis and the forest's degrees-C axis are separately
  labelled with their opposite sign conventions.
- All-model and formal-pair denominators are separately labelled.
- Figure 2 and T2 share value IDs and rounding.
- Counts and NA rows remain visible.
- Every rung of panel (a) is target-period; none is development-period.
- Caption contains a receipt-rendered reading, mechanism-neutral explanation, and
  permanent scope boundary.

---

## Figure 3 — The spatial partition changes the transfer conclusion

**State:** `POST_TEMPLATE_ONLY_DEVELOPMENT`
**Promoted:** 2026-08-06.  This slot previously held the marginal-interval and
event-probability figure; the benchmark restructure needs a main figure for the
random-held-site versus whole-region contrast, and that content had no main-text
slot.  The interval and event-probability panels move to Figure 4(c) and to
Figure S5 (§6.4).
**Evidence period:** **development, 2019-01-01 through 2020-12-24** — not a
target-period result.  A mandatory in-panel scope band
(`fig03.scope.development_period_not_confirmation`) is rendered.
**Research question:** how much of a reported spatial transfer survives
whole-region holdout rather than a random held-site split?
**Narrative role:** the benchmark's second design lever, and the one with the
largest single effect on a reported number.
**First citation:** close of the Results subsection on whole-region holdout.

### Why this figure is development-period, and why it is still POST-gated

The one-time opening produces **no** held-region artifact.  The confirmatory
protocol registers a temporal cohort and a site-identifier-disjoint external
cohort; a leave-one-HUC2-region-out arm is not among them, and
`docs/R13_POSTOPEN_TABLE_RENDERER.md` §6 records that Table 4.6's held-region
fragment renders as `NOT_EMITTED_BY_THE_ONE_TIME_OPENING`.  The whole-region
holdout evidence in this paper is therefore the Stage-13c development-period
evidence, permanently, and the figure says so inside the panel rather than in a
caption a reader may skip.

It is nonetheless gated on the verified opening receipt, exactly like every other
POST figure.  A development-period main figure is legitimate only when the
submission is past the one-shot boundary and the reader can see what the opening
did and did not produce; publishing it earlier would let a development display
stand in for a target-period result that was never attempted.

### Panel structure and plot types

**(a) The three arms.** Plot median station skill against persistence for the
temporal, random-held-site warm-start, and held-region gauged-transfer arms at
each lead, with the against-damped-persistence values on a paired secondary
panel.  Slope lines between arms make the reduction readable.  Both axes are
dimensionless and labelled *positive favours the candidate*.  The panel is the
development-period reference ladder as well as the partition contrast: the
temporal arm's two rows are the same predictions scored against two references.
*Artifact:* the Stage-13b/13c transfer-arm report and row table.

**(b) Fold geometry.** Draw the 15 HUC2 groups packed into four folds of
[30, 30, 31, 29] stations on the same base as Figure 1(a), one colour and one
hatch per fold, with held-out stations outlined.  *Artifact:* the Stage-13c
region-transfer table and report; `data_usgs/station_registry_v1.csv`.

**(c) Distance is the association.** Plot per-station held-region skill against
the distance to that station's nearest training gauge, marking the 289 km mean,
with the random-held-site arm overplotted in a muted style at its own much
smaller distances.  The panel asserts association only.  The in-panel note and
the caption both say so; no fitted line, correlation coefficient, or causal verb
appears.

**(d) The ranking is unchanged.** Plot held-region station-median RMSE for
ThermoRoute, global LightGBM, and the global LSTM by lead, with the paired
station-level ΔRMSE and whole-HUC2 intervals beneath, in degrees C and labelled
*negative favours the candidate*.  *Artifact:* the Stage-13c region-transfer
report and the LSTM baseline report.

### Caption takeaway

> Holding out whole hydrologic regions rather than random gauges reduces reported
> skill on this panel, at a mean nearest-training-gauge distance two orders of
> magnitude larger than the spacing inside the intact cohort. These are
> development-period diagnostics on a fixed availability-selected cohort over
> 2019--2020; the one-time evaluation produces no held-region arm, so no value
> here has a target-period counterpart and none may be compared numerically with
> Figure 2, Figure 4, or any SI figure bound to the evaluation period. The
> distance panel reports association, not a transfer mechanism.

### Data fields and value IDs

| Panel | Required values |
|---|---|
| (a) | arm ID, reference model ID, horizon, median station skill, station count, cluster count, exact-key digest, sign convention |
| (b) | fold ID, HUC2 unit, station ID, coordinates, held-out flag, fold station counts |
| (c) | site ID, nearest-training-gauge distance, per-station held-region skill, arm ID, mean-distance reference, association-only status |
| (d) | model ID, horizon, held-region station-median RMSE, paired ΔRMSE, whole-HUC2 interval endpoints, win rate, station count |

Every arm-level value binds its own exact-key digest and denominator.  The
temporal, random-held-site, and held-region arms are three different key sets and
are never pooled into one mark.

### Gate

- **PRE:** schema text only.  No axes, no fold map, no placeholder arm.
- **POST:** verified opening receipt; Stage-13c region-transfer table and report;
  the transfer-arm report carrying the random-held-site fold results; the
  confirmatory protocol; the frozen station registry; a bound evidence-period
  declaration; the in-panel scope band; render receipt.

### Prohibited semantics

Target-period coverage of any statement in this figure; ungauged prediction;
river-network or hydraulic transfer; independent river-network components;
national inference; a causal reading of the distance panel; superiority,
non-inferiority, equivalence, or parity; numerical comparison against Figure 2,
Figure 4, or any target-period SI figure as if the two were one cohort; a
held-region arm described as an evaluation-period result.

### Acceptance

- All three arms, all three leads, and both reference models appear or bind an
  explicit NA.
- The in-panel scope band is legible at final placed size and names the
  development span and the 2021-01-01 target start.
- The fold map reconciles to 120 stations and to fold sizes [30, 30, 31, 29].
- Skill (dimensionless) and ΔRMSE (degrees C) never share an axis.
- The distance panel carries its association-only statement inside the panel.

## Figure 4 — Regional and seasonal heterogeneity, and what coverage costs

**State:** `POST_TEMPLATE_ONLY`
**Reassigned:** 2026-08-06.  This slot previously held the mechanism and
applicability-boundary figure.  Its architecture-intervention panel moves to
Figure S10, its attrition waterfall to Figure S6, and its external-arm panel to
Figure S8; its temporal-sensitivity panel stays here as panel (b), joined by the
regional heterogeneity of Figure S7's aggregate and the coverage--width plane of
the former Figure 3(a).  See §6.4.
**Evidence period:** target, 2021-01-01 through 2023-12-31.
**Research question:** is the remaining skill uniform across regions and
seasons, and what does a calibrated interval cost?
**Narrative role:** the benchmark's third lever — heterogeneity — plus the price
of the uncertainty statement.
**First citation:** close of the Results subsection on regional uniformity and
interval width.

### Panel structure and plot types

**(a) Regional heterogeneity.** Plot per-HUC2 median skill against persistence
and against damped persistence at each lead, ordered by region, with the pooled
median and the region-weighted mean drawn as named reference lines and station
count per region encoded by marker size.  The panel's subject is the compression
between the two reference columns, not the ranking of regions.  *Artifact:*
`trusted/spatial_sensitivity_v1.json` `comparisons[].per_huc[]`.  Note that
`per_huc[].huc2` is a cluster label of the form `HUC2:01` or
`UNMAPPED:<site_no>`, not a bare two-digit code; `UNMAPPED` units render as
themselves and are never folded into a neighbour.

**(b) Seasonal and annual heterogeneity.** Plot all eight frozen descriptive
candidates per formal row: equal weighting of the 12 year-by-season cells, three
leave-one-year values, and four leave-one-season values.  Mark the deterministic
most-adverse value; retain the formal effect as a distinct reference and never
replace it.  *Artifact:* `trusted/temporal_coverage_audit_v1.json`.

**(c) Coverage--width plane.** Plot station-balanced empirical 90% marginal
coverage against mean interval width for every eligible learned model and
horizon.  Use horizon markers and model colours; draw the 0.90 nominal reference
without implying a formal coverage test.  Point-only models bind a
`NOT_AVAILABLE` status and never receive invented heads.  Coverage is never shown
without the width that buys it.  *Artifact:*
`trusted/probabilistic_evaluation_v2.json` `coverage_90`,
`mean_interval_width_c`.

There is no development-period panel in this figure.  The Stage-22
split-CQR / block-maximum / delayed-ACI contrast that the restructure brief
proposed as panel (d) is 2019--2020 evidence and is carried by Figure S9.

### Caption takeaway

> Skill on this cohort is close to uniform across hydrologic regions once the
> reference model already exploits seasonality, and it is close to uniform across
> years and seasons under all eight predeclared coverage candidates. The interval
> panel reports achieved marginal coverage together with the width that buys it.
> These are station-balanced descriptive diagnostics on a fixed observable
> cohort: they do not establish conditional coverage, year or season stability,
> missing-at-random outcomes, or independence between hydrologic units, and a
> favourable coverage candidate never replaces a formal row.

### Data fields and value IDs

| Panel | Required values |
|---|---|
| (a) | comparison ID, HUC2 cluster label, per-HUC effect/skill, station count, pooled median, region-weighted mean, interval/status, registry binding |
| (b) | test ID, sensitivity ID/order, candidate definition, effect, support, formal-effect reference, deterministic-worst flag |
| (c) | cohort, model, horizon, `coverage_90`, `mean_interval_width_c`, station-weight audit, pre/post-reportability forecast counts, site count, minimum targets, `NOT_AVAILABLE` reason, probability-pipeline source identity |

### Gate

- **PRE:** schema text only.  No axes, no dummy coverage point, no placeholder
  candidate.
- **POST:** verified opening receipt; `trusted/spatial_sensitivity_v1.json`;
  `trusted/temporal_coverage_audit_v1.json`;
  `trusted/probabilistic_evaluation_v2.json`;
  `trusted/temporal_predictions_v1.parquet`;
  `trusted/availability_registry_v1.csv`; erratum binding; confirmatory protocol;
  SI08, SI10, SI11; render receipt.  If any panel lacks authority, Figure 4 is
  not generated, and a development panel is never substituted.

### Pre-opening guard (blocking risk, not a figure risk)

`opening.py:7093-7094` applies a **strict** `q05 < q95` to the member-averaged
nominal heads and raises `OpeningContractError` — aborting the entire one-time
opening — on violation.  This is the Stage-19 degeneracy trap one layer up.
Measured on the development panel: member averaging clears every affected
LightGBM key (5 members); the only 12 survivors are single-member
`LightGBM-perstation` keys, and that model appears in neither `PRIMARY_MODELS`
nor the confirmatory protocol.  **Re-run this check on the target-period
predictions before executing the one-time opening.**  The qualifier travels with
panel (c) and with Figure S5.

### Prohibited semantics

Conditional coverage; distribution-free target-period guarantee; CRPS; operational
forecast reliability; economic value; coverage without width; “all calendar
days”; stability across years or seasons; missing-at-random; HUC2 as an
independent river-network component; national inference; favourable sensitivity
used to replace or rescue a formal row; "quantile crossing" as the Stage-19
cause; development-period conformal numbers presented as target-period results.

### Acceptance

- All HUC2 units, all eight temporal candidates, and every eligible model ×
  horizon coverage cell appear or bind an explicit NA.
- Coverage and sharpness are read together in one panel.
- The pooled median and the region-weighted mean are separately named.
- The formal effect is visually distinct from every sensitivity candidate.
- No value in this figure is development-period.

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
and event heads plus the anchor-bound identity, **including the bounded-correction
schematic relocated from Figure 1(b) on 2026-08-06**: the damped anchor
$A_{t+h}$, the unrestricted learned displacement
$z_{t+h}=P_{t+h}-A_{t+h}+r_{\theta,t+h}$, and

\[
\widehat y_{t+h}=A_{t+h}+\delta\tanh(z_{t+h}/\delta),
\qquad \delta=1.0\ ^\circ\mathrm C,
\]

drawn as an anchor line with a shaded $A\pm\delta$ envelope, carrying the
in-panel warning “Deviation from anchor; not an error or safety bound” as part of
the panel rather than as a caption footnote; `(d)` member averaging, 2018 CQR,
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
operate on the correct artifacts/period; the non-safety statement is legible; the
relocated $A\pm\delta$ envelope carries its in-panel warning and is not presented
as an error bar.

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

## Figure S5 — Event score, reliability, and expanded probabilistic diagnostics

**State:** `POST_TEMPLATE_ONLY`
**Restored:** 2026-08-05, superseding the same-day interim rebinding onto
Stage-22.  See `docs/FIGURE_PLAN_STAGE19_INDEPENDENT_20260805.md` §3.
**Extended:** 2026-08-06.  The benchmark restructure keeps only the
coverage--width plane in the main text, as Figure 4(c).  The event-score panel
and the three per-horizon reliability panels of the former Figure 3 are
**relocated here** and are now this figure's primary content rather than an
expansion of a main-text panel that no longer exists.  Nothing is dropped.
**Question:** Do the aggregate coverage and width summaries in Figure 4(c)
conceal model, horizon, station, or bin-level failure, and how do the event
probabilities behave against a frozen seasonal reference?
**Evidence period:** target, 2021-01-01 through 2023-12-31.

**Stage-19 does not affect this figure.** The SI08 metric family is produced at
target period by the trusted scorer inside the one-time opening
(`src/thermoroute/opening.py:8932-8948`), per cohort × model × horizon, with
station-balanced reliability bins.  The original expanded design is retained in
full and no panel is dropped.  The development-period Stage-22 conformal evidence
is confined to Figure S9 and is never mixed into this figure.

**Panels/plot types:** `(a)` full model-by-horizon coverage/width dot matrix,
every cell bound or explicit NA; `(b)` **event score** — Brier skill against the
frozen seasonal reference by model and horizon, with the zero-skill line and the
bound reference identity drawn, the reference being the frozen seasonal
climatology and never confirmation-period event prevalence — shown together with
the pinball, interval, and log-score matrix under an explicit scoring-stage
legend distinguishing nominal pre-CQR heads, the deployed CQR interval, and the
post-Platt probability; `(c)` **reliability, one sub-panel per horizon**:
observed station-balanced event frequency against mean forecast probability with
the identity line, point area representing the bound bin denominator or
station-balanced effective weight, every registered bin and count present, empty
bins retained as explicit annotations, and station-balanced bin weights
reconciling to one; `(d)` calibration slope/intercept and AUROC/AUPRC/ECE
discrimination diagnostics with bound NA reasons for single-class or fit-failure
cases.  This figure expands Figure 4(c) and does not repeat its aggregate plane.

**Caption takeaway:** Probability diagnostics are reported with their scoring
stage, station-balanced weighting, support, and non-estimability state; event
probabilities are scored against a frozen seasonal reference whose fit interval
and observation count are bound; and no single coverage number is treated as a
conditional or distribution-free guarantee.

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
support reconciles to parent counts; identity, nominal, and zero-skill references
are visually distinct and named; the scoring-stage legend prevents pre-/post-
calibration conflation; no value in this figure is development-period.

---


## Figure S6 — Temporal opportunity, missingness, and attrition

**State:** `POST_TEMPLATE_ONLY`
**Extended:** 2026-08-06.  The attrition waterfall of the former Figure 4(b) is
**relocated here**; this figure already carried the same denominator-preserving
construction, so the relocation is an absorption rather than an addition, and
panel `(a)` now carries the reportable-cluster stage the main-text panel had.
**Question:** Which calendar opportunities become observable and reportable
forecast keys, and how do frozen time sensitivities change the descriptive
effect?
**Panels/plot types:** `(a)` denominator-preserving Sankey/waterfall keeping
calendar opportunities, eligible issues, observed issue WTEMP, observed target
WTEMP, exact paired keys, retained stations, **and reportable clusters** as
distinct named stages; `(b)` horizon-by-year and season opportunity heatmap;
`(c)` complete eight-candidate sensitivity dot plot; `(d)` retained-row
block/feedback sensitivity with the exact block semantics.

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
**Relationship to the main text (2026-08-06):** this figure expands Figure 4(a).
Figure 4(a) carries the per-HUC2 medians against both references and the two
aggregate reference lines; S7 carries every unit, every leave-one-HUC2 omission,
the cluster-share diagnostics, and the permanent gate box.  The two must agree in
value and rounding on the per-HUC2 effects they share.
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
**Extended:** 2026-08-06.  The external history-dependent arm of the former
Figure 4(d) is **relocated here** into panel `(b)`, which already owned the
external cohort's scope statement; the panel now also carries the arm's results.
**Question:** What target evidence survived frozen QC, what exactly is external
about the external arm, and which failures remain in the record?
**Panels/plot types:** `(a)` raw-response -> normalized-series -> exact-A retained
QC waterfall with qualifier/conflict categories; `(b)` development/external
site-ID disjointness and history-dependence diagram **plus the six primary models
by horizon on the exact external-arm key set**, with the in-panel title clause
“site-ID disjoint, history-dependent; not ungauged” and no river-network transfer
map; `(c)` failure/attrition matrix by declared reason and disposition.  If these
panels are not legible as one figure, keep their exact tables in SI12--SI14
rather than compressing them into decorative graphics.

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
**Confirmed as the sole home of this evidence:** 2026-08-06.  The benchmark
restructure proposed a “what calibration costs” panel inside target-period
Figure 4.  That panel's content — split-CQR against the block-maximum and delayed
adaptive-conformal variants, with the widths and the non-finite interval scores
they buy — is exactly panels `(a)` and `(b)` below, and it is 2019--2020
evidence.  It stays here, where the evidence-period declaration and the scope
band already exist.
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
they cannot be compared numerically with Figure 4(c) or S5, and no adaptive variant
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
rates merged into one mark; numerical comparison against Figure 4(c) or S5 as if the
two were the same cohort.

**Acceptance:** every method × slice cell appears or binds explicit NA;
non-finite widths appear with their counts; the two weighting denominators are
separately labelled; the evidence-period scope band is legible inside the figure.

---

## Figure S10 — Registered architecture interventions and the bounded-deviation audit

**State:** `POST_TEMPLATE_ONLY`
**Added:** 2026-08-06.  This figure receives the architecture-intervention panel
demoted from the former Figure 4(a) when the benchmark restructure made the
architecture the object under test rather than the contribution.  It is a
demotion in prominence only: the evidence, its period, and its gate are unchanged.
**Question:** Which registered one-factor interventions change the prediction on
the evaluation keys, and does the algebraic deviation bound hold where it is
claimed to?
**Evidence period:** target, 2021-01-01 through 2023-12-31.

The seven controls are not an afterthought of the model registry; they are part
of it.  The confirmatory protocol's `mandatory_exploratory_architecture_controls`
list is resolved into the temporal cohort's required model set alongside the six
primary models, so the trusted scorer emits a row for every control on the same
exact common keys, and Table 4.2 of the manuscript transcribes them.  This figure
is the graphical reading of those rows.

**Panels/plot types:** `(a)` horizon-by-control dot matrix over all seven
registered one-factor controls — prior-only, no dynamic prior, fixed relaxation,
no router, no mixture, no TCN, and unbounded residual — plotting the paired
station-level RMSE difference from full ThermoRoute in degrees C, labelled
*negative favours the candidate*, with seed and member completeness shown per
cell and no control selected after its value was seen; `(b)` the
bounded/unbounded algebraic-deviation audit: pointwise violation count and rate
and the maximum absolute correction against the configured limit, over every
station-by-lead cell, with the in-panel statement that the bound is relative to
the named anchor and is not a truth-error, ecological, regulatory, or
deployment-safety bound.

**Caption takeaway:** The registered controls are single-factor deletions and
interventions on a fixed observable cohort, scored on the same exact keys as the
primary models. They diagnose sensitivity; they do not establish component
necessity, capacity-matched attribution, or a causal mechanism, and the deviation
audit verifies an algebraic construction rather than a fitted outcome.

**Fields/value IDs:** control/model ID, exact intervention dictionary,
seed/member registry, exact-key digest, site and horizon effect, paired target
count, bound-violation count and rate, maximum absolute correction, configured
delta, suite and receipt lineage, reportability/NA reason.

**Gate:** verified opening receipt; all seven target control rows under the final
model/seed contract; `trusted/temporal_predictions_v1.parquet`;
`trusted/availability_registry_v1.csv`; Stage-09 and Stage-16 receipts; the
confirmatory protocol; SI09; render receipt.  PRE rendering is prohibited.  If
any control row lacks authority, the figure is not generated and a development
control is never substituted for it.

**Development-only companion.** The Stage-09b information-matched controls — the
plain causal TCN and the plain multilayer perceptron, with their parameter and
optimiser-step budgets — are development-period evidence and may **not** appear
in this figure.  They are tabulated in SI09 under their own evidence role, and
the Stage-09b receipt is a non-blocking provenance qualifier here, never a source
of a coordinate.

**Forbidden:** component necessity; causal attribution; capacity-matched claim
unless the exact comparison proves it; post hoc control selection; a
development-period control row on a target-period axis; the deviation bound
described as an error, safety, ecological, or regulatory bound; rescue of a
failed or unfavourable formal row; significance stars.

**Acceptance:** all seven interventions and all three leads appear or bind an
explicit NA with its reason; seed and member completeness is visible per cell;
the audit reports its denominator (station-by-lead cells) alongside its rate; the
non-safety statement is legible in-panel; no Stage-09b value appears.

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
- Reuse the old non-result information-boundary idea in Figure 1's
  cluster-geometry panel at thesis level and in Figure S2 at full detail; the old
  Figure 1d forest itself is not reused.
- ~~Move the bounded equation to Figure 1b at thesis level and to Figure S3 in
  full detail.~~ **Superseded 2026-08-06:** the bounded equation is no longer in
  Figure 1 at all.  It lives only in Figure S3(c).  See §6.4.
- Move expanded cohort/registry geometry to Figure S1; Figure 1's
  cluster-geometry panel retains only the HUC2 counts, the gate checks, and the
  HUC2/4/6/8 ladder needed for the thesis.
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

### 6.4 Benchmark-restructure reassignment (2026-08-06)

No panel is deleted.  Every panel of the previous four main figures either stays
in the main text with a new number or moves to a named SI figure, and the two
new main panels are built from evidence that already existed.  The full record,
including what was refused and why, is
`docs/PAPER_FIGURE_SI_RECONCILIATION.md`.

| Was | Content | Is now | Evidence period |
|---|---|---|---|
| Figure 1(a) | persistence challenge | Figure 1(c) | PRE structural |
| Figure 1(b) | bounded-correction schematic | **Figure S3(c)** | PRE structural |
| Figure 1(c) | cluster geometry against the gate | Figure 1(b), extended with the HUC2/4/6/8 ladder | PRE structural |
| — | station map and nearest-neighbour scale | **Figure 1(a)** — new; the full registry geometry stays in Figure S1 | PRE structural |
| — | reference ladder | **Figure 2(a)** — new, built at target period from the trusted scorer's own reference models | target |
| Figure 2(a)--(c) | all-model station distributions | Figure 2(b)--(d) | target |
| Figure 2(d) | registered five-row forest | Figure 2(e) | target |
| Figure 3(a) | coverage--width plane | **Figure 4(c)** | target |
| Figure 3(b) | event score | **Figure S5(b)** | target |
| Figure 3(c)--(e) | reliability by horizon | **Figure S5(c)** | target |
| — | three transfer arms, fold geometry, distance association, held-region ranking | **Figure 3(a)--(d)** — promoted to a main figure; development-period, scope-banded | development 2019--2020 |
| Figure 4(a) | seven registered architecture interventions | **Figure S10(a)--(b)** | target |
| Figure 4(b) | attrition waterfall | **Figure S6(a)** | target |
| Figure 4(c) | eight temporal-coverage candidates | **Figure 4(b)** | target |
| Figure 4(d) | external history-dependent arm | **Figure S8(b)** | target |
| — | per-HUC2 regional heterogeneity, aggregated | **Figure 4(a)**; every unit and every leave-one omission stays in Figure S7 | target |

Two consequences must not be lost:

1. **Figure 1's committed PRE bytes are stale.**  Panel (b) changed content, so
   `fig01_preopening_concept.{svg,pdf,png,json,csv}` no longer match this
   specification and must be re-rendered by the PRE renderer.  The PRE
   supporting-figure manifest also binds this file's SHA-256 as a source; both
   PRE manifests record a spec digest that this revision changes.
2. **Figure 3 is the only main figure the one-time opening cannot fill.**  The
   opening emits no held-region artifact, so Figure 3 is permanently
   development-period.  A future decision to run a target-period regional
   holdout would be a protocol amendment, not a figure change.

---

## 7. Manuscript placement and cross-reference contract

The current Markdown and PRE TeX contain no figure cross-reference or embedded
figure.  A complete paper projection must add references only through the
appropriate PRE/POST renderer.

The manuscript marks each first-citation position with an **inert placement
anchor**, not a reference:

```
<!-- FIGURE_ANCHOR id=<F1|F2|F3|F4|S1..S10> state=<PRE|POST|POST_DEVELOPMENT> role=first_citation source=paper/FIGURE_REDRAW_SPEC.md#figure-<n> -->
```

An anchor carries no number, no caption, and no image path, so it cannot
substitute for a render; the renderer resolves each one into a numbered reference
at the anchored position.  There are **fourteen** anchors, and their ids are the
fourteen figures of this specification:

| Figure | Anchor `state` | Required first citation |
|---|---|---|
| Figure 1 | `PRE` | close of the Introduction, after the four research questions |
| Figure 2 | `POST` | close of the Results subsection on the reference model |
| Figure 3 | `POST_DEVELOPMENT` | close of the Results subsection on whole-region holdout |
| Figure 4 | `POST` | close of the Results subsection on regional uniformity and interval width |
| Figure S1 | `PRE` | Data/cohort description |
| Figure S2 | `PRE` | Target-period inputs and issue-time boundary |
| Figure S3 | `PRE` | Model-design overview before component prose |
| Figure S4 | `POST` | close of the Results subsection on model ranking, which it expands |
| Figures S5, S6, S8, S10 | `POST` | the evaluation-period results section, beside the table each expands |
| Figure S7 | `POST` | close of the Results subsection on regional heterogeneity, beside Figure 4 |
| Figure S9 | `POST_DEVELOPMENT` | the Limitations paragraph on interval calibration, cited only as a development-period sensitivity |

The PRE-only `paper/agu_submission/build_agu.py` is not a POST figure renderer.
It strips every HTML comment before conversion, so the anchors never reach the
TeX and never become a hand-inserted reference.  No manual `\includegraphics`,
caption, or result transcription is authorized by this document.  The future POST
builder must verify that every cited figure exists, every generated figure is
cited, the anchor id set equals the manifest's figure-id set exactly, numbering
is unique, and caption/body claims resolve to the same value IDs.

---

## 8. Final acceptance checklist

### Evidence

- [ ] Current materialization is limited to Figures S1--S3; Figure 1's committed
      bytes predate the 2026-08-06 panel reassignment and are re-rendered before
      submission.
- [ ] Figures 2--4 and S4--S10 have no rendered submission artifact before POST.
- [ ] Every empirical mark and caption number resolves to one value ID.
- [ ] Every value ID binds value, unit, evidence role, source, derivation, and rounding.
- [ ] PRE training diagnostics contain no post-2020 outcome and no development score substitution.
- [ ] All five registered rows, all six primary models, all seven registered controls, and all eight temporal candidates appear where required.
- [ ] Missing, adverse, conflicting, failed, and non-estimable states remain visible.

### Argument

- [ ] Figure 1 alone communicates where the cohort is, how coarse its grouping
      is, and how persistent the target is.
- [ ] Figure 2 supplies end-to-end point evidence and the reference ladder
      without confirmatory wording.
- [ ] Figure 3 states inside the panel that it is development-period and that the
      opening produces no held-region arm.
- [ ] Figure 4 reports coverage together with the width that buys it, and reports
      regional and seasonal heterogeneity against both references.
- [ ] No figure or caption attributes the Stage-19 withdrawal to "quantile
      crossing"; the measured cause is a zero-width nominal interval and strict
      crossings were 0.
- [ ] Every figure declares one evidence period, no panel declares a different
      one from its figure, and any development-period figure renders an in-panel
      scope band.
- [ ] No development-period value is compared numerically with a target-period
      value, and no development row appears on a target-period axis.
- [ ] Figure S10 carries the registered interventions with their deviation audit,
      and Stage-09b values appear only in SI09.
- [ ] SI figures expand rather than duplicate main figures.
- [ ] Every claimed mechanism has a registered sensitivity or is described only as design.
- [ ] The manuscript's fourteen `FIGURE_ANCHOR` ids equal the fourteen figure ids
      of this specification, with matching `state` tokens.

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

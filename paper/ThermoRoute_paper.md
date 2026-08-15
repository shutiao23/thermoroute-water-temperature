# Benchmark design governs reported skill in daily river water-temperature prediction

**[AUTHOR LIST TO BE COMPLETED]**

^1^ [AFFILIATION 1 TO BE COMPLETED]

**Corresponding author:** [TO BE COMPLETED]

*[Author names, affiliations, ORCIDs, and corresponding-author details to be
completed before submission.]*

## Key Points

- Confirmatory: seven-day skill falls from +0.250 against persistence to +0.038
  against damped persistence, and every model collapses alike.
- Confirmatory: a hybrid reporting +0.218 against persistence is
  indistinguishable from the damped baseline; the reference set the sign.
- Confirmatory: keys a model may decline are 17-25% harder than the keys it
  keeps, across six models from persistence to deep learning.

## Abstract

Daily river water temperature is strongly persistent and seasonal, so reported
machine-learning skill depends on what a model is compared against and on which
days it is scored. On one frozen panel of 120 U.S. gauges across 15 hydrologic
regions we hold the models and data fixed and vary the evaluation design,
scoring every arm on one common registry of station/date/horizon keys with all
preprocessing fitted strictly backwards in time, on a pre-specified 2021-2023
held-out window of 116 reportable stations. Two channels are quantified. The
reference: seven-day station-median skill falls from +0.250 against persistence
to +0.038 against damped persistence, a property of the panel and not of one
model — all six compared models lie between +0.218 and +0.250 against
persistence and between −0.011 and +0.038 against damped persistence, so an
air2stream-style hybrid appearing to capture a fifth of the seven-day error
budget is indistinguishable from the baseline. The key set: one model failed on
0.75% of keys, and six witness models all score 17-25% worse on the keys it
declined than on those it kept, so a complete-case convention would have scored
it on a measurably easier subset. A gradient-boosted tree with site identity is
most accurate at one and three days. Post-outcome descriptive analyses then find
that withholding a gauge's own recent water temperature costs 1.3-1.7 °C, two
orders of magnitude more than the gap between the two estimator implementations
compared — hypotheses this cohort generated, not findings it confirmed, and each
an input ablation at gauged sites.

**Plain Language Summary.** River temperature changes slowly, so a forecast
repeating yesterday's reading is already fairly accurate, and nudging it toward
the season's usual value is better still. That makes the comparison a model is
judged against decisive. We tested predictions at 120 U.S. gauges under one
demanding evaluation: every model scored on exactly the same days and sites,
none allowed to see anything dated after its issue time. Judged
against the naive forecast all six methods looked about equally good and all
looked strong; judged against the slightly smarter one they separated, and one
established method proved no better than the simple rule it had appeared to beat
by a fifth. The days a method is allowed to skip, in studies that let it, are
also markedly harder than the days it keeps, so skipping flatters it. Separately
and more tentatively, what a model was *given* mattered more than which method
it was: denying it the recent readings from the gauge it predicts cost 1.3 to
1.7 °C, against under 0.01 °C for switching model implementations. Much of what is reported as a modelling advance is, on this evidence, a
statement about the comparison and the days scored.

**Keywords:** river water temperature; benchmark design; strong baselines;
spatial holdout; comparative evaluation; information value.

---

## 1. Introduction

Stream thermal regimes set dissolved-oxygen saturation, metabolic rates,
life-stage timing, and habitat suitability, and respond to atmospheric forcing,
discharge, groundwater exchange, shading, channel geometry, and regulation
([Caissie, 2006](https://doi.org/10.1111/j.1365-2427.2006.01597.x)). Daily
records from national monitoring networks have made the variable one of the most
heavily modeled targets in applied hydrological machine learning, and the
prevailing account is that deep sequence models have substantially advanced it:
nonlinear air–water regressions
([Mohseni et al., 1998](https://doi.org/10.1029/98WR01877)) and hybrid
air-temperature and discharge formulations
([Toffolon and Piccolroaz, 2015](https://doi.org/10.1088/1748-9326/10/11/114011))
were succeeded by recurrent, multi-task, and physics-guided river-network
architectures reporting large error reductions
([Rahmani, Lawson, et al., 2021](https://doi.org/10.1088/1748-9326/abd501);
[Sadler et al., 2022](https://doi.org/10.1029/2021WR030138);
[Barclay et al., 2023](https://doi.org/10.1029/2023WR035327)) and by graph
architectures making spatial dependence explicit
([Topp et al., 2023](https://doi.org/10.1029/2022WR033880)).

Those gains are heterogeneous and the heterogeneity does not organize itself by
architecture. Published accuracies are not comparable because reference models,
key sets, and splitting conventions differ
([Corona and Hogue, 2025](https://doi.org/10.5194/hess-29-2521-2025)), and where
families *are* compared under a common protocol the spread is small: neither a
comparison of statistical and machine-learning water-temperature models
including an air2stream formulation
([Feigl et al., 2021](https://doi.org/10.5194/hess-25-2951-2021)) nor one across
recurrent, convolutional, and transformer architectures
([Liu et al., 2025](https://doi.org/10.5194/hess-29-6811-2025)) finds an ordering
that would explain the published dispersion. That a score is a statement about
its benchmark is itself uncontested, from the differential split-sample test
([Klemeš, 1986](https://doi.org/10.1080/02626668609491024)) through explicit
upper and lower benchmarks
([Seibert et al., 2018](https://doi.org/10.1002/hyp.11476)) to benchmarking as
the centre of what machine learning can tell us about a hydrological system
([Nearing et al., 2021](https://doi.org/10.1029/2020WR028091)). What is missing
for daily river temperature is the measurement.

**This paper measures two channels of it on identical keys.** The first is the
reference: daily mean water temperature has enough thermal inertia that relaxing
the last observation toward a seasonal climatology already removes most of the
error a learned model can remove, so the denominator of a skill score is a
modelling choice with the size of a result. The second is the key set: models
are routinely scored on model-specific complete-case sets, so a model can gain
apparent accuracy by declining to predict on hard days. A third channel,
preprocessing fitted on windows that include the evaluated interval
([Arsenault et al., 2018](https://doi.org/10.1016/j.jhydrol.2018.09.027)), is
controlled here by construction but not quantified, and a fourth — random
held-site splits, where a held gauge's neighbours remain in training
([Kratzert et al., 2019](https://doi.org/10.1029/2019WR026065)) — turns out to
be below this design's resolution (Section 4.4). None of the four is visible in
a reported RMSE.

Two boundaries follow, and recent work sits beyond both. Every model here
consumes issue-time information only and is scored on retrospective gridded
meteorology, not on what a forecaster could have run
([Zwart et al., 2023](https://doi.org/10.1111/1752-1688.13093);
[Padrón et al., 2025](https://doi.org/10.5194/hess-29-1685-2025)); and every
confirmatory arm retains the target gauge's own observed history, so nothing here
speaks to reaches that were never instrumented, now addressed at CONUS scale by
frameworks built for that case
([Siddik et al., 2026](https://doi.org/10.1016/j.jhydrol.2025.134780);
[Philippus et al., 2026](https://doi.org/10.1016/j.jhydrol.2026.135620)) and in
far sparser networks
([Chang et al., 2025](https://doi.org/10.1029/2024WR039053)).

The panel is 657,480 site-days from 120 stable USGS site numbers over 2006–2020,
in 34 states and 15 two-digit hydrologic unit (HUC2) regions; every model is
tuned on data through 2020 and scored on one common set of station/date/horizon
keys at 1-, 3- and 7-day leads, with a held-out 2021–2023 window. The cohort is
availability-selected rather than drawn, bounding what it can generalize to
(Sections 3.4, 6). ThermoRoute, whose skill is decomposed, is described in
Section 3.1; its architecture is the object under test, not the contribution.

---

## 2. Data and design

### 2.1 The frozen panel and the cohort it defines

The development record comprises a single derived panel and a fixed station
registry (SI01). Each of 120 sites carries one row per calendar day from
2006-01-01 through 2020-12-31, giving 657,480 rows, missing observations being
explicit nulls rather than absent rows. The models consume seven raw variables —
water temperature, streamflow, air temperature, precipitation, a
relative-humidity proxy, daylight-period mean shortwave radiation and wind
speed — from USGS NWIS daily values
([U.S. Geological Survey, 1994](https://doi.org/10.5066/F7P55KJN)), Daymet V4
([Thornton et al., 2022](https://doi.org/10.3334/ORNLDAAC/2129)) and gridMET
([Abatzoglou, 2013](https://doi.org/10.1002/joc.3413)).

The cohort was not sampled at random and nothing treats it as though it were:
selection rejected 1,345 of 1,465 candidates, mostly for lacking joint
water-temperature and discharge availability, and 2019–2020 participated in its
construction. Water temperature is absent on 15.8% of rows and discharge on 2.8%;
38 retained stations share a hydrologic unit code with another and 19 lie within
10 km of another, so station-level independence is not tenable.

### 2.2 Temporal roles and the common key set

Temporal roles are fixed before any model is fitted, and a training sample is
admitted only if its issue date and all target dates fall in the same partition:
training 2006–2015 (438,240 rows, 341,646 with observed water temperature),
validation 2016–2017, calibration 2018, development evaluation 2019–2020, and
the held-out test 2021–2023 (counts in SI01). Models are tuned exclusively on
data through 2020. The held-out window was opened once for the pre-specified
Section 4.3 evaluation and then *reused* for the explicitly post-outcome
analyses of Sections 4.4–4.6; it never informed tuning.

**The key registry is a data rule, not a model outcome.** A key is admissible
when its issue and target dates fall inside the window, issue-date water
temperature is finite and observed, a 32-day history can be built, and
target-date water temperature is finite and observed under the inclusive
qualifier policy, evaluated independently at each lead. No condition mentions a
model, and every declared model must produce a row on every admissible key:
failing to predict is a failure, not an exclusion. Section 4.1 measures what
that rule is worth. The development registry holds 249,072 keys, 83,024 per
lead, and never carries a held-out claim.

![Study sites and evaluation design.](figures/fig01_study_design.pdf)

**Figure 1. Study sites and evaluation design.** (a) The 120 U.S. gauges coloured
by the four deterministic whole-HUC2-region folds. (b) Temporal partitions.
(c) The three evaluation tasks: known-site forecasting, random held-site
transfer, and whole-region gauged transfer. (d) The issue-time information
boundary — every model scored on the identical common forecast-key registry, no
model-specific complete-case set permitted.

### 2.3 Evaluation-period inputs and the information boundary

The test interval is 2021-01-01 through 2023-12-31 for the same cohort, with
meteorology taken at each station coordinate rather than aggregated over
upstream catchments — a real limitation for large basins (Section 6). The
primary information set uses provider values dated no later than each issue date
and consumes no horizon-specific forecast: a date-indexed retrospective
hindcast, not a re-execution of what a forecaster could have run, because the
as-issued provisional vintage of a gridded product cannot be reconstructed
afterwards.

Water temperature (NWIS 00010, °C), discharge (00060) and gage height (00065)
come from NWIS as the provider's 00003 daily mean on site-local dates.
**Discharge is served in ft³ s⁻¹ and consumed in those units**: it enters the
models only after per-station standardization, so the unit affects no reported
quantity, but the panel column is not m³ s⁻¹. Values are retained irrespective
of approval qualifier; 2,059 of 657,480 rows carry negative discharge, a real
measurement at tidal and regulated reaches, and observed water temperature spans
−0.90 to 34.30 °C. The qualifier ledger and the provider-specific day
definitions are in SI12. Test-window daily means are used only as outcomes and
do not enter model fitting, feature construction, threshold estimation,
calibration, or station selection.

---

## 3. Methods

### 3.1 The predictor under test

ThermoRoute predicts at issue time *t*, station *i* and lead *h* by adding a
bounded learned residual to a damped-persistence anchor,

> **(1)**  $A_{i,t+h} = c_{i,t+h} + \phi_i^h\,(y_{i,t} - c_{i,t})$

where $c$ is a seasonal climatology and $\phi_i$ a per-station decay fitted by
no-intercept least squares on training-period consecutive-day anomaly pairs. The
same object, fitted identically, is the damped-persistence reference. The
residual comes from a strictly left-looking temporal convolutional encoder over
a 14-lag window, with a horizon-conditioned router, a three-expert mixture and a
$\pm1\,{}^\circ$C algebraic bound on the deviation from the anchor (Figure 2);
one model serves all three leads, 38,505 trainable parameters, five seeds
averaged (Text S3). The architecture is the object under test rather than the
contribution: Section 4.2 shows an information-matched plain causal network
reproduces it to within 0.023 °C.

![One shared run, one key registry, and a single fork at the regression
target.](figures/fig02_model_concept.pdf)

**Figure 2. Everything the compared models share, and the one thing they do not.**
A single run supplies every model the same information set — nothing dated after
the issue time (regime F0) — and the same damped-persistence anchor $A$ (inset),
which three of the four predict a residual around while LightGBM predicts the
raw target $y$ directly with site identity. That target is the only fork. All
models are then scored on one registry of station × date × lead keys, on which
declining to predict counts as a failure rather than an exclusion, and reduced
by the same station-first path: RMSE within station, paired contrast, median
across stations. The $\pm1\,{}^\circ$C bound constrains the residual about the
anchor, not the forecast error. The figure is structural and carries no evidence
from either period; tuning is not equalised and the optional F3 forcing axis is
not drawn.

### 3.2 The reference set

The reference set is what a reported gain is a gain over, and each member removes
one alternative explanation for apparent skill. **Persistence**
($\hat y_{t+h}=y_t$) quantifies raw series memory; **damped persistence**, equation
(1) fitted on the training period, is the primary reference; **climatology**
isolates the seasonal cycle. **LightGBM**
([Ke et al., 2017](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html))
is the principal learned reference, run globally with site identity and per
station; a **global LSTM** with a station embedding represents the deep sequence
family ([Rahmani, Lawson, et al., 2021](https://doi.org/10.1088/1748-9326/abd501);
[Zwart et al., 2023](https://doi.org/10.3389/frwa.2023.1184992)); a **plain
causal temporal convolutional network** is the information-matched deep
reference, with the same outcome-free inputs, a parameter count matched to
within 2% (38,346 against 38,505) and an unrestricted residual around the same
anchor; and an **air2stream-style hybrid** (Toffolon and Piccolroaz, 2015) is
fitted per station on the same keys (SI07). Tuning budgets are documented but
*not* equalized across classes, so any statement about LightGBM is scoped to
this four-candidate procedure and feature schema rather than to gradient
boosting in general.

### 3.3 Leakage control and spatial partitions

Leakage control was enforced by construction: every predictor carries a date no
later than the issue date, the encoder is strictly left-looking, and
standardization constants, q90 event thresholds (2006–2015), 2018 conformal
offsets, and Platt calibrators are each estimated strictly before the interval
to which they apply. Predictions were regenerated independently from each
fitted model for verification. The same mask is applied to the primary
predictors and to the issue-date auxiliary quantities used by the relaxation
proposal and regime gate; fill-value invariance is therefore established for
the predictors but not separately for those auxiliary quantities (SI02).

Random held-site splits can be optimistic, because a held gauge's neighbours
remain in training and the model reaches its thermal regime through correlated
forcing and shared preprocessing. We therefore compare four balanced random
folds against four leave-HUC2-region folds of [30, 30, 31, 29] stations, at a
mean 289 km from the nearest training gauge (SI11). Held sites
still supply their own water temperature through the issue date, so this is
**gauged** transfer to a region whose gauges were not fitted, not prediction at
a site with no record. Split-conformalized intervals are calibrated on 2018 only
and reported as a marginal coverage diagnostic (SI08).

### 3.4 Estimand, metrics, and comparison set

**Two differently signed quantities are reported and never combined:** the
paired effect **(2)** $\Delta\mathrm{RMSE} = \mathrm{RMSE}(\mathrm{cand}) -
\mathrm{RMSE}(\mathrm{ref})$ in ${}^\circ$C, where **negative** favors the
candidate, and the skill score **(3)** $\mathrm{skill} = 1 -
\mathrm{RMSE}(\mathrm{cand})/\mathrm{RMSE}(\mathrm{ref})$, dimensionless,
where **positive** does. Every value in Section 4 is labeled with which it is.
The sampling unit is the station: unweighted RMSE is computed on the common
daily keys per reportable station and the primary effect is the unweighted
median across stations, a station/lead cell being reportable only with at least
100 valid paired targets.

The comparison set is five rows and they are not the same kind of test: three
compare ThermoRoute with damped persistence at 1, 3 and 7 days as *superiority*
tests at a 0.00 °C margin, two compare it with LightGBM at 3 and 7 days as
*non-inferiority* tests at a +0.05 °C margin. That margin is a numerical ceiling
on allowable degradation fixed before the outcomes were opened; it is not derived
from sensor precision, biological response or regulation, and nothing here is
written as an equivalence or parity statement. Two uncertainty procedures
accompany each row, both clustered at HUC2: a one-sided p from exact enumeration
of all 2^K whole-cluster sign vectors against that row's own margin, and a
10,000-draw cluster bootstrap percentile interval, Holm-adjusted
([Holm, 1979](https://www.jstor.org/stable/4615733)) over the five. A row is
supported only when both agree. The cohort came from an availability filter
rather than probability sampling of regions, so exchangeability is not
established and the clustered intervals are approximate sensitivities, not
decision evidence.

---

## 4. Results

**Three analysis categories are used and they are not interchangeable.** Sections
4.1 and 4.3 are *confirmatory*: suite, family, estimand, margins and decision
rules were fixed on data through 2020 and the window opened only to score them.
Section 4.2 is a *development diagnostic* on the 2019–2020 partition, which
participated in cohort construction. Sections 4.4–4.6 are *post-outcome
descriptive*, specified after the window had been opened; they carry that label
wherever they are quoted. SI20 records each analysis's category and
specification date. Every value below is an unweighted station median over
reportable stations on the relevant common key registry, in °C.

### 4.1 Two channels of benchmark design, measured on identical keys

**The reference collapse is a property of the panel, not of one model.** On
identical keys the same fixed model reports a seven-day median station skill of
+0.251 against persistence and +0.038 against damped persistence, a factor of
6.6 between two numbers differing only in the denominator of equation (3). That
would be a fact about ThermoRoute if it stopped there. It does not: every model
in the suite behaves the same way (Table 1). All six lie between +0.218 and
+0.250 against persistence — a band narrow enough that a reader comparing two
papers quoting persistence-relative skill is comparing numbers that never
separated the models — and spread from −0.011 to +0.038 against damped
persistence, half again as wide.

**One model changes sign.** The air2stream-style hybrid reports +0.218
[0.167, 0.233] against persistence and −0.011 [−0.031, +0.007] against damped
persistence. The same fitted object, on the same days, either captures a fifth
of the seven-day error budget or is indistinguishable from the baseline,
according only to which reference sits in the denominator. Rebuilding the anchor
under seven predeclared one-factor variants moves ThermoRoute's seven-day skill
from +0.025 to +0.063 (SI20); the better
the anchor, the less any learned model adds.

**The key set is the second channel, and this cohort measures it.** Fourteen of
fifteen scored models predict on all 358,807 held-out keys; the hybrid produces
356,131, failing on 2,676 (0.75%) concentrated in nine of 118 sites. Under the
common-key rule that is a failure and everyone is scored on the intersection.
Under the complete-case convention much of the literature uses it would be an
exclusion, and the hybrid would be scored on a subset selected by its own
ability to handle it. Whether that subset is easier needs no counterfactual
model, because fourteen models did predict those keys: **every one of six
witnesses spanning persistence to deep learning scores 17–25% worse on the
declined keys than on the rest** (median ratio 1.21; ThermoRoute 1.560 against
1.300 °C, damped persistence 1.720 against 1.396 °C). The days a model drops, when it is
allowed to drop days, are materially harder than the days it keeps. That is the
size of the advantage the common-key registry removes by construction.

**Table 1 — every model, its accuracy, and what it reports against each
reference.** Station-median RMSE in °C at each lead and median station skill
(equation 3) at seven days, all on the identical held-out registry, 116
reportable stations, with whole-HUC2 cluster-bootstrap intervals. Reading down a
column compares models; reading across a row shows what the reference alone does
to one fitted object. The three reference models carry no skill against
themselves. MAE and bias are the same rows in SI07.

<!-- TABLE all-model scores (generated) -->

| Model | RMSE 1 d | RMSE 3 d | RMSE 7 d | Skill 7 d vs. persistence | Skill 7 d vs. damped |
| --- | ---: | ---: | ---: | --- | --- |
| Persistence | 0.813 | 1.638 | 2.202 | — | — |
| Damped persistence | 0.789 | 1.454 | 1.773 | — | — |
| Climatology | 1.899 | 1.902 | 1.903 | — | — |
| LightGBM | 0.589 | 1.304 | 1.735 | +0.248 [+0.230, +0.263] | +0.030 [+0.027, +0.037] |
| LSTM | 0.663 | 1.358 | 1.712 | +0.244 [+0.225, +0.256] | +0.028 [+0.027, +0.034] |
| Plain MLP | 0.690 | 1.374 | 1.720 | +0.243 [+0.220, +0.255] | +0.023 [+0.020, +0.032] |
| Plain causal TCN | 0.646 | 1.330 | 1.710 | +0.246 [+0.223, +0.258] | +0.028 [+0.024, +0.034] |
| air2stream | 0.719 | 1.459 | 1.825 | +0.218 [+0.167, +0.233] | -0.011 [-0.031, +0.007] |
| ThermoRoute | 0.640 | 1.337 | 1.694 | +0.250 [+0.235, +0.268] | +0.038 [+0.033, +0.046] |

![Decomposition of reported skill on the held-out window.](figures/fig03_skill_decomposition.pdf)

**Figure 3. Decomposition of reported skill on the held-out 2021–2023 window
(116 reportable stations).** (a) Station-median RMSE by lead, every compared
model. (b) The seven-day error budget: 0.49 °C of the reduction comes from
damping the anchor and 0.07 °C from the learned residual. (c) Paired station
effects for the five prespecified comparisons, median and interquartile range; negative
values favor ThermoRoute. Panel (c) shows the dispersion of the paired effects,
not the narrower cluster-bootstrap intervals of Table 2.

### 4.2 Development-window ranking and architectural reproducibility

On the development window the tree has the lowest station-median RMSE at all
three leads (0.578, 1.280 and 1.649 °C against ThermoRoute's 0.631, 1.291 and
1.657 °C), with paired differences of +0.046, +0.008 and −0.005 °C: on this
panel the ranking does not favor architectural constraint, which is the result
this study is built to detect. The architecture's distinguishing property is
instead the ±1 °C algebraic bound, which held on 100.00% of audited rows as a
guarantee of the construction rather than a fitted outcome. The deep predictor
is also reproducible from a much simpler object — an information-matched plain
causal network tracks it to within 0.023 °C at every lead, and of the one-factor
deletions only removing the temporal encoder moves the 1-day RMSE materially
(SI09).

### 4.3 Held-out 2021–2023 evaluation

The held-out evaluation applies the Section 3 suite and comparison set, fixed on
data through 2020, to 2021–2023 on the common registry (*n* = 116; per-model
metrics in SI07). Station-median skill is +0.250 against persistence and +0.038
against damped persistence, the tree with site identity leads at one and three
days (0.589 and 1.304 °C), and at seven days ThermoRoute reaches 1.694 °C
against the tree's 1.735 °C. Against the information-matched plain convolutional
network the paired station median is −0.015 °C at seven days in ThermoRoute's
favour, a contrast outside the prespecified comparison set and therefore
interpreted descriptively.

**Each comparison is evaluated against its prespecified margin** (Section 3.4).
All five meet their prespecified decision rule. Two readings are worth stating: row 4
meets a non-inferiority margin at a lead where the tree is the better model
(+0.015 °C, interval excluding zero from above), and at seven days the median of
−0.009 °C has an interval covering zero, so that row establishes the ceiling
rather than a win. The comparison set, margins, and decision rules are
documented in SI05–SI06.

**Table 2 — the five prespecified comparisons, each evaluated at its
prespecified margin.** ThermoRoute is
the candidate and 116 stations are reportable throughout. Station-level ΔRMSE in
°C, negative favors ThermoRoute; whole-HUC2 cluster-bootstrap intervals; sign-flip
p computed against the row's own margin, Holm-adjusted (unadjusted p ≤ 6.1 × 10⁻⁵
throughout, the floor for 15 clusters). Rows 4 and 5 were prespecified as
non-inferiority tests, so a margin of zero would misreport them. "Meets margin"
is the literal reading and not a claim that the candidate won: row 4 meets its
margin at a lead where the tree is the better model.

<!-- TABLE 4.6 (generated) -->

| # | Reference | Lead | Margin (°C) | ΔRMSE (°C) | 95% CI | Win rate | Holm p | Decision |
| ---: | --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| 1 | Damped persistence | 1 d | +0.00 | -0.129 | [-0.199, -0.076] | 0.90 | 1.5 × 10⁻⁴ | Meets margin |
| 2 | Damped persistence | 3 d | +0.00 | -0.108 | [-0.143, -0.074] | 0.91 | 1.5 × 10⁻⁴ | Meets margin |
| 3 | Damped persistence | 7 d | +0.00 | -0.069 | [-0.084, -0.057] | 0.95 | 1.8 × 10⁻⁴ | Meets margin |
| 4 | LightGBM | 3 d | +0.05 | 0.015 | [0.010, 0.025] | 0.24 | 1.8 × 10⁻⁴ | Meets margin |
| 5 | LightGBM | 7 d | +0.05 | -0.009 | [-0.017, 0.003] | 0.59 | 1.8 × 10⁻⁴ | Meets margin |

<!-- FIGURE_ANCHOR id=S10 state=POST role=first_citation source=paper/FIGURE_REDRAW_SPEC.md#figure-s10 -->

### 4.4 Spatial transfer on 2021–2023

The development-period whole-region finding (SI19) is re-tested on the
independent window under four leave-HUC2-region folds and four balanced random
folds over five split seeds (SI11, Figure S11), moving the median
nearest-training-gauge distance from 60 to 263 km. The region-minus-random
paired penalty for the raw-target tree is +0.006, +0.009 and +0.007 °C.

That penalty is the size of its own resampling noise — the median per-site
standard deviation across five seeds is 0.004–0.007 °C — and four folds give
four independent regional observations, so the geometry contrast sits **below
this design's resolution** and no geometry effect is claimed in either
direction. The numbers are the local-adaptation condition; the pooled-adaptation
arm is reported separately in SI11, where a preprocessing defect confines it to
a sensitivity.

### 4.5 Hydrologic states where the remaining gain concentrates

The remaining gain concentrates in specific states. Station thermal memory — the
anomaly half-life of the train-fitted anchor, not an e-folding time — has a
median of 6.9 days (IQR 5.0–11.9). At seven days the median memory gain is
0.49 °C against a learned gain of 0.07 °C (Figure 3b), the median per-station
fraction of the reduction delivered by seasonal memory is 0.875, and the
correlation between log half-life and learned gain is −0.40 at one day:
longer-memory stations leave less to add. With thresholds fitted on 2006–2015
only, the seven-day median paired ΔRMSE over damped persistence is −0.069 °C on
all keys and largest under low thermal anomaly (−0.229 °C) and rapid recent
cooling (−0.170 °C), reaching −0.088 °C under rapid recent warming, with no
issue-time state reversing the ordering. The sixteen-stratum table, the
outcome-conditioned strata that a manager cannot identify at issue time, and the
reportability and multiplicity caveats that keep them descriptive are in SI11.

### 4.6 What the information is worth and what the estimator adds

Sections 4.4 and 4.5 leave one question open: the residual learned gain is
small, but is that a limit of the models or of what they were given? Four
crossed axes answer it, and **every result here is post-outcome and
descriptive**. **L** withholds the target site's own observations, **F**
substitutes realized meteorology over t+1 … t+h as an oracle bound, **G** is the
holdout geometry and **A** swaps the tree for the information-matched network.
Prohibited inputs are dropped rather than filled, all 24 mask-invariance proofs
return exactly zero, and a negative control re-admitting one water-temperature
lag is caught with a 147 °C shift. Levels, estimands, analysis status, and
per-cell results are in SI20.

**Local thermal state dominates.** Withholding it costs 1.3–1.7 °C at every lead
and at all 116 stations — two to thirteen times realized future meteorology and
roughly seventy times the architecture effect of Section 4.2 — and almost all of
that is the recent sequence rather than the long record (0.01–0.16 °C to pool a
station's climatology and damped rate, 1.17–1.71 °C to remove its recent
readings). Neither ablation tests prediction at a site that was never
instrumented: a station with no thermal record has no target to score against.

**Realized future meteorology is worth 0.13 °C at one day and 0.54–0.63 °C at
three and seven**, three to four times each cell's minimum detectable effect,
and is event-scale weather information rather than seasonal phase: a placebo
whose rules were fixed before its outcome existed retains 0.0–8.8% of the value,
and it is almost entirely air temperature. 

**Table 3 — value of realized future meteorology.** Station-first forcing value
$V_F = \mathrm{median}_i[\mathrm{RMSE}_i(F0) - \mathrm{RMSE}_i(F3)]$ in °C; CI
is a 10,000-draw whole-HUC2 cluster bootstrap and MDE the smallest effect this
cohort's cluster structure resolves at α = 0.05.

| Model | Lead | RMSE F0 | RMSE F3 | $V_F$ | 95% CI | MDE |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| LightGBM | 1 d | 0.614 | 0.451 | 0.130 | [0.081, 0.189] | 0.031 |
| LightGBM | 3 d | 1.330 | 0.765 | 0.542 | [0.415, 0.721] | 0.152 |
| LightGBM | 7 d | 1.742 | 1.063 | 0.627 | [0.423, 0.803] | 0.172 |
| ResidualLightGBM | 1 d | 0.606 | 0.446 | 0.125 | [0.084, 0.183] | 0.028 |
| ResidualLightGBM | 3 d | 1.327 | 0.752 | 0.578 | [0.406, 0.704] | 0.160 |
| ResidualLightGBM | 7 d | 1.722 | 1.087 | 0.605 | [0.445, 0.795] | 0.151 |

Crossed with L its marginal value is *lower* once local observations are
withheld, but only at short leads (Table 4) — a statistical interaction within
the evaluated design, not a general property of the two information sources.

**Table 4 — value of realized future meteorology by information level.**
Station-first paired values in °C under whole-region holdout, 116 reportable
stations, 10,000-draw whole-HUC2 cluster bootstrap.

| Forcing value | 1 d | 3 d | 7 d |
| --- | ---: | ---: | ---: |
| At L0, gauged | 0.116 [0.059, 0.174] | 0.474 [0.299, 0.693] | 0.580 [0.366, 0.779] |
| At L2, obs. withheld | 0.038 [0.020, 0.072] | 0.331 [0.249, 0.496] | 0.596 [0.431, 0.804] |
| **Interaction, L2 − L0** | **−0.076 [−0.098, −0.056]** | **−0.147 [−0.193, −0.079]** | **+0.016 [−0.035, +0.099]** |

**The estimator matters only where the information does not.** At L0 the tree
and the network differ by at most 0.009 °C at any lead against station-median
errors of 0.53–1.74 °C, under both geometries — two fitted objects, not two
model families, since their tuning budgets were documented rather than equalized
(Section 3.2) and no graph, recurrent or differentiable-hybrid model is
represented. Remove the local gauge and the tree wins by 0.02–0.19 °C; given the oracle the
sign reverses, to −0.096 [−0.177, −0.038] at seven days, so which estimator wins
is itself conditional on the information regime. The two triple differences fall
below their cells' minimum detectable effect and are reported as unresolved.

![Effect sizes across the study's axes.](figures/fig05_information_axes.pdf)

**Figure 4. What each axis of the study is worth, at seven days.** Station-first
paired effects in °C on a log axis, damped-residual tree throughout; open markers
are negative effects plotted at their magnitude, bars that reach the 0.001 floor
are intervals spanning zero, and the tick marks the minimum detectable effect
where one is defined. Within *Future weather (oracle)* and *Estimator* the row
label gives only what distinguishes it — "gauged" is L0 and "observations
withheld" is L2 — and the bracket carries the rest, so *Estimator* rows are the
plain causal network minus the tree. Local observation, future weather, study
design and the estimator span three orders of magnitude on one continuous axis;
the axes are separate conditional designs and are compared, never summed. Every
row is post-outcome and descriptive except *reference construction*, whose bar is
also the one interval that is not a cluster bootstrap: it is the range across the
seven predeclared anchor variants of Section 4.1, a spread over specifications.
It is on this axis because every other row is conditional on one anchor while the
anchor is itself a fitted object, and it says how much of the figure is a
property of the reference rather than of the river.


## 5. Discussion

### 5.1 Why these numbers are smaller, and what transfers

The difference is design, not model quality, and Section 4.1 puts numbers on two
of the channels. A study quoting skill against persistence or climatology reports
something comparable to our +0.203 to +0.251 column rather than our +0.038 to
+0.168 column — up to a factor of 6.6 for the same fitted weights, and enough to
move a hybrid across zero — so we suggest damped persistence as the minimum
reference for this variable, alongside the naive one rather than instead of it. A
study scoring each model on its own complete-case set gives the model with the
most gaps the easiest days: here the declined keys ran 17–25% harder for every
model that attempted them.

The evaluation design is nonetheless stronger than its inferential reach.
Clustered inference rests on resampling whole spatial units and needs enough of
them, of comparable size, for the asymptotics to hold; fifteen HUC2 groups with
an inverse-Herfindahl effective count of 9.54 and a largest group holding 21.7%
of stations do not meet that bar, and the cohort was never exchangeable across
regions. Usual practice would report a station-level interval, and nothing in
such a paper would tell a reader the effective cluster count is under ten — the
difference is disclosure rather than the data or the model.

**Why our estimator gap is smaller than published comparisons, and why an
operational study would rank differently.** Two recent results are the right
external checks on ours and neither contradicts it. Feigl et al. (2021) compare
six model classes on Austrian catchments and report a median RMSE spread between
them of order 0.08 °C, where our L0 contrast between two implementations is
0.009 °C. The two numbers measure different things: theirs spans a wider set of
model families each tuned in its own way, ours holds the information set, the
regression target and the key registry fixed and varies only the estimator, so
ours is the narrower quantity and should be smaller. Read together they say the
same thing — the estimator is not where the difficulty lies — with ours putting a
floor under how small the gap becomes once the controls are matched.

Padrón et al. (2025) forecast stream temperature at extended range with archived
weather and find model choice mattering more than we do, and the reason is
visible in our own axes. Their setting supplies neither of the two conditions
under which our estimator contrast vanishes: their forcing is a real forecast
rather than issue-time-only information, and their harder cases withhold the
target site's history. Both of those are cells where our own estimator penalty
becomes non-zero and, given the oracle, changes sign. An operational study
should therefore expect a different ranking from ours, and that is a prediction
this design makes rather than a disagreement.

**What this implies for reporting.** Three things would make published river
temperature results comparable at negligible cost: quote skill against a damped
anchor alongside the naive one, since the pair costs one extra column and is
what separates the models; state how many keys each model actually scored, since
a model that declines the hard days is credited for an easier subset; and
declare whether preprocessing statistics were fitted before or across the
evaluated interval. None requires new computation, and the first two are what
this study measures.

The study is narrow by construction: a common-key clustered benchmark of point
predictors at gauged sites, not an attempt to replace process-based thermal
models, river-network graph models or differentiable hybrid formulations
([Jia et al., 2021](https://doi.org/10.1137/1.9781611976700.69);
[Rahmani et al., 2023](https://doi.org/10.1029/2023WR034420);
[Zwart et al., 2023](https://doi.org/10.3389/frwa.2023.1184992)). What is
portable is not the architecture but three controls: one key registry across
models, statistics fitted strictly backwards in time, and partitions by whole
hydrologic region.

---

## 6. Limitations

**Scope.** The estimand is a fixed availability-selected cohort of gauged sites,
and the clustered intervals are approximate sensitivities on it (Section 5.2).
Section 4.6 withholds a station's own record from the model's inputs, which is a
different question from prediction at a location that was never instrumented:
the cohort, the climatology and the evaluation keys all come from gauged sites,
and this study does not address the ungauged case. Throughout, "causal"
describes time ordering only.

**Cohort and measurement.** Requiring joint water-temperature and discharge
records buys a discharge channel worth +0.042 °C at 1 day (SI19) at the cost of
1,345 of 1,465 candidates, which enriches the panel for long, complete records
and bounds what it generalizes to. Meteorology is taken at station coordinates
rather than integrated over upstream catchments. The air2stream-style reference
is an empirical variant of the published formulation (Toffolon and Piccolroaz,
2015) rather than the official code, and is used as a reference rather than as
evidence about the process model.

**Reference and thresholds.** The damped anchor is a fitted object and the
seven-day headline moves by a factor of 2.5 across defensible constructions of
it (Section 4.1), so every number against damped persistence is conditional on
the anchor of Section 3.1, primary because it was fixed before the window
opened. The ±1 °C bound, the event quantiles and the 100-key rule are
development-selected, and q90 is a statistical tail diagnostic (SI19).

## 7. Conclusions

We evaluated daily river water-temperature prediction at 120 stable U.S. gauges
across 15 hydrologic regions on one common set of station/date/horizon keys, all
statistics fitted strictly backwards in time, on a held-out 2021–2023 window.

Two channels of benchmark design were quantified, and both are properties of the
panel rather than of any one model. The reference governs the reported gain:
seven-day station-median skill falls from +0.250 against persistence to +0.038
against damped persistence, every model in the suite collapses the same way, and
an air2stream-style hybrid that reports +0.218 against persistence is
indistinguishable from the damped baseline. The key set governs which days a
model is asked about: the keys one model declined are 17–25% harder than the
keys it kept, unanimously across six witnesses, so a complete-case convention
would have scored it on an easier subset than the registry it was drawn from.

Model ranking does not favor architectural constraint — a gradient-boosted tree
with site identity is most accurate at one and three days — and the
spatial-partition question is unresolvable in the regime it was posed in. A
post-outcome descriptive analysis suggests the information a model is given
dominates the estimator by two orders of magnitude, but that is a hypothesis
this cohort generated rather than one it confirmed.

A published skill score is not interpretable without its reference model, its
key set and its information set, and every comparison here is descriptive for
this fixed, availability-selected cohort.

## 8. Open Research

**Data.** The derived panel, station registry, and metadata describing
acquisition of the 2021–2023 evaluation data will be deposited as one versioned
dataset at `[DATA DOI TO BE MINTED]` under `[DATA LICENCE TO BE ASSIGNED]`,
subject to the redistribution terms of the source products. Where source data
cannot be redistributed, SI16 provides the corresponding acquisition
instructions and provenance metadata. Primary sources are USGS NWIS
([U.S. Geological Survey, 1994](https://doi.org/10.5066/F7P55KJN)), Daymet V4 R1
([Thornton et al., 2022](https://doi.org/10.3334/ORNLDAAC/2129)) and gridMET
([Abatzoglou, 2013](https://doi.org/10.1002/joc.3413)), each under its own terms.

**Software and reproduction.** The analysis software and a reproducible Python
3.12 computational environment will be archived at
`[SOFTWARE DOI TO BE MINTED]`. The MIT licence applies to the software but not
to observational data obtained from third-party providers. SI15 documents the
procedures needed to reproduce Sections 4.1–4.3 and the numerical tolerances
used for stochastic neural-network results. Independent reproduction on a
third-party host has not yet been performed.

## Acknowledgments

[FUNDING, COMPUTATIONAL RESOURCES, COMPETING INTERESTS, AND CRediT ROLES TO BE
COMPLETED BEFORE SUBMISSION.]

We acknowledge the U.S. Geological Survey, the ORNL DAAC and the Northwest
Knowledge Network for the products named in Open Research; acknowledgment of a
provider is not an endorsement of this analysis.

## Supporting Information

SI01 cohort, registry and partition counts; SI02 issue-time boundary; SI03
equations and units; SI04 analysis plan and chronology; SI05–SI06 the
comparison set; SI07 all-model scores with MAE and bias; SI08 probability and
reliability schemas; SI09 architecture and information-matched controls with
training and tuning details; SI10 temporal coverage; SI11 the spatial folds,
cluster geometry and the hydrologic-state strata; SI12 outcome quality control
and qualifiers; SI13 the history-dependent external arm; SI14 missingness and
failure cases; SI15 reproduction procedures and computational environment; SI16 rights and
data dictionary; SI19 development-period spatial analyses; SI20 the method,
estimands, prespecification status, analysis classification, and complete
Section 4.6 results.

Figures S1–S3 give cohort geometry, temporal roles and model dataflow; S4–S8
expand Sections 4.3 and 4.5; S9 is a development-period conformal sensitivity not
comparable with any held-out figure; S10 gives the architecture controls; and
S11 presents the matched spatial-transfer panels and the unresolved geometry
contrast.

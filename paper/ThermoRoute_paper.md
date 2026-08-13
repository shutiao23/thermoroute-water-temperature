# Benchmark design governs reported skill in daily river water-temperature prediction

**[AUTHOR LIST TO BE COMPLETED]** — one line per author in the final agreed
order, each carrying the preferred citation name, the affiliation number, and a
verified ORCID. The number of authors, their names, and their affiliations are
not asserted here. The signed intake schema that must produce this block is
`docs/FAIR_SUBMISSION_READINESS_AND_TEMPLATES.md` §2.

^1^ [AFFILIATION 1 TO BE COMPLETED — department or laboratory, institution, city, postcode, country]

^2^ [AFFILIATION 2 TO BE COMPLETED — add or delete affiliation lines to match the final author list]

**Corresponding author:** [CORRESPONDING AUTHOR TO BE COMPLETED — verified name,
ORCID, affiliation number, and institutional e-mail address]

## Key Points

- Withholding a gauge's own recent water temperature costs 1.3-1.7 °C; changing
  the model class costs under 0.01 °C at the same sites.
- Future weather is worth 0.13-0.63 °C and worth less, not more, once the local
  gauge is gone, so the two are complements.
- Seven-day skill of +0.250 against persistence falls to +0.038 against damped
  persistence; the information results are descriptive.

## Abstract

Daily river water temperature is strongly persistent and strongly seasonal, so
reported machine-learning skill depends on what a model is compared against and
on what it is allowed to see. Holding one frozen panel of 120 U.S. gauges across
15 hydrologic regions fixed, we vary the evaluation design and the information
supplied, scoring every arm on one common registry of station/date/horizon keys
with all preprocessing fitted strictly backwards in time. On a pre-specified
2021-2023 held-out window, a constrained deep predictor's seven-day
station-median skill is +0.250 against persistence but +0.038 against damped
persistence, and a gradient-boosted tree with site identity is the most accurate
model at one and three days. Descriptive analyses on the same keys then measure
what the information itself is worth. Withholding a gauge's own recent water
temperature costs 1.3-1.7 °C at every lead and at every one of 116 stations,
two orders of magnitude more than the model class, which is worth under
0.01 °C wherever that history is present. Realized future meteorology is worth
0.13-0.63 °C, is almost entirely future air temperature, survives two
pre-registered placebos, and about half of it survives substitution of an
archived fixed-lead forecast. Future weather and the local gauge are
complements rather than substitutes: forcing is worth less once the gauge is
removed. The estimator begins to matter only where information is scarce. The
held-out comparison is confirmatory; every information result is post-outcome
and descriptive.

**Plain Language Summary.** River temperature changes slowly from day to day, so
a forecast that repeats yesterday's reading is already fairly accurate, and one
that also nudges it toward the usual value for the time of year is better still.
We tested predictions at 120 U.S. gauges under a deliberately demanding
evaluation: every model scored on exactly the same days and sites, none allowed
to see anything from after the moment it was asked to predict. Two things
mattered far more than which method was used. The first is whether the river
being predicted has a thermometer in it: taking away a gauge's own recent
readings hurt accuracy roughly two hundred times more than switching between
entirely different families of model. The second is knowing the weather that
has not happened yet, which is worth about a third as much as the gauge, is
almost entirely about air temperature, and turns out to be worth *less* at a
river without a thermometer rather than more. Better forecasts of the weather
therefore cannot substitute for putting a sensor in the water. Much of what is
usually reported as a modelling advance is, on this evidence, a statement about
what the model was given and what it was compared against.

**Keywords:** river water temperature; benchmark design; strong baselines;
spatial holdout; comparative evaluation; conformal prediction.

---

## 1. Introduction

Stream and river thermal regimes set dissolved-oxygen saturation, metabolic
rates, life-stage timing, and habitat suitability, and they respond to
atmospheric forcing, discharge, groundwater exchange, riparian shading, channel
geometry, and regulation
([Caissie, 2006](https://doi.org/10.1111/j.1365-2427.2006.01597.x)). Water
managers need daily-resolution temperature at gauged reaches, and the supply of
daily records from national monitoring networks has made the variable one of the
most heavily modeled targets in applied hydrological machine learning.

The prevailing understanding in that literature is that deep sequence models have
substantially advanced daily river-temperature prediction. Nonlinear air–water
regressions ([Mohseni et al., 1998](https://doi.org/10.1029/98WR01877)) and
hybrid air-temperature and discharge formulations
([Toffolon and Piccolroaz, 2015](https://doi.org/10.1088/1748-9326/10/11/114011))
were succeeded by recurrent, multi-task, and physics-guided river-network
architectures that report large error reductions
([Rahmani, Lawson, et al., 2021](https://doi.org/10.1088/1748-9326/abd501);
[Sadler et al., 2022](https://doi.org/10.1029/2021WR030138);
[Barclay et al., 2023](https://doi.org/10.1029/2023WR035327)), and graph
architectures make spatial dependencies explicit
([Topp et al., 2023](https://doi.org/10.1029/2022WR033880)). Transfer to unseen
conditions is a question of spatial sensitivity rather than of raw accuracy, and
it is the one this study isolates by holding out whole regions rather than
neighbouring sites.

Those reported gains are extraordinarily heterogeneous, and the heterogeneity
does not organize itself by architecture. Published accuracies are not
comparable across studies because reference models, key sets, and data-splitting
conventions differ
([Corona and Hogue, 2025](https://doi.org/10.5194/hess-29-2521-2025)). The
dispersion survives within single panels: on the data used here, one fixed model
and one fixed set of prediction days produce a seven-day skill of +0.251 or of
+0.038 depending only on which reference model is placed in the denominator. An
architectural account cannot produce that spread, because the architecture does
not change between the two numbers.

Hydrology has known for a long time that a reported score is a statement about
its benchmark, from the differential split-sample test
([Klemeš, 1986](https://doi.org/10.1080/02626668609491024)) through explicit
upper and lower benchmarks
([Seibert et al., 2018](https://doi.org/10.1002/hyp.11476)) to benchmarking as
the centre of what machine learning can tell us about a hydrological system
([Nearing et al., 2021](https://doi.org/10.1029/2020WR028091)). None of this is
contested. What is missing for daily river temperature is not the argument but
the measurement: how large the benchmark effect actually is for this variable,
on identical prediction keys, with an estimand that pairs models station by
station rather than comparing marginal aggregates.

Three design choices make that measurement attributable. First, skill is usually
quoted against naive persistence, climatology, or an air–water regression. Daily
mean water temperature has large thermal inertia, so relaxing the last
observation toward a seasonal climatology already removes most of the error a
learned model can remove. Second, models are scored on model-specific
complete-case sets, so a model can gain apparent accuracy by declining to
predict on hard days, and preprocessing statistics are commonly fitted on
periods that include the evaluated interval
([Arsenault et al., 2018](https://doi.org/10.1016/j.jhydrol.2018.09.027)).
Third, spatial generalization is reported from random held-site splits; when a
held-out gauge's neighbours remain in training, the quantity measured is closer
to interpolation than to transfer
([Kratzert et al., 2019](https://doi.org/10.1029/2019WR026065)). None of these
choices is detectable from a reported RMSE, and each can inflate it.

Here we hold the model and the data fixed and vary the design choices instead.
We assembled a panel of 657,480 site-days from 120 stable U.S. Geological Survey
site numbers spanning 2006–2020, 34 states, and 15 two-digit hydrologic unit
(HUC2) regions, tuned all models on data through 2020, and evaluate them on one
common set of station/date/horizon keys at 1-, 3-, and 7-day leads, with a
held-out 2021–2023 test window. The cohort is availability-selected rather than
drawn, which bounds what any of it can generalize to; Section 3.6 and Section 6
state that bound rather than working around it.

Five questions organize the study. How much of a reported gain survives a strong
reference? Which model class is actually most accurate on identical keys? How
much of a reported spatial transfer survives whole-region holdout? In which
issue-time-identifiable hydrologic states does a learned model still add skill?
And — since every model above sees only issue-time information — how much
predictability is withheld by that restriction rather than by the models, which
we bound in Section 4.8 with realized future meteorology. The predictor whose
skill we decompose, ThermoRoute, is described in Section 3.1; its architecture is
the object under test rather than the contribution.

---

## 2. Data and design

### 2.1 The frozen panel and the cohort it defines

The development record is a single derived panel,
`data_usgs/panel_usgs_120v2.parquet`, bound by a manifest
(`data_usgs/frozen_panel_v1.json`) to a station registry
(`data_usgs/station_registry_v1.csv`). Each of 120 sites carries one row for
every calendar day from 2006-01-01 through 2020-12-31, giving 657,480 rows;
missing observations are explicit nulls, not absent rows. The models
consume seven raw variables: water temperature, streamflow, air temperature,
precipitation, a relative-humidity proxy, Daymet daylight-period mean incoming
shortwave radiation, and wind speed, from USGS NWIS daily values
([U.S. Geological Survey, 1994](https://doi.org/10.5066/F7P55KJN)), Daymet V4
([Thornton et al., 2022](https://doi.org/10.3334/ORNLDAAC/2129)) and gridMET
([Abatzoglou, 2013](https://doi.org/10.1002/joc.3413)).

The cohort was not sampled at random from U.S. rivers, and nothing here treats
it as though it were. Candidate selection rejected 1,345 of 1,465 candidate
stations, mostly for lacking joint water-temperature and discharge
availability. The retained cohort is availability-enriched, and the 2019–2020 interval
participated in its construction, so it is development data in the strict sense,
used for model tuning and diagnosis rather than as an independent test.

Observed water temperature is absent on about 15.8% of panel rows and discharge
on about 2.8%. Among retained stations, 38 share a repeated hydrologic unit code
with another retained station and 19 have another retained station within 10 km,
so station-level independence is not tenable.

### 2.2 Temporal roles and the common key set

Temporal roles are fixed before any model is fitted, and a training sample is
admitted only if its issue date and every one of its target dates fall inside the
same partition.

| Role | Dates | Rows | Observed `WTEMP` |
|---|---:|---:|---:|
| Training | 2006–2015 | 438,240 | 341,646 |
| Validation | 2016–2017 | 87,720 | 84,074 |
| Calibration | 2018 | 43,800 | 42,279 |
| Development evaluation | 2019–2020 | 87,720 | 85,621 |
| Held-out test | 2021–2023 | — | observed |

Models are tuned exclusively on data through 2020; the 2021–2023 window is held
out and used only for the comparative evaluation of Section 4.4. On the
development-evaluation partition, the intersection of admissible keys across all
primary models comprises 249,072 station/date/horizon keys, 83,024 per lead; the
held-out window carries its own common-key registry of the same form. The 249,072
figure describes the development registry only and is never used for a held-out
claim.

![Study sites and evaluation design.](figures/fig01_study_design.pdf)

**Figure 1. Study sites and evaluation design.** (a) The 120 U.S. gauges colored
by the four deterministic whole-HUC2-region folds. (b) Temporal partitions.
(c) The three evaluation tasks: known-site forecasting, random held-site
transfer, and whole-region gauged transfer. (d) Issue-time information boundary:
all models are scored on the identical common forecast-key registry. No
model-specific complete-case set is permitted.

### 2.3 Evaluation-period inputs and the information boundary

The test interval is 2021-01-01 through 2023-12-31 for the same 120-site cohort. Meteorology is represented at each station coordinate and is not aggregated over upstream catchments, a real limitation for large basins (Section 6).

The primary information set uses provider values dated no later than each historical issue date and consumes no horizon-specific future weather forecast. This is a date-indexed retrospective hindcast, and not a re-execution of what a forecaster could have run on the day (Section 6): the as-issued provisional vintage of a gridded product cannot be reconstructed after the fact, so archiving requests, responses, timestamps, and checksums freezes the dataset actually evaluated without proving that identical values were available operationally at the time.

Daily mean water temperature (00010, °C), discharge (00060, cfs), and raw-only gage height (00065, ft) come from USGS NWIS. Test-window daily means are outcomes only; they are not read by model-selection, feature-selection, threshold-selection, calibration, or station-inclusion code, all of which is fixed on data through 2020.

A pooled-preprocessing sensitivity re-fits the model suite with pooled transforms in place of per-station ones; it is not a site-disjoint cohort (Section 3.4).

---

## 3. Methods

### 3.1 The predictor under test

ThermoRoute predicts at issue time *t*, station *i*, and lead *h* by adding a
bounded learned residual to a damped-persistence anchor,

> **(1)**  $A_{i,t+h} = c_{i,t+h} + \phi_i^h\,(y_{i,t} - c_{i,t})$

where $c$ is a seasonal climatology and $\phi_i$ a per-station decay fitted by
no-intercept least squares on training-period consecutive-day anomaly pairs.
The same object, fitted identically, is the damped-persistence reference.

The residual comes from a strictly left-looking temporal convolutional encoder
over a 14-lag window, with a horizon-conditioned variable/lag router, a
three-expert mixture, and a $\pm1\,{}^\circ$C algebraic bound on the deviation
from the anchor. One model serves all three leads; 38,505 trainable parameters;
five seeds averaged. The full specification is in Text S3. The architecture is
the object under test, not the contribution: Section 4.3 shows an
information-matched plain causal network reproduces it to within 0.023 °C.

![Common anchor–residual formulation under matched
information.](figures/fig02_model_concept.pdf)

**Figure 2. The formulation every compared model shares.** LightGBM is a
*raw-target* model with site identity; ResidualLightGBM, the plain causal TCN,
and ThermoRoute predict a residual around the same anchor, with identical keys
and information set. The figure is structural and carries no evidence from
either period.

### 3.2 The reference set

The composition of the reference set is the most consequential methodological
choice in this study, because it is what the reported gain is a gain over. Each
member removes one alternative explanation for apparent skill.

**Persistence** (`ŷ_{t+h} = y_t`) quantifies the raw memory of the series.
**Damped persistence**, equation (1) fitted on the training period, absorbs
inertia and seasonality and is the reference for the primary comparisons.
**Seasonal climatology** isolates the seasonal cycle alone. **LightGBM**
([Ke et al., 2017](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html))
is the principal learned reference, run globally with site identity as a
categorical feature and per station. A **global LSTM** with a station embedding
represents the deep sequence family
([Rahmani, Lawson, et al., 2021](https://doi.org/10.1088/1748-9326/abd501);
[Zwart et al., 2023](https://doi.org/10.3389/frwa.2023.1184992)). A **plain
causal temporal convolutional network** is the information-matched deep
reference: it receives the same outcome-free issue-time inputs as the full
model, is matched to within 2% of its parameter count (38,346 against 38,505),
and predicts an unrestricted residual around the same anchor while omitting the
relaxation proposal, router, mixture, and residual bound.

Tuning budgets are documented but *not* equalized across model classes: any
statement about LightGBM here is scoped to this four-candidate procedure and
this feature schema, not to gradient boosting in general. Seven one-factor
architecture controls are deletion and intervention sensitivities and do not
prove component necessity or identify a mechanism.

An air2stream-style hybrid reference (Toffolon and Piccolroaz, 2015) is fitted
per station and scored on the same held-out keys as every other model (SI07).
It is an unofficial empirical thermal-recurrence variant of the published
formulation — not the official code and not a validated reproduction — so no
process-side claim is attached to its scores.

### 3.3 Leakage control

Leakage control is a set of mechanisms checkable against the code and the artifacts, not a statement of intent. Every predictor a primary model consumes carries a date no later than the issue date, and the encoder of Section 3.1 is strictly left-looking (Text S3). Standardization constants, the q90 event thresholds (2006–2015), the conformal offsets (2018), and the Platt calibrators (2018) are each fitted on data strictly preceding the interval on which they are applied. We record one honest gap: the issue-date auxiliary quantities behind the relaxation proposal and the regime gate carry no separate auxiliary-path mask, so the output is not guaranteed invariant to the fill values. A fresh interpreter replays every trained member and rejects merely self-consistent prediction files.

### 3.4 Two spatial partitions, separated from local adaptation

Random held-site splits are the usual way to report spatial generalization, but they can be optimistic: if a held-out gauge's neighbours remain in training, the model reaches that site's thermal regime through correlated forcing and shared preprocessing statistics. We therefore run a 2×2 factorial on the independent 2021–2023 window separating the spatial partition from the local-adaptation policy (protocol v1; Section 4.5 and SI11).

The **random held-site warm-start** arm uses four balanced folds in which held-site histories still contribute to global panel preprocessing; it is the arm most comparable to published random-split results, and for that reason not the one we treat as the spatial test. The **held-region gauged transfer** arm packs the 15 HUC2 groups into four folds of [30, 30, 31, 29] stations and holds out whole regions; the mean distance from a held-out station to its nearest training gauge is 289 km. *Local* adaptation lets a target station's own 2006–2015 history enter its climatology, damped anchor, and imputation medians; *pooled* adaptation fits those statistics over in-fold stations only.

Held-out sites still supply their own observed water temperature through the issue date. This is **gauged** transfer to a region whose gauges were not used in fitting; it is **not** prediction at a site with no observational record.

### 3.5 Conformal intervals

Split-conformalized quantile intervals are calibrated on the 2018 year only
and are reported as an empirical marginal coverage diagnostic with width and
interval score beside every coverage figure; no finite-sample or
conditional-coverage claim is made (details and sensitivities in SI08).

### 3.6 Estimand, metrics, and comparison set

**Two differently signed quantities are reported, and they are never combined.**
The paired effect is

> **(2)**  $\Delta\mathrm{RMSE} = \mathrm{RMSE}(\mathrm{candidate}) - \mathrm{RMSE}(\mathrm{reference})$, in ${}^\circ$C, **negative** favors the candidate

and the skill score is

> **(3)**  $\mathrm{skill} = 1 - \mathrm{RMSE}(\mathrm{candidate})/\mathrm{RMSE}(\mathrm{reference})$, dimensionless, **positive** favors the candidate

Equation (2) is the formal estimand and carries a °C unit; equation (3) is a
bare signed number. Every value in Section 4 is labeled with which it is.

The sampling unit is the station. For each lead, unweighted RMSE is computed on
the common daily keys per reportable station, and the primary effect is the
unweighted median across stations of equation (2) with ThermoRoute as candidate;
a station/lead cell is reportable only with at least 100 valid paired targets.

The comparison set is five head-to-head rows: ThermoRoute against damped
persistence at 1 d, 3 d, and 7 d with a 0.00 °C reference margin, and against
LightGBM at 3 d and 7 d with a +0.05 °C numerical ceiling. That ceiling is a
stated numerical reference, not a decision threshold, and carries no ecological
or regulatory importance.

Two uncertainty procedures accompany each row, both clustered at HUC2: a
one-sided p-value from exact enumeration of all 2^K whole-cluster sign vectors,
and a 10,000-draw whole-HUC2 cluster bootstrap percentile interval for the
median station effect, Holm-adjusted
([Holm, 1979](https://www.jstor.org/stable/4615733)) over the five p-values.
The cohort came from an availability filter, not probability sampling of HUC2
regions, so exchangeable region sampling is not established, and the clustered
intervals and p-values are approximate descriptive sensitivities, not decision
evidence. No comparison is written as a superiority, non-inferiority,
equivalence, or parity statement, and none generalizes to a national population.

---

## 4. Results

Sections 4.1–4.3 report development-period (2019–2020) benchmark diagnostics.
The 2019–2020 partition participated in cohort construction and informed model
selection, so these values are development diagnostics rather than an independent
test; they are reported because they are the basis on which the model suite and
the comparison set were fixed, and they anticipate the held-out evaluation.
Unless stated otherwise, all development values are five-seed ensemble means on
the 249,072 common keys, with station-median RMSE in °C. Section 4.4 reports the
held-out 2021–2023 comparative evaluation; all of its values are unweighted
station medians over the reportable stations on the common held-out key registry
(`outputs/final/`), the same station-first estimator used in Sections 4.1–4.3.
Pooled RMSE is reported only as a sensitivity in the Supporting Information and
is never labelled a station median.

### 4.1 The reference model, not the architecture, sets the reported gain

Most of a reported gain does not survive a strong reference. On identical keys,
the same fixed model reports a seven-day median station skill of +0.251 against
persistence and +0.038 against damped persistence — a factor of 6.6 between two
numbers that differ only in the denominator of equation (3). A study reporting
only the persistence column would describe this model as gaining a quarter of
the seven-day error budget. A study reporting both columns describes the same
fitted weights as gaining four percent.

| Lead | Persistence | Damped persistence | LightGBM | LSTM | ThermoRoute |
|---:|---:|---:|---:|---:|---:|
| 1 d | 0.803 | 0.774 | 0.578 | 0.662 | 0.631 |
| 3 d | 1.576 | 1.406 | 1.280 | 1.323 | 1.291 |
| 7 d | 2.217 | 1.738 | 1.649 | 1.679 | 1.657 |

*Development-period (2019–2020) station-median RMSE, °C.*

**The reference is itself a fitted object, and its construction moves the
headline.** We rebuilt the anchor under seven predeclared one-factor variants,
rescoring each against the *published* ThermoRoute predictions so that every
difference is attributable to the reference alone. Seven-day skill against damped
persistence then ranges from +0.025 to +0.063, with the published +0.038 in the
middle (`outputs/final/anchor_sensitivity.parquet`); the low end comes from a decay fitted directly at each lead, a
*stronger* reference. The seven-day headline is the least stable, so the number
this paper quotes most often is the most sensitive to how the reference was
specified: the better the anchor, the less the learned model adds.

![Decomposition of reported skill on the held-out window.](figures/fig03_skill_decomposition.pdf)

**Figure 3. Decomposition of reported skill on the held-out 2021–2023 window
(116 reportable stations).** (a) Station-median RMSE by lead. (b) Seven-day
memory gain 0.49 °C against learned gain 0.07 °C. (c) Per-station memory
fraction, median 0.875. (d) Paired ΔRMSE; negative values favor ThermoRoute.

### 4.2 The most accurate model on this panel is a gradient-boosted tree

It does, and we report it directly. On the development window the tree
ensemble has the lowest station-median RMSE at all three leads: 0.578, 1.280,
and 1.649 °C against ThermoRoute's 0.631, 1.291, and 1.657 °C. The paired
station-level differences (`ThermoRoute − LightGBM`, equation 2) are +0.046 °C
at 1 day, +0.008 °C at 3 days, and −0.005 °C at 7 days, with ThermoRoute win
rates of 0.00, 0.40, and 0.60 across the 120 stations. The one-day difference
is outside the frozen five-test family; as an exploratory check it is
+0.035 °C (sign-flip p = 1.0 under the one-sided family convention). The
architecture's remaining argument is not accuracy but the ±1 °C algebraic
bound of Section 3.1, which held on 100.00% of audited rows — a property of
the construction rather than a fitted outcome.

### 4.3 An information-matched plain convolutional network reproduces the architecture

Very little of the architecture's structure is doing work. Against the
information-matched plain causal temporal convolutional network (same inputs,
same anchor, same optimiser budget, 38,346 against 38,505 parameters) the full
architecture's paired median station effect, computed within seed on identical
keys, ranges across the five seeds from −0.0003 to −0.0105 °C at 1 day and
−0.0035 to −0.0226 °C at 7 days — every seed favors the full model. The
one-factor deletions agree: removing the temporal encoder is the largest single
effect (1-day station-median RMSE 0.631 to 0.679 °C), while removing the router,
mixture, dynamic prior, fixed-relaxation constraint, or residual bound moves the
1-day figure by at most 0.004 °C (SI09). These are five-seed deletion and
intervention sensitivities on paired keys; they do not prove component necessity.

### 4.4 Held-out 2021–2023 evaluation

The held-out evaluation applies the Section 3 model suite and comparison set,
fixed on data through 2020, to 2021-01-01 through 2023-12-31. Every number is an
unweighted station median over reportable stations on the common held-out key
registry (*n* = 116; per-model metrics in SI07). Station-median skill is +0.250
against persistence and +0.038 against damped persistence, and the tree with
site identity leads at one and three days (0.589 and 1.304 °C). At seven days
ThermoRoute reaches 1.694 °C and is not separated from the tree
(−0.009 °C paired) or from the information-matched plain convolutional network
(+0.015 °C paired). Row 5 is not
evidence of equivalence: the smallest effect this cohort's cluster structure can
resolve at α = 0.05 is larger than the effect observed, so it is underpowered
rather than null. The damped-persistence rows sit at three to seven times the
smallest detectable effect, so the family is read row by row.

**Table 1 — every model on the common held-out keys.** Station-median RMSE, MAE
and bias in °C at each lead, 116 reportable stations, identical
station/date/horizon keys for every row.

<!-- TABLE all-model scores (generated) -->

| Model | RMSE 1 | RMSE 3 | RMSE 7 | MAE 1 | MAE 3 | MAE 7 | bias 1 | bias 3 | bias 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Persistence | 0.813 | 1.638 | 2.202 | 0.597 | 1.235 | 1.686 | -0.000 | -0.001 | -0.004 |
| DampedPersistence | 0.789 | 1.454 | 1.773 | 0.591 | 1.100 | 1.340 | -0.029 | -0.076 | -0.143 |
| Climatology | 1.899 | 1.902 | 1.903 | 1.485 | 1.486 | 1.485 | -0.379 | -0.369 | -0.359 |
| LightGBM | 0.589 | 1.304 | 1.735 | 0.431 | 0.995 | 1.315 | -0.025 | -0.098 | -0.196 |
| LSTM | 0.663 | 1.358 | 1.712 | 0.503 | 1.044 | 1.292 | -0.001 | -0.051 | -0.137 |
| PlainMLP-7var | 0.690 | 1.374 | 1.720 | 0.520 | 1.054 | 1.311 | -0.004 | -0.059 | -0.128 |
| PlainCausalTCN-7var | 0.646 | 1.330 | 1.710 | 0.477 | 1.021 | 1.299 | 0.000 | -0.031 | -0.122 |
| Air2stream | 0.719 | 1.459 | 1.825 | 0.559 | 1.115 | 1.366 | -0.020 | -0.110 | -0.189 |
| ThermoRoute | 0.640 | 1.337 | 1.694 | 0.463 | 1.013 | 1.269 | 0.011 | -0.029 | -0.130 |

**Table 4.6 — paired comparisons on the held-out window.** Station-level ΔRMSE
(°C, negative favors ThermoRoute) for the frozen five-test family of Section
3.6; cluster-bootstrap intervals over whole HUC2 regions, Holm-adjusted
sign-flip p-values.

<!-- TABLE 4.6 (generated) -->

| # | Comparison | Lead | ΔRMSE (°C) | CI low | CI high | Win rate | Stations | p (sign flip) | Holm p |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | ThermoRoute vs. DampedPersistence | 1 d | -0.129 | -0.199 | -0.076 | 0.90 | 116 | 3.1e-05 | 1.5e-04 |
| 2 | ThermoRoute vs. DampedPersistence | 3 d | -0.108 | -0.143 | -0.073 | 0.91 | 116 | 3.1e-05 | 1.5e-04 |
| 3 | ThermoRoute vs. DampedPersistence | 7 d | -0.069 | -0.086 | -0.057 | 0.95 | 116 | 6.1e-05 | 1.8e-04 |
| 4 | ThermoRoute vs. LightGBM | 3 d | 0.015 | 0.010 | 0.025 | 0.24 | 116 | 1.0e+00 | 1.0e+00 |
| 5 | ThermoRoute vs. LightGBM | 7 d | -0.009 | -0.017 | 0.002 | 0.59 | 116 | 7.4e-02 | 1.5e-01 |

<!-- FIGURE_ANCHOR id=S10 state=POST role=first_citation source=paper/FIGURE_REDRAW_SPEC.md#figure-s10 -->

One-factor ablations neither help nor hurt systematically at any lead: fixing
the dynamic prior and
removing the bounded-residual constraint give station-median skill against
damped persistence of +0.180 and +0.176 at one day against +0.173 for the full
model (SI09). These are deletion and intervention sensitivities and do not prove
component necessity; neither adds material point accuracy on the independent
window.

### 4.5 Matched spatial-transfer experiment on 2021–2023

The development-period whole-region finding (SI19) is re-tested on the independent window with a matched 2×2 factorial design that separates spatial geometry from local adaptation. Station-agnostic gradient-boosted trees are fitted under four leave-HUC2-region folds and four balanced random folds over five split seeds, crossed with local and pooled preprocessing (protocol v1; SI11). The design moves the median nearest-training-gauge distance from 60 km under random held-site folds to 263 km under whole-region folds. Under local adaptation the region-minus-random paired station penalty is +0.006 °C at one day, +0.009 °C at three days, and +0.007 °C at seven days for the raw-target tree.

**This experiment does not resolve a geometry effect, and we do not report one.** The penalty is the same size as its own resampling noise — the median per-site standard deviation across the five split seeds is 0.004–0.007 °C — and four whole-region folds provide four independent regional observations. We report the penalty as *below this design's resolution*.

The pooled arm is defective: its preprocessing reused fold 0's statistics across folds 1–3, whose held-out stations lie inside fold 0's training set (a recorded defect, DLOG-012). We report no adaptation main effect from it, and no number from it is carried into the Discussion or the Conclusions.

![Spatial transfer under matched holdouts.](figures/fig04_spatial_transfer.pdf)

**Figure 4. Spatial transfer under matched random-site and whole-region holdouts on the independent 2021–2023 window** (panel detail and pooled-arm station-set caveats in SI11).

### 4.6 Hydrologic conditions governing incremental skill

The remaining learned gain concentrates in specific hydrologic states. Station thermal memory, the anomaly half-life of the train-fitted damped-persistence anchor — an anomaly half-life, not an e-folding time — has a station median of 6.9 days (interquartile range 5.0–11.9). At seven days the median memory gain is 0.49 °C and the median learned gain 0.07 °C (Figure 3b), and the median per-station fraction of the error reduction delivered by seasonal memory is 0.875. The station correlation between log half-life and learned gain is −0.40 at one day: longer-memory stations leave less for the learned model to add at short leads.

**Issue-time-identifiable states.** Thresholds are station-specific empirical quantiles fitted on the 2006–2015 training period only (SI11). At seven days the learned model's median paired ΔRMSE over damped persistence is −0.069 °C on all keys; it is largest under low thermal anomaly (−0.229 °C) and rapid recent cooling (−0.170 °C), and reaches −0.088 °C under rapid recent warming. No issue-time-identifiable state reverses the ordering.

**Table 3 — learned gain by hydrologic state at seven days.** Station-median
paired ΔRMSE against damped persistence; negative favors ThermoRoute. Issue-time
strata are identifiable when the forecast is made; outcome-conditioned strata
are not.

<!-- TABLE hydrologic states (generated) -->

| State (7-day keys) | ΔRMSE, ThermoRoute − damped (°C) | stations | median keys |
| --- | ---: | ---: | ---: |
| All keys | -0.069 | 116 | 1,060 |
| issue low anomaly | -0.229 | 105 | 71 |
| issue high anomaly | -0.080 | 114 | 139 |
| issue rapid recent warming | -0.088 | 116 | 107 |
| issue rapid recent cooling | -0.170 | 114 | 99 |
| issue strong warming pressure | -0.072 | 115 | 111 |
| issue strong cooling pressure | -0.058 | 116 | 90 |
| issue low flow | -0.072 | 92 | 143 |
| issue high flow | -0.060 | 104 | 89 |
| issue rapid flow rise | -0.054 | 116 | 103 |
| issue rapid flow recession | -0.072 | 116 | 99 |
| actual rapid warming | -0.300 | 116 | 109 |
| actual rapid cooling | -0.184 | 116 | 102 |
| actual warmest decile | -0.066 | 116 | 141 |
| actual coldest decile | 0.029 | 109 | 77 |
| actual high flow | -0.078 | 107 | 87 |
| actual low flow | -0.078 | 92 | 147 |

**Retrospective outcome-conditioned diagnostics.** Stratifying by the target outcome — retrospective, not an issue-time state — days that subsequently warm rapidly reach −0.30 °C at seven days, and days that subsequently cool rapidly −0.18 °C. On the coldest target decile the anchor alone is marginally better (+0.03 °C), the only stratum in which the learned correction does not pay for itself. Those strata select the days on which a damped anchor must fail, and a manager cannot identify them at issue time; we read them as a diagnostic of where the anchor is weak rather than an operational benefit.

### 4.7 What local thermal state is worth

Section 4.5 could not resolve a spatial effect because every arm kept the held
station's own thermal history. This section removes that history instead of the
geography, which is the contrast the design can actually resolve.

Four nested information levels are fitted under whole-region holdout: **L0**
keeps everything; **L1** pools the climatology and damped rate over training
stations; **L2** removes every target-site water-temperature input, leaving
discharge and meteorology; **L2-U2** removes discharge as well. Prohibited
inputs are dropped from the design matrix rather than filled; all 24 proofs
return a maximum absolute difference of exactly zero, and a negative control
re-admitting a single water-temperature lag is caught with a 147 °C shift
(`outputs/final/information_ladder_v6_authority_v1/`).

**Local thermal state is the dominant information in this problem.** Withholding
it costs 1.3–1.7 °C, at every lead, at every one of 116 stations. That is two to
thirteen times the value of realized future meteorology (Section 4.8) and
roughly seventy times the architecture effect of Section 4.3. A study that keeps
the target gauge's recent readings and then reports a model comparison is
comparing models inside the regime where the largest available information is
already present.

**Almost all of that value is the recent sequence, not the station's long
history.** Pooling a station's climatology and damped rate costs 0.01–0.16 °C;
removing its recent readings costs 1.17–1.71 °C. Cold-starting a gauged site is
therefore nearly free, while thermally ungauged prediction is a different
problem. Once water temperature is gone, also removing discharge changes
station-median RMSE by 0.009–0.084 °C with every interval covering zero.

**Spatial geometry matters, but only once local history is gone.** Section 4.5
measured a geometry penalty of 0.004–0.009 °C and could not resolve it. Running
the same four levels under random-site holdout shows why: the penalty is
0.006–0.014 °C at L0 and 0.073–0.129 °C at L2. The station-level
double difference is +0.066 to +0.124 °C, and every whole-HUC2 interval excludes
zero. A random-site split in a dense network measures something closer to
interpolation than transfer, but the cost of that substitution is invisible
while the held gauge still supplies its own thermal history.

---

### 4.8 What future meteorology would be worth

Every result above is issue-time-only, so none of it says whether the small
residual learned gain reflects a limit of the models or a limit of the
information they were given. This section refits the tree models with the
*realized* meteorology over t+1 … t+h substituted for the issue-time forcing, on
the same cohort and the same 116 reportable stations. F3 is a **retrospective
realized gridded meteorological oracle**: an upper bound on what perfect weather
information would be worth, not a forecast product and not an operational gain.
It is also *post-outcome* — the 2021–2023 window was already open when it was
specified — so it is a descriptive reanalysis and not a confirmatory test, and
it is reported here rather than in the Abstract or the Key Points for that
reason.

**Table 4.13 — value of realized future meteorology.** Station-first forcing
value $V_F = \mathrm{median}_i[\mathrm{RMSE}_i(F0) - \mathrm{RMSE}_i(F3)]$ in °C.
CI is a 10,000-draw whole-HUC2 cluster bootstrap; MDE is the smallest effect
this cohort's cluster structure can resolve at α = 0.05.

| Model | Lead | RMSE F0 | RMSE F3 | $V_F$ | 95% CI | MDE |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| LightGBM | 1 d | 0.614 | 0.451 | 0.130 | [0.081, 0.189] | 0.031 |
| LightGBM | 3 d | 1.330 | 0.765 | 0.542 | [0.415, 0.721] | 0.152 |
| LightGBM | 7 d | 1.742 | 1.063 | 0.627 | [0.423, 0.803] | 0.172 |
| ResidualLightGBM | 1 d | 0.606 | 0.446 | 0.125 | [0.084, 0.183] | 0.028 |
| ResidualLightGBM | 3 d | 1.327 | 0.752 | 0.578 | [0.406, 0.704] | 0.160 |
| ResidualLightGBM | 7 d | 1.722 | 1.087 | 0.605 | [0.445, 0.795] | 0.151 |

Every value is three to four times its own minimum detectable effect, where the
architecture effects of Section 4.3 are smaller than theirs, and no
leave-one-HUC2-out range changes sign. Formed as a station-level double
difference, the forcing value rises by +0.415 °C [0.301, 0.541] from one to
three days but by only +0.074 °C [0.046, 0.101] from three to seven (LightGBM).
Future weather buys most of what it can buy by day three; the seasonal and
per-year strata are in
`outputs/final/forcing_regime_v5_observed_inference_authority_v1/`.

**The gain is event-scale weather information, not seasonal phase.** A placebo
whose design and decision rules were sealed before the outcome existed deranges
the meteorology vector across dates *within each station-month*, preserving each
station's climate and seasonal phase and destroying only the correspondence
between a forecast key and its weather. It retains 0.0% of the true value at one
day, 2.0% at three and 8.8% at seven, and the sealed rule P1 is satisfied at
every model and lead. A second sealed control displaces the realized future by a
whole week, leaving local weather persistence intact and removing only the exact
dates; the +7-day arm retains −0.1%, 0.7% and 14.0%, and P4 is satisfied.

**What the forcing value is worth as a warning.** Both arms were scored on
warm-tail exceedance and signed rapid change, with thresholds fitted per station
on the 2006–2015 training period only. At seven days with the raw-target tree,
the probability of detection for exceedance of a station's training 95th
percentile rises from 0.18 to 0.55 while the false-alarm ratio *falls*, from
0.36 to 0.20, so this is not a base-rate trade. These are deterministic point
predictors and none emits a calibrated event probability; the comparison is an
oracle bound in event space exactly as it is in RMSE space.

**Which rivers depend on future weather.** Of six attributes declared before the
relationships were inspected, the strongest is the one the physics predicts: the
anomaly half-life of the station's own damped anchor, ρ = −0.49 [−0.66, −0.16]
at seven days. Rivers that hold a thermal anomaly for longer carry their own
state forward and need tomorrow's weather less. One attribute is a nuisance
control and it is not null — stations with more complete training records show a
smaller forcing value (ρ = −0.29 [−0.45, −0.07]) — so part of the apparent
physical gradient may be a record-quality gradient. We report these as
exploratory covariation on a fixed, non-probability cohort with about nine
effective spatial clusters, not as an attribution, and we do not fit a
multivariable model the design cannot carry.

**The forcing value is future air temperature.** At seven days, air temperature
alone recovers 0.595 °C of the 0.627 °C full value, while withholding it and
giving everything else its realized future retains only 0.129 °C.

**About half the oracle survives a real forecast.** The F2a arm reproduces that
temperature-only arm with an archived forecast substituted for realized air
temperature at prediction time, and recovers 49–62% of it across leads and both
trees. The series is a **fixed-lead composite**, not a coherent trajectory from
one initialization and not an as-issued operational archive; the archive spans
2021–2023, so this is a measurement of those years; and the arm supplies air
temperature only. The F2b archived-vintage arm remains unrun under the same
seal.

---

### 4.9 Future weather does not substitute for a local gauge

Read separately, Sections 4.7 and 4.8 invite a reading neither tested: that an ungauged reach could buy back with weather what it lost with the sensor. We crossed the axes, refitting the F3 forcing arm at L0 and at L2 under the same whole-region folds and level-legal anchor as Section 4.7.

**Table 4.17 — value of realized future meteorology by information level.**
Station-first paired values in °C under whole-region holdout, 116 reportable
stations, 10,000-draw whole-HUC2 cluster bootstrap.

| Quantity | 1 d | 3 d | 7 d |
| --- | ---: | ---: | ---: |
| Forcing value at L0 (gauged) | 0.116 [0.059, 0.174] | 0.474 [0.299, 0.693] | 0.580 [0.366, 0.779] |
| Forcing value at L2 (thermally ungauged) | 0.038 [0.020, 0.072] | 0.331 [0.249, 0.496] | 0.596 [0.431, 0.804] |
| **Interaction (L2 − L0)** | **−0.076 [−0.098, −0.056]** | **−0.147 [−0.193, −0.079]** | **+0.016 [−0.035, +0.099]** |

Future weather is worth *less* to a model that has lost the local gauge, not more: the interaction is negative at one and three days and excludes zero at both. Realized meteorology earns its value by correcting a trajectory, and at short leads that trajectory is the persistence anchor built from the station's own recent readings; without it the model falls back on a pooled climatology too coarse for tomorrow's weather to sharpen. The two information sources are complements at the leads where local state dominates.

This analysis is post-outcome and descriptive, not a confirmatory test, and is reported outside the Abstract and Key Points for that reason (`outputs/final/forcing_information_interaction_v1/`).

---

### 4.10 The model class matters only where the information does not

Every contrast to this point varies information and holds the model class
fixed. This one does the reverse, and it is the contrast on which the paper's
central claim can fail. A plain causal temporal convolutional network is fitted
in each cell of the crossing, information-matched to the trees, and compared
with the residual-target tree that shares its target formulation. Positive means
the network is worse (full crossing, both holdout geometries and per-cell
minimum detectable effects in
`outputs/final/architecture_geometry_interaction_v1/`).

**In the regime this literature reports, the model class is worth about a
hundredth of a degree.** At L0 under issue-time information the two model
classes differ by at most 0.009 °C at any lead, against station-median errors of
0.53–1.74 °C, and the null holds under random-site holdout as well, the geometry
this literature uses.

**Where information is scarce, the estimator starts to matter.** Remove the
local gauge and the tree wins by 0.02–0.19 °C. The architecture-by-information
interaction is positive in all six geometry-by-lead cells and resolved in three
of them — at one day under both holdouts and at three days under random-site —
while the two seven-day cells and whole-region at three days have intervals
covering zero. What makes the estimator matter is the missing local information,
not the spatial extrapolation the whole-region design bundles with it.

That result depends on the network having been trained far enough, and its own
lineage said it had not: under the shared 40-epoch budget the median best epoch
at L2 was 37, with a third of cells still improving at the cap, while L0
converged well inside it. A budget that binds only where the claim lives is the
information-mismatch failure relocated to the optimiser, so the L2 cells were
refitted with a 300-epoch cap and patience 20. The median best epoch moves to
119 and nothing reaches the new cap, and the penalty does not shrink — it grows,
to 0.249, 0.152 and 0.065 °C, with every change interval covering zero. Longer
training makes the network worse against the held region while its in-fold
validation still improves, which is what an information-scarce regime looks
like from the optimiser's side. The reported figures are the shorter-budget
ones, so they are the conservative pair
(`outputs/final/tcn_convergence_sensitivity_v1/`).

**Given perfect future weather, the sequence model is the better one.** At L0
with F3 the sign reverses, to −0.096 [−0.177, −0.038] at seven days. That is the
paper's one resolvable architecture gain, and it requires an oracle: an argument
for sequence models *conditional on good forcing*, not for the regime the
literature benchmarks. The L0 triple difference sits below each cell's minimum
detectable effect (0.006, 0.012 and 0.018 °C), so it is evidence of absence; at
L2 the same three-way contrast is underpowered, and we report that as a power
limit rather than as a null.

![Effect sizes across the study's axes.](figures/fig05_information_axes.pdf)

**Figure 5. What each axis of the study is worth, at seven days.** Station-first
paired effects in °C on a log axis, damped-residual tree throughout, with
whole-HUC2 cluster-bootstrap intervals; open markers are negative effects
plotted at their magnitude, and the tick marks the minimum detectable effect
where one is defined. Local observation, future weather, study design and the
estimator span three orders of magnitude. The axes are separate conditional
designs and are compared, never summed. Every row is post-outcome and
descriptive.

Post-outcome and descriptive, and excluded from the Abstract and Key Points on
that ground.

## 5. Discussion

### 5.1 Why these numbers are smaller than published gains

Our reported gains are far smaller than those commonly published for daily river
temperature, and the difference is one of design, not model quality.
Three choices account for it.

The first is the reference. Where a study quotes skill against persistence or
climatology, the number it reports is comparable to our +0.203 to +0.251 column,
not to our +0.038 to +0.168 column; on a variable with this much day-to-day
memory the two differ by up to a factor of 6.6 for the same fitted weights. We
suggest damped persistence as the minimum reference for this variable.

The second is the spatial partition, and here our design was not able to measure
what it set out to measure. Holding out whole regions moves the median distance
to the nearest training gauge from 60 to 263 km, but every arm retains the held
station's thermal history, and the measured geometry penalty (0.004–0.009 °C)
sits inside its own resampling noise (Section 4.5). We make no claim about the
size of the spatial effect on this cohort, in either direction. A random-site
split in a dense network measures something closer to interpolation than to
transfer.

The third is the key set and the preprocessing boundary. Scoring each model on
its own complete-case set, and fitting scaling, imputation, climatology, or
interval offsets on windows that include the evaluated interval, move error
downward without leaving a visible trace. We do not assert that the published
literature is affected; removing both channels here left the constrained deep
model behind a gradient-boosted tree at the 1- and 3-day leads and matched at
7 days.

### 5.2 What a benchmark of this geometry can carry

The evaluation design of this study is stronger than its inferential reach. Clustered inference over a gauge panel rests on resampling or sign-flipping whole spatial units, and its accuracy depends on having enough units, of comparable size, that the asymptotics they invoke are not a fiction. Fifteen HUC2 groups with an inverse-Herfindahl effective count of 9.54 and a largest group holding 21.7% of stations is not enough, and the sampling design was never exchangeable across regions. We therefore present the clustered intervals and p-values as approximate descriptive sensitivities and do not attach decision weight to them.

The usual practice would report a station-level interval or a paired test across sites, and nothing in such a paper would tell a reader that the effective number of clusters is under ten; the difference is disclosure, not the data and not the model.

### 5.3 What transfers

This study is narrow by construction: a common-key, clustered comparative
benchmark of point predictors at gauged sites, not an attempt to replace
process-based thermal models, river-network graph models, or differentiable
hybrid formulations
([Jia et al., 2021](https://doi.org/10.1137/1.9781611976700.69);
[Rahmani et al., 2023](https://doi.org/10.1029/2023WR034420);
[Zwart et al., 2023](https://doi.org/10.3389/frwa.2023.1184992)). Every arm
consumes the target site's observed water temperature through the issue date
(Section 3.4), so the paper is silent on prediction at locations with no thermal
record.

What is portable is not the architecture but three controls: an identical key
registry across models, calibration and preprocessing statistics fit strictly
backwards in time, and spatial partitions by whole hydrologic region rather than
by site. On this cohort the reference choice moved a reported number by more
than the differences between architectures did; whether that ordering holds
across the published literature is a question this study motivates rather than
settles.

---

## 6. Limitations

**Scope.** The design is a descriptive benchmark on a fixed cohort:
comparisons are station-first and approximate (15 HUC2 clusters; effective
cluster count 9.54; largest share 21.7%; non-probability sampling), and no
interval, p-value, or ranking claim is decision evidence. Section 4.7 withholds a station's own thermal record from the model, which is
not the same as predicting at a location that has never been instrumented: the
cohort, the climatology and the evaluation keys all still come from gauged
sites. Genuinely ungauged prediction is not quantified here, and no
operational-replay claim is made. Throughout, "causal" describes time ordering only, never
causal inference.

**Cohort, measurement, and design.** The cohort trades a modest discharge
channel (removal costs +0.042 °C at 1 day; SI19) against wide spatial
coverage; 950 of 1,465 candidates were excluded for missing joint flow.
Point-scale meteorology is used at station coordinates rather than basin
integrals. The air2stream-style hybrid is an unofficial empirical variant of
the published formulation (Toffolon and Piccolroaz, 2015) — not the official
code, not a validated reproduction — so no process-side claim is attached to
its scores.

**Reference construction.** The damped anchor is a fitted object and the
seven-day headline moves by a factor of 2.5 across defensible constructions of
it (Section 4.1). Every number stated against damped persistence is therefore
conditional on the anchor specified in Section 3.1, which is primary because
it was fixed before the held-out window was opened, not because it is the
strongest available.

**Threshold and archive scope.** The ±1 °C algebraic bound, event-threshold
quantiles, and reportability thresholds (100 paired keys) are
development-selected; the q90 event threshold is a statistical tail diagnostic
with no biological, ecological, or regulatory interpretation.
Development-period spatial analyses and robustness probes are archived in SI19.

## 7. Conclusions

We evaluated daily river water-temperature prediction at 120 stable U.S. gauges
across 15 hydrologic regions, scoring every model on one common set of
station/date/horizon keys at 1-, 3-, and 7-day leads, with all statistics
fitted strictly backwards in time and a held-out 2021–2023 test window.

Three findings survive. First, the reference model governs the reported gain:
the deep predictor's seven-day station-median skill is +0.250 against
persistence but +0.038 against damped persistence. Second, model ranking does
not favor architectural constraint: a gradient-boosted tree with site identity
has the lowest station-median RMSE at 1- and 3-day leads (0.589 and 1.304 °C),
and the deep model's 7-day point estimate (1.694 versus 1.735 °C) is not
separated from the tree at the cluster level, though that row is underpowered
rather than null. Third, the spatial-partition question the design was built to ask is
unresolvable in the regime it was posed in and resolvable just outside it. With
the held station's own thermal history retained, holding out whole regions
rather than random sites changes station-median RMSE by less than the five-seed
resampling spread. Withhold that history and the same contrast separates
(Section 4.7), which is why the geometry of a split cannot be assessed
independently of how much local information the model keeps.

Every model here is issue-time-only. A retrospective analysis (Section 4.8)
bounds what perfect future meteorology could add at 0.13 °C at one day and
0.54–0.63 °C at three and seven days. A placebo sealed before the outcome
existed removes 91–100% of that value: the gain is event-scale weather
information, not seasonal phase.

A published skill score is not interpretable without its reference model, its
spatial partition, and its information set. The comparisons are descriptive for
this fixed, availability-selected cohort: clustered intervals and p-values are
approximate sensitivities, not decision evidence.

## 8. Open Research

**Data availability.** The derived daily panel (`panel_usgs_120v2.parquet`), the
stable station registry (`station_registry_v1.csv`), their manifests, and the
test-window acquisition record are deposited as one versioned dataset at
`[DATA DOI TO BE MINTED]` under `[DATA LICENCE TO BE ASSIGNED]`. Neither can be
assigned before the byte-level rights review of Section 6 completes, because a
derived product does not inherit the most permissive of its upstream terms; the
deposit is described here as planned and specified, not as existing.

**Primary observational sources.** Water temperature and discharge come from USGS
NWIS ([U.S. Geological Survey, 1994](https://doi.org/10.5066/F7P55KJN)),
meteorology from Daymet V4 R1
([Thornton et al., 2022](https://doi.org/10.3334/ORNLDAAC/2129)), and wind speed
from gridMET ([Abatzoglou, 2013](https://doi.org/10.1002/joc.3413)), each under
its own terms.

**Material that cannot currently be redistributed.** Where redistribution terms
are unresolved, the deposit carries the request specification and a SHA-256
manifest in place of the bytes; exclusions are enumerated in SI16.

**Software availability.** The analysis software, the per-stage manifests, and
the Python 3.12 dependency lock with package hashes are archived at
`[SOFTWARE DOI TO BE MINTED]`. The MIT licence on the source code covers neither
the observational data nor the archived provider responses.

**Reproduction.** Sections 4.1–4.3 reproduce from the archived panel, bundles and
pinned environment; Section 4.4 from the archived test-window acquisition record.
Bit-level equality is expected for the tree models and agreement to a documented
tolerance for the neural members; tolerances and digests are in the Supporting
Information. Reproduction has not yet been carried out by an operator
independent of the authors on an independent host, and this statement is not a
claim that it has.

## Acknowledgments

[FUNDING TO BE COMPLETED — each funder with its award number, or an explicit
statement that the work received no external funding.]

[COMPUTATIONAL RESOURCES TO BE COMPLETED — the facility or facilities on which
the model suite was fitted.]

We acknowledge the U.S. Geological Survey for the National Water Information
System, the ORNL Distributed Active Archive Center for Daymet V4 R1, and the
Northwest Knowledge Network for gridMET. Acknowledgment of a data provider is not
an endorsement of this analysis by that provider.

**Competing interests.** [COMPETING INTERESTS TO BE COMPLETED — a declaration is
required from every author, including the explicit statement that none exists if
that is the case.]

**Author contributions.** [CREDIT ROLES TO BE COMPLETED — assign each author to
the applicable CRediT roles: conceptualization, methodology, software,
validation, formal analysis, investigation, resources, data curation, writing —
original draft, writing — review and editing, visualization, supervision, project
administration, funding acquisition. The signed intake form is
`docs/FAIR_SUBMISSION_READINESS_AND_TEMPLATES.md` §2.]

## Supporting Information

Supporting Information contains the cohort and registry description with the
candidate-rejection ledger (SI01); the issue-time information boundary and the
product-compatibility bridge (SI02); model equations and unit conventions
(SI03); the analysis protocol, redesign chronology and decision log (SI04); the
comparison set and its fields (SI05–SI06); all-model scores on the exact common
keys, including the fitted air2stream-style hybrid reference (SI07);
probability and reliability schemas with the measured degenerate-interval table
(SI08); architecture and information-matched controls with seed and budget
provenance (SI09); the temporal coverage audit (SI10); the matched spatial
factorial and the cluster geometry at HUC2 through HUC8 (SI11); outcome quality
control and qualifier evidence (SI12); the history-dependent external arm
(SI13); missingness and failure cases (SI14); reproduction hashes, commands and
environment parity fields (SI15); and the rights and data dictionary (SI16).

Figures S1–S3 describe the cohort geometry, the temporal roles and issue-time
boundary, and the model and calibration dataflow; Figures S4–S8 expand Section
4.4 and the hydrologic-state mechanism of Section 4.6; Figure S9 is a development-period conformal-calibration sensitivity,
labelled as such in the figure, and must not be compared numerically with any
held-out-window figure.


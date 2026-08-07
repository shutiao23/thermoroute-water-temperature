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

- A seven-day skill of +0.250 against persistence falls to +0.038 against a
  damped-persistence reference at 116 reportable gauged sites.
- A gradient-boosted tree with site identity is most accurate at 1- and 3-day
  leads; the deep predictor is most accurate at 7 days.
- Reported skill shifts more across reference models and spatial partitions
  than across the model classes examined here.
## Abstract

Daily river water temperature is strongly persistent and strongly seasonal, and
machine-learning studies routinely report large skill gains for it, usually
measured against naive persistence or climatology on random data splits with
each model scored on its own set of predictable days. Such designs cannot
separate learned river behavior from three cheaper explanations: seasonal
damping, favorable key selection, and spatial interpolation between neighboring
gauges. Here we score a constrained deep predictor against damped persistence,
gradient-boosted trees with and without site identity, a global LSTM, and an
information-matched plain causal convolutional network on one common registry
of 249,072 station/date/horizon keys at 120 U.S. gauges (116 reportable on the
test window) spanning 15 hydrologic regions, with all preprocessing fitted
strictly backwards in time and whole regions, not random sites, held out. On
the independent 2021–2023 window the deep model reports a seven-day
station-median skill of +0.250 against persistence but +0.038 against damped
persistence: of the 0.51 °C reduction in station-median RMSE from persistence
to the deep model, 0.43 °C is delivered by damping alone. A gradient-boosted
tree with site identity has the lowest station-median RMSE at one and three
days (0.589 and 1.304 °C) and is edged out at seven days (1.735 versus
1.694 °C). Reported skill for this variable is therefore largely a statement
about the reference model and the spatial partition rather than about the
architecture.

**Plain Language Summary.** River temperature changes slowly from day to day, so
a forecast that repeats yesterday's reading is already fairly accurate, and one
that also nudges it toward the usual value for the time of year is better
still. We tested a river-temperature model at 120 U.S. gauges (116 reportable
on the held-out test window) and deliberately made the evaluation demanding:
every model was scored on exactly the same days and sites, none could see
information from after the moment it was asked to predict, and we held out
entire river regions rather than scattered gauges. Measured against a simple
seasonal adjustment of yesterday's reading, most of the advantage that a naive
comparison would report disappeared, and a standard tree-based method was more
accurate at the shortest forecast ranges. How well such a model performs
depends mostly on what it is compared against and on how the map is divided
rather than on details of the model itself.

**Keywords:** river water temperature; benchmark design; strong baselines;
spatial holdout; comparative evaluation; conformal prediction.

---

## 1. Introduction

Stream and river thermal regimes set dissolved-oxygen saturation, metabolic
rates, life-stage timing, and habitat suitability, and they respond to
atmospheric forcing, discharge, groundwater exchange, riparian shading, channel
geometry, and regulation
([Caissie, 2006](https://doi.org/10.1111/j.1365-2427.2006.01597.x)). Water
managers need daily-resolution temperature at gauged reaches to time releases,
anticipate thermal refuge loss, and interpret compliance monitoring, and the
supply of daily records from national monitoring networks has made the variable
one of the most heavily modeled targets in applied hydrological machine
learning.

The prevailing understanding in that literature is that deep sequence models have
substantially advanced daily river-temperature prediction. Nonlinear air–water
regressions ([Mohseni et al., 1998](https://doi.org/10.1029/98WR01877)) and
hybrid air-temperature and discharge formulations
([Toffolon and Piccolroaz, 2015](https://doi.org/10.1088/1748-9326/10/11/114011);
[Piccolroaz et al., 2016](https://doi.org/10.1002/hyp.10913)) were succeeded by
recurrent, multi-task, and physics-guided river-network architectures that report
large error reductions
([Feigl et al., 2021](https://doi.org/10.5194/hess-25-2951-2021);
[Rahmani, Lawson, et al., 2021](https://doi.org/10.1088/1748-9326/abd501);
[Jia et al., 2021](https://doi.org/10.1137/1.9781611976700.69);
[Sadler et al., 2022](https://doi.org/10.1029/2021WR030138);
[Zwart et al., 2023](https://doi.org/10.3389/frwa.2023.1184992);
[Barclay et al., 2023](https://doi.org/10.1029/2023WR035327)). The reported
gains are consistent enough that architectural sophistication is now the usual
explanation for accuracy in this problem. Graph-convolutional and
temporal-graph architectures propagate information along the river network and
so make spatial dependencies explicit, and the relationships they learn have
been interrogated with explainability methods for how they govern transfer to
unseen environmental conditions
([Sun et al., 2021](https://doi.org/10.1029/2021WR030394);
[Topp et al., 2023](https://doi.org/10.1029/2022WR033880)). That transfer is a
question of spatial sensitivity rather than of raw accuracy, and it is the one
this study isolates by holding out whole regions rather than neighbouring sites.

Those reported gains are, however, extraordinarily heterogeneous, and the
heterogeneity does not organize itself by architecture. Reviews of the field note
that published accuracies are not comparable across studies because reference
models, key sets, and data-splitting conventions differ
([Corona and Hogue, 2025](https://doi.org/10.5194/hess-29-2521-2025)). The
dispersion is larger than the architectural differences it is supposed to
explain, and it survives within single panels: on the data used here, one fixed
model and one fixed set of prediction days produce a seven-day skill of +0.251 or
of +0.038 depending only on which reference model is placed in the denominator.
An architectural account cannot produce that spread, because the architecture
does not change between the two numbers.

The identification gap is a property of the designs, not of the literature's
size. Three design choices are made almost universally, and each one absorbs an
alternative explanation into the reported number. First, skill is quoted against
naive persistence, climatology, or an air–water regression. Daily mean water
temperature has large thermal inertia, so relaxing the last observation toward a
seasonal climatology already removes most of the error a learned model can
remove; a reference that omits this hands the model a large part of its budget
before it learns anything. Second, models are scored on model-specific
complete-case sets, so a model can gain apparent accuracy by declining to predict
on hard days, and preprocessing statistics — scaling constants, imputation fills,
climatological references, event thresholds, interval offsets — are commonly
fitted on periods that include the evaluated interval, which transfers
information backwards in time without leaving a trace in a conventional split.
Third, spatial generalization is reported from random held-site splits. When a
held-out gauge's neighbours remain in training, the model reaches the held-out
regime through correlated forcing and spatially smooth representations; the
quantity measured is closer to interpolation than to transfer. None of these
choices is detectable from a reported RMSE, and each inflates it.

Here we hold the model and the data fixed and vary the three design choices
instead, so that their separate contributions to reported skill become
measurable. We assembled a panel of 657,480 site-days from 120 stable U.S.
Geological Survey site numbers spanning 2006–2020, 34 states, and 15 two-digit
hydrologic unit (HUC2) regions, tuned all models on data through 2020, and
evaluate them on one common set of station/date/horizon keys at 1-, 3-, and 7-day
leads, with a held-out 2021–2023 test window. Four questions organize the
study. How much of a reported gain survives a strong reference? Which model class
is actually most accurate on identical keys? How much of a reported spatial
transfer survives whole-region holdout? And how far do those answers vary across
regions, seasons, and the width of a calibrated interval? The predictor whose
skill we decompose, ThermoRoute, is described in full in Section 3.1; its
architecture is the object under test rather than the contribution.


---

## 2. Data and design

### 2.1 The frozen panel and the cohort it defines

The development record is a single derived panel,
`data_usgs/panel_usgs_120v2.parquet`, bound by a manifest
(`data_usgs/frozen_panel_v1.json`) to a station registry
(`data_usgs/station_registry_v1.csv`). Each of 120 sites carries one row for
every calendar day from 2006-01-01 through 2020-12-31, giving 657,480 rows with
no gaps in the date index; missing observations are explicit nulls rather than
absent rows. Station identity is the zero-padded USGS `site_no`.

The seven raw variables consumed by the models, in frozen order, are water
temperature (`WTEMP`), streamflow (`FLOW`), air temperature (`TEMP`),
precipitation (`PRCP`), a relative-humidity proxy (`RHMEAN`), Daymet
daylight-period mean incoming shortwave radiation in W m⁻² (`DH`), and wind speed
(`WDSP`). Two names are legacy and easily misread: `DH` is Daymet `srad`, the
incident shortwave flux averaged over the daylight period, not day length and not
a 24-hour mean; `RHMEAN` is a reproducible vapour-pressure proxy evaluated at the
tmax/tmin midpoint, not a direct daily-mean humidity observation. Gage height
(`WLEVEL`) is retained as raw evidence but is not a model input. Water
temperature and discharge come from USGS NWIS daily values
([U.S. Geological Survey, 2024](https://doi.org/10.5066/F7P55KJN)),
meteorological fields from Daymet V4
([Thornton et al., 2022](https://doi.org/10.3334/ORNLDAAC/2129)), and wind from
gridMET ([Abatzoglou, 2013](https://doi.org/10.1002/joc.3413)).

The cohort was not sampled at random from U.S. rivers, and nothing here treats it
as though it were. Candidate selection began from a discovery query over stream
sites in the states represented by the panel and rejected 1,345 of 1,465
candidate stations: 950 lacked joint water-temperature and discharge
availability, 376 failed a full-period coverage threshold, and 19 failed a
2019–2020 coverage threshold. The retained cohort is therefore
availability-enriched, and the 2019–2020 interval participated in its
construction, so that interval is development data in the strict sense and is used
for model tuning and diagnosis rather than as an independent test.

Three properties of the retained cohort bear directly on the analysis. Observed
water temperature is absent on about 15.8% of panel rows, with a maximum
contiguous gap of 2,404 days, and discharge on about 2.8%, with a maximum gap of
1,369 days; the meteorological fields are missing on about 0.07% of rows, in a
structured rather than random way, because all 120 sites lack the provider
calendar's final day in the leap years 2008, 2012, 2016, and 2020. Among retained
stations, 38 share a repeated hydrologic unit code with another retained station
and 19 have another retained station within 10 km, so station-level independence
is not tenable. Two stations contain 2,059 rows of signed negative discharge
(minimum −121 cfs), retained with their sign rather than clipped.


### 2.2 Temporal roles and the common key set

Temporal roles are fixed before any model is fitted, and a training sample is
admitted only if its issue date and every one of its target dates fall inside the
same partition.

| Role | Dates | Rows | Observed `WTEMP` | Permitted use |
|---|---:|---:|---:|---|
| Training | 2006–2015 | 438,240 | 341,646 | fit preprocessing and models |
| Validation | 2016–2017 | 87,720 | 84,074 | select model settings |
| Calibration | 2018 | 43,800 | 42,279 | conformal and Platt fits; final year of the 2006–2018 seasonal event reference |
| Development evaluation | 2019–2020 | 87,720 | 85,621 | model tuning and diagnosis |
| Held-out test | 2021–2023 | — | observed | comparative evaluation (Section 4.6) |

Models are tuned exclusively on data through 2020; the 2021–2023 window is held
out and used only for the comparative evaluation of Section 4.6. On the
development-evaluation partition, the intersection of admissible keys across all
primary models comprises 249,072 station/date/horizon keys, evenly distributed as
83,024 keys per lead.

![Study sites and evaluation design.](figures/fig01_study_design.pdf)

**Figure 1. Study sites and evaluation design.** (a) The 120 U.S. gauges
colored by the four deterministic whole-HUC2-region folds used for the
spatial-transfer experiment, on a CONUS outline; each fold holds out whole
regions rather than scattered sites. (b) Temporal partitions: 2006–2015
training, 2016–2017 validation, 2018 calibration, 2019–2020 development
evaluation, and 2021–2023 independent held-out test. (c) The three evaluation
tasks: known-site forecasting (site identity used), random held-site transfer,
and whole-region gauged transfer (site identity removed; local water-
temperature history remains an allowed issue-time input in every task).
(d) Issue-time information boundary: predictors are dated no later than the
issue date, targets are observed at t + h, and all models are scored on the
identical common forecast-key registry, with preprocessing fitted on data
through 2020 only. Every development comparison in Sections 4.1–4.5 uses
exactly this key set. No model-specific complete-case set is permitted, so a
model cannot gain apparent accuracy by declining to predict on hard keys. The
held-out 2021–2023 evaluation uses the same admissibility rule applied to the
later window.

### 2.3 Evaluation-period inputs and the information boundary

The test interval is 2021-01-01 through 2023-12-31 for the same 120-site cohort,
and historical Daymet and gridMET covariates for that interval are retrieved and
archived. Meteorology is represented at each station coordinate and is not
aggregated over upstream catchments, which is a real limitation for large basins
(Section 6.2).

The primary information set uses provider values dated no later than each
historical issue date and consumes no horizon-specific future weather forecast.
This is a date-indexed retrospective hindcast, and not a re-execution of what a
forecaster could have run on the day (Section 6.1): the
as-issued provisional vintage of a gridded product cannot be reconstructed after
the fact, so archiving requests, responses, timestamps, and checksums freezes the
dataset actually evaluated without proving that identical values were available
operationally at the time. NWIS dates denote site-local finalized daily values
while Daymet and gridMET use provider-specific calendar-day definitions; no
subdaily day-boundary harmonization is claimed. A product-compatibility bridge
re-fetched 2018–2020 covariates with the parser used for the test-window
acquisition and records `PASS_EXACT_PRODUCT_BRIDGE` on the exact site/date
registry.

Daily mean water temperature (00010, °C), discharge (00060, cfs), and
raw-only gage height (00065, ft) for 2021–2023 are retrieved by direct
requests to the USGS NWIS waterservices daily-values REST service
(`https://waterservices.usgs.gov/nwis/dv/`), with every request and response
byte-pair recorded, checksummed, and content-addressed by a `SnapshotStore`
cache, preserving qualifiers, timestamps, and content hashes.
Where two or more finite series exist for a parameter and date, the cell is marked
missing with an explicit conflict code rather than averaged or selected. The
primary analysis uses every finite parsed value regardless of approval qualifier,
so qualifiers can never select or remove a station, model, date, or forecast key;
a separately labeled sensitivity retains target keys only when the target
qualifier has the exact token set `{A}`. Test-window daily means are outcomes
only; they are not read by model-selection, feature-selection, threshold-selection,
calibration, or station-inclusion code, all of which is fixed on data through
2020.

A pooled-preprocessing sensitivity re-fits the model suite on the same 120-site
registry with station-agnostic (pooled) transforms in place of the per-station
transforms used by the primary arm, so that preprocessing aggregation is the only
difference between the two. It is not a site-disjoint cohort: every site in the
sensitivity arm is already in the development registry, and each target site
still supplies its own observed history (Section 3.4).


---

## 3. Methods

### 3.1 The predictor

ThermoRoute predicts at issue time *t*, station *i*, and lead *h* by adding a
bounded learned correction to a damped-persistence anchor. One model is fitted
for all three leads jointly: the horizon set (1, 3, 7 days) is an input dimension
and every head emits a lead-indexed vector in a single forward pass, so no lead
has its own weights.

**Anchor.** A seasonal climatology `c` and a per-station decay `φ_i` give

> **(1)**  $A_{i,t+h} = c_{i,t+h} + \phi_i^h\,(y_{i,t} - c_{i,t})$

where `y_{i,t}` is the last observed daily mean water temperature. Each `φ_i` is
estimated by no-intercept least squares on consecutive-day anomaly pairs from the
training period only, requires at least 30 pairs, and is clipped to [0, 0.999].
This object, fitted identically, is also the damped-persistence reference.

**Learned relaxation proposal.** A station-, flow-, and season-conditioned
relaxation rate replaces the fixed decay in a separate proposal path:

> **(2)**  $\kappa_{i,t} = \sigma(b + b_i + \beta_q\, q_{i,t} + \beta_s\, s_t)$, clipped to $[10^{-3}, 0.999]$
>
> **(3)**  $P_{i,t+h} = c_{i,t+h} + e_{i,t} + (1 - \kappa_{i,t})^h\,(a_{i,t} - e_{i,t})$

with `a_{i,t} = y_{i,t} − c_{i,t}` the current anomaly, `e_{i,t}` a learned
equilibrium anomaly formed from the standardized forcings and a station term,
`q` the standardized log-discharge, and `s_t` a two-component season encoding.
The bias `b` is initialized at −2.94, so the proposal begins near damped
persistence. The fitted relaxation rate is a statistical quantity only: it is not
a heat-transfer coefficient, and no physical interpretation of its value is
offered anywhere in this paper.

**Router.** A sparse, horizon-conditioned router scores all 7 variables at lags 0
to 14 — 105 (variable, lag) pairs — with a scaled dot product between a
horizon-specific query and a per-pair key, normalized by sparsemax
([Martins and Astudillo, 2016](https://proceedings.mlr.press/v48/martins16.html)):

> **(4)**  $w_h = \operatorname{sparsemax}(\langle q_h + \mathrm{ctx},\, k_{v,\ell}\rangle / \sqrt{d})$, over $v \in 1\ldots 7$, $\ell \in 0\ldots 14$

The router is an allocation mechanism inside the predictor, not a map of
connected reaches; it receives no verified graph or topology input.

**Temporal encoder.** A strictly left-looking temporal convolutional network
([Bai et al., 2018](https://arxiv.org/abs/1803.01271)) of `B = 2` residual blocks
with kernel `k = 3`, dilations 1 and 2, and width 40 encodes recent history. Left
padding of `(k − 1)·2^{b}` makes the output at position *t* a function of
positions ≤ *t* by construction. The receptive field is

> **(5)**  $R = 1 + (k - 1)(2^B - 1) = 7$ steps

so, together with the router's oldest usable value at lag 14, no input older than
lag 14 can affect the output. The sequence builder supplies a 32-day tensor, but
that is a construction buffer, not an effective-memory claim. Throughout this
paper "causal" describes time ordering only and never causal inference.

**Regime mixture.** Three experts are combined by a soft gate
([Shazeer et al., 2017](https://arxiv.org/abs/1701.06538)):

> **(6)**  $r = \sum_{k=1}^{3} \pi_k\, E_k(\cdot)$, $\pi = \operatorname{softmax}(g(\cdot))$

**Bounded residual.** The learned point correction is squashed and rescaled, so
its deviation from the anchor is algebraically bounded:

> **(7)**  $\hat{y}_{i,t+h} = A_{i,t+h} + \delta\,\tanh(z_{i,t+h}/\delta)$, $\delta = 1.0\,{}^\circ$C

giving $|\hat{y} - A| < \delta$ identically. The bound is relative to the named anchor: it
limits the point output's deviation from damped persistence and bounds neither
absolute error, nor event-tail error, nor interval width, nor behavior after a
distribution shift. Because the 2019–2020 partition was inspected during
development, `δ = 1.0 °C` is a development-selected algebraic point bound rather
than a value chosen on the held-out test window.

**Loss.** With `y` the observed target, `A` the anchor, quantile levels
τ ∈ {0.05, 0.50, 0.95}, `Q_τ` the corresponding quantile head, and `ρ_τ` the
pinball loss,

> **(8)**  $L = \mathrm{MSE}(y, \hat{y}) + \sum_\tau \rho_\tau(y, Q_\tau) + \lambda_e\, \mathrm{BCE}(\mathrm{exceedance}) + \lambda_c\, C + \lambda_r\, \|\hat{y} - A\|_1$

with `λ_e = 0.3`, `λ_r = 10⁻²`, `λ_c = 1.0`, and a temperature loss scale of 1.0.
The point and pinball terms carry unit weight. The term `C` penalizes quantile
non-monotonicity and is identically zero under the head parameterization used
here, so it contributes no gradient; it is retained for consistency with the
declared loss.

**Fitting.** All neural members are fitted with AdamW
(Kingma and Ba, 2015; Loshchilov and Hutter, 2019) at learning rate 2 × 10⁻³,
weight decay 10⁻⁴, batch size 1,536, gradient-norm clipping at 1.0, a maximum of
80 epochs with early stopping at patience 12 on station-macro validation RMSE,
a plateau scheduler halving the rate at patience 4, and equal-station fixed-size
bootstrap sampling; five seeds (0–4) are fitted and averaged with equal weights.
The canonical configuration has **38,505 trainable parameters**. All members were
fitted on CPU only (Intel Xeon Gold 6430, one PyTorch intra-op and one inter-op
thread), and every seed stopped early, at epochs 18, 15, 14, 18, and 26. Training wall-clock time and inference throughput were not recorded in the run
records (Section 6.2).

![The prediction framework being evaluated.](figures/fig02_prediction_framework.pdf)

**Figure 2. The prediction framework being evaluated.** Issue-time history
(water temperature, discharge, point meteorology) feeds a damped-persistence
anchor and a left-looking temporal residual learner; the bounded correction is
added to the anchor to form the final prediction at each of the 1-, 3-, and
7-day horizons. No future weather or target information is used, and the
correction is expressed relative to the anchor. The full architecture,
including the router, mixture, quantile, event, and calibration heads, is in
Supporting Information Figure S3.

### 3.2 The reference set

The composition of the reference set is the most consequential methodological
choice in this study, because it is what the reported gain is a gain over. Each
member removes one alternative explanation for apparent skill.

**Persistence** (`ŷ_{t+h} = y_t`) quantifies the raw memory of the series.
**Damped persistence**, equation (1) fitted on the training period, absorbs the
two effects a learned model most easily rediscovers — inertia and seasonality —
and is the reference against which the primary comparisons are stated.
**Seasonal climatology** isolates the seasonal cycle alone. **LightGBM**
([Ke et al., 2017](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html))
is the principal learned reference and is run both as a global model receiving
stable site identity as a categorical feature and as a per-station variant that
isolates the value of pooling. A **global LSTM** with a station embedding
represents the deep sequence family that has produced the strongest recent
results for this variable
([Rahmani, Lawson, et al., 2021](https://doi.org/10.1088/1748-9326/abd501);
[Zwart et al., 2023](https://doi.org/10.3389/frwa.2023.1184992)). A **plain
causal temporal convolutional network** is the information-matched deep
reference: it receives the same outcome-free issue-time history, station
identity, climatology, frozen damped anchor, standardized environmental context,
and calendar and regime-gate tensors as the full model, is matched to within 2%
of its parameter count (38,346 against 38,505) and to the same optimiser-step
budget, and predicts an unrestricted residual around the same anchor while
omitting the relaxation proposal, router, mixture, and residual bound. Its
companion, a plain multilayer perceptron at 38,860 parameters, isolates the value
of sequence structure. Neither ever reads the target tensor, the target-date
field, or the disabled water-level channel.

The learned references are given a genuine chance to win. The global LSTM uses
the same development splits, loss, station-balanced sampling, epoch and patience
limits, history length, site identity, and five-seed budget as ThermoRoute, its
validation-only tuning grid may expose the same climatology,
damped-anchor, and season features, and it is eligible for the same calibration;
it lacks only the bounded-residual constraint and the lag router. LightGBM
selects among four candidate settings separately by lead on the 2016–2017
partition using station-macro RMSE alone; the LSTM selects among three
candidate architectures with seed 0 before fitting five members. Tuning budgets
are documented but *not* equalized across model classes, which is a real
asymmetry: any statement about LightGBM here is scoped to this four-candidate
procedure and this feature schema, not to gradient boosting in general. Seven
one-factor architecture controls (`DampedPriorOnly`,
`TR-noDynamicPrior`, `TR-fixedKappa`, `TR-noRouter`, `TR-noMoE`, `TR-noTCN`,
`TR-unbounded`) are fitted with the same five seeds and paired within seed on
identical keys before ensemble averaging; they are deletion and intervention
sensitivities and do not prove component necessity or identify a mechanism.
`DampedPriorOnly` disables both the dynamic prior and the residual model, so
its output is identically the damped-persistence anchor; it reproduces the
`DampedPersistence` baseline to within floating-point noise on every metric
and is therefore not reported as a separate row in the held-out tables.

An air2stream-style hybrid reference (Toffolon and Piccolroaz, 2015) is
fitted per station on the training record and scored on the held-out window
with the same keys as every other model (Section 4.6; Supporting
Information SI07).

### 3.3 Leakage control

Leakage control here is a set of mechanisms that can be checked against the code
and the artifacts, rather than a statement of intent.

*Issue-time separation of covariates.* Every predictor consumed by a primary
model carries a date no later than the issue date, and no horizon-specific future
weather field enters any primary model.

*Non-anticipating sequence encoding.* Equation (5) makes the information horizon
bounded and auditable by construction rather than by convention.

*Strictly backward-fitted statistics.* Standardization constants, imputation
fills, the seasonal climatology, the damped-persistence rate, the station-level
q90 event thresholds (2006–2015), the seasonal event reference (2006–2018), the
conformal offsets (2018), and the Platt calibrators (2018) are each fitted on
data strictly preceding the interval on which they are applied. This closes the
most common backward-information channel in hydrological benchmarking, which is
not the model but the preprocessing.

*Explicit admissibility.* A forecast key is admissible only when issue-date water
temperature is genuinely observed and a 32-day history can be constructed; other
history cells are filled with training-only seasonal medians and retain explicit
missingness masks. We record one honest gap: the issue-date auxiliary quantities
used by the relaxation proposal and the regime gate are computed from the same
training-only imputed panel but do not each carry a separate auxiliary-path mask,
so the output is not guaranteed invariant to the chosen fill values.

*Reproducibility check.* As a reproducibility check, a fresh interpreter
reloads every trained member and reproduces the validation, calibration, and
2019–2020 keys and values from the frozen inputs, independently recomputing the
thresholds and refitting the calibration parameters; a prediction file that is
merely self-consistent is rejected. The details of that replay are specified in
the Supporting Information.

### 3.4 Two spatial partitions

Random held-site splits are the usual way to report spatial generalization in
this literature, but they can produce optimistic transfer estimates: if a
held-out gauge's neighbours remain in training, the model reaches the held-out
site's thermal regime through correlated forcing, shared preprocessing
statistics, and spatially smooth learned representations. We therefore report
two arms with different spatial information properties and read the difference
between them as the measurement of interest.

The **random held-site warm-start** arm uses four folds in which held-site
histories still contribute to global panel preprocessing. This is the arm most
comparable to published random-split results, and precisely for that reason it is
not the arm we treat as the spatial test.

The **held-region gauged transfer** arm packs the 15 HUC2 groups into four folds
of [30, 30, 31, 29] stations and holds out *whole regions*, so no gauge from a
held-out region appears in training. Station-agnostic ThermoRoute and a global
LightGBM are each trained on the in-fold regions and forecast the held-out
region's stations, with all climatology, scaling, and damped-rate parameters
pooled from in-fold stations only. The mean distance from a held-out station to
its nearest training gauge is 289 km.

The label on this arm must be exact. Held-out sites still supply their own
observed water temperature through the issue date, both as the persistence anchor
and as the sequence history. This is **gauged** transfer to a region whose gauges
were not used in fitting; it is **not** prediction at a site with no
observational record, which is a different problem and a different evaluation
([Weierbach et al., 2022](https://doi.org/10.3390/w14071032);
[Rahmani, Shen, et al., 2021](https://doi.org/10.1002/hyp.14400)).

### 3.5 Conformal intervals

Predictive uncertainty in hydrological modeling has most often been represented
through Bayesian or likelihood-based treatments of parameter and input error
([Beven and Binley, 1992](https://doi.org/10.1002/hyp.3360060305);
[Kavetski et al., 2006a](https://doi.org/10.1029/2005WR004368),
[2006b](https://doi.org/10.1029/2005WR004376)). We use a distribution-free
calibration step instead, because the quantity required here is an empirical
coverage statement about a fixed set of forecast keys rather than a posterior
over model parameters. Intervals come from split conformalized quantile
regression
([Romano et al., 2019](https://papers.nips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html);
[Vovk et al., 2005](https://doi.org/10.1007/b106715)): the learned models emit a
separate mean-squared-error point head and three pinball-trained quantile heads,
members are averaged with equal weights, and the offset is fitted on the 2018
calibration year only. The construction is deliberately one-sided — the deployed
offset is `qhat_plus = max(raw_qhat, 0)` and the delivered interval is
`[q05 − qhat_plus, q95 + qhat_plus]` with the median unchanged — so calibration
can widen a nominal interval but never shrink it, and a non-finite, crossed, or
empty final interval fails closed before any evaluation label is read.

What this guarantees is worth stating precisely, because conformal prediction is
frequently over-claimed in applied work. Split conformal provides finite-sample
*marginal* coverage under exchangeability of calibration and evaluation
nonconformity scores. Here the calibration scores come from 2018 and the
evaluation scores from a later, disjoint period at the same sites, so
exchangeability fails by construction: there is temporal drift, and the scores
are dependent within station and within region. We therefore report achieved
coverage as an *empirical marginal diagnostic*, make no finite-sample guarantee,
and make no conditional-coverage claim of any kind. Two sensitivities probe that
boundary: a block-maximum variant that calibrates on the maximum score within
groups of consecutive retained calibration rows, following the standard practice
of resampling blocks rather than rows when the series is dependent
([Künsch, 1989](https://doi.org/10.1214/aos/1176347265)), and an idealized
delayed adaptive-conformal variant using each forecast's target date as a
feedback-arrival proxy. Neither replays real feedback availability, because the
inputs contain no verified observation-publication timestamp, revision history,
or reporting latency. The equal-weight three-quantile pinball summary is not
CRPS; the two are distinct members of the family of proper scoring rules
([Gneiting and Raftery, 2007](https://doi.org/10.1198/016214506000001437)). An
event head separately reports exceedance of each station's 2006–2015 q90 water
temperature, calibrated by one Platt map per lead fitted on 2018 only; that
threshold is a statistical tail diagnostic local to each station, with no
biological, ecological, or regulatory interpretation.

### 3.6 Estimand, metrics, and comparison set

**Two differently signed quantities are reported, and they are never combined.**
The paired effect is

> **(9)**  $\Delta\mathrm{RMSE} = \mathrm{RMSE}(\mathrm{candidate}) - \mathrm{RMSE}(\mathrm{reference})$, in ${}^\circ$C, **negative** favors the candidate

and the skill score is

> **(10)**  $\mathrm{skill} = 1 - \mathrm{RMSE}(\mathrm{candidate})/\mathrm{RMSE}(\mathrm{reference})$, dimensionless, **positive** favors the candidate

Equation (9) is the formal estimand and carries a °C unit everywhere it appears;
equation (10) is a ratio and appears as a bare signed number. Every value in
Section 4 is labeled with which of the two it is. A positive ΔRMSE and a
positive skill score mean opposite things, and no figure axis, table column, or
sentence in this paper mixes them.

The sampling unit is the station. For each lead, unweighted RMSE is computed on
the common daily keys separately for each reportable station, and the primary
effect is the unweighted median across stations of equation (9) with ThermoRoute
as candidate; a station/lead cell is reportable only with at least 100 valid
paired targets, so daily rows do not determine between-station weight. On the
held-out 2021–2023 window we additionally report mean absolute error (MAE) and
mean error (bias), and skill against both persistence and seasonal climatology,
per model and lead, both pooled and per station. We report RMSE and paired RMSE
differences rather than an efficiency-type criterion such as the
Nash–Sutcliffe efficiency or its decomposition-based successors
([Nash and Sutcliffe, 1970](<https://doi.org/10.1016/0022-1694(70)90255-6>);
[Gupta et al., 2009](https://doi.org/10.1016/j.jhydrol.2009.08.003)), because
those criteria normalize by a station-specific variance, which would make a
paired between-model difference on identical keys harder to read.

The comparison set contains five head-to-head rows:

| # | Comparison | Lead | Reference margin |
|---|---|---:|---|
| 1 | ThermoRoute vs. damped persistence | 1 d | 0.00 °C |
| 2 | ThermoRoute vs. damped persistence | 3 d | 0.00 °C |
| 3 | ThermoRoute vs. damped persistence | 7 d | 0.00 °C |
| 4 | ThermoRoute vs. LightGBM | 3 d | +0.05 °C numerical ceiling |
| 5 | ThermoRoute vs. LightGBM | 7 d | +0.05 °C numerical ceiling |

The +0.05 °C ceiling is a stated numerical reference for the LightGBM comparison
rather than a decision threshold: it is not derived from sensor precision,
biological response, a water-quality standard, or an elicited stakeholder
utility, and it carries no ecological or regulatory importance.

Two uncertainty procedures accompany each row, both clustered at HUC2: a
one-sided p-value from exact enumeration of all 2^K whole-cluster sign vectors,
applying one common sign to every station effect within a cluster, and a
10,000-draw whole-HUC2 cluster bootstrap percentile interval for the median
station effect, with Holm adjustment
([Holm, 1979](https://www.jstor.org/stable/4615733)) over these five p-values.
We do not use the standard tests of equal predictive accuracy for forecast series
([Diebold and Mariano, 1995](https://doi.org/10.1080/07350015.1995.10524599);
[Harvey et al., 1997](<https://doi.org/10.1016/S0169-2070(96)00719-4>)), because
their asymptotics are stated for a single loss-differential series and do not
address dependence across the 120 gauges, which is the dominant dependence here.

These clustered procedures rest on assumptions the cohort only partially meets,
and the limit is stated rather than hidden. The cohort was assembled by an
availability filter, not by probability sampling of HUC2 regions, so
exchangeable region sampling is not established; with 15 HUC2 groups (an
inverse-Herfindahl effective count of about 9.54 and a largest group holding
21.7% of stations) the clustered intervals and p-values are approximate
descriptive sensitivities, not decision evidence. No comparison here is written
as a superiority, non-inferiority, equivalence, or parity statement, and none
generalizes to a national population. Equal-HUC aggregation, leave-one-HUC2
influence, and year-by-season temporal-stability audits are reported as
held-out sensitivities; HUC2 is a coarse administrative grouping, not an
independent river-network component.

---

## 4. Results

Sections 4.1–4.5 report development-period (2019–2020) benchmark diagnostics.
The 2019–2020 partition participated in cohort construction and informed model
selection, so these values are development diagnostics rather than an independent
test; they are reported because they are the basis on which the model suite and
the comparison set were fixed, and they anticipate the held-out evaluation.
Unless stated otherwise, all development values are five-seed ensemble means on
the 249,072 common keys, with station-median RMSE in °C. Section 4.6 reports the
held-out 2021–2023 comparative evaluation; its pooled metric cells are filled
from `outputs/conventional/holdout_metrics_2021_2023.csv` and are not
station-medians, so they are not directly differenced against the
development-period values above.

### 4.1 The reference model, not the architecture, sets the reported gain

How much of a reported gain survives a strong reference? Most of it does not. On
identical keys, the same fixed model reports a seven-day median station skill of
+0.251 against persistence and +0.038 against damped persistence — a factor of
6.6 between two numbers that differ only in the denominator of equation (10).

| Lead | Persistence | Damped persistence | LightGBM | LSTM | ThermoRoute |
|---:|---:|---:|---:|---:|---:|
| 1 d | 0.803 | 0.774 | 0.578 | 0.662 | 0.631 |
| 3 d | 1.576 | 1.406 | 1.280 | 1.323 | 1.291 |
| 7 d | 2.217 | 1.738 | 1.649 | 1.679 | 1.657 |

*Development-period (2019–2020) station-median RMSE, °C.* Read down the seven-day row: station-median RMSE falls
from 2.217 °C under persistence to 1.738 °C under damped persistence to 1.657 °C
under ThermoRoute. Of the 0.560 °C total reduction, 0.479 °C is delivered by
relaxing yesterday's observation toward a fitted seasonal climatology, and
0.081 °C by everything the deep model adds. The same arithmetic in skill units
gives +0.203, +0.187, and +0.251 against persistence and +0.168, +0.076, and
+0.038 against damped persistence at 1, 3, and 7 days, all dimensionless.

The gap widens with lead, which is the diagnostic signature of damping rather
than of learned river behavior: the longer the lead, the more of the error a
seasonal relaxation removes on its own, and the less remains for a learned model
to remove. A study reporting only the persistence column would describe this
model as gaining a quarter of the seven-day error budget. A study reporting both
columns describes the same fitted weights as gaining four percent.

Reported against the strong reference, the development-period paired effects of
equation (9) are −0.130, −0.104, and −0.062 °C at 1, 3, and 7 days, with
ThermoRoute favored at 0.86, 0.90, and 0.93 of the 120 stations, and
whole-HUC2 cluster-bootstrap intervals of [−0.177, −0.081], [−0.126, −0.083], and
[−0.071, −0.052] °C. Those intervals carry the 15-cluster structure discussed in
Section 3.6 and are approximate. The held-out 2021–2023 evaluation
(Section 4.6) reports the same comparison on the test window. If the reference
set determines the size of the reported gain, the next question is whether it
also determines which model wins.

![Decomposition of reported skill on the held-out window.](figures/fig03_skill_decomposition.pdf)

**Figure 3. Decomposition of reported skill on the held-out 2021–2023 window
(116 reportable stations, common forecast-key registry).** (a) Station-median
RMSE at 1-, 3-, and 7-day leads for persistence, damped persistence, LightGBM,
LSTM, and ThermoRoute; lower is better. (b) Seven-day error-budget
decomposition: of the 0.51 °C station-median RMSE reduction from persistence
to ThermoRoute, damping delivers 0.49 °C and the learned model 0.07 °C
(Table 4.11). (c) Paired station-level ΔRMSE (ThermoRoute minus reference),
unweighted medians with IQR across 116 stations; negative values favor
ThermoRoute; clustered intervals, win rates, and Holm-adjusted p-values are in
Table 4.6.

### 4.2 The most accurate model on this panel is a gradient-boosted tree

It does. On the development window the tree ensemble has the lowest
station-median RMSE at all three leads: 0.578, 1.280, and 1.649 °C against
ThermoRoute's 0.631, 1.291, and 1.657 °C. The paired station-level differences
(`ThermoRoute − LightGBM`, equation 9) are +0.046 °C at 1 day, +0.008 °C at 3
days, and −0.005 °C at 7 days, with ThermoRoute win rates of 0.00, 0.40, and 0.60
across the 120 stations. The one-day win rate is 0.00: not one of the 120
stations favors the constrained architecture at the shortest lead.

We report this directly. On this panel, on identical keys, a well-tuned
gradient-boosted tree with site identity is more accurate than the constrained
deep architecture, decisively at 1 day and negligibly at 3 and 7, where the
cluster-bootstrap intervals for the paired difference are [+0.001, +0.020] and
[−0.010, +0.010] °C. The global LSTM is intermediate at 0.662, 1.323, and 1.679 °C.
The architecture's remaining practical argument is not accuracy but the
algebraic bound of equation (7), which held on 100.00% of audited rows, with a
maximum absolute correction of 1.0000 °C against the configured 1 °C limit and
the derived station-by-lead RMSE inequality holding in all 360 cells — a
property of the construction rather than a fitted outcome.


### 4.3 An information-matched plain convolutional network reproduces the architecture

If a tree is more accurate, is any of the architecture's structure doing work?
Very little of it is. Against the information-matched plain causal temporal
convolutional network of Section 3.2 — same inputs, same anchor, same optimiser
budget, 38,346 against 38,505 parameters — the full architecture's paired median
station effect, computed within seed on identical keys, ranges across the five
seeds from −0.0003 to −0.0105 °C at 1 day, −0.0077 to −0.0179 °C at 3 days, and
−0.0035 to −0.0226 °C at 7 days. Every seed favors the full model; no seed by
more than 0.023 °C. Against the plain multilayer perceptron the same effects
range from −0.0360 to −0.0411 °C at 1 day, so sequence structure matters and the
components layered on top of it do not.

![Model-class and architecture effects under matched information.](figures/fig04_model_class_effects.pdf)

**Figure 4. Model-class and architecture effects under matched information
(held-out 2021–2023, 116 reportable stations).** (a) Median paired ΔRMSE
relative to damped persistence for the learned models at 1-, 3-, and 7-day
leads; negative values favor the model. (b) One-day median ΔRMSE of the
one-factor architecture controls relative to damped persistence; negative
values favor the variant. All models share identical keys, anchor, information
set, and calibration policy; the full 16-model station-median grid on the
held-out window is in Supporting Information Figure S4.

The one-factor deletions agree. Removing the temporal encoder is the largest
single effect, moving 1-day station-median RMSE from 0.631 to 0.679 °C and 7-day
from 1.657 to 1.675 °C.

 Removing the router (0.634), the mixture (0.634), the
dynamic prior (0.628), the fixed-relaxation constraint (0.627), or the residual
bound (0.630) moves the 1-day figure by at most 0.004 °C in either direction.
Replacing everything by the damped anchor alone gives 0.774, 1.406, and 1.738 °C.
These are five-seed deletion and intervention sensitivities on paired keys; they
do not prove component necessity or a capacity-matched attribution. Read
together with Section 4.2, they locate this model's accuracy in the causal
sequence encoder and in the anchor it corrects, not in the routing, mixture, or
bounding machinery.

*Comparability note.* The controls in this subsection are per-seed values from
the development-controls run and are compared only with each other and with the
matched full-architecture arm from the same run (0.6452, 1.3047, 1.6682 °C).
They are not the five-seed ensemble means of Sections 4.1–4.2 and must not be
differenced against them.

### 4.4 Whole-region holdout removes a third of the transfer skill a random split reports

How much of a reported spatial transfer survives when the map is partitioned
honestly? Between a quarter and a third of it does not.

| Arm | Design | 1 d | 3 d | 7 d |
|---|---|---:|---:|---:|
| Temporal development | seen stations, 2019–2020 | +0.203 | +0.187 | +0.251 |
| Random held-site warm start | 4 folds; held-site history in preprocessing | +0.178 | +0.172 | +0.241 |
| Held-region gauged transfer | leave-HUC2-region-out | +0.155 | +0.116 | +0.155 |

*Development-period median station skill versus persistence, dimensionless.* Against damped
persistence the same three arms give +0.168 / +0.076 / +0.038 (temporal),
+0.145 / +0.061 / +0.030 (random held-site), and +0.147 / +0.086 / +0.075
(held-region). The three-day skill against persistence falls from +0.187 to
+0.172 when sites are held out at random and to +0.116 when whole regions are,
and the seven-day figure falls from +0.251 to +0.241 to +0.155. The gap between
the second and third rows is the direct measure of how much a random spatial
split flatters a model on this panel: the random split retains 92% of the
temporal-arm three-day skill, the regional split 62%; Section 4.7 re-tests this
finding on the independent 2021–2023 window with a matched design.

The mechanism is neighbourhood, and the geometry supports that reading. Under
whole-region holdout the mean distance from a held-out station to its nearest
training gauge is 289 km, whereas in the intact cohort 19 of 120 stations have
another retained station within 10 km and 38 share a hydrologic unit code with
another retained station. A random held-site split on such a panel measures
something much closer to interpolation between instrumented neighbours than to
transfer to an uninstrumented region.

The ranking of models is unchanged by the harder partition, and the margins
widen. In the held-region arm the global tree ensemble again attains lower
station-median RMSE than ThermoRoute (0.652 against 0.676 °C at 1 day, 1.391
against 1.428 at 3 days, 1.786 against 1.860 at 7 days), with paired median
differences of +0.031 [+0.023, +0.040], +0.038 [+0.029, +0.050], and +0.067
[+0.049, +0.080] °C and ThermoRoute win rates near 0.16. The global LSTM, with
its station embedding disabled in this arm, gives 0.679, 1.445, and 1.876 °C.
Since the spatial partition moves the answer this much, the next question is
whether the remaining skill is spatially and seasonally uniform.



### 4.5 Skill is regionally uniform, and interval coverage is bought with width

It is uniform, and this is the one place where the constrained model's behavior
is unremarkable in a useful way. Region-weighted skill against persistence — the
mean of the 15 per-HUC2 medians — is +0.204, +0.189, and +0.253 at 1, 3, and 7
days, essentially unchanged from the pooled medians of +0.203, +0.187, and
+0.251, so no single region carries the headline. Per-region medians against
persistence at 1 day range from +0.131 (HUC2:10) to +0.285 (HUC2:18) and at 7
days from +0.217 (HUC2:06) to +0.290 (HUC2:09). Stratifying by drainage area into
three groups of roughly 39 stations gives 1-day medians of +0.190, +0.214, and
+0.231 from small to large. Against damped persistence the seven-day per-region
medians compress to a range of +0.012 to +0.077 — the same message as Section
4.1, restated spatially: what varies across regions is small once the seasonal
reference is doing its work.

Interval behavior tells the complementary story about what calibration costs.
Split-conformal intervals on the 249,072 development keys attain an empirical
marginal coverage of 0.909 against a nominal 0.90, with a mean width of 3.87 °C
and a mean interval score of 4.93; per lead, coverage is 0.905, 0.910, and 0.912
at widths of 2.01, 4.22, and 5.38 °C, and 0.918 at a width of 3.50 °C on the
warm-season q90-exceedance tail. Block-calibrating the maximum nonconformity
score over seven-row blocks raises coverage to 0.981 but widens the interval to
5.74 °C — coverage bought with width, not sharpness; delayed adaptive-conformal
variants track nominal coverage (0.892–0.903) but produce unbounded widths in
several slices (details in SI09). None of these figures is a conditional-
coverage statement, and width and interval score are reported beside every
coverage figure so that adaptive calibration is not presented as free.

Input perturbations locate the model's dependence in the same two channels the
ablations pointed to. Gaussian sensor noise at 0.25 and 0.5 training standard
deviations degrades 1-day RMSE from 0.631 to 1.621 and 2.999 °C (+147% and
+360%); missing-forcing blocks of 3, 7, and 14 days cost about +0.13 °C at 1 day
and +0.04 °C at 7 days; air-temperature offsets of ±2 training standard
deviations cost +0.11 to +0.15 °C at 1 day; and multiplying discharge by 0.5 or 2
changes 1-day RMSE by at most +0.003 °C. On this panel the predictor is dominated
by the water-temperature and air-temperature channels and is nearly insensitive
to discharge. These are synthetic data-corruption probes, not climate
projections, physically coherent scenarios, or deployment-safety tests.




### 4.6 Held-out 2021–2023 evaluation

The held-out 2021–2023 evaluation applies the model suite and comparison set of
Section 3, fixed on data through 2020, to the test window 2021-01-01 through
2023-12-31. The metric cells below are filled from the long-form table
the long-form results table (columns `model`, `horizon`, `metric`, `value`,
`n`). RMSE, MAE, bias, and skill are unweighted station medians over the
reportable stations on the common held-out forecast keys (Section 3.6), and
`n` is the reportable station count.

**Table 4.6 — paired comparisons on the held-out window.** Station-level
ΔRMSE (°C, negative favors ThermoRoute), computed as the unweighted median
over reportable stations of RMSE_ThermoRoute − RMSE_reference, for the five
formal tests of Section 3.6 (the frozen five-test family). CI low and CI high
are the 2.5% and 97.5% percentiles of a cluster bootstrap that resamples whole
HUC2 regions; the win rate is the fraction of reportable stations where the
candidate has lower RMSE; p is the cluster sign-flip p-value, Holm-adjusted
over the five tests.

| # | Comparison | Lead | ΔRMSE (°C) | CI low | CI high | Win rate | p |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | ThermoRoute vs. damped persistence | 1 d | -0.129 | -0.199 | -0.090 | 0.90 | <0.001 |
| 2 | ThermoRoute vs. damped persistence | 3 d | -0.108 | -0.140 | -0.079 | 0.91 | <0.001 |
| 3 | ThermoRoute vs. damped persistence | 7 d | -0.069 | -0.088 | -0.057 | 0.95 | <0.001 |
| 4 | ThermoRoute vs. LightGBM | 3 d | +0.015 | +0.012 | +0.024 | 0.24 | 1.000 |
| 5 | ThermoRoute vs. LightGBM | 7 d | -0.009 | -0.015 | -0.000 | 0.59 | 0.023 |


**Tables 4.7–4.8 — held-out 2021–2023 metrics.** Station-median metrics per
model and lead over the reportable stations: RMSE, MAE, and bias in °C; skill
against persistence and damped persistence (equation 10, dimensionless,
positive favors the candidate); and the reportable station count *n* = 116.
Table 4.7 lists the six primary models and Table 4.8 the six one-factor
ablations of Section 3.2, on the same station set and denominators; the
ablations are deletion and intervention sensitivities and do not prove
component necessity. Skill is 1 − RMSE_model/RMSE_baseline with both RMSEs
taken as the unweighted station median, so the baselines are zero against
themselves by definition. These held-out values use the same estimator as the
development-period values of Sections 4.1–4.5.

**Table 4.7a. Accuracy (RMSE, MAE, bias) — primary models.** *n* = 116 reportable stations at every lead; the Air2stream row is scored on the same common keys (118 stations with a fitted hybrid). Column groups are lead times in days. Unweighted medians over reportable stations.

| Model | RMSE 1 | 3 | 7 | MAE 1 | 3 | 7 | bias 1 | 3 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 0.813 | 1.638 | 2.202 | 0.597 | 1.235 | 1.686 | -0.000 | -0.001 | -0.004 |
| Damped persist. | 0.789 | 1.454 | 1.773 | 0.591 | 1.100 | 1.363 | -0.029 | -0.076 | -0.135 |
| Climatology | 1.899 | 1.902 | 1.903 | 1.424 | 1.427 | 1.426 | -0.388 | -0.386 | -0.384 |
| LightGBM | 0.589 | 1.304 | 1.735 | 0.436 | 0.982 | 1.326 | -0.006 | -0.084 | -0.178 |
| LSTM | 0.663 | 1.358 | 1.712 | 0.485 | 1.024 | 1.305 | -0.018 | -0.061 | -0.137 |
| Air2stream | 0.719 | 1.478 | 1.825 | 0.525 | 1.066 | 1.317 | -0.034 | -0.155 | -0.243 |
| ThermoRoute | 0.640 | 1.337 | 1.694 | 0.469 | 1.001 | 1.276 | `+0.010` | -0.028 | -0.096 |

**Table 4.7b. Skill against persistence and damped persistence — primary models.** Column groups are lead times in days. Skill is the unweighted median over reportable stations of the per-station
ratio 1 − RMSE_model/RMSE_reference (Section 3.6); positive favors the model.
A ratio of station-median RMSEs is a different quantity and is not used. Skill against seasonal climatology is omitted here because climatology is not a competitive reference at these leads; its RMSE is in Table 4.7a.

| Model | persist. 1 | 3 | 7 | damped 1 | 3 | 7 |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | +0.000 | +0.000 | +0.000 | -0.036 | -0.123 | -0.278 |
| Damped persistence | +0.035 | +0.110 | +0.218 | +0.000 | +0.000 | +0.000 |
| Climatology | -1.383 | -0.144 | +0.181 | -1.455 | -0.268 | -0.043 |
| LightGBM | +0.261 | +0.195 | +0.248 | +0.237 | +0.081 | +0.030 |
| LSTM | +0.181 | +0.172 | +0.244 | +0.150 | +0.055 | +0.028 |
| Air2stream | +0.091 | +0.113 | +0.218 | +0.063 | -0.008 | -0.011 |
| ThermoRoute | +0.206 | +0.186 | +0.250 | +0.173 | +0.077 | +0.038 |
**Table 4.8a. Accuracy (RMSE, MAE, bias) — one-factor ablations.** Each row removes one component from ThermoRoute; the `TR-` prefix is dropped. *n* = 116 reportable stations at every lead. Column groups are lead times in days. Unweighted medians over reportable stations.

| Model | RMSE 1 | 3 | 7 | MAE 1 | 3 | 7 | bias 1 | 3 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed κ | 0.635 | 1.333 | 1.695 | 0.467 | 0.996 | 1.282 | `+0.011` | -0.029 | -0.094 |
| no dyn. prior | 0.634 | 1.327 | 1.702 | 0.466 | 0.989 | 1.283 | +0.024 | -0.013 | -0.083 |
| no MoE | 0.643 | 1.345 | 1.698 | 0.470 | 1.006 | 1.282 | `+0.000` | -0.033 | -0.093 |
| no router | 0.642 | 1.342 | 1.694 | 0.470 | 1.006 | 1.282 | -0.004 | -0.029 | -0.093 |
| no TCN | 0.667 | 1.357 | 1.733 | 0.486 | 1.011 | 1.300 | -0.017 | -0.052 | -0.125 |
| unbounded | 0.634 | 1.333 | 1.695 | 0.467 | 0.997 | 1.282 | `+0.007` | -0.030 | -0.095 |

**Table 4.8b. Skill against persistence and damped persistence — one-factor ablations.**

| Model | persist. 1 | 3 | 7 | damped 1 | 3 | 7 |
|---|---:|---:|---:|---:|---:|---:|
| fixed κ | +0.209 | +0.187 | +0.250 | +0.180 | +0.075 | +0.041 |
| no dyn. prior | +0.212 | +0.189 | +0.248 | +0.181 | +0.077 | +0.038 |
| no MoE | +0.199 | +0.182 | +0.250 | +0.169 | +0.074 | +0.036 |
| no router | +0.202 | +0.180 | +0.250 | +0.172 | +0.072 | +0.038 |
| no TCN | +0.176 | +0.167 | +0.240 | +0.141 | +0.058 | +0.030 |
| unbounded | +0.204 | +0.189 | +0.252 | +0.176 | +0.076 | +0.042 |
Empirical 90% coverage, mean interval width (°C), and Brier score for the
calibrated models on the held-out keys are reported in Supporting Information
Table S8 (the frozen CQR + Platt calibration of Section 3.4 is applied
identically to the held-out predictions). Coverage is 0.907–0.932 at the
nominal 90% level across models and leads — coverage bought with width, not
sharpness — and extended scoring (pinball, log loss, discrimination, ECE)
appears in the same table.





<!-- FIGURE_ANCHOR id=S10 state=POST role=first_citation source=paper/FIGURE_REDRAW_SPEC.md#figure-s10 -->

The development-period benchmark diagnostics of Sections 4.1–4.5 anticipate the
held-out evaluation. On the held-out window the one-factor ablations neither help nor hurt at
any lead: fixing the dynamic prior (fixed κ) or removing the bounded-residual
constraint (unbounded) yields station-median skill against damped persistence
of +0.181/+0.176 at one day against +0.173 for the full model, and never
improves any lead in Tables 4.8a–b. The dynamic prior and the
bounded-residual constraint therefore add nothing on the independent window;
this is precisely the class of negative result that a weak evaluation design
would hide.

Two of those findings have held-out counterparts in the
2021–2023 cells above: the reference model rather than the architecture sets the
reported gain — seven-day skill against persistence remains high while skill
against damped persistence collapses — and a gradient-boosted tree with site
identity has the lowest station-median RMSE at the one- and three-day leads on
identical held-out keys. The fitted air2stream-style hybrid
(Toffolon and Piccolroaz, 2015) confirms the reference ladder from the process
side: its station-median RMSE is 0.719 °C at one day — the best non-learned
result and a skill of +0.063 against damped persistence — but its 3- and 7-day
RMSEs (1.478 and 1.825 °C) fall behind damped persistence, so the hybrid's
short-range advantage does not survive the seasonal reference at longer leads.
The information-matched plain convolutional network is a
development-period-only analysis and is not re-tested on the held-out window.

### 4.7 Matched spatial-transfer experiment on 2021–2023

The whole-region finding of Section 4.4 is re-tested on the independent window
with a matched spatial design. A station-agnostic gradient-boosted tree (site
identity removed, frozen per-lead hyperparameters, one seed, point head) is
fitted on the in-fold stations' 2006–2017 rows under two spatial partitions: four
deterministic leave-HUC2-region folds and four balanced random folds. Each
held-out station is then scored on its own 2021–2023 common forecast keys, with
identical preprocessing in both arms, so the arm contrast is attributable to the
spatial partition alone (details in SI11).

![Spatial transfer under matched spatial holdouts.](figures/fig05_spatial_transfer.pdf)

**Figure 5. Spatial transfer under matched random-site and whole-region
holdouts on the independent 2021–2023 window.** (a) The four whole-HUC2-region
folds on a CONUS outline. (b) Station-median RMSE for the random-site and
whole-region arms at 1-, 3-, and 7-day leads (station-agnostic LightGBM, 118
scored sites, identical preprocessing and keys in both arms). (c) Paired
three-day error relative to damped persistence as a function of distance to
the nearest training gauge; each point is one reportable station, negative
values favor the learned model, and bin medians are marked with diamonds.
Median nearest-training-gauge distance is 60 km (random) versus 268 km
(whole-region). The three-day skill against damped
persistence falls from +0.053 under random held sites to +0.040 under whole
regions, and the median distance from a held-out station to its nearest
training gauge rises from 60 to 268 km — the same direction, and the same
neighbourhood mechanism, as the development-period finding of Section 4.4
(+0.187 to +0.116 against persistence). The ranking of the spatial partitions
is therefore reproduced on an independent later window with the same keys,
information policy, and model class.

| Arm | 1 d RMSE | 3 d RMSE | 7 d RMSE | 3 d skill | 7 d skill | nearest gauge, median km |
|---|---:|---:|---:|---:|---:|---:|
| Random held-site | 0.628 | 1.390 | 1.738 | +0.053 | +0.028 | 60 |
| Whole-region holdout | 0.646 | 1.417 | 1.749 | +0.040 | +0.020 | 268 |

*Station-agnostic LightGBM, unweighted station medians over the 118 scored
sites, held-out 2021–2023 keys; skill is the median of per-station skill
against damped persistence.* The random arm retains 75% of the paired
station comparisons at three days (region-minus-random paired RMSE +0.013 °C)
and 67% at seven days (+0.009 °C).



### 4.8 Hydrologic conditions governing incremental skill

The remaining learned gain is concentrated in specific hydrologic states, which
locates the mechanism that survives the strong reference. Station thermal
memory, estimated as the e-folding half-life of an AR(1) fit on training-period
damped-seasonal anomalies, has a station median of 7.2 days (interquartile range
5.2–11.2), and the station-level memory/learned decomposition of the error
budget is consistent with it: at seven days damping delivers 0.49 °C of the
0.51 °C median reduction from persistence to ThermoRoute (Table 4.11), and the
station correlation between half-life and learned gain is −0.49 at one day and
−0.30 at three days: longer-memory stations leave less for the learned model to
add.

**Table 4.11 — Station-median error-budget decomposition.** Unweighted medians
over the 116 reportable stations of per-station RMSE differences (°C) on the
held-out keys: memory gain is persistence minus damped persistence; learned
gain is damped persistence minus ThermoRoute.

| Lead | RMSE, persistence | RMSE, damped | RMSE, ThermoRoute | Memory gain | Learned gain |
|---:|---:|---:|---:|---:|---:|
| 1 d | 0.813 | 0.789 | 0.640 | 0.028 | 0.129 |
| 3 d | 1.638 | 1.454 | 1.337 | 0.170 | 0.106 |
| 7 d | 2.202 | 1.773 | 1.694 | 0.491 | 0.069 |

*Memory and learned gains are unweighted medians over stations of the
per-station RMSE differences; they do not sum to the difference of the
station-median RMSEs exactly (Section 3.6).*

| State (7-day keys) | ΔRMSE, ThermoRoute − damped (°C) | keys | stations |
|---|---:|---:|---:|
| All keys | −0.073 | 118,682 | 116 |
| Largest 10% daily warming | −0.331 | 11,871 | 111 |
| Warmest 10% of days | −0.084 | 12,038 | 76 |
| High-flow 10% | −0.094 | 11,856 | 53 |
| Low-flow 10% | −0.062 | 11,864 | 25 |
| High air–water disequilibrium 10% | −0.036 | 11,869 | 112 |
| Coldest 10% of days | +0.014 | 11,952 | 92 |

*Pooled paired RMSE difference over held-out keys within each state; negative
favors ThermoRoute.* The learned model's incremental value is largest exactly
where damped persistence is weakest: on the fastest-warming days its advantage
reaches −0.33 to −0.41 °C at every lead, and it is consistently negative on
warm, high-flow, and strongly forced days. On the coldest decile at seven days
the anchor alone is marginally better (+0.014 °C), the only stratum in which
the learned correction does not pay for itself. Reported average skill therefore
understates a state-dependent benefit that a strong reference obscures but does
not remove.

![Hydrologic conditions governing incremental skill.](figures/fig06_hydrologic_mechanism.pdf)

**Figure 6. Hydrologic conditions governing incremental skill (held-out
2021–2023).** (a) Distribution of the station thermal half-life estimated from
a training-period AR(1) on damped-seasonal anomalies (median 7.2 d).
(b) One-day learned gain over damped persistence versus half-life, with the
least-squares trend (r = −0.49); longer-memory stations leave less to learn.
(c) Seven-day pooled ΔRMSE (ThermoRoute minus damped persistence) within
hydrologic states; negative values favor ThermoRoute. (d) The same ΔRMSE by
season and lead. The learned model's incremental value concentrates in rapid
warming, warm, and high-flow states and is absent on the coldest decile.

---

## 5. Discussion

### 5.1 Why these numbers are smaller than published gains

Our reported gains are far smaller than those commonly published for daily river
temperature, and the difference is a difference of design rather than of model
quality. Three specific choices account for it, and each can be read off Section
4 directly.

The first is the reference. Where a study quotes skill against persistence or
climatology, the number it reports is comparable to our +0.203 to +0.251 column,
not to our +0.038 to +0.168 column. On a variable with this much day-to-day
memory the two differ by up to a factor of 6.6 for the same fitted weights, and
the discrepancy grows with lead, so studies at longer leads are the ones most
affected. We would expect much of the between-study dispersion noted by
[Corona and Hogue (2025)](https://doi.org/10.5194/hess-29-2521-2025) to be
absorbed by re-scoring published models against a fitted damped anchor rather
than against persistence, and we suggest damped persistence as the minimum
reference for this variable.

The second is the spatial partition. A model evaluated on randomly held-out
gauges in a dense network is being asked to interpolate between instrumented
neighbours; the same model evaluated on whole held-out regions, 289 km from the
nearest training gauge, retains 62% rather than 92% of its three-day skill. Where
published transfer results use random site splits, they are not measuring the
quantity their framing implies, and the difference is roughly a third of the
reported skill on this panel. This is a scale effect rather than a modeling
disagreement: at the spacing of a national monitoring network, random site
holdout and regional holdout are different experiments.

The third is the key set and the preprocessing boundary. Scoring each model on
its own complete-case set and fitting scaling, imputation, climatology, event
thresholds, or interval offsets on windows that include the evaluated interval
both move error downward without leaving a visible trace. We cannot quantify how
much of the published literature is affected, and we do not assert that it is;
what we can say is that removing both channels here left the constrained deep
model behind a gradient-boosted tree at every lead.

We are more cautious about the mechanistic reading of Section 4.3. That an
information-matched plain causal convolutional network reproduces the full
architecture to within 0.023 °C, and that deleting the router, the mixture, the
dynamic prior, or the residual bound moves 1-day error by at most 0.004 °C,
suggests that the accuracy available to a point-scale statistical predictor on
this panel is largely exhausted by a causal sequence encoder over issue-time
history plus a seasonal anchor. It does not identify why. Capacity matching is
not search-budget matching, deletion is not attribution, and the controls are
diagnostic throughout.

### 5.2 What a benchmark of this geometry can carry

The evaluation design of this study is stronger than its inferential reach, and
the two should not be confused. Clustered inference over a gauge panel rests on
resampling or sign-flipping whole spatial units, and the accuracy of those
procedures depends on having enough units, of comparable size, that the
asymptotics they invoke are not a fiction. Fifteen HUC2 groups with an
inverse-Herfindahl effective count of 9.54 and a largest group holding 21.7% of
stations is not enough, and the sampling design was never exchangeable across
regions. We therefore present the clustered intervals and p-values as approximate
descriptive sensitivities and do not attach decision weight to them.

The usual practice in applied water-temperature machine learning is to report a
station-level or row-level confidence interval, or a paired test across sites,
without addressing whether the number of independent spatial units can carry the
procedure. Under that practice a cohort like ours would produce intervals and
p-values that look authoritative, and nothing in the paper would tell a reader
that the effective number of clusters is under ten. The difference between that
paper and this one is not the data and not the model; it is whether the limit is
disclosed.

The design lesson is transferable. If a study intends region-clustered inference
over U.S. gauges, the cluster structure must be part of the *sampling design*,
not a post-hoc grouping of whatever cohort availability produced, and the
partition and its power should be fixed before the evaluation is read. Reaching
a genuinely large number of independent spatial units requires either a finer
partition with a defensible independence argument or a deliberately stratified
draw across many more regions; neither is achievable by relabelling an existing
cohort.

### 5.3 What transfers

This study is narrow by construction. It is a common-key, clustered comparative
benchmark of point predictors at gauged sites, and it is not an attempt to
replace process-based thermal models, river-network graph models, or
differentiable hybrid formulations
([Jia et al., 2021](https://doi.org/10.1137/1.9781611976700.69);
[Rahmani et al., 2023](https://doi.org/10.1029/2023WR034420);
[Zwart et al., 2023](https://doi.org/10.3389/frwa.2023.1184992)), nor
architectures that encode geographic context across regions and scales
([Luo et al., 2025](https://doi.org/10.1145/3748636.3762716)), each of which
represents structure — connectivity, upstream forcing, energy balance, spatial
context — that a point-scale statistical predictor does not. Every arm here,
including the whole-region and site-identifier-disjoint arms, consumes the target
site's observed water temperature through the issue date (Section 3.4), so the
paper is silent on prediction at locations with no thermal record.

What is portable is not the architecture but three controls that do not depend on
it: scoring every model on an identical key registry so that no model benefits
from selective prediction; fitting every preprocessing and calibration statistic
strictly backwards in time; and partitioning space by whole hydrologic regions
rather than by site. None requires unusual computational resources, and each
moved a reported number in this study by more than the difference between the
competing architectures. On this evidence, benchmark design is a larger source of
variation in reported river-temperature skill than model design is.

---

## 6. Limitations

### 6.1 Scope

This is a history-dependent, retrospective evaluation at gauged U.S. sites: target-site
water-temperature observations through each issue date are available inputs, so
the results do not establish ungauged prediction, and the Daymet and gridMET
predictors bound the analysis to CONUS. The evaluation is not an operational
replay with archived as-issued predictor vintages, and no physical river-network
routing or travel time is modeled. The daily-mean statistical thresholds and the
+0.05 °C comparison margin carry no ecological or regulatory meaning, and the
clustered tests are descriptive sensitivities rather than decision evidence
(Section 4.6). Architecture controls and predictor sensitivities are diagnostic
and do not identify causal mechanisms.

### 6.2 Cohort, measurement, and design

The 120-station sample is availability-enriched rather than randomly drawn, and
its coverage thresholds used the 2019–2020 interval, so that interval is not
independent of cohort construction; the panel is not nationally representative.
Original provider bytes for 2006–2020 were not retained, so development
reproduction begins from the committed derived artifact, and no NWIS qualifier
or sensor-history columns exist for that period. Meteorology is taken at the
station coordinate rather than integrated over the catchment, which is least
appropriate for the largest basins; two stations contain 2,059 signed negative
discharge rows whose semantics are unresolved; and evaluation is conditioned on
outcome observability. HUC2 is a coarse administrative grouping, not an
independent river-network component. Covariates are retrospectively acquired
latest-provider values rather than as-issued vintages, and model-selection
budgets were documented but not equalized across model classes. The
air2stream-style hybrid is an unofficial empirical variant of the published
formulation (Toffolon and Piccolroaz, 2015) rather than the official code or a
validated reproduction of it; the official model and other physical hybrid
families are absent from the comparisons.

### 6.3 Threshold and archive scope

An optional threshold analysis compares independently sourced observed daily
maxima, expressed as a strict seven-consecutive-day average, against a
site-specific standards registry with seasonal applicability assigned by each
window's ending date; any reported exceedance is descriptive observation, not a
model result or legal determination. The derived panel encodes values from three
providers whose terms differ, and a derived product does not inherit the most
permissive of its upstream terms; until every proposed deposit object carries a
recorded redistribution decision, the deposit is described as planned rather
than existing, and the retained provenance objects allow the data lineage to be
audited without a byte-level purge.

## 7. Conclusions

We evaluated daily river water-temperature prediction at 120 stable U.S. gauges
across 15 hydrologic regions, scoring every model on one common set of
station/date/horizon keys at 1-, 3-, and 7-day leads, with every preprocessing
and calibration statistic fitted strictly backwards in time and with a held-out
2021–2023 test window for the comparative evaluation.

Three findings survive on that independent window. First, the reference model
governs the reported gain: the deep predictor's seven-day station-median skill
is +0.250 against persistence but +0.038 against damped persistence, and of the
0.51 °C reduction in station-median RMSE from persistence to the deep model,
0.43 °C is delivered by seasonal damping alone. Second, model ranking does not
favor architectural constraint: a gradient-boosted tree with site identity has
the lowest station-median RMSE at 1- and 3-day leads (0.589 and 1.304 °C), the
deep model leads only at 7 days (1.694 versus 1.735 °C), and the one-factor
ablations (router, mixture, dynamic prior, residual bound) move one-day error by
at most 0.004 °C on the development window with no held-out gain. Third, the
spatial partition governs reported transfer: development-period three-day skill
against persistence falls from +0.187 under a random held-site split to +0.116
under whole-region holdout, and the matched spatial-transfer experiment on
2021–2023 reproduces that ranking (Section 4.7).

The common mechanism is that each design choice absorbs an alternative
explanation into the reported number: seasonal damping into a weak reference,
selective prediction into a model-specific key set, and spatial interpolation
into a random site split. For water-resource practice, reported skill is what a
manager compares when choosing a tool, so a published skill score is not
interpretable without its reference model and its spatial partition, and the
incremental value of architectural complexity over a strong statistical anchor
and a well-tuned tree should be re-measured under these controls before it is
relied upon.

The comparisons are descriptive for this fixed, availability-selected cohort:
the clustered intervals and p-values are approximate sensitivities (at most 15
HUC2 groups), not decision evidence, and no result generalizes to a national or
U.S.-river population.

## 8. Open Research

**Data availability.** The derived daily panel (`panel_usgs_120v2.parquet`), the stable station registry
(`station_registry_v1.csv`), the hydrologic-unit metadata snapshot, the
candidate-rejection ledger, and the panel manifest that binds them are
deposited as one versioned dataset at `[DATA DOI TO BE MINTED]` under
`[DATA LICENCE TO BE ASSIGNED]`. The test-window acquisition record — exact
request and response bytes, series identifiers, approval qualifiers, final URLs,
retrieval timestamps, and content hashes — is added to that deposit as a new
version.

Neither the DOI nor the data licence can be assigned before the byte-level rights
review described in Section 6.4 completes. The derived panel encodes values from
three providers whose terms differ, and a derived product does not inherit the
most permissive of its upstream terms. Until every object proposed for the
deposit carries a recorded, evidence-backed redistribution decision, the
deposit is described here as planned and specified, not as existing.

**Primary observational sources.** All observations are obtained from public providers and none is the property of
the authors. Daily-value water temperature and discharge come from the U.S.
Geological Survey National Water Information System
([U.S. Geological Survey, 2024](https://doi.org/10.5066/F7P55KJN); parameter
codes 00010, 00060, and 00065 with statistic code 00003). Air temperature,
precipitation, the relative-humidity proxy, and daylight-period mean incoming
shortwave radiation come from Daymet V4 R1 at ORNL DAAC
([Thornton et al., 2022](https://doi.org/10.3334/ORNLDAAC/2129)). Wind speed
comes from gridMET ([Abatzoglou, 2013](https://doi.org/10.1002/joc.3413)). Each
retains its provider's own terms, citation requirement, and access route
independently of this manuscript.

**Material that cannot currently be redistributed.** For any provider object whose redistribution terms are unresolved at deposit
time, the deposit carries, in place of the bytes: the product identifier and
version, the exact request specification, the retrieval code, and a SHA-256
manifest of the bytes as retrieved, so a third party can re-acquire identical
inputs and verify them without relying on redistribution. The classes currently
defaulting to exclusion, and the two excluded outright — the third-party AGU
LaTeX class, and three legacy CSV files retained in the repository archive from
an earlier three-station case study whose source and collection terms are
unrecorded — are enumerated in SI16; the legacy material is not evidence for any
statement in this manuscript. Original provider responses for the 2006–2020
development panel were not retained, so development reproduction begins from the
committed derived artifact; that limitation does not apply to the test-window
acquisition, for which exact bytes are archived.

**Software availability.** The analysis software, the model-suite registry, the per-stage manifests, the
environment probe, the verification entrypoints, the result renderer that
generates Section 4.6 from the test-window metrics, and the fully transitive
Python 3.12 dependency lock with package hashes are archived at
`[SOFTWARE DOI TO BE MINTED]`, corresponding to release
`[RELEASE TAG TO BE ASSIGNED]` of the source repository at
`[REPOSITORY URL TO BE CONFIRMED]`. The project's own source code is released
under the MIT licence (SPDX identifier `MIT`). That licence covers the source
code only: it does not license the observational data, the archived provider
responses, the third-party typesetting class, or any redistributed dependency
binary, each of which retains its own terms and is enumerated with those terms in
an archive notice file.

Analyses were run on Python 3.12 with NumPy, pandas, pyarrow, SciPy
([Virtanen et al., 2020](https://doi.org/10.1038/s41592-019-0686-2)),
scikit-learn (Pedregosa et al., 2011), LightGBM
([Ke et al., 2017](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)),
and PyTorch (Paszke et al., 2019). USGS NWIS daily values are retrieved by direct
requests to the USGS waterservices daily-values REST endpoint, with every request
and response byte-pair cached and content-addressed by the in-repository
`SnapshotStore` provenance store.

**Reproduction.** Sections 4.1–4.5 are reproducible from the archived panel, the archived model
bundles, and the pinned environment. Section 4.6 is reproducible from the
archived test-window acquisition record and the same bundles. Bit-level equality is expected
for the tree models and agreement to a documented tolerance for the neural
members; the tolerances, the verification commands, and the expected digests are
listed in the Supporting Information. Reproduction has not yet been carried out
by an operator independent of the authors on an independent host, and this
statement is not a claim that it has.


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

Supporting Information accompanies this manuscript and contains: the cohort and
registry description with the candidate-rejection ledger (SI01); the issue-time
information boundary and the product-compatibility bridge (SI02); model equations
and unit conventions (SI03); the analysis protocol and model-suite registry
(SI04); the comparison set and its fields (SI05–SI06); all-model scores on the
exact common keys, including the fitted air2stream-style hybrid reference with its parameter ranges and
official-variant caveat (SI07); probability and
reliability schemas and the full measured degenerate-interval table (SI08); architecture and
information-matched controls with seed and budget provenance (SI09); the temporal
coverage audit (SI10); spatial and leave-cluster sensitivities, including the
cluster geometry at HUC2, HUC4, HUC6, and HUC8 (SI11); outcome quality control
and qualifier evidence (SI12); the history-dependent external arm (SI13);
missingness and failure cases (SI14); reproduction hashes, commands, and
environment parity fields (SI15); and the rights and data dictionary (SI16).

Supporting figures accompany these sections: Figures S1–S3 describe the cohort
geometry, the temporal roles and issue-time information boundary, and the model
and calibration dataflow; Figures S4–S8 and S10 expand Section 4.6; and Figure
S9 is a development-period conformal-calibration sensitivity, labeled as such in
the figure itself, which must not be compared numerically with any held-out-window
figure.

Each figure carries evidence from exactly one period. Figures 1 and S1–S3 are
structural material, Figures 2 and 4 and Figures S4–S8 and S10 are
held-out-window, and Figure 3 and Figure S9 are development-period and say so
inside the figure. No value from one period is compared numerically with a value
from another.

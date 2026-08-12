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
  leads; the deep model matches it at 7 days.
- Learned skill beyond the seasonal reference is largest at issue-time low
  thermal anomaly and during rapid recent cooling.

## Abstract

Daily river water temperature is strongly persistent and strongly seasonal, and
machine-learning studies for it routinely quote skill against naive persistence
or climatology, on random splits, with each model scored on its own predictable days. Such designs cannot separate learned river behavior from
cheaper explanations: seasonal damping, favorable key selection, and spatial
interpolation between instrumented neighbors. We hold the data fixed and vary the evaluation design,
scoring a constrained deep predictor,
damped persistence, gradient-boosted trees with and without site identity, a
global LSTM, an information-matched plain causal convolutional network, and an
empirical thermal-recurrence comparator on one common registry of
station/date/horizon keys at 120 U.S. gauges (116 reportable on the 2021–2023
test window) across 15 hydrologic regions, with all preprocessing fitted strictly
backwards in time. On the independent window the deep model's
seven-day station-median skill is +0.250 against persistence but +0.038 against
damped persistence; the median station-level memory gain is 0.49 °C against a
learned gain of 0.07 °C, and the median fraction of the station-level error
reduction delivered by seasonal memory is 0.875. A gradient-boosted tree with
site identity has the lowest station-median RMSE at one and three days (0.589
and 1.304 °C); at seven days the two are not separated, on a comparison the
design is underpowered to make. Every model here is issue-time-only, and a
separate retrospective analysis bounds what perfect future weather could add at
several times the architecture effect. Within that regime, reported skill is
largely a statement about the reference model rather than the architecture.

**Plain Language Summary.** River temperature changes slowly from day to day, so
a forecast that repeats yesterday's reading is already fairly accurate, and one
that also nudges it toward the usual value for the time of year is better
still. We tested a river-temperature model at 120 U.S. gauges (116 reportable
on the held-out test window) and deliberately made the evaluation demanding:
every model was scored on exactly the same days and sites, none could see
information from after the moment it was asked to predict, and in a separate
experiment we held out entire river regions rather than scattered gauges.
Measured against a simple seasonal adjustment of yesterday's reading, most of
the advantage that a naive comparison would report disappeared, and a standard
tree-based method was more accurate at the shortest forecast ranges. How well
such a model performs depends mostly on what it is compared against and, to a
smaller and measurable degree, on how the map is divided rather than on details
of the model itself.

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
gains are consistent enough that architectural sophistication is a common
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

Hydrology has known for a long time that a reported score is a statement about
its benchmark. [Klemeš (1986)](https://doi.org/10.1080/02626668609491024) set
out the differential split-sample test precisely so that a model would be
evaluated under conditions it had not been fitted for;
[Seibert (2001)](https://doi.org/10.1002/hyp.446) and
[Schaefli and Gupta (2007)](https://doi.org/10.1002/hyp.6825) argued that
efficiency criteria are uninterpretable without an explicit benchmark, and
[Seibert et al. (2018)](https://doi.org/10.1002/hyp.11476) formalized upper and
lower benchmarks as the pair a reported score must sit between.
[Andréassian et al. (2009)](https://doi.org/10.5194/hess-13-1757-2009) proposed
standardized crash tests, [Pappenberger et al.
(2015)](https://doi.org/10.1016/j.jhydrol.2015.01.024) asked how a forecaster
can know that a forecast is better,
[Knoben et al. (2019)](https://doi.org/10.5194/hess-23-4323-2019) showed that a
familiar criterion carries an implicit benchmark that is not the one most
readers assume, and [Nearing et al.
(2018](https://doi.org/10.1175/JHM-D-17-0209.1),
[2021)](https://doi.org/10.1029/2020WR028091) placed benchmarking at the centre
of what machine learning can and cannot tell us about a hydrological system.
None of this is contested. What is missing for daily river temperature is not
the argument but the measurement: how large the benchmark effect actually is
for this variable, on identical prediction keys, with an estimand that pairs
models station by station rather than comparing marginal aggregates.

That measurement is what this paper supplies, and three design choices make it
possible to attribute the result. First, skill is usually quoted against naive
persistence, climatology, or an air–water regression. Daily mean water
temperature has large thermal inertia, so relaxing the last observation toward a
seasonal climatology already removes most of the error a learned model can
remove; a reference that omits this hands the model a large part of its budget
before it learns anything. Second, models are scored on model-specific
complete-case sets, so a model can gain apparent accuracy by declining to predict
on hard days, and preprocessing statistics — scaling constants, imputation fills,
climatological references, event thresholds, interval offsets — are commonly
fitted on periods that include the evaluated interval, which transfers
information backwards in time without leaving a trace in a conventional split
([Arsenault et al., 2018](https://doi.org/10.1016/j.jhydrol.2018.09.027)).
Third, spatial generalization is reported from random held-site splits. When a
held-out gauge's neighbours remain in training, the model reaches the held-out
regime through correlated forcing and spatially smooth representations; the
quantity measured is closer to interpolation than to transfer, which is the
distinction the predictions-in-ungauged-basins literature is organized around
([Hrachowitz et al., 2013](https://doi.org/10.1080/02626667.2013.803183);
[Kratzert et al., 2019](https://doi.org/10.1029/2019WR026065)). None of these
choices is detectable from a reported RMSE, and each can inflate it.

Here we hold the model and the data fixed and vary the design choices instead,
so that their separate contributions to reported skill become measurable. We
assembled a panel of 657,480 site-days from 120 stable U.S. Geological Survey
site numbers spanning 2006–2020, 34 states, and 15 two-digit hydrologic unit
(HUC2) regions, tuned all models on data through 2020, and evaluate them on one
common set of station/date/horizon keys at 1-, 3-, and 7-day leads, with a
held-out 2021–2023 test window. The cohort is availability-selected rather than
drawn, which bounds what any of it can generalize to
([Gupta et al., 2014](https://doi.org/10.5194/hess-18-463-2014)); Section 3.6
and Section 6 state that bound rather than working around it.

Five questions organize the study. How much of a reported gain survives a strong
reference? Which model class is actually most accurate on identical keys? How
much of a reported spatial transfer survives whole-region holdout, once local
adaptation is separated from spatial geometry? In which issue-time-identifiable
hydrologic states does a learned model still add skill beyond the strong
reference? And — since every model above sees only issue-time information — how
much predictability is withheld by that restriction rather than by the models,
which we bound in Section 4.8 with realized future meteorology. The predictor
whose skill we decompose, ThermoRoute, is described in full in Section 3.1; its
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
([U.S. Geological Survey, 1994](https://doi.org/10.5066/F7P55KJN)),
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
| Held-out test | 2021–2023 | — | observed | comparative evaluation (Section 4.4) |

Models are tuned exclusively on data through 2020; the 2021–2023 window is held
out and used only for the comparative evaluation of Section 4.4. On the
development-evaluation partition, the intersection of admissible keys across all
primary models comprises 249,072 station/date/horizon keys, evenly distributed as
83,024 keys per lead; the held-out window carries its own common-key registry of
the same form (Section 4.4). The 249,072 figure describes the development
registry only and is never used for a held-out claim.

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
through 2020 only. Every development comparison in Sections 4.1–4.3 uses
exactly this key set. No model-specific complete-case set is permitted, so a
model cannot gain apparent accuracy by declining to predict on hard keys. The
held-out 2021–2023 evaluation uses the same admissibility rule applied to the
later window.

### 2.3 Evaluation-period inputs and the information boundary

The test interval is 2021-01-01 through 2023-12-31 for the same 120-site cohort,
and historical Daymet and gridMET covariates for that interval are retrieved and
archived. Meteorology is represented at each station coordinate and is not
aggregated over upstream catchments, which is a real limitation for large basins
(Section 6).

The primary information set uses provider values dated no later than each
historical issue date and consumes no horizon-specific future weather forecast.
This is a date-indexed retrospective hindcast, and not a re-execution of what a
forecaster could have run on the day (Section 6): the
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

### 3.1 The predictor under test

ThermoRoute predicts at issue time *t*, station *i*, and lead *h* by adding a
bounded learned residual to a damped-persistence anchor,

> **(1)**  $A_{i,t+h} = c_{i,t+h} + \phi_i^h\,(y_{i,t} - c_{i,t})$

where $c$ is a seasonal climatology and $\phi_i$ a per-station decay fitted by
no-intercept least squares on training-period consecutive-day anomaly pairs
(at least 30 pairs, clipped to [0, 0.999]). The same object, fitted
identically, is the damped-persistence reference.

The residual is produced by a strictly left-looking temporal convolutional
encoder over a 14-lag window, with a horizon-conditioned variable/lag router,
a three-expert mixture, and a $\pm1\,{}^\circ$C algebraic bound on the
deviation from the anchor. One model serves all three leads; 38,505 trainable
parameters; five seeds averaged. The full specification, loss, and fitting
protocol are in Text S3. The architecture is the object under test, not the
contribution: Section 4.3 shows an information-matched plain causal network
reproduces it to within 0.023 °C, and Section 4.4 quantifies what the
information set, not the architecture, is worth.

![Common anchor–residual formulation under matched
information.](figures/fig02_model_concept.pdf)

**Figure 2. The formulation every compared model shares.** (a) Issue-time
information: local thermal state, meteorological history, and — only in the
forcing-regime arm of Section 4.8 — future meteorology (F0 absent, F3 the
realized oracle). (b) The damped-persistence anchor of equation (1), written
without station and lead indices for legibility, and the two regression
targets fitted against it. LightGBM, the most accurate model at 1 and 3 days,
is a *raw-target* model with site identity; ResidualLightGBM, the plain causal
TCN, and ThermoRoute predict a residual around the same anchor. Anchor, keys,
and information set are identical across both; only the target differs.
(c) The point forecast, and the ±1 °C bound that ThermoRoute alone imposes on
its residual — a bound on deviation from the anchor, not on error. (d) The
evaluation contract of Section 3.6: one common key registry, RMSE within each
station, paired station-level contrast, unweighted median across stations. The
full ThermoRoute dataflow is Figure S3; this figure is structural and carries
no evidence from either period.

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
with the same keys as every other model (Section 4.4; Supporting
Information SI07). It is an unofficial empirical thermal-recurrence variant
of the published formulation — not the official code and not a validated
reproduction — so no process-side claim is attached to its scores (Section 6).

### 3.3 Leakage control

Leakage control here is a set of mechanisms that can be checked against the code
and the artifacts, rather than a statement of intent.

*Issue-time separation of covariates.* Every predictor consumed by a primary
model carries a date no later than the issue date, and no horizon-specific future
weather field enters any primary model.

*Non-anticipating sequence encoding.* The strictly left-looking encoder of
Section 3.1 makes the information horizon bounded and auditable by
construction rather than by convention (Text S3).

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

### 3.4 Two spatial partitions, separated from local adaptation

Random held-site splits are the usual way to report spatial generalization in
this literature, but they can produce optimistic transfer estimates: if a
held-out gauge's neighbours remain in training, the model reaches the held-out
site's thermal regime through correlated forcing, shared preprocessing
statistics, and spatially smooth learned representations. We therefore report
two arms with different spatial information properties and read the difference
between them as the measurement of interest, in a 2×2 factorial that separates
the spatial partition from the local-adaptation policy (protocol v1; Section
4.5 and SI11).

The **random held-site warm-start** arm uses four balanced folds in which
held-site histories still contribute to global panel preprocessing. This is the
arm most comparable to published random-split results, and precisely for that
reason it is not the arm we treat as the spatial test.

The **held-region gauged transfer** arm packs the 15 HUC2 groups into four folds
of [30, 30, 31, 29] stations and holds out *whole regions*, so no gauge from a
held-out region appears in training. Station-agnostic ThermoRoute and a global
LightGBM are each trained on the in-fold regions and forecast the held-out
region's stations, with all climatology, scaling, and damped-rate parameters
pooled from in-fold stations only. The mean distance from a held-out station to
its nearest training gauge is 289 km.

The factorial contrasts two adaptation policies within each geometry. *Local*
adaptation lets each target station's own 2006–2015 history contribute to its
climatology, damped anchor, and imputation medians — the definition of gauged
transfer used throughout this paper. *Pooled* adaptation fits those statistics
over the in-fold training stations only, so a held region's long histories never
enter preprocessing; the contrast between the two isolates the value of local
thermal history separately from the spatial partition. On the independent
2021–2023 window the experiment is run with a station-agnostic gradient-boosted
tree (raw and damped-residual targets) over four region folds and five repeated
random-split seeds (Section 4.5); the development-period ThermoRoute arm of
the spatial analysis is archived in SI19.

The label on this arm must be exact. Held-out sites still supply their own
observed water temperature through the issue date, both as the persistence anchor
and as the sequence history. This is **gauged** transfer to a region whose gauges
were not used in fitting; it is **not** prediction at a site with no
observational record, which is a different problem and a different evaluation
([Weierbach et al., 2022](https://doi.org/10.3390/w14071032);
[Rahmani, Shen, et al., 2021](https://doi.org/10.1002/hyp.14400)).

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

Equation (2) is the formal estimand and carries a °C unit everywhere it appears;
equation (3) is a ratio and appears as a bare signed number. Every value in
Section 4 is labeled with which of the two it is. A positive ΔRMSE and a
positive skill score mean opposite things, and no figure axis, table column, or
sentence in this paper mixes them.

The sampling unit is the station. For each lead, unweighted RMSE is computed on
the common daily keys separately for each reportable station, and the primary
effect is the unweighted median across stations of equation (2) with ThermoRoute
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

How much of a reported gain survives a strong reference? Most of it does not. On
identical keys, the same fixed model reports a seven-day median station skill of
+0.251 against persistence and +0.038 against damped persistence — a factor of
6.6 between two numbers that differ only in the denominator of equation (3).

| Lead | Persistence | Damped persistence | LightGBM | LSTM | ThermoRoute |
|---:|---:|---:|---:|---:|---:|
| 1 d | 0.803 | 0.774 | 0.578 | 0.662 | 0.631 |
| 3 d | 1.576 | 1.406 | 1.280 | 1.323 | 1.291 |
| 7 d | 2.217 | 1.738 | 1.649 | 1.679 | 1.657 |

*Development-period (2019–2020) station-median RMSE, °C.* Read down the seven-day row: station-median RMSE falls
from 2.217 °C under persistence to 1.738 °C under damped persistence to 1.657 °C
under ThermoRoute. The median over stations of the per-station memory gain
(persistence minus damped persistence) is 0.479 °C and the median learned gain
(damped persistence minus ThermoRoute) is 0.081 °C; these are medians of
per-station differences and, like all such quantities, do not add exactly to the
difference of the station-median RMSEs (Section 3.6). The same arithmetic in
skill units gives +0.203, +0.187, and +0.251 against persistence and +0.168,
+0.076, and +0.038 against damped persistence at 1, 3, and 7 days, all
dimensionless.

**The reference is itself a fitted object, and its construction moves the
headline.** If the argument of this paper is that the reference governs the
reported gain, the reference we built cannot be exempt from that argument. We
therefore rebuilt the anchor under seven predeclared one-factor variants —
one, three and five harmonics; a nonparametric day-of-year mean; a fitting
window extended through 2018; a pooled rather than per-station decay; a
lead-specific decay fitted directly on lag-*h* anomaly pairs instead of
$\phi^h$; and a relaxed clip — and rescored each against the *published*
ThermoRoute predictions, so that every difference is attributable to the
reference alone. The rebuilt baseline reproduces the frozen
`DampedPersistence` station RMSEs to within 1e-8 °C, which is what makes the
comparison meaningful.

Across those variants the seven-day skill against damped persistence ranges
from +0.025 to +0.063, with the published +0.038 in the middle
(`outputs/final/anchor_sensitivity.parquet`). Two ends of that range are
instructive. A single-harmonic climatology — a weaker seasonal reference of
the kind a study might reasonably adopt — inflates the reported gain to
+0.063. A decay fitted directly at each lead, rather than extrapolated as
$\phi^h$, produces a *stronger* reference and cuts the reported gain to
+0.025, because $\phi^h$ is exact only for a pure AR(1) and daily water
temperature is not one. The clip never binds. The one- and three-day
headlines are far more stable (factors of 1.1 and 1.3 across the same
variants) than the seven-day one (a factor of 2.5), so the number this paper
quotes most often is the one most sensitive to how carefully the reference was
specified. That is the paper's own thesis applied to itself, and it sharpens
rather than weakens it: the better the anchor, the less the learned model adds.

The gap widens with lead, which is the diagnostic signature of damping rather
than of learned river behavior: the longer the lead, the more of the error a
seasonal relaxation removes on its own, and the less remains for a learned model
to remove. A study reporting only the persistence column would describe this
model as gaining a quarter of the seven-day error budget. A study reporting both
columns describes the same fitted weights as gaining four percent.

Reported against the strong reference, the development-period paired effects of
equation (2) are −0.130, −0.104, and −0.062 °C at 1, 3, and 7 days, with
ThermoRoute favored at 0.86, 0.90, and 0.93 of the 120 stations, and
whole-HUC2 cluster-bootstrap intervals of [−0.177, −0.081], [−0.126, −0.083], and
[−0.071, −0.052] °C. Those intervals carry the 15-cluster structure discussed in
Section 3.6 and are approximate. The held-out 2021–2023 evaluation
(Section 4.4) reports the same comparison on the test window. If the reference
set determines the size of the reported gain, the next question is whether it
also determines which model wins.

![Decomposition of reported skill on the held-out window.](figures/fig03_skill_decomposition.pdf)

**Figure 3. Decomposition of reported skill on the held-out 2021–2023 window
(116 reportable stations, common forecast-key registry).** (a) Station-median
RMSE at 1-, 3-, and 7-day leads for persistence, damped persistence, LightGBM,
LSTM, and ThermoRoute; lower is better. (b) Seven-day error-budget
decomposition: per-station memory gain (persistence minus damped persistence)
and learned gain (damped persistence minus ThermoRoute), summarized as the
equal-station mean waterfall (mean memory gain 0.456 °C plus mean learned gain
0.070 °C equals the mean total gain 0.527 °C exactly) with the median and IQR of
the per-station gains shown alongside (station-median memory gain 0.491 °C and
learned gain 0.069 °C at seven days). (c) Per-station memory fraction
(memory gain divided by total gain, computed only where the total gain is
positive; median 0.875 at seven days). (d) Paired station-level
ΔRMSE (ThermoRoute minus reference), unweighted medians with IQR across 116
stations; negative values favor ThermoRoute; clustered intervals, win rates,
and Holm-adjusted p-values are in Table 4.6.

### 4.2 The most accurate model on this panel is a gradient-boosted tree

It does. On the development window the tree ensemble has the lowest
station-median RMSE at all three leads: 0.578, 1.280, and 1.649 °C against
ThermoRoute's 0.631, 1.291, and 1.657 °C. The paired station-level differences
(`ThermoRoute − LightGBM`, equation 2) are +0.046 °C at 1 day, +0.008 °C at 3
days, and −0.005 °C at 7 days, with ThermoRoute win rates of 0.00, 0.40, and 0.60
across the 120 stations. The one-day win rate is 0.00: not one of the 120
stations favors the constrained architecture at the shortest lead.

We report this directly. On this panel, on identical keys, a well-tuned
gradient-boosted tree with site identity is more accurate than the constrained
deep architecture, decisively at 1 day and negligibly at 3 and 7, where the
cluster-bootstrap intervals for the paired difference are [+0.001, +0.020] and
[−0.010, +0.010] °C. The one-day paired difference is outside the frozen
five-test family; as an exploratory check it is +0.035 °C (ThermoRoute minus
LightGBM, station median, HUC2-cluster bootstrap CI [+0.031, +0.045],
sign-flip p = 1.0 under the one-sided family convention), with the tree
favored at 98% of the 116 reportable stations — the abstract's one-day claim
is therefore paired, cluster-aware, and consistent with the development-window
win rate of 0.00. The global LSTM is intermediate at 0.662, 1.323, and 1.679 °C.
The architecture's remaining practical argument is not accuracy but the
±1 °C algebraic bound of Section 3.1, which held on 100.00% of audited rows, with a
maximum absolute correction of 1.0000 °C against the configured 1 °C limit and
the derived station-by-lead RMSE inequality holding in all 360 cells — a
property of the construction rather than a fitted outcome.


### 4.3 An information-matched plain convolutional network reproduces the architecture

Very little of the architecture's structure is doing work. Against the
information-matched plain causal temporal convolutional network (same inputs,
same anchor, same optimiser budget, 38,346 against 38,505 parameters) the full
architecture's paired median station effect, computed within seed on identical
keys, ranges across the five seeds from −0.0003 to −0.0105 °C at 1 day and
−0.0035 to −0.0226 °C at 7 days — every seed favors the full model, none by
more than 0.023 °C. The one-factor deletions agree: removing the temporal
encoder is the largest single effect (1-day station-median RMSE 0.631 to
0.679 °C), while removing the router, mixture, dynamic prior, fixed-relaxation
constraint, or residual bound moves the 1-day figure by at most 0.004 °C
(held-out ablation values in SI09). Against the plain multilayer perceptron
the same effects are −0.036 to −0.041 °C at 1 day, so sequence structure
matters and the components layered on top of it do not. These are five-seed
deletion and intervention sensitivities on paired keys; they do not prove
component necessity.

### 4.4 Held-out 2021–2023 evaluation

The held-out 2021–2023 evaluation applies the model suite and comparison set of
Section 3, fixed on data through 2020, to the test window 2021-01-01 through
2023-12-31. Every number in this section is an unweighted station median over
the reportable stations on the common held-out key registry, derived from
`outputs/final/` by `scripts/final/build_results_authority.py` (pooled RMSE
appears only as a Supporting Information sensitivity, Section SI07). On this
window the deep model's station-median skill is +0.250 against persistence
and +0.038 against damped persistence, and the tree with site identity leads
at one and three days (station-median RMSE 0.589 and 1.304 °C; Table 4.7a) and
is not separated from it at seven days (1.694 versus 1.735 °C; paired
station-median ΔRMSE −0.009 °C, 95% interval [−0.017, +0.002], Table 4.6 row
5). That last row is not evidence of equivalence, and we do not read it as
such: the smallest effect this cohort's cluster structure can resolve at
α = 0.05 is larger than the effect observed, so the comparison is underpowered
rather than null (`outputs/final/family_power.parquet`). The three
damped-persistence rows of the family are in the opposite position — their
effects are three to seven times the smallest detectable effect — so the
family's power is not uniform and Table 4.6 should be read row by row. The
information-matched plain neural controls are scored on this window with the
same frozen weights and keys as every other model; they carry no calibration
(`NO_FROZEN_CALIBRATION`) and enter only the point comparisons.

**Table 4.6 — paired comparisons on the held-out window.** Station-level
ΔRMSE (°C, negative favors ThermoRoute), computed as the unweighted median
over reportable stations of RMSE_ThermoRoute − RMSE_reference, for the five
formal tests of Section 3.6 (the frozen five-test family). CI low and CI high
are the 2.5% and 97.5% percentiles of a cluster bootstrap that resamples whole
HUC2 regions; the win rate is the fraction of reportable stations where the
candidate has lower RMSE. p (sign flip) is the whole-HUC2-cluster sign-flip
p-value and Holm p is its Holm adjustment over the five formal tests (Section
3.6). Rows are generated by
`scripts/final/generate_manuscript_tables.py` from `outputs/final/`.

<!-- TABLE 4.6 (generated) -->

| # | Comparison | Lead | ΔRMSE (°C) | CI low | CI high | Win rate | Stations | p (sign flip) | Holm p |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | ThermoRoute vs. DampedPersistence | 1 d | -0.129 | -0.199 | -0.076 | 0.90 | 116 | 3.1e-05 | 1.5e-04 |
| 2 | ThermoRoute vs. DampedPersistence | 3 d | -0.108 | -0.143 | -0.073 | 0.91 | 116 | 3.1e-05 | 1.5e-04 |
| 3 | ThermoRoute vs. DampedPersistence | 7 d | -0.069 | -0.086 | -0.057 | 0.95 | 116 | 6.1e-05 | 1.8e-04 |
| 4 | ThermoRoute vs. LightGBM | 3 d | 0.015 | 0.010 | 0.025 | 0.24 | 116 | 1.0e+00 | 1.0e+00 |
| 5 | ThermoRoute vs. LightGBM | 7 d | -0.009 | -0.017 | 0.002 | 0.59 | 116 | 7.4e-02 | 1.5e-01 |


**Tables 4.7–4.8 — held-out 2021–2023 metrics.** Station-median metrics per
model and lead over the reportable stations (*n* = 116 at every lead): RMSE,
MAE, and bias in °C; skill against persistence and damped persistence
(equation 3, dimensionless, positive favors the candidate). The
information-matched plain controls are included on the same station set and
denominators; the ablations are deletion and intervention sensitivities and do
not prove component necessity. Skill is the unweighted median over reportable
stations of the per-station ratio 1 − RMSE_model/RMSE_baseline; a ratio of
station-median RMSEs is a different quantity and is not used. Rows are
generated by `scripts/final/generate_manuscript_tables.py` from
`outputs/final/`.

<!-- TABLE 4.7a (generated) -->

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











<!-- FIGURE_ANCHOR id=S10 state=POST role=first_citation source=paper/FIGURE_REDRAW_SPEC.md#figure-s10 -->

The development-period benchmark diagnostics of Sections 4.1–4.3 anticipate the
held-out evaluation. On the held-out window the one-factor ablations neither
help nor hurt systematically at any lead: fixing the dynamic prior (fixed κ) or
removing the bounded-residual constraint (unbounded) yields station-median
skill against damped persistence of +0.180/+0.176 at one day against +0.173 for
the full model, with small, non-monotonic differences that never exceed +0.009
skill units across the leads and tables (SI09). The dynamic prior and
the bounded-residual constraint therefore add no material point-accuracy gain
on the independent window; this is precisely the class of negative result that
a weak evaluation design would hide.

Two of those findings have held-out counterparts in the
2021–2023 cells above: the reference model rather than the architecture sets the
reported gain — seven-day skill against persistence remains high while skill
against damped persistence collapses — and a gradient-boosted tree with site
identity has the lowest station-median RMSE at the one- and three-day leads on
identical held-out keys. The fitted air2stream-style hybrid
(Toffolon and Piccolroaz, 2015) — an unofficial empirical thermal-recurrence
variant of the published formulation, not the official code and not a
validated reproduction — reaches a station-median RMSE of 0.719 °C at one day,
the best non-learned result on the common 116-station set, and falls behind
damped persistence at 3 and 7 days (1.459 and 1.825 °C); because the
implementation is not the published model, no process-side claim is attached to
it. The information-matched plain convolutional network and plain multilayer
perceptron are re-scored on this held-out window (Tables 4.7a and SI07): the
plain TCN's station-median RMSE is 0.646, 1.330, and 1.710 °C at 1, 3, and
7 days against ThermoRoute's 0.640, 1.337, and 1.694 °C — the full architecture
retains a small, consistent edge (median paired ΔRMSE with the plain TCN as
candidate and ThermoRoute as reference, +0.010, +0.011, and +0.015 °C; under
equation (2) positive values favor the reference, and the plain-TCN win
fractions are 0.30, 0.30, and 0.16) that is an order of
magnitude smaller than the reference-model effect, and the plain MLP is worse
still (0.690, 1.374, and 1.720 °C), so sequence structure matters while the
router, mixture, and bounding machinery add little beyond the causal encoder
and the anchor.

### 4.5 Matched spatial-transfer experiment on 2021–2023

The development-period whole-region finding (SI19) is re-tested on the independent window
with a matched 2×2 factorial design that separates spatial geometry from local
adaptation (protocol v1; details in SI11). Station-agnostic gradient-boosted
trees (site identity removed, frozen per-lead hyperparameters, deterministic
fits) are fitted on the in-fold stations' 2006–2017 rows under two geometries —
four deterministic leave-HUC2-region folds and four balanced random folds
repeated over five split seeds — and two adaptation policies: *local*, in which
each held station's own 2006–2015 history contributes to its climatology,
damped anchor and imputation medians (gauged transfer with local thermal
history), and *pooled*, in which those statistics are fitted over the in-fold
training stations only, so a held region's long histories never enter
preprocessing. Each held-out station is scored on its own 2021–2023 common
forecast keys; both a raw-target and a damped-anchor-residual variant are
fitted.

Under local adaptation the median nearest-training-gauge distance is 60 km for
the random arm and 263 km for the whole-region arm. The region-minus-random
paired station penalty (region cell minus the mean of the five random-split
cells, per station) is +0.006 °C at one day (IQR −0.000 to +0.014), +0.009 °C
at three days (IQR +0.001 to +0.026), and +0.007 °C at seven days (IQR +0.001
to +0.030) for the raw-target tree, with the random arm winning at 72–78% of
stations and the penalty positive in all five split seeds at every lead
(+0.004 to +0.010 °C per seed). The residual-target tree gives the same
picture (+0.006, +0.007, and +0.004 °C).

**This experiment does not resolve a geometry effect, and we do not report one.**
The penalty is the same size as its own resampling noise: the median per-site
standard deviation across the five split seeds is 0.004–0.007 °C, against a
penalty of 0.004–0.009 °C. Four whole-region folds provide four independent
regional observations, and the sign consistency across seeds is a statement
about those same four folds re-used, not about four additional samples of
space. Within this design there is likewise no gradient of the penalty against
distance or hydroclimatic novelty (correlations −0.18 to +0.17 across leads and
models). We therefore report the whole-region penalty as *below this design's
resolution* rather than as a small confirmed effect, and we draw no conclusion
from it about spatial generalization.

Two features of the design explain why it could not have resolved one. Every
held-out station in every cell still supplies its own observed water
temperature through the issue date, its own climatology, and its own damped
rate, so the anchor carries the prediction and the learned component that would
have to transfer across space is worth about 0.07 °C at seven days in the first
place (Section 4.1). A geometry effect must be found inside that 0.07 °C. The
informative contrast is therefore the one that removes local information
entirely, and the arm that attempts it in this run is defective: its pooled
preprocessing reused fold 0's statistics across folds 1–3, whose held-out
stations lie inside fold 0's training set (a recorded defect, DLOG-012). We
report no adaptation main effect from that arm. The corrected contrast is the
L0/L1/L2 information ladder of protocol v2, which fits preprocessing per fold
and adds levels with no local thermal record at all; it had not completed when
this manuscript was prepared, and no number from the defective arm is carried
into the Discussion or the Conclusions.

![Spatial transfer under matched random-site and whole-region
holdouts.](figures/fig05_spatial_transfer.pdf)

**Figure 4. Spatial transfer under matched random-site and whole-region
holdouts on the independent 2021–2023 window.** (a) The four whole-HUC2-region
folds on a CONUS outline. (b) Station-median RMSE for the four factorial cells
(geometry × adaptation) at 1-, 3-, and 7-day leads (station-agnostic LightGBM,
identical preprocessing and keys within each cell). (c) Paired region-minus-
random three-day error as a function of distance to the nearest training gauge;
each point is one station–seed cell, positive values favor the random arm, and
bin medians are marked with diamonds. (d) The same penalty against a
standardized hydroclimatic-novelty distance to the nearest training station.
The penalty is small (about 0.01 °C), consistently signed across five split
seeds and two model targets, and does not show a strong novelty gradient
(cell-level station-set caveats for the pooled arm are recorded in SI11).



### 4.6 Hydrologic conditions governing incremental skill

The remaining learned gain is concentrated in specific hydrologic states, which
locates the mechanism that survives the strong reference. Station thermal
memory is measured as the anomaly half-life of the official train-fitted
damped-persistence anchor, $t_{1/2} = \ln(0.5)/\ln(\phi_i)$ — an anomaly
half-life, not an e-folding time — with a station median of 6.9 days
(interquartile range 5.0–11.9). The station-level memory/learned decomposition
of the error budget is consistent with it: at seven days the median memory gain
is 0.49 °C and the median learned gain 0.07 °C (Figure 3b). Whole-HUC2 cluster
bootstrap intervals separate the two without overlap — [0.386, 0.570] °C for
the memory gain against [0.057, 0.086] °C for the learned gain — and the
leave-one-HUC2-out ranges ([0.477, 0.531] and [0.066, 0.074] °C) do not
approach each other either. Per-station memory fractions, computed over the
113 of 116 stations whose total gain is positive, have a median of 0.875
([0.861, 0.894]). The three stations the rule excludes are ones where the
learned model is worse than raw persistence at seven days, and at all three the
damped anchor is worse than raw persistence too, so the fraction would have no
interpretable denominator there. Values
are from `outputs/final/decomposition_intervals.parquet` and carry the same
15-cluster approximation as every other clustered interval here (Section 3.6). The station correlation between log half-life and learned
gain is −0.40 at one day and −0.14 at three days — longer-memory stations leave
less for the learned model to add at short leads — and reverses sign at seven
days (+0.21), where the remaining learned gain is small everywhere.

**Issue-time-identifiable states.** All thresholds are station-specific (or
station-month-specific) empirical quantiles fitted on the 2006–2015 training
period only; state metrics are computed station-first with a 30-key minimum per
station-state cell, then summarized by the median across stations (Table 4.12;
details in SI11). At seven days the learned model's median paired ΔRMSE over
damped persistence is −0.069 °C on all keys (Table 4.6, row 3). It is largest — in favor of the
learned model — when the current thermal anomaly is low (−0.23 °C, 105
stations) and during rapid recent cooling (−0.17 °C) and recent warming
(−0.09 °C); the gain is present under both low and high station-relative flow
(−0.07 and −0.06 °C) and across all seasons (−0.05 to −0.10 °C). No
issue-time-identifiable state reverses the ordering: the learned correction pays
for itself under every state examined here, but the magnitude of its advantage
is state-dependent rather than uniform.

**Retrospective outcome-conditioned diagnostics.** Stratifying by the target
outcome — an explicitly retrospective diagnostic, not an issue-time state —
shows where the anchor is weakest: on days that subsequently warm rapidly
(actual change at or above the station's training q90, positive direction only)
the median paired ΔRMSE reaches −0.30 °C at seven days, and on days that
subsequently cool rapidly (training q10 or below) it is −0.18 °C. On the
coldest target decile the anchor alone is marginally better (+0.03 °C), the
only stratum in which the learned correction does not pay for itself.

**Table 4.12 — Hydrologic states (7-day keys).** Median across
stations of the per-station paired ΔRMSE (ThermoRoute − damped persistence,
°C; negative favors ThermoRoute), station count, and median keys per
station-state cell. `issue_*` states are issue-time-identifiable with
station-specific training-period (2006–2015) quantiles (station-month
quantiles for flow); `actual_*` states are retrospective outcome-conditioned
diagnostics with station-specific signed thresholds. Rows are generated by
`scripts/final/generate_manuscript_tables.py`.

<!-- TABLE 4.12 (generated) -->

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

Reported average skill therefore understates a state-dependent benefit that a
strong reference obscures but does not remove. The largest values in Table 4.12
belong to the two `actual_*` strata, and those are conditioned on the realized
target change: they select the days on which a damped anchor must fail, so a
model that corrects the anchor will always appear strongest there, and a
manager cannot identify those days at issue time. We therefore read them as a
diagnostic of where the anchor is weak rather than as an operational benefit,
and the issue-time statement — largest under low thermal anomaly (−0.229 °C)
and during rapid recent cooling (−0.170 °C) — is the one carried into the Key
Points.

![Hydrologic conditions governing incremental skill.](figures/fig06_hydrologic_mechanism.pdf)

**Figure 5. Hydrologic conditions governing incremental skill (held-out
2021–2023).** (a) Distribution of the station anomaly half-life from the
official train-fitted damped-persistence anchor (median 6.9 d; an anomaly
half-life, not an e-folding time). (b) One-day learned gain over damped
persistence versus log half-life, with the least-squares trend (r = −0.40);
longer-memory stations leave less to learn at short leads. (c) Seven-day median
station paired ΔRMSE (ThermoRoute minus damped persistence) within
issue-time-identifiable states; negative values favor ThermoRoute.
(d) Retrospective outcome-conditioned strata (actual rapid warming and cooling
computed separately with station-specific signed thresholds); negative values
favor ThermoRoute. The learned model's incremental value concentrates in rapid
warming and cooling states and under low thermal anomaly; it is absent on the
coldest target decile.

### 4.7 What local thermal state is worth

Section 4.5 could not resolve a spatial effect because every arm kept the held
station's own thermal history, leaving the transferable component worth about
0.07 °C. This section removes that history instead of the geography, which is
the contrast the design can actually resolve.

Four nested information levels are fitted under whole-region holdout, so no
gauge from a held region appears in training and each level withholds one more
local observation: **L0** keeps everything; **L1** pools the climatology and
damped rate over training stations, so the target site contributes no long-term
statistic; **L2** removes every target-site water-temperature input — lags,
rolling statistics, deltas, their observedness flags, the climatology anomaly
and the persistence anchor — leaving discharge and meteorology; **L2-U2**
removes discharge as well.

Prohibited inputs are dropped from the design matrix rather than filled, and
before each cell is fitted a proof perturbs the held stations' own observations
and requires a bit-identical design matrix. All 24 proofs return a maximum
absolute difference of exactly zero; a negative control that re-admits a single
water-temperature lag is caught with a 147 °C shift, so the proof is not
vacuous.

**Table 4.15 — value of local thermal information.** Station-first paired
differences in °C under whole-region holdout, raw-target tree, 116 reportable
stations; positive means the withheld information was worth that much. CI is a
10,000-draw whole-HUC2 cluster bootstrap.

| Withheld | Lead | Value (°C) | 95% CI | Stations worse |
| --- | ---: | ---: | --- | ---: |
| Own long-term statistics (L1−L0) | 1 d | 0.010 | [0.007, 0.012] | 0.82 |
| | 3 d | 0.062 | [0.041, 0.072] | 0.91 |
| | 7 d | 0.161 | [0.139, 0.195] | 0.91 |
| Recent temperature sequence (L2−L1) | 1 d | 1.707 | [1.346, 2.432] | 1.00 |
| | 3 d | 1.201 | [0.889, 1.982] | 1.00 |
| | 7 d | 1.169 | [0.850, 1.638] | 1.00 |
| **All local thermal state (L2−L0)** | **1 d** | **1.716** | **[1.363, 2.442]** | **1.00** |
| | **3 d** | **1.255** | **[0.989, 2.064]** | **1.00** |
| | **7 d** | **1.325** | **[1.092, 1.725]** | **1.00** |
| Local discharge (L2-U2−L2) | 1 d | 0.084 | [−0.033, 0.227] | 0.55 |
| | 3 d | 0.043 | [−0.051, 0.180] | 0.54 |
| | 7 d | 0.009 | [−0.043, 0.116] | 0.53 |

Three things follow, and the first reorders the paper.

**Local thermal state is the dominant information in this problem.** Withholding
it costs 1.3–1.7 °C, at every lead, at every one of 116 stations, with
leave-one-HUC2-out ranges that never approach zero. That is two to thirteen
times the value of realized future meteorology (Section 4.8), roughly seventy
times the architecture effect of Section 4.3, and two orders of magnitude above
the geometry penalty of Section 4.5. A study that keeps the target gauge's
recent readings and then reports a model comparison is comparing models inside
the regime where the largest available information is already present.

**Almost all of that value is the recent sequence, not the station's long
history.** Pooling a station's climatology and damped rate over the training
stations costs 0.01–0.16 °C; removing its recent readings costs 1.17–1.71 °C, a
factor of ten to one hundred and seventy. Cold-starting a gauged site is
therefore nearly free, while thermally ungauged prediction is a different
problem — which is the distinction the transfer literature draws and the one a
random-site split cannot see.

**Discharge does not substitute for thermal history.** Once water temperature is
gone, also removing discharge changes station-median RMSE by 0.009–0.084 °C with
every interval covering zero and stations splitting about evenly. The
"hydrology observed" rung is, on this cohort, barely distinguishable from having
no local observation at all.

**Spatial geometry matters, but only once local history is gone.** Section 4.5
measured a geometry penalty of 0.004–0.009 °C and could not resolve it. Running
the same four levels under random-site holdout as well, with the station
contrast formed inside each of five split seeds and the paired contrasts then
averaged, shows why: the penalty is 0.006–0.014 °C at L0 and 0.073–0.129 °C at
L2, about thirteen times larger. The station-level double difference — the
geometry penalty at L2 minus the penalty at L0 — is +0.066 to +0.124 °C, and
every whole-HUC2 interval excludes zero at every model and lead, with
leave-one-HUC2-out ranges that keep their sign.

This is the interaction the transfer literature's concern predicts, and it
reconciles Section 4.5 with that concern rather than contradicting it. A
random-site split in a dense network measures something closer to interpolation
than transfer, but the cost of that substitution is invisible while the held
gauge still supplies its own thermal history: the model barely needs its
neighbours. Remove the history and the neighbours start to matter. A study
reporting spatial generalization from random splits at a gauged site is
therefore not merely optimistic by a small margin; it is measuring a quantity
whose sensitivity to the split design is itself conditional on how much local
information the model retains.

---

### 4.8 What future meteorology would be worth

Every result above is issue-time-only, so none of it says whether the small
residual learned gain reflects a limit of the models or a limit of the
information they were given. This section separates the two by refitting the
tree models with the *realized* meteorology over t+1 … t+h substituted for the
issue-time forcing, on the same cohort, the same 116 reportable stations, and
the same forecast keys.

The status of this analysis differs from the rest of Section 4 and the
difference is not cosmetic. F3 is a **retrospective realized gridded
meteorological oracle**: it is Daymet and gridMET estimates at the station
coordinate for days that had already happened, so it is an upper bound on what
perfect weather information would be worth, not a forecast product and not an
operational gain. The analysis is also *post-outcome*: the 2021–2023 window was
already open and earlier forcing results had been inspected when it was
specified, so it is a descriptive reanalysis and not a confirmatory test. It is
reported here, and not in the Abstract or the Key Points, for that reason.
Values come from
`outputs/final/forcing_regime_v5_observed_inference_authority_v1/`.

**Table 4.13 — value of realized future meteorology.** Station-first forcing
value $V_F = \mathrm{median}_i[\mathrm{RMSE}_i(F0) - \mathrm{RMSE}_i(F3)]$ in
°C, so a larger number means more information value. CI is a 10,000-draw
whole-HUC2 cluster bootstrap; LOCO is the leave-one-HUC2-out range of the same
statistic; MDE is the smallest effect this cohort's cluster structure can
resolve at α = 0.05 under exact whole-cluster sign-flip enumeration.

| Model | Lead | RMSE F0 | RMSE F3 | $V_F$ | 95% CI | LOCO range | Win rate | MDE |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| LightGBM | 1 d | 0.614 | 0.451 | 0.130 | [0.081, 0.189] | [0.117, 0.154] | 0.91 | 0.031 |
| LightGBM | 3 d | 1.330 | 0.765 | 0.542 | [0.415, 0.721] | [0.511, 0.641] | 0.92 | 0.152 |
| LightGBM | 7 d | 1.742 | 1.063 | 0.627 | [0.423, 0.803] | [0.558, 0.753] | 0.96 | 0.172 |
| ResidualLightGBM | 1 d | 0.606 | 0.446 | 0.125 | [0.084, 0.183] | [0.117, 0.148] | 0.91 | 0.028 |
| ResidualLightGBM | 3 d | 1.327 | 0.752 | 0.578 | [0.406, 0.704] | [0.507, 0.668] | 0.94 | 0.160 |
| ResidualLightGBM | 7 d | 1.722 | 1.087 | 0.605 | [0.445, 0.795] | [0.553, 0.716] | 0.95 | 0.151 |

Three properties make this a usable measurement rather than a single number.
It is *large relative to what the design can resolve*: every value is three to
four times its own minimum detectable effect, where the architecture effects of
Section 4.3 are smaller than theirs. It is *spatially stable*: the
leave-one-HUC2-out range never changes sign, and the equal-HUC aggregate
(0.151, 0.623, 0.694 °C for LightGBM) is slightly larger than the
station-weighted median, so the effect is not carried by an over-represented
region. And it is *not a target-formulation artifact*: the station-level
contrast between the raw-target and damped-residual trees is −0.002, +0.000,
and −0.026 °C at 1, 3, and 7 days, at most 4% of the forcing value itself.

The lead structure is the more interesting result. Formed as a station-level
double difference, the forcing value rises by +0.415 °C [0.301, 0.541] from one
to three days but by only +0.074 °C [0.046, 0.101] from three to seven
(LightGBM; +0.425 and +0.051 °C for the residual-target tree). Future weather
therefore buys most of what it can buy by day three, and the three-to-seven-day
increment, while still resolvable, is five to eight times smaller. Seasonally,
the value is largest in spring (0.67 °C at three days) and smallest in summer
(0.40 °C), consistent with a regime in which spring temperature is driven by
synoptic variability and summer sits nearer a radiation-controlled equilibrium;
it is stable across the three test years (0.54–0.56 °C at three days).

**The gain is event-scale weather information, not seasonal phase.** A value
computed against F0 could in principle come from a sharper seasonal signal
rather than from knowing what specific weather followed a given issue. We
tested that with a placebo whose design, thresholds and decision rules were
sealed before its outcome existed
(`protocols/wrr_forcing_placebo_protocol_v5a_seal.json`): the meteorology
vector is deranged across dates *within each station-month*, so every station's
climate, seasonal phase and month-level realized conditions are preserved
exactly and only the day-to-day correspondence between a forecast key and its
weather is destroyed. All five permutation seeds were refitted end to end.

The forcing value collapses. Of the true value, the placebo retains 0.0% at one
day, 2.0% at three and 8.8% at seven for the raw-target tree (−0.5%, 1.7% and
8.4% for the residual-target tree). The station-level placebo-minus-true
contrast is +0.129, +0.525 and +0.577 °C at 1, 3 and 7 days, every interval
excludes zero, every leave-one-HUC2-out range keeps its sign, and the placebo
is worse at 91–95% of stations. The sealed rule P1 — retention below 25% — is
satisfied at every model and lead.

**Table 4.14 — shuffled-forcing placebo (sealed pre-outcome).** $V_F$ is the
station-first forcing value in °C; *retained* is the placebo's share of the
true value, the quantity rules P1–P3 are stated in.

| Model | Lead | $V_F$ true | $V_F$ shuffled | Retained | Placebo worse at | Rule |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| LightGBM | 1 d | 0.130 | 0.000 | 0.0% | 0.91 | P1 |
| LightGBM | 3 d | 0.542 | 0.011 | 2.0% | 0.92 | P1 |
| LightGBM | 7 d | 0.627 | 0.055 | 8.8% | 0.94 | P1 |
| ResidualLightGBM | 1 d | 0.125 | −0.001 | −0.5% | 0.91 | P1 |
| ResidualLightGBM | 3 d | 0.578 | 0.010 | 1.7% | 0.94 | P1 |
| ResidualLightGBM | 7 d | 0.605 | 0.051 | 8.4% | 0.95 | P1 |

The residue is itself informative and rises monotonically with lead. At one day
the placebo is worth nothing at all: tomorrow's weather is useful only as
tomorrow's weather. By seven days about a twelfth of the value survives a
within-month shuffle, which is the part attributable to knowing the general
level of a month rather than the sequence of its days — the longer the lead,
the more the anchor has decayed and the more even a month-level statement is
worth.

**What the forcing value is worth as a warning.** A station-median RMSE
improvement is a statistics result; what a release schedule or a thermal-refuge
warning acts on is whether a threshold will be crossed. We therefore scored
both arms on warm-tail exceedance and on signed rapid change, with every
threshold fitted per station on the 2006–2015 training period only and a
minimum of ten events per station-cell
(`outputs/final/forcing_event_metrics_summary.parquet`).

The event result is larger in relative terms than the RMSE result. At seven
days with the raw-target tree, the probability of detection for exceedance of a
station's training 95th percentile rises from 0.18 to 0.55, and for rapid
warming from 0.18 to 0.52; the critical success index roughly triples in both
cases. Crucially this is not a base-rate trade: the false-alarm ratio *falls*
at the same time, from 0.36 to 0.20 for the 95th-percentile exceedance and from
0.43 to 0.21 for rapid cooling. Every station-first paired interval excludes
zero, and the lead structure matches the RMSE result — the event gain is small
at one day and largest at three to seven.

| Event (7 d, LightGBM) | POD F0 → F3 | FAR F0 → F3 | ΔCSI | Stations |
| --- | --- | --- | ---: | ---: |
| Warm exceedance, station q90 | 0.59 → 0.77 | 0.24 → 0.15 | +0.155 | 116 |
| Warm exceedance, station q95 | 0.18 → 0.55 | 0.36 → 0.20 | +0.231 | 114 |
| Rapid warming | 0.18 → 0.52 | 0.28 → 0.20 | +0.274 | 116 |
| Rapid cooling | 0.26 → 0.62 | 0.43 → 0.21 | +0.274 | 116 |

These are deterministic point predictors, so the Brier column that usually
accompanies such a table is here the misclassification rate of a 0/1 indicator
and is reported in the artifact rather than the manuscript; it is not a
reliability result, and none of these models emits a calibrated event
probability. The comparison is also an oracle bound in event space exactly as
it is in RMSE space: it states what perfect weather information would be worth
for warning, not what a forecast product delivers.

**Which rivers depend on future weather.** The station spread in the forcing
value is wide — the middle half runs from 0.30 to 0.86 °C at seven days — and
that spread is structured. Against six attributes declared before the
relationships were inspected, the strongest is the one the physics predicts:
the anomaly half-life of the station's own damped anchor, with Spearman
ρ = −0.49 [−0.66, −0.16] at seven days under whole-HUC2 resampling. Rivers that
hold a thermal anomaly for longer carry their own state forward and need
tomorrow's weather less. The relationship is strongest at one day (ρ = −0.82)
and weakens with lead (−0.63, −0.47), which is the expected direction: local
memory protects a forecast most over the interval it can bridge. Flashier
rivers depend more on future forcing (discharge coefficient of variation,
ρ = +0.48 [0.26, 0.63]), and warmer, lower-latitude stations depend more
(ρ = +0.33 and −0.44). Catchment area is not resolved (ρ = −0.16, interval
covering zero).

One of the six is a nuisance control and it is not null: stations with more
complete training water-temperature records show a smaller forcing value
(ρ = −0.29 [−0.45, −0.07]). Part of the apparent physical gradient may
therefore be a record-quality gradient, and the two cannot be separated on this
cohort. Latitude and mean annual air temperature are also two views of one
gradient. We report these as exploratory covariation on a fixed,
non-probability cohort with about nine effective spatial clusters, not as an
attribution, and we do not fit a multivariable model the design cannot carry
(`outputs/final/forcing_heterogeneity.parquet`).

**The model is using exact event timing.** The shuffle cannot separate
day-to-day correspondence from sub-monthly synoptic persistence, because a
same-month donor can land within a few days of the true valid time. The second
sealed control displaces the realized future by a whole week in each direction,
which leaves climate, near-seasonal phase and local weather persistence intact
and removes only the exact dates. Keys whose displaced valid time falls outside
the record are dropped from both shift arms *and* from the true arm, so all
three are scored on one common shift-registry named separately from the primary
one.

The forcing value does not survive a week's displacement either. The +7-day arm
— the primary timing control that decision rule P4 is stated in — retains
−0.1%, 0.7% and 14.0% of the true value at 1, 3 and 7 days for the raw-target
tree (−0.5%, 1.0% and 13.8% for the residual-target tree), and is worse than
the true arm at 90–95% of stations. The −7-day arm retains essentially nothing
at any lead; it is the weaker control of the pair, because displacing backwards
moves the future window toward information already available at issue time, and
the two are reported separately and never averaged. P4 is satisfied, so the
lead-structure interpretation above stands.

The residues of the two controls are consistent and mildly informative. At
seven days the within-month shuffle leaves 8.8% and the one-week displacement
14.0%: displacing by a week preserves more than shuffling within a month
because synoptic weather is autocorrelated over several days, so the shifted
window still resembles the true one. Both are small, and the direction is what
a weather-information reading predicts.

**The forcing value is future air temperature.** The last sealed arms attribute
it. A component arm gives the selected variable its realized future values and
every other meteorological variable its training climatology, so exactly one
variable carries event-scale information; the complementary family does the
reverse. Both are reported because the variables are correlated and neither
family alone is honest.

At seven days, air temperature alone recovers 0.595 °C of the 0.627 °C full
value (95%), while withholding air temperature and giving everything else its
realized future retains only 0.129 °C (21%). Withholding radiation,
precipitation or humidity-and-wind instead costs nothing measurable: each
`without` arm returns 0.62 °C, indistinguishable from the full value. The same
ordering holds at one and three days, where air temperature alone recovers 87%
and 92%.

The two families together say more than either does alone. Radiation and
humidity-and-wind are individually informative — 0.105 and 0.117 °C on their
own — but entirely substitutable, which is what a correlated predictor looks
like when the variable it tracks is still available. Air temperature is the
only variable that is both sufficient on its own and not replaceable by the
others. Shares therefore exceed 100% in sum and this is not a variance
decomposition.

**About half the oracle survives a real forecast.** Everything above is an
oracle, so the question a practitioner asks — how much of it is attainable —
needs a forecast in the design, not a reanalysis. The F2a arm supplies one. It
reproduces the `only_air_temperature` arm exactly, with a single substitution:
at each valid time the realized air temperature is replaced by an archived
fixed-lead forecast for that day, and every other meteorological variable stays
climatological. Its comparator is therefore that same arm, which is why the
component result above matters operationally: air temperature alone carries 95%
of the full oracle at seven days, so a temperature-only product is not
structurally barred from recovering most of what is there.

Training is where the obvious implementation is wrong, and we record the error
because it is easy to make. The forecast archive covers 2021–2023 only, so a
model *trained* on F2a features sees a future-temperature column that is
climatology on every training row; it learns to ignore that column, and
substituting real forecasts at evaluation then changes nothing. A first run of
this arm did exactly that and reported 1.8% recovery, which measured the mistake
rather than the product. The arm therefore trains on realized air temperature
and substitutes the forecast only when predicting — which is also what an
operational system does.

**Table 4.16 — what an archived fixed-lead temperature forecast recovers.**
Station-first values in °C against F0 on 116 reportable stations; the
denominator is the realized-temperature oracle (`only_air_temperature`), never
the five-variable F3.

| Model | Lead | Oracle $V$ | Forecast $V$ | Recovery |
| --- | ---: | ---: | ---: | ---: |
| LightGBM | 1 d | 0.113 | 0.056 | 49% |
| LightGBM | 3 d | 0.499 | 0.308 | 62% |
| LightGBM | 7 d | 0.595 | 0.299 | 50% |
| ResidualLightGBM | 1 d | 0.111 | 0.054 | 49% |
| ResidualLightGBM | 3 d | 0.532 | 0.320 | 60% |
| ResidualLightGBM | 7 d | 0.564 | 0.326 | 58% |

Roughly half to three-fifths of the temperature oracle survives contact with a
real forecast, and the shortfall grows with lead in the way forecast skill does.
The ratio is reported as a point estimate: each difference carries its own
whole-HUC2 cluster interval, but an interval on a quotient of two estimated
medians would be one in name only.

Three constraints travel with this number and none of them is rhetorical. The
series is a **fixed-lead composite** — for each valid time, what a run issued
*h* days earlier said about that day — not a coherent trajectory from one
initialization and not an as-issued operational archive; the sealed protocol
forbids both readings. The archive spans 2021–2023 at these stations, so this is
a measurement of those years, not a climatological expectation. And the arm
supplies air temperature only, so it is not an estimate of what a full
operational forcing product would deliver. What remains unrun under the same
seal is the F2b archived-vintage arm, which is the one that would answer that
last question.

---

### 4.9 Future weather does not substitute for a local gauge

Sections 4.7 and 4.8 each varied one information axis while holding the other at
its default, and reported that way they invite a reading neither tested: that
the two are substitutes, so an ungauged reach could buy back with weather what
it lost with the sensor. That reading has practical consequences — it is the
argument for instrumenting a basin with forecasts rather than thermistors — so
we crossed the axes and asked directly.

The F3 forcing arm was refitted at L0 and at L2 under the same whole-region
folds, the same mask-invariance proof, and the same level-legal anchor as
Section 4.7, giving a forcing value at each information level and a
station-level double difference between them. Because these cells use
whole-region holdout rather than the split of Section 4.8, the L0 forcing values
below are slightly smaller than Table 4.13's; the comparison that matters is
between the two levels within this table, where the folds are identical.

**Table 4.17 — value of realized future meteorology by information level.**
Station-first paired values in °C under whole-region holdout, raw-target tree,
116 reportable stations, 10,000-draw whole-HUC2 cluster bootstrap.

| Quantity | 1 d | 3 d | 7 d |
| --- | ---: | ---: | ---: |
| Forcing value at L0 (gauged) | 0.116 [0.059, 0.174] | 0.474 [0.299, 0.693] | 0.580 [0.366, 0.779] |
| Forcing value at L2 (thermally ungauged) | 0.038 [0.020, 0.072] | 0.331 [0.249, 0.496] | 0.596 [0.431, 0.804] |
| **Interaction (L2 − L0)** | **−0.076 [−0.098, −0.056]** | **−0.147 [−0.193, −0.079]** | **+0.016 [−0.035, +0.099]** |

The interaction is negative at one and three days and its interval excludes zero
at both, for both target formulations. Future weather is worth *less* to a model
that has lost the local gauge, not more: at one day it retains a third of its
gauged value, at three days seven-tenths. Only by seven days do the two levels
converge, and there the interaction is indistinguishable from zero
(+0.016 [−0.035, +0.099]).

The mechanism is the one the anchor makes visible. Realized future meteorology
earns its value by correcting a trajectory, and at short leads the trajectory it
corrects is the persistence anchor built from the station's own recent readings.
Remove that anchor and the model falls back on a pooled climatology, which is
too coarse a starting point for tomorrow's weather to sharpen — there is less
error of the right kind left to remove. By seven days the anchor has decayed far
enough that a gauged model is not much better positioned than an ungauged one,
and the forcing value stops depending on which it is.

This closes the substitution argument in the unfavourable direction. The two
information sources are complements at the leads where local state dominates,
so a thermally ungauged reach is harder than either Section 4.7 or Section 4.8
implies on its own: it loses 1.3–1.7 °C to the missing gauge and then recovers
less from forecasts than a gauged site would. The one qualification is the
seven-day cell, where forcing is worth the same either way — which is also the
lead at which the archived forecast of Section 4.8 recovers only half the
oracle, so the regime where forcing substitutes best is the regime where it is
least attainable.

Geometry is held at whole-region throughout. Letting a third axis vary would
produce a three-way contrast 116 stations across nine effective clusters cannot
support; the local-information-by-geometry interaction is measured separately in
Section 4.7. Like that one, this analysis is post-outcome and descriptive, not a
confirmatory test, and is reported outside the Abstract and Key Points for that
reason
(`outputs/final/forcing_information_interaction_v1/`).

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
affected. Whether this accounts for the between-study dispersion catalogued by
[Corona and Hogue (2025)](https://doi.org/10.5194/hess-29-2521-2025) is a
testable question that we have not tested: it requires re-scoring published
models against a fitted damped anchor on a common registry, not the single
cohort studied here. What this study does establish is that the choice is
consequential enough to be worth testing, and we suggest damped persistence as
the minimum reference for this variable. The point is not new — benchmark-
relative evaluation has been argued for in hydrology for decades
([Klemeš, 1986](https://doi.org/10.1080/02626668609491024);
[Schaefli and Gupta, 2007](https://doi.org/10.1002/hyp.6825);
[Seibert et al., 2018](https://doi.org/10.5194/hess-22-4323-2018)) — but the
size of the effect for this particular variable, measured on identical keys
with a station-first paired estimand, has not previously been quantified.

The second is the spatial partition, and here our design was not able to
measure what it set out to measure. A model evaluated on randomly held-out
gauges in a dense network is being asked to interpolate between instrumented
neighbours, and holding out whole regions moves the median distance to the
nearest training gauge from 60 to 263 km. But because every arm retains the
held station's own thermal history, the transferable part of the model is worth
about 0.07 °C at seven days, and the measured geometry penalty (0.004–0.009 °C)
sits inside its own resampling noise (Section 4.5). We therefore make no claim
about the size of the spatial effect on this cohort, in either direction. The
design lesson stands independently of the measurement: a random-site split in a
dense network measures something closer to interpolation than to transfer, and
separating the spatial partition from the local-information level is a
prerequisite for measuring either — the argument Klemeš (1986) made for
differential split-sample testing, applied to space rather than to climate.

The third is the key set and the preprocessing boundary. Scoring each model on
its own complete-case set and fitting scaling, imputation, climatology, event
thresholds, or interval offsets on windows that include the evaluated interval
both move error downward without leaving a visible trace. We cannot quantify how
much of the published literature is affected, and we do not assert that it is;
what we can say is that removing both channels here left the constrained deep
model behind a gradient-boosted tree at the 1- and 3-day leads and matched at
7 days.

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
rather than by site. None requires unusual computational resources, and the
reference choice alone moved a reported number in this study by far more than
the differences between the tested architectures. On this cohort, the reference choice and the information set each moved a
reported number by more than the differences between architectures did; whether
that ordering holds across the published literature is a question this study
motivates rather than settles, because it would require re-scoring published
models on a common registry. The key-selection and preprocessing channels are
closed by construction in this design rather than quantified as separate
effects (Section 4.5, SI11).

---

## 6. Limitations

All caveats are consolidated here; earlier sections refer to them by number.

**Scope.** The design is a descriptive benchmark on a fixed cohort:
comparisons are station-first and approximate (15 HUC2 clusters; effective
cluster count 9.54; largest share 21.7%; non-probability sampling), and no
interval, p-value, or ranking claim is decision evidence. Prediction at
locations with no water-temperature record is not quantified in the current
evidence package: the earlier L2/L3 and forcing-ladder learned results were
withdrawn after the raw-observedness lineage defect was identified, and the
corrected score-producing authorities remain pending. The manuscript therefore
makes no operational-replay or ungauged-location performance claim from those
withdrawn runs. The fitted relaxation rate is
a statistical quantity only, not a heat-transfer coefficient, and no physical
interpretation of its value is offered. Throughout the paper "causal"
describes time ordering only, never causal inference.

**Cohort, measurement, and design.** The cohort trades a real, modest
discharge channel (removal costs +0.042 °C at 1 day; SI19) against wide
spatial coverage; 950 of 1,465 candidates were excluded for missing joint
flow. Point-scale meteorology is used at station coordinates rather than
basin integrals. The air2stream-style hybrid is an unofficial empirical
thermal-recurrence variant of the published formulation (Toffolon and
Piccolroaz, 2015) — not the official code, not a validated reproduction — so
no process-side claim is attached to its scores; running the official code is
planned (protocol v3, Phase 5b). Training and inference wall-clock costs were
not recorded. The development-period spatial analysis and robustness probes
of the earlier draft are archived in SI19.

**Reference construction.** The damped anchor is a fitted object and the
seven-day headline moves by a factor of 2.5 across defensible constructions of
it (Section 4.1). Every number in this paper that is stated against damped
persistence is therefore conditional on the anchor specified in Section 3.1,
and a study adopting a cruder seasonal reference would report a larger gain
from the same fitted weights. We report the published construction as the
primary one because it was fixed before the held-out window was opened, not
because it is the strongest anchor available; the directly fitted lead-specific
decay is stronger.

**Threshold and archive scope.** The ±1 °C algebraic bound, event-threshold
quantiles, and reportability thresholds (100 paired keys) are
development-selected; the q90 event threshold is a statistical tail
diagnostic with no biological, ecological, or regulatory interpretation.
Public data licenses and rights are discussed in SI16.
## 7. Conclusions

We evaluated daily river water-temperature prediction at 120 stable U.S. gauges
across 15 hydrologic regions, scoring every model on one common set of
station/date/horizon keys at 1-, 3-, and 7-day leads, with every preprocessing
and calibration statistic fitted strictly backwards in time and with a held-out
2021–2023 test window for the comparative evaluation.

Three findings survive on that independent window. First, the reference model
governs the reported gain: the deep predictor's seven-day station-median skill
is +0.250 against persistence but +0.038 against damped persistence, with a
median station-level memory gain of 0.49 °C and a median learned gain of 0.07 °C
at seven days (the per-station memory fraction has a median of 0.875). Second,
model ranking does not
favor architectural constraint: a gradient-boosted tree with site identity has
the lowest station-median RMSE at 1- and 3-day leads (0.589 and 1.304 °C), and
the deep model's 7-day point estimate (1.694 versus 1.735 °C) is not
separated from the tree at the cluster level, though that row is underpowered
rather than null (Table 4.6; Section 4.4); the one-factor
ablations (router, mixture, dynamic prior, residual bound) move one-day error by
at most 0.004 °C on the development window with no material held-out gain.
Third, and negatively, this design cannot resolve the spatial-partition
question it was built to ask. Holding out whole regions rather than random
sites moves the median distance to the nearest training gauge from 60 to
263 km but changes station-median RMSE by less than the five-seed resampling
spread of that same quantity (Section 4.5); and because every arm retains the held
station's own thermal history, the model component that would have to transfer
is worth about 0.07 °C in the first place. The contrast that removes local
information is the informative one, and the arm attempting it here carries a
recorded preprocessing defect (DLOG-012), so we report no value for it. The
corrected four-level information ladder of protocol v2 is the experiment that
answers this question, and it is not yet complete.

Two boundaries define what these findings mean. Every model evaluated here is
issue-time-only, so all of the above describes a single information regime: the
one in which recent local thermal history is available and future meteorology
is not. A separate retrospective analysis on the same cohort and keys
(Section 4.8) puts an upper bound on what perfect future meteorology could add:
0.13 °C at one day and 0.54–0.63 °C at three and seven days, an order of
magnitude above the architecture effect measured under issue-time information.
That bound is an information ceiling from a realized-weather oracle. A
placebo sealed before its outcome existed shows it is event-scale weather
information rather than seasonal phase: deranging the meteorology within each
station-month, which preserves climate and seasonal phase exactly, removes
91–100% of the value (Section 4.8). A second sealed control shows the same
under a one-week displacement of the future, so the model is using exact event
timing rather than the weather window around it. Reported
skill for this variable is nonetheless governed jointly by the reference model
and by the information the model is given, and neither is a property of the
architecture.

The common mechanism on the design side is that each choice absorbs an
alternative explanation into the reported number: seasonal damping into a weak
reference, selective prediction into a model-specific key set, and spatial
interpolation into a random site split. For water-resource practice, a
published skill score is not interpretable without its reference model, its
spatial partition, and its information set, and the incremental value of
architectural complexity over a strong statistical anchor and a well-tuned tree
should be re-measured under these controls before it is relied upon.

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
review described in Section 6 completes. The derived panel encodes values from
three providers whose terms differ, and a derived product does not inherit the
most permissive of its upstream terms. Until every object proposed for the
deposit carries a recorded, evidence-backed redistribution decision, the
deposit is described here as planned and specified, not as existing.

**Primary observational sources.** All observations are obtained from public providers and none is the property of
the authors. Daily-value water temperature and discharge come from the U.S.
Geological Survey National Water Information System
([U.S. Geological Survey, 1994](https://doi.org/10.5066/F7P55KJN); parameter
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
generates Section 4.4 from the test-window metrics, and the fully transitive
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

**Reproduction.** Sections 4.1–4.3 are reproducible from the archived panel, the archived model
bundles, and the pinned environment. Section 4.4 is reproducible from the
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
and unit conventions (SI03); the analysis protocol, redesign chronology and
decision log (SI04); the comparison set and its fields (SI05–SI06); all-model scores on the
exact common keys, including the fitted air2stream-style hybrid reference with its parameter ranges and
official-variant caveat (SI07); probability and
reliability schemas and the full measured degenerate-interval table (SI08); architecture and
information-matched controls with seed and budget provenance and their
independent-window re-test (SI09); the temporal
coverage audit (SI10); the matched spatial factorial, local-adaptation policy,
repeated-split distribution and the
cluster geometry at HUC2, HUC4, HUC6, and HUC8 (SI11); outcome quality control
and qualifier evidence (SI12); the history-dependent external arm (SI13);
missingness and failure cases with the key-history completeness strata (SI14);
reproduction hashes, commands, and
environment parity fields, including the results-authority commands and manifest (SI15); and the rights and data dictionary (SI16).

Supporting figures accompany these sections: Figures S1–S3 describe the cohort
geometry, the temporal roles and issue-time information boundary, and the model
and calibration dataflow; Figures S4–S8 and S10 expand Section 4.4; and Figure
S9 is a development-period conformal-calibration sensitivity, labeled as such in
the figure itself, which must not be compared numerically with any held-out-window
figure.

Each figure carries evidence from exactly one period. Figures 1 and 2 and Figures
S1–S3 are structural material; Figures 3–5 and Figures S4–S8 and S10 are
held-out-window; Figure S9 is a development-period conformal-calibration
sensitivity. Each figure states its period inside the figure, and no value from
one period is compared numerically with a value from another.

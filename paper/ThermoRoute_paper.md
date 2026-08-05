# Strong baselines, controlled leakage, and a pre-specified inference gate for daily river water-temperature hindcasting

[Author One]^a,\*^, [Author Two]^a^, [Author Three]^b^

^a^ [Department / Laboratory, Institution, City, Postcode, Country]

^b^ [Department / Laboratory, Institution, City, Postcode, Country]

\* Corresponding author: [replace with verified name, ORCID, affiliation, and e-mail]

## Key Points

- A 120-gauge, 657,480 site-day panel is used to benchmark a constrained
  river-temperature predictor against damped persistence, a tree ensemble, and a
  global LSTM on identical station/date/horizon keys.
- Evaluation labels for 2021–2023 are acquired exactly once, after the model
  suite, the input set, and the analysis code are frozen and independently
  replayed.
- A pre-specified inference gate on cohort cluster structure fails on the frozen
  cohort, so the formal comparisons are reported as fixed-cohort descriptive
  effects rather than as inferential conclusions.

## Manuscript status

The one-time 2021–2023 evaluation has not been executed at the time of writing.
Section 4 therefore contains the development-period descriptive results that
exist, together with clearly marked slots for the five formal comparisons and
their supporting tables. Every slot names the artifact that will fill it. The
author block, the archive DOIs, and the data licence fields are placeholders.

## Abstract

Daily river water temperature is difficult to hindcast honestly. The signal is
strongly persistent, so a copy of the most recent observation is already a hard
reference; meteorological and hydrological covariates leak future information
unless their issue-time availability is defined explicitly; sensor records are
incomplete in ways that are correlated with season and with site; and gauges in
the same region are not statistically independent. We assembled a panel of
657,480 site-days from 120 stable U.S. Geological Survey site numbers spanning
2006–2020, 34 states, and 15 two-digit hydrologic unit (HUC2) groups, and used
it to evaluate ThermoRoute, a bounded-deviation predictor that combines a
damped-persistence anchor, a learned relaxation proposal, a sparse
horizon-conditioned variable and lag router, a strictly left-looking temporal
convolutional encoder, a regime mixture, and a residual whose deviation from the
anchor is algebraically bounded. Three design choices define the study. First,
the comparison set is a strong baseline suite — persistence, damped persistence,
seasonal climatology, an air2stream-style reference, a global gradient-boosted
tree ensemble with and without site identity, and a global LSTM — scored on
identical station/date/horizon keys, because a comparison against persistence
alone is uninformative for this variable. Second, leakage is controlled
mechanically rather than by assertion: all predictors are dated no later than the
issue date, no horizon-specific future weather field is consumed, the temporal
encoder is non-anticipating by construction, and every preprocessing, calibration,
and threshold statistic is fitted strictly before the interval on which it is
applied. Third, the 2021–2023 evaluation labels are downloaded exactly once,
after the model suite and the input set are frozen, hash-sealed, and replayed by
an isolated process that cannot reach the label namespace. A pre-specified,
outcome-free inference gate requires at least 30 reportable clusters, an
effective-cluster fraction of at least 0.75, and a largest-cluster share below
0.25. The frozen cohort contains at most 15 HUC2 groups with an inverse-Herfindahl
effective cluster count of about 9.54, so the gate fails independently of any
outcome. The five formal comparisons are consequently reported as fixed-cohort
descriptive effects under the verdict `DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED`,
with whole-HUC2 sign-flip p-values, cluster-bootstrap intervals, and Holm
adjustments retained only as assumption-conditional sensitivities. We argue that
stating this limit explicitly, before the outcomes are seen, is more useful to
the field than reporting cluster-naive intervals as though they were inferential.

**Plain Language Summary.** River temperature changes slowly from one day to the
next, so a forecast that simply repeats yesterday's reading is already fairly
accurate. That makes it easy to publish a model that looks impressive but is not.
We built a river-temperature model and then went to unusual lengths to test it
fairly: we compared it against several strong alternatives on exactly the same
days and sites, we made sure the model can never see information from after the
moment it is asked to predict, and we agreed in advance — in writing, sealed
against later editing — what would count as a result. We also decided in advance
that our set of gauges is spread over too few large river regions to make
statistical statements about rivers in general. That turned out to be true, so we
report our numbers as a description of these 120 gauges rather than as a general
conclusion. The final test data, covering 2021 to 2023, are downloaded only once,
after everything else is locked.

**Keywords:** river water temperature; retrospective hindcasting; strong
baselines; clustered inference; conformal prediction; pre-registration.

---

## 1. Introduction

Stream and river thermal regimes control dissolved-oxygen saturation, metabolic
rates, life-stage timing, and habitat suitability, and they are set by a
combination of atmospheric forcing, discharge, groundwater exchange, riparian
shading, channel geometry, and regulation
([Caissie, 2006](https://doi.org/10.1111/j.1365-2427.2006.01597.x)). Daily
water-temperature prediction has consequently become a common application for
statistical and machine-learning models, from nonlinear air–water regressions
([Mohseni et al., 1998](https://doi.org/10.1029/98WR01877)) through hybrid
air-temperature and discharge formulations
([Toffolon and Piccolroaz, 2015](https://doi.org/10.1088/1748-9326/10/11/114011);
[Piccolroaz et al., 2016](https://doi.org/10.1002/hyp.10913)) to deep sequence
models, multi-task learners, and physics-guided river-network architectures
([Feigl et al., 2021](https://doi.org/10.5194/hess-25-2951-2021);
[Rahmani et al., 2021](https://doi.org/10.1088/1748-9326/abd501);
[Jia et al., 2021](https://doi.org/10.1137/1.9781611976700.69);
[Sadler et al., 2022](https://doi.org/10.1029/2021WR030138);
[Zwart et al., 2023](https://doi.org/10.3389/frwa.2023.1184992)). Recent reviews
of this literature note that reported accuracies are difficult to compare because
reference models, key sets, and data-splitting conventions differ across studies
([Corona and Hogue, 2025](https://doi.org/10.5194/hess-29-2521-2025)).

The difficulty is not that daily water temperature is hard to predict. It is that
it is easy to predict badly and still report an attractive number. Four
properties of the problem make this so, and each one motivates a specific design
decision in this study.

**Persistence is a strong reference.** Daily mean water temperature has a large
thermal inertia relative to a one-day step. On the panel used here, the
station-median root-mean-square error (RMSE) of naive persistence is 0.803 °C at
a one-day lead and 2.217 °C at a seven-day lead, and simply relaxing the last
observation toward a fitted seasonal climatology reduces the seven-day figure to
1.738 °C. Any model that is compared only against naive persistence therefore has
roughly a quarter of the seven-day error budget handed to it before it learns
anything. A reported skill score against persistence is not, by itself, evidence
that a model has learned river thermodynamics; it may only be evidence that the
model has learned to damp.

**Issue-time leakage is easy and invisible.** Water temperature covaries strongly
with same-day and next-day air temperature. If a model is given a gridded
meteorological field whose timestamp is the target date rather than the issue
date, its apparent skill increases substantially, and nothing in a conventional
train/test split will reveal this. The same problem arises more subtly through
preprocessing: scaling constants, imputation fills, climatological references,
event thresholds, and interval-calibration offsets are all statistics of the
data, and fitting any of them on a period that includes the evaluated interval
transfers information backwards in time.

**Sensor records are incomplete, and the incompleteness is structured.** In the
frozen panel, roughly 15.8% of daily water-temperature values are absent, with
maximum contiguous gaps of 2,404 days for water temperature and 1,369 days for
discharge. Missingness of this kind is not missing-at-random: gaps cluster in
winter at ice-affected sites and around instrument service intervals. Any
evaluation conditioned on the observability of the target inherits that
conditioning, and any station-selection rule based on coverage thresholds makes
the resulting cohort an availability-enriched sample rather than a random one.

**Rows from the same station and region are not independent.** A daily panel of
120 gauges over fifteen years contains hundreds of thousands of rows, but it does
not contain hundreds of thousands of independent observations. Treating daily
rows as exchangeable produces confidence intervals that are implausibly narrow.
Treating stations as the sampling unit is better but still insufficient, because
neighbouring stations share weather, geology, and regulation. The natural
clustering unit for a U.S. panel is a large hydrologic region, and a study with a
few dozen gauges typically has only a handful of such regions.

This study addresses the narrow empirical question that these four properties
leave open: on fixed common station/date/horizon keys, at 1-, 3-, and 7-day
leads, how does a constrained learned correction behave relative to a strong
statistical reference set, to a tuned gradient-boosted tree ensemble, and to a
global recurrent network, when the information boundary is enforced mechanically
and when the evaluation labels are seen exactly once? It also asks whether the
associated intervals and event probabilities are useful empirical descriptions of
uncertainty. It does not attempt to infer an energy budget, a hydraulic state, or
a management utility from the fitted architecture, and it does not claim
prediction at ungauged locations.

Our contribution is therefore methodological as much as it is architectural. We
report: (i) a comparison set that includes references most river-temperature
studies omit; (ii) a leakage-control mechanism that is checkable rather than
asserted; (iii) a regional-transfer design that holds out whole HUC2 regions and
that we label honestly as *gauged* transfer, because issue-time water-temperature
history remains available at the held-out sites; (iv) split-conformal intervals
with an explicit statement of what they do and do not guarantee; (v) a single,
irreversible acquisition of the 2021–2023 evaluation labels after the model suite
is frozen; and (vi) a pre-specified inference gate on cohort cluster structure
that we allowed to fail, and whose failure we report as a result about the study
design rather than as a footnote.

The repository underlying this study also contains an older three-station case
study using the identifiers b1, s2, and p3. b1, s2, and p3 are ordinary
monitoring stations, not reservoirs. No verified metadata establish any
upstream/downstream ordering, hydraulic connectivity, regulation status, or
travel time among b1, s2, and p3. That legacy case is outside the present
evaluation: its source provenance, station-identity metadata, measurement
dictionary, and redistribution authorization are unverified, and its outputs
predate the stable USGS site numbers and lineage records used here. It is
described in Section 2.5 only so that readers encountering it in the archive know
its status.

---

## 2. Data and cohort

### 2.1 The frozen panel and its registry

The canonical development record is a single derived panel,
`data_usgs/panel_usgs_120v2.parquet`, bound by a manifest
(`data_usgs/frozen_panel_v1.json`) to a station registry
(`data_usgs/station_registry_v1.csv`). Each of 120 sites carries one row for
every calendar day from 2006-01-01 through 2020-12-31, giving 657,480 rows with
no gaps in the date index; missing observations are explicit nulls rather than
absent rows. Station identity is the zero-padded USGS `site_no`. Legacy
sequential aliases of the form `n000`–`n119` exist in older artifacts and are
never joined without first resolving them through the registry.

The registry spans 34 states and 15 HUC2 groups. The seven raw variables consumed
by the models, in frozen order, are water temperature (`WTEMP`), streamflow
(`FLOW`), air temperature (`TEMP`), precipitation (`PRCP`), a relative-humidity
proxy (`RHMEAN`), Daymet daylight-period mean incoming shortwave radiation in
W m⁻² (`DH`), and wind speed (`WDSP`). Two of these names are legacy and are
easily misread: `DH` is Daymet `srad`, the incident shortwave flux averaged over
the daylight period, not day length and not a 24-hour mean; `RHMEAN` is a
reproducible vapour-pressure proxy evaluated at the tmax/tmin midpoint, not a
direct daily-mean humidity observation. Gage height (`WLEVEL`) is retained as raw
evidence but is not a model input. Water-temperature and discharge values come
from USGS NWIS daily values
([U.S. Geological Survey, 2024](https://doi.org/10.5066/F7P55KJN)),
meteorological fields from Daymet V4
([Thornton et al., 2022](https://doi.org/10.3334/ORNLDAAC/2129)), and wind from
gridMET ([Abatzoglou, 2013](https://doi.org/10.1002/joc.3413)).

Observed-value missingness in the committed panel is approximately 15.8% for
`WTEMP`, 2.8% for `FLOW`, about 0.07% for `TEMP`, `PRCP`, `RHMEAN`, and `DH`, and
zero for `WDSP`. These rates describe the derived panel; they do not reconstruct
provider request history. The small meteorological gap is structured rather than
random: all 120 sites lack the provider-calendar final day in the leap years
2008, 2012, 2016, and 2020. `WLEVEL`, retained only as provenance, is missing on
77.65% of rows.

### 2.2 Why the cohort is what it is

The cohort was not sampled at random from U.S. rivers, and the manuscript does
not treat it as though it were. Candidate selection began from a discovery query
over stream sites in the states represented by the panel and rejected 1,345
candidate stations: 950 lacked joint water-temperature and discharge
availability, 376 failed a full-period coverage threshold, and 19 failed a
2019–2020 coverage threshold. The surviving ledgers are consistent with
full-period coverage thresholds of 0.55 for `WTEMP` and 0.70 for `FLOW`, and
2019–2020 thresholds of 0.80 for both. This has an important consequence that we
state plainly rather than bury: the 2019–2020 interval participated in cohort
construction, so it is development data in the strict sense and has additionally
informed model and narrative development. It is exploratory throughout this
manuscript.

Two further properties of the retained cohort bear on the statistical treatment
in Section 3.6. Among retained stations, 38 share a repeated HUC code with
another retained station and 19 have another retained station within 10 km, so
station-level independence is not tenable. Water temperature and discharge have
maximum contiguous missing runs of 2,404 and 1,369 days respectively. Two
stations contain 2,059 rows with signed negative discharge (minimum −121 cfs);
these are retained with their sign rather than clipped, and their tidal,
backwater, or measurement semantics are not resolved here.

The original discovery responses, retrieval timestamps, exact command line, and
complete run configuration for the 1,465-candidate selection were not retained,
so that step is auditable from committed ledgers but is not source-replayable.
The panel also retains no NWIS qualifier columns, method or sensor history, or
original daily-value response bytes, so measurement discontinuities and qualifier
sensitivity cannot be reconstructed for 2006–2020. Both limitations apply only to
the development record; the requirements for newly acquired evaluation inputs are
stricter (Section 3.7).

### 2.3 Temporal roles

Temporal roles are fixed before any model is fitted, and a training sample is
admitted only if its issue date and every one of its target dates fall inside the
same partition.

| Role | Dates | Rows | Observed `WTEMP` | Permitted use |
|---|---:|---:|---:|---|
| Training | 2006–2015 | 438,240 | 341,646 | fit preprocessing and models |
| Validation | 2016–2017 | 87,720 | 84,074 | choose frozen settings |
| Calibration | 2018 | 43,800 | 42,279 | conformal and Platt fits; final year of the 2006–2018 seasonal event reference |
| Development evaluation | 2019–2020 | 87,720 | 85,621 | previously inspected; exploratory diagnosis only |
| One-time evaluation | 2021–2023 | — | sealed | opened once, after all freezes (Section 3.7) |

On the development evaluation partition, the intersection of admissible keys
across all primary models comprises 249,072 station/date/horizon keys, evenly
distributed as 83,024 keys per lead. All development comparisons reported in
Section 4 use exactly this common key set.

### 2.4 Evaluation-period inputs and the information boundary

The evaluation interval is 2021-01-01 through 2023-12-31 for the same 120-site
cohort. Historical Daymet and gridMET covariates for that interval are retrieved
and archived *before* any outcome is accessed. Meteorology is represented at each
station coordinate and is not aggregated over upstream catchments, which is a
real limitation for large basins and is listed as such in Section 6.

The primary information set uses provider values dated no later than each
historical issue date and consumes no horizon-specific future weather forecast.
This is a date-indexed retrospective hindcast. It is not an operational replay:
the as-issued provisional vintage of a gridded product at a historical issue date
cannot be reconstructed after the fact, so archiving requests, responses,
timestamps, and checksums freezes the dataset actually evaluated without proving
that identical values were available operationally at that time. NWIS dates
denote site-local finalized daily values while Daymet and gridMET use
provider-specific calendar-day definitions; no subdaily day-boundary
harmonization is claimed.

Because the 2006–2020 panel lacks its original HTTP responses, a separate
outcome-free bridge re-fetched 2018–2020 Daymet and gridMET with the same parser
used for the evaluation-period acquisition and compared every predictor on the
exact site/date registry. Its committed manifest records `PASS_EXACT_PRODUCT_BRIDGE`.
That gate detects product-version, parsing, scaling, missingness, and one-day
alignment drift; it cannot prove that provider calendar days share NWIS local-day
boundaries, and it cannot reconstruct as-issued availability.

A separate, deliberately restricted external cohort is defined after the model
suite is committed. A fixed 34-state metadata query identifies stream sites
advertising daily water-temperature capability, without requesting values,
coverage dates, or event rates, and a deterministic seed selects 30 stations
whose USGS site identifiers do not occur in the development registry. This is
site-identifier disjointness only; it is not proof of HUC8, river-network, or
distance separation. Models for that cohort are pooled and station-agnostic, but
they still consume each target site's observed water-temperature history through
the issue date. It is a history-dependent new-gauge exercise, it sits outside the
formal comparison family, and it cannot change model or site selection.

### 2.5 Legacy three-station material

The archive contains three legacy CSV files identified as b1, s2, and p3. b1, s2,
and p3 are ordinary monitoring stations, not reservoirs. No verified metadata
establish any upstream/downstream ordering, hydraulic connectivity, regulation
status, or travel time among b1, s2, and p3. Older outputs derived from those
files predate stable USGS site numbers, the current panel contract, and the
prediction-lineage records used here, and their source provenance, measurement
dictionary, and redistribution authorization are unverified. Re-running the
current pipeline would not bring that material inside the present evaluation. It
is not evidence for any statement in this manuscript and it is not redistributed.

---

## 3. Methods

### 3.1 The predictor

ThermoRoute produces, for issue time *t* and lead *h*, a point prediction built
from four parts. A damped-persistence anchor moves the last observed daily water
temperature `y_t` toward a frozen seasonal climatology `c_{t+h}` at a fitted rate
and supplies the conservative reference trajectory. A learned flow- and
season-conditioned relaxation proposal modifies that decay behaviour; its
parameter is a fitted statistical quantity and is not interpreted as a measured
residence time or heat-transfer coefficient. A sparse, horizon-conditioned router
assigns weights over the seven variables at lags 0–14 following the sparsemax
construction ([Martins and Astudillo, 2016](https://proceedings.mlr.press/v48/martins16.html)),
and a strictly left-looking temporal convolutional encoder
([Bai et al., 2018](https://arxiv.org/abs/1803.01271)) with a regime mixture of
experts ([Shazeer et al., 2017](https://arxiv.org/abs/1701.06538)) encodes recent
history. Finally, the learned point correction is passed through `tanh` and
scaled by a fixed `delta_scale`, so its deviation from the anchor is
algebraically bounded.

Three clarifications matter for interpretation. First, "causal" in this paper
describes time ordering only — the encoder is non-anticipating — and never causal
inference. Second, the router is an allocation mechanism inside the predictor,
not a map of connected reaches; it receives no verified graph or topology input
and does not identify physical transport, travel time, residence time, or
regulation. Third, the bound is relative to the named anchor: it yields a finite
algebraic limit on the point output's deviation from damped persistence, and it
bounds neither absolute error, nor event-tail error, nor interval width, nor
behaviour after a distribution shift. The sequence builder supplies a 32-day
tensor, but this is a construction buffer rather than an effective-memory claim:
the current two-block, kernel-three encoder has a theoretical seven-step
receptive field and the router's oldest usable value is lag 14, so no input older
than lag 14 can affect the output.

The canonical configuration sets `delta_scale` to 1.0 °C. Because the 2019–2020
partition was inspected during development, this value is a development-selected
algebraic point bound and is not presented as a selection from an untouched
holdout.

### 3.2 A strong baseline suite

The most consequential methodological choice in this study is the composition of
the comparison set. Much of the applied river-temperature literature reports skill
relative to persistence, to climatology, or to an air–water regression. For a
variable with this much day-to-day memory, those references are too weak to
separate a model that has learned useful structure from a model that has learned
to be conservative. We therefore fixed, before any evaluation, a suite in which
each member removes a specific alternative explanation for apparent skill.

**Persistence** (`ŷ_{t+h} = y_t`) establishes the floor and quantifies the raw
memory of the series. **Damped persistence**, fitted on the training period,
relaxes `y_t` toward a frozen seasonal climatology and is the reference that
matters: it absorbs the two effects — inertia and seasonality — that a learned
model can most easily rediscover, and it is the reference against which the
primary hypotheses are stated. **Seasonal climatology** isolates the
seasonal-cycle component on its own. An **air2stream-style a4/a8 reference**,
fitted by deterministic six-start bounded least squares on measured drivers,
represents the hybrid air-temperature/discharge family that dominates the process
literature; we describe below why its current status limits what can be said
about it. **LightGBM** ([Ke et al., 2017](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html))
is the principal learned reference and is run in two configurations: a global
model that receives stable site identity as a categorical feature, and a
per-station variant that isolates the value of pooling. A **global LSTM** with a
station embedding represents the deep sequence family that has produced the
strongest recent results for this variable
([Rahmani et al., 2021](https://doi.org/10.1088/1748-9326/abd501);
[Zwart et al., 2023](https://doi.org/10.3389/frwa.2023.1184992)).

The learned references are given a genuine chance to win. The global LSTM uses
the same development splits, loss, station-balanced sampling, epoch and patience
limits, history length, site identity, and five-seed budget as ThermoRoute, and
its predeclared validation-only grid may expose the same frozen climatology,
damped-anchor, and season features. It is eligible for the same conformal and
probability calibration. It lacks only ThermoRoute's bounded-residual constraint
and lag router. LightGBM selects among four predeclared candidate settings
separately by lead on the 2016–2017 validation partition, using station-macro
RMSE alone; the LSTM selects among three predeclared architectures with seed 0
before fitting five members. These tuning budgets are documented and are *not*
identical across model classes, which is a real asymmetry and is stated as a
limitation rather than smoothed over. Consequently any statement about LightGBM
in this paper is scoped to this frozen procedure — four candidates, this feature
schema, this seed set — and is not a statement about gradient boosting in
general or about a larger search.

The air2stream-style reference requires a specific caveat. The implementation
used here is an unofficial style reference, not the official air2stream code and
not a validated reproduction of it, and in the current artifact set it is
recorded as `NOT_RUN`. A comparison that could speak to the published model would
require the official implementation and its documented calibration search. We
therefore report it as an unavailable, non-primary reference rather than
quietly dropping it from the suite.

Beyond the primary suite, seven one-factor architecture controls
(`DampedPriorOnly`, `TR-noDynamicPrior`, `TR-fixedKappa`, `TR-noRouter`,
`TR-noMoE`, `TR-noTCN`, `TR-unbounded`) are fitted with the same five seeds as
ThermoRoute and paired within seed on identical forecast keys and exact target
values before ensemble averaging. A second control family (a plain multilayer
perceptron and a plain causal temporal convolutional network) is
information-matched as well as parameter-matched: both receive the same
outcome-free issue-time history, station identity, climatology, frozen damped
anchor, standardized environmental context, and calendar and regime-gate tensors
available to the full model, and both predict unrestricted residuals around the
same anchor while omitting the relaxation proposal, router, mixture, and residual
bound. Neither ever reads the target tensor, the target-date field, or the
disabled water-level channel. These are deletion and intervention sensitivities.
They remove a direct information-set confound; they do not equalize historical
hyperparameter-search budgets, prove component necessity, or identify a
mechanism.

All comparisons use identical station/date/horizon keys for the two models being
compared. No model-specific complete-case set is permitted, so a model cannot
gain apparent accuracy by declining to predict on hard keys.

### 3.3 Leakage control

Leakage control in this study is a set of mechanisms, each of which can be
checked against the code and the artifacts, rather than a statement of intent.

*Issue-time separation of covariates.* Every predictor consumed by a primary
model carries a date no later than the issue date. No horizon-specific future
weather field enters any primary model. An optional secondary analysis using
archived GFS 2-m temperature composites is defined separately, is restricted by
the archive's own start date, and is explicitly outside the primary registry.

*Non-anticipating sequence encoding.* The temporal convolutional encoder uses
left-only padding, so output at position *t* is a function of positions ≤ *t*
by construction rather than by convention. The router's admissible lag range is
0–14 and the encoder's receptive field is seven steps, so the effective
information horizon is bounded and auditable, independent of the 32-day buffer
used to assemble tensors.

*Strictly backward-fitted statistics.* Standardization constants, imputation
fills, the seasonal climatology, the damped-persistence rate, the station-level
q90 event thresholds (2006–2015), the seasonal event reference (2006–2018), the
conformal offsets (2018), and the Platt calibrators (2018) are each fitted on
data strictly preceding the interval on which they are applied. In the temporal
arm these are station-specific and train-only; in the pooled arms they are pooled
train-only transforms. This closes the most common backward-information channel
in hydrological benchmarking, which is not the model but the preprocessing.

*Explicit admissibility.* A forecast key is admissible only when issue-date water
temperature is genuinely observed and a 32-day history can be constructed. Other
history cells may be filled with training-only seasonal medians and retain
explicit missingness masks in the sequence input; no minimum observed-history
fraction is imposed. We record one honest gap in this mechanism: the current
issue-date auxiliary quantities used by the learned relaxation proposal and the
regime gate (standardized forcings, discharge, and the water-temperature
tendency) are computed from the same training-only imputed panel but do not each
carry a separate auxiliary-path mask. The model sees the corresponding masks
through its sequence branch, but its output is not guaranteed invariant to the
chosen fill values. This is a declared design limitation, not a claim that the
auxiliary path is mask-isolated.

*Transform conventions that do not encode outcomes.* Discharge uses a signed
`log1p`, preserving observed negative values rather than clipping them or
relabelling them as drought; precipitation uses a nonnegative `log1p`.

*Replay under isolation.* Before any label acquisition is authorized, a fresh
isolated interpreter running under `python -I -B`, with network access, child
processes, repository writes, and reads from the evaluation namespaces all
denied, reloads every trained member and every prediction head and reproduces the
validation, calibration, and 2019–2020 development keys and values from the
frozen inputs. The replay maps historical panel aliases through the registry,
requires every selected station/target-date truth value to equal the frozen-panel
value, independently recomputes the q90 thresholds and seasonal references, and
refits the conformal and Platt parameters from the exact 2018 member-averaged
rows. A prediction file that is merely self-consistent, or a re-hashed
calibration object, is rejected. This is what prevents a rewritten artifact from
substituting invented truth or calibration parameters for the real ones.

### 3.4 Regional transfer design

Random held-site splits are the usual way to report spatial generalization in
this literature, and they leak. If a held-out gauge's neighbours remain in
training, the model can reach the held-out site's thermal regime through
correlated forcing, shared preprocessing statistics, and spatially smooth
learned representations. We therefore report two arms with different leakage
properties and label them differently.

The **random held-site warm-start** arm uses four folds in which held-site
histories still contribute to global panel preprocessing. This is the arm most
comparable to published random-split results, and precisely for that reason it is
not the arm we treat as the spatial test.

The **held-region gauged transfer** arm packs the 15 HUC2/unknown groups into
four folds of [30, 30, 31, 29] stations and holds out *whole regions*, so no
gauge from a held-out region appears in training. Station-agnostic ThermoRoute
and a global LightGBM are each trained on the in-fold regions and forecast the
held-out region's stations. All climatology, scaling, and damped-rate parameters
are pooled from in-fold stations only. The mean distance from a held-out station
to its nearest training gauge is 289 km.

The label on this arm must be exact, because the distinction is routinely blurred
in this literature. Held-out sites still provide their own observed water
temperature through the issue date, both as the persistence anchor and as the
sequence history. This is **gauged** transfer to a region whose gauges were not
used in fitting; it is **not** prediction at an ungauged location, and no
statement in this paper should be read as the latter. Ungauged prediction would
require the model to operate with no target-site water-temperature record at all,
which is a different problem and a different evaluation
([Weierbach et al., 2022](https://doi.org/10.3390/w14071032)). The same caveat
applies to the 30-site external cohort of Section 2.4, which is site-identifier
disjoint but still history-dependent.

### 3.5 Conformal intervals

Interval estimates are produced by split conformalized quantile regression
([Romano et al., 2019](https://papers.nips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html);
[Vovk et al., 2005](https://doi.org/10.1007/b106715)). The learned models emit a
separate mean-squared-error point head and three pinball-trained quantile heads
at levels 0.05, 0.50, and 0.95. Members are averaged with equal weights, and the
conformal offset is fitted on the 2018 calibration year only.

The construction is deliberately one-sided in its effect. The exact
split-conformal order statistic is retained as a signed raw audit value, but the
deployed offset is `qhat_plus = max(raw_qhat, 0)`, and the delivered interval is
`[q05 − qhat_plus, q95 + qhat_plus]` with the median left unchanged. Calibration
may therefore leave the nominal interval unchanged or widen it, but it can never
shrink it. A non-finite offset, or a non-finite, crossed, or empty final
interval, fails closed before any evaluation label is read. In the temporal arm
offsets are fitted per station and lead; in the pooled arm they are fitted per
lead only.

What this does and does not guarantee is worth stating precisely, because
conformal prediction is frequently over-claimed in applied work. Split conformal
provides finite-sample *marginal* coverage under exchangeability of the
calibration and evaluation nonconformity scores. Here the calibration scores come
from 2018 and the evaluation scores from a later, disjoint period at the same
sites, so exchangeability is not satisfied by construction: there is temporal
drift, and the scores are dependent within station and within region. We
therefore report achieved coverage as an *empirical marginal diagnostic* and make
no finite-sample guarantee. We make no conditional-coverage claim of any kind:
nothing here establishes that coverage holds within a season, within a
temperature regime, within a region, or at a particular station. Two
pre-specified sensitivities probe that boundary — a block-maximum calibration
variant that groups consecutive retained calibration rows, and an idealized
delayed adaptive-conformal variant that uses each forecast's target date as a
feedback-arrival proxy. Neither replays real feedback availability, because the
inputs contain no verified observation-publication timestamp, revision history,
data vintage, or reporting latency; both report empirical marginal coverage only.
The equal-weight three-quantile pinball summary is a three-quantile score and is
not called CRPS.

An event head reports exceedance of each station's 2006–2015 q90 water
temperature, calibrated by one Platt map per lead fitted on 2018 only, against a
seasonal event reference fitted on 2006–2018. That threshold is an absolute
statistical tail diagnostic local to each station. It has no biological,
ecological, or regulatory meaning and is not comparable across stations. Both the
Platt fit and the probability summaries give every retained station equal total
weight, so stations with more retained days cannot dominate the calibration map.

### 3.6 Estimand, comparison family, and the pre-specified inference gate

The sampling unit is the station. For each lead, unweighted RMSE is computed on
the common daily keys separately for each reportable station, and the primary
effect is the unweighted median across stations of the paired difference
`RMSE(ThermoRoute) − RMSE(reference)`. A station/lead cell is reportable only
with at least 100 valid paired targets. Daily rows therefore do not determine
between-station weight, and a station with a long record does not dominate the
estimand. This estimates performance conditional on observable issue and target
water temperature within the frozen availability-enriched cohort.

The frozen comparison family contains exactly five rows:

| # | Comparison | Lead | Predeclared margin |
|---|---|---:|---|
| 1 | ThermoRoute vs. damped persistence | 1 d | 0.00 °C |
| 2 | ThermoRoute vs. damped persistence | 3 d | 0.00 °C |
| 3 | ThermoRoute vs. damped persistence | 7 d | 0.00 °C |
| 4 | ThermoRoute vs. LightGBM | 3 d | +0.05 °C numerical ceiling |
| 5 | ThermoRoute vs. LightGBM | 7 d | +0.05 °C numerical ceiling |

The +0.05 °C ceiling was fixed before outcomes as a strict numerical limit on
allowable degradation. It is not derived from sensor precision, biological
response, a water-quality standard, or an elicited stakeholder utility, and it
carries no ecological or regulatory importance.

Two uncertainty procedures accompany each row, both clustered at HUC2. Because
the registry contains at most 15 clusters, the one-sided p-value is obtained by
*exact enumeration* of all 2^K whole-cluster sign vectors rather than by Monte
Carlo, applying one common sign to every station effect within a cluster. A
10,000-draw whole-HUC2 cluster bootstrap gives the percentile interval for the
median station effect, resampling complete clusters so that all station effects
in a sampled region are retained together. Holm adjustment
([Holm, 1979](https://www.jstor.org/stable/4615733)) covers exactly these five
p-values. Exact enumeration removes Monte Carlo error but does not make the
procedure distribution-free: it still assumes joint sign symmetry of each
complete cluster effect vector around the tested margin.

This is where the study's central methodological decision was taken, and taken
before any outcome was visible. An outcome-free amendment to the protocol
overlays the original decision rule with a gate on the cluster structure of the
cohort itself. The gate requires at least 30 reportable clusters, an
effective-cluster fraction of at least 0.75, and a largest-cluster share below
0.25. Missing, unknown, or failed components fail closed.

The frozen cohort cannot pass it. Before any reportability attrition, the 15 HUC2
groups contain between 2 and 26 stations; the largest holds 21.7% of stations;
and the inverse-Herfindahl effective cluster count is 9.54, an effective fraction
of approximately 0.636. The cluster-count component fails outright and the
effective-fraction component fails as well; only the largest-share component
would pass on its own, and the gate is a conjunction. The verdict is therefore
fixed by the geography of the frozen cohort, independently of the eventual
numbers, and is permanently `DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED`.

The reporting consequences are strict and are followed throughout Section 4.
Every one of the five rows is still rendered exactly once, including adverse,
conflicting, or non-estimable outcomes; the effect estimate, the win rate, the
station and cluster counts, and the interval are all shown. But the effect is a
fixed-cohort descriptive quantity, and the exact sign-flip p-value, the
cluster-bootstrap interval, and the Holm-adjusted value are retained solely as
assumption-conditional sensitivities. They are not decision evidence. No row may
be written as superiority, non-inferiority, equivalence, or parity, and no row
may be generalized to a national or U.S.-river population. Section 5.1 explains
why we consider this a feature of the study rather than a defect.

Secondary and exploratory analyses — a qualifier-restricted target sensitivity,
equal-HUC and leave-one-HUC influence summaries, station-balanced probability
diagnostics, architecture controls, and synthetic missingness, noise, flow, and
weather perturbations — are labelled as such and cannot promote or replace a
formal row. Relative economic value is recorded as
`REV_NOT_EVALUATED_NO_PREDECLARED_COST_LOSS_RATIOS`: no cost–loss ratios,
observed management costs, actions, or stakeholder utility model were frozen, so
no economic value is computed, and a future cost–loss grid without those observed
quantities would remain a hypothetical sensitivity.

A predeclared temporal coverage and calendar-balance audit accompanies each
formal row and reports eight fixed descriptive candidates: equal weighting of the
12 year-by-season cells, leave-one-year for each evaluation year, and
leave-one-season for DJF, MAM, JJA, and SON. The most adverse
candidate-minus-reference effect is taken as the deterministic unfavorable value,
with ties broken by frozen candidate order. The audit is restricted to keys where
both issue-date and target-date water temperature are retained finite
observations. It does not impute unavailable outcomes, test a missing-at-random
assumption, estimate all-calendar-day performance, or establish stability across
years or seasons, and a favourable sensitivity can never rescue an unfavourable
primary row.

### 3.7 The one-time 2021–2023 evaluation

The single strongest methodological property of this study is that the evaluation
labels are acquired exactly once, and that everything capable of being tuned is
frozen and independently verified before that acquisition is permitted.

The evaluation interval is 2021-01-01 through 2023-12-31 for the frozen 120-site
cohort. Those labels may not be read by model-selection, feature-selection,
threshold-selection, calibration, or station-inclusion code. Enforcement is
ordered, and the order is itself a gate. First, the protocol and its seal, the
claim renderer and validator, the source tree, the dependency locks, and all
temporal and station-agnostic model members are frozen and exactly replayed under
the isolation described in Section 3.3. Second, the model-suite registry is
committed *while* the external candidate registry and the evaluation-period
predictor artifacts are still absent from the repository. Only then may
metadata-only candidate discovery and retrospective predictor acquisition run.
Authorization replays the resulting Git ancestry and blob hashes and verifies
that the model-suite commit genuinely precedes the candidate metadata and the
evaluation-period predictor artifacts. Any source change after the model freeze
invalidates authorization. Failure of the chronology gate does not produce a
weaker result: it demotes the whole exercise to retrospective exploration and
forbids the confirmatory reading entirely.

The acquisition itself is deliberately narrow and deliberately irreversible. A
raw-only child process is restricted to USGS daily-value requests for daily mean
water temperature, discharge, and gage height. It records exact request and
response bytes, series identifiers, approval qualifiers, final URLs, retrieval
timestamps, and content hashes. Parameter and statistic identities are fixed in
advance as daily mean (`00003`) water temperature (`00010`, °C), discharge
(`00060`, cfs), and raw-only gage height (`00065`, ft). Every daily-mean series
column is retained. On a given date, zero finite series values means missing;
exactly one means that value, its qualifier, and its series identifier are used;
two or more finite series means the parameter/date is marked missing with an
explicit conflict code, with every constituent retained in the quality audit.
Values are never averaged or selected across conflicting series, and **no site
may be replaced after outcomes are accessed**.

The primary analysis uses every finite parsed daily-mean water-temperature and
discharge value regardless of approval qualifier, so qualifiers can never select
or remove a station, model, date, or forecast key. A separately labelled,
non-confirmatory sensitivity — fixed before opening — retains target keys only
when the target qualifier has the exact token set `{A}`, without refitting any
model. Counts by cohort, station, variable, raw qualifier string, and value
presence are reported immutably, and unknown qualifier strings are preserved and
counted rather than interpreted.

There is one logical opening and one fixed request ledger. This is not a claim of
exactly-once HTTP delivery, and we do not pretend otherwise: transport may retry,
and a response received before its transaction directory is durable may be
requested again. What is guaranteed is that a complete, durable, verifiable
canonical response is never replaced; that a partial, invalid, or non-canonical
transaction fails closed without overwrite; that cleanup is confined to
unpublished owner-private state; and that once the acquisition manifest exists,
raw network continuation is permanently disabled and only network-free
deterministic recomputation may proceed. Normalized tables and the manifest are
generated and validated in a private same-filesystem staging area and published
by a single directory rename, as is the trusted scoring layer. These are
honest-owner crash and replay guards; they are not protection against a malicious
owner or a same-UID adversary, and we say so rather than implying a security
property the design does not have.

The first successful scoring run writes a sealed result bound to the protocol
commit, the data manifest hash, the resolved configuration hash, and the
source-tree hash. A second scoring run after any model change is exploratory and
must carry a new run identity.

### 3.8 Pre-registration and reproducibility

The analysis protocol was written and frozen before any post-2020 outcome was
requested or inspected. It fixes the cohort and its hashes, the temporal roles,
the model registry, the estimand, the five-row comparison family and its margins,
the clustered p-value and interval procedures, the multiplicity correction, the
probabilistic and event contracts, the outcome quality-control policy, and the
decision rules — including the requirement that every row be reported exactly
once regardless of direction. Two later amendments are themselves outcome-free
and separately sealed: one narrows inferential wording through the cluster gate
of Section 3.6, and one replaces the original single-seed rule for the
architecture controls with the fixed five-seed matrix used here. A probability
metric erratum, also sealed before any label access, corrects the source heads
used for the three-quantile score. Protocol bytes are bound by SHA-256 to both
their original preregistration commit and the final pre-label commit. We state
the limit of that evidence plainly: the seals are repository-internal and assume
an honest owner; there is no external timestamp, public registration service, or
independent custodian, and no local seal protects against an owner rewriting
local history.

Reproducibility is receipt-based. Each training stage terminates in a
content-bound completion record covering its members, heads, predictions,
transforms, and lineage, and the model-suite freeze and the independent release
verifier both require the complete set; a missing, stale, incomplete, or
re-sealed partial record fails closed. Development reproduction is conditional on
the committed panel bytes, because the original provider responses for 2006–2020
were not retained (Section 2.2). The full receipt chain, the fully transitive
Python 3.12 dependency lock with package hashes, the environment probe, the
protocol and amendments, and the per-stage manifests are archived with the
software release (Section 8) and enumerated in the Supporting Information; the
main text does not list individual receipts, stage numbers, or run identifiers,
which belong in the archive rather than in the narrative.

---

## 4. Results

Section 4.1–4.5 report development-period quantities. **All of these are
exploratory.** The 2019–2020 partition participated in cohort construction and
has informed model and narrative development; it is not an independent test, and
no number in these subsections is decision-eligible. Section 4.6 contains the
formal rows, which are unfilled at the time of writing. Unless stated otherwise,
all development values are five-seed ensemble means on the 249,072 common keys,
with station-median RMSE in °C.

### 4.1 Point accuracy on the development partition (exploratory)

| Lead | Persistence | Damped persistence | air2stream-style | LightGBM | LSTM | ThermoRoute |
|---:|---:|---:|---:|---:|---:|---:|
| 1 d | 0.803 | 0.774 | NOT_RUN | 0.578 | 0.662 | 0.631 |
| 3 d | 1.576 | 1.406 | NOT_RUN | 1.280 | 1.323 | 1.291 |
| 7 d | 2.217 | 1.738 | NOT_RUN | 1.649 | 1.679 | 1.657 |

The shape of this table is the reason the baseline suite matters. Measured
against naive persistence, the ThermoRoute ensemble's median station skill is
+0.203, +0.187, and +0.251 at 1, 3, and 7 days. Measured against damped
persistence, the same quantities are +0.168, +0.076, and +0.038. Roughly
four-fifths of the apparent seven-day gain over persistence is attributable to
damping alone. A study reporting only the first row of numbers would describe the
same model very differently from one reporting both.

The tree ensemble has the lowest development-period station-median RMSE at all
three leads. The paired station-level differences (`ThermoRoute − LightGBM`) are
+0.046 °C at 1 day, +0.008 °C at 3 days, and −0.005 °C at 7 days, with
ThermoRoute win rates of 0.00, 0.40, and 0.60 across the 120 stations. The 1-day
win rate is reported as 0.00. We report this directly: on this partition,
the constrained architecture does not have an accuracy advantage over a
well-tuned tree ensemble, and its behaviour relative to that ensemble is
lead-dependent. Against damped persistence, the paired median differences are
−0.130, −0.104, and −0.062 °C with win rates of 0.86, 0.90, and 0.93.

The exploratory whole-HUC2 cluster-bootstrap intervals for those paired
differences are [−0.177, −0.081], [−0.126, −0.083], and [−0.071, −0.052] °C
against damped persistence, and [+0.043, +0.057], [+0.001, +0.020], and
[−0.010, +0.010] °C against LightGBM. These are development-period mirrors of the
frozen procedure. They carry the same 15-cluster structure that fails the gate of
Section 3.6, and every row of the development mirror is flagged
`NO_STRONG_INFERENCE` with the warnings `SMALL_CLUSTER_COUNT_LT_30` and
`LOW_EFFECTIVE_CLUSTER_FRACTION`.

### 4.2 Regional structure and heterogeneity (exploratory)

Development skill against persistence is not carried by a single region or by a
single size class. Region-weighted skill — the mean of the 15 per-HUC2 medians —
is +0.204, +0.189, and +0.253 at 1, 3, and 7 days, essentially unchanged from the
pooled medians of +0.203, +0.187, and +0.251. Per-region medians against
persistence at 1 day range from +0.131 (HUC2:10) to +0.285 (HUC2:18); at 7 days
from +0.217 (HUC2:06) to +0.290 (HUC2:09). Stratifying by drainage area into
three groups of roughly 39 stations gives 1-day medians of +0.190, +0.214, and
+0.231 from small to large. Against damped persistence, per-region 7-day medians
compress to a range of +0.012 to +0.077, which is the same message as Section
4.1 restated spatially.

### 4.3 Transfer arms (exploratory)

| Arm | Design | 1 d | 3 d | 7 d |
|---|---|---:|---:|---:|
| Temporal development | seen stations, 2019–2020 | +0.203 | +0.187 | +0.251 |
| Random held-site warm start | 4 folds; held-site history in preprocessing | +0.178 | +0.172 | +0.241 |
| Held-region gauged transfer | leave-HUC2-region-out | +0.155 | +0.116 | +0.155 |

*Median station skill versus persistence.* Against damped persistence the same
three arms give +0.168 / +0.076 / +0.038 (temporal), +0.145 / +0.061 / +0.030
(random held-site), and +0.147 / +0.086 / +0.075 (held-region).

The ordering is the expected one and is informative in itself: skill degrades
from the temporal arm to the random held-site arm and again to the held-region
arm. The gap between the second and third rows is a direct measure of how much a
random spatial split flatters a model on this panel.

In the held-region arm the global tree ensemble again attains lower station-median
RMSE than ThermoRoute (0.652 vs 0.676 at 1 day, 1.391 vs 1.428 at 3 days, 1.786
vs 1.860 at 7 days), with paired median differences of +0.031 [+0.023, +0.040],
+0.038 [+0.029, +0.050], and +0.067 [+0.049, +0.080] °C and ThermoRoute win rates
near 0.16. The global LSTM, with its station embedding disabled in this arm,
gives 0.679, 1.445, and 1.876. Every one of these arms retains issue-time
water-temperature history at the held-out sites, so none of them speaks to
ungauged prediction.

### 4.4 Interval behaviour (exploratory)

Split-conformal intervals on the 249,072 development keys attain an empirical
marginal coverage of 0.909 against a nominal 0.90, with a mean width of 3.87 °C
and a mean interval score of 4.93. Per lead, coverage is 0.905, 0.910, and 0.912
with widths of 2.01, 4.22, and 5.38 °C. On the warm-season tail slice defined by
training-period q90 exceedance (30,531 keys), coverage is 0.918 at a width of
3.50 °C.

Two pre-specified sensitivities show how sensitive these numbers are to the
calibration convention. Calibrating on the maximum nonconformity score within
blocks of seven consecutive retained calibration rows raises overall coverage to
0.981 but widens the interval to 5.74 °C — coverage bought with width, not with
sharpness. The idealized delayed adaptive-conformal variants track nominal
coverage more closely as the step size grows (0.903, 0.900, and 0.892 for
γ = 0.005, 0.02, 0.05) but produce unbounded widths in several slices, so their
interval scores are not finite. We report width and interval score beside every
coverage figure precisely so that adaptive calibration is not presented as free.
None of these figures is a conditional-coverage statement.

### 4.5 Component and input sensitivities (exploratory)

Deleting individual components changes development RMSE by small amounts:
removing the temporal encoder is the largest single effect (0.631 → 0.679 at 1
day, 1.657 → 1.675 at 7 days), while removing the router, the mixture, the
dynamic prior, or the residual bound moves the 1-day figure within ±0.005 °C of
the full model. Replacing everything by the damped anchor alone gives 0.774,
1.406, and 1.738. These are five-seed deletion sensitivities on paired keys; they
do not establish component necessity or a capacity-matched attribution.

Synthetic corruption of issue-time inputs, applied without changing the forecast
keys or the targets, orders the input channels by influence. Gaussian sensor
noise at 0.25 and 0.5 training standard deviations degrades 1-day RMSE from 0.631
to 1.621 and 2.999 °C (+147% and +360%). Blocks of missing forcing values of 3,
7, and 14 days cost about +0.13 °C at 1 day and about +0.04 °C at 7 days.
Air-temperature offsets of ±2 training standard deviations cost +0.11 to +0.15 °C
at 1 day. Multiplying discharge by 0.5 or 2 changes 1-day RMSE by at most
+0.003 °C. The predictor is, on this panel, dominated by the water-temperature
and air-temperature channels and is nearly insensitive to discharge
perturbations. These are synthetic data-corruption probes; they are not climate
projections, physically coherent scenarios, or deployment-safety tests.

Finally, the bounded-deviation contract was audited directly on the development
partition. The pointwise anchor contract holds on 100.00% of rows, the derived
pointwise error inequality holds on 100.00% of rows, the maximum absolute
correction is 1.0000 °C against the configured 1 °C bound, and the derived
station-by-lead RMSE inequality holds in all 360 station-by-lead cells. This
verifies only that the implementation cannot move farther than the configured
distance from its named anchor. It is not evidence of accuracy, extrapolation,
calibrated uncertainty, or safety.

### 4.6 Evaluation-period results

The following slots are rendered deterministically from the verified acquisition
receipt after the single opening described in Section 3.7. Each is reported
exactly once, including adverse, conflicting, and non-estimable outcomes, and
each carries the fixed-cohort descriptive verdict of Section 3.6. Handwritten
substitution into these slots is rejected by body hashes and evidence bindings.

**Table 4.1 — the five formal rows.** Fields: model pair, lead, predeclared
margin, reportable station count, reportable cluster count, median paired station
RMSE difference, whole-HUC2 cluster-bootstrap 95% interval, win rate, exact
sign-flip raw p-value, Holm-adjusted p-value, and the bound gate verdict.

| # | Comparison | Lead | Margin | Effect and interval | Sensitivities (p_raw / p_Holm) | Verdict |
|---|---|---:|---|---|---|---|
| 1 | ThermoRoute vs. damped persistence | 1 d | 0.00 °C | `[TO BE FILLED AFTER OPENING]` | `[TO BE FILLED AFTER OPENING]` | `DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED` |
| 2 | ThermoRoute vs. damped persistence | 3 d | 0.00 °C | `[TO BE FILLED AFTER OPENING]` | `[TO BE FILLED AFTER OPENING]` | `DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED` |
| 3 | ThermoRoute vs. damped persistence | 7 d | 0.00 °C | `[TO BE FILLED AFTER OPENING]` | `[TO BE FILLED AFTER OPENING]` | `DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED` |
| 4 | ThermoRoute vs. LightGBM | 3 d | +0.05 °C | `[TO BE FILLED AFTER OPENING]` | `[TO BE FILLED AFTER OPENING]` | `DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED` |
| 5 | ThermoRoute vs. LightGBM | 7 d | +0.05 °C | `[TO BE FILLED AFTER OPENING]` | `[TO BE FILLED AFTER OPENING]` | `DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED` |

**Table 4.2 — all-model scores on the evaluation common keys.**
`[TO BE FILLED AFTER OPENING]` — station-median RMSE for every primary model at
every lead on the exact common key registry, with the reportable station count
and the air2stream-style reference resolved as either fitted or unavailable.

**Table 4.3 — temporal coverage and calendar balance.**
`[TO BE FILLED AFTER OPENING]` — for each of the five rows, all eight predeclared
descriptive candidates (equal-weighted 12 year-by-season cells; leave-one-year
for 2021, 2022, 2023; leave-one-season for DJF, MAM, JJA, SON) and the
deterministic most-adverse value.

**Table 4.4 — interval behaviour on the evaluation keys.**
`[TO BE FILLED AFTER OPENING]` — achieved empirical marginal coverage, mean
width, and interval score by model and lead, with the block-maximum and delayed
adaptive-conformal sensitivities.

**Table 4.5 — outcome quality control.**
`[TO BE FILLED AFTER OPENING]` — immutable counts by cohort, station, variable,
raw qualifier string, and value presence; multiple-finite-series conflict counts;
and the qualifier-restricted `{A}` sensitivity for each formal row.

**Table 4.6 — held-region and external arms on the evaluation keys.**
`[TO BE FILLED AFTER OPENING]` — leave-HUC2-region-out results and the 30-site
site-identifier-disjoint arm, both labelled exploratory and both
history-dependent.

**Probabilistic metric suite — not reported.** The pre-registered probabilistic
metric stage does not execute on the frozen artifacts, for the reason given in
Section 6.3, and its downstream analysis stage inherits the same condition.
Interval evidence is reported from the split-conformal analysis instead.

---

## 5. Discussion

### 5.1 A pre-specified gate that failed is a result, not an excuse

The central inferential fact of this study is that we wrote down, before seeing
any evaluation outcome, a condition under which our own headline comparisons
would not be allowed to carry inferential weight — and that condition was not
met.

The condition is about cohort geometry, not about performance. Clustered
inference on a panel of gauges rests on resampling or sign-flipping whole
clusters, and the accuracy of those procedures depends on having enough clusters,
of reasonably comparable size, that the asymptotics they invoke are not a
fiction. Fifteen HUC2 groups with an inverse-Herfindahl effective count of 9.54
and a largest group holding 21.7% of stations is not enough. This was
determinable from the registry alone. It required no outcome, and it could have
been determined by anyone with the station list.

That is exactly why we consider stating it a contribution. The usual practice in
applied water-temperature machine learning is to report a station-level or
row-level confidence interval, or a paired test across sites, without addressing
whether the number of independent spatial units can carry the procedure. Under
that practice, a cohort like ours would produce intervals and p-values that look
authoritative, and nothing in the paper would tell a reader that the effective
number of clusters is under ten. The difference between that paper and this one
is not the data and not the model; it is whether the limit is disclosed. We think
the disclosed version is more useful, and we would rather publish a benchmark
with an honest scope than an inference with a hidden denominator.

The practical consequence is that all five formal rows are fixed-cohort
descriptive effects. The numbers are reported in full, because withholding them
would be its own distortion, and the clustered p-values, bootstrap intervals, and
Holm adjustments are reported alongside them as assumption-conditional
sensitivities — quantities that describe what would follow *if* the cluster
sign-symmetry and resampling assumptions held, which we do not assert. What they
are not is decision evidence, and no favourable point estimate, narrow interval,
or small p-value can convert them into decision evidence after the fact. The
verdict was determined by the cohort before the outcomes existed, and it is not
reopened by them.

There is a second instance of the same principle in this paper, described in
Section 6.3, in which a pre-registered data contract rejected an output that we
would have preferred to report. We left that contract alone as well. A
pre-registration that never binds against its authors' convenience has not been
tested; these two cases are the evidence that this one has.

The corresponding design lesson is concrete and transferable. If a study intends
region-clustered inference over U.S. gauges, the cluster structure must be part
of the *sampling design*, not a post-hoc grouping of whatever cohort availability
produced. Reaching 30 or more effective clusters requires either a finer spatial
partition with a defensible independence argument, or a deliberately stratified
draw across many more regions, and either choice must be made — and its power
computed — before any outcome is accessible. The alternative route, of enlarging
the cohort until the gate passes, was considered for this submission and set
aside; retrofitting it onto the present frozen cohort would have meant choosing
the cluster definition after seeing which definition helped, which is precisely
the failure mode the gate exists to prevent.

### 5.2 What the development evidence does and does not indicate

Read as description of these 120 gauges over 2019–2020, the development results
are informative in three ways and silent in a fourth.

They indicate, first, that the reference set determines the story. The same
ensemble that shows median skill of +0.251 against persistence at 7 days shows
+0.038 against damped persistence. Both numbers are correct. Only the second
answers the question a reader of a modelling paper actually has.

Second, they indicate that a well-tuned gradient-boosted tree ensemble with site
identity is a demanding reference on this panel: it attains the lowest
station-median RMSE at all three leads in the temporal arm and in the
held-region arm, with a reported ThermoRoute win rate of 0.00 at 1 day in the
temporal arm. Reports in which a
proposed architecture is compared only to persistence, climatology, or an
air–water regression should be read with this in mind. The constrained
architecture's practical argument here is not accuracy but the algebraic bound on
its deviation from an interpretable anchor, which held in 100% of audited rows
and in all 360 station-by-lead cells, and which is a property of the construction
rather than a fitted outcome.

Third, they indicate that the spatial evaluation convention matters. Median skill
against persistence drops from +0.187 to +0.172 to +0.116 at 3 days as the design
moves from temporal, to random held-site, to whole-region holdout, with a mean
nearest-training-gauge distance of 289 km in the last. A random spatial split on
a panel where 19 stations have a neighbour within 10 km is measuring something
closer to interpolation than to transfer.

They are silent on the question of prediction without target-site records. Every
arm in this study, including the held-region and the site-identifier-disjoint
arms, consumes the target site's observed water temperature through the issue
date, as anchor and as history. The problem of predicting at a location with no
thermal record is a different problem, and this design cannot address it.

### 5.3 Relation to prior work

The comparison this study makes is narrow by construction. It is a common-key,
clustered, pre-registered benchmark of a bounded correction against strong
statistical and learned references at gauged sites. It is not an attempt to
replace process-based thermal models, river-network graph models, or
differentiable hybrid formulations
([Jia et al., 2021](https://doi.org/10.1137/1.9781611976700.69);
[Rahmani et al., 2023](https://doi.org/10.1029/2023WR034420);
[Zwart et al., 2023](https://doi.org/10.3389/frwa.2023.1184992)), each of which
addresses structure — connectivity, upstream forcing, energy balance — that a
point-scale statistical predictor does not represent. Nor does the learned
relaxation proposal recover any of that structure: it receives no verified graph
or topology input and identifies no transport, travel time, residence time, or
regulation.

Where we think the design is most portable is in the three controls that do not
depend on the architecture at all: scoring every model on an identical key
registry so that no model benefits from selective prediction; fitting every
preprocessing and calibration statistic strictly backwards in time; and fixing
the analysis, the margins, and the reporting rules before the evaluation labels
exist. None of the three requires unusual computational resources. Together they
change what a reported skill score means.

---

## 6. Limitations

### 6.1 Machine-verifiable scope statements

The following statements are part of the evidence contract and remain in the
manuscript regardless of the eventual numerical outcome.

<!-- ROUTE_A_CLAIM LIMIT_NOT_UNGAUGED sha256=187d9b062cd6d02d17d3c86b3c0401da96ae3b0d2e2b6aa5d24cd53d0c34c661 -->
Route A is history-dependent and uses target-site water-temperature observations through each issue date; it does not establish ungauged prediction.
<!-- ROUTE_A_CLAIM_ENTRY {"claim_id":"LIMIT_NOT_UNGAUGED","evidence":{"artifact":"protocol","constraint_id":"P01_NOT_UNGAUGED"},"kind":"NEGATED_LIMITATION","phase_allowed":["PRE_CONFIRMATION_LABELS_SEALED","POST_CONFIRMATION_VERIFIED"],"polarity":"NEGATED","render_targets":["paper/ThermoRoute_paper.md"],"scope":"Route-A external history-dependent new-gage evaluation","template_id":"NEGATED_LIMITATION_NOT_UNGAUGED"} -->
<!-- END ROUTE_A_CLAIM -->

<!-- ROUTE_A_CLAIM LIMIT_NO_NETWORK_ROUTING_PROOF sha256=03d1ca5c0805e6bcdd3a64583b34c7af94f7f11c76a6e0ac2cf208f4c0ac4bfb -->
Route A does not establish independent river-network separation, physical river-network routing, or hydraulic travel time.
<!-- ROUTE_A_CLAIM_ENTRY {"claim_id":"LIMIT_NO_NETWORK_ROUTING_PROOF","evidence":{"artifact":"protocol","constraint_id":"P02_NO_NETWORK_ROUTING_PROOF"},"kind":"NEGATED_LIMITATION","phase_allowed":["PRE_CONFIRMATION_LABELS_SEALED","POST_CONFIRMATION_VERIFIED"],"polarity":"NEGATED","render_targets":["paper/ThermoRoute_paper.md"],"scope":"Route-A physics-inspired architecture and HUC2 sensitivity","template_id":"NEGATED_LIMITATION_NO_NETWORK_ROUTING_PROOF"} -->
<!-- END ROUTE_A_CLAIM -->

<!-- ROUTE_A_CLAIM LIMIT_NOT_OPERATIONAL_REPLAY sha256=69bf41372c0a367028673b4725f9e3a139494c12457570c44425425544ebb0c0 -->
Route A is a one-shot retrospective historical-information evaluation, not an operational replay with archived as-issued predictor vintages or future NWP.
<!-- ROUTE_A_CLAIM_ENTRY {"claim_id":"LIMIT_NOT_OPERATIONAL_REPLAY","evidence":{"artifact":"protocol","constraint_id":"P03_NOT_OPERATIONAL_REPLAY"},"kind":"NEGATED_LIMITATION","phase_allowed":["PRE_CONFIRMATION_LABELS_SEALED","POST_CONFIRMATION_VERIFIED"],"polarity":"NEGATED","render_targets":["paper/ThermoRoute_paper.md"],"scope":"Route-A retrospective forecast evaluation","template_id":"NEGATED_LIMITATION_NOT_OPERATIONAL_REPLAY"} -->
<!-- END ROUTE_A_CLAIM -->

<!-- ROUTE_A_CLAIM LIMIT_NO_REGULATORY_ECOLOGICAL_MEANING sha256=f485bcba32456a7272600b4e71fa3d5b1bc4da5dce8d72daf27476318a7455f7 -->
Route A uses daily-mean statistical thresholds and a numerical non-inferiority margin; neither has ecological, biological, or regulatory meaning.
<!-- ROUTE_A_CLAIM_ENTRY {"claim_id":"LIMIT_NO_REGULATORY_ECOLOGICAL_MEANING","evidence":{"artifact":"protocol","constraint_id":"P04_NO_REGULATORY_ECOLOGICAL_MEANING"},"kind":"NEGATED_LIMITATION","phase_allowed":["PRE_CONFIRMATION_LABELS_SEALED","POST_CONFIRMATION_VERIFIED"],"polarity":"NEGATED","render_targets":["paper/ThermoRoute_paper.md"],"scope":"Route-A daily-mean statistical thresholds and numerical margin","template_id":"NEGATED_LIMITATION_NO_REGULATORY_ECOLOGICAL_MEANING"} -->
<!-- END ROUTE_A_CLAIM -->

<!-- ROUTE_A_CLAIM LIMIT_NO_SAFETY_GUARANTEE sha256=2af26c4eccae44cb3cc411411d34c264a22334cd0e0b35f9e654f93fcc8fb4c2 -->
Route A provides no physical, deployment, regulatory, or distribution-free safety guarantee, and failure to reject is not equivalence.
<!-- ROUTE_A_CLAIM_ENTRY {"claim_id":"LIMIT_NO_SAFETY_GUARANTEE","evidence":{"artifact":"protocol","constraint_id":"P05_NO_SAFETY_GUARANTEE"},"kind":"NEGATED_LIMITATION","phase_allowed":["PRE_CONFIRMATION_LABELS_SEALED","POST_CONFIRMATION_VERIFIED"],"polarity":"NEGATED","render_targets":["paper/ThermoRoute_paper.md"],"scope":"Route-A bounded residual and confirmatory decisions","template_id":"NEGATED_LIMITATION_NO_SAFETY_GUARANTEE"} -->
<!-- END ROUTE_A_CLAIM -->

<!-- ROUTE_A_CLAIM LIMIT_NO_CAUSAL_MECHANISM sha256=71f71405c7dcaa360e28348f686b6aa41f74166553071ba75185c60bbe30a3f5 -->
Route-A architecture controls and predictor sensitivities are descriptive or exploratory and do not identify causal mechanisms.
<!-- ROUTE_A_CLAIM_ENTRY {"claim_id":"LIMIT_NO_CAUSAL_MECHANISM","evidence":{"artifact":"protocol","constraint_id":"P06_NO_CAUSAL_MECHANISM"},"kind":"NEGATED_LIMITATION","phase_allowed":["PRE_CONFIRMATION_LABELS_SEALED","POST_CONFIRMATION_VERIFIED"],"polarity":"NEGATED","render_targets":["paper/ThermoRoute_paper.md"],"scope":"Route-A architecture controls and predictor sensitivities","template_id":"NEGATED_LIMITATION_NO_CAUSAL_MECHANISM"} -->
<!-- END ROUTE_A_CLAIM -->

<!-- ROUTE_A_CLAIM LIMIT_NOT_NATIONALLY_REPRESENTATIVE sha256=fd2f6b595b45a9c38c2236da5dca8fccbfb12ff0ad785ee1e6e8e9ecc2633901 -->
Route A evaluates a fixed availability-enriched 120-site cohort and is not nationally representative of all U.S. rivers or all calendar days.
<!-- ROUTE_A_CLAIM_ENTRY {"claim_id":"LIMIT_NOT_NATIONALLY_REPRESENTATIVE","evidence":{"artifact":"protocol","constraint_id":"P07_NOT_NATIONALLY_REPRESENTATIVE"},"kind":"NEGATED_LIMITATION","phase_allowed":["PRE_CONFIRMATION_LABELS_SEALED","POST_CONFIRMATION_VERIFIED"],"polarity":"NEGATED","render_targets":["paper/ThermoRoute_paper.md"],"scope":"Route-A fixed availability-enriched site cohort","template_id":"NEGATED_LIMITATION_NOT_NATIONALLY_REPRESENTATIVE"} -->
<!-- END ROUTE_A_CLAIM -->

<!-- ROUTE_A_CLAIM LIMIT_NO_CONDITIONAL_COVERAGE_OR_CRPS_CLAIM sha256=3a2e5070e24b899f28d35bdfccdb24175836b334f4f1bde6ce1b4ad129ab94c1 -->
Route A does not establish conditional coverage, and its equal-weight three-quantile pinball summary is not CRPS.
<!-- ROUTE_A_CLAIM_ENTRY {"claim_id":"LIMIT_NO_CONDITIONAL_COVERAGE_OR_CRPS_CLAIM","evidence":{"artifact":"protocol","constraint_id":"P08_NO_CONDITIONAL_COVERAGE_OR_CRPS_CLAIM"},"kind":"NEGATED_LIMITATION","phase_allowed":["PRE_CONFIRMATION_LABELS_SEALED","POST_CONFIRMATION_VERIFIED"],"polarity":"NEGATED","render_targets":["paper/ThermoRoute_paper.md"],"scope":"Route-A descriptive probability evaluation","template_id":"NEGATED_LIMITATION_NO_CONDITIONAL_COVERAGE_OR_CRPS_CLAIM"} -->
<!-- END ROUTE_A_CLAIM -->

### 6.2 Cohort, measurement, and design limitations

The station sample is availability-enriched rather than randomly drawn, and the
coverage thresholds that produced it used the 2019–2020 interval, so that
interval is not independent of cohort construction. The original provider bytes
for the 2006–2020 panel are unavailable, so development reproduction begins from
the committed derived artifact and the 1,465-candidate discovery execution is
auditable but not replayable. The panel retains no NWIS qualifier columns, method
or sensor history for 2006–2020, so measurement discontinuities cannot be
reconstructed for the development record. Meteorology is represented at the
station coordinate rather than integrated over the contributing catchment, which
is least appropriate for the largest basins in the cohort. Two stations contain
2,059 signed negative discharge rows whose semantics are unresolved. Outcomes are
daily means, and the evaluation is conditioned on outcome observability. The
auxiliary prior and gate features are computed from a training-only imputed panel
without separate path-specific validity flags, so output invariance to fill
values is not guaranteed. HUC2 is a coarse administrative grouping and is not an
independent river-network component. The +0.05 °C margin is a numerical ceiling
with no stakeholder-derived importance. Covariates are retrospectively acquired
latest-provider values rather than as-issued vintages. Model-selection budgets
were documented but not equalized across model classes, so any statement about a
reference model is scoped to its frozen procedure. No latency, memory, energy, or
multi-hardware benchmark is reported. The evidence chronology is
owner-controlled, with no external timestamp or independent custodian. The
air2stream-style reference has not been fitted and no official air2stream
calibration run exists.

### 6.3 The probabilistic metric suite is not reported

The pre-registered probabilistic contract treats a zero-width nominal prediction
interval as a fatal condition and prohibits evaluation-time repair. In the
development panel, 135 of 26,993,675 member-level rows carrying complete quantile
heads — 0.0005%, or about one row in 200,000 — have identical 5th, 50th, and 95th
percentile predictions. All 135 come from the tree ensembles (123 from the global
model, 12 from the per-station variant); no other model is affected. We emphasise
what the artifact is *not*: there is no strict quantile-ordering violation
anywhere in the panel, and the maximum monotonicity violation is exactly
0.000 °C. The rejected rows are correctly ordered but degenerate.

Their distribution is interpretable, and we flag the reading that follows as
interpretation rather than as a measured attribution. The 135 rows fall on 12
distinct sites, all in hydrologic regions 04, 05, and 06 — the Great Lakes, upper
Mississippi, and Missouri basins — and 129 of them occur at the 1-day lead, 6 at
3 days, and none at 7 days. In 95.6% of them the observed target is below 1 °C.
This is the signature of ice-affected winter conditions, in which daily mean
water temperature is pinned near the freezing point and the conditional 5th,
50th, and 95th percentiles genuinely coincide: for example site 04027000 at a
1-day lead, with all three quantiles equal to 0.0010 °C and to −0.0191 °C on
different dates. A zero-width interval is arguably the correct answer under those
conditions rather than a broken one. The 12 per-station cases, such as site
01542500 with all three quantiles equal to 13.1107 °C, are constant-leaf
predictions from a model fitted on a small station-specific sample, which is an
expected property of per-station gradient boosting. The model outputs are
defensible; it is the contract's inclusive equality test that refuses them.

We did not amend the contract. It was frozen before any access to the evaluation
labels, and relaxing it would have invalidated the frozen training receipts on
which the chronology gate of Section 3.7 depends. The pre-registered
probabilistic metric suite — interval coverage from the nominal heads,
three-quantile pinball mean, reliability, and Brier score — is therefore not
reported, and the analysis stage that consumes it is likewise not reported. Two
things follow. First, interval validity in this paper rests entirely on the
split-conformal analysis of Sections 3.5 and 4.4, which is unaffected: the
conformal offset is non-negative by construction, so the delivered interval for
every one of those 135 rows is strictly wider than the degenerate nominal one,
and the deployed contract separately requires positive width. No delivered
interval is degenerate. Second, event-probability metrics — Brier score,
reliability, discrimination, and calibration slope and intercept — are absent
from this manuscript, and no claim about probabilistic calibration beyond
empirical marginal interval coverage should be read into it. Resolving this
requires a sealed erratum and a new training lineage, which is future work rather
than a repair applied after seeing which resolution was convenient.

### 6.4 Scope of the observed-threshold analysis

An optional threshold analysis is separate from prediction evaluation. It can
only compare independently sourced observed daily maxima, expressed as a strict
seven-consecutive-day average of daily maxima, against a complete site-specific
standards registry, with seasonal applicability assigned by each window's ending
date. Season-external or incomplete windows remain unclassified, and missing,
unrelated, or ambiguous standard matches fail closed. Any reported exceedance is
descriptive: it is an observation, not a model result and not a legal
determination, and it does not establish compliance with any law or regulation.

### 6.5 Archive scope

The active release namespace excludes withdrawn generated outputs and rendered
manuscripts, but the self-contained history bundle deliberately retains reachable
deleted objects so that chronology can be audited. It is not a byte-level purge.
Those historical objects are provenance rather than current evidence and require
a separate licence and privacy review before public redistribution. The current
archive profile is machine-marked local-owner evidence: public redistribution and
third-party transfer are disabled and the public build mode fails closed pending
a byte-bound rights review. This profile must not be read as a public software or
data release.

---

## 7. Conclusions

We have described a river water-temperature hindcasting benchmark whose design
choices are aimed at a single problem: making a reported skill number mean what a
reader assumes it means. The comparison set includes damped persistence, a tuned
gradient-boosted tree ensemble with and without site identity, and a global LSTM,
all scored on identical station/date/horizon keys, because for a variable this
persistent a weaker reference set does not discriminate. The information boundary
is enforced by construction — issue-time-dated covariates, a non-anticipating
encoder with a bounded lag range, and every preprocessing and calibration
statistic fitted strictly backwards in time — and verified by an isolated replay
that cannot reach the evaluation labels. Spatial generalization is reported from
whole-region holdout and labelled as gauged transfer, because issue-time water
temperature remains available at held-out sites. Intervals are split-conformal
and are reported as empirical marginal diagnostics with width and interval score
attached, with no conditional-coverage claim.

On the development partition, and as description of these 120 gauges only, the
ensemble's median station skill against damped persistence is +0.168, +0.076, and
+0.038 at 1, 3, and 7 days, against +0.203, +0.187, and +0.251 versus naive
persistence; the tree ensemble attains the lowest station-median RMSE at all
three leads; and skill degrades monotonically from the temporal arm through the
random held-site arm to the whole-region holdout. The evaluation-period rows
remain unfilled until the single acquisition described in Section 3.7 occurs.

Irrespective of those numbers, the formal comparisons will be fixed-cohort
descriptive effects. The pre-specified inference gate on cohort cluster structure
fails on a cohort with at most 15 HUC2 groups and an effective cluster count near
9.54, and that determination was made from the registry before any outcome
existed. We report it as a finding about the design space of this literature: a
benchmark of this size cannot carry region-clustered inference about rivers in
general, and reaching that regime requires the cluster structure to be part of
the sampling design rather than a grouping applied afterwards.

---

## 8. Open Research

**Data availability.** The derived daily panel (`panel_usgs_120v2.parquet`), the
stable station registry, the hydrologic-unit metadata snapshot, and the frozen
panel manifest that binds them are archived at
[DOI: 10.5281/zenodo.XXXXXXX] under [DATA LICENCE — to be assigned after the
byte-level rights review described in Section 6.5]. The evaluation-period
acquisition record — raw request and response bytes, series identifiers, approval
qualifiers, retrieval timestamps, and content hashes — is archived in the same
deposit after the acquisition described in Section 3.7.

Primary observations are redistributed from public providers and are also
available at source: daily-value water temperature and discharge from the U.S.
Geological Survey National Water Information System
([https://doi.org/10.5066/F7P55KJN](https://doi.org/10.5066/F7P55KJN));
meteorological fields from Daymet V4 at ORNL DAAC
([https://doi.org/10.3334/ORNLDAAC/2129](https://doi.org/10.3334/ORNLDAAC/2129));
and wind speed from gridMET
([https://doi.org/10.1002/joc.3413](https://doi.org/10.1002/joc.3413)). Original
provider responses and retrieval timestamps for the 2006–2020 development panel
were not retained and cannot be reconstructed, so development reproduction begins
from the committed derived artifact; this limitation does not apply to the
evaluation-period acquisition, for which exact bytes are archived.

**Software availability.** The analysis software, the frozen protocol and its
sealed amendments, the model-suite registry, the completion receipts, the
per-stage manifests, and the fully transitive Python 3.12 dependency lock with
package hashes are archived at [DOI: 10.5281/zenodo.YYYYYYY], corresponding to
release [vX.Y.Z] of the source repository at [REPOSITORY URL], under
[SOFTWARE LICENCE — SPDX identifier]. The deposit includes the environment probe,
the verification entrypoints, and the deterministic result renderer that
generates Section 4.6 from the acquisition receipt. Third-party components
redistributed within the archive are enumerated with their own licence terms in a
notice file; components whose redistribution terms are unresolved are excluded
from the public deposit rather than shipped under an assumed licence.

**Reproduction.** Section 4.1–4.5 are reproducible from the archived panel, the
archived model bundles, and the pinned environment. Section 4.6 is reproducible
from the archived acquisition record and the same bundles. Bit-level equality is
expected for the tree models and agreement to the documented tolerance for the
neural members; the tolerances, the verification commands, and the expected
digests are listed in the Supporting Information.

**Placeholders.** All bracketed identifiers above — DOIs, repository URL, release
tag, and licence identifiers — are unassigned at the time of writing and must be
completed, with verified author metadata and contributor roles, before
submission.

## Acknowledgments

[To be completed: funding sources with award numbers, computational resources,
data provider acknowledgments, and a competing-interests statement. Author
contributions to be recorded using CRediT taxonomy terms.]

## Supporting Information

Supporting Information accompanies this manuscript and contains: the cohort and
registry description with the candidate-rejection ledger (SI01); the issue-time
information boundary and the product-compatibility bridge (SI02); model equations
and unit conventions (SI03); the sealed protocol, its amendments, and the
probability-metric erratum (SI04); the registered five-row family and its fields
(SI05–SI06); all-model scores on the exact common keys (SI07); probability and
reliability schemas together with the non-reporting disposition of Section 6.3
(SI08); architecture and information-matched controls with seed and budget
provenance (SI09); the temporal coverage audit (SI10); spatial and leave-cluster
sensitivities (SI11); outcome quality control and qualifier evidence (SI12); the
history-dependent external arm (SI13); missingness and failure cases (SI14);
reproduction hashes, commands, and environment parity fields (SI15); and the
rights and data dictionary (SI16). Supporting figures S1–S8 accompany the
corresponding sections.

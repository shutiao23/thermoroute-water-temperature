# ThermoRoute Route-A confirmatory protocol

Protocol status: **frozen before acquisition or inspection of post-2020 outcome data**.

This document separates the already-inspected 2006--2020 development record from
the Route-A confirmation exercise.  The 2019--2020 partition is exploratory: it
has previously informed model and narrative development and must not be described
as a blind or one-shot test.

## 1. Frozen development cohort and identities

- Canonical development panel:
  `data_usgs/panel_usgs_120v2.parquet`
  (`sha256=0427a07ea4514ba29ce7d0cf89594e6c35c7f9134cc4d1d96fdc90daeaf5ba69`).
- Canonical station metadata:
  `data_usgs/stations_meta_120v2.csv`
  (`sha256=92449bde5760ba82c53c61a67bc56240b4f76c52ea3425af320f09792ec7a33a`).
- Reproducibility of development training is conditional on those frozen panel
  bytes.  The original WTEMP/FLOW/Daymet/gridMET HTTP responses, request records,
  and retrieval timestamps used to build this legacy 2006--2020 panel were not
  retained.  They cannot be reconstructed retroactively, so Route A does not call
  development-data preparation a source-level replay.  This limitation does not
  relax the raw-snapshot requirements for newly acquired confirmation inputs.
- Station identity is the zero-padded USGS `site_no`.  The legacy `n00`--`n119`
  values are aliases only and may never be joined without checking `site_no`.
- The development partitions remain train 2006--2015, validation 2016--2017,
  calibration 2018, and exploratory evaluation 2019--2020.  A sample is admitted
  only if its issue date and every target date remain inside the same partition.

## 2. Untouched temporal confirmation set and input-availability audit

- The sealed outcome/history acquisition period remains 2021-01-01 through
  2023-12-31 for the frozen 120-site cohort.  These labels must not be read by
  model-selection, feature-selection, threshold-selection, calibration, or
  station-inclusion code.
- The primary target registry still starts on 2021-01-01.  Primary models use
  only WTEMP/FLOW history and retrospective gridded TEMP/PRCP/RHMEAN/DH/
  WDSP values dated on or before the issue date.  They do **not** consume a
  horizon-specific future-NWP field.  NWIS gage height (`WLEVEL`) may be retained
  in a raw acquisition response for provenance, but it is not in the frozen
  seven-variable model feature order and may not enter a Route-A prediction.
- The gridded meteorology is a retrospectively retrieved historical product.  Its
  as-issued provisional vintage at each historical issue date cannot be
  reconstructed.  Raw or normalized responses, retrieval times, request details,
  and checksums freeze the one-shot dataset actually evaluated; they do not prove
  that the same values were operationally available then.  Route A is therefore
  a one-shot retrospective historical-information evaluation, not a complete
  operational replay.
- The legacy column `DH` is Daymet `srad`: incident shortwave flux averaged over
  the daylight period (W m-2), not a 24-hour mean or daily energy total.
  `RHMEAN` is a reproducible vapour-pressure/Tetens proxy evaluated at the
  tmax/tmin midpoint, not a direct daily-mean RH observation.  gridMET `WDSP`
  is decoded from the packed NCSS integer only after the frozen OPeNDAP metadata
  proves units `m/s`, `scale_factor=0.1`, and `add_offset=0`.  These variables
  may be predictors; their learned weights are not an identified energy budget.
- **Secondary pre-label availability audit (2026-07-21):** official source
  documentation says the GFS 2-m temperature archive begins in March 2021.  The
  archive-run start is 2021-03-23; consequently the first valid dates for 1-, 3-,
  and 7-day offsets are 2021-03-24, 2021-03-26, and 2021-03-30.  This limits only
  an optional NWP availability/sensitivity table to target dates from 2021-03-30;
  it does not alter the primary registry or become a label-opening requirement.
- If that secondary artifact is acquired, it uses the Open-Meteo Previous Runs
  API with NOAA NCEP GFS global (`models=gfs_global`) and only
  `temperature_2m_previous_day1`, `...day3`, and `...day7`, in GMT and Celsius.
- A `previous_dayN` value is aligned separately at every valid hour to the value
  forecast N x 24 hours earlier.  The daily predictor is the arithmetic mean of
  exactly 24 such valid-hour values and is available by 23:59 UTC on the issue
  date.  It is a rolling fixed-lead composite, **not** one identified model-run
  initialization and not a complete operational forecast replay.
- Secondary-NWP requests are partitioned by stable `site_no` and UTC calendar month.
  Every raw JSON response is content-addressed; parsed blocks are create-only and
  retain request/response checksums.  Incomplete 24-hour days remain explicitly
  unavailable rather than being averaged from a partial day.
- Before labels are opened, the mandatory input evidence is the resolved schema
  and immutable provenance/snapshots for the retrospective meteorological fields
  actually consumed by primary models.  The optional NWP artifact must merely be
  resolved as either acquired-and-frozen or explicitly not used.
- A station/horizon is reportable when it has at least 100 valid confirmation
  targets.  Availability is reported; it is not used to replace the frozen cohort.
- The acquisition process records request parameters, retrieval UTC, response or
  normalized-file SHA-256, source statistic code, units, and approval qualifiers.
- NWIS parameter/statistic identities are fixed as daily mean (`00003`) water
  temperature (`00010`, degrees C), discharge (`00060`, cubic feet per second),
  and raw-only gage height (`00065`, feet).  Every daily-mean time-series column
  is retained.  On each date, zero finite series values means missing and exactly
  one means that value, qualifier, and series identifier are used.  If two or
  more series are finite on the same date, the parameter/date is marked missing
  with `MULTIPLE_FINITE_SERIES_CONFLICT`; values are never averaged or selected
  and the station is never replaced.  Exact identifiers and conflict counts are
  part of the evidence package.
- The primary analysis uses every finite parsed daily-mean WTEMP and FLOW value,
  regardless of its approval qualifier.  Qualifiers can never select or replace
  a station, model, date, or primary forecast key.  The result package reports
  immutable counts by cohort, station, variable, raw qualifier string, and value
  presence; unknown strings are preserved and counted without interpretation.
- A separately labelled, nonconfirmatory data-quality sensitivity is fixed now,
  before outcomes are opened.  Without refitting any model or filtering its
  issue-date history, it retains target keys only when the target WTEMP qualifier
  has the exact comma-token set `{A}`.  Station/horizon cells need at least 100
  such targets.  This sensitivity is descriptive and cannot enter, remove, or
  reinterpret any member of the five-test Holm family.
- NWIS dates denote site-local finalized daily values, while Daymet and gridMET
  use provider-specific calendar-day definitions.  No subdaily issue-time/day-
  boundary harmonization is claimed.  The evaluation is therefore date-indexed
  and retrospective.  Raw series identifiers do not prove sensor continuity;
  gage-height datum is not harmonized because WLEVEL is never consumed.
- The first successful scoring run writes a sealed result with the protocol commit,
  data manifest hash, resolved configuration hash, and code-tree hash.  A second
  scoring run after model changes is exploratory and must use a new run identity.

The external-site candidate universe is also frozen before labels.  Discovery
queries only the USGS site-metadata endpoint for stream sites in the 34 states
represented by the development cohort whose metadata advertises daily-value
parameter 00010 capability.  It does not request values, a date range, holdout
coverage, event rates, or model errors.  The exact state responses and the
site-number-sorted candidate table precede deterministic selection of 30 sites.

The order of these operations is itself a gate.  The final protocol and seal,
claim renderer/validator, source tree, dependency locks, temporal models, and
station-agnostic five-seed models must first be frozen and exactly replayed.  The
model-suite registry is then committed while the external candidate registry and
2021--2023 predictor artifacts are still absent.  Only afterward may metadata-only
candidate discovery and retrospective predictor acquisition run.  Authorization
must replay that Git ancestry, the earlier model-suite blob, the later evidence
blobs, and their hashes.  Failure of this order demotes the exercise to transductive
retrospective exploration and prohibits confirmation claims.  This is an internal
repository chronology for an honest owner; no external timestamp, public
preregistration service, or independent custodian is presently claimed.

## 3. Frozen models and information set

Route A evaluates a physics-inspired statistical forecaster at historically
gauged sites.  It does not claim true ungauged prediction, river-network routing,
mechanistic parameter identification, regulatory compliance, or operational NWP
skill.

The primary confirmation comparison includes:

1. persistence;
2. train-fitted damped persistence;
3. seasonal climatology;
4. a validation-tuned LightGBM with exactly the same available feature schema;
5. a validation-tuned LSTM with the same seed and optimization budget;
6. the five-member ThermoRoute ensemble.

The following single-seed architecture controls are mandatory but exploratory:
`DampedPriorOnly`, `TR-noDynamicPrior`, `TR-fixedKappa`, `TR-noRouter`,
`TR-noMoE`, `TR-noTCN`, and `TR-unbounded`.  These exact executable definitions
must be recorded in the frozen model-suite registry.  They replace earlier draft
labels such as `plain_mlp` that did not correspond to an implemented, one-factor
control.

The separately frozen 30-site external analysis uses station-agnostic models and
only pooled transformations fitted on the 120-site development training period.
Those sites still provide observed WTEMP through the issue date, so this is a
history-dependent new-gage validation, not an ungauged forecast.  It is reported
as external/exploratory evidence and is not added to the five-test primary family.

All models are scored on an identical key registry
`(site_no, horizon, issue_date, target_date)`.  No model-specific complete-case
set may be used for the primary comparison.

## 4. Primary estimands and hypotheses

The estimand is conditional on an explicit observable-key registry.  An issue is
admissible only when the issue and selected target are inside 2021--2023,
issue-date WTEMP is genuinely observed, and a 32-day history can be constructed.
Other missing predictors may be filled only with development-period frozen
transforms and keep their missingness masks.  A horizon is independently
admissible only when its target-date WTEMP is observed under the inclusive
qualifier policy.  Every declared model must emit exactly one prediction for
every such key.  A station/horizon enters the primary summary only with at least
100 admissible targets.

The sampling unit is a station.  For each horizon, first compute unweighted RMSE
on the common daily keys for each reportable station.  The primary effect is the
median across stations of the paired difference
`RMSE(ThermoRoute) - RMSE(reference)`, so a station with more observed dates does
not receive more point-estimand weight.  This estimates performance conditional
on observable issue/target WTEMP in the availability-enriched frozen cohort; it
is not performance on every calendar day or every U.S. river.

- H1: ThermoRoute improves on damped persistence.  Report the HUC2-cluster
  bootstrap 95% confidence interval for the median paired difference and win rate.
- H2: ThermoRoute is non-inferior to LightGBM at 3 and 7 days under a predeclared
  margin of +0.05 degrees C in station-median RMSE difference.
- H3: at 1 day, report the paired effect without a superiority or parity claim.

The confirmatory family is H1 at three horizons plus H2 at two horizons.  Holm
correction is applied across these five tests.  All other subgroup, architecture,
mechanism, event, and OOD analyses are exploratory and labelled as such.

The +0.05 degrees C non-inferiority margin was fixed before confirmation
outcomes as a strict numerical ceiling on allowable degradation.  It is not
derived from sensor precision, biological response, a water-quality standard,
or an elicited stakeholder utility, and it has no ecological or regulatory
importance interpretation.

For every comparison, first compute RMSE on the common daily keys separately for
each station, then form `RMSE(ThermoRoute) - RMSE(reference)`.  The primary effect
is the median of those paired station-level differences.  Its 95% interval is the
2.5th--97.5th percentile interval from 10,000 bootstrap draws of whole HUC2
clusters: all station effects in a sampled HUC2 are retained together.  A station
without verified HUC metadata is a stable singleton cluster.

The one-sided p-value is not derived from the bootstrap.  Because the canonical
registry contains at most 15 HUC2 clusters, it exactly enumerates every one of
the `2^K` whole-cluster sign vectors, applying one common sign to all station
effects in each cluster.  This removes Monte-Carlo error but still explicitly
assumes cluster-level sign symmetry around the tested margin.  Legacy per-test
seed fields are inert identifiers.  Holm step-down adjustment covers exactly the
three damped-persistence superiority tests and the two LightGBM non-inferiority
tests at horizons 3 and 7.  Equal-HUC aggregation, leave-one-HUC influence, and
the full cluster-size/effect table are descriptive sensitivities only; HUC2 is a
broad region and is not claimed to be an independent river-network component.

Failure to reject a difference is not evidence of equivalence.  The words
`equivalent`, `parity`, or `non-inferior` are allowed only when the complete
confidence interval lies inside the corresponding predeclared margin.

The structured claim renderer calls a formal test `SUPPORTED` if and only if it
is estimable, its one-sided Holm-adjusted p-value is at most 0.05, and the complete
percentile interval lies strictly below its predeclared margin
(`ci_high_c < margin_c`).  If the Holm and interval criteria disagree, the fixed
output is `EVIDENCE_CONFLICT_NOT_SUPPORTED` and reports both values; it may not
select the favorable procedure.  A non-estimable test reports attrition and no
directional conclusion.  All five tests, including negative or conflicting ones,
must appear exactly once.  H2 support permits only “satisfies the predeclared
+0.05 °C numerical non-inferiority bound”; it never licenses equivalence, parity,
ecological insignificance, or regulatory acceptability.  External, probability,
spatial, qualifier, robustness, and architecture-control findings remain
exploratory or descriptive regardless of their numerical direction.

## 5. Probabilistic and event evaluation

- Quantile output is described as a three-quantile forecast.  The equal-weighted
  three-quantile score is not called CRPS.
- Interval calibration is fit only on 2018 data and evaluated as empirical marginal
  coverage; no exchangeability or conditional-coverage guarantee is claimed.
- The exceedance event remains the station-specific train-period q90 diagnostic,
  an absolute statistical tail threshold local to each observed station.  It is
  not a biological, ecological, regulatory, or cross-station-comparable limit.
  Event probabilities are calibrated on 2018 only.  Brier skill uses a seasonal
  train/calibration climatology fixed before confirmation, never the confirmation
  event rate.
- Reliability, log loss, Brier score, AUPRC, AUROC, calibration slope/intercept,
  and expected calibration error are reported.  REV is exploratory and is shown
  only for explicitly stated hypothetical cost-loss ratios.
- EPA 7DADM and other regulatory conclusions are out of scope until daily maximum
  observations and waterbody-specific standards are available.

The event reference is frozen before confirmation labels.  For the temporal
cohort it is a station-by-calendar-month event rate fitted on 2006--2018 under
each station's 2006--2015 train-q90 threshold, smoothed with weight 2 toward its
station and global rates.  For external warm-start sites it is a pooled monthly
rate under a pooled 2006--2015 q90, with no external-site outcome adaptation.
The pooled threshold is only an absolute statistical diagnostic; it is neither
site-local nor biologically, ecologically, or regulatorily meaningful.

Probability metrics are descriptive and outside the five-test family.  Within
each cohort/model/horizon, stations with at least 100 exact common targets
receive equal total weight.  The fixed output reports 90% interval coverage and
width, equal-weight three-quantile pinball mean (not CRPS), Brier score and skill
against the frozen seasonal reference, clipped log loss, AUROC, AUPRC, 10-bin
ECE, and evaluation calibration intercept/slope.  A deterministic model with no
probability head is reported as unavailable; a one-class or non-estimable metric
is `null` with a reason, never silently dropped or replaced.  No p-value,
confidence interval, Holm decision, or conditional-coverage claim is generated.

## 6. Robustness and stopping rules

Missingness, sensor-noise, heat-tail, low-flow, high-flow, and meteorological-shift
stress tests use frozen perturbations and are exploratory.  No architecture or
hyperparameter is changed after confirmation labels are opened.  If a required
input source is unavailable, the affected analysis is reported missing; a new
source or station cohort is not selected based on confirmation outcomes.

The irreversible opening intent and opening ID are created only once, but fixed-
ledger transport may resume after a transient network failure.  A resume first
validates every complete create-only request transaction and fetches only wholly
absent ledger entries.  A partial or inconsistent transaction is never replaced
and leaves the opening indeterminate.  Resume is allowed only while the acquisition
manifest, normalized outcomes, trusted predictions, statistics, report, and final
receipt are all absent; no site, request, model, or analysis choice can change.
Fixed code does not parse, normalize, score, summarize, or render outcomes until
all ledger transactions are complete.  This reduces transport fragility but does
not prevent an honest-filesystem owner from manually reading partial raw bytes and
is not a malicious-owner security guarantee.

The confirmation report must include all predeclared models and hypotheses,
including negative results.  Route A is complete only when a clean environment can
rebuild the station registry and common-key predictions from frozen inputs, and a
change to data, resolved configuration, or source code produces a different run ID.

Before the one-time label opening, an isolated interpreter must reload every frozen
learned model member and every available prediction head and exactly reproduce its
validation, calibration, and 2019--2020 development keys and values from the frozen
inputs.  Opening authorization is bound to the resulting deterministic replay
receipt.  The same authorization is also bound to a fully transitive Python 3.12
dependency lock containing exact versions and package hashes.  A model file merely
being present, or a direct dependency version merely being named, does not satisfy
either gate.  The replay uses the fixed entrypoint under `python -I -B`; it denies
network and child-process access, repository writes, and reads from confirmation
data/output namespaces, then records the complete repository-read path digest.  A
fresh authorization subprocess must recompute all predictions and exactly match the
receipt rather than trusting a self-attested JSON file.

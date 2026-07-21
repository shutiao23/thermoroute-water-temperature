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
- The first successful scoring run writes a sealed result with the protocol commit,
  data manifest hash, resolved configuration hash, and code-tree hash.  A second
  scoring run after model changes is exploratory and must use a new run identity.

The external-site candidate universe is also frozen before labels.  Discovery
queries only the USGS site-metadata endpoint for stream sites in the 34 states
represented by the development cohort whose metadata advertises daily-value
parameter 00010 capability.  It does not request values, a date range, holdout
coverage, event rates, or model errors.  The exact state responses and the
site-number-sorted candidate table precede deterministic selection of 30 sites.

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

The sampling unit is a station.  For each horizon, first compute RMSE on the
common daily keys for each station.  The primary effect is the paired station-level
RMSE difference `RMSE(ThermoRoute) - RMSE(reference)`.

- H1: ThermoRoute improves on damped persistence.  Report the HUC2-cluster
  bootstrap 95% confidence interval for the median paired difference and win rate.
- H2: ThermoRoute is non-inferior to LightGBM at 3 and 7 days under a predeclared
  margin of +0.05 degrees C in station-median RMSE difference.
- H3: at 1 day, report the paired effect without a superiority or parity claim.

The confirmatory family is H1 at three horizons plus H2 at two horizons.  Holm
correction is applied across these five tests.  All other subgroup, architecture,
mechanism, event, and OOD analyses are exploratory and labelled as such.

For every comparison, first compute RMSE on the common daily keys separately for
each station, then form `RMSE(ThermoRoute) - RMSE(reference)`.  The primary effect
is the median of those paired station-level differences.  Its 95% interval is the
2.5th--97.5th percentile interval from 10,000 bootstrap draws of whole HUC2
clusters: all station effects in a sampled HUC2 are retained together.  A station
without verified HUC metadata is a stable singleton cluster.

The one-sided p-value is not derived from the bootstrap.  It uses 50,000 whole-
HUC2 sign-flip randomisations, applying one common sign to all station effects in
each cluster.  This test explicitly assumes cluster-level sign symmetry around
the tested margin; it uses the add-one Monte Carlo correction.  Seeds are fixed
by horizon and reference in the machine-readable protocol.  Holm step-down
adjustment covers exactly the three damped-persistence superiority tests and the
two LightGBM non-inferiority tests at horizons 3 and 7.

Failure to reject a difference is not evidence of equivalence.  The words
`equivalent`, `parity`, or `non-inferior` are allowed only when the complete
confidence interval lies inside the corresponding predeclared margin.

## 5. Probabilistic and event evaluation

- Quantile output is described as a three-quantile forecast.  The equal-weighted
  three-quantile score is not called CRPS.
- Interval calibration is fit only on 2018 data and evaluated as empirical marginal
  coverage; no exchangeability or conditional-coverage guarantee is claimed.
- The exceedance event remains the station-specific train-period q90 diagnostic.
  Event probabilities are calibrated on 2018 only.  Brier skill uses a seasonal
  train/calibration climatology fixed before confirmation, never the confirmation
  event rate.
- Reliability, log loss, Brier score, AUPRC, AUROC, calibration slope/intercept,
  and expected calibration error are reported.  REV is exploratory and is shown
  only for explicitly stated hypothetical cost-loss ratios.
- EPA 7DADM and other regulatory conclusions are out of scope until daily maximum
  observations and waterbody-specific standards are available.

## 6. Robustness and stopping rules

Missingness, sensor-noise, heat-tail, low-flow, high-flow, and meteorological-shift
stress tests use frozen perturbations and are exploratory.  No architecture or
hyperparameter is changed after confirmation labels are opened.  If a required
input source is unavailable, the affected analysis is reported missing; a new
source or station cohort is not selected based on confirmation outcomes.

The confirmation report must include all predeclared models and hypotheses,
including negative results.  Route A is complete only when a clean environment can
rebuild the station registry and common-key predictions from frozen inputs, and a
change to data, resolved configuration, or source code produces a different run ID.

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

## 2. Untouched temporal confirmation set

- Confirmation period: 2021-01-01 through 2023-12-31 for the frozen 120-site
  cohort.  These labels must not be read by model-selection, feature-selection,
  threshold-selection, calibration, or station-inclusion code.
- Inputs are restricted to information available through the issue date.  This is
  a historical-information experiment, not an archived-NWP operational forecast.
- A station/horizon is reportable when it has at least 100 valid confirmation
  targets.  Availability is reported; it is not used to replace the frozen cohort.
- The acquisition process records request parameters, retrieval UTC, response or
  normalized-file SHA-256, source statistic code, units, and approval qualifiers.
- The first successful scoring run writes a sealed result with the protocol commit,
  data manifest hash, resolved configuration hash, and code-tree hash.  A second
  scoring run after model changes is exploratory and must use a new run identity.

## 3. Frozen models and information set

Route A evaluates a physics-inspired statistical forecaster at historically
gauged sites.  It does not claim true ungauged prediction, river-network routing,
mechanistic parameter identification, regulatory compliance, or operational NWP
skill.

The confirmation comparison includes:

1. persistence;
2. train-fitted damped persistence;
3. seasonal climatology;
4. a validation-tuned LightGBM with exactly the same available feature schema;
5. a validation-tuned LSTM with the same seed and optimization budget;
6. ThermoRoute and the predeclared nested variants `prior_only`, `plain_mlp`,
   `plain_tcn`, `bounded_residual`, and `unbounded_residual`.

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

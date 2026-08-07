# Route B measurement, missingness and space×time UQ draft (P0-B3)

| Field | Value |
| --- | --- |
| Date | 2026-08-01 |
| Status | **DRAFT ONLY / PRELABEL / NOT AN EXECUTABLE PROTOCOL** |
| Dependency | Requires P0-B1 sampling design and P0-B2 task/model information sets |
| Evidence boundary | Raw target values are not acquired by this draft; no Route-A/Stage-09 artifact is modified or reused |

## 1. Measurement evidence contract

Every retained target and issue-time observation must trace to a durable raw
transaction and a unique time-series identity. A normalized value without its raw
response, series metadata, qualifier/approval status and local-day rule is not a
complete measurement record.

### `measurement_qa_registry_v1.parquet`

```text
site_id
time
target_local_date
parameter_code
statistic_code
value
unit
timeseries_id
approval_status
qualifiers
method_id
instrument_id
sensor_start
sensor_end
timezone
raw_transaction_sha256
provider_last_modified
series_conflict_code
retained_primary
missing_metadata_reason
```

Unavailable method, instrument or calibration history is stored as null plus a
reason, not silently converted to “standard sensor.” If local Water Science Center
records are required, the request and response become separately bound evidence.

Duplicate series are resolved by an outcome-independent rule based on parameter,
statistic, approval, method continuity and frozen series identity. The pipeline
must retain the conflict ledger even when only one series enters scoring.

## 2. Flow and calendar semantics

Negative FLOW values are retained in raw evidence. They may represent tidal,
backwater or measurement semantics and must not be renamed drought flow or
silently clipped.

- the default robust numeric transform is signed `asinh`;
- tidal/backwater/negative-flow strata receive explicit sensitivity results;
- if official Air2stream requires positive flow, a positive-flow non-tidal domain
  must be declared in the sampling frame before targets are accessed;
- the restricted result cannot be generalized back to tidal/negative-flow sites.

USGS local-day, UTC/GMT meteorology and daylight-period predictor semantics are
recorded independently. The primary alignment rule is frozen before target
access, with a separately labelled local-day/UTC boundary sensitivity.

## 3. Opportunity and attrition registries

### `calendar_opportunity_registry_v1.parquet`

```text
site_id
issue_date
target_date
horizon
network_component_id
calendar_opportunity
issue_observed
history_fraction
max_history_gap_days
predictor_mask_sha256
target_observed
exclusion_stage
exclusion_reason
```

### `attrition_cells_v1.parquet`

```text
network_component_id
site_id
year
season
horizon
stage
n_opportunity
n_history_pass
n_target_observed
n_common_key
n_reportable
```

These registries distinguish the all-calendar opportunity population from the
observable-key estimand. Every abstract/table/caption reports the corresponding
station, component, year and key counts.

## 4. History completeness and fill invariance

The minimum WTEMP/predictor history fraction and maximum-gap rule are selected
from training/calibration behavior only, then frozen. Target-period error may not
be used to tune them.

Every path that consumes an imputed value, including thermal proposals and gates,
must also consume or propagate the corresponding mask. A required metamorphic
test changes every fill value where `mask=0` while holding observed inputs and
masks fixed. Predictions must remain bitwise equal or within a separately
declared deterministic tolerance. Failure blocks the model suite.

Primary results remain conditional on their frozen observable/history gate.
All-calendar attrition and missingness sensitivities accompany them; missing keys
are not treated as correctly predicted days.

## 5. Calibration inclusion

Calibration has one registry per horizon. The inclusion predicate for horizon
`h` must be exactly the scoring predicate for `h`, restricted to the frozen
calibration dates. Availability of another horizon may not remove the row.

### `calibration_registry_v1.parquet`

```text
horizon
issue_date
target_date
site_id
network_component_id
key_sha256
y_observed
included
exclusion_reason
calibrator_or_offset_sha256
```

The fit receipt binds every key, the target-free prediction head, the target
transaction, CQR offset/Platt model and post-fit coverage diagnostics. No target
from the formal evaluation period enters calibration.

## 6. Missingness sensitivity

The primary complete-key estimand is accompanied by predeclared, descriptive
sensitivities:

1. cross-fitted stabilized inverse-probability weighting under an explicit MAR
   model;
2. δ-pattern-mixture shifts for unobserved labels;
3. physically bounded worst-case paired-error/tipping calculations;
4. year/season/site/component attrition maps.

### `missingness_sensitivity_v1.parquet`

```text
method
assumption
horizon
comparison
delta_or_bound
effect
uncertainty_low
uncertainty_high
effective_sample_size
tipping_status
```

These cannot rescue a failed primary gate or be selected after results according
to favorability. MAR/IPW does not prove MAR; pattern-mixture bounds expose the
dependence on unobserved outcomes.

## 7. Crossed space×time uncertainty

The primary uncertainty procedure should propagate graph-component and finite-time
variation without treating graph disconnection as statistical independence.
Draft algorithm:

1. perform 20,000 preregistered bootstrap draws;
2. resample PSUs with the frozen design/strata and replicate weights;
3. choose the frozen year-scope branch:
   - with only three target years, keep those years fixed and interpret the result
     as conditional on that target period;
   - a year-superpopulation claim requires additional independent years and a
     preregistered small-sample/year-resampling method;
4. within each retained/resampled year×season, apply a 28-day moving block
   synchronously to all selected
   components, preserving shared national/regional weather-event dates;
5. recompute predictions-to-key joins, site RMSE, component effect and the
   design-weighted overall effect from daily rows in every draw;
6. retain failed/non-estimable draws and apply a predeclared maximum failure-rate
   rule.

The 28-day block is a draft value. It must be justified and frozen from
training/calibration residual dependence. Fourteen- and 56-day blocks may be
registered as sensitivities. A secondary hierarchical model may be added but
cannot replace the primary bootstrap after seeing which is more favorable.
Twenty thousand draws reduce Monte Carlo error; they do not create more
independent target years.

### `spacetime_uq_draws_v1.parquet`

```text
draw_id
comparison
horizon
component_sample_sha256
year_sample_sha256
block_sample_sha256
n_components
n_sites
n_keys
effect
estimable
failure_code
```

### `spacetime_uq_receipt_v1.json`

The self-hashed receipt binds the sampling design, seeds, block rule, every draw,
non-estimable rate, point estimand, interval, and sensitivity variants.

## 8. Probability reporting

Report, at minimum:

- station- and component-balanced interval score;
- pinball loss at every emitted quantile;
- marginal coverage and width by station, season and registered event class;
- event Brier score, reliability curve and calibration slope;
- calibration sample sizes and attrition.

Do not call three-quantile pinball CRPS. Do not claim distribution-free conditional
coverage from a marginal conformal procedure. A CRPS claim requires a full
distribution or a preregistered sufficiently dense quantile approximation.

## 9. Acceptance gates

P0-B3 passes only when:

1. 100% of retained issue/target rows trace to a raw transaction, time-series ID,
   qualifier and approval status; unavailable method/sensor fields have reasons.
2. Duplicate/conflicting series follow the frozen non-outcome rule.
3. Negative-flow and day-boundary sensitivities are produced without changing the
   primary cohort post hoc.
4. All imputation branches propagate masks and pass fill invariance.
5. History-completeness rules are frozen before target access.
6. Per-horizon calibration and scoring key predicates are identical.
7. Every formal cell has estimable crossed space×time uncertainty or reports
   `NOT_ESTIMABLE`; failed bootstrap draws remain visible.
8. Coverage/reliability language stays within the procedure actually evaluated.

## 10. Future implementation namespace

After the Stage-09 boundary is released, proposed new files include:

- `protocols/route_b_measurement_missingness_uq_v1.json`;
- `src/thermoroute/route_b_measurement.py`;
- `src/thermoroute/route_b_missingness.py`;
- `src/thermoroute/route_b_spacetime.py`;
- tests for provenance closure, negative-flow handling, local-day alignment,
  fill invariance, per-horizon calibration and crossed-bootstrap recovery on
  synthetic data.

Raw and derived artifacts belong only under `data_route_b/**` and
`outputs/route_b/**`. Opening/target acquisition remains a separate future human
authorization gate.

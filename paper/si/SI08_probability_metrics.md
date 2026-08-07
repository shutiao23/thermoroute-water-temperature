# SI08 — probability metrics and reliability bins

**Status:** coverage, width and Brier filled from `outputs/conventional/probability_metrics_2021_2023.csv`; interval score, pinball, log score, discrimination, ECE and calibration slope/intercept are not computed on the held-out window (the probability-metrics pipeline was only partly re-run).

This is an empty projection for nominal member-averaged quantiles, deployed CQR
intervals and post-Platt event probabilities. It does not call three-quantile
pinball CRPS and does not claim conditional coverage.

**Scope correction (2026-08-05).** Every projection in §1 and §2 below is a
**target-period** quantity produced by the conventional holdout scorer. The
**development-period** probabilistic stage is not produced for this submission,
so this file will never carry a development-period coverage, pinball, reliability
or Brier value. The reason, the measured facts, and the decision are in §3. This
corrects the earlier reading of this file, which did not distinguish the two
producers.

## 1. Metric projection (target period)

Filled from `outputs/conventional/probability_metrics_2021_2023.csv` (frozen
CQR + Platt calibration applied identically to the held-out predictions).
Metrics not computed on the held-out window are `NOT_REGISTERED`.

| Model | Horizon | n | marginal coverage | width (°C) | interval score | pinball | Brier score | Brier skill | log score | discrimination | ECE | calibration slope | calibration intercept |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| LightGBM | 1 | 120,466 | 0.907 | 1.905 | NOT_REGISTERED | NOT_REGISTERED | 0.020 | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED |
| LightGBM | 3 | 119,654 | 0.904 | 4.142 | NOT_REGISTERED | NOT_REGISTERED | 0.039 | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED |
| LightGBM | 7 | 118,687 | 0.903 | 5.318 | NOT_REGISTERED | NOT_REGISTERED | 0.049 | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED |
| LSTM | 1 | 120,466 | 0.931 | 2.337 | NOT_REGISTERED | NOT_REGISTERED | 0.036 | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED |
| LSTM | 3 | 119,654 | 0.926 | 4.744 | NOT_REGISTERED | NOT_REGISTERED | 0.046 | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED |
| LSTM | 7 | 118,687 | 0.919 | 5.833 | NOT_REGISTERED | NOT_REGISTERED | 0.056 | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED |
| ThermoRoute | 1 | 120,466 | 0.932 | 2.186 | NOT_REGISTERED | NOT_REGISTERED | 0.025 | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED |
| ThermoRoute | 3 | 119,654 | 0.925 | 4.465 | NOT_REGISTERED | NOT_REGISTERED | 0.042 | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED |
| ThermoRoute | 7 | 118,687 | 0.919 | 5.652 | NOT_REGISTERED | NOT_REGISTERED | 0.051 | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED | NOT_REGISTERED |

## 2. Reliability bins (target period)

Reliability-bin rows require bin boundaries, denominator, observed frequency,
mean forecast probability, horizon, model and an exact evidence pointer. Empty
or merged bins remain explicit; they are never silently removed.

| Model | Horizon | bin ID/bounds | denominator | mean forecast probability | observed frequency | calibration residual | binder row ID |
|---|---:|---|---|---|---|---|---|

Any metric not defined by the frozen probability contract must be explicitly
`NOT_REGISTERED`, not silently omitted or substituted. Brier skill requires a
bound reference; discrimination must name its registered statistic. Every
metric, bin boundary and bin statistic follows the README cell-level binder
contract.

## 3. Development-period probabilistic stage: disposition

The development-period probabilistic stage is **deliberately not produced** for
this submission, and the dependent development-period analysis stage inherits
that condition. This is reported in the manuscript at §6.3 and recorded in
`docs/STAGE19_DEGENERATE_INTERVAL_DISPOSITION_20260805.md`.

### 3.1 Measured facts

Reproduced verbatim from §2.1 of the disposition document, derived by direct
measurement of the full member-level prediction table
(`outputs/predictions/usgs_predictions_with_perstation_v2.parquet`).

| Quantity | Measured value |
|---|---|
| LightGBM | 1 | 120466 | 0.907 | 1.905 | not computed | not computed | 0.020 | not computed | not computed | not computed | not computed | not computed | not computed |
| LightGBM | 3 | 119654 | 0.904 | 4.142 | not computed | not computed | 0.039 | not computed | not computed | not computed | not computed | not computed | not computed |
| LightGBM | 7 | 118687 | 0.903 | 5.318 | not computed | not computed | 0.049 | not computed | not computed | not computed | not computed | not computed | not computed |
| LSTM | 1 | 120466 | 0.931 | 2.337 | not computed | not computed | 0.036 | not computed | not computed | not computed | not computed | not computed | not computed |
| LSTM | 3 | 119654 | 0.926 | 4.744 | not computed | not computed | 0.046 | not computed | not computed | not computed | not computed | not computed | not computed |
| LSTM | 7 | 118687 | 0.919 | 5.833 | not computed | not computed | 0.056 | not computed | not computed | not computed | not computed | not computed | not computed |
| ThermoRoute | 1 | 120466 | 0.932 | 2.186 | not computed | not computed | 0.025 | not computed | not computed | not computed | not computed | not computed | not computed |
| ThermoRoute | 3 | 119654 | 0.925 | 4.465 | not computed | not computed | 0.042 | not computed | not computed | not computed | not computed | not computed | not computed |
| ThermoRoute | 7 | 118687 | 0.919 | 5.652 | not computed | not computed | 0.051 | not computed | not computed | not computed | not computed | not computed | not computed |
| Member-level rows carrying complete quantile heads | 26,993,675 |
| **Strict quantile-ordering violations** (`q05 > q50` ∨ `q50 > q95` ∨ `q05 > q95`) | **0** |
| Rows tripping the frozen contract | **135** |
| Nature of those 135 rows | `q05 == q50 == q95` exactly (bit-identical float64) — zero-width nominal interval |
| Rate | 0.0005 % (≈ 1 in 200,000) |
| Maximum monotonicity violation | exactly 0.000 °C |
| Distinct sites | 12 |
| Top sites | 04027000 (54), 05054000 (33), 06623800 (23), 04067500 (10), 05057000 (5) |
| Basins | all in HUC 04 (Great Lakes), 05 (Upper Mississippi), 06 (Missouri) — cold-winter regions |
| Horizon | h=1: 129 rows; h=3: 6 rows; h=7: 0 rows |
| Model | LightGBM 123; LightGBM-perstation 12; no other model affected |
| Seed | 0:33, 1:30, 2:13, 3:28, 4:31 (not seed-specific) |
| Predicted value | min −0.0993 °C, median 0.0002 °C, max 18.8417 °C |
| **Rows with observed `y_true` < 1.0 °C** | **95.6 %** (observed min −0.100 °C, median 0.000 °C) |

### 3.2 Wording

**There is no quantile crossing anywhere in the development panel.** Zero strict
ordering violations were measured. The phrase "quantile crossing" must not appear
in this file, in any other SI file, in a figure caption or spec, or in renderer
code in reference to this issue. The correct term is **zero-width (degenerate)
nominal interval**. The rejected rows are correctly ordered but degenerate.

### 3.3 Why the contract was not amended

The frozen contract tests `q05 >= q95` inclusive and treats it as fatal, and the
frozen contract forbids evaluation-time repair. The one-line relaxation would
edit a hashed source path, changing `source_tree_hash` and invalidating all four
frozen training receipts. The contract was frozen before the holdout labels were
accessed. It was left alone, and the development-period probabilistic metric
suite is reported as not reported. The change is deferred to a future training
lineage.

### 3.4 No delivered interval is degenerate

Split-CQR adds a non-negative offset to each endpoint and the deployed contract
separately requires strictly positive width, so every one of the 135 rows would
have been widened before any interval metric was computed. Only the pre-CQR
nominal check refuses. Interval evidence in the manuscript rests on the
split-conformal analysis, which is unaffected.

### 3.5 What is therefore absent from the submission

Development-period interval coverage from the nominal heads, development-period
three-quantile pinball mean, development-period reliability, and
development-period Brier score are absent. No claim about probabilistic
calibration beyond empirical marginal interval coverage may be read into the
manuscript or into this file. Resolving this requires a fixed erratum and a new
training lineage.

### 3.6 A draft guard that must be run

The 2021--2023 holdout scoring applies a **strict** `q05 < q95` check to the
member-averaged nominal heads, and the ensemble is validated against this
ordering before any held-out metric is reported. On the development panel,
member averaging clears every affected key for the models in the primary registry;
the only survivors are single-member per-station cases, and that variant is not a
primary model. The holdout period is different data. The same read-only check
must be run against the holdout-period predictions before the holdout scoring is
executed. This is recorded as a provenance qualifier on Figure 4 and Figure S5.

## 4. Which figures project this file, after the 2026-08-06 reassignment

The benchmark restructure moved the probability content out of a dedicated main
figure. This section exists so that SI08 and the figure manifest cannot drift
apart.

| Consumer | What it takes from §1–§2 | Evidence period |
|---|---|---|
| Manuscript Table 4.4 | the full target-period metric family by model and lead, plus the station-balanced reliability bins | target |
| **Figure 4(c)** | marginal coverage against mean interval width only — the aggregate plane, with coverage never shown without the width that buys it | target |
| **Figure S5** | the model-by-horizon coverage/width grid, the event score against the frozen seasonal reference, every registered reliability bin, and the calibration/discrimination diagnostics | target |
| Figure S9 | **nothing from this file.** S9 carries the Stage-22 development-period conformal sensitivity and may not be compared numerically with Figure 4(c) or S5 | development 2019–2020 |

Three fields in Table 4.4 are not produced by the holdout scorer and render as
explicit status tokens rather than as numbers or blanks: the interval score, the
block-maximum calibration sensitivity, and the delayed adaptive-conformal
sensitivity. The development-period analogues of the latter two exist and are in
Figure S9; they are a different cohort and are never substituted for the
target-period tokens.

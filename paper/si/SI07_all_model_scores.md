# SI07 — all-model exact-common-key scores

**Status:** filled from `outputs/final/station_metrics.parquet` (unweighted
station medians over the 116 reportable stations on the common held-out key
registry; pooled RMSE from `outputs/final/pooled_metrics.parquet` is reported
here as a sensitivity only and is never a station median).

Rows are generated only from receipt-bound predictions after the renderer proves
one exact common key set for every compared model. Development scores are not
admissible.

| Model | Horizon | paired keys | stations | RMSE (°C) | MAE (°C) | bias (°C) | station-balanced summary | binder row ID |
|---|---:|---|---|---|---|---|---|---|

Every cell follows the README cell-level binder contract. The row must bind the
key-registry digest, metric formula, finite-value filter,
station aggregation, unit, rounding rule and parent prediction digest. Selective
model omission rejects the POST build.

The frozen model registry for the temporal cohort is the six primary models
**plus** the seven one-factor architecture controls, so this table carries a row
for each of the thirteen, on one exact common key set, with the controls
labelled exploratory. Figure S10 is the graphical reading of the control rows on
the held-out window
(`paper/si/figures/render_figS10_architecture_controls.py`); it must agree with
this table in value, unit, and rounding wherever the two report the same period.

## Discharge-channel ablation (Major Comment 7 retraining test)

`scripts/final/run_flow_ablation.py` retrains the station-agnostic global
LightGBM per lead on the development train/validation rows with every FLOW
column removed (28 columns at 1 d; 182 features remain) and scores the
identical 2021–2023 common keys. Station-first paired ΔRMSE (no-flow minus
with-flow, °C; positive favours with-flow):

| Horizon | RMSE no-flow | RMSE with-flow | median ΔRMSE | no-flow win fraction | stations |
|---|---:|---:|---:|---:|---:|
| 1 d | 0.622 | 0.589 | +0.042 | 0.03 | 116 |
| 3 d | 1.357 | 1.304 | +0.034 | 0.14 | 116 |
| 7 d | 1.738 | 1.735 | +0.009 | 0.37 | 116 |

The scale-perturbation probes (flow × 0.5 / × 2) leave 1-day RMSE nearly
unchanged; the retraining test shows the channel carries small but systematic
information at the shortest lead. The trade-off against the cohort's spatial
coverage (950 of 1,465 candidates excluded for missing joint flow) is quantified
by a no-flow core cohort only when the candidate registry is re-derived
(protocol P2, not run).

## Models named in the manuscript that carry no score, and why

One model row is absent from the manuscript tables and one model row is scored:
the official upstream air2stream remains unbuilt (see provenance below), while
the unofficial empirical a8 variant is fitted and scored on the held-out window.

### 1. The air2stream-style hybrid reference — fitted (a8 variant)

**Status.** The unofficial empirical a8 variant of `src/thermoroute/air2stream.py`
is calibrated per station on the 2006–2015 training record (multi-start bounded
least squares, six starts, `max_nfev = 6000`) and scored on the 2021–2023
**common-key registry** of the results authority (`outputs/final/forecast_keys.parquet`):
only stations with at least 100 paired common keys are reportable, so the
primary hybrid rows use the identical 116-station set as every other model
(118 stations carry a fitted hybrid, of which two fall below the 100-key
threshold). Station-median RMSE on the reportable set is 0.719 / 1.459 / 1.825 °C
at 1 / 3 / 7 days. A separately labelled sensitivity retains all 118 fitted
stations with **no reportability filter applied**, giving 0.719 / 1.478 /
1.825 °C; the 1- and 7-day figures coincide with the primary set to three
decimals and only the 3-day value differs. Median of per-station skill
against damped persistence is +0.063 / −0.008 / −0.011. The official upstream
model was not executed; the hybrid is an *unofficial empirical comparator*, not
a claim against the published model and not a process-based result (manuscript
Section 6.2).

**Provenance (official upstream).** The pinned-source audit
(`docs/AIR2STREAM_SOURCE_BUILD_AUDIT_20260801.md`) records verdict
`BLOCKED_NO_COMPILER / REFERENCE_CASE_NOT_ATTESTABLE` against upstream
commit `d4834bccf01657c03ab60efb4c18f8a256132c53`:

| Finding | Consequence for a comparison |
|---|---|
| Persistence | 1 | all | 116 | 0.813 | 0.597 | -0.000 | station median | si07.row |
| Persistence | 3 | all | 116 | 1.638 | 1.235 | -0.001 | station median | si07.row |
| Persistence | 7 | all | 116 | 2.202 | 1.686 | -0.004 | station median | si07.row |
| DampedPersistence | 1 | all | 116 | 0.789 | 0.591 | -0.029 | station median | si07.row |
| DampedPersistence | 3 | all | 116 | 1.454 | 1.100 | -0.076 | station median | si07.row |
| DampedPersistence | 7 | all | 116 | 1.773 | 1.340 | -0.143 | station median | si07.row |
| Climatology | 1 | all | 116 | 1.899 | 1.485 | -0.379 | station median | si07.row |
| Climatology | 3 | all | 116 | 1.902 | 1.486 | -0.369 | station median | si07.row |
| Climatology | 7 | all | 116 | 1.903 | 1.485 | -0.359 | station median | si07.row |
| LightGBM | 1 | all | 116 | 0.589 | 0.431 | -0.025 | station median | si07.row |
| LightGBM | 3 | all | 116 | 1.304 | 0.995 | -0.098 | station median | si07.row |
| LightGBM | 7 | all | 116 | 1.735 | 1.315 | -0.196 | station median | si07.row |
| LSTM | 1 | all | 116 | 0.663 | 0.503 | -0.001 | station median | si07.row |
| LSTM | 3 | all | 116 | 1.358 | 1.044 | -0.051 | station median | si07.row |
| LSTM | 7 | all | 116 | 1.712 | 1.292 | -0.137 | station median | si07.row |
| ThermoRoute | 1 | all | 116 | 0.640 | 0.463 | +0.011 | station median | si07.row |
| ThermoRoute | 3 | all | 116 | 1.337 | 1.013 | -0.029 | station median | si07.row |
| ThermoRoute | 7 | all | 116 | 1.694 | 1.269 | -0.130 | station median | si07.row |
| Air2stream (a8, unofficial) | 1 | all | 116 | 0.719 | 0.559 | -0.020 | station median | si07.row |
| Air2stream (a8, unofficial) | 3 | all | 116 | 1.459 | 1.115 | -0.110 | station median | si07.row |
| Air2stream (a8, unofficial) | 7 | all | 116 | 1.825 | 1.366 | -0.189 | station median | si07.row |
| Air2stream (a8, unofficial, no reportability filter) | 1 | all | 118 | 0.719 | 0.559 | -0.020 | station median | si07.sens |
| Air2stream (a8, unofficial, no reportability filter) | 3 | all | 118 | 1.478 | 1.128 | -0.111 | station median | si07.sens |
| Air2stream (a8, unofficial, no reportability filter) | 7 | all | 118 | 1.825 | 1.366 | -0.189 | station median | si07.sens |
| no supported Fortran compiler present (`gfortran`, `ifort`, `ifx`, `flang`, `nvfortran`, `f95`, `f90` all absent), no Makefile or documented compiler command, and Intel-specific `ifport` / `makedirqq` calls in `AIR2STREAM_READ.f90` | the pinned source was never built in this environment |
| the shipped `air2stream_1.0.0.out` binary was deliberately not executed | running an upstream binary would not show that the pinned source builds, and no golden-output checksum is published to attest it against |
| the commit is unsigned (`%G? = N`, `git verify-commit` fails) and no tag is advertised | provenance is TLS transport plus Git object identity, not signer attestation |
| no committed expected output; the default PSO path calls `random_seed`/`random_number` with no fixed-seed contract | any reference case must use tolerance-based declared metrics or an explicitly frozen deterministic seed path, never byte-identical optimizer output |
| README attributes preprocessed case inputs to FOEN and MeteoSwiss; the CC BY-SA 3.0 code licence does not by itself resolve case-data redistribution | the reference case cannot be redistributed with this work as it stands |

**What a defensible comparison would require**, all of it before any number is
quoted:

1. the official implementation built from the pinned source in a recorded
   environment, with the build command and toolchain versions in the receipt;
2. an attested reference case — either a frozen deterministic seed path or
   declared metrics with a stated tolerance — reproduced from that build;
3. a documented, symmetric calibration search: the same predeclared budget,
   objective, and selection partition the learned references receive, since
   tuning budgets are already an acknowledged asymmetry in this study;
4. scoring on the identical exact common key registry every other model uses, so
   the row shares this table's denominators; and
5. a resolved redistribution decision for the case data, recorded in SI16.

Until all five hold, the correct rendering is the status token, not a number and
not a blank.

### 2. Per-station LightGBM — named, unscored

The reference set names a per-station LightGBM variant, whose purpose is to
isolate the value of pooling against the global model with site identity as a
categorical feature. **No scored development summary of that variant exists**,
so the manuscript names the variant in §3.2 without quoting a number and omits it
from the §4.1 score table. It is not a primary model and appears in neither the
five-row comparison family nor the protocol's model registry, so its absence
weakens no registered claim. Closing this gap means scoring it on the same exact
common key registry and adding a row here; it does not mean reporting the
global-model number under the per-station label.


## Skill against persistence and damped persistence (relocated Table 4.7b)

<!-- TABLE 4.7b (generated) -->

| Model | persist. 1 | persist. 3 | persist. 7 | damped 1 | damped 3 | damped 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Persistence | — | — | — | — | — | — |
| DampedPersistence | — | — | — | — | — | — |
| Climatology | -1.383 | -0.144 | +0.181 | -1.455 | -0.268 | -0.043 |
| LightGBM | +0.261 | +0.195 | +0.248 | +0.237 | +0.081 | +0.030 |
| LSTM | +0.181 | +0.172 | +0.244 | +0.150 | +0.055 | +0.028 |
| PlainMLP-7var | +0.152 | +0.163 | +0.243 | +0.115 | +0.047 | +0.023 |
| PlainCausalTCN-7var | +0.190 | +0.180 | +0.246 | +0.164 | +0.062 | +0.028 |
| Air2stream | +0.091 | +0.113 | +0.218 | +0.063 | -0.008 | -0.011 |
| ThermoRoute | +0.206 | +0.186 | +0.250 | +0.173 | +0.077 | +0.038 |


## Skill against persistence and damped persistence (relocated Table 4.7b)

<!-- TABLE 4.7b (generated) -->
| Model | persist. 1 | persist. 3 | persist. 7 | damped 1 | damped 3 | damped 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Persistence | — | — | — | — | — | — |
| DampedPersistence | — | — | — | — | — | — |
| Climatology | -1.383 | -0.144 | +0.181 | -1.455 | -0.268 | -0.043 |
| LightGBM | +0.261 | +0.195 | +0.248 | +0.237 | +0.081 | +0.030 |
| LSTM | +0.181 | +0.172 | +0.244 | +0.150 | +0.055 | +0.028 |
| PlainMLP-7var | +0.152 | +0.163 | +0.243 | +0.115 | +0.047 | +0.023 |
| PlainCausalTCN-7var | +0.190 | +0.180 | +0.246 | +0.164 | +0.062 | +0.028 |
| Air2stream | +0.091 | +0.113 | +0.218 | +0.063 | -0.008 | -0.011 |
| ThermoRoute | +0.206 | +0.186 | +0.250 | +0.173 | +0.077 | +0.038 |


## Skill against persistence and damped persistence (relocated Table 4.7b)

<!-- TABLE 4.7b (generated) -->
| Model | persist. 1 | persist. 3 | persist. 7 | damped 1 | damped 3 | damped 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Persistence | — | — | — | — | — | — |
| DampedPersistence | — | — | — | — | — | — |
| Climatology | -1.383 | -0.144 | +0.181 | -1.455 | -0.268 | -0.043 |
| LightGBM | +0.261 | +0.195 | +0.248 | +0.237 | +0.081 | +0.030 |
| LSTM | +0.181 | +0.172 | +0.244 | +0.150 | +0.055 | +0.028 |
| PlainMLP-7var | +0.152 | +0.163 | +0.243 | +0.115 | +0.047 | +0.023 |
| PlainCausalTCN-7var | +0.190 | +0.180 | +0.246 | +0.164 | +0.062 | +0.028 |
| Air2stream | +0.091 | +0.113 | +0.218 | +0.063 | -0.008 | -0.011 |
| ThermoRoute | +0.206 | +0.186 | +0.250 | +0.173 | +0.077 | +0.038 |

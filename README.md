# ThermoRoute

ThermoRoute is a research repository for a retrospective daily river-water-
temperature hindcasting benchmark at 1-, 3-, and 7-day horizons. It does not
establish an as-issued operational forecast. The study follows a conventional
design with one development period (2006–2020) and one independently held-out
temporal window (2021–2023): models, preprocessing, and calibration are fixed
using development-period data only, and the holdout is scored once, with every
number derived from persisted artifacts.

The model combines a damped-persistence anchor, a learned thermal-relaxation
proposal, a horizon-conditioned variable/lag router, a strictly left-looking
(non-anticipating) temporal convolutional encoder, a regime mixture, bounded
residuals, a separate MSE point head, three pinball-trained quantile heads, and
split-conformal calibration of the nominal 90% interval.

## Held-out results (2021–2023)

Held-out station-median RMSE in °C by horizon (116 reportable stations on the
common key registry at every lead), from `outputs/final/station_metrics.parquet`
(see the results authority below):

| Model | 1 d | 3 d | 7 d |
|---|---:|---:|---:|
| ThermoRoute | 0.640 | 1.337 | 1.694 |
| LightGBM | 0.589 | 1.304 | 1.735 |
| LSTM | 0.663 | 1.358 | 1.712 |
| Plain causal TCN (info-matched) | 0.646 | 1.330 | 1.710 |
| Air2stream (unofficial variant) | 0.719 | 1.478 | 1.825 |
| Damped persistence | 0.789 | 1.454 | 1.773 |
| Persistence | 0.813 | 1.638 | 2.202 |
| Climatology | 1.899 | 1.902 | 1.903 |

ThermoRoute's station-median skill over persistence on the held-out window is
+0.207 / +0.186 / +0.250 and +0.173 / +0.077 / +0.038 over damped persistence at
1 / 3 / 7 d; the median station-level memory gain at 7 d is 0.49 °C and the
median learned gain 0.07 °C (median memory fraction 0.875). The core development
finding — that skill over the strong damped-persistence reference collapses with
lead time — replicates on the independent window. LightGBM leads at 1 d and 3 d;
at 7 d the ThermoRoute point estimate is lower (ΔRMSE −0.009) but not
distinguishable from the tree at the whole-HUC2 cluster level. All tables in the
manuscript regenerate from the persisted prediction table with `scripts/final/`
(results authority).

## Results authority (`outputs/final/`)

Every headline number in the manuscript traces to one row of one table under
`outputs/final/` (protocol v1, `protocols/wrr_strong_accept_protocol_v1.yaml`):

```text
forecast_keys.parquet         common-key registry + per-key history completeness
predictions.parquet           per-key ensemble predictions (regenerated)
station_metrics.parquet       station-first RMSE/MAE/bias/n per model x horizon
paired_effects.parquet        station-level DeltaRMSE and skill per contrast
decomposition_effects.parquet station-level exact G_total = G_memory + G_learned
pooled_metrics.parquet        pooled sensitivity (SI only)
hydrologic_state_effects.parquet  station-first state metrics (issue-time +
                                 outcome-conditioned, 30-key minimum)
basin_attributes.parquet      half-life from the official anchor + registry
spatial_effects.parquet       2x2 spatial factorial cells (geometry x adaptation)
flow_ablation_effects.parquet with-flow vs no-flow LightGBM retraining
paper_values.tex              \\newcommand macros bound to the claim ledger
claim_ledger_resolved.csv     claim-by-claim resolution
result_manifest.json          git SHA, protocol hash, output digests
```

Regeneration (no training, no network):

```bash
python scripts/final/build_results_authority.py     # outputs/final/* + paper_values.tex
python scripts/final/run_mechanism_analysis.py      # states + basin attributes
python scripts/final/run_missingness_sensitivity.py # key-history strata
python scripts/final/generate_manuscript_tables.py  # Section 4.6/4.8 table blocks
python scripts/final/check_manuscript_consistency.py
```

The experimental scripts that produce the spatial and flow tables are
`scripts/final/run_spatial_factorial.py` (LightGBM, 2×2 factorial, five random
split seeds; archived after DLOG-012), `scripts/final/run_information_ladder.py`
(protocol v2: four-level information ladder L0-L3 × geometry, fold-complete
sharded outputs, completeness and key-registry assertions; `--golden-l0`
reproduces the archived local cells bit-for-bit) and
`scripts/final/run_flow_ablation.py`; all are CPU-light and reproducible from
the committed panels. The protocol v2 design, the inference family N1–N15, the
pre-registered predictions P-1–P-4, and the stopping rules are frozen in
`protocols/wrr_strong_accept_protocol_v2.yaml` (seal:
`protocols/wrr_strong_accept_protocol_v2_seal.json`).

## Study design

### Data and splits

- The development panel is `data_usgs/panel_usgs_120v2.parquet`: 657,480 daily
  rows for 120 stable USGS site numbers from 2006-01-01 through 2020-12-31. The
  registry covers 34 states and 15 HUC2 groups and is bound to the panel by
  `data_usgs/frozen_panel_v1.json`.
- The 2021–2023 window is a true temporal holdout: it participated in no
  training, model selection, calibration, or station-inclusion decision. The
  holdout outcomes were acquired and scored after the full development pipeline
  was fixed.
- A reportable station needs at least 100 common valid target keys for the
  relevant horizon and model pair. The primary effect is the median across
  station-level RMSE differences; primary inference uses exact whole-HUC2
  sign-flip p-values, a whole-HUC2 cluster bootstrap for confidence intervals,
  and Holm adjustment across exactly the five comparisons of the primary family.
- The raw feature order is `WTEMP, FLOW, TEMP, PRCP, RHMEAN, DH, WDSP`. `DH` is
  the legacy name for Daymet daylight-period mean incoming shortwave radiation
  (W m⁻²), not day length; `RHMEAN` is a derived humidity proxy. `WLEVEL` may be
  archived as raw provenance but is not consumed by the models.
- The 32-day context is a construction buffer, not an effective 32-day memory
  claim: the router is restricted to lags 0–14 and the two-block, kernel-three
  TCN has a theoretical seven-step receptive field, so no input older than lag
  14 can affect a prediction.
- The pooled-training arms (`ThermoRoute-ext`, `LightGBM-ext`, `LSTM-ext`) are
  trained station-agnostically on the same 120 sites and still consume each
  target site's observed water-temperature history through the issue date. They
  are pooled-preprocessing sensitivity arms, not site-disjoint transfer
  evidence, and are exploratory.

### Development split

| Role | Dates | Use |
|---|---|---|
| Train | 2006–2015 | fit models and preprocessing |
| Validation | 2016–2017 | model selection only |
| Calibration | 2018 | CQR and Platt fit (2018 only); final year of the seasonal event-reference fit (2006–2018) |
| Development evaluation | 2019–2020 | exploratory diagnostics; also the frozen-bundle reproduction tie-back |
| Independent holdout | 2021–2023 | scored once; never used in a development decision |

## Model design

For issue time `t` and horizon `h`, the point forecast has four main pieces:

1. A damped-persistence/climatology anchor supplies a strong conservative
   reference trajectory.
2. A learned flow- and season-conditioned relaxation proposal changes the
   thermal statistical decay behavior, but remains a statistical component
   rather than an energy-balance estimate or measured physical timescale.
3. A sparse horizon-conditioned router selects among the variables and lags;
   a strictly left-looking (non-anticipating) TCN and regime mixture encode
   recent history.
4. A `tanh`-bounded residual limits the point prediction's deviation from its
   anchor. This is an algebraic bounded-deviation property, not a deployment
   or tail-risk guarantee.

The learned models emit an MSE point and distinct pinball q05/q50/q95 heads.
CQR is fitted only on 2018 after member-wise averaging. Its exact split-conformal
order statistic is retained as a signed raw audit value, while the deployed
offset is `qhat_plus = max(raw_qhat, 0)`. The final interval is therefore
`q05 - qhat_plus` to `q95 + qhat_plus`: it may stay unchanged or widen, but it
cannot shrink; q50 is not adjusted. Non-finite offsets or crossed/non-finite
final heads are rejected. Member-wise averaged quantiles are engineering
ensemble summaries, not mixture-distribution quantiles. Neural quantiles are
ordered by construction, so their retained crossing-loss field is
compatibility-only and contributes zero. LightGBM's independently fit heads are
never sorted: the bundle records raw development crossings by member and
horizon, clips q05/q95 to the nominal raw q50 when necessary, and leaves q50
exactly unchanged.

Exceedance events use a seasonal statistical reference fitted on 2006–2018
(development data only); horizon-specific Platt calibration uses 2018 only.
Pinball is evaluated on the nominal pre-CQR heads, coverage and width on the
deployed CQR endpoints, and event metrics on the post-Platt probability. The
Platt calibrator gives each retained station equal total weight in the fitted
probability map. Each model class uses the same five seeds (0–4) and is
compared on identical forecast keys and exact target bytes after ensemble
averaging; the one-factor ablations (no-dynamic-prior, fixed-kappa, no-router,
no-MoE, no-TCN, unbounded) are deletion/intervention sensitivities and do not
prove component necessity or identify a mechanism.

Temporal learned models receive stable site identity, while the pooled-training
arms do not. History cells may be filled by train-only seasonal medians while
retaining missingness masks in the sequence input; there is no minimum observed
fraction in the 32-day context. The optional Air2stream-style reference is an
unofficial style-based implementation, not the official Air2stream code or a
validated reproduction of it.

### Observed 7DADM description

A separate descriptive analysis reads independently sourced observations
identified as daily maximum water temperature, computes a strict
seven-consecutive-day average of daily maxima (7DADM), and compares those
observations with a site-specific standards registry. It does not read or
classify model predictions. An observed value above a registry threshold is
reported only as a descriptive exceedance: it is not a legal or regulatory
decision, does not determine compliance with any law or regulation, and is not
a forecast result.

## Scoring workflow

### 1. Holdout scoring — `scripts/conventional_holdout_2021_2023.py`

The scorer is standalone: it loads the frozen development-period bundles,
applies the preprocessing baked into each bundle, runs inference on the holdout
windows, and scores against the NWIS observed water temperature. It has two
stages:

1. **Development reproduction (2019–2020).** The frozen bundles replay the
   stored development test predictions on 249,072 keys per model-seed and the
   results are compared with the stored development table (currently within
   ~2e-6 maximum absolute difference, atol 1e-2). This is the faithfulness
   proof that loading, preprocessing, and inference reproduce the
   development-period pipeline.
2. **Holdout (2021–2023).** Acquisition is re-parse-only from a content-
   addressed snapshot cache (`SnapshotStore` under `outputs/conventional/
   raw_2021_2023/`) holding the exact raw NWIS, Daymet, and gridMET responses.
   Each registry site receives a typed acquisition status (`OK` / `NO_SERIES` /
   `ALL_SERIES_CONFLICT` / `PARSE_FAILED` / `HTTP_FAILED` / `SNAPSHOT_MISSING`)
   recorded in `cohort_2021_2023.csv`; the panel (`panel_2021_2023.parquet`)
   is content-keyed so a changed registry, interval, or parser version rebuilds
   it. The scorer then builds the frozen-transform temporal windows (118,865
   windows), runs every model ensemble — 13 in total: ThermoRoute and
   DampedPriorOnly, the LSTM and LightGBM baselines, the six one-factor
   ablations (no-dynamic-prior, fixed-kappa, no-router, no-MoE, no-TCN,
   unbounded), and the three pooled-training arms (Persistence /
   DampedPersistence / Climatology are computed in-scorer) — applies the
   frozen CQR + Platt calibration at the same
   point the development pipeline does, and writes:

   - `predictions_2021_2023.parquet` — the per-key prediction table carrying
     `q05/q50/q95` post-CQR with `q05_raw/q50_raw/q95_raw` twins, `p_exceed`
     post-Platt with its raw twin, `conformal_delta_c`, the Platt parameters,
     `calibration_state` (`FROZEN_CQR_PLATT_APPLIED` /
     `NO_FROZEN_CALIBRATION` / `NOT_APPLICABLE_POINT_ONLY`),
     `event_threshold_c`, `event_observed`, `huc2`, `n_members`,
     `bundle_sha256`, and `cohort`. Baselines carry `NOT_APPLICABLE_POINT_ONLY`;
     arms without a bundle carry `NO_FROZEN_CALIBRATION` and are skipped by the
     probability metrics.
   - `holdout_metrics_2021_2023.csv` — long-format `model × horizon × metric`
     table (RMSE, MAE, BIAS, skill vs persistence, skill vs climatology, n).
   - `holdout_summary_2021_2023.json` — interval, station failures, model list,
     window counts, and the wide metrics table.
   - `holdout_tables_2021_2023.tex`, `validation_report_2021_2023.json`,
     `run_manifest.json`, `cohort_2021_2023.csv`,
     `fetch_failures_2021_2023.json`, `panel_cache_key.txt`, and the run log.

The run writes nothing if any validation check fails (on failure it writes only
`validation_report_2021_2023.json`). The checks cover: per-member finiteness
and quantile ordering; strict `q05 < q95` before calibration; post-calibration
ordering, strict width, and `p_exceed ∈ [0,1]`; the prediction-schema check on
the canonical `PRED_COLS`; duplicate keys; registry membership of every
`site_id`; `y_true` tie-back to the rebuilt panel within 1e-6; recomputation of
`event_observed`; common forecast keys per cohort; exhaustive cohort accounting
with `n_stations_reportable` equal to the ≥100-count per horizon; and the
2019–2020 reproduction (which also covers the pooled-training arms). A site
that cannot be acquired is retried; an incomplete cohort is allowed only with
`--allow-incomplete-cohort`, which is stamped into the run manifest.

### 2. Statistic derivation — `scripts/conventional_derive_statistics.py`

All manuscript statistics are derived from the persisted prediction table plus
the station registry — never from a bundle, the panel, or the network, and the
module imports and runs with torch absent (enforced by an AST test). This is
the no-re-run guarantee: every number in the manuscript regenerates in 60–120 s
without touching a model. The derivation uses `src/thermoroute/
conventional_stats.py`:

- `station_metrics` — per-station RMSE with the ≥100-target reportability
  rule; the station-median effect across stations.
- `pooled_metrics` — the pooled rows that reproduce the scorer's numbers
  exactly.
- `skill_table` — skill vs Persistence, DampedPersistence, and Climatology.
- `paired_effects` — per-station effect = candidate − reference (negative
  favours the candidate).
- `cluster_inference` — whole-HUC2 cluster bootstrap, exact sign-flip
  p-values, Holm adjustment over the five-comparison family, and win rate
  (mean effect < 0), with leave-one-cluster-out sensitivity.
- `probability_metrics` — Brier score, coverage, interval width, and interval
  score, skipping models whose `calibration_state` is not
  `FROZEN_CQR_PLATT_APPLIED`.
- `reliability_bins` — ten-bin reliability table for `p_exceed`.

Supporting checks: `scripts/verify_holdout_metrics.py` runs the A1 sanity
checks (completeness, plausibility bands, NaN policy, baseline presence) on the
metrics CSV, and `scripts/build_results_table.py` renders the paper's held-out
result tables (Tables 4.7/4.8) as compilable LaTeX from the same CSV.

## Output artifacts

`outputs/conventional/` holds the complete holdout evidence:

```text
predictions_2021_2023.parquet     per-key prediction table (results.PRED_COLS + calibration columns)
holdout_metrics_2021_2023.csv     long-format model × horizon × metric table
holdout_summary_2021_2023.json    interval, model list, window counts, wide metrics
holdout_tables_2021_2023.tex      paper result tables (built by scripts/build_results_table.py)
panel_2021_2023.parquet           assembled 2021–2023 panel (content-keyed cache)
raw_2021_2023/                    content-addressed raw HTTP response cache (NWIS + Daymet + gridMET)
cohort_2021_2023.csv              per-site acquisition status and conflict-day counts
fetch_failures_2021_2023.json     transient fetch failures (retried)
validation_report_2021_2023.json  validation-check report
validation_2019_2020.json         development reproduction diagnostics (tie-back)
run_manifest.json                 run identity, options, bundle root
panel_cache_key.txt               cache key (registry ‖ interval ‖ parser version)
run_2021_2023.log                 scoring run log
```

## Model bundle provenance

The 15 frozen model bundles (≈700 MB, dominated by the two LightGBM sets) are
gitignored and live under `outputs/models/` by default. `data_usgs/
model_bundle_manifest_v1.json` records each bundle's relative path, `sha256`,
and size. `scripts/verify_model_bundles.py` verifies every declared file under
`--bundle-root` (default `outputs/models`; `THERMOROUTE_BUNDLE_ROOT` overrides),
so a clean checkout plus the deposited bundle pack is provably the scoring
input. The scorer takes `--bundle-root` the same way.

## Repository layout

```text
data/                         legacy inputs for three ordinary monitoring stations (b1, s2, p3;
                              no topology, regulation status, or travel time is established)
data_usgs/                    development panel, station registry, bundle manifest, raw snapshots
src/thermoroute/              model, data, inference, calibration, scoring, and statistics code
scripts/                      development, holdout scoring, derivation, and verification entrypoints
tests/                        unit and integration tests (synthetic fixtures; no bundles, no network)
paper/                        manuscript and submission sources
docs/                         working notes; docs/archive/ holds superseded design records
outputs/                      generated artifacts; outputs/conventional/ is the holdout evidence
```

### Module index (`src/thermoroute/`)

| Module | Purpose |
|---|---|
| `config.py` | central study configuration |
| `checkpoint.py` | training checkpoints and self-describing inference bundles |
| `features.py` | feature engineering: harmonic climatology, tabular lags, DOY terms |
| `frozen_inference.py` | frozen-bundle preprocessing and sequence-model inference |
| `frozen_calibration.py` | frozen CQR + Platt calibration applied at inference time |
| `conventional_acquisition.py` | strict NWIS re-parse from the snapshot cache with typed per-site status |
| `conventional_score.py` | standalone holdout scorer (validation reproduction + holdout scoring) |
| `conventional_stats.py` | pure-statistics derivation (no torch, no bundles) |
| `metrics.py` | point, probabilistic, and event scores |
| `quantiles.py` | quantile-head identity and crossing diagnostics |
| `registry.py` | prediction-key registry utilities for fair comparisons |
| `results.py` | canonical prediction schema and scoring aggregator |
| `significance.py` | significance tools respecting temporal autocorrelation |
| `spatial.py` | station-registry and HUC2-cluster accessors |
| `usgs.py` | large-sample USGS NWIS + Daymet acquisition |

## Tests and verification

The test suite runs on synthetic fixtures with no bundles and no network:

```bash
python -m pytest -q
ruff check src tests
ruff check --select F scripts
mypy src/thermoroute --ignore-missing-imports
python scripts/verify_holdout_metrics.py
python scripts/verify_model_bundles.py --bundle-root outputs/models
```

Key guarantees covered by tests: the derivation and scorer modules import
neither torch nor any deleted module (AST scans); calibration state routing
(raw vs calibrated families); rejection of corrupted prediction tables;
`y_true` tie-back sensitivity; station decode independent of module-level
global state; acquisition-status typing; and exhaustive cohort accounting.

## Reproducibility boundary

The development panel and station registry are committed as Parquet/CSV bytes,
but the original provider HTTP responses for the 2006–2020 development panel
are unavailable, so development reproduction is conditional on the committed
panel. Holdout acquisition, by contrast, archives exact requests, responses,
timestamps, qualifiers, and hashes through the content-addressed snapshot
cache. The model bundles are not in git; the tracked manifest plus
`verify_model_bundles.py` make a clean checkout plus the deposited bundle pack
the provable scoring input.

## License

Code is provided under the repository license. Data redistribution and provider
terms must be reviewed separately before any public release; the redistribution
terms of the bundled third-party AGU LaTeX class must likewise be verified and
recorded in a third-party notice, or that class must be excluded from any
public code archive.

See the [legacy three-site semantics notice](protocols/legacy_three_site_semantics_notice_v1.md) for the historical three-site material and its withdrawal status.

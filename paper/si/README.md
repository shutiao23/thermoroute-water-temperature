# ThermoRoute supporting-information scaffold

**Status:** partially filled. SI01 and SI05 now carry bound empirical cells;
SI02, SI03 and SI15 are normative specifications with no empirical cells by
design; the remaining documents still hold `[pending computation]` placeholders,
and each states which receipt or computation would discharge it.

This directory is the supporting information for the conventional
comparative-holdout manuscript `paper/ThermoRoute_paper.md`. It reports no
invented empirical result, rendered figure, or count: a numeric cell here is
either bound to a named artifact under `outputs/` or is an explicit
`[pending computation]` placeholder, never an estimate standing in for one. This
directory is subordinate to that manuscript and must not revise it.

## Uniform fill rule

The sole result placeholder is:

```text
[pending computation]
```

Use it verbatim for every numeric result, count derived from target outcomes,
score, interval, p-value, or plotted coordinate. Never replace it with a
development-cache value. A value may replace a placeholder only when it has a
bound source pointer, a unit, a derivation, and a verified binding to the
held-out 2021--2023 metric table
`outputs/conventional/holdout_metrics_2021_2023.csv`.

Most numbers written in SI01--SI05 are frozen design or cohort-geometry facts,
and each names its source and role. Two are not, and are labelled as such: the
SI01 HUC2 projection is registry- and cohort-derived (outcome-free, but a
derived count rather than a declared one), and the SI05 five-row family is a
performance result bound to
`outputs/conventional/cluster_inference_2021_2023.json`. All other quantities
remain `[pending computation]`.

## Contents

| File | Static purpose | Source if a value is ever added |
|---|---|---|
| `SI00_inventory.md` | SI inventory and evidence routing | held-out 2021--2023 metric table |
| `SI01_cohort_and_registry.md` | cohort and cluster geometry | registry/panel binding |
| `SI02_information_boundary.md` | issue-time boundary and predictor provenance | acquisition/bridge bindings |
| `SI03_model_equations.md` | identities, units and non-identifiability limits | frozen model suite |
| `SI04_protocol_and_amendments.md` | analysis protocol and model-suite registry | bound model-suite registry |
| `SI05_comparison_family.md` | five-row comparison geometry, estimands and fill schema | held-out-window statistics |
| `SI06_formal_five_rows.md` | five-row result projection (result cells pending) | held-out-window statistics |
| `SI07_all_model_scores.md` | all-model exact-common-key score projection (result cells pending) | held-out-window predictions |
| `SI08_probability_metrics.md` | held-out-window probability/reliability projection (result cells pending), plus the development-period non-reporting disposition | held-out-window probability evaluation |
| `SI09_development_controls.md` | Stage09/09b control geometry and provenance routing | frozen suite and development receipts |
| `SI10_temporal_coverage.md` | availability and temporal sensitivity projection (result cells pending) | temporal-coverage audit |
| `SI11_spatial_sensitivity.md` | HUC/leave-cluster sensitivity projection (result cells pending) | spatial-sensitivity receipt and registry binding |
| `SI12_qc_qualifiers.md` | outcome-QC, qualifier and exact-A projection (result cells pending) | raw-evidence and QC-cluster-geometry bindings |
| `SI13_external_history_arm.md` | external history-dependent arm, explicitly not ungauged | external-suite receipt |
| `SI14_missingness_failures.md` | missingness, attrition and failure-case projection (result cells pending) | bound attrition/failure-case evidence |
| `SI15_reproduction_hashes.md` | commands, environment and immutable binding projection | replay and reproduction receipts |
| `SI16_rights_data_dictionary.md` | rights-decision and machine-readable data-dictionary routing | qualified rights decisions and release manifest |

## Evidence boundary

Each visible value in the SI must bind the value, unit, evidence role, source
pointer, derivation, and rounding. The held-out 2021--2023 metric table
`outputs/conventional/holdout_metrics_2021_2023.csv` (long format:
`model,horizon,metric,value,n`) is the source of truth for the Section 4.6
metric cells; values derived from it are permitted, but they must not be
presented as independent of that table.

### Cell-level binder contract

One row-level source pointer is never sufficient for a multi-value row. SI06--SI16
and the supporting figures use this generic hidden binder shape:

```text
tables.<si_or_figure_id>.rows[<binder_row_id>].cells.<field>.value_id
```

Every visible empirical value, count, hash, status, decision, interval endpoint
and plotted coordinate has its own `value_id`. A composite display cell must map
each displayed subvalue separately. Reusing one value ID for different values, or
providing only a row-level `source value_id`, is an error.

## Prohibitions

- Do not read or cite Stage-09 development caches as held-out-window evidence.
- Do not add a score, effect, interval, p-value, coverage value, reliability
  point, or claim verdict before the held-out 2021--2023 metric computation.
- Do not alter a registered comparison, model budget, target key, cohort, or
  margin from this directory.
- The study is reported for a fixed availability-enriched cohort; no SI may turn
  its approximate clustered sensitivities into superiority, non-inferiority,
  equivalence, national, operational, ungauged, safety, or causal claims.
- Do not write "quantile crossing" in reference to the withheld development-period
  probabilistic stage. Zero strict ordering violations were measured; the correct
  term is "zero-width (degenerate) nominal interval". See SI08 §3.

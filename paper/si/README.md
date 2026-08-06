# ThermoRoute supporting-information scaffold

**Status:** DRAFT.

This directory is the supporting-information (SI) scaffold for the conventional
comparative-holdout manuscript `paper/ThermoRoute_paper.md`. It reports no
invented empirical result, rendered figure, or count beyond the frozen
design/cohort-geometry facts already written in SI01--SI05; every other numeric
result, score, interval, p-value, or plotted coordinate is a placeholder. This
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

Numbers already written in SI01--SI05 are frozen design or cohort-geometry
facts only. Each one names its source and role; none is a performance result.
All other quantities remain `[pending computation]`.

## Contents

| File | Static purpose | Source if a value is ever added |
|---|---|---|
| `SI00_inventory.md` | SI inventory and evidence routing | held-out 2021--2023 metric table |
| `SI01_cohort_and_registry.md` | cohort and cluster geometry | registry/panel binding |
| `SI02_information_boundary.md` | issue-time boundary and predictor provenance | acquisition/bridge bindings |
| `SI03_model_equations.md` | identities, units and non-identifiability limits | frozen model suite |
| `SI04_protocol_and_amendments.md` | analysis protocol and model-suite registry | bound model-suite registry |
| `SI05_comparison_family.md` | five-row comparison geometry, estimands and fill schema | held-out-window statistics |
| `SI06_formal_five_rows_RECEIPT.md` | empty five-row result projection | held-out-window statistics |
| `SI07_all_model_scores_RECEIPT.md` | empty all-model exact-common-key score projection | held-out-window predictions |
| `SI08_probability_metrics_RECEIPT.md` | empty held-out-window probability/reliability projection, plus the development-period non-reporting disposition | held-out-window probability evaluation |
| `SI09_development_controls_RECEIPT.md` | Stage09/09b control geometry and provenance routing | frozen suite and development receipts |
| `SI10_temporal_coverage_RECEIPT.md` | empty availability and temporal sensitivity projection | temporal-coverage audit |
| `SI11_spatial_sensitivity_RECEIPT.md` | empty HUC/leave-cluster sensitivity projection | spatial-sensitivity receipt and registry binding |
| `SI12_qc_qualifiers_RECEIPT.md` | empty outcome-QC, qualifier and exact-A projection | raw-evidence and QC-cluster-geometry bindings |
| `SI13_external_history_arm_RECEIPT.md` | external history-dependent arm, explicitly not ungauged | external-suite receipt |
| `SI14_missingness_failures_RECEIPT.md` | empty missingness, attrition and failure-case projection | bound attrition/failure-case evidence |
| `SI15_reproduction_hashes_RECEIPT.md` | commands, environment and immutable binding projection | replay and reproduction receipts |
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

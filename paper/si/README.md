# ThermoRoute supporting-information scaffold

**Status:** DRAFT / PRE-OPENING / NOT RECEIPT.

This directory is a static Route-A supporting-information (SI) scaffold. It
does not contain an empirical result, receipt, rendered figure, target-period
outcome, or authority to open the confirmation interval. The frozen PRE
manuscript is `paper/ThermoRoute_paper.md`; this directory is subordinate to
that manuscript and must not revise it.

## Uniform fill rule

The sole result placeholder is:

```text
[pending — 探索期数据，不可写入结论]
```

Use it verbatim for every numeric result, verdict, count derived from target
outcomes, score, interval, p-value, plotted coordinate, or receipt-derived
claim. Never replace it with a development-cache value. A verified POST build
may replace a placeholder only when its value has one `value_id`, a bound source
pointer, a unit, a derivation, and a verified receipt binding.

Numbers already written in SI01--SI05 are frozen PRE design or cohort-geometry
facts only. Each one names its source and role; none is a performance result.
All other quantities remain `[pending — 探索期数据，不可写入结论]`.

## Contents

| File | Static purpose | POST authority if a value is ever added |
|---|---|---|
| `SI00_inventory.md` | SI inventory and evidence routing | `post_paper_evidence_v1.json` plus `post_render_receipt_v1.json` |
| `SI01_cohort_and_registry.md` | frozen cohort and cluster geometry | registry/panel binding plus receipt-bound projection |
| `SI02_information_boundary.md` | issue-time boundary and predictor provenance | acquisition/bridge bindings plus opening receipt |
| `SI03_model_equations.md` | identities, units and non-identifiability limits | frozen model suite and receipt-bound model artifacts |
| `SI04_protocol_and_amendments.md` | protocol, amendments, erratum and chronology | bound authorization, claim registry and opening receipt |
| `SI05_comparison_family.md` | five-row geometry, estimands and fill schema | `formal_tests[*]` in the verified opening receipt |
| `SI06_formal_five_rows_RECEIPT.md` | empty five-row result projection | verified opening receipt plus bound inference/QC gates |
| `SI07_all_model_scores_RECEIPT.md` | empty all-model exact-common-key score projection | receipt-bound predictions and score derivations |
| `SI08_probability_metrics_RECEIPT.md` | empty probability/reliability projection | verified probability-evaluation receipt |
| `SI09_development_controls_RECEIPT.md` | Stage09/09b control geometry and provenance routing | complete development receipts and frozen suite |
| `SI10_temporal_coverage_RECEIPT.md` | empty availability and temporal sensitivity projection | temporal-coverage receipt and opening evidence |
| `SI11_spatial_sensitivity_RECEIPT.md` | empty HUC/leave-cluster sensitivity projection | spatial-sensitivity receipt and registry binding |
| `SI12_qc_qualifiers_RECEIPT.md` | empty outcome-QC, qualifier and exact-A projection | raw-evidence, QC-gate and opening bindings |
| `SI13_external_history_arm_RECEIPT.md` | external history-dependent arm, explicitly not ungauged | external-suite receipt and exact-key binding |
| `SI14_missingness_failures_RECEIPT.md` | empty missingness, attrition and failure-case projection | bound attrition/failure-case evidence |
| `SI15_reproduction_hashes_RECEIPT.md` | commands, environment and immutable binding projection | replay and POST-render receipts |
| `SI16_rights_data_dictionary.md` | rights-decision and machine-readable data-dictionary routing | qualified rights decisions and release manifest |

## Receipt boundary

The intended POST evidence manifest schema is the design specified in
`docs/POST_PAPER_PROJECTION_DESIGN.md`:

```text
format
route_id
authority
source_bindings
values
sections
tables
figures
si
outputs
```

For each visible value, `values[value_id]` must bind the value, unit, evidence
role, source pointer, derivation, and rounding. The later
`post_render_receipt_v1.json` is intended to bind the evidence manifest,
rendered SI and figures, submission artifacts, environment, and SHA-256 values.
Those files do not exist in this PRE scaffold; their names are design targets,
not evidence.

### Cell-level binder contract

One row-level source pointer is never sufficient for a multi-value row. SI06–SI16
and FigS1–FigS8 use this generic hidden binder shape:

```text
tables.<si_or_figure_id>.rows[<binder_row_id>].cells.<field>.value_id
```

Every visible empirical value, count, hash, status, decision, interval endpoint
and plotted coordinate has its own `value_id`. A composite display cell must map
each displayed subvalue separately. The referenced `values[value_id]` object
contains the unit (or explicit dimensionless/status/hash role), evidence role,
exact source pointer, derivation and rounding/format rule. Reusing one value ID
for different values, or providing only a row-level `source value_id`, rejects
the POST build.

## Prohibitions

- Do not read or cite Stage-09 development caches as confirmation evidence.
- Do not add a score, effect, interval, p-value, coverage value, reliability
  point, or claim verdict before receipt verification.
- Do not alter a registered comparison, model budget, target key, cohort, margin,
  inference gate, or claim language from this directory.
- Route A remains fixed-cohort descriptive; no SI may turn its
  assumption-conditional sensitivities into superiority, non-inferiority,
  equivalence, national, operational, ungauged, safety, or causal claims.

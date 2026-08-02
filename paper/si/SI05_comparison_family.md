# SI05 — frozen comparison family and fill schema

**Status:** DRAFT / PRE-OPENING / NOT RECEIPT.

This document fixes display geometry for the registered Route-A family. It
reports no effect, performance number, interval, p-value, Holm value or
decision. Every result-bearing cell is
`[pending — 探索期数据，不可写入结论]`.

## Registered five-row geometry

| Row | Candidate − reference | Horizon | Frozen margin / ceiling | PRE role |
|---:|---|---:|---:|---|
| 1 | ThermoRoute − damped persistence | 1 day | 0.00 °C | registered descriptive comparison |
| 2 | ThermoRoute − damped persistence | 3 days | 0.00 °C | registered descriptive comparison |
| 3 | ThermoRoute − damped persistence | 7 days | 0.00 °C | registered descriptive comparison |
| 4 | ThermoRoute − LightGBM | 3 days | +0.05 °C ceiling | registered descriptive comparison |
| 5 | ThermoRoute − LightGBM | 7 days | +0.05 °C ceiling | registered descriptive comparison |

Source: `paper/ThermoRoute_paper.md` §4.2 and
`paper/FIGURE1_AND_SI_SKELETON.md` §1d. These margins are frozen design
quantities, not ecological, biological, regulatory or stakeholder-derived
thresholds.

## Estimand and scope

`paper/ThermoRoute_paper.md` §§4.1--4.2 is the source for the following
registered analysis geometry. For a station and horizon, paired RMSE is eligible
only with at least 100 valid paired targets for both models. The primary effect
is the unweighted median of station-level candidate-minus-reference RMSE, so
daily rows do not determine between-station weighting directly. Comparisons use
identical station/date/horizon target keys.

The same PRE source specifies exact whole-HUC2 sign-flip enumeration and a
10,000-draw whole-HUC2 cluster bootstrap, with Holm adjustment over exactly
these five rows. These values are design geometry. Because the cohort has at
most 15 HUC2 groups, the minimum-30-cluster eligibility gate necessarily fails:
each reported row is fixed-cohort descriptive and any interval, sign-flip p-value
and Holm value is an assumption-conditional sensitivity. No row can support
superiority, non-inferiority, equivalence, parity, national-representativeness or
operational claims.

## Receipt-to-cell schema

The future renderer must take one and only one matching object from opening
receipt `formal_tests[*]` for every row. Required fields are:

```text
test_identity
candidate_model
reference_model
horizon
margin_or_ceiling
status
effect
effect_unit
ci_lower
ci_upper
station_count
cluster_count
win_rate
raw_p
holm_p
bound_checks
inference_gate_status
source_pointer
value_ids
derivations
rounding_rules
```

These are the declared projection schema from
`docs/POST_PAPER_PROJECTION_DESIGN.md`; the names do not assert a present
receipt schema or authorize schema drift. The renderer must reject a missing,
extra, non-finite, unbound or non-matching row.

Each visible result cell is resolved through a binder map rather than copied
directly from a familiar field name:

```text
tables.si06.rows[test_identity].cells.effect.value_id
tables.si06.rows[test_identity].cells.ci_lower.value_id
tables.si06.rows[test_identity].cells.ci_upper.value_id
tables.si06.rows[test_identity].cells.station_count.value_id
tables.si06.rows[test_identity].cells.cluster_count.value_id
tables.si06.rows[test_identity].cells.win_rate.value_id
tables.si06.rows[test_identity].cells.raw_p.value_id
tables.si06.rows[test_identity].cells.holm_p.value_id
tables.si06.rows[test_identity].cells.gate_verdict.value_id
```

Every referenced `values[value_id]` must contain the visible value, unit,
evidence role, exact source pointer, derivation and rounding rule. Figure 1d and
the main T2 table must use the same value IDs, not independently recompute or
transcribe the row.

## Empty result table

| Row | Effect: station-median ΔRMSE | cluster-bootstrap CI | sign-flip p | Holm p | bound/gate verdict |
|---:|---|---|---|---|---|
| 1 | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` |
| 2 | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` |
| 3 | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` |
| 4 | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` |
| 5 | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` |

Exploratory 2019–2020 diagnostics, older noncanonical outputs and Stage-09
caches are not admissible sources for this table.

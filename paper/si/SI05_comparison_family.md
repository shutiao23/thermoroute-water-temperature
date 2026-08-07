# SI05 — comparison family and fill schema

**Status:** scaffold finalized; empirical values `[pending computation]`.

This document fixes display geometry for the study's five-row comparison family.
It reports no effect, performance number, interval, p-value, Holm value or
decision. Every result-bearing cell is `[pending computation]`.

## Five-row comparison geometry

| Row | Candidate − reference | Horizon | Reference margin | Role |
|---:|---|---:|---:|---|
| 1 | ThermoRoute − damped persistence | 1 day | 0.00 °C | primary comparison |
| 2 | ThermoRoute − damped persistence | 3 days | 0.00 °C | primary comparison |
| 3 | ThermoRoute − damped persistence | 7 days | 0.00 °C | primary comparison |
| 4 | ThermoRoute − LightGBM | 3 days | +0.05 °C ceiling | primary comparison |
| 5 | ThermoRoute − LightGBM | 7 days | +0.05 °C ceiling | primary comparison |

Source: `paper/ThermoRoute_paper.md` §3.6 and `paper/FIGURE1_AND_SI_SKELETON.md`
§1d. The +0.05 °C ceiling is a stated numerical reference for the LightGBM
comparison, not a decision threshold; it is not an ecological, biological,
regulatory, or stakeholder-derived value.

## Estimand and scope

`paper/ThermoRoute_paper.md` §3.6 is the source for the following analysis
geometry. For a station and horizon, paired RMSE is eligible only with at least
100 valid paired targets for both models. The primary effect is the unweighted
median of station-level candidate-minus-reference RMSE, so daily rows do not
determine between-station weighting directly. Comparisons use identical
station/date/horizon target keys.

The same source specifies exact whole-HUC2 sign-flip enumeration and a
10,000-draw whole-HUC2 cluster bootstrap, with Holm adjustment over exactly
these five rows. These values are design geometry. Because the cohort has at
most 15 HUC2 groups (an inverse-Herfindahl effective cluster count of about
9.54) and was not probability-sampled, the clustered intervals, sign-flip
p-values, and Holm values are approximate descriptive sensitivities. No row can
support superiority, non-inferiority, equivalence, parity,
national-representativeness, or operational claims.

## Source-to-cell schema

The renderer takes one and only one matching object from the held-out-window
statistics for every row. Required fields are:

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
source_pointer
value_ids
derivations
rounding_rules
```

The renderer must reject a missing, extra, non-finite, unbound, or non-matching
row. Each visible result cell is resolved through a binder map rather than copied
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
```

Every referenced `values[value_id]` must contain the visible value, unit,
evidence role, exact source pointer, derivation, and rounding rule. Figure 1d and
the main Table 4.6 must use the same value IDs, not independently recompute or
transcribe the row.

## Empty result table

| Row | Effect: station-median ΔRMSE | cluster-bootstrap CI | sign-flip p | Holm p |
|---:|---|---|---|---|
| 1 | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` |
| 2 | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` |
| 3 | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` |
| 4 | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` |
| 5 | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` |

Development-period (2019--2020) diagnostics, older noncanonical outputs, and
Stage-09 caches are not admissible sources for this table.

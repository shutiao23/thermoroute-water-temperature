# SI06 — five-row result projection

**Status:** DRAFT.

This file fixes the result-table geometry only. It is not evidence that an
evaluation has run. Every result-bearing cell remains `[pending computation]`
until the held-out 2021--2023 statistics supply exactly one bound row.

| Row | Candidate − reference | Horizon | Margin / ceiling | Effect (°C) | 95% CI (°C) | stations | clusters | raw p | Holm p |
|---:|---|---:|---:|---|---|---|---|---|---|
| 1 | ThermoRoute − damped persistence | 1 day | 0.00 °C | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` |
| 2 | ThermoRoute − damped persistence | 3 days | 0.00 °C | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` |
| 3 | ThermoRoute − damped persistence | 7 days | 0.00 °C | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` |
| 4 | ThermoRoute − LightGBM | 3 days | +0.05 °C ceiling | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` |
| 5 | ThermoRoute − LightGBM | 7 days | +0.05 °C ceiling | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` |

The renderer must reject missing, duplicate, extra, or non-finite rows. The
clustered intervals and p-values are approximate descriptive sensitivities for a
fixed, availability-enriched cohort with at most 15 HUC2 groups; they must not be
written as superiority, non-inferiority, equivalence, parity, or
national/U.S.-river generalization.

## Hidden binder-map contract

The visible table never treats a row-level source pointer as sufficient. For
each `test_identity`, the source manifest must provide cell-specific `value_id`
references for effect, both CI endpoints, station count, cluster count, win rate,
raw p, and Holm p. Each `values[value_id]` object must bind unit, evidence role,
exact source pointer, derivation, and rounding rule. The same IDs populate
Figure 1d and the main Table 4.6. Missing, duplicated, or differently rounded
mappings reject the render.

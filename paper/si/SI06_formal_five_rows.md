# SI06 — five-row result projection

**Status:** filled from `outputs/final/cluster_inference_2021_2023.json`
(cluster map fixed 2026-08-08: HUC2 codes are zero-padded, giving exactly 15
whole-HUC2 clusters; earlier artifacts that treated single-digit codes as
per-site clusters are withdrawn).

This file fixes the result-table geometry only. It is not evidence that an
evaluation has run. Every result-bearing cell is filled from the persisted derivation tables
until the held-out 2021--2023 statistics supply exactly one bound row.

| Row | Candidate − reference | Horizon | Effect (°C) | 95% CI (°C) | stations | clusters | raw p | Holm p |
|---:|---|---:|---:|---|---|---|---|---|
| 1 | ThermoRoute − damped persistence | 1 d | −0.129 | [−0.199, −0.076] | 116 | 15 | <0.001 | 0.0002 |
| 2 | ThermoRoute − damped persistence | 3 d | −0.108 | [−0.143, −0.073] | 116 | 15 | <0.001 | 0.0002 |
| 3 | ThermoRoute − damped persistence | 7 d | −0.069 | [−0.086, −0.057] | 116 | 15 | <0.001 | 0.0002 |
| 4 | ThermoRoute − LightGBM | 3 d | +0.015 | [+0.010, +0.025] | 116 | 15 | 0.999 | 0.999 |
| 5 | ThermoRoute − LightGBM | 7 d | −0.009 | [−0.017, +0.002] | 116 | 15 | 0.074 | 0.148 |

The renderer must reject missing, duplicate, extra, or non-finite rows. The
clustered intervals and p-values are approximate descriptive sensitivities for a
fixed, availability-enriched cohort with at most 15 HUC2 groups; they must not be
written as superiority, non-inferiority, equivalence, parity, or
national/U.S.-river generalization. The seven-day ThermoRoute−LightGBM row is a
point estimate whose interval includes zero; the manuscript reports it as such
and does not claim significance for it.

## Hidden binder-map contract

The visible table never treats a row-level source pointer as sufficient. For
each `test_identity`, the source manifest must provide cell-specific `value_id`
references for effect, both CI endpoints, station count, cluster count, win rate,
raw p, and Holm p. Each `values[value_id]` object must bind unit, evidence role,
exact source pointer, derivation, and rounding rule. The same IDs populate
Figure 1d and the main Table 4.6. Missing, duplicated, or differently rounded
mappings reject the render.

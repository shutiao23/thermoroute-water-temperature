# SI10 — temporal coverage and time-block sensitivity

**Status:** scaffold finalized; empirical values `[pending computation]`.

This SI projects availability and time-block sensitivity for the held-out
2021--2023 window. It is not all-calendar-day performance: the renderer must
distinguish calendar availability, issue-history availability, target
observability, and exact paired-model availability, and a minimum-target rule
cannot be presented as all-calendar-day performance.

## Temporal-coverage projection (target period)

| Horizon | year/season/block | eligible issues | eligible targets | paired keys | stations | completeness | effect/score | binder row ID |
|---:|---|---|---|---|---|---|---|---|
| *(1/3/7)* | *(declared stratum)* | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` |

## Source and fill routing

Each `[pending computation]` cell binds, for the row's horizon (1, 3, or 7 days)
and declared stratum (year/season/block), the column's named quantity under the
frozen temporal-coverage policy
(`protocols/route_a_temporal_coverage_policy_v1.json`) and the opening/coverage
receipts:

- **eligible issues / eligible targets** — counts of issue dates and observable
  target dates, per stratum, derived from the held-out-window receipt.
- **paired keys / stations** — the exact common-key model-pair count and the
  contributing station count, per stratum.
- **completeness** — the fraction of declared targets that are observable and
  paired, per stratum.
- **effect/score** — a descriptive within-stratum sensitivity of the candidate
  effect or score, never a redefinition of the primary estimand.
- **binder row ID** — the `tables.si10.rows[<id>]` cell-level binder key.

## Relationship to the figures

Figure 4 carries regional/seasonal heterogeneity and what coverage costs, and
Figure S6 carries temporal opportunity, missingness, and attrition. Both consume
this table's strata, denominators, and completeness values and must agree with
them in value, unit, and rounding. Both figures are POST-gated and blocked on
the test-window evaluation receipt, so no coordinate in either is available yet.

Every visible denominator, count, completeness value and result follows the
README cell-level binder contract.

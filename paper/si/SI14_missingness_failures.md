# SI14 — missingness, attrition and failure cases

**Status:** scaffold finalized; empirical values `[pending computation]`.

This SI projects missingness, attrition, and failure cases for the held-out
2021--2023 window. Denominators, missingness reasons, and failed cases must
remain visible: outcome-dependent threshold changes, station replacement,
subgroup rescue, and omission of unfavourable rows are prohibited.

## Missingness and failure projection (target period)

| Stage/reason/stratum | eligible before | retained after | attrition | history completeness | effect/score | failure disposition | binder row ID |
|---|---|---|---|---|---|---|---|
| *(declared row)* | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` |

## Source and fill routing

Each `[pending computation]` cell binds, for the row's declared
stage/reason/stratum, the column's named quantity under the bound
attrition/failure-case evidence:

- **eligible before / retained after** — counts before and after attrition, per
  stratum.
- **attrition** — the fraction lost, with the recorded reason.
- **history completeness** — the fraction of required history present, per
  stratum.
- **effect/score** — a descriptive within-stratum sensitivity, never a
  redefinition of the primary estimand.
- **failure disposition** — the recorded fail-closed status and reason for any
  failed case.
- **binder row ID** — the `tables.si14.rows[<id>]` cell-level binder key.

A failed case renders with its disposition, never as a silent omission.

## Relationship to the figures

Figure S6 carries temporal opportunity, missingness, and attrition, and Figure
S8 carries failure cases. Both must agree with this table in denominators,
attrition, completeness, and failure disposition. Both figures are POST-gated
and blocked on the test-window evaluation receipt, so no coordinate in either
is available yet.

Every displayed denominator, attrition, completeness value, result and failure
status follows the README cell-level binder contract.

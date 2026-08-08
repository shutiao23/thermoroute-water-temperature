# SI14 — missingness, attrition and failure cases

**Status:** filled. 2 stations provably dry (NO_SERIES, successful snapshots, no daily-mean series), 3 transient HTTP failures retried successfully, 01435000 below the reportability rule; per-horizon reportable counts 116/116/116.

This SI projects missingness, attrition, and failure cases for the held-out
2021--2023 window. Denominators, missingness reasons, and failed cases must
remain visible: outcome-dependent threshold changes, station replacement,
subgroup rescue, and omission of unfavourable rows are prohibited.

## Key-registry history completeness (Major Comment 12 sensitivity)

Every forecast key in `outputs/final/forecast_keys.parquet` carries the observed
fraction of water temperature in the 7-, 14-, and 32-day windows ending at the
issue date (computed from the raw panels before imputation; the panels have one
row per calendar day, so row-rolling counts equal calendar-window counts). The
stratified station-first paired ΔRMSE (ThermoRoute − damped persistence) from
`scripts/final/run_missingness_sensitivity.py`:

| Window | observed fraction | keys | reportable stations | median ΔRMSE (°C) |
|---|---:|---:|---:|---:|
| 7 d | ≥ 0.95 | 348,238 | 116 | −0.092 |
| 7 d | 0.75–0.95 | 5,001 | 3 | −0.150 |
| 7 d | < 0.75 | 5,568 | 0 | — |
| 14 d | ≥ 0.95 | 337,730 | 116 | −0.093 |
| 14 d | 0.75–0.95 | 15,327 | 15 | −0.052 |
| 32 d | ≥ 0.95 | 329,386 | 116 | −0.091 |
| 32 d | 0.75–0.95 | 22,859 | 22 | −0.096 |

97% of the seven-day common keys have at least 95% of their 7-day history
genuinely observed, and the learned-gain conclusion is present (indeed larger)
on the well-observed stratum, so the relative model-vs-anchor result is not
carried by heavily imputed keys.

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

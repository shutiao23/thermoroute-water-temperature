# SI13 — external history-dependent cohort

**Status:** scaffold finalized; empirical values `[pending computation]`.

This arm is site-ID-disjoint but consumes target-site WTEMP history. It is a
known-gauge external arm, not ungauged prediction, river-network transfer,
national inference, or operational as-issued forecasting. It is excluded from
the development registry and bound to its own external-suite receipt.

## External-arm projection

| Model/comparison | Horizon | sites | paired keys | RMSE/effect (°C) | interval/status | binder row ID |
|---|---:|---|---|---|---|---|
| *(receipt row)* | *(1/3/7)* | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` |

## Source and fill routing

Each `[pending computation]` cell binds, for the row's external model/comparison
and horizon (1, 3, or 7 days), the column's named quantity under the
external-suite receipt:

- **sites** — the site-ID-disjoint external cohort count.
- **paired keys** — the exact common-key set shared with the comparison model.
- **RMSE/effect (°C)** — the candidate-minus-reference RMSE or effect, in °C.
- **interval/status** — the clustered interval endpoint or status token.
- **binder row ID** — the `tables.si13.rows[<id>]` cell-level binder key.

The receipt must bind cohort identity, exclusion from the development registry,
history requirement, predictor products, exact keys, and model suite. Metadata
disjointness alone is not evidence of hydrologic independence.

## Relationship to the figures

Figure 3 (development period, 2019--2020) and Figure S8 (target period) both
read this arm. The two carry different evidence periods and may never be
compared numerically with each other; each must agree with this table only on
the coordinates it actually shares. Figure 3 and Figure S8 are POST-gated and
blocked on the test-window evaluation receipt, so no coordinate in either is
available yet.

Every displayed count, score, effect, interval endpoint and status follows the
README cell-level binder contract.

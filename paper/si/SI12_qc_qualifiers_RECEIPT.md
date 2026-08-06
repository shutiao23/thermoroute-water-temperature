# SI12 — outcome QC, qualifiers and exact-A audit

**Status:** DRAFT.

No target outcome or provider response is represented here. A future row must be
traceable to immutable raw bytes, normalized series identity and the outcome-QC
gate.

| QC dimension | series/qualifier subset | raw rows | retained rows | exclusions/conflicts | sensitivity/status | binder row ID |
|---|---|---|---|---|---|---|
| statistic/unit/series identity | *(declared subset)* | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` |
| qualifier/method continuity | *(declared subset)* | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` |
| flatline/range/change-point audit | *(declared subset)* | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` |

`exact-A` means the declared primary analysis is reproduced from the bound raw
and normalized evidence under the frozen QC policy. It is not permission to
replace sites, tune thresholds or suppress an adverse QC status.
Every displayed count, exclusion/conflict value and status follows the README
cell-level binder contract.

# SI12 — outcome QC, qualifiers and exact-A audit

**Status:** scaffold finalized; empirical values `[pending computation]`.

This SI projects the outcome quality-control gate, qualifier evidence, and the
exact-A audit for the held-out 2021--2023 window. No target outcome or provider
response is represented here yet. A future row must be traceable to immutable
raw bytes, normalized series identity, and the outcome-QC gate.

## QC projection (target period)

| QC dimension | series/qualifier subset | raw rows | retained rows | exclusions/conflicts | sensitivity/status | binder row ID |
|---|---|---|---|---|---|---|
| statistic/unit/series identity | *(declared subset)* | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` |
| qualifier/method continuity | *(declared subset)* | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` |
| flatline/range/change-point audit | *(declared subset)* | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` |

## Source and fill routing

Each `[pending computation]` cell binds, for the row's QC dimension and declared
series/qualifier subset, the column's named quantity under the raw-evidence
bindings and the QC-cluster-geometry receipts:

- **raw rows / retained rows** — counts before and after the QC gate, per
  subset, derived from the bound raw bytes.
- **exclusions/conflicts** — the recorded exclusion reasons and qualifier
  conflicts, never silently removed.
- **sensitivity/status** — the QC status (pass/fail/exact-A) and any
  within-subset sensitivity.
- **binder row ID** — the `tables.si12.rows[<id>]` cell-level binder key.

`exact-A` means the declared primary analysis is reproduced from the bound raw
and normalized evidence under the frozen QC policy. It is not permission to
replace sites, tune thresholds, or suppress an adverse QC status. A passed QC
claim requires exact-A evidence; a missing value renders as the placeholder,
never as an implied pass.

## Relationship to Figure S8

Figure S8 carries outcome QC (with the external-history arm and failure cases).
Its QC panels must agree with this table in raw/retained counts, exclusions, and
status. Figure S8 is POST-gated and blocked on the test-window evaluation
receipt, so no coordinate in it is available yet.

Every displayed count, exclusion/conflict value and status follows the README
cell-level binder contract.

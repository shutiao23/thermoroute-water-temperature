# SI09 — Stage09 and Stage09b development controls

**Status:** development-period-only analysis. The information-matched plain controls (PlainMLP-7var, PlainCausalTCN-7var) reproduce the stored development arm predictions exactly (max abs diff 0.0, G15 gate) but were not admitted to the held-out window in this submission; their held-out cells are therefore not computed, not reported, and not invented.

Stage09 and Stage09b are development-only controls. Their completion receipts
may establish provenance and matrix completeness, but cannot by themselves fill
target-period performance or change the study comparison eligibility. They are
named in manuscript §4.1 as fixed-seed (Stage09) and exact 45-member (Stage09b)
matrix geometry, not as scored comparisons.

## Control geometry (development period)

| Stage | arm/model | seed registry | information set | parameter/search budget | completion receipt | projected sensitivity | binder row ID |
|---|---|---|---|---|---|---|---|
| Stage09 | *(frozen control/model)* | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` |
| Stage09b | *(frozen matched arm)* | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` |

## Source and fill routing

Each `[pending computation]` cell is a development-period provenance or
sensitivity value, never a target-period result. It binds, for the row's stage
and arm, the column's named field under the frozen model-suite registry and the
Stage09/Stage09b receipts:

- **seed registry** — the fixed seed list for the stage (Stage09: five seeds per
  control; Stage09b: the 45-member matrix), read from the frozen suite.
- **information set** — the matched predictor/feature set the arm may see, from
  the Stage-09b information-matched control definition.
- **parameter/search budget** — the predeclared tuning budget for the arm, from
  the model-matrix amendment.
- **completion receipt** — the bound `{path, sha256}` of the stage completion
  receipt, present only after the frozen suite has run.
- **projected sensitivity** — a descriptive development-period sensitivity of
  the control's effect, never a target-period performance number.
- **binder row ID** — the `tables.si09.rows[<id>]` cell-level binder key.

No Stage-09b value may appear inside a target-period figure, and no development
cache may substitute for a bound receipt.

## Relationship to Figure S10

Figure S10 is the graphical reading of the registered architecture interventions
and the bounded-deviation audit. Its control rows must agree with this table in
identity, seed, budget, and receipt status; this SI is the development-period
companion table to Figure S10. Figure S10 is POST-gated and blocked on the
test-window evaluation receipt, so no coordinate in it is available yet.

The POST evidence chain must bind authorization → frozen suite → all required
stage receipts → predictions/metrics → displayed value. Partial members, failed
runs and diagnostic caches cannot populate this SI. Every displayed count,
budget, receipt status and sensitivity follows the README cell-level binder
contract.

# SI09 — Stage09 and Stage09b development controls

**Status:** development-period analysis with an independent-window re-test. The
information-matched plain controls (PlainMLP-7var, PlainCausalTCN-7var)
reproduce the stored development arm predictions exactly (max abs diff 0.0, G15
gate) and, following the external review (Major Comment 5), were re-scored on
the 2021–2023 holdout window with the same frozen weights, anchor, inputs, and
common key registry (manuscript Section 4.5, Table 4.7a, and SI07). Their
held-out cells are uncalibrated (`NO_FROZEN_CALIBRATION`) and enter point
comparisons only.

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
companion table to Figure S10.

Figure S10 was for a long time described here as "POST-gated and blocked on the
test-window evaluation receipt, so no coordinate in it is available yet". That
was true when written and stopped being true when the held-out evaluation of
manuscript Section 4.3 ran; the blocker outlived the block. The figure is now
rendered from `outputs/final/paired_effects.parquet` on the held-out 2021–2023
window by `paper/si/figures/render_figS10_architecture_controls.py`, so it
carries held-out coordinates while this table remains the development-period
companion. The two must not be read as the same period, and neither value may be
compared numerically with the other.

What it shows: of the six one-factor controls, only removing the temporal
encoder separates from the rest, and every other deletion or intervention sits
within a few hundredths of a degree of the full model. These remain deletion and
intervention sensitivities on paired keys; they do not establish that any
component is necessary.

The POST evidence chain must bind authorization → frozen suite → all required
stage receipts → predictions/metrics → displayed value. Partial members, failed
runs and diagnostic caches cannot populate this SI. Every displayed count,
budget, receipt status and sensitivity follows the README cell-level binder
contract.


## One-factor ablation accuracy and skill (relocated Tables 4.8a-b)

<!-- TABLE 4.8a (generated) -->

| Model | RMSE 1 | RMSE 3 | RMSE 7 | MAE 1 | MAE 3 | MAE 7 | bias 1 | bias 3 | bias 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TR-fixedKappa | 0.635 | 1.333 | 1.695 | 0.464 | 1.017 | 1.266 | 0.012 | -0.036 | -0.121 |
| TR-noDynamicPrior | 0.634 | 1.327 | 1.702 | 0.464 | 1.011 | 1.268 | 0.027 | -0.017 | -0.109 |
| TR-noMoE | 0.643 | 1.345 | 1.698 | 0.475 | 1.021 | 1.267 | -0.001 | -0.025 | -0.130 |
| TR-noRouter | 0.642 | 1.342 | 1.694 | 0.470 | 1.026 | 1.265 | -0.011 | -0.041 | -0.119 |
| TR-noTCN | 0.667 | 1.357 | 1.733 | 0.496 | 1.040 | 1.310 | -0.010 | -0.049 | -0.161 |
| TR-unbounded | 0.634 | 1.333 | 1.695 | 0.466 | 0.999 | 1.271 | 0.003 | -0.047 | -0.125 |


## One-factor ablation accuracy and skill (relocated Tables 4.8a-b)

<!-- TABLE 4.8b (generated) -->

| Model | persist. 1 | persist. 3 | persist. 7 | damped 1 | damped 3 | damped 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TR-fixedKappa | +0.209 | +0.187 | +0.250 | +0.180 | +0.075 | +0.041 |
| TR-noDynamicPrior | +0.212 | +0.189 | +0.248 | +0.181 | +0.077 | +0.038 |
| TR-noMoE | +0.199 | +0.182 | +0.250 | +0.169 | +0.074 | +0.036 |
| TR-noRouter | +0.202 | +0.180 | +0.250 | +0.172 | +0.072 | +0.038 |
| TR-noTCN | +0.176 | +0.167 | +0.240 | +0.141 | +0.058 | +0.030 |
| TR-unbounded | +0.204 | +0.189 | +0.252 | +0.176 | +0.076 | +0.042 |


## One-factor ablation accuracy and skill (relocated Tables 4.8a-b)

<!-- TABLE 4.8a (generated) -->
| Model | RMSE 1 | RMSE 3 | RMSE 7 | MAE 1 | MAE 3 | MAE 7 | bias 1 | bias 3 | bias 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TR-fixedKappa | 0.635 | 1.333 | 1.695 | 0.464 | 1.017 | 1.266 | 0.012 | -0.036 | -0.121 |
| TR-noDynamicPrior | 0.634 | 1.327 | 1.702 | 0.464 | 1.011 | 1.268 | 0.027 | -0.017 | -0.109 |
| TR-noMoE | 0.643 | 1.345 | 1.698 | 0.475 | 1.021 | 1.267 | -0.001 | -0.025 | -0.130 |
| TR-noRouter | 0.642 | 1.342 | 1.694 | 0.470 | 1.026 | 1.265 | -0.011 | -0.041 | -0.119 |
| TR-noTCN | 0.667 | 1.357 | 1.733 | 0.496 | 1.040 | 1.310 | -0.010 | -0.049 | -0.161 |
| TR-unbounded | 0.634 | 1.333 | 1.695 | 0.466 | 0.999 | 1.271 | 0.003 | -0.047 | -0.125 |


## One-factor ablation accuracy and skill (relocated Tables 4.8a-b)

<!-- TABLE 4.8b (generated) -->
| Model | persist. 1 | persist. 3 | persist. 7 | damped 1 | damped 3 | damped 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TR-fixedKappa | +0.209 | +0.187 | +0.250 | +0.180 | +0.075 | +0.041 |
| TR-noDynamicPrior | +0.212 | +0.189 | +0.248 | +0.181 | +0.077 | +0.038 |
| TR-noMoE | +0.199 | +0.182 | +0.250 | +0.169 | +0.074 | +0.036 |
| TR-noRouter | +0.202 | +0.180 | +0.250 | +0.172 | +0.072 | +0.038 |
| TR-noTCN | +0.176 | +0.167 | +0.240 | +0.141 | +0.058 | +0.030 |
| TR-unbounded | +0.204 | +0.189 | +0.252 | +0.176 | +0.076 | +0.042 |


## One-factor ablation accuracy and skill (relocated Tables 4.8a-b)

<!-- TABLE 4.8a (generated) -->
| Model | RMSE 1 | RMSE 3 | RMSE 7 | MAE 1 | MAE 3 | MAE 7 | bias 1 | bias 3 | bias 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TR-fixedKappa | 0.635 | 1.333 | 1.695 | 0.464 | 1.017 | 1.266 | 0.012 | -0.036 | -0.121 |
| TR-noDynamicPrior | 0.634 | 1.327 | 1.702 | 0.464 | 1.011 | 1.268 | 0.027 | -0.017 | -0.109 |
| TR-noMoE | 0.643 | 1.345 | 1.698 | 0.475 | 1.021 | 1.267 | -0.001 | -0.025 | -0.130 |
| TR-noRouter | 0.642 | 1.342 | 1.694 | 0.470 | 1.026 | 1.265 | -0.011 | -0.041 | -0.119 |
| TR-noTCN | 0.667 | 1.357 | 1.733 | 0.496 | 1.040 | 1.310 | -0.010 | -0.049 | -0.161 |
| TR-unbounded | 0.634 | 1.333 | 1.695 | 0.466 | 0.999 | 1.271 | 0.003 | -0.047 | -0.125 |


## One-factor ablation accuracy and skill (relocated Tables 4.8a-b)

<!-- TABLE 4.8b (generated) -->
| Model | persist. 1 | persist. 3 | persist. 7 | damped 1 | damped 3 | damped 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TR-fixedKappa | +0.209 | +0.187 | +0.250 | +0.180 | +0.075 | +0.041 |
| TR-noDynamicPrior | +0.212 | +0.189 | +0.248 | +0.181 | +0.077 | +0.038 |
| TR-noMoE | +0.199 | +0.182 | +0.250 | +0.169 | +0.074 | +0.036 |
| TR-noRouter | +0.202 | +0.180 | +0.250 | +0.172 | +0.072 | +0.038 |
| TR-noTCN | +0.176 | +0.167 | +0.240 | +0.141 | +0.058 | +0.030 |
| TR-unbounded | +0.204 | +0.189 | +0.252 | +0.176 | +0.076 | +0.042 |

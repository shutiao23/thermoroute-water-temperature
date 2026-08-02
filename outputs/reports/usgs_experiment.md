# USGS large-sample experiment (120 stations, 5 seeds)

_Variables WTEMP, FLOW, TEMP, PRCP, RHMEAN, DH, WDSP. Observed targets only; identical samples across the five primary headline models. ThermoRoute = 5-seed mean. The same-station LightGBM is also a five-seed mean and receives stable site identity as a categorical feature; its small predeclared grid is selected by 2016–2017 station-macro RMSE only._

Air2stream-style a4/a8 (unofficial, non-primary): NOT_RUN; headline entry is NOT_RUN / NA.

| horizon | persist | damped | Air2stream-style a4/a8 (unofficial, non-primary) | LightGBM | ThermoRoute | skill vs persist | skill vs damped | win-rate vs damped |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.803 | 0.774 | NOT_RUN / NA | 0.578 | 0.629 | +0.204 | +0.172 | 0.87 |
| 3 | 1.576 | 1.406 | NOT_RUN / NA | 1.280 | 1.287 | +0.189 | +0.077 | 0.91 |
| 7 | 2.217 | 1.738 | NOT_RUN / NA | 1.649 | 1.659 | +0.249 | +0.038 | 0.94 |

## Random held-station warm-start diagnostic (90→30)

Held stations contribute historical observations to the global panel preprocessing; this arm is not zero-shot spatial transfer and does not establish unseen-basin skill.

| horizon | warm-start RMSE | persistence RMSE | warm-start skill |
|---|---|---|---|
| 1 | 0.660 | 0.784 | +0.158 |
| 3 | 1.274 | 1.525 | +0.165 |
| 7 | 1.548 | 2.025 | +0.236 |

## Module ablations (five-seed deletion/intervention sensitivity; ensemble-mean median per-station RMSE, delta_scale=1.0)

Audit: every mandatory control contains seeds 0--4 and uses the same five seeds as ThermoRoute, with identical forecast keys and exact y_true within each paired seed. Interpretation: this is a five-seed deletion/intervention sensitivity, not evidence of module necessity, causal mechanism, or capacity-matched attribution.

| variant | h1 | h3 | h7 |
|---|---|---|---|
| ThermoRoute | 0.629 | 1.287 | 1.659 |
| DampedPriorOnly | 0.774 | 1.406 | 1.738 |
| TR-noDynamicPrior | 0.629 | 1.287 | 1.669 |
| TR-fixedKappa | 0.627 | 1.290 | 1.656 |
| TR-noRouter | 0.634 | 1.298 | 1.670 |
| TR-noMoE | 0.634 | 1.290 | 1.658 |
| TR-noTCN | 0.679 | 1.340 | 1.674 |
| TR-unbounded | 0.630 | 1.283 | 1.651 |

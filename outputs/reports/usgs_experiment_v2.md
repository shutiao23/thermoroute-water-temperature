# USGS large-sample experiment (120 stations, 5 seeds)

_Variables WTEMP, FLOW, TEMP, PRCP, RHMEAN, DH, WDSP. Observed targets only; identical samples across models. ThermoRoute = 5-seed mean._

| horizon | persist | damped | air2stream-a8 | LightGBM | ThermoRoute | skill vs persist | skill vs damped | win-rate vs damped |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.797 | 0.774 | 0.797 | 0.620 | 0.630 | +0.204 | +0.172 | 0.88 |
| 3 | 1.581 | 1.406 | 1.464 | 1.300 | 1.289 | +0.186 | +0.078 | 0.94 |
| 7 | 2.235 | 1.778 | 1.809 | 1.669 | 1.658 | +0.247 | +0.035 | 0.92 |

## Leave-group-out transfer (90→30 unseen basins)

| horizon | TR transfer RMSE | persistence RMSE | transfer skill |
|---|---|---|---|
| 1 | 0.647 | 0.780 | +0.170 |
| 3 | 1.295 | 1.528 | +0.153 |
| 7 | 1.589 | 2.058 | +0.228 |

## Module ablations (median per-station RMSE, delta_scale=1.0)

| variant | h1 | h3 | h7 |
|---|---|---|---|
| ThermoRoute | 0.630 | 1.289 | 1.658 |
| TR-noPrior | 1.337 | 1.533 | 1.717 |
| TR-fixedKappa | 0.651 | 1.307 | 1.669 |
| TR-noRouter | 0.648 | 1.308 | 1.663 |
| TR-noMoE | 0.740 | 1.371 | 1.720 |
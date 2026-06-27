# USGS large-sample experiment (40 stations, 5 seeds)

_Variables WTEMP, FLOW, TEMP, PRCP, RHMEAN, DH, WDSP. Observed targets only; identical samples across models. ThermoRoute = 5-seed mean._

| horizon | persist | damped | LightGBM | ThermoRoute | skill vs persist | skill vs damped | win-rate vs damped |
|---|---|---|---|---|---|---|---|
| 1 | 0.671 | 0.645 | 0.560 | 0.554 | +0.163 | +0.127 | 0.72 |
| 3 | 1.420 | 1.258 | 1.153 | 1.175 | +0.180 | +0.057 | 0.82 |
| 7 | 1.952 | 1.525 | 1.458 | 1.490 | +0.253 | +0.034 | 0.78 |

## Leave-group-out transfer (30→10 unseen basins)

| horizon | TR transfer RMSE | persistence RMSE | transfer skill |
|---|---|---|---|
| 1 | 0.655 | 0.782 | +0.162 |
| 3 | 1.283 | 1.533 | +0.163 |
| 7 | 1.553 | 2.096 | +0.259 |

## Module ablations (median per-station RMSE, delta_scale=1.5)

| variant | h1 | h3 | h7 |
|---|---|---|---|
| ThermoRoute | 0.554 | 1.175 | 1.490 |
| TR-noPrior | 1.045 | 1.314 | 1.551 |
| TR-fixedKappa | 0.574 | 1.185 | 1.481 |
| TR-noRouter | 0.570 | 1.196 | 1.502 |
| TR-noMoE | 0.632 | 1.233 | 1.536 |
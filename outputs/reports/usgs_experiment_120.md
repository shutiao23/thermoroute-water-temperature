# USGS large-sample experiment (120 stations, 5 seeds)

_Variables WTEMP, FLOW, TEMP, PRCP, RHMEAN, DH, WDSP. Observed targets only; identical samples across models. ThermoRoute = 5-seed mean._

| horizon | persist | damped | air2stream-a8 | LightGBM | ThermoRoute | skill vs persist | skill vs damped | win-rate vs damped |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.797 | 0.774 | 0.811 | 0.616 | 0.628 | +0.207 | +0.175 | 0.84 |
| 3 | 1.581 | 1.406 | 1.370 | 1.290 | 1.281 | +0.189 | +0.081 | 0.88 |
| 7 | 2.235 | 1.778 | 1.695 | 1.665 | 1.652 | +0.251 | +0.039 | 0.88 |

## Leave-group-out transfer (90→30 unseen basins)

| horizon | TR transfer RMSE | persistence RMSE | transfer skill |
|---|---|---|---|
| 1 | 0.656 | 0.780 | +0.159 |
| 3 | 1.285 | 1.528 | +0.159 |
| 7 | 1.565 | 2.058 | +0.239 |

## Module ablations (median per-station RMSE, delta_scale=1.5)

| variant | h1 | h3 | h7 |
|---|---|---|---|
| ThermoRoute | 0.628 | 1.281 | 1.652 |
| TR-noPrior | 1.177 | 1.440 | 1.684 |
| TR-fixedKappa | 0.648 | 1.305 | 1.655 |
| TR-noRouter | 0.640 | 1.294 | 1.667 |
| TR-noMoE | 0.745 | 1.359 | 1.692 |
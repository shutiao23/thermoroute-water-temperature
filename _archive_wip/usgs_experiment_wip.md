# USGS large-sample experiment (40 stations, 2 seeds)

_Variables WTEMP, FLOW, TEMP, PRCP, RHMEAN, DH, WDSP. Observed targets only; identical samples across models. ThermoRoute = 2-seed mean._

| horizon | persist | damped | air2stream-a8 | LightGBM | ThermoRoute | skill vs persist | skill vs damped | win-rate vs damped |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.704 | 0.678 | 0.730 | 0.549 | nan | +nan | +nan | 0.00 |
| 3 | 1.440 | 1.300 | 1.303 | 1.189 | nan | +nan | +nan | 0.00 |
| 7 | 2.233 | 1.780 | 1.647 | 1.687 | nan | +nan | +nan | 0.00 |

## Leave-group-out transfer (30→10 unseen basins)

| horizon | TR transfer RMSE | persistence RMSE | transfer skill |
|---|---|---|---|
| 1 | nan | 0.908 | +nan |
| 3 | nan | 1.707 | +nan |
| 7 | nan | 2.224 | +nan |

## Module ablations (median per-station RMSE, delta_scale=1.5)

| variant | h1 | h3 | h7 |
|---|---|---|---|
| ThermoRoute | nan | nan | nan |
| TR-noPrior | nan | nan | nan |
| TR-fixedKappa | nan | nan | nan |
| TR-noRouter | nan | nan | nan |
| TR-noMoE | nan | nan | nan |
# Deep sequence baseline (global LSTM) — in-sample + region-transfer

A station-agnostic top-down LSTM (1 layer, hidden 64, 14-day context, persistence-anchored) trained under the SAME Track-H splits and composite loss as ThermoRoute — the field-standard deep baseline (Rahmani 2021 / Willard 2024), landing in the published LSTM accuracy band (Zwart 2023). The intended reading is the FOIL: it is competitive on RMSE (behind both the global LightGBM and ThermoRoute here, consistent with GBDT≥LSTM on autocorrelated daily data) yet carries no bounded-degradation floor, no distribution-free calibrated intervals, and no interpretable lag router.

## Leave-HUC2-region-out transfer (whole regions held out)

| horizon | n | TR RMSE | LGB RMSE | LSTM RMSE | TR−LSTM p | LGB−LSTM p | best |
|---|---|---|---|---|---|---|---|
| 1 | 114 | 0.667 | 0.650 | 0.671 | 2.4e-05 | 1e-11 | **LGB** |
| 3 | 114 | 1.323 | 1.321 | 1.393 | 1e-18 | 4.6e-17 | **LGB** |
| 7 | 114 | 1.675 | 1.683 | 1.825 | 6e-20 | 1.6e-19 | **TR** |

## In-sample (120 stations, 5-seed ensembles) median per-station RMSE

| horizon | persist | LightGBM | LSTM | ThermoRoute | TR−LSTM paired p | TR wins |
|---|---|---|---|---|---|---|
| 1 | 0.797 | 0.620 | 0.653 | 0.630 | 1.1e-15 | 88% |
| 3 | 1.581 | 1.300 | 1.362 | 1.289 | 4.8e-17 | 93% |
| 7 | 2.235 | 1.669 | 1.803 | 1.658 | 2.2e-17 | 91% |
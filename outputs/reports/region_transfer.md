# Leave-HUC2-region-out transfer — ThermoRoute vs the strong learned baseline

16 HUC2 regions packed into 4 folds ([30, 30, 30, 30] stations); each fold holds out **whole regions** so no held-out-region gage is in training (fixes the random-split spatial leak). Station-agnostic ThermoRoute and a global LightGBM are each trained on the in-fold regions and forecast the held-out region stations. Persistence/damped are training-free (read from v2).

| horizon | n | TR RMSE | LGB RMSE | persist | damped | TR skill/persist | LGB skill/persist | TR−LGB paired Wilcoxon p | winner |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 114 | 0.667 | 0.650 | 0.797 | 0.774 | +0.151 | +0.175 | 5e-09 | **LGB** |
| 3 | 114 | 1.323 | 1.321 | 1.581 | 1.406 | +0.169 | +0.168 | 0.95 | **tie** |
| 7 | 114 | 1.675 | 1.683 | 2.235 | 1.778 | +0.237 | +0.232 | 0.0014 | **TR** |

## Verdict

**TIE** — ThermoRoute and LightGBM transfer comparably out-of-region. Report parity honestly; the contribution is calibration + physics-vs-GBDT delineation, not a transfer-superiority claim (二区-strong framing).

Per-horizon winners: {1: 'LGB', 3: 'tie', 7: 'TR'}
Mean nearest-training-gage distance for held-out stations: 358 km (vs a random split where it would be near 0).
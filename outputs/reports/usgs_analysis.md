# USGS large-sample: calibration, decision value, mechanism

## Probabilistic & event metrics (conformal, test)

| model | h | PICP | MPIW | CRPS | Brier-skill | AUPRC |
|---|---|---|---|---|---|---|
| ThermoRoute | 1 | 0.904 | 2.02 | 0.246 | +0.742 | 0.923 |
| ThermoRoute | 3 | 0.909 | 4.26 | 0.505 | +0.596 | 0.817 |
| ThermoRoute | 7 | 0.911 | 5.45 | 0.633 | +0.508 | 0.744 |
| LightGBM | 1 | 0.906 | 2.01 | 0.240 | +0.733 | 0.918 |
| LightGBM | 3 | 0.906 | 4.24 | 0.503 | +0.573 | 0.794 |
| LightGBM | 7 | 0.907 | 5.35 | 0.630 | +0.490 | 0.716 |

## Decision value (peak REV)

| model | h | REV_max | REV@0.1 | REV@0.2 |
|---|---|---|---|---|
| DampedPersistence | 1 | 0.831 | 0.799 | 0.818 |
| LightGBM | 1 | 0.898 | 0.894 | 0.850 |
| Persistence | 1 | 0.843 | 0.814 | 0.828 |
| ThermoRoute | 1 | 0.907 | 0.902 | 0.859 |
| DampedPersistence | 3 | 0.679 | 0.616 | 0.656 |
| LightGBM | 3 | 0.844 | 0.841 | 0.760 |
| Persistence | 3 | 0.706 | 0.651 | 0.677 |
| ThermoRoute | 3 | 0.855 | 0.847 | 0.775 |
| DampedPersistence | 7 | 0.578 | 0.494 | 0.553 |
| LightGBM | 7 | 0.813 | 0.807 | 0.700 |
| Persistence | 7 | 0.589 | 0.511 | 0.547 |
| ThermoRoute | 7 | 0.823 | 0.814 | 0.718 |

## Dynamic thermal memory — κ flow-dependence

- κ_high/κ_low > 1 (faster relaxation at high flow) at **0% of stations** (median ratio 0.87).
- mean κ_low=0.134, κ_high=0.117.

## Router top drivers by horizon

- h=1d: FLOW (44%), PRCP (29%), RHMEAN (8%)
- h=3d: DH (30%), FLOW (29%), RHMEAN (15%)
- h=7d: DH (35%), FLOW (27%), RHMEAN (17%)
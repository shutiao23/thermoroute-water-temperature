# USGS large-sample: calibration, decision value, mechanism

## Probabilistic & event metrics (conformal, test)

| model | h | PICP | MPIW | CRPS | Brier-skill | AUPRC |
|---|---|---|---|---|---|---|
| ThermoRoute | 1 | 0.904 | 2.02 | 0.245 | +0.745 | 0.923 |
| ThermoRoute | 3 | 0.911 | 4.27 | 0.504 | +0.604 | 0.815 |
| ThermoRoute | 7 | 0.912 | 5.45 | 0.631 | +0.514 | 0.742 |
| LightGBM | 1 | 0.905 | 1.99 | 0.238 | +0.732 | 0.918 |
| LightGBM | 3 | 0.906 | 4.22 | 0.499 | +0.574 | 0.795 |
| LightGBM | 7 | 0.907 | 5.31 | 0.625 | +0.498 | 0.723 |

## Decision value (peak REV)

| model | h | REV_max | REV@0.1 | REV@0.2 |
|---|---|---|---|---|
| DampedPersistence | 1 | 0.831 | 0.799 | 0.818 |
| LightGBM | 1 | 0.899 | 0.895 | 0.847 |
| Persistence | 1 | 0.843 | 0.814 | 0.828 |
| ThermoRoute | 1 | 0.903 | 0.896 | 0.857 |
| DampedPersistence | 3 | 0.679 | 0.616 | 0.656 |
| LightGBM | 3 | 0.845 | 0.842 | 0.758 |
| Persistence | 3 | 0.706 | 0.651 | 0.677 |
| ThermoRoute | 3 | 0.855 | 0.848 | 0.783 |
| DampedPersistence | 7 | 0.578 | 0.494 | 0.553 |
| LightGBM | 7 | 0.817 | 0.810 | 0.708 |
| Persistence | 7 | 0.589 | 0.511 | 0.547 |
| ThermoRoute | 7 | 0.824 | 0.814 | 0.725 |

## Dynamic thermal memory — κ flow-dependence

- κ_high/κ_low > 1 (faster relaxation at high flow) at **2% of stations** (median ratio 0.87).
- mean κ_low=0.118, κ_high=0.105.

## Router top drivers by horizon

- h=1d: DH (56%), WTEMP (21%), WDSP (9%)
- h=3d: DH (54%), WTEMP (19%), TEMP (11%)
- h=7d: DH (54%), TEMP (21%), FLOW (11%)
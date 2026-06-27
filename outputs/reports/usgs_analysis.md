# USGS large-sample: calibration, decision value, mechanism

## Probabilistic & event metrics (conformal, test)

| model | h | PICP | MPIW | CRPS | Brier-skill | AUPRC |
|---|---|---|---|---|---|---|
| ThermoRoute | 1 | 0.904 | 1.86 | 0.230 | +0.296 | 0.570 |
| ThermoRoute | 3 | 0.906 | 3.76 | 0.454 | +0.254 | 0.514 |
| ThermoRoute | 7 | 0.909 | 4.84 | 0.570 | +0.242 | 0.488 |
| LightGBM | 1 | 0.900 | 1.90 | 0.230 | +0.334 | 0.623 |
| LightGBM | 3 | 0.889 | 3.77 | 0.464 | +0.298 | 0.570 |
| LightGBM | 7 | 0.885 | 4.72 | 0.581 | +0.284 | 0.548 |

## Decision value (peak REV)

| model | h | REV_max | REV@0.1 | REV@0.2 |
|---|---|---|---|---|
| DampedPersistence | 1 | 0.891 | 0.859 | 0.884 |
| LightGBM | 1 | 0.615 | 0.533 | 0.560 |
| Persistence | 1 | 0.893 | 0.861 | 0.885 |
| ThermoRoute | 1 | 0.619 | 0.529 | 0.556 |
| DampedPersistence | 3 | 0.800 | 0.738 | 0.788 |
| LightGBM | 3 | 0.616 | 0.540 | 0.529 |
| Persistence | 3 | 0.798 | 0.737 | 0.782 |
| ThermoRoute | 3 | 0.611 | 0.529 | 0.526 |
| DampedPersistence | 7 | 0.740 | 0.656 | 0.728 |
| LightGBM | 7 | 0.619 | 0.554 | 0.521 |
| Persistence | 7 | 0.727 | 0.645 | 0.705 |
| ThermoRoute | 7 | 0.598 | 0.528 | 0.502 |

## Dynamic thermal memory — κ flow-dependence

- κ_high/κ_low > 1 (faster relaxation at high flow) at **18% of stations** (median ratio 0.92).
- mean κ_low=0.102, κ_high=0.095.

## Router top drivers by horizon

- h=1d: WDSP (nan%), DH (nan%), RHMEAN (nan%)
- h=3d: WDSP (nan%), DH (nan%), RHMEAN (nan%)
- h=7d: WDSP (nan%), DH (nan%), RHMEAN (nan%)
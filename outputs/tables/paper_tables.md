# ThermoRoute — result tables

_All metrics on the 2019–2020 blind test. Deep models: mean ± std over seeds. Stations b1/s2/p3 averaged unless noted._

## Table 2 — Overall blind-test accuracy by horizon

RMSE (°C) and skill vs persistence, mean over the three stations.

| model | RMSE h1 | RMSE h3 | RMSE h7 | skill h1 | skill h3 | skill h7 |
|---|---|---|---|---|---|---|
| Persistence | 0.270 | 0.536 | 0.918 | +nan | +nan | +nan |
| Climatology | 1.279 | 1.279 | 1.280 | -5.034 | -2.001 | -0.730 |
| DampedPersistence | 0.261 | 0.483 | 0.724 | +0.027 | +0.081 | +0.177 |
| Air2streamLite | 0.265 | 0.498 | 0.760 | +0.013 | +0.056 | +0.144 |
| Ridge | 0.276 | 0.776 | 1.349 | -0.041 | -0.369 | -0.384 |
| LightGBM | 0.255 | 0.496 | 0.806 | +0.015 | +0.049 | +0.085 |
| GRU | 1.567 | 1.557 | 1.404 | -4.711 | -1.849 | -0.557 |
| ThermoRoute | 0.292 | 0.543 | 0.798 | -0.112 | -0.037 | +0.083 |

## Table 2b — ThermoRoute vs damped persistence, per station (+ significance)

ΔRMSE = RMSE(ThermoRoute) − RMSE(damped). Negative ⇒ ThermoRoute better. DM p<0.05 marked *.

| station | horizon | RMSE damped | RMSE ThermoRoute | ΔRMSE | DM p |
|---|---|---|---|---|---|
| b1 | 1 | 0.329 | 0.365 | +0.037 | 0.000* |
| b1 | 3 | 0.616 | 0.687 | +0.071 | 0.016* |
| b1 | 7 | 0.888 | 0.969 | +0.081 | 0.126 |
| s2 | 1 | 0.294 | 0.280 | -0.014 | 0.008* |
| s2 | 3 | 0.507 | 0.471 | -0.036 | 0.021* |
| s2 | 7 | 0.719 | 0.686 | -0.033 | 0.256 |
| p3 | 1 | 0.161 | 0.199 | +0.038 | 0.000* |
| p3 | 3 | 0.327 | 0.386 | +0.059 | 0.000* |
| p3 | 7 | 0.565 | 0.666 | +0.101 | 0.002* |

## Table 3 — Variable-set gains (RMSE °C)

Adding hydrology + meteorology to the autoregressive baseline.

| model | set | h1 | h3 | h7 |
|---|---|---|---|---|
| LightGBM | V1 | 0.268 | 0.514 | 0.780 |
| LightGBM | V2 | 0.255 | 0.497 | 0.803 |
| LightGBM | V3 | 0.255 | 0.496 | 0.806 |
| ThermoRoute | V1 | 0.275 | 0.505 | 0.759 |
| ThermoRoute | V2 | 0.293 | 0.524 | 0.762 |
| ThermoRoute | V3 | 0.292 | 0.543 | 0.798 |

## Table 4 — Probabilistic & high-temperature warning (blind test)

ThermoRoute (joint, conformal). PICP target = 0.90.

| station | horizon | PICP | MPIW (°C) | CRPS | Brier-skill | AUPRC |
|---|---|---|---|---|---|---|
| b1 | 1 | 0.791 | 0.99 | 0.161 | -0.902 | 0.607 |
| b1 | 3 | 0.860 | 2.52 | 0.320 | -1.587 | 0.221 |
| b1 | 7 | 0.868 | 3.41 | 0.430 | -2.045 | 0.091 |
| s2 | 1 | 0.871 | 0.86 | 0.113 | +0.629 | 0.848 |
| s2 | 3 | 0.896 | 1.52 | 0.190 | +0.392 | 0.698 |
| s2 | 7 | 0.912 | 2.32 | 0.275 | +0.207 | 0.420 |
| p3 | 1 | 0.863 | 0.62 | 0.083 | +0.338 | 0.587 |
| p3 | 3 | 0.882 | 1.22 | 0.160 | +0.118 | 0.422 |
| p3 | 7 | 0.826 | 1.84 | 0.276 | -0.021 | 0.258 |

## Table 5 — Leave-one-station-out spatial transfer (RMSE °C)

| held-out station | h1 joint | h1 LOSO | h3 joint | h3 LOSO | h7 joint | h7 LOSO |
|---|---|---|---|---|---|---|
| b1 | 0.388 | 0.431 | 0.757 | 0.752 | 1.029 | 0.979 |
| s2 | 0.283 | 0.289 | 0.477 | 0.478 | 0.692 | 0.675 |
| p3 | 0.206 | 0.281 | 0.395 | 0.553 | 0.674 | 0.866 |

## Table 6 — Module ablations (RMSE °C, V3 joint) + RMSE 95% CI

Block-bootstrap 95% CI for the full model (station-pooled).

| variant | h1 | h3 | h7 |
|---|---|---|---|
| ThermoRoute | 0.292 | 0.543 | 0.798 |
| TR-noPrior | 1.145 | 1.180 | 1.226 |
| TR-fixedKappa | 0.287 | 0.525 | 0.797 |
| TR-softmax | 0.305 | 0.540 | 0.780 |
| TR-noMoE | 0.260 | 0.491 | 0.757 |
| TR-noRouter | 0.345 | 0.631 | 0.875 |

**ThermoRoute RMSE 95% CI (moving-block bootstrap, station-averaged):**

- h=1: 0.282 [0.266, 0.297] °C
- h=3: 0.515 [0.469, 0.565] °C
- h=7: 0.774 [0.697, 0.853] °C

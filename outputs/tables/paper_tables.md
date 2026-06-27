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
| GRU | 1.542 | 1.537 | 1.475 | -4.652 | -1.832 | -0.625 |
| ThermoRoute | 0.343 | 0.557 | 0.808 | -0.273 | -0.071 | +0.077 |

## Table 2b — ThermoRoute vs damped persistence, per station (+ significance)

ΔRMSE = RMSE(ThermoRoute) − RMSE(damped). Negative ⇒ ThermoRoute better. DM p<0.05 marked *.

| station | horizon | RMSE damped | RMSE ThermoRoute | ΔRMSE | DM p |
|---|---|---|---|---|---|
| b1 | 1 | 0.329 | 0.486 | +0.157 | 0.000* |
| b1 | 3 | 0.616 | 0.756 | +0.139 | 0.000* |
| b1 | 7 | 0.888 | 1.002 | +0.114 | 0.085 |
| s2 | 1 | 0.294 | 0.279 | -0.015 | 0.002* |
| s2 | 3 | 0.507 | 0.465 | -0.042 | 0.016* |
| s2 | 7 | 0.719 | 0.668 | -0.051 | 0.041* |
| p3 | 1 | 0.161 | 0.207 | +0.046 | 0.000* |
| p3 | 3 | 0.327 | 0.407 | +0.079 | 0.000* |
| p3 | 7 | 0.565 | 0.653 | +0.088 | 0.016* |

## Table 3 — Variable-set gains (RMSE °C)

Adding hydrology + meteorology to the autoregressive baseline.

| model | set | h1 | h3 | h7 |
|---|---|---|---|---|
| LightGBM | V1 | 0.268 | 0.514 | 0.780 |
| LightGBM | V2 | 0.255 | 0.497 | 0.803 |
| LightGBM | V3 | 0.255 | 0.496 | 0.806 |
| ThermoRoute | V1 | 0.283 | 0.524 | 0.772 |
| ThermoRoute | V2 | 0.295 | 0.552 | 0.796 |
| ThermoRoute | V3 | 0.343 | 0.557 | 0.808 |

## Table 4 — Probabilistic & high-temperature warning (blind test)

ThermoRoute (joint, conformal). PICP target = 0.90.

| station | horizon | PICP | MPIW (°C) | CRPS | Brier-skill | AUPRC |
|---|---|---|---|---|---|---|
| b1 | 1 | 0.651 | 1.15 | 0.274 | +0.389 | 0.668 |
| b1 | 3 | 0.847 | 2.45 | 0.316 | +0.186 | 0.433 |
| b1 | 7 | 0.871 | 3.51 | 0.440 | +0.084 | 0.363 |
| s2 | 1 | 0.868 | 0.84 | 0.113 | +0.564 | 0.803 |
| s2 | 3 | 0.876 | 1.41 | 0.186 | +0.381 | 0.660 |
| s2 | 7 | 0.910 | 2.22 | 0.270 | +0.234 | 0.432 |
| p3 | 1 | 0.863 | 0.66 | 0.087 | +0.318 | 0.503 |
| p3 | 3 | 0.870 | 1.25 | 0.168 | +0.122 | 0.338 |
| p3 | 7 | 0.849 | 1.94 | 0.274 | +0.076 | 0.254 |

## Table 5 — Leave-one-station-out spatial transfer (RMSE °C)

| held-out station | h1 joint | h1 LOSO | h3 joint | h3 LOSO | h7 joint | h7 LOSO |
|---|---|---|---|---|---|---|
| b1 | 0.527 | 0.460 | 0.779 | 0.727 | 1.080 | 1.057 |
| s2 | 0.286 | 0.291 | 0.471 | 0.477 | 0.677 | 0.689 |
| p3 | 0.217 | 0.367 | 0.421 | 0.752 | 0.669 | 1.121 |

## Table 6 — Module ablations (RMSE °C, V3 joint) + RMSE 95% CI

Block-bootstrap 95% CI for the full model (station-pooled).

| variant | h1 | h3 | h7 |
|---|---|---|---|
| ThermoRoute | 0.343 | 0.557 | 0.808 |
| TR-noPrior | 1.091 | 1.181 | 1.305 |
| TR-fixedKappa | 0.287 | 0.532 | 0.788 |
| TR-softmax | 0.331 | 0.612 | 0.901 |
| TR-noMoE | 0.260 | 0.483 | 0.723 |
| TR-noRouter | 0.311 | 0.574 | 0.786 |

**ThermoRoute RMSE 95% CI (moving-block bootstrap, station-averaged):**

- h=1: 0.324 [0.306, 0.337] °C
- h=3: 0.542 [0.493, 0.593] °C
- h=7: 0.775 [0.694, 0.857] °C

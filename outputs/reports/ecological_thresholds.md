# Exceedance warnings at fixed ecological thresholds (EPA 7DADM salmonid criteria)

Calibrated exceedance probability at an absolute threshold T, read from the conformalised predictive distribution (no retraining), scored on the free-flowing stations where T is ecologically live (test base rate 0.05–0.60). Brier skill is vs the climatological base rate; higher is better. This complements the statistical train-q90 warning with a regulator-meaningful cut-off.


## 18 °C (salmonid rearing / 7DADM)

| horizon | n stn | base rate | model | Brier skill | AUROC |
|---|---|---|---|---|---|
| 1 | 71 | 0.31 | ThermoRoute | +0.927 | 0.998 |
| 1 | 71 | 0.31 | LightGBM | +0.928 | 0.998 |
| 1 | 71 | 0.31 | LSTM | +0.925 | 0.998 |
| 1 | 71 | 0.31 | Persistence (determ.) | +0.867 | 0.967 |
| 3 | 71 | 0.31 | ThermoRoute | +0.846 | 0.992 |
| 3 | 71 | 0.31 | LightGBM | +0.844 | 0.992 |
| 3 | 71 | 0.31 | LSTM | +0.840 | 0.991 |
| 3 | 71 | 0.31 | Persistence (determ.) | +0.731 | 0.933 |
| 7 | 71 | 0.31 | ThermoRoute | +0.806 | 0.987 |
| 7 | 71 | 0.31 | LightGBM | +0.804 | 0.987 |
| 7 | 71 | 0.31 | LSTM | +0.792 | 0.986 |
| 7 | 71 | 0.31 | Persistence (determ.) | +0.621 | 0.905 |

## 20 °C (migration-corridor max)

| horizon | n stn | base rate | model | Brier skill | AUROC |
|---|---|---|---|---|---|
| 1 | 62 | 0.29 | ThermoRoute | +0.929 | 0.998 |
| 1 | 62 | 0.29 | LightGBM | +0.930 | 0.998 |
| 1 | 62 | 0.29 | LSTM | +0.927 | 0.998 |
| 1 | 62 | 0.29 | Persistence (determ.) | +0.871 | 0.968 |
| 3 | 62 | 0.29 | ThermoRoute | +0.847 | 0.992 |
| 3 | 62 | 0.29 | LightGBM | +0.846 | 0.992 |
| 3 | 62 | 0.29 | LSTM | +0.839 | 0.992 |
| 3 | 62 | 0.29 | Persistence (determ.) | +0.738 | 0.935 |
| 7 | 62 | 0.29 | ThermoRoute | +0.807 | 0.988 |
| 7 | 62 | 0.29 | LightGBM | +0.806 | 0.988 |
| 7 | 62 | 0.29 | LSTM | +0.791 | 0.986 |
| 7 | 62 | 0.29 | Persistence (determ.) | +0.633 | 0.908 |

The calibrated probabilistic warnings retain clear positive Brier skill at the regulatory thresholds, and beat the deterministic persistence warning — so the exceedance contribution does not depend on the arbitrary 90th-percentile cut-off.
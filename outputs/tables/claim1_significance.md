# Claim 1 — accuracy vs baselines, per-station significance (n = 114 blind-test stations)

Two disclosed protocols. **per-seed**: each of the 5 ThermoRoute seeds is scored as a single model (same budget as each baseline) and per-station RMSE is averaged across seeds; ± is the across-seed std of the median skill. **ensemble**: the deployed 5-member seed-averaged forecaster vs single-model baselines — an ensemble-vs-single comparison, labelled as such. Skill = 1−RMSE/RMSE_ref. Win-rate over the n stations with a Wilson 95% CI. Wilcoxon = two-sided paired signed-rank; Holm-adjusted across the 9 (horizon × reference) tests per protocol.

| protocol | horizon | reference | n | median skill [boot 95% CI] | win-rate [Wilson 95% CI] | p (raw) | p (Holm) |
|---|---|---|---|---|---|---|---|
| per-seed | 1 | Persistence | 114 | +0.186 [+0.169, +0.208] (±0.005) | 0.90 (103/114) [0.84, 0.95] | 1.1e-18 | 7.4e-18* |
| per-seed | 1 | DampedPersistence | 114 | +0.157 [+0.134, +0.172] (±0.005) | 0.85 (97/114) [0.77, 0.90] | 6.2e-17 | 3.1e-16* |
| per-seed | 1 | LightGBM | 114 | -0.044 [-0.050, -0.034] (±0.007) | 0.10 (11/114) [0.05, 0.16] | 4.4e-15 | 1.3e-14* |
| per-seed | 3 | Persistence | 114 | +0.180 [+0.170, +0.187] (±0.003) | 0.98 (112/114) [0.94, 1.00] | 3.0e-20 | 2.4e-19* |
| per-seed | 3 | DampedPersistence | 114 | +0.072 [+0.060, +0.077] (±0.002) | 0.89 (102/114) [0.82, 0.94] | 3.0e-18 | 1.8e-17* |
| per-seed | 3 | LightGBM | 114 | +0.002 [-0.002, +0.005] (±0.003) | 0.53 (60/114) [0.44, 0.62] | 7.6e-01 | 7.6e-01 |
| per-seed | 7 | Persistence | 114 | +0.243 [+0.233, +0.254] (±0.002) | 0.99 (113/114) [0.95, 1.00] | 2.0e-20 | 1.8e-19* |
| per-seed | 7 | DampedPersistence | 114 | +0.029 [+0.025, +0.035] (±0.001) | 0.89 (102/114) [0.82, 0.94] | 1.1e-15 | 4.4e-15* |
| per-seed | 7 | LightGBM | 114 | -0.002 [-0.004, +0.003] (±0.002) | 0.46 (52/114) [0.37, 0.55] | 2.6e-01 | 5.2e-01 |
| ensemble | 1 | Persistence | 114 | +0.204 [+0.181, +0.222] | 0.92 (105/114) [0.86, 0.96] | 1.5e-19 | 1.1e-18* |
| ensemble | 1 | DampedPersistence | 114 | +0.172 [+0.156, +0.194] | 0.88 (100/114) [0.80, 0.93] | 2.3e-18 | 1.1e-17* |
| ensemble | 1 | LightGBM | 114 | -0.019 [-0.029, -0.010] | 0.33 (38/114) [0.25, 0.42] | 1.1e-06 | 3.4e-06* |
| ensemble | 3 | Persistence | 114 | +0.186 [+0.179, +0.195] | 0.98 (112/114) [0.94, 1.00] | 2.3e-20 | 1.8e-19* |
| ensemble | 3 | DampedPersistence | 114 | +0.078 [+0.070, +0.084] | 0.94 (107/114) [0.88, 0.97] | 2.0e-19 | 1.2e-18* |
| ensemble | 3 | LightGBM | 114 | +0.009 [+0.005, +0.014] | 0.66 (75/114) [0.57, 0.74] | 1.4e-05 | 2.8e-05* |
| ensemble | 7 | Persistence | 114 | +0.247 [+0.237, +0.258] | 1.00 (114/114) [0.97, 1.00] | 1.9e-20 | 1.7e-19* |
| ensemble | 7 | DampedPersistence | 114 | +0.035 [+0.031, +0.041] | 0.92 (105/114) [0.86, 0.96] | 2.9e-18 | 1.2e-17* |
| ensemble | 7 | LightGBM | 114 | +0.004 [+0.001, +0.009] | 0.61 (69/114) [0.51, 0.69] | 9.4e-02 | 9.4e-02 |

## HUC2 cluster-bootstrap robustness (per-seed protocol)

Median-skill 95% CI when whole HUC2 regions are resampled with replacement (stations within a region are spatially correlated and are not treated as independent):

| horizon | reference | station-boot CI | HUC2-cluster CI (k regions) |
|---|---|---|---|
| 1 | Persistence | [+0.169, +0.208] | [+0.161, +0.229] (k=16) |
| 1 | DampedPersistence | [+0.134, +0.172] | [+0.115, +0.200] (k=16) |
| 1 | LightGBM | [-0.050, -0.034] | [-0.054, -0.032] (k=16) |
| 3 | Persistence | [+0.170, +0.187] | [+0.165, +0.190] (k=16) |
| 3 | DampedPersistence | [+0.060, +0.077] | [+0.057, +0.082] (k=16) |
| 3 | LightGBM | [-0.002, +0.005] | [-0.005, +0.006] (k=16) |
| 7 | Persistence | [+0.233, +0.254] | [+0.225, +0.268] (k=16) |
| 7 | DampedPersistence | [+0.025, +0.035] | [+0.021, +0.037] (k=16) |
| 7 | LightGBM | [-0.004, +0.003] | [-0.008, +0.004] (k=16) |
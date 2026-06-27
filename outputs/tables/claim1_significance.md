# Claim 1 — accuracy vs baselines, per-station significance (n=40)

ThermoRoute = 5-seed mean. Skill = 1−RMSE/RMSE_ref; win-rate = fraction of stations where ThermoRoute is better; Wilcoxon = paired signed-rank p.

| horizon | reference | median skill [95% CI] | win-rate | Wilcoxon p |
|---|---|---|---|---|
| 1 | Persistence | +0.163 [+0.123, +0.204] | 0.92 | 9.6e-10* |
| 1 | DampedPersistence | +0.127 [+0.077, +0.171] | 0.81 | 5.5e-07* |
| 1 | LightGBM | -0.002 [-0.027, +0.009] | 0.47 | 7.5e-01 |
| 3 | Persistence | +0.180 [+0.160, +0.204] | 1.00 | 2.9e-11* |
| 3 | DampedPersistence | +0.057 [+0.050, +0.078] | 0.92 | 9.6e-10* |
| 3 | LightGBM | +0.014 [+0.005, +0.034] | 0.69 | 1.1e-03* |
| 7 | Persistence | +0.253 [+0.236, +0.273] | 1.00 | 2.9e-11* |
| 7 | DampedPersistence | +0.034 [+0.025, +0.043] | 0.86 | 6.3e-07* |
| 7 | LightGBM | +0.015 [+0.004, +0.034] | 0.69 | 6.7e-03* |
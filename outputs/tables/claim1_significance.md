# Claim 1 — accuracy vs baselines, per-station significance (n=40)

ThermoRoute = 5-seed mean. Skill = 1−RMSE/RMSE_ref; win-rate = fraction of stations where ThermoRoute is better; Wilcoxon = paired signed-rank p.

| horizon | reference | median skill [95% CI] | win-rate | Wilcoxon p |
|---|---|---|---|---|
| 1 | Persistence | +0.205 [+0.184, +0.225] | 0.93 | 2.2e-19* |
| 1 | DampedPersistence | +0.174 [+0.150, +0.196] | 0.88 | 3.9e-18* |
| 1 | LightGBM | -0.018 [-0.027, -0.009] | 0.33 | 3.3e-05* |
| 3 | Persistence | +0.189 [+0.183, +0.196] | 0.98 | 2.3e-20* |
| 3 | DampedPersistence | +0.080 [+0.070, +0.086] | 0.94 | 2.5e-19* |
| 3 | LightGBM | +0.014 [+0.011, +0.018] | 0.75 | 1.3e-08* |
| 7 | Persistence | +0.251 [+0.241, +0.260] | 0.99 | 2.0e-20* |
| 7 | DampedPersistence | +0.037 [+0.034, +0.046] | 0.93 | 3.8e-19* |
| 7 | LightGBM | +0.006 [+0.003, +0.012] | 0.67 | 8.6e-04* |
# Claim 2 — K-fold leave-group-out transfer (every station held out once)

4 folds. Mean ± std of transfer skill across folds.

| horizon | TR RMSE (mean) | skill vs persistence | skill vs damped |
|---|---|---|---|
| 1 | 0.682 | +0.126 ± 0.023 | +0.089 ± 0.025 |
| 3 | 1.289 | +0.141 ± 0.015 | +0.023 ± 0.019 |
| 7 | 1.538 | +0.234 ± 0.016 | +0.012 ± 0.003 |

# Claim 4 — ablations (3-seed mean median RMSE) + paired test vs full

| variant | h1 | h3 | h7 | Wilcoxon p (h3, vs full) |
|---|---|---|---|---|
| ThermoRoute | 0.554 | 1.175 | 1.490 | — |
| TR-noPrior | 0.995 | 1.302 | 1.531 | 2.9e-11* |
| TR-fixedKappa | 0.559 | 1.177 | 1.482 | 2.8e-05* |
| TR-noRouter | 0.568 | 1.189 | 1.497 | 2.0e-09* |
| TR-noMoE | 0.623 | 1.236 | 1.527 | 5.5e-10* |
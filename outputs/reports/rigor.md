# Claim 2 — K-fold leave-group-out transfer (every station held out once)

4 folds. Mean ± std of transfer skill across folds.

| horizon | TR RMSE (mean) | skill vs persistence | skill vs damped |
|---|---|---|---|
| 1 | 0.696 | +0.173 ± 0.021 | +0.140 ± 0.023 |
| 3 | 1.361 | +0.169 ± 0.010 | +0.057 ± 0.012 |
| 7 | 1.654 | +0.241 ± 0.011 | +0.028 ± 0.011 |

# Claim 4 — ablations (3-seed mean median RMSE) + paired test vs full

| variant | h1 | h3 | h7 | Wilcoxon p (h3, vs full) |
|---|---|---|---|---|
| ThermoRoute | 0.630 | 1.289 | 1.656 | — |
| TR-noPrior | 1.327 | 1.528 | 1.699 | 2.2e-20* |
| TR-fixedKappa | 0.642 | 1.294 | 1.658 | 2.3e-01 |
| TR-noRouter | 0.639 | 1.302 | 1.662 | 1.3e-06* |
| TR-noMoE | 0.733 | 1.343 | 1.699 | 9.5e-18* |
# Claim 2 — K-fold leave-group-out transfer (every station held out once)

4 folds. Mean ± std of transfer skill across folds.

| horizon | TR RMSE (mean) | skill vs persistence | skill vs damped |
|---|---|---|---|
| 1 | 0.688 | +0.181 ± 0.023 | +0.149 ± 0.024 |
| 3 | 1.360 | +0.170 ± 0.010 | +0.058 ± 0.012 |
| 7 | 1.654 | +0.240 ± 0.010 | +0.028 ± 0.010 |

# Claim 4 — ablations (3-seed mean median RMSE) + paired test vs full

| variant | h1 | h3 | h7 | Wilcoxon p (h3, vs full) |
|---|---|---|---|---|
| ThermoRoute | 0.629 | 1.282 | 1.655 | — |
| TR-noPrior | 1.149 | 1.416 | 1.670 | 3.9e-20* |
| TR-fixedKappa | 0.635 | 1.286 | 1.659 | 1.1e-05* |
| TR-noRouter | 0.640 | 1.293 | 1.666 | 2.4e-16* |
| TR-noMoE | 0.733 | 1.345 | 1.690 | 1.0e-16* |
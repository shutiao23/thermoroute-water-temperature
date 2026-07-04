# TUURT transfer triad — temporal / unseen-station / ungaged-region

Skill = 1 − RMSE(ThermoRoute)/RMSE(reference), median across held-out stations. The HESS review (Corona & Hogue 2025, 29:2521) prescribes these three tests as the evaluation standard for extrapolation confidence; most ML-SWT studies run none. Positive skill at every arm and lead means the physics-biased forecaster generalises in time AND space.

| arm | protocol | h1 skill vs persist | h3 | h7 |
|---|---|---|---|---|
| **Temporal** | future years, seen stations (n=114) | +0.204 | +0.186 | +0.247 |
| **Unseen station** | random 4-fold leave-group-out | +0.173 | +0.169 | +0.241 |
| **Ungaged region** | leave-HUC2-region-out (~358 km) | +0.151 | +0.169 | +0.237 |

### vs damped persistence (the harder reference)

| arm | h1 | h3 | h7 |
|---|---|---|---|
| Temporal | +0.172 | +0.078 | +0.035 |
| Unseen station | +0.140 | +0.057 | +0.028 |
| Ungaged region | +0.120 | +0.052 | +0.026 |

All three arms show positive skill against both references at every lead — the transfer holds in time (temporal), to unseen gages (random), and to whole unseen regions (~358 km extrapolation). Under the ungaged-region arm ThermoRoute ties the strong global LightGBM (see `region_transfer.md`); the differentiator there is the retained Proposition-1 floor and calibrated intervals, not point skill.
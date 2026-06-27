# Decision value — Relative Economic Value (blind test)

Peak REV and REV at representative cost–loss ratios α. 1.0 = perfect-forecast value; 0 = no better than climatology. Probabilistic forecasts sweep the optimal threshold; persistence is a fixed deterministic warning.

| model | h | base rate | REV_max | α* | REV@0.05 | REV@0.1 | REV@0.2 | REV@0.5 |
|---|---|---|---|---|---|---|---|---|
| DampedPersistence | 1 | 0.067 | 0.806 | 0.07 | 0.742 | 0.798 | 0.767 | 0.599 |
| LightGBM | 1 | 0.067 | 0.889 | 0.07 | 0.871 | 0.855 | 0.743 | 0.435 |
| Persistence | 1 | 0.067 | 0.803 | 0.07 | 0.736 | 0.796 | 0.770 | 0.633 |
| ThermoRoute | 1 | 0.071 | 0.812 | 0.07 | 0.789 | 0.762 | 0.679 | 0.213 |
| DampedPersistence | 3 | 0.067 | 0.556 | 0.07 | 0.400 | 0.545 | 0.503 | 0.279 |
| LightGBM | 3 | 0.067 | 0.808 | 0.05 | 0.808 | 0.664 | 0.456 | 0.129 |
| Persistence | 3 | 0.067 | 0.642 | 0.07 | 0.519 | 0.630 | 0.583 | 0.333 |
| ThermoRoute | 3 | 0.071 | 0.791 | 0.07 | 0.762 | 0.715 | 0.405 | -0.019 |
| DampedPersistence | 7 | 0.068 | 0.249 | 0.07 | -0.024 | 0.234 | 0.179 | -0.122 |
| LightGBM | 7 | 0.068 | 0.719 | 0.02 | 0.706 | 0.587 | 0.270 | -0.122 |
| Persistence | 7 | 0.068 | 0.408 | 0.07 | 0.200 | 0.388 | 0.311 | -0.102 |
| ThermoRoute | 7 | 0.071 | 0.689 | 0.07 | 0.545 | 0.646 | 0.281 | -0.039 |
# Relative Economic Value (REV) of the exceedance warning — full cost–loss curve

REV(α) for the decision 'protect iff p>α' against a high-temperature (train-q90) exceedance, over the full cost–loss grid. Probabilistic models (ThermoRoute/LightGBM/LSTM) use calibrated p_exceed; persistence/climatology issue a deterministic warning when their point forecast crosses the threshold. Framing follows Modi et al. 2025 (HESS 29:5593): value, not RMSE.

| horizon | model | REV_max | REV@0.05 | REV@0.1 | REV@0.2 | REV@0.5 |
|---|---|---|---|---|---|---|
| 1 | ThermoRoute | 0.907 | 0.877 | 0.902 | 0.859 | 0.683 |
| 1 | LightGBM | 0.898 | 0.870 | 0.894 | 0.850 | 0.672 |
| 1 | LSTM | 0.880 | 0.845 | 0.873 | 0.824 | 0.618 |
| 1 | Persistence | 0.843 | 0.628 | 0.814 | 0.828 | 0.724 |
| 1 | Climatology | 0.437 | -0.399 | 0.322 | 0.412 | 0.249 |
| 3 | ThermoRoute | 0.855 | 0.817 | 0.847 | 0.775 | 0.481 |
| 3 | LightGBM | 0.844 | 0.810 | 0.841 | 0.760 | 0.446 |
| 3 | LSTM | 0.839 | 0.794 | 0.831 | 0.754 | 0.451 |
| 3 | Persistence | 0.706 | 0.302 | 0.651 | 0.677 | 0.483 |
| 3 | Climatology | 0.436 | -0.399 | 0.322 | 0.412 | 0.248 |
| 7 | ThermoRoute | 0.823 | 0.774 | 0.814 | 0.718 | 0.353 |
| 7 | LightGBM | 0.813 | 0.777 | 0.807 | 0.700 | 0.336 |
| 7 | LSTM | 0.810 | 0.757 | 0.802 | 0.702 | 0.308 |
| 7 | Persistence | 0.589 | 0.022 | 0.511 | 0.547 | 0.276 |
| 7 | Climatology | 0.437 | -0.399 | 0.322 | 0.412 | 0.250 |
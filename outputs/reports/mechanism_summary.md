# Mechanism summary (ThermoRoute, seed 0)

- Trained 3 epochs, params=33664, val median-RMSE=0.5045 °C

## Dynamic relaxation rate κ (per-day memory)

| station | mean κ | κ low-flow | κ high-flow | implied memory 1/κ (d) |
|---|---|---|---|---|
| b1 | 0.066 | 0.039 | 0.098 | 15.1 |
| s2 | 0.052 | 0.031 | 0.076 | 19.4 |
| p3 | 0.045 | 0.030 | 0.064 | 22.3 |

## Top variable×lag drivers by horizon (router weight share)

- **h=1d**: DH (46%), WLEVEL (21%), PRCP (20%); dominant WTEMP lag = 1 d
- **h=3d**: TEMP (42%), PRCP (20%), WLEVEL (17%); dominant WTEMP lag = 2 d
- **h=7d**: TEMP (37%), PRCP (24%), FLOW (14%); dominant WTEMP lag = 2 d
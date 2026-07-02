# Mechanism summary (ThermoRoute, seed 0)

- Trained 21 epochs, params=33664, val median-RMSE=0.5100 °C

## Dynamic relaxation rate κ (per-day memory)

| station | mean κ | κ low-flow | κ high-flow | implied memory 1/κ (d) |
|---|---|---|---|---|
| b1 | 0.157 | 0.117 | 0.207 | 6.4 |
| s2 | 0.109 | 0.081 | 0.145 | 9.1 |
| p3 | 0.055 | 0.040 | 0.073 | 18.1 |

## Top variable×lag drivers by horizon (router weight share)

- **h=1d**: WLEVEL (68%), FLOW (14%), TEMP (9%); dominant WTEMP lag = 0 d
- **h=3d**: WLEVEL (51%), WTEMP (18%), FLOW (13%); dominant WTEMP lag = 11 d
- **h=7d**: WLEVEL (67%), WDSP (26%), PRCP (7%); dominant WTEMP lag = 0 d
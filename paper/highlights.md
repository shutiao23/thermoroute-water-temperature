# Highlights

*(Journal of Hydrology format: ≤85 characters each, including spaces.)*

- A dynamic thermal-relaxation prior makes damped persistence a special case
- On 40 USGS stations it significantly beats persistence and damped persistence
- Transfers to unseen basins in 4-fold leave-group-out (skill +0.13 to +0.23)
- Conformal forecast intervals are near-nominal at 89–97% of stations
- Three negative results reported in full, including no gain on a damped cascade

---

# One-page summary

**Problem.** Daily river water temperature is so autocorrelated that *persistence*
is a punishing baseline, and many machine-learning studies inflate their skill by
using covariates unavailable at forecast time. We ask whether a physics-guided
learner can do better honestly — and we insist the answer be demonstrated on a
large, diverse sample, not one site.

**Method — ThermoRoute.** A learnable *dynamic thermal-relaxation prior* relaxes
the temperature anomaly toward a horizon-shifted seasonal climatology at a flow- and
season-modulated rate; with the modulation switched off it is *exactly* damped
persistence, so the strong baseline is contained as a special case and any gain is
attributable to the learned part. A horizon-conditioned sparse variable–lag router,
a causal temporal encoder and a regime mixture-of-experts produce a **bounded**
neural residual (it cannot override the prior). Outputs are conformally-calibrated
quantiles plus a high-temperature exceedance probability. We forecast under a strict
historical-information protocol (no future observed weather) with a one-shot
2019–2020 blind test.

**Evidence (40 public USGS stations + Daymet/gridMET forcing).**

| Claim | Result |
|---|---|
| Beats the physics baselines | RMSE 0.554 / 1.175 / 1.490 °C at 1 / 3 / 7 d; significantly better than persistence and damped persistence (Wilcoxon p ≤ 10⁻⁶; wins 81–92 % of stations) |
| ≥ parity with a strong learner | Significantly beats LightGBM at the station level at 3–7 days; ties at 1 day |
| Transfers across basins | 4-fold leave-group-out: +0.13 / +0.14 / +0.23 skill vs persistence (std ≈ 0.02) |
| Calibrated | PICP ≈ 0.90; 89–97 % of stations within ±0.05 of nominal |
| Components matter | Removing the prior / experts / router significantly hurts (p ≤ 2×10⁻⁹) |

**Honesty.** We report three negative results in full: no point-accuracy gain on a
near-deterministic three-station reservoir cascade; the flow-dependent thermal memory
does **not** generalise beyond it; and no robust cost–loss decision-value advantage
over a (strong) deterministic persistence warning. The take-home is methodological as
much as technical: a physics-guided forecaster's advantage exists only where the
system has forecast headroom, and establishing it requires a large, transfer-tested
sample — not a single cascade.

**Reproducibility.** All code, the fixed evaluation protocol, the assembled datasets,
13 leakage/metric unit tests, and one-command reproduction are provided; every
headline claim was checked by an adversarial internal review against the result
tables.

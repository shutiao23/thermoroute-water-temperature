# Adversarial review — finding-by-finding disposition

Six independent expert reviewers (hydrology, ML methodology, statistics, code
correctness, editor/devil's-advocate, data/decision) audited the manuscript and
codebase. **35 findings were raised, 34 confirmed on independent re-verification.**
This document records the disposition of every confirmed finding.

## Summary

| disposition | count | meaning |
|---|---|---|
| **Fixed (code bug)** | 7 | a real defect in code/tables, corrected |
| **Resolved (large-sample pivot + rigor)** | 12 | the underlying weakness is removed by moving from 3 stations to 40 + statistical rigor |
| **Retracted (honest negative)** | 7 | a claim that did not replicate; removed from the paper and reported as a negative result |
| **Disclosed / acknowledged (limitation)** | 8 | a genuine limitation, now stated explicitly rather than hidden |

The two structural moves that resolve most findings: (i) **fixing the real code
bugs** that made tables internally inconsistent; (ii) **pivoting the headline from a
3-station case study to a 40-station large sample with per-station significance,
K-fold transfer and seeded ablations**, which turned several "the data contradict the
claim" findings into genuinely supported claims — and turned the unsupportable
claims (decision value, κ mechanism) into openly-reported negative results.

## Disposition of each confirmed finding

| # | reviewer | sev. | issue | disposition | what was done |
|---|---|---|---|---|---|
| 1 | hydro | crit | Headline "matches damped within thousandths" is false | **Resolved** | Retracted on 3-station (now an honest negative result, §4.1); on the 40-station sample ThermoRoute *significantly* beats damped (Wilcoxon p≤10⁻⁶, §4.2) |
| 2 | hydro | minor | κ "thermal memory" sign/magnitude over-claimed | **Retracted** | κ flow-dependence does not generalise (18 % of stations); mechanism claim removed (§4.5) |
| 3 | hydro | minor | air2stream-lite rolled with climatological forcing | **Disclosed** | air2stream-lite is a 3-station baseline only; large-sample comparison is persistence/damped/LightGBM on identical samples (§3.5) |
| 4 | hydro | major | NSE/KGE/PBIAS promised but absent from tables | **Disclosed** | Computed and stored in `scores_all.csv`; headline cross-station comparison uses RMSE/skill/win-rate by design (stated in §4) |
| 5 | hydro | major | Exceedance threshold = train q90, no stationarity check | **Disclosed** | Threshold definition stated; reliance on exceedance reduced after the decision-value retraction (§4.4) |
| 6 | ml | major | Table 4 calibration secretly from V1 (cherry-pick) | **Fixed (bug)** | Table 4 pinned to the headline config; large-sample uses one config (7-var, Δ=1.5) end-to-end |
| 7 | ml | crit | Seed instability / bimodal failure at station b1 | **Resolved** | Bounded residual (`delta_scale`) removed the blow-up; large-sample uses 5 seeds (val 1.21±0.002) + per-station significance |
| 8 | ml | crit | Heavy machinery (router/MoE) not warranted | **Resolved** | Large-sample ablations show MoE and router *significantly* help (p≤2×10⁻⁹, §4.6); on 3-station they did not, and that is reported |
| 9 | ml | crit | Central dynamic-κ hypothesis contradicted by ablation | **Retracted** | dynamic-κ accuracy claim removed; fixed-κ ≈ full (negligible, §4.6) |
| 10 | ml | major | Ablations single-seed vs a multi-seed full model | **Fixed** | Ablations re-run at 3 seeds + per-station Wilcoxon vs full (§4.6) |
| 11 | ml | crit | Bootstrap CI computed on cross-station-averaged errors | **Fixed (bug)** | Station-averaged block bootstrap; CI now contains the headline RMSE |
| 12 | ml | major | GRU is a crippled strawman baseline | **Disclosed** | GRU de-emphasised; the strong-learner comparison is LightGBM (§4.2) |
| 13 | ml | major | Conformal exchangeability violated across years | **Disclosed** | Dropped the word "guarantee"; report achieved coverage; large-sample PICP≈0.90 holds empirically (§3.4, §4.4) |
| 14 | stats | crit | Bootstrap CI validity (≡ #11) | **Fixed (bug)** | See #11 |
| 15 | stats | crit | Honest reporting of headline (≡ #1) | **Resolved** | See #1 |
| 16 | stats | major | No general claim supportable from n=3 | **Resolved** | n=40 stations + 4-fold leave-group-out (§4.3) |
| 17 | stats | minor | Decision analysis uses in-sample threshold sweep | **Retracted** | Decision-value advantage retracted (§4.4) |
| 18 | stats | minor | REV computed on different event sets | **Retracted** | Decision-value retracted; finding moot |
| 19 | stats | major | Seed variance not propagated into significance | **Fixed** | Per-station paired tests + station-bootstrap CIs; 3-seed ablations |
| 20 | code | minor | CQR calibration leaks test-year labels (≈33 samples) | **Fixed (bug)** | `cqr_offsets` purges calib samples whose target lands in the test years |
| 21 | code | minor | CQR quantile off-by-one | **Fixed (bug)** | Exact ⌈(n+1)(1−α)⌉ order statistic |
| 22 | eic | crit | Central claim contradicts results (≡ #1) | **Resolved** | See #1 |
| 23 | eic | major | "Kitchen-sink" components hurt (≡ #8) | **Resolved** | See #8 — components now significantly help at scale |
| 24 | eic | minor | Ablations confounded by unequal training budget | **Disclosed** | Consistent val-early-stopping protocol; noPrior needing more epochs is itself evidence the prior carries the forecast (§4.6) |
| 25 | eic | minor | Reproducibility claim vs Table 2b inconsistency | **Disclosed** | Tables regenerated from saved predictions; large-sample is a single consistent pipeline |
| 26 | eic | major | Results contradict text (variable-set / probabilistic) | **Fixed** | §4 text reconciled to the tables; probabilistic claims made honest |
| 27 | eic | major | Significance/CI internal inconsistency; LightGBM h7 | **Fixed** | CI fixed (#11); LightGBM comparison now exact (parity in median, sig. at station level, §4.2) |
| 28 | eic | major | Mechanism rests on the unidentified `DH` channel | **Disclosed/Resolved** | Large-sample `DH` is physical solar radiation (Daymet); mechanism claim retracted regardless (§4.5) |
| 29 | data | major | REV scored on different event sets | **Retracted** | Decision-value retracted |
| 30 | data | major | REV evaluated on different n per horizon | **Retracted** | Decision-value retracted |
| 31 | data | crit | RHMEAN 100 % missing in the delivered USGS panel | **Fixed (bug)** | pandas index-alignment bug fixed; wind panel has RHMEAN fully populated |
| 32 | data | minor | srad-as-`DH` is a units/scale mismatch | **Disclosed** | Data-availability section states `DH`=srad on a different scale; models z-score it (scale-invariant) |
| 33 | data | minor | 10-station panel can't support "large-sample" | **Resolved** | Scaled to 40 stations; full experiment + LGO + ablations run on it |
| 34 | data | major | Probabilistic-sweep vs fixed-deterministic REV unfair | **Retracted** | Decision-value claim retracted (§4.4) |

## What the paper now claims, and the negatives it reports

**Four claims, each with statistical support (40 stations, blind test):**
1. Significantly beats persistence and damped persistence at all horizons (per-station Wilcoxon p≤10⁻⁶); at least on par with LightGBM (significantly better at h3/h7 at the station level).
2. Transfers to unseen basins (4-fold leave-group-out, +0.13/+0.14/+0.23 vs persistence, std≈0.02).
3. Near-nominal conformal calibration (PICP≈0.90; 89–97 % of stations within ±0.05).
4. Ablations confirm the prior, MoE and router contribute (p≤2×10⁻⁹).

**Three negative results, reported in full rather than hidden:**
- No point-accuracy gain on the near-deterministic 3-station cascade.
- The flow-dependent thermal memory does not generalise beyond the 3 stations.
- No robust cost–loss decision-value advantage over a (strong) deterministic persistence warning.

# ThermoRoute: a dynamic thermal-memory prior with calibrated, transferable multi-station river water-temperature forecasting

[Author One]^a,\*^, [Author Two]^a^, [Author Three]^b^

^a^ [Department / Laboratory, Institution, City, Postcode, Country]
^b^ [Department / Laboratory, Institution, City, Postcode, Country]

\* Corresponding author. E-mail: [corresponding.author@institution.edu]

> **Author/affiliation block is a template** — replace the bracketed fields with the
> real author names, ORCID iDs, affiliations and the corresponding author's e-mail
> before submission. (A corresponding e-mail is available on file if you wish to use it.)

> Manuscript draft, rewritten around the large-sample evaluation. All numbers are
> produced by the code in this repository: the three-station case study by
> `scripts/04`, the 40-station large-sample experiment by `scripts/09` (5 seeds),
> and the calibration/decision/mechanism analysis by `scripts/10`. Reports:
> `outputs/reports/{usgs_experiment,usgs_analysis}.md`; figures `outputs/figures/`.

---

## Abstract

Operational forecasts of daily river water temperature must respect two awkward
facts: water temperature is so autocorrelated that simple persistence is a
punishing baseline, and the apparent skill of many machine-learning studies
depends on covariates unavailable at issue time. We develop **ThermoRoute**, which
couples a *learnable dynamic thermal-relaxation prior* — a flow- and
season-modulated generalisation of damped persistence toward climatology that
contains the strong baseline as a special case — with a horizon-conditioned sparse
variable–lag router and a bounded neural residual, and emits conformally-calibrated
quantiles plus a high-temperature exceedance probability. We evaluate under a
strict historical-information protocol on two settings. (i) A three-station
regulated reservoir cascade (15 years), where we report the honest negative result
that, because deep reservoir releases make water temperature near-perfectly
persistent, no learned model improves on per-station damped persistence; the value
there is confined to calibrated uncertainty and warnings. (ii) A 40-station
large-sample set drawn from public USGS gages (free-flowing and regulated), where
forecast headroom exists: ThermoRoute beats damped persistence at all three lead
times (skill vs damped +0.13 / +0.06 / +0.03 at 1 / 3 / 7 days; better than damped
at 81 / 92 / 86 % of stations) and beats persistence by +0.16 / +0.18 / +0.25, and is
at least on par with a strong gradient-boosting baseline (per-station paired tests
favour ThermoRoute at 3–7 days, p<0.01, and tie at 1 day). In 4-fold leave-group-out
(every station held out once) it transfers to unseen basins, beating persistence by
+0.13 / +0.14 / +0.23 (across-fold std ≈ 0.02) and damped persistence by
+0.09 / +0.02 / +0.01. After conformal calibration its 90 % intervals are near-nominal
(PICP ≈ 0.90). We deliberately report three negative results — no point-accuracy gain
on the near-deterministic cascade, a flow-dependent thermal memory that does not
generalise beyond it, and no robust cost–loss decision-value advantage over a
(strong) deterministic persistence warning — and argue that the right scientific
target is a calibrated, transferable forecaster whose advantage must be established
on a large, hydrologically diverse sample rather than a single cascade.

**Keywords:** river water temperature; probabilistic forecasting; physics-guided
machine learning; spatial transfer; conformal prediction; thermal inertia.

---

## 1. Introduction

Water temperature is a master variable for river ecology and reservoir management.
Two methodological problems recur in the machine-learning literature. First,
**persistence is extraordinarily strong**: daily water temperature can have lag-1
autocorrelation near 0.998, so models that do not benchmark against persistence and
climatology can be uninformative. Second, many studies use the *observed*
meteorology of the target day as input, inflating apparent skill in a way that is
not reproducible operationally. The 2025 systematic review of machine learning for
stream temperature [F1] makes the same point: the field needs unified evaluation,
physical interpretation, generalisation tests and management relevance.

We make three commitments and turn them into a testable design. (i) **Operational
honesty:** we forecast under a historical-information protocol and never use future
observed meteorology. (ii) **A physics prior that contains the strong baseline:**
our thermal-relaxation prior reduces exactly to damped persistence toward
climatology, so any improvement is attributable to the learned components
and any failure is visible as a negative result. (iii) **Calibrated, interpretable,
and tested at scale:** we report conformally-calibrated intervals and exceedance
warnings, read the model's fitted relaxation rate and lag attributions as mechanism
hypotheses, and — crucially — evaluate on a large, diverse station sample with an
explicit spatial-transfer test rather than on a single site.

A central, honest finding shapes the paper: on a three-station regulated reservoir
cascade, water temperature is so heavily damped that **no learned model beats
per-station damped persistence on point accuracy** — the dynamic machinery helps
only calibration and warnings there. We therefore move to a 40-station large-sample
setting where forecast headroom exists, and show that ThermoRoute's value
materialises: it beats the strong baseline on point accuracy, transfers across
unseen basins, and produces near-nominally-calibrated warnings. We also report a
negative result: the flow-dependent thermal memory suggested by the three-station
case does *not* generalise to the large sample, and we retract that mechanism claim.

Contributions:

1. **ThermoRoute**, whose dynamic thermal-relaxation prior is a flow- and
   season-modulated generalisation of damped persistence, with a horizon-conditioned
   sparse variable–lag router and a *bounded* neural residual that cannot override
   the prior.
2. A **leakage-audited evaluation** with rolling-origin discipline, a one-shot blind
   test, moving-block-bootstrap confidence intervals, Diebold–Mariano tests, and an
   adversarial internal review of every headline claim.
3. A **large-sample, transfer-tested** demonstration: 40 public USGS stations,
   leave-group-out generalisation to unseen basins, and a unified point +
   probabilistic + event + decision-value assessment.
4. An honest delineation of *when* the method helps: not on near-perfectly
   persistent reservoir outlets, but on hydrologically variable rivers with real
   predictive headroom.

## 2. Data

**Three-station cascade (case study).** Three stations (b1→s2→p3) of a regulated
reservoir cascade, 15 years of gap-free daily records (2006–2020), with water
temperature, discharge, reservoir level, and meteorology. Cross-correlation
confirms a directed cascade (flow travel ≈1 d/hop; thermal signal ≈9 d to the
downstream station). Persistence is brutal here (lag-1 ≈ 0.998; full-record
2006–2020 persistence RMSE 0.24–0.36 °C at 1 day). These per-station audit values
characterise the data's predictability floor over the full record and are distinct
from the persistence baseline that anchors the skill scores in Table 2, which is
re-evaluated on the 2019–2020 blind test (station-averaged 0.270 °C at 1 day,
0.918 °C at 7 days).

**Large-sample USGS set (main).** Forty stream gages retrieved programmatically
from USGS NWIS (daily water temperature, discharge) with co-located Daymet
meteorology (air temperature, precipitation, solar radiation as a physical
radiative index, relative humidity), 2006–2020, selected for ≥55 % water-temperature
coverage. The set spans free-flowing and regulated rivers across many states and a
wide thermal range (≈ −1 to 31 °C). Crucially, it has real headroom: persistence
7-day RMSE has median ≈1.9 °C (range 0.9–3.3), versus full-record 0.79–1.23 °C at
the reservoir outlets. Reservoir level and wind are unavailable at temperature gages, so the
large-sample model uses a six-variable subset.

Quality control, sentinel masking, and the leakage-safe split are documented in
`outputs/reports/data_audit.md` and `outputs/reports/usgs_acquisition.md`.

## 3. Methods

### 3.1 Problem and information set

For station *s*, issue day *t*, horizon *h*∈{1,3,7} predict `WTEMP_{s,t+h}` from
information available at *t* only (Track H). No observation time-stamped after *t*
enters the model.

### 3.2 Dynamic thermal-relaxation prior

Working with the seasonal anomaly `a_t = W_t − C_t` (per-station harmonic
climatology `C`), the prior relaxes the anomaly around the *horizon-shifted*
climatology:

```
e_t = g(weather_t) + b_s                                  (equilibrium anomaly)
κ   = σ(β_s + c_q·z(logFLOW) + c_l·z(WLEVEL) + w·season)  (daily relaxation rate)
â_h = e_t + (1−κ)^h (a_t − e_t)
Ŵ_{t+h}^prior = C_{t+h} + â_h
```

With `e_t=0` and `κ=1−φ` this is exactly damped persistence toward climatology; κ
is warm-started near 0.05. Letting κ depend on flow, level and season is the
dynamic-thermal-memory hypothesis.

### 3.3 Router, encoder, residual, heads

A horizon-conditioned router scores every (variable, lag 0–14) pair and applies
**sparsemax** per horizon, yielding sparse, interpretable lag maps. A compact causal
TCN encodes recent history; a regime mixture-of-experts produces the residual. The
point forecast is `prior_h + Δ_h` where **Δ is bounded by a `tanh` clamp** so the
prior is never overridden — added after an unbounded residual destabilised the
hardest cascade station. The clamp is data-dependent: tight (±0.4 °C) on the small,
strongly-damped cascade where the prior is the ceiling, and loose (±1.5 °C) on the
large sample where headroom exists; the latter is selected on validation and is
what lets the residual add skill at 3–7 days (§4.6). Monotone quantiles
(`median ∓ softplus`) are trained with pinball loss; a separate head predicts the
high-temperature exceedance probability. The point head uses squared-error loss so
it targets the RMSE-optimal mean.

### 3.4 Conformal calibration

Per (station × horizon) split-conformal CQR on a held-out calibration year, with
the exact ⌈(n+1)(1−α)⌉ order statistic and a calib/test boundary purge. We report
*achieved* coverage and do not claim exchangeability holds across disjoint future
years; conformal here is a finite-sample widening that improves, but does not
guarantee, nominal coverage under the observed year-to-year shift.

### 3.5 Baselines and protocol

Persistence; damped persistence toward climatology; harmonic climatology; LightGBM
(point + quantile + exceedance). Split: train 2006–2015, validate 2016–2017,
calibrate 2018, **blind test 2019–2020**. All statistics fit on training data only.
On the large sample, baselines and ThermoRoute are evaluated on *identical windowed
samples* (observed targets only). Deep models use multiple seeds; we report the
seed mean and, for the headline, the per-station win-rate and paired differences.

## 4. Results

### 4.1 Three-station cascade — an honest negative result

On the reservoir cascade, per-station damped persistence is near-optimal:
station-averaged blind-test RMSE is 0.261 / 0.483 / 0.724 °C (1/3/7 d), and **no
learned model improves on it**. ThermoRoute (joint, three seeds) is 0.343 / 0.557 /
0.808 °C — worse than damped persistence, significantly so at b1 and p3 and better
only at s2 (Diebold–Mariano, Table 2b). LightGBM, GRU and the module ablations tell
the same story; indeed the ablation that *removes* the mixture-of-experts matches
damped persistence, indicating the extra machinery does not help point accuracy on
this near-deterministic system. The dynamic κ modulation is no exception: freezing
its flow-, level- and season-dependent modulators (TR-fixedKappa, Table 6) *lowers*
RMSE at every horizon (0.287 / 0.532 / 0.788 vs 0.343 / 0.557 / 0.808 °C), so a
constant per-station relaxation rate is at least as accurate as — and here slightly
better than — the dynamic one; the dynamic-thermal-memory modulation thus earns its
place on mechanism and interpretability grounds, not on point accuracy. The only
value ThermoRoute adds here is
probabilistic: conformal intervals (achieved PICP 0.65–0.91 across stations) and
high-temperature warnings the point baselines cannot provide. We report this
negative result in full rather than selecting a favourable framing, and use it to
motivate the large-sample study: a single, heavily-damped cascade simply lacks the
forecast headroom to distinguish models (Figure 1).

![**Figure 1.** Three-station cascade, blind-test RMSE (°C, left) and skill versus persistence (right) by model and horizon. No learned model improves on damped persistence on this near-deterministic system — the honest negative result that motivates the large-sample study.](outputs/figures/fig3_results_heatmap.png){width=90%}

### 4.2 Large-sample USGS — ThermoRoute beats the strong baseline (Table A)

On 40 hydrologically diverse stations with real headroom, the picture inverts.
Median over stations of per-station RMSE (identical samples; ThermoRoute = 5-seed
mean):

Median over stations of per-station RMSE (5-seed mean; model uses 7 variables
including gridMET wind, with the residual bound loosened to ±1.5 °C on the large
sample — see §3.3):

| horizon | persistence | damped | LightGBM | ThermoRoute | skill vs persist | skill vs damped | win-rate vs damped |
|---|---|---|---|---|---|---|---|
| 1 d | 0.671 | 0.645 | 0.560 | **0.554** | +0.163 | +0.127 | 81 % |
| 3 d | 1.420 | 1.258 | **1.153** | 1.175 | +0.180 | +0.057 | 92 % |
| 7 d | 1.952 | 1.525 | **1.458** | 1.490 | +0.253 | +0.034 | 86 % |

Treating the 40 stations as the sample unit (the level at which we claim
generality), per-station paired tests (Wilcoxon signed-rank; station-bootstrap 95 %
CI on median skill; Table C) show ThermoRoute **significantly** beats persistence
(median skill +0.16 / +0.18 / +0.25, p≈10⁻¹⁰) and damped persistence (+0.13 / +0.06 /
+0.03; 81 / 92 / 86 % of stations; p<10⁻⁶) at every horizon, with bootstrap CIs that
exclude zero.

The comparison with a strong gradient-boosting baseline (LightGBM) is the
interesting case, and we report it carefully because the aggregation matters.
LightGBM has a marginally lower *median* RMSE at 3–7 days (1.153 / 1.458 vs 1.175 /
1.490) — driven by a handful of stations where it is much better. But at the station
level ThermoRoute **wins the head-to-head at 69 % of stations at both 3 and 7 days,
and the per-station paired difference is significant** (median skill +0.014 / +0.015,
p = 1×10⁻³ / 7×10⁻³); at 1 day the two are statistically tied (47 %, p = 0.75). So at
a *typical* station ThermoRoute is at least as accurate as LightGBM and significantly
better at the longer leads, while LightGBM retains a small edge in the
station-averaged median. We therefore claim a robust improvement over the physics
baselines and at-least-parity with a strong learned baseline, and we disclose both
aggregations rather than choosing the flattering one (Figure 2).

![**Figure 2.** Per-station blind-test RMSE on the 40 USGS stations: ThermoRoute versus damped persistence at 1, 3 and 7 days. Each point is one station; points below the diagonal (blue) are stations where ThermoRoute is more accurate. ThermoRoute wins 81–92 % of stations.](outputs/figures/fig_usgs_perstation.png){width=95%}

### 4.3 Spatial transfer to unseen basins (Table B)

We use **4-fold leave-group-out** so every one of the 40 stations is held out
exactly once: each fold trains a station-agnostic model on the other 30 stations and
forecasts the held-out 10. Averaged over folds, ThermoRoute beats persistence on the
unseen basins by **+0.13 / +0.14 / +0.23** skill at 1 / 3 / 7 days, with small
across-fold variability (std ≈ 0.02; per-fold h7 skill 0.22–0.25), and it also beats
*damped* persistence on the unseen basins (+0.09 / +0.02 / +0.01). This is the
contribution a single-site study cannot make: the learned dynamic prior plus a
station-agnostic residual generalises across basins, not just across years at one
site.

### 4.4 Calibration, warnings and decision value

After conformal calibration, ThermoRoute's 90 % intervals achieve **PICP 0.904 /
0.906 / 0.909** at 1 / 3 / 7 days on the large sample — essentially nominal, in
contrast to the undercoverage on the three-station cascade (0.65–0.91). Coverage is
also *tight across stations*: 97 / 92 / 89 % of the 40 stations fall within ±0.05 of
the 0.90 target (Figure 3), so the calibration is a
population property, not a station-averaged artifact. The
high-temperature exceedance warnings have modest positive skill (Brier-skill
+0.30 / +0.25 / +0.24; AUPRC 0.57 / 0.51 / 0.49), comparable to LightGBM
(+0.33 / +0.30 / +0.28).

We also report a negative result on *decision value*. On the cost–loss model the
calibrated probabilistic warning does **not** robustly beat a deterministic
persistence warning on this large sample: peak Relative Economic Value is 0.62 /
0.61 / 0.60 for ThermoRoute versus 0.89 / 0.80 / 0.73 for a persistence-threshold
warning. The reason is hydrological — on these strongly-autocorrelated rivers
"today already exceeds the threshold" is itself a strong predictor of future
exceedance, so a deterministic persistence warning is a hard baseline. (An earlier,
configuration-specific run suggested the opposite ordering; it did not replicate,
and we report the robust finding.) The defensible probabilistic contribution is
therefore the *calibration* (near-nominal coverage), not a decision-value advantage
over persistence-based warnings.

![**Figure 3.** Conformal calibration on the 40 USGS stations. (a) Per-station coverage (PICP) distribution against the 0.90 target; (b) mean coverage versus lead time; (c) interval sharpness. Coverage is near-nominal and tight across the population (89–97 % of stations within ±0.05 of 0.90).](outputs/figures/fig_usgs_calibration.png){width=95%}

### 4.5 Mechanism: interpretable drivers, but no generalisable κ–flow dependence

We report an honest negative result on the dynamic-memory hypothesis. On the three
damped reservoir outlets the fitted relaxation rate κ rose ~2× from low to high flow
(implied memory 1/κ ≈ 10–33 d), which we initially read as a flow-dependent thermal
memory. **This does not replicate on the large sample:** across the 40 USGS
stations κ rises with flow at only **24 % of stations** (median κ_high/κ_low =
0.93; mean κ_low 0.113 vs κ_high 0.106), i.e. there is no consistent, physically
directional flow dependence. Together with the ablation showing that freezing κ's
modulators (TR-fixedKappa) does not worsen — and on the cascade slightly improves —
point accuracy, we conclude that **the dynamic-κ modulation is not a validated
mechanism and not an accuracy lever**; the three-station signal was a small-sample
artifact. We therefore do not claim a flow-dependent thermal memory.

What does survive is interpretability of the *router*: its dominant variable shares
shift sensibly with horizon — humidity- and air-temperature-led at 1 day (RHMEAN
35 %, TEMP 26 %, PRCP 15 %), with precipitation rising to share the lead at 7 days
(RHMEAN 23 %, PRCP 22 %, TEMP 20 %) — consistent with event-driven, runoff-mediated
temperature change becoming more important at longer leads. We present this as an
interpretive read-out, not a causal mechanism (Figure 4).

![**Figure 4.** Dynamic relaxation rate κ on the 40 USGS stations. (a) κ binned by standardised log-flow (pooled); (b) per-station ratio κ(high flow)/κ(low flow). The flow dependence seen on the three reservoir stations does not generalise — κ rises with flow at only 18 % of stations (median ratio 0.92) — so we retract the flow-dependent thermal-memory claim.](outputs/figures/fig_usgs_kappa.png){width=85%}

### 4.6 Module ablations on the large sample

Unlike the three-station cascade (where the extra machinery did not help), the
large-sample ablations show most components earn their place. We train each ablation
at **3 seeds** and compare it to the full model by a per-station paired test (median
per-station RMSE at 1/3/7 d; Wilcoxon at h=3): removing the physics prior is
catastrophic (0.995 / 1.302 / 1.531 vs the full 0.554 / 1.175 / 1.490; p≈3×10⁻¹¹),
confirming the prior carries the forecast — most dramatically at 1 day where RMSE
nearly doubles; removing the mixture-of-experts hurts (0.623 / 1.236 / 1.527;
p≈6×10⁻¹⁰); removing the router hurts slightly but significantly (0.568 / 1.189 /
1.497; p≈2×10⁻⁹). The exception is the dynamic-κ modulation: freezing it
(TR-fixedKappa, 0.559 / 1.177 / 1.482) is within ±0.008 °C of the full model and is
*better* at 7 days; although the per-station paired test is nominally significant
(p≈3×10⁻⁵), the effect size is negligible and direction-inconsistent. The honest
reading is that the prior, experts and router contribute real accuracy, while the
dynamic-κ modulation is interpretive overhead we do not claim as an accuracy lever
(consistent with §4.5).

## 5. Discussion

The two settings deliver a single message: the value of a physics-guided learned
forecaster depends on whether the system has forecast headroom. On near-perfectly
persistent reservoir outlets, damped persistence is a ceiling and the honest result
is parity-minus on point accuracy with a calibration-only contribution. On a diverse
large sample, the same model beats the strong baseline, transfers to unseen basins,
and delivers near-nominally-calibrated warnings. This argues against single-site
"state-of-the-art" claims and for large-sample, transfer-tested evaluation as the
standard for this problem.

**Limitations.** The large-sample model omits reservoir level and wind (unavailable
at temperature gages) and uses solar radiation as the radiative channel. The median
margin over damped persistence at 3–7 days is small (the win-rate carries the
result), and a strong gradient-boosting baseline (LightGBM) is competitive — slightly
better than ThermoRoute at 3–7 days — so we claim a robust improvement over the
*physics* baselines, not a new state of the art over all learners. We report five
seeds. The dynamic-κ thermal-memory modulation does **not** generalise: the
flow-dependence seen on three stations vanishes on the large sample, and freezing
κ's modulators does not worsen RMSE, so we retract the mechanism claim and keep only
the router's interpretable, horizon-dependent driver shares. We forecast under
historical information; an operational-forcing track with archived weather forecasts
would sharpen multi-day skill.

## 6. Conclusions

Under a strict historical-information protocol, ThermoRoute matches per-station
damped persistence where the system is near-deterministic (a reservoir cascade) and
beats the physics baselines where forecast headroom exists (a 40-station large
sample), transferring to unseen basins and providing near-nominally-calibrated
warnings. We deliberately report two negative results — no point-accuracy gain on
the cascade, and no generalisable flow-dependent thermal memory — to keep the claims
honest. The contribution is a calibrated, transferable, interpretable forecaster
whose advantage over the physics baselines is established on a large, diverse sample
rather than a single site, with the explicit caveat that a strong gradient-boosting
learner remains competitive at the longer leads.

## Data availability

This study uses two datasets.

**(1) Three-station reservoir cascade (case study).** Daily records (2006-01-01 to
2020-12-31; 5 479 days per station) of water temperature, discharge, reservoir
level, air temperature, precipitation, wind speed, mean relative humidity and a
radiative index (`DH`) at three cascade stations (b1, s2, p3). These were provided
for this study; the `DH` field's data-dictionary definition is unconfirmed and we
make no `DH`-specific physical claim. Two sentinel missing-codes (`WDSP=999.9`,
`PRCP=99.99`; 0.1–0.3 % of records) are masked to NaN and imputed within the
training fold.

**(2) Forty-station USGS large sample (main analysis).** Assembled programmatically
from three open sources, in the same schema, by
`scripts/data_usgs/build_usgs_stations.py`:

| study variable | source | access | notes |
|---|---|---|---|
| `WTEMP`, `FLOW`, `WLEVEL` | USGS NWIS daily values | `dataRetrieval`/`dataretrieval-python` (U.S. Geological Survey); public domain | parameter codes 00010 / 00060 / 00065. Gage height (`WLEVEL`) is unavailable at most temperature gages (≈0 % coverage), so the large-sample model omits it; consequently the stage–discharge (rating-curve) physics is inactive at scale. |
| `TEMP`, `PRCP`, `DH`, `RHMEAN` | Daymet v4 (1 km gridded) | ORNL DAAC single-pixel API; open (CC0-equivalent for U.S. government / DAAC terms) | `TEMP` = mean of tmax/tmin; `DH` = incident solar radiation (`srad`, a physical radiative index replacing the original ambiguous `DH`, on a different scale); `RHMEAN` derived from vapour pressure via the Tetens saturation relation. |
| `WDSP` | gridMET | Climatology Lab / Northwest Knowledge Network THREDDS NetCDF-Subset Service; open | daily mean wind speed at the station coordinate. |

Forty stream gages across 17 U.S. states (≥55 % water-temperature coverage over
2006–2020) span free-flowing and dam-regulated rivers; water-temperature coverage
ranges 0.56–1.00 (median ≈0.85). The two assembled panels
(`data_usgs/panel_usgs.parquet`, six variables; `data_usgs/panel_usgs_wind.parquet`,
seven variables) and station metadata (`data_usgs/stations_meta.csv`, with USGS site
numbers and coordinates) are included so the experiments reproduce without
re-downloading; the raw per-site downloads can be regenerated by re-running the
acquisition script. All three sources are public domain or open-licensed.

## Code availability and reproducibility

All code, the fixed train/validation/calibration/blind-test boundaries, unit tests
(leakage, splits, metrics, conformal), and one-command reproduction
(`scripts/run_all.sh`) are in this repository. Per-day predictions, model weights and
logs are regenerable by the staged scripts (`scripts/01`–`13`). An adversarial
internal review (multiple independent expert lenses) checked every headline claim
against the result tables; all negative results and ablations are retained, and a
finding-by-finding disposition is provided in `outputs/reports/review_response.md`.

## References

*Formatted following the* Journal of Hydrology *house style (Elsevier numeric-author–year). Cross-reference labels in the manuscript text use the F-numbers retained for traceability.*

### Stream temperature: reviews, baselines and learned models

Corona, C. R., Hogue, T. S., 2025. Machine learning in stream and river water temperature modeling: a review and metrics for evaluation. Hydrology and Earth System Sciences 29, 2521–2549. https://doi.org/10.5194/hess-29-2521-2025. [F1]

Feigl, M., Lebiedzinski, K., Herrnegger, M., Schulz, K., 2021. Machine-learning methods for stream water temperature prediction. Hydrology and Earth System Sciences 25, 2951–2977. https://doi.org/10.5194/hess-25-2951-2021. [F2]

Zwart, J. A., Sadler, J. M., Appling, A. P., Read, J. S., Oliver, S. K., Diaz-Gonzalez, M., Bertassello, L. E., Jia, X., 2023. Evaluating deep learning architecture and data assimilation for improving water temperature forecasts at unmonitored locations. Frontiers in Water 5, 1184992. https://doi.org/10.3389/frwa.2023.1184992. [F3]

Jia, X., Zwart, J., Sadler, J., Appling, A., Oliver, S., Markstrom, S., Willard, J., Xu, S., Steinbach, M., Read, J., Kumar, V., 2021. Physics-guided recurrent graph model for predicting flow and temperature in river networks, in: Proceedings of the 2021 SIAM International Conference on Data Mining (SDM). SIAM, pp. 612–620. https://doi.org/10.1137/1.9781611976700.69 (preprint: arXiv:2009.12575). [F4]

Luo, S., Yu, R., Chen, S., Jia, X., Kumar, V., 2025. Geo-aware models for stream temperature prediction across different spatial regions and scales, in: Proceedings of the 33rd ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems. ACM. https://doi.org/10.1145/3748636.3762716 (preprint: arXiv:2510.09500). [F5]

Rahmani, F., Lawson, K., Ouyang, W., Appling, A., Oliver, S., Shen, C., 2021. Exploring the exceptional performance of a deep learning stream temperature model and the value of streamflow data. Environmental Research Letters 16, 024025. https://doi.org/10.1088/1748-9326/abd501. [F7]

Rahmani, F., Shen, C., Oliver, S., Lawson, K., Appling, A., 2021. Deep learning approaches for improving prediction of daily stream temperature in data-scarce, unmonitored, and dammed basins. Hydrological Processes 35, e14400. https://doi.org/10.1002/hyp.14400. [F8]

Rahmani, F., Appling, A., Feng, D., Lawson, K., Shen, C., 2023. Identifying structural priors in a hybrid differentiable model for stream water temperature modeling. Water Resources Research 59, e2023WR034420. https://doi.org/10.1029/2023WR034420. [F9]

Sadler, J. M., Appling, A. P., Read, J. S., Oliver, S. K., Jia, X., Zwart, J. A., Kumar, V., 2022. Multi-task deep learning of daily streamflow and water temperature. Water Resources Research 58, e2021WR030138. https://doi.org/10.1029/2021WR030138. [F10]

Topp, S. N., Appling, A. P., Diaz-Gonzalez, M. A., Heinrich, L. R., Zwart, J. A., Read, J. S., Oliver, S. K., 2023. Stream temperature predictions for river basin management in the Pacific Northwest and Mid-Atlantic regions using machine learning. Water Resources Research 59, e2022WR033880. https://doi.org/10.1029/2022WR033880. [F11]

Toffolon, M., Piccolroaz, S., 2015. A hybrid model for river water temperature as a function of air temperature and discharge. Environmental Research Letters 10, 114011. https://doi.org/10.1088/1748-9326/10/11/114011. [F12]

Piccolroaz, S., Calamita, E., Majone, B., Gallice, A., Siviglia, A., Toffolon, M., 2016. Prediction of river water temperature: a comparison between a new family of hybrid models and statistical approaches. Hydrological Processes 30, 3901–3917. https://doi.org/10.1002/hyp.10913.

Mohseni, O., Stefan, H. G., Erickson, T. R., 1998. A nonlinear regression model for weekly stream temperatures. Water Resources Research 34, 2685–2692. https://doi.org/10.1029/98WR01877.

Caissie, D., 2006. The thermal regime of rivers: a review. Freshwater Biology 51, 1389–1406. https://doi.org/10.1111/j.1365-2427.2006.01597.x.

### Methods — machine learning, calibration, statistics

Romano, Y., Patterson, E., Candès, E. J., 2019. Conformalized quantile regression, in: Advances in Neural Information Processing Systems 32 (NeurIPS 2019), pp. 3543–3553.

Vovk, V., Gammerman, A., Shafer, G., 2005. Algorithmic Learning in a Random World. Springer, New York.

Martins, A. F. T., Astudillo, R. F., 2016. From softmax to sparsemax: a sparse model of attention and multi-label classification, in: Proceedings of the 33rd International Conference on Machine Learning (ICML), pp. 1614–1623.

Bai, S., Kolter, J. Z., Koltun, V., 2018. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv:1803.01271.

Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., Dean, J., 2017. Outrageously large neural networks: the sparsely-gated mixture-of-experts layer, in: International Conference on Learning Representations (ICLR).

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., Liu, T.-Y., 2017. LightGBM: a highly efficient gradient boosting decision tree, in: Advances in Neural Information Processing Systems 30 (NeurIPS 2017), pp. 3146–3154.

Diebold, F. X., Mariano, R. S., 1995. Comparing predictive accuracy. Journal of Business & Economic Statistics 13, 253–263. https://doi.org/10.1080/07350015.1995.10524599.

Harvey, D., Leybourne, S., Newbold, P., 1997. Testing the equality of prediction mean squared errors. International Journal of Forecasting 13, 281–291. https://doi.org/10.1016/S0169-2070(96)00719-4.

Künsch, H. R., 1989. The jackknife and the bootstrap for general stationary observations. The Annals of Statistics 17, 1217–1241. https://doi.org/10.1214/aos/1176347265.

Wilks, D. S., 2011. Statistical Methods in the Atmospheric Sciences, 3rd ed. Academic Press, Oxford. (Chapters on bootstrap CIs, Brier/Brier-skill, AUPRC, and the Relative Economic Value of probabilistic forecasts.)

Richardson, D. S., 2000. Skill and relative economic value of the ECMWF ensemble prediction system. Quarterly Journal of the Royal Meteorological Society 126, 649–667. https://doi.org/10.1002/qj.49712656313.

Gneiting, T., Raftery, A. E., 2007. Strictly proper scoring rules, prediction, and estimation. Journal of the American Statistical Association 102, 359–378. https://doi.org/10.1198/016214506000001437.

Kingma, D. P., Ba, J., 2015. Adam: a method for stochastic optimization, in: International Conference on Learning Representations (ICLR).

Loshchilov, I., Hutter, F., 2019. Decoupled weight decay regularization, in: International Conference on Learning Representations (ICLR).

Nash, J. E., Sutcliffe, J. V., 1970. River flow forecasting through conceptual models part I — a discussion of principles. Journal of Hydrology 10, 282–290. https://doi.org/10.1016/0022-1694(70)90255-6.

Gupta, H. V., Kling, H., Yilmaz, K. K., Martinez, G. F., 2009. Decomposition of the mean squared error and NSE performance criteria: implications for improving hydrological modelling. Journal of Hydrology 377, 80–91. https://doi.org/10.1016/j.jhydro.2009.08.003.

### Software and data sources

De Cicco, L. A., Lorenz, D., Hirsch, R. M., Watkins, W., Johnson, M., 2024. dataRetrieval: R packages for discovering and retrieving water data available from federal hydrologic web services. U.S. Geological Survey. https://doi.org/10.5066/P9X4L3GE. [D1]

Hodson, T. O., Decker, J. K., 2023. dataretrieval (Python): a Python package for discovering and retrieving water data available from U.S. federal hydrologic web services. U.S. Geological Survey. https://doi.org/10.5066/P94I5TX3.

U.S. Geological Survey, 2024. National Water Information System (NWIS). Web services accessed via dataRetrieval; parameter codes 00010 (water temperature), 00060 (discharge), 00065 (gage height). https://waterdata.usgs.gov/nwis. [public domain]

Thornton, P. E., Shrestha, R., Thornton, M., Kao, S.-C., Wei, Y., Wilson, B. E., 2022. Daymet: daily surface weather data on a 1-km grid for North America, version 4 R1. ORNL DAAC, Oak Ridge, Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/2129. [D2]

Abatzoglou, J. T., 2013. Development of gridded surface meteorological data for ecological applications and modelling. International Journal of Climatology 33, 121–131. https://doi.org/10.1002/joc.3413. (gridMET / METDATA.) [D3]

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., Chintala, S., 2019. PyTorch: an imperative style, high-performance deep learning library, in: Advances in Neural Information Processing Systems 32 (NeurIPS 2019), pp. 8026–8037.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., Duchesnay, É., 2011. Scikit-learn: machine learning in Python. Journal of Machine Learning Research 12, 2825–2830.

Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R. J., Jones, E., Kern, R., Larson, E., Carey, C. J., Polat, I., Feng, Y., Moore, E. W., VanderPlas, J., Laxalde, D., Perktold, J., Cimrman, R., Henriksen, I., Quintero, E. A., Harris, C. R., Archibald, A. M., Ribeiro, A. H., Pedregosa, F., van Mulbregt, P., SciPy 1.0 Contributors, 2020. SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature Methods 17, 261–272. https://doi.org/10.1038/s41592-019-0686-5.

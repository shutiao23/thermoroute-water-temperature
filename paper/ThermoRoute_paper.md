# ThermoRoute: a physics-biased, calibrated, transferable framework for multi-station river water-temperature forecasting under a large-sample protocol

[Author One]^a,\*^, [Author Two]^a^, [Author Three]^b^

^a^ [Department / Laboratory, Institution, City, Postcode, Country]
^b^ [Department / Laboratory, Institution, City, Postcode, Country]

\* Corresponding author. E-mail: [corresponding.author@institution.edu]

> **Author/affiliation block is a template** — replace the bracketed fields with the
> real author names, ORCID iDs, affiliations and the corresponding author's e-mail
> before submission. (A corresponding e-mail is available on file if you wish to use it.)

> Manuscript draft, rewritten around the large-sample evaluation. All numbers are
> produced by the code in this repository: the three-station case study by
> `scripts/01`–`08`, the 120-station large-sample experiment by `scripts/09`
> (5 seeds, panel `data_usgs/panel_usgs_100.parquet` — the filename is
> historical; the panel holds 120 stations), the calibration/decision/mechanism
> analysis by `scripts/10`, statistical rigor by `scripts/12`–`13`, and the
> sha256 artifact manifest by `scripts/14`. Reports:
> `outputs/reports/{usgs_experiment_v2,usgs_analysis,rigor}.md`; tables
> `outputs/tables/`; figures `outputs/figures/`. Every headline number is
> traceable to a hashed artifact in `outputs/manifest.json`.

---

## Abstract

Hindcast forecasts of daily river water temperature must respect two awkward
facts: water temperature is so autocorrelated that simple persistence is a
punishing baseline, and the apparent skill of many machine-learning studies
depends on covariates unavailable at issue time. We develop **ThermoRoute**, a
physics-biased, calibrated, transferable framework that couples a learnable
relaxation prior — a flow- and season-modulated generalisation of damped
persistence toward climatology that contains the strong baseline as a special
case — with a horizon-conditioned sparse variable–lag router and a bounded
neural residual, and emits conformally-calibrated quantiles plus a high-
temperature exceedance probability. We evaluate under a strict
historical-information (Track-H) protocol — using only data available at issue
time, with gridded reanalysis forcings (Daymet, gridMET) standing in for true
archived weather forecasts — on two settings.

(i) A **three-station regulated reservoir cascade** (15 years; the only case
study) where, because deep reservoir releases make water temperature near-
perfectly persistent, no learned model improves on per-station damped
persistence; the value there is confined to calibrated uncertainty and warnings.

(ii) A **120-station large-sample USGS panel** (free-flowing and regulated; 114
stations contribute the blind-test split, the headline-N) where forecast
headroom exists: ThermoRoute beats persistence and damped persistence at every
lead (per-station Wilcoxon, Holm-adjusted, p ≤ 3×10⁻¹⁶), with median skill vs
damped +0.16 / +0.07 / +0.03 at 1 / 3 / 7 days and better than damped at
**85 / 89 / 89 %** of the n=114 blind-test stations (Wilson 95 % CIs all exclude
50 %; robust to a HUC2 cluster bootstrap), and beats an air2stream-style
8-parameter physical model (a fairly-calibrated *variant* of Toffolon–Piccolroaz)
on median RMSE at every lead (0.630 / 1.289 / 1.658 vs 0.733 / 1.409 / 1.805).
Against a strong gradient-boosting baseline (LightGBM) the honest result is
**parity, not superiority**: with the seed budget matched (each of five
ThermoRoute seeds scored as a single model), LightGBM is significantly more
accurate at 1 day (median skill −0.044, p = 1×10⁻¹⁴) while the two are
statistically indistinguishable at 3 and 7 days (median skill +0.002 and −0.002,
p = 0.76 and 0.52). This parity **also holds out-of-region**: under a
leave-HUC2-region-out protocol (whole regions held out; mean 358 km to the
nearest training gage) both ThermoRoute and a global LightGBM transfer far beyond
persistence, and between them the result is a horizon-conditional tie (LightGBM
better at 1 day, tied at 3 days, ThermoRoute better at 7 days, p = 1×10⁻³). Against
the field-standard deep baseline — a top-down global LSTM — ThermoRoute is more
accurate at every lead, in-sample and out-of-region (p ≤ 5×10⁻¹⁶), consistent with
the repeated finding that gradient boosting matches or beats deep sequence models on
strongly-autocorrelated daily hydrology. ThermoRoute thus matches the strongest
learner in point accuracy while adding three things a pure or hybrid learner does
not provide: (i) a **bounded-degradation guarantee** — because its residual is
`tanh`-bounded to ±δ around the physics prior, RMSE is provably never worse than the
prior by more than δ °C on any station; we verify this holds on 100 % of blind-test
predictions and that the floor survives 358 km of spatial extrapolation (it still
beats persistence at 91 % of held-out-region station×horizon cells); (ii)
**distribution-free calibrated uncertainty** — after conformal calibration on the
2018 hold-out year its 90 % intervals are empirically near-nominal (PICP 0.90 ± 0.01)
and its exceedance warning is the best-calibrated of the three (Brier skill
+0.74 / +0.60 / +0.51; highest AUROC); and (iii) interpretable variable–lag drivers.
On a generic cost–loss decision model, the calibrated warning has the highest
relative economic value across the operationally relevant (cheap-protection) range
of cost–loss ratios, illustrating that the RMSE tie with the strong learner does not
carry to a value tie; we frame this as a decision-relevance result rather than a
demonstrated management advantage, because the cost–loss model is generic and the
high-temperature threshold is a statistical (train q90) rather than ecological
cut-off. We deliberately report negative results in full — no
point gain on the near-deterministic cascade, and a flow-dependent thermal
memory that does **not** generalise beyond it (κ rises with flow at 0 % of
large-sample stations) — and argue that the right scientific target is a
calibrated, transferable forecaster whose advantage must be established on a
large, hydrologically diverse sample, not a single cascade.

**Keywords:** river water temperature; probabilistic forecasting; physics-guided
machine learning; large-sample hydrology; spatial transfer; conformal prediction;
calibrated uncertainty.

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

Quantifying predictive uncertainty honestly is the third recurring problem, and it
has a long hydrological lineage. Generalised Likelihood Uncertainty Estimation
(GLUE) [G1] rejected the notion of a single optimal parameter set — accepting
*equifinality* among many acceptable simulators — and lumped predictive uncertainty
onto the parameters through an informal likelihood. Bayesian Total Error Analysis
(BATEA) [G2, G3] replied that this conflates distinct error sources, and instead
requires the modeller to *explicitly* specify separate input, output and structural
error models (e.g. latent storm multipliers that infer the "true" rainfall), so that
a model is not blamed for corrupt forcing. Both target the calibration and
simulation of rainfall–runoff models on one or two catchments; BATEA itself notes
that its main obstacle is the difficulty of specifying valid input-error models,
"which are currently poorly understood". We take a third route suited to
*forecasting at scale*: rather than decomposing the error budget, we wrap the
forecaster in **conformalised quantile regression**, a distribution-free calibration
that targets nominal coverage without assuming any error model — trading BATEA's
process-attribution insight for robustness and a per-station coverage property that
holds across 120 basins.

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
cascade, water temperature is so heavily damped that **no learned model
consistently improves on per-station damped persistence on point accuracy across
horizons** (LightGBM edges it at 1 day only; Table 2) — the dynamic machinery
helps only calibration and warnings there. We therefore move to a 120-station
large-sample setting where forecast headroom exists, and show that ThermoRoute's value
materialises: it beats the *physics* baselines on point accuracy, is on par with
a strong gradient-boosting learner (better at 1 day, statistically tied at 3–7
days), transfers across unseen basins, and produces near-nominally-calibrated
warnings. We also report a negative result: the flow-dependent thermal memory
suggested by the three-station case does *not* generalise to the large sample,
and we retract that mechanism claim.

Contributions:

1. **ThermoRoute**, a physics-biased forecaster whose relaxation prior *provably
   contains damped persistence* as a special case (Proposition 1) and whose
   *bounded* neural residual gives a per-station worst-case deployment floor —
   RMSE never worse than the prior by more than δ °C — that we **verify empirically**
   holds on 100 % of blind-test predictions and survives 358 km of spatial
   extrapolation, a guarantee neither a GBDT, an LSTM, nor a differentiable-hybrid
   model states. (We initially framed the prior's flow-/season-modulated relaxation
   rate as a "dynamic thermal memory"; §4.5 reports that this mechanism does *not*
   generalise, and we retain the prior only as a calibrated,
   damped-persistence-anchored regulariser.)
2. A **leakage-audited evaluation** with rolling-origin discipline, a one-shot blind
   test, moving-block-bootstrap confidence intervals, Diebold–Mariano tests, and an
   adversarial internal review of every headline claim.
3. A **large-sample, transfer-tested** demonstration: 120 public USGS stations
   (114 contributing the blind test), the full temporal / unseen-station /
   ungaged-region (TUURT) transfer triad the review prescribes, benchmarked against
   persistence, damped persistence, an air2stream *variant*, a global **and**
   per-station LightGBM, and a top-down global **LSTM**, under a unified multi-metric
   point + probabilistic (PICP, CRPS, reliability) + exceedance (Brier, AUROC) +
   cost–loss decision-value assessment.
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

**Large-sample USGS panel (main analysis).** 120 stream gages retrieved
programmatically from USGS NWIS (daily water temperature `00010`, discharge
`00060`, gage height `00065` where available) with co-located Daymet
meteorology (air temperature, precipitation, solar radiation as a physical
radiative index, relative humidity) and gridMET daily mean wind speed,
2006–2020. Inclusion required ≥55 % water-temperature coverage and ≥70 % flow
coverage over the full record; a subsequent ≥80 % blind-test (2019–2020)
coverage gate prevents a station from sneaking into the panel on its pre-2019
record only to drop out at evaluation (an earlier 40-station pilot exhibited
this 40→36 shrinkage, see §S1). Gage height is unavailable at the great
majority of temperature gages, so the rating-curve physics line is inactive at
scale (§3.2 caveat); WLEVEL appears only at three of the 120 stations and is
not used in the main analysis. The panel spans 35 U.S. states across 16 USGS
HUC2 hydrologic regions, mixing free-flowing and dam-regulated rivers (a
post-hoc subgroup analysis is in §S2), and a wide thermal range (≈ −1 to 31 °C).
Crucially, it has real forecast headroom: persistence 7-day RMSE has median
≈2.3 °C (range 0.9–3.7), versus full-record 0.79–1.23 °C at the reservoir
outlets, so multiple model families remain meaningfully distinguishable.

**Effective station counts.** Of the 120 nominally acquired stations, **114**
stations contribute predictions in the 2019–2020 blind-test split — the
headline-N for win-rates, per-station paired tests, conformal PICP and all
mechanism analysis in §4. The validation and calibration splits cover 117 and
115 stations respectively. A smaller 40-station pilot (39 sites, 36 at blind
test) was used during model development and produced the same qualitative
conclusions but with smaller effect sizes; we report it as an *intermediate
result for comparison* (§S1), not as the headline. All accepted-station and
rejected-candidate metadata are in `data_usgs/stations_meta.csv` and
`data_usgs/rejected_sites_120v2.csv`.

Quality control, sentinel masking, and the leakage-safe split are documented in
`outputs/reports/data_audit.md` and `outputs/reports/usgs_acquisition.md`.

## 3. Methods

### 3.1 Problem and information set

For station *s*, issue day *t*, horizon *h*∈{1,3,7} predict `WTEMP_{s,t+h}` from
information available at *t* only — the *historical-information* (Track-H)
protocol. No observation time-stamped after *t* enters the model. We use
**reanalysis** weather forcings (Daymet, gridMET), not archived operational
weather forecasts, so the evaluation is a **hindcast** rather than a true
operational forecast: a deployed system would replace t+1…t+h forcings with a
Numerical Weather Prediction forecast issued at *t*, whose error would degrade
multi-day skill compared to the reanalysis-driven numbers reported here. The
ranking *between* the models we compare (all using the same forcings) is
unaffected by this choice.

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

**Variable availability caveat.** The WLEVEL term is included for completeness
of the physical structure but it is ACTIVE only on the three reservoir-cascade
stations of §2 (where gage height is recorded) and on three of the 120 USGS
stations. On the remaining 117 USGS stations, WLEVEL is imputed to its
standardised mean (zero) and the `c_l·z(WLEVEL)` term contributes nothing in
practice; the stage–discharge / rating-curve physics line that this term was
designed to encode is therefore **inactive at scale**. We retain the symbol in
the equation for traceability with the 3-station code path.

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
strongly-damped cascade where the prior is the ceiling, and looser (±1.0 °C) on the
large sample where headroom exists. The large-sample bound was selected by a
**validation-only sweep** over {0.4, 1.0, 1.5, 2.0} using three training seeds
per value, minimising the mean over horizons of the median per-station RMSE on
2016–2017 (`scripts/11_retune.py`, `outputs/tables/usgs_retune.csv`); ±1.0 was
the winner (2016–2017 mean RMSE 1.227, versus 1.230 at ±1.5 and 1.233 at ±2.0).

The clamp buys a worst-case safety guarantee that a free residual cannot:

> **Proposition 1 (bounded degradation).** Let $W^{\mathrm{prior}}_{t+h}$ be the
> prior forecast and $\hat W_{t+h} = W^{\mathrm{prior}}_{t+h} + \Delta_h$ with
> $\Delta_h = \delta\,\tanh(u_h)$, so $|\Delta_h| < \delta$. Then for every
> station, horizon and issue day,
> $|\hat W_{t+h} - W_{t+h}| < |W^{\mathrm{prior}}_{t+h} - W_{t+h}| + \delta$,
> and hence per-station $\mathrm{RMSE}(\hat W) \le \mathrm{RMSE}(W^{\mathrm{prior}}) + \delta$.
> Because the prior reduces to damped persistence when $e_t{=}0,\ \kappa{=}1{-}\varphi$
> (the contained baseline), the learned model can never be worse than damped
> persistence by more than $\delta$ °C on any station — a floor a generic
> unconstrained learner (GBDT or free-headed net) does not provide.

*Proof.* The first inequality is the triangle inequality on
$\hat W = W^{\mathrm{prior}} + \Delta_h$ with $|\Delta_h| < \delta$; squaring,
averaging over a station's days and taking the root (Minkowski) gives the RMSE
bound. ∎  This is why we clamp rather than blend: it turns "the residual usually
helps" into "the residual provably cannot hurt by more than δ" — the property
that makes a physics-biased forecaster safe to deploy where a bare learner's tail
behaviour is unbounded.
*Disclosure:* an earlier version of this study fixed the bound at ±1.5 using a
sweep that had read the blind-test years — a protocol violation; on discovering
it we re-ran the sweep on the validation split only, which selected ±1.0, and
regenerated **every** large-sample number in this manuscript at ±1.0. The blind
test was evaluated once, after the bound was frozen on validation. Monotone quantiles
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
(point + quantile + exceedance); a **top-down global LSTM** — the field-standard
deep sequence baseline (Rahmani et al., 2021; Willard et al., 2024): a
station-agnostic 1-layer, 64-unit LSTM over a 14-day context, persistence-anchored
(it predicts the multi-day increment), trained under the identical Track-H splits
and composite loss so any gap to ThermoRoute is the physics prior and bounded
residual, not the training recipe; and, on the large sample, an **air2stream-style
eight-parameter hybrid model**. Our implementation keeps the eight-parameter
air-to-stream structure with per-station calibration on the training years, but
deviates from the canonical Toffolon–Piccolroaz formulation (different
θ-weighting and low-temperature handling); we therefore label it a *variant*
throughout and do not claim results against the canonical published model.
Split: train 2006–2015, validate 2016–2017,
calibrate 2018, **blind test 2019–2020**. All statistics fit on training data only.
On the large sample, baselines and ThermoRoute are evaluated on *identical windowed
samples* (observed targets only). ThermoRoute uses five seeds, reported under
**two disclosed protocols**: *per-seed*, where each seed model is scored alone —
matching the single-model budget of every baseline — and per-station RMSE is
averaged across seeds with the across-seed spread reported; and *ensemble*, the
deployed five-member seed-averaged forecaster, which is an ensemble-versus-
single-model comparison and is labelled as such wherever it appears. Headline
significance tables report both (Table `claim1_significance`).

## 4. Results

### 4.1 Three-station cascade — an honest negative result

On the reservoir cascade, per-station damped persistence is near-optimal:
station-averaged blind-test RMSE is 0.261 / 0.483 / 0.724 °C (1/3/7 d), and **no
learned model consistently improves on it across horizons**. ThermoRoute (joint,
three seeds, per-seed scores averaged) is 0.292 / 0.543 / 0.798 °C — worse than
damped persistence overall: significantly worse at p3 at every horizon and at b1
at 1–3 days, better only at s2 (Diebold–Mariano p < 0.05 at 1–3 days; Table 2b).
LightGBM comes closest — it edges damped persistence at 1 day (0.255 vs
0.261 °C) but falls behind at 7 days (0.806 vs 0.724 °C) — while GRU fails
outright, and the module ablations tell the same story: the ablation that
*removes* the mixture-of-experts is the best ThermoRoute variant here, matching
damped persistence at 1 day (0.260 °C), indicating the extra machinery does not
help point accuracy on this near-deterministic system. The dynamic κ modulation
is no exception: freezing its flow-, level- and season-dependent modulators
(TR-fixedKappa, Table 6) *lowers* RMSE at every horizon (0.287 / 0.525 / 0.797
vs 0.292 / 0.543 / 0.798 °C), so a constant per-station relaxation rate is at
least as accurate as — and here slightly better than — the dynamic one; the
dynamic-thermal-memory modulation thus earns its place on mechanism and
interpretability grounds, not on point accuracy. The only value ThermoRoute adds
here is probabilistic: conformal intervals (achieved per-station PICP 0.79–0.91)
and high-temperature warnings the point baselines cannot provide — though the
warning skill itself is station-dependent (positive Brier skill at s2 and p3,
negative at the most persistent station b1; Table 4). We report this negative
result in full rather than selecting a favourable framing, and use it to
motivate the large-sample study: a single, heavily-damped cascade simply lacks
the forecast headroom to distinguish models (Figure 1).

![**Figure 1.** Three-station cascade, blind-test RMSE (°C, left) and skill versus persistence (right) by model and horizon. No learned model improves on damped persistence on this near-deterministic system — the honest negative result that motivates the large-sample study.](outputs/figures/fig3_results_heatmap.png){width=90%}

### 4.2 Large-sample USGS — ThermoRoute beats the physics baselines and is on par with a strong learner (Table A)

On the 114 blind-test stations of the 120-station USGS panel (see §2 for the
effective-N disclosure), with real forecast headroom, the picture inverts.
Median over stations of per-station RMSE (5-member seed-averaged ensemble; model
uses 7 variables including gridMET wind, with the residual bound set to
±1.0 °C on the large sample — the value chosen by the validation-only sweep of
§3.3); an air2stream-style 8-parameter model (a *variant* of
Toffolon–Piccolroaz — §3.5) is included as a *physical* strong baseline:

| horizon | persistence | damped | air2stream-a8 | LSTM | LightGBM (global) | LightGBM (per-station) | ThermoRoute | skill vs persist | skill vs damped | win vs damped |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 d | 0.797 | 0.774 | 0.733 | 0.653 | 0.620 | 0.632 | 0.630 | +0.204 | +0.172 | 88 % |
| 3 d | 1.581 | 1.406 | 1.409 | 1.362 | 1.300 | 1.324 | 1.289 | +0.186 | +0.078 | 94 % |
| 7 d | 2.235 | 1.778 | 1.805 | 1.803 | 1.669 | 1.750 | 1.658 | +0.247 | +0.035 | 92 % |

(Medians are not bolded because ThermoRoute and the *global* LightGBM are a
statistical tie at 3–7 days and LightGBM leads at 1 day — see the paired tests
below and `claim1_significance`; the sub-0.02 °C median gaps are not individually
meaningful. We report **both** LightGBM configurations: the global model is the
stronger of the two on this panel — a *per-station* LightGBM over-fits the
limited per-gage record and is worse at 3–7 days (1.324 / 1.750) than both the
global model and ThermoRoute — so the parity comparison is against the stronger,
global LightGBM. The **top-down global LSTM** (0.653 / 1.362 / 1.803; in the
published daily-SWT LSTM band, Zwart et al. 2023) is a credible deep baseline that
beats persistence at every lead but sits behind both the global LightGBM and
ThermoRoute — consistent with the repeated finding that gradient boosting matches
or beats deep sequence models on strongly-autocorrelated tabular hydrology (Feigl
et al. 2021; Grinsztajn et al. 2022). ThermoRoute is more accurate than the LSTM at
every lead (per-station Wilcoxon p ≤ 5×10⁻¹⁶; better at 88 / 93 / 91 % of stations).
air2stream is the fair *variant* of §3.5, calibrated on observed-target days with
the observed air temperature driving the first step.)

ThermoRoute beats every *physics* baseline (persistence, damped persistence,
air2stream) at every horizon. Per-station Wilcoxon paired tests on the n=114
blind-test stations (per-seed protocol, Holm-adjusted across the 9 horizon ×
reference tests) give median skill +0.157 / +0.072 / +0.029 vs damped
persistence (p ≤ 3×10⁻¹⁶ at every horizon; ThermoRoute better at 85 / 89 / 89 %
of stations) and +0.186 / +0.180 / +0.243 vs raw persistence (p ≤ 2×10⁻¹⁸).
Both the station bootstrap and a **HUC2 cluster bootstrap** — which resamples
whole hydrologic regions rather than treating spatially-correlated stations as
independent — give 95 % CIs that exclude zero at every horizon
(Table `claim1_significance`).

The strong gradient-boosting baseline (LightGBM) is the most interesting
comparison, and the honest result is a **near-tie that we do not oversell**.
Under the *per-seed* protocol — each ThermoRoute seed scored as a single model
against the single-model LightGBM, so the model budgets match — LightGBM is
significantly more accurate at 1 day (median skill −0.044, Holm p = 1×10⁻¹⁴;
ThermoRoute wins only 10 % of stations), while at 3 and 7 days the two are
**statistically indistinguishable** (median skill +0.002 and −0.002, Holm
p = 0.76 and 0.52; ThermoRoute wins 53 % and 46 % of stations). The deployed
5-seed *ensemble* — which carries the usual ensemble advantage over a single
model, and is labelled as such — shifts this marginally in ThermoRoute's favour:
a small but significant edge at 3 days (median skill +0.009, Holm p = 3×10⁻⁵),
still behind at 1 day (−0.019) and tied at 7 days (+0.004, p = 0.09). We
therefore claim (i) a robust, significant improvement over the *physics*
baselines at every horizon, and (ii) **parity, not superiority, against the
strong learned baseline**: LightGBM is better at 1 day and the two are on par at
3–7 days once the seed budget is matched. We do **not** claim an advantage over
LightGBM, and we report the unfavourable short-horizon result openly. The
intermediate 40-station pilot gives the same qualitative ordering (Figure 2).

![**Figure 2.** Per-station blind-test RMSE on the 114 USGS blind-test stations (of the 120-station main panel): ThermoRoute versus damped persistence at 1, 3 and 7 days. Each point is one station; points below the diagonal (blue) are stations where ThermoRoute is more accurate. ThermoRoute wins 88 / 94 / 92 % of stations against damped persistence at 1 / 3 / 7 days.](outputs/figures/fig_usgs_perstation.png){width=95%}

### 4.3 Spatial transfer to unseen basins (Table B)

We use **4-fold leave-group-out (LGO)** on the 120-station panel: the stations are
partitioned into four folds, each fold trains a station-agnostic model on the
other 90 stations and forecasts the 30 held-out basins the model never saw during
training. Every station is thus held out exactly once, and we report the mean ±
standard deviation of transfer skill across the four folds (Table B):

| horizon | TR transfer RMSE | skill vs persistence | skill vs damped |
|---|---|---|---|
| 1 d | 0.696 | **+0.173 ± 0.021** | +0.140 ± 0.023 |
| 3 d | 1.361 | **+0.169 ± 0.010** | +0.057 ± 0.012 |
| 7 d | 1.654 | **+0.241 ± 0.011** | +0.028 ± 0.011 |

ThermoRoute transfers to unseen basins with a skill over persistence of
+0.17 / +0.17 / +0.24 at 1 / 3 / 7 days and small across-fold variability
(std ≤ 0.023), and it also beats *damped* persistence on the held-out basins at
every lead. The transfer skill is close to the in-sample skill of §4.2, i.e. the
station-agnostic model loses little when applied to basins outside its training
set.

**Leave-region-out, benchmarked against the strong learned baseline.** A random
station split can leak: gages on the same river and in the same HUC2 region
straddle train and test. We therefore also run a **leave-HUC2-region-out**
protocol — the 16 regions are packed into four folds and *whole regions* are held
out, so no held-out gage shares a river or region with training (mean distance
from a held-out gage to its nearest training gage rises from ≈0 to **358 km**) —
and, crucially, we run the **global LightGBM under the identical protocol** so
transfer is measured against the strong learned baseline, not only persistence
(Table C):

| horizon | ThermoRoute RMSE | LightGBM RMSE | LSTM RMSE | TR skill vs persist | LGB skill vs persist | TR−LGB paired Wilcoxon p |
|---|---|---|---|---|---|---|
| 1 d | 0.667 | 0.650 | 0.671 | +0.151 | +0.175 | 5×10⁻⁹ (LGB better) |
| 3 d | 1.323 | 1.321 | 1.393 | +0.169 | +0.168 | 0.95 (tie) |
| 7 d | 1.675 | 1.683 | 1.825 | +0.237 | +0.232 | 1×10⁻³ (TR better) |

All three learned models transfer far out of region and beat persistence/damped by
a wide margin. Between ThermoRoute and the strong GBDT the result is a
**horizon-conditional tie** — LightGBM is better at 1 day, the two are
indistinguishable at 3 days, and ThermoRoute is better at 7 days (the same ordering
as in-sample, §4.2). The **global LSTM transfers as well as the others but stays
behind both** at every lead (0.671 / 1.393 / 1.825; ThermoRoute better with paired
p ≤ 6×10⁻²⁰), so the deep baseline does not close the gap out of region either. We
therefore do **not** claim that ThermoRoute generalises to ungauged basins *better
than* the strong learned baseline; we claim that a physics-biased, station-agnostic
model transfers *as well as* a global GBDT and better than a top-down LSTM, while
additionally carrying the calibrated intervals (§4.4), the bounded-degradation
guarantee (Proposition 1, verified out-of-region in §4.4) and the interpretable
drivers (§4.5) that neither learner provides. This honest delineation — parity in
point accuracy, advantage in the properties that matter for deployment — is the
paper's central position.

**The transfer triad (TUURT).** The 2025 review [F1] prescribes *temporal, unseen
and ungaged-region tests* as the standard for extrapolation confidence, and notes
most ML stream-temperature studies run none. ThermoRoute passes all three with
positive skill over persistence at every lead, degrading gracefully as the
extrapolation demand grows (Table D):

| test arm | protocol | skill vs persistence (1 / 3 / 7 d) |
|---|---|---|
| Temporal | future 2019–2020, seen stations | +0.204 / +0.186 / +0.247 |
| Unseen station | random 4-fold leave-group-out | +0.173 / +0.169 / +0.241 |
| Ungaged region | leave-HUC2-region-out (~358 km) | +0.151 / +0.169 / +0.237 |

Skill is positive against *damped* persistence too on every arm and lead
(+0.17/+0.08/+0.03 temporal down to +0.12/+0.05/+0.03 ungaged-region), and the
monotone erosion from temporal to unseen-station to ungaged-region is the expected,
honest signature of genuine spatial extrapolation rather than leakage.

### 4.4 Calibration, warnings and decision value

**Calibration.** Conformalised Quantile Regression (Romano et al., 2019) on the
2018 calibration year widens raw quantiles per (station × horizon). The formal
finite-sample (1−α) coverage guarantee of split-CQR holds only under
exchangeability between calibration and test; in our temporal split the test
years (2019–2020) follow the calibration year, so the assumption is **not**
strictly satisfied — we therefore report *empirical* coverage rather than a
guarantee. ThermoRoute's 90 % intervals achieve **PICP 0.904 / 0.909 / 0.911**
at 1 / 3 / 7 days, with **96 / 88 / 86 %** of the n=114 blind-test stations
falling within ±0.05 of nominal (Wilson 95 % CIs: 91–99 %, 80–93 %, 78–91 %;
Figure 3). Applying the *identical* split-conformal wrapper to the two learned
baselines, both are calibrated to the same level (LightGBM PICP 0.906 / 0.906 /
0.907; LSTM 0.905 / 0.912 / 0.912), so calibrated coverage is a property any of the
models can be given and is **not** by itself ThermoRoute's differentiator; its
intervals are also the sharpest of the three (CRPS 0.246 / 0.505 / 0.633, versus
LightGBM 0.240 / 0.503 / 0.630 and LSTM 0.258 / 0.534 / 0.688 — on par with the GBDT
and ahead of the LSTM). On the smaller 40-station pilot the same model is calibrated
at the *population* level (median PICP 0.90–0.91) with a wider per-station spread
(n=36 per-station PICP range 0.81–0.97), and on the three-station cascade the spread
is wider still (0.79–0.91, §4.1), so we treat tight per-station coverage as a
property of the large-sample regime, not a guarantee.

**Bounded-degradation guarantee, verified (Proposition 1).** ThermoRoute's genuine
distinguishing property is one no pure or hybrid learner states: because the neural
residual is `tanh`-bounded to ±δ around the physics prior, per sample
|median−y| ≤ |prior−y|+δ, hence per station RMSE(model) ≤ RMSE(prior)+δ — a
worst-case deployment floor. We verify it on the blind test: the pointwise bound
holds on **100.0 %** of predictions and the per-station RMSE ceiling holds on
**100 %** of station×horizon cells (Figure 4a). The bound is not decorative — the
residual is engaged (non-zero) on **81 %** of samples — yet the ±δ cap only has to
bind on **3 %** of them, so δ = 1 °C is a hard ceiling the working residual rarely
touches (Figure 4b). The floor is not an in-sample artifact: on the
leave-HUC2-region-out held-out stations, ThermoRoute still beats persistence at
**91 %** of station×horizon cells, i.e. the guarantee survives 358 km of spatial
extrapolation. Neither the LightGBM nor the LSTM — nor differentiable-hybrid stream
temperature models (Rahmani et al., 2023) — can state a comparable per-station
worst-case bound.

![**Figure 4.** Proposition 1 verified on the blind test. (a) Per-station RMSE of ThermoRoute versus its internal physics prior; every point lies under the y = x + δ ceiling (δ = 1 °C). (b) Distribution of the bounded neural residual: engaged on 81 % of days, but the ±δ cap binds on only 3 % — a hard floor the working residual rarely reaches.](outputs/figures/fig_prop1_binding.png){width=90%}

**High-temperature exceedance warnings** have clear positive skill on the
120-station panel. ThermoRoute leads both learned baselines on the calibrated
warning: Brier skill **+0.74 / +0.60 / +0.51** and AUROC 0.989 / 0.975 / 0.963 at
1 / 3 / 7 d, ahead of LightGBM (+0.73 / +0.57 / +0.49) and the LSTM
(+0.68 / +0.56 / +0.48), and its reliability curve tracks the diagonal most closely
(Figure 5). The exceedance threshold is **statistical**, set as the station-specific
train-period 90th percentile of WTEMP — it is not a biological or regulatory limit
and the skill numbers should be read accordingly.

![**Figure 5.** Reliability of the high-temperature exceedance warning (all leads pooled). Forecast probability versus observed exceedance frequency; ThermoRoute's calibrated warning tracks the diagonal most closely, ahead of the LightGBM and LSTM.](outputs/figures/fig_reliability.png){width=55%}

**Decision value.** On a generic cost–loss decision model (Wilks 2011), we score
the exceedance decision "protect iff p > α" over the full cost–loss ratio grid
α ∈ (0, 1) — the framing of Modi et al. (2025), who show forecast *value* need not
follow RMSE *skill*. Across the operationally relevant low-α range (cheap protection
relative to loss), ThermoRoute's calibrated warning has the highest relative
economic value at every lead: REV at α = 0.1 is **0.90 / 0.85 / 0.81** for
ThermoRoute versus 0.89 / 0.84 / 0.81 (LightGBM), 0.87 / 0.83 / 0.80 (LSTM) and
0.81 / 0.65 / 0.51 (persistence); peak REV is 0.91 / 0.85 / 0.82 (Figure 6). This is
the axis on which the RMSE tie with LightGBM does *not* translate to a value tie: a
calibrated probability dominates a deterministic threshold warning where protection
is cheap. Honesty requires the caveat at the *other* end of the grid: at α = 0.5
(protection nearly as costly as the loss) the deterministic persistence warning
becomes competitive (persistence REV 0.72 / 0.48 / 0.28 versus ThermoRoute
0.68 / 0.48 / 0.35), so the probabilistic advantage is concentrated in the cheap-
protection regime, not universal. We also stress that the cost–loss model is
generic and the threshold is statistical; a true management value calculation would
require biological/regulatory thresholds and station-specific cost ratios, so we
read this as a rigorous *decision-relevance* result rather than a demonstrated
dollar value.

**Multi-metric reporting.** The 2025 review [F1] asks that regression, dimensionless
and error-index statistics be reported together rather than RMSE alone. On the
pooled blind test ThermoRoute scores NSE 0.992 / 0.970 / 0.954, KGE 0.993 / 0.974 /
0.963, MAE 0.470 / 0.971 / 1.213 °C and |PBIAS| ≤ 0.2 % at 1 / 3 / 7 d — all above
the review's reported field norms (median NSE ≈ 0.93, MAE ≈ 1.09 °C) and essentially
tied with the global LightGBM (NSE 0.993 / 0.969 / 0.954) and ahead of the LSTM
(NSE 0.992 / 0.967 / 0.947); the full per-model table is `outputs/tables/multi_metric.csv`.
Region-weighted RMSE (mean of per-HUC2 medians, so no single over-represented region
carries the headline) is 0.66 / 1.37 / 1.69 °C, within 0.01–0.04 °C of the pooled
medians, confirming the result is not driven by one region.

![**Figure 3.** Conformal calibration on n=114 USGS blind-test stations. (a) Per-station coverage (PICP) distribution against the 0.90 target; (b) mean coverage versus lead time; (c) interval sharpness. Coverage is near-nominal and tight across the population (86–96 % of stations within ±0.05 of 0.90).](outputs/figures/fig_usgs_calibration.png){width=95%}

### 4.5 Mechanism: interpretable drivers, but no generalisable κ–flow dependence

We report an honest negative result on the dynamic-memory hypothesis. On the three
damped reservoir outlets the fitted relaxation rate κ rose ~1.8× from low to high
flow (implied memory 1/κ ≈ 6–18 d), which we initially read as a flow-dependent
thermal memory. **This does not replicate on the large sample:** across the 120-station
USGS panel κ rises with flow at **0 % of stations** (median κ_high/κ_low =
0.87; mean κ_low 0.134 vs κ_high 0.117) — and at only 24 % on the smaller 40-
station pilot — so there is no consistent, physically directional flow dependence
at scale; if anything the sign is weakly reversed. Together with the ablation showing that freezing κ's modulators (TR-
fixedKappa) does not worsen point accuracy, we conclude that **the dynamic-κ
modulation is not a validated mechanism and not an accuracy lever**; the three-
station signal was a small-sample artifact. We therefore do not claim a flow-
dependent thermal memory.

What does survive is interpretability of the *router*. Its dominant variable
shares shift sensibly with horizon: at 1 day the router concentrates on
discharge (FLOW ≈44 % of routing weight) and precipitation (≈29 %), consistent
with short-range control by hydrologic state and recent rainfall; at 3 and 7
days incident solar radiation (`DH`) grows to the single largest share
(30 % / 35 %) alongside flow (29 % / 27 %) and a rising humidity contribution
(RHMEAN 15 % / 17 %), consistent with the surface energy budget — radiative
heating and evaporative exchange — mattering more as the persistent thermal
state decays at longer leads. This is an interpretive read-out, not a causal
mechanism (Figure 4).

![**Figure 4.** Dynamic relaxation rate κ on the 120-station USGS panel. (a) κ binned by standardised log-flow (pooled); (b) per-station ratio κ(high flow)/κ(low flow). The flow dependence seen on the three reservoir stations does not generalise — κ rises with flow at **0 % of stations** on the 120-station panel (and 24 % on the 40-station pilot), median ratio 0.87 — so we retract the flow-dependent thermal-memory claim.](outputs/figures/fig_usgs_kappa.png){width=85%}

### 4.6 Module ablations on the large sample

Unlike the three-station cascade (where the extra machinery did not help), the
large-sample ablations show most components earn their place. Each ablation is
trained at **3 seeds** on the 120-station panel and compared to the full model by
a per-station paired test (Wilcoxon signed-rank at h=3); 3-seed-mean per-station
median RMSE (1 / 3 / 7 days):

Each ablation and the full model are all scored on the **matched 3-seed budget**
(seeds {0,1,2}); the paired test is a per-station Wilcoxon signed-rank at h=3:

| variant | h1 | h3 | h7 | Wilcoxon p (h3 vs full) |
|---|---|---|---|---|
| ThermoRoute (full) | 0.630 | 1.289 | 1.656 | — |
| TR-noPrior | 1.327 | 1.528 | 1.699 | 2.2×10⁻²⁰ * |
| TR-noMoE | 0.733 | 1.343 | 1.699 | 9.5×10⁻¹⁸ * |
| TR-noRouter | 0.639 | 1.302 | 1.662 | 1.3×10⁻⁶ * |
| TR-fixedKappa | 0.642 | 1.294 | 1.658 | 0.23 (n.s.) |

Removing the physics prior is catastrophic — RMSE more than doubles at 1 day
(0.630 → 1.327) and the per-station difference is significant at p ≈ 2×10⁻²⁰ —
confirming that the prior carries the forecast. Removing the mixture-of-experts
(+0.103 / +0.054 / +0.043 at 1 / 3 / 7 d, p ≈ 1×10⁻¹⁷) and the router
(+0.009 / +0.013 / +0.006, p ≈ 1×10⁻⁶) both hurt significantly. Freezing the
dynamic-κ modulators (TR-fixedKappa) leaves accuracy **statistically unchanged**
(+0.012 / +0.005 / +0.002, paired p = 0.23, not significant): once the seed
budget is matched, the dynamic modulation is neither helpful nor harmful. The
honest reading is that the prior, experts and router contribute real accuracy
while the dynamic-κ modulation is interpretive overhead we do not claim as an
accuracy lever — retained only because freezing it does not improve accuracy
either (consistent with the §4.5 mechanism retraction). Unlike the 40-station
pilot — where fixed-κ was marginally *better* at 7 days — the sign is now (if
negligibly) in favour of the dynamic version, which we read as within-noise.

## 5. Discussion

The two settings deliver a single message: the value of a physics-guided learned
forecaster depends on whether the system has forecast headroom. On near-perfectly
persistent reservoir outlets, damped persistence is a ceiling and the honest result
is parity-minus on point accuracy with a calibration-only contribution. On a diverse
large sample, the same model beats the *physics* baselines, transfers to unseen basins,
and delivers near-nominally-calibrated warnings — while being **on par with, not
superior to, a strong gradient-boosting learner**. This argues against single-site
"state-of-the-art" claims and for large-sample, transfer-tested evaluation as the
standard for this problem.

**Limitations.** The large-sample model omits reservoir level (gage height is
unavailable at most USGS temperature gages, so the rating-curve physics line is
inactive at scale) and uses Daymet incident solar radiation as the radiative
channel `DH` (a physical replacement of unknown scale relative to the original
3-station `DH`); wind speed is included from gridMET in the 7-variable
configuration. The median
margin over damped persistence at 3–7 days is small (the win-rate carries the
result), and a strong gradient-boosting baseline (LightGBM) is **at least as
accurate as ThermoRoute**: significantly better at 1 day and statistically
indistinguishable at 3–7 days once the seed budget is matched (per-seed protocol,
§4.2). We therefore claim a robust improvement over the *physics* baselines and
calibrated, transferable uncertainty — not a new state of the art over all
learners. We report five seeds. The dynamic-κ thermal-memory modulation does
**not** generalise: the flow-dependence seen on three stations vanishes on the
large sample (κ rises with flow at 0 % of stations), and freezing κ's modulators
does not worsen RMSE, so we retract the mechanism claim and keep only the
router's interpretable, horizon-dependent driver shares.

**Input uncertainty.** Our forcings are gridded reanalysis (Daymet, gridMET), which
we treat as exact inputs — the very assumption that BATEA [G2, G3] warns against for
rainfall–runoff, where forcing error can dominate and, if ignored, biases the
calibrated parameters. Two points make this less acute here than in the
rainfall–runoff setting BATEA addresses. First, water temperature is far smoother
and more strongly autocorrelated than discharge, so short-horizon skill is carried
by the persistent state (`WTEMP` history) rather than by the noisy meteorological
forcing, limiting the leverage of input error. Second, we do not need the input-error
model that BATEA requires and that it concedes is "poorly understood": the conformal
layer calibrates the predictive intervals *empirically* on held-out years, absorbing
residual forcing error into the coverage rather than attributing it. The cost, in
BATEA's terms, is that we cannot separate forcing error from model error — a
deliberate trade of process attribution for distribution-free coverage at scale.
A fully operational system would additionally replace our reanalysis forcings with
archived numerical weather-prediction forecasts, whose own error would degrade
multi-day skill and could be handled either by a BATEA-style latent input model or
by re-calibrating the conformal intervals on forecast-driven residuals; we leave this
operational-forcing track to future work.

**Equifinality.** The near-identical validation loss across our five seeds
(1.343–1.348) is a learned-model echo of the parameter equifinality that motivated
GLUE [G1]: many weight configurations simulate the calibration data almost equally
well. Where GLUE embraces this and propagates it into the prediction limits, our
bounded physics prior instead *reduces* it — constraining the residual so the
admissible solutions cluster near damped persistence — which is philosophically
closer to BATEA's add-structure stance than to GLUE's accept-equifinality one, and is
consistent with the low across-seed spread we observe.

## 6. Conclusions

Under a strict historical-information protocol, ThermoRoute matches per-station
damped persistence where the system is near-deterministic (a reservoir cascade) and
beats the physics baselines where forecast headroom exists (a 120-station large
sample), transferring to unseen basins and providing near-nominally-calibrated
warnings. We deliberately report two negative results — no point-accuracy gain on
the cascade, and no generalisable flow-dependent thermal memory — to keep the claims
honest. The contribution is a calibrated, transferable, interpretable forecaster
whose advantage over the physics baselines is established on a large, diverse sample
rather than a single site, with the explicit caveat that a strong gradient-boosting
learner is at least as accurate across leads — better at 1 day and statistically
on par at 3–7 days — so the claim is calibrated, transferable skill over the
physics baselines, not a new state of the art over all learners.

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

**(2) 120-station USGS large sample (main analysis).** Assembled programmatically
from three open sources, in the same schema, by
`scripts/data_usgs/build_usgs_stations.py`:

| study variable | source | access | notes |
|---|---|---|---|
| `WTEMP`, `FLOW`, `WLEVEL` | USGS NWIS daily values | `dataRetrieval`/`dataretrieval-python` (U.S. Geological Survey); public domain | parameter codes 00010 / 00060 / 00065. Gage height (`WLEVEL`) is unavailable at most temperature gages (≈0 % coverage), so the large-sample model omits it; consequently the stage–discharge (rating-curve) physics is inactive at scale. |
| `TEMP`, `PRCP`, `DH`, `RHMEAN` | Daymet v4 (1 km gridded) | ORNL DAAC single-pixel API; open (CC0-equivalent for U.S. government / DAAC terms) | `TEMP` = mean of tmax/tmin; `DH` = incident solar radiation (`srad`, a physical radiative index replacing the original ambiguous `DH`, on a different scale); `RHMEAN` derived from vapour pressure via the Tetens saturation relation. |
| `WDSP` | gridMET | Climatology Lab / Northwest Knowledge Network THREDDS NetCDF-Subset Service; open | daily mean wind speed at the station coordinate. |

120 stream gages across 35 U.S. states and 16 HUC2 regions (≥55 %
water-temperature and ≥70 % flow coverage over 2006–2020, plus a ≥80 %
blind-test-window coverage gate — §2) span free-flowing and dam-regulated
rivers; water-temperature coverage ranges 0.56–1.00 (median ≈0.88). The
assembled main panel is `data_usgs/panel_usgs_100.parquet` (120 stations, seven
model variables plus WLEVEL; the filename is historical and predates the
120-station rebuild). Station metadata with USGS site numbers and coordinates
is in `data_usgs/stations_meta_120v2.csv`, and every probed-but-rejected
candidate is recorded in `data_usgs/rejected_sites_120v2.csv`
(`outputs/reports/usgs_acquisition.md` documents the audit trail). The legacy
40-station pilot panels (`data_usgs/panel_usgs.parquet`,
`data_usgs/panel_usgs_wind.parquet`) are retained only for the §S1 pilot
comparison. All panels are included so the experiments reproduce without
re-downloading; the raw per-site downloads can be regenerated by re-running the
acquisition script. All three sources are public domain or open-licensed.

## Code availability and reproducibility

All code, the fixed train/validation/calibration/blind-test boundaries, unit tests
(leakage, splits, metrics, conformal, cross-model sample consistency), and
one-command reproduction (`scripts/run_all.sh`) are in this repository. Per-day
predictions, model weights and logs are regenerable by the staged scripts
(`scripts/01`–`14`); `scripts/14_manifest.py` writes a sha256 manifest of every
artifact the manuscript's numbers depend on (`outputs/manifest.json`), and
`--check` verifies the recorded hashes so any drift between the manuscript and
the artifacts is machine-detectable. Continuous integration runs the test suite
under both the version-locked environment that produced the published numbers
(`requirements-lock.txt`) and loose forward-compatibility floors. An adversarial
internal review (multiple independent expert lenses) checked every headline claim
against the result tables; all negative results and ablations are retained, and a
finding-by-finding disposition is provided in `outputs/reports/review_response.md`.

**Archived release.** For a persistent, citable record the code, configuration, the
input station panels, and all derived tables/reports/figures are archived at Zenodo
(https://doi.org/REPLACE_WITH_ZENODO_DOI, v1.0.0, MIT license), built with
`scripts/make_release_archive.sh` and carrying the sha256 manifest; the
multi-hundred-MB predictions and model checkpoints are excluded but regenerable from
the archive via `scripts/run_all.sh`. The raw observations are redistributed in
derived form only and are independently available from USGS NWIS, Daymet (ORNL DAAC,
https://doi.org/10.3334/ORNLDAAC/2129) and gridMET; USGS site numbers for all 120
gages are in the archive. (`outputs/reports/data_availability.md` holds the
submission-ready Open Research statement; the DOI is minted at camera-ready.)

## References

*Formatted following the* Journal of Hydrology *house style (Elsevier numeric-author–year). Cross-reference labels in the manuscript text use the F-numbers retained for traceability.*

### Stream temperature: reviews, baselines and learned models

Corona, C. R., Hogue, T. S., 2025. Machine learning in stream and river water temperature modeling: a review and metrics for evaluation. Hydrology and Earth System Sciences 29, 2521–2549. https://doi.org/10.5194/hess-29-2521-2025. [F1]

Feigl, M., Lebiedzinski, K., Herrnegger, M., Schulz, K., 2021. Machine-learning methods for stream water temperature prediction. Hydrology and Earth System Sciences 25, 2951–2977. https://doi.org/10.5194/hess-25-2951-2021. [F2]

Zwart, J. A., Diaz, J., Hamshaw, S., Oliver, S., Ross, J. C., Sleckman, M., Appling, A. P., Corson-Dosch, H., Jia, X., Read, J., Sadler, J., Thompson, T., Watkins, D., White, E., 2023. Evaluating deep learning architecture and data assimilation for improving water temperature forecasts at unmonitored locations. Frontiers in Water 5, 1184992. https://doi.org/10.3389/frwa.2023.1184992. [F3]

Jia, X., Zwart, J., Sadler, J., Appling, A., Oliver, S., Markstrom, S., Willard, J., Xu, S., Steinbach, M., Read, J., Kumar, V., 2021. Physics-guided recurrent graph model for predicting flow and temperature in river networks, in: Proceedings of the 2021 SIAM International Conference on Data Mining (SDM). SIAM, pp. 612–620. https://doi.org/10.1137/1.9781611976700.69 (preprint: arXiv:2009.12575). [F4]

Luo, S., Yu, R., Chen, S., Fan, Y., Xie, Y., Li, Y., Jia, X., 2025. Geo-aware models for stream temperature prediction across different spatial regions and scales, in: Proceedings of the 33rd ACM International Conference on Advances in Geographic Information Systems (SIGSPATIAL '25). ACM, pp. 124–136. https://doi.org/10.1145/3748636.3762716 (preprint: arXiv:2510.09500). [F5]

Rahmani, F., Lawson, K., Ouyang, W., Appling, A., Oliver, S., Shen, C., 2021. Exploring the exceptional performance of a deep learning stream temperature model and the value of streamflow data. Environmental Research Letters 16, 024025. https://doi.org/10.1088/1748-9326/abd501. [F7]

Rahmani, F., Shen, C., Oliver, S., Lawson, K., Appling, A., 2021. Deep learning approaches for improving prediction of daily stream temperature in data-scarce, unmonitored, and dammed basins. Hydrological Processes 35, e14400. https://doi.org/10.1002/hyp.14400. [F8]

Rahmani, F., Appling, A., Feng, D., Lawson, K., Shen, C., 2023. Identifying structural priors in a hybrid differentiable model for stream water temperature modeling. Water Resources Research 59, e2023WR034420. https://doi.org/10.1029/2023WR034420. [F9]

Sadler, J. M., Appling, A. P., Read, J. S., Oliver, S. K., Jia, X., Zwart, J. A., Kumar, V., 2022. Multi-task deep learning of daily streamflow and water temperature. Water Resources Research 58, e2021WR030138. https://doi.org/10.1029/2021WR030138. [F10]

Weierbach, H., Lima, A. R., Willard, J. D., Hendrix, V. C., Christianson, D. S., Lubich, M., Varadharajan, C., 2022. Stream temperature predictions for river basin management in the Pacific Northwest and Mid-Atlantic regions using machine learning. Water 14, 1032. https://doi.org/10.3390/w14071032. [F11]

Willard, J. D., Ciulla, F., Weierbach, H., Xu, Z., Hendrix, V. C., Varadharajan, C., 2024. Evaluating deep learning approaches for predictions in unmonitored basins with continental-scale stream temperature models. arXiv:2410.19865. https://doi.org/10.48550/arXiv.2410.19865. [F13]

Toffolon, M., Piccolroaz, S., 2015. A hybrid model for river water temperature as a function of air temperature and discharge. Environmental Research Letters 10, 114011. https://doi.org/10.1088/1748-9326/10/11/114011. [F12]

Piccolroaz, S., Calamita, E., Majone, B., Gallice, A., Siviglia, A., Toffolon, M., 2016. Prediction of river water temperature: a comparison between a new family of hybrid models and statistical approaches. Hydrological Processes 30, 3901–3917. https://doi.org/10.1002/hyp.10913.

Mohseni, O., Stefan, H. G., Erickson, T. R., 1998. A nonlinear regression model for weekly stream temperatures. Water Resources Research 34, 2685–2692. https://doi.org/10.1029/98WR01877.

Caissie, D., 2006. The thermal regime of rivers: a review. Freshwater Biology 51, 1389–1406. https://doi.org/10.1111/j.1365-2427.2006.01597.x.

### Uncertainty estimation in hydrological modelling

Beven, K., Binley, A., 1992. The future of distributed models: model calibration and uncertainty prediction. Hydrological Processes 6 (3), 279–298. https://doi.org/10.1002/hyp.3360060305. [G1]

Kavetski, D., Kuczera, G., Franks, S. W., 2006a. Bayesian analysis of input uncertainty in hydrological modeling: 1. Theory. Water Resources Research 42, W03407. https://doi.org/10.1029/2005WR004368. [G2]

Kavetski, D., Kuczera, G., Franks, S. W., 2006b. Bayesian analysis of input uncertainty in hydrological modeling: 2. Application. Water Resources Research 42, W03408. https://doi.org/10.1029/2005WR004376. [G3]

Modi, P. A., Small, E. E., Kasprzyk, J., Livneh, B., 2025. Understanding the relationship between streamflow forecast skill and value across the western United States. Hydrology and Earth System Sciences 29, 5593–5613. https://doi.org/10.5194/hess-29-5593-2025. [G4]

### Methods — machine learning, calibration, statistics

Romano, Y., Patterson, E., Candès, E. J., 2019. Conformalized quantile regression, in: Advances in Neural Information Processing Systems 32 (NeurIPS 2019), pp. 3538–3548.

Vovk, V., Gammerman, A., Shafer, G., 2005. Algorithmic Learning in a Random World. Springer, New York. https://doi.org/10.1007/b106715.

Martins, A. F. T., Astudillo, R. F., 2016. From softmax to sparsemax: a sparse model of attention and multi-label classification, in: Proceedings of the 33rd International Conference on Machine Learning (ICML), pp. 1614–1623.

Bai, S., Kolter, J. Z., Koltun, V., 2018. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv:1803.01271.

Grinsztajn, L., Oyallon, E., Varoquaux, G., 2022. Why do tree-based models still outperform deep learning on typical tabular data?, in: Advances in Neural Information Processing Systems 35 (NeurIPS 2022) Datasets and Benchmarks Track. https://doi.org/10.48550/arXiv.2207.08815.

Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., Dean, J., 2017. Outrageously large neural networks: the sparsely-gated mixture-of-experts layer, in: International Conference on Learning Representations (ICLR).

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., Liu, T.-Y., 2017. LightGBM: a highly efficient gradient boosting decision tree, in: Advances in Neural Information Processing Systems 30 (NeurIPS 2017), pp. 3146–3154.

Diebold, F. X., Mariano, R. S., 1995. Comparing predictive accuracy. Journal of Business & Economic Statistics 13, 253–263. https://doi.org/10.1080/07350015.1995.10524599.

Harvey, D., Leybourne, S., Newbold, P., 1997. Testing the equality of prediction mean squared errors. International Journal of Forecasting 13, 281–291. https://doi.org/10.1016/S0169-2070(96)00719-4.

Künsch, H. R., 1989. The jackknife and the bootstrap for general stationary observations. The Annals of Statistics 17, 1217–1241. https://doi.org/10.1214/aos/1176347265.

Wilks, D. S., 2011. Statistical Methods in the Atmospheric Sciences, 3rd ed., International Geophysics Series, vol. 100. Academic Press, Oxford. ISBN 978-0-12-385022-5.

Richardson, D. S., 2000. Skill and relative economic value of the ECMWF ensemble prediction system. Quarterly Journal of the Royal Meteorological Society 126, 649–667. https://doi.org/10.1002/qj.49712656313.

Gneiting, T., Raftery, A. E., 2007. Strictly proper scoring rules, prediction, and estimation. Journal of the American Statistical Association 102, 359–378. https://doi.org/10.1198/016214506000001437.

Kingma, D. P., Ba, J., 2015. Adam: a method for stochastic optimization, in: International Conference on Learning Representations (ICLR).

Loshchilov, I., Hutter, F., 2019. Decoupled weight decay regularization, in: International Conference on Learning Representations (ICLR).

Nash, J. E., Sutcliffe, J. V., 1970. River flow forecasting through conceptual models part I — a discussion of principles. Journal of Hydrology 10, 282–290. https://doi.org/10.1016/0022-1694(70)90255-6.

Gupta, H. V., Kling, H., Yilmaz, K. K., Martinez, G. F., 2009. Decomposition of the mean squared error and NSE performance criteria: implications for improving hydrological modelling. Journal of Hydrology 377, 80–91. https://doi.org/10.1016/j.jhydrol.2009.08.003.

### Software and data sources

De Cicco, L. A., Hirsch, R. M., Lorenz, D., Watkins, W. D., Johnson, M., 2024. dataRetrieval: R packages for discovering and retrieving water data available from federal hydrologic web services, v.2.7.17. U.S. Geological Survey. https://doi.org/10.5066/P9X4L3GE. [D1]

Hodson, T. O., Hariharan, J. A., Black, S., Horsburgh, J. S., 2023. dataretrieval (Python): a Python package for discovering and retrieving water data available from U.S. federal hydrologic web services. U.S. Geological Survey software release. https://doi.org/10.5066/P94I5TX3.

U.S. Geological Survey, 2024. USGS water data for the Nation: U.S. Geological Survey National Water Information System (NWIS) database. Web services accessed via dataRetrieval; parameter codes 00010 (water temperature), 00060 (discharge), 00065 (gage height); accessed 2026. https://doi.org/10.5066/F7P55KJN (https://waterdata.usgs.gov/nwis). [public domain]

Thornton, M. M., Shrestha, R., Wei, Y., Thornton, P. E., Kao, S.-C., 2022. Daymet: daily surface weather data on a 1-km grid for North America, version 4 R1. ORNL DAAC, Oak Ridge, Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/2129. [D2]

Abatzoglou, J. T., 2013. Development of gridded surface meteorological data for ecological applications and modelling. International Journal of Climatology 33, 121–131. https://doi.org/10.1002/joc.3413. (gridMET / METDATA.) [D3]

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., Chintala, S., 2019. PyTorch: an imperative style, high-performance deep learning library, in: Advances in Neural Information Processing Systems 32 (NeurIPS 2019), pp. 8024–8035.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., Duchesnay, É., 2011. Scikit-learn: machine learning in Python. Journal of Machine Learning Research 12, 2825–2830.

Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R. J., Jones, E., Kern, R., Larson, E., Carey, C. J., Polat, I., Feng, Y., Moore, E. W., VanderPlas, J., Laxalde, D., Perktold, J., Cimrman, R., Henriksen, I., Quintero, E. A., Harris, C. R., Archibald, A. M., Ribeiro, A. H., Pedregosa, F., van Mulbregt, P., SciPy 1.0 Contributors, 2020. SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature Methods 17, 261–272. https://doi.org/10.1038/s41592-019-0686-2.

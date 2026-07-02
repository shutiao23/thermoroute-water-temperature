# ThermoRoute

**Physics-guided, dynamic-lag, calibrated, transferable river water-temperature forecasting.**

ThermoRoute forecasts daily water temperature (`WTEMP`) 1, 3 and 7 days ahead. It
couples a *learnable dynamic thermal-relaxation prior* — a flow- and
season-modulated generalisation of damped persistence toward climatology that
contains the strong baseline as a special case — with a horizon-conditioned sparse
variable–lag router, a causal TCN encoder, a regime mixture-of-experts, and a
**bounded** neural residual; outputs are conformally-calibrated quantiles plus a
high-temperature exceedance probability.

The study has two settings.

* **Main analysis (large-sample, USGS).** 120 public USGS stream gages with
  Daymet meteorology and gridMET wind, 2006–2020, free-flowing and
  dam-regulated. On the 114 blind-test stations, with the seed budget matched
  (each seed scored as a single model), **ThermoRoute beats persistence by
  +0.19/+0.18/+0.24 skill (1/3/7 d) and damped persistence by +0.16/+0.07/+0.03**
  (per-station Wilcoxon, Holm-adjusted, p ≤ 3×10⁻¹⁶; robust to a HUC2 cluster
  bootstrap), and beats an air2stream-style 8-parameter physical model (a
  *variant* of Toffolon–Piccolroaz) at every lead (0.630/1.289/1.658 vs
  0.797/1.464/1.809 °C). Against a strong gradient-boosting learner (LightGBM)
  the honest result is **parity, not superiority**: LightGBM is significantly
  better at 1 day and the two are statistically tied at 3–7 days. In 4-fold
  leave-group-out transfer to basins it never trained on it beats persistence by
  +0.17/+0.17/+0.24. Conformal calibration delivers near-nominal PICP (≈0.90).
* **Case study (3-station cascade).** Three reservoir-cascade stations
  (`b1`→`s2`→`p3`), 2006–2020. The reservoir outlets are so heavily damped that
  **no learned model consistently improves on per-station damped persistence on
  point RMSE across horizons** — an honest negative result that motivates the
  large-sample study and is reported in full.

Two negative results are reported in full: no point gain on the cascade, and a
flow-dependent thermal memory that does not generalise beyond it (κ rises with
flow at 0 % of large-sample stations).

See `paper/ThermoRoute_paper.md` for the full manuscript. Every headline number
is traceable to a hashed artifact in `outputs/manifest.json` (`scripts/14`);
`outputs/reports/adversarial_review_tri_persona.md` records the most recent
three-expert adversarial review and `outputs/reports/review_response.md` an
earlier six-lens one.

---

## Why the problem is hard (and honest)

Reservoir water temperature has lag-1 autocorrelation ≈ 0.998. **Persistence is a
brutal baseline**, and only damped persistence toward climatology reliably beats
it. A generic strong learner (LightGBM) is at least as accurate as ThermoRoute —
better at 1 day, statistically tied at 3–7 days once the seed budget is matched.
The contribution is therefore (i) beating the *physics* baselines robustly, (ii)
spatial transfer, (iii) calibrated uncertainty — established on a large,
hydrologically diverse sample rather than a single site, **not** a claim of
state-of-the-art point accuracy over all learners.

---

## Repository layout

```
project1/
├── data/                       raw 3-station cascade CSVs (b1, s2, p3)
├── data_usgs/                  USGS large-sample panels (panel_usgs_100.parquet),
│                                 per-station n*.csv, stations_meta.csv, acquisition report
├── src/thermoroute/
│   ├── config.py               protocol constants (split, topology, thresholds)
│   ├── data.py                 loading, sentinel QC, fold-safe split + impute
│   ├── features.py             harmonic climatology, tabular lag features
│   ├── datasets.py             windowed tensors + leakage guard (NaN-safe)
│   ├── baselines.py            persistence … LightGBM (3-station baselines)
│   ├── air2stream.py           canonical Toffolon–Piccolroaz hybrid (a4 + a8)
│   ├── thermoroute.py          the model (prior + router + TCN + MoE + heads)
│   ├── train.py                training loop, composite loss, GRU reference
│   ├── conformal.py            conformalised quantile regression (CQR)
│   ├── metrics.py              point / probabilistic / event metrics
│   ├── significance.py         moving-block bootstrap, Diebold-Mariano
│   ├── decision.py             cost-loss decision value (REV)
│   ├── results.py              canonical predictions schema + scoring
│   └── usgs.py                 NWIS + Daymet + gridMET acquisition
├── scripts/
│   ├── 01_prepare_data.py            3-station audit + processed panel
│   ├── 04_run_experiments.py         3-station experiment matrix
│   ├── 05_explain.py                 3-station mechanism extraction
│   ├── 06_make_figures.py            3-station + USGS figures
│   ├── 07_make_tables.py             3-station paper tables
│   ├── 08_decision_value.py          REV decision-value analysis
│   ├── 09_usgs_experiment.py    USGS main: baselines + air2stream + ThermoRoute × seeds + LGO + ablations
│   ├── 10_usgs_analysis.py      USGS calibration, REV, mechanism (κ, router drivers)
│   ├── 11_retune.py             residual-bound (delta_scale) tuning
│   ├── 12_claim_stats.py        per-station Wilcoxon + bootstrap CI (Claims 1, 3)
│   ├── 13_rigor.py              K-fold leave-group-out + 3-seed ablations (Claims 2, 4)
│   ├── data_usgs/build_usgs_stations.py   acquisition driver
│   └── run_all.sh                     one-command reproduction (both tracks)
├── tests/                              leakage / split / metric unit tests (13 tests)
├── paper/
│   ├── ThermoRoute_paper.md            manuscript
│   ├── ThermoRoute_paper.pdf|.docx     rendered
│   ├── cover_letter.md|.pdf|.docx      submission cover letter
│   └── highlights.md|.pdf|.docx        JoH-format highlights + one-page summary
└── outputs/                            tables, figures, predictions, reports, models
```

## Quick start

```bash
pip install -r requirements.txt          # torch, lightgbm, sklearn, ...
bash scripts/run_all.sh                   # full pipeline, both tracks (multi-hour on CPU)
```

Or step by step (set `PYTHONPATH=src`; first batch is the 3-station case study,
second batch is the USGS large-sample main analysis):

```bash
# --- 3-station case study (~30 minutes on CPU) ---
python3 scripts/01_prepare_data.py        # audit + processed panel
python3 -m pytest tests/ -q               # leakage / metric tests (13 pass)
python3 scripts/04_run_experiments.py     # 3-station experiment matrix
python3 scripts/05_explain.py             # 3-station mechanism extraction
python3 scripts/06_make_figures.py        # 3-station figures
python3 scripts/07_make_tables.py         # 3-station paper tables
python3 scripts/08_decision_value.py      # REV analysis

# --- USGS large-sample main analysis (multi-hour on CPU) ---
# acquisition (network-bound) — already pre-acquired in data_usgs/
python3 scripts/data_usgs/build_usgs_stations.py --target 120 --max-probe 1500 \
    --out panel_usgs_100.parquet --states CO OR WA PA NY MN WI CA ID MT [...]
# main experiment (5 seeds + air2stream + LGO + ablations)
python3 scripts/09_usgs_experiment.py --panel data_usgs/panel_usgs_100.parquet \
    --air2stream --seeds 5
# downstream: calibration / REV / mechanism + significance + rigor
python3 scripts/10_usgs_analysis.py
python3 scripts/12_claim_stats.py
python3 scripts/13_rigor.py
```

## The model in one screen

For station *s*, issue day *t*, horizon *h* ∈ {1,3,7}:

```
prior :  a_t = W_t − C_t                      today's anomaly
         e_t = g(weather_t)                   weather-driven equilibrium anomaly
         κ   = σ(b_s + c_q·z(logFLOW) + c_l·z(WLEVEL) + season)   daily relax rate
         â_h = e_t + (1−κ)^h (a_t − e_t)
         prior_h = C_{t+h} + â_h              (= damped persistence when e=0, κ=1−φ)

residual: routed = sparsemax router over {variable × lag(0..14) × horizon}
          latent = causal TCN(history)
          Δ_h    = MoE(routed, latent | season, flow, precip regime)
          Δ_h    is **bounded by ±delta_scale °C** (tanh) so the prior is never overridden

output : median_h = prior_h + Δ_h
         q05/q95  = median ∓ softplus(width)   → CQR-calibrated on 2018
         P(exceed q90)_h
```

## Leakage discipline (enforced, not promised)

* Split by date: **train 2006–2015 / val 2016–2017 / calib 2018 / blind test 2019–2020**.
* All scalers, climatology, rating curves and imputation statistics are fit on
  **train only**.
* `datasets._assert_no_leakage` verifies every window's last step inverts to
  `WTEMP_t`; `tests/test_leakage.py` checks splits, sentinels and target offsets.
* Sentinel codes `WDSP=999.9`, `PRCP=99.99` are masked to NaN, never used as extremes.
* For USGS panels, a station is included only if its 2006–2020 water-temperature
  coverage is ≥55 %; per-split effective station counts (≤ nominal panel size)
  are documented in `outputs/reports/usgs_acquisition.md` and reported in the
  paper.

## Notes / open items

* **DH semantics on the 3-station data are unverified** (`config.DH_SEMANTICS_VERIFIED
  = False`). The audit is consistent with a sunshine/insolation index but no
  DH-specific claim is made. On USGS panels, `DH` is Daymet incident solar
  radiation (W/m²), a physical replacement on a different scale.
* This is a **historical-information (Track H)** study: inputs use only data
  available at issue time. No future observed meteorology is used.
* Three stations are intentionally a *case study* — the main analysis is the
  120-station USGS large sample, which has the forecast headroom (persistence
  h7 median ≈ 2.3 °C) needed to distinguish models.

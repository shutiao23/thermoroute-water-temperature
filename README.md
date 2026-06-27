# ThermoRoute

**Physics-guided, dynamic-lag, calibrated multi-station river water-temperature forecasting.**

ThermoRoute forecasts daily water temperature (`WTEMP`) 1, 3 and 7 days ahead at
three reservoir-cascade stations (**b1 → s2 → p3**, 2006–2020, 15 years of daily
records). It pairs a **learnable dynamic thermal-relaxation physics prior** with a
**horizon-conditioned sparse variable–lag router**, a causal TCN encoder and a
regime mixture-of-experts, and emits **conformally-calibrated quantiles** plus a
**high-temperature exceedance probability**.

The scientific claim is not "a neural net fits better". It is:

> The river's thermal response time-constant is **not fixed** — it varies with
> flow, level and season — and a horizon-conditioned sparse lag router recovers
> *which drivers and which lags* matter at 1, 3 and 7 days. The physics prior
> makes **damped persistence a strict special case**, so any gain is attributable
> to the dynamic mechanism, and every interval is calibrated with a finite-sample
> guarantee.

---

## Why this problem is hard (and honest)

Reservoir water temperature is extremely autocorrelated (lag-1 ≈ 0.998).
**Persistence is a brutal baseline**, and only *damped persistence toward
climatology* reliably beats it. A generic strong learner (LightGBM with 126
features) does **not** beat that physics-aware baseline at 3–7 days — it overfits.
ThermoRoute is therefore designed to (a) match/beat the damped-persistence floor
on point accuracy by *generalising* it, and (b) add what the floor cannot:
calibrated uncertainty, high-temperature warning, and an interpretable,
flow-and-season-dependent thermal-memory mechanism.

See `outputs/reports/data_audit.md` for the numbers behind every claim above.

---

## Repository layout

```
project1/
├── data/                       raw CSVs (b1,s2,p3) + processed/panel.parquet
├── src/thermoroute/
│   ├── config.py               protocol constants (split, topology, thresholds)
│   ├── data.py                 loading, sentinel QC, fold-safe split + impute
│   ├── features.py             harmonic climatology, tabular lag features
│   ├── datasets.py             windowed tensors + leakage guard
│   ├── baselines.py            persistence … air2stream-lite … LightGBM
│   ├── thermoroute.py          the model (prior + router + TCN + MoE + heads)
│   ├── train.py                training loop, composite loss, GRU reference
│   ├── conformal.py            conformalised quantile regression (CQR)
│   ├── metrics.py              point / probabilistic / event metrics
│   ├── significance.py         moving-block bootstrap, Diebold-Mariano
│   └── results.py              canonical predictions schema + scoring
├── scripts/
│   ├── 01_prepare_data.py      → data_audit.md, panel.parquet
│   ├── 04_run_experiments.py   → predictions.parquet, scores_all.csv
│   ├── 05_explain.py           → explain.npz, mechanism_summary.md
│   ├── 06_make_figures.py      → outputs/figures/*.png|pdf
│   ├── 07_make_tables.py       → outputs/tables/paper_tables.md
│   └── run_all.sh              one-command reproduction
├── tests/                      leakage / split / metric unit tests
├── paper/                      manuscript draft (Methods + Results, real numbers)
└── outputs/                    tables, figures, predictions, reports, models
```

## Quick start

```bash
pip install -r requirements.txt          # torch, lightgbm, sklearn, ...
bash scripts/run_all.sh                   # ~30–60 min on a laptop CPU
```

Or step by step (set `PYTHONPATH=src`):

```bash
python3 scripts/01_prepare_data.py        # audit + processed panel
python3 -m pytest tests/ -q               # 13 leakage / metric tests
python3 scripts/04_run_experiments.py     # full matrix (the long step)
python3 scripts/05_explain.py             # router + κ extraction
python3 scripts/06_make_figures.py        # 10 figures
python3 scripts/07_make_tables.py         # 6 paper tables
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

## Notes / open items

* **DH semantics are unverified** against a data dictionary (`config.DH_SEMANTICS_VERIFIED
  = False`). The audit is consistent with a sunshine/insolation index; it enters
  models as a generic radiative channel and no DH-specific claim is made.
* This is a **historical-information (Track H)** study: inputs use only data
  available at issue time. No future observed meteorology is used; an Oracle
  upper bound is intentionally omitted from headline results.
* Three stations are too few for a deep graph network; cross-station structure is
  modelled through the directed travel-time prior and LOSO transfer, not a GNN.

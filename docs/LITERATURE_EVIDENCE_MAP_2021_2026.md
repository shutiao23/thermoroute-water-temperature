# River-temperature prediction literature evidence map, 2021–2026

| Field | Value |
| --- | --- |
| Date | 2026-08-01 |
| Status | Literature-positioning evidence; not a numerical meta-analysis |
| ThermoRoute Route A | Daily-mean h=1/3/7 retrospective known-gauge hindcast using target-site recent WTEMP |
| ThermoRoute Route B | Must separate strict ungauged, graph-disconnected-network holdout and as-issued operational arms; graph disconnection alone is not statistical independence |
| Manuscript boundary | This map is under `docs/`; frozen PRE paper/bibliography files are not modified here |

## 1. Comparison axes

Cross-paper RMSE ranking is invalid unless all of the following align:

- daily mean versus daily maximum/monthly target;
- reconstruction/hindcast versus real forecast issue dates;
- access to target-site recent WTEMP and observed flow;
- latest retrospective meteorology versus archived as-issued NWP;
- temporal holdout, random site holdout, spatial block or disconnected-network
  holdout;
- point output versus calibrated distribution/quantiles/events.

Route A's 30 external station identities still provide their own recent WTEMP.
They are monitored/known-gauge transfer, not strict ungauged prediction.

## 2. Core evidence map

| Study | Task / validation | Target-site history and probability | ThermoRoute role |
| --- | --- | --- | --- |
| [Corona & Hogue 2025, HESS review](https://hess.copernicus.org/articles/29/2521/2025/) | Review of 57 stream-temperature ML studies and evaluation gaps | Mixed; highlights uncertainty, extremes and ungauged generalization | Primary 2025 positioning source; not a numerical comparator |
| [Leach et al. 2023, process primer](https://wires.onlinelibrary.wiley.com/doi/10.1002/wat2.1643) | Process synthesis of heat exchange, advection and inflows | Not a prediction evaluation | Supports process boundaries; cannot validate ThermoRoute latent physics |
| [Toffolon & Piccolroaz 2015, Air2stream](https://iris.unitn.it/handle/11572/143425) | Low-order air-temperature/discharge model at three rivers | Site calibration; point prediction | Theoretical official-baseline origin; not strict ungauged/probabilistic |
| [Feigl et al. 2021, HESS](https://hess.copernicus.org/articles/25/2951/2021/) | Ten Austrian basins; RF/XGB/FNN/GRU/LSTM versus linear/Air2stream; within-basin temporal tests | Local records; point prediction | Shows tuned ML can beat Air2stream locally; weak spatial-transfer comparator |
| [Qiu et al. 2021, Journal of Hydrology](https://www.sciencedirect.com/science/article/pii/S0022169421000639) | Nine stations across China/US/Switzerland; LSTM versus Air2stream/RF/BPNN | Site-calibrated; point prediction | Core China/cross-region citation; dam-altered thermal regimes are an important failure domain |
| [Tao et al. 2021, Journal of Hydrology](https://doi.org/10.1016/j.jhydrol.2021.126430) | Daily Yangtze water temperature at Cuntan, Datong and Yichang; C-vine conditional quantiles versus logistic regression and GRNN | Site-calibrated probabilistic reconstruction using lagged air temperature and discharge | China-basin probabilistic context; not a multi-horizon untouched forecast and not evidence of independent spatial transfer |
| [Huang et al. 2023, Journal of Hydrology](https://doi.org/10.1016/j.jhydrol.2022.128857) | Monthly 1960--2020 reconstruction for four Dongting Lake Basin tributaries using LSTM plus time-series analyses | Long-term monthly reconstruction, not an issue-time predictive distribution | China-basin data-scarcity and thermal-regime context; target aggregation and reconstruction task prevent direct ranking against Route A |
| [Rahmani et al. 2021, ERL/USGS](https://pubs.usgs.gov/publication/70224248) | Shared LSTM across 118 data-rich CONUS basins; temporal validation | Known gauges; point prediction | Strong Route-A large-sample background, not unmonitored spatial evidence |
| [Rahmani et al. 2021, Hydrological Processes/USGS](https://pubs.usgs.gov/publication/70238318) | 400+ data-scarce, unmonitored and dammed basins; includes PUB evaluation | No target WT in PUB configurations; some use target observed flow | High Route-B comparator only when information sets match |
| [Weierbach et al. 2022, Water](https://doi.org/10.3390/w14071032) | Regional monthly WT, including PUB XGBoost/SVR/MLR | No calibrated distribution | Shows regional conventional ML transfer; monthly target is not directly comparable |
| [Siegel et al. 2023, PLOS Water](https://journals.plos.org/water/article?id=10.1371%2Fjournal.pwat.0000119) | Daily PNW reach reconstruction with leave-year and leave-region evaluation | No local forecast history requirement; point mapping | Strong spatial-block design reference; reconstruction rather than h=1/3/7 forecast |
| [Almeida & Coelho 2023, GMD](https://gmd.copernicus.org/articles/16/4083/2023/) | 83 sparse-record stations; Air2stream/ML/ensemble under limited forcing | Within-station tests; point prediction | Supports heterogeneous best-model and missing-data arguments; weak Route-B comparator |
| [Zwart et al. 2023, near-term forecasts/USGS](https://www.usgs.gov/publications/near-term-forecasts-stream-temperature-using-deep-learning-and-data-assimilation) | Real 2021 warm-season 1–7 day forecasts at five managed Delaware sites using GEFS and data assimilation | Recent target WT; ensemble intervals with undercoverage | Closest operational Route-A comparator; retrospective ThermoRoute cannot inherit its operational label |
| [Zwart et al. 2023, Frontiers in Water](https://www.frontiersin.org/journals/water/articles/10.3389/frwa.2023.1184992/full) | LSTM/RGCN ± assimilation across about 70 Delaware reaches with spatial/temporal tests | Gauged/ungauged configurations; probabilistic metrics | Shows spatial holdout worsens error/calibration; verify neighbor-WT information before calling strict ungauged |
| [Wade et al. 2024, NWM physical WT](https://andrewsforest.oregonstate.edu/publications/5361) | Explicit physical heat-flux model in an NWM framework at an Oregon research basin | Process simulation; point output | Architecture/process reference; not a continental or probability benchmark |
| [Padrón et al. 2025, HESS](https://hess.copernicus.org/articles/29/1685/2025/) | Swiss 54-station daily-maximum direct forecasts to 32 days; TFT/RNNED/NHITS; new-site and no-target-WT configurations | Past WT and ECMWF ensembles in main arm; quantile/CRPS evaluation | Highest-priority Route-A probabilistic sequence comparator and important Route-B ablation reference |
| [Chang et al. 2025, WRR](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024WR039053) | Alaska 142-basin LSTM with 42 fully held-out spatial test basins | No target WT or observed flow in spatial test | High Route-B comparator; random site holdout is still weaker than disconnected-network holdout |
| [Diaz et al. 2025, CONUS RGCN/USGS](https://www.usgs.gov/publications/evaluation-daily-stream-temperature-predictions-1979-2021-across-contiguous-united) | Daily min/mean/max reconstruction for 57,810 CONUS reaches using graph ML | Observations/meteorology/static graph; normal distribution/interval evaluation | Establishes continental graph-probability prior art; public summary alone does not prove strict component holdout |
| [Philippus et al. 2025, TempEst 2](https://www.sciencedirect.com/science/article/pii/S0022169425006596) | CONUS 1-km daily mean/max mapping; grid leave-cell-out and year walk-forward | No target-site local calibration; point prediction | Strong Route-B spatial OOD design; not a multi-horizon forecast |
| [Siddik et al. 2025, OSTI](https://www.osti.gov/pages/biblio/3010625) | About 300 CONUS basins; large-sample LSTM with thermally relevant upstream contributing areas and spatial/time test folds | No local WT for reach expansion; point prediction | Highest-priority Route-B architecture/scale comparator; probability/operational claims remain separate |
| [Philippus, Corona & Hogue 2026, TempEst-NEXT](https://www.sciencedirect.com/science/article/pii/S0022169426007171) | CONUS daily 1–16 day ungaged forecast without local calibration | Forecast weather; point prediction | Establishes CONUS ungaged multi-horizon prior art; blocks “first” claims |
| [Yang & Xue 2026, PGDL-A2S](https://www.sciencedirect.com/science/article/pii/S0952197626006676) | Air2stream ODE plus GRU-generated parameters at 13 U.S. stations | Historical WT/air temperature/flow; uncertainty claims require full-text qualification | Close physics-guided Route-A architecture comparator; public evidence does not establish strict component holdout or calibrated distribution |
| [Yang et al. 2026, Journal of Environmental Management](https://doi.org/10.1016/j.jenvman.2025.128460) | CNN-LSTM-attention reconstruction of 1960--2009 Yangtze Basin water temperature and retrospective heatwave analysis | Historical air temperature, streamflow and day-of-year; reconstruction rather than prospective forecast | Recent China-basin deep-learning context and an explicit heatwave application; not an operational, strict-ungauged or graph-disconnected comparator |

## 3. Comparator groups

### Route A: known-gauge h=1/3/7

Highest priority: Padrón 2025 and both Zwart 2023 studies. Feigl, Qiu and the
known-gauge Rahmani study provide secondary temporal/local context.

### Route B: strict ungauged

Highest priority: Chang 2025, Siddik 2025, TempEst 2 and the unmonitored Rahmani
study. Siegel 2023 supplies a strong region-held-out design reference.

### China-basin reconstruction and regulation context

Qiu 2021, Tao 2021, Huang 2023 and Yang 2026 now provide four verified
2021--2026 anchors spanning daily point prediction, daily conditional-quantile
modelling, monthly reconstruction and basin-scale deep-learning reconstruction.
They close the narrow bibliography-discovery gap but do not create a matched
benchmark: all are retrospective/site-calibrated or reconstruction studies, and
none proves a strict ungauged, as-issued operational, graph-disconnected
h=1/3/7 comparison under ThermoRoute's information set.

### Independent-network evaluation

Graph/network studies by Zwart and Diaz are architectural neighbors, but random
site folds or connected withheld reaches do not replace an experiment that holds
out complete disconnected drainage components.

### Probability and events

Padrón, Zwart and Diaz establish the need for horizon-resolved interval score or
CRPS/WIS, marginal coverage, sharpness, Brier score and reliability. Emitting
three quantiles or a CQR interval alone is not a calibrated probabilistic headline.

### Official process baseline

Air2stream originates with Toffolon & Piccolroaz and is used in Feigl, Qiu and
Almeida; PGDL-A2S is a recent physics-guided neighbor. The exact official code,
version, license, forcing and calibration budget must be bound before calling a
local reimplementation “official.”

## 4. Safe positioning statements

1. “Route A evaluates transfer to unseen station identities while retaining each
   target station's recent temperature history; we therefore describe it as
   monitored-site transfer, not prediction in strictly ungauged basins.”
2. “Strict-ungauged performance is reserved for Route B, where target-site
   temperature history is excluded and entire connected river-network components
   are held out.”
3. “The h=1/3/7 experiments are retrospective hindcasts; operational forecasting
   claims require archived as-issued NWP and issue-time data vintages.”
4. “Prior work separately demonstrates monitored-site probabilistic forecasts,
   observation-free spatial extrapolation and continental graph modeling; the
   proposed Route-B contribution is their more demanding intersection, subject
   to an untouched disconnected-network evaluation.”
5. “Direct metric ranking is avoided when target aggregation, horizon, issue-time
   information, spatial split or probability representation differs.”

## 5. Prohibited positioning statements

- first strict-ungauged river-temperature model;
- first continental or river-network water-temperature model;
- first probabilistic or operational river-temperature forecast;
- site-ID-disjoint validation proves ungauged-basin generalization;
- lower reported RMSE proves SOTA across incompatible targets/information/splits.

Before these sources enter `paper/references.bib`, reconcile author lists, year,
volume/pages/article number and DOI against the publisher record and record the
retrieval date. This map intentionally does not edit the frozen PRE manuscript.

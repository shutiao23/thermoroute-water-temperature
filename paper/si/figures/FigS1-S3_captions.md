# PRE supporting-figure captions (FigS1–FigS3)

These captions apply only to the outcome-free PRE figures rendered by
`render_pre_supporting_figures.py`. They contain no target-period result,
performance coordinate, effect estimate, interval, or p-value.

## Fig. S1 — Frozen cohort selection and registry geometry

**Question answered:** Which candidates were retained, and how is the fixed
cohort distributed in coordinate and HUC2 space? **(a)** The committed ledgers
contain 1,465 unique candidate site IDs: 1,345 rejected sites (950 without joint
NWIS water-temperature and streamflow availability, 376 with low full-period
coverage, and 19 with low 2019–2020 development-period coverage) and 120 retained
sites. The frozen registry, rejection ledger, and development panel hashes and
station identities are checked by the renderer before materialization. The
committed ledgers audit this selection, but the original discovery responses,
retrieval timestamps, command line, and full run configuration were not retained,
so the discovery process is not source-replayable. **(b)** Longitude–latitude
scatter of the 120 retained station coordinates. Fifteen unique marker
shape-by-fill combinations identify the HUC2 groups, with color as a redundant
secondary cue. This is deliberately not a map:
it has no national boundary or basemap and supports no national-coverage or
representativeness claim. **(c)** Registry-derived retained-site counts for the
15 HUC2 groups. **(d)** The outcome-free environmental audit records 97 unique
HUC codes, 38 stations in repeated-HUC groups, a 53.856 km median and 0.752 km
minimum nearest-station distance, and 19 stations with a neighbor within 10 km.
The marker projection reconciles exactly to all 120 registry rows. No basemap or
HUC boundary is rendered, so map-source rights are not applicable. HUC2 is a
coarse descriptive grouping, and neither HUC overlap nor coordinate proximity
establishes hydraulic connectivity or an independent river-network component.
The cohort is availability-enriched, not a probability sample of U.S. rivers.
Sources: `station_registry_v1.csv`, `panel_usgs_120v2.parquet`,
`rejected_sites_120v2.csv`, and `development_environmental_audit_v1.json`.

## Fig. S2 — Temporal roles and issue-time information boundary

**Question answered:** When may information enter the retrospective prediction
workflow? **(a)** The frozen temporal roles are 2006–2015 training, 2016–2017
validation, 2018 calibration, 2019–2020 inspected exploratory development, and a
one-time 2021–2023 target-period evaluation. The 2019–2020 partition informed
cohort, model, and narrative development and is not independently primary.
**(b)** At historical issue date *t*, predictor inputs may use observed
target-site water-temperature history and dated FLOW, Daymet, and gridMET values
only through *t*, together with frozen climatology and damped-anchor inputs.
Target water temperature at *t* + *h*, horizon-specific future weather forecasts,
and anticipatory future-input vintages are excluded from predictor inputs for
registered horizons *h* ∈ {1, 3, 7} days. This is a date-indexed retrospective
hindcast. **(c)** WTEMP and FLOW are dated NWIS inputs through *t*; TEMP, PRCP,
RHMEAN, and DH are dated Daymet inputs through *t*; and WDSP is a dated gridMET
input through *t*. All are the latest values served at retrospective retrieval.
**(d)** The outcome-free 2018–2020 bridge covers 120 sites and 131,520 site-days.
For exactly five meteorological fields (TEMP, PRCP, RHMEAN, DH, and WDSP), it
establishes product/parser compatibility, exact missing-pattern agreement, and
best-or-tied zero-day alignment in a ±1-day check; it does not test WTEMP or FLOW.
The capability and limitation columns have equal visual weight. The bridge
requested or read no outcomes, does not establish archived as-issued availability,
does not prove that Daymet/gridMET calendar days share NWIS local-day boundaries,
and does not establish target-period predictor availability. Route A is not an
operational replay, ungauged prediction, or causal transport analysis.

## Fig. S3 — ThermoRoute PRE model and calibration architecture

**Question answered:** How do the frozen inputs, left-looking representation,
bounded point output, and separate calibrated heads connect? **(a)** The sequence builder
constructs a 32-day tensor over the frozen variable order WTEMP, FLOW, TEMP, PRCP,
RHMEAN, DH, and WDSP with explicit missingness masks. WLEVEL may be retained as
raw evidence but is excluded from all Route-A model inputs. The 32-day tensor is
a construction buffer, not an effective-memory claim. **(b)** A horizon-specific sparse
router allocates the seven variables across lags 0–14, while a strictly
left-looking two-block, kernel-three TCN has theoretical receptive field 7; no
input older than router lag 14 can affect the current model output. A
mixture-of-experts combines the resulting representations. The learned
flow-and-season-conditioned relaxation produces proposal *P*, while frozen
damped persistence supplies anchor *A*. The point head is algebraically bounded
about *A* by δ = 1.0 °C through the shown tanh identity; this is not a bound on
truth error, event risk, interval width, or safety. **(c)** Point, nominal quantile, and
event heads are separate. The independently parameterized q0.50 head uses the
same anchor and algebraic δ bound as the point head; q0.05 and q0.95 are then
constructed as q0.50 minus/plus separate positive softplus widths, so the
interval endpoints are not themselves constrained to the ±δ envelope. **(d)** After
member averaging, CQR produces the final calibrated interval and one
horizon-specific Platt calibrator produces the final calibrated event probability;
both are fitted on 2018 only. These implementation facts are asserted against the
protocol, ordered-variable registry, configuration, and model source before
rendering. Learned κ and router weights are internal predictor allocations. The
router does not reveal physical drivers. These latent allocations support no
causal transport interpretation.

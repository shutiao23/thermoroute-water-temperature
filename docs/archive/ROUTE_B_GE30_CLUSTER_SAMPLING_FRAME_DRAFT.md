# Route B — graph-disconnected network sampling frame draft (P0-B1)

| Field | Value |
| --- | --- |
| Date | 2026-08-01 |
| Status | **DRAFT ONLY / OUTCOME-FREE / NOT A FROZEN PROTOCOL** |
| Route-A boundary | Does not change Route A, its HUC2 gate, its source identity, or its permanently descriptive claim eligibility |
| Runtime boundary | No target values, Stage-09 artifacts, experiment process, or `outputs/**` path may be read or written by this design work |

## 1. Feasibility correction

The earlier draft tentatively used HUC2 as the Route-B cluster key while requiring
at least 30 clusters. That is impossible: the official USGS WBD data dictionary
states that the national WBD has only 22 two-digit hydrologic units. HUC2 may be
used for geographic stratification and reporting, but it cannot be the Route-B
primary sampling unit (PSU) under a ≥30-cluster design.

Official references:

- [USGS WBD data dictionary](https://www.usgs.gov/ngp-standards-and-specifications/watershed-boundary-dataset-wbd-data-dictionary)
- [USGS Watershed Boundary Dataset](https://www.usgs.gov/national-hydrography/watershed-boundary-dataset)
- [USGS NHDPlus HR user guide](https://pubs.usgs.gov/sir/2025/5031/sir20255031.pdf)

Route B must have its own outcome-free protocol. It must not silently split the
Route-A HUC2 units, reuse the Route-A gate, or describe the existing 15-HUC2
cohort as “almost confirmatory.”

## 2. Proposed population and estimand

The recommended primary population is **eligible monitored graph-disconnected
drainage-network components**, not every U.S. river and not every calendar day.

The proposed primary PSU is a frozen weakly connected drainage-network component
derived from an immutable NHDPlus HR `PlusFlow` graph. The graph must include
documented diversion and artificial-path edges. A candidate gauge is snapped to
one flowline under a predeclared distance/ambiguity rule, then inherits its
`network_component_id`.

HUC2/HUC4, climate region and stream-size classes are sampling strata and report
labels only. They are not evidence that two sites are hydrologically independent.
Likewise, graph disconnection proves only that the frozen, audited PlusFlow graph
contains no encoded connection. It does not establish statistical independence or
exclude omitted diversions, groundwater exchange, reservoir operations or shared
weather. Those limitations require input-completeness gates, sensitivity analyses
and space×time UQ.

For inference to the eligible monitored-component frame, the recommended point
estimand is a Hájek design-weighted mean paired difference in RMSE:

```text
within site:       Δ_s = RMSE(candidate) - RMSE(reference)
component mean:    Δ_g = (1/N_g) × sum_sample[Δ_s / π_site|component]
raw design weight: u_g = 1 / π_component
normalized weight: w_g = u_g / sum(u_g)
Route-B effect:    sum(w_g × Δ_g)
```

Here `N_g` is the number of eligible sites in component g. With one uniformly
selected primary site, `π_site|component=1/N_g` and the component estimator is the
selected site's Δ. With a self-weighting PSU design, the overall estimator then
coincides with the component-equal sample mean. Otherwise the exact two-stage
weights/finite-population factors and a design-consistent replicate bootstrap are
mandatory; the unweighted component mean is descriptive sensitivity only. This differs deliberately from
Route A's fixed-cohort station-median estimand.
If a median or other robust functional is preferred, it must be chosen before
power calculations and target access; it is not inherited automatically.

## 3. Sampling design

### 3.1 Frame construction

Only pre-target, outcome-free information may define eligibility:

- monitoring-location and time-series identity metadata;
- pre-target record start/end metadata and predictor availability;
- site type and coordinates;
- frozen WBD/NHDPlus topology and snap quality;
- predeclared domain exclusions, such as a positive-flow non-tidal domain if an
  official process baseline requires it.

Target-period water-temperature values, errors, event rates and target-period
completeness must not be inspected or used to select sites/components.

All Route-A sites and every network component containing a Route-A site are
excluded from the primary Route-B frame. This prevents a nominally new site on a
development river network from being called graph-disconnected transfer.

### 3.2 Two-stage probability sample

1. Stratify eligible network components by outcome-free HUC4/HUC2, climate and
   size metadata.
2. Select components using recorded positive inclusion probabilities and a
   public seed.
3. Within each selected component, select one primary gauge using a recorded
   uniform positive probability over the frozen eligible sites. A later unequal
   within-component design requires its exact Horvitz–Thompson component-mean
   estimator to be frozen before target access.
4. Pre-rank reserve gauges using the same outcome-free seed and rules.
5. Reserve activation is allowed only before any target request and only for a
   predeclared metadata/topology failure. Once target acquisition begins, no site
   or component substitution is allowed.

Let `K_required = max(45, power-derived K)`. If `K_required > 60`, the draft frame
is `FRAME_INFEASIBLE` unless the upper bound is formally increased before target
access. Otherwise select `K_required` components, so attrition can occur while
every formal comparison/horizon still has a chance to retain at least 30
reportable PSUs. The final number must come from a prelabel power/MDE grid rather
than from the desired journal result.

### 3.3 Task arms must stay separate

| Arm | Target-site history | Permitted label |
| --- | --- | --- |
| Known-gauge | WTEMP observed through issue date | history-dependent known-gauge hindcast |
| Strict ungauged | no target-site WTEMP used in fit, transforms, selection or issue-time input | ungauged transfer |
| Operational | archived as-issued predictors/NWP and latency/vintage replay | operational only if the archive actually exists |

The user must select the primary arm before protocol freeze. A result from one arm
cannot rescue or rename another.

## 4. Required artifacts and schemas

### `candidate_frame_v1.parquet`

```text
site_id
timeseries_id_wtemp
timeseries_id_flow
latitude
longitude
site_type
huc2
huc4
nhdplus_id
snap_distance_m
snap_status
network_component_id
record_start_metadata
record_end_metadata
route_a_component_overlap
eligibility
exclusion_code
source_snapshot_sha256
```

### `network_edges_v1.parquet`

```text
from_nhdplus_id
to_nhdplus_id
edge_type
direction
source_release
source_row_sha256
```

### `cluster_selection_v1.parquet`

```text
network_component_id
stratum
frame_size
eligible_sites_in_component
cluster_inclusion_probability
selection_seed
selection_rank
site_id
within_cluster_probability
raw_component_design_weight
normalized_analysis_weight
role                         # primary | reserve
reserve_rank
activated_prelabel
```

### `sampling_frame_receipt_v1.json`

The self-hashed receipt binds every input/output byte, the graph release, allowed
metadata fields, randomization algorithm/seed, target-access attestation and an
independent replay summary.

### `inference_gate_v1.json`

```text
K_selected
K_reportable_expected
normalized_weight_shares     # sum to one
n_effective                  # 1 / sum(normalized_weight_share_g^2)
effective_fraction           # n_effective / K
largest_weight_share
unresolved_snap_count
graph_input_completeness_pass
unresolved_diversion_or_artificial_path_count
unresolved_cross_basin_connection_count
route_a_component_overlap_count
target_period_years
power_grid_binding
pass
failure_codes
```

All artifacts must live in `data_route_b/prelabel/**` or
`outputs/route_b/prelabel/**` in the future implementation. They must never reuse
the Route-A Stage-09 run tree.

## 5. Acceptance gates

The sampling frame passes only if all of the following are true:

1. `K_required <= 60`, `K_selected = K_required`, and every selected PSU/site has
   positive, replayable inclusion probability and a normalized design weight.
2. Every selected site has one resolved, non-ambiguous graph snap.
3. The declared graph inputs pass completeness checks; no unresolved diversion,
   artificial path or known cross-basin connection remains. Zero encoded edges
   between constructed components is recorded as graph topology, not treated as
   proof of statistical independence.
4. Route-A site and component overlap equal zero.
5. Nominal balance passes `n_effective/K >= 0.75` and
   `largest_weight_share < 0.25`.
6. Every formal comparison/horizon is separately required to retain
   `K_reportable >= 30`, `n_effective/K >= 0.75`, and the largest-share rule at
   analysis time. Missing/failed cells are `NOT_ESTIMABLE`; they do not trigger
   station replacement.
7. The target contains at least three untouched years, or a separately justified
   untouched retrospective period plus a prospective season.
8. Target-label paths are absent/unaccessed at freeze; protocol and frame hashes
   receive an external timestamp or independent custodian receipt.

If the eligible network frame is too small, the failure closure is a monitored
fixed-frame descriptive study. HUC4, HUC8, distance bins or station singletons
must not be substituted after seeing that failure merely to reach 30.

## 6. User/external decisions required before implementation

- geography: CONUS only or Alaska/Hawaii/territories;
- primary task: known-gauge, strict ungauged, or operational;
- target population: eligible monitored network components, not “all rivers”;
- untouched target period;
- scientific margin/MDE and power assumptions; Route-A +0.05 °C has no automatic
  scientific meaning here;
- positive-flow/non-tidal primary domain, if any;
- permission for metadata/topology downloads and an external timestamp/custodian;
- explicit later authorization for any target acquisition/opening.

## 7. Current status

No frame, graph, candidate registry, selection, gate, timestamp or target-period
artifact has been created. This document is a design artifact only and cannot be
cited as a preregistration or as evidence that Route B is feasible.

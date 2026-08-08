# SI11 — spatial, local-adaptation, and leave-cluster sensitivity

**Status:** the matched spatial-transfer experiment on the independent
2021–2023 window (manuscript Section 4.7) is a 2×2 factorial — geometry
{random-site, whole-region} × adaptation {target-local, training-pooled} —
computed by `scripts/final/run_spatial_factorial.py` from `outputs/final/spatial_effects.parquet`
(station-first metrics; identical keys and preprocessing policy within each
cell). The development-period analysis of manuscript Section 4.4 remains a
separate diagnostic and is not folded into the independent-window numbers.

## Factorial design (protocol v1)

| Geometry | Adaptation | Local history in preprocessing | Task label |
|---|---|---|---|
| Random site | target-local | yes (per-station 2006–2015) | Random-local |
| Whole region | target-local | yes (per-station 2006–2015) | Region-local |
| Random site | training-pooled | no (in-fold pooled stats) | Random-pooled |
| Whole region | training-pooled | no (in-fold pooled stats) | Region-pooled |

Every cell: station-agnostic LightGBM (raw target and damped-anchor residual),
frozen main-design per-lead hyperparameters, four folds, five random-split
seeds for the random arm, deterministic leave-HUC2 folds for the region arm,
held-out 2021–2023 common keys. Per-fold hyperparameter re-tuning is not
performed; the frozen main-design selections (2016–2017 validation) are reused
and this is a documented limitation.

| Cell | 1 d RMSE | 3 d RMSE | 7 d RMSE | stations | nearest gauge, median km |
|---|---:|---:|---:|---:|---:|
| Random-local (LightGBM) | 0.641 | 1.344 | 1.700 | 120 | 60 |
| Region-local (LightGBM) | 0.635 | 1.324 | 1.725 | 120 | 263 |
| Random-pooled (LightGBM) | 0.665 | 1.392 | 1.815 | 120 | 60 |
| Region-pooled (LightGBM) | 0.677 | 1.401 | 1.804 | 120 | 263 |
| Random-local (ResidualLightGBM) | 0.636 | 1.304 | 1.689 | 120 | 60 |
| Region-local (ResidualLightGBM) | 0.647 | 1.334 | 1.693 | 120 | 263 |

## Repeated-split paired penalty (region minus random, °C)

Per station the region-cell RMSE is paired with the mean of the five random-split
cells, and the median over stations is reported with the IQR, the random win
fraction, the per-seed medians, and the median per-site split spread:

| Horizon | median penalty | IQR | random win fraction | per-seed medians | split spread |
|---|---:|---:|---:|---|---:|
| 1 d | +0.006 | [−0.000, +0.014] | 0.72 | 0.008/0.004/0.007/0.005/0.007 | 0.004 |
| 3 d | +0.009 | [+0.001, +0.026] | 0.78 | 0.010/0.007/0.009/0.006/0.010 | 0.007 |
| 7 d | +0.007 | [+0.001, +0.030] | 0.76 | 0.010/0.008/0.009/0.010/0.008 | 0.007 |

(LightGBM, local adaptation.) The residual-target tree gives +0.006 / +0.007 /
+0.004 °C at 1 / 3 / 7 d with the same sign pattern. The penalty is small and
consistently signed; per decision rule R1 of the protocol the spatial partition
is reported as a secondary finding, not a headline.

## Distance and hydroclimatic novelty

For each held station the nearest-training-gauge great-circle distance (km) and
a standardized hydroclimatic novelty (Euclidean distance in z-scores of mean
annual air temperature, log drainage area, and |latitude|, fitted on training
stations only) are correlated with the station penalty (descriptive). No strong
gradient is found (LightGBM, local adaptation):

| Horizon | corr(penalty, log distance) | corr(penalty, hydro novelty) |
|---|---:|---:|
| 1 d | −0.07 | +0.16 |
| 3 d | −0.05 | +0.14 |
| 7 d | −0.05 | +0.17 |

## Leave-cluster geometry at HUC2, HUC4, HUC6, and HUC8

| Cluster definition | omitted unit | reportable clusters | effective fraction | largest share | effect (°C) | interval/status | binder row ID |
|---|---|---|---|---|---|---|---|
| *(HUC/network/distance rule)* | *(declared unit)* | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` |

the study remains descriptive (fixed cohort) even if a leave-one-cluster sensitivity
is numerically stable. Route B requires its own prelabel registry, balance gates
and receipt namespace; this table cannot retrofit those conditions.
Every displayed count, balance diagnostic, effect and interval endpoint follows
the README cell-level binder contract.

## Recomputed cluster geometry at HUC2, HUC4, HUC6, and HUC8

Manuscript §3.6 states the cluster ladder in one sentence and this file carries
it in full. The values below are **outcome-free structural facts about the frozen
registry**, computed before any reportability attrition and before any
evaluation label exists. They are not results, and none of them is a
sensitivity of an effect.

Recomputed with the cluster-structure caveat's own `cluster_geometry` function against
`data_usgs/station_registry_v1.csv`, the file the gate itself reads. Source:
`docs/OPTION_A_DESCRIPTIVE_BENCHMARK_SCOPE.md` §1.

| Unit | n_clusters | effective count (1/Σs²) | effective fraction | largest share | Gate |
|---|---:|---:|---:|---:|---|
| **HUC2 (used)** | 15 | 9.536 | 0.6358 | 0.2167 | FAIL (count and fraction) |
| HUC4 | 64 | 32.432 | 0.5068 | 0.1000 | FAIL (fraction) |
| HUC6 | 75 | 36.364 | 0.4848 | 0.1000 | FAIL (fraction) |
| HUC8 | 95 | 72.000 | 0.7579 | 0.0417 | **PASS** all three |

Gate thresholds: `n_clusters ≥ 30`, `effective_cluster_fraction ≥ 0.75`,
`largest_cluster_share < 0.25`. Under HUC2 the cluster sizes are
`[2, 2, 3, 3, 4, 5, 5, 7, 8, 8, 10, 10, 13, 14, 26]` over 120 stations, with
`cluster_size_cv = 0.7569`.

### Why HUC8 was not adopted

HUC8 clears all three thresholds and would change nothing, for two reasons that
must be read together.

1. **It would satisfy the arithmetic while defeating its purpose.** Adjacent
   HUC8 units on the same river are plainly not independent. A partition chosen
   after seeing which partition passes is precisely the failure mode an
   outcome-free gate exists to prevent. HUC2 remains a coarse administrative
   grouping and is never an independent river-network component; a finer
   administrative grouping does not become one.
2. **Eligibility is not computed from the cluster diagnostics at all.**
   `claim_eligible` is a hardcoded literal `False`, and two further gate
   components fail independently of cohort size: both structural assumptions are
   recorded as unmet, and the null-simulation component was never implemented and
   fails closed on every run. A HUC8 partition would clear the cluster gate and
   return the same verdict, `descriptive (fixed cohort, few clusters)`.

The cluster count is therefore the **least** binding of the three reasons this
study is descriptive, and the threshold itself is a recorded specification
error: the Watershed Boundary Dataset defines about 21 HUC2 regions in total
(18 CONUS + 19 Alaska + 20 Hawaii + 21 Caribbean), so `n_clusters ≥ 30` is
unreachable under a HUC2 partition for any U.S. cohort whatsoever. That error was
made when the amendment was written and identified before any evaluation label
was accessed.

### A trap for anyone repeating the recomputation

`data_usgs/station_registry_v1.csv` stores `huc_cd` and `huc2` **without leading
zeros** (`1060003`, not `01060003`; `1`, not `01`), while
`data_usgs/huc_metadata_usgs_v1.csv` preserves them. Slicing `huc_cd[:2]` on the
registry yields **18 clusters, not 15**; 64 of 120 rows are affected. Zero-pad to
eight (or twelve) digits first. The gate itself is unaffected because it reads
the `huc2` column directly, and the unpadded `'1'` and the padded `'01'` are the
same partition.

### Relationship to the figures

Figure 4(a) carries the per-HUC2 medians against both reference models with the
pooled median and the region-weighted mean as reference lines; Figure S7 expands
it with every unit, every leave-one-HUC2 omission, the cluster-share diagnostics,
and the permanent gate status box. The two must agree in value, unit, and
rounding on every per-HUC effect they share. Figure 1(b) renders the ladder above
as an outcome-free geometry panel, with HUC8 marked as passing the arithmetic
while its units are not independent. `per_huc[].huc2` is a cluster label of the
form `HUC2:01` or `UNMAPPED:<site_no>`, never a bare two-digit code, and
`UNMAPPED` units render as themselves rather than being folded into a neighbour.

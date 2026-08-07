# SI11 — spatial and leave-cluster sensitivity

**Status:** development-period analysis; the whole-region holdout has no held-out counterpart (fold weights do not exist) and is reported as development-period-only in Section 4.4.

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

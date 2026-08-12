# SI01 — frozen cohort and registry projection

**Status:** filled. 120-station registry; held-out acquisition is 118 OK / 2 NO_SERIES (03058000, 03544970) with the three transient failures retried successfully (03456991, 04043238, 14150800); 01435000 retains 17 of 1,110 conflicting series days and falls below the 100-target rule; 116 stations are reportable per lead (`outputs/conventional/cohort_2021_2023.csv`).

No registry bytes or outcomes are reproduced in this document. Any redraw value
or target-period count is `[pending computation]` until it is bound
to a verified receipt.

## Frozen PRE geometry

| Fact | Value | Source and role |
|---|---:|---|
| stable USGS sites | 120 | `paper/ThermoRoute_paper.md` §2.1; fixed cohort geometry, not a result |
| calendar span | 2006-01-01 to 2020-12-31 | `paper/ThermoRoute_paper.md` §2.1; development-panel span |
| site-days | 657,480 | `paper/ThermoRoute_paper.md` §2.1; panel geometry |
| states | 34 | `paper/ThermoRoute_paper.md` §2.1; cohort descriptor |
| HUC2 groups | 15 | `paper/ThermoRoute_paper.md` §2.1; outcome-free cluster geometry |
| HUC2 station range before reportability attrition | 2–26 | `paper/ThermoRoute_paper.md` §4.2; cluster-geometry input only |
| largest HUC2 share before attrition | 21.7% | `paper/ThermoRoute_paper.md` §4.2; cluster-geometry input only |
| inverse-Herfindahl effective cluster count | 9.54 | `paper/ThermoRoute_paper.md` §4.2; cluster-geometry diagnostic only |
| minimum reportable-cluster gate | 30 | `paper/ThermoRoute_paper.md` §§1, 4.2; comparison-eligibility requirement |

The frozen cohort has at most 15 HUC2 groups, so it necessarily fails the
minimum-30-cluster component before target outcomes are viewed. This is a PRE
scope fact: study effects are descriptive (fixed cohort), while bootstrap
intervals and sign-flip p-values remain approximate sensitivities.

## Source bindings and projection schema

The canonical references named in the PRE manuscript are:

```text
data_usgs/panel_usgs_120v2.parquet
data_usgs/station_registry_v1.csv
data_usgs/frozen_panel_v1.json
```

This SI intentionally does not assert uninspected file headers. A later
registry-to-SI projection must instead declare this minimum projection schema:

| Field | Meaning | Required binding |
|---|---|---|
| `registry_ref` | registry path and SHA-256 | manifest `source_bindings` entry |
| `panel_ref` | panel path and SHA-256 | manifest `source_bindings` entry |
| `site_id` | stable site identity used in the row | registry contract and cohort binding |
| `huc2` | reportable cluster label | registry contract and cluster-geometry binding |
| `state` | state label used in a descriptive map/table | registry contract |
| `cohort_role` | temporal/external eligibility role | model-suite and receipt binding |
| `count_role` | pre-attrition, reportable, or retained-after-filter count | derivation and source pointer |

`site_id`, `huc2`, and `state` are projection-field names, not claims about the
literal column names in any unread data file.

## Figure/table fill rules

- A HUC2 station-count bar chart may use only receipt- or registry-bound counts;
  before that it must be labelled `registry-derived (redraw TODO)`.
- No map or bar chart may annotate RMSE, skill, national coverage, or a
  superpopulation interpretation.
- A post-opening count still needs a `value_id`, source pointer, derivation,
  rounding rule and gate role; a number visible in a figure is never self-proving.
- The shell below is now filled, and it is filled from the category these rules
  permit: both counts are registry- and cohort-derived, not outcome-derived.
  Pre-attrition counts come from the frozen station registry; reportable counts
  are the stations carrying at least 100 scored keys at *every* lead in the
  primary F0 shards — an intersection across leads, not a union, because a
  station reportable at one day and not at seven is not in the cohort the
  paper's tables describe. Nothing in this table reads a prediction or a score
  (`outputs/final/si01_huc2_projection_v1/`,
  builder `scripts/final/build_si01_huc2_projection.py`).

| HUC2 | pre-attrition station count | reportable station count | binder row ID |
|---|---:|---:|---|
| 01 | 5 | 5 | `SI01.HUC2.01` |
| 02 | 13 | 11 | `SI01.HUC2.02` |
| 03 | 14 | 14 | `SI01.HUC2.03` |
| 04 | 10 | 10 | `SI01.HUC2.04` |
| 05 | 10 | 9 | `SI01.HUC2.05` |
| 06 | 2 | 1 | `SI01.HUC2.06` |
| 07 | 8 | 8 | `SI01.HUC2.07` |
| 09 | 2 | 2 | `SI01.HUC2.09` |
| 10 | 7 | 7 | `SI01.HUC2.10` |
| 11 | 4 | 4 | `SI01.HUC2.11` |
| 12 | 5 | 5 | `SI01.HUC2.12` |
| 14 | 8 | 8 | `SI01.HUC2.14` |
| 16 | 3 | 3 | `SI01.HUC2.16` |
| 17 | 26 | 26 | `SI01.HUC2.17` |
| 18 | 3 | 3 | `SI01.HUC2.18` |
| **Total** | **120** | **116** | `SI01.HUC2.TOTAL` |

Each projected HUC2 label and count follows the README cell-level binder shape;
the pre-attrition and reportable counts have distinct `value_id` objects even
when they share one source registry.

Attrition is not uniform across regions and the paper's inference depends on
which way it went. Four stations are lost — two from HUC2 02, one each from 05
and 06 — so the largest region's share of the reportable cohort *rises* from
21.7% to 22.4%, and HUC2 06 falls to a single station. Attrition therefore
concentrates the cluster structure slightly rather than balancing it, which is
the direction that makes whole-HUC2 resampling more conservative, not less. No
region is emptied.

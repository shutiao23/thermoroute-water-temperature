# SI01 — frozen cohort and registry projection

**Status:** DRAFT / PRE-OPENING / NOT RECEIPT.

No registry bytes or outcomes are reproduced in this document. Any redraw value
or target-period count is `[pending — 探索期数据，不可写入结论]` until it is bound
to a verified receipt.

## Frozen PRE geometry

| Fact | Value | Source and role |
|---|---:|---|
| stable USGS sites | 120 | `paper/ThermoRoute_paper.md` §2.1; fixed cohort geometry, not a result |
| calendar span | 2006-01-01 to 2020-12-31 | `paper/ThermoRoute_paper.md` §2.1; development-panel span |
| site-days | 657,480 | `paper/ThermoRoute_paper.md` §2.1; panel geometry |
| states | 34 | `paper/ThermoRoute_paper.md` §2.1; cohort descriptor |
| HUC2 groups | 15 | `paper/ThermoRoute_paper.md` §2.1; outcome-free cluster geometry |
| HUC2 station range before reportability attrition | 2–26 | `paper/ThermoRoute_paper.md` §4.2; gate input only |
| largest HUC2 share before attrition | 21.7% | `paper/ThermoRoute_paper.md` §4.2; gate input only |
| inverse-Herfindahl effective cluster count | 9.54 | `paper/ThermoRoute_paper.md` §4.2; gate diagnostic only |
| minimum reportable-cluster gate | 30 | `paper/ThermoRoute_paper.md` §§1, 4.2; claim-eligibility requirement |

The frozen cohort has at most 15 HUC2 groups, so it necessarily fails the
minimum-30-cluster component before target outcomes are viewed. This is a PRE
scope fact: Route-A effects are fixed-cohort descriptive, while bootstrap
intervals and sign-flip p-values remain assumption-conditional sensitivities.

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
| `huc2` | reportable cluster label | registry contract and gate binding |
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
- The following shell is deliberately empty of values:

| HUC2 | pre-attrition station count | reportable station count | binder row ID |
|---|---|---|---|
| `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` |

Each projected HUC2 label and count follows the README cell-level binder shape;
the pre-attrition and reportable counts have distinct `value_id` objects even
when they share one source registry.

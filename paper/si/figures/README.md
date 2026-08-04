# Supporting-figure inventory

**Status:** PRE-OPENING. Outcome-free FigS1–FigS3 are materialized; no
receipt-derived result figure is materialized.

This directory records the canonical FigS1–FigS8 responsibilities. It contains
no target result, plotted receipt coordinate or authority to render a POST
figure. Static geometry figures may be generated from bound PRE inputs; all
result figures require the verified opening POST evidence manifest, claim
registry and inference/QC gates.

| Figure | Planned content | Current authority | Materialization gate |
|---|---|---|---|
| FigS1 | cohort ledger, coordinate/HUC2 geometry, counts, and environmental-audit diagnostics | PRE cohort facts only | **MATERIALIZED** from hash-bound registry/panel/rejection/audit evidence; QA complete |
| FigS2 | chronology, issue-time boundary, source matrix, and predictor bridge | PRE design and outcome-free bridge evidence | **MATERIALIZED** with bridge capability/limitation parity; QA complete |
| FigS3 | complete PRE model and calibration architecture | PRE design only | **MATERIALIZED** from asserted protocol/configuration/implementation facts; QA complete |
| FigS4 | all-model station-level effects/scores | none | SI07 exact-common-key evidence |
| FigS5 | interval, probability and reliability diagnostics | none | SI08 probability evidence and bin map |
| FigS6 | temporal coverage, missingness and attrition | none | SI10/SI14 evidence with declared denominators |
| FigS7 | spatial and leave-cluster sensitivities | none | SI11 registry/UQ evidence and gate status |
| FigS8 | QC, external-arm and failure-case diagnostics | none | SI12/SI13/SI14 evidence with scope labels |

Every rendered result coordinate must resolve to one manifest `value_id` with a
unit, evidence role, source pointer, derivation and rounding rule. Route A stays
fixed-cohort descriptive; FigS8 must label the external arm history-dependent
and not ungauged. A missing figure is preferable to a development-cache fill.

## Reproducible PRE renderer and artifacts

Run from the repository root:

```bash
python paper/si/figures/render_pre_supporting_figures.py
```

The renderer uses Matplotlib 3.8-compatible APIs and PyArrow for the frozen-panel
identity check. It fixes the SVG hash salt and file metadata, sorts data-derived
geometry, and fails closed unless the registry, panel, rejection ledger,
environmental audit, predictor-bridge chain, and model-architecture assertions
match their bound PRE sources. S2 chronology/horizons are AST-projected from the
configuration and checked against the protocol; S3 quantiles, missing-mask state,
and 2018 CQR/Platt contracts are likewise source-projected. Semantic colors are
bound to the exact palette tokens in `paper/FIGURE_REDRAW_SPEC.md`; allowed and
verified paths use `ALLOWED_TEAL`, independently of the categorical LSTM green.
Each 140-mm-wide figure is delivered as vector SVG, embedded-TrueType vector PDF,
and 300 dpi PNG on white. Body text targets 8 pt; the hard final-size floor is
7.5 pt.

| Figure | SVG / PDF / PNG basename | Dimensions |
|---|---|---:|
| FigS1 | `figS1_cohort_registry` | 140 × 156 mm |
| FigS2 | `figS2_information_boundary` | 140 × 182 mm |
| FigS3 | `figS3_model_architecture` | 140 × 122 mm |

`figS1_cohort_registry_data.csv` binds one `site_no`/state/lat/lon/HUC2 row and
distinct site/state/x/y/category `value_id`s to each of the 120 visible coordinate markers.
The 15 HUC2 categories use unique combinations from eight shapes crossed with
filled/open state; color is only an auxiliary cue, preserving grayscale identity.
`pre_supporting_figures_manifest.json` is the mark-level value binder: every
value object declares value, unit, evidence role, source pointer, derivation, and
rounding. It also binds the environmental audit; predictor-bridge manifest,
panel, registry, report, request map, three raw indexes, and two normalized
tables. The renderer also traverses the indexes’ 241 unique request records and
verifies containment, existence, SHA-256, and byte count for all 482 response and
metadata leaves. The bridge `source_tree_sha256` is explicitly frozen
implementation provenance, not a raw-leaf tree digest. The manifest also binds
protocol/configuration/model sources; all artifacts; panel order; caption values;
scope status; semantic palette; and render profile. No basemap is used, map-source rights
are therefore not applicable, and the coordinate projection reconciles exactly
to the 120-row registry. Long-form interpretation limits are in
`FigS1-S3_captions.md`.

## QA record

- Frozen-input SHA/count/site-identity gate: **PASS**.
- Environmental-audit SHA/selection/geography gate: **PASS**.
- Predictor-bridge nested path/SHA/status/field/limitation/outcome gate: **PASS**.
- Raw snapshot index/version/count/uniqueness and 482-leaf path/hash/byte gate:
  **PASS**.
- S2 config/protocol chronology, horizon, issue-cutoff, vintage, and provider
  source assertions: **PASS**.
- Architecture variable-order/WLEVEL/context/router/TCN/delta assertions: **PASS**.
- S3 quantile/missing-mask/calibration-period/CQR/Platt/erratum assertions: **PASS**.
- Figure 1/SI shared canonical `value_id` value-and-unit parity: **PASS**.
- Automated text-canvas containment and registered 2-mm module-inset checks:
  **PASS**.
- Minimum visible SVG stroke width ≥0.6 pt and 15-way grayscale marker encoding:
  **PASS**.
- SVG XML parse and accessibility title/description check: **PASS**.
- Minimum nominal SVG text size ≥7.5 pt: **PASS**.
- `rsvg-convert` rasterization and page-level visual inspection: **PASS**.
- PDF font inspection: **PASS**; DejaVu Sans and DejaVu Sans Bold are embedded
  CID TrueType fonts (`pdf.fonttype=42`).
- Two consecutive full renderer runs produced identical SVG, PDF, PNG, marker
  CSV and manifest SHA-256 values: **PASS**.

FigS1 is a coordinate scatter rather than a national-coverage map and makes no
hydraulic-connectivity inference. FigS2 states the exact allowed/excluded
information contract: its exact bridge covers only the five meteorological
fields and establishes neither as-issued/local-day equivalence nor target-period
availability. FigS3 contains no fitted weight, performance result, truth-error or
safety bound, or physical-routing interpretation.

FigS4–FigS8 remain closed behind the receipt/evidence gates in the inventory
table. No development cache may fill them.

# Figure 1 restructure re-render (2026-08-06)

| Field | Value |
| --- | --- |
| Date | 2026-08-06 |
| Scope | Re-render Figure 1 (`fig01_preopening_concept`) to the restructured spec and relocate the bounded-correction schematic to Figure S3(c). Paper/figures only; no `src/`, `scripts/`, `tests/`, `protocols/`, `pyproject.toml`, or `requirements*` byte changed. |
| Spec | `paper/FIGURE_REDRAW_SPEC.md` §4 (Figure 1), §6.4, and the §5 Figure-S3 relocation note (~line 835). |
| Authoritative inputs (frozen, retrain-independent) | `data_usgs/station_registry_v1.csv`, `data_usgs/panel_usgs_120v2.parquet`, `data_usgs/development_environmental_audit_v1.json`, `src/thermoroute/config.py`, `protocols/route_a_inference_amendment_v2.json`, `protocols/route_a_confirmatory_v1.json`, `protocols/route_a_claim_registry_v1.json`. No model prediction or 2021–2023/target value was read. |

## What changed

Panel reassignment per §6.4:

- **New (a)** = station map: 120 retained gauges on a CONUS coordinate scatter
  (no basemap, `NO_BASEMAP`), coloured by HUC2 with a 15-way unique
  (marker, fill) encoding mirroring Figure S1, sized by retained 2006–2015
  observed-WTEMP day count. Inset: zero-based nearest-neighbour-distance
  histogram annotating the 10 km mark (19 stations) and the 289 km
  whole-region-holdout mean.
- **New (b)** = cluster geometry vs the claim gate (was old panel c), extended
  with the HUC2/4/6/8 ladder (15/64/75/95 clusters; effective fractions
  0.636/0.507/0.485/0.758; HUC8 marked *passes the arithmetic; adjacent units
  on one river are not independent*).
- **New (c)** = persistence challenge (was old panel a): station median |Δ_h T|
  by horizon from observed exact-day pairs in 2006–2015.
- The bounded-correction tanh schematic (old panel b) was **removed** from
  Figure 1 and **relocated to Figure S3(c)** as an anchor line with a shaded
  A±δ envelope carrying the in-panel warning
  *“Deviation from anchor; not an error or safety bound.”*

## Binding decisions recorded

1. **289 km whole-region-holdout mean.** The Stage-13c region-transfer table
   (`MC/outputs/reports/region_transfer.md`) is not present in this tree, so the
   289 km value is bound as a **declared structural-geometry** value
   (`fig01.a.whole_region_holdout.mean_nearest_training_gauge_km`,
   `PRE_STRUCTURAL_GEOMETRY`) with a spec + Option-A source pointer, not as a
   hashed file source. It is never presented as a score.
2. **Marker size period.** Panel-(a) marker size uses the **2006–2015
   training-period** finite-observed-WTEMP day count, keeping panel (a) inside
   its declared PRE-structural evidence period (no 2016–2020 development data).
3. **Nearest-neighbour diagnostics.** Per-station great-circle (haversine)
   nearest-neighbour distances are computed from the frozen registry
   coordinates and validated against the frozen environmental audit
   (median 53.856…, minimum 0.7518…, 19 stations within 10 km).
4. **HUC2/4/6/8 ladder.** Ladder values are taken verbatim from
   `docs/OPTION_A_DESCRIPTIVE_BENCHMARK_SCOPE.md` §1 (the gate’s own
   `cluster_geometry`), which is bound as a hashed source.
5. **Layout.** Panel (b) is the content-heavy panel (15 bars + 3 gate gauges +
   ladder + excluded-input band + evidence spine) and is given the full-width
   bottom row; panels (a) and (c) share the top row. Panel labels (a)/(b)/(c)
  are bold and the caption orders them; the L-arrangement is the only
  legible option at 140 × 92 mm with a 7.5 pt body floor.
6. **Frozen-input integrity.** Every frozen-input SHA-256 check
   (panel, registry, audit, config, amendment, protocol) and the cohort
   geometry checks (120 stations, 15 HUC2, `EXPECTED_HUC_COUNTS`) are retained.
   The cross-figure value check (Figure 1 ↔ S1/S2/S3) passes for all 36 shared
   `route_a.*` value IDs.

## Verification

- `py_compile` of both renderers passes.
- Figure 1: SVG well-formed; PDF 140 × 92 mm with embedded TrueType fonts;
  visible-stroke guard ≥ 0.6 pt; binder 1,673 values / 520 marks; CSV 360 rows.
- Figure S3: text-layout guard passes; minimum nominal text 7.5 pt; the
  relocated schematic mark `figs3.bounded_correction_schematic` is bound.
- Both PNGs visually confirmed (vision describe + OCR): labels legible, no text
  overlap, panel content matches the (a)/(b)/(c) assignment, colour encoding
  redundant (marker shape + fill).

## State token

Figure 1 state token moved from `PRE_MATERIALIZED_REDRAW_STALE` to
`PRE_MATERIALIZED` in `paper/FIGURE_REDRAW_SPEC.md`, `paper/si/SI00_inventory.md`,
and `docs/PAPER_FIGURE_SI_RECONCILIATION.md`.

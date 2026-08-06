# AGU Figure 1

`render_fig01_preopening_concept.py` deterministically renders the 140 × 92 mm,
three-panel proposition figure:

1. observed training-period thermal mismatch;
2. the anchor-bounded correction mechanism; and
3. registry geometry, the cluster-structure caveat, and the resulting claim boundary.

Run from the repository root:

```bash
python paper/agu_submission/figures/render_fig01_preopening_concept.py
```

The renderer writes:

- `fig01_preopening_concept.svg` — primary vector artwork;
- `fig01_preopening_concept.pdf` — embedded-font submission/render check;
- `fig01_preopening_concept.png` — 300 dpi review raster;
- `fig01_preopening_concept_data.csv` — one traceability row per plotted
  station/horizon marker, including stable `mark_id`, `site_value_id`,
  `x_value_id`, `y_value_id`, and `n_pairs_value_id`; and
- `fig01_preopening_concept.json` — the `thermoroute.figure-binder.v2` values
  registry and mark/cell binder, plus formulas, cohort geometry, gate
  evaluations, caption, source/output SHA-256 bindings, render profile, and scope
  qualifiers.

Panel a uses only exact-day observed `WTEMP` pairs with both endpoints in
2006-01-01 through 2015-12-31. It first takes the median absolute change within
each station/horizon and then plots the 120 stations with equal visual weight.
Its y-axis is the fixed publication scale 0--2.5 °C with 0.5 °C ticks,
declared independently of the observed extrema; the renderer fails if a station
point falls outside that scale. The axis minimum, maximum, and tick interval each
have a bound value ID and share a dedicated axis-scale mark.
Panel c counts HUC2 membership directly from the frozen station registry. The
renderer fails if the expected 657,480-row / 120-station / 15-HUC2 geometry
drifts. It also fails if the frozen panel or registry SHA changes, if panel
`site_id` values differ from registry `legacy_site_id` aliases, or if the binder
does not contain exactly 360 station marks with separately bound site, x, y, and
pair-count cells. The renderer validates the frozen `config.py` and inference
amendment SHA values, parses `DELTA_SCALE` with Python's AST, parses the three
gate thresholds and failed-gate verdict from the amendment, and rejects any
semantic drift before drawing. It also SHA-binds
`protocols/route_a_confirmatory_v1.json` and verifies the exact historical-input
cutoff plus the frozen `horizon_specific_future_nwp_consumed = false` and
`operational_replay_claim_allowed = false` fields. Panel c therefore binds and
marks `target WTEMP at t+h` and `horizon-specific future weather` independently
as excluded predictor inputs.

Cross-figure cohort facts use canonical value IDs so main and supporting figures
can bind the same objects directly: `route_a.panel.row_count`,
`route_a.registry.station_count`, `route_a.registry.huc2_group_count`, and
`route_a.registry.huc2_<code>.station_count`.

The dated covariates are latest-available retrospective products, not as-issued
operational vintages. This limitation and the fixed-cohort descriptive claim
boundary are preserved in the JSON caption and scope fields rather than expanded
into dense on-figure prose.

All visible SVG strokes are checked after accessibility metadata insertion and
must be at least 0.6 pt. The render profile also records the exact redraw palette:
teal `#008C7A`, ink `#202020`, grid `#D0D0D0`, pale blue `#DCEAF4`, pale
vermilion `#F9E3D6`, and pale teal `#DCEFEA`. Panel c additionally enforces 2 mm
internal padding around the excluded-input labels and 2 mm separation between
their header band and both the bar axes and all 15 bar-top labels.

Minimal QA commands:

```bash
python -m py_compile paper/agu_submission/figures/render_fig01_preopening_concept.py
xmllint --noout paper/agu_submission/figures/fig01_preopening_concept.svg
rsvg-convert -o /tmp/fig01.png paper/agu_submission/figures/fig01_preopening_concept.svg
pdfinfo paper/agu_submission/figures/fig01_preopening_concept.pdf
pdffonts paper/agu_submission/figures/fig01_preopening_concept.pdf
```

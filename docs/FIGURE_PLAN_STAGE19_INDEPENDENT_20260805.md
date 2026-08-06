# Stage-19-independent figure plan

**Date:** 2026-08-05
**Status:** DECIDED — skeletons revised, spec updated, no figure rendered.
**§5 partially superseded 2026-08-06.** The benchmark restructure
(`docs/PAPER_BENCHMARK_RESTRUCTURE.md`) reassigned the four main figures and
added Figure S10, so the nine-figure manifest in §5 and the numbering used in §3,
§4 and §7 no longer match the submission. The authoritative manifest is now
`paper/FIGURE_REDRAW_SPEC.md` §4–§5, with the panel-by-panel delta in its §6.4
and in `docs/PAPER_FIGURE_SI_RECONCILIATION.md`. Everything else in this
document stands unchanged and is still the authority for it: the corrected
Stage-19 facts (§1), the skeleton audit (§2), the determination that the trusted
scorer supplies target-period interval evidence (§3), the pre-opening degeneracy
guard (§4a), the exact POST artifact paths, the render-ordering rationale (§6),
and the AGU size targets (§8.1). Where this document says "Figure 3", read
"Figure 4(c) and Figure S5"; where it says "Figure 4", read "Figure S10, Figure
S6(a), Figure 4(b) and Figure S8(b)".
**Supersedes:** the interim Stage-22 rebinding drafted earlier the same day; see §3.
**Scope:** the POST-gated figures (Figures 2–4, S4–S9) managed by
`paper/agu_submission/figures/render_post_main_figures_skeleton.py` and
`paper/si/figures/render_post_supporting_figures_skeleton.py`.
**Companion:** [`STAGE19_DEGENERATE_INTERVAL_DISPOSITION_20260805.md`](STAGE19_DEGENERATE_INTERVAL_DISPOSITION_20260805.md)
(why Stage-19 is not produced). This document covers only what the figures do
about it.

---

## 1. Corrected Stage-19 facts

Recorded verbatim from direct measurement of
`outputs/predictions/usgs_predictions_with_perstation_v2.parquet`. These numbers
supersede every earlier project note.

| Quantity | Measured value |
|---|---|
| Member-level rows with complete quantile heads | 26,993,675 |
| **Strict quantile-ordering violations** (`q05 > q50` ∨ `q50 > q95` ∨ `q05 > q95`) | **0** |
| Rows tripping the Stage-19 contract | **135** |
| Nature of those rows | `q05 == q50 == q95` bit-identical float64 — zero-width nominal interval |
| Rate | 135 / 26,993,675 = 0.0005 % (≈ 1 in 200,000) |
| Attribution | LightGBM 123, LightGBM-perstation 12 |
| Maximum monotonicity violation | exactly 0.000 °C |
| Cause | the frozen check in `scripts/19_probabilistic.py` uses `q05 >= q95` — equality included — and the protocol forbids evaluation-time repair |
| Effect on delivered intervals | none: split-CQR would have widened all 135 rows, so no delivered interval is degenerate |

**There is no quantile crossing.** Zero strict crossings were observed. This fact
is going into the manuscript's Limitations section, and the figure documentation
must not contradict it.

**Wording ban.** The phrase "quantile crossing" must not appear in any figure
caption, figure spec, or renderer code in reference to this issue. The correct
term is **zero-width (degenerate) nominal interval**. Both skeletons now list
`"quantile crossing as the Stage-19 cause"` in the `prohibited_semantics` tuple
of the affected figures.

**Disposition (unchanged and binding):** Stage-19 is not produced for this
submission, because fixing the contract means editing `scripts/`, which changes
`source_tree_hash` and invalidates all four training receipts (~2 days of
recompute). Stage-10 validates the Stage-19 receipt before reading any table and
therefore cascades.

---

## 2. Audit of the claimed skeleton features

Audited against the code as committed, not against the description.

| # | Claimed feature | Verdict | Evidence |
|---|---|---|---|
| 1 | Per-figure panel manifest + chart type + binding contracts | **Holds** | `PanelSpec(panel_id, plot_type, contract)` present for all 8 figures |
| 2 | Required value IDs declared per figure | **Holds** | `FigureSpec.required_value_ids` populated for all 8 |
| 3 | `shares_value_ids_with` **cross-checks** | **Does not hold** | The field existed; no code anywhere read it. No cross-check was performed. Now enforced by `validate_manifest()` (namespace validity, no self-reference) and implemented by `ValueBinder.cross_check_shared_value_ids()` (value + unit + rounding equality) |
| 4 | Dependency table mirroring `model_suite.py` constants | **Holds** | Verified: `STAGE9/16/25_COMPLETION_RECEIPT_PATH` in `src/thermoroute/model_suite.py:145,153,172`; `STAGE09B_COMPLETION_RECEIPT_PATH` in `src/thermoroute/development_controls_gate.py:87`; opening-receipt glob matches `src/thermoroute/opening.py:1226,1278`. All four paths correct |
| 5 | SI06–SI14 receipt docs bound | **Holds** | All nine referenced receipt documents exist in `paper/si/` |
| 6 | `declared_later` markers for POST-only paths | **Holds, but harmful** | `resolve()` returned `False` unconditionally, making a placeholder indistinguishable from a genuinely missing dependency — and it, not the `blocked_by_stage19` flag, is what actually blocked fig03/figS5. Removed; replaced by real paths plus an explicit `Severity` |
| 7 | Forbidden-semantics lists | **Holds as data** | Present and now extended; never machine-enforced, which is acceptable since enforcement is a POST caption/QA task, but it is a declaration and not a control |
| 8 | `blocked_by_stage19` on fig03/figS5 | **Holds as a field; misleading as a mechanism** | The refusal came from the `declared_later` Stage-19 dependency. `blocked_by_stage19` only appended a **duplicate** failing row. Clearing the flag alone would not have unblocked either figure |
| 9 | Double gate (spec `POST_TEMPLATE_ONLY` → refuse; opening receipt missing → refuse; any dependency missing → refuse) | **Holds** | All three refusal paths verified |
| 10 | Exits 2 writing no files and creating no axes | **Partially holds** | No file or axes is ever created — correct. But only `PostGateNotPassed` was caught in `main()`; `PanelBuilderNotImplemented` escaped as an uncaught traceback with **exit 1**. Fixed: both are caught and exit 2 |
| 11 | `--status` read-only gate report | **Holds, but was inaccurate** | Read-only and renders nothing — correct. It reported `stage09b/16/25` and Stage-13c as *missing* when they exist in the multicore worktree, because every path resolved against the paper tree only. Fixed by a configurable evidence root |
| 12 | matplotlib imported only after the gate passes | **Vacuously holds** | matplotlib is never imported anywhere in either file; only a comment asserts the rule. True, but untested and undemonstrated |
| 13 | `NotImplementedError` panel-builder stubs | **Does not hold as stated** | There are no stubs — `PANEL_BUILDERS` is an empty dict — and the exception is a custom `PanelBuilderNotImplemented(RuntimeError)`, not `NotImplementedError`. The refusal behaviour is equivalent; the description is inaccurate |
| 14 | `ValueBinder` with the same semantics as the PRE renderer | **Partially holds** | `value()`/`cell()` reproduce the PRE closure (`render_pre_supporting_figures.py:1652–1677`) exactly, including error strings. But (a) the PRE binder is a closure, not a class; (b) the skeleton **omitted `mark()`**, the PRE function that routes visible cells through `cell()` — without it, "every visible mark binds a declared `value_id`" cannot be enforced; (c) `ValueBinder` was never instantiated and `require_contract_ids()` never called anywhere: entirely dead code. All three fixed |

### 2.1 Additional defects found during the audit

| Defect | Location | Fix |
|---|---|---|
| `assert target is not None` reachable for any unknown dependency kind | main skeleton `Dependency.resolve` | explicit `ManifestError` |
| State regex `([A-Z_]+)` cannot match a token containing a digit | both `spec_state` | widened to `[A-Z0-9_]+` |
| Refusal test was exact-match `state != "POST_TEMPLATE_ONLY"`, so **any** new state token would silently pass the spec gate | both `gate_report` | substring test on `TEMPLATE_ONLY`; verified against the new `POST_TEMPLATE_ONLY_DEVELOPMENT` token, which still refuses |
| figS7 bound a non-existent `declared_later` "13c POST binding" while the real artifacts existed | SI skeleton | bound to `outputs/tables/region_transfer.csv` (required) and `outputs/reports/region_transfer.md` (qualifier) |
| No AGU geometry/type/format constraint expressed in code | both | `RenderProfile` with a `validate()` that rejects non-AGU widths, sub-7.5 pt type, non-vector output, unembedded fonts, and colour-alone encoding |
| POST dependency paths were invented placeholders | both | replaced with the **exact** artifact names from the `opening.py:1226-1281` state table; nine `trusted/` artifacts now bound by real name |

### 2.2 Defect-fix confirmation

All six defects the coordinator asked about are fixed and covered by the
41-check harness in §7:

| Defect | Status | Where |
|---|---|---|
| `ValueBinder` dead code (never instantiated, `require_contract_ids` never called) | **Fixed** | instantiated in `render_figure`; `require_contract_ids()` and `require_scope_band()` called after panel builders; `mark()` added to match the PRE closure |
| `shares_value_ids_with` never read | **Fixed** | `validate_manifest()` checks namespace validity and self-reference; `ValueBinder.cross_check_shared_value_ids()` compares value, unit, and rounding |
| `PanelBuilderNotImplemented` escaping as exit 1 | **Fixed** | caught in `main()` alongside `PostGateNotPassed`; both exit 2 writing nothing |
| State regex could not match a digit | **Fixed** | `[A-Z0-9_]+` |
| Exact-match state test would let a new token pass | **Fixed** | substring `TEMPLATE_ONLY` test |
| Reachable `AssertionError` in `Dependency.resolve` | **Fixed** | explicit `ManifestError` on unknown kind |

---

## 3. Determination: does the opening supply target-period interval evidence?

**Question posed 2026-08-05 (coordinator):** the manuscript rewrite created a
Results slot for Table 4.4 "interval behaviour". Does the opening receipt /
trusted scorer supply target-period interval coverage, width, and interval score
at the granularity a main-text figure needs?

### 3.1 Answer: YES — and it supplies considerably more than that

Evidence, all read-only:

**`src/thermoroute/opening.py:8932-8948`** defines the metric family the trusted
scorer computes:

```
coverage_90, mean_interval_width_c,
pinball_q05_c, pinball_q50_c, pinball_q95_c,
equal_weight_three_quantile_pinball_mean_c,
brier_score, frozen_reference_brier_score, brier_skill_frozen_seasonal,
log_loss, auroc, auprc, ece_10_equal_width,
calibration_intercept, calibration_slope, event_rate
```

**Granularity** (`opening.py:9053-9060`): the emitting loop is
`for cohort in ("temporal","external")` × `for model in expected_models` ×
`for horizon in (1,3,7)`. That is cohort × model × horizon — exactly the grid
Figure 3(a) and Figure S5(a) specify.

**Reliability bins** (`opening.py:8920-8926`): station-balanced bins exist, their
`station_balanced_weight` must sum to 1.0 within 1e-12, and ECE is returned
alongside. Figure 3(c)-(e) and S5(c) are satisfiable.

**Frozen event reference** (`opening.py:9010-9016`): the reference climatology is
validated by `validate_frozen_seasonal_event_reference` with a recorded
`fit_interval` and `fit_observation_count`, per-site for the temporal cohort and
pooled for the external cohort. Figure 3(b)'s "Brier skill against the frozen
seasonal reference" is satisfiable, and the spec's prohibition on using
confirmation-period prevalence as the reference is structurally enforced.

**Reportability** (`opening.py:9060+`): per-site `n_valid_targets` from the
availability registry, against the protocol's
`minimum_valid_targets_per_station_horizon = 100`.

**Period** (`protocols/route_a_confirmatory_v1.json`):
`time_holdout.primary_target_start = 2021-01-01`; the primary estimand's
`admissible_issue` runs 2021-01-01 through 2023-12-31.

**Artifact** (`opening.py:1264-1265`):
`outputs/confirmatory/route_a_<namespace>/trusted/probabilistic_evaluation_v2.json`.

### 3.2 Why this was not obvious earlier

`scripts/19_probabilistic.py` and the opening's trusted scorer compute
overlapping metric families on *different periods* by *different code paths*.
Stage-19 is a development-period tool; the trusted scorer is the target-period
authority and runs inside the one-time opening. The withdrawal of Stage-19
removes development-period probability diagnostics only. Everything Figure 3 and
Figure S5 were originally specified to show is a target-period quantity and is
therefore unaffected.

The interim plan drafted earlier the same day assumed Stage-19 was the only
producer of these metrics. That assumption was wrong, and the interim rebinding
is **superseded**.

### 3.3 Resulting action

- **Figure 3** — restored to its original five-panel target-period design
  (coverage–width plane, Brier skill, three reliability panels), bound to
  `trusted/probabilistic_evaluation_v2.json`. Stem returns to
  `fig03_intervals_probability`. Nothing is dropped. All main-text figures are
  now target-period and the Results narrative is internally consistent.
- **Figure S5** — restored to its original four-panel expanded design
  (model × horizon coverage/width matrix, score matrix, reliability bins,
  calibration/discrimination), same binding. Nothing is dropped.
- **Figure S9 (new)** — carries the Stage-22 development-period conformal
  sensitivity built for the interim plan, with every guard retained:
  `EvidencePeriod.DEVELOPMENT`, a mandatory in-panel scope band, and
  prohibited-semantics entries forbidding target-period reading and forbidding
  numerical comparison against Figures 3 or S5.

### 3.4 Why S9 is a separate figure rather than a panel of S5

The coordinator's instruction was to move the development conformal diagnostics
into S5. That was correct under the assumption that S5 would otherwise be empty.
Since S5's original target-period content turns out to be available, folding
2019–2020 development evidence into the same figure as 2021–2023 target results
would recreate precisely the authority-layer hazard the interim plan existed to
avoid — a reader comparing a coverage number across panels would be comparing
two different cohorts.

The governing rule, now written into the spec: **one figure never mixes two
evidence periods.** S5 expands Figure 3 (spec §8: "SI figures expand rather than
duplicate main figures"); S9 stands alone as a development-period sensitivity.

**S9 is optional.** Nothing in the confirmatory five-test family depends on it,
and the probability-metric erratum already classifies the whole probabilistic
family as `DESCRIPTIVE_NOT_IN_CONFIRMATORY_FIVE_TEST_FAMILY` with
`inference_allowed: false`. If page budget is tight, drop S9 without weakening a
registered claim.

---

## 4. Nothing is dropped

The interim plan proposed dropping Figure 3(b), Figure 3(c)–(e), and three
Figure S5 panels for want of a Stage-19-independent substitute. **All of those
panels are restored.** The trusted scorer supplies event probabilities, Brier
skill against a frozen seasonal reference, reliability bins with denominators,
calibration slope and intercept, AUROC, AUPRC, and ECE at target period.

No figure or panel is dropped from the submission set. The model dimension is
retained. The only addition is optional Figure S9.

---

## 4a. Pre-opening blocking risk (new, and more serious than any figure issue)

`src/thermoroute/opening.py:7093-7094` applies a **strict** `q05 < q95` to the
member-averaged nominal heads and raises `OpeningContractError` — aborting the
entire one-time opening — on violation:

```python
or not ((q05 <= q50).all() and (q50 <= q95).all())
or not (q05 < q95).all()
```

This is the same degeneracy condition that stopped Stage-19, one layer up, and
the opening is one-shot.

**Measurement (read-only, development panel, 27,740,891 rows):**

| Quantity | Value |
|---|---|
| Member-level rows with `q05 == q95` | 135 (LightGBM 123, LightGBM-perstation 12) |
| Distinct forecast keys affected | 120 |
| Keys still degenerate **after member averaging** | **12** |
| Attribution of those 12 | **all** `LightGBM-perstation`, which has 1 member |
| LightGBM keys surviving averaging | **0** — all 108 cleared by the 5-member mean |

`LightGBM-perstation` is absent from `model_suite.PRIMARY_MODELS`
(`Persistence, DampedPersistence, Climatology, LightGBM, LSTM, ThermoRoute`) and
the string "perstation" appears **0** times in
`protocols/route_a_confirmatory_v1.json`. On development data, no model in the
opening's registry would trip the check.

**This is reassuring, not conclusive:** the target period (2021–2023) is
different data, and a 5-member mean can in principle still tie exactly at an
ice-affected site.

**Recommended action before the one-time opening:** run the same read-only
member-averaged check against the target-period predictions. It costs minutes and
guards a one-shot, ~2-day-to-recover operation. This is recorded as a provenance
qualifier on Figure 3 so it cannot be lost.

## 5. Final figure manifest

Nine POST-gated figures (eight required + one optional). Widths are the AGU
targets from `FIGURE_REDRAW_SPEC.md` §3.2: full width 140 mm (=
`agujournal2019.cls` `\textwidth 5.5in` = 139.7 mm), single column 85 mm.

All POST dependency paths below are the **exact** artifact names from the state
table in `src/thermoroute/opening.py:1226-1281`; only the run namespace is
globbed. They are fail-closed guards, not placeholders.

| Fig | Stem | Panels | Period | Required dependencies | Qualifier (non-blocking) |
|---|---|---|---|---|---|
| 2 | `fig02_point_performance` | 4 | target | opening receipt; Stage-09; SI06; SI07; confirmatory protocol; `trusted/temporal_predictions_v1.parquet`; `trusted/availability_registry_v1.csv`; `trusted/statistics_v1.json` | — |
| 3 | `fig03_intervals_probability` | 5 | target | opening receipt; `trusted/probabilistic_evaluation_v2.json`; `trusted/temporal_predictions_v1.parquet`; `trusted/availability_registry_v1.csv`; erratum contract; confirmatory protocol; SI08 | Stage-19 withheld (informational); pre-opening degeneracy guard |
| 4 | `fig04_mechanism_boundary` | 4 | target | opening receipt; Stage-09; Stage-16; Stage-25; SI10; SI14; `trusted/temporal_predictions_v1.parquet`; `trusted/external_predictions_v1.parquet`; `trusted/temporal_coverage_audit_v1.json` | Stage-09b (development-only; SI09 only) |
| S4 | `figS4_point_heterogeneity` | 3 | target | opening receipt; Stage-09; SI06; SI07; confirmatory protocol; `trusted/temporal_predictions_v1.parquet`; `trusted/availability_registry_v1.csv`; `trusted/statistics_v1.json` | — |
| S5 | `figS5_probability_diagnostics` | 4 | target | opening receipt; `trusted/probabilistic_evaluation_v2.json`; `trusted/temporal_predictions_v1.parquet`; `trusted/external_predictions_v1.parquet`; `trusted/availability_registry_v1.csv`; erratum contract; SI08; confirmatory protocol | Stage-19 withheld (informational) |
| S6 | `figS6_temporal_attrition` | 4 | target | opening receipt; SI10; SI14; `trusted/temporal_coverage_audit_v1.json`; `trusted/availability_registry_v1.csv` | — |
| S7 | `figS7_spatial_leave_huc2` | 4 | target | opening receipt; SI11; `trusted/spatial_sensitivity_v1.json`; Stage-13c region-transfer table | Stage-13c report |
| S8 | `figS8_qc_external_failures` | 3 | target | opening receipt; Stage-25; SI12; SI13; SI14; `trusted/outcome_qc_gate_v1.json`; `trusted/outcome_quality_audit_v1.json`; `trusted/external_predictions_v1.parquet` | — |
| **S9** *(optional)* | `figS9_conformal_sensitivity_development` | 4 | **development 2019–2020** | opening receipt; Stage-22 row table; Stage-22 report; confirmatory protocol | evidence-period band; span record |

Eight of nine are target-period. S9 is the only development-period figure and is
the only one carrying a mandatory in-panel scope band.

All figures share one render profile: vector PDF + SVG, embedded fonts, ≥ 8 pt
body text with a 7.5 pt absolute floor, ≥ 0.6 pt strokes, white background,
colourblind-safe Okabe–Ito semantic palette, redundant non-colour encoding on
every series. `RenderProfile.validate()` rejects any violation, including the
non-AGU 95 mm and 190 mm widths.

PRE figures (1, S1–S3) are already materialized and outside this plan.

---

## 6. POST rendering order

Ordered so shared value IDs are minted by their owning figure before any consumer
reads them.

| Step | Action | Rationale |
|---|---|---|
| 0 | **Pre-opening degeneracy check** on target-period predictions (§4a) | guards a one-shot operation against `OpeningContractError` |
| 1 | One-time opening → `outputs/confirmatory/route_a_*/` | every figure's first required gate; materializes all nine `trusted/` artifacts |
| 2 | Point `--evidence-root` / `THERMOROUTE_FIGURE_EVIDENCE_ROOT` at the worktree that ran the opening | receipts and tables are not in the paper tree |
| 3 | Flip spec state tokens for the figures being rendered | skeletons refuse on the substring `TEMPLATE_ONLY` |
| 4 | **Table T2** | mints `forest_mark.*` / `t2_companion.*` reused by Figure 2 |
| 5 | **Figure 2** | owns all-model and forest value IDs; cross-checks against T2 |
| 6 | **Figure S4** | consumes Figure 2 / T2 IDs; every retained/NA station reconciles |
| 7 | **Figure 3** | owns the `coverage_90` / `brier_*` / `reliability_bin.*` IDs |
| 8 | **Figure S5** | expands Figure 3; must agree in value, unit, and rounding |
| 9 | **Figure 4** | independent except for shared cohort counts |
| 10 | **Figure S6** | consumes Figure 4 attrition/temporal IDs |
| 11 | **Figure S7** | needs SI11 + `trusted/spatial_sensitivity_v1.json` + Stage-13c |
| 12 | **Figure S8** | needs Stage-25 + SI12–SI14 + QC artifacts |
| 13 | **Figure S9** *(optional)* | Stage-22 only; evidence already complete; no upstream dependency |
| 14 | Render receipt + visual QA | font embedding, bounding boxes, grayscale and colour-vision simulation, placed-size type check |

---

## 7. Verification

```
$ python -m py_compile paper/agu_submission/figures/render_post_main_figures_skeleton.py \
                       paper/si/figures/render_post_supporting_figures_skeleton.py
COMPILE OK
```

41 behavioural checks pass, covering: manifest validation for both modules;
rejection of 95 mm and 190 mm widths, 6 pt type, colour-alone encoding,
unembedded fonts, and raster-only output; every figure refusing to render under
real roots; Figures 3 and S5 being target-period with no dropped panels and a
**required** binding to `trusted/probabilistic_evaluation_v2.json`; those two
figures refusing on spec state, missing opening receipt, and missing target-period
probability artifact; Figure S9 being development-period with a mandatory scope
band, a recorded span, a target-period prohibition, and refusal when the Stage-22
table is absent; every development-period figure carrying a scope band; POST paths
matching the `opening.py` state table; and the binder rejecting conflicting
values, undeclared cells, unknown panels, and undeclared required IDs.

Render attempts exit 2 and write nothing. No figure artifact was created at any
point (the twelve files under `paper/*/figures/` are the pre-existing PRE
artifacts for Figures 1 and S1–S3, mtime 2026-08-04).

### 7.1 `--status`, main figures, evidence root = multicore worktree

```
# evidence root : /home/lzq/workspace/parttime/thermoroute-water-temperature-multicore
# repo root     : /home/lzq/workspace/parttime/thermoroute-water-temperature
# schema        : 4.0.0-post-skeleton-stage19-independent
figure   authority                    sev        ok   detail
fig02    spec_state                   required   no   spec state = POST_TEMPLATE_ONLY
fig02    opening_receipt              required   no   0 matches for outputs/confirmatory/route_a_*/opening_receipt_v1.json
fig02    stage09_receipt              required   yes  outputs/models/route_a_stage09_completion.json
fig02    si06_formal_rows             required   yes  paper/si/SI06_formal_five_rows_RECEIPT.md
fig02    si07_all_model_scores        required   yes  paper/si/SI07_all_model_scores_RECEIPT.md
fig02    confirmatory_protocol        required   yes  protocols/route_a_confirmatory_v1.json
fig02    post_temporal_predictions    required   no   0 matches for outputs/confirmatory/route_a_*/trusted/temporal_predictions_v1.parquet
fig02    post_availability_registry   required   no   0 matches for outputs/confirmatory/route_a_*/trusted/availability_registry_v1.csv
fig02    post_statistics              required   no   0 matches for outputs/confirmatory/route_a_*/trusted/statistics_v1.json
fig03    spec_state                   required   no   spec state = POST_TEMPLATE_ONLY
fig03    opening_receipt              required   no   0 matches for outputs/confirmatory/route_a_*/opening_receipt_v1.json
fig03    post_probabilistic_evaluation required   no   0 matches for outputs/confirmatory/route_a_*/trusted/probabilistic_evaluation_v2.json
fig03    post_temporal_predictions    required   no   0 matches for outputs/confirmatory/route_a_*/trusted/temporal_predictions_v1.parquet
fig03    post_availability_registry   required   no   0 matches for outputs/confirmatory/route_a_*/trusted/availability_registry_v1.csv
fig03    erratum_contract             required   yes  protocols/route_a_probability_metric_erratum_v1.json
fig03    confirmatory_protocol        required   yes  protocols/route_a_confirmatory_v1.json
fig03    si08_probability_metrics     required   yes  paper/si/SI08_probability_metrics_RECEIPT.md
fig03    provenance_qualifier         qualifier  yes  The Stage-19 development-period probabilistic script is withheld for this submission (zero-width nominal intervals; 0 strict quantile crossings in 26,993,675 member-level rows). This figure does NOT depend on it: the target-period metrics are computed by the trusted scorer inside the one-time opening (src/thermoroute/opening.py:8932-8948).
fig03    provenance_qualifier         qualifier  yes  Pre-opening guard: opening.py:7093-7094 enforces a strict q05 < q95 on member-averaged nominal heads and aborts the whole opening on violation. On the development panel, member averaging clears every affected LightGBM key (5 members); the only survivors are 12 single-member LightGBM-perstation keys, and that model is not in PRIMARY_MODELS or the confirmatory protocol. Re-run the check on the target-period predictions before executing the one-time opening.
fig04    spec_state                   required   no   spec state = POST_TEMPLATE_ONLY
fig04    opening_receipt              required   no   0 matches for outputs/confirmatory/route_a_*/opening_receipt_v1.json
fig04    stage09_receipt              required   yes  outputs/models/route_a_stage09_completion.json
fig04    stage16_receipt              required   yes  outputs/models/route_a_stage16_completion.json
fig04    stage25_receipt              required   yes  outputs/models/route_a_stage25_completion.json
fig04    si10_temporal_coverage       required   yes  paper/si/SI10_temporal_coverage_RECEIPT.md
fig04    si14_missingness_failures    required   yes  paper/si/SI14_missingness_failures_RECEIPT.md
fig04    post_temporal_predictions    required   no   0 matches for outputs/confirmatory/route_a_*/trusted/temporal_predictions_v1.parquet
fig04    post_external_predictions    required   no   0 matches for outputs/confirmatory/route_a_*/trusted/external_predictions_v1.parquet
fig04    post_temporal_coverage_audit required   no   0 matches for outputs/confirmatory/route_a_*/trusted/temporal_coverage_audit_v1.json
fig04    stage09b_receipt             qualifier  yes  outputs/models/route_a_stage09b_completion.json [development-only controls; SI09 tabulation only, never substituted into panel (a)]
# exit 2 = at least one REQUIRED gate is unmet (expected pre-POST); qualifier rows never block
```

Exit status 2.

### 7.2 `--status`, supporting figures, evidence root = multicore worktree

```
# evidence root : /home/lzq/workspace/parttime/thermoroute-water-temperature-multicore
# repo root     : /home/lzq/workspace/parttime/thermoroute-water-temperature
# schema        : 4.0.0-post-skeleton-stage19-independent
figure   authority                        sev        ok   detail
figS4    spec_state                       required   no   spec state = POST_TEMPLATE_ONLY
figS4    opening_receipt                  required   no   0 matches for outputs/confirmatory/route_a_*/opening_receipt_v1.json
figS4    stage09_receipt                  required   yes  outputs/models/route_a_stage09_completion.json
figS4    si06_formal_rows                 required   yes  paper/si/SI06_formal_five_rows_RECEIPT.md
figS4    si07_all_model_scores            required   yes  paper/si/SI07_all_model_scores_RECEIPT.md
figS4    confirmatory_protocol            required   yes  protocols/route_a_confirmatory_v1.json
figS4    post_temporal_predictions        required   no   0 matches for outputs/confirmatory/route_a_*/trusted/temporal_predictions_v1.parquet
figS4    post_availability_registry       required   no   0 matches for outputs/confirmatory/route_a_*/trusted/availability_registry_v1.csv
figS4    post_statistics                  required   no   0 matches for outputs/confirmatory/route_a_*/trusted/statistics_v1.json
figS5    spec_state                       required   no   spec state = POST_TEMPLATE_ONLY
figS5    opening_receipt                  required   no   0 matches for outputs/confirmatory/route_a_*/opening_receipt_v1.json
figS5    post_probabilistic_evaluation    required   no   0 matches for outputs/confirmatory/route_a_*/trusted/probabilistic_evaluation_v2.json
figS5    post_temporal_predictions        required   no   0 matches for outputs/confirmatory/route_a_*/trusted/temporal_predictions_v1.parquet
figS5    post_external_predictions        required   no   0 matches for outputs/confirmatory/route_a_*/trusted/external_predictions_v1.parquet
figS5    post_availability_registry       required   no   0 matches for outputs/confirmatory/route_a_*/trusted/availability_registry_v1.csv
figS5    erratum_contract                 required   yes  protocols/route_a_probability_metric_erratum_v1.json
figS5    si08_probability_metrics         required   yes  paper/si/SI08_probability_metrics_RECEIPT.md
figS5    confirmatory_protocol            required   yes  protocols/route_a_confirmatory_v1.json
figS5    provenance_qualifier             qualifier  yes  The Stage-19 development-period probabilistic script is withheld for this submission (zero-width nominal intervals; 0 strict quantile crossings in 26,993,675 member-level rows). This figure does NOT depend on it: the target-period metrics come from the trusted scorer inside the one-time opening (src/thermoroute/opening.py:8932-8948).
figS6    spec_state                       required   no   spec state = POST_TEMPLATE_ONLY
figS6    opening_receipt                  required   no   0 matches for outputs/confirmatory/route_a_*/opening_receipt_v1.json
figS6    si10_temporal_coverage           required   yes  paper/si/SI10_temporal_coverage_RECEIPT.md
figS6    si14_missingness_failures        required   yes  paper/si/SI14_missingness_failures_RECEIPT.md
figS6    post_temporal_coverage_audit     required   no   0 matches for outputs/confirmatory/route_a_*/trusted/temporal_coverage_audit_v1.json
figS6    post_availability_registry       required   no   0 matches for outputs/confirmatory/route_a_*/trusted/availability_registry_v1.csv
figS7    spec_state                       required   no   spec state = POST_TEMPLATE_ONLY
figS7    opening_receipt                  required   no   0 matches for outputs/confirmatory/route_a_*/opening_receipt_v1.json
figS7    si11_spatial_sensitivity         required   yes  paper/si/SI11_spatial_sensitivity_RECEIPT.md
figS7    post_spatial_sensitivity         required   no   0 matches for outputs/confirmatory/route_a_*/trusted/spatial_sensitivity_v1.json
figS7    stage13c_region_transfer_table   required   yes  outputs/tables/region_transfer.csv
figS7    stage13c_region_transfer_report  qualifier  yes  outputs/reports/region_transfer.md [narrative companion to the region-transfer table]
figS8    spec_state                       required   no   spec state = POST_TEMPLATE_ONLY
figS8    opening_receipt                  required   no   0 matches for outputs/confirmatory/route_a_*/opening_receipt_v1.json
figS8    stage25_receipt                  required   yes  outputs/models/route_a_stage25_completion.json
figS8    si12_qc_qualifiers               required   yes  paper/si/SI12_qc_qualifiers_RECEIPT.md
figS8    si13_external_history_arm        required   yes  paper/si/SI13_external_history_arm_RECEIPT.md
figS8    si14_missingness_failures        required   yes  paper/si/SI14_missingness_failures_RECEIPT.md
figS8    post_outcome_qc_gate             required   no   0 matches for outputs/confirmatory/route_a_*/trusted/outcome_qc_gate_v1.json
figS8    post_outcome_quality_audit       required   no   0 matches for outputs/confirmatory/route_a_*/trusted/outcome_quality_audit_v1.json
figS8    post_external_predictions        required   no   0 matches for outputs/confirmatory/route_a_*/trusted/external_predictions_v1.parquet
figS9    spec_state                       required   no   spec state = POST_TEMPLATE_ONLY_DEVELOPMENT
figS9    opening_receipt                  required   no   0 matches for outputs/confirmatory/route_a_*/opening_receipt_v1.json
figS9    stage22_row_table                required   yes  outputs/tables/aci_coverage.csv
figS9    stage22_report                   required   yes  outputs/reports/adaptive_conformal.md
figS9    confirmatory_protocol            required   yes  protocols/route_a_confirmatory_v1.json
figS9    provenance_qualifier             qualifier  yes  Panels bind Stage-22 development-period evidence (2019-01-01..2020-12-24) over 249,072 exact keys, 120 sites, 15 HUC2 groups, leads {1,3,7}. The confirmatory target period starts 2021-01-01; no panel in this figure is a target-period result and no value here may be compared numerically with Figure 3 or S5.
figS9    evidence_period                  qualifier  yes  development_2019_2020 (Stage-22 span 2019-01-01..2020-12-24; confirmatory target starts 2021-01-01); in-panel scope band figS9.scope.development_period_not_confirmation is mandatory
# exit 2 = at least one REQUIRED gate is unmet (expected pre-POST); qualifier rows never block
```

Exit status 2.

### 7.3 Reading the output

Every figure is correctly blocked. The binding blocker for all nine is the
**missing opening receipt** — `outputs/confirmatory/` exists in neither worktree —
followed by the nine `trusted/` artifacts the opening will create. Stage-19 never
appears as a blocker anywhere; it is informational on Figures 3 and S5 only.

Figure S9 is the one figure whose substantive evidence already resolves
(`aci_coverage.csv`, `adaptive_conformal.md` both `yes`); it is blocked solely by
the spec state and the opening receipt.

The `--evidence-root` flag (or `THERMOROUTE_FIGURE_EVIDENCE_ROOT`) is what makes
Stage-09/09b/16/25, Stage-13c, and Stage-22 resolve at all. With the default root
— this paper tree — those rows read `no`, which is the inaccuracy the audit found
in item 11.

---

## 8. Remaining risk

| Risk | Severity | Note |
|---|---|---|
| **Pre-opening degeneracy abort** (§4a) | **High — act before opening** | `opening.py:7093-7094` can abort the one-shot opening on a single zero-width member-averaged interval. Development-panel measurement is clean for every model in `PRIMARY_MODELS`, but target-period data is untested. Run the read-only check first |
| Opening not yet run | High (expected) | All nine figures blocked; nothing renderable until the one-time opening completes |
| No event-probability evidence if the opening's probability step fails | Medium | Figures 3 and S5 both depend on `probabilistic_evaluation_v2.json`. If that step fails at opening time, the fallback is Figure S9's development-period conformal evidence plus an honest Limitations paragraph — the interim plan in this document's history is the recipe |
| Figure S9 mistaken for a target-period result | Medium — mitigated | Mandatory in-panel scope band, `EvidencePeriod.DEVELOPMENT`, explicit prohibition on numerical comparison with Figures 3/S5, and a manifest guard that refuses to render it unlabelled |
| Panel builders remain unimplemented | Low (expected) | `PANEL_BUILDERS` is empty by design; every figure refuses at the builder stage with exit 2 |
| Nine figures may exceed page budget | Low | S9 is explicitly optional and droppable without weakening a registered claim |
| Two `WRR_FIGURE_STYLE_GUIDE.md` copies | **Resolved** | The `docs/` copy is renamed `docs/WRR_FIGURE_STYLE_TEARDOWN.md` and all three cross-references in `paper/WRR_FIGURE_STYLE_GUIDE.md` are updated. They were never duplicates: `paper/` is the layer-2 design manual, `docs/` the layer-3 sample teardown |

### 8.1 A note on the AGU size targets

The original brief specified "single-column 95 mm or full-width 190 mm".
**Neither is an AGU size.** AGU's published ranges are single column 50–85 mm and
double column 105–170 mm, maximum height 228 mm. The repository had already
resolved this on 2026-08-05: full width 140 mm (matching `agujournal2019.cls`
`\textwidth 5.5in` = 139.7 mm) and single column 85 mm, with the recorded
rationale that authoring at 88 mm would trigger a 96.6 % production rescale and
push 7.5 pt ticks to 7.24 pt. `RenderProfile.validate()` now rejects 95 mm and
190 mm explicitly so the wrong values cannot re-enter. Correction accepted by the
coordinator 2026-08-05.

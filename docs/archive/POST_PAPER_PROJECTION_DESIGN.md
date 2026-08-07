# Receipt-to-paper POST projection design (P0-P)

| Field | Value |
| --- | --- |
| Date | 2026-08-01 |
| Status | **DESIGN ONLY / NO EMPIRICAL VALUE / PRE PAPER UNCHANGED** |
| Principle | POST is a deterministic read-only projection of verified receipts, never an editor of evidence |

The current claim script can append structured rows to Markdown, while the AGU
builder is intentionally PRE-only and rejects post-opening use. P0-P therefore
requires a separate POST pipeline. It must not overwrite the frozen PRE TeX or
read development Stage-09 caches as though they were opening evidence.

## 1. Main figures

### Figure 1 — design, scope and five registered rows

- issue-time information boundary;
- exact bounded-correction equation and “not a safety bound” warning;
- station/HUC2 imbalance, effective cluster count and outcome-free gate;
- five-row ΔRMSE forest.

For Route A, every row is labelled fixed-cohort descriptive and all CI/p/Holm
values are assumption-conditional sensitivities.

### Figure 2 — point performance

- station-level RMSE distributions for all six main models by horizon;
- ThermoRoute−damped h1/h3/h7 station effects;
- ThermoRoute−LightGBM h3/h7 effects with the +0.05 °C numerical ceiling;
- HUC2-sorted site×horizon ΔRMSE heatmap.

### Figure 3 — probability behavior

- 90% marginal coverage;
- interval width;
- Brier skill against the frozen seasonal reference;
- reliability curves with bin counts.

Only models with a frozen probabilistic head appear. Three emitted quantiles are
not called CRPS and marginal calibration is not called conditional coverage.

### Figure 4 — robustness and scope

- horizon availability and attrition;
- equal-cell, leave-year and season sensitivity;
- per-HUC/leave-one-HUC effects and cluster warnings;
- external 30-site history-dependent arm, explicitly “not ungauged.”

Route B uses separate figure templates, component/space×time UQ and its own
receipt namespace. Route-A panels cannot be relabelled as Route-B evidence.

## 2. Main tables and SI

Main tables:

- T1 cohort, split, information set and availability;
- T2 all five registered comparisons, counts, effect, interval, p/Holm and
  descriptive-only verdict;
- T3 all-model pooled and station-balanced RMSE/MAE/bias.

SI index:

```text
SI00 inventory and evidence map
SI01 cohort/registry
SI02 issue-time information boundary
SI03 equations, units and identifiability limits
SI04 protocols, amendments and chronology
SI05 five-row comparison geometry and estimands
SI06 formal five rows
SI07 all-model scores
SI08 probability metrics and reliability bins
SI09 Stage09/09b model matrix, budgets, seeds and controls
SI10 temporal coverage
SI11 spatial/leave-HUC sensitivity
SI12 QC, qualifiers and exact-A audit
SI13 external history-dependent arm
SI14 missingness, attrition and failure cases
SI15 reproduction hashes and commands
SI16 rights matrix and data dictionary
FigS1–FigS8 enlarged diagnostics
```

## 3. Evidence-field routing

| Consumer | Required authority |
| --- | --- |
| Fig1d / T2 | opening receipt `formal_tests[*]`: test identity, model pair, horizon, margin, status, effect, CI, station/cluster counts, win rate, raw/Holm p and bound checks |
| Claim eligibility | bound inference gate and outcome-QC gate statuses; no renderer override |
| Fig2 / T3 | receipt-bound temporal predictions on exact common keys; every recomputation records formula, filters and parent SHA |
| Fig3 | bound probabilistic evaluation rows, counts, coverage, width, pinball, Brier/skill, log score, discrimination, ECE/calibration and reliability bins |
| Fig4 temporal | availability registry and temporal-coverage sensitivities |
| Fig4 spatial | spatial-sensitivity comparisons and cluster diagnostics |
| External arm | separately receipt-bound external predictions/metrics |
| QC SI | outcome quality audit, QC gate and approved-target sensitivity |
| Controls/budgets | authorization→model suite→Stage09/09b/16/25 receipts→bound predictions/metrics/budget ledger |

The builder follows declared bindings rather than guessing familiar filenames.

## 4. POST evidence manifest

`post_paper_evidence_v1.json` contains:

```text
format
route_id
authority                       # opening, authorization, claim registry, model suite, stage receipts
source_bindings
values                          # value_id → value, unit, evidence role, source pointer, derivation, rounding
sections
tables
figures
si
outputs
```

Every visible number, including n, dates, percentages and axis annotations, has
one `value_id`. The configured reporting policy distinguishes measurement
resolution, sensor uncertainty and display precision.

`post_render_receipt_v1.json` binds the evidence manifest, rendered Markdown,
figure-data tables, SVG/PDF figures, SI, TeX/PDF, environment, fonts, page count,
page-QA report and all SHA-256 values.

## 5. Fail-closed acceptance

1. Every input path, format, self-hash and byte binding validates.
2. Missing, non-finite or extra undeclared evidence rejects the build.
3. The five Route-A rows appear exactly once and retain their gate verdict.
4. Every visible value resolves to one manifest entry and evidence pointer.
5. A second build from identical inputs is byte-identical, or any platform
   variance is covered by a stricter declared rendering tolerance/normalization.
6. TeX has no missing figure/citation/undefined reference; fonts are embedded,
   figure text is at least 8 pt and colors pass the chosen color-vision check.
7. Page QA records overflow, clipping, raster resolution and caption/table
   completeness.
8. Route A rejects superiority, non-inferiority, equivalence, national and
   operational language regardless of favorable numerical sensitivities.
9. Route B requires its independent registry, ≥30 reportable components, balance
   gates, external timestamp, fair-budget receipt and space×time UQ before its
   own template is eligible.
10. PRE placeholders or `[pending — 探索期数据，不可写入结论]` in a POST build are
    errors; their absence never permits fabricated replacement values.

## 6. Future file layout

```text
publication/contracts/post_paper_contract_v1.json
publication/templates/route_a/*.md.j2
publication/templates/route_b/*.md.j2
scripts/31_build_post_evidence.py
scripts/32_render_post_paper.py
scripts/33_verify_post_paper.py
tests/test_post_paper_projection.py
outputs/publication/<route>_<inputhash>/
  post_paper_evidence_v1.json
  results.md
  discussion.md
  tables/
  figures/
  si/
  submission/ThermoRoute_WRR.tex
  submission/ThermoRoute_WRR.pdf
  page_qa_v1.json
  post_render_receipt_v1.json
```

Implementation must wait for the protected source boundary and use the clean-room
contract. The current `paper/agu_submission/**` PRE package remains immutable.

# ThermoRoute — AGU Key Points

This file is the authoring source for the AGU **Key Points** block. Its three
items are byte-identical to the Key Points block of `paper/ThermoRoute_paper.md`
and to the `keypoints` macro generated into
`paper/agu_submission/ThermoRoute_WRR.tex`. Changing one without the other two is
an inconsistency, not an edit.

**AGU constraint.** At most three Key Points; each is one complete sentence of at
most 140 characters including spaces; each states a fact about the work rather
than a claim about its importance.

## Key Points

1. Station-median skill at a seven-day lead is +0.251 against persistence but
   +0.038 against damped persistence at 120 U.S. gauges. *(128 characters)*
2. A gradient-boosted tree with site identity has the lowest station-median RMSE
   at 1, 3, and 7 days on identical prediction keys. *(127 characters)*
3. Holding out whole hydrologic regions instead of random sites lowers three-day
   skill against persistence from +0.187 to +0.116. *(126 characters)*

## What a reader can check, and where

| Key Point | Checkable assertion | Where it is specified |
|---|---|---|
| 1 | Skill is `1 − RMSE(candidate)/RMSE(reference)`, dimensionless, defined at manuscript equation (10); station-median RMSE 2.217 (persistence), 1.738 (damped persistence), 1.657 °C (ThermoRoute) at a 7-day lead on one common 249,072-key set | §3.6, §4.1 of the manuscript; SI05, SI07 |
| 2 | Global LightGBM with stable site identity as a categorical feature scores 0.578 / 1.280 / 1.649 °C against 0.631 / 1.291 / 1.657 °C, with a ThermoRoute station win rate of 0.00 at a 1-day lead across 120 stations | §3.2, §4.2 of the manuscript; SI07, SI09 |
| 3 | Four folds of [30, 30, 31, 29] stations holding out whole HUC2 regions, mean nearest-training-gauge distance 289 km, against a four-fold random held-site warm-start arm on the same panel | §3.4, §4.4 of the manuscript; SI11 |

All three Key Points are development-period (2019–2020) benchmark diagnostics.
The 2019–2020 interval participated in cohort construction and model tuning; the
held-out 2021–2023 evaluation is reported in §4.6 of the manuscript.

## Scope statements that accompany these Key Points

- The comparisons are reported for a fixed availability-enriched 120-site cohort.
  With at most 15 HUC2 groups and an effective cluster count of about 9.54, and a
  cohort that was not probability-sampled, the clustered intervals and p-values
  are approximate descriptive sensitivities rather than decision evidence.
- No comparison is written as a statement of superiority, non-inferiority,
  equivalence, or parity, and none generalises to a national or U.S.-river
  population. Failure to reject a clustered test is not evidence of equivalence.
- The cohort is availability-enriched rather than randomly drawn, and every
  quantity reported from 2019–2020 is a development-period diagnostic.
- Every arm, including the whole-region holdout and the site-identifier-disjoint
  external arm, consumes the target site's observed water temperature through the
  issue date. This is gauged transfer; it is not prediction at an ungauged
  location.
- Split-conformal coverage is reported as an empirical marginal diagnostic. No
  conditional-coverage statement is made, and the equal-weight three-quantile
  pinball summary is not CRPS.
- The development-period raw quantile-head probabilistic suite is not reported;
  the reason and the measured facts are in §6.3 of the manuscript. Zero strict
  quantile-ordering violations were measured; the flagged rows are zero-width
  (degenerate) nominal intervals.
- The air2stream-style hybrid reference is `NOT_RUN` and its implementation is
  unofficial. It is excluded from every comparison, and its status is reported in
  SI07.
- The held-out 2021–2023 metric cells are `<<...>>` placeholders pending
  computation from `outputs/conventional/holdout_metrics_2021_2023.csv`.
- The data archive is described as planned pending a byte-level rights review;
  no public release is asserted.

## Status of this file

`paper/highlights.md` mirrors the Key Points block and scope statements of the
conventional comparative-holdout manuscript `paper/ThermoRoute_paper.md`. The
superseded earlier-era bytes remain recoverable in version control.

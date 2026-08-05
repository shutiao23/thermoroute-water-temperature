# ThermoRoute — AGU Key Points

This file is the authoring source for the AGU **Key Points** block. Its three
items are byte-identical to the Key Points block of `paper/ThermoRoute_paper.md`
and to the `keypoints` environment generated into
`paper/agu_submission/ThermoRoute_WRR.tex`. Changing one without the other two is
an inconsistency, not an edit.

**AGU constraint.** At most three Key Points; each is one complete sentence of at
most 140 characters including spaces; each states a fact about the work rather
than a claim about its importance.

## Key Points

1. Daily river water temperature at 120 U.S. gauges is scored against damped
   persistence, a tree ensemble, and an LSTM on identical keys. *(134 characters)*
2. The 2021–2023 evaluation labels are opened once, after the model suite, the
   inputs, and the analysis code are frozen and replayed. *(130 characters)*
3. A pre-specified cluster gate fails on the frozen 15-region cohort, so all five
   formal comparisons are fixed-cohort descriptive effects. *(135 characters)*

## What a reader can check, and where

| Key Point | Checkable assertion | Where it is specified |
|---|---|---|
| 1 | 120 stable USGS site numbers; 657,480 site-days; 2006–2020; damped persistence, LightGBM, and a global LSTM scored on one common station/date/horizon key set | §2.1, §2.3, §3.2, §4.1 of the manuscript; SI01, SI05, SI07 |
| 2 | The evaluation interval is 2021-01-01 to 2023-12-31; the model-suite commit must precede the evaluation-period predictor artifacts; an isolated replay precedes acquisition | §3.3, §3.7 of the manuscript; SI02, SI04, SI15 |
| 3 | Gate thresholds `n_clusters ≥ 30`, `effective_cluster_fraction ≥ 0.75`, `largest_cluster_share < 0.25`; frozen cohort has at most 15 HUC2 groups with an inverse-Herfindahl effective cluster count of about 9.54 | §3.6, §5.1 of the manuscript; `docs/B02_PERMANENT_DESCRIPTIVE_CLAIM_WORDING.md`; SI01, SI04 |

Key Point 3 is determined by the registry alone. It required no evaluation
outcome and is not reopened by one.

## Scope statements that accompany these Key Points

- The five formal comparisons are fixed-cohort descriptive effects under the
  verdict `DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED`. Whole-HUC2 cluster-bootstrap
  intervals, exact sign-flip p-values, and Holm adjustments accompany them as
  assumption-conditional sensitivities and are not decision evidence.
- No row may be written as superiority, non-inferiority, equivalence, or parity,
  and no row may be generalized to a national or U.S.-river population.
- The cohort is availability-enriched rather than randomly drawn, and the
  2019–2020 interval participated in cohort construction, so every quantity
  reported from it is exploratory.
- Every arm, including the whole-region holdout and the site-identifier-disjoint
  external arm, consumes the target site's observed water temperature through the
  issue date. This is gauged transfer; it is not prediction at an ungauged
  location.
- Split-conformal coverage is reported as an empirical marginal diagnostic. No
  conditional-coverage statement is made, and the equal-weight three-quantile
  pinball summary is not CRPS.
- The pre-registered probabilistic metric suite is not reported; the reason and
  the measured facts are in §6.3 of the manuscript.
- The evaluation-period result slots are unfilled at the time of writing.
- The archive chronology is sealed inside the repository and assumes an honest
  owner. There is no external timestamp, public registration service, or
  independent custodian.

## Status of this file

`paper/highlights.md` is one of the documents whose bytes are SHA-256 frozen in
`protocols/route_a_claim_registry_v1.json` (`preopen_document_sha256`). This
revision changes those bytes and therefore invalidates that binding until the
registry is re-sealed under a separately authorized `protocols/` change. The
superseded bytes remain recoverable at commit `b0699a8`.

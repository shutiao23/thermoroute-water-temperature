# Cover letter — Water Resources Research

> **Send condition.** This letter is written in its final form. It must not be
> sent until (i) the held-out 2021–2023 metric cells of Section 4.6 are filled
> from `outputs/conventional/holdout_metrics_2021_2023.csv`, and (ii) every
> bracketed placeholder below and in the manuscript is closed. The send
> condition is tracked in `docs/WRR_SUBMISSION_CHECKLIST.md`.

[DATE TO BE COMPLETED]

Editor-in-Chief
*Water Resources Research*

Dear Editor,

We submit for your consideration the manuscript **"Reported skill in daily river
water-temperature prediction shrinks under strong baselines and whole-region
holdout."**

Daily river water temperature is one of the most heavily modelled targets in
applied hydrological machine learning, and the prevailing account of that
literature is that deep sequence architectures have substantially advanced it.
Reported gains, however, are far more dispersed than the architectural
differences that are supposed to explain them, and they are not comparable across
studies. Our manuscript asks what that dispersion is made of. Holding the model
and the data fixed, we vary the three design choices that almost every study in
this field makes the same way — the reference model, the key set and
preprocessing boundary, and the spatial partition — and measure how much each one
moves the reported number.

The answer is that they move it more than the architecture does. On one common
set of 249,072 station/date/horizon keys at 120 U.S. gauges, the same fixed model
reports a seven-day median station skill of +0.251 against naive persistence and
+0.038 against damped persistence; of the 0.560 °C reduction in station-median
RMSE from persistence to the deep model, 0.479 °C is delivered by relaxing
yesterday's observation toward a fitted seasonal climatology. A well-tuned
gradient-boosted tree with site identity has the lowest station-median RMSE at
all three leads, and at a one-day lead it is more accurate at every one of the
120 stations. An information-matched plain causal convolutional network, given
the same inputs, the same anchor, the same optimiser budget and 38,346 against
38,505 parameters, reproduces the full architecture to within 0.023 °C, while
deleting the router, the mixture, the dynamic prior or the residual bound moves
one-day error by at most 0.004 °C. And three-day skill against persistence falls
from +0.187 under a random held-site split to +0.116 under whole-region holdout,
at a mean nearest-training-gauge distance of 289 km on a panel where 19 of 120
stations have a neighbour within 10 km.

We report those results directly, including the ones that are unfavourable to our
own architecture, because they are the paper's contribution. The manuscript is an
evaluation-design study for a hydrological prediction problem rather than a
performance record, and its transferable content is three controls that require
no unusual computational resources: scoring every model on one identical key
registry so that no model benefits from selective prediction; fitting every
preprocessing and calibration statistic strictly backwards in time; and
partitioning space by whole hydrologic regions rather than by site. Each of the
three moved a reported number in our study by more than the difference between
the competing architectures.

Three features of the submission warrant explanation in advance.

**The design is a conventional comparative holdout.** All models are trained and
tuned on data through 2020, with 2019–2020 as the development-evaluation window
that informed model selection, and the same model suite is then evaluated on a
held-out 2021–2023 test window. Every preprocessing and calibration statistic is
fitted strictly backwards in time, and the held-out daily means are outcomes
only: they are not read by model-selection, feature-selection, threshold-selection,
calibration, or station-inclusion code, all of which is fixed on data through
2020. The development-period (2019–2020) benchmark diagnostics are reported in
Sections 4.1–4.5, and the held-out 2021–2023 comparative metrics in Section 4.6;
the latter are clearly marked placeholders pending a parallel metric computation.

**The comparisons are descriptive for this fixed cohort, and we say why in full.**
The cohort was assembled by an availability filter over gauges in 34 states, not
by probability sampling of regions, so the independent, exchangeable sampling of
HUC2 regions that clustered inference assumes is not established. With only 15
HUC2 groups (an inverse-Herfindahl effective cluster count of about 9.54 and a
largest group holding 21.7% of stations) the whole-HUC2 cluster-bootstrap
intervals, exact sign-flip p-values, and Holm adjustments are therefore
approximate and are presented as descriptive sensitivities rather than as
decision evidence. The manuscript makes **no claim of superiority,
non-inferiority, equivalence, or parity** for any model over any other, and no
claim that its results generalize to a national or U.S.-river population.

**Two declared analyses are absent, and both absences are reported with their
measured cause.** An air2stream-style hybrid reference was planned but never
fitted, and the available implementation is unofficial, so we removed it from the
comparison set entirely rather than carry an empty column; its status is
documented in the Supporting Information. Separately, on the development panel 135
of 26,993,675 member-level rows — 0.0005%, concentrated at twelve ice-affected
sites where the conditional percentiles of water temperature genuinely coincide
near the freezing point — have degenerate (zero-width) nominal intervals, with
zero strict quantile-ordering violations anywhere in the panel. We report
split-conformal intervals (which are non-degenerate by construction) for the
development period and do not report the raw quantile-head probabilistic suite
for that period; the held-out-window probability family is reported in full.

Supporting Information accompanies the manuscript and contains the cohort
ledgers, the analysis protocol and model-suite registry, the full comparison-set
specification, all-model scores on the exact common keys, the cluster geometry at
four hydrologic partition levels, the coverage and quality-control audits, and
the reproduction hashes and commands.

The manuscript has not been published elsewhere and is not under consideration by
another journal. All authors have approved the submission and have declared their
contributions, funding, and competing interests as recorded in the Acknowledgments.

[SUGGESTED REVIEWERS TO BE COMPLETED — names, affiliations, e-mail addresses, and
a statement of no conflict with the author list.]

[OPPOSED REVIEWERS TO BE COMPLETED, or an explicit statement that there are none.]

Thank you for considering this work.

Sincerely,

[CORRESPONDING AUTHOR TO BE COMPLETED — verified name, ORCID, affiliation, and
institutional e-mail address]
on behalf of all authors

---

## Status of this file

`paper/cover_letter.md` is the conventional comparative-holdout cover letter for
`paper/ThermoRoute_paper.md`. The superseded earlier-era bytes remain
recoverable in version control.

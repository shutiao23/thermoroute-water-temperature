# Cover letter — Water Resources Research

> **Send condition.** This letter is written in its final form. It must not be
> sent until (i) the one-time evaluation of Section 3.7 has been executed and the
> Section 4.6 slots are filled from the verified receipt, and (ii) every bracketed
> placeholder below and in the manuscript is closed. The send condition is
> tracked in `docs/WRR_SUBMISSION_CHECKLIST.md`.

[DATE TO BE COMPLETED]

Editor-in-Chief
*Water Resources Research*

Dear Editor,

We submit for your consideration the manuscript **"Strong baselines, controlled
leakage, and a pre-specified inference gate for daily river water-temperature
hindcasting."**

Daily river water temperature is a variable for which it is easy to publish an
attractive number and hard to publish an honest one. The series is strongly
persistent, so a copy of yesterday's observation is already a demanding
reference; meteorological covariates leak future information unless their
issue-time availability is defined explicitly; sensor gaps are correlated with
season and site; and gauges in the same hydrologic region are not statistically
independent. The manuscript is organised around those four difficulties rather
than around the architecture we happen to have built.

The study assembled a panel of 657,480 site-days from 120 stable USGS site
numbers over 2006–2020, spanning 34 states and 15 two-digit hydrologic unit
groups, and used it for five design decisions that we believe are the
transferable part of the work.

**A strong baseline suite.** Every model was scored on identical
station/date/horizon keys against persistence, damped persistence, seasonal
climatology, an air2stream-style reference, a gradient-boosted tree ensemble with
and without site identity, and a global LSTM. No model was permitted its own
complete-case key set, so accuracy cannot be gained by declining to predict on
hard keys. We report the tuning budgets, which were documented but not equalized
across model classes, as an asymmetry rather than smoothing over it.

**Leakage control by mechanism rather than by assertion.** All predictors carry a
date no later than the issue date, no horizon-specific future weather field is
consumed, the temporal encoder is non-anticipating by construction with a bounded
lag range, and every preprocessing, calibration, and threshold statistic was
fitted strictly before the interval on which it is applied. Before any label was
acquired, an isolated interpreter with no network access and no read access to
the evaluation namespaces reloaded every trained member and reproduced the frozen
predictions, thresholds, and calibration parameters from the frozen inputs.

**A regional-transfer arm that is labelled honestly.** Whole hydrologic regions
were held out, so no gauge from a held-out region appears in training and the
mean distance from a held-out station to its nearest training gauge is 289 km. We
state explicitly that this is *gauged* transfer, because held-out sites still
supply their own observed water-temperature history through the issue date. It is
not prediction at an ungauged location, and we have taken care that no sentence
in the manuscript can be read as the latter.

**Conformal intervals with their assumptions stated.** Split conformalized
quantile regression was calibrated on a single earlier year, and the deployed
offset is clipped below at zero so that calibration can widen a nominal interval
but never shrink it. We report coverage as an empirical marginal diagnostic,
state that exchangeability is violated by construction, and make no
conditional-coverage claim of any kind.

**A single acquisition of the evaluation labels.** The held-out labels covering
2021 through 2023 were downloaded exactly once, after the model suite, the input
set, and the analysis code were frozen, hash-sealed, and independently replayed,
and after a chronology gate verified that the model-suite commit genuinely
precedes the evaluation-period artifacts. Failure of that gate would not have
produced a weaker result; it would have demoted the whole exercise to
retrospective exploration.

We draw the Editor's attention to one feature of the submission that is unusual
enough to warrant explanation in advance. Before any outcome was visible, we
wrote down a gate on the cluster structure of our own cohort — at least 30
reportable clusters, an effective-cluster fraction of at least 0.75, and a
largest-cluster share below 0.25 — under which our own headline comparisons would
not be allowed to carry inferential weight. The frozen cohort contains at most 15
hydrologic regions with an inverse-Herfindahl effective cluster count of about
9.54, so the gate fails on cohort geometry alone, independently of any number the
evaluation produced. We report that failure rather than hiding it. All five
formal comparisons are therefore reported as fixed-cohort descriptive effects
under the verdict `DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED`; the clustered
p-values, bootstrap intervals, and Holm adjustments are retained beside them as
assumption-conditional sensitivities and are not decision evidence.

We want to be unambiguous about what this means for the paper's conclusions. The
manuscript makes **no claim of superiority, non-inferiority, equivalence, or
parity** for the proposed predictor over any reference model, and no claim that
its results generalize to a national or U.S.-river population. Where the
development evidence indicates that a well-tuned tree ensemble attains lower
station-median error than our own architecture, we say so plainly in the Results
and again in the Discussion. A second pre-registered contract also bound against
our convenience: a frozen probabilistic-metric check rejects a small number of
correctly ordered but zero-width nominal intervals, and because relaxing it would
have invalidated the frozen training receipts, we left it alone and report the
probabilistic metric suite as not reported, with the measured facts given in
full.

We believe the paper is a good fit for *Water Resources Research* because its
contribution is an evaluation design for a hydrological prediction problem, not a
performance record. The three controls we consider most portable — scoring every
model on one key registry, fitting every preprocessing and calibration statistic
strictly backwards in time, and fixing the analysis and reporting rules before
the evaluation labels exist — require no unusual computational resources and
change what a reported skill score means.

Supporting Information accompanies the manuscript and contains the cohort
ledgers, the sealed protocol and its amendments, the full comparison-family
specification, all-model scores on the exact common keys, the coverage and
quality-control audits, and the reproduction hashes and commands.

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

`paper/cover_letter.md` is one of the documents whose bytes are SHA-256 frozen in
`protocols/route_a_claim_registry_v1.json` (`preopen_document_sha256`). This
revision changes those bytes and therefore invalidates that binding until the
registry is re-sealed under a separately authorized `protocols/` change. The
superseded bytes remain recoverable at commit `b0699a8`.

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
set of 249,072 station/date/horizon keys at 120 U.S. gauges over 2006–2020, the
same fixed model reports a seven-day median station skill of +0.251 against naive
persistence and +0.038 against damped persistence; of the 0.560 °C reduction in
station-median RMSE from persistence to the deep model, 0.479 °C is delivered by
relaxing yesterday's observation toward a fitted seasonal climatology. A
well-tuned gradient-boosted tree with site identity has the lowest station-median
RMSE at all three leads, and at a one-day lead it is more accurate at every one
of the 120 stations. An information-matched plain causal convolutional network,
given the same inputs, the same anchor, the same optimiser budget and 38,346
against 38,505 parameters, reproduces the full architecture to within 0.023 °C,
while deleting the router, the mixture, the dynamic prior or the residual bound
moves one-day error by at most 0.004 °C. And three-day skill against persistence
falls from +0.187 under a random held-site split to +0.116 under whole-region
holdout, at a mean nearest-training-gauge distance of 289 km on a panel where 19
of 120 stations have a neighbour within 10 km.

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

**The evaluation labels are opened once.** The held-out labels covering 2021
through 2023 are downloaded exactly once, after the model suite, the input set,
and the analysis code are frozen, hash-sealed, and replayed by an isolated
process that cannot reach the label namespace, and after a chronology gate
verifies that the model-suite commit genuinely precedes the evaluation-period
artifacts. Failure of that gate would not have produced a weaker result; it would
have demoted the whole exercise to retrospective exploration.

**The formal comparisons are permanently descriptive, and we say why in full.**
Before any outcome was visible we wrote down conditions under which our own
headline comparisons would not be allowed to carry inferential weight. Three of
those conditions fail independently and permanently. The two structural sampling
assumptions behind whole-cluster resampling and sign flipping are recorded as
unmet, because the cohort was assembled by an availability filter rather than
probability-sampled; the null-simulation calibration the gate requires was never
implemented; and the cohort has 15 hydrologic regions with an inverse-Herfindahl
effective count of about 9.54 against a threshold of 30. We draw the Editor's
attention to the last of these in particular. That threshold is unreachable under
a two-digit hydrologic partition for *any* U.S. cohort, because about 21 such
regions exist in total. It was a specification error, it was identified before
any evaluation label was accessed, and Section 3.6 discloses it as an error
rather than defending it. We also report that subdividing the map to an eight-digit
partition would clear the threshold, and explain why we did not do that: adjacent
eight-digit units on the same river are not independent, and eligibility is fixed
by the first two reasons regardless. All five formal comparisons are therefore
reported as fixed-cohort descriptive effects under the verdict
`DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED`, with the clustered p-values, bootstrap
intervals, and Holm adjustments retained beside them as assumption-conditional
sensitivities rather than as decision evidence. The manuscript makes **no claim
of superiority, non-inferiority, equivalence, or parity** for any model over any
other, and no claim that its results generalize to a national or U.S.-river
population.

**Two declared analyses are absent, and both absences are reported with their
measured cause.** An air2stream-style hybrid reference was declared but never
fitted, and the available implementation is unofficial, so we removed it from the
comparison set entirely rather than carry an empty column; its status is
documented in the Supporting Information. Separately, a frozen probabilistic
contract for the development period treats a zero-width nominal interval as fatal
and forbids evaluation-time repair. On 135 of 26,993,675 member-level rows —
0.0005%, concentrated at twelve ice-affected sites where the conditional
percentiles of water temperature genuinely coincide near the freezing point — that
condition trips, with zero strict ordering violations anywhere in the panel.
Relaxing it would have invalidated the frozen training receipts, so we left it
alone and report the development-period suite as not reported. The
evaluation-period probability family is computed by a separate path inside the
single opening and is reported in full.

Supporting Information accompanies the manuscript and contains the cohort
ledgers, the sealed protocol and its amendments, the full comparison-set
specification, all-model scores on the exact common keys, the recomputed cluster
geometry at four hydrologic partition levels, the coverage and quality-control
audits, and the reproduction hashes and commands.

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
registry is re-sealed under a separately authorized `protocols/` change, or until
the block-binding change recorded in `docs/OPTION_A_DESCRIPTIVE_BENCHMARK_SCOPE.md`
§2.6 lands. The superseded bytes remain recoverable at commit `b0699a8`.

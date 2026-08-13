# Response to the advisor review of 2026-08-13

Every item below was checked against the repository before being acted on. Where
the review was right, it says so and points at the fix. Where the review was
working from the 23-page PDF alone and the repository holds evidence it could
not see, it says that too — twice, and one of those is the most consequential
change in this revision.

**Verdict accepted in substance.** The submission was not ready, and two of the
four blockers were defects I introduced. The revision closes everything that is
editable, statistical, or bibliographic. It does not close the items that need
new data, new compute, or the authors' own information, and those are listed at
the end with reasons rather than deferred quietly.

---

## The two findings that changed the paper most

### B02 — the manuscript contradicted itself, and every gate passed

**Confirmed exactly as described.** Three sections stated that the post-outcome
information results were reported *outside* the Abstract and the Key Points.
They were the Abstract's core and two of the three Key Points.

The cause is ordinary. Those sentences were true when written; the results were
promoted later on the authors' instruction; nothing connected the promotion to
the prose describing the old arrangement. The existing gate checked whether the
word "descriptive" appeared in a promoted claim's span — and it did, on *both*
sides of the contradiction.

Fixed by stating the promotion as a deliberate choice at the head of Section 4,
with its cost stated, labelling every quotation at the point of use, and adding
`check_placement_denials_are_true`, which fails the build when the manuscript
denies a placement the claim ledger records. SI20 §9 now maps every headline
number to its grade, its specification date, and its artifact.

### B04 — the fallacy was real, and the correct test was already sealed and unreported

**The fallacy is confirmed.** Section 4.10 said an effect below the minimum
detectable effect was "evidence of absence". That is backwards, and it
contradicted this study's own sealed contract, which records
`failure_to_reject_is_equivalence: false`. Removed.

**But the recommended fix — declare a negligibility margin and test against
it — was already done, in 2026-07-22, and the manuscript was not reporting it.**
`protocols/route_a_confirmatory_v1.json` registers the two LightGBM rows as
**non-inferiority** tests at a +0.05 °C margin, sealed under an attestation that
no post-2020 outcome had been requested or inspected. The manuscript computed
all five sign-flip p-values against a margin of zero, turning a registered
non-inferiority test into a superiority test nobody registered, and reported the
two rows as failures (Holm 1.0 and 0.15).

At the sealed margins:

| Test | Kind | Margin | ΔRMSE | 95% CI | p | Holm p | Sealed decision |
|---|---|---:|---:|---|---:|---:|---|
| H1-h1-vs-damped | superiority | 0.00 | −0.129 | [−0.199, −0.076] | 3.1e-05 | 1.5e-04 | SUPPORTED |
| H1-h3-vs-damped | superiority | 0.00 | −0.108 | [−0.143, −0.074] | 3.1e-05 | 1.5e-04 | SUPPORTED |
| H1-h7-vs-damped | superiority | 0.00 | −0.069 | [−0.084, −0.057] | 6.1e-05 | 1.8e-04 | SUPPORTED |
| H2-h3-vs-lightgbm | **non-inferiority** | +0.05 | +0.015 | [0.010, 0.025] | 6.1e-05 | 1.8e-04 | SUPPORTED |
| H2-h7-vs-lightgbm | **non-inferiority** | +0.05 | −0.009 | [−0.017, 0.003] | 6.1e-05 | 1.8e-04 | SUPPORTED |

Verified independently before being written down, because a correction that
favours the paper deserves more scrutiny than one that does not: at both leads
all 15 HUC2 cluster medians fall below the margin, and 90.5% (3 d) and 96.6%
(7 d) of individual stations do. The result is not an artefact of the median.

Three limits are stated in the manuscript and are not negotiable:

1. The protocol's own rationale records that +0.05 °C is **not** derived from
   sensor precision, biological response, or regulation, and its
   `noninferiority_wording_limit` forbids "equivalent", "parity", "equally
   good", "ecologically negligible". The manuscript says *numerical ceiling*.
2. Row 4 is a supported non-inferiority result at a lead where the tree is
   nevertheless the better model, by +0.015 °C.
3. The seal is repository-internal Git and SHA-256 evidence. Its own
   attestation block records no external timestamp, no public pre-registration,
   and no independent custodian, and its scope field says it is sufficient for
   an honest owner and is not proof against a repository owner rewriting
   history. **The review's caution on this point was right and is now printed in
   Section 4.4**: a referee should read the confirmatory label as auditable, not
   as attested.

No margin is declared for any architecture contrast, and one chosen now would be
chosen knowing the answer. Those stay underpowered.

---

## Blockers

| Item | Status | What was done |
|---|---|---|
| B01 submission completeness / FAIR | **not closed — authors'** | Nine placeholders, both DOIs, licence, rights review, and independent reproduction are unchanged. Not assertable by me. |
| B02 evidence governance | **closed** | Above. New gate + regression tests + SI20 §9 provenance table. |
| B03 construct validity / sensor overclaim | **closed** | "Better forecasts of the weather therefore cannot substitute for putting a sensor in the water" is gone. L2 is "observations withheld" in the text, Table 4, and the Figure 5 row labels. §4.9's heading no longer asserts a substitution result. The PLS now says explicitly that none of these comparisons tests where a thermometer should be installed. |
| B04 MDE misused as equivalence | **closed** | Above. |

On B03 the review offers "do a true-ungauged validation, or delete the
language". I took the second, and the reason is stronger than expedience: this
design cannot supply the first. A site with no thermal record supplies no target
to score against, and the cohort itself was selected on water-temperature
availability. The manuscript now says that rather than approximating it.

## Majors

| Item | Status | What was done |
|---|---|---|
| M05 cluster inference / extrapolation | **already stated; strengthened** | §3.6, §5.2 and §6 already reported 15 clusters, effective count 9.54, 21.7% largest share, and non-probability sampling. Section 4's new preamble makes the grade of every result explicit. Adding regions is data work. |
| M06 anchor equifinality | **closed** | Figure 5 gains a *Reference construction* row: the seven-day effect across the seven predeclared anchor variants, −0.045 to −0.108 °C. It is the row that shows the reference specification is larger than several effects the figure invites a reader to compare. Flagged in the caption and in the data file as a range over specifications, not a bootstrap interval. |
| M07 baseline fairness | **closed as scoping** | Every "model class" claim is now about two fitted objects with documented, unequalized budgets. §4.10's heading changed accordingly. Equalizing budgets across families is compute work. |
| M08 spatial validation | **already disclosed** | §4.5 already reported the geometry effect as unresolved and refused to claim one. The gauged-vs-ungauged naming split the review asks for is now carried consistently. |
| M09 operational validity | **already disclosed; boundary sharpened** | F2b is unrun and labelled; F2a is named a fixed-lead composite, not a coherent trajectory. The Introduction now places the operational literature on the far side of this study's boundary. |
| M10 QA/QC and units | **partly closed** | Discharge is served *and consumed* in ft³ s⁻¹ — the manuscript briefly claimed a conversion that does not exist in this pipeline, and now states the served unit, the SI factor, and that standardization makes the unit immaterial to results. Negative discharge (2,059 rows) and the observed temperature range are disclosed. The full station×variable qualifier ledger remains SI12's. |
| M11 physical interpretation | **not closed** | Point-scale forcing is already declared a limitation. Catchment-integrated forcing and a heat-budget check are new compute. |
| M12 common-key auditability | **partly closed** | §2.2 described the registry as "the intersection of admissible keys across all primary models", which reads as model-dependent and is not what the protocol specifies. Admissibility is four data conditions — window, observed issue-date temperature, a buildable 32-day history, observed target — none of which mentions a model, and every declared model must then produce a row on every admissible key, so declining a hard day is a failure rather than an exclusion. That is now stated. What remains open is the review's stronger ask: a published registry-generation script with hashes, so a reader can re-derive the key set rather than take the rule on trust. |
| M13 implementation defects | **already disclosed** | The pooled-arm fold defect and the auxiliary-path mask gap are both stated, and no number from the pooled arm reaches the Discussion. |
| M14 figures and version consistency | **closed** | Tables renumbered 1–4. Figure 3's caption described four panels and a per-station memory-fraction panel that has never existed; corrected, and the claim formerly bound to that caption unbound. Figure 4's caption now names the model *and* the artifact, and explains the +0.02/+0.03/+0.01 against +0.006/+0.009/+0.007 discrepancy as two different models. A sign-convention error in §4.4's plain-TCN contrast is fixed. |
| M15 literature | **closed** | Six works added and engaged, not listed. Feigl et al. (2021) was already in the bibliography and had never been cited. |
| M16 management relevance | **closed by deletion** | The deployment framing is gone; no cost model is claimed. |
| M17 stratified analysis | **closed by relocation** | The sixteen-stratum table is in SI11, with both reasons stated: stations qualified on ≥100 all-keys targets are scored inside strata on a median of 71, and the strata are not a multiplicity family. |
| m18 AGU format | **closed** | PLS 205 → 191 words against a 200-word limit. The Abstract had *also* drifted, to 289 against a 250-word limit, which the review did not flag and nothing in the build measured; it is 249. Both limits are now enforced by `_check_front_matter_lengths` in the generator, beside the Key Point character limit that was already checked. Key Points each ≤140 characters and each self-labelling ("Confirmatory:" / "Descriptive:"). Both CONUS maps carry a graticule. |
| m19 conformal keyword | **closed** | Removed; no conformal result is in the main text. |

---

## Two places where the review was working from less than the repository holds

Both are the review's own stated caveat — it saw 23 pages of PDF and no
`outputs/`, `protocols/`, or SI — so neither is a criticism of it.

1. **Section 4.4's confirmatory identity is not merely self-claimed.**
   `route_a_protocol_seal_v1.json` exists, is dated 2026-07-22, carries SHA-256
   hashes of the protocol at two commits, and attests that no post-2020 outcome
   had been requested. What the review is right about is the *limit* of that
   evidence, and the protocol says so more bluntly than the manuscript did. That
   sentence is now in the paper.

2. **A pre-declared negligibility margin already existed.** See B04. The review
   asks for one to be defined and tested; the correct move was to report the one
   that had been sealed thirteen months of project time earlier and then
   mis-rendered.

The review is also right that neither of these was visible to a reader, which is
the actual failure. Both are now printed.

---

## What a referee will still be able to say, and I agree with them

- The information results are the paper's most interesting findings and its
  least tested ones. They are post-outcome on a cohort that had already been
  read. No amount of labelling changes that, and the paper now says so in the
  same breath as it quotes them.
- Fifteen HUC2 clusters with an effective count of 9.54 will not carry national
  inference, and the paper does not attempt it.
- One tree and one convolutional network is a thin basis for anything said about
  estimators, which is why the paper no longer says anything about model
  *classes*.

## Current state

| | |
|---|---|
| Publication units | 26.4 against a threshold of 25 — inside the two-unit tolerance; a fee, not a defect |
| Pages | 36, double-spaced submission format (`draft`) |
| Figures / tables | 5 / 4 (one table relocated to SI11) |
| Consistency gate | passes, with two new checks |
| Figure typography | 16 figures, 0 problems |
| Blocking on the authors | 9 placeholders, 2 DOIs, licence, rights review, independent reproduction |

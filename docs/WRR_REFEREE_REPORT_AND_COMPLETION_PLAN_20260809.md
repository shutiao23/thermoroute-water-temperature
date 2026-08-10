# WRR referee report and completion plan — 2026-08-09

Two documents in one file.

- **Part I** is a referee report written as a *Water Resources Research* reviewer
  would write it after reading `paper/ThermoRoute_paper.md`,
  `paper/agu_submission/ThermoRoute_WRR.pdf` (41 pp.), `paper/si/`, and the
  Open Research statement. It reviews the manuscript *as it currently stands*,
  not the intended information-regime rewrite.
- **Part II** is the complete inventory of work that has not been done, tiered
  and sequenced, with a definition of done for each item.

Everything asserted in Part I was checked against the repository on 2026-08-09.
Line numbers refer to `paper/ThermoRoute_paper.md` at that state.

---

# Part I — Referee report

**Manuscript:** *Benchmark design governs reported skill in daily river
water-temperature prediction*
**Recommendation:** **Reject with encouragement to resubmit.** The manuscript is
not currently reviewable at editorial level (Major 10), and the scientific core
as framed does not clear the WRR novelty bar (Major 1). The underlying study is
careful, honest, and unusually well controlled; the problem is what it is being
asked to claim, not how it was executed.

## Summary of the submission

The authors assemble a 120-gauge USGS panel (2006–2023; 116 gauges reportable on
a 2021–2023 held-out window), score a constrained deep predictor
("ThermoRoute" = damped-persistence anchor + bounded TCN residual) against
persistence, damped persistence, climatology, LightGBM with and without site
identity, a global LSTM, an information-matched plain causal TCN, a plain MLP,
and an unofficial air2stream-style hybrid, all on one common
station/date/horizon key registry with strictly backward-fitted preprocessing.
They report that seven-day skill is +0.250 against persistence but +0.038
against damped persistence; that a gradient-boosted tree with site identity is
the most accurate model at 1 and 3 days; that whole-region holdout costs about
0.01 °C relative to random-site holdout while removing local thermal history
costs about 0.2 °C; and that residual learned skill concentrates in rapid
warming and cooling states.

The evaluation machinery is the strongest part of the paper. Common-key scoring
with no model-specific complete-case sets, strictly backward-fitted
preprocessing statistics, station-first paired estimands, explicit separation of
ΔRMSE from skill, and a self-disclosed effective cluster count are all things I
routinely have to ask authors for and rarely get. I want to say that clearly
before the criticisms.

## Major comments

### Major 1 — The central claim is not new, and the manuscript does not engage the literature in which it is old

The proposition that reported skill is a statement about the reference model,
and that a seasonal/damped anchor absorbs most of the apparent gain for a
strongly autocorrelated variable, is established in hydrological forecast
evaluation. The manuscript's 44 references contain none of it. There is no
Klemeš (1986) differential split-sample test — which is the direct ancestor of
the whole-region holdout argument in §3.4 — no Schaefli & Gupta (2007) on
benchmark-relative efficiency, no Seibert (2001) / Seibert et al. (2018) on
upper and lower benchmarks, no Pappenberger et al. (2015) on how to know
whether a forecast is better, no Knoben et al. (2019) on the mean-flow
benchmark, no Nearing et al. (2018, 2021) on benchmarking and information
limits, no Kratzert / Newman / Addor line on split design in
deep-learning hydrology. There is likewise no Blöschl, Sivapalan, Wagener,
Hrachowitz, Kirchner, or Troch: the comparative-hydrology community whose
journal this is does not appear in the reference list at all.

As written, §1 constructs a strawman ("the prevailing understanding is that deep
sequence models have substantially advanced daily river-temperature
prediction") and then refutes it with a result that a reader of the
benchmarking literature already expects. That is not a WRR paper.

The genuinely new material is the *quantitative* one: an exact station-first
paired decomposition of a reported skill number into memory gain and learned
gain on a fixed key registry, with 0.875 of the median station-level error
reduction attributable to seasonal memory. That is worth publishing, but only if
it is positioned as a measurement inside a known frame, with the frame properly
cited, and only if it is measured on a cohort that can carry it (Major 3).

**Required:** rebuild §1 and §5.1 around the existing benchmarking literature;
state precisely which quantity has not previously been measured for this
variable; delete every sentence that implies the field is unaware of baseline
sensitivity.

### Major 2 — The headline contrast is partly definitional and is not labelled as such

Equation (1) makes the damped-persistence anchor a *component of ThermoRoute*.
The model's prediction is anchor + bounded residual. Therefore
"skill of ThermoRoute against damped persistence" is, by construction, the value
of ThermoRoute's own residual head, and "skill against persistence" is that plus
the value of a component the model already contains. The 6.6× ratio in §4.1 is
substantially an arithmetic property of the model's own decomposition, not an
empirical discovery about deep learning.

This matters for the generalization in §5.1 and in the Conclusions. LightGBM and
the LSTM do *not* contain the anchor; their skill against damped persistence is
a genuinely different quantity and is the one that speaks to the literature's
architectures. The manuscript never separates these two cases and quotes only
the ThermoRoute number in the abstract and Key Point 1.

**Required:** state explicitly that the anchor is internal to ThermoRoute; move
the anchor-free models (LightGBM, LSTM, plain TCN, plain MLP) to the front of
the reference-sensitivity analysis, since they are the ones the argument
actually generalizes to; and report the persistence-vs-damped skill ratio for
each of them, not only for the model that contains the anchor.

### Major 3 — The cohort cannot support the paper's generality, and the manuscript oscillates between admitting this and ignoring it

The cohort is 120 gauges retained from 1,465 candidates by an availability
filter (950 rejected for lacking joint temperature *and* discharge), spanning 15
HUC2 regions with an inverse-Herfindahl effective cluster count of 9.54 and a
largest group holding 21.7% of stations. The manuscript says so, repeatedly and
creditably (§2.1, §3.6, §5.2, §6). It then writes, in §5.3: *"benchmark design is
a larger source of variation in reported river-temperature skill than model
design is"* — a statement about the field — and in §5.1: *"We would expect much
of the between-study dispersion noted by Corona and Hogue (2025) to be absorbed
by re-scoring published models against a fitted damped anchor."*

That expectation is the paper's most interesting claim and it is untested. A
disclosure paragraph does not license the claim; it just documents that the
authors know it is unlicensed.

**Required, choose one:**
(a) Test it. Take a set of published daily river-temperature results, re-score
    them (or re-implement the models) against a fitted damped anchor on your
    registry, and report the shift in the reported number. This turns the paper
    from an assertion into a measurement and would make it a strong WRR paper.
(b) Expand the cohort so the claim is at least defensible in-sample. The
    temperature-only cohort (no joint-flow requirement) is the obvious move: it
    removes the 950-station availability filter that drives the selection
    problem, plausibly reaches 300+ gauges, and raises the effective cluster
    count above the level at which the clustered inference in §3.6 becomes
    something other than decorative.
(c) Demote every field-level statement to "on this cohort", including in the
    title, the abstract, Key Point 1, and the Conclusions. This is the cheapest
    path and the one that most reduces the paper's interest.

### Major 4 — The spatial-transfer result is at the noise floor, and the informative arm is contaminated

§4.5 reports a region-minus-random penalty of +0.006 / +0.009 / +0.007 °C at
1/3/7 days, against a median per-site standard deviation across the five random
split seeds of 0.004–0.007 °C. The effect and its own resampling noise are the
same size. Describing this as "a small, consistently signed transfer penalty
that is reproducible across models, seeds, and leads" over-reads five seeds and
four folds. With 4 region folds there are effectively four independent
whole-region observations; the sign consistency across seeds is largely a
statement about the same four folds being re-used.

Separately, the reason the geometry effect is near-null is structural and should
be stated as a limitation of the design rather than as a finding: every held-out
station in every arm still supplies its own observed WTEMP through issue time,
its own climatology, and its own φ. The anchor therefore carries the prediction,
and the learned component that would have to transfer spatially is contributing
~0.07 °C in total at seven days. A design in which the transferable part of the
model is 4% of the error budget cannot resolve a spatial-transfer effect.

The arm that *does* remove local information — pooled adaptation, +0.20 °C at
seven days — is the scientifically interesting one, and the manuscript itself
declares it defective: DLOG-012 records that fold 0's pooled preprocessing
statistics were reused across folds 1–3, whose held-out stations are inside fold
0's training set. §4.5 calls the pooled numbers "a lower bound". A reviewer
cannot accept a headline conclusion ("the spatial partition is secondary
specifically given local thermal history", stated in the abstract, §4.5, and the
Conclusions) whose supporting contrast is a contaminated arm the authors have
labelled a lower bound.

**Required:** rerun the pooled arm with per-fold preprocessing before any version
of this claim is made; report the geometry effect as unresolved at this design's
precision, with the seed spread shown beside it; and add the no-local-history
levels (L2 / L2-U2 in your protocol v2/v4) so that the transfer question is
asked of a model that actually has to transfer.

### Major 5 — Key Point 3 and the abstract promote an outcome-conditioned diagnostic to a headline

Key Point 3: *"Learned skill beyond the seasonal reference concentrates in rapid
warming and rapid cooling states."* Abstract: *"the learned model's residual
value concentrates in rapid warming and rapid cooling states (−0.30 and −0.18 °C
station-median paired ΔRMSE at seven days)."*

Those two values are Table 4.12's `actual rapid warming` and `actual rapid
cooling` rows — strata defined by the *realized* change at the target date. The
manuscript labels them correctly inside §4.6 as "retrospective
outcome-conditioned diagnostics, not issue-time states", and then puts them in
the abstract and a Key Point without that qualifier.

Conditioning on the realized outcome selects exactly the days on which a
persistence-type anchor must fail. A model that corrects the anchor will
therefore always look best there. The result is close to tautological and has no
operational content: a manager cannot identify these days at issue time.

The issue-time counterparts in the same table are −0.088 (`issue rapid recent
warming`) and −0.170 (`issue rapid recent cooling`), and the largest issue-time
effect is −0.229 on `issue low anomaly` — which is not what the Key Point says.

**Required:** rewrite Key Point 3 and the abstract sentence using issue-time
states only; keep the outcome-conditioned strata in §4.6 as a labelled
diagnostic; do not carry them into any summary-level statement.

### Major 6 — No uncertainty on the primary decomposition, and no sensitivity on the anchor the whole paper depends on

The memory-gain / learned-gain decomposition (0.49 °C / 0.07 °C, memory fraction
0.875) appears in the abstract, Key Point 1's argument, Figure 3, §4.6, and the
Conclusions. It carries no interval anywhere in the manuscript or the SI. Every
other headline in the paper has a cluster-bootstrap CI; this one does not.

More fundamentally, the damped anchor is the paper's instrument, and its fitting
choices are unexamined. φ is fitted per station by no-intercept least squares on
training-period consecutive-day anomaly pairs, requires ≥30 pairs, and is
clipped to [0, 0.999]; the lead-h anchor is φ^h. There is no sensitivity to:
the climatology form (day-of-year vs harmonic, window width, robustness to the
2,404-day WTEMP gaps); the φ fitting window (2006–2015 vs 2006–2018); pooled vs
per-station φ; φ^h extrapolation vs a directly fitted lead-specific φ_h; or the
clip. Since "the reference model governs the reported gain" is the paper's
thesis, the paper must show that the specific reference it constructed is not
itself a tuning choice that sets the answer. A reviewer will read the current
version as: the authors built a strong baseline, did not test how strong it
could have been, and then concluded that baselines matter.

**Required:** cluster-bootstrap intervals on the decomposition quantities; a
one-page anchor-sensitivity table in the SI covering at minimum the climatology
form, the φ window, pooled-vs-per-station φ, and the direct φ_h fit; and a
statement of how much the headline +0.038 moves across those variants.

### Major 7 — The inference is disowned and then used

§3.6 and §5.2 state that the clustered intervals and p-values are "approximate
descriptive sensitivities, not decision evidence" and that "no comparison here
is written as a superiority, non-inferiority, equivalence, or parity statement".
Table 4.6 nonetheless prints Holm-adjusted p-values to two significant figures,
and the abstract states that the seven-day ThermoRoute–LightGBM difference "is
not distinguishable from zero at the whole-HUC2 cluster level". That is an
equivalence statement, drawn from a procedure the authors say cannot support
one, based on a non-significant result with no power analysis.

This is self-inconsistent in a way that a reviewer must flag, because it is the
one place where the paper's own stated epistemic standard is violated.

**Required, choose one:** (a) commit to the clustered procedure, justify it, and
add the leave-one-HUC2-out range and a minimum-detectable-effect calculation
beside every p-value; or (b) drop p-values entirely and report effect sizes with
station-level intervals plus the LOCO range, and replace "not distinguishable
from zero" with the interval itself. Option (b) is more honest and costs the
paper nothing.

### Major 8 — Missing references and benchmarks

- **No valid process benchmark.** The air2stream-style model is described five
  times as "an unofficial empirical thermal-recurrence variant of the published
  formulation — not the official code and not a validated reproduction". A
  reference that the authors will not stand behind is not a reference. For a WRR
  audience, a paper about river-temperature prediction with no defensible
  process-based or hybrid comparator is incomplete. Either run the published
  air2stream code, or freeze a transparent equilibrium-response model you *will*
  defend, or remove the arm.
- **No future-forcing arm.** All primary models are issue-time-only, so the
  seven-day results are not comparable with any operational forecasting study,
  and the paper cannot say whether the small learned gain at seven days reflects
  a model limitation or an information limitation. This is the single largest
  scientific gap. An oracle (realized gridded meteorology at t+1…t+h) and a
  realistic forecast arm would let the paper state how much of the seven-day
  error is irreducible without future weather — a far more interesting result
  than the one currently reported.
- **No spatial-analog baseline.** For the transfer arms, the natural cheap
  reference is nearest-gauge or regional-regression analog transfer. Without it,
  "whole-region transfer costs 0.01 °C" has no scale.
- **No upper benchmark.** In the Seibert et al. (2018) sense: what is the best
  achievable on these keys? Without it the 0.07 °C learned gain has no
  denominator.

### Major 9 — Open Research statement is not compliant

§8 currently says: the data DOI is `[DATA DOI TO BE MINTED]`, the licence is
`[DATA LICENCE TO BE ASSIGNED]`, the software DOI is `[SOFTWARE DOI TO BE
MINTED]`, the repository URL is `[REPOSITORY URL TO BE CONFIRMED]`, the deposit
"is described here as planned and specified, not as existing", original provider
responses for the 2006–2020 development panel were not retained, and
"reproduction has not yet been carried out by an operator independent of the
authors on an independent host".

AGU's Open Research policy requires that data and software be *available* at
submission, in a FAIR-compliant repository, with a persistent identifier. A plan
is not a deposit. This alone would return the manuscript at editorial check.

The candour is admirable and should be preserved in the Limitations; it cannot
substitute for the deposit.

### Major 10 — Production defects; the manuscript would be returned before review

I list these because several are the kind that make an editor stop reading.

1. **Abstract is 359 words** against WRR's 250-word limit.
2. **Eleven unresolved placeholders**: author list, two affiliations,
   corresponding author, funding, computational resources, competing interests,
   CRediT roles, data DOI, data licence, software DOI, release tag, repository
   URL.
3. **A status note at the top of the manuscript** ("this manuscript is under
   active revision following external review… labelled `[PENDING AUTHORITY
   VALUE]` until regeneration completes"). This must not exist in a submitted
   file.
4. **Figure numbering is broken in the compiled PDF.** Each figure is
   double-captioned: a LaTeX `\caption{}` (auto-numbered 1, 2, 3, 4) sits
   directly above a hand-written body paragraph numbered 1, 3, 5, 6. The PDF
   therefore contains "Figure 2. Decomposition of reported skill on the held-out
   window." immediately followed by "Figure 3. Decomposition of reported skill
   on the held-out 2021–2023 window…", and the same collision at 3/5 and 4/6.
   There is no Figure 2 and no Figure 4 in the main text; the numbering has gaps
   left by figures relocated to the SI.
5. **Dangling section cross-references.** §2.3, §3.1, §3.2 and §8 cite
   Sections 6.1, 6.2 and 6.4; Section 6 has no subsections. §3.4 and SI11 cite
   "Section 4.7"; the spatial factorial is §4.5.
6. **Duplicated heading**: `## 6. Limitations` immediately followed by
   `### 6 Limitations`.
7. **Undefined equations.** The text cites equations (5), (7), (9) and (10);
   only (1), (2) and (3) are defined. §3.6 says "Equation (9) is the formal
   estimand" — the estimand is equation (2). The Table 4.7 caption cites
   "equation (10)" for the skill score — that is equation (3). §3.3 cites
   "Equation (5)" for non-anticipating encoding and §4.2 cites "equation (7)"
   for the algebraic bound; neither appears.
8. **Missing tables.** The caption block reads "Tables 4.7–4.8"; only Table 4.7a
   is rendered, followed by ~10 blank lines.
9. **Table numbering** is section-relative (4.6, 4.7a, 4.12), implying Tables
   4.1–4.5 and 4.8–4.11 that do not exist. AGU requires Table 1, 2, 3, …. Two
   further tables (the §3.6 comparison set, the §4.1 development RMSE table) are
   unnumbered and uncaptioned.
10. **Markdown image paths do not resolve.** `figures/fig01_study_design.pdf`
    etc. are relative to `paper/`, but the assets are in
    `paper/agu_submission/figures/`; `paper/figures/` does not exist.
11. **Supporting Information is incomplete.** Figures S4–S10 are referenced;
    only S1–S3 exist in `paper/si/figures/`. `[pending computation]` cells
    remain in SI02, SI03, SI04, SI05, SI09, SI10, SI14, SI15 and `si/README.md`.
    SI08 reports nine probability metrics as "not computed" on every row
    (interval score, pinball, log score, discrimination, ECE, calibration slope
    and intercept). SI12's qualifier-restricted sensitivity was not recomputed
    on the held-out window. SI11's `claim_eligible` is hardcoded `False`.
12. **41 pages** against AGU's ~25 publication-unit guidance.
13. The final SI paragraph still refers to "Figures 2 and 4" in the TeX, which
    no longer exist.

### Major 11 — Title, Key Points, and Conclusions do not describe the same paper

The title says benchmark design governs reported skill. The largest measured
effect in the manuscript is the local-adaptation effect (+0.20 °C at seven
days), which is about *information availability*, not benchmark design — and it
comes from the contaminated arm (Major 4). The three Key Points are
(i) reference sensitivity, (ii) model ranking, (iii) hydrologic states; the
three findings in the Conclusions are (i) reference sensitivity, (ii) model
ranking, (iii) spatial partition. Key Point 3 and Conclusion 3 are different
results.

## Minor comments

1. "Causal" is used throughout for time-ordering. The disclosure in §6 helps,
   but "non-anticipating" is the standard term and avoids the issue entirely.
2. §2.1 retains 2,059 rows of signed negative discharge without discussing the
   effect on log or ratio features, or on the flow-state quantiles used in
   Table 4.12.
3. §2.1 notes that 38 stations share a HUC with another retained station and 19
   have a neighbour within 10 km. This is the real independence problem and it
   is stronger than the HUC2-cluster machinery addresses; a distance-decay or
   spatial-block analysis would be more convincing than HUC2 resampling.
4. The 0.07% meteorological missingness is fully structured (all 120 sites lack
   the provider calendar's last day in four leap years). This is fine but should
   be stated as a calendar artifact rather than as missingness.
5. Legacy variable names `DH` (Daymet `srad`) and `RHMEAN` (a vapour-pressure
   proxy) are explained but should simply be renamed.
6. Tuning budgets are explicitly not equalized across model classes (§3.2). The
   scoping sentence is good; a reviewer will still ask for a matched-budget
   run of LightGBM and the LSTM, since the paper's second headline is a model
   ranking.
7. Training and inference wall-clock costs were not recorded (§6). For a paper
   whose practical recommendation is "prefer the tree", cost is part of the
   recommendation.
8. §4.4's "1.694 versus 1.735 °C" and the abstract's "1.735 versus 1.694 °C"
   order the same pair differently; the consistency gate flags this as a
   contradiction warning.
9. Figure 3's caption mixes an equal-station *mean* waterfall (0.456 + 0.070 =
   0.527) with *median* per-station gains (0.491, 0.069) in the same panel. Pick
   one and put the other in the SI.
10. §4.2 reports the 1-day ThermoRoute-vs-LightGBM comparison as "outside the
    frozen five-test family… as an exploratory check". Good practice, but the
    result then appears in Key Point 2 and the Conclusions as a headline. An
    exploratory result cannot be a Key Point.

## What would change my recommendation

A resubmission that (i) is positioned inside the existing benchmarking
literature with a specific, measured contribution; (ii) reports a
future-forcing arm so that the seven-day results speak to an information limit
rather than only to a baseline choice; (iii) reruns the local-adaptation
contrast without the fold-0 preprocessing defect; (iv) drops or fixes the
outcome-conditioned headline; (v) either expands the cohort or scopes every
claim to it; and (vi) arrives with a real data and software deposit and a clean
PDF, would be a strong candidate.

---

# Part II — What has not been done

Tiers are ordered by what blocks what. Within a tier, items are independent
unless a dependency is stated. "DoD" = definition of done.

## Tier 0 — Framing and scope decisions (do these first; they change what you build)

These are decisions, not engineering, and every downstream item depends on them.
Making them after the experiments run is how the current situation arose.

### 0.1 Decide whether this is one paper or two

My recommendation is two, and this is the single most consequential item in this
document.

- **Paper A — the benchmark-design paper.** Essentially the current manuscript,
  descoped: reference sensitivity, model ranking on identical keys, the
  memory/learned decomposition, issue-time state analysis. Drop the spatial
  claim to a limitation. Fix Part I Major 1, 2, 5, 6, 7, 9, 10, 11. Do **not**
  wait for the F/L ladders. Realistic scope: 3–5 weeks. Risk: Major 1 and
  Major 3 remain; a reviewer may still find it thin.
- **Paper B — the information-regime paper.** The F × L × G × A response
  surface: how much short-range river-temperature predictability comes from
  local thermal state, from future atmospheric forcing, from spatial
  information, and from model class. This is a genuinely novel, quantitative,
  and useful question, it is what the v4 protocol is built for, and it is where
  the remaining engineering should go. Realistic scope: months.

The current plan — rewrite the single manuscript into Paper B and hold
everything until the ladders are correct — means nothing is submitted until the
entire program closes. Splitting lets Paper A absorb the production, literature,
and FAIR work now, and lets Paper B inherit a clean apparatus.

**DoD:** a written decision in `docs/strong_accept_decision_log.md` naming which
manuscript each frozen result belongs to.

### 0.2 Cut the governance investment to ~5% of remaining effort

#### The ledger

Measured on 2026-08-09.

| Subsystem | Lines (code + tests) |
|---|---:|
| `src/thermoroute/formal/` + `test_formal_authority.py` | 1,458 |
| `build_semantic_data_registries_v4.py` + test | 3,238 |
| `build_semantic_contract_registries_v4.py` + test | 4,477 |
| `build_information_regime_key_registries_v4.py` + test | 4,438 |
| `build_forcing_regime_authority.py` + test | 2,280 |
| `build_information_regime_authority.py` + test | 1,473 |
| `build_preprocessing_lineage_defect_authority_v1.py` + test | 2,822 |
| **Authority / registry subtotal** | **20,186** |
| `run_forcing_ladder_v5_observed.py` + test | 4,686 |
| `run_neural_information_regimes.py` + test | 6,669 |
| `run_l2_u2_information_regime_v4.py` + test | 3,866 |
| **Plan-only runner subtotal** | **15,221** |
| **Total** | **35,407** |

None of the three "runners" imports `torch` or `lightgbm`. They cannot fit a
model. They are 15,221 lines of admission control for 5,332 fits that do not
exist. The whole apparatus has produced zero fitted models.

For scale, the one governance subsystem with genuine reviewer-visible value —
`check_manuscript_consistency.py` + `generate_manuscript_tables.py` +
`audit_used_in.py` + the claim ledger — is 646 lines.

#### Why more of it has no expected value

DLOG-025 is the only defect in this project's history that destroyed results.
The authority layer did not catch it and **structurally could not**: that layer
verifies artifacts against manifests, i.e. self-consistency. The defect was
semantic — imputed values entered training labels — and a manifest of an
imputed label is a perfectly valid manifest. It was found by two people reading
the production `loader → imputer → feature builder → training-row selection`
path, which is a one-day activity.

So the marginal line of authority/seal/attack-audit code has ~zero expected
value against the next defect of the same class. This is the argument for
cutting; the budget argument is secondary.

The order was also inverted: the turnstile was built before the stadium.

#### Keep / freeze / cancel

**Keep and extend (~650 lines, the only governance worth more effort):**
- `check_manuscript_consistency.py`, `generate_manuscript_tables.py`,
  `audit_used_in.py`, `paper/claim_ledger.yaml`, `build_results_authority.py`.
  This makes "every number in the paper comes from an artifact" true and
  testable, and it is the only part a referee benefits from.
- The **raw-observedness capture** inside `build_semantic_data_registries_v4.py`.
  This is not governance — it is the scientific fix to DLOG-025.

**Freeze (do not extend, do not delete, ~30,000 lines):**
- `src/thermoroute/formal/authority.py` and its symlink / hardlink / FIFO /
  device / path-escape / Unicode-surrogate hardening. This defends against an
  adversary with write access to your own filesystem. There is no adversary.
- The three plan-only runners.
- `build_forcing_regime_authority.py`, `build_information_regime_authority.py`,
  `build_preprocessing_lineage_defect_authority_v1.py`. All three have served
  their purpose; DLOG-025/026 require their bytes to be preserved anyway.
- All seal machinery beyond one digest check.

Freezing means: a `# FROZEN` header, an entry in the decision log, no new
features, no new audits, no refactors. Bug fixes only when a test breaks.
Do **not** spend time deleting them — deletion is also effort, and the
withdrawn-result bytes must be preserved.

**Cancel outright:**
- The planned independent contract-registry attack audit. Replace it with the
  lineage test below.
- Any further work on `formal/authority.py`.
- Any new create-only authority for a result that has not been computed yet.

#### The four contract-registry fixes, re-scoped

Two of your four items are science, two are governance. Treat them differently.

| Item | Verdict | Scope |
|---|---|---|
| 1. Separate folds from evaluation partitions | **Science — do fully.** It defines the estimator; 45 folds vs 44 changes what the numbers mean | Tier 1.1 as written |
| 2. Geometry-specific split-seed contracts | **Science — do fully.** Seed-mean-of-within-seed-pairs vs single-partition is an estimand definition | Tier 1.2 as written |
| 3. Verify the full runtime environment | **Governance — do the 30-line version.** *Record* `pip freeze` + versions + thread counts into a JSON beside every output. Do not build declared-vs-observed verification, do not fail closed | ~30 lines |
| 4. Data-candidate trust boundary | **Governance — do only the one test.** Construct a candidate with one imputed WTEMP cell flagged `observed=True` and assert rejection. Skip the provenance allow-list and the sampled-slice re-verification framework | ~60 lines |

#### What replaces 35,000 lines

Four things, roughly 600 lines total:

1. **The lineage test** (~150 lines). Runs the real loader / imputer / feature
   builder over a fixture panel with known missing cells and asserts:
   no imputed WTEMP reaches a training label; no imputed issue-date WTEMP passes
   admissibility; history-missingness features equal the raw record; the
   validation set is disjoint from the training rows. This is the single
   highest-value test in the repository and it does not exist.
2. **The two-sided key assertion** (~40 lines, mostly exists in
   `assert_shards_in_registry.py`): no required key missing from a cell, no
   outside key admitted, target bytes identical across cells sharing a key.
3. **The environment record** (~30 lines): write it, don't verify it.
4. **The consistency gate and claim ledger**, already built.

Plus one rule that is not code: run each experiment once, record what you did in
the decision log, and stop.

#### Stopping rule, so this does not recur

Before writing any verification code, name the specific defect in this project's
history that it would have caught. If you cannot name one, do not write it.
By that test, DLOG-025 justifies item 1 above and nothing else in the 35,407
lines.

**Budget:** cap all remaining governance work at ~5% of effort. Every hour in
that layer is an hour not spent on the literature program (0.4), the anchor
sensitivity analysis (4.8), or the expanded cohort (0.3) — the three items with
the highest reviewer-visible value per unit effort in this document.

### 0.3 Decide the cohort

Choose now, because it determines whether the corrected ladder runs once or
twice:

- keep the 120-station WTEMP+FLOW cohort and scope every claim to it, or
- build the temperature-only cohort (no joint-flow requirement, target ≥300
  gauges, more HUC2 regions) and run the ladder on it.

Option 2 fixes Major 3 and the effective-cluster-count problem simultaneously
and is the highest-scientific-value item in this entire document. It also costs
one full re-acquisition and re-run. Deciding it after the 432-cell ladder
completes would waste that run.

**DoD:** frozen cohort definition, candidate-rejection ledger, and HUC2
distribution with effective cluster count, recorded before any corrected fit.

### 0.4 Literature program

Not optional, and not something to do at writing time.

- Read and cite: Klemeš (1986); Seibert (2001); Schaefli & Gupta (2007);
  Pappenberger et al. (2015); Seibert, Vis, Lewis & van Meerveld (2018);
  Knoben, Freer & Woods (2019); Nearing et al. (2018, 2021); Kratzert et al.
  (2019); Newman et al. (2015); Addor et al. (2017); Blöschl et al. (2019, the
  23 unsolved problems); at least one Wagener/Hrachowitz/Kirchner
  comparative-hydrology position piece.
- Add the river-temperature benchmarking studies that already use strong
  baselines, so the paper is not claiming to be first.
- Re-read Corona & Hogue (2025) and extract the actual reported spread; the
  paper currently gestures at it.

**DoD:** ≥15 new references; §1 and §5.1 rewritten around them; a paragraph that
states, in one sentence, what has not previously been measured.

## Tier 1 — Contract-registry corrections (blocks all execution)

Your four items, plus three I would add. All are in
`scripts/final/build_semantic_contract_registries_v4.py`.

### 1.1 Separate CV folds from evaluation partitions *(your item 1)*
Current build emits 45 fold records because v5's `known_site_temporal`
evaluation partition is counted as a fold. Correct structure: 40 random-site
folds (10 seeds × 4) + 4 whole-region folds = exactly 44 CV folds; v5's
known-site temporal split becomes an `evaluation_partition_id`, and v5 fits
reference it as such.
**DoD:** registry asserts `len(folds) == 44`; no fit record carries both a
`fold_id` and an `evaluation_partition_id`; test injects the old shape and
fails.

### 1.2 Geometry-specific split-seed contracts *(your item 2)*
Contrast records currently declare seeds `0..9` uniformly. Correct rule:
random-site = 10 seeds, station effect paired **within** seed then averaged
across the 10; whole-region = one deterministic HUC2 partition, no seed
averaging; the geometry contrast compares the random seed-mean station effect
against the region single-partition station effect. This is exactly the
`random_seed_first_rule` in `protocols/wrr_information_regimes_protocol_v4.yaml`.
**DoD:** contrast records carry an explicit `aggregation` field
(`seed_mean_of_within_seed_pairs` vs `single_partition`); a test asserts no
whole-region contrast declares more than one seed.

### 1.3 Verify the full runtime environment *(your item 3)*
Python/NumPy/pandas/PyArrow are verified; LightGBM, scikit-learn, SciPy, and
Torch come from declared values. Any of those four changes fit results.
**DoD:** the environment registry imports each package and records
`__version__` plus the resolved distribution metadata (name, version, and the
wheel/dist-info hash where available) at build time; a declared-vs-observed
mismatch fails closed. Also record BLAS/OpenMP thread counts and
`OMP_NUM_THREADS`/`LIGHTGBM` determinism settings — LightGBM is not
bit-reproducible across thread counts, and your existing
`tests/test_stage09_completion.py` already fails collection on a thread-cap
policy mismatch, which is the same class of problem.

### 1.4 Harden the data-candidate trust boundary *(your item 4)*
Currently a self-consistent manifest is sufficient. Add: (a) the data-builder's
own provenance (builder source hash, input panel hashes, build parameters) must
match; (b) independent semantic re-verification of a sampled slice — not just
schema and row counts — specifically that `*_observed` still equals Arrow
validity for all eight variables and that `y_true` still matches the daily
registry; (c) an explicit allow-list of accepted candidate provenance roots so
an arbitrary self-describing directory cannot be admitted.
**DoD:** an attack test that constructs a self-consistent but semantically wrong
candidate (e.g. one imputed WTEMP cell with `WTEMP_observed=True`) and shows the
builder rejects it.

### 1.5 *(added)* Lineage regression test on the production path
The defect DLOG-025 found existed because tests exercised helper functions with
hand-made flags and never the production `raw → flags → train-only fit → impute
→ feature` lifecycle. This is the highest-value test in the repository and it
does not exist yet.
**DoD:** a test that runs the real loader/imputer/feature builder over a
fixture panel with known missing cells and asserts (i) no imputed WTEMP reaches
a training label, (ii) no imputed issue-date WTEMP passes admissibility,
(iii) history-missingness features equal the raw record, (iv) the LightGBM
validation set is disjoint from the training rows.

### 1.6 *(added)* Complete the primary matrix in the plan before sealing
The current 5,332-fit inventory omits the 24 LightGBM spatial logical cells of
the 48-cell F0/F3 matrix. Sealing a contract registry whose inventory cannot
produce the paper's primary matrix means sealing twice.
**DoD:** the cell registry contains every logical cell of the declared primary
matrix, or explicitly marks the missing ones `PLANNED_DEGRADED` with the reason.

### 1.7 *(added)* Reconcile the protocol-vs-plan cell-count contradiction
`SCIENTIFIC_EVIDENCE_STATUS.md` item 6 records that protocol v2 describes 576
ladder cells plus U2 while the executed subset was 432, and that L3, L2-U2 and
N07–N09 are absent and the N01–N15 Holm family was never computed. That
contradiction must be resolved in the v4 registry, not carried forward.
**DoD:** one table mapping every protocol-declared cell to `PLANNED` /
`REGISTERED` / `EXECUTED` / `WITHDRAWN`, generated from the registry rather than
hand-maintained.

### 1.8 Then: independent attack audit + real-input smoke, re-run
**DoD:** a written audit record in the style of the formal-authority audit;
93-test focused suite extended with the new assertions; end-to-end scratch smoke
regenerated after every fix above.

## Tier 2 — Governance to authorize execution

### 2.1 Formal protocol for the corrected runs
`protocols/wrr_information_regimes_protocol_v4.yaml` is `DRAFT_NOT_SEALED`,
`execution_authorized: false`. It must be completed (every pointer, model, fold
and contrast registry resolved) and sealed with a seal that binds the exact
protocol bytes — the v2 and v3 seals do not, which is why both are
`PROVISIONAL_NOT_YET_COMPLETE`.
**DoD:** `_seal.json` whose digest matches the YAML bytes; a test that
recomputes it.

### 2.2 Preserve the post-outcome disclosure
The 2021–2023 window is open and F/L outcomes were inspected. v4 already says
this. Keep it: every corrected result must be labelled post-outcome
normalization, never "preregistered" or "independent confirmation". The
reconstructed DLOG-015 must never be cited as prospective registration.
**DoD:** a gate that fails the manuscript build on any of the strings listed in
`prohibited_chronology_language`.

### 2.3 Clean commit, source registry, runtime registry, execution authority
The worktree currently has 27 modified and 45+ untracked files, no clean commit,
no tag, no PR. Nothing can be sealed against this state.
**DoD:** a clean commit whose tree hash is bound into the execution authority;
source registry enumerating every runner and library file with hashes.

### 2.4 Fix the two failing runtime-pin tests and the collection error
Two Route-A tests pin a Python 3.11 authority serialization environment;
`tests/test_stage09_completion.py` fails *collection* under a 1-thread cap
("process thread cap 1 differs from the frozen policy cap 8 for role stage09"),
which means the full suite cannot even be enumerated in some environments.
**DoD:** full suite (~1,169 tests) collects and runs green in both environments,
or the pins are explicitly retired with a decision-log entry.

## Tier 3 — The corrected experiments

### 3.1 Twelve-cell F0/F3 correction (`run_forcing_ladder_v5_observed.py`)
Fixed scope, plan-only today. Needs: execution authority, actual fits,
prediction shards, station metrics, paired forcing effects, and a new create-only
result authority.
**DoD:** twelve shards + station metrics + paired effects bound by a manifest;
independent re-verification; frozen.
**Note:** the old runner used the last 2,000 rows of the training table as the
LightGBM validation set. The corrected run must use the disjoint 2016–2017
partition. Verify this in the shard metadata, not only in the code.

### 3.2 Independent review and freeze of the 12-cell result
Do not skip. This is the first corrected learned result in the project; if its
lineage is wrong again, everything after it is wasted.
**DoD:** a second, independent read-only reconstruction of at least one cell's
training rows and metrics from raw inputs.

### 3.3 Corrected 432-cell historical ladder
20 random folds (5 seeds × 4) + 4 region folds × L0/L1/L2 ×
LightGBM/ResidualLightGBM × 3 leads. Not yet reimplemented.
**DoD:** 432 shards under a new versioned path; old bytes preserved untouched;
new authority; the L2 mask-invariance proof from the protocol
(`L2_mask_proof`) actually executed and recorded.

### 3.4 Extend random geometry from 5 to 10 seeds
The protocol declares seeds 0–9; seeds 5–9 are a pre-outcome extension and are
the only part of the geometry analysis that can be described as such.
**DoD:** 10 seeds × 4 folds with matched station and key counts, and the
whole-region result located within that distribution.

### 3.5 The 24 missing LightGBM spatial cells of the 48-cell F0/F3 matrix
2 F × 2 L × 2 G × 2 A × 3 leads = 48 logical cells. The current 5,332-fit plan
covers plain TCN and selected ThermoRoute but not the LightGBM spatial cells.
Until these run, no "48-cell matrix complete" statement is admissible.
**DoD:** all 48 logical cells `EXECUTED`, with the two-sided key assertion (no
required key missing, no outside key admitted) passing per cell.

### 3.6 F2b decision, and the 72-cell matrix
F2b is `PLANNED_DEGRADED`. NOAA/NCEI's visible THREDDS catalogs expose only
+0/+3/+6 h for old months; a full trajectory to +192 h may need HAS/tape
retrieval. Set a hard go/no-go date. If it fails, the primary matrix is 48
cells and the paper must say so.
**DoD:** either a frozen exact-initialization vintage archive passing the
vintage/coherence/coverage gates, or a decision-log entry closing F2b as
degraded.

### 3.7 F2a diagnostic, correctly scoped
Acquisition is verified (4,080 chunks, 0.998 coverage, label-free). It may only
be compared against `F3_temperature_only`, never `F3_full`, and never called
as-issued or operational.
**DoD:** the temperature-only recovery fraction computed on the frozen 329,628
reportable-key registry, with the denominator guard applied (no ratio if the
paired F3 value is non-positive, its interval covers zero, or |value| < 0.05 °C).

### 3.8 L2-U2 and L3
L2-U2 (40 fits) is plan-only; L3 requires
`outputs/final/basin_attributes_v2.parquet`, which does not exist —
`src/thermoroute/regionalize.py` is untracked and unrun.
**DoD (L2-U2):** flow and target-site preprocessing state masked; invariance
proof executed. **DoD (L3):** predeclared attribute set with documented source
coverage; the <70% degradation rule applied.

### 3.9 F3 placebos
Within-month shuffled and ±7-day time-shifted F3, on the identical key registry.
No implementation exists. Without these, an oracle result has no negative
control and a reviewer will assume the oracle gain is partly leakage through
calendar structure.
**DoD:** shuffle strata and seeds frozen before execution; shift semantics
documented; both run on the same registry as the real F3.

### 3.10 Forcing-component attribution
Air temperature / radiation / precipitation / humidity+wind / full, with
discharge kept separate. Not implemented.

### 3.11 F1 climatological-forcing negative control
The prior run was discarded. Requires a separate training-only climatology per
meteorological variable.

### 3.12 ThermoRoute hard-regime cells, bounded and unbounded
Only the F0–L0 architecture audit is frozen. The protocol requires an unbounded
variant in every hard-regime cell so that a residual bound is not mistaken for a
property of deep architecture.

### 3.13 Event and tail metrics
Warm-tail, signed rapid warming/cooling, exceedance: RMSE/MAE/bias plus
POD/FAR/CSI/Brier/onset on common keys with training-only thresholds. Not
implemented for the forcing arms. **This is the item most likely to produce the
paper's practical contribution** — managers care about exceedance, not median
RMSE — and it is currently last in the queue. Consider moving it forward.

### 3.14 Validated physical benchmark
Run the official air2stream code, or freeze a transparent equilibrium-response
model. See Part I Major 8.

### 3.15 Expanded cohort (see 0.3)
The WTEMP+meteorology cohort without the joint-flow filter.

### 3.16 Nested flow-complete cohort
Once 3.15 exists, the current 120 stations become a nested subset, which lets
you test whether the flow channel matters after cohort selection is removed —
a much better version of the current flow-ablation result.

### 3.17 2024–2025 audit window
Deferred by DLOG-007. This is the only route to a genuinely untouched
confirmation, and it is worth a great deal to a reviewer given the disclosed
post-outcome status of 2021–2023. Freeze cohort, matrix, models, thresholds,
estimands and figures *before* opening outcomes, then run once. If you do not
do this, say plainly in the paper that no untouched window exists.

## Tier 4 — Inference and estimands

None of this can start before Tier 3 produces results, but the code can be
written and tested against synthetic inputs now.

### 4.1 Station-first paired effects for every declared contrast
`median_i(R_candidate − R_reference)` on exactly-matched keys.

### 4.2 Two-stage aggregation
Random geometry: pair within seed, then average the ten seed-specific station
effects; preserve all ten as sensitivity evidence. Never pool daily rows, never
concatenate seeds. Fit-seed aggregation handled separately from split-seed
aggregation.

### 4.3 Whole-HUC2 cluster bootstrap (10,000 draws), exact sign-flip enumeration,
and leave-one-HUC2-out range — for every promoted effect, not only the primary
family.

### 4.4 Multiple testing
The N01–N15 Holm family declared in protocol v2 was never computed. Declare the
family before reading outcomes and compute it.

### 4.5 Interaction estimands
L×G, F×A, L×A, F×L as station-level double differences with their own
clustered sensitivities.

### 4.6 Recovery-fraction denominator guards
Already specified in the protocol; must be implemented, not just declared.

### 4.7 *(added)* Intervals on the decomposition
The memory/learned decomposition currently has none anywhere. See Part I
Major 6.

### 4.8 *(added)* Anchor sensitivity analysis
Climatology form, φ window, pooled vs per-station φ, φ^h vs direct φ_h, clip.
See Part I Major 6. This is small work with a large effect on reviewer
confidence.

### 4.9 *(added)* Minimum detectable effect
Given 15 HUC2 clusters, state what effect size the design can resolve. This
replaces "not distinguishable from zero" with something defensible.

## Tier 5 — Manuscript

### 5.1 Rewrite
For Paper B, the frame is the information regime, not predictive performance.
Keep the four tasks distinct throughout: oracle, retrospective forecast, gauged
transfer, cold-start. Follow the writing rules you set: confident connected
prose, all caveats consolidated in Limitations, no per-sentence hedging, no
SHA-256 as a selling point, F3 always "retrospective realized gridded
meteorological oracle", and no restoration of any withdrawn headline number.

### 5.2 Claim ledger closure
Every promoted number needs a ledger entry with source table, filters, paired
estimator, print precision, and every `used_in` span. Currently the ledger has
only the v1 conventional claims — no F/L, no interaction, no recovery. The
resolved ledger and TeX macros must be regenerated and the gate must fail on any
undeclared information-regime headline.

### 5.3 Unified result manifest
`outputs/final/result_manifest.json` binds protocol v1 only, omits the forcing
artifacts, and records a stale digest for `ladder_effects.parquet`. Generate one
clean-state manifest binding the governing protocol and amendments, source
commit, claim ledger, summaries, key-level and station-level artifacts, and
exact hashes.

### 5.4 Regenerate tables, figures, SI, PDF
Including everything in Part I Major 10: abstract to 250 words, all placeholders
resolved, figure numbering fixed (remove the duplicate `\caption{}`/body-caption
pattern; renumber to a contiguous 1–4), tables renumbered to AGU style,
equations (5)/(7)/(9)/(10) defined or the references corrected, Section 6
subsections created or the cross-references fixed, `Section 4.7` → `Section 4.5`,
duplicate Limitations heading removed, markdown figure paths repaired, the
status note deleted.

### 5.5 SI completion
Fill or explicitly retire every `[pending computation]` cell (SI02, SI03, SI04,
SI05, SI09, SI10, SI14, SI15, README). Compute or formally deregister SI08's
nine missing probability metrics. Recompute SI12's qualifier-restricted
sensitivity on the held-out window. Render Figures S4–S10 into
`paper/si/figures/` or drop the references. Replace SI11's hardcoded
`claim_eligible: False`.

### 5.6 Full-repository verification
Full test suite green in both environments, Ruff and format clean, consistency
gate clean on both `.md` and `.tex`, `audit_used_in.py` with no FIX rows, all
injection tests failing as designed.

## Tier 6 — Submission compliance (independent of the science; start now)

These are pure calendar time and are currently unstarted. They will block you at
the end if you leave them there.

1. **Author list, affiliations, ORCIDs, corresponding author** — collect via the
   intake schema in `docs/FAIR_SUBMISSION_READINESS_AND_TEMPLATES.md` §2.
2. **Funding, computational resources, competing interests, CRediT roles** — one
   declaration per author.
3. **Byte-level rights review** — blocks the data DOI and licence. The derived
   panel mixes USGS, Daymet V4 R1 (ORNL DAAC) and gridMET; a derived product
   does not inherit the most permissive upstream terms. Until every proposed
   object carries an evidence-backed redistribution decision, no deposit exists.
4. **Mint the data DOI and assign the licence**; deposit the panel, registry,
   HUC metadata, rejection ledger and manifest, plus the test-window acquisition
   record as a new version.
5. **Mint the software DOI**, tag a release, confirm the public repository URL.
6. **Independent reproduction** by an operator who is not an author, on an
   independent host, from the archived bundles and the pinned Python 3.12
   hashed lock. The manuscript currently states this has not happened; a
   reviewer will read that as an admission.
7. Note the development-panel gap honestly: original provider responses for
   2006–2020 were not retained, so development reproduction starts from the
   committed derived artifact. Keep this in Limitations.

## Suggested order

```
0.1 → 0.3 → 0.4          decisions and reading, in parallel with everything below
  │
  ├── Paper A track:  5.4 production fixes → 5.5 SI → Tier 6 (1,2)
  │                   → Major 2/5/6/7 text and analysis fixes
  │                   → 4.7, 4.8, 4.9 (small, high value)
  │                   → submit
  │
  └── Paper B track:  1.1–1.8 contracts → 1.8 audit
                      → 2.1–2.4 governance and clean commit
                      → 3.1 twelve-cell → 3.2 independent freeze
                      → 3.3 432-cell → 3.4 ten seeds → 3.5 48 cells
                      → 3.6 F2b go/no-go → 3.9 placebos → 3.13 event metrics
                      → Tier 4 inference → 5.1–5.3, 5.6
                      → 3.17 untouched 2024–2025 audit (freeze before opening)
```

Tier 6 items 3–6 run on their own clock and should be started immediately
regardless of which track you are on.

## Two honest observations

**The engineering is ahead of the science.** The lineage-defect detection, the
create-only authorities, the injection-tested consistency gate, and the
independent attack audits are genuinely better than the field's norm. But a
reviewer sees none of that. What a reviewer sees is: a well-known methodological
point, on 116 availability-selected gauges, with no future-forcing arm, no
process benchmark, no untouched window, and no data deposit. The gap between
those two sentences is the whole problem.

**The most valuable remaining work is cheap.** In rough order of
reviewer-visible value per unit effort: the literature program (0.4), the
anchor sensitivity analysis (4.8), the intervals on the decomposition (4.7),
the production fixes (5.4), the event/exceedance metrics (3.13), and the
expanded cohort (0.3/3.15). The 5,332-fit neural inventory is far down that
list.

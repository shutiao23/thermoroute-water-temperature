# Main-body restructure of `paper/ThermoRoute_paper.md` — 2026-08-05

| Field | Value |
|---|---|
| Target file | `paper/ThermoRoute_paper.md` |
| Branch | `feat/route-a-completion` |
| Prior bytes | SHA-256 `4843656b5858c8d0f997d0bc5dcc55aa79e77fc8c974b1b7a1ee17ce519104e7` (recoverable at commit `b0699a8`) |
| New bytes | rewritten in this revision; see §5 for the binding consequence |
| Files touched | `paper/ThermoRoute_paper.md`, `docs/PAPER_RESTRUCTURE_20260805.md` (this file) |
| Files deliberately NOT touched | `src/`, `scripts/`, `tests/`, `protocols/`, `pyproject.toml`, `requirements*.txt`, `.github/workflows/`, `ops/`, `outputs/`, the sibling `…-multicore` worktree (read-only) |
| Target venue | Water Resources Research (AGU) |
| Main-body length | ≈ 10,430 words (§1–§7); ≈ 11,700 including front and back matter |

---

## 1. Why the rewrite

The superseded manuscript was organised as a governance document with a research
paper embedded in it. Its section order was: problem → data → model → evaluation
design → **reproducibility and one-time opening** → **current progress** → permanent
limitations → results placeholder → availability → conclusion. Two of the nine
sections (≈ 2,100 words) described receipt chains, directory-rename publication
semantics, and stage-by-stage engineering status. The three genuinely
distinguishing scientific properties of the study — the strong baseline suite, the
mechanical leakage control, and the single-acquisition holdout — were distributed
across four sections and never argued as strengths. The opening status note read
as an apology ("no current empirical performance conclusion", "all numerical
statements … have been withdrawn"), and the inference-gate failure was presented
as an unfortunate constraint rather than as the study's most defensible
methodological decision.

The rewrite keeps every scope restriction and every permanent limitation, and
re-argues the same facts in the order a WRR reader expects.

---

## 2. Section-by-section changes

### New structure

| § | Section | Status |
|---|---|---|
| — | Title, Key Points, Manuscript status, Abstract, Plain Language Summary, Keywords | rewritten |
| 1 | Introduction | rewritten; absorbs old §1 + §1.1 |
| 2 | Data and cohort (2.1 panel/registry, 2.2 why the cohort is what it is, 2.3 temporal roles, 2.4 evaluation-period inputs and information boundary, 2.5 legacy three-station material) | rewritten from old §2 |
| 3 | Methods (3.1 predictor, **3.2 strong baseline suite**, **3.3 leakage control**, **3.4 regional transfer**, **3.5 conformal intervals**, 3.6 estimand/family/inference gate, **3.7 one-time 2021–2023 evaluation**, 3.8 pre-registration and reproducibility) | reordered and re-argued from old §3, §4, §5 |
| 4 | Results (4.1–4.5 development/exploratory, 4.6 evaluation-period slots) | new |
| 5 | Discussion (5.1 gate failure as contribution, 5.2 what development evidence indicates, 5.3 relation to prior work) | new |
| 6 | Limitations (6.1 machine-verifiable claim blocks, 6.2 cohort/measurement/design, 6.3 probabilistic suite not reported, 6.4 observed-threshold scope, 6.5 archive scope) | rebuilt from old §7 |
| 7 | Conclusions | rewritten from old §10 |
| 8 | Open Research | rewritten from old §9, AGU-compliant |
| — | Acknowledgments, Supporting Information | new |

### Detail

**Title.** Changed from "ThermoRoute: a bounded-deviation, physics-inspired
framework for retrospective daily river-temperature hindcasting" to "Strong
baselines, controlled leakage, and a pre-specified inference gate for daily river
water-temperature hindcasting". The old title advertised the architecture; the
new one advertises the evaluation, which is what the paper actually contributes.

**Key Points.** Added (three items, AGU requirement). Did not exist before.

**Manuscript status.** The old block-quoted status note (7 lines, apologetic) is
replaced by a five-line "Manuscript status" paragraph stating the fact
(evaluation not executed), what Section 4 contains instead, and that placeholders
exist. No withdrawal language, no self-criticism.

**Abstract.** Rewritten. Retains: cohort facts, the gate facts and verdict string,
the "causal = time ordering only" clarification, the no-topology clarification.
Adds: an explicit three-part statement of the design contribution (baseline suite,
leakage mechanism, single acquisition). Removes: the sentence "The contribution at
this stage is a testable architecture and a fail-closed evaluation system whose
remaining scientific and engineering limitations are stated explicitly", which
framed the paper as an incomplete deliverable.

**Plain Language Summary.** Added (AGU requirement). Did not exist before.

**§1 Introduction.** Restructured around four named difficulties (persistence
strength, issue-time leakage, structured incompleteness, spatial dependence), each
with a concrete number from the panel and each mapped to a later design decision.
Related work is folded in rather than given a subsection, following the requirement to
keep the introduction hydrologically motivated. Added citations to
`corona2025ml`, `sadler2022multitask` from `paper/references.bib`. The legacy
three-station paragraph is retained with both allowlisted sentences preserved
verbatim, but moved to the end of §1 and cross-referenced to §2.5.

**§2 Data and cohort.** Old §2.1/2.2/2.3 content preserved and reorganised. New
§2.2 ("Why the cohort is what it is") makes the availability-enrichment argument
explicitly rather than as a trailing sentence. New §2.3 adds the 2021–2023 row to
the temporal-role table and the 249,072 / 83,024-per-lead common-key counts. New
§2.5 gives the legacy three-station material a named, bounded home.

**§3 Methods — the substantive reordering.** Old §3 (model design) and §4
(evaluation design) are merged and re-emphasised:

- **§3.2 Strong baseline suite** is now the first methods subsection after the
  predictor description, and it *argues* the suite rather than listing it: each
  reference is introduced by the alternative explanation it removes. New material:
  the explicit statement that the LSTM receives the same splits, loss, sampling,
  epoch/patience limits, history length, site identity, and seed budget, and is
  eligible for the same calibration; the explicit statement that tuning budgets
  were not equalized and that any LightGBM statement is scoped to the frozen
  four-candidate procedure. The air2stream-style reference is reported as
  `NOT_RUN` and unofficial rather than dropped.
- **§3.3 Leakage control** is new as a standalone subsection with six named
  mechanisms (issue-time covariate dating; non-anticipating encoding with bounded
  lag range; strictly backward-fitted statistics; explicit admissibility including
  the declared auxiliary-mask gap; transform conventions; isolated replay under
  `python -I -B`). Previously this material was scattered across old §3.2, §4.1,
  and §5.
- **§3.4 Regional transfer design** is new as a standalone subsection. It states
  the leakage argument against random spatial splits, gives the fold geometry
  ([30, 30, 31, 29]) and the 289 km nearest-training-gauge distance, and carries
  an unambiguous paragraph that this is *gauged* transfer, not ungauged
  prediction. Cites `weierbach2022stream`.
- **§3.5 Conformal intervals** is new as a standalone subsection. Retains the
  `qhat_plus = max(raw_qhat, 0)` construction and the fail-closed conditions.
  Adds an explicit "what this guarantees" paragraph: marginal only, exchangeability
  violated by construction (2018 calibration vs. later evaluation, dependence
  within station and region), no finite-sample guarantee, no conditional coverage
  claim of any kind.
- **§3.6** carries the estimand, the five-row table, both clustered procedures,
  and the inference gate. The gate is now presented as a decision *taken by the
  authors in advance*, with its three thresholds, the three cohort values, and the
  arithmetic of the failure spelled out.
- **§3.7 One-time 2021–2023 evaluation** is expanded and given real space, per
  instruction. It states the ordering gate (model-suite commit precedes candidate
  metadata and predictor artifacts), the narrow raw-only acquisition, the
  parameter/statistic identities, the multiple-finite-series conflict rule, the
  no-site-replacement rule, the qualifier-inclusive primary policy plus the
  `{A}` sensitivity, and the honest statement that this is one logical opening
  and not exactly-once HTTP delivery, and an honest-owner guard rather than a
  security property.
- **§3.8 Pre-registration and reproducibility** is compressed to **exactly two
  paragraphs**, per instruction. No receipt names, no stage numbers, no run IDs.
  Points to SI and to the archive DOIs.

**§4 Results.** New section. §4.1–§4.5 report development-period numbers with a
standing exploratory label. §4.6 contains the six evaluation-period slots (§4 of
this document). The old "§8 Pre-opening results placeholder and stopping rule",
which described the renderer command line, is removed from the main body; the
renderer is referenced once in §8 Open Research.

**§5 Discussion.** New section. §5.1 makes the gate-failure-as-strength argument
with confidence: the condition was determinable from the registry alone, the
usual practice in this literature is to report cluster-naive intervals, the
difference between that paper and this one is disclosure, and the transferable
design lesson is that cluster structure must be part of the sampling design.
§5.1 also links forward to §6.3 as a second instance of a pre-registration
binding against the authors' convenience. §5.2 reads the development evidence as
description and states plainly that the tree ensemble attains the lowest
station-median RMSE at all three leads. §5.3 positions the work against
process/network/differentiable models without claiming to replace them.

**§6 Limitations.** The eight structured `ROUTE_A_CLAIM` blocks are preserved
**byte-for-byte** (see §5 of this document). §6.2 consolidates the old trailing
limitation paragraph. §6.3 is new (see §3 below). §6.4 and §6.5 carry the
observed-threshold and archive-scope paragraphs forward.

**§7 Conclusions.** Rewritten from the old §10. Removes the "specified research
system rather than a completed empirical finding" framing and the sentence about
old headline numbers. States the design contribution, the development numbers as
description, and the permanent descriptive scope.

**§8 Open Research.** Rewritten to AGU's Data Availability Statement
requirements: data deposit with DOI placeholder and licence placeholder, primary
sources with their own DOIs, software archive with DOI, release tag, repository
URL, and SPDX licence placeholder, plus a reproduction paragraph with tolerance
routing and a placeholder-inventory paragraph. The old text's admission that the
2006–2020 provider bytes are unavailable is retained.

---

## 3. Correction applied during the revision: the Stage-19 blocker

The initial plan described the blocker as "64 crossing rows out of
1,245,360 member-level rows (0.0051%) where exported q05/q50/q95 are
non-monotone", which matches the comment at
`…-multicore/ops/stage09/chain_stage09_remaining.sh:5-9`. That characterisation
was superseded during the revision by a direct measurement of the prediction table. The
manuscript uses the measured version, and the phrase "quantile crossing" does not
appear anywhere in the new §6.3.

| Measured fact | Value |
|---|---|
| Member-level rows scanned (rows carrying complete quantile heads) | 26,993,675 |
| Strict ordering violations (`q05 > q50`, `q50 > q95`, or `q05 > q95`) | **0** |
| Maximum monotonicity violation | exactly 0.000 °C |
| Rows tripping the frozen contract | **135** (all zero-width: `q05 == q50 == q95`, bit-identical float64) |
| Rate | 135 / 26,993,675 = 0.0005% (≈ 1 in 200,000) |
| Attribution | global LightGBM 123 rows; per-station LightGBM 12 rows; no other model |
| Mechanism | the frozen contract tests `q05 >= q95` inclusive and treats it as fatal; the protocol forbids evaluation-time repair, so a correctly ordered but degenerate interval is rejected |
| Example rows | site `04027000`, h=1, all three quantiles = 0.0010064127062574 °C and = −0.019121303577536868 °C; site `01542500` (per-station), all three quantiles = 13.110679188150272 °C |
| Distinct sites | 12, all in HUC 04 (Great Lakes), 05 (Upper Mississippi), 06 (Missouri) |
| By lead | h=1: 129 rows; h=3: 6 rows; h=7: 0 rows |
| Rows with observed `y_true` < 1.0 °C | 95.6% |
| Downstream effect | none on the delivered product: split-CQR adds a non-negative offset and the deployed contract separately requires positive width, so no delivered interval is degenerate |

**Provenance.** These values are recorded in
`docs/STAGE19_DEGENERATE_INTERVAL_DISPOSITION_20260805.md` (written concurrently
in this working tree), derived by direct measurement of
`/home/lzq/workspace/parttime/thermoroute-water-temperature-multicore/outputs/predictions/usgs_predictions_with_perstation_v2.parquet`.
That disposition document also records the decision not to amend the contract and
the retraining cost that would follow, and specifies that its §2.1 table be
reproduced verbatim in the SI. The measurement is **not yet materialised as a
receipt-bound output artifact**; before submission it should be re-derived by a
committed script and the artifact path and digest recorded here. The freeze-time
evidence that the stage did not run is at
`…-multicore/outputs/logs/multicore_remaining.log:917` and
`…-multicore/outputs/logs/monitor_chains.log:90,96`.

The interpretive attribution in §6.3 (ice-affected winter sites for the global
model; constant-leaf per-station fits) is written as interpretation and is
labelled as such in the manuscript text.

---

## 4. Unfilled result slots and the artifact that will fill each

All slots use the literal marker `[TO BE FILLED AFTER OPENING]`. Fifteen markers
occur across six slots (Table 4.1 carries two markers per row).

| Slot | Location | Filled by |
|---|---|---|
| **Table 4.1**, five formal rows (effect, cluster-bootstrap 95% interval, win rate, station/cluster counts, exact sign-flip raw p, Holm p) | §4.6 | `formal_tests[*]` of the verified opening receipt, rendered deterministically by `scripts/26_validate_claims.py --write-generated-results --require-complete` from the exact statistics artifact. Verdict column is pre-filled with `DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED`, which is outcome-independent. |
| **Table 4.2**, all-model scores on evaluation common keys | §4.6 | receipt-bound evaluation-period score table (evaluation-period analogue of `outputs/tables/usgs_scores.csv`), routed through SI07. |
| **Table 4.3**, temporal coverage / calendar balance (8 predeclared candidates + most-adverse) | §4.6 | the predeclared temporal-coverage audit artifact after physical replay and receipt binding, per `protocols/route_a_temporal_coverage_policy_v1.json`; routed through SI10. |
| **Table 4.4**, interval behaviour on evaluation keys | §4.6 | evaluation-period analogue of `outputs/reports/adaptive_conformal.md` + `outputs/tables/aci_coverage.csv`; routed through SI08. |
| **Table 4.5**, outcome quality control | §4.6 | the acquisition quality-audit tables produced under `protocols/route_a_outcome_qc_policy_v1.json` (qualifier counts, multiple-finite-series conflicts, `{A}` sensitivity); routed through SI12. |
| **Table 4.6**, held-region and external arms on evaluation keys | §4.6 | evaluation-period analogues of `outputs/reports/region_transfer.md` and the pooled external arm; routed through SI11 and SI13. |
| **Probabilistic metric suite** | §4.6, §6.3 | **Not a slot.** Explicitly marked "not reported" with the §6.3 justification. It must not be filled without a sealed erratum and a new training lineage. |
| DOIs, repository URL, release tag, licence identifiers, author metadata | §8, Acknowledgments | external; see `docs/FAIR_EXTERNAL_CLOSEOUT_REQUEST_PACK.md`. |

---

## 5. Binding consequence of this rewrite (must be read before any validator run)

`protocols/route_a_claim_registry_v1.json` contains
`preopen_document_sha256["paper/ThermoRoute_paper.md"] = 4843656b5858c8d0f997d0bc5dcc55aa79e77fc8c974b1b7a1ee17ce519104e7`,
and `free_text_policy` states
`REQUIRED_PREOPEN_DOCUMENT_BYTES_ARE_SHA256_FROZEN`. Rewriting the file
necessarily invalidates that binding, so `scripts/26_validate_claims.py` will
report a document-hash mismatch for this file until the registry is re-sealed.
Re-sealing is a `protocols/` edit and is **outside the permission boundary of
this revision**; it must be authorized separately and will create a new document
lineage. The previous bytes remain recoverable at commit `b0699a8`.

Mitigations applied so that nothing else in the validator chain breaks:

1. **All eight `ROUTE_A_CLAIM` blocks are preserved byte-for-byte** (5,982 bytes,
   SHA-256 `63698d1e52e58d893ba971df54ca7d1f2c9fc04e4151083b73c876b251946cee`),
   including their per-claim body hashes and JSON entries, and all eight
   `render_targets` still point at this file. They now live under §6.1.
2. **Both allowlisted legacy-semantics sentences are preserved verbatim**
   (`ALLOW_LEGACY_THREE_SITE_ORDINARY_MONITORING_CORRECTION` and
   `ALLOW_LEGACY_THREE_SITE_EPISTEMIC_TOPOLOGY_LIMIT`), and each appears twice
   (§1 and §2.5).
3. **All 21 `free_text_lints` regexes were run against the new text with
   whitespace normalisation.** Zero unallowed hits: every match falls inside a
   structured claim block or matches the allowlist.
4. **B-02 forbidden-verb scan** (`confirm`, `demonstrate`, `establish`,
   `support`, `achieve`, `outperform`, `prove`, `validate`, `show superiority`,
   `conclude equivalence`/`parity`): every remaining occurrence is either
   prohibitive ("no row may be written as superiority, non-inferiority,
   equivalence, or parity"), inside a preserved claim block, or a neutral
   non-claim use (`provenance`, `validated in a staging area`, `invalidates
   authorization`). No occurrence upgrades a five-row verdict.
5. `postopen_document_transform` mode is
   `EXACT_PREOPEN_BYTES_PLUS_DETERMINISTIC_RESULT_SUFFIX` with heading
   `# Route-A receipt-derived results`. The new §4.6 slot table is prose, not the
   generated suffix, and does not occupy that heading. The generated suffix
   remains appendable.

---

## 6. Every number used, and its source file

All paths under `MC/` are read-only reads from
`/home/lzq/workspace/parttime/thermoroute-water-temperature-multicore/`.
All paths under `PRE/` are from the superseded bytes of
`paper/ThermoRoute_paper.md` at commit `b0699a8` (carried forward unchanged, not
re-derived).

### Cohort and panel

| Value | Used in | Source |
|---|---|---|
| 657,480 site-days; 120 stable site numbers; 2006-01-01–2020-12-31 | Abstract, §2.1 | `README.md`; `protocols/route_a_confirmatory_protocol.md` §1; `PRE/§2.1` |
| 34 states; 15 HUC2 groups | Abstract, §2.1 | `README.md`; `PRE/§2.1` |
| Missingness 15.8% `WTEMP`, 2.8% `FLOW`, ~0.07% `TEMP`/`PRCP`/`RHMEAN`/`DH`, 0% `WDSP`; `WLEVEL` 77.65% | §1, §2.1 | `PRE/§2.1` |
| Leap-year provider-calendar final-day gap (2008, 2012, 2016, 2020) | §2.1 | `PRE/§2.1` |
| 1,345 rejected candidates = 950 + 376 + 19; thresholds 0.55/0.70 and 0.80/0.80; 1,465-candidate discovery | §2.2 | `PRE/§2.1` |
| 38 stations share a repeated HUC code; 19 have a neighbour within 10 km | §1, §2.2, §5.2 | `PRE/§2.1` |
| Max missing runs 2,404 d (`WTEMP`), 1,369 d (`FLOW`) | §1, §2.2 | `PRE/§2.1` |
| 2,059 negative-`FLOW` rows at 2 stations; minimum −121 cfs | §2.2, §6.2 | `PRE/§2.1` |
| Split rows 438,240 / 87,720 / 43,800 / 87,720 and observed `WTEMP` 341,646 / 84,074 / 42,279 / 85,621 | §2.3 table | `PRE/§2.1` table |
| 249,072 common evaluation keys | §2.3, §4, §4.4 | `MC/outputs/reports/usgs_robustness_v1.md` L3; `MC/outputs/reports/adaptive_conformal.md` |
| 83,024 keys per lead | §2.3 | `MC/outputs/reports/adaptive_conformal.md` (lead-1/3/7 rows) |
| Evaluation interval 2021-01-01 – 2023-12-31 | §2.4, §3.7 | `protocols/route_a_confirmatory_protocol.md` §2 |
| `PASS_EXACT_PRODUCT_BRIDGE` | §2.4 | `PRE/§2.2`; `README.md` |
| 30-site external cohort, deterministic seed, site-ID disjoint | §2.4 | `protocols/route_a_confirmatory_protocol.md` §2–§3; `PRE/§2.3` |

### Development-period point accuracy (exploratory)

| Value | Used in | Source |
|---|---|---|
| Persistence 0.803 / 1.576 / 2.217 | §1, §4.1 | `MC/outputs/reports/usgs_experiment.md` |
| Damped persistence 0.774 / 1.406 / 1.738 | §1, §4.1, §4.5 | same |
| LightGBM 0.578 / 1.280 / 1.649 | §4.1 | same |
| ThermoRoute 0.631 / 1.291 / 1.657 | §4.1, §4.5 | same; cross-checked in `MC/outputs/reports/lstm_baseline.md` |
| LSTM 0.662 / 1.323 / 1.679 | §4.1 | `MC/outputs/reports/lstm_baseline.md` |
| air2stream-style = `NOT_RUN` | §3.2, §4.1 | `MC/outputs/reports/usgs_experiment.md` L5 |
| Skill vs. persistence +0.203 / +0.187 / +0.251 | Abstract-adjacent, §4.1, §4.2, §4.3, §7 | `MC/outputs/reports/usgs_experiment.md`; `MC/outputs/reports/tuurt.md` |
| Skill vs. damped +0.168 / +0.076 / +0.038 | §4.1, §4.3, §7 | same |
| Paired median TR − damped −0.130 / −0.104 / −0.062; win 0.86 / 0.90 / 0.93 | §4.1 | `MC/outputs/tables/development_route_a_estimand_mirror.md` |
| Paired median TR − LightGBM +0.046 / +0.008 / −0.005; win 0.00 / 0.40 / 0.60 | §4.1, §5.2 | same |
| Cluster-bootstrap intervals [−0.177,−0.081], [−0.126,−0.083], [−0.071,−0.052]; [+0.043,+0.057], [+0.001,+0.020], [−0.010,+0.010] | §4.1 | same |
| `NO_STRONG_INFERENCE`, `SMALL_CLUSTER_COUNT_LT_30`, `LOW_EFFECTIVE_CLUSTER_FRACTION` | §4.1 | same, small-cluster sensitivity table |

### Regional structure and transfer (exploratory)

| Value | Used in | Source |
|---|---|---|
| Region-weighted skill +0.204 / +0.189 / +0.253 (mean of 15 per-HUC2 medians) | §4.2 | `MC/outputs/reports/stratified.md` |
| 1-day per-region range +0.131 (HUC2:10) to +0.285 (HUC2:18) | §4.2 | same |
| 7-day per-region range +0.217 (HUC2:06) to +0.290 (HUC2:09) | §4.2 | same |
| 7-day vs.-damped per-region range +0.012 to +0.077 | §4.2 | same |
| Drainage strata n = 39 / 38 / 39; 1-day +0.190 / +0.214 / +0.231 | §4.2 | same |
| Random held-site 4-fold skill vs. persistence +0.178 / +0.172 / +0.241 | §4.3 | `MC/outputs/reports/tuurt.md`; per-fold values in `MC/outputs/tables/claim2_kfold_lgo.csv` |
| Random held-site 4-fold skill vs. damped +0.145 / +0.061 / +0.030 | §4.3 | same |
| Held-region skill vs. persistence +0.155 / +0.116 / +0.155 | §4.3, §5.2 | `MC/outputs/reports/tuurt.md`; `MC/outputs/reports/region_transfer.md` |
| Held-region skill vs. damped +0.147 / +0.086 / +0.075 | §4.3 | `MC/outputs/reports/tuurt.md` |
| Held-region RMSE TR 0.676 / 1.428 / 1.860; LGB 0.652 / 1.391 / 1.786 | §4.3 | `MC/outputs/reports/region_transfer.md`; `MC/outputs/tables/region_transfer.csv` |
| Held-region median TR − LGB +0.031 [+0.023,+0.040], +0.038 [+0.029,+0.050], +0.067 [+0.049,+0.080]; win 0.158/0.175/0.158 (reported as "near 0.16") | §4.3 | same |
| Held-region LSTM 0.679 / 1.445 / 1.876 | §4.3 | `MC/outputs/reports/lstm_baseline.md` |
| Fold geometry [30, 30, 31, 29]; mean nearest-training-gauge distance 289 km | §3.4, §5.2 | `MC/outputs/reports/region_transfer.md` |

### Intervals (exploratory)

| Value | Used in | Source |
|---|---|---|
| split-CQR overall coverage 0.909, width 3.87, interval score 4.93, n = 249,072 | §4.4 | `MC/outputs/reports/adaptive_conformal.md` |
| Per-lead coverage 0.905 / 0.910 / 0.912; widths 2.01 / 4.22 / 5.38 | Abstract-adjacent, §4.4 | same |
| Warm train-q90 tail: n = 30,531, coverage 0.918, width 3.50 | §4.4 | same |
| Block-max sensitivity: coverage 0.981, width 5.74 | §4.4 | same |
| Delayed ACI coverage 0.903 / 0.900 / 0.892 for γ = 0.005 / 0.02 / 0.05; several slices report infinite width and interval score | §4.4 | same |

### Component and input sensitivities (exploratory)

| Value | Used in | Source |
|---|---|---|
| Ablations: TR-noTCN 0.679 / 1.338 / 1.675; TR-noRouter 0.634; TR-noMoE 0.634; TR-noDynamicPrior 0.628; TR-fixedKappa 0.627; TR-unbounded 0.630; DampedPriorOnly 0.774 / 1.406 / 1.738 | §4.5 | `MC/outputs/reports/usgs_experiment.md` |
| Sensor noise 0.25 sd → 1.621 (+146.9%); 0.5 sd → 2.999 (+360.0%) at h=1 | §4.5 | `MC/outputs/reports/usgs_robustness_v1.md` |
| Missing blocks 3/7/14 d → ΔRMSE ≈ +0.132/+0.125/+0.125 at h=1 and +0.039/+0.045/+0.048 at h=7 | §4.5 | same |
| Air-temperature shift ±2 sd → +0.106 to +0.150 at h=1 | §4.5 | same |
| Flow multipliers 0.5 / 2 → ≤ +0.003 at h=1 | §4.5 | same |
| Bounded-deviation audit: 100.00% pointwise anchor contract; 100.00% derived error inequality; max |correction| 1.0000 °C; 360 station-by-lead cells at 100.00% | §4.5, §5.2 | `MC/outputs/reports/prop1_binding.md` |
| `delta_scale` = 1.0 °C | §3.1, §4.5 | `PRE/§4.1`; `MC/outputs/reports/prop1_binding.md` |

### Inference gate and comparison family

| Value | Used in | Source |
|---|---|---|
| Gate thresholds: `n_clusters ≥ 30`, `effective_cluster_fraction ≥ 0.75`, `largest_cluster_share < 0.25` | Abstract, §3.6, §5.1 | `docs/B02_PERMANENT_DESCRIPTIVE_CLAIM_WORDING.md` |
| ≤ 15 HUC2; effective cluster count 9.54; effective fraction ≈ 0.636; largest share 0.217 (21.7%); cluster sizes 2 / 7.0 / 26 | Abstract, §3.6, §5.1, §7 | `docs/B02_PERMANENT_DESCRIPTIVE_CLAIM_WORDING.md`; `MC/outputs/tables/development_route_a_estimand_mirror.md` (small-cluster sensitivity table) |
| Verdict string `DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED` | Abstract, §3.6, §4.6, §5.1 | `docs/B02_PERMANENT_DESCRIPTIVE_CLAIM_WORDING.md`; `protocols/route_a_claim_registry_v1.json` `decision_rule.inference_gate_failure` |
| Five-row family, margins 0.00 / 0.00 / 0.00 / +0.05 / +0.05 °C | §3.6 table, §4.6 | `protocols/route_a_confirmatory_protocol.md` §4; `PRE/§4.2` |
| ≥ 100 valid paired targets per reportable station/lead | §3.6 | `protocols/route_a_confirmatory_protocol.md` §2, §4 |
| 10,000 whole-HUC2 bootstrap draws; exact 2^K sign enumeration; Holm over exactly five p-values | §3.6 | `protocols/route_a_confirmatory_protocol.md` §4 |
| `REV_NOT_EVALUATED_NO_PREDECLARED_COST_LOSS_RATIOS` | §3.6 | `MC/outputs/reports/rev_curve.md`; `MC/outputs/tables/rev_curve.csv` |
| Eight temporal-coverage candidates (12 year-by-season cells; leave-one-year ×3; leave-one-season ×4) | §3.6, §4.6 | `PRE/§4.4`; `protocols/route_a_temporal_coverage_policy_v1.json` |

### Model selection budgets

| Value | Used in | Source |
|---|---|---|
| LightGBM: 4 predeclared candidates per lead, selected on 2016–2017 station-macro RMSE | §3.2 | `MC/outputs/tables/lightgbm_joint_validation_selection.csv` |
| LSTM: 3 predeclared candidates, seed 0, then 5 members | §3.2 | `MC/outputs/tables/lstm_validation_selection.csv` |
| Seeds 0–4 for LightGBM, LSTM, ThermoRoute, and all seven controls | §3.2 | `MC/outputs/reports/usgs_experiment.md`; `PRE/§4.1` |
| Encoder: two blocks, kernel 3, 7-step receptive field; router lags 0–14; 32-day buffer | §3.1, §3.3 | `PRE/§3.2`; `README.md` |

### Stage-19 disposition

See §3 of this document. All values there are recorded in
`docs/STAGE19_DEGENERATE_INTERVAL_DISPOSITION_20260805.md`, derived from
`MC/outputs/predictions/usgs_predictions_with_perstation_v2.parquet`, with the
non-execution evidence at `MC/outputs/logs/multicore_remaining.log:917` and
`MC/outputs/logs/monitor_chains.log:90,96`. The superseded
"64 / 1,245,360 crossing" figure at
`MC/ops/stage09/chain_stage09_remaining.sh:5-9` is **not** used, and the phrase
"quantile crossing" does not appear in the manuscript.

---

## 7. What was deliberately deleted

| Deleted | Why | Where it now lives |
|---|---|---|
| Old §5 "Reproducibility and one-time opening" (≈ 1,000 words of receipt-chain, directory-rename, sidecar-repair, and stage-closure prose) | Instruction: pre-registration and reproducibility compressed to two paragraphs, no governance section, no receipt/stage/run-ID enumeration | Scientific content redistributed to §3.7 (acquisition and chronology gate, as a methodological argument) and §3.8 (two paragraphs). Engineering detail belongs in SI15 and the archive. |
| Old §6 "Current progress" in full, including the six-item outstanding-work list and the "all superseded noncanonical numeric outputs are considered stale" sentence | A submitted manuscript does not carry an engineering to-do list; it read as an apology | Replaced by the five-line "Manuscript status" note and by the §4.6 slot table, which names the filling artifact for each slot |
| Old §8 "Pre-opening results placeholder and stopping rule", including the literal `python scripts/26_validate_claims.py …` command line | Command lines belong in the archive and SI, not the main body | Renderer referenced once, without a command line, in §8 Open Research and in §4.6 |
| Old opening block-quoted status note ("All numerical statements from superseded noncanonical result artifacts have been withdrawn pending a complete stable-`site_no` Route-A rerun") | Instruction: keep the status note but make it short and dignified | "Manuscript status" paragraph |
| Enumeration of individual completion receipts and their contents (Stage-9 parent binding, 75-file member/head registry, 45-member matrix, 80-model-file closure, four-gate closure) | Instruction: do not enumerate receipts, stage numbers, or run IDs in the main body | One sentence in §3.8 ("each training stage terminates in a content-bound completion record … the model-suite freeze and the independent release verifier both require the complete set") |
| Detailed transport-continuation semantics (ledger entries without a durably published canonical transaction, unpublished owner-private temporary state, digest-sidecar regeneration, abandoned-stage deletion preconditions) | Operational detail with no bearing on the scientific claim | Condensed to one paragraph in §3.7 that preserves the three statements a reader needs: one logical opening, fail-closed on partial state, honest-owner guard rather than a security property |
| Sentence "The contribution at this stage is a testable architecture and a fail-closed evaluation system whose remaining scientific and engineering limitations are stated explicitly" | Framed the paper as an incomplete deliverable | Replaced by the six-item contribution list at the end of §1 |
| Old §1.1 as a separate related-work subsection | Instruction: keep the introduction tight and hydrologically motivated | Folded into §1 prose and §5.3 |
| Old §4.3 sentence on the Air2stream-style reference being "optional" | Understated; the reference is part of the declared suite and its status must be visible | §3.2 (unofficial, `NOT_RUN`, and why a published-model comparison would require the official implementation) and §4.1 table |
| Old references to "six primary model types … persistence, damped persistence, climatology, LightGBM, a global LSTM, and ThermoRoute" as the whole suite | Understated the suite: the per-station LightGBM variant and the air2stream-style reference were not visible in that list | §3.2 presents the full suite including both |

**Nothing was deleted from the permanent-limitation set.** Every limitation in
the old §7, including the trailing paragraph and the archive-scope paragraphs,
appears in the new §6. The eight structured claim blocks are byte-identical.

---

## 8. Known gaps to close before submission

1. Re-seal `preopen_document_sha256` for `paper/ThermoRoute_paper.md` under an
   authorized `protocols/` change (§5 above).
2. Materialise the §3 Stage-19 measurement as a committed, receipt-bound artifact
   and replace the provenance note with its path and digest.
3. No development-period score is available for the **per-station LightGBM**
   variant: `MC/outputs/predictions/usgs_predictions_with_perstation_v2.parquet`
   contains 249,072 aligned rows, but no scored summary report exists. §3.2
   therefore describes the variant without quoting a number, and §4.1 omits it
   from the score table. Either score it or state its absence in SI07.
4. Figure cross-references are absent. `paper/FIGURE_REDRAW_SPEC.md` §7 fixes the
   first-citation position for Figures 1–4 and S1–S8; those citations must be
   inserted by the PRE/POST renderer, not by hand.
5. `paper/highlights.md`, `paper/cover_letter.md`, and
   `paper/agu_submission/ThermoRoute_WRR.tex` still reflect the superseded
   structure and section numbering and need the same treatment.
6. Author metadata, DOIs, licences, and CRediT roles remain placeholders.

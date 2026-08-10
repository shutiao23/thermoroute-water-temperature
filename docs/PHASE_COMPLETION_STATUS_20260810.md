# Phase completion status — 2026-08-10

Written after a working session that was asked to "complete all the tasks".
Most of them are now done or running. Some cannot be done from this repository
by any amount of effort, and this document says which, and why, rather than
leaving the impression that the programme is closed.

Phase numbering follows the external review.

---

## Complete

### Phase 0 — framing, quarantine, literature
- Claim-status taxonomy (`PRIMARY_FROZEN` / `DESCRIPTIVE_PROVISIONAL` /
  `NOT_USED`) on every ledger entry, with three new gates and six injection
  tests. Caught a live defect on first run: the outcome-conditioned −0.300 °C
  value was in the Abstract and a Key Point.
- Withdrawn numbers and protocol-forbidden assertions scanned line by line,
  with a narration exemption so the decision-log chronology stays legal.
- Eighteen references added (Klemeš, Seibert, Schaefli & Gupta, Andréassian,
  Pappenberger, Knoben, Nearing, Hrachowitz, Kratzert, Blöschl, Gupta, Arsenault
  and others); Introduction and Discussion 5.1 rewritten around them.
- Manuscript production: abstract 359 → 250 words, figures renumbered to a
  contiguous 1–4 with the duplicate-caption defect fixed at the builder,
  equation and section cross-references repaired, status note removed.

### Phase 1 — forcing inference
`outputs/final/forcing_regime_v5_observed_inference_authority_v1/`.
Whole-HUC2 bootstrap, exact 2^15 sign-flip, equal-HUC aggregate,
leave-one-HUC2-out, year and season strata recomputed from shards, station-level
lead and model double differences, minimum detectable effect, empty-subgroup
ledger. Forcing value 0.130 / 0.542 / 0.627 °C (LightGBM) at 1/3/7 days; every
interval excludes zero, every LOCO range keeps its sign, every effect is three
to four times its own MDE. Lead interaction shows the value largely plateaus
after day three (+0.415 from 1→3 d against +0.074 from 3→7 d).

### Phase 2a — within-month shuffled placebo
Sealed **before** the outcome existed, then executed. Decision rule P1
satisfied at every model and lead: the placebo retains 0.0 / 2.0 / 8.8 %
(LightGBM) and −0.5 / 1.7 / 8.4 % (ResidualLightGBM). The forcing value is
event-scale weather information, not seasonal phase. This is the first result
in the project whose interpretation rule was frozen before its outcome, and P3
— under which the finding would have been withdrawn — was reachable and tested.

### Phase 8 — station heterogeneity
Six attributes declared before inspection. Anomaly half-life is the strongest
predictor of the forcing value (Spearman −0.49 [−0.66, −0.16] at seven days,
−0.82 at one day): rivers that hold a thermal anomaly need tomorrow's weather
less. Flashier rivers depend more (+0.48), warmer and lower-latitude stations
depend more. The nuisance control is **not** null — record completeness
covaries at −0.29 [−0.45, −0.07] — so part of the gradient may be data
availability, and the manuscript says so.

### Phase 9 — event and decision metrics
The largest practical result in the project. Seven days, F0 → F3: detection of
station-q95 exceedance rises 0.18 → 0.55 and rapid warming 0.18 → 0.52, while
the false-alarm ratio *falls* (0.36 → 0.20 and 0.28 → 0.20). Not a base-rate
trade; CSI improves on both counts. Thresholds are training-period only.

### Referee-report gaps closed with measurements
- **Anchor sensitivity (Major 6b).** Eight one-factor variants rescored against
  the published predictions. Seven-day headline ranges 0.025–0.063 with the
  published 0.038 in the middle; a cruder climatology inflates it, a directly
  fitted lead-h decay produces a stronger reference and cuts it. The clip never
  binds. The seven-day number is the paper's least robust (factor 2.5 against
  1.1 and 1.3 at one and three days).
- **Decomposition intervals (Major 6a).** Memory [0.386, 0.570] and learned
  [0.057, 0.086] do not overlap. The fraction is over 113 of 116 stations, and
  the three exclusions are explained rather than dropped.
- **Family power (Major 7).** The equivalence claim was on the two rows that
  cannot bear it: at seven days the ThermoRoute-vs-LightGBM effect is smaller
  than anything the cluster structure can resolve, and at three days it favours
  the reference, which a one-sided test cannot address. The three
  damped-persistence rows are three to seven times their MDE. Wording corrected
  throughout.

### Maintenance
Twenty pre-existing collection errors fixed (a tracker edit in `0ace001` had
silently disabled the v5 runner); five permanently-red tests re-scoped from
pre-run facts to invariants that outlive the run.

---

## Running at the time of writing

- **Phase 2b — ±7-day shift arms** (12 fits). Runner and twelve tests
  committed; watchdog armed. Closes decision rule P4, the timing-specificity
  control the shuffle cannot supply.
- **Phase 7 — forced-hybrid scoring** (~3 h). Calibration is complete and
  committed: converged in 28 iterations, median α 0.101, air-temperature
  coefficient 0.818 °C per °C, shortwave 0.641, precipitation and wind ≈ 0. The
  scorer rolls it forward on the tree arms' own keys under F0 and F3_full to ask
  whether the trees' forcing value is available to a simple thermal-response law.

---

## Not done, and honestly why

### Achievable here, not yet done
| Item | Cost | Note |
|---|---|---|
| Phase 2c — 8 component arms (single + leave-one-out) | 48 fits, ~2 h | Sealed under the same protocol; attributes the forcing value to variables |
| Phase 3 — corrected L0/L1/L2 ladder, L2-U2 | Large implementation + hours | **The blocker.** Needs L-masking built into the v5 feature path. Deliberately not rushed: a hurried masking implementation is precisely how DLOG-025 destroyed the last ladder |
| Phase 4 — 48-cell F×L×G×A matrix | Blocked on Phase 3 | Also needs 10-seed geometry and the 24 LightGBM spatial cells |
| Phase 5 — hard-regime architecture | Blocked | `run_neural_information_regimes.py` imports no `torch` and fits nothing; a real plain-TCN runner does not exist |
| Phase 6 — F2a temperature-only diagnostic | Moderate | Acquisition is verified and the key registry frozen; needs the matched `F3_temperature_only` arm and the denominator guard |

### Not achievable from this repository
| Item | Blocker |
|---|---|
| Phase 10a — expanded temperature-only cohort (≥300 gauges) | Requires a new USGS discovery query and Daymet/gridMET acquisition. Network retrieval and provider terms are outside this environment |
| Phase 10b — 2024–2025 frozen audit | Same: the outcome data does not exist in the repository |
| Author list, affiliations, ORCIDs, funding, CRediT, competing interests | Identity data only the authors hold |
| Data and software DOI, licence, release tag, repository URL | Requires the byte-level rights review and minting authority |
| Independent reproduction on an independent host | Requires an operator who is not an author |

### Deliberately not done
- **Phase 11 — information-regime rewrite.** The external review's own rule is
  not to rewrite the final Abstract before the 48-cell matrix exists, because
  it would change the paper's central claim a third time. The Abstract was
  corrected to be defect-free and correctly scoped, not reframed.

---

## Remaining manuscript debts

- Tables are still numbered 4.6 / 4.7a / 4.12 / 4.13 / 4.14 rather than
  Table 1..n. Renumbering touches the manuscript, the ledger span tokens, the
  table generator and the TeX simultaneously.
- The PDF is 49 pages against AGU's ~25 publication-unit guidance, and Section
  4.7 has grown substantially this session.
- SI: `[pending computation]` cells remain in SI02, SI03, SI04, SI05, SI09,
  SI10, SI14, SI15; SI08 reports nine probability metrics as not computed;
  SI12's qualifier sensitivity is unrecomputed; Figures S4–S10 have no files.
- 306 MB of placebo shards are tracked in git, consistent with the v5 precedent
  but worth a deliberate decision.

## One pattern worth naming

Three separate times this session a governance binding made its own subject
unreachable: the v5 execution authority pins the append-only decision log, a
tracker edit removed a literal token the v5 runner requires, and the
forced-hybrid seal binds the runner hash while the runner pins the seal hash.
Each was a hash or content binding on something the protocol itself requires to
change. The fix in each case was to bind the part that should be stable and
exclude the part that must move; the general lesson is that a binding whose
subject must change is not a safeguard, it is a time bomb with a long fuse.

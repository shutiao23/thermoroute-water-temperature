# Strong-Accept Redesign — Decision Log

Every material decision in the `wrr/strong-accept-v1` work stream is recorded
here BEFORE the change is implemented, so a reviewer can see which choices were
made without reading the held-out outcomes.  Entries are append-only; do not
edit an existing entry.

Template:

```markdown
## YYYY-MM-DD — <id>

Decision: ...
Why: ...
Evidence available before the decision: ...
Data periods already inspected: ...
Changes a primary hypothesis? yes/no
Requires a protocol version bump? yes/no
Commit(s): ...
```

---

## 2026-08-08 — DLOG-001: protocol v1 freeze

Decision: Adopt `protocols/wrr_strong_accept_protocol_v1.yaml` as the governing
protocol for the redesign; tag the pre-redesign state
`wrr-pre-strong-accept-4c282e9` and archive the pre-redesign result artifacts
under `outputs/archive/pre_strong_accept_4c282e9/`.

Why: The external review requires an auditable chronology and a frozen
estimand/decision-rule set before any new experiment runs.

Evidence available before the decision: The review report; the tracked results
under `outputs/conventional/`; the manuscript.

Data periods already inspected: 2019-2020 (development) and 2021-2023 (holdout)
were already inspected by the authors before this log was opened.

Changes a primary hypothesis? no
Requires a protocol version bump? no (this IS the v1 freeze)
Commit(s): (next commit on `wrr/strong-accept-v1`)

---

## 2026-08-08 — DLOG-002: single results authority

Decision: All headline numbers will be derived only from `outputs/final/`
(forecast_keys, predictions, station_metrics, paired_effects,
decomposition_effects, paper_values.tex) via
`scripts/final/build_results_authority.py`; the manuscript will reference
`\newcommand` macros from `paper_values.tex` for every headline value instead of
hand-entered numbers.

Why: The review found contradictory estimates in the manuscript (pooled vs
station-median), conflicting station counts (61/65/67 vs 116 vs 118), and a
non-additive decomposition.

Evidence available before the decision: `outputs/conventional/validation_report_2021_2023.json`
reports 61/65/67 reportable stations while `station_metrics_2021_2023.csv` and
the manuscript report 116; `skill_table_2021_2023.csv` and the manuscript
differ on the exact skill numbers; Table 4.11 medians do not add to the
station-median differences.

Data periods already inspected: 2021-2023 (the conflicting artifacts are all
holdout-period).

Changes a primary hypothesis? no
Requires a protocol version bump? no
Commit(s): (results-authority commit)

---

## 2026-08-08 — DLOG-003: holdout scorer re-run with plain controls

Decision: Re-run `scripts/conventional_holdout_2021_2023.py` against the same
frozen bundles (`*_665afcb4d674161943ae` set) with plain controls enabled, so
`PlainCausalTCN-7var` and `PlainMLP-7var` receive an independent 2021-2023
evaluation on the identical common keys (the reviewer's Major Comment 5), and
the stale validation report is regenerated with truthful gate details.

Why: The plain-control checkpoints and the G15 admission machinery already
exist; only inference cost is involved.  The current validation report is stale
and its gate detail strings are misleading templates.

Evidence available before the decision: `outputs/models/plain_control_run/checkpoints/`
exists with 5 seeds per arm; `plain_controls_g15.json` proves the frozen
preprocessing borrow reproduces the stored dev predictions to bit-level;
`validation_report_2021_2023.json` gates carry pass=true with
"violated/invalid/disagrees" detail strings.

Data periods already inspected: 2019-2020 (G15 dev reproduction was run when
the controls were trained) and 2021-2023 (the reported holdout metrics).

Changes a primary hypothesis? no (adds an independent-window re-test of an
existing claim; the architecture conclusion itself was already published in
the manuscript).
Requires a protocol version bump? no
Commit(s): (scorer-fix and results commit)

---

## 2026-08-08 — DLOG-004: mechanism terminology and state definitions

Decision: (1) rename "e-folding half-life" to "anomaly half-life"; (2) compute
half-life from the official train-fitted damped-persistence anchor phi
(`features.DampedPersistenceAnchor`) rather than a re-fitted AR(1); (3) split
"fastest warming" into actual rapid warming (positive station-specific
threshold) and actual rapid cooling (negative); (4) replace global flow/warmth
quantiles with station-specific (and station-month-specific) training-period
quantiles; (5) add issue-time-identifiable states as the primary regime
analysis, with target-conditioned states demoted to explicitly-labelled
retrospective diagnostics; (6) compute all state metrics station-first with a
30-key minimum per station-state cell.

Why: The review's Major Comment 6 documents a factual error ("largest 10% daily
warming" computed from |dT|), a terminology error (half-life vs e-folding
time), global-threshold stratification that mixes basin size and climate, and
pooled (not station-first) state metrics.

Evidence available before the decision: `scripts/conventional_mechanism_analysis.py`
and `outputs/conventional/mechanism_2021_2023.json`.

Data periods already inspected: 2021-2023 (mechanism strata) and 2006-2015
(training thresholds; the official anchor is fitted on training only).

Changes a primary hypothesis? no — it changes the definition of the
*diagnostic*, not the estimand.
Requires a protocol version bump? no (the mechanism state list is in v1).
Commit(s): (mechanism-rework commit)

---

## 2026-08-08 — DLOG-005: spatial factorial experiment

Decision: Replace the single-arm region-vs-random comparison with the 2x2
factorial (geometry {random-site, whole-region} x adaptation {target-local,
training-pooled}) defined in the protocol, run with station-agnostic LightGBM
(raw target and residual target), 5 random-split seeds, 4 deterministic
leave-HUC2 folds, and distance/hydroclimatic-novelty regression; the region
arm's preprocessing will exclude target-region long histories in the pooled
cells.

Why: The review's Major Comment 4 requires separating geometry from local
adaptation and repeated random splits before the spatial claim can stand.

Evidence available before the decision: `scripts/conventional_region_transfer_2021_2023.py`
(single model, one seed, one random split, local adaptation in both arms) and
`region_transfer_summary.json`.

Data periods already inspected: 2021-2023 (the held-out spatial scoring window).

Changes a primary hypothesis? no
Requires a protocol version bump? no
Commit(s): (spatial-factorial commit)

---

## 2026-08-08 — DLOG-006: residual model-class benchmark scope

Decision: The matched residual benchmark on 2021-2023 will cover, for the
primary comparison: residual LightGBM (per-lead), the information-matched plain
causal TCN and plain MLP (from the re-run scorer), and the full ThermoRoute —
all sharing the same damped anchor, inputs, keys and station policy.  Residual
LSTM per-lead training and an equalized 12-config-per-family tuning sweep are
deferred and recorded as limitations: the existing global LSTM is a
joint-horizon non-residual baseline and cannot be relabelled as matched.

Why: The review's Major Comment 5 asks for an independent-window matched
comparison; the plain controls already satisfy the information-matching
requirements by construction (same anchor, same inputs, 2%-parameter match,
same seeds), while a full per-lead 12-config neural sweep exceeds the available
CPU budget for this session and would not change the pre-registered decision
rules.

Evidence available before the decision: `plain_controls_g15.json` (dev
reproduction of the controls), `development_controls_metric_summary.csv`
(dev-period effects).

Data periods already inspected: 2019-2020 (dev controls), 2021-2023 (the
reported holdout metrics).

Changes a primary hypothesis? no
Requires a protocol version bump? no
Commit(s): (results commit)

---

## 2026-08-08 — DLOG-007: 2024-2025 audit and no-flow cohort deferred

Decision: The 2024-2025 frozen audit window and the no-flow core cohort
expansion are not run in this session.  The protocol defines them (P2 scope);
the decision log will record the exact freeze commit and hash BEFORE any such
audit is opened, and the manuscript will not claim either has been performed.

Why: The 2024-2025 provider data has not been acquired (no snapshot cache, no
network fetch in this environment), and the 1,465-station candidate registry
needed to re-derive a no-flow cohort is not present in the checkout (only the
rejection ledger is).  Running them would fabricate evidence.

Evidence available before the decision: `data_usgs/rejected_sites_120v2.csv`
(rejection ledger only); absence of any 2024-2025 acquisition cache under
`outputs/conventional/raw_2021_2023/`.

Data periods already inspected: none beyond 2023.

Changes a primary hypothesis? no
Requires a protocol version bump? no
Commit(s): none (documented only)

---

## 2026-08-08 — DLOG-008: spatial decision rule R1 applied

Decision: The independent-window matched spatial penalty is +0.006 / +0.009 /
+0.007 °C at 1 / 3 / 7 d (LightGBM, local adaptation; +0.006 / +0.007 / +0.004
for the residual-target tree), consistently signed across five split seeds and
two model targets but below the 0.02 °C threshold of decision rule R1.  The
spatial partition is therefore reported as a secondary finding: it is removed
from the Key Points, the abstract states it as "a small, consistently signed
additional error of about 0.01 °C — an order of magnitude smaller than the
reference-model effect", and the development-period "removes a third" framing
is kept only as a development-period descriptive result.

Why: The pre-registered decision rule was triggered; the alternative (keeping
spatial as a headline) would have contradicted the protocol.

Evidence available before the decision: `outputs/final/spatial_summary.json`
(computed after the factorial completed; the protocol was written before the
factorial ran).

Data periods already inspected: 2021-2023 (the factorial scoring window).

Changes a primary hypothesis? no
Requires a protocol version bump? no
Commit(s): (spatial-factorial commit)

---

## 2026-08-08 — DLOG-009: HUC2 cluster-map bug fix

Decision: `spatial.huc2_cluster_map` now zero-pads the registry's `huc2`
codes, restoring the true 15 whole-HUC2 clusters (the previous code matched
only the two-digit codes 10-18 and silently treated the nine single-digit
codes as per-site clusters, yielding 71 clusters).  All cluster inference was
regenerated; the five-test family p-values and intervals changed accordingly
(e.g., ThermoRoute−LightGBM at 7 d: CI now includes zero, Holm p = 0.148).

Why: The cluster structure is a protocol-level scientific fact; the bug
inflated the effective cluster count and could have changed conclusions.

Evidence available before the decision: the registry's `huc2` column stores
integers 1-18; the regex `\d{2}` only matched two-digit strings.

Data periods already inspected: none (structural fix, no outcomes re-read).

Changes a primary hypothesis? no (inference sensitivity only)
Requires a protocol version bump? no
Commit(s): (spatial-factorial commit)

---

## 2026-08-08 — DLOG-010: plain controls admitted to the holdout

Decision: Following the G15 preprocessing-borrow gate passing with
`max_abs_diff = 0` for both information-matched plain controls, the two arms
(PlainMLP-7var, PlainCausalTCN-7var) were scored on the 2021-2023 window with
their frozen weights.  The held-out result: the full architecture retains a
small, consistent edge (median paired ΔRMSE +0.010 / +0.011 / +0.015 °C at
1 / 3 / 7 d; plain-TCN win fractions 0.30 / 0.30 / 0.16).  This is below the
0.02 °C threshold of decision rule R2 but the direction is stable, so the
manuscript reports the full architecture as primary with the plain TCN as the
parsimonious description, and does not claim a material architecture
advantage.

Why: Major Comment 5 of the review asked for the independent-window re-test;
the frozen checkpoints and gate machinery made it an inference-only cost.

Evidence available before the decision: `plain_controls_g15.json` (dev
reproduction) and the scorer's G15 admission for the rerun.

Data periods already inspected: 2021-2023 (the scoring window).

Changes a primary hypothesis? no
Requires a protocol version bump? no
Commit(s): (results commit)

---

## 2026-08-08 — DLOG-011: flow retraining ablation result

Decision: Retraining the global LightGBM without the FLOW channel costs
+0.042 / +0.034 / +0.009 °C station-median RMSE at 1 / 3 / 7 d (no-flow win
fractions 0.03 / 0.14 / 0.37).  The manuscript now reports discharge as
carrying small but systematic information at the shortest lead, replacing the
earlier "nearly insensitive to discharge" reading of the scale-perturbation
probes; the cohort trade-off is quantified only when a no-flow core cohort is
re-derived (P2).

Why: Major Comment 7 of the review required a retraining test, not only
synthetic perturbations.

Evidence available before the decision: `outputs/final/flow_ablation_effects.parquet`.

Data periods already inspected: 2021-2023 (scoring window).

Changes a primary hypothesis? no
Requires a protocol version bump? no
Commit(s): (flow-ablation commit)

## 2026-08-08 — DLOG-012: spatial-factorial software defects disclosed and corrected

Decision: Three software defects in `scripts/final/run_spatial_factorial.py` are
recorded here and corrected by a new runner (`scripts/final/run_information_ladder.py`,
protocol v2), not by patching the archived script:

1. Resume path data loss: the resume set is built from an existing
   `spatial_effects.parquet` but cached cells are never re-read into the output;
   the final `to_parquet` overwrite therefore permanently drops resumed cells.
   Evidence: `spatial_factorial_run2.log` line 1 ("resume: 7 cells cached"),
   281 of 288 expected cell lines in the log, and seven region cells reporting
   90 of 120 stations in `spatial_effects.parquet`.
2. Pooled-preprocessing cache key omits the fold: `prep_key = (variant[0],
   variant[1], h)` with `variant[1]` the seed, so all four folds of a pooled
   cell reuse fold 0's in-fold climatology / damped anchor / imputation. Fold 1-3
   held-out stations are inside fold 0's training set, so pooled statistics are
   contaminated (about 33% of stations) and the pooled penalty is a lower bound.
3. Prediction loss: key-level predictions are written to
   `spatial_effects.parquet` and then overwritten by site-level metrics at the
   same path, so per-key diagnostics are unrecoverable from the artifact.

Why: The external review's cell-level audit found station-count mismatches
across cells; the log and parquet reconstruction above show the cause is data
loss plus cache leakage, not a reportability threshold.

Evidence available before the decision: the run logs, the parquet columns and
station counts, and the code itself.

Data periods already inspected: 2021-2023 (independent window) and the
2006-2015 fit period.

Changes a primary hypothesis? no (the affected quantities — pooled-arm
penalties and adaptation effects — are recomputed under protocol v2 with
fold-complete, leak-free preprocessing; the local-arm geometry results are
unaffected and remain as reported).

Requires a protocol version bump? yes (protocol v2, DLOG-013).

Commit(s): (this worktree)

## 2026-08-08 — DLOG-013: protocol v2 freeze — information ladder, regulation strata, and process classes

Decision: Freeze `protocols/wrr_strong_accept_protocol_v2.yaml` as the governing
protocol for the rerun of the spatial/information experiment and the
hydrologic-similarity analysis. It fixes, before any new outcome is read:

(a) a four-level information ladder L0/L1/L2/L3 x geometry {random, region}
with per-level input masks and a common key registry across all eight cells;
(b) regulation strata by upstream storage ratio (SR < 0.05 unregulated,
0.05-0.5 moderate, >= 0.5 strongly regulated) with a pre-fixed merge rule
(moderate into strong below 15 stations);
(c) a four-class process classification (regulated, snowmelt, groundwater,
rain) with a pre-fixed priority order;
(d) pre-registered model forms for the half-life attribute regression (main
and parsimonious) with leave-one-HUC2-out cross-validation and no stepwise
search;
(e) a new inference family of 15 tests (N1-N15) covering the ladder contrasts,
Holm-corrected within the family, using the existing HUC2 whole-cluster
sign-flip and cluster bootstrap;
(f) four quantitative predictions P-1..P-4 to be tested against the rerun;
(g) stopping rules and degradation paths (GAGES-II match rate < 70% degrades
L3 and the attribute regression to exploratory on the matched subset).

The original frozen five-test family of protocol v1 is unchanged. The protocol
v2 seal records SHA-256 digests of `outputs/final/` taken before any new
outcome was read; the digest file also serves as the pre-outcome snapshot.

Why: The external review requires the ungauged question to be answered within
this paper's experiment matrix, and any rerun must be fold-complete and
leak-free (DLOG-012) with the analysis committed before the outcomes.

Evidence available before the decision: the review report; DLOG-012's evidence;
the protocol v1 decision set.

Data periods already inspected: 2021-2023 (independent window) and the
2006-2015 fit period (these were inspected before this entry; no new outcome
was read for the v2 design after the seal was created).

Changes a primary hypothesis? no primary hypothesis of protocol v1 changes; v2
adds a new comparison family and four predictions.

Requires a protocol version bump? yes (this is the bump).

Commit(s): (this worktree)

## 2026-08-08 — DLOG-014: protocol v2 engineering amendment + ladder key-semantics fix

Decision: Record the second-round review findings and fixes:

1. The ladder now scores EXACTLY the common forecast-key registry (two-sided
   assertion; previously the assertion was one-sided and the cell scored a
   ~8% superset whose extra keys were mostly issue-dates with unobserved
   water temperature — 1,740 of 2,445 extra keys in the audited cell).  A
   --legacy-keys mode reproduces the archived key semantics ONLY for the G1
   side-effect-free refactor test; its outputs are never compared with the
   main held-out tables.
2. Training rows obey the same admissibility rule as the registry
   (issue-date WTEMP genuinely observed, unmasked panel); the admissibility
   column is excluded from features.
3. The y_observed filter is NaN-safe (numeric > 0, not astype(bool), which
   maps NaN to True); metrics count only keys with real labels and use
   ndarray means so NaN propagates; a reportable flag (n >= 100) is written
   into ladder_effects rather than silently truncating.
4. The consistency gate's used_in semantics are now all-mode (every declared
   span must print the value), enforced with an audited ledger
   (audit_used_in.py) and a warning-level contradiction scan.
5. An engineering clause requiring reverse-direction assertions is frozen as
   protocols/wrr_strong_accept_protocol_v2_engineering_amendment_v1.yaml;
   the v2 seal remains valid because no estimand, family, or threshold
   changed.
6. SI07 air2stream rows are split into the reportable 116-station set and an
   explicitly labelled 118-station no-reportability-filter sensitivity; the
   MAE/bias columns are corrected to station medians (they were pooled
   key-level values).  Six air2stream claims were added to the ledger and
   SI07 is scanned by the gate.
7. Markdown table header/alignment mismatches (SI11 x4, SI08 x1) fixed and a
   table-shape check added to the gate.

Why: second-round external review (registry dilution of C2, single-sided
assertions, NaN-as-True, presence-not-equality).

Evidence available before the decision: shard-vs-registry key analysis
(32,850 vs 30,405 keys; 1,740 unobserved-issue-date extra keys; 210 null
labels), the INJ4 injection result, and the n=1095 constant-count check.

Changes a primary hypothesis? no (the v2 estimates themselves were not yet
computed; the L0 production cells were re-scored on the registry).

Requires a protocol version bump? no (amendment v1 to protocol v2).

Commit(s): (this worktree)

## 2026-08-09 — DLOG-016: protocol v3 forcing ladder — P-5/P-6 verdicts

Decision: Protocol v3 (DLOG-015) sealed before any F-axis outcome; the
forcing ladder ran (scripts/final/run_forcing_ladder.py, registry-scored,
two-sided assertions).  Interim verdicts on the tree models (LightGBM and
ResidualLightGBM, frozen per-lead hyperparameters, whole-cohort fits):

  lead   F0 rmse   F3 rmse   F3-F0     learned minus damped (F0 -> F3)
  1 d    0.625     0.479     -0.146    -0.135 -> -0.277
  3 d    1.372     0.806     -0.566    -0.083 -> -0.584
  7 d    1.735     1.113     -0.622    -0.039 -> -0.603

P-5 (F3 expands the learned advantage over damped persistence at 7 d to at
least -0.30 C): CONFIRMED on the interim tree signal (-0.603 C; 7 d RMSE
1.113 inside the predicted 1.15-1.25 band).  The deep-architecture arm is
scheduled via its sequence-input channels; the protocol's stopping rule
explicitly allows the interim evaluation on the best tree model.

P-6 (forcing value grows with lead): CONFIRMED (0.622 > 0.566 > 0.146).

P-7 (F1 recovers a material fraction at 3/7 d): pending the F1 rerun
(the first F1 run carried a feature-list bug and was discarded).

Interpretation: the advisory review's D1 is quantitatively confirmed - the
F0 learned-gain null was an information-set consequence.  At 7 days perfect
forcing is worth ~0.62 C of station-median RMSE (a 36% reduction) while the
architecture is worth at most 0.023 C: a ~27:1 ratio.  The paper's central
claim moves from "benchmark design governs reported skill" to "the
information budget governs predictability: forcing value ~0.62 C, local
record value ~0.20 C (pooled vs local, protocol v2), geometry value
~0.005 C, architecture value <=0.023 C".

Why: advisory review D1 (unidentified central attribution without a
forcing-information arm).

Evidence available before the decision: the sealed protocol v3; the registry
assertions; forcing_effects.parquet.

Changes a primary hypothesis? yes - the D1 attribution question is answered;
the F0 framing of Sections 5.1/Conclusions/Abstract/Key Points is superseded.

Requires a protocol version bump? no (v3 governs).

Commit(s): (this worktree)

## 2026-08-09 — DLOG-017: L-axis completed (protocol v2) + full information budget

Decision: The 432-cell information ladder (L0 gauged-local, L1 pooled,
L2 thermally ungauged; region + random geometries; LightGBM +
ResidualLightGBM; leads 1/3/7) completed with the fold-complete,
registry-scored runner.  Seven-day station-median RMSE by level (reportable
stations, random arm): L0 1.736/1.714, L1 1.897/1.850, L2 2.832/2.784
(LightGBM/ResidualLightGBM); region arm: L0 1.756/1.725, L1 1.939/1.876,
L2 3.308/3.260.

Paired station deltas vs L0 at 7 d: L1 +0.16..+0.21 C; L2 +0.99..+1.47 C.
The geometry effect (region minus random) is +0.005..+0.02 C at L0 and
+0.48 C at L2 — a ~30-fold amplification when local thermal history is
absent, quantitatively confirming advisory finding D2 (the spatial null was
constructed, not discovered).

Combined with the forcing ladder (DLOG-016), the paper's information budget
at 7 days is now: future forcing -0.62 C (F3 vs F0); local thermal history
+1.0..+1.5 C (L2 vs L0); local statistics +0.16..+0.21 C (L1 vs L0);
geometry +0.005..+0.02 C (at L0) or +0.48 C (at L2); architecture
<=0.023 C.  This is the positive, quantitative, transferable statement the
advisory review required (S2).

Why: protocol v2 runner completed; advisory review's S2 acceptance criterion.

Evidence available before the decision: ladder_effects.parquet (12,744
station-cells), completeness gate 432/432, two-sided registry assertions,
reportable flags.

Changes a primary hypothesis? yes — the spatial-transfer conclusion is now
reported as the conditional statement with the L2 geometry amplification
quantified.

Requires a protocol version bump? no.

Commit(s): (this worktree)

---

## 2026-08-09 — DLOG-015: protocol v3 freeze record (post-outcome administrative reconstruction)

**Chronology status: administratively reconstructed from existing v3 seal;
entered after outcome, not new pre-registration.**  This entry is appended out
of numeric order to preserve the decision log's append-only rule.  It does not
repair the missing contemporaneous DLOG-015 entry and must never be cited as
evidence that a new decision was made before outcome access.

Decision: Record the decision that can be reconstructed from the already
existing `protocols/wrr_strong_accept_protocol_v3.yaml` and
`protocols/wrr_strong_accept_protocol_v3_seal.json`: protocol v3 introduced the
F0/F1/F2/F3 forcing axis, its cross-matrix plan, predictions P-5--P-7, and the
stopping rules later referenced by DLOG-016.  The seal states a freeze time of
2026-08-09T00:30:18Z at git commit
`c63eef57d46366c0248bcc3e21be995371c32b66`, with `git_dirty: true`.

This reconstruction records only what those existing objects assert.  The v3
protocol, seal, and runner were not tracked by the cited commit, and the seal
hashes the then-current `outputs/final/` files but not the protocol YAML, runner,
or decision log.  It therefore remains
`PROVISIONAL_NOT_YET_COMPLETE` chronology evidence rather than an immutable
pre-outcome registration receipt.

Why: DLOG-016 cites DLOG-015, but no DLOG-015 entry existed in this append-only
log.  Leaving the gap unexplained would overstate the audit trail; silently
inserting an apparently contemporaneous entry would be worse.  This explicit
administrative reconstruction preserves both the recovered intent and the true
evidence boundary.

Evidence available before this reconstructed entry: protocol v3, its existing
seal, DLOG-016 and DLOG-017, the current F/L artifacts, and the completed
evidence-status audit.  F- and L-axis outcomes had already been inspected.

Data periods already inspected: 2006-2017 fitting/validation data and the
2021-2023 evaluation window, including the reported F- and L-axis outcomes.

Changes a primary hypothesis? no — administrative chronology repair only; it
introduces no new prediction, threshold, estimand, or verdict.

Requires a protocol version bump? no.  Any future crossed information-regime
experiment requires its own prospective protocol rather than relying on this
reconstruction.

Commit(s): (administrative reconstruction in this worktree; to be committed
with DLOG-018 and `docs/SCIENTIFIC_EVIDENCE_STATUS.md`)

---

## 2026-08-09 — DLOG-018: station-first correction and information-budget headline withdrawal

Decision: Withdraw the following quantitative headline interpretations from
manuscript-eligible use: the `+0.48 C` L2 geometry effect, the approximately
`30x` geometry amplification, the forcing-to-architecture `27:1` ratio, the
claim that the separate F and L runs constitute a "full information budget",
and the current CONFIRMED labels for the F3 P-5/P-6 verdicts.  DLOG-016 and
DLOG-017 remain unchanged as the historical record of what was initially
concluded; this entry supersedes those interpretations.

The corrected disposition is:

1. Protocol v2 requires the five random seeds to be averaged within station
   before region-minus-random pairing.  On the current reportable
   `ladder_effects.parquet` rows, the seven-day paired median geometry effects
   are `+0.007968 C` (L0) and `+0.100697 C` (L2) for LightGBM, and
   `+0.003983 C` (L0) and `+0.103896 C` (L2) for ResidualLightGBM.  The
   corresponding absolute L-by-geometry interactions are `+0.092729 C` and
   `+0.099913 C`.  The earlier approximately `+0.48 C` value subtracts two
   marginal medians and is not the registered median station-paired estimand.
   The approximately `30x` ratio is therefore withdrawn.  These corrected raw
   contrasts remain `PROVISIONAL_NOT_YET_COMPLETE` until a governed contrast
   table, inference, robustness checks, and a matching manifest exist.
2. The current forcing authority is not the artifact cited by DLOG-016:
   `forcing_effects.parquet` and `forcing_summary.json` contain F0 only.  The
   table has 708 station-cells (118 stations per model and lead), but only 116
   stations per cell meet the frozen `n >= 100` reportability rule; it has no
   `reportable` column, and its summary includes all 118.  Moreover, the summary
   code subtracts marginal station medians rather than computing the median of
   paired station deltas.  The reported F3 values and the `-0.622 C` contrast
   are therefore `PROVISIONAL_NOT_YET_COMPLETE`, not current authority values.
3. DLOG-016 states that the reported F3 seven-day RMSE of `1.113 C` lies inside
   the pre-registered P-5 range `1.15--1.25 C`; it does not.  If the missing
   raw F3 artifact is recovered or reproduced, P-5's learned-advantage threshold
   and its RMSE-range prediction must be adjudicated separately.  P-5 and P-6
   are downgraded from CONFIRMED to `PROVISIONAL_NOT_YET_COMPLETE`; P-7 and F2
   remain `PLANNED`.
4. The `27:1` ratio combines an ungoverned F3 tree contrast with an architecture
   contrast measured under F0-L0.  The current F and L effects come from
   separate conditional designs, so they cannot be added, ranked as a universal
   ratio, or labelled a closed budget.  Until a common-key F-by-L-by-G-by-A
   experiment exists, the admissible framing is "conditional information
   contrasts," not an additive or full information budget.
5. The information-regime claims are absent from `paper/claim_ledger.yaml`,
   `outputs/final/claim_ledger_resolved.csv`, and
   `outputs/final/paper_values.tex`.  The current `result_manifest.json` binds
   protocol v1 only, omits forcing artifacts, and its recorded digest for
   `ladder_effects.parquet` does not match the current file.  A passing current
   manuscript gate therefore does not close any F/L headline.  No such number
   may enter the manuscript until the ledger, macros, result tables, protocol
   hashes, and manifest form one verified authority state.
6. The 432 registry-clean L0/L1/L2 cells remain useful provisional evidence,
   but they are a completed subset, not completion of protocol v2's declared
   L0--L3 matrix plus U2 sensitivity.  L3, L2-U2, the N1--N15 governed inference
   output, and the crossed design remain incomplete.

Why: A requirement-to-evidence audit applied the frozen primary estimand to the
current raw artifacts and checked the result manifest, claim ledger, protocol
seals, and decision chronology.  It found that the geometry headline used a
difference of marginal medians, the forcing headline lacks a current supporting
artifact and reportability enforcement, and the proposed budget combines
non-crossed conditional contrasts.

Evidence available before the decision: `protocols/wrr_strong_accept_protocol_v1.yaml`,
`protocols/wrr_strong_accept_protocol_v2.yaml`,
`protocols/wrr_strong_accept_protocol_v3.yaml`, the current
`outputs/final/ladder_effects.parquet`, `forcing_effects.parquet`,
`forcing_summary.json`, `result_manifest.json`, the claim ledger and generated
macros, and `docs/SCIENTIFIC_EVIDENCE_STATUS.md`.  The two-sided shard audit
passes for all 12,744 current L0/L1/L2 station-cells; that key-integrity result
does not cure the estimator, completeness, inference, or manifest gaps.

Data periods already inspected: 2021-2023 evaluation outcomes and 2006-2017
fit/validation inputs.

Changes a primary hypothesis? yes — it withdraws the current headline verdicts
and narrows the admissible claim to provisional conditional contrasts.  It adds
no favorable replacement hypothesis and does not change the registered
station-first estimand.

Requires a protocol version bump? no for this correction.  A future crossed
F-by-L-by-G-by-A experiment and any new headline decision rules require a new
prospective protocol before outcomes are read.

Commit(s): (this worktree)

---

## 2026-08-09 — DLOG-019: station-level L-by-geometry interaction correction

Decision: Correct only the L-by-geometry interaction interpretation in
DLOG-018.  Its seven-day values `+0.092729 C` (LightGBM) and `+0.099913 C`
(ResidualLightGBM) were calculated as
`median_station(G@L2) - median_station(G@L0)`.  Although each within-level
geometry contrast first averaged the five random seeds within station, this
difference of two marginal medians is still not the station-level
difference-in-differences required for the L-by-geometry interaction.

The protocol-compatible station-level estimand is, for every common reportable
station `s`,

`[(RMSE_region,L2,s - RMSE_random,L2,s) -
  (RMSE_region,L0,s - RMSE_random,L0,s)]`,

followed by the median across stations.  The new create-only information-regime
authority builder applies that ordering on the common 116-station seven-day
set.  Its seven-day L-by-geometry medians are
`+0.087489705415824 C` for LightGBM and `+0.100936172151032 C` for
ResidualLightGBM.

Supersession scope: this entry supersedes only DLOG-018's characterization of
`+0.092729 C` and `+0.099913 C` as L-by-geometry interactions and replaces
those two interaction values.  It does not supersede DLOG-018's within-level
G@L0 or G@L2 paired geometry contrasts, its withdrawal of the `+0.48 C`,
approximately `30x`, `27:1`, or "full information budget" claims, its forcing
artifact findings, or any provisional status.  The corrected interaction
values remain `PROVISIONAL_NOT_YET_COMPLETE`: the v4 protocol is an unsealed,
post-outcome draft, and governed inference, robustness, claim-ledger, and
manuscript bindings remain incomplete.  The published create-only authority's
own status is explicitly `COMPLETED_SUBSET_NOT_FULL_V2`; its existence does not
complete the v2 matrix or seal the v4 draft.

Why: An authority-level contrast audit distinguished the median of paired
station double differences from a difference between the medians of two
paired station contrasts.  Only the former preserves the registered
station-first estimand through the interaction.

Evidence available before the decision: the 432 current L0/L1/L2 shards,
`outputs/final/ladder_effects.parquet`,
`outputs/final/forecast_keys.parquet`, the two-sided key audit, the create-only
`scripts/final/build_information_regime_authority.py` authority builder,
`outputs/final/information_regime_v4/` and its verified output hashes,
DLOG-018, and the unsealed
`protocols/wrr_information_regimes_protocol_v4.yaml` draft.  The authority
contains 4,176 absolute station rows, 6,960 paired contrast rows, and 60
paired-only summary rows; all three leads use 116 common reportable stations.

Data periods already inspected: 2021-2023 evaluation outcomes and 2006-2017
fit/validation inputs.

Changes a primary hypothesis? no — estimator correction only.  It preserves
the conditional L-by-geometry question and does not introduce a favorable new
hypothesis or threshold.

Requires a protocol version bump? no.  This correction does not edit or seal
the existing v2/v3 protocols or the v4 draft.  Any execution of still-unrun v4
extensions requires a prospective seal before outcome access.

Commit(s): (this worktree)

---

## 2026-08-09 — DLOG-020: F0/F3_full station-first forcing normalization authority

Decision: Replace the withdrawn DLOG-016 forcing numbers with the create-only
v4 authority for the already-viewed F0/F3_full tree-model domain.  This is a
post-outcome normalization, not a prospective, preregistered, independent, or
confirmatory experiment.  The authority scores the exact forecast registry,
requires the same keys and registry target values across arms, filters every
cell to the same 116 stations with at least 100 paired targets, and computes
the registered station-first estimand
`median_i[RMSE_i(F0)-RMSE_i(F3_full)]`.

The resulting forcing values at 1/3/7 days are
`+0.118791/+0.492673/+0.540617 C` for LightGBM and
`+0.116222/+0.513690/+0.535455 C` for ResidualLightGBM.  The runner's
F3_full-minus-F0 deltas are the same values with the opposite sign.  The
ordering increases from 1 to 3 to 7 days in both tree models, so the observed
post-outcome normalization matches the qualitative P-6 ordering.  It is not
labelled a confirmatory P-6 verdict because the v3 chronology and seal do not
support that claim.

P-5 is mixed and has no automatic joint verdict.  At seven days, the
station-median F3_full RMSE is `1.113565 C` for LightGBM and `1.121000 C` for
ResidualLightGBM, both outside the registered `1.15--1.25 C` range.  The
within-station learned-minus-damped medians are `-0.607671 C` and
`-0.581723 C`, which meet the `-0.30 C` threshold, and their changes from F0
are `-0.566723 C` and `-0.539034 C`, which meet the registered expansion
component.  The authority therefore records the range component as not met,
the learned-threshold and expansion components as met, and the joint status as
`NOT_ADJUDICATED_BY_AUTHORITY_BUILDER`.  DLOG-016's `0.622 C`, `P-5
CONFIRMED`, and `P-6 CONFIRMED` statements remain historical and are
superseded by this disposition.

Why: DLOG-018 found that the cited forcing artifacts contained F0 only, used
118 unfiltered stations, and subtracted marginal medians.  The v4 runner and
independent authority now reproduce both arms from key-level predictions,
enforce reportability and cross-arm equality, retain canonical protocol-arm
semantics, reconstruct station metrics and contrasts independently, and bind
every input and output hash.

Evidence available before the decision: the twelve key-level shards in
`outputs/final/forcing_shards_v4/`, `forcing_effects_v4.parquet`,
`forcing_contrasts_v4.parquet`, `forcing_summary_v4.json`, the create-only
`outputs/final/forcing_regime_v4_authority/`, and its independently rebuilt
and verified manifest.  The authority contains 1,392 common-station metric
rows and 696 station-paired contrast rows.  Its manifest SHA-256 is
`260baa7e9fae08afb64280cb87fe320fc1712f4d2e0bd8b261ef824b133285b8`;
it binds v4 draft protocol SHA-256
`66e089baf37db1137cad23f148e31df39cc71aea872dec4a97dc6f8701d13a98`
and the twelve-shard aggregate SHA-256
`2e915e654ea302091ac0e35ff704d3c55949ffad1338d95278ef66f1921d93e1`.

Data periods already inspected: 2006-2017 fitting/validation inputs and the
2021-2023 evaluation outcomes, including earlier non-authoritative F0/F3
summaries.

Changes a primary hypothesis? no.  This entry applies the registered
station-first estimator and gives equal weight to met and unmet P-5
components; it introduces no new favorable threshold or hypothesis.

Requires a protocol version bump? no for this already-viewed normalization.
Still-unrun F1, F2, placebos, components, crossed, deep-model, or audit-window
experiments remain governed by a future exact-byte v4 seal.

Commit(s): (this worktree)

---

## 2026-08-09 — DLOG-021: completed-subset whole-HUC2 inference authority

Decision: Freeze approximate fixed-cohort descriptive sensitivities for the
60 model-stratified contrasts in the completed L0/L1/L2 subset.  The
create-only inference authority uses 116 common reportable stations and 15
canonical HUC2 clusters in every cell.  It reports a 10,000-draw whole-HUC2
cluster bootstrap of the equal-station median, complete `2^15` whole-cluster
sign-flip enumeration, equal-HUC2 medians, and all 15 leave-one-HUC2 results.
The analysis is permanently labelled `POST_OUTCOME_NORMALIZATION_ONLY` and
`COMPLETED_SUBSET_NOT_FULL_V2`.

At seven days, the station-first L-by-geometry interaction remains
`+0.087490 C` for LightGBM and `+0.100936 C` for ResidualLightGBM, but its
robustness is model dependent.  LightGBM has a cluster-bootstrap interval
`[-0.039911, +0.555379] C`, exact two-sided sign-flip sensitivity
`p=0.400146`, and equal-HUC2 median `+0.006568 C`.  ResidualLightGBM has
interval `[+0.006921, +0.301810] C`, sign-flip sensitivity `p=0.041016`, and
equal-HUC2 median `+0.079093 C`.  Both leave-one-HUC2 ranges remain positive,
but the LightGBM cluster resampling and equal-HUC result prohibit a
model-general or population-level interaction claim.

The seven-day G@L2 contrast shows the same pattern: LightGBM
`+0.100697 C`, interval `[-0.028897, +0.650207] C`, `p=0.387695`; and
ResidualLightGBM `+0.103896 C`, interval `[+0.011825, +0.319547] C`,
`p=0.041016`.  The smaller G@L0 contrasts are positive in both models, with
intervals excluding zero in this approximate sensitivity.  These raw values
are not Holm-adjusted confirmatory results.

Protocol v2's N01--N15 family is not adjudicated: N07--N09 require the
uncompleted L3 arm, and v2 does not specify which model or, for the C1--C3
presentations, which geometry selects a family member.  The authority records
candidate mappings only, emits no Holm value, and emits no confirmatory
verdict.  Outcome-informed selection is forbidden.

Why: DLOG-019 froze the correct station-level double difference but retained
provisional status pending governed inference and robustness.  The new
builder binds the upstream authority and canonical HUC2 registry, and an
independent audit exactly recomputed all 60 medians, equal-HUC summaries,
900 leave-one-HUC rows, all sign-flip tails, and representative bootstrap
draws before approving the formal artifact.

Evidence available before the decision:
`outputs/final/information_regime_v4/`,
`scripts/final/build_information_regime_inference_v4.py`, and
`outputs/final/information_regime_inference_v4/`.  The formal inference
manifest SHA-256 is
`1bce4a92c0b90193a6c64edf26eaaa43f323e5335374c12dbf78cb82d3fa515c`;
the builder SHA-256 is
`9ada6b7d2136af038b77be35edb8bd9063930c1909e4d0ee83d13e28cb6193ad`.
The summary has 60 rows, and the per-HUC2 and leave-one-HUC2 tables have 900
rows each.

Data periods already inspected: 2006-2017 fitting/validation inputs and the
2021-2023 evaluation outcomes, including all L0/L1/L2 subset outcomes.

Changes a primary hypothesis? no.  This is a post-outcome robustness
normalization that exposes model dependence and withholds the incomplete
multiple-comparison verdict.

Requires a protocol version bump? no.  It neither completes protocol v2 nor
authorizes any unrun v4 extension.

Commit(s): (this worktree)

---

## 2026-08-09 — DLOG-022: F0/F3_full whole-HUC2 descriptive inference authority

Decision: Freeze whole-HUC2 robustness summaries for the six already-viewed
F0-versus-F3_full tree-model forcing contrasts.  The create-only authority
uses the registered positive forcing-value statistic
`RMSE_i(F0)-RMSE_i(F3_full)`, 116 common reportable stations, and 15 canonical
HUC2 clusters in every model-by-horizon cell.  It reports a 10,000-draw
whole-HUC2 cluster bootstrap of the equal-station median, every one of the
`2^15` whole-cluster sign flips, an equal-HUC2 sensitivity, and all 15
leave-one-HUC2 estimates.  Its permanent status is
`POST_OUTCOME_NORMALIZATION_ONLY / COMPLETED_VIEWED_DOMAIN_SUBSET /
APPROXIMATE_FIXED_COHORT_DESCRIPTIVE`; it is neither prospective nor
confirmatory.

The LightGBM forcing values at 1/3/7 days are
`+0.118791/+0.492673/+0.540617 C`, with whole-HUC2 bootstrap intervals
`[+0.075163,+0.169223]`, `[+0.360625,+0.651083]`, and
`[+0.406486,+0.702479] C`.  The corresponding ResidualLightGBM values are
`+0.116222/+0.513690/+0.535455 C`, with intervals
`[+0.074042,+0.164942]`, `[+0.355890,+0.673587]`, and
`[+0.391430,+0.695267] C`.  The exact positive-tail counts are respectively
4/2/2 and 4/4/2 of 32,768 configurations; the two-sided absolute counts are
8/4/4 and 8/8/4.  These are raw descriptive sensitivities, not adjusted
p-values or binary significance decisions.

All 90 leave-one-HUC2 estimates remain positive.  The equal-HUC2 medians at
1/3/7 days are `+0.137610/+0.613368/+0.656035 C` for LightGBM and
`+0.131188/+0.606121/+0.602032 C` for ResidualLightGBM.  These summaries show
that the positive forcing contrast is not produced by one HUC2 in this fixed
cohort, but they do not turn the retrospective gridded oracle into an
operational forecast, a population law, or a prospective hypothesis test.
No Holm adjustment, claim verdict, P-5 adjudication, or architecture ratio is
computed by this authority.

Why: DLOG-020 froze corrected station-first point estimates but did not yet
bind clustered uncertainty or cluster-influence sensitivities.  The new
builder validates the upstream forcing authority's exact manifest, 696 paired
station rows, station registry, sign convention, common-station inventory and
hash chain before computing the six cells.  An independent implementation
then reproduced all six point estimates, all 90 per-HUC2 values, all 90 LOCO
values, all exact tail counts, deterministic seeds, and every bootstrap draw
summary with maximum absolute discrepancy zero.  A second formal-path check
verified create-only publication and byte equality of the four result files
against the audited temporary build.

Evidence available before the decision:
`outputs/final/forcing_regime_v4_authority/`,
`scripts/final/build_forcing_regime_inference_v4.py`, and the create-only
`outputs/final/forcing_regime_inference_v4/`.  The formal inference manifest
SHA-256 is
`580fed64c308267c1ed0c49e60dac8f21ff1657651de4ee70e3b179d6ed758d7`;
the input-set SHA-256 is
`c6bf211c565814943b05a353a77f3285b01dac4558980a1d69b777f36d40c1a1`;
the output-set SHA-256 is
`ed1cb4d4b8750a9d50e4a2f43627b64982ad28b17b1d6589d91538ac08f366c1`;
and the builder SHA-256 is
`9aa6035ed1477f2e435ac2a263d4ab91e1c67f5c813549e2270647f32497d9da`.

Data periods already inspected: 2006-2017 fitting/validation inputs and the
2021-2023 evaluation outcomes, including the earlier forcing summaries.

Changes a primary hypothesis? no.  This is a robustness normalization of the
already-viewed forcing domain and introduces no new threshold, favorable
contrast, or multiplicity choice.

Requires a protocol version bump? no.  It neither seals v4 nor authorizes F1,
F2, placebos, components, crossed, neural, U2, novelty, event, cohort, or
audit-window execution.

Commit(s): (this worktree)

---

## 2026-08-09 — DLOG-023: label-free F2a acquisition verified, not promoted

Decision: Freeze the completed input-acquisition verification for the
retrospective Open-Meteo Previous Runs/GFS fixed-lead air-temperature
composite.  This decision binds an input product only.  It does not promote an
F2a model arm, authorize execution under the still-draft v4 protocol, or add a
forecast-value result.

The immutable acquisition contains 4,080 station-month chunks for 120
stations and 362,520 rows over 2021-03-30 through 2023-12-31.  Of those rows,
361,800 satisfy the complete fixed-lead record contract.  Coverage is
120,600/120,840 station-days (`0.9980139026812314`) independently at each of
the registered 1-, 3-, and 7-day leads, and all 120 stations pass the frozen
0.90 coverage gate at every lead.  The full verifier found no water-temperature
or flow outcome access, no panel-artifact access, and no outcome-label field.

The admissible semantics are permanently narrow: each row is a retrospectively
retrieved fixed-valid-time-minus-lead daily air-temperature composite.  It is
not one coherent model initialization, an as-issued operational trajectory,
or a multi-variable forecast.  Any future recovery fraction must compare F2a
only with `F3_temperature_only` on an exact common-key registry; comparison
with `F3_full` is forbidden.

Why: acquisition completed after the v4 draft was written, so integrity,
coverage, source semantics, and the label-free access boundary had to be
verified before a common-key builder could consume the product.  The verifier
recomputed all chunk/response hashes, manifest and snapshot-index bindings,
schema and temporal identities, and the fixed-lead coverage inventory.  It
explicitly records `arm_promoted: false` and
`promotion_requires_separate_protocol_authorization: true`.  A second full
verification into an independent temporary directory reproduced the formal
report byte for byte, including SHA-256
`7d8a019d4966a941e9396a40930b2775ba95660347684701838721fe5032a60e`.

Evidence available before the decision:
`data_usgs/confirmatory_predictors/gfs-previous-runs-v1/manifest.json`,
`data_usgs/raw_snapshots/openmeteo-gfs-previous-runs-v1/snapshot_index.json`,
`scripts/data_usgs/verify_confirmatory_nwp.py`, and the create-only
`outputs/final/f2a_acquisition_verification_v1.json`.  Their SHA-256 values are,
respectively,
`54f85eef3ccd4b3cc69071f3c4ed7af5e3208274d3b9062a49c43c96ea43362c`,
`433f4b4885f225f5f9fe87ec9e94ba81493c07691bb4cc9a994be1931a406771`,
`2238015a5b1fb8b669df371096874f66d07aafa6d2718b4f97502ebe6bf149aa`,
and `7d8a019d4966a941e9396a40930b2775ba95660347684701838721fe5032a60e`.

Data periods already inspected: no model outcome was inspected by this
acquisition or verification step.  The atmospheric predictor dates are
2021-03-30 through 2023-12-31; the 2021-2023 water-temperature evaluation
domain and F3 oracle context had already been viewed separately.

Changes a primary hypothesis? no.  It verifies an input and its stopping gate
without computing a model score, contrast, event metric, or recovery fraction.

Requires a protocol version bump? no.  A new exact-byte v4 execution seal is
still required before F2a feature construction, training, scoring, or any
extension result can run.

Commit(s): (this worktree)

---

## 2026-08-09 — DLOG-024: score-independent v4 key-registry authority

Decision: Freeze the create-only Phase-1 primary, F2a-temperature, and
corrected as-of history-context registries at
`outputs/final/information_regime_key_registries_v4/`.  The formal manifest
status is `PHASE1_KEY_REGISTRIES_ONLY_NOT_MODEL_SCORE_AUTHORITY`.  This
decision freezes score-independent membership and context only: no model score
was read or accepted, no model output was published, and `arm_promoted`
remains `false` for F2a.

The primary registry reconstructs exactly 358,807 raw keys from 118 sites and
358,765 reportable keys from 116 sites.  Its 1-, 3-, and 7-day raw counts are
120,466, 119,654, and 118,687; its corresponding reportable counts are
120,444, 119,639, and 118,682.  The raw and reportable key-identity SHA-256
values are, respectively,
`6c4d26bb04884712ae07f7087a4e1782924a78e79e8066fc25e7ce795128c5aa`
and
`63a20255826c384a719d5287fa1563641b82df8b11b56d6a358084190a2487b4`.
This registry is the F0/F3_full base registry; it is not an F2b intersection.

The separate F2a-temperature registry contains 329,648 common keys from 117
sites and 329,628 reportable keys from 116 sites.  Its common counts at 1, 3,
and 7 days are 110,397, 109,857, and 109,394; its reportable counts are
110,385, 109,850, and 109,393.  The common and reportable key-identity SHA-256
values are, respectively,
`a71bab597c0d70504219ab0d3f497e00ab1736e847b170e08b6c68adfc11bd2b`
and
`bc30f2aa5f9101033fdade46454516dd172378a8522f9e4c66883d44d84be8a4`.
It matches the fixed-lead composite only to target-day
`F3_temperature_only`; `F3_full` is forbidden as its recovery denominator.

The corrected context uses the last finite water-temperature observation on
or before each issue date and never uses a future observation.  The raw
Hall/H75/H100 membership counts are 358,807/352,245/315,336, and the
reportable counts are 358,765/352,223/315,314.  The defective legacy context
columns were not consumed, and the authority-bound
`outputs/final/forecast_keys.parquet` was not overwritten.  These are frozen
stratum memberships, not a history-quality model result.

Why: the still-draft v4 design requires exact common keys, pre-score
reportability, and as-of context before any future model runner can be sealed.
The builder reconstructs the registries twice from canonical hash-bound
inputs, compares all seven serialized files byte for byte, validates their
schemas and semantic inventories, and publishes create-only.  This closes the
key-registry construction gate without opening the score, recovery, or
execution gates.

Evidence available before the decision:
`scripts/final/build_information_regime_key_registries_v4.py` and the seven
files under `outputs/final/information_regime_key_registries_v4/`.  The formal
manifest SHA-256 is
`ac0c256907264022e1fe7c4e407e0f95ece1f03233ecd6bb1841e27eb40b49ea`,
and the builder SHA-256 is
`ae22113b0011d4d0dec902f65f98df0ec45a47cd0842c07cdf0b83c9c0584b94`.
The production-byte claim is limited to the exact pinned runtime: CPython
3.11.7, NumPy 1.26.4, pandas 2.1.4, PyArrow/Arrow 14.0.2, GCC 11.2.0, and
x86_64 little-endian Linux.  The manifest discloses that both repository locks
target a different Python/pandas/PyArrow environment; cross-environment
Parquet byte identity is not claimed.

Data periods already inspected: the 2021–2023 key, target, acquired F2a, and
retrospective F3-temperature context required to construct these registries.
No model score, prediction, contrast, recovery fraction, or event metric was
read or computed by this step.

Changes a primary hypothesis? no.  This is a governance and data-contract
artifact, not an arm promotion, model result, or prospective preregistration.

Requires a protocol version bump? no.  The v4 document remains
`DRAFT_NOT_SEALED` with `execution_authorized: false`; a new exact-byte seal
binding the completed implementation is still required before any unrun model
feature construction, training, scoring, or result publication.

Commit(s): (this worktree)

---

## 2026-08-09 — DLOG-025: F/L tree results withdrawn for lost observedness lineage

Decision: Withdraw the current information-ladder and forcing-ladder tree-model
results from scientific use.  Preserve their files and hashes as forensic
records of a reproducible but invalid training lineage; do not delete,
overwrite, relabel, or use them as manuscript evidence.  This decision
supersedes the result-bearing parts of DLOG-018 through DLOG-022.  DLOG-018's
withdrawal of the older `+0.48 C`, `30x`, `27:1`, and full-budget claims still
stands; none of those values is restored.

The defect occurs before fitting.  The two exact input Parquets contain no
`*_observed` columns.  The legacy final runners concatenate them without
creating raw observedness flags, impute every variable, and then pass that
imputed panel both as the feature panel and as the purported true-label panel.
The feature builder consequently infers observedness from already-filled
values.  Imputed water-temperature targets enter training, missing issue-date
water temperatures pass admissibility, and history missingness features no
longer represent the raw record.  The forcing runner additionally uses the
last 2,000 rows of the same training table as its LightGBM evaluation set,
rather than a disjoint validation partition.

For the twelve forcing tree cells, the current versus correctly admissible
combined 2006--2017 row counts are 525,720 versus 423,266 at one day, 525,240
versus 420,673 at three days, and 524,280 versus 418,055 at seven days.  The
invalid unions are therefore 102,454, 104,567, and 106,225 rows.  Target values
were imputed in 100,174, 100,041, and 99,769 rows, respectively.  Even among
otherwise admissible rows, missingness features are wrong in 57,441, 56,619,
and 56,141 rows.  On the corrected 358,765-key evaluation registry, 51,723
keys have at least one erroneous history-missingness feature.  These errors
need not cancel between F0 and F3_full, between random-site and whole-region
geometry, or between L levels.

The withdrawn information descendants are all 432 files under
`outputs/final/ladder_shards/`, `ladder_effects.parquet`,
`ladder_summary.json`, and the formal `information_regime_v4/` and
`information_regime_inference_v4/` authorities.  Their manifest SHA-256 values
are `6984d561ceb537814837057707db726c20be5b569709ded27ebd2ddd15aef39b`
and `1bce4a92c0b90193a6c64edf26eaaa43f323e5335374c12dbf78cb82d3fa515c`.
The withdrawn forcing descendants are all twelve files under
`outputs/final/forcing_shards_v4/`, the v4 effects, contrasts, and summary,
and the `forcing_regime_v4_authority/` and
`forcing_regime_inference_v4/` authorities.  Their manifest SHA-256 values are
`260baa7e9fae08afb64280cb87fe320fc1712f4d2e0bd8b261ef824b133285b8`
and `580fed64c308267c1ed0c49e60dac8f21ff1657651de4ee70e3b179d6ed758d7`.
Downstream metric reconstruction and HUC2 resampling remain arithmetically
reproducible, but they operate on scientifically invalid predictions.

This defect does not invalidate the raw evaluation key identity, chronology,
or registry-valued `y_true`.  Across the 358,807 legacy evaluation keys, raw
issue and target water temperature are finite and the maximum panel-versus-
registry target difference remains
`1.5258789076710855e-6 C`.  The raw F3 future availability path is also
separate from the imputed history path.  The score-independent key-registry
authority from DLOG-024, the F2a acquisition verification from DLOG-023, and
the conventional Route-A authority are not withdrawn.  New work must use the
358,765-row `primary_reportable_key_registry_v4.parquet`, not the defective
legacy context metadata.

Why: two independent read-only implementations traced the exact production
bytes through loader, imputer, feature builder, training-row selection, and
formal authorities, then reproduced the affected row and mask inventories.
Existing tests exercised helper functions with hand-created observed flags but
never tested the production `raw -> flags -> train-only fit -> impute ->
feature` lifecycle.  The authority builders verified key, target, shard, and
downstream arithmetic consistency; they did not independently reconstruct the
training lineage.

Evidence available before the decision: the exact development and evaluation
panels (SHA-256
`0427a07ea4514ba29ce7d0cf89594e6c35c7f9134cc4d1d96fdc90daeaf5ba69`
and `cecdac459139456202240954e4c98fe18bba1fe8b63e9b06ab268683e0d1c03c`),
`scripts/final/run_information_ladder.py` (SHA-256
`8fa6b3c327df6dbb74f5e68f22dc0db861f4fc61cd46cab068bd6b79e206b49c`),
and `scripts/final/run_forcing_ladder_v4.py` (SHA-256
`6b370a4c1271e43f4797408bc1831a44882a7e3d727ccd71a485eb0cccac6eb8`).
The old authority directories listed above bind those exact defective source
and data bytes.

Data periods already inspected: 2006--2017 training/validation inputs and the
already-open 2021--2023 evaluation domain.  This audit read no unrun F2,
placebo, component, crossed, neural, L2_U2, L3, event, cohort, or audit-window
model outcome.

Changes a primary hypothesis? no.  It removes invalid evidence rather than
selecting a favorable result.  Every withdrawn F/L estimate, interval,
sign-flip tail, equal-HUC summary, LOCO range, P-5 component, and lead-order
interpretation must remain absent until a versioned observed-lineage rerun is
independently authorized.

Requires a protocol version bump? no for the withdrawal.  Corrected runs must
use new versioned runners and output paths, preserve raw flags before
imputation, bind pre-imputation training keys and labels, use disjoint
2006--2015 training and 2016--2017 validation, and issue new authorities.  The
draft v4 execution protocol remains unsealed and unauthorized.

Commit(s): (this worktree)

---

## 2026-08-09 — DLOG-026: observedness-lineage withdrawal authority frozen

Decision: Freeze the create-only, score-independent preprocessing-lineage
defect authority at
`outputs/final/preprocessing_lineage_defect_authority_v1/`.  Its status is
`SCIENTIFIC_RESULTS_WITHDRAWN_PENDING_OBSERVED_LINEAGE_RERUN`.  This authority
binds and independently reconstructs the adverse evidence underlying
DLOG-025; it is not a corrected model result, an execution authorization, or
a protocol seal.

The formal evaluation domain is the 358,765-row
`primary_reportable_key_registry_v4.parquet`, not the legacy 358,807-row key
file.  The authority reconstructs 17,478/17,231/17,014 keys with at least one
erroneous history-mask feature at 1/3/7 days, or 51,723 in total.  It retains
the legacy 51,765-key count separately as forensic chronology and identifies
the 42 legacy-only nonreportable keys.  The maximum raw-panel versus formal
`y_true` difference is `1.5258789076710855e-6 C`, within the fixed `2e-6 C`
tolerance.

The report's frozen claim token
`SCIENTIFIC_EVIDENCE_STATUS_ROWS_31_32_33_36_38` is a historical symbolic
identifier assigned before the new authority row was inserted into the status
document; it is not a live source-line pointer.  The withdrawn semantics are
the named L-subset, L2 geometry, L-by-geometry, F0 and F3 learned-result rows,
regardless of later Markdown line movement.  The published authority bytes
must not be rewritten to chase documentation line numbers.

The training reconstruction preserves DLOG-025's combined 2006--2017 counts:
525,720/423,266 current/admissible rows at one day,
525,240/420,673 at three days, and 524,280/418,055 at seven days.  It also
binds the exact old forcing source and records that its LightGBM evaluation
set was the last 2,000 rows of the same training table, rather than the
disjoint 2016--2017 validation partition required for a corrected rerun.

The published directory contains exactly the report and manifest JSON.  Their
SHA-256 values are
`5c1b1e05932538dba1b2a0d21ad44fdfe54a9c52e949b96b3a668356910e68d9`
and
`e69124409f49e4fb2aaaae319104251ca3078e535fca69121ed0eafddb23d908`.
The bound builder SHA-256 is
`03158b8b4ae23550b7501132bb0d148220617cd67f3cfda0805f7e9ae51a9fc7`;
the independently exercised test-file SHA-256 is
`971306c4264524fce04248f7e8776d1ba6810c9c3449b8694b41914502ecda97`.
The builder read zero prediction or score rows.  Two exact build passes,
strict production input/config/runtime pins, an anchored-directory
create-only publisher, staged byte verification and no-replace atomic rename
were independently verified before publication.  A post-publication check
then confirmed the exact two-file set and both published digests.

The withdrawal and preservation boundary is unchanged.  All old F/L learned
predictions and descendants remain scientifically withdrawn; raw key identity
and `y_true`, the DLOG-024 score-independent registries, the DLOG-023 F2a
acquisition verification, conventional Route-A, and the separate air2stream
raw path remain outside this specific defect.  No old numerical F/L claim is
restored, and no new model score is introduced.

Why: DLOG-025 recorded the adverse decision before this evidence artifact was
built.  A separate append-only entry is therefore required to freeze the
later create-only authority without implying that it existed at the time of
the original withdrawal.

Evidence available before the decision: the exact inputs and sources bound in
the manifest, two independent attack reviews of the builder and publisher,
17 focused tests, and two production reconstruction passes.

Data periods already inspected: 2006--2017 training/validation inputs and the
already-open 2021--2023 evaluation domain.  No unrun extension outcome and no
model prediction or score was read.

Changes a primary hypothesis? no.  This freezes adverse lineage evidence and
keeps every affected result withdrawn.

Requires a protocol version bump? no.  Any corrected model execution still
requires a versioned runner, independently authorized score-execution
protocol, source registry and new result authority; the v4 draft remains
unsealed and `execution_authorized: false`.

Commit(s): (this worktree)

---

## 2026-08-09 — DLOG-027: F2b remains planned-degraded; primary scope is 48 F0/F3 cells

Decision: Close the current F2b go/no-go as **NO-GO** and retain F2b as
`PLANNED_DEGRADED`.  Until a separately frozen archive pilot passes every
upgrade gate, the intended primary crossed design is the 48-cell F0/F3_full
subset rather than a completed 72-cell F0/F2b/F3_full matrix.  F2a remains a
separate retrospective fixed-lead, temperature-only diagnostic and cannot be
substituted for F2b.

Why: A read-only official-source investigation found that the NOAA GFS NODD
bucket currently exposes coherent 0.25-degree trajectories for sampled old
initializations.  The 2021-03-30 00Z and 2023-12-31 sampled directories contain
the expected three-hour f000 through f192 sequence without a missing forecast
hour.  This establishes a plausible acquisition route, not an execution-ready
archive.  The repository still lacks all of the following required evidence:

1. a frozen UTC issuance cutoff for each date-only issue key;
2. a complete 2021--2023 issue-day and required-variable inventory;
3. publication/revision-history evidence for objects backfilled after their
   nominal initialization date, including pre-2021-02-26 examples whose S3
   `Last-Modified` metadata is in 2024;
4. archived GRIB/message bytes, HTTP metadata, ETags and SHA-256 checksums; and
5. a score-independent common-key authority proving at least 0.90 coverage.

The official evidence checked was the NCEI GFS product description
(`https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast`),
the NCEP/NCO GFS product inventory
(`https://www.nco.ncep.noaa.gov/pmb/products/gfs/`), the NOAA GFS NODD registry
(`https://registry.opendata.aws/noaa-gfs-bdp-pds/`), NCEI THREDDS old-month
catalogs, and the NCEI HAS Grid-004 request interface.  NODD accessibility does
not by itself attest contemporaneous availability or immutable as-issued
vintage semantics.

Permitted manuscript wording after the 48-cell subset is actually complete:

> The archived-vintage F2b arm remained `PLANNED_DEGRADED` because an
> issue-time-complete, coherent, publication-time-bound, and checksum-frozen
> GFS archive was not demonstrated for the 2021--2023 evaluation keys. We
> therefore report the 48-cell F0/F3_full subset, not a completed 72-cell
> primary matrix. F2a remains a separate retrospective fixed-lead,
> temperature-only diagnostic and was not substituted for F2b. Consequently,
> we make no as-issued, operational-replay, or archived-forecast recovery claim.

Evidence available before the decision: only label-free source documentation,
directory listings and HTTP metadata.  No model, prediction, score, target
outcome or F2b-derived feature was read or computed.

Data periods already inspected: object inventories for sampled 2021 and 2023
initializations.  This was an archive-availability probe, not a model-result
inspection and not the full issue-key coverage audit.

Changes a primary hypothesis? no.  This applies the v4 F2b degradation rule;
it does not select a favorable result.

Requires a protocol version bump? no for the NO-GO.  Any future reopening must
be a separately frozen acquisition amendment made before feature extraction or
model scoring.

Commit(s): (this worktree)

---

## 2026-08-09 — DLOG-028: score-free semantic registries v4 authority frozen

Decision: Promote only the semantic data and contract registries to a
create-only Tier-1 authority. The canonical 11-file directory is
`outputs/final/semantic_registries_v4_authority/`; its authority-manifest
SHA-256 is
`12cdc355a06d2c39733a60386dfdeb8a2f6b234d2f8d6f641a995a3d3c41072c`.
The authority remains explicitly non-executable and is not a forcing-protocol
seal or model-result authority.

Why: The production publisher rebuilt the semantic-data candidate and the
semantic-contract candidate twice under the pinned Route-A Python 3.12
environment, verified exact serialized bytes and cross-bindings, and committed
the staged directory with create-only `RENAME_NOREPLACE` semantics. The
authority binds two Parquet data registries, six JSON contract registries, two
candidate manifests, the authority manifest, all relevant builders and tests,
source inputs and runtime receipts. The primary state inventory is exactly 72
logical cells: 48 `PLANNED`, 24 `REGISTERED`, zero `EXECUTED` and zero
`WITHDRAWN`. The separate 432 legacy logical cells remain forensic
`WITHDRAWN`.

Evidence available before the decision: score-independent raw panels, frozen
key and defect authorities, protocol/status documents, registry builders and
tests. The focused semantic test set passed 60 tests before publication. No
model checkpoint, prediction, score, effect, contrast result or runner output
was read or accepted.

Data periods already inspected: semantic inputs cover the previously declared
2006--2015 training and 2016--2017 validation periods; source authorities also
describe the already-open 2021--2023 evaluation domain. This publication adds
no outcome-derived model evidence.

Changes a primary hypothesis? no. It freezes data meanings and the current
planned/registered inventory only.

Requires a protocol version bump? no. The forcing-specific protocol candidate,
clean design commit, complete source registry and terminal seal remain separate
prerequisites. `execution_authorized` remains false throughout this authority.

Commit(s): (this worktree)

---

## 2026-08-10 — DLOG-027: forcing-v5 inference authority, claim-status quarantine, and manuscript scope correction

Decision: three linked actions, recorded together because each one depends on
the others.

**1. Publish the forcing-v5 inference authority.** The v5 point authority
reserved `outputs/final/forcing_regime_v5_observed_inference_authority_v1/` for
interval evidence and published none. That directory now holds a create-only
bundle built by `scripts/final/build_forcing_v5_inference_authority.py` from
the 696 published station-paired effects and the twelve key-level shards. It
adds, per model and horizon: a 10,000-draw whole-HUC2 cluster bootstrap
interval, an exact sign-flip tail over all 2^15 whole-cluster sign vectors, the
equal-HUC aggregate, the leave-one-HUC2-out range, year and season strata
recomputed from the shards rather than reweighted, station-level lead
interactions and a model contrast formed as double differences before any
median, and a minimum detectable effect obtained by translating the effect
vector until the sign-flip tail crosses 0.05.

The station-first forcing value is 0.130/0.542/0.627 degC at 1/3/7 days for
LightGBM and 0.125/0.578/0.605 for ResidualLightGBM. Every bootstrap interval
excludes zero, every leave-one-HUC2-out range keeps its sign, the equal-HUC
aggregate is slightly larger than the station-weighted median in all six cells,
and every effect is three to four times its own minimum detectable effect. The
lead interaction is +0.415 degC [0.301, 0.541] from one to three days against
+0.074 degC [0.046, 0.101] from three to seven, so the value largely plateaus
after day three. The model contrast is at most 0.026 degC, so the result is not
an artifact of the raw-versus-residual target formulation. All 42 year and
season strata are consistently signed; the seasonal maximum is spring
(0.67 degC at three days) and the minimum is summer (0.40 degC).

The builder independently recomputes every station effect from the shards and
fails closed if it drifts from the published point authority by more than 1e-9,
or if F0 and F3 disagree on `y_true` by more than the frozen 2e-6 degC. Both
checks passed. The authority declares `post_outcome: true`,
`confirmatory: false`, `prospectively_registered: false`; it does not authorize
execution of anything and does not promote F2a, F2b, or any L-axis arm.

**2. Make the Phase-0 quarantine mechanical.** Every entry in
`paper/claim_ledger.yaml` now carries `status:` from
`PRIMARY_FROZEN | DESCRIPTIVE_PROVISIONAL | NOT_USED`, and
`scripts/final/check_manuscript_consistency.py` gained three gates:
`check_claim_status` (a provisional claim may not declare a headline span),
`check_provisional_not_in_headlines` (a provisional value may not be *printed*
in the Abstract, Key Points, or Conclusions even if the ledger does not declare
it there), and `check_quarantined_content` (withdrawn headline numbers and
protocol-forbidden assertions, scanned line by line so a chronology entry that
records a withdrawal is distinguishable from a sentence restating it).

The second gate immediately caught a live defect: the outcome-conditioned
`actual_rapid_warming` value of -0.300 degC was a Key Point and an Abstract
sentence while Section 4.6 correctly labelled it retrospective. Six injection
tests (INJ11-INJ16) fix the gates' behaviour.

**3. Correct the manuscript's scope.** Removed from the Abstract, Key Points
and Conclusions: the outcome-conditioned rapid-warming headline; the +0.20 degC
pooled-adaptation effect, whose arm carries the DLOG-012 fold-0 preprocessing
defect and which the manuscript itself had labelled a lower bound; and the
claim that the whole-region geometry penalty is a small confirmed effect. The
geometry penalty (0.004-0.009 degC) is the size of its own five-seed
resampling spread and is now reported as below the design's resolution, with
the structural reason stated: every arm retains the held station's own thermal
history, so the component that must transfer is worth about 0.07 degC at seven
days and a geometry effect has to be found inside it.

Section 4.7 reports the forcing result with its inference, explicitly labelled
post-outcome and oracle-bounded, and deliberately not in the Abstract or Key
Points. The Introduction and Discussion were rewritten against the
benchmarking literature they had omitted (Klemes 1986; Seibert 2001; Schaefli
and Gupta 2007; Andreassian et al. 2009; Pappenberger et al. 2015; Seibert et
al. 2018; Knoben et al. 2019; Nearing et al. 2018, 2021; and the PUB and
large-sample lines), and the untested assertion that re-scoring published
models would absorb the between-study dispersion is now stated as a question
this study motivates rather than answers.

Why: the v5 point authority produced the project's first valid corrected
learned result, and the largest risk at that moment was reporting it inside a
manuscript whose Abstract still carried two numbers that the project's own
records had disqualified. Making the quarantine a gate rather than an editorial
intention is what prevents the third recurrence.

Evidence available before the decision: the v5 point authority and its
manifest; the twelve v5 shards; the frozen 15-cluster HUC2 map; the existing
`thermoroute.significance` estimators, reused rather than reimplemented.

Data periods already inspected: 2006-2017 training/validation inputs and the
already-open 2021-2023 evaluation window. No unrun placebo, component, F2, L2,
L3, neural, crossed, event, cohort, or audit-window outcome was read.

Changes a primary hypothesis? No. It adds interval evidence to an existing
point result and removes inadmissible claims from the manuscript.

Requires a protocol version bump? Not for the inference authority, which is
downstream of the already-published point authority. The placebo and component
controls are specified in the new
`protocols/wrr_forcing_placebo_protocol_v5a.yaml`, drafted but unsealed and
unauthorized; sealing it before the first placebo fit is the only way those
arms can ever be described as prospectively specified.

Commit(s): (this worktree)

### DLOG-027 addendum: the governance pins cannot survive their own protocol

Recording this entry exposed a circular dependency in the v5 execution
authority. `run_forcing_ladder_v5_observed.py` pins
`docs/strong_accept_decision_log.md`, `docs/SCIENTIFIC_EVIDENCE_STATUS.md` and
`docs/WRR_INFORMATION_REGIME_TODO_20260809.md` by exact SHA-256 and refuses to
run when any of them drifts. All three are documents the protocol *requires*
to be updated: the decision log is append-only by design. Every future DLOG
entry therefore breaks the runner, and the break is indistinguishable from
tampering.

Two consequences were already visible before this entry. Commit `0ace001`
updated the tracker's T05 row to `COMPLETE` after the seal and clean design
commit landed, which silently removed a literal token the runner pins; twenty
tests in `tests/final/test_forcing_v5_execution_authority.py` had been failing
at collection ever since, on a runner that can no longer be executed. That
token has been corrected to describe the current true state, and the two
protocol-payload byte pins that move with it (length 8,663 to 8,664, digest
`0d4d97a2...` to `2a7a8ccc...`) were recomputed. The three governance document
digests are refreshed below.

Refreshing forward-looking pins does not rewrite history: the completed v5 run
recorded what it bound at execution time in its own lineage manifest, and that
record is untouched.

The design should be changed rather than re-pinned every time. A content hash
is the wrong instrument for an append-only log. The pin should bind either the
log's *prefix* through a named entry, or the specific frozen artifacts the run
depended on, and leave the narrative documents to a token check that states
what must be present rather than what the bytes must be. Recorded as an open
item; it is not fixed here, because redesigning the pin semantics is a change
to the sealed execution path and belongs with the next protocol version.

---

## 2026-08-10 — DLOG-028: shuffled-forcing placebo executed under a pre-outcome seal; P1 satisfied

Decision: the F0/F3_full forcing value is reported as **event-scale weather
information** rather than seasonal-phase information. Sealed decision rule P1
is satisfied at every model and lead.

Chronology, which is what makes this entry different from every other result in
this project. Protocol v5a was sealed at commit `88d7578` on a clean tree with
no placebo output directory present; the sealer refuses to write when one
exists, so the claim that the decision rules preceded the outcome is checkable
rather than asserted. The seal binds the protocol bytes (`58183acf...`) and a
canonical digest of the decision-rule, estimand, arm and held-fixed blocks
(`f60a6e3d...`), fixing thresholds P1 < 25%, P2 25-60%, P3 > 60% before any
shuffled fit existed. P3, under which the forcing value would have been
withdrawn as an information claim, was reachable and is covered by a test.
**This is the first result in the project whose interpretation rule was frozen
before its outcome.**

Result. Retention of the true forcing value by the placebo: 0.0%, 2.0% and 8.8%
at 1, 3 and 7 days for the raw-target tree; -0.5%, 1.7% and 8.4% for the
residual-target tree. The station-level placebo-minus-true contrast is +0.129,
+0.525 and +0.577 degC, every interval excludes zero, every leave-one-HUC2-out
range keeps its sign, and the placebo is worse at 91-95% of stations.

Construction. `RawFutureRegistry` means "the exact realized future", so it was
not filled with permuted values. The derangement is applied upstream to the
meteorology columns of the raw panel the future lookup reads, and the registry
is then built through the ordinary public factory, so every v5 validator runs
unchanged on the placebo and no private constructor is touched. Preprocessing,
imputer, climatologies, damped anchor and the entire base feature table come
from the true panel; future forcing is the only thing that moves. The stratum
is the specific year-month, the stricter of the two readings the sealed
`[site_id, target_month]` admits.

What the arm does and does not settle. A within-month derangement preserves
each station-month mean exactly, so it removes within-month day-to-day
correspondence specifically, not "future weather" wholesale. The residue rising
monotonically with lead (0% to 9%) is the month-level component, and it grows
because the anchor has decayed further at longer leads. The arm does not
separate day-to-day correspondence from sub-monthly synoptic persistence, since
a donor from the same month can fall within a few days of the true valid time;
the +/-7-day shift arm is the control for that and has not run. It also
attributes nothing to individual meteorological variables; the component arms
are specified in the same seal and have not run.

Status boundary. The underlying F0/F3 contrast remains post-outcome and
descriptive: sealing a control before its own outcome does not make the
already-inspected reference result confirmatory. Section 4.7 stays out of the
Abstract and the Key Points. No ratio against an architecture, local-information
or geometry effect is admissible until the crossed matrix exists.

Engineering note. The first execution completed all thirty fits and then raised
TypeError on `transaction.commit(expected_files)`, which requires a keyword-only
`precommit_check`. The transaction rolled the bundle back, so nothing was
published and no partial arm could be mistaken for a result, but forty minutes
of compute was lost. The callback now re-verifies thirty shards, the manifest
cell count, the unchanged true panels and preprocessing record, and the seal's
bound inputs; two tests guard the call site. This is the same shape of mistake
as DLOG-025 -- verifying artifacts and authorizations while leaving the actual
call path unexercised -- and the three-line `inspect.signature` assertion that
now prevents it should have existed before the first run.

Evidence available before the decision: the sealed protocol and its seal, the
v5 point and inference authorities, the twelve reference shards, the thirty
placebo shards and their lineage manifest.

Data periods already inspected: 2006-2017 training/validation inputs and the
already-open 2021-2023 window. No shift, component, F2, L2, L3, neural,
crossed, event, cohort or audit-window outcome was read.

Changes a primary hypothesis? No. It supplies the control that the forcing
result required, and the control was passed.

Requires a protocol version bump? No. The shift and component arms remain
specified under the same v5a seal and unrun.

Commit(s): (this worktree)

---

## 2026-08-10 — DLOG-029: shift arms executed under the same seal; P4 satisfied

Decision: the F0/F3_full forcing value is reported as **exact-event-timing**
weather information. Sealed decision rule P4 is satisfied.

The shuffle arm (DLOG-028) removed within-month day-to-day correspondence but
could not separate that from sub-monthly synoptic persistence, because a
same-month donor can land within a few days of the true valid time. The +/-7-day
arms displace the realized future by a whole week, leaving climate,
near-seasonal phase and local weather persistence intact.

Result. The +7-day arm, the primary control P4 is stated in, retains -0.1%,
0.7% and 14.0% of the true forcing value at 1, 3 and 7 days for the raw-target
tree and -0.5%, 1.0% and 13.8% for the residual-target tree, and is worse than
the true arm at 90-95% of stations. The -7-day arm retains essentially nothing
at any lead. The two are reported separately and never averaged; the minus arm
is the weaker control because it moves the future window toward information
already available at issue time.

The two residues are coherent. At seven days the within-month shuffle leaves
8.8% and the week displacement 14.0%. Displacement preserves more because
synoptic weather is autocorrelated over several days, so the shifted window
still resembles the true one; a shuffle within a month does not. That ordering
is what a weather-information reading predicts and would be hard to obtain from
a seasonal-phase artifact.

Boundary rule. Displacement leaves seven days at one end of each station's
record without a donor; those keys took a climatology substitution and were
counted per key by the runner. The sealed rule is enforced in the scoring
authority: such keys are dropped from both shift arms *and* from the true arm,
so all three share one common shift-registry, named separately and never mixed
with the primary 358,765-key registry. A test asserts the common registry is
strictly smaller than the primary one, so a silently unapplied boundary rule
cannot pass.

Chronology. Both arms run under the same specification seal as the shuffle,
written at commit 88d7578 before any placebo outcome existed. The underlying
F0/F3 contrast remains post-outcome and descriptive; Section 4.7 stays out of
the Abstract and the Key Points.

Still open under the same seal: the four single-component and four
leave-one-out arms, which would attribute the value to individual
meteorological variables. Nothing here attributes it.

Evidence available before the decision: the sealed protocol and seal, the v5
point and inference authorities, the twelve reference shards, the twelve shift
shards and their lineage manifest.

Data periods already inspected: 2006-2017 inputs and the already-open 2021-2023
window. No component, F2, L2, L3, neural, crossed, cohort or audit-window
outcome was read.

Changes a primary hypothesis? No. It supplies the timing control the forcing
result required, and the control was passed.

Requires a protocol version bump? No.

Commit(s): (this worktree)

---

## 2026-08-11 — DLOG-030: corrected L ladder executed; local thermal state dominates

Decision: the 432-cell ladder withdrawn by DLOG-025 is replaced. Local thermal
state is worth 1.3-1.7 degC under whole-region holdout, and almost all of it is
the recent water-temperature sequence rather than the station's long-term
statistics.

Result, raw-target tree, 116 reportable stations, station-first paired medians
with whole-HUC2 bootstrap intervals:

  L1 - L0      own long-term statistics   0.010 / 0.062 / 0.161 degC at 1/3/7 d
  L2 - L1      recent temperature sequence 1.707 / 1.201 / 1.169
  L2 - L0      all local thermal state     1.716 / 1.255 / 1.325
  L2_U2 - L2   local discharge             0.084 / 0.043 / 0.009

Every L2-L0 interval excludes zero, all 116 stations are worse at every lead,
and every leave-one-HUC2-out range stays far from zero. Every discharge
interval covers zero and stations split about evenly.

This reorders the paper. Local thermal state is two to thirteen times the
realized-future-meteorology value, roughly seventy times the architecture
effect, and two orders of magnitude above the geometry penalty. The three
quantities come from separate conditional designs and are not added; the
ordering is what matters.

Two further readings. Cold-starting a *gauged* site is nearly free -- pooling a
station's climatology and damped rate costs 0.01-0.16 degC -- while thermally
ungauged prediction is a different problem, at 1.17-1.71 degC. And discharge
does not substitute for thermal history: once water temperature is gone,
removing discharge as well changes almost nothing, so the "hydrology observed"
rung is barely distinguishable from having no local observation at all.

Lineage, which is the point of the rebuild. Prohibited inputs are dropped from
the design matrix rather than filled, so invariance holds by construction. A
proof runs immediately before each cell is fitted, perturbing the held
stations' own observations and requiring a bit-identical design matrix: all 24
proofs return exactly zero, and a negative control that re-admits a single
water-temperature lag is caught with a 147 degC shift. Three guards the old
ladder lacked fail closed: no imputed issue or target label may enter training
or validation, train and validation identities may not overlap, and no
held-region station may reach the training set. Every statistic a level
constrains -- imputer, climatology, damped rate -- is refitted on in-fold
stations only.

Two implementation facts recorded because both could have degraded the result
silently. The evaluation registry covers 116 reportable stations, not the
120-station cohort, and asserting 120 initially failed closed rather than
quietly scoring a subset. And bind_exact_evaluation_rows only accepts the
complete frozen namespace because that validator enforces two-sided equality
with the formal key registry, so the binding uses the full column set and the
level's subset is applied at fit time; passing the reduced set would have
traded away the key-registry check without any visible symptom.

Scope. Whole-region geometry only. The random-site arm and therefore the L-by-G
interaction are not yet run, so no statement is made here about how the local
information value depends on spatial geometry. Post-outcome and descriptive
like every other 2021-2023 result; the F and L axes remain separate conditional
designs and are never added or divided.

Evidence available before the decision: the 96 create-only ladder shards, their
lineage manifest with the per-cell mask-invariance proof, and the frozen key
registry.

Data periods already inspected: 2006-2017 inputs and the already-open 2021-2023
window. No component, F2, neural, crossed, cohort or audit-window outcome read.

Changes a primary hypothesis? It replaces withdrawn evidence and reorders which
information source the manuscript treats as dominant.

Requires a protocol version bump? No.

Commit(s): (this worktree)

---

## 2026-08-11 — DLOG-031: L-by-G interaction and component attribution

Two results, both from arms that are now complete.

**Spatial geometry matters about thirteen times more once local thermal history
is gone.** Running all four information levels under random-site holdout as
well as whole-region, with the station contrast formed inside each of five
split seeds and the paired contrasts then averaged per station, gives a
geometry penalty of 0.006-0.014 degC at L0 and 0.073-0.129 degC at L2. The
station-level double difference is +0.066 to +0.124 degC; every whole-HUC2
interval excludes zero at every model and lead and every leave-one-HUC2-out
range keeps its sign.

This reconciles Section 4.5 with the transfer literature rather than
contradicting it. The cost of substituting a random split for a regional one is
invisible while the held gauge still supplies its own thermal history, because
the model barely needs its neighbours. Remove the history and the neighbours
begin to matter. The sensitivity of a reported spatial result to the split
design is therefore itself conditional on how much local information the model
retains, which is not something a single-geometry study can discover.

**The forcing value is future air temperature.** At seven days air temperature
alone recovers 0.595 of the 0.627 degC full value (95%), while withholding air
temperature and giving every other variable its realized future retains only
0.129 degC (21%). Withholding radiation, precipitation or humidity-and-wind
instead costs nothing measurable: each returns 0.62 degC. Air temperature alone
recovers 87% and 92% at one and three days.

Reporting both families is what makes this readable. Radiation and
humidity-and-wind are individually informative -- 0.105 and 0.117 degC on their
own -- yet entirely substitutable, which is what a correlated predictor looks
like while the variable it tracks remains available. Air temperature is the
only variable both sufficient alone and not replaceable. Shares sum well past
100% and are not a variance decomposition.

Method notes. Random-site aggregation is seed-first throughout: the contrast is
formed inside each seed and the five paired contrasts averaged per station, not
the five risks averaged and differenced afterwards, which is a different
quantity whenever seeds disagree about which stations are hard. A component arm
gives the unselected variables their training climatology rather than dropping
their columns, so the feature namespace is identical across arms and "which
variable matters" is not confounded with "how many columns the model has".

Engineering note. Every fit is pinned to one LightGBM thread for determinism
and the runners were serial, so a 128-core machine was executing one experiment
at roughly two cores; the random-site ladder projected to 7.4 hours. Sharding
by (seed, level) across ten workers with per-worker lineage manifests cut it to
under an hour without touching what any single fit computes. While doing this I
misread `pgrep -f "a\|b"` -- ERE, so the alternation is literal -- concluded the
workers had died, and launched four duplicates that could have raced on the
same shard paths. They were killed within ninety seconds and all 75 shards
present at that moment were re-read and verified intact; no shard was corrupted.

Scope. Whole-region and random-site geometries at F0 only. The F-by-L crossing,
the plain-TCN architecture arm and the 48-cell matrix remain unrun, so no
interaction between forcing and either local information or architecture is
claimed. Post-outcome and descriptive; the axes are never added or divided.

Commit(s): (this worktree)

## 2026-08-12 — DLOG-032: what a real forecast recovers, and whether weather substitutes for a gauge

Two arms that were computable from work already on disk but had never been
scored into authorities, plus the manuscript text they support.

**F2a: about half the temperature oracle survives a real forecast.** The
recovery fraction is `median_i[R_i(F0) - R_i(F2a)]` over
`median_i[R_i(F0) - R_i(F3_temperature_only)]`, and it comes out at 49%, 62%
and 50% at 1, 3 and 7 days for the raw-target tree (49%, 60%, 58% residual).
The denominator is the realized-temperature arm and never `F3_full`: F2a is a
temperature-only product, and charging it for four variables it never claimed
to supply would be arithmetic, not measurement. The component result of
DLOG-031 is what makes the concession small -- air temperature alone already
carries 95% of the full oracle at seven days.

No interval is reported on the ratio. Both differences carry whole-HUC2 cluster
intervals; a quotient of two estimated medians does not, and a delta-method
band on it would be an interval in name only. The artifact says so in a
`forbidden` list next to "operational recovery fraction" and "recovery against
F3_full", so the constraint travels with the number rather than living only in
prose.

The training error is recorded because it is the natural implementation and it
is wrong. The forecast archive covers 2021-2023, so a model trained on F2a
features sees a future-temperature column that is climatology on every training
row, learns to ignore it, and is unaffected when real forecasts are substituted
at evaluation. That run reported 1.8% recovery -- a measurement of the mistake.
Training on realized temperature and substituting the forecast only at
prediction is both the fix and what an operational system does.

**F-by-L: future weather does not substitute for a local gauge.** Crossing the
two axes that Sections 4.7 and 4.8 varied separately answers a question the
paper was inviting and had not asked. Forcing value at L0 is 0.116/0.474/0.580
degC at 1/3/7 days; at L2 it is 0.038/0.331/0.596. The station-level double
difference is -0.076 [-0.098, -0.056], -0.147 [-0.193, -0.079] and
+0.016 [-0.035, +0.099].

The sign is the informative part and it is the unfavourable one. Realized
future weather is worth *less* to a thermally ungauged model, not more, at the
leads where local state dominates -- it earns its value by correcting a
trajectory, and with the persistence anchor gone the fallback climatology is
too coarse to sharpen. The two information sources are complements, so an
ungauged reach loses 1.3-1.7 degC to the missing gauge and then recovers less
from forecasts than a gauged site would. Only at seven days do the levels
converge, and that is also the lead where the archived forecast recovers just
half the oracle: the regime where forcing substitutes best is the regime where
it is least attainable.

Scope discipline. Geometry is pinned to whole_region: a three-way contrast is
not something 116 stations across about nine effective clusters can carry, and
the L-by-G interaction already has its own authority. Both analyses are
post-outcome and descriptive, are excluded from the Abstract and Key Points on
that ground, and the axes are never added or divided.

Artifacts: `outputs/final/f2a_recovery_authority_v1/`,
`outputs/final/forcing_information_interaction_v1/`.
Builders: `scripts/final/build_f2a_recovery_authority.py`,
`scripts/final/build_forcing_information_interaction.py`.

Commit(s): (this worktree)

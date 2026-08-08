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

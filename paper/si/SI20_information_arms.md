# SI20 — the information arms: local state, future forcing, geometry, architecture

**Status:** filled. Every value here is bound to a scored authority under
`outputs/final/`. Every result in this section is **post-outcome and
descriptive**: the 2021–2023 window was already open and earlier results had
been inspected when these arms were specified. They are not confirmatory tests,
and the manuscript reports them as descriptive in the Abstract and Key Points
for that reason.

This section exists because manuscript Sections 4.7–4.10 are the largest effects
the study measured and, until it was written, cited artifact paths and nothing
else. A reader could reproduce them and could not read the method anywhere.

## 1. What each arm varies

Four axes are crossed on one frozen panel, one common key registry, and one
evaluation set. Each axis withholds or supplies information; none changes the
estimand, the keys, or the anchor.

| Axis | Levels | What changes |
|---|---|---|
| **L** local information | L0, L1, L2, L2-U2 | which of the target site's own observations the model may use |
| **F** future forcing | F0, F3_full, and the component and placebo arms | whether the model sees meteorology after the issue time |
| **G** geometry | whole-region, random-site | how the holdout is drawn |
| **A** architecture | residual LightGBM, plain causal TCN | the model class, information held fixed |

`L1` pools the climatology and damped rate over training stations; `L2` removes
every target-site water-temperature input, including the persistence anchor and
the climatology anomaly; `L2-U2` additionally removes discharge. Two column
families are water-temperature-derived despite their names — `clim_anom` and
`persistence` — and both are withheld at L2. This is the defect that destroyed
an earlier ladder: a level that leaves either of them in is not the level it
claims to be.

## 2. Masking is proved, not asserted

A level that hides an input must produce a **bit-identical design matrix** when
that input is perturbed. Before each cell is fitted, the held stations' hidden
observations are replaced with noise and the design matrix is rebuilt; the two
must agree exactly.

* All 24 proofs return a maximum absolute difference of exactly zero.
* A negative control that re-admits a single water-temperature lag is caught
  with a 147 °C shift, so the proof is not vacuous.

Prohibited inputs are **dropped, not filled**. Filling would leave the mask
dependent on the fill value; dropping makes the invariance hold by construction.

Authority: `outputs/final/information_ladder_v6_observed/`, runner
`scripts/final/run_information_ladder_v6_observed.py`.

## 3. Estimands

Every contrast is station-first and paired, formed per station before any median
is taken:

```text
effect = median_i [ RMSE_i(candidate) − RMSE_i(reference) ]
```

A difference of two marginal medians is a different quantity and is never
reported. Interactions are station-level double differences formed the same way,
and the three-way contrast of §4.10 is a difference of two double differences.

Random-site aggregation is **seed-first**: the station contrast is formed inside
each of the five split seeds, against the model fitted on that seed's own folds,
and only then averaged. Pooling risks across seeds first would compare arms that
were never held out on the same stations.

Intervals are 10,000-draw whole-HUC2 cluster bootstraps; sign-flip tests
enumerate all 2^15 whole-cluster sign vectors exactly; leave-one-HUC2-out ranges
accompany every interaction.

## 4. Minimum detectable effect, and the two kinds of null

Several cells report a null, and they do not all mean the same thing. Each
interaction carries a minimum detectable effect: the observed vector is centred
and shifted until the whole-cluster sign-flip tail clears α = 0.05, which makes
the number a property of the dependence structure rather than of the effect that
happens to be present.

* At L0 the MDE is 0.003–0.005 °C, so a null there is **evidence of absence**.
* At L2 it is 0.064–0.190 °C, so a null there is **absence of evidence**.

The forty-fold gap is the result restated rather than a property of the design:
at L0 every station's effect sits against zero with almost no spread and the
sign-flip test is powerful; at L2 the effect varies widely across stations and
the same test on the same fifteen regions is not.

The artifact field is named `magnitude_at_or_above_mde`, deliberately not
`resolvable`. It compares an observed magnitude against what the cohort could
detect, which is a statement about power and not about the effect: at L2 and
seven days a magnitude clears its own MDE while its interval covers zero.

## 5. Pre-outcome sealing of the forcing controls

The two placebo arms and the component arms were specified, and their decision
rules fixed, before their outcomes existed. The protocol bytes and the decision
rules P1–P4 are sealed in
`protocols/wrr_forcing_placebo_protocol_v5a.yaml` with
`protocols/wrr_forcing_placebo_protocol_v5a_seal.json`.

* **Shuffle placebo** — the meteorology vector is deranged across dates *within
  each station-month*, so climate, seasonal phase and month-level conditions are
  preserved exactly and only the day-to-day correspondence is destroyed.
* **Shift placebo** — the realized future is displaced by a whole week in each
  direction, which leaves climate, near-seasonal phase and weather persistence
  intact and removes only the exact dates. The two directions are reported
  separately and never averaged; displacing backwards moves the window toward
  information already available at issue time and is the weaker control.
* **Component arms** — two families, `only_X` (X realized, everything else
  climatological) and `without_X` (the reverse). Single-variable values do not
  sum to the full forcing value and the shares exceed 100%; this is not a
  variance decomposition, because the meteorological variables are correlated.

## 6. What the F2a forecast arm is, and is not

F2a substitutes an archived fixed-lead air-temperature forecast for the realized
value and changes nothing else. Three constraints travel with it:

* it is a **fixed-lead composite** — for each valid time, what a run issued *h*
  days earlier said — not a coherent trajectory from one initialization, and not
  an as-issued operational archive;
* the archive covers **2021–2023** at these stations, so the recovery fraction
  measures those years and is not a climatological expectation;
* its only legal denominator is `F3_temperature_only`, never `F3_full`. A
  temperature-only product must not be charged for four variables it never
  claimed to supply.

The arm trains on **realized** air temperature and substitutes the forecast only
at prediction. Training on F2a features directly is wrong and was done once: the
archive is 2021–2023 only, so a model trained on them sees a future-temperature
column that is climatology on every training row, learns to ignore it, and is
unaffected when real forecasts arrive. That run reported 1.8% recovery, which
measured the mistake rather than the product.

No confidence interval is reported on the recovery fraction. Both differences
carry whole-HUC2 intervals; a quotient of two estimated medians does not, and a
delta-method band on it would be an interval in name only.

## 7. The architecture arm's information matching

The plain causal TCN consumes the frozen tabular namespace the trees consume,
under the same level mask, on the same folds and keys, adding its residual to
the same level-legal anchor. Matching is structural: the namespace is already a
lag window, so its lag columns become the sequence axis and every remaining
column enters as a static feature, and a test requires that partition to be
exhaustive at every level.

The dilations span the whole eight-position lag grid. A stack reaching back only
seven positions would be causal *and* blind to the oldest observation a tree can
split on, which is an information mismatch dressed up as an architecture.

Three concessions are recorded, ordered by how much each binds:

1. **Training budget.** A fixed budget is not a matched budget: LightGBM stops at
   its own best iteration, the network stops when the clock does. Under the
   original 40-epoch budget the median best epoch was 9–23 at L0 and 37 at L2,
   with a third of L2 cells still improving at the cap. Refitting those cells at
   a 300-epoch cap with patience 20 moves the median to 119 and the penalty
   **grows** rather than shrinks, with every change interval covering zero, so
   the published shorter-budget figures are the conservative pair
   (`outputs/final/tcn_convergence_sensitivity_v1/`).
2. **Missing values.** LightGBM routes NaN down a learned branch; the network
   sets missing cells to the training mean. Recorded, and it does not bind: the
   imputed fraction is 0.0 in all 1,008 cells.
3. **Ragged lag sets.** A variable whose lag set is only partly admissible is
   demoted to static features — still visible, not convolved.

## 8. Authorities

| Manuscript section | Authority |
|---|---|
| 4.7 local thermal state | `outputs/final/information_ladder_v6_authority_v1/` |
| 4.7 geometry interaction | `outputs/final/ladder_geometry_interaction_v1/` |
| 4.8 forcing value and inference | `outputs/final/forcing_regime_v5_observed_inference_authority_v1/` |
| 4.8 shuffle placebo | `outputs/final/forcing_placebo_v5a_authority_v1/` |
| 4.8 shift placebo | `outputs/final/forcing_placebo_v5a_shift_authority_v1/` |
| 4.8 component attribution | `outputs/final/forcing_component_authority_v1/` |
| 4.8 event metrics | `outputs/final/forcing_event_metrics_summary.parquet` |
| 4.8 heterogeneity | `outputs/final/forcing_heterogeneity.parquet` |
| 4.8 F2a recovery | `outputs/final/f2a_recovery_authority_v1/` |
| 4.9 forcing-by-information | `outputs/final/forcing_information_interaction_v1/` |
| 4.10 architecture crossing | `outputs/final/architecture_geometry_interaction_v1/` |
| 4.10 convergence sensitivity | `outputs/final/tcn_convergence_sensitivity_v1/` |

Prediction shards are excluded from version control as deterministically
regenerable and fingerprinted instead: `outputs/final/shard_digest_index.json`
carries one SHA-256 per shard, so a regenerated shard can be checked
byte-for-byte against the one these numbers were computed from.

## 9. What these arms do not establish

* Nothing here is confirmatory. Every effect in Sections 4.7–4.10 is
  post-outcome and descriptive.
* L2 withholds a station's own record from the model; it does not predict at a
  location that was never instrumented. The cohort, climatology and keys still
  come from gauged sites.
* F3 is a retrospective realized gridded meteorological oracle, not a forecast
  product and not an operational gain.
* The architecture axis carries one network. One network is one draw from what a
  deep model does on this panel.
* The forcing and information axes are separate conditional designs. They are
  compared and never summed.

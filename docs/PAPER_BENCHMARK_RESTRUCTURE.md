# Benchmark restructure of `paper/ThermoRoute_paper.md` — 2026-08-06

| Field | Value |
|---|---|
| Target files | `paper/ThermoRoute_paper.md`, `paper/highlights.md`, `paper/cover_letter.md`, this file |
| Branch | `feat/route-a-completion` |
| Supersedes | `docs/PAPER_RESTRUCTURE_20260805.md` for structure, figure routing, and Table 4.4/4.6 provenance; that document's number-source tables (§6) remain valid except where noted in §5 below |
| Files deliberately NOT touched | `src/`, `scripts/`, `tests/`, `protocols/`, `pyproject.toml`, `requirements*.txt`, `.github/`, `ops/`, `outputs/`, `paper/agu_submission/**`, `paper/si/**`, `paper/FIGURE_REDRAW_SPEC.md`, both sibling worktrees (read-only) |
| Target venue | Water Resources Research (AGU) |
| Prior manuscript bytes | SHA-256 `1507f20c3e83a648…` at commit `dee33fb` |

---

## 1. What changed, and why

The superseded manuscript argued that ThermoRoute's architecture solves daily
river-temperature prediction, with the evaluation apparatus as supporting
machinery. The development evidence does not support that reading:

- LightGBM has the lowest station-median RMSE at 1, 3, and 7 days;
- ThermoRoute's station win rate against LightGBM at 1 day is 0.00;
- ablating the router, mixture, dynamic prior or residual bound moves 1-day RMSE
  by ≤ 0.004 °C, and only removing the temporal encoder degrades meaningfully;
- an information-matched plain causal TCN reproduces the full architecture to
  within 0.023 °C at every seed and lead.

The evidence *does* support a different and stronger claim, which is now the main
line:

> Reported skill in river-temperature machine learning is highly sensitive to
> baseline choice, information boundary and spatial partition. Against damped
> persistence, scored on identical prediction keys, and with whole-region
> holdout, the incremental advantage of complex models shrinks substantially.

The paper is therefore an evaluation-benchmark paper. The architecture is the
object under test, not the contribution.

**Title.** `Strong baselines, controlled leakage, and a pre-specified inference
gate for daily river water-temperature hindcasting` →
`Reported skill in daily river water-temperature prediction shrinks under strong
baselines and whole-region holdout`. The old title named subjects; the new one
states a finding and a transition of control. The inference gate is deliberately
*not* in the title: it is governance, not a hydrological finding.

---

## 2. New outline

Argumentative spine (Yang-group pattern): important water problem → prevailing
understanding → heterogeneity it cannot explain → the design gap that hid it →
redesigned study → establish the phenomenon → show the heterogeneity → decompose
the mechanism → rise to water-resource meaning.

| § | Section | Role in the spine |
|---|---|---|
| — | Title, Key Points, Manuscript status, Abstract (5 moves, 248 words), Plain Language Summary, Keywords | — |
| 1 | Introduction, five paragraphs: (1) thermal regimes and why daily prediction matters; (2) prevailing understanding — deep models have advanced this problem; (3) the heterogeneity — reported gains are more dispersed than the architectures explain, and the same model reports +0.251 or +0.038 on one panel; (4) the identification gap, framed as three methodological failures of prevailing designs; (5) the redesigned study and its four questions | problem → prevailing view → heterogeneity → gap → design |
| 2 | Data and design: 2.1 frozen panel and the cohort it defines; 2.2 temporal roles and the common key set; 2.3 evaluation-period inputs and the information boundary | design |
| 3 | Methods: 3.1 the predictor (equations 1–8, parameter count, cost); 3.2 the reference set; 3.3 leakage control; 3.4 two spatial partitions; 3.5 conformal intervals; 3.6 estimand, metrics (equations 9–10), comparison set, and why this study is descriptive; 3.7 the one-time 2021–2023 evaluation | design |
| 4 | Results, question-first, headings that are judgements: 4.1 *The reference model, not the architecture, sets the reported gain*; 4.2 *The most accurate model on this panel is a gradient-boosted tree*; 4.3 *An information-matched plain convolutional network reproduces the architecture*; 4.4 *Whole-region holdout removes a third of the transfer skill a random split reports*; 4.5 *Skill is regionally uniform, and interval coverage is bought with width*; 4.6 evaluation-period slots | establish → heterogeneity → mechanism |
| 5 | Discussion: 5.1 why these numbers are smaller than published gains (scale, reference, design); 5.2 what a benchmark of this geometry can carry; 5.3 what transfers | disagreement as design difference → meaning |
| 6 | Limitations: 6.1 the eight machine-verifiable claim blocks; 6.2 cohort, measurement, design; 6.3 the *development-period* probabilistic suite is not reported; 6.4 threshold and archive scope | — |
| 7 | Conclusions: design and scale, three quantitative answers, mechanism, water-resource consequence, permanent scope | rise to meaning |
| 8 | Open Research | — |
| — | Acknowledgments, Supporting Information | — |

Every §4 subsection opens with the question, states the conclusion in the first
sentence, gives the numbers and the heterogeneity, and closes with the question
that motivates the next one. No sentence in §4 begins "Figure N shows".

---

## 3. The three audit corrections, as applied

### 3.1 §3.6 now leads with the two substantive reasons

The manuscript previously presented the cluster count as *the* reason the study
is descriptive, which invited the "a threshold designed to fail" reading. §3.6
now states three independent and permanent reasons, in order of how binding they
are, and states first that eligibility is a fixed property of the frozen gate
rather than a quantity recomputed from the cohort.

| # | Reason | Manuscript wording | Source of fact |
|---|---|---|---|
| 1 | Both structural assumptions are recorded as unmet — `INDEPENDENT_EXCHANGEABLE_HUC2_SAMPLING` (not probability sampled) and `JOINT_CLUSTER_VECTOR_SIGN_SYMMETRY` (no randomized sign assignment) | "recorded as unmet … design facts rather than measurements … would hold with 30 regions as surely as with 15" | `src/thermoroute/inference_gate.py:59-77`, both hardcoded `"status": "NOT_ESTABLISHED"` |
| 2 | The null-simulation component was never implemented and fails closed on every run | "a declared requirement that was not built, disclosed rather than quietly dropped" | `inference_gate.py:444-460`; `"pass": False` unconditional at `:451`; status token `NOT_IMPLEMENTED_FAIL_CLOSED` at `:449` |
| 3 | Too few clusters — least binding, presented last | thresholds, cohort values, then: "**the threshold is unreachable under a HUC2 partition for any U.S. cohort whatsoever**… That is a specification error, made when the amendment was written and identified before any evaluation label was accessed" | `inference_gate.py:55` `MIN_CLUSTERS = 30`; `:429` threshold check; WBD has ~21 HUC2 regions |

The manuscript also states that eligibility is not computed from the cluster
diagnostics and that a HUC8 partition would clear the cluster gate and return the
same verdict (`claim_eligible` is the hardcoded literal `False` at
`inference_gate.py:501`; the opening refuses to run unless the gate has failed
closed, `opening.py:5255`). The measured cluster ladder is cited in full:

| Partition | n_clusters | effective fraction | Gate |
|---|---:|---:|---|
| HUC2 (current) | 15 | 0.636 | FAIL (count and fraction) |
| HUC4 | 64 | 0.507 | FAIL (fraction) |
| HUC6 | 75 | 0.485 | FAIL (fraction) |
| HUC8 | 95 | 0.758 | PASS all three |

with the explicit statement that adjacent HUC8 units on the same river are not
independent, so passing that way would satisfy the arithmetic while defeating the
purpose. Source: `docs/OPTION_A_DESCRIPTIVE_BENCHMARK_SCOPE.md` §1, recomputed
with `thermoroute.inference_gate.cluster_geometry` against
`data_usgs/station_registry_v1.csv`.

**Note for anyone repeating the recomputation:** `station_registry_v1.csv` stores
`huc_cd`/`huc2` without leading zeros, so slicing `huc_cd[:2]` yields 18 clusters,
not 15. Zero-pad first. The gate itself reads the `huc2` column directly and is
unaffected.

### 3.2 The undefined metric is now defined once, prominently, and labelled everywhere

There was never a sign-convention conflict. Code, protocol, manuscript, SI and
figure spec all use `candidate − reference` with negative favouring ThermoRoute.
The defect was that `skill_score = 1 − RMSE/RMSE_ref`
(`src/thermoroute/metrics.py:59-61`) was printed as bare signed numbers and
defined nowhere, two paragraphs from a °C-signed ΔRMSE of the opposite polarity.

Fixed by adding equations (9) and (10) at the head of §3.6:

> **(9)** `ΔRMSE = RMSE(candidate) − RMSE(reference)`, in °C, **negative** favours the candidate
> **(10)** `skill = 1 − RMSE(candidate)/RMSE(reference)`, dimensionless, **positive** favours the candidate

with the explicit statement that a positive ΔRMSE and a positive skill score mean
opposite things and that no axis, column or sentence mixes them. Nothing was
flipped. Every occurrence in §4, §5, §7, the Abstract, the Key Points and the
cover letter now carries either a °C unit (ΔRMSE) or the word "skill" plus
"dimensionless" on first use in each subsection.

### 3.3 air2stream removed; the plain causal TCN promoted to a named primary baseline

`air2stream` status is `NOT_RUN` (`outputs/reports/usgs_experiment.md` L5 in the
multicore worktree) and the implementation is unofficial
(`docs/AIR2STREAM_SOURCE_BUILD_AUDIT_20260801.md`). It is removed from the
Abstract, the Key Points, the Introduction, §3.2's reference-set enumeration, the
§4.1 score table, the Conclusions and the cover letter. Its absence is explained
in §3.2 in two sentences and routed to SI07 for status, provenance and what a
defensible comparison would require. §6.2 states that the hybrid process family
is consequently absent from every comparison in the paper.

The **plain causal temporal convolutional network** is promoted from architecture
control to named primary baseline in §3.2 and given its own results subsection
(§4.3). This also answers the criticism that the only deep baseline was an LSTM.

---

## 4. Every number used, with its source

`MC/` = read-only reads from
`/home/lzq/workspace/parttime/thermoroute-water-temperature-multicore/`.
`SRC/` = read-only reads from this repository's `src/`.
`PRE/` = carried forward unchanged from the superseded manuscript bytes at commit
`b0699a8` (see `docs/PAPER_RESTRUCTURE_20260805.md` §6 for their own provenance).

### 4.1 Lineage warning — read before touching any number

**The manuscript is bound to the multicore lineage.** Three mutually inconsistent
copies of the headline development table exist:

| Copy | 1 d / 3 d / 7 d ThermoRoute RMSE | skill vs persist | skill vs damped |
|---|---|---|---|
| **`MC/outputs/reports/usgs_experiment.md` (used)** | 0.631 / 1.291 / 1.657 | +0.203 / +0.187 / +0.251 | +0.168 / +0.076 / +0.038 |
| `outputs/reports/usgs_experiment.md`, working tree (uncommitted `M`) | 0.630 / 1.293 / 1.662 | +0.203 / +0.187 / +0.250 | +0.168 / +0.076 / +0.037 |
| `outputs/reports/usgs_experiment.md` at `HEAD` | 0.629 / 1.287 / 1.659 | +0.204 / +0.189 / +0.249 | +0.172 / +0.077 / +0.038 |

The multicore copy is the one used, because it is the only copy consistent with
every other artifact the manuscript cites: `MC/outputs/reports/tuurt.md`,
`MC/outputs/reports/stratified.md`, `MC/outputs/reports/lstm_baseline.md`,
`MC/outputs/reports/region_transfer.md` and
`MC/outputs/tables/development_route_a_estimand_mirror.{md,csv}` all agree with
it. **The local working-tree modification to `outputs/reports/usgs_experiment.md`
must not be used to update the manuscript without first re-deriving the whole
citation set from the same lineage.**

### 4.2 Cohort and panel

| Value | Used in | Source |
|---|---|---|
| 657,480 site-days; 120 stable site numbers; 2006-01-01–2020-12-31 | §1, §2.1, §7 | `README.md`; `protocols/route_a_confirmatory_protocol.md` §1; `PRE/§2.1` |
| 34 states; 15 HUC2 groups | Abstract, §1, §2.1, §7 | `README.md`; `PRE/§2.1` |
| Missingness 15.8% `WTEMP`, 2.8% `FLOW`, ~0.07% meteorology; leap-year provider-calendar gap 2008/2012/2016/2020 | §2.1 | `PRE/§2.1` |
| Max contiguous gaps 2,404 d (`WTEMP`), 1,369 d (`FLOW`) | §2.1 | `PRE/§2.1` |
| 1,345 rejected of 1,465 candidates = 950 + 376 + 19 | §2.1, §6.2 | `PRE/§2.1` |
| 38 stations share a HUC code; 19 have a neighbour within 10 km | §2.1, §4.4, §5.1, §7 | `PRE/§2.1` |
| 2,059 negative-`FLOW` rows at 2 stations, minimum −121 cfs | §2.1, §6.2 | `PRE/§2.1` |
| Split rows 438,240 / 87,720 / 43,800 / 87,720; observed `WTEMP` 341,646 / 84,074 / 42,279 / 85,621 | §2.2 table | `PRE/§2.1` table |
| 249,072 common keys; 83,024 per lead | Abstract, §2.2, §4.5, §7 | `MC/outputs/reports/usgs_robustness_v1.md` L3; `MC/outputs/reports/adaptive_conformal.md` |
| Evaluation interval 2021-01-01 – 2023-12-31 | §2.3, §3.7 | `protocols/route_a_confirmatory_protocol.md` §2 |
| `PASS_EXACT_PRODUCT_BRIDGE` | §2.3 | `PRE/§2.2`; `README.md` |
| 30-site external cohort, deterministic seed, site-ID disjoint | §2.3 | `protocols/route_a_confirmatory_protocol.md` §2–§3; `PRE/§2.3` |

### 4.3 Model specification (new in this revision)

| Value | Used in | Source |
|---|---|---|
| Anchor `A = c_{t+h} + φ_i^h (y_t − c_t)`; per-station φ; no-intercept OLS on consecutive-day anomaly pairs; ≥30 pairs; clipped [0, 0.999] | eq. (1), §3.1 | `SRC/thermoroute/features.py:286-295` (predict), `:143-174` (fit on `train_mask`), `:234-244` (estimator + clip), `:27-29` (`DAMPED_MIN_PAIRS=30`, bounds) |
| Relaxation proposal: `κ = σ(b + b_i + β_q q̃ + β_s s)` clipped [1e-3, 0.999]; `P = c + e + (1−κ)^h (a − e)`; `b` initialised −2.94 | eqs. (2)–(3), §3.1 | `SRC/thermoroute/thermoroute.py:50-110`, bias at `:82` |
| Router: 7 variables × lags 0–14 = 105 pairs; horizon-conditioned query; scaled dot product; sparsemax | eq. (4), §3.1 | `SRC/thermoroute/thermoroute.py:116-160`, weighting at `:152-155`, sparsemax at `:34-44`; `MAX_ROUTER_LAG=14` at `config.py:132` |
| TCN: 2 blocks, kernel 3, dilations 1 and 2, width 40, left-only causal pad; receptive field `1 + (k−1)(2^B − 1) = 7` | eq. (5), §3.1, §3.3 | `SRC/thermoroute/thermoroute.py:166-184`, dilation `:172`, `ConstantPad1d` `:174`; `config.py:147-150` |
| Mixture: 3 experts, soft softmax gate | eq. (6), §3.1 | `SRC/thermoroute/thermoroute.py:190-209`; `n_experts=3` at `config.py:151` |
| Bounded residual `ŷ = A + δ tanh(z/δ)`, `δ = 1.0 °C` | eq. (7), §3.1, §4.2 | `SRC/thermoroute/thermoroute.py:424-439`; `DELTA_SCALE = 1.0` at `config.py:187` |
| Loss `MSE + Σ_τ ρ_τ + λ_e·BCE + λ_c·C + λ_r·‖ŷ−A‖₁`; `λ_e = 0.3`, `λ_r = 1e-2`, `λ_c = 1.0`, loss scale 1.0; quantiles 0.05/0.50/0.95; point and pinball at unit weight; `C ≡ 0` by construction | eq. (8), §3.1 | `SRC/thermoroute/train.py:154-213`, `:199-205`; weights `config.py:159-163, 169`; `QUANTILES` `config.py:94` |
| One model for all three leads (horizons are an input dimension; heads emit `[B, H]`) | §3.1 | `SRC/thermoroute/thermoroute.py:71, 236-249, 276-277, 445-446`; `train.py:665`; `config.py:93` |
| **38,505 trainable parameters** (canonical configuration) | §3.1, §4.3, cover letter | `SRC/thermoroute/development_controls.py:60` `THERMOROUTE_REFERENCE_PARAMETERS = 38_505`; cross-checked against `MC/outputs/runs/09b_development_controls/e87f141ad92c33ce1d3f/development_controls_architecture_budget.csv` (`ThermoRoute-ladder-07_plus_WDSP`, ratio 1.0000) |
| AdamW, lr 2e-3, wd 1e-4, batch 1,536, grad clip 1.0, ≤80 epochs, patience 12, plateau ×0.5 at patience 4, equal-station fixed-size bootstrap, 5 seeds | §3.1 | `SRC/thermoroute/train.py:500-501, 584, 593, 628`; `config.py:153-158, 191`; batch override `development_controls.py:56` and `MC/outputs/runs/09_usgs_experiment/.../run.json:539` |
| Best epoch per seed 18 / 15 / 14 / 18 / 26 | §3.1 | `outputs/runs/09_usgs_experiment/c5d4aebc81cf227531d7/checkpoints/seed{0..4}.pt.meta.json`, field `epoch` |
| CPU-only, Intel Xeon Gold 6430, 1 intra-op and 1 inter-op thread | §3.1 | `outputs/runs/09_usgs_experiment/c5d4aebc81cf227531d7/run.json:4, 285, 394-395, 414` |
| Plain causal TCN 38,346 params; plain MLP 38,860; both within 2% of full; same 17,280 optimiser steps; same inputs, anchor and exclusions | §3.2, §4.3, cover letter | `MC/outputs/runs/09b_development_controls/e87f141ad92c33ce1d3f/development_controls_architecture_budget.csv`, rows `PlainCausalTCN-7var`, `PlainMLP-7var` |
| LightGBM: 4 predeclared candidates per lead on 2016–2017 station-macro RMSE; LSTM: 3 candidates, seed 0, then 5 members | §3.2 | `MC/outputs/tables/lightgbm_joint_validation_selection.csv`; `MC/outputs/tables/lstm_validation_selection.csv` |

### 4.4 Development-period results

| Value | Used in | Source |
|---|---|---|
| Persistence 0.803 / 1.576 / 2.217 °C | §4.1, §4.4 | `MC/outputs/reports/usgs_experiment.md` |
| Damped persistence 0.774 / 1.406 / 1.738 °C | §4.1, §4.3 | same |
| LightGBM 0.578 / 1.280 / 1.649 °C | Abstract, §4.1, §4.2, §7, highlights | same |
| ThermoRoute 0.631 / 1.291 / 1.657 °C | §4.1, §4.2 | same; cross-checked `MC/outputs/reports/lstm_baseline.md` |
| LSTM 0.662 / 1.323 / 1.679 °C | §4.1, §4.2 | `MC/outputs/reports/lstm_baseline.md` |
| Skill vs. persistence +0.203 / +0.187 / +0.251 | Abstract, KP1, §1, §4.1, §4.4, §4.5, §5.1, §7 | `MC/outputs/reports/usgs_experiment.md`; `MC/outputs/reports/tuurt.md`; `MC/outputs/reports/stratified.md` |
| Skill vs. damped +0.168 / +0.076 / +0.038 | Abstract, KP1, §1, §4.1, §4.4, §7 | same |
| **0.560 °C total and 0.479 °C damping share at 7 d** — arithmetic on the three station-median RMSEs above (2.217 − 1.657 = 0.560; 2.217 − 1.738 = 0.479) | Abstract, §4.1, §7, cover letter | derived by subtraction from `MC/outputs/reports/usgs_experiment.md`; stated in the manuscript as a comparison of station-median RMSE *levels*, not as a decomposition of paired differences |
| ΔRMSE TR − damped −0.130 / −0.104 / −0.062 °C; win 0.86 / 0.90 / 0.93 | §4.1 | `MC/outputs/tables/development_route_a_estimand_mirror.md` and `.csv` |
| ΔRMSE TR − LightGBM +0.046 / +0.008 / −0.005 °C; win **0.00** / 0.40 / 0.60 | §4.2, §7, cover letter | same (`win_rate = 0.0` exactly, h=1 LightGBM row of the CSV) |
| Cluster-bootstrap intervals [−0.177,−0.081], [−0.126,−0.083], [−0.071,−0.052] (damped); [+0.043,+0.057], [+0.001,+0.020], [−0.010,+0.010] (LightGBM) | §4.1, §4.2 | same |
| `NO_STRONG_INFERENCE`, `SMALL_CLUSTER_COUNT_LT_30`, `LOW_EFFECTIVE_CLUSTER_FRACTION` | §4.1 | same, small-cluster sensitivity table |
| Region-weighted skill +0.204 / +0.189 / +0.253 | §4.5 | `MC/outputs/reports/stratified.md` |
| Per-region 1-d range +0.131 (HUC2:10) to +0.285 (HUC2:18); 7-d +0.217 (HUC2:06) to +0.290 (HUC2:09); 7-d vs damped +0.012 to +0.077 | §4.5 | same |
| Drainage strata n = 39 / 38 / 39; 1-d +0.190 / +0.214 / +0.231 | §4.5 | same |
| Random held-site 4-fold: +0.178 / +0.172 / +0.241 (persist), +0.145 / +0.061 / +0.030 (damped) | KP3, §4.4, §5.1, §7 | `MC/outputs/reports/tuurt.md` |
| Held-region: +0.155 / +0.116 / +0.155 (persist), +0.147 / +0.086 / +0.075 (damped) | Abstract, KP3, §4.4, §5.1, §7 | `MC/outputs/reports/tuurt.md`; `MC/outputs/reports/region_transfer.md` |
| **92% and 62% retention** — `0.172/0.187 = 0.920` and `0.116/0.187 = 0.620` | §4.4, §5.1 | derived by division from the two rows above |
| Held-region RMSE TR 0.676 / 1.428 / 1.860; LGB 0.652 / 1.391 / 1.786; LSTM 0.679 / 1.445 / 1.876 | §4.4 | `MC/outputs/reports/region_transfer.md`; `MC/outputs/reports/lstm_baseline.md` |
| Held-region ΔRMSE +0.031 [+0.023,+0.040], +0.038 [+0.029,+0.050], +0.067 [+0.049,+0.080] °C; win ≈ 0.16 | §4.4 | `MC/outputs/reports/region_transfer.md` |
| Fold geometry [30, 30, 31, 29]; mean nearest-training-gauge distance 289 km | Abstract, §3.4, §4.4, §5.1, §7, cover letter | `MC/outputs/reports/region_transfer.md` |
| Plain-TCN paired ΔRMSE per seed: h1 −0.0079/−0.0101/−0.0003/−0.0033/−0.0105; h3 −0.0079/−0.0120/−0.0077/−0.0117/−0.0179; h7 −0.0219/−0.0060/−0.0226/−0.0046/−0.0035 °C — quoted as ranges, and as "within 0.023 °C" (max |value| = 0.0226) | Abstract, §4.3, §5.1, §7, cover letter | `MC/outputs/runs/09b_development_controls/e87f141ad92c33ce1d3f/development_controls_report.md`, "Full ThermoRoute versus matched neural controls" |
| Plain-MLP paired ΔRMSE h1 range −0.0360 to −0.0411 °C | §4.3 | same |
| Stage-09b per-seed-mean median station RMSE: ThermoRoute-ladder-07 0.6452 / 1.3047 / 1.6682; PlainCausalTCN 0.6478 / 1.3241 / 1.6885; PlainMLP 0.7006 / 1.3424 / 1.6937 | §4.3 comparability note | same, "2019–2020 development-evaluation results" |
| Ablations: noTCN 0.679 / 1.338 / 1.675; noRouter 0.634; noMoE 0.634; noDynamicPrior 0.628; fixedKappa 0.627; unbounded 0.630; DampedPriorOnly 0.774 / 1.406 / 1.738 | Abstract, §4.3, §7 | `MC/outputs/reports/usgs_experiment.md` |
| **"at most 0.004 °C"** — max |0.631 − {0.634, 0.634, 0.628, 0.627, 0.630}| = 0.004 | Abstract, §4.3, §7, cover letter | derived by subtraction from the row above. The reviewer's brief said "≤ 0.005"; the measured maximum is 0.004, and the manuscript states 0.004 |
| Split-CQR coverage 0.909, width 3.87, interval score 4.93, n = 249,072; per lead 0.905/0.910/0.912 at 2.01/4.22/5.38; warm tail n = 30,531, 0.918 at 3.50 | §4.5 | `MC/outputs/reports/adaptive_conformal.md` |
| Block-max sensitivity 0.981 coverage at width 5.74 | §4.5 | same |
| Delayed ACI 0.903 / 0.900 / 0.892 for γ = 0.005 / 0.02 / 0.05, several slices infinite width | §4.5 | same |
| Sensor noise 0.25 sd → 1.621 (+147%); 0.5 sd → 2.999 (+360%) at h=1; missing blocks ≈ +0.13 at h=1 and +0.04 at h=7; air-T ±2 sd → +0.11 to +0.15; flow ×0.5/×2 → ≤ +0.003 | §4.5 | `MC/outputs/reports/usgs_robustness_v1.md` |
| Bounded-deviation audit 100.00% pointwise; max |correction| 1.0000 °C; 360 station-by-lead cells | §4.2 | `MC/outputs/reports/prop1_binding.md` |

### 4.5 Gate, comparison set, and the descriptive verdict

| Value | Used in | Source |
|---|---|---|
| Gate thresholds `n_clusters ≥ 30`, `effective_cluster_fraction ≥ 0.75`, `largest_cluster_share < 0.25` | §3.6, §5.2, §7, highlights, cover letter | `docs/B02_PERMANENT_DESCRIPTIVE_CLAIM_WORDING.md`; `SRC/thermoroute/inference_gate.py:55-57, 429-434` |
| ≤ 15 HUC2; effective count 9.54; effective fraction ≈ 0.636; largest share 0.217; cluster sizes 2 / 7.0 / 26 | §3.6, §5.2, §7 | `docs/B02_…md`; `MC/outputs/tables/development_route_a_estimand_mirror.md` small-cluster table |
| HUC2/4/6/8 ladder: 15/0.636, 64/0.507, 75/0.485, 95/0.758 | §3.6 | `docs/OPTION_A_DESCRIPTIVE_BENCHMARK_SCOPE.md` §1 |
| ~21 HUC2 regions nationally | §3.6, cover letter | `docs/OPTION_A_DESCRIPTIVE_BENCHMARK_SCOPE.md` §1 (WBD: 18 CONUS + 19 AK + 20 HI + 21 Caribbean) |
| `DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED` | §3.6, §4.6, §7, highlights, cover letter | `docs/B02_…md`; `protocols/route_a_claim_registry_v1.json` |
| Five-row family with margins 0.00/0.00/0.00/+0.05/+0.05 °C | §3.6 table, §4.6 | `protocols/route_a_confirmatory_protocol.md` §4; `PRE/§4.2` |
| ≥ 100 valid paired targets per reportable station/lead | §3.6 | `protocols/route_a_confirmatory_protocol.md` §2, §4 |
| 10,000 bootstrap draws; exact 2^K sign enumeration; Holm over five p-values | §3.6 | `protocols/route_a_confirmatory_protocol.md` §4 |
| `REV_NOT_EVALUATED_NO_PREDECLARED_COST_LOSS_RATIOS` | §3.6 | `MC/outputs/reports/rev_curve.md` |
| Eight temporal-coverage candidates | §3.6, §4.6 | `protocols/route_a_temporal_coverage_policy_v1.json`; `PRE/§4.4` |

### 4.6 Stage-19 degenerate intervals

135 of 26,993,675 member-level rows (0.0005%); **0 strict ordering violations**;
maximum monotonicity violation exactly 0.000 °C; 123 global LightGBM + 12
per-station LightGBM; 12 sites in HUC 04/05/06; 129 at h=1, 6 at h=3, 0 at h=7;
95.6% with observed target < 1 °C. Used in §6.3, highlights, cover letter.
Source: `docs/STAGE19_DEGENERATE_INTERVAL_DISPOSITION_20260805.md` §1 and §2.1,
measured from `MC/outputs/predictions/usgs_predictions_with_perstation_v2.parquet`.
The phrase **"quantile crossing" appears nowhere** in the manuscript, highlights,
or cover letter; verified by regex (0 matches).

---

## 5. Corrections to `docs/PAPER_RESTRUCTURE_20260805.md`

That document remains the provenance record for the `PRE/`-sourced values. Four
of its statements are superseded:

1. **§4 routes Table 4.4 to adaptive-conformal analogues.** Superseded. Table 4.4
   is filled from `trusted/probabilistic_evaluation_v2.json`, emitted
   unconditionally by the trusted scorer inside the single opening — a different
   code path and a different evidence period from the withheld development-period
   Stage-19 suite. Field map: `docs/R13_POSTOPEN_TABLE_RENDERER.md` M13.
2. **§4 states that `scripts/26_validate_claims.py --write-generated-results`
   renders Table 4.1.** It does not. It appends five HTML-comment-delimited claim
   blocks under the heading `# Route-A receipt-derived results`. Table rendering
   is a separate script (R13). The two layers are complementary.
3. **§2 describes §4.6 as containing six slots including a held-region
   evaluation-period arm.** The opening produces no held-region artifact, so that
   fragment is a status token, not a number. Table 4.6 is retitled "external arm"
   in the manuscript and states this explicitly.
4. **Known gap 3** (no per-station LightGBM development score) still stands and is
   still handled the same way: §3.2 names the variant without quoting a number and
   §4.1 omits it from the score table.

---

## 6. What was cut, and where it went

Prose word count (table rows excluded): **12,420 → 11,976, a 3.6% net
reduction**. That net figure understates the cut: roughly 2,500 words were
removed and roughly 2,050 words of content required by this brief were added.
The removals and additions are itemised below.

### 6.1 Removed

| Removed | Words (approx.) | Why | Where it now lives |
|---|---:|---|---|
| Legacy b1/s2/p3 material: the Introduction paragraph and the whole of old §2.5 | 230 | Brief: remove entirely from Introduction, Data and Limitations | One clause in §8 Open Research recording that three legacy CSVs are not redistributed, pointing at `protocols/legacy_three_site_semantics_notice_v1.md`. The two allowlisted corrective sentences are no longer needed, because the lint they were allowlisted against fires only on `b1`/`s2`/`p3` tokens, which no longer appear. `legacy_semantics_sentence_allowlist` is an allowlist, not a required-presence list; `required_permanent_coverage` names only the eight `LIMIT_*` blocks, all of which are retained |
| Old §3.7 transport/durability detail: exactly-once HTTP semantics, canonical-transaction replacement rules, unpublished owner-private state, directory-rename publication, body-hash substitution | 350 | Brief: move audit and filesystem language to SI; main text needs one or two sentences | Two sentences in §3.7 ("transactional… honest-owner crash and replay guards, not protection against a malicious owner or a same-privilege adversary") plus SI02 |
| Old §3.3 replay detail: `python -I -B`, child-process denial, per-parameter recomputation list | 90 | same | One sentence in §3.3 plus SI02 |
| Old §3.8 pre-registration and reproducibility, receipt-chain paragraph | 190 | Governance, not method | Folded into the last paragraph of §3.7; detail in SI04/SI15 |
| Old §5.1 "A pre-specified gate that failed is a result, not an excuse" (a standalone 640-word essay) | 260 net | The gate is governance and must not be the paper's headline | Compressed into §5.2 "What a benchmark of this geometry can carry", which now also carries the three-reason summary and the design lesson |
| Old §2.2 discovery/provenance narrative and old §2.4 external-cohort detail | 240 | Consolidation | §2.1 and §2.3, compressed; ledger detail in SI01 |
| air2stream paragraphs in §3.2 and the `NOT_RUN` column in the §4.1 table | 120 | Correction 3 | Two sentences in §3.2; full status in SI07 |
| Old §6.2/§6.4/§6.5 duplication of caveats already stated in Methods | 340 | Brief: each permanent caveat stated in full exactly once, cross-referenced elsewhere | §6.2 and §6.4, consolidated; cross-references replace repeats |
| Old §8 rights enumeration | 120 | AGU DAS does not need the full class list | SI16, with a pointer |
| Abstract and Plain Language Summary | 140 | AGU 250-word limit | — |
| Old Conclusions paragraph restating the gate at length | 60 | Duplication | Compressed final paragraph of §7 |

### 6.2 Added (required by this brief)

| Added | Words (approx.) | Required by |
|---|---:|---|
| §3.1 equations (1)–(8), parameter count, optimizer schedule, shared-lead evidence, training/inference-cost slots | 450 | "Methods completeness" |
| §3.6 equations (9)–(10) and the metric-labelling rule | 200 | Correction 2 |
| §3.6 three-reason rewrite, the HUC ladder, and the specification-error disclosure | 600 | Correction 1 |
| §4.3, the information-matched plain causal TCN as a named baseline | 310 | Correction 3 |
| Question-first opening and closing sentences in every §4 subsection | 200 | Writing model |
| §5.1 "Why these numbers are smaller than published gains" (new) | 290 | Writing model: disagreements as differences of scale and design |
| §7 quantitative answers and water-resource consequence | 130 | Writing model |
| §4.6/§6.3 evaluation-period vs development-period probability correction | 180 | Coordinator correction |

---

## 7. Main-text figure specifications

There are currently no main-text figures. Four are specified below, each with one
irreplaceable job, each answering the question the previous one raises.

**Citation mechanism.** `paper/FIGURE_REDRAW_SPEC.md` §7 forbids hand-inserted
figure references: "A complete paper projection must add references only through
the appropriate PRE/POST renderer… No manual `\includegraphics`, caption, or
result transcription is authorized." The manuscript therefore carries **inert
placement anchors**, not references, in the form

```
<!-- FIGURE_ANCHOR id=<F1|F2|F3|F4|S1..S9> state=<PRE|POST|POST_DEVELOPMENT> role=first_citation source=paper/FIGURE_REDRAW_SPEC.md#figure-<n> -->
```

Thirteen anchors are placed: F1 (§1 close, after the narrow research question,
and §2.1), F2 (§4.1 close), F3 (§4.4 close), F4 and S7 (§4.5 close), S1 (§2.1),
S2 (§2.3), S3 (§3.1), S4 (§4.3), S5/S6/S8 (§4.6), S9 (§6.3). They contain no
number, no caption and no image path, so they cannot substitute for a render, and
the renderer resolves each into a numbered reference at the anchored position.

**Ownership boundary.** `paper/FIGURE_REDRAW_SPEC.md` and the two POST renderer
skeletons are owned by the figure track and were **not modified**. The main-figure
re-assignment below is a required delta to that spec, recorded here for its owner.

| New | Job | Nearest current spec slot | Delta required in `FIGURE_REDRAW_SPEC.md` |
|---|---|---|---|
| **Figure 1** | station map and cohort geometry | Figure 1 (`PRE_MATERIALIZED_REDRAW`) panels (a) and (c) + Figure S1 | Replace panel (b) (the bounded-correction schematic) with the station map; keep the persistence-challenge panel and the cluster-geometry/gate panel; move the bounded-correction schematic to S3 |
| **Figure 2** | how baseline choice changes reported skill | Figure 2 (`POST_TEMPLATE_ONLY`) | Add a "reference ladder" panel; keep the all-model station distributions and the five-row forest |
| **Figure 3** | how random held-site vs whole-region holdout changes the transfer conclusion | currently only Figure S7 and Figure 4 panel (d) | **Promote** to a main figure; the mechanism/ablation content currently in Figure 4 moves to SI |
| **Figure 4** | regional and seasonal heterogeneity with conformal coverage–width | Figure 3 panel (a) + Figure 4 panel (c) + Figure S7 | **Merge**; the reliability and event-score panels of the current Figure 3 move to S5 |

Every panel below obeys the existing global rules: one evidence period per
figure; development-period figures carry a mandatory in-panel scope band; ΔRMSE
axes are labelled in °C with "negative favours the candidate" and skill axes are
labelled dimensionless with "positive favours the candidate"; no significance
stars; no development value compared numerically with an evaluation value; the
Stage-19 cause is never described as "quantile crossing".

### Figure 1 — The cohort, and what it can carry

*Question:* what panel is this, and what inferential weight can its geometry
support? *State:* PRE. *Evidence period:* frozen registry and 2006–2015 training
panel only.

- **(a) Station map.** 120 sites on a CONUS base, coloured by HUC2 region,
  symbol size by retained observed-`WTEMP` day count. Inset histogram of
  nearest-neighbour distance between retained stations, with the 10 km mark
  annotated (19 stations) and the 289 km whole-region-holdout mean marked on the
  same axis. *Artifact:* `data_usgs/station_registry_v1.csv`;
  `MC/outputs/reports/region_transfer.md` for 289 km.
- **(b) Cluster geometry against the gate.** Bar chart of the 15 HUC2 station
  counts (2 … 26), with three separate small gauges — never a shared false
  numeric axis — showing 15 vs ≥ 30 clusters, 0.636 vs ≥ 0.75 effective
  fraction, 0.217 vs < 0.25 largest share. A fourth strip shows the HUC2/4/6/8
  ladder (15/64/75/95 clusters; 0.636/0.507/0.485/0.758) with HUC8 marked
  "passes, but adjacent units are not independent". *Artifact:*
  `docs/OPTION_A_DESCRIPTIVE_BENCHMARK_SCOPE.md` §1;
  `MC/outputs/tables/development_route_a_estimand_mirror.md`.
- **(c) The persistence challenge.** Per-station distribution of median observed
  `|T_{t+h} − T_t|` at h = 1, 3, 7 over 2006–2015, exact calendar-day pairs,
  finite observed `WTEMP` at both ends, equal station weight. This is the
  motivation quantity, not a model score. *Artifact:* frozen panel, training
  partition only.

### Figure 2 — Baseline choice, not architecture, sets the reported gain

*Question:* how much of a reported gain survives a strong reference? *State:*
POST for panels (b)–(d); panel (a) is the development-period version and carries
a scope band. Cited at the close of §4.1.

- **(a) The reference ladder.** For each lead, the same ThermoRoute predictions
  scored against persistence, damped persistence, climatology, LightGBM and the
  plain causal TCN: five skill values on one dimensionless axis, connected, with
  the persistence and damped-persistence points labelled (+0.251 and +0.038 at
  7 d). One panel per lead or one panel with lead as a symbol. *Artifact:*
  `MC/outputs/reports/usgs_experiment.md`, `tuurt.md`.
- **(b)–(c) All-model station distributions by lead.** ECDF or station-level
  points of per-station RMSE for persistence, damped persistence, climatology,
  global LightGBM, global LSTM, plain causal TCN and ThermoRoute on the declared
  common-key set, with retained station counts. *Artifact:*
  `trusted/temporal_predictions_v1.parquet`, `trusted/availability_registry_v1.csv`.
- **(d) The five-row forest.** Median station ΔRMSE and whole-HUC2 bootstrap
  interval for all five registered rows, with the 0.00 °C line for the
  damped-persistence rows and the +0.05 °C ceiling for the LightGBM rows, and
  status, station count and cluster count beside each row. *Artifact:*
  `trusted/statistics_v1.json` `tests[*]`.

### Figure 3 — The spatial partition changes the transfer conclusion

*Question:* how much of a reported spatial transfer survives whole-region
holdout? *State:* development period; mandatory in-panel scope band. Cited at the
close of §4.4.

- **(a) The three arms.** Skill against persistence for temporal, random
  held-site, and held-region arms at each lead (+0.203/+0.187/+0.251;
  +0.178/+0.172/+0.241; +0.155/+0.116/+0.155), with the vs-damped values on a
  paired secondary panel. Slope lines make the collapse readable. *Artifact:*
  `MC/outputs/reports/tuurt.md`.
- **(b) Fold geometry.** The 15 HUC2 groups packed into four folds
  [30, 30, 31, 29], drawn on the same base map as Figure 1(a), one colour per
  fold, with held-out stations outlined. *Artifact:*
  `MC/outputs/reports/region_transfer.md`; `MC/outputs/tables/region_transfer.csv`.
- **(c) Distance is the mechanism.** Per-station held-region skill against
  distance to the nearest training gauge, with the 289 km mean marked and the
  random-held-site arm overplotted in a muted style at its own much smaller
  distances. The panel asserts association only; the caption says so.
- **(d) The ranking is unchanged.** Held-region station-median RMSE for
  ThermoRoute (0.676/1.428/1.860), LightGBM (0.652/1.391/1.786) and LSTM
  (0.679/1.445/1.876), with the paired ΔRMSE and HUC2 intervals beneath.
  *Artifact:* `region_transfer.md`, `lstm_baseline.md`.

### Figure 4 — Regional and seasonal heterogeneity, and what coverage costs

*Question:* is the remaining skill uniform, and what does a calibrated interval
cost? *State:* POST for (a)–(c); (d) is development period with a scope band.
Cited at the close of §4.5, together with S7.

- **(a) Regional heterogeneity.** Per-HUC2 median skill against persistence and
  against damped persistence at each lead, ordered by region, with the pooled
  median and the region-weighted mean as reference lines (+0.251 vs +0.253 at
  7 d against persistence; the +0.012 to +0.077 compression against damped
  persistence). Station count per region encoded by marker size. *Artifact:*
  `trusted/spatial_sensitivity_v1.json` `comparisons[].per_huc[]`;
  development analogue `MC/outputs/reports/stratified.md`.
- **(b) Seasonal and annual heterogeneity.** All eight predeclared temporal
  candidates — equal-weighted 12 year-by-season cells, three leave-one-year, four
  leave-one-season — plotted per formal row against the formal effect, with the
  deterministic most-adverse candidate marked and never substituted for the
  formal effect. *Artifact:* `trusted/temporal_coverage_audit_v1.json`.
- **(c) Coverage–width plane.** Empirical marginal coverage against mean interval
  width by model and lead, with the 0.90 nominal reference drawn and no implied
  coverage test. Point-only models bind a `NOT_AVAILABLE` status. *Artifact:*
  `trusted/probabilistic_evaluation_v2.json` `coverage_90`,
  `mean_interval_width_c`.
- **(d) What calibration costs.** The development-period split-CQR, block-maximum
  and delayed-ACI variants on the same coverage–width plane (0.909 at 3.87;
  0.981 at 5.74; 0.903/0.900/0.892 with unbounded widths in several slices),
  showing that coverage is bought with width. Infinite widths render as an
  explicit off-scale status marker, never as a clipped point. *Artifact:*
  `MC/outputs/reports/adaptive_conformal.md`.

---

## 8. Every remaining slot

### 8.1 The fifteen `[TO BE FILLED AFTER OPENING]` markers

Count verified at exactly **15**, unchanged. Mapping per
`docs/R13_POSTOPEN_TABLE_RENDERER.md` §2 (M01–M15).

| Slot | Markers | Filled from |
|---|---:|---|
| Table 4.1, five formal rows | 10 (M01–M10) | `trusted/statistics_v1.json` `tests[test_id].{median_effect_c, ci_low_c, ci_high_c, status, p_one_sided_raw, p_holm}`. The verdict column is pre-filled with `DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED`, which is outcome-independent |
| Table 4.2, all-model scores | 1 (M11) | `trusted/report_v1.md` `## All frozen models` → `### temporal:` table, transcribed under an exact-header contract |
| Table 4.3, temporal coverage | 1 (M12) | `trusted/temporal_coverage_audit_v1.json` `comparison_sensitivities[].frozen_sensitivity_candidates[]` and `.frozen_worst_unfavorable_sensitivity` |
| Table 4.4, interval and probability behaviour | 1 (M13) | `trusted/probabilistic_evaluation_v2.json` `rows[cohort=temporal,model,horizon]`. **Status tokens, never numbers:** `interval_score`, `block_maximum_sensitivity`, `delayed_aci_sensitivity` |
| Table 4.5, outcome quality control | 1 (M14) | `trusted/outcome_quality_audit_v1.json`, `trusted/outcome_qc_gate_v1.json`, `trusted/approved_target_sensitivity_v1.json` |
| Table 4.6, external arm | 1 (M15) | `trusted/report_v1.md` `### external:` table. **Status token:** the leave-one-HUC2-region-out arm, `NOT_EMITTED_BY_THE_ONE_TIME_OPENING` |

### 8.2 Non-opening slots introduced or retained

| Slot | Location | What closes it |
|---|---|---|
| `[TRAINING WALL-CLOCK TO BE RECORDED]` | §3.1, §8 placeholder inventory | **New.** No measured end-to-end training time exists: `outputs/logs/stage09_formal.log` has no duration and `run.json` has no elapsed field. `docs/RUN_PACKAGE_20260802/README.md:20-25` gives an *estimate* only, which must not be used |
| `[INFERENCE COST TO BE RECORDED]` | §3.1, §8 | **New.** No latency, throughput or ms/example figure exists anywhere in `outputs/`, `docs/` or `README.md` |
| `[AUTHOR LIST TO BE COMPLETED]`, `[AFFILIATION 1/2 …]`, `[CORRESPONDING AUTHOR …]` | header, Acknowledgments, cover letter | `docs/FAIR_SUBMISSION_READINESS_AND_TEMPLATES.md` §2 |
| `[DATA DOI TO BE MINTED]`, `[DATA LICENCE TO BE ASSIGNED]`, `[SOFTWARE DOI TO BE MINTED]`, `[RELEASE TAG TO BE ASSIGNED]`, `[REPOSITORY URL TO BE CONFIRMED]` | §8 | `docs/FAIR_EXTERNAL_CLOSEOUT_REQUEST_PACK.md`; blocked on the rights review |
| `[FUNDING …]`, `[COMPUTATIONAL RESOURCES …]`, `[COMPETING INTERESTS …]`, `[CREDIT ROLES …]` | Acknowledgments | external |
| `[DATE …]`, `[SUGGESTED REVIEWERS …]`, `[OPPOSED REVIEWERS …]` | cover letter | external |
| Per-station LightGBM development score | §3.2, §4.1 | No scored summary exists; §3.2 names the variant without a number and §4.1 omits it. Either score it or state its absence in SI07 |

---

## 9. Verification performed

All read-only. No test suite was run and nothing under `scripts/` was executed.

| Check | Result |
|---|---|
| `[TO BE FILLED AFTER OPENING]` marker count | 15, unchanged |
| Eight `ROUTE_A_CLAIM` blocks byte-identical to commit `dee33fb` | PASS (compared with `git show HEAD:paper/ThermoRoute_paper.md`) |
| `required_permanent_coverage`: all eight `LIMIT_*` ids present exactly once | PASS |
| All 21 `free_text_lints` regexes from `protocols/route_a_claim_registry_v1.json`, case-sensitive and case-insensitive, over whitespace-normalised free text with claim blocks removed | **0 hits** |
| B-02 forbidden verbs (`confirm`, `demonstrate`, `establish`, `support`, `achieve`, `outperform`, `prove`, `validate`, `show superiority`, `conclude equivalence/parity`) | No occurrence upgrades a five-row verdict. Remaining matches are `provenance`, `invalidates authorization`, negated `do not prove component necessity`, and `not a validated reproduction` |
| Phrase "quantile crossing" | 0 occurrences in manuscript, highlights and cover letter |
| Abstract length | 248 words (AGU limit 250; was 380) |
| Plain Language Summary | 170 words (AGU limit 200) |
| Key Points | 3 items at 128, 127 and 126 characters (AGU limit 140) |
| B-02 paste-ready descriptive sentence | present verbatim, once, at the end of §3.6 |

**One deliberate deviation from B-02.** B-02 offers its paste-ready sentence for
"Abstract / Conclusion". At 128 words it cannot fit a 250-word abstract. It is
therefore placed verbatim, exactly once, at the end of §3.6, where the reader
meets the gate; the Abstract and §7 carry compressed statements built only from
B-02's allowed-phrase whitelist and containing the verdict string, the
assumption-conditional labelling, and the superiority/non-inferiority/
equivalence/parity/national prohibition. This also satisfies the brief's rule
that each permanent caveat be stated in full exactly once.

---

## 10. Not done, and why

1. **`paper/agu_submission/ThermoRoute_WRR.tex` was not regenerated.**
   `build_agu.py` produces it and another process owns it. It must be regenerated
   from the new manuscript bytes, and its `\title`, `keypoints` environment and
   abstract will change.
2. **`paper/FIGURE_REDRAW_SPEC.md` was not modified.** The four-figure
   re-assignment in §7 above is a delta for its owner. Changing the spec without
   the two POST renderer skeletons (`paper/agu_submission/figures/…` and
   `paper/si/figures/…`), which read its state tokens, would desynchronise them.
3. **`paper/si/**` was not modified.** SI02, SI07, SI08, SI11 and SI16 now receive
   relocated main-text material and need updating.
4. **Nothing under `src/`, `scripts/`, `tests/`, `protocols/` was touched**, and
   nothing was committed. `preopen_document_sha256` for
   `paper/ThermoRoute_paper.md`, `paper/highlights.md` and `paper/cover_letter.md`
   is now stale — it was already stale before this session
   (`docs/OPTION_A_DESCRIPTIVE_BENCHMARK_SCOPE.md` §2.5). The escape is the R6
   block-binding change, already queued in the remediation lineage.

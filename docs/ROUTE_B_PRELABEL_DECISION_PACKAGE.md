# Route B pre-label decision package

| Field | Value |
| --- | --- |
| Date | 2026-08-02 |
| Status | **DECISION-READY DRAFT / OUTCOME-FREE / NOT SELECTED / NOT SEALED / FEASIBILITY NOT PROVEN** |
| Purpose | Consolidate the decisions and evidence required before a Route B protocol can be written, independently sealed and executed |
| Route A boundary | Route A remains a fixed-cohort descriptive analysis; this package does not change its HUC2 gate, claim eligibility, source identity or evidence |
| Runtime boundary | This package is documentation only. It does not authorize target access, downloads, model runs, source changes or writes to `protocols/**`, `src/**`, `scripts/**`, `tests/**`, `ops/**` or `outputs/**` |

This package integrates the current Route B sampling, power, comparator,
identifiability, measurement, missingness and uncertainty drafts. It is a
decision instrument, not a preregistration. No candidate frame, graph registry,
probability sample, power result, model matrix, measurement registry, clean-room
receipt or target-period outcome has been produced by it.

## 1. Decision scope and non-claims

Route B is intended to test a claim in a probability sample of eligible monitored,
graph-disconnected drainage-network components. It cannot be created by splitting
Route A's 15 HUC2 groups, relabelling Route A external sites, or changing a cluster
definition until 30 groups appear. The national HUC2 count is itself less than 30;
HUC2 may be a stratum and reporting label but cannot be the Route B PSU under the
proposed gate.

The following statements are true now:

- the proposed PSU is a frozen weakly connected NHDPlus drainage-network
  component, subject to topology-completeness and residual-dependence caveats;
- official Air2stream, N-HiTS and TFT have provenance-qualified candidate records;
- draft sampling, power, budget, QA, missingness and space×time UQ rules exist;
- existing literature blocks broad “first,” generic SOTA, strict-ungauged and
  operational claims unless the information set and validation design actually
  support them.

The following statements are **not** established:

- Route B is feasible or has at least 30 eligible/reportable PSUs;
- graph disconnection proves statistical independence;
- any task arm, population, estimand, margin, hypothesis family, comparator,
  target period, compute budget, QA threshold or UQ rule has been selected;
- a comparator has built, passed a reference case or received a fair pilot;
- a protocol, frame, model suite or predictor snapshot has been frozen or sealed;
- any result, power, coverage, calibration, superiority, non-inferiority,
  mechanism or operational performance exists.

All result-like language below is a falsifiable claim template. It is not a
finding and must not be copied into a manuscript as one.

## 2. Outcome boundary and custody model

### 2.1 Permitted before the opening boundary

Subject to an approved future work order, pre-label design may use:

- monitoring-location and time-series identity metadata that do not disclose
  target-period values or target-period completeness;
- record metadata ending before the untouched target period;
- frozen WBD/NHDPlus topology, site coordinates, site type, climate, stream-size
  and pre-target predictor-availability metadata;
- development-period data explicitly labelled as development evidence;
- external literature, synthetic stress scenarios and provider documentation;
- target-free model predictions only after the complete information-set,
  transformation and predictor-snapshot contracts are frozen.

Development data may inform a predeclared power grid or history/block-length rule,
but it may not be represented as Route B evidence. Route A target effects,
variances, attrition, favorable subgroups or opening results must never update the
Route B margin, PSU construction, sample size or decision rule.

### 2.2 Prohibited before the opening boundary

Before a separate opening authorization, no worker, analyst or model-selection
process may:

- read, request, list, probe or cache target-period WTEMP values;
- inspect target-period WTEMP availability, missingness, qualifier patterns,
  event rates, series conflicts or any derivative that reveals outcome presence;
- use target error, target completeness or target strata to select, replace,
  rank or exclude a site/component;
- tune a threshold, margin, history gate, block length, model, hyperparameter,
  calibration rule, headline or sensitivity from target results;
- use a Route A cache, receipt, model result or opened effect as Route B trial or
  design evidence;
- activate a reserve after any target request has begun.

If a provider endpoint mixes permitted metadata/predictors with target WTEMP, the
endpoint remains outcome-bearing. A filtered view created after retrieval is not
an outcome-free acquisition.

### 2.3 Separation of authority

The scientific owner/user chooses the scientific question. An independent
custodian verifies evidence separation and controls access. The custodian must
not silently choose scientific margins, arms or estimands, and the analyst must
not self-issue an opening receipt.

| Authority | Required responsibility |
| --- | --- |
| Scientific owner/user | Select the population, geography, task arm, target period/year scope, estimand, hypothesis family, scientific margins, power/precision targets, comparator policy, contribution class and any primary-domain restrictions |
| Independent custodian | Hold or mediate target credentials/paths; verify target paths were absent or inaccessible during design; verify hashes, chronology and external timestamp; independently replay selection and freeze receipts; reject an incomplete opening request |
| Dual sign-off | Approve the final four-stage freeze manifest and, later, a separate exact target-acquisition request ledger |
| Analyst | Implement only the sealed decisions, retain failures, and make no post-opening discretionary substitutions |

“Independent” must be defined before freeze. At minimum, the custodian must not
have participated in outcome-guided model/cohort selection and must have an
auditable identity, timestamp and signed/self-hashed receipt.

## 3. Decisions that remain open

Each decision below must be resolved explicitly. A default or recommendation is
not a selection. An omitted row is a freeze failure.

### D01 — Inferential target and population (user)

Choose whether Route B is intended to support a design-based claim for the
**eligible monitored-component frame** or only a fixed sampled-component
description. The current recommended inferential population is eligible monitored
graph-disconnected drainage-network components, not all U.S. rivers, all river
days or ungauged basins generally.

If design weights and positive inclusion probabilities cannot be implemented,
the only allowed closure is the fixed-frame descriptive branch.

### D02 — Geography and frame exclusions (user)

Choose CONUS only versus inclusion of Alaska, Hawaii and territories, and freeze
all outcome-free domain exclusions. Route A sites and every network component
containing a Route A site must be excluded. Geography cannot be expanded after
target access to restore power.

### D03 — Primary task arm/information set (user)

Choose exactly one primary arm:

1. **Known-gauge retrospective:** target-site WTEMP/FLOW observed through issue
   time; claim is history-dependent hindcast.
2. **Strict ungauged:** no target-site WTEMP in fitting, transforms, selection or
   issue-time inputs; claim is ungauged transfer.
3. **Operational replay:** archived as-issued observations, NWP vintages and
   latency/state as they existed at issue time; claim is operational only if the
   complete archive exists.

The arms may be separately registered, but one cannot rescue or rename another.
Masking target history from a model selected for target-history use does not
create a fair strict-ungauged arm.

### D04 — Untouched period and year scope (user)

Choose the exact untouched period and one of:

- a fixed-period estimand conditional on the named years; or
- a year-superpopulation estimand with enough independent years and a frozen
  small-sample/year-resampling method.

The sampling draft requires at least three untouched years, or a separately
justified untouched retrospective period plus a prospective season. Three years
alone are weak evidence for a year-superpopulation claim.

### D05 — Point estimand and weighting (user)

The recommended point estimand is the design-weighted mean paired RMSE difference,
negative favoring the candidate. Site effects are first estimated inside PSU×year,
years are equally weighted under the fixed-period branch, and PSUs use normalized
inverse inclusion probabilities. Equal PSU weights are valid only for a verified
self-weighting design.

Choose this estimand or predeclare another robust functional before power work.
Route A's station median is not inherited automatically.

### D06 — Hypothesis family, margins and multiplicity (user)

For each comparison/horizon, select superiority, non-inferiority or descriptive
estimation; identify the reference and direction; supply the scientific source
for every margin; and freeze familywise error control. Route A's `+0.05 °C`
ceiling, storage precision and the power formula do not provide a Route B
scientific margin.

### D07 — Assurance, power and precision (user)

Select:

- required power (`0.80` or `0.90` are draft grid values);
- PSU-retention assurance `A`;
- maximum CI half-width `W` in °C;
- family-adjusted planning alpha;
- the permissible upper bound on selected PSUs.

The draft search is `45 <= K <= 60` and chooses the smallest K meeting assurance,
power and precision while retaining at least 30 reportable PSUs. If K=60 fails,
the design is not feasible under the current frame/rules; no gate or margin may be
relaxed in response.

### D08 — Primary flow/tidal domain and calendar semantics (user)

Choose whether the primary population includes tidal/backwater/negative-flow
sites. If a positive-flow non-tidal restriction is necessary for an official
process comparator, it must enter the frame now and the resulting claim must stay
inside that domain. Negative flows remain in raw evidence and are not silently
clipped or renamed. Freeze the local-day/UTC alignment rule and its sensitivity.

### D09 — Comparator matrix and budget (user)

Select the current strong probabilistic comparator, the exact eligible models per
arm and a common resource policy. The current draft budget is 40 search trials,
3 tuning seeds and 5 final seeds per learned family, with the same search
algorithm and validation objective plus predeclared hardware-hour caps. These are
design values, not an approved allocation.

Current candidate evidence is:

- official Air2stream is pinned in the provenance record at commit
  `d4834bccf01657c03ab60efb4c18f8a256132c53`, but its source build/reference case
  remains `BLOCKED_NO_COMPILER / REFERENCE_CASE_NOT_ATTESTABLE`; its
  CC-BY-SA-3.0 evidence and exact build must be bound, and it is eligible only in
  a known-gauge arm after that blocker closes, or in an operational arm when
  as-issued forcing and issue-time state also pass their contracts;
- `neuralforecast==3.2.0` N-HiTS is the preferred candidate for a
  development-only budgeted pilot, not a selected comparator;
- `pytorch-forecasting==1.8.0` TFT is a proposed sensitivity candidate, not a
  selected comparator;
- a strict-ungauged exogenous-only derivative requires its own architecture,
  training-network and zero-history contract. It is not official Air2stream,
  N-HiTS or TFT by substitution.

### D10 — Contribution class and identifiability (user)

Choose:

- **predictive-only (recommended):** structured statistical predictor with
  neutral latent-variable language; or
- **mechanistic:** an identifiable heat-balance formulation plus independent
  heat flux, geometry, velocity/travel time, groundwater, shading and reservoir
  state measurements.

An analytic compensation direction, rank deficiency, flat profile or unstable
latent automatically yields `PREDICTIVE_ONLY_LATENTS_NOT_IDENTIFIED`. Prediction
benchmarking may continue, but learned κ, equilibrium, router weights and experts
cannot be physical findings.

### D11 — History, QA, missingness and calibration rules (user)

Using training/calibration evidence only, choose the minimum observed-history
fraction, maximum gap, drift/flatline/change-point tests, duplicate-series rule,
approval/qualifier primary rule, registered event classes, calibration inclusion
and missingness sensitivity parameters. The per-horizon calibration predicate
must equal the scoring predicate for that horizon, restricted to calibration
dates.

### D12 — Space×time UQ branch (user/statistical reviewer)

Select the fixed-year or year-superpopulation branch, block-length rule, bootstrap
failure ceiling, replicate-weight method and number of draws. The current draft
uses 20,000 draws and a 28-day synchronized moving block with 14- and 56-day
sensitivities. Those values remain unselected and must be justified from
training/calibration residual dependence, not target results.

### D13 — Custody, seal and opening mechanics (custodian; dual sign-off)

Name the custodian; define credential and target-path isolation; select the
external timestamp/signature mechanism; define independent replay; and specify
the exact one-time request scope. Completion of this row does not authorize
opening. The user must later issue a separate explicit target-acquisition
authorization after the clean-room gate passes.

## 4. Evidence gaps before any freeze can pass

| Gap ID | Missing evidence | Current status | Fail-closed consequence |
| --- | --- | --- | --- |
| G01 | Immutable WBD/NHDPlus releases, complete edge rules, diversions/artificial paths and source hashes | Not assembled | No PSU identities or independence language |
| G02 | Candidate frame, graph snaps, ambiguity ledger, Route A component-overlap audit | Not created | No sampling or feasibility claim |
| G03 | Positive inclusion probabilities, public seed, primary/reserve order and replay | Not created | No design-based estimand |
| G04 | Outcome-free power/assurance grid with MCSE, hard/null/informative-attrition cells and daily-error reconstruction | Not run | K cannot be selected; feasibility unknown |
| G05 | Scientific margin source, hypothesis family, alpha and precision threshold | Not selected | No power claim or directional decision rule |
| G06 | Primary arm, information set, target period and year scope | Not selected | No eligible model matrix or claim |
| G07 | Official Air2stream source-build/reference-case receipt | Blocked | Cannot be called a verified official comparison |
| G08 | Development-only N-HiTS/TFT pilot for covariate semantics, determinism, quantiles and compute | Not run | Modern comparator remains a candidate |
| G09 | Fully hashed dependency/license lock and complete trial/compute budget | Not created | No fair frozen suite |
| G10 | End-to-end °C↔°F training/prediction equivalence and identifiability verdict | Not run | Unit and mechanism claims blocked |
| G11 | Provider evidence that raw bytes, series IDs, qualifiers, approval, methods/instruments and local-day rules can be retained | Not demonstrated | Measurement contract not executable |
| G12 | Frozen history, mask propagation/fill-invariance, duplicate-series, negative-flow and day-boundary rules | Not implemented or tested | Model suite blocked |
| G13 | Per-horizon calibration registry design and target-free prediction binding | Not implemented | Probability/calibration claim blocked |
| G14 | Missingness model/bounds and crossed space×time bootstrap validated on synthetic/development data | Not implemented | Primary UQ not established |
| G15 | Four freeze manifests, external timestamp and independent replay | Not created | No preregistration or chronology claim |
| G16 | Clean-room reproduction of all pre-label artifacts and absence/inaccessibility attestation for target paths | Not performed | Opening prohibited |

No missing item may be converted to “passed” by the absence of an observed error.
Each needs affirmative, replayable evidence.

## 5. PSU, assurance and balance gates

### 5.1 Frame construction gate

The primary PSU is one complete frozen graph component; multiple sites in a
component do not create additional PSUs. The frame passes only if:

1. every selected site has one resolved, non-ambiguous snap;
2. the declared graph inputs are complete, with no unresolved diversion,
   artificial path or known cross-basin connection;
3. Route A site and component overlap both equal zero;
4. every selected component/site has a positive, replayable inclusion
   probability and a normalized design weight;
5. primary and reserve selection replay exactly from the frozen seed;
6. all reserves are either activated for a predeclared metadata/topology failure
   before target acquisition or remain permanently inactive.

Passing this gate records “no encoded connection in the audited graph,” not
statistical independence. Shared weather, omitted diversions, groundwater,
reservoir operations and management remain sources of dependence handled by the
UQ design and limitations.

### 5.2 Pre-opening sample-size and balance gate

Let `K_required = max(45, K_power)`, with `K_power` determined by the frozen
assurance/power/precision rule. Before opening:

- `K_required <= 60` under the current draft bound;
- `K_selected = K_required`;
- expected reportability satisfies the selected assurance of retaining at least
  30 PSUs for every formal comparison/horizon;
- `K_effective / K_selected >= 0.75`, where
  `K_effective = 1 / sum(normalized_weight_share^2)`;
- `largest_weight_share < 0.25`;
- graph/input completeness and Route A exclusion gates pass.

If the power-derived K exceeds 60, the current design is `FRAME_INFEASIBLE`. The
upper bound may be reconsidered only while outcomes remain inaccessible and only
through a newly reviewed design, never because a desired result needs rescue.

### 5.3 Analysis-time gate

Every comparison×horizon cell independently requires:

- `K_reportable >= 30`;
- `K_effective / K_reportable >= 0.75` under the actual cell weights;
- `largest_weight_share < 0.25`;
- the frozen minimum keys/history rule and an estimable UQ result.

A failed cell is `NOT_ESTIMABLE`. It cannot borrow PSUs from another horizon,
activate reserves, change weights, merge components or use a sensitivity as the
primary result. If the frame itself cannot meet the gates, close Route B as a
monitored fixed-frame descriptive study rather than inventing HUC4/HUC8,
distance-bin or singleton PSUs after the fact.

## 6. Comparator fairness and model-claim gates

### 6.1 Common information and keys

Within a formal arm, every model must use the same issue/target keys, training
years, development clusters, issue-time information and predictor vintage. A
model may not receive future meteorology, target-site history or a more complete
key set because its implementation expects it. Ineligible models are reported as
`NOT_ELIGIBLE`; they are not silently adapted.

The immutable model row binds implementation/package hashes, license evidence,
search-space hash, algorithm, trial/compute caps, hardware, tuning and final seeds,
parameter count, validation metric and common-key registry.

### 6.2 Equal opportunity, not equal rhetoric

- The same trial algorithm, validation objective and accounting rules apply to
  every learned family.
- Every model stops at its trial cap or hardware-hour cap; unused resources do
  not transfer.
- All registered final seeds are averaged; best-seed selection is prohibited.
- Failed, timed-out and unfavorable trials remain in the ledger.
- Runtime, memory, parameters and all search results are reported.
- A capacity-matched attribution control is a separate claim and must satisfy a
  frozen parameter band (draft `±2%`); equal trial counts alone do not establish
  capacity matching.

### 6.3 Unit and identifiability gates

The objective should be unit-covariant, using either dimensionless z-space or
correctly scaled raw-temperature losses, but never both normalizations at once.
An end-to-end °C↔°F replay must produce inverse-transformed predictions within the
frozen tolerance (draft `1e-5 °C`); scalar loss equality is insufficient.

Mechanism wording remains closed unless analytic reparameterization, Jacobian
rank/conditioning, profile loss and multi-seed latent stability all pass and
independent physical-state validation exists. Architecture ablation is not a
mechanism test.

## 7. Measurement, missingness and space×time UQ gates

### 7.1 Measurement QA

Every retained issue/target row must trace to raw transaction bytes, a unique
time-series identity, parameter/statistic code, unit, qualifier, approval status
and local-day rule. Missing method, instrument or calibration history is stored as
null with a reason. Conflicting series follow one frozen outcome-independent rule,
and the full conflict ledger remains visible.

Negative FLOW remains raw evidence. The default draft transform is signed
`asinh`; tidal/backwater/negative-flow and local-day/UTC boundary results are
predeclared sensitivities, not reasons to rebuild the cohort after opening.

### 7.2 Opportunity, history and imputation

The calendar-opportunity and attrition registries must distinguish all possible
issue/target dates from observed-history, observed-target, common-key and
reportable subsets. Every table/headline reports component, site, year and key
counts for its actual estimand.

The history fraction and maximum-gap thresholds are frozen from
training/calibration evidence. Every branch consuming an imputed value must
consume/propagate its mask. A metamorphic test changes every missing fill while
holding observed values and masks fixed; predictions must remain invariant within
the frozen deterministic tolerance. Failure blocks the suite.

Missing keys are never counted as correct days. Primary results remain conditional
on the frozen observable/history gate.

### 7.3 Missingness sensitivities

Predeclare cross-fitted stabilized IPW under an explicit MAR model,
δ-pattern-mixture shifts, physically bounded worst-case/tipping calculations and
year/season/site/component attrition maps. These are descriptive assumption
analyses. MAR/IPW does not prove MAR, and no sensitivity can rescue a failed
primary gate or be selected by favorability.

### 7.4 Crossed uncertainty

The primary bootstrap must reconstruct the daily candidate/reference errors and
RMSE on every draw, use the frozen design/strata and replicate weights, and apply
synchronized calendar blocks to all components so shared events are preserved.
It must retain non-estimable draws and enforce a frozen maximum failure rate.

With a fixed three-year target, the resulting inference is conditional on those
years. More bootstrap draws do not create independent years. A
year-superpopulation statement requires additional years and the separately
frozen year method. Graph disconnection does not remove the need for this
space×time treatment.

Probability reporting includes interval score, each emitted quantile's pinball
loss, coverage and width by registered strata, Brier/reliability/calibration slope
where an event is predeclared, and calibration/attrition counts. Three-quantile
pinball is not CRPS, and marginal conformal coverage is not distribution-free
conditional coverage.

## 8. Falsifiable headline register

The owner must select a small primary family before freeze. Each template below
has an automatic kill condition; multiplicity correction applies to every
selected directional headline.

| ID | Candidate headline template | Minimum evidence | Automatic closure if not met |
| --- | --- | --- | --- |
| H1 | “For the sampled eligible monitored components during the fixed untouched period, ThermoRoute improved design-weighted RMSE over [reference] in the [arm] information set.” | Passed PSU/UQ gates; exact common keys; frozen superiority boundary; family-adjusted interval/test crosses the registered favorable boundary | “The study did not establish improvement”; no trend, subgroup or sensitivity rescue |
| H2 | “ThermoRoute was not worse than the selected modern comparator by more than the scientific margin [m].” | Comparator provenance/fair-budget receipt; margin source; upper confidence bound below m; all cell gates pass | “Non-inferiority was not established”; never convert to parity/equivalence |
| H3 | “The bounded correction reduced registered worst-decile/OOD error without sacrificing average accuracy beyond [m].” | Capacity-matched unbounded control; predeclared tail/OOD strata; both tail and average decision rules pass | Bound remains an algebraic output constraint, not demonstrated utility |
| H4 | “The selected probabilistic model improved [interval score/pinball/Brier] while meeting the registered marginal coverage calibration criterion.” | Frozen distribution/quantiles/event; per-horizon calibration; coverage plus width and reliability; complete attrition counts | Report under/overcoverage and score failure; no calibrated-probability headline |
| H5 | “Performance transferred to strict-ungauged disconnected networks without target-site temperature history.” | Strict-ungauged arm selected before search; no target WT in fit/transforms/selection/input; exogenous comparator fairness; complete component holdout | Claim remains known-gauge or descriptive transfer; site-ID disjointness cannot substitute |
| H6 | “[Latent quantity] tracked an independently measured physical process.” | Mechanistic route selected; identifiable formulation; all latent gates; independent physical measurements and held-out validation | `PREDICTIVE_ONLY_LATENTS_NOT_IDENTIFIED`; delete mechanistic/driver wording |

Regardless of numerical results, the literature record prohibits “first
strict-ungauged,” “first continental/network,” “first probabilistic/operational”
and generic cross-study SOTA claims. Direct rankings are also prohibited when
target aggregation, horizon, information set, spatial split or probability
representation differ.

## 9. Four-stage freeze, clean-room gate and one-way opening

The following order is mandatory. A later stage cannot cure a failed earlier
stage, and none is authorized by this document.

### Freeze 1 — Scientific decision contract

Bind D01–D08 and D10: population, geography, primary arm, untouched period/year
scope, estimand, hypotheses/margins/multiplicity, power/assurance/precision,
primary domain and contribution class.

**Fail closed:** any missing choice yields `SCIENTIFIC_CONTRACT_INCOMPLETE`; do not
construct a selectively convenient final frame or model matrix.

### Freeze 2 — Frame, PSU and power contract

Build and replay the immutable topology/candidate frame, exclusions, probability
sample, reserves, inclusion probabilities, weights, power grid and pre-opening
balance/assurance gate.

**Fail closed:** unresolved graph inputs/snaps/Route A overlap, non-positive or
non-replayable probabilities, K above the frozen limit, or failed balance/power
yields `FRAME_INFEASIBLE`. Close to descriptive or redesign while outcomes remain
inaccessible; do not proceed to model selection as if strong inference survived.

### Freeze 3 — Comparator, QA, missingness and UQ contract

Bind D09–D12: arm-specific model matrix, implementations/licenses, information
sets, common keys, searches/resources/seeds, unit and identifiability tests,
measurement evidence, series/calendar/flow rules, history/fill masks,
per-horizon calibration, missingness sensitivities and crossed UQ.

**Fail closed:** missing comparator provenance, unequal information/resources,
unresolved measurement traceability, fill sensitivity, mismatched calibration
keys or unvalidated UQ yields `ANALYSIS_CONTRACT_INCOMPLETE`. No target request.

### Freeze 4 — Executable suite, predictor and custody contract

Using only frozen development inputs, close all registered trials and failures;
produce the model-suite receipt, predictor snapshot/acquisition request schema,
dependency lock, target-absence attestation and D13 custody/opening manifest.
No model, threshold, key rule or predictor transformation may remain mutable.

**Fail closed:** incomplete trials/receipts, dependency drift, replay mismatch,
target-path exposure or custody conflict yields `PRELABEL_EXECUTION_NOT_CLOSED`.

### Independent clean-room replay

In an isolated environment, the custodian/reviewer reproduces all four manifests,
PSU selection, power decisions, model/predictor identities and pre-label tests from
their frozen inputs. The replay verifies hashes and confirms that target outcomes
were absent/inaccessible throughout.

**Fail closed:** any discrepancy yields `CLEANROOM_REPLAY_FAILED`. Return to the
appropriate freeze under a new version while outcomes remain closed; never patch
a sealed artifact in place.

### Separate target-acquisition authorization

Only after all prior receipts pass may the user and custodian sign an exact
request ledger and the user issue a separate explicit authorization. That event
is the one-way opening boundary. It defines endpoints, parameter/statistic codes,
sites, dates, expected raw-response handling and credentials.

After the first target request:

- no station/component/reserve substitution is allowed;
- no change to model, margin, hypothesis, metric, event, QA/history gate, block,
  calibration or primary sensitivity is allowed;
- provider/measurement failures follow only the frozen failure codes;
- every analysis cell applies the ≥30/effective-balance/UQ gates independently;
- unfavorable and `NOT_ESTIMABLE` results remain reportable evidence.

## 10. Minimum contents of a future signed decision record

A future decision record is complete only when it contains:

```text
decision_package_version_and_sha256
scientific_owner_identity_and_signature
independent_custodian_identity_and_signature
primary_population_and_geography
primary_arm_and_information_set_id
untouched_period_and_year_scope
estimand_and_design_weights
hypothesis_family_margins_sources_alpha
assurance_power_precision_and_K_rule
domain_and_calendar_rules
model_matrix_and_budget_sha256
measurement_missingness_uq_contract_sha256
frame_power_and_balance_receipts
dependency_model_suite_predictor_receipts
target_absence_and_access-control_attestation
external_timestamp
cleanroom_replay_receipt
explicit_statement_target_acquisition_not_yet_authorized
```

The final line remains true even after the decision record is sealed. Opening is
a later, separate authorization.

## 11. Consistency crosswalk

| Source draft | Preserved decision/evidence boundary in this package |
| --- | --- |
| `ROUTE_B_GE30_CLUSTER_SAMPLING_FRAME_DRAFT.md` | Network-component PSU; HUC2 as stratum only; two-stage probability sample; Route A component exclusion; K/balance gates; pre-target reserve rule; graph disconnection caveat |
| `ROUTE_B_POWER_MDE_DRAFT.md` | Design-weighted paired RMSE; fixed-year versus year-superpopulation branch; daily-error power simulation; `45..60` draft search; assurance/power/precision choice; no Route A effect input |
| `ROUTE_B_MODEL_BUDGET_IDENTIFIABILITY_DRAFT.md` | Arm-specific common information/keys; draft `40/3/5` budget; official-baseline provenance gate; unit equivalence; predictive-only fail closure for unidentified latents |
| `ROUTE_B_MEASUREMENT_MISSINGNESS_UQ_DRAFT.md` | Raw transaction/series/qualifier traceability; flow/day semantics; opportunity and attrition registries; fill invariance; per-horizon calibration; descriptive missingness sensitivities; crossed synchronized-block UQ |
| `ROUTE_B_COMPARATOR_PROVENANCE_20260801.md` | Air2stream remains build/reference-blocked; N-HiTS/TFT remain candidates; arm eligibility and package evidence do not constitute selection/freeze |
| `LITERATURE_EVIDENCE_MAP_2021_2026.md` | Known-gauge, strict-ungauged and operational labels stay separate; graph disconnection is not independence; incompatible studies are not directly ranked; prohibited “first” and generic SOTA claims remain prohibited |

No numeric threshold or candidate status in this package upgrades a draft value to
a final choice. If a source draft is revised before a protocol is sealed, this
package must be versioned and reconciled; it must not silently override the more
specific evidence record.

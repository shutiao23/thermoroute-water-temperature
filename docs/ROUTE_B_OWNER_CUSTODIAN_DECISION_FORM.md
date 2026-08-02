# Route B owner/custodian pre-label decision form

| Field | Value |
| --- | --- |
| Form date | 2026-08-02 |
| Form status | **UNSIGNED / ALL DECISIONS OPEN / OUTCOME-FREE / NOT SEALED** |
| Governing package | `docs/ROUTE_B_PRELABEL_DECISION_PACKAGE.md` |
| Governing package SHA-256 | `47674176aaafe47c4b50fd0c4af1264c9e555e980fa5c90746288804799d15ef` |
| Purpose | Record owner, custodian and statistical-reviewer choices needed before the four-stage Route B freeze |
| Non-authority | **THIS FORM DOES NOT AUTHORIZE TARGET ACQUISITION, TARGET ACCESS, A PROTOCOL FREEZE OR MODEL EXECUTION** |

This is a blank decision instrument. Every option box is intentionally unchecked.
Bold “advisory recommendation” text records design advice only; it is not a
choice, default, approval or proxy signature. The named signer must select
exactly one mutually exclusive option for every decision, complete all associated
fields, date the choice and initial it. A custom value is valid only where an
option explicitly permits it and the value and evidence source are supplied
before freeze.

This form must be completed without reading, requesting, listing, probing or
caching target-period WTEMP values or any target-period availability,
missingness, qualifier, event-rate or conflict derivative. Route A outcomes,
effects, attrition and favorable subgroups are not Route B design evidence.

## How to use this form

1. Resolve D01–D13 in order. Check exactly one box in each decision and copy its
   letter into the `Selected option` field.
2. Complete every required free-text or numeric field. “TBD,” an empty field, two
   checked boxes or an unsigned selection all count as blank.
3. Attach immutable, hashed evidence for every applicable G01–G16 row. An
   unchecked evidence box never becomes a pass merely because no error was seen.
4. The statistical reviewer reviews D05–D07 and co-selects D12. The custodian
   selects D13 with owner dual sign-off and independently verifies the custody
   and clean-room evidence.
5. Signing this form closes choices only. It cannot cross the one-way opening
   boundary. Target acquisition requires a later, separate, exact request ledger
   and explicit owner/custodian authorization after all freezes and clean-room
   replay pass.

## D01 — Inferential target and population

**Required selecting/signing subject:** scientific owner/user.

**Advisory recommendation only:** option A, because it preserves the intended
design-based Route B claim while keeping the population inside the eligible
monitored-component frame. This advice is not a selection.

| Select one | Mutually exclusive option | Scientific and computational impact |
| --- | --- | --- |
| [ ] A | Design-based inference to eligible monitored, graph-disconnected drainage-network components with positive inclusion probabilities and design weights | Supports population inference only if frame, sampling and space×time UQ gates pass; requires graph construction, probability sampling, replicate weights and replay |
| [ ] B | Fixed sampled-component description only | Forfeits design-based population inference; lowers sampling/weighting burden but still requires honest cohort, common-key and UQ reporting |

Selected option: `____`  Population definition/version: `____________________`

Owner initials/date: `____________________`

**If blank:** `SCIENTIFIC_CONTRACT_INCOMPLETE`; no final frame, design-based
estimand, power claim or population headline may be constructed.

## D02 — Geography and frame exclusions

**Required selecting/signing subject:** scientific owner/user.

**Advisory recommendation only:** option A (CONUS only) until immutable topology,
provider and measurement evidence establishes equal frame coverage elsewhere.
This narrower feasibility recommendation is not a selection.

| Select one | Mutually exclusive option | Scientific and computational impact |
| --- | --- | --- |
| [ ] A | CONUS only | Bounds topology and provider heterogeneity and reduces frame/QA cost; claims cannot extend to Alaska, Hawaii or territories |
| [ ] B | CONUS, Alaska, Hawaii and named U.S. territories: `________________` | Broadens the claim but requires complete frozen topology, coverage, calendar and measurement contracts for every named geography; materially increases frame and QA work |

Mandatory exclusions under every option: every Route A site and every network
component containing a Route A site. Additional outcome-free exclusions and
sources: `____________________________________________________________`

Selected option: `____`  Owner initials/date: `____________________`

**If blank:** `SCIENTIFIC_CONTRACT_INCOMPLETE`; geography cannot be expanded or
contracted after target access, and no candidate frame may be promoted.

## D03 — Primary task arm and information set

**Required selecting/signing subject:** scientific owner/user.

**Advisory recommendation only:** option A is the lowest-evidence-risk starting
point if the scientific question genuinely concerns history-dependent hindcast.
It must not be chosen merely because it is easier, and this advice does not select
or rename the intended scientific task.

| Select one | Mutually exclusive option | Scientific and computational impact |
| --- | --- | --- |
| [ ] A | Known-gauge retrospective: target-site WTEMP/FLOW observed only through issue time | Claim is history-dependent hindcast; permits arm-eligible history models and, after provenance closure, official Air2stream; requires strict issue-time leakage controls |
| [ ] B | Strict ungauged: no target-site WTEMP in fitting, transforms, selection or issue-time inputs | Supports only genuine ungauged transfer; requires component holdout, a purpose-built exogenous architecture/training network and zero-history attestation; known-gauge comparators are `NOT_ELIGIBLE` |
| [ ] C | Operational replay using complete archived as-issued observations, NWP vintages, latency and issue-time state | Supports an operational claim only if the complete contemporaneous archive exists; highest storage, provenance and replay cost; reconstructed modern forecasts are insufficient |

Selected option: `____`  Primary arm/information-set ID: `____________________`

Issue-time cutoff and permitted input classes: `______________________________`

Owner initials/date: `____________________`

**If blank:** `SCIENTIFIC_CONTRACT_INCOMPLETE`; there is no eligible model matrix,
information contract, operational/ungauged label or primary claim.

## D04 — Untouched period and year scope

**Required selecting/signing subject:** scientific owner/user.

**Advisory recommendation only:** option A unless enough genuinely independent
untouched years and a justified small-sample year method support option B. Three
years alone are weak evidence for a year-superpopulation claim.

| Select one | Mutually exclusive option | Scientific and computational impact |
| --- | --- | --- |
| [ ] A | Fixed untouched-period estimand conditional on the exact named years | Inference is conditional on those years; uses equal year weighting under the current recommended estimand; crossed calendar-block UQ remains required |
| [ ] B | Year-superpopulation estimand | Requires enough independent years and a frozen small-sample/year-resampling method; adds a year uncertainty layer and cannot be created by increasing bootstrap draws |

Exact untouched start/end dates: `________________` to `________________`

Named years/seasons and time zone/calendar basis: `___________________________`

For option B, independent-year rationale and method: `________________________`

If fewer than three untouched years are proposed, separately justified untouched
retrospective period plus prospective-season plan: `__________________________`

Selected option: `____`  Owner initials/date: `____________________`

**If blank:** `SCIENTIFIC_CONTRACT_INCOMPLETE`; target dates may not be requested,
and neither fixed-period nor year-superpopulation inference is eligible.

## D05 — Point estimand and weighting

**Required selecting/signing subject:** scientific owner/user; statistical
reviewer concurrence required.

**Advisory recommendation only:** option A, the package's recommended point
estimand. It is not inherited from Route A and is not selected by appearing here.

| Select one | Mutually exclusive option | Scientific and computational impact |
| --- | --- | --- |
| [ ] A | Design-weighted mean paired RMSE difference, candidate minus reference; site effects within PSU×year, equal years for fixed-period inference, normalized inverse-inclusion-probability PSU weights | Negative values favor the candidate; requires daily-error reconstruction, exact paired keys, design/replicate weights and PSU×year bookkeeping |
| [ ] B | The same paired RMSE functional with equal PSU weights, permitted only after a verified self-weighting-design receipt | Simpler computation, but scientifically invalid without affirmative self-weighting evidence; absence of unequal weights is not proof |
| [ ] C | Another predeclared robust functional: `________________` with direction `________________` and weighting `________________` | Changes power, interpretation and UQ; must be fully specified and simulated outcome-free before K selection; Route A's station median is not inherited |

Formal estimand notation/implementation ID: `_______________________________`

Selected option: `____`  Owner initials/date: `____________________`

Statistical reviewer initials/date: `____________________`

**If blank:** `SCIENTIFIC_CONTRACT_INCOMPLETE`; no effect, power grid, confidence
interval or directional headline may be defined.

## D06 — Hypothesis family, margins and multiplicity

**Required selecting/signing subject:** scientific owner/user; statistical
reviewer concurrence required.

**Advisory recommendation only:** option A for a small, scientifically primary
family when improvement is the real question and zero is the registered boundary.
Choose B only with an external scientific margin source; use C when no defensible
directional boundary exists. This is advice, not a choice.

| Select one | Mutually exclusive option for the primary family | Scientific and computational impact |
| --- | --- | --- |
| [ ] A | Superiority | Requires frozen reference, direction and family-adjusted favorable boundary; supports an improvement claim only when the adjusted rule passes |
| [ ] B | Non-inferiority | Requires an externally justified scientific margin for every comparison/horizon; failure cannot be called parity or equivalence |
| [ ] C | Descriptive interval estimation | Forfeits confirmatory directional claims; still requires frozen estimand, multiplicity/reporting policy and valid UQ |

Complete one row per formal comparison×horizon; attach more rows if needed.

| Comparison ID | Horizon | Reference | Direction | Margin/boundary and unit | Scientific source | Family | Planning alpha | Multiplicity method |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `________` | `____` | `________` | `____` | `________` | `________` | `____` | `____` | `________` |
| `________` | `____` | `________` | `____` | `________` | `________` | `____` | `____` | `________` |

Selected primary headline IDs from H1–H6, if any: `________________`  (A headline
template is not a result.)

Selected option: `____`  Owner initials/date: `____________________`

Statistical reviewer initials/date: `____________________`

**If blank:** `SCIENTIFIC_CONTRACT_INCOMPLETE`; G05 remains open, no power or
directional decision rule exists, and Route A's `+0.05 °C` ceiling or storage
precision may not be substituted as a scientific margin.

## D07 — Assurance, power and precision

**Required selecting/signing subject:** scientific owner/user; statistical
reviewer concurrence required.

**Advisory recommendation only:** option B (0.90 power) for the primary family if
the frame remains feasible; 0.80 and 0.90 are draft grid values, not approved
allocations. The recommendation does not relax the assurance, precision or
balance gates.

| Select one | Mutually exclusive planning-power option | Scientific and computational impact |
| --- | --- | --- |
| [ ] A | Required power = 0.80 | Lower K may be feasible but tolerates a higher false-negative risk; still requires outcome-free Monte Carlo assurance/precision evidence and MCSE |
| [ ] B | Required power = 0.90 | Stronger sensitivity target, typically more PSUs and simulation/collection cost; infeasibility at the cap must be reported rather than repaired post hoc |
| [ ] C | Another justified power target = `________` | Requires a written scientific/statistical rationale and a newly reviewed outcome-free planning grid before freeze |

Required PSU-retention assurance `A`: `________`

Maximum CI half-width `W` (°C): `________`

Family-adjusted planning alpha: `________`

Permissible selected-PSU range/cap: lower `________`; upper `________`

K rule: smallest K meeting assurance, power and precision while retaining at
least 30 reportable PSUs: [ ] yes  [ ] no (if no, exact alternative: `________`)

Selected option: `____`  Owner initials/date: `____________________`

Statistical reviewer initials/date: `____________________`

**If blank:** `SCIENTIFIC_CONTRACT_INCOMPLETE`; K is unselected and feasibility
is unknown. If the frozen upper cap fails, status is `FRAME_INFEASIBLE`; no gate,
margin or power target may be relaxed in response to target evidence.

## D08 — Primary flow/tidal domain and calendar semantics

**Required selecting/signing subject:** scientific owner/user.

**Advisory recommendation only:** option A preserves the broader monitored-domain
question and avoids discarding difficult flow regimes, while marking ineligible
comparators `NOT_ELIGIBLE`. If the scientific priority is an official comparator
that requires positive, non-tidal flow, option B must be chosen before the frame.

| Select one | Mutually exclusive primary-domain option | Scientific and computational impact |
| --- | --- | --- |
| [ ] A | Include eligible tidal, backwater and negative-flow sites in the primary population | Broader monitored-domain claim; requires signed `asinh` or another frozen transform, regime strata/sensitivities and models that can consume these values; some process comparators may be `NOT_ELIGIBLE` |
| [ ] B | Restrict the primary population to verified positive-flow, non-tidal sites | May enable an eligible official process comparator; narrows the scientific claim and requires outcome-free frame exclusion evidence |

Primary local-day/UTC alignment rule: `______________________________________`

Registered day-boundary sensitivity: `______________________________________`

Raw negative FLOW retained without silent clipping/renaming: [ ] yes

Selected option: `____`  Owner initials/date: `____________________`

**If blank:** `SCIENTIFIC_CONTRACT_INCOMPLETE`; frame eligibility, measurement
semantics and comparator eligibility are undefined.

## D09 — Comparator matrix and common resource budget

**Required selecting/signing subject:** scientific owner/user.

**Advisory recommendation only:** option A, because the package identifies N-HiTS
as the preferred candidate for a development-only budgeted pilot. It remains
unselected unless the owner checks A after that pilot closes its gates.

| Select one | Mutually exclusive comparator-policy option | Scientific and computational impact |
| --- | --- | --- |
| [ ] A | N-HiTS (`neuralforecast==3.2.0`) as the strong probabilistic primary comparator, contingent on a passed development-only pilot | Preferred package candidate, but requires covariate-semantics, determinism, quantile and compute evidence plus common information/keys and equal search accounting |
| [ ] B | TFT (`pytorch-forecasting==1.8.0`) as the strong probabilistic primary comparator, contingent on a passed development-only pilot | Higher architecture/implementation burden; requires the same pilot, information, quantile, determinism and resource gates and cannot be selected as a favorable post hoc substitute |
| [ ] C | Another current strong probabilistic primary comparator: `________________`, version/hash `________________` | Requires outcome-free provenance, license, pilot and fair-budget evidence equivalent to A/B before selection; changing it later requires a new pre-label version |

Exact arm-specific model matrix and implementation/package hashes:

| Model/family | Role | Eligible D03 arm | Source/package/version hash | License evidence | Search-space hash | Status gate |
| --- | --- | --- | --- | --- | --- | --- |
| `________` | `________` | `________` | `________` | `________` | `________` | `________` |
| `________` | `________` | `________` | `________` | `________` | `________` | `________` |
| `________` | `________` | `________` | `________` | `________` | `________` | `________` |

Common resource policy: search algorithm `________`; validation objective
`________`; search trials/family `________`; tuning seeds `________`; final seeds
`________`; hardware-hour cap/family `________`; hardware `________`.

The package's `40 trials / 3 tuning seeds / 5 final seeds` values are: [ ] adopted
after review  [ ] rejected and replaced above. Neither box is preselected.

All final seeds averaged; failures/timeouts/unfavorable trials retained; unused
budget non-transferable: [ ] yes

Selected option: `____`  Owner initials/date: `____________________`

Air2stream role, if any: `________________`. It is eligible only in the selected
arm/domain and only after G07 passes; it cannot replace the selected strong
probabilistic comparator. Strict-ungauged use requires a separate exogenous-only
architecture and zero-history contract, not a renamed package model.

**If blank:** `ANALYSIS_CONTRACT_INCOMPLETE`; there is no fair model suite or
eligible comparator claim. Air2stream is not verified while G07 is blocked, and
N-HiTS/TFT remain candidates while G08 is open.

## D10 — Contribution class and identifiability

**Required selecting/signing subject:** scientific owner/user.

**Advisory recommendation only:** option A, matching the package's recommendation
and the current absence of independent physical-state/identifiability evidence.

| Select one | Mutually exclusive option | Scientific and computational impact |
| --- | --- | --- |
| [ ] A | Predictive-only structured statistical predictor with neutral latent-variable language | Permits prediction benchmarking; prohibits physical interpretation of learned κ, equilibrium, router weights and experts; requires ordinary model/unit tests but not a mechanistic headline |
| [ ] B | Mechanistic contribution | Requires an identifiable heat-balance formulation, analytic/Jacobian/profile/multi-seed gates and independent flux, geometry, velocity/travel-time, groundwater, shading and reservoir-state measurements; substantially increases data and validation cost |

Contribution wording and allowed latent terminology: `________________________`

Independent physical measurements/receipt IDs for option B: `________________`

Selected option: `____`  Owner initials/date: `____________________`

**If blank:** `SCIENTIFIC_CONTRACT_INCOMPLETE`. Any compensation direction, rank
deficiency, flat profile or unstable latent automatically closes mechanistic
language as `PREDICTIVE_ONLY_LATENTS_NOT_IDENTIFIED`, even if prediction continues.

## D11 — History, QA, missingness and calibration rules

**Required selecting/signing subject:** scientific owner/user.

**Advisory recommendation only:** option C: keep an observed/history-gated primary
analysis and allow a mask-aware imputed branch only as a registered sensitivity.
This preserves a legible primary estimand while testing missingness assumptions;
it is not a selection.

| Select one | Mutually exclusive primary data-handling option | Scientific and computational impact |
| --- | --- | --- |
| [ ] A | Observed/history-qualified rows only; no consumed imputation in primary or sensitivity models | Simplest fill semantics but may reduce reportability and generalizability; missing rows never count as correct and attrition/IPW/bounds still must be reported |
| [ ] B | Mask-aware imputation is part of the primary model contract | May preserve opportunity but requires every consumer to receive/propagate masks, frozen fit scope and fill-invariance metamorphic tests; failure blocks the suite |
| [ ] C | Observed/history-gated primary analysis with mask-aware imputation only in a predeclared sensitivity | Separates primary evidence from imputation assumptions; highest duplicated compute/registry burden and sensitivity cannot rescue a failed primary gate |

All fields below are mandatory regardless of option; use `not applicable` only
with a signed rationale.

| Rule | Frozen value, source and implementation/test ID |
| --- | --- |
| Minimum observed-history fraction by horizon | `________________________________________________` |
| Maximum gap and lookback window | `________________________________________________` |
| Drift, flatline and change-point tests/thresholds | `________________________________________________` |
| Duplicate/conflicting-series rule | `________________________________________________` |
| Approval/qualifier primary rule | `________________________________________________` |
| Registered event classes and thresholds | `________________________________________________` |
| Per-horizon calibration inclusion predicate | `________________________________________________` |
| Missingness/IPW model and stabilization/truncation | `________________________________________________` |
| δ-pattern-mixture grid | `________________________________________________` |
| Physical bounds and tipping rule | `________________________________________________` |
| Imputation fit scope, mask propagation and fill-invariance tolerance | `________________________________________________` |

Calibration predicate equals scoring predicate for each horizon, restricted to
calibration dates: [ ] yes

Selected option: `____`  Owner initials/date: `____________________`

**If blank:** `ANALYSIS_CONTRACT_INCOMPLETE`; history, QA, calibration and
reportability are undefined. Missing keys cannot be counted as correct, and no
post-opening threshold or series substitution is permitted.

## D12 — Space×time uncertainty branch

**Required selecting/signing subjects:** scientific owner/user and statistical
reviewer (joint selection).

**Advisory recommendation only:** option A when D04 selects a fixed period or only
three untouched years are available. The package's 20,000 draws and 28-day block
with 14-/56-day sensitivities are draft values, not selections.

| Select one | Mutually exclusive UQ option | Scientific and computational impact |
| --- | --- | --- |
| [ ] A | Fixed-year crossed space×time bootstrap, conditional on the named years | Reconstructs daily candidate/reference errors and RMSE on each draw with design/replicate weights and synchronized blocks; does not support a year-superpopulation statement |
| [ ] B | Year-superpopulation UQ with a separately frozen small-sample/year method plus crossed space×time dependence | Supports a broader year claim only with enough independent years; materially greater method, simulation and compute burden |

Block-length selection rule and development/calibration evidence: `____________`

Primary block length: `________` days; sensitivity lengths: `________________`

Bootstrap failure ceiling: `________`; number of draws: `________`

Replicate-weight method/design strata: `_____________________________________`

Non-estimable draws retained and synchronized calendar blocks applied to all
components: [ ] yes

Selected option: `____`

Owner initials/date: `____________________`

Statistical reviewer initials/date: `____________________`

**If blank:** `ANALYSIS_CONTRACT_INCOMPLETE`; primary UQ, coverage and
directional inference are unavailable. Graph disconnection does not license
independent-site or independent-day uncertainty.

## D13 — Custody, seal and opening mechanics

**Required selecting/signing subjects:** independent custodian selects the custody
implementation; scientific owner/user provides dual sign-off. The analyst cannot
self-issue an opening receipt.

**Advisory recommendation only:** option A where an institution can provide true
credential separation, immutable external timestamping and an isolated replay
environment. Independence must be evidenced, not inferred from a job title.

| Select one | Mutually exclusive custody implementation | Scientific and computational impact |
| --- | --- | --- |
| [ ] A | Institutionally independent named custodian holds target credentials/paths, externally timestamps all manifests and performs an isolated clean-room replay | Strong separation and auditability; requires institutional identity/access records, separate environment and replay resources |
| [ ] B | Independent external custodian/reviewer provides credential escrow, external signatures/timestamps and clean-room replay | Strong external separation; adds contractual, transfer, security and scheduling work and must preserve raw-response custody requirements |
| [ ] C | Internal dual-control custody with two named non-analyst approvers, technical deny controls and an independent replay reviewer | May be operationally feasible but requires affirmative conflict-of-interest, access-log, dual-control and reviewer-independence evidence; a shared analyst credential is prohibited |

Named custodian(s), affiliation and auditable identity: `______________________`

Why custodian/reviewer is independent: `_____________________________________`

Target credential/path isolation mechanism and deny evidence: `______________`

External timestamp/signature mechanism: `___________________________________`

Clean-room platform, inputs and independent replay method: `_________________`

Future one-time request schema (not an authorization): endpoint(s) `________`;
parameter/statistic codes `________`; site manifest hash `________`; exact dates
`________`; raw-response handling `________`; credential handle `________`.

Selected option: `____`

Custodian initials/date: `____________________`

Owner dual-sign-off initials/date: `____________________`

**If blank:** `PRELABEL_EXECUTION_NOT_CLOSED`; no custody/opening manifest or
target request is valid. Completing D13 still does **not** authorize opening.

## Central G01–G16 evidence checklist

Every item requires an immutable artifact or receipt, SHA-256, independent
verification identity/date and affirmative result. The package's current status
is recorded for orientation; it is not overwritten by a check mark. Attach a
versioned continuation sheet if one cell is too small.

| Gap | Evidence required | Package status at form creation | Artifact/receipt path and SHA-256 | Independent verifier/date | Check only after affirmative replay | Fail-closed consequence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G01 | Immutable WBD/NHDPlus releases, complete edge rules, diversions/artificial paths and source hashes | Not assembled | `________________` | `________________` | [ ] PASS | No PSU identities or independence language |
| G02 | Candidate frame, graph snaps, ambiguity ledger and Route A component-overlap audit | Not created | `________________` | `________________` | [ ] PASS | No sampling or feasibility claim |
| G03 | Positive inclusion probabilities, public seed, primary/reserve order and exact replay | Not created | `________________` | `________________` | [ ] PASS | No design-based estimand |
| G04 | Outcome-free power/assurance grid with MCSE, hard/null/informative-attrition cells and daily-error reconstruction | Not run | `________________` | `________________` | [ ] PASS | K cannot be selected; feasibility unknown |
| G05 | Scientific margin source, hypothesis family, alpha and precision threshold | Not selected | `________________` | `________________` | [ ] PASS | No power claim or directional decision rule |
| G06 | Primary arm, information set, target period and year scope | Not selected | `________________` | `________________` | [ ] PASS | No eligible model matrix or claim |
| G07 | Official Air2stream source-build/reference-case receipt | `BLOCKED_NO_COMPILER / REFERENCE_CASE_NOT_ATTESTABLE` | `________________` | `________________` | [ ] PASS | Cannot call it a verified official comparison |
| G08 | Development-only N-HiTS/TFT pilot for covariate semantics, determinism, quantiles and compute | Not run | `________________` | `________________` | [ ] PASS | Modern comparator remains a candidate |
| G09 | Fully hashed dependency/license lock and complete trial/compute budget | Not created | `________________` | `________________` | [ ] PASS | No fair frozen suite |
| G10 | End-to-end °C↔°F training/prediction equivalence and identifiability verdict | Not run | `________________` | `________________` | [ ] PASS | Unit and mechanism claims blocked |
| G11 | Provider evidence retaining raw bytes, series IDs, qualifiers, approval, methods/instruments and local-day rules | Not demonstrated | `________________` | `________________` | [ ] PASS | Measurement contract not executable |
| G12 | Frozen history, mask propagation/fill-invariance, duplicate-series, negative-flow and day-boundary rules | Not implemented or tested | `________________` | `________________` | [ ] PASS | Model suite blocked |
| G13 | Per-horizon calibration registry design and target-free prediction binding | Not implemented | `________________` | `________________` | [ ] PASS | Probability/calibration claim blocked |
| G14 | Missingness model/bounds and crossed space×time bootstrap validated on synthetic/development data | Not implemented | `________________` | `________________` | [ ] PASS | Primary UQ not established |
| G15 | Four freeze manifests, external timestamp and independent replay | Not created | `________________` | `________________` | [ ] PASS | No preregistration or chronology claim |
| G16 | Clean-room reproduction of all pre-label artifacts and target-path absence/inaccessibility attestation | Not performed | `________________` | `________________` | [ ] PASS | Opening prohibited |

Evidence reviewer notes and unresolved failures:

`__________________________________________________________________________`

`__________________________________________________________________________`

## Mandatory cross-decision consistency review

The signers must check these only after reviewing the completed selections:

- [ ] D01 population, D02 geography and D08 flow/tidal domain describe the same
  candidate frame and claim boundary.
- [ ] D03 arm exactly matches every D09 model's eligibility and information set;
  no known-gauge model has been relabelled strict-ungauged and no reconstructed
  forecast has been labelled operational.
- [ ] D04 year scope and D12 UQ branch match; a fixed-period design is not given a
  year-superpopulation interpretation.
- [ ] D05 estimand, D06 direction/margins and D07 outcome-free power simulation
  use the same comparison, horizon, daily-error reconstruction and weights.
- [ ] D07 K rule preserves at least 30 reportable PSUs per formal cell,
  `K_effective / K_reportable >= 0.75`, `largest_weight_share < 0.25`, and the
  frozen upper cap; failure is not repaired by redefining PSUs.
- [ ] D08 measurement/calendar semantics match D11 QA/history rules and D09
  comparator input capabilities.
- [ ] D10 contribution wording matches G10's identifiability verdict; no latent
  mechanism language is allowed on predictive-only closure.
- [ ] D11 calibration predicate equals the scoring predicate for each horizon on
  calibration dates, and every imputation consumer propagates a mask.
- [ ] D12 rebuilds daily errors/RMSE on each draw, preserves shared calendar
  events, retains non-estimable draws and uses the frozen design weights.
- [ ] D13 separates scientific choice from access control, binds exact immutable
  inputs and provides an independent clean-room replay.
- [ ] Route A sites/components, outcomes, effects, attrition and model results
  were not used to tune Route B margins, K, thresholds, cohort or model choice.
- [ ] No target-period outcome, availability, missingness, qualifier, conflict or
  derivative was read, requested, listed, probed or cached while completing this
  form.

Unresolved mismatch (any entry prevents signature): `________________________`

## Signatures and non-authorization attestation

### Scientific owner/user

I attest that the scientific choices recorded in D01–D12 are my explicit choices,
not defaults inferred from advisory recommendations. I understand the resulting
population, information-set, temporal, domain, estimand and claim limits. I have
not used Route A outcomes or Route B target-period outcome/availability evidence
to choose them.

Name/affiliation: `_________________________________________________________`

Signature: `________________________________`  Date/time/time zone: `________`

### Independent custodian

I attest that I meet the recorded independence definition, that target credentials
and paths were absent or inaccessible to design/model-selection workers, and that
I will reject incomplete or non-replayable freeze/opening requests. My signature
on this form does not grant target access.

Name/affiliation: `_________________________________________________________`

Signature: `________________________________`  Date/time/time zone: `________`

### Statistical reviewer

I attest that I reviewed D05–D07 and jointly selected D12; that their estimand,
margin/direction, multiplicity, power, assurance, precision, design weighting and
space×time UQ contracts are mutually consistent; and that no target-period result
was used in that review.

Name/affiliation: `_________________________________________________________`

Signature: `________________________________`  Date/time/time zone: `________`

### Joint target-acquisition status

By signing above, all parties affirm:

> **TARGET ACQUISITION IS NOT AUTHORIZED. TARGET ACCESS IS NOT AUTHORIZED.**
> **NO ENDPOINT, PATH, CREDENTIAL OR REQUEST MAY BE ACTIVATED BY THIS FORM.**

The statement remains true after this decision form is signed or sealed. Only
after all four freeze receipts, G01–G16 evidence, external timestamp and
independent clean-room replay pass may the owner and custodian consider a later,
separate exact target-acquisition request ledger. That later authorization must
be explicit and cannot be inferred from this document, a protocol, a signature,
a model-suite receipt or the absence of an observed error.

Owner initials acknowledging **NOT AUTHORIZED**: `________________`

Custodian initials acknowledging **NOT AUTHORIZED**: `________________`

Statistical reviewer initials acknowledging **NOT AUTHORIZED**: `____________`

Signed-form SHA-256 (computed only after all entries/signatures):
`__________________________________________________________________________`

External timestamp/signature receipt (does not authorize opening):
`__________________________________________________________________________`

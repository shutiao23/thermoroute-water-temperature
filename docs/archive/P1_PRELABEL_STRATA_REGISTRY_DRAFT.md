# P1-01 prelabel strata registry design

| Field | Value |
| --- | --- |
| Date | 2026-08-01 |
| Status | **DRAFT ONLY / OUTCOME-FREE / NOT A FROZEN PROTOCOL** |
| Primary scope | Route-B event, flow, season, year, qualifier and history-completeness reporting strata |
| Route-A use | Taxonomy may support descriptive reporting only; it cannot reopen Route-A inference eligibility |
| Outcome access | No target-period value, prediction, score, Stage-09 artifact or `outputs/**` path was read to prepare this draft |
| Unresolved sentinel | `UNRESOLVED_PRELABEL_DECISION` |
| Authority | Design document only; not a seal, target-access authorization, assignment receipt or analysis receipt |

This registry is the P1-01 design layer that is missing between the Route-B
sampling/model/UQ drafts and a reportable stratified analysis. It defines what
must be frozen before target access and how every opportunity is assigned after
access. It does not choose a favorable threshold, create a subgroup hypothesis or
authorize acquisition of any outcome.

The following existing boundaries remain controlling:

- the Route-B sampling unit, design weights and minimum-30 formal-cell gate come
  from `docs/ROUTE_B_GE30_CLUSTER_SAMPLING_FRAME_DRAFT.md`;
- measurement lineage, calendar opportunity, attrition, per-horizon calibration,
  missingness and crossed space×time UQ come from
  `docs/ROUTE_B_MEASUREMENT_MISSINGNESS_UQ_DRAFT.md`;
- information-set and model-budget parity come from
  `docs/ROUTE_B_MODEL_BUDGET_IDENTIFIABILITY_DRAFT.md`;
- every Route-A five-row result remains permanently
  `DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED` under
  `docs/B02_PERMANENT_DESCRIPTIVE_CLAIM_WORDING.md` and
  `docs/REPORTING_POLICY_DESCRIPTIVE_ONLY.md`.

## 1. Roles and non-negotiable estimand boundary

The unstratified, preregistered Route-B estimand remains the primary scientific
estimand. Strata have two distinct roles:

1. **Mandatory primary diagnostics:** opportunity, attrition, missingness and
   support counts by stratum. These diagnose what population the primary result
   actually covers; they do not create a second headline.
2. **Prelabelled descriptive secondary performance:** point/probability metrics
   and paired effects within frozen strata. These remain descriptive unless a
   stratum-specific hypothesis is separately added to the formal family, with its
   own scientific margin and multiplicity rule, before target access.

No stratum may rescue a failed overall gate, replace the primary cohort, trigger
site/component substitution, or be promoted because its result is favorable.
Post-outcome merging, splitting, relabelling or threshold tuning is prohibited.

## 2. Common assignment unit and source contract

The atomic row is one frozen forecast opportunity:

```text
(route_id, information_set_id, network_component_id, site_id,
 issue_date, target_local_date, horizon)
```

All dimensions are assigned to this same row. Assignment must be a left join from
the frozen opportunity registry; a missing measurement or metadata value creates
an explicit unknown/missing category and never deletes the opportunity row.

Permitted source fields are limited to:

- the sealed sampling frame and graph-component registry;
- `calendar_opportunity_registry_v1.parquet` and
  `measurement_qa_registry_v1.parquet` defined by the B3 draft;
- official provider codebooks whose exact bytes and retrieval dates are bound;
- training/calibration-only summaries explicitly permitted below;
- an externally cited scientific/decision threshold selected before target
  access.

Model predictions, target-period errors, target-period event frequency, target
completeness and any Stage-09 diagnostic are prohibited threshold-selection
inputs.

## 3. Dimension registry

### 3.1 Event strata

There is currently **no configured event threshold**. Each future event is one
independent Boolean definition with this required contract:

```text
event_definition_id
event_family                  # thermal | flow | compound | other
source_variable
source_statistic_code
time_basis                    # issue | target | antecedent_window | named_window
window_definition
source_unit
canonical_unit
conversion_rule
comparison_operator
threshold_value
threshold_unit
threshold_source_role         # external_scientific | training_only | metadata_rule
threshold_source_locator
threshold_source_sha256
classification_availability   # issue_time_available | posthoc_observed_context
frozen_at_utc
```

Until every field above is resolved, the event definition has status
`UNRESOLVED_PRELABEL_DECISION` and cannot classify a row or appear in a result
caption.

For a sealed event definition, each opportunity receives exactly one of:

| Level | Exact assignment |
| --- | --- |
| `EVENT_TRUE` | Required source value exists, units validate and the frozen predicate evaluates true |
| `EVENT_FALSE` | Required source value exists, units validate and the frozen predicate evaluates false |
| `EVENT_UNKNOWN` | Required value, unit, time alignment or source metadata is missing/invalid |

Rules:

- Levels are mutually exclusive within one `event_definition_id`.
- Different event definitions may overlap. A day may be both a thermal and flow
  event; the assignment table retains both flags rather than inventing a combined
  winner.
- A compound event must be separately registered as an exact Boolean expression
  over already sealed event IDs. It is not created after observing overlap.
- A target-date observed variable may be used only as post-hoc evaluation
  context. It cannot be described as an issue-time operational alert unless the
  underlying variable and threshold were genuinely available at issue time.
- Daily-mean WTEMP must not be equated with daily maximum, 7DADM, a regulatory
  threshold or ecological injury. An external threshold is eligible only when
  its endpoint, aggregation, unit and population match the registered target or
  the mismatch is explicitly labelled hypothetical/descriptive.
- Events may not be defined by model error, residual, rank, skill, coverage
  failure or any other evaluated outcome.

The protocol owner must choose the event family and threshold source. A training-derived
quantile probability, an absolute temperature, a flow recurrence threshold or a
compound rule remains `UNRESOLVED_PRELABEL_DECISION`; this draft assigns none.

### 3.2 Flow strata

The raw flow variable is the measurement-registry discharge value for the frozen
time basis. Native Route-A storage uses USGS daily mean `00060/00003` in cubic
feet per second (`ft3 s-1`, commonly `cfs`); Route B must bind the actual parameter,
statistic and unit from its own raw series rather than inherit them by name.

`flow_time_basis` is currently `UNRESOLVED_PRELABEL_DECISION`: the protocol
must choose issue-date flow, target-date post-hoc context or a named antecedent
window before seal.

The base sign dimension is exact and mutually exclusive:

| Level | Rule after unit validation |
| --- | --- |
| `FLOW_MISSING` | No retained value exists |
| `FLOW_UNIT_UNKNOWN` | A value exists but cannot be converted to the frozen canonical discharge unit |
| `FLOW_NEGATIVE` | Canonical discharge `< 0` |
| `FLOW_ZERO` | Canonical discharge `= 0` using the stored exact value; no tolerance is invented |
| `FLOW_POSITIVE` | Canonical discharge `> 0` |

Negative values are not clipped or renamed low/drought flow. Tidal, backwater and
series-semantics flags remain separate qualifier/context variables.

A positive-flow magnitude dimension may later use levels such as registered bins,
but the number of bins, cutpoints, reference population, weighting, time period
and tie convention are all currently `UNRESOLVED_PRELABEL_DECISION`. They must be
derived from training/calibration data only or an external hydrologic rule, then
stored as exact canonical-unit boundaries. Target-period flow distribution or
model performance may not set them.

Unit conversion must use an exact declared factor. A future implementation must
prove that equivalent cfs and SI inputs produce identical assignments at every
boundary after canonical conversion.

### 3.3 Season strata

The proposed default is the four meteorological seasons assigned from the
station's `target_local_date`:

| Level | Target-local calendar months |
| --- | --- |
| `DJF` | December, January, February |
| `MAM` | March, April, May |
| `JJA` | June, July, August |
| `SON` | September, October, November |
| `SEASON_UNKNOWN` | Target-local date/timezone cannot be established |

These levels are mutually exclusive and exhaustive after date resolution. Leap
day belongs to `DJF`. Season is based on the target local date, not the issue
month or UTC date. The timezone/local-day source must be the same bound source as
the B3 measurement/calendar registry.

This conventional definition is a **proposed draft choice**, not yet a seal. If
the protocol owner instead selects hydrologic, thermal or region-specific seasons, every
boundary, region map and provenance source must replace this table before target
access; no competing season definition may be chosen after results.

### 3.4 Year strata

`TARGET_YEAR_YYYY` is the integer year of `target_local_date`. It is mutually
exclusive and exhaustive after local-date resolution; unresolved local dates map
to `TARGET_YEAR_UNKNOWN`.

The untouched target period and therefore the allowed year-level set remain
`UNRESOLVED_PRELABEL_DECISION`. The seal must enumerate every allowed year and
reject any extra year. Year-equal aggregation versus a year-superpopulation claim
must match the B1/B3 estimand and UQ branch. Three target years do not by
themselves justify a general year-superpopulation claim.

Leave-year analyses use the same frozen year IDs. Years may not be pooled or
omitted because one has unfavorable performance or missingness.

### 3.5 Qualifier strata

Qualifier assignment consumes the raw provider qualifiers, approval status,
series-conflict code, method/series identity and missing-metadata reason from the
B3 measurement registry. The exact provider-code mapping is currently
`UNRESOLVED_PRELABEL_DECISION`; it must be frozen from an official codebook
snapshot before target access.

Once that mapping is sealed, each target row receives one mutually exclusive
analysis category under this precedence:

1. `QUALIFIER_SERIES_CONFLICT` — the frozen duplicate/series rule records an
   unresolved conflict;
2. `QUALIFIER_METADATA_UNKNOWN` — qualifier or approval metadata required by the
   mapping is unavailable;
3. `QUALIFIER_NOT_APPROVED` — official mapping identifies provisional,
   unapproved or rejected status;
4. `QUALIFIER_APPROVED_FLAGGED` — approved but at least one mapped substantive
   qualifier is present;
5. `QUALIFIER_APPROVED_UNFLAGGED` — approved and no mapped substantive qualifier
   is present;
6. `QUALIFIER_OTHER_MAPPED` — a code is validly mapped but does not belong to the
   preceding categories.

The raw qualifier vector remains lossless even though the analysis category is
single-valued. Multiple raw codes are allowed; the precedence above makes their
analysis category deterministic. Unknown codes fail the mapping validator and
become `QUALIFIER_METADATA_UNKNOWN`; they are not silently treated as clean.

Issue-time and target-time qualifier dimensions must use different IDs. A
target-time approval status is not assumed to have been known at issue time.

### 3.6 History-completeness strata

History assignment uses these B3 opportunity fields:

- `history_fraction`, dimensionless in `[0,1]`;
- `max_history_gap_days`, calendar days;
- predictor-specific observed/mask counts bound by `predictor_mask_sha256`;
- the exact required history window and information-set ID.

The minimum history fraction, maximum allowed gap, per-variable requirements and
any richer reporting-bin cutpoints are currently
`UNRESOLVED_PRELABEL_DECISION`. They must be selected from training/calibration
behavior only and frozen before target access.

After the primary completeness gate is frozen, the mandatory mutually exclusive
levels are:

| Level | Rule |
| --- | --- |
| `HISTORY_UNKNOWN` | Required metric/mask/window metadata is missing or invalid |
| `HISTORY_FAIL` | At least one frozen completeness predicate fails |
| `HISTORY_PASS` | Every frozen completeness predicate passes |

Optional richer `HISTORY_BIN_*` levels require an independently frozen ordered
cutpoint table. They may describe performance gradients but cannot alter the
primary inclusion gate. Fill values never count as observed history; every
imputation-consuming branch must satisfy the B3 fill-invariance test.

## 4. Cross-dimension overlap and atomic cells

Season, year, flow, qualifier and history dimensions overlap across dimensions
by design but are mutually exclusive within their own registered dimension.
Each Boolean event definition is separately mutually exclusive; different event
IDs may overlap.

An atomic cross-stratum cell is permitted only when its exact dimension product
is listed before target access, for example:

```text
event_definition_id × season × horizon
flow_sign × history_gate × horizon
```

The registry must not automatically generate every Cartesian product. Each
registered product records a scientific/reporting reason and a maximum order.
Unregistered intersections may appear only as counts in an exploratory appendix,
with no effect/interval/p-value, and cannot enter abstract, highlights or
headline figures.

## 5. Minimum support and reportability

Counts are always reported, including zeros and unknown categories. Performance
reportability is a separate deterministic gate.

The following thresholds are currently unresolved and must be frozen per
dimension/product, horizon and task arm:

```text
min_calendar_opportunities
min_target_observed
min_common_keys
min_keys_per_site
min_sites
min_network_components
min_years
min_event_occurrences              # event cells only
max_unknown_fraction
max_bootstrap_failure_fraction
```

No numerical value is supplied by this draft. Route-A's historical 100-paired-key
rule is not automatically imported into Route B, and sensor storage resolution
does not determine a statistical support threshold.

Every cell receives exactly one status:

| Status | Meaning |
| --- | --- |
| `REPORTABLE_DESCRIPTIVE` | All frozen descriptive-support rules pass |
| `LOW_SUPPORT_COUNTS_ONLY` | Counts are shown; performance and UQ are suppressed or explicitly marked unstable under the frozen rule |
| `NOT_ESTIMABLE` | Required score/effect cannot be computed |
| `THRESHOLD_UNRESOLVED` | Registry was not sealable; analysis must not proceed |

Any cell proposed for formal Route-B inference must additionally pass the B1
hard gates independently at that comparison/horizon: at least 30 reportable
network PSUs, effective fraction at least 0.75 and largest normalized weight
share below 0.25. Passing those gates does not itself promote a P1 stratum to a
formal hypothesis.

No low-support cell may be merged with a neighboring season/year/flow bin after
outcomes. Its failure status remains visible.

## 6. Statistics and uncertainty

For every registered reportable cell, record at minimum:

- calendar opportunities, target-observed rows, common keys, sites, network
  components, years and event occurrences as applicable;
- raw and design-weighted support, normalized weight shares, Kish effective PSU
  count and largest weight share;
- station/component-balanced RMSE, MAE and bias in degrees Celsius;
- paired candidate-minus-reference effect in degrees Celsius on exact common
  keys;
- for eligible probability models, interval score and width in degrees Celsius,
  pinball loss in degrees Celsius, and dimensionless coverage, Brier score,
  reliability and calibration slope;
- all-calendar attrition and unknown-category fractions.

Route-B effects use the frozen two-stage design weights. UQ must call the sealed
B3 crossed space×time procedure and bind its component/year/block resamples;
strata may not substitute an ordinary row bootstrap. Block length, draw count and
failure threshold inherit the final B3 contract and are currently unresolved
where B3 itself is draft.

Route-A stratified effects, if rendered later, remain fixed-cohort descriptive.
Whole-HUC2 intervals, sign-flip p-values or Holm values are assumption-conditional
sensitivities only and cannot support superiority, non-inferiority, equivalence,
parity or national inference.

## 7. Missing and unknown handling

1. Missing target values remain in opportunity/attrition denominators and cannot
   contribute a score.
2. Missing flow, qualifier, timezone/local date or history metadata maps to its
   explicit unknown category; it is never silently dropped.
3. `UNKNOWN` is not pooled with the reference/normal/clean category.
4. If an unknown fraction violates the frozen maximum, the cell fails
   reportability; the threshold is not relaxed.
5. Provider codebook changes create a new mapping version. Old rows are replayed
   under the sealed version rather than silently remapped.
6. Missingness sensitivities remain those specified by B3; a favorable IPW or
   pattern-mixture result cannot rescue a failed primary or support gate.

## 8. Multiplicity and claim control

The default P1-01 contract is:

```text
claim_role = DESCRIPTIVE_SECONDARY
hypothesis_test = NONE
decision_rule = NONE
multiplicity_family = NONE
```

Therefore no stratum-specific p-value, confidence interval or favorable rank may
be written as confirmatory evidence. UQ intervals quantify descriptive
uncertainty; they do not become a decision rule by comparison with zero or a
margin.

If the protocol owner wants a formal stratum-specific hypothesis, it must be defined in a
separate outcome-free Route-B amendment before target access. That amendment must
freeze the exact cell, estimand, direction, scientific margin, alpha allocation,
multiplicity method and failure rule. It cannot be added after seeing an overall
or subgroup result.

Prohibited rescue patterns include:

- reporting only the best season, year, event or flow class;
- dropping an unknown/low-support stratum;
- changing cutpoints to move a CI or p-value across a boundary;
- using a descriptive subgroup to reverse an unsupported overall headline;
- importing Route-B strata to upgrade Route-A's permanently descriptive five
  comparisons.

## 9. Machine-readable artifacts

### `strata_registry_v1.json`

```text
format
registry_id
status
route_id
information_set_ids
created_at_utc
outcome_access_attestation
sampling_frame_sha256
measurement_contract_sha256
model_budget_sha256
dimensions[]:
  dimension_id
  dimension_type
  assignment_unit
  source_fields
  source_units
  canonical_unit
  conversion_rule
  time_basis
  levels[]
  precedence
  mutually_exclusive_within_dimension
  overlap_group
  unknown_level
  threshold_spec
  threshold_source_role
  threshold_source_locator
  threshold_source_sha256
  threshold_status
  frozen_at_utc
registered_products[]
support_rules[]
statistics[]
uq_contract_id
claim_role
multiplicity_contract
unresolved_decisions[]
provider_codebook_bindings[]
self_sha256
```

`status=SEALED_PRELABEL` is forbidden while any required field contains
`UNRESOLVED_PRELABEL_DECISION`.

### `strata_assignment_v1.parquet`

```text
route_id
information_set_id
network_component_id
site_id
issue_date
target_local_date
horizon
opportunity_key_sha256
dimension_id
definition_version
raw_source_value
raw_source_unit
canonical_value
canonical_unit
assigned_level
unknown_reason
source_transaction_sha256
assignment_rule_sha256
```

One opportunity has one row per mutually exclusive dimension and one row per
registered Boolean event definition. The table must reconcile exactly to the
opportunity registry.

### `strata_cells_v1.parquet`

```text
registered_product_id
level_tuple
comparison_id
horizon
claim_role
n_calendar_opportunities
n_target_observed
n_common_keys
n_sites
n_components
n_effective_components
n_years
n_events
unknown_fraction
design_weight_sum
largest_weight_share
support_status
metric_id
value
unit
uq_low
uq_high
uq_role
evidence_pointer
```

### Receipts

- `strata_registry_seal_v1.json` binds the registry, provider codebooks,
  training/calibration-only threshold derivations, user decisions, source tree,
  sampling/B3 contracts, external timestamp/custodian and target-path absence.
- `strata_assignment_receipt_v1.json` binds every opportunity, raw transaction,
  assignment row, unknown reason and replay hash after authorized acquisition.
- `strata_analysis_receipt_v1.json` binds cell support decisions, statistics,
  UQ draws/parents, failures and rendered values.

Every receipt is create-once, self-hashed and independently replayable. A later
provider-code or threshold change requires a new registry/version and cannot
overwrite the sealed one.

## 10. Freeze sequence

```text
T0  this design draft; no authority
T1  user chooses Route-B arm, target period, event definitions and reporting products
T2  official qualifier codebook and unit/time semantics are byte-bound
T3  training/calibration-only derivations freeze flow/history/event cutpoints and support rules
T4  registry validator passes with zero unresolved required fields
T5  independent seal + external timestamp; target paths absent/unaccessed
T6  separately authorized target acquisition
T7  deterministic assignment, support/UQ analysis and receipts
T8  receipt-only paper/SI rendering
```

No T6 action is permitted because this draft exists. Any target access before T5
invalidates prelabel status for thresholds or products not already sealed.

## 11. Acceptance tests for the future implementation

P1-01 is complete only when all of the following pass:

1. The JSON schema rejects unknown keys, duplicate IDs, invalid units and
   unresolved required decisions in `SEALED_PRELABEL` state.
2. Target paths are absent/unaccessed at seal and the attestation is independently
   checked.
3. Every opportunity receives exactly one season, year, flow-sign, qualifier and
   history level, including explicit unknown levels.
4. Each event definition produces exactly one true/false/unknown state.
5. Within-dimension mutual exclusivity and cross-dimension/event overlap rules
   match the registry exactly.
6. Target-local date/timezone assignment handles year boundaries and leap days
   deterministically.
7. cfs/SI-equivalent flow values assign identically after the frozen exact
   conversion.
8. Negative flow is retained as negative and never clipped or renamed drought.
9. Official qualifier-code mapping is complete; unseen codes fail closed into an
   auditable unknown state.
10. History cutpoints use training/calibration inputs only; changing imputed fill
    values cannot change assignment when masks are fixed.
11. Assignment rows reconcile one-for-one with the opportunity registry and bind
    the raw transaction/mapping bytes.
12. All opportunity, observed, common-key, site, component, year, event and
    unknown counts reconcile across assignment, cell and attrition tables.
13. Support status is a pure function of sealed counts/thresholds and cannot be
    overridden by metric value.
14. Low-support, unknown and non-estimable cells remain present.
15. Design weights, effective component count and largest share reproduce the B1
    registry; any formal cell independently enforces the B1 gates.
16. Stratum UQ replays the sealed B3 space×time procedure rather than resampling
    rows as independent observations.
17. No unregistered Cartesian product emits a score/effect/interval.
18. No descriptive cell emits a confirmatory decision or multiplicity-adjusted
    claim.
19. Route-A rendering enforces the permanent descriptive-only wording regardless
    of favorable values.
20. Seal, assignment and analysis receipts validate their self-hashes and all
    parent byte bindings under an independent replay.

## 12. Decisions still required

The following cannot be inferred by the implementation team:

- primary Route-B task arm and target period;
- exact event definitions, endpoints, aggregation and threshold source;
- flow time basis and any positive-flow magnitude cutpoints;
- acceptance or replacement of the proposed DJF/MAM/JJA/SON convention;
- official qualifier codebook/version and mapping review;
- history fraction/gap/per-variable gates and optional bins;
- descriptive support thresholds and allowed cross-dimension products;
- final B3 block/UQ/failure contract;
- whether any stratum-specific test is promoted into a separately multiplicity-
  controlled formal family before outcome access.

Until those decisions are resolved, this document advances design completeness
but cannot be cited as P1-01 completion or preregistration.

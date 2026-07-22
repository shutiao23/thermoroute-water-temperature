# ThermoRoute

ThermoRoute is a research repository for a retrospective daily river-water-
temperature hindcasting benchmark at 1-, 3-, and 7-day horizons. It does not yet
establish an as-issued operational forecast. The model combines a damped-persistence anchor,
a learned thermal-relaxation proposal, a horizon-conditioned variable/lag router,
a causal temporal convolutional encoder, a regime mixture, bounded residuals, and
a separate MSE point head, three pinball-trained quantile heads, and split-conformal
calibration of the nominal 90% interval.

## Current evidence status

The repository is in a **pre-opening reconstruction state**.

- The canonical development panel is
  `data_usgs/panel_usgs_120v2.parquet`: 657,480 daily rows for 120 stable USGS
  site numbers from 2006-01-01 through 2020-12-31.
- The current registry covers 34 states and 15 HUC2 groups. It is bound to the
  panel by `data_usgs/frozen_panel_v1.json`.
- The 2019–2020 interval is development/exploratory data. It is not an independent
  final evaluation.
- The target interval for the one-time retrospective exercise is 2021-01-01
  through 2023-12-31. Its outcome values have not been requested or inspected in
  this workflow.
- The final pre-label protocol and its honest-owner Git seal are in `protocols/`.
  The seal has no external timestamp, public registration, independent custodian,
  or write-once storage.
- Old files under `outputs/` were produced by a legacy cohort and lack the current
  lineage sidecars. They are historical artifacts, not current evidence, and must
  not be cited as results of the present pipeline.
- The outcome-free 2018–2020 Daymet/gridMET compatibility bridge is complete and
  records `PASS_EXACT_PRODUCT_BRIDGE`. This proves exact agreement for the frozen
  bridge projection, not operational predictor availability or local-day
  equivalence with NWIS.
- The temporal-coverage policy, deterministic audit core, and physical replay
  path are implemented, but no target-period coverage result exists before the
  one-time opening and receipt binding.
- No current Route-A model suite, opening authorization, completed receipt, or
  verified formal result exists yet.

This status is deliberately fail-closed: `scripts/26_validate_claims.py` rejects
prohibited or malformed claims in the registered manuscript documents, and the
release tooling cannot produce a completed profile without a valid one-time
receipt. The validator is not a semantic proof over arbitrary repository prose;
unregistered generators require their own guards and tests.

## The problem

Daily stream temperature is highly persistent. A useful model therefore has to
beat strong, simple references on exactly the same station/date/horizon keys while
using finalized values dated on or before each historical issue date. This is a
date-indexed retrospective information rule, not proof that the same product
vintages were operationally available at that time. The benchmark also has to
report missingness, sensor qualifiers, geographic dependence, uncertainty, and
model-selection history without silently changing the cohort after outcomes are
seen.

ThermoRoute is designed to test whether a constrained learned correction can add
value over persistence, damped persistence, climatology, LightGBM, and a global
LSTM. The repository treats that as an empirical question; the architecture name
is not evidence of hydraulic transport or an identified physical mechanism.

## Frozen evaluation design

The development split is:

| Role | Dates | Use |
|---|---|---|
| Train | 2006–2015 | fit models and preprocessing |
| Validation | 2016–2017 | model selection only |
| Calibration | 2018 | conformal calibration and frozen event references |
| Development evaluation | 2019–2020 | previously inspected; exploratory diagnostics only |
| One-time target interval | 2021–2023 | labels remain sealed until all gates pass |

The exact raw feature order is
`WTEMP, FLOW, TEMP, PRCP, RHMEAN, DH, WDSP`. `WLEVEL` may be archived as raw
provenance but is not consumed by the Route-A models. `DH` is the legacy name for
Daymet daylight-period mean incoming shortwave radiation (W m⁻²), not day length;
`RHMEAN` is a derived humidity proxy. The external 30-site exercise
uses pooled, station-agnostic training but still uses each target site's observed
water-temperature history through the issue date; it is permanently exploratory.
The 2019–2020 period also informed station inclusion through minimum WTEMP/FLOW
coverage thresholds and is not blind, untouched, or independently confirmatory.

The formal family contains exactly five station-balanced comparisons. Every
reportable station needs at least 100 common valid target keys for the relevant
horizon and model pair. The effect is the median across station-level RMSE
differences. Primary p-values use exact whole-HUC2 sign-flip enumeration, confidence
intervals use a whole-HUC2 cluster bootstrap, and Holm adjustment covers exactly
the five frozen tests. A favorable statement requires both the adjusted p-value
and the predeclared confidence-bound rule; non-significance is never interpreted
as equality. The 15 HUC2 groups are small and unequal (inverse-Herfindahl effective
count 9.54 before attrition), so exact enumeration removes Monte Carlo error but
does not remove the joint sign-symmetry assumption or make cluster-bootstrap
coverage reliable in small samples.

The two H2 rows concern only the frozen LightGBM procedure: selection from four
predeclared candidate settings on 2016–2017, followed by the frozen five-seed fit.
They do not establish non-inferiority to LightGBM in general, to a larger search,
or to a differently resourced implementation. Historical model-selection budgets
are documented but were not equalized across model classes.

## Model design

For issue time `t` and horizon `h`, the point forecast has four main pieces:

1. A damped-persistence/climatology anchor supplies a strong conservative
   reference trajectory.
2. A learned flow- and season-conditioned relaxation proposal changes the thermal
   memory, but remains a statistical component rather than an energy-balance
   estimate.
3. A sparse horizon-conditioned router selects among predeclared variables and
   lags; a causal TCN and regime mixture encode recent history.
4. A `tanh`-bounded residual limits the point prediction's deviation from its
   anchor. This is an algebraic bounded-deviation property, not a deployment or
   tail-risk guarantee.

The learned models emit an MSE point and distinct pinball q05/q50/q95 heads. CQR
fitted on 2018 widens q05/q95 only; q50 is not adjusted. Member-wise averaged
quantiles are engineering ensemble summaries, not mixture-distribution quantiles.
Neural quantiles are ordered by construction, so their retained crossing-loss
field is compatibility-only and contributes zero. LightGBM's independently fit
heads are never sorted: the bundle records raw development crossings by member
and horizon, clips q05/q95 to the nominal raw q50 when necessary, and leaves q50
exactly unchanged.
Exceedance events use a frozen seasonal statistical reference derived without
target-period labels. Probability,
spatial-influence, exact-qualifier, and architecture-control outputs are descriptive
or exploratory unless the protocol explicitly places them in the five-test family.
The seven Stage 09 controls are seed-0-versus-seed-0, single-seed functionality and
intervention diagnostics. They do not prove that a component is necessary, identify
a causal mechanism, or establish cross-seed stability.

The Platt calibrator for each horizon is fitted with calibration rows as the fitting
units, whereas target-period probability metrics give each retained station equal
total weight. That frozen weighting difference must be kept visible when interpreting
calibration. The optional Air2stream-style reference is an unofficial style-based
implementation, not the official Air2stream code or a validated reproduction of it.

Temporal learned models receive stable site identity, while the pooled external
models do not. The latter still consume each new gauge's observed WTEMP history.
Other history cells may be filled by train-only seasonal medians while retaining
missingness masks; there is no minimum observed fraction in the 32-day context.
The 32 days are a construction buffer, not an effective 32-day memory claim: the
router is restricted to lags 0--14, and the current two-block, kernel-three TCN
has a theoretical seven-step receptive field. Consequently, no input older than
lag 14 can affect the current model output.

## Temporal coverage audit

The predeclared audit describes the fixed, availability-enriched Route-A cohort on
forecast keys where both the issue-date and target-date WTEMP values are retained
finite observations. It does not impute unobserved outcomes, test a missing-at-random
assumption, estimate performance on all calendar days, or establish stability by
year or season.

For each of the five formal model comparisons, the audit reports all eight frozen
descriptive alternatives: equal weighting of the 12 year-by-season cells, leave-one-
year for 2021, 2022, and 2023, and leave-one-season for DJF, MAM, JJA, and SON. The
largest candidate-minus-reference effect is the deterministic worst case; exact ties
use the frozen candidate order. These values cannot change the primary station set,
formal statistic, p-value, Holm decision, or claim eligibility, and a favorable
sensitivity cannot rescue an unfavorable or non-estimable primary result. When the
formal row is not estimable, its formal effect remains `NA`; any separately computable
prediction-derived effect remains descriptive only. The audit becomes evidence only
after all physical source files are independently replayed and its exact bytes are
bound into the opening receipt and completed release.

## Observed 7DADM description

Stage 21 is separate from Route A prediction evaluation. It reads independently
sourced observations explicitly identified as daily maximum water temperature,
computes a strict seven-consecutive-day average of daily maxima (7DADM), and
compares those observations with a site-specific standards registry. It does not
read or classify model predictions. Applicability is evaluated on the ending date
of each seven-day window, with one result per jurisdiction/designated-use/species-
life-stage/season context. A season-external row or a row lacking seven observed
daily maxima receives no exceedance classification.

The stage is fail-closed. Missing inputs produce only an empty table and an
explicit not-performed report; missing, unrelated, or ambiguous site/standard
matches abort threshold application. An observed value above a registry threshold
is reported only as a descriptive exceedance. It is not a legal or regulatory
decision, does not determine compliance with any law or regulation, and is not a
forecast result or evidence of management value.

## Evidence workflow

The intended order is strict:

1. Commit the protocol, seal, claim ledger, dependency locks, source, tests, and
   training code.
2. Validate the committed 2018–2020 Daymet/gridMET re-fetch, archived exact
   responses, and development predictor-product bridge against the frozen panel on
   the exact site/date registry. The committed gate records
   `PASS_EXACT_PRODUCT_BRIDGE`. This checks product and parser compatibility, but
   not subdaily local-day equivalence or as-issued availability. This outcome-free
   engineering gate was added after the local protocol seal and cannot change its
   hypotheses or decisions.
3. Rebuild the canonical development chain, including the separate matched
   MLP/TCN and multi-seed feature-ladder audit; freeze all five-member temporal
   and pooled external model bundles; replay every model head on development data.
   Stage 9 publishes its content-bound completion receipt only after the canonical
   predictions, tables, report, bundle parity checks, and all three formal pointers
   succeed. `scripts/run_all.sh` then explicitly runs Stage 09b. Stage 09b publishes
   its own receipt as its final atomic write only after the exact 31-member
   (5 MLP + 5 TCN + 21 feature-ladder) registry, common forecast keys and truth,
   architecture/optimisation budget, combined predictions, report, and every
   lineage sidecar validate. Stage 24 rejects either receipt when it is missing,
   stale, incomplete, tampered, or bound to another source/runtime/panel/registry.
   It binds both receipt paths and SHA-256 digests into the frozen suite identity;
   the independent release verifier enforces the same two-gate closure without
   executing archive code.
4. Commit the model-suite registry while candidate metadata and target-period
   predictor artifacts are absent.
5. Acquire metadata-only candidate evidence and retrospective Daymet/gridMET
   inputs without requesting outcome values; commit those exact bytes.
6. Replay Git ancestry and blob hashes, then create one immutable authorization.
7. Execute the fixed raw-only acquisition child and trusted scorer. Before the
   acquisition manifest exists, network transport may resume only within the same
   opening ID and one exact frozen request ledger, only for requests without a
   durably published canonical transaction, and only while normalized, derived,
   and trusted outputs are all absent. HTTP delivery is not exactly once: retries
   are allowed, and a response received before its transaction directory becomes
   durable may be requested again. A complete, durable, verifiable canonical
   response is never replaced. A partial, invalid, or noncanonical canonical
   transaction fails closed without overwrite; cleanup is limited to unpublished,
   owner-private temporary or pending state.
   Once every raw transaction is complete, the request map, two normalized tables,
   and manifest are generated and validated in one private same-filesystem stage
   and published by one directory rename. Once that manifest exists, the raw child
   is permanently disabled; a resume can only perform network-free deterministic
   trusted recomputation, atomic trusted-directory publication, receipt completion,
   or validated sidecar repair.
8. Render all five statements atomically and idempotently from the verified receipt
   with `python scripts/26_validate_claims.py --root . --registry
   protocols/route_a_claim_registry_v1.json --write-generated-results
   --require-complete`, then build a completed release profile.

There is one logical opening and one frozen request ledger, not a claim of
exactly-once HTTP delivery. Trusted recovery never creates another logical opening
or ledger. A wholly absent canonical
`trusted/` directory is recomputed as a complete generation in a private
same-filesystem staging directory and published by one directory rename. A complete
canonical trusted generation without a receipt is fully replayed before receipt
creation. An abandoned stage is deleted only after its canonical name,
owner-private mode, same-device read-only regular files, and absence of external
hard links all validate; any unsafe stage fails closed. Partial, invalid, extra,
linked, or otherwise noncanonical canonical trusted contents are not repaired or
replaced. A valid receipt with only its digest sidecar
missing may regenerate that sidecar after full validation; a sidecar without its
receipt fails closed. These guards address honest-owner interruption and
misoperation, not a malicious filesystem owner or same-UID adversary.

Failure of the chronology gate demotes the exercise to retrospective exploration.

## Repository layout

```text
data/                         legacy inputs for three ordinary monitoring stations
data_usgs/                    canonical panel, registries, and frozen evidence
protocols/                    protocol, protocol seal, and structured claim ledger
src/thermoroute/              model, data, inference, provenance, and opening code
scripts/                      development, freezing, opening, and release entrypoints
tests/                        leakage, replay, schema, failure, and release tests
paper/                        byte-frozen pre-opening snapshots; POST claims are generated only in the canonical Markdown evidence layer
outputs/                      legacy artifacts plus future content-addressed outputs
```

## Verification commands

These commands are safe before label opening:

```bash
python -m pytest -q
ruff check src tests
ruff check --select F scripts
mypy src/thermoroute --ignore-missing-imports
python -I -B scripts/27_verify_development_replay.py --help
python scripts/26_validate_claims.py \
  --root . --registry protocols/route_a_claim_registry_v1.json
```

The full development/model-freezing pipeline is computationally expensive and is
not represented by the legacy `outputs/` directory. Do not run the one-time opening
command until the model suite, predictor evidence, chronology receipt, clean-source
authorization, and all preflight tests are present.

## Reproducibility boundary

The canonical 2006–2020 derived panel and stable registry are byte-bound, but the
original provider HTTP responses and retrieval timestamps for that development
panel are unavailable. Development reproduction is therefore conditional on the
committed Parquet bytes. New candidate, meteorological, and opening acquisitions
are designed to archive exact requests, responses, timestamps, qualifiers, and
hashes.

The current protocol evidence is repository-internal and assumes an honest owner.
It does not protect against an owner rewriting local Git history. No remote push,
public registration, or external artifact publication is performed by this
workflow.

The release archive produced by `scripts/make_release_archive.sh` is a local
verification artifact, not permission to redistribute every bundled dataset and
not a public release. Zenodo deposit metadata is intentionally disabled while
creator identities and the redistribution terms for each data category remain
unverified. A `.zenodo.json` file may be restored only after verified creators and
separate, accurate data-license metadata are available.

## License

Code is provided under the repository license. Data redistribution and provider
terms must be reviewed separately before public release, especially for the three
station case-study files whose source and redistribution authorization are not yet
documented.

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
Exceedance events use a frozen seasonal statistical reference derived without
target-period labels. Probability,
spatial-influence, exact-qualifier, and architecture-control outputs are descriptive
or exploratory unless the protocol explicitly places them in the five-test family.

Temporal learned models receive stable site identity, while the pooled external
models do not. The latter still consume each new gauge's observed WTEMP history.
Other history cells may be filled by train-only seasonal medians while retaining
missingness masks; there is no minimum observed fraction in the 32-day context.
The 32 days are a construction buffer, not an effective 32-day memory claim: the
router is restricted to lags 0--14, and the current two-block, kernel-three TCN
has a theoretical seven-step receptive field. Consequently, no input older than
lag 14 can affect the current model output.

## Evidence workflow

The intended order is strict:

1. Commit the protocol, seal, claim ledger, dependency locks, source, tests, and
   training code.
2. Re-fetch 2018–2020 Daymet/gridMET with the confirmation parser, archive the
   exact responses, and require the development predictor-product bridge to
   reproduce the frozen panel on the exact site/date registry. This checks product
   and parser compatibility, but not subdaily local-day equivalence or as-issued
   availability. This outcome-free engineering gate was added after the local
   protocol seal and cannot change its hypotheses or decisions.
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
7. Execute the fixed raw-only acquisition child and trusted scorer. Network
   transport may continue only within the same opening ID and exact request ledger,
   before any normalized or trusted output exists.
8. Render all five statements automatically from the verified receipt and build a
   completed release profile.

Failure of the chronology gate demotes the exercise to retrospective exploration.

## Repository layout

```text
data/                         three-station legacy case-study inputs
data_usgs/                    canonical panel, registries, and frozen evidence
protocols/                    protocol, protocol seal, and structured claim ledger
src/thermoroute/              model, data, inference, provenance, and opening code
scripts/                      development, freezing, opening, and release entrypoints
tests/                        leakage, replay, schema, failure, and release tests
paper/                        pre-opening manuscript sources; results currently pending
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

## License

Code is provided under the repository license. Data redistribution and provider
terms must be reviewed separately before public release, especially for the three
station case-study files whose source and redistribution authorization are not yet
documented.

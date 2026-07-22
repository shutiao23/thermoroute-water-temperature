# ThermoRoute — pre-opening highlights

> Draft status: the canonical computation and one-time target-period evaluation
> remain incomplete. These are design highlights, not performance findings.
> This byte-frozen pre-opening status snapshot is not a live project-status page
> and is not rewritten after the one-time opening.

- Is designed to hindcast daily river water temperature at 1-, 3-, and 7-day horizons.
- Anchors a learned temporal model to damped persistence with a bounded residual.
- Uses a stable 120-site USGS development panel covering 2006–2020.
- Separates training, validation, calibration, and exploratory development years.
- The frozen protocol specifies six primary model types and seven one-factor controls.
- Stage 09 controls are single-seed functionality diagnostics, not causal ablations.
- Uses station-balanced RMSE effects on identical model-pair target keys.
- Enumerates whole-HUC2 signs and adjusts exactly five formal p-values with Holm.
- Restricts H2 to the frozen four-candidate, five-seed LightGBM procedure.
- Audits observable-key temporal coverage with eight fixed descriptive sensitivities.
- The opening design will archive raw target requests, identifiers, qualifiers, and conflicts.
- Requires a model-freeze commit before candidate metadata and later covariates.
- Generates formal statements only from a fully verified one-time receipt.

## Current status

The legacy headline numbers have been withdrawn because they belong to a different
station mapping and do not have the lineage sidecars required by the current code.
The 2018–2020 predictor-product bridge has passed. The outstanding work is to rerun
the canonical development chain, freeze and replay all model bundles, acquire the
remaining outcome-free metadata and historical covariates in the required order,
pass authorization, execute the fixed opening, receipt-bind the coverage audit, and
regenerate the manuscript and release evidence.

## Scope

The architecture is physics-inspired but remains a statistical predictor. Its
event threshold and numerical comparison margin are methodological quantities.
The external-site analysis uses local observation history and remains exploratory.
Coverage sensitivities condition on observed issue/target WTEMP and do not establish
missing-at-random, all-calendar, year-stability, or season-stability claims. The
current Git seal is repository-internal and has no independent timestamp or custodian.

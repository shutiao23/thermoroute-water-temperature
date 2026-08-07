# Route B model budget, unit invariance and identifiability draft (P0-B2)

| Field | Value |
| --- | --- |
| Date | 2026-08-01 |
| Status | **DRAFT ONLY / PRELABEL / NOT AN EXECUTABLE PROTOCOL** |
| Dependency | Requires an accepted P0-B1 population, estimand, task arm and graph-disconnected-network registry, with residual dependence limitations carried into UQ |
| Route-A boundary | Does not modify, reuse or upgrade Route-A Stage-09 evidence |

Official-source comparator candidates, exact package hashes and arm eligibility
are recorded separately in
`docs/ROUTE_B_COMPARATOR_PROVENANCE_20260801.md`. That evidence does not select or
freeze a comparator.

## 1. Contribution decision

The default Route-B contribution should be a **structured statistical predictor**.
The bounded correction and thermal proposal may provide useful inductive bias,
but they are not a heat-balance equation and do not identify physical states.

Two routes are allowed before freeze:

1. **Predictive-only (recommended):** report point/probability performance and
   failure boundaries; use neutral names for unvalidated latent variables.
2. **Mechanistic:** add an identifiable heat-balance formulation plus independent
   measurements of heat flux, geometry, velocity/travel time, groundwater,
   shading and reservoir state. Architecture ablations and multi-seed stability
   alone are insufficient.

## 2. Information-set contract

Every comparison must be assigned one exact `information_set_id`.

| Information set | Allowed information | Prohibited claim |
| --- | --- | --- |
| Known-gauge retrospective | target-site WTEMP/FLOW through issue date; frozen historical meteorology | ungauged or operational |
| Strict ungauged | no target-site WTEMP in training, transforms, selection or issue-time input | known-gauge result cannot be substituted |
| Operational replay | archived as-issued observations, predictor vintages, latency and future NWP available at issue time | operational if latest retrospective products are used |

All models in one formal family use identical issue/target keys and the same
information set. A model may not gain future meteorology, target-site history or a
more complete key set because its implementation expects it.

## 3. Fair model matrix

The model matrix is arm-specific. A model that requires target-site WTEMP is not
eligible merely because it appears in another arm. The known-gauge suite should
include, at minimum:

- persistence and damped persistence;
- climatology;
- an externally verified **official Air2stream implementation** with code version,
  license and reproduction case;
- LightGBM;
- a recurrent or convolutional sequence baseline;
- one preselected current strong probabilistic sequence model;
- ThermoRoute;
- separately labelled capacity-matched attribution controls.

In the strict-ungauged arm, persistence, damped persistence and any site-calibrated
Air2stream configuration are `NOT_ELIGIBLE`. A separately preregistered
regionalized process baseline may enter only if every parameter/initial state is
learned from development networks and no target-site WTEMP is used. The existing
in-repository Air2stream-style code is not automatically eligible
to represent the official published implementation. Until provenance and a
reproduction case are bound, label it as an unofficial process-style baseline.

### Proposed model-matrix fields

```text
model_id
implementation_uri
implementation_commit_or_container_sha256
license_evidence
information_set_id
search_space_sha256
search_algorithm
max_trials
max_cpu_seconds
max_gpu_seconds
hardware_id
tuning_seeds
final_seeds
parameter_count
parameter_band_role
development_cluster_registry_sha256
validation_metric
common_key_registry_sha256
```

### Draft budget rule

The initial design value is 40 search trials, 3 tuning seeds and 5 final seeds per
learned model family, using the same search algorithm and validation objective.
A one-time outcome-free pilot may set model-specific hardware-hour caps; the
pilot and cap must be frozen before the formal search.

Each model stops when either its trial limit or hardware-hour cap is reached.
Unused budget is not transferred. Final reporting averages all registered final
seeds; selecting the best seed is prohibited. Runtime, peak memory, parameter
count, failed trials and all search results are reported.

Capacity-matched attribution controls target parameter count within a predeclared
band (draft: ±2%). A main-algorithm comparison does not become capacity-matched
merely because both models received the same number of trials.

## 4. Trial and suite artifacts

### `route_b_model_budget_v1.json`

Binds the information sets, search spaces, trial/compute caps, seeds, hardware,
validation objective and failure rules.

### `model_matrix_v1.json`

One immutable row per model/arm with the fields above.

### `trial_ledger_v1.parquet`

```text
model_id
trial_id
hyperparameter_sha256
seed
started_at
ended_at
cpu_seconds
gpu_seconds
peak_memory_bytes
status
validation_score
log_sha256
```

### `model_suite_receipt_v1.json`

Self-hashed closure over implementations, legal/provenance evidence, information
sets, keys, searches, seeds, selected configurations and compute use.

## 5. Unit-covariant objective

Let `σ_T` be a positive robust temperature-anomaly scale computed only from the
training split and bound in the model receipt. The recommended implementation
trains temperature heads in dimensionless coordinates:

```text
z = (T - anchor) / σ_T
```

In z space, MSE, pinball, crossing and residual penalties are already
dimensionless and must **not** be divided by `σ_T` again; BCE is unchanged and λ
weights are dimensionless. Equivalently, an implementation may remain in the raw
temperature scale and divide MSE by `σ_T²` and pinball/L1/crossing terms by
`σ_T`. The two formulations are alternatives, not cumulative normalizations.

The recommended z-space objective therefore uses:

- MSE/pinball/crossing/residual penalties directly on z-space heads/targets;
- BCE unchanged;
- dimensionless λ weights;
- the bounded-correction scale stored as dimensionless `delta_z`.

Input temperature transformations, thresholds, climatology, anchors and output
inverse transforms must follow the same affine unit conversion.

### Unit-equivalence receipt

The receipt binds:

- training-only `σ_T` and all transformed constants;
- °C and °F toy datasets generated from identical physical series;
- identical deterministic initialization/search choice;
- both training logs and inverse-transformed predictions;
- `max_abs_prediction_difference <= 1e-5 °C` after inverse transformation.

A scalar loss equality test alone is insufficient; end-to-end training and output
equivalence are required.

## 6. Identifiability gate

The current additive internal-prior plus neural-residual form admits a compensation
direction: adding a function to the prior and subtracting it from the residual can
leave the final prediction unchanged. Outcome loss alone cannot identify either
term as a physical state.

The prelabel gate must record:

```text
test_id
parameter_block
analytic_symmetry_found
jacobian_rank
jacobian_condition_number
profile_range
multiseed_latent_stability
threshold
pass
allowed_claim_role
```

Required tests include analytic reparameterization, Jacobian/rank diagnostics,
profile loss along compensation directions, expert permutation handling and
multi-seed latent stability.

Fail-closed rule:

```text
analytic compensation OR rank deficiency OR flat profile OR unstable latent
    → PREDICTIVE_ONLY_LATENTS_NOT_IDENTIFIED
```

Prediction benchmarking may continue after this verdict, but κ, equilibrium,
router weights and expert assignments may not be interpreted as physical drivers
or mechanisms. Independent physical-state validation is required to reopen that
claim class.

## 7. Acceptance gates

P0-B2 passes only when:

1. every eligible model in an arm has the same formal information set and exact
   common keys; ineligible target-history baselines are explicitly absent rather
   than silently adapted;
2. the full search/seed/compute ledger closes with no cap overrun;
3. official Air2stream provenance, license and a reference reproduction case are
   verified, or the model is explicitly downgraded to unofficial;
4. the current probabilistic sequence comparator is selected and versioned before
   target access;
5. °C↔°F end-to-end equivalence passes;
6. the identifiability verdict automatically controls allowed mechanism wording;
7. unfavorable, failed and timed-out trials remain in the ledger.

## 8. Future implementation namespace

After the Stage-09 source boundary is released, use new Route-B-only files such as:

- `protocols/route_b_model_budget_v1.json`;
- `src/thermoroute/route_b_budget.py`;
- `src/thermoroute/route_b_loss.py`;
- `src/thermoroute/route_b_identifiability.py`;
- corresponding tests for budget overrun, common keys, C/F end-to-end training,
  compensation detection and claim gating.

All runtime artifacts belong under `outputs/route_b/**`; no Route-A run cache is a
valid Route-B trial.

## 9. User decisions required

- primary arm/information set;
- scientific margin/MDE and validation objective;
- hardware-hour cap and available hardware;
- selected current probabilistic sequence baseline;
- predictive-only or mechanistic contribution;
- authorization to obtain/verify the official Air2stream implementation and its
  redistribution terms.

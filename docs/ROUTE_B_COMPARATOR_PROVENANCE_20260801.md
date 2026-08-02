# Route B comparator provenance — candidate freeze inputs

| Field | Value |
| --- | --- |
| Snapshot | 2026-08-01 |
| Status | **EVIDENCE COLLECTED / COMPARATOR NOT YET SELECTED OR FROZEN** |
| Scope | Official Air2stream plus modern probabilistic sequence candidates |
| Safety | Official repositories, papers, documentation and package metadata inspected read-only; no package, model, data or target outcome accessed |

## 1. Air2stream

The maintained author repository is
[`spiccolroaz/air2stream`](https://github.com/spiccolroaz/air2stream). A live
read-only `git ls-remote` resolved `master` and `HEAD` to:

```text
d4834bccf01657c03ab60efb4c18f8a256132c53
```

No tag was advertised by that query. The README calls the executable version
`1.0.0`, cites Toffolon and Piccolroaz (2015), and supplies Fortran source,
precompiled executables and three Swiss reference cases. The repository
[license](https://raw.githubusercontent.com/spiccolroaz/air2stream/master/LICENSE)
is Creative Commons Attribution-ShareAlike 3.0 Unported. The original authors'
repository points users to this maintained repository.

Protocol consequence:

- pin the full commit, source-tree digest, compiler/runtime and license bytes;
- compile from source in the clean-room image rather than trusting the shipped
  executable;
- reproduce at least one included reference case and bind input/output hashes;
- use the official parameterization and objective selected before outcomes;
- give Air2stream the same issue-time air-temperature/discharge information and
  calibration budget as competing models.

Official Air2stream is an at-site calibrated model. Its calibration consumes
site water temperature, so it is eligible for the known-gauge arm only. A
regionalized parameter variant that sees no target-site water temperature would
be a separately named derivative, not the official Air2stream baseline and not a
silent substitute in the strict-ungauged arm.

The isolated source-build attempt is recorded in
`docs/AIR2STREAM_SOURCE_BUILD_AUDIT_20260801.md`. It verified the pinned Git tree
and source/input hashes but is fail-closed as
`BLOCKED_NO_COMPILER / REFERENCE_CASE_NOT_ATTESTABLE`: no Fortran compiler is
installed, the source uses Intel-specific `ifport`/`makedirqq`, and the repository
contains neither a committed golden output nor a fixed optimizer seed contract.

## 2. Modern probabilistic sequence candidate

### Preferred candidate for a budgeted pilot: N-HiTS + multi-quantile loss

Nixtla identifies its
[`NeuralForecast` N-HiTS](https://nixtlaverse.nixtla.io/neuralforecast/models.nhits.html)
as the official implementation of the AAAI 2023 model. Its documentation exposes
multi-quantile, parametric distribution and implicit-quantile losses rather than
requiring a point forecast to masquerade as a distribution.

The current PyPI candidate snapshot is `neuralforecast==3.2.0`, Apache-2.0, with
trusted-publishing provenance to source commit
`a98b8d028f73838b792b32c3f905186b42cb39aa`:

| Artifact | SHA-256 |
| --- | --- |
| `neuralforecast-3.2.0.tar.gz` | `eea25c7f4a885b56d2f2a96b1aebaa425da730f25b5176783018c5f62fc11b13` |
| `neuralforecast-3.2.0-py3-none-any.whl` | `e46d72ff3534de9333d6ce222e16663f36e467c35d86382a30a5014d3b05eb84` |

This is a candidate, not yet a dependency or protocol choice. Before freezing,
the pilot must demonstrate exact exogenous-covariate semantics, output-quantile
ordering, deterministic replay and feasible CPU cost under the common budget.

### Predeclared sensitivity candidate: Temporal Fusion Transformer

TFT is designed for multi-horizon forecasts with static, observed and known
time-varying covariates; the standard PyTorch Forecasting implementation supports
quantile loss. The current PyPI snapshot is `pytorch-forecasting==1.8.0`, MIT,
with trusted-publishing provenance to source commit
`4d8d97cd3e85a15a9b90a38dfb0afc819d8e8aa4`:

| Artifact | SHA-256 |
| --- | --- |
| `pytorch_forecasting-1.8.0.tar.gz` | `f7de2af2fb7ce0c4ef73287666b8b79391485ff55883c974b6267b505adf0422` |
| `pytorch_forecasting-1.8.0-py3-none-any.whl` | `d77ad9a86a310513b3835570f2119200d7a8818fe6acf42106bb97dbba66dfff` |

TFT is more tuning- and memory-intensive. It should not displace the primary
candidate merely by receiving a larger search. If retained, it receives the same
trial accounting and stopping rules and is reported even when it fails.

## 3. Arm eligibility

| Comparator | Known-gauge | Strict ungauged | Operational replay |
| --- | --- | --- | --- |
| Official at-site Air2stream | Eligible with frozen calibration | `NOT_ELIGIBLE` | Eligible only with as-issued forcing and issue-time state |
| N-HiTS/TFT using target-series encoder history | Eligible | `NOT_ELIGIBLE` | Eligible only if all encoder/decoder inputs existed as issued |
| Exogenous-only global derivative | Separate candidate requiring its own freeze | Potentially eligible | Depends on forcing vintage |

Masking target-site history after selecting a target-history model is not a fair
strict-ungauged baseline. The exogenous-only architecture, training network,
static descriptors and zero-history behavior must be specified and budgeted
before target access.

## 4. Freeze gates still missing

1. User selection of primary arm, comparator and compute ceiling.
2. Fully hashed dependency lock including transitive packages and license notices.
3. Air2stream source build and reference-case reproduction receipt; the current
   isolated attempt records the exact compiler/portability/golden-output blockers
   but is not a passing receipt.
4. Candidate smoke/pilot results using development-only data and common budgets.
5. Exact information-set audit for every horizon and arm.
6. Frozen quantile/distribution outputs and calibration policy.
7. Failure rules that retain unsuccessful or over-budget candidates in the trial
   ledger.

Until these gates exist, `Air2stream`, `N-HiTS` and `TFT` are provenance-qualified
candidates, not completed Route-B comparisons.

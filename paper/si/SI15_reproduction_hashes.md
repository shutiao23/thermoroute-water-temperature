# SI15 — reproduction hashes, commands and parity

**Status:** filled from the persisted conventional-holdout artifacts; source-tree and protocol bindings are not applicable in the conventional design (the pre-registration apparatus was removed).

This SI projects the reproduction bindings — source tree, protocol/model suite,
input closure, runtime/container, and replay/render receipts — for the held-out
2021--2023 window. A local successful run is not an independent clean-room
replay, and every binding below must carry a literal command and a declared
numerical tolerance once filled.

## Reproduction binding projection

| Binding | path/identifier | SHA-256 or version | verification command/role | status | binder row ID |
|---|---|---|---|---|---|
| model bundles | `data_usgs/model_bundle_manifest_v1.json` | `507bed94de033e7c9929d77964f5e637ba4329b6a33995ea19a3c86af4477dbe` | `scripts/verify_model_bundles.py` (174 files) | verified | si15.bundle-manifest |
| per-key predictions | `outputs/conventional/predictions_2021_2023.parquet` | `6aa05b4d737fc18361ff6dd352165a0491399a5ee06ac6df015bd2efaa5ce959` | G1-G14 gates in `validation_report_2021_2023.json` | verified | si15.predictions |
| 2019-2020 reproduction | `outputs/conventional/validation_2019_2020.json` | `e10f28382c2c9a95214acfc01ff11fddf20ac9cbb0f6805550656f602d900edd` | max abs diff 9.5e-7 (ThermoRoute), 0.0 (LightGBM/LSTM/plain arms) | verified | si15.validation |
| held-out metrics | `outputs/conventional/holdout_metrics_2021_2023.csv` | `b79fdbaf63100193e4def61798e6d71d58d42b3f7fc8ed36e1fba585e8f96c11` | `scripts/verify_holdout_metrics.py` (129 checks) | verified | si15.metrics |

## Source and fill routing

Each `[pending computation]` cell binds, for the row's binding class, the
column's named quantity under the replay and reproduction receipts:

- **path/identifier** — the exact repository path or external identifier of the
  bound object.
- **SHA-256 or version** — the content hash (or declared version) of that
  object.
- **verification command/role** — the literal top-level command and the evidence
  role it discharges (source identity, chronology/model authority, immutable
  inputs, environment identity, parity/publication closure).
- **status** — the bound verification result, never an assumed pass.
- **binder row ID** — the `tables.si15.rows[<id>]` cell-level binder key.

The final SI must provide literal top-level commands and declared numerical
tolerances. Reusing one value ID for different objects, or providing only a
row-level source pointer, is an error.

## Relationship to the release manifest

This SI has no figure; its consumer is the POST verifier and the public-archive
release manifest. It is gated on the replay/render receipts and on the rights
decisions in SI16, so no binding is available yet.

Every displayed path, hash/version and status follows the README cell-level
binder contract.

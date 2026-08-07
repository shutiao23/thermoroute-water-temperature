# Stage-09 cross-source provenance bridge — design only

| Field | Value |
| --- | --- |
| Date | 2026-08-01 |
| Status | **DESIGN ONLY / NOT IMPLEMENTED / NOT AN ADMISSION RECEIPT** |
| Candidate producer source | `19289553aa0929bdb5803a8a3eaa96b38a651b3d52da441fb6298f2ac9228b55` |
| Candidate run | `7cb2bfb18c1f9aa3dba7` |
| Target access | No 2021–2023 outcome was requested or read for this audit |

## 1. Current verdict

The current repository does **not** permit cross-source reuse. Stage-09 receipt
validation requires the receipt source to equal the executing source tree;
model-suite validation then requires the development contract and every learned
bundle from Stages 09, 09b, 16 and 25 to share that same source hash. Opening and
release verification repeat the equality checks.

Consequently, neither of these shortcuts is valid:

- replacing the old hash in model metadata or sidecars with a new hash;
- treating a successful old-source receipt as if a new source produced it.

Both would erase provenance rather than bridge it.

## 2. Recoverable state, without claiming completion

Read-only inspection found the current run's 35/35 control-member receipt and
six of the nine canonical Stage-09 artifact classes: run manifest, 912 MB
prediction Parquet, prediction sidecar, scores, report and LightGBM selection.
The ThermoRoute pointer, LightGBM pointer, component pointer and canonical
completion receipt are absent. The prior process was killed during
prepublication validation, after predictions were saved. Therefore this remains
an incomplete candidate, not a reusable formal upstream.

No recovery, pointer publication or watcher change is authorized by this design.

## 3. Lowest-risk execution order

1. Under the unchanged producer source, obtain a native, independently validated
   Stage-09 receipt. The existing guarded relaunch can reuse caches, but has a
   known memory-risk in final validation and requires explicit user approval.
2. Prefer completing Stages 09b, 16 and 25 and freezing the entire A1 development
   suite under that same producer source. Do not mix old-source Stage 09 with
   new-source downstream training implicitly.
3. Stop the old-source formal chain at a governed terminal state. Verify that no
   related process is live and record the watcher/lock transition.
4. Implement and seal P0-A0 and P0-R0 under a new governance source, still before
   any confirmation-outcome acquisition.
5. Admit the immutable old-source A1 suite through one explicit dual-hash bridge.
   Only then build P0-A2 chronology and opening authorization under the new
   governance source.

This ordering avoids retraining only if the complete producer suite first passes
its native validators and the bridge gates below all pass. The conservative
fallback is a full new-source rerun.

## 4. Required dual-hash contract

The adopted suite must retain, rather than overwrite, two identities:

- `producer_source_sha256`: source that computed predictions, trained weights and
  emitted the native A1 receipts;
- `governance_source_sha256`: source that validates the erratum, chronology,
  authorization, opening and release.

Add a create-once `route_a_producer_suite_adoption_receipt_v1` that binds:

- producer Git commit and source-tree hash;
- governance Git commit and source-tree hash;
- all four native completion receipts, frozen model suite, development replay,
  input closure, numerical runtime and every learned-artifact hash;
- a complete producer-to-governance diff inventory;
- the sealed A0 erratum and clean-room receipt;
- an attestation that no confirmation outcome was requested or read;
- an external pre-opening timestamp and a self hash.

The bridge must be a new authority object. It must not rewrite producer bundle
metadata, sidecars, run IDs, native receipts or the old model suite.

## 5. Fail-closed eligibility

Adoption is allowed only when all conditions hold:

1. The producer suite passes its original validators in an isolated checkout of
   the exact producer commit.
2. Every producer artifact is content-addressed and byte-identical to the native
   receipt closure.
3. The governance diff changes no training target, inclusion rule, input feature,
   preprocessing, architecture, loss, seed, hyperparameter, calibration fit,
   prediction, score or model-selection computation used by the producer suite.
4. The A0 changes only add outcome-free governance, full-key equivalence tests,
   ambiguity resolution and downstream validation. If any computational
   dependency differs, adoption fails and a new-source rerun is mandatory.
5. An independent verifier checks both Git trees and the adoption receipt without
   importing producer Python into the governance process. A clean-room replay of
   the producer validators is a separate required check.
6. Opening, release and archive verifiers require the adoption receipt and expose
   both hashes in all downstream manifests.
7. Missing, extra or unclassified diff paths fail closed. A reviewer cannot waive
   the failure after seeing outcomes.

Static call-graph reasoning alone is insufficient for condition 3. The allowlist
must be exact and reviewable, and the old producer computation remains labelled
as old-source computation even after adoption.

## 6. Implementation impact

Supporting this contract requires coordinated source changes in model-suite,
chronology, opening, release and archive validators plus exact-schema tests. The
present equality checks must not simply be deleted; they should remain the native
path, with adoption as a second, narrower path requiring the dual-hash receipt.

Until those changes are implemented and tested, the repository's only valid path
is same-source completion followed by a same-source model suite, or a complete
rerun under the eventual new source.

## 7. Evidence locations in the current code

- `src/thermoroute/model_suite.py`: Stage-09 current-source check near
  `_load_formal_stage09_manifest`; development and learned-bundle source equality
  in `validate_model_suite_document`.
- `src/thermoroute/opening.py`: live source equality in
  `_validate_development_contract` and bundle-lineage checks.
- `scripts/24_freeze_model_suite.py`: Stage-09/09b/16/25 source equality before
  freezing.
- `scripts/verify_release.py`: independent receipt and Git replay checks repeat
  the single-source contract.

These checks are evidence that a bridge is a new governed protocol feature, not
an interpretation already supported by the repository.

# P0-A0 governance closure design

| Field | Value |
| --- | --- |
| Date | 2026-08-01 |
| Current verdict | `CODE_FIXED / NEW_RUN_ID_BOUND / GOVERNANCE_UNSEALED / FULL_KEY_EVIDENCE_PENDING` |
| Status | **DESIGN ONLY — implementation would change the protected source hash** |
| Target-access state | No canonical 2021–2023 confirmation directory was found or read during this audit |

## 1. What is already satisfied

- Commit `008e1f34` is an ancestor of the current HEAD.
- The pre-fix run `bb02498a8396ea7c6110` is void.
- The current candidate `7cb2bfb18c1f9aa3dba7` is bound to the post-fix
  `source_sha256=19289553aa0929bdb5803a8a3eaa96b38a651b3d52da441fb6298f2ac9228b55`.
- Development calibration uses independent horizon masks on the 2018 calibration
  split; train/validation/test retain their declared joint-label semantics.
- The composite loss implements MSE/σ², pinball/crossing/residual terms divided by
  σ and unscaled BCE, rejecting non-positive/non-finite scale.
- Static tests exist for °C↔°F term and ratio invariance, although they were not
  rerun during the protected Stage-09 window.
- The executable five-row JSON family is exact and the opening/claim validators
  already require each structured test ID once.

## 2. What prevents closure

1. `protocols/route_a_calibration_inclusion_erratum_v1.json` remains
   `DRAFT_PRELABEL_OUTCOME_FREE` and has no separate seal.
2. No dedicated validator checks that erratum or binds it into chronology,
   opening authorization, opening receipts and release verification.
3. Development-calibration and confirmation inclusion are duplicated code paths;
   no full 2018 direct test proves per-horizon key/target/count/hash equality.
4. Existing Stage-09b semantics retain one aggregate calibration registry hash,
   not h1/h3/h7 counts and hashes separately.
5. The loss correction has static unit tests but no formal end-to-end unit
   equivalence receipt.
6. The sealed Markdown protocol contains the H1/H3 prose collision documented in
   `docs/ROUTE_A_H1_H3_CLARIFICATION_DRAFT.md`; the structured family has no
   formally sealed priority rule over that prose.

## 3. One integrated future erratum

To avoid three more interacting patches, create one immutable
`route_a_p0_a0_governance_erratum_v1` after the current Stage-09 candidate reaches
an explicit terminal state. It should bind:

- commit `008e1f34` and the exact corrected source blobs;
- the existing calibration discrepancy draft;
- per-horizon development/confirmation inclusion equivalence;
- the dimensionless composite-loss and affine-unit contract;
- the existing five-test family SHA-256;
- the rule that the Markdown “H3 at one day” sentence is stale prose and does not
  create/remove a test;
- no change to candidate, reference, horizon, margin, alternative, estimand,
  multiplicity or decision rule;
- an outcome-free attestation and absent target-path check.

Create the seal in a later independent commit, binding the erratum document
commit and all prior governance seals. Do not silently edit the historical sealed
Markdown.

## 4. Required implementation changes

At the next allowed source boundary:

1. add an exact erratum validator and tests;
2. bind its document/seal hashes into chronology, model-suite freeze, opening
   authorization, opening intent/receipt, release verifier and archive manifest;
3. add a full-panel 2018 equivalence test that compares development-calibration
   and confirmation-mode h1/h3/h7 keys, targets, counts and SHA-256 values;
4. emit per-horizon calibration count/hash fields in the Stage-09b semantic
   receipt while retaining the aggregate registry binding;
5. add an end-to-end deterministic °C/°F toy-training receipt or downgrade the
   formal claim to static objective covariance only;
6. run the clean-room/stationary test plan in
   `docs/CLEANROOM_REPRODUCTION_DESIGN.md`.

## 5. Run-identity consequence

`DEFAULT_SOURCE_PATTERNS` includes protocols, source, scripts and tests. Every
change above creates a new source hash. The current formal validator also requires
a Stage-09 receipt's source hash to match the live source.

Therefore, under the **current single-source implementation**, after P0-A0
governance closure:

- `7cb2bfb18c1f9aa3dba7` remains development/diagnostic evidence;
- it cannot be relabelled as belonging to the new source;
- all source-bound formal stages must start from a new run identity;
- the new erratum seal hash must enter the opening namespace before any target
  acquisition.

There is one design-only alternative: first complete the entire A1 producer suite
under the unchanged source, then admit it without relabelling through the explicit
dual-hash contract in `docs/STAGE09_CROSS_SOURCE_PROVENANCE_BRIDGE_DESIGN.md`.
That path is not implemented or approved. It avoids a later formal rerun only if
the native producer receipts and every fail-closed bridge gate pass; otherwise a
new-source rerun remains mandatory.

## 6. Safe execution cut

1. Keep all current hashed paths unchanged while the user considers the Stage-09
   terminal/relaunch decision.
2. Do not start Stage-09b/16/25 until the user chooses between (a) a complete
   same-source A1 producer suite followed by a separately implemented dual-hash
   adoption, or (b) a full formal chain under the future A0 source.
3. Release the source boundary only after either (a) a valid completion receipt,
   or (b) explicit user abandonment of the candidate. Under (b), stop/reconfigure
   the watcher through its governed lock procedure, verify no related formal
   process is running, record the candidate as terminal diagnostic, and only then
   make the integrated erratum/document commit.
4. Create the seal in a separate commit, then implement/bind validators.
5. Run stationary tests and compute the final governance source identity. Start
   the entire formal model chain once unless an independently verified A1
   producer suite qualifies for the explicit adoption path.
6. Preserve the later human gate before any opening authorization or target
   request.

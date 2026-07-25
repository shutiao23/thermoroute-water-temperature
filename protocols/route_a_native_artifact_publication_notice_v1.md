# Route-A native-artifact publication notice v1

Status: pre-label corrective engineering record. No 2021--2023 target water
temperature was requested, acquired, inspected, scored, or inferred before
this notice and its implementation were written. This record supplements the
earlier native-thread enforcement notice; it does not alter any scientific or
statistical choice.

Evidence boundary: this is a repository-internal honest-owner attestation backed
by the recorded local files and Git history.  It has no independent custodian,
external timestamp, or write-once proof, so it must not be represented as
independent evidence that no person accessed the target outcomes.

## Discovery

On 2026-07-24, adversarial review of the first rerun under live native-thread
enforcement found a second, distinct resumability gap. The process-lifetime
thread limiter, environment variables, Torch settings, and all four discovered
BLAS/OpenMP pools were at one thread at startup. No sampled live-policy check
reported thread drift; the process was not continuously monitored.
However, several reusable artifacts could be published before the next live
policy assertion.

In particular, Stage 09 could train a LightGBM member and publish its native
model object and shard manifest before a post-fit assertion. Transient live
thread counts are deliberately excluded from the stable runtime identity, so a
failed process and a later compliant retry can share the same run id. A later
retry could therefore accept a checksum-valid shard without proving that the
policy was live at that shard's original publication boundary. Neural
checkpoints, member predictions, model bundles, later formal stages, replay
receipts, and the one-time opening had analogous boundary windows.

This is a provenance and transaction-integrity defect. It is not evidence that
the interrupted models were actually trained with more than one thread.

## Disposition of the interrupted run

- Git commit: `efde4df631b08b73c3d7d6712bfe6b8b3bc1446c`
- run id: `658febb19ff3c72b9c21`
- source SHA-256:
  `dd90adae9a018486dff5f20aef6b55537fb16cc7e01b8c4cdada0767cc556fbc`
- runtime SHA-256:
  `a8a04e905f06b06e9e559efdfb8da200767ce640f2bea3bb47887d1e57290864`
- run-manifest SHA-256:
  `c4838782db19dd9450b480df026b06af9221fb2f5c0f2f4734cd8fa02220a704`
- completed cache state at interruption: seven LightGBM objects and seven shard
  manifests, with no complete shard-set manifest
- canonical predictions: absent
- model/component pointer: absent
- Stage-09 completion receipt: absent
- confirmation outcomes requested or read: no

The 39 MB diagnostic directory was moved intact out of the active formal
namespace to the local quarantine
`stale-artifacts-20260724/native-cache-publication-gap-run-658febb19ff3c72b9c21`.
It is recoverable for audit only within the honest-owner local repository
boundary, and is inadmissible for cache reuse, model freezing, performance
reporting, or later opening. It is outside the release archive; the archive
alone cannot recover or independently replay this interrupted run. A new clean
source commit must create a new source hash and run id.

## Corrective publication contract

The repaired contract is `compute or replay -> fully stage and fsync -> assert
the live numerical policy -> atomically publish`. The assertion is passed into
the publication primitive, rather than being placed only near an outer caller.

The implementation applies that rule to:

1. training checkpoint payloads and sidecars, including load and sidecar
   recovery paths;
2. LightGBM content objects, shard manifests, and complete shard-set manifests,
   including cache acceptance;
3. member and canonical prediction bytes and lineage sidecars;
4. Torch and LightGBM inference-bundle directory publication;
5. Stage-09, Stage-09b, Stage-16, and Stage-25 component pointers and/or
   completion receipts, including Stage-16's exact parent, selection,
   seed-prediction, derived-prediction, bundle, pointer, and replay closure;
6. Stage-24's content-addressed model-suite document, direct opening-registry
   alias, and current-suite pointer;
7. Stage-27's create-only development-replay receipt; and
8. the Opening trusted-directory rename, one-time receipt, receipt sidecar, and
   sidecar-recovery path.

Completion publishers validate the candidate closure, assert, stage the exact
receipt bytes, assert again inside the atomic writer immediately before
`replace`, `link`, or `rename`, reopen and validate the authoritative artifact,
then assert before reporting success. Existing valid caches are also rejected
when the current live policy is not compliant.

Fault-injection tests cover first-boundary and later-boundary failures. They
require the receipt, sidecar, manifest, pointer, or directory publication that
confers authority to remain absent or unchanged. Depending on which boundary
fails, recoverable residue may be private staging, a content-addressed orphan,
or a payload already at its final path but still inadmissible without the next
required authority artifact. Examples of the last case include a checkpoint
payload, prediction bytes without their lineage sidecar, and the Stage-09b
combined Parquet without a valid completion receipt. Validators and cache
acceptance must reject such incomplete residue; retry may recover it only by
completing and revalidating the full transaction. This contract does not claim
that every failed publication leaves only private staging.

## Scientific invariance

This correction changes no station cohort, date split, variable, target,
horizon, architecture, loss, hyperparameter, seed, calibration rule, estimand,
comparison, noninferiority margin, multiplicity rule, or outcome-access rule.
It changes only when an execution artifact is allowed to become authoritative.
All formal artifacts must be regenerated.

The frozen protocol's phrase `2021-2025` in its external-site forbidden-
selection list is broader than the frozen 2021--2023 target interval. It is
therefore conservative, not permission to use 2024--2025 information and not a
change to the target interval. The sealed base protocol is not silently edited.

## Unrelated legacy-site semantics

This engineering correction has no bearing on `b1`, `s2`, or `p3`. They are
ordinary monitoring stations, not reservoirs. No verified metadata establish any
upstream/downstream ordering, hydraulic connectivity, regulation status, or
travel time among b1, s2, and p3.

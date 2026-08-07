> Historical record (superseded by the conventional design).

# Route-A PRE/POST static claim audit

| Field | Value |
| --- | --- |
| Date | 2026-08-01 |
| Verdict | **PASS_PRE_BYTES_AND_COVERAGE / POST_NOT_PRESENT / GOVERNANCE_UNSEALED** |
| Scope | Read-only static inspection of the frozen PRE documents, five-test registry and result placeholders |
| Exclusions | No Stage-09 process, source, data, output, lock or log was changed; no 2021--2023 label was requested or read; no validator or test suite was executed |

This audit is evidence about the current pre-opening text only. It is not an
opening authorization, protocol amendment, completion receipt, empirical result
or submission approval.

## 1. Frozen PRE document bytes

The eight documents listed in
`protocols/route_a_claim_registry_v1.json:/preopen_document_sha256` all match
their registered SHA-256 values:

| Document | Registered/current SHA-256 | Result |
| --- | --- | --- |
| `README.md` | `42255e3d9e7ccc36363664a551730db97b175a3d3b6c0bada38f7c8deeac7f27` | PASS |
| `paper/ThermoRoute_paper.md` | `4843656b5858c8d0f997d0bc5dcc55aa79e77fc8c974b1b7a1ee17ce519104e7` | PASS |
| `paper/highlights.md` | `b312c8a1010287f6c7243c7ffca1e7b978062cb93a69eb85eb1e2c8bc0e5aa8b` | PASS |
| `paper/cover_letter.md` | `da55fed9ff52c4678fd47a74d96d74ae286a5295de3aa19fb6323ee9d9f5ab56` | PASS |
| `paper/agu_submission/ThermoRoute_WRR.tex` | `34a34af4f727f57831bb48bb3520b39a5c4cd4e7d1d4999675961e908824da92` | PASS |
| `protocols/route_a_native_thread_enforcement_notice_v1.md` | `f2b60bf9dbe7108ae1498a322e86d66033c6cc82a3febcda11271d67ea40977e` | PASS |
| `protocols/route_a_native_artifact_publication_notice_v1.md` | `3e2075a5ee3af6f4216ad96ab0d647c3d63ea16be2fd6e43df5754d8e299d81f` | PASS |
| `protocols/legacy_three_site_semantics_notice_v1.md` | `2a41271ad7f9d927cb4f1a5f23b9c0abb786353e98ce15e581065e64eb87122f` | PASS |

The comparison is against the JSON registry rather than a separately maintained
hash list.

## 2. Permanent limitation coverage

The canonical Markdown contains exactly eight opening markers, eight structured
`ROUTE_A_CLAIM_ENTRY` records and eight closing markers. These correspond to the
registry's eight `required_permanent_coverage.claim_ids`, each with
`EACH_EXACTLY_ONCE` multiplicity:

1. not ungauged;
2. no network-routing or hydraulic-travel-time proof;
3. not an operational archived-vintage replay;
4. no regulatory or ecological meaning for the numerical thresholds/margin;
5. no physical, deployment, regulatory or distribution-free safety guarantee;
6. no causal-mechanism identification;
7. not nationally representative; and
8. no conditional-coverage or CRPS claim.

The registered PRE TeX byte projection contains the same eight limitation
sentences. It also states that empirical results are intentionally absent and
that no result can support superiority, non-inferiority, equivalence, parity or a
U.S.-river superpopulation claim under the already-failed cluster gate.

## 3. Five-test identity and placeholder coverage

The authoritative family at
`protocols/route_a_confirmatory_v1.json:/primary_inference_contract/confirmatory_family`
contains exactly:

| Test ID | Candidate | Reference | Horizon | Margin |
| --- | --- | --- | ---: | ---: |
| `H1-h1-vs-damped` | ThermoRoute | DampedPersistence | 1 | 0.00 C |
| `H1-h3-vs-damped` | ThermoRoute | DampedPersistence | 3 | 0.00 C |
| `H1-h7-vs-damped` | ThermoRoute | DampedPersistence | 7 | 0.00 C |
| `H2-h3-vs-lightgbm` | ThermoRoute | LightGBM | 3 | +0.05 C |
| `H2-h7-vs-lightgbm` | ThermoRoute | LightGBM | 7 | +0.05 C |

`route_a_claim_registry_v1.json` maps those five IDs one-for-one to five
POST-only `result_claim_specs` and requires each claim ID exactly once after a
verified opening. The Markdown methods and the PRE TeX projection describe the
same five candidate/reference/horizon/margin rows in prose. The figure/SI
skeleton reserves the same five-row geometry and currently contains 83 exact
`[pending — 探索期数据，不可写入结论]` tokens across its result shells.

The canonical Markdown contains the explicit pre-opening statement, “No formal
result block is present in this byte-frozen pre-opening snapshot.” No generated
result block marker is present. Therefore missing POST claims are expected, not a
coverage failure, at the current phase. POST content cannot be audited until a
valid opening and completion receipt exists and the deterministic renderer has
produced it.

## 4. Residual blocker: unsealed H1/H3 prose clarification

The structured family is internally exact, but the sealed narrative protocol has
the known stale “H3 at one day” sentence documented in
`docs/ROUTE_A_H1_H3_CLARIFICATION_DRAFT.md`. The outcome-free clarification is
still a draft outside the protected source tree and is not a protocol amendment.
Accordingly, the current PRE package passes byte and structured-coverage checks
but does **not** close P0-A0 governance. Before any opening authorization, the
clarification must be reviewed, sealed and bound without changing the five
structured rows or using outcome information.

## 5. Fail-closed disposition

- Keep the current PRE manuscript and TeX bytes unchanged while Stage 09 is
  protected.
- Do not hand-fill any result or figure cell from exploratory caches.
- Do not run the POST renderer or `--require-complete` path without the unique
  verified opening/completion receipt.
- Treat all five eventual Route-A rows as fixed-cohort descriptive results; the
  maximum of 15 HUC2 groups cannot pass the predeclared minimum-30 gate.
- This audit should be rerun after any authorized protocol governance change and
  again against the exact POST artifact set after receipt generation.

# SI04 — protocol, amendments and chronology

**Status:** superseded by the conventional design; the pre-registration protocol apparatus was removed with the route-A deletion (docs/C1_DELETION_EXECUTION_ORDER.md).

This is a navigation layer for frozen protocol sources. It does not reseal,
amend, authorize or execute any stage. Any claimed completion state is
`[pending computation]` unless separately verified by its canonical
receipt.

## Protocol map

| Source path | PRE role described by the manuscript | SI rule |
|---|---|---|
| `protocols/route_a_primary_v1.json` | registered study family, chronology and evaluation design | cite as the source; do not copy/edit decisions here |
| `protocols/route_a_inference_amendment_v2.json` | outcome-free comparison-eligibility overlay | cannot be overridden by a table or caption |
| `protocols/route_a_model_matrix_amendment_v1.json` | prospectively superseded control replication/matrix rule | does not alter five registered comparisons |
| `protocols/route_a_probability_metric_erratum_v1.json` | draft correction to probability-metric application | preserves the frozen metric contract |
| `protocols/route_a_temporal_coverage_policy_v1.json` | temporal-coverage audit policy | receipt binding is required before any audit result |
| `protocols/route_a_claim_registry_v1.json` | machine-readable claim-rendering authority | no handwritten replacement claim is permitted |

The matching `*_seal_*.json` files are seal references, not a license to
recompute, modify or reopen a protocol from SI.

## Outcome-free inference overlay

`paper/ThermoRoute_paper.md` §4.2 is the source for the following
outcome-free gate inputs: at least 30 reportable clusters, effective-cluster
fraction at least 0.75, largest-cluster share below 0.25, and passing
falsification evidence. Their role is comparison eligibility, not performance.
Unknown, missing or failed components fail closed. The same PRE source records
at most 15 HUC2 groups, so the cohort permanently fails the minimum-cluster
component before outcome access. Therefore every formal row is fixed-cohort
descriptive with the verdict
`descriptive (fixed cohort, few clusters)`; any interval and p-value are
approximate sensitivities, not a claim override.

## Amendment and erratum boundaries

- `paper/ThermoRoute_paper.md` §4.1 describes the model-matrix amendment:
  five fixed seeds for each Stage-09 control and an exact 45-member Stage-09b
  matrix are design/budget geometry, not results. It did not alter the five
  primary rows, splits, margins, primary models, CQR/Platt contracts or
  claim rule.
- The probability-metric erratum records that the three-quantile pinball mean is
  evaluated on nominal member-averaged heads before CQR; coverage/width use the
  deployed CQR interval and event metrics use post-Platt probabilities. Earlier
  incompatible artifacts are withdrawn and cannot fill SI cells.
- `paper/ThermoRoute_paper.md` §2.2 describes the 2018–2020 product-bridge
  gate as a post-seal reproducibility strengthening step. Its date span is a
  PRE engineering-gate scope, not outcome evidence; it does not alter
  hypotheses, estimands, model registry, margins or decision rules.

## Chronology projection fields

A later chronology appendix must bind the following fields rather than narrate
them from memory:

```text
protocol_ref
protocol_seal_ref
amendment_ref
amendment_seal_ref
claim_registry_ref
model_suite_ref
authorization_ref
opening_ref
receipt_ref
source_tree_hash
chronology_status
```

Each reference needs an exact path, SHA-256 and relationship role in
`post_paper_evidence_v1.json`. This scaffold holds no such bound values. The
placeholder for a missing or unresolved field remains
`[pending computation]`.

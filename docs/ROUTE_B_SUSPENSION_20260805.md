# Route B: suspended for this submission

**Date:** 2026-08-05
**Decision:** SUSPENDED — not abandoned destructively; the option is preserved at near-zero cost.
**Applies to:** the WRR submission currently in preparation.

---

## 1. Decision

Route B — the enlarged-cohort route intended to support a **superiority** claim — is
**removed from the critical path** of this submission. No Route-B compute, download,
cohort construction, or opening will be undertaken before the Route-A manuscript is
submitted.

## 2. What Route B required

| Requirement | Status |
|---|---|
| ≥ 50–60 reportable clusters (HUC4 / network clustering) | frozen Route-A cohort has at most 15 HUC2 groups |
| New, substantially larger station cohort | not acquired |
| Protocol freeze + seal **before** any opening | not done |
| New model training over the enlarged cohort | not started |
| Independent second opening | not started |
| Formal superiority analysis | not started |

Meeting the cluster requirement means a cohort spanning most of CONUS: a new
discovery pass, a new rate-limited multi-day download of predictors, a full retrain,
a second sealed opening, and a second QA/QC pass. Realistically **two to three
months**, and it is a second paper's worth of work, not a section of this one.

## 3. Why suspending is the right call

1. **It is not what this literature rewards.** Water-temperature hindcasting papers
   in WRR are accepted on the honesty and strength of the comparison, not on a
   formal superiority verdict. The existing baseline suite — persistence, damped
   persistence, air2stream, global LightGBM, per-station LightGBM, global LSTM — is
   already stronger than the field norm, where a comparison against persistence
   alone is common.

2. **The descriptive result is already the publishable contribution.** Development-
   period skill against persistence is positive at every horizon and every transfer
   arm, and the leave-HUC2-region-out arm is the honest hard case. Adding a
   superiority verdict on a different cohort would not change what a reader learns.

3. **It would delay the paper by a quarter for a claim class the reviewers did not
   ask for.**

4. **The inference-gate story is stronger without it.** The manuscript's Discussion
   argues that pre-specifying a cluster-structure gate and honouring its failure is
   a contribution. Immediately constructing a second cohort chosen to pass that gate
   would undercut the argument.

## 4. What is preserved, and how

Suspension is not deletion. The following are retained so Route B can be resumed as
a separate study without repeating design work:

- `docs/ROUTE_B_MEASUREMENT_MISSINGNESS_UQ_DRAFT.md`
- `docs/ROUTE_B_MODEL_BUDGET_IDENTIFIABILITY_DRAFT.md`
- `docs/ROUTE_B_POWER_MDE_DRAFT.md`
- `docs/ROUTE_B_PRELABEL_DECISION_PACKAGE.md`
- `docs/ROUTE_B_COMPARATOR_PROVENANCE_20260801.md`
- `docs/ROUTE_B_OWNER_CUSTODIAN_DECISION_FORM.md`

**No Route-B document is to be deleted or rewritten.** They are the design record of
a deferred study.

### 4.1 The protocol-seal question

A Route-B protocol seal must precede any Route-B opening, or the superiority claim is
permanently forfeit. Sealing costs one JSON file and one commit.

**It is deliberately NOT done now**, because `protocols/**/*.json` is inside
`source_tree_hash` and adding a file there would invalidate all four Route-A training
receipts — the same two-day cost documented in
[`CODE_FREEZE_DISCIPLINE_20260805.md`](CODE_FREEZE_DISCIPLINE_20260805.md).

The seal is therefore queued into the post-opening remediation lineage; see
[`SOURCE_HASH_SCOPE_REMEDIATION_PLAN.md`](SOURCE_HASH_SCOPE_REMEDIATION_PLAN.md).
Route B loses nothing by this, because it cannot open before then in any case.

## 5. Manuscript consequences

- The manuscript makes **no** superiority, non-inferiority, equivalence or parity
  claim. This was already required by the permanent descriptive wording in
  `docs/B02_PERMANENT_DESCRIPTIVE_CLAIM_WORDING.md`; suspending Route B simply means
  no future arm will change that.
- Future work may state that an enlarged-cohort inferential study is designed and
  deferred, citing the retained Route-B design documents. It must not promise a
  result or a date.

## 6. Reversal condition

Route B resumes only when all of the following hold:

1. The Route-A manuscript is submitted.
2. A cohort meeting `n_clusters ≥ 30`, `effective_cluster_fraction ≥ 0.75`,
   `largest_cluster_share < 0.25` is demonstrated **outcome-free**, from metadata
   alone, before any protocol is sealed.
3. The post-opening remediation lineage has landed, so that the Route-B protocol seal
   can be added without a further retrain.

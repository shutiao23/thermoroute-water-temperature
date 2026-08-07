> Historical record (superseded by the conventional design).

# Stage-19 disposition: degenerate (zero-width) nominal intervals

**Date:** 2026-08-05
**Status:** DECIDED — Stage-19 and Stage-10 are not produced for the current submission.
**Decision owner:** project lead (delegated execution).
**Applies to:** Route-A development chain, WRR submission.

---

## 1. Summary

Stage-19 (probabilistic evaluation: PICP, three-quantile pinball, reliability, Brier)
and its dependent Stage-10 are **deliberately not produced** for this submission.

The blocking condition was previously described in project notes as *"quantile
crossing"*. **That description is factually wrong.** Direct measurement of the full
member-level prediction table shows:

| Quantity | Measured value |
|---|---|
| Member-level rows carrying complete quantile heads | 26,993,675 |
| **Strict quantile-ordering violations** (`q05 > q50` ∨ `q50 > q95` ∨ `q05 > q95`) | **0** |
| Rows tripping the Stage-19 contract | **135** |
| Nature of those 135 rows | `q05 == q50 == q95` exactly (bit-identical float64) — zero-width nominal interval |
| Rate | 0.0005 % (≈ 1 in 200,000) |
| Maximum monotonicity violation | exactly 0.000 °C |

There is **no quantile crossing anywhere in the development panel.** The contract is
tripped only by *degenerate but correctly ordered* intervals.

## 2. What actually happens

`scripts/19_probabilistic.py` rejects a row when

```python
raw_crossing = (heads[:, 0] > heads[:, 1])      # q05 > q50
             | (heads[:, 1] > heads[:, 2])      # q50 > q95
             | (heads[:, 0] >= heads[:, 2])     # q05 >= q95   <- equality included
```

and the frozen protocol prohibits evaluation-time repair
(`"evaluation-time repair is prohibited"`, `"Stage19 never repairs them"`).

The third clause uses `>=`, not `>`. A row where the three quantile heads return the
same number is therefore fatal, even though its ordering is valid.

### 2.1 Characterisation of the 135 rows

| Dimension | Distribution |
|---|---|
| Distinct sites | 12 |
| Top sites | 04027000 (54), 05054000 (33), 06623800 (23), 04067500 (10), 05057000 (5) |
| Basins | all in HUC 04 (Great Lakes), 05 (Upper Mississippi), 06 (Missouri) — cold-winter regions |
| Horizon | h=1: 129 rows; h=3: 6 rows; h=7: 0 rows |
| Model | LightGBM 123; LightGBM-perstation 12; no other model affected |
| Seed | 0:33, 1:30, 2:13, 3:28, 4:31 (not seed-specific) |
| Predicted value | min −0.0993 °C, median 0.0002 °C, max 18.8417 °C |
| **Rows with observed `y_true` < 1.0 °C** | **95.6 %** (observed min −0.100 °C, median 0.000 °C) |

### 2.2 Interpretation

This is a physically coherent result, not a numerical defect.

At ice-affected sites in winter, daily mean water temperature is pinned near the
freezing point. The conditional 5th, 50th and 95th percentiles of water temperature
genuinely coincide there, and a one-day-ahead prediction that returns a zero-width
interval is the correct answer, not a broken one. The concentration at h=1
(129 of 135 rows), the confinement to twelve cold-region sites, and the fact that
95.6 % of affected rows have an observed temperature below 1 °C are all consistent
with this reading.

The remaining LightGBM-perstation cases (e.g. site 01542500,
`q05 = q50 = q95 = 13.110679188150272 °C`) are constant-leaf predictions from a
per-station model fitted on a small per-site sample. That is a known and expected
property of per-station gradient boosting, and the per-station foil is an
exploratory comparator, not a primary model.

### 2.3 No delivered product is degenerate

The split-CQR step computes `cqr_q05 = q05 − δ` and `cqr_q95 = q95 + δ` with δ ≥ 0,
and the code separately requires the calibrated interval to have strictly positive
width. Every one of the 135 rows would have been widened to a valid interval before
any interval metric was computed. **Only the pre-CQR nominal check refuses.**

## 3. Why we are not fixing it

The one-line fix (`>=` → `>`, plus a recorded degeneracy count) would edit
`scripts/19_probabilistic.py`.

`scripts/**/*.py` is inside `DEFAULT_SOURCE_PATTERNS`
([`src/thermoroute/repro.py`](../src/thermoroute/repro.py)), so the edit changes
`source_tree_hash`. Every consumer of the four training receipts then fails closed —
for example [`src/thermoroute/model_suite.py`](../src/thermoroute/model_suite.py):

```python
if identity["source_sha256"] != source_tree_hash(root):
    raise ModelSuiteError("Stage-9 completion receipt is stale for the run or current source")
```

The cost of the one-line fix is therefore a **full retrain of Stage-09, 09b, 16 and
25** — approximately two days of wall clock, plus the restart risk demonstrated by
the preceding five lineages. Relaxing the check via the protocol files is not a way
out either: `protocols/**/*.json` and `protocols/**/*.md` are also hashed.

Weighed against that, Stage-19 delivers:

- metrics whose own erratum classifies them as
  `DESCRIPTIVE_NOT_IN_CONFIRMATORY_FIVE_TEST_FAMILY` with `inference_allowed: false`;
- i.e. **nothing in the confirmatory five-comparison family depends on Stage-19.**

Interval validity is separately available from Stage-22 (adaptive conformal), which
completed successfully over 249,072 keys: split-CQR marginal coverage 0.909 overall,
0.905 / 0.910 / 0.912 at leads 1 / 3 / 7, with widths and interval scores reported.

**Decision: do not amend. Do not retrain. Report honestly.**

## 4. How this is reported

### 4.1 In the manuscript (Limitations, cross-referenced from Discussion)

> The pre-registered probabilistic-metric contract treats a zero-width nominal
> prediction interval as a fatal condition and prohibits evaluation-time repair.
> In the development panel, 135 of 26,993,675 member-level rows (0.0005 %) have
> identical 5th, 50th and 95th percentile predictions, predominantly at ice-affected
> sites where daily mean water temperature is pinned near the freezing point and the
> three percentiles genuinely coincide. No strict quantile-ordering violation occurs
> anywhere in the panel. Because the contract was frozen before any access to the
> evaluation labels, and because relaxing it would have invalidated the frozen
> training receipts, we did not amend it; the probabilistic metric suite is
> therefore not reported. Interval validity is instead reported from the
> split-conformal analysis, which is unaffected.

This paragraph is evidence that the pre-registration bound the authors against their
own convenience. It belongs next to the inference-gate discussion, and should be
written in that register — not as an apology.

### 4.2 In the SI

Report Table §2.1 of this document verbatim, including the zero strict-crossing
result.

### 4.3 Wording ban

The phrase **"quantile crossing"** must not appear in the manuscript, SI, figure
captions, figure specs or renderer code in reference to this issue. The correct term
is **"zero-width (degenerate) nominal interval"**. Zero crossings were observed.

## 5. Consequences accepted

| Lost | Mitigation |
|---|---|
| Stage-19 probabilistic metric suite (PICP, pinball, reliability, Brier) | Stage-22 split-conformal coverage/width/interval score |
| Stage-10 USGS calibration/latent diagnostics (cascades from 19) | not on the confirmatory path |
| Original Fig 3 and Fig S5 bindings | rebound onto Stage-22 evidence — see [`FIGURE_PLAN_STAGE19_INDEPENDENT_20260805.md`](FIGURE_PLAN_STAGE19_INDEPENDENT_20260805.md) |
| Development chain completeness | 17 of 19 stages; recorded, not concealed |

## 6. Deferred remediation

The `>=` → `>` change is scheduled into the single post-opening remediation lineage
described in
[`SOURCE_HASH_SCOPE_REMEDIATION_PLAN.md`](SOURCE_HASH_SCOPE_REMEDIATION_PLAN.md).
It must not be applied before the one-time opening completes. See
[`CODE_FREEZE_DISCIPLINE_20260805.md`](CODE_FREEZE_DISCIPLINE_20260805.md).

## 7. Reproducing the measurement

Read-only; touches no hashed path.

```python
import pyarrow.parquet as pq, numpy as np
f = pq.ParquetFile('outputs/predictions/usgs_predictions_with_perstation_v2.parquet')
strict = degenerate = 0
for b in f.iter_batches(batch_size=2_000_000, columns=['q05', 'q50', 'q95']):
    q5, q50, q95 = (b.column(c).to_numpy(zero_copy_only=False).astype('float64')
                    for c in ('q05', 'q50', 'q95'))
    ok = np.isfinite(q5) & np.isfinite(q50) & np.isfinite(q95)
    sc = ok & ((q5 > q50) | (q50 > q95) | (q5 > q95))
    strict += int(sc.sum())
    degenerate += int((ok & (q5 == q95) & ~sc).sum())
print(strict, degenerate)   # -> 0 135
```

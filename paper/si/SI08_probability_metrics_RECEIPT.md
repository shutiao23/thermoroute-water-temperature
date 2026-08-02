# SI08 — probability metrics and reliability bins

**Status:** DRAFT / PRE-OPENING / NOT RECEIPT.

This is an empty projection for nominal member-averaged quantiles, deployed CQR
intervals and post-Platt event probabilities. It does not call three-quantile
pinball CRPS and does not claim conditional coverage.

| Model | Horizon | n | marginal coverage | width (°C) | interval score | pinball | Brier score | Brier skill | log score | discrimination | ECE | calibration slope | calibration intercept | binder row ID |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|
| *(receipt model)* | *(1/3/7)* | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` |

Reliability-bin rows require bin boundaries, denominator, observed frequency,
mean forecast probability, horizon, model and an exact evidence pointer. Empty
or merged bins remain explicit; they are never silently removed.

| Model | Horizon | bin ID/bounds | denominator | mean forecast probability | observed frequency | calibration residual | binder row ID |
|---|---:|---|---|---|---|---|---|
| *(receipt model)* | *(1/3/7)* | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` |

Any metric not defined by the frozen probability contract must be explicitly
`NOT_REGISTERED`, not silently omitted or substituted. Brier skill requires a
bound reference; discrimination must name its registered statistic. Every
metric, bin boundary and bin statistic follows the README cell-level binder
contract.

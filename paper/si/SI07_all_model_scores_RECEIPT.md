# SI07 — all-model exact-common-key scores

**Status:** DRAFT / PRE-OPENING / NOT RECEIPT.

Rows are generated only from receipt-bound predictions after the renderer proves
one exact common key set for every compared model. Development scores are not
admissible.

| Model | Horizon | paired keys | stations | RMSE (°C) | MAE (°C) | bias (°C) | station-balanced summary | binder row ID |
|---|---:|---|---|---|---|---|---|---|
| *(receipt model)* | *(1/3/7)* | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` |

Every cell follows the README cell-level binder contract. The row must bind the
key-registry digest, metric formula, finite-value filter,
station aggregation, unit, rounding rule and parent prediction digest. Selective
model omission rejects the POST build.

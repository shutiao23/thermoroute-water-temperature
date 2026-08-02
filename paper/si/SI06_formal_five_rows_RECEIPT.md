# SI06 — formal five-row result projection

**Status:** DRAFT / PRE-OPENING / NOT RECEIPT.

This file fixes the POST table geometry only. It is not evidence that an
opening occurred. Every result-bearing cell remains
`[pending — 探索期数据，不可写入结论]` until a verified opening receipt supplies
exactly one bound row.

| Row | Candidate − reference | Horizon | Margin / ceiling | Effect (°C) | 95% CI (°C) | stations | clusters | raw p | Holm p | gate/verdict |
|---:|---|---:|---:|---|---|---|---|---|---|---|
| 1 | ThermoRoute − damped persistence | 1 day | 0.00 °C | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` |
| 2 | ThermoRoute − damped persistence | 3 days | 0.00 °C | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` |
| 3 | ThermoRoute − damped persistence | 7 days | 0.00 °C | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` |
| 4 | ThermoRoute − LightGBM | 3 days | +0.05 °C ceiling | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` |
| 5 | ThermoRoute − LightGBM | 7 days | +0.05 °C ceiling | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` |

The POST verifier must reject missing, duplicate, extra or non-finite rows and
must preserve Route A's descriptive-only eligibility status.

## Hidden binder-map contract

The visible table never treats a row-level receipt pointer as sufficient. For
each frozen `test_identity`, the POST manifest must provide cell-specific
`value_id` references for effect, both CI endpoints, station count, cluster
count, raw p, Holm p and gate/verdict. Each `values[value_id]` object must bind
unit, evidence role, exact source pointer, derivation and rounding rule. The
same IDs populate Figure 1d and the main T2 table. Missing, duplicated or
differently rounded mappings reject the render.

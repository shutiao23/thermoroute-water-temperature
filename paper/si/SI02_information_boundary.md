# SI02 — issue-time information boundary

**Status:** DRAFT / PRE-OPENING / NOT RECEIPT.

This document specifies a date-indexed retrospective hindcast boundary. It is
not an operational replay, an ungauged design, a causal transport model, or a
receipt. Any realized availability, score or audit value is
`[pending — 探索期数据，不可写入结论]`.

## Allowed and forbidden information

| Category | PRE rule | Source and role |
|---|---|---|
| issue time | calendar day `t` | `paper/FIGURE1_AND_SI_SKELETON.md` §1a; design symbol |
| target horizons | `h ∈ {1, 3, 7}` days | Figure-1 skeleton §1a; registered geometry |
| WTEMP history | observed target-site history through issue date only | `paper/ThermoRoute_paper.md` §§2.2, 4.1; history-dependent design |
| meteorology | date-indexed Daymet/gridMET values dated no later than historical issue date | PRE manuscript §2.2; retrospective predictor rule |
| frozen references | climatology and damped-anchor inputs | Figure-1 skeleton §1a; model-input role |
| prohibited target | target-date WTEMP | Figure-1 skeleton §1a; leakage prohibition |
| prohibited future predictor | horizon-specific future weather forecast or anticipatory vintage | Figure-1 skeleton §1a; leakage prohibition |

The PRE manuscript records a committed `PASS_EXACT_PRODUCT_BRIDGE` parser/product
compatibility gate for 2018–2020. This SI reports that statement as a frozen
engineering-gate description only; it is neither an opening receipt nor proof of
as-issued provider availability, matching, operational latency, or local-day
alignment.

## Event-level projection schema

Any later information-boundary audit must bind, rather than infer, the following
fields:

| Field | Required meaning |
|---|---|
| `site_id` | stable target-site identity |
| `issue_date` | date at which the prediction is issued |
| `target_date` | outcome date associated with the horizon |
| `horizon_days` | one registered horizon value |
| `variable_id` | predictor or outcome identifier |
| `source_date` | date represented by a predictor value |
| `vintage_or_retrieval_ref` | bound provider/raw-evidence reference, where applicable |
| `admissibility` | allowed, forbidden, missing, or fail-closed state |
| `source_binding` | manifest path, digest and derivation pointer |

These are SI projection fields. They are not an assertion that current raw
provider bytes, archival vintages, target outcomes, or a completed audit are
available in the repository.

## Required rendering language

Every information-boundary figure/table must state all of the following:

1. “date-indexed retrospective hindcast”;
2. no horizon-specific future forecasts and no target outcomes as predictor
   inputs; and
3. not operational as-issued availability, not ungauged prediction, and not
   causal transport inference.

If a future audit needs an unavailable field, it must show
`[pending — 探索期数据，不可写入结论]` or fail closed; it must not substitute a
development cache or reconstructed timestamp.

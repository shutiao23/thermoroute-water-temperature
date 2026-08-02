# Route A H1/H3 wording clarification — draft only

| Field | Value |
| --- | --- |
| Defect | `m-02` in the submission blueprint |
| Date | 2026-08-01 |
| Status | **DRAFT / OUTCOME-FREE / NOT A PROTOCOL AMENDMENT** |
| Scope | Clarifies an internal label collision; does not change any candidate, reference, horizon, margin, alternative, multiplicity rule, estimand, or decision |
| Source boundary | This file is outside the Stage-09 `source_tree_hash`; do not copy it into `protocols/` until a new source boundary is explicitly allowed |

## 1. The ambiguity

`protocols/route_a_confirmatory_protocol.md` currently says all three of the
following:

1. “H1: ThermoRoute improves on damped persistence.”
2. “H3: at 1 day, report the paired effect without a superiority or parity
   claim.”
3. “The confirmatory family is H1 at three horizons plus H2 at two horizons.”

Items 1 and 3 agree with the machine-readable family, while item 2 can be read as
removing the h=1 superiority comparison or creating a sixth hypothesis called
H3. That reading is inconsistent with every structured registry used by the
execution and rendering chain.

## 2. Existing machine-readable authority

The existing five-row family is exact and unchanged:

| Test ID | Candidate | Reference | Horizon | Margin | Registered role |
| --- | --- | --- | ---: | ---: | --- |
| `H1-h1-vs-damped` | ThermoRoute | DampedPersistence | 1 | 0.00 °C | H1 superiority row |
| `H1-h3-vs-damped` | ThermoRoute | DampedPersistence | 3 | 0.00 °C | H1 superiority row |
| `H1-h7-vs-damped` | ThermoRoute | DampedPersistence | 7 | 0.00 °C | H1 superiority row |
| `H2-h3-vs-lightgbm` | ThermoRoute | LightGBM | 3 | +0.05 °C | H2 numerical non-inferiority row |
| `H2-h7-vs-lightgbm` | ThermoRoute | LightGBM | 7 | +0.05 °C | H2 numerical non-inferiority row |

This family appears in `protocols/route_a_confirmatory_v1.json`, is repeated in
`protocols/route_a_inference_amendment_v2.json`, is bound by
`confirmatory_family_sha256=473b2ddae17b353b3499413ecd266e684e9b411412bba5463435e8400aedb799`,
and is mapped one-for-one in `protocols/route_a_claim_registry_v1.json`.

## 3. Proposed future erratum text

When the protected Stage-09 source boundary is released, a separately reviewed
outcome-free erratum should state:

> The sentence “H3: at 1 day, report the paired effect without a superiority or
> parity claim” is a stale prose label and does not define a separate hypothesis.
> The registered family remains exactly the five machine-readable rows: H1
> compares ThermoRoute with damped persistence at horizons 1, 3, and 7 days with
> margin 0.00 °C; H2 compares ThermoRoute with LightGBM at horizons 3 and 7 days
> with the +0.05 °C numerical ceiling. No candidate, reference, horizon, margin,
> alternative, multiplicity family, estimand, or decision rule is changed.

The erratum should preserve the existing document rather than silently editing
its historical bytes, have its own immutable identifier and seal, and be bound
by any later authorization/chronology verifier.

## 4. Reporting consequence

The clarification does not restore confirmatory eligibility. The outcome-free
spatial gate already fixes every one of the five rows as
`DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED` for Route-A claim wording. Therefore:

- the h=1 row must still be rendered exactly once;
- it may report only a fixed-cohort descriptive effect and
  assumption-conditional sensitivity values;
- it may not be rewritten as superiority, non-inferiority, equivalence, parity,
  or national generalization, regardless of its numerical result.

## 5. Acceptance tests for the eventual erratum

1. The canonical serialization and SHA-256 of the five structured comparison
   objects are unchanged.
2. Exactly five test IDs remain, with no standalone `H3-*` row.
3. Opening and claim rendering still require every test ID exactly once.
4. The erratum changes no prediction, label inclusion, statistic, margin, seed,
   or multiplicity input.
5. The erratum is recorded and sealed before any opening authorization and
   before any 2021–2023 outcome acquisition.
6. The Stage-09 candidate is not relabelled or promoted because of this draft.


# SI00–SI16 static integrity audit

| Field | Value |
| --- | --- |
| Date | 2026-08-02 |
| Scope | `paper/si/README.md`, SI00–SI16, `paper/si/figures/README.md`, Figure-1 skeleton/SVG and POST projection design |
| Evidence role | Internal second-pass static review of PRE shells; not a receipt, opening, result audit or independent submission review |
| Outcome boundary | No `outputs/**` artifact or 2021–2023 outcome was read to perform this audit |

## Checks performed

1. All SI00–SI16 files and the FigS1–FigS8 planned inventory exist.
2. Every SI document declares a DRAFT/PRE status; `_RECEIPT` filenames are empty
   projection shells, not receipts.
3. All target-result cells use the exact pending token and no development score
   is inserted.
4. Route A remains fixed-cohort descriptive, the external arm remains explicitly
   history-dependent/not ungauged, and rights rows remain non-approval/PUBLIC
   fail-closed.
5. The point equation in SI03, the Figure-1 skeleton and the materialized SVG is
   the same frozen implementation identity:

   \[
   \hat y_{t+h}=A_{t+h}+\delta\tanh\{(P_{t+h}-A_{t+h}+r_{\theta,t+h})/\delta\}.
   \]

6. SI05 is the five-row comparison geometry; SI09 carries Stage09/09b matrix,
   budget, seed and control routing.
7. SI08 includes coverage, width, interval score, pinball, Brier score/skill,
   log score, discrimination, ECE, calibration slope/intercept and explicit
   reliability-bin routing without calling three-quantile pinball CRPS.
8. SI01 and SI06–SI16 implement the generic cell-level binder contract:
   `rows[binder_row_id].cells[field].value_id`. Every visible value/subvalue must
   bind unit/role, exact source, derivation and formatting independently.
9. Figure/SI target values require a verified opening POST manifest plus bound
   claim registry and inference/QC gates. A development completion receipt alone
   is insufficient.
10. The SVG remains well-formed XML; Markdown diff/whitespace checks pass.

## Result

**PASS_STATIC_SHELL_INTEGRITY**, limited to the scope above.

This result proves only that the no-result shells are internally routed and
fail-closed. It does not prove scientific completion, filled SI, visual/page QA,
receipt-field compatibility, a working POST renderer, rights approval, DOI,
clean-room reproduction or submission readiness. Any future shell or renderer
change requires a new audit version.

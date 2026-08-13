# SI closeout — what is discharged, and what is not (2026-08-13)

A twelve-file audit of the Supporting Information ran every placeholder and
factual assertion against the artifacts that exist. This records the outcome so
the remaining `[pending computation]` cells are a known, bounded list rather
than an open question.

**The audit did not finish.** It hit the account's monthly spend limit with 66
of 141 checks unrun, all of them in the verification stage. Only the 22
discharges that were actually verified against their artifact were applied;
proposed discharges whose verification never ran were discarded rather than
trusted, because the verification stage is the entire reason the audit is
worth anything. The list below is therefore a lower bound on what is
dischargeable, not a complete one.

## Applied

22 verified discharges and corrections, across SI00, SI03, SI04, SI13, SI14 and
the SI README. The substantive ones are recorded in the commit history; the
pattern worth noting is that most were not missing values but *stale* ones --
counts that stopped being right when a figure was added, cross-references to
sections that had been renamed, and blockers that outlived the thing blocking
them.

## Blocked, by file

| File | Blocked cells | Representative blocker |
|---|---:|---|
| `SI04_protocol_and_amendments.md` | 7 | No such file exists anywhere in the repo. Repo-wide, the name appears only in SI04:102 and docs/archive/POST_PAPER_PROJECTION_DESIGN.md, which is arch |
| `SI09_development_controls.md` | 7 | The required final authority does not exist. outputs/models/plain_control_run/stage09b_member_precompute_v1/coordinator_receipt.json self-declares evi |
| `SI10_temporal_coverage.md` | 7 | No availability/calendar-opportunity registry exists. outputs/final/forecast_keys.parquet holds only realized keys: unique (site, issue_date) pairs eq |
| `SI00_inventory.md` | 6 | No rendered S10 artifact exists anywhere in the repo (repo-wide search for *S10* returns only paper/si/figures/render_figS10_architecture_controls.py) |
| `SI15_reproduction_hashes.md` | 6 | No environment or container receipt exists anywhere under outputs/. The repo has requirements.txt, requirements-lock.txt, requirements-lock-py312-hash |
| `SI03_model_equations.md` | 5 | No artifact carries these in symbolic form. The implementation exists and is hash-bound (src/thermoroute/thermoroute.py#DynamicThermalRelaxationPrior. |
| `SI13_external_history_arm.md` | 5 | No such candidate/reference pair exists in any scored artifact. outputs/conventional/paired_effects_2021_2023.csv contains 40 (candidate, reference) p |
| `SI14_missingness_failures.md` | 5 | No artifact records three HTTP failures, or any HTTP failure. outputs/conventional/fetch_failures_2021_2023.json is `[]`; outputs/conventional/cohort_ |
| `SI16_rights_data_dictionary.md` | 5 | No qualified rights reviewer decision exists anywhere in the repo. docs/FAIR_RIGHTS_ACCEPTANCE_MATRIX.md header records status EVIDENCE_COLLECTED + BL |
| `README.md` | 5 | This is a normative contract clause, not a data lookup. SI00 already declares the results authority to be "outputs/final/, protocol v1" and SI06/SI07/ |
| `SI01_cohort_and_registry.md` | 4 | No artifact records a fetch failure. outputs/conventional/fetch_failures_2021_2023.json is the empty list `[]`; all 121 snapshot metadata files under  |
| `SI07_all_model_scores.md` | 4 | Correctly blocked and the status token is the right rendering. docs/AIR2STREAM_SOURCE_BUILD_AUDIT_20260801.md L8 records verdict BLOCKED_NO_COMPILER / |

## Why they are blocked

Every one falls into a category no amount of work inside this repository can
close:

* **Receipts that were never produced.** Several SI sections project fields from
  a one-time opening receipt or a replay receipt that does not exist under
  `outputs/`. The projection contract is specified; the receipt is not.
* **`value_id` binder objects.** SI06-SI16 specify a cell-level binder shape,
  and no `tables.si*.cells.*.value_id` keys exist anywhere in the repository.
  Writing them would be inventing a provenance layer, not recording one.
* **Artifacts that were planned and never built**, including at least one
  protocol path with no history in git.
* **Identity and rights data** — authors, ORCIDs, funding, CRediT, DOI, licence
  scope, independent reproduction — which only the authors hold.

None of these blocks a scientific claim in the manuscript. Every number in
Sections 4.1-4.10 is bound to a scored authority under `outputs/final/`, and
SI20 now documents the method behind the four sections that previously cited
artifact paths and nothing else.

## The rule that produced this list

A placeholder may be discharged only when someone has read the value out of a
named artifact. "It is probably X" is not a discharge, and a plausible number
in a supporting table is worse than an honest placeholder, because a reader
cannot tell the two apart.

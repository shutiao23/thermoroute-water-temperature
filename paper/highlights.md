# ThermoRoute — AGU Key Points

This file is the authoring source for the AGU **Key Points** block. Its three
items are byte-identical to the Key Points block of `paper/ThermoRoute_paper.md`.
That correspondence used to be an instruction and nothing enforced it, so this
file drifted three revisions behind the manuscript — still printing +0.251, a
cohort of 120 rather than 116 reportable stations, and a Key Point 3 that had
been replaced. `check_manuscript_consistency.py` now compares the two and fails
the build when they differ, which is the only version of "keep these in sync"
that survives contact with editing.

**AGU constraint.** At most three Key Points; each is one complete sentence of at
most 140 characters including spaces; each states a fact about the work rather
than a claim about its importance.

## Key Points

1. Confirmatory: seven-day skill falls from +0.250 against persistence to +0.038
   against damped persistence, and every model collapses alike.
2. Confirmatory: a hybrid reporting +0.218 against persistence is
   indistinguishable from the damped baseline; the reference set the sign.
3. Confirmatory: keys a model may decline are 17-25% harder than the keys it
   keeps, across six models from persistence to deep learning.

## What a reader can check, and where

| Key Point | Checkable assertion | Where it is specified |
|---|---|---|
| 1 | Skill is `1 − RMSE(candidate)/RMSE(reference)`, dimensionless, defined at manuscript equation (10), on one common key set over the held-out 2021–2023 window | §3.6, §4.4 of the manuscript; SI05, SI07 |
| 2 | Withholding every target-site water-temperature input (level L2 against L0) under whole-region holdout costs a station-first paired median of 1.716, 1.255 and 1.325 °C at 1, 3 and 7 days, positive at all 116 reportable stations; the plain causal TCN differs from the residual tree by at most 0.009 °C at the same level | §4.7, §4.10 of the manuscript; `outputs/final/information_ladder_v6_authority_v1/`, `outputs/final/architecture_geometry_interaction_v1/` |
| 3 | Realized future meteorology is worth 0.130-0.627 °C against F0; the station-level double difference of that value between L2 and L0 is −0.076 and −0.147 °C at 1 and 3 days, both intervals excluding zero, so the value falls rather than rises when the gauge's readings are withheld | §4.8, §4.9 of the manuscript; `outputs/final/forcing_information_interaction_v1/` |

Key Point 1 is the confirmatory result: its comparison was fixed before the
held-out window opened. Key Points 2 and 3 are post-outcome and descriptive, and
each now says so in its own first word rather than relying on a disclosure
carried by a neighbouring point. That change followed a real failure: three
sections of the manuscript stated that these results were reported *outside* the
Abstract and Key Points while they were the Abstract's core, and every gate
passed because the word "descriptive" appeared on both sides of the
contradiction. `check_placement_denials_are_true` now fails the build when the
manuscript denies a placement the claim ledger records.

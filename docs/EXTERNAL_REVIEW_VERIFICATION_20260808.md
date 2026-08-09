# External review verification record — 2026-08-08

Every claim in the external review was checked two independent ways before any
fix was applied: (1) direct code reading of the producing script, and (2)
reconstruction from the persisted artifacts (logs, parquet columns and counts,
summary JSONs).  This record states the verdict for each claim and the
evidence.  All P0 defects were then fixed and re-verified by injection tests;
the consistency gate now fails on every defect class.

## A. Reviewer's factual claims

| # | Claim | Verdict | Evidence (approach 1 = code, 2 = artifacts) |
|---|---|---|---|
| 1 | 44 references, zero comparative-hydrology authors | CONFIRMED | 1: `grep -c "^@" paper/references.bib` = 44; 2: no sivapalan/blöschl/wagener/klemeš/hrachowitz/kirchner/troch hits |
| 2 | t½ median 6.9 d, IQR 5.0–11.9, range 2.26–52.1 | CONFIRMED | 2: `basin_attributes.parquet` describe (exact match) |
| 3 | Relaxation-rate disclaimer at paper:340 | CONFIRMED | 1: verbatim |
| 4 | Local adaptation means each arm has its own local calibration | CONFIRMED | 1: `run_spatial_factorial.py` local branch fits per-station clim/phi |
| 5 | Geometry effect +0.004…+0.007, 73–76% consistency | CONFIRMED (range 0.0043–0.0090, 72–80%) | 2: matched-site 2×2 recomputation from `spatial_effects.parquet` |
| 6 | Adaptation effect +0.163…+0.198, 92–97% | CONFIRMED (0.1633–0.2017, 91–97%) | 2: same recomputation |
| 7 | LightGBM 7d pooled penalty −0.0036, win 0.47 | CONFIRMED | 2: `spatial_summary.json` (exact) |
| 8 | "same ordering" sentence is wrong for pooled arm | CONFIRMED | 2: sign flip at 7 d pooled |
| 9 | Region cell station counts 60/90/120 vs SI11's "120" | CONFIRMED (7 of 12 region cells at 90; intersection 60 for LGBM 3 d) | 2: parquet nunique; SI11 lines 29–34 |
| 10 | SI11 pooled rows stale (0.677/1.401/1.804; 0.665/1.392/1.815) | CONFIRMED (authority: 0.692/1.551/2.051; 0.667/1.373/1.824) | 2: `spatial_summary.json` |
| 11 | plain-TCN sentence omits candidate/reference | CONFIRMED | 1: paper:1039–1040; storage row candidate=PlainCausalTCN-7var, reference=ThermoRoute (+0.0104/+0.0106/+0.0154) |
| 12 | air2stream 3 d 1.478 vs authority 1.459 | CONFIRMED | 1: md:1033, tex:1216; 2: Table 4.7a row and `station_metrics.parquet` |
| 13 | Table 4.12 All keys −0.073 vs Table 4.6 row 3 −0.069; correct −0.0688 | CONFIRMED | 2: `state[...].delta_rmse.median()` = −0.0730; TR-vs-DP 7d site median = −0.0688 |
| 14 | "116"/"1,023" hardcoded | CONFIRMED | 1: `generate_manuscript_tables.py:153` |
| 15 | Figure 3 caption promises Holm p-values, Table 4.6 has none | CONFIRMED | 1: caption line 748; generator emits no p columns; authority JSON has them (7 d: 6.1e−5 / 1.83e−4) |
| 16 | 1-day GBDT headline lacks paired inference | CONFIRMED | 1: Table 4.6 has only 3 d/7 d TR-vs-LGBM; 2: `paired_effects.parquet` has no h=1 TR-vs-LGBM rows |
| 17 | Gate check #2 never compares numbers | CONFIRMED | 1: `check_manuscript_numbers` loop had no assertion |
| 18 | Gate table regex returns only last row per table | CONFIRMED | 1: `re.findall` with capture group returns the group; verified by injection |
| 19 | Gate passes with all five defects present | CONFIRMED | 2: ran it; exit 0 |
| 20 | 950/1465 candidates excluded for missing flow | CONFIRMED | 1: paper:201 |
| 21 | SI11 `[pending computation]` table at line ~71 | CONFIRMED | 1: verbatim |
| 22 | SI11 claim_eligible hardcoded False / null-simulation never implemented | CONFIRMED | 1: SI11 lines 114–119 |
| 23 | 8+ placeholder sites for author/affiliation/ORCID/funding/CRediT/DOI | CONFIRMED (both md and tex; they are identity data, not fillable here) | 1: grep |

## B. The three software defects

| Bug | Verdict | Evidence |
|---|---|---|
| 1. Resume silently drops cached cells | CONFIRMED | Code: `done` set built from existing parquet, rows never re-read; final `to_parquet` overwrites. Artifacts: run2 log line 1 "resume: 7 cells cached"; 281 of 288 expected result lines; 7 region cells report 90/120 stations; key-level predictions unrecoverable (Bug 3). Reviewer's "6 cells" is corrected to 7 |
| 2. Pooled cache key drops the fold | CONFIRMED | Code: `prep_key = (variant[0], variant[1], h)` with `variant[1] = seed`; fold 0's pooled stats reused for folds 1–3, whose held-out stations are in fold 0's training set (≈33% leakage) |
| 3. Predictions overwritten by metrics | CONFIRMED | Code: line 279 writes predictions to `spatial_effects.parquet`, line 307 overwrites with site metrics; no fold column, no key-level rows survive |

## C. Reviewer's recommendations implemented

| Item | Status |
|---|---|
| Table 4.12 All keys estimand from paired effects, no hardcoded values | DONE (All keys = −0.069, stations 116, median keys 1,060 computed from data) |
| air2stream 1.478 → 1.459 (md + tex) | DONE; SI07's 118-station rows are verified correct on their declared basis (0.719/1.478/1.825) and kept |
| plain-TCN candidate/reference explicit + sign explained | DONE (md + tex) |
| Table 4.6 p (sign flip) and Holm p columns | DONE (values from authority JSON; caption updated; Figure 3 promise now satisfied) |
| SI11 pooled rows regenerated + true station counts + mismatch disclosure | DONE |
| Gate: real numeric comparison, section-scoped | DONE; injection tests I1–I5 all fail the gate as required |
| Gate: full table-block matching (finditer, non-capturing) | DONE; immediately caught a real header-dash drift the old gate missed |
| Gate: tex-vs-md claim consistency | DONE (every claim value must appear in both documents) |
| SI11 pending table removed; neutral gate wording | DONE |
| RQ3 conditional wording in abstract / 4.7 / conclusions | DONE, with the matched-set 2×2 effects (+0.20 adaptation vs +0.005 geometry at 7 d, pooled sign flip) |
| 1-day exploratory paired inference | DONE: +0.035 °C, CI [+0.031, +0.045], p=1.0, 98% stations, labelled outside the frozen family |
| Protocol v2 (ladder, strata, classes, regression forms, N1–N15, P-1–P-4, stopping rules) + seal + DLOG-012/013 + SI04 rows | DONE |
| `run_information_ladder.py` with shards, completeness gate, fold-aware prep keys, registry assertions, L2/L3 masks, mask proof | DONE; golden L0 reproduces the archived cells: region arm worst \|diff\| = 0.0, random arm (seed 0) worst \|diff\| = 4e-15 (720 station-cells per arm); mask proof passes (max \|diff\| = 0.0) |
| Full validation | 845 passed, 1 skipped; gate passes on md + tex; TeX compiles (52 pp) |

## E. Second-round findings (2026-08-08) — verified, then fixed

| # | Finding | Verification | Fix |
|---|---|---|---|
| N1 | gate uses any not all; single-span drift passes (INJ4 reproduced: abstract-only 1.694->1.999 exits 0) | CONFIRMED by injection | used_in audited (`audit_used_in.py`, 22/22 claims), semantics switched to all-mode with explicit `used_in_mode: any` escape hatch; figure-span regex greedy bug found and fixed (swallowed the document tail); warning-level contradiction scan added; INJ4/5/6/7/8 all FAIL now |
| N2 | ladder scores a ~8% superset of the registry; extra keys mostly unobserved-issue-date WTEMP (1,740/2,445 in the audited cell); assertion one-sided | CONFIRMED from artifacts (32,850 vs 30,405 keys; 1,740; 210 null labels; per-station 1,095 vs 844) | registry inner-join scoring; two-sided assertion; training admissibility from the unmasked panel; y_observed NaN->True bug fixed; `--legacy-keys` G1/G2 split |
| N3 | metrics n counts unlabeled keys (constant 1,095 across all 1,440 cells) | CONFIRMED | dropna on y_pred/y_damped/y_true; ndarray means; reportable flag (n>=100) |
| N4 | golden wording overstated (0.0 vs machine precision) | CONFIRMED: region 4.441e-15/660 cells, random 3.997e-15/720 cells (recomputed from artifacts) | docstring/error-message wording; unmatched-station NOTE in G1 |
| N5 | SI07 prose/table mismatch (prose claims 116, values and rows are 118/no-filter) | CONFIRMED, one correction: the MAE/bias columns are POOLED key-level values (0.525/1.066/1.317 match pooled exactly), a third inconsistency | SI07 split into reportable (116) and no-reportability-filter (118) rows with station-median MAE/bias; prose rewritten; 6 ledger claims + SI07 scanned by the gate (injections fail) |
| N6 | SI11 header/alignment mismatch (and more) | CONFIRMED: 4 tables in SI11 + 1 in SI08 | fixed; `check_table_shapes` added to the gate |

## G. Round-2 acceptance results (all reviewer criteria)

| Criterion | Result |
|---|---|
| audit_used_in.py: no FIX rows | 22/22 claims OK |
| gate baseline | exit 0 |
| INJ4 abstract-only 1.694->1.999 | FAIL (was pass) |
| INJ5 whole-doc, INJ6 tex-only, INJ7 conclusions-only 0.250, INJ8 key-points-only 0.038 | all FAIL |
| INJ2 Table 4.7a internal row | FAIL |
| INJ9/INJ10 SI07 reportable/unfiltered drift | FAIL |
| G1 legacy golden (refactor side-effect-free) | 1,380 cells, worst \|diff\| = 4.44e-15 (region 4.441e-15/660, random 3.997e-15/720); DLOG-012 NOTE printed for the 30-station archive gaps |
| G2 production L0 (registry-scored) | 48/48 cells; region_L0_LGBM_s0_f0_h7 = 30,405 keys (was 32,850); 0 unobserved-issue-date keys; 0 null labels; two-sided assertions clean |
| assert_shards_in_registry.py | all 1,416 station-cells two-sided equal |
| ladder_effects n | 136 distinct values, median 1,069, max 1,094 (was constant 1,095); reportable flag: 24 cells below 100 keys flagged |
| Table shapes (SI11 x4, SI08 x1) | fixed; check_table_shapes in the gate |
| Full pytest | 855 passed, 1 skipped (10 new injection tests) |
| TeX compile | OK |

Full ladder (L0-L2, 432 cells) launched persistent on 2026-08-08; L3 will be
appended when basin_attributes_v2.parquet exists (Phase 2 prerequisite).

## H. Reviewer claims corrected by this verification

1. SI07's MAE/bias columns (0.525/1.066/1.317, -0.034/-0.155/-0.243) are not
   118-station station medians: they are POOLED key-level values (reproduced
   exactly as pooled).  The fix goes one step further than the review asked:
   both sets now report station-median MAE/bias.
2. The review's expected post-fix n median (~1,013) is actually 1,069; the
   per-station registry key counts drive it (844-1094 at 7 d).
3. One gate bug the review did not list was found and fixed: the figure-caption
   span regex used greedy `.*` with a `(?=\n\n|\Z)` lookahead, which
   swallowed the document tail (the `\Z` branch always holds at EOF).
4. The used_in audit initially passed 22/22 only because generated-table
   blocks leaked into section spans (the same blank-line regex flaw); with
   correct stripping, 7 claims were repaired and 5 prose gaps were closed by
   printing the values where they are discussed (4.6 intro, Figure 3 caption).

## I. Original reviewer claims corrected by this verification

1. "12 region cells中 6 个少了一个 fold" → 7 of 12 (their own list names seven).
2. "Random-pooled authority 0.668/1.374/1.824" → 0.667/1.373/1.824 (rounding).
3. The pooled-arm numbers in the reviewer's own recomputation are a lower bound because of Bug 2 leakage — the reviewer said this; the manuscript now states it too.
4. SI07's 1.478 values are NOT stale: they are the declared 118-station sensitivity (verified from `air2stream_2021_2023.parquet`: station-median RMSE 0.719/1.478/1.825 exactly). The reviewer's blanket "grep 1.478 must have no hits" cannot apply there; the fix scope is the manuscript prose and Table 4.7a (116-station basis).


## J. Advisory review round (2026-08-09) — D1-D5 verdicts and actions

| Finding | Verdict | Action |
|---|---|---|
| D1 (fatal): no future forcing → damped persistence is approx-Bayes-optimal; the null is a design consequence; Section 5.1 attribution unidentified | CONFIRMED (code: features.py shifts >= 0; air2stream.py Ta_clim roll-out; historical_inputs.py:352 future_nwp False) | protocol v3 + run_forcing_ladder.py; F3 vs F0: 7 d 1.735 -> 1.113 (-0.622 C, 36%); learned-vs-damped at 7 d -0.039 -> -0.603 (P-5 CONFIRMED, interim tree signal); forcing : architecture value ratio ~27:1 |
| D2 (fatal): spatial null has no test power (target's own history always provided) | CONFIRMED (0.20 C pooled vs 0.005 C geometry = 40:1) | L-axis L2/L3 running (protocol v2); conclusions/abstract rewritten conditionally (done in round 1) |
| D3 (structural): paper doesn't know what it is; ThermoRoute over-represented | CONFIRMED (4.2 win rate 0.00; 4.3 <= 0.023 C; 4.8b <= 0.004 C) | Batch A: 3.1 compressed to ~180 words + 1 equation; 4.3 to one paragraph; Figures 2/4 removed; ablation tables to SI09 |
| D4: inference strength vs Key Points overclaim | CONFIRMED (15 HUC2, eff. 9.54, 21.7%, descriptive-only) | Batch A consolidated all caveats into one Section 6; Key Points ranking claims to be downgraded in Phase 6 |
| D5: air2stream disclaimed to uselessness | CONFIRMED (6+ disclaimers) | Batch A: disclaimers consolidated into Section 6 (Batch B continues); official code run = Phase 5b |

Batch A (advisory length plan) applied to both md and tex: sections 4.4/4.5
deleted (-> SI19), 3.5 -> one sentence (-> SI08), Tables 4.7b/4.8a/4.8b/4.11
removed (4.11 deleted as a Figure 3b duplicate; others -> SI07/SI09),
Figures 2/4 removed, Section 6.1-6.3 merged, sections renumbered 4.6-4.8 ->
4.4-4.6, equations (9)/(10) -> (2)/(3).  Manuscript: 99,468 -> 78,432 bytes;
tex: 50 -> 41 pages, 0 overfull/too-narrow warnings; PU estimate ~25.9
(6,100-word target + Batch B tone surgery to follow).  During the port two
self-inflicted failures occurred and were fixed: an empty-span
str.replace("", new) blowup (a bare-string argument where a tuple was
expected) and a wrapper search that missed \scriptsize-converted tables.


## K. L-axis completion and the full information budget (2026-08-09)

The 432-cell information ladder (protocol v2 runner) completed: 432/432
cells, two-sided registry assertions clean, 12,744 station-cells,
reportable flags applied.

| 7-day effect | delta RMSE (station median) |
|---|---|
| Forcing value (F3 - F0) | -0.62 C (1.735 -> 1.113) |
| Local thermal history (L2 - L0) | +0.99..+1.47 C |
| Local statistics (L1 - L0) | +0.16..+0.21 C |
| Geometry at L0 | +0.005..+0.02 C |
| Geometry at L2 | +0.48 C (~30x amplification) |
| Architecture (plain TCN vs full) | <= 0.023 C |

Advisory D2 quantitatively confirmed: the spatial null was constructed
(geometry is second-order ONLY given local thermal history; without it the
geometry effect amplifies ~30-fold).  The paper's central claim is now the
positive information-budget statement (advisory S2).

Batch A status: applied to md (99,468 -> 78,432 bytes) and tex (50 -> 41
pages, 6-table inventory, 0 overfull warnings); gate + 855 tests green.
Batch B (tone surgery) and Phase 6 (full rewrite with the information-budget
framing) are the next pass.


## L. Former v4 correction to Sections J--K, withdrawn by DLOG-025 (2026-08-09)

Sections J and K above are retained as a chronological record, but their
forcing and information-budget interpretations are **superseded** by
DLOG-018--DLOG-025.  They must not be cited as current evidence.  In
particular, the earlier `0.622 C`, `36%`, `27:1`, `+0.48 C`, `~30x`, "full
information budget", and "P-5 CONFIRMED" statements were produced before the
estimands, target registry, chronology and authority bindings were corrected.

**DLOG-025 now also withdraws every learned F/L number in the remainder of
this section.**  The v4 files and hashes are retained only as forensic records
of reproducible downstream arithmetic over invalid learned predictions.  They
are not current result authorities, and no point estimate, interval, sign-flip
tail, equal-HUC summary, LOCO range, P-5 component, or lead-order statement
below may be cited.  No corrected replacement result has yet been issued.

The now-withdrawn post-outcome-normalization forcing artifact contains exactly
12 F0/F3-full tree-model cells over the completed 116-station cohort.  Its
intended station-level forcing estimand was
`median_i[RMSE_i(F0) - RMSE_i(F3_full)]`, not a difference of aggregate
medians.  The formerly reported, now-withdrawn values were:

| Model | 1 d | 3 d | 7 d | F3-full 7 d RMSE |
|---|---:|---:|---:|---:|
| LightGBM | +0.119 C | +0.493 C | +0.541 C | 1.114 C |
| Residual LightGBM | +0.116 C | +0.514 C | +0.535 C | 1.121 C |

Here F3-full used realized future Daymet/gridMET gridded estimates at station
coordinates.  It is a retrospective oracle/proxy, not direct station forcing,
not a catchment-average product, and not an operational or deployable
forecast.  The corresponding withdrawn seven-day learned-minus-damped
effects were -0.608 C and -0.582 C.  The registered P-5 absolute-RMSE interval
[1.15, 1.25] C fails for both tree models, while its learned-model threshold
and expansion conditions appeared to pass on these invalid predictions; no
P-5 or P-6 component or verdict survives DLOG-025.

The formerly reported, now-withdrawn seven-day L-axis geometry estimates on
the completed 432-shard L0--L2 subset were:

| Estimand | LightGBM | Residual LightGBM |
|---|---:|---:|
| Geometry at L0, `G@L0` | +0.007968 C | +0.003983 C |
| Geometry at L2, `G@L2` | +0.100697 C | +0.103896 C |
| Interaction, `LxG = G@L2 - G@L0` | +0.087490 C | +0.100936 C |

These were not the earlier `+0.48 C` or `~30x` quantities, which remain
withdrawn independently.  The withdrawn whole-HUC2 arithmetic was
model-dependent: the LightGBM LxG interval included zero and its exact raw
sign-flip p-value was 0.400146; the residual-LightGBM interval was
[+0.006921, +0.301810] C with raw p = 0.041016.  Equal-HUC summaries also
differed materially from station-weighted summaries.  These values now carry
no scientific interpretation because their prediction lineage is invalid.

Both forensic authorities were labelled `POST_OUTCOME_NORMALIZATION_ONLY`.
The forcing artifact was additionally scoped `COMPLETED_VIEWED_DOMAIN_SUBSET`;
the L-axis inference was scoped `COMPLETED_SUBSET_NOT_FULL_V2` and
`APPROXIMATE_FIXED_COHORT_DESCRIPTIVE_SENSITIVITY`.  Their preserved manifests
are identified as follows; DLOG-025 withdraws each from scientific use:

| Forensic artifact | Manifest SHA-256 | Protocol SHA-256 | Current status |
|---|---|---|---|
| Forcing regime v4 | `260baa7e9fae08afb64280cb87fe320fc1712f4d2e0bd8b261ef824b133285b8` | `66e089baf37db1137cad23f148e31df39cc71aea872dec4a97dc6f8701d13a98` | withdrawn by DLOG-025 |
| Forcing-regime inference v4 | `580fed64c308267c1ed0c49e60dac8f21ff1657651de4ee70e3b179d6ed758d7` | inherited exact binding `66e089baf37db1137cad23f148e31df39cc71aea872dec4a97dc6f8701d13a98` | withdrawn by DLOG-025 |
| Information-regime inference v4 | `1bce4a92c0b90193a6c64edf26eaaa43f323e5335374c12dbf78cb82d3fa515c` | `66e089baf37db1137cad23f148e31df39cc71aea872dec4a97dc6f8701d13a98` | withdrawn by DLOG-025 |

The withdrawn forcing-inference artifact recorded whole-HUC2 95% bootstrap
intervals of
[+0.075163, +0.169223], [+0.360625, +0.651083], and
[+0.406486, +0.702479] C for LightGBM at 1/3/7 days, and
[+0.074042, +0.164942], [+0.355890, +0.673587], and
[+0.391430, +0.695267] C for Residual LightGBM.  All 90 leave-one-HUC2
values were positive.  These intervals, leave-one-HUC2 values, and exact
sign-flip tails are retained only to document what was withdrawn.

Protocol v4 remains draft and unsealed.  A versioned observed-lineage rerun and
new independent authority are required before any learned F/L result can be
reported; F2, L3, U2, neural, crossed-design, event and prospective
confirmation work also remain incomplete.

## M. F2a input acquisition gate (2026-08-09)

The retrospective Open-Meteo Previous Runs/GFS acquisition has now completed
and passed its independent, create-only full verification.  The inventory is
4,080 station-month chunks, 120 stations and 362,520 rows over 2021-03-30
through 2023-12-31; 361,800 rows satisfy the complete fixed-lead record
contract.  Coverage is 120,600/120,840 station-days (`0.9980139`) at each of
the 1-, 3- and 7-day leads, and all 120 stations pass the frozen 0.90 gate.

This closes an input-integrity gate only.  The verifier records no
water-temperature/flow outcome access, `arm_promoted: false`, and a mandatory
separate protocol authorization before use.  The product remains a
retrospectively retrieved fixed-lead air-temperature composite, not a coherent
initialization or as-issued operational forecast.  No F2a model score or
recovery fraction has been computed.  A future recovery analysis must use an
exact F0/F2a/F3_temperature_only common registry and must never use F3_full as
the denominator.  DLOG-024 subsequently freezes that score-independent
registry but does not authorize the analysis.

The verification-report, acquisition-manifest and snapshot-index SHA-256
values are respectively
`7d8a019d4966a941e9396a40930b2775ba95660347684701838721fe5032a60e`,
`54f85eef3ccd4b3cc69071f3c4ed7af5e3208274d3b9062a49c43c96ea43362c`,
and `433f4b4885f225f5f9fe87ec9e94ba81493c07691bb4cc9a994be1931a406771`.

## N. Score-independent v4 key registries (2026-08-09)

DLOG-024 freezes a create-only Phase-1 key-registry authority at
`outputs/final/information_regime_key_registries_v4/`.  The manifest status is
`PHASE1_KEY_REGISTRIES_ONLY_NOT_MODEL_SCORE_AUTHORITY`.  The builder read no
model score and the authority publishes no model output.  It therefore is not
an F2a arm promotion, a forecast-value or history-quality result, an execution
authorization, or prospective preregistration.  The v4 protocol remains
`DRAFT_NOT_SEALED` with `execution_authorized: false`, and the F2a manifest
field remains `arm_promoted: false`.

The primary base registry contains 358,807 raw keys from 118 sites and 358,765
reportable keys from 116 sites.  Raw counts at 1/3/7 days are
120,466/119,654/118,687; reportable counts are
120,444/119,639/118,682.  This is the F0/F3_full base registry, not an F2b
intersection.  The separate F2a-temperature registry intersects the acquired
fixed-lead composite only with the target-day `F3_temperature_only` field: it
contains 329,648 common keys from 117 sites and 329,628 reportable keys from
116 sites.  Its 1/3/7-day common counts are 110,397/109,857/109,394 and its
reportable counts are 110,385/109,850/109,393.  `F3_full` remains a forbidden
recovery denominator.

The same authority repairs the issue-date history context without changing
the legacy authority-bound `outputs/final/forecast_keys.parquet`.  It uses only
finite water-temperature observations on or before each issue date.  The
raw Hall/H75/H100 counts are 358,807/352,245/315,336 and the reportable counts
are 358,765/352,223/315,314.  These membership counts freeze future
history-quality strata; no stratified model contrast has been run.

The formal manifest SHA-256 is
`ac0c256907264022e1fe7c4e407e0f95ece1f03233ecd6bb1841e27eb40b49ea`,
and the bound builder SHA-256 is
`ae22113b0011d4d0dec902f65f98df0ec45a47cd0842c07cdf0b83c9c0584b94`.
Byte determinism is claimed only for the pinned production runtime: CPython
3.11.7, NumPy 1.26.4, pandas 2.1.4, PyArrow/Arrow 14.0.2, GCC 11.2.0,
x86_64 little-endian Linux.  The manifest explicitly discloses that the two
repository locks target a different Python/pandas/PyArrow environment and
does not claim cross-environment Parquet byte identity.

## O. Observedness-lineage defect and F/L withdrawal (2026-08-09)

DLOG-025 records a pre-fit defect in both legacy final runners.  The exact
development and evaluation Parquets contain no `*_observed` columns.  The
runners did not create those flags before imputation, then supplied the
imputed panel as both feature input and purported true-label input.  As a
result, imputed water-temperature targets entered fitting, missing issue-date
water temperatures passed admissibility, and history-missingness features
were inferred from filled rather than raw values.  The forcing runner also
selected its final 2,000 training-table rows as the LightGBM evaluation set
instead of using a disjoint temporal validation partition.

For the twelve forcing tree cells, current versus correctly admissible
combined 2006--2017 row counts are 525,720/423,266 at one day,
525,240/420,673 at three days, and 524,280/418,055 at seven days.  The invalid
unions are 102,454/104,567/106,225 rows, including
100,174/100,041/99,769 imputed targets.  Even among otherwise admissible rows,
57,441/56,619/56,141 rows have erroneous missingness masks.  On the formal
358,765-key primary reportable registry, 51,723 evaluation keys have at least
one erroneous history-mask feature, so no learned prediction in the old F/L
trees is salvageable.

The withdrawal covers all 432 `ladder_shards`, their effects/summary and both
information-regime v4 result/inference authorities, plus all 12
`forcing_shards_v4`, their effects/contrasts/summary and both forcing-regime v4
result/inference authorities.  The files remain immutable forensic records.
It does not withdraw raw evaluation key identity or `y_true`, the
score-independent DLOG-024 key-registry authority, the label-free DLOG-023 F2a
acquisition verification, the conventional Route-A authority, or the separate
air2stream raw path.  Corrected work must use
`primary_reportable_key_registry_v4.parquet`, preserve raw flags before any
imputation, bind raw training labels, and separate 2006--2015 training from
2016--2017 validation under new versioned runners and output paths.

DLOG-026 subsequently freezes the adverse evidence in the create-only
`outputs/final/preprocessing_lineage_defect_authority_v1/` authority.  Its
report independently reconstructs 17,478/17,231/17,014 affected formal keys
at 1/3/7 days (51,723 total), while retaining the 51,765-key legacy count only
as forensic chronology.  It also binds the old training-tail evaluation-set
defect and the unchanged preservation boundary.  The report and manifest
SHA-256 values are respectively
`5c1b1e05932538dba1b2a0d21ad44fdfe54a9c52e949b96b3a668356910e68d9`
and
`e69124409f49e4fb2aaaae319104251ca3078e535fca69121ed0eafddb23d908`.
The builder read zero prediction or score rows.  This freezes a withdrawal,
not a corrected result, execution authorization, or protocol seal.

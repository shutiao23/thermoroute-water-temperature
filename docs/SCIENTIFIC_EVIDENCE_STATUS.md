# Scientific evidence status

Snapshot date: 2026-08-09 UTC
Scope: WRR information-regime redesign
Governing rule: a result may enter the manuscript as a quantitative finding
only when its row is `VERIFIED_AND_FROZEN`.

This register separates a computed result from a result that is currently
reproducible and bound to the paper.  It is intentionally conservative.  The
only permitted status values are:

- `VERIFIED_AND_FROZEN`: the complete required artifact exists, its producing
  contract and estimator are explicit, and a current integrity check binds the
  artifact to the repository state.
- `PROVISIONAL_NOT_YET_COMPLETE`: some evidence exists, but at least one
  required arm, estimator, integrity binding, or independent check is missing
  or contradictory.
- `PLANNED`: no admissible current result exists for the stated deliverable.

## Current evidence register

| Result or deliverable | Status | Current authoritative evidence | Requirement before promotion |
|---|---|---|---|
| V1 common-key point-model authority (2021–2023) | `VERIFIED_AND_FROZEN` | `outputs/final/{forecast_keys,station_metrics,paired_effects}.parquet`; their current SHA-256 values match `outputs/final/result_manifest.json`. This status covers the key/target and point-model quantities, not the defective legacy `issue_wtemp_observed` or `days_since_last_observed_wtemp` metadata columns | Preserve the matched hashes when the manifest is regenerated; use the separately named, now-frozen v4 context registry for any future recency/issue-observed analysis and never overwrite the authority-bound v1 file |
| Matched plain-TCN architecture contrast under F0–L0 | `VERIFIED_AND_FROZEN` | `outputs/final/station_metrics.parquet` and `outputs/final/paired_effects.parquet`; current v1 claim ledger and consistency gate | Keep the conclusion explicitly conditional on F0–L0 until hard-regime cells exist |
| Protocol v2 information-ladder freeze | `PROVISIONAL_NOT_YET_COMPLETE` | `protocols/wrr_strong_accept_protocol_v2.yaml` and `_seal.json` exist | Seal must bind the protocol bytes themselves; resolve the mismatch between the required 576 + U2 design and the reported 432-cell subset |
| Protocol v3 forcing-ladder freeze | `PROVISIONAL_NOT_YET_COMPLETE` | `protocols/wrr_strong_accept_protocol_v3.yaml` and `_seal.json` exist; DLOG-015 has now been administratively reconstructed from those existing objects after outcome access | Seal must bind the protocol bytes themselves; the reconstructed DLOG-015 must remain explicitly post-outcome and cannot substitute for a contemporaneous pre-registration record; bind complete forcing artifacts |
| Information-regimes protocol v4 draft | `PROVISIONAL_NOT_YET_COMPLETE` | `protocols/wrr_information_regimes_protocol_v4.yaml` explicitly declares `DRAFT_NOT_SEALED`, `execution_authorized: false`, and post-outcome normalization for the already-viewed 2021–2023 domains. DLOG-024 freezes score-independent keys; DLOG-025 withdraws the current F/L tree results; DLOG-026 freezes only the adverse-lineage evidence. None changes the protocol flags | Correct and independently authorize the observed-lineage runners, bind the frozen key authority, resolve every remaining pointer and model/fold/contrast registry, and only then create a new exact-byte seal; never describe the draft or its normalization analyses as prospective preregistration |
| Score-independent v4 key-registry authority | `VERIFIED_AND_FROZEN` | The create-only `outputs/final/information_regime_key_registries_v4/` authority has manifest status `PHASE1_KEY_REGISTRIES_ONLY_NOT_MODEL_SCORE_AUTHORITY`. It freezes 358,807 primary raw keys/118 sites, 358,765 primary reportable keys/116 sites, 329,648 F2a-temperature common keys/117 sites, and 329,628 F2a-temperature reportable keys/116 sites. Its manifest SHA-256 is `ac0c256907264022e1fe7c4e407e0f95ece1f03233ecd6bb1841e27eb40b49ea`; no model score was read or accepted | Preserve the create-only bytes and exact-runtime disclosure. Use the primary registry only as the F0/F3_full base, not an F2b intersection; use the F2a registry only with `F3_temperature_only`. This artifact does not promote F2a, authorize execution, seal v4, or constitute a model-result authority |
| Semantic registries v4 authority | `VERIFIED_AND_FROZEN` | DLOG-028 records the create-only 11-file `outputs/final/semantic_registries_v4_authority/`. Manifest SHA-256 `12cdc355a06d2c39733a60386dfdeb8a2f6b234d2f8d6f641a995a3d3c41072c` binds the two independently rebuilt data registries, six contract registries, both candidate manifests, builders/tests/source inputs and the pinned Route-A runtime. The 72 primary logical-cell states are exactly 48 `PLANNED`, 24 `REGISTERED`, zero `EXECUTED` and zero `WITHDRAWN`; the separate 432 legacy cells remain forensic `WITHDRAWN` | Preserve the exact bytes and code bindings. This score-free authority has `execution_authorized: false`, binds no forcing seal and cannot promote any arm, model result or protocol state; a separate forcing-specific terminal seal is still required before scoring |
| Preprocessing-lineage defect authority v1 | `VERIFIED_AND_FROZEN` | The create-only `outputs/final/preprocessing_lineage_defect_authority_v1/` contains exactly two JSON files. Manifest SHA-256 `e69124409f49e4fb2aaaae319104251ca3078e535fca69121ed0eafddb23d908` binds a report SHA-256 `5c1b1e05932538dba1b2a0d21ad44fdfe54a9c52e949b96b3a668356910e68d9`, the formal 358,765-key domain, 51,723 erroneous-mask keys, the separate 358,807-key legacy inventory, and zero prediction/score rows read | Preserve the exact bytes. Its frozen `SCIENTIFIC_EVIDENCE_STATUS_ROWS_31_32_33_36_38` token is a historical symbolic label, not a live Markdown line pointer. This authority freezes a withdrawal, not a replacement result or execution permission; corrected F/L runs still require new runners, governance, outputs and independent result authorities |
| DLOG-012–028 scientific chronology | `PROVISIONAL_NOT_YET_COMPLETE` | DLOG-012–014 and DLOG-016–028 record the observed decisions and corrections; DLOG-015 is explicitly labelled an after-outcome administrative reconstruction; DLOG-023 binds only label-free F2a acquisition, DLOG-024 only score-independent keys/context, DLOG-025 withdraws DLOG-018–022 results, DLOG-026 freezes the adverse-lineage evidence, DLOG-027 closes F2b as currently `PLANNED_DEGRADED`, and DLOG-028 freezes only score-free semantic registries | Commit the chronology and referenced artifacts together. Never cite DLOG-020–022 as current evidence, reconstructed DLOG-015 as prospective registration, DLOG-023 as an F2a result, DLOG-024 as arm promotion/model evidence, DLOG-026 as a corrected model result, DLOG-027 as proof that an F2b archive/result exists, or DLOG-028 as execution permission/model evidence |
| L0/L1/L2 × random/region × two tree models completed subset | `PROVISIONAL_NOT_YET_COMPLETE` | The 432 shards and both formal authorities remain byte-reproducible forensic records, but DLOG-025 withdraws them from scientific use: raw observedness flags were lost before imputation, so imputed targets/issues entered training and history missingness features were corrupted | Rerun the exact viewed scope under a new observed-lineage version with pre-imputation flags, raw training labels, disjoint 2006–2015 train/2016–2017 validation, corrected common keys, and a new independent authority; never overwrite the old bytes |
| Seven-day L2 geometry contrast | `PROVISIONAL_NOT_YET_COMPLETE` | The previously reported +0.100697/+0.103896 °C estimates and their clustered sensitivities are withdrawn by DLOG-025 because the underlying predictions inherit invalid training and feature-mask lineage | Recompute on corrected versioned shards before reporting any L2 geometry value. The older incompatible +0.48 °C / 30× interpretation remains withdrawn and is not restored |
| Seven-day L×G station-level interaction | `PROVISIONAL_NOT_YET_COMPLETE` | The previously reported +0.087489705415824/+0.100936172151032 °C double differences and their clustered sensitivities are withdrawn by DLOG-025; the downstream arithmetic was reproducible but consumed invalid predictions | Recompute the station-level double difference and whole-HUC2 sensitivities only from a new observed-lineage authority; no old point estimate or interval may enter the claim ledger or manuscript |
| L2-U2 no-local-temperature/no-local-flow sensitivity | `PLANNED` | Runner has a code path, but no `L2U2` shard or governed summary exists | Complete the predeclared cells, prove that target identity/adaptation paths are absent, and bind outputs |
| L3 attribute-informed cold-start arm | `PLANNED` | `src/thermoroute/regionalize.py` exists; required `outputs/final/basin_attributes_v2.parquet` does not | Freeze hydrologically motivated attributes, document source coverage, run the arm, and apply the <70% degradation rule |
| F0 tree forcing baseline | `PROVISIONAL_NOT_YET_COMPLETE` | The twelve v4 shards and authorities remain forensic records, but DLOG-025 withdraws their learned-model scores and contrasts: roughly 100,000 imputed targets per lead entered fitting and 51,723 corrected evaluation keys have at least one erroneous history-mask feature | Rerun only the already-viewed F0/F3_full tree scope under a new observed-lineage version, corrected 358,765-key registry, disjoint temporal validation, and new create-only authority; preserve but never overwrite the v4 bytes |
| F1 climatological-forcing diagnostic | `PLANNED` | No admissible current result; the prior run was discarded | Fit a separate training-only climatology for each meteorological variable, test it, and rerun on common keys |
| F3 retrospective realized gridded meteorological oracle | `PROVISIONAL_NOT_YET_COMPLETE` | Raw future F3 values/masks and exact evaluation keys/targets remain usable inputs, but every learned F0/F3 score, forcing value, HUC2 interval, LOCO summary, lead-order interpretation, and P-5 component from the v4 tree shards is withdrawn by DLOG-025. The old authorities prove only downstream arithmetic consistency | Rerun the viewed twelve-cell correction with preserved raw masks and labels, then issue new point/inference authorities before quoting any forcing number. F3 must still be called a retrospective gridded oracle, never an operational gain |
| F3 retrospective realized gridded meteorological oracle shuffled and ±7-day time-shift placebos | `PLANNED` | No implementation or result artifact | Freeze shuffle strata/seeds and shift semantics, then run on the identical key registry |
| Forcing-component attribution | `PLANNED` | No implementation or result artifact | Run predeclared air-temperature, radiation, precipitation, humidity/wind, and full-forcing arms; keep discharge separate |
| F2a retrospective fixed-lead air-temperature forecast composite | `PROVISIONAL_NOT_YET_COMPLETE` | The create-only `outputs/final/f2a_acquisition_verification_v1.json` independently verifies all 4,080 immutable station-month chunks, fixed-lead semantics, no outcome-label access, and 0.998014 coverage at each lead. The separate score-independent v4 authority now freezes 329,648 F2a/F3-temperature common keys and 329,628 reportable keys, but records `arm_promoted: false`, `model_scores_read_or_accepted: false`, and no model output | Bind the already-frozen common-key registry plus exact feature/runner code in a new v4 execution seal, then compute only the `F3_temperature_only`-matched diagnostic. Never call the product coherent/as-issued/operational, treat the key registry as arm promotion, or use `F3_full` as its recovery denominator |
| F2b archived-vintage coherent multi-variable forecast trajectory | `PLANNED` | DLOG-027 records the current NO-GO and `PLANNED_DEGRADED` decision. NOAA NODD samples expose coherent 3-hour f000--f192 GFS trajectories for old initializations, but the repository has no frozen issue-time cutoff, full 2021--2023 inventory, publication/revision-history attestation for backfilled objects, common-key coverage authority, or archived GRIB bytes and SHA-256 values | Keep F2b outside the primary result matrix and declare the 48-cell F0/F3 degradation. Reopen only under a separately frozen acquisition pilot that binds one available initialization per issue date, publication semantics, complete trajectories, required variables, original bytes/checksums, and the >=0.90 common-key coverage gate; never substitute F2a or claim as-issued/operational replay |
| Primary F×L×G×A crossed response surface | `PLANNED` | Existing F and L results come from separate conditional designs | Complete the frozen common-key matrix for LightGBM and plain TCN and emit station-paired main effects/interactions |
| Information-regime claim-ledger closure | `PLANNED` | `paper/claim_ledger.yaml`, `outputs/final/claim_ledger_resolved.csv`, and `outputs/final/paper_values.tex` contain the earlier v1 claims but no governed entries/macros for the F3 retrospective realized gridded meteorological oracle, L1/L2, the L×G interaction, or operational recovery | Add each promoted information-regime number with its exact source table, filters, paired estimator, precision, and every `used_in` span; regenerate the resolved ledger and macros and require the gate to fail on any undeclared information-regime headline |
| Unified information-regime result manifest | `PROVISIONAL_NOT_YET_COMPLETE` | The current top-level `result_manifest.json` binds protocol v1 only, omits forcing artifacts, and records a stale digest for `ladder_effects.parquet`; the new `information_regime_v4_manifest.json` validly binds the completed L0–L2 subset but is not a unified F/L manuscript authority | Generate one clean-state manifest that binds the governing protocol/amendments, source commit, claim ledger, summaries, key-level and station-level F/L artifacts, and exact output hashes; verify every digest before manuscript use |
| ThermoRoute hard-regime cells, bounded/unbounded | `PLANNED` | Only the F0–L0 architecture audit is frozen | Run the selected F2, F3 retrospective realized gridded meteorological oracle, L2, and region cells with matched inputs, objective, seeds, and bounded/unbounded variants |
| Validated physical/forced-hybrid reference | `PLANNED` | A transparent constrained equilibrium-response implementation and score-free fail-closed calibration runner now exist, with train-only calibration, unit/sign/range contracts, four forcing schemas and recursive 1/3/7-day rollout tests. No governed parameter authority, 2021–2023 prediction or score exists | Bind the exact implementation, inputs and reviewed seal; publish create-only training parameters; independently validate reference cases before using process-comparator language or computing any holdout score |
| Local-history quality sensitivity (H100/H75/Hall) | `PROVISIONAL_NOT_YET_COMPLETE` | The separately named v4 key authority reconstructs issue-date context from the raw holdout panel using only finite WTEMP on or before each issue date. Raw Hall/H75/H100 counts are 358,807/352,245/315,336; reportable counts are 358,765/352,223/315,314. The defective legacy context columns were not consumed and `outputs/final/forecast_keys.parquet` was not overwritten. No stratified model contrast exists | Preserve the score-independent registry and compute paired L0−L2 effects within its frozen H100/H75/Hall strata only after an exact-byte v4 execution seal authorizes the runner; do not infer a history-quality result from registry membership counts |
| Ten-seed matched random-partition distribution | `PLANNED` | L ladder currently has five random seeds | Complete 10 seeds × 4 folds with matched station/key counts and locate the whole-region result in that distribution |
| Geographic/hydroclimatic novelty analysis | `PLANNED` | Coordinates and limited attributes exist; no governed v4 novelty artifact exists | Compute training-only geographic and environmental distances and relate them to station-paired transfer penalties |
| Warm-tail, signed rapid-change, and exceedance value of F0, F2, and the F3 retrospective realized gridded meteorological oracle | `PLANNED` | General state/event utilities exist, but no forcing-arm event comparison exists | Freeze training-only thresholds and report RMSE/MAE/bias plus POD/FAR/CSI/Brier/onset metrics on common keys |
| Larger no-flow temperature cohort | `PLANNED` | Current analysis is the 120-site WTEMP+FLOW availability-selected cohort | Build the WTEMP+meteorology cohort and reproduce the frozen headline cells, or retain an explicit fixed-cohort scope limitation |
| Frozen 2024–2025 audit | `PLANNED` | DLOG-007 explicitly deferred it | Freeze cohort, matrix, models, thresholds, estimands, and figures before opening outcomes, then run once |
| Information-regime manuscript rewrite | `PLANNED` | `paper/ThermoRoute_paper.md` still has the old benchmark-design title and section order | Rewrite only from promoted evidence; keep oracle, retrospective forecast, gauged transfer, and cold-start tasks distinct |

## Remaining contradictions that block manuscript closure

1. The current 432 L shards and 12 forcing shards lost raw observedness before
   imputation.  Their learned scores and every downstream F/L contrast or
   sensitivity are withdrawn by DLOG-025.  Exact evaluation keys and targets
   remain valid, but no old learned prediction can be salvaged.  DLOG-026 now
   freezes the adverse evidence, not a correction; a versioned observed-lineage
   rerun and new result authority are still required.
2. The current top-level `result_manifest.json` binds protocol v1, omits both
   information-regime chains, records a stale digest for
   `ladder_effects.parquet`, and does not record DLOG-025's withdrawal.  A
   unified manuscript authority has not yet been generated.
3. The claim ledger, resolved ledger, TeX macros, tables, figures, and
   manuscript consistency gate contain no governed F/L claims.  The artifact
   values are now withdrawn rather than merely unbound; no old or hand-entered
   information-regime number is allowed in the paper build.
4. The v4 document remains `DRAFT_NOT_SEALED` with
   `execution_authorized: false`.  DLOG-024 freezes Phase-1 keys but supplies no
   execution seal or model-score authority.  No canonical `_sealed.yaml` or
   seal file should exist until the observed-lineage correction, semantic
   registries, loaders, trainers, scorers, writers, environment, and clean
   design commit are complete.
5. The F and L authorities are separate conditional experiments.  They cannot
   be added, divided, or presented as a closed information budget even after
   correction.  The primary F×L×G×A response surface remains planned.
6. Protocol v2 describes 576 ladder cells plus U2.  L3, L2-U2, and N07–N09
   remain absent; the N01–N15 Holm family is not computed.  The old 432-cell
   subset is both incomplete and scientifically withdrawn.
7. DLOG-015 is an after-outcome administrative reconstruction, and the old
   v2/v3 seal JSON files do not bind their claimed YAML protocol bytes.  Neither
   object supplies prospective status to the 2021–2023 normalizations.

## Gates closed or withdrawn through DLOG-028

- DLOG-020–022 corrected key pairing and downstream estimand arithmetic but
  did not validate the upstream training lineage.  DLOG-025 therefore
  withdraws every numerical F/L result and P-5/P-6 interpretation from those
  decisions.  The historical 0.622 °C, +0.48 °C, 30×, 27:1, and full-budget
  claims remain withdrawn as well; no replacement value is currently
  admissible.
- DLOG-023 closes only the F2a acquisition-integrity and 90% coverage gate:
  120/120 stations pass at every fixed lead and the verifier confirms a
  label-free access boundary.  It does not close the model, recovery,
  execution-seal, or result-authority gates.
- DLOG-024 closes the score-independent primary, F2a-temperature and corrected
  as-of history-context registry gates.  It does not promote F2a, read or
  publish model scores, seal the draft v4 protocol, authorize execution, or
  supply any model or history-quality result.
- DLOG-025 closes the disclosure/withdrawal gate for lost observedness lineage.
  It preserves the invalid bytes for forensics, preserves the independent
  score-free input authorities, and requires new versioned outputs rather than
  an in-place rewrite.
- DLOG-026 closes the create-only adverse-lineage evidence-freeze gate.  Its
  authority binds the formal 358,765-key domain and 51,723 affected keys while
  reading zero prediction or score rows.  It neither authorizes execution nor
  restores or replaces any learned F/L result.
- DLOG-027 applies the F2b degradation rule after a label-free official-source
  archive probe. It closes the current 72-cell completion claim and retains a
  48-cell F0/F3 scope; it does not freeze F2b data, features or model evidence.
- DLOG-028 freezes the score-free semantic data and contract registries in an
  11-file create-only authority. Its 48 `PLANNED` plus 24 `REGISTERED` primary
  cells contain zero executed cells; it neither seals a forcing protocol nor
  authorizes a score-producing process.

Until corrected outputs are independently frozen, the manuscript must report
no learned F/L number at all.  Any future replacement must remain explicitly
post-outcome, fixed-cohort, conditional, and non-additive.

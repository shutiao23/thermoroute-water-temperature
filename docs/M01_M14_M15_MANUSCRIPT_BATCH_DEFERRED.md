# M-01 / M-14 / M-15 manuscript wording change batch — governance-closure carry-over

| Field | Value |
| --- | --- |
| Date | 2026-08-02 |
| Status | `DEFERRED_TO_P0_A0_GOVERNANCE_CLOSURE` — diffs prepared, not applied |
| Review items | M-01 (physics-inspired → constrained statistical), M-14 (variable dictionary), M-15 (literature coverage) |
| Why deferred | PRE-phase manuscript bytes are triple-bound in the fail-closed claim chain: `protocols/route_a_claim_registry_v1.json` `preopen_document_sha256` → `protocols/route_a_model_matrix_amendment_v1.json` `governance_inputs` → `scripts/26_validate_claims.py` `MODEL_MATRIX_CLAIM_REGISTRY_SHA256`. Editing the manuscript bytes breaks `test_every_required_preopen_document_matches_its_frozen_sha256` and the model-matrix validator unless the whole chain is updated in one governed batch. |

## 1. The frozen chain (verified 2026-08-02)

```text
paper/ThermoRoute_paper.md  ─┐
paper/agu_submission/ThermoRoute_WRR.tex ─┤
README.md, highlights, cover_letter …    ─┼─> preopen_document_sha256
                                          │     in protocols/route_a_claim_registry_v1.json
                                          ▼
protocols/route_a_claim_registry_v1.json  ──> sha256 referenced by
                                          │     route_a_model_matrix_amendment_v1.json
                                          │     governance_inputs.route_a_claim_registry_v1
                                          ▼
scripts/26_validate_claims.py  MODEL_MATRIX_CLAIM_REGISTRY_SHA256 constant
                                          │
                                          ▼
source_tree_hash  (scripts/** in DEFAULT_SOURCE_PATTERNS)
```

Current registry file SHA-256 = `bc3d489b6f2cfe945789a57990a15291b332ae63be6c8df0211aa2b6ff8b70b6`,
which matches `MODEL_MATRIX_CLAIM_REGISTRY_SHA256` exactly. Any manuscript edit
requires updating: the two `preopen_document_sha256` entries, the registry file
hash constant in `scripts/26_validate_claims.py`, and the amendment's
`governance_inputs` binding — all inside one authorized source-boundary change.

## 2. Prepared diffs (apply only inside the governed batch)

### M-01 — remove physics-inspired mechanism wording

Files: `paper/ThermoRoute_paper.md`, `paper/agu_submission/ThermoRoute_WRR.tex`.

1. Title: `a bounded-deviation, physics-inspired framework` →
   `a bounded-deviation, constrained statistical framework` (md line 1; TeX `\title`).
2. Abstract first sentence: `physics-inspired statistical predictor` →
   `constrained statistical predictor`; append after "causal drivers":
   `; the relaxation dynamics are learned statistical surrogates, not solved
   heat-transfer or energy-balance equations, and no thermal-energy closure is claimed`.
3. Keywords: `physics-inspired machine learning` →
   `constrained statistical machine learning`.

### M-14 — variable dictionary

Files: `paper/ThermoRoute_paper.md` (§2.1, after the variable sentence),
`paper/si/SI16_rights_data_dictionary.md`.

Add Table 1 with rows WTEMP (°C, NWIS 00010 daily mean), FLOW (cfs, 00060,
negative values stratified), WLEVEL (ft, 00065, raw evidence only), TEMP (°C,
Daymet daily mean), PRCP (mm day⁻¹), RHMEAN (%, derived proxy, not measured),
DH (W m⁻², daylight-period mean, not daily energy), WDSP (m s⁻¹, gridMET packed,
scale 0.1). Add the LOD/LOQ N/A statement and source-precision rounding note.
SI16 already contains the full QA/QC dictionary text prepared in this revision.

### M-15 — literature coverage

Files: `paper/ThermoRoute_paper.md` (§1.1), `paper/agu_submission/ThermoRoute_WRR.tex`
(§1.1), `paper/references.bib`.

Add: Corona & Hogue 2025 HESS review (57 studies); China-basin LSTM/hybrid work
(Qiu et al. 2021 J. Hydrol. 595:126016, DOI 10.1016/j.jhydrol.2021.126016; Tao et
al. 2021 J. Hydrol. 598:126430, DOI 10.1016/j.jhydrol.2021.126430; Huang et al.
2023 J. Hydrol. 616:128857, DOI 10.1016/j.jhydrol.2022.128857); Luo et al. 2025
SIGSPATIAL geo-aware models (DOI 10.1145/3748636.3762716); Rahmani et al. 2021
PUB unmonitored basins (pubs.usgs.gov 70238318); a cross-study comparability
disclaimer. All three new bib entries were verified against CrossRef on
2026-08-02 (author lists, titles, volumes, DOIs).

## 3. When this batch may be applied

Inside P0-A0 governance closure (same authorized source-boundary change that
seals the final protocol), together with:

1. registry `preopen_document_sha256` update for both files;
2. `MODEL_MATRIX_CLAIM_REGISTRY_SHA256` constant update in
   `scripts/26_validate_claims.py`;
3. amendment `governance_inputs` binding update + re-seal;
4. recompute `source_tree_hash` and propagate through pins/run package.

Until then the manuscript keeps the pre-opening frozen wording, which is
internally consistent and passes the full suite (claim registry tests included).

## 4. Non-goals

This record does not authorize opening, does not change any scientific value in
the frozen protocol, and does not weaken the fail-closed chain. It only preserves
review-driven wording changes so they are not lost while the governed batch is
scheduled.

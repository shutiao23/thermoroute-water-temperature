# SI16 — rights matrix and data dictionary

**Status:** scaffold finalized; not a rights approval.

This file is a projection schema, not a redistribution decision, DOI record or
FAIR completion claim.

## Rights projection

| Byte class/object | SHA-256 | provider/rightsholder | evidence locator | distribution scope | licence/terms | reviewer decision | binder row ID |
|---|---|---|---|---|---|---|---|
| *(content-deduplicated object)* | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` | `[pending computation]` |

Unknown, conflicting or unreviewed rights remain PUBLIC-excluded. Repository
visibility, provider openness and a software licence are not substitutes for an
exact-byte redistribution decision.

## Redistribution class enumeration

Manuscript §8 states that the classes defaulting to exclusion, and the two
excluded outright, are enumerated here; the AGU data-availability statement does
not need the full list. This section is that enumeration. It is a **status
record, not a rights approval**: every row below is the current recorded state,
and no row may be read as permission.

**Substitute for the bytes.** For any provider object whose redistribution terms
are unresolved at deposit time, the deposit carries, in place of the bytes: the
product identifier and version, the exact request specification (site list,
parameter and statistic codes, and date range), the retrieval code, and a SHA-256
manifest of the bytes as retrieved. A third party can therefore re-acquire
identical inputs from the provider and verify them against the manifest without
relying on redistribution at all.

### Classes with no recorded redistribution decision — default to exclusion

| # | Class | Note |
|---|---|---|
| 1 | USGS NWIS response bytes | the raw request/response record of the test-window acquisition |
| 2 | Daymet V4 subsets, and any derived field that materially encodes Daymet values | a derived field does not escape the upstream terms by being derived |
| 3 | gridMET responses and derived fields | as above |
| 4 | the mixed derived panel and the registries built from them | a derived product does not inherit the most permissive of its upstream terms |

These four default to exclusion until each object proposed for the deposit
carries a recorded, evidence-backed decision. Until then the project's release
tooling refuses to build a public archive, which is why manuscript §8 describes
the deposit as planned and specified rather than as existing, and why the data
DOI and the data licence remain unassigned.

### Classes excluded outright rather than pending

| # | Class | Basis |
|---|---|---|
| 5 | the third-party AGU LaTeX class and its companion assets — `agujournal2025.cls`, `agujournal2019.cls`, `tweaklist-git-moderncv-fixed.sty`, `wiley-macros.tex`, `agu-logo-small.pdf`, `agu-logo-large.pdf` | AGU/Wiley and moderncv bytes supplied for submission use, not archive redistribution. `docs/RIGHTS_PROVIDER_EVIDENCE_20260801.md` §5 records `EXCLUDE_PUBLIC` and `docs/FAIR_RIGHTS_ACCEPTANCE_MATRIX.md` §3 keeps that default for the whole family; no file-level licence or written permission exists. The release verifier rejects them by name, with the rights basis in the error message, so re-adding them is refused rather than silently accepted |
| 6 | three legacy CSV files retained in the repository archive from an earlier three-station case study | no recorded source, collection terms, or redistribution authorization. They are not evidence for any statement in the manuscript; their status is recorded in `protocols/legacy_three_site_semantics_notice_v1.md`. Not redistributed in any form |

### What the software licence does and does not cover

The project's own source code is released under the MIT licence (SPDX
identifier `MIT`). That licence covers the source code only. It does **not**
license the observational data, the archived provider responses, the third-party
typesetting class, or any redistributed dependency binary, each of which retains
its own terms and is enumerated with those terms in an archive notice file. Any
component whose terms are unresolved is excluded from the deposit rather than
shipped under an assumed licence.

### An irreversible gap, recorded rather than repaired

Original provider responses and retrieval timestamps for the 2006–2020
development panel were not retained and cannot be reconstructed. Development
reproduction therefore begins from the committed derived artifact. That
limitation does not extend to the evaluation-period acquisition, for which exact
request and response bytes, qualifiers, timestamps, and content hashes are
archived.

## Data-dictionary projection

The POST verifier must bind this SI to the release inventory, qualified reviewer
decisions, DOI metadata and public-archive receipt. Until then P0-F remains
incomplete and the PUBLIC build must fail closed. Every displayed object hash,
scope, decision and data-dictionary value follows the README cell-level binder
contract.

## Variable dictionary (frozen 120v2 panel and predictor bridge)

Source definitions: `src/thermoroute/usgs.py` (acquisition), `src/thermoroute/predictor_bridge.py`
(bridge contract), `data_usgs/development_predictor_bridge_v1.json`
(`PASS_EXACT_PRODUCT_BRIDGE`). No laboratory chemical assay was performed;
analytical LOD/LOQ are not applicable.

| Variable | Semantic definition | Source product/parameter | Unit | Aggregation/statistic | Missing/qualifier rule | QA notes |
|---|---|---|---|---|---|---|
| WTEMP | Daily-mean river water temperature | USGS NWIS parameter 00010, statistic 00003 (daily mean) | °C | Daily mean of instantaneous values | Broad plausibility gate −2 to 50 °C; qualifier `{A}` (approved) retained as secondary sensitivity; other qualifiers retained in raw evidence | Primary outcome; not instantaneous, daily maximum, or 7DADM |
| FLOW | Daily-mean discharge | USGS NWIS parameter 00060, statistic 00003 | cfs (ft³ s⁻¹) | Daily mean | Negative values possible in tidal/backwater settings; retained as finite values in primary analysis, stratified as sensitivity | 2 sites contain 2,059 negative records (min ≈ −121 cfs); semantics unresolved, reported separately |
| WLEVEL | Daily-mean gage height | USGS NWIS parameter 00065, statistic 00003 | ft | Daily mean | Raw evidence only; not an input to study models | Stored for provenance, not modelled |
| TEMP | Daily-mean air temperature proxy | Daymet single-pixel at station coordinates; tmax/tmin | °C | Daily mean of Daymet tmax/tmin | Daymet 365-day calendar; leap years omit Dec 31, retained as explicit missingness | Point meteorology, not catchment forcing |
| PRCP | Daily precipitation | Daymet `prcp` | mm/day | Daily total | As above | — |
| RHMEAN | Relative-humidity proxy derived from vapour pressure | Derived: 100 · vp / sᵥₚ(tmean), Tetens saturation | % | Daily mean | Clipped to [0, 100] | Not a directly measured daily-mean RH; name is a proxy label |
| DH | Daylight-period mean solar flux | Daymet `srad` (legacy feature name `DH`) | W m⁻² | Daylight-period mean (not daily energy total) | As above | Do not interpret as daily irradiance integral |
| WDSP | Daily wind speed | gridMET packed value, scale factor 0.1, add offset 0.0 | m s⁻¹ | Daily mean after CF packing decode | Values outside [0, 100] set to NaN | gridMET, not Daymet |

Reported-precision note: model outputs are float64, but source sensor/product
precision does not justify reporting 0.001 °C-level differences. Rounding rules
follow source precision (WTEMP reported precision is 0.1 °C); the +0.05 °C
comparison margin is a pre-frozen numerical threshold, not a minimum important
difference derived from measurement error, ecological effect, or stakeholder
utility.

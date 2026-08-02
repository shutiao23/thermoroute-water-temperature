# SI16 — rights matrix and data dictionary

**Status:** DRAFT / PRE-OPENING / NOT RIGHTS APPROVAL.

This file is a projection schema, not a redistribution decision, DOI record or
FAIR completion claim.

## Rights projection

| Byte class/object | SHA-256 | provider/rightsholder | evidence locator | distribution scope | licence/terms | reviewer decision | binder row ID |
|---|---|---|---|---|---|---|---|
| *(content-deduplicated object)* | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` | `[pending — 探索期数据，不可写入结论]` |

Unknown, conflicting or unreviewed rights remain PUBLIC-excluded. Repository
visibility, provider openness and a software licence are not substitutes for an
exact-byte redistribution decision.

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
| WLEVEL | Daily-mean gage height | USGS NWIS parameter 00065, statistic 00003 | ft | Daily mean | Raw evidence only; not an input to Route-A models | Stored for provenance, not modelled |
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

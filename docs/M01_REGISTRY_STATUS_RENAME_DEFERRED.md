# M-01 minor: registry status enum rename — deferred change record

| Field | Value |
| --- | --- |
| Date | 2026-08-02 |
| Status | `DEFERRED_TO_NEW_LINEAGE` — design only, not applied |
| Review item | Review `m-01` — `USGS_SNAPSHOT_SITE_NO_MATCH` is treated as "verified" by code while its name reads like a match failure |
| Source boundary | Must be coordinated with the new Stage-09 lineage (post `a4b174e1...`); it changes `registry_sha256` and therefore run identity |

## 1. Why it is deferred

`data_usgs/station_registry_v1.csv` holds the status value on all 120 rows, and
the registry file hash enters every Route-A run identity
(`registry_sha256 = 090e7c0d...` for the current Stage-09 `7cb2...`). Renaming
the enum therefore changes the registry hash, which changes the content-addressed
run identity and would invalidate the just-completed Stage-09b config-shape
repair lineage boundary if mixed in.

The Stage09b protected change work order explicitly requires the nine-arm /
five-seed registry to remain unchanged. The rename must therefore be applied as
a **separate allowlisted change on the new source identity**, together with a
regenerated frozen panel and a recomputed Stage-09 lineage.

## 2. Affected files (when executed)

| File | Change |
| --- | --- |
| `scripts/data_usgs/freeze_panel.py:147` | Emit the new enum value for USGS-snapshot site_no joins |
| `data_usgs/station_registry_v1.csv` | 120 rows: `USGS_SNAPSHOT_SITE_NO_MATCH` → new value |
| `src/thermoroute/spatial.py:81` | Match the new value for verified HUC2 clusters |
| `src/thermoroute/opening.py:2097` | Match the new value in the opening completeness contract |
| `src/thermoroute/evidence.py:122` | Counts follow the renamed enum automatically |
| `tests/test_data_evidence.py:46-47`, `tests/test_confirmatory_inputs.py:288,1482`, `tests/test_confirmatory_opening.py:3671-3672` | Fixture values updated |

## 3. Proposed enum name

`MATCHED_BY_SITE_NO_USGS_SNAPSHOT` — states explicitly that the HUC metadata was
matched by stable USGS `site_no` against the frozen USGS snapshot and is treated
as verified. The legacy `LEGACY_SOURCE_SITE_NO_MATCH` path (used when
`huc_source_kind != "usgs_snapshot"`) remains unchanged in meaning.

## 4. Migration steps (with the new lineage)

1. Apply the rename within the same authorized protected-change batch that
   creates the new Stage-09 lineage (after the `a4b174e1...` boundary exists).
2. Regenerate the frozen panel and registry via `scripts/data_usgs/freeze_panel.py`
   with the USGS snapshot source; verify the status counts stay 120/120.
3. Run `pytest -q tests/test_data_evidence.py tests/test_confirmatory_inputs.py
   tests/test_confirmatory_opening.py tests/test_spatial.py` (and full suite).
4. Record the new `registry_sha256` and the resulting new run identity.
5. Document that the pre-rename registry hash remains historical evidence only.

## 5. Non-goals

This record does not change the scientific semantics of any station, HUC
cluster, opening contract, or protocol value. It is a metadata naming fix only.

# Raw API snapshots

`scripts/data_usgs/build_usgs_stations.py` stores every exact USGS, Daymet and
gridMET response under a versioned directory here. Each request has:

- `response.bin`: the unmodified response bytes;
- `metadata.json`: canonical URL/request, UTC retrieval time, HTTP metadata,
  byte count, request fingerprint and response SHA-256;
- the store-level `snapshot_index.json` generated after acquisition.

Raw snapshots can be large and are not reconstructed retroactively for the
legacy 2006–2020 panel. New acquisitions must archive the versioned snapshot
directory in the project's release/data repository. `--offline` verifies and
replays it without network access; a missing or corrupt response aborts.

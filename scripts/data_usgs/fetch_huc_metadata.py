#!/usr/bin/env python3
"""Fetch frozen-cohort HUC metadata by stable USGS site_no.

This is a small metadata-only acquisition.  It does not request water-temperature
values or post-2020 outcomes.  The exact RDB response is retained by
``SnapshotStore`` and the derived CSV is tied to that response in a sidecar.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute.evidence import FrozenPanelSpec  # noqa: E402
from thermoroute.provenance import (  # noqa: E402
    SnapshotStore,
    canonical_json_bytes,
    sha256_file,
)
from thermoroute.usgs import _parse_nwis_rdb  # noqa: E402


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--panel-spec", type=Path,
        default=ROOT / "data_usgs" / "frozen_panel_v1.json",
    )
    parser.add_argument(
        "--snapshot-dir", type=Path,
        default=ROOT / "data_usgs" / "raw_snapshots" / "huc-v1",
    )
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "data_usgs" / "huc_metadata_usgs_v1.csv",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    sidecar = args.out.with_suffix(".provenance.json")
    if not args.force and (args.out.exists() or sidecar.exists()):
        raise FileExistsError("refusing to overwrite frozen HUC metadata without --force")

    spec = FrozenPanelSpec.load(args.panel_spec)
    registry = spec.load_registry()
    site_nos = sorted(registry["site_no"].astype(str))
    url = "https://waterservices.usgs.gov/nwis/site/?" + urlencode({
        "format": "rdb",
        "sites": ",".join(site_nos),
        "siteOutput": "expanded",
        "siteStatus": "all",
    })
    store = SnapshotStore(args.snapshot_dir, offline=args.offline)
    payload, record = store.fetch(provider="usgs-nwis-site-metadata", url=url)
    raw = _parse_nwis_rdb(payload)
    required = {"site_no", "station_nm", "huc_cd", "dec_lat_va", "dec_long_va"}
    missing_columns = required - set(raw.columns)
    if missing_columns:
        raise RuntimeError(f"USGS metadata response missing {sorted(missing_columns)}")
    raw["site_no"] = raw["site_no"].astype("string").str.strip()
    if raw["site_no"].duplicated().any():
        raise RuntimeError("USGS metadata response contains duplicate site_no")
    raw = raw[raw["site_no"].isin(site_nos)].copy()
    missing_sites = set(site_nos) - set(raw["site_no"].astype(str))
    if missing_sites:
        raise RuntimeError(f"USGS metadata missing frozen sites: {sorted(missing_sites)}")
    raw["huc_cd"] = raw["huc_cd"].astype("string").str.strip()
    # NWIS expanded metadata may report either HUC8 or the more specific HUC12.
    valid_huc = raw["huc_cd"].str.fullmatch(r"(?:\d{8}|\d{12})")
    if valid_huc.ne(True).any():
        bad = raw.loc[valid_huc.ne(True), "site_no"]
        raise RuntimeError(f"invalid or missing HUC code for {bad.tolist()}")
    raw["huc2"] = raw["huc_cd"].str[:2]
    columns = [
        "site_no", "station_nm", "dec_lat_va", "dec_long_va", "huc_cd", "huc2",
    ]
    if "drain_area_va" in raw:
        columns.append("drain_area_va")
    result = raw[columns].sort_values("site_no").reset_index(drop=True)
    atomic_write(args.out, result.to_csv(index=False, lineterminator="\n").encode())
    index_path = store.write_index()
    provenance = {
        "schema_version": 1,
        "development_panel_sha256": spec.document["panel"]["sha256"],
        "development_metadata_sha256": (
            spec.document["station_registry"]["source_metadata_sha256"]),
        "site_count": len(result),
        "join_key": "site_no",
        "request_sha256": record.request_sha256,
        "response_sha256": record.response_sha256,
        "retrieved_at_utc": record.retrieved_at_utc,
        "raw_snapshot_index": os.path.relpath(index_path.resolve(), ROOT),
        "raw_snapshot_index_sha256": sha256_file(index_path),
        "derived_csv_sha256": sha256_file(args.out),
        "outcome_data_requested": False,
    }
    atomic_write(sidecar, canonical_json_bytes(provenance))
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create a deterministic stable-key registry and immutable panel contract.

This is a one-time *freeze* operation, not a downloader.  It converts the legacy
``nXX`` mapping in the metadata generated alongside a panel into a registry whose
primary key is the USGS ``site_no``.  The panel bytes are not rewritten; loaders
verify the contract and map the legacy alias at runtime.

Example (the checked-in development freeze):

    PYTHONPATH=src python scripts/data_usgs/freeze_panel.py \
      --panel data_usgs/panel_usgs_120v2.parquet \
      --metadata data_usgs/stations_meta_120v2.csv \
      --huc-source data_usgs/huc_metadata_usgs_v1.csv \
      --huc-source-kind usgs_snapshot \
      --huc-provenance data_usgs/huc_metadata_usgs_v1.provenance.json \
      --registry data_usgs/station_registry_v1.csv \
      --spec data_usgs/frozen_panel_v1.json \
      --panel-id usgs120-development-v1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute.provenance import canonical_json_bytes, sha256_file  # noqa: E402


REQUIRED_PANEL_COLUMNS = (
    "DATE", "site_id", "WTEMP", "FLOW", "WLEVEL", "TEMP", "PRCP",
    "WDSP", "RHMEAN", "DH",
)


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


def relative_to_spec(path: Path, spec_path: Path) -> str:
    return os.path.relpath(path.resolve(), spec_path.resolve().parent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--huc-source", type=Path,
        help="optional legacy HUC table; joined strictly by site_no, never nXX",
    )
    parser.add_argument(
        "--huc-source-kind", choices=("legacy", "usgs_snapshot"), default="legacy")
    parser.add_argument(
        "--huc-provenance", type=Path,
        help="provenance sidecar for --huc-source, when available",
    )
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--panel-id", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for output in (args.registry, args.spec):
        if output.exists() and not args.force:
            raise FileExistsError(f"refusing to overwrite frozen artifact: {output}")

    panel = pd.read_parquet(args.panel)
    missing_panel = set(REQUIRED_PANEL_COLUMNS) - set(panel.columns)
    if missing_panel:
        raise ValueError(f"panel missing columns: {sorted(missing_panel)}")
    panel["DATE"] = pd.to_datetime(panel["DATE"])
    if panel.duplicated(["site_id", "DATE"]).any():
        raise ValueError("panel contains duplicate (site_id, DATE) keys")

    metadata = pd.read_csv(
        args.metadata, dtype={"site": "string", "site_id": "string"},
        keep_default_na=False,
    )
    required_meta = {"site", "site_id", "station_nm", "lat", "lon", "state"}
    missing_meta = required_meta - set(metadata.columns)
    if missing_meta:
        raise ValueError(f"metadata missing columns: {sorted(missing_meta)}")
    registry = metadata.rename(columns={"site": "site_no", "site_id": "legacy_site_id"})
    registry["site_no"] = registry["site_no"].astype("string").str.strip()
    registry["legacy_site_id"] = registry["legacy_site_id"].astype("string").str.strip()
    for key in ("site_no", "legacy_site_id"):
        if registry[key].eq("").any() or registry[key].duplicated().any():
            raise ValueError(f"metadata {key} is empty or non-unique")
    panel_ids = set(panel["site_id"].astype(str))
    registry_ids = set(registry["legacy_site_id"].astype(str))
    if panel_ids != registry_ids:
        raise ValueError(
            "panel and metadata are not the same generation: "
            f"panel_only={sorted(panel_ids - registry_ids)[:5]}, "
            f"metadata_only={sorted(registry_ids - panel_ids)[:5]}")

    huc_contract: dict[str, object] = {
        "status": "NOT_ATTACHED",
        "site_no_joined_count": 0,
        "missing_count": len(registry),
    }
    if args.huc_source is not None:
        huc = pd.read_csv(
            args.huc_source, dtype={"site": "string", "site_no": "string"},
            keep_default_na=False)
        if "site_no" not in huc and "site" in huc:
            huc = huc.rename(columns={"site": "site_no"})
        if "site_no" not in huc.columns:
            raise ValueError("HUC source requires a stable USGS site_no column")
        huc["site_no"] = huc["site_no"].astype("string").str.strip()
        if huc["site_no"].eq("").any() or huc["site_no"].duplicated().any():
            raise ValueError("HUC source site_no is empty or non-unique")
        # Deliberately ignore the source's nXX site_id: it belongs to a different
        # panel generation.  A true site_no join is the only admissible mapping.
        huc_columns = [
            c for c in ("site_no", "huc_cd", "huc2", "huc2_name", "drain_area_va")
            if c in huc.columns
        ]
        registry = registry.merge(
            huc[huc_columns], on="site_no", how="left", validate="one_to_one")
        joined = (
            registry["huc2"].notna()
            & registry["huc2"].astype(str).str.strip().ne("")
        ) if "huc2" in registry else pd.Series(False, index=registry.index)
        joined_status = (
            "USGS_SNAPSHOT_SITE_NO_MATCH"
            if args.huc_source_kind == "usgs_snapshot"
            else "LEGACY_SOURCE_SITE_NO_MATCH"
        )
        registry["huc_metadata_status"] = joined.map({
            True: joined_status,
            False: "UNVERIFIED_MISSING",
        })
        huc_contract = {
            "status": (
                "COMPLETE_USGS_RAW_SNAPSHOT"
                if args.huc_source_kind == "usgs_snapshot" and bool(joined.all())
                else "PARTIAL_LEGACY_SOURCE"
            ),
            "join_key": "site_no",
            "legacy_alias_used": False,
            "joined_status": joined_status,
            "source_path": relative_to_spec(args.huc_source, args.spec),
            "source_sha256": sha256_file(args.huc_source),
            "embedded_in_station_registry": True,
            "runtime_dependency": args.huc_source_kind == "usgs_snapshot",
            "site_no_joined_count": int(joined.sum()),
            "missing_count": int((~joined).sum()),
            "disclosure": (
                "Only exact site_no matches were carried forward. Missing HUC "
                "metadata is not inferred from legacy nXX aliases. The matched "
                "values are embedded in the frozen station registry."
            ),
        }
        if args.huc_provenance is not None:
            huc_contract["provenance_path"] = relative_to_spec(
                args.huc_provenance, args.spec)
            huc_contract["provenance_sha256"] = sha256_file(args.huc_provenance)

    columns = ["site_no", "legacy_site_id"] + [
        c for c in registry.columns if c not in {"site_no", "legacy_site_id"}
    ]
    registry = registry[columns].sort_values("site_no").reset_index(drop=True)
    registry_payload = registry.to_csv(index=False, lineterminator="\n").encode("utf-8")
    atomic_write(args.registry, registry_payload)

    spec = {
        "schema_version": 1,
        "panel_id": args.panel_id,
        "evidence_role": "development_exploratory",
        "evaluation_disclosure": (
            "The 2019-2020 outcomes have already informed model development and "
            "must not be described as blind or untouched."
        ),
        "panel": {
            "path": relative_to_spec(args.panel, args.spec),
            "sha256": sha256_file(args.panel),
            "format": "parquet",
            "row_count": int(len(panel)),
            "station_count": int(panel["site_id"].nunique()),
            "date_start": str(panel["DATE"].min().date()),
            "date_end": str(panel["DATE"].max().date()),
            "legacy_station_key": "site_id",
            "required_columns": list(REQUIRED_PANEL_COLUMNS),
        },
        "station_registry": {
            "path": relative_to_spec(args.registry, args.spec),
            "sha256": sha256_file(args.registry),
            "station_count": int(len(registry)),
            "primary_key": "site_no",
            "legacy_alias": "legacy_site_id",
            "source_metadata_path": relative_to_spec(args.metadata, args.spec),
            "source_metadata_sha256": sha256_file(args.metadata),
            "huc_metadata": huc_contract,
        },
        "legacy_raw_provenance": {
            "status": "UNAVAILABLE_FOR_EXISTING_PANEL",
            "disclosure": (
                "Exact raw HTTP response bytes and request timestamps were not "
                "retained when this legacy panel was created. They cannot be "
                "reconstructed retroactively; future acquisition must use "
                "thermoroute.provenance.SnapshotStore."
            ),
        },
    }
    atomic_write(args.spec, canonical_json_bytes(spec))
    print(json.dumps({
        "spec": str(args.spec),
        "spec_sha256": sha256_file(args.spec),
        "panel_sha256": spec["panel"]["sha256"],
        "registry_sha256": spec["station_registry"]["sha256"],
        "stations": len(registry),
        "rows": len(panel),
    }, indent=2))


if __name__ == "__main__":
    main()

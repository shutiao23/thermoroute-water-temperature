"""Strict accessors for the frozen USGS station registry and HUC2 clusters."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


DEFAULT_STATION_REGISTRY = (
    Path(__file__).resolve().parents[2] / "data_usgs" / "station_registry_v1.csv"
)


def load_station_registry(path: str | Path = DEFAULT_STATION_REGISTRY) -> pd.DataFrame:
    """Load registry identifiers as text and validate HUC2 against HUC8.

    CSV inference drops leading zeroes from USGS and hydrologic identifiers.
    Normalising them centrally prevents scripts from silently creating HUC2
    ``1`` and ``01`` as different clusters or joining stable site numbers to
    legacy ``nXX`` aliases.
    """
    frame = pd.read_csv(
        path,
        dtype={
            "site_no": "string",
            "legacy_site_id": "string",
            "huc_cd": "string",
            "huc2": "string",
            "huc_metadata_status": "string",
        },
        keep_default_na=False,
    )
    required = {
        "site_no", "legacy_site_id", "lat", "lon", "huc_cd", "huc2",
        "huc_metadata_status",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"station registry missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["site_no"] = frame.site_no.str.strip().str.zfill(8)
    frame["legacy_site_id"] = frame.legacy_site_id.str.strip()
    if frame.site_no.str.fullmatch(r"\d{8,15}").ne(True).any():
        raise ValueError("site_no must be a stable numeric USGS identifier")
    if frame.site_no.duplicated().any() or frame.legacy_site_id.duplicated().any():
        raise ValueError("station registry identifiers must be unique")
    # NWIS returns a mixture of HUC8 and HUC12 values.  CSV conversion may have
    # removed one leading zero, so restore to the nearest valid width.
    frame["huc_cd"] = frame.huc_cd.str.strip().map(
        lambda value: value.zfill(8 if len(value) <= 8 else 12) if value else ""
    )
    frame["huc2"] = frame.huc2.str.strip().map(
        lambda value: value.zfill(2) if value else ""
    )
    for row in frame.itertuples(index=False):
        if row.huc_cd and not re.fullmatch(r"(?:\d{8}|\d{12})", row.huc_cd):
            raise ValueError(f"invalid HUC code for site {row.site_no}: {row.huc_cd!r}")
        if row.huc2 and not re.fullmatch(r"\d{2}", row.huc2):
            raise ValueError(f"invalid HUC2 for site {row.site_no}: {row.huc2!r}")
        if row.huc_cd and row.huc2 and row.huc_cd[:2] != row.huc2:
            raise ValueError(f"HUC2/HUC8 mismatch for site {row.site_no}")
    for column in ("lat", "lon", "drain_area_va"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("site_no").reset_index(drop=True)


def huc2_cluster_map(registry: pd.DataFrame) -> dict[str, str]:
    """Return verified HUC2 clusters; unresolved sites remain separate."""
    required = {"site_no", "huc2", "huc_metadata_status"}
    missing = required - set(registry)
    if missing:
        raise ValueError(f"station registry missing columns: {sorted(missing)}")
    result: dict[str, str] = {}
    for row in registry.itertuples(index=False):
        site = str(row.site_no).zfill(8)
        huc2 = str(row.huc2)
        verified = str(row.huc_metadata_status) == "USGS_SNAPSHOT_SITE_NO_MATCH"
        result[site] = f"HUC2:{huc2}" if verified and re.fullmatch(r"\d{2}", huc2) \
            else f"UNMAPPED:{site}"
    return result

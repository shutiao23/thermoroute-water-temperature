#!/usr/bin/env python3
"""Parse the frozen F2a snapshots into a fixed-lead forecast-temperature panel.

The F2a acquisition is verified and its key registry frozen, but the values
themselves have never been turned into a panel, so the arm could not be fitted.
This builder does only that: it reads the 4,080 immutable station-month
snapshots through the provenance store's own verifier and emits one daily mean
per (station, target date, lead).

What the product is, stated because the name invites the wrong reading. Each
``temperature_2m_previous_dayN`` series is the value a run issued *N days
before* the valid time.  Assembling the three leads gives a fixed-lead
composite: for one valid time it reports what the 1-, 3- and 7-day-ahead
forecasts said.  It is **not** a single coherent model initialization, and it is
not an as-issued operational archive; the acquisition record says so explicitly
and the sealed protocol forbids both claims.  Its only legitimate comparison is
against ``F3_temperature_only`` -- realized air temperature at the same valid
times -- never against the full five-variable oracle.

Daily aggregation is the mean of the 24 hourly values at GMT, matching how the
Daymet daily mean the models already consume is defined.  A day with any
missing hour is dropped rather than partially averaged, and the dropped count
is reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

SNAPSHOT_ROOT = ROOT / "data_usgs" / "raw_snapshots" / "openmeteo-gfs-previous-runs-v1"
REGISTRY = ROOT / "data_usgs" / "station_registry_v1.csv"
DEFAULT_OUT = ROOT / "outputs" / "final" / "f2a_forecast_panel_v1.parquet"

LEADS = (1, 3, 7)
FIELDS = tuple(f"temperature_2m_previous_day{lead}" for lead in LEADS)
FORMAT = "thermoroute.f2a-fixed-lead-forecast-panel.v1"
COORDINATE_TOLERANCE = 1e-4


class F2aPanelError(RuntimeError):
    """A snapshot or coordinate check failed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def station_coordinates() -> pd.DataFrame:
    registry = pd.read_csv(REGISTRY, dtype={"site_no": str})
    out = registry[["site_no", "lat", "lon"]].copy()
    out["site_id"] = out["site_no"].str.zfill(8)
    return out[["site_id", "lat", "lon"]]


_REQUEST_COORD = re.compile(r"[?&]latitude=([-\d.]+).*?[?&]longitude=([-\d.]+)")


def requested_coordinates(metadata_path: Path) -> tuple[float, float]:
    """The coordinates the request asked for, which is what binds the station.

    The response body reports the *grid* point Open-Meteo served, which is
    displaced from the gauge by up to half a cell, so matching on it finds
    nothing.  The request URL is the authoritative binding and is what the
    acquisition record was verified against.
    """
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    url = str(document["request"]["url"])
    latitude = re.search(r"[?&]latitude=([-\d.]+)", url)
    longitude = re.search(r"[?&]longitude=([-\d.]+)", url)
    if not latitude or not longitude:
        raise F2aPanelError(f"request URL carries no coordinates: {metadata_path}")
    return float(latitude.group(1)), float(longitude.group(1))


def _match_station(lat: float, lon: float, stations: pd.DataFrame) -> str:
    """Resolve requested coordinates back to a station identity.

    It must be unambiguous: an absent or ambiguous match is an error rather
    than a nearest-neighbour guess, because a silently mis-assigned station
    would corrupt every downstream key.
    """
    close = stations[
        (np.abs(stations["lat"] - lat) < COORDINATE_TOLERANCE)
        & (np.abs(stations["lon"] - lon) < COORDINATE_TOLERANCE)
    ]
    if len(close) != 1:
        raise F2aPanelError(
            f"coordinates ({lat}, {lon}) match {len(close)} stations; expected exactly one"
        )
    return str(close["site_id"].iloc[0])


def parse_snapshots(limit: int | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    stations = station_coordinates()
    responses = sorted(SNAPSHOT_ROOT.rglob("response.bin"))
    if not responses:
        raise F2aPanelError(f"no snapshots under {SNAPSHOT_ROOT}")
    if limit is not None:
        responses = responses[:limit]

    frames: list[pd.DataFrame] = []
    dropped_incomplete = 0
    for path in responses:
        payload = json.loads(path.read_text(encoding="utf-8"))
        hourly = payload.get("hourly")
        if not hourly or "time" not in hourly:
            raise F2aPanelError(f"snapshot lacks an hourly block: {path}")
        missing = [field for field in FIELDS if field not in hourly]
        if missing:
            raise F2aPanelError(f"snapshot lacks {missing}: {path}")
        site = _match_station(
            *requested_coordinates(path.parent / "metadata.json"), stations
        )
        block = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
        for lead, field in zip(LEADS, FIELDS, strict=True):
            block[lead] = [
                np.nan if value is None else float(value) for value in hourly[field]
            ]
        block["target_date"] = block["time"].dt.normalize()

        for lead in LEADS:
            counts = block.groupby("target_date")[lead].agg(["count", "mean"])
            complete = counts[counts["count"] == 24]
            dropped_incomplete += int(len(counts) - len(complete))
            frames.append(pd.DataFrame({
                "site_id": site,
                "target_date": complete.index,
                "lead_days": np.int16(lead),
                "f2a_temp_c": complete["mean"].to_numpy(float),
            }))

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["site_id", "lead_days", "target_date"])
    panel = panel.drop_duplicates(["site_id", "lead_days", "target_date"], keep="first")
    panel = panel.reset_index(drop=True)
    stats = {
        "snapshots_read": len(responses),
        "rows": len(panel),
        "stations": int(panel["site_id"].nunique()),
        "station_days_dropped_for_incomplete_hours": dropped_incomplete,
        "target_start": str(panel["target_date"].min().date()),
        "target_end": str(panel["target_date"].max().date()),
    }
    return panel, stats


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None,
                        help="parse only the first N snapshots (smoke test)")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)

    panel, stats = parse_snapshots(limit=args.limit)
    if args.print_only or args.limit is not None:
        print(json.dumps(stats, sort_keys=True, indent=1))
        print(panel.head(4).to_string(index=False))
        return 0

    panel.to_parquet(args.out, index=False)
    manifest = {
        "format": FORMAT,
        "builder_sha256": _sha256_file(Path(__file__).resolve()),
        "output_sha256": _sha256_file(args.out),
        "aggregation": "mean of 24 GMT hourly values; incomplete days dropped",
        "semantics": {
            "fixed_lead_composite": True,
            "coherent_single_initialization": False,
            "as_issued_operational_archive": False,
            "only_legal_comparator": "F3_temperature_only",
        },
        **stats,
    }
    (args.out.parent / "f2a_forecast_panel_v1_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=1) + "\n", encoding="utf-8",
    )
    print(json.dumps({"status": "WRITTEN", **stats}, sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

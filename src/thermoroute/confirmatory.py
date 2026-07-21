"""Metadata-only discovery for the sealed Route-A new-site cohort.

Candidate discovery is intentionally narrower than data acquisition.  It asks
the USGS *site metadata* endpoint which stream sites advertise daily-value
water-temperature capability; it never requests a daily-value record, a date
range, an outcome value, or holdout-period coverage.  Selection remains a
separate deterministic operation in :mod:`thermoroute.evidence`.
"""

from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import urlencode

import pandas as pd

from .evidence import EvidenceError, FORBIDDEN_CONFIRMATORY_COLUMNS
from .usgs import _parse_nwis_rdb


USGS_SITE_ENDPOINT = "https://waterservices.usgs.gov/nwis/site/"

# Geographic support is frozen to the states represented in the 120-site
# development registry.  This universe is metadata-derived and was fixed before
# any post-2020 labels were acquired.  Callers may supply a different *explicit*
# universe, which must then be recorded in the discovery provenance document.
ROUTE_A_STATE_UNIVERSE = (
    "AL", "AR", "AZ", "CA", "CO", "FL", "GA", "IA", "ID", "IN", "KY",
    "MA", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NH", "NJ", "NV",
    "NY", "OH", "OK", "OR", "PA", "SC", "TX", "UT", "WA", "WI", "WV",
    "WY",
)

CANDIDATE_COLUMNS = (
    "site_no", "station_nm", "lat", "lon", "state", "site_type", "huc_cd",
    "drain_area_va",
)


def normalise_states(states: Iterable[str]) -> tuple[str, ...]:
    """Return a unique, sorted, explicit two-letter state universe."""
    normalised = tuple(sorted({str(state).strip().upper() for state in states}))
    if not normalised:
        raise ValueError("candidate discovery requires at least one state")
    invalid = [state for state in normalised if len(state) != 2 or not state.isalpha()]
    if invalid:
        raise ValueError(f"invalid two-letter state codes: {invalid}")
    return normalised


def build_usgs_candidate_url(state: str) -> str:
    """Build one canonical metadata-only NWIS request URL.

    ``parameterCd=00010`` and ``hasDataTypeCd=dv`` describe advertised station
    capability.  The site endpoint returns metadata, not the WTEMP observations.
    No start/end date or coverage predicate is permitted here.
    """
    state = normalise_states([state])[0]
    params = {
        "agencyCd": "USGS",
        "format": "rdb",
        "hasDataTypeCd": "dv",
        "parameterCd": "00010",
        "siteOutput": "expanded",
        "siteStatus": "all",
        "siteType": "ST",
        "stateCd": state,
    }
    return USGS_SITE_ENDPOINT + "?" + urlencode(sorted(params.items()))


def _assert_metadata_only_columns(columns: Iterable[object]) -> None:
    lowered = {str(column).strip().lower() for column in columns}
    forbidden = sorted(
        column for column in lowered
        if any(token in column for token in FORBIDDEN_CONFIRMATORY_COLUMNS)
    )
    if forbidden:
        raise EvidenceError(
            "candidate discovery output contains outcome/coverage fields: "
            f"{forbidden}"
        )


def parse_usgs_candidate_metadata(payload: bytes, *, state: str) -> pd.DataFrame:
    """Parse one state response into the frozen metadata-only candidate schema."""
    state = normalise_states([state])[0]
    raw = _parse_nwis_rdb(payload)
    required = {"site_no", "station_nm", "dec_lat_va", "dec_long_va"}
    missing = required - set(raw.columns)
    if missing:
        raise EvidenceError(
            f"USGS candidate metadata for {state} lacks {sorted(missing)}"
        )
    if "site_tp_cd" in raw and raw["site_tp_cd"].astype(str).str.strip().ne("ST").any():
        raise EvidenceError(f"USGS metadata for {state} returned a non-stream site")
    if "agency_cd" in raw and raw["agency_cd"].astype(str).str.strip().ne("USGS").any():
        raise EvidenceError(f"USGS metadata for {state} returned a non-USGS site")

    out = pd.DataFrame({
        "site_no": raw["site_no"].astype("string").str.strip(),
        "station_nm": raw["station_nm"].astype("string").str.strip(),
        "lat": pd.to_numeric(raw["dec_lat_va"], errors="coerce"),
        "lon": pd.to_numeric(raw["dec_long_va"], errors="coerce"),
        "state": state,
        "site_type": (
            raw["site_tp_cd"].astype("string").str.strip().fillna("")
            if "site_tp_cd" in raw else "ST"
        ),
        "huc_cd": (
            raw["huc_cd"].astype("string").str.strip().fillna("")
            if "huc_cd" in raw else pd.Series("", index=raw.index, dtype="string")
        ),
        "drain_area_va": (
            pd.to_numeric(raw["drain_area_va"], errors="coerce")
            if "drain_area_va" in raw else float("nan")
        ),
    })
    if out["site_no"].eq("").any() or out["site_no"].isna().any():
        raise EvidenceError(f"USGS candidate metadata for {state} has an empty site_no")
    out["station_nm"] = out["station_nm"].fillna("")
    if out["site_no"].duplicated().any():
        raise EvidenceError(f"USGS candidate metadata for {state} duplicates site_no")
    valid_coordinates = (
        out["lat"].between(-90.0, 90.0)
        & out["lon"].between(-180.0, 180.0)
    )
    if valid_coordinates.ne(True).any():
        bad = out.loc[valid_coordinates.ne(True), "site_no"].tolist()
        raise EvidenceError(f"invalid candidate coordinates for {bad[:10]}")
    out = out[list(CANDIDATE_COLUMNS)].sort_values("site_no").reset_index(drop=True)
    _assert_metadata_only_columns(out.columns)
    return out


def merge_candidate_metadata(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Combine state responses with deterministic ordering and strict identity."""
    materialised = list(frames)
    if not materialised:
        raise EvidenceError("candidate discovery returned no state responses")
    for frame in materialised:
        if tuple(frame.columns) != CANDIDATE_COLUMNS:
            raise EvidenceError("candidate frame does not match the metadata-only schema")
        _assert_metadata_only_columns(frame.columns)
    candidates = pd.concat(materialised, ignore_index=True)
    if candidates.empty:
        raise EvidenceError("candidate discovery returned no sites")
    duplicates = candidates[candidates["site_no"].duplicated(keep=False)]
    if not duplicates.empty:
        sites = sorted(duplicates["site_no"].astype(str).unique())
        raise EvidenceError(f"candidate site_no appears in multiple responses: {sites[:10]}")
    return candidates.sort_values(["site_no", "state"]).reset_index(drop=True)

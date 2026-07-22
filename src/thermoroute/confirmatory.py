"""Metadata-only discovery for the sealed Route-A new-site cohort.

Candidate discovery is intentionally narrower than data acquisition.  It asks
the USGS *site metadata* endpoint which stream sites advertise daily-value
water-temperature capability; it never requests a daily-value record, a date
range, an outcome value, or holdout-period coverage.  Selection remains a
separate deterministic operation in :mod:`thermoroute.evidence`.
"""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode

import pandas as pd

from .evidence import EvidenceError, FORBIDDEN_CONFIRMATORY_COLUMNS
from .usgs import _parse_nwis_rdb
from .provenance import canonical_json_bytes, sha256_bytes, sha256_file


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
CANDIDATE_PROVIDER = "usgs-nwis-confirmatory-site-metadata"
CANDIDATE_USER_AGENT = "ThermoRoute/1.0 Route-A metadata-only discovery"


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


def replay_candidate_evidence(
    candidates_path: str | Path,
    provenance_path: str | Path,
    snapshot_index_path: str | Path,
    *,
    protocol_sha256: str,
    state_universe: Iterable[str],
) -> pd.DataFrame:
    """Strictly replay metadata-only raw bytes into the candidate universe."""
    candidates_path = Path(candidates_path).resolve()
    provenance_path = Path(provenance_path).resolve()
    snapshot_index_path = Path(snapshot_index_path).resolve()
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        index = json.loads(snapshot_index_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise EvidenceError("candidate provenance/index is absent or invalid") from exc
    expected_states = normalise_states(state_universe)
    expected_provenance_keys = {
        "schema_version", "artifact_role", "protocol_sha256", "state_universe",
        "state_universe_rule", "candidate_rule", "candidate_count",
        "site_primary_key", "sort_order", "columns", "outcome_endpoint_requested",
        "outcome_values_requested", "holdout_coverage_requested_or_computed",
        "raw_snapshot_index", "raw_snapshot_index_sha256",
        "candidate_table_sha256", "requests",
    }
    if not isinstance(provenance, Mapping) or set(provenance) != expected_provenance_keys:
        raise EvidenceError("candidate provenance schema changed")
    expected_flags: dict[str, Any] = {
        "schema_version": 1,
        "artifact_role": "PRE_LABEL_METADATA_ONLY_CANDIDATE_UNIVERSE",
        "protocol_sha256": protocol_sha256,
        "state_universe": list(expected_states),
        "site_primary_key": "site_no",
        "sort_order": ["site_no", "state"],
        "columns": list(CANDIDATE_COLUMNS),
        "outcome_endpoint_requested": False,
        "outcome_values_requested": False,
        "holdout_coverage_requested_or_computed": False,
        "raw_snapshot_index_sha256": sha256_file(snapshot_index_path),
        "candidate_table_sha256": sha256_file(candidates_path),
    }
    if any(provenance.get(key) != value for key, value in expected_flags.items()):
        raise EvidenceError("candidate provenance identity/flags changed")
    if index.get("schema_version") != 1:
        raise EvidenceError("candidate snapshot index schema changed")
    records = index.get("records")
    if not isinstance(records, list) or index.get("snapshot_count") != len(records):
        raise EvidenceError("candidate snapshot index count changed")
    if len(records) != len(expected_states):
        raise EvidenceError("candidate snapshot count differs from state universe")
    indexed: dict[str, Mapping[str, Any]] = {}
    snapshot_root = snapshot_index_path.parent
    for record in records:
        required_record = {
            "provider", "request_sha256", "response_sha256", "retrieved_at_utc",
            "byte_count", "request", "metadata_path", "response_path",
        }
        if not isinstance(record, Mapping) or set(record) != required_record:
            raise EvidenceError("candidate snapshot record schema changed")
        request = record.get("request")
        if not isinstance(request, Mapping):
            raise EvidenceError("candidate snapshot request is malformed")
        request_sha = sha256_bytes(canonical_json_bytes(dict(request)))
        if request_sha != record.get("request_sha256") or request_sha in indexed:
            raise EvidenceError("candidate request fingerprint changed or duplicated")
        response_path = (snapshot_root / str(record["response_path"])).resolve()
        metadata_path = (snapshot_root / str(record["metadata_path"])).resolve()
        if (
            snapshot_root not in response_path.parents
            or snapshot_root not in metadata_path.parents
            or response_path.name != "response.bin"
            or metadata_path.name != "metadata.json"
            or response_path.parent != metadata_path.parent
            or response_path.parent.name != request_sha
        ):
            raise EvidenceError("candidate snapshot path layout changed")
        payload = response_path.read_bytes()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        response_sha = sha256_bytes(payload)
        expected_metadata = {
            "schema_version": 1,
            "request": dict(request),
            "request_sha256": request_sha,
            "http_status": 200,
            "byte_count": len(payload),
            "response_sha256": response_sha,
            "response_file": "response.bin",
        }
        if (
            response_sha != record.get("response_sha256")
            or int(record.get("byte_count", -1)) != len(payload)
            or any(metadata.get(key) != value for key, value in expected_metadata.items())
            or metadata.get("retrieved_at_utc") != record.get("retrieved_at_utc")
        ):
            raise EvidenceError("candidate snapshot response/metadata binding changed")
        indexed[request_sha] = record

    requests = provenance.get("requests")
    if not isinstance(requests, list) or len(requests) != len(expected_states):
        raise EvidenceError("candidate provenance request registry changed")
    frames: list[pd.DataFrame] = []
    seen_states: list[str] = []
    for request_record in requests:
        if not isinstance(request_record, Mapping) or set(request_record) != {
            "state", "candidate_count", "request_sha256", "response_sha256",
            "retrieved_at_utc", "byte_count",
        }:
            raise EvidenceError("candidate provenance request row changed")
        state = str(request_record["state"])
        canonical_request = {
            "schema_version": 1,
            "provider": CANDIDATE_PROVIDER,
            "method": "GET",
            "url": build_usgs_candidate_url(state),
            "headers": {"User-Agent": CANDIDATE_USER_AGENT},
        }
        request_sha = sha256_bytes(canonical_json_bytes(canonical_request))
        if request_sha != request_record.get("request_sha256") or request_sha not in indexed:
            raise EvidenceError("candidate provenance request is not canonical")
        indexed_record = indexed[request_sha]
        if dict(indexed_record["request"]) != canonical_request:
            raise EvidenceError("candidate indexed request byte contract changed")
        for field in (
            "request_sha256", "response_sha256", "retrieved_at_utc", "byte_count"
        ):
            if request_record.get(field) != indexed_record.get(field):
                raise EvidenceError("candidate provenance does not bind raw snapshot")
        response_path = (snapshot_root / str(indexed_record["response_path"])).resolve()
        frame = parse_usgs_candidate_metadata(response_path.read_bytes(), state=state)
        if int(request_record["candidate_count"]) != len(frame):
            raise EvidenceError("candidate per-state count changed")
        frames.append(frame)
        seen_states.append(state)
    if tuple(sorted(seen_states)) != expected_states or len(seen_states) != len(set(seen_states)):
        raise EvidenceError("candidate raw-response state universe changed")
    rebuilt = merge_candidate_metadata(frames)
    if int(provenance.get("candidate_count", -1)) != len(rebuilt):
        raise EvidenceError("candidate total count changed")
    provided = pd.read_csv(
        candidates_path,
        dtype={
            "site_no": "string", "station_nm": "string", "state": "string",
            "site_type": "string", "huc_cd": "string",
        },
        keep_default_na=False,
    )
    if tuple(provided.columns) != CANDIDATE_COLUMNS:
        raise EvidenceError("candidate table schema changed")
    for column in ("lat", "lon", "drain_area_va"):
        provided[column] = pd.to_numeric(provided[column], errors="coerce")
    provided["huc_cd"] = provided["huc_cd"].fillna("")
    try:
        pd.testing.assert_frame_equal(
            rebuilt, provided, check_dtype=False, rtol=0.0, atol=0.0
        )
    except AssertionError as exc:
        raise EvidenceError("candidate table cannot be replayed from raw bytes") from exc
    return provided

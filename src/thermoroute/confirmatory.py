"""Metadata-only discovery for the sealed Route-A new-site cohort.

Candidate discovery is intentionally narrower than data acquisition.  It asks
the USGS *site metadata* endpoint which stream sites advertise daily-value
water-temperature capability; it never requests a daily-value record, a date
range, an outcome value, or holdout-period coverage.  Selection remains a
separate deterministic operation in :mod:`thermoroute.evidence`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
import io
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping
from urllib.parse import urlencode

import pandas as pd

from .evidence import EvidenceError, FORBIDDEN_CONFIRMATORY_COLUMNS
from .usgs import _parse_nwis_rdb
from .provenance import (
    ProvenanceError,
    canonical_json_bytes,
    read_single_link_regular,
    require_canonical_utc,
    sha256_bytes,
)


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
CANDIDATE_STATE_UNIVERSE_RULE = (
    "states represented in the frozen 120-site development registry; "
    "no post-2020 outcome or coverage information"
)
CANDIDATE_SELECTION_RULE = (
    "USGS stream sites whose site metadata advertises daily-value "
    "parameter 00010 capability; siteStatus=all"
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


def canonical_candidate_request(state: str) -> dict[str, object]:
    """Return the complete, identity-bearing metadata request for one state."""
    return {
        "schema_version": 1,
        "provider": CANDIDATE_PROVIDER,
        "method": "GET",
        "url": build_usgs_candidate_url(state),
        "headers": {"User-Agent": CANDIDATE_USER_AGENT},
    }


def _strict_canonical_json(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite JSON number {value}")
        return parsed

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        document: dict[str, Any] = {}
        for key, value in pairs:
            if key in document:
                raise ValueError(f"duplicate JSON key {key!r}")
            document[key] = value
        return document

    try:
        document = json.loads(
            payload.decode("utf-8", errors="strict"),
            parse_constant=reject_constant,
            parse_float=finite_float,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise EvidenceError(f"{label} is not strict JSON") from exc
    if not isinstance(document, dict) or payload != canonical_json_bytes(document):
        raise EvidenceError(f"{label} is not canonical producer JSON")
    return document


def _candidate_raw_paths(states: tuple[str, ...]) -> set[str]:
    paths: set[str] = set()
    for state in states:
        request = canonical_candidate_request(state)
        request_sha = sha256_bytes(canonical_json_bytes(request))
        base = Path(CANDIDATE_PROVIDER) / request_sha
        paths.add((base / "metadata.json").as_posix())
        paths.add((base / "response.bin").as_posix())
    return paths


def audit_candidate_snapshot_tree(
    snapshot_root: str | Path,
    states: Iterable[str],
    *,
    allow_index: bool,
    require_all_raw: bool,
) -> None:
    """Require the exact candidate snapshot namespace, with no linked nodes."""
    root = Path(snapshot_root)
    expected_raw_files = _candidate_raw_paths(normalise_states(states))
    if not os.path.lexists(root):
        if require_all_raw:
            raise EvidenceError("candidate raw snapshot root is absent")
        return
    root_metadata = root.lstat()
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise EvidenceError("candidate raw snapshot root must be one real directory")
    expected_files = set(expected_raw_files)
    if allow_index:
        expected_files.add("snapshot_index.json")
    expected_directories: set[str] = set()
    for relative in expected_raw_files:
        parent = Path(relative).parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for directory, names, files in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in names:
            path = directory_path / name
            metadata = path.lstat()
            relative = path.relative_to(root).as_posix()
            if not stat.S_ISDIR(metadata.st_mode):
                raise EvidenceError(f"candidate snapshot directory is unsafe: {relative}")
            actual_directories.add(relative)
        for name in files:
            path = directory_path / name
            metadata = path.lstat()
            relative = path.relative_to(root).as_posix()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise EvidenceError(f"candidate snapshot file is linked/unsafe: {relative}")
            actual_files.add(relative)
    unexpected_files = actual_files - expected_files
    unexpected_directories = actual_directories - expected_directories
    if unexpected_files or unexpected_directories:
        raise EvidenceError(
            "candidate snapshot store contains extra nodes: "
            f"files={sorted(unexpected_files)[:5]}, "
            f"directories={sorted(unexpected_directories)[:5]}"
        )
    for relative in expected_raw_files:
        peer = (
            relative.removesuffix("metadata.json") + "response.bin"
            if relative.endswith("metadata.json")
            else relative.removesuffix("response.bin") + "metadata.json"
        )
        if (relative in actual_files) != (peer in actual_files):
            raise EvidenceError("candidate raw snapshot transaction is incomplete")
    if require_all_raw and not expected_raw_files <= actual_files:
        missing = sorted(expected_raw_files - actual_files)
        raise EvidenceError(f"candidate raw snapshot requests are missing: {missing[:5]}")


@dataclass(frozen=True)
class CandidateSnapshotAudit:
    index_document: Mapping[str, Any]
    index_payload: bytes
    responses_by_state: Mapping[str, bytes]


def audit_candidate_snapshot_store(
    snapshot_root: str | Path,
    state_universe: Iterable[str],
    *,
    trusted_root: str | Path,
    published_index_payload: bytes | None = None,
) -> CandidateSnapshotAudit:
    """Reconstruct and optionally byte-verify the unique canonical raw index."""
    root = Path(os.path.abspath(snapshot_root))
    states = normalise_states(state_universe)
    allow_index = published_index_payload is not None
    audit_candidate_snapshot_tree(
        root, states, allow_index=allow_index, require_all_raw=True
    )
    records: list[dict[str, Any]] = []
    responses: dict[str, bytes] = {}
    metadata_keys = {
        "schema_version", "request", "request_sha256", "retrieved_at_utc",
        "http_status", "response_headers", "byte_count", "response_sha256",
        "response_file", "final_url", "retrieval_semantics",
    }
    for state in states:
        request = canonical_candidate_request(state)
        request_sha = sha256_bytes(canonical_json_bytes(request))
        base = Path(CANDIDATE_PROVIDER) / request_sha
        metadata_path = root / base / "metadata.json"
        response_path = root / base / "response.bin"
        try:
            metadata_payload = read_single_link_regular(
                metadata_path,
                label=f"candidate metadata for {state}",
                trusted_root=trusted_root,
            )
            response_payload = read_single_link_regular(
                response_path,
                label=f"candidate response for {state}",
                trusted_root=trusted_root,
            )
            metadata = _strict_canonical_json(
                metadata_payload, label=f"candidate metadata for {state}"
            )
            retrieved = require_canonical_utc(
                metadata.get("retrieved_at_utc"),
                label=f"candidate retrieved_at_utc for {state}",
            )
        except ProvenanceError as exc:
            raise EvidenceError(str(exc)) from exc
        response_sha = sha256_bytes(response_payload)
        embedded_request = metadata.get("request")
        response_headers = metadata.get("response_headers")
        if (
            set(metadata) != metadata_keys
            or type(metadata.get("schema_version")) is not int
            or metadata.get("schema_version") != 2
            or not isinstance(embedded_request, Mapping)
            or embedded_request != request
            or sha256_bytes(canonical_json_bytes(dict(embedded_request)))
            != request_sha
            or metadata.get("request_sha256") != request_sha
            or type(metadata.get("http_status")) is not int
            or metadata.get("http_status") != 200
            or type(response_headers) is not dict
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in response_headers.items()
            )
            or type(metadata.get("byte_count")) is not int
            or metadata.get("byte_count") != len(response_payload)
            or metadata.get("response_sha256") != response_sha
            or metadata.get("response_file") != "response.bin"
            or metadata.get("final_url") != request["url"]
            or metadata.get("retrieval_semantics") not in {
                "DIRECT_HTTP_RESPONSE",
                "BYTE_IDENTICAL_REFETCH_COMPLETED_RESPONSE_ONLY_TRANSACTION",
            }
        ):
            raise EvidenceError("candidate snapshot response/metadata binding changed")
        records.append({
            "provider": CANDIDATE_PROVIDER,
            "request_sha256": request_sha,
            "response_sha256": response_sha,
            "retrieved_at_utc": retrieved,
            "byte_count": len(response_payload),
            "metadata_sha256": sha256_bytes(metadata_payload),
            "metadata_byte_count": len(metadata_payload),
            "request": request,
            "metadata_path": (base / "metadata.json").as_posix(),
            "response_path": (base / "response.bin").as_posix(),
        })
        responses[state] = response_payload
    records.sort(key=lambda record: str(record["metadata_path"]))
    retrieved_datetimes = [
        datetime.fromisoformat(str(record["retrieved_at_utc"]))
        for record in records
    ]
    if max(retrieved_datetimes) - min(retrieved_datetimes) > timedelta(days=1):
        raise EvidenceError(
            "candidate acquisition exceeded its 24-hour session; use a new snapshot version"
        )
    index_document: dict[str, Any] = {
        "schema_version": 2,
        "snapshot_count": len(records),
        "records": records,
    }
    index_payload = canonical_json_bytes(index_document)
    if published_index_payload is not None and published_index_payload != index_payload:
        raise EvidenceError("candidate snapshot index differs from exact raw replay")
    audit_candidate_snapshot_tree(
        root, states, allow_index=allow_index, require_all_raw=True
    )
    return CandidateSnapshotAudit(
        index_document=index_document,
        index_payload=index_payload,
        responses_by_state=responses,
    )


@dataclass(frozen=True)
class CandidateEvidenceAudit:
    candidates: pd.DataFrame
    candidate_payload: bytes
    provenance_payload: bytes
    snapshot_index_payload: bytes
    provenance: Mapping[str, Any]
    snapshot: CandidateSnapshotAudit


def audit_candidate_evidence(
    candidates_path: str | Path,
    provenance_path: str | Path,
    snapshot_index_path: str | Path,
    *,
    protocol_sha256: str | None,
    state_universe: Iterable[str] | None,
    trusted_root: str | Path | None = None,
) -> CandidateEvidenceAudit:
    """Strictly replay metadata-only raw bytes into the candidate universe."""
    candidates_path = Path(os.path.abspath(candidates_path))
    provenance_path = Path(os.path.abspath(provenance_path))
    snapshot_index_path = Path(os.path.abspath(snapshot_index_path))
    if trusted_root is None:
        trusted_root = Path(os.path.commonpath([
            candidates_path.parent,
            provenance_path.parent,
            snapshot_index_path.parent,
        ]))
    trusted_root = Path(os.path.abspath(trusted_root))
    try:
        candidate_payload = read_single_link_regular(
            candidates_path, label="candidate table", trusted_root=trusted_root
        )
        provenance_payload = read_single_link_regular(
            provenance_path, label="candidate provenance", trusted_root=trusted_root
        )
        index_payload = read_single_link_regular(
            snapshot_index_path,
            label="candidate snapshot index",
            trusted_root=trusted_root,
        )
        provenance = _strict_canonical_json(
            provenance_payload, label="candidate provenance"
        )
    except (ProvenanceError, EvidenceError) as exc:
        raise EvidenceError("candidate provenance/index is absent or invalid") from exc
    if protocol_sha256 is None:
        protocol_sha256 = str(provenance.get("protocol_sha256", ""))
    if state_universe is None:
        state_universe = provenance.get("state_universe", [])
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
    if (
        type(provenance.get("schema_version")) is not int
        or type(provenance.get("candidate_count")) is not int
        or any(
            type(provenance.get(field)) is not bool
            for field in (
                "outcome_endpoint_requested",
                "outcome_values_requested",
                "holdout_coverage_requested_or_computed",
            )
        )
    ):
        raise EvidenceError("candidate provenance scalar types changed")
    expected_flags: dict[str, Any] = {
        "schema_version": 1,
        "artifact_role": "PRE_LABEL_METADATA_ONLY_CANDIDATE_UNIVERSE",
        "protocol_sha256": protocol_sha256,
        "state_universe": list(expected_states),
        "state_universe_rule": CANDIDATE_STATE_UNIVERSE_RULE,
        "candidate_rule": CANDIDATE_SELECTION_RULE,
        "site_primary_key": "site_no",
        "sort_order": ["site_no", "state"],
        "columns": list(CANDIDATE_COLUMNS),
        "outcome_endpoint_requested": False,
        "outcome_values_requested": False,
        "holdout_coverage_requested_or_computed": False,
        "raw_snapshot_index_sha256": sha256_bytes(index_payload),
        "candidate_table_sha256": sha256_bytes(candidate_payload),
    }
    try:
        expected_index_reference = snapshot_index_path.relative_to(
            trusted_root
        ).as_posix()
    except ValueError as exc:
        raise EvidenceError("candidate snapshot index escapes the trusted root") from exc
    expected_flags["raw_snapshot_index"] = expected_index_reference
    if provenance.get("candidate_table_sha256") != sha256_bytes(candidate_payload):
        raise EvidenceError("candidate table checksum differs from provenance")
    if provenance.get("raw_snapshot_index_sha256") != sha256_bytes(index_payload):
        raise EvidenceError("candidate snapshot index checksum differs from provenance")
    if any(provenance.get(key) != value for key, value in expected_flags.items()):
        raise EvidenceError("candidate provenance identity/flags changed")
    snapshot_root = snapshot_index_path.parent
    snapshot = audit_candidate_snapshot_store(
        snapshot_root,
        expected_states,
        trusted_root=trusted_root,
        published_index_payload=index_payload,
    )
    indexed = {
        str(record["request_sha256"]): record
        for record in snapshot.index_document["records"]
    }

    requests = provenance.get("requests")
    if not isinstance(requests, list) or len(requests) != len(expected_states):
        raise EvidenceError("candidate provenance request registry changed")
    for request_record in requests:
        if (
            not isinstance(request_record, Mapping)
            or set(request_record) != {
                "state", "candidate_count", "request_sha256", "response_sha256",
                "retrieved_at_utc", "byte_count",
            }
            or not isinstance(request_record.get("state"), str)
            or type(request_record.get("candidate_count")) is not int
            or type(request_record.get("byte_count")) is not int
            or not isinstance(request_record.get("request_sha256"), str)
            or not isinstance(request_record.get("response_sha256"), str)
            or not isinstance(request_record.get("retrieved_at_utc"), str)
        ):
            raise EvidenceError("candidate provenance request scalar types changed")
    frames: list[pd.DataFrame] = []
    expected_requests: list[dict[str, Any]] = []
    for state in expected_states:
        canonical_request = canonical_candidate_request(state)
        request_sha = sha256_bytes(canonical_json_bytes(canonical_request))
        if request_sha not in indexed:
            raise EvidenceError("candidate provenance request is not canonical")
        indexed_record = indexed[request_sha]
        if dict(indexed_record["request"]) != canonical_request:
            raise EvidenceError("candidate indexed request byte contract changed")
        frame = parse_usgs_candidate_metadata(
            snapshot.responses_by_state[state], state=state
        )
        frames.append(frame)
        expected_requests.append({
            "state": state,
            "candidate_count": len(frame),
            "request_sha256": request_sha,
            "response_sha256": indexed_record["response_sha256"],
            "retrieved_at_utc": indexed_record["retrieved_at_utc"],
            "byte_count": indexed_record["byte_count"],
        })
    if requests != expected_requests:
        raise EvidenceError(
            "candidate provenance requests differ from canonical state order/replay"
        )
    rebuilt = merge_candidate_metadata(frames)
    if provenance.get("candidate_count") != len(rebuilt):
        raise EvidenceError("candidate total count changed")
    expected_candidate_payload = rebuilt.to_csv(
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    ).encode("utf-8")
    if candidate_payload != expected_candidate_payload:
        raise EvidenceError("candidate table is not exact canonical producer CSV")
    provided = pd.read_csv(
        io.BytesIO(candidate_payload),
        dtype={
            "site_no": "string", "station_nm": "string", "state": "string",
            "site_type": "string", "huc_cd": "string",
        },
        keep_default_na=False,
        float_precision="round_trip",
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
    return CandidateEvidenceAudit(
        candidates=provided,
        candidate_payload=candidate_payload,
        provenance_payload=provenance_payload,
        snapshot_index_payload=index_payload,
        provenance=provenance,
        snapshot=snapshot,
    )


def replay_candidate_evidence(
    candidates_path: str | Path,
    provenance_path: str | Path,
    snapshot_index_path: str | Path,
    *,
    protocol_sha256: str,
    state_universe: Iterable[str],
    trusted_root: str | Path | None = None,
) -> pd.DataFrame:
    """Compatibility wrapper returning the replayed candidate table."""
    return audit_candidate_evidence(
        candidates_path,
        provenance_path,
        snapshot_index_path,
        protocol_sha256=protocol_sha256,
        state_universe=state_universe,
        trusted_root=trusted_root,
    ).candidates

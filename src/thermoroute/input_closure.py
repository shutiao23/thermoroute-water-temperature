"""Closed, immutable inventory of the Route-A development inputs.

The development evidence chain has two roots: the frozen 2006--2020 panel
specification and the outcome-free predictor-product bridge.  This module walks
only their canonical development paths.  It deliberately has no generic
"follow every path in JSON" mechanism: a corrupted declaration must be rejected
before its target is opened, especially when confirmatory outcomes exist beside
the development tree.

The returned :class:`InputClosure` binds the raw bytes and repository-relative
path of every declared file.  File-system identity is retained separately so a
caller can detect a time-of-check/time-of-use change immediately before training
or publication.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any

from .evidence import FrozenPanelSpec
from .predictor_bridge import validate_development_bridge_manifest_offline
from .provenance import canonical_json_bytes


INPUT_CLOSURE_FORMAT = "thermoroute.development-input-closure.v1"
COMPOSED_INPUT_CLOSURE_FORMAT = "thermoroute.composed-input-closure.v1"

FROZEN_SPEC_PATH = PurePosixPath("data_usgs/frozen_panel_v1.json")
PANEL_PATH = PurePosixPath("data_usgs/panel_usgs_120v2.parquet")
REGISTRY_PATH = PurePosixPath("data_usgs/station_registry_v1.csv")
SOURCE_METADATA_PATH = PurePosixPath("data_usgs/stations_meta_120v2.csv")
HUC_SOURCE_PATH = PurePosixPath("data_usgs/huc_metadata_usgs_v1.csv")
HUC_PROVENANCE_PATH = PurePosixPath(
    "data_usgs/huc_metadata_usgs_v1.provenance.json"
)
HUC_INDEX_PATH = PurePosixPath(
    "data_usgs/raw_snapshots/huc-v1/snapshot_index.json"
)

BRIDGE_MANIFEST_PATH = PurePosixPath(
    "data_usgs/development_predictor_bridge_v1.json"
)
BRIDGE_FROZEN_PATH = PurePosixPath(
    "data_usgs/development_predictor_bridge_v1/"
    "frozen_panel_predictors_2018_2020.parquet"
)
BRIDGE_REFRESHED_PATH = PurePosixPath(
    "data_usgs/development_predictor_bridge_v1/"
    "refreshed_predictors_2018_2020.parquet"
)
BRIDGE_REPORT_PATH = PurePosixPath(
    "data_usgs/development_predictor_bridge_v1/bridge_report_v1.json"
)
BRIDGE_REQUEST_MAP_PATH = PurePosixPath(
    "data_usgs/development_predictor_bridge_v1/source_request_map_v1.json"
)
BRIDGE_RAW_INDEXES: tuple[tuple[str, PurePosixPath, str, int], ...] = (
    (
        "daymet",
        PurePosixPath(
            "data_usgs/raw_snapshots/development-predictor-bridge-v1/"
            "daymet-v1/snapshot_index_v2.json"
        ),
        "ornl-daymet-single-pixel-route-a",
        120,
    ),
    (
        "gridmet",
        PurePosixPath(
            "data_usgs/raw_snapshots/development-predictor-bridge-v1/"
            "gridmet-v1/snapshot_index_v2.json"
        ),
        "gridmet-ncss-route-a",
        120,
    ),
    (
        "gridmet_schema",
        PurePosixPath(
            "data_usgs/raw_snapshots/development-predictor-bridge-v1/"
            "gridmet-schema-v1/snapshot_index_v2.json"
        ),
        "gridmet-opendap-schema-route-a",
        1,
    ),
)

_HEX_DIGITS = frozenset("0123456789abcdef")
_SNAPSHOT_METADATA_FIELDS = {
    "schema_version",
    "request",
    "request_sha256",
    "retrieved_at_utc",
    "http_status",
    "response_headers",
    "byte_count",
    "response_sha256",
    "response_file",
}


class InputClosureError(RuntimeError):
    """The fixed Route-A development input closure is unsafe or inconsistent."""


@dataclass(frozen=True, slots=True)
class InputInventoryEntry:
    """One byte-level binding in the sorted development inventory."""

    path: str
    raw_sha256: str
    byte_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "raw_sha256": self.raw_sha256,
            "bytes": self.byte_count,
        }


@dataclass(frozen=True, slots=True)
class FileStatSnapshot:
    """File-system identity captured while the corresponding bytes were read."""

    path: str
    device: int
    inode: int
    mode: int
    link_count: int
    uid: int
    gid: int
    byte_count: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, path: str, value: os.stat_result) -> FileStatSnapshot:
        return cls(
            path=path,
            device=value.st_dev,
            inode=value.st_ino,
            mode=value.st_mode,
            link_count=value.st_nlink,
            uid=value.st_uid,
            gid=value.st_gid,
            byte_count=value.st_size,
            mtime_ns=value.st_mtime_ns,
            ctime_ns=value.st_ctime_ns,
        )


@dataclass(frozen=True, slots=True)
class InputClosure:
    """Resolved, immutable Route-A development input binding."""

    repo_root: Path
    inventory: tuple[InputInventoryEntry, ...]
    binding_digest: str
    stat_snapshot: tuple[FileStatSnapshot, ...]
    semantic_checks: tuple[str, ...]

    def binding_document(self) -> dict[str, object]:
        """Return a fresh serialisable document; the closure itself stays immutable."""
        return {
            "format": INPUT_CLOSURE_FORMAT,
            "binding_digest": self.binding_digest,
            "inventory": [entry.as_dict() for entry in self.inventory],
            "semantic_checks": list(self.semantic_checks),
        }

    def assert_unchanged(self) -> None:
        """Re-read every fixed path and fail on byte or file-identity drift."""
        if len(self.inventory) != len(self.stat_snapshot):
            raise InputClosureError("input closure inventory/stat snapshot diverged")
        refreshed: list[InputInventoryEntry] = []
        for expected, expected_stat in zip(
            self.inventory, self.stat_snapshot, strict=True
        ):
            if expected.path != expected_stat.path:
                raise InputClosureError("input closure inventory/stat paths diverged")
            payload, actual_stat = _read_regular_file(
                self.repo_root, PurePosixPath(expected.path)
            )
            actual = InputInventoryEntry(
                path=expected.path,
                raw_sha256=hashlib.sha256(payload).hexdigest(),
                byte_count=len(payload),
            )
            if actual != expected:
                raise InputClosureError(
                    f"development input bytes changed after resolution: {expected.path}"
                )
            if actual_stat != expected_stat:
                raise InputClosureError(
                    "development input stat identity changed after resolution: "
                    f"{expected.path}"
                )
            refreshed.append(actual)
        if _binding_digest(tuple(refreshed)) != self.binding_digest:
            raise InputClosureError("development input binding digest changed")


FrozenSpecVerifier = Callable[[Path], object]
BridgeVerifier = Callable[..., object]


def _normalise_relative_path(value: object, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise InputClosureError(f"{label} path is malformed")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise InputClosureError(f"{label} path escapes or is non-canonical")
    return path


def _fixed_path(value: object, expected: PurePosixPath, *, label: str) -> None:
    path = _normalise_relative_path(value, label=label)
    if path != expected:
        # This check happens before any file operation on the declaration.
        raise InputClosureError(
            f"{label} is outside the fixed development closure: {path.as_posix()}"
        )


def _fixed_spec_child(value: object, expected: PurePosixPath, *, label: str) -> None:
    child = _normalise_relative_path(value, label=label)
    declared = FROZEN_SPEC_PATH.parent / child
    if declared != expected:
        raise InputClosureError(
            f"{label} is outside the fixed development closure: {declared.as_posix()}"
        )


def _require_digest(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise InputClosureError(f"{label} is not a lowercase SHA-256")
    return value


def compose_input_closure_digest(
    components: Mapping[str, str],
) -> str:
    """Compose named component digests into one canonical input-closure digest.

    Names are part of the commitment, so exchanging two semantically different
    inputs with identical container positions cannot preserve the digest.  The
    sorted list representation makes the result independent of mapping insertion
    order while retaining an exact, independently reproducible composition.
    """
    if not isinstance(components, Mapping) or not components:
        raise InputClosureError("input-closure composition must not be empty")
    records: list[dict[str, str]] = []
    for name, digest in components.items():
        if (
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or any(character in name for character in ("\\", "\x00"))
        ):
            raise InputClosureError("input-closure component name is malformed")
        records.append({
            "name": name,
            "sha256": _require_digest(
                digest, label=f"input-closure component {name!r}"
            ),
        })
    records.sort(key=lambda record: record["name"])
    if len({record["name"] for record in records}) != len(records):
        raise InputClosureError("input-closure component names are duplicated")
    document = {
        "format": COMPOSED_INPUT_CLOSURE_FORMAT,
        "components": records,
    }
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def _json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputClosureError(f"{label} is not a UTF-8 JSON object") from exc
    if not isinstance(document, dict):
        raise InputClosureError(f"{label} is not a JSON object")
    return document


def _same_stat(left: os.stat_result, right: os.stat_result) -> bool:
    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_uid",
        "st_gid",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    return all(getattr(left, field) == getattr(right, field) for field in fields)


def _read_regular_file(
    root: Path, relative: PurePosixPath
) -> tuple[bytes, FileStatSnapshot]:
    """Read through no-follow directory descriptors and reject special/linked files."""
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise InputClosureError("internal input-closure path is not relative")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | os.O_CLOEXEC | nofollow | getattr(os, "O_NONBLOCK", 0)
    descriptors: list[int] = []
    try:
        current = os.open(root, directory_flags | nofollow)
        descriptors.append(current)
        for component in relative.parts[:-1]:
            current = os.open(
                component,
                directory_flags | nofollow,
                dir_fd=current,
            )
            descriptors.append(current)
        handle = os.open(relative.name, file_flags, dir_fd=current)
        descriptors.append(handle)
        before = os.fstat(handle)
        if not stat.S_ISREG(before.st_mode):
            raise InputClosureError(
                f"development input is not a regular file: {relative.as_posix()}"
            )
        if before.st_nlink != 1:
            raise InputClosureError(
                f"development input has a hard-link alias: {relative.as_posix()}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(handle, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(handle)
        if not _same_stat(before, after):
            raise InputClosureError(
                f"development input changed while being read: {relative.as_posix()}"
            )
        return b"".join(chunks), FileStatSnapshot.from_stat(
            relative.as_posix(), after
        )
    except InputClosureError:
        raise
    except OSError as exc:
        raise InputClosureError(
            f"development input is absent, linked, or unsafe: {relative.as_posix()}"
        ) from exc
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


class _Collector:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.entries: dict[str, InputInventoryEntry] = {}
        self.stats: dict[str, FileStatSnapshot] = {}

    def add(
        self,
        path: PurePosixPath,
        *,
        expected_sha256: object | None = None,
        label: str,
        allow_shared: bool = False,
    ) -> bytes:
        relative = path.as_posix()
        expected = (
            _require_digest(expected_sha256, label=f"{label} checksum")
            if expected_sha256 is not None
            else None
        )
        if relative in self.entries:
            if not allow_shared:
                raise InputClosureError(
                    f"duplicate development input declaration: {relative}"
                )
            entry = self.entries[relative]
            if expected is not None and entry.raw_sha256 != expected:
                raise InputClosureError(
                    f"shared development input checksum disagrees: {relative}"
                )
            payload, snapshot = _read_regular_file(self.root, path)
            if (
                hashlib.sha256(payload).hexdigest() != entry.raw_sha256
                or len(payload) != entry.byte_count
                or snapshot != self.stats[relative]
            ):
                raise InputClosureError(
                    f"shared development input changed during closure: {relative}"
                )
            return payload
        payload, snapshot = _read_regular_file(self.root, path)
        digest = hashlib.sha256(payload).hexdigest()
        if expected is not None and digest != expected:
            raise InputClosureError(
                f"{label} raw checksum mismatch: expected {expected}, got {digest}"
            )
        self.entries[relative] = InputInventoryEntry(
            path=relative, raw_sha256=digest, byte_count=len(payload)
        )
        self.stats[relative] = snapshot
        return payload

    def freeze(self) -> InputClosure:
        paths = sorted(self.entries)
        inventory = tuple(self.entries[path] for path in paths)
        snapshots = tuple(self.stats[path] for path in paths)
        return InputClosure(
            repo_root=self.root,
            inventory=inventory,
            binding_digest=_binding_digest(inventory),
            stat_snapshot=snapshots,
            semantic_checks=(
                "FrozenPanelSpec.verify",
                "validate_development_bridge_manifest_offline",
            ),
        )


def _binding_digest(inventory: tuple[InputInventoryEntry, ...]) -> str:
    document = {
        "format": INPUT_CLOSURE_FORMAT,
        "inventory": [entry.as_dict() for entry in inventory],
    }
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def _binding(
    value: object,
    *,
    expected_path: PurePosixPath,
    label: str,
) -> tuple[PurePosixPath, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise InputClosureError(f"{label} binding is malformed")
    _fixed_path(value.get("path"), expected_path, label=label)
    return expected_path, _require_digest(value.get("sha256"), label=f"{label} checksum")


def _verify_snapshot_metadata(
    metadata: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    request_sha256: str,
    response_sha256: str,
    retrieved_at_utc: object,
    byte_count: int,
) -> None:
    headers = metadata.get("response_headers")
    if (
        set(metadata) != _SNAPSHOT_METADATA_FIELDS
        or metadata.get("schema_version") != 1
        or metadata.get("request") != request
        or metadata.get("request_sha256") != request_sha256
        or metadata.get("retrieved_at_utc") != retrieved_at_utc
        or metadata.get("http_status") != 200
        or not isinstance(headers, Mapping)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in headers.items()
        )
        or metadata.get("byte_count") != byte_count
        or metadata.get("response_sha256") != response_sha256
        or metadata.get("response_file") != "response.bin"
    ):
        raise InputClosureError("raw snapshot metadata/index/response disagree")


def _collect_snapshot_index(
    collector: _Collector,
    *,
    index_path: PurePosixPath,
    index_payload: bytes,
    schema_version: int,
    expected_provider: str,
    expected_count: int,
    label: str,
) -> tuple[dict[str, Any], ...]:
    index = _json_object(index_payload, label=label)
    records = index.get("records")
    if (
        set(index) != {"schema_version", "snapshot_count", "records"}
        or index.get("schema_version") != schema_version
        or type(index.get("snapshot_count")) is not int
        or index.get("snapshot_count") != expected_count
        or not isinstance(records, list)
        or len(records) != expected_count
    ):
        raise InputClosureError(f"{label} contract changed")
    base_fields = {
        "provider",
        "request_sha256",
        "response_sha256",
        "retrieved_at_utc",
        "byte_count",
        "request",
        "metadata_path",
        "response_path",
    }
    expected_fields = (
        base_fields | {"metadata_sha256", "metadata_byte_count"}
        if schema_version == 2
        else base_fields
    )
    seen_requests: set[str] = set()
    seen_declared_paths: set[str] = set()
    parsed: list[dict[str, Any]] = []
    for position, raw_record in enumerate(records):
        record_label = f"{label} record {position}"
        if not isinstance(raw_record, Mapping) or set(raw_record) != expected_fields:
            raise InputClosureError(f"{record_label} contract changed")
        record = dict(raw_record)
        provider = record.get("provider")
        request = record.get("request")
        request_sha = _require_digest(
            record.get("request_sha256"), label=f"{record_label} request checksum"
        )
        response_sha = _require_digest(
            record.get("response_sha256"), label=f"{record_label} response checksum"
        )
        byte_count = record.get("byte_count")
        if (
            provider != expected_provider
            or not isinstance(request, Mapping)
            or set(request) != {"schema_version", "provider", "method", "url", "headers"}
            or request.get("schema_version") != 1
            or request.get("provider") != expected_provider
            or request.get("method") != "GET"
            or not isinstance(request.get("url"), str)
            or not request.get("url")
            or not isinstance(request.get("headers"), Mapping)
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in request["headers"].items()
            )
            or hashlib.sha256(canonical_json_bytes(dict(request))).hexdigest()
            != request_sha
            or request_sha in seen_requests
            or type(byte_count) is not int
            or byte_count < 1
            or not isinstance(record.get("retrieved_at_utc"), str)
            or not record["retrieved_at_utc"]
        ):
            raise InputClosureError(f"{record_label} identity is malformed")
        seen_requests.add(request_sha)
        expected_base = PurePosixPath(expected_provider) / request_sha
        metadata_relative = _normalise_relative_path(
            record.get("metadata_path"), label=f"{record_label} metadata"
        )
        response_relative = _normalise_relative_path(
            record.get("response_path"), label=f"{record_label} response"
        )
        if (
            metadata_relative != expected_base / "metadata.json"
            or response_relative != expected_base / "response.bin"
        ):
            raise InputClosureError(f"{record_label} paths are not content addressed")
        for declared in (metadata_relative, response_relative):
            declared_text = declared.as_posix()
            if declared_text in seen_declared_paths:
                raise InputClosureError(
                    f"duplicate raw snapshot path declaration: {declared_text}"
                )
            seen_declared_paths.add(declared_text)
        metadata_path = index_path.parent / metadata_relative
        response_path = index_path.parent / response_relative
        metadata_payload = collector.add(
            metadata_path,
            expected_sha256=(
                record.get("metadata_sha256") if schema_version == 2 else None
            ),
            label=f"{record_label} metadata",
        )
        if (
            schema_version == 2
            and record.get("metadata_byte_count") != len(metadata_payload)
        ):
            raise InputClosureError(f"{record_label} metadata byte count changed")
        response_payload = collector.add(
            response_path,
            expected_sha256=response_sha,
            label=f"{record_label} response",
        )
        if len(response_payload) != byte_count:
            raise InputClosureError(f"{record_label} response byte count changed")
        metadata = _json_object(metadata_payload, label=f"{record_label} metadata")
        _verify_snapshot_metadata(
            metadata,
            request=request,
            request_sha256=request_sha,
            response_sha256=response_sha,
            retrieved_at_utc=record.get("retrieved_at_utc"),
            byte_count=byte_count,
        )
        parsed.append(record)
    return tuple(parsed)


def _collect_frozen_panel(collector: _Collector) -> int:
    spec_payload = collector.add(FROZEN_SPEC_PATH, label="frozen panel spec")
    spec = _json_object(spec_payload, label="frozen panel spec")
    panel = spec.get("panel")
    registry = spec.get("station_registry")
    if (
        spec.get("schema_version") != 1
        or spec.get("evidence_role") != "development_exploratory"
        or not isinstance(panel, Mapping)
        or not isinstance(registry, Mapping)
        or panel.get("date_end") != "2020-12-31"
        or panel.get("date_start") != "2006-01-01"
        or type(panel.get("station_count")) is not int
        or panel.get("station_count") != 120
    ):
        raise InputClosureError("frozen panel is not the fixed Route-A development cohort")
    _fixed_spec_child(panel.get("path"), PANEL_PATH, label="frozen panel")
    panel_sha = _require_digest(panel.get("sha256"), label="frozen panel checksum")
    collector.add(
        PANEL_PATH, expected_sha256=panel_sha, label="frozen development panel"
    )
    _fixed_spec_child(registry.get("path"), REGISTRY_PATH, label="station registry")
    registry_sha = _require_digest(
        registry.get("sha256"), label="station registry checksum"
    )
    collector.add(
        REGISTRY_PATH, expected_sha256=registry_sha, label="station registry"
    )
    _fixed_spec_child(
        registry.get("source_metadata_path"),
        SOURCE_METADATA_PATH,
        label="station source metadata",
    )
    metadata_sha = _require_digest(
        registry.get("source_metadata_sha256"),
        label="station source metadata checksum",
    )
    collector.add(
        SOURCE_METADATA_PATH,
        expected_sha256=metadata_sha,
        label="station source metadata",
    )
    huc = registry.get("huc_metadata")
    if (
        not isinstance(huc, Mapping)
        or huc.get("runtime_dependency") is not True
        or huc.get("status") != "COMPLETE_USGS_RAW_SNAPSHOT"
        or huc.get("join_key") != "site_no"
    ):
        raise InputClosureError("frozen HUC metadata closure is incomplete")
    _fixed_spec_child(huc.get("source_path"), HUC_SOURCE_PATH, label="HUC source")
    huc_source_sha = _require_digest(
        huc.get("source_sha256"), label="HUC source checksum"
    )
    collector.add(
        HUC_SOURCE_PATH, expected_sha256=huc_source_sha, label="HUC source"
    )
    _fixed_spec_child(
        huc.get("provenance_path"), HUC_PROVENANCE_PATH, label="HUC provenance"
    )
    huc_provenance_sha = _require_digest(
        huc.get("provenance_sha256"), label="HUC provenance checksum"
    )
    provenance_payload = collector.add(
        HUC_PROVENANCE_PATH,
        expected_sha256=huc_provenance_sha,
        label="HUC provenance",
    )
    provenance = _json_object(provenance_payload, label="HUC provenance")
    if (
        provenance.get("schema_version") != 1
        or provenance.get("outcome_data_requested") is not False
        or provenance.get("join_key") != "site_no"
        or provenance.get("site_count") != 120
        or provenance.get("development_panel_sha256") != panel_sha
        or provenance.get("development_metadata_sha256") != metadata_sha
        or provenance.get("derived_csv_sha256") != huc_source_sha
    ):
        raise InputClosureError("HUC provenance is not bound to the development cohort")
    _fixed_path(
        provenance.get("raw_snapshot_index"), HUC_INDEX_PATH, label="HUC raw index"
    )
    index_payload = collector.add(
        HUC_INDEX_PATH,
        expected_sha256=provenance.get("raw_snapshot_index_sha256"),
        label="HUC raw index",
    )
    records = _collect_snapshot_index(
        collector,
        index_path=HUC_INDEX_PATH,
        index_payload=index_payload,
        schema_version=1,
        expected_provider="usgs-nwis-site-metadata",
        expected_count=1,
        label="HUC raw snapshot index",
    )
    record = records[0]
    if (
        provenance.get("request_sha256") != record["request_sha256"]
        or provenance.get("response_sha256") != record["response_sha256"]
        or provenance.get("retrieved_at_utc") != record["retrieved_at_utc"]
    ):
        raise InputClosureError("HUC provenance/raw snapshot identity changed")
    return 120


def _collect_predictor_bridge(collector: _Collector) -> None:
    manifest_payload = collector.add(
        BRIDGE_MANIFEST_PATH, label="development predictor bridge manifest"
    )
    manifest = _json_object(
        manifest_payload, label="development predictor bridge manifest"
    )
    if (
        manifest.get("format") != "thermoroute.development-predictor-bridge.v1"
        or manifest.get("status") != "PASS_EXACT_PRODUCT_BRIDGE"
        or manifest.get("outcome_values_requested_or_read") is not False
    ):
        raise InputClosureError("development predictor bridge is not an outcome-free PASS")
    panel_path, panel_sha = _binding(
        manifest.get("panel"), expected_path=PANEL_PATH, label="bridge panel"
    )
    collector.add(
        panel_path,
        expected_sha256=panel_sha,
        label="bridge panel",
        allow_shared=True,
    )
    registry_path, registry_sha = _binding(
        manifest.get("registry"),
        expected_path=REGISTRY_PATH,
        label="bridge registry",
    )
    collector.add(
        registry_path,
        expected_sha256=registry_sha,
        label="bridge registry",
        allow_shared=True,
    )
    normalized = manifest.get("normalized")
    if not isinstance(normalized, Mapping) or set(normalized) != {"frozen", "refreshed"}:
        raise InputClosureError("bridge normalized registry changed")
    for name, expected_path in (
        ("frozen", BRIDGE_FROZEN_PATH),
        ("refreshed", BRIDGE_REFRESHED_PATH),
    ):
        path, digest = _binding(
            normalized[name], expected_path=expected_path, label=f"bridge {name} table"
        )
        collector.add(path, expected_sha256=digest, label=f"bridge {name} table")
    for field, expected_path in (
        ("report", BRIDGE_REPORT_PATH),
        ("request_map", BRIDGE_REQUEST_MAP_PATH),
    ):
        path, digest = _binding(
            manifest.get(field), expected_path=expected_path, label=f"bridge {field}"
        )
        collector.add(path, expected_sha256=digest, label=f"bridge {field}")
    raw = manifest.get("raw_snapshot_indexes")
    if not isinstance(raw, Mapping) or set(raw) != {
        "daymet",
        "gridmet",
        "gridmet_schema",
    }:
        raise InputClosureError("bridge raw snapshot registry changed")
    for name, expected_path, provider, count in BRIDGE_RAW_INDEXES:
        path, digest = _binding(
            raw[name], expected_path=expected_path, label=f"bridge raw/{name}"
        )
        payload = collector.add(
            path, expected_sha256=digest, label=f"bridge raw/{name}"
        )
        _collect_snapshot_index(
            collector,
            index_path=path,
            index_payload=payload,
            schema_version=2,
            expected_provider=provider,
            expected_count=count,
            label=f"bridge raw/{name} index",
        )


def resolve_development_input_closure(
    repo_root: str | Path,
    *,
    frozen_spec_verifier: FrozenSpecVerifier | None = None,
    bridge_verifier: BridgeVerifier | None = None,
) -> InputClosure:
    """Resolve and semantically verify the fixed Route-A development closure.

    The optional callables are dependency-injection seams for isolated tests.
    Production callers should omit them so :meth:`FrozenPanelSpec.verify` and
    :func:`validate_development_bridge_manifest_offline` are used directly.
    """
    supplied_root = Path(repo_root)
    if supplied_root.is_symlink():
        raise InputClosureError("repository root must not be a symlink")
    try:
        root = supplied_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise InputClosureError("repository root is absent or invalid") from exc
    if not root.is_dir():
        raise InputClosureError("repository root is not a directory")

    collector = _Collector(root)
    expected_sites = _collect_frozen_panel(collector)
    _collect_predictor_bridge(collector)
    closure = collector.freeze()

    verify_frozen = frozen_spec_verifier
    if verify_frozen is None:
        def verify_frozen(path: Path) -> object:
            return FrozenPanelSpec.load(path).verify()
    verify_bridge = bridge_verifier or validate_development_bridge_manifest_offline
    try:
        frozen_result = verify_frozen(root / FROZEN_SPEC_PATH)
        bridge_result = verify_bridge(
            repo_root=root,
            manifest_path=root / BRIDGE_MANIFEST_PATH,
            expected_sites=expected_sites,
        )
    except Exception as exc:
        raise InputClosureError("development input semantic verification failed") from exc
    if (
        isinstance(frozen_result, Mapping)
        and frozen_result.get("evidence_role") != "development_exploratory"
    ):
        raise InputClosureError("FrozenPanelSpec.verify returned the wrong evidence role")
    if (
        isinstance(bridge_result, Mapping)
        and bridge_result.get("status") != "PASS_EXACT_PRODUCT_BRIDGE"
    ):
        raise InputClosureError("bridge offline validator did not return an exact PASS")
    closure.assert_unchanged()
    return closure


# Short public alias for callers that already operate inside a development-only stage.
resolve_input_closure = resolve_development_input_closure

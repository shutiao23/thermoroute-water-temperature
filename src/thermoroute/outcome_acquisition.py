"""Raw-only, create-only NWIS acquisition for the Route-A opening.

The first process freezes the complete request ledger before its first HTTPS
request.  A later process may continue the *same* irreversible opening, but it
can only reuse byte-verified complete transactions and fetch ledger entries
whose canonical transaction directory is wholly absent.  No outcome is parsed
until every raw transaction is complete.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence
import urllib.request

import pandas as pd

from .opening_contract import (
    ACQUISITION_ATTEMPT_INDEX_FORMAT,
    ACQUISITION_ATTEMPT_RESULT_FORMAT,
    ACQUISITION_ATTEMPT_START_FORMAT,
    ACQUISITION_MANIFEST_FORMAT,
    ACQUISITION_REQUEST_LEDGER_FORMAT,
    ACQUISITION_REQUEST_MAP_FORMAT,
    MAX_CONFIRMATORY_NWIS_RESPONSE_BYTES,
    assert_no_symlink_components,
    validate_acquisition_work_order,
    validate_frozen_source_identity,
)
from .provenance import canonical_json_bytes, sha256_bytes, sha256_file
from .usgs import (
    CONFIRMATORY_NWIS_PROVIDER,
    build_nwis_confirmatory_url,
    nwis_confirmatory_series_registry,
    parse_nwis_confirmatory_daily,
)


class OutcomeAcquisitionError(RuntimeError):
    """The fixed raw acquisition could not satisfy its immutable contract."""


_RESPONSE_CHUNK_BYTES = 1024 * 1024
_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


@contextmanager
def _open_directory_chain(path: Path, *, create: bool) -> Iterator[int]:
    """Open an absolute directory component-by-component without symlinks."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    if not absolute.is_absolute():
        raise OutcomeAcquisitionError("secure directory path is not absolute")
    descriptor = os.open(os.path.sep, _DIRECTORY_OPEN_FLAGS)
    try:
        for component in absolute.parts[1:]:
            try:
                child = os.open(
                    component,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, 0o755, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(
                    component,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=descriptor,
                )
            os.close(descriptor)
            descriptor = child
        yield descriptor
    except OSError as exc:
        raise OutcomeAcquisitionError(
            f"secure directory traversal failed: {absolute}"
        ) from exc
    finally:
        os.close(descriptor)


def _secure_mkdirs(path: Path) -> None:
    with _open_directory_chain(path, create=True) as descriptor:
        os.fsync(descriptor)


def _secure_create_directory(path: Path) -> None:
    with _open_directory_chain(path.parent, create=True) as parent_descriptor:
        try:
            os.mkdir(path.name, 0o755, dir_fd=parent_descriptor)
        except FileExistsError as exc:
            raise OutcomeAcquisitionError(
                f"refusing to replace immutable directory: {path}"
            ) from exc
        child = os.open(
            path.name,
            _DIRECTORY_OPEN_FLAGS,
            dir_fd=parent_descriptor,
        )
        try:
            os.fsync(child)
            os.fsync(parent_descriptor)
        finally:
            os.close(child)


def _read_regular_file_no_follow(path: Path) -> bytes:
    with _open_directory_chain(path.parent, create=False) as parent_descriptor:
        descriptor = os.open(
            path.name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise OutcomeAcquisitionError(
                    f"acquisition artifact is not a regular file: {path}"
                )
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                return handle.read()
        finally:
            os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    with _open_directory_chain(path, create=False) as descriptor:
        os.fsync(descriptor)


def _create_bytes(path: Path, payload: bytes) -> None:
    path = Path(os.path.abspath(os.fspath(path)))
    with _open_directory_chain(path.parent, create=True) as parent_descriptor:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            descriptor = os.open(
                path.name,
                flags,
                0o444,
                dir_fd=parent_descriptor,
            )
        except FileExistsError as exc:
            raise OutcomeAcquisitionError(
                f"refusing to replace immutable artifact: {path}"
            ) from exc
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.fsync(parent_descriptor)


def _create_or_validate_bytes(path: Path, payload: bytes, *, label: str) -> None:
    if path.exists():
        if (
            not path.is_file()
            or path.is_symlink()
            or _read_regular_file_no_follow(path) != payload
        ):
            raise OutcomeAcquisitionError(f"existing {label} differs from replay")
        return
    _create_bytes(path, payload)


def _create_parquet(path: Path, frame: pd.DataFrame) -> None:
    path = Path(os.path.abspath(os.fspath(path)))
    with _open_directory_chain(path.parent, create=True) as parent_descriptor:
        try:
            os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise OutcomeAcquisitionError(
                f"refusing to replace immutable artifact: {path}"
            )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        temporary_name_only = temporary.name
        os.close(descriptor)
        try:
            frame.to_parquet(temporary, index=False)
            temporary_descriptor = os.open(
                temporary_name_only,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_descriptor,
            )
            try:
                if not stat.S_ISREG(os.fstat(temporary_descriptor).st_mode):
                    raise OutcomeAcquisitionError(
                        "temporary parquet output is not a regular file"
                    )
                os.fsync(temporary_descriptor)
                os.fchmod(temporary_descriptor, 0o444)
            finally:
                os.close(temporary_descriptor)
            try:
                os.link(
                    temporary_name_only,
                    path.name,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError as exc:
                raise OutcomeAcquisitionError(
                    f"refusing to replace immutable artifact: {path}"
                ) from exc
            os.fsync(parent_descriptor)
        finally:
            try:
                os.unlink(temporary_name_only, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(_read_regular_file_no_follow(path).decode("utf-8"))
    except (
        FileNotFoundError,
        UnicodeError,
        json.JSONDecodeError,
        OSError,
    ) as exc:
        raise OutcomeAcquisitionError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise OutcomeAcquisitionError(f"{label} is not a JSON object")
    return value


def _self_hashed_document(
    path: Path, *, label: str, self_field: str, format_name: str
) -> dict[str, Any]:
    document = _read_json(path, label=label)
    stable = dict(document)
    claimed = stable.pop(self_field, None)
    if (
        document.get("format") != format_name
        or claimed != sha256_bytes(canonical_json_bytes(stable))
        or _read_regular_file_no_follow(path) != canonical_json_bytes(document)
    ):
        raise OutcomeAcquisitionError(f"{label} identity or self hash changed")
    return document


def _binding(root: Path, path: Path) -> dict[str, str]:
    try:
        resolved = assert_no_symlink_components(
            root,
            Path(os.path.abspath(os.fspath(path))),
            require_file=True,
        )
    except Exception as exc:
        raise OutcomeAcquisitionError("cannot bind acquisition artifact") from exc
    return {
        "path": resolved.relative_to(root).as_posix(),
        "sha256": sha256_file(resolved),
    }


def _evidence_paths(state: Mapping[str, Path]) -> dict[str, Path]:
    acquisition_root = state["raw_nwis_root"].parent
    return {
        "root": acquisition_root,
        "request_ledger": acquisition_root / "request_ledger_v1.json",
        "attempts_root": acquisition_root / "transport_attempts_v1",
        "attempt_index": acquisition_root / "transport_attempt_index_v1.json",
        "lock": acquisition_root / ".transport_resume.lock",
    }


@contextmanager
def _exclusive_transport_lock(path: Path) -> Iterator[None]:
    path = Path(os.path.abspath(os.fspath(path)))
    with _open_directory_chain(path.parent, create=True) as parent_descriptor:
        descriptor = os.open(
            path.name,
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise OutcomeAcquisitionError(
                    "transport lock is not a regular file"
                )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise OutcomeAcquisitionError(
                    "another process is already continuing this opening"
                ) from exc
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def _request_specs(
    work_order: Mapping[str, Any], authorization: Mapping[str, Any]
) -> list[dict[str, Any]]:
    plan = authorization["acquisition_plan"]
    history_start = str(plan["history_start"])
    target_end = str(plan["target_end"])
    rows: list[dict[str, Any]] = []
    ordinal = 0
    for cohort in ("temporal", "external"):
        for site_no in work_order["site_registries"][cohort]["sites"]:
            ordinal += 1
            request = {
                "schema_version": 1,
                "provider": CONFIRMATORY_NWIS_PROVIDER,
                "method": "GET",
                "url": build_nwis_confirmatory_url(
                    site_no, history_start, target_end
                ),
                "headers": {},
            }
            rows.append({
                "ordinal": ordinal,
                "cohort": cohort,
                "site_no": str(site_no),
                "request": request,
                "request_sha256": sha256_bytes(canonical_json_bytes(request)),
            })
    return rows


def _expected_request_ledger(
    *,
    work_order: Mapping[str, Any],
    authorization: Mapping[str, Any],
    work_order_path: Path,
) -> dict[str, Any]:
    stable = {
        "format": ACQUISITION_REQUEST_LEDGER_FORMAT,
        "status": "FROZEN_BEFORE_FIRST_HTTPS_REQUEST",
        "opening_id": authorization["opening_id"],
        "authorization_sha256": work_order["authorization_sha256"],
        "work_order_self_sha256": work_order["work_order_self_sha256"],
        "work_order_file_sha256": sha256_file(work_order_path),
        "provider": CONFIRMATORY_NWIS_PROVIDER,
        "maximum_response_bytes_per_request": (
            MAX_CONFIRMATORY_NWIS_RESPONSE_BYTES
        ),
        "request_order": "temporal_then_external_each_site_no_ascending",
        "request_count": sum(
            len(work_order["site_registries"][cohort]["sites"])
            for cohort in ("temporal", "external")
        ),
        "requests": _request_specs(work_order, authorization),
        "station_or_request_replacement_allowed": False,
    }
    return {
        **stable,
        "request_ledger_self_sha256": sha256_bytes(canonical_json_bytes(stable)),
    }


def _load_or_create_request_ledger(
    *,
    path: Path,
    expected: Mapping[str, Any],
    raw_root: Path,
    resume: bool,
) -> dict[str, Any]:
    if path.exists():
        if not resume:
            raise OutcomeAcquisitionError(
                "initial acquisition refuses a preexisting request ledger"
            )
        actual = _self_hashed_document(
            path,
            label="acquisition request ledger",
            self_field="request_ledger_self_sha256",
            format_name=ACQUISITION_REQUEST_LEDGER_FORMAT,
        )
        if actual != dict(expected):
            raise OutcomeAcquisitionError(
                "acquisition request ledger differs from the fixed work order"
            )
        return actual
    if raw_root.exists() and any(raw_root.iterdir()):
        raise OutcomeAcquisitionError(
            "raw acquisition evidence exists without its frozen request ledger"
        )
    _create_bytes(path, canonical_json_bytes(dict(expected)))
    return dict(expected)


def _attempt_number(path: Path, suffix: str) -> int:
    prefix = "attempt_"
    if not path.name.startswith(prefix) or not path.name.endswith(suffix):
        raise OutcomeAcquisitionError("transport attempt filename is noncanonical")
    value = path.name[len(prefix):-len(suffix)]
    if len(value) != 6 or not value.isdigit() or int(value) < 1:
        raise OutcomeAcquisitionError("transport attempt number is noncanonical")
    return int(value)


def _load_attempt_history(
    *,
    attempts_root: Path,
    opening_id: str,
    authorization_sha256: str,
    work_order_self_sha256: str,
    request_ledger_sha256: str,
    request_ids: set[str],
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    if not attempts_root.exists():
        return {}, {}
    if not attempts_root.is_dir() or attempts_root.is_symlink():
        raise OutcomeAcquisitionError("transport-attempt root is malformed")
    starts: dict[int, dict[str, Any]] = {}
    results: dict[int, dict[str, Any]] = {}
    for path in sorted(attempts_root.iterdir()):
        if not path.is_file() or path.is_symlink():
            raise OutcomeAcquisitionError("transport-attempt evidence contains extras")
        if path.name.endswith("_start.json"):
            number = _attempt_number(path, "_start.json")
            document = _self_hashed_document(
                path,
                label="transport attempt start",
                self_field="attempt_start_self_sha256",
                format_name=ACQUISITION_ATTEMPT_START_FORMAT,
            )
            target = starts
        elif path.name.endswith("_result.json"):
            number = _attempt_number(path, "_result.json")
            document = _self_hashed_document(
                path,
                label="transport attempt result",
                self_field="attempt_result_self_sha256",
                format_name=ACQUISITION_ATTEMPT_RESULT_FORMAT,
            )
            target = results
        else:
            raise OutcomeAcquisitionError("transport-attempt evidence contains extras")
        if number in target or document.get("attempt_number") != number:
            raise OutcomeAcquisitionError("transport-attempt evidence is duplicated")
        if path.stat().st_mode & 0o222:
            raise OutcomeAcquisitionError("transport-attempt evidence is mutable")
        common = {
            "opening_id": opening_id,
            "authorization_sha256": authorization_sha256,
            "work_order_self_sha256": work_order_self_sha256,
            "request_ledger_sha256": request_ledger_sha256,
            "opening_count": 1,
        }
        if any(document.get(key) != value for key, value in common.items()):
            raise OutcomeAcquisitionError("transport-attempt identity changed")
        target[number] = document
    if sorted(starts) != list(range(1, len(starts) + 1)):
        raise OutcomeAcquisitionError("transport-attempt starts are not contiguous")
    if not set(results) <= set(starts):
        raise OutcomeAcquisitionError("transport result lacks its start evidence")
    for number, start in starts.items():
        start_fields = {
            "format", "status", "opening_id", "authorization_sha256",
            "work_order_self_sha256", "request_ledger_sha256",
            "attempt_number", "mode", "opening_count",
            "completed_before_attempt_request_sha256",
            "missing_at_start_request_sha256", "response_replacement_allowed",
            "started_at_utc", "attempt_start_self_sha256",
        }
        completed = start.get("completed_before_attempt_request_sha256")
        missing = start.get("missing_at_start_request_sha256")
        if (
            set(start) != start_fields
            or start.get("status") != "TRANSPORT_ATTEMPT_STARTED"
            or start.get("response_replacement_allowed") is not False
            or not isinstance(completed, list)
            or not isinstance(missing, list)
            or len(completed) != len(set(completed))
            or len(missing) != len(set(missing))
            or set(completed) & set(missing)
            or set(completed) | set(missing) != request_ids
        ):
            raise OutcomeAcquisitionError(
                "transport-attempt start does not partition the request ledger"
            )
        _validate_timestamp(start.get("started_at_utc"))
        mode = start.get("mode")
        if mode not in {"INITIAL_OPENING_TRANSPORT", "RESUME_SAME_OPENING"}:
            raise OutcomeAcquisitionError("transport-attempt mode changed")
        if number > 1 and mode != "RESUME_SAME_OPENING":
            raise OutcomeAcquisitionError("later transport attempt is not a resume")
        result = results.get(number)
        if result is None:
            continue
        result_fields = {
            "format", "status", "opening_id", "authorization_sha256",
            "work_order_self_sha256", "request_ledger_sha256",
            "attempt_number", "attempt_start_sha256", "opening_count",
            "completed_request_sha256", "missing_request_sha256",
            "failure_class", "response_replacement_count", "completed_at_utc",
            "attempt_result_self_sha256",
        }
        if result.get("attempt_start_sha256") != sha256_file(
            attempts_root / f"attempt_{number:06d}_start.json"
        ) or set(result) != result_fields or result.get(
            "response_replacement_count"
        ) != 0:
            raise OutcomeAcquisitionError("transport result does not bind its start")
        final_completed = result.get("completed_request_sha256")
        final_missing = result.get("missing_request_sha256")
        if (
            not isinstance(final_completed, list)
            or not isinstance(final_missing, list)
            or len(final_completed) != len(set(final_completed))
            or len(final_missing) != len(set(final_missing))
            or set(final_completed) & set(final_missing)
            or set(final_completed) | set(final_missing) != request_ids
        ):
            raise OutcomeAcquisitionError(
                "transport-attempt result does not partition the request ledger"
            )
        status = result.get("status")
        if status == "ALL_LEDGER_TRANSACTIONS_COMPLETE":
            if final_missing or result.get("failure_class") is not None:
                raise OutcomeAcquisitionError(
                    "completed transport result still reports a failure"
                )
        elif status == "TRANSPORT_INCOMPLETE_MAY_RESUME_SAME_OPENING":
            if not final_missing or not isinstance(result.get("failure_class"), str):
                raise OutcomeAcquisitionError(
                    "incomplete transport result lacks a failure/missing request"
                )
        else:
            raise OutcomeAcquisitionError("transport-attempt result status changed")
        _validate_timestamp(result.get("completed_at_utc"))
    return starts, results


def _write_attempt_start(
    *,
    attempts_root: Path,
    attempt_number: int,
    resume: bool,
    work_order: Mapping[str, Any],
    request_ledger_sha256: str,
    completed: Iterable[str],
    missing: Iterable[str],
) -> tuple[Path, dict[str, Any]]:
    stable = {
        "format": ACQUISITION_ATTEMPT_START_FORMAT,
        "status": "TRANSPORT_ATTEMPT_STARTED",
        "opening_id": work_order["opening_id"],
        "authorization_sha256": work_order["authorization_sha256"],
        "work_order_self_sha256": work_order["work_order_self_sha256"],
        "request_ledger_sha256": request_ledger_sha256,
        "attempt_number": attempt_number,
        "mode": "RESUME_SAME_OPENING" if resume else "INITIAL_OPENING_TRANSPORT",
        "opening_count": 1,
        "completed_before_attempt_request_sha256": sorted(completed),
        "missing_at_start_request_sha256": sorted(missing),
        "response_replacement_allowed": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    document = {
        **stable,
        "attempt_start_self_sha256": sha256_bytes(canonical_json_bytes(stable)),
    }
    path = attempts_root / f"attempt_{attempt_number:06d}_start.json"
    _create_bytes(path, canonical_json_bytes(document))
    return path, document


def _write_attempt_result(
    *,
    attempts_root: Path,
    attempt_number: int,
    attempt_start_path: Path,
    work_order: Mapping[str, Any],
    request_ledger_sha256: str,
    completed: Iterable[str],
    missing: Iterable[str],
    status: str,
    failure_class: str | None,
) -> Path:
    stable = {
        "format": ACQUISITION_ATTEMPT_RESULT_FORMAT,
        "status": status,
        "opening_id": work_order["opening_id"],
        "authorization_sha256": work_order["authorization_sha256"],
        "work_order_self_sha256": work_order["work_order_self_sha256"],
        "request_ledger_sha256": request_ledger_sha256,
        "attempt_number": attempt_number,
        "attempt_start_sha256": sha256_file(attempt_start_path),
        "opening_count": 1,
        "completed_request_sha256": sorted(completed),
        "missing_request_sha256": sorted(missing),
        "failure_class": failure_class,
        "response_replacement_count": 0,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    document = {
        **stable,
        "attempt_result_self_sha256": sha256_bytes(canonical_json_bytes(stable)),
    }
    path = attempts_root / f"attempt_{attempt_number:06d}_result.json"
    _create_bytes(path, canonical_json_bytes(document))
    return path


def _validate_timestamp(value: object) -> None:
    if not isinstance(value, str):
        raise OutcomeAcquisitionError("NWIS retrieval timestamp is malformed")
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise OutcomeAcquisitionError("NWIS retrieval timestamp is malformed") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(None):
        raise OutcomeAcquisitionError("NWIS retrieval timestamp is not UTC")


def _validate_transaction(
    *,
    directory: Path,
    spec: Mapping[str, Any],
    starts: Mapping[int, Mapping[str, Any]],
) -> tuple[bytes, dict[str, Any], Path, Path]:
    if not directory.is_dir() or directory.is_symlink():
        raise OutcomeAcquisitionError("NWIS transaction path is malformed")
    response_path = directory / "response.bin"
    metadata_path = directory / "metadata.json"
    if set(path.name for path in directory.iterdir()) != {
        "response.bin", "metadata.json"
    }:
        raise OutcomeAcquisitionError(
            "partial or extraneous NWIS transaction is indeterminate"
        )
    if any(
        not path.is_file() or path.is_symlink()
        for path in (response_path, metadata_path)
    ):
        raise OutcomeAcquisitionError("NWIS transaction files are malformed")
    payload = _read_regular_file_no_follow(response_path)
    metadata = _read_json(metadata_path, label="NWIS transaction metadata")
    expected_fields = {
        "schema_version", "opening_id", "authorization_sha256",
        "work_order_self_sha256", "request_ledger_sha256", "attempt_number",
        "request", "request_sha256", "retrieved_at_utc", "http_status",
        "response_headers", "final_url", "byte_count", "response_sha256",
        "response_file", "maximum_response_bytes_per_request",
    }
    if set(metadata) != expected_fields:
        raise OutcomeAcquisitionError("NWIS transaction metadata schema changed")
    request = dict(spec["request"])
    expected = {
        "schema_version": 1,
        "request": request,
        "request_sha256": spec["request_sha256"],
        "http_status": 200,
        "final_url": request["url"],
        "byte_count": len(payload),
        "response_sha256": sha256_bytes(payload),
        "response_file": "response.bin",
        "maximum_response_bytes_per_request": (
            MAX_CONFIRMATORY_NWIS_RESPONSE_BYTES
        ),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise OutcomeAcquisitionError("NWIS transaction request/response binding changed")
    if not payload or not isinstance(metadata.get("response_headers"), Mapping):
        raise OutcomeAcquisitionError("NWIS transaction response is empty or malformed")
    _validate_timestamp(metadata.get("retrieved_at_utc"))
    attempt_number = metadata.get("attempt_number")
    if not isinstance(attempt_number, int) or attempt_number not in starts:
        raise OutcomeAcquisitionError("NWIS transaction lacks a valid attempt binding")
    start = starts[attempt_number]
    if (
        spec["request_sha256"]
        not in start["missing_at_start_request_sha256"]
        or metadata.get("opening_id") != start.get("opening_id")
        or metadata.get("authorization_sha256")
        != start.get("authorization_sha256")
        or metadata.get("work_order_self_sha256")
        != start.get("work_order_self_sha256")
        or metadata.get("request_ledger_sha256")
        != start.get("request_ledger_sha256")
    ):
        raise OutcomeAcquisitionError("NWIS transaction attempt identity changed")
    if any(path.stat().st_mode & 0o222 for path in (response_path, metadata_path)):
        raise OutcomeAcquisitionError("NWIS transaction is not immutable")
    if _read_regular_file_no_follow(metadata_path) != canonical_json_bytes(metadata):
        raise OutcomeAcquisitionError("NWIS transaction metadata is noncanonical")
    return payload, metadata, response_path, metadata_path


def _recover_complete_pending_transactions(
    *,
    raw_root: Path,
    specs_by_sha: Mapping[str, Mapping[str, Any]],
    starts: Mapping[int, Mapping[str, Any]],
) -> None:
    provider_root = raw_root / CONFIRMATORY_NWIS_PROVIDER
    pending_root = provider_root / ".pending"
    if not pending_root.exists():
        return
    if not pending_root.is_dir() or pending_root.is_symlink():
        raise OutcomeAcquisitionError("NWIS pending-transaction root is malformed")
    for pending in sorted(pending_root.iterdir()):
        request_sha = pending.name
        spec = specs_by_sha.get(request_sha)
        canonical = provider_root / request_sha
        if spec is None or canonical.exists():
            raise OutcomeAcquisitionError(
                "pending NWIS transaction is unknown or duplicates canonical bytes"
            )
        _validate_transaction(directory=pending, spec=spec, starts=starts)
        with _open_directory_chain(
            pending_root, create=False
        ) as pending_descriptor, _open_directory_chain(
            provider_root, create=False
        ) as provider_descriptor:
            os.rename(
                pending.name,
                canonical.name,
                src_dir_fd=pending_descriptor,
                dst_dir_fd=provider_descriptor,
            )
        _fsync_directory(provider_root)


def _scan_transactions(
    *,
    raw_root: Path,
    specs: Sequence[Mapping[str, Any]],
    starts: Mapping[int, Mapping[str, Any]],
) -> tuple[
    dict[str, tuple[bytes, dict[str, Any], Path, Path]], list[str]
]:
    specs_by_sha = {str(spec["request_sha256"]): spec for spec in specs}
    provider_root = raw_root / CONFIRMATORY_NWIS_PROVIDER
    allowed_raw_names = {CONFIRMATORY_NWIS_PROVIDER, "snapshot_index.json"}
    if raw_root.exists():
        extras = {path.name for path in raw_root.iterdir()} - allowed_raw_names
        if extras:
            raise OutcomeAcquisitionError(
                f"raw NWIS root contains extraneous entries: {sorted(extras)}"
            )
    if provider_root.exists():
        if not provider_root.is_dir() or provider_root.is_symlink():
            raise OutcomeAcquisitionError("NWIS provider root is malformed")
        allowed_provider_names = {*specs_by_sha, ".pending"}
        extras = {path.name for path in provider_root.iterdir()} - allowed_provider_names
        if extras:
            raise OutcomeAcquisitionError(
                f"NWIS provider root contains extraneous entries: {sorted(extras)}"
            )
    complete: dict[str, tuple[bytes, dict[str, Any], Path, Path]] = {}
    missing: list[str] = []
    for spec in specs:
        request_sha = str(spec["request_sha256"])
        directory = provider_root / request_sha
        if not directory.exists():
            missing.append(request_sha)
            continue
        complete[request_sha] = _validate_transaction(
            directory=directory, spec=spec, starts=starts
        )
    return complete, missing


def _read_bounded_response(response: Any) -> tuple[bytes, str]:
    """Read one response incrementally without crossing the fixed byte cap."""
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    byte_count = 0
    while True:
        remaining = MAX_CONFIRMATORY_NWIS_RESPONSE_BYTES - byte_count
        requested = min(_RESPONSE_CHUNK_BYTES, remaining + 1)
        chunk = response.read(requested)
        if not isinstance(chunk, bytes):
            raise OutcomeAcquisitionError("NWIS response reader returned non-bytes")
        if not chunk:
            break
        if len(chunk) > remaining:
            raise OutcomeAcquisitionError(
                "NWIS response exceeds the fixed per-request byte limit"
            )
        chunks.append(chunk)
        digest.update(chunk)
        byte_count += len(chunk)
    return b"".join(chunks), digest.hexdigest()


def _fetch_create_only(
    *,
    raw_root: Path,
    spec: Mapping[str, Any],
    work_order: Mapping[str, Any],
    request_ledger_sha256: str,
    attempt_number: int,
    attempts: int = 3,
) -> tuple[bytes, dict[str, Any], Path, Path]:
    request = dict(spec["request"])
    request_sha = str(spec["request_sha256"])
    provider_root = raw_root / CONFIRMATORY_NWIS_PROVIDER
    directory = provider_root / request_sha
    pending = provider_root / ".pending" / request_sha
    if directory.exists() or pending.exists():
        raise OutcomeAcquisitionError(
            "missing-request fetch found a preexisting transaction path"
        )

    last_error: Exception | None = None
    payload: bytes | None = None
    response_values: tuple[int, dict[str, str], str, str] | None = None
    for attempt in range(max(1, attempts)):
        try:
            http_request = urllib.request.Request(
                str(request["url"]), headers={}, method="GET"
            )

            class _RejectRedirects(urllib.request.HTTPRedirectHandler):
                def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
                    return None

            with urllib.request.build_opener(_RejectRedirects).open(
                http_request, timeout=120.0
            ) as response:
                candidate, candidate_sha256 = _read_bounded_response(response)
                status = int(getattr(response, "status", 200) or 200)
                response_headers = {
                    str(key): str(value) for key, value in response.headers.items()
                }
                final_url = str(response.geturl())
            if status != 200 or not candidate or final_url != str(request["url"]):
                raise OutcomeAcquisitionError(
                    "NWIS returned an empty, non-200, or redirected response"
                )
            payload = candidate
            response_values = (
                status,
                response_headers,
                final_url,
                candidate_sha256,
            )
            break
        except Exception as exc:  # preserve exact cause below
            last_error = exc
            if attempt + 1 < max(1, attempts):
                time.sleep(min(2 ** attempt, 4))
    if payload is None or response_values is None:
        raise OutcomeAcquisitionError("fixed NWIS acquisition failed") from last_error

    status, response_headers, final_url, response_sha256 = response_values
    metadata = {
        "schema_version": 1,
        "opening_id": work_order["opening_id"],
        "authorization_sha256": work_order["authorization_sha256"],
        "work_order_self_sha256": work_order["work_order_self_sha256"],
        "request_ledger_sha256": request_ledger_sha256,
        "attempt_number": attempt_number,
        "request": request,
        "request_sha256": request_sha,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "http_status": status,
        "response_headers": response_headers,
        "final_url": final_url,
        "byte_count": len(payload),
        "response_sha256": response_sha256,
        "response_file": "response.bin",
        "maximum_response_bytes_per_request": (
            MAX_CONFIRMATORY_NWIS_RESPONSE_BYTES
        ),
    }
    _secure_create_directory(pending)
    _create_bytes(pending / "response.bin", payload)
    _create_bytes(pending / "metadata.json", canonical_json_bytes(metadata))
    _fsync_directory(pending)
    if directory.exists():
        raise OutcomeAcquisitionError(
            "refusing to replace an existing NWIS request transaction"
        )
    with _open_directory_chain(
        pending.parent, create=False
    ) as pending_descriptor, _open_directory_chain(
        provider_root, create=False
    ) as provider_descriptor:
        os.rename(
            pending.name,
            directory.name,
            src_dir_fd=pending_descriptor,
            dst_dir_fd=provider_descriptor,
        )
    _fsync_directory(provider_root)
    return payload, metadata, directory / "response.bin", directory / "metadata.json"


def _attempt_index(
    *,
    root: Path,
    attempts_root: Path,
    work_order: Mapping[str, Any],
    request_ledger_path: Path,
    transactions: Mapping[str, tuple[bytes, Mapping[str, Any], Path, Path]],
) -> dict[str, Any]:
    request_ids = set(transactions)
    starts, results = _load_attempt_history(
        attempts_root=attempts_root,
        opening_id=str(work_order["opening_id"]),
        authorization_sha256=str(work_order["authorization_sha256"]),
        work_order_self_sha256=str(work_order["work_order_self_sha256"]),
        request_ledger_sha256=sha256_file(request_ledger_path),
        request_ids=request_ids,
    )
    if (
        not starts
        or max(starts) not in results
        or results[max(starts)].get("status")
        != "ALL_LEDGER_TRANSACTIONS_COMPLETE"
    ):
        raise OutcomeAcquisitionError("final transport attempt evidence is incomplete")
    attempts = []
    for number in sorted(starts):
        start_path = attempts_root / f"attempt_{number:06d}_start.json"
        result_path = attempts_root / f"attempt_{number:06d}_result.json"
        result = results.get(number)
        attempts.append({
            "attempt_number": number,
            "mode": starts[number]["mode"],
            "status": (
                result["status"]
                if result is not None
                else "NO_RESULT_PROCESS_TERMINATED"
            ),
            "start": _binding(root, start_path),
            "result": (
                _binding(root, result_path) if result is not None else None
            ),
        })
    timestamps = sorted(
        str(transaction[1]["retrieved_at_utc"])
        for transaction in transactions.values()
    )
    final_start = starts[max(starts)]
    stable = {
        "format": ACQUISITION_ATTEMPT_INDEX_FORMAT,
        "status": "ALL_LEDGER_TRANSACTIONS_COMPLETE",
        "opening_id": work_order["opening_id"],
        "authorization_sha256": work_order["authorization_sha256"],
        "work_order_self_sha256": work_order["work_order_self_sha256"],
        "request_ledger": _binding(root, request_ledger_path),
        "request_count": len(request_ids),
        "attempt_count": len(starts),
        "resume_count": sum(
            start["mode"] == "RESUME_SAME_OPENING" for start in starts.values()
        ),
        "opening_count": 1,
        "response_replacement_count": 0,
        "completed_before_final_attempt_request_sha256": final_start[
            "completed_before_attempt_request_sha256"
        ],
        "retrieval_span_utc": {
            "first": timestamps[0],
            "last": timestamps[-1],
        },
        "attempts": attempts,
    }
    return {
        **stable,
        "attempt_index_self_sha256": sha256_bytes(canonical_json_bytes(stable)),
    }


def _forbidden_resume_outputs(
    state: Mapping[str, Path]
) -> list[str]:
    keys = (
        "acquisition_manifest", "temporal_outcomes", "external_outcomes",
        "availability_registry", "outcome_quality_audit",
        "approved_target_sensitivity", "spatial_sensitivity",
        "probabilistic_evaluation", "temporal_predictions",
        "external_predictions", "statistics", "report", "receipt",
        "receipt_sha256",
    )
    return sorted(key for key in keys if key in state and state[key].exists())


def _unexpected_transport_namespace_entries(
    *, state: Mapping[str, Path], evidence: Mapping[str, Path]
) -> list[str]:
    acquisition_root = evidence["root"]
    if not acquisition_root.exists():
        return []
    allowed = {
        state["raw_nwis_root"].name,
        state["acquisition_request_map"].name,
        evidence["request_ledger"].name,
        evidence["attempts_root"].name,
        evidence["attempt_index"].name,
        evidence["lock"].name,
    }
    return sorted(path.name for path in acquisition_root.iterdir() if path.name not in allowed)


def inspect_transport_resume_state(
    *,
    work_order_path: Path,
    work_order: Mapping[str, Any],
    authorization: Mapping[str, Any],
    state: Mapping[str, Path],
) -> dict[str, Any]:
    """Read-only, fail-closed classification of the raw transport namespace."""
    evidence = _evidence_paths(state)
    raw_root = state["raw_nwis_root"]
    unexpected = _unexpected_transport_namespace_entries(
        state=state, evidence=evidence
    )
    if unexpected:
        raise OutcomeAcquisitionError(
            f"raw transport namespace contains extraneous entries: {unexpected}"
        )
    expected_ledger = _expected_request_ledger(
        work_order=work_order,
        authorization=authorization,
        work_order_path=work_order_path,
    )
    if not evidence["request_ledger"].exists():
        if (
            evidence["attempts_root"].exists()
            or evidence["attempt_index"].exists()
            or (raw_root.exists() and any(raw_root.iterdir()))
        ):
            raise OutcomeAcquisitionError(
                "raw evidence exists without the frozen request ledger"
            )
        return {
            "classification": "RESUMABLE_BEFORE_REQUEST_LEDGER_PUBLICATION",
            "completed_request_count": 0,
            "missing_request_count": int(expected_ledger["request_count"]),
            "recoverable_pending_request_count": 0,
            "attempt_count": 0,
        }
    ledger = _self_hashed_document(
        evidence["request_ledger"],
        label="acquisition request ledger",
        self_field="request_ledger_self_sha256",
        format_name=ACQUISITION_REQUEST_LEDGER_FORMAT,
    )
    if ledger != expected_ledger:
        raise OutcomeAcquisitionError(
            "acquisition request ledger differs from the fixed work order"
        )
    specs = list(ledger["requests"])
    specs_by_sha = {str(spec["request_sha256"]): spec for spec in specs}
    request_ids = set(specs_by_sha)
    starts, _results = _load_attempt_history(
        attempts_root=evidence["attempts_root"],
        opening_id=str(work_order["opening_id"]),
        authorization_sha256=str(work_order["authorization_sha256"]),
        work_order_self_sha256=str(work_order["work_order_self_sha256"]),
        request_ledger_sha256=sha256_file(evidence["request_ledger"]),
        request_ids=request_ids,
    )
    pending_ids: list[str] = []
    pending_root = raw_root / CONFIRMATORY_NWIS_PROVIDER / ".pending"
    if pending_root.exists():
        if not pending_root.is_dir() or pending_root.is_symlink():
            raise OutcomeAcquisitionError("pending NWIS root is malformed")
        for pending in sorted(pending_root.iterdir()):
            request_sha = pending.name
            spec = specs_by_sha.get(request_sha)
            canonical = pending.parent.parent / request_sha
            if spec is None or canonical.exists():
                raise OutcomeAcquisitionError(
                    "pending NWIS transaction is unknown or duplicated"
                )
            _validate_transaction(directory=pending, spec=spec, starts=starts)
            pending_ids.append(request_sha)
    complete, missing = _scan_transactions(
        raw_root=raw_root, specs=specs, starts=starts
    )
    if not set(pending_ids) <= set(missing):
        raise OutcomeAcquisitionError("pending/canonical transaction registry changed")
    snapshot_index = raw_root / "snapshot_index.json"
    request_map_path = state["acquisition_request_map"]
    if missing and (snapshot_index.exists() or request_map_path.exists()):
        raise OutcomeAcquisitionError(
            "derived raw index/request map appeared before every transaction"
        )
    if evidence["attempt_index"].exists():
        raise OutcomeAcquisitionError(
            "final attempt index appeared before normalized outcome publication"
        )
    if not missing:
        index_rows: list[dict[str, Any]] = []
        request_rows: list[dict[str, Any]] = []
        for spec in specs:
            request_sha = str(spec["request_sha256"])
            payload, metadata, response_path, metadata_path = complete[request_sha]
            series_registry = nwis_confirmatory_series_registry(payload)
            request_rows.append({
                "cohort": spec["cohort"],
                "site_no": spec["site_no"],
                "request_sha256": request_sha,
                "response_sha256": metadata["response_sha256"],
                "retrieved_at_utc": metadata["retrieved_at_utc"],
                "byte_count": metadata["byte_count"],
                "attempt_number": metadata["attempt_number"],
                "series_registry": series_registry,
            })
            index_rows.append({
                "provider": CONFIRMATORY_NWIS_PROVIDER,
                "request_sha256": request_sha,
                "response_sha256": metadata["response_sha256"],
                "retrieved_at_utc": metadata["retrieved_at_utc"],
                "byte_count": metadata["byte_count"],
                "attempt_number": metadata["attempt_number"],
                "request": spec["request"],
                "metadata_path": metadata_path.relative_to(raw_root).as_posix(),
                "metadata_sha256": sha256_file(metadata_path),
                "response_path": response_path.relative_to(raw_root).as_posix(),
                "series_registry": series_registry,
            })
        index_rows.sort(key=lambda row: str(row["request_sha256"]))
        request_rows.sort(
            key=lambda row: (str(row["cohort"]), str(row["site_no"]))
        )
        expected_index = canonical_json_bytes({
            "schema_version": 1,
            "snapshot_count": len(index_rows),
            "records": index_rows,
        })
        expected_map = canonical_json_bytes({
            "format": ACQUISITION_REQUEST_MAP_FORMAT,
            "opening_id": authorization["opening_id"],
            "authorization_sha256": work_order["authorization_sha256"],
            "provider": CONFIRMATORY_NWIS_PROVIDER,
            "request_count": len(request_rows),
            "requests": request_rows,
        })
        for path, payload, label in (
            (snapshot_index, expected_index, "NWIS snapshot index"),
            (request_map_path, expected_map, "NWIS request map"),
        ):
            if path.exists() and (
                not path.is_file()
                or path.is_symlink()
                or _read_regular_file_no_follow(path) != payload
            ):
                raise OutcomeAcquisitionError(
                    f"existing {label} differs from raw replay"
                )
    effective_missing = set(missing) - set(pending_ids)
    classification = (
        "RESUMABLE_RAW_COMPLETE_DERIVATION_NOT_PUBLISHED"
        if not missing
        else "RESUMABLE_RECOVERABLE_PENDING_BYTES"
        if pending_ids
        else "RESUMABLE_MISSING_REQUESTS"
    )
    return {
        "classification": classification,
        "completed_request_count": len(complete),
        "missing_request_count": len(effective_missing),
        "recoverable_pending_request_count": len(pending_ids),
        "attempt_count": len(starts),
    }


def acquire_from_work_order(
    work_order_path: str | Path,
    *,
    root: str | Path,
    entrypoint_path: str | Path,
    resume: bool = False,
) -> Path:
    """Execute or continue one fixed work order and return its manifest path."""
    root = Path(root).resolve()
    work_order_path = Path(os.path.abspath(os.fspath(work_order_path)))
    try:
        work_order, authorization, state = validate_acquisition_work_order(
            work_order_path, root=root, entrypoint_path=entrypoint_path
        )
    except Exception as exc:
        raise OutcomeAcquisitionError("acquisition contract validation failed") from exc
    evidence = _evidence_paths(state)
    for path in (*state.values(), *evidence.values()):
        assert_no_symlink_components(root, path)
    raw_root = state["raw_nwis_root"]
    with _exclusive_transport_lock(evidence["lock"]):
        forbidden = _forbidden_resume_outputs(state)
        if forbidden:
            raise OutcomeAcquisitionError(
                "raw acquisition cannot continue after derived/trusted publication: "
                f"{forbidden}"
            )
        unexpected = _unexpected_transport_namespace_entries(
            state=state, evidence=evidence
        )
        if unexpected:
            raise OutcomeAcquisitionError(
                f"raw transport namespace contains extraneous entries: {unexpected}"
            )
        if not resume and raw_root.exists():
            raise OutcomeAcquisitionError("production raw NWIS root already exists")

        expected_ledger = _expected_request_ledger(
            work_order=work_order,
            authorization=authorization,
            work_order_path=work_order_path,
        )
        ledger = _load_or_create_request_ledger(
            path=evidence["request_ledger"],
            expected=expected_ledger,
            raw_root=raw_root,
            resume=resume,
        )
        request_ledger_sha256 = sha256_file(evidence["request_ledger"])
        specs = list(ledger["requests"])
        request_ids = {str(spec["request_sha256"]) for spec in specs}
        starts, results = _load_attempt_history(
            attempts_root=evidence["attempts_root"],
            opening_id=str(work_order["opening_id"]),
            authorization_sha256=str(work_order["authorization_sha256"]),
            work_order_self_sha256=str(work_order["work_order_self_sha256"]),
            request_ledger_sha256=request_ledger_sha256,
            request_ids=request_ids,
        )
        if not resume and (starts or results):
            raise OutcomeAcquisitionError(
                "initial acquisition refuses preexisting attempt evidence"
            )
        _secure_mkdirs(raw_root)
        _recover_complete_pending_transactions(
            raw_root=raw_root,
            specs_by_sha={str(spec["request_sha256"]): spec for spec in specs},
            starts=starts,
        )
        complete, missing = _scan_transactions(
            raw_root=raw_root, specs=specs, starts=starts
        )
        attempt_number = len(starts) + 1
        start_path, start_document = _write_attempt_start(
            attempts_root=evidence["attempts_root"],
            attempt_number=attempt_number,
            resume=resume,
            work_order=work_order,
            request_ledger_sha256=request_ledger_sha256,
            completed=complete,
            missing=missing,
        )
        starts = {**starts, attempt_number: start_document}

        try:
            specs_by_sha = {
                str(spec["request_sha256"]): spec for spec in specs
            }
            if missing:
                # This second complete replay is deliberately adjacent to the
                # first possible socket operation.  It catches source changes
                # after the parent/orchestrator preflight but before transport.
                validate_frozen_source_identity(
                    root=root,
                    authorization=authorization,
                )
            for request_sha in missing:
                complete[request_sha] = _fetch_create_only(
                    raw_root=raw_root,
                    spec=specs_by_sha[request_sha],
                    work_order=work_order,
                    request_ledger_sha256=request_ledger_sha256,
                    attempt_number=attempt_number,
                )
        except Exception as exc:
            try:
                completed_after, missing_after = _scan_transactions(
                    raw_root=raw_root, specs=specs, starts=starts
                )
                _write_attempt_result(
                    attempts_root=evidence["attempts_root"],
                    attempt_number=attempt_number,
                    attempt_start_path=start_path,
                    work_order=work_order,
                    request_ledger_sha256=request_ledger_sha256,
                    completed=completed_after,
                    missing=missing_after,
                    status="TRANSPORT_INCOMPLETE_MAY_RESUME_SAME_OPENING",
                    failure_class=type(exc).__name__,
                )
            except Exception:
                # A partial/corrupt transaction is deliberately left indeterminate.
                pass
            raise OutcomeAcquisitionError(
                "fixed-ledger NWIS transport did not complete"
            ) from exc

        complete, missing = _scan_transactions(
            raw_root=raw_root, specs=specs, starts=starts
        )
        if missing or set(complete) != request_ids:
            _write_attempt_result(
                attempts_root=evidence["attempts_root"],
                attempt_number=attempt_number,
                attempt_start_path=start_path,
                work_order=work_order,
                request_ledger_sha256=request_ledger_sha256,
                completed=complete,
                missing=missing,
                status="TRANSPORT_INCOMPLETE_MAY_RESUME_SAME_OPENING",
                failure_class="INCOMPLETE_LEDGER",
            )
            raise OutcomeAcquisitionError("not every frozen request is complete")

        # Only this point crosses the frozen pre-completion visibility boundary.
        frames: dict[str, list[pd.DataFrame]] = {"temporal": [], "external": []}
        request_rows: list[dict[str, Any]] = []
        index_rows: list[dict[str, Any]] = []
        for spec in specs:
            request_sha = str(spec["request_sha256"])
            payload, metadata, response_path, metadata_path = complete[request_sha]
            parsed = parse_nwis_confirmatory_daily(
                payload,
                site_no=str(spec["site_no"]),
                start=str(authorization["acquisition_plan"]["history_start"]),
                end=str(authorization["acquisition_plan"]["target_end"]),
            )
            series_registry = nwis_confirmatory_series_registry(payload)
            frames[str(spec["cohort"])].append(parsed)
            request_rows.append({
                "cohort": spec["cohort"],
                "site_no": spec["site_no"],
                "request_sha256": request_sha,
                "response_sha256": metadata["response_sha256"],
                "retrieved_at_utc": metadata["retrieved_at_utc"],
                "byte_count": metadata["byte_count"],
                "attempt_number": metadata["attempt_number"],
                "series_registry": series_registry,
            })
            index_rows.append({
                "provider": CONFIRMATORY_NWIS_PROVIDER,
                "request_sha256": request_sha,
                "response_sha256": metadata["response_sha256"],
                "retrieved_at_utc": metadata["retrieved_at_utc"],
                "byte_count": metadata["byte_count"],
                "attempt_number": metadata["attempt_number"],
                "request": spec["request"],
                "metadata_path": metadata_path.relative_to(raw_root).as_posix(),
                "metadata_sha256": sha256_file(metadata_path),
                "response_path": response_path.relative_to(raw_root).as_posix(),
                "series_registry": series_registry,
            })

        index_rows.sort(key=lambda row: str(row["request_sha256"]))
        index_path = raw_root / "snapshot_index.json"
        _create_or_validate_bytes(
            index_path,
            canonical_json_bytes({
                "schema_version": 1,
                "snapshot_count": len(index_rows),
                "records": index_rows,
            }),
            label="NWIS snapshot index",
        )
        request_rows.sort(key=lambda row: (str(row["cohort"]), str(row["site_no"])))
        request_map_path = state["acquisition_request_map"]
        _create_or_validate_bytes(
            request_map_path,
            canonical_json_bytes({
                "format": ACQUISITION_REQUEST_MAP_FORMAT,
                "opening_id": authorization["opening_id"],
                "authorization_sha256": work_order["authorization_sha256"],
                "provider": CONFIRMATORY_NWIS_PROVIDER,
                "request_count": len(request_rows),
                "requests": request_rows,
            }),
            label="NWIS request map",
        )

        outcome_paths = {
            "temporal": state["temporal_outcomes"],
            "external": state["external_outcomes"],
        }
        for cohort, output_path in outcome_paths.items():
            combined = pd.concat(frames[cohort], ignore_index=True)
            combined["site_no"] = combined.site_no.astype("string")
            combined["DATE"] = pd.to_datetime(combined.DATE)
            combined = combined.sort_values(["site_no", "DATE"]).reset_index(
                drop=True
            )
            _create_parquet(output_path, combined)

        result_path = _write_attempt_result(
            attempts_root=evidence["attempts_root"],
            attempt_number=attempt_number,
            attempt_start_path=start_path,
            work_order=work_order,
            request_ledger_sha256=request_ledger_sha256,
            completed=complete,
            missing=(),
            status="ALL_LEDGER_TRANSACTIONS_COMPLETE",
            failure_class=None,
        )
        del result_path
        attempt_index = _attempt_index(
            root=root,
            attempts_root=evidence["attempts_root"],
            work_order=work_order,
            request_ledger_path=evidence["request_ledger"],
            transactions=complete,
        )
        _create_or_validate_bytes(
            evidence["attempt_index"],
            canonical_json_bytes(attempt_index),
            label="transport-attempt index",
        )

        manifest_path = state["acquisition_manifest"]
        _create_bytes(manifest_path, canonical_json_bytes({
            "format": ACQUISITION_MANIFEST_FORMAT,
            "opening_id": authorization["opening_id"],
            "authorization_sha256": work_order["authorization_sha256"],
            "protocol_sha256": authorization["protocol"]["sha256"],
            "labels_state": "OPENED_ONCE",
            "site_replacement_count": 0,
            "response_replacement_count": 0,
            "history_start": str(authorization["acquisition_plan"]["history_start"]),
            "target_start": str(authorization["acquisition_plan"]["target_start"]),
            "target_end": str(authorization["acquisition_plan"]["target_end"]),
            "maximum_response_bytes_per_request": (
                MAX_CONFIRMATORY_NWIS_RESPONSE_BYTES
            ),
            "request_ledger": _binding(root, evidence["request_ledger"]),
            "transport_attempt_index": _binding(root, evidence["attempt_index"]),
            "transport_summary": {
                "opening_count": 1,
                "attempt_count": attempt_index["attempt_count"],
                "resume_count": attempt_index["resume_count"],
                "completed_before_final_attempt_request_sha256": attempt_index[
                    "completed_before_final_attempt_request_sha256"
                ],
                "retrieval_span_utc": attempt_index["retrieval_span_utc"],
            },
            "raw_nwis_snapshot_index": _binding(root, index_path),
            "request_map": _binding(root, request_map_path),
            "normalized_outcome_tables": {
                cohort: _binding(root, path)
                for cohort, path in outcome_paths.items()
            },
            "producer_role": "RAW_ONLY_NO_PREDICTIONS_OR_STATISTICS",
        }))
        return manifest_path

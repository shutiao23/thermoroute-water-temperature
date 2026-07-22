"""Raw-only, create-only NWIS acquisition for the Route-A opening.

The first process freezes the complete request ledger before its first HTTPS
request.  A later process may continue the *same* irreversible opening, but it
can only reuse byte-verified, durably published canonical transactions and
retry ledger entries for which no such transaction exists.  HTTP delivery is
therefore at least once, not exactly once.  No outcome is parsed until every
raw transaction is complete.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import tempfile
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
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
    RAW_ACQUISITION_FORBIDDEN_STATE_KEYS,
    TRUSTED_STATE_KEYS,
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
_ACQUISITION_STAGE_PREFIX = ".acquisition-stage-v1-"
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
            created = False
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
                    created = True
                except FileExistsError:
                    pass
                child = os.open(
                    component,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=descriptor,
                )
            if created:
                os.fsync(child)
                os.fsync(descriptor)
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
            os.mkdir(path.name, 0o700, dir_fd=parent_descriptor)
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


def _require_immutable_atomic_final(path: Path, *, label: str) -> None:
    """Require one immutable final inode or its sole known post-link temp."""
    path = Path(os.path.abspath(os.fspath(path)))
    with _open_directory_chain(path.parent, create=False) as parent_descriptor:
        parent = os.fstat(parent_descriptor)
        descriptor = os.open(
            path.name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_dev != parent.st_dev
                or metadata.st_mode & 0o222
                or metadata.st_nlink not in {1, 2}
            ):
                raise OutcomeAcquisitionError(
                    f"{label} is not one immutable atomic final file"
                )
            linked_temps = 0
            temporary_pattern = re.compile(
                rf"\.{re.escape(path.name)}\.[a-z0-9_]{{8}}\.tmp"
            )
            for name in os.listdir(parent_descriptor):
                if temporary_pattern.fullmatch(name) is None:
                    continue
                temporary = os.stat(
                    name, dir_fd=parent_descriptor, follow_symlinks=False
                )
                if (
                    stat.S_ISREG(temporary.st_mode)
                    and temporary.st_dev == metadata.st_dev
                    and temporary.st_ino == metadata.st_ino
                ):
                    linked_temps += 1
            if (
                metadata.st_nlink == 2 and linked_temps != 1
            ) or (metadata.st_nlink == 1 and linked_temps != 0):
                raise OutcomeAcquisitionError(
                    f"{label} has an unknown hard link"
                )
        finally:
            os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    with _open_directory_chain(path, create=False) as descriptor:
        os.fsync(descriptor)


def _atomic_create_fault(_point: str, _path: Path) -> None:
    """No-op hook replaced only by atomic-publication crash tests."""


def _acquisition_transport_fault(_point: str) -> None:
    """No-op hook replaced only by transport-boundary crash tests."""


def _cleanup_atomic_create_temps(
    parent_descriptor: int,
    *,
    final_name: str,
    expected_payload: bytes,
) -> None:
    """Remove only safe temp remnants created for this exact final name."""
    parent = os.fstat(parent_descriptor)
    if (
        not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != os.geteuid()
        or parent.st_mode & 0o022
    ):
        raise OutcomeAcquisitionError(
            "atomic-create parent is not owner-controlled"
        )
    try:
        final_metadata = os.stat(
            final_name, dir_fd=parent_descriptor, follow_symlinks=False
        )
    except FileNotFoundError:
        final_metadata = None
    temporary_pattern = re.compile(
        rf"\.{re.escape(final_name)}\.[a-z0-9_]{{8}}\.tmp"
    )
    for name in os.listdir(parent_descriptor):
        if temporary_pattern.fullmatch(name) is None:
            continue
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
        try:
            metadata = os.fstat(descriptor)
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                payload = handle.read()
            published_link = bool(
                final_metadata is not None
                and stat.S_ISREG(final_metadata.st_mode)
                and metadata.st_dev == final_metadata.st_dev
                and metadata.st_ino == final_metadata.st_ino
                and metadata.st_nlink == 2
            )
            safe_owner_file = bool(
                stat.S_ISREG(metadata.st_mode)
                and metadata.st_uid == os.geteuid()
                and metadata.st_dev == parent.st_dev
                and not metadata.st_mode & 0o022
                and (metadata.st_nlink == 1 or published_link)
            )
            if not safe_owner_file:
                raise OutcomeAcquisitionError(
                    "atomic-create temporary artifact has unsafe metadata"
                )
            # A prefix left before link is not authoritative.  A complete temp
            # hard-linked to the final is redundant.  Both may be unlinked
            # without changing a published final name.
            if payload != expected_payload and published_link:
                raise OutcomeAcquisitionError(
                    "published atomic-create temp differs from final payload"
                )
            os.unlink(name, dir_fd=parent_descriptor)
        finally:
            os.close(descriptor)
    os.fsync(parent_descriptor)


def _create_bytes(path: Path, payload: bytes) -> None:
    path = Path(os.path.abspath(os.fspath(path)))
    with _open_directory_chain(path.parent, create=True) as parent_descriptor:
        _cleanup_atomic_create_temps(
            parent_descriptor,
            final_name=path.name,
            expected_payload=payload,
        )
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
        try:
            with os.fdopen(descriptor, "wb") as handle:
                split = max(1, len(payload) // 2) if payload else 0
                handle.write(payload[:split])
                handle.flush()
                _atomic_create_fault("after_temporary_prefix_write", path)
                handle.write(payload[split:])
                handle.flush()
                os.fchmod(handle.fileno(), 0o444)
                _atomic_create_fault(
                    "after_final_mode_before_inode_fsync", path
                )
                os.fsync(handle.fileno())
            _atomic_create_fault("before_no_replace_link", path)
            try:
                os.link(
                    temporary.name,
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
            _atomic_create_fault("after_no_replace_link", path)
        finally:
            try:
                os.unlink(temporary.name, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
            except FileNotFoundError:
                pass


def _create_or_validate_bytes(path: Path, payload: bytes, *, label: str) -> None:
    if os.path.lexists(path):
        if (
            not path.is_file()
            or path.is_symlink()
            or _read_regular_file_no_follow(path) != payload
        ):
            raise OutcomeAcquisitionError(f"existing {label} differs from replay")
        _require_immutable_atomic_final(path, label=label)
        return
    _create_bytes(path, payload)


def _safe_atomic_temp_names(
    directory: Path,
    *,
    allowed_final: Callable[[str], str | None],
    remove: bool,
) -> set[str]:
    """Validate, optionally remove, and return recognized SIGKILL remnants."""
    if not os.path.lexists(directory):
        return set()
    safe: set[str] = set()
    with _open_directory_chain(directory, create=False) as parent_descriptor:
        parent = _assert_owner_controlled_directory(parent_descriptor)
        for name in os.listdir(parent_descriptor):
            if not name.startswith(".") or not name.endswith(".tmp"):
                continue
            final_name = allowed_final(name)
            if final_name is None:
                continue
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_descriptor,
            )
            try:
                metadata = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or metadata.st_dev != parent.st_dev
                    or metadata.st_mode & 0o022
                ):
                    raise OutcomeAcquisitionError(
                        "atomic temporary artifact has unsafe metadata"
                    )
                try:
                    final = os.stat(
                        final_name,
                        dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    if metadata.st_nlink != 1:
                        raise OutcomeAcquisitionError(
                            "unpublished atomic temp has an external hard link"
                        )
                else:
                    if (
                        not stat.S_ISREG(final.st_mode)
                        or final.st_dev != metadata.st_dev
                        or final.st_ino != metadata.st_ino
                        or metadata.st_nlink != 2
                    ):
                        raise OutcomeAcquisitionError(
                            "published atomic temp does not bind its final file"
                        )
                safe.add(name)
            finally:
                os.close(descriptor)
        if remove:
            for name in safe:
                os.unlink(name, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
    return safe


def _named_atomic_temp_parser(final_names: set[str]):
    ordered = tuple(sorted(final_names, key=len, reverse=True))

    def parse(name: str) -> str | None:
        for final_name in ordered:
            if re.fullmatch(
                rf"\.{re.escape(final_name)}\.[a-z0-9_]{{8}}\.tmp",
                name,
            ) is not None:
                return final_name
        return None

    return parse


def _attempt_atomic_temp_parser(name: str) -> str | None:
    match = re.fullmatch(
        r"\.(attempt_[0-9]{6}_(?:start|result)\.json)\."
        r"[a-z0-9_]{8}\.tmp",
        name,
    )
    return None if match is None else match.group(1)


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
                os.fchmod(temporary_descriptor, 0o444)
                os.fsync(temporary_descriptor)
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
                os.fsync(parent_descriptor)
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
    _require_immutable_atomic_final(path, label=label)
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
    _require_immutable_atomic_final(resolved, label="bound acquisition artifact")
    return {
        "path": resolved.relative_to(root).as_posix(),
        "sha256": sha256_file(resolved),
    }


def _evidence_paths(state: Mapping[str, Path]) -> dict[str, Path]:
    transport_root = state["transport_root"]
    if state["raw_nwis_root"].parent != transport_root:
        raise OutcomeAcquisitionError(
            "raw NWIS root is outside the canonical transport namespace"
        )
    if state["raw_nwis_snapshot_index"].parent != state["raw_nwis_root"]:
        raise OutcomeAcquisitionError(
            "raw snapshot index is outside the canonical raw namespace"
        )
    return {
        "root": transport_root,
        "request_ledger": transport_root / "request_ledger_v1.json",
        "attempts_root": transport_root / "transport_attempts_v1",
        "attempt_index": transport_root / "transport_attempt_index_v1.json",
        "lock": transport_root / ".transport_resume.lock",
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
            parent = os.fstat(parent_descriptor)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_dev != parent.st_dev
                or metadata.st_nlink != 1
                or metadata.st_mode & 0o077
            ):
                raise OutcomeAcquisitionError(
                    "transport lock has unsafe metadata"
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
    if os.path.lexists(path):
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
    if os.path.lexists(raw_root) and any(raw_root.iterdir()):
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
    if not os.path.lexists(attempts_root):
        return {}, {}
    if not attempts_root.is_dir() or attempts_root.is_symlink():
        raise OutcomeAcquisitionError("transport-attempt root is malformed")
    safe_temps = _safe_atomic_temp_names(
        attempts_root,
        allowed_final=_attempt_atomic_temp_parser,
        remove=False,
    )
    starts: dict[int, dict[str, Any]] = {}
    results: dict[int, dict[str, Any]] = {}
    for path in sorted(attempts_root.iterdir()):
        if path.name in safe_temps:
            continue
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
    open_attempts = set(starts) - set(results)
    if len(open_attempts) > 1 or (
        open_attempts and open_attempts != {max(starts)}
    ):
        raise OutcomeAcquisitionError(
            "only the latest transport attempt may lack a result"
        )
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
            or completed != sorted(completed)
            or missing != sorted(missing)
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
            or final_completed != sorted(final_completed)
            or final_missing != sorted(final_missing)
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
    if starts:
        first = starts[1]
        if (
            first.get("mode") != "INITIAL_OPENING_TRANSPORT"
            or first["completed_before_attempt_request_sha256"]
            or set(first["missing_at_start_request_sha256"]) != request_ids
        ):
            raise OutcomeAcquisitionError(
                "first transport attempt does not start from the full ledger"
            )
    for number in sorted(starts):
        start = starts[number]
        start_completed = set(
            start["completed_before_attempt_request_sha256"]
        )
        start_missing = set(start["missing_at_start_request_sha256"])
        if number > 1:
            previous = results.get(number - 1)
            if previous is None:
                raise OutcomeAcquisitionError(
                    "transport attempt follows an attempt without a result"
                )
            if previous.get("status") == "ALL_LEDGER_TRANSACTIONS_COMPLETE":
                raise OutcomeAcquisitionError(
                    "transport attempt exists after ledger completion"
                )
            if (
                start["completed_before_attempt_request_sha256"]
                != previous["completed_request_sha256"]
                or start["missing_at_start_request_sha256"]
                != previous["missing_request_sha256"]
            ):
                raise OutcomeAcquisitionError(
                    "transport attempt does not continue the prior result partition"
                )
        result = results.get(number)
        if result is None:
            continue
        result_completed = set(result["completed_request_sha256"])
        result_missing = set(result["missing_request_sha256"])
        if (
            not start_completed <= result_completed
            or not result_missing <= start_missing
        ):
            raise OutcomeAcquisitionError(
                "transport attempt result rolls back completed requests"
            )
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
        # Attempt numbering is logical, not process-local.  A resume after a
        # crash between ledger publication and the first start still creates
        # logical attempt 1, which must remain the INITIAL transition.
        "mode": (
            "INITIAL_OPENING_TRANSPORT"
            if attempt_number == 1
            else "RESUME_SAME_OPENING"
        ),
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
    results: Mapping[int, Mapping[str, Any]],
) -> tuple[bytes, dict[str, Any], Path, Path]:
    if not directory.is_dir() or directory.is_symlink():
        raise OutcomeAcquisitionError("NWIS transaction path is malformed")
    directory_metadata = os.lstat(directory)
    if (
        directory_metadata.st_uid != os.geteuid()
        or directory_metadata.st_mode & 0o077
    ):
        raise OutcomeAcquisitionError(
            "NWIS transaction directory is not owner-private"
        )
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
    result = results.get(attempt_number)
    if (
        result is not None
        and spec["request_sha256"]
        not in result["completed_request_sha256"]
    ):
        raise OutcomeAcquisitionError(
            "NWIS transaction is absent from its attempt result"
        )
    for path in (response_path, metadata_path):
        metadata_status = os.lstat(path)
        if (
            metadata_status.st_uid != os.geteuid()
            or metadata_status.st_dev != directory_metadata.st_dev
            or metadata_status.st_nlink != 1
            or metadata_status.st_mode & 0o222
        ):
            raise OutcomeAcquisitionError(
                "NWIS transaction is linked, mutable, or has unsafe ownership"
            )
    if _read_regular_file_no_follow(metadata_path) != canonical_json_bytes(metadata):
        raise OutcomeAcquisitionError("NWIS transaction metadata is noncanonical")
    return payload, metadata, response_path, metadata_path


def _recover_complete_pending_transactions(
    *,
    raw_root: Path,
    specs_by_sha: Mapping[str, Mapping[str, Any]],
    starts: Mapping[int, Mapping[str, Any]],
    results: Mapping[int, Mapping[str, Any]],
) -> None:
    provider_root = raw_root / CONFIRMATORY_NWIS_PROVIDER
    pending_root = provider_root / ".pending"
    if not os.path.lexists(pending_root):
        return
    if not pending_root.is_dir() or pending_root.is_symlink():
        raise OutcomeAcquisitionError("NWIS pending-transaction root is malformed")
    for pending in sorted(pending_root.iterdir()):
        request_sha = pending.name
        spec = specs_by_sha.get(request_sha)
        canonical = provider_root / request_sha
        if spec is None or os.path.lexists(canonical):
            raise OutcomeAcquisitionError(
                "pending NWIS transaction is unknown or duplicates canonical bytes"
            )
        try:
            _validate_transaction(
                directory=pending,
                spec=spec,
                starts=starts,
                results=results,
            )
        except OutcomeAcquisitionError:
            _remove_safe_pending_transaction(pending)
            continue
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
            # A cross-directory rename mutates both directories.  Persist the
            # source removal and destination addition before this transaction
            # is considered canonical.
            os.fsync(pending_descriptor)
            os.fsync(provider_descriptor)


def _remove_safe_pending_transaction(
    directory: Path, *, remove: bool = True
) -> None:
    """Validate and optionally remove a crashed noncanonical transaction."""
    with _open_directory_chain(
        directory.parent, create=False
    ) as parent_descriptor:
        parent = os.fstat(parent_descriptor)
        descriptor = os.open(
            directory.name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_descriptor
        )
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_dev != parent.st_dev
                or metadata.st_mode & 0o077
            ):
                raise OutcomeAcquisitionError(
                    "incomplete pending transaction is not owner-private"
                )
            entries = os.listdir(descriptor)
            children: dict[str, os.stat_result] = {}
            inode_counts: dict[tuple[int, int], int] = {}
            for name in entries:
                child = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=descriptor,
                )
                try:
                    child_metadata = os.fstat(child)
                finally:
                    os.close(child)
                if (
                    not stat.S_ISREG(child_metadata.st_mode)
                    or child_metadata.st_uid != os.geteuid()
                    or child_metadata.st_dev != metadata.st_dev
                    or child_metadata.st_mode & 0o022
                ):
                    raise OutcomeAcquisitionError(
                        "incomplete pending transaction has an unsafe entry"
                    )
                children[name] = child_metadata
                inode = (child_metadata.st_dev, child_metadata.st_ino)
                inode_counts[inode] = inode_counts.get(inode, 0) + 1
            if any(
                child.st_nlink
                != inode_counts[(child.st_dev, child.st_ino)]
                for child in children.values()
            ):
                raise OutcomeAcquisitionError(
                    "incomplete pending transaction has an external hard link"
                )
            if remove:
                for name in entries:
                    os.unlink(name, dir_fd=descriptor)
                os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if remove:
            os.rmdir(directory.name, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)


def _scan_transactions(
    *,
    raw_root: Path,
    specs: Sequence[Mapping[str, Any]],
    starts: Mapping[int, Mapping[str, Any]],
    results: Mapping[int, Mapping[str, Any]],
) -> tuple[
    dict[str, tuple[bytes, dict[str, Any], Path, Path]], list[str]
]:
    specs_by_sha = {str(spec["request_sha256"]): spec for spec in specs}
    provider_root = raw_root / CONFIRMATORY_NWIS_PROVIDER
    allowed_raw_names = {CONFIRMATORY_NWIS_PROVIDER, "snapshot_index.json"}
    if os.path.lexists(raw_root):
        safe_temps = _safe_atomic_temp_names(
            raw_root,
            allowed_final=_named_atomic_temp_parser({"snapshot_index.json"}),
            remove=False,
        )
        extras = {
            path.name for path in raw_root.iterdir()
        } - allowed_raw_names - safe_temps
        if extras:
            raise OutcomeAcquisitionError(
                f"raw NWIS root contains extraneous entries: {sorted(extras)}"
            )
    if os.path.lexists(provider_root):
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
        if not os.path.lexists(directory):
            missing.append(request_sha)
            continue
        complete[request_sha] = _validate_transaction(
            directory=directory,
            spec=spec,
            starts=starts,
            results=results,
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
    if os.path.lexists(directory) or os.path.lexists(pending):
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
    if os.path.lexists(directory):
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
        # Durably record both halves of this cross-directory rename.  If the
        # process dies earlier, the response is not claimed as durable and may
        # be requested again under the frozen ledger.
        os.fsync(pending_descriptor)
        os.fsync(provider_descriptor)
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
    for request_sha, transaction in transactions.items():
        attempt_number = transaction[1].get("attempt_number")
        if not isinstance(attempt_number, int):
            raise OutcomeAcquisitionError(
                "canonical transaction has a malformed attempt number"
            )
        result = results.get(attempt_number)
        if (
            result is None
            or request_sha not in result["completed_request_sha256"]
        ):
            raise OutcomeAcquisitionError(
                "canonical transaction is not closed by its attempt result"
            )
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


def _acquisition_directory(state: Mapping[str, Path]) -> Path:
    keys = (
        "acquisition_request_map",
        "temporal_outcomes",
        "external_outcomes",
        "acquisition_manifest",
    )
    if set(keys) - set(state):
        raise OutcomeAcquisitionError("acquisition bundle state is incomplete")
    parents = {state[key].parent for key in keys}
    if len(parents) != 1:
        raise OutcomeAcquisitionError(
            "acquisition bundle paths do not share one canonical directory"
        )
    directory = next(iter(parents))
    if directory.parent != state["run_directory"]:
        raise OutcomeAcquisitionError(
            "acquisition bundle is outside the canonical run directory"
        )
    return directory


def _acquisition_state_at_directory(
    state: Mapping[str, Path], directory: Path
) -> dict[str, Path]:
    canonical = _acquisition_directory(state)
    directory = Path(os.path.abspath(os.fspath(directory)))
    if directory.parent != canonical.parent:
        raise OutcomeAcquisitionError(
            "acquisition stage is not a same-filesystem sibling"
        )
    staged = dict(state)
    for key in (
        "acquisition_request_map",
        "temporal_outcomes",
        "external_outcomes",
        "acquisition_manifest",
    ):
        staged[key] = directory / state[key].name
    return staged


def _assert_owner_controlled_directory(descriptor: int) -> os.stat_result:
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o022
    ):
        raise OutcomeAcquisitionError(
            "acquisition publication parent is not owner-controlled"
        )
    return metadata


def _remove_safe_abandoned_stage(
    *,
    parent_descriptor: int,
    name: str,
    parent_metadata: os.stat_result,
    remove: bool,
) -> None:
    descriptor = os.open(
        name,
        _DIRECTORY_OPEN_FLAGS,
        dir_fd=parent_descriptor,
    )
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_dev != parent_metadata.st_dev
            or metadata.st_mode & 0o077
        ):
            raise OutcomeAcquisitionError(
                "abandoned acquisition stage has unsafe metadata"
            )
        entries = os.listdir(descriptor)
        file_metadata: dict[str, os.stat_result] = {}
        inode_counts: dict[tuple[int, int], int] = {}
        for entry in entries:
            child = os.open(
                entry,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=descriptor,
            )
            try:
                child_metadata = os.fstat(child)
            finally:
                os.close(child)
            if (
                not stat.S_ISREG(child_metadata.st_mode)
                or child_metadata.st_uid != os.geteuid()
                or child_metadata.st_dev != metadata.st_dev
                or child_metadata.st_mode & 0o222
            ):
                raise OutcomeAcquisitionError(
                    "abandoned acquisition stage contains an unsafe entry"
                )
            file_metadata[entry] = child_metadata
            inode = (child_metadata.st_dev, child_metadata.st_ino)
            inode_counts[inode] = inode_counts.get(inode, 0) + 1
        if any(
            child.st_nlink
            != inode_counts[(child.st_dev, child.st_ino)]
            for child in file_metadata.values()
        ):
            raise OutcomeAcquisitionError(
                "abandoned acquisition stage contains an external hard link"
            )
        if remove:
            for entry in entries:
                os.unlink(entry, dir_fd=descriptor)
            os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if remove:
        os.rmdir(name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)


def _cleanup_abandoned_acquisition_stages(state: Mapping[str, Path]) -> None:
    run_directory = state["run_directory"]
    with _open_directory_chain(run_directory, create=True) as parent_descriptor:
        parent_metadata = _assert_owner_controlled_directory(parent_descriptor)
        for name in sorted(os.listdir(parent_descriptor)):
            if not name.startswith(_ACQUISITION_STAGE_PREFIX):
                continue
            if not re.fullmatch(r"\.acquisition-stage-v1-[0-9a-f]{32}", name):
                raise OutcomeAcquisitionError(
                    "acquisition staging namespace contains a noncanonical entry"
                )
            _remove_safe_abandoned_stage(
                parent_descriptor=parent_descriptor,
                name=name,
                parent_metadata=parent_metadata,
                remove=True,
            )


def _validate_abandoned_acquisition_stages(
    state: Mapping[str, Path]
) -> None:
    run_directory = state["run_directory"]
    if not os.path.lexists(run_directory):
        return
    with _open_directory_chain(
        run_directory, create=False
    ) as parent_descriptor:
        parent_metadata = _assert_owner_controlled_directory(parent_descriptor)
        for name in sorted(os.listdir(parent_descriptor)):
            if not name.startswith(_ACQUISITION_STAGE_PREFIX):
                continue
            if not re.fullmatch(r"\.acquisition-stage-v1-[0-9a-f]{32}", name):
                raise OutcomeAcquisitionError(
                    "acquisition staging namespace contains a noncanonical entry"
                )
            _remove_safe_abandoned_stage(
                parent_descriptor=parent_descriptor,
                name=name,
                parent_metadata=parent_metadata,
                remove=False,
            )


def _new_acquisition_stage_directory(state: Mapping[str, Path]) -> Path:
    canonical = _acquisition_directory(state)
    with _open_directory_chain(
        state["run_directory"], create=True
    ) as parent_descriptor:
        parent_metadata = _assert_owner_controlled_directory(parent_descriptor)
        for _attempt in range(128):
            name = f"{_ACQUISITION_STAGE_PREFIX}{secrets.token_hex(16)}"
            try:
                os.mkdir(name, 0o700, dir_fd=parent_descriptor)
            except FileExistsError:
                continue
            descriptor = os.open(
                name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_descriptor
            )
            try:
                metadata = os.fstat(descriptor)
                if (
                    not stat.S_ISDIR(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or metadata.st_dev != parent_metadata.st_dev
                    or metadata.st_mode & 0o077
                ):
                    raise OutcomeAcquisitionError(
                        "new acquisition stage has unsafe metadata"
                    )
                os.fsync(descriptor)
                os.fsync(parent_descriptor)
            finally:
                os.close(descriptor)
            return canonical.parent / name
    raise OutcomeAcquisitionError("cannot allocate an acquisition stage")


def _assert_exact_acquisition_directory(
    directory: Path,
    state: Mapping[str, Path],
    *,
    allow_recoverable_canonical_mode: bool = False,
) -> None:
    canonical = _acquisition_directory(state)
    expected = {
        state[key].name
        for key in (
            "acquisition_request_map",
            "temporal_outcomes",
            "external_outcomes",
            "acquisition_manifest",
        )
    }
    directory = Path(os.path.abspath(os.fspath(directory)))
    if directory != canonical and directory.parent != canonical.parent:
        raise OutcomeAcquisitionError("acquisition directory is noncanonical")
    is_canonical = directory == canonical
    with _open_directory_chain(directory, create=False) as descriptor:
        directory_metadata = os.fstat(descriptor)
        actual_mode = stat.S_IMODE(directory_metadata.st_mode)
        allowed_modes = (
            {0o555, 0o700}
            if is_canonical and allow_recoverable_canonical_mode
            else {0o555}
            if is_canonical
            else {0o700}
        )
        if (
            not stat.S_ISDIR(directory_metadata.st_mode)
            or directory_metadata.st_uid != os.geteuid()
            or actual_mode not in allowed_modes
            or set(os.listdir(descriptor)) != expected
        ):
            raise OutcomeAcquisitionError(
                "acquisition bundle layout or ownership changed"
            )
        for name in expected:
            child = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=descriptor,
            )
            try:
                metadata = os.fstat(child)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or metadata.st_dev != directory_metadata.st_dev
                    or metadata.st_nlink != 1
                    or metadata.st_mode & 0o222
                ):
                    raise OutcomeAcquisitionError(
                        "acquisition bundle contains an unsafe artifact"
                    )
            finally:
                os.close(child)


def _acquisition_directory_mode(state: Mapping[str, Path]) -> int:
    canonical = _acquisition_directory(state)
    with _open_directory_chain(canonical, create=False) as descriptor:
        return stat.S_IMODE(os.fstat(descriptor).st_mode)


def _harden_recoverable_acquisition_directory(
    state: Mapping[str, Path],
) -> None:
    """Finish only the rename-before-chmod crash state after full replay."""
    canonical = _acquisition_directory(state)
    _assert_exact_acquisition_directory(
        canonical, state, allow_recoverable_canonical_mode=True
    )
    if _acquisition_directory_mode(state) != 0o700:
        raise OutcomeAcquisitionError(
            "acquisition permission recovery requires exact mode 0700"
        )
    with _open_directory_chain(
        state["run_directory"], create=False
    ) as parent_descriptor:
        descriptor = os.open(
            canonical.name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_descriptor
        )
        try:
            os.fchmod(descriptor, 0o555)
            os.fsync(descriptor)
            os.fsync(parent_descriptor)
        finally:
            os.close(descriptor)
    _assert_exact_acquisition_directory(canonical, state)


def _logical_binding(
    root: Path, *, physical_path: Path, canonical_path: Path
) -> dict[str, str]:
    physical = assert_no_symlink_components(
        root, physical_path, require_file=True
    )
    _require_immutable_atomic_final(
        physical, label="staged acquisition artifact"
    )
    canonical = assert_no_symlink_components(root, canonical_path)
    return {
        "path": canonical.relative_to(root).as_posix(),
        "sha256": sha256_file(physical),
    }


def _acquisition_publication_fault(_point: str) -> None:
    """No-op production hook replaced only by crash tests."""


def _publish_acquisition_directory(
    stage: Path, state: Mapping[str, Path]
) -> Path:
    canonical = _acquisition_directory(state)
    if re.fullmatch(r"\.acquisition-stage-v1-[0-9a-f]{32}", stage.name) is None:
        raise OutcomeAcquisitionError(
            "acquisition staging directory name is noncanonical"
        )
    _assert_exact_acquisition_directory(stage, state)
    with _open_directory_chain(
        state["run_directory"], create=False
    ) as parent_descriptor:
        parent_metadata = _assert_owner_controlled_directory(parent_descriptor)
        stage_descriptor = os.open(
            stage.name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_descriptor
        )
        try:
            stage_metadata = os.fstat(stage_descriptor)
            if (
                stage_metadata.st_dev != parent_metadata.st_dev
                or stage_metadata.st_uid != os.geteuid()
                or stage_metadata.st_mode & 0o077
            ):
                raise OutcomeAcquisitionError(
                    "acquisition stage changed before publication"
                )
            try:
                os.stat(
                    canonical.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise OutcomeAcquisitionError(
                    "canonical acquisition bundle already exists"
                )
            os.fsync(stage_descriptor)
            _acquisition_publication_fault("before_directory_rename")
            os.rename(
                stage.name,
                canonical.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            _acquisition_publication_fault(
                "after_directory_rename_before_hardening"
            )
            os.fchmod(stage_descriptor, 0o555)
            os.fsync(stage_descriptor)
            _acquisition_publication_fault(
                "after_directory_hardening_before_parent_fsync"
            )
            os.fsync(parent_descriptor)
            _acquisition_publication_fault("after_directory_rename")
        finally:
            os.close(stage_descriptor)
    _assert_exact_acquisition_directory(canonical, state)
    return canonical


def _forbidden_resume_outputs(
    state: Mapping[str, Path]
) -> list[str]:
    missing = set(RAW_ACQUISITION_FORBIDDEN_STATE_KEYS) - set(state)
    if missing:
        raise OutcomeAcquisitionError(
            f"raw forbidden-state registry is incomplete: {sorted(missing)}"
        )
    forbidden = {
        key
        for key in RAW_ACQUISITION_FORBIDDEN_STATE_KEYS
        if os.path.lexists(state[key])
    }
    trusted_parents = {state[key].parent for key in TRUSTED_STATE_KEYS}
    if len(trusted_parents) != 1:
        raise OutcomeAcquisitionError(
            "trusted output paths do not share one canonical directory"
        )
    trusted_directory = next(iter(trusted_parents))
    if os.path.lexists(trusted_directory):
        forbidden.add("trusted_directory")
    acquisition_parents = {
        state[key].parent
        for key in (
            "acquisition_manifest",
            "acquisition_request_map",
            "temporal_outcomes",
            "external_outcomes",
        )
    }
    if len(acquisition_parents) != 1:
        raise OutcomeAcquisitionError(
            "acquisition outputs do not share one canonical directory"
        )
    acquisition_directory = next(iter(acquisition_parents))
    if os.path.lexists(acquisition_directory):
        forbidden.add("acquisition_directory")
    return sorted(forbidden)


def _unexpected_transport_namespace_entries(
    *, state: Mapping[str, Path], evidence: Mapping[str, Path]
) -> list[str]:
    acquisition_root = evidence["root"]
    if not os.path.lexists(acquisition_root):
        return []
    if os.path.lexists(evidence["lock"]):
        parent = os.lstat(acquisition_root)
        lock = os.lstat(evidence["lock"])
        if (
            not stat.S_ISREG(lock.st_mode)
            or lock.st_uid != os.geteuid()
            or lock.st_dev != parent.st_dev
            or lock.st_nlink != 1
            or lock.st_mode & 0o077
        ):
            raise OutcomeAcquisitionError(
                "raw transport lock has unsafe metadata"
            )
    allowed = {
        state["raw_nwis_root"].name,
        evidence["request_ledger"].name,
        evidence["attempts_root"].name,
        evidence["attempt_index"].name,
        evidence["lock"].name,
    }
    safe_temps = _safe_atomic_temp_names(
        acquisition_root,
        allowed_final=_named_atomic_temp_parser({
            evidence["request_ledger"].name,
            evidence["attempt_index"].name,
        }),
        remove=False,
    )
    return sorted(
        path.name
        for path in acquisition_root.iterdir()
        if path.name not in allowed and path.name not in safe_temps
    )


def inspect_transport_resume_state(
    *,
    root: Path,
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
    if not os.path.lexists(evidence["request_ledger"]):
        if (
            os.path.lexists(evidence["attempts_root"])
            or os.path.lexists(evidence["attempt_index"])
            or (os.path.lexists(raw_root) and any(raw_root.iterdir()))
        ):
            raise OutcomeAcquisitionError(
                "raw evidence exists without the frozen request ledger"
            )
        return {
            "classification": "RESUMABLE_BEFORE_REQUEST_LEDGER_PUBLICATION",
            "completed_request_count": 0,
            "missing_request_count": int(expected_ledger["request_count"]),
            "recoverable_pending_request_count": 0,
            "refetchable_nondurable_response_count": 0,
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
    starts, results = _load_attempt_history(
        attempts_root=evidence["attempts_root"],
        opening_id=str(work_order["opening_id"]),
        authorization_sha256=str(work_order["authorization_sha256"]),
        work_order_self_sha256=str(work_order["work_order_self_sha256"]),
        request_ledger_sha256=sha256_file(evidence["request_ledger"]),
        request_ids=request_ids,
    )
    pending_ids: list[str] = []
    refetchable_pending_ids: list[str] = []
    pending_root = raw_root / CONFIRMATORY_NWIS_PROVIDER / ".pending"
    if os.path.lexists(pending_root):
        if not pending_root.is_dir() or pending_root.is_symlink():
            raise OutcomeAcquisitionError("pending NWIS root is malformed")
        for pending in sorted(pending_root.iterdir()):
            request_sha = pending.name
            spec = specs_by_sha.get(request_sha)
            canonical = pending.parent.parent / request_sha
            if spec is None or os.path.lexists(canonical):
                raise OutcomeAcquisitionError(
                    "pending NWIS transaction is unknown or duplicated"
                )
            try:
                _validate_transaction(
                    directory=pending,
                    spec=spec,
                    starts=starts,
                    results=results,
                )
            except OutcomeAcquisitionError:
                _remove_safe_pending_transaction(pending, remove=False)
                refetchable_pending_ids.append(request_sha)
            else:
                pending_ids.append(request_sha)
    complete, missing = _scan_transactions(
        raw_root=raw_root,
        specs=specs,
        starts=starts,
        results=results,
    )
    if not (set(pending_ids) | set(refetchable_pending_ids)) <= set(missing):
        raise OutcomeAcquisitionError("pending/canonical transaction registry changed")
    snapshot_index = state["raw_nwis_snapshot_index"]
    if missing and os.path.lexists(snapshot_index):
        raise OutcomeAcquisitionError(
            "derived raw index appeared before every transaction"
        )
    if missing and os.path.lexists(evidence["attempt_index"]):
        raise OutcomeAcquisitionError(
            "final attempt index appeared before every raw transaction"
    )
    if not missing:
        index_rows: list[dict[str, Any]] = []
        for spec in specs:
            request_sha = str(spec["request_sha256"])
            payload, metadata, response_path, metadata_path = complete[request_sha]
            series_registry = nwis_confirmatory_series_registry(payload)
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
        expected_index = canonical_json_bytes({
            "schema_version": 1,
            "snapshot_count": len(index_rows),
            "records": index_rows,
        })
        if os.path.lexists(snapshot_index) and (
            not snapshot_index.is_file()
            or snapshot_index.is_symlink()
            or _read_regular_file_no_follow(snapshot_index) != expected_index
        ):
            raise OutcomeAcquisitionError(
                "existing NWIS snapshot index differs from raw replay"
            )
        if os.path.lexists(evidence["attempt_index"]):
            expected_attempt_index = _attempt_index(
                root=root,
                attempts_root=evidence["attempts_root"],
                work_order=work_order,
                request_ledger_path=evidence["request_ledger"],
                transactions=complete,
            )
            if _read_regular_file_no_follow(
                evidence["attempt_index"]
            ) != canonical_json_bytes(expected_attempt_index):
                raise OutcomeAcquisitionError(
                    "existing transport-attempt index differs from raw replay"
                )
    effective_missing = set(missing) - set(pending_ids)
    classification = (
        "RESUMABLE_RAW_COMPLETE_DERIVATION_NOT_PUBLISHED"
        if not missing
        else "RESUMABLE_RECOVERABLE_PENDING_BYTES"
        if pending_ids
        else "RESUMABLE_NON_DURABLE_RESPONSE_REFETCH"
        if refetchable_pending_ids
        else "RESUMABLE_MISSING_REQUESTS"
    )
    return {
        "classification": classification,
        "completed_request_count": len(complete),
        "missing_request_count": len(effective_missing),
        "recoverable_pending_request_count": len(pending_ids),
        "refetchable_nondurable_response_count": len(
            refetchable_pending_ids
        ),
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
    forbidden = _forbidden_resume_outputs(state)
    if forbidden:
        raise OutcomeAcquisitionError(
            "raw acquisition cannot continue after derived/trusted publication: "
            f"{forbidden}"
        )
    with _exclusive_transport_lock(evidence["lock"]):
        forbidden = _forbidden_resume_outputs(state)
        if forbidden:
            raise OutcomeAcquisitionError(
                "raw acquisition cannot continue after derived/trusted publication: "
                f"{forbidden}"
            )
        _cleanup_abandoned_acquisition_stages(state)
        _safe_atomic_temp_names(
            evidence["root"],
            allowed_final=_named_atomic_temp_parser({
                evidence["request_ledger"].name,
                evidence["attempt_index"].name,
            }),
            remove=True,
        )
        if os.path.lexists(evidence["attempts_root"]):
            _safe_atomic_temp_names(
                evidence["attempts_root"],
                allowed_final=_attempt_atomic_temp_parser,
                remove=True,
            )
        if os.path.lexists(raw_root):
            _safe_atomic_temp_names(
                raw_root,
                allowed_final=_named_atomic_temp_parser({
                    "snapshot_index.json"
                }),
                remove=True,
            )
        # Recheck after safe crash-remnant cleanup, immediately before reading
        # or extending transport state.  The raw child never mutates transport
        # after any canonical derived/trusted publication is visible.
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
        if not resume and os.path.lexists(raw_root):
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
            results=results,
        )
        provider_root = raw_root / CONFIRMATORY_NWIS_PROVIDER
        if os.path.lexists(provider_root):
            if not provider_root.is_dir() or provider_root.is_symlink():
                raise OutcomeAcquisitionError("NWIS provider root is malformed")
            _fsync_directory(provider_root)
        pending_root = provider_root / ".pending"
        if os.path.lexists(pending_root):
            _fsync_directory(pending_root)
        complete, missing = _scan_transactions(
            raw_root=raw_root,
            specs=specs,
            starts=starts,
            results=results,
        )
        if missing and os.path.lexists(state["raw_nwis_snapshot_index"]):
            raise OutcomeAcquisitionError(
                "derived raw index appeared before every transaction"
            )
        if missing and os.path.lexists(evidence["attempt_index"]):
            raise OutcomeAcquisitionError(
                "final attempt index appeared before every raw transaction"
            )
        open_attempts = sorted(set(starts) - set(results))
        if missing and open_attempts:
            interrupted = open_attempts[0]
            _write_attempt_result(
                attempts_root=evidence["attempts_root"],
                attempt_number=interrupted,
                attempt_start_path=(
                    evidence["attempts_root"]
                    / f"attempt_{interrupted:06d}_start.json"
                ),
                work_order=work_order,
                request_ledger_sha256=request_ledger_sha256,
                completed=complete,
                missing=missing,
                status="TRANSPORT_INCOMPLETE_MAY_RESUME_SAME_OPENING",
                failure_class="PREVIOUS_PROCESS_TERMINATED",
            )
            results[interrupted] = _self_hashed_document(
                evidence["attempts_root"]
                / f"attempt_{interrupted:06d}_result.json",
                label="reconstructed interrupted transport result",
                self_field="attempt_result_self_sha256",
                format_name=ACQUISITION_ATTEMPT_RESULT_FORMAT,
            )
            open_attempts = []

        active_attempt: int | None = None
        active_start_path: Path | None = None
        if missing:
            active_attempt = len(starts) + 1
            active_start_path, start_document = _write_attempt_start(
                attempts_root=evidence["attempts_root"],
                attempt_number=active_attempt,
                resume=resume,
                work_order=work_order,
                request_ledger_sha256=request_ledger_sha256,
                completed=complete,
                missing=missing,
            )
            starts = {**starts, active_attempt: start_document}
            try:
                specs_by_sha = {
                    str(spec["request_sha256"]): spec for spec in specs
                }
                # This complete replay is deliberately adjacent to the first
                # possible socket operation.
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
                        attempt_number=active_attempt,
                    )
            except Exception as exc:
                try:
                    # If transport failed after both immutable pending files
                    # were durable but before their directory rename, promote
                    # them before sealing this result.  The next start can then
                    # exactly equal this immutable result partition.
                    _recover_complete_pending_transactions(
                        raw_root=raw_root,
                        specs_by_sha=specs_by_sha,
                        starts=starts,
                        results=results,
                    )
                    if os.path.lexists(provider_root):
                        if (
                            not provider_root.is_dir()
                            or provider_root.is_symlink()
                        ):
                            raise OutcomeAcquisitionError(
                                "NWIS provider root is malformed"
                            )
                        _fsync_directory(provider_root)
                    if os.path.lexists(pending_root):
                        _fsync_directory(pending_root)
                    completed_after, missing_after = _scan_transactions(
                        raw_root=raw_root,
                        specs=specs,
                        starts=starts,
                        results=results,
                    )
                    transport_complete = not missing_after
                    _write_attempt_result(
                        attempts_root=evidence["attempts_root"],
                        attempt_number=active_attempt,
                        attempt_start_path=active_start_path,
                        work_order=work_order,
                        request_ledger_sha256=request_ledger_sha256,
                        completed=completed_after,
                        missing=missing_after,
                        status=(
                            "ALL_LEDGER_TRANSACTIONS_COMPLETE"
                            if transport_complete
                            else "TRANSPORT_INCOMPLETE_MAY_RESUME_SAME_OPENING"
                        ),
                        failure_class=(
                            None if transport_complete else type(exc).__name__
                        ),
                    )
                except Exception:
                    # Unsafe/corrupt canonical evidence remains indeterminate.
                    pass
                raise OutcomeAcquisitionError(
                    "fixed-ledger NWIS transport did not complete"
                ) from exc

        complete, missing = _scan_transactions(
            raw_root=raw_root,
            specs=specs,
            starts=starts,
            results=results,
        )
        if missing or set(complete) != request_ids:
            if active_attempt is not None and active_start_path is not None:
                _write_attempt_result(
                    attempts_root=evidence["attempts_root"],
                    attempt_number=active_attempt,
                    attempt_start_path=active_start_path,
                    work_order=work_order,
                    request_ledger_sha256=request_ledger_sha256,
                    completed=complete,
                    missing=missing,
                    status="TRANSPORT_INCOMPLETE_MAY_RESUME_SAME_OPENING",
                    failure_class="INCOMPLETE_LEDGER",
                )
            raise OutcomeAcquisitionError("not every frozen request is complete")

        # Complete a result only from the exact validated request-key set and
        # exact immutable transactions.  No count-only repair is permitted.
        _acquisition_transport_fault("after_complete_transaction_replay")
        open_attempts = sorted(set(starts) - set(results))
        if open_attempts:
            completed_attempt = open_attempts[0]
            completed_start = (
                evidence["attempts_root"]
                / f"attempt_{completed_attempt:06d}_start.json"
            )
        elif (
            not results
            or results[max(results)].get("status")
            != "ALL_LEDGER_TRANSACTIONS_COMPLETE"
        ):
            if results:
                # An immutable prior result must be the exact partition for
                # the next start.  Complete canonical transactions alongside
                # a prior result that still calls them missing are therefore
                # contradictory evidence, not grounds for a synthetic jump.
                raise OutcomeAcquisitionError(
                    "complete transactions contradict the latest attempt result"
                )
            completed_attempt = len(starts) + 1
            completed_start, start_document = _write_attempt_start(
                attempts_root=evidence["attempts_root"],
                attempt_number=completed_attempt,
                resume=True,
                work_order=work_order,
                request_ledger_sha256=request_ledger_sha256,
                completed=complete,
                missing=(),
            )
            starts[completed_attempt] = start_document
        else:
            completed_attempt = None
            completed_start = None
        if completed_attempt is not None and completed_start is not None:
            _write_attempt_result(
                attempts_root=evidence["attempts_root"],
                attempt_number=completed_attempt,
                attempt_start_path=completed_start,
                work_order=work_order,
                request_ledger_sha256=request_ledger_sha256,
                completed=complete,
                missing=(),
                status="ALL_LEDGER_TRANSACTIONS_COMPLETE",
                failure_class=None,
            )

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
        index_path = state["raw_nwis_snapshot_index"]
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
        request_map_bytes = canonical_json_bytes({
            "format": ACQUISITION_REQUEST_MAP_FORMAT,
            "opening_id": authorization["opening_id"],
            "authorization_sha256": work_order["authorization_sha256"],
            "provider": CONFIRMATORY_NWIS_PROVIDER,
            "request_count": len(request_rows),
            "requests": request_rows,
        })

        normalized_frames: dict[str, pd.DataFrame] = {}
        for cohort in ("temporal", "external"):
            combined = pd.concat(frames[cohort], ignore_index=True)
            combined["site_no"] = combined.site_no.astype("string")
            combined["DATE"] = pd.to_datetime(combined.DATE)
            combined = combined.sort_values(["site_no", "DATE"]).reset_index(
                drop=True
            )
            normalized_frames[cohort] = combined

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

        stage = _new_acquisition_stage_directory(state)
        staged_state = _acquisition_state_at_directory(state, stage)
        _create_bytes(
            staged_state["acquisition_request_map"], request_map_bytes
        )
        for cohort, key in (
            ("temporal", "temporal_outcomes"),
            ("external", "external_outcomes"),
        ):
            _create_parquet(staged_state[key], normalized_frames[cohort])
        manifest = {
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
            "request_map": _logical_binding(
                root,
                physical_path=staged_state["acquisition_request_map"],
                canonical_path=state["acquisition_request_map"],
            ),
            "normalized_outcome_tables": {
                cohort: _logical_binding(
                    root,
                    physical_path=staged_state[key],
                    canonical_path=state[key],
                )
                for cohort, key in (
                    ("temporal", "temporal_outcomes"),
                    ("external", "external_outcomes"),
                )
            },
            "producer_role": "RAW_ONLY_NO_PREDICTIONS_OR_STATISTICS",
        }
        _create_bytes(
            staged_state["acquisition_manifest"],
            canonical_json_bytes(manifest),
        )
        _acquisition_publication_fault("after_stage_generation")
        if (
            _read_regular_file_no_follow(
                staged_state["acquisition_request_map"]
            )
            != request_map_bytes
            or _read_regular_file_no_follow(
                staged_state["acquisition_manifest"]
            )
            != canonical_json_bytes(manifest)
        ):
            raise OutcomeAcquisitionError(
                "staged acquisition JSON differs from deterministic replay"
            )
        for cohort, key in (
            ("temporal", "temporal_outcomes"),
            ("external", "external_outcomes"),
        ):
            try:
                actual = pd.read_parquet(staged_state[key])
                pd.testing.assert_frame_equal(
                    actual,
                    normalized_frames[cohort],
                    check_dtype=False,
                    check_exact=True,
                )
            except (OSError, ValueError, AssertionError) as exc:
                raise OutcomeAcquisitionError(
                    f"staged {cohort} normalized outcomes differ from raw replay"
                ) from exc
        _assert_exact_acquisition_directory(stage, state)
        _acquisition_publication_fault("after_stage_validation")
        _publish_acquisition_directory(stage, state)
        return state["acquisition_manifest"]

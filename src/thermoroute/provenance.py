"""Immutable raw-HTTP snapshots for auditable environmental data acquisition.

The acquisition pipeline used to retain only parsed tables.  That is not enough
to distinguish a provider revision from a parser or code revision.  This module
stores the exact response bytes beside a canonical request document and records
retrieval time, HTTP metadata, byte count and SHA-256.

Snapshots are content addressed by the *request*.  A repeated request reuses and
verifies the stored response; it never silently overwrites it.  Deliberate data
refreshes therefore require a new snapshot root/version.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import time
from typing import Any, Callable, Iterator, Mapping
import urllib.request


class ProvenanceError(RuntimeError):
    """Raised when a raw snapshot is missing, corrupt, or cannot be acquired."""


def canonical_json_bytes(value: object) -> bytes:
    """Stable UTF-8 JSON representation used for request fingerprints."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_canonical_utc(value: object, *, label: str) -> str:
    """Require the one UTC spelling emitted by :func:`utc_now_iso`.

    ``datetime.fromisoformat`` deliberately accepts several equivalent forms
    (a space in place of ``T``, ``Z``, and shortened fractional seconds).  A
    frozen evidence chain must not: the timestamp is an identity-bearing byte
    string, so one instant has exactly one accepted producer representation.
    """
    if type(value) is not str:
        raise ProvenanceError(f"{label} must be a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ProvenanceError(f"{label} must be a canonical UTC timestamp") from exc
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != timedelta(0)
        or parsed.isoformat() != value
        or "T" not in value
        or not value.endswith("+00:00")
    ):
        raise ProvenanceError(f"{label} must be a canonical UTC timestamp")
    return value


def utc_now_iso() -> str:
    """Return the canonical UTC spelling used in immutable evidence."""
    return datetime.now(timezone.utc).isoformat()


def strict_canonical_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    """Decode one duplicate-free, finite, canonical producer JSON object."""
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
        raise ProvenanceError(f"{label} is not strict JSON") from exc
    if not isinstance(document, dict) or payload != canonical_json_bytes(document):
        raise ProvenanceError(f"{label} is not canonical producer JSON")
    return document


def _secure_flags(*names: str) -> int:
    missing = [name for name in names if not hasattr(os, name)]
    if missing:
        raise ProvenanceError(
            f"platform lacks fail-closed filesystem flags: {', '.join(missing)}"
        )
    flags = 0
    for name in names:
        flags |= int(getattr(os, name))
    return flags


def _normalise_under_root(path: str | Path, trusted_root: str | Path) -> tuple[Path, Path]:
    root = Path(os.path.abspath(trusted_root))
    candidate = Path(os.path.abspath(path))
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ProvenanceError(f"path escapes trusted root: {candidate}") from exc
    if not relative.parts:
        raise ProvenanceError(f"path names the trusted root, not a file: {candidate}")
    return root, relative


def _open_parent_fd(
    path: str | Path,
    *,
    trusted_root: str | Path,
    create_parents: bool,
) -> tuple[int, str]:
    """Pin every directory component and return ``(parent_fd, basename)``."""
    root, relative = _normalise_under_root(path, trusted_root)
    directory_flags = _secure_flags(
        "O_RDONLY", "O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC"
    )
    try:
        current = os.open(root, directory_flags)
    except OSError as exc:
        raise ProvenanceError(f"trusted root is absent or unsafe: {root}") from exc
    try:
        for component in relative.parts[:-1]:
            try:
                child = os.open(component, directory_flags, dir_fd=current)
            except FileNotFoundError:
                if not create_parents:
                    raise
                try:
                    os.mkdir(component, 0o755, dir_fd=current)
                except FileExistsError:
                    pass
                os.fsync(current)
                child = os.open(component, directory_flags, dir_fd=current)
            metadata = os.fstat(child)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(child)
                raise ProvenanceError(
                    f"path parent is not a real directory: {component}"
                )
            os.close(current)
            current = child
        return current, relative.parts[-1]
    except BaseException:
        os.close(current)
        raise


def read_single_link_regular(
    path: str | Path,
    *,
    label: str,
    trusted_root: str | Path,
) -> bytes:
    """Read a regular single-link file from one pinned, no-follow descriptor."""
    parent_fd, name = _open_parent_fd(
        path, trusted_root=trusted_root, create_parents=False
    )
    descriptor = -1
    try:
        flags = _secure_flags(
            "O_RDONLY", "O_NOFOLLOW", "O_CLOEXEC", "O_NONBLOCK"
        )
        try:
            descriptor = os.open(name, flags, dir_fd=parent_fd)
        except OSError as exc:
            raise ProvenanceError(
                f"{label} is absent or not a regular single-link file: {path}"
            ) from exc
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ProvenanceError(
                f"{label} is not a regular single-link file: {path}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        identity_fields = (
            "st_dev", "st_ino", "st_mode", "st_nlink", "st_uid", "st_gid",
            "st_size", "st_mtime_ns", "st_ctime_ns",
        )
        if (
            any(getattr(before, field) != getattr(after, field) for field in identity_fields)
            or len(payload) != before.st_size
        ):
            raise ProvenanceError(f"{label} changed while it was read: {path}")
        try:
            linked = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as exc:
            raise ProvenanceError(f"{label} changed while it was read: {path}") from exc
        if (
            not stat.S_ISREG(linked.st_mode)
            or linked.st_nlink != 1
            or (linked.st_dev, linked.st_ino) != (after.st_dev, after.st_ino)
        ):
            raise ProvenanceError(f"{label} changed while it was read: {path}")
        return payload
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def create_single_link_regular(
    path: str | Path,
    payload: bytes,
    *,
    trusted_root: str | Path,
    mode: int = 0o444,
) -> None:
    """Create one immutable file through a pinned parent and fsync its entry."""
    parent_fd, name = _open_parent_fd(
        path, trusted_root=trusted_root, create_parents=True
    )
    descriptor = -1
    try:
        flags = _secure_flags(
            "O_WRONLY", "O_CREAT", "O_EXCL", "O_NOFOLLOW", "O_CLOEXEC"
        )
        descriptor = os.open(name, flags, mode, dir_fd=parent_fd)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ProvenanceError(f"short write while publishing {path}")
            view = view[written:]
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != len(payload)
        ):
            raise ProvenanceError(f"published file identity changed: {path}")
        os.fsync(parent_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


@dataclass(frozen=True)
class AdvisoryLockHandle:
    fd: int
    token: str


@contextmanager
def candidate_publication_lock(
    path: str | Path,
    *,
    trusted_root: str | Path,
    shared: bool = False,
    inherited_fd: int | None = None,
    inherited_token: str | None = None,
) -> Iterator[AdvisoryLockHandle]:
    """Lock one existing fixed regular-file anchor, including across a child.

    An inherited descriptor is a duplicate of the parent's open-file
    description.  The child verifies its inode and lock state, then only closes
    its duplicate; it never issues ``LOCK_UN`` and therefore cannot release the
    parent's transaction early.  This serializes cooperating local processes;
    it is not an external timestamp, independent custodian, or defense against
    a hostile repository/OS owner.
    """
    operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
    descriptor = -1
    owns_unlock = inherited_fd is None
    parent_fd = -1
    try:
        parent_fd, name = _open_parent_fd(
            path, trusted_root=trusted_root, create_parents=False
        )
        if inherited_fd is None:
            flags = _secure_flags("O_RDONLY", "O_NOFOLLOW", "O_CLOEXEC")
            descriptor = os.open(name, flags, dir_fd=parent_fd)
            token = secrets.token_hex(32)
        else:
            if (
                type(inherited_fd) is not int
                or inherited_fd < 0
                or not isinstance(inherited_token, str)
                or re.fullmatch(r"[0-9a-f]{64}", inherited_token) is None
            ):
                raise ProvenanceError("inherited publication-lock capability is malformed")
            descriptor = os.dup(inherited_fd)
            token = inherited_token
        metadata = os.fstat(descriptor)
        linked = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.getuid()
            or not stat.S_ISREG(linked.st_mode)
            or linked.st_nlink != 1
            or (linked.st_dev, linked.st_ino) != (metadata.st_dev, metadata.st_ino)
        ):
            raise ProvenanceError("candidate publication lock is linked or unsafe")
        try:
            fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProvenanceError("candidate publication transaction is busy") from exc
        linked_after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (linked_after.st_dev, linked_after.st_ino) != (
            metadata.st_dev, metadata.st_ino
        ):
            raise ProvenanceError("candidate publication lock entry changed")
        try:
            yield AdvisoryLockHandle(fd=descriptor, token=token)
        finally:
            descriptor_after = os.fstat(descriptor)
            linked_exit = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(descriptor_after.st_mode)
                or descriptor_after.st_nlink != 1
                or descriptor_after.st_uid != os.getuid()
                or not stat.S_ISREG(linked_exit.st_mode)
                or linked_exit.st_nlink != 1
                or (linked_exit.st_dev, linked_exit.st_ino)
                != (descriptor_after.st_dev, descriptor_after.st_ino)
                or (descriptor_after.st_dev, descriptor_after.st_ino)
                != (metadata.st_dev, metadata.st_ino)
                or descriptor_after.st_size != metadata.st_size
                or descriptor_after.st_mtime_ns != metadata.st_mtime_ns
            ):
                raise ProvenanceError(
                    "candidate publication lock anchor changed during transaction"
                )
    finally:
        if descriptor >= 0:
            if owns_unlock:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError:
                    pass
            os.close(descriptor)
        if parent_fd >= 0:
            os.close(parent_fd)


def _atomic_write(path: Path, payload: bytes) -> None:
    """Write one complete artifact, then atomically publish it."""
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


def _exclusive_create(
    path: Path,
    payload: bytes,
    *,
    trusted_root: Path | None = None,
) -> None:
    """Create one immutable suffix without replacing any existing directory entry."""
    if trusted_root is not None:
        create_single_link_regular(
            path, payload, trusted_root=trusted_root, mode=0o444
        )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _read_single_link_regular(
    path: Path,
    *,
    label: str,
    trusted_root: Path | None = None,
) -> bytes:
    """Read exact bytes while rejecting links and path replacement."""
    if trusted_root is not None:
        return read_single_link_regular(
            path, label=label, trusted_root=trusted_root
        )
    try:
        before = path.lstat()
    except OSError as exc:
        raise ProvenanceError(f"{label} is absent or unsafe: {path}") from exc
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode) or before.st_nlink != 1:
        raise ProvenanceError(f"{label} is not a single-link regular file: {path}")
    payload = path.read_bytes()
    try:
        after = path.lstat()
    except OSError as exc:
        raise ProvenanceError(f"{label} changed while it was read: {path}") from exc
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_nlink)
        != (after.st_dev, after.st_ino, after.st_size, after.st_nlink)
        or len(payload) != before.st_size
    ):
        raise ProvenanceError(f"{label} changed while it was read: {path}")
    return payload


@dataclass(frozen=True)
class SnapshotRecord:
    provider: str
    request_sha256: str
    response_sha256: str
    response_path: Path
    metadata_path: Path
    retrieved_at_utc: str
    byte_count: int
    request: Mapping[str, Any]


class SnapshotStore:
    """Content-addressed store for immutable public API responses."""

    def __init__(
        self,
        root: str | Path,
        *,
        offline: bool = False,
        trusted_root: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.offline = offline
        self.trusted_root = Path(trusted_root) if trusted_root is not None else None

    @staticmethod
    def _provider_name(provider: str) -> str:
        name = re.sub(r"[^a-z0-9_.-]+", "-", provider.strip().lower()).strip("-")
        if not name:
            raise ValueError("provider must contain at least one safe character")
        return name

    @staticmethod
    def request_document(
        *, provider: str, url: str, method: str = "GET",
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        # Only declared request headers enter the fingerprint.  urllib defaults
        # can vary by Python version and are not part of the scientific query.
        return {
            "schema_version": 1,
            "provider": SnapshotStore._provider_name(provider),
            "method": method.upper(),
            "url": url,
            "headers": dict(sorted((headers or {}).items())),
        }

    def _paths(self, provider: str, request_sha256: str) -> tuple[Path, Path]:
        base = self.root / self._provider_name(provider) / request_sha256
        return base / "response.bin", base / "metadata.json"

    def _read_verified(
        self, response_path: Path, metadata_path: Path,
        request_sha256: str,
        *,
        expected_final_url: str | None = None,
    ) -> tuple[bytes, SnapshotRecord]:
        try:
            metadata_bytes = _read_single_link_regular(
                metadata_path,
                label="snapshot metadata",
                trusted_root=self.trusted_root,
            )
            payload = _read_single_link_regular(
                response_path,
                label="snapshot response",
                trusted_root=self.trusted_root,
            )
            meta = strict_canonical_json_object(
                metadata_bytes, label="snapshot metadata"
            )
        except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProvenanceError(f"incomplete snapshot for request {request_sha256}") from exc
        required_v1 = {
            "schema_version", "request", "request_sha256", "retrieved_at_utc",
            "http_status", "response_headers", "byte_count", "response_sha256",
            "response_file",
        }
        required_v2 = required_v1 | {"final_url", "retrieval_semantics"}
        request = meta.get("request") if isinstance(meta, dict) else None
        metadata_schema = meta.get("schema_version") if isinstance(meta, dict) else None
        if (
            not isinstance(meta, dict)
            or (
                metadata_schema == 1 and set(meta) != required_v1
            )
            or (
                metadata_schema == 2 and set(meta) != required_v2
            )
            or metadata_schema not in {1, 2}
            or not isinstance(request, Mapping)
            or sha256_bytes(canonical_json_bytes(request)) != request_sha256
            or meta.get("response_file") != "response.bin"
            or type(meta.get("http_status")) is not int
            or int(meta["http_status"]) != 200
            or not isinstance(meta.get("response_headers"), Mapping)
            or (
                metadata_schema == 2
                and not isinstance(meta.get("final_url"), str)
            )
            or (
                metadata_schema == 2
                and meta.get("retrieval_semantics") not in {
                    "DIRECT_HTTP_RESPONSE",
                    "BYTE_IDENTICAL_REFETCH_COMPLETED_RESPONSE_ONLY_TRANSACTION",
                }
            )
            or (
                expected_final_url is not None
                and (
                    metadata_schema != 2
                    or meta.get("final_url") != expected_final_url
                )
            )
        ):
            raise ProvenanceError(f"snapshot metadata contract changed: {metadata_path}")
        require_canonical_utc(
            meta.get("retrieved_at_utc"), label="snapshot retrieved_at_utc"
        )
        actual = sha256_bytes(payload)
        if meta.get("request_sha256") != request_sha256:
            raise ProvenanceError(f"request fingerprint mismatch in {metadata_path}")
        if meta.get("response_sha256") != actual:
            raise ProvenanceError(f"raw response checksum mismatch in {response_path}")
        if meta.get("byte_count") != len(payload):
            raise ProvenanceError(f"raw response byte-count mismatch in {response_path}")
        record = SnapshotRecord(
            provider=str(request["provider"]),
            request_sha256=request_sha256,
            response_sha256=actual,
            response_path=response_path,
            metadata_path=metadata_path,
            retrieved_at_utc=str(meta["retrieved_at_utc"]),
            byte_count=len(payload),
            request=dict(request),
        )
        return payload, record

    def fetch(
        self,
        *,
        provider: str,
        url: str,
        headers: Mapping[str, str] | None = None,
        timeout: float = 60.0,
        retries: int = 3,
        resume_incomplete: bool = False,
        expected_final_url: str | None = None,
        _fault_injector: Callable[[str, Path], object] | None = None,
    ) -> tuple[bytes, SnapshotRecord]:
        """Fetch or reuse one exact response and return its verified bytes.

        With ``resume_incomplete=True``, a complete single-link response whose
        metadata suffix is absent may be re-fetched, byte-compared, and then
        completed by create-only metadata publication.  No existing byte is
        replaced.  Offline mode cannot reconstruct missing retrieval metadata.
        """
        request_doc = self.request_document(
            provider=provider, url=url, method="GET", headers=headers)
        request_sha = sha256_bytes(canonical_json_bytes(request_doc))
        response_path, metadata_path = self._paths(provider, request_sha)

        response_exists = os.path.lexists(response_path)
        metadata_exists = os.path.lexists(metadata_path)
        if response_exists and metadata_exists:
            return self._read_verified(
                response_path,
                metadata_path,
                request_sha,
                expected_final_url=expected_final_url,
            )
        if metadata_exists:
            raise ProvenanceError(f"incomplete snapshot for request {request_sha}")
        response_only = response_exists
        if response_only:
            if not resume_incomplete:
                return self._read_verified(response_path, metadata_path, request_sha)
            existing_response = _read_single_link_regular(
                response_path,
                label="incomplete snapshot response",
                trusted_root=self.trusted_root,
            )
        else:
            existing_response = None
        if self.offline:
            if response_only:
                raise ProvenanceError(
                    f"offline snapshot metadata miss for {provider} request "
                    f"{request_sha}: {url}"
                )
            raise ProvenanceError(
                f"offline snapshot miss for {provider} request {request_sha}: {url}")

        attempts = max(1, int(retries))
        last_error: Exception | None = None
        downloaded: tuple[bytes, int, dict[str, str], str, str] | None = None
        for attempt in range(attempts):
            try:
                req = urllib.request.Request(url, headers=dict(headers or {}), method="GET")
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    payload = response.read()
                    status = int(getattr(response, "status", 200) or 200)
                    response_headers = dict(response.headers.items())
                    geturl = getattr(response, "geturl", None)
                    if expected_final_url is not None and not callable(geturl):
                        raise ProvenanceError(
                            "HTTP transport did not expose its final URL"
                        )
                    final_url = str(geturl()) if callable(geturl) else url
                if not payload:
                    raise ProvenanceError(f"empty response from {url}")
                if status != 200:
                    raise ProvenanceError(f"unexpected HTTP status {status} from {url}")
                if expected_final_url is not None and final_url != expected_final_url:
                    raise ProvenanceError(
                        f"unexpected HTTP redirect for immutable request: {final_url}"
                    )
                retrieved = utc_now_iso()
                downloaded = (
                    payload, status, response_headers, retrieved, final_url
                )
                break
            except Exception as exc:  # preserve the concrete cause below
                last_error = exc
                if attempt + 1 < attempts:
                    time.sleep(min(2.0 ** attempt, 8.0))
        if downloaded is None:
            raise ProvenanceError(
                f"failed to acquire {provider} after {attempts} attempts: {url}"
            ) from last_error
        payload, status, response_headers, retrieved, final_url = downloaded
        if existing_response is not None:
            if payload != existing_response:
                raise ProvenanceError(
                    f"re-fetched response differs from incomplete snapshot: {response_path}"
                )
        else:
            if resume_incomplete:
                try:
                    _exclusive_create(
                        response_path,
                        payload,
                        trusted_root=self.trusted_root,
                    )
                except FileExistsError as exc:
                    raise ProvenanceError(
                        f"refusing to replace snapshot response: {response_path}"
                    ) from exc
            else:
                _atomic_write(response_path, payload)
            if _fault_injector is not None:
                _fault_injector("after_raw_response_publish", response_path)
        response_sha = sha256_bytes(payload)
        metadata = {
            "schema_version": 2 if expected_final_url is not None else 1,
            "request": request_doc,
            "request_sha256": request_sha,
            "retrieved_at_utc": retrieved,
            "http_status": status,
            "response_headers": response_headers,
            "byte_count": len(payload),
            "response_sha256": response_sha,
            "response_file": "response.bin",
        }
        if expected_final_url is not None:
            metadata["final_url"] = final_url
            metadata["retrieval_semantics"] = (
                "BYTE_IDENTICAL_REFETCH_COMPLETED_RESPONSE_ONLY_TRANSACTION"
                if existing_response is not None
                else "DIRECT_HTTP_RESPONSE"
            )
        try:
            _exclusive_create(
                metadata_path,
                canonical_json_bytes(metadata),
                trusted_root=self.trusted_root,
            )
        except FileExistsError as exc:
            raise ProvenanceError(
                f"refusing to replace snapshot metadata: {metadata_path}"
            ) from exc
        return self._read_verified(
            response_path,
            metadata_path,
            request_sha,
            expected_final_url=expected_final_url,
        )

    def write_index(self) -> Path:
        """Verify every snapshot and publish a deterministic store index."""
        records: list[dict[str, Any]] = []
        for metadata_path in sorted(self.root.glob("*/*/metadata.json")):
            request_sha = metadata_path.parent.name
            response_path = metadata_path.parent / "response.bin"
            _, record = self._read_verified(response_path, metadata_path, request_sha)
            records.append({
                "provider": record.provider,
                "request_sha256": record.request_sha256,
                "response_sha256": record.response_sha256,
                "retrieved_at_utc": record.retrieved_at_utc,
                "byte_count": record.byte_count,
                "request": dict(record.request),
                "metadata_path": str(metadata_path.relative_to(self.root)),
                "response_path": str(response_path.relative_to(self.root)),
            })
        index = {
            "schema_version": 1,
            "snapshot_count": len(records),
            "records": records,
        }
        path = self.root / "snapshot_index.json"
        _atomic_write(path, canonical_json_bytes(index))
        return path

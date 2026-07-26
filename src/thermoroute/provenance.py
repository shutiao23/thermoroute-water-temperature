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

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
from typing import Any, Callable, Mapping
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


def _exclusive_create(path: Path, payload: bytes) -> None:
    """Create one immutable suffix without replacing any existing directory entry."""
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


def _read_single_link_regular(path: Path, *, label: str) -> bytes:
    """Read exact bytes while rejecting links and path replacement."""
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


class SnapshotStore:
    """Content-addressed store for immutable public API responses."""

    def __init__(self, root: str | Path, *, offline: bool = False) -> None:
        self.root = Path(root)
        self.offline = offline

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
    ) -> tuple[bytes, SnapshotRecord]:
        try:
            metadata_bytes = _read_single_link_regular(
                metadata_path, label="snapshot metadata"
            )
            payload = _read_single_link_regular(
                response_path, label="snapshot response"
            )
            meta = json.loads(metadata_bytes.decode("utf-8"))
        except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProvenanceError(f"incomplete snapshot for request {request_sha256}") from exc
        required = {
            "schema_version", "request", "request_sha256", "retrieved_at_utc",
            "http_status", "response_headers", "byte_count", "response_sha256",
            "response_file",
        }
        request = meta.get("request") if isinstance(meta, dict) else None
        if (
            not isinstance(meta, dict)
            or set(meta) != required
            or meta.get("schema_version") != 1
            or not isinstance(request, Mapping)
            or sha256_bytes(canonical_json_bytes(request)) != request_sha256
            or meta.get("response_file") != "response.bin"
            or type(meta.get("http_status")) is not int
            or int(meta["http_status"]) != 200
            or not isinstance(meta.get("response_headers"), Mapping)
        ):
            raise ProvenanceError(f"snapshot metadata contract changed: {metadata_path}")
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
            return self._read_verified(response_path, metadata_path, request_sha)
        if metadata_exists:
            raise ProvenanceError(f"incomplete snapshot for request {request_sha}")
        response_only = response_exists
        if response_only:
            if not resume_incomplete:
                return self._read_verified(response_path, metadata_path, request_sha)
            existing_response = _read_single_link_regular(
                response_path, label="incomplete snapshot response"
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
        downloaded: tuple[bytes, int, dict[str, str], str] | None = None
        for attempt in range(attempts):
            try:
                req = urllib.request.Request(url, headers=dict(headers or {}), method="GET")
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    payload = response.read()
                    status = int(getattr(response, "status", 200) or 200)
                    response_headers = dict(response.headers.items())
                if not payload:
                    raise ProvenanceError(f"empty response from {url}")
                if status != 200:
                    raise ProvenanceError(f"unexpected HTTP status {status} from {url}")
                retrieved = datetime.now(timezone.utc).isoformat()
                downloaded = (payload, status, response_headers, retrieved)
                break
            except Exception as exc:  # preserve the concrete cause below
                last_error = exc
                if attempt + 1 < attempts:
                    time.sleep(min(2.0 ** attempt, 8.0))
        if downloaded is None:
            raise ProvenanceError(
                f"failed to acquire {provider} after {attempts} attempts: {url}"
            ) from last_error
        payload, status, response_headers, retrieved = downloaded
        if existing_response is not None:
            if payload != existing_response:
                raise ProvenanceError(
                    f"re-fetched response differs from incomplete snapshot: {response_path}"
                )
        else:
            if resume_incomplete:
                try:
                    _exclusive_create(response_path, payload)
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
            "schema_version": 1,
            "request": request_doc,
            "request_sha256": request_sha,
            "retrieved_at_utc": retrieved,
            "http_status": status,
            "response_headers": response_headers,
            "byte_count": len(payload),
            "response_sha256": response_sha,
            "response_file": "response.bin",
        }
        try:
            _exclusive_create(metadata_path, canonical_json_bytes(metadata))
        except FileExistsError as exc:
            raise ProvenanceError(
                f"refusing to replace snapshot metadata: {metadata_path}"
            ) from exc
        return self._read_verified(response_path, metadata_path, request_sha)

    def write_index(self) -> Path:
        """Verify every snapshot and publish a deterministic store index."""
        records: list[dict[str, Any]] = []
        for metadata_path in sorted(self.root.glob("*/*/metadata.json")):
            request_sha = metadata_path.parent.name
            response_path = metadata_path.parent / "response.bin"
            _, record = self._read_verified(response_path, metadata_path, request_sha)
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            records.append({
                "provider": record.provider,
                "request_sha256": record.request_sha256,
                "response_sha256": record.response_sha256,
                "retrieved_at_utc": record.retrieved_at_utc,
                "byte_count": record.byte_count,
                "request": meta["request"],
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

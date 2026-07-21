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
import time
from typing import Any, Mapping
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
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            payload = response_path.read_bytes()
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ProvenanceError(f"incomplete snapshot for request {request_sha256}") from exc
        actual = sha256_bytes(payload)
        if meta.get("request_sha256") != request_sha256:
            raise ProvenanceError(f"request fingerprint mismatch in {metadata_path}")
        if meta.get("response_sha256") != actual:
            raise ProvenanceError(f"raw response checksum mismatch in {response_path}")
        if meta.get("byte_count") != len(payload):
            raise ProvenanceError(f"raw response byte-count mismatch in {response_path}")
        record = SnapshotRecord(
            provider=str(meta["request"]["provider"]),
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
    ) -> tuple[bytes, SnapshotRecord]:
        """Fetch or reuse one exact response and return its verified bytes."""
        request_doc = self.request_document(
            provider=provider, url=url, method="GET", headers=headers)
        request_sha = sha256_bytes(canonical_json_bytes(request_doc))
        response_path, metadata_path = self._paths(provider, request_sha)

        if response_path.exists() or metadata_path.exists():
            return self._read_verified(response_path, metadata_path, request_sha)
        if self.offline:
            raise ProvenanceError(
                f"offline snapshot miss for {provider} request {request_sha}: {url}")

        attempts = max(1, int(retries))
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                req = urllib.request.Request(url, headers=dict(headers or {}), method="GET")
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    payload = response.read()
                    status = int(getattr(response, "status", 200) or 200)
                    response_headers = dict(response.headers.items())
                if not payload:
                    raise ProvenanceError(f"empty response from {url}")
                retrieved = datetime.now(timezone.utc).isoformat()
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
                # Response first, metadata second.  A crash between the two is
                # intentionally detected as an incomplete transaction later.
                _atomic_write(response_path, payload)
                _atomic_write(metadata_path, canonical_json_bytes(metadata))
                return self._read_verified(response_path, metadata_path, request_sha)
            except Exception as exc:  # preserve the concrete cause below
                last_error = exc
                if attempt + 1 < attempts:
                    time.sleep(min(2.0 ** attempt, 8.0))
        raise ProvenanceError(
            f"failed to acquire {provider} after {attempts} attempts: {url}") from last_error

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

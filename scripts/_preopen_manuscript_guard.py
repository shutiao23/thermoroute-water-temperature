#!/usr/bin/env python3
"""Standard-library-only guard for PRE-OPEN manuscript rendering."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


CLAIM_REGISTRY_RELATIVE = "protocols/route_a_claim_registry_v1.json"
PREOPEN_MANUSCRIPT_SOURCES = (
    "paper/ThermoRoute_paper.md",
    "paper/cover_letter.md",
    "paper/highlights.md",
)
EXPECTED_PHASE_RESOLVER = {
    "mode": "DERIVE_NEVER_CLI_OVERRIDE",
    "canonical_authorization": (
        "data_usgs/confirmatory_opening_authorization_v1.json"
    ),
    "namespace_glob": "outputs/confirmatory/route_a_*",
}


class PreopenManuscriptGuardError(RuntimeError):
    """The repository is no longer the frozen PRE-OPEN render state."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PreopenManuscriptGuardError(f"{label} is absent or malformed")
    raw = Path(value)
    if raw.is_absolute() or raw.as_posix() != value or ".." in raw.parts:
        raise PreopenManuscriptGuardError(
            f"{label} must be a canonical repository-relative POSIX path"
        )
    return value


def _inside_file(root: Path, relative: str, *, label: str) -> Path:
    candidate = root / relative
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise PreopenManuscriptGuardError(f"{label} is absent or unsafe: {relative}") from exc
    if root not in resolved.parents or not resolved.is_file():
        raise PreopenManuscriptGuardError(
            f"{label} escapes the repository or is not a regular file: {relative}"
        )
    return resolved


def _load_registry(root: Path, registry_relative: str) -> Mapping[str, Any]:
    registry_path = _inside_file(root, registry_relative, label="claim registry")
    try:
        document = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreopenManuscriptGuardError("claim registry is unreadable or invalid") from exc
    if (
        not isinstance(document, Mapping)
        or document.get("format") != "thermoroute.route-a-claim-ledger.v2"
    ):
        raise PreopenManuscriptGuardError("claim registry format is not frozen v2")
    return document


def _validate_phase_absence(root: Path, registry: Mapping[str, Any]) -> None:
    resolver = registry.get("phase_resolver")
    if not isinstance(resolver, Mapping) or dict(resolver) != EXPECTED_PHASE_RESOLVER:
        raise PreopenManuscriptGuardError("claim registry phase resolver changed")

    authorization_relative = _relative_path(
        resolver.get("canonical_authorization"), label="canonical authorization path"
    )
    authorization = root / authorization_relative
    if os.path.lexists(authorization):
        raise PreopenManuscriptGuardError(
            "canonical opening authorization exists; PRE-OPEN rendering is forbidden"
        )

    namespace_glob = _relative_path(
        resolver.get("namespace_glob"), label="confirmation namespace glob"
    )
    namespace_root = root / Path(namespace_glob).parent
    if os.path.lexists(namespace_root):
        raise PreopenManuscriptGuardError(
            "confirmation output namespace exists; PRE-OPEN rendering is forbidden"
        )


def assert_preopen_manuscript_render_allowed(
    root: str | Path,
    *,
    registry_relative: str = CLAIM_REGISTRY_RELATIVE,
) -> dict[str, str]:
    """Require PRE phase evidence and exact frozen hashes for all render inputs."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise PreopenManuscriptGuardError("repository root is absent or not a directory")
    registry_relative = _relative_path(registry_relative, label="claim registry path")
    registry = _load_registry(root, registry_relative)
    _validate_phase_absence(root, registry)

    required_documents = registry.get("required_documents")
    expected_hashes = registry.get("preopen_document_sha256")
    if (
        not isinstance(required_documents, list)
        or not all(isinstance(value, str) for value in required_documents)
        or len(required_documents) != len(set(required_documents))
        or not isinstance(expected_hashes, Mapping)
        or set(expected_hashes) != set(required_documents)
    ):
        raise PreopenManuscriptGuardError(
            "claim registry frozen-document hash map is not exact"
        )

    actual_hashes: dict[str, str] = {}
    for relative in PREOPEN_MANUSCRIPT_SOURCES:
        expected = expected_hashes.get(relative)
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise PreopenManuscriptGuardError(
                f"claim registry lacks a frozen SHA-256 for {relative}"
            )
        source = _inside_file(root, relative, label="PRE-OPEN manuscript source")
        actual = _sha256_file(source)
        if actual != expected:
            raise PreopenManuscriptGuardError(
                f"PRE-OPEN manuscript source differs from its frozen SHA-256: {relative}"
            )
        actual_hashes[relative] = actual
    return actual_hashes


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        hashes = assert_preopen_manuscript_render_allowed(args.root)
    except PreopenManuscriptGuardError as exc:
        print(f"PRE-OPEN manuscript render refused: {exc}", file=sys.stderr)
        return 2
    print(f"PASS frozen PRE-OPEN manuscript sources ({len(hashes)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

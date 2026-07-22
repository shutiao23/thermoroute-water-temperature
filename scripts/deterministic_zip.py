#!/usr/bin/env python3
"""Create a byte-reproducible ZIP from a staged release directory.

The archive order, timestamps, owner-independent modes and compression settings
are fixed.  Symlinks and transient interpreter/editor files are rejected or
excluded so a release cannot silently depend on the machine that built it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
import zipfile


FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
EXCLUDED_NAMES = {".DS_Store"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
FIXED_MANIFEST_UTC = "1980-01-01T00:00:00+00:00"


def _archive_mode(relative: PurePosixPath, *, directory: bool) -> int:
    if directory:
        return stat.S_IFDIR | 0o755
    executable = (
        len(relative.parts) > 1
        and relative.parts[1] == "scripts"
        and relative.suffix in {".py", ".sh"}
    )
    return stat.S_IFREG | (0o755 if executable else 0o644)


def _excluded(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return (
        "__pycache__" in relative.parts
        or path.name in EXCLUDED_NAMES
        or path.suffix in EXCLUDED_SUFFIXES
    )


def _entries(source: Path, archive_root: str) -> list[tuple[Path | None, PurePosixPath, bool]]:
    root_name = PurePosixPath(archive_root)
    if root_name.is_absolute() or len(root_name.parts) != 1 or root_name.name in {"", ".", ".."}:
        raise ValueError("archive root must be one safe path component")
    entries: list[tuple[Path | None, PurePosixPath, bool]] = [
        (None, root_name, True)
    ]
    for path in sorted(source.rglob("*"), key=lambda value: value.relative_to(source).as_posix()):
        if _excluded(path, source):
            continue
        if path.is_symlink():
            raise ValueError(f"release staging contains a symlink: {path}")
        if not (path.is_dir() or path.is_file()):
            raise ValueError(f"release staging contains a non-regular entry: {path}")
        relative = root_name / PurePosixPath(path.relative_to(source).as_posix())
        entries.append((path, relative, path.is_dir()))
    return sorted(
        entries,
        key=lambda item: item[1].as_posix() + ("/" if item[2] else ""),
    )


def create_deterministic_zip(
    source: str | Path,
    destination: str | Path,
    *,
    archive_root: str = "thermoroute",
) -> None:
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination == source or source in destination.parents:
        raise ValueError("destination cannot be inside the staged release")
    manifest = source / "outputs" / "manifest.json"
    if manifest.is_file():
        document = json.loads(manifest.read_text(encoding="utf-8"))
        if "generated_utc" not in document:
            raise ValueError("release manifest lacks generated_utc")
        document["generated_utc"] = FIXED_MANIFEST_UTC
        profile_marker = source / "data_usgs" / "release_profile_v1.json"
        if profile_marker.is_file():
            profile = json.loads(profile_marker.read_text(encoding="utf-8"))
            revision = profile.get("authorized_worktree_dirt_policy")
            if revision is not None:
                document["release_revision"] = revision
            document["release_evidence"] = {
                "profile": profile.get("profile"),
                "claim_validation": profile.get("claim_validation"),
                "git_history_evidence": profile.get("git_history_evidence"),
                "reproducibility_lock": profile.get("artifact_closure", {}).get(
                    "reproducibility_lock"
                ),
            }
        payload = (
            json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        descriptor, temporary_manifest_name = tempfile.mkstemp(
            prefix=".manifest.", suffix=".tmp", dir=manifest.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_manifest_name, manifest)
        except BaseException:
            try:
                os.unlink(temporary_manifest_name)
            except FileNotFoundError:
                pass
            raise
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            strict_timestamps=True,
        ) as archive:
            for path, relative, directory in _entries(source, archive_root):
                name = relative.as_posix() + ("/" if directory else "")
                info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIMESTAMP)
                info.create_system = 3
                info.external_attr = _archive_mode(relative, directory=directory) << 16
                info.compress_type = zipfile.ZIP_STORED if directory else zipfile.ZIP_DEFLATED
                info.flag_bits |= 0x800
                payload = b"" if directory else path.read_bytes()  # type: ignore[union-attr]
                archive.writestr(info, payload, compress_type=info.compress_type, compresslevel=9)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--archive-root", default="thermoroute")
    args = parser.parse_args()
    create_deterministic_zip(
        args.source, args.destination, archive_root=args.archive_root
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

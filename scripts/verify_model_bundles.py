#!/usr/bin/env python3
"""Verify the frozen model bundles against the tracked manifest.

Reads ``data_usgs/model_bundle_manifest_v1.json`` and checks every declared
file's sha256 and size under ``--bundle-root`` (default ``outputs/models``).
Used by CI and the release verifier so a clean checkout plus the deposited
bundle pack is provably the scoring input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "data_usgs" / "model_bundle_manifest_v1.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", default="outputs/models", type=Path)
    parser.add_argument("--manifest", default=MANIFEST, type=Path)
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("format") != "thermoroute.model-bundle-manifest.v1":
        print("manifest format unsupported", file=sys.stderr)
        return 2
    failures = []
    checked = 0
    for bundle_name, entry in sorted(manifest["bundles"].items()):
        bundle_dir = args.bundle_root / bundle_name
        for relative, record in entry["files"].items():
            path = bundle_dir / relative
            checked += 1
            if not path.exists():
                failures.append(f"{bundle_name}/{relative}: missing")
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != record["sha256"]:
                failures.append(f"{bundle_name}/{relative}: sha256 mismatch")
            elif path.stat().st_size != record["size"]:
                failures.append(f"{bundle_name}/{relative}: size mismatch")
    print(f"checked {checked} files across {len(manifest['bundles'])} bundles")
    if failures:
        print("FAILED:")
        for failure in failures[:20]:
            print("  " + failure)
        return 1
    print("all bundles verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Record what every prediction shard is, without storing its bytes in git.

The shards are the evidence behind every number in the paper, and they are also
87% redundant: six columns are constant within a file and repeated once per row,
``key_id`` is already frozen in the key registry, and ``y_true`` is too.  A
2.84 MB shard carries about 0.37 MB of genuinely new information.  They are
also deterministically regenerable -- pinned inputs, fixed seeds, single-thread
LightGBM -- so committing three quarters of a gigabyte of them buys storage, not
verifiability.

This builder writes the thing that does buy verifiability: one SHA-256 per
shard, with its row count, station count and the identity columns, so anyone
who reruns a producer can check byte-for-byte that they reproduced the file this
paper was computed from.  A digest proves identity; the bytes only take up
space.  Same principle as the governance cleanup -- record, do not lock.

The repository already worked this way for ``outputs/final/predictions.parquet``
and the withdrawn ``ladder_shards/``; this makes the rule uniform and gives the
excluded bytes a verifiable fingerprint they previously lacked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "outputs" / "final"
DEFAULT_OUT = FINAL / "shard_digest_index.json"

#: Directories whose bytes are excluded from git and fingerprinted here.
SHARD_DIRS = (
    "forcing_regime_v5_observed/forcing_shards_v5_observed",
    "forcing_placebo_v5a/placebo_shards_v5a_shuffle",
    "forcing_placebo_v5a_shift/placebo_shards_v5a_shift",
    "forcing_components_v5a/component_shards_v5a",
    "forcing_f2a_v5a/f2a_shards_v5a",
    "information_ladder_v6_observed/ladder_shards_v6_observed",
)

FORMAT = "thermoroute.shard-digest-index.v1"
IDENTITY_COLUMNS = ("horizon", "model", "arm", "level", "geometry", "fold")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(path)
    identity = {
        column: str(frame[column].iloc[0])
        for column in IDENTITY_COLUMNS
        if column in frame.columns and frame[column].nunique() == 1
    }
    return {
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": len(frame),
        "stations": int(frame["site_id"].nunique()),
        **identity,
    }


def build(directories: Sequence[str]) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    total_bytes = 0
    total_shards = 0
    for relative in directories:
        directory = FINAL / relative
        if not directory.exists():
            groups[relative] = {"present": False}
            continue
        entries = {}
        for path in sorted(directory.glob("*.parquet")):
            entries[path.name] = describe(path)
            total_bytes += entries[path.name]["bytes"]
            total_shards += 1
        groups[relative] = {
            "present": True,
            "shard_count": len(entries),
            "shards": entries,
        }
    return {
        "format": FORMAT,
        "purpose": (
            "verify a regenerated shard byte-for-byte against the one this "
            "paper was computed from; the bytes themselves are excluded from "
            "version control as regenerable"
        ),
        "regenerable": True,
        "determinism": (
            "pinned inputs, fixed seeds, single-thread LightGBM; parallel "
            "workers change scheduling only, never what a fit computes"
        ),
        "total_shards": total_shards,
        "total_bytes_excluded": total_bytes,
        "groups": groups,
    }


def verify(index_path: Path) -> dict[str, Any]:
    """Re-hash every shard present on disk and report disagreements."""
    index = json.loads(index_path.read_text(encoding="utf-8"))
    checked = missing = mismatched = 0
    problems: list[str] = []
    for relative, group in index["groups"].items():
        if not group.get("present"):
            continue
        for name, record in group["shards"].items():
            path = FINAL / relative / name
            if not path.exists():
                missing += 1
                continue
            checked += 1
            if _sha256_file(path) != record["sha256"]:
                mismatched += 1
                problems.append(f"{relative}/{name}")
    return {
        "checked": checked, "absent": missing, "mismatched": mismatched,
        "mismatches": problems[:10],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verify", action="store_true",
                        help="re-hash the shards on disk against the index")
    args = parser.parse_args(argv)

    if args.verify:
        result = verify(args.out)
        print(json.dumps(result, sort_keys=True, indent=1))
        return 1 if result["mismatched"] else 0

    index = build(SHARD_DIRS)
    args.out.write_text(
        json.dumps(index, sort_keys=True, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "WRITTEN",
        "total_shards": index["total_shards"],
        "gigabytes_excluded": round(index["total_bytes_excluded"] / 1e9, 2),
        "index_kilobytes": round(args.out.stat().st_size / 1e3, 1),
    }, sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

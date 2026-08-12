#!/usr/bin/env python3
"""Merge the per-worker TCN lineage manifests into one, and prove it is complete.

Sharding the arm across workers means each writes its own manifest under its own
tag, which is what keeps parallel workers from overwriting one another's lineage
record. The cost is that the record ends up in two dozen files, and a reader
cannot tell from any one of them whether the set is complete.

This merges them and, more usefully, checks the merge against the shards on
disk. The two failure directions are different problems and are reported
separately: a shard with no manifest entry is a fit whose evidence was lost,
and a manifest entry with no shard is a record of a fit that is not there. The
first is what actually happened during this arm's run -- two workers were
launched on the same tag, and the second overwrote the first's manifest after
skipping the cells it had already written -- so the check exists because the
failure did, not in anticipation of it.

Duplicate entries for one shard are collapsed only when they agree; a
disagreement means two fits produced different evidence for the same cell,
which is a determinism failure and is raised rather than resolved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.final.run_plain_tcn_arm as TCN

FORMAT = "thermoroute.plain-tcn-arm.consolidated.v1"
OUT_NAME = "tcn_lineage_manifest_consolidated_v1.json"


class LineageError(RuntimeError):
    """The lineage record and the shards on disk disagree."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def collect(directory: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Merge every per-worker manifest, keyed by shard filename."""
    cells: dict[str, dict[str, Any]] = {}
    sources: list[str] = []
    for path in sorted(directory.glob("tcn_lineage_manifest_v1_*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        sources.append(path.name)
        for cell in document["cells"]:
            name = cell["shard"]
            if name in cells and cells[name] != cell:
                raise LineageError(
                    f"two fits recorded different evidence for {name}; "
                    "the arm is not deterministic"
                )
            cells[name] = cell
    return cells, sources


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=TCN.OUTPUT_DIR)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="report gaps instead of failing on them")
    args = parser.parse_args(argv)

    shard_dir = args.dir / TCN.SHARD_DIRNAME
    cells, sources = collect(args.dir)
    on_disk = {path.name for path in shard_dir.glob("*.parquet")}

    unrecorded = sorted(on_disk - set(cells))
    orphaned = sorted(set(cells) - on_disk)
    if (unrecorded or orphaned) and not args.allow_incomplete:
        raise LineageError(
            f"{len(unrecorded)} shard(s) carry no lineage entry and "
            f"{len(orphaned)} entr(ies) name a shard that is absent"
        )

    merged = {
        "format": FORMAT,
        "status": TCN.STATUS,
        "architecture": {
            "kind": "plain causal TCN over the lag axis",
            "channels": TCN.CHANNELS, "kernel": TCN.KERNEL,
            "dilations": list(TCN.DILATIONS),
            "receptive_field_positions": 1 + (TCN.KERNEL - 1) * sum(TCN.DILATIONS),
            "epochs": TCN.EPOCHS, "batch": TCN.BATCH,
            "learning_rate": TCN.LEARNING_RATE,
            "bound_degC": TCN.BOUND_DEGC,
            "model_selection": "lowest in-fold validation MSE across epochs",
        },
        "runner_sha256": _sha256_file(Path(TCN.__file__).resolve()),
        "worker_manifests": sources,
        "shards_on_disk": len(on_disk),
        "cells_recorded": len(cells),
        "shards_without_a_lineage_entry": unrecorded,
        "lineage_entries_without_a_shard": orphaned,
        "cells": [cells[name] for name in sorted(cells)],
    }
    out = args.dir / OUT_NAME
    out.write_text(json.dumps(merged, sort_keys=True, indent=1, default=str) + "\n",
                   encoding="utf-8")
    print(json.dumps({
        "status": "WRITTEN", "worker_manifests": len(sources),
        "shards_on_disk": len(on_disk), "cells_recorded": len(cells),
        "unrecorded": len(unrecorded), "orphaned": len(orphaned),
    }, sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

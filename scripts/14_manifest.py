#!/usr/bin/env python3
"""Stage 14 — full artifact manifest (sha256 evidence chain).

Walks every artifact the paper's numbers depend on — prediction parquets and
their checkpoints, tables, reports, trained models, figures, and the input
panels — and records path, byte size and sha256 together with the git commit
of the working tree that produced them.  The manifest is the arbiter of "which
file is the current truth": any number in the manuscript must be traceable to
a hash listed here.

Run:  python3 scripts/14_manifest.py            # writes outputs/manifest.json
      python3 scripts/14_manifest.py --check    # verify hashes instead
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCAN = [
    ("data", "*.csv"),
    ("data_usgs", "panel_usgs*.parquet"),
    ("data_usgs", "stations_meta*.csv"),
    ("data_usgs", "rejected_sites*.csv"),
    ("outputs/predictions", "**/*.parquet"),
    ("outputs/tables", "*.csv"),
    ("outputs/tables", "*.md"),
    ("outputs/tables", "*.npz"),
    ("outputs/models", "*.pt"),
    ("outputs/reports", "*.md"),
    ("outputs/figures", "*.png"),
    ("outputs/figures", "*.pdf"),
]


def sha256(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def collect() -> dict[str, dict]:
    files = {}
    for base, pattern in SCAN:
        for p in sorted((ROOT / base).glob(pattern)):
            if p.is_file():
                rel = str(p.relative_to(ROOT))
                files[rel] = {"sha256": sha256(p), "bytes": p.stat().st_size,
                              "mtime": datetime.fromtimestamp(
                                  p.stat().st_mtime, tz=timezone.utc).isoformat()}
    return files


def git_state() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify existing manifest hashes; non-zero exit on drift")
    args = ap.parse_args()
    mpath = ROOT / "outputs" / "manifest.json"

    if args.check:
        old = json.loads(mpath.read_text())
        drift = []
        for rel, meta in old.get("files", {}).items():
            p = ROOT / rel
            if not p.exists():
                drift.append(f"MISSING  {rel}")
            elif sha256(p) != meta["sha256"]:
                drift.append(f"CHANGED  {rel}")
        print("\n".join(drift) if drift else
              f"manifest OK: {len(old.get('files', {}))} files verified")
        return 1 if drift else 0

    files = collect()
    manifest = {
        "generated_utc": datetime.now(tz=timezone.utc).isoformat(),
        "git": git_state(),
        "n_files": len(files),
        "current_truth": {
            "usgs_predictions": "outputs/predictions/usgs_predictions_v2.parquet",
            "usgs_panel": "data_usgs/panel_usgs_100.parquet",
            "usgs_scores": "outputs/tables/usgs_scores_v2.csv",
            "cascade_predictions": "outputs/predictions/predictions.parquet",
            "cascade_scores": "outputs/tables/scores_all.csv",
        },
        "files": files,
    }
    mpath.write_text(json.dumps(manifest, indent=1))
    print(f"wrote {mpath.relative_to(ROOT)}: {len(files)} artifacts hashed, "
          f"commit {manifest['git']['commit'][:12]}"
          f"{' (dirty tree)' if manifest['git']['dirty'] else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

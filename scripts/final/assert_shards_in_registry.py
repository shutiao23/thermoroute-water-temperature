#!/usr/bin/env python3
"""Verify every production ladder shard scores exactly the common registry.

One-off production gate (protocol v2 runner requirement, two-sided): for each
shard under ``outputs/final/ladder_shards/`` the key set must equal the common
forecast-key registry's subset for its held stations and horizon — no
registry key may be declined and no key outside the registry may be scored.
Legacy-mode shards (``ladder_shards_legacy/``) are exempt by design.

Usage::

    python scripts/final/assert_shards_in_registry.py [--shards DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "outputs" / "final"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default=FINAL / "ladder_shards")
    args = ap.parse_args()

    ref = pd.read_parquet(FINAL / "forecast_keys.parquet",
                          columns=["site_id", "horizon", "issue_date"])
    ref["site_id"] = ref["site_id"].astype(str).str.zfill(8)
    ref["issue_date"] = pd.to_datetime(ref["issue_date"])
    ref_sets = {}
    for (site, h), g in ref.groupby(["site_id", "horizon"]):
        ref_sets[(site, int(h))] = set(
            g["issue_date"].dt.strftime("%Y-%m-%d"))

    problems = 0
    checked = 0
    for sp in sorted(Path(args.shards).glob("*.parquet")):
        if sp.stat().st_size == 0:
            continue
        d = pd.read_parquet(sp)
        d["site_id"] = d["site_id"].astype(str).str.zfill(8)
        d["issue_date"] = pd.to_datetime(d["issue_date"])
        h = int(d["horizon"].iloc[0])
        for site, g in d.groupby("site_id"):
            cell = set(g["issue_date"].dt.strftime("%Y-%m-%d"))
            ref_set = ref_sets.get((site, h), set())
            missing = ref_set - cell
            extra = cell - ref_set
            checked += 1
            if missing or extra:
                problems += 1
                print(f"{sp.name} {site} h{h}: {len(missing)} registry keys "
                      f"missing, {len(extra)} outside the registry")
    if problems:
        print(f"REGISTRY BREACH: {problems} station-cells violate the "
              f"two-sided registry equality ({checked} checked)")
        return 1
    print(f"all {checked} station-cells score exactly the common registry "
          f"subset (two-sided) - OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

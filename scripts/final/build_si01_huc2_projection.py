#!/usr/bin/env python3
"""Fill SI01's empty HUC2 shell from the registry and the reportable cohort.

SI01 declares a projection schema and then leaves the table itself empty, with
every cell marked ``[pending computation]``.  That was the right call while the
counts had no source pointer, but both counts are registry- and cohort-derived
rather than outcome-derived, which is the category SI01's own fill rules permit:
a HUC2 station-count table "may use only receipt- or registry-bound counts", and
these are the latter.  Nothing here reads a prediction or a score.

Two counts per HUC2, kept distinct because they answer different questions.
*Pre-attrition* is the frozen cohort geometry -- how the 120 stable sites
distribute across regions, fixed before any outcome existed, and the input to
every cluster-bootstrap and sign-flip statement in the paper.  *Reportable* is
what survived the per-lead key-count rule at evaluation.  They share one source
registry but are separate ``value_id`` objects, as the README's binder shape
requires, because a reader who conflates them will mis-read the cluster
structure the inference rests on.

The reportable set is taken from the primary forcing shards rather than from a
station list, so it is the set of stations that actually carried scored keys
rather than a set that was intended to.  If those two ever disagree, the shards
are right.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

REGISTRY = ROOT / "data_usgs" / "station_registry_v1.csv"
REFERENCE_SHARDS = (
    ROOT / "outputs" / "final" / "forcing_regime_v5_observed"
    / "forcing_shards_v5_observed"
)
DEFAULT_OUT = ROOT / "outputs" / "final" / "si01_huc2_projection_v1"

FORMAT = "thermoroute.si01-huc2-projection.v1"
MIN_KEYS = 100


class ProjectionError(RuntimeError):
    """A source the projection depends on is missing or inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reportable_stations() -> set[str]:
    """Stations carrying at least ``MIN_KEYS`` scored keys at every lead.

    "Per lead" is the operative phrase: a station reportable at one day and not
    at seven is not in the cohort the paper's tables describe, so the rule is an
    intersection across leads and not a union.
    """
    per_lead: list[set[str]] = []
    for horizon in (1, 3, 7):
        path = REFERENCE_SHARDS / f"F0_LightGBM_h{horizon}_v5_observed.parquet"
        if not path.exists():
            raise ProjectionError(f"missing reference shard: {path}")
        counts = pd.read_parquet(path, columns=["site_id"]).value_counts("site_id")
        per_lead.append(set(counts[counts >= MIN_KEYS].index.astype(str)))
    return set.intersection(*per_lead)


def build() -> tuple[pd.DataFrame, dict[str, Any]]:
    registry = pd.read_csv(REGISTRY, dtype={"site_no": str, "huc2": str})
    registry["site_id"] = registry["site_no"].str.zfill(8)
    registry["huc2"] = registry["huc2"].str.zfill(2)
    reportable = reportable_stations()

    registry["is_reportable"] = registry["site_id"].isin(reportable)
    grouped = registry.groupby("huc2", sort=True)
    table = pd.DataFrame({
        "huc2": grouped.size().index,
        "pre_attrition_stations": grouped.size().to_numpy(),
        "reportable_stations": grouped["is_reportable"].sum().to_numpy(),
    })
    table["binder_row_id"] = [
        f"SI01.HUC2.{huc}" for huc in table["huc2"]
    ]
    unmatched = reportable - set(registry["site_id"])
    if unmatched:
        raise ProjectionError(
            f"{len(unmatched)} scored stations are absent from the registry: "
            f"{sorted(unmatched)[:5]}"
        )
    totals = {
        "pre_attrition_total": int(table["pre_attrition_stations"].sum()),
        "reportable_total": int(table["reportable_stations"].sum()),
        "huc2_groups": int(len(table)),
        "huc2_groups_with_no_reportable_station": int(
            (table["reportable_stations"] == 0).sum()
        ),
        "largest_reportable_share": float(
            table["reportable_stations"].max() / table["reportable_stations"].sum()
        ),
    }
    return table, totals


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)

    table, totals = build()
    if args.print_only:
        print(table.to_string(index=False))
        print(json.dumps(totals, sort_keys=True, indent=1))
        print()
        print("| HUC2 | pre-attrition station count | reportable station count "
              "| binder row ID |")
        print("|---|---:|---:|---|")
        for row in table.itertuples(index=False):
            print(f"| {row.huc2} | {row.pre_attrition_stations} | "
                  f"{row.reportable_stations} | `{row.binder_row_id}` |")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame_path = args.out_dir / "si01_huc2_projection.parquet"
    table.to_parquet(frame_path, index=False)
    summary_path = args.out_dir / "si01_huc2_projection_summary.json"
    summary_path.write_text(json.dumps({
        "format": FORMAT,
        "derivation": (
            "pre-attrition counts from the frozen station registry; reportable "
            "counts from stations carrying at least 100 scored keys at every "
            "lead in the primary F0 shards"
        ),
        "outcome_free": True,
        "min_keys_per_lead": MIN_KEYS,
        "source_bindings": {
            "registry": {"path": str(REGISTRY.relative_to(ROOT)),
                         "sha256": _sha256_file(REGISTRY)},
        },
        "totals": totals,
        "rows": table.to_dict(orient="records"),
    }, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"status": "WRITTEN", **totals}, sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

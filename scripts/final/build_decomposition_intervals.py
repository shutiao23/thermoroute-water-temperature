#!/usr/bin/env python3
"""Attach whole-HUC2 cluster intervals to the memory/learned decomposition.

The station-level error-budget decomposition -- median memory gain 0.49 degC,
median learned gain 0.07 degC, median memory fraction 0.875 at seven days --
appears in the Abstract, in Figure 2, in Section 4.6 and in the Conclusions,
and carries no uncertainty anywhere in the manuscript or the Supporting
Information.  Every other headline in the paper has a clustered interval; this
one did not, which is the gap an external review flagged.

This builder reads the frozen ``outputs/final/decomposition_effects.parquet``
and adds, per model and lead, a 10,000-draw whole-HUC2 cluster bootstrap
percentile interval, the equal-HUC aggregate beside the station-weighted one,
and the leave-one-HUC2-out range, using the same estimators and the same
15-cluster map as every other clustered quantity in the results authority.

It computes no new prediction and refits nothing: the decomposition is a
function of already-published station RMSEs, so this is an uncertainty
statement about an existing frozen result, not a new result.  The intervals
inherit that result's status and its limits -- 15 HUC2 groups, effective
cluster count about 9.5, non-probability cohort -- and are approximate
descriptive sensitivities rather than decision evidence.

The memory fraction is summarised only over stations whose total gain is
positive, exactly as the manuscript defines it; the stations excluded by that
rule are counted in the output rather than dropped silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from thermoroute.significance import (
    cluster_bootstrap_paired_effect,
    cluster_inference_sensitivity,
)
from thermoroute.spatial import huc2_cluster_map, load_station_registry

FINAL = ROOT / "outputs" / "final"
SOURCE = FINAL / "decomposition_effects.parquet"
REGISTRY_CSV = ROOT / "data_usgs" / "station_registry_v1.csv"
DEFAULT_OUT = FINAL / "decomposition_intervals.parquet"

FORMAT = "thermoroute.decomposition-cluster-intervals.v1"
N_BOOT = 10_000
BOOT_SEED = 0

#: Quantities the manuscript prints.  ``memory_fraction`` is restricted to
#: stations with a positive total gain, which is how Section 4.6 defines it.
QUANTITIES = (
    ("g_memory", False, "median station memory gain, degC"),
    ("g_learned", False, "median station learned gain, degC"),
    ("g_total", False, "median station total gain, degC"),
    ("memory_fraction", True, "median station memory fraction, dimensionless"),
)


class IntervalError(RuntimeError):
    """An input or coverage check failed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_clusters() -> dict[str, str]:
    return huc2_cluster_map(load_station_registry(REGISTRY_CSV))


def build_rows() -> list[dict[str, Any]]:
    if not SOURCE.exists():
        raise IntervalError(f"decomposition artifact missing: {SOURCE}")
    frame = pd.read_parquet(SOURCE)
    clusters = load_clusters()
    rows: list[dict[str, Any]] = []

    for (model, horizon), block in frame.groupby(["model", "horizon"], sort=True):
        block = block.sort_values("site_id")
        for column, positive_only, label in QUANTITIES:
            subset = block[block["g_total_positive"]] if positive_only else block
            excluded = len(block) - len(subset)
            values = subset[column].to_numpy(float)
            if not np.isfinite(values).all():
                raise IntervalError(f"non-finite {column} for {model} h{horizon}")
            keys = subset["site_id"].map(clusters)
            if keys.isna().any():
                raise IntervalError("a station is absent from the HUC2 map")
            groups = keys.to_numpy()

            # These are one-sample location statistics, not paired candidate
            # -minus-reference effects, so the sign-flip test does not apply and
            # is deliberately not computed here.
            boot = cluster_bootstrap_paired_effect(
                values, groups, statistic="median", n_boot=N_BOOT, seed=BOOT_SEED,
            )
            sens = cluster_inference_sensitivity(values, groups, statistic="median")
            per_cluster = [
                float(np.median(values[groups == cluster]))
                for cluster in np.unique(groups)
            ]
            rows.append({
                "model": model,
                "horizon": int(horizon),
                "quantity": column,
                "label": label,
                "median": boot["effect"],
                "ci_low": boot["ci_low"],
                "ci_high": boot["ci_high"],
                "iqr_low": float(np.percentile(values, 25)),
                "iqr_high": float(np.percentile(values, 75)),
                "equal_huc": float(np.mean(per_cluster)),
                "loco_min": sens["loco_effect_min"],
                "loco_max": sens["loco_effect_max"],
                "n_stations": len(values),
                "n_clusters": len(np.unique(groups)),
                "effective_cluster_count": sens[
                    "effective_cluster_count_inverse_herfindahl"
                ],
                "stations_excluded_by_positive_total_gain_rule": int(excluded),
                "inference_role": (
                    "APPROXIMATE_DESCRIPTIVE_SENSITIVITY_NOT_DECISION_EVIDENCE"
                ),
            })
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)

    rows = build_rows()
    table = pd.DataFrame(rows)
    if args.print_only:
        seven = table[(table["horizon"] == 7) & (table["model"] == "ThermoRoute")]
        print(seven[[
            "quantity", "median", "ci_low", "ci_high", "equal_huc",
            "loco_min", "loco_max", "n_stations",
        ]].to_string(index=False))
        return 0

    table.to_parquet(args.out, index=False)
    manifest = {
        "format": FORMAT,
        "source_sha256": _sha256_file(SOURCE),
        "builder_sha256": _sha256_file(Path(__file__).resolve()),
        "output_sha256": _sha256_file(args.out),
        "bootstrap_draws": N_BOOT,
        "bootstrap_seed": BOOT_SEED,
        "cluster_unit": "HUC2",
        "note": (
            "Uncertainty for an existing frozen decomposition; no prediction "
            "was recomputed and no model was refitted."
        ),
    }
    (args.out.parent / "decomposition_intervals_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=1) + "\n", encoding="utf-8",
    )
    print(json.dumps({"status": "WRITTEN", "rows": len(table),
                      "out": str(args.out)}, sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

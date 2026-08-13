#!/usr/bin/env python3
"""Is the L2 architecture penalty a model class, or a training budget?

Section 4.10 reports that the tree beats the plain causal TCN by 0.11-0.19 degC
once the local gauge is withheld. That number is only about the model class if
both models were trained to convergence, and the arm's lineage says the network
was not: under the original fixed 40-epoch budget the median best epoch at L2
was 37, with roughly a third of cells still improving when training stopped. At
L0, by contrast, the median was 9-23 and almost nothing hit the cap.

So the budget bound exactly where the disputed claim lives. That is the same
failure the arm was careful about on the information axis -- a contrast that
looks like architecture but is really something else -- relocated to the
optimiser.

This compares the primary cells against a rerun with a 300-epoch cap and
patience 20, on identical folds, seeds and keys. The comparison is per station
and paired, like every other contrast in the paper, and it reports the change in
the architecture penalty rather than the change in the network's own error,
because the penalty is what Section 4.10 claims.

A shrinking penalty means the section overstated the model class and the number
must be replaced. An unchanged penalty means the budget was not what produced
it, and the section stands with the check recorded.
"""

from __future__ import annotations

import argparse
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

import scripts.final.build_architecture_authority as ARCH
import scripts.final.run_information_ladder_v6_observed as L
import scripts.final.run_plain_tcn_arm as TCN
from thermoroute.significance import cluster_bootstrap_paired_effect
from thermoroute.spatial import huc2_cluster_map, load_station_registry

TREE_SHARDS = L.OUTPUT_DIR / L.SHARD_DIRNAME
TCN_SHARDS = TCN.OUTPUT_DIR / TCN.SHARD_DIRNAME
DEFAULT_OUT = L.V5.FINAL_OUTPUT_ROOT / "tcn_convergence_sensitivity_v1"
SUFFIX = "_conv300"
COMPARATOR = ARCH.COMPARATOR
N_BOOT, BOOT_SEED = 10_000, 0


class SensitivityError(RuntimeError):
    """A precondition failed."""


def _tcn(level: str, horizon: int, fit_seed: int, suffix: str) -> pd.Series:
    return ARCH._read([
        TCN_SHARDS / f"{level}_whole_region_fold{fold}"
                     f"_PlainTCN_unbounded_seed{fit_seed}_h{horizon}{suffix}.parquet"
        for fold in range(L.N_FOLDS)
    ])


def _tree(level: str, horizon: int) -> pd.Series:
    return ARCH._read([
        TREE_SHARDS / f"{level}_whole_region_fold{fold}"
                      f"_{COMPARATOR}_h{horizon}.parquet"
        for fold in range(L.N_FOLDS)
    ])


def build_rows(levels: Sequence[str]) -> list[dict[str, Any]]:
    clusters = huc2_cluster_map(
        load_station_registry(ROOT / "data_usgs" / "station_registry_v1.csv")
    )
    rows: list[dict[str, Any]] = []
    for level in levels:
        for horizon in L.V5.HORIZONS:
            tree = _tree(level, horizon)
            base = {s: _tcn(level, horizon, s, "") for s in TCN.FIT_SEEDS}
            longer = {s: _tcn(level, horizon, s, SUFFIX) for s in TCN.FIT_SEEDS}
            index = tree.index
            for series in (*base.values(), *longer.values()):
                index = index.intersection(series.index)
            groups = index.map(clusters).to_numpy()

            penalties = {}
            for name, block in (("budget_40", base), ("patience_300", longer)):
                penalties[name] = np.mean(
                    [(block[s].loc[index] - tree.loc[index]).to_numpy()
                     for s in sorted(block)], axis=0)
                boot = cluster_bootstrap_paired_effect(
                    penalties[name], groups, statistic="median",
                    n_boot=N_BOOT, seed=BOOT_SEED)
                rows.append({
                    "quantity": "architecture_penalty_tcn_minus_tree",
                    "training": name, "level": level, "horizon": int(horizon),
                    "median_degC": float(np.median(penalties[name])),
                    "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
                    "n_stations": int(len(index)),
                })
            change = penalties["patience_300"] - penalties["budget_40"]
            boot = cluster_bootstrap_paired_effect(
                change, groups, statistic="median", n_boot=N_BOOT, seed=BOOT_SEED)
            rows.append({
                "quantity": "change_from_longer_training",
                "training": "patience_300 - budget_40",
                "level": level, "horizon": int(horizon),
                "median_degC": float(np.median(change)),
                "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
                "n_stations": int(len(index)),
            })
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="*", default=["L0", "L2"])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)

    table = pd.DataFrame(build_rows(args.levels))
    if args.print_only:
        print(table.to_string(index=False))
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out_dir / "tcn_convergence_sensitivity.parquet",
                     index=False)
    (args.out_dir / "tcn_convergence_sensitivity_summary.json").write_text(
        json.dumps({
            "format": "thermoroute.tcn-convergence-sensitivity.v1",
            "question": (
                "whether the L2 architecture penalty reflects the model class "
                "or the 40-epoch training budget that bound there"
            ),
            "primary_budget": {"epochs": 40, "patience": 0},
            "sensitivity_budget": {"epochs": 300, "patience": 20},
            "unchanged": "folds, fit seeds, evaluation keys, anchor, comparator",
            "rows": table.to_dict(orient="records"),
        }, sort_keys=True, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

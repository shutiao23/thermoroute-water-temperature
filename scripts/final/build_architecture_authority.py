#!/usr/bin/env python3
"""The architecture axis: what the model class is worth once information is fixed.

Every other contrast in this paper varies information and holds the model class
constant.  This one does the reverse.  The plain causal TCN and the residual
LightGBM see the same frozen namespace under the same level mask, predict the
same keys on the same folds, and add their residual to the same level-legal
anchor, so

    architecture penalty at level L, lead h
        A(L, h) = median_i [ R_i(L, PlainTCN) - R_i(L, ResidualLightGBM) ]

is a difference of model class and nothing else.  Positive means the network is
worse.  The paper's claim is that the information available at issue time, not
the estimator, is what bounds skill here; that claim is falsifiable exactly to
the extent that this number is small, so it is reported whichever way it comes
out.

Three secondary quantities.  The bounded variant applies the same +/-1 degC
algebraic limit as the manuscript's constrained model, so ``bounded - plain``
measures what that constraint costs a network rather than a tree.  The A-by-L
interaction asks whether the model class starts to matter once the local gauge
is gone -- the regime where a flexible estimator would have the most room to
substitute pattern for observation:

    I = median_i { [R_i(L2,TCN) - R_i(L2,tree)] - [R_i(L0,TCN) - R_i(L0,tree)] }

And the A-by-F interaction asks the same question of future weather: given a
five-variable forcing vector over the whole horizon, does a convolutional
sequence model exploit it differently from a tree?  Both are station-level
double differences formed before any median is taken.

That second one is why the arm is fitted at F0 and F3 rather than at F0 alone.
Section 4.8 could not divide a forcing value by an architecture effect because
the two came from uncrossed cells; crossing them is what makes the ratio a
measurable quantity instead of a category error.

Fit seeds are aggregated seed-first: the station contrast is formed inside each
seed and the paired contrasts are then averaged.  For a deterministic tree
comparator that happens to coincide with averaging the network's risks first,
but the two differ the moment the comparator is stochastic, and writing it the
robust way costs nothing.
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

import scripts.final.run_information_ladder_v6_observed as L
import scripts.final.run_plain_tcn_arm as TCN
from thermoroute.significance import (
    cluster_bootstrap_paired_effect,
    cluster_inference_sensitivity,
)
from thermoroute.spatial import huc2_cluster_map, load_station_registry

TREE_SHARDS = L.OUTPUT_DIR / L.SHARD_DIRNAME
TCN_SHARDS = TCN.OUTPUT_DIR / TCN.SHARD_DIRNAME
DEFAULT_OUT = L.V5.FINAL_OUTPUT_ROOT / "architecture_authority_v1"

FORMAT = "thermoroute.architecture-authority.v1"
STATUS = "POST_OUTCOME_OBSERVED_LINEAGE_ARCHITECTURE"
LEVELS = ("L0", "L2")
FORCINGS = ("F0", "F3_full")
GEOMETRY = "whole_region"
COMPARATOR = "ResidualLightGBM"
N_BOOT, BOOT_SEED = 10_000, 0
MIN_KEYS = 100


class ArchitectureError(RuntimeError):
    """A scoring precondition failed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _station_rmse(frames: list[pd.DataFrame]) -> pd.Series:
    frame = pd.concat(frames, ignore_index=True)
    if frame["key_id"].duplicated().any():
        raise ArchitectureError("a key appears in more than one fold")
    squared = (frame["y_pred"].to_numpy() - frame["y_true"].to_numpy()) ** 2
    counts = frame.groupby("site_id").size()
    mean = pd.Series(squared, index=frame["site_id"].to_numpy()).groupby(level=0).mean()
    return np.sqrt(mean.loc[counts[counts >= MIN_KEYS].index]).sort_index()


def _read(paths: list[Path]) -> pd.Series:
    frames = []
    for path in paths:
        if not path.exists():
            raise ArchitectureError(f"missing shard: {path}")
        frames.append(pd.read_parquet(
            path, columns=["key_id", "site_id", "y_true", "y_pred"]))
    return _station_rmse(frames)


def _prefix(forcing: str) -> str:
    return "" if forcing == "F0" else f"{forcing}_"


def tree_risk(
    level: str, forcing: str, horizon: int, model: str = COMPARATOR
) -> pd.Series:
    return _read([
        TREE_SHARDS / f"{_prefix(forcing)}{level}_{GEOMETRY}_fold{fold}"
                      f"_{model}_h{horizon}.parquet"
        for fold in range(L.N_FOLDS)
    ])


def tcn_risk_by_seed(
    level: str, forcing: str, horizon: int, variant: str
) -> dict[int, pd.Series]:
    return {
        seed: _read([
            TCN_SHARDS / f"{_prefix(forcing)}{level}_{GEOMETRY}_fold{fold}"
                         f"_PlainTCN_{variant}_seed{seed}_h{horizon}.parquet"
            for fold in range(L.N_FOLDS)
        ])
        for seed in TCN.FIT_SEEDS
    }


def _summarise(
    values: np.ndarray, groups: np.ndarray, quantity: str, level: str,
    forcing: str, horizon: int, *, with_loco: bool = False,
) -> dict[str, Any]:
    """Median with a whole-HUC2 cluster interval, oriented so positive is worse."""
    boot = cluster_bootstrap_paired_effect(
        values, groups, statistic="median", n_boot=N_BOOT, seed=BOOT_SEED)
    row = {
        "quantity": quantity, "level": level, "forcing": forcing,
        "horizon": int(horizon),
        "median_degC": float(np.median(values)),
        "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
        "station_fraction_positive": float(np.mean(values > 0)),
        "n_stations": int(len(values)),
    }
    if with_loco:
        sens = cluster_inference_sensitivity(values, groups, statistic="median")
        row.update({
            "loco_min": sens["loco_effect_min"],
            "loco_max": sens["loco_effect_max"],
            "loco_sign_stable": bool(sens["loco_direction_stable"]),
        })
    return row


def build_rows() -> list[dict[str, Any]]:
    clusters = huc2_cluster_map(
        load_station_registry(ROOT / "data_usgs" / "station_registry_v1.csv")
    )
    rows: list[dict[str, Any]] = []
    for horizon in L.V5.HORIZONS:
        cached: dict[tuple[str, str], Any] = {}
        index: pd.Index | None = None
        for level in LEVELS:
            for forcing in FORCINGS:
                tree = tree_risk(level, forcing, horizon)
                plain = tcn_risk_by_seed(level, forcing, horizon, "unbounded")
                bounded = tcn_risk_by_seed(level, forcing, horizon, "bounded")
                shared = tree.index
                for series in (*plain.values(), *bounded.values()):
                    shared = shared.intersection(series.index)
                cached[level, forcing] = (tree, plain, bounded)
                index = shared if index is None else index.intersection(shared)

        # one station set across every cell of the crossing, so the two double
        # differences below are taken over the same stations
        groups = index.map(clusters).to_numpy()
        penalty: dict[tuple[str, str], np.ndarray] = {}
        for (level, forcing), (tree, plain, bounded) in cached.items():
            # seed-first: contrast inside each seed, then average
            unbounded_contrast = np.mean(
                [(plain[s].loc[index] - tree.loc[index]).to_numpy()
                 for s in sorted(plain)], axis=0)
            bounded_contrast = np.mean(
                [(bounded[s].loc[index] - tree.loc[index]).to_numpy()
                 for s in sorted(bounded)], axis=0)
            bound_cost = np.mean(
                [(bounded[s].loc[index] - plain[s].loc[index]).to_numpy()
                 for s in sorted(plain)], axis=0)
            penalty[level, forcing] = unbounded_contrast
            rows.append(_summarise(
                unbounded_contrast, groups,
                "architecture_penalty_tcn_minus_tree", level, forcing, horizon))
            rows.append(_summarise(
                bounded_contrast, groups,
                "architecture_penalty_bounded_tcn_minus_tree",
                level, forcing, horizon))
            rows.append(_summarise(
                bound_cost, groups, "cost_of_the_one_degree_bound",
                level, forcing, horizon))

        for forcing in FORCINGS:
            rows.append(_summarise(
                penalty["L2", forcing] - penalty["L0", forcing], groups,
                "interaction_L2_minus_L0", "L2-L0", forcing, horizon,
                with_loco=True))
        for level in LEVELS:
            rows.append(_summarise(
                penalty[level, "F3_full"] - penalty[level, "F0"], groups,
                "interaction_F3_minus_F0", level, "F3_full-F0", horizon,
                with_loco=True))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)

    rows = build_rows()
    table = pd.DataFrame(rows)
    if args.print_only:
        print(table[["quantity", "level", "forcing", "horizon", "median_degC",
                     "ci_low", "ci_high", "station_fraction_positive"]]
              .to_string(index=False))
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame_path = args.out_dir / "architecture_effects.parquet"
    table.to_parquet(frame_path, index=False)
    summary_path = args.out_dir / "architecture_summary.json"
    summary_path.write_text(json.dumps({
        "format": FORMAT,
        "authority_status": STATUS,
        "comparator": COMPARATOR,
        "geometry": GEOMETRY,
        "forcings": list(FORCINGS),
        "levels": list(LEVELS),
        "orientation": "positive means the network is worse than the tree",
        "information_matching": (
            "identical frozen namespace, identical level mask, identical folds "
            "and evaluation keys, identical level-legal anchor; the network's "
            "only disadvantage is mean imputation where LightGBM routes NaN"
        ),
        "seed_aggregation": (
            "station contrast formed inside each fit seed, then the paired "
            "contrasts averaged per station"
        ),
        "fit_seeds": list(TCN.FIT_SEEDS),
        "chronology": {"post_outcome": True, "confirmatory": False},
        "rows": rows,
    }, sort_keys=True, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    (args.out_dir / "architecture_authority_manifest.json").write_text(
        json.dumps({
            "format": FORMAT + ".manifest",
            "builder_sha256": _sha256_file(Path(__file__).resolve()),
            "runner_sha256": _sha256_file(
                ROOT / "scripts" / "final" / "run_plain_tcn_arm.py"),
            "outputs_sha256": {
                frame_path.name: _sha256_file(frame_path),
                summary_path.name: _sha256_file(summary_path),
            },
        }, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"status": "WRITTEN", "rows": len(table)},
                     sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

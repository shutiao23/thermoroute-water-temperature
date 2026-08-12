#!/usr/bin/env python3
"""Does the architecture penalty depend on how the holdout was drawn?

Section 4.10 reports its headline null -- the model class is worth at most
0.009 degC once local information is present -- under whole-region holdout. The
literature that null speaks to overwhelmingly uses random splits, so the
evidence is not currently in the geometry of the papers it is aimed at. This
builder puts it there.

    architecture penalty at level L, geometry G
        P(L, G) = median_i [ R_i(L, G, PlainTCN) - R_i(L, G, tree) ]

    interaction
        I(L) = median_i { [R_i(L,random,TCN) - R_i(L,random,tree)]
                        - [R_i(L,region,TCN) - R_i(L,region,tree)] }

The sharper question is L2, and it is a question about what the earlier result
*meant*. Whole-region holdout withholds the local gauge and performs spatial
extrapolation at the same time, and an architecture penalty appeared there
(+0.185 degC at one day) alongside a geometry penalty of similar size
(+0.127 degC, Section 4.7). Nothing so far separates them. Random-site holdout
at L2 keeps the missing gauge and hands back the near neighbours, so if the
architecture gap collapses there it was about extrapolation rather than about
scarce local information, and Section 4.10's reading has to change. If it
survives, the reading stands and its scope widens to the split design the
literature actually uses.

Random-site aggregation is seed-first and the distinction matters here more
than anywhere else in the paper: the network contrast is formed *inside* each
split seed, against the tree fitted on that same seed's folds, and only then
averaged. Pooling risks across split seeds first would compare a network and a
tree that were never held out on the same stations.

F0 only. The forcing axis is crossed with architecture at whole-region holdout
in Section 4.10; crossing all three at once is a contrast 116 stations across
about nine effective clusters cannot carry.
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

import scripts.final.build_architecture_authority as ARCH
import scripts.final.run_information_ladder_v6_observed as L
import scripts.final.run_plain_tcn_arm as TCN
from thermoroute.significance import (
    cluster_bootstrap_paired_effect,
    cluster_inference_sensitivity,
)
from thermoroute.spatial import huc2_cluster_map, load_station_registry

TREE_SHARDS = L.OUTPUT_DIR / L.SHARD_DIRNAME
TCN_SHARDS = TCN.OUTPUT_DIR / TCN.SHARD_DIRNAME
DEFAULT_OUT = L.V5.FINAL_OUTPUT_ROOT / "architecture_geometry_interaction_v1"

FORMAT = "thermoroute.architecture-geometry-interaction.v1"
STATUS = "POST_OUTCOME_OBSERVED_LINEAGE_INTERACTION"
LEVELS = ("L0", "L2")
FORCING = "F0"
VARIANT = "unbounded"
COMPARATOR = ARCH.COMPARATOR
N_BOOT, BOOT_SEED = 10_000, 0


class InteractionError(RuntimeError):
    """A scoring precondition failed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree(level: str, horizon: int, split_seed: int | None) -> pd.Series:
    tag = "" if split_seed is None else f"_seed{split_seed}"
    geometry = "whole_region" if split_seed is None else "random_site"
    return ARCH._read([
        TREE_SHARDS / f"{level}_{geometry}{tag}_fold{fold}"
                      f"_{COMPARATOR}_h{horizon}.parquet"
        for fold in range(L.N_FOLDS)
    ])


def _tcn(level: str, horizon: int, split_seed: int | None, fit_seed: int) -> pd.Series:
    tag = "" if split_seed is None else f"_seed{split_seed}"
    geometry = "whole_region" if split_seed is None else "random_site"
    return ARCH._read([
        TCN_SHARDS / f"{level}_{geometry}{tag}_fold{fold}"
                     f"_PlainTCN_{VARIANT}_seed{fit_seed}_h{horizon}.parquet"
        for fold in range(L.N_FOLDS)
    ])


def _cell(level: str, horizon: int) -> dict[str, Any]:
    """Every risk series this level and lead needs, keyed by split seed."""
    risks: dict[Any, Any] = {"region": {"tree": _tree(level, horizon, None)}}
    risks["region"]["tcn"] = {
        f: _tcn(level, horizon, None, f) for f in TCN.FIT_SEEDS
    }
    risks["random"] = {}
    for split_seed in L.RANDOM_SEEDS:
        risks["random"][split_seed] = {
            "tree": _tree(level, horizon, split_seed),
            "tcn": {f: _tcn(level, horizon, split_seed, f) for f in TCN.FIT_SEEDS},
        }
    return risks


def _shared(risks: dict[str, Any]) -> pd.Index:
    index = risks["region"]["tree"].index
    for series in risks["region"]["tcn"].values():
        index = index.intersection(series.index)
    for block in risks["random"].values():
        index = index.intersection(block["tree"].index)
        for series in block["tcn"].values():
            index = index.intersection(series.index)
    return index


def _penalties(
    risks: dict[str, Any], index: pd.Index
) -> tuple[np.ndarray, np.ndarray]:
    """Station-level architecture penalty under each geometry, seed-first."""
    region = np.mean(
        [(risks["region"]["tcn"][f].loc[index]
          - risks["region"]["tree"].loc[index]).to_numpy()
         for f in sorted(risks["region"]["tcn"])],
        axis=0,
    )
    random = np.mean(
        [(block["tcn"][f].loc[index] - block["tree"].loc[index]).to_numpy()
         for split_seed, block in sorted(risks["random"].items())
         for f in sorted(block["tcn"])],
        axis=0,
    )
    return region, random


def _summarise(
    values: np.ndarray, groups: np.ndarray, quantity: str, level: str,
    geometry: str, horizon: int, *, with_loco: bool = False,
) -> dict[str, Any]:
    boot = cluster_bootstrap_paired_effect(
        values, groups, statistic="median", n_boot=N_BOOT, seed=BOOT_SEED)
    row = {
        "quantity": quantity, "level": level, "geometry": geometry,
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
        cells = {level: _cell(level, horizon) for level in LEVELS}
        index = None
        for risks in cells.values():
            shared = _shared(risks)
            index = shared if index is None else index.intersection(shared)
        groups = index.map(clusters).to_numpy()

        penalty: dict[tuple[str, str], np.ndarray] = {}
        for level in LEVELS:
            region, random = _penalties(cells[level], index)
            penalty[level, "whole_region"] = region
            penalty[level, "random_site"] = random
            rows.append(_summarise(
                region, groups, "architecture_penalty_tcn_minus_tree",
                level, "whole_region", horizon))
            rows.append(_summarise(
                random, groups, "architecture_penalty_tcn_minus_tree",
                level, "random_site", horizon))
            rows.append(_summarise(
                random - region, groups, "interaction_random_minus_region",
                level, "random_site-whole_region", horizon, with_loco=True))

        # does the architecture-by-information effect itself survive the split?
        for geometry in ("whole_region", "random_site"):
            rows.append(_summarise(
                penalty["L2", geometry] - penalty["L0", geometry], groups,
                "interaction_L2_minus_L0", "L2-L0", geometry, horizon,
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
        print(table[["quantity", "level", "geometry", "horizon", "median_degC",
                     "ci_low", "ci_high", "station_fraction_positive"]]
              .to_string(index=False))
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame_path = args.out_dir / "architecture_geometry_interaction.parquet"
    table.to_parquet(frame_path, index=False)
    summary_path = args.out_dir / "architecture_geometry_interaction_summary.json"
    summary_path.write_text(json.dumps({
        "format": FORMAT,
        "authority_status": STATUS,
        "comparator": COMPARATOR,
        "forcing": FORCING,
        "variant": VARIANT,
        "orientation": "positive means the network is worse than the tree",
        "seed_aggregation": (
            "the network contrast is formed inside each split seed, against "
            "the tree fitted on that seed's folds, and only then averaged over "
            "split seeds and fit seeds"
        ),
        "split_seeds": list(L.RANDOM_SEEDS),
        "fit_seeds": list(TCN.FIT_SEEDS),
        "question": (
            "whether the architecture penalty that appears at L2 under "
            "whole-region holdout is about scarce local information or about "
            "spatial extrapolation, which that geometry confounds"
        ),
        "chronology": {"post_outcome": True, "confirmatory": False},
        "rows": rows,
    }, sort_keys=True, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    (args.out_dir / "architecture_geometry_interaction_manifest.json").write_text(
        json.dumps({
            "format": FORMAT + ".manifest",
            "builder_sha256": _sha256_file(Path(__file__).resolve()),
            "runner_sha256": _sha256_file(Path(TCN.__file__).resolve()),
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

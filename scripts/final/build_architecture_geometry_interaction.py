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

The forcing axis is included, which an earlier version of this note said the
cohort could not carry. That was asserted rather than measured, and measuring
it showed the assertion was wrong at L0 and right at L2. The minimum detectable
effect for the two-way architecture-by-geometry contrast is 0.003-0.005 degC at
L0 and 0.064-0.190 degC at L2, and the reason for the forty-fold gap is the
result itself: at L0 every station's penalty sits against zero with almost no
spread, so the sign-flip test is powerful, while at L2 the penalty varies widely
across stations and the same test is not. "Unresolvable" and "zero" therefore
mean opposite things in the two rows, and the triple difference is reported with
its own MDE at every cell so a reader can tell which one they are looking at.
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
    cluster_sign_flip_pvalue,
)
from thermoroute.spatial import huc2_cluster_map, load_station_registry

TREE_SHARDS = L.OUTPUT_DIR / L.SHARD_DIRNAME
TCN_SHARDS = TCN.OUTPUT_DIR / TCN.SHARD_DIRNAME
DEFAULT_OUT = L.V5.FINAL_OUTPUT_ROOT / "architecture_geometry_interaction_v1"

FORMAT = "thermoroute.architecture-geometry-interaction.v1"
STATUS = "POST_OUTCOME_OBSERVED_LINEAGE_INTERACTION"
LEVELS = ("L0", "L2")
FORCINGS = ("F0", "F3_full")
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


def _stem(level: str, forcing: str, split_seed: int | None) -> str:
    prefix = "" if forcing == "F0" else f"{forcing}_"
    tag = "" if split_seed is None else f"_seed{split_seed}"
    geometry = "whole_region" if split_seed is None else "random_site"
    return f"{prefix}{level}_{geometry}{tag}"


def _tree(
    level: str, forcing: str, horizon: int, split_seed: int | None
) -> pd.Series:
    stem = _stem(level, forcing, split_seed)
    return ARCH._read([
        TREE_SHARDS / f"{stem}_fold{fold}_{COMPARATOR}_h{horizon}.parquet"
        for fold in range(L.N_FOLDS)
    ])


def _tcn(
    level: str, forcing: str, horizon: int, split_seed: int | None, fit_seed: int
) -> pd.Series:
    stem = _stem(level, forcing, split_seed)
    return ARCH._read([
        TCN_SHARDS / f"{stem}_fold{fold}"
                     f"_PlainTCN_{VARIANT}_seed{fit_seed}_h{horizon}.parquet"
        for fold in range(L.N_FOLDS)
    ])


def _cell(level: str, forcing: str, horizon: int) -> dict[str, Any]:
    """Every risk series this cell needs, keyed by split seed."""
    risks: dict[Any, Any] = {
        "region": {
            "tree": _tree(level, forcing, horizon, None),
            "tcn": {f: _tcn(level, forcing, horizon, None, f) for f in TCN.FIT_SEEDS},
        },
        "random": {},
    }
    for split_seed in L.RANDOM_SEEDS:
        risks["random"][split_seed] = {
            "tree": _tree(level, forcing, horizon, split_seed),
            "tcn": {f: _tcn(level, forcing, horizon, split_seed, f)
                    for f in TCN.FIT_SEEDS},
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


def minimum_detectable_effect(
    values: np.ndarray, groups: np.ndarray, *, alpha: float = 0.05
) -> float:
    """Smallest constant effect this cohort's cluster structure can resolve.

    The observed vector is centred and then shifted until the whole-cluster
    sign-flip tail clears ``alpha``.  Centring first makes the answer a property
    of the dependence structure rather than of the effect that happens to be
    there, so it is comparable across cells and answers "could we have seen it"
    rather than "did we".

    This is the number that separates the two ways an interval can cover zero.
    A cell whose MDE is far below the effect being looked for is evidence of
    absence; a cell whose MDE is above it is absence of evidence, and reporting
    them the same way would be the more damaging of the two errors.
    """
    centred = values - float(np.median(values))
    def tail(shift: float) -> float:
        return cluster_sign_flip_pvalue(centred - shift, groups, statistic="median")
    low, high = 0.0, max(1.0, 4.0 * float(np.std(values)))
    if tail(high) > alpha:
        return float("nan")
    for _ in range(40):
        mid = 0.5 * (low + high)
        if tail(mid) <= alpha:
            high = mid
        else:
            low = mid
    return high


def _summarise(
    values: np.ndarray, groups: np.ndarray, quantity: str, level: str,
    geometry: str, horizon: int, *, forcing: str = "F0",
    with_loco: bool = False, with_mde: bool = False,
) -> dict[str, Any]:
    boot = cluster_bootstrap_paired_effect(
        values, groups, statistic="median", n_boot=N_BOOT, seed=BOOT_SEED)
    row = {
        "quantity": quantity, "level": level, "geometry": geometry,
        "forcing": forcing, "horizon": int(horizon),
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
    if with_mde:
        mde = minimum_detectable_effect(values, groups)
        row["mde_degC"] = mde
        # Deliberately not called "resolvable".  This compares the observed
        # magnitude with what the cohort could detect, which is a statement
        # about power, not about this effect: a cell can clear its own MDE and
        # still have an interval covering zero, and at L2/7d one does.  The
        # interval remains the inference; this field only tells a reader
        # whether a null there is evidence of absence or absence of evidence.
        row["magnitude_at_or_above_mde"] = (
            bool(abs(row["median_degC"]) >= mde) if mde == mde else False
        )
    return row


def build_rows() -> list[dict[str, Any]]:
    clusters = huc2_cluster_map(
        load_station_registry(ROOT / "data_usgs" / "station_registry_v1.csv")
    )
    rows: list[dict[str, Any]] = []
    for horizon in L.V5.HORIZONS:
        cells = {
            (level, forcing): _cell(level, forcing, horizon)
            for level in LEVELS for forcing in FORCINGS
        }
        index = None
        for risks in cells.values():
            shared = _shared(risks)
            index = shared if index is None else index.intersection(shared)
        groups = index.map(clusters).to_numpy()

        penalty: dict[tuple[str, str, str], np.ndarray] = {}
        for (level, forcing), risks in cells.items():
            region, random = _penalties(risks, index)
            penalty[level, forcing, "whole_region"] = region
            penalty[level, forcing, "random_site"] = random
            for geometry, values in (("whole_region", region),
                                     ("random_site", random)):
                rows.append(_summarise(
                    values, groups, "architecture_penalty_tcn_minus_tree",
                    level, geometry, horizon, forcing=forcing, with_mde=True))

        for level in LEVELS:
            for forcing in FORCINGS:
                rows.append(_summarise(
                    penalty[level, forcing, "random_site"]
                    - penalty[level, forcing, "whole_region"],
                    groups, "interaction_random_minus_region", level,
                    "random_site-whole_region", horizon, forcing=forcing,
                    with_loco=True, with_mde=True))
            # does the network's F3 advantage depend on the split design?
            for geometry in ("whole_region", "random_site"):
                rows.append(_summarise(
                    penalty[level, "F3_full", geometry]
                    - penalty[level, "F0", geometry],
                    groups, "interaction_F3_minus_F0", level, geometry,
                    horizon, forcing="F3_full-F0",
                    with_loco=True, with_mde=True))
            # the triple difference: does architecture-by-forcing move with
            # geometry?  Reported with its MDE at every cell, because at L2 the
            # question is whether the cohort could have answered it at all.
            triple = (
                (penalty[level, "F3_full", "random_site"]
                 - penalty[level, "F3_full", "whole_region"])
                - (penalty[level, "F0", "random_site"]
                   - penalty[level, "F0", "whole_region"])
            )
            rows.append(_summarise(
                triple, groups, "triple_difference_AxFxG", level,
                "random_site-whole_region", horizon,
                forcing="F3_full-F0", with_loco=True, with_mde=True))

        for forcing in FORCINGS:
            for geometry in ("whole_region", "random_site"):
                rows.append(_summarise(
                    penalty["L2", forcing, geometry]
                    - penalty["L0", forcing, geometry],
                    groups, "interaction_L2_minus_L0", "L2-L0", geometry,
                    horizon, forcing=forcing, with_loco=True, with_mde=True))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)

    rows = build_rows()
    table = pd.DataFrame(rows)
    if args.print_only:
        print(table[["quantity", "level", "geometry", "forcing", "horizon",
                     "median_degC", "ci_low", "ci_high", "mde_degC",
                     "magnitude_at_or_above_mde"]].to_string(index=False))
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame_path = args.out_dir / "architecture_geometry_interaction.parquet"
    table.to_parquet(frame_path, index=False)
    summary_path = args.out_dir / "architecture_geometry_interaction_summary.json"
    summary_path.write_text(json.dumps({
        "format": FORMAT,
        "authority_status": STATUS,
        "comparator": COMPARATOR,
        "forcings": list(FORCINGS),
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
        "mde_semantics": (
            "mde_degC is the smallest constant effect this cohort's whole-HUC2 "
            "dependence structure resolves at alpha = 0.05, obtained by "
            "centring the observed vector and shifting until the sign-flip "
            "tail clears alpha. It answers 'could we have seen it', not 'did "
            "we'; the cluster bootstrap interval is the inference. A null "
            "whose MDE lies far below the effect sought is evidence of "
            "absence, and one whose MDE lies above it is absence of evidence."
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

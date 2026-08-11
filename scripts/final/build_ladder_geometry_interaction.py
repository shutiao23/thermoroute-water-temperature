#!/usr/bin/env python3
"""The L-by-G interaction: does spatial geometry matter more without local history?

Section 4.5 measured a geometry penalty of 0.004-0.009 degC and could not
resolve it, because every arm kept the held station's own thermal history and
the transferable component was worth about 0.07 degC.  The obvious hypothesis is
that geometry only bites once local information is gone.  This builder tests it.

    geometry penalty at level L
        P(L) = median_i [ R_i(L, whole_region) - R_i(L, random_site) ]

    interaction
        I = median_i { [R_i(L2,region) - R_i(L2,random)]
                     - [R_i(L0,region) - R_i(L0,random)] }

formed as a station-level double difference before any median is taken.  A
positive interaction means whole-region holdout costs more when the model is
thermally ungauged than when it is not -- the reading that would justify the
transfer literature's concern.

Random-site geometry is averaged the way the protocol specifies and the way
that matters: the station contrast is formed *inside* each split seed and the
five paired contrasts are then averaged per station.  Averaging the five risks
first and differencing afterwards is a different quantity, and the difference
is not cosmetic when seeds disagree about which stations are hard.
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
from thermoroute.significance import (
    cluster_bootstrap_paired_effect,
    cluster_inference_sensitivity,
)
from thermoroute.spatial import huc2_cluster_map, load_station_registry

SHARDS = L.OUTPUT_DIR / L.SHARD_DIRNAME
DEFAULT_OUT = L.V5.FINAL_OUTPUT_ROOT / "ladder_geometry_interaction_v1"

FORMAT = "thermoroute.ladder-geometry-interaction.v1"
STATUS = "POST_OUTCOME_OBSERVED_LINEAGE_INTERACTION"
LEVELS = ("L0", "L2")
N_BOOT = 10_000
BOOT_SEED = 0
MIN_KEYS = 100


class InteractionError(RuntimeError):
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
        raise InteractionError("a key appears in more than one fold")
    squared = (frame["y_pred"].to_numpy() - frame["y_true"].to_numpy()) ** 2
    counts = frame.groupby("site_id").size()
    mean = pd.Series(squared, index=frame["site_id"].to_numpy()).groupby(level=0).mean()
    return np.sqrt(mean.loc[counts[counts >= MIN_KEYS].index]).sort_index()


def region_risk(level: str, model: str, horizon: int) -> pd.Series:
    frames = [
        pd.read_parquet(
            SHARDS / f"{level}_whole_region_fold{fold}_{model}_h{horizon}.parquet",
            columns=["key_id", "site_id", "y_true", "y_pred"],
        )
        for fold in range(L.N_FOLDS)
    ]
    return _station_rmse(frames)


def random_risk_by_seed(level: str, model: str, horizon: int) -> dict[int, pd.Series]:
    out: dict[int, pd.Series] = {}
    for seed in L.RANDOM_SEEDS:
        frames = [
            pd.read_parquet(
                SHARDS / f"{level}_random_site_seed{seed}_fold{fold}"
                         f"_{model}_h{horizon}.parquet",
                columns=["key_id", "site_id", "y_true", "y_pred"],
            )
            for fold in range(L.N_FOLDS)
        ]
        out[seed] = _station_rmse(frames)
    return out


def build_rows() -> list[dict[str, Any]]:
    clusters = huc2_cluster_map(
        load_station_registry(ROOT / "data_usgs" / "station_registry_v1.csv")
    )
    rows: list[dict[str, Any]] = []
    for model in L.V5.MODELS:
        for horizon in L.V5.HORIZONS:
            penalty: dict[str, np.ndarray] = {}
            index: pd.Index | None = None
            for level in LEVELS:
                region = region_risk(level, model, horizon)
                per_seed = random_risk_by_seed(level, model, horizon)
                shared = region.index
                for series in per_seed.values():
                    shared = shared.intersection(series.index)
                # seed-first: contrast inside each seed, then average
                contrasts = [
                    (region.loc[shared] - per_seed[seed].loc[shared]).to_numpy()
                    for seed in sorted(per_seed)
                ]
                penalty[level] = np.mean(contrasts, axis=0)
                index = shared if index is None else index.intersection(shared)

            for level in LEVELS:
                values = penalty[level]
                groups = index.map(clusters).to_numpy()
                boot = cluster_bootstrap_paired_effect(
                    values, groups, statistic="median",
                    n_boot=N_BOOT, seed=BOOT_SEED,
                )
                rows.append({
                    "quantity": f"geometry_penalty_at_{level}",
                    "model": model, "horizon": int(horizon),
                    "median_degC": boot["effect"],
                    "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
                    "station_fraction_positive": float(np.mean(values > 0)),
                    "n_stations": len(values),
                })

            double = penalty["L2"] - penalty["L0"]
            groups = index.map(clusters).to_numpy()
            boot = cluster_bootstrap_paired_effect(
                double, groups, statistic="median", n_boot=N_BOOT, seed=BOOT_SEED,
            )
            sens = cluster_inference_sensitivity(-double, groups, statistic="median")
            rows.append({
                "quantity": "interaction_L2_minus_L0",
                "model": model, "horizon": int(horizon),
                "median_degC": boot["effect"],
                "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
                "station_fraction_positive": float(np.mean(double > 0)),
                "loco_min": -sens["loco_effect_max"],
                "loco_max": -sens["loco_effect_min"],
                "loco_sign_stable": bool(sens["loco_direction_stable"]),
                "n_stations": len(double),
            })
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)

    rows = build_rows()
    table = pd.DataFrame(rows)
    if args.print_only:
        print(table[["quantity", "model", "horizon", "median_degC",
                     "ci_low", "ci_high", "station_fraction_positive"]]
              .to_string(index=False))
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame_path = args.out_dir / "ladder_geometry_interaction.parquet"
    table.to_parquet(frame_path, index=False)
    summary_path = args.out_dir / "ladder_geometry_interaction_summary.json"
    summary_path.write_text(json.dumps({
        "format": FORMAT,
        "authority_status": STATUS,
        "seed_aggregation": (
            "station contrast formed inside each split seed, then the five "
            "paired contrasts averaged per station; risks are never averaged "
            "across seeds and differenced afterwards"
        ),
        "random_seeds": list(L.RANDOM_SEEDS),
        "chronology": {"post_outcome": True, "confirmatory": False},
        "rows": rows,
    }, sort_keys=True, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    (args.out_dir / "ladder_geometry_interaction_manifest.json").write_text(
        json.dumps({
            "format": FORMAT + ".manifest",
            "builder_sha256": _sha256_file(Path(__file__).resolve()),
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

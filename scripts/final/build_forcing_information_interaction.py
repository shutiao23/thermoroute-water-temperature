#!/usr/bin/env python3
"""The F-by-L interaction: is future weather worth more when the gauge is gone?

Sections 4.7 and 4.8 measured two things separately -- what local thermal state
is worth, and what future meteorology is worth -- each holding the other fixed
at its default.  Reported that way they invite a reading the data has not been
asked about: that the two are substitutes, so a thermally ungauged reach could
buy back with weather what it lost with the sensor.  This builder asks directly.

    forcing value at level L
        V(L) = median_i [ R_i(L, F0) - R_i(L, F3_full) ]

    interaction
        I = median_i { [R_i(L2,F0) - R_i(L2,F3)] - [R_i(L0,F0) - R_i(L0,F3)] }

formed as a station-level double difference before any median is taken, so the
quantity is a difference of paired contrasts and not a difference of two
marginal medians.  A positive interaction is the substitution reading: forcing
buys more once local water temperature is unavailable.  A negative one is the
complement reading -- forcing is worth *less* to a model that has no thermal
state to apply it to -- which would be the more consequential finding, because
it says the two information sources reinforce rather than replace each other and
that the ungauged case is harder than either axis alone suggests.

Geometry is held at ``whole_region`` throughout.  Both interacting axes vary
here already; letting the third vary as well would produce a three-way contrast
the folds cannot support at 116 stations, and the L-by-G interaction is measured
separately in ``build_ladder_geometry_interaction.py``.

Intervals are whole-HUC2 cluster bootstraps and the interaction additionally
carries a leave-one-HUC2-out range, because a double difference is the quantity
most easily manufactured by one unusual region.
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
DEFAULT_OUT = L.V5.FINAL_OUTPUT_ROOT / "forcing_information_interaction_v1"

FORMAT = "thermoroute.forcing-information-interaction.v1"
STATUS = "POST_OUTCOME_OBSERVED_LINEAGE_INTERACTION"
LEVELS = ("L0", "L2")
GEOMETRY = "whole_region"
N_BOOT, BOOT_SEED = 10_000, 0
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


def risk(level: str, forcing: str, model: str, horizon: int) -> pd.Series:
    """Pooled-across-folds station risk for one (level, forcing) cell."""
    prefix = "" if forcing == "F0" else f"{forcing}_"
    frames = []
    for fold in range(L.N_FOLDS):
        path = (SHARDS / f"{prefix}{level}_{GEOMETRY}_fold{fold}"
                         f"_{model}_h{horizon}.parquet")
        if not path.exists():
            raise InteractionError(f"missing shard: {path}")
        frames.append(pd.read_parquet(
            path, columns=["key_id", "site_id", "y_true", "y_pred"]))
    return _station_rmse(frames)


def build_rows() -> list[dict[str, Any]]:
    clusters = huc2_cluster_map(
        load_station_registry(ROOT / "data_usgs" / "station_registry_v1.csv")
    )
    rows: list[dict[str, Any]] = []
    for model in L.V5.MODELS:
        for horizon in L.V5.HORIZONS:
            risks = {
                (level, forcing): risk(level, forcing, model, horizon)
                for level in LEVELS
                for forcing in ("F0", "F3_full")
            }
            # one station set for every cell, so the double difference below is
            # taken over the same stations at both levels
            index = risks[LEVELS[0], "F0"].index
            for series in risks.values():
                index = index.intersection(series.index)
            value = {
                level: (risks[level, "F0"].loc[index]
                        - risks[level, "F3_full"].loc[index]).to_numpy()
                for level in LEVELS
            }
            groups = index.map(clusters).to_numpy()
            for level in LEVELS:
                gain = value[level]
                boot = cluster_bootstrap_paired_effect(
                    -gain, groups, statistic="median",
                    n_boot=N_BOOT, seed=BOOT_SEED)
                rows.append({
                    "quantity": f"forcing_value_at_{level}",
                    "model": model, "horizon": int(horizon),
                    "median_degC": float(np.median(gain)),
                    "ci_low": -boot["ci_high"], "ci_high": -boot["ci_low"],
                    "station_fraction_positive": float(np.mean(gain > 0)),
                    "n_stations": len(gain),
                })

            double = value["L2"] - value["L0"]
            boot = cluster_bootstrap_paired_effect(
                -double, groups, statistic="median", n_boot=N_BOOT, seed=BOOT_SEED)
            sens = cluster_inference_sensitivity(-double, groups, statistic="median")
            rows.append({
                "quantity": "interaction_L2_minus_L0",
                "model": model, "horizon": int(horizon),
                "median_degC": float(np.median(double)),
                "ci_low": -boot["ci_high"], "ci_high": -boot["ci_low"],
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
    frame_path = args.out_dir / "forcing_information_interaction.parquet"
    table.to_parquet(frame_path, index=False)
    summary_path = args.out_dir / "forcing_information_interaction_summary.json"
    summary_path.write_text(json.dumps({
        "format": FORMAT,
        "authority_status": STATUS,
        "estimand": (
            "station-level double difference of paired F0-minus-F3 contrasts "
            "across information levels, medianed once at the end"
        ),
        "geometry": GEOMETRY,
        "reading": {
            "positive": "forcing substitutes for the lost local gauge",
            "negative": "forcing complements it; the ungauged case is harder "
                        "than either axis alone implies",
        },
        "chronology": {"post_outcome": True, "confirmatory": False},
        "rows": rows,
    }, sort_keys=True, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    (args.out_dir / "forcing_information_interaction_manifest.json").write_text(
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

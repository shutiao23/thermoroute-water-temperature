#!/usr/bin/env python3
"""Score the corrected L ladder: what local thermal information is worth.

Replaces every withdrawn number from the 432-cell ladder.  Each contrast is a
station-level paired difference formed on the exact keys the two rungs share,
then summarised by the unweighted median across stations, with whole-HUC2
cluster resampling -- the same estimand family as the forcing arms, so the two
can be read side by side without being added.

The rungs are nested, so the differences decompose the local-information budget
in a way the F and L axes cannot be combined into:

  L1 - L0     the value of the target site's own long-term statistics, with its
              recent history still visible
  L2 - L1     the value of the recent water-temperature sequence itself
  L2 - L0     the total value of local thermal state, the headline
  L2_U2 - L2  the value of local discharge once thermal history is already gone

Every station is scored under whole-region holdout, so each contrast is a
transfer statement: what a model loses at a gauge whose region never appeared
in training, as successive local observations are withheld.
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
DEFAULT_OUT = L.V5.FINAL_OUTPUT_ROOT / "information_ladder_v6_authority_v1"

FORMAT = "thermoroute.information-ladder-v6-authority.v1"
STATUS = "POST_OUTCOME_OBSERVED_LINEAGE_LADDER_RESULT"
N_BOOT = 10_000
BOOT_SEED = 0
MIN_KEYS = 100
REPORTABLE_STATIONS = 116

CONTRASTS = (
    ("L1", "L0", "value of the target site's own long-term statistics"),
    ("L2", "L1", "value of the recent water-temperature sequence"),
    ("L2", "L0", "total value of local thermal state"),
    ("L2_U2", "L2", "value of local discharge once thermal history is gone"),
)


class LadderAuthorityError(RuntimeError):
    """A scoring precondition failed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_level(level_name: str, model: str, horizon: int) -> pd.DataFrame:
    """All four whole-region folds of one cell, concatenated over held stations."""
    frames = []
    for fold in range(L.N_FOLDS):
        path = SHARDS / f"{level_name}_{L.GEOMETRY}_fold{fold}_{model}_h{horizon}.parquet"
        if not path.exists():
            raise LadderAuthorityError(f"ladder shard missing: {path}")
        frames.append(pd.read_parquet(
            path, columns=["key_id", "site_id", "y_true", "y_pred", "y_anchor"]
        ))
    out = pd.concat(frames, ignore_index=True)
    # The folds partition the 120-station cohort, but scoring happens on the
    # formal registry, which is the 116 reportable stations: two gauges are dry
    # on the test window and one falls below the 100-key rule.
    if out["site_id"].nunique() != REPORTABLE_STATIONS:
        raise LadderAuthorityError(
            f"{level_name} {model} h{horizon} covers {out['site_id'].nunique()} "
            f"stations; the four folds must together cover the {REPORTABLE_STATIONS} "
            "reportable stations exactly once"
        )
    if out["key_id"].duplicated().any():
        raise LadderAuthorityError("a key is scored in more than one fold")
    return out


def station_rmse(frame: pd.DataFrame, keys: set[str]) -> pd.Series:
    block = frame[frame["key_id"].isin(keys)]
    squared = (block["y_pred"].to_numpy() - block["y_true"].to_numpy()) ** 2
    counts = block.groupby("site_id").size()
    mean = pd.Series(squared, index=block["site_id"].to_numpy()).groupby(level=0).mean()
    keep = counts[counts >= MIN_KEYS].index
    return np.sqrt(mean.loc[keep]).sort_index()


def build_rows() -> list[dict[str, Any]]:
    clusters = huc2_cluster_map(
        load_station_registry(ROOT / "data_usgs" / "station_registry_v1.csv")
    )
    rows: list[dict[str, Any]] = []
    for model in L.V5.MODELS:
        for horizon in L.V5.HORIZONS:
            loaded = {
                name: load_level(name, model, horizon) for name in L.LEVEL_NAMES
            }
            # one common key set across every rung, so no contrast is scored on
            # keys another rung declined
            common: set[str] | None = None
            for frame in loaded.values():
                keys = set(frame["key_id"])
                common = keys if common is None else (common & keys)
            if not common:
                raise LadderAuthorityError("rungs share no key")

            risks = {
                name: station_rmse(frame, common) for name, frame in loaded.items()
            }
            for high, low, meaning in CONTRASTS:
                shared = risks[high].index.intersection(risks[low].index)
                if len(shared) < 40:
                    raise LadderAuthorityError(
                        f"{high}-{low} shares only {len(shared)} stations"
                    )
                # candidate-minus-reference: the *worse* rung is the candidate,
                # so a positive value is what the withheld information was worth
                delta = (risks[high].loc[shared] - risks[low].loc[shared]).to_numpy()
                groups = shared.map(clusters).to_numpy()
                boot = cluster_bootstrap_paired_effect(
                    delta, groups, statistic="median", n_boot=N_BOOT, seed=BOOT_SEED,
                )
                sens = cluster_inference_sensitivity(
                    -delta, groups, statistic="median"
                )
                rows.append({
                    "contrast": f"{high}-{low}",
                    "meaning": meaning,
                    "model": model,
                    "horizon": int(horizon),
                    "median_value_degC": boot["effect"],
                    "ci_low": boot["ci_low"],
                    "ci_high": boot["ci_high"],
                    "iqr_low": float(np.percentile(delta, 25)),
                    "iqr_high": float(np.percentile(delta, 75)),
                    "station_fraction_positive": float(np.mean(delta > 0)),
                    "loco_min": -sens["loco_effect_max"],
                    "loco_max": -sens["loco_effect_min"],
                    "loco_sign_stable": bool(sens["loco_direction_stable"]),
                    "median_rmse_high": float(risks[high].loc[shared].median()),
                    "median_rmse_low": float(risks[low].loc[shared].median()),
                    "n_stations": len(shared),
                    "n_common_keys": len(common),
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
        view = table[table["model"] == "LightGBM"]
        print(view[["contrast", "horizon", "median_rmse_low", "median_rmse_high",
                    "median_value_degC", "ci_low", "ci_high",
                    "station_fraction_positive"]].to_string(index=False))
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame_path = args.out_dir / "information_ladder_v6_contrasts.parquet"
    table.to_parquet(frame_path, index=False)
    summary = {
        "format": FORMAT,
        "authority_status": STATUS,
        "geometry": L.GEOMETRY,
        "levels": list(L.LEVEL_NAMES),
        "estimand": "median_i[RMSE_i(withheld rung) - RMSE_i(richer rung)]",
        "chronology": {
            "post_outcome": True,
            "confirmatory": False,
            "replaces": "the 432-cell ladder withdrawn by DLOG-025",
        },
        "non_additivity": (
            "The F and L axes are separate conditional designs and are never "
            "added, divided, or presented as one information budget."
        ),
        "rows": rows,
    }
    summary_path = args.out_dir / "information_ladder_v6_summary.json"
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=1, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "information_ladder_v6_authority_manifest.json").write_text(
        json.dumps({
            "format": FORMAT + ".manifest",
            "builder_sha256": _sha256_file(Path(__file__).resolve()),
            "ladder_manifest_sha256": _sha256_file(L.OUTPUT_DIR / L.MANIFEST_FILENAME),
            "outputs_sha256": {
                frame_path.name: _sha256_file(frame_path),
                summary_path.name: _sha256_file(summary_path),
            },
        }, sort_keys=True, indent=1) + "\n", encoding="utf-8",
    )
    print(json.dumps({"status": "WRITTEN", "rows": len(table)},
                     sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

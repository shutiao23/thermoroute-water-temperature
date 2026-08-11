#!/usr/bin/env python3
"""Attribute the forcing value to meteorological variables (protocol v5a).

The shuffle and shift arms showed the F0/F3 gain is event-scale weather tied to
exact dates.  This says which weather.  Two families are reported side by side
because neither alone is honest about correlated predictors:

  only_X      X realized, every other variable climatological
  without_X   X climatological, every other variable realized

Single-variable values do not sum to the full forcing value -- air temperature,
radiation and humidity move together -- so the shares below exceed 100% and are
not a variance decomposition.  The pair to read is ``only_X`` against
``without_X``: a variable that carries the signal alone *and* whose removal
costs little is one the others can substitute for.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.final.run_forcing_components_v5a as CP
import scripts.final.run_forcing_ladder_v5_observed as V5
from thermoroute.significance import cluster_bootstrap_paired_effect
from thermoroute.spatial import huc2_cluster_map, load_station_registry

SHARDS = CP.OUTPUT_DIR / CP.SHARD_DIRNAME
REFERENCE = V5.FINAL_OUTPUT_ROOT / "forcing_regime_v5_observed" / "forcing_shards_v5_observed"
DEFAULT_OUT = V5.FINAL_OUTPUT_ROOT / "forcing_component_authority_v1"
N_BOOT, BOOT_SEED = 10_000, 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _station_rmse(path: Path) -> pd.Series:
    frame = pd.read_parquet(path, columns=["site_id", "y_true", "y_pred"])
    squared = (frame["y_pred"].to_numpy() - frame["y_true"].to_numpy()) ** 2
    return np.sqrt(
        pd.Series(squared, index=frame["site_id"].to_numpy()).groupby(level=0).mean()
    ).sort_index()


def build_rows() -> list[dict[str, Any]]:
    clusters = huc2_cluster_map(
        load_station_registry(ROOT / "data_usgs" / "station_registry_v1.csv")
    )
    rows: list[dict[str, Any]] = []
    for model in V5.MODELS:
        for horizon in V5.HORIZONS:
            f0 = _station_rmse(
                REFERENCE / f"F0_{model}_h{horizon}_v5_observed.parquet")
            full = _station_rmse(
                REFERENCE / f"F3_full_{model}_h{horizon}_v5_observed.parquet")
            shared = f0.index.intersection(full.index)
            full_value = float(np.median((f0.loc[shared] - full.loc[shared]).to_numpy()))
            for arm_name, _ in CP.arms():
                arm = _station_rmse(SHARDS / f"{arm_name}_{model}_h{horizon}.parquet")
                index = shared.intersection(arm.index)
                value = (f0.loc[index] - arm.loc[index]).to_numpy()
                groups = index.map(clusters).to_numpy()
                boot = cluster_bootstrap_paired_effect(
                    -value, groups, statistic="median",
                    n_boot=N_BOOT, seed=BOOT_SEED)
                median = float(np.median(value))
                rows.append({
                    "arm": arm_name,
                    "family": "only" if arm_name.startswith("only_") else "without",
                    "group": arm_name.split("_", 1)[1],
                    "model": model, "horizon": int(horizon),
                    "forcing_value_degC": median,
                    "ci_low": -boot["ci_high"], "ci_high": -boot["ci_low"],
                    "full_forcing_value_degC": full_value,
                    "share_of_full": median / full_value if full_value > 0 else np.nan,
                    "n_stations": len(index),
                })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)
    rows = build_rows()
    table = pd.DataFrame(rows)
    if args.print_only:
        view = table[(table["model"] == "LightGBM") & (table["horizon"] == 7)]
        print(view[["arm", "forcing_value_degC", "ci_low", "ci_high",
                    "share_of_full"]].to_string(index=False))
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame_path = args.out_dir / "forcing_component_values.parquet"
    table.to_parquet(frame_path, index=False)
    summary_path = args.out_dir / "forcing_component_summary.json"
    summary_path.write_text(json.dumps({
        "format": "thermoroute.forcing-component-authority.v1",
        "authority_status": "SEALED_PRE_OUTCOME_COMPONENT_RESULT",
        "non_additivity": (
            "shares exceed 100% because the meteorological variables are "
            "correlated; this is not a variance decomposition"
        ),
        "chronology": {"specification_sealed_before_outcome": True,
                       "post_outcome": False},
        "rows": rows,
    }, sort_keys=True, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    (args.out_dir / "forcing_component_authority_manifest.json").write_text(
        json.dumps({"builder_sha256": _sha256_file(Path(__file__).resolve()),
                    "outputs_sha256": {
                        frame_path.name: _sha256_file(frame_path),
                        summary_path.name: _sha256_file(summary_path)}},
                   sort_keys=True, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"status": "WRITTEN", "rows": len(table)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

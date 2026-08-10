#!/usr/bin/env python3
"""Score the transparent forced-hybrid reference under F0 and F3, and compare.

The manuscript has had no process benchmark it would stand behind: the
air2stream-style arm is described five times as an unofficial, unvalidated
variant, and a reference the authors will not defend is not a reference.  The
transparent forced thermal-response model is the alternative that referee
comment proposed --

    T_{next} = T_{current} + alpha_i * (T_eq(F_{next}) - T_{current})

with a per-station response rate and a linear equilibrium in the five
atmospheric fields, calibrated on 2006-2015 only.  It is not a substitute for a
full process model; it is a physically-directed reference simple enough to
defend line by line.

The scientific question this answers is not "does the hybrid beat the tree".
It is whether the forcing value the trees found is available to a simple
thermal-response law.  Both arms are rolled forward on the same held-out keys
and the same channel definitions the trees used:

* if V_F(hybrid) is close to V_F(tree), the value of future weather is mostly
  simple thermal response and the machine learning is not adding mechanism;
* if V_F(tree) is materially larger, the trees are exploiting nonlinearity or
  state dependence the linear equilibrium cannot express.

The hybrid reads only its calibrated parameters and the evaluation-window
forcing; no holdout outcome enters its fitting, and the parameters were frozen
by a separate create-only calibration under its own execution seal.
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

import scripts.final.run_forcing_ladder_v5_observed as V5
import scripts.final.run_forcing_placebo_v5a_shuffle as P
from thermoroute.forced_hybrid import (
    ATMOSPHERIC_VARIABLES,
    FORCING_UNITS,
    ForcedHybridModel,
)
from thermoroute.significance import cluster_bootstrap_paired_effect
from thermoroute.spatial import huc2_cluster_map, load_station_registry

FINAL = ROOT / "outputs" / "final"
MODEL_JSON = FINAL / "forced_hybrid_reference_v1_model.json"
SHARDS = FINAL / "forcing_regime_v5_observed" / "forcing_shards_v5_observed"
DEFAULT_OUT = FINAL / "forced_hybrid_forcing_value.parquet"

FORMAT = "thermoroute.forced-hybrid-forcing-value.v1"
CHANNELS = {"F0": "F0", "F3_full": "F3_full"}
N_BOOT = 10_000
BOOT_SEED = 0
MIN_KEYS = 100


class HybridScoreError(RuntimeError):
    """An input or coverage check failed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model() -> ForcedHybridModel:
    if not MODEL_JSON.exists():
        raise HybridScoreError(
            f"calibrated parameters missing: {MODEL_JSON}; run "
            "scripts/final/run_forced_hybrid_reference.py --execute first"
        )
    return ForcedHybridModel.from_json(MODEL_JSON.read_text(encoding="utf-8"))


def evaluation_keys(horizon: int) -> pd.DataFrame:
    """The exact keys the tree arms were scored on, at one lead."""
    path = SHARDS / f"F0_LightGBM_h{horizon}_v5_observed.parquet"
    return pd.read_parquet(
        path, columns=["site_id", "issue_date", "target_date", "y_true"]
    )


def score(model: ForcedHybridModel, panel: pd.DataFrame,
          horizon: int) -> pd.DataFrame:
    keys = evaluation_keys(horizon)
    indexed = panel.set_index(["site_id", "DATE"]).sort_index()
    # the hybrid contract fixes the schema order; V5 lists the same five
    # variables in a different order and the model rejects that
    atmospheric = list(ATMOSPHERIC_VARIABLES)

    rows: list[dict[str, Any]] = []
    for (station, issue_date), block in keys.groupby(
        ["site_id", "issue_date"], sort=True
    ):
        try:
            issue_row = indexed.loc[(station, issue_date)]
        except KeyError:
            continue
        current = float(issue_row["WTEMP"])
        if not np.isfinite(current):
            continue
        issue_forcing = {v: float(issue_row[v]) for v in atmospheric}
        if not all(np.isfinite(list(issue_forcing.values()))):
            continue
        valid = pd.date_range(issue_date, periods=horizon + 1, freq="D")[1:]
        try:
            future = indexed.loc[(station, valid[0]):(station, valid[-1])]
        except KeyError:
            continue
        future = future.reset_index()
        future = future[future["DATE"].isin(valid)]
        if len(future) != horizon or not np.isfinite(
            future[atmospheric].to_numpy(float)
        ).all():
            continue
        # the contract fixes the future schema exactly, step column included
        future_forcing = future[["DATE", *atmospheric]].rename(
            columns={"DATE": "valid_date"}
        )
        future_forcing.insert(
            0, "step", np.arange(1, horizon + 1, dtype=np.int16)
        )
        truth = float(block["y_true"].iloc[0])
        record: dict[str, Any] = {
            "site_id": station, "issue_date": issue_date, "horizon": horizon,
            "y_true": truth,
        }
        for channel in CHANNELS:
            # F0 forbids a future payload by contract: it persists the issue-day
            # atmosphere through the rollout, which is exactly the information
            # boundary the F0 tree arm has.
            path_out = model.rollout(
                station=station, issue_date=issue_date,
                temperature_current=current, issue_forcing=issue_forcing,
                future_forcing=None if channel == "F0" else future_forcing,
                horizon=horizon,
                channel=channel, forcing_units=FORCING_UNITS,
            )
            record[f"pred_{channel}"] = float(path_out["y_pred"].iloc[-1])
        rows.append(record)
    if not rows:
        raise HybridScoreError(f"no key survived the hybrid rollout at h{horizon}")
    return pd.DataFrame(rows)


def summarise(scored: pd.DataFrame, clusters: dict[str, str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for horizon, block in scored.groupby("horizon", sort=True):
        per_station = []
        for station, rows in block.groupby("site_id", sort=True):
            if len(rows) < MIN_KEYS:
                continue
            truth = rows["y_true"].to_numpy()
            entry = {"site_id": station}
            for channel in CHANNELS:
                residual = rows[f"pred_{channel}"].to_numpy() - truth
                entry[channel] = float(np.sqrt(np.mean(residual ** 2)))
            per_station.append(entry)
        frame = pd.DataFrame(per_station)
        if len(frame) < 40:
            continue
        value = (frame["F0"] - frame["F3_full"]).to_numpy()
        groups = frame["site_id"].map(clusters).to_numpy()
        boot = cluster_bootstrap_paired_effect(
            -value, groups, statistic="median", n_boot=N_BOOT, seed=BOOT_SEED,
        )
        out.append({
            "model": "TransparentForcedThermalResponse",
            "horizon": int(horizon),
            "median_station_rmse_F0": float(frame["F0"].median()),
            "median_station_rmse_F3": float(frame["F3_full"].median()),
            "median_forcing_value": float(np.median(value)),
            "forcing_value_ci_low": -boot["ci_high"],
            "forcing_value_ci_high": -boot["ci_low"],
            "station_win_fraction_F3_better": float(np.mean(value > 0)),
            "n_stations": len(frame),
        })
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)

    model = load_model()
    bounds = P.capture_placebo_inputs()
    raw, _stations, _ = V5._load_raw_panel_from_bounds(bounds)
    panel = V5._validated_raw_frame(raw)
    clusters = huc2_cluster_map(
        load_station_registry(ROOT / "data_usgs" / "station_registry_v1.csv")
    )

    scored = pd.concat(
        [score(model, panel, horizon) for horizon in V5.HORIZONS], ignore_index=True
    )
    rows = summarise(scored, clusters)
    table = pd.DataFrame(rows)
    if args.print_only:
        print(table.to_string(index=False))
        return 0

    table.to_parquet(args.out, index=False)
    manifest = {
        "format": FORMAT,
        "calibrated_model_sha256": _sha256_file(MODEL_JSON),
        "builder_sha256": _sha256_file(Path(__file__).resolve()),
        "output_sha256": _sha256_file(args.out),
        "note": (
            "Physically-directed reference, calibrated on 2006-2015 only and "
            "rolled forward on the tree arms' own held-out keys.  Answers "
            "whether the trees' forcing value is available to a simple thermal-"
            "response law, not whether the hybrid beats the trees."
        ),
    }
    (args.out.parent / "forced_hybrid_forcing_value_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=1) + "\n", encoding="utf-8",
    )
    print(json.dumps({"status": "WRITTEN", "rows": len(table)},
                     sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

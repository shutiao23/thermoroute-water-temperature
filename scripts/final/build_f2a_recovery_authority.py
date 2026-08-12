#!/usr/bin/env python3
"""What a real fixed-lead temperature forecast recovers of the oracle's value.

Everything upstream of this is an oracle: F3 hands the model the *realized*
future weather, which no operational system has.  The F2a arm replaces realized
air temperature with an archived forecast at the same fixed lead and changes
nothing else, so the ratio below is the only number in the paper that speaks to
what is actually attainable.

    forecast value      V_f = median_i [ R_i(F0) - R_i(F2a) ]
    oracle value        V_o = median_i [ R_i(F0) - R_i(F3_temperature_only) ]
    recovery            V_f / V_o

The denominator is ``F3_temperature_only`` -- realized air temperature, all
other meteorology climatological -- and never ``F3_full``.  F2a is a
temperature-only product, so dividing by the five-variable oracle would charge
the forecast for variables it never claimed to supply.  The component arms make
this a small concession rather than a large one: at seven days air temperature
alone already carries 95% of the full oracle gain.

The ratio is a ratio of medians and is reported as a point estimate with no
interval, because the paired-bootstrap machinery gives intervals for
differences, not for quotients of them, and a delta-method interval on a ratio
whose denominator is itself estimated would be an interval in name only.  The
two differences each carry their own clustered interval; a reader who wants
uncertainty on the ratio should read those.

Two constraints inherited from the sealed protocol and repeated because the
shorthand is tempting.  The forecast series is a *fixed-lead composite* -- for
each valid time, what a run issued h days earlier said -- not a coherent
trajectory from one initialization and not an as-issued operational archive.
And the archive covers 2021-2023, so this measures those three years at these
stations, not a climatological expectation.
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

import scripts.final.run_f2a_arm_v5a as F2A
import scripts.final.run_forcing_components_v5a as CP
import scripts.final.run_forcing_ladder_v5_observed as V5
from thermoroute.significance import cluster_bootstrap_paired_effect
from thermoroute.spatial import huc2_cluster_map, load_station_registry

F2A_SHARDS = F2A.OUTPUT_DIR / F2A.SHARD_DIRNAME
COMPONENT_SHARDS = CP.OUTPUT_DIR / CP.SHARD_DIRNAME
REFERENCE = (
    V5.FINAL_OUTPUT_ROOT / "forcing_regime_v5_observed" / "forcing_shards_v5_observed"
)
DEFAULT_OUT = V5.FINAL_OUTPUT_ROOT / "f2a_recovery_authority_v1"

FORMAT = "thermoroute.f2a-recovery-authority.v1"
STATUS = "SEALED_PRE_OUTCOME_F2A_RECOVERY"
#: the only legal denominator; ``only_air_temperature`` is F3_temperature_only
DENOMINATOR_ARM = "only_air_temperature"
N_BOOT, BOOT_SEED = 10_000, 0


class RecoveryError(RuntimeError):
    """A scoring precondition failed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _station_rmse(path: Path) -> pd.Series:
    if not path.exists():
        raise RecoveryError(f"missing shard: {path}")
    frame = pd.read_parquet(path, columns=["site_id", "y_true", "y_pred"])
    squared = (frame["y_pred"].to_numpy() - frame["y_true"].to_numpy()) ** 2
    return np.sqrt(
        pd.Series(squared, index=frame["site_id"].to_numpy()).groupby(level=0).mean()
    ).sort_index()


def _paired(
    baseline: pd.Series, arm: pd.Series, index: pd.Index, clusters: dict[str, str]
) -> dict[str, Any]:
    """Median station-level gain over F0, with a whole-HUC2 cluster interval."""
    value = (baseline.loc[index] - arm.loc[index]).to_numpy()
    groups = index.map(clusters).to_numpy()
    # the bootstrap is written for "candidate minus reference"; negate so the
    # reported interval brackets the gain rather than its mirror image
    boot = cluster_bootstrap_paired_effect(
        -value, groups, statistic="median", n_boot=N_BOOT, seed=BOOT_SEED
    )
    return {
        "value_degC": float(np.median(value)),
        "ci_low": -boot["ci_high"],
        "ci_high": -boot["ci_low"],
        "station_fraction_positive": float(np.mean(value > 0)),
    }


def build_rows() -> list[dict[str, Any]]:
    clusters = huc2_cluster_map(
        load_station_registry(ROOT / "data_usgs" / "station_registry_v1.csv")
    )
    rows: list[dict[str, Any]] = []
    for model in V5.MODELS:
        for horizon in V5.HORIZONS:
            f0 = _station_rmse(REFERENCE / f"F0_{model}_h{horizon}_v5_observed.parquet")
            oracle = _station_rmse(
                COMPONENT_SHARDS / f"{DENOMINATOR_ARM}_{model}_h{horizon}.parquet"
            )
            forecast = _station_rmse(
                F2A_SHARDS / f"{F2A.ARM}_{model}_h{horizon}.parquet"
            )
            index = f0.index.intersection(oracle.index).intersection(forecast.index)
            if len(index) < 100:
                raise RecoveryError(
                    f"only {len(index)} shared stations at {model}/h{horizon}"
                )
            oracle_gain = _paired(f0, oracle, index, clusters)
            forecast_gain = _paired(f0, forecast, index, clusters)
            denominator = oracle_gain["value_degC"]
            rows.append({
                "model": model,
                "horizon": int(horizon),
                "n_stations": len(index),
                "oracle_value_degC": denominator,
                "oracle_ci_low": oracle_gain["ci_low"],
                "oracle_ci_high": oracle_gain["ci_high"],
                "forecast_value_degC": forecast_gain["value_degC"],
                "forecast_ci_low": forecast_gain["ci_low"],
                "forecast_ci_high": forecast_gain["ci_high"],
                "forecast_station_fraction_positive":
                    forecast_gain["station_fraction_positive"],
                "recovery_fraction": (
                    forecast_gain["value_degC"] / denominator
                    if denominator > 0 else float("nan")
                ),
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
        print(table[["model", "horizon", "oracle_value_degC",
                     "forecast_value_degC", "recovery_fraction",
                     "n_stations"]].to_string(index=False))
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame_path = args.out_dir / "f2a_recovery_values.parquet"
    table.to_parquet(frame_path, index=False)
    summary_path = args.out_dir / "f2a_recovery_summary.json"
    summary_path.write_text(json.dumps({
        "format": FORMAT,
        "authority_status": STATUS,
        "estimand": (
            "median_i[RMSE_i(F0) - RMSE_i(F2a)] divided by "
            "median_i[RMSE_i(F0) - RMSE_i(F3_temperature_only)]"
        ),
        "denominator": DENOMINATOR_ARM,
        "forbidden": [
            "recovery against F3_full",
            "operational recovery fraction",
            "a confidence interval on the ratio",
        ],
        "semantics": {
            "fixed_lead_composite": True,
            "coherent_single_initialization": False,
            "as_issued_operational_archive": False,
            "forecast_archive_years": "2021-2023",
        },
        "interval_note": (
            "each difference carries a whole-HUC2 cluster bootstrap interval; "
            "the ratio is a point estimate because an interval on a quotient of "
            "two estimated medians would not be one"
        ),
        "chronology": {"specification_sealed_before_outcome": True,
                       "post_outcome": False},
        "rows": rows,
    }, sort_keys=True, indent=1, allow_nan=False, default=str) + "\n",
        encoding="utf-8")
    (args.out_dir / "f2a_recovery_authority_manifest.json").write_text(
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

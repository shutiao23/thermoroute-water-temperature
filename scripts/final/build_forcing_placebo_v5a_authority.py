#!/usr/bin/env python3
"""Score the shuffled-forcing placebo and apply the sealed decision rules.

Reads the placebo shards written by
``scripts/final/run_forcing_placebo_v5a_shuffle.py`` and the reference F0/F3
shards from the v5 observed run, and produces, per model and horizon:

* the placebo forcing value ``V_F(shuffle) = median_i[RMSE_i(F0) - RMSE_i(shuffle)]``
  formed station-first inside each permutation seed and then averaged over the
  five seeds per station, exactly as the sealed protocol specifies;
* the station-level placebo contrast against the true oracle arm,
  ``median_i[RMSE_i(shuffle) - RMSE_i(F3_true)]``, formed per station before any
  median is taken;
* whole-HUC2 cluster bootstrap intervals, exact sign-flip tails, equal-HUC
  aggregates and leave-one-HUC2-out ranges for both;
* the retained fraction ``V_F(shuffle) / V_F(true)``, which is the quantity
  decision rules P1-P3 are stated in.

It then applies the sealed rules and prints the verdict.  The rules were fixed
in ``protocols/wrr_forcing_placebo_protocol_v5a.yaml`` and sealed before any
placebo outcome existed, so the verdict here is a lookup, not a judgement:

* P1  retained < 25%  -> the forcing value is event-scale weather information
* P2  25% <= retained <= 60% -> substantially seasonal phase; the headline is
      restated as the shuffle-corrected difference
* P3  retained > 60% -> the F0/F3 forcing value is withdrawn as an information
      claim and reported as a calendar/seasonal-phase artifact

P3 is an admissible outcome and must be reported if it occurs.
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

import scripts.final.run_forcing_placebo_v5a_shuffle as P
from thermoroute.significance import (
    cluster_bootstrap_paired_effect,
    cluster_inference_sensitivity,
)
from thermoroute.spatial import huc2_cluster_map, load_station_registry

FINAL = ROOT / "outputs" / "final"
REFERENCE_SHARDS = FINAL / "forcing_regime_v5_observed" / "forcing_shards_v5_observed"
PLACEBO_SHARDS = P.OUTPUT_DIR / P.SHARD_DIRNAME
DEFAULT_OUT_DIR = FINAL / "forcing_placebo_v5a_authority_v1"

AUTHORITY_FORMAT = "thermoroute.forcing-placebo-v5a-shuffle-authority.v1"
AUTHORITY_STATUS = "SEALED_PRE_OUTCOME_PLACEBO_RESULT"

N_BOOT = 10_000
BOOT_SEED = 0
MIN_KEYS = 100

#: Sealed thresholds; changing either of these is a protocol amendment.
P1_MAX_RETAINED = 0.25
P3_MIN_RETAINED = 0.60


class AuthorityError(RuntimeError):
    """A placebo scoring precondition failed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rmse(residual: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(residual))))


def _station_rmse(frame: pd.DataFrame) -> pd.Series:
    squared = (frame["y_pred"].to_numpy() - frame["y_true"].to_numpy()) ** 2
    mean = pd.Series(squared, index=frame["site_id"].to_numpy()).groupby(level=0).mean()
    return np.sqrt(mean).sort_index()


def _station_keys(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby("site_id", sort=True).size()


def load_reference(model: str, horizon: int) -> tuple[pd.Series, pd.Series]:
    out = []
    for arm in ("F0", "F3_full"):
        path = REFERENCE_SHARDS / f"{arm}_{model}_h{horizon}_v5_observed.parquet"
        if not path.exists():
            raise AuthorityError(f"reference shard missing: {path}")
        out.append(pd.read_parquet(path, columns=["site_id", "y_true", "y_pred"]))
    return _station_rmse(out[0]), _station_rmse(out[1])


def load_placebo(model: str, horizon: int) -> dict[int, pd.Series]:
    per_seed: dict[int, pd.Series] = {}
    for seed in P.SHUFFLE_SEEDS:
        path = PLACEBO_SHARDS / P.shard_filename(model, horizon, seed)
        if not path.exists():
            raise AuthorityError(f"placebo shard missing: {path}")
        frame = pd.read_parquet(path, columns=["site_id", "y_true", "y_pred"])
        if int(_station_keys(frame).min()) < MIN_KEYS:
            raise AuthorityError(f"placebo shard below the 100-key rule: {path}")
        per_seed[seed] = _station_rmse(frame)
    return per_seed


def _infer(deltas: np.ndarray, clusters: np.ndarray) -> dict[str, Any]:
    boot = cluster_bootstrap_paired_effect(
        deltas, clusters, statistic="median", n_boot=N_BOOT, seed=BOOT_SEED,
    )
    sens = cluster_inference_sensitivity(deltas, clusters, statistic="median")
    per_cluster = [
        float(np.median(deltas[clusters == cluster])) for cluster in np.unique(clusters)
    ]
    return {
        "median": boot["effect"],
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
        "equal_huc": float(np.mean(per_cluster)),
        "loco_min": sens["loco_effect_min"],
        "loco_max": sens["loco_effect_max"],
        "loco_sign_stable": bool(sens["loco_direction_stable"]),
        "sign_flip_p_one_sided": sens["sign_flip_p_one_sided_sensitivity"],
        "n_stations": len(deltas),
        "n_clusters": len(np.unique(clusters)),
    }


def verdict(retained: float) -> tuple[str, str]:
    if retained < P1_MAX_RETAINED:
        return "P1_event_scale_supported", (
            "the forcing value is reported as event-scale weather information"
        )
    if retained > P3_MIN_RETAINED:
        return "P3_finding_withdrawn", (
            "the F0/F3 forcing value is withdrawn as an information claim and "
            "reported as a calendar/seasonal-phase artifact"
        )
    return "P2_partly_seasonal", (
        "the forcing value contains a substantial seasonal-phase component; the "
        "headline is restated as the shuffle-corrected difference"
    )


def build_rows() -> list[dict[str, Any]]:
    cluster_map = huc2_cluster_map(load_station_registry(ROOT / "data_usgs" / "station_registry_v1.csv"))
    rows: list[dict[str, Any]] = []
    for model in ("LightGBM", "ResidualLightGBM"):
        for horizon in (1, 3, 7):
            rmse_f0, rmse_true = load_reference(model, horizon)
            per_seed = load_placebo(model, horizon)
            sites = rmse_f0.index
            for name, series in [("F3_true", rmse_true), *per_seed.items()]:
                if not series.index.equals(sites):
                    raise AuthorityError(
                        f"station set differs between F0 and {name} at {model} h{horizon}"
                    )

            # seed-first: station contrast inside each seed, then averaged
            placebo_value = np.mean(
                [(rmse_f0 - series).to_numpy() for series in per_seed.values()], axis=0
            )
            placebo_vs_true = np.mean(
                [(series - rmse_true).to_numpy() for series in per_seed.values()], axis=0
            )
            true_value = (rmse_f0 - rmse_true).to_numpy()

            clusters = sites.map(cluster_map).to_numpy()
            if pd.isna(clusters).any():
                raise AuthorityError("a station is absent from the HUC2 map")

            # candidate-minus-reference convention: negative favours the candidate
            placebo = _infer(-placebo_value, clusters)
            contrast = _infer(placebo_vs_true, clusters)
            true_median = float(np.median(true_value))
            placebo_median = float(np.median(placebo_value))
            retained = placebo_median / true_median if true_median > 0 else float("nan")
            rule, meaning = verdict(retained)
            rows.append({
                "model": model,
                "horizon": horizon,
                "true_forcing_value": true_median,
                "placebo_forcing_value": placebo_median,
                "placebo_forcing_value_ci_low": -placebo["ci_high"],
                "placebo_forcing_value_ci_high": -placebo["ci_low"],
                "retained_fraction": retained,
                "placebo_minus_true_median": contrast["median"],
                "placebo_minus_true_ci_low": contrast["ci_low"],
                "placebo_minus_true_ci_high": contrast["ci_high"],
                "placebo_minus_true_sign_flip_p": contrast["sign_flip_p_one_sided"],
                "placebo_minus_true_loco_stable": contrast["loco_sign_stable"],
                "station_win_fraction_placebo_worse": float(np.mean(placebo_vs_true > 0)),
                "n_stations": placebo["n_stations"],
                "n_clusters": placebo["n_clusters"],
                "n_seeds": len(per_seed),
                "decision_rule": rule,
                "decision_meaning": meaning,
            })
    return rows


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary = [r for r in rows if r["horizon"] in (3, 7)]
    rules = sorted({r["decision_rule"] for r in primary})
    return {
        "format": AUTHORITY_FORMAT,
        "authority_status": AUTHORITY_STATUS,
        "arm": P.PLACEBO_ARM,
        "chronology": {
            "specification_sealed_before_outcome": True,
            "seal": "protocols/wrr_forcing_placebo_protocol_v5a_seal.json",
            "post_outcome": False,
            "note": (
                "The reference F0/F3 result is post-outcome; this control arm was "
                "specified and sealed before its own outcome existed."
            ),
        },
        "sealed_thresholds": {
            "P1_max_retained": P1_MAX_RETAINED,
            "P3_min_retained": P3_MIN_RETAINED,
        },
        "decision_rules_triggered_at_3_and_7_days": rules,
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)

    rows = build_rows()
    summary = summarise(rows)
    if args.print_only:
        print(json.dumps(summary, sort_keys=True, indent=1))
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame_path = args.out_dir / "forcing_placebo_v5a_shuffle_rows.parquet"
    pd.DataFrame(rows).to_parquet(frame_path, index=False)
    summary_path = args.out_dir / "forcing_placebo_v5a_shuffle_summary.json"
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=1, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "format": AUTHORITY_FORMAT + ".manifest",
        "builder_sha256": _sha256_file(Path(__file__).resolve()),
        "placebo_lineage_manifest_sha256": _sha256_file(
            P.OUTPUT_DIR / P.MANIFEST_FILENAME
        ),
        "outputs_sha256": {
            frame_path.name: _sha256_file(frame_path),
            summary_path.name: _sha256_file(summary_path),
        },
    }
    (args.out_dir / "forcing_placebo_v5a_authority_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=1) + "\n", encoding="utf-8",
    )
    print(json.dumps({"status": "WRITTEN", "out_dir": str(args.out_dir),
                      "rules": summary["decision_rules_triggered_at_3_and_7_days"]},
                     sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

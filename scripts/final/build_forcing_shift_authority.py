#!/usr/bin/env python3
"""Score the +/-7-day shift arms and apply sealed decision rule P4.

The shuffle arm established that the forcing value is event-scale rather than
seasonal-phase information.  It could not separate day-to-day correspondence
from sub-monthly synoptic persistence, because a donor drawn from the same
station-month can land within a few days of the true valid time.  These arms
displace the realized future by a whole week, which leaves climate,
near-seasonal phase and local weather persistence in place and removes exact
timing.

The sealed boundary rule is enforced here rather than in the runner.
Displacement leaves seven days at one end of each station's record without a
donor; those keys received a climatology substitution and are recorded per key.
Rule: such keys are dropped from *both* shift arms **and from the true arm**,
so all three are scored on one common shift-registry that is named separately
and never mixed with the primary 358,765-key registry.

Decision rule P4, fixed before any shift outcome existed: the +7-day arm is the
primary timing control.  If its forcing value is not materially below the true
value, the model is using seasonal level rather than event timing and the
lead-structure interpretation of Section 4.7 is withdrawn.  The -7 arm is the
weaker control of the pair -- displacing backwards moves the future window
toward information already available at issue time -- and the two are reported
separately and never averaged.
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
import scripts.final.run_forcing_placebo_v5a_shift as SH
from thermoroute.significance import (
    cluster_bootstrap_paired_effect,
    cluster_inference_sensitivity,
)
from thermoroute.spatial import huc2_cluster_map, load_station_registry

FINAL = ROOT / "outputs" / "final"
TRUE_SHARDS = FINAL / "forcing_regime_v5_observed" / "forcing_shards_v5_observed"
SHIFT_SHARDS = SH.OUTPUT_DIR / SH.SHARD_DIRNAME
DEFAULT_OUT = FINAL / "forcing_placebo_v5a_shift_authority_v1"

FORMAT = "thermoroute.forcing-placebo-v5a-shift-authority.v1"
STATUS = "SEALED_PRE_OUTCOME_PLACEBO_RESULT"
PRIMARY_ARM = "F3_shift_plus7"
N_BOOT = 10_000
BOOT_SEED = 0
MIN_KEYS = 100
#: Sealed: "materially below" is read as retaining less than this share of the
#: true forcing value, the same 25% boundary rule P1 uses.
P4_MAX_RETAINED = 0.25


class ShiftAuthorityError(RuntimeError):
    """A scoring precondition failed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def common_shift_registry(model: str, horizon: int) -> set[str]:
    """Keys usable in the shift contrast: no boundary substitution in either arm."""
    usable: set[str] | None = None
    for arm in SH.SHIFT_ARMS:
        path = SHIFT_SHARDS / SH.shard_filename(arm, model, horizon)
        if not path.exists():
            raise ShiftAuthorityError(f"shift shard missing: {path}")
        frame = pd.read_parquet(path, columns=["key_id", "forcing_substitution_count"])
        clean = set(frame.loc[frame["forcing_substitution_count"] == 0, "key_id"])
        usable = clean if usable is None else (usable & clean)
    if not usable:
        raise ShiftAuthorityError(f"empty common shift registry for {model} h{horizon}")
    return usable


def station_rmse(path: Path, keys: set[str]) -> pd.Series:
    frame = pd.read_parquet(path, columns=["key_id", "site_id", "y_true", "y_pred"])
    frame = frame[frame["key_id"].isin(keys)]
    squared = (frame["y_pred"].to_numpy() - frame["y_true"].to_numpy()) ** 2
    counts = frame.groupby("site_id").size()
    mean = pd.Series(squared, index=frame["site_id"].to_numpy()).groupby(level=0).mean()
    keep = counts[counts >= MIN_KEYS].index
    return np.sqrt(mean.loc[keep]).sort_index()


def build_rows() -> list[dict[str, Any]]:
    clusters = huc2_cluster_map(
        load_station_registry(ROOT / "data_usgs" / "station_registry_v1.csv")
    )
    rows: list[dict[str, Any]] = []
    for model in V5.MODELS:
        for horizon in V5.HORIZONS:
            keys = common_shift_registry(model, horizon)
            f0 = station_rmse(
                TRUE_SHARDS / f"F0_{model}_h{horizon}_v5_observed.parquet", keys
            )
            true_f3 = station_rmse(
                TRUE_SHARDS / f"F3_full_{model}_h{horizon}_v5_observed.parquet", keys
            )
            shared = f0.index.intersection(true_f3.index)
            arms = {}
            for arm in SH.SHIFT_ARMS:
                arms[arm] = station_rmse(
                    SHIFT_SHARDS / SH.shard_filename(arm, model, horizon), keys
                )
                shared = shared.intersection(arms[arm].index)
            if len(shared) < 40:
                raise ShiftAuthorityError(
                    f"only {len(shared)} stations shared for {model} h{horizon}"
                )
            groups = shared.map(clusters).to_numpy()
            true_value = (f0.loc[shared] - true_f3.loc[shared]).to_numpy()
            true_median = float(np.median(true_value))

            for arm in SH.SHIFT_ARMS:
                value = (f0.loc[shared] - arms[arm].loc[shared]).to_numpy()
                against_true = (arms[arm].loc[shared] - true_f3.loc[shared]).to_numpy()
                boot = cluster_bootstrap_paired_effect(
                    -value, groups, statistic="median",
                    n_boot=N_BOOT, seed=BOOT_SEED,
                )
                contrast = cluster_bootstrap_paired_effect(
                    against_true, groups, statistic="median",
                    n_boot=N_BOOT, seed=BOOT_SEED,
                )
                sens = cluster_inference_sensitivity(
                    against_true, groups, statistic="median"
                )
                median_value = float(np.median(value))
                retained = median_value / true_median if true_median > 0 else np.nan
                rows.append({
                    "arm": arm,
                    "is_primary_timing_control": arm == PRIMARY_ARM,
                    "model": model,
                    "horizon": int(horizon),
                    "true_forcing_value": true_median,
                    "shift_forcing_value": median_value,
                    "shift_forcing_value_ci_low": -boot["ci_high"],
                    "shift_forcing_value_ci_high": -boot["ci_low"],
                    "retained_fraction": retained,
                    "shift_minus_true_median": contrast["effect"],
                    "shift_minus_true_ci_low": contrast["ci_low"],
                    "shift_minus_true_ci_high": contrast["ci_high"],
                    "shift_minus_true_loco_stable": bool(sens["loco_direction_stable"]),
                    "station_win_fraction_shift_worse": float(np.mean(against_true > 0)),
                    "n_stations": len(shared),
                    "n_common_shift_keys": len(keys),
                })
    return rows


def verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary = [
        r for r in rows
        if r["is_primary_timing_control"] and r["horizon"] in (3, 7)
    ]
    worst = max(r["retained_fraction"] for r in primary)
    passed = worst < P4_MAX_RETAINED
    return {
        "rule": "P4_timing_specificity",
        "primary_arm": PRIMARY_ARM,
        "max_retained_fraction_at_3_and_7_days": worst,
        "threshold": P4_MAX_RETAINED,
        "satisfied": passed,
        "meaning": (
            "the model is using event timing, not seasonal level; the "
            "lead-structure interpretation of Section 4.7 stands"
            if passed else
            "the model is using seasonal level rather than event timing; the "
            "lead-structure interpretation of Section 4.7 is withdrawn"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)

    rows = build_rows()
    table = pd.DataFrame(rows)
    decision = verdict(rows)
    summary = {
        "format": FORMAT,
        "authority_status": STATUS,
        "chronology": {
            "specification_sealed_before_outcome": True,
            "seal": "protocols/wrr_forcing_placebo_protocol_v5a_seal.json",
            "post_outcome": False,
        },
        "boundary_rule": (
            "keys with any boundary substitution in either shift arm are dropped "
            "from both shift arms and from the true arm; all three share one "
            "common shift-registry, never mixed with the primary registry"
        ),
        "weaker_control": "F3_shift_minus7 overlaps issue-time information",
        "decision": decision,
        "rows": rows,
    }
    if args.print_only:
        view = table[["arm", "model", "horizon", "true_forcing_value",
                      "shift_forcing_value", "retained_fraction",
                      "station_win_fraction_shift_worse"]]
        print(view.to_string(index=False))
        print("\nP4:", json.dumps(decision, indent=1))
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame_path = args.out_dir / "forcing_shift_rows.parquet"
    table.to_parquet(frame_path, index=False)
    summary_path = args.out_dir / "forcing_shift_summary.json"
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=1, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "forcing_shift_authority_manifest.json").write_text(
        json.dumps({
            "format": FORMAT + ".manifest",
            "builder_sha256": _sha256_file(Path(__file__).resolve()),
            "shift_lineage_manifest_sha256": _sha256_file(
                SH.OUTPUT_DIR / SH.MANIFEST_FILENAME
            ),
            "outputs_sha256": {
                frame_path.name: _sha256_file(frame_path),
                summary_path.name: _sha256_file(summary_path),
            },
        }, sort_keys=True, indent=1) + "\n", encoding="utf-8",
    )
    print(json.dumps({"status": "WRITTEN", "decision": decision},
                     sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

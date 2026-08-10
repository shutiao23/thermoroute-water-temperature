#!/usr/bin/env python3
"""Build the forcing-v5 *inference* authority from the published point authority.

The v5 point authority (``forcing_regime_v5_observed_point_authority_v1``)
publishes station-first point estimates only and explicitly reserves this path
for interval, sign-flip, and stability evidence.  This builder fills that
reservation.

What it computes, per ``model x horizon`` cell, from the 116 station-level
paired effects:

* the station-median paired effect and its interquartile range,
* a 10,000-draw whole-HUC2 cluster bootstrap percentile interval,
* an *exact* whole-HUC2 sign-flip tail (15 clusters, all 2^15 sign vectors),
* the equal-HUC aggregate beside the station-weighted one,
* the leave-one-HUC2-out range,
* a minimum detectable effect obtained by shifting the effect vector until the
  sign-flip tail crosses 0.05 under the observed dependence structure,
* year (2021/2022/2023) and season (DJF/MAM/JJA/SON) strata recomputed from the
  key-level shards rather than reweighted from the pooled numbers,
* station-level paired lead interactions (3d-1d, 7d-3d) and the
  LightGBM-vs-ResidualLightGBM model contrast, both as double differences
  formed per station before any median is taken.

Every subgroup that is empty or falls below its key threshold is written to an
explicit ledger, so a stratum can never disappear silently into a NaN.

Chronology.  The 2021-2023 window was open and F/L outcomes were inspected
before this analysis.  Everything here is therefore a *post-outcome*,
descriptive, fixed-cohort sensitivity.  It is not confirmatory, not
prospectively registered, and does not license an operational-forecast claim:
F3 is a retrospective realized gridded meteorological oracle.
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

from thermoroute.significance import (
    cluster_bootstrap_paired_effect,
    cluster_inference_sensitivity,
    cluster_sign_flip_pvalue,
)
from thermoroute.spatial import huc2_cluster_map, load_station_registry

AUTHORITY_FORMAT = "thermoroute.forcing-v5-observed-inference-authority.v1"
AUTHORITY_STATUS = "POST_OUTCOME_DESCRIPTIVE_INFERENCE_NOT_CONFIRMATORY"
ESTIMAND = "median_i[RMSE_i(F0)-RMSE_i(F3_full)]"

FINAL = ROOT / "outputs" / "final"
POINT_DIR = FINAL / "forcing_regime_v5_observed_point_authority_v1"
SHARD_DIR = FINAL / "forcing_regime_v5_observed" / "forcing_shards_v5_observed"
REGISTRY_CSV = ROOT / "data_usgs" / "station_registry_v1.csv"
DEFAULT_OUT_DIR = FINAL / "forcing_regime_v5_observed_inference_authority_v1"

POINT_EFFECTS = POINT_DIR / "forcing_v5_point_paired_effects.parquet"
POINT_STATIONS = POINT_DIR / "forcing_v5_point_station_metrics.parquet"
POINT_SUMMARY = POINT_DIR / "forcing_v5_point_summary.json"

N_BOOT = 10_000
BOOT_SEED = 0
#: Primary cells keep the protocol's 100-paired-key reportability rule.  Strata
#: are necessarily smaller; they use their own, lower, explicitly named
#: threshold and are never mixed with a primary number.
PRIMARY_MIN_KEYS = 100
STRATUM_MIN_KEYS = 30
#: A stratum is only given an interval when it retains enough spatial units for
#: whole-cluster resampling to mean anything.
STRATUM_MIN_STATIONS = 20
STRATUM_MIN_CLUSTERS = 5

SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


class InferenceError(RuntimeError):
    """An input, pairing, or coverage check failed closed."""


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_cluster_map() -> dict[str, str]:
    """Return the frozen 15-cluster HUC2 map for the cohort."""
    registry = load_station_registry(REGISTRY_CSV)
    return huc2_cluster_map(registry)


def load_point_effects() -> pd.DataFrame:
    if not POINT_EFFECTS.exists():
        raise InferenceError(f"point authority missing: {POINT_EFFECTS}")
    frame = pd.read_parquet(POINT_EFFECTS)
    required = {
        "model", "horizon", "site_id", "n_common_keys", "reportable",
        "rmse_F0", "rmse_F3_full", "delta_rmse_F3_minus_F0",
    }
    missing = required - set(frame.columns)
    if missing:
        raise InferenceError(f"point effects missing columns: {sorted(missing)}")
    if not frame["reportable"].all():
        raise InferenceError("point authority contains non-reportable rows")
    if (frame["n_common_keys"] < PRIMARY_MIN_KEYS).any():
        raise InferenceError("point authority row below the 100-key primary rule")
    return frame


def load_shard_pairs() -> dict[tuple[str, int], pd.DataFrame]:
    """Return F0/F3 key-level pairs per ``(model, horizon)``.

    Pairing is an inner join on ``key_id``; both arms must agree on ``y_true``
    to the frozen 2e-6 degC representation tolerance before any stratum is cut.
    """
    pairs: dict[tuple[str, int], pd.DataFrame] = {}
    for model in ("LightGBM", "ResidualLightGBM"):
        for horizon in (1, 3, 7):
            f0 = SHARD_DIR / f"F0_{model}_h{horizon}_v5_observed.parquet"
            f3 = SHARD_DIR / f"F3_full_{model}_h{horizon}_v5_observed.parquet"
            for path in (f0, f3):
                if not path.exists():
                    raise InferenceError(f"v5 shard missing: {path}")
            cols = ["key_id", "site_id", "target_date", "y_true", "y_pred"]
            left = pd.read_parquet(f0, columns=cols)
            right = pd.read_parquet(f3, columns=cols)
            merged = left.merge(
                right, on=["key_id", "site_id", "target_date"],
                suffixes=("_F0", "_F3"), how="inner", validate="one_to_one",
            )
            drift = float(np.max(np.abs(merged["y_true_F0"] - merged["y_true_F3"])))
            if drift > 2e-6:
                raise InferenceError(
                    f"y_true disagreement {drift} exceeds 2e-6 for {model} h{horizon}"
                )
            merged["y_true"] = merged["y_true_F0"]
            merged["year"] = merged["target_date"].dt.year.astype(int)
            merged["season"] = merged["target_date"].dt.month.map(SEASONS)
            pairs[(model, horizon)] = merged
    return pairs


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------


def _rmse(residual: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(residual))))


def station_effects_from_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Return per-station RMSE(F0), RMSE(F3) and the paired delta on shared keys."""
    rows = []
    for site, block in frame.groupby("site_id", sort=True):
        rmse_f0 = _rmse(block["y_pred_F0"].to_numpy() - block["y_true"].to_numpy())
        rmse_f3 = _rmse(block["y_pred_F3"].to_numpy() - block["y_true"].to_numpy())
        rows.append({
            "site_id": site,
            "n_keys": len(block),
            "rmse_F0": rmse_f0,
            "rmse_F3_full": rmse_f3,
            "delta_rmse_F3_minus_F0": rmse_f3 - rmse_f0,
            "forcing_value_rmse_F0_minus_F3": rmse_f0 - rmse_f3,
        })
    return pd.DataFrame(rows)


def equal_cluster_effect(deltas: np.ndarray, clusters: np.ndarray) -> float:
    """Mean over clusters of the within-cluster median station effect."""
    per_cluster = [
        float(np.median(deltas[clusters == cluster]))
        for cluster in np.unique(clusters)
    ]
    return float(np.mean(per_cluster))


def minimum_detectable_effect(
    deltas: np.ndarray, clusters: np.ndarray, *, alpha: float = 0.05,
) -> dict[str, Any]:
    """Smallest |shift| whose sign-flip tail still clears ``alpha``.

    The observed effect vector is translated toward zero until the one-sided
    whole-cluster sign-flip tail rises above ``alpha``.  The crossing point is
    the smallest constant effect this cohort's dependence structure can resolve,
    which is a more honest statement of power than a non-significant p-value.
    """
    observed = float(np.median(deltas))
    floor = 1.0 / (2 ** len(np.unique(clusters)))
    if observed >= 0.0:
        return {
            "resolvable": False,
            "reason": "observed median does not favour the candidate",
            "minimum_detectable_effect_degC": None,
            "smallest_attainable_p_value": floor,
        }

    def tail(shift: float) -> float:
        return cluster_sign_flip_pvalue(deltas - shift, clusters, statistic="median")

    # Translating by ``shift`` moves the median to ``observed - shift``.  At
    # shift = 0 the effect is the observed one (small tail); at shift = observed
    # the effect is exactly zero (tail ~ 0.5).  The tail is therefore decreasing
    # in ``shift`` over [observed, 0], and the crossing point is the smallest
    # effect location this dependence structure can still resolve.
    if tail(0.0) > alpha:
        return {
            "resolvable": False,
            "reason": "the observed effect itself does not clear alpha",
            "minimum_detectable_effect_degC": None,
            "smallest_attainable_p_value": floor,
        }
    if tail(observed) <= alpha:
        return {
            "resolvable": False,
            "reason": "a zero-centred effect would still clear alpha; test degenerate",
            "minimum_detectable_effect_degC": None,
            "smallest_attainable_p_value": floor,
        }
    low, high = observed, 0.0  # low: not detectable, high: detectable
    for _ in range(40):
        mid = 0.5 * (low + high)
        if tail(mid) <= alpha:
            high = mid
        else:
            low = mid
    return {
        "resolvable": True,
        "reason": None,
        "minimum_detectable_effect_degC": abs(observed - high),
        "alpha": alpha,
        "smallest_attainable_p_value": floor,
    }


def infer_cell(
    deltas: np.ndarray,
    clusters: np.ndarray,
    *,
    with_mde: bool = True,
) -> dict[str, Any]:
    """Full descriptive inference for one vector of station paired effects.

    ``deltas`` use the repository-wide ``candidate - reference`` convention with
    candidate ``F3_full``; negative favours the oracle.  ``forcing_value_*``
    fields restate the same quantity with the sign flipped so that a larger
    number means more information value.
    """
    boot = cluster_bootstrap_paired_effect(
        deltas, clusters, statistic="median", n_boot=N_BOOT, seed=BOOT_SEED,
    )
    sens = cluster_inference_sensitivity(deltas, clusters, statistic="median")
    equal_huc = equal_cluster_effect(deltas, clusters)
    result: dict[str, Any] = {
        "n_stations": len(deltas),
        "n_clusters": len(np.unique(clusters)),
        "effective_cluster_count": sens["effective_cluster_count_inverse_herfindahl"],
        "largest_cluster_share": sens["largest_cluster_share"],
        "median_delta_rmse_F3_minus_F0": boot["effect"],
        "median_forcing_value_rmse_F0_minus_F3": -boot["effect"],
        "forcing_value_q25": float(np.percentile(-deltas, 25)),
        "forcing_value_q75": float(np.percentile(-deltas, 75)),
        "station_win_fraction_F3_better": float(np.mean(deltas < 0.0)),
        "cluster_bootstrap_ci_low_delta": boot["ci_low"],
        "cluster_bootstrap_ci_high_delta": boot["ci_high"],
        # negate and swap so the forcing-value interval reads low..high
        "cluster_bootstrap_ci_low_forcing_value": -boot["ci_high"],
        "cluster_bootstrap_ci_high_forcing_value": -boot["ci_low"],
        "equal_huc_delta": equal_huc,
        "equal_huc_forcing_value": -equal_huc,
        "loco_delta_min": sens["loco_effect_min"],
        "loco_delta_max": sens["loco_effect_max"],
        "loco_forcing_value_min": -sens["loco_effect_max"],
        "loco_forcing_value_max": -sens["loco_effect_min"],
        "loco_sign_stable": bool(sens["loco_direction_stable"]),
        "loco_max_abs_shift": sens["loco_max_abs_shift_from_full"],
        "sign_flip_p_one_sided": sens["sign_flip_p_one_sided_sensitivity"],
        "sign_flip_is_exact": bool(sens["exact_enumeration"]),
        "cluster_warning_codes": ";".join(sens["warning_codes"]),
        "inference_strength": sens["inference_strength"],
        "inference_role": "POST_OUTCOME_DESCRIPTIVE_SENSITIVITY_NOT_DECISION_EVIDENCE",
    }
    if with_mde:
        result["minimum_detectable_effect"] = minimum_detectable_effect(deltas, clusters)
    return result


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------


def _attach_clusters(
    frame: pd.DataFrame, cluster_map: dict[str, str]
) -> tuple[np.ndarray, np.ndarray]:
    clusters = frame["site_id"].map(cluster_map)
    if clusters.isna().any():
        missing = sorted(frame.loc[clusters.isna(), "site_id"].unique())
        raise InferenceError(f"stations absent from the HUC2 map: {missing}")
    return frame["delta_rmse_F3_minus_F0"].to_numpy(float), clusters.to_numpy()


def build_primary(effects: pd.DataFrame, cluster_map: dict[str, str]) -> list[dict]:
    rows = []
    for (model, horizon), block in effects.groupby(["model", "horizon"], sort=True):
        block = block.sort_values("site_id")
        deltas, clusters = _attach_clusters(block, cluster_map)
        row = {"model": model, "horizon": int(horizon), "stratum": "ALL",
               "stratum_kind": "primary"}
        row.update(infer_cell(deltas, clusters))
        row["median_station_rmse_F0"] = float(block["rmse_F0"].median())
        row["median_station_rmse_F3_full"] = float(block["rmse_F3_full"].median())
        rows.append(row)
    return rows


def build_strata(
    pairs: dict[tuple[str, int], pd.DataFrame], cluster_map: dict[str, str]
) -> tuple[list[dict], list[dict]]:
    """Return stratum inference rows and the excluded-subgroup ledger."""
    rows: list[dict] = []
    ledger: list[dict] = []
    for (model, horizon), frame in sorted(pairs.items()):
        for kind, column, levels in (
            ("year", "year", [2021, 2022, 2023]),
            ("season", "season", ["DJF", "MAM", "JJA", "SON"]),
        ):
            for level in levels:
                subset = frame[frame[column] == level]
                station = station_effects_from_pairs(subset) if len(subset) else pd.DataFrame()
                if len(station):
                    below = station[station["n_keys"] < STRATUM_MIN_KEYS]
                    for record in below.itertuples(index=False):
                        ledger.append({
                            "model": model, "horizon": int(horizon),
                            "stratum_kind": kind, "stratum": str(level),
                            "site_id": record.site_id, "n_keys": int(record.n_keys),
                            "reason": f"below STRATUM_MIN_KEYS={STRATUM_MIN_KEYS}",
                        })
                    station = station[station["n_keys"] >= STRATUM_MIN_KEYS]
                if not len(station):
                    ledger.append({
                        "model": model, "horizon": int(horizon),
                        "stratum_kind": kind, "stratum": str(level),
                        "site_id": None, "n_keys": 0,
                        "reason": "stratum empty after key threshold",
                    })
                    continue
                deltas, clusters = _attach_clusters(station, cluster_map)
                n_clusters = len(np.unique(clusters))
                row = {"model": model, "horizon": int(horizon),
                       "stratum": str(level), "stratum_kind": kind,
                       "median_delta_rmse_F3_minus_F0": float(np.median(deltas)),
                       "median_forcing_value_rmse_F0_minus_F3": -float(np.median(deltas)),
                       "station_win_fraction_F3_better": float(np.mean(deltas < 0.0)),
                       "n_stations": len(deltas), "n_clusters": n_clusters,
                       "median_keys_per_station": float(station["n_keys"].median())}
                if len(deltas) >= STRATUM_MIN_STATIONS and n_clusters >= STRATUM_MIN_CLUSTERS:
                    row.update(infer_cell(deltas, clusters, with_mde=False))
                    row["interval_status"] = "COMPUTED"
                else:
                    row["interval_status"] = (
                        "NOT_COMPUTED_INSUFFICIENT_SPATIAL_UNITS"
                    )
                    ledger.append({
                        "model": model, "horizon": int(horizon),
                        "stratum_kind": kind, "stratum": str(level),
                        "site_id": None, "n_keys": int(station["n_keys"].sum()),
                        "reason": (
                            f"stations={len(deltas)} clusters={n_clusters} below "
                            f"({STRATUM_MIN_STATIONS}, {STRATUM_MIN_CLUSTERS})"
                        ),
                    })
                rows.append(row)
    return rows, ledger


def build_contrasts(effects: pd.DataFrame, cluster_map: dict[str, str]) -> list[dict]:
    """Station-level double differences: lead interactions and model contrast.

    Each contrast is formed *per station* before any median is taken, so no
    quantity here is a difference of two marginal medians.
    """
    rows: list[dict] = []
    value = "forcing_value_rmse_F0_minus_F3"

    for model, block in effects.groupby("model", sort=True):
        wide = block.pivot(index="site_id", columns="horizon", values=value).dropna()
        for high, low, label in ((3, 1, "V_F(3d) - V_F(1d)"), (7, 3, "V_F(7d) - V_F(3d)")):
            diff = (wide[high] - wide[low]).sort_index()
            frame = pd.DataFrame({
                "site_id": diff.index,
                # keep the candidate-reference sign convention: more negative
                # delta means the longer lead gains more forcing value
                "delta_rmse_F3_minus_F0": -diff.to_numpy(),
            })
            deltas, clusters = _attach_clusters(frame, cluster_map)
            row = {"contrast_kind": "lead_interaction", "model": model,
                   "label": label, "horizon_high": high, "horizon_low": low}
            row.update(infer_cell(deltas, clusters, with_mde=False))
            rows.append(row)

    for horizon, block in effects.groupby("horizon", sort=True):
        wide = block.pivot(index="site_id", columns="model", values=value).dropna()
        if not {"LightGBM", "ResidualLightGBM"} <= set(wide.columns):
            continue
        diff = (wide["ResidualLightGBM"] - wide["LightGBM"]).sort_index()
        frame = pd.DataFrame({
            "site_id": diff.index,
            "delta_rmse_F3_minus_F0": -diff.to_numpy(),
        })
        deltas, clusters = _attach_clusters(frame, cluster_map)
        row = {"contrast_kind": "model_contrast", "model": "ResidualLightGBM-LightGBM",
               "label": "V_F(ResidualLightGBM) - V_F(LightGBM)",
               "horizon_high": int(horizon), "horizon_low": int(horizon)}
        row.update(infer_cell(deltas, clusters, with_mde=False))
        rows.append(row)
    return rows


def build_all() -> dict[str, Any]:
    cluster_map = load_cluster_map()
    effects = load_point_effects()
    pairs = load_shard_pairs()

    # The strata must reproduce the published point estimate when pooled; this
    # is the check that the shard-side recomputation and the point authority
    # describe the same experiment.
    for (model, horizon), frame in pairs.items():
        rebuilt = station_effects_from_pairs(frame).set_index("site_id")
        published = effects[
            (effects["model"] == model) & (effects["horizon"] == horizon)
        ].set_index("site_id")
        shared = rebuilt.index.intersection(published.index)
        if len(shared) != len(published):
            raise InferenceError(
                f"shard/point station mismatch for {model} h{horizon}: "
                f"{len(shared)} vs {len(published)}"
            )
        drift = float(np.max(np.abs(
            rebuilt.loc[shared, "delta_rmse_F3_minus_F0"]
            - published.loc[shared, "delta_rmse_F3_minus_F0"]
        )))
        if drift > 1e-9:
            raise InferenceError(
                f"shard recomputation drifts from the point authority by {drift}"
            )

    primary = build_primary(effects, cluster_map)
    strata, ledger = build_strata(pairs, cluster_map)
    contrasts = build_contrasts(effects, cluster_map)
    return {
        "primary": primary,
        "strata": strata,
        "contrasts": contrasts,
        "excluded_subgroups": ledger,
    }


def summary_payload(parts: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": AUTHORITY_FORMAT,
        "authority_status": AUTHORITY_STATUS,
        "station_first_estimand": ESTIMAND,
        "sign_convention": (
            "delta = RMSE(F3_full) - RMSE(F0); negative favours the oracle. "
            "forcing_value = -delta."
        ),
        "chronology": {
            "post_outcome": True,
            "confirmatory": False,
            "prospectively_registered": False,
            "already_open_window": "2021-2023",
        },
        "interpretation_limits": {
            "f3_is": "retrospective realized gridded meteorological oracle",
            "f3_is_not": [
                "operational forecast gain",
                "deployable improvement",
                "as-issued forecast value",
            ],
            "additive_information_budget": False,
            "cross_experiment_ratio_allowed": False,
        },
        "inference_settings": {
            "bootstrap_draws": N_BOOT,
            "bootstrap_seed": BOOT_SEED,
            "cluster_unit": "HUC2",
            "sign_flip": "exact enumeration of all 2^K whole-cluster sign vectors",
            "primary_min_keys": PRIMARY_MIN_KEYS,
            "stratum_min_keys": STRATUM_MIN_KEYS,
            "stratum_min_stations": STRATUM_MIN_STATIONS,
            "stratum_min_clusters": STRATUM_MIN_CLUSTERS,
            "season_key": "target_date month",
            "year_key": "target_date year",
        },
        "primary": parts["primary"],
        "contrasts": parts["contrasts"],
        "excluded_subgroup_count": len(parts["excluded_subgroups"]),
    }


def write_outputs(parts: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summary_payload(parts)
    files = {
        "forcing_v5_inference_summary.json": summary,
        "forcing_v5_inference_strata.parquet": pd.DataFrame(parts["strata"]),
        "forcing_v5_inference_contrasts.parquet": pd.DataFrame(parts["contrasts"]),
        "forcing_v5_inference_excluded_subgroups.parquet": pd.DataFrame(
            parts["excluded_subgroups"]
            or [{"model": None, "horizon": None, "stratum_kind": None,
                 "stratum": None, "site_id": None, "n_keys": None, "reason": None}]
        ),
    }
    digests: dict[str, str] = {}
    for name, payload in files.items():
        path = out_dir / name
        if isinstance(payload, pd.DataFrame):
            payload.to_parquet(path, index=False)
        else:
            path.write_text(
                json.dumps(payload, sort_keys=True, indent=1, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        digests[name] = _sha256_file(path)

    manifest = {
        "format": AUTHORITY_FORMAT + ".manifest",
        "authority_status": AUTHORITY_STATUS,
        "inputs": {
            "point_paired_effects_sha256": _sha256_file(POINT_EFFECTS),
            "point_station_metrics_sha256": _sha256_file(POINT_STATIONS),
            "point_summary_sha256": _sha256_file(POINT_SUMMARY),
            "station_registry_sha256": _sha256_file(REGISTRY_CSV),
            "shard_sha256": {
                path.name: _sha256_file(path)
                for path in sorted(SHARD_DIR.glob("*.parquet"))
            },
        },
        "builder_sha256": _sha256_file(Path(__file__).resolve()),
        "outputs_sha256": digests,
        "summary_canonical_sha256": _canonical_json_sha256(summary),
    }
    manifest_path = out_dir / "forcing_v5_inference_authority_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=1, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    digests[manifest_path.name] = _sha256_file(manifest_path)
    return digests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--print-only", action="store_true",
        help="compute and print the primary rows without writing any file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    parts = build_all()
    if args.print_only:
        print(json.dumps(summary_payload(parts), sort_keys=True, indent=1))
        return 0
    digests = write_outputs(parts, args.out_dir)
    print(json.dumps({
        "status": "WRITTEN",
        "out_dir": str(args.out_dir),
        "files": digests,
        "primary_rows": len(parts["primary"]),
        "stratum_rows": len(parts["strata"]),
        "contrast_rows": len(parts["contrasts"]),
        "excluded_subgroups": len(parts["excluded_subgroups"]),
    }, sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

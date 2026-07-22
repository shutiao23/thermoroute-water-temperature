#!/usr/bin/env python3
"""Route-A station-level inference on a strict common forecast registry.

The 2019--2020 results are exploratory.  This script mirrors the pre-registered
confirmatory estimands without retroactively calling this period blind.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thermoroute import config as C
from thermoroute import results as R
from thermoroute.probability import ensemble_prediction_frame
from thermoroute.registry import enforce_common_forecast_keys
from thermoroute.repro import atomic_write_bytes
from thermoroute.spatial import huc2_cluster_map, load_station_registry
from thermoroute.significance import (
    cluster_bootstrap_paired_effect,
    cluster_inference_sensitivity,
    cluster_sign_flip_pvalue,
    equivalence_decision,
    holm_adjust,
    noninferiority_decision,
)


PREDICTIONS = C.PREDICTIONS / "usgs_predictions_v2.parquet"
PANEL = ROOT / "data_usgs" / "panel_usgs_120v2.parquet"
REGISTRY = ROOT / "data_usgs" / "station_registry_v1.csv"
MODELS = ("ThermoRoute", "DampedPersistence", "LightGBM")
NONINFERIORITY_MARGIN_C = 0.05


def _huc_clusters() -> dict[str, str]:
    return huc2_cluster_map(load_station_registry(REGISTRY))


def _common_predictions() -> pd.DataFrame:
    raw = R.load_route_a_predictions(
        PREDICTIONS, root=ROOT, panel_path=PANEL, registry_path=REGISTRY
    )
    frames = []
    for model in MODELS:
        frame = ensemble_prediction_frame(raw, model)
        frame["model"] = model
        frame["scope"] = "route_a_common"
        frame["feature_set"] = "canonical"
        frame["seed"] = 0
        # Registry utility only needs identity/model/split/y fields and keeps extras.
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    common, audit = enforce_common_forecast_keys(combined, MODELS, split="test")
    print(
        f"common registry={audit.common_unique}; dropped={audit.dropped_rows}; "
        f"before={audit.before_unique}", flush=True
    )
    return common[(common.split == "test") & common.model.isin(MODELS)]


def _station_rmse(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, horizon, site), group in frame.groupby(["model", "horizon", "site_id"]):
        rows.append({
            "model": model,
            "horizon": int(horizon),
            "site_id": str(site),
            "RMSE": float(np.sqrt(np.mean((group.y_pred - group.y_true) ** 2))),
            "n": len(group),
        })
    return pd.DataFrame(rows)


def main() -> None:
    common = _common_predictions()
    per_station = _station_rmse(common)
    clusters = _huc_clusters()
    unmapped_count = sum(value.startswith("UNMAPPED:") for value in clusters.values())
    records = []
    sensitivity_records = []
    loco_records = []
    confirmatory_indices = []
    for horizon in C.HORIZONS:
        wide = per_station[per_station.horizon == horizon].pivot(
            index="site_id", columns="model", values="RMSE"
        ).dropna(subset=list(MODELS))
        for reference in ("DampedPersistence", "LightGBM"):
            effect = (wide.ThermoRoute - wide[reference]).to_numpy(float)
            cluster = np.array([clusters.get(str(site), f"UNMAPPED:{site}") for site in wide.index])
            margin = NONINFERIORITY_MARGIN_C if (
                reference == "LightGBM" and horizon in (3, 7)
            ) else 0.0
            inference = cluster_bootstrap_paired_effect(
                effect,
                cluster,
                n_boot=10000,
                seed=1000 + horizon + (100 if reference == "LightGBM" else 0),
                null_margin=margin,
            )
            record = {
                "horizon": horizon,
                "reference": reference,
                "margin_c": margin,
                **inference,
                "win_rate": float(np.mean(effect < 0)),
                "equivalent_margin_0p05": equivalence_decision(
                    inference["ci_low"], inference["ci_high"], NONINFERIORITY_MARGIN_C
                ),
                "noninferior": (
                    noninferiority_decision(inference["ci_high"], margin)
                    if margin > 0 else False
                ),
            }
            record["p_cluster_sign_flip"] = cluster_sign_flip_pvalue(
                effect,
                cluster,
                null_margin=margin,
                n_randomisations=50000,
                seed=5000 + horizon + (100 if reference == "LightGBM" else 0),
            )
            sensitivity = cluster_inference_sensitivity(
                effect,
                cluster,
                null_margin=margin,
                n_randomisations=50000,
                seed=5000 + horizon + (100 if reference == "LightGBM" else 0),
            )
            sensitivity_p = sensitivity["sign_flip_p_one_sided_sensitivity"]
            if sensitivity_p is None or float(sensitivity_p) != float(
                record["p_cluster_sign_flip"]
            ):
                raise RuntimeError("cluster sensitivity disagrees with frozen p-value path")
            sensitivity_records.append({
                "horizon": horizon,
                "reference": reference,
                "margin_c": margin,
                **{
                    key: value
                    for key, value in sensitivity.items()
                    if key not in {"leave_one_cluster_out", "warning_codes"}
                },
                "warning_codes": ";".join(sensitivity["warning_codes"]),
            })
            for loco in sensitivity["leave_one_cluster_out"]:
                loco_records.append({
                    "horizon": horizon,
                    "reference": reference,
                    "margin_c": margin,
                    **loco,
                })
            records.append(record)
            if reference == "DampedPersistence" or (
                reference == "LightGBM" and horizon in (3, 7)
            ):
                confirmatory_indices.append(len(records) - 1)

    result = pd.DataFrame(records)
    p_values = np.array([
        result.loc[index, "p_cluster_sign_flip"] for index in confirmatory_indices
    ])
    adjusted = holm_adjust(p_values)
    result["p_holm_confirmatory_family"] = np.nan
    for index, value in zip(confirmatory_indices, adjusted):
        result.loc[index, "p_holm_confirmatory_family"] = value
    atomic_write_bytes(
        C.TABLES / "claim1_significance.csv", result.to_csv(index=False).encode()
    )
    sensitivity_result = pd.DataFrame(sensitivity_records)
    atomic_write_bytes(
        C.TABLES / "claim1_cluster_sensitivity.csv",
        sensitivity_result.to_csv(index=False).encode(),
    )
    atomic_write_bytes(
        C.TABLES / "claim1_cluster_loco.csv",
        pd.DataFrame(loco_records).to_csv(index=False).encode(),
    )

    lines = [
        "# Route-A paired station-level inference\n",
        "The 2019--2020 partition is previously inspected development evidence. "
        "Effects are station RMSE(ThermoRoute) minus RMSE(reference), so negative is "
        f"better. Confidence intervals resample complete HUC2 clusters; {unmapped_count} "
        "stations without verified HUC metadata are treated as separate clusters.\n",
        "The predeclared non-inferiority margin is +0.05 degrees C at h=3/7. "
        "A non-significant difference is not called parity; equivalence requires the "
        "entire interval inside [-0.05,+0.05]. One-sided p-values use a whole-HUC2 "
        "cluster sign-flip randomisation (cluster-level sign symmetry assumption). "
        "Holm adjustment covers the three damped-superiority and two LightGBM-"
        "non-inferiority tests.\n",
        "| h | reference | n | clusters | median difference [95% CI] | win rate | "
        "margin | Holm p | equivalent | non-inferior |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in result.iterrows():
        p = row.p_holm_confirmatory_family
        p_text = "exploratory" if pd.isna(p) else f"{p:.4g}"
        lines.append(
            f"| {int(row.horizon)} | {row.reference} | {int(row.n_stations)} | "
            f"{int(row.n_clusters)} | {row.effect:+.3f} "
            f"[{row.ci_low:+.3f},{row.ci_high:+.3f}] | {row.win_rate:.2f} | "
            f"{row.margin_c:.2f} | {p_text} | {bool(row.equivalent_margin_0p05)} | "
            f"{bool(row.noninferior)} |"
        )
    lines.extend([
        "",
        "## Exploratory small-cluster sensitivity (not a new test)",
        "",
        "These diagnostics do not change the frozen five-test family, its p-values, "
        "Holm adjustment, confidence intervals, or decisions. Exact enumeration "
        "removes Monte-Carlo error, but it does not remove the whole-HUC sign-symmetry "
        "assumption or make 15 clusters a large sample. `NO_STRONG_INFERENCE` is "
        "therefore reported whenever fewer than 30 clusters remain, one cluster has "
        "at least 25% of stations, effective cluster count is below 75% of nominal "
        "count, or leave-one-HUC effects cross the tested margin.",
        "",
        "| h | reference | G | cluster sizes min/median/max | effective G | largest "
        "share | LOCO effect-minus-margin range | direction | strength | warnings |",
        "|---|---|---:|---|---:|---:|---|---|---|---|",
    ])
    for _, row in sensitivity_result.iterrows():
        lines.append(
            f"| {int(row.horizon)} | {row.reference} | {int(row.n_clusters)} | "
            f"{int(row.cluster_size_min)}/{row.cluster_size_median:.1f}/"
            f"{int(row.cluster_size_max)} | "
            f"{row.effective_cluster_count_inverse_herfindahl:.2f} | "
            f"{row.largest_cluster_share:.3f} | "
            f"[{row.loco_effect_minus_null_margin_min:+.3f},"
            f"{row.loco_effect_minus_null_margin_max:+.3f}] | "
            f"{row.loco_direction} | {row.inference_strength} | "
            f"{row.warning_codes or 'none'} |"
        )
    atomic_write_bytes(C.TABLES / "claim1_significance.md", "\n".join(lines).encode())

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for axis, horizon in zip(axes, C.HORIZONS):
        wide = per_station[per_station.horizon == horizon].pivot(
            index="site_id", columns="model", values="RMSE"
        ).dropna(subset=["ThermoRoute", "DampedPersistence"])
        axis.scatter(wide.DampedPersistence, wide.ThermoRoute, s=14, alpha=0.65)
        limit = max(wide.DampedPersistence.max(), wide.ThermoRoute.max()) * 1.05
        axis.plot([0, limit], [0, limit], "k--", linewidth=1)
        axis.set_xlim(0, limit)
        axis.set_ylim(0, limit)
        axis.set_title(f"h={horizon} d")
        axis.set_xlabel("damped-persistence RMSE")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("ThermoRoute RMSE")
    fig.suptitle("Development-period station RMSE on common forecast keys")
    fig.savefig(C.FIGURES / "fig_usgs_perstation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("\n".join(lines))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Route-A station-level inference on a strict common forecast registry.

The 2019--2020 results are exploratory. This script mirrors the later frozen
estimand registry without retroactively calling this period blind or formal.
"""
# ruff: noqa: E402
from __future__ import annotations

from io import BytesIO
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
from thermoroute.model_suite import (
    STAGE16_COMPLETION_RECEIPT_PATH,
    validate_stage16_completion_receipt,
)
from thermoroute.probability import ensemble_prediction_frame
from thermoroute.registry import ROUTE_A_PRIMARY_MODELS, enforce_common_forecast_keys
from thermoroute.repro import (
    advisory_file_lock,
    atomic_write_bytes,
    sha256_file,
    source_tree_hash,
)
from thermoroute.spatial import huc2_cluster_map, load_station_registry
from thermoroute.significance import (
    cluster_bootstrap_paired_effect,
    cluster_inference_sensitivity,
    cluster_sign_flip_pvalue,
    equivalence_decision,
    holm_adjust,
    noninferiority_decision as numerical_margin_decision,
)


PREDICTIONS = C.PREDICTIONS / "usgs_predictions_v2.parquet"
STAGE16_RECEIPT = ROOT / STAGE16_COMPLETION_RECEIPT_PATH
PANEL = ROOT / "data_usgs" / "panel_usgs_120v2.parquet"
REGISTRY = ROOT / "data_usgs" / "station_registry_v1.csv"
MODELS = ("ThermoRoute", "DampedPersistence", "LightGBM")
NUMERICAL_MARGIN_C = 0.05
MINIMUM_SITE_HORIZON_TARGETS = 100
DEVELOPMENT_MIRROR_FORMAT = "thermoroute.development-route-a-estimand-mirror.v1"
DEVELOPMENT_EVIDENCE_START = "2019-01-01"
DEVELOPMENT_EVIDENCE_END = "2020-12-31"
DEVELOPMENT_EVIDENCE_ROLE = "DEVELOPMENT_EXPLORATORY_PREVIOUSLY_INSPECTED"
FROZEN_FAMILY_TEST_IDS = {
    (1, "DampedPersistence"): "H1-h1-vs-damped",
    (3, "DampedPersistence"): "H1-h3-vs-damped",
    (7, "DampedPersistence"): "H1-h7-vs-damped",
    (3, "LightGBM"): "H2-h3-vs-lightgbm",
    (7, "LightGBM"): "H2-h7-vs-lightgbm",
}
DEVELOPMENT_IDENTITY_COLUMNS = (
    "analysis_format",
    "evidence_interval_start",
    "evidence_interval_end",
    "evidence_role",
    "previously_inspected",
    "formal_confirmatory_result",
    "decision_eligible",
    "mirror_test_id",
    "comparison_role",
    "in_frozen_five_test_family",
    "p_value_role",
    "equivalence_role",
    "minimum_common_targets",
    "canonical_registry_models",
    "summarized_models",
    "source_prediction_sha256",
    "stage16_completion_receipt_sha256",
    "source_tree_sha256",
)


def _development_row_identity(
    *,
    horizon: int,
    reference: str,
    source_prediction_sha256: str,
    stage16_completion_receipt_sha256: str,
    source_tree_sha256: str,
) -> dict[str, object]:
    """Return the non-negotiable exploratory identity for one output row."""
    if horizon not in C.HORIZONS or reference not in {
        "DampedPersistence", "LightGBM"
    }:
        raise ValueError("development mirror comparison is outside the fixed registry")
    for label, digest in (
        ("source prediction", source_prediction_sha256),
        ("Stage-16 completion receipt", stage16_completion_receipt_sha256),
        ("source tree", source_tree_sha256),
    ):
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"{label} SHA-256 is malformed")
    mirror_test_id = FROZEN_FAMILY_TEST_IDS.get((horizon, reference))
    in_family = mirror_test_id is not None
    return {
        "analysis_format": DEVELOPMENT_MIRROR_FORMAT,
        "evidence_interval_start": DEVELOPMENT_EVIDENCE_START,
        "evidence_interval_end": DEVELOPMENT_EVIDENCE_END,
        "evidence_role": DEVELOPMENT_EVIDENCE_ROLE,
        "previously_inspected": True,
        "formal_confirmatory_result": False,
        "decision_eligible": False,
        "mirror_test_id": mirror_test_id or "",
        "comparison_role": (
            "FROZEN_FIVE_TEST_FAMILY_DEVELOPMENT_MIRROR"
            if in_family
            else "EXPLORATORY_H1_LIGHTGBM_OUTSIDE_FROZEN_FAMILY"
        ),
        "in_frozen_five_test_family": in_family,
        "p_value_role": "ASSUMPTION_CONDITIONAL_DEVELOPMENT_SENSITIVITY",
        "equivalence_role": "EXPLORATORY_DIAGNOSTIC_NO_CLAIM",
        "minimum_common_targets": MINIMUM_SITE_HORIZON_TARGETS,
        "canonical_registry_models": "|".join(ROUTE_A_PRIMARY_MODELS),
        "summarized_models": "|".join(MODELS),
        "source_prediction_sha256": source_prediction_sha256,
        "stage16_completion_receipt_sha256": (
            stage16_completion_receipt_sha256
        ),
        "source_tree_sha256": source_tree_sha256,
    }


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


def _retain_reportable_station_rmse(
    per_station: pd.DataFrame,
    *,
    models: tuple[str, ...] = MODELS,
    minimum_targets: int = MINIMUM_SITE_HORIZON_TARGETS,
) -> pd.DataFrame:
    """Keep only sufficiently populated exact-common station/horizon cells.

    ``_common_predictions`` has already intersected the exact forecast keys
    across ``models``.  This downstream guard prevents a malformed or partially
    regenerated station table from silently changing that population: every
    station/horizon cell must contain one row per model and all model-specific
    target counts must be identical before the minimum-count rule is applied.
    """
    if (
        isinstance(minimum_targets, bool)
        or not isinstance(minimum_targets, int)
        or minimum_targets <= 0
    ):
        raise ValueError("minimum_targets must be a positive integer")
    requested = tuple(dict.fromkeys(str(model) for model in models))
    if len(requested) < 2 or len(requested) != len(models):
        raise ValueError("models must contain at least two distinct model names")
    required = {"model", "horizon", "site_id", "RMSE", "n"}
    missing = sorted(required - set(per_station.columns))
    if missing:
        raise ValueError(f"station RMSE table is missing columns: {missing}")

    selected = per_station[per_station.model.astype(str).isin(requested)].copy()
    if selected.empty:
        raise RuntimeError("station RMSE table contains none of the requested models")
    selected["model"] = selected.model.astype(str)
    selected["site_id"] = selected.site_id.astype(str)
    if selected.duplicated(["model", "horizon", "site_id"]).any():
        raise RuntimeError("station RMSE table duplicates a model/station/horizon cell")

    counts = selected.pivot(
        index=["site_id", "horizon"], columns="model", values="n"
    ).reindex(columns=list(requested))
    if counts.isna().any().any():
        raise RuntimeError(
            "station/horizon model coverage is incomplete on the exact common registry"
        )
    count_values = counts.to_numpy(float)
    if not (
        np.isfinite(count_values).all()
        and (count_values > 0).all()
        and np.equal(count_values, np.floor(count_values)).all()
    ):
        raise RuntimeError("station/horizon target counts must be finite positive integers")
    if not np.equal(count_values, count_values[:, [0]]).all():
        raise RuntimeError(
            "station/horizon target counts disagree across models on the exact common registry"
        )

    eligible = {
        (str(site), int(horizon))
        for site, horizon in counts.index[count_values[:, 0] >= minimum_targets]
    }
    retained = selected[
        [
            (str(site), int(horizon)) in eligible
            for site, horizon in zip(selected.site_id, selected.horizon)
        ]
    ].copy()
    return retained.sort_values(["horizon", "site_id", "model"]).reset_index(drop=True)


def _run(*, stage16_completion_receipt_sha256: str) -> None:
    common = _common_predictions()
    per_station = _retain_reportable_station_rmse(_station_rmse(common))
    prediction_sha256 = sha256_file(PREDICTIONS)
    code_sha256 = source_tree_hash(ROOT)
    clusters = _huc_clusters()
    unmapped_count = sum(value.startswith("UNMAPPED:") for value in clusters.values())
    records = []
    sensitivity_records = []
    loco_records = []
    family_mirror_indices = []
    for horizon in C.HORIZONS:
        wide = per_station[per_station.horizon == horizon].pivot(
            index="site_id", columns="model", values="RMSE"
        ).dropna(subset=list(MODELS))
        if wide.empty:
            raise RuntimeError(
                f"no station has at least {MINIMUM_SITE_HORIZON_TARGETS} "
                f"exact common targets at h={horizon}"
            )
        for reference in ("DampedPersistence", "LightGBM"):
            identity = _development_row_identity(
                horizon=horizon,
                reference=reference,
                source_prediction_sha256=prediction_sha256,
                stage16_completion_receipt_sha256=(
                    stage16_completion_receipt_sha256
                ),
                source_tree_sha256=code_sha256,
            )
            effect = (wide.ThermoRoute - wide[reference]).to_numpy(float)
            cluster = np.array([clusters.get(str(site), f"UNMAPPED:{site}") for site in wide.index])
            margin = NUMERICAL_MARGIN_C if (
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
                **identity,
                "horizon": horizon,
                "reference": reference,
                "margin_c": margin,
                **inference,
                "win_rate": float(np.mean(effect < 0)),
                "development_equivalence_diagnostic_0p05": equivalence_decision(
                    inference["ci_low"], inference["ci_high"], NUMERICAL_MARGIN_C
                ),
                "development_ci_below_numerical_margin_exploratory": (
                    numerical_margin_decision(inference["ci_high"], margin)
                    if margin > 0 else False
                ),
            }
            record["p_cluster_sign_flip_development_sensitivity"] = (
                cluster_sign_flip_pvalue(
                    effect,
                    cluster,
                    null_margin=margin,
                    n_randomisations=50000,
                    seed=5000 + horizon + (100 if reference == "LightGBM" else 0),
                )
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
                record["p_cluster_sign_flip_development_sensitivity"]
            ):
                raise RuntimeError("cluster sensitivity disagrees with frozen p-value path")
            sensitivity_records.append({
                **identity,
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
                    **identity,
                    "horizon": horizon,
                    "reference": reference,
                    "margin_c": margin,
                    **loco,
                })
            records.append(record)
            if bool(identity["in_frozen_five_test_family"]):
                family_mirror_indices.append(len(records) - 1)

    result = pd.DataFrame(records)
    if len(family_mirror_indices) != 5 or set(
        result.loc[family_mirror_indices, "mirror_test_id"]
    ) != set(FROZEN_FAMILY_TEST_IDS.values()):
        raise RuntimeError("development mirror does not cover the frozen five-test family")
    p_values = np.array([
        result.loc[index, "p_cluster_sign_flip_development_sensitivity"]
        for index in family_mirror_indices
    ])
    adjusted = holm_adjust(p_values)
    result["p_holm_frozen_family_development_mirror"] = np.nan
    for index, value in zip(family_mirror_indices, adjusted):
        result.loc[index, "p_holm_frozen_family_development_mirror"] = value
    atomic_write_bytes(
        C.TABLES / "development_route_a_estimand_mirror.csv",
        result.to_csv(index=False).encode(),
    )
    sensitivity_result = pd.DataFrame(sensitivity_records)
    atomic_write_bytes(
        C.TABLES / "development_route_a_estimand_mirror_cluster_sensitivity.csv",
        sensitivity_result.to_csv(index=False).encode(),
    )
    atomic_write_bytes(
        C.TABLES / "development_route_a_estimand_mirror_cluster_loco.csv",
        pd.DataFrame(loco_records).to_csv(index=False).encode(),
    )

    lines = [
        "# Exploratory development-period mirror of frozen Route-A estimands\n",
        "The 2019--2020 partition is previously inspected development evidence, "
        "not a formal confirmatory result, and no row is decision-eligible. "
        "Effects are station RMSE(ThermoRoute) minus RMSE(reference), so negative is "
        f"better. Confidence intervals resample complete HUC2 clusters; {unmapped_count} "
        "stations without verified HUC metadata are treated as separate clusters.\n",
        "The frozen numerical margin is +0.05 degrees C at h=3/7. "
        "A non-significant difference is not called parity; equivalence requires the "
        "entire interval inside [-0.05,+0.05]. One-sided p-values use a whole-HUC2 "
        "cluster sign-flip randomisation (cluster-level sign symmetry assumption). "
        "A development-only mirror of the frozen Holm procedure covers the three "
        "damped-superiority and two LightGBM numerical-bound rows; it cannot support "
        "a confirmatory decision.\n",
        f"Only station-horizon cells with at least {MINIMUM_SITE_HORIZON_TARGETS} "
        "targets from the canonical common-key registry across all six primary "
        "models are retained. This artifact summarizes ThermoRoute, damped "
        "persistence, and LightGBM; their counts are asserted identical before "
        "scoring. The h=1 LightGBM comparison and every equivalence diagnostic are "
        "exploratory and excluded from the frozen five-test family.\n",
        "The exact final Stage-16 completion receipt SHA-256 for every row and "
        f"rendered artifact is `{stage16_completion_receipt_sha256}`.\n",
        "| h | reference | n | clusters | median difference [95% CI] | win rate | "
        "margin | frozen-family Holm mirror (development only) | equivalence "
        "diagnostic (exploratory) | CI below numerical margin (exploratory) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in result.iterrows():
        p = row["p_holm_frozen_family_development_mirror"]
        p_text = "outside family" if pd.isna(p) else f"{p:.4g}"
        lines.append(
            f"| {int(row.horizon)} | {row.reference} | {int(row.n_stations)} | "
            f"{int(row.n_clusters)} | {row.effect:+.3f} "
            f"[{row.ci_low:+.3f},{row.ci_high:+.3f}] | {row.win_rate:.2f} | "
            f"{row.margin_c:.2f} | {p_text} | "
            f"{bool(row.development_equivalence_diagnostic_0p05)} | "
            f"{bool(row.development_ci_below_numerical_margin_exploratory)} |"
        )
    lines.extend([
        "",
        "## Exploratory small-cluster sensitivity (not a new test)",
        "",
        "These development diagnostics do not create or change any formal p-value, "
        "Holm adjustment, confidence interval, or decision. Exact enumeration "
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
    atomic_write_bytes(
        C.TABLES / "development_route_a_estimand_mirror.md",
        "\n".join(lines).encode(),
    )

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
    figure_payload = BytesIO()
    fig.savefig(
        figure_payload,
        format="png",
        dpi=300,
        bbox_inches="tight",
        metadata={
            "Stage16CompletionReceiptSHA256": stage16_completion_receipt_sha256,
        },
    )
    plt.close(fig)
    atomic_write_bytes(
        C.FIGURES / "development_route_a_estimand_mirror.png",
        figure_payload.getvalue(),
    )
    print("\n".join(lines))


def main() -> None:
    # Stage 16 takes this path exclusively while replacing the V2 prediction,
    # bundle, pointers, and receipt.  Keep the shared lock from gate validation
    # through every table/report/figure publication to prevent mixed generations.
    with advisory_file_lock(C.STAGE16_TRANSACTION_LOCK, exclusive=False):
        validate_stage16_completion_receipt(
            STAGE16_RECEIPT,
            root=ROOT,
            replay_bundle=False,
        )
        _run(
            stage16_completion_receipt_sha256=sha256_file(STAGE16_RECEIPT),
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Route-A probabilistic and multi-metric verification.

Event probabilities are calibrated on the calibration split.  Brier skill uses a
seasonal climatology fitted from training observations, never the evaluation event
rate.  The 2019--2020 partition is called the development evaluation because it
has previously informed this project; the post-2020 confirmation set is scored by
the separate sealed confirmation runner.
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
from thermoroute import conformal as CF
from thermoroute import data as D
from thermoroute import metrics as M
from thermoroute import results as R
from thermoroute.evidence import FrozenPanelSpec
from thermoroute.probability import (
    calibrated_event_frame,
    calibration_intercept_slope,
    ensemble_prediction_frame,
    expected_calibration_error,
    fit_seasonal_climatology,
)
from thermoroute.repro import atomic_write_bytes, atomic_write_json
from thermoroute.spatial import huc2_cluster_map, load_station_registry


PANEL = ROOT / "data_usgs" / "panel_usgs_120v2.parquet"
STATION_REGISTRY = ROOT / "data_usgs" / "station_registry_v1.csv"
PREDICTIONS = C.PREDICTIONS / "usgs_predictions_v2.parquet"
PROB_MODELS = ("ThermoRoute", "LightGBM", "LSTM")
POINT_MODELS = ("Persistence", "DampedPersistence", "LightGBM", "LSTM", "ThermoRoute")
COLORS = {"ThermoRoute": "#B3132B", "LightGBM": "#185FA5", "LSTM": "#6A4C93"}
IDENTITY = ["site_id", "horizon", "issue_date", "target_date"]


def _training_references() -> tuple[dict[str, float], object]:
    panel = FrozenPanelSpec.load().load_panel(stable_site_ids=True)
    panel["DATE"] = pd.to_datetime(panel["DATE"])
    masks = D.split_masks(panel["DATE"])
    train = panel.loc[masks.train].copy()
    thresholds = {
        str(site): float(group.WTEMP.quantile(C.EXCEEDANCE_QUANTILE))
        for site, group in train.groupby("site_id")
    }
    return thresholds, fit_seasonal_climatology(train, thresholds)


def _cqr_evaluation(ensemble: pd.DataFrame) -> pd.DataFrame:
    calibration = ensemble[ensemble.split == "calib"]
    evaluation = ensemble[ensemble.split == "test"].copy()
    if calibration.empty or not {"q05", "q95"}.issubset(calibration):
        return evaluation
    valid = calibration.q05.notna() & calibration.q95.notna()
    if not valid.any():
        return evaluation
    return CF.apply_cqr(evaluation, CF.cqr_offsets(calibration.loc[valid], alpha=0.10))


def _common_point_frames(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames = {
        model: ensemble_prediction_frame(predictions, model)
        for model in POINT_MODELS
    }
    frames = {model: frame[frame.split == "test"].copy()
              for model, frame in frames.items() if not frame.empty}
    if set(frames) != set(POINT_MODELS):
        missing = sorted(set(POINT_MODELS) - set(frames))
        raise RuntimeError(f"missing point models for common-key evaluation: {missing}")
    for horizon in C.HORIZONS:
        key_sets = []
        for frame in frames.values():
            rows = frame[frame.horizon == horizon]
            key_sets.append(set(map(tuple, rows[IDENTITY].itertuples(index=False, name=None))))
        common = set.intersection(*key_sets)
        if not common:
            raise RuntimeError(f"no common evaluation keys at horizon {horizon}")
        common_index = pd.MultiIndex.from_tuples(sorted(common), names=IDENTITY)
        for model, frame in frames.items():
            selected = frame[frame.horizon == horizon].set_index(IDENTITY)
            selected = selected.loc[common_index].reset_index()
            # Every model must carry exactly the same outcome on every key.
            frames[model] = pd.concat(
                [frames[model][frames[model].horizon != horizon], selected], ignore_index=True
            )
    reference = frames[POINT_MODELS[0]][IDENTITY + ["y_true"]].sort_values(IDENTITY)
    for model in POINT_MODELS[1:]:
        candidate = frames[model][IDENTITY + ["y_true"]].sort_values(IDENTITY)
        if not reference[IDENTITY].reset_index(drop=True).equals(
            candidate[IDENTITY].reset_index(drop=True)
        ) or not np.allclose(reference.y_true, candidate.y_true, equal_nan=True):
            raise AssertionError(f"{model} does not share the common outcome registry")
    return frames


def main() -> None:
    thresholds, climatology = _training_references()
    predictions = R.load_route_a_predictions(
        PREDICTIONS,
        root=ROOT,
        panel_path=PANEL,
        registry_path=STATION_REGISTRY,
    )
    huc = huc2_cluster_map(load_station_registry(STATION_REGISTRY))

    score_rows: list[dict] = []
    reliability: dict[str, pd.DataFrame] = {}
    calibrator_record: dict[str, dict] = {}
    for model in PROB_MODELS:
        ensemble = ensemble_prediction_frame(predictions, model)
        interval_eval = _cqr_evaluation(ensemble)
        event_eval, calibrators = calibrated_event_frame(
            predictions,
            model,
            thresholds=thresholds,
            climatology=climatology,
            calibration_split="calib",
            evaluation_split="test",
        )
        calibrator_record[model] = {
            str(h): calibrator.as_dict() for h, calibrator in calibrators.items()
        }
        reliability[model] = event_eval
        for horizon in C.HORIZONS:
            interval = interval_eval[interval_eval.horizon == horizon]
            events = event_eval[event_eval.horizon == horizon]
            if interval.empty or events.empty:
                continue
            quantiles = {
                0.05: interval.q05.to_numpy(float),
                0.50: interval.q50.to_numpy(float),
                0.95: interval.q95.to_numpy(float),
            }
            probabilistic = M.probabilistic_scores(
                interval.y_true.to_numpy(float), quantiles
            )
            y_event = events.event.to_numpy(int)
            p_raw = events.p_exceed.to_numpy(float)
            p_cal = events.p_exceed_calibrated.to_numpy(float)
            p_ref = events.p_reference.to_numpy(float)
            raw_scores = M.event_scores(y_event, p_raw, p_ref)
            cal_scores = M.event_scores(y_event, p_cal, p_ref)
            intercept, slope = calibration_intercept_slope(y_event, p_cal)
            score_rows.append({
                "model": model,
                "horizon": horizon,
                "PICP": probabilistic["PICP"],
                "MPIW": probabilistic["MPIW"],
                "WINKLER": probabilistic["WINKLER"],
                "THREE_QUANTILE_SCORE": probabilistic["THREE_QUANTILE_SCORE"],
                "Brier_raw": raw_scores["BRIER"],
                "Brier_calibrated": cal_scores["BRIER"],
                "BrierSkill_calibrated": cal_scores["BRIER_SKILL"],
                "LogLoss_calibrated": cal_scores["LOG_LOSS"],
                "ECE_calibrated": expected_calibration_error(y_event, p_cal),
                "calibration_intercept": intercept,
                "calibration_slope": slope,
                "AUPRC": cal_scores.get("AUPRC", np.nan),
                "AUROC": cal_scores.get("AUROC", np.nan),
                "base_rate_evaluation": cal_scores["BASE_RATE"],
                "reference_brier": M.brier(y_event, p_ref),
            })
    probabilistic_scores = pd.DataFrame(score_rows)
    atomic_write_bytes(
        C.TABLES / "probabilistic_scores.csv",
        probabilistic_scores.to_csv(index=False).encode(),
    )
    atomic_write_json(C.TABLES / "event_calibrators.json", calibrator_record)

    point_rows: list[dict] = []
    point_frames = _common_point_frames(predictions)
    for model, evaluation in point_frames.items():
        for horizon in C.HORIZONS:
            group = evaluation[evaluation.horizon == horizon]
            scores = M.point_scores(group.y_true.to_numpy(float), group.y_pred.to_numpy(float))
            per_station = {
                site: M.rmse(part.y_true.to_numpy(float), part.y_pred.to_numpy(float))
                for site, part in group.groupby("site_id")
            }
            regions: dict[str, list[float]] = {}
            for site, value in per_station.items():
                regions.setdefault(huc.get(site, "unmapped"), []).append(value)
            point_rows.append({
                "model": model,
                "horizon": horizon,
                "n_common": len(group),
                "RMSE": scores["RMSE"],
                "MAE": scores["MAE"],
                "NSE": scores["NSE"],
                "KGE": scores["KGE"],
                "PBIAS": scores["PBIAS"],
                "RMSE_region_wtd": float(np.mean([np.median(v) for v in regions.values()])),
            })
    point_scores = pd.DataFrame(point_rows)
    atomic_write_bytes(C.TABLES / "multi_metric.csv", point_scores.to_csv(index=False).encode())

    fig, axes = plt.subplots(1, len(C.HORIZONS), figsize=(12, 4), sharex=True, sharey=True)
    bins = np.linspace(0, 1, 11)
    for axis, horizon in zip(axes, C.HORIZONS):
        axis.plot([0, 1], [0, 1], color="#888", linestyle="--", linewidth=1)
        for model, frame in reliability.items():
            group = frame[frame.horizon == horizon]
            y = group.event.to_numpy(int)
            for column, style, label_suffix in (
                ("p_exceed", ":", " raw"),
                ("p_exceed_calibrated", "-", " calibrated"),
            ):
                p = group[column].to_numpy(float)
                membership = np.digitize(p, bins[1:-1])
                points = [
                    (float(p[membership == b].mean()), float(y[membership == b].mean()))
                    for b in range(10) if (membership == b).sum() >= 30
                ]
                if points:
                    x, observed = zip(*points)
                    axis.plot(x, observed, style, color=COLORS[model], linewidth=1.5,
                              label=model + label_suffix)
        axis.set_title(f"h={horizon} d")
        axis.grid(alpha=0.25)
        axis.set_xlabel("forecast probability")
    axes[0].set_ylabel("observed frequency")
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Train-q90 event reliability: raw and 2018-calibrated probabilities")
    fig.savefig(C.FIGURES / "fig_reliability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# Route-A probabilistic verification\n",
        "2019--2020 is a previously inspected development evaluation, not a blind test. "
        "Event probabilities are Platt-calibrated on 2018 only. Brier skill is against "
        "a station-month climatology fitted on 2006--2015; the evaluation event rate "
        "is never used as the reference forecast.\n",
        "The distribution score is the three-quantile score over q05/q50/q95, not CRPS. "
        "CQR coverage is empirical marginal coverage under this temporal protocol.\n",
        "| model | h | PICP | width | 3Q score | Brier raw | Brier cal | BSS cal | "
        "log loss | ECE | cal slope | AUPRC |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in probabilistic_scores.iterrows():
        lines.append(
            f"| {row.model} | {int(row.horizon)} | {row.PICP:.3f} | {row.MPIW:.2f} | "
            f"{row.THREE_QUANTILE_SCORE:.3f} | {row.Brier_raw:.3f} | "
            f"{row.Brier_calibrated:.3f} | {row.BrierSkill_calibrated:+.3f} | "
            f"{row.LogLoss_calibrated:.3f} | {row.ECE_calibrated:.3f} | "
            f"{row.calibration_slope:.2f} | {row.AUPRC:.3f} |"
        )
    lines.extend([
        "",
        "All point models in `multi_metric.csv` are evaluated on the exact same "
        "(site, horizon, issue date, target date) registry.",
        "",
        "![reliability](../figures/fig_reliability.png)",
    ])
    atomic_write_bytes(C.REPORTS / "probabilistic.md", "\n".join(lines).encode())
    print("\n".join(lines))


if __name__ == "__main__":
    main()

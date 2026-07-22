#!/usr/bin/env python3
"""Stage 8 — exploratory cost–loss value on the three-station case study.

This is a *hypothetical* decision analysis.  Event thresholds, probability
calibration, and the seasonal reference forecast are all fitted outside the
2019–2020 evaluation rows.  Every model is scored on the same forecast keys.
No operational benefit or realised management saving is claimed.
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
from thermoroute import data as D
from thermoroute import decision as DEC
from thermoroute import probability as P
from thermoroute import results as R


MODELS = [
    ("ThermoRoute", {"scope": "joint", "feature_set": "V3"}, True, "#185FA5"),
    ("LightGBM", {"feature_set": "V3"}, True, "#7F77DD"),
    ("DampedPersistence", {}, False, "#BA7517"),
    ("Persistence", {}, False, "#888780"),
]
KEYS = ["site_id", "horizon", "issue_date", "target_date", "split"]


def _thresholds_and_reference() -> tuple[dict[str, float], P.SeasonalClimatology]:
    panel = pd.read_parquet(C.DATA_PROCESSED / "panel.parquet")
    masks = D.split_masks(panel["DATE"])
    thresholds = R.exceedance_thresholds(panel, masks)
    reference = P.fit_seasonal_climatology(panel.loc[masks.train], thresholds)
    return thresholds, reference


def _select(predictions: pd.DataFrame, model: str, filters: dict[str, str]) -> pd.DataFrame:
    selected = predictions[predictions.model == model].copy()
    for name, value in filters.items():
        selected = selected[selected[name] == value]
    return selected


def _model_frame(
    predictions: pd.DataFrame,
    model: str,
    filters: dict[str, str],
    probabilistic: bool,
    thresholds: dict[str, float],
    reference: P.SeasonalClimatology,
) -> pd.DataFrame:
    selected = _select(predictions, model, filters)
    if probabilistic:
        evaluation, _ = P.calibrated_event_frame(
            selected,
            model,
            thresholds=thresholds,
            climatology=reference,
            calibration_split="calib",
            evaluation_split="test",
            min_calibration_samples=100,
        )
        evaluation["score"] = evaluation["p_exceed_calibrated"]
    else:
        evaluation = P.ensemble_prediction_frame(selected, model)
        evaluation = evaluation[evaluation.split == "test"].copy()
        evaluation["threshold"] = evaluation.site_id.astype(str).map(thresholds)
        if evaluation.threshold.isna().any():
            raise KeyError("a deterministic forecast has no pre-fitted event threshold")
        evaluation["event"] = (evaluation.y_true > evaluation.threshold).astype(int)
        evaluation["score"] = (evaluation.y_pred > evaluation.threshold).astype(float)
        evaluation["p_reference"] = reference.predict(
            evaluation.site_id.astype(str).to_numpy(), evaluation.target_date.to_numpy()
        )
    if evaluation.duplicated(KEYS).any():
        raise AssertionError(f"{model} contains duplicate forecast keys")
    return evaluation[KEYS + ["y_true", "event", "score", "p_reference"]]


def _common_frames(frames: dict[str, pd.DataFrame], horizon: int) -> dict[str, pd.DataFrame]:
    keys: set[tuple[object, ...]] | None = None
    indexed: dict[str, pd.DataFrame] = {}
    for model, frame in frames.items():
        subset = frame[frame.horizon == horizon].copy().set_index(KEYS).sort_index()
        indexed[model] = subset
        current = set(subset.index.tolist())
        keys = current if keys is None else keys & current
    if not keys:
        raise RuntimeError(f"no common forecast keys at horizon {horizon}")
    ordered = sorted(keys)
    aligned = {model: frame.loc[ordered].reset_index() for model, frame in indexed.items()}
    truth = next(iter(aligned.values()))["y_true"].to_numpy(float)
    reference = next(iter(aligned.values()))["p_reference"].to_numpy(float)
    for model, frame in aligned.items():
        if not np.allclose(frame.y_true, truth, rtol=0, atol=1e-8):
            raise AssertionError(f"y_true differs on common keys for {model}")
        if not np.allclose(frame.p_reference, reference, rtol=0, atol=1e-12):
            raise AssertionError(f"reference probability differs for {model}")
    return aligned


def main() -> None:
    predictions = pd.read_parquet(C.PREDICTIONS / "predictions.parquet")
    thresholds, reference = _thresholds_and_reference()
    model_frames = {
        model: _model_frame(predictions, model, filters, probabilistic,
                            thresholds, reference)
        for model, filters, probabilistic, _ in MODELS
    }
    alphas = np.linspace(0.01, 0.99, 99)
    rows: list[dict[str, float | str | int]] = []
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, horizon in zip(axes, C.HORIZONS):
        aligned = _common_frames(model_frames, horizon)
        for model, _, probabilistic, colour in MODELS:
            frame = aligned[model]
            events = frame.event.to_numpy(int)
            score = frame.score.to_numpy(float)
            p_ref = frame.p_reference.to_numpy(float)
            curve = DEC.rev_curve(
                events, score, alphas, probabilistic=probabilistic,
                reference_probability=p_ref,
            )
            ax.plot(
                alphas, np.clip(curve, -0.2, 1.0),
                "-" if probabilistic else "--", color=colour, lw=1.6,
                label=model + ("" if probabilistic else " (det.)"),
            )
            summary = DEC.value_summary(
                events, score, probabilistic=probabilistic,
                reference_probability=p_ref,
            )
            summary.update({"model": model, "horizon": horizon, "n_common": len(frame)})
            rows.append(summary)
        ax.axhline(0, color="#888780", lw=0.7)
        ax.set_title(f"h = {horizon} d")
        ax.set_xlabel("hypothetical cost–loss ratio α = C/L")
        ax.set_ylim(-0.2, 1.0)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Relative Economic Value")
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("Exploratory hypothetical decision value (2019–2020 development period)", y=1.03)
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / "fig11_rev_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(C.FIGURES / "fig11_rev_curves.pdf", bbox_inches="tight")
    plt.close(fig)

    values = pd.DataFrame(rows).sort_values(["horizon", "model"])
    C.TABLES.mkdir(parents=True, exist_ok=True)
    values.to_csv(C.TABLES / "decision_value.csv", index=False)
    lines = [
        "# Exploratory hypothetical decision value (2019–2020 development period)\n",
        "Probabilities are calibrated on 2018. The reference is a smoothed "
        "station-by-month climatology fitted on 2006–2015. Models share exact "
        "forecast keys. These generic cost–loss scenarios do not establish "
        "operational benefit, realised savings, or a management recommendation.\n",
        "| model | h | common n | observed event rate | REV max | α at max | REV@0.05 | REV@0.1 | REV@0.2 | REV@0.5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in values.iterrows():
        lines.append(
            f"| {row.model} | {int(row.horizon)} | {int(row.n_common)} | "
            f"{row.base_rate:.3f} | {row.REV_max:.3f} | {row.alpha_at_max:.2f} | "
            f"{row['REV@0.05']:.3f} | {row['REV@0.1']:.3f} | "
            f"{row['REV@0.2']:.3f} | {row['REV@0.5']:.3f} |"
        )
    (C.TABLES / "decision_value.md").write_text("\n".join(lines) + "\n")
    print("wrote exploratory decision-value outputs", flush=True)


if __name__ == "__main__":
    main()

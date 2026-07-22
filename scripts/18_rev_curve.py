#!/usr/bin/env python3
"""Exploratory Route-A cost--loss analysis with calibrated probabilities.

This is a hypothetical decision model, not evidence of regulatory or operational
value.  Event calibration is fitted on 2018.  The reference action uses a
2006--2015 station-month climatology and never the 2019--2020 event rate.
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
from thermoroute import results as R
from thermoroute.decision import cluster_bootstrap_rev, rev_curve
from thermoroute.evidence import FrozenPanelSpec
from thermoroute.probability import (
    calibrated_event_frame,
    ensemble_prediction_frame,
    fit_seasonal_climatology,
)
from thermoroute.repro import atomic_write_bytes
from thermoroute.spatial import huc2_cluster_map, load_station_registry


PANEL = ROOT / "data_usgs" / "panel_usgs_120v2.parquet"
PREDICTIONS = C.PREDICTIONS / "usgs_predictions_v2.parquet"
STATION_REGISTRY = ROOT / "data_usgs" / "station_registry_v1.csv"
GRID = np.linspace(0.01, 0.99, 99)
PROBABILISTIC = {"ThermoRoute": "#B3132B", "LightGBM": "#185FA5", "LSTM": "#6A4C93"}
DETERMINISTIC = {"Persistence": "#777777", "DampedPersistence": "#BA7517"}
IDENTITY = ["site_id", "horizon", "issue_date", "target_date"]


def _references():
    panel = FrozenPanelSpec.load().load_panel(stable_site_ids=True)
    panel["DATE"] = pd.to_datetime(panel.DATE)
    masks = D.split_masks(panel.DATE)
    train = panel.loc[masks.train].copy()
    thresholds = {
        str(site): float(group.WTEMP.quantile(C.EXCEEDANCE_QUANTILE))
        for site, group in train.groupby("site_id")
    }
    return thresholds, fit_seasonal_climatology(train, thresholds)


def _all_event_frames(predictions, thresholds, climatology):
    frames = {}
    for model in PROBABILISTIC:
        frame, _ = calibrated_event_frame(
            predictions, model, thresholds=thresholds, climatology=climatology
        )
        frame["score"] = frame.p_exceed_calibrated
        frame["probabilistic"] = True
        frames[model] = frame
    for model in DETERMINISTIC:
        frame = ensemble_prediction_frame(predictions, model)
        frame = frame[frame.split == "test"].copy()
        frame["threshold"] = frame.site_id.astype(str).map(thresholds)
        frame["event"] = (frame.y_true > frame.threshold).astype(int)
        frame["score"] = (frame.y_pred > frame.threshold).astype(float)
        frame["p_reference"] = climatology.predict(
            frame.site_id.astype(str).to_numpy(), frame.target_date.to_numpy()
        )
        frame["probabilistic"] = False
        frames[model] = frame

    # One registry for every method at each horizon.
    aligned = {model: [] for model in frames}
    for horizon in C.HORIZONS:
        key_sets = [
            set(map(tuple, frame.loc[frame.horizon == horizon, IDENTITY]
                    .itertuples(index=False, name=None)))
            for frame in frames.values()
        ]
        common = set.intersection(*key_sets)
        if not common:
            raise RuntimeError(f"no common event keys at horizon {horizon}")
        index = pd.MultiIndex.from_tuples(sorted(common), names=IDENTITY)
        for model, frame in frames.items():
            selected = frame[frame.horizon == horizon].set_index(IDENTITY).loc[index].reset_index()
            aligned[model].append(selected)
    return {model: pd.concat(parts, ignore_index=True) for model, parts in aligned.items()}


def main() -> None:
    thresholds, climatology = _references()
    predictions = R.load_route_a_predictions(
        PREDICTIONS,
        root=ROOT,
        panel_path=PANEL,
        registry_path=STATION_REGISTRY,
    )
    frames = _all_event_frames(predictions, thresholds, climatology)
    huc = huc2_cluster_map(load_station_registry(STATION_REGISTRY))

    records = []
    fig, axes = plt.subplots(1, len(C.HORIZONS), figsize=(12.6, 4), sharey=True)
    for axis, horizon in zip(axes, C.HORIZONS):
        for model, frame in frames.items():
            group = frame[frame.horizon == horizon]
            events = group.event.to_numpy(int)
            score = group.score.to_numpy(float)
            reference = group.p_reference.to_numpy(float)
            is_probability = model in PROBABILISTIC
            values = rev_curve(
                events, score, GRID, is_probability, reference_probability=reference
            )
            color = PROBABILISTIC.get(model, DETERMINISTIC[model])
            axis.plot(GRID, values, color=color, linewidth=1.7,
                      linestyle="-" if is_probability else "--", label=model)
            clusters = group.site_id.astype(str).map(huc).fillna(group.site_id).to_numpy()
            for alpha in (0.05, 0.10, 0.20, 0.50):
                value = rev_curve(
                    events, score, np.array([alpha]), is_probability,
                    reference_probability=reference,
                )[0]
                lo, hi = cluster_bootstrap_rev(
                    events, score, reference, clusters, alpha,
                    probabilistic=is_probability, n_boot=2000, seed=100 + horizon,
                )
                records.append({
                    "horizon": horizon,
                    "model": model,
                    "alpha": alpha,
                    "REV": value,
                    "REV_ci_low": lo,
                    "REV_ci_high": hi,
                    "n_common": len(group),
                    "n_clusters": len(np.unique(clusters)),
                })
        axis.axhline(0, color="#333", linewidth=0.8)
        axis.set_xlabel("hypothetical cost-loss ratio C/L")
        axis.set_title(f"h={horizon} d")
        axis.grid(alpha=0.25)
        axis.set_ylim(-0.2, 1.0)
    axes[0].set_ylabel("relative economic value")
    axes[-1].legend(fontsize=8, loc="lower center")
    fig.suptitle("Exploratory cost-loss sensitivity; 2018-calibrated event probabilities")
    fig.savefig(C.FIGURES / "fig_rev_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    result = pd.DataFrame(records)
    atomic_write_bytes(C.TABLES / "rev_curve.csv", result.to_csv(index=False).encode())
    lines = [
        "# Exploratory hypothetical cost--loss sensitivity\n",
        "This analysis does not use observed management costs, actions, ecological "
        "losses, or a regulatory endpoint. It therefore does not establish operational "
        "or economic value. Probabilities are fitted on 2018; the reference policy is "
        "a 2006--2015 station-month climatology. All methods use identical daily keys.\n",
        "| h | model | C/L | REV [HUC2 bootstrap 95% CI] | n |",
        "|---|---|---|---|---|",
    ]
    for _, row in result.iterrows():
        lines.append(
            f"| {int(row.horizon)} | {row.model} | {row.alpha:.2f} | "
            f"{row.REV:.3f} [{row.REV_ci_low:.3f}, {row.REV_ci_high:.3f}] | "
            f"{int(row.n_common)} |"
        )
    atomic_write_bytes(C.REPORTS / "rev_curve.md", "\n".join(lines).encode())
    print("\n".join(lines))


if __name__ == "__main__":
    main()

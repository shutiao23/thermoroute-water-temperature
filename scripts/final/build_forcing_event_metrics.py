#!/usr/bin/env python3
"""Does the forcing value show up in thermal events, or only in mean RMSE?

A station-median RMSE improvement of 0.63 degC is a statistics result.  What a
water manager acts on is whether a warm threshold will be crossed, and when.
This builder scores the F0 and F3 arms on warm-tail exceedance and on signed
rapid change, so the forcing value can be stated in terms a release schedule or
a thermal-refuge warning would use.

Thresholds are fitted on the 2006-2015 training period only, per station, and
never on the evaluation window:

* ``warm_q90`` / ``warm_q95``  -- station exceedance of its own training warm
  quantile at the target date;
* ``rapid_warming`` / ``rapid_cooling`` -- the signed h-day change exceeding the
  station's training q90 (or falling below its q10), computed separately in
  each direction because a model that is good at one is not automatically good
  at the other.

For each station, arm, lead and event class the builder reports POD, FAR, CSI
and the Brier score of a calibration-free probability surrogate, plus the
station-first paired F0-minus-F3 differences that make the forcing value
comparable with the RMSE result.

The probability surrogate needs stating plainly.  These are point predictors
with no fitted event probability, so the "probability" used for the Brier score
is a deterministic 0/1 indicator of the predicted event.  A Brier score on a
0/1 forecast is the misclassification rate; it is reported because it is the
conventional column, not because the models emit calibrated probabilities, and
it must not be read as a reliability result.

Chronology matches the arms it scores: the F0/F3 contrast is post-outcome and
descriptive, so everything here inherits that status.
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
from thermoroute.significance import cluster_bootstrap_paired_effect
from thermoroute.spatial import huc2_cluster_map, load_station_registry

FINAL = ROOT / "outputs" / "final"
SHARDS = FINAL / "forcing_regime_v5_observed" / "forcing_shards_v5_observed"
DEFAULT_OUT = FINAL / "forcing_event_metrics.parquet"

FORMAT = "thermoroute.forcing-event-metrics.v1"
TRAIN_START, TRAIN_END = pd.Timestamp("2006-01-01"), pd.Timestamp("2015-12-31")
MIN_EVENTS = 10          # a station-event cell needs this many positives
N_BOOT = 10_000
BOOT_SEED = 0

EVENTS = ("warm_q90", "warm_q95", "rapid_warming", "rapid_cooling")


class EventError(RuntimeError):
    """A threshold or coverage check failed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def training_thresholds(horizon: int) -> pd.DataFrame:
    """Per-station warm and signed-change thresholds from 2006-2015 only."""
    bounds = P.capture_placebo_inputs()
    raw, _stations, _ = V5._load_raw_panel_from_bounds(bounds)
    frame = V5._validated_raw_frame(raw)
    train = frame[
        (frame["DATE"] >= TRAIN_START) & (frame["DATE"] <= TRAIN_END)
    ][["site_id", "DATE", "WTEMP", "WTEMP_observed"]]

    rows = []
    for station, block in train.groupby("site_id", sort=True):
        block = block.sort_values("DATE")
        values = block["WTEMP"].to_numpy(float)
        observed = block["WTEMP_observed"].to_numpy(bool)
        dates = pd.to_datetime(block["DATE"]).to_numpy(dtype="datetime64[D]")
        finite = values[observed]
        if len(finite) < 100:
            continue
        gap = (dates[horizon:] - dates[:-horizon]) == np.timedelta64(horizon, "D")
        pair = observed[horizon:] & observed[:-horizon] & gap
        change = values[horizon:][pair] - values[:-horizon][pair]
        if len(change) < 100:
            continue
        rows.append({
            "site_id": station,
            "horizon": horizon,
            "warm_q90": float(np.quantile(finite, 0.90)),
            "warm_q95": float(np.quantile(finite, 0.95)),
            "change_q90": float(np.quantile(change, 0.90)),
            "change_q10": float(np.quantile(change, 0.10)),
        })
    if not rows:
        raise EventError(f"no station produced training thresholds at h{horizon}")
    return pd.DataFrame(rows)


def _contingency(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    hits = float(np.sum(observed & predicted))
    misses = float(np.sum(observed & ~predicted))
    false_alarms = float(np.sum(~observed & predicted))
    pod = hits / (hits + misses) if (hits + misses) > 0 else np.nan
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else np.nan
    denominator = hits + misses + false_alarms
    csi = hits / denominator if denominator > 0 else np.nan
    # deterministic 0/1 "probability": the Brier score is the misclassification
    # rate, reported as the conventional column and not as a reliability result
    brier = float(np.mean((predicted.astype(float) - observed.astype(float)) ** 2))
    return {"pod": pod, "far": far, "csi": csi, "brier": brier,
            "n_events": int(hits + misses)}


def score_arm(arm: str, model: str, horizon: int,
              thresholds: pd.DataFrame) -> pd.DataFrame:
    path = SHARDS / f"{arm}_{model}_h{horizon}_v5_observed.parquet"
    if not path.exists():
        raise EventError(f"shard missing: {path}")
    frame = pd.read_parquet(
        path, columns=["site_id", "issue_date", "target_date", "y_true", "y_pred"]
    )
    merged = frame.merge(
        thresholds[thresholds["horizon"] == horizon], on="site_id", how="inner"
    )
    if merged.empty:
        raise EventError(f"no station survived the threshold join for {arm} h{horizon}")

    # Warm-tail events only: they are a level comparison at the target date and
    # need nothing but the shard.  Signed rapid change needs the issue-date
    # temperature and is handled by score_change_events.
    rows = []
    for station, block in merged.groupby("site_id", sort=True):
        truth, pred = block["y_true"].to_numpy(), block["y_pred"].to_numpy()
        record: dict[str, Any] = {
            "site_id": station, "arm": arm, "model": model, "horizon": horizon,
            "n_keys": len(block),
        }
        for event in EVENTS:
            if event.startswith("warm_"):
                level = float(block[event].iloc[0])
                observed, predicted = truth >= level, pred >= level
            else:
                continue  # signed-change events: see score_change_events
            stats = _contingency(observed, predicted)
            if stats["n_events"] < MIN_EVENTS:
                continue
            record.update({f"{event}_{k}": v for k, v in stats.items()})
        rows.append(record)
    return pd.DataFrame(rows)


def score_change_events(arm: str, model: str, horizon: int,
                        thresholds: pd.DataFrame,
                        issue_values: pd.DataFrame) -> pd.DataFrame:
    """Signed rapid-change events, which need the issue-date temperature."""
    path = SHARDS / f"{arm}_{model}_h{horizon}_v5_observed.parquet"
    frame = pd.read_parquet(
        path, columns=["site_id", "issue_date", "y_true", "y_pred"]
    ).merge(issue_values, on=["site_id", "issue_date"], how="inner")
    frame = frame.merge(
        thresholds[thresholds["horizon"] == horizon], on="site_id", how="inner"
    )
    rows = []
    for station, block in frame.groupby("site_id", sort=True):
        actual = block["y_true"].to_numpy() - block["issue_wtemp"].to_numpy()
        forecast = block["y_pred"].to_numpy() - block["issue_wtemp"].to_numpy()
        record: dict[str, Any] = {
            "site_id": station, "arm": arm, "model": model, "horizon": horizon,
        }
        for event, comparison in (
            ("rapid_warming", lambda v, t: v >= t),
            ("rapid_cooling", lambda v, t: v <= t),
        ):
            level = float(
                block["change_q90"].iloc[0] if event == "rapid_warming"
                else block["change_q10"].iloc[0]
            )
            stats = _contingency(comparison(actual, level), comparison(forecast, level))
            if stats["n_events"] < MIN_EVENTS:
                continue
            record.update({f"{event}_{k}": v for k, v in stats.items()})
        rows.append(record)
    return pd.DataFrame(rows)


def load_issue_values() -> pd.DataFrame:
    bounds = P.capture_placebo_inputs()
    raw, _, _ = V5._load_raw_panel_from_bounds(bounds)
    frame = V5._validated_raw_frame(raw)
    out = frame.loc[
        frame["WTEMP_observed"].to_numpy(bool), ["site_id", "DATE", "WTEMP"]
    ].rename(columns={"DATE": "issue_date", "WTEMP": "issue_wtemp"})
    return out


def build() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    issue_values = load_issue_values()
    clusters = huc2_cluster_map(
        load_station_registry(ROOT / "data_usgs" / "station_registry_v1.csv")
    )
    per_station: list[pd.DataFrame] = []
    for horizon in V5.HORIZONS:
        thresholds = training_thresholds(horizon)
        for model in V5.MODELS:
            for arm in V5.ARMS:
                warm = score_arm(arm, model, horizon, thresholds)
                change = score_change_events(
                    arm, model, horizon, thresholds, issue_values
                )
                merged = warm.merge(
                    change, on=["site_id", "arm", "model", "horizon"], how="outer"
                )
                per_station.append(merged)
    table = pd.concat(per_station, ignore_index=True)

    # station-first paired F0 - F3 differences, one row per model/lead/event/metric
    summary: list[dict[str, Any]] = []
    for model in V5.MODELS:
        for horizon in V5.HORIZONS:
            f0 = table[(table["arm"] == "F0") & (table["model"] == model)
                       & (table["horizon"] == horizon)].set_index("site_id")
            f3 = table[(table["arm"] == "F3_full") & (table["model"] == model)
                       & (table["horizon"] == horizon)].set_index("site_id")
            shared = f0.index.intersection(f3.index)
            for event in EVENTS:
                for metric in ("pod", "far", "csi", "brier"):
                    column = f"{event}_{metric}"
                    if column not in f0.columns:
                        continue
                    pair = pd.concat(
                        [f0.loc[shared, column], f3.loc[shared, column]], axis=1
                    ).dropna()
                    if len(pair) < 20:
                        continue
                    # F3 minus F0: positive means the oracle arm scores higher
                    delta = (pair.iloc[:, 1] - pair.iloc[:, 0]).to_numpy()
                    groups = pair.index.map(clusters).to_numpy()
                    boot = cluster_bootstrap_paired_effect(
                        delta, groups, statistic="median",
                        n_boot=N_BOOT, seed=BOOT_SEED,
                    )
                    summary.append({
                        "model": model, "horizon": horizon, "event": event,
                        "metric": metric,
                        "median_F0": float(pair.iloc[:, 0].median()),
                        "median_F3": float(pair.iloc[:, 1].median()),
                        "median_delta_F3_minus_F0": boot["effect"],
                        "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
                        "n_stations": len(pair),
                        "higher_is_better": metric in ("pod", "csi"),
                    })
    return table, summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)

    table, summary = build()
    frame = pd.DataFrame(summary)
    if args.print_only:
        view = frame[(frame["horizon"] == 7) & (frame["model"] == "LightGBM")
                     & (frame["metric"].isin(["pod", "csi"]))]
        print(view[["event", "metric", "median_F0", "median_F3",
                    "median_delta_F3_minus_F0", "ci_low", "ci_high",
                    "n_stations"]].to_string(index=False))
        return 0

    table.to_parquet(args.out, index=False)
    summary_path = args.out.parent / "forcing_event_metrics_summary.parquet"
    frame.to_parquet(summary_path, index=False)
    manifest = {
        "format": FORMAT,
        "builder_sha256": _sha256_file(Path(__file__).resolve()),
        "outputs_sha256": {
            args.out.name: _sha256_file(args.out),
            summary_path.name: _sha256_file(summary_path),
        },
        "threshold_period": "2006-01-01..2015-12-31 (training only)",
        "minimum_events_per_station_cell": MIN_EVENTS,
        "brier_caveat": (
            "point predictors emit no calibrated probability; the Brier score "
            "here is the misclassification rate of a deterministic 0/1 "
            "indicator and is not a reliability result"
        ),
        "chronology": "inherits the post-outcome descriptive status of F0/F3",
    }
    (args.out.parent / "forcing_event_metrics_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=1) + "\n", encoding="utf-8",
    )
    print(json.dumps({"status": "WRITTEN", "station_rows": len(table),
                      "summary_rows": len(frame)}, sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Derive every manuscript statistic from the persisted prediction table.

Consumes ONLY the persisted per-key prediction parquet plus the station
registry — never a model bundle, never the panel, never the network — and
writes the full set of result tables under ``outputs/conventional/``.  This
is the "no re-run" guarantee: re-deriving every number takes seconds and
touches no model.  The module must import and run with torch absent
(enforced by an AST test).

Usage::

    python scripts/conventional_derive_statistics.py \\
        --predictions outputs/conventional/predictions_2021_2023.parquet \\
        --registry data_usgs/station_registry_v1.csv \\
        --out outputs/conventional/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from thermoroute import conventional_stats as CS  # noqa: E402


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    for column in ("model", "horizon", "site_id"):
        if column not in frame.columns:
            raise ValueError(f"prediction table lacks required column {column!r}")
    return frame


def load_registry(path: Path) -> pd.DataFrame:
    registry = pd.read_csv(path, dtype={"site_no": str})
    registry = registry[
        ["site_no", "lat", "lon", "huc2", "huc_metadata_status"]
    ].copy()
    registry["site_no"] = registry["site_no"].str.strip()
    return registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--out", default="outputs/conventional", type=Path)
    parser.add_argument("--n-boot", default=10000, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    pred = load_predictions(args.predictions)
    registry = load_registry(args.registry)
    pred["site_id"] = pred["site_id"].astype(str).str.zfill(8)

    station = CS.station_metrics(pred)
    station["site_id"] = station["site_id"].astype(str).str.zfill(8)
    pooled = CS.pooled_metrics(pred)
    skill = CS.skill_table(station, pooled)

    # Paired contrasts: every non-baseline model vs each baseline per lead,
    # plus the frozen five-test family's LightGBM references (H2 tests).
    contrasts = []
    non_baselines = sorted(set(pred["model"]) - set(CS.BASELINE_MODELS))
    for model in non_baselines:
        for base in CS.BASELINE_MODELS:
            for horizon in sorted(set(int(h) for h in pred["horizon"])):
                contrasts.append((model, base, horizon))
    for formal in CS.HOLM_FAMILY_CONTRASTS:
        if formal not in contrasts:
            contrasts.append(formal)
    effects = CS.paired_effects(station, contrasts=contrasts)
    effects["site_id"] = effects["site_id"].astype(str).str.zfill(8)
    inference = CS.cluster_inference(
        effects, registry, n_boot=args.n_boot, seed=args.seed,
        holm_family_contrasts=CS.HOLM_FAMILY_CONTRASTS,
    )
    probability = CS.probability_metrics(pred)
    reliability = CS.reliability_bins(pred)

    station.to_csv(args.out / "station_metrics_2021_2023.csv", index=False)
    pooled.to_csv(args.out / "pooled_metrics_2021_2023.csv", index=False)
    skill.to_csv(args.out / "skill_table_2021_2023.csv", index=False)
    effects.to_csv(args.out / "paired_effects_2021_2023.csv", index=False)
    probability.to_csv(args.out / "probability_metrics_2021_2023.csv", index=False)
    reliability.to_csv(args.out / "reliability_bins_2021_2023.csv", index=False)
    (args.out / "cluster_inference_2021_2023.json").write_text(
        json.dumps(inference, indent=2, default=str)
    )

    summary = {
        "n_prediction_rows": int(len(pred)),
        "n_stations": int(pred.site_id.nunique()),
        "n_reportable_stations_h1": int(
            station[station.horizon == 1].groupby("model").site_id.nunique().max()
        ),
        "models": sorted(set(map(str, pred["model"]))),
        "horizons": sorted(set(int(h) for h in pred["horizon"])),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

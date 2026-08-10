#!/usr/bin/env python3
"""Why do some rivers depend on future weather and others do not?

The seven-day forcing value has a wide station spread -- the middle half runs
from about 0.30 to 0.86 degC -- and a box plot of that spread is a description,
not an explanation.  This builder relates the station-level forcing value to
physically interpretable station properties, so the result can be stated as
hydrology rather than as a model score.

Attributes are declared here, before the relationships are inspected, and are
restricted to quantities already frozen in the repository:

* ``half_life_days``      -- the anomaly half-life of the train-fitted damped
  anchor, i.e. how long the river holds a thermal anomaly.  The primary
  hypothesis: a river with a long thermal memory carries its own state forward
  and has less need of tomorrow's weather.
* ``log_drain_area``      -- catchment area.  Large rivers have more thermal
  mass and longer response times.
* ``mean_annual_air_temp`` -- a coarse climate position.
* ``latitude`` / ``elevation_proxy`` -- position, with latitude standing in for
  the seasonal-amplitude gradient.
* ``flow_cv``             -- the coefficient of variation of training discharge,
  a crude index of hydrologic flashiness.
* ``wtemp_observed_fraction`` -- record completeness, included as a nuisance
  control so a data-availability gradient cannot masquerade as a physical one.

The analysis is deliberately modest.  With 116 stations, roughly nine effective
spatial clusters and a non-probability cohort, a multivariable model would
overstate what the design can carry.  We therefore report Spearman rank
correlations with whole-HUC2 cluster bootstrap intervals, plus a single
bivariate ordinary least squares fit against the primary hypothesis, and we
label the whole thing exploratory.  It is a description of covariation on a
fixed cohort, not an attribution.
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
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.final.run_forcing_ladder_v5_observed as V5
import scripts.final.run_forcing_placebo_v5a_shuffle as P
from thermoroute.spatial import huc2_cluster_map, load_station_registry

FINAL = ROOT / "outputs" / "final"
POINT_EFFECTS = (
    FINAL / "forcing_regime_v5_observed_point_authority_v1"
    / "forcing_v5_point_paired_effects.parquet"
)
BASIN = FINAL / "basin_attributes.parquet"
REGISTRY = ROOT / "data_usgs" / "station_registry_v1.csv"
DEFAULT_OUT = FINAL / "forcing_heterogeneity.parquet"

FORMAT = "thermoroute.forcing-heterogeneity.v1"
N_BOOT = 10_000
BOOT_SEED = 0
TRAIN_START, TRAIN_END = pd.Timestamp("2006-01-01"), pd.Timestamp("2015-12-31")

#: Declared before inspection.  The first is the primary hypothesis.
ATTRIBUTES = (
    ("log_half_life", "log anomaly half-life of the train-fitted anchor (days)"),
    ("log_drain_area", "log catchment drainage area"),
    ("mean_annual_air_temp_c", "mean annual air temperature"),
    ("latitude", "station latitude"),
    ("flow_cv", "coefficient of variation of training discharge"),
    ("wtemp_observed_fraction", "training water-temperature record completeness"),
)
PRIMARY = "log_half_life"


class HeterogeneityError(RuntimeError):
    """An input or coverage check failed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def station_attributes() -> pd.DataFrame:
    basin = pd.read_parquet(BASIN)
    registry = pd.read_csv(REGISTRY, dtype={"site_no": str})

    bounds = P.capture_placebo_inputs()
    raw, _stations, _ = V5._load_raw_panel_from_bounds(bounds)
    panel = V5._validated_raw_frame(raw)
    train = panel[(panel["DATE"] >= TRAIN_START) & (panel["DATE"] <= TRAIN_END)]

    derived = []
    for station, block in train.groupby("site_id", sort=True):
        flow = block["FLOW"].to_numpy(float)
        finite = flow[np.isfinite(flow)]
        # signed discharge exists in this cohort; a CV on a signed series is
        # meaningless, so it is computed on the positive part and stations with
        # too little positive flow are left missing rather than imputed
        positive = finite[finite > 0]
        flow_cv = (
            float(np.std(positive) / np.mean(positive))
            if len(positive) >= 100 and np.mean(positive) > 0 else np.nan
        )
        derived.append({
            "site_id": station,
            "flow_cv": flow_cv,
            "wtemp_observed_fraction": float(
                np.mean(block["WTEMP_observed"].to_numpy(bool))
            ),
        })

    frame = (
        basin[["site_id", "lat", "drain_area_va", "half_life_days",
               "mean_annual_air_temp_c"]]
        .merge(pd.DataFrame(derived), on="site_id", how="left")
        .merge(registry[["site_no", "state"]].rename(columns={"site_no": "site_id"}),
               on="site_id", how="left")
    )
    frame["latitude"] = frame["lat"]
    frame["log_half_life"] = np.log(frame["half_life_days"].clip(lower=1e-3))
    frame["log_drain_area"] = np.log(frame["drain_area_va"].clip(lower=1e-3))
    return frame


def cluster_bootstrap_spearman(
    x: np.ndarray, y: np.ndarray, clusters: np.ndarray,
) -> dict[str, float]:
    """Spearman rho with whole-HUC2 resampling, which the station spread needs."""
    unique = np.unique(clusters)
    groups = {c: np.flatnonzero(clusters == c) for c in unique}
    rng = np.random.default_rng(BOOT_SEED)
    draws = np.empty(N_BOOT)
    for index in range(N_BOOT):
        picked = np.concatenate(
            [groups[c] for c in rng.choice(unique, size=len(unique), replace=True)]
        )
        if len(np.unique(x[picked])) < 3:
            draws[index] = np.nan
            continue
        draws[index] = stats.spearmanr(x[picked], y[picked]).statistic
    finite = draws[np.isfinite(draws)]
    point = stats.spearmanr(x, y)
    return {
        "spearman_rho": float(point.statistic),
        "ci_low": float(np.percentile(finite, 2.5)),
        "ci_high": float(np.percentile(finite, 97.5)),
        "n_stations": len(x),
        "n_clusters": len(unique),
    }


def build_rows() -> list[dict[str, Any]]:
    effects = pd.read_parquet(POINT_EFFECTS)
    attributes = station_attributes()
    clusters = huc2_cluster_map(load_station_registry(REGISTRY))
    rows: list[dict[str, Any]] = []

    for (model, horizon), block in effects.groupby(["model", "horizon"], sort=True):
        merged = block.merge(attributes, on="site_id", how="left")
        for name, label in ATTRIBUTES:
            usable = merged[
                np.isfinite(merged[name].to_numpy(float))
                & np.isfinite(merged["forcing_value_rmse_F0_minus_F3"].to_numpy(float))
            ]
            if len(usable) < 40:
                rows.append({
                    "model": model, "horizon": int(horizon), "attribute": name,
                    "label": label, "status": "SKIPPED_INSUFFICIENT_COVERAGE",
                    "n_stations": len(usable),
                })
                continue
            x = usable[name].to_numpy(float)
            y = usable["forcing_value_rmse_F0_minus_F3"].to_numpy(float)
            groups = usable["site_id"].map(clusters).to_numpy()
            record = {
                "model": model, "horizon": int(horizon), "attribute": name,
                "label": label, "status": "COMPUTED",
                "is_primary_hypothesis": name == PRIMARY,
                "inference_role": "EXPLORATORY_COVARIATION_NOT_ATTRIBUTION",
            }
            record.update(cluster_bootstrap_spearman(x, y, groups))
            if name == PRIMARY:
                fit = stats.linregress(x, y)
                record["ols_slope"] = float(fit.slope)
                record["ols_r_squared"] = float(fit.rvalue ** 2)
            rows.append(record)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)

    rows = build_rows()
    table = pd.DataFrame(rows)
    if args.print_only:
        view = table[(table["horizon"] == 7) & (table["model"] == "LightGBM")
                     & (table["status"] == "COMPUTED")]
        print(view[["attribute", "spearman_rho", "ci_low", "ci_high",
                    "n_stations"]].to_string(index=False))
        return 0

    table.to_parquet(args.out, index=False)
    manifest = {
        "format": FORMAT,
        "builder_sha256": _sha256_file(Path(__file__).resolve()),
        "output_sha256": _sha256_file(args.out),
        "declared_attributes": [name for name, _ in ATTRIBUTES],
        "primary_hypothesis": PRIMARY,
        "bootstrap": {"draws": N_BOOT, "seed": BOOT_SEED, "unit": "HUC2"},
        "note": (
            "Exploratory covariation on a fixed non-probability cohort with "
            "about nine effective spatial clusters; not an attribution and not "
            "a multivariable model."
        ),
    }
    (args.out.parent / "forcing_heterogeneity_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=1) + "\n", encoding="utf-8",
    )
    print(json.dumps({"status": "WRITTEN", "rows": len(table)},
                     sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Add the exploratory per-station LightGBM as an immutable derived artifact.

The Stage-9 prediction file is never overwritten.  This script trains on the
same window registry as the sequence models, aligns its evaluation rows to the
frozen Stage-9 primary keys, and writes a new content-addressed derivative.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")

ROOT = Path(__file__).resolve().parents[1]


def _isolate_project_bytecode() -> None:
    if __name__ != "__main__":
        return
    prefix = Path(sys.pycache_prefix).resolve() if sys.pycache_prefix else None
    if (
        sys.flags.isolated
        and prefix is not None
        and prefix != ROOT
        and ROOT not in prefix.parents
    ):
        return
    with tempfile.TemporaryDirectory(prefix="thermoroute-perstation-pycache-") as cache:
        result = subprocess.run(
            [sys.executable, "-I", "-X", f"pycache_prefix={cache}",
             str(Path(__file__).resolve()), *sys.argv[1:]],
            cwd=ROOT,
            env=os.environ.copy(),
            check=False,
        )
    raise SystemExit(result.returncode)


_isolate_project_bytecode()
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from thermoroute import baselines as B
from thermoroute import config as C
from thermoroute import data as D
from thermoroute import datasets as DS
from thermoroute import features as F
from thermoroute import results as R
from thermoroute.registry import (
    FORECAST_KEY,
    restrict_tabular_to_window_registry,
    targets_match_at_model_precision,
)
from thermoroute.repro import (
    cache_is_valid,
    resolve_run_identity,
    seal_artifact,
    sha256_file,
    sidecar_path,
)


USGS_VARS = ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP")
DEFAULT_PANEL = ROOT / "data_usgs" / "panel_usgs_120v2.parquet"
DEFAULT_REGISTRY = ROOT / "data_usgs" / "station_registry_v1.csv"
DEFAULT_INPUT = C.PREDICTIONS / "usgs_predictions_stage9_v2.parquet"
DEFAULT_OUTPUT = C.PREDICTIONS / "usgs_predictions_with_perstation_v2.parquet"


def _verify_parent(path: Path) -> None:
    metadata_path = sidecar_path(path)
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Stage-9 predictions and sidecar are required: {path}")
    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("artifact_sha256") != sha256_file(path)
        or metadata.get("artifact_bytes") != path.stat().st_size
        or metadata.get("content_schema") != R.PREDICTION_SCHEMA_VERSION
    ):
        raise ValueError("Stage-9 parent predictions fail their lineage checksum/schema")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    panel_path = args.panel.resolve()
    registry_path = args.registry.resolve()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    _verify_parent(input_path)
    parent_sha = sha256_file(input_path)
    run_config = {
        "stage": "per_station_lightgbm",
        "role": "exploratory_derived",
        "parent_sha256": parent_sha,
        "variables": USGS_VARS,
        "horizons": C.HORIZONS,
        "missingness_features": True,
        "registry_alignment": "window_registry_all_splits_and_stage9_primary_test_keys",
    }
    identity = resolve_run_identity(
        root=ROOT, panel=panel_path, registry=registry_path, config=run_config,
    )
    if cache_is_valid(output_path, identity, schema=R.PREDICTION_SCHEMA_VERSION):
        print(f"verified cache: {output_path}")
        return

    started = time.time()
    bundle = D.prepare_dataset_from_panel(str(panel_path))
    panel, panel_imputed, masks = (
        bundle["panel_raw"], bundle["panel"], bundle["masks"]
    )
    stations = bundle["stations"]
    climatology = F.HarmonicClimatology.fit(panel, masks.train)
    thresholds = {
        site: float(panel.loc[masks.train & panel.site_id.eq(site).to_numpy(), "WTEMP"].quantile(0.9))
        for site in stations
    }
    windows = DS.build_windows(
        panel_imputed, masks, climatology, variables=USGS_VARS,
        require_observed_target=True,
    )

    tabs: dict[int, pd.DataFrame] = {}
    for horizon in C.HORIZONS:
        tab = F.attach_split(F.build_tabular(
            panel_imputed, horizon, USGS_VARS, climatology,
            drop_feature_nans=False, require_observed_target=True,
            include_missingness=True,
        ))
        tab = restrict_tabular_to_window_registry(tab, windows, C.STATIONS, horizon)
        for column in F.feature_columns(tab):
            tab[column] = pd.to_numeric(tab[column], errors="coerce").fillna(0.0)
        tabs[horizon] = tab

    predictions = B.run_lightgbm(tabs, thresholds, feature_set="USGS")
    predictions["model"] = "LightGBM-perstation"
    predictions = predictions[predictions.split.eq("test")].copy()

    parent = pd.read_parquet(input_path)
    R.validate_predictions(parent)
    primary = parent[
        parent.model.eq("ThermoRoute") & parent.split.eq("test")
    ][list(FORECAST_KEY) + ["y_true"]].drop_duplicates(list(FORECAST_KEY))
    aligned = predictions.merge(
        primary.rename(columns={"y_true": "__parent_y"}),
        on=list(FORECAST_KEY), how="inner", validate="one_to_one",
    )
    if len(aligned) != len(primary):
        raise ValueError(
            f"per-station LightGBM covers {len(aligned)}/{len(primary)} Stage-9 primary keys"
        )
    # The tabular path inherits parquet float64 labels while Stage 9 stores the
    # float32 labels consumed by the sequence models.  Compare the shared model
    # precision exactly: this accepts representation-only drift but still
    # rejects the next float32 ULP.
    if not targets_match_at_model_precision(aligned.y_true, aligned.__parent_y):
        raise ValueError("per-station LightGBM target labels differ from Stage-9")
    aligned = aligned.drop(columns="__parent_y")

    derived = pd.concat(
        [parent[parent.model.ne("LightGBM-perstation")], aligned], ignore_index=True
    )
    R.write_predictions(derived, output_path)
    seal_artifact(
        output_path, identity, kind="derived_predictions_with_perstation_lightgbm",
        schema=R.PREDICTION_SCHEMA_VERSION,
        parents={input_path.name: parent_sha},
        extra={"model_role": "exploratory", "added_rows": len(aligned)},
    )
    print(
        f"wrote {output_path} with {len(aligned)} aligned rows in "
        f"{time.time() - started:.0f}s"
    )


if __name__ == "__main__":
    main()

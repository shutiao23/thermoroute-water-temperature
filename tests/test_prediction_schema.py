from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoroute.registry import ROUTE_A_PRIMARY_MODELS
from thermoroute.repro import resolve_run_identity, seal_artifact
from thermoroute.results import (
    PRED_COLS,
    load_route_a_predictions,
    validate_predictions,
    write_predictions,
)


def _valid():
    frame = pd.DataFrame({
        "model": ["m"], "scope": ["joint"], "feature_set": ["V1"], "seed": [0],
        "site_id": ["01234567"], "horizon": [3], "split": ["test"],
        "issue_date": ["2020-01-01"], "target_date": ["2020-01-04"],
        "y_true": [10.0], "y_pred": [10.1], "q05": [9.0], "q50": [10.0],
        "q95": [11.0], "p_exceed": [0.2],
    })
    return frame[PRED_COLS]


def test_prediction_schema_accepts_valid_and_writes_atomically(tmp_path):
    frame = _valid()
    validate_predictions(frame)
    path = tmp_path / "predictions.parquet"
    write_predictions(frame, path)
    assert path.exists() and pd.read_parquet(path).equals(frame)


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d.assign(target_date="2020-01-03"), "target_date"),
    (lambda d: d.assign(q05=12.0), "quantiles"),
    (lambda d: d.assign(p_exceed=1.2), "p_exceed"),
    (lambda d: pd.concat([d, d], ignore_index=True), "duplicate"),
    (lambda d: d.assign(y_pred=np.nan), "null"),
])
def test_prediction_schema_rejects_corruption(mutation, match):
    with pytest.raises(ValueError, match=match):
        validate_predictions(mutation(_valid()))


def _sealed_route_a_fixture(tmp_path: Path, *, site_id: str = "01234567"):
    root = tmp_path / "repo"
    for directory in ("src", "scripts", "tests"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    (root / "requirements-lock.txt").write_text("numpy==1\n")
    panel = root / "panel.csv"
    panel.write_text("site_id,y\n01234567,1\n")
    registry = root / "registry.csv"
    registry.write_text("site_no\n01234567\n")
    frames = []
    for model in ROUTE_A_PRIMARY_MODELS:
        frame = _valid().assign(model=model, site_id=site_id)
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    path = root / "predictions.parquet"
    write_predictions(predictions, path)
    identity = resolve_run_identity(
        root=root, panel=panel, registry=registry, config={"stage": "final"}
    )
    seal_artifact(
        path,
        identity,
        kind="final_route_a_development_predictions",
        schema="thermoroute.predictions.v1",
        parents={"parent.parquet": "0" * 64},
    )
    return root, panel, registry, path


def test_route_a_loader_requires_lineage_and_stable_site_registry(tmp_path):
    root, panel, registry, path = _sealed_route_a_fixture(tmp_path)
    loaded = load_route_a_predictions(
        path,
        root=root,
        panel_path=panel,
        registry_path=registry,
        require_current_source=False,
    )
    assert set(loaded.model) == set(ROUTE_A_PRIMARY_MODELS)

    path.with_name(path.name + ".meta.json").unlink()
    with pytest.raises(FileNotFoundError, match="lineage sidecar"):
        load_route_a_predictions(
            path,
            root=root,
            panel_path=panel,
            registry_path=registry,
            require_current_source=False,
        )


def test_route_a_loader_rejects_legacy_station_aliases(tmp_path):
    root, panel, registry, path = _sealed_route_a_fixture(tmp_path, site_id="n00")
    with pytest.raises(ValueError, match="legacy or unknown"):
        load_route_a_predictions(
            path,
            root=root,
            panel_path=panel,
            registry_path=registry,
            require_current_source=False,
        )

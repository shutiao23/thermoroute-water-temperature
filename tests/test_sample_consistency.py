"""Identical-keys enforcement test for the USGS predictions registry.

If a USGS predictions file (``usgs_predictions*.parquet``) is present, this test
asserts that every comparable model emits predictions on the SAME
(site_id, horizon, issue_date) keys on the test split. Without this guarantee,
RMSE/skill/win-rate numbers between models are not comparable — the issue raised
in the advisor's adversarial review (LightGBM walked ``panel_raw + dropna``
while ThermoRoute walked ``panel_imp + require_observed_target``).

The test is skipped when no predictions file exists (so the suite still runs
on a fresh checkout without re-training).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

from thermoroute import config as C

COMPARABLE_MODELS = ("Persistence", "Climatology", "DampedPersistence",
                     "Air2stream-a4", "Air2stream-a8", "LightGBM", "ThermoRoute")


def _candidate_files():
    return [
        C.PREDICTIONS / "usgs_predictions_120.parquet",
        C.PREDICTIONS / "usgs_predictions.parquet",
    ]


@pytest.mark.parametrize("path", _candidate_files())
def test_test_split_keys_identical_across_models(path):
    if not path.exists():
        pytest.skip(f"{path} not found — run scripts/09_usgs_experiment.py to generate")
    df = pd.read_parquet(path, columns=["model", "split", "site_id",
                                        "horizon", "issue_date"])
    df = df[df["split"] == "test"]
    present = sorted(set(df["model"]) & set(COMPARABLE_MODELS))
    if len(present) < 2:
        pytest.skip(f"only {present} present in {path.name}; need ≥2 for comparison")
    df["key"] = list(zip(df["site_id"], df["horizon"], df["issue_date"].astype(str)))
    keys_per_model = {m: set(g["key"]) for m, g in df.groupby("model") if m in present}
    sizes = {m: len(k) for m, k in keys_per_model.items()}
    common = set.intersection(*keys_per_model.values())
    for m, k in keys_per_model.items():
        extra = k - common
        assert not extra, (
            f"{m} has {len(extra)} test keys not shared by every other model in "
            f"{path.name}. Per-model sizes: {sizes}. Inner-join enforcement in "
            f"scripts/09_usgs_experiment.py:main() must be applied before saving."
        )

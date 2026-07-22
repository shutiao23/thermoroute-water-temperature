"""Unit tests for metrics, conformal coverage and the sparse router."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest
import torch

from thermoroute import metrics as M
from thermoroute.thermoroute import sparsemax
from thermoroute.conformal import (
    apply_cqr,
    block_cqr_offsets,
    conformal_quantile,
    cqr_offsets,
    hierarchical_cqr_offsets,
)


def test_perfect_forecast_scores():
    y = np.linspace(5, 25, 200)
    assert M.rmse(y, y) == 0.0
    assert M.r2(y, y) > 0.999
    assert abs(M.kge(y, y) - 1.0) < 1e-6


def test_skill_score_sign():
    y = np.array([10.0, 11, 12, 13])
    good = y + 0.1
    bad = y + 2.0
    assert M.skill_score(y, good, bad) > 0   # good beats bad reference


def test_pinball_median_is_half_mae():
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    q = np.zeros_like(y)
    assert np.isclose(M.pinball(y, q, 0.5), 0.5 * M.mae(y, q), atol=1e-9)


def test_sparsemax_is_sparse_and_normalised():
    z = torch.tensor([[2.0, 1.0, 0.1, -1.0, -3.0]])
    p = sparsemax(z, dim=-1)
    assert torch.allclose(p.sum(-1), torch.ones(1), atol=1e-6)
    assert (p == 0).sum().item() >= 2          # genuinely sparse
    assert (p >= 0).all()


def test_softmax_vs_sparsemax_density():
    z = torch.randn(4, 20)
    sp = sparsemax(z, -1)
    sm = torch.softmax(z, -1)
    assert (sp == 0).sum() > (sm == 0).sum()   # sparsemax zeros more entries


def test_cqr_improves_coverage_toward_nominal():
    rng = np.random.default_rng(1)
    n = 800
    y = rng.normal(0, 1, n)
    # deliberately too-narrow intervals
    df = pd.DataFrame({
        "site_id": "x", "horizon": 1,
        "y_true": y, "q05": -0.2 * np.ones(n), "q95": 0.2 * np.ones(n),
        "split": ["calib"] * (n // 2) + ["test"] * (n // 2),
    })
    cal = df[df.split == "calib"]
    off = cqr_offsets(cal, alpha=0.10)
    out = apply_cqr(df, off)
    test = out[out.split == "test"]
    picp = M.coverage(test.y_true.to_numpy(), test.q05.to_numpy(), test.q95.to_numpy())
    assert picp >= 0.80     # widened toward the nominal 90%


def test_conformal_small_sample_reports_unattainable_quantile():
    assert np.isinf(conformal_quantile(np.array([0.1, 0.2]), alpha=0.10))


def test_apply_cqr_rejects_an_unfrozen_site_horizon_key():
    frame = pd.DataFrame({
        "site_id": ["known", "unfrozen"],
        "horizon": [1, 3],
        "q05": [1.0, 2.0],
        "q95": [3.0, 4.0],
    })
    with pytest.raises(KeyError, match="missing prediction"):
        apply_cqr(frame, {("known", 1): 0.25})


def test_block_and_hierarchical_conformal_are_explicitly_pooled():
    n = 140
    df = pd.DataFrame({
        "site_id": np.repeat(["a", "b"], n // 2),
        "huc2": np.repeat(["west", "east"], n // 2),
        "horizon": 1,
        "issue_date": pd.date_range("2018-01-01", periods=n),
        "y_true": np.linspace(0, 1, n),
        "q05": np.linspace(0, 1, n) - 0.1,
        "q95": np.linspace(0, 1, n) + 0.1,
    })
    block = block_cqr_offsets(df, alpha=0.2, block_days=7)
    assert set(block) == {("a", 1), ("b", 1)}
    hierarchical = hierarchical_cqr_offsets(df, "huc2", alpha=0.2, min_group=20)
    assert ("__global__", 1) in hierarchical
    assert ("west", 1) in hierarchical and ("east", 1) in hierarchical

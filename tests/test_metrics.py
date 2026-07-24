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
    CQRContractError,
    apply_cqr,
    apply_hierarchical_cqr,
    block_cqr_offsets,
    conformal_quantile,
    cqr_offsets,
    cqr_offsets_with_audit,
    cqr_policy_contract,
    finalise_cqr_offsets,
    hierarchical_cqr_offsets,
    validate_cqr_offset_bundle,
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
        "y_true": y, "q05": -0.2 * np.ones(n), "q50": np.zeros(n),
        "q95": 0.2 * np.ones(n),
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
        "q50": [2.0, 3.0],
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


def test_negative_signed_cqr_offset_is_audited_and_clipped_to_zero():
    raw = {("site", 1): -0.75, ("site", 3): 0.0, ("site", 7): 0.4}
    deployed, audit = finalise_cqr_offsets(raw)

    assert deployed == {("site", 1): 0.0, ("site", 3): 0.0, ("site", 7): 0.4}
    assert audit["raw_negative_count"] == 1
    assert audit["raw_min"] == -0.75
    assert audit["raw_max"] == 0.4
    assert audit["clipped_count"] == 1
    assert audit["deployed_min"] == 0.0
    validate_cqr_offset_bundle(deployed, cqr_policy_contract(), audit)


def test_overcovered_calibration_can_never_shrink_the_nominal_interval():
    frame = pd.DataFrame({
        "site_id": ["site"] * 20,
        "horizon": [1] * 20,
        "y_true": np.zeros(20),
        "q05": np.full(20, -2.0),
        "q50": np.zeros(20),
        "q95": np.full(20, 2.0),
    })
    offsets, audit = cqr_offsets_with_audit(frame, alpha=0.2)
    calibrated = apply_cqr(frame, offsets)

    assert offsets[("site", 1)] == 0.0
    assert audit["raw_negative_count"] == 1
    assert audit["clipped_count"] == 1
    assert np.array_equal(calibrated.q05, frame.q05)
    assert np.array_equal(calibrated.q50, frame.q50)
    assert np.array_equal(calibrated.q95, frame.q95)


@pytest.mark.parametrize("value", [-0.01, np.nan, np.inf, -np.inf])
def test_apply_cqr_rejects_negative_or_nonfinite_offsets(value):
    frame = pd.DataFrame({
        "site_id": ["site"], "horizon": [1],
        "q05": [1.0], "q50": [2.0], "q95": [3.0],
    })
    with pytest.raises(CQRContractError, match="offset"):
        apply_cqr(frame, {("site", 1): value})


@pytest.mark.parametrize(
    ("q05", "q50", "q95"),
    (
        (2.0, 1.0, 3.0),
        (1.0, 3.0, 2.0),
        (3.0, 2.0, 1.0),
        (2.0, 2.0, 2.0),
    ),
)
def test_apply_cqr_rejects_crossed_or_empty_nominal_intervals(q05, q50, q95):
    frame = pd.DataFrame({
        "site_id": ["site"], "horizon": [1],
        "q05": [q05], "q50": [q50], "q95": [q95],
    })
    with pytest.raises(CQRContractError, match="cross|empty"):
        apply_cqr(frame, {("site", 1): 0.0})


def test_apply_cqr_rejects_an_empty_prediction_registry():
    frame = pd.DataFrame(columns=["site_id", "horizon", "q05", "q50", "q95"])
    with pytest.raises(CQRContractError, match="non-finite|empty"):
        apply_cqr(frame, {("site", 1): 0.0})


@pytest.mark.parametrize("horizon", [1.5, True, 0, -1])
def test_apply_cqr_rejects_noncanonical_horizons(horizon):
    frame = pd.DataFrame({
        "site_id": ["site"], "horizon": [horizon],
        "q05": [1.0], "q50": [2.0], "q95": [3.0],
    })
    with pytest.raises(CQRContractError, match="horizon"):
        apply_cqr(frame, {("site", 1): 0.0})


def test_nonfinite_calibration_evidence_fails_closed_instead_of_dropping_rows():
    frame = pd.DataFrame({
        "site_id": ["site", "site"], "horizon": [1, 1],
        "y_true": [0.0, np.nan], "q05": [-1.0, -1.0],
        "q50": [0.0, 0.0], "q95": [1.0, 1.0],
    })
    with pytest.raises(CQRContractError, match="non-finite"):
        cqr_offsets(frame, alpha=0.5)


def test_hierarchical_apply_rejects_unsafe_unused_registry_offset():
    frame = pd.DataFrame({
        "huc2": ["west"], "horizon": [1],
        "q05": [1.0], "q50": [2.0], "q95": [3.0],
    })
    offsets = {("west", 1): 0.1, ("unused", 1): -0.1, ("__global__", 1): 0.2}
    with pytest.raises(CQRContractError, match="negative"):
        apply_hierarchical_cqr(frame, offsets, "huc2")


def test_cqr_audit_tamper_is_rejected():
    deployed, audit = finalise_cqr_offsets({("site", 1): -0.2})
    attacked = dict(audit)
    attacked["clipped_count"] = 0
    with pytest.raises(CQRContractError, match="tampered"):
        validate_cqr_offset_bundle(deployed, cqr_policy_contract(), attacked)


@pytest.mark.parametrize(
    "key", ("site|01", "site|0", "site|-1", "site|1|extra", "|1", "site|x")
)
def test_cqr_registry_rejects_malformed_or_alias_text_keys(key):
    with pytest.raises(CQRContractError, match="key|horizon|separator"):
        validate_cqr_offset_bundle(
            {key: 0.0},
            cqr_policy_contract(),
            {"raw_signed_offsets": {key: 0.0}},
        )

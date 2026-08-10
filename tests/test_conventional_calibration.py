"""Tests for the frozen CQR + Platt calibration port (frozen_calibration.py).

The module is a verbatim lift of the removed opening.py calibration path;
these tests pin its observable behaviour on synthetic fixtures shaped like
the frozen bundle metadata (see thermoroute_usgs_bundle_665afcb4d674161943ae).
"""

from __future__ import annotations

import numpy as np
import pytest

from thermoroute.frozen_calibration import (
    FrozenCalibrationError,
    apply_frozen_calibration,
)
from thermoroute.conformal import cqr_policy_contract, _build_cqr_offset_audit
from thermoroute.probability import logit

SITES = ("01073319", "01104415")
HORIZONS = (1, 3, 7)


def _audit(offsets):
    # deployed = max(raw, 0); audit must reproduce the signed raw registry.
    raw = {k: (v if v > 0.0 else -0.05) for k, v in offsets.items()}
    return _build_cqr_offset_audit(raw, offsets)


def _offsets(sites=SITES, horizons=HORIZONS, *, pooled=False):
    if pooled:
        return {f"__pooled__|{h}": 0.1 * h for h in horizons}
    return {f"{s}|{h}": 0.1 * h for s in sites for h in horizons}


def _calibrators(horizons=HORIZONS):
    return {
        str(h): {"intercept": -0.012114527849004617, "slope": 1.3643651327511683, "constant": None}
        for h in horizons
    }


def _thresholds(sites=SITES, *, pooled=False):
    if pooled:
        return {"__pooled__": 20.0}
    return {s: 20.0 + i for i, s in enumerate(sites)}


def _metadata(*, sites=SITES, horizons=HORIZONS, pooled=False, **overrides):
    offsets = _offsets(sites, horizons, pooled=pooled)
    meta = {
        "conformal_offsets": offsets,
        "event_calibrators": _calibrators(horizons),
        "event_thresholds": _thresholds(sites, pooled=pooled),
        "conformal_policy": cqr_policy_contract(),
        "conformal_offset_audit": _audit(offsets),
    }
    meta.update(overrides)
    return meta


def _heads(n_sites=2, n_h=3, *, seed=0):
    rng = np.random.default_rng(seed)
    q50 = rng.normal(12.0, 2.0, size=(n_sites, n_h))
    q05 = q50 - np.abs(rng.normal(0.5, 0.2, size=(n_sites, n_h))) - 0.2
    q95 = q50 + np.abs(rng.normal(0.5, 0.2, size=(n_sites, n_h))) + 0.2
    prob = rng.uniform(0.05, 0.95, size=(n_sites, n_h))
    return q05, q50, q95, prob


def test_cqr_delta_applies_to_outer_quantiles_only():
    meta = _metadata()
    q05, q50, q95, prob = _heads()
    out_q05, out_q50, out_q95, _ = apply_frozen_calibration(
        meta, np.asarray(SITES), HORIZONS, q05, q50, q95, prob, external=False, label="t"
    )
    for col, h in enumerate(HORIZONS):
        delta = float(meta["conformal_offsets"][f"{SITES[0]}|{h}"])
        assert np.allclose(out_q05[:, col], q05[:, col] - delta)
        assert np.allclose(out_q95[:, col], q95[:, col] + delta)
        assert np.allclose(out_q50[:, col], q50[:, col])


def test_platt_identity_and_known_mapping():
    # identity: intercept=0, slope=1, constant=None -> probability unchanged
    meta = _metadata(**{
        "event_calibrators": {
            str(h): {"intercept": 0.0, "slope": 1.0, "constant": None}
            for h in HORIZONS
        },
    })
    q05, q50, q95, prob = _heads()
    _, _, _, out_prob = apply_frozen_calibration(
        meta, np.asarray(SITES), HORIZONS, q05, q50, q95, prob, external=False, label="t"
    )
    assert np.allclose(out_prob, prob)
    # known mapping: intercept=-0.012114527849004617, slope=1.3643651327511683
    meta2 = _metadata()
    _, _, _, out2 = apply_frozen_calibration(
        meta2, np.asarray(SITES), HORIZONS, q05, q50, q95, prob, external=False, label="t"
    )
    expected = 1.0 / (1.0 + np.exp(-(1.3643651327511683 * logit(prob) - 0.012114527849004617)))
    assert np.allclose(out2, expected, atol=1e-12)


def test_missing_metadata_raises():
    with pytest.raises(FrozenCalibrationError, match="lacks frozen calibration"):
        apply_frozen_calibration(
            {"event_calibrators": _calibrators()}, np.asarray(SITES), HORIZONS,
            *_heads(), external=False, label="t",
        )


def test_wrong_offset_key_set_raises():
    meta = _metadata()
    meta["conformal_offsets"] = {"01073319|1": 0.1}  # incomplete key set
    with pytest.raises(FrozenCalibrationError, match="offset registry changed"):
        apply_frozen_calibration(
            meta, np.asarray(SITES), HORIZONS, *_heads(), external=False, label="t"
        )


def test_external_routes_to_pooled_offsets_and_threshold():
    meta = _metadata(pooled=True)
    q05, q50, q95, prob = _heads()
    out_q05, _, out_q95, _ = apply_frozen_calibration(
        meta, np.asarray(SITES), HORIZONS, q05, q50, q95, prob, external=True, label="t"
    )
    for col, h in enumerate(HORIZONS):
        delta = float(meta["conformal_offsets"][f"__pooled__|{h}"])
        assert np.allclose(out_q05[:, col], q05[:, col] - delta)
        assert np.allclose(out_q95[:, col], q95[:, col] + delta)
    with pytest.raises(FrozenCalibrationError, match="not pooled"):
        apply_frozen_calibration(
            _metadata(pooled=False), np.asarray(SITES), HORIZONS,
            q05, q50, q95, prob, external=True, label="t",
        )


def test_raw_and_calibrated_families_differ_and_stay_valid():
    meta = _metadata()
    q05, q50, q95, prob = _heads(seed=3)
    out = apply_frozen_calibration(
        meta, np.asarray(SITES), HORIZONS, q05, q50, q95, prob, external=False, label="t"
    )
    out_q05, out_q50, out_q95, out_prob = out
    assert not np.allclose(out_q05, q05)  # offsets were applied
    assert not np.allclose(out_prob, prob)  # Platt was applied
    assert np.isfinite(out_q05).all() and np.isfinite(out_q95).all()
    assert (out_q05 < out_q95).all()
    assert ((0.0 <= out_prob) & (out_prob <= 1.0)).all()

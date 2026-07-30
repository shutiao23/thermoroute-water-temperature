"""Unit covariance of the nondimensional composite loss under °C↔°F.

Exact invariant (documented and asserted):
  For affine T_F = a * T_C + b with a = 1.8, b = 32, set σ_F = a * σ_C and
  transform every temperature tensor (y, point, q*, prior) and the exceedance
  threshold the same way.  Recompute ybin from the transformed threshold.
  Then each nondimensional term from ``composite_loss_terms`` (point, quantile,
  event, crossing, residual, total) and every pairwise ratio of weighted parts
  is identical up to floating-point tolerance.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoroute import config as C
from thermoroute.thermoroute import ThermoRouteOutputs
from thermoroute.train import (
    CELSIUS_TO_FAHRENHEIT_OFFSET,
    CELSIUS_TO_FAHRENHEIT_SCALE,
    composite_loss,
    composite_loss_terms,
    resolve_temperature_loss_scale,
    training_temperature_loss_scale,
)


def _celsius_batch():
    torch.manual_seed(0)
    n, h = 8, 3
    y = torch.tensor(
        [
            [8.0, 9.5, 11.0],
            [12.0, 13.0, 14.5],
            [6.5, 7.0, 7.5],
            [18.0, 19.0, 20.0],
            [10.0, 10.5, 11.5],
            [15.0, 16.0, 17.0],
            [4.0, 5.0, 6.0],
            [21.0, 22.0, 23.5],
        ],
        dtype=torch.float64,
    )
    assert y.shape == (n, h)
    point = y + torch.tensor(
        [[0.4, -0.2, 0.1], [-0.5, 0.3, 0.0], [0.2, 0.2, -0.4],
         [-0.1, 0.6, -0.3], [0.0, -0.5, 0.4], [0.3, -0.1, 0.2],
         [-0.6, 0.1, 0.5], [0.2, -0.4, -0.2]],
        dtype=torch.float64,
    )
    q50 = y + torch.tensor(
        [[0.1, 0.0, -0.1], [-0.2, 0.1, 0.2], [0.0, -0.2, 0.1],
         [0.3, -0.1, 0.0], [-0.1, 0.2, -0.2], [0.2, 0.0, 0.1],
         [-0.3, 0.1, 0.2], [0.0, -0.2, 0.3]],
        dtype=torch.float64,
    )
    # Deliberately crossed quantiles so the crossing term is non-zero under both
    # units and participates in the invariance check.
    q05 = q50 - torch.tensor(
        [[0.8, 0.6, 0.7], [0.5, 0.9, 0.4], [0.7, 0.5, 0.6],
         [0.4, 0.8, 0.5], [0.6, 0.7, 0.9], [0.5, 0.4, 0.8],
         [0.9, 0.6, 0.5], [0.7, 0.8, 0.4]],
        dtype=torch.float64,
    )
    q95 = q50 + torch.tensor(
        [[0.7, 0.5, 0.8], [0.6, 0.4, 0.9], [0.5, 0.7, 0.6],
         [0.9, 0.5, 0.4], [0.4, 0.8, 0.7], [0.8, 0.6, 0.5],
         [0.5, 0.9, 0.7], [0.6, 0.4, 0.8]],
        dtype=torch.float64,
    )
    q05 = q05.clone()
    q05[:, 1] = q50[:, 1] + 0.3  # force q05 > q50 on horizon 1
    q95 = q95.clone()
    q95[:, 2] = q50[:, 2] - 0.2  # force q95 < q50 on horizon 2
    prior = y + torch.tensor(
        [[-0.3, 0.1, 0.2], [0.4, -0.2, -0.1], [-0.1, 0.3, 0.0],
         [0.2, -0.4, 0.1], [-0.2, 0.0, 0.3], [0.1, 0.2, -0.3],
         [0.0, -0.1, 0.4], [-0.4, 0.3, -0.2]],
        dtype=torch.float64,
    )
    prior = prior.clone()
    prior[0, 0] = float("nan")  # exercise the finite-prior residual mask
    exceed_logit = torch.tensor(
        [[-1.2, 0.3, 1.1], [0.4, -0.5, 0.2], [-0.8, 0.9, -0.1],
         [1.5, 0.0, -1.0], [-0.3, 0.7, 0.4], [0.2, -1.1, 0.8],
         [-0.6, 0.1, 1.2], [0.9, -0.4, -0.7]],
        dtype=torch.float64,
    )
    threshold_c = torch.tensor(
        [10.0, 12.0, 7.0, 19.0, 11.0, 16.0, 5.0, 22.0], dtype=torch.float64
    )
    out = ThermoRouteOutputs(
        point=point,
        q05=q05,
        q50=q50,
        q95=q95,
        exceed_logit=exceed_logit,
        prior=prior,
        kappa=torch.zeros(n, dtype=torch.float64),
        teq=torch.zeros(n, dtype=torch.float64),
        lag_weights=torch.zeros(n, h, 1, 1, dtype=torch.float64),
        pi=torch.zeros(n, 1, dtype=torch.float64),
    )
    return out, y, threshold_c


def _to_fahrenheit_temperature(values: torch.Tensor) -> torch.Tensor:
    return CELSIUS_TO_FAHRENHEIT_SCALE * values + CELSIUS_TO_FAHRENHEIT_OFFSET


def test_resolve_temperature_loss_scale_rejects_non_positive():
    with pytest.raises(ValueError, match="temperature_loss_scale"):
        resolve_temperature_loss_scale(replace(C.TRAIN, temperature_loss_scale=0.0))
    with pytest.raises(ValueError, match="temperature_loss_scale"):
        resolve_temperature_loss_scale(replace(C.TRAIN, temperature_loss_scale=-2.0))


def test_default_celsius_reference_matches_unscaled_historical_formula():
    """With σ = 1 °C the nondimensional loss equals the pre-M-04 °C formula."""
    from thermoroute.train import pinball_loss

    out, y, threshold_c = _celsius_batch()
    ybin = (y > threshold_c[:, None]).to(dtype=y.dtype)
    cfg = replace(C.TRAIN, temperature_loss_scale=1.0)
    terms = composite_loss_terms(out, y, ybin, cfg)

    point = torch.mean((y - out.point) ** 2)
    lq = (
        pinball_loss(y, out.q05, 0.05)
        + pinball_loss(y, out.q50, 0.50)
        + pinball_loss(y, out.q95, 0.95)
    )
    cross = (torch.relu(out.q05 - out.q50) + torch.relu(out.q50 - out.q95)).mean()
    evt = torch.nn.functional.binary_cross_entropy_with_logits(out.exceed_logit, ybin)
    finite = torch.isfinite(out.prior)
    resid = (out.point[finite] - out.prior[finite]).abs().mean()
    legacy = (
        point + lq + cfg.lambda_event * evt
        + cfg.lambda_crossing * cross + cfg.lambda_residual * resid
    )
    assert torch.allclose(terms.total, legacy, rtol=0.0, atol=0.0)
    assert torch.allclose(composite_loss(out, y, ybin, cfg), legacy, rtol=0.0, atol=0.0)


def test_celsius_fahrenheit_nondimensional_loss_invariant():
    out_c, y_c, threshold_c = _celsius_batch()
    ybin_c = (y_c > threshold_c[:, None]).to(dtype=y_c.dtype)
    sigma_c = 2.5
    cfg_c = replace(C.TRAIN, temperature_loss_scale=sigma_c)
    terms_c = composite_loss_terms(out_c, y_c, ybin_c, cfg_c)

    a = CELSIUS_TO_FAHRENHEIT_SCALE
    out_f = ThermoRouteOutputs(
        point=_to_fahrenheit_temperature(out_c.point),
        q05=_to_fahrenheit_temperature(out_c.q05),
        q50=_to_fahrenheit_temperature(out_c.q50),
        q95=_to_fahrenheit_temperature(out_c.q95),
        exceed_logit=out_c.exceed_logit.clone(),  # logits are dimensionless
        prior=_to_fahrenheit_temperature(out_c.prior),
        kappa=out_c.kappa,
        teq=_to_fahrenheit_temperature(out_c.teq),
        lag_weights=out_c.lag_weights,
        pi=out_c.pi,
    )
    # Preserve the intentional NaN in prior under the affine map.
    out_f.prior[0, 0] = float("nan")
    y_f = _to_fahrenheit_temperature(y_c)
    threshold_f = _to_fahrenheit_temperature(threshold_c)
    ybin_f = (y_f > threshold_f[:, None]).to(dtype=y_f.dtype)
    assert torch.equal(ybin_f, ybin_c)

    cfg_f = replace(C.TRAIN, temperature_loss_scale=a * sigma_c)
    terms_f = composite_loss_terms(out_f, y_f, ybin_f, cfg_f)

    for name in ("point", "quantile", "event", "crossing", "residual", "total"):
        left = getattr(terms_c, name)
        right = getattr(terms_f, name)
        assert torch.allclose(left, right, rtol=1e-12, atol=1e-12), name

    parts_c = terms_c.as_weighted_parts(cfg_c)
    parts_f = terms_f.as_weighted_parts(cfg_f)
    names = list(parts_c)
    for i, left_name in enumerate(names):
        for right_name in names[i + 1:]:
            num_c = parts_c[left_name]
            den_c = parts_c[right_name]
            num_f = parts_f[left_name]
            den_f = parts_f[right_name]
            if float(den_c.abs()) < 1e-15 or float(den_f.abs()) < 1e-15:
                continue
            ratio_c = num_c / den_c
            ratio_f = num_f / den_f
            assert torch.allclose(ratio_c, ratio_f, rtol=1e-12, atol=1e-12), (
                left_name, right_name
            )


def test_omitting_scale_covariation_breaks_fahrenheit_invariance():
    """Guard: forgetting σ' = aσ reintroduces the M-04 defect."""
    out_c, y_c, threshold_c = _celsius_batch()
    ybin_c = (y_c > threshold_c[:, None]).to(dtype=y_c.dtype)
    cfg = replace(C.TRAIN, temperature_loss_scale=1.0)
    loss_c = composite_loss(out_c, y_c, ybin_c, cfg)

    out_f = ThermoRouteOutputs(
        point=_to_fahrenheit_temperature(out_c.point),
        q05=_to_fahrenheit_temperature(out_c.q05),
        q50=_to_fahrenheit_temperature(out_c.q50),
        q95=_to_fahrenheit_temperature(out_c.q95),
        exceed_logit=out_c.exceed_logit.clone(),
        prior=_to_fahrenheit_temperature(out_c.prior),
        kappa=out_c.kappa,
        teq=out_c.teq,
        lag_weights=out_c.lag_weights,
        pi=out_c.pi,
    )
    out_f.prior[0, 0] = float("nan")
    y_f = _to_fahrenheit_temperature(y_c)
    threshold_f = _to_fahrenheit_temperature(threshold_c)
    ybin_f = (y_f > threshold_f[:, None]).to(dtype=y_f.dtype)
    loss_f_wrong = composite_loss(out_f, y_f, ybin_f, cfg)
    assert not torch.allclose(loss_c, loss_f_wrong, rtol=1e-3, atol=1e-3)


def test_training_temperature_loss_scale_helper_scales_with_fahrenheit():
    class _WD:
        wtemp_t = None

        def idx(self, split: str):
            assert split == "train"
            return __import__("numpy").arange(4)

    wd = _WD()
    wd.wtemp_t = __import__("numpy").asarray([5.0, 10.0, 15.0, 20.0], dtype=float)
    sigma_c = training_temperature_loss_scale(wd)
    wd.wtemp_t = CELSIUS_TO_FAHRENHEIT_SCALE * wd.wtemp_t + CELSIUS_TO_FAHRENHEIT_OFFSET
    sigma_f = training_temperature_loss_scale(wd)
    assert sigma_c == pytest.approx(
        __import__("numpy").std([5.0, 10.0, 15.0, 20.0], ddof=0)
    )
    assert sigma_f == pytest.approx(CELSIUS_TO_FAHRENHEIT_SCALE * sigma_c)

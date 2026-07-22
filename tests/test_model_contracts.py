"""Architectural and safety contracts for the Route-A ThermoRoute design."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from thermoroute import config as C
from thermoroute.baselines import station_equal_sample_weight
from thermoroute.thermoroute import ThermoRoute
from thermoroute.train import LSTMForecaster, pinball_loss


CFG = C.TrainConfig(d_model=8, encoder_blocks=1, kernel_size=2,
                    dropout=0.0, n_experts=2, max_epochs=1,
                    batch_size=4, patience=1)


def _batch(batch_size=4, n_vars=3, horizons=(1, 3)):
    torch.manual_seed(7)
    length = C.MAX_ROUTER_LAG + 2
    return {
        "X": torch.randn(batch_size, length, n_vars),
        "Mask": torch.ones(batch_size, length, n_vars),
        "wtemp_t": torch.full((batch_size,), 10.0),
        "clim_t": torch.full((batch_size,), 9.0),
        "clim_tgt": torch.full((batch_size, len(horizons)), 9.5),
        "damped_prior": torch.full((batch_size, len(horizons)), 10.25),
        "phys_std": torch.randn(batch_size, 2),
        "logflowz": torch.randn(batch_size),
        "wlevelz": torch.randn(batch_size),
        "season": torch.randn(batch_size, 2),
        "gate": torch.randn(batch_size, 6),
        "station": torch.zeros(batch_size, dtype=torch.long),
    }


def _model(**kwargs):
    return ThermoRoute(n_vars=3, n_stations=2, horizons=(1, 3), cfg=CFG,
                       n_phys=2, **kwargs)


def test_strict_forecast_is_bounded_around_fixed_damped_anchor():
    model = _model(safety_anchor="damped", delta_scale=0.4)
    # Force the learned proposal far away.  The guarantee must still reference
    # damped_prior, not this drifting internal prior.
    with torch.no_grad():
        model.prior.eq_lin.bias.fill_(100.0)
        model.head_delta.bias.fill_(50.0)
    out = model(_batch())

    assert torch.equal(out.prior, _batch()["damped_prior"])
    assert out.internal_prior is not None
    assert torch.max(torch.abs(out.internal_prior - out.prior)) > 1.0
    assert torch.all(torch.abs(out.point - out.prior) <= 0.4 + 1e-6)


def test_unbounded_control_can_leave_the_damped_band():
    model = _model(safety_anchor="damped", delta_scale=None)
    with torch.no_grad():
        model.prior.eq_lin.bias.fill_(20.0)
        model.head_delta.bias.fill_(5.0)
    out = model(_batch())
    assert torch.max(torch.abs(out.point - out.prior)) > 0.4


def test_fixed_kappa_removes_all_time_varying_modulators():
    model = _model(fixed_kappa=True)
    batch = _batch()
    batch["logflowz"] = torch.tensor([-10.0, -1.0, 2.0, 20.0])
    batch["wlevelz"] = torch.tensor([8.0, -4.0, 0.0, 3.0])
    batch["season"] = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    _, kappa, _ = model.prior(batch)

    assert torch.count_nonzero(model.prior.k_season.weight) == 0
    assert torch.count_nonzero(model.prior.k_season.bias) == 0
    assert torch.count_nonzero(model.prior.k_flow) == 0
    assert torch.count_nonzero(model.prior.k_level) == 0
    assert torch.allclose(kappa, kappa[0].expand_as(kappa))


def test_route_a_default_makes_wlevel_structurally_inert():
    model = _model()
    assert model.use_wlevel is False
    assert model.prior.k_level.requires_grad is False
    first = _batch()
    second = {key: value.clone() for key, value in first.items()}
    second["wlevelz"] = torch.tensor([100.0, -100.0, 50.0, -50.0])
    with torch.no_grad():
        _, first_kappa, _ = model.prior(first)
        _, second_kappa, _ = model.prior(second)
    assert torch.equal(first_kappa, second_kappa)


def test_clean_ablation_paths_remove_only_the_named_branch():
    no_router = _model(use_router=False, safety_anchor="internal", delta_scale=None)
    assert no_router.router is None
    assert no_router.encoder is not None
    assert no_router.moe is not None

    with torch.no_grad():
        no_router.head_delta.weight.fill_(1.0)
    no_router(_batch()).point.sum().backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad)
               for p in no_router.encoder.parameters())

    no_moe = _model(use_moe=False)
    assert no_moe.moe is None
    assert no_moe.single_expert is not None
    assert no_moe.router is not None and no_moe.encoder is not None

    no_tcn = _model(use_tcn=False)
    assert no_tcn.encoder is None
    assert no_tcn.router is not None and no_tcn.moe is not None


def test_damped_prior_only_has_exact_point_forecast():
    model = _model(use_prior=False, residual_model=False,
                   safety_anchor="damped", delta_scale=0.4)
    batch = _batch()
    out = model(batch)
    assert torch.equal(out.point, batch["damped_prior"])


def test_lstm_matches_context_and_site_identity_contract_by_arm():
    same_station = LSTMForecaster(
        n_vars=3, horizons=(1, 3), d=8, n_stations=2,
        context=C.CONTEXT_LENGTH, station_agnostic=False, station_embed_dim=4)
    held_region = LSTMForecaster(
        n_vars=3, horizons=(1, 3), d=8, n_stations=2,
        context=C.CONTEXT_LENGTH, station_agnostic=True, station_embed_dim=4)
    assert same_station.context == C.CONTEXT_LENGTH
    assert held_region.context == C.CONTEXT_LENGTH
    assert same_station.rnn.input_size == 6
    assert same_station.station_embedding is not None
    assert held_region.station_embedding is None

    batch = _batch(n_vars=3, horizons=(1, 3))
    batch["X"] = torch.randn(4, C.CONTEXT_LENGTH, 3)
    batch["Mask"] = torch.ones_like(batch["X"])
    output = same_station(batch)
    output.point.sum().backward()
    assert output.point.shape == (4, 2)
    assert same_station.station_embedding.weight.grad is not None


def test_fair_lstm_candidate_uses_only_frozen_derived_context_and_damped_anchor():
    model = LSTMForecaster(
        n_vars=3, horizons=(1, 3), d=8, n_stations=2,
        context=C.CONTEXT_LENGTH, station_agnostic=False, station_embed_dim=4,
        use_derived_context=True, anchor="damped",
    )
    # recurrent state (8) + site embedding (4) + clim target (2) + damped
    # target (2) + issue-date sin/cos season (2)
    assert model.head_point.in_features == 18
    assert model.head_q50.in_features == 18
    batch = _batch(n_vars=3, horizons=(1, 3))
    batch["X"] = torch.randn(4, C.CONTEXT_LENGTH, 3)
    batch["Mask"] = torch.ones_like(batch["X"])
    with torch.no_grad():
        model.head_point.weight.zero_()
        model.head_point.bias.zero_()
    output = model(batch)
    assert torch.equal(output.point, batch["damped_prior"])
    assert torch.equal(output.prior, batch["damped_prior"])


def test_point_and_q50_are_independent_and_point_is_not_quantile_sorted():
    model = _model(safety_anchor="damped", delta_scale=None)
    batch = _batch()
    with torch.no_grad():
        model.head_delta.weight.zero_()
        model.head_delta.bias.fill_(20.0)
        model.head_q50.weight.zero_()
        model.head_q50.bias.fill_(-20.0)
    output = model(batch)

    assert torch.all(output.point > output.q95)
    assert torch.all(output.q05 <= output.q50)
    assert torch.all(output.q50 <= output.q95)


def test_q50_pinball_gradient_reaches_only_its_direct_head():
    model = _model(safety_anchor="damped", delta_scale=None)
    output = model(_batch())
    target = output.q50.detach() + 10.0
    pinball_loss(target, output.q50, 0.50).backward()

    assert model.head_q50.bias.grad is not None
    assert torch.count_nonzero(model.head_q50.bias.grad)
    assert model.head_delta is not None
    assert model.head_delta.weight.grad is None
    assert model.head_delta.bias.grad is None


def test_lstm_q50_pinball_does_not_update_point_head():
    model = LSTMForecaster(
        n_vars=3, horizons=(1, 3), d=8, n_stations=2,
        context=C.CONTEXT_LENGTH, station_agnostic=False,
    )
    batch = _batch(n_vars=3, horizons=(1, 3))
    batch["X"] = torch.randn(4, C.CONTEXT_LENGTH, 3)
    batch["Mask"] = torch.ones_like(batch["X"])
    output = model(batch)
    target = output.q50.detach() + 10.0
    pinball_loss(target, output.q50, 0.50).backward()

    assert model.head_q50.bias.grad is not None
    assert torch.count_nonzero(model.head_q50.bias.grad)
    assert model.head_point.weight.grad is None
    assert model.head_point.bias.grad is None


def test_lightgbm_station_weights_equalise_total_training_loss():
    sites = np.array(["a", "a", "a", "b", "c", "c"])
    weights = station_equal_sample_weight(sites)
    totals = {
        site: float(weights[sites == site].sum()) for site in np.unique(sites)
    }
    assert np.mean(weights) == 1.0
    assert len(set(round(value, 12) for value in totals.values())) == 1

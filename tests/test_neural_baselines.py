from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from thermoroute.neural_baselines import (
    PlainCausalTCNForecaster,
    PlainMLPForecaster,
)


BATCH_SIZE = 4
CONTEXT = 12
N_VARS = 5
N_STATIONS = 7
HORIZONS = (1, 3, 7)


def _batch(*, station: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260722)
    X = torch.randn(BATCH_SIZE, CONTEXT, N_VARS, generator=generator)
    mask = (torch.rand(BATCH_SIZE, CONTEXT, N_VARS, generator=generator) > 0.2).float()
    result = {
        "X": X,
        "Mask": mask,
        # Every field below is available in WindowedData.batch but must not be
        # an undeclared input to either pure-neural control.
        "y": torch.randn(BATCH_SIZE, len(HORIZONS), generator=generator),
        "clim_tgt": torch.randn(BATCH_SIZE, len(HORIZONS), generator=generator),
        "damped_prior": torch.randn(BATCH_SIZE, len(HORIZONS), generator=generator),
        "wtemp_t": torch.randn(BATCH_SIZE, generator=generator),
    }
    result["station"] = (
        station
        if station is not None
        else torch.arange(BATCH_SIZE, dtype=torch.long) % N_STATIONS
    )
    return result


def _mlp(*, station_agnostic: bool, init_seed: int = 17) -> PlainMLPForecaster:
    return PlainMLPForecaster(
        n_vars=N_VARS,
        context_length=CONTEXT,
        horizons=HORIZONS,
        n_stations=N_STATIONS,
        station_agnostic=station_agnostic,
        init_seed=init_seed,
        hidden_dim=24,
        depth=2,
        dropout=0.1,
    )


def _tcn(*, station_agnostic: bool, init_seed: int = 17) -> PlainCausalTCNForecaster:
    return PlainCausalTCNForecaster(
        n_vars=N_VARS,
        context_length=CONTEXT,
        horizons=HORIZONS,
        n_stations=N_STATIONS,
        station_agnostic=station_agnostic,
        init_seed=init_seed,
        channels=16,
        blocks=3,
        kernel_size=3,
        dropout=0.1,
    )


MODEL_FACTORIES: tuple[Callable[..., torch.nn.Module], ...] = (_mlp, _tcn)


@pytest.mark.parametrize("factory", MODEL_FACTORIES)
@pytest.mark.parametrize("station_agnostic", [False, True])
def test_outputs_have_finite_monotone_quantiles(factory, station_agnostic: bool) -> None:
    model = factory(station_agnostic=station_agnostic).eval()
    with torch.no_grad():
        output = model(_batch())

    expected = (BATCH_SIZE, len(HORIZONS))
    assert output.point.shape == expected
    assert output.q_lo.shape == expected
    assert output.q_med.shape == expected
    assert output.q_hi.shape == expected
    assert output.event_logit.shape == expected
    assert torch.isfinite(output.q_lo).all()
    assert torch.isfinite(output.q_med).all()
    assert torch.isfinite(output.q_hi).all()
    assert torch.isfinite(output.event_logit).all()
    assert torch.all(output.q_lo < output.q_med)
    assert torch.all(output.q_med < output.q_hi)

    # Existing fit/export code can consume the controls without relabelling a
    # neural output as a physical prior.
    assert output.q05 is output.q_lo
    assert output.q50 is output.q_med
    assert output.q95 is output.q_hi
    assert output.exceed_logit is output.event_logit
    assert torch.isnan(output.prior).all()
    assert output.lag_weights.shape == (BATCH_SIZE, len(HORIZONS), 0, 0)
    assert output.pi.shape == (BATCH_SIZE, 0)


@pytest.mark.parametrize("factory", MODEL_FACTORIES)
def test_future_and_target_fields_do_not_affect_predictions(factory) -> None:
    model = factory(station_agnostic=False).eval()
    first = _batch()
    second = {key: value.clone() for key, value in first.items()}
    second["y"].fill_(float("nan"))
    second["clim_tgt"].fill_(1e20)
    second["damped_prior"].fill_(-1e20)
    second["wtemp_t"].fill_(123456.0)
    second["unknown_future_feature"] = torch.randn(BATCH_SIZE, 99)

    with torch.no_grad():
        out_first = model(first)
        out_second = model(second)
    for name in ("point", "q_lo", "q_med", "q_hi", "event_logit"):
        torch.testing.assert_close(
            getattr(out_first, name), getattr(out_second, name), rtol=0.0, atol=0.0
        )


@pytest.mark.parametrize("factory", MODEL_FACTORIES)
def test_station_agnostic_mode_is_invariant_to_station_identity(factory) -> None:
    model = factory(station_agnostic=True).eval()
    first = _batch(station=torch.tensor([0, 1, 2, 3], dtype=torch.long))
    second = {key: value.clone() for key, value in first.items()}
    second["station"] = torch.tensor([6, 5, 4, 3], dtype=torch.long)

    with torch.no_grad():
        out_first = model(first)
        out_second = model(second)
    torch.testing.assert_close(out_first.q_med, out_second.q_med, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        out_first.event_logit, out_second.event_logit, rtol=0.0, atol=0.0
    )
    assert model.station_embedding is None
    assert model.architecture_metadata()["input_keys_read"] == ("X", "Mask")


def test_tcn_hidden_prefix_is_invariant_to_any_suffix_change() -> None:
    model = _tcn(station_agnostic=True).eval()
    first = _batch()
    second_X = first["X"].clone()
    second_mask = first["Mask"].clone()
    prefix_length = 7
    second_X[:, prefix_length:, :] = 1000.0 * torch.randn_like(
        second_X[:, prefix_length:, :]
    )
    second_mask[:, prefix_length:, :] = 1.0 - second_mask[:, prefix_length:, :]

    with torch.no_grad():
        hidden_first = model.encode_sequence(first["X"], first["Mask"])
        hidden_second = model.encode_sequence(second_X, second_mask)
    torch.testing.assert_close(
        hidden_first[:, :prefix_length, :],
        hidden_second[:, :prefix_length, :],
        # CPU convolution backends may change last-bit accumulation order when
        # unrelated suffix columns change; the causal values remain equal to
        # float32 numerical precision.
        rtol=1e-6,
        atol=1e-6,
    )
    assert not torch.equal(hidden_first[:, prefix_length:, :], hidden_second[:, prefix_length:, :])


@pytest.mark.parametrize("factory", MODEL_FACTORIES)
def test_history_schema_is_strictly_validated(factory) -> None:
    model = factory(station_agnostic=True).eval()
    valid = _batch()

    with pytest.raises(KeyError, match="Mask"):
        model({"X": valid["X"]})

    wrong_context = dict(valid)
    wrong_context["X"] = valid["X"][:, :-1, :]
    wrong_context["Mask"] = valid["Mask"][:, :-1, :]
    with pytest.raises(ValueError, match="must have shape"):
        model(wrong_context)

    wrong_mask_shape = dict(valid)
    wrong_mask_shape["Mask"] = valid["Mask"][:, :, :-1]
    with pytest.raises(ValueError, match="exactly the same shape"):
        model(wrong_mask_shape)

    nonbinary_mask = {key: value.clone() for key, value in valid.items()}
    nonbinary_mask["Mask"][0, 0, 0] = 0.5
    with pytest.raises(ValueError, match="must be binary"):
        model(nonbinary_mask)

    nonfinite_history = {key: value.clone() for key, value in valid.items()}
    nonfinite_history["X"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        model(nonfinite_history)


@pytest.mark.parametrize("factory", MODEL_FACTORIES)
def test_station_aware_schema_rejects_invalid_station_tensors(factory) -> None:
    model = factory(station_agnostic=False).eval()
    valid = _batch()

    missing = {key: value for key, value in valid.items() if key != "station"}
    with pytest.raises(KeyError, match="station"):
        model(missing)

    wrong_dtype = dict(valid)
    wrong_dtype["station"] = valid["station"].float()
    with pytest.raises(TypeError, match="torch.long"):
        model(wrong_dtype)

    out_of_range = {key: value.clone() for key, value in valid.items()}
    out_of_range["station"][0] = N_STATIONS
    with pytest.raises(ValueError, match="must lie"):
        model(out_of_range)


@pytest.mark.parametrize("factory", MODEL_FACTORIES)
def test_init_seed_is_deterministic_and_metadata_reconstructs(factory) -> None:
    torch.manual_seed(111)
    first = factory(station_agnostic=False, init_seed=29).eval()
    torch.manual_seed(999999)
    second = factory(station_agnostic=False, init_seed=29).eval()

    first_state = first.state_dict()
    second_state = second.state_dict()
    assert first_state.keys() == second_state.keys()
    for name in first_state:
        torch.testing.assert_close(first_state[name], second_state[name], rtol=0.0, atol=0.0)

    metadata = first.architecture_metadata()
    rebuilt = type(first)(**metadata["constructor_kwargs"]).eval()
    for name, value in first_state.items():
        torch.testing.assert_close(value, rebuilt.state_dict()[name], rtol=0.0, atol=0.0)
    assert metadata["trainable_parameters"] == first.n_params()
    assert metadata["format_version"] == 2
    assert metadata["q50_is_independent_from_point"] is True
    assert metadata["output_keys"] == (
        "point", "q_lo", "q_med", "q_hi", "event_logit"
    )
    assert "externally" in str(metadata["budget_matching_note"])
    assert metadata["future_keys_never_read"] == (
        "y",
        "clim_tgt",
        "damped_prior",
        "target_date",
    )

    different = factory(station_agnostic=False, init_seed=30)
    assert any(
        not torch.equal(value, different.state_dict()[name])
        for name, value in first_state.items()
        if value.is_floating_point()
    )


def test_default_models_are_small_and_parameter_budget_friendly() -> None:
    common = {
        "n_vars": 7,
        "context_length": 32,
        "horizons": (1, 3, 7, 14),
        "station_agnostic": True,
        "init_seed": 0,
    }
    mlp = PlainMLPForecaster(**common)
    tcn = PlainCausalTCNForecaster(**common)
    counts = (mlp.n_params(), tcn.n_params())
    assert max(counts) < 100_000
    assert max(counts) / min(counts) < 2.0


@pytest.mark.parametrize("factory", MODEL_FACTORIES)
def test_plain_point_and_q50_have_disjoint_direct_head_gradients(factory) -> None:
    model = factory(station_agnostic=True)
    output = model(_batch())
    assert output.point.data_ptr() != output.q50.data_ptr()
    # The final linear packs [point, q50, lo, hi, event] per horizon.  A q50-only
    # loss must not touch the point rows of that head.
    target = output.q50.detach() + 5.0
    loss = torch.maximum(0.5 * (target - output.q50), -0.5 * (target - output.q50)).mean()
    loss.backward()
    gradient = model.head.weight.grad.reshape(len(HORIZONS), 5, -1)
    assert torch.count_nonzero(gradient[:, 1, :])
    assert torch.count_nonzero(gradient[:, 0, :]) == 0


def test_constructor_rejects_ambiguous_or_invalid_architecture_schema() -> None:
    common = {
        "n_vars": N_VARS,
        "context_length": CONTEXT,
        "horizons": HORIZONS,
        "init_seed": 0,
    }
    with pytest.raises(ValueError, match="station_agnostic"):
        PlainMLPForecaster(**common, station_agnostic=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="strictly increasing"):
        PlainMLPForecaster(
            **{**common, "horizons": (1, 7, 3)},
            station_agnostic=True,
        )
    with pytest.raises(ValueError, match="kernel_size"):
        PlainCausalTCNForecaster(
            **common,
            station_agnostic=True,
            kernel_size=1,
        )

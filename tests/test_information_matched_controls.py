from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from thermoroute import development_controls as DC
from thermoroute.neural_baselines import (
    INFORMATION_MATCHED_EXCLUDED_KEYS,
    INFORMATION_MATCHED_INPUT_KEYS,
    PlainCausalTCNForecaster,
    PlainMLPForecaster,
)


BATCH_SIZE = 3
CONTEXT = 8
N_VARS = 7
N_STATIONS = 6
HORIZONS = (1, 3, 7)
N_PHYS = 4
GATE_DIM = 6


def _batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260725)
    return {
        "X": torch.randn(
            BATCH_SIZE, CONTEXT, N_VARS, generator=generator, dtype=torch.float32
        ),
        "Mask": torch.ones(BATCH_SIZE, CONTEXT, N_VARS, dtype=torch.float32),
        "station": torch.tensor([0, 1, 2], dtype=torch.long),
        "wtemp_t": torch.randn(BATCH_SIZE, generator=generator),
        "clim_t": torch.randn(BATCH_SIZE, generator=generator),
        "clim_tgt": torch.randn(BATCH_SIZE, len(HORIZONS), generator=generator),
        "damped_prior": torch.randn(
            BATCH_SIZE, len(HORIZONS), generator=generator
        ),
        "phys_std": torch.randn(BATCH_SIZE, N_PHYS, generator=generator),
        "logflowz": torch.randn(BATCH_SIZE, generator=generator),
        "season": torch.randn(BATCH_SIZE, 2, generator=generator),
        "gate": torch.randn(BATCH_SIZE, GATE_DIM, generator=generator),
        # These coexist in WindowedData.batch but are excluded from the model.
        "y": torch.randn(BATCH_SIZE, len(HORIZONS), generator=generator),
        "target_date": torch.arange(BATCH_SIZE * len(HORIZONS)).reshape(
            BATCH_SIZE, len(HORIZONS)
        ),
        "wlevelz": torch.randn(BATCH_SIZE, generator=generator),
    }


def _mlp() -> PlainMLPForecaster:
    return PlainMLPForecaster(
        n_vars=N_VARS,
        context_length=CONTEXT,
        horizons=HORIZONS,
        n_stations=N_STATIONS,
        station_agnostic=False,
        init_seed=13,
        hidden_dim=24,
        depth=2,
        dropout=0.0,
        use_information_matched_context=True,
        n_phys=N_PHYS,
        gate_dim=GATE_DIM,
    )


def _tcn() -> PlainCausalTCNForecaster:
    return PlainCausalTCNForecaster(
        n_vars=N_VARS,
        context_length=CONTEXT,
        horizons=HORIZONS,
        n_stations=N_STATIONS,
        station_agnostic=False,
        init_seed=13,
        channels=12,
        blocks=2,
        kernel_size=3,
        dropout=0.0,
        use_information_matched_context=True,
        n_phys=N_PHYS,
        gate_dim=GATE_DIM,
    )


MODEL_FACTORIES: tuple[Callable[[], torch.nn.Module], ...] = (_mlp, _tcn)


def _clone_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in batch.items()}


@pytest.mark.parametrize("factory", MODEL_FACTORIES)
def test_information_matched_metadata_is_exact_and_noncausal(factory) -> None:
    model = factory()
    metadata = model.architecture_metadata()

    assert metadata["format_version"] == 3
    assert metadata["input_keys_read"] == INFORMATION_MATCHED_INPUT_KEYS
    assert metadata["input_keys_explicitly_excluded"] == (
        INFORMATION_MATCHED_EXCLUDED_KEYS
    )
    assert metadata["common_anchor"] == "damped_prior"
    assert metadata["forecast_parameterization"] == (
        "damped_prior_plus_unrestricted_additive_neural_residual"
    )
    assert metadata["uses_learned_physics_prior"] is False
    assert metadata["uses_dynamic_lag_router"] is False
    assert metadata["uses_mixture_of_experts"] is False
    assert metadata["uses_bounded_residual"] is False
    assert metadata["causal_interpretation_allowed"] is False
    assert "does not identify a causal effect" in str(metadata["noncausal_boundary"])


@pytest.mark.parametrize("factory", MODEL_FACTORIES)
@pytest.mark.parametrize("key", INFORMATION_MATCHED_INPUT_KEYS)
def test_every_declared_information_field_enters_the_prediction(factory, key: str) -> None:
    model = factory().eval()
    original = _batch()
    perturbed = _clone_batch(original)
    if key == "X":
        perturbed[key][0, -1, 0] += 9.0
    elif key == "Mask":
        perturbed[key][0, -1, 0] = 0.0
    elif key == "station":
        perturbed[key][0] = 4
    elif perturbed[key].ndim == 1:
        perturbed[key][0] += 9.0
    else:
        perturbed[key][0, 0] += 9.0

    with torch.no_grad():
        before = model(original).point
        after = model(perturbed).point
    assert not torch.equal(before, after), f"{key} did not enter the model computation"


@pytest.mark.parametrize("factory", MODEL_FACTORIES)
def test_excluded_outcome_date_and_level_fields_never_affect_outputs(factory) -> None:
    model = factory().eval()
    original = _batch()
    perturbed = _clone_batch(original)
    perturbed["y"].fill_(float("nan"))
    perturbed["target_date"].fill_(2**31 - 1)
    perturbed["wlevelz"].fill_(float("inf"))

    with torch.no_grad():
        before = model(original)
        after = model(perturbed)
    for name in ("point", "q_lo", "q_med", "q_hi", "event_logit", "prior"):
        torch.testing.assert_close(
            getattr(before, name), getattr(after, name), rtol=0.0, atol=0.0
        )


@pytest.mark.parametrize("factory", MODEL_FACTORIES)
def test_damped_anchor_and_residual_are_exact_and_unrestricted(factory) -> None:
    model = factory().eval()
    batch = _batch()
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        model.head.bias[0::5] = 10_000.0
        model.head.bias[1::5] = -20_000.0
        output = model(batch)

    torch.testing.assert_close(output.prior, batch["damped_prior"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        output.point - batch["damped_prior"],
        torch.full_like(batch["damped_prior"], 10_000.0),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        output.q_med - batch["damped_prior"],
        torch.full_like(batch["damped_prior"], -20_000.0),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("factory", MODEL_FACTORIES)
@pytest.mark.parametrize("key", INFORMATION_MATCHED_INPUT_KEYS)
def test_missing_information_field_fails_closed(factory, key: str) -> None:
    model = factory().eval()
    batch = _batch()
    del batch[key]
    with pytest.raises(KeyError):
        model(batch)


@pytest.mark.parametrize("factory", MODEL_FACTORIES)
@pytest.mark.parametrize("key", INFORMATION_MATCHED_INPUT_KEYS)
def test_wrong_information_shape_fails_closed(factory, key: str) -> None:
    model = factory().eval()
    batch = _batch()
    value = batch[key]
    if key in {"X", "Mask"}:
        batch[key] = value[:, :-1, :]
    elif key == "station" or value.ndim == 1:
        batch[key] = value[:, None]
    else:
        batch[key] = value[:, :-1]
    with pytest.raises(ValueError, match="shape"):
        model(batch)


FLOAT_INPUT_KEYS = tuple(
    key for key in INFORMATION_MATCHED_INPUT_KEYS if key != "station"
)


@pytest.mark.parametrize("factory", MODEL_FACTORIES)
@pytest.mark.parametrize("key", FLOAT_INPUT_KEYS)
def test_nonfinite_information_field_fails_closed(factory, key: str) -> None:
    model = factory().eval()
    batch = _batch()
    batch[key].reshape(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(batch)


@pytest.mark.parametrize("factory", MODEL_FACTORIES)
@pytest.mark.parametrize("key", INFORMATION_MATCHED_INPUT_KEYS)
def test_wrong_information_dtype_fails_closed(factory, key: str) -> None:
    model = factory().eval()
    batch = _batch()
    if key == "station":
        batch[key] = batch[key].to(torch.float32)
    elif key == "Mask":
        batch[key] = batch[key].to(torch.int64)
    else:
        batch[key] = batch[key].to(torch.float64)
    with pytest.raises((TypeError, ValueError), match="dtype|floating"):
        model(batch)


@pytest.mark.parametrize("factory", MODEL_FACTORIES)
@pytest.mark.parametrize("key", INFORMATION_MATCHED_INPUT_KEYS)
def test_wrong_information_device_fails_closed(factory, key: str) -> None:
    model = factory().eval()
    batch = _batch()
    batch[key] = torch.empty_like(batch[key], device="meta")
    with pytest.raises(ValueError, match="device"):
        model(batch)


def test_stage09b_explicitly_enables_matched_mode_and_freezes_new_budget() -> None:
    controls = DC.declared_arms()[:2]
    counts = DC.assert_parameter_budgets(DC.declared_arms(), n_stations=120)
    assert counts["PlainMLP-7var"] == DC.MLP_EXPECTED_PARAMETERS == 38_860
    assert counts["PlainCausalTCN-7var"] == DC.TCN_EXPECTED_PARAMETERS == 38_346
    for arm in controls:
        count = counts[arm.arm_id]
        assert abs(count / DC.THERMOROUTE_REFERENCE_PARAMETERS - 1.0) <= 0.02
        model = DC.build_arm_model(arm, seed=0, n_stations=120)
        assert model.use_information_matched_context is True
        assert model.architecture_metadata()["input_keys_read"] == (
            INFORMATION_MATCHED_INPUT_KEYS
        )

    budget = DC.architecture_budget_rows(
        DC.declared_arms(), n_stations=120, train_examples=3_073
    ).set_index("arm_id")
    for arm in controls:
        row = budget.loc[arm.arm_id]
        assert bool(row["information_matched_context"])
        assert row["model_input_keys_read"] == ",".join(
            INFORMATION_MATCHED_INPUT_KEYS
        )
        assert row["model_input_keys_explicitly_excluded"] == ",".join(
            INFORMATION_MATCHED_EXCLUDED_KEYS
        )
        assert row["common_forecast_anchor"] == "damped_prior"
        assert row["residual_parameterization"] == (
            "unrestricted_additive_neural_residual"
        )
        assert not bool(row["uses_learned_physics_prior"])
        assert not bool(row["uses_dynamic_lag_router"])
        assert not bool(row["uses_mixture_of_experts"])


@pytest.mark.parametrize("factory", (_mlp, _tcn))
def test_generic_station_agnostic_use_of_matched_schema_is_rejected(factory) -> None:
    kwargs = factory().architecture_kwargs()
    kwargs["station_agnostic"] = True
    with pytest.raises(ValueError, match="station"):
        type(factory())(**kwargs)

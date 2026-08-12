"""Tests for the plain causal TCN arm (Phase 5).

The architecture contrast is only meaningful if the network is fed the same
information as the trees and if the word "causal" in its name is true.  These
tests cover exactly that, plus the algebraic bound the manuscript's constrained
model relies on.  They are unit tests over the layout and the module -- the
fitted cells live in shards and are checked by the authority builder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.final.run_forcing_ladder_v5_observed as V5
import scripts.final.run_plain_tcn_arm as T
from thermoroute.information_levels import admissible_columns


# ------------------------------------------------------- information matching


def test_the_layout_partitions_the_namespace_exactly() -> None:
    """Every admissible column lands in exactly one branch: none lost, none invented."""
    columns = V5.FROZEN_BASE_FEATURE_COLUMNS
    variables, lags, static = T.sequence_layout(columns)
    reconstructed = [f"{v}_lag{lag}" for v in variables for lag in lags] + static
    assert sorted(reconstructed) == sorted(columns)
    assert len(reconstructed) == len(set(reconstructed))


def test_the_masked_namespace_also_partitions_exactly() -> None:
    """The same must hold at every level, or the mask leaks through the reshape."""
    for level_name in ("L0", "L1", "L2", "L2_U2"):
        columns = admissible_columns(level_name, V5.FROZEN_BASE_FEATURE_COLUMNS)
        variables, lags, static = T.sequence_layout(columns)
        rebuilt = [f"{v}_lag{lag}" for v in variables for lag in lags] + static
        assert sorted(rebuilt) == sorted(columns), level_name


def test_a_level_that_hides_water_temperature_keeps_it_out_of_the_sequence() -> None:
    """L2 forbids local water temperature; it must not survive as a channel."""
    columns = admissible_columns("L2", V5.FROZEN_BASE_FEATURE_COLUMNS)
    variables, _, static = T.sequence_layout(columns)
    assert "WTEMP" not in variables
    assert not any(c.startswith("WTEMP_lag") for c in static)


def test_the_lag_axis_runs_oldest_to_newest() -> None:
    """A causal convolution over a reversed axis would look backwards in time."""
    _, lags, _ = T.sequence_layout(V5.FROZEN_BASE_FEATURE_COLUMNS)
    assert lags == sorted(lags, reverse=True)
    assert lags[-1] == 0, "the newest position must be lag 0"


# -------------------------------------------------------------------- causality


def test_the_convolution_cannot_see_the_future() -> None:
    """Perturbing a *newer* position must not move an older output position.

    This is the property the name claims.  It is checked on the trunk output
    rather than on ``forward``, because ``forward`` reads only the newest
    position and would pass trivially.
    """
    torch.manual_seed(0)
    model = T.PlainCausalTCN(3, 4, bounded=False).eval()
    base = torch.randn(2, 3, 8)
    with torch.no_grad():
        before = model.convolve(base)
        moved = base.clone()
        moved[:, :, -1] += 10.0            # perturb the newest position only
        after = model.convolve(moved)
    assert torch.allclose(before[:, :, :-1], after[:, :, :-1]), (
        "an older output position moved when a newer input did"
    )
    assert not torch.allclose(before[:, :, -1], after[:, :, -1])


def test_an_older_position_does_reach_the_readout() -> None:
    """The mirror check: causality must not be achieved by ignoring history."""
    torch.manual_seed(0)
    model = T.PlainCausalTCN(3, 4, bounded=False).eval()
    base = torch.randn(2, 3, 8)
    moved = base.clone()
    moved[:, :, 0] += 10.0
    with torch.no_grad():
        assert not torch.allclose(model.convolve(base)[:, :, -1],
                                  model.convolve(moved)[:, :, -1])


def test_the_receptive_field_covers_the_whole_lag_grid() -> None:
    """A stack blind to the oldest lag would be an information mismatch.

    The trees can split on ``{VAR}_lag14``; a convolution whose receptive field
    stops short of it cannot, and the architecture contrast would then be
    partly a contrast of available history.
    """
    _, lags, _ = T.sequence_layout(V5.FROZEN_BASE_FEATURE_COLUMNS)
    reach = 1 + (T.KERNEL - 1) * sum(T.DILATIONS)
    assert reach >= len(lags), (
        f"receptive field {reach} leaves part of the {len(lags)}-position "
        "history unreachable"
    )


# ------------------------------------------------------------------ the bound


def test_the_bounded_variant_cannot_exceed_one_degree() -> None:
    """Algebraically, not empirically: extreme inputs must still be clamped."""
    torch.manual_seed(0)
    model = T.PlainCausalTCN(3, 4, bounded=True).eval()
    with torch.no_grad():
        out = model(torch.full((16, 3, 8), 1e4), torch.full((16, 4), 1e4))
    assert torch.all(out.abs() <= T.BOUND_DEGC + 1e-6)


def test_the_unbounded_variant_is_genuinely_unbounded() -> None:
    """Otherwise the two variants would measure the same thing."""
    torch.manual_seed(0)
    model = T.PlainCausalTCN(3, 4, bounded=False).eval()
    with torch.no_grad():
        out = model(torch.full((16, 3, 8), 1e4), torch.full((16, 4), 1e4))
    assert out.abs().max() > T.BOUND_DEGC


# ---------------------------------------------------------------- tensorising


def _frame(rows: int = 32) -> tuple[pd.DataFrame, list[str], list[int], list[str]]:
    columns = [f"{v}_lag{lag}" for v in ("A", "B") for lag in (2, 1, 0)] + ["S"]
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(rows, len(columns))), columns=columns)
    variables, lags, static = T.sequence_layout(columns)
    return frame, variables, lags, static


def test_evaluation_rows_are_standardised_with_training_statistics() -> None:
    """Refitting the scaler on held rows would leak their distribution."""
    frame, variables, lags, static = _frame()
    _, _, stats, _ = T.to_tensors(frame, variables, lags, static)
    shifted = frame + 100.0
    seq, flat, _, _ = T.to_tensors(shifted, variables, lags, static, stats)
    assert float(seq.mean()) > 10.0, "held rows were re-centred on themselves"
    assert float(flat.mean()) > 10.0


def test_missing_cells_become_the_training_mean_and_are_counted() -> None:
    frame, variables, lags, static = _frame()
    frame.iloc[0, 0] = np.nan
    seq, flat, _, missing = T.to_tensors(frame, variables, lags, static)
    assert torch.isfinite(seq).all() and torch.isfinite(flat).all()
    assert missing == pytest.approx(1 / (seq.numel() + flat.numel()))


def test_the_sequence_axis_holds_the_column_it_claims_to() -> None:
    """Guards against a transposed stack silently mixing variables and lags."""
    frame, variables, lags, static = _frame()
    seq, _, _, _ = T.to_tensors(frame, variables, lags, static,
                                stats={"seq_mean": 0.0, "seq_std": 1.0,
                                       "flat_mean": 0.0, "flat_std": 1.0})
    for v_index, variable in enumerate(variables):
        for l_index, lag in enumerate(lags):
            assert seq[:, v_index, l_index].numpy() == pytest.approx(
                frame[f"{variable}_lag{lag}"].to_numpy(np.float32), rel=1e-6
            )

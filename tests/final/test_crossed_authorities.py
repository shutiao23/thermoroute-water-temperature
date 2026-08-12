"""Tests for the two crossing authorities: F2a recovery and the F-by-L interaction.

Both report a quantity built from other quantities, which is where this project
has previously gone wrong: a ratio taken against the wrong denominator, or an
interaction formed by differencing two medians instead of medianing a
difference. These tests pin the estimand and the denominator rather than the
values, so they survive a rerun and fail a redefinition.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.final.build_f2a_recovery_authority as R
import scripts.final.build_forcing_information_interaction as I


# ------------------------------------------------------------ F2a denominator


def test_the_denominator_is_the_temperature_only_arm() -> None:
    """F3_full would charge a temperature product for four variables it never
    claimed to supply, which is arithmetic rather than measurement."""
    assert R.DENOMINATOR_ARM == "only_air_temperature"


def test_the_artifact_forbids_the_two_readings_that_would_overstate_it() -> None:
    summary = R.DEFAULT_OUT / "f2a_recovery_summary.json"
    if not summary.exists():
        pytest.skip("recovery authority not built in this tree")
    document = json.loads(summary.read_text(encoding="utf-8"))
    forbidden = set(document["forbidden"])
    assert "recovery against F3_full" in forbidden
    assert "operational recovery fraction" in forbidden
    assert "a confidence interval on the ratio" in forbidden
    assert document["semantics"]["coherent_single_initialization"] is False
    assert document["semantics"]["as_issued_operational_archive"] is False


def test_the_ratio_is_reported_without_an_interval() -> None:
    """A quotient of two estimated medians has no interval here; each of its two
    differences does, and those are what a reader should use."""
    frame_path = R.DEFAULT_OUT / "f2a_recovery_values.parquet"
    if not frame_path.exists():
        pytest.skip("recovery authority not built in this tree")
    columns = set(pd.read_parquet(frame_path).columns)
    assert {"oracle_ci_low", "oracle_ci_high",
            "forecast_ci_low", "forecast_ci_high"} <= columns
    assert not {"recovery_ci_low", "recovery_ci_high"} & columns


def test_the_ratio_is_the_two_reported_differences() -> None:
    frame_path = R.DEFAULT_OUT / "f2a_recovery_values.parquet"
    if not frame_path.exists():
        pytest.skip("recovery authority not built in this tree")
    table = pd.read_parquet(frame_path)
    assert (table["recovery_fraction"].to_numpy()
            == pytest.approx((table["forecast_value_degC"]
                              / table["oracle_value_degC"]).to_numpy()))


def test_a_gain_is_oriented_so_that_larger_is_better() -> None:
    """The bootstrap is written for candidate-minus-reference; a sign slip here
    would report the interval mirrored about zero."""
    index = pd.Index([f"s{i:02d}" for i in range(20)])
    baseline = pd.Series(np.full(20, 1.0), index=index)
    arm = pd.Series(np.full(20, 0.4), index=index)
    clusters = {site: ("A" if i < 10 else "B") for i, site in enumerate(index)}
    out = R._paired(baseline, arm, index, clusters)
    assert out["value_degC"] == pytest.approx(0.6)
    assert out["ci_low"] <= 0.6 <= out["ci_high"]
    assert out["station_fraction_positive"] == 1.0


# ------------------------------------------------------- the interaction shape


def _fake_risk_table(values: dict[tuple[str, str], np.ndarray],
                     index: pd.Index):
    def risk(level: str, forcing: str, model: str, horizon: int) -> pd.Series:
        return pd.Series(values[level, forcing], index=index)
    return risk


def test_the_interaction_is_a_median_of_differences_not_a_difference_of_medians(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two coincide only when the medians happen to move together, and the
    whole point of a paired estimand is that they need not.

    Here two of twenty stations gain a great deal at L2 and the other eighteen
    gain nothing extra. That is enough to drag the marginal median of the L2
    gain from 0 to 10 while the *typical station* is unaffected, so a difference
    of medians reports +10 and the paired quantity reports 0. The paired one is
    the honest answer to "does the ungauged case change what forcing is worth".
    """
    index = pd.Index([f"s{i:02d}" for i in range(20)])
    gain_l0 = np.array([0.0] * 11 + [10.0] * 9)
    gain_l2 = np.array([0.0] * 9 + [20.0] * 2 + [10.0] * 9)
    zeros = np.zeros(20)
    values = {
        ("L0", "F0"): gain_l0, ("L0", "F3_full"): zeros,
        ("L2", "F0"): gain_l2, ("L2", "F3_full"): zeros,
    }
    monkeypatch.setattr(I, "risk", _fake_risk_table(values, index))
    monkeypatch.setattr(I.L.V5, "MODELS", ["LightGBM"])
    monkeypatch.setattr(I.L.V5, "HORIZONS", [7])
    monkeypatch.setattr(
        I, "huc2_cluster_map",
        lambda registry: {s: ("A" if i < 10 else "B") for i, s in enumerate(index)})
    monkeypatch.setattr(I, "load_station_registry", lambda path: None)

    rows = {row["quantity"]: row for row in I.build_rows()}
    difference_of_medians = (rows["forcing_value_at_L2"]["median_degC"]
                             - rows["forcing_value_at_L0"]["median_degC"])
    assert difference_of_medians == pytest.approx(10.0)
    # the reported interaction is the paired quantity, which disagrees
    assert rows["interaction_L2_minus_L0"]["median_degC"] == pytest.approx(0.0)


def test_every_level_is_scored_on_the_same_station_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A double difference over two different station sets is not one."""
    wide = pd.Index([f"s{i:02d}" for i in range(20)])
    narrow = wide[:15]

    def risk(level: str, forcing: str, model: str, horizon: int) -> pd.Series:
        support = wide if level == "L0" else narrow
        return pd.Series(np.linspace(1.0, 2.0, len(support)), index=support)

    monkeypatch.setattr(I, "risk", risk)
    monkeypatch.setattr(I.L.V5, "MODELS", ["LightGBM"])
    monkeypatch.setattr(I.L.V5, "HORIZONS", [7])
    monkeypatch.setattr(
        I, "huc2_cluster_map",
        lambda registry: {s: ("A" if i < 10 else "B") for i, s in enumerate(wide)})
    monkeypatch.setattr(I, "load_station_registry", lambda path: None)

    counts = {row["n_stations"] for row in I.build_rows()}
    assert counts == {len(narrow)}


def test_geometry_is_pinned_to_whole_region() -> None:
    """Letting a third axis vary would be a three-way contrast nine effective
    clusters cannot carry; the L-by-G interaction has its own authority."""
    assert I.GEOMETRY == "whole_region"


def test_a_missing_shard_is_an_error_and_not_a_silent_omission() -> None:
    with pytest.raises(I.InteractionError, match="missing shard"):
        I.risk("L0", "F0", "NoSuchModel", 999)

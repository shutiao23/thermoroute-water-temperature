"""Tests for the within-station-month shuffled-forcing placebo (protocol v5a).

The placebo is only a control if the permutation does what the sealed protocol
says: preserve station climate, seasonal phase and the joint marginal
distribution of the future variables, and destroy nothing except the
correspondence between a forecast key and the weather that followed it.  These
tests check each of those properties separately, on a synthetic panel where the
right answer is known and on the real panel where it is not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.final.run_forcing_ladder_v5_observed as V5
import scripts.final.run_forcing_placebo_v5a_shuffle as P
from thermoroute import config as C

MET = list(V5.METEOROLOGY_VARIABLES)


def _panel(n_sites: int = 3, days: int = 400, *, seed: int = 7) -> pd.DataFrame:
    """A schema-exact raw panel with a seasonal cycle and a station offset."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=days, freq="D")
    frames = []
    for index in range(n_sites):
        site = f"{index + 1:08d}"
        doy = dates.dayofyear.to_numpy()
        season = np.sin(2 * np.pi * doy / 365.25)
        frame = pd.DataFrame({"site_id": site, "DATE": dates})
        for offset, variable in enumerate(C.ALL_VARS):
            base = 10.0 * index + offset + 5.0 * season
            frame[variable] = base + rng.normal(0.0, 0.5, size=days)
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    for variable in C.ALL_VARS:
        panel[f"{variable}_observed"] = np.isfinite(panel[variable].to_numpy())
    return panel[list(V5.RAW_PANEL_COLUMNS)].reset_index(drop=True)


def _month_key(frame: pd.DataFrame) -> pd.Series:
    return frame["DATE"].dt.year * 100 + frame["DATE"].dt.month


# ------------------------------------------------------------------ preserved


def test_permutation_preserves_the_multiset_of_values_in_every_station_month() -> None:
    true = _panel()
    shuffled = P.permute_meteorology_within_station_month(true, seed=0)
    for keys, block in true.groupby(["site_id", _month_key(true)], sort=True):
        other = shuffled[
            (shuffled["site_id"] == keys[0]) & (_month_key(shuffled) == keys[1])
        ]
        for variable in MET:
            np.testing.assert_allclose(
                np.sort(block[variable].to_numpy()),
                np.sort(other[variable].to_numpy()),
            )


def test_permutation_preserves_station_climate_and_seasonal_phase() -> None:
    true = _panel()
    shuffled = P.permute_meteorology_within_station_month(true, seed=1)
    # station mean is preserved exactly; month-of-year mean is preserved exactly
    for variable in MET:
        for group in ("site_id",):
            np.testing.assert_allclose(
                true.groupby(group)[variable].mean().to_numpy(),
                shuffled.groupby(group)[variable].mean().to_numpy(),
            )
        np.testing.assert_allclose(
            true.groupby(_month_key(true))[variable].mean().to_numpy(),
            shuffled.groupby(_month_key(shuffled))[variable].mean().to_numpy(),
        )


def test_permutation_moves_the_five_variables_as_one_vector() -> None:
    """Within-day meteorological coherence must survive the shuffle."""
    true = _panel()
    shuffled = P.permute_meteorology_within_station_month(true, seed=2)
    true_rows = {tuple(np.round(row, 12)) for row in true[MET].to_numpy()}
    shuffled_rows = {tuple(np.round(row, 12)) for row in shuffled[MET].to_numpy()}
    # every shuffled row vector is a row vector that genuinely occurred
    assert shuffled_rows == true_rows


def test_permutation_leaves_identity_water_temperature_and_flow_untouched() -> None:
    true = _panel()
    shuffled = P.permute_meteorology_within_station_month(true, seed=3)
    for column in ("site_id", "DATE", "WTEMP", "FLOW"):
        if column in MET:  # pragma: no cover - guards the fixture, not the code
            continue
        np.testing.assert_array_equal(
            true[column].to_numpy(), shuffled[column].to_numpy()
        )


def test_observedness_still_equals_isfinite_after_the_shuffle() -> None:
    true = _panel()
    true.loc[5, "TEMP"] = np.nan
    true["TEMP_observed"] = np.isfinite(true["TEMP"].to_numpy())
    shuffled = P.permute_meteorology_within_station_month(true, seed=4)
    for variable in C.ALL_VARS:
        np.testing.assert_array_equal(
            shuffled[f"{variable}_observed"].to_numpy(),
            np.isfinite(shuffled[variable].to_numpy()),
        )
    # the raw-panel invariant is what the v5 factory enforces
    V5._validate_raw_frame(shuffled)


# ------------------------------------------------------------------ destroyed


def test_permutation_is_a_derangement_within_every_multi_day_stratum() -> None:
    true = _panel()
    shuffled = P.permute_meteorology_within_station_month(true, seed=5)
    same = np.all(
        np.isclose(true[MET].to_numpy(), shuffled[MET].to_numpy()), axis=1
    )
    ledger = shuffled.attrs["placebo_ledger"]
    # no row keeps its own vector except in strata that cannot be deranged
    assert int(same.sum()) <= ledger["singleton_rows_left_in_place"]
    assert ledger["residual_fixed_points"] == 0


def test_within_stratum_anomaly_correspondence_is_destroyed() -> None:
    """The day-to-day anomaly must decorrelate; the seasonal level must not.

    A within-month shuffle preserves each station-month's mean exactly, so the
    seasonal signal survives almost untouched and the raw lag-1 autocorrelation
    barely moves.  That is the point of this arm rather than a weakness of it:
    it isolates the *within-month, day-to-day* anomaly, which is the part of
    future weather that a forecast key could correspond to.  The property to
    check is therefore that the anomaly about the stratum mean is decorrelated
    from the true anomaly at the same dates.
    """
    true = _panel(n_sites=3, days=900)
    shuffled = P.permute_meteorology_within_station_month(true, seed=6)
    key = true["site_id"].astype(str) + "|" + _month_key(true).astype(str)
    for variable in MET:
        a = (
            true[variable].to_numpy()
            - true.groupby(key)[variable].transform("mean").to_numpy()
        )
        b = (
            shuffled[variable].to_numpy()
            - shuffled.groupby(key.to_numpy())[variable].transform("mean").to_numpy()
        )
        assert abs(float(np.corrcoef(a, b)[0, 1])) < 0.15


def test_the_station_month_mean_is_preserved_exactly() -> None:
    """The complement of the test above: the level the anomaly is taken about."""
    true = _panel()
    shuffled = P.permute_meteorology_within_station_month(true, seed=6)
    key_true = true["site_id"].astype(str) + "|" + _month_key(true).astype(str)
    key_shuf = shuffled["site_id"].astype(str) + "|" + _month_key(shuffled).astype(str)
    for variable in MET:
        np.testing.assert_allclose(
            true.groupby(key_true)[variable].mean().to_numpy(),
            shuffled.groupby(key_shuf)[variable].mean().to_numpy(),
        )


def test_different_seeds_give_different_permutations() -> None:
    true = _panel()
    a = P.permute_meteorology_within_station_month(true, seed=0)
    b = P.permute_meteorology_within_station_month(true, seed=1)
    assert not np.allclose(a[MET].to_numpy(), b[MET].to_numpy())


def test_permutation_is_deterministic_for_a_seed() -> None:
    true = _panel()
    a = P.permute_meteorology_within_station_month(true, seed=11)
    b = P.permute_meteorology_within_station_month(true, seed=11)
    np.testing.assert_array_equal(a[MET].to_numpy(), b[MET].to_numpy())


def test_a_donor_never_comes_from_another_station_or_another_month() -> None:
    true = _panel()
    shuffled = P.permute_meteorology_within_station_month(true, seed=8)
    key = list(zip(true["site_id"], _month_key(true), strict=True))
    lookup: dict[tuple, set] = {}
    for index, entry in enumerate(key):
        lookup.setdefault(entry, set()).add(
            tuple(np.round(true[MET].to_numpy()[index], 12))
        )
    for index, entry in enumerate(key):
        assert tuple(np.round(shuffled[MET].to_numpy()[index], 12)) in lookup[entry]


# ------------------------------------------------------------------- plumbing


def test_shuffled_panel_passes_the_v5_raw_factory() -> None:
    true = _panel()
    panel = V5.make_raw_label_panel(true, source_sha256="0" * 64)
    shuffled, ledger = P.build_shuffled_panel(panel, seed=0)
    assert shuffled.source_sha256 != panel.source_sha256
    assert ledger["strata"] > 0
    frame = V5._validated_raw_frame(shuffled)
    np.testing.assert_array_equal(
        frame["WTEMP"].to_numpy(), V5._validated_raw_frame(panel)["WTEMP"].to_numpy()
    )


def test_build_shuffled_panel_rejects_a_permutation_that_moved_water_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = V5.make_raw_label_panel(_panel(), source_sha256="0" * 64)

    original = P.permute_meteorology_within_station_month

    def sabotage(frame: pd.DataFrame, *, seed: int) -> pd.DataFrame:
        out = original(frame, seed=seed)
        out["WTEMP"] = out["WTEMP"].to_numpy()[::-1]
        out["WTEMP_observed"] = np.isfinite(out["WTEMP"].to_numpy())
        return out

    monkeypatch.setattr(P, "permute_meteorology_within_station_month", sabotage)
    with pytest.raises(P.PlaceboError, match="WTEMP"):
        P.build_shuffled_panel(panel, seed=0)


def test_runner_declares_the_sealed_arm_and_five_seeds() -> None:
    assert P.PLACEBO_ARM == "F3_shuffle_month"
    assert P.SHUFFLE_SEEDS == (0, 1, 2, 3, 4)
    assert P.FEATURE_ARM == "F3_full"


def test_execution_authority_binds_runner_runtime_and_commit() -> None:
    authority = P.build_execution_authority(require_clean_tree=False)
    assert authority["runner_sha256"] == P._sha256_file(
        Path(P.__file__).resolve()
    )
    assert authority["protocol_sha256"] == P._sha256_file(P.SEALER.PROTOCOL)
    assert set(P.RUNTIME_PACKAGES) <= set(authority["runtime"]["packages"])
    assert authority["arm"] == "F3_shuffle_month"

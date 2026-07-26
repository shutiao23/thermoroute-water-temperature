"""Leakage and split-integrity tests — the checks reviewers care about most.

Run:  PYTHONPATH=src python3 -m pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from thermoroute import config as C
from thermoroute import baselines as B
from thermoroute import data as D
from thermoroute import features as F
from thermoroute import datasets as DS


SYNTHETIC_SITES = ("site_alpha", "site_beta", "site_gamma")


def _synthetic_daily_panel() -> pd.DataFrame:
    """Create a deterministic 2006--2020 panel with controlled missingness."""
    dates = pd.date_range("2006-01-01", "2020-12-31", freq="D")
    day = np.arange(len(dates), dtype=float)
    annual = 2.0 * np.pi * dates.dayofyear.to_numpy(float) / C.SEASONAL_PERIOD
    frames: list[pd.DataFrame] = []
    for site_index, site_id in enumerate(SYNTHETIC_SITES):
        offset = float(site_index)
        frame = pd.DataFrame({
            "DATE": dates,
            "site_id": site_id,
            "WTEMP": 11.0 + offset + 7.0 * np.sin(annual) + 0.15 * np.sin(day / 9.0),
            "FLOW": 45.0 + 4.0 * offset + 12.0 * np.cos(annual) + day % 11.0,
            "WLEVEL": 2.0 + 0.1 * offset + 0.25 * np.cos(annual),
            "TEMP": 9.0 + offset + 11.0 * np.sin(annual - 0.2),
            "PRCP": np.where(
                (day.astype(int) + site_index) % 17 == 0,
                4.0 + 0.2 * offset,
                0.0,
            ),
            "WDSP": 2.5 + 0.15 * offset + 0.5 * np.cos(day / 13.0),
            "RHMEAN": 60.0 + 5.0 * np.sin(annual + 0.4) + offset,
            "DH": 180.0 + 25.0 * np.sin(annual - 0.5) + 2.0 * offset,
        })
        for variable_index, variable in enumerate(C.ALL_VARS):
            missing = (
                np.arange(len(frame))
                + 37 * site_index
                + 53 * variable_index
            ) % 503 == 0
            frame.loc[missing, variable] = np.nan
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["site_id", "DATE"]
    ).reset_index(drop=True)


@pytest.fixture(scope="module")
def synthetic_panel_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("leakage-panel") / "daily_panel.parquet"
    _synthetic_daily_panel().to_parquet(path, index=False)
    return path


@pytest.fixture(autouse=True)
def _forbid_legacy_file_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("leakage tests must use only the synthetic panel")

    monkeypatch.setattr(D, "load_panel", forbidden)
    monkeypatch.setattr(D, "prepare_dataset", forbidden)


@pytest.fixture
def synthetic_bundle(
    synthetic_panel_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(C, "STATIONS", SYNTHETIC_SITES)
    bundle = D.prepare_dataset_from_panel(str(synthetic_panel_path))
    climatology = F.HarmonicClimatology.fit(
        bundle["panel"], bundle["masks"].train
    )
    return bundle, climatology


def test_split_disjoint_and_ordered():
    D.assert_split_disjoint()
    s = C.SPLIT.as_dict()
    assert s["train"][1] < s["val"][0] < s["val"][1] < s["calib"][0]
    assert s["calib"][1] < s["test"][0]


def test_no_calendar_gaps_or_dupes(synthetic_bundle):
    bundle, _climatology = synthetic_bundle
    panel = bundle["panel_raw"]
    for st in C.STATIONS:
        sub = panel[panel.site_id == st]
        full = pd.date_range(sub.DATE.min(), sub.DATE.max(), freq="D")
        assert len(full) == len(sub)
        assert sub.DATE.duplicated().sum() == 0


def test_window_and_tabular_builders_fail_closed_on_synthetic_calendar_gap(
    synthetic_bundle,
):
    bundle, clim = synthetic_bundle
    panel = bundle["panel"].copy()
    station = str(C.STATIONS[0])
    station_rows = panel.index[panel.site_id.astype(str).eq(station)]
    assert len(station_rows) > 3
    gappy = panel.drop(index=station_rows[len(station_rows) // 2]).reset_index(drop=True)
    with pytest.raises(ValueError, match="calendar gap"):
        DS.build_windows(gappy, bundle["masks"], clim)
    with pytest.raises(ValueError, match="calendar gap"):
        F.build_tabular(gappy, 3, C.FEATURE_SETS["V2"], clim)


def test_sentinels_masked():
    raw = pd.DataFrame({
        "WDSP": [2.0, C.SENTINELS["WDSP"], C.SENTINELS["WDSP"] + 1.0],
        "PRCP": [0.0, C.SENTINELS["PRCP"], C.SENTINELS["PRCP"] + 1.0],
    })
    panel, counts = D._mask_sentinels(raw)
    assert counts == {"WDSP": 2, "PRCP": 2}
    assert panel["WDSP"].max() < C.SENTINELS["WDSP"]
    assert panel["PRCP"].max() < C.SENTINELS["PRCP"]


def test_imputation_only_uses_train(synthetic_bundle):
    """An imputer fit on train must be reproducible from train rows alone."""
    bundle, _climatology = synthetic_bundle
    panel = bundle["panel_raw"]
    masks = bundle["masks"]
    imp = bundle["imputer"]
    # global medians equal the train-only medians (no future leak)
    tr = panel.loc[masks.train]
    for st in C.STATIONS:
        for v in C.ALL_VARS:
            ref = float(tr[tr.site_id == st][v].median())
            assert np.isclose(imp.global_median[(st, v)], ref, equal_nan=True)


def test_window_tail_equals_issue_value(synthetic_bundle):
    """The last history step must invert to WTEMP_t — no future bleed."""
    b, clim = synthetic_bundle
    wd = DS.build_windows(b["panel"], b["masks"], clim)
    DS._assert_no_leakage(wd, b["panel"])   # raises on any mismatch
    assert len(wd.X) > 10000


def test_window_splits_are_target_closed(synthetic_bundle):
    """No horizon target may cross train/val/calib/test boundaries."""
    b, clim = synthetic_bundle
    wd = DS.build_windows(b["panel"], b["masks"], clim)
    expected = (wd.issue_date[:, None]
                + np.asarray(wd.horizons)[None, :] * np.timedelta64(1, "D"))
    assert np.array_equal(wd.target_date, expected)
    for name, (lo, hi) in C.SPLIT.as_dict().items():
        sel = wd.split == name
        assert sel.any()
        assert (wd.issue_date[sel] >= np.datetime64(lo)).all()
        assert (wd.issue_date[sel] <= np.datetime64(hi)).all()
        assert (wd.target_date[sel] >= np.datetime64(lo)).all()
        assert (wd.target_date[sel] <= np.datetime64(hi)).all()
        # Multi-horizon windows share one sample registry, so max(horizon) days
        # at the end of each partition are embargoed.
        assert wd.issue_date[sel].max() <= (
            np.datetime64(hi) - np.timedelta64(max(wd.horizons), "D"))


def test_confirmation_targets_are_available_independently_by_horizon(
    synthetic_bundle,
):
    """Late h=1 issues and asynchronous labels must not be lost to h=7."""
    bundle, clim = synthetic_bundle
    panel = bundle["panel"].copy()
    site = str(C.STATIONS[0])
    # One missing target invalidates only the station/horizon issue pairs that
    # actually point at this date; it cannot complete-case-filter other heads.
    missing_date = pd.Timestamp("2020-12-29")
    panel.loc[
        panel.site_id.eq(site) & panel.DATE.eq(missing_date),
        "WTEMP_observed",
    ] = False
    wd = DS.build_windows(
        panel,
        bundle["masks"],
        clim,
        require_observed_target=True,
        evaluation_interval=("2020-12-20", "2020-12-31"),
        evaluation_split="confirm",
        independent_horizon_targets=True,
    )
    assert wd.target_valid.shape == wd.y.shape
    for column, horizon in enumerate(wd.horizons):
        selected = wd.target_valid[:, column]
        assert selected.any()
        assert wd.issue_date[selected].max() == (
            np.datetime64("2020-12-31") - np.timedelta64(horizon, "D")
        )
        assert (wd.target_date[selected, column] <= np.datetime64("2020-12-31")).all()
        assert np.isfinite(wd.y[selected, column]).all()
        assert np.isnan(wd.y[~selected, column]).all()

    station_rows = wd.station == 0
    issue = np.datetime64("2020-12-22")
    row = np.flatnonzero(station_rows & (wd.issue_date == issue))
    assert len(row) == 1
    # 2020-12-29 is h=7 from the chosen issue; h=1 and h=3 remain observed.
    assert wd.target_valid[row[0]].tolist() == [True, True, False]


def test_target_is_strictly_future(synthetic_bundle):
    """The stored target y[:, hi] must equal panel WTEMP at issue_date + h —
    verified against the panel itself on a subsample, not just h > 0."""
    b, clim = synthetic_bundle
    wd = DS.build_windows(b["panel"], b["masks"], clim)
    lookup = {(s, d): w for s, d, w in zip(
        b["panel"].site_id, pd.to_datetime(b["panel"].DATE).to_numpy(),
        b["panel"].WTEMP.to_numpy(float))}
    rng = np.random.default_rng(0)
    sample = rng.choice(len(wd.X), size=min(500, len(wd.X)), replace=False)
    checked = 0
    for hi, h in enumerate(wd.horizons):
        assert h > 0
        for i in sample:
            st = C.STATIONS[wd.station[i]]
            tgt = lookup.get((st, wd.issue_date[i] + np.timedelta64(h, "D")))
            if tgt is not None and np.isfinite(tgt) and np.isfinite(wd.y[i, hi]):
                assert np.isclose(wd.y[i, hi], tgt, atol=1e-6), (
                    f"window {i}: y[h={h}]={wd.y[i, hi]} != panel WTEMP "
                    f"{tgt} at issue+{h}d for {st}")
                checked += 1
    assert checked > 500, "too few verifiable (window, horizon) pairs"
    # tabular path: target_date strictly after issue_date
    tab = F.attach_split(F.build_tabular(b["panel"], 3, C.FEATURE_SETS["V2"], clim))
    assert (pd.to_datetime(tab.target_date) > pd.to_datetime(tab.issue_date)).all()


def test_tabular_split_is_target_closed(synthetic_bundle):
    b, clim = synthetic_bundle
    tab = F.attach_split(F.build_tabular(b["panel"], 7, C.FEATURE_SETS["V3"], clim))
    for name, (lo, hi) in C.SPLIT.as_dict().items():
        rows = tab[tab.split == name]
        assert not rows.empty
        assert pd.to_datetime(rows.issue_date).min() >= pd.Timestamp(lo)
        assert pd.to_datetime(rows.issue_date).max() <= pd.Timestamp(hi) - pd.Timedelta(days=7)
        assert pd.to_datetime(rows.target_date).min() >= pd.Timestamp(lo)
        assert pd.to_datetime(rows.target_date).max() <= pd.Timestamp(hi)

    # Boundary rows still exist for traceability but are explicitly excluded.
    train_end_issue = tab[
        pd.to_datetime(tab.issue_date).between("2015-12-25", "2015-12-31")
    ]
    assert not train_end_issue.empty
    assert set(train_end_issue["split"]) == {"none"}


def test_feature_schema_blocks_hidden_forcing_paths(synthetic_bundle):
    """V1 must be invariant to forcings, including physics/gate side paths."""
    b, clim = synthetic_bundle
    v1 = C.FEATURE_SETS["V1"]
    first = DS.build_windows(b["panel"], b["masks"], clim, variables=v1)
    perturbed = b["panel"].copy()
    rng = np.random.default_rng(41)
    for var in C.FORCINGS:
        perturbed[var] = rng.normal(1000.0, 500.0, len(perturbed))
    second = DS.build_windows(perturbed, b["masks"], clim, variables=v1)

    assert first.feature_schema.variables == ("WTEMP",)
    assert first.phys_vars == () and first.phys_std.shape[1] == 0
    assert np.array_equal(first.X, second.X)
    assert np.array_equal(first.damped_prior, second.damped_prior)
    assert np.count_nonzero(first.logflowz) == 0
    assert np.count_nonzero(first.wlevelz) == 0
    # Gate layout: sin, cos, TEMP, FLOW, PRCP, WTEMP tendency.
    assert np.count_nonzero(first.gate[:, 2:5]) == 0
    assert np.array_equal(first.gate, second.gate)


def test_damped_anchor_is_immune_to_post_train_targets(synthetic_bundle):
    b, clim = synthetic_bundle
    original = F.DampedPersistenceAnchor.fit(b["panel"], b["masks"].train, clim)
    altered = b["panel"].copy()
    altered.loc[~b["masks"].train, "WTEMP"] += 1000.0
    refit = F.DampedPersistenceAnchor.fit(altered, b["masks"].train, clim)
    assert original.phi == refit.phi


def test_damped_baseline_reuses_the_window_anchor(synthetic_bundle):
    b, clim = synthetic_bundle
    tabs = B._tab_by_horizon(b["panel"], clim, C.FEATURE_SETS["V3"])
    _predictions, phi = B.run_damped_persistence(
        b["panel"], b["masks"], tabs, clim
    )
    anchor = F.DampedPersistenceAnchor.fit(
        b["panel"], b["masks"].train, clim
    )
    assert phi == anchor.phi


def test_zero_shot_preprocessors_ignore_held_station_history(synthetic_bundle):
    b, _ = synthetic_bundle
    held_site = SYNTHETIC_SITES[-1]
    train_stations = tuple(s for s in C.STATIONS if s != held_site)
    clim1 = F.HarmonicClimatology.fit(
        b["panel"], b["masks"].train, fit_stations=train_stations, pooled=True)
    scale1 = D.StandardScalerPerStation.fit(
        b["panel"], b["masks"].train, variables=("WTEMP",),
        fit_stations=train_stations, pooled=True)
    damp1 = F.DampedPersistenceAnchor.fit(
        b["panel"], b["masks"].train, clim1,
        fit_stations=train_stations, pooled=True)

    altered = b["panel"].copy()
    held_train = b["masks"].train & altered.site_id.eq(held_site).to_numpy()
    altered.loc[held_train, "WTEMP"] += 500.0
    clim2 = F.HarmonicClimatology.fit(
        altered, b["masks"].train, fit_stations=train_stations, pooled=True)
    scale2 = D.StandardScalerPerStation.fit(
        altered, b["masks"].train, variables=("WTEMP",),
        fit_stations=train_stations, pooled=True)
    damp2 = F.DampedPersistenceAnchor.fit(
        altered, b["masks"].train, clim2,
        fit_stations=train_stations, pooled=True)

    assert np.array_equal(clim1.coef[held_site], clim2.coef[held_site])
    assert scale1.mean[(held_site, "WTEMP")] == scale2.mean[(held_site, "WTEMP")]
    assert scale1.std[(held_site, "WTEMP")] == scale2.std[(held_site, "WTEMP")]
    assert damp1.phi == damp2.phi


def test_signed_flow_transform_preserves_reverse_flow_and_round_trips():
    raw = np.array([-121.0, -0.01, 0.0, 3.0, 900.0])
    transformed = D.stabilising_transform("FLOW", raw)
    restored = D.inverse_stabilising_transform("FLOW", transformed)
    assert transformed[0] < transformed[1] < 0
    assert transformed[2] == 0
    assert np.allclose(restored, raw)


def test_tabular_learned_baseline_can_receive_missingness_information(
    synthetic_bundle,
):
    bundle, clim = synthetic_bundle
    tab = F.build_tabular(
        bundle["panel"], 1, C.FEATURE_SETS["V3"], clim,
        drop_feature_nans=False, include_missingness=True,
    )
    mask_columns = [column for column in tab if "_observed_" in column]
    assert mask_columns
    values = tab[mask_columns].to_numpy(float)
    assert np.nanmin(values) == 0.0
    assert np.nanmax(values) == 1.0

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute import config as C  # noqa: E402
from thermoroute import data as D  # noqa: E402
from thermoroute import features as F  # noqa: E402
from thermoroute.checkpoint import neural_output_head_schema  # noqa: E402
from thermoroute.frozen_inference import (  # noqa: E402
    FrozenInferenceError,
    build_frozen_confirmation_windows,
    reconstruct_frozen_transforms,
    thermoroute_factory_from_metadata,
)
from thermoroute.weighting import (  # noqa: E402
    STATION_EQUAL_WEIGHTING,
    STATION_SUMMARY_EQUAL_WEIGHTING,
)


FEATURES = ("WTEMP", "FLOW", "TEMP")


def _metadata(*, pooled: bool = False, station_agnostic: bool = False):
    training_sites = ("01000001", "01000002")
    seasonal = {}
    global_medians = {}
    means = {}
    stds = {}
    coefficients = {}
    phi = {}
    for index, site in enumerate(training_sites):
        for variable in FEATURES:
            base = 10.0 + (0.0 if pooled else index)
            seasonal[f"{site}|{variable}"] = {
                str(day): base for day in range(1, 367)
            }
            global_medians[f"{site}|{variable}"] = base
            means[f"{site}|{variable}"] = 0.0 if pooled else float(index)
            stds[f"{site}|{variable}"] = 2.0
        coefficients[site] = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        phi[site] = 0.8
    return {
        "run_id": "fixture",
        "output_head_schema": neural_output_head_schema(),
        "architecture": {
            "class": "thermoroute.thermoroute.ThermoRoute",
            "kwargs": {
                "n_vars": len(FEATURES),
                "n_stations": len(training_sites),
                "n_phys": 1,
                "delta_scale": 1.0,
                "safety_anchor": "damped",
                "station_agnostic": station_agnostic,
            },
        },
        "feature_order": list(FEATURES),
        "horizons": [1, 3, 7],
        "station_to_index": {site: index for index, site in enumerate(training_sites)},
        "preprocessing": {
            "input_schema": {
                "variables": list(FEATURES),
                "context_length": C.CONTEXT_LENGTH,
                "transforms": {"FLOW": "signed_log1p"},
                "missingness_mask": True,
            },
            "imputer": {
                "method": (
                    D.POOLED_STATION_BALANCED_IMPUTER_METHOD if pooled
                    else D.PER_STATION_IMPUTER_METHOD
                ),
                "pooled": pooled,
                "pool_weighting": (
                    STATION_SUMMARY_EQUAL_WEIGHTING if pooled else None
                ),
                "fit_stations": list(training_sites),
                "seasonal_medians": seasonal,
                "global_medians": global_medians,
            },
            "scaler": {
                "method": D.POOLED_SCALER_METHOD if pooled
                          else D.PER_STATION_SCALER_METHOD,
                "pooled": pooled,
                "pool_weighting": STATION_EQUAL_WEIGHTING if pooled else None,
                "variance": D.POOLED_SCALER_VARIANCE if pooled
                            else "within_station_sample_variance_ddof_1",
                "mean": means,
                "std": stds,
                "fit_stations": list(training_sites),
            },
            "climatology": {
                "method": F.POOLED_HARMONIC_METHOD if pooled
                          else F.PER_STATION_HARMONIC_METHOD,
                "pooled": pooled,
                "pool_weighting": STATION_EQUAL_WEIGHTING if pooled else None,
                "harmonics": 3,
                "coefficients": coefficients,
                "fit_stations": list(training_sites),
            },
            "damped_anchor": {
                "method": F.POOLED_DAMPED_AR_METHOD if pooled
                          else F.DAMPED_AR_METHOD,
                "pooled": pooled,
                "phi": phi,
                "fit_stations": list(training_sites),
                "fallback": 0.9,
                "min_pairs": F.DAMPED_MIN_PAIRS,
                "coefficient_bounds": [
                    F.DAMPED_LOWER_BOUND, F.DAMPED_UPPER_BOUND,
                ],
                "minimum_lagged_anomaly_mean_square": F.DAMPED_MIN_MEAN_SQUARE,
                "pair_rule": F.DAMPED_PAIR_RULE,
                "pool_weighting": STATION_EQUAL_WEIGHTING,
                "eligibility_rule": F.DAMPED_ELIGIBILITY_RULE,
                "eligible_fit_stations": list(training_sites),
                "pair_counts": {
                    site: F.DAMPED_MIN_PAIRS for site in training_sites
                },
                "lagged_anomaly_mean_squares": {
                    site: 1.0 for site in training_sites
                },
            },
        },
    }


def _panel(sites=("01000001", "01000002")):
    dates = pd.date_range("2020-12-01", "2021-01-12", freq="D")
    frames = []
    for offset, site in enumerate(sites):
        values = 10.0 + offset + np.linspace(0.0, 1.0, len(dates))
        frame = pd.DataFrame({
            "DATE": dates,
            "site_id": site,
            "WTEMP": values,
            "FLOW": np.full(len(dates), 5.0 + offset),
            "TEMP": np.full(len(dates), 8.0 + offset),
        })
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_same_station_frozen_windows_use_bundle_transforms_and_confirmation_interval():
    old_stations, old_upstream = C.STATIONS, C.UPSTREAM
    try:
        metadata = _metadata()
        panel = _panel()
        # Missing weather is filled from bundle medians; target observation masks
        # remain true only for actually observed WTEMP.
        panel.loc[panel.index[5], "TEMP"] = np.nan
        wd, transforms, imputed = build_frozen_confirmation_windows(
            panel,
            metadata,
            ("01000001", "01000002"),
            interval=("2021-01-01", "2021-01-10"),
            external=False,
        )
        assert set(wd.split) == {"confirm"}
        assert wd.issue_date.min() == np.datetime64("2021-01-01")
        assert wd.target_valid is not None
        assert wd.target_date[wd.target_valid].max() <= np.datetime64("2021-01-10")
        assert tuple(wd.var_names) == FEATURES
        assert transforms.scaler.mean[("01000002", "WTEMP")] == 1.0
        assert imputed.loc[5, "TEMP"] == 10.0
        # Horizon-specific confirmation support retains legal h1/h3 forecasts
        # instead of dropping them merely because h7 crosses the boundary.
        assert len(wd.issue_date) == 2 * 9
        assert wd.target_valid.sum(axis=0).tolist() == [18, 14, 6]
    finally:
        C.STATIONS, C.UPSTREAM = old_stations, old_upstream


def test_external_bundle_rejects_station_embedding_and_nonpooled_transforms():
    sites = ("02000001", "02000002")
    with pytest.raises(FrozenInferenceError, match="station_agnostic"):
        reconstruct_frozen_transforms(
            _metadata(pooled=True, station_agnostic=False), sites, external=True
        )
    with pytest.raises(FrozenInferenceError, match="imputer must declare pooled"):
        reconstruct_frozen_transforms(
            _metadata(pooled=False, station_agnostic=True), sites, external=True
        )


def test_external_bundle_expands_only_verified_pooled_statistics():
    sites = ("02000001", "02000002")
    transforms = reconstruct_frozen_transforms(
        _metadata(pooled=True, station_agnostic=True), sites, external=True
    )
    assert transforms.station_agnostic
    assert transforms.scaler.pooled
    assert transforms.climatology.pooled
    assert transforms.damped_anchor.pooled
    assert transforms.scaler.mean[(sites[0], "FLOW")] == 0.0
    assert transforms.scaler.mean[(sites[1], "FLOW")] == 0.0


def test_external_bundle_rejects_false_pooled_declaration():
    metadata = _metadata(pooled=True, station_agnostic=True)
    metadata["preprocessing"]["scaler"]["mean"]["01000002|FLOW"] = 9.0
    with pytest.raises(FrozenInferenceError, match="stored values differ"):
        reconstruct_frozen_transforms(
            metadata, ("02000001",), external=True
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda block: block["eligible_fit_stations"].pop(),
            "eligible-station registry is stale",
        ),
        (
            lambda block: block["pair_counts"].pop("01000002"),
            "eligibility evidence differs",
        ),
        (
            lambda block: block["pair_counts"].update({"extra": 40}),
            "eligibility evidence differs",
        ),
        (
            lambda block: block.update({"unknown": True}),
            "schema is not exact",
        ),
    ],
)
def test_damped_eligibility_metadata_tampering_fails_closed(mutation, message):
    metadata = _metadata(pooled=True, station_agnostic=True)
    mutation(metadata["preprocessing"]["damped_anchor"])
    with pytest.raises(FrozenInferenceError, match=message):
        reconstruct_frozen_transforms(metadata, ("02000001",), external=True)


def test_thermoroute_architecture_is_reconstructed_without_pickle():
    model = thermoroute_factory_from_metadata(_metadata())
    assert model.n_vars == len(FEATURES)
    assert model.safety_anchor == "damped"
    assert not model.prior.station_agnostic


def test_legacy_head_alias_metadata_cannot_be_reconstructed():
    metadata = _metadata()
    metadata.pop("output_head_schema")
    with pytest.raises(FrozenInferenceError, match="independent point/q50"):
        thermoroute_factory_from_metadata(metadata)


def test_legacy_unsigned_flow_transform_bundle_is_rejected():
    metadata = _metadata()
    metadata["preprocessing"]["input_schema"].pop("transforms")
    metadata["preprocessing"]["input_schema"]["log1p_variables"] = ["FLOW"]
    with pytest.raises(FrozenInferenceError, match="transform map"):
        reconstruct_frozen_transforms(
            metadata, ("01000001", "01000002"), external=False
        )

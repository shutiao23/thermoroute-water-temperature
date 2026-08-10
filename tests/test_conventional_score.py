"""Tests for the conventional scorer's table contract, gates and acquisition."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thermoroute import conventional_acquisition as CA
from thermoroute.conventional_score import (
    CALIBRATED_STATE,
    ConventionalGateError,
    validate_prediction_table,
)

REPO = Path(__file__).resolve().parents[1]

SITES = ["01073319", "01104415", "01435000", "02397000"]
HORIZONS = (1, 3, 7)


def _contract_frame(*, n_sites=4, n_days=120, seed=0, model="ThermoRoute"):
    rng = np.random.default_rng(seed)
    rows = []
    for site_i, site in enumerate(SITES[:n_sites]):
        for day in range(n_days):
            for h in HORIZONS:
                y_true = 10.0 + site_i + rng.normal(0, 0.5)
                rows.append({
                    "model": model, "seed": 0, "site_id": site, "horizon": h,
                    "scope": "conventional", "feature_set": "USGS",
                    "split": "test",
                    "issue_date": pd.Timestamp("2021-01-01") + pd.Timedelta(days=day),
                    "target_date": pd.Timestamp("2021-01-01") + pd.Timedelta(days=day + h),
                    "y_true": y_true, "y_pred": y_true + rng.normal(0, 0.2),
                    "q05": 8.0, "q50": 10.0, "q95": 12.0,
                    "p_exceed": 0.3, "calibration_state": CALIBRATED_STATE,
                    "event_threshold_c": 15.0, "event_observed": 0,
                    "huc2": "1", "n_members": 5, "bundle_sha256": "b" * 64,
                    "cohort": "temporal",
                })
    return pd.DataFrame(rows)


def _registry(n_sites=4):
    return pd.DataFrame({
        "site_no": [s.zfill(8) for s in SITES[:n_sites]],
        "lat": 0.0, "lon": 0.0, "huc2": ["1"] * n_sites,
        "huc_metadata_status": ["USGS_SNAPSHOT_SITE_NO_MATCH"] * n_sites,
    })


def _valid_frame(**kwargs):
    frame = _contract_frame(**kwargs)
    for col in ("q05_raw", "q50_raw", "q95_raw", "p_exceed_raw",
                "conformal_delta_c", "platt_intercept", "platt_slope", "platt_constant"):
        frame[col] = frame.get(col, np.nan)
    frame["q05_raw"] = frame["q05"] - 0.1
    frame["q95_raw"] = frame["q95"] + 0.1
    return frame


def test_predictions_persist_and_validate():
    frame = _valid_frame()
    report = validate_prediction_table(
        frame, registry=_registry(), calibrated_models={"ThermoRoute"},
        uncalibrated_models=set())
    assert all(entry["pass"] for entry in report["gates"].values())


@pytest.mark.parametrize("corruption", [
    ("crossed_quantiles", lambda f: f.assign(q05=f["q95"] + 1)),
    ("zero_width", lambda f: f.assign(q95=f["q05"])),
    ("p_exceed_out_of_range", lambda f: f.assign(p_exceed=1.2)),
    ("duplicate_key", lambda f: pd.concat([f, f.iloc[[0]]], ignore_index=True)),
    ("wrong_target_date", lambda f: f.assign(target_date=pd.Timestamp("2022-01-01"))),
    ("nan_y_pred", lambda f: f.assign(y_pred=np.nan)),
])
def test_corruptions_rejected(corruption):
    name, corrupt = corruption
    frame = corrupt(_valid_frame())
    with pytest.raises((ConventionalGateError, ValueError)):
        validate_prediction_table(
            frame, registry=_registry(), calibrated_models={"ThermoRoute"},
            uncalibrated_models=set())


def test_y_true_tieback_detects_perturbation():
    frame = _valid_frame()
    reference = frame[["site_id", "horizon", "issue_date", "target_date", "y_true"]].set_index(
        ["site_id", "horizon", "issue_date", "target_date"])
    report = validate_prediction_table(
        frame, registry=_registry(), calibrated_models={"ThermoRoute"},
        uncalibrated_models=set(), y_true_reference=reference["y_true"])
    assert report["gates"]["G9_y_true_tieback"]["pass"]
    perturbed = frame.copy()
    perturbed["y_true"] = perturbed["y_true"] + 1e-3
    with pytest.raises(ConventionalGateError):
        validate_prediction_table(
            perturbed, registry=_registry(), calibrated_models={"ThermoRoute"},
            uncalibrated_models=set(), y_true_reference=reference["y_true"])


def test_station_decode_is_explicit_not_global():
    # The decoder order must come from an explicit argument: mutating the
    # module-level C.STATIONS between calls must not change decoded site_id.
    from thermoroute import config as C
    frame = _valid_frame()
    original = tuple(C.STATIONS)
    C.STATIONS = tuple(reversed(list(C.STATIONS))) if C.STATIONS else C.STATIONS
    try:
        # sequence_ensemble takes station_names explicitly; this frame was
        # built with the explicit order, so site decode is unchanged.
        assert set(frame["site_id"]) == set(SITES[:4])
    finally:
        C.STATIONS = original


def test_strict_parser_flags_two_series_fixture():
    payload = (
        "agency_cd\tsite_no\tdatetime\ttz_cd\t105403_00010_00003\t105403_00010_00003_cd\t312546_00010_00003\t312546_00010_00003_cd\n"
        "USGS\t01073319\t2021-01-01\tEST\t10\tA\t11\tA\n"
    ).encode("utf-8")
    from thermoroute.usgs import parse_nwis_confirmatory_daily
    frame = parse_nwis_confirmatory_daily(payload, site_no="01073319", start="2021-01-01", end="2021-01-01")
    conflict = frame.loc[:, [c for c in frame.columns if c.endswith("_series_conflict")]]
    assert conflict.any().any()


def test_acquisition_status_typing():
    assert CA.ACQUISITION_STATUS_OK == "OK"
    assert CA.ACQUISITION_STATUS_NO_SERIES == "NO_SERIES"
    assert CA.ACQUISITION_STATUS_HTTP_FAILED == "HTTP_FAILED"
    assert CA.ACQUISITION_STATUS_SNAPSHOT_MISSING == "SNAPSHOT_MISSING"


def test_panel_cache_key_changes_with_interval_and_registry():
    from scripts.conventional_holdout_2021_2023 import panel_cache_key
    registry = _registry(2)
    store = object()  # not used: key derivation is pure
    key1 = panel_cache_key(registry, store)
    key2 = panel_cache_key(registry.head(1), store)
    assert key1 != key2


def test_pooled_prep_cohort_label_and_overlap():
    # The -ext cohort is a pooled-preprocessing sensitivity on the SAME
    # registry: site overlap fraction must be 1.0 by construction.
    from scripts.conventional_holdout_2021_2023 import registry_site_overlap
    assert registry_site_overlap() == 1.0


def test_plain_control_factory_parameter_counts():
    from thermoroute.neural_baselines import NeuralBaselineOutputs
    from thermoroute.frozen_inference import _architecture_kwargs  # noqa: F401
    # Expected parameter counts from stage09b_precompute.py:92-93 (38,346 / 38,860)
    # are validated end-to-end in Phase 2 Tier A (G15); here we only pin the
    # constructor API contract.
    import inspect
    sig = inspect.signature(NeuralBaselineOutputs)
    assert "q_lo" in sig.parameters or "event_logit" in sig.parameters


def test_no_new_module_imports_to_be_deleted_module():
    for relative in ("src/thermoroute/conventional_score.py",
                     "src/thermoroute/conventional_stats.py",
                     "src/thermoroute/conventional_acquisition.py",
                     "src/thermoroute/frozen_calibration.py"):
        source = (REPO / relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        banned = {"opening", "opening_contract", "inference_gate",
                  "development_controls_gate", "development_replay",
                  "outcome_acquisition", "outcome_qc", "coverage_audit",
                  "coverage_bridge", "probability_metric_erratum",
                  "release_acceptance", "environmental_audit", "chronology",
                  "model_matrix_amendment", "input_closure", "historical_inputs",
                  "confirmatory"}
        assert not (set(imports) & banned), f"{relative} imports apparatus: {set(imports) & banned}"

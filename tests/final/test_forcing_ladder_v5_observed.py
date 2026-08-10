"""Adversarial contracts for the create-only v5 observed-lineage correction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import scripts.final.run_forcing_ladder_v5_observed as V5  # noqa: E402
from thermoroute import config as C  # noqa: E402
from thermoroute import data as D  # noqa: E402
from thermoroute import features as F  # noqa: E402

SITE = "01000001"
SYNTHETIC_SHA256 = hashlib.sha256(b"synthetic-raw-panel").hexdigest()


def _synthetic_source() -> pd.DataFrame:
    dates = pd.date_range("2006-01-01", periods=120, freq="D")
    phase = np.arange(len(dates), dtype=np.float64)
    frame = pd.DataFrame(
        {
            "DATE": dates,
            "site_id": np.full(len(dates), SITE, dtype=object),
            "WTEMP": 10.0 + np.sin(phase / 9.0),
            "FLOW": 20.0 + phase / 20.0,
            "WLEVEL": 2.0 + phase / 1_000.0,
            "TEMP": 8.0 + np.cos(phase / 11.0),
            "PRCP": np.mod(phase, 5.0),
            "WDSP": 3.0 + np.mod(phase, 4.0) / 10.0,
            "RHMEAN": 60.0 + np.mod(phase, 7.0),
            "DH": 150.0 + phase / 10.0,
        },
        columns=V5.EXPECTED_PANEL_COLUMNS,
    )
    frame.loc[20, "WTEMP"] = np.nan
    frame.loc[25, "TEMP"] = np.nan
    return frame


def _synthetic_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[V5.RawLabelPanel, V5.ImputedFeaturePanel, F.HarmonicClimatology]:
    monkeypatch.setattr(C, "STATIONS", (SITE,))
    flagged = V5.add_raw_observed_flags(_synthetic_source())
    raw = V5.make_raw_label_panel(flagged, source_sha256=SYNTHETIC_SHA256)
    training = np.ones(len(flagged), dtype=np.bool_)
    imputer = D.Imputer.fit(flagged, training, fit_stations=(SITE,))
    climatology = F.HarmonicClimatology.fit(
        flagged,
        training,
        fit_stations=(SITE,),
    )
    imputed = V5.impute_feature_panel(raw, imputer)
    return raw, imputed, climatology


def _canonical_key(site: str, issue: pd.Timestamp, target: pd.Timestamp, lead: int) -> str:
    payload = f"temporal|known_site|{site}|{issue:%Y-%m-%d}|{target:%Y-%m-%d}|{lead}"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _reference_rows(issues: list[pd.Timestamp], lead: int = 1) -> pd.DataFrame:
    rows = []
    for index, issue in enumerate(issues):
        target = issue + pd.Timedelta(days=lead)
        rows.append(
            {
                "key_id": _canonical_key(SITE, issue, target, lead),
                "site_id": SITE,
                "issue_date": issue,
                "target_date": target,
                "lead_days": np.int16(lead),
                "y_true": float(index + 10),
                "issue_wtemp_observed": True,
                "days_since_last_observed_wtemp": np.int32(0),
                "n_observed_wtemp_7d": np.int16(7),
                "fraction_observed_wtemp_7d": 1.0,
                "n_observed_wtemp_14d": np.int16(14),
                "fraction_observed_wtemp_14d": 1.0,
                "n_observed_wtemp_32d": np.int16(32),
                "fraction_observed_wtemp_32d": 1.0,
                "history_H100": True,
                "history_H75": True,
                "history_Hall": True,
                "reportable_primary": True,
            }
        )
    return pd.DataFrame(rows, columns=V5.REFERENCE_COLUMNS)


def _base_table(
    issues: list[pd.Timestamp],
    splits: list[str],
    *,
    horizon: int = 1,
    y: list[float] | None = None,
) -> pd.DataFrame:
    count = len(issues)
    target = pd.Series(issues, dtype="datetime64[ns]") + pd.Timedelta(days=horizon)
    values = [float(index + 10) for index in range(count)] if y is None else y
    data: dict[str, object] = {
        "site_id": np.full(count, SITE, dtype=object),
        "issue_date": pd.Series(issues, dtype="datetime64[ns]"),
        "target_date": target,
        "y": np.asarray(values, dtype=np.float64),
        "issue_wtemp_observed": np.ones(count, dtype=np.bool_),
        "target_wtemp_observed": np.ones(count, dtype=np.bool_),
        "split": np.asarray(splits, dtype=object),
        "horizon": np.full(count, horizon, dtype=np.int16),
    }
    for feature in V5.FROZEN_BASE_FEATURE_COLUMNS:
        data[feature] = np.arange(1, count + 1, dtype=np.float64)
    data["clim_t"] = np.full(count, 9.0, dtype=np.float64)
    data["clim_target"] = np.full(count, 9.0, dtype=np.float64)
    data["persistence"] = np.full(count, 10.0, dtype=np.float64)
    return pd.DataFrame(data, columns=(*V5.BASE_TABLE_COLUMNS, *V5.FROZEN_BASE_FEATURE_COLUMNS))


def _candidate_table(issues: list[pd.Timestamp], *, lead: int = 1) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "site_id": np.full(len(issues), SITE, dtype=object),
            "issue_date": pd.Series(issues, dtype="datetime64[ns]"),
            "target_date": pd.Series(issues, dtype="datetime64[ns]") + pd.Timedelta(days=lead),
            "y": np.arange(10, 10 + len(issues), dtype=np.float64),
            "split": np.full(len(issues), "confirm", dtype=object),
        }
    )
    features = pd.DataFrame(
        np.ones((len(table), len(V5.FROZEN_BASE_FEATURE_COLUMNS)), dtype=np.float64),
        columns=V5.FROZEN_BASE_FEATURE_COLUMNS,
    )
    return pd.concat([table.reset_index(drop=True), features], axis=1)


@pytest.fixture(scope="module")
def pinned_inputs() -> tuple[V5.RawLabelPanel, tuple[str, ...], pd.DataFrame]:
    raw, stations, _registry = V5.load_pinned_raw_panel()
    reference = V5.load_pinned_reference_registry()
    return raw, stations, reference


def test_dry_run_is_data_free_scope_exact_and_gate_locked(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run attempted data or byte access")

    monkeypatch.setattr(V5.pd, "read_parquet", forbidden)
    monkeypatch.setattr(V5.pd, "read_csv", forbidden)
    monkeypatch.setattr(V5, "_sha256_file", forbidden)
    monkeypatch.setattr(V5, "_read_stable_regular", forbidden)
    assert V5.main(["--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["analysis_status"] == V5.STATUS
    assert plan["execution_performed"] is False
    assert plan["scope"] == {
        "arms": ["F0", "F3_full"],
        "models": ["LightGBM", "ResidualLightGBM"],
        "horizons": [1, 3, 7],
        "cell_count": 12,
        "other_arms_authorized": False,
    }
    assert plan["defect_authority_gate"] == {
        "manifest_path": (
            "outputs/final/preprocessing_lineage_defect_authority_v1/"
            "preprocessing_lineage_defect_authority_v1_manifest.json"
        ),
        "expected_sha256": ("e69124409f49e4fb2aaaae319104251ca3078e535fca69121ed0eafddb23d908"),
        "state": "PIN_DECLARED_FILESYSTEM_NOT_READ_BY_DRY_RUN",
    }
    assert plan["score_execution_authority_gate"]["state"] == (
        "LOCKED_NO_CANONICAL_SCORE_EXECUTION_SEAL"
    )
    assert plan["score_execution_authority_gate"]["draft_protocol_execution_authorized"] is False
    assert (
        plan["score_execution_authority_gate"]["defect_authority_is_score_authorization"] is False
    )
    assert plan["frozen_base_feature_count"] == 210
    assert plan["publication"]["single_linux_RENAME_NOREPLACE"] is True


def test_execute_is_locked_before_data_fit_lock_or_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert V5.EXPECTED_DEFECT_AUTHORITY_SHA256 == (
        "e69124409f49e4fb2aaaae319104251ca3078e535fca69121ed0eafddb23d908"
    )
    assert not os.path.lexists(V5.DEFAULT_OUTPUT_DIR)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("locked execution crossed a production side-effect boundary")

    monkeypatch.setattr(V5, "_read_stable_regular", forbidden)
    monkeypatch.setattr(V5.pd, "read_parquet", forbidden)
    monkeypatch.setattr(V5, "_lgb_fit", forbidden)
    monkeypatch.setattr(V5, "_BundleTransaction", forbidden)
    with pytest.raises(RuntimeError, match="canonical sealed score-execution"):
        V5.execute(V5.DEFAULT_OUTPUT_DIR)
    assert not os.path.lexists(V5.DEFAULT_OUTPUT_DIR)


def test_only_exact_production_destination_is_accepted_before_capture(tmp_path: Path) -> None:
    args = argparse.Namespace(output_dir=tmp_path / "external", execute=True, dry_run=False)
    with pytest.raises(ValueError, match="fixed to"):
        V5.run(args)
    traversal = V5.FINAL_OUTPUT_ROOT / ".." / "final" / V5.DEFAULT_OUTPUT_DIR.name
    with pytest.raises(ValueError, match="fixed to"):
        V5.run(argparse.Namespace(output_dir=traversal, execute=True, dry_run=False))


def test_production_source_protocol_key_pins_and_formal_registry_binding(
    pinned_inputs: tuple[V5.RawLabelPanel, tuple[str, ...], pd.DataFrame],
) -> None:
    evidence = V5.verify_production_pins()
    assert {name: item["sha256"] for name, item in evidence.items()} == dict(V5.PINNED_INPUT_SHA256)
    assert V5.PINNED_GOVERNANCE_SHA256["protocol"] == (
        "66e089baf37db1137cad23f148e31df39cc71aea872dec4a97dc6f8701d13a98"
    )
    assert V5.PINNED_GOVERNANCE_SHA256["key_authority_manifest"] == (
        "ac0c256907264022e1fe7c4e407e0f95ece1f03233ecd6bb1841e27eb40b49ea"
    )
    assert V5.PINNED_GOVERNANCE_SHA256["semantic_authority_manifest"] == (
        "12cdc355a06d2c39733a60386dfdeb8a2f6b234d2f8d6f641a995a3d3c41072c"
    )
    raw, stations, reference = pinned_inputs
    assert len(raw.frame) == 788_880
    assert len(stations) == 120
    assert raw.frame["DATE"].min() == pd.Timestamp("2006-01-01")
    assert raw.frame["DATE"].max() == pd.Timestamp("2023-12-31")
    assert len(reference) == 358_765
    assert reference.groupby("lead_days").size().to_dict() == dict(V5.EXPECTED_REFERENCE_COUNTS)
    assert reference["issue_wtemp_observed"].all()
    assert reference["reportable_primary"].all()


def test_production_train_validation_counts_are_raw_admissible_and_disjoint(
    pinned_inputs: tuple[V5.RawLabelPanel, tuple[str, ...], pd.DataFrame],
) -> None:
    raw, _stations, _reference = pinned_inputs
    training_mask, validation_mask = V5.period_masks(raw.frame)
    assert training_mask.sum() == 438_240
    assert validation_mask.sum() == 87_720
    assert not (training_mask & validation_mask).any()
    for horizon in V5.HORIZONS:
        examples = V5.build_raw_example_registry(raw, horizon)
        for split in ("train", "val"):
            subset = examples[examples["split"].eq(split)]
            assert len(subset) == V5.EXPECTED_EXAMPLE_COUNTS[split][horizon]
            assert subset["issue_wtemp_observed"].all()
            assert subset["target_wtemp_observed"].all()
            assert np.isfinite(subset["y"]).all()


def test_42_legacy_nonformal_keys_are_rejected(
    pinned_inputs: tuple[V5.RawLabelPanel, tuple[str, ...], pd.DataFrame],
) -> None:
    raw, _stations, reference = pinned_inputs
    rejected_by_horizon: dict[int, int] = {}
    for horizon in V5.HORIZONS:
        candidates = V5.build_raw_example_registry(raw, horizon)
        candidates = candidates[candidates["split"].eq("confirm")]
        identity = ["site_id", "issue_date", "target_date"]
        candidate_keys = set(candidates[identity].itertuples(index=False, name=None))
        ref_h = reference[reference["lead_days"].eq(horizon)]
        formal_keys = set(ref_h[identity].itertuples(index=False, name=None))
        assert not formal_keys - candidate_keys
        rejected_by_horizon[horizon] = len(candidate_keys - formal_keys)
    assert rejected_by_horizon == {1: 22, 3: 15, 7: 5}
    assert sum(rejected_by_horizon.values()) == 42

    legacy = pd.read_parquet(V5.LEGACY_KEY_REGISTRY)
    with pytest.raises(AssertionError, match="schema/order"):
        V5.validate_reference_registry(legacy, require_production_counts=False)


def test_production_evaluation_binding_is_two_sided_and_rejects_all_42(
    pinned_inputs: tuple[V5.RawLabelPanel, tuple[str, ...], pd.DataFrame],
) -> None:
    raw, _stations, reference = pinned_inputs
    rejected = 0
    for horizon in V5.HORIZONS:
        candidates = V5.build_raw_example_registry(raw, horizon)
        candidates = candidates[candidates["split"].eq("confirm")].reset_index(drop=True)
        features = pd.DataFrame(
            np.ones((len(candidates), len(V5.FROZEN_BASE_FEATURE_COLUMNS)), dtype=np.float64),
            columns=V5.FROZEN_BASE_FEATURE_COLUMNS,
        )
        candidates = pd.concat([candidates, features], axis=1)
        bound = V5.bind_exact_evaluation_rows(
            candidates,
            reference,
            horizon,
            V5.FROZEN_BASE_FEATURE_COLUMNS,
        )
        assert len(bound) == V5.EXPECTED_REFERENCE_COUNTS[horizon]
        assert set(bound["key_id"]) == set(
            reference.loc[reference["lead_days"].eq(horizon), "key_id"]
        )
        rejected += bound.attrs["raw_admissible_candidates_outside_formal_registry"]
    assert rejected == 42


@pytest.mark.parametrize("bad_value", ["False", 0, 1])
def test_raw_observed_masks_reject_truthy_or_string_coercion(bad_value: object) -> None:
    flagged = V5.add_raw_observed_flags(_synthetic_source())
    attacked = flagged.copy()
    attacked["WTEMP_observed"] = bad_value
    with pytest.raises(TypeError, match="bool dtype"):
        V5.make_raw_label_panel(attacked, source_sha256=SYNTHETIC_SHA256)


@pytest.mark.parametrize("column", ["issue_wtemp_observed", "history_H100", "reportable_primary"])
@pytest.mark.parametrize("bad_value", ["False", 1])
def test_reference_masks_reject_truthy_or_string_coercion(
    column: str,
    bad_value: object,
) -> None:
    reference = _reference_rows([pd.Timestamp("2021-01-01")])
    reference[column] = bad_value
    with pytest.raises(TypeError, match="bool dtype"):
        V5.validate_reference_registry(reference, require_production_counts=False)


@pytest.mark.parametrize("bad_lead", [1.0, 1.5])
def test_reference_lead_must_be_exact_integer(bad_lead: float) -> None:
    reference = _reference_rows([pd.Timestamp("2021-01-01")])
    reference["lead_days"] = bad_lead
    with pytest.raises(TypeError, match="signed-integer"):
        V5.validate_reference_registry(reference, require_production_counts=False)


@pytest.mark.parametrize("bad_site", ["１２３４５６７８", "0100000a", "1000001", " 01000001"])
def test_reference_site_must_be_ascii_digits_of_exact_length(bad_site: str) -> None:
    reference = _reference_rows([pd.Timestamp("2021-01-01")])
    reference["site_id"] = bad_site
    with pytest.raises(ValueError, match="ASCII station"):
        V5.validate_reference_registry(reference, require_production_counts=False)


def test_reference_dates_are_exact_naive_midnight_and_target_chronology() -> None:
    reference = _reference_rows([pd.Timestamp("2021-01-01")])
    attacked = reference.copy()
    attacked["issue_date"] = "2021-01-01"
    with pytest.raises(TypeError, match=r"datetime64\[ns\]"):
        V5.validate_reference_registry(attacked, require_production_counts=False)

    attacked = reference.copy()
    attacked["issue_date"] += pd.Timedelta(hours=1)
    attacked["key_id"] = _canonical_key(
        SITE, attacked.loc[0, "issue_date"], attacked.loc[0, "target_date"], 1
    )
    with pytest.raises(ValueError, match="midnight"):
        V5.validate_reference_registry(attacked, require_production_counts=False)

    attacked = reference.copy()
    attacked["issue_date"] = attacked["issue_date"].dt.tz_localize("UTC")
    with pytest.raises(TypeError, match="timezone-naive"):
        V5.validate_reference_registry(attacked, require_production_counts=False)

    attacked = reference.copy()
    attacked["target_date"] += pd.Timedelta(days=1)
    with pytest.raises(AssertionError, match="target_date"):
        V5.validate_reference_registry(attacked, require_production_counts=False)


def test_reference_history_count_fraction_types_and_arithmetic_are_exact() -> None:
    reference = _reference_rows([pd.Timestamp("2021-01-01")])
    attacked = reference.copy()
    attacked["n_observed_wtemp_7d"] = 7.0
    with pytest.raises(TypeError, match="signed-integer"):
        V5.validate_reference_registry(attacked, require_production_counts=False)
    attacked = reference.copy()
    attacked["fraction_observed_wtemp_7d"] = 0.9
    with pytest.raises(AssertionError, match="arithmetic"):
        V5.validate_reference_registry(attacked, require_production_counts=False)


def test_every_raw_mask_is_created_before_imputation_and_survives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, imputed, climatology = _synthetic_pipeline(monkeypatch)
    assert set(V5.OBSERVED_COLUMNS).issubset(raw.frame.columns)
    V5.assert_observed_masks_survive_transform(raw, imputed.frame)
    missing_date = pd.Timestamp("2006-01-26")
    imputed_row = imputed.frame[imputed.frame["DATE"].eq(missing_date)].iloc[0]
    assert np.isfinite(imputed_row["TEMP"])
    assert not bool(imputed_row["TEMP_observed"])

    table, columns = V5.build_observed_feature_table(raw, imputed, climatology, 1)
    assert columns == V5.FROZEN_BASE_FEATURE_COLUMNS
    feature_row = table[table["issue_date"].eq(missing_date)].iloc[0]
    assert feature_row["TEMP_observed_lag0"] == 0.0
    assert np.isfinite(feature_row["TEMP_lag0"])


def test_missing_raw_issue_and_target_are_excluded(monkeypatch: pytest.MonkeyPatch) -> None:
    raw, imputed, climatology = _synthetic_pipeline(monkeypatch)
    table, _columns = V5.build_observed_feature_table(raw, imputed, climatology, 1)
    missing_issue = pd.Timestamp("2006-01-21")
    issue_whose_target_is_missing = missing_issue - pd.Timedelta(days=1)
    assert missing_issue not in set(table["issue_date"])
    assert issue_whose_target_is_missing not in set(table["issue_date"])
    assert table["issue_wtemp_observed"].all()
    assert table["target_wtemp_observed"].all()


def test_wrapper_constructor_defensive_copy_and_raw_mutation_attack() -> None:
    flagged = V5.add_raw_observed_flags(_synthetic_source())
    dummy = "0" * 64
    with pytest.raises(TypeError, match="issued only"):
        V5.RawLabelPanel(flagged, dummy, dummy, dummy, dummy, dummy)

    original = flagged.copy(deep=True)
    raw = V5.make_raw_label_panel(original, source_sha256=SYNTHETIC_SHA256)
    original.loc[0, "WTEMP"] = 999.0
    assert raw.frame.loc[0, "WTEMP"] != 999.0
    exposed = raw.frame
    exposed.loc[0, "WTEMP"] = 888.0
    assert raw.frame.loc[0, "WTEMP"] != 888.0

    internal = object.__getattribute__(raw, "_frame")
    internal.loc[0, "WTEMP"] += 0.125
    with pytest.raises(ValueError, match="full content changed"):
        _ = raw.frame


def test_wrapper_source_seal_and_imputed_content_mutation_attack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, imputed, _climatology = _synthetic_pipeline(monkeypatch)
    object.__setattr__(raw, "_source_sha256", "0" * 64)
    with pytest.raises(ValueError, match="source/content seal"):
        _ = raw.frame

    raw, imputed, _climatology = _synthetic_pipeline(monkeypatch)
    internal = object.__getattribute__(imputed, "_frame")
    internal.loc[0, "TEMP"] += 1.0
    with pytest.raises(ValueError, match="full content changed"):
        _ = imputed.frame


def test_imputed_as_raw_attack_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    raw, imputed, climatology = _synthetic_pipeline(monkeypatch)
    with pytest.raises(TypeError, match="RawLabelPanel"):
        V5.build_observed_feature_table(imputed, imputed, climatology, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="imputed panel cannot be used"):
        V5.make_raw_label_panel(imputed.frame, source_sha256=raw.source_sha256)


@pytest.mark.parametrize("variable", ["WTEMP", "TEMP"])
def test_imputer_cannot_change_any_observed_value(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
) -> None:
    raw, imputed, _climatology = _synthetic_pipeline(monkeypatch)

    class ObservedValueMutatingImputer:
        def transform(self, _frame: pd.DataFrame) -> pd.DataFrame:
            attacked = imputed.frame
            observed_index = attacked.index[attacked[f"{variable}_observed"]][0]
            attacked.loc[observed_index, variable] += 0.25
            return attacked

    with pytest.raises(AssertionError, match=f"observed {variable} value"):
        V5.impute_feature_panel(raw, ObservedValueMutatingImputer())  # type: ignore[arg-type]


def test_evaluation_binding_is_exact_and_rejects_missing_formal_key() -> None:
    issues = [pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-02")]
    reference = _reference_rows(issues)
    table = _candidate_table([*issues, pd.Timestamp("2021-01-03")])
    bound = V5.bind_exact_evaluation_rows(
        table,
        reference,
        1,
        V5.FROZEN_BASE_FEATURE_COLUMNS,
    )
    assert len(bound) == 2
    assert set(bound["key_id"]) == set(reference["key_id"])
    assert bound.attrs["raw_admissible_candidates_outside_formal_registry"] == 1

    with pytest.raises(AssertionError, match="formal keys missing"):
        V5.bind_exact_evaluation_rows(
            table.iloc[[0, 2]],
            reference,
            1,
            V5.FROZEN_BASE_FEATURE_COLUMNS,
        )
    shifted = reference.copy()
    shifted["y_true"] += 1.0
    with pytest.raises(AssertionError, match="targets differ"):
        V5.bind_exact_evaluation_rows(
            table,
            shifted,
            1,
            V5.FROZEN_BASE_FEATURE_COLUMNS,
        )


def test_future_registry_rejects_duplicate_identities_and_public_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, imputed, climatology = _synthetic_pipeline(monkeypatch)
    table, _columns = V5.build_observed_feature_table(raw, imputed, climatology, 1)
    identities = table.head(3)[["site_id", "issue_date"]]
    duplicated = pd.concat([identities, identities.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="exactly unique"):
        V5.build_raw_future_registry(raw, duplicated, 1)
    dummy = "0" * 64
    with pytest.raises(TypeError, match="issued only"):
        V5.RawFutureRegistry(pd.DataFrame(), 1, dummy, dummy, dummy, dummy, dummy)


def test_future_registry_exact_grid_masks_and_mutation_revalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, imputed, climatology = _synthetic_pipeline(monkeypatch)
    table, _columns = V5.build_observed_feature_table(raw, imputed, climatology, 3)
    identities = table.head(4)[["site_id", "issue_date"]]
    future = V5.build_raw_future_registry(raw, identities, 3)
    assert len(future.frame) == len(identities) * 3
    assert set(future.frame["step"]) == {1, 2, 3}
    for variable in V5.METEOROLOGY_VARIABLES:
        assert future.frame[f"{variable}_observed"].dtype == np.dtype("bool")
        assert np.array_equal(
            future.frame[f"{variable}_observed"].to_numpy(copy=True),
            np.isfinite(future.frame[f"{variable}_raw"].to_numpy(copy=True)),
        )
    exposed = future.frame
    exposed.loc[0, "TEMP_raw"] += 3.0
    assert not future.frame.equals(exposed)

    internal = object.__getattribute__(future, "_frame")
    internal.loc[0, "TEMP_raw"] += 0.25
    with pytest.raises(ValueError, match="full content changed"):
        _ = future.frame


@pytest.mark.parametrize("attack", ["step", "valid_date", "mask", "source"])
def test_future_registry_rejects_grid_and_lineage_forgery(
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    raw, imputed, climatology = _synthetic_pipeline(monkeypatch)
    table, _columns = V5.build_observed_feature_table(raw, imputed, climatology, 1)
    future = V5.build_raw_future_registry(raw, table.head(2)[["site_id", "issue_date"]], 1)
    internal = object.__getattribute__(future, "_frame")
    if attack == "step":
        internal.loc[0, "step"] = 2
    elif attack == "valid_date":
        internal.loc[0, "valid_date"] += pd.Timedelta(hours=1)
    elif attack == "mask":
        internal["TEMP_observed"] = "False"
    else:
        object.__setattr__(future, "_source_sha256", "0" * 64)
    with pytest.raises((TypeError, ValueError), match="step|midnight|bool dtype|source"):
        _ = future.frame


def test_f0_exact_allowlist_and_no_future_registry_capability() -> None:
    base = _base_table([pd.Timestamp("2006-01-01")], ["train"])
    with pytest.raises(ValueError, match="F0 cannot receive"):
        V5.materialize_forcing_features(
            base,
            V5.FROZEN_BASE_FEATURE_COLUMNS,
            arm="F0",
            horizon=1,
            raw_future_registry=object(),  # type: ignore[arg-type]
            meteorology_climatologies={},
        )
    clean, columns = V5.materialize_forcing_features(
        base,
        V5.FROZEN_BASE_FEATURE_COLUMNS,
        arm="F0",
        horizon=1,
        raw_future_registry=None,
        meteorology_climatologies=None,
    )
    assert columns == V5.FROZEN_BASE_FEATURE_COLUMNS
    assert clean["forcing_substitution_count"].eq(0).all()

    attacked = base.copy()
    attacked["TEMP_fut1"] = 1.0
    with pytest.raises(ValueError, match="future namespace"):
        V5.materialize_forcing_features(
            attacked,
            V5.FROZEN_BASE_FEATURE_COLUMNS,
            arm="F0",
            horizon=1,
            raw_future_registry=None,
            meteorology_climatologies=None,
        )
    with pytest.raises(ValueError, match="exact frozen"):
        V5.materialize_forcing_features(
            base,
            (*V5.FROZEN_BASE_FEATURE_COLUMNS[1:], V5.FROZEN_BASE_FEATURE_COLUMNS[0]),
            arm="F0",
            horizon=1,
            raw_future_registry=None,
            meteorology_climatologies=None,
        )


def test_f3_raw_future_values_and_masks_are_not_mutated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, imputed, climatology = _synthetic_pipeline(monkeypatch)
    table, columns = V5.build_observed_feature_table(raw, imputed, climatology, 1)
    table = table.head(30).copy()
    future = V5.build_raw_future_registry(raw, table[["site_id", "issue_date"]], 1)
    before = future.frame
    training = np.ones(len(raw.frame), dtype=np.bool_)
    met_clims = V5.V4.fit_meteorological_climatologies(raw.frame, training, (SITE,))
    materialized, model_columns = V5.materialize_forcing_features(
        table,
        columns,
        arm="F3_full",
        horizon=1,
        raw_future_registry=future,
        meteorology_climatologies=met_clims,
    )
    pd.testing.assert_frame_equal(future.frame, before)
    assert model_columns[: len(columns)] == columns
    assert model_columns[len(columns) :] == V5._future_feature_columns(1)
    assert np.isfinite(materialized[list(model_columns)].to_numpy(dtype=np.float64)).all()
    V5.validate_substitution_accounting(materialized, arm="F3_full", horizon=1)


@pytest.mark.parametrize("attack", ["float", "negative", "sum", "bound", "f0_nonzero"])
def test_substitution_accounting_rejects_forgery(attack: str) -> None:
    base = _base_table([pd.Timestamp("2006-01-01")], ["train"])
    frame, _columns = V5.materialize_forcing_features(
        base,
        V5.FROZEN_BASE_FEATURE_COLUMNS,
        arm="F0",
        horizon=1,
        raw_future_registry=None,
        meteorology_climatologies=None,
    )
    attacked = frame.copy()
    if attack == "float":
        attacked[V5.SUBSTITUTION_COLUMNS[0]] = 0.0
    elif attack == "negative":
        attacked[V5.SUBSTITUTION_COLUMNS[0]] = -1
    elif attack == "sum":
        attacked[V5.SUBSTITUTION_COLUMNS[0]] = 1
    elif attack == "bound":
        for column in V5.SUBSTITUTION_COLUMNS:
            attacked[column] = 2
        attacked["forcing_substitution_count"] = 10
    else:
        attacked[V5.SUBSTITUTION_COLUMNS[0]] = 1
        attacked["forcing_substitution_count"] = 1
    with pytest.raises((TypeError, ValueError), match="integer|non-negative|sum|bound|F0"):
        V5.validate_substitution_accounting(attacked, arm="F0", horizon=1)


def test_tree_fit_binds_actual_matrices_validation_params_and_best_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    issues = [
        pd.Timestamp("2006-01-01"),
        pd.Timestamp("2006-01-02"),
        pd.Timestamp("2016-01-01"),
        pd.Timestamp("2016-01-02"),
    ]
    base = _base_table(issues, ["train", "train", "val", "val"], y=[10, 11, 20, 21])
    table, model_columns = V5.materialize_forcing_features(
        base,
        V5.FROZEN_BASE_FEATURE_COLUMNS,
        arm="F0",
        horizon=1,
        raw_future_registry=None,
        meteorology_climatologies=None,
    )
    evaluation = table.iloc[[0]].copy()
    evaluation["issue_date"] = pd.Series([pd.Timestamp("2021-01-01")])
    evaluation["target_date"] = pd.Series([pd.Timestamp("2021-01-02")])
    evaluation["y_true"] = np.asarray([12.0], dtype=np.float64)
    evaluation["key_id"] = _canonical_key(
        SITE, evaluation.loc[0, "issue_date"], evaluation.loc[0, "target_date"], 1
    )
    anchor = F.DampedPersistenceAnchor(phi={SITE: 0.9}, fit_stations=(SITE,))
    preprocessing = V5.ObservedPreprocessing(
        imputer=None,  # type: ignore[arg-type]
        water_climatology=None,  # type: ignore[arg-type]
        damped_anchor=anchor,
        meteorology_climatologies={},
        station_ids=(SITE,),
        training_row_count=2,
        validation_row_count=2,
    )
    captured: dict[str, object] = {}

    class Fitted:
        best_iteration_ = 17

        def predict(self, rows: pd.DataFrame, *, num_threads: int) -> np.ndarray:
            assert num_threads == 1
            return np.full(len(rows), 11.0)

    def fake_fit(Xtr, ytr, Xval, yval, objective, **kwargs):
        captured.update(
            {
                "Xtr": Xtr.to_numpy(dtype=np.float64).copy(),
                "ytr": np.asarray(ytr).copy(),
                "Xval": Xval.to_numpy(dtype=np.float64).copy(),
                "yval": np.asarray(yval).copy(),
                "objective": objective,
                **kwargs,
            }
        )
        return Fitted()

    monkeypatch.setattr(V5, "_lgb_fit", fake_fit)
    cell = V5.Cell("F0", "LightGBM", 1)
    frame, evidence = V5.fit_tree_cell(
        table,
        evaluation,
        model_columns,
        preprocessing,
        cell,
    )
    assert captured["Xtr"].shape == (2, 210)
    assert captured["Xval"].shape == (2, 210)
    np.testing.assert_array_equal(captured["Xtr"][:, 0], [1.0, 2.0])
    np.testing.assert_array_equal(captured["Xval"][:, 0], [3.0, 4.0])
    assert captured["n_est"] == V5.BEST_ITER_UPPER_BOUND[1]
    assert evidence["validation_role"] == "actual_lgb_eval_set_not_training_tail"
    assert evidence["best_iteration"] == 17
    assert evidence["resolved_lightgbm_parameters"]["seed"] == 0
    assert evidence["resolved_lightgbm_parameters"]["deterministic"] is True
    for split in ("train", "validation", "evaluation"):
        assert len(evidence["datasets"][split]["actual_feature_matrix_sha256"]) == 64
        assert len(evidence["datasets"][split]["ordered_identity_and_y_sha256"]) == 64
    assert len(frame) == 1

    formal_reference = pd.DataFrame(
        {
            "key_id": evaluation["key_id"],
            "site_id": [SITE],
            "issue_date": pd.Series([pd.Timestamp("2021-01-01")]),
            "target_date": pd.Series([pd.Timestamp("2021-01-02")]),
            "lead_days": np.asarray([1], dtype=np.int16),
            "y_true": np.asarray([12.0], dtype=np.float64),
        }
    )
    V5.validate_v5_shard(frame, cell, formal_reference)
    attacked = frame.copy()
    attacked["key_id"] = "0" * 64
    with pytest.raises(AssertionError, match="keys differ"):
        V5.validate_v5_shard(attacked, cell, formal_reference)


def test_dataset_matrix_and_label_hashes_are_ordered_and_mutation_sensitive() -> None:
    table = _base_table(
        [pd.Timestamp("2006-01-01"), pd.Timestamp("2006-01-02")],
        ["train", "train"],
    )
    first = V5._ordered_dataset_binding(
        table,
        label_column="y",
        feature_columns=V5.FROZEN_BASE_FEATURE_COLUMNS,
        horizon=1,
    )
    attacked_feature = table.copy()
    attacked_feature.loc[0, V5.FROZEN_BASE_FEATURE_COLUMNS[0]] += 0.5
    second = V5._ordered_dataset_binding(
        attacked_feature,
        label_column="y",
        feature_columns=V5.FROZEN_BASE_FEATURE_COLUMNS,
        horizon=1,
    )
    assert first["actual_feature_matrix_sha256"] != second["actual_feature_matrix_sha256"]
    assert first["ordered_identity_and_y_sha256"] == second["ordered_identity_and_y_sha256"]
    attacked_y = table.copy()
    attacked_y.loc[0, "y"] += 0.5
    third = V5._ordered_dataset_binding(
        attacked_y,
        label_column="y",
        feature_columns=V5.FROZEN_BASE_FEATURE_COLUMNS,
        horizon=1,
    )
    assert first["ordered_identity_and_y_sha256"] != third["ordered_identity_and_y_sha256"]
    reversed_rows = table.iloc[::-1].reset_index(drop=True)
    fourth = V5._ordered_dataset_binding(
        reversed_rows,
        label_column="y",
        feature_columns=V5.FROZEN_BASE_FEATURE_COLUMNS,
        horizon=1,
    )
    assert first["ordered_identity_sha256"] != fourth["ordered_identity_sha256"]


def test_training_lineage_is_one_identity_per_horizon_across_all_cells() -> None:
    table = _base_table(
        [pd.Timestamp("2006-01-01"), pd.Timestamp("2016-01-01")],
        ["train", "val"],
    )
    lineage = V5._training_lineage(table, 1)
    cells = [cell for cell in V5.fixed_cells() if cell.horizon == 1]
    per_cell = {
        (cell.arm, cell.model): (
            lineage["train"]["key_sha256"],
            lineage["val"]["key_sha256"],
        )
        for cell in cells
    }
    assert len(set(per_cell.values())) == 1
    assert len(per_cell) == 4
    arm_table = table.copy()
    arm_table["future_feature"] = [3.0, 4.0]
    V5.assert_fit_identities_unchanged(table, arm_table)
    attacked = arm_table.iloc[::-1].reset_index(drop=True)
    with pytest.raises(AssertionError, match="changed training/validation"):
        V5.assert_fit_identities_unchanged(table, attacked)


def test_snapshot_revalidation_detects_toctou(tmp_path: Path) -> None:
    source = tmp_path / "dependency.txt"
    source.write_bytes(b"first")
    bound = V5._read_stable_regular(source, root=tmp_path, label="synthetic dependency")
    V5._revalidate_snapshot({"dependency": bound}, root=tmp_path)
    source.write_bytes(b"second")
    with pytest.raises(RuntimeError, match="changed after capture"):
        V5._revalidate_snapshot({"dependency": bound}, root=tmp_path)


def test_bundle_transaction_publishes_complete_bundle_with_one_commit(tmp_path: Path) -> None:
    destination = tmp_path / "bundle"
    expected: dict[str, dict[str, object]] = {}
    with V5._BundleTransaction(
        destination,
        allowed_root=tmp_path,
        expected_destination_name="bundle",
    ) as transaction:
        expected["shards/a.bin"] = transaction.write_bytes("shards/a.bin", b"alpha")
        expected["manifest.json"] = transaction.write_bytes("manifest.json", b"{}\n")
        transaction.commit(expected, precommit_check=lambda: None)
    assert (destination / "shards" / "a.bin").read_bytes() == b"alpha"
    assert (destination / "manifest.json").read_bytes() == b"{}\n"
    assert not any(path.name.startswith(".bundle.staging.") for path in tmp_path.iterdir())
    assert not (tmp_path / ".bundle.create.lock").exists()


def test_bundle_root_atomic_swap_fails_and_cleans_only_anchored_owned_inode(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    root = parent / "root"
    root.mkdir()
    moved_root = parent / "root-moved"
    destination = root / "bundle"
    transaction = V5._BundleTransaction(
        destination,
        allowed_root=root,
        expected_destination_name="bundle",
    )
    try:
        binding = transaction.write_bytes("manifest.json", b"{}\n")
        root.rename(moved_root)
        root.mkdir()
        marker = root / "attacker-marker"
        marker.write_bytes(b"do-not-delete")
        with pytest.raises(RuntimeError, match="no longer names the anchored directory"):
            transaction.commit(
                {"manifest.json": binding},
                precommit_check=lambda: None,
            )
    finally:
        transaction.close()
    assert marker.read_bytes() == b"do-not-delete"
    assert not (root / "bundle").exists()
    assert not (moved_root / "bundle").exists()
    assert not any(path.name.startswith(".bundle.staging.") for path in moved_root.iterdir())
    assert not (moved_root / ".bundle.create.lock").exists()


def test_bundle_transaction_rejects_preexisting_lock_external_traversal_and_symlink(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="destination exists"):
        V5._BundleTransaction(
            existing,
            allowed_root=tmp_path,
            expected_destination_name="existing",
        )
    with pytest.raises(ValueError, match="escapes"):
        V5._BundleTransaction(
            tmp_path.parent / "external",
            allowed_root=tmp_path,
            expected_destination_name="external",
        )
    with pytest.raises(ValueError, match=r"\.\."):
        V5._BundleTransaction(
            tmp_path / ".." / "escape",
            allowed_root=tmp_path,
            expected_destination_name="escape",
        )

    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "linked-root"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symbolic link"):
        V5._BundleTransaction(
            link / "bundle",
            allowed_root=link,
            expected_destination_name="bundle",
        )

    first = V5._BundleTransaction(
        tmp_path / "locked",
        allowed_root=tmp_path,
        expected_destination_name="locked",
    )
    try:
        with pytest.raises(RuntimeError, match="another create-only publication"):
            V5._BundleTransaction(
                tmp_path / "locked",
                allowed_root=tmp_path,
                expected_destination_name="locked",
            )
    finally:
        first.close()


@pytest.mark.parametrize("attack", ["forged_hash", "extra", "symlink", "rename_failure"])
def test_bundle_failure_never_publishes_partial_formal_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    destination = tmp_path / f"bundle-{attack}"
    transaction = V5._BundleTransaction(
        destination,
        allowed_root=tmp_path,
        expected_destination_name=destination.name,
    )
    try:
        binding = transaction.write_bytes("manifest.json", b"{}\n")
        expected = {"manifest.json": dict(binding)}
        if attack == "forged_hash":
            expected["manifest.json"]["sha256"] = "0" * 64
        elif attack == "extra":
            transaction.write_bytes("extra.bin", b"extra")
        elif attack == "symlink":
            assert transaction.staging is not None
            (transaction.staging / "evil-link").symlink_to(tmp_path / "outside")
        else:
            monkeypatch.setattr(
                V5,
                "_rename_directory_noreplace_at",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected rename")),
            )
        with pytest.raises(RuntimeError):
            transaction.commit(expected, precommit_check=lambda: None)
    finally:
        transaction.close()
    assert not destination.exists()
    assert not any(
        path.name.startswith(f".{destination.name}.staging.") for path in tmp_path.iterdir()
    )


def test_versioned_names_and_no_formal_output_exist() -> None:
    names = {V5.shard_filename(cell) for cell in V5.fixed_cells()}
    assert len(names) == 12
    assert all(name.endswith("_v5_observed.parquet") for name in names)
    assert V5.MANIFEST_FILENAME.endswith("_v5_observed.json")
    assert not os.path.lexists(V5.DEFAULT_OUTPUT_DIR)

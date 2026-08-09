"""Focused adversarial tests for scratch-only semantic data registries v4."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

S = importlib.import_module("scripts.final.build_semantic_data_registries_v4")

SITES = ("01000001", "01000002")
LEGACY = ("n00", "n01")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(path: Path, relative: str) -> dict[str, object]:
    return {"path": relative, "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _station_registry() -> pd.DataFrame:
    rows = []
    for site, legacy in zip(SITES, LEGACY, strict=True):
        rows.append(
            {
                "site_no": site,
                "legacy_site_id": legacy,
                "ok": "True",
                "wtemp_cov": "1.0",
                "flow_cov": "1.0",
                "wtemp_cov_test": "1.0",
                "flow_cov_test": "1.0",
                "wlevel_cov": "0.0",
                "n_days": "24",
                "station_nm": f"Synthetic {site}",
                "lat": "40.0",
                "lon": "-75.0",
                "state": "PA",
                "huc_cd": "02040101",
                "huc2": "02",
                "drain_area_va": "1.0",
                "huc_metadata_status": "SYNTHETIC",
            }
        )
    return pd.DataFrame(rows, columns=S.STATION_REGISTRY_COLUMNS)


def _panel(dates: pd.DatetimeIndex, site_ids: tuple[str, ...], *, offset: float) -> pd.DataFrame:
    rows = []
    for site_index, site in enumerate(site_ids):
        for day_index, timestamp in enumerate(dates):
            value = offset + 10.0 * site_index + day_index
            rows.append(
                {
                    "DATE": timestamp,
                    "site_id": site,
                    "WTEMP": value,
                    "FLOW": value + 100.0,
                    "WLEVEL": np.nan if day_index % 2 == 0 else value + 1.0,
                    "TEMP": value + 2.0,
                    "PRCP": value / 10.0,
                    "WDSP": value / 20.0,
                    "RHMEAN": value + 50.0,
                    "DH": value + 200.0,
                }
            )
    return pd.DataFrame(rows, columns=S.PANEL_COLUMNS)


def _raw_union(development: pd.DataFrame, evaluation: pd.DataFrame) -> pd.DataFrame:
    alias = dict(zip(LEGACY, SITES, strict=True))
    dev = development.copy()
    dev["site_id"] = dev["site_id"].map(alias)
    union = pd.concat([dev, evaluation], ignore_index=True)
    return (
        union.drop_duplicates(["site_id", "DATE"], keep="last")
        .sort_values(["site_id", "DATE"], kind="mergesort")
        .reset_index(drop=True)
    )


def _independent_counts(raw: pd.DataFrame, config: S.BuildConfig) -> dict[str, dict[str, int]]:
    result = {split: {str(lead): 0 for lead in S.LEADS} for split in S.SPLITS}
    ranges = {
        "train": (pd.Timestamp(config.train_start), pd.Timestamp(config.train_end)),
        "validation": (
            pd.Timestamp(config.validation_start),
            pd.Timestamp(config.validation_end),
        ),
    }
    for _site, group in raw.groupby("site_id"):
        values = group.set_index("DATE")["WTEMP"]
        for split, (lower, upper) in ranges.items():
            for lead in S.LEADS:
                for issue, value in values.items():
                    target = issue + pd.Timedelta(days=lead)
                    if (
                        lower <= issue <= upper
                        and lower <= target <= upper
                        and np.isfinite(value)
                        and target in values.index
                        and np.isfinite(values.loc[target])
                    ):
                        result[split][str(lead)] += 1
    return result


def _write_canonical_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(S._canonical_json_bytes(document))


def _make_evidence(root: Path) -> tuple[S.RegistryInputPaths, S.BuildConfig, pd.DataFrame]:
    station_path = root / S.SOURCE_RELATIVE_PATHS["station_registry"]
    development_path = root / S.SOURCE_RELATIVE_PATHS["development_panel"]
    evaluation_path = root / S.SOURCE_RELATIVE_PATHS["evaluation_panel"]
    report_path = root / S.SOURCE_RELATIVE_PATHS["defect_authority_report"]
    key_path = (
        root / "outputs/final/information_regime_key_registries_v4/"
        "information_regime_key_registries_v4_manifest.json"
    )
    defect_path = (
        root / "outputs/final/preprocessing_lineage_defect_authority_v1/"
        "preprocessing_lineage_defect_authority_v1_manifest.json"
    )
    station_path.parent.mkdir(parents=True, exist_ok=True)
    development_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    _station_registry().to_csv(station_path, index=False)

    development = _panel(
        pd.date_range("2006-01-01", "2006-01-14", freq="D"),
        LEGACY,
        offset=0.0,
    )
    evaluation = _panel(
        pd.date_range("2006-01-12", "2006-01-24", freq="D"),
        SITES,
        offset=1_000.0,
    )
    development.loc[
        (development["site_id"].eq("n00")) & (development["DATE"].eq(pd.Timestamp("2006-01-04"))),
        "WTEMP",
    ] = np.nan
    evaluation.loc[
        (evaluation["site_id"].eq("01000002"))
        & (evaluation["DATE"].eq(pd.Timestamp("2006-01-18"))),
        "WTEMP",
    ] = np.nan
    evaluation.loc[evaluation["DATE"].eq(pd.Timestamp("2006-01-20")), "TEMP"] = np.nan
    development.to_parquet(development_path, index=False)
    evaluation.to_parquet(evaluation_path, index=False)

    key_document = {
        "artifact_id": "thermoroute-information-regime-key-registries-v4-phase1",
        "status": "PHASE1_KEY_REGISTRIES_ONLY_NOT_MODEL_SCORE_AUTHORITY",
        "registry_publication_scope": "PHASE1_CONTRACT_ONLY",
        "execution_authorized_by_protocol": False,
        "model_or_score_output_published": False,
        "model_scores_read_or_accepted": False,
        "input_bindings": {
            "station_registry_v1": _binding(
                station_path,
                S.SOURCE_RELATIVE_PATHS["station_registry"],
            ),
            "raw_holdout_panel": _binding(
                evaluation_path,
                S.SOURCE_RELATIVE_PATHS["evaluation_panel"],
            ),
        },
    }
    _write_canonical_json(key_path, key_document)

    config = S.BuildConfig.synthetic(
        train_start=date(2006, 1, 1),
        train_end=date(2006, 1, 10),
        validation_start=date(2006, 1, 11),
        validation_end=date(2006, 1, 20),
    )
    raw = _raw_union(development, evaluation)
    counts = _independent_counts(raw, config)
    missing = {
        variable: int((~np.isfinite(raw[variable].to_numpy(dtype=float))).sum())
        for variable in S.PANEL_VARIABLES
        if variable != "WLEVEL"
    }
    report_document = {
        "artifact_id": "thermoroute-preprocessing-lineage-defect-authority-v1",
        "status": "SCIENTIFIC_RESULTS_WITHDRAWN_PENDING_OBSERVED_LINEAGE_RERUN",
        "required_versioned_rerun": {
            "raw_observed_flags_must_be_created_before_imputation": True,
            "observed_masks_must_survive_imputation_bit_exactly": True,
            "training_issue_and_target_admissibility_must_be_joined_from_raw_lineage": True,
            "training_key_registries_or_digests_must_be_authority_bound": True,
            "training_must_use_only_2006_2015_rows": True,
            "validation_must_use_only_independent_2016_2017_rows": True,
        },
        "raw_panel_inventory": {
            "combined_rows": len(raw),
            "stations": len(SITES),
            "combined_start": raw["DATE"].min().date().isoformat(),
            "combined_end": raw["DATE"].max().date().isoformat(),
            "combined_missing": missing,
        },
        "training_lineage_inventory": {
            "by_horizon": {
                str(lead): {
                    "train": {"admissible_rows": counts["train"][str(lead)]},
                    "val": {"admissible_rows": counts["validation"][str(lead)]},
                }
                for lead in S.LEADS
            }
        },
    }
    _write_canonical_json(report_path, report_document)
    defect_document = {
        "artifact_id": "thermoroute-preprocessing-lineage-defect-authority-v1",
        "status": "SCIENTIFIC_RESULTS_WITHDRAWN_PENDING_OBSERVED_LINEAGE_RERUN",
        "input_bindings": {
            "score_independent_key_manifest": _binding(
                key_path,
                (
                    "outputs/final/information_regime_key_registries_v4/"
                    "information_regime_key_registries_v4_manifest.json"
                ),
            ),
            "station_registry": _binding(
                station_path,
                S.SOURCE_RELATIVE_PATHS["station_registry"],
            ),
            "training_panel": _binding(
                development_path,
                S.SOURCE_RELATIVE_PATHS["development_panel"],
            ),
            "evaluation_panel": _binding(
                evaluation_path,
                S.SOURCE_RELATIVE_PATHS["evaluation_panel"],
            ),
        },
        "output": {
            "name": report_path.name,
            "sha256": _sha256(report_path),
            "size_bytes": report_path.stat().st_size,
        },
    }
    _write_canonical_json(defect_path, defect_document)
    paths = S.RegistryInputPaths(
        key_authority_manifest=key_path,
        defect_authority_manifest=defect_path,
        defect_authority_report=report_path,
        station_registry=station_path,
        development_panel=development_path,
        evaluation_panel=evaluation_path,
    )
    return paths, config, raw


@pytest.fixture()
def synthetic_evidence(
    tmp_path: Path,
) -> tuple[Path, S.RegistryInputPaths, S.BuildConfig, pd.DataFrame]:
    root = tmp_path / "evidence"
    root.mkdir()
    paths, config, raw = _make_evidence(root)
    return root, paths, config, raw


@pytest.fixture()
def synthetic_bundle(
    synthetic_evidence: tuple[Path, S.RegistryInputPaths, S.BuildConfig, pd.DataFrame],
) -> tuple[S.CandidateBundle, pd.DataFrame, S.BuildConfig]:
    root, paths, config, raw = synthetic_evidence
    return S.build_twice(paths, config=config, evidence_root=root), raw, config


def test_exact_schemas_observedness_overlap_order_digests_and_counts(
    synthetic_bundle: tuple[S.CandidateBundle, pd.DataFrame, S.BuildConfig],
) -> None:
    bundle, raw, config = synthetic_bundle
    assert set(bundle.files) == {S.DAILY_FILENAME, S.TRAINING_FILENAME, S.MANIFEST_FILENAME}
    manifest = bundle.manifest
    assert manifest["status"] == "CANDIDATE_NOT_AUTHORITY"
    assert manifest["formal_authority"] is False
    assert manifest["execution_authorized"] is False
    assert manifest["model_execution_authorized"] is False
    assert manifest["model_predictions_scores_effects_or_contrasts_read"] is False
    digest_contract = manifest["semantic_contract"]["ordered_semantic_digest"]
    assert digest_contract["algorithm"] == "SHA-256"
    assert digest_contract["domain_prefix_utf8"] == "thermoroute-semantic-data-frame-v4\0"
    assert digest_contract["row_order"] == "physical_registry_order"
    assert digest_contract["parquet_metadata_included"] is False
    assert manifest["semantic_contract"]["daily_raw_observed_panel_registry"] == {
        "identity": ["site_id", "DATE"],
        "order": ["site_id", "DATE"],
        "raw_observed_definition": "isfinite(value) in captured source before imputation",
        "imputation_performed": False,
        "overlap_policy": "evaluation_panel_wins",
        "overlap_selection_precedence": ["development_panel", "evaluation_panel"],
        "strict_daily_calendar_per_station": True,
        "mask_digest_scope": "all_8_panel_variables_including_WLEVEL",
        "defect_authority_mask_digest_relationship": (
            "SEPARATE_NOT_EQUAL: the v1 defect authority mask digest covers its seven "
            "model-used variables and omits WLEVEL"
        ),
        "raw_missing_values_preserved": True,
        "raw_missing_values_filled": False,
        "raw_missing_physical_encoding": (
            "nullable Arrow float64; validity bitmap equals the corresponding observed "
            "bool exactly; every valid value is finite"
        ),
    }
    daily_schema = pq.read_schema(io.BytesIO(bundle.files[S.DAILY_FILENAME]))
    training_schema = pq.read_schema(io.BytesIO(bundle.files[S.TRAINING_FILENAME]))
    assert daily_schema.equals(S.DAILY_SCHEMA)
    assert training_schema.equals(S.TRAINING_SCHEMA)
    daily_table = pq.read_table(io.BytesIO(bundle.files[S.DAILY_FILENAME]))
    for variable in S.PANEL_VARIABLES:
        raw_column = daily_table[variable]
        observed_column = daily_table[f"{variable}_observed"]
        valid = raw_column.is_valid().to_numpy(zero_copy_only=False)
        observed = observed_column.to_numpy(zero_copy_only=False)
        assert daily_schema.field(variable).nullable is True
        assert np.array_equal(valid, observed)
        assert raw_column.null_count == int((~observed).sum())
        assert np.isfinite(raw_column.to_numpy(zero_copy_only=False)[valid]).all()
    for column in ("DATE", "site_id", *S.OBSERVED_COLUMNS):
        assert daily_table[column].null_count == 0
        assert daily_schema.field(column).nullable is False
    assert tuple(bundle.daily.columns) == S.DAILY_COLUMNS
    assert tuple(bundle.training_examples.columns) == S.TRAINING_COLUMNS
    pd.testing.assert_frame_equal(
        bundle.daily[list(S.PANEL_COLUMNS)],
        raw[list(S.PANEL_COLUMNS)],
        check_dtype=True,
    )
    for variable in S.PANEL_VARIABLES:
        assert bundle.daily[f"{variable}_observed"].dtype == np.dtype("bool")
        assert np.array_equal(
            bundle.daily[f"{variable}_observed"].to_numpy(),
            np.isfinite(bundle.daily[variable].to_numpy(dtype=float)),
        )
    overlap = bundle.daily[bundle.daily["DATE"].between("2006-01-12", "2006-01-14")]
    assert overlap["WTEMP"].min() >= 1_000.0
    assert manifest["source_panel_inventory"]["overlap_rows"] == 6
    assert (
        manifest["source_panel_inventory"]["overlap_raw_cell_differences_by_variable"]["WTEMP"] == 6
    )
    expected_counts = _independent_counts(raw, config)
    assert manifest["training_example_counts"]["from_materialized_registry"] == expected_counts
    assert (
        manifest["training_example_counts"]["independent_raw_calendar_reconstruction"]
        == expected_counts
    )
    assert manifest["training_example_counts"]["counts_match"] is True
    assert bundle.training_examples["issue_wtemp_observed"].all()
    assert bundle.training_examples["target_wtemp_observed"].all()
    assert manifest["outputs"][S.DAILY_FILENAME]["ordered_semantic_content_sha256"] == (
        S._canonical_frame_sha256(bundle.daily, S.DAILY_COLUMNS)
    )
    assert manifest["outputs"][S.TRAINING_FILENAME][
        "ordered_semantic_content_sha256"
    ] == S._canonical_frame_sha256(bundle.training_examples, S.TRAINING_COLUMNS)


@pytest.mark.parametrize("replacement", ["True", 1])
def test_string_and_numeric_observed_bools_fail_closed(
    synthetic_bundle: tuple[S.CandidateBundle, pd.DataFrame, S.BuildConfig],
    replacement: object,
) -> None:
    bundle, _raw, _config = synthetic_bundle
    attacked = bundle.daily.copy()
    attacked["WTEMP_observed"] = replacement
    with pytest.raises(S.SemanticRegistryError, match="bool dtype"):
        S.validate_daily_registry(attacked)


def test_strict_site_date_nonfinite_and_training_bool_contracts(
    synthetic_bundle: tuple[S.CandidateBundle, pd.DataFrame, S.BuildConfig],
) -> None:
    bundle, _raw, config = synthetic_bundle
    attacked_site = bundle.daily.copy()
    attacked_site.loc[0, "site_id"] = "1000001"
    with pytest.raises(S.SemanticRegistryError, match="station identifier"):
        S.validate_daily_registry(attacked_site)

    attacked_date = bundle.daily.copy()
    attacked_date["DATE"] = attacked_date["DATE"].dt.strftime("%Y-%m-%d")
    with pytest.raises(S.SemanticRegistryError, match=r"datetime64\[ns\]"):
        S.validate_daily_registry(attacked_date)

    attacked_infinite = bundle.daily.copy()
    attacked_infinite.loc[0, "FLOW"] = np.inf
    with pytest.raises(S.SemanticRegistryError, match="infinity"):
        S.validate_daily_registry(attacked_infinite)

    attacked_training = bundle.training_examples.copy()
    attacked_training["issue_wtemp_observed"] = np.int8(1)
    with pytest.raises(S.SemanticRegistryError, match="bool dtype"):
        S.validate_training_registry(attacked_training, daily=bundle.daily, config=config)

    attacked_y = bundle.training_examples.copy()
    attacked_y.loc[0, "y_true"] = np.inf
    with pytest.raises(S.SemanticRegistryError, match="infinity"):
        S.validate_training_registry(attacked_y, daily=bundle.daily, config=config)


def test_arrow_validity_bitmap_and_valid_nan_attacks_fail_closed(
    synthetic_bundle: tuple[S.CandidateBundle, pd.DataFrame, S.BuildConfig],
) -> None:
    bundle, _raw, _config = synthetic_bundle
    table = pq.read_table(io.BytesIO(bundle.files[S.DAILY_FILENAME]))
    flags = table["WTEMP_observed"].to_numpy(zero_copy_only=False).copy()
    flags[0] = ~flags[0]
    attacked_flags = table.set_column(
        table.schema.get_field_index("WTEMP_observed"),
        table.schema.field("WTEMP_observed"),
        pa.array(flags, type=pa.bool_(), from_pandas=False),
    )
    with pytest.raises(S.SemanticRegistryError, match="validity differs from observedness"):
        S._validate_daily_arrow_table(attacked_flags, label="attacked daily")

    raw = table["WTEMP"]
    valid = raw.is_valid().to_numpy(zero_copy_only=False)
    values = raw.to_numpy(zero_copy_only=False).copy()
    selected = int(np.flatnonzero(valid)[0])
    values[selected] = np.nan
    attacked_raw = table.set_column(
        table.schema.get_field_index("WTEMP"),
        table.schema.field("WTEMP"),
        pa.array(
            values,
            mask=~valid,
            type=pa.float64(),
            from_pandas=False,
        ),
    )
    with pytest.raises(S.SemanticRegistryError, match="valid non-finite"):
        S._validate_daily_arrow_table(attacked_raw, label="attacked daily")


def test_duplicate_identity_and_order_attacks_fail_closed(
    synthetic_bundle: tuple[S.CandidateBundle, pd.DataFrame, S.BuildConfig],
) -> None:
    bundle, _raw, config = synthetic_bundle
    duplicate_daily = pd.concat([bundle.daily, bundle.daily.iloc[[0]]], ignore_index=True)
    with pytest.raises(S.SemanticRegistryError, match="duplicates a station-day"):
        S.validate_daily_registry(duplicate_daily)

    reversed_daily = bundle.daily.iloc[::-1].reset_index(drop=True)
    with pytest.raises(S.SemanticRegistryError, match="ordering"):
        S.validate_daily_registry(reversed_daily)

    shifted_calendar = bundle.daily.copy()
    selected_site = shifted_calendar["site_id"].eq(SITES[1])
    shifted_calendar.loc[selected_site, "DATE"] += pd.Timedelta(days=1)
    shifted_calendar = shifted_calendar.sort_values(
        ["site_id", "DATE"],
        kind="mergesort",
    ).reset_index(drop=True)
    with pytest.raises(S.SemanticRegistryError, match="one exact date vector"):
        S.validate_daily_registry(shifted_calendar)

    duplicate_example = pd.concat(
        [bundle.training_examples, bundle.training_examples.iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(S.SemanticRegistryError, match="duplicates training_key_id"):
        S.validate_training_registry(duplicate_example, daily=bundle.daily, config=config)

    reversed_training = bundle.training_examples.iloc[::-1].reset_index(drop=True)
    with pytest.raises(S.SemanticRegistryError, match="order is not"):
        S.validate_training_registry(reversed_training, daily=bundle.daily, config=config)

    changed_key = bundle.training_examples.copy()
    changed_key.loc[0, "training_key_id"] = "0" * 64
    with pytest.raises(S.SemanticRegistryError, match="training_key_id changed"):
        S.validate_training_registry(changed_key, daily=bundle.daily, config=config)


def test_duplicate_source_identity_is_rejected_before_overlap_selection(
    synthetic_evidence: tuple[Path, S.RegistryInputPaths, S.BuildConfig, pd.DataFrame],
) -> None:
    root, paths, _config, _raw = synthetic_evidence
    frame = pd.read_parquet(paths.evaluation_panel)
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    frame.to_parquet(paths.evaluation_panel, index=False)
    bound = S._read_stable_regular(
        paths.evaluation_panel,
        root=root,
        label="duplicated evaluation panel",
    )
    with pytest.raises(S.SemanticRegistryError, match="duplicates a source station-day"):
        S._load_panel(bound, label="evaluation panel")


def test_input_symlink_and_hardlink_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"evidence")
    symlink = tmp_path / "symlink.bin"
    symlink.symlink_to(source)
    with pytest.raises(S.SemanticRegistryError, match="symbolic link"):
        S._read_stable_regular(symlink, root=tmp_path, label="symlink evidence")

    hardlink = tmp_path / "hardlink.bin"
    os.link(source, hardlink)
    with pytest.raises(S.SemanticRegistryError, match="exactly one hard link"):
        S._read_stable_regular(source, root=tmp_path, label="hard-linked evidence")


def test_create_only_writer_and_candidate_manifest(
    synthetic_evidence: tuple[Path, S.RegistryInputPaths, S.BuildConfig, pd.DataFrame],
    tmp_path: Path,
) -> None:
    root, paths, config, _raw = synthetic_evidence
    output_parent = tmp_path / "scratch"
    output_parent.mkdir()
    destination = output_parent / "candidate"
    published, bundle = S.write_candidate(
        paths,
        destination,
        config=config,
        evidence_root=root,
    )
    assert published == destination
    assert {item.name for item in destination.iterdir()} == set(bundle.files)
    assert all(item.stat().st_nlink == 1 for item in destination.iterdir())
    disk_manifest = json.loads((destination / S.MANIFEST_FILENAME).read_text())
    assert disk_manifest["status"] == "CANDIDATE_NOT_AUTHORITY"
    assert disk_manifest["execution_authorized"] is False
    with pytest.raises(S.SemanticRegistryError, match="already exists"):
        S.write_candidate(paths, destination, config=config, evidence_root=root)
    assert not list(output_parent.glob(".*.candidate-stage.*"))
    assert not list(output_parent.glob(".*.candidate-create.lock"))


def test_output_symlink_final_root_and_rename_toctou_fail_closed(
    synthetic_evidence: tuple[Path, S.RegistryInputPaths, S.BuildConfig, pd.DataFrame],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, paths, config, _raw = synthetic_evidence
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(S.SemanticRegistryError, match="symbolic link"):
        S.write_candidate(
            paths,
            linked_parent / "candidate",
            config=config,
            evidence_root=root,
        )

    prohibited = S.ROOT / "outputs" / "final" / "semantic-candidate-must-not-exist"
    assert not os.path.lexists(prohibited)
    with pytest.raises(S.SemanticRegistryError, match="prohibited under outputs/final"):
        S.write_candidate(paths, prohibited, config=config, evidence_root=root)
    assert not os.path.lexists(prohibited)

    original = S._rename_directory_noreplace_at

    def race(
        old_descriptor: int,
        source_name: str,
        new_descriptor: int,
        destination_name: str,
    ) -> None:
        os.mkdir(destination_name, dir_fd=new_descriptor)
        original(old_descriptor, source_name, new_descriptor, destination_name)

    monkeypatch.setattr(S, "_rename_directory_noreplace_at", race)
    raced = real_parent / "raced-candidate"
    with pytest.raises(S.SemanticRegistryError, match="refusing to overwrite"):
        S.write_candidate(paths, raced, config=config, evidence_root=root)
    assert raced.is_dir()
    assert list(raced.iterdir()) == []
    assert not list(real_parent.glob(".*.candidate-stage.*"))
    assert not list(real_parent.glob(".*.candidate-create.lock"))


def test_manifest_string_false_cannot_authorize(
    synthetic_bundle: tuple[S.CandidateBundle, pd.DataFrame, S.BuildConfig],
) -> None:
    bundle, _raw, config = synthetic_bundle
    attacked_manifest = bundle.manifest
    attacked_manifest["execution_authorized"] = "false"
    payload = S._canonical_json_bytes(attacked_manifest)
    files = dict(bundle.files)
    files[S.MANIFEST_FILENAME] = payload
    attacked = S.CandidateBundle(
        files=files,
        daily=bundle.daily,
        training_examples=bundle.training_examples,
        manifest_payload=payload,
        snapshot=bundle.snapshot,
        _token=S._BUNDLE_TOKEN,
    )
    with pytest.raises(S.SemanticRegistryError, match="non-authorizing status"):
        S._verify_bundle(attacked, config=config)


def test_route_a_runtime_is_explicit_and_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    active = {
        **dict(S.ROUTE_A_CANDIDATE_RUNTIME),
    }
    monkeypatch.setattr(S, "_runtime_identity", lambda: active)
    record = S._validate_candidate_runtime(enforce=True)
    assert record["candidate_runtime_enforced"] is True
    assert record["authorization_effect"] == "NONE"
    bad = dict(active)
    bad["python_version"] = "3.11.7"
    monkeypatch.setattr(S, "_runtime_identity", lambda: bad)
    with pytest.raises(S.SemanticRegistryError, match="route-a candidate runtime"):
        S._validate_candidate_runtime(enforce=True)


def test_production_config_cannot_disable_runtime_enforcement() -> None:
    config = S.BuildConfig.production()
    with pytest.raises(S.SemanticRegistryError, match="must enforce route-a runtime"):
        S.BuildConfig(
            mode=config.mode,
            train_start=config.train_start,
            train_end=config.train_end,
            validation_start=config.validation_start,
            validation_end=config.validation_end,
            enforce_route_a_runtime=False,
            expected_counts=config.expected_counts,
        )

"""Synthetic fail-closed tests for the v1 preprocessing-lineage authority."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.final.build_preprocessing_lineage_defect_authority_v1 as authority

SITES = ("00000001", "00000002")
LEGACY_SITES = ("n00", "n01")


def _panel_rows(dates: pd.DatetimeIndex, sites: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for site_index, site in enumerate(sites):
        for index, day in enumerate(dates):
            water = 8.0 + site_index + index / 100.0
            rows.append(
                {
                    "DATE": day,
                    "site_id": site,
                    "WTEMP": water,
                    "FLOW": 20.0 + site_index,
                    "WLEVEL": 3.0,
                    "TEMP": 12.0 + index / 200.0,
                    "PRCP": 0.5,
                    "WDSP": 2.0,
                    "RHMEAN": 60.0,
                    "DH": 150.0,
                }
            )
    return pd.DataFrame(rows, columns=authority.PANEL_COLUMNS)


def _write_inputs(root: Path) -> tuple[authority.LineageInputPaths, authority.BuildConfig]:
    root.mkdir(parents=True)
    station_registry = root / "station_registry.csv"
    pd.DataFrame({"site_no": SITES, "legacy_site_id": LEGACY_SITES, "huc2": ("01", "02")}).to_csv(
        station_registry, index=False
    )

    training_panel = root / "training.parquet"
    evaluation_panel = root / "evaluation.parquet"
    train_dates = pd.date_range("2006-01-01", "2006-02-20", freq="D")
    evaluation_dates = pd.date_range("2006-02-15", "2006-04-30", freq="D")
    train = _panel_rows(train_dates, LEGACY_SITES)
    evaluation = _panel_rows(evaluation_dates, SITES)

    # Genuine pre-imputation gaps in train/validation rows.
    train.loc[
        train["site_id"].eq("n00")
        & train["DATE"].isin(pd.to_datetime(["2006-01-10", "2006-01-11"])),
        "WTEMP",
    ] = np.nan
    train.loc[
        train["site_id"].eq("n01") & train["DATE"].eq(pd.Timestamp("2006-01-20")),
        "WTEMP",
    ] = np.nan
    train.loc[
        train["site_id"].eq("n01") & train["DATE"].eq(pd.Timestamp("2006-02-10")),
        "WTEMP",
    ] = np.nan
    # History gaps before, but never on, the evaluation issue/target keys.
    evaluation.loc[
        evaluation["site_id"].eq(SITES[0]) & evaluation["DATE"].eq(pd.Timestamp("2006-03-01")),
        "WTEMP",
    ] = np.nan
    evaluation.loc[
        evaluation["site_id"].eq(SITES[0]) & evaluation["DATE"].eq(pd.Timestamp("2006-03-02")),
        "FLOW",
    ] = np.nan
    evaluation.loc[
        evaluation["site_id"].eq(SITES[1]) & evaluation["DATE"].eq(pd.Timestamp("2006-03-03")),
        "TEMP",
    ] = np.nan
    train.to_parquet(training_panel, index=False)
    evaluation.to_parquet(evaluation_panel, index=False)

    combined = pd.concat(
        [
            train.assign(site_id=train["site_id"].map(dict(zip(LEGACY_SITES, SITES, strict=True)))),
            evaluation,
        ],
        ignore_index=True,
    ).drop_duplicates(["site_id", "DATE"], keep="last")
    y_lookup = combined.set_index(["site_id", "DATE"])["WTEMP"]
    registry_rows: list[dict[str, object]] = []
    for lead in authority.LEADS:
        for site in SITES:
            for issue in pd.date_range("2006-03-05", "2006-04-20", periods=10):
                issue = issue.normalize()
                target = issue + pd.Timedelta(days=lead)
                y_true = float(y_lookup.loc[(site, target)])
                registry_rows.append(
                    {
                        "key_id": f"{site}|{issue.date()}|h{lead}",
                        "site_id": site,
                        "horizon": lead,
                        "issue_date": issue,
                        "target_date": target,
                        "y_true": y_true,
                    }
                )
    forecast_registry = root / "forecast_keys.parquet"
    registry_frame = pd.DataFrame(registry_rows)
    registry_frame.to_parquet(forecast_registry, index=False)

    formal_reportable_key_registry = root / "primary_reportable_key_registry_v4.parquet"
    formal = registry_frame.rename(columns={"horizon": "lead_days"}).copy()
    formal["issue_wtemp_observed"] = True
    formal["days_since_last_observed_wtemp"] = 0
    formal["n_observed_wtemp_7d"] = 7
    formal["fraction_observed_wtemp_7d"] = 1.0
    formal["n_observed_wtemp_14d"] = 14
    formal["fraction_observed_wtemp_14d"] = 1.0
    formal["n_observed_wtemp_32d"] = 32
    formal["fraction_observed_wtemp_32d"] = 1.0
    formal["history_H100"] = True
    formal["history_H75"] = True
    formal["history_Hall"] = True
    formal["reportable_primary"] = True
    arrow_types = {
        "string": pa.string(),
        "timestamp[ns]": pa.timestamp("ns"),
        "int16": pa.int16(),
        "int32": pa.int32(),
        "double": pa.float64(),
        "bool": pa.bool_(),
    }
    formal_schema = pa.schema(
        [
            pa.field(name, arrow_types[type_name], nullable=nullable)
            for name, type_name, nullable in authority.FORMAL_REPORTABLE_KEY_ARROW_SCHEMA
        ]
    )
    formal = formal[[field.name for field in formal_schema]]
    pq.write_table(
        pa.Table.from_pandas(formal, schema=formal_schema, preserve_index=False, safe=True),
        formal_reportable_key_registry,
    )

    ordinary_files = {
        "information_runner": "information runner placeholder\n",
        "forcing_runner": "forcing runner placeholder\n",
        "features_source": "features placeholder\n",
        "data_source": "data placeholder\n",
        "config_source": "config placeholder\n",
        "information_point_manifest": '{"opaque":"information-point"}\n',
        "information_inference_manifest": '{"opaque":"information-inference"}\n',
        "forcing_point_manifest": '{"opaque":"forcing-point"}\n',
        "forcing_inference_manifest": '{"opaque":"forcing-inference"}\n',
        "score_independent_key_manifest": '{"preserved":"keys"}\n',
        "f2a_acquisition_verification": '{"preserved":"acquisition"}\n',
        "requirements_lock": "pandas==synthetic\n",
        "requirements_lock_py312_hashed": "pandas==synthetic --hash=sha256:00\n",
        "pyproject": "[project]\nname='synthetic'\n",
    }
    paths: dict[str, Path] = {
        "station_registry": station_registry,
        "training_panel": training_panel,
        "evaluation_panel": evaluation_panel,
        "forecast_registry": forecast_registry,
        "formal_reportable_key_registry": formal_reportable_key_registry,
    }
    for role, payload in ordinary_files.items():
        path = root / f"{role}.txt"
        path.write_text(payload, encoding="utf-8")
        paths[role] = path

    config = authority.BuildConfig(
        mode="synthetic",
        train_start=pd.Timestamp("2006-01-01").date(),
        train_end=pd.Timestamp("2006-01-31").date(),
        val_start=pd.Timestamp("2006-02-01").date(),
        val_end=pd.Timestamp("2006-02-28").date(),
    )
    return authority.LineageInputPaths(**paths), config


def _report(bundle: authority._ArtifactBundle) -> dict[str, object]:
    return json.loads(bundle.report_payload)


def test_two_pass_synthetic_reconstruction_is_canonical_and_score_free(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    paths, config = _write_inputs(root)
    first = authority.build_artifacts(paths, config=config, repository_root=root)
    second = authority.build_artifacts(paths, config=config, repository_root=root)
    authority._verify_two_pass(first, second)

    assert set(first.files) == {authority.REPORT_FILENAME, authority.MANIFEST_FILENAME}
    assert first.report_payload == authority._canonical_json_bytes(json.loads(first.report_payload))
    assert first.manifest_payload == authority._canonical_json_bytes(
        json.loads(first.manifest_payload)
    )
    report = _report(first)
    assert report["status"] == authority.STATUS
    assert report["evaluation_inventory"]["authority_role"] == ("formal_reportable_key_registry")
    assert report["legacy_forecast_registry_inventory"]["scientific_evaluation_domain"] is False
    assert report["scope_guard"] == {
        "prediction_shards_read": False,
        "model_score_tables_read": False,
        "model_extension_outcomes_read": False,
        "affected_manifests_hashed_as_opaque_bytes_only": True,
        "new_model_result_computed": False,
    }
    assert all(
        split["invalid_union_rows"] > 0
        for horizon in report["training_lineage_inventory"]["by_horizon"].values()
        for split in horizon.values()
        if split["current_rows"]
    )
    assert all(
        item["raw_issue_missing_rows"] == item["raw_target_missing_rows"] == 0
        for item in report["evaluation_inventory"]["by_horizon"].values()
    )
    assert (
        sum(
            item["mask_error_rows"]
            for item in report["evaluation_inventory"]["by_horizon"].values()
        )
        > 0
    )
    assert (
        report["evaluation_inventory"][
            "F3_evaluation_substitution_count_independently_reconstructed"
        ]
        == 0
    )
    assert all(
        item["F3_total_raw_future_substitutions"] == 0
        for item in report["evaluation_inventory"]["by_horizon"].values()
    )
    assert "0.540617" not in first.report_payload.decode("utf-8")


def test_formal_reportable_registry_is_primary_and_legacy_is_separate(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    paths, config = _write_inputs(root)
    table = pq.read_table(paths.formal_reportable_key_registry)
    pq.write_table(table.slice(1), paths.formal_reportable_key_registry)

    report = _report(authority.build_artifacts(paths, config=config, repository_root=root))
    formal_rows = sum(
        item["registry_rows"] for item in report["evaluation_inventory"]["by_horizon"].values()
    )
    legacy_rows = sum(
        item["registry_rows"]
        for item in report["legacy_forecast_registry_inventory"]["by_horizon"].values()
    )
    relationship = report["legacy_forecast_registry_inventory"][
        "relationship_to_formal_reportable_registry"
    ]
    assert formal_rows == legacy_rows - 1
    assert relationship["legacy_only_nonreportable_rows"] == 1
    assert relationship["formal_reportable_is_exact_subset_of_legacy_keys"] is True


def test_imputed_as_raw_and_observed_schema_drift_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    paths, config = _write_inputs(root)
    training = pd.read_parquet(paths.training_panel)
    evaluation = pd.read_parquet(paths.evaluation_panel)
    training["WTEMP"] = training["WTEMP"].fillna(10.0)
    evaluation["WTEMP"] = evaluation["WTEMP"].fillna(10.0)
    training.to_parquet(paths.training_panel, index=False)
    evaluation.to_parquet(paths.evaluation_panel, index=False)
    with pytest.raises(authority.LineageAuthorityError, match="imputed-as-raw"):
        authority.build_artifacts(paths, config=config, repository_root=root)

    root = tmp_path / "schema"
    paths, config = _write_inputs(root)
    training = pd.read_parquet(paths.training_panel)
    training["WTEMP_observed"] = training["WTEMP"].notna()
    training.to_parquet(paths.training_panel, index=False)
    with pytest.raises(authority.LineageAuthorityError, match="schema/order"):
        authority.build_artifacts(paths, config=config, repository_root=root)


def test_infinite_panel_value_and_formal_flag_schema_drift_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "infinite"
    paths, config = _write_inputs(root)
    training = pd.read_parquet(paths.training_panel)
    training.loc[training.index[0], "WTEMP"] = np.inf
    training.to_parquet(paths.training_panel, index=False)
    with pytest.raises(authority.LineageAuthorityError, match="infinite value"):
        authority.build_artifacts(paths, config=config, repository_root=root)

    root = tmp_path / "flag-schema"
    paths, config = _write_inputs(root)
    formal = pd.read_parquet(paths.formal_reportable_key_registry)
    formal["issue_wtemp_observed"] = formal["issue_wtemp_observed"].astype("int8")
    formal.to_parquet(paths.formal_reportable_key_registry, index=False)
    with pytest.raises(authority.LineageAuthorityError, match="Arrow schema changed"):
        authority.build_artifacts(paths, config=config, repository_root=root)


def test_evaluation_mask_or_raw_label_drift_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    paths, config = _write_inputs(root)
    registry = pd.read_parquet(paths.forecast_registry)
    first = registry.iloc[0]
    evaluation = pd.read_parquet(paths.evaluation_panel)
    selected = evaluation["site_id"].eq(first["site_id"]) & evaluation["DATE"].eq(
        first["issue_date"]
    )
    assert selected.sum() == 1
    evaluation.loc[selected, "WTEMP"] = np.nan
    evaluation.to_parquet(paths.evaluation_panel, index=False)
    with pytest.raises(authority.LineageAuthorityError, match="raw-missing issue"):
        authority.build_artifacts(paths, config=config, repository_root=root)


def test_f3_future_substitution_inventory_is_reconstructed_from_raw_panel(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    paths, config = _write_inputs(root)
    registry = pd.read_parquet(paths.forecast_registry)
    first = registry.loc[registry["horizon"].eq(1)].iloc[0]
    future_date = first["issue_date"] + pd.Timedelta(days=1)
    evaluation = pd.read_parquet(paths.evaluation_panel)
    selected = evaluation["site_id"].eq(first["site_id"]) & evaluation["DATE"].eq(future_date)
    assert selected.sum() == 1
    evaluation.loc[selected, "TEMP"] = np.nan
    evaluation.to_parquet(paths.evaluation_panel, index=False)

    report = _report(authority.build_artifacts(paths, config=config, repository_root=root))
    by_horizon = report["evaluation_inventory"]["by_horizon"]
    reconstructed = report["evaluation_inventory"][
        "F3_evaluation_substitution_count_independently_reconstructed"
    ]
    assert reconstructed > 0
    assert reconstructed == sum(
        item["F3_total_raw_future_substitutions"] for item in by_horizon.values()
    )
    assert by_horizon["1"]["F3_keys_with_any_raw_future_substitution"] > 0
    assert by_horizon["1"]["F3_raw_future_substitutions_by_variable"]["TEMP"] > 0


def test_expected_inventory_drift_and_canonical_hash_drift_fail(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    paths, config = _write_inputs(root)
    baseline = authority.build_artifacts(paths, config=config, repository_root=root)
    temporal = _report(baseline)["training_lineage_inventory"]["by_horizon"]
    expected = {
        int(lead): {split: dict(item) for split, item in splits.items()}
        for lead, splits in temporal.items()
    }
    expected[1]["train"]["current_rows"] += 1
    strict = replace(config, expected_temporal=expected)
    with pytest.raises(authority.LineageAuthorityError, match="temporal lineage counts"):
        authority.build_artifacts(paths, config=strict, repository_root=root)

    observed = {
        role: {"path": path, "sha256": digest}
        for role, (path, digest) in authority.PRODUCTION_CANONICAL_FILES.items()
    }
    first_role = next(iter(observed))
    observed[first_role]["sha256"] = "0" * 64
    with pytest.raises(authority.LineageAuthorityError, match="path/hash"):
        authority._validate_canonical_bindings(observed)


def test_production_config_and_output_location_are_exactly_pinned(tmp_path: Path) -> None:
    paths = authority.LineageInputPaths.production(authority.ROOT)
    noncanonical = replace(
        authority.BuildConfig.production(),
        train_end=pd.Timestamp("2014-12-31").date(),
        val_start=pd.Timestamp("2015-01-01").date(),
        expected_temporal=None,
        expected_evaluation=None,
        expected_legacy_evaluation=None,
    )
    with pytest.raises(authority.LineageAuthorityError, match="exact canonical config"):
        authority.build_artifacts(
            paths,
            config=noncanonical,
            repository_root=authority.ROOT,
        )

    with pytest.raises(authority.LineageAuthorityError, match="allowed output root"):
        authority.publish_authority(
            paths,
            tmp_path / authority.DEFAULT_OUTPUT_NAME,
            config=authority.BuildConfig.production(),
            repository_root=authority.ROOT,
            allowed_output_root=tmp_path,
        )


def test_parent_symlink_input_is_rejected(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (external / "payload").write_bytes(b"external")
    (repository / "linked").symlink_to(external, target_is_directory=True)
    with pytest.raises(authority.LineageAuthorityError, match="symbolic link"):
        authority._read_stable_regular(
            repository / "linked" / "payload",
            root=repository,
            label="parent-symlink input",
        )


def test_path_replacement_during_capture_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    target = repository / "payload"
    target.write_bytes(b"original")
    replacement = repository / "replacement"
    replacement.write_bytes(b"replaced")
    real_fdopen = authority.os.fdopen

    class SwapAfterRead:
        def __init__(self, descriptor: int, mode: str, *, closefd: bool) -> None:
            self.handle = real_fdopen(descriptor, mode, closefd=closefd)

        def __enter__(self) -> Self:
            return self

        def read(self) -> bytes:
            payload = self.handle.read()
            authority.os.replace(replacement, target)
            return payload

        def __exit__(self, *args: object) -> None:
            self.handle.close()

    monkeypatch.setattr(authority.os, "fdopen", SwapAfterRead)
    with pytest.raises(authority.LineageAuthorityError, match="path changed"):
        authority._read_stable_regular(target, root=repository, label="TOCTOU input")


def test_forged_bundle_scope_and_publisher_inputs_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(authority.LineageAuthorityError, match="not issued"):
        authority._ArtifactBundle(
            files={}, report_payload=b"{}\n", manifest_payload=b"{}\n", _token=object()
        )
    with pytest.raises(authority.LineageAuthorityError, match="withdrawn artifact scope"):
        authority._validate_declared_scope(
            (*authority.WITHDRAWN_PATHS, "outputs/final/extra"),
            authority.WITHDRAWN_CLAIMS,
            authority.PRESERVED_SCOPE,
        )

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    config = authority.BuildConfig(
        mode="synthetic",
        train_start=pd.Timestamp("2006-01-01").date(),
        train_end=pd.Timestamp("2006-01-31").date(),
        val_start=pd.Timestamp("2006-02-01").date(),
        val_end=pd.Timestamp("2006-02-28").date(),
    )
    with pytest.raises(authority.LineageAuthorityError, match="LineageInputPaths only"):
        authority.publish_authority(
            object(),  # type: ignore[arg-type]
            allowed / "authority",
            config=config,
            repository_root=tmp_path,
            allowed_output_root=allowed,
        )


def test_create_only_publication_has_no_resume_or_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    paths, config = _write_inputs(root)
    allowed = root / "outputs"
    allowed.mkdir()
    destination = allowed / "lineage_authority"
    published, bundle = authority.publish_authority(
        paths,
        destination,
        config=config,
        repository_root=root,
        allowed_output_root=allowed,
    )
    assert published == destination
    assert {path.name for path in destination.iterdir()} == set(bundle.files)
    assert not any(
        "staging" in path.name or "create.lock" in path.name for path in allowed.iterdir()
    )
    with pytest.raises(authority.LineageAuthorityError, match="refusing to overwrite"):
        authority.publish_authority(
            paths,
            destination,
            config=config,
            repository_root=root,
            allowed_output_root=allowed,
        )
    with pytest.raises(SystemExit):
        authority._parser().parse_args(["--resume"])


def test_partial_publication_failure_cleans_stage_lock_and_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repository"
    paths, config = _write_inputs(root)
    allowed = root / "outputs"
    allowed.mkdir()
    destination = allowed / "lineage_authority"

    def fail_rename(
        _old_directory_descriptor: int,
        _source_name: str,
        _new_directory_descriptor: int,
        _destination_name: str,
    ) -> None:
        raise OSError("injected rename failure")

    monkeypatch.setattr(authority, "_rename_directory_noreplace_at", fail_rename)
    with pytest.raises(OSError, match="injected rename failure"):
        authority.publish_authority(
            paths,
            destination,
            config=config,
            repository_root=root,
            allowed_output_root=allowed,
        )
    assert not destination.exists()
    assert not any(
        "staging" in path.name or "create.lock" in path.name for path in allowed.iterdir()
    )


def test_root_symlink_replacement_before_rename_fails_and_cleans_anchored_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    paths, config = _write_inputs(repository)
    allowed = repository / "outputs"
    allowed.mkdir()
    detached = repository / "outputs-detached"
    external = repository / "external-root"
    external.mkdir()
    destination = allowed / "lineage_authority"
    real_build_twice = authority.build_twice

    def build_then_replace_root(*args: object, **kwargs: object) -> authority._ArtifactBundle:
        bundle = real_build_twice(*args, **kwargs)  # type: ignore[arg-type]
        allowed.rename(detached)
        allowed.symlink_to(external, target_is_directory=True)
        return bundle

    monkeypatch.setattr(authority, "build_twice", build_then_replace_root)
    with pytest.raises(authority.LineageAuthorityError, match="no longer names the anchored"):
        authority.publish_authority(
            paths,
            destination,
            config=config,
            repository_root=repository,
            allowed_output_root=allowed,
        )
    assert allowed.is_symlink()
    assert list(external.iterdir()) == []
    assert list(detached.iterdir()) == []


def test_root_inode_replacement_after_rename_fails_and_removes_published_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    paths, config = _write_inputs(repository)
    allowed = repository / "outputs"
    allowed.mkdir()
    detached = repository / "outputs-detached"
    destination = allowed / "lineage_authority"
    real_rename = authority._rename_directory_noreplace_at

    def rename_then_replace_root(
        old_directory_descriptor: int,
        source_name: str,
        new_directory_descriptor: int,
        destination_name: str,
    ) -> None:
        real_rename(
            old_directory_descriptor,
            source_name,
            new_directory_descriptor,
            destination_name,
        )
        allowed.rename(detached)
        allowed.mkdir()

    monkeypatch.setattr(authority, "_rename_directory_noreplace_at", rename_then_replace_root)
    with pytest.raises(authority.LineageAuthorityError, match="no longer names the anchored"):
        authority.publish_authority(
            paths,
            destination,
            config=config,
            repository_root=repository,
            allowed_output_root=allowed,
        )
    assert allowed.is_dir() and not allowed.is_symlink()
    assert list(allowed.iterdir()) == []
    assert list(detached.iterdir()) == []


def test_affected_and_preserved_scope_are_exact_and_nonoverlapping(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    paths, config = _write_inputs(root)
    report = _report(authority.build_artifacts(paths, config=config, repository_root=root))
    assert tuple(report["withdrawal"]["artifact_paths"]) == authority.WITHDRAWN_PATHS
    assert tuple(report["withdrawal"]["claims"]) == authority.WITHDRAWN_CLAIMS
    assert tuple(report["preserved"]["scope"]) == authority.PRESERVED_SCOPE
    assert not set(report["withdrawal"]["claims"]) & set(report["preserved"]["scope"])
    bindings = report["input_bindings"]
    assert all(
        bindings[role]["opaque_bytes_not_semantically_parsed"]
        for role in authority.AFFECTED_MANIFEST_ROLES
    )
    assert all(
        not bindings[role]["opaque_bytes_not_semantically_parsed"]
        for role in authority.PRESERVED_AUTHORITY_ROLES
    )
    assert report["preserved"]["forecast_registry_context_columns_preserved"] is False
    assert report["preserved"]["conventional_route_a"]["preserved"] is True
    assert report["preserved"]["air2stream"]["preserved_for_this_specific_defect"] is True
    assert (
        report["defect_verdict"]["forcing_lightgbm_eval_set_is_tail_of_same_train_plus_val_table"]
        is True
    )
    assert (
        report["defect_verdict"]["independent_2016_2017_validation_passed_to_old_lightgbm_fit"]
        is False
    )
    eval_defect = report["training_lineage_inventory"]["old_forcing_lightgbm_eval_set_defect"]
    assert eval_defect["validation_size"] == "min(2000, len(training_pool))"
    assert eval_defect["independent_2016_2017_validation_table_used"] is False
    assert (
        "CONVENTIONAL_ROUTE_A_PRE_IMPUTATION_OBSERVED_FLAG_LIFECYCLE_AND_RESULTS"
        in report["preserved"]["scope"]
    )
    assert (
        "AIR2STREAM_RAW_PANEL_OBSERVEDNESS_PATH_NOT_AFFECTED_BY_THIS_DEFECT"
        in report["preserved"]["scope"]
    )
    assert (
        report["required_versioned_rerun"]["unrun_extensions_may_be_opened_by_this_authority"]
        is False
    )
    expected_f3 = authority.PRODUCTION_EXPECTED_EVALUATION[1][
        "F3_raw_future_substitutions_by_variable"
    ]
    assert type(expected_f3) is type(authority.PRODUCTION_EXPECTED_EVALUATION)
    with pytest.raises(TypeError):
        expected_f3["TEMP"] = 1


def test_two_pass_rejects_self_consistent_but_different_payloads(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    paths, config = _write_inputs(root)
    first = authority.build_artifacts(paths, config=config, repository_root=root)
    report = json.loads(first.report_payload)
    report["defect_verdict"]["confirmed"] = False
    forged_report = authority._canonical_json_bytes(report)
    manifest = json.loads(first.manifest_payload)
    manifest["output"]["sha256"] = authority._sha256(forged_report)
    manifest["output"]["size_bytes"] = len(forged_report)
    forged_manifest = authority._canonical_json_bytes(manifest)
    forged = authority._ArtifactBundle(
        files={
            authority.REPORT_FILENAME: forged_report,
            authority.MANIFEST_FILENAME: forged_manifest,
        },
        report_payload=forged_report,
        manifest_payload=forged_manifest,
        _token=authority._BUNDLE_TOKEN,
    )
    with pytest.raises(authority.LineageAuthorityError, match="two-pass bytes differ"):
        authority._verify_two_pass(first, forged)

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import thermoroute.coverage_bridge as coverage_bridge
from thermoroute.coverage_audit import (
    MODEL_REGISTRY,
    POLICY_FILE_SHA256,
    POLICY_RELATIVE,
    PREDICTION_COLUMNS,
    ROUTE_A_FORMAL_TESTS,
)
from thermoroute.coverage_bridge import (
    CONSTRUCTION_BUFFER_START,
    CORE_PROJECTION_START,
    PARQUET_BATCH_ROWS,
    CoverageBridgeError,
    replay_temporal_coverage_from_physical_files,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _prediction_rows(site: str, models: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    offsets = {model: (index + 1) * 0.01 for index, model in enumerate(models)}
    start = pd.Timestamp("2021-01-01")
    end = pd.Timestamp("2023-12-31")
    for model in models:
        for horizon in (1, 3, 7):
            for issue in pd.date_range(start, end - pd.Timedelta(days=horizon)):
                target = issue + pd.Timedelta(days=horizon)
                truth = 10.0 + target.dayofyear / 1000.0
                rows.append(
                    {
                        "model": model,
                        "site_id": site,
                        "horizon": horizon,
                        "issue_date": issue,
                        "target_date": target,
                        "y_true": truth,
                        "y_pred": truth + offsets[model],
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["model", "site_id", "horizon", "issue_date", "target_date"],
        kind="mergesort",
    ).reset_index(drop=True)


def _synthetic_root(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "repo"
    (root / "protocols").mkdir(parents=True)
    (root / "data_usgs").mkdir()
    policy = root / POLICY_RELATIVE
    policy.write_bytes((ROOT / POLICY_RELATIVE).read_bytes())
    protocol = root / "protocols/route_a_confirmatory_v1.json"
    family = [
        {
            **row,
            "alternative": "candidate_minus_reference_below_margin",
            "bootstrap_seed": 1000 + index,
            "sign_flip_seed": 5000 + index,
            "description": row["test_id"],
        }
        for index, row in enumerate(ROUTE_A_FORMAL_TESTS)
    ]
    _write_json(
        protocol,
        {"primary_inference_contract": {"confirmatory_family": family}},
    )
    registries: dict[str, Path] = {}
    for cohort, site in (("development", "t1"), ("external", "e1")):
        path = root / f"data_usgs/{cohort}.csv"
        pd.DataFrame({"site_no": [site], "huc2": ["01"]}).to_csv(
            path, index=False, lineterminator="\n"
        )
        registries[cohort] = path

    base = "outputs/confirmatory/route_a_" + "a" * 24
    state = {
        "namespace": "a" * 24,
        "run_directory": base,
        "acquisition_manifest": f"{base}/acquisition/acquisition_manifest_v1.json",
        "temporal_outcomes": f"{base}/acquisition/temporal_outcomes_v1.parquet",
        "external_outcomes": f"{base}/acquisition/external_outcomes_v1.parquet",
        "availability_registry": f"{base}/trusted/availability_registry_v1.csv",
        "temporal_predictions": f"{base}/trusted/temporal_predictions_v1.parquet",
        "external_predictions": f"{base}/trusted/external_predictions_v1.parquet",
        "statistics": f"{base}/trusted/statistics_v1.json",
        "temporal_coverage_audit": f"{base}/trusted/temporal_coverage_audit_v1.json",
    }
    for cohort, site in (("temporal", "t1"), ("external", "e1")):
        dates = pd.date_range(CONSTRUCTION_BUFFER_START, "2023-12-31")
        outcomes = pd.DataFrame(
            {
                "site_no": site,
                "DATE": dates,
                "WTEMP": 10.0 + dates.dayofyear / 1000.0,
                "WTEMP_value_status": "RETAINED_FINITE_VALUE",
            }
        )
        path = root / state[f"{cohort}_outcomes"]
        path.parent.mkdir(parents=True, exist_ok=True)
        outcomes.to_parquet(path, index=False)
        predictions = _prediction_rows(site, MODEL_REGISTRY[cohort])
        prediction_path = root / state[f"{cohort}_predictions"]
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(prediction_path, index=False)

    availability_rows = []
    for cohort, site in (("temporal", "t1"), ("external", "e1")):
        for horizon in (1, 3, 7):
            count = len(
                pd.date_range(
                    "2021-01-01",
                    pd.Timestamp("2023-12-31") - pd.Timedelta(days=horizon),
                )
            )
            availability_rows.append(
                {
                    "cohort": cohort,
                    "site_no": site,
                    "horizon": horizon,
                    "n_valid_targets": count,
                    "reportable": True,
                }
            )
    availability_path = root / state["availability_registry"]
    availability_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(availability_rows).to_csv(
        availability_path, index=False, lineterminator="\n"
    )
    statistics = {
        "format": "thermoroute.route-a-confirmatory-statistics.v1",
        "tests": [
            {
                "test_id": row["test_id"],
                "status": "NOT_ESTIMABLE_INSUFFICIENT_STATIONS_OR_CLUSTERS",
                "median_effect_c": None,
                "n_stations": 1,
                "n_clusters": 1,
            }
            for row in ROUTE_A_FORMAL_TESTS
        ],
    }
    _write_json(root / state["statistics"], statistics)
    acquisition = {
        "normalized_outcome_tables": {
            cohort: {
                "path": state[f"{cohort}_outcomes"],
                "sha256": _sha(root / state[f"{cohort}_outcomes"]),
            }
            for cohort in ("temporal", "external")
        }
    }
    _write_json(root / state["acquisition_manifest"], acquisition)
    authorization = {
        "protocol": {"path": protocol.relative_to(root).as_posix(), "sha256": _sha(protocol)},
        "temporal_coverage_policy": {
            "path": POLICY_RELATIVE,
            "sha256": POLICY_FILE_SHA256,
        },
        "registries": {
            cohort: {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha(path),
            }
            for cohort, path in registries.items()
        },
        "required_models": {
            cohort: list(MODEL_REGISTRY[cohort])
            for cohort in ("temporal", "external")
        },
        "acquisition_plan": {
            "history_start": CONSTRUCTION_BUFFER_START,
            "target_start": "2021-01-01",
            "target_end": "2023-12-31",
        },
        "state_paths": state,
    }
    return root, authorization


def test_physical_bridge_replays_all_eleven_sources_and_two_pass_predictions(
    tmp_path: Path,
) -> None:
    root, authorization = _synthetic_root(tmp_path)
    audit = replay_temporal_coverage_from_physical_files(
        root=root, authorization=authorization
    )
    assert audit["status"] == "DERIVED_CORE_REQUIRES_RECEIPT_BINDING"
    assert len(audit["source_bindings"]) == 11
    assert audit["source_bindings"]["policy"] == {
        "path": POLICY_RELATIVE,
        "sha256": POLICY_FILE_SHA256,
    }
    assert all(
        row["formal_median_effect_c"] is None
        and row["prediction_derived_descriptive_median_effect_c"] is not None
        and len(row["frozen_sensitivity_candidates"]) == 8
        for row in audit["comparison_sensitivities"]
    )


def test_construction_buffer_is_bound_but_excluded_from_core_projection(
    tmp_path: Path,
) -> None:
    root, authorization = _synthetic_root(tmp_path)
    first = replay_temporal_coverage_from_physical_files(
        root=root, authorization=authorization
    )
    path = root / authorization["state_paths"]["temporal_outcomes"]
    frame = pd.read_parquet(path)
    mask = frame.DATE.eq(pd.Timestamp(CONSTRUCTION_BUFFER_START))
    assert mask.sum() == 1
    frame.loc[mask, "WTEMP"] = np.nan
    frame.loc[mask, "WTEMP_value_status"] = "MISSING_NO_FINITE_SERIES"
    frame.to_parquet(path, index=False)
    manifest_path = root / authorization["state_paths"]["acquisition_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["normalized_outcome_tables"]["temporal"]["sha256"] = _sha(path)
    _write_json(manifest_path, manifest)
    second = replay_temporal_coverage_from_physical_files(
        root=root, authorization=authorization
    )
    assert first["coverage_cells"] == second["coverage_cells"]
    assert first["semantic_inputs"]["observability"] == second["semantic_inputs"][
        "observability"
    ]
    assert first["source_bindings"]["temporal_normalized_outcomes"]["sha256"] != second[
        "source_bindings"
    ]["temporal_normalized_outcomes"]["sha256"]
    assert CORE_PROJECTION_START == "2020-12-01"


def test_missing_construction_buffer_row_fails_closed(tmp_path: Path) -> None:
    root, authorization = _synthetic_root(tmp_path)
    path = root / authorization["state_paths"]["external_outcomes"]
    frame = pd.read_parquet(path)
    frame = frame[~frame.DATE.eq(pd.Timestamp(CONSTRUCTION_BUFFER_START))]
    frame.to_parquet(path, index=False)
    manifest_path = root / authorization["state_paths"]["acquisition_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["normalized_outcome_tables"]["external"]["sha256"] = _sha(path)
    _write_json(manifest_path, manifest)
    with pytest.raises(CoverageBridgeError, match="construction buffer"):
        replay_temporal_coverage_from_physical_files(
            root=root, authorization=authorization
        )


@pytest.mark.parametrize("attack", ["symlink", "hardlink", "path_alias"])
def test_physical_path_and_inode_alias_attacks_fail_closed(
    tmp_path: Path, attack: str
) -> None:
    root, authorization = _synthetic_root(tmp_path)
    if attack == "path_alias":
        authorization["registries"]["external"] = deepcopy(
            authorization["registries"]["development"]
        )
    else:
        binding = authorization["registries"]["external"]
        target = root / binding["path"]
        replacement = target.with_name("external-replacement.csv")
        target.rename(replacement)
        if attack == "symlink":
            target.symlink_to(replacement.name)
        else:
            os.link(replacement, target)
        binding["sha256"] = _sha(replacement)
    with pytest.raises(CoverageBridgeError, match="alias|regular|unsafe"):
        replay_temporal_coverage_from_physical_files(
            root=root, authorization=authorization
        )


def test_unsorted_full_prediction_file_fails_before_core(tmp_path: Path) -> None:
    root, authorization = _synthetic_root(tmp_path)
    path = root / authorization["state_paths"]["temporal_predictions"]
    frame = pd.read_parquet(path)
    frame.iloc[::-1].reset_index(drop=True).to_parquet(path, index=False)
    with pytest.raises(CoverageBridgeError, match="stable-sorted"):
        replay_temporal_coverage_from_physical_files(
            root=root, authorization=authorization
        )


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        ("model", pd.Series([1], dtype="int64")),
        ("site_id", pd.Series([1], dtype="int64")),
        ("horizon", pd.Series([1.0], dtype="float64")),
        ("issue_date", pd.Series(["2021-01-01"], dtype="object")),
        (
            "target_date",
            pd.Series(pd.DatetimeIndex(["2021-01-02"], tz="UTC")),
        ),
        ("y_true", pd.Series([True], dtype="bool")),
        ("y_pred", pd.Series([10], dtype="int64")),
    ],
)
def test_prediction_dtype_aliases_fail_before_streaming_projection(
    tmp_path: Path,
    column: str,
    replacement: pd.Series,
) -> None:
    frame = pd.DataFrame(
        {
            "model": ["ThermoRoute"],
            "site_id": ["t1"],
            "horizon": pd.Series([1], dtype="int64"),
            "issue_date": pd.to_datetime(["2021-01-01"]),
            "target_date": pd.to_datetime(["2021-01-02"]),
            "y_true": pd.Series([10.0], dtype="float64"),
            "y_pred": pd.Series([10.1], dtype="float64"),
        }
    )
    frame[column] = replacement
    path = tmp_path / "prediction.parquet"
    frame.to_parquet(path, index=False)
    reader = coverage_bridge._PhysicalReader(tmp_path)
    with pytest.raises(CoverageBridgeError, match="prediction schema"):
        coverage_bridge._stream_prediction_pass_one(
            reader,
            key="temporal_full_predictions",
            logical="prediction.parquet",
            physical=path,
            expected_sha256=None,
            cohort="temporal",
            sites={"t1"},
        )


def test_prediction_timestamp_must_be_calendar_normalized(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "model": ["ThermoRoute"],
            "site_id": ["t1"],
            "horizon": pd.Series([1], dtype="int64"),
            "issue_date": pd.to_datetime(["2021-01-01 01:00:00"]),
            "target_date": pd.to_datetime(["2021-01-02 01:00:00"]),
            "y_true": pd.Series([10.0], dtype="float64"),
            "y_pred": pd.Series([10.1], dtype="float64"),
        }
    )
    path = tmp_path / "prediction.parquet"
    frame.to_parquet(path, index=False)
    reader = coverage_bridge._PhysicalReader(tmp_path)
    with pytest.raises(CoverageBridgeError, match="frozen contract"):
        coverage_bridge._stream_prediction_pass_one(
            reader,
            key="temporal_full_predictions",
            logical="prediction.parquet",
            physical=path,
            expected_sha256=None,
            cohort="temporal",
            sites={"t1"},
        )


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        pytest.param(
            "site_no", pd.Series([1], dtype="int64"), id="numeric-site"
        ),
        pytest.param(
            "DATE",
            pd.Series(["2020-11-30"], dtype="object"),
            id="string-date",
        ),
        pytest.param(
            "DATE",
            pd.Series(pd.DatetimeIndex(["2020-11-30"], tz="UTC")),
            id="timezone-date",
        ),
        pytest.param("WTEMP", pd.Series([10], dtype="int64"), id="integer-wtemp"),
        pytest.param(
            "WTEMP_value_status",
            pd.Series([1], dtype="int64"),
            id="numeric-status",
        ),
    ],
)
def test_outcome_dtype_aliases_fail_at_physical_boundary(
    tmp_path: Path,
    column: str,
    replacement: pd.Series,
) -> None:
    frame = pd.DataFrame(
        {
            "site_no": ["t1"],
            "DATE": pd.to_datetime([CONSTRUCTION_BUFFER_START]),
            "WTEMP": pd.Series([10.0], dtype="float64"),
            "WTEMP_value_status": ["RETAINED_FINITE_VALUE"],
        }
    )
    frame[column] = replacement
    path = tmp_path / "outcomes.parquet"
    frame.to_parquet(path, index=False)
    reader = coverage_bridge._PhysicalReader(tmp_path)
    with pytest.raises(CoverageBridgeError, match="outcome schema"):
        coverage_bridge._read_observability(
            reader,
            key="temporal_normalized_outcomes",
            logical="outcomes.parquet",
            physical=path,
            expected_sha256=_sha(path),
            sites=("t1",),
        )


def test_outcome_timestamp_must_be_calendar_normalized(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "site_no": ["t1"],
            "DATE": pd.to_datetime(["2020-11-30 01:00:00"]),
            "WTEMP": pd.Series([10.0], dtype="float64"),
            "WTEMP_value_status": ["RETAINED_FINITE_VALUE"],
        }
    )
    path = tmp_path / "outcomes.parquet"
    frame.to_parquet(path, index=False)
    reader = coverage_bridge._PhysicalReader(tmp_path)
    with pytest.raises(CoverageBridgeError, match="invalid site/date keys"):
        coverage_bridge._read_observability(
            reader,
            key="temporal_normalized_outcomes",
            logical="outcomes.parquet",
            physical=path,
            expected_sha256=_sha(path),
            sites=("t1",),
        )


@pytest.mark.parametrize(
    "row",
    [
        pytest.param("temporal,t1,1.0,10,True", id="float-horizon"),
        pytest.param("temporal,t1,True,10,True", id="boolean-horizon"),
        pytest.param("temporal,t1,1,10.0,True", id="float-count"),
        pytest.param("temporal,t1,1,False,True", id="boolean-count"),
        pytest.param("temporal,t1,1,10,1", id="integer-reportable"),
    ],
)
def test_availability_scalar_aliases_fail_at_physical_boundary(
    tmp_path: Path, row: str
) -> None:
    path = tmp_path / "availability.csv"
    path.write_text(
        "cohort,site_no,horizon,n_valid_targets,reportable\n" + row + "\n",
        encoding="utf-8",
    )
    reader = coverage_bridge._PhysicalReader(tmp_path)
    with pytest.raises(CoverageBridgeError, match="physical type contract"):
        coverage_bridge._read_availability(
            reader,
            logical="availability.csv",
            physical=path,
            expected_sha256=_sha(path),
        )


def test_receipt_binding_tamper_fails_without_trusting_audit_paths(
    tmp_path: Path,
) -> None:
    root, authorization = _synthetic_root(tmp_path)
    state = authorization["state_paths"]
    receipt = {
        key: {
            "path": state[state_key],
            "sha256": _sha(root / state[state_key]),
        }
        for key, state_key in {
            "acquisition_manifest": "acquisition_manifest",
            "temporal_normalized_outcomes": "temporal_outcomes",
            "external_normalized_outcomes": "external_outcomes",
            "temporal_predictions": "temporal_predictions",
            "external_predictions": "external_predictions",
            "availability_registry": "availability_registry",
            "statistics": "statistics",
        }.items()
    }
    receipt["statistics"]["sha256"] = "0" * 64
    with pytest.raises(CoverageBridgeError, match="statistics SHA-256 changed"):
        replay_temporal_coverage_from_physical_files(
            root=root,
            authorization=authorization,
            receipt_artifacts=receipt,
        )


def test_same_descriptor_metadata_mutation_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "source.json"
    source.write_bytes(b"{}\n")
    reader = coverage_bridge._PhysicalReader(root)

    with pytest.raises(CoverageBridgeError, match="metadata changed during same-fd read"):
        with reader.open(
            key="source",
            logical_relative="source.json",
            physical_path=source,
            expected_sha256=None,
        ) as handle:
            assert handle.read() == b"{}\n"
            source.write_bytes(b'{"changed":true}\n')


def test_prediction_projection_uses_fixed_size_streaming_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, authorization = _synthetic_root(tmp_path)
    calls: list[dict[str, Any]] = []
    original = coverage_bridge.pq.ParquetFile.iter_batches

    def recording_iter_batches(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        coverage_bridge.pq.ParquetFile,
        "iter_batches",
        recording_iter_batches,
    )
    replay_temporal_coverage_from_physical_files(
        root=root,
        authorization=authorization,
    )
    prediction_calls = [
        call for call in calls if call.get("columns") == list(PREDICTION_COLUMNS)
    ]
    assert len(prediction_calls) == 3
    assert all(call.get("batch_size") == PARQUET_BATCH_ROWS for call in prediction_calls)

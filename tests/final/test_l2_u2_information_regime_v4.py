"""Synthetic-only adversarial tests for the v4 L2_U2 Phase-1 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.final.run_l2_u2_information_regime_v4 as U


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _fold(index: int = 0) -> U.FoldDefinition:
    huc = U.load_station_huc2_metadata()
    folds = U.build_canonical_huc2_folds(huc)
    U.validate_canonical_repository_folds(folds, huc)
    return folds[index]


def _preprocessing_sources(
    fold: U.FoldDefinition | None = None,
    *,
    row_shift: float = 0.0,
) -> U.PreprocessingSources:
    fold = _fold() if fold is None else fold
    dates: dict[str, tuple[str, ...]] = {}
    rows: dict[str, np.ndarray] = {}
    curves: dict[str, np.ndarray] = {}
    for index, station in enumerate(fold.train_stations):
        base = float(index + 1) + row_shift
        dates[station] = ("2006-01-01", "2015-12-31")
        rows[station] = np.array(
            [
                [base, base + 1, base + 2, base + 3, base + 4],
                [base + 1, base + 2, base + 3, base + 4, base + 5],
            ],
            dtype=float,
        )
        curves[station] = np.arange(366, dtype=float) + base
    return U.PreprocessingSources(
        fold=fold.fold,
        meteorology_registry_sha256=_digest("synthetic-training-meteorology-registry"),
        anchor_registry_sha256=_digest("synthetic-training-anchor-registry"),
        meteorology_dates_by_station=dates,
        meteorology_rows_by_station=rows,
        anchor_curve_by_station=curves,
    )


def _preprocessor(fold: U.FoldDefinition | None = None) -> U.TrainingOnlyPreprocessor:
    fold = _fold() if fold is None else fold
    return U.fit_training_only_preprocessor(fold, _preprocessing_sources(fold))


def _forecast_keys(fold: U.FoldDefinition | None = None) -> tuple[U.ForecastKey, ...]:
    fold = _fold() if fold is None else fold
    sites = fold.held_stations[:2]
    keys = (
        U.ForecastKey(sites[0], "2021-01-10", "2021-01-17"),
        U.ForecastKey(sites[1], "2021-01-11", "2021-01-18"),
    )
    return tuple(sorted(keys))


def _raw_predictors(
    fold: U.FoldDefinition | None = None,
    *,
    value_shift: float = 0.0,
) -> U.RawPredictorBatch:
    keys = _forecast_keys(fold)
    n, context, width = len(keys), U.HISTORY_CONTEXT_DAYS, len(U.ALLOWED_METEOROLOGY)
    values = np.arange(n * context * width, dtype=float).reshape(n, context, width)
    values += value_shift
    timestamps = tuple(
        tuple(
            (key.issue_date - U.timedelta(days=offset)).isoformat()
            for offset in range(U.HISTORY_CONTEXT_DAYS - 1, -1, -1)
        )
        for key in keys
    )
    return U.RawPredictorBatch(
        keys=keys,
        key_ids=tuple(key.key_id for key in keys),
        history_timestamps=timestamps,
        history_variables=U.ALLOWED_METEOROLOGY,
        history_values=values,
        history_observed=np.ones_like(values, dtype=bool),
        predictor_registry_sha256=_digest("synthetic-f0-predictor-registry"),
    )


def _canonical() -> U.CanonicalL2U2Inputs:
    fold = _fold()
    sources = _preprocessing_sources(fold)
    state = U.fit_training_only_preprocessor(fold, sources)
    return U.build_canonical_l2_u2_inputs(_raw_predictors(fold), state, sources, role="evaluation")


def _labels(keys: tuple[U.ForecastKey, ...] | None = None) -> U.LabelRegistry:
    keys = _forecast_keys() if keys is None else keys
    return U._construct_label_registry(
        keys=keys,
        y_true=np.arange(len(keys), dtype=np.float64) + 10.0,
        source_file_sha256=_digest("synthetic-label-parquet-bytes"),
    )


def _shared_primary_frame(site: str) -> pd.DataFrame:
    rows = []
    issue = pd.Timestamp("2021-01-10")
    for lead in (1, 3, 7):
        target = issue + pd.Timedelta(days=lead)
        issue_text = issue.strftime("%Y-%m-%d")
        target_text = target.strftime("%Y-%m-%d")
        key_id = hashlib.sha256(
            f"temporal|known_site|{site}|{issue_text}|{target_text}|{lead}".encode()
        ).hexdigest()
        rows.append(
            {
                "key_id": key_id,
                "site_id": site,
                "issue_date": issue,
                "target_date": target,
                "lead_days": lead,
                "y_true": 10.0 + lead,
                "issue_wtemp_observed": True,
                "days_since_last_observed_wtemp": 0,
                "n_observed_wtemp_7d": 7,
                "fraction_observed_wtemp_7d": 1.0,
                "n_observed_wtemp_14d": 14,
                "fraction_observed_wtemp_14d": 1.0,
                "n_observed_wtemp_32d": 32,
                "fraction_observed_wtemp_32d": 1.0,
                "history_H100": True,
                "history_H75": True,
                "history_Hall": True,
                "reportable_primary": True,
            }
        )
    frame = pd.DataFrame(rows, columns=U.SHARED_PRIMARY_REPORTABLE_COLUMNS)
    return frame.astype(
        {
            "key_id": object,
            "site_id": object,
            "issue_date": "datetime64[ns]",
            "target_date": "datetime64[ns]",
            "lead_days": np.int16,
            "y_true": np.float64,
            "issue_wtemp_observed": bool,
            "days_since_last_observed_wtemp": np.int32,
            "n_observed_wtemp_7d": np.int16,
            "fraction_observed_wtemp_7d": np.float64,
            "n_observed_wtemp_14d": np.int16,
            "fraction_observed_wtemp_14d": np.float64,
            "n_observed_wtemp_32d": np.int16,
            "fraction_observed_wtemp_32d": np.float64,
            "history_H100": bool,
            "history_H75": bool,
            "history_Hall": bool,
            "reportable_primary": bool,
        }
    )


def _arrow_schema_with_field(
    field_name: str,
    *,
    field_type: pa.DataType | None = None,
    nullable: bool | None = None,
) -> pa.Schema:
    fields = []
    for field in U.SHARED_PRIMARY_REPORTABLE_ARROW_SCHEMA:
        if field.name != field_name:
            fields.append(field)
            continue
        fields.append(
            pa.field(
                field.name,
                field.type if field_type is None else field_type,
                nullable=field.nullable if nullable is None else nullable,
            )
        )
    return pa.schema(fields)


def _shared_primary_parquet_bytes(
    *,
    schema: pa.Schema = U.SHARED_PRIMARY_REPORTABLE_ARROW_SCHEMA,
    overrides: dict[str, list[object]] | None = None,
) -> bytes:
    columns = _shared_primary_frame("255534081324000").to_dict(orient="list")
    if overrides:
        columns.update(overrides)
    table = pa.Table.from_pydict(columns, schema=schema)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


def test_plan_resolves_legacy_24_contradiction_as_2_8_40() -> None:
    plan = U.build_fit_plan()
    arithmetic = U.plan_arithmetic(plan)

    assert len(plan) == 40
    assert arithmetic["logical_cell_count"] == 2
    assert arithmetic["architecture_by_fold_unit_count"] == 8
    assert arithmetic["model_fit_seed_count"] == 5
    assert arithmetic["fit_unit_count"] == 40
    assert not arithmetic["legacy_24_inherited"]
    assert {cell.architecture for cell in plan} == set(U.Architecture)
    assert {cell.fold for cell in plan} == set(U.FOLDS)
    assert {cell.fit_seed for cell in plan} == set(U.MODEL_FIT_SEEDS)
    assert U.fit_plan_sha256(plan) == U.EXPECTED_PLAN_SHA256


def test_plan_rejects_axis_or_replication_drift() -> None:
    base = U.build_fit_plan()[0]
    with pytest.raises(TypeError, match="Architecture"):
        replace(base, architecture="LightGBM")  # type: ignore[arg-type]
    with pytest.raises(U.ContractError, match="fold"):
        replace(base, fold=4)
    with pytest.raises(U.ContractError, match="fit_seed"):
        replace(base, fit_seed=5)
    with pytest.raises(U.ContractError, match="immutable"):
        replace(base, forcing="F3_full")


def test_static_audit_names_every_material_v4_legacy_mismatch() -> None:
    _payload, protocol = U._read_frozen_draft()
    findings = U.audit_v4_and_legacy(
        protocol,
        U.LEGACY_PROTOCOL.read_text(encoding="utf-8"),
        U.LEGACY_RUNNER.read_text(encoding="utf-8"),
    )
    by_id = {finding.finding_id: finding for finding in findings}

    assert set(by_id) == {
        "V2_24_CELL_ARITHMETIC_CONTRADICTION",
        "LEGACY_SCOPE_NOT_EXECUTABLE_AS_WRITTEN",
        "LEGACY_ARCHITECTURE_MISMATCH",
        "LEGACY_MASK_PROOF_INCOMPLETE",
        "LEGACY_PREPROCESSING_LINEAGE_NOT_TRAIN_ONLY",
        "LEGACY_LABEL_AUTHORITY_NOT_SEPARATE",
        "LEGACY_RESUME_IS_EXISTENCE_ONLY",
        "V4_L2_U2_FORCING_AXIS_UNSPECIFIED",
        "V4_COUNT_RESOLUTION",
    }
    assert by_id["V4_L2_U2_FORCING_AXIS_UNSPECIFIED"].severity == "BLOCKS_SEAL"


def test_repository_huc2_folds_are_exact_intact_and_hash_frozen() -> None:
    huc = U.load_station_huc2_metadata()
    folds = U.build_canonical_huc2_folds(huc)
    U.validate_canonical_repository_folds(folds, huc)

    assert len(huc) == 120
    assert "255534081324000" in huc
    assert U.station_huc2_sha256(huc) == U.EXPECTED_STATION_HUC2_SHA256
    assert U.fold_registry_sha256(folds, huc) == U.EXPECTED_FOLD_REGISTRY_SHA256
    assert tuple(len(fold.held_stations) for fold in folds) == (30, 30, 31, 29)
    for fold in folds:
        U.validate_canonical_fold_definition(fold)
        assert U.canonical_sha256(fold.to_record()) == U.EXPECTED_FOLD_RECORD_SHA256[fold.fold]


def test_preprocessor_uses_exact_dated_training_members_only() -> None:
    fold = _fold()
    sources = _preprocessing_sources(fold)
    state = U.fit_training_only_preprocessor(fold, sources)

    assert set(sources.meteorology_rows_by_station) == set(fold.train_stations)
    assert set(sources.anchor_curve_by_station).isdisjoint(fold.held_stations)
    assert state.training_period == "2006-01-01/2015-12-31"
    assert state.training_stations == fold.train_stations
    assert state.held_stations == fold.held_stations
    assert state.variables == U.ALLOWED_METEOROLOGY
    U.validate_training_only_preprocessor(state)

    changed = U.fit_training_only_preprocessor(fold, _preprocessing_sources(fold, row_shift=10.0))
    assert changed.source_sha256 != state.source_sha256
    assert changed.state_sha256 != state.state_sha256


def test_preprocessor_rejects_held_sources_wrong_dates_and_noncanonical_fold() -> None:
    fold = _fold()
    sources = _preprocessing_sources(fold)
    rows = dict(sources.meteorology_rows_by_station)
    rows[fold.held_stations[0]] = np.ones((2, 5))
    with pytest.raises(U.ContractError, match="training stations only"):
        U.fit_training_only_preprocessor(fold, replace(sources, meteorology_rows_by_station=rows))

    dates = dict(sources.meteorology_dates_by_station)
    dates[fold.train_stations[0]] = ("2006-01-01", "2016-01-01")
    with pytest.raises(U.ContractError, match="leaves canonical"):
        U.fit_training_only_preprocessor(fold, replace(sources, meteorology_dates_by_station=dates))

    bad_fold = replace(
        fold,
        train_stations=tuple(sorted((*fold.train_stations[1:], fold.held_stations[0]))),
        held_stations=tuple(sorted((fold.train_stations[0], *fold.held_stations[1:]))),
    )
    with pytest.raises(U.ContractError, match="not the frozen canonical"):
        U.fit_training_only_preprocessor(bad_fold, sources)


def test_preprocessor_direct_construction_replace_and_mutation_fail_closed() -> None:
    fold = _fold()
    sources = _preprocessing_sources(fold)
    state = U.fit_training_only_preprocessor(fold, sources)
    with pytest.raises(U.ContractError, match="no public constructor"):
        U.TrainingOnlyPreprocessor()
    with pytest.raises(U.ContractError, match="no public constructor"):
        replace(state, source_sha256=_digest("forged"))
    for array in (
        state.imputer_fill,
        state.scaler_center,
        state.scaler_scale,
        state.pooled_anchor_curve,
    ):
        with pytest.raises(ValueError):
            array.setflags(write=True)

    forged = _preprocessor()
    object.__setattr__(
        forged,
        "imputer_fill",
        U._immutable_array(np.asarray(forged.imputer_fill) + 1.0, np.float64),
    )
    with pytest.raises(U.ContractError, match="state hash"):
        U.build_canonical_l2_u2_inputs(
            _raw_predictors(), forged, _preprocessing_sources(), role="evaluation"
        )

    drifted_sources = replace(
        sources,
        meteorology_registry_sha256=_digest("different-current-meteorology-file"),
    )
    with pytest.raises(U.ContractError, match="freshly rehashed exact training-source bytes"):
        U.build_canonical_l2_u2_inputs(
            _raw_predictors(fold), state, drifted_sources, role="evaluation"
        )


def test_forecast_key_is_exact_tuple_hash_and_rejects_forged_fields() -> None:
    key = U.ForecastKey("01000000", "2021-01-10", "2021-01-17")
    expected = hashlib.sha256(b"temporal|known_site|01000000|2021-01-10|2021-01-17|7").hexdigest()
    assert key.key_id == expected
    U.ForecastKey.from_record({"key_id": expected, **key.to_record()})

    production_15 = U.ForecastKey("255534081324000", "2021-01-10", "2021-01-17")
    leading_zero_15 = U.ForecastKey("000000012345678", "2021-01-10", "2021-01-17")
    assert production_15.site_id == "255534081324000"
    assert leading_zero_15.site_id == "000000012345678"
    assert "000000012345678" in (
        f"temporal|known_site|{leading_zero_15.site_id}|{leading_zero_15.issue_time}|"
        f"{leading_zero_15.target_time}|7"
    )

    with pytest.raises(U.ContractError, match="exactly 8 or 15"):
        U.ForecastKey("1", "2021-01-10", "2021-01-17")
    for invalid in ("123456789", "12345678901234", "1234567890123456", "1234567x"):
        with pytest.raises(U.ContractError, match="exactly 8 or 15"):
            U.ForecastKey(invalid, "2021-01-10", "2021-01-17")
    with pytest.raises(U.ContractError, match="exact digit string"):
        U.ForecastKey(255534081324000, "2021-01-10", "2021-01-17")  # type: ignore[arg-type]
    with pytest.raises(U.ContractError, match="canonical YYYY"):
        U.ForecastKey("01000000", "2021-1-10", "2021-01-17")
    with pytest.raises(U.ContractError, match="exactly issue_time"):
        U.ForecastKey("01000000", "2021-01-10", "2021-01-18")
    with pytest.raises(U.ContractError, match="does not hash"):
        U.ForecastKey.from_record({"key_id": _digest("forged"), **key.to_record()})


def test_shared_label_parser_accepts_real_15_digit_id_and_rejects_other_shapes() -> None:
    normalized = U._normalize_shared_primary_reportable_frame(
        _shared_primary_frame("255534081324000")
    )
    assert {record["site_id"] for record in normalized} == {"255534081324000"}
    h7 = next(record for record in normalized if record["lead_days"] == 7)
    key = U.ForecastKey.from_record(
        {name: h7[name] for name in ("key_id", "site_id", "issue_time", "target_time", "lead_days")}
    )
    assert key.site_id == "255534081324000"

    for invalid in ("123456789", "12345678901234", "1234567890123456"):
        with pytest.raises(U.ContractError, match="exactly 8 or 15"):
            U._normalize_shared_primary_reportable_frame(_shared_primary_frame(invalid))


def test_shared_label_parquet_reader_accepts_only_the_exact_physical_schema() -> None:
    frame = U._read_shared_primary_reportable_parquet(_shared_primary_parquet_bytes())

    assert list(frame.columns) == list(U.SHARED_PRIMARY_REPORTABLE_COLUMNS)
    assert frame.dtypes.to_dict() == dict(U.SHARED_PRIMARY_REPORTABLE_PANDAS_DTYPES)
    normalized = U._normalize_shared_primary_reportable_frame(frame)
    assert {record["lead_days"] for record in normalized} == {1, 3, 7}


@pytest.mark.parametrize(
    ("field_name", "field_type", "overrides"),
    [
        ("lead_days", pa.string(), {"lead_days": ["1", "3", "7"]}),
        ("lead_days", pa.float64(), {"lead_days": [1.0, 3.0, 7.0]}),
        (
            "reportable_primary",
            pa.string(),
            {"reportable_primary": ["true", "true", "true"]},
        ),
        ("issue_date", pa.timestamp("ms"), None),
        ("y_true", pa.float32(), None),
    ],
)
def test_shared_label_parquet_reader_rejects_schema_confusion(
    field_name: str,
    field_type: pa.DataType,
    overrides: dict[str, list[object]] | None,
) -> None:
    schema = _arrow_schema_with_field(field_name, field_type=field_type)
    payload = _shared_primary_parquet_bytes(schema=schema, overrides=overrides)

    with pytest.raises(U.ContractError, match="physical Arrow/Parquet schema"):
        U._read_shared_primary_reportable_parquet(payload)


def test_shared_label_parquet_reader_rejects_nullable_and_order_drift() -> None:
    nullable_schema = _arrow_schema_with_field("lead_days", nullable=True)
    with pytest.raises(U.ContractError, match="physical Arrow/Parquet schema"):
        U._read_shared_primary_reportable_parquet(
            _shared_primary_parquet_bytes(schema=nullable_schema)
        )

    reordered_fields = list(U.SHARED_PRIMARY_REPORTABLE_ARROW_SCHEMA)
    reordered_fields[0], reordered_fields[1] = reordered_fields[1], reordered_fields[0]
    with pytest.raises(U.ContractError, match="physical Arrow/Parquet schema"):
        U._read_shared_primary_reportable_parquet(
            _shared_primary_parquet_bytes(schema=pa.schema(reordered_fields))
        )


def test_shared_label_schema_confusion_fails_before_row_or_pandas_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = _arrow_schema_with_field("lead_days", field_type=pa.float64())
    payload = _shared_primary_parquet_bytes(
        schema=schema,
        overrides={"lead_days": [1.0, 3.0, 7.0]},
    )
    metadata = pq.ParquetFile(pa.BufferReader(payload))

    class MetadataOnlyParquet:
        schema_arrow = metadata.schema_arrow
        schema = metadata.schema

        @staticmethod
        def read(*_args: object, **_kwargs: object) -> pa.Table:
            raise AssertionError("schema-confused rows were read before rejection")

    monkeypatch.setattr(U.pq, "ParquetFile", lambda *_args, **_kwargs: MetadataOnlyParquet())
    with pytest.raises(U.ContractError, match="physical Arrow/Parquet schema"):
        U._read_shared_primary_reportable_parquet(payload)


@pytest.mark.parametrize(
    ("column", "values"),
    [
        ("lead_days", pd.Series(["1", "3", "7"], dtype=object)),
        ("lead_days", pd.Series([1.0, 3.0, 7.0], dtype=np.float64)),
        (
            "reportable_primary",
            pd.Series(["true", "true", "true"], dtype=object),
        ),
        (
            "issue_date",
            pd.Series(["2021-01-10"] * 3, dtype="datetime64[ms]"),
        ),
    ],
)
def test_shared_label_normalizer_rejects_permissive_pandas_dtype_coercion(
    column: str,
    values: pd.Series,
) -> None:
    frame = _shared_primary_frame("255534081324000")
    frame[column] = values

    with pytest.raises(U.ContractError, match="exact pandas dtype"):
        U._normalize_shared_primary_reportable_frame(frame)


def test_raw_f0_derives_site_doy_and_season_only_from_exact_keys() -> None:
    raw = _raw_predictors()
    assert raw.station_ids == tuple(key.site_id for key in raw.keys)
    np.testing.assert_array_equal(
        raw.target_day_of_year,
        np.array([key.target_day_of_year for key in raw.keys], dtype=np.int16),
    )
    phase = 2.0 * np.pi * (raw.target_day_of_year.astype(float) - 1.0) / 366.0
    np.testing.assert_allclose(raw.season[:, 0], np.sin(phase))
    np.testing.assert_allclose(raw.season[:, 1], np.cos(phase))
    assert raw.history_variables == U.ALLOWED_METEOROLOGY
    for row, key in zip(raw.history_timestamps, raw.keys, strict=True):
        assert len(row) == U.HISTORY_CONTEXT_DAYS
        assert row[0] == (key.issue_date - U.timedelta(days=31)).isoformat()
        assert row[-1] == key.issue_time


def test_raw_rejects_future_history_prohibited_channels_and_key_misjoin() -> None:
    raw = _raw_predictors()
    future = list(raw.history_timestamps)
    future[0] = (*future[0][:-1], raw.keys[0].target_time)
    with pytest.raises(U.ContractError, match="32 contiguous daily dates ending at issue_time"):
        replace(raw, history_timestamps=tuple(future))

    with pytest.raises(U.ContractError, match="prohibited/local"):
        replace(
            raw,
            history_variables=("WTEMP", *U.ALLOWED_METEOROLOGY),
            history_values=np.zeros((2, U.HISTORY_CONTEXT_DAYS, 6)),
            history_observed=np.ones((2, U.HISTORY_CONTEXT_DAYS, 6), dtype=bool),
        )
    with pytest.raises(U.ContractError, match="do not hash"):
        replace(raw, key_ids=tuple(reversed(raw.key_ids)))
    with pytest.raises(TypeError):
        U.RawPredictorBatch(  # type: ignore[call-arg]
            **{field: getattr(raw, field) for field in raw.__dataclass_fields__},
            station_ids=raw.station_ids,
        )


def test_raw_rejects_zero_short_gapped_or_non_issue_ending_context() -> None:
    raw = _raw_predictors()
    with pytest.raises(U.ContractError, match=r"exact shape \[N,32,5\]"):
        replace(
            raw,
            history_timestamps=((), ()),
            history_values=np.empty((2, 0, 5)),
            history_observed=np.empty((2, 0, 5), dtype=bool),
        )
    with pytest.raises(U.ContractError, match=r"exact shape \[N,32,5\]"):
        replace(
            raw,
            history_timestamps=tuple(row[1:] for row in raw.history_timestamps),
            history_values=np.asarray(raw.history_values)[:, 1:, :],
            history_observed=np.asarray(raw.history_observed)[:, 1:, :],
        )

    gapped = [list(row) for row in raw.history_timestamps]
    gapped[0][10] = gapped[0][9]
    with pytest.raises(U.ContractError, match="32 contiguous daily dates ending at issue_time"):
        replace(raw, history_timestamps=tuple(tuple(row) for row in gapped))

    not_issue_ending = [list(row) for row in raw.history_timestamps]
    not_issue_ending[0] = not_issue_ending[0][:-1] + [raw.keys[0].target_time]
    with pytest.raises(U.ContractError, match="32 contiguous daily dates ending at issue_time"):
        replace(raw, history_timestamps=tuple(tuple(row) for row in not_issue_ending))


def test_raw_rejects_user_supplied_season_doy_identity_or_label_channels() -> None:
    raw = _raw_predictors()
    values = {field: getattr(raw, field) for field in raw.__dataclass_fields__}
    for forbidden_name, forbidden_value in (
        ("season", np.zeros((2, 2))),
        ("target_day_of_year", np.ones(2)),
        ("site_identity", ("a", "b")),
        ("y_true", np.ones(2)),
    ):
        with pytest.raises(TypeError):
            U.RawPredictorBatch(**values, **{forbidden_name: forbidden_value})


def test_parsed_key_station_membership_controls_fold_role() -> None:
    fold = _fold()
    state = _preprocessor(fold)
    raw = _raw_predictors(fold)
    sources = _preprocessing_sources(fold)
    U.build_canonical_l2_u2_inputs(raw, state, sources, role="evaluation")
    with pytest.raises(U.ContractError, match="outside their fold"):
        U.build_canonical_l2_u2_inputs(raw, state, sources, role="train")

    forged_key = U.ForecastKey(
        fold.train_stations[0], raw.keys[0].issue_time, raw.keys[0].target_time
    )
    keys = tuple(sorted((forged_key, raw.keys[1])))
    misjoined = replace(raw, keys=keys, key_ids=tuple(key.key_id for key in keys))
    with pytest.raises(U.ContractError, match="outside their fold"):
        U.build_canonical_l2_u2_inputs(misjoined, state, sources, role="evaluation")


def test_both_architectures_share_exact_met_only_projection() -> None:
    canonical = _canonical()
    tree = U.build_lightgbm_inputs(canonical)
    tcn = U.build_plain_tcn_inputs(canonical)

    assert tree.key_ids == tcn.key_ids == canonical.key_ids
    assert tree.preprocessing_sha256 == tcn.preprocessing_sha256
    assert tree.canonical_input_sha256 == tcn.canonical_input_sha256 == canonical.state_sha256
    assert not any(
        prohibited.lower() in name.lower()
        for name in tree.feature_names
        for prohibited in (*U.PROHIBITED_LOCAL_VARIABLES, "site", "label", "y_true")
    )
    assert tcn.channel_names == U.ALLOWED_METEOROLOGY
    assert tcn.static_names == ("sin_doy", "cos_doy", "pooled_anchor")


def test_allowed_meteorology_changes_both_architecture_inputs() -> None:
    fold = _fold()
    sources = _preprocessing_sources(fold)
    state = U.fit_training_only_preprocessor(fold, sources)
    baseline = U.build_canonical_l2_u2_inputs(
        _raw_predictors(fold), state, sources, role="evaluation"
    )
    changed = U.build_canonical_l2_u2_inputs(
        _raw_predictors(fold, value_shift=2.5), state, sources, role="evaluation"
    )
    assert U.architecture_input_sha256(
        U.build_lightgbm_inputs(baseline)
    ) != U.architecture_input_sha256(U.build_lightgbm_inputs(changed))
    assert U.architecture_input_sha256(
        U.build_plain_tcn_inputs(baseline)
    ) != U.architecture_input_sha256(U.build_plain_tcn_inputs(changed))


def test_canonical_and_adapter_direct_construction_or_replace_is_rejected() -> None:
    canonical = _canonical()
    tree = U.build_lightgbm_inputs(canonical)
    tcn = U.build_plain_tcn_inputs(canonical)
    with pytest.raises(U.ContractError, match="no public constructor"):
        U.CanonicalL2U2Inputs()
    with pytest.raises(U.ContractError, match="no public constructor"):
        U.LightGBMInputs()
    with pytest.raises(U.ContractError, match="no public constructor"):
        U.PlainTCNInputs()
    with pytest.raises(U.ContractError, match="no public constructor"):
        replace(canonical, role="train")
    with pytest.raises(U.ContractError, match="no public constructor"):
        replace(tree, preprocessing_sha256=_digest("forged"))
    with pytest.raises(U.ContractError, match="no public constructor"):
        replace(tcn, preprocessing_sha256=_digest("forged"))


def test_every_model_boundary_array_is_bytes_backed_and_rehashed() -> None:
    canonical = _canonical()
    tree = U.build_lightgbm_inputs(canonical)
    tcn = U.build_plain_tcn_inputs(canonical)
    arrays = (
        canonical.meteorology_values,
        canonical.meteorology_observed,
        canonical.season,
        canonical.pooled_anchor,
        tree.matrix,
        tcn.sequence,
        tcn.sequence_observed,
        tcn.static_context,
    )
    for array in arrays:
        with pytest.raises(ValueError):
            array.setflags(write=True)

    object.__setattr__(
        tree,
        "matrix",
        U._immutable_array(np.asarray(tree.matrix) + 1.0, np.float32),
    )
    with pytest.raises(U.ContractError, match="does not bind current bytes"):
        U.architecture_input_sha256(tree)


def test_key_ids_are_metadata_not_forward_tensors_but_are_hash_bound() -> None:
    canonical = _canonical()
    tree = U.build_lightgbm_inputs(canonical)
    tcn = U.build_plain_tcn_inputs(canonical)
    tree_tensor_hash = U.architecture_tensor_sha256(tree)
    tcn_tensor_hash = U.architecture_tensor_sha256(tcn)

    object.__setattr__(tree, "key_ids", tuple(reversed(tree.key_ids)))
    object.__setattr__(tcn, "key_ids", tuple(reversed(tcn.key_ids)))
    with pytest.raises(U.ContractError, match="hash does not bind"):
        U.architecture_input_sha256(tree)
    with pytest.raises(U.ContractError, match="hash does not bind"):
        U.architecture_input_sha256(tcn)
    assert tree_tensor_hash == U.architecture_tensor_sha256(U.build_lightgbm_inputs(canonical))
    assert tcn_tensor_hash == U.architecture_tensor_sha256(U.build_plain_tcn_inputs(canonical))


def test_labels_are_path_file_and_exact_y_bytes_bound() -> None:
    labels = _labels()
    U.validate_label_registry_structure(labels)
    assert labels.source_path == U._repository_relative(U.AUTHORITY_FILE_PATHS["label_registry"])
    assert (
        labels.y_true_bytes_sha256 == hashlib.sha256(labels.y_true.tobytes(order="C")).hexdigest()
    )
    with pytest.raises(ValueError):
        labels.y_true.setflags(write=True)
    with pytest.raises(U.ContractError, match="no public constructor"):
        U.LabelRegistry()
    with pytest.raises(U.ContractError, match="no public constructor"):
        replace(labels, source_file_sha256=_digest("forged"))


def test_common_keys_and_labels_share_one_primary_reportable_file_authority() -> None:
    expected = (
        U.ROOT
        / "outputs"
        / "final"
        / "information_regime_key_registries_v4"
        / "primary_reportable_key_registry_v4.parquet"
    )
    assert U.SHARED_KEY_REGISTRY == expected
    assert U.AUTHORITY_FILE_PATHS["common_key_registry"] == expected
    assert U.AUTHORITY_FILE_PATHS["label_registry"] == expected
    assert U.SEALED_PROTOCOL_SEAL == (
        U.ROOT / "protocols" / "wrr_information_regimes_protocol_v4_seal.json"
    )


def test_unscoped_or_all_h7_labels_are_never_execution_ready() -> None:
    labels = _labels()
    assert labels.fold is None
    assert labels.view_status == U.UNSCOPED_LABEL_VIEW_STATUS
    assert not labels.execution_authorized
    with pytest.raises(U.Phase1ExecutionUnavailable, match="all-h7 label view"):
        U.authorize_label_view_for_execution(labels)


def test_fold_label_boundary_requires_two_sided_exact_held_predictor_keys() -> None:
    fold = _fold()
    keys = _forecast_keys(fold)
    labels = U._construct_label_registry(
        keys=keys,
        y_true=np.array([10.0, 11.0]),
        source_file_sha256=_digest("synthetic-shared-primary-registry"),
        fold=fold.fold,
        view_status=U.FOLD_LABEL_VIEW_STATUS,
    )
    U.validate_fold_label_evaluation_boundary(labels, fold, keys)

    with pytest.raises(U.ContractError, match="not two-sided exact"):
        U.validate_fold_label_evaluation_boundary(labels, fold, keys[:-1])
    outside = U.ForecastKey(fold.train_stations[0], keys[0].issue_time, keys[0].target_time)
    with pytest.raises(U.ContractError, match="outside fold-held membership"):
        U.validate_fold_label_evaluation_boundary(
            labels,
            fold,
            tuple(sorted((keys[0], outside))),
        )
    with pytest.raises(U.Phase1ExecutionUnavailable, match="predictor loader"):
        U.authorize_label_view_for_execution(labels)


def test_model_label_alignment_is_two_sided_and_revalidated() -> None:
    canonical = _canonical()
    inputs = U.build_lightgbm_inputs(canonical)
    labels = _labels()
    U.validate_model_label_alignment(inputs, labels)

    missing = _labels((_forecast_keys()[0],))
    with pytest.raises(U.ContractError, match="different exact keys"):
        U.validate_model_label_alignment(inputs, missing)


def test_prediction_requires_exact_authority_keys_and_y_true_byte_copy() -> None:
    labels = _labels()
    valid = U.PredictionBatch(
        keys=labels.keys,
        y_true=np.array(labels.y_true, copy=True),
        y_pred=np.array([9.5, 11.5]),
        label_registry_file_sha256=labels.source_file_sha256,
    )
    assert len(U._validate_prediction_content_structure(valid, labels)) == 64

    wrong_y = U.PredictionBatch(
        keys=labels.keys,
        y_true=np.array([10.0, 999.0]),
        y_pred=np.array([9.5, 11.5]),
        label_registry_file_sha256=labels.source_file_sha256,
    )
    with pytest.raises(U.ContractError, match="exact float64 byte copy"):
        U._validate_prediction_content_structure(wrong_y, labels)
    wrong_binding = replace(valid, label_registry_file_sha256=_digest("other-label-file"))
    with pytest.raises(U.ContractError, match="different label authority"):
        U._validate_prediction_content_structure(wrong_binding, labels)


def test_artifact_paths_are_exact_architecture_specific_and_safe() -> None:
    plan = U.build_fit_plan()
    tree = next(cell for cell in plan if cell.architecture is U.Architecture.LIGHTGBM)
    tcn = next(cell for cell in plan if cell.architecture is U.Architecture.PLAIN_TCN)
    tree_paths = U.artifact_paths(tree)
    tcn_paths = U.artifact_paths(tcn)
    assert tree_paths["checkpoint"].endswith(".txt")
    assert tcn_paths["checkpoint"].endswith(".pt")
    assert tree_paths != tcn_paths
    for path in (*tree_paths.values(), *tcn_paths.values()):
        assert not Path(path).is_absolute()
        assert ".." not in Path(path).parts


def test_arbitrary_provenance_and_planned_self_attestation_are_unavailable() -> None:
    arbitrary = {key: _digest(key) for key in U.REQUIRED_PROVENANCE_HASHES}
    with pytest.raises(U.Phase1ExecutionUnavailable, match="arbitrary provenance"):
        U.validate_provenance_hashes(arbitrary)
    with pytest.raises(U.Phase1ExecutionUnavailable, match="cannot emit or accept"):
        U.planned_fit_manifest(U.build_fit_plan()[0], arbitrary)


def test_internally_consistent_structural_inputs_cannot_gain_execution_authority() -> None:
    fold = _fold()
    sources = _preprocessing_sources(fold)
    state = U.fit_training_only_preprocessor(fold, sources)
    raw = _raw_predictors(fold)
    U.build_canonical_l2_u2_inputs(raw, state, sources, role="evaluation")

    with pytest.raises(U.Phase1ExecutionUnavailable, match="structural tests only"):
        U.authorize_structural_inputs_for_execution(
            raw,
            state,
            sources,
            {"authority_sha256": _digest("self-consistent-but-self-attested")},
        )
    assert not U.PREDICTOR_BOUND_BYTE_LOADER_AUTHORIZED
    assert not U.PREPROCESSING_BOUND_BYTE_LOADERS_AUTHORIZED
    assert not U.STRUCTURAL_INPUTS_CAN_AUTHORIZE_EXECUTION


def test_resume_is_terminal_and_reads_no_arbitrary_bytes(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-be-read"
    marker.write_bytes(b"arbitrary self-attested prediction")
    with pytest.raises(U.Phase1CompletionUnavailable, match="terminally unavailable"):
        U.validate_resume_manifest(
            {"status": "COMPLETE", "prediction": str(marker)},
            U.build_fit_plan()[0],
            {key: _digest(key) for key in U.REQUIRED_PROVENANCE_HASHES},
            artifact_root=tmp_path / "absent-root",
        )


def test_completion_rejects_cross_fit_provenance_and_self_attestations_terminally() -> None:
    forged = []
    for index, cell in enumerate(U.build_fit_plan()):
        forged.append(
            {
                "cell": cell.to_record(),
                "status": "COMPLETE",
                "authority_sha256": _digest(f"drifting-authority-{index % 2}"),
                "prediction_content_validated": True,
            }
        )
    with pytest.raises(U.Phase1CompletionUnavailable, match="40 fits"):
        U.validate_completion_inventory_metadata(forged)


def test_execution_authority_direct_construction_is_rejected() -> None:
    with pytest.raises(U.GovernanceError, match="no public constructor"):
        U.ExecutionAuthority()

    forged_assignments = [
        (cell, {"authority_sha256": _digest(f"forged-{cell.fit_id}")})
        for cell in U.build_fit_plan()
    ]
    with pytest.raises(U.GovernanceError, match="exact trusted structural type"):
        U.validate_uniform_fit_authorities(forged_assignments)  # type: ignore[arg-type]


def test_external_self_signed_protocol_and_seal_are_rejected_before_any_data_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    external_protocol = tmp_path / "wrr_information_regimes_protocol_v4_sealed.yaml"
    external_seal = tmp_path / "seal.json"
    external_protocol.write_text(
        "protocol_id: thermoroute_wrr_information_regimes_v4\n"
        "version: 4\nstatus: SEALED\nexecution_authorized: true\n"
        f"seal_path: {external_seal}\n",
        encoding="utf-8",
    )
    external_seal.write_text(
        json.dumps(
            {
                "format": U.SEAL_FORMAT,
                "status": "SEALED",
                "execution_authorized": True,
            }
        ),
        encoding="utf-8",
    )

    def forbidden_data_read(*_args, **_kwargs):
        raise AssertionError("data was touched before external authority rejection")

    monkeypatch.setattr(U, "load_station_huc2_metadata", forbidden_data_read)
    with pytest.raises(U.GovernanceError, match="only the fixed derived protocol path"):
        U.validate_execution_authority(external_protocol)


def test_current_draft_and_future_canonical_path_are_both_terminally_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_data_read(*_args, **_kwargs):
        raise AssertionError("station/panel/outcome data was touched")

    monkeypatch.setattr(U, "load_station_huc2_metadata", forbidden_data_read)
    with pytest.raises(U.GovernanceError, match="only the fixed derived protocol path"):
        U.validate_execution_authority(U.DEFAULT_PROTOCOL)
    with pytest.raises(U.GovernanceError, match="no pre-pinned SHA-256"):
        U.validate_execution_authority(U.SEALED_PROTOCOL)


def test_non_dry_run_fails_governance_before_station_or_outcome_access(monkeypatch) -> None:
    def forbidden_data_read(*_args, **_kwargs):
        raise AssertionError("station metadata was touched before governance failed")

    monkeypatch.setattr(U, "load_station_huc2_metadata", forbidden_data_read)
    args = argparse.Namespace(protocol=U.DEFAULT_PROTOCOL, dry_run=False, print_fits=False)
    with pytest.raises(U.GovernanceError, match="before station/panel/outcome access"):
        U.run(args)


def test_dry_run_is_frozen_draft_only_and_reports_no_experiment_data_or_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_data_read(*_args, **_kwargs):
        raise AssertionError("dry-run attempted a station/panel/outcome data read")

    monkeypatch.setattr(U, "load_station_huc2_metadata", forbidden_data_read)
    manifest = U.dry_run_manifest(protocol_path=U.DEFAULT_PROTOCOL)
    assert manifest["status"] == U.PHASE_STATUS
    assert not manifest["execution_implemented"]
    assert not manifest["reads_2021_2023_panel_or_outcomes"]
    assert not manifest["trains_or_scores_models"]
    assert not manifest["writes_artifacts"]
    assert manifest["inventory"]["fit_unit_count"] == 40
    assert manifest["scope"]["F"] == "F0"
    assert manifest["scope"]["F_status"].endswith("PINNED_DERIVED_PROTOCOL")
    assert manifest["artifact_contract"]["resume"] == "TERMINALLY_UNAVAILABLE_IN_PHASE1"
    common_keys = manifest["common_key_contract"]
    assert common_keys["physical_schema_validated_before_pandas_conversion"]
    assert not common_keys["permissive_pandas_dtype_coercion_allowed"]
    assert len(common_keys["shared_registry_arrow_schema"]) == 18
    assert not any(field["nullable"] for field in common_keys["shared_registry_arrow_schema"])
    assert manifest["future_execution_authority"]["derived_protocol_sha256_pinned"] is None
    future = manifest["future_execution_authority"]
    assert future["canonical_seal_path"].endswith(
        "protocols/wrr_information_regimes_protocol_v4_seal.json"
    )
    assert (
        future["canonical_file_paths"]["common_key_registry"]
        == future["canonical_file_paths"]["label_registry"]
    )
    assert not future["predictor_bound_byte_loader_authorized"]
    assert not future["preprocessing_bound_byte_loaders_authorized"]
    assert not future["structural_inputs_can_authorize_execution"]
    assert not manifest["fold_registry"]["station_membership_read_in_dry_run"]
    assert all("panel" not in path.lower() for path in manifest["safe_dry_run_reads"])

    external = tmp_path / "draft.yaml"
    external.write_bytes(U.DEFAULT_PROTOCOL.read_bytes())
    with pytest.raises(U.ContractError, match="only the frozen canonical"):
        U.dry_run_manifest(protocol_path=external)


def test_even_hypothetical_mapping_cannot_execute_phase1() -> None:
    with pytest.raises(U.GovernanceError, match="exact trusted structural type"):
        U.execute_training({"scope": "L2_U2_EXECUTION_AUTHORITY"})  # type: ignore[arg-type]

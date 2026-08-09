"""Synthetic-only tests for the v4 neural Phase-1 contract runner.

Nothing in this module reads production panels or forecast outcomes.  Synthetic
temporary bytes exercise fail-closed resume and governance contracts; no model
is constructed or trained and no formal output path is touched.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import fields, replace
from datetime import date, timedelta
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

import scripts.final.run_neural_information_regimes as N

T0 = "01000001"
T1 = "01000002"
T2 = "01000003"
S0 = "02000001"
S1 = "02000002"
S2 = "02000003"


def _key_id(site_no: str, issue_date: str, lead: int) -> str:
    return N.canonical_forecast_key_id(
        site_no=site_no,
        issue_date=issue_date,
        target_date=(date.fromisoformat(issue_date) + timedelta(days=lead)).isoformat(),
        lead=lead,
    )


def _rebind_forecast_keys(document: dict[str, object]) -> dict[str, object]:
    document["key_authority_binding_sha256"] = N.canonical_sha256(
        {
            "format": N.FORECAST_KEY_AUTHORITY_BINDING_FORMAT,
            "authority_kind": N.SYNTHETIC_FORECAST_KEY_AUTHORITY_KIND,
            "production_authority_status": N.PRODUCTION_COMMON_KEY_AUTHORITY_STATUS,
            "canonical_registry_path": N.CANONICAL_COMMON_KEY_REGISTRY_PATH,
            "evaluation_period": ["2021-01-01", "2023-12-31"],
            "key_id_formula": N.CANONICAL_FORECAST_KEY_FORMULA,
            "geometry": document["geometry"],
            "split_seed": document["split_seed"],
            "fold": document["fold"],
            "records": document["records"],
        }
    )
    return document


def test_exact_logical_inventory_and_fit_arithmetic() -> None:
    plan = N.build_cell_plan()
    arithmetic = N.inventory_arithmetic(plan)

    assert len(plan) == 5_280
    assert arithmetic == N.EXPECTED_ARITHMETIC
    assert arithmetic.protocol_f0_f3_primary_logical_cells == 48
    assert arithmetic.protocol_lightgbm_primary_logical_cells == 24
    assert arithmetic.protocol_plain_tcn_primary_logical_cells == 24
    assert arithmetic.selected_thermoroute_protocol_cells == 12
    assert arithmetic.selected_thermoroute_variant_configurations == 24
    assert arithmetic.neural_runner_variant_configurations == 48
    assert arithmetic.plain_tcn_random_fits == 2_400
    assert arithmetic.plain_tcn_region_fits == 240
    assert arithmetic.thermoroute_random_fits == 2_400
    assert arithmetic.thermoroute_region_fits == 240
    assert N.cell_plan_sha256(plan) == N.EXPECTED_CELL_PLAN_SHA256

    configurations = {cell.variant_configuration_id for cell in plan}
    assert len(configurations) == 48
    assert len({cell.fit_id for cell in plan}) == len(plan)
    assert all(type(cell) is N.CellPlan for cell in plan)


def test_selected_thermoroute_inventory_is_exact_and_has_both_variants() -> None:
    plan = N.build_cell_plan()
    thermoroute = [cell for cell in plan if cell.architecture is N.Architecture.THERMOROUTE]
    regimes = {(cell.forcing, cell.local_level, cell.geometry) for cell in thermoroute}
    assert regimes == N.SELECTED_THERMOROUTE_REGIMES
    by_logical_regime = {}
    for cell in thermoroute:
        key = (cell.forcing, cell.local_level, cell.geometry, cell.lead)
        by_logical_regime.setdefault(key, set()).add(cell.constraint)
    assert len(by_logical_regime) == 12
    assert all(set(N.ConstraintVariant) == variants for variants in by_logical_regime.values())

    plain = [cell for cell in plan if cell.architecture is N.Architecture.PLAIN_TCN]
    assert {cell.constraint for cell in plain} == {N.ConstraintVariant.UNBOUNDED}


def test_cell_plan_rejects_wrong_types_regimes_and_replication_axes() -> None:
    base = next(
        cell for cell in N.build_cell_plan() if cell.architecture is N.Architecture.PLAIN_TCN
    )
    with pytest.raises(TypeError, match="ForcingArm"):
        replace(base, forcing="F0")  # type: ignore[arg-type]
    with pytest.raises(N.ContractError, match="only the unbounded"):
        replace(base, constraint=N.ConstraintVariant.BOUNDED)
    with pytest.raises(N.ContractError, match="split_seed"):
        replace(base, split_seed=10)

    selected = next(
        cell for cell in N.build_cell_plan() if cell.architecture is N.Architecture.THERMOROUTE
    )
    with pytest.raises(N.ContractError, match="four selected"):
        replace(
            selected,
            forcing=N.ForcingArm.F0,
            local_level=N.LocalLevel.L2,
            geometry=N.Geometry.RANDOM,
        )


def test_random_folds_match_region_sizes_and_partition_stations() -> None:
    stations = tuple(f"s{index:03d}" for index in range(120))
    folds = N.build_matched_random_folds(stations, 7)
    assert [len(fold.hold_stations) for fold in folds] == [30, 30, 31, 29]
    assert [len(fold.train_stations) for fold in folds] == [90, 90, 89, 91]
    N.validate_fold_collection(folds, stations)
    held = [station for fold in folds for station in fold.hold_stations]
    assert Counter(held) == Counter(stations)
    assert N.build_matched_random_folds(stations, 7) == folds
    assert N.build_matched_random_folds(stations, 8) != folds


def _canonical_synthetic_huc_registry() -> tuple[tuple[str, ...], dict[str, str]]:
    stations = tuple(f"s{index:03d}" for index in range(120))
    group_sizes = (
        ("HUC2:17", 26),
        ("HUC2:03", 14),
        ("HUC2:02", 13),
        ("HUC2:04", 10),
        ("HUC2:05", 10),
        ("HUC2:07", 8),
        ("HUC2:14", 8),
        ("HUC2:10", 7),
        ("HUC2:01", 5),
        ("HUC2:12", 5),
        ("HUC2:11", 4),
        ("HUC2:16", 3),
        ("HUC2:18", 3),
        ("HUC2:06", 2),
        ("HUC2:09", 2),
    )
    huc: dict[str, str] = {}
    start = 0
    for group, size in group_sizes:
        for station in stations[start : start + size]:
            huc[station] = group
        start += size
    assert start == len(stations)
    return stations, huc


def test_fold_registry_hash_requires_exact_validated_44_fold_inventory() -> None:
    stations, huc = _canonical_synthetic_huc_registry()
    folds = N.build_canonical_fold_registry(stations, huc)
    assert len(folds) == 44
    assert [len(fold.hold_stations) for fold in folds[-4:]] == [30, 30, 31, 29]
    digest = N.fold_registry_sha256(folds, stations, huc)
    assert digest == N.fold_registry_sha256(
        N.build_canonical_fold_registry(stations, huc), stations, huc
    )

    bad = list(folds)
    first = bad[0]
    bad[0] = replace(
        first,
        train_stations=tuple(sorted((*first.train_stations, first.hold_stations[0]))),
    )
    with pytest.raises(N.ContractError, match="station overlap"):
        N.fold_registry_sha256(tuple(bad), stations, huc)
    with pytest.raises(N.ContractError, match="exactly 44"):
        N.fold_registry_sha256(folds[:-1], stations, huc)


def test_station_and_whole_huc_disjoint_validators_fail_closed() -> None:
    with pytest.raises(N.ContractError, match="station overlap"):
        N.validate_station_disjoint(("a", "b"), ("b", "c"))

    huc = {"a": "01", "b": "01", "c": "02", "d": "02"}
    with pytest.raises(N.ContractError, match="splits HUC2"):
        N.validate_huc_disjoint(("b", "d"), ("a", "c"), huc)
    N.validate_huc_disjoint(("c", "d"), ("a", "b"), huc)


def test_region_builder_keeps_huc_groups_intact() -> None:
    stations = tuple(f"s{index:02d}" for index in range(10))
    huc = {
        **{station: "01" for station in stations[0:3]},
        **{station: "02" for station in stations[3:6]},
        **{station: "03" for station in stations[6:8]},
        **{station: "04" for station in stations[8:10]},
    }
    folds = N.build_region_folds(stations, huc)
    N.validate_fold_collection(
        folds,
        stations,
        huc2_by_station=huc,
        expected_hold_sizes=tuple(len(fold.hold_stations) for fold in folds),
    )
    assert [len(fold.hold_stations) for fold in folds] == [3, 3, 2, 2]


def _synthetic_authority(
    *,
    lead: int = 1,
    issue_dates: tuple[str, ...] = ("2021-01-01", "2021-01-02", "2021-01-03"),
) -> tuple[N.FoldDefinition, dict[str, str], bytes, bytes]:
    fold = N.FoldDefinition(
        geometry=N.Geometry.REGION,
        split_seed=0,
        fold=0,
        train_stations=(T0, T1),
        hold_stations=(S0, S1, S2),
    )
    huc2 = {
        T0: "HUC2:01",
        T1: "HUC2:01",
        S0: "HUC2:02",
        S1: "HUC2:02",
        S2: "HUC2:02",
    }
    fold_bytes = N.build_pooled_l2_fold_registry_bytes((fold,), huc2)
    key_bytes = N.build_forecast_keys_registry_bytes(
        fold,
        tuple(
            N.ForecastKeyRecord(
                key_id=_key_id((S0, S1, S2)[index], issue_date, lead),
                site_no=(S0, S1, S2)[index],
                issue_date=issue_date,
                target_date=(date.fromisoformat(issue_date) + timedelta(days=lead)).isoformat(),
                lead=lead,
                huc2="HUC2:02",
            )
            for index, issue_date in enumerate(issue_dates)
        ),
    )
    return fold, huc2, fold_bytes, key_bytes


class _ExplodingMapping(Mapping[str, np.ndarray]):
    def __getitem__(self, _key: str) -> np.ndarray:
        raise AssertionError("F0 inspected prohibited future forcing")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("F0 iterated prohibited future forcing")

    def __len__(self) -> int:
        raise AssertionError("F0 sized prohibited future forcing")


def test_f0_has_canonical_width_and_never_reads_future_values() -> None:
    _fold, _huc2, fold_bytes, key_bytes = _synthetic_authority(lead=7)
    with pytest.raises(N.ContractError, match="caller-supplied"):
        N.build_canonical_future_batch(
            N.ForcingArm.F0,
            7,
            key_ids=("k0", "k1", "k2"),
            trajectories=_ExplodingMapping(),
            substitution_masks=_ExplodingMapping(),
        )
    future = N.build_canonical_future_batch(
        N.ForcingArm.F0,
        7,
        forecast_keys_registry_bytes=key_bytes,
        fold_registry_bytes=fold_bytes,
    )
    assert future.values.shape == (3, 10)
    assert not future.mask.any()
    assert not future.values.any()
    assert not future.substitution_mask.any()
    assert future.substitution_audit_record()["substitution_count_total"] == 0
    assert future.feature_names == N.FUTURE_FEATURE_POLICY.feature_names(7)
    assert not N.FUTURE_FEATURE_POLICY.raw_trajectory_exposed_to_model
    assert not N.FUTURE_FEATURE_POLICY.substitution_flags_are_model_inputs


def test_forecastkeys_enforce_formal_hash_domain_and_synthetic_authority_binding() -> None:
    _fold, _huc2, fold_bytes, key_bytes = _synthetic_authority()
    document = json.loads(key_bytes)
    assert document["authority_kind"] == N.SYNTHETIC_FORECAST_KEY_AUTHORITY_KIND
    assert document["production_authority_status"] == N.PRODUCTION_COMMON_KEY_AUTHORITY_STATUS
    assert document["canonical_registry_path"] == N.CANONICAL_COMMON_KEY_REGISTRY_PATH
    assert document["evaluation_period"] == ["2021-01-01", "2023-12-31"]
    assert document["key_id_formula"] == N.CANONICAL_FORECAST_KEY_FORMULA
    assert len(document["key_authority_binding_sha256"]) == 64

    forged_binding = copy.deepcopy(document)
    forged_binding["key_authority_binding_sha256"] = "0" * 64
    with pytest.raises(N.ContractError, match="key-authority binding"):
        N.build_canonical_future_batch(
            N.ForcingArm.F0,
            1,
            forecast_keys_registry_bytes=N._canonical_json_bytes(forged_binding),
            fold_registry_bytes=fold_bytes,
        )

    forged_key = copy.deepcopy(document)
    forged_key["records"][0]["key_id"] = "0" * 64
    _rebind_forecast_keys(forged_key)
    with pytest.raises(N.ContractError, match="does not hash its canonical tuple"):
        N.build_canonical_future_batch(
            N.ForcingArm.F0,
            1,
            forecast_keys_registry_bytes=N._canonical_json_bytes(forged_key),
            fold_registry_bytes=fold_bytes,
        )

    training_site_key = copy.deepcopy(document)
    record = training_site_key["records"][0]
    record["site_no"] = T0
    record["huc2"] = "HUC2:01"
    record["key_id"] = _key_id(T0, record["issue_date"], record["lead"])
    _rebind_forecast_keys(training_site_key)
    with pytest.raises(N.ContractError, match="not held by the selected fold"):
        N.build_canonical_future_batch(
            N.ForcingArm.F0,
            1,
            forecast_keys_registry_bytes=N._canonical_json_bytes(training_site_key),
            fold_registry_bytes=fold_bytes,
        )

    with pytest.raises(N.ContractError, match="exact 2021-01-01..2023-12-31"):
        N.ForecastKeyRecord(
            key_id="0" * 64,
            site_no=S0,
            issue_date="2099-01-01",
            target_date="2099-01-02",
            lead=1,
            huc2="HUC2:02",
        )
    with pytest.raises(N.ContractError, match="ASCII-only"):
        N.ForecastKeyRecord(
            key_id="0" * 64,
            site_no="s0",
            issue_date="2021-01-01",
            target_date="2021-01-02",
            lead=1,
            huc2="HUC2:02",
        )


@pytest.mark.parametrize(
    "bad_site",
    (
        "０２０００００１",
        "٠٢٠٠٠٠٠١",
        "²²²²²²²²",
    ),
)
def test_non_ascii_digit_sites_fail_at_key_fold_and_materialization_boundaries(
    bad_site: str,
) -> None:
    assert bad_site.isdigit()
    with pytest.raises(N.ContractError, match="ASCII-only"):
        N.canonical_forecast_key_id(
            site_no=bad_site,
            issue_date="2021-01-01",
            target_date="2021-01-02",
            lead=1,
        )

    _fold, _huc2, fold_bytes, key_bytes = _synthetic_authority()
    forged_keys = json.loads(key_bytes)
    forged_record = forged_keys["records"][0]
    forged_record["site_no"] = bad_site
    forged_record["key_id"] = hashlib.sha256(
        (
            f"temporal|known_site|{bad_site}|{forged_record['issue_date']}|"
            f"{forged_record['target_date']}|{forged_record['lead']}"
        ).encode()
    ).hexdigest()
    _rebind_forecast_keys(forged_keys)
    with pytest.raises(N.ContractError, match="ASCII-only"):
        N.build_canonical_future_batch(
            N.ForcingArm.F0,
            1,
            forecast_keys_registry_bytes=N._canonical_json_bytes(forged_keys),
            fold_registry_bytes=fold_bytes,
        )

    forged_fold = json.loads(fold_bytes)
    for station_record in forged_fold["stations"]:
        if station_record["site_no"] == S0:
            station_record["site_no"] = bad_site
    forged_fold["folds"][0]["hold_stations"][0] = bad_site
    with pytest.raises(N.ContractError, match="ASCII-only"):
        N.build_canonical_future_batch(
            N.ForcingArm.F0,
            1,
            forecast_keys_registry_bytes=key_bytes,
            fold_registry_bytes=N._canonical_json_bytes(forged_fold),
        )

    with pytest.raises(N.ContractError, match="ASCII-only"):
        N.FutureRawRecord(
            key_id="0" * 64,
            site_no=bad_site,
            huc2="HUC2:02",
            issue_date="2021-01-01",
            target_date="2021-01-02",
            lead=1,
            raw_values={variable: [0.0] for variable in N.METEOROLOGICAL_VARIABLES},
        )


def _future_fit_evidence() -> tuple[N.FoldDefinition, bytes, bytes]:
    fold = N.FoldDefinition(
        geometry=N.Geometry.RANDOM,
        split_seed=0,
        fold=0,
        train_stations=(T0, T1),
        hold_stations=(S0, S1),
    )
    source_dates = N._canonical_training_source_dates()
    source_doys = np.asarray(
        [date.fromisoformat(value).timetuple().tm_yday - 1 for value in source_dates],
        dtype=float,
    )
    daily_values = {
        station: {
            variable: source_doys + station_offset + variable_offset
            for variable_offset, variable in enumerate(N.METEOROLOGICAL_VARIABLES)
        }
        for station_offset, station in enumerate(fold.train_stations)
    }
    huc2 = {T0: "HUC2:01", T1: "HUC2:01", S0: "HUC2:02", S1: "HUC2:02"}
    shared_fold_bytes = N.build_pooled_l2_fold_registry_bytes((fold,), huc2)
    source_dates = {
        station: N._canonical_training_source_dates() for station in fold.train_stations
    }
    fold_bytes, source_bytes = N.build_training_only_future_fit_evidence(
        fold=fold,
        fold_registry_bytes=shared_fold_bytes,
        huc2_by_station=huc2,
        training_station_variable_daily_values=daily_values,
        training_source_dates_by_station=source_dates,
    )
    return fold, fold_bytes, source_bytes


def _raw_future_registry(fold: N.FoldDefinition) -> tuple[bytes, bytes]:
    row0 = {
        variable: [
            1.0 + offset,
            None if variable == "PRCP" else 2.0 + offset,
            6.0 + offset,
        ]
        for offset, variable in enumerate(N.METEOROLOGICAL_VARIABLES)
    }
    row1 = {variable: [2.0, 4.0, 9.0] for variable in N.METEOROLOGICAL_VARIABLES}
    records = (
        N.FutureRawRecord(
            key_id=_key_id(S0, "2021-01-01", 3),
            site_no=S0,
            huc2="HUC2:02",
            issue_date="2021-01-01",
            target_date="2021-01-04",
            lead=3,
            raw_values=row0,
        ),
        N.FutureRawRecord(
            key_id=_key_id(S1, "2021-01-02", 3),
            site_no=S1,
            huc2="HUC2:02",
            issue_date="2021-01-02",
            target_date="2021-01-05",
            lead=3,
            raw_values=row1,
        ),
    )
    return (
        N.build_raw_future_registry_bytes(fold=fold, records=records),
        N.build_forecast_keys_registry_bytes(
            fold,
            tuple(
                N.ForecastKeyRecord(
                    key_id=record.key_id,
                    site_no=record.site_no,
                    issue_date=record.issue_date,
                    target_date=record.target_date,
                    lead=record.lead,
                    huc2=record.huc2,
                )
                for record in records
            ),
        ),
    )


def test_f3_exposes_only_target_day_and_prefix_mean_in_canonical_order() -> None:
    fold, fold_bytes, fit_source_bytes = _future_fit_evidence()
    _fit_fold, _fit_huc2, recomputed_climatology, fit_semantic_sha = (
        N._parse_future_fit_evidence_bytes(fold_bytes, fit_source_bytes)
    )
    assert recomputed_climatology[T0]["PRCP"][2] == 3.0
    assert len(fit_semantic_sha) == 64
    fit_document = json.loads(fit_source_bytes)
    assert "daily_records" in fit_document["records"][0]
    assert "variable_climatology" not in fit_document["records"][0]
    raw_bytes, key_bytes = _raw_future_registry(fold)
    future = N.build_canonical_future_batch(
        N.ForcingArm.F3_FULL,
        3,
        forecast_keys_registry_bytes=key_bytes,
        fold_registry_bytes=fold_bytes,
        raw_future_registry_bytes=raw_bytes,
        substitution_fit_source_bytes=fit_source_bytes,
    )
    assert future.values.shape == (2, 10)
    assert future.mask.all()
    assert future.station_ids == (S0, S1)
    assert future.issue_dates == ("2021-01-01", "2021-01-02")
    assert future.target_dates == ("2021-01-04", "2021-01-05")
    assert (future.geometry, future.split_seed, future.fold) == (
        N.Geometry.RANDOM,
        0,
        0,
    )
    np.testing.assert_allclose(future.values[:, 0], [6.0, 9.0])
    np.testing.assert_allclose(future.values[:, 1], [3.0, 5.0])
    assert future.feature_names[:2] == ("TEMP_fut3", "TEMP_futmean3")
    assert not hasattr(future, "trajectories")
    audit = future.substitution_audit_record()
    assert audit["substitution_count_total"] == 1
    assert audit["substitution_count_by_variable"]["PRCP"] == 1
    assert len(audit["source_value_sha256"]) == 64
    assert len(audit["substitution_registry_sha256"]) == 64
    assert len(audit["fit_provenance_sha256"]) == 64
    assert len(audit["canonical_value_sha256"]) == 64
    assert len(audit["canonical_mask_sha256"]) == 64
    assert len(audit["substitution_mask_sha256"]) == 64
    assert len(audit["factory_seal_sha256"]) == 64
    assert not future.execution_validated
    N.validate_canonical_future_batch(future)
    derived = json.loads(future._substitution_registry_bytes)
    prcp_steps = derived["records"][0]["variables"]["PRCP"]
    assert prcp_steps[0]["value"] == 2.0
    assert prcp_steps[0]["substituted"] is False
    assert prcp_steps[1]["value"] == 3.5
    assert prcp_steps[1]["substituted"] is True

    with pytest.raises(N.ContractError, match="caller-supplied"):
        N.build_canonical_future_batch(
            N.ForcingArm.F3_FULL,
            3,
            trajectories={variable: np.ones((2, 3)) for variable in N.METEOROLOGICAL_VARIABLES},
        )

    with pytest.raises(N.ContractError, match="substitution-fit source bytes"):
        N.build_canonical_future_batch(
            N.ForcingArm.F3_FULL,
            3,
            forecast_keys_registry_bytes=key_bytes,
            fold_registry_bytes=fold_bytes,
            raw_future_registry_bytes=raw_bytes,
        )

    held_derived = json.loads(fit_source_bytes)
    held_derived["records"].append(copy.deepcopy(held_derived["records"][0]))
    held_derived["records"][-1]["station_id"] = S0
    with pytest.raises(N.ContractError, match="membership/order mismatch"):
        N.build_canonical_future_batch(
            N.ForcingArm.F3_FULL,
            3,
            forecast_keys_registry_bytes=key_bytes,
            fold_registry_bytes=fold_bytes,
            raw_future_registry_bytes=raw_bytes,
            substitution_fit_source_bytes=N._canonical_json_bytes(held_derived),
        )

    for field, forged_value, message in (
        ("is_missing", True, "missing future raw values"),
        ("valid_date", "2021-01-31", "does not align"),
    ):
        forged_raw = json.loads(raw_bytes)
        step = forged_raw["records"][0]["variables"]["TEMP"][0]
        step[field] = forged_value
        with pytest.raises(N.ContractError, match=message):
            N.build_canonical_future_batch(
                N.ForcingArm.F3_FULL,
                3,
                forecast_keys_registry_bytes=key_bytes,
                fold_registry_bytes=fold_bytes,
                raw_future_registry_bytes=N._canonical_json_bytes(forged_raw),
                substitution_fit_source_bytes=fit_source_bytes,
            )

    forged_huc = json.loads(raw_bytes)
    forged_huc["records"][0]["huc2"] = "HUC2:99"
    with pytest.raises(N.ContractError, match="HUC2 mismatch"):
        N.build_canonical_future_batch(
            N.ForcingArm.F3_FULL,
            3,
            forecast_keys_registry_bytes=key_bytes,
            fold_registry_bytes=fold_bytes,
            raw_future_registry_bytes=N._canonical_json_bytes(forged_huc),
            substitution_fit_source_bytes=fit_source_bytes,
        )

    forged_site = json.loads(raw_bytes)
    forged_site["records"][0]["site_no"] = "99999999"
    with pytest.raises(N.ContractError, match="outside the selected fold"):
        N.build_canonical_future_batch(
            N.ForcingArm.F3_FULL,
            3,
            forecast_keys_registry_bytes=key_bytes,
            fold_registry_bytes=fold_bytes,
            raw_future_registry_bytes=N._canonical_json_bytes(forged_site),
            substitution_fit_source_bytes=fit_source_bytes,
        )

    forged_variables = json.loads(raw_bytes)
    del forged_variables["records"][0]["variables"]["WDSP"]
    with pytest.raises(N.ContractError, match="variables mismatch"):
        N.build_canonical_future_batch(
            N.ForcingArm.F3_FULL,
            3,
            forecast_keys_registry_bytes=key_bytes,
            fold_registry_bytes=fold_bytes,
            raw_future_registry_bytes=N._canonical_json_bytes(forged_variables),
            substitution_fit_source_bytes=fit_source_bytes,
        )

    forged_fit_dates = json.loads(fit_source_bytes)
    forged_fit_dates["records"][0]["daily_records"][0]["date"] = "2006-01-02"
    with pytest.raises(N.ContractError, match="not exact 2006--2015 dates"):
        N.build_canonical_future_batch(
            N.ForcingArm.F3_FULL,
            3,
            forecast_keys_registry_bytes=key_bytes,
            fold_registry_bytes=fold_bytes,
            raw_future_registry_bytes=raw_bytes,
            substitution_fit_source_bytes=N._canonical_json_bytes(forged_fit_dates),
        )

    with pytest.raises(N.ContractError, match="key_ids must be derived"):
        N.build_canonical_future_batch(
            N.ForcingArm.F3_FULL,
            3,
            key_ids=("k0", "k1"),
            forecast_keys_registry_bytes=key_bytes,
            fold_registry_bytes=fold_bytes,
            raw_future_registry_bytes=raw_bytes,
            substitution_fit_source_bytes=fit_source_bytes,
        )

    forged_audit = _forged_dataclass_instance(
        future,
        _substitution_registry_bytes=N._canonical_json_bytes({"forged": True}),
    )
    with pytest.raises(N.ContractError, match="derived audit bytes"):
        N.validate_canonical_future_batch(forged_audit)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="unexpected keyword"):
        N.build_training_only_future_fit_evidence(  # type: ignore[call-arg]
            fold=fold,
            fold_registry_bytes=fold_bytes,
            huc2_by_station={
                T0: "HUC2:01",
                T1: "HUC2:01",
                S0: "HUC2:02",
                S1: "HUC2:02",
            },
            training_station_variable_climatology={
                station: {variable: np.ones(366) for variable in N.METEOROLOGICAL_VARIABLES}
                for station in fold.train_stations
            },
            training_source_dates_by_station={
                station: N._canonical_training_source_dates() for station in fold.train_stations
            },
        )


def _forged_dataclass_instance(instance: object, **changes: object) -> object:
    forged = object.__new__(type(instance))
    for descriptor in fields(instance):
        value = changes.get(descriptor.name, getattr(instance, descriptor.name))
        object.__setattr__(forged, descriptor.name, value)
    return forged


def test_future_batch_direct_construction_and_invented_hash_or_values_are_rejected() -> None:
    with pytest.raises(TypeError, match="cannot be constructed directly"):
        N.CanonicalFutureBatch()  # type: ignore[call-arg]

    future = _f0()
    N.validate_canonical_future_batch(future)
    invented_hash = _forged_dataclass_instance(future, source_value_sha256="0" * 64)
    with pytest.raises(N.ContractError, match="metadata differs"):
        N.validate_canonical_future_batch(invented_hash)  # type: ignore[arg-type]

    writeable_values = _forged_dataclass_instance(
        future,
        values=np.array(future.values, copy=True),
    )
    with pytest.raises(N.ContractError, match="durable immutable"):
        N.validate_canonical_future_batch(writeable_values)  # type: ignore[arg-type]

    invented_values = N._readonly_array(np.ones_like(future.values), np.float32)
    forged_values = _forged_dataclass_instance(future, values=invented_values)
    with pytest.raises(N.ContractError, match="values differs"):
        N.build_separated_batch(
            N.LocalLevel.L0,
            _synthetic_sources(),
            _synthetic_labels(),
            forged_values,  # type: ignore[arg-type]
        )


def test_daily_sources_and_raw_future_reject_overflow_even_without_substitution() -> None:
    fold, fold_bytes, fit_source_bytes = _future_fit_evidence()
    raw_bytes, key_bytes = _raw_future_registry(fold)
    raw_without_substitution = json.loads(raw_bytes)
    for record in raw_without_substitution["records"]:
        for variable in N.METEOROLOGICAL_VARIABLES:
            for step in record["variables"][variable]:
                if step["is_missing"]:
                    step["is_missing"] = False
                    step["raw_value"] = 0.0
    assert not any(
        step["is_missing"]
        for record in raw_without_substitution["records"]
        for variable in N.METEOROLOGICAL_VARIABLES
        for step in record["variables"][variable]
    )

    overflow_fit = json.loads(fit_source_bytes)
    overflow_fit["records"][0]["daily_records"][0]["values"]["TEMP"] = 1e308
    with pytest.raises(N.ContractError, match="physically/overflow-safe"):
        N.build_canonical_future_batch(
            N.ForcingArm.F3_FULL,
            3,
            forecast_keys_registry_bytes=key_bytes,
            fold_registry_bytes=fold_bytes,
            raw_future_registry_bytes=N._canonical_json_bytes(raw_without_substitution),
            substitution_fit_source_bytes=N._canonical_json_bytes(overflow_fit),
        )

    overflow_raw = copy.deepcopy(raw_without_substitution)
    overflow_raw["records"][0]["variables"]["TEMP"][0]["raw_value"] = 1e308
    with pytest.raises(N.ContractError, match="physically/overflow-safe"):
        N.build_canonical_future_batch(
            N.ForcingArm.F3_FULL,
            3,
            forecast_keys_registry_bytes=key_bytes,
            fold_registry_bytes=fold_bytes,
            raw_future_registry_bytes=N._canonical_json_bytes(overflow_raw),
            substitution_fit_source_bytes=fit_source_bytes,
        )

    _l2_fold, l2_fold_bytes, l2_key_bytes, l2_source_bytes = _pooled_context_evidence()
    overflow_l2 = json.loads(l2_source_bytes)
    overflow_l2["records"][0]["daily_records"][0]["wtemp"] = 1e308
    with pytest.raises(N.ContractError, match="physically/overflow-safe"):
        N.build_training_only_pooled_l2_context(
            fold_registry_bytes=l2_fold_bytes,
            forecast_keys_registry_bytes=l2_key_bytes,
            training_source_registry_bytes=N._canonical_json_bytes(overflow_l2),
        )
    with pytest.raises(N.ContractError, match="pooled aggregation"):
        N._safe_pooled_curve(
            (np.full(366, 1e308), np.full(366, 1e308)),
            label="overflow exploit",
        )


def _synthetic_sources() -> N.RawInputSources:
    _fold, _huc2, fold_bytes, key_bytes = _synthetic_authority()
    authority = N._materialize_forecast_key_authority(
        fold_registry_bytes=fold_bytes,
        forecast_keys_registry_bytes=key_bytes,
    )
    n, context, variables = 3, N.HISTORY_LENGTH, len(N.HISTORY_VARIABLES)
    history = np.arange(n * context * variables, dtype=float).reshape(n, context, variables)
    return N.RawInputSources(
        key_ids=tuple(record.key_id for record in authority.records),
        station_identity=tuple(record.site_no for record in authority.records),
        history_timestamps=tuple(
            tuple(
                (
                    date.fromisoformat(record.issue_date)
                    - timedelta(days=N.HISTORY_LENGTH - 1 - offset)
                ).isoformat()
                for offset in range(N.HISTORY_LENGTH)
            )
            for record in authority.records
        ),
        history_values=history,
        history_mask=np.ones_like(history, dtype=bool),
        wtemp_t=np.array([10.0, 11.0, 12.0]),
        clim_t=np.array([8.0, 9.0, 10.0]),
        clim_tgt=np.array([8.5, 9.5, 10.5]),
        damped_prior=np.array([9.5, 10.5, 11.5]),
        phys_std=np.arange(n * 4, dtype=float).reshape(n, 4),
        logflowz=np.array([0.1, 0.2, 0.3]),
        season=np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]),
        gate=np.arange(n * len(N.GATE_FEATURES), dtype=float).reshape(n, len(N.GATE_FEATURES)),
        forecast_keys_registry_bytes=key_bytes,
        fold_registry_bytes=fold_bytes,
    )


def _synthetic_labels(target: np.ndarray | None = None) -> N.LabelTargets:
    _fold, _huc2, fold_bytes, key_bytes = _synthetic_authority()
    authority = N._materialize_forecast_key_authority(
        fold_registry_bytes=fold_bytes,
        forecast_keys_registry_bytes=key_bytes,
    )
    return N.LabelTargets(
        key_ids=tuple(record.key_id for record in authority.records),
        station_ids=tuple(record.site_no for record in authority.records),
        target=np.array([20.0, 21.0, 22.0]) if target is None else target,
        forecast_keys_registry_bytes=key_bytes,
        fold_registry_bytes=fold_bytes,
    )


def _pooled_context_evidence() -> tuple[N.FoldDefinition, bytes, bytes, bytes]:
    fold, _huc2, fold_bytes, forecast_key_bytes = _synthetic_authority()
    source_dates = N._canonical_training_source_dates()
    source_doys = np.asarray(
        [date.fromisoformat(value).timetuple().tm_yday - 1 for value in source_dates],
        dtype=float,
    )
    daily_wtemp = {
        T0: source_doys + 5.0,
        T1: source_doys + 9.0,
    }
    source_bytes = N.build_pooled_l2_training_source_registry_bytes(
        fold,
        daily_wtemp,
        {station: N._canonical_training_source_dates() for station in fold.train_stations},
    )
    return fold, fold_bytes, forecast_key_bytes, source_bytes


def _pooled_context() -> N.PooledL2Context:
    _fold, fold_bytes, forecast_key_bytes, source_bytes = _pooled_context_evidence()
    return N.build_training_only_pooled_l2_context(
        fold_registry_bytes=fold_bytes,
        forecast_keys_registry_bytes=forecast_key_bytes,
        training_source_registry_bytes=source_bytes,
    )


def _f0() -> N.CanonicalFutureBatch:
    _fold, _huc2, fold_bytes, key_bytes = _synthetic_authority()
    return N.build_canonical_future_batch(
        N.ForcingArm.F0,
        1,
        forecast_keys_registry_bytes=key_bytes,
        fold_registry_bytes=fold_bytes,
    )


@pytest.mark.parametrize(
    "mask_value",
    (False, True),
    ids=("mask-false", "masked-values"),
)
def test_numeric_masked_arrays_fail_closed_at_array_l2_and_f3_boundaries(
    mask_value: bool,
) -> None:
    fold, huc2, fold_bytes, _forecast_key_bytes = _synthetic_authority()
    source_dates = N._canonical_training_source_dates()
    masked_values = np.ma.MaskedArray(
        np.ones(len(source_dates), dtype=float),
        mask=mask_value,
    )

    with pytest.raises(N.ContractError, match="must not be a NumPy MaskedArray"):
        N._readonly_array(masked_values, np.float32)

    l2_daily = {station: np.ones(len(source_dates), dtype=float) for station in fold.train_stations}
    l2_daily[fold.train_stations[0]] = masked_values
    with pytest.raises(N.ContractError, match="must not be a NumPy MaskedArray"):
        N.build_pooled_l2_training_source_registry_bytes(
            fold,
            l2_daily,
            {station: source_dates for station in fold.train_stations},
        )

    future_daily = {
        station: {
            variable: np.ones(len(source_dates), dtype=float)
            for variable in N.METEOROLOGICAL_VARIABLES
        }
        for station in fold.train_stations
    }
    future_daily[fold.train_stations[0]]["TEMP"] = masked_values
    with pytest.raises(N.ContractError, match="must not be a NumPy MaskedArray"):
        N.build_training_only_future_fit_evidence(
            fold=fold,
            fold_registry_bytes=fold_bytes,
            huc2_by_station=huc2,
            training_station_variable_daily_values=future_daily,
            training_source_dates_by_station={
                station: source_dates for station in fold.train_stations
            },
        )


@pytest.mark.parametrize("dtype", (float, bool), ids=("float", "bool"))
@pytest.mark.parametrize(
    "mask_value",
    (False, True),
    ids=("mask-false", "masked-values"),
)
def test_point_mse_rejects_numeric_and_boolean_masked_arrays(
    dtype: type,
    mask_value: bool,
) -> None:
    plain = np.asarray([0, 1], dtype=dtype)
    masked = np.ma.MaskedArray(plain, mask=mask_value)
    with pytest.raises(N.ContractError, match="must not be a NumPy MaskedArray"):
        N.point_mse(masked, plain)
    with pytest.raises(N.ContractError, match="must not be a NumPy MaskedArray"):
        N.point_mse(plain, masked)


@pytest.mark.parametrize(
    "mask_value",
    (False, True),
    ids=("mask-false", "masked-values"),
)
def test_safe_daily_curve_rejects_masked_values_and_doy_indices(mask_value: bool) -> None:
    values = np.ones(366, dtype=float)
    date_doys = np.arange(366, dtype=np.int16)
    masked_values = np.ma.MaskedArray(values, mask=mask_value)
    masked_doys = np.ma.MaskedArray(date_doys, mask=mask_value)
    with pytest.raises(N.ContractError, match="must not be a NumPy MaskedArray"):
        N._safe_daily_doy_curve(masked_values, date_doys, label="masked daily values")
    with pytest.raises(N.ContractError, match="must not be a NumPy MaskedArray"):
        N._safe_daily_doy_curve(values, masked_doys, label="masked daily indices")


@pytest.mark.parametrize(
    "mask_value",
    (False, True),
    ids=("mask-false", "masked-values"),
)
def test_safe_pooled_curve_rejects_masked_collections_and_members(mask_value: bool) -> None:
    plain = np.ones(366, dtype=float)
    masked_member = np.ma.MaskedArray(plain, mask=mask_value)
    masked_collection = np.ma.MaskedArray(np.ones((2, 366), dtype=float), mask=mask_value)
    with pytest.raises(N.ContractError, match="must not be a NumPy MaskedArray"):
        N._safe_pooled_curve((plain, masked_member), label="masked member")
    with pytest.raises(N.ContractError, match="must not be a NumPy MaskedArray"):
        N._safe_pooled_curve(masked_collection, label="masked collection")


def test_l2_wtemp_identity_and_label_perturbations_cannot_change_inputs_or_anchor() -> None:
    source = _synthetic_sources()
    labels = _synthetic_labels()
    pooled = _pooled_context()
    baseline = N.build_separated_batch(N.LocalLevel.L2, source, labels, _f0(), pooled_l2=pooled)

    history = np.array(source.history_values, copy=True)
    history[:, :, N.WTEMP_HISTORY_INDEX] += 100_000.0
    mask = np.array(source.history_mask, copy=True)
    mask[:, :, N.WTEMP_HISTORY_INDEX] = ~mask[:, :, N.WTEMP_HISTORY_INDEX]
    gate = np.array(source.gate, copy=True)
    gate[:, N.WTEMP_TENDENCY_INDEX] -= 200_000.0
    perturbed_source = replace(
        source,
        history_values=history,
        history_mask=mask,
        wtemp_t=np.array([-1e6, -2e6, -3e6]),
        clim_t=np.array([1000.0, 2000.0, 3000.0]),
        clim_tgt=np.array([4000.0, 5000.0, 6000.0]),
        damped_prior=np.array([7000.0, 8000.0, 9000.0]),
        gate=gate,
    )
    perturbed_labels = _synthetic_labels(np.array([999.0, 998.0, 997.0]))
    perturbed = N.build_separated_batch(
        N.LocalLevel.L2,
        perturbed_source,
        perturbed_labels,
        _f0(),
        pooled_l2=pooled,
    )

    assert N.model_input_sha256(baseline.model_inputs) == N.model_input_sha256(
        perturbed.model_inputs
    )
    assert N.label_sha256(baseline.labels) != N.label_sha256(perturbed.labels)
    np.testing.assert_array_equal(baseline.model_inputs.damped_prior, pooled.clim_tgt)
    assert not baseline.model_inputs.Mask[:, :, N.WTEMP_HISTORY_INDEX].any()
    assert not baseline.model_inputs.X[:, :, N.WTEMP_HISTORY_INDEX].any()
    assert not baseline.model_inputs.gate[:, N.WTEMP_TENDENCY_INDEX].any()
    assert "station" not in baseline.model_inputs.tensor_mapping()
    assert baseline.future_audit is not None
    assert baseline.future_audit.substitution_audit_record()["substitution_count_total"] == 0
    assert not pooled.execution_validated
    N.validate_pooled_l2_context(pooled)
    assert pooled.lineage.attestation == N.POOLED_L2_ATTESTATION
    assert set(pooled.lineage.training_stations).isdisjoint(pooled.lineage.held_stations)
    assert not hasattr(N, "_MODEL_INPUT_CONSTRUCTION_TOKEN")
    assert not hasattr(N.PooledL2Context, "_from_training_only")
    with pytest.raises(TypeError, match="cannot be constructed directly"):
        N.ModelInputBatch()  # type: ignore[call-arg]
    with pytest.raises(N.ContractError, match="structured ForecastKeys"):
        N.build_separated_batch(
            N.LocalLevel.L2,
            replace(
                source,
                station_identity=("99999991", "99999992", "99999993"),
            ),
            labels,
            _f0(),
            pooled_l2=pooled,
        )


def test_l2_context_rejects_direct_or_held_station_construction() -> None:
    with pytest.raises(TypeError, match="cannot be constructed directly"):
        N.PooledL2Context(  # type: ignore[call-arg]
            key_ids=("k0",),
            clim_t=np.array([1.0]),
            clim_tgt=np.array([2.0]),
        )
    with pytest.raises(TypeError, match="cannot be constructed directly"):
        N.PooledL2Lineage()  # type: ignore[call-arg]

    fold, fold_bytes, forecast_key_bytes, source_bytes = _pooled_context_evidence()
    source_document = json.loads(source_bytes)
    assert "daily_records" in source_document["records"][0]
    assert "climatology" not in source_document["records"][0]
    pooled = N.build_training_only_pooled_l2_context(
        fold_registry_bytes=fold_bytes,
        forecast_keys_registry_bytes=forecast_key_bytes,
        training_source_registry_bytes=source_bytes,
    )
    assert pooled.clim_t[0] == 7.0
    assert pooled.clim_tgt[0] == 8.0

    with pytest.raises(TypeError, match="unexpected keyword"):
        N.build_pooled_l2_training_source_registry_bytes(  # type: ignore[call-arg]
            fold=fold,
            training_station_climatology={station: np.ones(366) for station in fold.train_stations},
            training_source_dates_by_station={
                station: N._canonical_training_source_dates() for station in fold.train_stations
            },
        )
    with pytest.raises(N.ContractError, match="one finite value"):
        N.build_pooled_l2_training_source_registry_bytes(
            fold,
            {station: np.ones(366) for station in fold.train_stations},
            {station: N._canonical_training_source_dates() for station in fold.train_stations},
        )

    held_source = json.loads(source_bytes)
    held_source["records"].append(
        {
            "station_id": S0,
            "daily_records": copy.deepcopy(held_source["records"][0]["daily_records"]),
        }
    )
    with pytest.raises(N.ContractError, match="membership/order mismatch"):
        N.build_training_only_pooled_l2_context(
            fold_registry_bytes=fold_bytes,
            forecast_keys_registry_bytes=forecast_key_bytes,
            training_source_registry_bytes=N._canonical_json_bytes(held_source),
        )

    with pytest.raises(TypeError, match="unexpected keyword"):
        N.build_training_only_pooled_l2_context(  # type: ignore[call-arg]
            fold=fold,
            key_ids=("k0",),
            key_station_ids=("wrong",),
            issue_day_of_year=(99,),
            target_day_of_year=(100,),
        )

    forged_keys = json.loads(forecast_key_bytes)
    forged_keys["records"][0]["huc2"] = "HUC2:99"
    _rebind_forecast_keys(forged_keys)
    with pytest.raises(N.ContractError, match="HUC2 mismatch"):
        N.build_training_only_pooled_l2_context(
            fold_registry_bytes=fold_bytes,
            forecast_keys_registry_bytes=N._canonical_json_bytes(forged_keys),
            training_source_registry_bytes=source_bytes,
        )

    forged_dates = json.loads(forecast_key_bytes)
    forged_dates["records"][0]["target_date"] = "2021-01-03"
    with pytest.raises(N.ContractError, match="must equal issue_date"):
        N.build_training_only_pooled_l2_context(
            fold_registry_bytes=fold_bytes,
            forecast_keys_registry_bytes=N._canonical_json_bytes(forged_dates),
            training_source_registry_bytes=source_bytes,
        )

    forged_source_dates = json.loads(source_bytes)
    forged_source_dates["records"][0]["daily_records"][-1]["date"] = "2015-12-30"
    with pytest.raises(N.ContractError, match="not exact 2006--2015 dates"):
        N.build_training_only_pooled_l2_context(
            fold_registry_bytes=fold_bytes,
            forecast_keys_registry_bytes=forecast_key_bytes,
            training_source_registry_bytes=N._canonical_json_bytes(forged_source_dates),
        )


def test_l2_context_recomputes_exact_fold_source_and_lineage_bytes_on_every_use() -> None:
    pooled = _pooled_context()
    N.validate_pooled_l2_context(pooled)
    assert (
        pooled.lineage.fold_registry_sha256
        == hashlib.sha256(pooled._fold_registry_bytes).hexdigest()
    )
    assert (
        pooled.lineage.training_climatology_source_sha256
        == hashlib.sha256(pooled._training_source_registry_bytes).hexdigest()
    )
    assert len(pooled.factory_seal_sha256) == 64

    injected_lineage = _forged_dataclass_instance(
        pooled.lineage,
        training_climatology_source_sha256="0" * 64,
    )
    forged_lineage = _forged_dataclass_instance(pooled, lineage=injected_lineage)
    with pytest.raises(N.ContractError, match="metadata differs"):
        N.build_separated_batch(
            N.LocalLevel.L2,
            _synthetic_sources(),
            _synthetic_labels(),
            _f0(),
            pooled_l2=forged_lineage,  # type: ignore[arg-type]
        )

    writeable_clim = _forged_dataclass_instance(
        pooled,
        clim_t=np.array(pooled.clim_t, copy=True),
    )
    with pytest.raises(N.ContractError, match="durable immutable"):
        N.validate_pooled_l2_context(writeable_clim)  # type: ignore[arg-type]

    forged_anchor = _forged_dataclass_instance(
        pooled,
        clim_tgt=N._readonly_array(np.full_like(pooled.clim_tgt, 999.0), np.float32),
    )
    with pytest.raises(N.ContractError, match="target climatology differs"):
        N.build_separated_batch(
            N.LocalLevel.L2,
            _synthetic_sources(),
            _synthetic_labels(),
            _f0(),
            pooled_l2=forged_anchor,  # type: ignore[arg-type]
        )

    with pytest.raises(N.ContractError, match="structured ForecastKeys"):
        replace(
            _synthetic_labels(),
            station_ids=(T0, T1, T0),
        )

    tampered_source = json.loads(pooled._training_source_registry_bytes)
    tampered_source["records"].append(
        {
            "station_id": S0,
            "daily_records": copy.deepcopy(tampered_source["records"][0]["daily_records"]),
        }
    )
    forged_held_source = _forged_dataclass_instance(
        pooled,
        _training_source_registry_bytes=N._canonical_json_bytes(tampered_source),
    )
    with pytest.raises(N.ContractError, match="membership/order mismatch"):
        N.validate_pooled_l2_context(forged_held_source)  # type: ignore[arg-type]


def test_shared_semantic_fold_fingerprint_rejects_same_numeric_fold_different_universe() -> None:
    source = _synthetic_sources()
    labels = _synthetic_labels()
    pooled = _pooled_context()
    baseline_future = _f0()
    assert (
        source._forecast_key_authority.semantic_fold_sha256
        == labels._forecast_key_authority.semantic_fold_sha256
        == baseline_future.semantic_fold_sha256
        == pooled.lineage.semantic_fold_sha256
    )
    assert (
        source._forecast_key_authority.key_authority_binding_sha256
        == labels._forecast_key_authority.key_authority_binding_sha256
        == baseline_future.key_authority_binding_sha256
        == pooled.lineage.key_authority_binding_sha256
    )

    _fold, _huc2, _base_fold_bytes, key_bytes = _synthetic_authority()
    alternate_fold = N.FoldDefinition(
        geometry=N.Geometry.REGION,
        split_seed=0,
        fold=0,
        train_stations=(T0, T2),
        hold_stations=(S0, S1, S2),
    )
    alternate_huc2 = {
        T0: "HUC2:01",
        T2: "HUC2:01",
        S0: "HUC2:02",
        S1: "HUC2:02",
        S2: "HUC2:02",
    }
    alternate_fold_bytes = N.build_pooled_l2_fold_registry_bytes((alternate_fold,), alternate_huc2)
    alternate_future = N.build_canonical_future_batch(
        N.ForcingArm.F0,
        1,
        forecast_keys_registry_bytes=key_bytes,
        fold_registry_bytes=alternate_fold_bytes,
    )
    assert alternate_future.fold == baseline_future.fold == 0
    assert alternate_future.semantic_fold_sha256 != baseline_future.semantic_fold_sha256
    with pytest.raises(N.ContractError, match="exact ForecastKeys/fold bytes"):
        N.build_separated_batch(
            N.LocalLevel.L0,
            source,
            labels,
            alternate_future,
        )


def test_f3_fit_builder_rejects_same_numeric_fold_from_different_shared_registry() -> None:
    fold, huc2, _fold_bytes, _key_bytes = _synthetic_authority()
    alternate_fold = N.FoldDefinition(
        geometry=N.Geometry.REGION,
        split_seed=fold.split_seed,
        fold=fold.fold,
        train_stations=(T0, T2),
        hold_stations=fold.hold_stations,
    )
    alternate_huc2 = {
        T0: "HUC2:01",
        T2: "HUC2:01",
        S0: "HUC2:02",
        S1: "HUC2:02",
        S2: "HUC2:02",
    }
    alternate_fold_bytes = N.build_pooled_l2_fold_registry_bytes((alternate_fold,), alternate_huc2)

    with pytest.raises(N.ContractError, match="differs from exact shared fold-registry bytes"):
        N.build_training_only_future_fit_evidence(
            fold=fold,
            fold_registry_bytes=alternate_fold_bytes,
            huc2_by_station=huc2,
            training_station_variable_daily_values={},
            training_source_dates_by_station={},
        )


def test_f3_and_l2_must_share_exact_forecastkey_dates_and_fold_identity() -> None:
    fold, shared_fold_bytes, _key_bytes, _source_bytes = _pooled_context_evidence()
    huc2 = {
        T0: "HUC2:01",
        T1: "HUC2:01",
        S0: "HUC2:02",
        S1: "HUC2:02",
        S2: "HUC2:02",
    }
    source_dates = N._canonical_training_source_dates()
    source_doys = np.asarray(
        [date.fromisoformat(value).timetuple().tm_yday - 1 for value in source_dates],
        dtype=float,
    )
    daily_values = {
        station: {
            variable: source_doys + station_index + variable_index
            for variable_index, variable in enumerate(N.METEOROLOGICAL_VARIABLES)
        }
        for station_index, station in enumerate(fold.train_stations)
    }
    fit_fold_bytes, fit_source_bytes = N.build_training_only_future_fit_evidence(
        fold=fold,
        fold_registry_bytes=shared_fold_bytes,
        huc2_by_station=huc2,
        training_station_variable_daily_values=daily_values,
        training_source_dates_by_station={
            station: N._canonical_training_source_dates() for station in fold.train_stations
        },
    )
    assert fit_fold_bytes == shared_fold_bytes

    def future_for_dates(issue_dates: tuple[str, ...], target_dates: tuple[str, ...]):
        records = tuple(
            N.FutureRawRecord(
                key_id=_key_id((S0, S1, S2)[index], issue_dates[index], 1),
                site_no=(S0, S1, S2)[index],
                huc2="HUC2:02",
                issue_date=issue_dates[index],
                target_date=target_dates[index],
                lead=1,
                raw_values={
                    variable: [float(index + variable_index)]
                    for variable_index, variable in enumerate(N.METEOROLOGICAL_VARIABLES)
                },
            )
            for index in range(3)
        )
        raw_bytes = N.build_raw_future_registry_bytes(
            fold=fold,
            records=records,
        )
        key_bytes = N.build_forecast_keys_registry_bytes(
            fold,
            tuple(
                N.ForecastKeyRecord(
                    key_id=record.key_id,
                    site_no=record.site_no,
                    issue_date=record.issue_date,
                    target_date=record.target_date,
                    lead=record.lead,
                    huc2=record.huc2,
                )
                for record in records
            ),
        )
        return N.build_canonical_future_batch(
            N.ForcingArm.F3_FULL,
            1,
            forecast_keys_registry_bytes=key_bytes,
            fold_registry_bytes=fit_fold_bytes,
            raw_future_registry_bytes=raw_bytes,
            substitution_fit_source_bytes=fit_source_bytes,
        )

    pooled = _pooled_context()
    aligned = future_for_dates(
        ("2021-01-01", "2021-01-02", "2021-01-03"),
        ("2021-01-02", "2021-01-03", "2021-01-04"),
    )
    N.build_separated_batch(
        N.LocalLevel.L2,
        _synthetic_sources(),
        _synthetic_labels(),
        aligned,
        pooled_l2=pooled,
    )
    shifted = future_for_dates(
        ("2021-02-01", "2021-02-02", "2021-02-03"),
        ("2021-02-02", "2021-02-03", "2021-02-04"),
    )
    with pytest.raises(N.ContractError, match="key registries differ|exact ForecastKeys"):
        N.build_separated_batch(
            N.LocalLevel.L2,
            _synthetic_sources(),
            _synthetic_labels(),
            shifted,
            pooled_l2=pooled,
        )


def test_all_boundaries_reject_missing_or_2099_structured_forecastkeys() -> None:
    fold, fold_bytes, fit_source_bytes = _future_fit_evidence()
    raw_bytes, key_bytes = _raw_future_registry(fold)
    with pytest.raises(N.ContractError, match="structured ForecastKeys bytes"):
        N.build_canonical_future_batch(
            N.ForcingArm.F3_FULL,
            3,
            fold_registry_bytes=fold_bytes,
            raw_future_registry_bytes=raw_bytes,
            substitution_fit_source_bytes=fit_source_bytes,
        )

    raw_2099 = json.loads(raw_bytes)
    for index, record in enumerate(raw_2099["records"], start=1):
        issue = date(2099, 1, index)
        record["issue_date"] = issue.isoformat()
        record["target_date"] = (issue + timedelta(days=3)).isoformat()
        for variable in N.METEOROLOGICAL_VARIABLES:
            for step, step_record in enumerate(record["variables"][variable], start=1):
                step_record["valid_date"] = (issue + timedelta(days=step)).isoformat()
    with pytest.raises(N.ContractError, match="exactly match structured ForecastKeys"):
        N.build_canonical_future_batch(
            N.ForcingArm.F3_FULL,
            3,
            forecast_keys_registry_bytes=key_bytes,
            fold_registry_bytes=fold_bytes,
            raw_future_registry_bytes=N._canonical_json_bytes(raw_2099),
            substitution_fit_source_bytes=fit_source_bytes,
        )

    _fold, _huc2, fold_2099, valid_keys = _synthetic_authority()
    keys_2099 = json.loads(valid_keys)
    for index, record in enumerate(keys_2099["records"], start=1):
        issue = date(2099, 1, index)
        target = issue + timedelta(days=int(record["lead"]))
        record["issue_date"] = issue.isoformat()
        record["target_date"] = target.isoformat()
        record["key_id"] = hashlib.sha256(
            (
                f"temporal|known_site|{record['site_no']}|{record['issue_date']}|"
                f"{record['target_date']}|{record['lead']}"
            ).encode()
        ).hexdigest()
    _rebind_forecast_keys(keys_2099)
    with pytest.raises(N.ContractError, match="exact 2021-01-01..2023-12-31"):
        N.build_canonical_future_batch(
            N.ForcingArm.F0,
            1,
            forecast_keys_registry_bytes=N._canonical_json_bytes(keys_2099),
            fold_registry_bytes=fold_2099,
        )


def test_l0_preserves_allowed_wtemp_but_remains_label_separated() -> None:
    source = _synthetic_sources()
    baseline = N.build_separated_batch(N.LocalLevel.L0, source, _synthetic_labels(), _f0())
    label_perturbed = N.build_separated_batch(
        N.LocalLevel.L0,
        source,
        _synthetic_labels(np.array([500.0, 600.0, 700.0])),
        _f0(),
    )
    assert N.model_input_sha256(baseline.model_inputs) == N.model_input_sha256(
        label_perturbed.model_inputs
    )

    history = np.array(source.history_values, copy=True)
    history[:, :, N.WTEMP_HISTORY_INDEX] += 1.0
    input_perturbed = N.build_separated_batch(
        N.LocalLevel.L0,
        replace(source, history_values=history),
        _synthetic_labels(),
        _f0(),
    )
    assert N.model_input_sha256(baseline.model_inputs) != N.model_input_sha256(
        input_perturbed.model_inputs
    )


def test_history_requires_exact_32_contiguous_authority_aligned_dates() -> None:
    source = _synthetic_sources()
    for masked in (
        np.ma.MaskedArray(
            np.ones(source.history_mask.shape, dtype=bool),
            mask=False,
        ),
        np.ma.MaskedArray(
            np.ones(source.history_mask.shape, dtype=bool),
            mask=np.ones(source.history_mask.shape, dtype=bool),
        ),
    ):
        with pytest.raises(N.ContractError, match="must not be a NumPy MaskedArray"):
            replace(source, history_mask=masked)
    with pytest.raises(N.ContractError, match="exact boolean"):
        replace(
            source,
            history_mask=np.ones(source.history_mask.shape, dtype=np.int8),
        )
    with pytest.raises(N.ContractError, match=r"shape .*32, 7"):
        replace(
            source,
            history_values=np.array(source.history_values[:, :-1, :], copy=True),
            history_mask=np.array(source.history_mask[:, :-1, :], copy=True),
        )

    gapped = [list(row) for row in source.history_timestamps]
    gapped[0][10] = gapped[0][9]
    with pytest.raises(N.ContractError, match="contiguous issue-31..issue"):
        replace(source, history_timestamps=tuple(tuple(row) for row in gapped))

    dates_2099 = tuple(
        tuple(
            (date(2099, 1, 31) - timedelta(days=N.HISTORY_LENGTH - 1 - offset)).isoformat()
            for offset in range(N.HISTORY_LENGTH)
        )
        for _index in range(3)
    )
    with pytest.raises(N.ContractError, match="contiguous issue-31..issue"):
        replace(source, history_timestamps=dates_2099)


def test_label_source_and_model_arrays_are_durably_immutable_and_rechecked() -> None:
    source = _synthetic_sources()
    labels = _synthetic_labels()
    batch = N.build_separated_batch(N.LocalLevel.L0, source, labels, _f0())
    for array in (
        source.history_values,
        labels.target,
        batch.model_inputs.X,
        batch.model_inputs.future_context,
    ):
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError):
            array.flat[0] = 123.0

    forged_labels = _forged_dataclass_instance(
        labels,
        target=np.array(labels.target, copy=True),
    )
    with pytest.raises(N.ContractError, match="durable immutable"):
        N.label_sha256(forged_labels)  # type: ignore[arg-type]

    forged_sources = _forged_dataclass_instance(
        source,
        history_values=np.array(source.history_values, copy=True),
    )
    with pytest.raises(N.ContractError, match="durable immutable"):
        N.build_separated_batch(
            N.LocalLevel.L0,
            forged_sources,  # type: ignore[arg-type]
            labels,
            _f0(),
        )

    forged_model = _forged_dataclass_instance(
        batch.model_inputs,
        X=np.array(batch.model_inputs.X, copy=True),
    )
    with pytest.raises(N.ContractError, match="durable immutable"):
        forged_model.tensor_mapping()  # type: ignore[union-attr]


def test_point_objective_and_active_parameter_budget_are_machine_checked() -> None:
    assert N.point_mse(np.array([1.0, 3.0]), np.array([0.0, 1.0])) == 2.5
    policy = N.POINT_OBJECTIVE_POLICY.to_record()
    assert policy["trained_output_heads"] == ["point"]
    assert policy["forbidden_auxiliary_losses"] == [
        "quantile_pinball",
        "event_bce",
        "quantile_crossing",
        "residual_magnitude",
    ]
    assert N.FIT_BUDGET_POLICY.model_fit_seeds == (0, 1, 2, 3, 4)
    N.validate_active_parameter_counts(
        {
            "plain_TCN:unbounded": 1_010,
            "ThermoRoute:unbounded": 1_000,
            "ThermoRoute:bounded": 1_000,
        }
    )
    with pytest.raises(N.ContractError, match="differ"):
        N.validate_active_parameter_counts(
            {
                "plain_TCN:unbounded": 1_100,
                "ThermoRoute:unbounded": 1_000,
                "ThermoRoute:bounded": 1_000,
            }
        )


def _provenance_hashes() -> dict[str, str]:
    hashes = {
        key: hashlib.sha256(key.encode("utf-8")).hexdigest() for key in N.REQUIRED_PROVENANCE_HASHES
    }
    hashes.update(N.fixed_contract_hashes())
    hashes["runner_sha256"] = hashlib.sha256(Path(N.__file__).resolve().read_bytes()).hexdigest()
    return hashes


def _complete_manifest(
    cell: N.CellPlan,
    artifact_root: Path,
    *,
    checkpoint: bytes = b"checkpoint",
    prediction: bytes = b"synthetic prediction bytes",
) -> dict[str, object]:
    provenance = _provenance_hashes()
    document = {
        **copy.deepcopy(N._fit_manifest_identity(cell)),
        "status": N.ArtifactStatus.COMPLETE.value,
        "provenance_hashes": provenance,
        "data_lineage_hashes": {key: provenance[key] for key in N.DATA_LINEAGE_HASHES},
        "execution_lineage_hashes": {key: provenance[key] for key in N.EXECUTION_LINEAGE_HASHES},
        "artifact_hashes": {},
        "training_execution_implemented": True,
    }
    checkpoint_path = artifact_root / N.checkpoint_relative_path(cell)
    prediction_path = artifact_root / N.prediction_relative_path(cell)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(checkpoint)
    prediction_path.write_bytes(prediction)
    document["artifact_hashes"] = {
        "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
        "prediction_sha256": hashlib.sha256(prediction).hexdigest(),
    }
    return document


def test_artifact_paths_are_unique_but_resume_and_subset_fail_closed(tmp_path: Path) -> None:
    plan = N.build_cell_plan()
    checkpoint_paths = {N.checkpoint_relative_path(cell) for cell in plan}
    prediction_paths = {N.prediction_relative_path(cell) for cell in plan}
    manifest_paths = {N.fit_manifest_relative_path(cell) for cell in plan}
    assert len(checkpoint_paths) == len(prediction_paths) == len(manifest_paths) == len(plan)

    subset = plan[:2]
    documents = [_complete_manifest(cell, tmp_path) for cell in subset]
    for document, cell in zip(documents, subset):
        with pytest.raises(
            N.ResumeContentValidationUnavailable,
            match=N.RESUME_CONTENT_VALIDATION_UNAVAILABLE_STATUS,
        ):
            N.validate_resume_manifest(
                document,
                cell,
                None,
                artifact_root=tmp_path,
            )
    with pytest.raises(
        N.ResumeContentValidationUnavailable,
        match=N.RESUME_CONTENT_VALIDATION_UNAVAILABLE_STATUS,
    ):
        N.validate_subset_completion_diagnostic(
            documents,
            subset,
            None,
            artifact_root=tmp_path,
        )
    with pytest.raises(N.ContractError, match="completion inventory"):
        N.validate_subset_completion_diagnostic(
            documents[:1],
            subset,
            None,
            artifact_root=tmp_path,
        )
    with pytest.raises(N.ContractError, match="duplicate fit"):
        N.validate_subset_completion_diagnostic(
            documents,
            (subset[0], subset[0]),
            None,
            artifact_root=tmp_path,
        )

    wrong = copy.deepcopy(documents[0])
    wrong["provenance_hashes"]["panel_sha256"] = "not-a-digest"
    with pytest.raises(N.ContractError, match="SHA-256"):
        N.validate_resume_manifest(
            wrong,
            subset[0],
            None,
            artifact_root=tmp_path,
        )

    unknown = copy.deepcopy(documents[0])
    unknown["invented_field"] = "silently accepted before hardening"
    with pytest.raises(N.ContractError, match="schema mismatch"):
        N.validate_resume_manifest(
            unknown,
            subset[0],
            None,
            artifact_root=tmp_path,
        )


def test_resume_never_treats_arbitrary_or_missing_artifact_bytes_as_valid(tmp_path: Path) -> None:
    cell = N.build_cell_plan()[0]
    document = _complete_manifest(cell, tmp_path)
    checkpoint_path = tmp_path / N.checkpoint_relative_path(cell)
    checkpoint_path.unlink()
    with pytest.raises(
        N.ResumeContentValidationUnavailable,
        match=N.RESUME_CONTENT_VALIDATION_UNAVAILABLE_STATUS,
    ):
        N.validate_resume_manifest(
            document,
            cell,
            None,
            artifact_root=tmp_path,
        )

    extra_nested = copy.deepcopy(document)
    extra_nested["artifact_hashes"]["invented"] = "0" * 64
    with pytest.raises(N.ContractError, match="artifact hash schema mismatch"):
        N.validate_resume_manifest(
            extra_nested,
            cell,
            None,
            artifact_root=tmp_path,
        )

    with pytest.raises(N.ContractError, match="immutable authority snapshot"):
        N.planned_fit_manifest(cell, _provenance_hashes())  # type: ignore[arg-type]


def test_full_completion_requires_canonical_plan_and_never_returns_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = N.build_cell_plan()
    with pytest.raises(N.ContractError, match="cell-plan arithmetic|canonical"):
        N.validate_full_completion_inventory(
            (),
            canonical[:2],
            None,
            artifact_root=tmp_path,
        )

    monkeypatch.setattr(N, "_validate_completion_documents", lambda *_args, **_kwargs: None)
    with pytest.raises(
        N.PredictionContentValidationUnavailable,
        match=N.FULL_COMPLETION_UNAVAILABLE_STATUS,
    ):
        N.validate_full_completion_inventory(
            (),
            canonical,
            None,
            artifact_root=tmp_path,
        )


def test_phase1_manifest_exposes_hashes_and_unimplemented_modules() -> None:
    manifest = N.phase1_plan_manifest(N.build_cell_plan())
    assert manifest["execution_implemented"] is False
    assert manifest["reads_panels"] is False
    assert manifest["writes_artifacts"] is False
    assert manifest["inventory"]["total_fit_units"] == 5_280
    assert manifest["inventory"]["selected_thermoroute_protocol_cells"] == 12
    assert manifest["inventory"]["selected_thermoroute_variant_configurations"] == 24
    assert manifest["artifact_contract"]["full_completion_status"] == (
        N.FULL_COMPLETION_UNAVAILABLE_STATUS
    )
    assert manifest["artifact_contract"]["resume_status"] == (
        N.RESUME_CONTENT_VALIDATION_UNAVAILABLE_STATUS
    )
    assert manifest["neural_execution_authority_contract"]["seal_read_count_per_validation"] == 1
    assert manifest["neural_execution_authority_contract"]["seal_object_is_recursively_immutable"]
    assert manifest["neural_execution_authority_contract"]["common_key_authority_status"] == (
        N.PRODUCTION_COMMON_KEY_AUTHORITY_STATUS
    )
    assert manifest["input_schema"]["forecast_key_authority"]["structural_authority_kind"] == (
        N.SYNTHETIC_FORECAST_KEY_AUTHORITY_KIND
    )
    assert len(manifest["hashes"]["cell_plan_sha256"]) == 64
    assert manifest["unimplemented_execution_modules"]
    assert "cells" not in manifest


def test_dry_run_reads_and_writes_no_files(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def prohibited(*_args, **_kwargs):
        raise AssertionError("dry-run performed filesystem I/O")

    monkeypatch.setattr(Path, "read_bytes", prohibited)
    monkeypatch.setattr(Path, "write_bytes", prohibited)
    monkeypatch.setattr(Path, "write_text", prohibited)
    args = N.build_parser().parse_args(["--dry-run"])
    assert N.run(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["phase"] == "PHASE_1_CONTRACT_ONLY"
    assert output["inventory"]["total_fit_units"] == 5_280


def test_draft_protocol_rejects_before_execution_or_output_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = tmp_path / "v4.yaml"
    protocol.write_text(
        "\n".join(
            (
                f"protocol_id: {N.EXPECTED_PROTOCOL_ID}",
                f"version: {N.EXPECTED_PROTOCOL_VERSION}",
                "status: DRAFT_NOT_SEALED",
                "execution_authorized: false",
                "seal_path: null",
            )
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "must-not-exist"

    def unexpected_execution(*_args, **_kwargs):
        raise AssertionError("draft reached the execution boundary")

    monkeypatch.setattr(N, "execute_training", unexpected_execution)
    args = N.build_parser().parse_args(
        ["--protocol", str(protocol), "--output-dir", str(output_dir)]
    )
    with pytest.raises(N.GovernanceError, match="pinned to the canonical"):
        N.run(args)
    assert not output_dir.exists()


def _write_test_seal(
    tmp_path: Path,
    *,
    neural_execution_bindings: Mapping[str, object] | None = None,
) -> Path:
    protocol = tmp_path / "v4.yaml"
    seal_path = tmp_path / "v4_seal.json"
    protocol.write_text(
        "\n".join(
            (
                f"protocol_id: {N.EXPECTED_PROTOCOL_ID}",
                f"version: {N.EXPECTED_PROTOCOL_VERSION}",
                "status: SEALED",
                "execution_authorized: true",
                f"seal_path: {json.dumps(str(seal_path))}",
            )
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(protocol.read_bytes()).hexdigest()
    seal: dict[str, object] = {
        "format": N.PROTOCOL_SEAL_FORMAT,
        "status": "SEALED",
        "execution_authorized": True,
        "protocol": {
            "protocol_id": N.EXPECTED_PROTOCOL_ID,
            "version": N.EXPECTED_PROTOCOL_VERSION,
            "path": str(protocol.resolve()),
            "sha256": digest,
        },
    }
    if neural_execution_bindings is not None:
        seal["neural_execution_bindings"] = dict(neural_execution_bindings)
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    return protocol


def _execution_bindings(tmp_path: Path) -> dict[str, object]:
    registry_dir = tmp_path / "registries"
    registry_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "runner_sha256": Path(N.__file__).resolve(),
        "cell_plan_sha256": registry_dir / "cell_plan.json",
        "model_registry_sha256": registry_dir / "model_registry.json",
        "fold_registry_sha256": registry_dir / "fold_registry.json",
        "input_registry_sha256": registry_dir / "input_registry.json",
        "contrast_registry_sha256": registry_dir / "contrast_registry.json",
    }
    paths["cell_plan_sha256"].write_bytes(
        N._canonical_json_bytes([cell.to_record() for cell in N.build_cell_plan()])
    )
    for key, registry_format in N.NEURAL_REGISTRY_FORMATS_REQUIRING_PARSERS.items():
        paths[key].write_bytes(N._canonical_json_bytes({"format": registry_format}))
    bindings = {}
    for key, path in paths.items():
        payload = path.read_bytes()
        bindings[key] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return bindings


def _patch_synthetic_canonical_authority(
    monkeypatch: pytest.MonkeyPatch,
    protocol: Path,
    canonical_bindings: Mapping[str, object],
) -> None:
    paths: dict[str, Path] = {}
    for key in N.NEURAL_EXECUTION_BINDING_KEYS:
        record = canonical_bindings[key]
        assert isinstance(record, Mapping)
        paths[key] = Path(str(record["path"])).resolve()
    monkeypatch.setattr(N, "CANONICAL_SEALED_PROTOCOL", protocol.resolve())
    monkeypatch.setattr(
        N,
        "CANONICAL_PROTOCOL_SEAL",
        (protocol.parent / "v4_seal.json").resolve(),
    )
    monkeypatch.setattr(
        N,
        "CANONICAL_NEURAL_REGISTRY_PATHS",
        MappingProxyType(paths),
    )


def test_even_a_valid_test_seal_stops_at_explicit_phase1_execution_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _write_test_seal(tmp_path)
    byte_binding = N.validate_protocol_byte_binding_only(protocol)
    assert byte_binding["scope"] == "PROTOCOL_BYTE_BINDING_ONLY"
    assert byte_binding["neural_execution_authorized"] is False
    with pytest.raises(N.GovernanceError, match="pinned to the canonical"):
        N.validate_neural_execution_authority(protocol)
    monkeypatch.setattr(N, "CANONICAL_SEALED_PROTOCOL", protocol.resolve())
    monkeypatch.setattr(
        N,
        "CANONICAL_PROTOCOL_SEAL",
        (tmp_path / "v4_seal.json").resolve(),
    )
    with pytest.raises(N.GovernanceError, match="lacks neural_execution_bindings"):
        N.validate_neural_execution_authority(protocol)
    output_dir = tmp_path / "must-not-exist"
    args = N.build_parser().parse_args(
        ["--protocol", str(protocol), "--output-dir", str(output_dir)]
    )
    with pytest.raises(N.GovernanceError, match="lacks neural_execution_bindings"):
        N.run(args)
    assert not output_dir.exists()


def test_full_neural_authority_is_an_immutable_but_semantically_terminal_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bindings = _execution_bindings(tmp_path)
    protocol = _write_test_seal(tmp_path, neural_execution_bindings=bindings)
    _patch_synthetic_canonical_authority(monkeypatch, protocol, bindings)
    snapshot = N._capture_neural_authority_snapshot(protocol)
    assert type(snapshot) is N.NeuralAuthoritySnapshot
    assert snapshot.semantic_validation_complete is False
    assert set(snapshot.binding_bytes) == set(N.NEURAL_EXECUTION_BINDING_KEYS)
    assert len(snapshot.snapshot_sha256) == 64
    with pytest.raises(TypeError):
        snapshot.binding_bytes["input_registry_sha256"] = b"forged"  # type: ignore[index]
    with pytest.raises(TypeError, match="cannot be constructed directly"):
        N.NeuralAuthoritySnapshot()  # type: ignore[call-arg]
    with pytest.raises(
        N.NeuralAuthoritySemanticValidationUnavailable,
        match=N.NEURAL_AUTHORITY_SEMANTIC_VALIDATION_UNAVAILABLE_STATUS,
    ):
        N.validate_neural_execution_authority(protocol)
    with pytest.raises(
        N.NeuralAuthoritySemanticValidationUnavailable,
        match=N.NEURAL_AUTHORITY_SEMANTIC_VALIDATION_UNAVAILABLE_STATUS,
    ):
        N.planned_fit_manifest(N.build_cell_plan()[0], snapshot)
    cell = N.build_cell_plan()[0]
    document = _complete_manifest(cell, tmp_path / "artifacts")
    with pytest.raises(
        N.ResumeContentValidationUnavailable,
        match=N.RESUME_CONTENT_VALIDATION_UNAVAILABLE_STATUS,
    ):
        N.validate_resume_manifest(
            document,
            cell,
            snapshot,
            artifact_root=tmp_path / "artifacts",
        )
    with pytest.raises(N.Phase1ExecutionUnavailable, match="immutable authority snapshot"):
        N.execute_training((), {"neural_execution_authorized": True}, tmp_path, resume=False)  # type: ignore[arg-type]

    input_record = bindings["input_registry_sha256"]
    assert isinstance(input_record, Mapping)
    captured_input = snapshot.binding_bytes["input_registry_sha256"]
    Path(str(input_record["path"])).write_bytes(b"tampered after seal")
    assert snapshot.binding_bytes["input_registry_sha256"] == captured_input
    with pytest.raises(N.GovernanceError, match="SHA-256 mismatch"):
        N._capture_neural_authority_snapshot(protocol)


def test_non_dry_phase1_never_reads_neural_registries_or_writes_even_with_full_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bindings = _execution_bindings(tmp_path)
    protocol = _write_test_seal(tmp_path, neural_execution_bindings=bindings)
    _patch_synthetic_canonical_authority(monkeypatch, protocol, bindings)
    output_dir = tmp_path / "must-not-exist"
    args = N.build_parser().parse_args(
        ["--protocol", str(protocol), "--output-dir", str(output_dir)]
    )
    with pytest.raises(
        N.NeuralAuthoritySemanticValidationUnavailable,
        match=N.NEURAL_AUTHORITY_SEMANTIC_VALIDATION_UNAVAILABLE_STATUS,
    ):
        N.run(args)
    assert not output_dir.exists()


def test_neural_authority_requires_exact_canonical_paths_before_registry_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _execution_bindings(tmp_path)
    wrong = copy.deepcopy(base)
    wrong_path = tmp_path / "wrong-but-existing.json"
    wrong_path.write_bytes(b"opaque")
    wrong["input_registry_sha256"] = {
        "path": str(wrong_path),
        "sha256": hashlib.sha256(b"opaque").hexdigest(),
    }
    protocol = _write_test_seal(tmp_path, neural_execution_bindings=wrong)
    _patch_synthetic_canonical_authority(monkeypatch, protocol, base)
    with pytest.raises(N.GovernanceError, match="path is not canonical"):
        N.validate_neural_execution_authority(protocol)

    missing = tmp_path / "missing.json"
    with pytest.raises(N.GovernanceError, match="is missing"):
        N._read_regular_declared_file(str(missing), label="synthetic registry")
    with pytest.raises(N.GovernanceError, match="regular non-symlink"):
        N._read_regular_declared_file(str(tmp_path), label="synthetic registry")

    real_parent = tmp_path / "real-registry-parent"
    real_parent.mkdir()
    parent_target = real_parent / "input.json"
    parent_target.write_bytes(b"parent target")
    parent_symlink = tmp_path / "linked-registry-parent"
    parent_symlink.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(N.GovernanceError, match="symlink path component"):
        N._read_regular_declared_file(
            str(parent_symlink / "input.json"), label="synthetic registry"
        )

    target = tmp_path / "symlink-target.json"
    target.write_bytes(b"target")
    symlink = tmp_path / "symlink-registry.json"
    symlink.symlink_to(target)
    with pytest.raises(N.GovernanceError, match="symlink"):
        N._read_regular_declared_file(str(symlink), label="synthetic registry")

    with pytest.raises(N.GovernanceError, match="path-escape"):
        N._read_regular_declared_file("../escaped.json", label="synthetic registry")


def test_neural_authority_binds_executing_runner_and_semantic_cell_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bindings = _execution_bindings(tmp_path)
    runner_record = bindings["runner_sha256"]
    assert isinstance(runner_record, Mapping)
    copied_runner = tmp_path / "copied-runner.py"
    copied_runner.write_bytes(Path(str(runner_record["path"])).read_bytes())
    wrong_runner = copy.deepcopy(bindings)
    wrong_runner["runner_sha256"]["path"] = str(copied_runner)
    protocol = _write_test_seal(tmp_path, neural_execution_bindings=wrong_runner)
    _patch_synthetic_canonical_authority(monkeypatch, protocol, bindings)
    with pytest.raises(N.GovernanceError, match="path is not canonical"):
        N.validate_neural_execution_authority(protocol)

    wrong_plan = copy.deepcopy(bindings)
    plan_record = wrong_plan["cell_plan_sha256"]
    assert isinstance(plan_record, dict)
    plan_path = Path(str(plan_record["path"]))
    plan_path.write_bytes(b"[]")
    plan_record["sha256"] = hashlib.sha256(b"[]").hexdigest()
    protocol = _write_test_seal(tmp_path, neural_execution_bindings=wrong_plan)
    _patch_synthetic_canonical_authority(monkeypatch, protocol, bindings)
    with pytest.raises(N.GovernanceError, match="exact canonical 5,280-fit plan"):
        N.validate_neural_execution_authority(protocol)


def test_neural_authority_reads_and_validates_one_immutable_seal_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bindings = _execution_bindings(tmp_path)
    protocol = _write_test_seal(tmp_path, neural_execution_bindings=bindings)
    _patch_synthetic_canonical_authority(monkeypatch, protocol, bindings)
    original = N._read_regular_declared_file
    seal_reads = 0

    def race_probe(*args, **kwargs):
        nonlocal seal_reads
        if kwargs.get("label") == "v4 seal":
            seal_reads += 1
            if seal_reads > 1:
                raise AssertionError("validator reread a potentially changed seal")
        return original(*args, **kwargs)

    monkeypatch.setattr(N, "_read_regular_declared_file", race_probe)
    snapshot = N._capture_neural_authority_snapshot(protocol)
    assert snapshot.seal_bytes
    assert seal_reads == 1


def test_production_authority_paths_are_pinned_and_currently_absent() -> None:
    assert N.CANONICAL_SEALED_PROTOCOL.name == ("wrr_information_regimes_protocol_v4_sealed.yaml")
    assert N.CANONICAL_PROTOCOL_SEAL.name == ("wrr_information_regimes_protocol_v4_seal.json")
    assert N.DEFAULT_PROTOCOL == N.CANONICAL_SEALED_PROTOCOL
    assert not N.CANONICAL_SEALED_PROTOCOL.exists()
    assert not N.CANONICAL_PROTOCOL_SEAL.exists()
    with pytest.raises(N.GovernanceError, match="is missing"):
        N.validate_neural_execution_authority(N.DEFAULT_PROTOCOL)

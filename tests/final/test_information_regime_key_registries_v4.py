"""Synthetic-only adversarial tests for the v4 Phase-1 key registries."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import scripts.final.build_information_regime_key_registries_v4 as registry

SITES = ("00000001", "00000002", "123456789012345")
PRIMARY_START = date(2021, 3, 23)
F2A_START = date(2021, 3, 30)
TARGET_END = date(2021, 4, 8)
MINIMUM_TARGETS = 2


@dataclass
class SyntheticFixture:
    root: Path
    protocol_v4: Path
    route_protocol: Path
    station_registry: Path
    panel: Path
    legacy: Path
    acquisition_root: Path
    acquisition_manifest: Path
    snapshot_index: Path
    verification_report: Path
    code_paths: dict[str, Path]
    config: registry.RegistryBuildConfig

    @property
    def paths(self) -> registry.RegistryInputPaths:
        return registry.RegistryInputPaths(
            protocol_v4=self.protocol_v4,
            route_a_protocol=self.route_protocol,
            station_registry=self.station_registry,
            raw_holdout_panel=self.panel,
            legacy_forecast_registry=self.legacy,
            f2a_verification_report=self.verification_report,
            f2a_acquisition_manifest=self.acquisition_manifest,
            f2a_acquisition_root=self.acquisition_root,
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_write(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(registry._canonical_json_bytes(document))


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _v4_protocol_text() -> str:
    return f"""\
protocol_id: {registry.EXPECTED_V4_PROTOCOL_ID}
version: 4
status: DRAFT_NOT_SEALED
execution_authorized: false
seal_path: null
axes:
  F_future_forcing:
    F2a:
      coherent_single_initialization: false
      recovery_denominator: F3_temperature_only
      frozen_acquisition_scope:
        lead_days: [1, 3, 7]
        value_per_forecast_key: target-day daily-mean air temperature
        all_intermediate_leads_acquired: false
common_key_and_reportability_contract:
  key_fields: [site_id, issue_time, target_time, lead_days]
  reportability:
    minimum_paired_targets_per_station_lead: {MINIMUM_TARGETS}
history_quality_sensitivity:
  window_days: 32
  raw_observation_basis: before_imputation
"""


def _route_protocol_document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "protocol_id": registry.EXPECTED_ROUTE_A_PROTOCOL_ID,
        "status": "FROZEN_NOT_ACQUIRED",
        "time_holdout": {
            "start": PRIMARY_START.isoformat(),
            "end": TARGET_END.isoformat(),
            "secondary_nwp_common_lead_target_start": F2A_START.isoformat(),
        },
        "secondary_archived_nwp_contract": {
            "status": "IMPLEMENTED_NOT_ACQUIRED",
            "role": "OPTIONAL_SECONDARY_AVAILABILITY_SENSITIVITY_NOT_PRIMARY_INPUT",
            "contains_outcome_labels": False,
            "consumed_by_primary_models": False,
            "primary_evaluation_dependency": False,
            "provider": registry.EXPECTED_PROVIDER,
            "provider_documentation": registry.EXPECTED_PROVIDER_DOCUMENTATION,
            "api_endpoint": "https://previous-runs-api.open-meteo.com/v1/forecast",
            "upstream_model": registry.EXPECTED_UPSTREAM_MODEL,
            "open_meteo_model_parameter": registry.EXPECTED_OPEN_METEO_MODEL,
            "variable": registry.EXPECTED_SOURCE_VARIABLE,
            "unit": "degrees_C",
            "lead_days": list(registry.LEADS),
            "lead_fields": [f"temperature_2m_previous_day{lead}" for lead in registry.LEADS],
            "lead_semantics": (
                "For each valid hour, previous_dayN is aligned N times 24 hours "
                "before that valid hour. This is a rolling fixed-lead composite, "
                "not the output of one identified model initialization."
            ),
            "timezone": "GMT_UTC_PLUS_00",
            "daily_aggregation": (
                "arithmetic mean only when all 24 valid-hour values are finite; "
                "otherwise retain an unavailable row and hour count"
            ),
            "archive_run_start": "2021-03-23",
            "secondary_common_lead_target_start": F2A_START.isoformat(),
            "request_partition": (
                "one stable site_no and one UTC calendar month per immutable raw snapshot"
            ),
            "raw_snapshot_required": True,
            "derived_blocks_may_be_overwritten": False,
            "selection_or_tuning_from_predictor_availability": False,
        },
    }


def _station_registry() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site_no": SITES,
            "legacy_site_id": SITES,
            "ok": [True] * 3,
            "wtemp_cov": [1.0] * 3,
            "flow_cov": [1.0] * 3,
            "wtemp_cov_test": [1.0] * 3,
            "flow_cov_test": [1.0] * 3,
            "wlevel_cov": [1.0] * 3,
            "n_days": [100] * 3,
            "station_nm": ["Synthetic A", "Synthetic B", "Synthetic C"],
            "lat": [40.0, 41.0, 42.0],
            "lon": [-100.0, -101.0, -102.0],
            "state": ["AA", "BB", "CC"],
            "huc_cd": ["01000000", "02000000", "03000000"],
            "huc2": ["01", "02", "03"],
            "drain_area_va": [10.0, 20.0, 30.0],
            "huc_metadata_status": ["SYNTHETIC"] * 3,
        },
        columns=registry.STATION_REGISTRY_COLUMNS,
    )


def _panel() -> pd.DataFrame:
    start = pd.Timestamp(PRIMARY_START) - pd.Timedelta(days=31)
    dates = pd.date_range(start, pd.Timestamp(TARGET_END), freq="D")
    sparse_c = {
        pd.Timestamp("2021-03-29"),
        pd.Timestamp("2021-03-30"),
        pd.Timestamp("2021-04-01"),
        pd.Timestamp("2021-04-05"),
    }
    rows: list[dict[str, object]] = []
    for site_index, site in enumerate(SITES):
        for day_index, current in enumerate(dates):
            wtemp = float(5.0 + site_index + day_index / 100.0)
            if site == SITES[2] and current not in sparse_c:
                wtemp = float("nan")
            rows.append(
                {
                    "DATE": current,
                    "site_id": site,
                    "WTEMP": wtemp,
                    "FLOW": float(10 + site_index),
                    "WLEVEL": float(1 + site_index),
                    "TEMP": float(2 + site_index + day_index / 10.0),
                    "PRCP": 0.0,
                    "WDSP": 1.0,
                    "RHMEAN": 50.0,
                    "DH": 100.0,
                }
            )
    return pd.DataFrame(rows, columns=registry.PANEL_COLUMNS)


def _legacy_from_panel(panel: pd.DataFrame) -> pd.DataFrame:
    lookup = {
        (str(row.site_id), pd.Timestamp(row.DATE)): float(row.WTEMP)
        for row in panel.itertuples(index=False)
    }
    rows: list[dict[str, object]] = []
    for row in panel.itertuples(index=False):
        issue = pd.Timestamp(row.DATE)
        if issue.date() < PRIMARY_START or not np.isfinite(row.WTEMP):
            continue
        for lead in registry.LEADS:
            target = issue + pd.Timedelta(days=lead)
            y_panel = lookup.get((str(row.site_id), target), float("nan"))
            if target.date() > TARGET_END or not np.isfinite(y_panel):
                continue
            # Preserve a deliberately different but authority-legal binary
            # representation so the builder must copy legacy y_true.
            y_legacy = y_panel + (1e-6 if len(rows) % 2 == 0 else -1e-6)
            rows.append(
                {
                    "key_id": registry._key_id(str(row.site_id), issue, target, lead),
                    "cohort": "temporal",
                    "task": "known_site",
                    "site_id": str(row.site_id),
                    "issue_date": issue,
                    "target_date": target,
                    "horizon": lead,
                    "period": "2021-2023",
                    "target_observed": True,
                    "y_true": y_legacy,
                    # Deliberately defective legacy context metadata.  These
                    # columns must be ignored, not repaired in-place.
                    "issue_wtemp_observed": False,
                    "days_since_last_observed_wtemp": -999,
                    "n_observed_wtemp_7d": 7.0,
                    "fraction_observed_wtemp_7d": 1.0,
                    "n_observed_wtemp_14d": 14.0,
                    "fraction_observed_wtemp_14d": 1.0,
                    "n_observed_wtemp_32d": 32.0,
                    "fraction_observed_wtemp_32d": 1.0,
                }
            )
    return (
        pd.DataFrame(rows, columns=registry.LEGACY_REGISTRY_COLUMNS)
        .sort_values(["site_id", "horizon", "issue_date", "target_date"])
        .reset_index(drop=True)
    )


def _f2a_frame(
    site: str,
    chunk_start: date,
    chunk_end: date,
    *,
    request_sha: str,
    response_sha: str,
    coordinates: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    lat, lon = coordinates[site]
    rows: list[dict[str, object]] = []
    for lead in registry.LEADS:
        for target in pd.date_range(chunk_start, chunk_end, freq="D"):
            rows.append(
                {
                    "site_no": site,
                    "requested_lat": lat,
                    "requested_lon": lon,
                    "horizon": lead,
                    "issue_date": target - pd.Timedelta(days=lead),
                    "target_date": target,
                    "air_temp_2m_mean_c": float(lead + target.day / 100.0),
                    "available_hour_count": 24,
                    "complete_target_day": True,
                    "lead_field": f"temperature_2m_previous_day{lead}",
                    "source_provider": registry.EXPECTED_PROVIDER,
                    "upstream_model": registry.EXPECTED_UPSTREAM_MODEL,
                    "open_meteo_model": registry.EXPECTED_OPEN_METEO_MODEL,
                    "issue_semantics": registry.EXPECTED_ISSUE_SEMANTICS,
                    "response_grid_lat": lat + 0.01,
                    "response_grid_lon": lon - 0.01,
                    "request_sha256": request_sha,
                    "response_sha256": response_sha,
                }
            )
    return pd.DataFrame(rows, columns=registry.F2A_PARQUET_COLUMNS)


def _coverage_documents(
    acquisition_root: Path,
    *,
    station_count: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int, int]:
    frames = [pd.read_parquet(path) for path in sorted(acquisition_root.glob("*/*.parquet"))]
    acquired = pd.concat(frames, ignore_index=True)
    expected_days = (TARGET_END - F2A_START).days + 1
    rows: list[dict[str, object]] = []
    for site in SITES:
        for lead in registry.LEADS:
            group = acquired[(acquired["site_no"] == site) & (acquired["horizon"] == lead)]
            complete = int(group["complete_target_day"].sum())
            fraction = complete / expected_days
            rows.append(
                {
                    "site_no": site,
                    "lead_days": lead,
                    "expected_target_days": expected_days,
                    "complete_target_days": complete,
                    "coverage_fraction": fraction,
                    "passes_0_90_gate": fraction >= registry.EXPECTED_COVERAGE_THRESHOLD,
                }
            )
    by_lead: list[dict[str, object]] = []
    for lead in registry.LEADS:
        lead_frame = acquired[acquired["horizon"] == lead]
        complete = int(lead_frame["complete_target_day"].sum())
        denominator = station_count * expected_days
        by_lead.append(
            {
                "lead_days": lead,
                "expected_station_days": denominator,
                "complete_station_days": complete,
                "coverage_fraction": complete / denominator,
                "stations_passing_0_90_gate": sum(
                    row["passes_0_90_gate"] for row in rows if row["lead_days"] == lead
                ),
                "station_count": station_count,
            }
        )
    return rows, by_lead, len(acquired), int(acquired["complete_target_day"].sum())


def _refresh_manifest_and_report(fixture: SyntheticFixture) -> None:
    sidecars: list[dict[str, Any]] = []
    for artifact in sorted(fixture.acquisition_root.glob("*/*.parquet")):
        sidecar = artifact.with_suffix(".provenance.json")
        document = json.loads(sidecar.read_text(encoding="utf-8"))
        frame = pd.read_parquet(artifact)
        document["artifact_sha256"] = _sha256(artifact)
        document["row_count"] = len(frame)
        document["complete_row_count"] = int(frame["complete_target_day"].sum())
        _canonical_write(sidecar, document)
        sidecars.append(document)
    sidecars.sort(key=lambda row: (str(row["site_no"]), str(row["chunk_start"])))
    manifest = json.loads(fixture.acquisition_manifest.read_text(encoding="utf-8"))
    manifest["chunks"] = sidecars
    manifest["row_count"] = sum(int(row["row_count"]) for row in sidecars)
    manifest["complete_row_count"] = sum(int(row["complete_row_count"]) for row in sidecars)
    _canonical_write(fixture.acquisition_manifest, manifest)
    coverage, by_lead, total, complete = _coverage_documents(
        fixture.acquisition_root, station_count=len(SITES)
    )
    report = json.loads(fixture.verification_report.read_text(encoding="utf-8"))
    report["evidence_hashes"]["acquisition_manifest"]["sha256"] = _sha256(
        fixture.acquisition_manifest
    )
    report["inventory"]["forecast_rows"] = total
    report["inventory"]["complete_forecast_rows"] = complete
    report["coverage_by_station_and_lead"] = coverage
    report["coverage_by_lead"] = by_lead
    gate = all(row["passes_0_90_gate"] for row in coverage)
    report["coverage_gate_passed"] = gate
    report["status"] = (
        "VERIFIED_COMPLETE_COVERAGE_GATE_PASSED"
        if gate
        else "VERIFIED_COMPLETE_COVERAGE_GATE_FAILED"
    )
    _canonical_write(fixture.verification_report, report)


def _build_fixture(root: Path) -> SyntheticFixture:
    root.mkdir(parents=True)
    (root / "requirements-lock.txt").write_text(
        "numpy==1.26.4\npandas==2.2.2\npyarrow==24.0.0\n", encoding="utf-8"
    )
    (root / "requirements-lock-py312-hashed.txt").write_text(
        "# synthetic Python 3.12 hashed lock\n", encoding="utf-8"
    )
    (root / "pyproject.toml").write_text(
        "[project]\nname='synthetic-key-registry'\nversion='0.0.0'\n", encoding="utf-8"
    )
    protocol_v4 = root / "protocols" / "wrr_information_regimes_protocol_v4.yaml"
    protocol_v4.parent.mkdir()
    protocol_v4.write_text(_v4_protocol_text(), encoding="utf-8")
    route_protocol = root / "protocols" / "route_a_confirmatory_v1.json"
    route_protocol.write_text(
        json.dumps(_route_protocol_document(), indent=2) + "\n", encoding="utf-8"
    )

    station_registry = root / "data_usgs" / "station_registry_v1.csv"
    station_registry.parent.mkdir()
    station_frame = _station_registry()
    station_frame.to_csv(station_registry, index=False)
    panel_path = root / "outputs" / "conventional" / "panel_2021_2023.parquet"
    panel_path.parent.mkdir(parents=True)
    panel = _panel()
    panel.to_parquet(panel_path, index=False)
    legacy_path = root / "outputs" / "final" / "forecast_keys.parquet"
    legacy_path.parent.mkdir(parents=True)
    _legacy_from_panel(panel).to_parquet(legacy_path, index=False)

    acquisition_root = root / "data_usgs" / "confirmatory_predictors" / "gfs-previous-runs-v1"
    acquisition_root.mkdir(parents=True)
    snapshot_index = (
        root
        / "data_usgs"
        / "raw_snapshots"
        / "openmeteo-gfs-previous-runs-v1"
        / "snapshot_index.json"
    )
    _canonical_write(snapshot_index, {"schema_version": 1, "snapshot_count": 6, "records": []})
    coordinates = {
        str(row.site_no): (float(row.lat), float(row.lon))
        for row in station_frame.itertuples(index=False)
    }
    route_sha = _sha256(route_protocol)
    chunks: list[dict[str, object]] = []
    for site in SITES:
        for chunk_start, chunk_end in registry._month_chunks(F2A_START, TARGET_END):
            request_sha = hashlib.sha256(
                f"request|{site}|{chunk_start}|{chunk_end}".encode()
            ).hexdigest()
            response_sha = hashlib.sha256(
                f"response|{site}|{chunk_start}|{chunk_end}".encode()
            ).hexdigest()
            artifact = acquisition_root / site / f"{chunk_start:%Y-%m}.parquet"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            frame = _f2a_frame(
                site,
                chunk_start,
                chunk_end,
                request_sha=request_sha,
                response_sha=response_sha,
                coordinates=coordinates,
            )
            frame.to_parquet(artifact, index=False)
            chunk = {
                "schema_version": 1,
                "artifact": _relative(artifact, root),
                "artifact_sha256": _sha256(artifact),
                "site_no": site,
                "chunk_start": chunk_start.isoformat(),
                "chunk_end": chunk_end.isoformat(),
                "row_count": len(frame),
                "complete_row_count": int(frame["complete_target_day"].sum()),
                "request_sha256": request_sha,
                "response_sha256": response_sha,
                "retrieved_at_utc": "2026-08-09T00:00:00+00:00",
                "labels_requested_or_read": False,
                "protocol_sha256": route_sha,
            }
            _canonical_write(artifact.with_suffix(".provenance.json"), chunk)
            chunks.append(chunk)

    registry_lineage = [
        {
            "path": _relative(station_registry, root),
            "sha256": _sha256(station_registry),
            "columns_read": ["site_no", "lat", "lon"],
            "row_count": len(SITES),
        }
    ]
    manifest_document = {
        "schema_version": 1,
        "artifact_role": "OPTIONAL_SECONDARY_NWP_AVAILABILITY_SENSITIVITY",
        "protocol": _relative(route_protocol, root),
        "protocol_sha256": route_sha,
        "source_provider": registry.EXPECTED_PROVIDER,
        "source_documentation": registry.EXPECTED_PROVIDER_DOCUMENTATION,
        "upstream_model": registry.EXPECTED_UPSTREAM_MODEL,
        "open_meteo_model": registry.EXPECTED_OPEN_METEO_MODEL,
        "source_variable": registry.EXPECTED_SOURCE_VARIABLE,
        "lead_days": list(registry.LEADS),
        "lead_fields": [f"temperature_2m_previous_day{lead}" for lead in registry.LEADS],
        "issue_semantics": registry.EXPECTED_ISSUE_SEMANTICS,
        "timezone": "GMT (UTC+00:00)",
        "daily_aggregation": (
            "arithmetic mean of exactly 24 finite valid-hour values; incomplete "
            "days retained with NaN predictor and availability count"
        ),
        "archive_run_start": "2021-03-23",
        "common_valid_target_start": F2A_START.isoformat(),
        "target_end": TARGET_END.isoformat(),
        "station_count": len(SITES),
        "row_count": sum(int(row["row_count"]) for row in chunks),
        "complete_row_count": sum(int(row["complete_row_count"]) for row in chunks),
        "request_partition": "one stable site_no by one UTC calendar month",
        "immutable_outputs": True,
        "consumed_by_primary_route_a_models": False,
        "primary_evaluation_dependency": False,
        "labels_requested_or_read": False,
        "outcome_endpoint_called": False,
        "registry_inputs": registry_lineage,
        "raw_snapshot_index": _relative(snapshot_index, root),
        "raw_snapshot_index_sha256": _sha256(snapshot_index),
        "chunks": chunks,
    }
    acquisition_manifest = acquisition_root / "manifest.json"
    _canonical_write(acquisition_manifest, manifest_document)

    code_paths: dict[str, Path] = {}
    for role in (
        "acquisition_script",
        "independent_verifier",
        "nwp_contract_module",
        "snapshot_store_module",
    ):
        path = root / "code" / f"{role}.py"
        path.parent.mkdir(exist_ok=True)
        path.write_text(f"# synthetic {role}\n", encoding="utf-8")
        code_paths[role] = path
    coverage, by_lead, total, complete = _coverage_documents(
        acquisition_root, station_count=len(SITES)
    )
    report_document = {
        "schema_version": 1,
        "verifier_id": registry.EXPECTED_VERIFIER_ID,
        "status": "VERIFIED_COMPLETE_COVERAGE_GATE_PASSED",
        "integrity_verification_passed": True,
        "coverage_gate_threshold": registry.EXPECTED_COVERAGE_THRESHOLD,
        "coverage_gate_passed": True,
        "arm_promoted": False,
        "promotion_requires_separate_protocol_authorization": True,
        "contains_or_reads_outcome_labels": False,
        "input_access_boundary": {
            "registry_columns_read": ["site_no", "lat", "lon"],
            "water_temperature_outcomes_read": False,
            "flow_outcomes_read": False,
            "panel_artifacts_read": False,
        },
        "information_semantics": {
            "fixed_lead_composite": True,
            "coherent_single_initialization": False,
            "as_issued_operational_forecast_archive": False,
            "issue_semantics": registry.EXPECTED_ISSUE_SEMANTICS,
        },
        "inventory": {
            "target_start": F2A_START.isoformat(),
            "target_end": TARGET_END.isoformat(),
            "lead_days": list(registry.LEADS),
            "station_count": len(SITES),
            "calendar_months_per_station": 2,
            "station_month_chunks": 6,
            "forecast_rows": total,
            "complete_forecast_rows": complete,
        },
        "coverage_by_station_and_lead": coverage,
        "coverage_by_lead": by_lead,
        "evidence_hashes": {
            "protocol": {"path": _relative(route_protocol, root), "sha256": route_sha},
            "input_registries": registry_lineage,
            "code": {
                role: {"path": _relative(path, root), "sha256": _sha256(path)}
                for role, path in sorted(code_paths.items())
            },
            "acquisition_manifest": {
                "path": _relative(acquisition_manifest, root),
                "sha256": _sha256(acquisition_manifest),
            },
            "snapshot_index": {
                "path": _relative(snapshot_index, root),
                "sha256": _sha256(snapshot_index),
            },
        },
    }
    verification_report = root / "outputs" / "final" / "f2a_verification.json"
    _canonical_write(verification_report, report_document)
    config = registry.RegistryBuildConfig(
        mode="synthetic",
        primary_issue_start=PRIMARY_START,
        primary_target_end=TARGET_END,
        f2a_target_start=F2A_START,
        minimum_targets=MINIMUM_TARGETS,
        expected_registry_stations=None,
        expected_raw_stations=3,
        expected_reportable_stations=2,
        expected_excluded_sites=(SITES[2],),
    )
    return SyntheticFixture(
        root=root,
        protocol_v4=protocol_v4,
        route_protocol=route_protocol,
        station_registry=station_registry,
        panel=panel_path,
        legacy=legacy_path,
        acquisition_root=acquisition_root,
        acquisition_manifest=acquisition_manifest,
        snapshot_index=snapshot_index,
        verification_report=verification_report,
        code_paths=code_paths,
        config=config,
    )


@pytest.fixture
def synthetic(tmp_path: Path) -> SyntheticFixture:
    return _build_fixture(tmp_path / "synthetic-repository")


def _build(fixture: SyntheticFixture) -> registry.RegistryArtifactBundle:
    return registry.build_registry_artifacts(
        fixture.paths,
        config=fixture.config,
        repository_root=fixture.root,
        input_root=fixture.root,
    )


def _mutate_block_and_reseal(
    fixture: SyntheticFixture,
    mutate: Any,
) -> Path:
    artifact = min(fixture.acquisition_root.glob("*/*.parquet"))
    frame = pd.read_parquet(artifact)
    mutated = mutate(frame.copy())
    mutated.to_parquet(artifact, index=False)
    _refresh_manifest_and_report(fixture)
    return artifact


def test_builds_primary_and_temperature_only_registries_before_scores(
    synthetic: SyntheticFixture,
) -> None:
    bundle = _build(synthetic)
    primary = bundle.manifest["primary_registry"]
    assert primary["raw_station_count"] == 3
    assert primary["reportable_station_count"] == 2
    assert primary["excluded_site_ids"] == [SITES[2]]
    assert primary["legacy_key_identity_exact_match"] is True
    assert primary["maximum_panel_vs_legacy_y_difference_c"] == pytest.approx(1e-6)
    assert set(bundle.primary_reportable["site_id"]) == set(SITES[:2])
    assert bundle.primary_raw["issue_wtemp_observed"].all()
    assert bundle.primary_raw["days_since_last_observed_wtemp"].eq(0).all()
    legacy = pd.read_parquet(synthetic.legacy)[["key_id", "y_true"]]
    observed = bundle.primary_raw[["key_id", "y_true"]].merge(
        legacy, on="key_id", suffixes=("_new", "_legacy"), validate="one_to_one"
    )
    assert np.array_equal(observed["y_true_new"], observed["y_true_legacy"])
    assert bundle.manifest["model_scores_read_or_accepted"] is False
    f2a = bundle.manifest["f2a_temperature_registry"]
    assert f2a["arm_promoted"] is False
    assert f2a["matched_oracle"] == "F3_temperature_only_target_day_Daymet_TEMP"
    assert f2a["forbidden_denominator"] == "F3_full"
    assert f2a["information_semantics"] == {
        "fixed_lead_composite": True,
        "one_target_day_value_per_registered_lead": True,
        "intermediate_trajectory_constructed": False,
        "prefix_mean_constructed": False,
        "coherent_single_initialization": False,
        "as_issued_operational_forecast_archive": False,
        "operational_wording_allowed": False,
    }
    assert not any("f3_full" in column.lower() for column in bundle.f2a_common.columns)
    assert bundle.f2a_common["f2a_complete_target_day"].all()
    assert bundle.f2a_common["f2a_available_hour_count"].eq(24).all()


def test_fifteen_digit_site_survives_every_registry_layer(
    synthetic: SyntheticFixture,
) -> None:
    bundle = _build(synthetic)
    long_site = SITES[2]
    assert len(long_site) == 15 and long_site.isdecimal()
    assert long_site in set(bundle.primary_raw["site_id"])
    assert long_site in set(bundle.primary_station_inventory["site_id"])
    assert long_site in set(bundle.f2a_common["site_id"])
    assert long_site in set(bundle.f2a_station_inventory["site_id"])
    assert long_site in bundle.manifest["primary_registry"]["raw_site_ids"]


@pytest.mark.parametrize(
    "invalid",
    [
        "1234567",
        "123456789",
        "12345678901234",
        "1234567890123456",
        "abcdefgh",
        "1234-678",
        " 00000001",
        "00000001 ",
        "",
    ],
)
def test_site_identity_rejects_every_non_8_or_15_digit_shape(invalid: str) -> None:
    with pytest.raises(registry.KeyRegistryError, match="8 or 15|whitespace"):
        registry._normalise_sites(pd.Series([invalid], dtype="string"), label="synthetic site")


def test_site_identity_accepts_exact_8_and_15_digit_shapes() -> None:
    observed = registry._normalise_sites(
        pd.Series(["00000001", "123456789012345"], dtype="string"),
        label="synthetic site",
    )
    assert observed.tolist() == ["00000001", "123456789012345"]


def test_asof_recency_never_uses_a_global_future_last_observation() -> None:
    dates = pd.date_range("2021-01-01", periods=40, freq="D")
    values = np.full(len(dates), np.nan)
    values[0] = 1.0
    values[-1] = 2.0
    panel = pd.DataFrame({"site_id": SITES[0], "DATE": dates, "WTEMP": values})
    context = registry._build_context_table(panel)
    middle = context.loc[context["DATE"].eq(pd.Timestamp("2021-01-20"))].iloc[0]
    assert middle["issue_wtemp_observed"] is np.False_ or not middle["issue_wtemp_observed"]
    assert middle["days_since_last_observed_wtemp"] == 19
    assert context["days_since_last_observed_wtemp"].min() >= 0
    final = context.iloc[-1]
    assert final["days_since_last_observed_wtemp"] == 0


@pytest.mark.parametrize(
    "drift_date",
    [pd.Timestamp("2021-03-23"), pd.Timestamp("2021-04-08")],
    ids=("issue-observability-drift", "target-observability-drift"),
)
def test_issue_or_target_observability_drift_fails_closed(
    synthetic: SyntheticFixture, drift_date: pd.Timestamp
) -> None:
    panel = pd.read_parquet(synthetic.panel)
    panel.loc[panel["site_id"].eq(SITES[0]) & panel["DATE"].eq(drift_date), "WTEMP"] = np.nan
    panel.to_parquet(synthetic.panel, index=False)
    with pytest.raises(registry.KeyRegistryError, match="identities differ"):
        _build(synthetic)


def test_y_reconstruction_beyond_absolute_tolerance_fails_closed(
    synthetic: SyntheticFixture,
) -> None:
    legacy = pd.read_parquet(synthetic.legacy)
    legacy.loc[0, "y_true"] += registry.Y_ABS_TOLERANCE_C * 2
    legacy.to_parquet(synthetic.legacy, index=False)
    with pytest.raises(registry.KeyRegistryError, match="exceeds absolute"):
        _build(synthetic)


def test_reportability_rejects_score_columns() -> None:
    keys = pd.DataFrame(
        {
            "site_id": [SITES[0]] * 3,
            "lead_days": list(registry.LEADS),
            "issue_date": [pd.Timestamp("2021-01-01")] * 3,
            "target_date": [
                pd.Timestamp("2021-01-02"),
                pd.Timestamp("2021-01-04"),
                pd.Timestamp("2021-01-08"),
            ],
            "model_score": [0.1, 0.2, 0.3],
        }
    )
    config = registry.RegistryBuildConfig(
        mode="synthetic",
        primary_issue_start=date(2021, 1, 1),
        primary_target_end=date(2021, 3, 31),
        f2a_target_start=date(2021, 3, 30),
        minimum_targets=1,
        expected_registry_stations=None,
        expected_raw_stations=None,
        expected_reportable_stations=None,
        expected_excluded_sites=None,
    )
    with pytest.raises(registry.KeyRegistryError, match="score-like"):
        registry._freeze_primary_reportability(keys, config=config)


def test_fixed_lead_identity_mismatch_fails_even_after_resealing(
    synthetic: SyntheticFixture,
) -> None:
    def mutate(frame: pd.DataFrame) -> pd.DataFrame:
        frame.loc[0, "lead_field"] = "temperature_2m_previous_day3"
        return frame

    _mutate_block_and_reseal(synthetic, mutate)
    with pytest.raises(registry.KeyRegistryError, match="fixed-lead field identity"):
        _build(synthetic)


def test_target_chronology_mismatch_fails_even_after_resealing(
    synthetic: SyntheticFixture,
) -> None:
    def mutate(frame: pd.DataFrame) -> pd.DataFrame:
        frame.loc[0, "issue_date"] = pd.Timestamp(frame.loc[0, "issue_date"]) - pd.Timedelta(days=1)
        return frame

    _mutate_block_and_reseal(synthetic, mutate)
    with pytest.raises(registry.KeyRegistryError, match="target/issue chronology"):
        _build(synthetic)


@pytest.mark.parametrize("mutation", ("duplicate", "outside"))
def test_duplicate_or_outside_f2a_keys_fail_even_after_resealing(
    synthetic: SyntheticFixture, mutation: str
) -> None:
    def mutate(frame: pd.DataFrame) -> pd.DataFrame:
        if mutation == "duplicate":
            return pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        frame.loc[0, "target_date"] = pd.Timestamp(TARGET_END) + pd.Timedelta(days=1)
        frame.loc[0, "issue_date"] = frame.loc[0, "target_date"] - pd.Timedelta(
            days=int(frame.loc[0, "horizon"])
        )
        return frame

    _mutate_block_and_reseal(synthetic, mutate)
    message = (
        "incomplete or outside target-day inventory"
        if mutation == "duplicate"
        else "outside target date"
    )
    with pytest.raises(registry.KeyRegistryError, match=message):
        _build(synthetic)


def test_incomplete_target_day_is_never_admitted_even_when_coverage_is_exactly_090(
    synthetic: SyntheticFixture,
) -> None:
    def mutate(frame: pd.DataFrame) -> pd.DataFrame:
        first = frame.index[0]
        frame.loc[first, "available_hour_count"] = 23
        frame.loc[first, "complete_target_day"] = False
        frame.loc[first, "air_temp_2m_mean_c"] = np.nan
        return frame

    artifact = _mutate_block_and_reseal(synthetic, mutate)
    changed = pd.read_parquet(artifact).iloc[0]
    report = json.loads(synthetic.verification_report.read_text(encoding="utf-8"))
    assert min(row["coverage_fraction"] for row in report["coverage_by_station_and_lead"]) == 0.9
    bundle = _build(synthetic)
    identity = (
        str(changed.site_no),
        int(changed.horizon),
        pd.Timestamp(changed.issue_date),
        pd.Timestamp(changed.target_date),
    )
    output_identities = set(
        map(
            tuple,
            bundle.f2a_common[["site_id", "lead_days", "issue_date", "target_date"]].itertuples(
                index=False, name=None
            ),
        )
    )
    assert identity not in output_identities


def test_parquet_hash_tamper_fails_closed(synthetic: SyntheticFixture) -> None:
    artifact = min(synthetic.acquisition_root.glob("*/*.parquet"))
    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    with pytest.raises(registry.KeyRegistryError, match="checksum mismatch"):
        _build(synthetic)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fixed_lead_composite", False),
        ("coherent_single_initialization", True),
        ("as_issued_operational_forecast_archive", True),
    ],
)
def test_wrong_f2a_semantics_fail_closed(
    synthetic: SyntheticFixture, field: str, value: bool
) -> None:
    report = json.loads(synthetic.verification_report.read_text(encoding="utf-8"))
    report["information_semantics"][field] = value
    _canonical_write(synthetic.verification_report, report)
    with pytest.raises(registry.KeyRegistryError, match="semantics changed"):
        _build(synthetic)


def test_f3_full_denominator_column_leakage_fails_even_after_resealing(
    synthetic: SyntheticFixture,
) -> None:
    def mutate(frame: pd.DataFrame) -> pd.DataFrame:
        frame["F3_full_temperature"] = frame["air_temp_2m_mean_c"]
        return frame

    _mutate_block_and_reseal(synthetic, mutate)
    with pytest.raises(registry.KeyRegistryError, match="schema changed"):
        _build(synthetic)


def test_manifest_path_escape_fails_closed(synthetic: SyntheticFixture) -> None:
    manifest = json.loads(synthetic.acquisition_manifest.read_text(encoding="utf-8"))
    manifest["chunks"][0]["artifact"] = "../outside.parquet"
    _canonical_write(synthetic.acquisition_manifest, manifest)
    report = json.loads(synthetic.verification_report.read_text(encoding="utf-8"))
    report["evidence_hashes"]["acquisition_manifest"]["sha256"] = _sha256(
        synthetic.acquisition_manifest
    )
    _canonical_write(synthetic.verification_report, report)
    with pytest.raises(registry.KeyRegistryError, match="escapes root"):
        _build(synthetic)


def test_symlink_input_is_rejected(synthetic: SyntheticFixture) -> None:
    real_panel = synthetic.panel.with_name("real-panel.parquet")
    synthetic.panel.rename(real_panel)
    synthetic.panel.symlink_to(real_panel.name)
    with pytest.raises(registry.KeyRegistryError, match="symlink"):
        _build(synthetic)


def test_publish_is_atomic_create_only_and_has_no_resume_path(
    synthetic: SyntheticFixture,
) -> None:
    destination = synthetic.root / "key-registry-v4"
    published, bundle = registry.publish_registry_artifacts(
        synthetic.paths,
        destination,
        config=synthetic.config,
        repository_root=synthetic.root,
        input_root=synthetic.root,
        allowed_output_root=synthetic.root,
    )
    assert published == destination
    assert {path.name for path in destination.iterdir()} == set(bundle.files)
    with pytest.raises(registry.KeyRegistryError, match="overwrite"):
        registry.publish_registry_artifacts(
            synthetic.paths,
            destination,
            config=synthetic.config,
            repository_root=synthetic.root,
            input_root=synthetic.root,
            allowed_output_root=synthetic.root,
        )
    with pytest.raises(registry.KeyRegistryError, match="partial/resume"):
        registry.publish_registry_artifacts(
            synthetic.paths,
            synthetic.root / "partial-key-registry",
            config=synthetic.config,
            repository_root=synthetic.root,
            input_root=synthetic.root,
            allowed_output_root=synthetic.root,
        )
    with pytest.raises(SystemExit):
        registry.main(["--resume"])


def test_deterministic_rebuild_is_byte_identical(synthetic: SyntheticFixture) -> None:
    first = _build(synthetic)
    second = _build(synthetic)
    assert first.files.keys() == second.files.keys()
    assert first.files == second.files
    assert first.manifest == second.manifest


def test_payload_mapping_is_deeply_immutable(synthetic: SyntheticFixture) -> None:
    bundle = _build(synthetic)
    with pytest.raises(TypeError):
        bundle.files["injected.bin"] = b"forged"  # type: ignore[index]
    assert all(type(payload) is bytes for payload in bundle.files.values())
    changed = bundle.manifest
    changed["status"] = "FORGED"
    assert bundle.manifest["status"] != "FORGED"


def test_injected_or_replaced_payload_fails_publication_integrity(
    synthetic: SyntheticFixture,
) -> None:
    bundle = _build(synthetic)
    injected_files = dict(bundle.files)
    injected_files["injected.bin"] = b"forged"
    injected = replace(bundle, files=injected_files)
    with pytest.raises(registry.KeyRegistryError, match="file set is not exact"):
        registry._verify_bundle_integrity(injected)

    replaced_files = dict(bundle.files)
    replaced_files[registry.PRIMARY_RAW_FILENAME] = b"not parquet"
    replaced_payload = replace(bundle, files=replaced_files)
    with pytest.raises(registry.KeyRegistryError, match="SHA-256 mismatch"):
        registry._verify_bundle_integrity(replaced_payload)


def test_resealed_forged_manifest_cannot_replace_independent_reconstruction(
    synthetic: SyntheticFixture,
) -> None:
    reconstruction = _build(synthetic)
    forged_manifest = reconstruction.manifest
    forged_manifest["input_bindings"]["builder_code"] = {
        "path": "scripts/final/forged_builder.py",
        "sha256": "f" * 64,
        "size_bytes": 1,
    }
    forged_manifest.pop("deterministic_reconstruction_receipt")
    forged_manifest["deterministic_reconstruction_receipt"] = (
        registry._deterministic_reconstruction_receipt(forged_manifest)
    )
    forged_manifest_payload = registry._canonical_json_bytes(forged_manifest)
    forged_files = dict(reconstruction.files)
    forged_files[registry.MANIFEST_FILENAME] = forged_manifest_payload
    forged = replace(
        reconstruction,
        files=forged_files,
        manifest_payload=forged_manifest_payload,
    )
    # A self-consistent, resealed receipt is not authority.  Only comparison
    # to a fresh build from canonical captured inputs closes this bypass.
    registry._verify_bundle_integrity(forged)
    with pytest.raises(registry.KeyRegistryError, match="independent reconstruction"):
        registry._verify_bundle_against_independent_reconstruction(forged, reconstruction)


def test_direct_or_dataclasses_replaced_bundle_is_not_accepted_by_closed_publisher(
    synthetic: SyntheticFixture,
) -> None:
    bundle = _build(synthetic)
    direct = registry.RegistryArtifactBundle(
        files=dict(bundle.files),
        manifest_payload=bundle.manifest_payload,
        primary_raw=bundle.primary_raw.copy(),
        primary_reportable=bundle.primary_reportable.copy(),
        primary_station_inventory=bundle.primary_station_inventory.copy(),
        f2a_common=bundle.f2a_common.copy(),
        f2a_reportable=bundle.f2a_reportable.copy(),
        f2a_station_inventory=bundle.f2a_station_inventory.copy(),
    )
    replaced_bundle = replace(bundle, files=dict(bundle.files))
    for candidate in (direct, replaced_bundle):
        with pytest.raises(registry.KeyRegistryError, match="never a caller bundle"):
            registry.publish_registry_artifacts(
                candidate,  # type: ignore[arg-type]
                synthetic.root / "must-not-publish",
                config=synthetic.config,
                repository_root=synthetic.root,
                input_root=synthetic.root,
                allowed_output_root=synthetic.root,
            )
    assert not (synthetic.root / "must-not-publish").exists()


def test_runtime_receipt_discloses_lock_mismatch_and_no_cross_environment_claim(
    synthetic: SyntheticFixture,
) -> None:
    bundle = _build(synthetic)
    runtime = bundle.manifest["runtime_determinism"]
    assert runtime["status"] == registry.RUNTIME_STATUS
    assert runtime["determinism_class"] == "PINNED_RUNTIME_DETERMINISTIC"
    assert runtime["repository_locks_are_active_runtime_authority"] is False
    assert runtime["repository_locks_match_active_runtime"] is False
    assert "not claimed" in runtime["byte_identity_scope"]
    assert runtime["active_runtime"] == registry._current_runtime_identity()
    assert runtime["active_runtime"]["python_compiler"] == "GCC 11.2.0"
    assert runtime["production_exact_runtime_pin"]["python_compiler"] == "GCC 11.2.0"
    registry._verify_bundle_integrity(bundle)
    references = runtime["repository_dependency_references"]
    assert references["requirements_lock"]["sha256"] == _sha256(
        synthetic.root / "requirements-lock.txt"
    )
    assert references["requirements_lock_py312_hashed"]["sha256"] == _sha256(
        synthetic.root / "requirements-lock-py312-hashed.txt"
    )
    assert references["pyproject"]["sha256"] == _sha256(synthetic.root / "pyproject.toml")


def test_current_audited_production_runtime_pin_is_exact() -> None:
    assert dict(registry.PRODUCTION_RUNTIME) == {
        "python_implementation": "CPython",
        "python_version": "3.11.7",
        "python_compiler": "GCC 11.2.0",
        "numpy_version": "1.26.4",
        "pandas_version": "2.1.4",
        "pyarrow_python_version": "14.0.2",
        "arrow_cpp_version": "14.0.2",
    }
    active = registry._current_runtime_identity()
    assert {key: active[key] for key in registry.PRODUCTION_RUNTIME} == dict(
        registry.PRODUCTION_RUNTIME
    )
    assert active["platform_system"] == "Linux"
    assert active["platform_machine"] == "x86_64"
    assert active["byteorder"] == "little"


def test_production_runtime_gate_fails_closed_on_forged_python_compiler(
    synthetic: SyntheticFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def captured(relative: str) -> registry.BoundBytes:
        return registry._read_stable_regular(
            synthetic.root / relative,
            root=synthetic.root,
            label=relative,
        )

    monkeypatch.setattr(registry.platform, "python_compiler", lambda: "GCC 11.2.0 FORGED")
    with pytest.raises(
        registry.KeyRegistryError,
        match="production Parquet runtime differs from the exact current-runtime pin",
    ):
        registry._runtime_evidence(
            requirements_lock=captured("requirements-lock.txt"),
            requirements_lock_py312=captured("requirements-lock-py312-hashed.txt"),
            pyproject=captured("pyproject.toml"),
            repository_root=synthetic.root,
            production=True,
        )


def test_canonical_production_paths_and_hashes_are_purely_validated_without_reads() -> None:
    assert dict(registry.PRODUCTION_CANONICAL_FILES) == {
        "protocol_v4_draft": (
            "protocols/wrr_information_regimes_protocol_v4.yaml",
            "66e089baf37db1137cad23f148e31df39cc71aea872dec4a97dc6f8701d13a98",
        ),
        "route_a_protocol": (
            "protocols/route_a_confirmatory_v1.json",
            "93c32e9dbfe976eaa7ef31cc5181ae1f4a415ad2b2e30674a5df64c46b968c05",
        ),
        "station_registry_v1": (
            "data_usgs/station_registry_v1.csv",
            "090e7c0daf39ac38ceefeb1af8a12c178283e18347905d8e72ada969ad5460c9",
        ),
        "raw_holdout_panel": (
            "outputs/conventional/panel_2021_2023.parquet",
            "cecdac459139456202240954e4c98fe18bba1fe8b63e9b06ab268683e0d1c03c",
        ),
        "authority_bound_legacy_forecast_registry": (
            "outputs/final/forecast_keys.parquet",
            "56e00e8befe256c9c90a7245cdc892e8d7539982c16fadeadd34d404079a1e0f",
        ),
        "independent_f2a_verification_report": (
            "outputs/final/f2a_acquisition_verification_v1.json",
            "7d8a019d4966a941e9396a40930b2775ba95660347684701838721fe5032a60e",
        ),
        "f2a_acquisition_manifest": (
            "data_usgs/confirmatory_predictors/gfs-previous-runs-v1/manifest.json",
            "54f85eef3ccd4b3cc69071f3c4ed7af5e3208274d3b9062a49c43c96ea43362c",
        ),
        "f2a_snapshot_index": (
            "data_usgs/raw_snapshots/openmeteo-gfs-previous-runs-v1/snapshot_index.json",
            "433f4b4885f225f5f9fe87ec9e94ba81493c07691bb4cc9a994be1931a406771",
        ),
        "requirements_lock": (
            "requirements-lock.txt",
            "ff2d67915ccaabb750cdf4c630d59500d6c8841b305d5fffb1ffd549195dc047",
        ),
        "requirements_lock_py312_hashed": (
            "requirements-lock-py312-hashed.txt",
            "fa325e30e8e69b9e8ec458e9e76a466a765530a8bffa46f1d1f8e34614f4ab76",
        ),
        "pyproject": (
            "pyproject.toml",
            "ef49ffacf3c73a2e566b89abac22d663bdd997a3f68adc5dcf0f5f187860fcc6",
        ),
    }
    observed = {
        role: {"path": path, "sha256": sha256}
        for role, (path, sha256) in registry.PRODUCTION_CANONICAL_FILES.items()
    }
    registry._validate_canonical_path_hash_contract(
        observed,
        expected_files=registry.PRODUCTION_CANONICAL_FILES,
        observed_acquisition_root=registry.PRODUCTION_ACQUISITION_ROOT,
        expected_acquisition_root=registry.PRODUCTION_ACQUISITION_ROOT,
        observed_builder_path=registry.PRODUCTION_BUILDER_PATH,
        expected_builder_path=registry.PRODUCTION_BUILDER_PATH,
    )


@pytest.mark.parametrize("tamper", ("path", "sha256", "root", "builder", "role"))
def test_any_canonical_production_binding_drift_fails_without_production_reads(
    tamper: str,
) -> None:
    observed = {
        role: {"path": path, "sha256": sha256}
        for role, (path, sha256) in registry.PRODUCTION_CANONICAL_FILES.items()
    }
    root = registry.PRODUCTION_ACQUISITION_ROOT
    builder = registry.PRODUCTION_BUILDER_PATH
    first_role = next(iter(observed))
    if tamper == "path":
        observed[first_role]["path"] = "elsewhere"
    elif tamper == "sha256":
        observed[first_role]["sha256"] = "0" * 64
    elif tamper == "root":
        root = "data_usgs/another-root"
    elif tamper == "builder":
        builder = "scripts/final/another_builder.py"
    else:
        observed["unexpected"] = {"path": "x", "sha256": "0" * 64}
    with pytest.raises(registry.KeyRegistryError, match="canonical production"):
        registry._validate_canonical_path_hash_contract(
            observed,
            expected_files=registry.PRODUCTION_CANONICAL_FILES,
            observed_acquisition_root=root,
            expected_acquisition_root=registry.PRODUCTION_ACQUISITION_ROOT,
            observed_builder_path=builder,
            expected_builder_path=registry.PRODUCTION_BUILDER_PATH,
        )


def test_wrong_production_path_is_rejected_before_any_file_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = registry.ROOT
    paths = registry.RegistryInputPaths(
        protocol_v4=root / "protocols" / "wrong-v4.yaml",
        route_a_protocol=root / registry.PRODUCTION_CANONICAL_FILES["route_a_protocol"][0],
        station_registry=root / registry.PRODUCTION_CANONICAL_FILES["station_registry_v1"][0],
        raw_holdout_panel=root / registry.PRODUCTION_CANONICAL_FILES["raw_holdout_panel"][0],
        legacy_forecast_registry=root
        / registry.PRODUCTION_CANONICAL_FILES["authority_bound_legacy_forecast_registry"][0],
        f2a_verification_report=root
        / registry.PRODUCTION_CANONICAL_FILES["independent_f2a_verification_report"][0],
        f2a_acquisition_manifest=root
        / registry.PRODUCTION_CANONICAL_FILES["f2a_acquisition_manifest"][0],
        f2a_acquisition_root=root / registry.PRODUCTION_ACQUISITION_ROOT,
    )

    def forbidden_read(*args: object, **kwargs: object) -> registry.BoundBytes:
        raise AssertionError("production bytes were opened before path pinning")

    monkeypatch.setattr(registry, "_read_stable_regular", forbidden_read)
    with pytest.raises(registry.KeyRegistryError, match="canonical production path"):
        registry.build_registry_artifacts(
            paths,
            config=registry.RegistryBuildConfig.production(),
            repository_root=root,
            input_root=root,
            production_read_authorized=True,
        )


def test_audited_production_scientific_rows_and_parquet_hashes_are_pinned() -> None:
    assert dict(registry.PRODUCTION_EXPECTED_ROWS) == {
        registry.PRIMARY_RAW_FILENAME: 358807,
        registry.PRIMARY_REPORTABLE_FILENAME: 358765,
        registry.F2A_COMMON_FILENAME: 329648,
        registry.F2A_REPORTABLE_FILENAME: 329628,
    }
    assert dict(registry.PRODUCTION_EXPECTED_PARQUET_SHA256) == {
        registry.PRIMARY_RAW_FILENAME: (
            "7d6c5cfa2ae2905fb725e56f8187df809ade200918fbe5c60291ed7861058268"
        ),
        registry.PRIMARY_REPORTABLE_FILENAME: (
            "9a135dcffcfd467cf1e2dda4fc711ba6cf66c8ad54bf3b6f799f4a2d5a3c1d24"
        ),
        registry.PRIMARY_STATIONS_FILENAME: (
            "d06e9bd51e15d704bce7c96f678bfd2d08b656483345ce9988e50cda438d0783"
        ),
        registry.F2A_COMMON_FILENAME: (
            "1861b9cc81551bbcfc408b83f9c7184de730017982a40e03bcfc65ef16ab27a7"
        ),
        registry.F2A_REPORTABLE_FILENAME: (
            "5cc75fda17abe44cd5201184d156c5a53726960d78182b726edf0620895754b6"
        ),
        registry.F2A_STATIONS_FILENAME: (
            "0ef0cc3bebe70179e691250689d28b1bb077393b95255cc341691a8089692ee4"
        ),
    }


def test_synthetic_mode_cannot_access_production_repository_root(
    synthetic: SyntheticFixture,
) -> None:
    with pytest.raises(registry.KeyRegistryError, match="synthetic mode"):
        registry.build_registry_artifacts(
            synthetic.paths,
            config=synthetic.config,
            repository_root=registry.ROOT,
            input_root=registry.ROOT,
        )


def test_noncanonical_verification_report_fails_closed(
    synthetic: SyntheticFixture,
) -> None:
    report = json.loads(synthetic.verification_report.read_text(encoding="utf-8"))
    synthetic.verification_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(registry.KeyRegistryError, match="not canonical JSON"):
        _build(synthetic)

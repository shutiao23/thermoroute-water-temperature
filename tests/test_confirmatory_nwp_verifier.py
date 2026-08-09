from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from scripts.data_usgs import verify_confirmatory_nwp as verifier

TARGET_START = date(2021, 3, 30)
TARGET_END = date(2021, 3, 31)
SITE_NO = "01234567"
REQUESTED_LAT = 35.25
REQUESTED_LON = -119.75
GRID_LAT = 35.2
GRID_LON = -119.8
RETRIEVED = "2026-08-09T12:34:56.123456+00:00"


@dataclass
class SyntheticAcquisition:
    root: Path
    output_dir: Path
    snapshot_dir: Path
    registry: Path
    protocol: Path
    artifact: Path
    sidecar: Path
    response: Path
    metadata: Path
    snapshot_index: Path
    manifest: Path
    request_sha: str


def _write_canonical(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(verifier._canonical_json_bytes(value))


def _protocol_document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "protocol_id": verifier.EXPECTED_PROTOCOL_ID,
        "time_holdout": {
            "secondary_nwp_common_lead_target_start": TARGET_START.isoformat(),
            "end": TARGET_END.isoformat(),
        },
        "secondary_archived_nwp_contract": {
            "status": "IMPLEMENTED_NOT_ACQUIRED",
            "role": "OPTIONAL_SECONDARY_AVAILABILITY_SENSITIVITY_NOT_PRIMARY_INPUT",
            "contains_outcome_labels": False,
            "consumed_by_primary_models": False,
            "primary_evaluation_dependency": False,
            "provider": verifier.EXPECTED_PROVIDER,
            "provider_documentation": verifier.EXPECTED_DOCUMENTATION,
            "api_endpoint": verifier.EXPECTED_ENDPOINT,
            "upstream_model": verifier.EXPECTED_UPSTREAM_MODEL,
            "open_meteo_model_parameter": verifier.EXPECTED_OPEN_METEO_MODEL,
            "variable": verifier.EXPECTED_SOURCE_VARIABLE,
            "unit": verifier.EXPECTED_UNIT,
            "lead_days": list(verifier.EXPECTED_LEADS),
            "lead_fields": [
                f"temperature_2m_previous_day{lead}" for lead in verifier.EXPECTED_LEADS
            ],
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
            "archive_run_start": verifier.EXPECTED_ARCHIVE_START.isoformat(),
            "secondary_common_lead_target_start": TARGET_START.isoformat(),
            "request_partition": (
                "one stable site_no and one UTC calendar month per immutable raw snapshot"
            ),
            "raw_snapshot_required": True,
            "derived_blocks_may_be_overwritten": False,
            "selection_or_tuning_from_predictor_availability": False,
        },
    }


def _raw_document(*, missing_first_hour_for_lead7: bool) -> dict[str, object]:
    times = pd.date_range(
        pd.Timestamp(TARGET_START), pd.Timestamp(TARGET_END) + pd.Timedelta(hours=23), freq="h"
    )
    hourly: dict[str, object] = {"time": times.strftime("%Y-%m-%dT%H:%M").tolist()}
    units: dict[str, str] = {"time": "iso8601"}
    for lead in verifier.EXPECTED_LEADS:
        values: list[float | None] = [float(lead * 10 + hour % 24) for hour in range(len(times))]
        if lead == 7 and missing_first_hour_for_lead7:
            values[0] = None
        field = f"temperature_2m_previous_day{lead}"
        hourly[field] = values
        units[field] = "°C"
    return {
        "latitude": GRID_LAT,
        "longitude": GRID_LON,
        "generationtime_ms": 0.1,
        "utc_offset_seconds": 0,
        "timezone": "GMT",
        "timezone_abbreviation": "GMT",
        "elevation": 100.0,
        "hourly_units": units,
        "hourly": hourly,
    }


def _derived_frame(raw: dict[str, object], request_sha: str, response_sha: str) -> pd.DataFrame:
    hourly = raw["hourly"]
    assert isinstance(hourly, dict)
    records: list[dict[str, object]] = []
    for lead in verifier.EXPECTED_LEADS:
        field = f"temperature_2m_previous_day{lead}"
        values = hourly[field]
        assert isinstance(values, list)
        for day_offset in range(2):
            target = TARGET_START + timedelta(days=day_offset)
            block = values[day_offset * 24 : (day_offset + 1) * 24]
            finite = [float(value) for value in block if value is not None]
            complete = len(finite) == 24
            records.append(
                {
                    "site_no": SITE_NO,
                    "requested_lat": REQUESTED_LAT,
                    "requested_lon": REQUESTED_LON,
                    "horizon": lead,
                    "issue_date": pd.Timestamp(target - timedelta(days=lead)),
                    "target_date": pd.Timestamp(target),
                    "air_temp_2m_mean_c": sum(finite) / 24 if complete else float("nan"),
                    "available_hour_count": len(finite),
                    "complete_target_day": complete,
                    "lead_field": field,
                    "source_provider": verifier.EXPECTED_PROVIDER,
                    "upstream_model": verifier.EXPECTED_UPSTREAM_MODEL,
                    "open_meteo_model": verifier.EXPECTED_OPEN_METEO_MODEL,
                    "issue_semantics": verifier.EXPECTED_ISSUE_SEMANTICS,
                    "response_grid_lat": GRID_LAT,
                    "response_grid_lon": GRID_LON,
                    "request_sha256": request_sha,
                    "response_sha256": response_sha,
                }
            )
    return pd.DataFrame.from_records(records, columns=verifier.PARQUET_COLUMNS)


def _build_synthetic(
    root: Path, *, missing_first_hour_for_lead7: bool = False
) -> SyntheticAcquisition:
    output_dir = root / "predictors"
    snapshot_dir = root / "snapshots"
    registry = root / "station_registry.csv"
    protocol = root / "route_a_protocol.json"
    registry.write_text(
        f"site_no,lat,lon,metadata_note\n{SITE_NO},{REQUESTED_LAT},{REQUESTED_LON},synthetic\n",
        encoding="utf-8",
    )
    protocol.write_text(json.dumps(_protocol_document(), indent=2) + "\n", encoding="utf-8")
    protocol_sha = verifier._sha256_file(protocol, label="synthetic protocol")

    station = verifier.Station(SITE_NO, REQUESTED_LAT, REQUESTED_LON)
    url = verifier._request_url(station, TARGET_START, TARGET_END)
    request = verifier._request_document(url)
    request_sha = verifier._sha256_bytes(verifier._canonical_json_bytes(request))
    raw = _raw_document(missing_first_hour_for_lead7=missing_first_hour_for_lead7)
    raw_payload = (json.dumps(raw, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    response_sha = verifier._sha256_bytes(raw_payload)

    artifact = output_dir / SITE_NO / "2021-03.parquet"
    sidecar = artifact.with_suffix(".provenance.json")
    response = snapshot_dir / verifier.EXPECTED_PROVIDER_SLUG / request_sha / "response.bin"
    metadata = response.with_name("metadata.json")
    response.parent.mkdir(parents=True)
    response.write_bytes(raw_payload)
    snapshot_metadata = {
        "schema_version": 1,
        "request": request,
        "request_sha256": request_sha,
        "retrieved_at_utc": RETRIEVED,
        "http_status": 200,
        "response_headers": {"Content-Type": "application/json"},
        "byte_count": len(raw_payload),
        "response_sha256": response_sha,
        "response_file": "response.bin",
    }
    _write_canonical(metadata, snapshot_metadata)

    frame = _derived_frame(raw, request_sha, response_sha)
    artifact.parent.mkdir(parents=True)
    frame.to_parquet(artifact, index=False)
    sidecar_document = {
        "schema_version": 1,
        "artifact": verifier._rel_repo(artifact),
        "artifact_sha256": verifier._sha256_file(artifact, label="synthetic Parquet"),
        "site_no": SITE_NO,
        "chunk_start": TARGET_START.isoformat(),
        "chunk_end": TARGET_END.isoformat(),
        "row_count": len(frame),
        "complete_row_count": int(frame["complete_target_day"].sum()),
        "request_sha256": request_sha,
        "response_sha256": response_sha,
        "retrieved_at_utc": RETRIEVED,
        "labels_requested_or_read": False,
        "protocol_sha256": protocol_sha,
    }
    _write_canonical(sidecar, sidecar_document)

    snapshot_index = snapshot_dir / "snapshot_index.json"
    index_document = {
        "schema_version": 1,
        "snapshot_count": 1,
        "records": [
            {
                "provider": verifier.EXPECTED_PROVIDER_SLUG,
                "request_sha256": request_sha,
                "response_sha256": response_sha,
                "retrieved_at_utc": RETRIEVED,
                "byte_count": len(raw_payload),
                "request": request,
                "metadata_path": str(metadata.relative_to(snapshot_dir)),
                "response_path": str(response.relative_to(snapshot_dir)),
            }
        ],
    }
    _write_canonical(snapshot_index, index_document)

    manifest = output_dir / "manifest.json"
    registry_lineage = [
        {
            "path": verifier._rel_repo(registry),
            "sha256": verifier._sha256_file(registry, label="synthetic registry"),
            "columns_read": ["site_no", "lat", "lon"],
            "row_count": 1,
        }
    ]
    manifest_document = {
        "schema_version": 1,
        "artifact_role": "OPTIONAL_SECONDARY_NWP_AVAILABILITY_SENSITIVITY",
        "protocol": verifier._rel_repo(protocol),
        "protocol_sha256": protocol_sha,
        "source_provider": verifier.EXPECTED_PROVIDER,
        "source_documentation": verifier.EXPECTED_DOCUMENTATION,
        "upstream_model": verifier.EXPECTED_UPSTREAM_MODEL,
        "open_meteo_model": verifier.EXPECTED_OPEN_METEO_MODEL,
        "source_variable": verifier.EXPECTED_SOURCE_VARIABLE,
        "lead_days": list(verifier.EXPECTED_LEADS),
        "lead_fields": [f"temperature_2m_previous_day{lead}" for lead in verifier.EXPECTED_LEADS],
        "issue_semantics": verifier.EXPECTED_ISSUE_SEMANTICS,
        "timezone": "GMT (UTC+00:00)",
        "daily_aggregation": (
            "arithmetic mean of exactly 24 finite valid-hour values; incomplete "
            "days retained with NaN predictor and availability count"
        ),
        "archive_run_start": verifier.EXPECTED_ARCHIVE_START.isoformat(),
        "common_valid_target_start": TARGET_START.isoformat(),
        "target_end": TARGET_END.isoformat(),
        "station_count": 1,
        "row_count": len(frame),
        "complete_row_count": int(frame["complete_target_day"].sum()),
        "request_partition": "one stable site_no by one UTC calendar month",
        "immutable_outputs": True,
        "consumed_by_primary_route_a_models": False,
        "primary_evaluation_dependency": False,
        "labels_requested_or_read": False,
        "outcome_endpoint_called": False,
        "registry_inputs": registry_lineage,
        "raw_snapshot_index": verifier._rel_repo(snapshot_index),
        "raw_snapshot_index_sha256": verifier._sha256_file(
            snapshot_index, label="synthetic snapshot index"
        ),
        "chunks": [sidecar_document],
    }
    _write_canonical(manifest, manifest_document)
    return SyntheticAcquisition(
        root=root,
        output_dir=output_dir,
        snapshot_dir=snapshot_dir,
        registry=registry,
        protocol=protocol,
        artifact=artifact,
        sidecar=sidecar,
        response=response,
        metadata=metadata,
        snapshot_index=snapshot_index,
        manifest=manifest,
        request_sha=request_sha,
    )


def _verify(fixture: SyntheticAcquisition) -> dict[str, object]:
    return verifier.verify_acquisition(
        output_dir=fixture.output_dir,
        snapshot_dir=fixture.snapshot_dir,
        registry_paths=[fixture.registry],
        protocol_path=fixture.protocol,
        _expected_target_end=TARGET_END,
        _required_station_count=1,
    )


def _reseal_after_artifact_change(fixture: SyntheticAcquisition) -> None:
    frame = pd.read_parquet(fixture.artifact)
    sidecar = json.loads(fixture.sidecar.read_text(encoding="utf-8"))
    sidecar["artifact_sha256"] = verifier._sha256_file(
        fixture.artifact, label="mutated synthetic Parquet"
    )
    sidecar["row_count"] = len(frame)
    sidecar["complete_row_count"] = int(frame["complete_target_day"].sum())
    _write_canonical(fixture.sidecar, sidecar)
    manifest = json.loads(fixture.manifest.read_text(encoding="utf-8"))
    manifest["row_count"] = len(frame)
    manifest["complete_row_count"] = int(frame["complete_target_day"].sum())
    manifest["chunks"] = [sidecar]
    _write_canonical(fixture.manifest, manifest)


def _reseal_after_response_change(fixture: SyntheticAcquisition) -> None:
    payload = fixture.response.read_bytes()
    response_sha = verifier._sha256_bytes(payload)
    metadata = json.loads(fixture.metadata.read_text(encoding="utf-8"))
    metadata["response_sha256"] = response_sha
    metadata["byte_count"] = len(payload)
    _write_canonical(fixture.metadata, metadata)
    frame = pd.read_parquet(fixture.artifact)
    frame["response_sha256"] = response_sha
    frame.to_parquet(fixture.artifact, index=False)
    sidecar = json.loads(fixture.sidecar.read_text(encoding="utf-8"))
    sidecar["response_sha256"] = response_sha
    sidecar["artifact_sha256"] = verifier._sha256_file(
        fixture.artifact, label="response-resealed Parquet"
    )
    _write_canonical(fixture.sidecar, sidecar)
    index = json.loads(fixture.snapshot_index.read_text(encoding="utf-8"))
    index["records"][0]["response_sha256"] = response_sha
    index["records"][0]["byte_count"] = len(payload)
    _write_canonical(fixture.snapshot_index, index)
    manifest = json.loads(fixture.manifest.read_text(encoding="utf-8"))
    manifest["raw_snapshot_index_sha256"] = verifier._sha256_file(
        fixture.snapshot_index, label="response-resealed snapshot index"
    )
    manifest["chunks"] = [sidecar]
    _write_canonical(fixture.manifest, manifest)


def test_complete_synthetic_chain_passes_without_promoting_arm(tmp_path: Path) -> None:
    fixture = _build_synthetic(tmp_path)
    report = _verify(fixture)
    assert report["status"] == "VERIFIED_COMPLETE_COVERAGE_GATE_PASSED"
    assert report["integrity_verification_passed"] is True
    assert report["coverage_gate_passed"] is True
    assert report["arm_promoted"] is False
    assert report["contains_or_reads_outcome_labels"] is False
    assert report["inventory"] == {
        "target_start": "2021-03-30",
        "target_end": "2021-03-31",
        "lead_days": [1, 3, 7],
        "station_count": 1,
        "calendar_months_per_station": 1,
        "station_month_chunks": 1,
        "forecast_rows": 6,
        "complete_forecast_rows": 6,
    }
    semantics = report["information_semantics"]
    assert semantics["fixed_lead_composite"] is True
    assert semantics["coherent_single_initialization"] is False
    assert semantics["as_issued_operational_forecast_archive"] is False
    hashes = report["evidence_hashes"]
    assert hashes["protocol"]["sha256"] == verifier._sha256_file(
        fixture.protocol, label="protocol assertion"
    )
    assert hashes["input_registries"][0]["sha256"] == verifier._sha256_file(
        fixture.registry, label="registry assertion"
    )
    assert set(hashes["code"]) == {
        "acquisition_script",
        "independent_verifier",
        "nwp_contract_module",
        "snapshot_store_module",
    }


def test_complete_but_low_coverage_is_reported_and_not_promoted(tmp_path: Path) -> None:
    fixture = _build_synthetic(tmp_path, missing_first_hour_for_lead7=True)
    report = _verify(fixture)
    assert report["status"] == "VERIFIED_COMPLETE_COVERAGE_GATE_FAILED"
    assert report["integrity_verification_passed"] is True
    assert report["coverage_gate_passed"] is False
    assert report["arm_promoted"] is False
    lead7 = [row for row in report["coverage_by_station_and_lead"] if row["lead_days"] == 7]
    assert lead7 == [
        {
            "site_no": SITE_NO,
            "lead_days": 7,
            "expected_target_days": 2,
            "complete_target_days": 1,
            "coverage_fraction": 0.5,
            "passes_0_90_gate": False,
        }
    ]


@pytest.mark.parametrize("missing_role", ["artifact", "sidecar", "response", "metadata"])
def test_missing_inventory_member_fails_closed(tmp_path: Path, missing_role: str) -> None:
    fixture = _build_synthetic(tmp_path)
    getattr(fixture, missing_role).unlink()
    with pytest.raises(verifier.NWPVerificationError, match="inventory mismatch"):
        _verify(fixture)


def test_unregistered_extra_file_fails_closed(tmp_path: Path) -> None:
    fixture = _build_synthetic(tmp_path)
    (fixture.output_dir / "unexpected.txt").write_text("not registered\n", encoding="utf-8")
    with pytest.raises(verifier.NWPVerificationError, match="inventory mismatch"):
        _verify(fixture)


def test_raw_checksum_tampering_fails_closed(tmp_path: Path) -> None:
    fixture = _build_synthetic(tmp_path)
    fixture.response.write_bytes(fixture.response.read_bytes() + b" ")
    with pytest.raises(verifier.NWPVerificationError, match="checksum mismatch"):
        _verify(fixture)


def test_forbidden_raw_label_key_fails_even_after_resealing(tmp_path: Path) -> None:
    fixture = _build_synthetic(tmp_path)
    raw = json.loads(fixture.response.read_text(encoding="utf-8"))
    raw["WTEMP"] = [1.0]
    fixture.response.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    _reseal_after_response_change(fixture)
    with pytest.raises(verifier.NWPVerificationError, match="forbidden label-like key"):
        _verify(fixture)


def test_schema_tampering_fails_even_after_resealing(tmp_path: Path) -> None:
    fixture = _build_synthetic(tmp_path)
    frame = pd.read_parquet(fixture.artifact)
    frame["outcome_label"] = 1.0
    frame.to_parquet(fixture.artifact, index=False)
    _reseal_after_artifact_change(fixture)
    with pytest.raises(verifier.NWPVerificationError, match="schema changed"):
        _verify(fixture)


@pytest.mark.parametrize(
    ("column", "new_value", "message"),
    [
        ("issue_date", pd.Timestamp("2021-01-01"), "issue_date"),
        ("upstream_model", "different model", "upstream model"),
        ("requested_lat", 0.0, "requested latitude"),
        ("available_hour_count", 23, "available_hour_count"),
    ],
)
def test_derived_identity_and_24_hour_semantics_fail_closed_after_resealing(
    tmp_path: Path, column: str, new_value: object, message: str
) -> None:
    fixture = _build_synthetic(tmp_path)
    frame = pd.read_parquet(fixture.artifact)
    frame.loc[0, column] = new_value
    frame.to_parquet(fixture.artifact, index=False)
    _reseal_after_artifact_change(fixture)
    with pytest.raises(verifier.NWPVerificationError, match=message):
        _verify(fixture)


def test_duplicate_forecast_key_fails_after_resealing(tmp_path: Path) -> None:
    fixture = _build_synthetic(tmp_path)
    frame = pd.read_parquet(fixture.artifact)
    for column in ("site_no", "horizon", "issue_date", "target_date"):
        frame.loc[1, column] = frame.loc[0, column]
    frame.to_parquet(fixture.artifact, index=False)
    _reseal_after_artifact_change(fixture)
    with pytest.raises(verifier.NWPVerificationError, match="forecast key is duplicated"):
        _verify(fixture)


def test_snapshot_index_chain_tampering_fails_closed(tmp_path: Path) -> None:
    fixture = _build_synthetic(tmp_path)
    index = json.loads(fixture.snapshot_index.read_text(encoding="utf-8"))
    index["records"][0]["byte_count"] += 1
    _write_canonical(fixture.snapshot_index, index)
    manifest = json.loads(fixture.manifest.read_text(encoding="utf-8"))
    manifest["raw_snapshot_index_sha256"] = verifier._sha256_file(
        fixture.snapshot_index, label="tampered index"
    )
    _write_canonical(fixture.manifest, manifest)
    with pytest.raises(verifier.NWPVerificationError, match="index record"):
        _verify(fixture)


def test_noncanonical_sidecar_fails_closed(tmp_path: Path) -> None:
    fixture = _build_synthetic(tmp_path)
    sidecar = json.loads(fixture.sidecar.read_text(encoding="utf-8"))
    fixture.sidecar.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(verifier.NWPVerificationError, match="not canonical JSON"):
        _verify(fixture)


def test_partial_diagnostic_is_read_only_and_non_authoritative(tmp_path: Path) -> None:
    fixture = _build_synthetic(tmp_path)
    fixture.manifest.unlink()
    fixture.snapshot_index.unlink()
    before = {
        path: path.stat().st_mtime_ns
        for path in (fixture.artifact, fixture.sidecar, fixture.response, fixture.metadata)
    }
    report = verifier.partial_inventory_diagnostic(
        output_dir=fixture.output_dir,
        snapshot_dir=fixture.snapshot_dir,
        registry_paths=[fixture.registry],
        protocol_path=fixture.protocol,
        _expected_target_end=TARGET_END,
        _required_station_count=1,
    )
    assert report["status"] == "PARTIAL_INVENTORY_DIAGNOSTIC_NON_AUTHORITATIVE"
    assert report["integrity_verification_passed"] is False
    assert report["coverage_gate_evaluated"] is False
    assert report["arm_promoted"] is False
    assert report["manifest_present"] is False
    assert report["snapshot_index_present"] is False
    after = {path: path.stat().st_mtime_ns for path in before}
    assert after == before

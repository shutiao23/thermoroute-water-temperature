from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pandas as pd
import pytest

from thermoroute.historical_inputs import (
    ACTUAL_FEATURE_ORDER,
    DAYMET_PROVIDER,
    GRIDMET_PROVIDER,
    GRIDMET_SCHEMA_PROVIDER,
    HistoricalInputError,
    PRELABEL_FIELDS,
    USER_AGENT,
    acquire_historical_inputs,
)
from thermoroute.opening import validate_prelabel_inputs
from thermoroute.provenance import (
    SnapshotStore,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from thermoroute.usgs import (
    build_daymet_url,
    build_gridmet_wind_metadata_url,
    build_gridmet_wind_url,
    parse_daymet_daily,
    parse_gridmet_wind_daily,
    parse_gridmet_wind_metadata,
)


def _daymet_payload(*, offset: float = 0.0) -> bytes:
    return (
        "Daymet Software Version: 4.0\n"
        "year,yday,tmax (deg c),tmin (deg c),prcp (mm/day),"
        "srad (W/m^2),vp (Pa)\n"
        f"2020,335,{10 + offset},{2 + offset},1.5,200,700\n"
        f"2021,1,{12 + offset},{4 + offset},-9999,210,710\n"
    ).encode()


def _gridmet_payload(*, offset: float = 0.0) -> bytes:
    return (
        'time,daily_mean_wind_speed[unit="m/s"]\n'
        f"2020-11-30T00:00:00Z,{35 + offset}\n"
        f"2021-01-01T00:00:00Z,{40 + offset}\n"
    ).encode()


def _gridmet_schema_payload(*, scale: float = 0.1) -> bytes:
    return (
        "Attributes {\n"
        "  daily_mean_wind_speed {\n"
        '    String units "m/s";\n'
        f"    Float64 scale_factor {scale};\n"
        "    Float64 add_offset 0.0;\n"
        "  }\n"
        "}\n"
    ).encode()


def _protocol(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes({
        "schema_version": 1,
        "status": "FROZEN_NOT_ACQUIRED",
        "time_holdout": {
            "primary_target_start": "2021-01-01",
            "end": "2023-12-31",
        },
        "primary_historical_input_contract": {
            "horizon_specific_future_nwp_consumed": False,
            "retrospective_meteorological_inputs": list(PRELABEL_FIELDS),
            "provisional_vintage_limitation": (
                "The as-issued provisional vintage cannot be reconstructed from "
                "the retrospective gridded products."
            ),
        },
        "primary_inference_contract": {
            "feature_order": list(ACTUAL_FEATURE_ORDER),
            "wlevel_consumed": False,
        },
    }))


def _registry(path: Path, site_no: str, lat: float, lon: float) -> None:
    pd.DataFrame({
        "site_no": [site_no],
        "lat": [lat],
        "lon": [lon],
        # This forbidden-looking value proves acquisition uses only usecols and
        # never loads a legacy outcome/availability field.
        "wtemp_cov_test": [0.99],
    }).to_csv(path, index=False)


def _seed_snapshot(root: Path, *, provider: str, url: str, payload: bytes) -> None:
    headers = {"User-Agent": USER_AGENT}
    request = SnapshotStore.request_document(
        provider=provider, url=url, headers=headers
    )
    request_sha = sha256_bytes(canonical_json_bytes(request))
    base = root / SnapshotStore._provider_name(provider) / request_sha
    base.mkdir(parents=True, exist_ok=False)
    (base / "response.bin").write_bytes(payload)
    (base / "metadata.json").write_bytes(canonical_json_bytes({
        "schema_version": 1,
        "request": request,
        "request_sha256": request_sha,
        "retrieved_at_utc": "2026-07-21T00:00:00+00:00",
        "http_status": 200,
        "response_headers": {"Content-Type": "text/csv"},
        "byte_count": len(payload),
        "response_sha256": sha256_bytes(payload),
        "response_file": "response.bin",
    }))


def _seed_site(root: Path, *, lat: float, lon: float, offset: float) -> None:
    start, end = "2020-11-30", "2023-12-31"
    _seed_snapshot(
        root / "daymet-v1",
        provider=DAYMET_PROVIDER,
        url=build_daymet_url(lat, lon, start, end),
        payload=_daymet_payload(offset=offset),
    )
    _seed_snapshot(
        root / "gridmet-v1",
        provider=GRIDMET_PROVIDER,
        url=build_gridmet_wind_url(lat, lon, start, end),
        payload=_gridmet_payload(offset=offset),
    )


def _seed_gridmet_schema(root: Path) -> None:
    _seed_snapshot(
        root / "gridmet-schema-v1",
        provider=GRIDMET_SCHEMA_PROVIDER,
        url=build_gridmet_wind_metadata_url(),
        payload=_gridmet_schema_payload(),
    )


def test_meteorology_parsers_keep_complete_calendar_and_exact_schema():
    daymet = parse_daymet_daily(
        _daymet_payload(), start="2020-11-30", end="2021-01-02"
    )
    assert list(daymet.columns) == ["TEMP", "PRCP", "RHMEAN", "DH"]
    assert len(daymet) == 34
    assert daymet.index.min() == pd.Timestamp("2020-11-30")
    assert daymet.loc["2020-11-30", "TEMP"] == pytest.approx(6.0)
    assert np.isnan(daymet.loc["2021-01-01", "PRCP"])
    assert daymet.loc["2020-12-01"].isna().all()

    wind = parse_gridmet_wind_daily(
        _gridmet_payload(), start="2020-11-30", end="2021-01-02"
    )
    assert wind.name == "WDSP"
    assert len(wind) == 34
    assert wind.loc["2020-11-30"] == pytest.approx(3.5)
    assert np.isnan(wind.loc["2020-12-01"])
    contract = parse_gridmet_wind_metadata(_gridmet_schema_payload())
    assert contract == {"units": "m/s", "scale_factor": 0.1, "add_offset": 0.0}
    with pytest.raises(ValueError, match="scale_factor changed"):
        parse_gridmet_wind_metadata(_gridmet_schema_payload(scale=1.0))


def test_daymet_leap_calendar_keeps_feb29_and_omits_dec31():
    # Official Daymet convention: every year has yday 1..365; a leap year
    # retains February 29 and discards December 31 (so yday365 is Dec 30).
    header = (
        "year,yday,tmax (deg c),tmin (deg c),prcp (mm/day),"
        "srad (W/m^2),vp (Pa)\n"
    )
    payload = (header + "2020,59,1,0,0,1,600\n"
               + "2020,60,2,0,0,1,600\n"
               + "2020,365,3,0,0,1,600\n").encode()
    frame = parse_daymet_daily(payload, start="2020-02-28", end="2020-12-31")
    assert frame.loc["2020-02-28", "TEMP"] == pytest.approx(0.5)
    assert frame.loc["2020-02-29", "TEMP"] == pytest.approx(1.0)
    assert frame.loc["2020-12-30", "TEMP"] == pytest.approx(1.5)
    assert frame.loc["2020-12-31"].isna().all()


def test_meteorology_urls_are_canonical_and_cannot_name_outcomes():
    daymet = build_daymet_url(40.123456789, -105.987654321, "2020-11-30", "2023-12-31")
    query = parse_qs(urlsplit(daymet).query)
    assert query["lat"] == ["40.12345679"]
    assert query["lon"] == ["-105.98765432"]
    assert query["vars"] == ["tmax,tmin,prcp,srad,vp"]
    assert query["start"] == ["2020-01-01"]
    assert query["end"] == ["2023-12-31"]

    gridmet = build_gridmet_wind_url(
        40.123456789, -105.987654321, "2020-11-30", "2023-12-31"
    )
    grid_query = parse_qs(urlsplit(gridmet).query)
    assert grid_query["var"] == ["daily_mean_wind_speed"]
    assert grid_query["time_start"] == ["2020-11-30T00:00:00Z"]
    assert grid_query["time_end"] == ["2023-12-31T00:00:00Z"]
    for url in (daymet, gridmet):
        lowered = url.lower()
        assert "/nwis/" not in lowered
        assert not any(code in lowered for code in ("00010", "00060", "00065"))


def test_offline_fixture_freezes_opening_compatible_inputs(tmp_path, monkeypatch):
    protocol_path = tmp_path / "protocols" / "route_a_confirmatory_v1.json"
    temporal_registry = tmp_path / "data_usgs" / "temporal.csv"
    external_registry = tmp_path / "data_usgs" / "external.csv"
    temporal_registry.parent.mkdir(parents=True)
    _protocol(protocol_path)
    _registry(temporal_registry, "01234567", 40.0, -105.0)
    _registry(external_registry, "07654321", 41.0, -104.0)
    snapshot_root = tmp_path / "data_usgs" / "raw_snapshots" / "historical-v1"
    _seed_gridmet_schema(snapshot_root)
    _seed_site(snapshot_root, lat=40.0, lon=-105.0, offset=0.0)
    _seed_site(snapshot_root, lat=41.0, lon=-104.0, offset=1.0)

    def network_forbidden(*_args, **_kwargs):
        raise AssertionError("offline replay attempted network access")

    monkeypatch.setattr("urllib.request.urlopen", network_forbidden)
    output_dir = tmp_path / "data_usgs" / "confirmatory_predictors" / "historical-v1"
    manifest_path = tmp_path / "data_usgs" / "confirmatory_actual_inputs_v1.json"
    manifest = acquire_historical_inputs(
        repo_root=tmp_path,
        protocol_path=protocol_path,
        temporal_registry_path=temporal_registry,
        external_registry_path=external_registry,
        snapshot_root=snapshot_root,
        output_dir=output_dir,
        manifest_path=manifest_path,
        offline=True,
        expected_temporal_sites=1,
        expected_external_sites=1,
    )
    assert manifest["contains_outcome_labels"] is False
    assert manifest["post_2020_wtemp_requested_or_inspected"] is False
    assert manifest["retrospective_provisional_vintage_reconstructable"] is False
    assert manifest["secondary_nwp_resolution"] == "EXPLICITLY_NOT_USED"
    assert manifest["history_start"] == "2020-11-30"

    expected_columns = {"site_no", "DATE", *PRELABEL_FIELDS}
    for cohort in ("temporal", "external"):
        binding = manifest["cohort_tables"][cohort]
        table_path = tmp_path / binding["path"]
        table = pd.read_parquet(table_path)
        assert set(table.columns) == expected_columns
        assert len(table) == len(pd.date_range("2020-11-30", "2023-12-31"))
        assert sha256_file(table_path) == binding["sha256"]
        assert not {"WTEMP", "FLOW", "WLEVEL"} & set(table.columns)

    validated = validate_prelabel_inputs(
        manifest_path,
        root=tmp_path,
        protocol_info={
            "protocol_sha256": sha256_file(protocol_path),
            "target_start": "2021-01-01",
            "target_end": "2023-12-31",
        },
        registries={
            "development": pd.DataFrame({
                "site_no": ["01234567"], "lat": [40.0], "lon": [-105.0],
            }),
            "external": pd.DataFrame({
                "site_no": ["07654321"], "lat": [41.0], "lon": [-104.0],
            }),
            "development_sha256": sha256_file(temporal_registry),
            "external_sha256": sha256_file(external_registry),
        },
        suite={"feature_order": ACTUAL_FEATURE_ORDER},
    )
    assert validated["history_start"] == "2020-11-30"

    # A fresh normalized destination can be reproduced entirely from immutable
    # raw snapshots.  Existing indexes are byte-verified, never rewritten.
    index_hashes = {
        provider: sha256_file(snapshot_root / provider / "snapshot_index.json")
        for provider in ("daymet-v1", "gridmet-v1", "gridmet-schema-v1")
    }
    replay = acquire_historical_inputs(
        repo_root=tmp_path,
        protocol_path=protocol_path,
        temporal_registry_path=temporal_registry,
        external_registry_path=external_registry,
        snapshot_root=snapshot_root,
        output_dir=output_dir.with_name("historical-v1-offline-replay"),
        manifest_path=manifest_path.with_name("confirmatory_actual_inputs_replay.json"),
        offline=True,
        expected_temporal_sites=1,
        expected_external_sites=1,
    )
    assert replay["cohort_summaries"] == manifest["cohort_summaries"]
    assert index_hashes == {
        provider: sha256_file(snapshot_root / provider / "snapshot_index.json")
        for provider in ("daymet-v1", "gridmet-v1", "gridmet-schema-v1")
    }

    with pytest.raises(HistoricalInputError, match="replace immutable"):
        acquire_historical_inputs(
            repo_root=tmp_path,
            protocol_path=protocol_path,
            temporal_registry_path=temporal_registry,
            external_registry_path=external_registry,
            snapshot_root=snapshot_root,
            output_dir=output_dir,
            manifest_path=manifest_path,
            offline=True,
            expected_temporal_sites=1,
            expected_external_sites=1,
        )

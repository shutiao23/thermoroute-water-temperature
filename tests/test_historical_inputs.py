from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
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
    ProvenanceError,
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


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "data_usgs" / "fetch_confirmatory_historical_inputs.py"


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


def _input_fixture(
    tmp_path: Path,
    *,
    omit_final_gridmet: bool = False,
) -> dict[str, object]:
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
    if omit_final_gridmet:
        _seed_snapshot(
            snapshot_root / "daymet-v1",
            provider=DAYMET_PROVIDER,
            url=build_daymet_url(
                41.0, -104.0, "2020-11-30", "2023-12-31"
            ),
            payload=_daymet_payload(offset=1.0),
        )
    else:
        _seed_site(snapshot_root, lat=41.0, lon=-104.0, offset=1.0)
    return {
        "repo_root": tmp_path,
        "protocol_path": protocol_path,
        "temporal_registry_path": temporal_registry,
        "external_registry_path": external_registry,
        "snapshot_root": snapshot_root,
        "output_dir": (
            tmp_path / "data_usgs" / "confirmatory_predictors" / "historical-v1"
        ),
        "manifest_path": tmp_path / "data_usgs" / "confirmatory_actual_inputs_v1.json",
        "offline": not omit_final_gridmet,
        "expected_temporal_sites": 1,
        "expected_external_sites": 1,
    }


class _FakeResponse:
    status = 200
    headers = {"Content-Type": "text/csv"}

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        return False

    def read(self) -> bytes:
        return self._payload


def _tree_file_state(root: Path) -> dict[str, tuple[str, int, int, int]]:
    state: dict[str, tuple[str, int, int, int]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        info = path.stat()
        state[path.relative_to(root).as_posix()] = (
            sha256_file(path), info.st_ino, info.st_mtime_ns, info.st_nlink,
        )
    return state


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

    # A complete retry is a strict, idempotent offline verification.  It does
    # not replace any already frozen evidence.
    assert acquire_historical_inputs(
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
    ) == manifest


def test_raw_response_only_crash_resumes_by_byte_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _input_fixture(tmp_path, omit_final_gridmet=True)
    expected_payload = _gridmet_payload(offset=1.0)
    calls: list[str] = []

    def fake_urlopen(request, **_kwargs):
        calls.append(request.full_url)
        return _FakeResponse(expected_payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    def crash(stage: str, _path: Path) -> None:
        if stage == "after_raw_response_publish":
            raise RuntimeError("simulated response-metadata crash")

    with pytest.raises(RuntimeError, match="response-metadata crash"):
        acquire_historical_inputs(**fixture, _fault_injector=crash)

    url = build_gridmet_wind_url(
        41.0, -104.0, "2020-11-30", "2023-12-31"
    )
    request = SnapshotStore.request_document(
        provider=GRIDMET_PROVIDER,
        url=url,
        headers={"User-Agent": USER_AGENT},
    )
    request_sha = sha256_bytes(canonical_json_bytes(request))
    request_dir = (
        Path(fixture["snapshot_root"])
        / "gridmet-v1"
        / SnapshotStore._provider_name(GRIDMET_PROVIDER)
        / request_sha
    )
    response_path = request_dir / "response.bin"
    metadata_path = request_dir / "metadata.json"
    response_state = (
        sha256_file(response_path), response_path.stat().st_ino,
        response_path.stat().st_mtime_ns,
    )
    assert response_path.read_bytes() == expected_payload
    assert not metadata_path.exists()

    manifest = acquire_historical_inputs(**fixture)
    assert manifest["status"] == "FROZEN_PRELABEL_NO_OUTCOMES"
    assert metadata_path.is_file()
    assert response_state == (
        sha256_file(response_path), response_path.stat().st_ino,
        response_path.stat().st_mtime_ns,
    )
    assert calls == [url, url]


def test_raw_response_resume_rejects_provider_byte_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _input_fixture(tmp_path, omit_final_gridmet=True)
    first_payload = _gridmet_payload(offset=1.0)
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_args, **_kwargs: _FakeResponse(first_payload)
    )

    def crash(stage: str, _path: Path) -> None:
        if stage == "after_raw_response_publish":
            raise RuntimeError("simulated raw crash")

    with pytest.raises(RuntimeError, match="simulated raw crash"):
        acquire_historical_inputs(**fixture, _fault_injector=crash)
    # Select the response-only request rather than either already complete one.
    response = next(
        path
        for path in Path(fixture["snapshot_root"]).glob("gridmet-v1/*/*/response.bin")
        if not (path.parent / "metadata.json").exists()
    )
    original = response.read_bytes()
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: _FakeResponse(_gridmet_payload(offset=9.0)),
    )

    with pytest.raises(ProvenanceError, match="differs from incomplete snapshot"):
        acquire_historical_inputs(**fixture)

    assert response.read_bytes() == original
    assert not (response.parent / "metadata.json").exists()


@pytest.mark.parametrize(
    "fault_stage",
    ("after_daymet_index_publish", "after_normalized_bundle_publish"),
)
def test_derived_publication_boundaries_resume_only_missing_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_stage: str,
) -> None:
    fixture = _input_fixture(tmp_path)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: pytest.fail("offline resume attempted network"),
    )

    def crash(stage: str, _path: Path) -> None:
        if stage == fault_stage:
            raise RuntimeError(f"simulated {fault_stage}")

    with pytest.raises(RuntimeError, match=fault_stage):
        acquire_historical_inputs(**fixture, _fault_injector=crash)
    before = _tree_file_state(tmp_path)
    if fault_stage == "after_normalized_bundle_publish":
        assert Path(fixture["output_dir"]).is_dir()
        assert not Path(fixture["manifest_path"]).exists()
    else:
        assert (
            Path(fixture["snapshot_root"])
            / "daymet-v1"
            / "snapshot_index.json"
        ).is_file()
        assert not Path(fixture["output_dir"]).exists()

    manifest = acquire_historical_inputs(**fixture)
    assert manifest["status"] == "FROZEN_PRELABEL_NO_OUTCOMES"
    after = _tree_file_state(tmp_path)
    assert all(after[path] == state for path, state in before.items())


def test_check_existing_is_fully_offline_and_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _input_fixture(tmp_path)
    expected = acquire_historical_inputs(**fixture)
    before = _tree_file_state(tmp_path)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: pytest.fail("--check-existing attempted network"),
    )
    checked = acquire_historical_inputs(
        **{**fixture, "offline": False},
        check_existing=True,
    )
    assert checked == expected
    assert _tree_file_state(tmp_path) == before


def test_check_existing_never_completes_a_missing_manifest(
    tmp_path: Path,
) -> None:
    fixture = _input_fixture(tmp_path)

    def crash(stage: str, _path: Path) -> None:
        if stage == "after_normalized_bundle_publish":
            raise RuntimeError("simulated bundle-only state")

    with pytest.raises(RuntimeError, match="bundle-only"):
        acquire_historical_inputs(**fixture, _fault_injector=crash)
    before = _tree_file_state(tmp_path)
    with pytest.raises(HistoricalInputError, match="complete manifest"):
        acquire_historical_inputs(**fixture, check_existing=True)
    assert _tree_file_state(tmp_path) == before
    assert not Path(fixture["manifest_path"]).exists()


@pytest.mark.parametrize(
    "attack",
    ("table", "request_map", "index", "manifest", "extra_bundle", "extra_raw"),
)
def test_existing_state_tamper_and_extra_entries_fail_closed(
    tmp_path: Path,
    attack: str,
) -> None:
    fixture = _input_fixture(tmp_path)
    acquire_historical_inputs(**fixture)
    output_dir = Path(fixture["output_dir"])
    snapshot_root = Path(fixture["snapshot_root"])
    manifest_path = Path(fixture["manifest_path"])
    attacked_path: Path
    if attack == "table":
        attacked_path = output_dir / "temporal_retrospective_meteorology_v1.parquet"
        attacked_path.chmod(0o600)
        attacked_path.write_bytes(attacked_path.read_bytes() + b"attacker")
    elif attack == "request_map":
        attacked_path = output_dir / "source_request_map_v1.json"
        attacked_path.chmod(0o600)
        attacked_path.write_bytes(attacked_path.read_bytes() + b" ")
    elif attack == "index":
        attacked_path = snapshot_root / "daymet-v1" / "snapshot_index.json"
        attacked_path.chmod(0o600)
        attacked_path.write_bytes(attacked_path.read_bytes() + b" ")
    elif attack == "manifest":
        attacked_path = manifest_path
        attacked_path.chmod(0o600)
        attacked_path.write_bytes(attacked_path.read_bytes() + b" ")
    elif attack == "extra_bundle":
        attacked_path = output_dir / "unexpected.bin"
        attacked_path.write_bytes(b"extra")
    else:
        attacked_path = snapshot_root / "daymet-v1" / "unexpected-provider"
        attacked_path.mkdir()
    before = _tree_file_state(tmp_path)

    with pytest.raises(HistoricalInputError):
        acquire_historical_inputs(**fixture, check_existing=True)

    assert attacked_path.exists()
    assert _tree_file_state(tmp_path) == before


def test_existing_symlink_and_hardlink_evidence_fail_closed(
    tmp_path: Path,
) -> None:
    symlink_fixture = _input_fixture(tmp_path / "symlink")
    acquire_historical_inputs(**symlink_fixture)
    manifest = Path(symlink_fixture["manifest_path"])
    retained = manifest.with_name("retained-manifest-evidence.json")
    manifest.rename(retained)
    manifest.symlink_to(retained.name)
    with pytest.raises(HistoricalInputError, match="symlink"):
        acquire_historical_inputs(**symlink_fixture, check_existing=True)
    assert manifest.is_symlink() and retained.is_file()

    hardlink_fixture = _input_fixture(tmp_path / "hardlink")
    acquire_historical_inputs(**hardlink_fixture)
    hardlinked_manifest = Path(hardlink_fixture["manifest_path"])
    external_link = hardlinked_manifest.with_name("external-hardlink-evidence.json")
    os.link(hardlinked_manifest, external_link)
    with pytest.raises(HistoricalInputError, match="single-link"):
        acquire_historical_inputs(**hardlink_fixture, check_existing=True)
    assert hardlinked_manifest.stat().st_nlink == 2
    assert external_link.read_bytes() == hardlinked_manifest.read_bytes()


def test_partial_staging_and_partial_manifest_are_preserved_and_rejected(
    tmp_path: Path,
) -> None:
    staging_fixture = _input_fixture(tmp_path / "staging")

    def staging_crash(stage: str, _path: Path) -> None:
        if stage == "after_temporal_table_staged":
            raise RuntimeError("simulated half-staged bundle")

    with pytest.raises(RuntimeError, match="half-staged"):
        acquire_historical_inputs(**staging_fixture, _fault_injector=staging_crash)
    output_dir = Path(staging_fixture["output_dir"])
    staging = list(output_dir.parent.glob(f".{output_dir.name}.*"))
    assert len(staging) == 1
    before = _tree_file_state(tmp_path / "staging")
    with pytest.raises(HistoricalInputError, match="staging evidence"):
        acquire_historical_inputs(**staging_fixture)
    assert staging[0].exists()
    assert _tree_file_state(tmp_path / "staging") == before

    manifest_fixture = _input_fixture(tmp_path / "manifest")

    def bundle_crash(stage: str, _path: Path) -> None:
        if stage == "after_normalized_bundle_publish":
            raise RuntimeError("simulated pre-manifest crash")

    with pytest.raises(RuntimeError, match="pre-manifest"):
        acquire_historical_inputs(**manifest_fixture, _fault_injector=bundle_crash)
    partial_manifest = Path(manifest_fixture["manifest_path"])
    partial_manifest.write_bytes(b"{")
    with pytest.raises(HistoricalInputError, match="deterministic replay"):
        acquire_historical_inputs(**manifest_fixture)
    assert partial_manifest.read_bytes() == b"{"


def test_cli_exposes_check_existing_mode() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--check-existing" in result.stdout
    assert "fully offline" in result.stdout

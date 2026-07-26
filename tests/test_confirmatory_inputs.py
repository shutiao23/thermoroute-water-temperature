from __future__ import annotations

from datetime import date
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute.confirmatory import (  # noqa: E402
    CANDIDATE_COLUMNS,
    CANDIDATE_PROVIDER,
    CANDIDATE_SELECTION_RULE,
    CANDIDATE_STATE_UNIVERSE_RULE,
    CANDIDATE_USER_AGENT,
    ROUTE_A_STATE_UNIVERSE,
    build_usgs_candidate_url,
    canonical_candidate_request,
    merge_candidate_metadata,
    parse_usgs_candidate_metadata,
    replay_candidate_evidence,
)
from thermoroute.evidence import EvidenceError  # noqa: E402
import thermoroute.opening as opening_module  # noqa: E402
from thermoroute.nwp import (  # noqa: E402
    ISSUE_SEMANTICS,
    NWP_COMMON_VALID_TIME_START,
    NWPContractError,
    build_previous_runs_url,
    iter_month_chunks,
    parse_previous_runs_daily,
)
from thermoroute.provenance import (  # noqa: E402
    ProvenanceError,
    candidate_publication_lock,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)


def _candidate_payload(site_no: str = "01234567") -> bytes:
    return (
        "agency_cd\tsite_no\tstation_nm\tsite_tp_cd\tdec_lat_va\t"
        "dec_long_va\thuc_cd\tdrain_area_va\n"
        "5s\t15s\t50s\t7s\t16n\t16n\t16s\t16n\n"
        f"USGS\t{site_no}\tExample River\tST\t40.125\t-105.250\t"
        "10190005\t42.5\n"
    ).encode("utf-8")


def _nwp_payload(*, missing: tuple[int, int] | None = None) -> bytes:
    times = pd.date_range("2021-04-01", periods=48, freq="h")
    hourly: dict[str, object] = {
        "time": [timestamp.strftime("%Y-%m-%dT%H:%M") for timestamp in times]
    }
    units: dict[str, str] = {"time": "iso8601"}
    for lead in (1, 3, 7):
        values: list[float | None] = [float(index + lead) for index in range(48)]
        if missing is not None and missing[0] == lead:
            values[missing[1]] = None
        field = f"temperature_2m_previous_day{lead}"
        hourly[field] = values
        units[field] = "°C"
    return json.dumps({
        "latitude": 40.13,
        "longitude": -105.25,
        "generationtime_ms": 0.2,
        "utc_offset_seconds": 0,
        "timezone": "GMT",
        "timezone_abbreviation": "GMT",
        "elevation": 1600.0,
        "hourly_units": units,
        "hourly": hourly,
    }).encode("utf-8")


def _load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_url_is_stable_metadata_only_query():
    url = build_usgs_candidate_url("co")
    parsed = urlsplit(url)
    query = parse_qs(parsed.query)
    assert parsed.path == "/nwis/site/"
    assert query == {
        "agencyCd": ["USGS"],
        "format": ["rdb"],
        "hasDataTypeCd": ["dv"],
        "parameterCd": ["00010"],
        "siteOutput": ["expanded"],
        "siteStatus": ["all"],
        "siteType": ["ST"],
        "stateCd": ["CO"],
    }
    assert not ({"startDT", "endDT", "start_date", "end_date"} & set(query))
    assert "/nwis/dv/" not in url


def test_candidate_parser_keeps_only_metadata_and_merges_stably():
    colorado = parse_usgs_candidate_metadata(_candidate_payload("01234567"), state="CO")
    oregon = parse_usgs_candidate_metadata(_candidate_payload("09876543"), state="OR")
    assert tuple(colorado.columns) == CANDIDATE_COLUMNS
    assert colorado.loc[0, "site_no"] == "01234567"
    assert not any("wtemp" in column.lower() for column in colorado.columns)
    merged = merge_candidate_metadata([oregon, colorado])
    assert merged["site_no"].tolist() == ["01234567", "09876543"]

    with pytest.raises(EvidenceError, match="multiple responses"):
        merge_candidate_metadata([colorado, colorado])


def test_discovery_freeze_command_calls_existing_holdout_script(tmp_path):
    module = _load_script(
        "discover_confirmatory_candidates_test",
        "scripts/data_usgs/discover_confirmatory_candidates.py",
    )
    command = module.holdout_freeze_command(
        candidates=tmp_path / "candidates.csv",
        snapshot_index=tmp_path / "snapshot_index.json",
        candidate_provenance=tmp_path / "candidate.provenance.json",
        out_registry=tmp_path / "registry.csv",
        out_lock=tmp_path / "lock.json",
        protocol=tmp_path / "protocol.json",
        n_sites=30,
        selection_seed="route-a-confirmatory-v1-public-seed",
    )
    assert Path(command[1]).name == "confirmatory_holdout.py"
    assert "freeze-candidates" in command
    assert command[command.index("--n-sites") + 1] == "30"
    assert "--candidate-provenance" in command
    assert command[command.index("--selection-seed") + 1] == (
        "route-a-confirmatory-v1-public-seed"
    )

    check_command = module.holdout_freeze_command(
        candidates=tmp_path / "candidates.csv",
        snapshot_index=tmp_path / "snapshot_index.json",
        candidate_provenance=tmp_path / "candidate.provenance.json",
        out_registry=tmp_path / "registry.csv",
        out_lock=tmp_path / "lock.json",
        protocol=tmp_path / "protocol.json",
        n_sites=30,
        selection_seed="route-a-confirmatory-v1-public-seed",
        check_existing=True,
    )
    assert "check-candidates" in check_command


def test_holdout_isolation_reexec_preserves_inherited_lock_fd(
    tmp_path, monkeypatch,
):
    module = _load_script(
        "confirmatory_holdout_isolation_fd_test",
        "scripts/data_usgs/confirmatory_holdout.py",
    )
    lock_path = tmp_path / "publication.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    captured = {}

    def fake_run(*_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(returncode=0)

    try:
        monkeypatch.setattr(module, "__name__", "__main__")
        monkeypatch.setattr(module.subprocess, "run", fake_run)
        monkeypatch.setattr(
            module.sys,
            "argv",
            [
                str(ROOT / "scripts/data_usgs/confirmatory_holdout.py"),
                "freeze-candidates",
                "--inherited-publication-lock-fd",
                str(descriptor),
                "--inherited-publication-lock-token",
                "a" * 64,
            ],
        )
        with pytest.raises(SystemExit) as exited:
            module._isolate_project_bytecode()
        assert exited.value.code == 0
        assert captured["pass_fds"] == (descriptor,)
    finally:
        os.close(descriptor)


def test_real_holdout_isolation_chain_retains_parent_lock_capability():
    lock_path = ROOT / "protocols" / "route_a_confirmatory_v1.json"
    missing_prefix = "outputs/__candidate_fd_probe_missing"
    with candidate_publication_lock(lock_path, trusted_root=ROOT) as owner:
        command = [
            sys.executable,
            str(ROOT / "scripts/data_usgs/confirmatory_holdout.py"),
            "freeze-candidates",
            "--protocol", "protocols/route_a_confirmatory_v1.json",
            "--development-spec", "data_usgs/frozen_panel_v1.json",
            "--candidates", f"{missing_prefix}.csv",
            "--candidate-snapshot-index", f"{missing_prefix}_index.json",
            "--candidate-provenance", f"{missing_prefix}_provenance.json",
            "--out-registry", f"{missing_prefix}_registry.csv",
            "--out-lock", f"{missing_prefix}_registry.lock.json",
            "--n-sites", "30",
            "--selection-seed", "route-a-confirmatory-v1-public-seed",
            "--inherited-publication-lock-fd", str(owner.fd),
            "--inherited-publication-lock-token", owner.token,
        ]
        result = subprocess.run(
            command,
            cwd=ROOT,
            pass_fds=(owner.fd,),
            text=True,
            capture_output=True,
            check=False,
        )
    assert result.returncode != 0
    assert "candidate table cannot be rebuilt" in result.stderr
    assert "Bad file descriptor" not in result.stderr
    assert "descriptor is not open" not in result.stderr
    assert "transaction is busy" not in result.stderr


def _candidate_publication_fixture(tmp_path: Path):
    discovery = _load_script(
        f"discover_candidate_resume_{tmp_path.name}",
        "scripts/data_usgs/discover_confirmatory_candidates.py",
    )
    holdout = _load_script(
        f"confirmatory_holdout_resume_{tmp_path.name}",
        "scripts/data_usgs/confirmatory_holdout.py",
    )
    discovery.ROOT = tmp_path
    discovery.ROUTE_A_STATE_UNIVERSE = ("CO",)
    holdout.ROOT = tmp_path

    data_root = tmp_path / "data_usgs"
    snapshot_root = data_root / "raw_snapshots" / "confirmatory-candidates-v1"
    candidate_path = data_root / "confirmatory_candidate_sites_v1.csv"
    provenance_path = candidate_path.with_suffix(".provenance.json")
    registry_path = data_root / "confirmatory_site_registry_v1.csv"
    lock_path = data_root / "confirmatory_site_registry_v1.lock.json"
    protocol_path = tmp_path / "protocols" / "route_a_confirmatory_v1.json"
    protocol_path.parent.mkdir(parents=True)
    selection_seed = "route-a-confirmatory-v1-public-seed"
    protocol_path.write_bytes(canonical_json_bytes({
        "schema_version": 1,
        "status": "PLANNED_NOT_ACQUIRED",
        "protocol_id": "route-a-fixture",
        "authoritative_protocol_commit": "b" * 40,
        "pre_label_amendments": [],
        "new_site_external_validation": {
            "status": "PLANNED_NOT_ACQUIRED",
            "planned_site_count": 1,
            "selection_seed": selection_seed,
        },
        "metadata_candidate_contract": {"state_universe": ["CO"]},
        "time_holdout": {"start": "2021-01-01", "end": "2023-12-31"},
    }))

    development_registry = data_root / "station_registry_v1.csv"
    development_registry.parent.mkdir(parents=True, exist_ok=True)
    development = pd.DataFrame({
        "site_no": [f"{70_000_000 + index:08d}" for index in range(120)],
        "legacy_site_id": [f"n{index + 1:03d}" for index in range(120)],
        "station_nm": [f"Development River {index + 1}" for index in range(120)],
        "lat": [39.5 + index / 1000 for index in range(120)],
        "lon": [-104.5 - index / 1000 for index in range(120)],
        "state": ["CO"] * 120,
        "huc_cd": ["10190005"] * 120,
        "huc2": ["10"] * 120,
        "huc_metadata_status": ["USGS_SNAPSHOT_SITE_NO_MATCH"] * 120,
    })
    development_registry.write_text(
        development.to_csv(
            index=False,
            float_format="%.17g",
            lineterminator="\n",
        ),
        encoding="utf-8",
    )
    source_metadata = data_root / "stations_meta_120v2.csv"
    source_metadata.write_text("fixture\n", encoding="utf-8")
    development_spec = data_root / "frozen_panel_v1.json"
    development_spec.write_bytes(canonical_json_bytes({
        "schema_version": 1,
        "station_registry": {
            "path": development_registry.name,
            "sha256": sha256_file(development_registry),
            "source_metadata_path": source_metadata.name,
            "source_metadata_sha256": sha256_file(source_metadata),
            "station_count": 120,
        },
    }))

    raw_payload = _candidate_payload()
    request = {
        "schema_version": 1,
        "provider": CANDIDATE_PROVIDER,
        "method": "GET",
        "url": build_usgs_candidate_url("CO"),
        "headers": {"User-Agent": CANDIDATE_USER_AGENT},
    }
    request_sha = sha256_bytes(canonical_json_bytes(request))
    response_sha = sha256_bytes(raw_payload)
    transaction = snapshot_root / CANDIDATE_PROVIDER / request_sha
    transaction.mkdir(parents=True)
    response_path = transaction / "response.bin"
    metadata_path = transaction / "metadata.json"
    response_path.write_bytes(raw_payload)
    metadata_path.write_bytes(canonical_json_bytes({
        "schema_version": 2,
        "request": request,
        "request_sha256": request_sha,
        "retrieved_at_utc": "2026-01-01T00:00:00+00:00",
        "http_status": 200,
        "response_headers": {},
        "byte_count": len(raw_payload),
        "response_sha256": response_sha,
        "response_file": "response.bin",
        "final_url": request["url"],
        "retrieval_semantics": "DIRECT_HTTP_RESPONSE",
    }))

    discovery_args = SimpleNamespace(
        states=["CO"],
        snapshot_dir=snapshot_root,
        out=candidate_path,
        protocol=protocol_path,
        offline=True,
        check_existing=False,
        retries=1,
        freeze_selection=True,
        out_registry=registry_path,
        out_lock=lock_path,
        n_sites=1,
    )
    holdout_args = SimpleNamespace(
        protocol=protocol_path,
        development_spec=development_spec,
        candidates=candidate_path,
        candidate_snapshot_index=snapshot_root / "snapshot_index.json",
        candidate_provenance=provenance_path,
        out_registry=registry_path,
        out_lock=lock_path,
        n_sites=1,
        selection_seed=selection_seed,
    )

    def local_holdout(command, **_kwargs):
        check_only = "check-candidates" in command
        holdout_args.inherited_publication_lock_fd = int(
            command[command.index("--inherited-publication-lock-fd") + 1]
        )
        holdout_args.inherited_publication_lock_token = command[
            command.index("--inherited-publication-lock-token") + 1
        ]
        holdout.publish_or_validate(holdout_args, check_only=check_only)
        return SimpleNamespace(returncode=0)

    discovery.subprocess = SimpleNamespace(run=local_holdout)
    paths = {
        "table": candidate_path,
        "index": snapshot_root / "snapshot_index.json",
        "provenance": provenance_path,
        "registry": registry_path,
        "lock": lock_path,
    }
    return discovery, holdout, discovery_args, holdout_args, paths


def _opening_registry_lock_validation_fixture(tmp_path: Path, monkeypatch):
    discovery, _holdout, args, holdout_args, paths = (
        _candidate_publication_fixture(tmp_path)
    )
    discovery.discover(args)
    development_spec_document = json.loads(
        holdout_args.development_spec.read_text(encoding="utf-8")
    )
    development_registry = (
        holdout_args.development_spec.parent
        / development_spec_document["station_registry"]["path"]
    ).resolve()
    frozen_spec = SimpleNamespace(
        registry_path=development_registry,
        verify=lambda: {"registry_sha256": sha256_file(development_registry)},
    )
    monkeypatch.setattr(
        opening_module.FrozenPanelSpec,
        "load",
        classmethod(lambda _cls, _path: frozen_spec),
    )
    protocol = json.loads(holdout_args.protocol.read_text(encoding="utf-8"))
    lock = json.loads(paths["lock"].read_text(encoding="utf-8"))
    validation_kwargs = {
        "root": tmp_path,
        "protocol_info": {
            "document": protocol,
            "protocol_sha256": sha256_file(holdout_args.protocol),
            "authoritative_commit": protocol["authoritative_protocol_commit"],
            "amendments_sha256": lock["pre_label_amendments_sha256"],
        },
        "development_registry": development_registry,
        "external_registry": paths["registry"],
        "external_lock": paths["lock"],
    }
    return paths["lock"], lock, validation_kwargs


def test_opening_accepts_exact_registry_lock(tmp_path, monkeypatch):
    _lock_path, _lock, validation_kwargs = _opening_registry_lock_validation_fixture(
        tmp_path, monkeypatch
    )

    registries = opening_module.validate_registry_lock(**validation_kwargs)

    assert registries["external"]["site_no"].tolist() == ["01234567"]


def test_opening_rejects_registry_lock_extra_top_level_key(tmp_path, monkeypatch):
    lock_path, lock, validation_kwargs = _opening_registry_lock_validation_fixture(
        tmp_path, monkeypatch
    )
    lock["unbound_extra"] = "accepted-by-old-opening-validator"
    lock_path.write_bytes(canonical_json_bytes(lock))

    with pytest.raises(opening_module.OpeningContractError, match="schema changed"):
        opening_module.validate_registry_lock(**validation_kwargs)


def test_opening_rejects_registry_lock_extra_nested_binding_key(
    tmp_path, monkeypatch,
):
    lock_path, lock, validation_kwargs = _opening_registry_lock_validation_fixture(
        tmp_path, monkeypatch
    )
    lock["frozen_artifacts"]["candidate_table"]["extra"] = "unbound"
    lock_path.write_bytes(canonical_json_bytes(lock))

    with pytest.raises(
        opening_module.OpeningContractError, match="binding schema changed"
    ):
        opening_module.validate_registry_lock(**validation_kwargs)


def test_opening_rejects_noncanonical_registry_lock_bytes(tmp_path, monkeypatch):
    lock_path, _lock, validation_kwargs = _opening_registry_lock_validation_fixture(
        tmp_path, monkeypatch
    )
    lock_path.write_bytes(lock_path.read_bytes() + b" ")

    with pytest.raises(
        opening_module.OpeningContractError, match="canonical producer JSON"
    ):
        opening_module.validate_registry_lock(**validation_kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", True), ("site_count", True), ("opening_count", False)],
)
def test_opening_rejects_registry_lock_integer_boolean_alias(
    tmp_path, monkeypatch, field, value,
):
    lock_path, lock, validation_kwargs = _opening_registry_lock_validation_fixture(
        tmp_path, monkeypatch
    )
    lock[field] = value
    lock_path.write_bytes(canonical_json_bytes(lock))

    with pytest.raises(
        opening_module.OpeningContractError, match="scalar types changed"
    ):
        opening_module.validate_registry_lock(**validation_kwargs)


def test_opening_rejects_noncanonical_external_registry_csv(tmp_path, monkeypatch):
    lock_path, lock, validation_kwargs = _opening_registry_lock_validation_fixture(
        tmp_path, monkeypatch
    )
    registry_path = Path(validation_kwargs["external_registry"])
    frame = pd.read_csv(registry_path, dtype=str, keep_default_na=False)
    registry_path.write_text(
        frame.to_csv(index=False, lineterminator="\n", quoting=csv.QUOTE_ALL),
        encoding="utf-8",
    )
    lock["confirmatory_registry_sha256"] = sha256_file(registry_path)
    lock_path.write_bytes(canonical_json_bytes(lock))

    with pytest.raises(
        opening_module.OpeningContractError, match="canonical seeded-selection CSV"
    ):
        opening_module.validate_registry_lock(**validation_kwargs)


@pytest.mark.parametrize("field", ["registry_frozen_at_utc", "created_at_utc"])
@pytest.mark.parametrize(
    "value",
    ["2026-07-26T00:00:00Z", "not-a-timestamp"],
)
def test_opening_rejects_noncanonical_or_invalid_registry_lock_timestamp(
    tmp_path, monkeypatch, field, value,
):
    lock_path, lock, validation_kwargs = _opening_registry_lock_validation_fixture(
        tmp_path, monkeypatch
    )
    lock[field] = value
    lock_path.write_bytes(canonical_json_bytes(lock))

    with pytest.raises(
        opening_module.OpeningContractError,
        match=r"canonical UTC timestamp",
    ):
        opening_module.validate_registry_lock(**validation_kwargs)


def test_opening_rejects_1900_registry_timestamp_against_raw_metadata(
    tmp_path, monkeypatch,
):
    lock_path, lock, validation_kwargs = _opening_registry_lock_validation_fixture(
        tmp_path, monkeypatch
    )
    lock["registry_frozen_at_utc"] = "1900-01-01T00:00:00+00:00"
    lock["created_at_utc"] = "1900-01-01T00:00:00+00:00"
    lock_path.write_bytes(canonical_json_bytes(lock))

    with pytest.raises(
        opening_module.OpeningContractError,
        match="raw/registry/lock chronology",
    ):
        opening_module.validate_registry_lock(**validation_kwargs)


def test_opening_rebuilds_candidate_acquisition_session_from_raw_metadata(
    tmp_path, monkeypatch,
):
    lock_path, lock, validation_kwargs = _opening_registry_lock_validation_fixture(
        tmp_path, monkeypatch
    )
    lock["candidate_acquisition_session"] = {
        "maximum_duration_seconds": 86400,
        "retrieved_at_min_utc": "2025-01-01T00:00:00+00:00",
        "retrieved_at_max_utc": "2025-01-01T00:00:00+00:00",
        "clock_source": "LOCAL_SYSTEM_CLOCK_NOT_EXTERNALLY_ATTESTED",
    }
    lock_path.write_bytes(canonical_json_bytes(lock))

    with pytest.raises(
        opening_module.OpeningContractError,
        match="candidate acquisition session changed",
    ):
        opening_module.validate_registry_lock(**validation_kwargs)


def test_opening_rejects_changed_registry_chronology_trust_boundary(
    tmp_path, monkeypatch,
):
    lock_path, lock, validation_kwargs = _opening_registry_lock_validation_fixture(
        tmp_path, monkeypatch
    )
    lock["chronology_trust_boundary"] = "EXTERNALLY_ATTESTED"
    lock_path.write_bytes(canonical_json_bytes(lock))

    with pytest.raises(opening_module.OpeningContractError, match="lock mismatch"):
        opening_module.validate_registry_lock(**validation_kwargs)


@pytest.mark.parametrize(
    ("boundary", "published_count"),
    [("table", 1), ("index", 2), ("provenance", 3)],
)
def test_candidate_discovery_resumes_each_derived_publication_boundary(
    tmp_path, monkeypatch, boundary, published_count,
):
    discovery, _holdout, args, _holdout_args, paths = (
        _candidate_publication_fixture(tmp_path)
    )
    args.freeze_selection = False
    ordered = [paths["table"], paths["index"], paths["provenance"]]
    original_create = discovery.atomic_create

    def publish_then_interrupt(path, payload):
        original_create(path, payload)
        if path == paths[boundary]:
            raise RuntimeError(f"injected interruption after {boundary}")

    monkeypatch.setattr(discovery, "atomic_create", publish_then_interrupt)
    with pytest.raises(RuntimeError, match="injected interruption"):
        discovery.discover(args)
    assert [path.exists() for path in ordered] == [
        index < published_count for index in range(3)
    ]

    monkeypatch.setattr(discovery, "atomic_create", original_create)
    discovery.discover(args)
    assert all(path.is_file() for path in ordered)
    before = {path: path.read_bytes() for path in ordered}
    discovery.discover(args)
    assert {path: path.read_bytes() for path in ordered} == before


def test_candidate_discovery_recovers_response_only_by_exact_refetch(
    tmp_path, monkeypatch,
):
    discovery, _holdout, args, _holdout_args, paths = (
        _candidate_publication_fixture(tmp_path)
    )
    request_dir = next(
        path for path in args.snapshot_dir.glob(f"{CANDIDATE_PROVIDER}/*")
        if path.is_dir()
    )
    response_path = request_dir / "response.bin"
    metadata_path = request_dir / "metadata.json"
    expected = response_path.read_bytes()
    original_inode = response_path.stat().st_ino
    metadata_path.unlink()
    args.offline = False
    args.freeze_selection = False
    calls = []

    class ExactResponse:
        status = 200
        headers = {"Date": "Sun, 26 Jul 2026 00:00:00 GMT"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            calls.append("read")
            return expected

        def geturl(self):
            return build_usgs_candidate_url("CO")

    monkeypatch.setattr(
        "thermoroute.provenance.urllib.request.urlopen",
        lambda *_args, **_kwargs: ExactResponse(),
    )
    discovery.discover(args)
    assert calls == ["read"]
    assert response_path.read_bytes() == expected
    assert response_path.stat().st_ino == original_inode
    assert metadata_path.is_file()
    recovered_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert recovered_metadata["retrieval_semantics"] == (
        "BYTE_IDENTICAL_REFETCH_COMPLETED_RESPONSE_ONLY_TRANSACTION"
    )
    assert all(paths[name].is_file() for name in ("table", "index", "provenance"))


def test_candidate_discovery_rejects_http_redirect_during_recovery(
    tmp_path, monkeypatch,
):
    discovery, _holdout, args, _holdout_args, paths = (
        _candidate_publication_fixture(tmp_path)
    )
    request_dir = next(args.snapshot_dir.glob(f"{CANDIDATE_PROVIDER}/*"))
    response_path = request_dir / "response.bin"
    metadata_path = request_dir / "metadata.json"
    expected = response_path.read_bytes()
    metadata_path.unlink()
    args.offline = False
    args.freeze_selection = False

    class RedirectedResponse:
        status = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return expected

        def geturl(self):
            return "https://attacker.example/redirected"

    monkeypatch.setattr(
        "thermoroute.provenance.urllib.request.urlopen",
        lambda *_args, **_kwargs: RedirectedResponse(),
    )
    with pytest.raises(RuntimeError, match="failed to acquire"):
        discovery.discover(args)
    assert response_path.read_bytes() == expected
    assert not metadata_path.exists()
    assert not any(paths[name].exists() for name in ("table", "index", "provenance"))


@pytest.mark.parametrize("boundary", ["registry", "lock"])
def test_holdout_resumes_each_registry_publication_boundary(
    tmp_path, monkeypatch, boundary,
):
    discovery, holdout, args, holdout_args, paths = (
        _candidate_publication_fixture(tmp_path)
    )
    args.freeze_selection = False
    discovery.discover(args)
    original_write = holdout.atomic_write

    def publish_then_interrupt(path, payload):
        original_write(path, payload)
        if path == paths[boundary]:
            raise RuntimeError(f"injected interruption after {boundary}")

    monkeypatch.setattr(holdout, "atomic_write", publish_then_interrupt)
    with pytest.raises(RuntimeError, match="injected interruption"):
        holdout.publish_or_validate(holdout_args, check_only=False)
    assert paths["registry"].exists()
    assert paths["lock"].exists() is (boundary == "lock")

    monkeypatch.setattr(holdout, "atomic_write", original_write)
    holdout.publish_or_validate(holdout_args, check_only=False)
    before = {
        paths["registry"]: paths["registry"].read_bytes(),
        paths["lock"]: paths["lock"].read_bytes(),
    }
    holdout.publish_or_validate(holdout_args, check_only=True)
    assert {path: path.read_bytes() for path in before} == before


def test_candidate_check_existing_is_network_free_and_write_free(
    tmp_path, monkeypatch,
):
    discovery, holdout, args, _holdout_args, _paths = (
        _candidate_publication_fixture(tmp_path)
    )
    discovery.discover(args)

    def tree_identity(root: Path) -> dict[str, tuple[int, int, int, int, bytes | None]]:
        identity: dict[str, tuple[int, int, int, int, bytes | None]] = {}
        for path in (root, *sorted(root.rglob("*"))):
            metadata = path.lstat()
            identity[path.relative_to(root).as_posix()] = (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_mtime_ns,
                path.read_bytes() if path.is_file() else None,
            )
        return identity

    before = tree_identity(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("check-existing attempted a network or write operation")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    monkeypatch.setattr(discovery, "atomic_create", forbidden)
    monkeypatch.setattr(holdout, "atomic_write", forbidden)
    args.check_existing = True
    args.offline = True
    discovery.discover(args)
    assert tree_identity(tmp_path) == before


def test_candidate_publication_lock_is_busy_but_accepts_inherited_capability(
    tmp_path,
):
    lock_path = tmp_path / "protocols" / "route_a_confirmatory_v1.json"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_bytes(b"fixed protocol anchor\n")

    with candidate_publication_lock(
        lock_path, trusted_root=tmp_path
    ) as owner:
        # A separately opened file description is a competing publisher even
        # in this process and must fail immediately instead of blocking.
        with pytest.raises(ProvenanceError, match="busy"):
            with candidate_publication_lock(lock_path, trusted_root=tmp_path):
                pass

        # A child receives the parent's existing open-file description through
        # pass_fds.  The helper may duplicate/verify it, but must never unlock
        # the parent's transaction when the borrowed context exits.
        with candidate_publication_lock(
            lock_path,
            trusted_root=tmp_path,
            inherited_fd=owner.fd,
            inherited_token=owner.token,
        ) as inherited:
            assert (
                os.fstat(inherited.fd).st_dev,
                os.fstat(inherited.fd).st_ino,
            ) == (
                os.fstat(owner.fd).st_dev,
                os.fstat(owner.fd).st_ino,
            )

        with pytest.raises(ProvenanceError, match="busy"):
            with candidate_publication_lock(lock_path, trusted_root=tmp_path):
                pass

    # Kernel release after the true owner exits permits the next transaction.
    with candidate_publication_lock(lock_path, trusted_root=tmp_path) as successor:
        assert successor.fd >= 0


def test_candidate_publication_lock_serializes_two_processes(tmp_path):
    lock_path = tmp_path / "protocols" / "route_a_confirmatory_v1.json"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_bytes(b"fixed protocol anchor\n")
    worker = "\n".join([
        "import sys",
        f"sys.path.insert(0, {str(ROOT / 'src')!r})",
        "from thermoroute.provenance import (",
        "    ProvenanceError, candidate_publication_lock,",
        ")",
        "try:",
        "    with candidate_publication_lock(sys.argv[1], trusted_root=sys.argv[2]):",
        "        pass",
        "except ProvenanceError as exc:",
        "    print(str(exc), file=sys.stderr)",
        "    raise SystemExit(23)",
    ])
    command = [sys.executable, "-c", worker, str(lock_path), str(tmp_path)]
    with candidate_publication_lock(lock_path, trusted_root=tmp_path):
        contender = subprocess.run(command, text=True, capture_output=True, check=False)
        assert contender.returncode == 23
        assert "busy" in contender.stderr
    successor = subprocess.run(command, text=True, capture_output=True, check=False)
    assert successor.returncode == 0, successor.stderr


@pytest.mark.parametrize(
    "artifact", ["table", "index", "provenance", "registry", "lock"]
)
def test_candidate_resume_rejects_tampered_publication(
    tmp_path, artifact,
):
    discovery, _holdout, args, _holdout_args, paths = (
        _candidate_publication_fixture(tmp_path)
    )
    discovery.discover(args)
    target = paths[artifact]
    target.write_bytes(target.read_bytes() + b"\n")
    args.check_existing = True
    with pytest.raises(RuntimeError):
        discovery.discover(args)


@pytest.mark.parametrize(
    ("artifact", "link_kind"),
    [("table", "symlink"), ("registry", "symlink"),
     ("table", "hardlink"), ("registry", "hardlink")],
)
def test_candidate_resume_rejects_symlink_and_hardlink_publications(
    tmp_path, artifact, link_kind,
):
    discovery, _holdout, args, _holdout_args, paths = (
        _candidate_publication_fixture(tmp_path)
    )
    discovery.discover(args)
    target = paths[artifact]
    alias = tmp_path / f"{artifact}-{link_kind}-alias"
    if link_kind == "symlink":
        alias.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(alias)
    else:
        os.link(target, alias)
    args.check_existing = True
    with pytest.raises(RuntimeError, match="regular single-link"):
        discovery.discover(args)


def test_candidate_resume_rejects_extra_snapshot_and_partial_temp(
    tmp_path,
):
    discovery, _holdout, args, _holdout_args, paths = (
        _candidate_publication_fixture(tmp_path)
    )
    discovery.discover(args)
    extra = args.snapshot_dir / "unexpected.bin"
    extra.write_bytes(b"unexpected")
    args.check_existing = True
    with pytest.raises(RuntimeError, match="extra nodes"):
        discovery.discover(args)

    extra.unlink()
    paths["provenance"].unlink()
    paths["registry"].unlink()
    paths["lock"].unlink()
    partial = paths["provenance"].with_name(
        f".{paths['provenance'].name}.interrupted.tmp"
    )
    partial.write_bytes(b"{")
    args.check_existing = False
    args.freeze_selection = False
    with pytest.raises(RuntimeError, match="temporary evidence"):
        discovery.discover(args)


def _rewrite_candidate_index_and_rebind_provenance(
    paths: dict[str, Path],
    index: dict[str, object],
) -> dict[str, object]:
    """Publish a self-consistently re-signed adversarial index/provenance pair."""
    paths["index"].write_bytes(canonical_json_bytes(index))
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    provenance["raw_snapshot_index_sha256"] = sha256_file(paths["index"])
    paths["provenance"].write_bytes(canonical_json_bytes(provenance))
    return provenance


@pytest.mark.parametrize(
    "attack",
    ["state_universe_rule", "candidate_rule", "raw_snapshot_index", "quote_all_csv"],
)
def test_candidate_replay_rejects_self_consistent_noncanonical_provenance_or_csv(
    tmp_path, attack,
):
    discovery, holdout, args, holdout_args, paths = (
        _candidate_publication_fixture(tmp_path)
    )
    discovery.discover(args)
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    if attack == "quote_all_csv":
        frame = pd.read_csv(paths["table"], dtype=str, keep_default_na=False)
        payload = frame.to_csv(
            index=False,
            float_format="%.17g",
            lineterminator="\n",
            quoting=csv.QUOTE_ALL,
        ).encode("utf-8")
        paths["table"].write_bytes(payload)
        provenance["candidate_table_sha256"] = sha256_bytes(payload)
    else:
        provenance[attack] = f"FORGED_{attack}"
    paths["provenance"].write_bytes(canonical_json_bytes(provenance))

    with pytest.raises(
        EvidenceError,
        match=r"(?i)(identity|flags|snapshot index|canonical producer CSV)",
    ):
        holdout.verify_candidate_evidence(
            holdout_args.candidates,
            holdout_args.candidate_provenance,
            holdout_args.candidate_snapshot_index,
            holdout_args.protocol,
        )


def test_candidate_replay_rejects_reordered_two_state_request_registry(tmp_path):
    discovery, holdout, args, holdout_args, paths = (
        _candidate_publication_fixture(tmp_path)
    )
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    protocol["metadata_candidate_contract"]["state_universe"] = ["CO", "OR"]
    args.protocol.write_bytes(canonical_json_bytes(protocol))
    discovery.ROUTE_A_STATE_UNIVERSE = ("CO", "OR")
    args.states = ["CO", "OR"]

    payload = _candidate_payload("09876543")
    request = canonical_candidate_request("OR")
    request_sha = sha256_bytes(canonical_json_bytes(request))
    transaction = args.snapshot_dir / CANDIDATE_PROVIDER / request_sha
    transaction.mkdir(parents=True)
    response_path = transaction / "response.bin"
    metadata_path = transaction / "metadata.json"
    response_path.write_bytes(payload)
    metadata_path.write_bytes(canonical_json_bytes({
        "schema_version": 2,
        "request": request,
        "request_sha256": request_sha,
        "retrieved_at_utc": "2026-01-01T00:00:00+00:00",
        "http_status": 200,
        "response_headers": {},
        "byte_count": len(payload),
        "response_sha256": sha256_bytes(payload),
        "response_file": "response.bin",
        "final_url": request["url"],
        "retrieval_semantics": "DIRECT_HTTP_RESPONSE",
    }))
    discovery.discover(args)
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    provenance["requests"].reverse()
    paths["provenance"].write_bytes(canonical_json_bytes(provenance))

    with pytest.raises(EvidenceError, match=r"canonical state order"):
        holdout.verify_candidate_evidence(
            holdout_args.candidates,
            holdout_args.candidate_provenance,
            holdout_args.candidate_snapshot_index,
            holdout_args.protocol,
        )


def test_holdout_rejects_noncanonical_utc_after_all_timestamp_bindings_are_updated(
    tmp_path,
):
    _discovery, holdout, args, holdout_args, paths = (
        _candidate_publication_fixture(tmp_path)
    )
    _discovery.discover(args)

    # ``Z`` denotes UTC and Python accepts it, but it is not the producer's one
    # canonical ``+00:00`` representation.  Rebind all three timestamp-bearing
    # layers so a checksum-only verifier cannot detect the attack.
    forged_timestamp = "2026-01-01T00:00:00Z"
    index = json.loads(paths["index"].read_text(encoding="utf-8"))
    record = index["records"][0]
    metadata_path = args.snapshot_dir / record["metadata_path"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["retrieved_at_utc"] = forged_timestamp
    metadata_path.write_bytes(canonical_json_bytes(metadata))
    record["retrieved_at_utc"] = forged_timestamp
    provenance = _rewrite_candidate_index_and_rebind_provenance(paths, index)
    provenance["requests"][0]["retrieved_at_utc"] = forged_timestamp
    paths["provenance"].write_bytes(canonical_json_bytes(provenance))

    with pytest.raises(RuntimeError, match=r"(?i)(canonical.*utc|utc.*canonical|timestamp)"):
        holdout.verify_candidate_evidence(
            holdout_args.candidates,
            holdout_args.candidate_provenance,
            holdout_args.candidate_snapshot_index,
            holdout_args.protocol,
        )


@pytest.mark.parametrize("field", ["registry_frozen_at_utc", "created_at_utc"])
def test_holdout_rejects_noncanonical_registry_lock_timestamp(tmp_path, field):
    discovery, _holdout, args, _holdout_args, paths = (
        _candidate_publication_fixture(tmp_path)
    )
    discovery.discover(args)
    lock = json.loads(paths["lock"].read_text(encoding="utf-8"))
    lock[field] = "2026-07-26T00:00:00Z"
    paths["lock"].write_bytes(canonical_json_bytes(lock))
    args.check_existing = True
    with pytest.raises(RuntimeError, match=r"(?i)(canonical.*utc|timestamp)"):
        discovery.discover(args)


@pytest.mark.parametrize("mutation", ["wrong_count", "extra_duplicate_record"])
def test_holdout_rejects_nonexact_snapshot_index_after_provenance_is_rebound(
    tmp_path,
    mutation,
):
    _discovery, holdout, args, holdout_args, paths = (
        _candidate_publication_fixture(tmp_path)
    )
    _discovery.discover(args)
    index = json.loads(paths["index"].read_text(encoding="utf-8"))
    if mutation == "wrong_count":
        index["snapshot_count"] = 2
    else:
        index["records"].append(dict(index["records"][0]))
        index["snapshot_count"] = 2
    _rewrite_candidate_index_and_rebind_provenance(paths, index)

    with pytest.raises(RuntimeError, match=r"(?i)(snapshot|index|count|duplicate)"):
        holdout.verify_candidate_evidence(
            holdout_args.candidates,
            holdout_args.candidate_provenance,
            holdout_args.candidate_snapshot_index,
            holdout_args.protocol,
        )


@pytest.mark.parametrize("attack", ["response_symlink", "extra_root_node"])
def test_direct_holdout_rejects_unsafe_or_nonexact_raw_snapshot_tree(
    tmp_path,
    attack,
):
    _discovery, holdout, args, holdout_args, paths = (
        _candidate_publication_fixture(tmp_path)
    )
    _discovery.discover(args)
    index = json.loads(paths["index"].read_text(encoding="utf-8"))
    response_path = args.snapshot_dir / index["records"][0]["response_path"]
    if attack == "response_symlink":
        alias = tmp_path / "candidate-response-alias.bin"
        alias.write_bytes(response_path.read_bytes())
        response_path.unlink()
        response_path.symlink_to(alias)
    else:
        (args.snapshot_dir / "unexpected-direct-holdout-node.bin").write_bytes(
            b"not part of the frozen raw namespace"
        )

    with pytest.raises(RuntimeError, match=r"(?i)(snapshot|extra|unsafe|link|regular)"):
        holdout.verify_candidate_evidence(
            holdout_args.candidates,
            holdout_args.candidate_provenance,
            holdout_args.candidate_snapshot_index,
            holdout_args.protocol,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "response_headers", "retrieved_at", "final_url", "extra_key",
        "whitespace", "duplicate_key",
    ],
)
def test_candidate_index_binds_exact_metadata_bytes(tmp_path, mutation):
    discovery, holdout, args, holdout_args, paths = (
        _candidate_publication_fixture(tmp_path)
    )
    discovery.discover(args)
    index = json.loads(paths["index"].read_text(encoding="utf-8"))
    record = index["records"][0]
    metadata_path = args.snapshot_dir / record["metadata_path"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert record["metadata_sha256"] == sha256_file(metadata_path)
    assert record["metadata_byte_count"] == metadata_path.stat().st_size

    if mutation == "response_headers":
        metadata["response_headers"]["X-Adversarial"] = "changed"
        tampered = canonical_json_bytes(metadata)
    elif mutation == "retrieved_at":
        metadata["retrieved_at_utc"] = "2026-01-01T00:00:01+00:00"
        tampered = canonical_json_bytes(metadata)
    elif mutation == "final_url":
        metadata["final_url"] = "https://attacker.example/redirected"
        tampered = canonical_json_bytes(metadata)
    elif mutation == "extra_key":
        metadata["unbound_extra"] = "changed"
        tampered = canonical_json_bytes(metadata)
    elif mutation == "whitespace":
        tampered = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()
    else:
        canonical = canonical_json_bytes(metadata)
        tampered = b'{"schema_version":2,' + canonical[1:]
    metadata_path.write_bytes(tampered)

    with pytest.raises(RuntimeError, match=r"(?i)(metadata|snapshot|canonical|exact)"):
        holdout.verify_candidate_evidence(
            holdout_args.candidates,
            holdout_args.candidate_provenance,
            holdout_args.candidate_snapshot_index,
            holdout_args.protocol,
        )


def test_candidate_replay_rejects_resigned_nonstr_header_value(tmp_path):
    discovery, holdout, args, holdout_args, paths = (
        _candidate_publication_fixture(tmp_path)
    )
    discovery.discover(args)
    index = json.loads(paths["index"].read_text(encoding="utf-8"))
    record = index["records"][0]
    metadata_path = args.snapshot_dir / record["metadata_path"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["response_headers"] = {"X-Adversarial": 1}
    metadata_path.write_bytes(canonical_json_bytes(metadata))
    record["metadata_sha256"] = sha256_file(metadata_path)
    record["metadata_byte_count"] = metadata_path.stat().st_size
    _rewrite_candidate_index_and_rebind_provenance(paths, index)

    with pytest.raises(
        EvidenceError, match=r"(?i)(response/metadata binding|header)"
    ):
        holdout.verify_candidate_evidence(
            holdout_args.candidates,
            holdout_args.candidate_provenance,
            holdout_args.candidate_snapshot_index,
            holdout_args.protocol,
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), True),
        (("candidate_count",), True),
        (("outcome_endpoint_requested",), 0),
        (("requests", 0, "candidate_count"), True),
        (("requests", 0, "byte_count"), True),
    ],
)
def test_candidate_replay_rejects_json_bool_integer_aliases(
    tmp_path, path, value,
):
    discovery, holdout, args, holdout_args, paths = (
        _candidate_publication_fixture(tmp_path)
    )
    discovery.discover(args)
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    target = provenance
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    paths["provenance"].write_bytes(canonical_json_bytes(provenance))

    with pytest.raises(EvidenceError, match=r"scalar types changed"):
        holdout.verify_candidate_evidence(
            holdout_args.candidates,
            holdout_args.candidate_provenance,
            holdout_args.candidate_snapshot_index,
            holdout_args.protocol,
        )


def test_holdout_freezer_replays_raw_candidate_evidence(tmp_path):
    module = _load_script(
        "confirmatory_holdout_evidence_test",
        "scripts/data_usgs/confirmatory_holdout.py",
    )
    payload = _candidate_payload().replace(
        b"\t42.5\n", b"\t4.6127834116537587e-10\n"
    )
    candidates = parse_usgs_candidate_metadata(payload, state="CO")
    candidate_path = tmp_path / "candidates.csv"
    candidate_path.write_text(
        candidates.to_csv(
            index=False,
            float_format="%.17g",
            lineterminator="\n",
        )
    )

    request = {
        "schema_version": 1,
        "provider": CANDIDATE_PROVIDER,
        "method": "GET",
        "url": build_usgs_candidate_url("CO"),
        "headers": {"User-Agent": CANDIDATE_USER_AGENT},
    }
    request_sha = sha256_bytes(canonical_json_bytes(request))
    response_sha = sha256_bytes(payload)
    retrieved_at = "2026-01-01T00:00:00+00:00"
    snapshot_root = tmp_path / "raw"
    transaction = snapshot_root / CANDIDATE_PROVIDER / request_sha
    transaction.mkdir(parents=True)
    response_path = transaction / "response.bin"
    metadata_path = transaction / "metadata.json"
    response_path.write_bytes(payload)
    metadata_path.write_bytes(canonical_json_bytes({
        "schema_version": 2,
        "request": request,
        "request_sha256": request_sha,
        "retrieved_at_utc": retrieved_at,
        "http_status": 200,
        "response_headers": {},
        "byte_count": len(payload),
        "response_sha256": response_sha,
        "response_file": "response.bin",
        "final_url": request["url"],
        "retrieval_semantics": "DIRECT_HTTP_RESPONSE",
    }))
    index_path = snapshot_root / "snapshot_index.json"
    index_path.write_bytes(canonical_json_bytes({
        "schema_version": 2,
        "snapshot_count": 1,
        "records": [{
            "provider": CANDIDATE_PROVIDER,
            "request_sha256": request_sha,
            "response_sha256": response_sha,
            "retrieved_at_utc": retrieved_at,
            "byte_count": len(payload),
            "metadata_sha256": sha256_file(metadata_path),
            "metadata_byte_count": metadata_path.stat().st_size,
            "request": request,
            "response_path": response_path.relative_to(snapshot_root).as_posix(),
            "metadata_path": metadata_path.relative_to(snapshot_root).as_posix(),
        }],
    }))
    provenance_path = tmp_path / "candidates.provenance.json"
    provenance_path.write_bytes(canonical_json_bytes({
        "schema_version": 1,
        "artifact_role": "PRE_LABEL_METADATA_ONLY_CANDIDATE_UNIVERSE",
        "protocol_sha256": "a" * 64,
        "state_universe": ["CO"],
        "state_universe_rule": CANDIDATE_STATE_UNIVERSE_RULE,
        "candidate_rule": CANDIDATE_SELECTION_RULE,
        "candidate_count": 1,
        "site_primary_key": "site_no",
        "sort_order": ["site_no", "state"],
        "columns": list(CANDIDATE_COLUMNS),
        "candidate_table_sha256": sha256_file(candidate_path),
        "raw_snapshot_index": index_path.relative_to(tmp_path).as_posix(),
        "raw_snapshot_index_sha256": sha256_file(index_path),
        "outcome_endpoint_requested": False,
        "outcome_values_requested": False,
        "holdout_coverage_requested_or_computed": False,
        "requests": [{
            "state": "CO",
            "candidate_count": 1,
            "request_sha256": request_sha,
            "response_sha256": response_sha,
            "retrieved_at_utc": retrieved_at,
            "byte_count": len(payload),
        }],
    }))
    rebuilt = module.verify_candidate_evidence(
        candidate_path, provenance_path, index_path
    )
    pd.testing.assert_frame_equal(rebuilt, candidates, check_dtype=False)

    candidate_path.write_text(candidate_path.read_text() + "\n")
    with pytest.raises(RuntimeError, match="checksum"):
        module.verify_candidate_evidence(candidate_path, provenance_path, index_path)


def test_candidate_evidence_replay_roundtrips_pathological_decimal(
    tmp_path, monkeypatch,
):
    payload = _candidate_payload().replace(
        b"\t42.5\n", b"\t4.6127834116537587e-10\n"
    )
    candidates = parse_usgs_candidate_metadata(payload, state="CO")
    candidate_path = tmp_path / "candidates.csv"
    candidate_path.write_text(
        candidates.to_csv(
            index=False,
            float_format="%.17g",
            lineterminator="\n",
        ),
        encoding="utf-8",
    )

    request = {
        "schema_version": 1,
        "provider": CANDIDATE_PROVIDER,
        "method": "GET",
        "url": build_usgs_candidate_url("CO"),
        "headers": {"User-Agent": CANDIDATE_USER_AGENT},
    }
    request_sha = sha256_bytes(canonical_json_bytes(request))
    response_sha = sha256_bytes(payload)
    retrieved_at = "2026-01-01T00:00:00+00:00"
    snapshot_root = tmp_path / "raw"
    snapshot_dir = snapshot_root / CANDIDATE_PROVIDER / request_sha
    snapshot_dir.mkdir(parents=True)
    response_path = snapshot_dir / "response.bin"
    metadata_path = snapshot_dir / "metadata.json"
    response_path.write_bytes(payload)
    metadata_path.write_bytes(canonical_json_bytes({
        "schema_version": 2,
        "request": request,
        "request_sha256": request_sha,
        "http_status": 200,
        "response_headers": {},
        "byte_count": len(payload),
        "response_sha256": response_sha,
        "response_file": "response.bin",
        "retrieved_at_utc": retrieved_at,
        "final_url": request["url"],
        "retrieval_semantics": "DIRECT_HTTP_RESPONSE",
    }))
    index_path = snapshot_root / "snapshot_index.json"
    index_path.write_bytes(canonical_json_bytes({
        "schema_version": 2,
        "snapshot_count": 1,
        "records": [{
            "provider": CANDIDATE_PROVIDER,
            "request_sha256": request_sha,
            "response_sha256": response_sha,
            "retrieved_at_utc": retrieved_at,
            "byte_count": len(payload),
            "metadata_sha256": sha256_file(metadata_path),
            "metadata_byte_count": metadata_path.stat().st_size,
            "request": request,
            "metadata_path": metadata_path.relative_to(snapshot_root).as_posix(),
            "response_path": response_path.relative_to(snapshot_root).as_posix(),
        }],
    }))
    protocol_path = tmp_path / "protocols" / "route_a_confirmatory_v1.json"
    protocol_path.parent.mkdir(parents=True)
    selection_seed = "route-a-confirmatory-v1-public-seed"
    protocol_path.write_bytes(canonical_json_bytes({
        "schema_version": 1,
        "status": "PLANNED_NOT_ACQUIRED",
        "protocol_id": "route-a-fixture",
        "authoritative_protocol_commit": "b" * 40,
        "pre_label_amendments": [],
        "new_site_external_validation": {
            "status": "PLANNED_NOT_ACQUIRED",
            "planned_site_count": 1,
            "selection_seed": selection_seed,
        },
        "metadata_candidate_contract": {"state_universe": ["CO"]},
        "time_holdout": {"start": "2021-01-01", "end": "2023-12-31"},
    }))
    protocol_sha = sha256_file(protocol_path)
    provenance_path = tmp_path / "candidates.provenance.json"
    provenance_path.write_bytes(canonical_json_bytes({
        "schema_version": 1,
        "artifact_role": "PRE_LABEL_METADATA_ONLY_CANDIDATE_UNIVERSE",
        "protocol_sha256": protocol_sha,
        "state_universe": ["CO"],
        "state_universe_rule": CANDIDATE_STATE_UNIVERSE_RULE,
        "candidate_rule": CANDIDATE_SELECTION_RULE,
        "candidate_count": 1,
        "site_primary_key": "site_no",
        "sort_order": ["site_no", "state"],
        "columns": list(CANDIDATE_COLUMNS),
        "outcome_endpoint_requested": False,
        "outcome_values_requested": False,
        "holdout_coverage_requested_or_computed": False,
        "raw_snapshot_index": index_path.relative_to(tmp_path).as_posix(),
        "raw_snapshot_index_sha256": sha256_file(index_path),
        "candidate_table_sha256": sha256_file(candidate_path),
        "requests": [{
            "state": "CO",
            "candidate_count": 1,
            "request_sha256": request_sha,
            "response_sha256": response_sha,
            "retrieved_at_utc": retrieved_at,
            "byte_count": len(payload),
        }],
    }))

    replayed = replay_candidate_evidence(
        candidate_path,
        provenance_path,
        index_path,
        protocol_sha256=protocol_sha,
        state_universe=("CO",),
    )

    pd.testing.assert_frame_equal(
        replayed, candidates, check_dtype=False, rtol=0.0, atol=0.0
    )

    tampered = candidates.copy()
    tampered.loc[0, "drain_area_va"] = np.nextafter(
        float(tampered.loc[0, "drain_area_va"]), np.inf
    )
    tampered_path = tmp_path / "candidates-adjacent.csv"
    tampered_path.write_text(
        tampered.to_csv(
            index=False,
            float_format="%.17g",
            lineterminator="\n",
        ),
        encoding="utf-8",
    )
    tampered_provenance = json.loads(
        provenance_path.read_text(encoding="utf-8")
    )
    tampered_provenance["candidate_table_sha256"] = sha256_file(tampered_path)
    tampered_provenance_path = tmp_path / "candidates-adjacent.provenance.json"
    tampered_provenance_path.write_bytes(
        canonical_json_bytes(tampered_provenance)
    )
    with pytest.raises(
        EvidenceError, match=r"(?i)(canonical producer CSV|cannot be replayed)"
    ):
        replay_candidate_evidence(
            tampered_path,
            tampered_provenance_path,
            index_path,
            protocol_sha256=protocol_sha,
            state_universe=("CO",),
        )

    development_registry = tmp_path / "development.csv"
    development = pd.DataFrame({
        "site_no": [f"{70_000_000 + index:08d}" for index in range(120)],
        "legacy_site_id": [f"n{index + 1:03d}" for index in range(120)],
        "station_nm": [f"Development River {index + 1}" for index in range(120)],
        "lat": [39.5 + index / 1000 for index in range(120)],
        "lon": [-104.5 - index / 1000 for index in range(120)],
        "state": ["CO"] * 120,
        "huc_cd": ["10190005"] * 120,
        "huc2": ["10"] * 120,
        "huc_metadata_status": ["USGS_SNAPSHOT_SITE_NO_MATCH"] * 120,
    })
    development_registry.write_text(
        development.to_csv(
            index=False,
            float_format="%.17g",
            lineterminator="\n",
        ),
        encoding="utf-8",
    )
    source_metadata = tmp_path / "development_source.csv"
    source_metadata.write_text("fixture\n", encoding="utf-8")
    development_spec = tmp_path / "frozen_panel_v1.json"
    development_spec.write_bytes(canonical_json_bytes({
        "schema_version": 1,
        "station_registry": {
            "path": development_registry.name,
            "sha256": sha256_file(development_registry),
            "source_metadata_path": source_metadata.name,
            "source_metadata_sha256": sha256_file(source_metadata),
            "station_count": 120,
        },
    }))
    out_registry = tmp_path / "external_registry.csv"
    out_lock = tmp_path / "external_lock.json"
    holdout = _load_script(
        "confirmatory_holdout_freeze_roundtrip_test",
        "scripts/data_usgs/confirmatory_holdout.py",
    )
    holdout.ROOT = tmp_path
    rejected_registry = tmp_path / "rejected_external_registry.csv"
    rejected_lock = tmp_path / "rejected_external_lock.json"
    with pytest.raises(RuntimeError, match="cannot be rebuilt"):
        holdout.freeze(SimpleNamespace(
            protocol=protocol_path,
            development_spec=development_spec,
            candidates=tampered_path,
            candidate_snapshot_index=index_path,
            candidate_provenance=tampered_provenance_path,
            out_registry=rejected_registry,
            out_lock=rejected_lock,
            n_sites=1,
            selection_seed=selection_seed,
        ))
    assert not rejected_registry.exists()
    assert not rejected_lock.exists()

    holdout.freeze(SimpleNamespace(
        protocol=protocol_path,
        development_spec=development_spec,
        candidates=candidate_path,
        candidate_snapshot_index=index_path,
        candidate_provenance=provenance_path,
        out_registry=out_registry,
        out_lock=out_lock,
        n_sites=1,
        selection_seed=selection_seed,
    ))
    frozen = pd.read_csv(
        out_registry,
        dtype={"site_no": "string"},
        float_precision="round_trip",
    )
    np.testing.assert_array_max_ulp(
        frozen["drain_area_va"].to_numpy(float),
        candidates["drain_area_va"].to_numpy(float),
        maxulp=0,
    )
    assert frozen["site_no"].tolist() == ["01234567"]

    frozen_spec = SimpleNamespace(
        registry_path=development_registry.resolve(),
        verify=lambda: {"registry_sha256": sha256_file(development_registry)},
    )
    monkeypatch.setattr(
        opening_module.FrozenPanelSpec,
        "load",
        classmethod(lambda _cls, _path: frozen_spec),
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    lock = json.loads(out_lock.read_text(encoding="utf-8"))
    protocol_info = {
        "document": protocol,
        "protocol_sha256": protocol_sha,
        "authoritative_commit": protocol["authoritative_protocol_commit"],
        "amendments_sha256": lock["pre_label_amendments_sha256"],
    }
    registries = opening_module.validate_registry_lock(
        root=tmp_path,
        protocol_info=protocol_info,
        development_registry=development_registry,
        external_registry=out_registry,
        external_lock=out_lock,
    )
    np.testing.assert_array_max_ulp(
        registries["external"]["drain_area_va"].to_numpy(float),
        candidates["drain_area_va"].to_numpy(float),
        maxulp=0,
    )

    adjacent = frozen.copy()
    adjacent.loc[0, "drain_area_va"] = np.nextafter(
        float(adjacent.loc[0, "drain_area_va"]), np.inf
    )
    out_registry.write_text(
        adjacent.to_csv(
            index=False,
            float_format="%.17g",
            lineterminator="\n",
        ),
        encoding="utf-8",
    )
    lock["confirmatory_registry_sha256"] = sha256_file(out_registry)
    out_lock.write_bytes(canonical_json_bytes(lock))
    with pytest.raises(
        opening_module.OpeningContractError,
        match=(
            r"(?i)(deterministic seeded candidate selection|"
            r"canonical seeded-selection CSV)"
        ),
    ):
        opening_module.validate_registry_lock(
            root=tmp_path,
            protocol_info=protocol_info,
            development_registry=development_registry,
            external_registry=out_registry,
            external_lock=out_lock,
        )


def test_previous_runs_url_freezes_model_variable_leads_and_timezone():
    url = build_previous_runs_url(
        latitude=40.0,
        longitude=-105.0,
        start_date="2021-04-01",
        end_date="2021-04-30",
    )
    query = parse_qs(urlsplit(url).query)
    assert query["models"] == ["gfs_global"]
    assert query["timezone"] == ["GMT"]
    assert query["temperature_unit"] == ["celsius"]
    assert query["hourly"] == [
        "temperature_2m_previous_day1,temperature_2m_previous_day3,"
        "temperature_2m_previous_day7"
    ]
    assert query["latitude"] == ["40.000000"]
    assert query["longitude"] == ["-105.000000"]


def test_previous_runs_parser_derives_daily_issue_and_target_keys():
    daily = parse_previous_runs_daily(
        _nwp_payload(),
        site_no="01234567",
        requested_start="2021-04-01",
        requested_end="2021-04-02",
    )
    assert len(daily) == 6
    assert daily["complete_target_day"].all()
    lead3 = daily[(daily["horizon"] == 3) & (daily["target_date"] == "2021-04-01")]
    assert len(lead3) == 1
    assert lead3.iloc[0]["issue_date"] == pd.Timestamp("2021-03-29")
    assert lead3.iloc[0]["air_temp_2m_mean_c"] == pytest.approx(14.5)
    assert lead3.iloc[0]["issue_semantics"] == ISSUE_SEMANTICS
    assert not any("wtemp" in column.lower() for column in daily.columns)


def test_previous_runs_parser_retains_but_does_not_average_partial_days():
    daily = parse_previous_runs_daily(
        _nwp_payload(missing=(7, 5)),
        site_no="01234567",
        requested_start="2021-04-01",
        requested_end="2021-04-02",
    )
    row = daily[(daily["horizon"] == 7) & (daily["target_date"] == "2021-04-01")].iloc[0]
    assert row["available_hour_count"] == 23
    assert not row["complete_target_day"]
    assert np.isnan(row["air_temp_2m_mean_c"])


def test_previous_runs_parser_rejects_wrong_grid_or_value_contract():
    document = json.loads(_nwp_payload())
    document["hourly"]["temperature_2m_previous_day1"][0] = "not-a-number"
    with pytest.raises(NWPContractError, match="non-numeric"):
        parse_previous_runs_daily(
            json.dumps(document).encode(),
            site_no="01234567",
            requested_start="2021-04-01",
            requested_end="2021-04-02",
        )

    document = json.loads(_nwp_payload())
    document["timezone"] = "America/Denver"
    with pytest.raises(NWPContractError, match="timezone"):
        parse_previous_runs_daily(
            json.dumps(document).encode(),
            site_no="01234567",
            requested_start="2021-04-01",
            requested_end="2021-04-02",
        )


def test_month_chunks_and_prelabel_availability_date_are_frozen():
    assert NWP_COMMON_VALID_TIME_START == date(2021, 3, 30)
    chunks = list(iter_month_chunks("2021-03-30", "2021-05-02"))
    assert chunks == [
        (date(2021, 3, 30), date(2021, 3, 31)),
        (date(2021, 4, 1), date(2021, 4, 30)),
        (date(2021, 5, 1), date(2021, 5, 2)),
    ]


def test_candidate_state_universe_matches_frozen_development_support():
    registry = pd.read_csv(ROOT / "data_usgs" / "station_registry_v1.csv")
    assert tuple(sorted(registry["state"].unique())) == ROUTE_A_STATE_UNIVERSE


def test_machine_protocol_records_prelabel_input_amendment():
    protocol = json.loads(
        (ROOT / "protocols" / "route_a_confirmatory_v1.json").read_text()
    )
    assert protocol["time_holdout"]["primary_target_start"] == "2021-01-01"
    assert protocol["time_holdout"]["secondary_nwp_common_lead_target_start"] == (
        "2021-03-30"
    )
    nwp = protocol["secondary_archived_nwp_contract"]
    assert nwp["contains_outcome_labels"] is False
    assert nwp["consumed_by_primary_models"] is False
    assert nwp["lead_days"] == [1, 3, 7]
    assert protocol["primary_historical_input_contract"][
        "horizon_specific_future_nwp_consumed"
    ] is False
    inference = protocol["primary_inference_contract"]
    assert inference["primary_effect"] == (
        "median of paired station-level RMSE differences"
    )
    assert inference["confidence_interval"]["method"] == (
        "whole-HUC2 cluster percentile bootstrap"
    )
    assert inference["one_sided_p_value"]["method"] == (
        "exact whole-HUC2 cluster sign-flip enumeration"
    )
    probability = inference["probabilistic_event_contract"]
    assert probability["event_reference_climatology"]["fit_interval"] == [
        "2006-01-01", "2018-12-31"
    ]
    assert "confirmation event rate" in probability[
        "event_reference_climatology"
    ]["brier_skill"]
    assert inference["one_sided_p_value"][
        "maximum_configurations_for_frozen_cohort"
    ] == 32768
    assert inference["one_sided_p_value"]["monte_carlo_correction"] == (
        "not applicable to exact enumeration"
    )
    assert len(inference["confirmatory_family"]) == 5
    assert inference["multiplicity"].startswith("Holm step-down")
    amendment = protocol["pre_label_amendments"][0]
    assert amendment["post_2020_wtemp_requested_or_inspected"] is False
    assert amendment["outcome_independent"] is True

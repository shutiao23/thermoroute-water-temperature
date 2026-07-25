from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

import thermoroute.input_closure as input_closure
from thermoroute.input_closure import (
    BRIDGE_MANIFEST_PATH,
    BRIDGE_RAW_INDEXES,
    FROZEN_SPEC_PATH,
    INPUT_CLOSURE_FORMAT,
    InputClosureError,
    resolve_development_input_closure,
)
from thermoroute.provenance import canonical_json_bytes


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _binding(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha(path.read_bytes()),
    }


def _snapshot_index(
    root: Path,
    index_path: Path,
    *,
    provider: str,
    schema_version: int,
) -> tuple[dict[str, object], bytes]:
    request = {
        "schema_version": 1,
        "provider": provider,
        "method": "GET",
        "url": f"https://example.invalid/{provider}/development-only",
        "headers": {},
    }
    request_sha = _sha(canonical_json_bytes(request))
    response = f"development bytes for {provider}\n".encode()
    response_sha = _sha(response)
    retrieved = "2026-01-01T00:00:00+00:00"
    metadata = canonical_json_bytes({
        "schema_version": 1,
        "request": request,
        "request_sha256": request_sha,
        "retrieved_at_utc": retrieved,
        "http_status": 200,
        "response_headers": {"content-type": "application/octet-stream"},
        "byte_count": len(response),
        "response_sha256": response_sha,
        "response_file": "response.bin",
    })
    relative_base = Path(provider) / request_sha
    metadata_relative = relative_base / "metadata.json"
    response_relative = relative_base / "response.bin"
    _write(index_path.parent / metadata_relative, metadata)
    _write(index_path.parent / response_relative, response)
    record: dict[str, object] = {
        "provider": provider,
        "request_sha256": request_sha,
        "response_sha256": response_sha,
        "retrieved_at_utc": retrieved,
        "byte_count": len(response),
        "request": request,
        "metadata_path": metadata_relative.as_posix(),
        "response_path": response_relative.as_posix(),
    }
    if schema_version == 2:
        record.update({
            "metadata_sha256": _sha(metadata),
            "metadata_byte_count": len(metadata),
        })
    index = canonical_json_bytes({
        "schema_version": schema_version,
        "snapshot_count": 1,
        "records": [record],
    })
    _write(index_path, index)
    return record, index


def _development_fixture(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data = root / "data_usgs"
    panel = data / "panel_usgs_120v2.parquet"
    registry = data / "station_registry_v1.csv"
    source_metadata = data / "stations_meta_120v2.csv"
    huc_source = data / "huc_metadata_usgs_v1.csv"
    for path, payload in (
        (panel, b"fixture development panel through 2020\n"),
        (registry, b"site_no,legacy_site_id\n1,n00\n"),
        (source_metadata, b"site_no,lat,lon\n1,1.0,2.0\n"),
        (huc_source, b"site_no,huc2\n1,01\n"),
    ):
        _write(path, payload)

    huc_index = data / "raw_snapshots/huc-v1/snapshot_index.json"
    huc_record, huc_index_bytes = _snapshot_index(
        root,
        huc_index,
        provider="usgs-nwis-site-metadata",
        schema_version=1,
    )
    huc_provenance = data / "huc_metadata_usgs_v1.provenance.json"
    _write(huc_provenance, canonical_json_bytes({
        "schema_version": 1,
        "outcome_data_requested": False,
        "join_key": "site_no",
        "site_count": 120,
        "development_panel_sha256": _sha(panel.read_bytes()),
        "development_metadata_sha256": _sha(source_metadata.read_bytes()),
        "derived_csv_sha256": _sha(huc_source.read_bytes()),
        "raw_snapshot_index": huc_index.relative_to(root).as_posix(),
        "raw_snapshot_index_sha256": _sha(huc_index_bytes),
        "request_sha256": huc_record["request_sha256"],
        "response_sha256": huc_record["response_sha256"],
        "retrieved_at_utc": huc_record["retrieved_at_utc"],
    }))
    frozen_spec = data / "frozen_panel_v1.json"
    _write(frozen_spec, canonical_json_bytes({
        "schema_version": 1,
        "evidence_role": "development_exploratory",
        "panel": {
            "path": panel.name,
            "sha256": _sha(panel.read_bytes()),
            "date_start": "2006-01-01",
            "date_end": "2020-12-31",
            "station_count": 120,
        },
        "station_registry": {
            "path": registry.name,
            "sha256": _sha(registry.read_bytes()),
            "source_metadata_path": source_metadata.name,
            "source_metadata_sha256": _sha(source_metadata.read_bytes()),
            "huc_metadata": {
                "runtime_dependency": True,
                "status": "COMPLETE_USGS_RAW_SNAPSHOT",
                "join_key": "site_no",
                "source_path": huc_source.name,
                "source_sha256": _sha(huc_source.read_bytes()),
                "provenance_path": huc_provenance.name,
                "provenance_sha256": _sha(huc_provenance.read_bytes()),
            },
        },
    }))

    compact_indexes = tuple(
        (name, path, provider, 1)
        for name, path, provider, _count in BRIDGE_RAW_INDEXES
    )
    monkeypatch.setattr(input_closure, "BRIDGE_RAW_INDEXES", compact_indexes)
    raw_bindings: dict[str, dict[str, str]] = {}
    for name, relative, provider, _count in compact_indexes:
        index_path = root / relative
        _snapshot_index(
            root, index_path, provider=provider, schema_version=2
        )
        raw_bindings[name] = _binding(root, index_path)

    bridge_root = data / "development_predictor_bridge_v1"
    frozen_table = bridge_root / "frozen_panel_predictors_2018_2020.parquet"
    refreshed_table = bridge_root / "refreshed_predictors_2018_2020.parquet"
    report = bridge_root / "bridge_report_v1.json"
    request_map = bridge_root / "source_request_map_v1.json"
    for path, payload in (
        (frozen_table, b"frozen development predictors\n"),
        (refreshed_table, b"refreshed development predictors\n"),
        (report, b"{}\n"),
        (request_map, b"{}\n"),
    ):
        _write(path, payload)
    bridge_manifest = data / "development_predictor_bridge_v1.json"
    _write(bridge_manifest, canonical_json_bytes({
        "format": "thermoroute.development-predictor-bridge.v1",
        "status": "PASS_EXACT_PRODUCT_BRIDGE",
        "outcome_values_requested_or_read": False,
        "panel": _binding(root, panel),
        "registry": _binding(root, registry),
        "normalized": {
            "frozen": _binding(root, frozen_table),
            "refreshed": _binding(root, refreshed_table),
        },
        "report": _binding(root, report),
        "request_map": _binding(root, request_map),
        "raw_snapshot_indexes": raw_bindings,
    }))
    return root


def _semantic_verifiers() -> tuple[object, object, list[tuple[str, object]]]:
    calls: list[tuple[str, object]] = []

    def frozen(path: Path) -> dict[str, str]:
        calls.append(("frozen", path))
        return {"evidence_role": "development_exploratory"}

    def bridge(**kwargs: object) -> dict[str, str]:
        calls.append(("bridge", kwargs))
        return {"status": "PASS_EXACT_PRODUCT_BRIDGE"}

    return frozen, bridge, calls


def test_resolves_sorted_recursive_immutable_development_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _development_fixture(tmp_path / "repo", monkeypatch)
    frozen, bridge, calls = _semantic_verifiers()
    closure = resolve_development_input_closure(
        root,
        frozen_spec_verifier=frozen,  # type: ignore[arg-type]
        bridge_verifier=bridge,  # type: ignore[arg-type]
    )

    paths = [entry.path for entry in closure.inventory]
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))
    assert all(path.startswith("data_usgs/") for path in paths)
    assert all("confirmatory" not in path and "2021" not in path for path in paths)
    assert FROZEN_SPEC_PATH.as_posix() in paths
    assert BRIDGE_MANIFEST_PATH.as_posix() in paths
    assert any(path.endswith("/metadata.json") for path in paths)
    assert any(path.endswith("/response.bin") for path in paths)
    assert calls == [
        ("frozen", root / FROZEN_SPEC_PATH),
        (
            "bridge",
            {
                "repo_root": root,
                "manifest_path": root / BRIDGE_MANIFEST_PATH,
                "expected_sites": 120,
            },
        ),
    ]
    expected_digest = _sha(canonical_json_bytes({
        "format": INPUT_CLOSURE_FORMAT,
        "inventory": [entry.as_dict() for entry in closure.inventory],
    }))
    assert closure.binding_digest == expected_digest
    assert [row.path for row in closure.stat_snapshot] == paths
    assert all(set(entry.as_dict()) == {"path", "raw_sha256", "bytes"}
               for entry in closure.inventory)
    closure.assert_unchanged()
    with pytest.raises(FrozenInstanceError):
        closure.binding_digest = "0" * 64  # type: ignore[misc]


def test_assert_unchanged_detects_bytes_and_stat_only_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _development_fixture(tmp_path / "repo", monkeypatch)
    frozen, bridge, _calls = _semantic_verifiers()
    closure = resolve_development_input_closure(
        root,
        frozen_spec_verifier=frozen,  # type: ignore[arg-type]
        bridge_verifier=bridge,  # type: ignore[arg-type]
    )
    source = root / "data_usgs/stations_meta_120v2.csv"
    original = source.read_bytes()
    source.write_bytes(b"X" * len(original))
    with pytest.raises(InputClosureError, match="bytes changed"):
        closure.assert_unchanged()

    root = _development_fixture(tmp_path / "repo-stat", monkeypatch)
    closure = resolve_development_input_closure(
        root,
        frozen_spec_verifier=frozen,  # type: ignore[arg-type]
        bridge_verifier=bridge,  # type: ignore[arg-type]
    )
    source = root / "data_usgs/stations_meta_120v2.csv"
    source.chmod(0o600)
    with pytest.raises(InputClosureError, match="stat identity changed"):
        closure.assert_unchanged()


@pytest.mark.parametrize("attack", ("symlink", "parent_symlink", "fifo", "hardlink"))
def test_rejects_link_and_special_file_attacks_before_semantic_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attack: str,
) -> None:
    root = _development_fixture(tmp_path / "repo", monkeypatch)
    source = root / "data_usgs/stations_meta_120v2.csv"
    if attack == "symlink":
        target = source.with_name("source-target.csv")
        source.rename(target)
        source.symlink_to(target.name)
    elif attack == "parent_symlink":
        bridge_dir = root / "data_usgs/development_predictor_bridge_v1"
        target = root / "data_usgs/bridge-target"
        bridge_dir.rename(target)
        bridge_dir.symlink_to(target.name, target_is_directory=True)
    elif attack == "fifo":
        source.unlink()
        os.mkfifo(source)
    else:
        alias = source.with_name("source-alias.csv")
        os.link(source, alias)
    frozen, bridge, calls = _semantic_verifiers()
    with pytest.raises(InputClosureError, match="linked|unsafe|regular|hard-link"):
        resolve_development_input_closure(
            root,
            frozen_spec_verifier=frozen,  # type: ignore[arg-type]
            bridge_verifier=bridge,  # type: ignore[arg-type]
        )
    assert not calls


def test_rejects_escape_without_opening_declared_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _development_fixture(tmp_path / "repo", monkeypatch)
    forbidden = root / "confirmatory_labels_2021_2023.parquet"
    os.mkfifo(forbidden)
    spec_path = root / FROZEN_SPEC_PATH
    spec = json.loads(spec_path.read_text())
    spec["panel"]["path"] = "../confirmatory_labels_2021_2023.parquet"
    spec_path.write_bytes(canonical_json_bytes(spec))
    frozen, bridge, calls = _semantic_verifiers()
    with pytest.raises(InputClosureError, match="escapes|fixed development closure"):
        resolve_development_input_closure(
            root,
            frozen_spec_verifier=frozen,  # type: ignore[arg-type]
            bridge_verifier=bridge,  # type: ignore[arg-type]
        )
    assert not calls
    assert forbidden.is_fifo()


def test_rejects_duplicate_raw_record_and_path_declarations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _development_fixture(tmp_path / "repo", monkeypatch)
    daymet = root / BRIDGE_RAW_INDEXES[0][1]
    index = json.loads(daymet.read_text())
    index["records"].append(dict(index["records"][0]))
    index["snapshot_count"] = 2
    daymet.write_bytes(canonical_json_bytes(index))
    manifest_path = root / BRIDGE_MANIFEST_PATH
    manifest = json.loads(manifest_path.read_text())
    manifest["raw_snapshot_indexes"]["daymet"] = _binding(root, daymet)
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    compact = list(input_closure.BRIDGE_RAW_INDEXES)
    compact[0] = (*compact[0][:3], 2)
    monkeypatch.setattr(input_closure, "BRIDGE_RAW_INDEXES", tuple(compact))
    frozen, bridge, calls = _semantic_verifiers()
    with pytest.raises(InputClosureError, match="identity is malformed|duplicate"):
        resolve_development_input_closure(
            root,
            frozen_spec_verifier=frozen,  # type: ignore[arg-type]
            bridge_verifier=bridge,  # type: ignore[arg-type]
        )
    assert not calls


def test_semantic_verification_is_mandatory_and_postchecked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _development_fixture(tmp_path / "repo", monkeypatch)

    def mutate_after_verification(path: Path) -> dict[str, str]:
        source = root / "data_usgs/stations_meta_120v2.csv"
        source.write_bytes(b"changed during semantic verification\n")
        return {"evidence_role": "development_exploratory"}

    def passing_bridge(**_kwargs: object) -> dict[str, str]:
        return {"status": "PASS_EXACT_PRODUCT_BRIDGE"}

    with pytest.raises(InputClosureError, match="bytes changed"):
        resolve_development_input_closure(
            root,
            frozen_spec_verifier=mutate_after_verification,
            bridge_verifier=passing_bridge,
        )

    root = _development_fixture(tmp_path / "repo-no-go", monkeypatch)
    with pytest.raises(InputClosureError, match="did not return an exact PASS"):
        resolve_development_input_closure(
            root,
            frozen_spec_verifier=lambda _path: {
                "evidence_role": "development_exploratory"
            },
            bridge_verifier=lambda **_kwargs: {"status": "NO_GO"},
        )


def test_repository_root_symlink_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _development_fixture(tmp_path / "repo", monkeypatch)
    alias = tmp_path / "repo-alias"
    alias.symlink_to(root, target_is_directory=True)
    frozen, bridge, calls = _semantic_verifiers()
    with pytest.raises(InputClosureError, match="root must not be a symlink"):
        resolve_development_input_closure(
            alias,
            frozen_spec_verifier=frozen,  # type: ignore[arg-type]
            bridge_verifier=bridge,  # type: ignore[arg-type]
        )
    assert not calls


def test_fixture_contains_no_unexpected_development_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guard the test helper itself against accidentally weakening path assertions."""
    root = _development_fixture(tmp_path / "repo", monkeypatch)
    frozen, bridge, _calls = _semantic_verifiers()
    closure = resolve_development_input_closure(
        root,
        frozen_spec_verifier=frozen,  # type: ignore[arg-type]
        bridge_verifier=bridge,  # type: ignore[arg-type]
    )
    inventory = {entry.path for entry in closure.inventory}
    all_regular_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    assert inventory == all_regular_files
    shutil.rmtree(root)

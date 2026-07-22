from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute.predictor_bridge import (  # noqa: E402
    BRIDGE_FIELDS,
    PredictorBridgeError,
    _resolve_manifest_binding,
    assert_exact_predictor_table,
    compare_predictor_bridge,
    frozen_bridge_slice,
    replay_predictor_bridge_offline,
)
from thermoroute.provenance import canonical_json_bytes  # noqa: E402


def test_manifest_binding_rejects_symlink_and_hardlink_aliases(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    artifact = root / "artifact.json"
    artifact.write_bytes(b"bound bytes\n")
    binding = {
        "path": "artifact.json",
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    assert _resolve_manifest_binding(root, binding, label="fixture") == artifact

    alias = root / "hardlink.json"
    os.link(artifact, alias)
    with pytest.raises(PredictorBridgeError, match="bytes/path changed"):
        _resolve_manifest_binding(root, binding, label="fixture")
    alias.unlink()

    target = root / "target.json"
    artifact.rename(target)
    artifact.symlink_to(target.name)
    linked = {
        "path": "artifact.json",
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    with pytest.raises(PredictorBridgeError, match="linked"):
        _resolve_manifest_binding(root, linked, label="fixture")


def _table() -> pd.DataFrame:
    dates = pd.date_range("2020-12-28", "2020-12-31", freq="D")
    rows = []
    for site_index, site in enumerate(("1", "2")):
        for day_index, date in enumerate(dates):
            base = float(10 * site_index + day_index)
            rows.append({
                "site_no": site,
                "DATE": date,
                "TEMP": base,
                "PRCP": base + 1,
                "RHMEAN": base + 2,
                "DH": np.nan if date == pd.Timestamp("2020-12-31") else base + 3,
                "WDSP": base + 4,
            })
    return pd.DataFrame(rows)


def test_exact_predictor_bridge_passes_and_attests_calendar():
    table = _table()
    report = compare_predictor_bridge(
        table,
        table.copy(),
        expected_site_count=2,
        start=pd.Timestamp("2020-12-28"),
        end=pd.Timestamp("2020-12-31"),
    )
    assert report["status"] == "PASS_EXACT_PRODUCT_BRIDGE"
    assert report["outcome_values_requested_or_read"] is False
    assert report["daymet_calendar_attestation"][
        "leap_year_omitted_dates_in_interval"
    ] == ["2020-12-31"]
    assert all(
        report["fields"][field]["exact_product_compatibility"]
        for field in BRIDGE_FIELDS
    )


def test_predictor_bridge_fails_product_drift_and_date_shift():
    frozen = _table()
    changed = frozen.copy()
    changed["TEMP"] = changed.groupby("site_no").TEMP.shift(1)
    report = compare_predictor_bridge(
        frozen,
        changed,
        expected_site_count=2,
        start=pd.Timestamp("2020-12-28"),
        end=pd.Timestamp("2020-12-31"),
    )
    assert report["status"] == "NO_GO_PRODUCT_BRIDGE_MISMATCH"
    assert report["fields"]["TEMP"]["exact_product_compatibility"] is False
    assert report["fields"]["TEMP"]["zero_day_alignment_best_or_tied"] is False


def test_predictor_bridge_rejects_incomplete_calendar():
    table = _table().iloc[:-1].copy()
    with pytest.raises(PredictorBridgeError, match="key registry is incomplete"):
        compare_predictor_bridge(
            table,
            table,
            expected_site_count=2,
            start=pd.Timestamp("2020-12-28"),
            end=pd.Timestamp("2020-12-31"),
        )


def test_frozen_bridge_slice_requires_one_to_one_stable_mapping():
    panel = _table().rename(columns={"site_no": "site_id"})
    registry = pd.DataFrame({
        "site_no": ["01000001", "01000002"],
        "legacy_site_id": ["1", "1"],
    })
    with pytest.raises(PredictorBridgeError, match="one-to-one"):
        frozen_bridge_slice(
            panel,
            registry,
            start=pd.Timestamp("2020-12-28"),
            end=pd.Timestamp("2020-12-31"),
        )


def test_legacy_rejection_wording_correction_is_byte_bound():
    correction = json.loads(
        (ROOT / "data_usgs/rejected_sites_120v2_corrections_v1.json")
        .read_text(encoding="utf-8")
    )
    source = ROOT / correction["source"]["path"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == correction["source"][
        "sha256"
    ]
    rows = pd.read_csv(source)
    legacy = correction["corrections"][0]
    assert int(rows.reason.eq(legacy["legacy_value"]).sum()) == legacy["row_count"]
    assert "development-evaluation-period" in legacy["corrected_interpretation"]


def _one_site_raw_replay_fixture(tmp_path: Path) -> dict[str, Path]:
    source_manifest = json.loads(
        (ROOT / "data_usgs/development_predictor_bridge_v1.json").read_text()
    )
    source_request_map = json.loads(
        (ROOT / source_manifest["request_map"]["path"]).read_text()
    )
    request = source_request_map["requests"][0]
    site = str(request["site_no"])
    registry = pd.read_csv(
        ROOT / "data_usgs/station_registry_v1.csv", dtype={"site_no": "string"}
    )
    registry_path = tmp_path / "registry.csv"
    registry.loc[registry.site_no.astype(str).eq(site)].to_csv(registry_path, index=False)
    request_map_path = tmp_path / "request_map.json"
    request_map_path.write_bytes(canonical_json_bytes({
        **source_request_map,
        "request_count": 1,
        "requests": [request],
    }))
    output: dict[str, Path] = {
        "registry": registry_path,
        "request_map": request_map_path,
    }
    for label, source_binding in source_manifest["raw_snapshot_indexes"].items():
        source_index_path = ROOT / source_binding["path"]
        source_index = json.loads(source_index_path.read_text())
        wanted = (
            source_request_map["gridmet_provider_contract"]["request_sha256"]
            if label == "gridmet_schema"
            else request[label]["request_sha256"]
        )
        record = next(
            row for row in source_index["records"]
            if row["request_sha256"] == wanted
        )
        destination_root = tmp_path / label
        for field in ("metadata_path", "response_path"):
            source = source_index_path.parent / record[field]
            destination = destination_root / record[field]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        destination_index = destination_root / "snapshot_index_v2.json"
        destination_index.write_bytes(canonical_json_bytes({
            "schema_version": 2,
            "snapshot_count": 1,
            "records": [record],
        }))
        output[label] = destination_index
    output["expected"] = ROOT / source_manifest["normalized"]["refreshed"]["path"]
    return output


def _replay_one_site(paths: dict[str, Path]) -> pd.DataFrame:
    return replay_predictor_bridge_offline(
        registry_path=paths["registry"],
        request_map_path=paths["request_map"],
        daymet_index_path=paths["daymet"],
        gridmet_index_path=paths["gridmet"],
        gridmet_schema_index_path=paths["gridmet_schema"],
        expected_sites=1,
    )


def test_raw_predictor_replay_rebuilds_current_frozen_values(tmp_path: Path) -> None:
    paths = _one_site_raw_replay_fixture(tmp_path)
    replayed = _replay_one_site(paths)
    site = pd.read_csv(paths["registry"], dtype={"site_no": "string"}).site_no.iloc[0]
    expected = pd.read_parquet(paths["expected"])
    expected = expected.loc[expected.site_no.astype(str).eq(str(site))]
    assert_exact_predictor_table(
        replayed, expected, label="one-site archived parser replay"
    )


@pytest.mark.parametrize(
    "attack",
    (
        "index_extra_field", "metadata_header", "metadata_hash", "request_coordinate",
        "missing_record", "extra_record", "response_symlink", "response_hardlink",
        "arbitrary_response_with_rebound_hashes", "path_traversal",
    ),
)
def test_raw_predictor_replay_rejects_lineage_and_filesystem_attacks(
    tmp_path: Path, attack: str,
) -> None:
    paths = _one_site_raw_replay_fixture(tmp_path)
    index_path = paths["daymet"]
    index = json.loads(index_path.read_text())
    record = index["records"][0]
    metadata_path = index_path.parent / record["metadata_path"]
    response_path = index_path.parent / record["response_path"]
    request_map = json.loads(paths["request_map"].read_text())

    if attack == "index_extra_field":
        record["unexpected"] = True
    elif attack == "metadata_header":
        metadata = json.loads(metadata_path.read_text())
        metadata["response_headers"]["x-forged"] = "yes"
        metadata_path.write_bytes(canonical_json_bytes(metadata))
    elif attack == "metadata_hash":
        record["metadata_sha256"] = "0" * 64
    elif attack == "request_coordinate":
        request_map["requests"][0]["requested_lat"] += 0.01
        paths["request_map"].write_bytes(canonical_json_bytes(request_map))
    elif attack == "missing_record":
        index["records"] = []
        index["snapshot_count"] = 0
    elif attack == "extra_record":
        index["records"].append(dict(record))
        index["snapshot_count"] = 2
    elif attack == "response_symlink":
        backup = response_path.with_name("response-backup.bin")
        shutil.copy2(response_path, backup)
        response_path.unlink()
        response_path.symlink_to(backup.name)
    elif attack == "response_hardlink":
        backup = response_path.with_name("response-backup.bin")
        shutil.copy2(response_path, backup)
        response_path.unlink()
        os.link(backup, response_path)
    elif attack == "arbitrary_response_with_rebound_hashes":
        payload = b"arbitrary copied-refreshed attack bytes\n"
        response_path.write_bytes(payload)
        response_sha = hashlib.sha256(payload).hexdigest()
        metadata = json.loads(metadata_path.read_text())
        metadata["byte_count"] = len(payload)
        metadata["response_sha256"] = response_sha
        metadata_path.write_bytes(canonical_json_bytes(metadata))
        record["byte_count"] = len(payload)
        record["response_sha256"] = response_sha
        record["metadata_byte_count"] = metadata_path.stat().st_size
        record["metadata_sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
        request_map["requests"][0]["daymet"]["byte_count"] = len(payload)
        request_map["requests"][0]["daymet"]["response_sha256"] = response_sha
        paths["request_map"].write_bytes(canonical_json_bytes(request_map))
    elif attack == "path_traversal":
        record["response_path"] = "../outside.bin"
    else:  # pragma: no cover - parametrization is closed
        raise AssertionError(attack)
    if attack not in {"metadata_header", "request_coordinate", "response_symlink", "response_hardlink"}:
        index_path.write_bytes(canonical_json_bytes(index))
    with pytest.raises((PredictorBridgeError, ValueError, KeyError)):
        _replay_one_site(paths)


def test_exact_predictor_table_rejects_copied_keys_with_one_value_changed(
    tmp_path: Path,
) -> None:
    replayed = _replay_one_site(_one_site_raw_replay_fixture(tmp_path))
    forged = replayed.copy()
    forged.loc[0, "TEMP"] = float(forged.loc[0, "TEMP"]) + 0.001
    with pytest.raises(PredictorBridgeError, match="TEMP"):
        assert_exact_predictor_table(replayed, forged, label="forged refreshed table")

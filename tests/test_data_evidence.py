"""Strong tests for frozen data identity, raw provenance and holdout honesty."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoroute.evidence import (
    DEFAULT_FROZEN_PANEL_SPEC,
    EvidenceError,
    FrozenPanelSpec,
    load_confirmatory_protocol,
    select_confirmatory_sites,
)
from thermoroute import data as data_module
from thermoroute.provenance import ProvenanceError, SnapshotStore, sha256_bytes
from thermoroute.usgs import _parse_nwis_rdb, fetch_nwis_daily


def test_checked_in_panel_registry_is_frozen_and_uses_site_no():
    spec = FrozenPanelSpec.load(DEFAULT_FROZEN_PANEL_SPEC)
    evidence = spec.verify()
    assert evidence == {
        "panel_id": "usgs120-development-v1",
        "panel_sha256": "0427a07ea4514ba29ce7d0cf89594e6c35c7f9134cc4d1d96fdc90daeaf5ba69",
        "registry_sha256": "090e7c0daf39ac38ceefeb1af8a12c178283e18347905d8e72ada969ad5460c9",
        "row_count": 657480,
        "station_count": 120,
        "site_primary_key": "site_no",
        "evidence_role": "development_exploratory",
    }
    registry = spec.load_registry()
    assert registry["site_no"].nunique() == 120
    assert registry["legacy_site_id"].str.fullmatch(r"n\d{2,3}").all()
    assert registry["huc_metadata_status"].value_counts().to_dict() == {
        "USGS_SNAPSHOT_SITE_NO_MATCH": 120,
    }
    panel = spec.load_panel(stable_site_ids=True)
    assert set(panel["site_id"].astype(str)) == set(registry["site_no"].astype(str))
    assert panel["legacy_site_id"].str.fullmatch(r"n\d{2,3}").all()
    assert not panel["site_id"].str.fullmatch(r"n\d{2,3}").any()


def test_frozen_panel_rejects_checksum_drift(tmp_path):
    original = json.loads(DEFAULT_FROZEN_PANEL_SPEC.read_text(encoding="utf-8"))
    panel_copy = tmp_path / "panel.parquet"
    panel_copy.write_bytes(FrozenPanelSpec.load().panel_path.read_bytes())
    registry_copy = tmp_path / "registry.csv"
    registry_copy.write_bytes(FrozenPanelSpec.load().registry_path.read_bytes())
    metadata_copy = tmp_path / "metadata.csv"
    metadata_copy.write_bytes(FrozenPanelSpec.load().source_metadata_path.read_bytes())
    original["panel"]["path"] = panel_copy.name
    original["station_registry"]["path"] = registry_copy.name
    original["station_registry"]["source_metadata_path"] = metadata_copy.name
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(original), encoding="utf-8")
    panel_copy.write_bytes(panel_copy.read_bytes() + b"tampered")
    with pytest.raises(EvidenceError, match="checksum mismatch"):
        FrozenPanelSpec.load(spec_path).load_panel()


def test_retired_usgs_panels_require_explicit_legacy_opt_in():
    retired = DEFAULT_FROZEN_PANEL_SPEC.parent / "panel_usgs_100.parquet"
    with pytest.raises(EvidenceError, match="non-canonical USGS panel"):
        data_module.prepare_dataset_from_panel(str(retired))


class _FakeResponse:
    status = 200
    headers = {"Content-Type": "text/plain", "ETag": "fixture-v1"}

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self.payload


def test_raw_snapshot_records_request_time_and_checksum(monkeypatch, tmp_path):
    payload = b"provider response bytes\n"
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        return _FakeResponse(payload)

    monkeypatch.setattr("thermoroute.provenance.urllib.request.urlopen", fake_urlopen)
    store = SnapshotStore(tmp_path / "raw")
    first, record = store.fetch(
        provider="fixture", url="https://example.test/api?site=123", retries=1)
    assert first == payload
    assert record.response_sha256 == sha256_bytes(payload)
    meta = json.loads(record.metadata_path.read_text(encoding="utf-8"))
    assert meta["request"]["url"] == "https://example.test/api?site=123"
    assert meta["retrieved_at_utc"].endswith("+00:00")
    assert meta["byte_count"] == len(payload)
    assert meta["response_sha256"] == sha256_bytes(payload)

    # Repeating offline reuses verified bytes and never contacts the provider.
    second, second_record = SnapshotStore(store.root, offline=True).fetch(
        provider="fixture", url="https://example.test/api?site=123")
    assert second == payload
    assert second_record.request_sha256 == record.request_sha256
    assert len(calls) == 1

    index = store.write_index()
    indexed = json.loads(index.read_text(encoding="utf-8"))
    assert indexed["snapshot_count"] == 1
    record.response_path.write_bytes(b"corrupt")
    with pytest.raises(ProvenanceError, match="checksum mismatch"):
        SnapshotStore(store.root, offline=True).fetch(
            provider="fixture", url="https://example.test/api?site=123")


def test_nwis_raw_rdb_parser_preserves_site_number_and_mean_statistic():
    payload = (
        b"# fixture\n"
        b"agency_cd\tsite_no\tdatetime\t123_00010_00003\t123_00010_00003_cd\n"
        b"5s\t15s\t20d\t14n\t10s\n"
        b"USGS\t01234567\t2021-01-01\t4.2\tA\n"
    )
    parsed = _parse_nwis_rdb(payload)
    assert parsed.loc[0, "site_no"] == "01234567"
    assert parsed.loc[0, "123_00010_00003"] == "4.2"
    assert parsed.loc[0, "123_00010_00003_cd"] == "A"


def test_nwis_daily_can_rebuild_from_snapshotted_rdb():
    payload = (
        b"agency_cd\tsite_no\tdatetime\t123_00010_00003\t123_00060_00003\t123_00065_00003\n"
        b"5s\t15s\t20d\t14n\t14n\t14n\n"
        b"USGS\t01234567\t2021-01-01\t4.2\t10.0\t2.0\n"
        b"USGS\t01234567\t2021-01-02\t4.4\t11.0\t2.1\n"
    )

    class FixtureStore:
        def fetch(self, **_kwargs):
            return payload, None

    daily = fetch_nwis_daily(
        "01234567", "2021-01-01", "2021-01-02",
        snapshot_store=FixtureStore(),
    )
    assert daily is not None
    assert list(daily.columns) == ["WTEMP", "FLOW", "WLEVEL"]
    assert daily.loc[pd.Timestamp("2021-01-01")].to_dict() == {
        "WTEMP": 4.2, "FLOW": 10.0, "WLEVEL": 2.0,
    }


def test_confirmatory_selection_is_new_metadata_only_and_deterministic():
    candidates = pd.DataFrame({
        "site_no": ["100", "200", "300", "400"],
        "station_nm": ["a", "b", "c", "d"],
        "lat": [1.0, 2.0, 3.0, 4.0],
        "lon": [-1.0, -2.0, -3.0, -4.0],
    })
    first = select_confirmatory_sites(
        candidates, {"100"}, n_sites=2, selection_seed="public-seed")
    second = select_confirmatory_sites(
        candidates.sample(frac=1, random_state=7), {"100"},
        n_sites=2, selection_seed="public-seed")
    pd.testing.assert_frame_equal(first, second)
    assert "100" not in set(first["site_no"])

    leaky = candidates.assign(wtemp_test_coverage=0.99)
    with pytest.raises(EvidenceError, match="cannot inspect"):
        select_confirmatory_sites(
            leaky, set(), n_sites=2, selection_seed="public-seed")


def test_checked_in_confirmatory_protocol_does_not_claim_completion():
    path = Path(__file__).resolve().parents[1] / "protocols" / "route_a_confirmatory_v1.json"
    protocol = load_confirmatory_protocol(path)
    assert protocol["status"] == "FROZEN_NOT_ACQUIRED"
    assert protocol["authoritative_protocol_commit"].startswith("6fce087")
    assert protocol["development_evidence"]["existing_2019_2020_role"] == (
        "EXPLORATORY_NOT_BLIND")
    assert protocol["availability_contract"]["labels_default_state"] == (
        "SEALED_NOT_ACQUIRED")

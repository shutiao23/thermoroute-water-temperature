from __future__ import annotations

from datetime import date
import importlib.util
import json
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute.confirmatory import (  # noqa: E402
    CANDIDATE_COLUMNS,
    ROUTE_A_STATE_UNIVERSE,
    build_usgs_candidate_url,
    merge_candidate_metadata,
    parse_usgs_candidate_metadata,
)
from thermoroute.evidence import EvidenceError  # noqa: E402
from thermoroute.nwp import (  # noqa: E402
    ISSUE_SEMANTICS,
    NWP_COMMON_VALID_TIME_START,
    NWPContractError,
    build_previous_runs_url,
    iter_month_chunks,
    parse_previous_runs_daily,
)
from thermoroute.provenance import canonical_json_bytes, sha256_bytes, sha256_file  # noqa: E402


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


def test_holdout_freezer_replays_raw_candidate_evidence(tmp_path):
    module = _load_script(
        "confirmatory_holdout_evidence_test",
        "scripts/data_usgs/confirmatory_holdout.py",
    )
    payload = _candidate_payload()
    candidates = parse_usgs_candidate_metadata(payload, state="CO")
    candidate_path = tmp_path / "candidates.csv"
    candidate_path.write_text(candidates.to_csv(index=False, lineterminator="\n"))

    response_path = tmp_path / "provider" / "request-one" / "response.bin"
    response_path.parent.mkdir(parents=True)
    response_path.write_bytes(payload)
    request_sha = "request-one"
    response_sha = sha256_bytes(payload)
    index_path = tmp_path / "snapshot_index.json"
    index_path.write_bytes(canonical_json_bytes({
        "schema_version": 1,
        "snapshot_count": 1,
        "records": [{
            "provider": "usgs-nwis-confirmatory-site-metadata",
            "request_sha256": request_sha,
            "response_sha256": response_sha,
            "request": {
                "provider": "usgs-nwis-confirmatory-site-metadata",
                "method": "GET",
                "url": build_usgs_candidate_url("CO"),
                "headers": {},
            },
            "response_path": str(response_path.relative_to(tmp_path)),
            "metadata_path": "unused-in-derived-verification.json",
        }],
    }))
    provenance_path = tmp_path / "candidates.provenance.json"
    provenance_path.write_bytes(canonical_json_bytes({
        "schema_version": 1,
        "artifact_role": "PRE_LABEL_METADATA_ONLY_CANDIDATE_UNIVERSE",
        "state_universe": ["CO"],
        "candidate_count": 1,
        "candidate_table_sha256": sha256_file(candidate_path),
        "raw_snapshot_index_sha256": sha256_file(index_path),
        "outcome_endpoint_requested": False,
        "outcome_values_requested": False,
        "holdout_coverage_requested_or_computed": False,
        "requests": [{
            "state": "CO",
            "request_sha256": request_sha,
            "response_sha256": response_sha,
        }],
    }))
    rebuilt = module.verify_candidate_evidence(
        candidate_path, provenance_path, index_path
    )
    pd.testing.assert_frame_equal(rebuilt, candidates, check_dtype=False)

    candidate_path.write_text(candidate_path.read_text() + "\n")
    with pytest.raises(RuntimeError, match="checksum"):
        module.verify_candidate_evidence(candidate_path, provenance_path, index_path)


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

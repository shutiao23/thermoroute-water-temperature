from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import thermoroute.opening as opening_module  # noqa: E402
import thermoroute.outcome_acquisition as outcome_acquisition  # noqa: E402
from thermoroute.opening import (  # noqa: E402
    CONTROL_INTERVENTIONS,
    OpeningAlreadyStarted,
    OpeningContractError,
    _required_models,
    _assert_worker_predictions_equal_trusted,
    _build_outcome_quality_audit,
    _frozen_calibration,
    _probabilistic_evaluation,
    _score_sequence_bundle,
    _spatial_cluster_diagnostics,
    _validate_development_prediction_parity,
    _validate_hashed_requirements_lock,
    _verify_opened_nwis_index,
    _verify_snapshot_index,
    _validate_temporal_controls,
    compute_confirmatory_statistics,
    exclusive_create_json,
    freeze_opening_authorization,
    opening_status,
    run_opening_once,
    validate_authorization,
    validate_protocol_seal,
)
from thermoroute.checkpoint import (  # noqa: E402
    load_inference_bundle,
    neural_output_head_schema,
    save_inference_bundle,
)
from thermoroute.provenance import (  # noqa: E402
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from thermoroute.model_suite import development_prediction_binding  # noqa: E402
import thermoroute.opening_contract as acquisition_contract  # noqa: E402
from thermoroute.probability import (  # noqa: E402
    fit_frozen_seasonal_event_reference,
)
from thermoroute.repro import RunIdentity, seal_artifact  # noqa: E402
from thermoroute.train import LSTMForecaster  # noqa: E402


def _opening_git_safety_repository(tmp_path: Path):
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    git("init", "-q")
    git("config", "user.email", "fixture@example.test")
    git("config", "user.name", "Fixture")
    tracked = repository / "tracked.txt"
    tracked.write_text("first\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "-q", "-m", "first")
    tracked.write_text("second\n", encoding="utf-8")
    git("commit", "-q", "-a", "-m", "second")
    return repository, git


def _assert_unsafe_git_blocks_authorization(
    repository: Path, *, match: str
) -> None:
    authorization = (
        repository
        / "data_usgs"
        / "confirmatory_opening_authorization_v1.json"
    )
    missing = repository / "must-not-be-read"
    with pytest.raises(OpeningContractError, match=match):
        freeze_opening_authorization(
            authorization,
            root=repository,
            protocol_path=missing,
            development_registry=missing,
            external_registry=missing,
            external_lock=missing,
            model_suite=missing,
            input_manifest=missing,
        )
    assert not authorization.exists()
    with pytest.raises(OpeningContractError, match=match):
        validate_authorization(authorization, root=repository)
    assert not authorization.exists()


def test_spatial_cluster_diagnostics_warns_without_creating_inference() -> None:
    comparison = {
        "n_stations": 10,
        "n_clusters": 3,
        "per_huc": [
            {"huc2": "01", "n_stations": 8, "median_station_effect_c": -0.1},
            {"huc2": "02", "n_stations": 1, "median_station_effect_c": 0.1},
            {"huc2": "03", "n_stations": 1, "median_station_effect_c": 0.2},
        ],
        "leave_one_huc": [
            {"held_out_huc2": "01", "effect_minus_margin_c": 0.1},
            {"held_out_huc2": "02", "effect_minus_margin_c": -0.2},
            {"held_out_huc2": "03", "effect_minus_margin_c": -0.1},
        ],
    }
    actual = _spatial_cluster_diagnostics(comparison)

    assert actual["effective_cluster_count_inverse_herfindahl"] < 2.0
    assert actual["largest_cluster_share"] == 0.8
    assert actual["loho_direction"] == "CROSSES_OR_TOUCHES_MARGIN"
    assert actual["inference_strength"] == "NO_STRONG_INFERENCE"
    assert set(actual["warning_codes"]) == {
        "SMALL_CLUSTER_COUNT_LT_30",
        "DOMINANT_CLUSTER_SHARE_GE_0_25",
        "EFFECTIVE_CLUSTER_FRACTION_LT_0_75",
        "LOHO_MARGIN_DIRECTION_UNSTABLE_OR_NOT_ESTIMABLE",
    }
    assert not ({"p_value", "confidence_interval", "decision"} & set(actual))


def test_spatial_cluster_diagnostics_rejects_inconsistent_totals() -> None:
    with pytest.raises(OpeningContractError, match="totals disagree"):
        _spatial_cluster_diagnostics({
            "n_stations": 3,
            "n_clusters": 2,
            "per_huc": [{"huc2": "01", "n_stations": 2}],
            "leave_one_huc": [{
                "held_out_huc2": "01", "effect_minus_margin_c": -0.1,
            }],
        })


def test_authorization_rejects_real_git_replace_ref_before_any_opening(tmp_path):
    repository, git = _opening_git_safety_repository(tmp_path)
    git("replace", git("rev-parse", "HEAD"), git("rev-parse", "HEAD^"))
    _assert_unsafe_git_blocks_authorization(repository, match="replacement refs")


def test_authorization_rejects_real_git_graft_before_any_opening(tmp_path):
    repository, git = _opening_git_safety_repository(tmp_path)
    graft = repository / ".git" / "info" / "grafts"
    graft.write_text(git("rev-parse", "HEAD") + "\n", encoding="utf-8")
    _assert_unsafe_git_blocks_authorization(repository, match="graft file")


def test_authorization_rejects_shallow_history_before_any_opening(tmp_path):
    repository, git = _opening_git_safety_repository(tmp_path)
    shallow = repository / ".git" / "shallow"
    shallow.write_text(git("rev-parse", "HEAD") + "\n", encoding="utf-8")
    _assert_unsafe_git_blocks_authorization(repository, match="shallow")


def test_authorization_rejects_ambient_git_override_before_any_opening(
    tmp_path, monkeypatch,
):
    repository, _git = _opening_git_safety_repository(tmp_path)
    monkeypatch.setenv("GIT_DIR", str(repository / ".git"))
    _assert_unsafe_git_blocks_authorization(repository, match="ambient Git")


def test_protocol_seal_replays_final_git_bytes_and_supports_bound_gitless_archive(
    tmp_path,
):
    repository = tmp_path / "repository"
    protocols = repository / "protocols"
    protocols.mkdir(parents=True)

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments], cwd=repository, check=True,
            text=True, capture_output=True,
        )
        return result.stdout.strip()

    git("init")
    git("config", "user.email", "fixture@example.test")
    git("config", "user.name", "Fixture")
    markdown = protocols / "route_a_confirmatory_protocol.md"
    protocol = protocols / "route_a_confirmatory_v1.json"
    markdown.write_text("# Original preregistration\n", encoding="utf-8")
    git("add", markdown.relative_to(repository).as_posix())
    git("commit", "-m", "original protocol")
    original_commit = git("rev-parse", "HEAD")
    original_markdown_sha256 = sha256_file(markdown)

    protocol.write_text('{"schema_version":1}\n', encoding="utf-8")
    markdown.write_text("# Final prelabel protocol\n", encoding="utf-8")
    git(
        "add",
        protocol.relative_to(repository).as_posix(),
        markdown.relative_to(repository).as_posix(),
    )
    git("commit", "-m", "final prelabel protocol")
    final_commit = git("rev-parse", "HEAD")

    seal = protocols / "route_a_protocol_seal_v1.json"
    seal_document = {
        "format": "thermoroute.route-a-protocol-seal.v1",
        "status": "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED",
        "protocol_id": "route-a-confirmatory-v1",
        "recorded_date": "2026-07-22",
        "original_preregistration": {
            "commit": original_commit,
            "markdown": {
                "path": markdown.relative_to(repository).as_posix(),
                "sha256": original_markdown_sha256,
            },
        },
        "final_prelabel_protocol": {
            "commit": final_commit,
            "json": {
                "path": protocol.relative_to(repository).as_posix(),
                "sha256": sha256_file(protocol),
            },
            "markdown": {
                "path": markdown.relative_to(repository).as_posix(),
                "sha256": sha256_file(markdown),
            },
        },
        "history_contract": {
            "original_commit_must_be_ancestor_of_final_commit": True,
            "final_commit_must_be_ancestor_of_authorization_commit": True,
            "git_show_bytes_must_match_every_declared_hash": True,
            "current_protocol_bytes_must_match_final_commit": True,
        },
        "prelabel_attestation": {
            "post_2020_wtemp_requested_or_inspected": False,
            "post_2020_flow_or_wlevel_requested_or_inspected": False,
            "confirmation_outcome_artifact_present": False,
            "external_timestamp_or_public_preregistration": False,
            "independent_custodian_or_worm_storage": False,
            "scope": "fixture honest-owner evidence",
        },
    }
    seal.write_bytes(canonical_json_bytes(seal_document))
    seal_sha256 = sha256_file(seal)

    validated = validate_protocol_seal(
        seal,
        protocol_path=protocol,
        root=repository,
        authoritative_commit=original_commit,
    )
    assert validated["final_commit"] == final_commit
    assert validated["sha256"] == seal_sha256

    original_protocol = protocol.read_bytes()
    protocol.write_text('{"schema_version":2}\n', encoding="utf-8")
    with pytest.raises(OpeningContractError, match="checksum changed"):
        validate_protocol_seal(
            seal,
            protocol_path=protocol,
            root=repository,
            authoritative_commit=original_commit,
        )
    protocol.write_bytes(original_protocol)

    archive = tmp_path / "archive"
    archive_protocols = archive / "protocols"
    archive_protocols.mkdir(parents=True)
    for source in (protocol, markdown, seal):
        (archive_protocols / source.name).write_bytes(source.read_bytes())
    gitless = validate_protocol_seal(
        archive_protocols / seal.name,
        protocol_path=archive_protocols / protocol.name,
        root=archive,
        authoritative_commit=original_commit,
        allow_gitless_archive=True,
        frozen_seal_sha256=seal_sha256,
    )
    assert gitless["final_commit"] == final_commit


def test_opening_state_machine_treats_any_intent_as_irreversible(tmp_path):
    intent = tmp_path / "intent.json"
    receipt = tmp_path / "receipt.json"
    assert opening_status(intent_path=intent, receipt_path=receipt) == (
        "SEALED_READY_OR_NOT_AUTHORIZED"
    )
    exclusive_create_json(intent, {"status": "OPENING_STARTED_IRREVERSIBLE"})
    assert opening_status(intent_path=intent, receipt_path=receipt) == (
        "OPENING_INCOMPLETE_SAME_OPENING_RESUME_REQUIRES_VALIDATION"
    )
    with pytest.raises(OpeningAlreadyStarted):
        exclusive_create_json(intent, {"status": "forged retry"})
    exclusive_create_json(receipt, {"status": "OPENED_AND_SCORED_ONCE"})
    assert opening_status(intent_path=intent, receipt_path=receipt) == (
        "OPENED_AND_SCORED_ONCE"
    )


def test_completed_opening_allows_only_document_commit_descendant(tmp_path):
    def git(*arguments):
        result = subprocess.run(
            ["git", *arguments],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    git("init", "-q")
    git("config", "user.email", "fixture@example.test")
    git("config", "user.name", "Fixture")
    (tmp_path / "README.md").write_text("compute\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-q", "-m", "compute")
    compute = git("rev-parse", "HEAD")
    receipt_relative = "outputs/confirmatory/route_a_fixture/opening_receipt_v1.json"
    receipt = tmp_path / receipt_relative
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}\n", encoding="utf-8")
    paper = tmp_path / "paper" / "main.md"
    paper.parent.mkdir()
    paper.write_text("results\n", encoding="utf-8")
    git("add", "paper/main.md")
    git("commit", "-q", "-m", "manuscript")
    manuscript = git("rev-parse", "HEAD")
    authorization = {"state_paths": {"receipt": receipt_relative}}
    assert opening_module._is_document_only_postopening_descendant(
        tmp_path,
        authorization,
        compute_commit=compute,
        current_commit=manuscript,
    )

    source = tmp_path / "src" / "forbidden.py"
    source.parent.mkdir()
    source.write_text("changed = True\n", encoding="utf-8")
    git("add", "src/forbidden.py")
    git("commit", "-q", "-m", "forbidden source")
    forbidden = git("rev-parse", "HEAD")
    assert not opening_module._is_document_only_postopening_descendant(
        tmp_path,
        authorization,
        compute_commit=compute,
        current_commit=forbidden,
    )
    git("rm", "src/forbidden.py")
    git("commit", "-q", "-m", "revert forbidden source in final tree")
    reverted = git("rev-parse", "HEAD")
    assert not opening_module._is_document_only_postopening_descendant(
        tmp_path,
        authorization,
        compute_commit=compute,
        current_commit=reverted,
    )


def test_authorization_prepublication_snapshot_rejects_git_and_ignored_source_races(
    tmp_path,
):
    def git(*arguments):
        result = subprocess.run(
            ["git", *arguments],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    git("init", "-q")
    git("config", "user.email", "fixture@example.test")
    git("config", "user.name", "Fixture")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/base.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("src/ignored.py\n", encoding="utf-8")
    git("add", "src/base.py", ".gitignore")
    git("commit", "-q", "-m", "frozen source")
    initial_state = opening_module._live_git_state(tmp_path)
    initial_inventory = opening_module.source_inventory(tmp_path)

    ignored = tmp_path / "src/ignored.py"
    ignored.write_text("ATTACK = True\n", encoding="utf-8")
    with pytest.raises(OpeningContractError, match="source inventory changed"):
        opening_module._assert_prepublication_source_snapshot(
            tmp_path,
            initial_git_state=initial_state,
            frozen_source_inventory=initial_inventory,
        )
    ignored.unlink()

    (tmp_path / "src/base.py").write_text("VALUE = 2\n", encoding="utf-8")
    git("add", "src/base.py")
    git("commit", "-q", "-m", "concurrent source commit")
    with pytest.raises(OpeningContractError, match="HEAD/worktree changed"):
        opening_module._assert_prepublication_source_snapshot(
            tmp_path,
            initial_git_state=initial_state,
            frozen_source_inventory=initial_inventory,
        )


def test_opening_lightgbm_target_check_uses_exact_float32_semantics(monkeypatch):
    truth64 = np.asarray([32.1], dtype=np.float64)
    truth32 = truth64.astype(np.float32)
    expected = pd.DataFrame(
        {
            "site_id": ["1"],
            "horizon": [1],
            "issue_date": pd.to_datetime(["2021-01-01"]),
            "target_date": pd.to_datetime(["2021-01-02"]),
            "y_true": truth64,
        }
    )

    def tabular_with(value):
        return pd.DataFrame(
            {
                "site_id": ["1"],
                "issue_date": pd.to_datetime(["2021-01-01"]),
                "target_date": pd.to_datetime(["2021-01-02"]),
                "y": [value],
                "feature": [1.0],
            }
        )

    monkeypatch.setattr(
        opening_module.F,
        "build_tabular",
        lambda *_args, **_kwargs: tabular_with(truth32[0]),
    )
    monkeypatch.setattr(
        opening_module.F, "feature_columns", lambda _frame: ["feature"]
    )
    design = opening_module._confirmation_tabular_design(
        pd.DataFrame(),
        object(),
        expected,
        feature_order=("WTEMP",),
        horizon=1,
        station_order=("1",),
        manifest={"station_categories": [], "design_feature_order": ["feature"]},
        external=True,
    )
    assert design.feature.tolist() == [1.0]

    different = np.nextafter(truth32[0], np.float32(np.inf), dtype=np.float32)
    monkeypatch.setattr(
        opening_module.F,
        "build_tabular",
        lambda *_args, **_kwargs: tabular_with(different),
    )
    with pytest.raises(OpeningContractError, match="tabular target changed"):
        opening_module._confirmation_tabular_design(
            pd.DataFrame(),
            object(),
            expected,
            feature_order=("WTEMP",),
            horizon=1,
            station_order=("1",),
            manifest={"station_categories": [], "design_feature_order": ["feature"]},
            external=True,
        )


def test_production_opening_api_rejects_callbacks_and_alternate_commands():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        run_opening_once(  # type: ignore[call-arg]
            "authorization.json",
            root=".",
            worker=lambda _order: None,  # type: ignore[arg-type,return-value]
            preflight_validator=lambda *_args, **_kwargs: {},
        )
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        run_opening_once(  # type: ignore[call-arg]
            "authorization.json",
            root=".",
            command=["python", "forged.py"],
        )


def test_acquisition_source_replay_binds_usgs_and_provenance(
    tmp_path,
    monkeypatch,
):
    required = {
        "thermoroute.opening_contract": "src/thermoroute/opening_contract.py",
        "thermoroute.outcome_acquisition": (
            "src/thermoroute/outcome_acquisition.py"
        ),
        "thermoroute.provenance": "src/thermoroute/provenance.py",
        "thermoroute.usgs": "src/thermoroute/usgs.py",
    }
    for relative in required.values():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    for name in tuple(sys.modules):
        if name == "thermoroute" or name.startswith("thermoroute."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    for name, relative in required.items():
        monkeypatch.setitem(
            sys.modules,
            name,
            SimpleNamespace(__file__=str(tmp_path / relative)),
        )
    inventory = acquisition_contract._source_inventory(tmp_path)
    authorization = {
        "source": {
            "source_inventory": inventory,
            "source_tree_sha256": opening_module.sha256_json(inventory),
        }
    }
    assert acquisition_contract.validate_frozen_source_identity(
        root=tmp_path,
        authorization=authorization,
    ) == inventory

    (tmp_path / required["thermoroute.provenance"]).write_text(
        "# changed after parent preflight\n",
        encoding="utf-8",
    )
    with pytest.raises(
        acquisition_contract.AcquisitionContractError,
        match="source tree differs",
    ):
        acquisition_contract.validate_frozen_source_identity(
            root=tmp_path,
            authorization=authorization,
        )

    provenance = tmp_path / required["thermoroute.provenance"]
    provenance.write_text(
        f"# {required['thermoroute.provenance']}\n",
        encoding="utf-8",
    )
    outside_module = tmp_path.parent / f"{tmp_path.name}-injected-module.py"
    outside_module.write_text("# injected\n", encoding="utf-8")
    monkeypatch.setitem(
        sys.modules,
        "thermoroute.injected",
        SimpleNamespace(__file__=str(outside_module)),
    )
    with pytest.raises(
        acquisition_contract.AcquisitionContractError,
        match="loaded project module escapes frozen source",
    ):
        acquisition_contract.validate_frozen_source_identity(
            root=tmp_path,
            authorization=authorization,
        )


def test_acquisition_rechecks_frozen_source_before_first_network_io(
    tmp_path,
    monkeypatch,
):
    work_order_path = tmp_path / "state" / "work_order.json"
    work_order_path.parent.mkdir(parents=True)
    work_order_path.write_text("{}\n", encoding="utf-8")
    state = {
        "run_directory": tmp_path / "state",
        "work_order": work_order_path,
        "intent": tmp_path / "state" / "intent.json",
        "transport_root": tmp_path / "state" / "transport",
        "raw_nwis_root": tmp_path / "state" / "transport" / "raw",
        "raw_nwis_snapshot_index": (
            tmp_path / "state" / "transport" / "raw" / "snapshot_index.json"
        ),
        "acquisition_request_map": (
            tmp_path / "state" / "acquisition" / "request-map.json"
        ),
        "temporal_outcomes": (
            tmp_path / "state" / "acquisition" / "temporal.parquet"
        ),
        "external_outcomes": (
            tmp_path / "state" / "acquisition" / "external.parquet"
        ),
        "acquisition_manifest": (
            tmp_path / "state" / "acquisition" / "manifest.json"
        ),
        "receipt": tmp_path / "state" / "receipt.json",
        "receipt_sha256": tmp_path / "state" / "receipt.sha256",
        **{
            key: tmp_path / "state" / "trusted" / f"{key}.artifact"
            for key in acquisition_contract.TRUSTED_STATE_KEYS
        },
    }
    work_order = {
        "opening_id": "fixture",
        "authorization_sha256": "a" * 64,
        "work_order_self_sha256": "b" * 64,
        "site_registries": {
            "temporal": {"sites": ["01000001"]},
            "external": {"sites": []},
        },
    }
    authorization = {
        "opening_id": "fixture",
        "acquisition_plan": {
            "history_start": "2021-01-01",
            "target_end": "2021-01-02",
        },
    }
    monkeypatch.setattr(
        outcome_acquisition,
        "validate_acquisition_work_order",
        lambda *_args, **_kwargs: (work_order, authorization, state),
    )
    monkeypatch.setattr(
        outcome_acquisition,
        "validate_frozen_source_identity",
        lambda **_kwargs: (_ for _ in ()).throw(
            acquisition_contract.AcquisitionContractError(
                "changed after parent preflight"
            )
        ),
    )
    network_calls = 0

    def forbidden_network(*_args, **_kwargs):
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("network must remain sealed")

    monkeypatch.setattr(
        outcome_acquisition.urllib.request,
        "build_opener",
        forbidden_network,
    )
    with pytest.raises(
        outcome_acquisition.OutcomeAcquisitionError,
        match="transport did not complete",
    ):
        outcome_acquisition.acquire_from_work_order(
            work_order_path,
            root=tmp_path,
            entrypoint_path=work_order_path,
        )
    assert network_calls == 0
    assert not state["acquisition_manifest"].exists()


def test_raw_entrypoint_rejects_dangling_provider_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "state"
    work_order_path = run / "work_order.json"
    transport = run / "transport"
    raw_root = transport / "raw"
    acquisition = run / "acquisition"
    trusted = run / "trusted"
    state = {
        "run_directory": run,
        "work_order": work_order_path,
        "intent": run / "intent.json",
        "transport_root": transport,
        "raw_nwis_root": raw_root,
        "raw_nwis_snapshot_index": raw_root / "snapshot_index.json",
        "acquisition_request_map": acquisition / "request-map.json",
        "temporal_outcomes": acquisition / "temporal.parquet",
        "external_outcomes": acquisition / "external.parquet",
        "acquisition_manifest": acquisition / "manifest.json",
        "receipt": run / "receipt.json",
        "receipt_sha256": run / "receipt.sha256",
        **{
            key: trusted / f"{key}.artifact"
            for key in acquisition_contract.TRUSTED_STATE_KEYS
        },
    }
    work_order = {
        "opening_id": "fixture",
        "authorization_sha256": "a" * 64,
        "work_order_self_sha256": "b" * 64,
        "site_registries": {
            "temporal": {"sites": ["01000001"]},
            "external": {"sites": []},
        },
    }
    authorization = {
        "opening_id": "fixture",
        "acquisition_plan": {
            "history_start": "2021-01-01",
            "target_end": "2021-01-02",
        },
    }
    run.mkdir()
    work_order_path.write_bytes(canonical_json_bytes(work_order))
    work_order_path.chmod(0o444)
    expected_ledger = outcome_acquisition._expected_request_ledger(
        work_order=work_order,
        authorization=authorization,
        work_order_path=work_order_path,
    )
    ledger_path = transport / "request_ledger_v1.json"
    outcome_acquisition._create_bytes(
        ledger_path, canonical_json_bytes(expected_ledger)
    )
    raw_root.mkdir()
    provider = raw_root / outcome_acquisition.CONFIRMATORY_NWIS_PROVIDER
    provider.symlink_to(tmp_path / "missing-provider", target_is_directory=True)
    monkeypatch.setattr(
        outcome_acquisition,
        "validate_acquisition_work_order",
        lambda *_args, **_kwargs: (work_order, authorization, state),
    )
    network_calls = 0

    def forbidden_network(**_kwargs):
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("dangling provider reached network")

    monkeypatch.setattr(
        outcome_acquisition, "_fetch_create_only", forbidden_network
    )
    with pytest.raises(
        outcome_acquisition.OutcomeAcquisitionError,
        match="provider root is malformed",
    ):
        outcome_acquisition.acquire_from_work_order(
            work_order_path,
            root=tmp_path,
            entrypoint_path=work_order_path,
            resume=True,
        )
    assert network_calls == 0


def test_canonical_opening_state_rejects_symlink_component_and_write(
    tmp_path,
):
    root = tmp_path / "repository"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "outputs").symlink_to(outside, target_is_directory=True)
    state_paths = {
        "namespace": "fixture",
        "intent": "outputs/confirmatory/route_a_fixture/intent.json",
    }
    with pytest.raises(OpeningContractError, match="state path is unsafe"):
        opening_module._secure_canonical_state_paths(root, state_paths)
    with pytest.raises(OpeningContractError, match="directory traversal is unsafe"):
        exclusive_create_json(
            root / state_paths["intent"],
            {"must_not_escape": True},
        )
    assert not (outside / "confirmatory" / "route_a_fixture" / "intent.json").exists()


def test_manifest_binding_rejects_same_bytes_at_noncanonical_path(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical" / "request_map.json"
    alias = tmp_path / "alias" / "request_map.json"
    canonical.parent.mkdir()
    alias.parent.mkdir()
    canonical.write_bytes(b"same immutable bytes\n")
    alias.write_bytes(canonical.read_bytes())
    binding = {
        "path": alias.relative_to(tmp_path).as_posix(),
        "sha256": sha256_file(alias),
    }

    with pytest.raises(OpeningContractError, match="path is noncanonical"):
        opening_module._verify_canonical_file_binding(
            tmp_path,
            binding,
            expected_path=canonical,
            label="opened request map",
        )


class _StreamingResponse:
    status = 200
    headers: dict[str, str] = {}

    def __init__(self, payload: bytes, url: str = "https://example.test/"):
        self.payload = payload
        self.url = url
        self.offset = 0
        self.request_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.request_sizes.append(size)
        if self.offset >= len(self.payload):
            return b""
        end = len(self.payload) if size < 0 else self.offset + size
        chunk = self.payload[self.offset:end]
        self.offset += len(chunk)
        return chunk

    def geturl(self) -> str:
        return self.url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_nwis_response_read_is_chunked_hashed_and_capped(monkeypatch, tmp_path):
    monkeypatch.setattr(
        outcome_acquisition,
        "MAX_CONFIRMATORY_NWIS_RESPONSE_BYTES",
        10,
    )
    monkeypatch.setattr(outcome_acquisition, "_RESPONSE_CHUNK_BYTES", 4)
    response = _StreamingResponse(b"0123456789")
    payload, digest = outcome_acquisition._read_bounded_response(response)
    assert payload == b"0123456789"
    assert digest == sha256_bytes(payload)
    assert response.request_sizes == [4, 4, 3, 1]

    request = {
        "url": "https://example.test/too-large",
        "method": "GET",
        "headers": {},
    }
    spec = {
        "request": request,
        "request_sha256": sha256_bytes(canonical_json_bytes(request)),
    }
    oversized = _StreamingResponse(b"01234567890", url=request["url"])

    class _Opener:
        def open(self, *_args, **_kwargs):
            return oversized

    monkeypatch.setattr(
        outcome_acquisition.urllib.request,
        "build_opener",
        lambda *_args: _Opener(),
    )
    raw_root = tmp_path / "raw"
    with pytest.raises(
        outcome_acquisition.OutcomeAcquisitionError,
        match="fixed NWIS acquisition failed",
    ):
        outcome_acquisition._fetch_create_only(
            raw_root=raw_root,
            spec=spec,
            work_order={
                "opening_id": "fixture",
                "authorization_sha256": "a" * 64,
                "work_order_self_sha256": "b" * 64,
            },
            request_ledger_sha256="c" * 64,
            attempt_number=1,
            attempts=1,
        )
    assert not raw_root.exists()


def test_raw_acquisition_entrypoint_runs_isolated_two_site_offline_e2e():
    """Exercise the real raw-only child without allowing a network connection."""
    payloads = {
        "01000001": (
            b"# temporal fixture\n"
            b"agency_cd\tsite_no\tdatetime\t123_00010_00003\t"
            b"123_00010_00003_cd\t456_00010_00003\t"
            b"456_00010_00003_cd\t123_00060_00003\t"
            b"123_00060_00003_cd\t123_00065_00003\t"
            b"123_00065_00003_cd\n"
            b"5s\t15s\t20d\t14n\t10s\t14n\t10s\t14n\t10s\t14n\t10s\n"
            b"USGS\t01000001\t2021-01-01\t4.2\tA\t\t\t-0.5\tP\t2.0\tA\n"
            b"USGS\t01000001\t2021-01-02\t5.1\tA\t5.2\tX-UNKNOWN\t"
            b"11.0\tA\t\t\n"
        ),
        "02000001": (
            b"# external fixture\n"
            b"agency_cd\tsite_no\tdatetime\t789_00010_00003\t"
            b"789_00010_00003_cd\t789_00060_00003\t"
            b"789_00060_00003_cd\t789_00065_00003\t"
            b"789_00065_00003_cd\n"
            b"5s\t15s\t20d\t14n\t10s\t14n\t10s\t14n\t10s\n"
            b"USGS\t02000001\t2021-01-01\t8.4\tA\t21.0\tA\t3.1\tP\n"
        ),
    }

    with tempfile.TemporaryDirectory(
        prefix=".route-a-acquisition-e2e-", dir=ROOT
    ) as fixture_name:
        fixture_root = Path(fixture_name).resolve()
        fixture_relative = fixture_root.relative_to(ROOT).as_posix()

        def write_json(path: Path, value: object) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(canonical_json_bytes(value))

        def fixed_binding(relative: str) -> dict[str, str]:
            path = (ROOT / relative).resolve()
            return {
                "path": relative,
                "realpath": str(path),
                "sha256": sha256_file(path),
            }

        registries = {}
        for cohort, site in (("development", "01000001"), ("external", "02000001")):
            path = fixture_root / f"{cohort}_registry.csv"
            path.write_text(f"site_no\n{site}\n", encoding="utf-8")
            registries[cohort] = {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
            }

        state_base = f"{fixture_relative}/state"
        state_paths = {
            "namespace": "offline-two-site-fixture",
            "run_directory": state_base,
            "work_order": f"{state_base}/acquisition_work_order_v1.json",
            "intent": f"{state_base}/opening_intent_v1.json",
            "transport_root": f"{state_base}/transport",
            "raw_nwis_root": f"{state_base}/transport/raw_nwis_v1",
            "raw_nwis_snapshot_index": (
                f"{state_base}/transport/raw_nwis_v1/snapshot_index.json"
            ),
            "acquisition_request_map": (
                f"{state_base}/acquisition/source_request_map_v1.json"
            ),
            "temporal_outcomes": (
                f"{state_base}/acquisition/temporal_outcomes_v1.parquet"
            ),
            "external_outcomes": (
                f"{state_base}/acquisition/external_outcomes_v1.parquet"
            ),
            "acquisition_manifest": (
                f"{state_base}/acquisition/acquisition_manifest_v1.json"
            ),
            "receipt": f"{state_base}/opening_receipt_v1.json",
            "receipt_sha256": f"{state_base}/opening_receipt_v1.sha256",
            **{
                key: f"{state_base}/trusted/{key}.artifact"
                for key in acquisition_contract.TRUSTED_STATE_KEYS
            },
        }
        source_inventory = opening_module.source_inventory(ROOT)
        authorization_path = fixture_root / "authorization.json"
        authorization = {
            "format": acquisition_contract.AUTHORIZATION_FORMAT,
            "status": "AUTHORIZED_LABELS_STILL_SEALED",
            "opening_id": "offline-two-site-fixture",
            "protocol": {"sha256": "c" * 64},
            "registries": registries,
            "source": {
                "authorization_path": authorization_path.relative_to(ROOT).as_posix(),
                "source_tree_sha256": opening_module.sha256_json(
                    source_inventory
                ),
                "source_inventory": source_inventory,
            },
            "runtime": {"runtime_sha256": "b" * 64},
            "fixed_code": {
                "sha256": "d" * 64,
                "entrypoints": {
                    "acquisition": fixed_binding(
                        "scripts/route_a_outcome_acquisition.py"
                    )
                },
                "files": {
                    relative: fixed_binding(relative)
                    for relative in (
                        "src/thermoroute/opening_contract.py",
                        "src/thermoroute/outcome_acquisition.py",
                    )
                },
            },
            "acquisition_plan": {
                "history_start": "2021-01-01",
                "target_start": "2021-01-01",
                "target_end": "2021-01-02",
                "maximum_response_bytes_per_request": (
                    acquisition_contract.MAX_CONFIRMATORY_NWIS_RESPONSE_BYTES
                ),
            },
            "state_paths": state_paths,
        }
        authorization["authorization_self_sha256"] = opening_module.sha256_json(
            authorization
        )
        write_json(authorization_path, authorization)
        authorization_sha256 = sha256_file(authorization_path)

        work_order_path = ROOT / state_paths["work_order"]
        work_order = {
            "format": acquisition_contract.ACQUISITION_WORK_ORDER_FORMAT,
            "opening_id": authorization["opening_id"],
            "authorization_path": authorization["source"]["authorization_path"],
            "authorization_sha256": authorization_sha256,
            "source_tree_sha256": authorization["source"]["source_tree_sha256"],
            "runtime_sha256": authorization["runtime"]["runtime_sha256"],
            "fixed_code_sha256": authorization["fixed_code"]["sha256"],
            "acquisition_plan": authorization["acquisition_plan"],
            "state_paths": state_paths,
            "site_registries": {
                "temporal": {
                    "sha256": registries["development"]["sha256"],
                    "sites": ["01000001"],
                },
                "external": {
                    "sha256": registries["external"]["sha256"],
                    "sites": ["02000001"],
                },
            },
        }
        work_order["work_order_self_sha256"] = opening_module.sha256_json(work_order)
        write_json(work_order_path, work_order)
        work_order_path.chmod(0o444)

        intent_path = ROOT / state_paths["intent"]
        intent = {
            "format": acquisition_contract.INTENT_FORMAT,
            "status": "OPENING_STARTED_IRREVERSIBLE",
            "opening_id": authorization["opening_id"],
            "authorization_sha256": authorization_sha256,
            "work_order_self_sha256": work_order["work_order_self_sha256"],
            "work_order_file_sha256": sha256_file(work_order_path),
            "fixed_code_sha256": authorization["fixed_code"]["sha256"],
            "runtime_sha256": authorization["runtime"]["runtime_sha256"],
                "maximum_openings": 1,
                "retry_after_failure_allowed": False,
                "same_opening_transport_resume_allowed": True,
            }
        intent["intent_self_sha256"] = opening_module.sha256_json(intent)
        write_json(intent_path, intent)
        intent_path.chmod(0o444)

        encoded_payloads = {
            site: base64.b64encode(payload).decode("ascii")
            for site, payload in payloads.items()
        }
        allow_external_success = fixture_root / "allow_external_success"
        kill_after_complete_replay = fixture_root / "kill_after_complete_replay"
        wrapper_path = fixture_root / "offline_http_wrapper.py"
        wrapper_path.write_text(
            "\n".join([
                "import base64",
                "from pathlib import Path",
                "import runpy",
                "import sys",
                f"sys.path.insert(0, {str(ROOT / 'src')!r})",
                "import time",
                "from urllib.parse import parse_qs, urlsplit",
                "import urllib.request",
                "import thermoroute.outcome_acquisition as acquisition",
                f"PAYLOADS = {encoded_payloads!r}",
                f"ALLOW_EXTERNAL_SUCCESS = Path({str(allow_external_success)!r})",
                f"KILL_AFTER_COMPLETE_REPLAY = Path({str(kill_after_complete_replay)!r})",
                f"REQUEST_LEDGER = Path({str(ROOT / state_paths['transport_root'] / 'request_ledger_v1.json')!r})",
                "RESUME = sys.argv[1:] == ['resume']",
                "time.sleep = lambda _seconds: None",
                "CALL_COUNT = 0",
                "def transport_fault(point):",
                "    if point == 'after_complete_transaction_replay' and KILL_AFTER_COMPLETE_REPLAY.exists():",
                "        raise RuntimeError('injected death after exact complete replay')",
                "acquisition._acquisition_transport_fault = transport_fault",
                "class Response:",
                "    status = 200",
                "    def __init__(self, request, payload, call_count):",
                "        self.request = request",
                "        self.payload = payload",
                "        self.offset = 0",
                "        self.headers = {'X-Fixture-Call': str(call_count)}",
                "    def read(self, size=-1):",
                "        if self.offset >= len(self.payload): return b''",
                "        end = len(self.payload) if size < 0 else self.offset + size",
                "        chunk = self.payload[self.offset:end]",
                "        self.offset += len(chunk)",
                "        return chunk",
                "    def geturl(self): return self.request.full_url",
                "    def __enter__(self): return self",
                "    def __exit__(self, *_args): return False",
                "class Opener:",
                "    def open(self, request, timeout):",
                "        global CALL_COUNT",
                "        CALL_COUNT += 1",
                "        if not REQUEST_LEDGER.is_file():",
                "            raise AssertionError('HTTPS attempted before ledger freeze')",
                "        site = parse_qs(urlsplit(request.full_url).query)['sites'][0]",
                "        if site == '02000001' and not ALLOW_EXTERNAL_SUCCESS.exists():",
                "            raise OSError('injected transient network failure')",
                "        return Response(request, base64.b64decode(PAYLOADS[site]), CALL_COUNT)",
                "urllib.request.build_opener = lambda *_args: Opener()",
                f"entrypoint = Path({str(ROOT / 'scripts/route_a_outcome_acquisition.py')!r})",
                f"sys.argv = [str(entrypoint), '--work-order', {str(work_order_path)!r}] + (['--resume'] if RESUME else [])",
                "runpy.run_path(str(entrypoint), run_name='__main__')",
            ]) + "\n",
            encoding="utf-8",
        )
        child_tmp = fixture_root / "tmp"
        child_tmp.mkdir()
        child_environment = opening_module._sanitized_child_environment(
            temporary_root=child_tmp
        )
        command = [sys.executable, "-I", "-B", str(wrapper_path)]
        initial = subprocess.run(
            command,
            cwd=ROOT,
            env=child_environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert initial.returncode != 0
        assert not (ROOT / state_paths["acquisition_manifest"]).exists()
        assert not (ROOT / state_paths["temporal_outcomes"]).exists()
        assert not (ROOT / state_paths["external_outcomes"]).exists()
        raw_root = ROOT / state_paths["raw_nwis_root"]
        assert not (raw_root / "snapshot_index.json").exists()
        transport_root = ROOT / state_paths["transport_root"]
        ledger_path = transport_root / "request_ledger_v1.json"
        frozen_ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        assert frozen_ledger["request_count"] == 2
        temporal_request = next(
            row for row in frozen_ledger["requests"]
            if row["site_no"] == "01000001"
        )
        temporal_transaction = (
            raw_root
            / "usgs-nwis-confirmatory-dv"
            / temporal_request["request_sha256"]
        )
        temporal_before = {
            path.name: (sha256_file(path), path.stat().st_mtime_ns)
            for path in temporal_transaction.iterdir()
        }
        first_result = json.loads(
            (
                transport_root
                / "transport_attempts_v1"
                / "attempt_000001_result.json"
            ).read_text(encoding="utf-8")
        )
        assert first_result["status"] == (
            "TRANSPORT_INCOMPLETE_MAY_RESUME_SAME_OPENING"
        )
        assert first_result["completed_request_sha256"] == [
            temporal_request["request_sha256"]
        ]

        allow_external_success.write_text("allow fixture response\n", encoding="utf-8")
        kill_after_complete_replay.write_text("crash once\n", encoding="utf-8")
        interrupted = subprocess.run(
            [*command, "resume"],
            cwd=ROOT,
            env=child_environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert interrupted.returncode != 0
        assert not (ROOT / state_paths["acquisition_manifest"]).exists()
        attempts_root = transport_root / "transport_attempts_v1"
        assert (attempts_root / "attempt_000002_start.json").is_file()
        assert not (attempts_root / "attempt_000002_result.json").exists()
        complete_before_reconstruction = {
            path.relative_to(raw_root).as_posix(): (
                sha256_file(path), path.stat().st_mtime_ns
            )
            for path in raw_root.rglob("*")
            if path.is_file()
        }

        kill_after_complete_replay.unlink()
        result = subprocess.run(
            [*command, "resume"],
            cwd=ROOT,
            env=child_environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        complete_after_reconstruction = {
            path.relative_to(raw_root).as_posix(): (
                sha256_file(path), path.stat().st_mtime_ns
            )
            for path in raw_root.rglob("*")
            if path.is_file() and path.name != "snapshot_index.json"
        }
        assert complete_after_reconstruction == complete_before_reconstruction
        temporal_after = {
            path.name: (sha256_file(path), path.stat().st_mtime_ns)
            for path in temporal_transaction.iterdir()
        }
        assert temporal_after == temporal_before

        manifest_path = ROOT / state_paths["acquisition_manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["format"] == acquisition_contract.ACQUISITION_MANIFEST_FORMAT
        assert manifest["producer_role"] == "RAW_ONLY_NO_PREDICTIONS_OR_STATISTICS"
        assert manifest["raw_nwis_snapshot_index"]["path"] == state_paths[
            "raw_nwis_snapshot_index"
        ]
        assert manifest["request_map"]["path"] == state_paths[
            "acquisition_request_map"
        ]
        assert manifest["normalized_outcome_tables"]["temporal"][
            "path"
        ] == state_paths["temporal_outcomes"]
        assert manifest["normalized_outcome_tables"]["external"][
            "path"
        ] == state_paths["external_outcomes"]
        request_map = json.loads(
            (ROOT / state_paths["acquisition_request_map"]).read_text(encoding="utf-8")
        )
        assert request_map["request_count"] == 2
        assert {
            (row["cohort"], row["site_no"])
            for row in request_map["requests"]
        } == {("temporal", "01000001"), ("external", "02000001")}

        snapshot_index = json.loads(
            (raw_root / "snapshot_index.json").read_text(encoding="utf-8")
        )
        assert snapshot_index["snapshot_count"] == 2
        observed_calls = []
        for record in snapshot_index["records"]:
            site = record["request"]["url"].split("sites=")[1].split("&", 1)[0]
            response_path = raw_root / record["response_path"]
            metadata_path = raw_root / record["metadata_path"]
            assert response_path.read_bytes() == payloads[site]
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            observed_calls.append(int(metadata["response_headers"]["X-Fixture-Call"]))
            assert metadata["response_sha256"] == sha256_bytes(payloads[site])
            assert response_path.stat().st_mode & 0o222 == 0
            assert metadata_path.stat().st_mode & 0o222 == 0
        assert observed_calls == [1, 1]
        assert sorted(
            record["attempt_number"] for record in snapshot_index["records"]
        ) == [1, 2]
        attempt_index = json.loads(
            (transport_root / "transport_attempt_index_v1.json").read_text(
                encoding="utf-8"
            )
        )
        assert attempt_index["attempt_count"] == 2
        assert attempt_index["resume_count"] == 1
        assert attempt_index["opening_count"] == 1
        assert attempt_index["response_replacement_count"] == 0
        assert attempt_index[
            "completed_before_final_attempt_request_sha256"
        ] == [temporal_request["request_sha256"]]
        assert manifest["transport_summary"]["resume_count"] == 1
        assert manifest["response_replacement_count"] == 0
        independently_verified = opening_module._verify_opened_transport_evidence(
            root=ROOT,
            acquisition=manifest,
            records=snapshot_index["records"],
            request_rows=request_map["requests"],
            opening_id=authorization["opening_id"],
            authorization_sha256=authorization_sha256,
            work_order_path=work_order_path,
            raw_root=raw_root,
        )
        assert independently_verified == manifest["transport_summary"]

        temporal = pd.read_parquet(ROOT / state_paths["temporal_outcomes"])
        external = pd.read_parquet(ROOT / state_paths["external_outcomes"])
        assert temporal.site_no.astype(str).tolist() == ["01000001", "01000001"]
        assert temporal.DATE.dt.strftime("%Y-%m-%d").tolist() == [
            "2021-01-01", "2021-01-02",
        ]
        assert temporal.loc[0, "FLOW"] == -0.5
        assert temporal.loc[1, "WTEMP_value_status"] == (
            "MULTIPLE_FINITE_SERIES_CONFLICT"
        )
        assert bool(temporal.loc[1, "WTEMP_series_conflict"])
        assert external.site_no.astype(str).tolist() == ["02000001", "02000001"]
        assert external.loc[0, "WTEMP"] == 8.4
        assert external.loc[1, "WTEMP_value_status"] == "MISSING_NO_FINITE_SERIES"

        immutable_before = {
            path.relative_to(fixture_root).as_posix(): sha256_file(path)
            for path in fixture_root.rglob("*")
            if path.is_file() and path != wrapper_path
        }
        retry = subprocess.run(
            [*command, "resume"],
            cwd=ROOT,
            env=child_environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert retry.returncode != 0
        immutable_after = {
            path.relative_to(fixture_root).as_posix(): sha256_file(path)
            for path in fixture_root.rglob("*")
            if path.is_file() and path != wrapper_path
        }
        assert immutable_after == immutable_before


def test_raw_resume_rejects_partial_canonical_transaction(tmp_path):
    request = {
        "schema_version": 1,
        "provider": "usgs-nwis-confirmatory-dv",
        "method": "GET",
        "url": opening_module.build_nwis_confirmatory_url(
            "01000001", "2021-01-01", "2021-01-02"
        ),
        "headers": {},
    }
    request_sha = sha256_bytes(canonical_json_bytes(request))
    transaction = tmp_path / request_sha
    transaction.mkdir(mode=0o700)
    response = transaction / "response.bin"
    response.write_bytes(b"complete bytes but missing immutable metadata")
    response.chmod(0o444)
    spec = {
        "ordinal": 1,
        "cohort": "temporal",
        "site_no": "01000001",
        "request": request,
        "request_sha256": request_sha,
    }
    starts = {
        1: {
            "opening_id": "fixture",
            "authorization_sha256": "a" * 64,
            "work_order_self_sha256": "b" * 64,
            "request_ledger_sha256": "c" * 64,
            "missing_at_start_request_sha256": [request_sha],
        }
    }
    with pytest.raises(
        outcome_acquisition.OutcomeAcquisitionError,
        match="partial or extraneous NWIS transaction is indeterminate",
    ):
        outcome_acquisition._validate_transaction(
            directory=transaction,
            spec=spec,
            starts=starts,
            results={},
        )


def _write_attempt_history_fixture(
    attempts_root: Path,
    attempts: list[dict[str, object]],
) -> None:
    attempts_root.mkdir()
    common = {
        "opening_id": "fixture-opening",
        "authorization_sha256": "a" * 64,
        "work_order_self_sha256": "b" * 64,
        "request_ledger_sha256": "c" * 64,
        "opening_count": 1,
    }
    for number, specification in enumerate(attempts, start=1):
        start_stable = {
            "format": outcome_acquisition.ACQUISITION_ATTEMPT_START_FORMAT,
            "status": "TRANSPORT_ATTEMPT_STARTED",
            **common,
            "attempt_number": number,
            "mode": specification["mode"],
            "completed_before_attempt_request_sha256": specification[
                "start_completed"
            ],
            "missing_at_start_request_sha256": specification[
                "start_missing"
            ],
            "response_replacement_allowed": False,
            "started_at_utc": "2026-07-22T00:00:00+00:00",
        }
        start = {
            **start_stable,
            "attempt_start_self_sha256": sha256_bytes(
                canonical_json_bytes(start_stable)
            ),
        }
        start_path = attempts_root / f"attempt_{number:06d}_start.json"
        start_path.write_bytes(canonical_json_bytes(start))
        start_path.chmod(0o444)
        if "result_completed" not in specification:
            continue
        status = specification.get(
            "status", "TRANSPORT_INCOMPLETE_MAY_RESUME_SAME_OPENING"
        )
        result_stable = {
            "format": outcome_acquisition.ACQUISITION_ATTEMPT_RESULT_FORMAT,
            "status": status,
            **common,
            "attempt_number": number,
            "attempt_start_sha256": sha256_file(start_path),
            "completed_request_sha256": specification["result_completed"],
            "missing_request_sha256": specification["result_missing"],
            "failure_class": (
                None
                if status == "ALL_LEDGER_TRANSACTIONS_COMPLETE"
                else "FixtureFailure"
            ),
            "response_replacement_count": 0,
            "completed_at_utc": "2026-07-22T00:00:00+00:00",
        }
        result = {
            **result_stable,
            "attempt_result_self_sha256": sha256_bytes(
                canonical_json_bytes(result_stable)
            ),
        }
        result_path = attempts_root / f"attempt_{number:06d}_result.json"
        result_path.write_bytes(canonical_json_bytes(result))
        result_path.chmod(0o444)


@pytest.mark.parametrize(
    ("scenario", "message"),
    [
        ("first_resume", "first transport attempt"),
        ("first_partition_jump", "first transport attempt"),
        ("partition_jump", "does not continue the prior result partition"),
        ("completed_rollback", "rolls back completed requests"),
        ("after_all_complete", "exists after ledger completion"),
    ],
)
def test_attempt_history_rejects_self_consistent_state_machine_attacks(
    tmp_path: Path,
    scenario: str,
    message: str,
) -> None:
    first, second = "1" * 64, "2" * 64
    attempt_one: dict[str, object] = {
        "mode": "INITIAL_OPENING_TRANSPORT",
        "start_completed": [],
        "start_missing": [first, second],
        "result_completed": [first],
        "result_missing": [second],
    }
    attempt_two: dict[str, object] = {
        "mode": "RESUME_SAME_OPENING",
        "start_completed": [first],
        "start_missing": [second],
        "result_completed": [first, second],
        "result_missing": [],
        "status": "ALL_LEDGER_TRANSACTIONS_COMPLETE",
    }
    if scenario == "first_resume":
        attempt_one["mode"] = "RESUME_SAME_OPENING"
        attempts = [attempt_one]
    elif scenario == "first_partition_jump":
        attempt_one["start_completed"] = [first]
        attempt_one["start_missing"] = [second]
        attempts = [attempt_one]
    elif scenario == "partition_jump":
        attempt_two["start_completed"] = [second]
        attempt_two["start_missing"] = [first]
        attempts = [attempt_one, attempt_two]
    elif scenario == "completed_rollback":
        attempt_two["result_completed"] = []
        attempt_two["result_missing"] = [first, second]
        attempt_two.pop("status")
        attempts = [attempt_one, attempt_two]
    else:
        attempt_one["result_completed"] = [first, second]
        attempt_one["result_missing"] = []
        attempt_one["status"] = "ALL_LEDGER_TRANSACTIONS_COMPLETE"
        attempt_two["start_completed"] = [first, second]
        attempt_two["start_missing"] = []
        attempts = [attempt_one, attempt_two]
    attempts_root = tmp_path / "attempts"
    _write_attempt_history_fixture(attempts_root, attempts)

    with pytest.raises(
        outcome_acquisition.OutcomeAcquisitionError, match=message
    ):
        outcome_acquisition._load_attempt_history(
            attempts_root=attempts_root,
            opening_id="fixture-opening",
            authorization_sha256="a" * 64,
            work_order_self_sha256="b" * 64,
            request_ledger_sha256="c" * 64,
            request_ids={first, second},
        )


def test_canonical_transaction_must_be_closed_by_its_attempt_result(
    tmp_path: Path,
) -> None:
    request = {
        "schema_version": 1,
        "provider": "usgs-nwis-confirmatory-dv",
        "method": "GET",
        "url": opening_module.build_nwis_confirmatory_url(
            "01000001", "2021-01-01", "2021-01-02"
        ),
        "headers": {},
    }
    request_sha = sha256_bytes(canonical_json_bytes(request))
    spec = {
        "ordinal": 1,
        "cohort": "temporal",
        "site_no": "01000001",
        "request": request,
        "request_sha256": request_sha,
    }
    transaction = tmp_path / request_sha
    transaction.mkdir(mode=0o700)
    payload = b"durable fixture response\n"
    response = transaction / "response.bin"
    response.write_bytes(payload)
    response.chmod(0o444)
    metadata = {
        "schema_version": 1,
        "opening_id": "fixture-opening",
        "authorization_sha256": "a" * 64,
        "work_order_self_sha256": "b" * 64,
        "request_ledger_sha256": "c" * 64,
        "attempt_number": 1,
        "request": request,
        "request_sha256": request_sha,
        "retrieved_at_utc": "2026-07-22T00:00:00+00:00",
        "http_status": 200,
        "response_headers": {},
        "final_url": request["url"],
        "byte_count": len(payload),
        "response_sha256": sha256_bytes(payload),
        "response_file": "response.bin",
        "maximum_response_bytes_per_request": (
            acquisition_contract.MAX_CONFIRMATORY_NWIS_RESPONSE_BYTES
        ),
    }
    metadata_path = transaction / "metadata.json"
    metadata_path.write_bytes(canonical_json_bytes(metadata))
    metadata_path.chmod(0o444)
    starts = {
        1: {
            "opening_id": "fixture-opening",
            "authorization_sha256": "a" * 64,
            "work_order_self_sha256": "b" * 64,
            "request_ledger_sha256": "c" * 64,
            "missing_at_start_request_sha256": [request_sha],
        }
    }
    results = {1: {"completed_request_sha256": []}}

    with pytest.raises(
        outcome_acquisition.OutcomeAcquisitionError,
        match="absent from its attempt result",
    ):
        outcome_acquisition._validate_transaction(
            directory=transaction,
            spec=spec,
            starts=starts,
            results=results,
        )


def test_transport_lock_rejects_concurrent_resume(tmp_path):
    lock_path = tmp_path / ".transport_resume.lock"
    contender = "\n".join([
        "import fcntl, os, sys",
        "descriptor = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)",
        "try:",
        "    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)",
        "except BlockingIOError:",
        "    raise SystemExit(0)",
        "raise SystemExit(1)",
    ])
    with outcome_acquisition._exclusive_transport_lock(lock_path):
        result = subprocess.run(
            [sys.executable, "-c", contender, str(lock_path)],
            text=True,
            capture_output=True,
            check=False,
        )
    assert result.returncode == 0, result.stderr


_RAW_RESUME_FORBIDDEN_STATE_KEYS = (
    acquisition_contract.RAW_ACQUISITION_FORBIDDEN_STATE_KEYS
)


def _stub_resume_inspection(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    acquisition = run / "acquisition"
    transport = run / "transport"
    trusted = run / "trusted"
    state = {
        "run_directory": run,
        "intent": run / "opening_intent_v1.json",
        "work_order": run / "acquisition_work_order_v1.json",
        "transport_root": transport,
        "raw_nwis_root": transport / "raw_nwis_v1",
        "raw_nwis_snapshot_index": (
            transport / "raw_nwis_v1" / "snapshot_index.json"
        ),
        "acquisition_request_map": acquisition / "request_map.json",
        "temporal_outcomes": acquisition / "temporal.parquet",
        "external_outcomes": acquisition / "external.parquet",
        "acquisition_manifest": acquisition / "manifest.json",
        "receipt": run / "receipt.json",
        "receipt_sha256": run / "receipt.sha256",
        **{
            key: trusted / f"{key}.artifact"
            for key in acquisition_contract.TRUSTED_STATE_KEYS
        },
    }
    state["intent"].write_text("{}\n", encoding="utf-8")
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text("{}\n", encoding="utf-8")
    work_order = {
        "site_registries": {
            "temporal": {"sites": ["01000001"]},
            "external": {"sites": ["02000001"]},
        }
    }
    state["work_order"].write_bytes(canonical_json_bytes(work_order))
    state["work_order"].chmod(0o444)
    monkeypatch.setattr(
        opening_module,
        "validate_authorization",
        lambda *_args, **_kwargs: {
            "state_paths": state,
            "authorization": {"opening_id": "fixture"},
        },
    )
    monkeypatch.setattr(
        opening_module,
        "_expected_acquisition_work_order",
        lambda *_args, **_kwargs: work_order,
    )
    monkeypatch.setattr(
        opening_module,
        "_validated_intent",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        outcome_acquisition,
        "inspect_transport_resume_state",
        lambda **_kwargs: {
            "classification": "RESUMABLE_MISSING_REQUESTS",
            "completed_request_count": 1,
            "missing_request_count": 1,
            "recoverable_pending_request_count": 0,
            "refetchable_nondurable_response_count": 0,
            "attempt_count": 1,
        },
    )
    return authorization_path, state


@pytest.mark.parametrize(
    "forbidden_kind",
    ["outcome_qc_gate", "trusted_directory", "acquisition_directory"],
)
def test_raw_entrypoint_directly_rejects_every_publication_boundary(
    tmp_path, monkeypatch, forbidden_kind,
):
    authorization_path, state = _stub_resume_inspection(tmp_path, monkeypatch)
    work_order = json.loads(state["work_order"].read_text(encoding="utf-8"))
    authorization = {"opening_id": "fixture"}
    monkeypatch.setattr(
        outcome_acquisition,
        "validate_acquisition_work_order",
        lambda *_args, **_kwargs: (work_order, authorization, state),
    )

    network_calls = 0

    def forbidden_network(**_kwargs):
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("raw forbidden check occurred after network access")

    monkeypatch.setattr(
        outcome_acquisition, "_fetch_create_only", forbidden_network
    )
    if forbidden_kind == "outcome_qc_gate":
        state["outcome_qc_gate"].parent.mkdir(parents=True)
        state["outcome_qc_gate"].write_bytes(b"forbidden")
    elif forbidden_kind == "trusted_directory":
        state["outcome_qc_gate"].parent.mkdir(parents=True)
    else:
        state["acquisition_manifest"].parent.mkdir(parents=True)

    with pytest.raises(
        outcome_acquisition.OutcomeAcquisitionError,
        match="cannot continue after derived/trusted publication",
    ):
        outcome_acquisition.acquire_from_work_order(
            state["work_order"],
            root=tmp_path,
            entrypoint_path=authorization_path,
            resume=True,
        )
    assert network_calls == 0
    assert not state["transport_root"].exists()


def test_status_marks_only_validated_raw_partial_state_resumable(
    tmp_path, monkeypatch,
):
    authorization_path, _state = _stub_resume_inspection(tmp_path, monkeypatch)
    result = opening_module.inspect_same_opening_transport_resume(
        authorization_path, root=tmp_path
    )
    assert result["status"] == (
        "OPENING_INCOMPLETE_SAME_OPENING_RAW_TRANSPORT_VALIDATED"
    )
    assert result["resume_phase"] == "RAW_TRANSPORT"
    assert result["raw_transport_resume_allowed"] is True
    assert result["network_free_trusted_completion_allowed"] is False
    assert result["transport"]["classification"] == "RESUMABLE_MISSING_REQUESTS"


def test_status_marks_partial_transaction_indeterminate(tmp_path, monkeypatch):
    authorization_path, _state = _stub_resume_inspection(tmp_path, monkeypatch)

    def reject_partial(**_kwargs):
        raise outcome_acquisition.OutcomeAcquisitionError(
            "partial canonical transaction"
        )

    monkeypatch.setattr(
        outcome_acquisition, "inspect_transport_resume_state", reject_partial
    )
    result = opening_module.inspect_same_opening_transport_resume(
        authorization_path, root=tmp_path
    )
    assert result["status"] == (
        "OPENING_INDETERMINATE_CORRUPT_OR_PARTIAL_TRANSACTION_NO_RESUME"
    )
    assert result["raw_transport_resume_allowed"] is False


@pytest.mark.parametrize(
    (
        "checkpoint",
        "expected_phase",
        "raw_allowed",
        "acquisition_allowed",
        "trusted_allowed",
    ),
    [
        (
            "raw_complete",
            "ACQUISITION_FINALIZATION_NETWORK_FREE",
            False,
            True,
            False,
        ),
        (
            "manifest",
            "TRUSTED_RECOMPUTE_NETWORK_FREE",
            False,
            False,
            True,
        ),
        (
            "trusted",
            "RECEIPT_COMPLETION_AFTER_FULL_REPLAY",
            False,
            False,
            True,
        ),
        (
            "receipt",
            "SIDECAR_RECOVERY_AFTER_FULL_VALIDATION",
            False,
            False,
            True,
        ),
        ("sidecar", "TERMINAL_COMPLETE", False, False, False),
    ],
)
def test_resume_status_phase_matrix(
    tmp_path,
    monkeypatch,
    checkpoint,
    expected_phase,
    raw_allowed,
    acquisition_allowed,
    trusted_allowed,
):
    authorization_path, state = _stub_resume_inspection(tmp_path, monkeypatch)
    monkeypatch.setattr(
        outcome_acquisition,
        "_assert_exact_acquisition_directory",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        opening_module,
        "_assert_exact_trusted_directory",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        opening_module,
        "_read_completed_receipt",
        lambda **_kwargs: {"status": "OPENED_AND_SCORED_ONCE"},
    )
    if checkpoint == "raw_complete":
        monkeypatch.setattr(
            outcome_acquisition,
            "inspect_transport_resume_state",
            lambda **_kwargs: {
                "classification": (
                    "RESUMABLE_RAW_COMPLETE_DERIVATION_NOT_PUBLISHED"
                ),
                "completed_request_count": 2,
                "missing_request_count": 0,
                "recoverable_pending_request_count": 0,
                "refetchable_nondurable_response_count": 0,
                "attempt_count": 1,
            },
        )
    else:
        state["acquisition_manifest"].parent.mkdir(parents=True)
        state["acquisition_manifest"].write_bytes(b"manifest")
        if checkpoint in {"trusted", "receipt", "sidecar"}:
            state["availability_registry"].parent.mkdir()
        if checkpoint in {"receipt", "sidecar"}:
            state["receipt"].write_bytes(b"receipt")
        if checkpoint == "sidecar":
            state["receipt_sha256"].write_bytes(b"sidecar")

    result = opening_module.inspect_same_opening_transport_resume(
        authorization_path, root=tmp_path
    )
    assert result["resume_phase"] == expected_phase
    assert result["raw_transport_resume_allowed"] is raw_allowed
    assert (
        result["network_free_acquisition_finalization_allowed"]
        is acquisition_allowed
    )
    assert result["network_free_trusted_completion_allowed"] is trusted_allowed


@pytest.mark.parametrize(
    "forbidden_key",
    tuple(key for key in _RAW_RESUME_FORBIDDEN_STATE_KEYS if key != "receipt"),
)
def test_status_forbids_raw_resume_after_any_derived_or_trusted_publication(
    tmp_path, monkeypatch, forbidden_key,
):
    authorization_path, state = _stub_resume_inspection(tmp_path, monkeypatch)
    state[forbidden_key].parent.mkdir(parents=True, exist_ok=True)
    state[forbidden_key].write_bytes(b"published")
    result = opening_module.inspect_same_opening_transport_resume(
        authorization_path, root=tmp_path
    )
    if forbidden_key == "acquisition_manifest":
        assert result["status"] == (
            "OPENING_INDETERMINATE_INVALID_ACQUISITION_BUNDLE_NO_RESUME"
        )
        return
    assert result["status"] == (
        "OPENING_INDETERMINATE_DERIVED_OR_TRUSTED_OUTPUT_EXISTS_NO_RESUME"
    )
    assert result["raw_transport_resume_allowed"] is False
    expected = {forbidden_key}
    if forbidden_key in {
        "acquisition_request_map", "temporal_outcomes", "external_outcomes"
    }:
        expected.add("acquisition_directory")
    if forbidden_key in acquisition_contract.TRUSTED_STATE_KEYS:
        expected.add("trusted_directory")
    assert set(result["forbidden_existing_outputs"]) == expected
def test_authorization_freeze_then_preflight_allows_only_its_own_untracked_file(
    tmp_path, monkeypatch,
):
    def git(*arguments):
        subprocess.run(
            ["git", *arguments], cwd=tmp_path, check=True,
            text=True, capture_output=True,
        )

    git("init")
    git("config", "user.email", "fixture@example.test")
    git("config", "user.name", "Fixture")
    relative_paths = (
        "protocol.json", "development.csv", "external.csv", "lock.json",
        "suite.json", "inputs.json", "panel_spec.json", "candidates.csv",
        "candidate_provenance.json", "candidate_snapshot_index.json",
        "development_replay.json", "protocol_seal.json", "inference_gate.json",
        "inference_amendment.json", "inference_amendment_seal.json",
        "outcome_qc_policy.json",
    )
    paths = {name: tmp_path / name for name in relative_paths}
    for name, path in paths.items():
        path.write_text(f"{name}\n", encoding="utf-8")
    chronology_path = (
        tmp_path / "outputs" / "prelabel" / "route_a_prelabel_chronology_v1.json"
    )
    chronology_path.parent.mkdir(parents=True)
    chronology_path.write_text("chronology\n", encoding="utf-8")
    replay_entrypoint = tmp_path / "scripts" / "27_verify_development_replay.py"
    replay_entrypoint.parent.mkdir(parents=True)
    replay_entrypoint.write_text(
        "raise SystemExit(0)\n", encoding="utf-8"
    )
    git("add", *paths.values(), replay_entrypoint, chronology_path)
    git("commit", "-m", "sealed pre-authorization fixture")

    development = pd.DataFrame({"site_no": ["01234567"]})
    external = pd.DataFrame({"site_no": ["07654321"]})
    protocol_info = {
        "protocol_sha256": sha256_file(paths["protocol.json"]),
        "authoritative_commit": "fixture-authoritative",
        "authoritative_markdown_sha256": "a" * 64,
        "amendments_sha256": "b" * 64,
        "target_start": "2021-01-01",
        "target_end": "2023-12-31",
        "document": {"primary_inference_contract": {"fixture": True}},
        "seal": {
            "path": paths["protocol_seal.json"],
            "sha256": sha256_file(paths["protocol_seal.json"]),
            "final_commit": "f" * 40,
        },
    }
    registries = {
        "development": development,
        "external": external,
        "development_sha256": sha256_file(paths["development.csv"]),
        "external_sha256": sha256_file(paths["external.csv"]),
        "lock_sha256": sha256_file(paths["lock.json"]),
        "development_panel_spec": paths["panel_spec.json"],
        "candidate_table": paths["candidates.csv"],
        "candidate_provenance": paths["candidate_provenance.json"],
        "candidate_snapshot_index": paths["candidate_snapshot_index.json"],
    }
    runtime_sha = "e" * 64
    suite = {
        "sha256": sha256_file(paths["suite.json"]),
        "runtime_sha256": runtime_sha,
        "feature_order": ("WTEMP", "FLOW"),
        "required_models": {
            "temporal": ("Persistence",), "external": ("Persistence",),
        },
    }
    inputs = {
        "sha256": sha256_file(paths["inputs.json"]),
        "history_start": "2020-12-01",
        "document": {},
    }
    monkeypatch.setattr(
        opening_module, "validate_protocol", lambda *_args, **_kwargs: protocol_info
    )
    monkeypatch.setattr(
        opening_module, "validate_registry_lock", lambda **_kwargs: registries
    )
    monkeypatch.setattr(
        opening_module, "validate_model_suite", lambda *_args, **_kwargs: suite
    )
    monkeypatch.setattr(
        opening_module, "validate_prelabel_inputs", lambda *_args, **_kwargs: inputs
    )
    outcome_qc_policy_document = {
        "format": "thermoroute.route-a-outcome-qc-policy.v1",
        "status": "FROZEN_PRELABEL_OUTCOME_FREE",
        "policy_id": "fixture-outcome-qc-policy",
    }
    amendment_document = {
        "format": "thermoroute.route-a-inference-amendment.v1",
        "amendment_id": "route-a-prelabel-inference-scope-014",
        "additional_preopen_gates": {
            "outcome_qc_policy": {
                "path": "outcome_qc_policy.json",
                "sha256": sha256_file(paths["outcome_qc_policy.json"]),
            },
        },
    }
    amendment_seal_document = {"final_prelabel_commit": "4" * 40}
    gate_document = {
        "format": "thermoroute.route-a-inference-gate.v1",
        "status": "FAIL_CLOSED_DESCRIPTIVE_ONLY",
        "claim_eligible": False,
        "analysis_mode": "FIXED_COHORT_DESCRIPTIVE_ONLY",
        "policy_sha256": "7" * 64,
    }
    monkeypatch.setattr(
        opening_module,
        "validate_inference_amendment",
        lambda *_args, **_kwargs: amendment_document,
    )
    monkeypatch.setattr(
        opening_module,
        "validate_inference_amendment_seal",
        lambda *_args, **_kwargs: amendment_seal_document,
    )
    monkeypatch.setattr(
        opening_module,
        "validate_inference_gate_document",
        lambda *_args, **_kwargs: gate_document,
    )
    monkeypatch.setattr(
        opening_module,
        "validate_outcome_qc_policy",
        lambda *_args, **_kwargs: outcome_qc_policy_document,
    )
    replay_document = {
        "format": "thermoroute.route-a-development-replay.v1",
        "status": "PASS_FULL_DEVELOPMENT_REPLAY_NO_CONFIRMATION_DATA",
    }
    monkeypatch.setattr(
        opening_module,
        "validate_development_replay_receipt",
        lambda *_args, **_kwargs: replay_document,
    )
    chronology_document = {
        "format": "thermoroute.route-a-prelabel-chronology.v1",
        "status": "PASS_REPOSITORY_INTERNAL_PRELABEL_ORDER",
        "order": {
            "model_freeze_commit": "1" * 40,
            "input_evidence_commit": "2" * 40,
            "receipt_creation_base_commit": "3" * 40,
            "strict_order_verified": True,
        },
        "paths": {
            "protocol_seal": "protocol_seal.json",
            "model_suite": "suite.json",
            "development_replay": "development_replay.json",
            "candidate_table": "candidates.csv",
            "candidate_provenance": "candidate_provenance.json",
            "candidate_snapshot_index": "candidate_snapshot_index.json",
            "external_registry": "external.csv",
            "external_lock": "lock.json",
            "input_manifest": "inputs.json",
        },
        "protocol_history": {
            "seal": {
                "path": "protocol_seal.json",
                "sha256": sha256_file(paths["protocol_seal.json"]),
            },
            "original_commit": "fixture-authoritative",
            "final_prelabel_commit": "f" * 40,
        },
        "external_timestamp_or_public_preregistration": False,
        "independent_custodian_or_worm_storage": False,
        "evidence_scope": "fixture repository-internal honest-owner chronology",
    }
    monkeypatch.setattr(
        opening_module,
        "validate_prelabel_chronology",
        lambda *_args, **_kwargs: chronology_document,
    )
    runtime = {"format": "fixture-runtime", "runtime_sha256": runtime_sha}
    fixed_code = {"format": "fixture-fixed-code", "sha256": "f" * 64}
    monkeypatch.setattr(
        opening_module, "_route_a_environment_contract", lambda _root: runtime
    )
    monkeypatch.setattr(
        opening_module, "_fixed_code_identity", lambda _root: fixed_code
    )
    authorization = tmp_path / "data_usgs" / "confirmatory_opening_authorization_v1.json"
    frozen = freeze_opening_authorization(
        authorization,
        root=tmp_path,
        protocol_path=paths["protocol.json"],
        development_registry=paths["development.csv"],
        external_registry=paths["external.csv"],
        external_lock=paths["lock.json"],
        model_suite=paths["suite.json"],
        input_manifest=paths["inputs.json"],
        development_replay_receipt=paths["development_replay.json"],
        prelabel_chronology_receipt=chronology_path,
        inference_gate=paths["inference_gate.json"],
        inference_amendment=paths["inference_amendment.json"],
        inference_amendment_seal=paths["inference_amendment_seal.json"],
        outcome_qc_policy=paths["outcome_qc_policy.json"],
    )
    assert frozen["source"]["post_freeze_allowed_git_status"] == (
        "?? data_usgs/confirmatory_opening_authorization_v1.json"
    )
    dry_run = validate_authorization(authorization, root=tmp_path)
    assert dry_run["authorization"]["status"] == "AUTHORIZED_LABELS_STILL_SEALED"
    assert dry_run["development_replay"] == replay_document
    assert dry_run["prelabel_chronology"] == chronology_document
    assert dry_run["inference_gate"] == gate_document
    assert dry_run["outcome_qc_policy"] == outcome_qc_policy_document
    assert not Path(dry_run["intent_path"]).exists()

    authorization_original = authorization.read_bytes()
    authorization.chmod(0o644)
    missing_gate = json.loads(authorization_original)
    missing_gate.pop("inference_gate")
    missing_gate.pop("authorization_self_sha256")
    missing_gate["authorization_self_sha256"] = opening_module.sha256_json(
        missing_gate
    )
    authorization.write_text(json.dumps(missing_gate), encoding="utf-8")
    with pytest.raises(OpeningContractError, match="lacks inference-gate binding"):
        validate_authorization(authorization, root=tmp_path)
    authorization.write_bytes(authorization_original)
    authorization.chmod(0o444)

    chronology_original = chronology_path.read_bytes()
    chronology_path.write_text("tampered chronology\n", encoding="utf-8")
    with pytest.raises(OpeningContractError, match="chronology receipt checksum"):
        validate_authorization(authorization, root=tmp_path)
    chronology_path.write_bytes(chronology_original)

    chronology_document["paths"]["input_manifest"] = "another-inputs.json"
    with pytest.raises(OpeningContractError, match="another protocol/model/input"):
        validate_authorization(authorization, root=tmp_path)
    chronology_document["paths"]["input_manifest"] = "inputs.json"

    gate_original = paths["inference_gate.json"].read_bytes()
    paths["inference_gate.json"].write_text("tampered gate\n", encoding="utf-8")
    with pytest.raises(OpeningContractError, match="inference gate checksum"):
        validate_authorization(authorization, root=tmp_path)
    paths["inference_gate.json"].write_bytes(gate_original)

    original = authorization.read_bytes()
    tampered = json.loads(original)
    tampered["created_at_utc"] = "2099-01-01T00:00:00+00:00"
    authorization.chmod(0o644)
    authorization.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(OpeningContractError, match="self-hash"):
        validate_authorization(authorization, root=tmp_path)
    authorization.write_bytes(original)
    authorization.chmod(0o444)

    (tmp_path / "another-untracked.txt").write_text("not allowed", encoding="utf-8")
    with pytest.raises(OpeningContractError, match="only the create-only untracked"):
        validate_authorization(authorization, root=tmp_path)


def test_exclusive_json_never_replaces_existing_bytes(tmp_path):
    path = tmp_path / "lock.json"
    exclusive_create_json(path, {"first": True})
    original = path.read_bytes()
    with pytest.raises(OpeningAlreadyStarted):
        exclusive_create_json(path, {"first": False})
    assert path.read_bytes() == original


def test_raw_acquisition_rejects_proxy_or_custom_ca_environment(
    tmp_path, monkeypatch,
):
    allowed = {
        "PATH": os.defpath,
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "TMPDIR": str(tmp_path.resolve()),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
    }
    monkeypatch.setattr(acquisition_contract.os, "environ", allowed.copy())
    acquisition_contract._validate_acquisition_environment()
    acquisition_contract.os.environ["HTTPS_PROXY"] = "http://attacker.invalid:8080"
    acquisition_contract.os.environ["SSL_CERT_FILE"] = "/tmp/attacker-ca.pem"
    with pytest.raises(
        acquisition_contract.AcquisitionContractError,
        match="complete allowlist",
    ):
        acquisition_contract._validate_acquisition_environment()


def test_hashed_runtime_lock_requires_sha256_on_every_requirement(tmp_path):
    checked = _validate_hashed_requirements_lock(
        ROOT / "requirements-lock-py312-hashed.txt"
    )
    assert checked["every_stanza_hashed"] is True
    assert checked["requirement_stanza_count"] > 0
    assert checked["sha256_hash_count"] >= checked["requirement_stanza_count"]

    incomplete = tmp_path / "incomplete-lock.txt"
    incomplete.write_text(
        "numpy==2.2.6 \\\n+    --hash=sha256:" + "a" * 64 + "\n"
        "pandas==2.3.3\n",
        encoding="utf-8",
    )
    with pytest.raises(OpeningContractError, match="not fully SHA-256 bound"):
        _validate_hashed_requirements_lock(incomplete)


def _architecture(**overrides):
    kwargs = {
        "n_vars": 7,
        "n_stations": 120,
        "n_phys": 4,
        "delta_scale": 1.0,
        "safety_anchor": "damped",
    }
    kwargs.update(overrides)
    return {
        "architecture": {"kwargs": kwargs},
        "horizons": [1, 3, 7],
    }


def test_control_registry_allows_only_exact_one_factor_interventions():
    metadata = {"ThermoRoute": _architecture()}
    entries = {}
    for model_id, intervention in CONTROL_INTERVENTIONS.items():
        metadata[model_id] = _architecture(**intervention)
        entries[model_id] = {
            "model_id": model_id,
            "member_count": 1,
            "intervention": dict(intervention),
        }
    _validate_temporal_controls(entries, metadata)

    metadata["TR-noRouter"] = _architecture(use_router=False, use_moe=False)
    with pytest.raises(OpeningContractError, match="beyond its intervention"):
        _validate_temporal_controls(entries, metadata)


def test_protocol_is_authoritative_for_temporal_and_external_model_suites():
    protocol = json.loads((ROOT / "protocols" / "route_a_confirmatory_v1.json").read_text())
    temporal = _required_models(protocol, cohort="temporal")
    external = _required_models(protocol, cohort="external")
    assert temporal == tuple(
        protocol["primary_inference_contract"]["primary_models"]
        + protocol["primary_inference_contract"][
            "mandatory_exploratory_architecture_controls"
        ]
    )
    assert external == tuple(protocol["primary_inference_contract"]["primary_models"])
    assert "TR-noRouter" in temporal and "TR-noRouter" not in external


def _probability_evaluation_fixture(*, single_class: bool = False):
    protocol = json.loads(
        (ROOT / "protocols" / "route_a_confirmatory_v1.json").read_text()
    )
    sites = ("01000001", "02000001")
    reference_rows = []
    for site_index, site in enumerate(sites):
        for index, date in enumerate(pd.date_range("2006-01-01", periods=365)):
            reference_rows.append({
                "site_id": site,
                "DATE": date,
                "WTEMP": float((index + site_index) % 3),
            })
    reference_panel = pd.DataFrame(reference_rows)
    thresholds = {site: 0.5 for site in sites}
    temporal_reference = fit_frozen_seasonal_event_reference(
        reference_panel, thresholds, pooled=False
    )
    external_reference = fit_frozen_seasonal_event_reference(
        reference_panel, 0.5, pooled=True
    )
    required = ("Persistence", "ThermoRoute")
    suite = {
        "required_models": {"temporal": required, "external": required},
        "metadata": {
            "temporal": {
                "ThermoRoute": {
                    "event_thresholds": thresholds,
                    "event_reference_climatology": temporal_reference,
                }
            },
            "external": {
                "ThermoRoute": {
                    "event_thresholds": {"__pooled__": 0.5},
                    "event_reference_climatology": external_reference,
                }
            },
        },
    }
    predictions: dict[str, pd.DataFrame] = {}
    availability_rows = []
    for cohort in ("temporal", "external"):
        cohort_rows = []
        for horizon in (1, 3, 7):
            for site_index, (site, count) in enumerate(zip(sites, (100, 200))):
                issue_dates = pd.date_range("2021-01-01", periods=count)
                y_true = 1.0 if single_class or site_index == 1 else 0.0
                for model in required:
                    learned = model == "ThermoRoute"
                    for issue_date in issue_dates:
                        cohort_rows.append({
                            "model": model,
                            "scope": f"route_a_{cohort}_confirmation",
                            "feature_set": "fixture",
                            "seed": -1,
                            "site_id": site,
                            "horizon": horizon,
                            "split": "confirm",
                            "issue_date": issue_date,
                            "target_date": issue_date + pd.Timedelta(days=horizon),
                            "y_true": y_true,
                            "y_pred": y_true,
                            "q05": (
                                (0.0 if single_class else -1.0)
                                if learned and site_index == 0 else
                                1.5 if learned else np.nan
                            ),
                            "q50": (
                                1.0 if learned and site_index == 0 else
                                2.0 if learned else np.nan
                            ),
                            "q95": (
                                2.0 if learned and site_index == 0 else
                                2.5 if learned else np.nan
                            ),
                            "p_exceed": (
                                (0.2 if site_index == 0 else 0.8)
                                if learned else np.nan
                            ),
                        })
                availability_rows.append({
                    "cohort": cohort,
                    "site_no": site,
                    "horizon": horizon,
                    "n_valid_targets": count,
                    "reportable": True,
                })
        predictions[cohort] = pd.DataFrame(cohort_rows)
    return predictions, suite, pd.DataFrame(availability_rows), protocol


def test_probability_metrics_are_station_balanced_not_row_pooled():
    predictions, suite, availability, protocol = _probability_evaluation_fixture()
    result = _probabilistic_evaluation(
        trusted_predictions=predictions,
        suite=suite,
        availability=availability,
        protocol=protocol,
    )
    row = next(
        item for item in result["rows"]
        if item["cohort"] == "temporal"
        and item["model"] == "ThermoRoute"
        and item["horizon"] == 1
    )
    assert row["coverage_90"] == pytest.approx(0.5)
    assert row["coverage_90"] != pytest.approx(100 / 300)
    assert row["event_rate"] == pytest.approx(0.5)
    assert row["event_rate"] != pytest.approx(200 / 300)
    assert row["n_forecasts"] == 300 and row["n_sites"] == 2
    assert row["minimum_site_total_weight"] == pytest.approx(0.5)
    assert row["maximum_site_total_weight"] == pytest.approx(0.5)
    assert sum(
        item["station_balanced_weight"] for item in row["reliability_bins"]
    ) == pytest.approx(1.0)
    builtin = next(
        item for item in result["rows"]
        if item["cohort"] == "temporal"
        and item["model"] == "Persistence"
        and item["horizon"] == 1
    )
    assert builtin["status"] == "NOT_AVAILABLE"
    assert result["rev_status"] == (
        "REV_NOT_EVALUATED_NO_PREDECLARED_COST_LOSS_RATIOS"
    )


def test_probability_evaluation_reports_zero_reportable_cells_explicitly():
    predictions, suite, availability, protocol = _probability_evaluation_fixture()
    availability["n_valid_targets"] = 99
    availability["reportable"] = False
    result = _probabilistic_evaluation(
        trusted_predictions=predictions,
        suite=suite,
        availability=availability,
        protocol=protocol,
    )
    learned = [row for row in result["rows"] if row["model"] == "ThermoRoute"]
    assert learned and all(row["status"] == "NOT_ESTIMABLE" for row in learned)
    assert all(row["reason"] == "NO_STATION_HAS_100_COMMON_TARGETS" for row in learned)


def test_probability_single_class_metrics_are_null_with_per_metric_reasons():
    predictions, suite, availability, protocol = _probability_evaluation_fixture(
        single_class=True
    )
    result = _probabilistic_evaluation(
        trusted_predictions=predictions,
        suite=suite,
        availability=availability,
        protocol=protocol,
    )
    row = next(
        item for item in result["rows"]
        if item["cohort"] == "external"
        and item["model"] == "ThermoRoute"
        and item["horizon"] == 7
    )
    for metric in (
        "auroc", "auprc", "calibration_intercept", "calibration_slope"
    ):
        assert row[metric] is None
        assert row["undefined_metric_reasons"][metric] == (
            "SINGLE_CLASS_RETAINED_OUTCOMES"
        )


def test_quality_audit_closes_every_multi_series_conflict_constituent():
    def payload(site: str) -> bytes:
        return (
            "agency_cd\tsite_no\tdatetime\t123_00010_00003\t"
            "123_00010_00003_cd\t456_00010_00003\t456_00010_00003_cd\n"
            "5s\t15s\t20d\t14n\t10s\t14n\t10s\n"
            f"USGS\t{site}\t2021-01-01\t4.2\tP\t\t\n"
            f"USGS\t{site}\t2021-01-02\t5.1\tA\t5.2\tX-UNKNOWN\n"
        ).encode()

    normalized = {}
    requests = []
    for cohort, site, digest in (
        ("temporal", "01000001", "a"),
        ("external", "02000001", "b"),
    ):
        raw = payload(site)
        normalized[cohort] = opening_module.parse_nwis_confirmatory_daily(
            raw, site_no=site, start="2021-01-01", end="2021-01-02"
        )
        requests.append({
            "cohort": cohort,
            "site_no": site,
            "request_sha256": digest * 64,
            "response_sha256": digest.upper().lower() * 64,
            "series_registry": opening_module.nwis_confirmatory_series_registry(raw),
        })
    protocol = json.loads(
        (ROOT / "protocols" / "route_a_confirmatory_v1.json").read_text()
    )
    audit = _build_outcome_quality_audit(
        normalized=normalized,
        request_rows=requests,
        protocol=protocol,
    )
    temporal = [
        row for row in audit["conflict_constituents"]
        if row["cohort"] == "temporal" and row["variable"] == "WTEMP"
    ]
    assert len(temporal) == 2
    assert {row["raw_qualifier"] for row in temporal} == {"A", "X-UNKNOWN"}
    assert {row["raw_value"] for row in temporal} == {"5.1", "5.2"}
    assert {row["value_column"] for row in temporal} == {
        "123_00010_00003", "456_00010_00003",
    }
    assert all(row["qualifier_column"].endswith("_cd") for row in temporal)
    assert all(row["response_sha256"] == "a" * 64 for row in temporal)
    temporal_status = [
        row for row in audit["value_status_day_counts"]
        if row["cohort"] == "temporal"
        and row["variable"] == "WTEMP"
        and row["value_status"] == "MULTIPLE_FINITE_SERIES_CONFLICT"
    ]
    assert temporal_status == [{
        "cohort": "temporal",
        "site_no": "01000001",
        "variable": "WTEMP",
        "value_status": "MULTIPLE_FINITE_SERIES_CONFLICT",
        "day_count": 1,
    }]
    counted = [
        row for row in audit["counts"]
        if row["cohort"] == "temporal" and row["variable"] == "WTEMP"
    ]
    assert any(
        row["raw_qualifier"] == "P"
        and row["value_status"] == "RETAINED_FINITE_VALUE"
        for row in counted
    )
    assert any(row["raw_qualifier"] == "X-UNKNOWN" for row in counted)


def test_five_test_statistics_are_recomputed_with_cluster_sign_flips():
    protocol = json.loads((ROOT / "protocols" / "route_a_confirmatory_v1.json").read_text())
    protocol["primary_inference_contract"]["confidence_interval"]["draws"] = 100
    protocol["primary_inference_contract"]["one_sided_p_value"][
        "randomisation_draws"
    ] = 100
    dates = pd.date_range("2021-01-01", periods=3, freq="D")
    frames = []
    for site_index, site in enumerate(("01000001", "02000001")):
        for horizon in (1, 3, 7):
            truth = np.asarray([10.0, 11.0, 12.0]) + site_index
            for model, error in (
                ("ThermoRoute", 0.1),
                ("DampedPersistence", 0.3),
                ("LightGBM", 0.12),
            ):
                frames.append(pd.DataFrame({
                    "model": model,
                    "site_id": site,
                    "horizon": horizon,
                    "issue_date": dates,
                    "target_date": dates + pd.to_timedelta(horizon, unit="D"),
                    "y_true": truth,
                    "y_pred": truth + error,
                }))
    predictions = pd.concat(frames, ignore_index=True)
    registry = pd.DataFrame({
        "site_no": ["01000001", "02000001"],
        "huc2": ["01", "02"],
        "huc_metadata_status": [
            "USGS_SNAPSHOT_SITE_NO_MATCH", "USGS_SNAPSHOT_SITE_NO_MATCH"
        ],
    })
    result = compute_confirmatory_statistics(
        predictions, registry, protocol, minimum_targets=1
    )
    assert len(result["tests"]) == 5
    assert all(row["status"] == "ESTIMABLE" for row in result["tests"])
    assert all(0.0 < row["p_one_sided_raw"] <= 1.0 for row in result["tests"])
    assert all(0.0 < row["p_holm"] <= 1.0 for row in result["tests"])


def test_snapshot_verifier_binds_request_metadata_and_response(tmp_path):
    request = {
        "schema_version": 1,
        "provider": "usgs-nwis-confirmatory-dv",
        "method": "GET",
        "url": (
            "https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites=01234567"
            "&startDT=2020-12-01&endDT=2023-12-31"
            "&parameterCd=00010%2C00060%2C00065&statCd=00003&siteStatus=all"
        ),
        "headers": {},
    }
    request_sha = sha256_bytes(canonical_json_bytes(request))
    directory = tmp_path / request["provider"] / request_sha
    directory.mkdir(parents=True)
    payload = (
        b"agency_cd\tsite_no\tdatetime\t123_00010_00003\n"
        b"5s\t15s\t20d\t14n\n"
        b"USGS\t01234567\t2021-01-01\t4.2\n"
    )
    response = directory / "response.bin"
    response.write_bytes(payload)
    metadata = {
        "schema_version": 1,
        "opening_id": "fixture-opening",
        "authorization_sha256": "a" * 64,
        "work_order_self_sha256": "b" * 64,
        "request_ledger_sha256": "c" * 64,
        "attempt_number": 1,
        "request": request,
        "request_sha256": request_sha,
        "retrieved_at_utc": "2026-07-21T00:00:00+00:00",
        "http_status": 200,
        "response_headers": {},
        "final_url": request["url"],
        "byte_count": len(payload),
        "response_sha256": sha256_bytes(payload),
        "response_file": "response.bin",
        "maximum_response_bytes_per_request": (
            acquisition_contract.MAX_CONFIRMATORY_NWIS_RESPONSE_BYTES
        ),
    }
    metadata_path = directory / "metadata.json"
    metadata_path.write_bytes(canonical_json_bytes(metadata))
    record = {
        "provider": request["provider"],
        "request_sha256": request_sha,
        "response_sha256": sha256_bytes(payload),
        "retrieved_at_utc": metadata["retrieved_at_utc"],
        "byte_count": len(payload),
        "attempt_number": 1,
        "request": request,
        "metadata_path": metadata_path.relative_to(tmp_path).as_posix(),
        "metadata_sha256": sha256_file(metadata_path),
        "response_path": response.relative_to(tmp_path).as_posix(),
        "series_registry": opening_module.nwis_confirmatory_series_registry(
            payload
        ),
    }
    index = tmp_path / "snapshot_index.json"
    index.write_bytes(canonical_json_bytes({
        "schema_version": 1, "snapshot_count": 1, "records": [record],
    }))
    records = _verify_snapshot_index(tmp_path, index, prelabel=False)
    _verify_opened_nwis_index(
        records,
        expected_sites={"01234567"},
        history_start="2020-12-01",
        target_end="2023-12-31",
    )

    metadata["byte_count"] += 1
    metadata_path.write_bytes(canonical_json_bytes(metadata))
    with pytest.raises(OpeningContractError, match="metadata does not bind"):
        _verify_snapshot_index(tmp_path, index, prelabel=False)


def _trusted_fixture_frame() -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=2, freq="D")
    return pd.DataFrame({
        "model": ["ThermoRoute"] * 2,
        "scope": ["route_a_temporal_confirmation"] * 2,
        "feature_set": ["WTEMP+FLOW+TEMP+PRCP+RHMEAN+DH+WDSP"] * 2,
        "seed": [-1, -1],
        "site_id": ["01234567"] * 2,
        "horizon": [1, 1],
        "split": ["confirm", "confirm"],
        "issue_date": dates,
        "target_date": dates + pd.Timedelta(days=1),
        "y_true": [10.0, 11.0],
        "y_pred": [10.2, 10.8],
        "q05": [9.0, 9.5],
        "q50": [10.2, 10.8],
        "q95": [11.5, 12.0],
        "p_exceed": [0.2, 0.3],
    })


def test_worker_prediction_cannot_substitute_labels_for_frozen_model_output():
    trusted = _trusted_fixture_frame()
    digest = _assert_worker_predictions_equal_trusted(
        trusted.copy(), trusted, cohort="temporal"
    )
    assert len(digest) == 64
    cheating = trusted.copy()
    cheating["y_pred"] = cheating["y_true"]
    with pytest.raises(OpeningContractError, match="differs from trusted scorer"):
        _assert_worker_predictions_equal_trusted(
            cheating, trusted, cohort="temporal"
        )


def test_frozen_calibration_is_applied_after_member_average_and_external_is_pooled():
    metadata = {
        "conformal_offsets": {"01234567|1": 0.2, "01234567|3": 0.3,
                              "01234567|7": 0.4},
        "event_thresholds": {"01234567": 20.0},
        "event_calibrators": {
            str(horizon): {"intercept": 0.0, "slope": 1.0, "constant": None}
            for horizon in (1, 3, 7)
        },
    }
    q05, q50, q95, probability = _frozen_calibration(
        metadata,
        np.asarray(["01234567"]),
        (1, 3, 7),
        np.asarray([[9.0, 8.0, 7.0]]),
        np.asarray([[10.0, 10.0, 10.0]]),
        np.asarray([[11.0, 12.0, 13.0]]),
        np.asarray([[0.2, 0.4, 0.8]]),
        external=False,
        label="fixture",
    )
    assert np.allclose(q05, [[8.8, 7.7, 6.6]])
    assert np.allclose(q50, [[10.0, 10.0, 10.0]])
    assert np.allclose(q95, [[11.2, 12.3, 13.4]])
    assert np.allclose(probability, [[0.2, 0.4, 0.8]])

    with pytest.raises(OpeningContractError, match="external event threshold is not pooled"):
        _frozen_calibration(
            metadata,
            np.asarray(["99999999"]),
            (1, 3, 7),
            q05, q50, q95, probability,
            external=True,
            label="external-fixture",
        )


class _TinyWindowedData:
    def __init__(self):
        self.horizons = (1, 3, 7)
        self.y = np.asarray([[10.5, 11.0, 11.5], [12.5, 13.0, 13.5]])
        self.station = np.zeros(2, dtype=int)
        self.issue_date = np.asarray(["2021-01-01", "2021-01-02"], dtype="datetime64[D]")
        self.target_date = self.issue_date[:, None] + np.asarray(self.horizons) * np.timedelta64(1, "D")
        generator = np.random.default_rng(7)
        self.X = generator.normal(size=(2, 5, 2)).astype("float32")
        self.Mask = np.ones_like(self.X, dtype="float32")
        self.wtemp_t = np.asarray([10.0, 12.0], dtype="float32")
        self.clim_t = np.asarray([9.0, 9.5], dtype="float32")

    def batch(self, index, device="cpu"):
        return {
            "X": torch.as_tensor(self.X[index], device=device),
            "Mask": torch.as_tensor(self.Mask[index], device=device),
            "station": torch.as_tensor(self.station[index], dtype=torch.long, device=device),
            "wtemp_t": torch.as_tensor(self.wtemp_t[index], device=device),
            "clim_t": torch.as_tensor(self.clim_t[index], device=device),
        }


def test_trusted_sequence_scorer_loads_all_members_and_emits_one_ensemble_row(tmp_path):
    architecture = {
        "class": "thermoroute.train.LSTMForecaster",
        "kwargs": {
            "n_vars": 2, "horizons": [1, 3, 7], "d": 4, "layers": 1,
            "dropout": 0.0, "context": 5, "n_stations": 1,
            "station_agnostic": False, "station_embed_dim": 2,
        },
    }
    metadata = {
        "run_id": "fixture",
        "output_head_schema": neural_output_head_schema(),
        "architecture": architecture,
        "feature_order": ["WTEMP", "FLOW"],
        "horizons": [1, 3, 7],
        "station_to_index": {"01234567": 0},
        "preprocessing": {"fixture": True},
        "event_thresholds": {"01234567": 20.0},
        "event_calibrators": {
            str(horizon): {"intercept": 0.0, "slope": 1.0, "constant": None}
            for horizon in (1, 3, 7)
        },
        "conformal_offsets": {
            f"01234567|{horizon}": 0.1 for horizon in (1, 3, 7)
        },
        "source_sha256": "s",
        "panel_sha256": "p",
        "registry_sha256": "r",
        "runtime_sha256": "t",
    }
    members = {}
    for seed in range(5):
        torch.manual_seed(seed)
        members[f"seed{seed}"] = LSTMForecaster(**architecture["kwargs"])
    directory = save_inference_bundle(
        tmp_path / "bundle",
        members=members,
        metadata=metadata,
        expected_member_count=5,
    )
    _, frozen_metadata = load_inference_bundle(directory, expected_member_count=5)
    scored = _score_sequence_bundle(
        root=tmp_path,
        entry={"artifact": {"path": "bundle"}, "member_count": 5},
        metadata=frozen_metadata,
        wd=_TinyWindowedData(),
        station_order=("01234567",),
        cohort="temporal",
        model_id="LSTM",
        external=False,
        batch_size=1,
    )
    assert len(scored) == 2 * 3
    assert scored.seed.eq(-1).all()
    assert scored.q05.notna().all() and scored.p_exceed.between(0, 1).all()


def test_producer_development_binding_replays_through_opening_validator(tmp_path):
    predictions = _trusted_fixture_frame().copy()
    predictions["model"] = "LSTM"
    predictions["seed"] = 0
    predictions["split"] = "test"
    artifact = tmp_path / "development.parquet"
    predictions.to_parquet(artifact, index=False)
    lineage = {
        "panel_sha256": "a" * 64,
        "registry_sha256": "b" * 64,
        "source_sha256": "c" * 64,
    }
    config_sha = "d" * 64
    identity = RunIdentity(
        run_id="fixture-development",
        panel_sha256=lineage["panel_sha256"],
        registry_sha256=lineage["registry_sha256"],
        config_sha256=config_sha,
        source_sha256=lineage["source_sha256"],
        runtime_sha256="f" * 64,
    )
    seal_artifact(
        artifact,
        identity,
        kind="fixture_development_predictions",
        schema="thermoroute.predictions.v1",
    )
    binding = development_prediction_binding(
        tmp_path,
        artifact,
        predictions,
        max_abs_difference=0.0,
        atol=1e-5,
    )
    _validate_development_prediction_parity(
        {"development_prediction": binding, "config_sha256": config_sha},
        root=tmp_path,
        cohort="temporal",
        model_id="LSTM",
        lineage=lineage,
        member_count=1,
    )

    with pytest.raises(OpeningContractError, match="parity selection changed"):
        _validate_development_prediction_parity(
            {"development_prediction": binding, "config_sha256": config_sha},
            root=tmp_path,
            cohort="temporal",
            model_id="LSTM",
            lineage=lineage,
            member_count=2,
        )

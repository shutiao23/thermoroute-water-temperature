from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "26_validate_claims.py"
PRODUCTION_REGISTRY = ROOT / "protocols" / "route_a_claim_registry_v1.json"
PRODUCTION_PROTOCOL = ROOT / "protocols" / "route_a_confirmatory_v1.json"
PRODUCTION_OUTCOME_QC_POLICY = (
    ROOT / "protocols" / "route_a_outcome_qc_policy_v1.json"
)


def _module():
    spec = importlib.util.spec_from_file_location("claim_validator_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _v1_registry(path: Path, *, status: str = "PENDING_SINGLE_OPENING") -> Path:
    path.write_text(
        json.dumps({
            "format": "thermoroute.route-a-claim-registry.v1",
            "documents": ["paper/*.md"],
            "claims": [{
                "claim_id": "C1",
                "status": status,
                "forbidden_regex": ["blind test"],
                "required_artifacts": ["outputs/receipt.json"],
            }],
        }),
        encoding="utf-8",
    )
    return path


def _v2_fixture(tmp_path: Path) -> tuple[object, Path, dict, dict]:
    module = _module()
    protocol_path = tmp_path / "protocols" / "route_a_confirmatory_v1.json"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_bytes(PRODUCTION_PROTOCOL.read_bytes())
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    registry = json.loads(PRODUCTION_REGISTRY.read_text(encoding="utf-8"))
    registry["protocol_binding"] = {
        "path": "protocols/route_a_confirmatory_v1.json",
        "sha256": _sha256(protocol_path),
    }
    registry["documents"] = [
        "paper/main.md",
        "outputs/confirmatory/route_a_*/trusted/report_v1.md",
    ]
    registry["required_documents"] = ["paper/main.md"]
    for constraint in registry["permanent_constraints"]:
        constraint["claim"]["render_targets"] = ["paper/main.md"]
    for spec in registry["result_claim_specs"]:
        spec["render_targets"] = ["paper/main.md"]
    registry_path = tmp_path / "protocols" / "claims.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    paper = tmp_path / "paper" / "main.md"
    paper.parent.mkdir()
    loaded = module._load_registry(registry_path)
    expected = module._expected_claims(
        registry=loaded, phase=module.PRE_PHASE, statistics=None
    )
    permanent = [
        expected[claim_id]["block"]
        for claim_id in loaded["required_permanent_coverage"]["claim_ids"]
    ]
    paper.write_bytes(b"Development evaluation only.\n\n" + b"\n\n".join(permanent))
    return module, registry_path, registry, protocol


def _state(namespace: str = "a" * 24) -> dict[str, str]:
    base = f"outputs/confirmatory/route_a_{namespace}"
    return {
        "namespace": namespace,
        "run_directory": base,
        "intent": f"{base}/opening_intent_v1.json",
        "statistics": f"{base}/trusted/statistics_v1.json",
        "outcome_qc_gate": f"{base}/trusted/outcome_qc_gate_v1.json",
        "report": f"{base}/trusted/report_v1.md",
        "receipt": f"{base}/opening_receipt_v1.json",
        "receipt_sha256": f"{base}/opening_receipt_v1.sha256",
    }


def _write_authorization(
    tmp_path: Path,
    registry: dict,
    *,
    inference_claim_eligible: bool = False,
) -> tuple[Path, dict[str, str]]:
    state = _state()
    policy_path = tmp_path / "protocols" / "route_a_outcome_qc_policy_v1.json"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_bytes(PRODUCTION_OUTCOME_QC_POLICY.read_bytes())
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    authorization = {
        "format": "thermoroute.route-a-opening-authorization.v1",
        "status": "AUTHORIZED_LABELS_STILL_SEALED",
        "source": {
            "authorization_path": "data_usgs/confirmatory_opening_authorization_v1.json"
        },
        "protocol": dict(registry["protocol_binding"]),
        "inference_gate": {
            "format": "thermoroute.route-a-inference-gate.v1",
            "status": (
                "PASS_CLAIM_ELIGIBLE"
                if inference_claim_eligible
                else "FAIL_CLOSED_DESCRIPTIVE_ONLY"
            ),
            "claim_eligible": inference_claim_eligible,
            "analysis_mode": "FIXED_COHORT_DESCRIPTIVE_ONLY",
        },
        "outcome_qc_policy": {
            "path": "protocols/route_a_outcome_qc_policy_v1.json",
            "sha256": _sha256(policy_path),
            "format": policy["format"],
            "policy_id": policy["policy_id"],
            "required": True,
        },
        "state_paths": state,
    }
    path = tmp_path / "data_usgs" / "confirmatory_opening_authorization_v1.json"
    path.parent.mkdir()
    path.write_text(json.dumps(authorization), encoding="utf-8")
    return path, state


def _holm(raw: list[float]) -> list[float]:
    order = sorted(range(len(raw)), key=lambda index: (raw[index], index))
    adjusted = [0.0] * len(raw)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(raw) - rank) * raw[index]))
        adjusted[index] = running
    return adjusted


def _statistics(protocol: dict, *, conflict: bool = False) -> dict:
    family = protocol["primary_inference_contract"]["confirmatory_family"]
    raw = [0.01] * 5
    holm = _holm(raw)
    rows = []
    for index, (test, adjusted) in enumerate(zip(family, holm)):
        margin = float(test["margin_c"])
        ci_high = margin - 0.01
        reject = adjusted <= 0.05
        if conflict and index == 0:
            # Keep the exact five-way Holm calculation coherent while making the
            # confidence-bound rule disagree with the p-value decision.
            ci_high = margin + 0.01
        rows.append({
            "test_id": test["test_id"],
            "candidate": test["candidate"],
            "reference": test["reference"],
            "horizon": int(test["horizon"]),
            "margin_c": margin,
            "effect_convention": "station_RMSE_ThermoRoute-minus-reference",
            "status": "ESTIMABLE",
            "median_effect_c": margin - 0.1,
            "ci_low_c": margin - 0.2,
            "ci_high_c": ci_high,
            "n_stations": 120,
            "n_clusters": 2,
            "win_rate": 0.8,
            "p_one_sided_raw": raw[index],
            "bootstrap_seed": test["bootstrap_seed"],
            "sign_flip_seed_legacy_ignored": test["sign_flip_seed"],
            "sign_flip_configurations": 4,
            "p_holm": adjusted,
            "reject_at_0_05": reject,
            "confidence_bound_supports_margin": ci_high < margin,
        })
    return {
        "format": "thermoroute.route-a-confirmatory-statistics.v1",
        "confidence_interval": {
            "method": family and protocol["primary_inference_contract"][
                "confidence_interval"
            ]["method"],
            "draws": protocol["primary_inference_contract"]["confidence_interval"][
                "draws"
            ],
        },
        "p_value": {
            "method": protocol["primary_inference_contract"]["one_sided_p_value"][
                "method"
            ],
            "maximum_configurations_for_frozen_cohort": protocol[
                "primary_inference_contract"
            ]["one_sided_p_value"]["maximum_configurations_for_frozen_cohort"],
            "monte_carlo_used": False,
            "assumption": protocol["primary_inference_contract"][
                "one_sided_p_value"
            ]["null_assumption"],
            "enumeration_rule": protocol["primary_inference_contract"][
                "one_sided_p_value"
            ]["enumeration_rule"],
            "legacy_seed_field": protocol["primary_inference_contract"][
                "one_sided_p_value"
            ]["legacy_seed_field"],
        },
        "multiplicity": "Holm step-down across exactly five tests",
        "tests": rows,
    }


def _post_fixture(
    module: object,
    tmp_path: Path,
    registry: dict,
    protocol: dict,
    monkeypatch: pytest.MonkeyPatch,
    *,
    conflict: bool = False,
    inference_claim_eligible: bool = False,
    outcome_qc_pass: bool = True,
) -> tuple[dict, dict[str, str]]:
    authorization_path, state = _write_authorization(
        tmp_path,
        registry,
        inference_claim_eligible=inference_claim_eligible,
    )
    for key in ("intent", "receipt"):
        path = tmp_path / state[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    report_path = tmp_path / state["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("# Receipt-bound trusted report\n", encoding="utf-8")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    policy_path = tmp_path / authorization["outcome_qc_policy"]["path"]
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    family = protocol["primary_inference_contract"]["confirmatory_family"]
    test_ids = [str(row["test_id"]) for row in family]
    outside_records = [] if outcome_qc_pass else [
        {"site_no": "fixture", "date": "2021-01-01", "wtemp_c": 51.0}
    ]
    components = {
        "target_plausibility_pass": outcome_qc_pass,
        "single_extreme_influence_pass": True,
        "leave_one_huc_direction_pass": True,
    }
    gate = {
        "format": "thermoroute.route-a-outcome-qc-gate.v1",
        "status": (
            "PASS_DIRECTIONAL_REPORTING_QC"
            if outcome_qc_pass
            else "FAIL_WITHHOLD_DIRECTIONAL_CLAIMS"
        ),
        "policy": {
            "path": authorization["outcome_qc_policy"]["path"],
            "sha256": authorization["outcome_qc_policy"]["sha256"],
            "policy_id": authorization["outcome_qc_policy"]["policy_id"],
        },
        "confirmatory_family_sha256": hashlib.sha256(
            json.dumps(
                family,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        "minimum_valid_targets_per_station_horizon": protocol[
            "availability_contract"
        ]["minimum_valid_targets_per_station_horizon"],
        "primary_statistics_filtered_or_recomputed_on_selected_rows": False,
        "models_retrained_or_recalibrated": False,
        "sites_or_primary_keys_removed_by_qc": False,
        "target_plausibility": {
            "outside_range_count": len(outside_records),
            "outside_range_records": outside_records,
            "outside_range_values_retained_in_primary_analysis": True,
            "pass": outcome_qc_pass,
        },
        "single_extreme_influence": [
            {"test_id": test_id, "pass": True} for test_id in test_ids
        ],
        "leave_one_huc_direction": [
            {"test_id": test_id, "pass": True} for test_id in test_ids
        ],
        "components": components,
        "pass": outcome_qc_pass,
        "directional_claims_allowed_by_outcome_qc": outcome_qc_pass,
        "failure_action": policy["decision"]["failure_action"],
    }
    gate["gate_self_sha256"] = hashlib.sha256(
        json.dumps(
            gate,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    gate_path = tmp_path / state["outcome_qc_gate"]
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    gate_binding = {
        "path": state["outcome_qc_gate"],
        "sha256": _sha256(gate_path),
    }

    statistics = _statistics(protocol, conflict=conflict)
    statistics["outcome_qc_gate"] = {
        **gate_binding,
        "format": gate["format"],
        "status": gate["status"],
        "pass": outcome_qc_pass,
        "directional_claims_allowed": outcome_qc_pass,
    }
    statistics_path = tmp_path / state["statistics"]
    statistics_path.parent.mkdir(parents=True, exist_ok=True)
    statistics_path.write_text(json.dumps(statistics), encoding="utf-8")
    receipt = {
        "artifacts": {
            "statistics": {
                "path": state["statistics"],
                "sha256": _sha256(statistics_path),
            },
            "outcome_qc_gate": gate_binding,
        },
        "formal_tests": statistics["tests"],
    }
    monkeypatch.setattr(
        module,
        "_validate_completed_receipt",
        lambda authorization_path, *, root, allow_gitless_archive: receipt,
    )
    return statistics, state


def test_legacy_v1_lints_but_empty_file_cannot_prove_completion(tmp_path):
    module = _module()
    paper = tmp_path / "paper" / "main.md"
    paper.parent.mkdir()
    paper.write_text("A 2019 blind test.\n", encoding="utf-8")
    receipt = tmp_path / "outputs" / "receipt.json"
    receipt.parent.mkdir()
    receipt.write_text("{}\n", encoding="utf-8")
    registry = _v1_registry(tmp_path / "claims.json")
    violations = module.validate_claims(
        root=tmp_path, registry_path=registry, require_complete=True
    )
    assert any("blind test" in value for value in violations)
    assert any("cannot establish a verified completed opening" in value for value in violations)


def test_legacy_v1_nonpending_fixture_remains_release_compatible(tmp_path):
    module = _module()
    paper = tmp_path / "paper" / "main.md"
    paper.parent.mkdir()
    paper.write_text("Scoped development language.\n", encoding="utf-8")
    registry = _v1_registry(tmp_path / "claims.json", status="SUPPORTED_AFTER_OPENING")
    assert module.validate_claims(
        root=tmp_path, registry_path=registry, require_complete=True
    ) == []


def test_v2_pre_phase_is_derived_and_require_complete_fails(tmp_path):
    module, registry_path, _, _ = _v2_fixture(tmp_path)
    registry = module._load_registry(registry_path)
    assert module.resolve_phase(root=tmp_path, registry=registry)["phase"] == module.PRE_PHASE
    assert module.validate_claims(root=tmp_path, registry_path=registry_path) == []
    assert module.validate_claims(
        root=tmp_path, registry_path=registry_path, require_complete=True
    ) == ["PHASE: --require-complete requires a verified completed receipt"]


def test_v2_pre_allows_exact_fixed_negative_disclosure_but_lints_overclaim(tmp_path):
    module, registry_path, registry, _ = _v2_fixture(tmp_path)
    template = registry["claim_templates"]["NEGATED_LIMITATION_NOT_UNGAUGED"]
    paper = tmp_path / "paper" / "main.md"
    paper.write_text(paper.read_text(encoding="utf-8") + "\n" + template + "\n", encoding="utf-8")
    assert module.validate_claims(root=tmp_path, registry_path=registry_path) == []
    paper.write_text(
        paper.read_text(encoding="utf-8")
        + "\nThe confirmatory evaluation establishes ungauged superiority.\n",
        encoding="utf-8",
    )
    violations = module.validate_claims(root=tmp_path, registry_path=registry_path)
    assert any("P01_NOT_UNGAUGED" in value for value in violations)
    assert any("LINT_UNSTRUCTURED_ROUTE_A_RESULT" in value for value in violations)


def test_v2_requires_every_canonical_document_and_permanent_disclosure(tmp_path):
    module, registry_path, registry, _ = _v2_fixture(tmp_path)
    extra = tmp_path / "paper" / "highlights.md"
    extra.write_text("Scoped limitations.\n", encoding="utf-8")
    registry["documents"].append("paper/highlights.md")
    registry["required_documents"].append("paper/highlights.md")
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    extra.unlink()
    with pytest.raises(module.ClaimRegistryError, match="canonical claim document closure"):
        module.validate_claims(root=tmp_path, registry_path=registry_path)

    # Restore the document, then remove one otherwise-valid structured limit.
    extra.write_text("Scoped limitations.\n", encoding="utf-8")
    paper = tmp_path / "paper" / "main.md"
    text = paper.read_text(encoding="utf-8")
    blocks, _, _ = module._parse_blocks(text)
    first = blocks[0]
    paper.write_text(
        text[: int(first["start"])] + text[int(first["stop"]):],
        encoding="utf-8",
    )
    violations = module.validate_claims(root=tmp_path, registry_path=registry_path)
    assert any("permanent limitation" in value and "appears 0 times" in value for value in violations)


def test_v2_protocol_sha_and_predicates_fail_closed(tmp_path):
    module, registry_path, registry, _ = _v2_fixture(tmp_path)
    protocol_path = tmp_path / registry["protocol_binding"]["path"]
    changed = json.loads(protocol_path.read_text(encoding="utf-8"))
    changed["external_new_gage_inference_contract"]["ungauged_claim_allowed"] = True
    protocol_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(module.ClaimRegistryError, match="protocol SHA-256 changed"):
        module.validate_claims(root=tmp_path, registry_path=registry_path)
    registry["protocol_binding"]["sha256"] = _sha256(protocol_path)
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(module.ClaimRegistryError, match="P01_NOT_UNGAUGED"):
        module.validate_claims(root=tmp_path, registry_path=registry_path)


def test_v2_orphan_namespace_and_partial_opening_are_indeterminate(tmp_path):
    module, registry_path, registry_document, _ = _v2_fixture(tmp_path)
    registry = module._load_registry(registry_path)
    orphan = tmp_path / "outputs" / "confirmatory" / f"route_a_{'b' * 24}" / "x.json"
    orphan.parent.mkdir(parents=True)
    orphan.write_text("{}\n", encoding="utf-8")
    with pytest.raises(module.ClaimRegistryError, match="without the canonical authorization"):
        module.resolve_phase(root=tmp_path, registry=registry)
    orphan.unlink()
    _, state = _write_authorization(tmp_path, registry_document)
    intent = tmp_path / state["intent"]
    intent.parent.mkdir(parents=True)
    intent.write_text("{}\n", encoding="utf-8")
    with pytest.raises(module.ClaimRegistryError, match="intent/receipt completion is partial"):
        module.resolve_phase(root=tmp_path, registry=registry)


def test_v2_invalid_completed_receipt_is_indeterminate(tmp_path, monkeypatch):
    module, registry_path, registry, _ = _v2_fixture(tmp_path)
    _, state = _write_authorization(tmp_path, registry)
    for key in ("intent", "receipt"):
        path = tmp_path / state[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    def fail(*args, **kwargs):
        raise module.ClaimRegistryError("canonical completed receipt did not verify")

    monkeypatch.setattr(module, "_validate_completed_receipt", fail)
    with pytest.raises(module.ClaimRegistryError, match="did not verify"):
        module.validate_claims(root=tmp_path, registry_path=registry_path)


def test_v2_post_renders_exact_five_blocks_and_validates_them(
    tmp_path, monkeypatch
):
    module, registry_path, registry, protocol = _v2_fixture(tmp_path)
    _post_fixture(module, tmp_path, registry, protocol, monkeypatch)
    rendered = module.render_result_claim_blocks(
        root=tmp_path, registry_path=registry_path
    )
    assert set(rendered) == {"paper/main.md"}
    assert rendered["paper/main.md"].count(b"<!-- ROUTE_A_CLAIM RESULT_") == 5
    assert rendered["paper/main.md"].count(
        b"DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED"
    ) == 15  # rendered verdict plus structured polarity/template_id per block
    assert b"SUPERIORITY_SUPPORTED for" not in rendered["paper/main.md"]
    assert b"NONINFERIORITY_SUPPORTED for" not in rendered["paper/main.md"]
    paper = tmp_path / "paper" / "main.md"
    paper.write_bytes(
        paper.read_bytes() + b"\n\n# Results\n\n" + rendered["paper/main.md"] + b"\n"
    )
    assert module.validate_claims(
        root=tmp_path,
        registry_path=registry_path,
        require_complete=True,
    ) == []


def test_v2_post_requires_every_test_exactly_once(tmp_path, monkeypatch):
    module, registry_path, registry, protocol = _v2_fixture(tmp_path)
    _post_fixture(module, tmp_path, registry, protocol, monkeypatch)
    payload = module.render_result_claim_blocks(
        root=tmp_path, registry_path=registry_path
    )["paper/main.md"]
    blocks = payload.split(b"\n\n")
    paper = tmp_path / "paper" / "main.md"
    permanent = paper.read_bytes()
    paper.write_bytes(permanent + b"\n\n" + b"\n\n".join(blocks[:-1]))
    violations = module.validate_claims(root=tmp_path, registry_path=registry_path)
    assert any("appears 0 times; expected exactly once" in value for value in violations)
    paper.write_bytes(permanent + b"\n\n" + payload + b"\n\n" + blocks[0])
    violations = module.validate_claims(root=tmp_path, registry_path=registry_path)
    assert any("appears 2 times" in value for value in violations)


def test_v2_self_consistent_manual_number_still_differs_from_renderer(
    tmp_path, monkeypatch
):
    module, registry_path, registry, protocol = _v2_fixture(tmp_path)
    _post_fixture(module, tmp_path, registry, protocol, monkeypatch)
    payload = module.render_result_claim_blocks(
        root=tmp_path, registry_path=registry_path
    )["paper/main.md"]
    text = payload.decode("utf-8")
    first_start = text.index("-->\n") + 4
    first_end = text.index("\n<!-- END ROUTE_A_CLAIM -->")
    body = text[first_start:first_end].replace("effect=-0.1", "effect=-9.9")
    digest = hashlib.sha256(body.encode()).hexdigest()
    header_end = text.index("-->\n")
    changed_header = text[:header_end].split("sha256=")[0] + f"sha256={digest} -->\n"
    tampered = changed_header + body + text[first_end:]
    paper = tmp_path / "paper" / "main.md"
    paper.write_bytes(paper.read_bytes() + b"\n\n" + tampered.encode("utf-8"))
    violations = module.validate_claims(root=tmp_path, registry_path=registry_path)
    assert any("differs from deterministic evidence rendering" in value for value in violations)
    assert not any("body hash is invalid" in value for value in violations)


def test_v2_statistics_must_match_receipt_binding_and_exact_family(
    tmp_path, monkeypatch
):
    module, registry_path, registry, protocol = _v2_fixture(tmp_path)
    statistics, state = _post_fixture(
        module, tmp_path, registry, protocol, monkeypatch
    )
    statistics["tests"][0]["test_id"] = "FORGED"
    path = tmp_path / state["statistics"]
    path.write_text(json.dumps(statistics), encoding="utf-8")
    # The receipt still binds the prior digest and formal-test bytes.
    with pytest.raises(module.ClaimRegistryError, match="SHA-256 changed"):
        module.validate_claims(root=tmp_path, registry_path=registry_path)


def test_v2_outcome_qc_gate_must_match_receipt_and_self_hash(
    tmp_path, monkeypatch,
) -> None:
    module, registry_path, registry, protocol = _v2_fixture(tmp_path)
    _, state = _post_fixture(
        module, tmp_path, registry, protocol, monkeypatch
    )
    gate_path = tmp_path / state["outcome_qc_gate"]
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["directional_claims_allowed_by_outcome_qc"] = False
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(module.ClaimRegistryError, match="SHA-256 changed"):
        module.validate_claims(root=tmp_path, registry_path=registry_path)


def test_v2_conflicting_p_and_ci_rules_render_not_supported(
    tmp_path, monkeypatch
):
    module, registry_path, registry, protocol = _v2_fixture(tmp_path)
    _post_fixture(
        module, tmp_path, registry, protocol, monkeypatch, conflict=True
    )
    rendered = module.render_result_claim_blocks(
        root=tmp_path, registry_path=registry_path
    )["paper/main.md"]
    assert b"DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED" in rendered
    assert b"EVIDENCE_CONFLICT_NOT_SUPPORTED" not in rendered
    assert b"SUPERIORITY_SUPPORTED for" not in rendered


def test_strong_p_values_and_favorable_intervals_cannot_override_failed_gate() -> None:
    module = _module()
    row = {
        "status": "ESTIMABLE",
        "reject_at_0_05": True,
        "confidence_bound_supports_margin": True,
        "margin_c": 0.0,
    }

    assert module._result_verdict(
        row,
        inference_claim_eligible=False,
        outcome_qc_claim_eligible=True,
    ) == "DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED"
    assert module._result_verdict(
        row,
        inference_claim_eligible=True,
        outcome_qc_claim_eligible=True,
    ) == "SUPERIORITY_SUPPORTED"
    assert module._result_verdict(
        row,
        inference_claim_eligible=True,
        outcome_qc_claim_eligible=False,
    ) == "DESCRIPTIVE_ONLY_OUTCOME_QC_GATE_FAILED"


def test_failed_outcome_qc_gate_withholds_directional_claims(
    tmp_path, monkeypatch,
) -> None:
    module, registry_path, registry, protocol = _v2_fixture(tmp_path)
    _post_fixture(
        module,
        tmp_path,
        registry,
        protocol,
        monkeypatch,
        inference_claim_eligible=True,
        outcome_qc_pass=False,
    )
    rendered = module.render_result_claim_blocks(
        root=tmp_path, registry_path=registry_path
    )["paper/main.md"]
    assert b"DESCRIPTIVE_ONLY_OUTCOME_QC_GATE_FAILED" in rendered
    assert b"SUPERIORITY_SUPPORTED for" not in rendered
    assert b"NONINFERIORITY_SUPPORTED for" not in rendered


def test_cli_has_no_phase_override(tmp_path):
    module, registry_path, _, _ = _v2_fixture(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--registry",
            str(registry_path),
            "--phase",
            "POST_CONFIRMATION_VERIFIED",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "unrecognized arguments: --phase" in result.stderr

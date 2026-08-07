"""Regression checks for the three ordinary legacy monitoring sites."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from _legacy_site_semantics import (  # noqa: E402
    EPISTEMIC_TOPOLOGY_SENTENCE,
    LEGACY_DATA_AUDIT_FILENAME,
    LEGACY_DATA_AUDIT_TOMBSTONE,
    LEGACY_FIGURE_BASENAMES,
    LEGACY_REPORT_FILENAME,
    LEGACY_REPORT_TOMBSTONE,
    ORDINARY_MONITORING_SENTENCE,
    REQUIRED_ALLOW_IDS,
    REQUIRED_LINT_IDS,
    find_legacy_semantic_violations,
    load_legacy_semantic_policy,
    retire_legacy_figure_outputs,
    retire_legacy_report_output,
)

sys.path.insert(0, str(ROOT / "src"))

from thermoroute import config as C  # noqa: E402


def test_legacy_monitoring_sites_do_not_expose_a_network_api() -> None:
    assert C.LEGACY_NETWORK_METADATA_VERIFIED is False
    for legacy_network_name in (
        "UPSTREAM",
        "FLOW_TRAVEL_DAYS",
        "THERMAL_TRAVEL_DAYS",
    ):
        assert not hasattr(C, legacy_network_name)


def test_legacy_narratives_do_not_reintroduce_cascade_claims() -> None:
    paths = {ROOT / "README.md", ROOT / "outputs/README.md"}
    paths.update(
        path
        for root_name in ("paper", "protocols")
        for path in (ROOT / root_name).rglob("*")
        if path.is_file() and path.suffix in {".md", ".tex"}
    )
    paths = sorted(paths)
    policy = load_legacy_semantic_policy(ROOT)
    for path in paths:
        assert not find_legacy_semantic_violations(
            path.read_text(encoding="utf-8"), policy
        ), path
    producer_roots = (
        ROOT / "src",
        ROOT / "scripts",
        ROOT / ".github" / "workflows",
        ROOT / "paper" / "agu_submission",
    )
    legacy_text_producers = tuple(sorted({
        path
        for producer_root in producer_roots
        for path in producer_root.rglob("*")
        if path.is_file() and path.suffix in {".py", ".sh", ".yml", ".yaml"}
    }))
    assert legacy_text_producers
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (*paths, *legacy_text_producers)
    )
    forbidden = (
        r"b1\s*(?:→|->)\s*s2",
        r"s2\s*(?:→|->)\s*p3",
        r"confirms?\s+b1",
        r"station topology\s*\(directed cascade",
        r"near[- ]deterministic\s+cascade",
        r'"cascade_(?:predictions|scores)"',
        r"三站级联",
        r"级联(?:实验|主表|\s*Track A|的\s*PICP|决策价值)",
        r"b1\s+is\s+more\s+regulated",
        r"(?:the\s+)?three\s+(?:sites|stations).{0,30}"
        r"(?:form|constitute|comprise).{0,30}(?:cascade|reservoir)",
        r"(?:b1|s2|p3)\s+(?:is|are|was|were)\s+(?:an?\s+)?"
        r"(?:reservoir|regulated|upstream|downstream)",
        r"(?:b1|s2|p3).{0,30}(?:feeds?|flows?\s+into|discharges?\s+to)"
        r".{0,30}(?:b1|s2|p3)",
        r"(?:b1|s2|p3)\s+(?:is\s+)?(?:upstream|downstream)\s+of\s+"
        r"(?:b1|s2|p3)",
    )
    for pattern in forbidden:
        assert re.search(pattern, text, flags=re.IGNORECASE) is None, pattern


def test_legacy_generated_diagnostics_do_not_overclaim_mechanism_or_transfer() -> None:
    explain = (ROOT / "scripts/05_explain.py").read_text(encoding="utf-8")
    figures = (ROOT / "scripts/06_make_figures.py").read_text(encoding="utf-8")
    runner = (ROOT / "scripts/run_all.sh").read_text(encoding="utf-8")
    generated_sources = "\n".join((explain, figures, runner))
    for phrase in (
        "mechanism analysis",
        "Mechanism summary",
        "per-day memory",
        "implied memory 1/κ",
        "Top variable×lag drivers",
        "Leave-one-station-out spatial transfer",
        "Router variable×lag importance",
        "Router lag importance",
        "blindtest",
    ):
        assert phrase.lower() not in generated_sources.lower()
    for required in (
        "ordinary monitoring stations",
        "encodes no river network",
        "not physical parameters",
        "not zero-shot transfer",
    ):
        assert required.lower() in generated_sources.lower()
    explain_retirement = explain.index("legacy_output = retire_legacy_report_output")
    first_data_operation = explain.index("bundle = D.prepare_dataset()")
    assert explain_retirement < first_data_operation
    assert "retire_legacy_figure_outputs(FIG)" in figures


def test_user_facing_model_terms_do_not_upgrade_statistical_components() -> None:
    relatives = (
        "src/thermoroute/air2stream.py",
        "src/thermoroute/baselines.py",
        "src/thermoroute/datasets.py",
        "src/thermoroute/neural_baselines.py",
        "src/thermoroute/thermoroute.py",
        "src/thermoroute/robustness.py",
        "src/thermoroute/development_controls.py",
        "scripts/09_usgs_experiment.py",
        "scripts/13c_region_transfer.py",
        "scripts/23_robustness.py",
    )
    text = "\n".join(
        (ROOT / relative).read_text(encoding="utf-8") for relative in relatives
    )
    normalized = " ".join(text.lower().split())
    for forbidden in (
        "half-physical",
        "physical reference",
        "physical-reference",
        "physical baseline",
        "safety anchor",
        "certified band",
        "physical-anchor",
        "no gage on a held-out river",
    ):
        assert forbidden not in normalized, forbidden
    assert "unofficial empirical comparator" in normalized
    assert "not as an identified physical model" in normalized
    assert "not a bound on truth error" in normalized
    assert "synthetic data-corruption" in normalized


def test_shared_legacy_lints_catch_adversarial_paraphrases() -> None:
    policy = load_legacy_semantic_policy(ROOT)
    assert tuple(policy.forbidden) == REQUIRED_LINT_IDS
    assert tuple(policy.allowed) == REQUIRED_ALLOW_IDS
    assert all(pattern.flags & re.DOTALL for pattern in policy.forbidden.values())
    for positive_claim in (
        "A three-station regulated reservoir cascade was studied.",
        "A three-site regulated reservoir cascade was studied.",
        "A regulated reservoir cascade of b1, s2, and p3 was studied.",
        "The three stations are a reservoir cascade.",
        "The three monitoring stations form a reservoir cascade.",
        "The three gauges comprise a reservoir cascade.",
        "The three sites form a hydraulic cascade.",
        "The three sites form a reservoir cascade.",
        "b1, s2, and p3 form a reservoir cascade.",
        "p3, b1, and s2 form a reservoir cascade.",
        "b1 -> s2 -> p3",
        "b1 => s2 => p3",
        "b1 ⟶ s2 ⟶ p3",
        "b1 is upstream of s2.",
        "b1 lies upstream of s2.",
        "b1 hydraulically precedes s2.",
        "p3 receives flow from s2.",
        "b1 drains to s2.",
        "b1 drains toward s2.",
        "b1 empties into s2.",
        "b1 precedes s2 along the river.",
        "b1 precedes s2 along the channel.",
        "b1 receives dam releases from p3.",
        "b1 is hydrologically above s2.",
        "Flow propagates from b1 to s2.",
        "Water travels from b1 toward s2.",
        "b1 routes water toward s2.",
        "b1 and s2 are hydraulically connected.",
        "b1 and s2 are connected.",
        "The b1 and s2 records encode downstream propagation.",
        "b1 feeds s2.",
        "b1 serves as a reservoir.",
        "b1 is part of a reservoir.",
        "p3 acts as a reservoir.",
        "p3 functions as an impoundment.",
        "p3 is an impoundment.",
        "The reservoir at p3 regulates flow.",
        "p3 is the downstream outlet.",
        "p3 is regulated.",
        "b1, s2, and p3 are reservoirs.",
        "p3, b1, and s2 are reservoirs.",
        "The sites were ordered upstream to downstream as b1, s2, p3.",
        "The three stations form a hydraulic chain.",
        "The gauges follow the sequence b1, s2, p3 along the river.",
        "κ identifies residence time.",
        "kappa estimates travel time.",
        "κ estimates travel-time.",
        "κ estimates turnover time.",
        "The retention time is encoded by kappa.",
        "1/kappa has units of days.",
        "κ corresponds to residence time.",
        "κ is a proxy for travel time.",
        "Kappa captures physical memory.",
        "Kappa represents physical memory.",
        "1/κ is the residence time.",
        "The router recovers travel time.",
        "The router reveals physical drivers.",
        "The router reveals flow pathways.",
        "Router weights are mechanistically interpretable.",
        "Physical drivers are revealed by the router.",
        "Allocation weights quantify causal influence.",
        "Router weights are feature importance.",
        "`b1` &rarr; `s2`.",
        r"$b1 \rightarrow s2$.",
        "b1 --> s2.",
        "b1 supplies water to s2.",
        "b1 is hydrologically linked to s2.",
        "b1 is hydraulically-connected to s2.",
        "From b1 via s2, water reaches p3.",
        "Thermal anomalies travel from b1 to s2.",
        "p3 represents a reservoir.",
        "p3 constitutes a regulated reach.",
        "p3 reservoir.",
        "p3 tailwater station.",
        "Regulated station p3.",
        "b1, s2, and p3 form a hydrologic network.",
        "b1 is above s2 along the river.",
        "s2 is below b1 in the reach.",
        "The flow path runs b1 then s2 then p3.",
        "Residence duration is represented by kappa.",
        "Dominant lag estimates propagation delay.",
        "The learned proposal represents heat transfer.",
        "The learned proposal recovers causal physics.",
        "b1 represents the upper reservoir.",
        "s2 receives b1 discharge.",
        "s2 gets water from b1.",
        "Water from b1 reaches s2.",
        "b1 comes before s2 hydrologically.",
        "The river connects b1, s2, and p3 in that order.",
        "b1, s2, then p3 follow the flow direction.",
        "Router weights expose physical pathways.",
        "Router weights map the river network.",
        "The router traces connectivity.",
        "The learned proposal is a heat-transfer model.",
        "p3 is a tailwater.",
        "b1, e.g. s2 and p3, form a reservoir cascade.",
        "b1, Fig. 1, s2, and p3 are reservoirs.",
        "b1, i.e. the first, s2, and p3 were ordered upstream to downstream.",
        "b1 (U.S. site), s2, and p3 constitute a hydraulic chain.",
        "b1、s2和p3构成水库级联。",
        "p3是一座水库。",
        "b1流向s2。",
        "p3 was a reservoir.",
        "s2 is fed by b1.",
        "b1 conveys water to s2.",
        "p3 denotes a reservoir.",
        "The trio b1, s2, and p3 is a cascade.",
        "1/κ = 4 days.",
        "κ has dimensions day^-1.",
        "b1 is up<strong>stream</strong> of s2.",
        r"b1 is \textbf{up}stream of s2.",
        r"\textbf{\emph{b1}} is upstream of s2.",
        r"b1 \longrightarrow s2.",
        r"b1 \xrightarrow{flow} s2.",
        '<img alt="b1 is upstream of s2">',
        "<span title='p3 is a reservoir'>neutral</span>",
        '<div aria-label="b1 feeds s2">neutral</div>',
        r"\href{https://example.invalid}{b1 is upstream of s2}.",
        r"\hyperref[fig:sites]{p3 is a reservoir}.",
        "[b1 is upstream of s2](https://example.invalid).",
        "![p3 is a reservoir](figure.png)",
        "b1: upstream; s2: middle; p3: downstream.",
        "b1、s2、p3依次位于河流的上、中、下游。",
        "水从b1经过s2最终到达p3。",
        "Higher FLOW causes a larger κ.",
        "The learned relaxation is a semi-physical heat-transfer model.",
    ):
        violations = find_legacy_semantic_violations(positive_claim, policy)
        assert violations, positive_claim
    for valid_limitation in (
        "b1, s2, and p3 are ordinary monitoring stations, not reservoirs.",
        "Their display order does not establish a cascade or travel time.",
        "Twenty-three stations form a reservoir cascade in an unrelated quotation.",
        "ab1 -> s2x is an arbitrary token sequence.",
        "xp3 was a reservoir code.",
        "κ does not identify residence time.",
        "κ does not directly identify residence time.",
        "It is false that κ identifies residence time.",
        "We reject the claim that κ identifies residence time.",
        "κ cannot reliably estimate travel time.",
        "We reject the claim that p3 was a reservoir.",
        "No verified metadata establish any upstream/downstream ordering, "
        "hydraulic connectivity, regulation status, or travel time among b1, "
        "s2, and p3.",
        "In an unrelated benchmark, stations are ordered from upstream to downstream.",
        "The router does not reveal physical drivers.",
        "The router does not empirically reveal physical drivers.",
        "The router cannot reliably reveal causal mechanisms.",
        "There is no evidence that the router reveals physical drivers.",
    ):
        assert not find_legacy_semantic_violations(valid_limitation, policy), (
            valid_limitation
        )
    for laundering_attack in (
        "b1 is upstream of s2, not downstream.",
        "The router reveals physical drivers, not merely correlations.",
        "κ represents residence time, not travel time.",
        "It is not only plausible: b1 feeds s2.",
        "No evidence was available before; b1 is upstream of s2.",
        "The prior claim was unsupported, but b1 is a reservoir.",
        "The router reveals physical drivers but does not report confidence "
        "intervals.",
        "κ represents residence time but does not estimate travel time.",
        "The learned proposal identifies physical mechanisms and does not use a "
        "river graph.",
        "It is false that critics were right, and b1 is upstream of s2.",
        "There is no evidence that the sample is biased, and b1 feeds s2.",
        "We reject the claim that the data are sparse; p3 is a reservoir.",
        "Those claims were unsupported before, but b1 is a reservoir.",
        "Historical outputs were withdrawn, but b1 feeds s2.",
        "b1, s2, and p3 are ordinary monitoring stations, not reservoirs; b1 is "
        "upstream but does not feed s2.",
        "b1, s2, and p3 are ordinary monitoring stations, not reservoirs, but "
        "they form a hydraulic cascade.",
        "b1, s2, and p3 are ordinary monitoring stations, not reservoirs and are "
        "arranged upstream to downstream.",
        "No verified metadata establish any upstream/downstream ordering, "
        "hydraulic connectivity, regulation status, or travel time among b1, "
        "s2, and p3, yet they are hydraulically linked in series.",
        "κ does not identify residence time but it estimates travel time.",
        "The router does not reveal physical drivers but it recovers travel time.",
        "There is no evidence that b1 is upstream of s2.",
        "It is false that b1 is upstream of s2.",
        "Neither b1 nor s2 is upstream of p3.",
        "b1 and s2 are not hydraulically connected.",
    ):
        assert find_legacy_semantic_violations(laundering_attack, policy), (
            laundering_attack
        )
    for unrelated_valid_statement in (
        "b1, s2, and p3 form the legacy monitoring cohort.",
        "Data from b1, s2, and p3 comprise 15 years of observations.",
        "The b1, s2, and p3 records constitute the optional case study.",
        "b1、s2和p3是普通监测站，不是水库。",
    ):
        assert not find_legacy_semantic_violations(
            unrelated_valid_statement, policy
        ), unrelated_valid_statement


def test_manuscript_generators_load_the_legacy_semantic_lints() -> None:
    renderer = (ROOT / "scripts/29_render_preopen_manuscripts.py").read_text(
        encoding="utf-8"
    )
    assert "find_legacy_semantic_violations" in renderer
    main_body = renderer.split("def main(", 1)[1]
    assert main_body.index("_audit_markdown_semantics_before_render") < main_body.index(
        "_build_main"
    )


def test_legacy_semantic_policy_rejects_epistemic_overclaim() -> None:
    policy = load_legacy_semantic_policy(ROOT)
    violations = find_legacy_semantic_violations(
        "b1 drains toward s2.", policy
    )
    assert violations


def test_legacy_report_tombstone_is_atomic_and_precedes_training(tmp_path) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    legacy = report_dir / LEGACY_REPORT_FILENAME
    legacy.write_text("STALE-LEGACY-CLAIM\n", encoding="utf-8")
    unrelated = report_dir / "keep.md"
    unrelated.write_text("keep\n", encoding="utf-8")
    written = retire_legacy_report_output(report_dir)
    assert written == legacy
    assert legacy.read_text(encoding="utf-8") == LEGACY_REPORT_TOMBSTONE
    assert ORDINARY_MONITORING_SENTENCE in LEGACY_REPORT_TOMBSTONE
    assert EPISTEMIC_TOPOLOGY_SENTENCE in LEGACY_REPORT_TOMBSTONE
    assert unrelated.read_text(encoding="utf-8") == "keep\n"
    assert not list(report_dir.glob(f".{LEGACY_REPORT_FILENAME}.*.tmp"))


def test_stage01_retires_stale_topology_report_before_data_failure(
    tmp_path, monkeypatch,
) -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/01_prepare_data.py"))
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    stale = report_dir / LEGACY_DATA_AUDIT_FILENAME
    stale.write_text(
        "Station topology: b1 -> s2 -> p3; flow travels one day per hop.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(namespace["C"], "REPORTS", report_dir)

    def fail_after_retirement():
        raise RuntimeError("synthetic data failure")

    monkeypatch.setattr(namespace["D"], "prepare_dataset", fail_after_retirement)
    with pytest.raises(RuntimeError, match="synthetic data failure"):
        namespace["main"]()
    assert stale.read_text(encoding="utf-8") == LEGACY_DATA_AUDIT_TOMBSTONE
    assert ORDINARY_MONITORING_SENTENCE in LEGACY_DATA_AUDIT_TOMBSTONE
    assert EPISTEMIC_TOPOLOGY_SENTENCE in LEGACY_DATA_AUDIT_TOMBSTONE


def test_figure_entrypoint_retires_exact_stale_names_before_fig1(tmp_path) -> None:
    for basename in LEGACY_FIGURE_BASENAMES:
        for suffix in (".png", ".pdf"):
            (tmp_path / f"{basename}{suffix}").write_bytes(b"STALE-LEGACY-CLAIM")
    unrelated = tmp_path / "keep.png"
    unrelated.write_bytes(b"keep")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/06_make_figures.py"),
            "--fig1-only",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "retired 12 withdrawn legacy figure files" in completed.stdout
    for basename in LEGACY_FIGURE_BASENAMES:
        for suffix in (".png", ".pdf"):
            assert not (tmp_path / f"{basename}{suffix}").exists()
    assert unrelated.read_bytes() == b"keep"
    assert (tmp_path / "fig1_monitoring_site_identifiers.png").is_file()


def test_figure_retirement_helper_is_exact_when_nothing_exists(tmp_path) -> None:
    unrelated_paths = (
        tmp_path / "fig7_router_allocation.png",
        tmp_path / "fig8_latent_decay_coefficient.png",
    )
    for unrelated in unrelated_paths:
        unrelated.write_bytes(b"keep")
    assert retire_legacy_figure_outputs(tmp_path) == ()
    assert all(path.read_bytes() == b"keep" for path in unrelated_paths)


def test_explain_npz_contract_and_figure_text_guard_fail_closed(tmp_path) -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/06_make_figures.py"))
    contract = {
        "semantic_contract_version": np.array(
            "thermoroute.legacy-monitoring-latent-diagnostics.v1"
        ),
        "site_classification": np.array(
            "ORDINARY_MONITORING_STATIONS_NOT_RESERVOIRS"
        ),
        "verified_network_metadata": np.array(False),
        "topology_inference_allowed": np.array(False),
        "physical_interpretation_allowed": np.array(False),
        "causal_interpretation_allowed": np.array(False),
        "analysis_role": np.array("SINGLE_SEED_DESCRIPTIVE_DIAGNOSTIC"),
        "diagnostic_seed": np.array(0),
        "site_ids": np.array(C.STATIONS),
    }
    namespace["_validate_explain_contract"](contract)
    bad_contract = dict(contract)
    bad_contract["topology_inference_allowed"] = np.array(True)
    with pytest.raises(RuntimeError, match="topology_inference_allowed"):
        namespace["_validate_explain_contract"](bad_contract)

    save_figure = namespace["_save"]
    save_figure.__globals__["_SEMANTIC_POLICY"] = load_legacy_semantic_policy(ROOT)
    save_figure.__globals__["FIG"] = tmp_path
    figure, axis = namespace["plt"].subplots()
    axis.set_title("b1 drains toward s2")
    with pytest.raises(RuntimeError, match="legacy semantic guard refused figure"):
        save_figure(figure, "bad_semantic_figure")
    namespace["plt"].close(figure)
    assert not (tmp_path / "bad_semantic_figure.png").exists()

    split_figure, split_axis = namespace["plt"].subplots()
    split_axis.set_title("Reservoir cascade and travel times")
    for index, site_id in enumerate(("b1", "s2", "p3")):
        split_axis.text(index, index, site_id)
    with pytest.raises(RuntimeError, match="legacy semantic guard refused figure"):
        save_figure(split_figure, "split_context_attack")
    namespace["plt"].close(split_figure)
    assert not (tmp_path / "split_context_attack.png").exists()

    arrow_figure, arrow_axis = namespace["plt"].subplots()
    arrow_axis.add_patch(
        namespace["matplotlib"].patches.FancyArrowPatch((0.2, 0.5), (0.8, 0.5))
    )
    with pytest.raises(RuntimeError, match="topology geometry"):
        save_figure(arrow_figure, "fig1_monitoring_site_identifiers")
    namespace["plt"].close(arrow_figure)
    assert not (tmp_path / "fig1_monitoring_site_identifiers.png").exists()


def test_stale_adversarial_reports_are_not_tracked_as_current_evidence() -> None:
    assert not (ROOT / "outputs/reports/adversarial_review_tri_persona.md").exists()
    assert not (ROOT / "outputs/reports/adversarial_review_strong_accept.md").exists()
    note = (ROOT / "outputs/README.md").read_text(encoding="utf-8")
    assert "Historical files formerly stored" in note
    assert "ordinary monitoring sites" in note
    assert "not an authority document" in note


def test_historical_three_site_notice_is_exactly_registered() -> None:
    relative = "protocols/legacy_three_site_semantics_notice_v1.md"
    notice_path = ROOT / relative
    notice = notice_path.read_text(encoding="utf-8")
    assert ORDINARY_MONITORING_SENTENCE in " ".join(notice.replace("`", "").split())
    assert EPISTEMIC_TOPOLOGY_SENTENCE in " ".join(notice.split())
    assert "f0a09ba44296a896205e219984b95e8bfe728219" in notice
    assert "7c034e8976042899cbe2cc2a2f04322488010d84" in notice
    assert "08536ce771529b189c4a63a069168b1d0a78c490" in notice
    assert "withdrawn historical provenance, not results" in notice

    registry = json.loads(
        (ROOT / "protocols/route_a_claim_registry_v1.json").read_text(
            encoding="utf-8"
        )
    )
    digest = hashlib.sha256(notice_path.read_bytes()).hexdigest()
    assert relative in registry["documents"]
    assert relative in registry["required_documents"]
    assert registry["preopen_document_sha256"][relative] == digest
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "[legacy three-site semantics notice]" in root_readme
    assert f"({relative})" in root_readme


def test_legacy_wlevel_report_has_only_raw_unknown_unit_qc_semantics() -> None:
    source = (ROOT / "scripts/01_prepare_data.py").read_text(encoding="utf-8")
    assert "Raw FLOW–WLEVEL association (QC only)" in source
    assert "raw legacy units" in source
    assert "units, and vertical datum are unverified" in source
    assert "rating curve" not in source.lower()
    assert "stage–discharge" not in source.lower()
    assert "WLEVEL drift 2006→2020 (m)" not in source


def test_canonical_run_does_not_silently_execute_the_legacy_case() -> None:
    script = (ROOT / "scripts/run_all.sh").read_text(encoding="utf-8")
    assert "if (( $# != 0 )); then" in script
    assert "--include-legacy-monitoring-case" not in script
    assert "INCLUDE_LEGACY_MONITORING_CASE" not in script
    assert "data/processed" not in script
    assert "scripts/14_manifest.py --check-route-a-boundary" in script
    assert "scripts/14_manifest.py --development-prelabel" in script
    assert "scripts/21_ecological_thresholds.py" not in script
    assert (
        'CANONICAL_USGS_STATION_REGISTRY="data_usgs/station_registry_v1.csv"'
        in script
    )
    assert 'export USGS_STATION_REGISTRY="$CANONICAL_USGS_STATION_REGISTRY"' in script
    assert "custom USGS_STATION_REGISTRY is unsupported" in script
    for legacy_entrypoint in (
        "scripts/01_prepare_data.py",
        "scripts/04_run_experiments.py",
        "scripts/05_explain.py",
        "scripts/06_make_figures.py",
        "scripts/07_make_tables.py",
        "scripts/08_decision_value.py",
    ):
        assert legacy_entrypoint not in script
    assert "b1/s2/p3 ordinary monitoring stations" in script
    assert "ROUTE A: USGS DEVELOPMENT ANALYSIS" in script
    assert "TRACK B" not in script

    for argument in ("--include-legacy-monitoring-case", "--help", "unexpected"):
        completed = subprocess.run(
            ["bash", str(ROOT / "scripts/run_all.sh"), argument],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 2, argument
        assert completed.stdout == ""
        assert completed.stderr == "usage: bash scripts/run_all.sh\n"


def test_release_archive_excludes_legacy_site_data_but_keeps_correction() -> None:
    script = (ROOT / "scripts/make_release_archive.sh").read_text(
        encoding="utf-8"
    )
    for legacy_data_path in ("data/b1.csv", "data/s2.csv", "data/p3.csv"):
        assert legacy_data_path not in script
    assert '$STAGE/data' not in script
    assert "protocols/legacy_three_site_semantics_notice_v1.md" in script


def test_legacy_site_figure_has_no_network_or_travel_time_semantics() -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/06_make_figures.py"))
    figure_function = namespace["fig_study_area"]
    captured: dict[str, object] = {}

    def capture(figure: object, name: str) -> None:
        captured["figure"] = figure
        captured["name"] = name

    figure_function.__globals__["_save"] = capture
    panel = pd.DataFrame(
        {
            "site_id": ["b1", "b1", "s2", "s2", "p3", "p3"],
            "WTEMP": [4.0, 12.0, 6.0, 14.0, 8.0, 16.0],
        }
    )
    figure_function(panel)

    assert captured["name"] == "fig1_monitoring_site_identifiers"
    figure = captured["figure"]
    axes = figure.axes  # type: ignore[attr-defined]
    rendered_text = "\n".join(
        text.get_text() for axis in axes for text in axis.texts
    )
    assert rendered_text.count("monitoring\nsite") == 3
    assert "Unordered identifiers" in rendered_text
    assert "positions have no geographic or hydrologic meaning" in rendered_text
    assert "No reservoir, connectivity, regulation, or travel-time claim" in rendered_text
    for forbidden in (
        "Reservoir cascade",
        "flow ~",
        "thermal ~",
        "upstream",
        "downstream",
        "2480 m",
        "1819 m",
        "989 m",
    ):
        assert forbidden not in rendered_text

    left_axis = axes[0]
    assert not left_axis.lines
    assert all(
        patch.__class__.__name__ == "FancyBboxPatch"
        for patch in left_axis.patches
    )
    assert not {
        "FancyArrow",
        "FancyArrowPatch",
        "ConnectionPatch",
    } & {artist.__class__.__name__ for artist in left_axis.get_children()}
    site_positions = {
        text.get_text(): text.get_position()
        for text in left_axis.texts
        if text.get_text() in {"b1", "s2", "p3"}
    }
    assert set(site_positions) == {"b1", "s2", "p3"}
    assert [
        site for site, _ in sorted(site_positions.items(), key=lambda item: item[1][0])
    ] == sorted(site_positions)
    (x1, y1), (x2, y2), (x3, y3) = [
        site_positions[site] for site in sorted(site_positions)
    ]
    twice_triangle_area = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    assert abs(twice_triangle_area) > 1e-9
    namespace["plt"].close(figure)

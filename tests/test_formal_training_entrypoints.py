from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
FORMAL_ENTRYPOINTS = (
    ("scripts/09_usgs_experiment.py", "--_thermoroute-stage09-worker"),
    ("scripts/09b_development_controls.py", "--_thermoroute-stage09b-worker"),
    ("scripts/16_lstm_baseline.py", "--_thermoroute-stage16-worker"),
    ("scripts/25_train_external_pooled_suite.py", "--_thermoroute-stage25-worker"),
)

FORMAL_WORKER_ENVIRONMENT_KEYS = {
    "scripts/09_usgs_experiment.py": (
        "THERMOROUTE_STAGE09_PYCACHE",
        "THERMOROUTE_STAGE09_NONCE",
    ),
    "scripts/09b_development_controls.py": (
        "THERMOROUTE_STAGE09B_PYCACHE",
        "THERMOROUTE_STAGE09B_NONCE",
    ),
    "scripts/16_lstm_baseline.py": (
        "THERMOROUTE_STAGE16_PYCACHE",
        "THERMOROUTE_STAGE16_NONCE",
    ),
    "scripts/25_train_external_pooled_suite.py": (
        "THERMOROUTE_STAGE25_PYCACHE",
        "THERMOROUTE_STAGE25_NONCE",
    ),
}

RUNTIME_ENFORCED_ENTRYPOINTS = {
    "scripts/09_usgs_experiment.py": 4,
    "scripts/09b_development_controls.py": 3,
    "scripts/16_lstm_baseline.py": 3,
    "scripts/25_train_external_pooled_suite.py": 4,
}

FINAL_PUBLICATION_GUARDS = {
    "scripts/09_usgs_experiment.py": {"publish_stage09_completion_receipt"},
    "scripts/09b_development_controls.py": {"publish_stage09b_completion_receipt"},
    "scripts/16_lstm_baseline.py": {"publish_stage16_completion_receipt"},
    "scripts/25_train_external_pooled_suite.py": {"publish_stage25_completion_receipt"},
}


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _is_native_policy_assertion(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and _call_name(statement.value) == "assert_formal_numerical_policy"
    )


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    matches = [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, f"expected exactly one function named {name!r}"
    return matches[0]


def _keyword_is_name(call: ast.Call, keyword: str, name: str) -> bool:
    return any(
        item.arg == keyword and isinstance(item.value, ast.Name) and item.value.id == name
        for item in call.keywords
    )


def _keyword_is_true(call: ast.Call, keyword: str) -> bool:
    return any(
        item.arg == keyword and isinstance(item.value, ast.Constant) and item.value.value is True
        for item in call.keywords
    )


def _statement_context(
    tree: ast.AST,
) -> tuple[dict[ast.AST, ast.AST], dict[ast.stmt, list[ast.stmt]]]:
    parent: dict[ast.AST, ast.AST] = {}
    statement_block: dict[ast.stmt, list[ast.stmt]] = {}
    for node in ast.walk(tree):
        for _field, value in ast.iter_fields(node):
            children = value if isinstance(value, list) else [value]
            for child in children:
                if isinstance(child, ast.AST):
                    parent[child] = node
            if isinstance(value, list) and all(isinstance(child, ast.stmt) for child in value):
                for child in value:
                    statement_block[child] = value
    return parent, statement_block


def _containing_statement(node: ast.AST, parent: dict[ast.AST, ast.AST]) -> ast.stmt:
    while not isinstance(node, ast.stmt):
        node = parent[node]
    return node


def _preceding_statement_on_execution_path(
    statement: ast.stmt,
    parent: dict[ast.AST, ast.AST],
    statement_block: dict[ast.stmt, list[ast.stmt]],
) -> ast.stmt | None:
    current: ast.AST = statement
    while True:
        if isinstance(current, ast.stmt) and current in statement_block:
            block = statement_block[current]
            position = block.index(current)
            if position:
                return block[position - 1]
        if current not in parent:
            return None
        current = parent[current]


@pytest.mark.parametrize(
    "relative",
    (
        "scripts/09_usgs_experiment.py",
        "scripts/16_lstm_baseline.py",
        "scripts/25_train_external_pooled_suite.py",
    ),
)
def test_every_formal_fit_propagates_the_live_publication_guard(relative):
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) == "fit_model"
    ]
    assert calls, f"{relative} no longer contains a formal fit_model call"
    missing = [
        call.lineno
        for call in calls
        if not _keyword_is_name(
            call,
            "artifact_publication_guard",
            "assert_formal_numerical_policy",
        )
    ]
    assert not missing, f"{relative} fit_model calls without the live publication guard: {missing}"


def test_stage16_retains_every_authority_split_before_bundle_parity():
    source = (ROOT / "scripts" / "16_lstm_baseline.py").read_text(
        encoding="utf-8"
    )
    assert "usgs_predictions_with_perstation_v2.parquet" not in source
    tree = ast.parse(source)
    main = _function(tree, "insample")

    parent_verifier = _function(tree, "_verify_parent")
    receipt_validations = [
        node
        for node in ast.walk(parent_verifier)
        if isinstance(node, ast.Call)
        and _call_name(node) == "validate_stage09_completion_receipt"
    ]
    assert len(receipt_validations) == 1
    assert _keyword_is_name(
        receipt_validations[0],
        "publication_guard",
        "assert_formal_numerical_policy",
    )

    parent_reads = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and _call_name(node) == "read_parquet"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "PARENT"
    ]
    assert len(parent_reads) == 1

    split_assignments: dict[str, str] = {}
    for node in ast.walk(main):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"lv", "lc", "lt"}
        ):
            constants = {
                child.value
                for child in ast.walk(node.value)
                if isinstance(child, ast.Constant)
                and isinstance(child.value, str)
            }
            expected = {"lv": "val", "lc": "calib", "lt": "test"}[
                node.targets[0].id
            ]
            assert expected in constants
            split_assignments[node.targets[0].id] = expected
    assert split_assignments == {"lv": "val", "lc": "calib", "lt": "test"}

    concatenations = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and _call_name(node) == "concat"
        and node.args
        and isinstance(node.args[0], (ast.List, ast.Tuple))
    ]
    assert any(
        {item.id for item in call.args[0].elts if isinstance(item, ast.Name)}
        >= {"parent_non_lstm", "lv", "lc", "lt"}
        for call in concatenations
    ), "Stage 16 no longer retains val/calib/test LSTM rows in its authority frame"

    common_key_calls = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and _call_name(node) == "enforce_common_forecast_keys"
    ]
    v2_writes = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and _call_name(node) == "write_predictions"
        and any(isinstance(arg, ast.Name) and arg.id == "V2" for arg in node.args)
    ]
    fail_closed_guards = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Attribute)
            and child.attr == "dropped_rows"
            for child in ast.walk(node.test)
        )
        and sum(
            isinstance(child, ast.Call)
            and _call_name(child) == "canonical_frame_digest"
            for child in ast.walk(node.test)
        ) >= 2
        and any(isinstance(child, ast.Raise) for child in ast.walk(node))
    ]
    assert len(common_key_calls) >= 1
    assert len(v2_writes) == 1
    assert len(fail_closed_guards) == 1
    assert (
        common_key_calls[-1].lineno
        < fail_closed_guards[0].lineno
        < v2_writes[0].lineno
    ), "Stage 16 must reject key loss before publishing canonical V2"

    parity_calls = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and _call_name(node) == "verify_sequence_prediction_parity"
    ]
    assert len(parity_calls) == 1
    splits = next(
        keyword.value
        for keyword in parity_calls[0].keywords
        if keyword.arg == "splits"
    )
    assert isinstance(splits, ast.Tuple)
    assert tuple(
        item.value for item in splits.elts if isinstance(item, ast.Constant)
    ) == ("val", "calib", "test")


def test_stage09_lightgbm_shard_acquisition_cannot_publish_or_return_unguarded():
    tree = ast.parse((ROOT / "scripts" / "09_usgs_experiment.py").read_text(encoding="utf-8"))
    acquire = _function(tree, "acquire_shard")
    _parent, statement_block = _statement_context(acquire)

    shard_saves = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) == "save_lightgbm_shard"
    ]
    assert shard_saves
    assert all(
        _keyword_is_name(call, "publication_guard", "assert_formal_numerical_policy")
        for call in shard_saves
    )

    returns = [node for node in ast.walk(acquire) if isinstance(node, ast.Return)]
    assert returns
    for statement in returns:
        value = statement.value
        if isinstance(value, ast.Name) and value.id in {"cached", "fitted"}:
            block = statement_block[statement]
            position = block.index(statement)
            assert position > 0 and _is_native_policy_assertion(block[position - 1]), (
                f"unguarded {value.id} return at line {statement.lineno}"
            )
            continue
        assert isinstance(value, ast.Call) and _call_name(value) == "save_lightgbm_shard", (
            f"unexpected shard acquisition return at line {statement.lineno}"
        )
        assert _keyword_is_name(value, "publication_guard", "assert_formal_numerical_policy")

    finalizers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) == "finalize_shard_set"
    ]
    assert finalizers
    assert all(
        _keyword_is_name(call, "publication_guard", "assert_formal_numerical_policy")
        for call in finalizers
    )


def test_stage09b_training_and_final_publication_propagate_the_live_guard():
    tree = ast.parse((ROOT / "scripts" / "09b_development_controls.py").read_text(encoding="utf-8"))
    guarded_calls = {
        "train_arm_group",
        "publish_final_artifacts",
        "publish_stage09b_completion_receipt",
    }
    found: set[str] = set()
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        name = _call_name(call)
        if name not in guarded_calls:
            continue
        found.add(name)
        assert _keyword_is_name(call, "publication_guard", "assert_formal_numerical_policy"), (
            f"{name} at line {call.lineno} lacks the live publication guard"
        )
    assert found == guarded_calls


def test_stage09b_receipt_validation_propagates_guard_to_checkpoint_replay():
    gate_tree = ast.parse(
        (
            ROOT / "src" / "thermoroute" / "development_controls_gate.py"
        ).read_text(encoding="utf-8")
    )
    validator = _function(gate_tree, "validate_stage09b_completion_receipt")
    assert "publication_guard" in {
        argument.arg for argument in validator.args.kwonlyargs
    }
    member_validation = [
        node
        for node in ast.walk(validator)
        if isinstance(node, ast.Call)
        and _call_name(node) == "_validate_member_predictions"
    ]
    assert len(member_validation) == 1
    assert _keyword_is_name(
        member_validation[0], "publication_guard", "publication_guard"
    )

    member_validator = _function(gate_tree, "_validate_member_predictions")
    member_replays = [
        node
        for node in ast.walk(member_validator)
        if isinstance(node, ast.Call)
        and _call_name(node) == "_replay_member_best_state"
    ]
    assert len(member_replays) == 1
    assert _keyword_is_name(
        member_replays[0], "publication_guard", "publication_guard"
    )

    member_replay = _function(gate_tree, "_replay_member_best_state")
    checkpoint_loads = [
        node
        for node in ast.walk(member_replay)
        if isinstance(node, ast.Call)
        and _call_name(node) == "load_training_checkpoint"
    ]
    assert len(checkpoint_loads) == 1
    assert _keyword_is_name(
        checkpoint_loads[0], "publication_guard", "publication_guard"
    )

    suite_tree = ast.parse(
        (ROOT / "src" / "thermoroute" / "model_suite.py").read_text(
            encoding="utf-8"
        )
    )
    suite_validator = _function(suite_tree, "validate_model_suite_document")
    receipt_replays = [
        node
        for node in ast.walk(suite_validator)
        if isinstance(node, ast.Call)
        and _call_name(node) == "validate_stage09b_completion_receipt"
    ]
    assert len(receipt_replays) == 1
    assert _keyword_is_name(
        receipt_replays[0], "publication_guard", "publication_guard"
    )


def test_stage16_all_zero_wilcoxon_diagnostic_is_nonfatal():
    path = ROOT / "scripts" / "16_lstm_baseline.py"
    spec = importlib.util.spec_from_file_location(
        "stage16_zero_wilcoxon_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = module.np.asarray([0.1, 0.2, 0.3], dtype=float)
    assert module._safe_wilcoxon_pvalue(values, values.copy()) == 1.0


def test_opening_trusted_publications_have_pre_and_post_policy_guards():
    tree = ast.parse((ROOT / "src" / "thermoroute" / "opening.py").read_text(encoding="utf-8"))
    scorer = _function(tree, "isolated_score_and_receipt")
    parent, statement_block = _statement_context(scorer)

    directory_publications = [
        node
        for node in ast.walk(scorer)
        if isinstance(node, ast.Call) and _call_name(node) == "_atomic_publish_trusted_directory"
    ]
    assert len(directory_publications) == 1
    directory_statement = _containing_statement(directory_publications[0], parent)
    block = statement_block[directory_statement]
    position = block.index(directory_statement)
    assert position > 0 and _is_native_policy_assertion(block[position - 1])
    published_validation = next(
        index
        for index, statement in enumerate(block[position + 1 :], position + 1)
        if any(
            isinstance(node, ast.Call)
            and _call_name(node) == "_assert_validated_artifacts_published"
            for node in ast.walk(statement)
        )
    )
    assert published_validation + 1 < len(block) and _is_native_policy_assertion(
        block[published_validation + 1]
    )

    receipt_publications = [
        node
        for node in ast.walk(scorer)
        if isinstance(node, ast.Call)
        and _call_name(node) == "_atomic_create_bytes"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id in {"receipt_path", "sidecar_path"}
    ]
    assert len(receipt_publications) == 3
    for publication in receipt_publications:
        publication_statement = _containing_statement(publication, parent)
        preceding = _preceding_statement_on_execution_path(
            publication_statement, parent, statement_block
        )
        assert preceding is not None and _is_native_policy_assertion(preceding), (
            f"unguarded {publication.args[0].id} publication at line {publication.lineno}"
        )

    receipt_publication = next(
        call
        for call in receipt_publications
        if isinstance(call.args[0], ast.Name) and call.args[0].id == "receipt_path"
    )
    receipt_statement = _containing_statement(receipt_publication, parent)
    receipt_block = statement_block[receipt_statement]
    receipt_position = receipt_block.index(receipt_statement)
    assert receipt_position > 0 and _is_native_policy_assertion(receipt_block[receipt_position - 1])

    sidecar_positions = [
        receipt_block.index(statement)
        for statement in receipt_block
        if any(
            isinstance(node, ast.Call)
            and _call_name(node) == "_atomic_create_bytes"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "sidecar_path"
            for node in ast.walk(statement)
        )
    ]
    sidecar_positions = [item for item in sidecar_positions if item > receipt_position]
    assert len(sidecar_positions) == 1
    sidecar_position = sidecar_positions[0]
    assert any(
        _is_native_policy_assertion(statement)
        for statement in receipt_block[receipt_position + 1 : sidecar_position]
    )
    assert any(
        _is_native_policy_assertion(statement)
        for statement in receipt_block[sidecar_position + 1 :]
    )


def test_opening_release_replay_checks_policy_before_and_after_validation():
    tree = ast.parse((ROOT / "src" / "thermoroute" / "opening.py").read_text(encoding="utf-8"))
    verifier = _function(tree, "isolated_verify_release")
    validation_position = next(
        index
        for index, statement in enumerate(verifier.body)
        if any(
            isinstance(node, ast.Call) and _call_name(node) == "validate_opening_products"
            for node in ast.walk(statement)
        )
    )
    assert any(
        _is_native_policy_assertion(statement)
        or any(
            isinstance(node, ast.Call) and _call_name(node) == "assert_formal_numerical_policy"
            for node in ast.walk(statement)
        )
        for statement in verifier.body[:validation_position]
    )
    receipt_position = next(
        index
        for index, statement in enumerate(verifier.body)
        if any(
            isinstance(node, ast.Call) and _call_name(node) == "_read_completed_receipt"
            for node in ast.walk(statement)
        )
    )
    result_position = next(
        index
        for index, statement in enumerate(verifier.body)
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "result" for target in statement.targets
        )
    )
    assert validation_position < receipt_position < result_position
    assert any(
        _is_native_policy_assertion(statement)
        for statement in verifier.body[receipt_position + 1 : result_position]
    )
    returns = [
        (index, statement)
        for index, statement in enumerate(verifier.body)
        if isinstance(statement, ast.Return)
    ]
    assert len(returns) == 1
    return_position, returned = returns[0]
    assert isinstance(returned.value, ast.Name) and returned.value.id == "result"
    assert return_position > result_position
    assert _is_native_policy_assertion(verifier.body[return_position - 1])


@pytest.mark.parametrize(
    ("relative", "publishers"),
    FINAL_PUBLICATION_GUARDS.items(),
)
def test_final_receipt_publication_has_same_block_native_policy_guard(
    relative,
    publishers,
):
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    parent: dict[ast.AST, ast.AST] = {}
    statement_block: dict[ast.stmt, list[ast.stmt]] = {}
    for node in ast.walk(tree):
        for _field, value in ast.iter_fields(node):
            children = value if isinstance(value, list) else [value]
            for child in children:
                if isinstance(child, ast.AST):
                    parent[child] = node
            if isinstance(value, list) and all(isinstance(child, ast.stmt) for child in value):
                for child in value:
                    statement_block[child] = value

    found: set[str] = set()
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        name = _call_name(call)
        if name not in publishers:
            continue
        found.add(name)
        statement: ast.AST = call
        while not isinstance(statement, ast.stmt):
            statement = parent[statement]
        block = statement_block[statement]
        position = block.index(statement)
        assert any(
            _is_native_policy_assertion(item)
            for item in block[max(0, position - 2) : position]
        )
    assert found == publishers


@pytest.mark.parametrize(
    ("relative", "minimum_assertions"),
    RUNTIME_ENFORCED_ENTRYPOINTS.items(),
)
def test_formal_numeric_entrypoints_use_full_native_runtime_contract(
    relative,
    minimum_assertions,
):
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    imports = {
        (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert ("thermoroute.repro", "configure_deterministic_runtime") in imports
    assert ("thermoroute.train", "configure_deterministic_runtime") not in imports
    assertions = sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "assert_formal_numerical_policy"
        for node in ast.walk(tree)
    )
    assert assertions >= minimum_assertions


def test_stage09_formal_publication_gate_is_cpu_only():
    path = ROOT / "scripts" / "09_usgs_experiment.py"
    spec = importlib.util.spec_from_file_location("stage09_cpu_gate_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    canonical = ROOT / "data_usgs" / "panel_usgs_120v2.parquet"
    assert module.formal_publication_candidate(
        panel_path=canonical, training_device="cpu", exploratory=False
    )
    for device in ("auto", "mps", "cuda"):
        assert not module.formal_publication_candidate(
            panel_path=canonical, training_device=device, exploratory=False
        )
    assert not module.formal_publication_candidate(
        panel_path=canonical, training_device="cpu", exploratory=True
    )


def test_stage09_incomplete_and_exploratory_outputs_are_run_scoped():
    path = ROOT / "scripts" / "09_usgs_experiment.py"
    spec = importlib.util.spec_from_file_location("stage09_output_scope_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    canonical_panel = ROOT / "data_usgs" / "panel_usgs_120v2.parquet"
    requested = {
        "requested_predictions": (
            ROOT / module.STAGE09_ARTIFACT_PATHS["predictions"]
        ),
        "requested_scores": ROOT / module.STAGE09_ARTIFACT_PATHS["scores"],
        "requested_report": ROOT / module.STAGE09_ARTIFACT_PATHS["report"],
    }
    complete = {
        "panel_path": canonical_panel,
        "training_device": "cpu",
        "exploratory": False,
        "canonical_output_paths": True,
        "seeds": len(module.C.USGS_SEEDS),
        "station_sampling": "balanced",
        "delta_scale": module.DELTA_SCALE,
        "ablations": True,
        "air2stream": False,
    }
    assert module.formal_stage09_configuration(**complete)

    run_dir = (
        ROOT / "outputs" / "runs" / "09_usgs_experiment" / "unit-test-run"
    )
    formal_paths = module.resolve_stage09_publication_paths(
        root=ROOT,
        run_dir=run_dir,
        **requested,
        formal_output_eligible=True,
    )
    assert formal_paths == {
        "predictions": requested["requested_predictions"].resolve(),
        "scores": requested["requested_scores"].resolve(),
        "report": requested["requested_report"].resolve(),
        "lightgbm_selection": (
            ROOT / module.STAGE09_ARTIFACT_PATHS["lightgbm_selection"]
        ).resolve(),
    }

    incomplete_variants = (
        {"exploratory": True},
        {"panel_path": canonical_panel.with_name("another_panel.parquet")},
        {"training_device": "mps"},
        {"canonical_output_paths": False},
        {"seeds": len(module.C.USGS_SEEDS) - 1},
        {"station_sampling": "natural"},
        {"delta_scale": module.DELTA_SCALE + 0.01},
        {"ablations": False},
        {"air2stream": True},
    )
    canonical_destinations = set(formal_paths.values())
    for change in incomplete_variants:
        configuration = {**complete, **change}
        eligible = module.formal_stage09_configuration(**configuration)
        assert not eligible, change
        diagnostic_paths = module.resolve_stage09_publication_paths(
            root=ROOT,
            run_dir=run_dir,
            **requested,
            formal_output_eligible=eligible,
        )
        assert set(diagnostic_paths) == {
            "predictions",
            "scores",
            "report",
            "lightgbm_selection",
        }
        assert {
            destination.parent for destination in diagnostic_paths.values()
        } == {run_dir / "diagnostic_outputs"}
        assert not (set(diagnostic_paths.values()) & canonical_destinations)

    with pytest.raises(
        ValueError, match="formal Stage-9 publication requires"
    ):
        module.resolve_stage09_publication_paths(
            root=ROOT,
            run_dir=run_dir,
            requested_predictions=run_dir / "custom.parquet",
            requested_scores=requested["requested_scores"],
            requested_report=requested["requested_report"],
            formal_output_eligible=True,
        )


def test_stage09_main_resolves_authority_scope_before_result_publication():
    tree = ast.parse(
        (ROOT / "scripts" / "09_usgs_experiment.py").read_text(encoding="utf-8")
    )
    main = _function(tree, "main")
    resolver_calls = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and _call_name(node) == "resolve_stage09_publication_paths"
    ]
    assert len(resolver_calls) == 1
    assert _keyword_is_name(
        resolver_calls[0],
        "formal_output_eligible",
        "formal_configuration_complete",
    )

    for name, key in (
        ("output_predictions", "predictions"),
        ("score_path", "scores"),
        ("report_path", "report"),
        ("lightgbm_selection_path", "lightgbm_selection"),
    ):
        assignments = [
            node
            for node in ast.walk(main)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            )
        ]
        assert len(assignments) == 1, name
        value = assignments[0].value
        assert (
            isinstance(value, ast.Subscript)
            and isinstance(value.value, ast.Name)
            and value.value.id == "publication_paths"
            and isinstance(value.slice, ast.Constant)
            and value.slice.value == key
        ), name

    formal_assignments = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "formal_configuration_complete"
            for target in node.targets
        )
    ]
    assert len(formal_assignments) == 1
    assert isinstance(formal_assignments[0].value, ast.Call)
    assert _call_name(formal_assignments[0].value) == "formal_stage09_configuration"

    formal_completion = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "formal_complete"
            for target in node.targets
        )
    ]
    assert len(formal_completion) == 1
    assert any(
        isinstance(node, ast.Name)
        and node.id == "formal_configuration_complete"
        for node in ast.walk(formal_completion[0].value)
    )

    formal_publication_blocks = [
        node
        for node in main.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "formal_complete"
        and any(
            isinstance(descendant, ast.Call)
            and _call_name(descendant) == "complete_stage09_transaction"
            for descendant in ast.walk(node)
        )
    ]
    assert len(formal_publication_blocks) == 1
    publisher_names = {
        _call_name(node)
        for node in ast.walk(formal_publication_blocks[0])
        if isinstance(node, ast.Call)
    }
    assert {
        "write_component_pointer",
        "publish_stage09_completion_receipt",
        "complete_stage09_transaction",
    } <= publisher_names


def _artifact_snapshot() -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    for relative in (
        "outputs/runs",
        "outputs/models",
        "outputs/predictions",
        "outputs/tables",
    ):
        base = ROOT / relative
        if not base.exists():
            continue
        base_stat = base.stat()
        snapshot[relative] = (-1, int(base_stat.st_mtime_ns))
        for path in base.rglob("*"):
            stat = path.stat()
            if path.is_dir():
                snapshot[path.relative_to(ROOT).as_posix()] = (
                    -1,
                    int(stat.st_mtime_ns),
                )
            elif path.is_file():
                snapshot[path.relative_to(ROOT).as_posix()] = (
                    int(stat.st_size),
                    int(stat.st_mtime_ns),
                )
    return snapshot


@pytest.mark.parametrize(("relative", "_worker_argument"), FORMAL_ENTRYPOINTS)
def test_formal_entrypoint_help_is_zero_training_and_zero_artifact_output(
    relative,
    _worker_argument,
):
    before = _artifact_snapshot()
    result = subprocess.run(
        [sys.executable, str(ROOT / relative), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert _artifact_snapshot() == before


@pytest.mark.parametrize(("relative", "worker_argument"), FORMAL_ENTRYPOINTS)
def test_formal_worker_cannot_bypass_controller_handshake(
    tmp_path,
    relative,
    worker_argument,
):
    cache = tmp_path / "untrusted-cache"
    cache.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-X",
            f"pycache_prefix={cache}",
            str(ROOT / relative),
            worker_argument,
            "--help",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=os.environ.copy(),
    )
    assert result.returncode != 0
    assert "worker handshake is incomplete" in result.stderr


@pytest.mark.parametrize(("relative", "worker_argument"), FORMAL_ENTRYPOINTS)
def test_formal_worker_rejects_every_inherited_environment_variable(
    tmp_path,
    relative,
    worker_argument,
):
    cache = (tmp_path / "controller-cache").resolve()
    cache.mkdir()
    nonce = "formal-controller-nonce"
    (cache / ".controller-nonce").write_text(nonce, encoding="utf-8")
    cache_key, nonce_key = FORMAL_WORKER_ENVIRONMENT_KEYS[relative]
    environment = {
        "PATH": os.defpath,
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "TMPDIR": str(cache),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
        cache_key: str(cache),
        nonce_key: nonce,
        "UNDECLARED_HOST_INJECTION": "must-be-rejected",
    }
    if relative == "scripts/16_lstm_baseline.py":
        environment["WORKER_THREADS"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-X",
            f"pycache_prefix={cache}",
            str(ROOT / relative),
            worker_argument,
            "--help",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    assert result.returncode != 0
    assert "formal worker isolation contract failed" in result.stderr


def test_formal_controllers_never_copy_the_host_environment():
    for relative, _worker_argument in FORMAL_ENTRYPOINTS:
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "os.environ.copy()" not in source, relative
        assert "dict(os.environ)" in source, relative

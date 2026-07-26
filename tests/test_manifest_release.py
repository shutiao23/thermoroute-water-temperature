from __future__ import annotations

import ast
import importlib.util
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SCRIPT = ROOT / "scripts" / "14_manifest.py"
VERIFY_SCRIPT = ROOT / "scripts" / "verify_release.py"
ZIP_SCRIPT = ROOT / "scripts" / "deterministic_zip.py"
MAKE_RELEASE_SCRIPT = ROOT / "scripts" / "make_release_archive.sh"
FIXTURE_DEVELOPMENT_INPUT_CLOSURE_SHA256 = "6" * 64
FIXTURE_DEVELOPMENT_INPUT_CLOSURE_FILE_COUNT = 1


def _stage09b_fixture_config(
    expected_bridge: dict[str, str],
    *,
    eval_batch_size: int = 2,
    input_closure_sha256: str = FIXTURE_DEVELOPMENT_INPUT_CLOSURE_SHA256,
    input_closure_file_count: int = FIXTURE_DEVELOPMENT_INPUT_CLOSURE_FILE_COUNT,
) -> dict[str, object]:
    """Build the real formal Stage-09b configuration for release fixtures."""
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from thermoroute import config as C
        from thermoroute.development_controls import (
            DEVELOPMENT_DISCLOSURE,
            FULL_VARIABLES,
            TRAIN_CONFIG,
            architecture_template,
            assert_parameter_budgets,
            declared_arms,
            expected_member_registry,
        )
    finally:
        sys.path.pop(0)
    arms = declared_arms()
    hash_policy = (
        "canonical-sort-identity-collections-independent-of-hash-secret"
    )
    formal_policy = {
        "thread_environment": {
            name: "1" for name in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
            )
        },
        "cublas_workspace_config": ":4096:8",
        "python_hash_environment_declaration": "0",
        "python_hash_randomization_enabled": True,
        "python_hash_policy": hash_policy,
        "required": {
            "threads": 1,
            "cublas_workspace_config": ":4096:8",
            "python_hash_policy": hash_policy,
            "torch_deterministic_algorithms": True,
            "tf32": False,
            "float32_matmul_precision": "highest",
        },
        "torch": {
            "num_threads": 1,
            "num_interop_threads": 1,
            "deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
        },
    }
    return {
        "stage": "09b_development_controls",
        "format": "thermoroute.development-controls.v2",
        "execution_role": (
            "prelabel_relative_to_unopened_post_2020_confirmation"
        ),
        "evidence_role": "development_only_exploratory",
        "development_disclosure": DEVELOPMENT_DISCLOSURE,
        "panel_date_range": ["2006-01-01", "2020-12-31"],
        "development_evaluation_interval": list(C.SPLIT.test),
        "blind_or_confirmatory": False,
        "suite_pointer_written": False,
        "training_device": "cpu",
        "variables": list(FULL_VARIABLES),
        "context_length": C.CONTEXT_LENGTH,
        "horizons": list(C.HORIZONS),
        "time_split": C.SPLIT.as_dict(),
        "station_sampling": "balanced",
        "selection_metric": "station_macro",
        "train_config": asdict(TRAIN_CONFIG),
        "arms": [asdict(arm) for arm in arms],
        "expected_member_registry": [
            list(member) for member in expected_member_registry(arms)
        ],
        "parameter_counts": assert_parameter_budgets(arms, n_stations=120),
        "architecture_templates": {
            arm.arm_id: architecture_template(arm, n_stations=120)
            for arm in arms
        },
        "parameter_match_tolerance_fraction": 0.02,
        "architecture_candidates_per_arm": 1,
        "historical_tuning_budget_equalized": False,
        "development_predictor_bridge": expected_bridge,
        "formal_numerical_policy": formal_policy,
        "eval_batch_size": eval_batch_size,
        "input_closure_sha256": input_closure_sha256,
        "input_closure_file_count": input_closure_file_count,
    }


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage09b_independent_release_mirror_matches_central_contract() -> None:
    """Prevent the archive-independent verifier from drifting from Stage-09b."""
    from thermoroute.chronology import STAGE09B_MEMBERS
    from thermoroute.development_controls import expected_member_registry

    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_stage09b_member_contract_test",
    )
    producer_members = expected_member_registry()
    release_members = verifier._stage09b_release_members()
    assert len(producer_members) == 45
    assert release_members == producer_members
    assert tuple(STAGE09B_MEMBERS) == producer_members


def _write_fixture(root: Path) -> Path:
    files = {
        "src/thermoroute/config.py": (
            "from dataclasses import dataclass\n"
            "HORIZONS = (1, 3, 7)\n"
            "STATIONS = ('s1',)\n"
            "@dataclass(frozen=True)\n"
            "class TrainConfig:\n"
            "    seed: int = 7\n"
            "TRAIN = TrainConfig()\n"
        ),
        "scripts/demo.py": "print('fixture')\n",
        "tests/test_demo.py": "def test_demo(): assert True\n",
        "pyproject.toml": "[project]\nname='fixture'\nversion='0.0.0'\n",
        "requirements.txt": "pandas>=2\n",
        "requirements-lock.txt": "pandas==2.2.2\n",
        "README.md": "# fixture\n",
        "data_usgs/station_input.csv": "x\n1\n",
        "outputs/predictions/copied_legacy.parquet": "renamed old bytes\n",
        "outputs/tables/result.csv": "score\n1.0\n",
    }
    for rel, payload in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    return root / "outputs" / "manifest.json"


def _write_bytes(root: Path, relative: str, payload: bytes = b"fixture\n") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _write_canonical_json(
    verifier, root: Path, relative: str, document: object
) -> Path:
    return _write_bytes(
        root, relative, verifier._canonical_json_bytes(document)
    )


def _commit_git_fixture(root: Path, message: str) -> str:
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c", "user.name=Fixture",
            "-c", "user.email=fixture@example.invalid",
            "commit", "-q", "-m", message,
        ],
        cwd=root,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def _binding(verifier, root: Path, relative: str) -> dict[str, str]:
    path = root / relative
    return {"path": relative, "sha256": verifier.sha256_file(path)}


def _write_formal_meteorology_snapshot(
    verifier,
    root: Path,
    index_path: str,
    *,
    provider: str,
    url: str,
    payload: bytes,
) -> set[str]:
    request = {
        "schema_version": 1,
        "provider": provider,
        "method": "GET",
        "url": url,
        "headers": {
            "User-Agent": "ThermoRoute/1.0 Route-A pre-label meteorology"
        },
    }
    request_sha = hashlib.sha256(
        verifier._canonical_json_bytes(request)
    ).hexdigest()
    base = PurePosixPath(index_path).parent
    transaction = base / provider / request_sha
    metadata_path = (transaction / "metadata.json").as_posix()
    response_path = (transaction / "response.bin").as_posix()
    retrieved = "2026-07-26T00:00:00+00:00"
    metadata_payload = verifier._canonical_json_bytes({
        "schema_version": 2,
        "request": request,
        "request_sha256": request_sha,
        "retrieved_at_utc": retrieved,
        "http_status": 200,
        "response_headers": {},
        "byte_count": len(payload),
        "response_sha256": hashlib.sha256(payload).hexdigest(),
        "response_file": "response.bin",
        "final_url": url,
        "retrieval_semantics": "DIRECT_HTTP_RESPONSE",
    })
    _write_bytes(root, metadata_path, metadata_payload)
    _write_bytes(root, response_path, payload)
    _write_bytes(
        root,
        index_path,
        verifier._canonical_json_bytes({
            "schema_version": 2,
            "snapshot_count": 1,
            "records": [{
                "provider": provider,
                "request_sha256": request_sha,
                "response_sha256": hashlib.sha256(payload).hexdigest(),
                "metadata_sha256": hashlib.sha256(metadata_payload).hexdigest(),
                "metadata_byte_count": len(metadata_payload),
                "retrieved_at_utc": retrieved,
                "byte_count": len(payload),
                "request": request,
                "metadata_path": PurePosixPath(metadata_path).relative_to(base).as_posix(),
                "response_path": PurePosixPath(response_path).relative_to(base).as_posix(),
            }],
        }),
    )
    return {index_path, metadata_path, response_path}


def _write_development_panel_fixture(root: Path) -> tuple[str, str, str]:
    """Write a small canonical panel spanning every frozen fit interval."""
    panel_relative = "data_usgs/panel_usgs_120v2.parquet"
    registry_relative = "data_usgs/station_registry_v1.csv"
    spec_relative = "data_usgs/frozen_panel_v1.json"
    records = [
        {
            "DATE": pd.Timestamp(year=year, month=month, day=15),
            "site_id": "n90",
            "WTEMP": float(5.0 + month + (year - 2006) / 100.0),
            "FLOW": 1.0,
            "WLEVEL": float("nan"),
            "TEMP": 10.0,
            "PRCP": 0.0,
            "WDSP": 1.0,
            "RHMEAN": 50.0,
            "DH": 100.0,
        }
        for year in range(2006, 2021)
        for month in range(1, 13)
    ]
    split_issue_dates = {
        "val": [pd.Timestamp("2016-01-01"), pd.Timestamp("2016-02-01")],
        "calib": [
            pd.Timestamp("2018-01-01") + pd.Timedelta(days=index)
            for index in range(100)
        ],
        "test": [pd.Timestamp("2019-01-01"), pd.Timestamp("2019-02-01")],
    }
    forecast_target_dates = {
        issue_date + pd.Timedelta(days=horizon)
        for issue_dates in split_issue_dates.values()
        for issue_date in issue_dates
        for horizon in (1, 3, 7)
    }
    records.extend(
        {
            "DATE": target_date,
            "site_id": "n90",
            # Stage 9 persists float32-canonical truth in a float64 Parquet
            # column. This value deliberately differs across those two
            # representations, so every positive replay exercises that rule.
            "WTEMP": 0.1,
            "FLOW": 1.0,
            "WLEVEL": float("nan"),
            "TEMP": 10.0,
            "PRCP": 0.0,
            "WDSP": 1.0,
            "RHMEAN": 50.0,
            "DH": 100.0,
        }
        for target_date in sorted(forecast_target_dates)
    )
    panel_path = root / panel_relative
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel = (
        pd.DataFrame.from_records(records)
        .drop_duplicates(["DATE", "site_id"], keep="last")
        .sort_values(["site_id", "DATE"], kind="mergesort")
        .reset_index(drop=True)
    )
    panel.to_parquet(panel_path, index=False)
    _write_bytes(
        root,
        registry_relative,
        b"site_no,legacy_site_id,huc2\n01073319,n90,01\n",
    )
    registry_path = root / registry_relative
    specification = {
        "schema_version": 1,
        "panel_id": "usgs120-development-v1",
        "panel": {
            "path": "panel_usgs_120v2.parquet",
            "format": "parquet",
            "legacy_station_key": "site_id",
            "sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
            "date_start": "2006-01-01",
            "date_end": "2020-12-31",
            "row_count": len(panel),
            "station_count": 1,
            "required_columns": list(panel.columns),
        },
        "station_registry": {
            "path": "station_registry_v1.csv",
            "primary_key": "site_no",
            "legacy_alias": "legacy_site_id",
            "sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
            "station_count": 1,
        },
    }
    _write_bytes(
        root,
        spec_relative,
        json.dumps(specification, sort_keys=True).encode("utf-8") + b"\n",
    )
    return panel_relative, registry_relative, spec_relative


def _fixture_run_identity(
    verifier,
    configuration: dict[str, object],
    *,
    panel_sha256: str,
    registry_sha256: str,
    source_sha256: str,
    runtime_sha256: str,
    input_closure_sha256: str,
) -> dict[str, str]:
    stable = {
        "schema_version": "thermoroute.run.v2",
        "panel_sha256": panel_sha256,
        "registry_sha256": registry_sha256,
        "config_sha256": verifier._sha256_json(configuration),
        "source_sha256": source_sha256,
        "runtime_sha256": runtime_sha256,
        "input_closure_sha256": input_closure_sha256,
    }
    return {"run_id": verifier._sha256_json(stable)[:20], **stable}


def _write_development_model_fixtures(
    verifier, root: Path, *, runtime_sha256: str
) -> tuple[dict[str, list[dict[str, object]]], set[str]]:
    """Create a complete suite registry backed by producer-shaped metadata."""
    panel_relative, registry_relative, spec_relative = (
        _write_development_panel_fixture(root)
    )
    panel_contracts = verifier._independent_development_panel_contracts(
        (root / panel_relative).read_bytes(),
        (root / registry_relative).read_bytes(),
        (root / spec_relative).read_bytes(),
    )
    panel_truth = pd.read_parquet(root / panel_relative).set_index("DATE")["WTEMP"]
    model_entries: dict[str, list[dict[str, object]]] = {}
    artifact_paths: set[str] = set()
    builtins = {"Persistence", "DampedPersistence", "Climatology"}
    primary_model_order = (
        "Persistence", "DampedPersistence", "Climatology", "LightGBM",
        "LSTM", "ThermoRoute",
    )
    control_model_order = (
        "DampedPriorOnly", "TR-noDynamicPrior", "TR-fixedKappa",
        "TR-noRouter", "TR-noMoE", "TR-noTCN", "TR-unbounded",
    )
    for cohort in ("temporal", "external"):
        entries: list[dict[str, object]] = []
        model_order = (
            primary_model_order
            if cohort == "external"
            else (*primary_model_order, *control_model_order)
        )
        assert set(model_order) == verifier._required_model_ids(cohort)
        for model in model_order:
            if model in builtins:
                entries.append({"model_id": model, "executor": "builtin"})
                continue
            members = verifier.DEVELOPMENT_REPLAY_MODEL_CONTRACTS[cohort][
                model
            ][1]
            executor = (
                "lightgbm_bundle" if model == "LightGBM"
                else "lstm_bundle" if model == "LSTM"
                else "thermoroute_bundle"
            )
            replay_atol = verifier.DEVELOPMENT_REPLAY_MODEL_CONTRACTS[cohort][
                model
            ][2]
            slug = model.replace("-", "_").lower()
            prediction = f"outputs/development/{cohort}_{slug}.parquet"
            prediction_sidecar = prediction + ".meta.json"
            prediction_path = root / prediction
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            prediction_records = []
            split_dates = {
                "val": [pd.Timestamp("2016-01-01"), pd.Timestamp("2016-02-01")],
                "calib": [
                    pd.Timestamp("2018-01-01") + pd.Timedelta(days=index)
                    for index in range(100)
                ],
                "test": [pd.Timestamp("2019-01-01"), pd.Timestamp("2019-02-01")],
            }
            for split, issue_dates in split_dates.items():
                for issue_date in issue_dates:
                    for horizon in (1, 3, 7):
                        for seed in range(members):
                            target_date = issue_date + pd.Timedelta(days=horizon)
                            panel_value = float(panel_truth.loc[target_date])
                            truth = float(np.float32(panel_value))
                            assert truth != panel_value
                            prediction_records.append({
                                "model": model,
                                "scope": "development_only_2006_2020",
                                "feature_set": "USGS",
                                "seed": seed,
                                "site_id": "01073319",
                                "horizon": horizon,
                                "split": split,
                                "issue_date": issue_date,
                                "target_date": target_date,
                                "y_true": truth,
                                "y_pred": truth,
                                "q05": truth - 2.0,
                                "q50": truth,
                                "q95": truth + 2.0,
                                "p_exceed": 0.25,
                            })
            prediction_frame = pd.DataFrame.from_records(prediction_records)
            prediction_frame.to_parquet(prediction_path, index=False)
            _write_canonical_json(verifier, root, prediction_sidecar, {})
            artifact_paths.update({prediction, prediction_sidecar})
            development_prediction = {
                "artifact": {
                    **_binding(verifier, root, prediction),
                    "sidecar": _binding(verifier, root, prediction_sidecar),
                },
                "rows": len(prediction_frame),
                "selection": {"model": model, "seeds": list(range(members))},
                "forecast_key_columns": [
                    "site_id", "horizon", "issue_date", "target_date"
                ],
                "prediction_columns": list(
                    verifier.DEVELOPMENT_REPLAY_PREDICTION_COLUMNS
                ),
                "forecast_key_registry_sha256": "a" * 64,
                "prediction_sha256": "b" * 64,
                "max_abs_difference": 0.0,
                "atol": replay_atol,
            }
            cqr_group = "__pooled__" if cohort == "external" else "01073319"
            raw_cqr_offsets = {
                f"{cqr_group}|{horizon}": -2.0 for horizon in (1, 3, 7)
            }
            cqr_offsets = {key: 0.0 for key in raw_cqr_offsets}
            cqr_audit = verifier._independent_cqr_audit(
                raw_cqr_offsets, cqr_offsets
            )
            constant = 0.5 / 101.0
            event_calibrators = {
                str(horizon): {
                    "intercept": math.log(constant / (1.0 - constant)),
                    "slope": 0.0,
                    "constant": constant,
                }
                for horizon in (1, 3, 7)
            }
            calibration_fit_contract = {
                "format": "thermoroute.route-a-calibration-fit.v1",
                "model_training_interval_inclusive": [
                    "2006-01-01", "2015-12-31"
                ],
                "hyperparameter_selection_interval_inclusive": [
                    "2016-01-01", "2017-12-31"
                ],
                "source_split": "calib",
                "issue_date_interval_inclusive": [
                    "2018-01-01", "2018-12-31"
                ],
                "post_2018_rows_allowed": False,
                "member_aggregation_before_fit": "equal_weight_member_mean",
                "event_reference_fit_interval_inclusive": [
                    "2006-01-01", "2018-12-31"
                ],
                "cqr": {
                    "alpha": 0.10,
                    "grouping": (
                        "pooled_by_horizon"
                        if cohort == "external" else "site_by_horizon"
                    ),
                    "target_date_boundary_purge": "2018-12-31",
                    "deployed_offset": "qhat_plus=max(raw_qhat,0)",
                },
                "event_probability": {
                    "method": "Platt_logistic_by_horizon",
                    "grouping": "pooled_by_horizon",
                    "fit_interval": ["2018-01-01", "2018-12-31"],
                    "fit_weighting": "equal_total_weight_per_station",
                },
            }
            cqr_metadata = {
                "panel_sha256": verifier.sha256_file(root / panel_relative),
                "registry_sha256": verifier.sha256_file(root / registry_relative),
                "event_thresholds": panel_contracts[cohort][
                    "event_thresholds"
                ],
                "event_reference_climatology": panel_contracts[cohort][
                    "event_reference_climatology"
                ],
                "event_calibrators": event_calibrators,
                "conformal_offsets": cqr_offsets,
                "conformal_policy": verifier.CQR_POLICY,
                "conformal_offset_audit": cqr_audit,
                "calibration_fit_contract": calibration_fit_contract,
            }
            if cohort == "external":
                cqr_metadata["event_threshold_estimator"] = {
                    "method": "pooled_training_empirical_quantile_v1",
                    "quantile": 0.90,
                    "pool_weighting": "equal_weight_per_finite_training_row",
                    "station_balanced": False,
                }
            if executor == "lightgbm_bundle":
                bundle = f"outputs/models/{cohort}/{slug}/manifest.json"
                member_names = [f"seed{seed}" for seed in range(members)]
                model_bindings: dict[str, object] = {}
                audit_members: dict[str, object] = {}
                for member in member_names:
                    heads: dict[str, object] = {}
                    for head in ("point", "q05", "q50", "q95", "event"):
                        relative = (
                            f"outputs/models/{cohort}/{slug}/{member}_h1_{head}.txt"
                        )
                        _write_bytes(root, relative, f"{member}/{head}\n".encode())
                        heads[head] = {
                            "path": PurePosixPath(relative).name,
                            "sha256": verifier.sha256_file(root / relative),
                        }
                        artifact_paths.add(relative)
                    model_bindings[member] = {"1": heads}
                    audit_members[member] = {
                        "1": {
                            "rows": 12,
                            "forecast_key_sha256": "c" * 64,
                            "raw_prediction_sha256": "d" * 64,
                            "q05_above_q50_count": 0,
                            "q50_above_q95_count": 0,
                            "any_crossing_count": 0,
                            "any_crossing_rate": 0.0,
                            "maximum_crossing_gap_c": 0.0,
                        }
                    }
                crossing = {
                    "format": "thermoroute.raw-quantile-crossing-audit.v1",
                    "scope": "development_export_rows_before_repair",
                    "key_columns": [
                        "site_id", "horizon", "split", "issue_date", "target_date"
                    ],
                    "repair_method": "median_preserving_endpoint_clip_v1",
                    "members": audit_members,
                }
                crossing["audit_sha256"] = verifier._sha256_json(crossing)
                manifest = {
                    "format": "thermoroute.lightgbm-bundle.v2",
                    "training_device": "cpu",
                    "runtime_sha256": runtime_sha256,
                    "heads": ["point", "q05", "q50", "q95", "event"],
                    "members": member_names,
                    "member_count": members,
                    "horizons": [1],
                    "quantile_repair": {
                        "method": "median_preserving_endpoint_clip_v1",
                        "version": 1,
                        "nominal_head_levels": {
                            "q05": 0.05, "q50": 0.50, "q95": 0.95,
                        },
                        "q05_operation": "minimum(raw_q05,raw_q50)",
                        "q50_operation": "raw_q50_unchanged",
                        "q95_operation": "maximum(raw_q95,raw_q50)",
                        "nominal_median_preserved_exactly": True,
                    },
                    "raw_quantile_crossing_audit": crossing,
                    "models": model_bindings,
                    "development_prediction": development_prediction,
                    **cqr_metadata,
                }
                _write_canonical_json(verifier, root, bundle, manifest)
                artifact_paths.add(bundle)
                artifact = _binding(verifier, root, bundle)
            else:
                directory = f"outputs/models/{cohort}/{slug}"
                weights = f"{directory}/weights.pt"
                metadata_path = f"{directory}/metadata.json"
                _write_bytes(root, weights, f"{cohort}/{model} weights\n".encode())
                metadata = {
                    "training_device": "cpu",
                    "runtime_sha256": runtime_sha256,
                    "member_count": members,
                    "weights_sha256": verifier.sha256_file(root / weights),
                    "development_prediction": development_prediction,
                    **cqr_metadata,
                }
                _write_canonical_json(verifier, root, metadata_path, metadata)
                artifact_paths.update({weights, metadata_path})
                artifact = {
                    "path": directory,
                    "metadata_sha256": verifier.sha256_file(root / metadata_path),
                    "weights_sha256": verifier.sha256_file(root / weights),
                }
            entries.append({
                "model_id": model,
                "executor": executor,
                "member_count": members,
                "artifact": artifact,
            })
        model_entries[cohort] = entries
    return model_entries, artifact_paths


def _write_stage25_gate_fixture(
    verifier,
    root: Path,
    *,
    model_entries: dict[str, list[dict[str, object]]],
    development_contract: dict[str, object],
    source_sha256: str,
    runtime_sha256: str,
) -> tuple[dict[str, str], set[str]]:
    """Create the exact canonical Stage-25 2+2+76 closure for release tests."""
    input_closure_sha256 = verifier._compose_input_closure_digest({
        "development": FIXTURE_DEVELOPMENT_INPUT_CLOSURE_SHA256,
    })
    configuration = verifier._stage25_expected_formal_configuration(
        development_contract["predictor_bridge"],
        input_closure_sha256=input_closure_sha256,
        input_closure_file_count=FIXTURE_DEVELOPMENT_INPUT_CLOSURE_FILE_COUNT,
    )
    identity = _fixture_run_identity(
        verifier,
        configuration,
        panel_sha256=development_contract["panel"]["sha256"],
        registry_sha256=development_contract["registry"]["sha256"],
        source_sha256=source_sha256,
        runtime_sha256=runtime_sha256,
        input_closure_sha256=input_closure_sha256,
    )
    run_id = identity["run_id"]
    feature_order = ["WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP"]
    created: set[str] = set()

    learned = {
        str(entry["model_id"]): entry
        for entry in model_entries["external"]
        if entry["executor"] != "builtin"
    }
    canonical_entries: dict[str, dict[str, object]] = {}
    for model_id, executor, slug in (
        ("ThermoRoute", "thermoroute_bundle", "thermoroute"),
        ("LSTM", "lstm_bundle", "lstm"),
    ):
        old_artifact = learned[model_id]["artifact"]
        old_directory = root / old_artifact["path"]
        directory = f"outputs/models/external_{slug}_bundle_{run_id}"
        metadata = f"{directory}/metadata.json"
        weights = f"{directory}/weights.pt"
        _write_bytes(root, metadata, (old_directory / "metadata.json").read_bytes())
        _write_bytes(root, weights, (old_directory / "weights.pt").read_bytes())
        created.update({metadata, weights})
        canonical_entries[model_id] = {
            "model_id": model_id,
            "executor": executor,
            "raw_feature_order": feature_order,
            "member_count": 5,
            "artifact": {
                "path": directory,
                "metadata_sha256": verifier.sha256_file(root / metadata),
                "weights_sha256": verifier.sha256_file(root / weights),
            },
        }

    old_lightgbm = learned["LightGBM"]["artifact"]
    old_manifest = json.loads(
        (root / old_lightgbm["path"]).read_text(encoding="utf-8")
    )
    lightgbm_directory = f"outputs/models/external_lightgbm_bundle_{run_id}"
    lightgbm_models: dict[str, object] = {}
    for seed in range(5):
        member = f"seed{seed}"
        lightgbm_models[member] = {}
        for horizon in (1, 3, 7):
            heads: dict[str, dict[str, str]] = {}
            for head in ("point", "q05", "q50", "q95", "event"):
                filename = f"{member}_h{horizon}_{head}.txt"
                relative = f"{lightgbm_directory}/{filename}"
                _write_bytes(
                    root,
                    relative,
                    f"{member}/h{horizon}/{head}\n".encode(),
                )
                created.add(relative)
                heads[head] = {
                    "path": filename,
                    "sha256": verifier.sha256_file(root / relative),
                }
            lightgbm_models[member][str(horizon)] = heads
    old_manifest["members"] = [f"seed{seed}" for seed in range(5)]
    old_manifest["member_count"] = 5
    old_manifest["horizons"] = [1, 3, 7]
    old_manifest["raw_feature_order"] = feature_order
    old_manifest["models"] = lightgbm_models
    audit_members = {
        f"seed{seed}": {
            str(horizon): {
                "rows": 12,
                "forecast_key_sha256": "c" * 64,
                "raw_prediction_sha256": "d" * 64,
                "q05_above_q50_count": 0,
                "q50_above_q95_count": 0,
                "any_crossing_count": 0,
                "any_crossing_rate": 0.0,
                "maximum_crossing_gap_c": 0.0,
            }
            for horizon in (1, 3, 7)
        }
        for seed in range(5)
    }
    crossing_audit = {
        "format": "thermoroute.raw-quantile-crossing-audit.v1",
        "scope": "development_export_rows_before_repair",
        "key_columns": [
            "site_id", "horizon", "split", "issue_date", "target_date"
        ],
        "repair_method": "median_preserving_endpoint_clip_v1",
        "members": audit_members,
    }
    crossing_audit["audit_sha256"] = verifier._sha256_json(crossing_audit)
    old_manifest["raw_quantile_crossing_audit"] = crossing_audit
    manifest_relative = f"{lightgbm_directory}/manifest.json"
    _write_canonical_json(verifier, root, manifest_relative, old_manifest)
    created.add(manifest_relative)
    canonical_entries["LightGBM"] = {
        "model_id": "LightGBM",
        "executor": "lightgbm_bundle",
        "raw_feature_order": feature_order,
        "member_count": 5,
        "artifact": _binding(verifier, root, manifest_relative),
    }
    assert len(created) == 80

    model_entries["external"] = [
        canonical_entries.get(str(entry["model_id"]), entry)
        for entry in model_entries["external"]
    ]
    prediction = (
        "outputs/predictions/"
        f"external_pooled_development_{run_id}.parquet"
    )
    _write_bytes(root, prediction, b"fixture predictions\n")
    prediction_path = root / prediction
    _write_canonical_json(verifier, root, prediction + ".meta.json", {
        "schema_version": "thermoroute.artifact.v1",
        "kind": "external_pooled_development_predictions",
        "artifact": prediction_path.name,
        "artifact_sha256": verifier.sha256_file(prediction_path),
        "artifact_bytes": prediction_path.stat().st_size,
        "content_schema": "thermoroute.predictions.v1",
        "run": identity,
        "parents": {},
        "extra": {
            "common_test_keys": 1,
            "post_2020_data_read": False,
        },
        "created_utc": "2026-07-22T00:00:00+00:00",
    })
    prediction_binding = {
        **_binding(verifier, root, prediction),
        "sidecar": _binding(verifier, root, prediction + ".meta.json"),
    }
    pointer = "outputs/models/route_a_external_components.json"
    component_entries = [
        canonical_entries[model_id]
        for model_id in ("ThermoRoute", "LSTM", "LightGBM")
    ]
    _write_canonical_json(verifier, root, pointer, {
        "format": "thermoroute.route-a-model-components.v1",
        "status": "COMPLETE",
        "training_device": "cpu",
        "run_id": run_id,
        "cohort": "external",
        "raw_feature_order": feature_order,
        "models": component_entries,
        "development_contract": development_contract,
        "development_prediction_artifact": prediction_binding,
    })
    run_manifest = f"outputs/runs/25_external_pooled/{run_id}/run.json"
    _write_canonical_json(verifier, root, run_manifest, {
        "schema_version": "thermoroute.run.v2",
        "identity": identity,
        "resolved_config": configuration,
        "created_utc": "2026-07-22T00:00:00+00:00",
        "environment": {},
        "git": {},
        "provenance": {
            "outcome_status": "NO_POST_2020_DATA_READ",
            "training_device": "cpu",
        },
    })
    model_files = [_binding(verifier, root, relative) for relative in sorted(created)]
    artifacts = {
        "run_manifest": _binding(verifier, root, run_manifest),
        "components_pointer": _binding(verifier, root, pointer),
        "development_predictions": _binding(verifier, root, prediction),
        "development_prediction_sidecar": _binding(
            verifier, root, prediction + ".meta.json"
        ),
        "frozen_panel_spec": development_contract["frozen_panel_spec"],
        "development_panel": development_contract["panel"],
        "station_registry": development_contract["registry"],
        "development_predictor_bridge": development_contract[
            "predictor_bridge"
        ],
        "model_files": model_files,
    }
    receipt = {
        "format": "thermoroute.stage25-completion-receipt.v1",
        "status": "COMPLETE",
        "stage": "25_train_external_pooled_suite",
        "run_id": run_id,
        "run_identity": identity,
        "formal_configuration": configuration,
        "training_device": "cpu",
        "confirmation_outcomes_requested_or_read": False,
        "artifacts": artifacts,
        "artifact_closure_sha256": verifier._sha256_json(artifacts),
    }
    receipt["receipt_self_sha256"] = verifier._sha256_json(receipt)
    receipt_path = "outputs/models/route_a_stage25_completion.json"
    _write_canonical_json(verifier, root, receipt_path, receipt)
    created.update({
        prediction,
        prediction + ".meta.json",
        pointer,
        run_manifest,
        receipt_path,
    })
    return _binding(verifier, root, receipt_path), created


def _write_stage16_gate_fixture(
    verifier,
    root: Path,
    *,
    model_entries: dict[str, list[dict[str, object]]],
    development_contract: dict[str, object],
    stage9_receipt: dict[str, object],
    stage9_path: str,
    source_sha256: str,
    runtime_sha256: str,
) -> tuple[dict[str, str], set[str]]:
    """Create the canonical Stage-16 receipt and its exact 24-file closure."""
    feature_order = [
        "WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP"
    ]
    parent = "outputs/predictions/usgs_predictions_stage9_v2.parquet"
    parent_sidecar = parent + ".meta.json"
    old_lstm_entry = next(
        entry for entry in model_entries["temporal"]
        if entry["model_id"] == "LSTM"
    )
    old_metadata_path = root / str(old_lstm_entry["artifact"]["path"]) / "metadata.json"
    old_metadata = json.loads(old_metadata_path.read_text(encoding="utf-8"))
    source_prediction = root / old_metadata["development_prediction"]["artifact"]["path"]
    parent_path = root / parent
    parent_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_prediction, parent_path)
    _write_canonical_json(verifier, root, parent_sidecar, {
        "artifact_sha256": verifier.sha256_file(parent_path),
    })
    stage9_receipt["artifacts"]["predictions"] = _binding(
        verifier, root, parent
    )
    stage9_receipt["artifacts"]["prediction_sidecar"] = _binding(
        verifier, root, parent_sidecar
    )
    stage9_receipt["receipt_self_sha256"] = verifier._sha256_json({
        key: value for key, value in stage9_receipt.items()
        if key != "receipt_self_sha256"
    })
    _write_canonical_json(verifier, root, stage9_path, stage9_receipt)

    input_closure_sha256 = verifier._compose_input_closure_digest({
        "development": stage9_receipt["run_identity"][
            "input_closure_sha256"
        ],
        "stage09_parent_prediction": verifier.sha256_file(parent_path),
        "stage09_parent_sidecar": verifier.sha256_file(root / parent_sidecar),
        "stage09_completion_receipt": verifier.sha256_file(root / stage9_path),
        "stage09_components": stage9_receipt["artifacts"][
            "components_pointer"
        ]["sha256"],
    })

    configuration = {
        "stage": "16_lstm_baseline_insample",
        "role": "final_route_a_development_predictions",
        "parent_sha256": verifier.sha256_file(parent_path),
        "models": [
            "Persistence", "DampedPersistence", "Climatology",
            "LightGBM", "LSTM", "ThermoRoute",
        ],
        "seeds": [0, 1, 2, 3, 4],
        "variables": feature_order,
        "horizons": [1, 3, 7],
        "context_length": 32,
        "station_embedding": True,
        "station_balanced": True,
        "selection_metric": "station_macro",
        "validation_grid": verifier._stage16_validation_grid(),
        "validation_selection_seed": 0,
        "validation_selection_split": "2016-2017 only",
        "event_reference_fit_interval": ["2006-01-01", "2018-12-31"],
        "train_config": verifier._stage16_train_config(),
        "training_device": "cpu",
        "formal_numerical_policy": {"worker_threads": 1},
        "input_closure_sha256": input_closure_sha256,
        "input_closure_file_count": (
            stage9_receipt["formal_configuration"][
                "input_closure_file_count"
            ] + 4
        ),
    }
    identity = _fixture_run_identity(
        verifier,
        configuration,
        panel_sha256=development_contract["panel"]["sha256"],
        registry_sha256=development_contract["registry"]["sha256"],
        source_sha256=source_sha256,
        runtime_sha256=runtime_sha256,
        input_closure_sha256=input_closure_sha256,
    )
    run_id = identity["run_id"]
    run_dir = f"outputs/runs/16_lstm_baseline/{run_id}"
    run_manifest = f"{run_dir}/run.json"
    _write_canonical_json(verifier, root, run_manifest, {
        "schema_version": "thermoroute.run.v2",
        "identity": identity,
        "resolved_config": configuration,
        "created_utc": "2026-07-25T00:00:00+00:00",
        "environment": {},
        "git": {},
        "provenance": {
            "evidence_role": "prelabel_route_a_model_build_development_only",
            "training_device": "cpu",
        },
    })

    selection = "outputs/tables/lstm_validation_selection.csv"
    # Candidate 1 wins, but candidate 0 sits just outside the frozen 1e-5
    # band so adversarial tests can move another metric view inside the band.
    metrics = [0.200011, 0.20, 0.40]
    header = (
        "candidate_id,d,layers,dropout,station_embed_dim,"
        "use_derived_context,anchor,val_station_macro_rmse,selected,"
        "selection_split\n"
    )
    rows = []
    for candidate_id, candidate in enumerate(verifier._stage16_validation_grid()):
        rows.append(
            f"{candidate_id},{candidate['d']},{candidate['layers']},"
            f"{candidate['dropout']},{candidate['station_embed_dim']},"
            f"{candidate['use_derived_context']},{candidate['anchor']},"
            f"{metrics[candidate_id]},{candidate_id == 1},"
            "2016-2017 validation\n"
        )
    _write_bytes(root, selection, (header + "".join(rows)).encode("utf-8"))

    created: set[str] = {parent, parent_sidecar, run_manifest, selection}

    def sidecar_document(
        artifact_relative: str,
        *,
        kind: str,
        extra: dict[str, object],
        parents: dict[str, str] | None = None,
    ) -> dict[str, object]:
        artifact = root / artifact_relative
        return {
            "schema_version": "thermoroute.artifact.v1",
            "kind": kind,
            "artifact": artifact.name,
            "artifact_sha256": verifier.sha256_file(artifact),
            "artifact_bytes": artifact.stat().st_size,
            "content_schema": "thermoroute.predictions.v1",
            "run": identity,
            "parents": parents or {},
            "extra": extra,
            "created_utc": "2026-07-25T00:00:00+00:00",
        }

    seed_files: list[dict[str, str]] = []
    for seed in range(5):
        prediction = f"{run_dir}/predictions/seed{seed}.parquet"
        sidecar = prediction + ".meta.json"
        shutil.copy2(source_prediction, _write_bytes(root, prediction, b""))
        _write_canonical_json(
            verifier,
            root,
            sidecar,
            sidecar_document(
                prediction, kind="lstm_seed_predictions", extra={}
            ),
        )
        seed_files.extend((
            _binding(verifier, root, prediction),
            _binding(verifier, root, sidecar),
        ))
        created.update({prediction, sidecar})

    candidate_files: list[dict[str, str]] = []
    for candidate_id, candidate in enumerate(verifier._stage16_validation_grid()):
        prediction = f"{run_dir}/selection/candidate{candidate_id}.parquet"
        prediction_sidecar = prediction + ".meta.json"
        checkpoint = f"{run_dir}/selection/candidate{candidate_id}.pt"
        checkpoint_sidecar = checkpoint + ".meta.json"
        shutil.copy2(source_prediction, _write_bytes(root, prediction, b""))
        _write_canonical_json(
            verifier,
            root,
            prediction_sidecar,
            sidecar_document(
                prediction,
                kind="lstm_validation_candidate_predictions",
                extra={
                    "candidate_id": candidate_id,
                    "candidate": candidate,
                    "selection_split": "2016-2017 validation",
                },
            ),
        )
        checkpoint_config = {
            **configuration,
            "candidate_id": candidate_id,
            "candidate": candidate,
        }
        checkpoint_path = _write_bytes(root, checkpoint, b"")
        checkpoint_config_json = json.dumps(
            checkpoint_config,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        checkpoint_extra = {
            "bad_epochs": 0,
            "train_rng_state": np.random.default_rng(candidate_id).bit_generator.state,
        }
        checkpoint_extra_json = json.dumps(
            checkpoint_extra,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        torch.save({
            "format": "thermoroute.training-checkpoint.v3",
            "run_id": run_id,
            "resolved_config_json": checkpoint_config_json,
            "resolved_config_sha256": verifier._sha256_json(checkpoint_config),
            "extra_json": checkpoint_extra_json,
            "extra_sha256": verifier._sha256_json(checkpoint_extra),
            "epoch": 79,
            "best_epoch": 1,
            "best_metric": float(metrics[candidate_id]),
            "model_class": "thermoroute.train.LSTMForecaster",
            "optimizer_class": "torch.optim.adamw.AdamW",
            "scheduler_class": (
                "torch.optim.lr_scheduler.ReduceLROnPlateau"
            ),
            "model_state": {"fixture": torch.tensor([candidate_id])},
            "best_model_state": {"fixture": torch.tensor([candidate_id])},
            "optimizer_state": {"state": {}, "param_groups": []},
            "scheduler_present": True,
            "scheduler_state": {"best": float(metrics[candidate_id])},
            "rng_state": {},
        }, checkpoint_path)
        _write_canonical_json(verifier, root, checkpoint_sidecar, {
            "format": "thermoroute.training-checkpoint-metadata.v2",
            "checkpoint_format": "thermoroute.training-checkpoint.v3",
            "run_id": run_id,
            "epoch": 79,
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "checkpoint_sha256": verifier.sha256_file(checkpoint_path),
            "resolved_config_sha256": verifier._sha256_json(checkpoint_config),
            "extra_sha256": verifier._sha256_json(checkpoint_extra),
            "model_class": "thermoroute.train.LSTMForecaster",
            "optimizer_class": "torch.optim.adamw.AdamW",
            "scheduler_class": "torch.optim.lr_scheduler.ReduceLROnPlateau",
            "scheduler_present": True,
        })
        candidate_files.extend(
            _binding(verifier, root, relative)
            for relative in (
                prediction, prediction_sidecar, checkpoint, checkpoint_sidecar
            )
        )
        created.update({
            prediction, prediction_sidecar, checkpoint, checkpoint_sidecar
        })

    final_prediction = "outputs/predictions/usgs_predictions_v2.parquet"
    shutil.copy2(source_prediction, _write_bytes(root, final_prediction, b""))
    final_sidecar = final_prediction + ".meta.json"
    final_extra = {
        "parent_run_id": stage9_receipt["run_id"],
        "primary_models": configuration["models"],
        "primary_common_test_keys": 6,
        "dropped_primary_rows": 0,
        "lstm_validation_rows": 6,
        "lstm_calibration_rows": 300,
    }
    _write_canonical_json(
        verifier,
        root,
        final_sidecar,
        sidecar_document(
            final_prediction,
            kind="final_route_a_development_predictions",
            parents={PurePosixPath(parent).name: verifier.sha256_file(parent_path)},
            extra=final_extra,
        ),
    )
    created.update({final_prediction, final_sidecar})

    bundle_dir = f"outputs/models/lstm_usgs_bundle_{run_id}"
    metadata_path = f"{bundle_dir}/metadata.json"
    weights_path = f"{bundle_dir}/weights.pt"
    old_weights = root / str(old_lstm_entry["artifact"]["path"]) / "weights.pt"
    shutil.copy2(old_weights, _write_bytes(root, weights_path, b""))
    development_prediction = dict(old_metadata["development_prediction"])
    development_prediction["artifact"] = {
        **_binding(verifier, root, final_prediction),
        "sidecar": _binding(verifier, root, final_sidecar),
    }
    metadata = {
        **old_metadata,
        "run_id": run_id,
        "source_sha256": source_sha256,
        "panel_sha256": identity["panel_sha256"],
        "registry_sha256": identity["registry_sha256"],
        "config_sha256": identity["config_sha256"],
        "input_closure_sha256": identity["input_closure_sha256"],
        "members": [f"seed{seed}" for seed in range(5)],
        "development_prediction": development_prediction,
    }
    _write_canonical_json(verifier, root, metadata_path, metadata)
    created.update({metadata_path, weights_path})
    entry = {
        "model_id": "LSTM",
        "executor": "lstm_bundle",
        "raw_feature_order": feature_order,
        "member_count": 5,
        "artifact": {
            "path": bundle_dir,
            "metadata_sha256": verifier.sha256_file(root / metadata_path),
            "weights_sha256": verifier.sha256_file(root / weights_path),
        },
    }
    old_lstm_entry.clear()
    old_lstm_entry.update(entry)
    pointer = "outputs/models/route_a_lstm_components.json"
    prediction_binding = {
        **_binding(verifier, root, final_prediction),
        "sidecar": _binding(verifier, root, final_sidecar),
    }
    _write_canonical_json(verifier, root, pointer, {
        "format": "thermoroute.route-a-model-components.v1",
        "status": "COMPLETE",
        "training_device": "cpu",
        "run_id": run_id,
        "cohort": "temporal_lstm",
        "raw_feature_order": feature_order,
        "models": [entry],
        "development_contract": development_contract,
        "development_prediction_artifact": prediction_binding,
    })
    shortcut = "outputs/models/lstm_usgs_bundle.json"
    _write_canonical_json(verifier, root, shortcut, {
        "run_id": run_id,
        "bundle_path": bundle_dir,
        "member_count": 5,
        "metadata_sha256": entry["artifact"]["metadata_sha256"],
        "weights_sha256": entry["artifact"]["weights_sha256"],
    })
    created.update({pointer, shortcut})

    model_files = [
        _binding(verifier, root, metadata_path),
        _binding(verifier, root, weights_path),
    ]
    artifacts = {
        "run_manifest": _binding(verifier, root, run_manifest),
        "stage09_completion_receipt": _binding(verifier, root, stage9_path),
        "stage09_parent_predictions": _binding(verifier, root, parent),
        "stage09_parent_prediction_sidecar": _binding(
            verifier, root, parent_sidecar
        ),
        "lstm_validation_selection": _binding(verifier, root, selection),
        "development_predictions": _binding(
            verifier, root, final_prediction
        ),
        "development_prediction_sidecar": _binding(
            verifier, root, final_sidecar
        ),
        "model_files": model_files,
        "lstm_seed_prediction_files": seed_files,
        "selection_candidate_files": candidate_files,
        "shortcut_pointer": _binding(verifier, root, shortcut),
        "components_pointer": _binding(verifier, root, pointer),
    }
    threshold_registry = {"01073319": 10.0}
    threshold_contract = {
        "target": "WTEMP",
        "fit_split": "canonical development train mask",
        "scope": "station-specific",
        "estimator": "pandas Series.quantile(q=0.90, interpolation=linear)",
        "quantile": 0.90,
        "registry": threshold_registry,
        "registry_sha256": verifier._sha256_json(threshold_registry),
    }
    selection_inputs = {
        "run_manifest": artifacts["run_manifest"],
        "panel": development_contract["panel"],
        "frozen_panel_spec": development_contract["frozen_panel_spec"],
        "station_registry": development_contract["registry"],
        "selection": artifacts["lstm_validation_selection"],
        "candidate_files": candidate_files,
        "event_threshold_contract": threshold_contract,
    }
    selection_audit = {
        "format": "thermoroute.stage16-selection-audit.v1",
        "status": "PASS_BEST_STATE_REPLAY_AND_VALIDATION_SELECTION_PARITY",
        "metric": "mean_station_rmse_across_all_horizons",
        "selection_split": "2016-2017 validation",
        "metric_atol": 1e-5,
        "replay_atol": 1e-5,
        "winner_candidate_id": 1,
        "candidates": [
            {
                "candidate_id": candidate_id,
                "recomputed_val_station_macro_rmse": metric,
                "reported_val_station_macro_rmse": metric,
                "checkpoint_best_metric": metric,
                "best_state_max_abs_difference": 0.0,
                "selected": candidate_id == 1,
            }
            for candidate_id, metric in enumerate(metrics)
        ],
        "input_closure": selection_inputs,
        "input_closure_sha256": verifier._sha256_json(selection_inputs),
    }
    parity_inputs = {
        "panel": development_contract["panel"],
        "frozen_panel_spec": development_contract["frozen_panel_spec"],
        "station_registry": development_contract["registry"],
        "bundle_metadata": model_files[0],
        "bundle_weights": model_files[1],
        "development_prediction": development_prediction,
    }
    parity = {
        "format": "thermoroute.stage16-bundle-parity.v1",
        "status": "PASS_FIVE_MEMBER_VAL_CALIB_TEST_REPLAY",
        "members": [f"seed{seed}" for seed in range(5)],
        "splits": ["val", "calib", "test"],
        "atol": 1e-5,
        "max_abs_difference": 0.0,
        "input_closure": parity_inputs,
        "input_closure_sha256": verifier._sha256_json(parity_inputs),
    }
    receipt = {
        "format": "thermoroute.stage16-completion-receipt.v1",
        "status": "PASS_FORMAL_STAGE16_COMPLETE",
        "stage": "16_lstm_baseline_insample",
        "run_id": run_id,
        "parent_stage09_run_id": stage9_receipt["run_id"],
        "run_identity": identity,
        "formal_configuration": configuration,
        "training_device": "cpu",
        "confirmation_outcomes_requested_or_read": False,
        "selection_audit": selection_audit,
        "bundle_prediction_parity": parity,
        "artifacts": artifacts,
        "artifact_closure_sha256": verifier._sha256_json(artifacts),
    }
    receipt["receipt_self_sha256"] = verifier._sha256_json(receipt)
    receipt_path = "outputs/models/route_a_stage16_completion.json"
    _write_canonical_json(verifier, root, receipt_path, receipt)
    created.add(receipt_path)
    return _binding(verifier, root, receipt_path), created


def _bind_stage16_receipt_fixture(
    verifier,
    root: Path,
    suite: dict[str, object],
    receipt: dict[str, object],
) -> dict[str, object]:
    """Re-seal a deliberately mutated Stage-16 fixture and its suite binding."""
    receipt.pop("receipt_self_sha256", None)
    receipt["artifact_closure_sha256"] = verifier._sha256_json(
        receipt["artifacts"]
    )
    receipt["receipt_self_sha256"] = verifier._sha256_json(receipt)
    relative = str(
        suite["preopening_gates"]["stage16_lstm_completion"]["path"]
    )
    _write_canonical_json(verifier, root, relative, receipt)
    rebound = json.loads(json.dumps(suite))
    rebound["preopening_gates"]["stage16_lstm_completion"] = _binding(
        verifier, root, relative
    )
    return rebound


def _mutate_stage16_checkpoint_fixture(
    verifier,
    root: Path,
    suite: dict[str, object],
    *,
    candidate_id: int,
    mode: str,
) -> dict[str, object]:
    """Mutate one checkpoint while consistently re-binding all evidence bytes."""
    gate = suite["preopening_gates"]["stage16_lstm_completion"]
    receipt_path = root / str(gate["path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    candidate_files = receipt["artifacts"]["selection_candidate_files"]
    offset = 4 * candidate_id
    checkpoint_relative = str(candidate_files[offset + 2]["path"])
    sidecar_relative = str(candidate_files[offset + 3]["path"])
    checkpoint_path = root / checkpoint_relative
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if mode == "missing_scheduler_state":
        payload.pop("scheduler_state")
    elif mode == "nonterminal":
        payload["epoch"] = 1
        payload["best_epoch"] = 1
        extra = json.loads(payload["extra_json"])
        extra["bad_epochs"] = 0
        payload["extra_json"] = json.dumps(
            extra,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        payload["extra_sha256"] = verifier._sha256_json(extra)
    else:  # pragma: no cover - fixture misuse
        raise AssertionError(mode)
    torch.save(payload, checkpoint_path)
    sidecar_path = root / sidecar_relative
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["epoch"] = payload["epoch"]
    sidecar["checkpoint_bytes"] = checkpoint_path.stat().st_size
    sidecar["checkpoint_sha256"] = verifier.sha256_file(checkpoint_path)
    sidecar["extra_sha256"] = payload["extra_sha256"]
    _write_canonical_json(verifier, root, sidecar_relative, sidecar)
    candidate_files[offset + 2] = _binding(
        verifier, root, checkpoint_relative
    )
    candidate_files[offset + 3] = _binding(verifier, root, sidecar_relative)
    selection_audit = receipt["selection_audit"]
    selection_audit["input_closure"]["candidate_files"] = candidate_files
    selection_audit["input_closure_sha256"] = verifier._sha256_json(
        selection_audit["input_closure"]
    )
    return _bind_stage16_receipt_fixture(verifier, root, suite, receipt)


def _commit_all_fixture_files(root: Path, message: str) -> str:
    """Commit fixture outputs even when the repository template ignores them."""
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-f", "-A"], cwd=root, check=True)
    return _commit_git_fixture(root, message)


def _development_model_metadata(
    root: Path, suite: dict[str, object]
) -> dict[tuple[str, str], dict[str, object]]:
    metadata: dict[tuple[str, str], dict[str, object]] = {}
    for cohort in ("temporal", "external"):
        for entry in suite["cohorts"][cohort]["models"]:
            if entry["executor"] == "builtin":
                continue
            artifact = entry["artifact"]
            relative = str(artifact["path"])
            path = root / relative
            if entry["executor"] != "lightgbm_bundle":
                path = path / "metadata.json"
            metadata[(cohort, str(entry["model_id"]))] = json.loads(
                path.read_text(encoding="utf-8")
            )
    return metadata


def _development_replay_fixture(
    verifier,
    root: Path,
    *,
    suite: dict[str, object],
    source_sha256: str,
    runtime_sha256: str,
    python_sha256: str = "d" * 64,
) -> dict[str, object]:
    suite_path = "data_usgs/confirmatory_model_suite_v1.json"
    receipt_path = verifier.DEVELOPMENT_REPLAY_RECEIPT
    entrypoint = verifier.DEVELOPMENT_REPLAY_ENTRYPOINT
    metadata = _development_model_metadata(root, suite)
    panel_contracts = verifier._independent_development_panel_contracts(
        (root / "data_usgs/panel_usgs_120v2.parquet").read_bytes(),
        (root / "data_usgs/station_registry_v1.csv").read_bytes(),
        (root / "data_usgs/frozen_panel_v1.json").read_bytes(),
    )
    rows = []
    for cohort in ("temporal", "external"):
        entries = {
            str(entry["model_id"]): entry
            for entry in suite["cohorts"][cohort]["models"]
        }
        for model in verifier.DEVELOPMENT_REPLAY_LEARNED_MODELS[cohort]:
            entry = entries[model]
            model_metadata = metadata[(cohort, model)]
            prediction = model_metadata["development_prediction"]
            prediction_path = root / prediction["artifact"]["path"]
            calibrated_head_gate = (
                verifier._independent_development_calibration_replay(
                    prediction_path.read_bytes(),
                    model_metadata,
                    model=model,
                    seeds=tuple(range(entry["member_count"])),
                    external=cohort == "external",
                    panel_contract=panel_contracts[cohort],
                    label=f"fixture {cohort}/{model}",
                )
            )
            rows.append({
                "cohort": cohort,
                "model": model,
                "executor": entry["executor"],
                "members": entry["member_count"],
                "rows": prediction["rows"],
                "atol": prediction["atol"],
                "max_abs_difference": 0.0,
                "calibrated_head_gate": calibrated_head_gate,
                "status": "PASS",
            })
    read_paths = sorted([entrypoint, suite_path])
    document = {
        "format": verifier.DEVELOPMENT_REPLAY_FORMAT,
        "status": "PASS_FULL_DEVELOPMENT_REPLAY_NO_CONFIRMATION_DATA",
        "isolated_process_required": True,
        "suite": _binding(verifier, root, suite_path),
        "source_tree_sha256": source_sha256,
        "runtime_sha256": runtime_sha256,
        "suite_numerical_runtime_sha256": runtime_sha256,
        "development_contract_sha256": verifier._sha256_json(
            suite["development_contract"]
        ),
        "replayed_splits": ["val", "calib", "test_2019_2020_development"],
        "confirmation_period_read": False,
        "builtins_validated_by_suite_contract": [
            "Climatology", "DampedPersistence", "Persistence"
        ],
        "models": rows,
        "execution_attestation": {
            "format": verifier.DEVELOPMENT_REPLAY_EXECUTION_FORMAT,
            "entrypoint": _binding(verifier, root, entrypoint),
            "interpreter": {
                "invoked_path": "/fixture/python",
                "realpath": "/fixture/python-real",
                "sha256": python_sha256,
                "implementation": "CPython",
                "version": "3.12.0",
            },
            "isolated_mode": True,
            "required_python_flags": {
                "isolated": 1,
                "ignore_environment": 1,
                "no_user_site": 1,
                "safe_path": True,
                "dont_write_bytecode": 0,
            },
            "fresh_pycache_policy": {
                "required": True,
                "controller_created_initially_empty_prefix": True,
                "repository_local_cache_allowed": False,
                "preexisting_repository_pyc_eligible": False,
                "prefix_lifetime": "one_isolated_child",
            },
            "logical_command_contract": {
                "create": [
                    "<bound-python>", "-I", "-X",
                    "pycache_prefix=<fresh-temporary-directory>", entrypoint,
                    "--_isolated-worker", "--suite", suite_path,
                    "--receipt", receipt_path,
                ],
                "fresh_check": [
                    "<bound-python>", "-I", "-X",
                    "pycache_prefix=<different-fresh-temporary-directory>",
                    entrypoint, "--_isolated-worker", "--suite", suite_path,
                    "--receipt", receipt_path, "--check",
                ],
            },
            "formal_environment": {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "PYTHONHASHSEED": "0",
            },
            "python_hash_seed_interpreter_effect": (
                "environment declaration present but ignored by CPython -I; replay "
                "code must sort identity-bearing collections"
            ),
            "io_guard": {
                "format": verifier.DEVELOPMENT_REPLAY_IO_GUARD_FORMAT,
                "network_access_allowed": False,
                "subprocess_allowed": False,
                "repository_writes_allowed": False,
                "confirmation_read_policy": (
                    verifier._development_replay_confirmation_read_policy()
                ),
                "repo_read_path_count": len(read_paths),
                "repo_read_paths_sha256": verifier._sha256_json(read_paths),
                "repo_read_paths": read_paths,
                "violations": [],
            },
            "security_boundary": (
                "fresh-process honest-owner replay guard; not protection against "
                "replacement of CPython, the operating system, or the repository owner"
            ),
        },
    }
    document["receipt_self_sha256"] = verifier._sha256_json(document)
    return document


@pytest.mark.parametrize(
    "alias",
    (
        "./artifacts/value.bin",
        "artifacts//value.bin",
        "artifacts/subdir/../value.bin",
    ),
)
def test_release_binding_reader_rejects_noncanonical_paths(tmp_path, alias):
    verifier = _load_script(VERIFY_SCRIPT, "verify_release_noncanonical_alias")
    artifact = _write_bytes(tmp_path, "artifacts/value.bin", b"bound bytes\n")
    binding = {"path": alias, "sha256": verifier.sha256_file(artifact)}

    with pytest.raises(ValueError, match="not canonical"):
        verifier._add_binding(
            tmp_path, {}, "fixture", binding, label="fixture binding"
        )


def test_release_binding_reader_rejects_symlink_alias(tmp_path):
    verifier = _load_script(VERIFY_SCRIPT, "verify_release_symlink_alias")
    artifact = _write_bytes(tmp_path, "artifacts/value.bin", b"bound bytes\n")
    alias = tmp_path / "artifacts" / "alias.bin"
    alias.symlink_to(artifact.name)
    binding = {
        "path": "artifacts/alias.bin",
        "sha256": verifier.sha256_file(artifact),
    }

    with pytest.raises(ValueError, match="symlink"):
        verifier._add_binding(
            tmp_path, {}, "fixture", binding, label="fixture binding"
        )


def test_release_binding_reader_rejects_hardlinked_artifact(tmp_path):
    verifier = _load_script(VERIFY_SCRIPT, "verify_release_hardlink")
    artifact = _write_bytes(tmp_path, "artifacts/value.bin", b"bound bytes\n")
    os.link(artifact, tmp_path / "artifacts" / "duplicate.bin")
    binding = {
        "path": "artifacts/value.bin",
        "sha256": verifier.sha256_file(artifact),
    }

    with pytest.raises(ValueError, match="hard-linked"):
        verifier._add_binding(
            tmp_path, {}, "fixture", binding, label="fixture binding"
        )


@pytest.mark.parametrize(
    "attack",
    [
        "legacy_index_v1",
        "metadata_bytes",
        "metadata_schema_v1",
        "outcome_url",
        "redirected_final_url",
        "fabricated_semantics",
        "extra_blob",
        "unindexed_symlink",
        "unindexed_submodule",
    ],
)
def test_release_candidate_snapshot_dependencies_require_metadata_bound_v2(
    tmp_path, attack,
):
    verifier = _load_script(
        VERIFY_SCRIPT, f"thermoroute_candidate_index_{attack}_test"
    )
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.test"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Fixture"], cwd=repository, check=True
    )
    raw = repository / "raw"
    request_url = (
        "https://waterservices.usgs.gov/nwis/site/?agencyCd=USGS&format=rdb&"
        "hasDataTypeCd=dv&parameterCd=00010&siteOutput=expanded&siteStatus=all&"
        "siteType=ST&stateCd=CO"
    )
    if attack == "outcome_url":
        request_url = (
            "https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites=01234567&"
            "parameterCd=00010"
        )
    request = {
        "schema_version": 1,
        "provider": "usgs-nwis-confirmatory-site-metadata",
        "method": "GET",
        "url": request_url,
        "headers": {
            "User-Agent": "ThermoRoute/1.0 Route-A metadata-only discovery"
        },
    }
    request_sha = hashlib.sha256(
        verifier._canonical_json_bytes(request)
    ).hexdigest()
    transaction = raw / request["provider"] / request_sha
    transaction.mkdir(parents=True)
    metadata_path = transaction / "metadata.json"
    response_path = transaction / "response.bin"
    response_path.write_bytes(b"response\n")
    response_sha = verifier.sha256_file(response_path)
    metadata = {
        "schema_version": 1 if attack == "metadata_schema_v1" else 2,
        "request": request,
        "request_sha256": request_sha,
        "retrieved_at_utc": "2026-07-26T00:00:00+00:00",
        "http_status": 200,
        "response_headers": {},
        "byte_count": response_path.stat().st_size,
        "response_sha256": response_sha,
        "response_file": "response.bin",
        "final_url": (
            "https://redirected.example.invalid/final"
            if attack == "redirected_final_url"
            else request_url
        ),
        "retrieval_semantics": (
            "FABRICATED"
            if attack == "fabricated_semantics"
            else "DIRECT_HTTP_RESPONSE"
        ),
    }
    original_metadata = verifier._canonical_json_bytes(metadata)
    metadata_path.write_bytes(original_metadata)
    record = {
        "provider": "usgs-nwis-confirmatory-site-metadata",
        "request_sha256": request_sha,
        "response_sha256": response_sha,
        "metadata_sha256": hashlib.sha256(original_metadata).hexdigest(),
        "metadata_byte_count": len(original_metadata),
        "retrieved_at_utc": "2026-07-26T00:00:00+00:00",
        "byte_count": response_path.stat().st_size,
        "request": request,
        "metadata_path": metadata_path.relative_to(raw).as_posix(),
        "response_path": response_path.relative_to(raw).as_posix(),
    }
    index = {
        "schema_version": 1 if attack == "legacy_index_v1" else 2,
        "snapshot_count": 1,
        "records": [record],
    }
    (raw / "snapshot_index.json").write_bytes(
        verifier._canonical_json_bytes(index)
    )
    if attack == "metadata_bytes":
        metadata_path.write_bytes(
            verifier._canonical_json_bytes({"schema_version": 2, "forged": True})
        )
    elif attack == "extra_blob":
        (raw / "unindexed.bin").write_bytes(b"extra candidate raw bytes\n")
    elif attack == "unindexed_symlink":
        (raw / "unindexed-link").symlink_to("snapshot_index.json")
    elif attack == "unindexed_submodule":
        submodule = raw / "unindexed-submodule"
        submodule.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=submodule, check=True)
        subprocess.run(
            ["git", "config", "user.email", "fixture@example.test"],
            cwd=submodule,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Fixture"],
            cwd=submodule,
            check=True,
        )
        (submodule / "payload.txt").write_text("submodule\n", encoding="utf-8")
        subprocess.run(["git", "add", "payload.txt"], cwd=submodule, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "submodule fixture"],
            cwd=submodule,
            check=True,
        )
    subprocess.run(["git", "add", "raw"], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "fixture"], cwd=repository, check=True
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    with pytest.raises(
        ValueError,
        match=r"(?i)(metadata|schema|malformed|SHA-256|namespace|non-file)",
    ):
        verifier._snapshot_dependency_paths(
            repository,
            commit,
            "raw/snapshot_index.json",
            require_metadata_binding=True,
            require_candidate_metadata_contract=True,
        )


@pytest.mark.parametrize(
    "attack",
    [
        "legacy_index_v1",
        "redirected_final_url",
        "fabricated_semantics",
        "noncanonical_metadata",
        "wrong_expected_provider",
        "extra_blob",
        "unindexed_symlink",
        "unindexed_submodule",
    ],
)
def test_release_formal_meteorology_requires_semantic_metadata_v2(
    tmp_path: Path,
    attack: str,
) -> None:
    verifier = _load_script(
        VERIFY_SCRIPT, f"thermoroute_formal_meteorology_{attack}_test"
    )
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.test"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Fixture"],
        cwd=repository,
        check=True,
    )
    index_path = "raw/daymet-v1/snapshot_index.json"
    provider = "ornl-daymet-single-pixel-route-a"
    url = "https://daymet.ornl.gov/single-pixel/api/data?lat=40.00000000"
    paths = _write_formal_meteorology_snapshot(
        verifier,
        repository,
        index_path,
        provider=provider,
        url=url,
        payload=b"formal meteorology\n",
    )
    index_file = repository / index_path
    index = json.loads(index_file.read_text(encoding="utf-8"))
    record = index["records"][0]
    metadata_relative = (
        PurePosixPath(index_path).parent / record["metadata_path"]
    ).as_posix()
    metadata_path = repository / metadata_relative
    if attack == "legacy_index_v1":
        index["schema_version"] = 1
        index_file.write_bytes(verifier._canonical_json_bytes(index))
    elif attack in {
        "redirected_final_url", "fabricated_semantics", "noncanonical_metadata"
    }:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if attack == "redirected_final_url":
            metadata["final_url"] = "https://example.invalid/redirected"
            metadata_payload = verifier._canonical_json_bytes(metadata)
        elif attack == "fabricated_semantics":
            metadata["retrieval_semantics"] = "FABRICATED"
            metadata_payload = verifier._canonical_json_bytes(metadata)
        else:
            metadata_payload = verifier._canonical_json_bytes(metadata) + b" "
        metadata_path.write_bytes(metadata_payload)
        record["metadata_sha256"] = hashlib.sha256(metadata_payload).hexdigest()
        record["metadata_byte_count"] = len(metadata_payload)
        index_file.write_bytes(verifier._canonical_json_bytes(index))
    elif attack == "extra_blob":
        _write_bytes(repository, "raw/daymet-v1/unindexed.bin", b"extra\n")
    elif attack == "unindexed_symlink":
        (repository / "raw/daymet-v1/unindexed-link").symlink_to(
            "snapshot_index.json"
        )
    elif attack == "unindexed_submodule":
        submodule = repository / "raw/daymet-v1/unindexed-submodule"
        submodule.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=submodule, check=True)
        subprocess.run(
            ["git", "config", "user.email", "fixture@example.test"],
            cwd=submodule,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Fixture"],
            cwd=submodule,
            check=True,
        )
        _write_bytes(submodule, "payload.txt", b"submodule\n")
        subprocess.run(["git", "add", "payload.txt"], cwd=submodule, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "submodule fixture"],
            cwd=submodule,
            check=True,
        )
    subprocess.run(["git", "add", "raw"], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", f"formal attack {attack}"],
        cwd=repository,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    expected_provider = (
        "gridmet-ncss-route-a"
        if attack == "wrong_expected_provider"
        else provider
    )
    with pytest.raises(
        ValueError,
        match=r"(?i)(formal|snapshot|metadata|provider|namespace|non-file)",
    ):
        verifier._snapshot_dependency_paths(
            repository,
            commit,
            index_path,
            require_metadata_binding=True,
            require_formal_prelabel_metadata_contract=True,
            expected_provider=expected_provider,
        )
    assert paths <= {
        path.relative_to(repository).as_posix()
        for path in repository.rglob("*")
        if path.is_file() and ".git" not in path.parts
    }


def test_release_truth_binding_accepts_stage09_float32_round_trip(tmp_path):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_float32_truth_round_trip"
    )
    model_entries, _ = _write_development_model_fixtures(
        verifier, tmp_path, runtime_sha256="c" * 64
    )
    suite = {
        "cohorts": {
            cohort: {"models": entries}
            for cohort, entries in model_entries.items()
        }
    }
    metadata = _development_model_metadata(tmp_path, suite)[
        ("temporal", "LightGBM")
    ]
    panel_contract = verifier._independent_development_panel_contracts(
        (tmp_path / "data_usgs/panel_usgs_120v2.parquet").read_bytes(),
        (tmp_path / "data_usgs/station_registry_v1.csv").read_bytes(),
        (tmp_path / "data_usgs/frozen_panel_v1.json").read_bytes(),
    )["temporal"]
    prediction_path = tmp_path / metadata["development_prediction"]["artifact"][
        "path"
    ]

    gate = verifier._independent_development_calibration_replay(
        prediction_path.read_bytes(),
        metadata,
        model="LightGBM",
        seeds=tuple(range(5)),
        external=False,
        panel_contract=panel_contract,
        label="float32 truth round trip",
    )

    assert gate["status"] == "PASS_NONNEGATIVE_CQR_WIDENS_ONLY"


@pytest.mark.parametrize(
    "attack",
    (
        "extra_top", "missing_top", "source_binding", "policy_allowlist",
        "underscore_sibling_read", "read_count", "read_hash", "io_extra_key",
        "execution_flags", "fresh_pycache", "execution_environment",
        "execution_command", "interpreter", "model_missing", "model_duplicate",
        "model_reordered", "wrong_executor", "wrong_members", "failed_status",
        "wrong_ablation_members",
        "difference_over_tolerance", "entrypoint_binding", "suite_binding",
        "suite_missing_cohorts", "inflated_atol", "reduced_rows",
        "coordinated_inflated_atol", "coordinated_reduced_rows",
        "nested_receipt", "resealed_cqr_metadata",
        "resealed_platt_metadata", "resealed_threshold_metadata",
        "resealed_legacy_mapping", "resealed_y_true",
    ),
)
def test_release_verifier_rejects_forged_development_replay_receipt(
    tmp_path, attack,
):
    verifier = _load_script(
        VERIFY_SCRIPT, f"thermoroute_verify_replay_attack_{attack}"
    )
    _write_bytes(
        tmp_path,
        verifier.DEVELOPMENT_REPLAY_ENTRYPOINT,
        b"#!/usr/bin/env python3\n",
    )
    source_sha256 = "a" * 64
    runtime_sha256 = "c" * 64
    model_entries, _ = _write_development_model_fixtures(
        verifier, tmp_path, runtime_sha256=runtime_sha256
    )
    suite = {
        "numerical_runtime_sha256": runtime_sha256,
        "development_contract": {"source_sha256": source_sha256},
        "cohorts": {
            cohort: {"models": entries}
            for cohort, entries in model_entries.items()
        },
    }
    suite_path = "data_usgs/confirmatory_model_suite_v1.json"
    _write_bytes(
        tmp_path,
        suite_path,
        verifier._lineage_canonical_json_bytes(suite),
    )
    replay = _development_replay_fixture(
        verifier,
        tmp_path,
        suite=suite,
        source_sha256=source_sha256,
        runtime_sha256=runtime_sha256,
    )
    model_metadata = _development_model_metadata(tmp_path, suite)
    execution = replay["execution_attestation"]
    io_guard = execution["io_guard"]
    models = replay["models"]
    if attack == "extra_top":
        replay["forged"] = True
    elif attack == "missing_top":
        replay.pop("builtins_validated_by_suite_contract")
    elif attack == "source_binding":
        replay["source_tree_sha256"] = "0" * 64
    elif attack == "policy_allowlist":
        io_guard["confirmation_read_policy"]["allowed_exact_paths"].append(
            "data_usgs/confirmatory_outcomes/labels.parquet"
        )
    elif attack == "underscore_sibling_read":
        paths = sorted([
            *io_guard["repo_read_paths"],
            "data_usgs/confirmatory_model_suite_v1.json_backup",
        ])
        io_guard["repo_read_paths"] = paths
        io_guard["repo_read_path_count"] = len(paths)
        io_guard["repo_read_paths_sha256"] = verifier._sha256_json(paths)
    elif attack == "read_count":
        io_guard["repo_read_path_count"] += 1
    elif attack == "read_hash":
        io_guard["repo_read_paths_sha256"] = "0" * 64
    elif attack == "io_extra_key":
        io_guard["forged"] = False
    elif attack == "execution_flags":
        execution["required_python_flags"]["isolated"] = 0
    elif attack == "fresh_pycache":
        execution["fresh_pycache_policy"][
            "controller_created_initially_empty_prefix"
        ] = False
    elif attack == "execution_environment":
        execution["formal_environment"]["OMP_NUM_THREADS"] = "2"
    elif attack == "execution_command":
        execution["logical_command_contract"]["create"].append("--forged")
    elif attack == "interpreter":
        execution["interpreter"]["realpath"] = "/attacker/python"
    elif attack == "model_missing":
        models.pop()
    elif attack == "model_duplicate":
        models.append(dict(models[-1]))
    elif attack == "model_reordered":
        models[0], models[1] = models[1], models[0]
    elif attack == "wrong_executor":
        models[0]["executor"] = "thermoroute_bundle"
    elif attack == "wrong_members":
        models[0]["members"] = 4
    elif attack == "wrong_ablation_members":
        next(
            row for row in models
            if row["cohort"] == "temporal"
            and row["model"] == "TR-noRouter"
        )["members"] = 4
    elif attack == "failed_status":
        models[0]["status"] = "FAIL"
    elif attack == "difference_over_tolerance":
        models[0]["max_abs_difference"] = 2e-12
    elif attack == "entrypoint_binding":
        execution["entrypoint"]["sha256"] = "0" * 64
    elif attack == "suite_binding":
        replay["suite"]["sha256"] = "0" * 64
    elif attack == "suite_missing_cohorts":
        suite.pop("cohorts")
    elif attack == "inflated_atol":
        models[0]["atol"] = 1.0
        models[0]["max_abs_difference"] = 0.5
    elif attack == "reduced_rows":
        models[0]["rows"] -= 1
    elif attack == "coordinated_inflated_atol":
        prediction = model_metadata[("temporal", "LightGBM")][
            "development_prediction"
        ]
        prediction["atol"] = 1_000_000.0
        prediction["max_abs_difference"] = 999_999.0
        models[0]["atol"] = 1_000_000.0
        models[0]["max_abs_difference"] = 999_999.0
    elif attack == "coordinated_reduced_rows":
        prediction = model_metadata[("temporal", "LightGBM")][
            "development_prediction"
        ]
        prediction["rows"] -= 1
        models[0]["rows"] -= 1
    elif attack == "nested_receipt":
        execution["security_boundary"] = "forged but self-consistent"
    elif attack == "resealed_cqr_metadata":
        # Model-bundle and enclosing release hashes are outside this static
        # function and are assumed to have been refreshed by the attacker.
        # The forged raw/deployed registry and its audit remain internally
        # self-consistent, so only replay from the bound rows can reject it.
        metadata = model_metadata[("temporal", "LightGBM")]
        deployed = metadata["conformal_offsets"]
        forged_raw = {key: -1.0 for key in deployed}
        metadata["conformal_offset_audit"] = verifier._independent_cqr_audit(
            forged_raw, deployed
        )
    elif attack == "resealed_platt_metadata":
        # This is a valid constant-logit parameterization, but it was not fit
        # from the frozen calibration rows and must therefore still fail.
        metadata = model_metadata[("temporal", "LightGBM")]
        constant = 0.10
        metadata["event_calibrators"] = {
            str(horizon): {
                "intercept": math.log(constant / (1.0 - constant)),
                "slope": 0.0,
                "constant": constant,
            }
            for horizon in (1, 3, 7)
        }
    elif attack == "resealed_threshold_metadata":
        # Re-sealing a plausible q90-like value cannot substitute for reading
        # and recomputing the threshold from the canonical panel bytes.
        metadata = model_metadata[("temporal", "LightGBM")]
        thresholds = dict(metadata["event_thresholds"])
        thresholds["01073319"] += 0.25
        metadata["event_thresholds"] = thresholds
    elif attack == "resealed_legacy_mapping":
        # Keep the panel's n90 legacy key and every nested hash self-consistent,
        # but redirect it to another plausible USGS identifier. The verifier
        # must map before computation and detect that predictions target the
        # formerly mapped station.
        registry_payload = (
            b"site_no,legacy_site_id,huc2\n01073320,n90,01\n"
        )
        _write_bytes(
            tmp_path, "data_usgs/station_registry_v1.csv", registry_payload
        )
        registry_sha256 = hashlib.sha256(registry_payload).hexdigest()
        specification_path = tmp_path / "data_usgs/frozen_panel_v1.json"
        specification = json.loads(specification_path.read_text(encoding="utf-8"))
        specification["station_registry"]["sha256"] = registry_sha256
        _write_bytes(
            tmp_path,
            "data_usgs/frozen_panel_v1.json",
            json.dumps(specification, sort_keys=True).encode("utf-8") + b"\n",
        )
        for metadata in model_metadata.values():
            metadata["registry_sha256"] = registry_sha256
    elif attack == "resealed_y_true":
        # Change every member's val truth for one forecast key, then refresh
        # the prediction artifact binding as an enclosing release re-seal
        # would. Calibration objects and the calibrated-head digest are still
        # self-consistent because calibration rows and predicted heads did not
        # change; only a strict join to frozen panel outcomes can reject this.
        metadata = model_metadata[("temporal", "LightGBM")]
        artifact = metadata["development_prediction"]["artifact"]
        prediction_path = tmp_path / artifact["path"]
        prediction_frame = pd.read_parquet(prediction_path)
        forged = (
            prediction_frame["model"].eq("LightGBM")
            & prediction_frame["split"].eq("val")
            & prediction_frame["horizon"].eq(1)
            & prediction_frame["issue_date"].eq(pd.Timestamp("2016-01-01"))
        )
        assert forged.sum() == 5
        canonical_truth = np.float32(
            prediction_frame.loc[forged, "y_true"].iloc[0]
        )
        forged_truth = np.nextafter(
            canonical_truth, np.float32(np.inf), dtype=np.float32
        )
        assert forged_truth != canonical_truth
        prediction_frame.loc[forged, "y_true"] = float(forged_truth)
        prediction_frame.to_parquet(prediction_path, index=False)
        artifact["sha256"] = hashlib.sha256(
            prediction_path.read_bytes()
        ).hexdigest()
    replay.pop("receipt_self_sha256", None)
    replay["receipt_self_sha256"] = verifier._sha256_json(replay)

    with pytest.raises(ValueError, match="development replay"):
        verifier._validate_development_replay_document(
            replay,
            receipt_bytes=verifier._lineage_canonical_json_bytes(replay),
            suite=suite,
            model_metadata=model_metadata,
            prediction_payloads=verifier._filesystem_development_prediction_payloads(
                tmp_path, model_metadata
            ),
            development_panel_payload=(
                tmp_path / "data_usgs/panel_usgs_120v2.parquet"
            ).read_bytes(),
            development_registry_payload=(
                tmp_path / "data_usgs/station_registry_v1.csv"
            ).read_bytes(),
            frozen_panel_spec_payload=(
                tmp_path / "data_usgs/frozen_panel_v1.json"
            ).read_bytes(),
            suite_binding=_binding(verifier, tmp_path, suite_path),
            replay_path=verifier.DEVELOPMENT_REPLAY_RECEIPT,
            source_sha256=source_sha256,
            runtime_sha256=runtime_sha256,
            entrypoint_binding=_binding(
                verifier, tmp_path, verifier.DEVELOPMENT_REPLAY_ENTRYPOINT
            ),
            expected_python_identity={
                "invoked_path": "/fixture/python",
                "realpath": "/fixture/python-real",
                "sha256": "d" * 64,
            },
        )


def test_release_verifier_replay_contract_matches_producer_constants():
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_replay_contract_mirror_test"
    )
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from thermoroute.model_suite import DEVELOPMENT_REPLAY_MODEL_CONTRACTS
    finally:
        sys.path.pop(0)

    assert (
        verifier.DEVELOPMENT_REPLAY_MODEL_CONTRACTS
        == DEVELOPMENT_REPLAY_MODEL_CONTRACTS
    )


def test_development_prediction_payload_loader_reuses_shared_filesystem_bytes(
    tmp_path,
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_replay_payload_fs_cache_test"
    )
    payload = bytes(range(256)) * 8
    relative = "outputs/development/shared_predictions.parquet"
    _write_bytes(tmp_path, relative, payload)
    digest = hashlib.sha256(payload).hexdigest()
    metadata = {
        ("temporal", "LightGBM"): {
            "development_prediction": {
                "artifact": {"path": relative, "sha256": digest}
            }
        },
        ("temporal", "TFT"): {
            "development_prediction": {
                "artifact": {"path": relative, "sha256": digest}
            }
        },
    }

    loaded = verifier._filesystem_development_prediction_payloads(
        tmp_path, metadata
    )

    assert loaded[("temporal", "LightGBM")] == payload
    assert (
        loaded[("temporal", "LightGBM")]
        is loaded[("temporal", "TFT")]
    )

    metadata[("temporal", "TFT")]["development_prediction"]["artifact"][
        "sha256"
    ] = "0" * 64
    with pytest.raises(ValueError, match="conflicting checksums"):
        verifier._filesystem_development_prediction_payloads(tmp_path, metadata)


def test_development_prediction_payload_loader_reuses_shared_git_blob(
    tmp_path, monkeypatch
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_replay_payload_git_cache_test"
    )
    payload = bytes(range(255, -1, -1)) * 8
    relative = "outputs/development/shared_predictions.parquet"
    digest = hashlib.sha256(payload).hexdigest()
    metadata = {
        ("external", "LightGBM"): {
            "development_prediction": {
                "artifact": {"path": relative, "sha256": digest}
            }
        },
        ("external", "TFT"): {
            "development_prediction": {
                "artifact": {"path": relative, "sha256": digest}
            }
        },
    }
    calls: list[tuple[object, ...]] = []

    def git_blob(*args, **_kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout=payload, stderr=b"")

    monkeypatch.setattr(verifier, "_run_git", git_blob)
    loaded = verifier._git_development_prediction_payloads(
        tmp_path / "audit.git", "a" * 40, metadata
    )

    assert len(calls) == 1
    assert loaded[("external", "LightGBM")] == payload
    assert loaded[("external", "LightGBM")] is loaded[("external", "TFT")]

    metadata[("external", "TFT")]["development_prediction"]["artifact"][
        "sha256"
    ] = "0" * 64
    calls.clear()
    with pytest.raises(ValueError, match="conflicting checksums"):
        verifier._git_development_prediction_payloads(
            tmp_path / "audit.git", "a" * 40, metadata
        )
    assert len(calls) == 1


def test_development_replay_filesystem_requires_canonical_producer_json(tmp_path):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_replay_filesystem_canonical_test"
    )
    _write_bytes(
        tmp_path,
        verifier.DEVELOPMENT_REPLAY_ENTRYPOINT,
        b"#!/usr/bin/env python3\n",
    )
    source_sha256 = "a" * 64
    runtime_sha256 = "c" * 64
    model_entries, _ = _write_development_model_fixtures(
        verifier, tmp_path, runtime_sha256=runtime_sha256
    )
    suite = {
        "numerical_runtime_sha256": runtime_sha256,
        "development_contract": {"source_sha256": source_sha256},
        "cohorts": {
            cohort: {"models": entries}
            for cohort, entries in model_entries.items()
        },
    }
    suite_path = "data_usgs/confirmatory_model_suite_v1.json"
    _write_bytes(
        tmp_path, suite_path, verifier._lineage_canonical_json_bytes(suite)
    )
    replay = _development_replay_fixture(
        verifier,
        tmp_path,
        suite=suite,
        source_sha256=source_sha256,
        runtime_sha256=runtime_sha256,
    )
    receipt_path = _write_bytes(
        tmp_path,
        verifier.DEVELOPMENT_REPLAY_RECEIPT,
        json.dumps(replay, indent=2).encode("utf-8") + b"\n",
    )
    with pytest.raises(ValueError, match="canonical producer JSON"):
        verifier._validate_development_replay_document(
            replay,
            receipt_bytes=receipt_path.read_bytes(),
            suite=suite,
            model_metadata=(
                model_metadata := _development_model_metadata(tmp_path, suite)
            ),
            prediction_payloads=verifier._filesystem_development_prediction_payloads(
                tmp_path, model_metadata
            ),
            development_panel_payload=(
                tmp_path / "data_usgs/panel_usgs_120v2.parquet"
            ).read_bytes(),
            development_registry_payload=(
                tmp_path / "data_usgs/station_registry_v1.csv"
            ).read_bytes(),
            frozen_panel_spec_payload=(
                tmp_path / "data_usgs/frozen_panel_v1.json"
            ).read_bytes(),
            suite_binding=_binding(verifier, tmp_path, suite_path),
            replay_path=verifier.DEVELOPMENT_REPLAY_RECEIPT,
            source_sha256=source_sha256,
            runtime_sha256=runtime_sha256,
            entrypoint_binding=_binding(
                verifier, tmp_path, verifier.DEVELOPMENT_REPLAY_ENTRYPOINT
            ),
            expected_python_identity={
                "invoked_path": "/fixture/python",
                "realpath": "/fixture/python-real",
                "sha256": "d" * 64,
            },
        )


def _chronology_binding(verifier, root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    return {
        **_binding(verifier, root, relative),
        "byte_count": path.stat().st_size,
        "git_blob_oid": "a" * 40,
    }


def _fixture_confirmatory_family() -> list[dict[str, object]]:
    specifications = (
        ("H1-h1-vs-damped", "DampedPersistence", 1, 0.0, 1001, 5001),
        ("H1-h3-vs-damped", "DampedPersistence", 3, 0.0, 1003, 5003),
        ("H1-h7-vs-damped", "DampedPersistence", 7, 0.0, 1007, 5007),
        ("H2-h3-vs-lightgbm", "LightGBM", 3, 0.05, 1103, 5103),
        ("H2-h7-vs-lightgbm", "LightGBM", 7, 0.05, 1107, 5107),
    )
    return [
        {
            "test_id": test_id,
            "candidate": "ThermoRoute",
            "reference": reference,
            "horizon": horizon,
            "margin_c": margin,
            "alternative": "candidate_minus_reference_below_margin",
            "bootstrap_seed": bootstrap_seed,
            "sign_flip_seed": sign_flip_seed,
            "description": f"fixture comparison {test_id}",
        }
        for test_id, reference, horizon, margin, bootstrap_seed, sign_flip_seed
        in specifications
    ]


def _fixture_model_metadata_registry(
    root: Path,
    model_entries: dict[str, list[dict[str, object]]],
) -> dict[tuple[str, str], dict[str, object]]:
    output: dict[tuple[str, str], dict[str, object]] = {}
    for cohort, entries in model_entries.items():
        for entry in entries:
            if entry["executor"] == "builtin":
                continue
            model = str(entry["model_id"])
            artifact = entry["artifact"]
            assert isinstance(artifact, dict)
            artifact_path = root / str(artifact["path"])
            metadata_path = (
                artifact_path
                if entry["executor"] == "lightgbm_bundle"
                else artifact_path / "metadata.json"
            )
            output[(cohort, model)] = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
    return output


def _fixture_probability_pipeline_contracts(verifier) -> dict[str, object]:
    output: dict[str, object] = {}
    for pipeline_class, pipeline in verifier.PROBABILISTIC_PIPELINES.items():
        contract = {
            "format": "thermoroute.fixture-member-quantile-contract.v1",
            "pipeline_class": pipeline_class,
        }
        output[pipeline_class] = {
            "pipeline": pipeline,
            "member_quantile_contract": contract,
            "member_quantile_contract_sha256": verifier._sha256_json(contract),
        }
    return output


def _fixture_probability_reliability_bins(
    event: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray,
    sites: np.ndarray,
) -> tuple[list[dict[str, object]], float]:
    clipped = np.clip(probability.astype(float), 1e-6, 1 - 1e-6)
    edges = np.linspace(0.0, 1.0, 11)
    assignments = np.clip(np.digitize(clipped, edges[1:-1]), 0, 9)
    rows: list[dict[str, object]] = []
    ece = 0.0
    for index in range(10):
        selected = assignments == index
        count = int(selected.sum())
        bin_weight = float(weights[selected].sum())
        mean_probability = (
            None
            if not count
            else float(np.average(clipped[selected], weights=weights[selected]))
        )
        event_rate = (
            None
            if not count
            else float(np.average(event[selected], weights=weights[selected]))
        )
        if mean_probability is not None and event_rate is not None:
            ece += bin_weight * abs(event_rate - mean_probability)
        rows.append({
            "bin_index": index + 1,
            "lower_bound": float(edges[index]),
            "upper_bound": float(edges[index + 1]),
            "upper_bound_inclusive": index == 9,
            "n": count,
            "n_sites": int(np.unique(sites[selected]).size),
            "station_balanced_weight": bin_weight,
            "mean_probability": mean_probability,
            "event_rate": event_rate,
        })
    return rows, float(ece)


def _fixture_probability_reference_predictions(
    reference: dict[str, object],
    sites: np.ndarray,
    target_dates: np.ndarray,
) -> np.ndarray:
    dates = pd.to_datetime(target_dates)
    if reference["mode"] == "pooled_month":
        monthly = reference["month_probability"]
        return np.asarray([
            float(monthly[str(int(month))]) for month in dates.month
        ])
    station_month = reference["station_month_probability"]
    return np.asarray([
        float(station_month[f"{site}|{int(month)}"])
        for site, month in zip(sites.astype(str), dates.month, strict=True)
    ])


def _fixture_probability_classification_diagnostics(
    event: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray,
) -> tuple[dict[str, float | None], dict[str, str]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score

    if np.unique(event).size < 2:
        reason = "SINGLE_CLASS_RETAINED_OUTCOMES"
        return (
            {
                "auroc": None,
                "auprc": None,
                "calibration_intercept": None,
                "calibration_slope": None,
            },
            {
                name: reason for name in (
                    "auroc", "auprc", "calibration_intercept",
                    "calibration_slope",
                )
            },
        )
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit_probability = np.log(clipped / (1 - clipped))
    calibration = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    calibration.fit(
        logit_probability.reshape(-1, 1), event, sample_weight=weights
    )
    return ({
        "auroc": float(roc_auc_score(
            event, probability, sample_weight=weights
        )),
        "auprc": float(average_precision_score(
            event, probability, sample_weight=weights
        )),
        "calibration_intercept": float(calibration.intercept_[0]),
        "calibration_slope": float(calibration.coef_[0, 0]),
    }, {})


def _fixture_probabilistic_evaluation_v2(
    verifier,
    *,
    protocol: dict[str, object],
    authorization: dict[str, object],
    frames: dict[str, pd.DataFrame],
    availability: pd.DataFrame,
    model_metadata: dict[tuple[str, str], dict[str, object]],
    required_models: dict[str, list[str]],
) -> dict[str, object]:
    contract = protocol["primary_inference_contract"][
        "probabilistic_event_contract"
    ]
    minimum_targets = protocol["availability_contract"][
        "minimum_valid_targets_per_station_horizon"
    ]
    assert minimum_targets == 100
    base_sha256 = verifier._sha256_json(contract)
    erratum_binding = authorization["probability_metric_erratum"]
    pipelines = _fixture_probability_pipeline_contracts(verifier)
    effective_contract = {
        "base_probabilistic_event_contract_sha256": base_sha256,
        "probability_metric_erratum": erratum_binding,
        "effective_output_artifact": verifier.PROBABILISTIC_EFFECTIVE_ARTIFACT_PATH,
        "metric_source_fields": dict(verifier.PROBABILISTIC_SOURCE_FIELDS),
        "nominal_quantile_handling": dict(
            verifier.PROBABILISTIC_NOMINAL_QUANTILE_HANDLING
        ),
        "bundle_scoring_pipeline_contracts": dict(
            verifier.PROBABILISTIC_PIPELINES
        ),
    }
    cohort_contracts: dict[str, object] = {}
    thresholds_by_cohort: dict[str, dict[str, float]] = {}
    for cohort in ("temporal", "external"):
        learned = [
            model for model in required_models[cohort]
            if model not in verifier.PROBABILISTIC_BUILTIN_MODELS
        ]
        primary = "ThermoRoute" if "ThermoRoute" in learned else learned[0]
        metadata = model_metadata[(cohort, primary)]
        thresholds = {
            str(site): float(value)
            for site, value in metadata["event_thresholds"].items()
        }
        reference = metadata["event_reference_climatology"]
        external = cohort == "external"
        cohort_contracts[cohort] = {
            "threshold_scope": (
                "pooled_development_train_q90"
                if external else "station_specific_development_train_q90"
            ),
            "threshold_registry_sha256": verifier._sha256_json(thresholds),
            "event_reference_format": reference["format"],
            "event_reference_mode": reference["mode"],
            "event_reference_sha256": verifier._sha256_json(reference),
            "event_reference_fit_interval": list(reference["fit_interval"]),
            "event_reference_fit_observation_count": int(
                reference["fit_observation_count"]
            ),
            "interpretation": (
                "exploratory pooled statistical tail threshold; non-ecological "
                "and not a waterbody-specific standard"
                if external
                else "site-local statistical tail diagnostic; not biological, "
                "regulatory, or cross-station comparable"
            ),
        }
        thresholds_by_cohort[cohort] = thresholds

    rows: list[dict[str, object]] = []
    for cohort in ("temporal", "external"):
        frame = frames[cohort]
        cohort_availability = availability.loc[
            availability["cohort"].astype(str).eq(cohort)
        ].copy()
        for model in required_models[cohort]:
            for horizon in (1, 3, 7):
                all_selected = frame.loc[
                    frame["model"].astype(str).eq(model)
                    & frame["horizon"].astype(int).eq(horizon)
                ].copy()
                available_horizon = cohort_availability.loc[
                    cohort_availability["horizon"].astype(int).eq(horizon)
                ]
                reportable_flags = available_horizon["reportable"].astype(
                    str
                ).str.lower().isin({"true", "1"})
                reportable_sites = set(
                    available_horizon.loc[reportable_flags, "site_no"].astype(str)
                )
                selected = all_selected.loc[
                    all_selected["site_id"].astype(str).isin(reportable_sites)
                ].copy()
                site_counts = selected["site_id"].astype(str).value_counts()
                n_sites = int(len(site_counts))
                if n_sites:
                    raw_weights = selected["site_id"].astype(str).map(
                        {site: 1.0 / int(count) for site, count in site_counts.items()}
                    ).to_numpy(float)
                    weights = raw_weights / raw_weights.sum()
                    site_total_weight = 1.0 / n_sites
                else:
                    weights = np.asarray([], dtype=float)
                    site_total_weight = None
                base = {
                    "cohort": cohort,
                    "model": model,
                    "horizon": horizon,
                    "n_forecasts_before_reportability_filter": len(all_selected),
                    "n_forecasts": len(selected),
                    "n_sites_before_reportability_filter": int(
                        all_selected["site_id"].astype(str).nunique()
                    ),
                    "n_sites": n_sites,
                    "minimum_targets_per_retained_site": minimum_targets,
                    "station_balanced_weight_sum": 0.0 if not n_sites else 1.0,
                    "minimum_site_total_weight": site_total_weight,
                    "maximum_site_total_weight": site_total_weight,
                    "threshold_scope": (
                        "pooled_development_train_q90"
                        if cohort == "external"
                        else "station_specific_development_train_q90"
                    ),
                }
                if model in verifier.PROBABILISTIC_BUILTIN_MODELS or not n_sites:
                    reason = (
                        "POINT_ONLY_BUILTIN_HAS_NO_FROZEN_PROBABILISTIC_HEAD"
                        if model in verifier.PROBABILISTIC_BUILTIN_MODELS
                        else "NO_STATION_HAS_100_COMMON_TARGETS"
                    )
                    rows.append({
                        **base,
                        "status": (
                            "NOT_AVAILABLE"
                            if model in verifier.PROBABILISTIC_BUILTIN_MODELS
                            else "NOT_ESTIMABLE"
                        ),
                        "reason": reason,
                        "event_count": None,
                        "non_event_count": None,
                        **{
                            name: None
                            for name in verifier.PROBABILISTIC_METRIC_FIELDS
                        },
                        "undefined_metric_reasons": {
                            name: reason
                            for name in verifier.PROBABILISTIC_METRIC_FIELDS
                        },
                        "reliability_bins": [],
                    })
                    continue
                metadata = model_metadata[(cohort, model)]
                sites = selected["site_id"].astype(str).to_numpy()
                offset_keys = [
                    f"{'__pooled__' if cohort == 'external' else site}|{horizon}"
                    for site in sites
                ]
                delta = np.asarray([
                    float(metadata["conformal_offsets"][key])
                    for key in offset_keys
                ])
                truth = selected["y_true"].to_numpy(float)
                cqr_q05 = selected["q05"].to_numpy(float)
                q50 = selected["q50"].to_numpy(float)
                cqr_q95 = selected["q95"].to_numpy(float)
                nominal_q05 = cqr_q05 + delta
                nominal_q95 = cqr_q95 - delta
                probability = selected["p_exceed"].to_numpy(float)
                threshold_registry = thresholds_by_cohort[cohort]
                thresholds = np.asarray([
                    threshold_registry[
                        "__pooled__" if cohort == "external" else site
                    ]
                    for site in sites
                ])
                event = (truth > thresholds).astype(int)
                p_clip = np.clip(probability, 1e-6, 1 - 1e-6)
                pinballs = {
                    "pinball_q05_c": float(np.sum(np.maximum(
                        0.05 * (truth - nominal_q05),
                        -0.95 * (truth - nominal_q05),
                    ) * weights)),
                    "pinball_q50_c": float(np.sum(np.maximum(
                        0.50 * (truth - q50),
                        -0.50 * (truth - q50),
                    ) * weights)),
                    "pinball_q95_c": float(np.sum(np.maximum(
                        0.95 * (truth - nominal_q95),
                        -0.05 * (truth - nominal_q95),
                    ) * weights)),
                }
                brier = float(np.sum((probability - event) ** 2 * weights))
                reference_probability = (
                    _fixture_probability_reference_predictions(
                        metadata["event_reference_climatology"],
                        sites,
                        selected["target_date"].to_numpy(),
                    )
                )
                reference_brier = float(np.sum(
                    (reference_probability - event) ** 2 * weights
                ))
                bins, ece = _fixture_probability_reliability_bins(
                    event, probability, weights, sites
                )
                diagnostics, undefined = (
                    _fixture_probability_classification_diagnostics(
                        event, probability, weights
                    )
                )
                pipeline_class = (
                    "LightGBM" if model == "LightGBM"
                    else "deep_LSTM_and_deterministic_controls"
                )
                pipeline = pipelines[pipeline_class]
                rows.append({
                    **base,
                    "status": "AVAILABLE",
                    "reason": None,
                    "event_count": int(event.sum()),
                    "non_event_count": int(len(event) - event.sum()),
                    "coverage_90": float(np.sum(
                        ((truth >= cqr_q05) & (truth <= cqr_q95)) * weights
                    )),
                    "mean_interval_width_c": float(np.sum(
                        (cqr_q95 - cqr_q05) * weights
                    )),
                    **dict(verifier.PROBABILISTIC_SOURCE_FIELDS),
                    "quantile_pipeline_class": pipeline_class,
                    "quantile_pipeline": pipeline["pipeline"],
                    "member_quantile_contract": pipeline[
                        "member_quantile_contract"
                    ],
                    "member_quantile_contract_sha256": pipeline[
                        "member_quantile_contract_sha256"
                    ],
                    "deployed_cqr_offset_min_c": float(delta.min()),
                    "deployed_cqr_offset_max_c": float(delta.max()),
                    "cqr_offset_scope": (
                        "pooled_external"
                        if cohort == "external"
                        else "station_horizon_temporal"
                    ),
                    "direct_nominal_forward_cqr_parity_bitwise": True,
                    "direct_nominal_q50_parity_bitwise": True,
                    "endpoint_inversion_used": False,
                    **pinballs,
                    "equal_weight_three_quantile_pinball_mean_c": float(
                        np.mean(list(pinballs.values()))
                    ),
                    "brier_score": brier,
                    "frozen_reference_brier_score": reference_brier,
                    "brier_skill_frozen_seasonal": 1 - brier / reference_brier,
                    "log_loss": float(np.sum(
                        -(event * np.log(p_clip)
                          + (1 - event) * np.log(1 - p_clip)) * weights
                    )),
                    "auroc": diagnostics["auroc"],
                    "auprc": diagnostics["auprc"],
                    "ece_10_equal_width": ece,
                    "calibration_intercept": diagnostics[
                        "calibration_intercept"
                    ],
                    "calibration_slope": diagnostics["calibration_slope"],
                    "event_rate": float(np.sum(event * weights)),
                    "undefined_metric_reasons": undefined,
                    "reliability_bins": bins,
                })
    return {
        "format": verifier.PROBABILISTIC_EVALUATION_FORMAT,
        "role": contract["role"],
        "contract_sha256": base_sha256,
        "base_probabilistic_event_contract_sha256": base_sha256,
        "probability_metric_erratum": erratum_binding,
        "effective_output_artifact": verifier.PROBABILISTIC_EFFECTIVE_ARTIFACT_PATH,
        "effective_contract": effective_contract,
        "effective_contract_sha256": verifier._sha256_json(effective_contract),
        "probabilistic_heads": ["q05", "q50", "q95", "p_exceed"],
        "aggregation": contract["aggregation"],
        "minimum_valid_targets_per_station_horizon": minimum_targets,
        "metric_weighting": (
            "station-balanced: each retained station total weight is 1/n_sites"
        ),
        "event_count_definition": (
            "unweighted raw counts of retained forecast rows by observed event class"
        ),
        "event_rate_and_probability_metric_weighting": (
            "station-balanced using the same per-row weights as all reported "
            "probability metrics"
        ),
        "central_interval_nominal_coverage": 0.90,
        "metric_sources": {
            "coverage_90_and_mean_interval_width_c": (
                verifier.PROBABILISTIC_INTERVAL_ENDPOINT_SOURCE
            ),
            "pinball_q05_q50_q95": (
                verifier.PROBABILISTIC_PINBALL_QUANTILE_SOURCE
            ),
            "event_probability": verifier.PROBABILISTIC_EVENT_PROBABILITY_SOURCE,
            "event_outcome": verifier.PROBABILISTIC_EVENT_OUTCOME_SOURCE,
        },
        "nominal_quantile_handling": dict(
            verifier.PROBABILISTIC_NOMINAL_QUANTILE_HANDLING
        ),
        "bundle_scoring_pipeline_contracts": pipelines,
        "interval_coverage_claim": (
            "station-balanced empirical marginal coverage only; no "
            "conditional-coverage or exchangeability guarantee"
        ),
        "three_quantile_score_definition": (
            "unscaled equal-weight arithmetic mean of nominal pre-CQR "
            "q05/q50/q95 pinball loss"
        ),
        "three_quantile_score_is_crps": False,
        "event_probability_calibration_period": "2018_only_before_confirmation",
        "evaluation_calibration_regression": (
            "weighted logistic regression of event on clipped forecast logit; "
            "sklearn lbfgs, C=1e6, max_iter=2000"
        ),
        "event_probability_clip_for_log_and_calibration_diagnostics": [
            1e-6, 1 - 1e-6
        ],
        "reliability_bins": "10_equal_width_bins_on_[0,1]",
        "single_class_auroc_auprc_and_calibration_parameters": "NA",
        "brier_skill_reference": (
            "bundle-frozen seasonal development train/calibration climatology"
        ),
        "confirmation_event_rate_used_as_brier_reference": False,
        "rev_status": "REV_NOT_EVALUATED_NO_PREDECLARED_COST_LOSS_RATIOS",
        "inference_computed": False,
        "cohort_contracts": cohort_contracts,
        "rows": rows,
    }


def _write_independent_probability_v2_fixture(
    verifier, root: Path, *, cqr_offset: float = 0.25,
) -> dict[str, object]:
    protocol = {
        "primary_inference_contract": {
            "probabilistic_event_contract": {
                "role": "DESCRIPTIVE_NOT_IN_CONFIRMATORY_FAMILY",
                "aggregation": "station_balanced_by_model_and_horizon",
            },
        },
        "availability_contract": {
            "minimum_valid_targets_per_station_horizon": 100,
        },
    }
    authorization = {
        "probability_metric_erratum": {
            "path": verifier.PROBABILITY_METRIC_ERRATUM_PATH,
            "sha256": "1" * 64,
            "format": verifier.PROBABILITY_METRIC_ERRATUM_FORMAT,
            "erratum_id": verifier.PROBABILITY_METRIC_ERRATUM_ID,
            "seal": {
                "path": verifier.PROBABILITY_METRIC_ERRATUM_SEAL_PATH,
                "sha256": "2" * 64,
            },
            "erratum_document_commit": "3" * 40,
        },
    }
    required_models = {
        "temporal": ["LightGBM"],
        "external": ["LightGBM"],
    }
    frames: dict[str, pd.DataFrame] = {}
    model_metadata: dict[tuple[str, str], dict[str, object]] = {}
    availability_rows: list[dict[str, object]] = []
    prediction_paths: dict[str, Path] = {}
    for cohort, site in (
        ("temporal", "01073319"),
        ("external", "02000001"),
    ):
        records: list[dict[str, object]] = []
        for horizon in (1, 3, 7):
            for index, issue in enumerate(
                pd.date_range("2021-01-01", periods=100, freq="D")
            ):
                truth = 10.0 if index % 2 == 0 else 12.0
                records.append({
                    "model": "LightGBM",
                    "scope": (
                        "route_a_temporal_confirmation"
                        if cohort == "temporal"
                        else "route_a_external_history_dependent_new_gage"
                    ),
                    "feature_set": "WTEMP+FLOW+TEMP+PRCP+RHMEAN+DH+WDSP",
                    "seed": -1,
                    "site_id": site,
                    "horizon": horizon,
                    "split": "confirm",
                    "issue_date": issue,
                    "target_date": issue + pd.Timedelta(days=horizon),
                    "y_true": truth,
                    "y_pred": truth,
                    "q05": truth - 1.0 - cqr_offset,
                    "q50": truth,
                    "q95": truth + 1.0 + cqr_offset,
                    "p_exceed": 0.25 if truth == 10.0 else 0.75,
                })
            availability_rows.append({
                "cohort": cohort,
                "site_no": site,
                "horizon": horizon,
                "n_valid_targets": 100,
                "reportable": True,
            })
        frame = pd.DataFrame.from_records(records).loc[
            :, list(verifier.DEVELOPMENT_REPLAY_PREDICTION_COLUMNS)
        ]
        prediction_path = root / f"{cohort}.parquet"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(prediction_path, index=False)
        frames[cohort] = frame
        prediction_paths[cohort] = prediction_path
        offset_group = "__pooled__" if cohort == "external" else site
        if cohort == "external":
            reference = {
                "format": "thermoroute.frozen-seasonal-event-reference.v1",
                "mode": "pooled_month",
                "threshold_scope": "pooled_development_train_q90",
                "fit_interval": ["2006-01-01", "2018-12-31"],
                "smoothing": 2.0,
                "global_probability": 0.5,
                "month_probability": {
                    str(month): 0.5 for month in range(1, 13)
                },
                "fit_observation_count": 100,
            }
        else:
            reference = {
                "format": "thermoroute.frozen-seasonal-event-reference.v1",
                "mode": "station_month",
                "threshold_scope": "station_development_train_q90",
                "fit_interval": ["2006-01-01", "2018-12-31"],
                "smoothing": 2.0,
                "global_probability": 0.5,
                "station_probability": {site: 0.5},
                "station_month_probability": {
                    f"{site}|{month}": 0.5 for month in range(1, 13)
                },
                "fit_observation_count": 100,
            }
        model_metadata[(cohort, "LightGBM")] = {
            "conformal_offsets": {
                f"{offset_group}|{horizon}": cqr_offset
                for horizon in (1, 3, 7)
            },
            "event_thresholds": {
                "__pooled__" if cohort == "external" else site: 11.0
            },
            "event_reference_climatology": reference,
        }
    availability = pd.DataFrame.from_records(availability_rows)
    availability_path = root / "availability.csv"
    availability.to_csv(availability_path, index=False, lineterminator="\n")
    artifact = _fixture_probabilistic_evaluation_v2(
        verifier,
        protocol=protocol,
        authorization=authorization,
        frames=frames,
        availability=availability,
        model_metadata=model_metadata,
        required_models=required_models,
    )
    artifact_path = root / "probabilistic_evaluation_v2.json"
    _write_canonical_json(verifier, root, artifact_path.name, artifact)
    return {
        "artifact_path": artifact_path,
        "availability_path": availability_path,
        "prediction_paths": prediction_paths,
        "model_metadata": model_metadata,
        "required_models": required_models,
        "protocol": protocol,
        "authorization": authorization,
    }


def _validate_independent_probability_fixture(verifier, fixture: dict[str, object]):
    return verifier._validate_probabilistic_evaluation_v2(
        artifact_path=fixture["artifact_path"],
        availability_path=fixture["availability_path"],
        prediction_paths=fixture["prediction_paths"],
        model_metadata=fixture["model_metadata"],
        required_models=fixture["required_models"],
        protocol=fixture["protocol"],
        authorization=fixture["authorization"],
    )


def _minimal_canonical_release(verifier, root: Path) -> None:
    _write_bytes(
        root,
        verifier.REPRODUCIBILITY_LOCK,
        ("fixture==1 \\\n    --hash=sha256:" + "0" * 64 + "\n").encode(),
    )
    for relative in verifier.CANONICAL_DEVELOPMENT_PATHS:
        _write_bytes(root, relative, b"{}\n" if relative.endswith(".json") else b"fixture\n")
    _write_bytes(root, "data_usgs/raw_snapshots/huc-v1/snapshot_index.json", b"{}\n")
    _write_bytes(root, "data_usgs/raw_snapshots/huc-v1/response.rdb")


def _write_protocol_seal_fixture(
    verifier,
    root: Path,
    *,
    original_commit: str = "1" * 40,
    final_commit: str = "2" * 40,
    original_markdown_sha256: str | None = None,
) -> tuple[Path, Path, Path]:
    protocol = _write_bytes(
        root,
        "protocols/route_a_confirmatory_v1.json",
        json.dumps({
            "protocol_id": "route-a-confirmatory-v1",
            "authoritative_protocol_commit": original_commit,
            "primary_inference_contract": {
                "confirmatory_family": _fixture_confirmatory_family(),
                "primary_models": [
                    "Persistence", "DampedPersistence", "Climatology",
                    "LightGBM", "LSTM", "ThermoRoute",
                ],
                "mandatory_exploratory_architecture_controls": [
                    "DampedPriorOnly", "TR-noDynamicPrior", "TR-fixedKappa",
                    "TR-noRouter", "TR-noMoE", "TR-noTCN", "TR-unbounded",
                ],
                "probabilistic_event_contract": json.loads(
                    (ROOT / "protocols/route_a_confirmatory_v1.json").read_text(
                        encoding="utf-8"
                    )
                )["primary_inference_contract"]["probabilistic_event_contract"],
            },
            "availability_contract": {
                "minimum_valid_targets_per_station_horizon": 100
            },
            "time_holdout": {
                "primary_target_start": "2021-01-01",
                "end": "2021-12-31",
            },
        }, sort_keys=True).encode() + b"\n",
    )
    markdown = _write_bytes(
        root,
        "protocols/route_a_confirmatory_protocol.md",
        b"# Final prelabel fixture protocol\n",
    )
    seal = {
        "format": verifier.PROTOCOL_SEAL_FORMAT,
        "status": "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED",
        "protocol_id": "route-a-confirmatory-v1",
        "original_preregistration": {
            "commit": original_commit,
            "markdown": {
                "path": "protocols/route_a_confirmatory_protocol.md",
                "sha256": original_markdown_sha256 or "3" * 64,
            },
        },
        "final_prelabel_protocol": {
            "commit": final_commit,
            "json": {
                "path": "protocols/route_a_confirmatory_v1.json",
                "sha256": verifier.sha256_file(protocol),
            },
            "markdown": {
                "path": "protocols/route_a_confirmatory_protocol.md",
                "sha256": verifier.sha256_file(markdown),
            },
        },
        "prelabel_attestation": {
            "external_timestamp_or_public_preregistration": False,
            "independent_custodian_or_worm_storage": False,
        },
    }
    seal_path = _write_bytes(
        root,
        verifier.PROTOCOL_SEAL_PATH,
        json.dumps(seal, sort_keys=True).encode() + b"\n",
    )
    return protocol, markdown, seal_path


def _write_claim_fixture_files(stage: Path) -> None:
    validator = stage / "scripts" / "26_validate_claims.py"
    validator.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "scripts" / "26_validate_claims.py", validator)
    registry = stage / "protocols" / "route_a_claim_registry_v1.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(json.dumps({
        "format": "thermoroute.route-a-claim-registry.v1",
        "documents": ["paper/main.md"],
        "claims": [{
            "claim_id": "FIXTURE",
            "status": "SUPPORTED_AFTER_OPENING",
            "forbidden_regex": ["NEVER_MATCH_THIS_FIXTURE"],
        }],
    }), encoding="utf-8")
    _write_bytes(stage, "paper/main.md", b"scoped development language\n")


def _materialize_claim_fixture(verifier, stage: Path, profile: str) -> None:
    _write_claim_fixture_files(stage)
    validator = stage / "scripts" / "26_validate_claims.py"
    registry = stage / "protocols" / "route_a_claim_registry_v1.json"
    scanned = [stage / "paper" / "main.md"]
    audit = {
        "format": "thermoroute.route-a-release-claim-audit.v1",
        "profile": profile,
        "require_complete": profile == verifier.POSTOPEN_PROFILE,
        "validator": verifier._binding_for(stage, validator),
        "registry": verifier._binding_for(stage, registry),
        "scanned_documents": [
            verifier._binding_for(stage, path) for path in scanned
        ],
        "violation_count": 0,
        "validator_stdout": "fixture static claim audit",
    }
    audit_path = _write_bytes(
        stage,
        verifier.CLAIM_AUDIT_PATH,
        json.dumps(audit, sort_keys=True).encode() + b"\n",
    )
    marker_path = stage / verifier.PROFILE_MARKER
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["claim_validation"] = verifier._binding_for(stage, audit_path)
    marker_path.write_bytes(verifier._canonical_json_bytes(marker))


def _write_postopen_fixture(verifier, root: Path) -> tuple[Path, dict[str, str]]:
    verifier._filesystem_development_input_closure = lambda _root: (
        FIXTURE_DEVELOPMENT_INPUT_CLOSURE_SHA256,
        FIXTURE_DEVELOPMENT_INPUT_CLOSURE_FILE_COUNT,
        ("fixture-development-input",),
    )
    _minimal_canonical_release(verifier, root)
    _write_bytes(
        root,
        "data_usgs/station_registry_v1.csv",
        b"site_no,huc2\n01073319,01\n",
    )
    _write_bytes(root, "requirements-lock.txt", b"numpy==1.0\n")
    _write_protocol_seal_fixture(verifier, root)
    _write_bytes(
        root,
        "data_usgs/external.csv",
        b"site_no,huc2\n02000001,02\n",
    )
    _write_bytes(root, "data_usgs/external.lock.json", b"{}\n")
    _write_bytes(root, "data_usgs/candidates.csv")
    _write_bytes(root, "data_usgs/candidates.provenance.json", b"{}\n")
    _write_bytes(root, "data_usgs/candidate-raw/snapshot_index.json", b"{}\n")
    _write_bytes(root, "data_usgs/candidate-raw/response.rdb")
    _write_bytes(root, "src/thermoroute/opening.py", b"# fixed\n")
    _write_bytes(root, "src/thermoroute/chronology.py", b"# chronology gate\n")
    for relative in (
        "src/thermoroute/outcome_qc.py",
        "src/thermoroute/probability_metric_erratum.py",
        "src/thermoroute/model_matrix_amendment.py",
        "src/thermoroute/coverage_audit.py",
        "src/thermoroute/coverage_bridge.py",
        "src/thermoroute/provenance.py",
        "src/thermoroute/repro.py",
        "src/thermoroute/usgs.py",
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    for relative in (
        "protocols/route_a_model_matrix_amendment_v1.json",
        "protocols/route_a_model_matrix_amendment_seal_v1.json",
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    coverage_policy_path = verifier.TEMPORAL_COVERAGE_POLICY_PATH
    coverage_policy_destination = root / coverage_policy_path
    coverage_policy_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / coverage_policy_path, coverage_policy_destination)
    _write_bytes(
        root, "scripts/28_freeze_prelabel_chronology.py", b"# chronology gate\n"
    )
    _write_bytes(root, "tests/test_chronology.py", b"# chronology gate\n")
    _write_bytes(root, "scripts/route_a_trusted_scorer.py", b"# fixed\n")
    _write_claim_fixture_files(root)

    protocol_binding = _binding(
        verifier, root, "protocols/route_a_confirmatory_v1.json"
    )
    protocol_seal_binding = _binding(verifier, root, verifier.PROTOCOL_SEAL_PATH)
    family = _fixture_confirmatory_family()
    family_sha256 = verifier._sha256_json(family)
    outcome_qc_policy_path = "protocols/route_a_outcome_qc_policy_v1.json"
    outcome_qc_policy = json.loads(
        (ROOT / outcome_qc_policy_path).read_text(encoding="utf-8")
    )
    outcome_qc_policy["base_protocol"] = protocol_binding
    outcome_qc_policy["confirmatory_family_sha256"] = family_sha256
    _write_bytes(
        root, outcome_qc_policy_path, json.dumps(outcome_qc_policy).encode()
    )
    amendment_path = "protocols/route_a_inference_amendment_v2.json"
    trusted_scoring_recovery_contract = json.loads(
        (ROOT / amendment_path).read_text(encoding="utf-8")
    )["trusted_scoring_recovery_contract"]
    policy_overlay = {
        **_binding(verifier, root, outcome_qc_policy_path),
        "required": True,
        "role": verifier.OUTCOME_QC_AMENDMENT_ROLE,
    }
    coverage_policy = json.loads(
        coverage_policy_destination.read_text(encoding="utf-8")
    )
    coverage_policy_overlay = {
        **_binding(verifier, root, coverage_policy_path),
        "required": True,
        "role": verifier.TEMPORAL_COVERAGE_AMENDMENT_ROLE,
    }
    amendment = {
        "format": "thermoroute.route-a-inference-amendment.v2",
        "status": "FROZEN_PRELABEL_OUTCOME_FREE",
        "amendment_id": "route-a-prelabel-inference-cqr-015",
        "recorded_date": "2026-07-24",
        "post_2020_wtemp_requested_or_inspected": False,
        "outcome_independent": True,
        "base_protocol": protocol_binding,
        "base_protocol_seal": protocol_seal_binding,
        "scientific_comparisons": {
            "count": 5,
            "confirmatory_family_sha256": family_sha256,
            "objects": family,
            "change_allowed": False,
        },
        "estimand_scope": {"fixture": "fixed cohort"},
        "inference_scope": {"fixture": "assumption conditional"},
        "decision_overlay": {
            "gate_artifact": "outputs/prelabel/route_a_inference_gate_v1.json",
            "all_gate_components_must_pass": True,
            "missing_unknown_or_not_run_is_failure": True,
            "gate_failure_verdict": "DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED",
            "supported_claim_allowed_when_gate_fails": False,
            "strong_p_value_or_favorable_interval_cannot_override_gate": True,
            "all_five_comparisons_must_still_be_rendered_exactly_once": True,
        },
        "additional_preopen_gates": {
            "outcome_qc_policy": policy_overlay,
            "temporal_coverage_policy": coverage_policy_overlay,
        },
        "trusted_scoring_recovery_contract": trusted_scoring_recovery_contract,
        "cqr_calibration_contract": verifier.CQR_AMENDMENT_CONTRACT,
        "lineage_contract": {
            "base_v1_files_remain_immutable": True,
            "separate_amendment_seal_required": True,
            "seal_path": "protocols/route_a_inference_amendment_seal_v2.json",
            "amendment_commit_must_precede_seal_commit": True,
        },
    }
    _write_bytes(root, amendment_path, json.dumps(amendment).encode())
    amendment_commit = "7" * 40
    amendment_seal_path = "protocols/route_a_inference_amendment_seal_v2.json"
    amendment_seal = {
        "format": "thermoroute.route-a-inference-amendment-seal.v2",
        "status": "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED",
        "amendment_id": amendment["amendment_id"],
        "amendment": _binding(verifier, root, amendment_path),
        "base_protocol_seal": protocol_seal_binding,
        "final_prelabel_commit": amendment_commit,
        "history_contract": {
            "base_protocol_commit_must_be_ancestor": True,
            "amendment_blob_must_match_commit": True,
            "amendment_commit_must_be_ancestor_of_authorization": True,
            "seal_is_created_only_after_amendment_commit": True,
        },
        "prelabel_attestation": {
            "post_2020_wtemp_requested_or_inspected": False,
            "outcome_independent": True,
        },
    }
    _write_bytes(root, amendment_seal_path, json.dumps(amendment_seal).encode())
    erratum_module = verifier._load_canonical_probability_metric_erratum_module(
        root
    )
    erratum = erratum_module.expected_probability_metric_erratum_document(
        root=root
    )
    _write_bytes(
        root,
        verifier.PROBABILITY_METRIC_ERRATUM_PATH,
        json.dumps(erratum).encode(),
    )
    erratum_seal = {
        "format": verifier.PROBABILITY_METRIC_ERRATUM_SEAL_FORMAT,
        "status": "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED",
        "erratum_id": verifier.PROBABILITY_METRIC_ERRATUM_ID,
        "erratum": _binding(
            verifier, root, verifier.PROBABILITY_METRIC_ERRATUM_PATH
        ),
        "governance_seals": {
            "base_protocol_seal": protocol_seal_binding,
            "inference_amendment_seal": _binding(
                verifier, root, amendment_seal_path
            ),
        },
        "erratum_document_commit": "8" * 40,
        "history_contract": dict(erratum_module.HISTORY_CONTRACT),
        "prelabel_attestation": dict(erratum_module.PRELABEL_ATTESTATION),
    }
    _write_bytes(
        root,
        verifier.PROBABILITY_METRIC_ERRATUM_SEAL_PATH,
        json.dumps(erratum_seal).encode(),
    )
    model_matrix_amendment = json.loads(
        (root / verifier.MODEL_MATRIX_AMENDMENT_PATH).read_text(
            encoding="utf-8"
        )
    )
    model_matrix_amendment_seal = json.loads(
        (root / verifier.MODEL_MATRIX_AMENDMENT_SEAL_PATH).read_text(
            encoding="utf-8"
        )
    )
    assert (
        verifier.sha256_file(root / verifier.MODEL_MATRIX_AMENDMENT_PATH)
        == verifier.MODEL_MATRIX_AMENDMENT_SHA256
    )
    assert (
        model_matrix_amendment_seal["governance_seals"]
        == verifier.MODEL_MATRIX_GOVERNANCE_SEALS
    )
    # The chronology gate registry has one exact producer order.  This fixture
    # needs only inert bytes for gates that are not executed by these tests,
    # while the semantic matrix documents above retain their real content.
    for relative in verifier.CHRONOLOGY_REQUIRED_GATE_PATHS:
        destination = root / relative
        if not destination.exists():
            _write_bytes(root, relative, f"# fixture gate: {relative}\n".encode())
    for relative in verifier.TRUSTED_VALIDATOR_PATHS:
        destination = root / relative
        if not destination.exists():
            _write_bytes(
                root, relative, f"# trusted-validator fixture: {relative}\n".encode()
            )

    runtime_sha256 = verifier._sha256_json({"fixture": True})
    model_entries, _model_artifact_paths = _write_development_model_fixtures(
        verifier, root, runtime_sha256=runtime_sha256
    )
    _write_bytes(
        root,
        verifier.DEVELOPMENT_REPLAY_ENTRYPOINT,
        b"#!/usr/bin/env python3\n# frozen replay fixture\n",
    )
    model_control_paths = sorted(verifier._working_model_control_paths(root))
    source_inventory = {
        relative: verifier.sha256_file(root / relative)
        for relative in model_control_paths
        if verifier._matches_source_inventory(relative)
    }
    source_sha256 = verifier._sha256_json(source_inventory)
    inference_policy = {
        "estimand_scope": {
            "fixed_cohort": "fixture fixed cohort",
            "superpopulation": "assumption conditional only",
        },
        "cluster_thresholds": {
            "minimum_clusters": 30,
            "minimum_effective_cluster_fraction": 0.75,
            "maximum_largest_cluster_share_exclusive": 0.25,
        },
        "structural_assumptions": [
            {
                "assumption_id": "INDEPENDENT_EXCHANGEABLE_HUC2_SAMPLING",
                "status": "NOT_ESTABLISHED",
            },
            {
                "assumption_id": "JOINT_CLUSTER_VECTOR_SIGN_SYMMETRY",
                "status": "NOT_ESTABLISHED",
            },
        ],
        "null_simulation": {
            "role": "FALSIFICATION_ONLY_NEVER_ESTABLISHES_STRUCTURAL_ASSUMPTIONS",
            "required_before_inferential_claims": True,
            "synthetic_boundary_null_only": True,
            "post_2020_outcomes_allowed": False,
            "caller_supplied_effects_allowed": False,
            "network_allowed": False,
            "scenarios": ["fixture"],
        },
        "decision": {
            "all_components_must_pass": True,
            "missing_unknown_or_not_run_is_failure": True,
            "failed_mode": "FIXED_COHORT_DESCRIPTIVE_ONLY",
            "failed_verdict": "DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED",
        },
    }
    gate_path = "outputs/prelabel/route_a_inference_gate_v1.json"
    registry_path = root / "data_usgs/station_registry_v1.csv"
    gate_geometry = verifier._independent_station_geometry(registry_path)
    threshold_failures = [
        "SMALL_CLUSTER_COUNT_LT_30",
        "DOMINANT_CLUSTER_SHARE_GE_0_25",
    ]
    structural_failures = [
        "INDEPENDENT_EXCHANGEABLE_HUC2_SAMPLING",
        "JOINT_CLUSTER_VECTOR_SIGN_SYMMETRY",
    ]
    null_gate = {
        **inference_policy["null_simulation"],
        "status": "NOT_RUN_BLOCKED_BY_STRUCTURAL_OR_CLUSTER_GATE",
        "pass": False,
        "outcomes_read": False,
        "network_used": False,
    }
    inference_gate = {
        "format": "thermoroute.route-a-inference-gate.v1",
        "status": "FAIL_CLOSED_DESCRIPTIVE_ONLY",
        "contains_confirmation_outcomes": False,
        "post_2020_outcomes_requested_or_inspected": False,
        "network_used": False,
        "inputs": {
            "base_protocol": protocol_binding,
            "base_protocol_seal": protocol_seal_binding,
            "station_registry": _binding(
                verifier, root, "data_usgs/station_registry_v1.csv"
            ),
            "source": {
                "source_tree_sha256": source_sha256,
                "source_inventory": source_inventory,
            },
        },
        "confirmatory_family": {
            "count": 5,
            "sha256": family_sha256,
            "objects": family,
            "candidate_reference_horizon_margin_unchanged": True,
        },
        "policy": inference_policy,
        "policy_sha256": verifier._sha256_json(inference_policy),
        "cluster_geometry": gate_geometry,
        "cluster_gate": {"pass": False, "failure_codes": threshold_failures},
        "structural_assumption_gate": {
            "pass": False,
            "failure_codes": structural_failures,
        },
        "null_simulation_gate": null_gate,
        "claim_eligible": False,
        "analysis_mode": "FIXED_COHORT_DESCRIPTIVE_ONLY",
        "blocking_reasons": [
            *(f"STRUCTURAL_ASSUMPTION_NOT_ESTABLISHED:{value}"
              for value in structural_failures),
            *threshold_failures,
            "NULL_SIMULATION_NOT_PASSING",
        ],
    }
    inference_gate["gate_self_sha256"] = verifier._sha256_json(inference_gate)
    _write_bytes(root, gate_path, json.dumps(inference_gate).encode())
    bridge_dependencies = {
        "frozen": "data_usgs/development_predictor_bridge_v1/frozen.parquet",
        "refreshed": "data_usgs/development_predictor_bridge_v1/refreshed.parquet",
        "report": "data_usgs/development_predictor_bridge_v1/report.json",
        "request_map": "data_usgs/development_predictor_bridge_v1/request_map.json",
        "daymet": "data_usgs/raw_snapshots/development-bridge/daymet.json",
        "gridmet": "data_usgs/raw_snapshots/development-bridge/gridmet.json",
        "gridmet_schema": "data_usgs/raw_snapshots/development-bridge/schema.json",
    }
    for relative in bridge_dependencies.values():
        _write_bytes(root, relative, b"{}\n" if relative.endswith(".json") else b"bridge\n")
    bridge_path = "data_usgs/development_predictor_bridge_v1.json"
    bridge = {
        "format": "thermoroute.development-predictor-bridge.v1",
        "status": "PASS_EXACT_PRODUCT_BRIDGE",
        "outcome_values_requested_or_read": False,
        "source_tree_sha256": source_sha256,
        "panel": _binding(verifier, root, "data_usgs/panel_usgs_120v2.parquet"),
        "registry": _binding(verifier, root, "data_usgs/station_registry_v1.csv"),
        "normalized": {
            name: _binding(verifier, root, bridge_dependencies[name])
            for name in ("frozen", "refreshed")
        },
        "report": _binding(verifier, root, bridge_dependencies["report"]),
        "request_map": _binding(verifier, root, bridge_dependencies["request_map"]),
        "raw_snapshot_indexes": {
            name: _binding(verifier, root, bridge_dependencies[name])
            for name in ("daymet", "gridmet", "gridmet_schema")
        },
    }
    _write_bytes(root, bridge_path, json.dumps(bridge).encode())

    def write_preopening_gate_fixtures() -> dict[str, dict[str, str]]:
        sys.path.insert(0, str(ROOT / "src"))
        try:
            from thermoroute.development_controls import (
                ARCHITECTURE_BUDGET_FORMAT,
                METRIC_SUMMARY_FORMAT,
                REPORT_FORMAT,
                SEMANTIC_AUDIT_FORMAT,
                architecture_budget_rows,
                budget_csv_bytes,
                declared_arms,
                recompute_metric_summary,
                recompute_paired_effect_summary,
                recompute_station_rmse,
                render_report,
                scientific_summary_document,
                summary_csv_bytes,
            )
        finally:
            sys.path.pop(0)

        expected_members = verifier._stage09b_release_members()
        controls_config = _stage09b_fixture_config(
            _binding(verifier, root, bridge_path), eval_batch_size=2,
        )
        identity = _fixture_run_identity(
            verifier,
            controls_config,
            panel_sha256=bridge["panel"]["sha256"],
            registry_sha256=bridge["registry"]["sha256"],
            source_sha256=source_sha256,
            runtime_sha256=runtime_sha256,
            input_closure_sha256=FIXTURE_DEVELOPMENT_INPUT_CLOSURE_SHA256,
        )
        run_dir = (
            f"outputs/runs/09b_development_controls/{identity['run_id']}"
        )
        matrix_audit = {
            "expected_members": len(expected_members),
            "prediction_rows": len(expected_members) * 9,
            "common_forecast_keys": 9,
            "splits": ["calib", "test", "val"],
            "reference_member": "PlainMLP-7var/seed0",
        }
        run_manifest_path = f"{run_dir}/run.json"
        _write_bytes(root, run_manifest_path, json.dumps({
            "schema_version": "thermoroute.run.v2",
            "identity": identity,
            "resolved_config": controls_config,
            "created_utc": "2026-07-22T00:00:00+00:00",
            "environment": {},
            "git": {},
            "provenance": {
                "development_only": True,
                "post_2020_outcomes_requested_or_read": False,
                "suite_pointer_written": False,
                "training_device": "cpu",
            },
        }).encode())
        arm_documents = {
            str(arm["arm_id"]): arm for arm in controls_config["arms"]
        }
        arm_features = {
            arm_id: str(document["feature_set"])
            for arm_id, document in arm_documents.items()
        }
        member_registry = []
        member_frames = {}
        base_parents = {
            "frozen_panel": identity["panel_sha256"],
            "frozen_station_registry": identity["registry_sha256"],
            "development_predictor_bridge": verifier.sha256_file(root / bridge_path),
        }
        for arm_id, seed in expected_members:
            relative = f"{run_dir}/arm_predictions/{arm_id}/seed{seed}.parquet"
            checkpoint_relative = f"{run_dir}/checkpoints/{arm_id}/seed{seed}.pt"
            checkpoint = _write_bytes(
                root, checkpoint_relative,
                f"fixture checkpoint {arm_id}/seed{seed}\n".encode(),
            )
            checkpoint_sidecar_relative = checkpoint_relative + ".meta.json"
            arm_document = arm_documents[arm_id]
            arm_config = {
                **controls_config,
                "arm": arm_document,
                "seed": seed,
                "trainable_parameters": controls_config["parameter_counts"][arm_id],
            }
            expected_model_class = {
                "PlainMLP": "thermoroute.neural_baselines.PlainMLPForecaster",
                "PlainCausalTCN": (
                    "thermoroute.neural_baselines.PlainCausalTCNForecaster"
                ),
                "ThermoRoute": "thermoroute.thermoroute.ThermoRoute",
            }[str(arm_document["family"])]
            _write_bytes(root, checkpoint_sidecar_relative, json.dumps({
                "format": "thermoroute.training-checkpoint-metadata.v2",
                "checkpoint_format": "thermoroute.training-checkpoint.v3",
                "run_id": identity["run_id"],
                "epoch": 4,
                "checkpoint_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": verifier.sha256_file(checkpoint),
                "resolved_config_sha256": verifier._sha256_json(arm_config),
                "extra_sha256": "e" * 64,
                "model_class": expected_model_class,
                "optimizer_class": "torch.optim.adamw.AdamW",
                "scheduler_class": "torch.optim.lr_scheduler.ReduceLROnPlateau",
                "scheduler_present": True,
            }).encode())
            records = []
            for split_index, split in enumerate(("val", "calib", "test")):
                for horizon in (1, 3, 7):
                    issue_date = pd.Timestamp("2017-01-01") + pd.Timedelta(
                        days=split_index * 30 + horizon
                    )
                    truth = float(split_index + horizon / 10)
                    prediction_value = truth + float(seed + 1) / 100
                    records.append({
                        "model": arm_id,
                        "scope": "development_only_2006_2020",
                        "feature_set": arm_features[arm_id],
                        "seed": seed,
                        "site_id": "01073319",
                        "horizon": horizon,
                        "split": split,
                        "issue_date": issue_date,
                        "target_date": issue_date + pd.Timedelta(days=horizon),
                        "y_true": truth,
                        "y_pred": prediction_value,
                        "q05": prediction_value - 1.0,
                        "q50": prediction_value,
                        "q95": prediction_value + 1.0,
                        "p_exceed": 0.25,
                    })
            frame = pd.DataFrame.from_records(records)
            prediction = root / relative
            prediction.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(prediction, index=False)
            member_frames[(arm_id, seed)] = frame
            sidecar_relative = relative + ".meta.json"
            checkpoint_binding = _binding(
                verifier, root, checkpoint_relative,
            )
            checkpoint_sidecar_binding = _binding(
                verifier, root, checkpoint_sidecar_relative,
            )
            member_parents = {
                **base_parents,
                "training_checkpoint": checkpoint_binding["sha256"],
                "training_checkpoint_sidecar": checkpoint_sidecar_binding["sha256"],
            }
            training_summary = {
                "best_validation_metric": 0.25,
                "selected_epoch": 2,
                "checkpoint_final_epoch": 4,
            }
            member_extra = {
                "format": "thermoroute.development-control-arm.v2",
                "arm_id": arm_id,
                "family": arm_document["family"],
                "feature_set": arm_document["feature_set"],
                "variables": arm_document["variables"],
                "seed": seed,
                "trainable_parameters": controls_config["parameter_counts"][arm_id],
                "architecture": controls_config["architecture_templates"][arm_id],
                "training_device": "cpu",
                "station_balanced": True,
                "selection_metric": "station_macro",
                "train_config": controls_config["train_config"],
                "context_length": 32,
                "horizons": [1, 3, 7],
                "development_only": True,
                "development_evaluation_interval": [
                    "2019-01-01", "2020-12-31"
                ],
                "blind_or_confirmatory": False,
                "suite_pointer_written": False,
                "eval_batch_size": 2,
                "training_summary": training_summary,
            }
            _write_bytes(root, sidecar_relative, json.dumps({
                "schema_version": "thermoroute.artifact.v1",
                "kind": "development_control_arm_predictions",
                "artifact": prediction.name,
                "artifact_sha256": verifier.sha256_file(prediction),
                "artifact_bytes": prediction.stat().st_size,
                "content_schema": "thermoroute.predictions.v1",
                "run": identity,
                "parents": dict(sorted(member_parents.items())),
                "extra": member_extra,
                "created_utc": "2026-07-22T00:00:00+00:00",
            }).encode())
            member_registry.append({
                "arm_id": arm_id,
                "seed": seed,
                "checkpoint": checkpoint_binding,
                "checkpoint_sidecar": checkpoint_sidecar_binding,
                "prediction": _binding(verifier, root, relative),
                "prediction_sidecar": _binding(verifier, root, sidecar_relative),
            })
        final_paths = {
            "predictions": f"{run_dir}/development_controls_predictions.parquet",
            "architecture_budget": (
                f"{run_dir}/development_controls_architecture_budget.csv"
            ),
            "metric_summary": f"{run_dir}/development_controls_metric_summary.csv",
            "report": f"{run_dir}/development_controls_report.md",
            "semantic_audit": f"{run_dir}/development_controls_semantic_audit.json",
        }
        combined = pd.concat(
            [member_frames[member] for member in expected_members], ignore_index=True
        )
        combined.to_parquet(root / final_paths["predictions"], index=False)
        budget = architecture_budget_rows(
            declared_arms(), n_stations=120, train_examples=9
        )
        _write_bytes(
            root, final_paths["architecture_budget"],
            budget_csv_bytes(budget),
        )
        summary = recompute_metric_summary(member_frames)
        paired_effects = recompute_paired_effect_summary(
            recompute_station_rmse(member_frames),
            exact_common_forecast_keys_verified=True,
        )
        _write_bytes(
            root, final_paths["metric_summary"], summary_csv_bytes(summary)
        )
        _write_bytes(
            root,
            final_paths["report"],
            render_report(
                run_id=identity["run_id"],
                audit=matrix_audit,
                budget=budget,
                summary=summary,
                paired_effects=paired_effects,
            ).encode("utf-8"),
        )
        final_specs = {
            "predictions": (
                "development_controls_combined_predictions",
                "thermoroute.predictions.v1", "combined_predictions",
            ),
            "architecture_budget": (
                "development_controls_budget", ARCHITECTURE_BUDGET_FORMAT,
                "architecture_budget",
            ),
            "metric_summary": (
                "development_controls_metric_summary", METRIC_SUMMARY_FORMAT,
                "metric_summary",
            ),
            "report": (
                "development_controls_report", REPORT_FORMAT, "report",
            ),
            "semantic_audit": (
                "development_controls_semantic_audit", SEMANTIC_AUDIT_FORMAT,
                "semantic_audit",
            ),
        }
        final_parents = {
            **base_parents,
            **{
                f"arm::{entry['arm_id']}::seed{entry['seed']}::prediction": (
                    entry["prediction"]["sha256"]
                )
                for entry in member_registry
            },
            **{
                f"arm::{entry['arm_id']}::seed{entry['seed']}::checkpoint": (
                    entry["checkpoint"]["sha256"]
                )
                for entry in member_registry
            },
            **{
                f"arm::{entry['arm_id']}::seed{entry['seed']}::checkpoint_sidecar": (
                    entry["checkpoint_sidecar"]["sha256"]
                )
                for entry in member_registry
            },
        }

        def write_final_sidecar(name: str) -> None:
            relative = final_paths[name]
            artifact = root / relative
            kind, content_schema, role = final_specs[name]
            _write_bytes(root, relative + ".meta.json", json.dumps({
                "schema_version": "thermoroute.artifact.v1",
                "kind": kind,
                "artifact": artifact.name,
                "artifact_sha256": verifier.sha256_file(artifact),
                "artifact_bytes": artifact.stat().st_size,
                "content_schema": content_schema,
                "run": identity,
                "parents": dict(sorted(final_parents.items())),
                "extra": verifier._stage09b_expected_final_extra(
                    matrix_audit, role=role,
                ),
                "created_utc": "2026-07-22T00:00:00+00:00",
            }).encode())

        for name in ("predictions", "architecture_budget", "metric_summary", "report"):
            write_final_sidecar(name)

        canonical_evaluation = verifier._normalise_stage09b_release_prediction(
            member_frames[expected_members[0]],
            arm_id=expected_members[0][0],
            seed=expected_members[0][1],
            feature_set=arm_features[expected_members[0][0]],
            reference=None,
        )[["split", "site_id", "horizon", "issue_date", "target_date", "y_true"]]
        canonical_evaluation_sha256 = verifier._stage09b_window_registry_digest(
            canonical_evaluation
        )
        canonical_train_sha256 = verifier._stage09b_window_registry_digest(
            canonical_evaluation
        )

        def fixture_canonical_windows(*_args, **_kwargs):
            return (
                canonical_evaluation.copy(), 9,
                canonical_evaluation_sha256, canonical_train_sha256,
                ("01073319",),
            )

        verifier._stage09b_rebuild_canonical_windows = fixture_canonical_windows
        semantic_members = []
        for entry in member_registry:
            prediction = root / entry["prediction"]["path"]
            prediction_sidecar = root / entry["prediction_sidecar"]["path"]
            checkpoint = root / entry["checkpoint"]["path"]
            checkpoint_sidecar = root / entry["checkpoint_sidecar"]["path"]
            member = (entry["arm_id"], entry["seed"])
            normalised = verifier._normalise_stage09b_release_prediction(
                member_frames[member], arm_id=entry["arm_id"], seed=entry["seed"],
                feature_set=arm_features[entry["arm_id"]],
                reference=canonical_evaluation,
            )
            semantic_members.append({
                "arm_id": entry["arm_id"],
                "seed": entry["seed"],
                "checkpoint": {
                    "sha256": verifier.sha256_file(checkpoint),
                    "bytes": checkpoint.stat().st_size,
                },
                "checkpoint_sidecar": {
                    "sha256": verifier.sha256_file(checkpoint_sidecar),
                    "bytes": checkpoint_sidecar.stat().st_size,
                },
                "prediction": {
                    "sha256": verifier.sha256_file(prediction),
                    "bytes": prediction.stat().st_size,
                },
                "prediction_sidecar": {
                    "sha256": verifier.sha256_file(prediction_sidecar),
                    "bytes": prediction_sidecar.stat().st_size,
                },
                "normalised_prediction_sha256": (
                    verifier._stage09b_prediction_content_digest(normalised)
                ),
                "best_model_state_prediction_replay_verified": True,
            })
        derived = {}
        for label, name in (
            ("architecture_budget", "architecture_budget"),
            ("combined_predictions", "predictions"),
            ("metric_summary", "metric_summary"),
            ("report", "report"),
        ):
            artifact = root / final_paths[name]
            sidecar = root / (final_paths[name] + ".meta.json")
            derived[label] = {
                "artifact": {
                    "sha256": verifier.sha256_file(artifact),
                    "bytes": artifact.stat().st_size,
                },
                "sidecar": {
                    "sha256": verifier.sha256_file(sidecar),
                    "bytes": sidecar.stat().st_size,
                },
            }
        semantic = {
            "format": "thermoroute.development-controls-semantic-audit.v3",
            "status": "PASS_BEST_MODEL_STATE_PREDICTION_REPLAY",
            "run_id": identity["run_id"],
            "evidence_scope": "best_model_state_prediction_replay",
            "best_model_state_prediction_replay_verified": True,
            "training_replay_verified": False,
            "post_2020_outcomes_requested_or_read": False,
            "matrix_audit": matrix_audit,
            "canonical_window_registry": {
                "sha256": canonical_evaluation_sha256,
                "common_forecast_keys": 9,
                "train_examples_per_epoch": 9,
                "train_registry_sha256": canonical_train_sha256,
            },
            "scientific_summary": scientific_summary_document(paired_effects),
            "members": semantic_members,
            "derived_artifacts": derived,
        }
        semantic["semantic_audit_self_sha256"] = verifier._sha256_json(semantic)
        _write_bytes(
            root, final_paths["semantic_audit"], json.dumps(semantic).encode()
        )
        write_final_sidecar("semantic_audit")
        controls_artifacts = {
            "run_manifest": _binding(verifier, root, run_manifest_path),
            "frozen_panel_spec": _binding(
                verifier, root, "data_usgs/frozen_panel_v1.json"
            ),
            "panel": bridge["panel"],
            "registry": bridge["registry"],
            "predictor_bridge": _binding(verifier, root, bridge_path),
            **{
                name: _binding(verifier, root, relative)
                for name, relative in final_paths.items()
            },
            "prediction_sidecar": _binding(
                verifier, root, final_paths["predictions"] + ".meta.json"
            ),
            "architecture_budget_sidecar": _binding(
                verifier, root, final_paths["architecture_budget"] + ".meta.json"
            ),
            "metric_summary_sidecar": _binding(
                verifier, root, final_paths["metric_summary"] + ".meta.json"
            ),
            "report_sidecar": _binding(
                verifier, root, final_paths["report"] + ".meta.json"
            ),
            "semantic_audit_sidecar": _binding(
                verifier, root, final_paths["semantic_audit"] + ".meta.json"
            ),
        }
        controls = {
            "format": "thermoroute.stage09b-completion-receipt.v3",
            "status": "PASS_STAGE09B_BEST_MODEL_STATE_PREDICTION_REPLAY",
            "stage": "09b_development_controls",
            "run_id": identity["run_id"],
            "run_identity": identity,
            "formal_configuration": controls_config,
            "evidence_scope": "best_model_state_prediction_replay",
            "best_model_state_prediction_replay_verified": True,
            "training_replay_verified": False,
            "matrix_audit": matrix_audit,
            "member_registry": member_registry,
            "artifacts": controls_artifacts,
            "post_2020_outcomes_requested_or_read": False,
        }
        controls["receipt_self_sha256"] = verifier._sha256_json(controls)
        controls_path = "outputs/models/route_a_stage09b_completion.json"
        _write_bytes(root, controls_path, json.dumps(controls).encode())

        stage9_config = {
            "stage": "09_usgs_experiment",
            "input_closure_sha256": (
                FIXTURE_DEVELOPMENT_INPUT_CLOSURE_SHA256
            ),
            "input_closure_file_count": (
                FIXTURE_DEVELOPMENT_INPUT_CLOSURE_FILE_COUNT
            ),
        }
        stage9_identity = _fixture_run_identity(
            verifier,
            stage9_config,
            panel_sha256=bridge["panel"]["sha256"],
            registry_sha256=bridge["registry"]["sha256"],
            source_sha256=source_sha256,
            runtime_sha256=runtime_sha256,
            input_closure_sha256=FIXTURE_DEVELOPMENT_INPUT_CLOSURE_SHA256,
        )
        stage9_run_manifest = (
            "outputs/runs/09_usgs_experiment/"
            f"{stage9_identity['run_id']}/run.json"
        )
        _write_canonical_json(verifier, root, stage9_run_manifest, {
            "schema_version": "thermoroute.run.v2",
            "identity": stage9_identity,
            "resolved_config": stage9_config,
            "created_utc": "2026-07-22T00:00:00+00:00",
            "environment": {},
            "git": {},
            "provenance": {
                "confirmation_outcomes_requested_or_read": False,
            },
        })
        stage9_artifacts = {
            "run_manifest": _binding(verifier, root, stage9_run_manifest),
        }
        for name in (
            "predictions", "prediction_sidecar", "scores", "report",
            "lightgbm_selection", "thermoroute_pointer",
            "lightgbm_pointer", "components_pointer",
        ):
            relative = (
                "outputs/runs/09_usgs_experiment/"
                f"{stage9_identity['run_id']}/{name}.json"
            )
            _write_bytes(root, relative, b"{}\n")
            stage9_artifacts[name] = _binding(verifier, root, relative)
        stage9 = {
            "format": "thermoroute.stage09-completion-receipt.v1",
            "status": "PASS_FORMAL_STAGE09_COMPLETE",
            "stage": "09_usgs_experiment",
            "run_id": stage9_identity["run_id"],
            "run_identity": stage9_identity,
            "formal_configuration": stage9_config,
            "confirmation_outcomes_requested_or_read": False,
            "artifacts": stage9_artifacts,
        }
        stage9["receipt_self_sha256"] = verifier._sha256_json(stage9)
        stage9_path = "outputs/models/route_a_stage09_completion.json"
        _write_bytes(root, stage9_path, json.dumps(stage9).encode())

        stage25_development = {
            "frozen_panel_spec": _binding(
                verifier, root, "data_usgs/frozen_panel_v1.json"
            ),
            "panel": bridge["panel"],
            "registry": bridge["registry"],
            "predictor_bridge": _binding(verifier, root, bridge_path),
            "source_sha256": source_sha256,
        }
        stage16_gate, _stage16_paths = _write_stage16_gate_fixture(
            verifier,
            root,
            model_entries=model_entries,
            development_contract=stage25_development,
            stage9_receipt=stage9,
            stage9_path=stage9_path,
            source_sha256=source_sha256,
            runtime_sha256=runtime_sha256,
        )
        stage25_gate, _stage25_paths = _write_stage25_gate_fixture(
            verifier,
            root,
            model_entries=model_entries,
            development_contract=stage25_development,
            source_sha256=source_sha256,
            runtime_sha256=runtime_sha256,
        )
        return {
            "stage09_completion": _binding(verifier, root, stage9_path),
            "stage09b_development_controls": _binding(
                verifier, root, controls_path
            ),
            "stage16_lstm_completion": stage16_gate,
            "stage25_external_completion": stage25_gate,
        }

    preopening_gates = write_preopening_gate_fixtures()
    suite = {
        "format": "thermoroute.route-a-model-suite.v1",
        "status": "FROZEN_BEFORE_LABEL_OPENING",
        "training_device": "cpu",
        "numerical_runtime_sha256": runtime_sha256,
        "protocol_sha256": protocol_binding["sha256"],
        "actual_feature_order": [
            "WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP"
        ],
        "model_matrix_amendment": {
            "format": verifier.MODEL_MATRIX_SUITE_BINDING_FORMAT,
            "document": {
                "path": verifier.MODEL_MATRIX_AMENDMENT_PATH,
                "sha256": verifier.sha256_file(
                    root / verifier.MODEL_MATRIX_AMENDMENT_PATH
                ),
                "format": verifier.MODEL_MATRIX_AMENDMENT_FORMAT,
                "status": verifier.MODEL_MATRIX_AMENDMENT_STATUS,
                "amendment_id": verifier.MODEL_MATRIX_AMENDMENT_ID,
                "amendment_document_commit": model_matrix_amendment_seal[
                    "amendment_document_commit"
                ],
            },
            "seal": {
                "path": verifier.MODEL_MATRIX_AMENDMENT_SEAL_PATH,
                "sha256": verifier.sha256_file(
                    root / verifier.MODEL_MATRIX_AMENDMENT_SEAL_PATH
                ),
                "format": verifier.MODEL_MATRIX_AMENDMENT_SEAL_FORMAT,
                "status": verifier.MODEL_MATRIX_AMENDMENT_SEAL_STATUS,
            },
            "contract_id": verifier._model_matrix_contract_id(
                model_matrix_amendment
            ),
        },
        "preopening_gates": preopening_gates,
        "development_contract": {
            "frozen_panel_spec": _binding(
                verifier, root, "data_usgs/frozen_panel_v1.json"
            ),
            "panel": bridge["panel"],
            "registry": bridge["registry"],
            "predictor_bridge": _binding(verifier, root, bridge_path),
            "source_sha256": source_sha256,
        },
        "cohorts": {
            cohort: {"models": entries}
            for cohort, entries in model_entries.items()
        },
    }
    suite_path = root / "data_usgs/confirmatory_model_suite_v1.json"
    suite_path.write_text(json.dumps(suite), encoding="utf-8")
    source_tree_sha256 = source_sha256
    development_replay = _development_replay_fixture(
        verifier,
        root,
        suite=suite,
        source_sha256=source_tree_sha256,
        runtime_sha256=runtime_sha256,
    )
    development_replay_path = root / verifier.DEVELOPMENT_REPLAY_RECEIPT
    development_replay_path.parent.mkdir(parents=True, exist_ok=True)
    development_replay_path.write_bytes(
        verifier._lineage_canonical_json_bytes(development_replay)
    )

    cohort_tables = {}
    for cohort in ("temporal", "external"):
        relative = f"data_usgs/prelabel/{cohort}.parquet"
        _write_bytes(root, relative)
        cohort_tables[cohort] = _binding(verifier, root, relative)
    evidence = []
    for index in range(4):
        relative = f"data_usgs/raw_snapshots/met-{index}/snapshot_index.json"
        _write_bytes(root, relative, b"{}\n")
        _write_bytes(root, f"data_usgs/raw_snapshots/met-{index}/response.bin")
        evidence.append({
            "contains_outcome": False,
            "contains_outcome_labels": False,
            "artifact": _binding(verifier, root, relative),
        })
    inputs = {
        "format": "thermoroute.route-a-prelabel-inputs.v1",
        "status": "FROZEN_PRELABEL_NO_OUTCOMES",
        "contains_outcome": False,
        "contains_outcome_labels": False,
        "labels_requested_or_read": False,
        "outcome_endpoint_called": False,
        "post_2020_wtemp_requested_or_inspected": False,
        "cohort_tables": cohort_tables,
        "source_evidence": evidence,
    }
    inputs_path = root / "data_usgs/confirmatory_actual_inputs_v1.json"
    inputs_path.write_text(json.dumps(inputs), encoding="utf-8")

    chronology_order = {
        "model_freeze_commit": "4" * 40,
        "input_evidence_commit": "5" * 40,
        "receipt_creation_base_commit": "6" * 40,
        "strict_order_verified": True,
    }
    chronology_stable = {
        "format": verifier.CHRONOLOGY_FORMAT,
        "status": verifier.CHRONOLOGY_STATUS,
        "order": chronology_order,
        "model_matrix_history": {
            "format": verifier.MODEL_MATRIX_HISTORY_FORMAT,
            "amendment": _chronology_binding(
                verifier, root, verifier.MODEL_MATRIX_AMENDMENT_PATH
            ),
            "seal": _chronology_binding(
                verifier, root, verifier.MODEL_MATRIX_AMENDMENT_SEAL_PATH
            ),
            "amendment_id": verifier.MODEL_MATRIX_AMENDMENT_ID,
            "amendment_document_commit": model_matrix_amendment_seal[
                "amendment_document_commit"
            ],
            "seal_commit": "3" * 40,
            "model_freeze_commit": chronology_order["model_freeze_commit"],
            "contract_id": verifier._model_matrix_contract_id(
                model_matrix_amendment
            ),
            "strict_order_verified": True,
            "immutable_to_release_tip": True,
            "evidence_scope": verifier.CHRONOLOGY_EVIDENCE_SCOPE,
        },
        "protocol_history": {
            "seal": _chronology_binding(
                verifier, root, verifier.PROTOCOL_SEAL_PATH
            ),
            "original_commit": "1" * 40,
            "final_prelabel_commit": "2" * 40,
            "declared_git_show_bindings": [
                {
                    "role": "original_markdown",
                    "commit": "1" * 40,
                    "path": "protocols/route_a_confirmatory_protocol.md",
                    "sha256": "3" * 64,
                },
                {
                    "role": "final_json",
                    "commit": "2" * 40,
                    "path": "protocols/route_a_confirmatory_v1.json",
                    "sha256": verifier.sha256_file(
                        root / "protocols/route_a_confirmatory_v1.json"
                    ),
                },
                {
                    "role": "final_markdown",
                    "commit": "2" * 40,
                    "path": "protocols/route_a_confirmatory_protocol.md",
                    "sha256": verifier.sha256_file(
                        root / "protocols/route_a_confirmatory_protocol.md"
                    ),
                },
            ],
        },
        "paths": {
            "protocol_seal": verifier.PROTOCOL_SEAL_PATH,
            "model_matrix_amendment": verifier.MODEL_MATRIX_AMENDMENT_PATH,
            "model_matrix_amendment_seal": (
                verifier.MODEL_MATRIX_AMENDMENT_SEAL_PATH
            ),
            "model_suite": "data_usgs/confirmatory_model_suite_v1.json",
            "development_replay": (
                "outputs/model_replay/route_a_development_replay_v1.json"
            ),
            "candidate_table": "data_usgs/candidates.csv",
            "candidate_provenance": "data_usgs/candidates.provenance.json",
            "candidate_snapshot_index": (
                "data_usgs/candidate-raw/snapshot_index.json"
            ),
            "external_registry": "data_usgs/external.csv",
            "external_lock": "data_usgs/external.lock.json",
            "input_manifest": "data_usgs/confirmatory_actual_inputs_v1.json",
        },
        "required_gate_files_at_model_freeze": [
            _chronology_binding(verifier, root, relative)
            for relative in verifier.CHRONOLOGY_REQUIRED_GATE_PATHS
        ],
        "model_source_control_artifacts": [
            _chronology_binding(verifier, root, relative)
            for relative in model_control_paths
        ],
        "source_tree_sha256": source_tree_sha256,
        "model_freeze_artifacts": [
            _chronology_binding(verifier, root, relative)
            for relative in (
                "data_usgs/confirmatory_model_suite_v1.json",
                "outputs/model_replay/route_a_development_replay_v1.json",
            )
        ],
        "input_evidence_artifacts": [
            _chronology_binding(verifier, root, relative)
            for relative in sorted((
                "data_usgs/candidates.csv",
                "data_usgs/candidates.provenance.json",
                "data_usgs/candidate-raw/snapshot_index.json",
                "data_usgs/external.csv",
                "data_usgs/external.lock.json",
                "data_usgs/confirmatory_actual_inputs_v1.json",
            ))
        ],
        "absence_at_model_freeze": {
            "checked_paths": [
                "data_usgs/candidates.csv",
                "data_usgs/confirmatory_actual_inputs_v1.json",
            ],
            "present_paths": [],
        },
        "post_model_control_audit": {
            "protected_directories": list(verifier.PROTECTED_DIRECTORIES),
            "protected_exact_files": list(verifier.PROTECTED_EXACT_FILES),
            "protected_root_patterns": list(verifier.PROTECTED_ROOT_PATTERNS),
            "committed_touches": [],
            "worktree_changes": [],
        },
        "post_freeze_artifact_mutation_count": 0,
        "external_timestamp_or_public_preregistration": False,
        "independent_custodian_or_worm_storage": False,
        "evidence_scope": verifier.CHRONOLOGY_EVIDENCE_SCOPE,
        "fallback_if_validation_fails": (
            "TRANSDUCTIVE_RETROSPECTIVE_EXPLORATION_"
            "CONFIRMATION_CLAIMS_PROHIBITED"
        ),
    }
    chronology = {
        **chronology_stable,
        "receipt_self_sha256": verifier._chronology_self_sha256(
            chronology_stable
        ),
    }
    chronology_path = root / verifier.CHRONOLOGY_PATH
    chronology_path.parent.mkdir(parents=True, exist_ok=True)
    chronology_path.write_bytes(verifier._canonical_json_bytes(chronology))

    namespace = "b" * 24
    base = f"outputs/confirmatory/route_a_{namespace}"
    state = {
        "namespace": namespace,
        "run_directory": base,
        "work_order": f"{base}/acquisition_work_order_v1.json",
        "intent": f"{base}/opening_intent_v1.json",
        "transport_root": f"{base}/transport",
        "raw_nwis_root": f"{base}/transport/raw_nwis_v1",
        "raw_nwis_snapshot_index": (
            f"{base}/transport/raw_nwis_v1/snapshot_index.json"
        ),
        "acquisition_request_map": f"{base}/acquisition/source_request_map_v1.json",
        "temporal_outcomes": f"{base}/acquisition/temporal_outcomes_v1.parquet",
        "external_outcomes": f"{base}/acquisition/external_outcomes_v1.parquet",
        "acquisition_manifest": f"{base}/acquisition/acquisition_manifest_v1.json",
        "availability_registry": f"{base}/trusted/availability_registry_v1.csv",
        "outcome_quality_audit": f"{base}/trusted/outcome_quality_audit_v1.json",
        "outcome_qc_gate": f"{base}/trusted/outcome_qc_gate_v1.json",
        "approved_target_sensitivity": f"{base}/trusted/approved_target_sensitivity_v1.json",
        "spatial_sensitivity": f"{base}/trusted/spatial_sensitivity_v1.json",
        "probabilistic_evaluation": f"{base}/trusted/probabilistic_evaluation_v2.json",
        "temporal_predictions": f"{base}/trusted/temporal_predictions_v1.parquet",
        "external_predictions": f"{base}/trusted/external_predictions_v1.parquet",
        "statistics": f"{base}/trusted/statistics_v1.json",
        "temporal_coverage_audit": (
            f"{base}/trusted/temporal_coverage_audit_v1.json"
        ),
        "report": f"{base}/trusted/report_v1.md",
        "receipt": f"{base}/opening_receipt_v1.json",
        "receipt_sha256": f"{base}/opening_receipt_v1.sha256",
    }
    authorization_path = root / "data_usgs/confirmatory_opening_authorization_v1.json"
    fixed_binding = _binding(verifier, root, "src/thermoroute/opening.py")
    provenance_binding = _binding(
        verifier, root, "src/thermoroute/provenance.py"
    )
    usgs_binding = _binding(verifier, root, "src/thermoroute/usgs.py")
    scorer_binding = _binding(verifier, root, "scripts/route_a_trusted_scorer.py")
    authorization = {
        "format": verifier.AUTHORIZATION_FORMAT,
        "status": "AUTHORIZED_LABELS_STILL_SEALED",
        "protocol": {
            **_binding(
                verifier, root, "protocols/route_a_confirmatory_v1.json"
            ),
            "seal": _binding(verifier, root, verifier.PROTOCOL_SEAL_PATH),
            "final_prelabel_commit": "2" * 40,
            "authoritative_commit": "1" * 40,
            "authoritative_markdown_sha256": "3" * 64,
        },
        "registries": {
            "development": _binding(verifier, root, "data_usgs/station_registry_v1.csv"),
            "external": _binding(verifier, root, "data_usgs/external.csv"),
            "external_lock": _binding(verifier, root, "data_usgs/external.lock.json"),
            "development_panel_spec": _binding(
                verifier, root, "data_usgs/frozen_panel_v1.json"
            ),
            "candidate_table": _binding(verifier, root, "data_usgs/candidates.csv"),
            "candidate_provenance": _binding(
                verifier, root, "data_usgs/candidates.provenance.json"
            ),
            "candidate_snapshot_index": _binding(
                verifier, root, "data_usgs/candidate-raw/snapshot_index.json"
            ),
        },
        "model_suite": _binding(
            verifier, root, "data_usgs/confirmatory_model_suite_v1.json"
        ),
        "development_replay": _binding(
            verifier, root, "outputs/model_replay/route_a_development_replay_v1.json"
        ),
        "prelabel_chronology": {
            **_binding(verifier, root, verifier.CHRONOLOGY_PATH),
            "format": verifier.CHRONOLOGY_FORMAT,
            "status": verifier.CHRONOLOGY_STATUS,
            "order": chronology_order,
            "evidence_scope": verifier.CHRONOLOGY_EVIDENCE_SCOPE,
        },
        "inference_amendment": {
            **_binding(verifier, root, amendment_path),
            "format": amendment["format"],
            "amendment_id": amendment["amendment_id"],
            "seal": _binding(verifier, root, amendment_seal_path),
            "final_prelabel_commit": amendment_commit,
        },
        "probability_metric_erratum": {
            **_binding(
                verifier, root, verifier.PROBABILITY_METRIC_ERRATUM_PATH
            ),
            "format": erratum["format"],
            "erratum_id": erratum["erratum_id"],
            "seal": _binding(
                verifier,
                root,
                verifier.PROBABILITY_METRIC_ERRATUM_SEAL_PATH,
            ),
            "erratum_document_commit": erratum_seal[
                "erratum_document_commit"
            ],
        },
        "model_matrix_amendment": {
            **_binding(
                verifier, root, verifier.MODEL_MATRIX_AMENDMENT_PATH
            ),
            "format": model_matrix_amendment["format"],
            "status": model_matrix_amendment["status"],
            "amendment_id": model_matrix_amendment["amendment_id"],
            "seal": _binding(
                verifier, root, verifier.MODEL_MATRIX_AMENDMENT_SEAL_PATH
            ),
            "amendment_document_commit": model_matrix_amendment_seal[
                "amendment_document_commit"
            ],
        },
        "inference_gate": {
            **_binding(verifier, root, gate_path),
            "format": inference_gate["format"],
            "status": inference_gate["status"],
            "claim_eligible": inference_gate["claim_eligible"],
            "analysis_mode": inference_gate["analysis_mode"],
            "policy_sha256": inference_gate["policy_sha256"],
        },
        "outcome_qc_policy": {
            **_binding(verifier, root, outcome_qc_policy_path),
            "format": outcome_qc_policy["format"],
            "policy_id": outcome_qc_policy["policy_id"],
            "required": True,
        },
        "temporal_coverage_policy": {
            **_binding(verifier, root, coverage_policy_path),
            "format": coverage_policy["format"],
            "policy_id": coverage_policy["policy_id"],
            "status": coverage_policy["status"],
            "required": True,
        },
        "acquisition_plan": {
            "history_start": "2020-11-30",
            "target_start": "2021-01-01",
            "target_end": "2023-12-31",
            "nwis_parameter_codes": ["00010", "00060", "00065"],
            "nwis_statistic_code": "00003",
            "request_partition": "one frozen site_no for the complete interval",
            "no_outcome_based_site_replacement": True,
            "provider": verifier.CONFIRMATORY_NWIS_PROVIDER,
            "canonical_endpoint": "https://waterservices.usgs.gov/nwis/dv/",
            "transport": "LIVE_HTTPS_ONLY_NO_PRESEEDED_OUTCOMES",
            "maximum_response_bytes_per_request": (
                verifier.MAX_CONFIRMATORY_NWIS_RESPONSE_BYTES
            ),
        },
        "actual_inputs": _binding(
            verifier, root, "data_usgs/confirmatory_actual_inputs_v1.json"
        ),
        "actual_feature_order": [
            "WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP"
        ],
        "statistics_contract_sha256": verifier._sha256_json(
            json.loads(
                (root / "protocols/route_a_confirmatory_v1.json").read_text(
                    encoding="utf-8"
                )
            )["primary_inference_contract"]
        ),
        "runtime": {
            "format": "thermoroute.route-a-runtime.v1",
            "requirements_lock": _binding(verifier, root, "requirements-lock.txt"),
            "hashed_requirements_lock": _binding(
                verifier, root, verifier.REPRODUCIBILITY_LOCK
            ),
            "installed_version_validation": "fixture validation",
            "installed_versions": {"fixture": "1"},
            "numerical_runtime_contract": {"fixture": True},
            "runtime_sha256": runtime_sha256,
            "python_executable": {
                "invoked_path": "/fixture/python",
                "realpath": "/fixture/python-real",
                "sha256": "d" * 64,
            },
            "golden_inference_sha256": "e" * 64,
            "formal_numerical_policy": {"status": "fixture"},
            "deterministic_child_policy": {"device": "cpu"},
        },
        "required_models": {
            "temporal": [
                "Persistence",
                "DampedPersistence",
                "Climatology",
                "LightGBM",
                "LSTM",
                "ThermoRoute",
                "DampedPriorOnly",
                "TR-noDynamicPrior",
                "TR-fixedKappa",
                "TR-noRouter",
                "TR-noMoE",
                "TR-noTCN",
                "TR-unbounded",
            ],
            "external": [
                "Persistence",
                "DampedPersistence",
                "Climatology",
                "LightGBM",
                "LSTM",
                "ThermoRoute",
            ],
        },
        "fixed_code": {
            "modules": {
                "opening": fixed_binding,
                "thermoroute.provenance": provenance_binding,
                "thermoroute.usgs": usgs_binding,
            },
            "files": {"opening": fixed_binding},
            "entrypoints": {"trusted_scorer": scorer_binding},
            "sha256": "9" * 64,
        },
        "source": {
            "authorization_path": "data_usgs/confirmatory_opening_authorization_v1.json",
            "git_commit_before_authorization": "0" * 40,
            "source_tree_sha256": source_tree_sha256,
            "source_inventory": source_inventory,
        },
        "state_paths": state,
    }
    # Keep the production-like fixture on the same content-addressed namespace
    # derivation as the producer.  A memorable placeholder here used to let
    # tests exercise downstream artifacts without proving the authorization's
    # namespace binding.
    authorization["state_paths"] = verifier._expected_authorization_state_paths(
        authorization
    )
    state = authorization["state_paths"]
    authorization["opening_id"] = verifier._sha256_json(authorization)[:24]
    authorization["created_at_utc"] = "2026-01-01T00:00:00+00:00"
    authorization["authorization_self_sha256"] = verifier._sha256_json(authorization)
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
    authorization_sha = verifier.sha256_file(authorization_path)
    work_order_stable = {
        "format": verifier.ACQUISITION_WORK_ORDER_FORMAT,
        "opening_id": authorization["opening_id"],
        "authorization_path": authorization["source"]["authorization_path"],
        "authorization_sha256": authorization_sha,
        "source_tree_sha256": authorization["source"]["source_tree_sha256"],
        "runtime_sha256": authorization["runtime"]["runtime_sha256"],
        "fixed_code_sha256": authorization["fixed_code"]["sha256"],
        "model_matrix_amendment_seal_sha256": authorization[
            "model_matrix_amendment"
        ]["seal"]["sha256"],
        "acquisition_plan": authorization["acquisition_plan"],
        "state_paths": state,
        "site_registries": {
            "temporal": {
                "sha256": authorization["registries"]["development"]["sha256"],
                "sites": ["01073319"],
            },
            "external": {
                "sha256": authorization["registries"]["external"]["sha256"],
                "sites": ["02000001"],
            },
        },
    }
    work_order = {
        **work_order_stable,
        "work_order_self_sha256": verifier._sha256_json(work_order_stable),
    }
    _write_canonical_json(verifier, root, state["work_order"], work_order)

    validator_files = {
        relative: verifier.sha256_file(root / relative)
        for relative in verifier.TRUSTED_VALIDATOR_PATHS
    }
    trusted_validator = {
        "implementation": verifier.TRUSTED_VALIDATOR_IMPLEMENTATION,
        "files": validator_files,
        "sha256": verifier._sha256_json(validator_files),
        "source_tree_sha256": source_tree_sha256,
    }
    preflight = {
        "authorization_sha256": authorization_sha,
        "opening_id": authorization["opening_id"],
        "protocol_sha256": authorization["protocol"]["sha256"],
        "development_registry_sha256": authorization["registries"][
            "development"
        ]["sha256"],
        "external_registry_sha256": authorization["registries"]["external"][
            "sha256"
        ],
        "external_lock_sha256": authorization["registries"]["external_lock"][
            "sha256"
        ],
        "model_suite_sha256": authorization["model_suite"]["sha256"],
        "development_replay_sha256": authorization["development_replay"][
            "sha256"
        ],
        "prelabel_chronology_sha256": authorization["prelabel_chronology"][
            "sha256"
        ],
        "inference_amendment_sha256": authorization["inference_amendment"][
            "sha256"
        ],
        "inference_amendment_seal_sha256": authorization[
            "inference_amendment"
        ]["seal"]["sha256"],
        "model_matrix_amendment_sha256": authorization[
            "model_matrix_amendment"
        ]["sha256"],
        "model_matrix_amendment_seal_sha256": authorization[
            "model_matrix_amendment"
        ]["seal"]["sha256"],
        "model_matrix_amendment_id": verifier.MODEL_MATRIX_AMENDMENT_ID,
        "model_matrix_amendment_status": (
            verifier.MODEL_MATRIX_AMENDMENT_STATUS
        ),
        "inference_gate_sha256": authorization["inference_gate"]["sha256"],
        "inference_gate_status": authorization["inference_gate"]["status"],
        "inference_claim_eligible": authorization["inference_gate"][
            "claim_eligible"
        ],
        "outcome_qc_policy_sha256": authorization["outcome_qc_policy"][
            "sha256"
        ],
        "temporal_coverage_policy_sha256": authorization[
            "temporal_coverage_policy"
        ]["sha256"],
        "prelabel_inputs_sha256": authorization["actual_inputs"]["sha256"],
        "actual_feature_order": list(authorization["actual_feature_order"]),
        "required_models": {
            cohort: list(values)
            for cohort, values in authorization["required_models"].items()
        },
        "source_tree_sha256": authorization["source"]["source_tree_sha256"],
        "runtime_sha256": authorization["runtime"]["runtime_sha256"],
        "requirements_lock_sha256": authorization["runtime"][
            "requirements_lock"
        ]["sha256"],
        "hashed_requirements_lock_sha256": authorization["runtime"][
            "hashed_requirements_lock"
        ]["sha256"],
        "golden_inference_sha256": authorization["runtime"][
            "golden_inference_sha256"
        ],
        "fixed_code_sha256": authorization["fixed_code"]["sha256"],
        "state_namespace": state["namespace"],
    }
    assert set(preflight) == set(verifier.RAW_PREFLIGHT_ATTESTATION_FIELDS)
    assert set(trusted_validator) == set(verifier.TRUSTED_VALIDATOR_FIELDS)
    intent = {
        "format": verifier.INTENT_FORMAT,
        "status": "OPENING_STARTED_IRREVERSIBLE",
        "opening_id": authorization["opening_id"],
        "authorization_sha256": authorization_sha,
        "preflight_attestation_sha256": verifier._sha256_json(preflight),
        "work_order_self_sha256": work_order["work_order_self_sha256"],
        "work_order_file_sha256": verifier.sha256_file(root / state["work_order"]),
        "fixed_code_sha256": authorization["fixed_code"]["sha256"],
        "runtime_sha256": authorization["runtime"]["runtime_sha256"],
        "model_matrix_amendment_seal_sha256": authorization[
            "model_matrix_amendment"
        ]["seal"]["sha256"],
        "maximum_openings": 1,
        "retry_after_failure_allowed": False,
        "same_opening_transport_resume_allowed": True,
        "trusted_validator": trusted_validator,
        "started_at_utc": "2026-01-01T00:00:00+00:00",
    }
    intent["intent_self_sha256"] = verifier._sha256_json(intent)
    _write_canonical_json(verifier, root, state["intent"], intent)

    request_ledger = f"{state['transport_root']}/request_ledger_v1.json"
    attempt_index = f"{state['transport_root']}/transport_attempt_index_v1.json"
    attempts_root = f"{state['transport_root']}/transport_attempts_v1"
    raw_index = state["raw_nwis_snapshot_index"]
    request_specs = []
    for ordinal, (cohort, site) in enumerate(
        (("temporal", "01073319"), ("external", "02000001")), start=1
    ):
        request = {
            "schema_version": 1,
            "provider": verifier.CONFIRMATORY_NWIS_PROVIDER,
            "method": "GET",
            "url": verifier._expected_confirmatory_nwis_url(
                site,
                authorization["acquisition_plan"]["history_start"],
                authorization["acquisition_plan"]["target_end"],
            ),
            "headers": {},
        }
        request_specs.append({
            "ordinal": ordinal,
            "cohort": cohort,
            "site_no": site,
            "request": request,
            "request_sha256": hashlib.sha256(
                verifier._canonical_json_bytes(request)
            ).hexdigest(),
        })
    request_ids = [row["request_sha256"] for row in request_specs]
    temporal_request, external_request = request_ids
    ledger_stable = {
        "format": verifier.ACQUISITION_REQUEST_LEDGER_FORMAT,
        "status": "FROZEN_BEFORE_FIRST_HTTPS_REQUEST",
        "opening_id": authorization["opening_id"],
        "authorization_sha256": authorization_sha,
        "work_order_self_sha256": work_order["work_order_self_sha256"],
        "work_order_file_sha256": verifier.sha256_file(root / state["work_order"]),
        "provider": verifier.CONFIRMATORY_NWIS_PROVIDER,
        "maximum_response_bytes_per_request": (
            verifier.MAX_CONFIRMATORY_NWIS_RESPONSE_BYTES
        ),
        "request_order": "temporal_then_external_each_site_no_ascending",
        "request_count": len(request_specs),
        "requests": request_specs,
        "station_or_request_replacement_allowed": False,
    }
    ledger = {
        **ledger_stable,
        "request_ledger_self_sha256": hashlib.sha256(
            verifier._canonical_json_bytes(ledger_stable)
        ).hexdigest(),
    }
    _write_canonical_json(verifier, root, request_ledger, ledger)
    ledger_sha256 = verifier.sha256_file(root / request_ledger)

    attempt_partitions = [
        {
            "completed_before": [],
            "missing_before": sorted(request_ids),
            "completed_after": [temporal_request],
            "missing_after": [external_request],
            "status": "TRANSPORT_INCOMPLETE_MAY_RESUME_SAME_OPENING",
            "failure_class": "FIXTURE_PROCESS_INTERRUPTION",
            "started_at": "2026-01-01T00:00:00+00:00",
            "completed_at": "2026-01-01T00:11:00+00:00",
        },
        {
            "completed_before": [temporal_request],
            "missing_before": [external_request],
            "completed_after": sorted(request_ids),
            "missing_after": [],
            "status": "ALL_LEDGER_TRANSACTIONS_COMPLETE",
            "failure_class": None,
            "started_at": "2026-01-01T00:12:00+00:00",
            "completed_at": "2026-01-01T00:21:00+00:00",
        },
    ]
    attempt_rows = []
    for number, partition in enumerate(attempt_partitions, start=1):
        mode = (
            "INITIAL_OPENING_TRANSPORT"
            if number == 1
            else "RESUME_SAME_OPENING"
        )
        start_stable = {
            "format": verifier.ACQUISITION_ATTEMPT_START_FORMAT,
            "status": "TRANSPORT_ATTEMPT_STARTED",
            "opening_id": authorization["opening_id"],
            "authorization_sha256": authorization_sha,
            "work_order_self_sha256": work_order["work_order_self_sha256"],
            "request_ledger_sha256": ledger_sha256,
            "attempt_number": number,
            "mode": mode,
            "opening_count": 1,
            "completed_before_attempt_request_sha256": sorted(
                partition["completed_before"]
            ),
            "missing_at_start_request_sha256": sorted(
                partition["missing_before"]
            ),
            "response_replacement_allowed": False,
            "started_at_utc": partition["started_at"],
        }
        start = {
            **start_stable,
            "attempt_start_self_sha256": hashlib.sha256(
                verifier._canonical_json_bytes(start_stable)
            ).hexdigest(),
        }
        start_relative = f"{attempts_root}/attempt_{number:06d}_start.json"
        _write_canonical_json(verifier, root, start_relative, start)
        result_stable = {
            "format": verifier.ACQUISITION_ATTEMPT_RESULT_FORMAT,
            "status": partition["status"],
            "opening_id": authorization["opening_id"],
            "authorization_sha256": authorization_sha,
            "work_order_self_sha256": work_order["work_order_self_sha256"],
            "request_ledger_sha256": ledger_sha256,
            "attempt_number": number,
            "attempt_start_sha256": verifier.sha256_file(root / start_relative),
            "opening_count": 1,
            "completed_request_sha256": sorted(partition["completed_after"]),
            "missing_request_sha256": sorted(partition["missing_after"]),
            "failure_class": partition["failure_class"],
            "response_replacement_count": 0,
            "completed_at_utc": partition["completed_at"],
        }
        result = {
            **result_stable,
            "attempt_result_self_sha256": hashlib.sha256(
                verifier._canonical_json_bytes(result_stable)
            ).hexdigest(),
        }
        result_relative = f"{attempts_root}/attempt_{number:06d}_result.json"
        _write_canonical_json(verifier, root, result_relative, result)
        attempt_rows.append({
            "attempt_number": number,
            "mode": mode,
            "status": partition["status"],
            "start": _binding(verifier, root, start_relative),
            "result": _binding(verifier, root, result_relative),
        })

    snapshot_records = []
    request_map_rows = []
    parsed_outcomes: dict[str, pd.DataFrame] = {}
    outcome_dates = pd.date_range("2020-11-30", "2023-12-31", freq="D")
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from thermoroute.usgs import parse_nwis_confirmatory_daily
    finally:
        sys.path.pop(0)
    retrieval_times = {
        temporal_request: "2026-01-01T00:10:00+00:00",
        external_request: "2026-01-01T00:20:00+00:00",
    }
    request_attempts = {temporal_request: 1, external_request: 2}
    for spec in request_specs:
        request_sha = spec["request_sha256"]
        transaction_root = (
            f"{state['raw_nwis_root']}/{verifier.CONFIRMATORY_NWIS_PROVIDER}/"
            f"{request_sha}"
        )
        response_relative = f"{transaction_root}/response.bin"
        metadata_relative = f"{transaction_root}/metadata.json"
        payload_lines = [
            f"# fixture NWIS response for {spec['site_no']}",
            "agency_cd\tsite_no\tdatetime\t00010_00003\t00010_00003_cd",
            "5s\t15s\t20d\t14n\t10s",
            *(
                "\t".join((
                    "USGS",
                    str(spec["site_no"]),
                    date.strftime("%Y-%m-%d"),
                    f"{10.0 + date.dayofyear / 1000.0:.3f}",
                    "A",
                ))
                for date in outcome_dates
            ),
        ]
        payload = ("\n".join(payload_lines) + "\n").encode("utf-8")
        series_registry = verifier._nwis_series_registry_from_payload(payload)
        parsed_outcomes[str(spec["site_no"])] = parse_nwis_confirmatory_daily(
            payload,
            site_no=str(spec["site_no"]),
            start=authorization["acquisition_plan"]["history_start"],
            end=authorization["acquisition_plan"]["target_end"],
        )
        _write_bytes(root, response_relative, payload)
        metadata = {
            "schema_version": 1,
            "opening_id": authorization["opening_id"],
            "authorization_sha256": authorization_sha,
            "work_order_self_sha256": work_order["work_order_self_sha256"],
            "request_ledger_sha256": ledger_sha256,
            "attempt_number": request_attempts[request_sha],
            "request": spec["request"],
            "request_sha256": request_sha,
            "retrieved_at_utc": retrieval_times[request_sha],
            "http_status": 200,
            "response_headers": {"Content-Type": "text/plain"},
            "final_url": spec["request"]["url"],
            "byte_count": len(payload),
            "response_sha256": hashlib.sha256(payload).hexdigest(),
            "response_file": "response.bin",
            "maximum_response_bytes_per_request": (
                verifier.MAX_CONFIRMATORY_NWIS_RESPONSE_BYTES
            ),
        }
        _write_canonical_json(verifier, root, metadata_relative, metadata)
        record = {
            "provider": verifier.CONFIRMATORY_NWIS_PROVIDER,
            "request_sha256": request_sha,
            "response_sha256": metadata["response_sha256"],
            "retrieved_at_utc": metadata["retrieved_at_utc"],
            "byte_count": metadata["byte_count"],
            "attempt_number": metadata["attempt_number"],
            "request": spec["request"],
            "metadata_path": (
                f"{verifier.CONFIRMATORY_NWIS_PROVIDER}/{request_sha}/metadata.json"
            ),
            "metadata_sha256": verifier.sha256_file(root / metadata_relative),
            "response_path": (
                f"{verifier.CONFIRMATORY_NWIS_PROVIDER}/{request_sha}/response.bin"
            ),
            "series_registry": series_registry,
        }
        snapshot_records.append(record)
        request_map_rows.append({
            "cohort": spec["cohort"],
            "site_no": spec["site_no"],
            "request_sha256": request_sha,
            "response_sha256": record["response_sha256"],
            "retrieved_at_utc": record["retrieved_at_utc"],
            "byte_count": record["byte_count"],
            "attempt_number": record["attempt_number"],
            "series_registry": series_registry,
        })
    # The live transport producer keeps this staging directory after atomically
    # publishing transactions.  It is intentionally empty and may disappear
    # when the release materializer copies only evidence files.
    (
        root
        / state["raw_nwis_root"]
        / verifier.CONFIRMATORY_NWIS_PROVIDER
        / ".pending"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )
    snapshot_records.sort(key=lambda row: row["request_sha256"])
    _write_canonical_json(verifier, root, raw_index, {
        "schema_version": 1,
        "snapshot_count": len(snapshot_records),
        "records": snapshot_records,
    })
    request_map_rows.sort(key=lambda row: (row["cohort"], row["site_no"]))
    _write_canonical_json(verifier, root, state["acquisition_request_map"], {
        "format": verifier.ACQUISITION_REQUEST_MAP_FORMAT,
        "opening_id": authorization["opening_id"],
        "authorization_sha256": authorization_sha,
        "provider": verifier.CONFIRMATORY_NWIS_PROVIDER,
        "request_count": len(request_map_rows),
        "requests": request_map_rows,
    })
    attempt_index_stable = {
        "format": verifier.ACQUISITION_ATTEMPT_INDEX_FORMAT,
        "status": "ALL_LEDGER_TRANSACTIONS_COMPLETE",
        "opening_id": authorization["opening_id"],
        "authorization_sha256": authorization_sha,
        "work_order_self_sha256": work_order["work_order_self_sha256"],
        "request_ledger": _binding(verifier, root, request_ledger),
        "request_count": len(request_ids),
        "attempt_count": len(attempt_rows),
        "resume_count": 1,
        "opening_count": 1,
        "response_replacement_count": 0,
        "completed_before_final_attempt_request_sha256": [temporal_request],
        "retrieval_span_utc": {
            "first": retrieval_times[temporal_request],
            "last": retrieval_times[external_request],
        },
        "attempts": attempt_rows,
    }
    attempt_index_document = {
        **attempt_index_stable,
        "attempt_index_self_sha256": hashlib.sha256(
            verifier._canonical_json_bytes(attempt_index_stable)
        ).hexdigest(),
    }
    _write_canonical_json(
        verifier, root, attempt_index, attempt_index_document
    )
    for cohort, site in (
        ("temporal", "01073319"),
        ("external", "02000001"),
    ):
        outcome_path = root / state[f"{cohort}_outcomes"]
        outcome_path.parent.mkdir(parents=True, exist_ok=True)
        parsed_outcomes[site].to_parquet(outcome_path, index=False)
    acquisition = {
        "format": verifier.ACQUISITION_MANIFEST_FORMAT,
        "opening_id": authorization["opening_id"],
        "authorization_sha256": authorization_sha,
        "protocol_sha256": authorization["protocol"]["sha256"],
        "labels_state": "OPENED_ONCE",
        "site_replacement_count": 0,
        "response_replacement_count": 0,
        "history_start": authorization["acquisition_plan"]["history_start"],
        "target_start": authorization["acquisition_plan"]["target_start"],
        "target_end": authorization["acquisition_plan"]["target_end"],
        "maximum_response_bytes_per_request": (
            verifier.MAX_CONFIRMATORY_NWIS_RESPONSE_BYTES
        ),
        "transport_summary": {
            "opening_count": 1,
            "attempt_count": 2,
            "resume_count": 1,
            "completed_before_final_attempt_request_sha256": [temporal_request],
            "retrieval_span_utc": {
                "first": retrieval_times[temporal_request],
                "last": retrieval_times[external_request],
            },
        },
        "request_ledger": _binding(verifier, root, request_ledger),
        "transport_attempt_index": _binding(verifier, root, attempt_index),
        "raw_nwis_snapshot_index": _binding(verifier, root, raw_index),
        "request_map": _binding(verifier, root, state["acquisition_request_map"]),
        "normalized_outcome_tables": {
            "temporal": _binding(verifier, root, state["temporal_outcomes"]),
            "external": _binding(verifier, root, state["external_outcomes"]),
        },
        "producer_role": "RAW_ONLY_NO_PREDICTIONS_OR_STATISTICS",
    }
    _write_canonical_json(
        verifier, root, state["acquisition_manifest"], acquisition
    )
    availability_rows: list[dict[str, object]] = []
    trusted_probability_frames: dict[str, pd.DataFrame] = {}
    fixture_model_metadata = _fixture_model_metadata_registry(
        root, model_entries
    )
    for cohort, site in (
        ("temporal", "01073319"),
        ("external", "02000001"),
    ):
        rows: list[dict[str, object]] = []
        for model_index, model in enumerate(
            authorization["required_models"][cohort]
        ):
            for horizon in (1, 3, 7):
                issue_dates = pd.date_range(
                    "2021-01-01",
                    pd.Timestamp("2023-12-31") - pd.Timedelta(days=horizon),
                    freq="D",
                )
                for issue in issue_dates:
                    target = issue + pd.Timedelta(days=horizon)
                    truth = 10.0 + target.dayofyear / 1000.0
                    builtin = model in verifier.PROBABILISTIC_BUILTIN_MODELS
                    delta = (
                        0.0
                        if builtin
                        else float(
                            fixture_model_metadata[(cohort, model)][
                                "conformal_offsets"
                            ][
                                f"{'__pooled__' if cohort == 'external' else site}|"
                                f"{horizon}"
                            ]
                        )
                    )
                    rows.append(
                        {
                            "model": model,
                            "scope": (
                                "route_a_temporal_confirmation"
                                if cohort == "temporal"
                                else "route_a_external_history_dependent_new_gage"
                            ),
                            "feature_set": (
                                "WTEMP+FLOW+TEMP+PRCP+RHMEAN+DH+WDSP"
                            ),
                            "seed": -1,
                            "site_id": site,
                            "horizon": horizon,
                            "split": "confirm",
                            "issue_date": issue,
                            "target_date": target,
                            "y_true": float(truth),
                            "y_pred": float(truth + (model_index + 1) / 100.0),
                            "q05": (
                                np.nan if builtin else float(truth - 1.0 - delta)
                            ),
                            "q50": np.nan if builtin else float(truth),
                            "q95": (
                                np.nan if builtin else float(truth + 1.0 + delta)
                            ),
                            "p_exceed": np.nan if builtin else 0.5,
                        }
                    )
        frame = pd.DataFrame.from_records(rows).loc[
            :, list(verifier.DEVELOPMENT_REPLAY_PREDICTION_COLUMNS)
        ].sort_values(
            ["model", "site_id", "horizon", "issue_date", "target_date"],
            kind="mergesort",
        ).reset_index(drop=True)
        prediction_path = root / state[f"{cohort}_predictions"]
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(prediction_path, index=False)
        trusted_probability_frames[cohort] = frame
        for horizon in (1, 3, 7):
            count = len(
                pd.date_range(
                    "2021-01-01",
                    pd.Timestamp("2023-12-31") - pd.Timedelta(days=horizon),
                    freq="D",
                )
            )
            availability_rows.append(
                {
                    "cohort": cohort,
                    "site_no": site,
                    "horizon": horizon,
                    "n_valid_targets": count,
                    "reportable": True,
                }
            )
    availability_path = root / state["availability_registry"]
    availability_path.parent.mkdir(parents=True, exist_ok=True)
    availability_frame = pd.DataFrame.from_records(availability_rows)
    availability_frame.to_csv(
        availability_path, index=False, lineterminator="\n"
    )
    _write_bytes(root, state["outcome_quality_audit"], b"{}\n")
    prediction_rows = []
    models_by_horizon = {
        1: ("ThermoRoute", "DampedPersistence"),
        3: ("ThermoRoute", "DampedPersistence", "LightGBM"),
        7: ("ThermoRoute", "DampedPersistence", "LightGBM"),
    }
    for horizon, models in models_by_horizon.items():
        # One more than the 100-row reportability floor so deleting the single
        # most influential row remains estimable in the outcome-QC fixture.
        for offset in range(101):
            issue = pd.Timestamp("2021-02-01") + pd.Timedelta(days=offset)
            for model in models:
                prediction_rows.append({
                    "model": model,
                    "site_id": "01073319",
                    "horizon": horizon,
                    "issue_date": issue,
                    "target_date": issue + pd.Timedelta(days=horizon),
                    "y_true": 10.0,
                    "y_pred": 10.0 if model == "ThermoRoute" else 11.0,
                })
    outcome_predictions = pd.DataFrame.from_records(prediction_rows)
    normalized_temporal = pd.DataFrame({
        "site_no": ["01073319"] * 108,
        "DATE": pd.date_range("2021-02-01", periods=108, freq="D"),
        "WTEMP": np.linspace(10.0, 12.0, 108),
    })
    spatial_sensitivity = {
        "comparisons": [
            {
                "test_id": row["test_id"],
                "station_weighted_median_effect_c": -1.0,
                "leave_one_huc": [{
                    "held_out_huc2": "01",
                    "effect_minus_margin_c": -1.0 - float(row["margin_c"]),
                }],
            }
            for row in family
        ]
    }
    outcome_qc_module = verifier._load_canonical_outcome_qc_module(root)
    outcome_qc_gate = outcome_qc_module.build_outcome_qc_gate_document(
        root=root,
        policy_path=root / outcome_qc_policy_path,
        protocol=json.loads(
            (root / "protocols/route_a_confirmatory_v1.json").read_text(
                encoding="utf-8"
            )
        ),
        temporal_predictions=outcome_predictions,
        normalized_temporal=normalized_temporal,
        spatial_sensitivity=spatial_sensitivity,
        minimum_targets=100,
    )
    _write_bytes(
        root, state["outcome_qc_gate"], json.dumps(outcome_qc_gate).encode()
    )
    _write_bytes(root, state["approved_target_sensitivity"], b"{}\n")
    _write_bytes(root, state["spatial_sensitivity"], b"{}\n")
    probabilistic_evaluation = _fixture_probabilistic_evaluation_v2(
        verifier,
        protocol=json.loads(
            (root / "protocols/route_a_confirmatory_v1.json").read_text(
                encoding="utf-8"
            )
        ),
        authorization=authorization,
        frames=trusted_probability_frames,
        availability=availability_frame,
        model_metadata=fixture_model_metadata,
        required_models=authorization["required_models"],
    )
    _write_canonical_json(
        verifier,
        root,
        state["probabilistic_evaluation"],
        probabilistic_evaluation,
    )
    tests = [
        {
            "test_id": row["test_id"],
            "status": "NOT_ESTIMABLE_INSUFFICIENT_STATIONS_OR_CLUSTERS",
            "median_effect_c": None,
            "n_stations": 1,
            "n_clusters": 1,
            "reject_at_0_05": False,
            "confidence_bound_supports_margin": False,
        }
        for row in family
    ]
    (root / state["statistics"]).write_text(json.dumps({
        "format": verifier.STATISTICS_FORMAT,
        "tests": tests,
        "outcome_qc_gate": {
            **_binding(verifier, root, state["outcome_qc_gate"]),
            "format": outcome_qc_gate["format"],
            "status": outcome_qc_gate["status"],
            "pass": True,
            "directional_claims_allowed": True,
        },
    }), encoding="utf-8")
    coverage_module = verifier._load_canonical_coverage_bridge_module(root)
    temporal_coverage_audit = (
        coverage_module.replay_temporal_coverage_from_physical_files(
            root=root,
            authorization=authorization,
        )
    )
    _write_bytes(
        root,
        state["temporal_coverage_audit"],
        json.dumps(
            temporal_coverage_audit,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n",
    )
    _write_bytes(root, state["report"], b"# trusted report\n")
    receipt_artifacts = {
        "acquisition_manifest": _binding(verifier, root, state["acquisition_manifest"]),
        "raw_nwis_snapshot_index": _binding(verifier, root, raw_index),
        "acquisition_request_map": _binding(verifier, root, state["acquisition_request_map"]),
        "temporal_normalized_outcomes": _binding(verifier, root, state["temporal_outcomes"]),
        "external_normalized_outcomes": _binding(verifier, root, state["external_outcomes"]),
        "availability_registry": _binding(verifier, root, state["availability_registry"]),
        "outcome_quality_audit": _binding(verifier, root, state["outcome_quality_audit"]),
        "outcome_qc_gate": _binding(verifier, root, state["outcome_qc_gate"]),
        "approved_target_sensitivity": _binding(
            verifier, root, state["approved_target_sensitivity"]
        ),
        "spatial_sensitivity": _binding(verifier, root, state["spatial_sensitivity"]),
        "probabilistic_evaluation": _binding(
            verifier, root, state["probabilistic_evaluation"]
        ),
        "temporal_predictions": _binding(verifier, root, state["temporal_predictions"]),
        "external_predictions": _binding(verifier, root, state["external_predictions"]),
        "statistics": _binding(verifier, root, state["statistics"]),
        "temporal_coverage_audit": _binding(
            verifier, root, state["temporal_coverage_audit"]
        ),
        "report": _binding(verifier, root, state["report"]),
    }
    release_artifacts = {
        key: {
            "format": (
                verifier.TEMPORAL_COVERAGE_AUDIT_FORMAT
                if key == "temporal_coverage_audit"
                else "fixture-format"
            ),
            **binding,
        }
        for key, binding in receipt_artifacts.items()
    }
    release_bindings = {
        "format": "thermoroute.route-a-release-bindings.v1",
        "opening_id": authorization["opening_id"],
        "state_namespace": state["namespace"],
        "authorization": {
            "format": verifier.AUTHORIZATION_FORMAT,
            "path": authorization["source"]["authorization_path"],
            "sha256": authorization_sha,
        },
        "artifacts": release_artifacts,
        "receipt": {
            "format": verifier.RECEIPT_FORMAT,
            "path": state["receipt"],
            "external_sha256_path": state["receipt_sha256"],
        },
    }
    receipt = {
        "format": verifier.RECEIPT_FORMAT,
        "status": "OPENED_AND_SCORED_ONCE",
        "opening_id": authorization["opening_id"],
        "authorization_sha256": authorization_sha,
        "model_matrix_amendment_seal_sha256": authorization[
            "model_matrix_amendment"
        ]["seal"]["sha256"],
        "intent_sha256": verifier.sha256_file(root / state["intent"]),
        "work_order_sha256": verifier.sha256_file(root / state["work_order"]),
        "preflight_attestation": preflight,
        "preflight_attestation_sha256": verifier._sha256_json(preflight),
        "fixed_code": authorization["fixed_code"],
        "authorized_runtime": authorization["runtime"],
        "completion_environment": {
            "numerical_runtime_sha256": authorization["runtime"]["runtime_sha256"]
        },
        "python_hash_seed_interpreter_effect": (
            "present_but_ignored_under_isolated_mode"
        ),
        "completed_at_utc": "2026-01-01T00:30:00+00:00",
        "opening_count": 1,
        "maximum_openings": 1,
        "retry_after_failure_allowed": False,
        "same_opening_transport_resume_allowed": True,
        "transport_recovery": acquisition["transport_summary"],
        "all_predeclared_models_reported": True,
        "reported_models": {
            cohort: sorted(values)
            for cohort, values in authorization["required_models"].items()
        },
        "trusted_validator": trusted_validator,
        "artifacts": receipt_artifacts,
        "trusted_prediction_hashes": {
            cohort: {
                "rows": 1,
                "sha256": digest,
                "schema": "thermoroute.predictions.v1",
                "ensemble_rule": (
                    "mean frozen members, then frozen CQR and horizon Platt "
                    "calibration"
                ),
            }
            for cohort, digest in (
                ("temporal", "a" * 64),
                ("external", "b" * 64),
            )
        },
        "formal_tests": tests,
        "temporal_coverage_audit": {
            **receipt_artifacts["temporal_coverage_audit"],
            "format": verifier.TEMPORAL_COVERAGE_AUDIT_FORMAT,
            "core_status": verifier.TEMPORAL_COVERAGE_CORE_STATUS,
            "physical_replay_verified": True,
            "source_binding_count": 11,
        },
        "state_paths": state,
        "release_bindings": release_bindings,
        "intent_self_sha256": intent["intent_self_sha256"],
        "security_boundary": (
            "misoperation/replay guard for an honest filesystem owner; not a "
            "defense against an owner who can replace the interpreter or files"
        ),
    }
    receipt["receipt_self_sha256"] = verifier._sha256_json(receipt)
    _write_canonical_json(verifier, root, state["receipt"], receipt)
    receipt_sha = verifier.sha256_file(root / state["receipt"])
    _write_bytes(
        root,
        state["receipt_sha256"],
        f"{receipt_sha}  opening_receipt_v1.json\n".encode(),
    )
    temporal_lstm_bundle = next(
        entry["artifact"]["path"]
        for entry in suite["cohorts"]["temporal"]["models"]
        if entry["model_id"] == "LSTM"
    )
    representatives = {
        "canonical_development": "data_usgs/panel_usgs_120v2.parquet",
        "authorization": "protocols/route_a_confirmatory_v1.json",
        "inference_gates": gate_path,
        "registries": "data_usgs/external.csv",
        "candidate_evidence": "data_usgs/candidates.csv",
        "model_suite": "data_usgs/confirmatory_model_suite_v1.json",
        "model_bundles": f"{temporal_lstm_bundle}/weights.pt",
        "prelabel_chronology": verifier.CHRONOLOGY_PATH,
        "prelabel_inputs": "data_usgs/prelabel/temporal.parquet",
        "raw_meteorology": "data_usgs/raw_snapshots/met-0/response.bin",
        "opening_intent": state["intent"],
        "raw_nwis": (
            f"{state['raw_nwis_root']}/{verifier.CONFIRMATORY_NWIS_PROVIDER}/"
            f"{temporal_request}/response.bin"
        ),
        "normalized_outcomes": state["temporal_outcomes"],
        "trusted_predictions": state["temporal_predictions"],
        "availability": state["availability_registry"],
        "sensitivity_audits": state["outcome_quality_audit"],
        "outcome_qc": state["outcome_qc_gate"],
        "probabilistic_evaluation": state["probabilistic_evaluation"],
        "statistics": state["statistics"],
        "temporal_coverage": state["temporal_coverage_audit"],
        "report": state["report"],
        "receipt": state["receipt_sha256"],
        "environment_attestations": "requirements-lock.txt",
        "reproducibility_lock": verifier.REPRODUCIBILITY_LOCK,
    }
    return authorization_path, representatives


def _refresh_postopen_coverage_evidence(
    verifier, root: Path, authorization_path: Path
) -> None:
    """Rebuild the fixture audit after a deliberately self-consistent mutation."""
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    state = authorization["state_paths"]
    coverage_module = verifier._load_canonical_coverage_bridge_module(root)
    audit = coverage_module.replay_temporal_coverage_from_physical_files(
        root=root,
        authorization=authorization,
    )
    audit_path = root / state["temporal_coverage_audit"]
    audit_path.write_text(
        json.dumps(audit, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    receipt_path = root / state["receipt"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    binding = _binding(verifier, root, state["temporal_coverage_audit"])
    receipt["artifacts"]["temporal_coverage_audit"] = binding
    released = receipt["release_bindings"]["artifacts"][
        "temporal_coverage_audit"
    ]
    released.update(binding)
    receipt["temporal_coverage_audit"] = {
        **binding,
        "format": verifier.TEMPORAL_COVERAGE_AUDIT_FORMAT,
        "core_status": verifier.TEMPORAL_COVERAGE_CORE_STATUS,
        "physical_replay_verified": True,
        "source_binding_count": 11,
    }
    receipt.pop("receipt_self_sha256", None)
    receipt["receipt_self_sha256"] = verifier._sha256_json(receipt)
    receipt_path.write_bytes(verifier._canonical_json_bytes(receipt))
    receipt_digest = verifier.sha256_file(receipt_path)
    (root / state["receipt_sha256"]).write_text(
        f"{receipt_digest}  opening_receipt_v1.json\n", encoding="utf-8"
    )


def _refresh_postopen_transport_evidence(
    verifier, root: Path, authorization_path: Path
) -> None:
    """Rebind a deliberately mutated transport chain through its outer receipt."""
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    state = authorization["state_paths"]
    acquisition_path = root / state["acquisition_manifest"]
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))

    ledger_relative = acquisition["request_ledger"]["path"]
    acquisition["request_ledger"] = _binding(verifier, root, ledger_relative)
    for key in ("raw_nwis_snapshot_index", "request_map"):
        relative = acquisition[key]["path"]
        acquisition[key] = _binding(verifier, root, relative)
    index_relative = acquisition["transport_attempt_index"]["path"]
    index_path = root / index_relative
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for row in index.get("attempts", []):
        for key in ("start", "result"):
            binding = row.get(key)
            if isinstance(binding, dict):
                row[key] = _binding(verifier, root, binding["path"])
    index.pop("attempt_index_self_sha256", None)
    index["attempt_index_self_sha256"] = hashlib.sha256(
        verifier._canonical_json_bytes(index)
    ).hexdigest()
    index_path.write_bytes(verifier._canonical_json_bytes(index))
    acquisition["transport_attempt_index"] = _binding(
        verifier, root, index_relative
    )
    for cohort in ("temporal", "external"):
        acquisition["normalized_outcome_tables"][cohort] = _binding(
            verifier, root, state[f"{cohort}_outcomes"]
        )
    acquisition_path.write_bytes(verifier._canonical_json_bytes(acquisition))

    receipt_path = root / state["receipt"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    acquisition_binding = _binding(
        verifier, root, state["acquisition_manifest"]
    )
    receipt["artifacts"]["acquisition_manifest"] = acquisition_binding
    receipt["release_bindings"]["artifacts"]["acquisition_manifest"].update(
        acquisition_binding
    )
    for acquisition_key, artifact_key in (
        ("raw_nwis_snapshot_index", "raw_nwis_snapshot_index"),
        ("request_map", "acquisition_request_map"),
    ):
        binding = dict(acquisition[acquisition_key])
        receipt["artifacts"][artifact_key] = binding
        receipt["release_bindings"]["artifacts"][artifact_key].update(binding)
    for cohort in ("temporal", "external"):
        binding = dict(acquisition["normalized_outcome_tables"][cohort])
        artifact_key = f"{cohort}_normalized_outcomes"
        receipt["artifacts"][artifact_key] = binding
        receipt["release_bindings"]["artifacts"][artifact_key].update(binding)
    receipt["transport_recovery"] = acquisition.get("transport_summary")
    _reseal_postopen_fixture_receipt(verifier, root, state, receipt)
    _refresh_postopen_coverage_evidence(verifier, root, authorization_path)


def _reseal_postopen_fixture_receipt(
    verifier, root: Path, state: dict[str, str], receipt: dict[str, object]
) -> None:
    receipt.pop("receipt_self_sha256", None)
    receipt["receipt_self_sha256"] = verifier._sha256_json(receipt)
    receipt_path = root / state["receipt"]
    receipt_path.write_bytes(verifier._canonical_json_bytes(receipt))
    (root / state["receipt_sha256"]).write_text(
        f"{verifier.sha256_file(receipt_path)}  opening_receipt_v1.json\n",
        encoding="utf-8",
    )


def _refresh_postopen_fixture_artifact_binding(
    verifier,
    root: Path,
    authorization_path: Path,
    artifact_key: str,
) -> None:
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    state = authorization["state_paths"]
    receipt_path = root / state["receipt"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    relative = receipt["artifacts"][artifact_key]["path"]
    binding = _binding(verifier, root, relative)
    receipt["artifacts"][artifact_key] = binding
    receipt["release_bindings"]["artifacts"][artifact_key].update(binding)
    _reseal_postopen_fixture_receipt(verifier, root, state, receipt)


def _validate_fixture_inference_closure(
    verifier, root: Path, authorization_path: Path
) -> dict[str, object]:
    """Rebind a mutated amendment and exercise its independent release check."""
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    amendment_binding = authorization["inference_amendment"]
    seal_path = root / amendment_binding["seal"]["path"]
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["amendment"] = _binding(verifier, root, amendment_binding["path"])
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    amendment_binding.update(_binding(verifier, root, amendment_binding["path"]))
    amendment_binding["seal"] = _binding(
        verifier, root, amendment_binding["seal"]["path"]
    )
    erratum_module = verifier._load_canonical_probability_metric_erratum_module(
        root
    )
    erratum_path = root / verifier.PROBABILITY_METRIC_ERRATUM_PATH
    erratum = erratum_module.expected_probability_metric_erratum_document(
        root=root
    )
    erratum_path.write_text(json.dumps(erratum), encoding="utf-8")
    erratum_seal_path = root / verifier.PROBABILITY_METRIC_ERRATUM_SEAL_PATH
    erratum_seal = json.loads(erratum_seal_path.read_text(encoding="utf-8"))
    erratum_seal["erratum"] = _binding(
        verifier, root, verifier.PROBABILITY_METRIC_ERRATUM_PATH
    )
    erratum_seal["governance_seals"]["inference_amendment_seal"] = (
        _binding(verifier, root, amendment_binding["seal"]["path"])
    )
    erratum_seal_path.write_text(json.dumps(erratum_seal), encoding="utf-8")
    authorization["probability_metric_erratum"].update(
        _binding(verifier, root, verifier.PROBABILITY_METRIC_ERRATUM_PATH)
    )
    authorization["probability_metric_erratum"]["seal"] = _binding(
        verifier, root, verifier.PROBABILITY_METRIC_ERRATUM_SEAL_PATH
    )

    protocol_binding = authorization["protocol"]
    protocol_document = json.loads(
        (root / protocol_binding["path"]).read_text(encoding="utf-8")
    )
    outcome_qc_policy = json.loads(
        (root / authorization["outcome_qc_policy"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    temporal_coverage_policy = json.loads(
        (root / authorization["temporal_coverage_policy"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    return verifier._validate_inference_closure(
        root,
        {},
        authorization,
        protocol_binding=protocol_binding,
        protocol_document=protocol_document,
        protocol_seal_path=root / protocol_binding["seal"]["path"],
        outcome_qc_policy=outcome_qc_policy,
        temporal_coverage_policy=temporal_coverage_policy,
    )


def _manifest_command(root: Path, manifest: Path, *extra: str):
    return [
        sys.executable,
        str(MANIFEST_SCRIPT),
        "--root",
        str(root),
        "--manifest",
        str(manifest),
        "--no-git",
        *extra,
    ]


def test_independent_probability_v2_recomputes_nominal_pre_cqr_pinball(
    tmp_path,
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_probability_v2_nominal_test"
    )
    fixture = _write_independent_probability_v2_fixture(
        verifier, tmp_path, cqr_offset=0.25
    )
    artifact = _validate_independent_probability_fixture(verifier, fixture)
    assert artifact["format"] == verifier.PROBABILISTIC_EVALUATION_FORMAT
    assert all(
        row["endpoint_inversion_used"] is False
        for row in artifact["rows"]
    )


def test_independent_probability_v2_rejects_post_cqr_endpoint_pinball_bug(
    tmp_path,
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_probability_v2_endpoint_bug_test"
    )
    fixture = _write_independent_probability_v2_fixture(
        verifier, tmp_path, cqr_offset=0.25
    )
    artifact_path = fixture["artifact_path"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    for row in artifact["rows"]:
        # This is the original bug: score stored post-CQR interval endpoints as
        # nominal q05/q95. The trusted parquet and suite offsets remain intact.
        row["pinball_q05_c"] = 0.0625
        row["pinball_q50_c"] = 0.0
        row["pinball_q95_c"] = 0.0625
        row["equal_weight_three_quantile_pinball_mean_c"] = 1.0 / 24.0
    artifact_path.write_bytes(verifier._canonical_json_bytes(artifact))
    with pytest.raises(ValueError, match="stored-parquet recomputation"):
        _validate_independent_probability_fixture(verifier, fixture)


def test_independent_probability_v2_rejects_self_consistent_metric_reseal(
    tmp_path,
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_probability_v2_metric_reseal_test"
    )
    fixture = _write_independent_probability_v2_fixture(verifier, tmp_path)
    artifact_path = fixture["artifact_path"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    for row in artifact["rows"]:
        # Keep the forged Brier skill internally consistent with the forged
        # reference Brier score. Only replaying the suite-frozen seasonal
        # reference against stored target dates can reject this reseal.
        row["frozen_reference_brier_score"] = 0.5
        row["brier_skill_frozen_seasonal"] = (
            1.0 - row["brier_score"] / row["frozen_reference_brier_score"]
        )
    artifact_path.write_bytes(verifier._canonical_json_bytes(artifact))
    with pytest.raises(ValueError, match="stored-parquet recomputation"):
        _validate_independent_probability_fixture(verifier, fixture)


@pytest.mark.parametrize(
    "attack",
    ["count", "duplicate", "unknown_zero"],
)
def test_independent_probability_v2_rejects_availability_registry_attacks(
    tmp_path, attack,
):
    verifier = _load_script(
        VERIFY_SCRIPT,
        f"thermoroute_verify_probability_v2_availability_{attack}_test",
    )
    fixture = _write_independent_probability_v2_fixture(verifier, tmp_path)
    availability_path = fixture["availability_path"]
    availability = pd.read_csv(availability_path, dtype={"site_no": "string"})
    if attack == "count":
        availability.loc[0, "n_valid_targets"] += 1
    elif attack == "duplicate":
        availability = pd.concat(
            [availability, availability.iloc[[0]]], ignore_index=True
        )
    else:
        availability = pd.concat([
            availability,
            pd.DataFrame.from_records([{
                "cohort": "temporal",
                "site_no": "09999999",
                "horizon": 1,
                "n_valid_targets": 0,
                "reportable": False,
            }]),
        ], ignore_index=True)
    availability.to_csv(availability_path, index=False, lineterminator="\n")
    with pytest.raises(ValueError, match="availability|counts differ"):
        _validate_independent_probability_fixture(verifier, fixture)


@pytest.mark.parametrize(
    "attack",
    ["duplicate_row", "unknown_row", "extra_top_level"],
)
def test_independent_probability_v2_rejects_row_and_schema_attacks(
    tmp_path, attack,
):
    verifier = _load_script(
        VERIFY_SCRIPT, f"thermoroute_verify_probability_v2_{attack}_test"
    )
    fixture = _write_independent_probability_v2_fixture(verifier, tmp_path)
    artifact_path = fixture["artifact_path"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if attack == "duplicate_row":
        artifact["rows"].append(dict(artifact["rows"][0]))
    elif attack == "unknown_row":
        forged = dict(artifact["rows"][0])
        forged["model"] = "UnknownModel"
        artifact["rows"].append(forged)
    else:
        artifact["undeclared_field"] = "forged"
    artifact_path.write_bytes(verifier._canonical_json_bytes(artifact))
    with pytest.raises(ValueError, match="schema|registry"):
        _validate_independent_probability_fixture(verifier, fixture)


def test_independent_probability_v2_rejects_transient_nominal_public_columns(
    tmp_path,
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_probability_v2_transient_column_test"
    )
    fixture = _write_independent_probability_v2_fixture(verifier, tmp_path)
    temporal_path = fixture["prediction_paths"]["temporal"]
    frame = pd.read_parquet(temporal_path)
    frame["_nominal_q05"] = frame["q05"] + 0.25
    frame["_nominal_q50"] = frame["q50"]
    frame["_nominal_q95"] = frame["q95"] - 0.25
    frame.to_parquet(temporal_path, index=False)
    with pytest.raises(ValueError, match="schema columns/order"):
        _validate_independent_probability_fixture(verifier, fixture)


def test_authorization_required_models_must_equal_protocol_and_suite_order():
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_suite_authorization_order_test"
    )
    primary = [
        "Persistence", "DampedPersistence", "Climatology", "LightGBM",
        "LSTM", "ThermoRoute",
    ]
    controls = [
        "DampedPriorOnly", "TR-noDynamicPrior", "TR-fixedKappa",
        "TR-noRouter", "TR-noMoE", "TR-noTCN", "TR-unbounded",
    ]
    protocol = {
        "primary_inference_contract": {
            "primary_models": primary,
            "mandatory_exploratory_architecture_controls": controls,
        }
    }
    required = {
        "temporal": [*primary, *controls],
        "external": list(primary),
    }
    cohorts = {
        cohort: {"models": [{"model_id": model} for model in models]}
        for cohort, models in required.items()
    }
    assert verifier._validate_authorized_suite_model_order(
        protocol, cohorts, required
    ) == {
        cohort: tuple(models) for cohort, models in required.items()
    }

    attacked_authorization = {
        cohort: list(models) for cohort, models in required.items()
    }
    attacked_authorization["external"].pop()
    with pytest.raises(ValueError, match="authorization required-model registry"):
        verifier._validate_authorized_suite_model_order(
            protocol, cohorts, attacked_authorization
        )

    attacked_cohorts = json.loads(json.dumps(cohorts))
    attacked_cohorts["external"]["models"][0:2] = reversed(
        attacked_cohorts["external"]["models"][0:2]
    )
    with pytest.raises(ValueError, match="model ID/order registry"):
        verifier._validate_authorized_suite_model_order(
            protocol, attacked_cohorts, required
        )


def test_manifest_binds_revision_source_config_data_and_detects_change(tmp_path):
    manifest_path = _write_fixture(tmp_path)
    commit, tree = "a" * 40, "b" * 40
    subprocess.run(
        _manifest_command(
            tmp_path,
            manifest_path,
            "--development-prelabel",
            "--source-git-commit",
            commit,
            "--source-git-tree",
            tree,
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert document["schema_version"] == "thermoroute.provenance-manifest.v2"
    assert document["manifest_role"] == (
        "DEVELOPMENT_PRELABEL_INVENTORY_NO_CONFIRMATORY_OR_OUTCOME_NAMESPACE"
    )
    assert document["scientific_evidence_authority"].startswith(
        "MANIFEST_ALONE_CONFERS_NO_SCIENTIFIC_AUTHORITY"
    )
    assert document["git"]["commit"] == commit
    assert "STATIONS" not in document["resolved_config"]["values"]
    assert not any(key.startswith("legacy_") for key in document["current_truth"])
    assert {"@git", "@source", "@config", "@dependencies"} <= set(document["dag"])
    assert set(document["dag"]["outputs/tables/result.csv"]["parents"]) == {
        "@git",
        "@source",
        "@config",
        "@dependencies",
    }
    renamed = document["dag"]["outputs/predictions/copied_legacy.parquet"]
    assert renamed["kind"] == "workspace_inventory"
    assert renamed["authority"] == (
        "WORKSPACE_INVENTORY_ONLY_REQUIRES_SEPARATE_RECEIPT_VALIDATION"
    )
    assert "data_usgs/station_input.csv" not in renamed["parents"]
    assert "outputs/predictions/copied_legacy.parquet" not in set(
        document["current_truth"].values()
    )

    attacked = json.loads(json.dumps(document))
    attacked["current_truth"]["usgs_predictions"] = (
        "outputs/predictions/copied_legacy.parquet"
    )
    attacked["dag"]["outputs/predictions/copied_legacy.parquet"]["authority"] = (
        "VALIDATED_CANONICAL_STAGE09_PREDICTION_LINEAGE"
    )
    attacked["dag"]["outputs/predictions/copied_legacy.parquet"]["parents"].append(
        "data_usgs/station_input.csv"
    )
    manifest_path.write_text(
        json.dumps(attacked, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tampered = subprocess.run(
        _manifest_command(tmp_path, manifest_path, "--check"),
        check=False,
        capture_output=True,
        text=True,
    )
    assert tampered.returncode == 1
    assert "CURRENT_TRUTH_CHANGED" in tampered.stderr
    assert "DAG_CONTENT_OR_AUTHORITY_CHANGED" in tampered.stderr

    wrong_role = json.loads(json.dumps(document))
    wrong_role["manifest_role"] = "WORKTREE_OR_STAGED_RELEASE_INVENTORY"
    wrong_role["scientific_evidence_authority"] = (
        "BYTE_INVENTORY_ONLY; RELEASE_PROFILE_AND_INDEPENDENT_VALIDATORS_DEFINE_"
        "SCIENTIFIC_AUTHORITY"
    )
    manifest_path.write_text(
        json.dumps(wrong_role, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    role_attack = subprocess.run(
        _manifest_command(
            tmp_path, manifest_path, "--check", "--development-prelabel"
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    assert role_attack.returncode == 1
    assert "MANIFEST_ROLE_EXPECTED" in role_attack.stderr

    manifest_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        _manifest_command(tmp_path, manifest_path, "--check"),
        check=True,
        capture_output=True,
        text=True,
    )
    (tmp_path / "data_usgs" / "station_input.csv").write_text(
        "x\n2\n", encoding="utf-8"
    )
    changed = subprocess.run(
        _manifest_command(tmp_path, manifest_path, "--check"),
        check=False,
        capture_output=True,
        text=True,
    )
    assert changed.returncode == 1
    assert "CHANGED data_usgs/station_input.csv" in changed.stderr


def test_development_manifest_stage09_is_inventory_only_despite_canonical_name(
    tmp_path,
):
    manifest_builder = _load_script(
        MANIFEST_SCRIPT, "thermoroute_manifest_exact_stage09_lineage_test"
    )
    files = {
        "data_usgs/panel_usgs_120v2.parquet": {"sha256": "1" * 64, "bytes": 1},
        "data_usgs/station_registry_v1.csv": {"sha256": "2" * 64, "bytes": 1},
        "data_usgs/station_retired_alias.csv": {"sha256": "3" * 64, "bytes": 1},
        "outputs/predictions/usgs_predictions_stage9_v2.parquet": {
            "sha256": "4" * 64,
            "bytes": 1,
        },
    }
    graph = manifest_builder.lineage_graph(
        tmp_path,
        files,
        "5" * 64,
        "6" * 64,
        "7" * 64,
        "8" * 64,
        development_prelabel=True,
    )
    node = graph["outputs/predictions/usgs_predictions_stage9_v2.parquet"]
    assert node["kind"] == "workspace_inventory"
    assert node["authority"] == (
        "WORKSPACE_INVENTORY_ONLY_REQUIRES_SEPARATE_RECEIPT_VALIDATION"
    )
    assert set(node["parents"]) == {
        "@git",
        "@source",
        "@config",
        "@dependencies",
    }
    assert "data_usgs/station_retired_alias.csv" not in node["parents"]


def test_manifest_filename_presence_never_creates_current_truth(tmp_path):
    manifest_builder = _load_script(
        MANIFEST_SCRIPT, "thermoroute_manifest_no_filename_truth_test"
    )
    for relative in (
        "data_usgs/panel_usgs_120v2.parquet",
        "data_usgs/station_registry_v1.csv",
        "outputs/predictions/usgs_predictions_stage9_v2.parquet",
        "outputs/tables/usgs_scores.csv",
    ):
        _write_bytes(tmp_path, relative, b"renamed-untrusted-bytes\n")
    assert manifest_builder._current_truth(tmp_path) == {}


@pytest.mark.parametrize("attack", ("file_symlink", "parent_symlink", "hardlink"))
def test_manifest_inventory_rejects_link_aliases(tmp_path, attack):
    manifest_builder = _load_script(
        MANIFEST_SCRIPT, f"thermoroute_manifest_link_alias_{attack}_test"
    )
    withdrawn = _write_bytes(tmp_path, "data/b1.csv", b"withdrawn bytes\n")
    alias = tmp_path / "data_usgs" / "station_alias.csv"
    if attack == "parent_symlink":
        target_parent = tmp_path / "alias-target"
        target_parent.mkdir()
        alias = target_parent / "station_alias.csv"
        alias.write_bytes(withdrawn.read_bytes())
        (tmp_path / "data_usgs").symlink_to(target_parent, target_is_directory=True)
        candidate = tmp_path / "data_usgs" / "station_alias.csv"
        with pytest.raises(RuntimeError, match="MANIFEST_UNSAFE_ARTIFACT"):
            manifest_builder._inventory_file_binding(tmp_path, candidate)
        return

    alias.parent.mkdir(parents=True)
    if attack == "file_symlink":
        alias.symlink_to(withdrawn)
    else:
        os.link(withdrawn, alias)
    with pytest.raises(RuntimeError, match="MANIFEST_UNSAFE_ARTIFACT"):
        manifest_builder.inventory(tmp_path, ("data_usgs/station*.csv",))


def test_release_boundary_requires_contract_and_rejects_traversal(tmp_path):
    from thermoroute.chronology import FIXED_PRELABEL_ABSENCE_PATHS

    verifier = _load_script(VERIFY_SCRIPT, "thermoroute_verify_release_test")
    manifest_builder = _load_script(
        MANIFEST_SCRIPT, "thermoroute_manifest_release_boundary_test"
    )
    semantics_tree = ast.parse(
        (ROOT / "scripts" / "_legacy_site_semantics.py").read_text(
            encoding="utf-8"
        )
    )
    legacy_figure_basenames = next(
        ast.literal_eval(node.value)
        for node in semantics_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "LEGACY_FIGURE_BASENAMES"
            for target in node.targets
        )
    )
    complete = (
        set(verifier.REQUIRED_MEMBERS)
        | set(verifier.REQUIRED_PAPER_MEMBERS)
        | {"outputs/reports/report.md"}
    )
    verifier.validate_members(complete)
    assert verifier.LEGACY_THREE_SITE_NOTICE_PATH in verifier.REQUIRED_MEMBERS
    native_notices = {
        verifier.NATIVE_THREAD_ENFORCEMENT_NOTICE_PATH,
        verifier.NATIVE_ARTIFACT_PUBLICATION_NOTICE_PATH,
    }
    assert native_notices <= set(verifier.REQUIRED_MEMBERS)
    route_a_governance = {
        "src/thermoroute/chronology.py",
        "src/thermoroute/model_matrix_amendment.py",
        "src/thermoroute/model_suite.py",
        verifier.RELEASE_MECHANICS_CORE_PATH,
        "scripts/make_release_archive.sh",
        "scripts/24_freeze_model_suite.py",
        "scripts/28_freeze_prelabel_chronology.py",
        verifier.RELEASE_MECHANICS_RUNNER_PATH,
        verifier.MODEL_MATRIX_AMENDMENT_PATH,
        verifier.MODEL_MATRIX_AMENDMENT_SEAL_PATH,
    }
    assert route_a_governance <= set(verifier.REQUIRED_MEMBERS)
    retired_inputs = {"data/b1.csv", "data/s2.csv", "data/p3.csv"}
    assert retired_inputs.isdisjoint(verifier.REQUIRED_MEMBERS)
    for retired_input in retired_inputs:
        with pytest.raises(ValueError, match="forbidden non-Route-A direct input"):
            verifier.validate_members(complete | {retired_input})
    for retired_output in verifier.RETIRED_MONITORING_CASE_OUTPUT_MEMBERS:
        with pytest.raises(ValueError, match="mixed-generation|rendered/binary"):
            verifier.validate_members(complete | {retired_output})
    assert (
        manifest_builder.RETIRED_MONITORING_CASE_OUTPUT_MEMBERS
        == verifier.RETIRED_MONITORING_CASE_OUTPUT_MEMBERS
    )
    historical_figure_members = {
        f"outputs/figures/{stem}.{suffix}"
        for stem in legacy_figure_basenames
        for suffix in ("png", "pdf")
    }
    assert historical_figure_members <= set(
        verifier.RETIRED_MONITORING_CASE_OUTPUT_MEMBERS
    )
    retired_output = "outputs/figures/fig1_study_area.png"
    _write_bytes(tmp_path, retired_output, b"retired monitoring-case fixture\n")
    with pytest.raises(
        RuntimeError, match="ROUTE_A_RETIRED_MONITORING_CASE_ARTIFACT_PRESENT"
    ):
        manifest_builder.assert_route_a_artifact_boundary(tmp_path)
    boundary_check = subprocess.run(
        [
            sys.executable,
            str(MANIFEST_SCRIPT),
            "--root",
            str(tmp_path),
            "--check-route-a-boundary",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert boundary_check.returncode == 1
    assert retired_output in boundary_check.stderr
    assert set(FIXED_PRELABEL_ABSENCE_PATHS) <= set(
        manifest_builder.PRE_MODEL_FORBIDDEN_PATHS
    )
    assert "data_usgs/wtemp_daily_max.parquet" in (
        manifest_builder.PRE_MODEL_FORBIDDEN_PATHS
    )
    assert "outputs/prelabel/route_a_inference_gate_v1.json" in (
        manifest_builder.PRE_MODEL_FORBIDDEN_PATHS
    )
    premodel_root = tmp_path / "premodel"
    _write_bytes(
        premodel_root,
        "outputs/confirmatory/synthetic_forbidden_outcome.parquet",
        b"synthetic test marker; must never be read\n",
    )
    with pytest.raises(
        RuntimeError, match="PREMODEL_FORBIDDEN_PATH_PRESENT_WITHOUT_READING"
    ):
        manifest_builder.assert_development_prelabel_boundary(premodel_root)
    assert (
        "outputs/confirmatory/synthetic_forbidden_outcome.parquet"
        not in manifest_builder.inventory(
            premodel_root,
            manifest_builder.DEVELOPMENT_PRELABEL_ARTIFACT_PATTERNS,
        )
    )
    with pytest.raises(ValueError, match="missing required members"):
        verifier.validate_members(
            complete - {verifier.LEGACY_THREE_SITE_NOTICE_PATH}
        )
    for notice in native_notices:
        with pytest.raises(ValueError, match="missing required members"):
            verifier.validate_members(complete - {notice})
    for evidence in route_a_governance:
        with pytest.raises(ValueError, match="missing required members"):
            verifier.validate_members(complete - {evidence})
    for missing_paper in verifier.REQUIRED_PAPER_MEMBERS:
        with pytest.raises(
            ValueError, match="missing registered manuscript sources"
        ):
            verifier.validate_members(complete - {missing_paper})
    with pytest.raises(ValueError, match="mixed-generation"):
        verifier.validate_members(
            complete | {"outputs/tables/usgs_stations_with_huc.csv"}
        )
    for stale, expected_error in (
        ("paper/ThermoRoute_paper.pdf", "rendered/binary manuscript artifact"),
        ("paper/ThermoRoute_paper.docx", "rendered/binary manuscript artifact"),
        ("paper/figures/stale-result.png", "rendered/binary manuscript artifact"),
        ("paper/unregistered_notes.md", "unregistered manuscript artifact"),
    ):
        with pytest.raises(ValueError, match=expected_error):
            verifier.validate_members(complete | {stale})
    for rendered_alias in (
        "Paper/stale.pdf",
        "ThermoRoute_paper.docx",
        "docs/manuscript.pdf",
        "figures/result.png",
        "assets/result.svg",
    ):
        with pytest.raises(ValueError, match="rendered/binary manuscript artifact"):
            verifier.validate_members(complete | {rendered_alias})
    for unsafe_alias in (
        "paper\\stale.txt",
        "Paper/stale.txt",
        "paper//stale.txt",
        "paper/./stale.txt",
        "paper/stale. ",
        "paper/NUL.txt",
        "paper/e\u0301vidence.txt",
    ):
        with pytest.raises(ValueError, match="release path|filesystem alias"):
            verifier.validate_members(complete | {unsafe_alias})

    shell = MAKE_RELEASE_SCRIPT.read_text(encoding="utf-8")
    assert "ALLOW_DIRTY_RELEASE" not in shell
    verify_temporary_archive = shell.index(
        'scripts/verify_release.py "$TMP_ZIP"'
    )
    publish_archive = shell.index('mv "$TMP_ZIP" "$OUT"')
    assert verify_temporary_archive < publish_archive
    assert verifier.LEGACY_THREE_SITE_NOTICE_PATH in shell
    assert shell.count(verifier.LEGACY_THREE_SITE_NOTICE_PATH) == 1
    for notice in native_notices:
        assert notice in shell
    paper_block = shell.split("paper_paths=(", 1)[1].split("\n)", 1)[0]
    shell_paper_paths = {
        line.strip() for line in paper_block.splitlines() if line.strip()
    }
    assert shell_paper_paths == set(verifier.ALLOWED_PAPER_MEMBERS)

    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("thermoroute/../escape.txt", "bad")
    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ValueError, match="unsafe archive path|unsafe platform alias"):
            verifier.normalised_members(archive)

    alias_archive_path = tmp_path / "alias.zip"
    with zipfile.ZipFile(alias_archive_path, "w") as archive:
        archive.writestr("thermoroute/paper\\stale.txt", "bad")
    with zipfile.ZipFile(alias_archive_path) as archive:
        with pytest.raises(ValueError, match="prohibited character"):
            verifier.normalised_members(archive)


@pytest.mark.parametrize(
    "relative",
    (
        "LICENSE",
        "data_usgs/frozen_panel_v1.json",
        (
            "data_usgs/raw_snapshots/huc-v1/usgs-nwis-site-metadata/"
            "request/response.bin"
        ),
    ),
)
def test_canonical_archive_data_and_license_are_bound_to_compute_git_blob(
    tmp_path: Path, relative: str,
) -> None:
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_git_bound_archive_data_test"
    )
    source = tmp_path / "source"
    archive = tmp_path / "archive"
    source.mkdir()
    archive.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    fixtures = {
        "LICENSE": b"fixture license\n",
        "data_usgs/frozen_panel_v1.json": b'{"schema_version":1}\n',
        (
            "data_usgs/raw_snapshots/huc-v1/usgs-nwis-site-metadata/"
            "request/response.bin"
        ): b"raw provider response\n",
    }
    for path, payload in fixtures.items():
        _write_bytes(source, path, payload)
        _write_bytes(archive, path, payload)
    commit = _commit_git_fixture(source, "canonical archive fixture")

    verifier._verify_canonical_archive_blobs_from_bundle(
        root=archive, bare=source, compute_commit=commit
    )
    (archive / relative).write_bytes(b"self-consistently replaced archive bytes\n")
    with pytest.raises(ValueError, match="differs from compute Git blob"):
        verifier._verify_canonical_archive_blobs_from_bundle(
            root=archive, bare=source, compute_commit=commit
        )


def test_v2_seal_history_and_manuscript_git_blobs_fail_closed(tmp_path):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_v2_seal_history_test"
    )
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)

    _write_bytes(root, "protocols/base_protocol.txt", b"frozen protocol\n")
    base_commit = _commit_git_fixture(root, "base protocol")
    protocol_seal = {
        "format": verifier.PROTOCOL_SEAL_FORMAT,
        "status": "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED",
        "final_prelabel_protocol": {"commit": base_commit},
    }
    _write_canonical_json(
        verifier, root, verifier.PROTOCOL_SEAL_PATH, protocol_seal
    )
    _commit_git_fixture(root, "base protocol seal")

    amendment = json.loads(
        (ROOT / verifier.INFERENCE_AMENDMENT_PATH).read_text(encoding="utf-8")
    )
    amendment["base_protocol_seal"] = _binding(
        verifier, root, verifier.PROTOCOL_SEAL_PATH
    )
    amendment_path = _write_canonical_json(
        verifier, root, verifier.INFERENCE_AMENDMENT_PATH, amendment
    )
    manuscript = _write_bytes(root, "paper/highlights.md", b"frozen source\n")
    amendment_commit = _commit_git_fixture(root, "v2 amendment")

    seal = {
        "format": "thermoroute.route-a-inference-amendment-seal.v2",
        "status": "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED",
        "amendment_id": amendment["amendment_id"],
        "amendment": _binding(verifier, root, verifier.INFERENCE_AMENDMENT_PATH),
        "base_protocol_seal": amendment["base_protocol_seal"],
        "final_prelabel_commit": amendment_commit,
        "history_contract": {
            "base_protocol_commit_must_be_ancestor": True,
            "amendment_blob_must_match_commit": True,
            "amendment_commit_must_be_ancestor_of_authorization": True,
            "seal_is_created_only_after_amendment_commit": True,
        },
        "prelabel_attestation": {
            "post_2020_wtemp_requested_or_inspected": False,
            "outcome_independent": True,
        },
    }
    seal_path = _write_canonical_json(
        verifier, root, verifier.INFERENCE_AMENDMENT_SEAL_PATH, seal
    )
    registry = {
        "format": "thermoroute.route-a-claim-ledger.v2",
        "inference_amendment_binding": {
            **_binding(verifier, root, verifier.INFERENCE_AMENDMENT_PATH),
            "format": amendment["format"],
            "amendment_id": amendment["amendment_id"],
            "seal": {
                **_binding(
                    verifier, root, verifier.INFERENCE_AMENDMENT_SEAL_PATH
                ),
                "status": "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED",
            },
        },
    }
    _write_canonical_json(
        verifier,
        root,
        "protocols/route_a_claim_registry_v1.json",
        registry,
    )
    seal_commit = _commit_git_fixture(root, "separate v2 seal")

    verifier._load_release_sealed_amendment(root)
    verifier._verify_amendment_seal_history_from_bundle(
        root=root, bare=root, compute_commit=seal_commit
    )
    verifier._verify_manuscript_blobs_from_bundle(
        root=root, bare=root, manuscript_commit=seal_commit
    )

    manuscript.write_bytes(b"uncommitted rewrite\n")
    with pytest.raises(ValueError, match="bytes differ from Git"):
        verifier._verify_manuscript_blobs_from_bundle(
            root=root, bare=root, manuscript_commit=seal_commit
        )
    manuscript.write_bytes(b"frozen source\n")

    seal_path.write_text(json.dumps(seal, indent=2) + "\n", encoding="utf-8")
    registry["inference_amendment_binding"]["seal"]["sha256"] = (
        verifier.sha256_file(seal_path)
    )
    _write_canonical_json(
        verifier,
        root,
        "protocols/route_a_claim_registry_v1.json",
        registry,
    )
    rewritten_commit = _commit_git_fixture(root, "forbidden seal rewrite")
    with pytest.raises(ValueError, match="differs from its creation commit"):
        verifier._verify_amendment_seal_history_from_bundle(
            root=root, bare=root, compute_commit=rewritten_commit
        )

    amendment["base_protocol_seal"]["path"] = "../../outside.json"
    _write_canonical_json(
        verifier, root, verifier.INFERENCE_AMENDMENT_PATH, amendment
    )
    seal["amendment"] = _binding(
        verifier, root, verifier.INFERENCE_AMENDMENT_PATH
    )
    seal["base_protocol_seal"] = amendment["base_protocol_seal"]
    _write_canonical_json(
        verifier, root, verifier.INFERENCE_AMENDMENT_SEAL_PATH, seal
    )
    registry["inference_amendment_binding"]["sha256"] = verifier.sha256_file(
        amendment_path
    )
    registry["inference_amendment_binding"]["seal"]["sha256"] = (
        verifier.sha256_file(seal_path)
    )
    _write_canonical_json(
        verifier,
        root,
        "protocols/route_a_claim_registry_v1.json",
        registry,
    )
    with pytest.raises(ValueError, match="amendment/seal semantics changed"):
        verifier._load_release_sealed_amendment(root)


def test_probability_metric_erratum_bundle_history_is_strictly_ordered(
    tmp_path,
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_probability_erratum_history_test"
    )
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)

    protocol_relative = "protocols/route_a_confirmatory_v1.json"
    protocol_path = root / protocol_relative
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / protocol_relative, protocol_path)
    base_commit = _commit_git_fixture(root, "base protocol")
    base_seal = {
        "format": verifier.PROTOCOL_SEAL_FORMAT,
        "status": "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED",
        "protocol_id": "route-a-confirmatory-v1",
        "final_prelabel_protocol": {
            "commit": base_commit,
            "json": _binding(verifier, root, protocol_relative),
        },
    }
    _write_canonical_json(
        verifier, root, verifier.PROTOCOL_SEAL_PATH, base_seal
    )
    _commit_git_fixture(root, "base protocol seal")

    amendment = {
        "format": "thermoroute.route-a-inference-amendment.v2",
        "status": "FROZEN_PRELABEL_OUTCOME_FREE",
        "post_2020_wtemp_requested_or_inspected": False,
        "outcome_independent": True,
    }
    _write_canonical_json(
        verifier, root, verifier.INFERENCE_AMENDMENT_PATH, amendment
    )
    amendment_commit = _commit_git_fixture(root, "inference amendment")
    amendment_seal = {
        "format": "thermoroute.route-a-inference-amendment-seal.v2",
        "status": "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED",
        "amendment": _binding(
            verifier, root, verifier.INFERENCE_AMENDMENT_PATH
        ),
        "final_prelabel_commit": amendment_commit,
    }
    _write_canonical_json(
        verifier,
        root,
        verifier.INFERENCE_AMENDMENT_SEAL_PATH,
        amendment_seal,
    )
    _commit_git_fixture(root, "inference amendment seal")

    erratum_module_path = root / "src/thermoroute/probability_metric_erratum.py"
    erratum_module_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        ROOT / "src/thermoroute/probability_metric_erratum.py",
        erratum_module_path,
    )
    erratum_module = verifier._load_canonical_probability_metric_erratum_module(
        root
    )
    erratum = erratum_module.expected_probability_metric_erratum_document(
        root=root
    )
    erratum_path = _write_canonical_json(
        verifier, root, verifier.PROBABILITY_METRIC_ERRATUM_PATH, erratum
    )
    document_commit = _commit_git_fixture(root, "probability metric erratum")
    erratum_seal = {
        "format": verifier.PROBABILITY_METRIC_ERRATUM_SEAL_FORMAT,
        "status": "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED",
        "erratum_id": verifier.PROBABILITY_METRIC_ERRATUM_ID,
        "erratum": _binding(
            verifier, root, verifier.PROBABILITY_METRIC_ERRATUM_PATH
        ),
        "governance_seals": {
            "base_protocol_seal": _binding(
                verifier, root, verifier.PROTOCOL_SEAL_PATH
            ),
            "inference_amendment_seal": _binding(
                verifier, root, verifier.INFERENCE_AMENDMENT_SEAL_PATH
            ),
        },
        "erratum_document_commit": document_commit,
        "history_contract": dict(erratum_module.HISTORY_CONTRACT),
        "prelabel_attestation": dict(erratum_module.PRELABEL_ATTESTATION),
    }
    _write_canonical_json(
        verifier,
        root,
        verifier.PROBABILITY_METRIC_ERRATUM_SEAL_PATH,
        erratum_seal,
    )
    seal_commit = _commit_git_fixture(root, "separate metric erratum seal")

    verifier._verify_probability_metric_erratum_history_from_bundle(
        root=root, bare=root, compute_commit=seal_commit
    )

    original = erratum_path.read_bytes()
    original_seal = (root / verifier.PROBABILITY_METRIC_ERRATUM_SEAL_PATH).read_bytes()
    original_branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "switch", "-q", "-c", "erratum-rebirth", seal_commit],
        cwd=root,
        check=True,
    )
    erratum_path.unlink()
    _commit_git_fixture(root, "delete probability metric erratum")
    erratum_path.write_bytes(original)
    rebirth_commit = _commit_git_fixture(
        root, "re-add identical probability metric erratum"
    )
    with pytest.raises(ValueError, match="document was not created exactly once"):
        verifier._verify_probability_metric_erratum_history_from_bundle(
            root=root, bare=root, compute_commit=rebirth_commit
        )
    subprocess.run(
        ["git", "switch", "-q", original_branch], cwd=root, check=True
    )

    attacked = json.loads(original)
    attacked["implementation_requirements"]["trusted_artifact_path"] = (
        "trusted/attacker_selected_probability.json"
    )
    _write_canonical_json(
        verifier, root, verifier.PROBABILITY_METRIC_ERRATUM_PATH, attacked
    )
    attacked_seal = json.loads(original_seal)
    attacked_seal["erratum"] = _binding(
        verifier, root, verifier.PROBABILITY_METRIC_ERRATUM_PATH
    )
    _write_canonical_json(
        verifier,
        root,
        verifier.PROBABILITY_METRIC_ERRATUM_SEAL_PATH,
        attacked_seal,
    )
    with pytest.raises(ValueError, match="complete contract changed"):
        verifier._load_release_probability_metric_erratum(root)
    erratum_path.write_bytes(original)
    (root / verifier.PROBABILITY_METRIC_ERRATUM_SEAL_PATH).write_bytes(
        original_seal
    )

    erratum_path.write_bytes(original + b" ")
    _commit_git_fixture(root, "forbidden erratum rewrite")
    erratum_path.write_bytes(original)
    restored_commit = _commit_git_fixture(root, "restore erratum bytes")
    with pytest.raises(ValueError, match="changed after freezing"):
        verifier._verify_probability_metric_erratum_history_from_bundle(
            root=root, bare=root, compute_commit=restored_commit
        )


def test_archive_resource_limits_are_checked_before_extraction(tmp_path, monkeypatch):
    verifier = _load_script(VERIFY_SCRIPT, "thermoroute_verify_zip_limits_test")

    def archive_with(payloads: list[bytes]) -> Path:
        path = tmp_path / f"fixture-{len(list(tmp_path.glob('fixture-*.zip')))}.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for index, payload in enumerate(payloads):
                info = zipfile.ZipInfo(f"thermoroute/data/{index}.bin")
                info.date_time = (1980, 1, 1, 0, 0, 0)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (0o100644 & 0xFFFF) << 16
                archive.writestr(info, payload)
        return path

    member_limited = archive_with([b"a", b"b"])
    monkeypatch.setattr(verifier, "MAX_ARCHIVE_MEMBERS", 1)
    with pytest.raises(ValueError, match="member count"):
        verifier._preflight_zip_container(member_limited)
    with zipfile.ZipFile(member_limited) as archive:
        with pytest.raises(ValueError, match="member count"):
            verifier.normalised_members(archive)

    monkeypatch.setattr(verifier, "MAX_ARCHIVE_MEMBERS", 10)
    member_bytes = archive_with([b"four"])
    monkeypatch.setattr(verifier, "MAX_ARCHIVE_MEMBER_BYTES", 3)
    with zipfile.ZipFile(member_bytes) as archive:
        with pytest.raises(ValueError, match="uncompressed safety limit"):
            verifier.normalised_members(archive)

    monkeypatch.setattr(verifier, "MAX_ARCHIVE_MEMBER_BYTES", 10)
    total_bytes = archive_with([b"abc", b"def"])
    monkeypatch.setattr(verifier, "MAX_ARCHIVE_TOTAL_BYTES", 5)
    with zipfile.ZipFile(total_bytes) as archive:
        with pytest.raises(ValueError, match="total uncompressed"):
            verifier.normalised_members(archive)

    monkeypatch.setattr(verifier, "MAX_ARCHIVE_TOTAL_BYTES", 10_000)
    compressed = archive_with([b"0" * 4096])
    monkeypatch.setattr(verifier, "MAX_ARCHIVE_MEMBER_BYTES", 10_000)
    monkeypatch.setattr(verifier, "MAX_ARCHIVE_COMPRESSION_RATIO", 2)
    with zipfile.ZipFile(compressed) as archive:
        with pytest.raises(ValueError, match="compression ratio"):
            verifier.normalised_members(archive)


def test_archive_python_is_not_executed_before_git_source_binding(tmp_path):
    verifier = _load_script(VERIFY_SCRIPT, "thermoroute_verify_preexec_gate_test")
    source, stage = tmp_path / "source", tmp_path / "stage"
    source.mkdir()
    stage.mkdir()
    _minimal_canonical_release(verifier, source)
    verifier.materialize_release_profile(source, stage, verifier.PREOPEN_PROFILE)
    sentinel = stage / "ARCHIVE_PYTHON_EXECUTED"
    _write_bytes(
        stage,
        "scripts/26_validate_claims.py",
        (
            "from pathlib import Path\n"
            "Path('ARCHIVE_PYTHON_EXECUTED').write_text('bad', encoding='utf-8')\n"
        ).encode(),
    )
    registry_path = _write_bytes(
        stage,
        "protocols/route_a_claim_registry_v1.json",
        json.dumps({"documents": ["paper/main.md"]}).encode(),
    )
    paper = _write_bytes(stage, "paper/main.md", b"preopen language\n")
    validator = stage / "scripts/26_validate_claims.py"
    with pytest.raises(ValueError, match="Git/preregistration evidence"):
        verifier.materialize_claim_audit(stage, verifier.PREOPEN_PROFILE)
    assert not sentinel.exists()
    audit = {
        "format": "thermoroute.route-a-release-claim-audit.v1",
        "profile": verifier.PREOPEN_PROFILE,
        "require_complete": False,
        "validator": verifier._binding_for(stage, validator),
        "registry": verifier._binding_for(stage, registry_path),
        "scanned_documents": [verifier._binding_for(stage, paper)],
        "violation_count": 0,
    }
    audit_path = _write_bytes(
        stage,
        verifier.CLAIM_AUDIT_PATH,
        json.dumps(audit, sort_keys=True).encode() + b"\n",
    )
    marker_path = stage / verifier.PROFILE_MARKER
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["claim_validation"] = verifier._binding_for(stage, audit_path)
    marker_path.write_bytes(verifier._canonical_json_bytes(marker))

    with pytest.raises(ValueError, match="Git/preregistration evidence"):
        verifier.verify_release_profile(stage, run_trusted_replay=True)
    assert not sentinel.exists()


def test_git_path_lifetime_walk_sees_add_delete_on_merged_side_branch(tmp_path):
    verifier = _load_script(VERIFY_SCRIPT, "thermoroute_verify_path_lifetime_test")
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    environment = os.environ.copy()
    environment.update({
        "GIT_AUTHOR_NAME": "Fixture",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_NAME": "Fixture",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
    })

    def commit(message: str) -> str:
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", message],
            cwd=root,
            env=environment,
            check=True,
        )
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()

    _write_bytes(root, "base.txt")
    commit("base")
    main_branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=root, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    subprocess.run(["git", "switch", "-q", "-c", "side"], cwd=root, check=True)
    receipt_relative = "outputs/prelabel/route_a_prelabel_chronology_v1.json"
    _write_bytes(root, receipt_relative, b"side birth\n")
    side_birth = commit("side receipt birth")
    (root / receipt_relative).unlink()
    commit("side receipt deletion")
    subprocess.run(["git", "switch", "-q", main_branch], cwd=root, check=True)
    _write_bytes(root, receipt_relative, b"canonical birth\n")
    main_birth = commit("canonical receipt birth")
    subprocess.run(
        ["git", "merge", "-q", "--no-ff", "side", "-m", "merge side"],
        cwd=root,
        env=environment,
        check=True,
    )
    tip = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        capture_output=True, check=True,
    ).stdout.strip()

    assert set(verifier._git_path_creation_commits(root, tip, receipt_relative)) == {
        side_birth,
        main_birth,
    }


def test_release_replay_accepts_one_strict_immutable_seal_birth(tmp_path):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_seal_lineage_valid_test"
    )
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _write_bytes(root, "base.txt")
    _commit_git_fixture(root, "base")
    relative = "protocols/route_a_inference_amendment_seal_v2.json"
    _write_bytes(root, "protocols/route_a_inference_amendment_v2.json", b"{}\n")
    amendment_commit = _commit_git_fixture(root, "amendment")
    payload = b'{"seal":"canonical"}\n'
    _write_bytes(root, relative, payload)
    creation = _commit_git_fixture(root, "seal")

    assert verifier._verify_unique_immutable_path_creation(
        root,
        tip="HEAD",
        predecessor=amendment_commit,
        relative=relative,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        label="inference amendment seal",
    ) == creation


@pytest.mark.parametrize(
    ("attack", "error"),
    (
        ("same_commit", "existed at its required predecessor commit"),
        ("preexisting", "existed at its required predecessor commit"),
        ("add_delete_readd", "exactly one reachable Git creation"),
        ("post_create_modify", "deleted or changed after creation"),
    ),
)
def test_release_replay_rejects_adversarial_seal_histories(
    tmp_path, attack, error,
):
    verifier = _load_script(
        VERIFY_SCRIPT, f"thermoroute_verify_seal_lineage_{attack}_test"
    )
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _write_bytes(root, "base.txt")
    _commit_git_fixture(root, "base")
    amendment = "protocols/route_a_inference_amendment_v2.json"
    relative = "protocols/route_a_inference_amendment_seal_v2.json"
    payload = b'{"seal":"canonical"}\n'

    if attack == "same_commit":
        _write_bytes(root, amendment, b"{}\n")
        _write_bytes(root, relative, payload)
        amendment_commit = _commit_git_fixture(root, "amendment and seal")
    elif attack == "preexisting":
        _write_bytes(root, relative, payload)
        _commit_git_fixture(root, "premature seal")
        _write_bytes(root, amendment, b"{}\n")
        amendment_commit = _commit_git_fixture(root, "later amendment")
    else:
        _write_bytes(root, amendment, b"{}\n")
        amendment_commit = _commit_git_fixture(root, "amendment")
        _write_bytes(root, relative, payload)
        _commit_git_fixture(root, "seal")
        if attack == "add_delete_readd":
            (root / relative).unlink()
            _commit_git_fixture(root, "delete seal")
            _write_bytes(root, relative, payload)
            _commit_git_fixture(root, "re-add seal")
        else:
            _write_bytes(root, relative, b'{"seal":"changed"}\n')
            _commit_git_fixture(root, "modify seal")
            _write_bytes(root, relative, payload)
            _commit_git_fixture(root, "restore seal")

    with pytest.raises(ValueError, match=error):
        verifier._verify_unique_immutable_path_creation(
            root,
            tip="HEAD",
            predecessor=amendment_commit,
            relative=relative,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            label="inference amendment seal",
        )


def test_manifest_does_not_promote_unsealed_canonical_stage09_filename(tmp_path):
    manifest_path = _write_fixture(tmp_path)
    prediction = (
        tmp_path
        / "outputs"
        / "predictions"
        / "usgs_predictions_stage9_v2.parquet"
    )
    prediction.parent.mkdir(parents=True, exist_ok=True)
    prediction.write_bytes(b"unsealed-stage09-bytes")
    scores = tmp_path / "outputs" / "tables" / "usgs_scores.csv"
    scores.parent.mkdir(parents=True, exist_ok=True)
    scores.write_text("horizon,site\n", encoding="utf-8")
    subprocess.run(
        _manifest_command(tmp_path, manifest_path),
        check=True,
        capture_output=True,
        text=True,
    )
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert document["current_truth"] == {}
    assert document["dag"][prediction.relative_to(tmp_path).as_posix()][
        "authority"
    ] == "CONTENT_HASH_INVENTORY"


def test_huc_verifier_replays_derived_rows_from_raw_nwis(tmp_path):
    verifier = _load_script(VERIFY_SCRIPT, "thermoroute_verify_huc_replay_test")
    source = ROOT / "data_usgs"
    target = tmp_path / "data_usgs"
    target.mkdir()
    for name in (
        "frozen_panel_v1.json",
        "station_registry_v1.csv",
        "huc_metadata_usgs_v1.csv",
        "huc_metadata_usgs_v1.provenance.json",
    ):
        shutil.copy2(source / name, target / name)
    shutil.copytree(
        source / "raw_snapshots" / "huc-v1",
        target / "raw_snapshots" / "huc-v1",
    )
    verifier.verify_canonical_huc_closure(tmp_path)

    # Forge a self-consistent registry/derived table/spec while retaining the
    # actual immutable NWIS response.  Digest-only closure would accept this;
    # raw replay must reject it.
    registry = (target / "station_registry_v1.csv").read_text(encoding="utf-8")
    registry = registry.replace(",1060003,1,55.7,", ",99060003,99,55.7,", 1)
    (target / "station_registry_v1.csv").write_text(registry, encoding="utf-8")
    huc = (target / "huc_metadata_usgs_v1.csv").read_text(encoding="utf-8")
    huc = huc.replace(",01060003,01,55.7", ",99060003,99,55.7", 1)
    (target / "huc_metadata_usgs_v1.csv").write_text(huc, encoding="utf-8")
    provenance_path = target / "huc_metadata_usgs_v1.provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["derived_csv_sha256"] = verifier.sha256_file(
        target / "huc_metadata_usgs_v1.csv"
    )
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    spec_path = target / "frozen_panel_v1.json"
    panel_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    station = panel_spec["station_registry"]
    station["sha256"] = verifier.sha256_file(target / "station_registry_v1.csv")
    station["huc_metadata"]["source_sha256"] = verifier.sha256_file(
        target / "huc_metadata_usgs_v1.csv"
    )
    station["huc_metadata"]["provenance_sha256"] = verifier.sha256_file(
        provenance_path
    )
    spec_path.write_text(json.dumps(panel_spec), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be replayed from raw NWIS"):
        verifier.verify_canonical_huc_closure(tmp_path)


def test_preopen_profile_is_explicit_and_rejects_any_result_or_label_path(tmp_path):
    verifier = _load_script(VERIFY_SCRIPT, "thermoroute_verify_preopen_profile_test")
    source, stage = tmp_path / "source", tmp_path / "stage"
    source.mkdir()
    stage.mkdir()
    _minimal_canonical_release(verifier, source)
    document = verifier.materialize_release_profile(
        source, stage, verifier.PREOPEN_PROFILE
    )
    _materialize_claim_fixture(verifier, stage, verifier.PREOPEN_PROFILE)
    assert document["profile"] == "PREOPEN_NOT_COMPLETE"
    assert document["confirmatory_scoring_completed"] is False
    assert document["directional_claims_allowed"] is False
    assert document["supported_test_ids"] == []
    assert document["supports_route_a_confirmatory_conclusions"] is False
    assert document["labels_included"] is False
    assert "cannot support" in document["warning"]
    assert document["distribution_scope"] == "LOCAL_OWNER_EVIDENCE_ONLY"
    assert (
        document["public_redistribution_authorized_by_this_release_evidence"]
        is False
    )
    assert (
        document["third_party_transfer_authorized_by_this_release_evidence"]
        is False
    )
    assert document["contains_unverified_redistribution_material"] is True
    assert set(document["known_minimum_unverified_redistribution_scopes"]) == {
        "data_usgs/**",
        verifier.GIT_BUNDLE_PATH,
        "paper/agu_submission/agujournal2019.cls",
    }
    assert document["known_unverified_scopes_are_exhaustive"] is False
    assert (
        document["rights_review_required_for_every_archive_member_by_exact_sha256"]
        is True
    )
    assert document["repository_code_license_authorizes_data"] is False
    assert document["public_profile_status"] == "BLOCKED_PENDING_RIGHTS_REVIEW"
    assert (
        document[
            "route_a_active_member_namespace_legacy_monitoring_inputs_included"
        ]
        is False
    )
    assert (
        document[
            "route_a_active_member_namespace_legacy_monitoring_outputs_included"
        ]
        is False
    )
    assert document["legacy_monitoring_case_is_route_a_scientific_evidence"] is False
    assert (
        document[
            "git_history_bundle_may_include_current_tip_legacy_monitoring_input_blobs"
        ]
        is True
    )
    assert (
        document[
            "git_history_bundle_may_include_reachable_historical_legacy_monitoring_output_blobs"
        ]
        is True
    )
    assert document["git_history_bundle_role"] == (
        "LOCAL_OWNER_GOVERNANCE_CHRONOLOGY_MAY_CONTAIN_WITHDRAWN_LEGACY_"
        "BYTES_NOT_CURRENT_ROUTE_A_SCIENTIFIC_EVIDENCE"
    )
    assert verifier.verify_release_profile(
        stage, run_trusted_replay=False
    ) == verifier.PREOPEN_PROFILE

    marker_path = stage / verifier.PROFILE_MARKER
    marker_bytes = marker_path.read_bytes()
    assert marker_bytes == verifier._canonical_json_bytes(json.loads(marker_bytes))
    unsafe_marker = json.loads(marker_bytes)
    unsafe_marker[
        "public_redistribution_authorized_by_this_release_evidence"
    ] = True
    marker_path.write_bytes(verifier._canonical_json_bytes(unsafe_marker))
    with pytest.raises(ValueError, match="exact local-only evidence profile"):
        verifier.verify_release_profile(stage, run_trusted_replay=False)
    marker_path.write_bytes(marker_bytes)

    unknown_marker = json.loads(marker_bytes)
    unknown_marker["public_release_allowed"] = True
    marker_path.write_bytes(verifier._canonical_json_bytes(unknown_marker))
    with pytest.raises(ValueError, match="unknown top-level fields"):
        verifier.verify_release_profile(stage, run_trusted_replay=False)
    marker_path.write_bytes(marker_bytes)

    duplicated = marker_bytes.replace(
        b'"public_redistribution_authorized_by_this_release_evidence":false,',
        b'"public_redistribution_authorized_by_this_release_evidence":true,'
        b'"public_redistribution_authorized_by_this_release_evidence":false,',
    )
    assert duplicated != marker_bytes
    marker_path.write_bytes(duplicated)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        verifier.verify_release_profile(stage, run_trusted_replay=False)
    marker_path.write_bytes(marker_bytes)

    for number in ("NaN", "Infinity", "-Infinity", "1e9999", "-1e9999"):
        marker_path.write_bytes(
            marker_bytes.replace(
                b"{", f'{{"strict_probe":{number},'.encode("ascii"), 1
            )
        )
        with pytest.raises(ValueError, match="non-finite JSON number"):
            verifier.verify_release_profile(stage, run_trusted_replay=False)
    marker_path.write_bytes(marker_bytes)

    for policy_field in ("forbidden_prefixes", "forbidden_path_components"):
        weakened_marker = json.loads(marker_bytes)
        weakened_marker[policy_field] = []
        marker_path.write_bytes(verifier._canonical_json_bytes(weakened_marker))
        with pytest.raises(ValueError, match="overstates its evidentiary status"):
            verifier.verify_release_profile(stage, run_trusted_replay=False)
        marker_path.write_bytes(marker_bytes)

    forbidden = _write_bytes(
        stage, "outputs/confirmatory/route_a_fake/trusted/statistics_v1.json", b"{}\n"
    )
    with pytest.raises(ValueError, match="confirmation/label/result"):
        verifier.verify_release_profile(stage, run_trusted_replay=False)
    forbidden.unlink()
    labels = _write_bytes(stage, "data_usgs/labels/post2020.parquet")
    with pytest.raises(ValueError, match="confirmation/label/result"):
        verifier.verify_release_profile(stage, run_trusted_replay=False)
    labels.unlink()


def test_public_distribution_mode_is_unconditionally_blocked(tmp_path):
    verifier = _load_script(VERIFY_SCRIPT, "thermoroute_distribution_gate_test")
    verifier.assert_distribution_mode_allowed(verifier.LOCAL_DISTRIBUTION)
    with pytest.raises(ValueError, match="public distribution is blocked"):
        verifier.assert_distribution_mode_allowed(verifier.PUBLIC_DISTRIBUTION)
    with pytest.raises(ValueError, match="public distribution is blocked"):
        verifier.verify_archive(
            tmp_path / "nonexistent.zip",
            distribution=verifier.PUBLIC_DISTRIBUTION,
        )
    environment = os.environ.copy()
    environment["THERMOROUTE_PYTHON"] = sys.executable
    completed = subprocess.run(
        ["bash", str(MAKE_RELEASE_SCRIPT), "--distribution", "PUBLIC"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "PUBLIC distribution is blocked" in completed.stderr

    verifier_cli = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT), "--distribution", "PUBLIC"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert verifier_cli.returncode == 1
    assert (
        "release verification failed: public distribution is blocked pending "
        "a byte-bound rights review"
    ) in verifier_cli.stderr

    implicit = subprocess.run(
        ["bash", str(MAKE_RELEASE_SCRIPT)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert implicit.returncode == 2
    assert "explicit --distribution is required" in implicit.stderr


def test_hosted_ci_cannot_build_or_upload_local_evidence_archive() -> None:
    workflow_paths = sorted((ROOT / ".github/workflows").glob("*.yml")) + sorted(
        (ROOT / ".github/workflows").glob("*.yaml")
    )
    assert workflow_paths
    workflows = {
        path: path.read_text(encoding="utf-8") for path in workflow_paths
    }
    combined = "\n".join(workflows.values())
    for forbidden in (
        "actions/upload-artifact",
        "softprops/action-gh-release",
        "gh release",
    ):
        assert forbidden not in combined
    builder_calls = [
        line.strip()
        for workflow in workflows.values()
        for line in workflow.splitlines()
        if "make_release_archive.sh" in line and not line.lstrip().startswith("#")
    ]
    assert builder_calls
    assert all("--distribution PUBLIC" in line for line in builder_calls)
    allowed_uses = {
        "actions/checkout@08eba0b27e820071cde6df949e0beb9ba4906955",
        "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065",
    }
    uses = {
        line.split("uses:", 1)[1].strip()
        for workflow in workflows.values()
        for line in workflow.splitlines()
        if "uses:" in line and not line.lstrip().startswith("#")
    }
    assert uses == allowed_uses
    assert (
        "github.event.pull_request.head.repo.full_name == github.repository"
        in combined
    )
    assert "persist-credentials: false" in combined
    assert "Fork pull requests are intentionally fail-closed" in combined
    assert "PUBLIC builder did not fail through the declared gate" in combined
    assert "PUBLIC verifier did not fail through the declared gate" in combined


def test_release_profile_v2_path_is_consistent_across_producers() -> None:
    verifier = _load_script(VERIFY_SCRIPT, "thermoroute_profile_path_verifier_test")
    zipper = _load_script(ZIP_SCRIPT, "thermoroute_profile_path_zipper_test")
    manifest = _load_script(MANIFEST_SCRIPT, "thermoroute_profile_path_manifest_test")
    assert verifier.PROFILE_MARKER == "evidence/release_profile_v2.json"
    assert zipper.PROFILE_MARKER == verifier.PROFILE_MARKER
    assert verifier.PROFILE_MARKER in manifest.ARTIFACT_PATTERNS


def test_postopen_profile_closes_every_required_category_and_missing_file_fails(
    tmp_path, monkeypatch
):
    verifier = _load_script(VERIFY_SCRIPT, "thermoroute_verify_postopen_profile_test")
    source, stage = tmp_path / "source", tmp_path / "stage"
    source.mkdir()
    stage.mkdir()
    authorization, representatives = _write_postopen_fixture(verifier, source)
    authorization_bytes = authorization.read_bytes()
    for runtime_attack in ("missing", "extra"):
        attacked = json.loads(authorization_bytes)
        if runtime_attack == "missing":
            attacked["runtime"].pop("installed_versions")
        else:
            attacked["runtime"]["attacker_extra"] = True
        _reseal_fixture_authorization(verifier, authorization, attacked)
        with pytest.raises(
            ValueError, match="environment attestation schema changed"
        ):
            verifier._validate_authorization_structure(source, authorization)
    authorization.write_bytes(authorization_bytes)
    authorization.write_bytes(
        authorization_bytes.replace(b"{", b'{"strict_probe":1e9999,', 1)
    )
    with pytest.raises(ValueError, match="non-finite JSON number"):
        verifier.build_release_profile(
            source,
            verifier.POSTOPEN_PROFILE,
            authorization_path=authorization,
        )
    authorization.write_bytes(authorization_bytes)
    document = verifier.materialize_release_profile(
        source,
        stage,
        verifier.POSTOPEN_PROFILE,
        authorization_path=authorization,
    )
    stage_authorization = stage / authorization.relative_to(source)
    stage_authorization_document = json.loads(
        stage_authorization.read_text(encoding="utf-8")
    )
    stage_state = stage_authorization_document["state_paths"]
    probability_path = stage / stage_state["probabilistic_evaluation"]
    receipt_path = stage / stage_state["receipt"]
    receipt_sidecar_path = stage / stage_state["receipt_sha256"]
    probability_bytes = probability_path.read_bytes()
    receipt_bytes = receipt_path.read_bytes()
    receipt_sidecar_bytes = receipt_sidecar_path.read_bytes()
    attacked_probability = json.loads(probability_bytes)
    available_row = next(
        row for row in attacked_probability["rows"]
        if row["status"] == "AVAILABLE"
    )
    available_row["pinball_q05_c"] += 1.0
    probability_path.write_bytes(
        verifier._canonical_json_bytes(attacked_probability)
    )
    _refresh_postopen_fixture_artifact_binding(
        verifier,
        stage,
        stage_authorization,
        "probabilistic_evaluation",
    )
    with pytest.raises(ValueError, match="stored-parquet recomputation"):
        verifier._gather_postopen_categories(stage, stage_authorization)
    probability_path.write_bytes(probability_bytes)
    receipt_path.write_bytes(receipt_bytes)
    receipt_sidecar_path.write_bytes(receipt_sidecar_bytes)

    _materialize_claim_fixture(verifier, stage, verifier.POSTOPEN_PROFILE)
    assert document["profile"] == "ROUTE_A_OPENED_COMPLETE"
    assert document["confirmatory_scoring_completed"] is True
    assert document["directional_claims_allowed"] is False
    assert document["inference_gate_claim_eligible"] is False
    assert document[
        "gross_plausibility_and_aggregate_sensitivity_gate_passed"
    ] is True
    assert document["supported_test_ids"] == []
    assert document["supports_route_a_confirmatory_conclusions"] is False
    assert set(document["artifact_closure"]) == verifier.REQUIRED_POSTOPEN_CATEGORIES
    replay_calls = []
    monkeypatch.setattr(
        verifier,
        "_verify_git_history_evidence",
        lambda *_args, **_kwargs: replay_calls.append("git"),
    )
    monkeypatch.setattr(
        verifier,
        "_verify_claim_audit",
        lambda *_args, **kwargs: replay_calls.append(
            f"claims:{kwargs['execute_validator']}"
        ),
    )
    monkeypatch.setattr(
        verifier,
        "_run_trusted_replay",
        lambda *_args, **_kwargs: replay_calls.append("trusted"),
    )
    assert verifier.verify_release_profile(
        stage, run_trusted_replay=True
    ) == verifier.POSTOPEN_PROFILE
    assert replay_calls == ["git", "claims:True", "trusted"]
    marker_path = stage / verifier.PROFILE_MARKER
    marker_bytes = marker_path.read_bytes()
    authorization_bytes = stage_authorization.read_bytes()
    attacked_authorization = json.loads(authorization_bytes)
    attacked_authorization["runtime"]["numerical_runtime_contract"] = {
        "fixture": True,
        "attacker": "coordinated-but-runtime-digest-left-stale",
    }
    _reseal_fixture_authorization(
        verifier, stage_authorization, attacked_authorization
    )
    attacked_marker = json.loads(marker_bytes)
    attacked_marker["authorization"]["sha256"] = verifier.sha256_file(
        stage_authorization
    )
    marker_path.write_bytes(verifier._canonical_json_bytes(attacked_marker))
    with pytest.raises(ValueError, match="runtime digest differs"):
        verifier.verify_release_profile(stage, run_trusted_replay=False)
    stage_authorization.write_bytes(authorization_bytes)
    marker_path.write_bytes(marker_bytes)

    unknown_marker = json.loads(marker_bytes)
    unknown_marker["public_release_allowed"] = True
    marker_path.write_bytes(verifier._canonical_json_bytes(unknown_marker))
    with pytest.raises(ValueError, match="unknown top-level fields"):
        verifier.verify_release_profile(stage, run_trusted_replay=False)
    marker_path.write_bytes(marker_bytes)

    altered_interface = json.loads(marker_bytes)
    altered_interface["trusted_replay_interface"]["entrypoint"] = "scripts/evil.py"
    marker_path.write_bytes(verifier._canonical_json_bytes(altered_interface))
    with pytest.raises(ValueError, match="trusted replay interface"):
        verifier.verify_release_profile(stage, run_trusted_replay=False)
    marker_path.write_bytes(marker_bytes)

    overstated = json.loads(marker_bytes)
    overstated["supports_route_a_confirmatory_conclusions"] = True
    overstated["directional_claims_allowed"] = True
    overstated["supported_test_ids"] = [
        _fixture_confirmatory_family()[0]["test_id"]
    ]
    marker_path.write_bytes(verifier._canonical_json_bytes(overstated))
    with pytest.raises(ValueError, match="claim status"):
        verifier.verify_release_profile(stage, run_trusted_replay=False)
    marker_path.write_bytes(marker_bytes)

    for category, relative in representatives.items():
        artifact = stage / relative
        payload = artifact.read_bytes()
        artifact.unlink()
        with pytest.raises(
            ValueError,
            match=(
                "absent|closure|missing|lacks|cannot read|identity|transport|"
                "checksum mismatch"
            ),
        ):
            verifier.verify_release_profile(stage, run_trusted_replay=False)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(payload)
        assert category in document["artifact_closure"]
    verifier.verify_release_profile(stage, run_trusted_replay=False)
    stale = _write_bytes(stage, "outputs/tables/usgs_scores_old_cohort.csv")
    with pytest.raises(ValueError, match="outside the authorization closure"):
        verifier.verify_release_profile(stage, run_trusted_replay=False)
    stale.unlink()


def test_postopen_coverage_replays_even_when_full_trusted_replay_is_disabled(
    tmp_path, monkeypatch
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_mandatory_coverage_replay_test"
    )
    source, stage = tmp_path / "source", tmp_path / "stage"
    source.mkdir()
    stage.mkdir()
    authorization, _ = _write_postopen_fixture(verifier, source)
    verifier.materialize_release_profile(
        source,
        stage,
        verifier.POSTOPEN_PROFILE,
        authorization_path=authorization,
    )
    _materialize_claim_fixture(verifier, stage, verifier.POSTOPEN_PROFILE)

    real_module = verifier._load_canonical_coverage_bridge_module(stage)
    real_replay_validator = verifier._validate_development_replay_document
    calls: list[str] = []

    class ReplayProxy:
        @staticmethod
        def replay_temporal_coverage_from_physical_files(**kwargs):
            calls.append("coverage")
            return real_module.replay_temporal_coverage_from_physical_files(
                **kwargs
            )

    monkeypatch.setattr(
        verifier,
        "_verify_git_history_evidence",
        lambda *_args, **_kwargs: calls.append("git"),
    )

    def validate_replay(*args, **kwargs):
        calls.append("development-replay-receipt")
        return real_replay_validator(*args, **kwargs)

    monkeypatch.setattr(
        verifier, "_validate_development_replay_document", validate_replay
    )
    monkeypatch.setattr(
        verifier,
        "_load_canonical_coverage_bridge_module",
        lambda _root: ReplayProxy,
    )
    monkeypatch.setattr(
        verifier, "_verify_claim_audit", lambda *_args, **_kwargs: None
    )

    def forbidden_trusted_replay(*_args, **_kwargs):
        raise AssertionError("full trusted replay must remain disabled")

    monkeypatch.setattr(verifier, "_run_trusted_replay", forbidden_trusted_replay)
    assert (
        verifier.verify_release_profile(stage, run_trusted_replay=False)
        == verifier.POSTOPEN_PROFILE
    )
    assert calls == ["git", "development-replay-receipt", "coverage"]


def test_postopen_archive_code_cannot_execute_before_git_identity_check(
    tmp_path, monkeypatch
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_postopen_preexec_gate_test"
    )
    source, stage = tmp_path / "source", tmp_path / "stage"
    source.mkdir()
    stage.mkdir()
    authorization, _ = _write_postopen_fixture(verifier, source)
    verifier.materialize_release_profile(
        source,
        stage,
        verifier.POSTOPEN_PROFILE,
        authorization_path=authorization,
    )
    _materialize_claim_fixture(verifier, stage, verifier.POSTOPEN_PROFILE)
    sentinel = tmp_path / "ARCHIVE_COVERAGE_PYTHON_EXECUTED"
    (stage / "src/thermoroute/coverage_bridge.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(sentinel)!r}).write_text('bad', encoding='utf-8')\n",
        encoding="utf-8",
    )

    def reject_git(*_args, **_kwargs):
        raise ValueError("Git identity rejected before archive import")

    monkeypatch.setattr(verifier, "_verify_git_history_evidence", reject_git)
    with pytest.raises(ValueError, match="before archive import"):
        verifier.verify_release_profile(stage, run_trusted_replay=False)
    assert not sentinel.exists()


def test_postopen_authorization_requires_exact_coverage_state_registry(
    tmp_path,
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_coverage_state_registry_test"
    )
    source = tmp_path / "source"
    source.mkdir()
    authorization_path, _ = _write_postopen_fixture(verifier, source)
    for attack in ("missing", "extra"):
        attacked = tmp_path / attack
        shutil.copytree(source, attacked)
        path = attacked / authorization_path.relative_to(source)
        authorization = json.loads(path.read_text(encoding="utf-8"))
        if attack == "missing":
            authorization["state_paths"].pop("temporal_coverage_audit")
        else:
            authorization["state_paths"]["coverage_alias"] = (
                authorization["state_paths"]["temporal_coverage_audit"]
            )
        created_at = authorization.pop("created_at_utc")
        authorization.pop("opening_id")
        authorization.pop("authorization_self_sha256")
        authorization["opening_id"] = verifier._sha256_json(authorization)[:24]
        authorization["created_at_utc"] = created_at
        authorization["authorization_self_sha256"] = verifier._sha256_json(
            authorization
        )
        path.write_text(json.dumps(authorization), encoding="utf-8")
        with pytest.raises(ValueError, match="state-path registry changed"):
            verifier._validate_authorization_structure(attacked, path)


def test_postopen_authorization_requires_exact_top_level_schema_and_opening_id(
    tmp_path,
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_authorization_identity_attacks_test"
    )
    source = tmp_path / "source"
    source.mkdir()
    authorization_path, _ = _write_postopen_fixture(verifier, source)

    for attack in ("missing", "extra"):
        attacked_root = tmp_path / f"authorization-{attack}"
        shutil.copytree(source, attacked_root)
        attacked_path = attacked_root / authorization_path.relative_to(source)
        authorization = json.loads(attacked_path.read_text(encoding="utf-8"))
        if attack == "missing":
            authorization.pop("actual_feature_order")
        else:
            authorization["attacker_extension"] = {"accepted": True}
        authorization.pop("authorization_self_sha256")
        authorization["authorization_self_sha256"] = verifier._sha256_json(
            authorization
        )
        attacked_path.write_text(json.dumps(authorization), encoding="utf-8")
        with pytest.raises(ValueError, match="top-level schema changed"):
            verifier._validate_authorization_structure(
                attacked_root, attacked_path
            )

    attacked_root = tmp_path / "authorization-opening-id"
    shutil.copytree(source, attacked_root)
    attacked_path = attacked_root / authorization_path.relative_to(source)
    authorization = json.loads(attacked_path.read_text(encoding="utf-8"))
    authorization["opening_id"] = "f" * 24
    authorization.pop("authorization_self_sha256")
    authorization["authorization_self_sha256"] = verifier._sha256_json(
        authorization
    )
    attacked_path.write_text(json.dumps(authorization), encoding="utf-8")
    with pytest.raises(ValueError, match="identity is inconsistent"):
        verifier._validate_authorization_structure(attacked_root, attacked_path)


def test_postopen_receipt_requires_exact_coverage_artifact_registry(
    tmp_path,
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_coverage_receipt_registry_test"
    )
    source = tmp_path / "source"
    source.mkdir()
    authorization_path, _ = _write_postopen_fixture(verifier, source)
    for attack in ("missing", "extra"):
        attacked = tmp_path / f"receipt-{attack}"
        shutil.copytree(source, attacked)
        attacked_authorization = attacked / authorization_path.relative_to(source)
        authorization = json.loads(
            attacked_authorization.read_text(encoding="utf-8")
        )
        state = authorization["state_paths"]
        receipt_path = attacked / state["receipt"]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if attack == "missing":
            receipt["artifacts"].pop("temporal_coverage_audit")
            receipt["release_bindings"]["artifacts"].pop(
                "temporal_coverage_audit"
            )
        else:
            receipt["artifacts"]["coverage_alias"] = dict(
                receipt["artifacts"]["temporal_coverage_audit"]
            )
            receipt["release_bindings"]["artifacts"]["coverage_alias"] = {
                "format": verifier.TEMPORAL_COVERAGE_AUDIT_FORMAT,
                **receipt["artifacts"]["coverage_alias"],
            }
        _reseal_postopen_fixture_receipt(
            verifier, attacked, state, receipt
        )
        with pytest.raises(ValueError, match="artifact registry"):
            verifier.build_release_profile(
                attacked,
                verifier.POSTOPEN_PROFILE,
                authorization_path=attacked_authorization,
            )


def test_postopen_release_rejects_recovery_contract_missing_tamper_and_extra(
    tmp_path,
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_recovery_contract_attacks_test"
    )
    source = tmp_path / "source"
    source.mkdir()
    authorization_path, _ = _write_postopen_fixture(verifier, source)
    amendment_relative = "protocols/route_a_inference_amendment_v2.json"
    amendment = json.loads(
        (source / amendment_relative).read_text(encoding="utf-8")
    )
    assert (
        amendment["trusted_scoring_recovery_contract"]
        == verifier.TRUSTED_SCORING_RECOVERY_CONTRACT
    )
    _validate_fixture_inference_closure(verifier, source, authorization_path)

    for attack in ("missing", "tamper", "extra"):
        attacked = tmp_path / f"recovery-{attack}"
        shutil.copytree(source, attacked)
        attacked_amendment_path = attacked / amendment_relative
        attacked_amendment = json.loads(
            attacked_amendment_path.read_text(encoding="utf-8")
        )
        recovery = attacked_amendment["trusted_scoring_recovery_contract"]
        if attack == "missing":
            recovery.pop("maximum_frozen_request_ledgers_per_opening")
        elif attack == "tamper":
            recovery["maximum_logical_openings"] = 2
        else:
            recovery["unfrozen_recovery_extension"] = True
        attacked_amendment_path.write_text(
            json.dumps(attacked_amendment), encoding="utf-8"
        )
        with pytest.raises(
            ValueError,
            match="trusted-scoring recovery contract changed",
        ):
            _validate_fixture_inference_closure(
                verifier,
                attacked,
                attacked / authorization_path.relative_to(source),
            )


def test_fast_release_requires_exact_opening_transport_document_schemas(
    tmp_path,
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_transport_top_level_schema_test"
    )
    source = tmp_path / "source"
    source.mkdir()
    authorization_path, _ = _write_postopen_fixture(verifier, source)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    state = authorization["state_paths"]
    attacks = (
        ("work-order-missing", state["work_order"], "source_tree_sha256"),
        ("work-order-extra", state["work_order"], None),
        ("intent-missing", state["intent"], "started_at_utc"),
        ("intent-extra", state["intent"], None),
        ("receipt-missing", state["receipt"], "security_boundary"),
        ("receipt-extra", state["receipt"], None),
        (
            "acquisition-missing",
            state["acquisition_manifest"],
            "protocol_sha256",
        ),
        ("acquisition-extra", state["acquisition_manifest"], None),
    )
    for attack, relative, missing_key in attacks:
        attacked = tmp_path / attack
        shutil.copytree(source, attacked)
        attacked_authorization = attacked / authorization_path.relative_to(source)
        path = attacked / relative
        document = json.loads(path.read_text(encoding="utf-8"))
        if missing_key is None:
            document["unfrozen_schema_extension"] = True
        else:
            document.pop(missing_key)
        if relative == state["work_order"]:
            document.pop("work_order_self_sha256", None)
            document["work_order_self_sha256"] = verifier._sha256_json(document)
            path.write_bytes(verifier._canonical_json_bytes(document))
        elif relative == state["intent"]:
            document.pop("intent_self_sha256", None)
            document["intent_self_sha256"] = verifier._sha256_json(document)
            path.write_bytes(verifier._canonical_json_bytes(document))
        elif relative == state["receipt"]:
            _reseal_postopen_fixture_receipt(
                verifier, attacked, state, document
            )
        else:
            path.write_bytes(verifier._canonical_json_bytes(document))
            _refresh_postopen_transport_evidence(
                verifier, attacked, attacked_authorization
            )
        with pytest.raises(
            ValueError,
            match="exact|schema|work-order|acquisition manifest",
        ):
            verifier.build_release_profile(
                attacked,
                verifier.POSTOPEN_PROFILE,
                authorization_path=attacked_authorization,
            )


def test_fast_release_rejects_forged_transport_chain_attacks(tmp_path):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_transport_chain_attacks_test"
    )
    source = tmp_path / "source"
    source.mkdir()
    authorization_path, _ = _write_postopen_fixture(verifier, source)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    state = authorization["state_paths"]
    verifier.build_release_profile(
        source,
        verifier.POSTOPEN_PROFILE,
        authorization_path=authorization_path,
    )

    for attack in (
        "forged",
        "missing",
        "extra",
        "reordered",
        "duplicate",
        "hash",
        "path",
        "partition",
        "attempt-chain",
        "attempt-time-format",
        "retrieval-time-format",
        "series",
        "oversize",
        "json-whitespace",
        "json-duplicate-key",
        "rdb-agency",
        "rdb-site",
        "rdb-date",
        "rdb-duplicate",
        "raw-normalized",
        "normalized",
        "normalized-tiny",
    ):
        attacked = tmp_path / f"transport-{attack}"
        shutil.copytree(source, attacked)
        attacked_authorization = attacked / authorization_path.relative_to(source)
        acquisition_path = attacked / state["acquisition_manifest"]
        acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
        ledger_path = attacked / acquisition["request_ledger"]["path"]
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if attack == "forged":
            ledger["requests"][0]["request"]["url"] += "&forged=1"
            ledger["requests"][0]["request_sha256"] = hashlib.sha256(
                verifier._canonical_json_bytes(
                    ledger["requests"][0]["request"]
                )
            ).hexdigest()
        elif attack == "missing":
            ledger.pop("request_order")
        elif attack == "extra":
            ledger["unfrozen_ledger_extension"] = True
        elif attack == "reordered":
            ledger["requests"] = list(reversed(ledger["requests"]))
        elif attack == "duplicate":
            ledger["requests"].append(dict(ledger["requests"][0]))
            ledger["request_count"] = len(ledger["requests"])
        elif attack == "path":
            alias = (
                Path(state["acquisition_manifest"]).parent
                / "request_ledger_alias_v1.json"
            ).as_posix()
            shutil.copy2(ledger_path, attacked / alias)
            acquisition["request_ledger"] = _binding(
                verifier, attacked, alias
            )
            acquisition_path.write_bytes(
                verifier._canonical_json_bytes(acquisition)
            )
        elif attack in {
            "partition",
            "attempt-chain",
            "attempt-time-format",
        }:
            index_path = attacked / acquisition["transport_attempt_index"]["path"]
            index = json.loads(index_path.read_text(encoding="utf-8"))
            row = index["attempts"][0 if attack == "partition" else 1]
            start_path = attacked / row["start"]["path"]
            start = json.loads(start_path.read_text(encoding="utf-8"))
            if attack == "partition":
                start["missing_at_start_request_sha256"].append(
                    start["missing_at_start_request_sha256"][0]
                )
            elif attack == "attempt-chain":
                all_requests = sorted(
                    item["request_sha256"] for item in ledger["requests"]
                )
                start["completed_before_attempt_request_sha256"] = []
                start["missing_at_start_request_sha256"] = all_requests
            else:
                start["started_at_utc"] = "2026-01-01T00:05:00Z"
            start.pop("attempt_start_self_sha256")
            start["attempt_start_self_sha256"] = hashlib.sha256(
                verifier._canonical_json_bytes(start)
            ).hexdigest()
            start_path.write_bytes(verifier._canonical_json_bytes(start))
        elif attack in {
            "retrieval-time-format",
            "series",
            "oversize",
            "rdb-agency",
            "rdb-site",
            "rdb-date",
            "rdb-duplicate",
            "raw-normalized",
        }:
            snapshot_path = attacked / acquisition["raw_nwis_snapshot_index"]["path"]
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            record = snapshot["records"][0]
            request_map_path = attacked / acquisition["request_map"]["path"]
            request_map = json.loads(request_map_path.read_text(encoding="utf-8"))
            request_row = next(
                row for row in request_map["requests"]
                if row["request_sha256"] == record["request_sha256"]
            )
            raw_root = attacked / state["raw_nwis_root"]
            metadata_path = raw_root / record["metadata_path"]
            response_path = raw_root / record["response_path"]
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if attack == "retrieval-time-format":
                forged_time = "2025-12-31T23:59:59Z"
                metadata["retrieved_at_utc"] = forged_time
                record["retrieved_at_utc"] = forged_time
                request_row["retrieved_at_utc"] = forged_time
            elif attack == "series":
                forged_series = {
                    "WTEMP": [{
                        "parameter_code": "00010",
                        "value_column": "FORGED_00010_00003",
                        "qualifier_column": None,
                    }],
                    "FLOW": [],
                    "WLEVEL": [],
                }
                record["series_registry"] = forged_series
                request_row["series_registry"] = forged_series
            elif attack == "oversize":
                payload = b"x" * (
                    verifier.MAX_CONFIRMATORY_NWIS_RESPONSE_BYTES + 1
                )
                response_path.write_bytes(payload)
            else:
                payload_lines = response_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                first_data = 3
                fields = payload_lines[first_data].split("\t")
                if attack == "rdb-agency":
                    fields[0] = "FAKE"
                elif attack == "rdb-site":
                    fields[1] = "99999999"
                elif attack == "rdb-date":
                    fields[2] = "2019-12-31"
                elif attack == "raw-normalized":
                    fields[3] = f"{float(fields[3]) + 1.0:.3f}"
                payload_lines[first_data] = "\t".join(fields)
                if attack == "rdb-duplicate":
                    payload_lines.insert(first_data + 1, payload_lines[first_data])
                payload = ("\n".join(payload_lines) + "\n").encode("utf-8")
                response_path.write_bytes(payload)
            if attack in {
                "oversize",
                "rdb-agency",
                "rdb-site",
                "rdb-date",
                "rdb-duplicate",
                "raw-normalized",
            }:
                metadata["byte_count"] = len(payload)
                metadata["response_sha256"] = hashlib.sha256(payload).hexdigest()
                record["byte_count"] = len(payload)
                record["response_sha256"] = metadata["response_sha256"]
                request_row["byte_count"] = len(payload)
                request_row["response_sha256"] = metadata["response_sha256"]
            metadata_path.write_bytes(verifier._canonical_json_bytes(metadata))
            record["metadata_sha256"] = verifier.sha256_file(metadata_path)
            snapshot_path.write_bytes(verifier._canonical_json_bytes(snapshot))
            request_map_path.write_bytes(
                verifier._canonical_json_bytes(request_map)
            )
        elif attack in {"normalized", "normalized-tiny"}:
            normalized_path = attacked / state["temporal_outcomes"]
            normalized = pd.read_parquet(normalized_path)
            increment = 1.0 if attack == "normalized" else 5e-13
            normalized.loc[0, "WTEMP"] = (
                float(normalized.loc[0, "WTEMP"]) + increment
            )
            normalized.to_parquet(normalized_path, index=False)
        elif attack == "json-whitespace":
            ledger_path.write_text(
                json.dumps(ledger, indent=2) + "\n", encoding="utf-8"
            )
        elif attack == "json-duplicate-key":
            canonical = verifier._canonical_json_bytes(ledger).decode("utf-8")
            duplicate = (
                "{\"format\":"
                + json.dumps(ledger["format"])
                + ","
                + canonical[1:]
            )
            ledger_path.write_text(duplicate, encoding="utf-8")
        elif attack == "hash":
            ledger["request_ledger_self_sha256"] = "0" * 64

        if attack not in {
            "path", "partition", "attempt-chain", "attempt-time-format",
            "retrieval-time-format", "series", "oversize", "hash",
            "json-whitespace", "json-duplicate-key", "rdb-agency",
            "rdb-site", "rdb-date", "rdb-duplicate", "raw-normalized",
            "normalized", "normalized-tiny",
        }:
            ledger.pop("request_ledger_self_sha256", None)
            ledger["request_ledger_self_sha256"] = hashlib.sha256(
                verifier._canonical_json_bytes(ledger)
            ).hexdigest()
            ledger_path.write_bytes(verifier._canonical_json_bytes(ledger))
        elif attack == "hash":
            ledger_path.write_bytes(verifier._canonical_json_bytes(ledger))
        _refresh_postopen_transport_evidence(
            verifier, attacked, attacked_authorization
        )
        with pytest.raises(
            ValueError,
            match=(
                "transport|request ledger|request-ledger|attempt|partition|"
                "canonical|exact contract|raw NWIS|response|normalized"
            ),
        ):
            verifier.build_release_profile(
                attacked,
                verifier.POSTOPEN_PROFILE,
                authorization_path=attacked_authorization,
            )


def test_transport_wall_clock_reversal_does_not_override_logical_chain(tmp_path):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_transport_wall_clock_test"
    )
    source = tmp_path / "source"
    source.mkdir()
    authorization_path, _ = _write_postopen_fixture(verifier, source)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    state = authorization["state_paths"]
    acquisition = json.loads(
        (source / state["acquisition_manifest"]).read_text(encoding="utf-8")
    )
    index = json.loads(
        (
            source / acquisition["transport_attempt_index"]["path"]
        ).read_text(encoding="utf-8")
    )
    second = index["attempts"][1]
    start_path = source / second["start"]["path"]
    result_path = source / second["result"]["path"]
    start = json.loads(start_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    start["started_at_utc"] = "2025-12-31T23:59:59+00:00"
    result["completed_at_utc"] = "2025-01-01T00:00:00+00:00"
    start.pop("attempt_start_self_sha256")
    start["attempt_start_self_sha256"] = hashlib.sha256(
        verifier._canonical_json_bytes(start)
    ).hexdigest()
    start_path.write_bytes(verifier._canonical_json_bytes(start))
    result["attempt_start_sha256"] = verifier.sha256_file(start_path)
    result.pop("attempt_result_self_sha256")
    result["attempt_result_self_sha256"] = hashlib.sha256(
        verifier._canonical_json_bytes(result)
    ).hexdigest()
    result_path.write_bytes(verifier._canonical_json_bytes(result))
    _refresh_postopen_transport_evidence(
        verifier, source, authorization_path
    )

    verifier.build_release_profile(
        source,
        verifier.POSTOPEN_PROFILE,
        authorization_path=authorization_path,
    )


def test_postopen_coverage_replay_rejects_tamper_and_path_topology_attacks(
    tmp_path,
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_coverage_physical_attacks_test"
    )
    source = tmp_path / "source"
    source.mkdir()
    authorization_path, _ = _write_postopen_fixture(verifier, source)
    for attack in (
        "audit_tamper",
        "prediction_tamper",
        "receipt_path_alias",
        "prediction_symlink",
        "prediction_hardlink",
    ):
        attacked = tmp_path / attack
        shutil.copytree(source, attacked)
        attacked_authorization = attacked / authorization_path.relative_to(source)
        authorization = json.loads(
            attacked_authorization.read_text(encoding="utf-8")
        )
        state = authorization["state_paths"]
        receipt_path = attacked / state["receipt"]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if attack == "audit_tamper":
            audit_path = attacked / state["temporal_coverage_audit"]
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["status"] = "FORGED_PASS"
            audit.pop("audit_self_sha256")
            audit["audit_self_sha256"] = verifier._sha256_json(audit)
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            binding = _binding(
                verifier, attacked, state["temporal_coverage_audit"]
            )
            receipt["artifacts"]["temporal_coverage_audit"] = binding
            receipt["release_bindings"]["artifacts"][
                "temporal_coverage_audit"
            ].update(binding)
            receipt["temporal_coverage_audit"].update(binding)
        elif attack == "prediction_tamper":
            prediction_path = attacked / state["temporal_predictions"]
            frame = pd.read_parquet(prediction_path)
            frame.loc[0, "y_pred"] = float(frame.loc[0, "y_pred"]) + 0.5
            frame.to_parquet(prediction_path, index=False)
            binding = _binding(
                verifier, attacked, state["temporal_predictions"]
            )
            receipt["artifacts"]["temporal_predictions"] = binding
            receipt["release_bindings"]["artifacts"][
                "temporal_predictions"
            ].update(binding)
        elif attack == "receipt_path_alias":
            binding = dict(receipt["artifacts"]["external_predictions"])
            receipt["artifacts"]["temporal_predictions"] = binding
            receipt["release_bindings"]["artifacts"][
                "temporal_predictions"
            ].update(binding)
        else:
            prediction_path = attacked / state["temporal_predictions"]
            backup = tmp_path / f"{attack}-target.parquet"
            shutil.copy2(prediction_path, backup)
            prediction_path.unlink()
            if attack == "prediction_symlink":
                prediction_path.symlink_to(backup)
            else:
                os.link(backup, prediction_path)
        _reseal_postopen_fixture_receipt(
            verifier, attacked, state, receipt
        )
        with pytest.raises(
            ValueError,
            match=(
                "temporal-coverage|coverage|canonical|leaves|escapes|"
                "regular|unsafe|hard-linked"
            ),
        ):
            verifier.build_release_profile(
                attacked,
                verifier.POSTOPEN_PROFILE,
                authorization_path=attacked_authorization,
            )


def test_postopen_release_rejects_self_consistent_nested_outcome_gate_forgery(
    tmp_path,
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_nested_outcome_gate_attack_test"
    )
    source = tmp_path / "source"
    source.mkdir()
    authorization_path, _ = _write_postopen_fixture(verifier, source)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    state = authorization["state_paths"]

    gate_path = source / state["outcome_qc_gate"]
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["single_extreme_influence"][0]["absolute_effect_change_c"] += 0.01
    gate.pop("gate_self_sha256")
    gate["gate_self_sha256"] = verifier._sha256_json(gate)
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    statistics_path = source / state["statistics"]
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    statistics["outcome_qc_gate"]["sha256"] = verifier.sha256_file(gate_path)
    statistics_path.write_text(json.dumps(statistics), encoding="utf-8")

    receipt_path = source / state["receipt"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for label, path in (
        ("outcome_qc_gate", gate_path),
        ("statistics", statistics_path),
    ):
        digest = verifier.sha256_file(path)
        receipt["artifacts"][label]["sha256"] = digest
        receipt["release_bindings"]["artifacts"][label]["sha256"] = digest
    receipt.pop("receipt_self_sha256")
    receipt["receipt_self_sha256"] = verifier._sha256_json(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_digest = verifier.sha256_file(receipt_path)
    (source / state["receipt_sha256"]).write_text(
        f"{receipt_digest}  opening_receipt_v1.json\n", encoding="utf-8"
    )
    _refresh_postopen_coverage_evidence(
        verifier, source, authorization_path
    )

    with pytest.raises(ValueError, match="outcome-QC gate semantics changed"):
        verifier.build_release_profile(
            source,
            verifier.POSTOPEN_PROFILE,
            authorization_path=authorization_path,
        )


def test_postopen_release_distinguishes_transport_resume_from_second_opening(
    tmp_path,
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_transport_completion_test"
    )
    source = tmp_path / "source"
    source.mkdir()
    authorization_path, _ = _write_postopen_fixture(verifier, source)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    state = authorization["state_paths"]
    acquisition_path = source / state["acquisition_manifest"]
    receipt_path = source / state["receipt"]
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    resumed_transport = dict(acquisition["transport_summary"])
    assert resumed_transport["opening_count"] == 1
    assert resumed_transport["attempt_count"] == 2
    assert resumed_transport["resume_count"] == 1
    verifier.build_release_profile(
        source,
        verifier.POSTOPEN_PROFILE,
        authorization_path=authorization_path,
    )

    second_opening = {**resumed_transport, "opening_count": 2}
    acquisition["transport_summary"] = second_opening
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")
    _refresh_postopen_transport_evidence(
        verifier, source, authorization_path
    )
    with pytest.raises(ValueError, match="transport summaries differ"):
        verifier.build_release_profile(
            source,
            verifier.POSTOPEN_PROFILE,
            authorization_path=authorization_path,
        )

    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    acquisition["transport_summary"] = resumed_transport
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")
    _refresh_postopen_transport_evidence(
        verifier, source, authorization_path
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["same_opening_transport_resume_allowed"] = False
    _reseal_postopen_fixture_receipt(
        verifier, source, state, receipt
    )
    with pytest.raises(ValueError, match="receipt exact production schema"):
        verifier.build_release_profile(
            source,
            verifier.POSTOPEN_PROFILE,
            authorization_path=authorization_path,
        )


def test_release_verifier_requires_both_receipts_and_exact_control_members(
    tmp_path, monkeypatch,
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_preopening_control_gates_test"
    )
    source = tmp_path / "source"
    source.mkdir()
    authorization_path, _representatives = _write_postopen_fixture(verifier, source)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    suite_path = source / authorization["model_suite"]["path"]
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    development = suite["development_contract"]
    original_paired_recompute = verifier._stage09b_recompute_paired_effects
    paired_recompute_calls: list[int] = []

    def observed_paired_recompute(station_metrics):
        paired_recompute_calls.append(len(set(zip(
            station_metrics["arm_id"].astype(str),
            station_metrics["seed"].astype(int),
            strict=True,
        ))))
        return original_paired_recompute(station_metrics)

    monkeypatch.setattr(
        verifier, "_stage09b_recompute_paired_effects", observed_paired_recompute,
    )
    verifier._validate_preopening_completion_gates(
        source, {}, suite, development, suite["numerical_runtime_sha256"]
    )
    assert paired_recompute_calls == [len(verifier._stage09b_release_members())]

    missing = json.loads(json.dumps(suite))
    missing["preopening_gates"].pop("stage09b_development_controls")
    with pytest.raises(ValueError, match="Stage-9/09b"):
        verifier._validate_preopening_completion_gates(
            source, {}, missing, development, suite["numerical_runtime_sha256"]
        )
    missing_stage25 = json.loads(json.dumps(suite))
    missing_stage25["preopening_gates"].pop("stage25_external_completion")
    with pytest.raises(ValueError, match="Stage-9/09b/16/25"):
        verifier._validate_preopening_completion_gates(
            source,
            {},
            missing_stage25,
            development,
            suite["numerical_runtime_sha256"],
        )
    missing_stage16 = json.loads(json.dumps(suite))
    missing_stage16["preopening_gates"].pop("stage16_lstm_completion")
    with pytest.raises(ValueError, match="Stage-9/09b/16/25"):
        verifier._validate_preopening_completion_gates(
            source,
            {},
            missing_stage16,
            development,
            suite["numerical_runtime_sha256"],
        )

    stage16_binding = suite["preopening_gates"]["stage16_lstm_completion"]
    stage16_path = source / stage16_binding["path"]
    original_stage16 = stage16_path.read_bytes()
    tampered_stage16 = json.loads(original_stage16)
    tampered_stage16["selection_audit"]["candidates"][1][
        "best_state_max_abs_difference"
    ] = 1.0
    tampered_stage16.pop("receipt_self_sha256")
    tampered_stage16["receipt_self_sha256"] = verifier._sha256_json(
        tampered_stage16
    )
    _write_canonical_json(
        verifier, source, stage16_binding["path"], tampered_stage16
    )
    tampered_stage16_suite = json.loads(json.dumps(suite))
    tampered_stage16_suite["preopening_gates"][
        "stage16_lstm_completion"
    ] = _binding(verifier, source, stage16_binding["path"])
    with pytest.raises(ValueError, match="candidate replay audit changed"):
        verifier._validate_preopening_completion_gates(
            source,
            {},
            tampered_stage16_suite,
            development,
            suite["numerical_runtime_sha256"],
        )
    stage16_path.write_bytes(original_stage16)

    stage25_binding = suite["preopening_gates"]["stage25_external_completion"]
    stage25_path = source / stage25_binding["path"]
    original_stage25 = stage25_path.read_bytes()
    tampered_stage25 = json.loads(original_stage25)
    tampered_stage25["status"] = "INCOMPLETE"
    stage25_path.write_text(json.dumps(tampered_stage25), encoding="utf-8")
    tampered_suite = json.loads(json.dumps(suite))
    tampered_suite["preopening_gates"]["stage25_external_completion"] = _binding(
        verifier, source, stage25_binding["path"]
    )
    with pytest.raises(ValueError, match="Stage-25 completion receipt is malformed"):
        verifier._validate_preopening_completion_gates(
            source,
            {},
            tampered_suite,
            development,
            suite["numerical_runtime_sha256"],
        )
    stage25_path.write_bytes(original_stage25)

    def bind_stage25_receipt(document: dict[str, object]) -> dict[str, object]:
        candidate = json.loads(json.dumps(document))
        candidate.pop("receipt_self_sha256", None)
        candidate["artifact_closure_sha256"] = verifier._sha256_json(
            candidate["artifacts"]
        )
        candidate["receipt_self_sha256"] = verifier._sha256_json(candidate)
        _write_canonical_json(
            verifier, source, stage25_binding["path"], candidate
        )
        candidate_suite = json.loads(json.dumps(suite))
        candidate_suite["preopening_gates"][
            "stage25_external_completion"
        ] = _binding(verifier, source, stage25_binding["path"])
        return candidate_suite

    pristine_stage25 = json.loads(original_stage25)
    sidecar_relative = pristine_stage25["artifacts"][
        "development_prediction_sidecar"
    ]["path"]
    sidecar_path = source / sidecar_relative
    original_sidecar = sidecar_path.read_bytes()
    malformed_sidecar = json.loads(original_sidecar)
    malformed_sidecar["content_schema"] = "attacker.predictions.v1"
    _write_canonical_json(verifier, source, sidecar_relative, malformed_sidecar)
    sidecar_receipt = json.loads(json.dumps(pristine_stage25))
    sidecar_receipt["artifacts"][
        "development_prediction_sidecar"
    ] = _binding(verifier, source, sidecar_relative)
    sidecar_suite = bind_stage25_receipt(sidecar_receipt)
    with pytest.raises(ValueError, match="Stage-25 prediction sidecar changed"):
        verifier._validate_preopening_completion_gates(
            source,
            {},
            sidecar_suite,
            development,
            suite["numerical_runtime_sha256"],
        )
    sidecar_path.write_bytes(original_sidecar)
    stage25_path.write_bytes(original_stage25)

    missing_seed = json.loads(json.dumps(pristine_stage25))
    missing_seed["formal_configuration"]["seeds"].pop()
    missing_seed_suite = bind_stage25_receipt(missing_seed)
    with pytest.raises(ValueError, match="formal configuration changed"):
        verifier._validate_preopening_completion_gates(
            source,
            {},
            missing_seed_suite,
            development,
            suite["numerical_runtime_sha256"],
        )
    stage25_path.write_bytes(original_stage25)

    missing_grid = json.loads(json.dumps(pristine_stage25))
    missing_grid["formal_configuration"]["lstm_validation_grid"].pop()
    missing_grid_suite = bind_stage25_receipt(missing_grid)
    with pytest.raises(ValueError, match="formal configuration changed"):
        verifier._validate_preopening_completion_gates(
            source,
            {},
            missing_grid_suite,
            development,
            suite["numerical_runtime_sha256"],
        )
    stage25_path.write_bytes(original_stage25)

    padding_relative = "outputs/models/unrelated_stage25_padding.bin"
    _write_bytes(source, padding_relative, b"unrelated padding\n")
    padded_receipt = json.loads(json.dumps(pristine_stage25))
    padded_model_files = padded_receipt["artifacts"]["model_files"]
    padded_model_files.pop()
    padded_model_files.append(_binding(verifier, source, padding_relative))
    padded_model_files.sort(key=lambda binding: binding["path"])
    padded_suite = bind_stage25_receipt(padded_receipt)
    with pytest.raises(ValueError, match="differ from canonical components"):
        verifier._validate_preopening_completion_gates(
            source,
            {},
            padded_suite,
            development,
            suite["numerical_runtime_sha256"],
        )
    stage25_path.write_bytes(original_stage25)

    controls_binding = suite["preopening_gates"]["stage09b_development_controls"]
    controls_path = source / controls_binding["path"]
    controls = json.loads(controls_path.read_text(encoding="utf-8"))
    controls["member_registry"].pop()
    stable = {
        key: value for key, value in controls.items()
        if key != "receipt_self_sha256"
    }
    controls["receipt_self_sha256"] = verifier._sha256_json(stable)
    controls_path.write_text(json.dumps(controls), encoding="utf-8")
    suite["preopening_gates"]["stage09b_development_controls"] = _binding(
        verifier, source, controls_binding["path"]
    )
    with pytest.raises(ValueError, match="matrix audit|declared members"):
        verifier._validate_preopening_completion_gates(
            source, {}, suite, development, suite["numerical_runtime_sha256"]
        )


def test_stage16_release_and_git_reject_three_view_winner_disagreement(
    tmp_path,
) -> None:
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_stage16_three_view_winner_test"
    )
    source = tmp_path / "source"
    source.mkdir()
    authorization_path, _ = _write_postopen_fixture(verifier, source)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    suite_path = source / authorization["model_suite"]["path"]
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    baseline_commit = _commit_all_fixture_files(source, "valid Stage-16 fixture")
    baseline_gates = suite["preopening_gates"]
    verifier._git_stage16_dependency_paths(
        source,
        baseline_commit,
        suite,
        baseline_gates["stage16_lstm_completion"],
        stage9_gate_binding=baseline_gates["stage09_completion"],
        development_input_closure_sha256=(
            FIXTURE_DEVELOPMENT_INPUT_CLOSURE_SHA256
        ),
        development_input_closure_file_count=(
            FIXTURE_DEVELOPMENT_INPUT_CLOSURE_FILE_COUNT
        ),
    )
    gate = suite["preopening_gates"]["stage16_lstm_completion"]
    receipt = json.loads((source / gate["path"]).read_text(encoding="utf-8"))
    candidates = receipt["selection_audit"]["candidates"]
    # Each change remains within 1e-5 of the reported CSV metric, but the
    # recomputed view now selects candidate 0 while the other views select 1.
    candidates[0]["recomputed_val_station_macro_rmse"] = 0.200009
    candidates[1]["recomputed_val_station_macro_rmse"] = 0.200001
    tampered_suite = _bind_stage16_receipt_fixture(
        verifier, source, suite, receipt
    )
    with pytest.raises(ValueError, match="three-view validation winner changed"):
        verifier._validate_preopening_completion_gates(
            source,
            {},
            tampered_suite,
            tampered_suite["development_contract"],
            tampered_suite["numerical_runtime_sha256"],
        )

    subprocess.run(["git", "add", "-f", "-A"], cwd=source, check=True)
    commit = _commit_git_fixture(source, "three-view winner mismatch")
    gates = tampered_suite["preopening_gates"]
    with pytest.raises(ValueError, match="three-view validation winner changed"):
        verifier._git_stage16_dependency_paths(
            source,
            commit,
            tampered_suite,
            gates["stage16_lstm_completion"],
            stage9_gate_binding=gates["stage09_completion"],
            development_input_closure_sha256=(
                FIXTURE_DEVELOPMENT_INPUT_CLOSURE_SHA256
            ),
            development_input_closure_file_count=(
                FIXTURE_DEVELOPMENT_INPUT_CLOSURE_FILE_COUNT
            ),
        )


@pytest.mark.parametrize("mode", ("missing_scheduler_state", "nonterminal"))
def test_stage16_release_and_git_reject_invalid_checkpoint_payload(
    tmp_path, mode: str,
) -> None:
    verifier = _load_script(
        VERIFY_SCRIPT, f"thermoroute_verify_stage16_checkpoint_{mode}_test"
    )
    source = tmp_path / "source"
    source.mkdir()
    authorization_path, _ = _write_postopen_fixture(verifier, source)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    suite_path = source / authorization["model_suite"]["path"]
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    tampered_suite = _mutate_stage16_checkpoint_fixture(
        verifier, source, suite, candidate_id=0, mode=mode
    )
    error = "top-level fields changed|structure or terminal state changed"
    with pytest.raises(ValueError, match=error):
        verifier._validate_preopening_completion_gates(
            source,
            {},
            tampered_suite,
            tampered_suite["development_contract"],
            tampered_suite["numerical_runtime_sha256"],
        )

    commit = _commit_all_fixture_files(source, f"invalid checkpoint {mode}")
    gates = tampered_suite["preopening_gates"]
    with pytest.raises(ValueError, match=error):
        verifier._git_stage16_dependency_paths(
            source,
            commit,
            tampered_suite,
            gates["stage16_lstm_completion"],
            stage9_gate_binding=gates["stage09_completion"],
            development_input_closure_sha256=(
                FIXTURE_DEVELOPMENT_INPUT_CLOSURE_SHA256
            ),
            development_input_closure_file_count=(
                FIXTURE_DEVELOPMENT_INPUT_CLOSURE_FILE_COUNT
            ),
        )


@pytest.mark.parametrize(
    "attack",
    (
        "seed_bool", "seed_float", "horizon_bool", "horizon_float",
        "site_integer", "string_date", "timezone", "intraday",
        "y_true_bool", "y_pred_integer", "q05_integer", "q50_bool",
        "q95_integer", "probability_bool",
    ),
)
def test_independent_release_normaliser_rejects_coercion_aliases(
    attack: str,
) -> None:
    verifier = _load_script(
        VERIFY_SCRIPT, f"thermoroute_verify_stage09b_{attack}_alias_test",
    )
    records = []
    for split_index, split in enumerate(("val", "calib", "test")):
        for horizon in (1, 3, 7):
            issue = pd.Timestamp("2017-01-01") + pd.Timedelta(days=split_index)
            records.append({
                "model": "PlainMLP-7var",
                "scope": "development_only_2006_2020",
                "feature_set": "all_7",
                "seed": 0,
                "site_id": "12345678",
                "horizon": horizon,
                "split": split,
                "issue_date": issue,
                "target_date": issue + pd.Timedelta(days=horizon),
                "y_true": 1.0,
                "y_pred": 1.1,
                "q05": 0.5,
                "q50": 1.1,
                "q95": 1.5,
                "p_exceed": 0.25,
            })
    frame = pd.DataFrame.from_records(records)
    if attack == "seed_bool":
        frame["seed"] = False
    elif attack == "seed_float":
        frame["seed"] = frame["seed"].astype("float64")
    elif attack == "horizon_bool":
        frame["horizon"] = frame["horizon"].astype(object)
        frame.loc[frame["horizon"].eq(1), "horizon"] = True
    elif attack == "horizon_float":
        frame["horizon"] = frame["horizon"].astype("float64")
    elif attack == "site_integer":
        frame["site_id"] = 12_345_678
    elif attack == "string_date":
        frame["issue_date"] = frame["issue_date"].dt.strftime("%Y-%m-%d")
    elif attack == "timezone":
        frame["issue_date"] = frame["issue_date"].dt.tz_localize("UTC")
        frame["target_date"] = frame["target_date"].dt.tz_localize("UTC")
    elif attack == "intraday":
        frame["issue_date"] += pd.Timedelta(hours=1)
        frame["target_date"] += pd.Timedelta(hours=1)
    elif attack == "y_true_bool":
        frame["y_true"] = False
    elif attack == "y_pred_integer":
        frame["y_pred"] = 1
    elif attack == "q05_integer":
        frame["q05"] = 0
    elif attack == "q50_bool":
        frame["q50"] = True
    elif attack == "q95_integer":
        frame["q95"] = 2
    else:
        frame["p_exceed"] = False
    with pytest.raises(ValueError, match="Stage-09b prediction"):
        verifier._normalise_stage09b_release_prediction(
            frame,
            arm_id="PlainMLP-7var",
            seed=0,
            feature_set="all_7",
            reference=None,
        )


@pytest.mark.parametrize("attack", ("string_date", "y_true_bool", "q50_integer"))
def test_independent_release_rejects_noncanonical_prediction_arrow_types(
    tmp_path: Path, attack: str,
) -> None:
    verifier = _load_script(
        VERIFY_SCRIPT, f"thermoroute_verify_stage09b_{attack}_arrow_test",
    )
    issue = pd.Timestamp("2017-01-01")
    frame = pd.DataFrame([{
        "model": "PlainMLP-7var",
        "scope": "development_only_2006_2020",
        "feature_set": "all_7",
        "seed": 0,
        "site_id": "12345678",
        "horizon": 1,
        "split": "test",
        "issue_date": issue,
        "target_date": issue + pd.Timedelta(days=1),
        "y_true": 1.0,
        "y_pred": 1.1,
        "q05": 0.5,
        "q50": 1.1,
        "q95": 1.5,
        "p_exceed": 0.25,
    }])
    if attack == "string_date":
        frame["issue_date"] = frame["issue_date"].dt.strftime("%Y-%m-%d")
    elif attack == "y_true_bool":
        frame["y_true"] = False
    else:
        frame["q50"] = 1
    path = tmp_path / "attacked.parquet"
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="Arrow type"):
        verifier._stage09b_assert_prediction_arrow_schema(path)


@pytest.mark.parametrize("attack", ("split", "station", "seed", "horizon"))
def test_independent_release_paired_recompute_rejects_registry_attacks(
    attack: str,
) -> None:
    verifier = _load_script(
        VERIFY_SCRIPT, f"thermoroute_verify_stage09b_{attack}_registry_test",
    )
    rows = [
        {
            "arm_id": arm_id,
            "seed": seed,
            "split": split,
            "horizon": horizon,
            "site_id": site,
            "forecast_keys": 2,
            "station_rmse_c": 1.0 + seed / 100,
        }
        for arm_id, seed in verifier._stage09b_release_members()
        for split in ("calib", "test", "val")
        for horizon in (1, 3, 7)
        for site in ("01234567", "12345678")
    ]
    station = pd.DataFrame.from_records(rows)
    if attack == "split":
        station = station.loc[~station["split"].eq("val")].copy()
    elif attack == "station":
        mask = (
            station["arm_id"].eq("ThermoRoute-ladder-07_plus_WDSP")
            & station["seed"].eq(0)
            & station["split"].eq("test")
            & station["horizon"].eq(1)
            & station["site_id"].eq("12345678")
        )
        station.loc[mask, "site_id"] = "99999999"
    elif attack == "seed":
        station = station.loc[
            ~(
                station["arm_id"].eq("PlainMLP-7var")
                & station["seed"].eq(4)
            )
        ].copy()
    else:
        station.loc[station["horizon"].eq(7), "horizon"] = 5
    with pytest.raises(ValueError, match="registry|declared members"):
        verifier._stage09b_recompute_paired_effects(station)


def test_release_lineage_hash_matches_live_opening_under_non_ascii_root(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    from thermoroute.repro import sha256_json

    verifier = _load_script(VERIFY_SCRIPT, "thermoroute_verify_unicode_hash_test")
    source = tmp_path / "[副业]论文" / "source"
    stage = tmp_path / "[副业]论文" / "stage"
    source.mkdir(parents=True)
    stage.mkdir(parents=True)
    authorization_path, _ = _write_postopen_fixture(verifier, source)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    stable = dict(authorization)
    claimed = stable.pop("authorization_self_sha256")

    assert claimed == sha256_json(stable)
    assert verifier._sha256_json(stable) == sha256_json(stable)
    document = verifier.materialize_release_profile(
        source,
        stage,
        verifier.POSTOPEN_PROFILE,
        authorization_path=authorization_path,
    )
    assert document["profile"] == verifier.POSTOPEN_PROFILE


def test_deterministic_zip_normalises_order_timestamp_modes_and_manifest_time(tmp_path):
    verifier = _load_script(VERIFY_SCRIPT, "thermoroute_verify_zip_shape_test")
    zipper = _load_script(ZIP_SCRIPT, "thermoroute_deterministic_zip_test")
    stage = tmp_path / "stage"
    _write_bytes(stage, "scripts/tool.py", b"print('x')\n")
    _write_bytes(stage, "data/value.txt", b"value\n")
    opened_root = (
        "outputs/confirmatory/route_a_" + "a" * 24
    )
    opened_receipt = _write_bytes(
        stage, f"{opened_root}/opening_receipt_v1.json", b"{}\n"
    )
    opened_statistic = _write_bytes(
        stage, f"{opened_root}/trusted/statistics_v1.json", b"{}\n"
    )
    manifest = _write_bytes(
        stage,
        "outputs/manifest.json",
        json.dumps({"generated_utc": "2099-01-01T00:00:00+00:00"}).encode(),
    )
    revision = {
        "compute_commit": "a" * 40,
        "manuscript_commit": "b" * 40,
        "committed_document_diff": [],
    }
    claim_validation = {"path": "evidence/claims.json", "sha256": "c" * 64}
    history_evidence = {"bundle": {"path": "evidence/history.bundle"}}
    lock_binding = {
        "path": "requirements-lock-py312-hashed.txt",
        "sha256": "d" * 64,
    }
    profile_marker = {
        "profile": verifier.POSTOPEN_PROFILE,
        **verifier.LOCAL_EVIDENCE_DISTRIBUTION_FIELDS,
        "authorized_worktree_dirt_policy": revision,
        "claim_validation": claim_validation,
        "git_history_evidence": history_evidence,
        "artifact_closure": {"reproducibility_lock": lock_binding},
    }
    _write_bytes(
        stage,
        verifier.PROFILE_MARKER,
        json.dumps(profile_marker).encode(),
    )
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    zipper.create_deterministic_zip(stage, first)
    first_sha = hashlib.sha256(first.read_bytes()).hexdigest()

    os.chmod(stage / "scripts/tool.py", 0o600)
    os.chmod(stage / "data/value.txt", 0o777)
    os.chmod(opened_receipt, 0o777)
    os.chmod(opened_statistic, 0o600)
    os.chmod(stage / opened_root / "trusted", 0o777)
    os.utime(stage / "data/value.txt", (2_000_000_000, 2_000_000_000))
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["generated_utc"] = "2100-01-01T00:00:00+00:00"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    zipper.create_deterministic_zip(stage, second)
    assert hashlib.sha256(second.read_bytes()).hexdigest() == first_sha

    with zipfile.ZipFile(second) as archive:
        members = verifier.normalised_members(archive)
        assert {"scripts/tool.py", "data/value.txt", "outputs/manifest.json"} <= members
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
        modes = {
            info.filename: (info.external_attr >> 16) & 0o777
            for info in archive.infolist()
        }
        assert modes["thermoroute/scripts/tool.py"] == 0o755
        assert modes["thermoroute/data/value.txt"] == 0o644
        assert modes[f"thermoroute/{opened_root}/"] == 0o555
        assert modes[f"thermoroute/{opened_root}/trusted/"] == 0o555
        assert modes[f"thermoroute/{opened_root}/opening_receipt_v1.json"] == 0o444
        assert modes[f"thermoroute/{opened_root}/trusted/statistics_v1.json"] == 0o444
        archived_manifest = json.loads(
            archive.read("thermoroute/outputs/manifest.json")
        )
        assert archived_manifest["release_revision"] == revision
        assert archived_manifest["release_evidence"] == {
            "profile": verifier.POSTOPEN_PROFILE,
            "distribution": {
                key: profile_marker.get(key)
                for key in verifier.LOCAL_EVIDENCE_DISTRIBUTION_FIELDS
            },
            "claim_validation": claim_validation,
            "git_history_evidence": history_evidence,
            "reproducibility_lock": lock_binding,
        }


@pytest.mark.parametrize(
    ("target_suffix", "attacked_mode", "error"),
    (
        (
            "/trusted/statistics_v1.json",
            stat.S_IFREG | 0o644,
            "non-canonical mode",
        ),
        ("/trusted/", stat.S_IFDIR | 0o755, "non-canonical mode"),
        (
            "/trusted/statistics_v1.json",
            stat.S_IFLNK | 0o777,
            "symbolic links",
        ),
    ),
)
def test_archive_rejects_writable_or_linked_opened_namespace_metadata(
    tmp_path, target_suffix: str, attacked_mode: int, error: str,
) -> None:
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_opened_mode_attack_test"
    )
    zipper = _load_script(
        ZIP_SCRIPT, "thermoroute_deterministic_opened_mode_attack_test"
    )
    stage = tmp_path / "stage"
    opened_root = "outputs/confirmatory/route_a_" + "b" * 24
    _write_bytes(
        stage, f"{opened_root}/trusted/statistics_v1.json", b"{}\n"
    )
    good = tmp_path / "good.zip"
    attacked = tmp_path / "attacked.zip"
    zipper.create_deterministic_zip(stage, good)
    with zipfile.ZipFile(good) as source, zipfile.ZipFile(
        attacked, "w"
    ) as destination:
        matched = False
        for info in source.infolist():
            payload = source.read(info.filename)
            if info.filename.endswith(target_suffix):
                info.external_attr = attacked_mode << 16
                matched = True
            destination.writestr(info, payload)
    assert matched
    extraction = tmp_path / "extraction"
    extraction.mkdir()
    with zipfile.ZipFile(attacked) as archive:
        with pytest.raises(ValueError, match=error):
            verifier._extract_archive_safely(archive, extraction)
    assert not any(extraction.iterdir())


def test_postopen_archive_modes_satisfy_real_immutable_directory_validators(
    tmp_path,
) -> None:
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_opened_mode_integration_test"
    )
    zipper = _load_script(
        ZIP_SCRIPT, "thermoroute_deterministic_opened_mode_integration_test"
    )
    stage = tmp_path / "stage"
    run_relative = "outputs/confirmatory/route_a_" + "c" * 24
    acquisition_names = {
        "acquisition_request_map": "source_request_map_v1.json",
        "temporal_outcomes": "temporal_outcomes_v1.parquet",
        "external_outcomes": "external_outcomes_v1.parquet",
        "acquisition_manifest": "acquisition_manifest_v1.json",
    }
    trusted_names = {
        "availability_registry": "availability_registry_v1.csv",
        "outcome_quality_audit": "outcome_quality_audit_v1.json",
        "outcome_qc_gate": "outcome_qc_gate_v1.json",
        "approved_target_sensitivity": "approved_target_sensitivity_v1.json",
        "spatial_sensitivity": "spatial_sensitivity_v1.json",
        "probabilistic_evaluation": "probabilistic_evaluation_v2.json",
        "temporal_predictions": "temporal_predictions_v1.parquet",
        "external_predictions": "external_predictions_v1.parquet",
        "statistics": "statistics_v1.json",
        "temporal_coverage_audit": "temporal_coverage_audit_v1.json",
        "report": "report_v1.md",
    }
    for name in acquisition_names.values():
        _write_bytes(stage, f"{run_relative}/acquisition/{name}")
    for name in trusted_names.values():
        _write_bytes(stage, f"{run_relative}/trusted/{name}")
    _write_bytes(stage, "data/ordinary.txt")

    archive_path = tmp_path / "opened.zip"
    zipper.create_deterministic_zip(stage, archive_path)
    extraction = tmp_path / "extraction"
    extraction.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        verifier._extract_archive_safely(archive, extraction)
    root = extraction / "thermoroute"
    run = root / run_relative
    assert stat.S_IMODE(run.stat().st_mode) == 0o555
    assert stat.S_IMODE((run / "acquisition").stat().st_mode) == 0o555
    assert stat.S_IMODE((run / "trusted").stat().st_mode) == 0o555
    assert stat.S_IMODE((root / "data").stat().st_mode) == 0o755
    assert stat.S_IMODE((root / "data/ordinary.txt").stat().st_mode) == 0o644

    state = {
        "run_directory": run,
        **{
            key: run / "acquisition" / name
            for key, name in acquisition_names.items()
        },
        **{
            key: run / "trusted" / name
            for key, name in trusted_names.items()
        },
    }
    for key in (*acquisition_names, *trusted_names):
        metadata = state[key].stat()
        assert stat.S_IMODE(metadata.st_mode) == 0o444
        assert metadata.st_nlink == 1

    def snapshot() -> dict[str, tuple[int, int, str | None]]:
        observed: dict[str, tuple[int, int, str | None]] = {}
        for path in sorted(run.rglob("*")):
            metadata = path.lstat()
            observed[path.relative_to(run).as_posix()] = (
                stat.S_IMODE(metadata.st_mode),
                metadata.st_nlink,
                hashlib.sha256(path.read_bytes()).hexdigest()
                if path.is_file()
                else None,
            )
        return observed

    before = snapshot()
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from thermoroute import opening, outcome_acquisition

        opening._assert_exact_trusted_directory(run / "trusted", state)
        outcome_acquisition._assert_exact_acquisition_directory(
            run / "acquisition", state
        )
    finally:
        sys.path.pop(0)
    assert snapshot() == before


def _write_exact_archive_member_fixture(verifier, stage: Path) -> dict[str, object]:
    """Create the smallest tree accepted by the exact member-set helper."""
    fixed = set(verifier.REQUIRED_MEMBERS) | set(verifier.ALLOWED_PAPER_MEMBERS)
    fixed.add(".gitignore")
    for relative in sorted(fixed):
        _write_bytes(stage, relative, f"fixture:{relative}\n".encode())
    marker = {
        "format": "thermoroute.release-profile.v1",
        "profile": verifier.PREOPEN_PROFILE,
        "artifact_closure": {},
    }
    _write_bytes(
        stage,
        verifier.PROFILE_MARKER,
        json.dumps(marker, sort_keys=True).encode() + b"\n",
    )
    _write_bytes(
        stage,
        "outputs/manifest.json",
        json.dumps({"generated_utc": "2099-01-01T00:00:00+00:00"}).encode(),
    )
    return marker


@pytest.mark.parametrize(
    ("extra", "forge_manifest"),
    (
        ("evidence/evil.bin", True),
        ("notes/unsupported_results.txt", False),
        ("data/secret_confirmation_labels.csv", False),
        ("unsupported_results.txt", False),
    ),
)
def test_exact_archive_layout_rejects_unregistered_files_even_with_new_manifest(
    tmp_path, extra: str, forge_manifest: bool,
) -> None:
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_exact_archive_file_test"
    )
    zipper = _load_script(
        ZIP_SCRIPT, "thermoroute_deterministic_exact_archive_file_test"
    )
    stage = tmp_path / "stage"
    marker = _write_exact_archive_member_fixture(verifier, stage)
    baseline = tmp_path / "baseline.zip"
    zipper.create_deterministic_zip(stage, baseline)
    with zipfile.ZipFile(baseline) as archive:
        members, directories = verifier._normalised_archive_layout(archive)
    verifier.validate_members(members)
    verifier._validate_exact_release_member_layout(
        stage, marker, members, directories
    )

    extra_path = _write_bytes(stage, extra, b"unregistered evidence\n")
    if forge_manifest:
        _write_bytes(
            stage,
            "outputs/manifest.json",
            json.dumps({
                "generated_utc": "2099-01-01T00:00:00+00:00",
                "forged_public_inventory": {
                    extra: hashlib.sha256(extra_path.read_bytes()).hexdigest()
                },
            }).encode(),
        )
    attacked = tmp_path / "attacked.zip"
    zipper.create_deterministic_zip(stage, attacked)
    with zipfile.ZipFile(attacked) as archive:
        members, directories = verifier._normalised_archive_layout(archive)
    if extra.startswith("data/"):
        with pytest.raises(ValueError, match="forbidden non-Route-A direct input"):
            verifier.validate_members(members)
        return
    # Canonical but unregistered paths are rejected by the exact-set layer.
    verifier.validate_members(members)
    with pytest.raises(ValueError, match="exact authorized set"):
        verifier._validate_exact_release_member_layout(
            stage, marker, members, directories
        )


def test_exact_archive_layout_rejects_unregistered_empty_directory(
    tmp_path,
) -> None:
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_exact_archive_directory_test"
    )
    zipper = _load_script(
        ZIP_SCRIPT, "thermoroute_deterministic_exact_archive_directory_test"
    )
    stage = tmp_path / "stage"
    marker = _write_exact_archive_member_fixture(verifier, stage)
    (stage / "notes/empty").mkdir(parents=True)
    attacked = tmp_path / "empty-directory.zip"
    zipper.create_deterministic_zip(stage, attacked)
    with zipfile.ZipFile(attacked) as archive:
        members, directories = verifier._normalised_archive_layout(archive)
    verifier.validate_members(members)
    with pytest.raises(ValueError, match="exact file parents"):
        verifier._validate_exact_release_member_layout(
            stage, marker, members, directories
        )


def test_release_manifest_revision_must_match_verified_bundle_commit_and_tree(
    tmp_path,
) -> None:
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_manifest_bundle_revision_test"
    )
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    _write_bytes(source, "tracked.txt", b"committed bytes\n")
    commit = _commit_git_fixture(source, "fixture revision")
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=source,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    bare = tmp_path / "audit.git"
    subprocess.run(
        ["git", "clone", "-q", "--bare", str(source), str(bare)],
        check=True,
    )
    release = tmp_path / "release"
    manifest_path = release / "outputs" / "manifest.json"
    expected_git = {
        "available": True,
        "commit": commit,
        "tree": tree,
        "dirty": False,
        "dirty_paths": [],
        "source": "release-builder",
    }
    _write_bytes(
        release,
        "outputs/manifest.json",
        json.dumps({"git": expected_git}, sort_keys=True).encode() + b"\n",
    )
    verifier._verify_manifest_revision_from_bundle(
        root=release,
        bare=bare,
        manuscript_commit=commit,
        profile=verifier.PREOPEN_PROFILE,
    )

    for field, forged in (("commit", "f" * 40), ("tree", "e" * 40)):
        attacked = {"git": {**expected_git, field: forged}}
        manifest_path.write_text(json.dumps(attacked) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="commit/tree differs"):
            verifier._verify_manifest_revision_from_bundle(
                root=release,
                bare=bare,
                manuscript_commit=commit,
                profile=verifier.PREOPEN_PROFILE,
            )

    attacked = {"git": {**expected_git, "dirty": True}}
    manifest_path.write_text(json.dumps(attacked) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="commit/tree differs"):
        verifier._verify_manifest_revision_from_bundle(
            root=release,
            bare=bare,
            manuscript_commit=commit,
            profile=verifier.PREOPEN_PROFILE,
        )


def test_git_bundle_replays_sealed_protocol_after_release_relocation(
    tmp_path, monkeypatch
):
    verifier = _load_script(VERIFY_SCRIPT, "thermoroute_verify_git_evidence_test")
    source, stage = tmp_path / "source", tmp_path / "stage"
    source.mkdir()
    stage.mkdir()
    protocol_relative = "protocols/route_a_confirmatory_protocol.md"
    protocol = _write_bytes(source, protocol_relative, b"# original fixture protocol\n")
    original_bytes = protocol.read_bytes()
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "add", protocol_relative], cwd=source, check=True)
    environment = os.environ.copy()
    environment.update({
        "GIT_AUTHOR_NAME": "Fixture",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_NAME": "Fixture",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
    })
    subprocess.run(
        ["git", "commit", "-q", "-m", "original protocol"],
        cwd=source,
        env=environment,
        check=True,
    )
    original_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    protocol.write_bytes(b"# final prelabel fixture protocol\n")
    protocol_json = _write_bytes(
        source,
        "protocols/route_a_confirmatory_v1.json",
        json.dumps({
            "protocol_id": "route-a-confirmatory-v1",
            "authoritative_protocol_commit": original_commit,
            "primary_inference_contract": {
                "confirmatory_family": _fixture_confirmatory_family(),
                "probabilistic_event_contract": json.loads(
                    (ROOT / "protocols/route_a_confirmatory_v1.json").read_text(
                        encoding="utf-8"
                    )
                )["primary_inference_contract"]["probabilistic_event_contract"],
            },
        }, sort_keys=True).encode() + b"\n",
    )
    subprocess.run(
        ["git", "add", protocol_relative, "protocols/route_a_confirmatory_v1.json"],
        cwd=source, check=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "final prelabel protocol"],
        cwd=source, env=environment, check=True,
    )
    final_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    seal_document = {
        "format": verifier.PROTOCOL_SEAL_FORMAT,
        "status": "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED",
        "protocol_id": "route-a-confirmatory-v1",
        "original_preregistration": {
            "commit": original_commit,
            "markdown": {
                "path": protocol_relative,
                "sha256": hashlib.sha256(original_bytes).hexdigest(),
            },
        },
        "final_prelabel_protocol": {
            "commit": final_commit,
            "json": {
                "path": "protocols/route_a_confirmatory_v1.json",
                "sha256": hashlib.sha256(protocol_json.read_bytes()).hexdigest(),
            },
            "markdown": {
                "path": protocol_relative,
                "sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
            },
        },
        "prelabel_attestation": {
            "external_timestamp_or_public_preregistration": False,
            "independent_custodian_or_worm_storage": False,
        },
    }
    seal_path = _write_bytes(
        source,
        verifier.PROTOCOL_SEAL_PATH,
        json.dumps(seal_document, sort_keys=True).encode() + b"\n",
    )
    subprocess.run(
        ["git", "add", verifier.PROTOCOL_SEAL_PATH], cwd=source, check=True
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "mechanical protocol seal"],
        cwd=source, env=environment, check=True,
    )
    amendment_path = _write_canonical_json(
        verifier,
        source,
        verifier.INFERENCE_AMENDMENT_PATH,
        {"fixture": "outcome-free inference amendment"},
    )
    amendment_commit = _commit_git_fixture(
        source, "outcome-free inference amendment"
    )
    amendment_seal_path = _write_canonical_json(
        verifier,
        source,
        verifier.INFERENCE_AMENDMENT_SEAL_PATH,
        {
            "format": "thermoroute.route-a-inference-amendment-seal.v2",
            "status": "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED",
            "amendment": _binding(
                verifier, source, verifier.INFERENCE_AMENDMENT_PATH
            ),
            "final_prelabel_commit": amendment_commit,
        },
    )
    _commit_git_fixture(source, "separate inference amendment seal")
    erratum = json.loads(
        (ROOT / verifier.PROBABILITY_METRIC_ERRATUM_PATH).read_text(
            encoding="utf-8"
        )
    )
    erratum["governance_inputs"] = {
        "base_protocol": _binding(
            verifier, source, "protocols/route_a_confirmatory_v1.json"
        ),
        "base_protocol_seal": _binding(
            verifier, source, verifier.PROTOCOL_SEAL_PATH
        ),
        "inference_amendment": _binding(
            verifier, source, verifier.INFERENCE_AMENDMENT_PATH
        ),
        "inference_amendment_seal": _binding(
            verifier, source, verifier.INFERENCE_AMENDMENT_SEAL_PATH
        ),
    }
    erratum["scientific_scope"]["confirmatory_family_sha256"] = (
        verifier._sha256_json(_fixture_confirmatory_family())
    )
    erratum_path = _write_canonical_json(
        verifier, source, verifier.PROBABILITY_METRIC_ERRATUM_PATH, erratum
    )
    erratum_document_commit = _commit_git_fixture(
        source, "probability metric erratum document"
    )
    erratum_seal_path = _write_canonical_json(
        verifier,
        source,
        verifier.PROBABILITY_METRIC_ERRATUM_SEAL_PATH,
        {
            "format": verifier.PROBABILITY_METRIC_ERRATUM_SEAL_FORMAT,
            "status": "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED",
            "erratum_id": verifier.PROBABILITY_METRIC_ERRATUM_ID,
            "erratum": _binding(
                verifier, source, verifier.PROBABILITY_METRIC_ERRATUM_PATH
            ),
            "governance_seals": {
                "base_protocol_seal": _binding(
                    verifier, source, verifier.PROTOCOL_SEAL_PATH
                ),
                "inference_amendment_seal": _binding(
                    verifier, source, verifier.INFERENCE_AMENDMENT_SEAL_PATH
                ),
            },
            "erratum_document_commit": erratum_document_commit,
            "history_contract": {
                "governance_seal_commits_must_be_ancestors": True,
                "erratum_blob_must_match_document_commit": True,
                "erratum_document_created_exactly_once": True,
                "document_commit_must_precede_seal_commit": True,
                "seal_created_exactly_once": True,
                "erratum_and_seal_immutable_to_release_tip": True,
            },
            "prelabel_attestation": {
                "post_2020_wtemp_requested_or_inspected": False,
                "confirmation_outcomes_requested_or_inspected": False,
                "outcome_endpoint_called": False,
                "outcome_independent": True,
                "network_used": False,
            },
        },
    )
    _commit_git_fixture(source, "separate probability metric erratum seal")
    model_matrix_attestation = {
        "post_2020_wtemp_requested_or_inspected": False,
        "confirmation_outcomes_requested_or_inspected": False,
        "confirmation_outcome_artifact_present": False,
        "outcome_endpoint_called": False,
        "outcome_independent": True,
        "network_used": False,
    }
    model_matrix_path = _write_canonical_json(
        verifier,
        source,
        verifier.MODEL_MATRIX_AMENDMENT_PATH,
        {
            "format": verifier.MODEL_MATRIX_AMENDMENT_FORMAT,
            "status": verifier.MODEL_MATRIX_AMENDMENT_STATUS,
            "amendment_id": verifier.MODEL_MATRIX_AMENDMENT_ID,
            "prelabel_attestation": model_matrix_attestation,
            "fixture": "outcome-free model matrix",
        },
    )
    model_matrix_document_commit = _commit_git_fixture(
        source, "outcome-free model-matrix amendment"
    )
    model_matrix_seal_path = _write_canonical_json(
        verifier,
        source,
        verifier.MODEL_MATRIX_AMENDMENT_SEAL_PATH,
        {
            "format": verifier.MODEL_MATRIX_AMENDMENT_SEAL_FORMAT,
            "status": verifier.MODEL_MATRIX_AMENDMENT_SEAL_STATUS,
            "amendment_id": verifier.MODEL_MATRIX_AMENDMENT_ID,
            "amendment": _binding(
                verifier, source, verifier.MODEL_MATRIX_AMENDMENT_PATH
            ),
            "amendment_document_commit": model_matrix_document_commit,
            "governance_seals": {
                "base_protocol_seal": _binding(
                    verifier, source, verifier.PROTOCOL_SEAL_PATH
                ),
                "inference_amendment_seal_v2": _binding(
                    verifier, source, verifier.INFERENCE_AMENDMENT_SEAL_PATH
                ),
                "probability_metric_erratum_seal_v1": _binding(
                    verifier,
                    source,
                    verifier.PROBABILITY_METRIC_ERRATUM_SEAL_PATH,
                ),
            },
            "history_contract": {
                "governance_seal_commits_must_be_strict_ancestors": True,
                "amendment_blob_must_match_document_commit": True,
                "amendment_document_created_exactly_once": True,
                "document_commit_must_precede_seal_commit": True,
                "seal_created_exactly_once": True,
                "amendment_and_seal_immutable_to_release_tip": True,
            },
            "prelabel_attestation": model_matrix_attestation,
        },
    )
    _commit_git_fixture(source, "separate model-matrix amendment seal")
    compute_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    _write_bytes(
        source, verifier.POSTOPEN_CLAIM_DOCUMENT, b"later documentation\n"
    )
    subprocess.run(
        ["git", "add", verifier.POSTOPEN_CLAIM_DOCUMENT],
        cwd=source,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "later docs"],
        cwd=source,
        env=environment,
        check=True,
    )

    _write_bytes(
        stage,
        verifier.PROFILE_MARKER,
        json.dumps({"profile": verifier.PREOPEN_PROFILE}).encode(),
    )
    for source_path, relative in (
        (protocol_json, "protocols/route_a_confirmatory_v1.json"),
        (protocol, protocol_relative),
        (seal_path, verifier.PROTOCOL_SEAL_PATH),
        (amendment_path, verifier.INFERENCE_AMENDMENT_PATH),
        (amendment_seal_path, verifier.INFERENCE_AMENDMENT_SEAL_PATH),
            (erratum_path, verifier.PROBABILITY_METRIC_ERRATUM_PATH),
            (erratum_seal_path, verifier.PROBABILITY_METRIC_ERRATUM_SEAL_PATH),
            (model_matrix_path, verifier.MODEL_MATRIX_AMENDMENT_PATH),
            (
                model_matrix_seal_path,
                verifier.MODEL_MATRIX_AMENDMENT_SEAL_PATH,
            ),
        ):
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)

    # This legacy fixture isolates portable protocol/Git replay and uses a tiny
    # synthetic model-matrix document. Dedicated tests below exercise the real
    # fixed whole-file matrix root and its exact Git document/seal chronology.
    def accept_synthetic_matrix(root, authorization):
        del authorization
        fixture_root = Path(root)
        document = fixture_root / verifier.MODEL_MATRIX_AMENDMENT_PATH
        seal = fixture_root / verifier.MODEL_MATRIX_AMENDMENT_SEAL_PATH
        return document, seal, verifier.sha256_file(seal)

    monkeypatch.setattr(
        verifier,
        "_validate_model_matrix_amendment_binding",
        accept_synthetic_matrix,
    )
    evidence = verifier.materialize_git_history_evidence(
        source, stage, verifier.PREOPEN_PROFILE
    )
    assert evidence["sealed_protocol_blob"]["sha256"] == hashlib.sha256(
        original_bytes
    ).hexdigest()
    assert evidence["final_prelabel_protocol_commit"] == final_commit
    assert evidence["external_timestamp_or_public_preregistration"] is False

    relocated = tmp_path / "a-different-absolute-path" / "release"
    shutil.copytree(stage, relocated)
    marker = json.loads(
        (relocated / verifier.PROFILE_MARKER).read_text(encoding="utf-8")
    )
    verifier._verify_git_history_evidence(
        relocated, marker, verifier.PREOPEN_PROFILE
    )

    # This fixture predates the chronology receipt and intentionally isolates
    # protocol replay plus the compute-to-manuscript document diff.  The
    # dedicated real-Git test below exercises the complete chronology verifier.
    monkeypatch.setattr(
        verifier,
        "_verify_prelabel_chronology_from_bundle",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        verifier,
        "_verify_authorized_compute_tree_from_bundle",
        lambda **_kwargs: None,
    )

    manuscript_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    manuscript_path = relocated / verifier.POSTOPEN_CLAIM_DOCUMENT
    manuscript_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / verifier.POSTOPEN_CLAIM_DOCUMENT, manuscript_path)
    authorization_relative = "data_usgs/opening_authorization.json"
    _write_bytes(
        relocated,
        authorization_relative,
        json.dumps({
            "protocol": {
                "authoritative_markdown_sha256": hashlib.sha256(original_bytes).hexdigest(),
                "final_prelabel_commit": final_commit,
                "seal": _binding(verifier, relocated, verifier.PROTOCOL_SEAL_PATH),
            },
        }).encode(),
    )
    document_binding = {
        **_binding(verifier, relocated, verifier.POSTOPEN_CLAIM_DOCUMENT),
        "bytes": manuscript_path.stat().st_size,
    }
    marker["profile"] = verifier.POSTOPEN_PROFILE
    marker["authorization"] = {"path": authorization_relative}
    marker["git_history_evidence"]["profile"] = verifier.POSTOPEN_PROFILE
    marker["git_history_evidence"]["compute_commit"] = compute_commit
    marker["git_history_evidence"]["manuscript_commit"] = manuscript_commit
    marker["authorized_worktree_dirt_policy"] = {
        "committed_document_diff": [document_binding],
    }
    verifier._verify_git_history_evidence(
        relocated, marker, verifier.POSTOPEN_PROFILE
    )

    missing_binding = json.loads(json.dumps(marker))
    missing_binding["authorized_worktree_dirt_policy"][
        "committed_document_diff"
    ] = []
    with pytest.raises(ValueError, match="diff differs"):
        verifier._verify_git_history_evidence(
            relocated, missing_binding, verifier.POSTOPEN_PROFILE
        )
    extra_binding = json.loads(json.dumps(marker))
    extra_binding["authorized_worktree_dirt_policy"][
        "committed_document_diff"
    ].append({"path": "paper/extra.md", "sha256": "1" * 64, "bytes": 1})
    with pytest.raises(ValueError, match="diff differs"):
        verifier._verify_git_history_evidence(
            relocated, extra_binding, verifier.POSTOPEN_PROFILE
        )
    wrong_blob = json.loads(json.dumps(marker))
    wrong_blob["authorized_worktree_dirt_policy"]["committed_document_diff"][0][
        "sha256"
    ] = "2" * 64
    with pytest.raises(ValueError, match="manuscript blob differs"):
        verifier._verify_git_history_evidence(
            relocated, wrong_blob, verifier.POSTOPEN_PROFILE
        )

    _write_bytes(source, "src/hidden_compute_change.py", b"hidden = True\n")
    subprocess.run(["git", "add", "src/hidden_compute_change.py"], cwd=source, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "forbidden hidden compute change"],
        cwd=source, env=environment, check=True,
    )
    (source / "src/hidden_compute_change.py").unlink()
    subprocess.run(["git", "add", "-A"], cwd=source, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "hide forbidden change by reverting it"],
        cwd=source, env=environment, check=True,
    )
    hidden_stage = tmp_path / "hidden-stage"
    _write_bytes(
        hidden_stage,
        verifier.PROFILE_MARKER,
        json.dumps({"profile": verifier.PREOPEN_PROFILE}).encode(),
    )
    for source_path, relative in (
        (protocol_json, "protocols/route_a_confirmatory_v1.json"),
        (protocol, protocol_relative),
        (seal_path, verifier.PROTOCOL_SEAL_PATH),
        (amendment_path, verifier.INFERENCE_AMENDMENT_PATH),
        (amendment_seal_path, verifier.INFERENCE_AMENDMENT_SEAL_PATH),
            (erratum_path, verifier.PROBABILITY_METRIC_ERRATUM_PATH),
            (erratum_seal_path, verifier.PROBABILITY_METRIC_ERRATUM_SEAL_PATH),
            (model_matrix_path, verifier.MODEL_MATRIX_AMENDMENT_PATH),
            (
                model_matrix_seal_path,
                verifier.MODEL_MATRIX_AMENDMENT_SEAL_PATH,
            ),
        ):
        destination = hidden_stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
    hidden_evidence = verifier.materialize_git_history_evidence(
        source, hidden_stage, verifier.PREOPEN_PROFILE
    )
    hidden_manuscript = hidden_stage / verifier.POSTOPEN_CLAIM_DOCUMENT
    hidden_manuscript.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        source / verifier.POSTOPEN_CLAIM_DOCUMENT, hidden_manuscript
    )
    _write_bytes(
        hidden_stage,
        authorization_relative,
        (relocated / authorization_relative).read_bytes(),
    )
    hidden_marker = json.loads(
        (hidden_stage / verifier.PROFILE_MARKER).read_text(encoding="utf-8")
    )
    hidden_marker["profile"] = verifier.POSTOPEN_PROFILE
    hidden_marker["authorization"] = {"path": authorization_relative}
    hidden_evidence["profile"] = verifier.POSTOPEN_PROFILE
    hidden_evidence["compute_commit"] = compute_commit
    hidden_marker["git_history_evidence"] = hidden_evidence
    hidden_marker["authorized_worktree_dirt_policy"] = {
        "committed_document_diff": [{
            **_binding(
                verifier, hidden_stage, verifier.POSTOPEN_CLAIM_DOCUMENT
            ),
            "bytes": hidden_manuscript.stat().st_size,
        }],
    }
    with pytest.raises(ValueError, match="forbidden compute-to-manuscript"):
        verifier._verify_git_history_evidence(
            hidden_stage, hidden_marker, verifier.POSTOPEN_PROFILE
        )

    main_branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=source, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "checkout", "-q", "-b", "unrelated-fixture", compute_commit],
        cwd=source, check=True,
    )
    _write_bytes(source, "paper/unrelated.md", b"unrelated history\n")
    subprocess.run(["git", "add", "paper/unrelated.md"], cwd=source, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "unrelated sibling"],
        cwd=source, env=environment, check=True,
    )
    unrelated_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    subprocess.run(["git", "checkout", "-q", main_branch], cwd=source, check=True)
    unrelated_stage = tmp_path / "unrelated-stage"
    shutil.copytree(hidden_stage, unrelated_stage)
    unrelated_bundle = unrelated_stage / verifier.GIT_BUNDLE_PATH
    unrelated_bundle.unlink()
    subprocess.run(
        ["git", "bundle", "create", str(unrelated_bundle), "--all"],
        cwd=source, check=True, capture_output=True, text=True,
    )
    unrelated_marker = json.loads(json.dumps(hidden_marker))
    unrelated_marker["git_history_evidence"]["bundle"] = _binding(
        verifier, unrelated_stage, verifier.GIT_BUNDLE_PATH
    )
    unrelated_marker["git_history_evidence"]["compute_commit"] = unrelated_commit
    with pytest.raises(ValueError, match="compute-to-manuscript"):
        verifier._verify_git_history_evidence(
            unrelated_stage, unrelated_marker, verifier.POSTOPEN_PROFILE
        )

    (relocated / verifier.GIT_BUNDLE_PATH).unlink()
    with pytest.raises(ValueError, match="absent"):
        verifier._verify_git_history_evidence(
            relocated, marker, verifier.POSTOPEN_PROFILE
        )


def test_postopen_git_bundle_replays_real_prelabel_chronology_and_rejects_tamper(
    tmp_path,
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_real_chronology_git_evidence_test"
    )
    # This test isolates immutable Git chronology.  The full development-input
    # closure is exercised by dedicated fixtures; retain one deterministic
    # synthetic closure here so the chronology graph stays tractable.
    verifier._git_development_input_closure = lambda _bare, _commit: (
        FIXTURE_DEVELOPMENT_INPUT_CLOSURE_SHA256,
        FIXTURE_DEVELOPMENT_INPUT_CLOSURE_FILE_COUNT,
        tuple(),
    )
    # This synthetic chronology uses synthetic protocol-seal bytes and therefore
    # cannot also satisfy the production matrix seal's four fixed historical
    # governance digests.  Exact production matrix Git lineage and coordinated
    # document+seal resealing are exercised against a real clone in
    # test_real_git_model_matrix_rejects_coordinated_document_and_seal_rehash.
    verifier._verify_model_matrix_amendment_history_from_bundle = (
        lambda **_arguments: None
    )
    source = tmp_path / "chronology-source"
    source.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    environment = os.environ.copy()
    environment.update({
        "GIT_AUTHOR_NAME": "Chronology Fixture",
        "GIT_AUTHOR_EMAIL": "chronology@example.invalid",
        "GIT_COMMITTER_NAME": "Chronology Fixture",
        "GIT_COMMITTER_EMAIL": "chronology@example.invalid",
    })

    def write_json(relative: str, value: object) -> Path:
        return _write_bytes(
            source,
            relative,
            json.dumps(value, sort_keys=True).encode("utf-8") + b"\n",
        )

    def git_text(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=source,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()

    def commit(message: str) -> str:
        subprocess.run(["git", "add", "-A"], cwd=source, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", message],
            cwd=source,
            env=environment,
            check=True,
        )
        return git_text("rev-parse", "HEAD")

    def git_blob_binding(commit_id: str, relative: str) -> dict[str, object]:
        payload = subprocess.run(
            ["git", "show", f"{commit_id}:{relative}"],
            cwd=source,
            capture_output=True,
            check=True,
        ).stdout
        oid = git_text("rev-parse", f"{commit_id}:{relative}")
        assert len(oid) == 40
        return {
            "path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_count": len(payload),
            "git_blob_oid": oid,
        }

    protocol_markdown_path = "protocols/route_a_confirmatory_protocol.md"
    protocol_json_path = "protocols/route_a_confirmatory_v1.json"
    original_markdown = _write_bytes(
        source, protocol_markdown_path, b"# Original preregistration\n"
    ).read_bytes()
    original_commit = commit("original preregistration")

    final_markdown = _write_bytes(
        source, protocol_markdown_path, b"# Final prelabel protocol\n"
    )
    final_protocol = write_json(
        protocol_json_path,
        {
            "schema_version": 1,
            "status": "PLANNED_NOT_ACQUIRED",
            "protocol_id": "route-a-confirmatory-v1",
            "authoritative_protocol_commit": original_commit,
            "pre_label_amendments": [],
            "new_site_external_validation": {
                "status": "PLANNED_NOT_ACQUIRED",
                "planned_site_count": 1,
                "selection_seed": "route-a-confirmatory-v1-public-seed",
            },
            "metadata_candidate_contract": {"state_universe": ["CO"]},
            "time_holdout": {
                "start": "2021-01-01",
                "end": "2023-12-31",
            },
            "primary_inference_contract": {
                "confirmatory_family": _fixture_confirmatory_family(),
                "probabilistic_event_contract": json.loads(
                    (ROOT / "protocols/route_a_confirmatory_v1.json").read_text(
                        encoding="utf-8"
                    )
                )["primary_inference_contract"]["probabilistic_event_contract"],
            },
        },
    )
    final_commit = commit("final prelabel protocol")

    seal = {
        "format": verifier.PROTOCOL_SEAL_FORMAT,
        "status": "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED",
        "protocol_id": "route-a-confirmatory-v1",
        "original_preregistration": {
            "commit": original_commit,
            "markdown": {
                "path": protocol_markdown_path,
                "sha256": hashlib.sha256(original_markdown).hexdigest(),
            },
        },
        "final_prelabel_protocol": {
            "commit": final_commit,
            "json": {
                "path": protocol_json_path,
                "sha256": hashlib.sha256(final_protocol.read_bytes()).hexdigest(),
            },
            "markdown": {
                "path": protocol_markdown_path,
                "sha256": hashlib.sha256(final_markdown.read_bytes()).hexdigest(),
            },
        },
        "prelabel_attestation": {
            "external_timestamp_or_public_preregistration": False,
            "independent_custodian_or_worm_storage": False,
        },
    }
    write_json(verifier.PROTOCOL_SEAL_PATH, seal)
    _write_bytes(source, "pyproject.toml", b"[project]\nname='chronology-fixture'\n")
    _write_bytes(source, "requirements.txt", b"fixture>=1\n")
    _write_bytes(source, "requirements-lock.txt", b"fixture==1\n")
    _write_bytes(
        source,
        verifier.REPRODUCIBILITY_LOCK,
        b"fixture==1 --hash=sha256:" + b"0" * 64 + b"\n",
    )
    gate_paths = tuple(verifier.CHRONOLOGY_REQUIRED_GATE_PATHS)
    for relative in gate_paths:
        if relative in {
            verifier.PROBABILITY_METRIC_ERRATUM_PATH,
            verifier.PROBABILITY_METRIC_ERRATUM_SEAL_PATH,
            verifier.MODEL_MATRIX_AMENDMENT_PATH,
            verifier.MODEL_MATRIX_AMENDMENT_SEAL_PATH,
        }:
            continue
        _write_bytes(source, relative, f"# frozen gate: {relative}\n".encode())
    _write_bytes(
        source,
        verifier.DEVELOPMENT_REPLAY_ENTRYPOINT,
        b"#!/usr/bin/env python3\n# frozen replay fixture\n",
    )
    fixed_modules = {
        "thermoroute.opening": "src/thermoroute/opening.py",
        "thermoroute.chronology": "src/thermoroute/chronology.py",
        "thermoroute.model_suite": "src/thermoroute/model_suite.py",
        "thermoroute.frozen_inference": "src/thermoroute/frozen_inference.py",
        "thermoroute.datasets": "src/thermoroute/datasets.py",
        "thermoroute.provenance": "src/thermoroute/provenance.py",
        "thermoroute.usgs": "src/thermoroute/usgs.py",
        "thermoroute.inference_gate": "src/thermoroute/inference_gate.py",
        "thermoroute.outcome_qc": "src/thermoroute/outcome_qc.py",
        "thermoroute.probability_metric_erratum": (
            "src/thermoroute/probability_metric_erratum.py"
        ),
        "thermoroute.model_matrix_amendment": (
            "src/thermoroute/model_matrix_amendment.py"
        ),
        "thermoroute.quantiles": "src/thermoroute/quantiles.py",
        "thermoroute.coverage_audit": "src/thermoroute/coverage_audit.py",
        "thermoroute.coverage_bridge": "src/thermoroute/coverage_bridge.py",
        "thermoroute.repro": "src/thermoroute/repro.py",
    }
    fixed_files = {
        relative: relative
        for relative in (
            "src/thermoroute/opening_contract.py",
            "src/thermoroute/outcome_acquisition.py",
        )
    }
    fixed_entrypoints = {
        "orchestrator": "scripts/route_a_opening_orchestrator.py",
        "acquisition": "scripts/route_a_outcome_acquisition.py",
        "trusted_scorer": "scripts/route_a_trusted_scorer.py",
    }
    for relative in {
        *fixed_modules.values(), *fixed_files.values(), *fixed_entrypoints.values()
    }:
        _write_bytes(source, relative, f"# fixed code: {relative}\n".encode())
    model_suite_path = "data_usgs/confirmatory_model_suite_v1.json"
    development_replay_path = (
        "outputs/model_replay/route_a_development_replay_v1.json"
    )
    development_paths = {
        "frozen_panel_spec": "data_usgs/frozen_panel_v1.json",
        "panel": "data_usgs/panel_usgs_120v2.parquet",
        "registry": "data_usgs/station_registry_v1.csv",
    }
    for relative in development_paths.values():
        _write_bytes(source, relative, b"{}\n" if relative.endswith(".json") else b"dev\n")
    runtime_sha256 = "e" * 64
    model_entries, development_model_artifact_paths = (
        _write_development_model_fixtures(
            verifier, source, runtime_sha256=runtime_sha256
        )
    )
    inference_amendment_path = "protocols/route_a_inference_amendment_v2.json"
    inference_amendment_seal_path = (
        "protocols/route_a_inference_amendment_seal_v2.json"
    )
    inference_gate_path = "outputs/prelabel/route_a_inference_gate_v1.json"
    write_json(inference_amendment_path, {"fixture": "outcome-free amendment"})
    amendment_commit = commit("freeze outcome-free inference amendment")
    write_json(
        inference_amendment_seal_path,
        {
            "format": "thermoroute.route-a-inference-amendment-seal.v2",
            "status": "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED",
            "amendment": _binding(
                verifier, source, inference_amendment_path
            ),
            "final_prelabel_commit": amendment_commit,
        },
    )
    seal_commit = commit("seal outcome-free inference amendment")
    probability_erratum = json.loads(
        (ROOT / verifier.PROBABILITY_METRIC_ERRATUM_PATH).read_text(
            encoding="utf-8"
        )
    )
    probability_erratum["governance_inputs"] = {
        "base_protocol": _binding(verifier, source, protocol_json_path),
        "base_protocol_seal": _binding(
            verifier, source, verifier.PROTOCOL_SEAL_PATH
        ),
        "inference_amendment": _binding(
            verifier, source, inference_amendment_path
        ),
        "inference_amendment_seal": _binding(
            verifier, source, inference_amendment_seal_path
        ),
    }
    probability_erratum["scientific_scope"][
        "confirmatory_family_sha256"
    ] = verifier._sha256_json(_fixture_confirmatory_family())
    write_json(verifier.PROBABILITY_METRIC_ERRATUM_PATH, probability_erratum)
    erratum_document_commit = commit("freeze probability metric erratum")
    probability_erratum_seal = {
        "format": verifier.PROBABILITY_METRIC_ERRATUM_SEAL_FORMAT,
        "status": "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED",
        "erratum_id": verifier.PROBABILITY_METRIC_ERRATUM_ID,
        "erratum": _binding(
            verifier, source, verifier.PROBABILITY_METRIC_ERRATUM_PATH
        ),
        "governance_seals": {
            "base_protocol_seal": _binding(
                verifier, source, verifier.PROTOCOL_SEAL_PATH
            ),
            "inference_amendment_seal": _binding(
                verifier, source, inference_amendment_seal_path
            ),
        },
        "erratum_document_commit": erratum_document_commit,
        "history_contract": {
            "governance_seal_commits_must_be_ancestors": True,
            "erratum_blob_must_match_document_commit": True,
            "erratum_document_created_exactly_once": True,
            "document_commit_must_precede_seal_commit": True,
            "seal_created_exactly_once": True,
            "erratum_and_seal_immutable_to_release_tip": True,
        },
        "prelabel_attestation": {
            "post_2020_wtemp_requested_or_inspected": False,
            "confirmation_outcomes_requested_or_inspected": False,
            "outcome_endpoint_called": False,
            "outcome_independent": True,
            "network_used": False,
        },
    }
    write_json(
        verifier.PROBABILITY_METRIC_ERRATUM_SEAL_PATH,
        probability_erratum_seal,
    )
    erratum_seal_commit = commit("seal probability metric erratum")
    model_matrix_attestation = dict(verifier.MODEL_MATRIX_PRELABEL_ATTESTATION)
    model_matrix_amendment = json.loads(
        (ROOT / verifier.MODEL_MATRIX_AMENDMENT_PATH).read_text(
            encoding="utf-8"
        )
    )
    _write_bytes(
        source,
        verifier.MODEL_MATRIX_AMENDMENT_PATH,
        (ROOT / verifier.MODEL_MATRIX_AMENDMENT_PATH).read_bytes(),
    )
    assert (
        verifier.sha256_file(source / verifier.MODEL_MATRIX_AMENDMENT_PATH)
        == verifier.MODEL_MATRIX_AMENDMENT_SHA256
    )
    model_matrix_document_commit = commit(
        "freeze outcome-free model-matrix amendment"
    )
    model_matrix_amendment_seal = {
        "format": verifier.MODEL_MATRIX_AMENDMENT_SEAL_FORMAT,
        "status": verifier.MODEL_MATRIX_AMENDMENT_SEAL_STATUS,
        "amendment_id": verifier.MODEL_MATRIX_AMENDMENT_ID,
        "amendment": _binding(
            verifier, source, verifier.MODEL_MATRIX_AMENDMENT_PATH
        ),
        "amendment_document_commit": model_matrix_document_commit,
        "governance_seals": dict(verifier.MODEL_MATRIX_GOVERNANCE_SEALS),
        "history_contract": dict(verifier.MODEL_MATRIX_SEAL_HISTORY_CONTRACT),
        "prelabel_attestation": model_matrix_attestation,
    }
    write_json(
        verifier.MODEL_MATRIX_AMENDMENT_SEAL_PATH,
        model_matrix_amendment_seal,
    )
    model_matrix_seal_commit = commit("seal outcome-free model-matrix amendment")
    write_json(inference_gate_path, {"fixture": "fail-closed inference gate"})
    frozen_source_inventory = {
        relative: verifier.sha256_file(source / relative)
        for relative in sorted(verifier._working_model_control_paths(source))
        if verifier._matches_source_inventory(relative)
    }
    frozen_source_sha = verifier._sha256_json(frozen_source_inventory)
    bridge_path = "data_usgs/development_predictor_bridge_v1.json"
    bridge_normalized = {
        "frozen": (
            "data_usgs/development_predictor_bridge_v1/"
            "frozen_panel_predictors_2018_2020.parquet"
        ),
        "refreshed": (
            "data_usgs/development_predictor_bridge_v1/"
            "refreshed_predictors_2018_2020.parquet"
        ),
    }
    bridge_report = "data_usgs/development_predictor_bridge_v1/bridge_report_v1.json"
    bridge_request_map = (
        "data_usgs/development_predictor_bridge_v1/source_request_map_v1.json"
    )
    for relative in (*bridge_normalized.values(), bridge_report, bridge_request_map):
        _write_bytes(source, relative, b"{}\n" if relative.endswith(".json") else b"bridge\n")
    bridge_indexes: dict[str, str] = {}
    bridge_raw_paths: set[str] = set()
    for name in ("daymet", "gridmet", "gridmet_schema"):
        index = (
            "data_usgs/raw_snapshots/development-predictor-bridge-v1/"
            f"{name}/snapshot_index_v2.json"
        )
        metadata = str(PurePosixPath(index).parent / "metadata.json")
        response = str(PurePosixPath(index).parent / "response.bin")
        metadata_path = write_json(metadata, {})
        _write_bytes(source, response, f"{name} raw\n".encode())
        _write_bytes(
            source,
            index,
            verifier._canonical_json_bytes({
                "schema_version": 2,
                "snapshot_count": 1,
                "records": [{
                    "provider": name,
                    "request_sha256": hashlib.sha256(
                        f"{name} request".encode()
                    ).hexdigest(),
                    "metadata_path": "metadata.json",
                    "response_path": "response.bin",
                    "response_sha256": verifier.sha256_file(source / response),
                    "metadata_sha256": verifier.sha256_file(metadata_path),
                    "metadata_byte_count": metadata_path.stat().st_size,
                    "retrieved_at_utc": "2026-07-22T00:00:00+00:00",
                    "byte_count": (source / response).stat().st_size,
                    "request": {"provider": name},
                }]
            }),
        )
        bridge_indexes[name] = index
        bridge_raw_paths.update({index, metadata, response})
    bridge = {
        "format": "thermoroute.development-predictor-bridge.v1",
        "status": "PASS_EXACT_PRODUCT_BRIDGE",
        "outcome_values_requested_or_read": False,
        "source_tree_sha256": frozen_source_sha,
        "panel": _binding(verifier, source, development_paths["panel"]),
        "registry": _binding(verifier, source, development_paths["registry"]),
        "normalized": {
            name: _binding(verifier, source, relative)
            for name, relative in bridge_normalized.items()
        },
        "report": _binding(verifier, source, bridge_report),
        "request_map": _binding(verifier, source, bridge_request_map),
        "raw_snapshot_indexes": {
            name: _binding(verifier, source, relative)
            for name, relative in bridge_indexes.items()
        },
    }
    write_json(bridge_path, bridge)

    stage09_config = {
        "stage": "09_usgs_experiment",
        "input_closure_sha256": FIXTURE_DEVELOPMENT_INPUT_CLOSURE_SHA256,
        "input_closure_file_count": (
            FIXTURE_DEVELOPMENT_INPUT_CLOSURE_FILE_COUNT
        ),
    }
    stage09_identity = _fixture_run_identity(
        verifier,
        stage09_config,
        panel_sha256=verifier.sha256_file(
            source / development_paths["panel"]
        ),
        registry_sha256=verifier.sha256_file(
            source / development_paths["registry"]
        ),
        source_sha256=frozen_source_sha,
        runtime_sha256=runtime_sha256,
        input_closure_sha256=FIXTURE_DEVELOPMENT_INPUT_CLOSURE_SHA256,
    )
    stage09_run_id = stage09_identity["run_id"]
    stage09_artifacts = {
        "run_manifest": (
            f"outputs/runs/09_usgs_experiment/{stage09_run_id}/run.json"
        ),
        "predictions": "outputs/predictions/usgs_predictions_stage9_v2.parquet",
        "prediction_sidecar": (
            "outputs/predictions/usgs_predictions_stage9_v2.parquet.meta.json"
        ),
        "scores": "outputs/tables/usgs_scores.csv",
        "report": "outputs/reports/usgs_experiment.md",
        "lightgbm_selection": (
            "outputs/tables/lightgbm_joint_validation_selection.csv"
        ),
        "thermoroute_pointer": "outputs/models/thermoroute_usgs_bundle.json",
        "lightgbm_pointer": "outputs/models/lightgbm_usgs_bundle.json",
        "components_pointer": "outputs/models/route_a_stage9_components.json",
    }
    for label, relative in stage09_artifacts.items():
        if label == "run_manifest":
            continue
        _write_bytes(source, relative, f"stage09 {label}\n".encode())
    write_json(
        stage09_artifacts["run_manifest"],
        {
            "schema_version": "thermoroute.run.v2",
            "identity": stage09_identity,
            "resolved_config": stage09_config,
            "created_utc": "2026-07-22T00:00:00+00:00",
            "environment": {},
            "git": {},
            "provenance": {},
        },
    )
    stage09_receipt_path = "outputs/models/route_a_stage09_completion.json"
    stage09_receipt = {
        "format": "thermoroute.stage09-completion-receipt.v1",
        "status": "PASS_FORMAL_STAGE09_COMPLETE",
        "stage": "09_usgs_experiment",
        "run_id": stage09_run_id,
        "run_identity": stage09_identity,
        "formal_configuration": stage09_config,
        "confirmation_outcomes_requested_or_read": False,
        "artifacts": {
            label: _binding(verifier, source, relative)
            for label, relative in stage09_artifacts.items()
        },
    }
    stage09_receipt["receipt_self_sha256"] = verifier._sha256_json(
        stage09_receipt
    )
    write_json(stage09_receipt_path, stage09_receipt)

    stage09b_config = _stage09b_fixture_config(
        _binding(verifier, source, bridge_path),
    )
    stage09b_identity = _fixture_run_identity(
        verifier,
        stage09b_config,
        panel_sha256=verifier.sha256_file(
            source / development_paths["panel"]
        ),
        registry_sha256=verifier.sha256_file(
            source / development_paths["registry"]
        ),
        source_sha256=frozen_source_sha,
        runtime_sha256=runtime_sha256,
        input_closure_sha256=FIXTURE_DEVELOPMENT_INPUT_CLOSURE_SHA256,
    )
    stage09b_run_id = stage09b_identity["run_id"]
    stage09b_run_dir = (
        f"outputs/runs/09b_development_controls/{stage09b_run_id}"
    )
    stage09b_artifacts = {
        "run_manifest": f"{stage09b_run_dir}/run.json",
        "frozen_panel_spec": development_paths["frozen_panel_spec"],
        "panel": development_paths["panel"],
        "registry": development_paths["registry"],
        "predictor_bridge": bridge_path,
        "predictions": f"{stage09b_run_dir}/development_controls_predictions.parquet",
        "prediction_sidecar": (
            f"{stage09b_run_dir}/development_controls_predictions.parquet.meta.json"
        ),
        "architecture_budget": (
            f"{stage09b_run_dir}/development_controls_architecture_budget.csv"
        ),
        "architecture_budget_sidecar": (
            f"{stage09b_run_dir}/development_controls_architecture_budget.csv.meta.json"
        ),
        "metric_summary": f"{stage09b_run_dir}/development_controls_metric_summary.csv",
        "metric_summary_sidecar": (
            f"{stage09b_run_dir}/development_controls_metric_summary.csv.meta.json"
        ),
        "report": f"{stage09b_run_dir}/development_controls_report.md",
        "report_sidecar": f"{stage09b_run_dir}/development_controls_report.md.meta.json",
        "semantic_audit": f"{stage09b_run_dir}/development_controls_semantic_audit.json",
        "semantic_audit_sidecar": (
            f"{stage09b_run_dir}/development_controls_semantic_audit.json.meta.json"
        ),
    }
    for label, relative in stage09b_artifacts.items():
        if label in {
            "frozen_panel_spec", "panel", "registry", "predictor_bridge",
            "run_manifest", "semantic_audit", "semantic_audit_sidecar",
        }:
            continue
        _write_bytes(source, relative, f"stage09b {label}\n".encode())
    write_json(
        stage09b_artifacts["run_manifest"],
        {
            "schema_version": "thermoroute.run.v2",
            "identity": stage09b_identity,
            "resolved_config": stage09b_config,
            "created_utc": "2026-07-22T00:00:00+00:00",
            "environment": {},
            "git": {},
            "provenance": {},
        },
    )
    stage09b_members = []
    semantic_members = []
    stage09b_member_paths: set[str] = set()
    for arm_id, seed in verifier._stage09b_release_members():
        member_path = (
            f"{stage09b_run_dir}/arm_predictions/{arm_id}/seed{seed}.parquet"
        )
        member_sidecar = f"{member_path}.meta.json"
        checkpoint_path = (
            f"{stage09b_run_dir}/checkpoints/{arm_id}/seed{seed}.pt"
        )
        checkpoint_sidecar = f"{checkpoint_path}.meta.json"
        _write_bytes(source, member_path, f"{arm_id}/seed{seed}\n".encode())
        _write_bytes(source, member_sidecar, b"{}\n")
        _write_bytes(
            source,
            checkpoint_path,
            f"checkpoint:{arm_id}:seed{seed}\n".encode(),
        )
        _write_bytes(source, checkpoint_sidecar, b"{}\n")
        stage09b_member_paths.update({
            member_path,
            member_sidecar,
            checkpoint_path,
            checkpoint_sidecar,
        })
        stage09b_members.append({
            "arm_id": arm_id,
            "seed": seed,
            "prediction": _binding(verifier, source, member_path),
            "prediction_sidecar": _binding(verifier, source, member_sidecar),
            "checkpoint": _binding(verifier, source, checkpoint_path),
            "checkpoint_sidecar": _binding(
                verifier, source, checkpoint_sidecar
            ),
        })
        semantic_members.append({
            "arm_id": arm_id,
            "seed": seed,
            "prediction": {
                "sha256": verifier.sha256_file(source / member_path),
                "bytes": (source / member_path).stat().st_size,
            },
            "prediction_sidecar": {
                "sha256": verifier.sha256_file(source / member_sidecar),
                "bytes": (source / member_sidecar).stat().st_size,
            },
            "checkpoint": {
                "sha256": verifier.sha256_file(source / checkpoint_path),
                "bytes": (source / checkpoint_path).stat().st_size,
            },
            "checkpoint_sidecar": {
                "sha256": verifier.sha256_file(source / checkpoint_sidecar),
                "bytes": (source / checkpoint_sidecar).stat().st_size,
            },
            "normalised_prediction_sha256": hashlib.sha256(
                f"normalised:{arm_id}:{seed}".encode()
            ).hexdigest(),
            "best_model_state_prediction_replay_verified": True,
        })

    stage09b_matrix = {
        "expected_members": len(stage09b_members),
        "prediction_rows": len(stage09b_members) * 3,
        "common_forecast_keys": 3,
        "splits": ["calib", "test", "val"],
        "reference_member": "PlainMLP-7var/seed0",
    }

    def stage09b_descriptor(relative: str) -> dict[str, object]:
        return {
            "sha256": verifier.sha256_file(source / relative),
            "bytes": (source / relative).stat().st_size,
        }

    stage09b_paired_records = [
        {
            "comparison_family": comparison["comparison_family"],
            "comparison_id": comparison["comparison_id"],
            "candidate_arm_id": comparison["candidate_arm_id"],
            "reference_arm_id": comparison["reference_arm_id"],
            "seed": seed,
            "split": split,
            "horizon": horizon,
            "common_forecast_keys": 1,
            "stations": 1,
            "median_paired_station_rmse_difference_c": 0.0,
        }
        for comparison in verifier._stage09b_paired_comparison_registry()
        for seed in comparison["seeds"]
        for split in ("calib", "test", "val")
        for horizon in (1, 3, 7)
    ]
    stage09b_scientific = verifier._stage09b_scientific_summary(
        pd.DataFrame.from_records(stage09b_paired_records)
    )
    semantic_audit = {
        "format": "thermoroute.development-controls-semantic-audit.v3",
        "status": "PASS_BEST_MODEL_STATE_PREDICTION_REPLAY",
        "run_id": stage09b_run_id,
        "evidence_scope": "best_model_state_prediction_replay",
        "training_replay_verified": False,
        "best_model_state_prediction_replay_verified": True,
        "post_2020_outcomes_requested_or_read": False,
        "matrix_audit": stage09b_matrix,
        "canonical_window_registry": {
            "sha256": "c" * 64,
            "common_forecast_keys": 3,
            "train_examples_per_epoch": 3,
            "train_registry_sha256": "d" * 64,
        },
        "scientific_summary": stage09b_scientific,
        "members": semantic_members,
        "derived_artifacts": {
            "architecture_budget": {
                "artifact": stage09b_descriptor(
                    stage09b_artifacts["architecture_budget"]
                ),
                "sidecar": stage09b_descriptor(
                    stage09b_artifacts["architecture_budget_sidecar"]
                ),
            },
            "combined_predictions": {
                "artifact": stage09b_descriptor(stage09b_artifacts["predictions"]),
                "sidecar": stage09b_descriptor(
                    stage09b_artifacts["prediction_sidecar"]
                ),
            },
            "metric_summary": {
                "artifact": stage09b_descriptor(
                    stage09b_artifacts["metric_summary"]
                ),
                "sidecar": stage09b_descriptor(
                    stage09b_artifacts["metric_summary_sidecar"]
                ),
            },
            "report": {
                "artifact": stage09b_descriptor(stage09b_artifacts["report"]),
                "sidecar": stage09b_descriptor(
                    stage09b_artifacts["report_sidecar"]
                ),
            },
        },
    }
    semantic_audit["semantic_audit_self_sha256"] = verifier._sha256_json(
        semantic_audit
    )
    write_json(stage09b_artifacts["semantic_audit"], semantic_audit)
    _write_bytes(source, stage09b_artifacts["semantic_audit_sidecar"], b"{}\n")
    stage09b_receipt_path = "outputs/models/route_a_stage09b_completion.json"
    stage09b_receipt = {
        "format": "thermoroute.stage09b-completion-receipt.v3",
        "status": "PASS_STAGE09B_BEST_MODEL_STATE_PREDICTION_REPLAY",
        "stage": "09b_development_controls",
        "run_id": stage09b_run_id,
        "run_identity": stage09b_identity,
        "formal_configuration": stage09b_config,
        "evidence_scope": "best_model_state_prediction_replay",
        "training_replay_verified": False,
        "best_model_state_prediction_replay_verified": True,
        "matrix_audit": stage09b_matrix,
        "member_registry": stage09b_members,
        "artifacts": {
            label: _binding(verifier, source, relative)
            for label, relative in stage09b_artifacts.items()
        },
        "post_2020_outcomes_requested_or_read": False,
    }
    stage09b_receipt["receipt_self_sha256"] = verifier._sha256_json(
        stage09b_receipt
    )
    write_json(stage09b_receipt_path, stage09b_receipt)

    development_contract = {
        **{
            name: _binding(verifier, source, relative)
            for name, relative in development_paths.items()
        },
        "predictor_bridge": _binding(verifier, source, bridge_path),
        "source_sha256": frozen_source_sha,
    }
    stage16_gate, stage16_paths = _write_stage16_gate_fixture(
        verifier,
        source,
        model_entries=model_entries,
        development_contract=development_contract,
        stage9_receipt=stage09_receipt,
        stage9_path=stage09_receipt_path,
        source_sha256=frozen_source_sha,
        runtime_sha256=runtime_sha256,
    )
    stage25_gate, stage25_paths = _write_stage25_gate_fixture(
        verifier,
        source,
        model_entries=model_entries,
        development_contract=development_contract,
        source_sha256=frozen_source_sha,
        runtime_sha256=runtime_sha256,
    )
    suite = {
        "format": "thermoroute.route-a-model-suite.v1",
        "status": "FROZEN_BEFORE_LABEL_OPENING",
        "training_device": "cpu",
        "numerical_runtime_sha256": runtime_sha256,
        "protocol_sha256": verifier.sha256_file(source / protocol_json_path),
        "actual_feature_order": [
            "WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP"
        ],
        "model_matrix_amendment": {
            "format": verifier.MODEL_MATRIX_SUITE_BINDING_FORMAT,
            "document": {
                "path": verifier.MODEL_MATRIX_AMENDMENT_PATH,
                "sha256": verifier.sha256_file(
                    source / verifier.MODEL_MATRIX_AMENDMENT_PATH
                ),
                "format": verifier.MODEL_MATRIX_AMENDMENT_FORMAT,
                "status": verifier.MODEL_MATRIX_AMENDMENT_STATUS,
                "amendment_id": verifier.MODEL_MATRIX_AMENDMENT_ID,
                "amendment_document_commit": model_matrix_document_commit,
            },
            "seal": {
                "path": verifier.MODEL_MATRIX_AMENDMENT_SEAL_PATH,
                "sha256": verifier.sha256_file(
                    source / verifier.MODEL_MATRIX_AMENDMENT_SEAL_PATH
                ),
                "format": verifier.MODEL_MATRIX_AMENDMENT_SEAL_FORMAT,
                "status": verifier.MODEL_MATRIX_AMENDMENT_SEAL_STATUS,
            },
            "contract_id": verifier._model_matrix_contract_id(
                model_matrix_amendment
            ),
        },
        "development_contract": development_contract,
        "preopening_gates": {
            "stage09_completion": _binding(
                verifier, source, stage09_receipt_path
            ),
            "stage09b_development_controls": _binding(
                verifier, source, stage09b_receipt_path
            ),
            "stage16_lstm_completion": stage16_gate,
            "stage25_external_completion": stage25_gate,
        },
        "cohorts": {
            cohort: {"models": entries}
            for cohort, entries in model_entries.items()
        },
    }
    write_json(model_suite_path, suite)
    _write_bytes(
        source,
        development_replay_path,
        verifier._lineage_canonical_json_bytes(
            _development_replay_fixture(
                verifier,
                source,
                suite=suite,
                source_sha256=frozen_source_sha,
                runtime_sha256=runtime_sha256,
            )
        ),
    )
    retained_development_model_artifact_paths = {
        relative
        for relative in development_model_artifact_paths
        if not relative.startswith("outputs/models/external/")
        and not relative.startswith("outputs/models/temporal/lstm/")
        and not relative.startswith("outputs/development/temporal_lstm.")
    }
    model_artifact_paths = {
        model_suite_path,
        development_replay_path,
        *development_paths.values(),
        *retained_development_model_artifact_paths,
        bridge_path,
        *bridge_normalized.values(),
        bridge_report,
        bridge_request_map,
        *bridge_raw_paths,
        stage09_receipt_path,
        *stage09_artifacts.values(),
        stage09b_receipt_path,
        *stage09b_artifacts.values(),
        *stage09b_member_paths,
        *stage16_paths,
        *stage25_paths,
    }
    model_commit = commit("freeze executable models and chronology gate")
    model_tree_paths = {
        value
        for value in git_text("ls-tree", "-r", "--name-only", model_commit).splitlines()
        if verifier._is_model_control_path(value)
    }
    model_source_inventory = {
        relative: str(git_blob_binding(model_commit, relative)["sha256"])
        for relative in sorted(model_tree_paths)
        if verifier._matches_source_inventory(relative)
    }
    source_tree_sha256 = verifier._sha256_json(model_source_inventory)
    assert source_tree_sha256 == frozen_source_sha

    input_paths = {
        "candidate_table": "data_usgs/confirmatory_candidate_sites_v1.csv",
        "candidate_provenance": (
            "data_usgs/confirmatory_candidate_sites_v1.provenance.json"
        ),
        "candidate_snapshot_index": (
            "data_usgs/raw_snapshots/confirmatory-candidates-v1/"
            "snapshot_index.json"
        ),
        "external_registry": "data_usgs/confirmatory_site_registry_v1.csv",
        "external_lock": "data_usgs/confirmatory_site_registry_v1.lock.json",
        "input_manifest": "data_usgs/confirmatory_actual_inputs_v1.json",
    }
    candidate_site_no = "99999999"
    candidate_table_payload = (
        "site_no,station_nm,lat,lon,state,site_type,huc_cd,drain_area_va\n"
        f"{candidate_site_no},Fixture River,40.125,-105.25,CO,ST,"
        "10190005,42.5\n"
    ).encode("utf-8")
    _write_bytes(
        source, input_paths["candidate_table"], candidate_table_payload
    )
    candidate_index = input_paths["candidate_snapshot_index"]
    candidate_url = (
        "https://waterservices.usgs.gov/nwis/site/?agencyCd=USGS&format=rdb&"
        "hasDataTypeCd=dv&parameterCd=00010&siteOutput=expanded&siteStatus=all&"
        "siteType=ST&stateCd=CO"
    )
    candidate_request = {
        "schema_version": 1,
        "provider": "usgs-nwis-confirmatory-site-metadata",
        "method": "GET",
        "url": candidate_url,
        "headers": {
            "User-Agent": "ThermoRoute/1.0 Route-A metadata-only discovery"
        },
    }
    candidate_request_sha = hashlib.sha256(
        verifier._canonical_json_bytes(candidate_request)
    ).hexdigest()
    candidate_base = (
        PurePosixPath(candidate_index).parent
        / str(candidate_request["provider"])
        / candidate_request_sha
    )
    candidate_metadata = str(candidate_base / "metadata.json")
    candidate_response = str(candidate_base / "response.bin")
    candidate_response_payload = (
        "agency_cd\tsite_no\tstation_nm\tsite_tp_cd\tdec_lat_va\t"
        "dec_long_va\thuc_cd\tdrain_area_va\n"
        "5s\t15s\t50s\t7s\t16s\t16s\t16s\t14n\n"
        f"USGS\t{candidate_site_no}\tFixture River\tST\t40.125\t-105.25\t"
        "10190005\t42.5\n"
    ).encode("utf-8")
    _write_bytes(source, candidate_response, candidate_response_payload)
    _write_bytes(
        source,
        candidate_metadata,
        verifier._canonical_json_bytes({
            "schema_version": 2,
            "request": candidate_request,
            "request_sha256": candidate_request_sha,
            "retrieved_at_utc": "2026-07-26T00:00:00+00:00",
            "http_status": 200,
            "response_headers": {},
            "byte_count": len(candidate_response_payload),
            "response_sha256": verifier.sha256_file(source / candidate_response),
            "response_file": "response.bin",
            "final_url": candidate_url,
            "retrieval_semantics": "DIRECT_HTTP_RESPONSE",
        }),
    )
    _write_bytes(
        source,
        candidate_index,
        verifier._canonical_json_bytes({
            "schema_version": 2,
            "snapshot_count": 1,
            "records": [{
                "provider": "usgs-nwis-confirmatory-site-metadata",
                "request_sha256": candidate_request_sha,
                "metadata_path": candidate_metadata.removeprefix(
                    str(PurePosixPath(candidate_index).parent) + "/"
                ),
                "metadata_sha256": verifier.sha256_file(source / candidate_metadata),
                "metadata_byte_count": (source / candidate_metadata).stat().st_size,
                "response_path": candidate_response.removeprefix(
                    str(PurePosixPath(candidate_index).parent) + "/"
                ),
                "response_sha256": verifier.sha256_file(source / candidate_response),
                "retrieved_at_utc": "2026-07-26T00:00:00+00:00",
                "byte_count": (source / candidate_response).stat().st_size,
                "request": candidate_request,
            }]
        }),
    )
    candidate_index_sha256 = verifier.sha256_file(source / candidate_index)
    candidate_provenance = {
        "schema_version": 1,
        "artifact_role": "PRE_LABEL_METADATA_ONLY_CANDIDATE_UNIVERSE",
        "protocol_sha256": verifier.sha256_file(source / protocol_json_path),
        "state_universe": ["CO"],
        "state_universe_rule": (
            "states represented in the frozen 120-site development registry; "
            "no post-2020 outcome or coverage information"
        ),
        "candidate_rule": (
            "USGS stream sites whose site metadata advertises daily-value "
            "parameter 00010 capability; siteStatus=all"
        ),
        "candidate_count": 1,
        "site_primary_key": "site_no",
        "sort_order": ["site_no", "state"],
        "columns": [
            "site_no", "station_nm", "lat", "lon", "state", "site_type",
            "huc_cd", "drain_area_va",
        ],
        "outcome_endpoint_requested": False,
        "outcome_values_requested": False,
        "holdout_coverage_requested_or_computed": False,
        "raw_snapshot_index": candidate_index,
        "raw_snapshot_index_sha256": candidate_index_sha256,
        "candidate_table_sha256": verifier.sha256_file(
            source / input_paths["candidate_table"]
        ),
        "requests": [{
            "state": "CO",
            "candidate_count": 1,
            "request_sha256": candidate_request_sha,
            "response_sha256": verifier.sha256_file(source / candidate_response),
            "retrieved_at_utc": "2026-07-26T00:00:00+00:00",
            "byte_count": len(candidate_response_payload),
        }],
    }
    _write_bytes(
        source,
        input_paths["candidate_provenance"],
        verifier._canonical_json_bytes(candidate_provenance),
    )
    selection_seed = "route-a-confirmatory-v1-public-seed"
    selection_rank = hashlib.sha256(
        f"{selection_seed}:{candidate_site_no}".encode("utf-8")
    ).hexdigest()
    external_registry_payload = (
        "site_no,station_nm,lat,lon,state,site_type,huc_cd,drain_area_va,"
        "selection_rank_sha256\n"
        f"{candidate_site_no},Fixture River,40.125,-105.25,CO,ST,10190005,"
        f"42.5,{selection_rank}\n"
    ).encode("utf-8")
    _write_bytes(
        source, input_paths["external_registry"], external_registry_payload
    )
    external_lock = {
        "schema_version": 1,
        "protocol_id": final_protocol.name.removesuffix(".json"),
        "protocol_sha256": verifier.sha256_file(source / protocol_json_path),
        "authoritative_protocol_commit": original_commit,
        "pre_label_amendments_sha256": verifier._sha256_json([]),
        "status": "REGISTRY_FROZEN_LABELS_SEALED",
        "site_count": 1,
        "site_primary_key": "site_no",
        "selection_seed": selection_seed,
        "holdout_start": "2021-01-01",
        "holdout_end": "2023-12-31",
        "development_panel_spec_sha256": verifier.sha256_file(
            source / development_paths["frozen_panel_spec"]
        ),
        "candidate_table_sha256": verifier.sha256_file(
            source / input_paths["candidate_table"]
        ),
        "candidate_provenance_sha256": verifier.sha256_file(
            source / input_paths["candidate_provenance"]
        ),
        "candidate_snapshot_index_sha256": candidate_index_sha256,
        "candidate_acquisition_session": {
            "maximum_duration_seconds": 86400,
            "retrieved_at_min_utc": "2026-07-26T00:00:00+00:00",
            "retrieved_at_max_utc": "2026-07-26T00:00:00+00:00",
            "clock_source": "LOCAL_SYSTEM_CLOCK_NOT_EXTERNALLY_ATTESTED",
        },
        "chronology_trust_boundary": (
            "LOCAL_HONEST_OWNER_ONLY_NO_EXTERNAL_TIMESTAMP_OR_CUSTODIAN"
        ),
        "confirmatory_registry_sha256": verifier.sha256_file(
            source / input_paths["external_registry"]
        ),
        "frozen_artifacts": {
            "development_panel_spec": _binding(
                verifier, source, development_paths["frozen_panel_spec"]
            ),
            "candidate_table": _binding(
                verifier, source, input_paths["candidate_table"]
            ),
            "candidate_provenance": _binding(
                verifier, source, input_paths["candidate_provenance"]
            ),
            "candidate_snapshot_index": _binding(
                verifier, source, candidate_index
            ),
        },
        "labels_state": "SEALED_NOT_ACQUIRED",
        "opening_count": 0,
        "registry_frozen_at_utc": "2026-07-27T00:00:00+00:00",
        "created_at_utc": "2026-07-27T00:00:00+00:00",
    }
    external_lock["protocol_id"] = json.loads(
        final_protocol.read_bytes()
    )["protocol_id"]
    _write_bytes(
        source,
        input_paths["external_lock"],
        verifier._canonical_json_bytes(external_lock),
    )
    temporal_table = "data_usgs/confirmatory_predictors/temporal.parquet"
    external_table = "data_usgs/confirmatory_predictors/external.parquet"
    request_map = "data_usgs/confirmatory_predictors/source_request_map_v1.json"
    for relative, payload in (
        (temporal_table, b"temporal\n"),
        (external_table, b"external\n"),
    ):
        _write_bytes(source, relative, payload)
    write_json(request_map, {})
    meteorology_contracts = {
        "ORNL Daymet single-pixel daily data": (
            "data_usgs/raw_snapshots/confirmatory-historical-inputs-v1/"
            "daymet-v1/snapshot_index.json",
            "ornl-daymet-single-pixel-route-a",
            "https://daymet.ornl.gov/single-pixel/api/data?lat=40.00000000",
            ["TEMP", "PRCP", "RHMEAN", "DH"],
        ),
        "gridMET daily mean wind via NWK NCSS": (
            "data_usgs/raw_snapshots/confirmatory-historical-inputs-v1/"
            "gridmet-v1/snapshot_index.json",
            "gridmet-ncss-route-a",
            "https://thredds.northwestknowledge.net/thredds/ncss/MET/wind.nc"
            "?var=daily_mean_wind_speed",
            ["WDSP"],
        ),
        "gridMET OPeNDAP dataset attributes": (
            "data_usgs/raw_snapshots/confirmatory-historical-inputs-v1/"
            "gridmet-schema-v1/snapshot_index.json",
            "gridmet-opendap-schema-route-a",
            "https://thredds.northwestknowledge.net/thredds/dodsC/"
            "MET/wind.nc.das",
            ["WDSP"],
        ),
    }
    meteorology_paths: set[str] = set()
    for source_name, (index_path, provider, url, _fields) in (
        meteorology_contracts.items()
    ):
        meteorology_paths |= _write_formal_meteorology_snapshot(
            verifier,
            source,
            index_path,
            provider=provider,
            url=url,
            payload=f"{source_name} fixture\n".encode("utf-8"),
        )
    write_json(
        input_paths["input_manifest"],
        {
            "format": "thermoroute.route-a-prelabel-inputs.v1",
            "status": "FROZEN_PRELABEL_NO_OUTCOMES",
            "contains_outcome": False,
            "contains_outcome_labels": False,
            "post_2020_wtemp_requested_or_inspected": False,
            "cohort_tables": {
                "temporal": _binding(verifier, source, temporal_table),
                "external": _binding(verifier, source, external_table),
            },
            "registry_inputs": {
                "temporal": _binding(
                    verifier, source, development_paths["registry"]
                ),
                "external": _binding(
                    verifier, source, input_paths["external_registry"]
                ),
            },
            "source_evidence": [
                *[
                    {
                        "source": source_name,
                        "evidence_type": "snapshot_index",
                        "contains_outcome": False,
                        "contains_outcome_labels": False,
                        "fields": fields,
                        "artifact": _binding(verifier, source, index_path),
                    }
                    for source_name, (index_path, _provider, _url, fields)
                    in meteorology_contracts.items()
                ],
                {
                    "source": "site-to-request normalization map",
                    "evidence_type": "normalized_immutable_snapshot",
                    "contains_outcome": False,
                    "contains_outcome_labels": False,
                    "fields": ["TEMP", "PRCP", "RHMEAN", "DH", "WDSP"],
                    "artifact": _binding(verifier, source, request_map),
                },
            ],
        },
    )
    input_artifact_paths = {
        *input_paths.values(),
        development_paths["frozen_panel_spec"],
        development_paths["registry"],
        candidate_metadata,
        candidate_response,
        temporal_table,
        external_table,
        request_map,
        *meteorology_paths,
    }
    input_commit = commit("freeze outcome-free input evidence")

    _write_bytes(
        source,
        "outputs/prelabel/chronology_receipt_base.marker",
        b"receipt creation base\n",
    )
    receipt_base_commit = commit("establish chronology receipt base")

    authorization_path = "data_usgs/confirmatory_opening_authorization_v1.json"
    chronology_order = {
        "model_freeze_commit": model_commit,
        "input_evidence_commit": input_commit,
        "receipt_creation_base_commit": receipt_base_commit,
        "strict_order_verified": True,
    }
    chronology_paths = {
        "protocol_seal": verifier.PROTOCOL_SEAL_PATH,
        "model_matrix_amendment": verifier.MODEL_MATRIX_AMENDMENT_PATH,
        "model_matrix_amendment_seal": (
            verifier.MODEL_MATRIX_AMENDMENT_SEAL_PATH
        ),
        "model_suite": model_suite_path,
        "development_replay": development_replay_path,
        **input_paths,
    }
    chronology = {
        "format": verifier.CHRONOLOGY_FORMAT,
        "status": verifier.CHRONOLOGY_STATUS,
        "order": chronology_order,
        "model_matrix_history": {
            "format": verifier.MODEL_MATRIX_HISTORY_FORMAT,
            "amendment": git_blob_binding(
                model_matrix_document_commit,
                verifier.MODEL_MATRIX_AMENDMENT_PATH,
            ),
            "seal": git_blob_binding(
                model_matrix_seal_commit,
                verifier.MODEL_MATRIX_AMENDMENT_SEAL_PATH,
            ),
            "amendment_id": verifier.MODEL_MATRIX_AMENDMENT_ID,
            "amendment_document_commit": model_matrix_document_commit,
            "seal_commit": model_matrix_seal_commit,
            "model_freeze_commit": model_commit,
            "contract_id": verifier._model_matrix_contract_id(
                model_matrix_amendment
            ),
            "strict_order_verified": True,
            "immutable_to_release_tip": True,
            "evidence_scope": verifier.CHRONOLOGY_EVIDENCE_SCOPE,
        },
        "protocol_history": {
            "seal": git_blob_binding(model_commit, verifier.PROTOCOL_SEAL_PATH),
            "original_commit": original_commit,
            "final_prelabel_commit": final_commit,
            "declared_git_show_bindings": [
                {
                    "role": "original_markdown",
                    "commit": original_commit,
                    "path": protocol_markdown_path,
                    "sha256": hashlib.sha256(original_markdown).hexdigest(),
                },
                {
                    "role": "final_json",
                    "commit": final_commit,
                    "path": protocol_json_path,
                    "sha256": hashlib.sha256(final_protocol.read_bytes()).hexdigest(),
                },
                {
                    "role": "final_markdown",
                    "commit": final_commit,
                    "path": protocol_markdown_path,
                    "sha256": hashlib.sha256(final_markdown.read_bytes()).hexdigest(),
                },
            ],
        },
        "paths": chronology_paths,
        "required_gate_files_at_model_freeze": [
            git_blob_binding(model_commit, relative) for relative in gate_paths
        ],
        "model_source_control_artifacts": [
            git_blob_binding(model_commit, relative)
            for relative in sorted(model_tree_paths)
        ],
        "source_tree_sha256": source_tree_sha256,
        "model_freeze_artifacts": [
            git_blob_binding(model_commit, relative)
            for relative in sorted(model_artifact_paths)
        ],
        "input_evidence_artifacts": [
            git_blob_binding(input_commit, relative)
            for relative in sorted(input_artifact_paths)
        ],
        "absence_at_model_freeze": {
            "checked_paths": sorted(
                set(verifier.CHRONOLOGY_FIXED_PRELABEL_ABSENCE_PATHS)
                | (
                    input_artifact_paths
                    - {
                        development_paths["frozen_panel_spec"],
                        development_paths["registry"],
                    }
                )
            ),
            "present_paths": [],
        },
        "post_model_control_audit": {
            "protected_directories": list(verifier.PROTECTED_DIRECTORIES),
            "protected_exact_files": list(verifier.PROTECTED_EXACT_FILES),
            "protected_root_patterns": list(verifier.PROTECTED_ROOT_PATTERNS),
            "committed_touches": [],
            "worktree_changes": [],
        },
        "post_freeze_artifact_mutation_count": 0,
        "external_timestamp_or_public_preregistration": False,
        "independent_custodian_or_worm_storage": False,
        "evidence_scope": verifier.CHRONOLOGY_EVIDENCE_SCOPE,
        "fallback_if_validation_fails": (
            "TRANSDUCTIVE_RETROSPECTIVE_EXPLORATION_CONFIRMATION_CLAIMS_PROHIBITED"
        ),
    }
    chronology["receipt_self_sha256"] = verifier._chronology_self_sha256(
        chronology
    )
    _write_bytes(
        source,
        verifier.CHRONOLOGY_PATH,
        verifier._canonical_json_bytes(chronology),
    )
    compute_commit = commit("freeze chronology receipt")

    def fixed_binding(relative: str) -> dict[str, str]:
        return {
            "path": relative,
            "realpath": str((source / relative).resolve()),
            "sha256": verifier.sha256_file(source / relative),
        }

    fixed_stable = {
        "modules": {
            name: fixed_binding(relative)
            for name, relative in fixed_modules.items()
        },
        "files": {
            name: fixed_binding(relative)
            for name, relative in fixed_files.items()
        },
        "entrypoints": {
            name: fixed_binding(relative)
            for name, relative in fixed_entrypoints.items()
        },
    }

    # Production authorization is create-only Git dirt after the compute
    # commit.  Keep it untracked while committing the later manuscript bytes.
    authorization = {
        "protocol": {
            "authoritative_commit": original_commit,
            "authoritative_markdown_sha256": hashlib.sha256(
                original_markdown
            ).hexdigest(),
            "final_prelabel_commit": final_commit,
            "seal": _binding(verifier, source, verifier.PROTOCOL_SEAL_PATH),
        },
        "registries": {
            "candidate_table": {"path": input_paths["candidate_table"]},
            "candidate_provenance": {"path": input_paths["candidate_provenance"]},
            "candidate_snapshot_index": {
                "path": input_paths["candidate_snapshot_index"]
            },
            "external": {"path": input_paths["external_registry"]},
            "external_lock": {"path": input_paths["external_lock"]},
        },
        "model_suite": {"path": model_suite_path},
        "development_replay": {"path": development_replay_path},
        "actual_inputs": {"path": input_paths["input_manifest"]},
        "prelabel_chronology": {
            **_binding(verifier, source, verifier.CHRONOLOGY_PATH),
            "format": verifier.CHRONOLOGY_FORMAT,
            "status": verifier.CHRONOLOGY_STATUS,
            "order": chronology_order,
            "evidence_scope": verifier.CHRONOLOGY_EVIDENCE_SCOPE,
        },
        "inference_amendment": {
            **_binding(verifier, source, inference_amendment_path),
            "seal": _binding(verifier, source, inference_amendment_seal_path),
            "final_prelabel_commit": amendment_commit,
        },
        "probability_metric_erratum": {
            **_binding(
                verifier, source, verifier.PROBABILITY_METRIC_ERRATUM_PATH
            ),
            "format": probability_erratum["format"],
            "erratum_id": probability_erratum["erratum_id"],
            "seal": _binding(
                verifier,
                source,
                verifier.PROBABILITY_METRIC_ERRATUM_SEAL_PATH,
            ),
            "erratum_document_commit": erratum_document_commit,
        },
        "model_matrix_amendment": {
            **_binding(
                verifier, source, verifier.MODEL_MATRIX_AMENDMENT_PATH
            ),
            "format": verifier.MODEL_MATRIX_AMENDMENT_FORMAT,
            "status": verifier.MODEL_MATRIX_AMENDMENT_STATUS,
            "amendment_id": verifier.MODEL_MATRIX_AMENDMENT_ID,
            "seal": _binding(
                verifier,
                source,
                verifier.MODEL_MATRIX_AMENDMENT_SEAL_PATH,
            ),
            "amendment_document_commit": model_matrix_document_commit,
        },
        "inference_gate": _binding(verifier, source, inference_gate_path),
        "runtime": {
            "requirements_lock": _binding(
                verifier, source, "requirements-lock.txt"
            ),
            "hashed_requirements_lock": _binding(
                verifier, source, verifier.REPRODUCIBILITY_LOCK
            ),
            "python_executable": {
                "invoked_path": "/fixture/python",
                "realpath": "/fixture/python-real",
                "sha256": "d" * 64,
            },
        },
        "fixed_code": {
            "format": "thermoroute.route-a-fixed-code.v1",
            **fixed_stable,
            "sha256": verifier._sha256_json(fixed_stable),
        },
        "source": {
            "git_commit_before_authorization": compute_commit,
            "source_tree_sha256": source_tree_sha256,
            "source_inventory": model_source_inventory,
        },
    }
    write_json(authorization_path, authorization)

    manuscript = _write_bytes(
        source,
        verifier.POSTOPEN_CLAIM_DOCUMENT,
        b"post-opening manuscript only\n",
    )
    subprocess.run(
        ["git", "add", verifier.POSTOPEN_CLAIM_DOCUMENT],
        cwd=source,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "render post-opening manuscript"],
        cwd=source,
        env=environment,
        check=True,
    )
    manuscript_commit = git_text("rev-parse", "HEAD")
    assert len({
        original_commit,
        final_commit,
        amendment_commit,
        seal_commit,
        erratum_document_commit,
        erratum_seal_commit,
        model_matrix_document_commit,
        model_matrix_seal_commit,
        model_commit,
        input_commit,
        receipt_base_commit,
        compute_commit,
        manuscript_commit,
    }) == 13

    stage = tmp_path / "chronology-stage"
    shutil.copytree(source, stage, ignore=shutil.ignore_patterns(".git"))
    document_binding = {
        **_binding(verifier, stage, verifier.POSTOPEN_CLAIM_DOCUMENT),
        "bytes": manuscript.stat().st_size,
    }
    _write_bytes(
        stage,
        verifier.PROFILE_MARKER,
        json.dumps({
            "profile": verifier.POSTOPEN_PROFILE,
            "authorization": {"path": authorization_path},
            "authorized_worktree_dirt_policy": {
                "compute_commit": compute_commit,
                "manuscript_commit": manuscript_commit,
                "committed_document_diff": [document_binding],
            },
        }, sort_keys=True).encode() + b"\n",
    )

    evidence = verifier.materialize_git_history_evidence(
        source, stage, verifier.POSTOPEN_PROFILE
    )
    receipt_oid = git_text(
        "rev-parse", f"{compute_commit}:{verifier.CHRONOLOGY_PATH}"
    )
    assert evidence["prelabel_chronology"] == {
        "receipt": {
            **_binding(verifier, stage, verifier.CHRONOLOGY_PATH),
            "bytes": (stage / verifier.CHRONOLOGY_PATH).stat().st_size,
        },
        "receipt_commit": compute_commit,
        "receipt_git_blob_oid": receipt_oid,
        "order": chronology_order,
        "model_source_control_artifact_count": len(model_tree_paths),
        "model_freeze_artifact_count": len(model_artifact_paths),
        "input_evidence_artifact_count": len(input_artifact_paths),
    }

    relocated = tmp_path / "relocated" / "release"
    shutil.copytree(stage, relocated)
    marker = json.loads(
        (relocated / verifier.PROFILE_MARKER).read_text(encoding="utf-8")
    )
    verifier._verify_git_history_evidence(
        relocated, marker, verifier.POSTOPEN_PROFILE
    )

    # Exercise the other two mandatory checks in the same unpatched test:
    # Stage 27 and exact raw-NWIS-to-Parquet reconstruction.
    combined_source = tmp_path / "combined-postopen-source"
    combined_source.mkdir()
    combined_authorization_path, _ = _write_postopen_fixture(
        verifier, combined_source
    )
    combined_categories, _, _ = verifier._gather_postopen_categories(
        combined_source, combined_authorization_path
    )
    assert combined_categories["model_suite"]
    assert combined_categories["model_bundles"]
    assert combined_categories["raw_nwis"]
    assert combined_categories["normalized_outcomes"]

    # Recomputing mutable archive JSON hashes after replacing executable code
    # must not defeat the immutable compute-commit blob comparison.
    attacked = tmp_path / "attacked-release"
    shutil.copytree(relocated, attacked)
    attacked_opening = attacked / "src/thermoroute/opening.py"
    attacked_opening.write_bytes(b"# attacker-replaced executable\n")
    attacked_sha = verifier.sha256_file(attacked_opening)
    attacked_chronology_path = attacked / verifier.CHRONOLOGY_PATH
    attacked_chronology = json.loads(
        attacked_chronology_path.read_text(encoding="utf-8")
    )
    for binding in attacked_chronology["model_source_control_artifacts"]:
        if binding["path"] == "src/thermoroute/opening.py":
            payload = attacked_opening.read_bytes()
            binding["sha256"] = attacked_sha
            binding["byte_count"] = len(payload)
            binding["git_blob_oid"] = hashlib.sha1(
                f"blob {len(payload)}\0".encode() + payload
            ).hexdigest()
    attacked_inventory = {
        item["path"]: item["sha256"]
        for item in attacked_chronology["model_source_control_artifacts"]
        if verifier._matches_source_inventory(item["path"])
    }
    attacked_tree_sha = verifier._sha256_json(attacked_inventory)
    attacked_chronology["source_tree_sha256"] = attacked_tree_sha
    attacked_chronology.pop("receipt_self_sha256")
    attacked_chronology["receipt_self_sha256"] = (
        verifier._chronology_self_sha256(attacked_chronology)
    )
    attacked_chronology_path.write_text(
        json.dumps(attacked_chronology, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    attacked_authorization_path = attacked / authorization_path
    attacked_authorization = json.loads(
        attacked_authorization_path.read_text(encoding="utf-8")
    )
    attacked_authorization["prelabel_chronology"]["sha256"] = (
        verifier.sha256_file(attacked_chronology_path)
    )
    attacked_authorization["source"]["source_inventory"] = attacked_inventory
    attacked_authorization["source"]["source_tree_sha256"] = attacked_tree_sha
    attacked_authorization["fixed_code"]["modules"][
        "thermoroute.opening"
    ]["sha256"] = attacked_sha
    attacked_fixed_stable = {
        group: attacked_authorization["fixed_code"][group]
        for group in ("modules", "files", "entrypoints")
    }
    attacked_authorization["fixed_code"]["sha256"] = verifier._sha256_json(
        attacked_fixed_stable
    )
    attacked_authorization_path.write_text(
        json.dumps(attacked_authorization, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    attacked_marker = json.loads(
        (attacked / verifier.PROFILE_MARKER).read_text(encoding="utf-8")
    )
    with pytest.raises(ValueError, match="compute Git blob"):
        verifier._verify_git_history_evidence(
            attacked, attacked_marker, verifier.POSTOPEN_PROFILE
        )

    added = tmp_path / "added-source-release"
    shutil.copytree(relocated, added)
    added_source = _write_bytes(
        added, "src/thermoroute/injected.py", b"INJECTED = True\n"
    )
    added_authorization_path = added / authorization_path
    added_authorization = json.loads(
        added_authorization_path.read_text(encoding="utf-8")
    )
    added_inventory = dict(added_authorization["source"]["source_inventory"])
    added_inventory["src/thermoroute/injected.py"] = verifier.sha256_file(
        added_source
    )
    added_authorization["source"]["source_inventory"] = added_inventory
    added_authorization["source"]["source_tree_sha256"] = verifier._sha256_json(
        added_inventory
    )
    added_authorization_path.write_text(
        json.dumps(added_authorization, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    added_marker = json.loads(
        (added / verifier.PROFILE_MARKER).read_text(encoding="utf-8")
    )
    with pytest.raises(ValueError, match="path set differs from compute Git tree"):
        verifier._verify_git_history_evidence(
            added, added_marker, verifier.POSTOPEN_PROFILE
        )

    tampered = json.loads(json.dumps(marker))
    tampered["git_history_evidence"]["prelabel_chronology"]["order"][
        "model_freeze_commit"
    ] = input_commit
    with pytest.raises(ValueError, match="chronology evidence differs from its receipt"):
        verifier._verify_git_history_evidence(
            relocated, tampered, verifier.POSTOPEN_PROFILE
        )

    def reseal_chronology_attack(
        attack_root: Path, chronology_document: dict
    ) -> dict:
        chronology_document.pop("receipt_self_sha256", None)
        chronology_document["receipt_self_sha256"] = (
            verifier._chronology_self_sha256(chronology_document)
        )
        chronology_attack_path = attack_root / verifier.CHRONOLOGY_PATH
        chronology_attack_path.write_bytes(
            verifier._canonical_json_bytes(chronology_document)
        )
        authorization_attack_path = attack_root / authorization_path
        authorization_attack = json.loads(
            authorization_attack_path.read_text(encoding="utf-8")
        )
        authorization_attack["prelabel_chronology"]["sha256"] = (
            verifier.sha256_file(chronology_attack_path)
        )
        authorization_attack_path.write_bytes(
            verifier._canonical_json_bytes(authorization_attack)
        )
        attack_marker_path = attack_root / verifier.PROFILE_MARKER
        attack_marker = json.loads(
            attack_marker_path.read_text(encoding="utf-8")
        )
        attack_marker["git_history_evidence"]["prelabel_chronology"][
            "receipt"
        ] = verifier._binding_for(attack_root, chronology_attack_path)
        attack_marker_path.write_bytes(
            verifier._canonical_json_bytes(attack_marker)
        )
        return attack_marker

    # The seal must be a separate strict ancestor of model freeze, not merely
    # another name for the model-freeze commit in consistently rehashed JSON.
    seal_at_model = tmp_path / "seal-at-model-release"
    shutil.copytree(relocated, seal_at_model)
    seal_at_model_chronology = json.loads(
        (seal_at_model / verifier.CHRONOLOGY_PATH).read_text(encoding="utf-8")
    )
    seal_at_model_chronology["model_matrix_history"]["seal_commit"] = (
        model_commit
    )
    seal_at_model_marker = reseal_chronology_attack(
        seal_at_model, seal_at_model_chronology
    )
    with pytest.raises(ValueError, match="model-matrix history changed"):
        verifier._verify_git_history_evidence(
            seal_at_model, seal_at_model_marker, verifier.POSTOPEN_PROFILE
        )

    # Coordinate the forged ID across both mutable JSON copies and refresh
    # every local byte binding.  The independent amendment-derived ID must
    # still reject it before trusting either metadata copy.
    coordinated = tmp_path / "coordinated-matrix-release"
    shutil.copytree(relocated, coordinated)
    coordinated_suite_path = coordinated / model_suite_path
    coordinated_suite = json.loads(
        coordinated_suite_path.read_text(encoding="utf-8")
    )
    forged_contract_id = "f" * 64
    coordinated_suite["model_matrix_amendment"]["contract_id"] = (
        forged_contract_id
    )
    coordinated_suite_path.write_bytes(
        json.dumps(coordinated_suite, sort_keys=True).encode("utf-8") + b"\n"
    )
    coordinated_chronology = json.loads(
        (coordinated / verifier.CHRONOLOGY_PATH).read_text(encoding="utf-8")
    )
    coordinated_chronology["model_matrix_history"]["contract_id"] = (
        forged_contract_id
    )
    for item in coordinated_chronology["model_freeze_artifacts"]:
        if item["path"] == model_suite_path:
            item.clear()
            item.update(
                _chronology_binding(verifier, coordinated, model_suite_path)
            )
            break
    coordinated_marker = reseal_chronology_attack(
        coordinated, coordinated_chronology
    )
    with pytest.raises(ValueError, match="contract ID changed"):
        verifier._verify_git_history_evidence(
            coordinated, coordinated_marker, verifier.POSTOPEN_PROFILE
        )

    history_extra = tmp_path / "matrix-history-extra-key-release"
    shutil.copytree(relocated, history_extra)
    history_extra_chronology = json.loads(
        (history_extra / verifier.CHRONOLOGY_PATH).read_text(encoding="utf-8")
    )
    history_extra_chronology["model_matrix_history"]["attacker_extra"] = True
    history_extra_marker = reseal_chronology_attack(
        history_extra, history_extra_chronology
    )
    with pytest.raises(ValueError, match="history schema changed"):
        verifier._verify_git_history_evidence(
            history_extra, history_extra_marker, verifier.POSTOPEN_PROFILE
        )

    chronology_extra = tmp_path / "chronology-top-extra-key-release"
    shutil.copytree(relocated, chronology_extra)
    chronology_extra_document = json.loads(
        (chronology_extra / verifier.CHRONOLOGY_PATH).read_text(encoding="utf-8")
    )
    chronology_extra_document["attacker_extra"] = True
    chronology_extra_marker = reseal_chronology_attack(
        chronology_extra, chronology_extra_document
    )
    with pytest.raises(ValueError, match="top-level schema changed"):
        verifier._verify_git_history_evidence(
            chronology_extra,
            chronology_extra_marker,
            verifier.POSTOPEN_PROFILE,
        )

    protocol_extra = tmp_path / "chronology-protocol-extra-key-release"
    shutil.copytree(relocated, protocol_extra)
    protocol_extra_document = json.loads(
        (protocol_extra / verifier.CHRONOLOGY_PATH).read_text(encoding="utf-8")
    )
    protocol_extra_document["protocol_history"]["attacker_extra"] = True
    protocol_extra_marker = reseal_chronology_attack(
        protocol_extra, protocol_extra_document
    )
    with pytest.raises(ValueError, match="lacks protocol history"):
        verifier._verify_git_history_evidence(
            protocol_extra,
            protocol_extra_marker,
            verifier.POSTOPEN_PROFILE,
        )

    # A forged Git receipt retaining plausible top-level PASS/source/runtime
    # fields must still fail on its nested execution contract.
    subprocess.run(
        ["git", "checkout", "-q", "--detach", model_commit],
        cwd=source,
        check=True,
    )
    forged_replay = json.loads(
        (source / development_replay_path).read_text(encoding="utf-8")
    )
    forged_replay["execution_attestation"]["fresh_pycache_policy"][
        "required"
    ] = False
    forged_replay.pop("receipt_self_sha256")
    forged_replay["receipt_self_sha256"] = verifier._sha256_json(forged_replay)
    _write_bytes(
        source,
        development_replay_path,
        verifier._lineage_canonical_json_bytes(forged_replay),
    )
    subprocess.run(
        ["git", "add", development_replay_path], cwd=source, check=True
    )
    subprocess.run(
        ["git", "commit", "-q", "--amend", "--no-edit"],
        cwd=source,
        env=environment,
        check=True,
    )
    forged_commit = git_text("rev-parse", "HEAD")
    forged_bundle = tmp_path / "forged-development-replay.bundle"
    subprocess.run(
        ["git", "bundle", "create", str(forged_bundle), "HEAD"],
        cwd=source,
        check=True,
    )
    forged_bare = tmp_path / "forged-audit.git"
    subprocess.run(["git", "init", "--bare", "-q", forged_bare], check=True)
    subprocess.run(
        [
            "git", "fetch", "-q", str(forged_bundle),
            "HEAD:refs/heads/forged",
        ],
        cwd=forged_bare,
        check=True,
    )
    with pytest.raises(ValueError, match="development replay execution identity"):
        verifier._reconstruct_model_dependency_paths(
            forged_bare,
            forged_commit,
            suite_path=model_suite_path,
            replay_path=development_replay_path,
            expected_python_identity=authorization["runtime"]["python_executable"],
            model_matrix_history=chronology["model_matrix_history"],
        )

    # A semantically identical receipt blob must still use the producer's exact
    # compact, sorted, newline-terminated JSON representation.
    forged_replay["execution_attestation"]["fresh_pycache_policy"][
        "required"
    ] = True
    forged_replay.pop("receipt_self_sha256")
    forged_replay["receipt_self_sha256"] = verifier._sha256_json(forged_replay)
    (source / development_replay_path).write_text(
        json.dumps(forged_replay, indent=2) + "\n", encoding="utf-8"
    )
    subprocess.run(
        ["git", "add", development_replay_path], cwd=source, check=True
    )
    subprocess.run(
        ["git", "commit", "-q", "--amend", "--no-edit"],
        cwd=source,
        env=environment,
        check=True,
    )
    noncanonical_commit = git_text("rev-parse", "HEAD")
    noncanonical_bundle = tmp_path / "noncanonical-development-replay.bundle"
    subprocess.run(
        ["git", "bundle", "create", str(noncanonical_bundle), "HEAD"],
        cwd=source,
        check=True,
    )
    noncanonical_bare = tmp_path / "noncanonical-audit.git"
    subprocess.run(["git", "init", "--bare", "-q", noncanonical_bare], check=True)
    subprocess.run(
        [
            "git", "fetch", "-q", str(noncanonical_bundle),
            "HEAD:refs/heads/noncanonical",
        ],
        cwd=noncanonical_bare,
        check=True,
    )
    with pytest.raises(ValueError, match="canonical producer JSON"):
        verifier._reconstruct_model_dependency_paths(
            noncanonical_bare,
            noncanonical_commit,
            suite_path=model_suite_path,
            replay_path=development_replay_path,
            expected_python_identity=authorization["runtime"]["python_executable"],
            model_matrix_history=chronology["model_matrix_history"],
        )


def _write_release_mechanics_v2_fixture(verifier, root: Path) -> tuple[Path, dict]:
    """Write one inert but byte-exact v2 mechanics receipt fixture."""
    for relative in (
        verifier.RELEASE_MECHANICS_CORE_PATH,
        verifier.RELEASE_MECHANICS_RUNNER_PATH,
        "scripts/verify_release.py",
    ):
        _write_bytes(root, relative, f"# fixture {relative}\n".encode())
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    commit = _commit_git_fixture(root, "freeze mechanics sources")
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()

    archive_relative = "dist/mécanique.zip"
    archive_path = _write_bytes(root, archive_relative, b"fixture zip bytes\n")
    archive_digest = verifier.sha256_file(archive_path)
    sidecar_path = _write_bytes(
        root,
        archive_relative + ".sha256",
        f"{archive_digest}  {archive_path.name}\n".encode("utf-8"),
    )
    python_invoked = Path(sys.executable)
    python_real = python_invoked.resolve()
    python_payload = python_real.read_bytes()
    profile_line = (
        "LOCAL EVIDENCE OK [DO NOT DISTRIBUTE; PREOPEN_NOT_COMPLETE]: <archive>"
    )
    sealed_archive = "../sealed-inputs/" + archive_path.name
    manifest_line = "manifest OK: 7 artifacts, source 012345abcdef, DAG 9 nodes"
    stdout_payload = (
        manifest_line
        + "\n"
        + profile_line.replace("<archive>", sealed_archive)
        + "\n"
    ).encode("utf-8")
    raw_peak = 4096
    if platform.system() == "Darwin":
        peak_unit, peak_bytes = "bytes", raw_peak
    elif platform.system() == "Linux":
        peak_unit, peak_bytes = "kibibytes", raw_peak * 1024
    else:
        pytest.skip("release-mechanics RSS contract supports Darwin and Linux")

    document = {
        "format": verifier.RELEASE_MECHANICS_RECEIPT_FORMAT,
        "status": verifier.RELEASE_MECHANICS_RECEIPT_STATUS,
        "profile": verifier.PREOPEN_PROFILE,
        "distribution": verifier.LOCAL_DISTRIBUTION,
        "evidence_scope": verifier.RELEASE_MECHANICS_EVIDENCE_SCOPE,
        "archive": {
            "path": archive_relative,
            "sha256": archive_digest,
            "bytes": len(archive_path.read_bytes()),
            "sidecar": verifier._binding_for(root, sidecar_path),
        },
        "source": {
            "git_commit": commit,
            "git_tree": tree,
            "git_clean_before_acceptance": True,
            "core": verifier._binding_for(
                root, root / verifier.RELEASE_MECHANICS_CORE_PATH
            ),
            "runner": verifier._binding_for(
                root, root / verifier.RELEASE_MECHANICS_RUNNER_PATH
            ),
            "verifier": verifier._binding_for(
                root, root / "scripts/verify_release.py"
            ),
        },
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "python_executable": {
                "invoked_path": str(python_invoked),
                "realpath": str(python_real),
                "sha256": hashlib.sha256(python_payload).hexdigest(),
                "bytes": len(python_payload),
            },
            "platform_system": platform.system(),
            "platform_release": platform.release(),
            "machine": platform.machine(),
        },
        "execution": {
            "command": [
                "<sealed-copy-of-bound-python-executable>",
                "-I",
                "-B",
                "<sealed-copy-of-git-bound-release-verifier>",
                "<sealed-copy-of-bound-archive>",
                "--distribution",
                verifier.LOCAL_DISTRIBUTION,
            ],
            "environment": {
                "PATH": os.defpath,
                "LANG": "C",
                "LC_ALL": "C",
                "TZ": "UTC",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "TMPDIR": "<fresh-temporary-root>",
            },
            "controller_pid": 101,
            "verifier_pid": 202,
            "python_isolated": True,
            "bytecode_disabled": True,
            "fresh_process": True,
            "fresh_working_directory": True,
            "fresh_temporary_extraction_root": True,
            "fresh_cwd_empty_before": True,
            "fresh_cwd_empty_after": True,
            "temporary_root_removed_after_exit": True,
            "sealed_archive_argument": sealed_archive,
            "process_group_cleanup_enforced": True,
            "returncode": 0,
            "stdout_sha256": hashlib.sha256(stdout_payload).hexdigest(),
            "stdout_bytes": len(stdout_payload),
            "stdout_transcript": [manifest_line, profile_line],
            "stderr_sha256": hashlib.sha256(b"").hexdigest(),
            "stderr_bytes": 0,
            "stderr_transcript": [],
            "profile_evidence_line": profile_line,
        },
        "resource_observation": {
            "wall_time_ns": 10,
            "user_cpu_time_ns": 2,
            "system_cpu_time_ns": 3,
            "total_cpu_time_ns": 5,
            "peak_rss_raw": raw_peak,
            "peak_rss_raw_unit": peak_unit,
            "peak_rss_bytes": peak_bytes,
            "measurement_backend": (
                verifier.RELEASE_MECHANICS_MEASUREMENT_BACKEND
            ),
            "measurement_scope": verifier.RELEASE_MECHANICS_MEASUREMENT_SCOPE,
            "repetitions": 1,
        },
        "isolation_limitations": dict(
            verifier.RELEASE_MECHANICS_ISOLATION_LIMITATIONS
        ),
        "authentication_limitations": dict(
            verifier.RELEASE_MECHANICS_AUTHENTICATION_LIMITATIONS
        ),
        "label_safety": dict(verifier.RELEASE_MECHANICS_LABEL_SAFETY),
    }
    document["receipt_self_sha256"] = (
        verifier._release_mechanics_self_sha256(document)
    )
    receipt = _write_canonical_json(
        verifier, root, verifier.RELEASE_MECHANICS_RECEIPT_PATH, document
    )
    return receipt, document


@pytest.mark.parametrize(
    ("attack", "message"),
    (
        ("v1", "version|scope|limitations"),
        ("top_extra", "exact schema"),
        ("nested_extra", "execution schema"),
        ("limitation", "version|scope|limitations"),
        ("duplicate", "duplicate JSON key"),
    ),
)
def test_independent_release_mechanics_v2_rejects_resealed_schema_attacks(
    tmp_path, attack, message
):
    verifier = _load_script(
        VERIFY_SCRIPT, f"thermoroute_verify_mechanics_{attack}_test"
    )
    root = tmp_path / "mechanics"
    root.mkdir()
    receipt, document = _write_release_mechanics_v2_fixture(verifier, root)
    assert verifier.validate_release_mechanics_acceptance(root, receipt) == document
    if attack == "v1":
        cli = subprocess.run(
            [
                sys.executable,
                str(VERIFY_SCRIPT),
                "--validate-release-mechanics-receipt",
                verifier.RELEASE_MECHANICS_RECEIPT_PATH,
                "--source-root",
                str(root),
            ],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        assert cli.returncode == 0, cli.stderr
        assert verifier.RELEASE_MECHANICS_RECEIPT_STATUS in cli.stdout

    if attack == "duplicate":
        receipt.write_bytes(
            receipt.read_bytes().replace(
                b"{", b'{"format":"thermoroute.attacker-alias.v1",', 1
            )
        )
    else:
        if attack == "v1":
            document["format"] = (
                "thermoroute.same-host-release-mechanics-acceptance.v1"
            )
        elif attack == "top_extra":
            document["attacker_extra"] = True
        elif attack == "nested_extra":
            document["execution"]["attacker_extra"] = True
        else:
            document["isolation_limitations"]["fresh_machine"] = True
        document.pop("receipt_self_sha256")
        document["receipt_self_sha256"] = (
            verifier._release_mechanics_self_sha256(document)
        )
        receipt.write_bytes(verifier._canonical_json_bytes(document))
    with pytest.raises(ValueError, match=message):
        verifier.validate_release_mechanics_acceptance(
            root, verifier.RELEASE_MECHANICS_RECEIPT_PATH
        )


@pytest.mark.parametrize("number", ("1e9999", "-1e9999"))
def test_strict_release_json_rejects_finite_syntax_that_overflows(number):
    verifier = _load_script(
        VERIFY_SCRIPT,
        "thermoroute_verify_overflow_"
        + ("negative" if number.startswith("-") else "positive"),
    )
    with pytest.raises(ValueError, match="non-finite JSON number"):
        verifier._strict_json_object_bytes(
            f'{{"value":{number}}}'.encode("ascii"),
            label="overflow fixture",
        )


def test_model_matrix_uses_fixed_document_and_exact_governance_registry(tmp_path):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_fixed_model_matrix_test"
    )
    amendment_path = ROOT / verifier.MODEL_MATRIX_AMENDMENT_PATH
    seal_path = ROOT / verifier.MODEL_MATRIX_AMENDMENT_SEAL_PATH
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    assert verifier.sha256_file(amendment_path) == (
        verifier.MODEL_MATRIX_AMENDMENT_SHA256
    )
    assert seal["governance_seals"] == verifier.MODEL_MATRIX_GOVERNANCE_SEALS
    assert verifier._validate_model_matrix_documents(
        amendment,
        seal,
        amendment_sha256=verifier.MODEL_MATRIX_AMENDMENT_SHA256,
        amendment_document_commit=seal["amendment_document_commit"],
    ) == verifier._model_matrix_contract_id(amendment)

    for attack in ("document", "governance_missing", "governance_extra", "governance_changed"):
        attacked_amendment = json.loads(json.dumps(amendment))
        attacked_seal = json.loads(json.dumps(seal))
        if attack == "document":
            attacked_amendment["scientific_scope"]["attacker_reseal"] = True
            attacked_payload = verifier._canonical_json_bytes(attacked_amendment)
            attacked_digest = hashlib.sha256(attacked_payload).hexdigest()
            attacked_seal["amendment"]["sha256"] = attacked_digest
        else:
            attacked_digest = verifier.MODEL_MATRIX_AMENDMENT_SHA256
            governance = attacked_seal["governance_seals"]
            if attack == "governance_missing":
                governance.pop("inference_amendment_seal_v1")
            elif attack == "governance_extra":
                governance["attacker"] = {
                    "path": "protocols/attacker.json",
                    "sha256": "f" * 64,
                }
            else:
                governance["base_protocol_seal"]["sha256"] = "f" * 64
        with pytest.raises(ValueError, match="amendment or seal semantics"):
            verifier._validate_model_matrix_documents(
                attacked_amendment,
                attacked_seal,
                amendment_sha256=attacked_digest,
                amendment_document_commit=seal["amendment_document_commit"],
            )


def test_real_git_model_matrix_rejects_coordinated_document_and_seal_rehash(
    tmp_path,
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_real_git_fixed_matrix_test"
    )
    source = tmp_path / "source"
    bare = tmp_path / "audit.git"
    subprocess.run(
        ["git", "clone", "-q", "--no-local", str(ROOT), str(source)],
        check=True,
    )
    subprocess.run(
        ["git", "clone", "-q", "--bare", str(source), str(bare)],
        check=True,
    )
    compute_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    verifier._verify_model_matrix_amendment_history_from_bundle(
        root=source,
        bare=bare,
        compute_commit=compute_commit,
    )

    # Build an actual forged Git history, not merely a coordinated filesystem
    # rewrite: prior governance seals -> forged document creation -> separately
    # committed forged seal -> later release tip.
    attacked = tmp_path / "attacked"
    attacked.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=attacked, check=True)
    subprocess.run(
        ["git", "config", "user.name", "Route A adversary"],
        cwd=attacked,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "adversary@example.invalid"],
        cwd=attacked,
        check=True,
    )
    for binding in verifier.MODEL_MATRIX_GOVERNANCE_SEALS.values():
        relative = binding["path"]
        destination = attacked / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
    subprocess.run(["git", "add", "protocols"], cwd=attacked, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "freeze prior governance"],
        cwd=attacked,
        check=True,
    )

    attacked_amendment_path = attacked / verifier.MODEL_MATRIX_AMENDMENT_PATH
    attacked_seal_path = attacked / verifier.MODEL_MATRIX_AMENDMENT_SEAL_PATH
    attacked_amendment_path.parent.mkdir(parents=True, exist_ok=True)
    attacked_amendment = json.loads(
        (source / verifier.MODEL_MATRIX_AMENDMENT_PATH).read_text(encoding="utf-8")
    )
    attacked_amendment["scientific_scope"]["attacker_reseal"] = True
    attacked_amendment_path.write_bytes(
        verifier._canonical_json_bytes(attacked_amendment)
    )
    subprocess.run(
        ["git", "add", verifier.MODEL_MATRIX_AMENDMENT_PATH],
        cwd=attacked,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "forge model matrix document"],
        cwd=attacked,
        check=True,
    )
    forged_document_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=attacked,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()

    attacked_seal = json.loads(
        (source / verifier.MODEL_MATRIX_AMENDMENT_SEAL_PATH).read_text(
            encoding="utf-8"
        )
    )
    attacked_seal["amendment"]["sha256"] = verifier.sha256_file(
        attacked_amendment_path
    )
    attacked_seal["amendment_document_commit"] = forged_document_commit
    attacked_seal_path.parent.mkdir(parents=True, exist_ok=True)
    attacked_seal_path.write_bytes(verifier._canonical_json_bytes(attacked_seal))
    subprocess.run(
        ["git", "add", verifier.MODEL_MATRIX_AMENDMENT_SEAL_PATH],
        cwd=attacked,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "forge matching matrix seal"],
        cwd=attacked,
        check=True,
    )
    forged_seal_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=attacked,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "commit", "-q", "--allow-empty", "-m", "forged release tip"],
        cwd=attacked,
        check=True,
    )
    attacked_tip = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=attacked,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    attacked_bare = tmp_path / "attacked-audit.git"
    subprocess.run(
        ["git", "clone", "-q", "--bare", str(attacked), str(attacked_bare)],
        check=True,
    )
    assert subprocess.run(
        ["git", "merge-base", "--is-ancestor", forged_document_commit, forged_seal_commit],
        cwd=attacked,
        check=False,
    ).returncode == 0
    assert subprocess.run(
        ["git", "merge-base", "--is-ancestor", forged_seal_commit, attacked_tip],
        cwd=attacked,
        check=False,
    ).returncode == 0
    with pytest.raises(ValueError, match="exact model-matrix amendment binding"):
        verifier._verify_model_matrix_amendment_history_from_bundle(
            root=attacked,
            bare=attacked_bare,
            compute_commit=attacked_tip,
        )


def test_opening_preflight_and_validator_reject_coordinated_nested_reseals(
    tmp_path,
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_opening_preflight_exact_test"
    )
    root = tmp_path / "postopen"
    root.mkdir()
    authorization_path, _representatives = _write_postopen_fixture(verifier, root)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    state = authorization["state_paths"]
    suite = json.loads(
        (root / authorization["model_suite"]["path"]).read_text(encoding="utf-8")
    )
    intent = json.loads((root / state["intent"]).read_text(encoding="utf-8"))
    receipt = json.loads((root / state["receipt"]).read_text(encoding="utf-8"))
    verifier._validate_opening_preflight_identity(
        root, authorization_path, authorization, suite, intent, receipt
    )

    attacks = (
        ("preflight_missing", "protocol_sha256"),
        ("preflight_extra", "attacker_extra"),
        ("preflight_changed", "protocol_sha256"),
        ("validator_missing", verifier.TRUSTED_VALIDATOR_PATHS[0]),
        ("validator_extra", "src/thermoroute/attacker.py"),
        ("validator_changed", verifier.TRUSTED_VALIDATOR_PATHS[0]),
        ("validator_source_tree", "source_tree_sha256"),
    )
    for attack, field in attacks:
        attacked_intent = json.loads(json.dumps(intent))
        attacked_receipt = json.loads(json.dumps(receipt))
        if attack.startswith("preflight"):
            attacked_preflight = attacked_receipt["preflight_attestation"]
            if attack.endswith("missing"):
                attacked_preflight.pop(field)
            elif attack.endswith("extra"):
                attacked_preflight[field] = True
            else:
                attacked_preflight[field] = "f" * 64
            digest = verifier._sha256_json(attacked_preflight)
            attacked_receipt["preflight_attestation_sha256"] = digest
            attacked_intent["preflight_attestation_sha256"] = digest
            message = "preflight attestation differs"
        else:
            attacked_validator = attacked_receipt["trusted_validator"]
            files = attacked_validator["files"]
            if attack.endswith("source_tree"):
                attacked_validator["source_tree_sha256"] = "f" * 64
            elif attack.endswith("missing"):
                files.pop(field)
            elif attack.endswith("extra"):
                files[field] = "f" * 64
            else:
                files[field] = "f" * 64
            attacked_validator["sha256"] = verifier._sha256_json(files)
            attacked_intent["trusted_validator"] = json.loads(
                json.dumps(attacked_validator)
            )
            message = "trusted-validator identity differs"
        with pytest.raises(ValueError, match=message):
            verifier._validate_opening_preflight_identity(
                root,
                authorization_path,
                authorization,
                suite,
                attacked_intent,
                attacked_receipt,
            )


def _reseal_fixture_authorization(verifier, path: Path, authorization: dict) -> None:
    created_at = authorization.pop("created_at_utc")
    authorization.pop("opening_id", None)
    authorization.pop("authorization_self_sha256", None)
    authorization["opening_id"] = verifier._sha256_json(authorization)[:24]
    authorization["created_at_utc"] = created_at
    authorization["authorization_self_sha256"] = verifier._sha256_json(
        authorization
    )
    path.write_bytes(verifier._canonical_json_bytes(authorization))


def test_authorization_recomputes_content_addressed_state_namespace(tmp_path):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_state_namespace_content_test"
    )
    root = tmp_path / "postopen"
    root.mkdir()
    authorization_path, _ = _write_postopen_fixture(verifier, root)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    old_base = authorization["state_paths"]["run_directory"]
    new_namespace = "f" * 24
    new_base = f"outputs/confirmatory/route_a_{new_namespace}"
    authorization["state_paths"] = {
        key: (
            new_namespace
            if key == "namespace"
            else value.replace(old_base, new_base, 1)
        )
        for key, value in authorization["state_paths"].items()
    }
    _reseal_fixture_authorization(verifier, authorization_path, authorization)
    with pytest.raises(ValueError, match="content-addressed canonical namespace"):
        verifier._validate_authorization_structure(root, authorization_path)


@pytest.mark.parametrize(
    ("attack", "message"),
    (
        ("feature_order", "actual feature order differs"),
        ("statistics_contract", "statistics contract differs"),
    ),
)
def test_postopen_gather_cross_checks_authorization_scientific_identity(
    tmp_path, attack, message
):
    verifier = _load_script(
        VERIFY_SCRIPT,
        f"thermoroute_verify_authorization_science_{attack}_test",
    )
    root = tmp_path / attack
    root.mkdir()
    authorization_path, _ = _write_postopen_fixture(verifier, root)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if attack == "feature_order":
        authorization["actual_feature_order"] = ["ATTACKER_FEATURE"]
    else:
        authorization["statistics_contract_sha256"] = "f" * 64
    _reseal_fixture_authorization(verifier, authorization_path, authorization)
    with pytest.raises(ValueError, match=message):
        verifier._gather_postopen_categories(root, authorization_path)


@pytest.mark.parametrize(
    "field",
    (
        "required_gate_files_at_model_freeze",
        "model_source_control_artifacts",
        "model_freeze_artifacts",
        "input_evidence_artifacts",
    ),
)
def test_release_chronology_rejects_resealed_binding_list_reordering(
    tmp_path, field
):
    verifier = _load_script(
        VERIFY_SCRIPT,
        f"thermoroute_verify_chronology_order_{field}_test",
    )
    root = tmp_path / field
    root.mkdir()
    authorization_path, _ = _write_postopen_fixture(verifier, root)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    chronology_path = root / verifier.CHRONOLOGY_PATH
    chronology = json.loads(chronology_path.read_text(encoding="utf-8"))
    chronology[field].reverse()
    chronology.pop("receipt_self_sha256")
    chronology["receipt_self_sha256"] = verifier._chronology_self_sha256(
        chronology
    )
    chronology_path.write_bytes(verifier._canonical_json_bytes(chronology))
    authorization["prelabel_chronology"]["sha256"] = verifier.sha256_file(
        chronology_path
    )
    authorization["state_paths"] = verifier._expected_authorization_state_paths(
        authorization
    )
    _reseal_fixture_authorization(verifier, authorization_path, authorization)

    with pytest.raises(ValueError, match="canonical producer order"):
        verifier._validate_prelabel_chronology_structure(
            root, {}, authorization
        )


def test_independent_chronology_rejects_json_aliases_after_binding_reseal(
    tmp_path,
):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_chronology_json_alias_test"
    )
    base = tmp_path / "base-postopen"
    base.mkdir()
    authorization_path, _ = _write_postopen_fixture(verifier, base)
    authorization_relative = authorization_path.relative_to(base)

    def rewrite_authorization(root: Path, authorization: dict) -> None:
        if "authorization_self_sha256" in authorization:
            authorization.pop("authorization_self_sha256")
            authorization["authorization_self_sha256"] = verifier._sha256_json(
                authorization
            )
        (root / authorization_relative).write_bytes(
            verifier._canonical_json_bytes(authorization)
        )

    for attack, message in (
        ("duplicate_chronology", "duplicate JSON key"),
        ("pretty_chronology", "canonical producer JSON"),
        ("duplicate_suite", "duplicate JSON key"),
    ):
        root = tmp_path / attack
        shutil.copytree(base, root)
        authorization = json.loads(
            (root / authorization_relative).read_text(encoding="utf-8")
        )
        chronology_path = root / verifier.CHRONOLOGY_PATH
        if attack == "duplicate_chronology":
            chronology_path.write_bytes(
                chronology_path.read_bytes().replace(
                    b"{", b'{"format":"thermoroute.attacker-alias.v1",', 1
                )
            )
        elif attack == "pretty_chronology":
            chronology = json.loads(chronology_path.read_text(encoding="utf-8"))
            chronology_path.write_text(
                json.dumps(chronology, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            suite_relative = authorization["model_suite"]["path"]
            suite_path = root / suite_relative
            suite_path.write_bytes(
                suite_path.read_bytes().replace(
                    b"{", b'{"format":"thermoroute.attacker-alias.v1",', 1
                )
            )
            authorization["model_suite"]["sha256"] = verifier.sha256_file(
                suite_path
            )
            chronology = json.loads(chronology_path.read_text(encoding="utf-8"))
            for item in chronology["model_freeze_artifacts"]:
                if item["path"] == suite_relative:
                    item.clear()
                    item.update(
                        _chronology_binding(verifier, root, suite_relative)
                    )
                    break
            chronology.pop("receipt_self_sha256")
            chronology["receipt_self_sha256"] = (
                verifier._chronology_self_sha256(chronology)
            )
            chronology_path.write_bytes(
                verifier._canonical_json_bytes(chronology)
            )
        authorization["prelabel_chronology"]["sha256"] = (
            verifier.sha256_file(chronology_path)
        )
        rewrite_authorization(root, authorization)
        with pytest.raises(ValueError, match=message):
            verifier._validate_prelabel_chronology_structure(
                root, {}, authorization
            )


def test_opening_identity_is_portable_in_real_python_i_process(tmp_path):
    relocated = tmp_path / "different-absolute-root" / "release"
    shutil.copytree(ROOT / "src", relocated / "src")
    scripts = relocated / "scripts"
    scripts.mkdir(parents=True)
    for name in (
        "route_a_opening_orchestrator.py",
        "route_a_outcome_acquisition.py",
        "route_a_trusted_scorer.py",
    ):
        shutil.copy2(ROOT / "scripts" / name, scripts / name)
    program = r"""
import copy
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "src"))
from thermoroute.opening import (
    OpeningContractError,
    _fixed_code_identity,
    _validate_portable_fixed_code_identity,
    _validate_portable_runtime_identity,
    sha256_json,
)

assert sys.flags.isolated
current = _fixed_code_identity(root)
frozen = copy.deepcopy(current)
for group in ("modules", "files", "entrypoints"):
    for binding in frozen[group].values():
        binding["realpath"] = "/sealed-on-another-machine/" + binding["path"]
stable = {group: frozen[group] for group in ("modules", "files", "entrypoints")}
frozen["sha256"] = sha256_json(stable)
_validate_portable_fixed_code_identity(frozen, current)

runtime = {
    "format": "fixture-runtime",
    "python_executable": {
        "invoked_path": sys.executable,
        "realpath": str(Path(sys.executable).resolve()),
        "sha256": "a" * 64,
    },
    "stable_contract": {"cpu": True},
}
frozen_runtime = copy.deepcopy(runtime)
frozen_runtime["python_executable"]["invoked_path"] = "/sealed/python"
frozen_runtime["python_executable"]["realpath"] = "/sealed/python-real"
_validate_portable_runtime_identity(frozen_runtime, runtime)

tampered = copy.deepcopy(frozen)
tampered["entrypoints"]["trusted_scorer"]["sha256"] = "0" * 64
tampered["sha256"] = sha256_json({
    group: tampered[group] for group in ("modules", "files", "entrypoints")
})
try:
    _validate_portable_fixed_code_identity(tampered, current)
except OpeningContractError:
    pass
else:
    raise AssertionError("relocation policy accepted changed trusted-scorer bytes")
print(json.dumps({"isolated": True, "relocated_root": str(root)}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program, str(relocated)],
        cwd=relocated,
        env={
            "PATH": os.defpath,
            "LANG": "C",
            "LC_ALL": "C",
            "TZ": "UTC",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    status = json.loads(result.stdout)
    assert status == {"isolated": True, "relocated_root": str(relocated.resolve())}


def test_release_materialization_rejects_hidden_git_index_flags(tmp_path):
    verifier = _load_script(
        VERIFY_SCRIPT, "thermoroute_verify_hidden_index_release_test"
    )
    source = tmp_path / "source"
    stage = tmp_path / "stage"
    source.mkdir()
    stage.mkdir()
    _minimal_canonical_release(verifier, source)
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "add", "-A"], cwd=source, check=True)
    subprocess.run(
        [
            "git", "-c", "user.name=Fixture",
            "-c", "user.email=fixture@example.invalid",
            "commit", "-q", "-m", "fixture",
        ],
        cwd=source,
        check=True,
    )
    relative = "requirements-lock-py312-hashed.txt"
    for enable, disable in (
        ("--assume-unchanged", "--no-assume-unchanged"),
        ("--skip-worktree", "--no-skip-worktree"),
    ):
        subprocess.run(
            ["git", "update-index", enable, relative], cwd=source, check=True
        )
        with pytest.raises(
            ValueError, match="assume-unchanged/skip-worktree"
        ):
            verifier.materialize_release_profile(
                source, stage, verifier.PREOPEN_PROFILE
            )
        subprocess.run(
            ["git", "update-index", disable, relative], cwd=source, check=True
        )
    _write_bytes(
        stage,
        verifier.PROFILE_MARKER,
        json.dumps({"profile": verifier.PREOPEN_PROFILE}).encode(),
    )
    dirty = _write_bytes(source, "untracked-release-dirt.txt")
    with pytest.raises(ValueError, match="clean Git worktree"):
        verifier.materialize_git_history_evidence(
            source, stage, verifier.PREOPEN_PROFILE
        )
    dirty.unlink()


def test_release_git_audit_rejects_history_overlays_and_ambient_redirects(
    tmp_path, monkeypatch
):
    verifier = _load_script(VERIFY_SCRIPT, "thermoroute_verify_git_overlay_test")
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    environment = os.environ.copy()
    environment.update({
        "GIT_AUTHOR_NAME": "Fixture",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_NAME": "Fixture",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
    })
    _write_bytes(root, "tracked.txt", b"one\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "one"], cwd=root,
        env=environment, check=True,
    )
    (root / "tracked.txt").write_bytes(b"two\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "two"], cwd=root,
        env=environment, check=True,
    )
    verifier._assert_safe_git_repository(root)

    subprocess.run(["git", "replace", "HEAD", "HEAD^"], cwd=root, check=True)
    with pytest.raises(ValueError, match="replacement refs"):
        verifier._assert_safe_git_repository(root)
    subprocess.run(["git", "replace", "-d", "HEAD"], cwd=root, check=True)

    graft = root / ".git/info/grafts"
    graft.parent.mkdir(parents=True, exist_ok=True)
    graft.write_text("forbidden\n", encoding="utf-8")
    with pytest.raises(ValueError, match="legacy grafts"):
        verifier._assert_safe_git_repository(root)
    graft.unlink()

    alternates = root / ".git/objects/info/alternates"
    alternates.parent.mkdir(parents=True, exist_ok=True)
    alternates.write_text("/tmp/forbidden\n", encoding="utf-8")
    with pytest.raises(ValueError, match="object alternates"):
        verifier._assert_safe_git_repository(root)
    alternates.unlink()

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    shallow = root / ".git/shallow"
    shallow.write_text(head + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="shallow"):
        verifier._assert_safe_git_repository(root)
    shallow.unlink()

    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "attacker-index"))
    with pytest.raises(ValueError, match="ambient Git"):
        verifier._assert_safe_git_repository(root)


def test_postopen_git_dirt_allows_only_authorization_and_canonical_namespace(
    tmp_path, monkeypatch
):
    verifier = _load_script(VERIFY_SCRIPT, "thermoroute_verify_postopen_dirt_test")
    root = tmp_path / "repo"
    root.mkdir()
    tracked = _write_bytes(root, "tracked.txt")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
    environment = os.environ.copy()
    environment.update({
        "GIT_AUTHOR_NAME": "Fixture",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_NAME": "Fixture",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
    })
    subprocess.run(
        ["git", "commit", "-q", "-m", "fixture"],
        cwd=root,
        env=environment,
        check=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    namespace = "1" * 24
    base = f"outputs/confirmatory/route_a_{namespace}"
    state = {
        "namespace": namespace,
        "run_directory": base,
        "work_order": f"{base}/acquisition_work_order_v1.json",
        "intent": f"{base}/opening_intent_v1.json",
        "transport_root": f"{base}/transport",
        "raw_nwis_root": f"{base}/transport/raw_nwis_v1",
        "raw_nwis_snapshot_index": (
            f"{base}/transport/raw_nwis_v1/snapshot_index.json"
        ),
        "acquisition_request_map": f"{base}/acquisition/source_request_map_v1.json",
        "temporal_outcomes": f"{base}/acquisition/temporal_outcomes_v1.parquet",
        "external_outcomes": f"{base}/acquisition/external_outcomes_v1.parquet",
        "acquisition_manifest": f"{base}/acquisition/acquisition_manifest_v1.json",
        "availability_registry": f"{base}/trusted/availability_registry_v1.csv",
        "outcome_quality_audit": f"{base}/trusted/outcome_quality_audit_v1.json",
        "outcome_qc_gate": f"{base}/trusted/outcome_qc_gate_v1.json",
        "approved_target_sensitivity": f"{base}/trusted/approved_target_sensitivity_v1.json",
        "spatial_sensitivity": f"{base}/trusted/spatial_sensitivity_v1.json",
        "probabilistic_evaluation": f"{base}/trusted/probabilistic_evaluation_v2.json",
        "temporal_predictions": f"{base}/trusted/temporal_predictions_v1.parquet",
        "external_predictions": f"{base}/trusted/external_predictions_v1.parquet",
        "statistics": f"{base}/trusted/statistics_v1.json",
        "temporal_coverage_audit": (
            f"{base}/trusted/temporal_coverage_audit_v1.json"
        ),
        "report": f"{base}/trusted/report_v1.md",
        "receipt": f"{base}/opening_receipt_v1.json",
        "receipt_sha256": f"{base}/opening_receipt_v1.sha256",
    }
    authorization_relative = "data_usgs/confirmatory_opening_authorization_v1.json"
    authorization = {
        "format": verifier.AUTHORIZATION_FORMAT,
        "status": "AUTHORIZED_LABELS_STILL_SEALED",
        "protocol": {"path": "protocols/fixture.json", "sha256": "1" * 64},
        "registries": {},
        "model_suite": {
            "path": "data_usgs/confirmatory_model_suite_v1.json",
            "sha256": "2" * 64,
        },
        "development_replay": {},
        "prelabel_chronology": {
            "path": verifier.CHRONOLOGY_PATH,
            "sha256": "3" * 64,
        },
        "source": {
            "authorization_path": authorization_relative,
            "git_commit_before_authorization": head,
            "source_tree_sha256": "4" * 64,
        },
        "outcome_qc_policy": {
            "path": "protocols/route_a_outcome_qc_policy_v1.json",
            "sha256": "7" * 64,
            "format": "thermoroute.route-a-outcome-qc-policy.v1",
            "policy_id": "route-a-outcome-qc-and-influence-001",
            "required": True,
        },
        "temporal_coverage_policy": {
            "path": verifier.TEMPORAL_COVERAGE_POLICY_PATH,
            "sha256": verifier.TEMPORAL_COVERAGE_POLICY_SHA256,
            "format": verifier.TEMPORAL_COVERAGE_POLICY_FORMAT,
            "policy_id": verifier.TEMPORAL_COVERAGE_POLICY_ID,
            "status": "FROZEN_PRELABEL_OUTCOME_FREE",
            "required": True,
        },
        "inference_amendment": {
            "path": "protocols/route_a_inference_amendment_v2.json",
            "sha256": "8" * 64,
            "format": "thermoroute.route-a-inference-amendment.v2",
            "amendment_id": "route-a-prelabel-inference-cqr-015",
            "seal": {
                "path": "protocols/route_a_inference_amendment_seal_v2.json",
                "sha256": "9" * 64,
            },
            "final_prelabel_commit": head,
        },
        "probability_metric_erratum": {
            "path": verifier.PROBABILITY_METRIC_ERRATUM_PATH,
            "sha256": "c" * 64,
            "format": verifier.PROBABILITY_METRIC_ERRATUM_FORMAT,
            "erratum_id": verifier.PROBABILITY_METRIC_ERRATUM_ID,
            "seal": {
                "path": verifier.PROBABILITY_METRIC_ERRATUM_SEAL_PATH,
                "sha256": "d" * 64,
            },
            "erratum_document_commit": head,
        },
        "model_matrix_amendment": {
            "path": verifier.MODEL_MATRIX_AMENDMENT_PATH,
            "sha256": verifier.MODEL_MATRIX_AMENDMENT_SHA256,
            "format": verifier.MODEL_MATRIX_AMENDMENT_FORMAT,
            "status": verifier.MODEL_MATRIX_AMENDMENT_STATUS,
            "amendment_id": verifier.MODEL_MATRIX_AMENDMENT_ID,
            "seal": {
                "path": verifier.MODEL_MATRIX_AMENDMENT_SEAL_PATH,
                "sha256": "0" * 64,
            },
            "amendment_document_commit": head,
        },
        "inference_gate": {
            "path": "outputs/prelabel/route_a_inference_gate_v1.json",
            "sha256": "a" * 64,
            "format": "thermoroute.route-a-inference-gate.v1",
            "status": "FAIL_CLOSED_DESCRIPTIVE_ONLY",
            "claim_eligible": False,
            "analysis_mode": "FIXED_COHORT_DESCRIPTIVE_ONLY",
            "policy_sha256": "b" * 64,
        },
        "actual_inputs": {
            "path": "data_usgs/confirmatory_prelabel_inputs_v1.json",
            "sha256": "5" * 64,
        },
        "actual_feature_order": [],
        "required_models": {},
        "statistics_contract_sha256": "e" * 64,
        "runtime": {
            "format": "thermoroute.route-a-runtime.v1",
            "requirements_lock": {
                "path": "requirements-lock.txt",
                "sha256": "3" * 64,
            },
            "hashed_requirements_lock": {
                "path": verifier.REPRODUCIBILITY_LOCK,
                "sha256": "6" * 64,
            },
            "installed_version_validation": "fixture validation",
            "installed_versions": {},
            "numerical_runtime_contract": {},
            "runtime_sha256": verifier._sha256_json({}),
            "python_executable": {},
            "golden_inference_sha256": "5" * 64,
            "formal_numerical_policy": {},
            "deterministic_child_policy": {},
        },
        "fixed_code": {},
        "acquisition_plan": {},
        "state_paths": state,
    }
    authorization["state_paths"] = verifier._expected_authorization_state_paths(
        authorization
    )
    state = authorization["state_paths"]
    base = state["run_directory"]
    monkeypatch.setattr(
        verifier,
        "_validate_model_matrix_amendment_binding",
        lambda *_args, **_kwargs: (
            root / verifier.MODEL_MATRIX_AMENDMENT_PATH,
            root / verifier.MODEL_MATRIX_AMENDMENT_SEAL_PATH,
            "0" * 64,
        ),
    )
    authorization["opening_id"] = verifier._sha256_json(authorization)[:24]
    authorization["created_at_utc"] = "2026-01-01T00:00:00+00:00"
    authorization["authorization_self_sha256"] = verifier._sha256_json(authorization)
    authorization_path = root / authorization_relative
    authorization_path.parent.mkdir(parents=True)
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
    namespace_artifact = _write_bytes(root, f"{base}/opening_receipt_v1.json")

    policy = verifier.validate_postopen_git_dirt(root, authorization_path)
    assert policy["untracked_exact"] == [authorization_relative]
    assert policy["untracked_prefixes"] == [base + "/"]

    manuscript = _write_bytes(
        root,
        verifier.POSTOPEN_CLAIM_DOCUMENT,
        b"post-opening manuscript\n",
    )
    subprocess.run(
        ["git", "add", verifier.POSTOPEN_CLAIM_DOCUMENT],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "docs after opening"],
        cwd=root, env=environment, check=True,
    )
    doc_policy = verifier.validate_postopen_git_dirt(root, authorization_path)
    assert doc_policy["compute_commit"] == head
    assert doc_policy["manuscript_commit"] != head
    assert [row["path"] for row in doc_policy["committed_document_diff"]] == [
        verifier.POSTOPEN_CLAIM_DOCUMENT
    ]
    assert manuscript.is_file()

    for relative in (
        "README.md",
        "paper/ThermoRoute_paper.tex",
        "paper/ThermoRoute_paper.docx",
        "paper/ThermoRoute_paper.pdf",
        "paper/supplement.md",
    ):
        _write_bytes(root, relative, f"forbidden {relative}\n".encode())
    subprocess.run(
        ["git", "add", "README.md", "paper"], cwd=root, check=True
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "forbidden document formats"],
        cwd=root,
        env=environment,
        check=True,
    )
    with pytest.raises(ValueError, match="outside the documentation whitelist"):
        verifier.validate_postopen_git_dirt(root, authorization_path)

    hidden = _write_bytes(root, "src/hidden_then_reverted.py", b"hidden = True\n")
    subprocess.run(["git", "add", "src/hidden_then_reverted.py"], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "forbidden hidden source"],
        cwd=root, env=environment, check=True,
    )
    hidden.unlink()
    subprocess.run(
        ["git", "add", "-u", "--", "src/hidden_then_reverted.py"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "revert forbidden hidden source"],
        cwd=root, env=environment, check=True,
    )
    with pytest.raises(ValueError, match="outside the documentation whitelist"):
        verifier.validate_postopen_git_dirt(root, authorization_path)

    extra = _write_bytes(root, "unexpected.txt")
    with pytest.raises(ValueError, match="extra Git dirt"):
        verifier.validate_postopen_git_dirt(root, authorization_path)
    extra.unlink()
    tracked.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tracked worktree"):
        verifier.validate_postopen_git_dirt(root, authorization_path)
    subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "forbidden compute edit"],
        cwd=root, env=environment, check=True,
    )
    with pytest.raises(ValueError, match="outside the documentation whitelist"):
        verifier.validate_postopen_git_dirt(root, authorization_path)
    assert namespace_artifact.is_file()


def test_real_preopen_materialization_keeps_frozen_panel_source_metadata_closed(tmp_path):
    verifier = _load_script(VERIFY_SCRIPT, "thermoroute_verify_real_panel_closure_test")
    stage = tmp_path / "stage"
    stage.mkdir()
    verifier.materialize_release_profile(ROOT, stage, verifier.PREOPEN_PROFILE)
    command = [
        sys.executable,
        "-c",
        (
            "from thermoroute.evidence import FrozenPanelSpec; "
            "e=FrozenPanelSpec.load(r'"
            + str(stage / "data_usgs" / "frozen_panel_v1.json")
            + "').verify(); assert e['station_count']==120"
        ),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(
        command,
        cwd=stage,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

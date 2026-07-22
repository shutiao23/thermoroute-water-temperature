"""Fresh-process full-development replay for the frozen Route-A model suite.

The training stages already compare their just-saved bundles with development
predictions.  This module deliberately repeats that proof in a later isolated
process: it reconstructs both temporal and pooled-external preprocessing from
the canonical panel, reloads every model member/head, and compares every
development prediction value on its exact forecast key.  The resulting receipt
contains no confirmation-period input or outcome.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence, cast

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import datasets as DS
from . import features as F
from .checkpoint import load_inference_bundle
from .frozen_inference import sequence_factory_from_metadata
from .model_suite import (
    BUILTIN_MODELS,
    ModelSuiteError,
    fit_pooled_imputer,
    load_lightgbm_bundle,
    serialise_preprocessing,
    validate_development_prediction_binding,
    validate_model_suite_document,
    verify_lightgbm_prediction_parity,
    verify_sequence_prediction_parity,
)
from .registry import restrict_tabular_to_window_registry
from .repro import (
    canonical_json,
    numerical_runtime_contract,
    sha256_file,
    sha256_json,
    source_tree_hash,
)


DEVELOPMENT_REPLAY_FORMAT = "thermoroute.route-a-development-replay.v1"
DEVELOPMENT_REPLAY_EXECUTION_FORMAT = (
    "thermoroute.route-a-development-replay-execution.v1"
)
DEVELOPMENT_REPLAY_IO_GUARD_FORMAT = (
    "thermoroute.route-a-development-replay-io-guard.v2"
)
DEVELOPMENT_REPLAY_CONFIRMATION_READ_POLICY_FORMAT = (
    "thermoroute.route-a-development-replay-confirmation-read-policy.v1"
)
REPLAY_ENTRYPOINT = "scripts/27_verify_development_replay.py"
REPLAY_FORBIDDEN_CONFIRMATION_NAMESPACE_STEMS = (
    "data_usgs/confirmatory",
    "data_usgs/raw_snapshots/confirmatory",
    "data_usgs/raw_snapshots/openmeteo-gfs-previous-runs-v1",
    "outputs/confirmatory",
)
REPLAY_ALLOWED_CONFIRMATION_READ_PATHS = (
    "data_usgs/confirmatory_model_suite_v1.json",
)
LEARNED_TEMPORAL = (
    "LightGBM", "LSTM", "ThermoRoute", "DampedPriorOnly",
    "TR-noDynamicPrior", "TR-fixedKappa", "TR-noRouter", "TR-noMoE",
    "TR-noTCN", "TR-unbounded",
)
LEARNED_EXTERNAL = ("LightGBM", "LSTM", "ThermoRoute")


def _confirmation_read_policy_attestation() -> dict[str, Any]:
    """Return the exact allow-before-deny confirmation-read policy."""
    return {
        "format": DEVELOPMENT_REPLAY_CONFIRMATION_READ_POLICY_FORMAT,
        "mode": "DENY_NAMESPACE_STEMS_EXCEPT_EXACT_ALLOWLIST",
        "path_representation": "RESOLVED_REPOSITORY_RELATIVE_POSIX",
        "deny_match": "STRING_STARTSWITH",
        "denied_namespace_stems": list(
            REPLAY_FORBIDDEN_CONFIRMATION_NAMESPACE_STEMS
        ),
        "allowed_exact_paths": list(REPLAY_ALLOWED_CONFIRMATION_READ_PATHS),
    }


def _is_forbidden_confirmation_read(relative: str) -> bool:
    """Apply exact-suite allowlisting before namespace-stem denial."""
    if relative in REPLAY_ALLOWED_CONFIRMATION_READ_PATHS:
        return False
    return any(
        relative.startswith(stem)
        for stem in REPLAY_FORBIDDEN_CONFIRMATION_NAMESPACE_STEMS
    )


class DevelopmentReplayIOGuard:
    """Audit one replay and fail closed on label, network or mutation access.

    Python audit hooks cannot be removed, so the hook remains installed but is
    active only inside the context manager.  This is an honest-owner execution
    guard, not a sandbox against an owner who can replace CPython or the kernel.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.active = False
        self.repo_reads: set[str] = set()
        self._installed = False

    def _relative_repo_path(self, value: object) -> str | None:
        if isinstance(value, int):
            return None
        if not isinstance(value, (str, bytes, os.PathLike)):
            return None
        try:
            raw = os.fsdecode(value)
        except (TypeError, ValueError):
            return None
        if not raw or raw.startswith("<"):
            return None
        try:
            path = Path(raw)
            if not path.is_absolute():
                path = Path.cwd() / path
            resolved = path.resolve(strict=False)
        except (OSError, RuntimeError):
            return None
        if resolved == self.root:
            return "."
        if self.root not in resolved.parents:
            return None
        return resolved.relative_to(self.root).as_posix()

    @staticmethod
    def _write_open(mode: object, flags: object) -> bool:
        if isinstance(mode, str) and any(token in mode for token in "wax+"):
            return True
        if isinstance(flags, int):
            access = flags & os.O_ACCMODE
            mutation = flags & (
                os.O_APPEND | os.O_CREAT | os.O_TRUNC
            )
            return access != os.O_RDONLY or bool(mutation)
        return False

    def _audit(self, event: str, args: tuple[object, ...]) -> None:
        if not self.active:
            return
        if event == "open" and args:
            relative = self._relative_repo_path(args[0])
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 else None
            if relative is None:
                return
            if self._write_open(mode, flags):
                raise PermissionError(
                    f"development replay may not write repository path: {relative}"
                )
            if _is_forbidden_confirmation_read(relative):
                raise PermissionError(
                    f"development replay may not read confirmation path: {relative}"
                )
            self.repo_reads.add(relative)
            return
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise PermissionError("development replay network access is prohibited")
        if (
            event == "subprocess.Popen"
            or event == "os.system"
            or event.startswith("os.exec")
            or event.startswith("os.spawn")
        ):
            raise PermissionError("development replay child processes are prohibited")

    def __enter__(self) -> DevelopmentReplayIOGuard:
        if not self._installed:
            sys.addaudithook(self._audit)
            self._installed = True
        self.active = True
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.active = False

    def attestation(self) -> dict[str, Any]:
        paths = sorted(self.repo_reads)
        return {
            "format": DEVELOPMENT_REPLAY_IO_GUARD_FORMAT,
            "network_access_allowed": False,
            "subprocess_allowed": False,
            "repository_writes_allowed": False,
            "confirmation_read_policy": _confirmation_read_policy_attestation(),
            "repo_read_path_count": len(paths),
            "repo_read_paths_sha256": sha256_json(paths),
            "repo_read_paths": paths,
            "violations": [],
        }


def _inside(root: Path, relative: object, *, directory: bool = False) -> Path:
    raw = Path(str(relative))
    if raw.is_absolute():
        raise ModelSuiteError("development replay path must be repository-relative")
    path = (root / raw).resolve()
    if path != root and root not in path.parents:
        raise ModelSuiteError("development replay path escapes repository")
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        raise ModelSuiteError(f"development replay artifact is missing: {raw}")
    return path


def _binding(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.resolve().relative_to(root).as_posix(),
        "sha256": sha256_file(path),
    }


def _relative_binding_path(root: Path, value: str | Path, *, label: str) -> Path:
    path = Path(value).resolve()
    if root not in path.parents or not path.is_file():
        raise ModelSuiteError(f"development replay {label} escapes or is absent")
    return path


def _execution_identity(
    *,
    root: Path,
    suite_path: Path,
    receipt_path: Path,
    entrypoint_path: Path,
) -> dict[str, Any]:
    entrypoint = _relative_binding_path(
        root, entrypoint_path, label="entrypoint"
    )
    suite = _relative_binding_path(root, suite_path, label="suite")
    receipt_relative = receipt_path.resolve()
    if root not in receipt_relative.parents:
        raise ModelSuiteError("development replay receipt escapes repository")
    executable = Path(sys.executable).resolve()
    if not executable.is_file():
        raise ModelSuiteError("development replay Python executable is absent")
    return {
        "format": DEVELOPMENT_REPLAY_EXECUTION_FORMAT,
        "entrypoint": _binding(root, entrypoint),
        "interpreter": {
            "invoked_path": sys.executable,
            "realpath": str(executable),
            "sha256": sha256_file(executable),
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
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
                "pycache_prefix=<fresh-temporary-directory>",
                _binding(root, entrypoint)["path"],
                "--_isolated-worker",
                "--suite", suite.relative_to(root).as_posix(),
                "--receipt", receipt_relative.relative_to(root).as_posix(),
            ],
            "fresh_check": [
                "<bound-python>", "-I", "-X",
                "pycache_prefix=<different-fresh-temporary-directory>",
                _binding(root, entrypoint)["path"],
                "--_isolated-worker",
                "--suite", suite.relative_to(root).as_posix(),
                "--receipt", receipt_relative.relative_to(root).as_posix(),
                "--check",
            ],
        },
        "formal_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
                "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED",
            )
        },
        "python_hash_seed_interpreter_effect": (
            "environment declaration present but ignored by CPython -I; replay "
            "code must sort identity-bearing collections"
        ),
    }


def _validate_formal_pycache_prefix(
    root: str | Path, prefix: str | Path | None
) -> Path:
    """Require a dedicated cache outside the repository for formal replay."""
    root = Path(root).resolve()
    if prefix is None:
        raise ModelSuiteError("development replay requires a fresh pycache prefix")
    pycache = Path(prefix).resolve()
    if pycache == root or root in pycache.parents or not pycache.is_dir():
        raise ModelSuiteError(
            "development replay pycache prefix is absent or repository-local"
        )
    return pycache


def _assert_formal_invocation(root: Path, entrypoint_path: Path) -> None:
    if not sys.flags.isolated:
        raise ModelSuiteError("development replay requires CPython isolated mode")
    expected_flags = {
        "isolated": 1,
        "ignore_environment": 1,
        "no_user_site": 1,
        "safe_path": True,
        "dont_write_bytecode": 0,
    }
    actual_flags = {
        "isolated": int(sys.flags.isolated),
        "ignore_environment": int(sys.flags.ignore_environment),
        "no_user_site": int(sys.flags.no_user_site),
        "safe_path": bool(sys.flags.safe_path),
        "dont_write_bytecode": int(sys.flags.dont_write_bytecode),
    }
    if actual_flags != expected_flags:
        raise ModelSuiteError("development replay isolated flags changed")
    _validate_formal_pycache_prefix(root, sys.pycache_prefix)
    if Path(sys.argv[0]).resolve() != entrypoint_path.resolve():
        raise ModelSuiteError("development replay executed through another entrypoint")


def _load_suite(root: Path, suite_path: Path) -> dict[str, Any]:
    try:
        value = json.loads(suite_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ModelSuiteError("development replay cannot read model suite") from exc
    if not isinstance(value, dict):
        raise ModelSuiteError("development replay model suite is not an object")
    # This comparison intentionally precedes suite validation, preprocessing,
    # artifact loading, and every model execution.  A replay under another
    # numerical stack is not allowed to touch learned artifacts and merely fail
    # after producing predictions.
    current_runtime_sha256 = sha256_json(numerical_runtime_contract())
    if value.get("numerical_runtime_sha256") != current_runtime_sha256:
        raise ModelSuiteError(
            "development replay runtime differs from frozen model suite"
        )
    if value.get("training_device") != "cpu":
        raise ModelSuiteError("development replay suite is not CPU-trained")
    validate_model_suite_document(value, root=root)
    return value


def _selected_predictions(
    root: Path,
    metadata: Mapping[str, Any],
    *,
    model_id: str,
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    binding = metadata.get("development_prediction")
    validate_development_prediction_binding(root, binding, label=model_id)
    assert isinstance(binding, Mapping)
    artifact = binding["artifact"]
    selection = binding["selection"]
    if not isinstance(artifact, Mapping) or not isinstance(selection, Mapping):
        raise ModelSuiteError(f"{model_id} development binding is malformed")
    if str(selection.get("model")) != model_id:
        raise ModelSuiteError(f"{model_id} development binding selects another model")
    path = _inside(root, artifact.get("path"))
    frame = pd.read_parquet(path)
    seeds = tuple(int(value) for value in selection.get("seeds", ()))
    selected = frame[
        frame.model.astype(str).eq(model_id)
        & frame.seed.astype(int).isin(seeds)
    ].copy()
    if len(selected) != int(binding["rows"]):
        raise ModelSuiteError(f"{model_id} replay selection row count changed")
    return selected, binding


def _member_seeds(
    member_names: Sequence[str], selection: Mapping[str, Any], *, model_id: str
) -> dict[str, int]:
    members = tuple(str(value) for value in member_names)
    seeds = tuple(int(value) for value in selection.get("seeds", ()))
    parsed: dict[str, int] = {}
    for member in members:
        if member.startswith("seed") and member[4:].isdigit():
            parsed[member] = int(member[4:])
    if len(parsed) == len(members) and set(parsed.values()) == set(seeds):
        return parsed
    if len(members) == len(seeds) == 1:
        return {members[0]: seeds[0]}
    raise ModelSuiteError(
        f"{model_id} bundle members cannot be matched to development seeds"
    )


def build_lightgbm_development_design(
    panel_imputed: pd.DataFrame,
    climatology: Any,
    windows: Any,
    *,
    station_order: Sequence[str],
    raw_feature_order: Sequence[str],
    station_agnostic: bool,
    expected_design_order: Sequence[str],
) -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    """Recreate the exact Stage-9 calibration/test LightGBM design."""
    original_stations = tuple(C.STATIONS)
    if tuple(station_order) != original_stations:
        raise ModelSuiteError("development replay station order differs from panel")
    expected = tuple(str(value) for value in expected_design_order)
    designs: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for horizon in tuple(int(value) for value in windows.horizons):
        table = F.attach_split(F.build_tabular(
            panel_imputed,
            horizon,
            tuple(str(value) for value in raw_feature_order),
            climatology,
            drop_feature_nans=False,
            require_observed_target=True,
            include_missingness=True,
        ))
        table = restrict_tabular_to_window_registry(
            table, windows, tuple(station_order), horizon
        )
        columns = F.feature_columns(table)
        for column in columns:
            table[column] = pd.to_numeric(
                table[column], errors="coerce"
            ).fillna(0.0)
        if not station_agnostic:
            table["station_code"] = pd.Categorical(
                table.site_id.astype(str), categories=list(station_order)
            )
            columns = [*columns, "station_code"]
        if tuple(columns) != expected:
            raise ModelSuiteError(
                f"LightGBM h{horizon} replay design order changed"
            )
        evaluation = table[table.split.isin(["calib", "test"])].copy()
        registry = evaluation[
            ["site_id", "split", "issue_date", "target_date", "y"]
        ].reset_index(drop=True)
        designs[horizon] = (
            registry,
            evaluation[columns].reset_index(drop=True),
        )
    return designs


def _prepare_temporal(
    root: Path, suite: Mapping[str, Any]
) -> tuple[
    pd.DataFrame,
    D.SplitMasks,
    tuple[str, ...],
    pd.DataFrame,
    F.HarmonicClimatology,
    DS.WindowedData,
    dict[str, Any],
]:
    contract = suite["development_contract"]
    panel_path = _inside(root, contract["panel"]["path"])
    prepared = D.prepare_dataset_from_panel(str(panel_path))
    panel = cast(pd.DataFrame, prepared["panel_raw"])
    panel_imputed = cast(pd.DataFrame, prepared["panel"])
    masks = cast(D.SplitMasks, prepared["masks"])
    station_values = cast(Sequence[object], prepared["stations"])
    stations = tuple(str(value) for value in station_values)
    imputer = cast(D.Imputer, prepared["imputer"])
    climatology = F.HarmonicClimatology.fit(panel, masks.train)
    features = tuple(str(value) for value in suite["actual_feature_order"])
    windows = DS.build_windows(
        panel_imputed,
        masks,
        climatology,
        variables=features,
        require_observed_target=True,
    )
    preprocessing = serialise_preprocessing(windows, climatology, imputer)
    return panel, masks, stations, panel_imputed, climatology, windows, preprocessing


def _prepare_external(
    panel: pd.DataFrame,
    masks: Any,
    stations: tuple[str, ...],
    features: tuple[str, ...],
):
    imputer = fit_pooled_imputer(panel, masks.train, fit_stations=stations)
    panel_imputed = imputer.transform(panel)
    climatology = F.HarmonicClimatology.fit(
        panel, masks.train, fit_stations=stations, pooled=True
    )
    windows = DS.build_windows(
        panel_imputed,
        masks,
        climatology,
        variables=features,
        require_observed_target=True,
        scaler_fit_stations=stations,
        pooled_scaler=True,
        damped_fit_stations=stations,
        pooled_damped=True,
    )
    preprocessing = serialise_preprocessing(windows, climatology, imputer)
    return panel_imputed, climatology, windows, preprocessing


def run_development_replay(
    *, root: str | Path, suite_path: str | Path
) -> dict[str, Any]:
    """Execute the full learned-model replay and return a deterministic receipt."""
    root = Path(root).resolve()
    suite_path = Path(suite_path).resolve()
    if root not in suite_path.parents:
        raise ModelSuiteError("development replay suite escapes repository")
    suite = _load_suite(root, suite_path)
    frozen_source_sha256 = str(
        suite["development_contract"]["source_sha256"]
    )
    if frozen_source_sha256 != source_tree_hash(root):
        raise ModelSuiteError(
            "development replay source differs from frozen model suite"
        )
    features = tuple(str(value) for value in suite["actual_feature_order"])
    (
        panel,
        masks,
        stations,
        temporal_panel,
        temporal_climatology,
        temporal_windows,
        temporal_preprocessing,
    ) = _prepare_temporal(root, suite)
    (
        external_panel,
        external_climatology,
        external_windows,
        external_preprocessing,
    ) = _prepare_external(panel, masks, stations, features)

    results: list[dict[str, Any]] = []
    for cohort, expected_models, panel_imputed, climatology, windows, preprocessing in (
        (
            "temporal", LEARNED_TEMPORAL, temporal_panel,
            temporal_climatology, temporal_windows, temporal_preprocessing,
        ),
        (
            "external", LEARNED_EXTERNAL, external_panel,
            external_climatology, external_windows, external_preprocessing,
        ),
    ):
        entries = suite["cohorts"][cohort]["models"]
        by_id = {str(entry["model_id"]): entry for entry in entries}
        learned = tuple(model for model in by_id if model not in BUILTIN_MODELS)
        if set(learned) != set(expected_models):
            raise ModelSuiteError(f"{cohort} replay learned-model registry changed")
        for model_id in expected_models:
            entry = by_id[model_id]
            executor = str(entry["executor"])
            artifact = entry["artifact"]
            if executor == "lightgbm_bundle":
                manifest_path = _inside(root, artifact["path"])
                models, metadata = load_lightgbm_bundle(manifest_path)
                expected, binding = _selected_predictions(
                    root, metadata, model_id=model_id
                )
                if metadata.get("preprocessing") != preprocessing:
                    raise ModelSuiteError(
                        f"{cohort}/{model_id} preprocessing replay changed"
                    )
                design = build_lightgbm_development_design(
                    panel_imputed,
                    climatology,
                    windows,
                    station_order=stations,
                    raw_feature_order=features,
                    station_agnostic=cohort == "external",
                    expected_design_order=metadata["design_feature_order"],
                )
                member_seeds = _member_seeds(
                    tuple(models), binding["selection"], model_id=model_id
                )
                difference = verify_lightgbm_prediction_parity(
                    manifest_path,
                    evaluation_design=design,
                    expected=expected,
                    member_seeds=member_seeds,
                    atol=float(binding["atol"]),
                )
            else:
                directory = _inside(root, artifact["path"], directory=True)
                weights, metadata = load_inference_bundle(
                    directory, expected_member_count=int(entry["member_count"])
                )
                expected, binding = _selected_predictions(
                    root, metadata, model_id=model_id
                )
                if metadata.get("preprocessing") != preprocessing:
                    raise ModelSuiteError(
                        f"{cohort}/{model_id} preprocessing replay changed"
                    )
                member_seeds = _member_seeds(
                    tuple(weights), binding["selection"], model_id=model_id
                )
                split_order = tuple(
                    split for split in ("val", "calib", "test")
                    if split in set(expected.split.astype(str))
                )
                difference = verify_sequence_prediction_parity(
                    directory,
                    wd=windows,
                    expected=expected,
                    model_factory=lambda _member, value:
                        sequence_factory_from_metadata(value),
                    member_seeds=member_seeds,
                    atol=float(binding["atol"]),
                    splits=split_order,
                )
            results.append({
                "cohort": cohort,
                "model": model_id,
                "executor": executor,
                "members": int(entry["member_count"]),
                "rows": int(len(expected)),
                "atol": float(binding["atol"]),
                "max_abs_difference": float(difference),
                "status": "PASS",
            })

    runtime = numerical_runtime_contract()
    document: dict[str, Any] = {
        "format": DEVELOPMENT_REPLAY_FORMAT,
        "status": "PASS_FULL_DEVELOPMENT_REPLAY_NO_CONFIRMATION_DATA",
        "isolated_process_required": True,
        "suite": _binding(root, suite_path),
        "source_tree_sha256": frozen_source_sha256,
        "runtime_sha256": sha256_json(runtime),
        "suite_numerical_runtime_sha256": suite["numerical_runtime_sha256"],
        "development_contract_sha256": sha256_json(
            suite["development_contract"]
        ),
        "replayed_splits": ["val", "calib", "test_2019_2020_development"],
        "confirmation_period_read": False,
        "builtins_validated_by_suite_contract": sorted(BUILTIN_MODELS),
        "models": results,
    }
    document["receipt_self_sha256"] = sha256_json(document)
    return document


def run_guarded_development_replay(
    *,
    root: str | Path,
    suite_path: str | Path,
    receipt_path: str | Path,
    entrypoint_path: str | Path,
) -> dict[str, Any]:
    """Run the full replay under the fixed isolated, no-I/O-side-effect policy."""
    root = Path(root).resolve()
    suite_path = Path(suite_path).resolve()
    receipt_path = Path(receipt_path).resolve()
    entrypoint_path = Path(entrypoint_path).resolve()
    _assert_formal_invocation(root, entrypoint_path)
    identity = _execution_identity(
        root=root,
        suite_path=suite_path,
        receipt_path=receipt_path,
        entrypoint_path=entrypoint_path,
    )
    guard = DevelopmentReplayIOGuard(root)
    with guard:
        document = run_development_replay(root=root, suite_path=suite_path)
    stable = {
        key: value for key, value in document.items()
        if key != "receipt_self_sha256"
    }
    stable["execution_attestation"] = {
        **identity,
        "io_guard": guard.attestation(),
        "security_boundary": (
            "fresh-process honest-owner replay guard; not protection against "
            "replacement of CPython, the operating system, or the repository owner"
        ),
    }
    return {**stable, "receipt_self_sha256": sha256_json(stable)}


def fresh_verify_development_replay_receipt(
    receipt_path: str | Path,
    *,
    root: str | Path,
    suite_path: str | Path,
    entrypoint_path: str | Path,
) -> dict[str, Any]:
    """Recompute all predictions and require exact receipt equivalence."""
    receipt_path = Path(receipt_path).resolve()
    existing = validate_development_replay_receipt(
        receipt_path, root=root, suite_path=suite_path
    )
    replayed = run_guarded_development_replay(
        root=root,
        suite_path=suite_path,
        receipt_path=receipt_path,
        entrypoint_path=entrypoint_path,
    )
    if replayed != existing:
        raise ModelSuiteError(
            "fresh full development replay differs from the frozen receipt"
        )
    return existing


def validate_development_replay_receipt(
    receipt_path: str | Path,
    *,
    root: str | Path,
    suite_path: str | Path,
) -> dict[str, Any]:
    """Validate a replay receipt against the current source and frozen suite."""
    root = Path(root).resolve()
    receipt_path = Path(receipt_path).resolve()
    suite_path = Path(suite_path).resolve()
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ModelSuiteError("development replay receipt is absent or invalid") from exc
    if not isinstance(receipt, dict) or receipt.get("format") != DEVELOPMENT_REPLAY_FORMAT:
        raise ModelSuiteError("unsupported development replay receipt")
    self_hashed = dict(receipt)
    self_digest = self_hashed.pop("receipt_self_sha256", None)
    if self_digest != sha256_json(self_hashed):
        raise ModelSuiteError("development replay receipt self hash changed")
    try:
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
        development = suite["development_contract"]
        frozen_source_sha256 = str(development["source_sha256"])
        suite_runtime_sha256 = str(suite["numerical_runtime_sha256"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ModelSuiteError(
            "development replay suite lacks a frozen source-tree SHA-256"
        ) from exc
    current_source_sha256 = source_tree_hash(root)
    if (
        len(frozen_source_sha256) != 64
        or frozen_source_sha256 != current_source_sha256
        or receipt.get("source_tree_sha256") != frozen_source_sha256
    ):
        raise ModelSuiteError(
            "development replay source differs from frozen model suite/current source"
        )
    current_runtime_sha256 = sha256_json(numerical_runtime_contract())
    if len(suite_runtime_sha256) != 64 or suite_runtime_sha256 != current_runtime_sha256:
        raise ModelSuiteError(
            "development replay runtime differs from frozen model suite"
        )
    expected_scalars = {
        "status": "PASS_FULL_DEVELOPMENT_REPLAY_NO_CONFIRMATION_DATA",
        "isolated_process_required": True,
        "source_tree_sha256": frozen_source_sha256,
        "runtime_sha256": suite_runtime_sha256,
        "suite_numerical_runtime_sha256": suite_runtime_sha256,
        "confirmation_period_read": False,
    }
    if any(receipt.get(key) != value for key, value in expected_scalars.items()):
        raise ModelSuiteError("development replay receipt no longer matches runtime/source")
    if receipt.get("suite") != _binding(root, suite_path):
        raise ModelSuiteError("development replay receipt binds another model suite")
    execution = receipt.get("execution_attestation")
    if not isinstance(execution, Mapping):
        raise ModelSuiteError("development replay lacks execution attestation")
    entrypoint_path = (root / REPLAY_ENTRYPOINT).resolve()
    expected_execution = _execution_identity(
        root=root,
        suite_path=suite_path,
        receipt_path=receipt_path,
        entrypoint_path=entrypoint_path,
    )
    for key, value in expected_execution.items():
        if execution.get(key) != value:
            raise ModelSuiteError(
                f"development replay execution identity changed: {key}"
            )
    io_guard = execution.get("io_guard")
    expected_io_guard_keys = {
        "format",
        "network_access_allowed",
        "subprocess_allowed",
        "repository_writes_allowed",
        "confirmation_read_policy",
        "repo_read_path_count",
        "repo_read_paths_sha256",
        "repo_read_paths",
        "violations",
    }
    if (
        not isinstance(io_guard, Mapping)
        or set(io_guard) != expected_io_guard_keys
        or io_guard.get("format") != DEVELOPMENT_REPLAY_IO_GUARD_FORMAT
        or io_guard.get("network_access_allowed") is not False
        or io_guard.get("subprocess_allowed") is not False
        or io_guard.get("repository_writes_allowed") is not False
        or io_guard.get("confirmation_read_policy")
        != _confirmation_read_policy_attestation()
        or io_guard.get("violations") != []
    ):
        raise ModelSuiteError("development replay I/O guard attestation changed")
    read_paths = io_guard.get("repo_read_paths")
    if (
        not isinstance(read_paths, list)
        or not all(isinstance(value, str) for value in read_paths)
        or read_paths != sorted(set(read_paths))
        or io_guard.get("repo_read_path_count") != len(read_paths)
        or io_guard.get("repo_read_paths_sha256") != sha256_json(read_paths)
        or any(_is_forbidden_confirmation_read(value) for value in read_paths)
    ):
        raise ModelSuiteError("development replay read-path evidence is malformed")
    if not isinstance(execution.get("security_boundary"), str):
        raise ModelSuiteError("development replay security boundary is absent")
    rows = receipt.get("models")
    if not isinstance(rows, list):
        raise ModelSuiteError("development replay receipt lacks model rows")
    actual = {(str(row.get("cohort")), str(row.get("model"))) for row in rows
              if isinstance(row, Mapping)}
    expected = {
        *(("temporal", model) for model in LEARNED_TEMPORAL),
        *(("external", model) for model in LEARNED_EXTERNAL),
    }
    if len(rows) != len(expected) or actual != expected:
        raise ModelSuiteError("development replay receipt model registry changed")
    for row in rows:
        if (
            row.get("status") != "PASS"
            or int(row.get("rows", 0)) < 1
            or int(row.get("members", 0)) < 1
            or not np.isfinite(float(row.get("max_abs_difference", np.nan)))
            or float(row["max_abs_difference"]) > float(row.get("atol", -1.0))
        ):
            raise ModelSuiteError("development replay receipt contains a failed row")
    return receipt


def write_replay_receipt(path: str | Path, document: Mapping[str, Any]) -> Path:
    """Create the deterministic receipt once; never replace different bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(document) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise FileExistsError(
                f"refusing to replace different development replay receipt: {path}"
            )
        return path
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path

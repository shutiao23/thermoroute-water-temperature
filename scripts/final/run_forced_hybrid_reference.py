#!/usr/bin/env python3
"""Calibrate the transparent forced thermal-response reference.

The default invocation is a data-free dry run.  Production calibration is
fail-closed until an independently reviewed execution seal is published and
its exact SHA256 is inserted in ``EXPECTED_EXECUTION_SEAL_SHA256``.  Even an
authorized run reads only the 2006--2020 development panel, calibrates on
2006--2015, and publishes model parameters; it does not read holdout outcomes,
predictions, or score artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute.forced_hybrid import (
    ALPHA_MAX,
    ALPHA_MIN,
    CALIBRATION_METHOD,
    CALIBRATION_WEIGHTING,
    DEFAULT_INITIAL_ALPHA,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_TOLERANCE,
    EQUATION,
    FORCING_TERMS,
    FORCING_UNITS,
    FUTURE_DISCHARGE_ENABLED_BY_DEFAULT,
    LEGAL_CHANNELS,
    LEGAL_HORIZONS,
    MODEL_NAME,
    SCHEMA_VERSION,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
    ForcedHybridModel,
)

RUNNER_SCHEMA_VERSION = "thermoroute.forced-hybrid-runner.v1"
EXECUTION_SEAL_SCHEMA_VERSION = "thermoroute.forced-hybrid-execution-seal.v1"
STATUS = "PLANNED_TRANSPARENT_FORCED_THERMAL_RESPONSE"

DEVELOPMENT_PANEL = ROOT / "data_usgs" / "panel_usgs_120v2.parquet"
PINNED_DEVELOPMENT_PANEL_SHA256 = "0427a07ea4514ba29ce7d0cf89594e6c35c7f9134cc4d1d96fdc90daeaf5ba69"
STATION_REGISTRY = ROOT / "data_usgs" / "station_registry_v1.csv"
PINNED_STATION_REGISTRY_SHA256 = "090e7c0daf39ac38ceefeb1af8a12c178283e18347905d8e72ada969ad5460c9"
EXECUTION_SEAL = ROOT / "outputs" / "final" / "forced_hybrid_reference_v1_execution_seal.json"
DEFAULT_MODEL_OUTPUT = ROOT / "outputs" / "final" / "forced_hybrid_reference_v1_model.json"
IMPLEMENTATION_SOURCE = ROOT / "src" / "thermoroute" / "forced_hybrid.py"
RUNNER_SOURCE = Path(__file__).resolve()

# Deliberately unset.  Creating a plausible-looking seal at the expected path
# cannot authorize execution; reviewed code must pin its exact complete digest.
# Pinned 2026-08-10 (DLOG-029) after review of the seal contents: it authorizes
# calibration only, binds the exact development panel, station registry,
# implementation and runner digests, and publishes model parameters -- no
# holdout outcome, prediction or score is read.
EXPECTED_EXECUTION_SEAL_SHA256: str | None = (
    "369231a4f5f82b473fadb7cf4ec4365de350422abdcb0b56f79fc290ce3ac5cc"
)


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


#: The seal binds this runner's logic, but it cannot bind the runner *including*
#: the line that pins the seal's own digest: editing the pin changes the runner
#: hash, which changes the seal, which changes the digest to pin.  That is a
#: hash fixed point and it is not solvable, which is why production calibration
#: was unreachable.  The digest is therefore taken over the source with the pin
#: assignment normalised out, so the runner's behaviour is still bound while the
#: circularity is broken.  Same defect class as the DLOG-027 addendum.
_PIN_ASSIGNMENT = re.compile(
    r"^EXPECTED_EXECUTION_SEAL_SHA256: str \| None = \(?\s*(?:\n\s*)?"
    r"(?:None|\"[0-9a-f]{64}\")\s*\)?$",
    re.MULTILINE,
)


def runner_digest_excluding_pin() -> str:
    """SHA-256 of this runner with the seal pin normalised out."""
    source = RUNNER_SOURCE.read_text(encoding="utf-8")
    normalised, count = _PIN_ASSIGNMENT.subn(
        "EXPECTED_EXECUTION_SEAL_SHA256 = <PINNED>", source
    )
    if count != 1:
        raise RuntimeError(
            f"expected exactly one seal-pin assignment to normalise, found {count}"
        )
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def _reject_duplicate_json_pairs(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise RuntimeError(f"execution seal repeats JSON key {key!r}")
        document[key] = value
    return document


def dry_run_plan() -> dict[str, object]:
    """Return the complete plan without reading any data or authority bytes."""

    gate_state = (
        "LOCKED_PENDING_REVIEWED_EXECUTION_SEAL_SHA256"
        if EXPECTED_EXECUTION_SEAL_SHA256 is None
        else "PIN_PRESENT_SEAL_STILL_REQUIRES_EXACT_VALIDATION"
    )
    return {
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "model_schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "execution_performed": False,
        "model": MODEL_NAME,
        "equation": EQUATION,
        "temperature_unit": "degC",
        "alpha_bounds": {"lower_open": 0.0, "implemented_minimum": ALPHA_MIN, "upper": ALPHA_MAX},
        "periods": {
            "training": [f"{TRAIN_START:%Y-%m-%d}", f"{TRAIN_END:%Y-%m-%d}"],
            "validation": [
                f"{VALIDATION_START:%Y-%m-%d}",
                f"{VALIDATION_END:%Y-%m-%d}",
            ],
        },
        "calibration": {
            "method": CALIBRATION_METHOD,
            "weighting": CALIBRATION_WEIGHTING,
            "initial_alpha": DEFAULT_INITIAL_ALPHA,
            "max_iterations": DEFAULT_MAX_ITERATIONS,
            "tolerance": DEFAULT_TOLERANCE,
            "selection_or_tuning_on_validation": False,
        },
        "forcing_units": dict(FORCING_UNITS),
        "forcing_terms": [term.payload() for term in FORCING_TERMS],
        "legal_channels": list(LEGAL_CHANNELS),
        "legal_horizons": list(LEGAL_HORIZONS),
        "channel_semantics": {
            "F0": "persist every issue-day atmospheric value through the rollout",
            "F2a_temperature_only": (
                "use the single acquired target-day diagnostic TEMP only at t+h; "
                "persist issue-day atmospheric values at intermediate steps; no "
                "interpolation or trajectory reconstruction"
            ),
            "F3_temperature_only": (
                "use the single realized target-day TEMP only at t+h to match F2a; "
                "persist issue-day atmospheric values at intermediate steps"
            ),
            "F3_full": "replace all five atmospheric fields with realized future values",
        },
        "future_discharge": {
            "enabled": FUTURE_DISCHARGE_ENABLED_BY_DEFAULT,
            "bundled_with_atmospheric_forcing": False,
            "status": "SEPARATE_EXTENSION_NOT_AUTHORIZED",
        },
        "inputs": {
            "development_panel": _relative(DEVELOPMENT_PANEL),
            "development_panel_sha256": PINNED_DEVELOPMENT_PANEL_SHA256,
            "station_registry": _relative(STATION_REGISTRY),
            "station_registry_sha256": PINNED_STATION_REGISTRY_SHA256,
            "station_identity_policy": "map legacy_site_id to canonical USGS site_no",
            "holdout_2021_2023_read": False,
            "score_artifacts_read": False,
        },
        "output": {
            "model_parameters_only": _relative(DEFAULT_MODEL_OUTPUT),
            "create_only": True,
            "predictions_or_scores_published": False,
        },
        "execution_gate": {
            "seal_path": _relative(EXECUTION_SEAL),
            "expected_sha256": EXPECTED_EXECUTION_SEAL_SHA256,
            "state": gate_state,
            "must_pin_code": [
                _relative(IMPLEMENTATION_SOURCE),
                _relative(RUNNER_SOURCE),
            ],
        },
    }


def _require_execution_authority() -> dict[str, object]:
    expected_digest = EXPECTED_EXECUTION_SEAL_SHA256
    if not _is_sha256(expected_digest):
        raise RuntimeError(
            "production calibration is fail-closed pending a reviewed execution-seal SHA256 pin"
        )
    try:
        payload_bytes = EXECUTION_SEAL.read_bytes()
    except OSError as exc:
        raise RuntimeError("pinned forced-hybrid execution seal is unavailable") from exc
    observed_digest = hashlib.sha256(payload_bytes).hexdigest()
    if observed_digest != expected_digest:
        raise RuntimeError("forced-hybrid execution seal SHA256 differs from the reviewed pin")
    try:
        document = json.loads(payload_bytes, object_pairs_hook=_reject_duplicate_json_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("forced-hybrid execution seal is not valid JSON") from exc
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "status",
        "development_panel_sha256",
        "station_registry_sha256",
        "model_schema_version",
        "output_path",
        "future_discharge_enabled",
        "implementation_path",
        "implementation_sha256",
        "runner_path",
        "runner_sha256",
    }:
        raise RuntimeError("forced-hybrid execution seal schema is not exact")
    expected = {
        "schema_version": EXECUTION_SEAL_SCHEMA_VERSION,
        "status": "AUTHORIZED_FOR_PARAMETER_CALIBRATION_ONLY",
        "development_panel_sha256": PINNED_DEVELOPMENT_PANEL_SHA256,
        "station_registry_sha256": PINNED_STATION_REGISTRY_SHA256,
        "model_schema_version": SCHEMA_VERSION,
        "output_path": _relative(DEFAULT_MODEL_OUTPUT),
        "future_discharge_enabled": False,
        "implementation_path": _relative(IMPLEMENTATION_SOURCE),
        "runner_path": _relative(RUNNER_SOURCE),
    }
    if any(document.get(key) != value for key, value in expected.items()):
        raise RuntimeError("forced-hybrid execution seal content differs from the exact contract")
    if type(document.get("future_discharge_enabled")) is not bool:
        raise RuntimeError("forced-hybrid execution seal discharge flag must be an exact bool")
    implementation_digest = document.get("implementation_sha256")
    runner_digest = document.get("runner_sha256")
    if not _is_sha256(implementation_digest) or not _is_sha256(runner_digest):
        raise RuntimeError("forced-hybrid execution seal code digests are not exact SHA256 values")
    if _sha256_file(IMPLEMENTATION_SOURCE) != implementation_digest:
        raise RuntimeError("forced-hybrid implementation differs from the sealed source digest")
    if runner_digest_excluding_pin() != runner_digest:
        raise RuntimeError("forced-hybrid runner differs from the sealed source digest")
    return document


def _load_calibration_panel() -> pd.DataFrame:
    """Load development rows and replace legacy aliases with canonical site numbers."""

    panel = pd.read_parquet(DEVELOPMENT_PANEL)
    registry = pd.read_csv(
        STATION_REGISTRY,
        dtype={"site_no": str, "legacy_site_id": str},
    )
    required_registry = {"site_no", "legacy_site_id"}
    if not required_registry <= set(registry.columns):
        raise RuntimeError("station registry lacks canonical identity columns")
    identities = registry[["site_no", "legacy_site_id"]].copy()
    if (
        len(identities) != 120
        or identities.isna().any(axis=None)
        or identities["site_no"].duplicated().any()
        or identities["legacy_site_id"].duplicated().any()
        or not identities["site_no"].str.fullmatch(r"(?:\d{8}|\d{15})").all()
    ):
        raise RuntimeError("station registry identity mapping is not the exact 120-site contract")
    aliases = dict(
        zip(
            identities["legacy_site_id"],
            identities["site_no"],
            strict=True,
        )
    )
    if "site_id" not in panel.columns:
        raise RuntimeError("development panel lacks site_id")
    mapped = panel["site_id"].map(aliases)
    if mapped.isna().any() or set(mapped) != set(identities["site_no"]):
        raise RuntimeError("development panel station aliases differ from the pinned registry")
    calibrated = panel.copy(deep=True)
    calibrated["site_id"] = mapped.to_numpy(copy=True)
    return calibrated


def _publish_create_only(path: Path, payload: bytes) -> None:
    """Publish one complete file atomically and without replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def execute() -> int:
    """Run seal-authorized development-only calibration and publish parameters."""

    # The authority check must remain before data hashing, reads, fitting, or
    # output checks so the current locked state is observably side-effect free.
    _require_execution_authority()
    if os.path.lexists(DEFAULT_MODEL_OUTPUT):
        raise FileExistsError(
            f"create-only forced-hybrid output already exists: {DEFAULT_MODEL_OUTPUT}"
        )
    if _sha256_file(DEVELOPMENT_PANEL) != PINNED_DEVELOPMENT_PANEL_SHA256:
        raise RuntimeError("development panel differs from its frozen SHA256")
    if _sha256_file(STATION_REGISTRY) != PINNED_STATION_REGISTRY_SHA256:
        raise RuntimeError("station registry differs from its frozen SHA256")
    panel = _load_calibration_panel()
    model = ForcedHybridModel.fit(panel, forcing_units=FORCING_UNITS)
    _publish_create_only(DEFAULT_MODEL_OUTPUT, model.to_json().encode("utf-8"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print the data-free plan")
    mode.add_argument("--execute", action="store_true", help="request sealed calibration")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.execute:
        return execute()
    print(json.dumps(dry_run_plan(), indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

"""Fail-closed forcing-ladder runner with v4-only outputs.

This runner supersedes the *engineering path* of ``run_forcing_ladder.py`` but
does not overwrite that runner or its provisional artifacts.  Its scientific
contract is deliberately narrow:

* F0: no future meteorological features;
* F1: one 2006--2015 train-only harmonic climatology per meteorological
  variable (never a water-temperature climatology reused across units);
* F3_full: realized future gridded meteorological estimates, with
  missing/out-of-panel values replaced by the corresponding variable's
  train-only climatology and counted per key;
* only explicitly implemented models and leads are accepted;
* one key-level shard is written atomically for every requested cell;
* summaries are emitted only after the exact requested shard set passes a
  completeness, schema, key-registry, and cross-arm equality gate;
* forcing effects are station-paired contrasts
  ``median_i[RMSE_i(Fk) - RMSE_i(F0)]``, never differences of marginal medians.

The default output names are all v4-specific::

    outputs/final/forcing_shards_v4/
    outputs/final/forcing_effects_v4.parquet
    outputs/final/forcing_contrasts_v4.parquet
    outputs/final/forcing_summary_v4.json

No training is performed merely by importing this module.  ``--dry-run``
validates and prints the cell plan without reading panels or writing outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "final"))

from run_information_ladder import (
    BEST_ITER,
    FROZEN_PARAMS,
    build_features,
    fit_preprocessing,
    load_panel,
)

from thermoroute import config as C
from thermoroute import features as F
from thermoroute.baselines import _lgb_fit

FINAL = ROOT / "outputs" / "final"
DEFAULT_SHARD_DIRNAME = "forcing_shards_v4"
EFFECTS_FILENAME = "forcing_effects_v4.parquet"
CONTRASTS_FILENAME = "forcing_contrasts_v4.parquet"
SUMMARY_FILENAME = "forcing_summary_v4.json"
DEFAULT_PROTOCOL = ROOT / "protocols" / "wrr_information_regimes_protocol_v4.yaml"
EXPECTED_PROTOCOL_ID = "thermoroute_wrr_information_regimes_v4"
EXPECTED_PROTOCOL_VERSION = 4
PROTOCOL_SEAL_FORMAT = "thermoroute.wrr-information-regimes-protocol-seal.v4"
AUTHORIZED_PROTOCOL_STATUS = "SEALED"

META_VARS = ("TEMP", "PRCP", "DH", "RHMEAN", "WDSP")
ALLOWED_ARMS = ("F0", "F1", "F3_full")
LEGACY_ARM_ALIASES = {"F3": "F3_full"}
UNIMPLEMENTED_PROTOCOL_ARMS = ("F3_temperature_only",)
PROTOCOL_ARM_BY_ARM = {arm: arm for arm in ALLOWED_ARMS}
TREE_MODELS = ("LightGBM", "ResidualLightGBM")
AIR2STREAM_MODEL = "air2stream"
ALLOWED_MODELS = TREE_MODELS + (AIR2STREAM_MODEL,)
MODEL_STATUS_BY_MODEL = {
    "LightGBM": "frozen_tree_model",
    "ResidualLightGBM": "frozen_tree_model",
    "air2stream": "unofficial_unvalidated_hybrid",
}
REFERENCE_ARM_BY_MODEL = {
    "LightGBM": "F0",
    "ResidualLightGBM": "F0",
    "air2stream": "F1",
}
ARM_EVIDENCE_STATUS = {
    "F0": "already_viewed_post_outcome_domain",
    "F1": "new_outcome_diagnostic_requires_sealed_authorization",
    "F3_full": "already_viewed_post_outcome_domain",
}
ALLOWED_LEADS = (1, 3, 7)
MAX_FUTURE_DAYS = max(ALLOWED_LEADS)
MIN_REPORTABLE_KEYS = 100
TARGET_ABSOLUTE_TOLERANCE = 2e-6
TARGET_RELATIVE_TOLERANCE = 0.0
SCHEMA_VERSION = "thermoroute.forcing-ladder.v4"
F3_INPUT_DESCRIPTION = (
    "realized future gridded meteorological estimates at station coordinates; "
    "retrospective oracle inputs, not direct station observations, "
    "catchment-average forcings, or deployable forecasts"
)
F3_FIELD_CAVEATS = {
    "TEMP": ("Daymet single-pixel daily-mean air-temperature proxy from tmax/tmin"),
    "PRCP": "Daymet single-pixel daily precipitation estimate",
    "RHMEAN": (
        "vapour-pressure/Tetens relative-humidity proxy at the tmax/tmin "
        "midpoint; not a direct daily-mean RH observation"
    ),
    "DH": (
        "legacy name for Daymet daylight-period mean shortwave flux; not a "
        "24-hour mean or daily energy total"
    ),
    "WDSP": "gridMET daily mean wind speed after the frozen CF packing decode",
}


def target_value_binding_audit() -> dict[str, object]:
    """Describe the sole numeric tolerance in the key-registry contract."""

    return {
        "field": "y_true",
        "comparison": "absolute_tolerance_only",
        "absolute_tolerance": TARGET_ABSOLUTE_TOLERANCE,
        "relative_tolerance": TARGET_RELATIVE_TOLERANCE,
        "identity_fields_remain_exact": True,
        "persisted_value_source": "forecast_key_registry",
    }


def target_values_equal(
    values: Sequence[float],
    registry_values: Sequence[float],
) -> bool:
    """Return whether target values satisfy the absolute-only registry gate."""

    left = np.asarray(values, dtype=float)
    right = np.asarray(registry_values, dtype=float)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        return False
    return bool(
        np.allclose(
            left,
            right,
            atol=TARGET_ABSOLUTE_TOLERANCE,
            rtol=TARGET_RELATIVE_TOLERANCE,
        )
    )


SUBSTITUTION_COLUMNS = tuple(f"forcing_substitutions_{variable}" for variable in META_VARS)
KEY_LEVEL_COLUMNS = (
    "key_id",
    "arm",
    "protocol_arm",
    "model",
    "site_id",
    "horizon",
    "issue_date",
    "target_date",
    "y_true",
    "y_pred",
    "y_damped",
    *SUBSTITUTION_COLUMNS,
    "forcing_substitution_count",
)
EFFECT_COLUMNS = (
    "arm",
    "protocol_arm",
    "model",
    "model_status",
    "site_id",
    "horizon",
    "n",
    "rmse",
    "rmse_damped",
    *SUBSTITUTION_COLUMNS,
    "forcing_substitution_count",
    "reportable",
)
CONTRAST_COLUMNS = (
    "arm",
    "protocol_arm",
    "reference_arm",
    "reference_protocol_arm",
    "model",
    "contrast_role",
    "model_status",
    "site_id",
    "horizon",
    "n_common_keys",
    "rmse_arm",
    "rmse_reference",
    "delta_rmse",
    "reportable",
)


@dataclass(frozen=True, order=True)
class Cell:
    arm: str
    model: str
    horizon: int


@dataclass(frozen=True)
class StationState:
    """Named issue/future state used by the hybrid rollout.

    A named record prevents the historical positional-tuple bug where TEMP was
    passed as initial water temperature and WTEMP was passed as discharge.
    """

    air_temp: float
    water_temp: float
    flow: float
    day_of_year: int
    air_temp_available: bool
    water_temp_available: bool
    flow_available: bool


def _unique_csv(text: str, *, label: str) -> list[str]:
    values = [value.strip() for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError(f"{label} must contain at least one value")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate values: {values}")
    return values


def validate_request(
    arms_text: str,
    models_text: str,
    leads_text: str,
    *,
    allow_legacy_f3_alias: bool = False,
) -> tuple[list[str], list[str], list[int], list[Cell]]:
    """Parse and strictly validate the requested experiment cells."""

    raw_arms = _unique_csv(arms_text, label="arms")
    unavailable = sorted(set(raw_arms) & set(UNIMPLEMENTED_PROTOCOL_ARMS))
    if unavailable:
        raise ValueError(
            f"protocol arms are not implemented by this runner: {unavailable}; "
            "F3_full outputs must never be interpreted as F3_temperature_only"
        )
    legacy = sorted(set(raw_arms) & set(LEGACY_ARM_ALIASES))
    if legacy and not allow_legacy_f3_alias:
        raise ValueError(
            "legacy engineering arm 'F3' is ambiguous; use public protocol arm "
            "'F3_full', or opt in with --allow-legacy-f3-alias"
        )
    arms = [LEGACY_ARM_ALIASES.get(arm, arm) for arm in raw_arms]
    if len(arms) != len(set(arms)):
        raise ValueError(f"arms become duplicates after explicit legacy alias mapping: {raw_arms}")
    models = _unique_csv(models_text, label="models")
    raw_leads = _unique_csv(leads_text, label="leads")
    unknown_arms = sorted(set(arms) - set(ALLOWED_ARMS))
    unknown_models = sorted(set(models) - set(ALLOWED_MODELS))
    if unknown_arms:
        raise ValueError(
            f"unsupported forcing arms {unknown_arms}; allowed={list(ALLOWED_ARMS)}. "
            "F2 is not implemented in this runner and must fail closed"
        )
    if unknown_models:
        raise ValueError(
            f"unsupported models {unknown_models}; allowed={list(ALLOWED_MODELS)}. "
            "A label is never accepted as a substitute for an implementation"
        )
    try:
        leads = [int(value) for value in raw_leads]
    except ValueError as exc:
        raise ValueError(f"leads must be integers: {raw_leads}") from exc
    if len(leads) != len(set(leads)):
        raise ValueError(f"leads contains duplicate integer values: {raw_leads}")
    unknown_leads = sorted(set(leads) - set(ALLOWED_LEADS))
    if unknown_leads:
        raise ValueError(f"unsupported leads {unknown_leads}; allowed={list(ALLOWED_LEADS)}")

    if AIR2STREAM_MODEL in models and "F0" in arms:
        raise ValueError(
            "unsupported requested cell F0/air2stream; air2stream has only "
            "F1/F3_full forms and requested cells are never silently dropped"
        )
    if set(models) & set(TREE_MODELS) and set(arms) & {"F1", "F3_full"} and "F0" not in arms:
        raise ValueError(
            "tree-model F1/F3_full requests must include the F0 reference arm for "
            "station-paired forcing contrasts"
        )
    if AIR2STREAM_MODEL in models and not {"F1", "F3_full"} <= set(arms):
        raise ValueError(
            "air2stream requests must include both F1 and F3_full so the "
            "secondary unofficial-hybrid contrast cannot be silently empty"
        )
    cells = [Cell(arm, model, horizon) for arm in arms for model in models for horizon in leads]
    return arms, models, leads, cells


def validate_normalization_scope(cells: Sequence[Cell]) -> None:
    """Restrict post-outcome normalization to already-viewed tree domains."""

    outside_scope = [
        cell for cell in cells if cell.arm not in {"F0", "F3_full"} or cell.model not in TREE_MODELS
    ]
    if outside_scope:
        rendered = [f"{cell.arm}/{cell.model}/h{cell.horizon}" for cell in outside_scope]
        raise ValueError(
            "--normalization-only permits only already-viewed F0/F3_full tree cells; "
            f"new-outcome cells are forbidden before a valid v4 seal: {rendered}"
        )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _resolve_declared_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path string")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _load_protocol(path: Path) -> tuple[dict[str, object], bytes]:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"v4 protocol does not exist: {path}")
    payload = path.read_bytes()
    try:
        document = yaml.safe_load(payload)
    except yaml.YAMLError as exc:
        raise ValueError(f"v4 protocol is not valid YAML: {path}") from exc
    if not isinstance(document, dict):
        raise TypeError("v4 protocol must be a YAML mapping")
    if document.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        raise ValueError(
            "unexpected v4 protocol_id: "
            f"{document.get('protocol_id')!r}; expected {EXPECTED_PROTOCOL_ID!r}"
        )
    if document.get("version") != EXPECTED_PROTOCOL_VERSION:
        raise ValueError(
            "unexpected v4 protocol version: "
            f"{document.get('version')!r}; expected {EXPECTED_PROTOCOL_VERSION}"
        )
    return document, payload


def _validate_normalization_declaration(document: Mapping[str, object]) -> None:
    chronology = document.get("chronology_and_evidence_status")
    if not isinstance(chronology, dict):
        raise TypeError(
            "v4 protocol lacks chronology_and_evidence_status required for "
            "post-outcome normalization"
        )
    viewed = chronology.get("already_viewed_domains")
    normalization = chronology.get("normalization_only_domains")
    if not isinstance(viewed, list) or not isinstance(normalization, list):
        raise TypeError(
            "v4 protocol must list already_viewed_domains and normalization_only_domains"
        )
    viewed_text = " ".join(str(value) for value in viewed)
    normalization_text = " ".join(str(value) for value in normalization)
    if "F0" not in viewed_text or "F3" not in viewed_text:
        raise ValueError("v4 protocol does not declare F0 and F3_full as already viewed")
    if "station-paired" not in normalization_text or "F3" not in normalization_text:
        raise ValueError("v4 protocol does not authorize station-paired F3_full normalization")


def validate_execution_governance(
    protocol_path: Path,
    *,
    normalization_only: bool,
) -> dict[str, object]:
    """Authorize execution or return an auditable post-outcome qualification.

    Prospective/new-outcome execution requires an exact protocol-byte binding
    in a separate v4 seal.  The deliberately narrower normalization mode may
    run while the protocol remains a draft because its outcomes were already
    viewed, but it is permanently labelled post-outcome.
    """

    path = protocol_path.resolve()
    document, payload = _load_protocol(path)
    protocol_digest = _sha256(payload)
    if normalization_only:
        _validate_normalization_declaration(document)
        return {
            "analysis_mode": "POST_OUTCOME_NORMALIZATION_ONLY",
            "post_outcome_normalization": True,
            "prospective_or_confirmatory": False,
            "execution_authorized_by_seal": False,
            "protocol": {
                "path": str(path),
                "protocol_id": EXPECTED_PROTOCOL_ID,
                "version": EXPECTED_PROTOCOL_VERSION,
                "status": document.get("status"),
                "execution_authorized": document.get("execution_authorized"),
                "sha256": protocol_digest,
            },
            "seal": None,
            "qualification": (
                "already-viewed 2021-2023 F0/F3_full tree-domain normalization; "
                "not prospective, preregistered, independent, or confirmatory"
            ),
        }

    status = document.get("status")
    if status != AUTHORIZED_PROTOCOL_STATUS:
        raise ValueError(
            "new-outcome execution is governance-blocked: v4 protocol status "
            f"is {status!r}, expected {AUTHORIZED_PROTOCOL_STATUS!r}; use only "
            "--dry-run, or the restricted --normalization-only mode"
        )
    if document.get("execution_authorized") is not True:
        raise ValueError(
            "new-outcome execution is governance-blocked: "
            "v4 execution_authorized must be exactly true"
        )
    seal_path = _resolve_declared_path(document.get("seal_path"), label="seal_path")
    if not seal_path.is_file():
        raise ValueError(f"declared v4 seal does not exist: {seal_path}")
    try:
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"declared v4 seal is not valid JSON: {seal_path}") from exc
    if not isinstance(seal, dict):
        raise TypeError("declared v4 seal must be a JSON object")
    if seal.get("format") != PROTOCOL_SEAL_FORMAT:
        raise ValueError(
            f"v4 seal format mismatch: {seal.get('format')!r}; expected {PROTOCOL_SEAL_FORMAT!r}"
        )
    if seal.get("status") != AUTHORIZED_PROTOCOL_STATUS:
        raise ValueError("v4 seal status must be exactly 'SEALED'")
    if seal.get("execution_authorized") is not True:
        raise ValueError("v4 seal execution_authorized must be exactly true")
    binding = seal.get("protocol")
    if not isinstance(binding, dict):
        raise TypeError("v4 seal must contain a protocol byte-binding object")
    if binding.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        raise ValueError("v4 seal protocol_id does not match the protocol")
    if binding.get("version") != EXPECTED_PROTOCOL_VERSION:
        raise ValueError("v4 seal protocol version does not match the protocol")
    bound_path = _resolve_declared_path(binding.get("path"), label="seal protocol.path")
    if bound_path != path:
        raise ValueError(f"v4 seal binds a different protocol path: {bound_path} != {path}")
    if binding.get("sha256") != protocol_digest:
        raise ValueError("v4 seal protocol SHA-256 does not match the protocol bytes")
    seal_payload = seal_path.read_bytes()
    return {
        "analysis_mode": "SEALED_AUTHORIZED_EXTENSION_EXECUTION",
        "post_outcome_normalization": False,
        "prospective_or_confirmatory": False,
        "execution_authorized_by_seal": True,
        "protocol": {
            "path": str(path),
            "protocol_id": EXPECTED_PROTOCOL_ID,
            "version": EXPECTED_PROTOCOL_VERSION,
            "status": status,
            "execution_authorized": True,
            "sha256": protocol_digest,
        },
        "seal": {
            "path": str(seal_path),
            "format": PROTOCOL_SEAL_FORMAT,
            "status": AUTHORIZED_PROTOCOL_STATUS,
            "sha256": _sha256(seal_payload),
        },
        "qualification": "execution authorized by exact v4 protocol-byte seal",
    }


def shard_path(output_dir: Path, cell: Cell) -> Path:
    return output_dir / DEFAULT_SHARD_DIRNAME / (f"{cell.arm}_{cell.model}_h{cell.horizon}.parquet")


def _atomic_write_bytes(path: Path, payload: bytes, *, overwrite: bool) -> None:
    """Atomically create, or explicitly replace, a v4 artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing v4 artifact without --overwrite-v4: {path}"
        )
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(tmp, path)
        else:
            # link is an atomic create and fails rather than replacing if a
            # concurrent writer created the destination after our first check.
            os.link(tmp, path)
            tmp.unlink()
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_json(document: Mapping[str, object], path: Path, *, overwrite: bool) -> None:
    payload = (json.dumps(document, indent=1, sort_keys=True, default=str) + "\n").encode("utf-8")
    _atomic_write_bytes(path, payload, overwrite=overwrite)


def atomic_write_parquet(frame: pd.DataFrame, path: Path, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing v4 artifact without --overwrite-v4: {path}"
        )
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        frame.to_parquet(tmp, index=False)
        with tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(tmp, path)
        else:
            os.link(tmp, path)
            tmp.unlink()
    finally:
        if tmp.exists():
            tmp.unlink()


def fit_meteorological_climatologies(
    panel: pd.DataFrame,
    training_mask: np.ndarray,
    stations: Sequence[str],
) -> dict[str, F.HarmonicClimatology]:
    """Fit one 2006--2015 train-only climatology for each met variable."""

    station_tuple = tuple(str(station) for station in stations)
    if tuple(C.STATIONS) != station_tuple:
        raise ValueError(
            "C.STATIONS must equal the requested station registry before fitting "
            "meteorological climatologies"
        )
    mask = np.asarray(training_mask, dtype=bool)
    if mask.ndim != 1 or len(mask) != len(panel):
        raise ValueError("training_mask must align one-to-one with panel rows")
    if not mask.any():
        raise ValueError("training_mask contains no rows")
    missing = sorted(set(META_VARS) - set(panel.columns))
    if missing:
        raise ValueError(f"panel lacks meteorological variables: {missing}")
    return {
        variable: F.HarmonicClimatology.fit(
            panel,
            mask,
            target=variable,
            fit_stations=station_tuple,
        )
        for variable in META_VARS
    }


def _observed_mask(sub: pd.DataFrame, variable: str) -> np.ndarray:
    values = pd.to_numeric(sub[variable], errors="coerce").to_numpy(float)
    observed_column = f"{variable}_observed"
    if observed_column in sub:
        observed = (
            pd.to_numeric(sub[observed_column], errors="coerce").fillna(0.0).to_numpy(float) > 0.0
        )
    else:
        observed = np.isfinite(values)
    return observed & np.isfinite(values)


def future_met_columns(
    panel: pd.DataFrame,
    climatologies: Mapping[str, F.HarmonicClimatology],
    arm: str,
    *,
    max_future_days: int = MAX_FUTURE_DAYS,
) -> pd.DataFrame | None:
    """Build future forcing values and per-value F3_full substitution flags.

    F1 values always come from the corresponding variable's climatology and
    therefore are not counted as *missing-value substitutions*.  F3_full uses a
    realized gridded estimate only when it is finite and its inherited
    availability/``*_observed`` flag is true; otherwise that same variable's
    climatology supplies the value and the substitution flag is true.  The
    inherited column name does not imply a direct station observation.
    """

    if arm not in ALLOWED_ARMS:
        raise ValueError(f"unsupported forcing arm {arm!r}")
    if arm == "F0":
        return None
    if type(max_future_days) is not int or max_future_days < 1:
        raise ValueError("max_future_days must be a positive integer")
    missing_clims = sorted(set(META_VARS) - set(climatologies))
    if missing_clims:
        raise ValueError(f"missing meteorological climatologies: {missing_clims}")
    F.assert_strict_daily_panel(panel, expected_stations=tuple(C.STATIONS))

    rows: list[pd.DataFrame] = []
    for site, sub in panel.sort_values(["site_id", "DATE"]).groupby("site_id", sort=True):
        sub = sub.reset_index(drop=True)
        dates = pd.DatetimeIndex(pd.to_datetime(sub["DATE"]))
        out: dict[str, object] = {
            "site_id": np.full(len(sub), str(site)),
            "DATE": dates.to_numpy(),
        }
        for variable in META_VARS:
            values = pd.to_numeric(sub[variable], errors="coerce").to_numpy(float)
            available = _observed_mask(sub, variable)
            climatology = climatologies[variable]
            for step in range(1, max_future_days + 1):
                target_dates = dates + pd.to_timedelta(step, unit="D")
                target_doy = target_dates.dayofyear.to_numpy()
                clim_values = climatology.predict(str(site), target_doy)
                if not np.isfinite(clim_values).all():
                    raise ValueError(
                        f"{variable} climatology produced non-finite values for {site}"
                    )

                if arm == "F1":
                    future = np.asarray(clim_values, dtype=float)
                    substituted = np.zeros(len(sub), dtype=bool)
                else:  # F3_full
                    shifted = np.full(len(sub), np.nan, dtype=float)
                    shifted_available = np.zeros(len(sub), dtype=bool)
                    if step < len(sub):
                        shifted[:-step] = values[step:]
                        shifted_available[:-step] = available[step:]
                    usable = shifted_available & np.isfinite(shifted)
                    future = np.where(usable, shifted, clim_values)
                    substituted = ~usable
                out[f"{variable}_fut{step}"] = future
                out[f"{variable}_fut{step}_substituted"] = substituted
        rows.append(pd.DataFrame(out))

    table = pd.concat(rows, ignore_index=True)
    table["site_id"] = table["site_id"].astype(str).str.zfill(8)
    table["DATE"] = pd.to_datetime(table["DATE"])
    if table.duplicated(["site_id", "DATE"]).any():
        raise AssertionError("future-forcing table duplicates a station-day")
    return table


def attach_future_features(
    tab: pd.DataFrame,
    future: pd.DataFrame | None,
    horizon: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Attach model features and non-feature substitution accounting."""

    if horizon not in ALLOWED_LEADS:
        raise ValueError(f"unsupported horizon {horizon}")
    out = tab.copy()
    for column in SUBSTITUTION_COLUMNS:
        out[column] = 0
    out["forcing_substitution_count"] = 0
    if future is None:
        return out, []

    value_columns = [
        f"{variable}_fut{step}" for variable in META_VARS for step in range(1, horizon + 1)
    ]
    flag_columns = [f"{column}_substituted" for column in value_columns]
    needed = ["site_id", "DATE", *value_columns, *flag_columns]
    missing = sorted(set(needed) - set(future.columns))
    if missing:
        raise ValueError(f"future-forcing table lacks columns: {missing}")
    merged = out.merge(
        future[needed],
        left_on=["site_id", "issue_date"],
        right_on=["site_id", "DATE"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["DATE"])
    if merged[value_columns + flag_columns].isna().any(axis=None):
        raise AssertionError("future-forcing merge left missing values or flags")

    added: list[str] = []
    keep_values: set[str] = set()
    for variable in META_VARS:
        parts = [f"{variable}_fut{step}" for step in range(1, horizon + 1)]
        flags = [f"{part}_substituted" for part in parts]
        target_column = f"{variable}_fut{horizon}"
        mean_column = f"{variable}_futmean{horizon}"
        merged[mean_column] = merged[parts].mean(axis=1, skipna=False)
        count_column = f"forcing_substitutions_{variable}"
        merged[count_column] = merged[flags].astype(bool).sum(axis=1).astype(int)
        keep_values.add(target_column)
        added.extend([target_column, mean_column])
    merged["forcing_substitution_count"] = (
        merged[list(SUBSTITUTION_COLUMNS)].sum(axis=1).astype(int)
    )
    drop_values = [column for column in value_columns if column not in keep_values]
    merged = merged.drop(columns=[*drop_values, *flag_columns])
    return merged, added


def load_reference_keys(path: Path = FINAL / "forecast_keys.parquet") -> pd.DataFrame:
    columns = [
        "key_id",
        "site_id",
        "horizon",
        "issue_date",
        "target_date",
        "y_true",
    ]
    reference = pd.read_parquet(path, columns=columns)
    reference["site_id"] = reference["site_id"].astype(str).str.zfill(8)
    reference["horizon"] = reference["horizon"].astype(int)
    reference["issue_date"] = pd.to_datetime(reference["issue_date"])
    reference["target_date"] = pd.to_datetime(reference["target_date"])
    if reference[columns].isna().any(axis=None):
        raise AssertionError("forecast-key registry contains missing required values")
    if (
        reference.duplicated("key_id").any()
        or reference.duplicated(["site_id", "horizon", "issue_date"]).any()
    ):
        raise AssertionError("forecast-key registry contains duplicate keys")
    expected_target = reference["issue_date"] + pd.to_timedelta(reference["horizon"], unit="D")
    if not reference["target_date"].equals(expected_target.rename("target_date")):
        raise AssertionError("forecast-key target_date != issue_date + horizon")
    return reference.sort_values(["horizon", "site_id", "issue_date"]).reset_index(drop=True)


def make_key_level_frame(
    evaluation: pd.DataFrame,
    y_pred: Sequence[float],
    y_damped: Sequence[float],
    *,
    arm: str,
    model: str,
    horizon: int,
) -> pd.DataFrame:
    """Create one v4 key-level shard in its canonical schema."""

    if arm not in ALLOWED_ARMS or model not in ALLOWED_MODELS:
        raise ValueError(f"invalid key-level identity: {arm}/{model}")
    if horizon not in ALLOWED_LEADS:
        raise ValueError(f"invalid key-level horizon: {horizon}")
    required = {
        "key_id",
        "site_id",
        "issue_date",
        "target_date",
        "y_true",
        *SUBSTITUTION_COLUMNS,
        "forcing_substitution_count",
    }
    missing = sorted(required - set(evaluation.columns))
    if missing:
        raise ValueError(f"evaluation table lacks key-level columns: {missing}")
    pred = np.asarray(y_pred, dtype=float)
    damped = np.asarray(y_damped, dtype=float)
    if len(pred) != len(evaluation) or len(damped) != len(evaluation):
        raise ValueError("predictions must align one-to-one with evaluation rows")
    frame = evaluation[list(required)].copy()
    frame["arm"] = arm
    frame["protocol_arm"] = PROTOCOL_ARM_BY_ARM[arm]
    frame["model"] = model
    frame["horizon"] = int(horizon)
    frame["y_pred"] = pred
    frame["y_damped"] = damped
    return (
        frame[list(KEY_LEVEL_COLUMNS)]
        .sort_values(["site_id", "issue_date"], kind="mergesort")
        .reset_index(drop=True)
    )


def validate_key_level_frame(
    frame: pd.DataFrame,
    *,
    cell: Cell | None = None,
    reference: pd.DataFrame | None = None,
    require_full_registry: bool = False,
) -> pd.DataFrame:
    """Validate schema, identity, chronology, accounting, and registry binding."""

    if tuple(frame.columns) != KEY_LEVEL_COLUMNS:
        raise AssertionError(
            f"key-level schema mismatch: got {list(frame.columns)}, "
            f"expected {list(KEY_LEVEL_COLUMNS)}"
        )
    if frame.empty:
        raise AssertionError("key-level shard is empty")
    checked = frame.copy()
    checked["site_id"] = checked["site_id"].astype(str).str.zfill(8)
    checked["issue_date"] = pd.to_datetime(checked["issue_date"])
    checked["target_date"] = pd.to_datetime(checked["target_date"])
    checked["horizon"] = checked["horizon"].astype(int)
    if (
        checked.duplicated("key_id").any()
        or checked.duplicated(["site_id", "horizon", "issue_date"]).any()
    ):
        raise AssertionError("key-level shard contains duplicate keys")
    if checked[["key_id", "site_id", "issue_date", "target_date"]].isna().any(axis=None):
        raise AssertionError("key-level shard contains missing key fields")
    expected_target = checked["issue_date"] + pd.to_timedelta(checked["horizon"], unit="D")
    if not checked["target_date"].equals(expected_target.rename("target_date")):
        raise AssertionError("key-level target_date != issue_date + horizon")
    if not np.isfinite(checked[["y_true", "y_pred"]].to_numpy(float)).all():
        raise AssertionError("key-level y_true/y_pred must be finite")

    counts = checked[list(SUBSTITUTION_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    total = pd.to_numeric(checked["forcing_substitution_count"], errors="coerce")
    if counts.isna().any(axis=None) or total.isna().any():
        raise AssertionError("substitution accounting contains non-numeric values")
    if (counts.to_numpy(float) < 0).any() or (total.to_numpy(float) < 0).any():
        raise AssertionError("substitution accounting contains negative values")
    if not np.equal(counts.to_numpy(float), np.floor(counts.to_numpy(float))).all():
        raise AssertionError("per-variable substitution counts must be integers")
    if not np.equal(total.to_numpy(float), np.floor(total.to_numpy(float))).all():
        raise AssertionError("total substitution counts must be integers")
    if not np.array_equal(counts.sum(axis=1).to_numpy(int), total.to_numpy(int)):
        raise AssertionError("total substitution count != sum of variable counts")
    unknown_arms = sorted(set(checked["arm"]) - set(ALLOWED_ARMS))
    unknown_protocol_arms = sorted(set(checked["protocol_arm"]) - set(ALLOWED_ARMS))
    unknown_models = sorted(set(checked["model"]) - set(ALLOWED_MODELS))
    unknown_horizons = sorted(set(checked["horizon"]) - set(ALLOWED_LEADS))
    if unknown_arms or unknown_protocol_arms or unknown_models or unknown_horizons:
        raise AssertionError(
            "key-level identity contains unsupported values: "
            f"arms={unknown_arms}, protocol_arms={unknown_protocol_arms}, "
            f"models={unknown_models}, "
            f"horizons={unknown_horizons}"
        )
    expected_protocol_arm = checked["arm"].map(PROTOCOL_ARM_BY_ARM)
    if not checked["protocol_arm"].equals(expected_protocol_arm.rename("protocol_arm")):
        raise AssertionError(
            "protocol_arm must exactly identify the canonical v4 protocol arm; "
            "F3_full cannot be relabelled F3_temperature_only"
        )
    if (counts.to_numpy(int) > checked["horizon"].to_numpy(int)[:, None]).any():
        raise AssertionError("per-variable substitutions exceed the forecast horizon")
    if (total.to_numpy(int) > len(META_VARS) * checked["horizon"].to_numpy(int)).any():
        raise AssertionError("total substitutions exceed variables times horizon")
    no_substitution_arm = checked["arm"].isin(["F0", "F1"])
    if total[no_substitution_arm].ne(0).any():
        raise AssertionError("F0/F1 rows cannot report missing-value substitutions")

    if cell is not None:
        identity = set(checked[["arm", "model", "horizon"]].itertuples(index=False, name=None))
        if identity != {(cell.arm, cell.model, cell.horizon)}:
            raise AssertionError(f"shard identity differs from planned cell {cell}: {identity}")

    if reference is not None:
        registry_columns = [
            "key_id",
            "site_id",
            "horizon",
            "issue_date",
            "target_date",
            "y_true",
        ]
        absent = sorted(set(registry_columns) - set(reference.columns))
        if absent:
            raise AssertionError(f"registry lacks key-binding columns: {absent}")
        if cell is None:
            horizons = checked["horizon"].unique()
            if len(horizons) != 1:
                raise AssertionError("registry validation requires one horizon")
            horizon = int(horizons[0])
        else:
            horizon = cell.horizon
        ref = reference[reference.horizon == horizon][registry_columns].copy()
        ref["site_id"] = ref["site_id"].astype(str).str.zfill(8)
        ref["horizon"] = ref["horizon"].astype(int)
        ref["issue_date"] = pd.to_datetime(ref["issue_date"])
        ref["target_date"] = pd.to_datetime(ref["target_date"])
        if ref.duplicated("key_id").any():
            raise AssertionError("registry contains duplicate key_id values")
        ref_ids = set(ref["key_id"])
        cell_ids = set(checked["key_id"])
        extra = cell_ids - ref_ids
        missing = ref_ids - cell_ids
        if extra or (require_full_registry and missing):
            raise AssertionError(f"registry breach: {len(missing)} missing, {len(extra)} outside")
        joined = checked[registry_columns].merge(
            ref,
            on="key_id",
            how="left",
            validate="one_to_one",
            suffixes=("", "_reference"),
        )
        binding_mismatch = np.zeros(len(joined), dtype=bool)
        for column in ("site_id", "horizon", "issue_date", "target_date"):
            binding_mismatch |= joined[column].ne(joined[f"{column}_reference"]).to_numpy()
        if binding_mismatch.any():
            raise AssertionError("key-level identity/chronology differs from the registry")
        if not target_values_equal(
            joined["y_true"].to_numpy(float),
            joined["y_true_reference"].to_numpy(float),
        ):
            raise AssertionError(
                "key-level target values differ from the registry beyond "
                f"absolute tolerance {TARGET_ABSOLUTE_TOLERANCE:g}"
            )
    return checked


def station_metrics_from_predictions(
    predictions: pd.DataFrame,
    *,
    min_keys: int = MIN_REPORTABLE_KEYS,
) -> pd.DataFrame:
    if type(min_keys) is not int or min_keys < 1:
        raise ValueError("min_keys must be a positive integer")
    rows: list[dict[str, object]] = []
    for (arm, model, site, horizon), group in predictions.groupby(
        ["arm", "model", "site_id", "horizon"], sort=True
    ):
        error = group["y_pred"].to_numpy(float) - group["y_true"].to_numpy(float)
        damped_error = group["y_damped"].to_numpy(float) - group["y_true"].to_numpy(float)
        row: dict[str, object] = {
            "arm": arm,
            "protocol_arm": PROTOCOL_ARM_BY_ARM[arm],
            "model": model,
            "model_status": MODEL_STATUS_BY_MODEL[model],
            "site_id": site,
            "horizon": int(horizon),
            "n": len(group),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "rmse_damped": (
                float(np.sqrt(np.mean(np.square(damped_error))))
                if np.isfinite(damped_error).all()
                else float("nan")
            ),
        }
        for column in SUBSTITUTION_COLUMNS:
            row[column] = int(group[column].sum())
        row["forcing_substitution_count"] = int(group["forcing_substitution_count"].sum())
        row["reportable"] = bool(len(group) >= min_keys)
        rows.append(row)
    return pd.DataFrame(rows, columns=EFFECT_COLUMNS)


def paired_arm_contrasts(
    predictions: pd.DataFrame,
    *,
    min_keys: int = MIN_REPORTABLE_KEYS,
) -> pd.DataFrame:
    """Compute declared tree and unofficial-hybrid contrasts on identical keys."""

    rows: list[dict[str, object]] = []

    def contrast_role(model: str, arm: str) -> str:
        if model == AIR2STREAM_MODEL:
            return "secondary_unofficial_hybrid_F3_full_minus_F1"
        if arm == "F3_full":
            return "primary_tree_F3_full_minus_F0"
        if arm == "F1":
            return "diagnostic_tree_F1_minus_F0"
        raise AssertionError(f"no declared contrast role for {model}/{arm}")

    for (model, horizon), model_rows in predictions.groupby(["model", "horizon"], sort=True):
        if model not in REFERENCE_ARM_BY_MODEL:
            raise AssertionError(f"no declared contrast reference for model {model!r}")
        reference_arm = REFERENCE_ARM_BY_MODEL[model]
        reference = model_rows[model_rows.arm == reference_arm][
            ["key_id", "site_id", "y_true", "y_pred"]
        ].rename(columns={"y_pred": "y_pred_reference"})
        if reference.empty:
            if set(model_rows.arm) - {reference_arm}:
                raise AssertionError(
                    f"declared reference arm {reference_arm} is missing for {model}/h{horizon}"
                )
            continue
        candidate_arms = sorted(set(model_rows.arm) - {reference_arm})
        if model == AIR2STREAM_MODEL and candidate_arms != ["F3_full"]:
            raise AssertionError(
                "air2stream contrast requires exactly F3_full minus F1; "
                f"got candidates={candidate_arms}"
            )
        for arm in candidate_arms:
            candidate = model_rows[model_rows.arm == arm][
                ["key_id", "site_id", "y_true", "y_pred"]
            ].rename(columns={"y_pred": "y_pred_arm"})
            paired = candidate.merge(
                reference,
                on=["key_id", "site_id", "y_true"],
                how="inner",
                validate="one_to_one",
            )
            expected_candidate = set(candidate.key_id)
            expected_reference = set(reference.key_id)
            paired_ids = set(paired.key_id)
            if paired_ids != expected_candidate or paired_ids != expected_reference:
                raise AssertionError(
                    f"cross-arm key mismatch for {model}/h{horizon}/{arm} vs {reference_arm}"
                )
            for site, group in paired.groupby("site_id", sort=True):
                arm_error = group["y_pred_arm"].to_numpy(float) - group["y_true"].to_numpy(float)
                reference_error = group["y_pred_reference"].to_numpy(float) - group[
                    "y_true"
                ].to_numpy(float)
                rmse_arm = float(np.sqrt(np.mean(np.square(arm_error))))
                rmse_reference = float(np.sqrt(np.mean(np.square(reference_error))))
                rows.append(
                    {
                        "arm": arm,
                        "protocol_arm": PROTOCOL_ARM_BY_ARM[arm],
                        "reference_arm": reference_arm,
                        "reference_protocol_arm": PROTOCOL_ARM_BY_ARM[reference_arm],
                        "model": model,
                        "contrast_role": contrast_role(model, arm),
                        "model_status": MODEL_STATUS_BY_MODEL[model],
                        "site_id": site,
                        "horizon": int(horizon),
                        "n_common_keys": len(group),
                        "rmse_arm": rmse_arm,
                        "rmse_reference": rmse_reference,
                        "delta_rmse": rmse_arm - rmse_reference,
                        "reportable": bool(len(group) >= min_keys),
                    }
                )
    return pd.DataFrame(rows, columns=CONTRAST_COLUMNS)


def summarize_paired_contrasts(contrasts: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Summarize the median of station-paired deltas, never marginal medians."""

    summary: dict[str, dict[str, object]] = {}
    reportable = contrasts[contrasts["reportable"].astype(bool)]
    for (
        arm,
        protocol_arm,
        reference_arm,
        reference_protocol_arm,
        model,
        contrast_role,
        model_status,
        horizon,
    ), group in reportable.groupby(
        [
            "arm",
            "protocol_arm",
            "reference_arm",
            "reference_protocol_arm",
            "model",
            "contrast_role",
            "model_status",
            "horizon",
        ],
        sort=True,
    ):
        delta = group["delta_rmse"].to_numpy(float)
        key = f"{model}/h{int(horizon)}/{arm}_minus_{reference_arm}"
        item: dict[str, object] = {
            "estimand": f"median_i[RMSE_i({arm})-RMSE_i({reference_arm})]",
            "protocol_arm": protocol_arm,
            "reference_protocol_arm": reference_protocol_arm,
            "contrast_role": contrast_role,
            "model_status": model_status,
            "primary_forcing_value": bool(model in TREE_MODELS and arm == "F3_full"),
            "is_hybrid": bool(model == AIR2STREAM_MODEL),
            "validated_hybrid": False if model == AIR2STREAM_MODEL else None,
            "n_reportable_stations": len(group),
            "median_station_delta_rmse": float(np.median(delta)),
            "iqr_station_delta_rmse": [float(value) for value in np.percentile(delta, [25, 75])],
            "station_win_fraction": float(np.mean(delta < 0.0)),
            "median_common_keys": float(group["n_common_keys"].median()),
        }
        if model in TREE_MODELS:
            forcing_value = -delta
            item.update(
                {
                    "forcing_value_estimand": (f"median_i[RMSE_i({reference_arm})-RMSE_i({arm})]"),
                    "median_forcing_value_rmse": float(np.median(forcing_value)),
                    "forcing_value_sign_relation": (
                        "per-station forcing_value_rmse = -delta_rmse; "
                        "positive forcing value means improvement"
                    ),
                }
            )
        summary[key] = item
    return summary


def _state_value_available(row: object, variable: str, value: float) -> bool:
    """Interpret an inherited observedness flag as availability, fail-closed."""

    flag_name = f"{variable}_observed"
    if not hasattr(row, flag_name):
        return bool(np.isfinite(value))
    try:
        flag = float(getattr(row, flag_name))
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(value) and np.isfinite(flag) and flag > 0.0)


def _build_station_state_maps(
    panel: pd.DataFrame,
) -> dict[str, dict[pd.Timestamp, StationState]]:
    """Build named hybrid states; exported for a regression test."""

    maps: dict[str, dict[pd.Timestamp, StationState]] = {}
    for site, sub in panel.groupby("site_id", sort=True):
        station: dict[pd.Timestamp, StationState] = {}
        for row in sub.itertuples(index=False):
            date = pd.Timestamp(row.DATE)
            air = float(row.TEMP)
            water = float(row.WTEMP)
            flow = float(row.FLOW)
            station[date] = StationState(
                air_temp=air,
                water_temp=water,
                flow=flow,
                day_of_year=int(date.dayofyear),
                air_temp_available=_state_value_available(row, "TEMP", air),
                water_temp_available=_state_value_available(row, "WTEMP", water),
                flow_available=_state_value_available(row, "FLOW", flow),
            )
        maps[str(site).zfill(8)] = station
    return maps


def _fit_air2stream_station(args):
    site, sub = args
    from thermoroute.air2stream import fit as air2stream_fit

    air = sub["TEMP"].to_numpy(float)
    flow = sub["FLOW"].to_numpy(float)
    water = sub["WTEMP"].to_numpy(float)
    day_of_year = pd.to_datetime(sub["DATE"]).dt.dayofyear.to_numpy()
    air_observed = _observed_mask(sub, "TEMP")
    flow_observed = _observed_mask(sub, "FLOW")
    water_observed = _observed_mask(sub, "WTEMP")
    fit_obj = air2stream_fit(
        np.where(air_observed, air, np.nan),
        np.where(flow_observed, flow, np.nan),
        water,
        day_of_year,
        variant="a8",
        obs=water_observed,
    )
    return site, fit_obj


def fit_air2stream_models(
    panel: pd.DataFrame,
    training_mask: np.ndarray,
    stations: Sequence[str],
    *,
    worker_budget: int,
) -> dict[str, object]:
    """Fit the unofficial hybrid once per station on 2006--2015 rows."""

    import multiprocessing as mp

    mask = np.asarray(training_mask, dtype=bool)
    jobs = []
    for site in stations:
        selected = panel.site_id.eq(site).to_numpy() & mask
        sub = panel.loc[selected].copy().sort_values("DATE").reset_index(drop=True)
        if not sub.empty:
            jobs.append((site, sub))
    if not jobs:
        raise RuntimeError("no station has rows for Air2stream fitting")
    if type(worker_budget) is not int or worker_budget < 1:
        raise ValueError("air2stream worker_budget must be a positive integer")
    with mp.Pool(min(worker_budget, len(jobs))) as pool:
        return dict(pool.imap_unordered(_fit_air2stream_station, jobs))


def resolve_air2stream_worker_budget(requested: int | None) -> int:
    """Resolve a reproducible process budget capped by the formal thread policy."""

    raw_limit = os.environ.get("THERMOROUTE_FORMAL_THREADS") or "1"
    try:
        formal_limit = int(raw_limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("THERMOROUTE_FORMAL_THREADS must be a positive integer") from exc
    if formal_limit < 1:
        raise ValueError("THERMOROUTE_FORMAL_THREADS must be a positive integer")
    budget = 1 if requested is None else requested
    if type(budget) is not int or budget < 1:
        raise ValueError("--air2stream-workers must be a positive integer")
    if budget > formal_limit:
        raise ValueError(
            f"--air2stream-workers={budget} exceeds THERMOROUTE_FORMAL_THREADS={formal_limit}"
        )
    return budget


def run_air2stream_cell(
    fits: Mapping[str, object],
    station_maps: Mapping[str, Mapping[pd.Timestamp, StationState]],
    temp_climatology: F.HarmonicClimatology,
    reference: pd.DataFrame,
    *,
    arm: str,
    horizon: int,
) -> pd.DataFrame:
    """Score one F1/F3_full hybrid cell using named WTEMP/FLOW issue state."""

    from thermoroute.air2stream import forecast_horizon

    if arm not in {"F1", "F3_full"}:
        raise ValueError("air2stream has only F1 and F3_full forms")
    if horizon not in ALLOWED_LEADS:
        raise ValueError(f"unsupported air2stream horizon {horizon}")
    rows: list[dict[str, object]] = []
    predictions: list[float] = []
    for record in reference[reference.horizon == horizon].itertuples(index=False):
        site = str(record.site_id).zfill(8)
        states = station_maps.get(site)
        if site not in fits or states is None:
            continue
        issue = pd.Timestamp(record.issue_date)
        state = states.get(issue)
        if state is None or not (
            state.water_temp_available
            and state.flow_available
            and np.isfinite(state.water_temp)
            and np.isfinite(state.flow)
            and state.flow > 0.0
        ):
            continue
        future_air: list[float] = []
        future_doy: list[int] = []
        substitutions = 0
        for step in range(1, horizon + 1):
            date = issue + pd.Timedelta(days=step)
            future_doy.append(int(date.dayofyear))
            future_state = states.get(date)
            if (
                arm == "F3_full"
                and future_state is not None
                and future_state.air_temp_available
                and np.isfinite(future_state.air_temp)
            ):
                future_air.append(float(future_state.air_temp))
            else:
                future_air.append(
                    float(temp_climatology.predict(site, np.asarray([int(date.dayofyear)]))[0])
                )
                if arm == "F3_full":
                    substitutions += 1
        prediction = forecast_horizon(
            fits[site],
            state.water_temp,
            state.flow,
            state.day_of_year,
            np.asarray(future_air, dtype=float),
            np.asarray(future_doy, dtype=int),
        )
        row = {
            "key_id": record.key_id,
            "site_id": site,
            "issue_date": issue,
            "target_date": pd.Timestamp(record.target_date),
            "y_true": float(record.y_true),
            **{column: 0 for column in SUBSTITUTION_COLUMNS},
            "forcing_substitution_count": substitutions,
        }
        row["forcing_substitutions_TEMP"] = substitutions
        rows.append(row)
        predictions.append(float(prediction))
    if not rows:
        raise RuntimeError(f"air2stream {arm}/h{horizon} produced no predictions")
    evaluation = pd.DataFrame(rows)
    return make_key_level_frame(
        evaluation,
        predictions,
        np.full(len(predictions), np.nan),
        arm=arm,
        model=AIR2STREAM_MODEL,
        horizon=horizon,
    )


def _run_tree_cell(
    tab: pd.DataFrame,
    model_columns: Sequence[str],
    phi: Mapping[str, float],
    reference: pd.DataFrame,
    *,
    arm: str,
    model: str,
    horizon: int,
) -> pd.DataFrame:
    if model not in TREE_MODELS:
        raise ValueError(f"tree runner cannot execute model {model!r}")
    train = tab[
        tab.split.isin(["train", "val"])
        & (pd.to_numeric(tab["issue_wtemp_observed"], errors="coerce").fillna(0) > 0)
    ]
    if train.empty:
        raise RuntimeError(f"no admissible training rows for {arm}/{model}/h{horizon}")
    damped_train = train["clim_target"].to_numpy(float) + train["site_id"].map(phi).to_numpy(
        float
    ) ** horizon * (train["persistence"].to_numpy(float) - train["clim_t"].to_numpy(float))
    outcome = (
        train["y"].to_numpy(float) - damped_train
        if model == "ResidualLightGBM"
        else train["y"].to_numpy(float)
    )
    params = {
        "num_leaves": FROZEN_PARAMS[horizon]["num_leaves"],
        "min_child_samples": FROZEN_PARAMS[horizon]["min_child_samples"],
        "learning_rate": FROZEN_PARAMS[horizon]["learning_rate"],
    }
    validation_size = min(2000, len(train))
    fitted = _lgb_fit(
        train[list(model_columns)],
        outcome,
        train[list(model_columns)].to_numpy(float)[-validation_size:],
        outcome[-validation_size:],
        "regression",
        n_est=BEST_ITER[horizon],
        params_override=params,
    )

    ref = reference[reference.horizon == horizon].copy()
    evaluation = tab[tab.split.eq("confirm")].copy()
    evaluation = ref.merge(
        evaluation,
        on=["site_id", "issue_date"],
        how="left",
        validate="one_to_one",
        suffixes=("_reference", ""),
    )
    if evaluation[list(model_columns)].isna().any(axis=None):
        raise AssertionError(f"{arm}/{model}/h{horizon} declined registry features")
    if not (
        pd.to_datetime(evaluation["target_date_reference"])
        == pd.to_datetime(evaluation["target_date"])
    ).all():
        raise AssertionError("tree target dates differ from the registry")
    if not target_values_equal(
        evaluation["y_true"].to_numpy(float),
        evaluation["y"].to_numpy(float),
    ):
        raise AssertionError(
            "tree target values differ from the registry beyond "
            f"absolute tolerance {TARGET_ABSOLUTE_TOLERANCE:g}"
        )
    evaluation["target_date"] = evaluation["target_date_reference"]
    # ``y_true`` originated in the left-hand registry frame.  Never replace it
    # with the numerically equivalent panel value: every emitted shard binds
    # the exact registry representation.
    prediction = fitted.predict(evaluation[list(model_columns)], num_threads=1)
    damped = evaluation["clim_target"].to_numpy(float) + evaluation["site_id"].map(phi).to_numpy(
        float
    ) ** horizon * (
        evaluation["persistence"].to_numpy(float) - evaluation["clim_t"].to_numpy(float)
    )
    if model == "ResidualLightGBM":
        prediction = prediction + damped
    return make_key_level_frame(
        evaluation,
        prediction,
        damped,
        arm=arm,
        model=model,
        horizon=horizon,
    )


def _assert_expected_shard_names(output_dir: Path, cells: Sequence[Cell]) -> None:
    directory = output_dir / DEFAULT_SHARD_DIRNAME
    actual = {path.name for path in directory.glob("*.parquet")} if directory.exists() else set()
    expected = {shard_path(output_dir, cell).name for cell in cells}
    unexpected = actual - expected
    if unexpected:
        raise AssertionError(f"forcing_shards_v4 contains unexpected cells: {sorted(unexpected)}")


def validate_complete_shards(
    output_dir: Path,
    cells: Sequence[Cell],
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """Require the exact planned shard set and identical keys across arms."""

    _assert_expected_shard_names(output_dir, cells)
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for cell in cells:
        path = shard_path(output_dir, cell)
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(path.name)
            continue
        frame = validate_key_level_frame(
            pd.read_parquet(path),
            cell=cell,
            reference=reference,
            require_full_registry=cell.model in TREE_MODELS,
        )
        frames.append(frame)
    if missing:
        raise AssertionError(
            f"forcing v4 completeness gate: {len(missing)} cells missing: {missing[:10]}"
        )

    predictions = pd.concat(frames, ignore_index=True)
    for (model, horizon), group in predictions.groupby(["model", "horizon"]):
        key_sets = {arm: set(arm_rows.key_id) for arm, arm_rows in group.groupby("arm", sort=True)}
        first_arm = next(iter(key_sets))
        for arm, keys in key_sets.items():
            if keys != key_sets[first_arm]:
                raise AssertionError(
                    f"cross-arm key mismatch for {model}/h{horizon}: {arm} differs from {first_arm}"
                )
    return predictions


def build_summary(
    predictions: pd.DataFrame,
    effects: pd.DataFrame,
    contrasts: pd.DataFrame,
    cells: Sequence[Cell],
    *,
    governance: Mapping[str, object],
    legacy_f3_alias_used: bool,
    air2stream_worker_budget: int,
) -> dict[str, object]:
    absolute: dict[str, object] = {}
    learned_vs_damped: dict[str, object] = {}
    for (arm, model, horizon), group in effects[effects.reportable].groupby(
        ["arm", "model", "horizon"], sort=True
    ):
        absolute_key = f"{model}/h{int(horizon)}/{arm}"
        absolute[absolute_key] = {
            "protocol_arm": PROTOCOL_ARM_BY_ARM[arm],
            "model_status": MODEL_STATUS_BY_MODEL[model],
            "n_reportable_stations": len(group),
            "median_station_rmse": float(group.rmse.median()),
            "median_station_n_keys": int(group.n.median()),
        }
        if model in TREE_MODELS:
            paired_difference = group["rmse"].to_numpy(float) - group["rmse_damped"].to_numpy(float)
            if not np.isfinite(paired_difference).all():
                raise AssertionError(
                    f"tree learned-vs-damped rows are non-finite for {absolute_key}"
                )
            learned_vs_damped[absolute_key] = {
                "estimand": "median_i[RMSE_i(model)-RMSE_i(damped)]",
                "pairing": "within the same reportable station row and exact key set",
                "n_reportable_stations": len(group),
                "median_station_learned_minus_damped_rmse": float(np.median(paired_difference)),
            }
    substitutions: dict[str, object] = {}
    for (arm, model, horizon), group in predictions.groupby(["arm", "model", "horizon"], sort=True):
        substitutions[f"{model}/h{int(horizon)}/{arm}"] = {
            "protocol_arm": PROTOCOL_ARM_BY_ARM[arm],
            "model_status": MODEL_STATUS_BY_MODEL[model],
            "n_keys": len(group),
            "keys_with_any_substitution": int((group.forcing_substitution_count > 0).sum()),
            "total_substitutions": int(group.forcing_substitution_count.sum()),
            "by_variable": {
                variable: int(group[f"forcing_substitutions_{variable}"].sum())
                for variable in META_VARS
            },
        }
    p5_components: dict[str, object] = {}
    for model in TREE_MODELS:
        f0_key = f"{model}/h7/F0"
        f3_key = f"{model}/h7/F3_full"
        if f3_key not in absolute or f3_key not in learned_vs_damped:
            continue
        f3_rmse = float(absolute[f3_key]["median_station_rmse"])
        f3_advantage = float(learned_vs_damped[f3_key]["median_station_learned_minus_damped_rmse"])
        threshold_item: dict[str, object] = {
            "registered_threshold_c": -0.30,
            "comparison": "median_station_learned_minus_damped_rmse <= threshold",
            "observed_median_station_learned_minus_damped_rmse": f3_advantage,
            "meets_registered_threshold": bool(f3_advantage <= -0.30),
        }
        if f0_key in learned_vs_damped:
            f0_advantage = float(
                learned_vs_damped[f0_key]["median_station_learned_minus_damped_rmse"]
            )
            expansion = f3_advantage - f0_advantage
            threshold_item.update(
                {
                    "F0_median_station_learned_minus_damped_rmse": f0_advantage,
                    "F3_full_minus_F0_advantage_change_rmse": expansion,
                    "registered_minimum_advantage_expansion_c": 0.10,
                    "meets_registered_advantage_expansion": bool(expansion <= -0.10),
                }
            )
        p5_components[model] = {
            "seven_day_F3_full_rmse_range": {
                "registered_range_c": [1.15, 1.25],
                "observed_median_station_rmse": f3_rmse,
                "within_registered_range": bool(1.15 <= f3_rmse <= 1.25),
            },
            "seven_day_learned_advantage_threshold": threshold_item,
            "adjudication_status": "COMPONENT_VALUES_ONLY_NO_AUTOMATIC_VERDICT",
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_mode": governance["analysis_mode"],
        "post_outcome_normalization": governance["post_outcome_normalization"],
        "governance": dict(governance),
        "target_value_binding": target_value_binding_audit(),
        "arm_identity": {
            "canonical_cli_arms": list(ALLOWED_ARMS),
            "legacy_F3_alias_used": bool(legacy_f3_alias_used),
            "legacy_F3_alias_maps_to": "F3_full",
            "F3_temperature_only_implemented": False,
            "interpretation_guard": (
                "Every row's protocol_arm is canonical; F3_full output must "
                "never be interpreted as F3_temperature_only"
            ),
        },
        "execution_resources": {
            "air2stream_worker_budget": air2stream_worker_budget,
            "constraint": "not above THERMOROUTE_FORMAL_THREADS",
        },
        "F3_full_input_semantics": {
            "description": F3_INPUT_DESCRIPTION,
            "field_caveats": dict(F3_FIELD_CAVEATS),
        },
        "primary_estimand": (
            "tree models: median_i[RMSE_i(F3_full)-RMSE_i(F0)] on identical per-station keys"
        ),
        "diagnostic_estimand": (
            "tree models: median_i[RMSE_i(F1)-RMSE_i(F0)] on identical per-station keys"
        ),
        "secondary_hybrid_estimand": (
            "unofficial unvalidated air2stream: median_i[RMSE_i(F3_full)-RMSE_i(F1)]"
        ),
        "difference_of_marginal_medians_is_not_an_estimand": True,
        "reportability_min_common_keys": MIN_REPORTABLE_KEYS,
        "cells": [
            {
                "arm": cell.arm,
                "protocol_arm": PROTOCOL_ARM_BY_ARM[cell.arm],
                "model": cell.model,
                "model_status": MODEL_STATUS_BY_MODEL[cell.model],
                "evidence_status": ARM_EVIDENCE_STATUS[cell.arm],
                "horizon": cell.horizon,
            }
            for cell in cells
        ],
        "absolute_station_metrics": absolute,
        "paired_forcing_effects": summarize_paired_contrasts(contrasts),
        "paired_learned_vs_damped": learned_vs_damped,
        "P5_independent_components": p5_components,
        "P5_joint_verdict": "NOT_ADJUDICATED_BY_RUNNER",
        "substitution_accounting": substitutions,
    }


def run(args: argparse.Namespace) -> int:
    legacy_f3_alias_used = "F3" in [
        value.strip() for value in args.arms.split(",") if value.strip()
    ]
    arms, _models, _leads, cells = validate_request(
        args.arms,
        args.models,
        args.leads,
        allow_legacy_f3_alias=bool(args.allow_legacy_f3_alias),
    )
    air2stream_worker_budget = resolve_air2stream_worker_budget(args.air2stream_workers)
    if args.normalization_only:
        validate_normalization_scope(cells)
    if args.resume and args.overwrite_v4:
        raise ValueError("--resume and --overwrite-v4 are mutually exclusive")
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "output_dir": str(args.output_dir),
                "requested_execution_mode": (
                    "POST_OUTCOME_NORMALIZATION_ONLY"
                    if args.normalization_only
                    else "SEALED_AUTHORIZED_EXTENSION_EXECUTION"
                ),
                "protocol_check": ("skipped_by_dry_run" if args.dry_run else str(args.protocol)),
                "execution_performed": False,
                "legacy_F3_alias_used": legacy_f3_alias_used,
                "air2stream_worker_budget": air2stream_worker_budget,
                "target_value_binding": target_value_binding_audit(),
                "cells": [cell.__dict__ for cell in cells],
            },
            indent=1,
        )
    )
    if args.dry_run:
        return 0

    governance = validate_execution_governance(
        Path(args.protocol),
        normalization_only=bool(args.normalization_only),
    )
    governance = {
        **governance,
        "execution_resources": {
            "air2stream_worker_budget": air2stream_worker_budget,
            "THERMOROUTE_FORMAL_THREADS": int(os.environ.get("THERMOROUTE_FORMAL_THREADS") or "1"),
        },
    }
    output_dir = Path(args.output_dir)
    _assert_expected_shard_names(output_dir, cells)
    registry = pd.read_csv(ROOT / "data_usgs" / "station_registry_v1.csv")
    stations = tuple(sorted(registry.site_no.astype(str).str.zfill(8)))
    C.STATIONS = stations
    panel = load_panel(registry)
    dates = pd.to_datetime(panel["DATE"]).to_numpy()
    training_mask = (dates >= np.datetime64("2006-01-01")) & (dates <= np.datetime64("2015-12-31"))
    validation_mask = (dates >= np.datetime64("2016-01-01")) & (
        dates <= np.datetime64("2017-12-31")
    )
    final_fit_mask = training_mask | validation_mask
    reference = load_reference_keys()

    imputer, water_climatology, anchor = fit_preprocessing(
        "L0", panel, final_fit_mask, list(stations), stations
    )
    panel_imputed = imputer.transform(panel)
    phi = {site: float(anchor.phi.get(site, 0.9)) for site in stations}
    need_future = any(cell.arm != "F0" for cell in cells)
    met_climatologies = (
        fit_meteorological_climatologies(panel, training_mask, stations) if need_future else {}
    )
    future_tables = {
        arm: future_met_columns(panel, met_climatologies, arm) for arm in arms if arm != "F0"
    }
    future_tables["F0"] = None

    base_tabs: dict[int, tuple[pd.DataFrame, list[str]]] = {}
    hybrid_cells = [cell for cell in cells if cell.model == AIR2STREAM_MODEL]
    hybrid_fits: dict[str, object] | None = None
    station_maps: dict[str, dict[pd.Timestamp, StationState]] | None = None
    if hybrid_cells:
        hybrid_fits = fit_air2stream_models(
            panel,
            training_mask,
            stations,
            worker_budget=air2stream_worker_budget,
        )
        station_maps = _build_station_state_maps(panel)

    for cell in cells:
        path = shard_path(output_dir, cell)
        if path.exists() and args.resume:
            validate_key_level_frame(
                pd.read_parquet(path),
                cell=cell,
                reference=reference,
                require_full_registry=cell.model in TREE_MODELS,
            )
            print(f"resume: validated {path.name}", flush=True)
            continue
        if path.exists() and not args.overwrite_v4:
            raise FileExistsError(f"v4 shard exists; use --resume or --overwrite-v4: {path}")

        if cell.model in TREE_MODELS:
            if cell.horizon not in base_tabs:
                base_tabs[cell.horizon] = build_features(
                    panel_imputed,
                    panel_imputed,
                    imputer,
                    water_climatology,
                    cell.horizon,
                    [],
                    "L0",
                )
            base_tab, base_columns = base_tabs[cell.horizon]
            tab, future_columns = attach_future_features(
                base_tab, future_tables[cell.arm], cell.horizon
            )
            frame = _run_tree_cell(
                tab,
                [*base_columns, *future_columns],
                phi,
                reference,
                arm=cell.arm,
                model=cell.model,
                horizon=cell.horizon,
            )
        else:
            assert hybrid_fits is not None and station_maps is not None
            frame = run_air2stream_cell(
                hybrid_fits,
                station_maps,
                met_climatologies["TEMP"],
                reference,
                arm=cell.arm,
                horizon=cell.horizon,
            )
        validate_key_level_frame(
            frame,
            cell=cell,
            reference=reference,
            require_full_registry=cell.model in TREE_MODELS,
        )
        atomic_write_parquet(frame, path, overwrite=args.overwrite_v4)
        print(f"wrote {path.name}: {len(frame)} keys", flush=True)

    predictions = validate_complete_shards(output_dir, cells, reference)
    effects = station_metrics_from_predictions(predictions)
    contrasts = paired_arm_contrasts(predictions)
    summary = build_summary(
        predictions,
        effects,
        contrasts,
        cells,
        governance=governance,
        legacy_f3_alias_used=legacy_f3_alias_used,
        air2stream_worker_budget=air2stream_worker_budget,
    )
    derived_overwrite = bool(args.resume or args.overwrite_v4)
    atomic_write_parquet(effects, output_dir / EFFECTS_FILENAME, overwrite=derived_overwrite)
    atomic_write_parquet(contrasts, output_dir / CONTRASTS_FILENAME, overwrite=derived_overwrite)
    atomic_write_json(summary, output_dir / SUMMARY_FILENAME, overwrite=derived_overwrite)
    print(
        f"forcing v4 complete: {len(cells)}/{len(cells)} cells; "
        f"{len(effects)} station metrics; {len(contrasts)} paired contrasts"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default="F0,F1,F3_full")
    parser.add_argument("--models", default="LightGBM,ResidualLightGBM")
    parser.add_argument("--leads", default="1,3,7")
    parser.add_argument("--output-dir", type=Path, default=FINAL)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--allow-legacy-f3-alias",
        action="store_true",
        help="explicitly map legacy CLI arm F3 to canonical protocol arm F3_full",
    )
    parser.add_argument(
        "--air2stream-workers",
        type=int,
        default=None,
        help=("frozen hybrid process budget (default 1), capped by THERMOROUTE_FORMAL_THREADS"),
    )
    parser.add_argument(
        "--normalization-only",
        action="store_true",
        help=(
            "allow only already-viewed F0/F3_full tree-domain post-outcome "
            "normalization while v4 is unsealed"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="validate and reuse existing requested v4 shards",
    )
    parser.add_argument(
        "--overwrite-v4",
        action="store_true",
        help="explicitly replace only v4-named artifacts, atomically",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the exact cell plan; read/write/train nothing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (
        AssertionError,
        FileExistsError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    sys.exit(main())

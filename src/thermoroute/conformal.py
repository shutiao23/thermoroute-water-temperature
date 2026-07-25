"""Conformalised quantile regression (Romano et al., 2019).

CQR adjusts the model's raw quantile interval using calibration-set conformity
scores, done per (station × horizon) — a Mondrian split that respects the
heteroscedasticity we expect across stations and lead times.  Any empirical
coverage value must come from the current lineage-bound experiment outputs;
this method description deliberately contains no withdrawn result.

Route A uses a conservative, versioned deployment policy.  The ordinary CQR
order statistic is a *signed* number and can be negative when the nominal band
already over-covers.  A negative value would shrink the interval and make the
manuscript's predeclared "widens only" statement false.  We therefore deploy
``qhat_plus = max(raw_qhat, 0)`` and preserve the raw signed registry in an
auditable metadata object.  Missing, non-numeric, or non-finite calibration
evidence fails closed instead of being silently discarded.

Note on the "guarantee": split-CQR's formal finite-sample (1−α) coverage holds
under exchangeability between calibration and test points. In this study the
2018 calibration rows, the temporally dependent 2019–2020 development rows, and
the predeclared 2021–2023 Route-A target years are not strictly exchangeable;
stations are not i.i.d. either. The development result is therefore only an
empirical marginal-coverage diagnostic. It is not an unconditional
finite-sample, conditional, or time-series coverage guarantee for Route A.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Hashable, Mapping, TypeVar, cast

import numpy as np
import pandas as pd

from . import config as C


CQR_POLICY_FORMAT = "thermoroute.cqr-nonnegative-offset-policy.v1"
CQR_OFFSET_AUDIT_FORMAT = "thermoroute.cqr-offset-audit.v1"


RegistryKey = TypeVar("RegistryKey", bound=Hashable)


class CQRContractError(ValueError):
    """A CQR input, frozen offset, or calibrated interval is unsafe."""


def cqr_policy_contract() -> dict[str, Any]:
    """Return the exact Route-A offset and interval deployment contract."""
    return {
        "format": CQR_POLICY_FORMAT,
        "raw_conformity_score": "max(q05_minus_y,y_minus_q95)",
        "raw_offset": "exact_split_conformal_signed_order_statistic",
        "deployed_offset": "qhat_plus=max(raw_qhat,0)",
        "negative_raw_offset_action": "clip_to_zero_never_shrink",
        "nonfinite_input_or_offset_action": "FAIL_CLOSED",
        "application": "q05_minus_qhat_plus;q50_unchanged;q95_plus_qhat_plus",
        "required_final_order": "finite_q05<=q50<=q95_and_nonempty_interval",
    }


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CQRContractError("CQR audit is not canonical finite JSON") from exc


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _normalise_registry_key(key: Hashable) -> str:
    if isinstance(key, tuple) and len(key) == 2:
        station, horizon = key
        parsed_horizon = _positive_integer(horizon, label="CQR horizon key")
        missing_station = pd.isna(station)
        if isinstance(missing_station, (bool, np.bool_)) and missing_station:
            raise CQRContractError("CQR station/group key is missing")
        station_value = str(station)
        if not station_value or "|" in station_value:
            raise CQRContractError("CQR station/group key is empty or ambiguous")
        return f"{station_value}|{parsed_horizon}"
    if isinstance(key, str):
        if key.count("|") != 1:
            raise CQRContractError(
                "canonical CQR text key must contain exactly one separator"
            )
        station, horizon_text = key.split("|", 1)
        if not station or not horizon_text.isascii() or not horizon_text.isdigit():
            raise CQRContractError("canonical CQR text key is malformed")
        horizon = int(horizon_text)
        if horizon < 1 or str(horizon) != horizon_text:
            raise CQRContractError(
                "canonical CQR text horizon must be a unique positive integer"
            )
        return f"{station}|{horizon}"
    raise CQRContractError("CQR registry key must be (station/group, horizon) or canonical text")


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise CQRContractError(f"{label} is not an integer")
    parsed = int(value)
    if parsed < 1:
        raise CQRContractError(f"{label} is not positive")
    return parsed


def _canonical_pair(station: object, horizon: object) -> tuple[str, int]:
    key = _normalise_registry_key(cast(Hashable, (station, horizon)))
    station_value, horizon_value = key.rsplit("|", 1)
    return station_value, int(horizon_value)


def _finite_registry(
    values: Mapping[RegistryKey, object], *, label: str
) -> dict[str, float]:
    if not isinstance(values, Mapping) or not values:
        raise CQRContractError(f"{label} registry is empty or malformed")
    output: dict[str, float] = {}
    for raw_key, raw_value in values.items():
        key = _normalise_registry_key(raw_key)
        if key in output:
            raise CQRContractError(f"{label} registry has duplicate canonical key {key}")
        if isinstance(raw_value, (bool, np.bool_)):
            raise CQRContractError(f"{label} offset is boolean")
        try:
            value = float(cast(Any, raw_value))
        except (TypeError, ValueError, OverflowError) as exc:
            raise CQRContractError(f"{label} offset is not numeric") from exc
        if not np.isfinite(value):
            raise CQRContractError(f"{label} offset is non-finite")
        output[key] = value
    return dict(sorted(output.items()))


def finalise_cqr_offsets(
    raw_offsets: Mapping[RegistryKey, object],
) -> tuple[dict[RegistryKey, float], dict[str, Any]]:
    """Clip signed raw CQR offsets at zero and return a reproducible audit.

    The returned deployed mapping preserves the caller's keys.  The audit uses
    canonical ``station|horizon`` keys so it can be written directly into a
    Torch or LightGBM bundle and independently checked before label opening.
    """
    raw = _finite_registry(raw_offsets, label="raw signed CQR")
    original_by_canonical = {
        _normalise_registry_key(key): key for key in raw_offsets
    }
    deployed_canonical = {
        key: (0.0 if value <= 0.0 else value) for key, value in raw.items()
    }
    deployed: dict[RegistryKey, float] = {
        original_by_canonical[key]: value
        for key, value in deployed_canonical.items()
    }
    audit = _build_cqr_offset_audit(raw, deployed_canonical)
    return deployed, audit


def _build_cqr_offset_audit(
    raw: Mapping[str, float], deployed: Mapping[str, float]
) -> dict[str, Any]:
    if set(raw) != set(deployed):
        raise CQRContractError("raw and deployed CQR registries have different keys")
    raw_values = np.asarray(list(raw.values()), dtype=float)
    deployed_values = np.asarray(list(deployed.values()), dtype=float)
    if (
        not len(raw_values)
        or not np.isfinite(raw_values).all()
        or not np.isfinite(deployed_values).all()
        or (deployed_values < 0.0).any()
    ):
        raise CQRContractError("CQR audit contains an empty, non-finite, or negative registry")
    expected = np.where(raw_values <= 0.0, 0.0, raw_values)
    if not np.array_equal(deployed_values, expected):
        raise CQRContractError("deployed CQR offsets do not equal max(raw, 0)")
    negative = raw_values < 0.0
    zero = raw_values == 0.0
    positive = raw_values > 0.0
    return {
        "format": CQR_OFFSET_AUDIT_FORMAT,
        "policy": cqr_policy_contract(),
        "offset_count": int(len(raw_values)),
        "raw_signed_offsets": dict(raw),
        "raw_signed_offsets_sha256": _sha256_json(dict(raw)),
        "raw_negative_count": int(negative.sum()),
        "raw_zero_count": int(zero.sum()),
        "raw_positive_count": int(positive.sum()),
        "raw_min": float(raw_values.min()),
        "raw_max": float(raw_values.max()),
        "clipped_count": int(negative.sum()),
        "deployed_offsets_sha256": _sha256_json(dict(deployed)),
        "deployed_min": float(deployed_values.min()),
        "deployed_max": float(deployed_values.max()),
        "deployed_all_nonnegative": True,
    }


def validate_cqr_offset_bundle(
    deployed_offsets: Mapping[Hashable, object],
    policy: object,
    audit: object,
) -> dict[str, Any]:
    """Validate a frozen CQR registry and its raw-signed provenance audit."""
    if policy != cqr_policy_contract():
        raise CQRContractError("frozen CQR policy differs from Route-A policy")
    deployed = _finite_registry(deployed_offsets, label="deployed CQR")
    if any(value < 0.0 for value in deployed.values()):
        raise CQRContractError("deployed CQR offset is negative")
    if not isinstance(audit, Mapping):
        raise CQRContractError("CQR offset audit is missing or malformed")
    raw_value = audit.get("raw_signed_offsets")
    raw = _finite_registry(raw_value, label="raw signed CQR") if isinstance(
        raw_value, Mapping
    ) else None
    if raw is None:
        raise CQRContractError("CQR audit lacks the raw signed offset registry")
    expected = _build_cqr_offset_audit(raw, deployed)
    if dict(audit) != expected:
        raise CQRContractError("CQR offset audit is stale, incomplete, or tampered")
    return expected


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Exact split-conformal order statistic, or +inf when unattainable.

    Non-finite observations are evidence failures, not rows that may be silently
    discarded.  Callers freezing a formal offset must also reject the +inf
    sentinel returned when the requested order statistic is unattainable.
    """
    if isinstance(alpha, (bool, np.bool_)):
        raise CQRContractError("alpha must be finite and strictly between zero and one")
    try:
        alpha_value = float(cast(Any, alpha))
    except (TypeError, ValueError, OverflowError) as exc:
        raise CQRContractError(
            "alpha must be finite and strictly between zero and one"
        ) from exc
    if not np.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
        raise CQRContractError("alpha must be finite and strictly between zero and one")
    try:
        scores = np.asarray(scores, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CQRContractError("conformity scores are not numeric") from exc
    if scores.ndim != 1 or len(scores) == 0:
        raise CQRContractError("conformity scores must be a non-empty vector")
    if not np.isfinite(scores).all():
        raise CQRContractError("conformity scores contain non-finite evidence")
    scores = np.sort(scores)
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha_value)))
    return float(scores[k - 1]) if k <= n else float("inf")


def _calibration_scores(frame: pd.DataFrame, *, label: str) -> np.ndarray:
    required = {"y_true", "q05", "q95"}
    missing = required - set(frame)
    if missing:
        raise CQRContractError(f"{label} frame missing columns: {sorted(missing)}")
    try:
        values = frame[["y_true", "q05", "q95"]].to_numpy(dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CQRContractError(f"{label} heads are not numeric") from exc
    if not len(values) or not np.isfinite(values).all():
        raise CQRContractError(f"{label} heads are empty or non-finite")
    y, lo, hi = values.T
    if (lo >= hi).any():
        raise CQRContractError(f"{label} nominal intervals are empty/crossed")
    return np.maximum(lo - y, y - hi)


def _raw_cqr_offsets(
    cal: pd.DataFrame, alpha: float, purge_boundary: bool
) -> dict[tuple[str, int], float]:
    required = {"site_id", "horizon", "y_true", "q05", "q95"}
    missing = required - set(cal)
    if missing:
        raise CQRContractError(f"calibration frame missing columns: {sorted(missing)}")
    cal = cal.copy()
    if purge_boundary and "target_date" in cal.columns:
        target_date = pd.to_datetime(cal["target_date"], errors="coerce")
        if target_date.isna().any():
            raise CQRContractError("calibration target_date contains invalid values")
        cal_end = pd.Timestamp(C.SPLIT.calib[1])
        cal = cal[target_date <= cal_end]
    if cal.empty:
        raise CQRContractError("calibration frame is empty after boundary purge")
    if cal[["site_id", "horizon"]].isna().any(axis=None):
        raise CQRContractError("calibration group keys contain missing evidence")
    raw: dict[tuple[str, int], float] = {}
    for (station, horizon), group in cal.groupby(["site_id", "horizon"], sort=True):
        score = _calibration_scores(group, label=f"CQR {station}/h{horizon}")
        key = _canonical_pair(station, horizon)
        if key in raw:
            raise CQRContractError("calibration groups alias one canonical CQR key")
        raw[key] = conformal_quantile(score, alpha)
    return raw


def cqr_offsets_with_audit(
    cal: pd.DataFrame,
    alpha: float = 0.10,
    purge_boundary: bool = True,
) -> tuple[dict[tuple[str, int], float], dict[str, Any]]:
    """Return conservative per-site offsets plus the raw-signed audit."""
    return finalise_cqr_offsets(_raw_cqr_offsets(cal, alpha, purge_boundary))


def cqr_offsets(
    cal: pd.DataFrame, alpha: float = 0.10, purge_boundary: bool = True
) -> dict[tuple[str, int], float]:
    """Per-(site, horizon) nonnegative CQR offset ``max(raw qhat, 0)``."""
    offsets, _audit = cqr_offsets_with_audit(cal, alpha, purge_boundary)
    return offsets


def block_cqr_offsets(
    cal: pd.DataFrame, alpha: float = 0.10, block_days: int = 7
) -> dict[tuple[str, int], float]:
    """Temporal-block CQR with conservative nonnegative deployed offsets."""
    block_size = _positive_integer(block_days, label="block_days")
    required = {"site_id", "horizon", "issue_date", "y_true", "q05", "q95"}
    missing = required - set(cal)
    if missing:
        raise CQRContractError(f"calibration frame missing columns: {sorted(missing)}")
    if cal[["site_id", "horizon"]].isna().any(axis=None):
        raise CQRContractError("block-CQR group keys contain missing evidence")
    raw: dict[tuple[str, int], float] = {}
    for (site, horizon), group in cal.groupby(["site_id", "horizon"], sort=True):
        issue_date = pd.to_datetime(group["issue_date"], errors="coerce")
        if issue_date.isna().any():
            raise CQRContractError("block-CQR issue_date contains invalid values")
        group = group.assign(_issue_date=issue_date).sort_values(
            "_issue_date", kind="mergesort"
        ).reset_index(drop=True)
        score = _calibration_scores(group, label=f"block CQR {site}/h{horizon}")
        block = np.arange(len(group)) // block_size
        block_maxima = np.asarray(
            [score[block == index].max() for index in np.unique(block)],
            dtype=float,
        )
        key = _canonical_pair(site, horizon)
        if key in raw:
            raise CQRContractError("block-CQR groups alias one canonical key")
        raw[key] = conformal_quantile(block_maxima, alpha)
    offsets, _audit = finalise_cqr_offsets(raw)
    return offsets


def hierarchical_cqr_offsets(
    cal: pd.DataFrame,
    group_col: str,
    alpha: float = 0.10,
    min_group: int = 100,
) -> dict[tuple[str, int], float]:
    """Pooled/fallback CQR with conservative nonnegative deployed offsets."""
    required = {group_col, "horizon", "y_true", "q05", "q95"}
    missing = required - set(cal)
    if missing:
        raise CQRContractError(f"calibration frame missing columns: {sorted(missing)}")
    if cal[[group_col, "horizon"]].isna().any(axis=None):
        raise CQRContractError(
            "hierarchical CQR group keys contain missing evidence"
        )
    minimum = _positive_integer(min_group, label="min_group")
    raw: dict[tuple[str, int], float] = {}
    for horizon, group in cal.groupby("horizon", sort=True):
        score = _calibration_scores(group, label=f"global CQR h{horizon}")
        key = _canonical_pair("__global__", horizon)
        if key in raw:
            raise CQRContractError("global CQR groups alias one canonical key")
        raw[key] = conformal_quantile(score, alpha)
    for (label, horizon), group in cal.groupby([group_col, "horizon"], sort=True):
        if len(group) < minimum:
            continue
        score = _calibration_scores(group, label=f"hierarchical CQR {label}/h{horizon}")
        key = _canonical_pair(label, horizon)
        if key in raw:
            raise CQRContractError("hierarchical CQR groups alias one canonical key")
        raw[key] = conformal_quantile(score, alpha)
    offsets, _audit = finalise_cqr_offsets(raw)
    return offsets


def _validate_prediction_heads(frame: pd.DataFrame, *, label: str) -> np.ndarray:
    required = {"q05", "q50", "q95"}
    missing = required - set(frame)
    if missing:
        raise CQRContractError(f"{label} prediction frame missing columns: {sorted(missing)}")
    try:
        heads = frame[["q05", "q50", "q95"]].to_numpy(dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CQRContractError(f"{label} prediction heads are not numeric") from exc
    if not len(heads) or not np.isfinite(heads).all():
        raise CQRContractError(f"{label} prediction heads contain non-finite values")
    if (
        (heads[:, 0] > heads[:, 1])
        | (heads[:, 1] > heads[:, 2])
        | (heads[:, 0] >= heads[:, 2])
    ).any():
        raise CQRContractError(f"{label} prediction heads cross or form an empty interval")
    return heads


def _validated_application_offsets(
    offsets: Mapping[Hashable, object], *, label: str
) -> dict[str, float]:
    values = _finite_registry(offsets, label=label)
    if any(value < 0.0 for value in values.values()):
        raise CQRContractError(f"{label} registry contains a negative offset")
    return values


def apply_hierarchical_cqr(
    pred: pd.DataFrame, offsets: Mapping[Hashable, object], group_col: str
) -> pd.DataFrame:
    """Apply pooled CQR while rejecting unsafe offsets and crossed heads."""
    output = pred.copy()
    _validate_prediction_heads(output, label="pre-CQR")
    if group_col not in output or "horizon" not in output:
        raise CQRContractError("hierarchical prediction frame lacks group/horizon")
    registry = _validated_application_offsets(offsets, label="hierarchical CQR")
    values: list[float] = []
    for label, horizon in zip(output[group_col], output["horizon"]):
        direct = "|".join(map(str, _canonical_pair(label, horizon)))
        fallback = "|".join(map(str, _canonical_pair("__global__", horizon)))
        if direct in registry:
            values.append(registry[direct])
        elif fallback in registry:
            values.append(registry[fallback])
        else:
            raise KeyError(f"hierarchical CQR lacks direct and global offset for {direct}")
    delta = np.asarray(values, dtype=float)
    output["q05"] = output["q05"].to_numpy(float) - delta
    output["q95"] = output["q95"].to_numpy(float) + delta
    _validate_prediction_heads(output, label="post-CQR")
    return output


def apply_cqr(
    pred: pd.DataFrame, offsets: Mapping[Hashable, object]
) -> pd.DataFrame:
    """Widen q05/q95 only; reject negative/non-finite offsets and bad heads."""
    out = pred.copy()
    _validate_prediction_heads(out, label="pre-CQR")
    if "site_id" not in out or "horizon" not in out:
        raise CQRContractError("prediction frame lacks site_id/horizon")
    registry = _validated_application_offsets(offsets, label="CQR")
    keys = [
        "|".join(map(str, _canonical_pair(site, horizon)))
        for site, horizon in zip(out["site_id"], out["horizon"])
    ]
    missing = sorted(set(keys) - set(registry))
    if missing:
        raise KeyError(
            "CQR offsets are missing prediction site×horizon keys: "
            f"{missing[:5]}"
        )
    delta = np.asarray([registry[key] for key in keys], dtype=float)
    out["q05"] = out["q05"].to_numpy(float) - delta
    out["q95"] = out["q95"].to_numpy(float) + delta
    _validate_prediction_heads(out, label="post-CQR")
    return out

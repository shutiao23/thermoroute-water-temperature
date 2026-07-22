"""Quantile-head identity and crossing diagnostics.

LightGBM fits q05, q50, and q95 as three independent nominal heads.  Sorting
their values exchanges those head identities whenever they cross.  The only
permitted repair therefore clips the two interval endpoints to the *nominal*
q50 head while leaving q50 byte-for-byte unchanged.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np


LIGHTGBM_QUANTILE_REPAIR_METHOD = "median_preserving_endpoint_clip_v1"
RAW_QUANTILE_CROSSING_AUDIT_FORMAT = (
    "thermoroute.raw-quantile-crossing-audit.v1"
)
RAW_QUANTILE_CROSSING_SUMMARY_FIELDS = frozenset({
    "rows",
    "forecast_key_sha256",
    "raw_prediction_sha256",
    "q05_above_q50_count",
    "q50_above_q95_count",
    "any_crossing_count",
    "any_crossing_rate",
    "maximum_crossing_gap_c",
})


class QuantileIdentityError(ValueError):
    """A quantile prediction or its identity audit is malformed."""


def lightgbm_quantile_repair_contract() -> dict[str, Any]:
    """Return the exact frozen meaning of LightGBM quantile post-processing."""
    return {
        "method": LIGHTGBM_QUANTILE_REPAIR_METHOD,
        "version": 1,
        "nominal_head_levels": {"q05": 0.05, "q50": 0.50, "q95": 0.95},
        "q05_operation": "minimum(raw_q05,raw_q50)",
        "q50_operation": "raw_q50_unchanged",
        "q95_operation": "maximum(raw_q95,raw_q50)",
        "nominal_median_preserved_exactly": True,
    }


def _three_finite_arrays(
    q05: Any, q50: Any, q95: Any, *, one_dimensional: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = (
        np.asarray(q05),
        np.asarray(q50),
        np.asarray(q95),
    )
    if len({array.shape for array in arrays}) != 1:
        raise QuantileIdentityError("quantile heads have different shapes")
    if one_dimensional and arrays[0].ndim != 1:
        raise QuantileIdentityError("crossing audit requires one-dimensional heads")
    if not all(
        np.issubdtype(array.dtype, np.number)
        and not np.issubdtype(array.dtype, np.complexfloating)
        for array in arrays
    ):
        raise QuantileIdentityError("quantile heads are not real numeric arrays")
    if not all(np.isfinite(array).all() for array in arrays):
        raise QuantileIdentityError("quantile heads contain non-finite values")
    return arrays


def repair_lightgbm_quantiles(
    q05: Any, q50: Any, q95: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Order endpoints without ever reassigning the nominal q50 head.

    This is not isotonic regression and does not sort the three values.  It is
    the deterministic map ``(min(q05, q50), q50, max(q95, q50))``.  In
    particular, the returned median is a copy of raw q50 and is exactly equal
    to it, including its floating-point bit pattern.
    """
    raw_q05, raw_q50, raw_q95 = _three_finite_arrays(q05, q50, q95)
    return (
        np.minimum(raw_q05, raw_q50),
        raw_q50.copy(),
        np.maximum(raw_q95, raw_q50),
    )


def raw_quantile_prediction_digest(q05: Any, q50: Any, q95: Any) -> str:
    """Hash nominal heads in fixed head order without repairing or sorting."""
    arrays = _three_finite_arrays(q05, q50, q95, one_dimensional=True)
    digest = hashlib.sha256()
    for name, array in zip(("q05", "q50", "q95"), arrays, strict=True):
        canonical = np.ascontiguousarray(array, dtype="<f8")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(str(canonical.shape).encode("ascii") + b"\0")
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def raw_quantile_crossing_summary(
    q05: Any,
    q50: Any,
    q95: Any,
    *,
    forecast_key_sha256: str,
) -> dict[str, Any]:
    """Summarise crossings before repair for one member and one horizon."""
    raw_q05, raw_q50, raw_q95 = _three_finite_arrays(
        q05, q50, q95, one_dimensional=True
    )
    if not _is_sha256(forecast_key_sha256):
        raise QuantileIdentityError("forecast-key digest is not SHA-256")
    rows = int(len(raw_q50))
    if rows < 1:
        raise QuantileIdentityError("crossing audit cannot be empty")
    lower_gap = np.maximum(raw_q05 - raw_q50, 0.0)
    upper_gap = np.maximum(raw_q50 - raw_q95, 0.0)
    lower_crossing = lower_gap > 0.0
    upper_crossing = upper_gap > 0.0
    any_crossing = lower_crossing | upper_crossing
    crossing_count = int(any_crossing.sum())
    return {
        "rows": rows,
        "forecast_key_sha256": str(forecast_key_sha256),
        "raw_prediction_sha256": raw_quantile_prediction_digest(
            raw_q05, raw_q50, raw_q95
        ),
        "q05_above_q50_count": int(lower_crossing.sum()),
        "q50_above_q95_count": int(upper_crossing.sum()),
        "any_crossing_count": crossing_count,
        "any_crossing_rate": float(crossing_count / rows),
        "maximum_crossing_gap_c": float(
            max(lower_gap.max(initial=0.0), upper_gap.max(initial=0.0))
        ),
    }


def validate_raw_quantile_crossing_summary(value: object) -> None:
    """Fail closed on an internally inconsistent raw-crossing summary."""
    if not isinstance(value, Mapping) or set(value) != RAW_QUANTILE_CROSSING_SUMMARY_FIELDS:
        raise QuantileIdentityError("raw crossing summary schema is not exact")
    try:
        rows = int(value["rows"])
        lower = int(value["q05_above_q50_count"])
        upper = int(value["q50_above_q95_count"])
        crossing = int(value["any_crossing_count"])
        rate = float(value["any_crossing_rate"])
        maximum_gap = float(value["maximum_crossing_gap_c"])
    except (TypeError, ValueError) as exc:
        raise QuantileIdentityError("raw crossing summary is not numeric") from exc
    integer_fields = (
        ("rows", rows),
        ("q05_above_q50_count", lower),
        ("q50_above_q95_count", upper),
        ("any_crossing_count", crossing),
    )
    if any(
        type(value[name]) is not int or value[name] != number
        for name, number in integer_fields
    ):
        raise QuantileIdentityError("raw crossing counts are not exact integers")
    if (
        isinstance(value["any_crossing_rate"], bool)
        or not isinstance(value["any_crossing_rate"], (int, float))
        or isinstance(value["maximum_crossing_gap_c"], bool)
        or not isinstance(value["maximum_crossing_gap_c"], (int, float))
    ):
        raise QuantileIdentityError("raw crossing rate or gap has a wrong type")
    if (
        rows < 1
        or min(lower, upper, crossing) < 0
        or max(lower, upper, crossing) > rows
        or crossing < max(lower, upper)
        or crossing > lower + upper
        or not np.isfinite(rate)
        or rate != crossing / rows
        or not np.isfinite(maximum_gap)
        or maximum_gap < 0.0
        or (crossing == 0) != (maximum_gap == 0.0)
    ):
        raise QuantileIdentityError("raw crossing counts, rate, or gap disagree")
    for field in ("forecast_key_sha256", "raw_prediction_sha256"):
        if not _is_sha256(value[field]):
            raise QuantileIdentityError(f"{field} is not SHA-256")


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str):
        return False
    text = value
    return (
        len(text) == 64
        and all(character in "0123456789abcdef" for character in text)
    )

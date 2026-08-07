"""Frozen CQR + Platt calibration applied at inference time.

Lifted verbatim from the removed ``opening.py`` calibration path
(``_frozen_calibration`` / ``_validate_frozen_calibration_registry``) so the
conventional scorer applies exactly the same calibration transform the
sealed pipeline did.  The 2019-2020 reproduction is the only proof the port
is faithful: G15 in the conventional scorer compares frozen-bundle
predictions against the stored development predictions within tolerance.

Only three external symbols are used: ``conformal.validate_cqr_offset_bundle``,
``conformal.CQRContractError``, and ``probability.PlattCalibrator`` /
``probability.logit`` — all KEEP modules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .conformal import CQRContractError, validate_cqr_offset_bundle
from .probability import PlattCalibrator, logit


class FrozenCalibrationError(RuntimeError):
    """Calibration parameters are missing, stale, or invalid."""


def validate_calibration_registry(
    metadata: Mapping[str, Any],
    station_ids: Sequence[str],
    horizons: Sequence[int],
    *,
    external: bool,
    label: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Validate calibration keys and values without reading confirmation labels."""
    offsets = metadata.get("conformal_offsets")
    calibrators = metadata.get("event_calibrators")
    thresholds = metadata.get("event_thresholds")
    if (
        not isinstance(offsets, Mapping)
        or not isinstance(calibrators, Mapping)
        or not isinstance(thresholds, Mapping)
    ):
        raise FrozenCalibrationError(f"{label} lacks frozen calibration parameters")
    expected_calibrators = {str(int(horizon)) for horizon in horizons}
    if set(calibrators) != expected_calibrators:
        raise FrozenCalibrationError(f"{label} event calibrator registry changed")
    if external:
        if set(thresholds) != {"__pooled__"}:
            raise FrozenCalibrationError(f"{label} external event threshold is not pooled")
        expected_offsets = {f"__pooled__|{int(horizon)}" for horizon in horizons}
    else:
        expected_sites = {str(value) for value in station_ids}
        if set(thresholds) != expected_sites:
            raise FrozenCalibrationError(f"{label} event-threshold site registry changed")
        expected_offsets = {
            f"{site}|{int(horizon)}"
            for site in expected_sites for horizon in horizons
        }
    if set(offsets) != expected_offsets:
        raise FrozenCalibrationError(f"{label} CQR offset registry changed")
    if not np.isfinite(np.asarray(list(thresholds.values()), dtype=float)).all():
        raise FrozenCalibrationError(f"{label} event threshold is non-finite")
    try:
        offset_values = np.asarray(list(offsets.values()), dtype=float)
    except (TypeError, ValueError) as exc:
        raise FrozenCalibrationError(f"{label} CQR offsets are not numeric") from exc
    if not np.isfinite(offset_values).all() or (offset_values < 0.0).any():
        raise FrozenCalibrationError(
            f"{label} contains a non-finite or negative CQR offset"
        )
    try:
        validate_cqr_offset_bundle(
            offsets,
            metadata.get("conformal_policy"),
            metadata.get("conformal_offset_audit"),
        )
    except CQRContractError as exc:
        raise FrozenCalibrationError(
            f"{label} CQR widens-only policy/audit is invalid"
        ) from exc
    for horizon in horizons:
        value = calibrators[str(int(horizon))]
        if not isinstance(value, Mapping) or set(value) != {
            "intercept", "slope", "constant"
        }:
            raise FrozenCalibrationError(f"{label} Platt calibrator is malformed")
        try:
            constant = value.get("constant")
            parameters = np.asarray(
                [float(value["intercept"]), float(value["slope"])], dtype=float
            )
            if constant is not None:
                constant = float(constant)
        except (KeyError, TypeError, ValueError) as exc:
            raise FrozenCalibrationError(f"{label} Platt calibrator is invalid") from exc
        if not np.isfinite(parameters).all() or (
            constant is not None
            and (
                not np.isfinite(constant)
                or not 0.0 < constant < 1.0
                or parameters[1] != 0.0
                or not np.isclose(
                    parameters[0],
                    float(logit(np.asarray([constant]))[0]),
                    rtol=0.0,
                    atol=1e-12,
                )
            )
        ):
            raise FrozenCalibrationError(f"{label} Platt calibrator is invalid")
    return offsets, calibrators, thresholds


def apply_frozen_calibration(
    metadata: Mapping[str, Any],
    station: np.ndarray,
    horizons: Sequence[int],
    q05: np.ndarray,
    q50: np.ndarray,
    q95: np.ndarray,
    raw_probability: np.ndarray,
    *,
    external: bool,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    offsets, calibrators, _ = validate_calibration_registry(
        metadata,
        tuple(str(value) for value in station),
        horizons,
        external=external,
        label=label,
    )
    q05, q50, q95 = (np.asarray(value, dtype=float).copy()
                     for value in (q05, q50, q95))
    probability = np.asarray(raw_probability, dtype=float).copy()
    expected_shape = (len(station), len(horizons))
    if (
        any(value.shape != expected_shape for value in (q05, q50, q95, probability))
        or not all(np.isfinite(value).all() for value in (q05, q50, q95, probability))
        or not ((q05 <= q50).all() and (q50 <= q95).all())
        or not (q05 < q95).all()
        or not ((0.0 <= probability) & (probability <= 1.0)).all()
    ):
        raise FrozenCalibrationError(f"{label} nominal heads are invalid before CQR")
    for column, horizon in enumerate(horizons):
        horizon = int(horizon)
        if external:
            delta = np.full(
                len(station), float(offsets[f"__pooled__|{horizon}"]), dtype=float
            )
        else:
            delta = np.asarray(
                [float(offsets[f"{site}|{horizon}"]) for site in station],
                dtype=float,
            )
        q05[:, column] -= delta
        q95[:, column] += delta
        value = calibrators[str(horizon)]
        constant = value.get("constant")
        calibrator = PlattCalibrator(
            intercept=float(value["intercept"]),
            slope=float(value["slope"]),
            constant=None if constant is None else float(constant),
        )
        probability[:, column] = calibrator.predict(probability[:, column])
    if not (
        np.isfinite(q05).all()
        and np.isfinite(q50).all()
        and np.isfinite(q95).all()
        and np.isfinite(probability).all()
        and (q05 <= q50).all()
        and (q50 <= q95).all()
        and (q05 < q95).all()
        and ((0.0 <= probability) & (probability <= 1.0)).all()
    ):
        raise FrozenCalibrationError(f"{label} calibrated heads are invalid")
    return q05, q50, q95, probability

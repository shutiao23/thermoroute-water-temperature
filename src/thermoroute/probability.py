"""Leakage-safe calibration and climatological references for event forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


EPS = 1e-6
EVENT_REFERENCE_FORMAT = "thermoroute.frozen-seasonal-event-reference.v1"


def _clip_probability(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)


def logit(values: np.ndarray) -> np.ndarray:
    values = _clip_probability(values)
    return np.log(values / (1.0 - values))


@dataclass(frozen=True)
class PlattCalibrator:
    """Logistic calibration on the raw forecast logit.

    ``constant`` is used when a calibration slice contains only one outcome
    class.  This is an explicit, smoothed calibration-period base rate rather
    than a model fitted to test outcomes.
    """

    intercept: float
    slope: float
    constant: float | None = None

    @classmethod
    def fit(cls, probabilities: np.ndarray, outcomes: np.ndarray) -> "PlattCalibrator":
        p = _clip_probability(probabilities)
        y = np.asarray(outcomes, dtype=int)
        ok = np.isfinite(p) & np.isfinite(y)
        p, y = p[ok], y[ok]
        if len(y) == 0:
            raise ValueError("cannot calibrate an empty sample")
        if np.unique(y).size < 2:
            # Jeffreys smoothing prevents exact 0/1 probabilities.
            constant = float((y.sum() + 0.5) / (len(y) + 1.0))
            return cls(intercept=float(logit(np.array([constant]))[0]), slope=0.0,
                       constant=constant)
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        model.fit(logit(p).reshape(-1, 1), y)
        return cls(intercept=float(model.intercept_[0]), slope=float(model.coef_[0, 0]))

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        p = _clip_probability(probabilities)
        if self.constant is not None:
            return np.full_like(p, self.constant, dtype=float)
        z = np.clip(self.intercept + self.slope * logit(p), -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-z))

    def as_dict(self) -> dict[str, float | None]:
        return {"intercept": self.intercept, "slope": self.slope, "constant": self.constant}


def fit_horizon_calibrators(calibration: pd.DataFrame, *,
                            probability_col: str = "p_exceed",
                            outcome_col: str = "event",
                            min_samples: int = 100) -> dict[int, PlattCalibrator]:
    """Fit one calibrator per horizon using calibration rows only."""
    required = {"horizon", probability_col, outcome_col}
    missing = required - set(calibration.columns)
    if missing:
        raise ValueError(f"calibration frame missing columns: {sorted(missing)}")
    calibrators: dict[int, PlattCalibrator] = {}
    for horizon, group in calibration.groupby("horizon"):
        valid = group[[probability_col, outcome_col]].dropna()
        if len(valid) < min_samples:
            raise ValueError(
                f"horizon {horizon} has {len(valid)} calibration rows; need {min_samples}"
            )
        calibrators[int(horizon)] = PlattCalibrator.fit(
            valid[probability_col].to_numpy(float), valid[outcome_col].to_numpy(int)
        )
    return calibrators


def apply_horizon_calibrators(frame: pd.DataFrame,
                              calibrators: Mapping[int, PlattCalibrator], *,
                              probability_col: str = "p_exceed",
                              output_col: str = "p_exceed_calibrated") -> pd.DataFrame:
    """Apply pre-fitted calibrators; missing horizons fail instead of falling back."""
    output = frame.copy()
    output[output_col] = np.nan
    for horizon, group_index in output.groupby("horizon").groups.items():
        if int(horizon) not in calibrators:
            raise KeyError(f"no event calibrator for horizon {horizon}")
        loc = np.asarray(list(group_index))
        values = output.loc[loc, probability_col].to_numpy(float)
        output.loc[loc, output_col] = calibrators[int(horizon)].predict(values)
    return output


@dataclass(frozen=True)
class SeasonalClimatology:
    """Station-by-calendar-month event probability fitted outside evaluation."""

    probabilities: Mapping[tuple[str, int], float]
    station_fallback: Mapping[str, float]
    global_fallback: float

    def predict(self, station: np.ndarray, target_date: np.ndarray) -> np.ndarray:
        dates = pd.to_datetime(target_date)
        result = np.empty(len(station), dtype=float)
        for i, (site, month) in enumerate(zip(station, dates.month)):
            site = str(site)
            result[i] = self.probabilities.get(
                (site, int(month)), self.station_fallback.get(site, self.global_fallback)
            )
        return _clip_probability(result)


def fit_seasonal_climatology(panel: pd.DataFrame, thresholds: Mapping[str, float], *,
                             date_col: str = "DATE", station_col: str = "site_id",
                             target_col: str = "WTEMP", smoothing: float = 2.0
                             ) -> SeasonalClimatology:
    """Fit a smoothed monthly reference from a pre-evaluation panel only."""
    frame = panel[[date_col, station_col, target_col]].dropna().copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame["event"] = [
        float(value > thresholds[str(site)])
        for site, value in zip(frame[station_col], frame[target_col])
    ]
    global_rate = float((frame["event"].sum() + 0.5) / (len(frame) + 1.0))
    station_rate: dict[str, float] = {}
    monthly: dict[tuple[str, int], float] = {}
    for site, group in frame.groupby(station_col):
        site = str(site)
        total = len(group)
        rate = float((group.event.sum() + smoothing * global_rate) / (total + smoothing))
        station_rate[site] = rate
        for month, month_group in group.groupby(group[date_col].dt.month):
            monthly[(site, int(month))] = float(
                (month_group.event.sum() + smoothing * rate) / (len(month_group) + smoothing)
            )
    return SeasonalClimatology(monthly, station_rate, global_rate)


def _smoothed_rate(events: np.ndarray, prior: float, smoothing: float) -> float:
    values = np.asarray(events, dtype=float)
    if values.size == 0:
        return float(prior)
    return float((values.sum() + smoothing * prior) / (len(values) + smoothing))


def fit_frozen_seasonal_event_reference(
    panel: pd.DataFrame,
    thresholds: Mapping[str, float] | float,
    *,
    pooled: bool,
    fit_interval: tuple[str, str] = ("2006-01-01", "2018-12-31"),
    date_col: str = "DATE",
    station_col: str = "site_id",
    target_col: str = "WTEMP",
    smoothing: float = 2.0,
) -> dict[str, object]:
    """Freeze a monthly event reference without reading evaluation outcomes.

    The temporal reference uses station-specific train-q90 thresholds and
    station-by-month rates.  The external warm-start reference uses one pooled
    development threshold and one pooled monthly rate, because an unseen site's
    outcome history is intentionally unavailable while the suite is frozen.
    Every month is materialised; a month with no fitting rows receives its
    predeclared smoothed fallback instead of consulting evaluation data.
    """
    if smoothing <= 0 or not np.isfinite(smoothing):
        raise ValueError("event-reference smoothing must be finite and positive")
    required = {date_col, station_col, target_col}
    missing = required - set(panel)
    if missing:
        raise ValueError(f"event-reference panel is missing: {sorted(missing)}")
    start, end = (pd.Timestamp(value) for value in fit_interval)
    if start > end:
        raise ValueError("event-reference fit interval is reversed")
    frame = panel[[date_col, station_col, target_col]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce")
    frame[station_col] = frame[station_col].astype(str)
    frame = frame[
        frame[date_col].between(start, end)
        & frame[target_col].notna()
        & np.isfinite(frame[target_col].to_numpy(float))
    ].copy()
    if frame.empty:
        raise ValueError("event-reference fitting interval has no finite outcomes")
    frame["month"] = frame[date_col].dt.month.astype(int)

    if pooled:
        if isinstance(thresholds, Mapping):
            if set(thresholds) != {"__pooled__"}:
                raise ValueError("pooled event reference requires only __pooled__")
            threshold = float(thresholds["__pooled__"])
        else:
            threshold = float(thresholds)
        if not np.isfinite(threshold):
            raise ValueError("pooled event threshold is not finite")
        frame["event"] = (frame[target_col].to_numpy(float) > threshold).astype(int)
        global_rate = float((frame.event.sum() + 0.5) / (len(frame) + 1.0))
        monthly = {
            str(month): _smoothed_rate(
                frame.loc[frame.month.eq(month), "event"].to_numpy(float),
                global_rate,
                smoothing,
            )
            for month in range(1, 13)
        }
        reference: dict[str, object] = {
            "format": EVENT_REFERENCE_FORMAT,
            "mode": "pooled_month",
            "threshold_scope": "pooled_development_train_q90",
            "fit_interval": [start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")],
            "smoothing": float(smoothing),
            "global_probability": global_rate,
            "month_probability": monthly,
            "fit_observation_count": int(len(frame)),
        }
        validate_frozen_seasonal_event_reference(reference, pooled=True)
        return reference

    if not isinstance(thresholds, Mapping) or not thresholds:
        raise ValueError("station event reference requires a non-empty threshold map")
    threshold_map = {str(site): float(value) for site, value in thresholds.items()}
    if any(not np.isfinite(value) for value in threshold_map.values()):
        raise ValueError("station event threshold is not finite")
    unknown = sorted(set(frame[station_col]) - set(threshold_map))
    if unknown:
        raise ValueError(f"event-reference panel has unknown sites: {unknown[:5]}")
    frame = frame[frame[station_col].isin(threshold_map)].copy()
    missing_sites = sorted(set(threshold_map) - set(frame[station_col]))
    if missing_sites:
        raise ValueError(f"event-reference sites have no fitting outcomes: {missing_sites[:5]}")
    frame["threshold"] = frame[station_col].map(threshold_map)
    frame["event"] = (
        frame[target_col].to_numpy(float) > frame.threshold.to_numpy(float)
    ).astype(int)
    global_rate = float((frame.event.sum() + 0.5) / (len(frame) + 1.0))
    station_rate: dict[str, float] = {}
    station_month: dict[str, float] = {}
    for site in sorted(threshold_map):
        group = frame[frame[station_col].eq(site)]
        rate = _smoothed_rate(group.event.to_numpy(float), global_rate, smoothing)
        station_rate[site] = rate
        for month in range(1, 13):
            station_month[f"{site}|{month}"] = _smoothed_rate(
                group.loc[group.month.eq(month), "event"].to_numpy(float),
                rate,
                smoothing,
            )
    reference = {
        "format": EVENT_REFERENCE_FORMAT,
        "mode": "station_month",
        "threshold_scope": "station_development_train_q90",
        "fit_interval": [start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")],
        "smoothing": float(smoothing),
        "global_probability": global_rate,
        "station_probability": station_rate,
        "station_month_probability": station_month,
        "fit_observation_count": int(len(frame)),
    }
    validate_frozen_seasonal_event_reference(
        reference, expected_sites=set(threshold_map), pooled=False
    )
    return reference


def validate_frozen_seasonal_event_reference(
    reference: Mapping[str, object],
    *,
    expected_sites: set[str] | None = None,
    pooled: bool | None = None,
) -> None:
    """Reject malformed or incomplete frozen event-reference metadata."""
    if reference.get("format") != EVENT_REFERENCE_FORMAT:
        raise ValueError("unsupported seasonal event-reference format")
    mode = reference.get("mode")
    if mode not in {"station_month", "pooled_month"}:
        raise ValueError("seasonal event-reference mode is invalid")
    if pooled is not None and (mode == "pooled_month") != pooled:
        raise ValueError("seasonal event-reference pooling mode changed")
    interval = reference.get("fit_interval")
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError("seasonal event-reference fit interval is malformed")
    try:
        start, end = (pd.Timestamp(str(value)) for value in interval)
        smoothing = float(cast(Any, reference["smoothing"]))
        global_probability = float(cast(Any, reference["global_probability"]))
        count = int(cast(Any, reference["fit_observation_count"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("seasonal event-reference scalar is malformed") from exc
    if start > end or smoothing <= 0 or count <= 0:
        raise ValueError("seasonal event-reference fit declaration is invalid")

    def validate_probability_map(value: object, expected: set[str], label: str) -> None:
        if not isinstance(value, Mapping) or set(map(str, value)) != expected:
            raise ValueError(f"seasonal event-reference {label} registry changed")
        try:
            probabilities = np.asarray([float(item) for item in value.values()])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"seasonal event-reference {label} is nonnumeric") from exc
        if not (np.isfinite(probabilities).all()
                and ((0.0 < probabilities) & (probabilities < 1.0)).all()):
            raise ValueError(f"seasonal event-reference {label} is outside (0,1)")

    if not np.isfinite(global_probability) or not 0.0 < global_probability < 1.0:
        raise ValueError("seasonal event-reference global probability is invalid")
    months = {str(month) for month in range(1, 13)}
    if mode == "pooled_month":
        validate_probability_map(reference.get("month_probability"), months, "months")
        if "station_probability" in reference or "station_month_probability" in reference:
            raise ValueError("pooled seasonal event reference contains station adaptation")
        return
    sites_value = reference.get("station_probability")
    if not isinstance(sites_value, Mapping):
        raise ValueError("station seasonal event reference lacks station probabilities")
    sites = set(map(str, sites_value))
    if expected_sites is not None and sites != set(map(str, expected_sites)):
        raise ValueError("seasonal event-reference station registry changed")
    validate_probability_map(sites_value, sites, "stations")
    expected_station_months = {f"{site}|{month}" for site in sites for month in range(1, 13)}
    validate_probability_map(
        reference.get("station_month_probability"),
        expected_station_months,
        "station-months",
    )
    if "month_probability" in reference:
        raise ValueError("station seasonal event reference contains pooled months")


def predict_frozen_seasonal_event_reference(
    reference: Mapping[str, object],
    stations: np.ndarray,
    target_dates: np.ndarray,
) -> np.ndarray:
    """Apply only the already-frozen monthly probabilities."""
    validate_frozen_seasonal_event_reference(reference)
    sites = np.asarray(stations).astype(str)
    dates = pd.to_datetime(target_dates)
    if len(sites) != len(dates):
        raise ValueError("event-reference station/date lengths differ")
    global_probability = float(cast(Any, reference["global_probability"]))
    output = np.empty(len(sites), dtype=float)
    if reference["mode"] == "pooled_month":
        monthly = reference["month_probability"]
        assert isinstance(monthly, Mapping)
        for index, month in enumerate(dates.month):
            output[index] = float(monthly.get(str(int(month)), global_probability))
        return _clip_probability(output)
    station_probability = reference["station_probability"]
    station_month = reference["station_month_probability"]
    assert isinstance(station_probability, Mapping) and isinstance(station_month, Mapping)
    for index, (site, month) in enumerate(zip(sites, dates.month)):
        if site not in station_probability:
            raise KeyError(f"seasonal event reference lacks station {site}")
        output[index] = float(station_month.get(
            f"{site}|{int(month)}", station_probability.get(site, global_probability)
        ))
    return _clip_probability(output)


def expected_calibration_error(outcomes: np.ndarray, probabilities: np.ndarray,
                               n_bins: int = 10) -> float:
    y = np.asarray(outcomes, dtype=float)
    p = _clip_probability(probabilities)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    value = 0.0
    for index in range(n_bins):
        selected = bins == index
        if selected.any():
            value += selected.mean() * abs(float(y[selected].mean() - p[selected].mean()))
    return float(value)


def calibration_intercept_slope(outcomes: np.ndarray,
                                probabilities: np.ndarray) -> tuple[float, float]:
    """Diagnostic logistic calibration intercept/slope on evaluation outcomes."""
    y = np.asarray(outcomes, dtype=int)
    p = _clip_probability(probabilities)
    if np.unique(y).size < 2:
        return float("nan"), float("nan")
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(logit(p).reshape(-1, 1), y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def ensemble_prediction_frame(predictions: pd.DataFrame, model: str) -> pd.DataFrame:
    """Average seed members on a strict sample identity.

    Quantiles are member-wise averages and are therefore an engineering ensemble
    summary, not quantiles of the mixture distribution.  The distinction is kept
    explicit in reports.
    """
    subset = predictions[predictions["model"] == model].copy()
    if subset.empty:
        return subset
    keys = ["site_id", "horizon", "issue_date", "target_date", "split"]
    for key in keys:
        if key not in subset:
            raise ValueError(f"prediction frame missing identity column {key}")
    aggregation = {
        "y_true": ("y_true", "first"),
        "y_pred": ("y_pred", "mean"),
    }
    for name in ("q05", "q50", "q95", "p_exceed"):
        if name in subset:
            aggregation[name] = (name, "mean")
    result = subset.groupby(keys, as_index=False).agg(**aggregation)
    if result.duplicated(keys).any():  # pragma: no cover - groupby should prevent this
        raise AssertionError("ensemble prediction keys are not unique")
    return result


def calibrated_event_frame(predictions: pd.DataFrame, model: str, *,
                           thresholds: Mapping[str, float],
                           climatology: SeasonalClimatology,
                           calibration_split: str = "calib",
                           evaluation_split: str = "test",
                           min_calibration_samples: int = 100
                           ) -> tuple[pd.DataFrame, dict[int, PlattCalibrator]]:
    """Calibrate a model's event head on one split and evaluate another."""
    ensemble = ensemble_prediction_frame(predictions, model)
    required = {"p_exceed", "y_true", "site_id", "target_date", "split", "horizon"}
    missing = required - set(ensemble)
    if missing:
        raise ValueError(f"event predictions missing columns: {sorted(missing)}")
    ensemble["threshold"] = ensemble.site_id.astype(str).map(thresholds)
    if ensemble["threshold"].isna().any():
        missing_sites = sorted(ensemble.loc[ensemble.threshold.isna(), "site_id"].unique())
        raise KeyError(f"missing event thresholds for sites: {missing_sites[:5]}")
    ensemble["event"] = (ensemble.y_true > ensemble.threshold).astype(int)
    calibration = ensemble[
        (ensemble.split == calibration_split) & ensemble.p_exceed.notna()
    ].copy()
    calibrators = fit_horizon_calibrators(
        calibration,
        probability_col="p_exceed",
        outcome_col="event",
        min_samples=min_calibration_samples,
    )
    evaluation = ensemble[
        (ensemble.split == evaluation_split) & ensemble.p_exceed.notna()
    ].copy()
    evaluation = apply_horizon_calibrators(evaluation, calibrators)
    evaluation["p_reference"] = climatology.predict(
        evaluation.site_id.astype(str).to_numpy(), evaluation.target_date.to_numpy()
    )
    return evaluation, calibrators

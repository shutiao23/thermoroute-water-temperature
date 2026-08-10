"""Transparent forced thermal-response reference model.

The reference is intentionally small and auditable.  For station ``i`` it
advances daily water temperature in degrees Celsius as

``T_next = T_current + alpha_i * (T_equilibrium(F_next) - T_current)``.

``alpha_i`` is constrained to ``(0, 1]``.  The equilibrium temperature is a
shared linear function of five atmospheric fields in their frozen physical
units.  Air temperature, shortwave radiation, and humidity have non-negative
effects; wind and precipitation have non-positive effects.  This is a
directionally constrained response reference, not a complete heat-budget
model and not an implementation of air2stream.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

MODEL_NAME = "TransparentForcedThermalResponse"
SCHEMA_VERSION = "thermoroute.forced-hybrid.v1"
EQUATION = "T_next=T_current+alpha_i*(T_equilibrium(F_next)-T_current)"

TRAIN_START = pd.Timestamp("2006-01-01")
TRAIN_END = pd.Timestamp("2015-12-31")
VALIDATION_START = pd.Timestamp("2016-01-01")
VALIDATION_END = pd.Timestamp("2017-12-31")

CHANNEL_F0 = "F0"
CHANNEL_F2A_TEMPERATURE_ONLY = "F2a_temperature_only"
CHANNEL_F3_TEMPERATURE_ONLY = "F3_temperature_only"
CHANNEL_F3_FULL = "F3_full"
LEGAL_CHANNELS = (
    CHANNEL_F0,
    CHANNEL_F2A_TEMPERATURE_ONLY,
    CHANNEL_F3_TEMPERATURE_ONLY,
    CHANNEL_F3_FULL,
)
LEGAL_HORIZONS = (1, 3, 7)
TEMPERATURE_ONLY_CHANNELS = (
    CHANNEL_F2A_TEMPERATURE_ONLY,
    CHANNEL_F3_TEMPERATURE_ONLY,
)

ATMOSPHERIC_VARIABLES = ("TEMP", "DH", "RHMEAN", "WDSP", "PRCP")
FORCING_UNITS = MappingProxyType(
    {
        "TEMP": "degC",
        "DH": "W m-2 daylight-mean",
        "RHMEAN": "percent proxy",
        "WDSP": "m s-1",
        "PRCP": "mm day-1",
    }
)
FUTURE_DISCHARGE_ENABLED_BY_DEFAULT = False

# Numerical calibration floor for the mathematically open lower bound.  Issued
# models validate the exact interval 0 < alpha <= 1; fitted values are clipped
# away from zero so the equilibrium coefficients remain identifiable.
ALPHA_MIN = 1.0e-6
ALPHA_MAX = 1.0
DEFAULT_INITIAL_ALPHA = 0.20
DEFAULT_MAX_ITERATIONS = 100
DEFAULT_TOLERANCE = 1.0e-10
MIN_TRAINING_PAIRS_PER_STATION = 30
CALIBRATION_METHOD = "station_equal_alternating_bounded_least_squares_v1"
CALIBRATION_WEIGHTING = "equal_station_total_weight_within_training_complete_pairs"


@dataclass(frozen=True)
class ForcingTerm:
    """One frozen linear equilibrium-temperature term."""

    variable: str
    input_unit: str
    reference_increment: float
    coefficient_unit: str
    lower_bound: float
    upper_bound: float
    physical_direction: str
    plausible_minimum: float
    plausible_maximum: float

    def payload(self) -> dict[str, object]:
        return {
            "variable": self.variable,
            "input_unit": self.input_unit,
            "reference_increment": self.reference_increment,
            "coefficient_unit": self.coefficient_unit,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "physical_direction": self.physical_direction,
            "plausible_minimum": self.plausible_minimum,
            "plausible_maximum": self.plausible_maximum,
        }


# Coefficients are equilibrium-temperature changes per declared increment.
# Bounds are deliberately broad; their primary purpose is physical direction,
# while still preventing an unconstrained fit from changing the model meaning.
FORCING_TERMS = (
    ForcingTerm(
        "TEMP",
        FORCING_UNITS["TEMP"],
        1.0,
        "degC equilibrium per 1 degC air temperature",
        0.0,
        2.0,
        "nonnegative",
        -80.0,
        60.0,
    ),
    ForcingTerm(
        "DH",
        FORCING_UNITS["DH"],
        100.0,
        "degC equilibrium per 100 W m-2 daylight-mean shortwave",
        0.0,
        10.0,
        "nonnegative",
        0.0,
        1_500.0,
    ),
    ForcingTerm(
        "RHMEAN",
        FORCING_UNITS["RHMEAN"],
        10.0,
        "degC equilibrium per 10 percentage-point humidity-proxy increase",
        0.0,
        5.0,
        "nonnegative",
        0.0,
        100.0,
    ),
    ForcingTerm(
        "WDSP",
        FORCING_UNITS["WDSP"],
        1.0,
        "degC equilibrium per 1 m s-1 wind-speed increase",
        -10.0,
        0.0,
        "nonpositive",
        0.0,
        100.0,
    ),
    ForcingTerm(
        "PRCP",
        FORCING_UNITS["PRCP"],
        10.0,
        "degC equilibrium per 10 mm day-1 precipitation increase",
        -10.0,
        0.0,
        "nonpositive",
        0.0,
        1_000.0,
    ),
)
TERM_BY_VARIABLE = MappingProxyType({term.variable: term for term in FORCING_TERMS})
COEFFICIENT_NAMES = ("intercept", *ATMOSPHERIC_VARIABLES)
COEFFICIENT_BOUNDS = MappingProxyType(
    {
        "intercept": (-50.0, 50.0),
        **{term.variable: (term.lower_bound, term.upper_bound) for term in FORCING_TERMS},
    }
)

PREDICTION_COLUMNS = (
    "site_id",
    "issue_date",
    "target_date",
    "lead_days",
    "channel",
    "temperature_equilibrium",
    "y_pred",
)
FORMAL_KEY_COLUMNS = ("key_id", "site_id", "issue_date", "target_date", "lead_days")


class ForcedHybridContractError(ValueError):
    """Raised when inputs violate the frozen forced-hybrid contract."""


def _exact_float(value: object, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ForcedHybridContractError(f"{label} must be one finite real number")
    converted = float(value)
    if not np.isfinite(converted):
        raise ForcedHybridContractError(f"{label} must be one finite real number")
    return converted


def _exact_positive_integer(value: object, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ForcedHybridContractError(f"{label} must be one exact positive integer")
    converted = int(value)
    if converted < 1:
        raise ForcedHybridContractError(f"{label} must be one exact positive integer")
    return converted


def _canonical_date(value: object, label: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is not None or timestamp != timestamp.normalize():
        raise ForcedHybridContractError(f"{label} must be a timezone-naive midnight date")
    return timestamp


def _canonical_site(value: object, label: str = "site_id") -> str:
    if type(value) is not str or not value or value.strip() != value or not value.isascii():
        raise ForcedHybridContractError(f"{label} must be one non-empty canonical string")
    return value


def validate_forcing_units(units: Mapping[str, str]) -> None:
    """Require the exact physical-unit declaration for all atmospheric fields."""

    if type(units) not in (dict, MappingProxyType) and not isinstance(units, Mapping):
        raise ForcedHybridContractError("forcing units must be a mapping")
    if dict(units) != dict(FORCING_UNITS):
        raise ForcedHybridContractError(
            f"forcing units must equal the frozen contract: {dict(FORCING_UNITS)}"
        )


def _forcing_frame(
    forcing: Mapping[str, object] | pd.DataFrame,
    *,
    units: Mapping[str, str],
) -> pd.DataFrame:
    validate_forcing_units(units)
    if isinstance(forcing, pd.DataFrame):
        frame = forcing.copy(deep=True)
    elif isinstance(forcing, Mapping):
        frame = pd.DataFrame([dict(forcing)])
    else:
        raise ForcedHybridContractError("forcing must be a mapping or DataFrame")
    if tuple(frame.columns) != ATMOSPHERIC_VARIABLES:
        raise ForcedHybridContractError(
            f"forcing schema/order must be exactly {ATMOSPHERIC_VARIABLES}"
        )
    if frame.empty:
        raise ForcedHybridContractError("forcing must contain at least one row")
    for term in FORCING_TERMS:
        series = frame[term.variable]
        if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(
            series.dtype
        ):
            raise ForcedHybridContractError(f"{term.variable} forcing must be numeric")
        values = series.to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(values).all():
            raise ForcedHybridContractError(f"{term.variable} forcing must be finite")
        if (values < term.plausible_minimum).any() or (values > term.plausible_maximum).any():
            raise ForcedHybridContractError(
                f"{term.variable} forcing is outside its frozen physical-unit range"
            )
        frame[term.variable] = values
    return frame.reset_index(drop=True)


def _scaled_design(frame: pd.DataFrame) -> np.ndarray:
    matrix = np.ones((len(frame), len(COEFFICIENT_NAMES)), dtype=np.float64)
    for index, term in enumerate(FORCING_TERMS, start=1):
        matrix[:, index] = (
            frame[term.variable].to_numpy(dtype=np.float64, copy=False) / term.reference_increment
        )
    return matrix


def forecast_key_id(
    site_id: str,
    issue_date: object,
    target_date: object,
    lead_days: int,
) -> str:
    """Return the canonical temporal-known-site forecast identity hash."""

    site = _canonical_site(site_id)
    issue = _canonical_date(issue_date, "issue_date")
    target = _canonical_date(target_date, "target_date")
    lead = _exact_positive_integer(lead_days, "lead_days")
    if target != issue + pd.Timedelta(days=lead):
        raise ForcedHybridContractError("target_date must equal issue_date + lead_days")
    payload = f"temporal|known_site|{site}|{issue:%Y-%m-%d}|{target:%Y-%m-%d}|{lead}"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _reject_duplicate_json_pairs(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise ForcedHybridContractError(f"serialized model repeats JSON key {key!r}")
        document[key] = value
    return document


@dataclass(frozen=True)
class ForcedHybridModel:
    """Fitted transparent thermal-response parameters and rollout methods."""

    coefficients: Mapping[str, float]
    alpha_by_station: Mapping[str, float]
    training_pair_count_by_station: Mapping[str, int]
    iterations: int
    converged: bool
    future_discharge_enabled: bool = FUTURE_DISCHARGE_ENABLED_BY_DEFAULT
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    tolerance: float = DEFAULT_TOLERANCE

    def __post_init__(self) -> None:
        if len(self.coefficients) != len(COEFFICIENT_NAMES) or set(self.coefficients) != set(
            COEFFICIENT_NAMES
        ):
            raise ForcedHybridContractError(
                f"coefficient schema must be exactly {COEFFICIENT_NAMES}"
            )
        checked_coefficients: dict[str, float] = {}
        for name in COEFFICIENT_NAMES:
            value = _exact_float(self.coefficients[name], f"coefficient {name}")
            lower, upper = COEFFICIENT_BOUNDS[name]
            if value < lower or value > upper:
                raise ForcedHybridContractError(
                    f"coefficient {name}={value} is outside [{lower}, {upper}]"
                )
            checked_coefficients[name] = value

        if not self.alpha_by_station:
            raise ForcedHybridContractError("alpha registry must not be empty")
        checked_alpha: dict[str, float] = {}
        for station, raw_alpha in sorted(self.alpha_by_station.items()):
            site = _canonical_site(station, "alpha station")
            alpha = _exact_float(raw_alpha, f"alpha {site}")
            if not 0.0 < alpha <= ALPHA_MAX:
                raise ForcedHybridContractError(
                    f"alpha {site} must be in the mathematical interval (0, {ALPHA_MAX}]"
                )
            checked_alpha[site] = alpha

        if set(self.training_pair_count_by_station) != set(checked_alpha):
            raise ForcedHybridContractError("training-pair registry must equal alpha registry")
        checked_counts: dict[str, int] = {}
        for site in checked_alpha:
            count = _exact_positive_integer(
                self.training_pair_count_by_station[site], f"training pairs {site}"
            )
            if count < MIN_TRAINING_PAIRS_PER_STATION:
                raise ForcedHybridContractError(
                    f"station {site} has fewer than {MIN_TRAINING_PAIRS_PER_STATION} train pairs"
                )
            checked_counts[site] = count

        iteration_count = _exact_positive_integer(self.iterations, "iterations")
        iterations_limit = _exact_positive_integer(self.max_iterations, "max_iterations")
        tolerance = _exact_float(self.tolerance, "tolerance")
        if iteration_count > iterations_limit:
            raise ForcedHybridContractError("iterations cannot exceed max_iterations")
        if tolerance <= 0.0:
            raise ForcedHybridContractError("tolerance must be positive")
        if type(self.converged) is not bool:
            raise ForcedHybridContractError("converged must be an exact bool")
        if not self.converged:
            raise ForcedHybridContractError("only a converged calibration may issue a model")
        if type(self.future_discharge_enabled) is not bool:
            raise ForcedHybridContractError("future_discharge_enabled must be an exact bool")
        if self.future_discharge_enabled:
            raise ForcedHybridContractError(
                "future discharge is a separate, currently disabled extension"
            )

        object.__setattr__(self, "coefficients", MappingProxyType(checked_coefficients))
        object.__setattr__(self, "alpha_by_station", MappingProxyType(checked_alpha))
        object.__setattr__(
            self,
            "training_pair_count_by_station",
            MappingProxyType(checked_counts),
        )
        object.__setattr__(self, "iterations", iteration_count)
        object.__setattr__(self, "max_iterations", iterations_limit)
        object.__setattr__(self, "tolerance", tolerance)

    @property
    def training_pair_count(self) -> int:
        return int(sum(self.training_pair_count_by_station.values()))

    @classmethod
    def fit(
        cls,
        panel: pd.DataFrame,
        *,
        forcing_units: Mapping[str, str],
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> ForcedHybridModel:
        """Fit only on complete consecutive pairs within 2006--2015.

        Validation dates are never used for coefficients, alpha values,
        stopping, or eligibility.  The fixed 2016--2017 window is metadata for
        downstream validation, not an optimization input.
        """

        validate_forcing_units(forcing_units)
        iterations_limit = _exact_positive_integer(max_iterations, "max_iterations")
        convergence_tolerance = _exact_float(tolerance, "tolerance")
        if convergence_tolerance <= 0.0:
            raise ForcedHybridContractError("tolerance must be positive")
        required = ("DATE", "site_id", "WTEMP", *ATMOSPHERIC_VARIABLES)
        missing = [column for column in required if column not in panel.columns]
        if missing:
            raise ForcedHybridContractError(f"fit panel lacks columns: {missing}")
        dates = pd.to_datetime(panel["DATE"], errors="coerce")
        if dates.isna().any() or getattr(dates.dt, "tz", None) is not None:
            raise ForcedHybridContractError("fit panel DATE must be timezone-naive and finite")
        if not dates.eq(dates.dt.normalize()).all():
            raise ForcedHybridContractError("fit panel DATE must contain midnight dates")
        sites = panel["site_id"]
        if (
            sites.isna().any()
            or not sites.map(lambda value: type(value) is str and bool(value)).all()
        ):
            raise ForcedHybridContractError("fit panel site_id must contain non-empty strings")
        if panel.assign(DATE=dates).duplicated(["site_id", "DATE"]).any():
            raise ForcedHybridContractError("fit panel duplicates a station-day")

        # Select the training window before converting or inspecting any
        # forcing/outcome values.  Later dates cannot influence calibration.
        calendar = panel.loc[dates.between(TRAIN_START, TRAIN_END), required].copy()
        calendar["DATE"] = dates.loc[calendar.index].to_numpy(copy=True)
        if calendar.empty:
            raise ForcedHybridContractError("fit panel contains no 2006--2015 rows")
        for column in ("WTEMP", *ATMOSPHERIC_VARIABLES):
            if not pd.api.types.is_numeric_dtype(
                calendar[column].dtype
            ) or pd.api.types.is_bool_dtype(calendar[column].dtype):
                raise ForcedHybridContractError(f"training {column} must be numeric")
            calendar[column] = calendar[column].to_numpy(dtype=np.float64, copy=True)

        current = calendar[["site_id", "DATE", "WTEMP"]].rename(
            columns={"DATE": "issue_date", "WTEMP": "temperature_current"}
        )
        current["target_date"] = current["issue_date"] + pd.Timedelta(days=1)
        following = calendar[["site_id", "DATE", "WTEMP", *ATMOSPHERIC_VARIABLES]].rename(
            columns={"DATE": "target_date", "WTEMP": "temperature_next"}
        )
        pairs = current.merge(
            following,
            on=["site_id", "target_date"],
            how="inner",
            sort=False,
            validate="one_to_one",
        )
        pairs = pairs.loc[pairs["target_date"].le(TRAIN_END)].copy()
        numeric = ("temperature_current", "temperature_next", *ATMOSPHERIC_VARIABLES)
        complete = np.isfinite(pairs[list(numeric)].to_numpy(dtype=np.float64)).all(axis=1)
        pairs = (
            pairs.loc[complete]
            .sort_values(["site_id", "issue_date"], kind="mergesort")
            .reset_index(drop=True)
        )
        if pairs.empty:
            raise ForcedHybridContractError("fit panel has no complete consecutive training pairs")

        forcing = _forcing_frame(pairs[list(ATMOSPHERIC_VARIABLES)], units=forcing_units)
        design = _scaled_design(forcing)
        # Eligibility is defined by the declared training panel, not by the
        # complete-case table.  A station with zero usable pairs must fail,
        # rather than silently disappearing from the fitted alpha registry.
        station_values = tuple(sorted(set(calendar["site_id"])))
        station_index = {station: index for index, station in enumerate(station_values)}
        row_station = np.asarray([station_index[site] for site in pairs["site_id"]], dtype=np.int64)
        counts = np.bincount(row_station, minlength=len(station_values))
        sparse = [
            station_values[index]
            for index, count in enumerate(counts)
            if int(count) < MIN_TRAINING_PAIRS_PER_STATION
        ]
        if sparse:
            raise ForcedHybridContractError(
                "stations lack the minimum complete training pairs: " + ", ".join(sparse[:5])
            )
        root_station_weight = np.sqrt(1.0 / counts[row_station].astype(np.float64))

        current_values = pairs["temperature_current"].to_numpy(dtype=np.float64)
        next_values = pairs["temperature_next"].to_numpy(dtype=np.float64)
        delta = next_values - current_values
        alphas = np.full(len(station_values), DEFAULT_INITIAL_ALPHA, dtype=np.float64)
        coefficient_lower = np.asarray(
            [COEFFICIENT_BOUNDS[name][0] for name in COEFFICIENT_NAMES], dtype=np.float64
        )
        coefficient_upper = np.asarray(
            [COEFFICIENT_BOUNDS[name][1] for name in COEFFICIENT_NAMES], dtype=np.float64
        )
        coefficients = np.zeros(len(COEFFICIENT_NAMES), dtype=np.float64)
        converged = False
        iteration = 0
        for iteration in range(1, iterations_limit + 1):
            previous_alpha = alphas.copy()
            previous_coefficients = coefficients.copy()
            row_alpha = alphas[row_station]
            response = delta + row_alpha * current_values
            result = lsq_linear(
                root_station_weight[:, None] * row_alpha[:, None] * design,
                root_station_weight * response,
                bounds=(coefficient_lower, coefficient_upper),
                method="trf",
                lsmr_tol="auto",
                max_iter=500,
            )
            if not result.success or not np.isfinite(result.x).all():
                raise ForcedHybridContractError(
                    f"bounded equilibrium fit failed: status={result.status} {result.message}"
                )
            coefficients = result.x.astype(np.float64, copy=True)
            gap = design @ coefficients - current_values
            for station_number in range(len(station_values)):
                selected = row_station == station_number
                station_gap = gap[selected]
                denominator = float(np.dot(station_gap, station_gap))
                if not np.isfinite(denominator) or denominator <= 1.0e-12:
                    raise ForcedHybridContractError(
                        f"station {station_values[station_number]} has no alpha-identifying energy"
                    )
                estimate = float(np.dot(station_gap, delta[selected]) / denominator)
                alphas[station_number] = float(np.clip(estimate, ALPHA_MIN, ALPHA_MAX))

            change = max(
                float(np.max(np.abs(alphas - previous_alpha))),
                float(np.max(np.abs(coefficients - previous_coefficients))),
            )
            if change <= convergence_tolerance:
                converged = True
                break

        if not converged:
            raise ForcedHybridContractError(
                f"alternating bounded calibration did not converge in {iterations_limit} iterations"
            )

        coefficient_mapping = {
            name: float(coefficients[index]) for index, name in enumerate(COEFFICIENT_NAMES)
        }
        alpha_mapping = {
            station: float(alphas[index]) for index, station in enumerate(station_values)
        }
        count_mapping = {
            station: int(counts[index]) for index, station in enumerate(station_values)
        }
        return cls(
            coefficients=coefficient_mapping,
            alpha_by_station=alpha_mapping,
            training_pair_count_by_station=count_mapping,
            iterations=iteration,
            converged=converged,
            max_iterations=iterations_limit,
            tolerance=convergence_tolerance,
        )

    def equilibrium_temperature(
        self,
        forcing: Mapping[str, object] | pd.DataFrame,
        *,
        forcing_units: Mapping[str, str],
    ) -> np.ndarray:
        """Evaluate equilibrium temperature in degrees Celsius."""

        frame = _forcing_frame(forcing, units=forcing_units)
        coefficients = np.asarray(
            [self.coefficients[name] for name in COEFFICIENT_NAMES], dtype=np.float64
        )
        return _scaled_design(frame) @ coefficients

    def step(
        self,
        station: str,
        temperature_current: float,
        forcing_next: Mapping[str, object],
        *,
        forcing_units: Mapping[str, str],
    ) -> float:
        """Advance exactly one day using the station's bounded response rate."""

        site = _canonical_site(station)
        if site not in self.alpha_by_station:
            raise ForcedHybridContractError(f"station {site} has no fitted alpha")
        current = _exact_float(temperature_current, "temperature_current")
        equilibrium = float(
            self.equilibrium_temperature(forcing_next, forcing_units=forcing_units)[0]
        )
        alpha = self.alpha_by_station[site]
        return current + alpha * (equilibrium - current)

    def _forcing_path(
        self,
        *,
        issue_date: pd.Timestamp,
        issue_forcing: Mapping[str, object],
        future_forcing: pd.DataFrame | None,
        horizon: int,
        channel: str,
        forcing_units: Mapping[str, str],
    ) -> pd.DataFrame:
        if channel not in LEGAL_CHANNELS:
            raise ForcedHybridContractError(f"unsupported forcing channel {channel!r}")
        issue = _forcing_frame(issue_forcing, units=forcing_units)
        issue_values = issue.iloc[0]
        if channel == CHANNEL_F0:
            if future_forcing is not None:
                raise ForcedHybridContractError("F0 forbids a future-forcing payload")
            path = pd.DataFrame(
                {
                    variable: np.full(horizon, issue_values[variable], dtype=np.float64)
                    for variable in ATMOSPHERIC_VARIABLES
                }
            )
            path.insert(0, "valid_date", pd.date_range(issue_date, periods=horizon + 1)[1:])
            path.insert(0, "step", np.arange(1, horizon + 1, dtype=np.int16))
            return path

        if not isinstance(future_forcing, pd.DataFrame):
            raise ForcedHybridContractError(f"{channel} requires a future-forcing DataFrame")
        future = future_forcing.copy(deep=True).reset_index(drop=True)
        target_day_only = channel in TEMPERATURE_ONLY_CHANNELS
        future_variables = ("TEMP",) if target_day_only else ATMOSPHERIC_VARIABLES
        expected_columns = ("step", "valid_date", *future_variables)
        if tuple(future.columns) != expected_columns:
            raise ForcedHybridContractError(
                f"{channel} future schema/order must be exactly {expected_columns}; "
                "future discharge is never bundled with atmospheric forcing"
            )
        expected_row_count = 1 if target_day_only else horizon
        if len(future) != expected_row_count:
            if target_day_only:
                raise ForcedHybridContractError(
                    f"{channel} accepts exactly one target-day value; intermediate "
                    "future temperatures are not acquired and must not be reconstructed"
                )
            raise ForcedHybridContractError(f"{channel} must provide exactly {horizon} steps")
        if not pd.api.types.is_integer_dtype(future["step"].dtype) or pd.api.types.is_bool_dtype(
            future["step"].dtype
        ):
            raise ForcedHybridContractError("future step must have an exact integer dtype")
        steps = future["step"].to_numpy(dtype=np.int64, copy=True)
        expected_steps = (
            np.asarray([horizon], dtype=np.int64)
            if target_day_only
            else np.arange(1, horizon + 1, dtype=np.int64)
        )
        if not np.array_equal(steps, expected_steps):
            raise ForcedHybridContractError("future steps differ from the exact channel grid")
        valid_dates = pd.to_datetime(future["valid_date"], errors="coerce")
        expected_dates = pd.Series(
            [issue_date + pd.Timedelta(days=int(step)) for step in expected_steps]
        )
        if (
            valid_dates.isna().any()
            or getattr(valid_dates.dt, "tz", None) is not None
            or not valid_dates.reset_index(drop=True).equals(expected_dates)
        ):
            raise ForcedHybridContractError(
                "future valid_date must equal issue_date + step on the exact daily grid"
            )

        materialized = pd.DataFrame(
            {
                variable: np.full(horizon, issue_values[variable], dtype=np.float64)
                for variable in ATMOSPHERIC_VARIABLES
            }
        )
        if target_day_only:
            # The frozen F2a acquisition has only t+h TEMP.  Both F2a and its
            # matched F3-temperature oracle therefore persist issue-day TEMP
            # at intermediate steps and replace TEMP only at t+h.  No
            # interpolation or hidden trajectory reconstruction is permitted.
            materialized.loc[horizon - 1, "TEMP"] = future.loc[0, "TEMP"]
        else:
            for variable in future_variables:
                materialized[variable] = future[variable].to_numpy(copy=True)
        checked = _forcing_frame(materialized, units=forcing_units)
        checked.insert(0, "valid_date", pd.date_range(issue_date, periods=horizon + 1)[1:])
        checked.insert(0, "step", np.arange(1, horizon + 1, dtype=np.int16))
        return checked

    def rollout(
        self,
        *,
        station: str,
        issue_date: object,
        temperature_current: float,
        issue_forcing: Mapping[str, object],
        future_forcing: pd.DataFrame | None,
        horizon: int,
        channel: str,
        forcing_units: Mapping[str, str],
        future_discharge: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Perform a recurrent multi-day rollout on one exact forcing channel."""

        site = _canonical_site(station)
        issue = _canonical_date(issue_date, "issue_date")
        lead_count = _exact_positive_integer(horizon, "horizon")
        if lead_count not in LEGAL_HORIZONS:
            raise ForcedHybridContractError(
                f"horizon must be one exact frozen value in {LEGAL_HORIZONS}"
            )
        if future_discharge is not None:
            raise ForcedHybridContractError(
                "future discharge is separate from atmospheric channels and disabled by default"
            )
        path = self._forcing_path(
            issue_date=issue,
            issue_forcing=issue_forcing,
            future_forcing=future_forcing,
            horizon=lead_count,
            channel=channel,
            forcing_units=forcing_units,
        )
        current = _exact_float(temperature_current, "temperature_current")
        rows: list[dict[str, object]] = []
        for row in path.itertuples(index=False):
            forcing_row = {variable: getattr(row, variable) for variable in ATMOSPHERIC_VARIABLES}
            equilibrium = float(
                self.equilibrium_temperature(forcing_row, forcing_units=forcing_units)[0]
            )
            current = self.step(
                site,
                current,
                forcing_row,
                forcing_units=forcing_units,
            )
            rows.append(
                {
                    "site_id": site,
                    "issue_date": issue,
                    "target_date": row.valid_date,
                    "lead_days": np.int16(row.step),
                    "channel": channel,
                    "temperature_equilibrium": equilibrium,
                    "y_pred": current,
                }
            )
        result = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
        if tuple(result.columns) != PREDICTION_COLUMNS:
            raise AssertionError("internal prediction schema changed")
        return result

    def forecast_endpoint(
        self,
        *,
        station: str,
        issue_date: object,
        temperature_current: float,
        issue_forcing: Mapping[str, object],
        future_forcing: pd.DataFrame | None,
        horizon: int,
        channel: str,
        forcing_units: Mapping[str, str],
        future_discharge: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Return only the ``t + horizon`` row after the full recurrent rollout."""

        trajectory = self.rollout(
            station=station,
            issue_date=issue_date,
            temperature_current=temperature_current,
            issue_forcing=issue_forcing,
            future_forcing=future_forcing,
            horizon=horizon,
            channel=channel,
            forcing_units=forcing_units,
            future_discharge=future_discharge,
        )
        return trajectory.tail(1).reset_index(drop=True)

    def to_payload(self) -> dict[str, object]:
        """Return the complete, versioned, unit-bearing model payload."""

        return {
            "schema_version": SCHEMA_VERSION,
            "model_name": MODEL_NAME,
            "equation": EQUATION,
            "temperature_unit": "degC",
            "training_period": [f"{TRAIN_START:%Y-%m-%d}", f"{TRAIN_END:%Y-%m-%d}"],
            "validation_period": [
                f"{VALIDATION_START:%Y-%m-%d}",
                f"{VALIDATION_END:%Y-%m-%d}",
            ],
            "forcing_terms": [term.payload() for term in FORCING_TERMS],
            "legal_channels": list(LEGAL_CHANNELS),
            "legal_horizons": list(LEGAL_HORIZONS),
            "future_discharge": {
                "enabled": self.future_discharge_enabled,
                "bundled_with_atmospheric_forcing": False,
            },
            "coefficients": dict(self.coefficients),
            "alpha_by_station": dict(self.alpha_by_station),
            "alpha_bounds": {
                "lower_open": 0.0,
                "calibration_numerical_floor": ALPHA_MIN,
                "upper_closed": ALPHA_MAX,
            },
            "training_pair_count_by_station": dict(self.training_pair_count_by_station),
            "training_pair_count": self.training_pair_count,
            "calibration": {
                "method": CALIBRATION_METHOD,
                "weighting": CALIBRATION_WEIGHTING,
                "initial_alpha": DEFAULT_INITIAL_ALPHA,
                "max_iterations": self.max_iterations,
                "tolerance": self.tolerance,
            },
            "iterations": self.iterations,
            "converged": self.converged,
        }

    def to_json(self) -> str:
        return (
            json.dumps(
                self.to_payload(),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> ForcedHybridModel:
        expected_keys = {
            "schema_version",
            "model_name",
            "equation",
            "temperature_unit",
            "training_period",
            "validation_period",
            "forcing_terms",
            "legal_channels",
            "legal_horizons",
            "future_discharge",
            "coefficients",
            "alpha_by_station",
            "alpha_bounds",
            "training_pair_count_by_station",
            "training_pair_count",
            "calibration",
            "iterations",
            "converged",
        }
        if set(payload) != expected_keys:
            raise ForcedHybridContractError("serialized model top-level schema is not exact")
        frozen = {
            "schema_version": SCHEMA_VERSION,
            "model_name": MODEL_NAME,
            "equation": EQUATION,
            "temperature_unit": "degC",
            "training_period": [f"{TRAIN_START:%Y-%m-%d}", f"{TRAIN_END:%Y-%m-%d}"],
            "validation_period": [
                f"{VALIDATION_START:%Y-%m-%d}",
                f"{VALIDATION_END:%Y-%m-%d}",
            ],
            "forcing_terms": [term.payload() for term in FORCING_TERMS],
            "legal_channels": list(LEGAL_CHANNELS),
            "legal_horizons": list(LEGAL_HORIZONS),
            "alpha_bounds": {
                "lower_open": 0.0,
                "calibration_numerical_floor": ALPHA_MIN,
                "upper_closed": ALPHA_MAX,
            },
        }
        for key, expected in frozen.items():
            if payload[key] != expected:
                raise ForcedHybridContractError(f"serialized model changed frozen field {key}")
        discharge = payload["future_discharge"]
        if discharge != {"enabled": False, "bundled_with_atmospheric_forcing": False}:
            raise ForcedHybridContractError("serialized future-discharge policy changed")
        coefficients = payload["coefficients"]
        alpha = payload["alpha_by_station"]
        counts = payload["training_pair_count_by_station"]
        calibration = payload["calibration"]
        if (
            not isinstance(coefficients, Mapping)
            or not isinstance(alpha, Mapping)
            or not isinstance(counts, Mapping)
            or not isinstance(calibration, Mapping)
        ):
            raise ForcedHybridContractError("serialized parameter registries must be mappings")
        if set(calibration) != {
            "method",
            "weighting",
            "initial_alpha",
            "max_iterations",
            "tolerance",
        }:
            raise ForcedHybridContractError("serialized calibration schema is not exact")
        if (
            calibration["method"] != CALIBRATION_METHOD
            or calibration["weighting"] != CALIBRATION_WEIGHTING
            or calibration["initial_alpha"] != DEFAULT_INITIAL_ALPHA
        ):
            raise ForcedHybridContractError("serialized calibration method changed")
        model = cls(
            coefficients=coefficients,  # type: ignore[arg-type]
            alpha_by_station=alpha,  # type: ignore[arg-type]
            training_pair_count_by_station=counts,  # type: ignore[arg-type]
            iterations=payload["iterations"],  # type: ignore[arg-type]
            converged=payload["converged"],  # type: ignore[arg-type]
            future_discharge_enabled=False,
            max_iterations=calibration["max_iterations"],  # type: ignore[arg-type]
            tolerance=calibration["tolerance"],  # type: ignore[arg-type]
        )
        if payload["training_pair_count"] != model.training_pair_count:
            raise ForcedHybridContractError("serialized total training-pair count changed")
        return model

    @classmethod
    def from_json(cls, document: str | bytes) -> ForcedHybridModel:
        try:
            payload = json.loads(document, object_pairs_hook=_reject_duplicate_json_pairs)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ForcedHybridContractError("serialized model is not valid JSON") from exc
        if not isinstance(payload, Mapping):
            raise ForcedHybridContractError("serialized model root must be an object")
        return cls.from_payload(payload)


def bind_formal_keys(predictions: pd.DataFrame, formal_keys: pd.DataFrame) -> pd.DataFrame:
    """Bind predictions only when candidate and formal key sets are identical."""

    if tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise ForcedHybridContractError("prediction schema/order is not exact")
    if set(predictions["channel"].to_numpy(copy=True)) - set(LEGAL_CHANNELS):
        raise ForcedHybridContractError("predictions contain an unsupported forcing channel")
    for column in ("temperature_equilibrium", "y_pred"):
        if not pd.api.types.is_numeric_dtype(
            predictions[column].dtype
        ) or pd.api.types.is_bool_dtype(predictions[column].dtype):
            raise ForcedHybridContractError(f"prediction {column} must be numeric")
        if not np.isfinite(predictions[column].to_numpy(dtype=np.float64, copy=True)).all():
            raise ForcedHybridContractError(f"prediction {column} must be finite")
    missing = [column for column in FORMAL_KEY_COLUMNS if column not in formal_keys.columns]
    if missing:
        raise ForcedHybridContractError(f"formal key registry lacks columns: {missing}")
    keys = formal_keys[list(FORMAL_KEY_COLUMNS)].copy(deep=True).reset_index(drop=True)
    if (
        keys.duplicated("key_id").any()
        or keys.duplicated(["site_id", "issue_date", "target_date", "lead_days"]).any()
    ):
        raise ForcedHybridContractError("formal key registry contains duplicate identities")
    if predictions.duplicated(["site_id", "issue_date", "target_date", "lead_days"]).any():
        raise ForcedHybridContractError("predictions contain duplicate identities")

    for frame, label in ((predictions, "prediction"), (keys, "formal key")):
        issue = pd.to_datetime(frame["issue_date"], errors="coerce")
        target = pd.to_datetime(frame["target_date"], errors="coerce")
        if (
            issue.isna().any()
            or target.isna().any()
            or getattr(issue.dt, "tz", None) is not None
            or getattr(target.dt, "tz", None) is not None
            or not issue.eq(issue.dt.normalize()).all()
            or not target.eq(target.dt.normalize()).all()
        ):
            raise ForcedHybridContractError(f"{label} dates are not canonical")
        lead_series = frame["lead_days"]
        if not pd.api.types.is_integer_dtype(lead_series.dtype) or pd.api.types.is_bool_dtype(
            lead_series.dtype
        ):
            raise ForcedHybridContractError(f"{label} lead_days must have integer dtype")
        lead = lead_series.to_numpy(dtype=np.int64, copy=True)
        if not set(lead) <= set(LEGAL_HORIZONS):
            raise ForcedHybridContractError(
                f"{label} lead_days must be in the frozen horizon set {LEGAL_HORIZONS}"
            )
        if not np.array_equal(
            target.to_numpy(dtype="datetime64[ns]"),
            issue.to_numpy(dtype="datetime64[ns]") + lead.astype("timedelta64[D]"),
        ):
            raise ForcedHybridContractError(f"{label} target boundary is invalid")

    for row in keys.itertuples(index=False):
        expected = forecast_key_id(
            row.site_id,
            row.issue_date,
            row.target_date,
            int(row.lead_days),
        )
        if row.key_id != expected:
            raise ForcedHybridContractError("formal key_id is not the canonical identity hash")

    identity = ["site_id", "issue_date", "target_date", "lead_days"]
    prediction_index = pd.MultiIndex.from_frame(predictions[identity])
    formal_index = pd.MultiIndex.from_frame(keys[identity])
    missing_predictions = formal_index.difference(prediction_index)
    outside_predictions = prediction_index.difference(formal_index)
    if len(missing_predictions) or len(outside_predictions):
        raise ForcedHybridContractError(
            "prediction/formal key boundary is not two-sided exact: "
            f"missing={len(missing_predictions)} outside={len(outside_predictions)}"
        )
    bound = keys.merge(predictions, on=identity, how="left", validate="one_to_one")
    if len(bound) != len(keys) or bound["y_pred"].isna().any():
        raise ForcedHybridContractError("formal key binding is not complete and one-to-one")
    return bound.reset_index(drop=True)

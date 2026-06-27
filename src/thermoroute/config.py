"""Central configuration for the ThermoRoute study.

Every constant that controls the experiment lives here so a single import gives
a reproducible description of the protocol.  Values were fixed *after* the data
audit (see ``outputs/reports/data_audit.md``) and *before* any model touched the
blind-test years, which is what keeps the evaluation honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
TABLES = OUTPUTS / "tables"
FIGURES = OUTPUTS / "figures"
PREDICTIONS = OUTPUTS / "predictions"
REPORTS = OUTPUTS / "reports"
MODELS = OUTPUTS / "models"
LOGS = OUTPUTS / "logs"

for _p in (DATA_PROCESSED, TABLES, FIGURES, PREDICTIONS, REPORTS, MODELS, LOGS):
    _p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Data schema
# --------------------------------------------------------------------------- #
TARGET = "WTEMP"
STATIONS: tuple[str, ...] = ("b1", "s2", "p3")
RAW_FILES: Mapping[str, str] = {"b1": "b1.csv", "s2": "s2.csv", "p3": "p3.csv"}

# All numeric channels carried through the pipeline.
ALL_VARS: tuple[str, ...] = (
    "WTEMP", "FLOW", "WLEVEL", "TEMP", "PRCP", "WDSP", "RHMEAN", "DH",
)
# Meteorological / hydrological forcings (everything that is not the target).
FORCINGS: tuple[str, ...] = ("FLOW", "WLEVEL", "TEMP", "PRCP", "WDSP", "RHMEAN", "DH")

# Sentinel "missing" codes discovered in the audit.  These are masked to NaN and
# imputed *within each fold*; they are NOT real extremes.
SENTINELS: Mapping[str, float] = {"WDSP": 999.9, "PRCP": 99.99}

# Variables for which a log1p transform stabilises the heavy right tail.
LOG1P_VARS: tuple[str, ...] = ("FLOW", "PRCP")


# --------------------------------------------------------------------------- #
# Station topology (confirmed by cross-correlation in the audit)
# --------------------------------------------------------------------------- #
# Directed downstream cascade b1 -> s2 -> p3.  Flow travel time ~1 day per hop;
# the thermal signal lags far more to p3 (~9 days), which is *why* a directed,
# variable-specific travel-time prior is physically motivated.
UPSTREAM: Mapping[str, str | None] = {"b1": None, "s2": "b1", "p3": "s2"}
FLOW_TRAVEL_DAYS: Mapping[tuple[str, str], int] = {("b1", "s2"): 1, ("s2", "p3"): 1}
THERMAL_TRAVEL_DAYS: Mapping[tuple[str, str], int] = {("b1", "s2"): 1, ("s2", "p3"): 9}
# Reservoir surface elevations differ by ~600-1500 m between stations, so WLEVEL
# must be standardised per station, never pooled on a common datum.

# DH semantics are NOT confirmed from a data dictionary.  The audit is consistent
# with a sunshine/insolation index (0-13, summer dip under monsoon cloud at the
# lower stations) but this remains an assumption; DH enters models as a generic
# radiative index and its data-dictionary meaning is an open verification item.
DH_SEMANTICS_VERIFIED = False


# --------------------------------------------------------------------------- #
# Task definition
# --------------------------------------------------------------------------- #
HORIZONS: tuple[int, ...] = (1, 3, 7)
QUANTILES: tuple[float, ...] = (0.05, 0.50, 0.95)          # for pinball / CQR
EXCEEDANCE_QUANTILE = 0.90                                  # high-temp threshold


# --------------------------------------------------------------------------- #
# Leakage-safe time split (years, inclusive).  Matches the plan's 10/2/1/2.
# The blind-test years are touched exactly once, at the very end.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TimeSplit:
    train: tuple[str, str] = ("2006-01-01", "2015-12-31")   # 10 y  fit everything
    val: tuple[str, str] = ("2016-01-01", "2017-12-31")     # 2 y   model selection
    calib: tuple[str, str] = ("2018-01-01", "2018-12-31")   # 1 y   conformal only
    test: tuple[str, str] = ("2019-01-01", "2020-12-31")    # 2 y   blind test

    def as_dict(self) -> Mapping[str, tuple[str, str]]:
        return {"train": self.train, "val": self.val,
                "calib": self.calib, "test": self.test}


SPLIT = TimeSplit()


# --------------------------------------------------------------------------- #
# Feature sets (mechanism ladder).  Each adds one driver group to isolate gains.
# --------------------------------------------------------------------------- #
FEATURE_SETS: Mapping[str, tuple[str, ...]] = {
    "V1": ("WTEMP",),                                         # thermal inertia only
    "V2": ("WTEMP", "FLOW", "WLEVEL", "TEMP", "PRCP"),        # + hydro + air temp
    "V3": ALL_VARS,                                           # + WDSP, RHMEAN, DH
}

# Candidate lags (days) exposed to tabular models and to the router.
SHORT_LAGS: tuple[int, ...] = (0, 1, 2, 3, 5, 7, 10, 14)
ROLLING_WINDOWS: tuple[int, ...] = (3, 7, 14, 30)
CONTEXT_LENGTH = 32          # days of history fed to sequence models
MAX_ROUTER_LAG = 14          # router attends over lags 0..14
SEASONAL_HARMONICS = 3       # K in the harmonic climatology


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEEDS: tuple[int, ...] = (0, 1, 2)        # deep models; baselines are deterministic
PRIMARY_SEED = 0
SEASONAL_PERIOD = 365.2425


@dataclass(frozen=True)
class TrainConfig:
    """Hyper-parameters for the PyTorch models (fixed on val, never on test)."""
    d_model: int = 40
    encoder_blocks: int = 2
    kernel_size: int = 3
    dropout: float = 0.15
    n_experts: int = 3
    station_embed_dim: int = 8
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    max_epochs: int = 80
    patience: int = 12
    grad_clip: float = 1.0
    lambda_event: float = 0.3       # weight on the exceedance BCE
    lambda_residual: float = 1e-2   # L1 keeping the net close to the physics prior
    lambda_crossing: float = 1.0    # quantile non-crossing penalty


TRAIN = TrainConfig()


def horizon_weights(horizons: Sequence[int] = HORIZONS) -> dict[int, float]:
    """Equal weight per horizon in the multi-task loss (documented, not tuned)."""
    return {h: 1.0 for h in horizons}

"""Central configuration for the ThermoRoute study.

Every constant that controls the legacy development experiment lives here so a
single import gives a reproducible protocol description.  The 2019--2020 rows
have already informed development and are exploratory evaluation, not an
untouched or blind test.  The separately frozen Route-A confirmation is defined
under ``protocols/``.
"""

from __future__ import annotations

from dataclasses import dataclass
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

OUTPUT_DIRECTORIES = (
    DATA_PROCESSED, TABLES, FIGURES, PREDICTIONS, REPORTS, MODELS, LOGS,
)


def ensure_output_directories() -> None:
    """Create runtime output directories explicitly.

    Importing experiment configuration is intentionally side-effect free; CLI
    entry points call this helper (or create their own destination) when needed.
    """
    for path in OUTPUT_DIRECTORIES:
        path.mkdir(parents=True, exist_ok=True)


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
# Legacy three-monitoring-site ordering (not used by Route A)
# --------------------------------------------------------------------------- #
# b1, s2 and p3 are ordinary monitoring-site identifiers, not reservoirs.  No
# verified station metadata in this repository establishes a directed hydraulic
# connection or travel time between them, so the executable configuration does
# not encode one.  Route A likewise has no river graph and makes no routing claim.
UPSTREAM: Mapping[str, str | None] = {"b1": None, "s2": None, "p3": None}
FLOW_TRAVEL_DAYS: Mapping[tuple[str, str], int] = {}
THERMAL_TRAVEL_DAYS: Mapping[tuple[str, str], int] = {}
# WLEVEL metadata/datum comparability is unverified, so it must be standardised
# per station and may not be interpreted as reservoir surface elevation.

# This flag concerns only the legacy three-site input files.  Their DH semantics
# are not confirmed by a data dictionary, so that channel is only a generic
# radiative index there.  Route A constructs its separate DH column from Daymet
# shortwave radiation and freezes that provider-specific definition in protocol.
DH_SEMANTICS_VERIFIED = False


# --------------------------------------------------------------------------- #
# Task definition
# --------------------------------------------------------------------------- #
HORIZONS: tuple[int, ...] = (1, 3, 7)
QUANTILES: tuple[float, ...] = (0.05, 0.50, 0.95)          # for pinball / CQR
EXCEEDANCE_QUANTILE = 0.90                                  # high-temp threshold


# --------------------------------------------------------------------------- #
# Leakage-safe development split (years, inclusive).  Matches 10/2/1/2.
# ``test`` is retained as a file-format/API name; its scientific role is an
# already-inspected exploratory evaluation partition.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TimeSplit:
    train: tuple[str, str] = ("2006-01-01", "2015-12-31")   # 10 y  fit everything
    val: tuple[str, str] = ("2016-01-01", "2017-12-31")     # 2 y   model selection
    calib: tuple[str, str] = ("2018-01-01", "2018-12-31")   # 1 y   conformal only
    test: tuple[str, str] = ("2019-01-01", "2020-12-31")    # 2 y development eval

    def as_dict(self) -> Mapping[str, tuple[str, str]]:
        return {"train": self.train, "val": self.val,
                "calib": self.calib, "test": self.test}


SPLIT = TimeSplit()


# --------------------------------------------------------------------------- #
# Legacy three-site feature ladder.  Route A uses its separately frozen seven-
# variable schema and never consumes WLEVEL.
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
    # Serialization compatibility only: neural quantiles are ordered by
    # construction, so the corresponding loss term is identically zero.
    lambda_crossing: float = 1.0


TRAIN = TrainConfig()


def horizon_weights(horizons: Sequence[int] = HORIZONS) -> dict[int, float]:
    """Equal weight per horizon in the multi-task loss (documented, not tuned)."""
    return {h: 1.0 for h in horizons}


# Current development-selected residual bound, imported by
# scripts 09/10/13/13b/13c.  The surviving validation ledger used seed 0 for all
# four scales and added seeds 1--2 only for the three near-tied scales; it was
# not a symmetric three-seed sweep.  Earlier 2019--2020 inspection also informed
# project development.  Route A therefore treats this as a frozen development
# configuration, not as selection on an untouched test or as a safety threshold.
# Do not hard-code delta elsewhere.
DELTA_SCALE: float = 1.0

# Exact seed set used for the large-sample deep models (scripts 09/13). The
# 3-station track uses SEEDS above; USGS uses these five.
USGS_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

"""ThermoRoute: physics-guided, dynamic-lag, calibrated multi-station river
water-temperature forecasting.

A compact, leakage-safe research codebase for 1/3/7-day daily water-temperature
prediction.  The legacy small-sample case uses three ordinary monitoring-site
identifiers (b1, s2, p3); they are not reservoirs, and their display order is
not evidence of a hydraulic cascade.

The public API is intentionally small; scripts under ``scripts/`` orchestrate
the full experimental matrix. See ``README.md`` for the run order.
"""

from __future__ import annotations

__version__ = "1.0.0"

from . import config, data, features  # noqa: F401

__all__ = ["config", "data", "features", "__version__"]

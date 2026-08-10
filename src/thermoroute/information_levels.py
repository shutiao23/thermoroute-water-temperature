"""Local-information levels for the corrected L ladder.

The previous ladder was destroyed because a feature-path change went in without
anyone checking what it did to the actual training rows (DLOG-025).  This module
exists to make the equivalent change checkable in isolation: it decides, for one
information level, exactly which of the 210 frozen base feature columns a model
may see and which climatology the anchor may use, and nothing else.

The central design choice is that prohibited inputs are **dropped, not filled**.
A masked-but-present column invites two failures that a dropped column cannot
have: the model can learn that a sentinel value marks a particular regime, and
the fill value itself carries information about the distribution it replaced.
Dropping makes the invariance requirement in the protocol -- predictions must
not change when prohibited inputs are perturbed -- true by construction rather
than by test, and the test then confirms the construction rather than
substituting for it.

Levels, following ``protocols/wrr_information_regimes_protocol_v4.yaml``:

``L0``
    History-rich gauged site.  Everything the v5 arms already use.
``L1``
    Gauged cold-start.  Issue-date water temperature and the recent sequence
    stay visible, but no target-site long-term statistic does: the climatology
    and the damped rate come from the training stations, pooled.
``L2``
    Thermally ungauged, hydrology observed.  Every target-site water-temperature
    input is gone -- lags, rolling statistics, deltas, their observedness flags,
    the climatology anomaly and the persistence anchor -- and the anchor is the
    pooled training climatology alone.  Discharge and meteorology remain.
``L2_U2``
    Fully local-observation-free.  ``L2`` with discharge removed as well, so no
    target-site observation of any kind reaches the model.

What this module deliberately does not decide: the spatial fold, the model
class, the forcing level, or the anchor's numerical values.  Those are the
runner's business.  Keeping the level definition this small is what makes it
reviewable.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

__all__ = [
    "LEVELS",
    "InformationLevel",
    "admissible_columns",
    "level",
    "prohibited_columns",
]

#: Columns derived from the target site's own water temperature.  ``clim_anom``
#: is the issue-time anomaly and therefore water-temperature-derived even though
#: its name does not say so; ``persistence`` is the issue-date reading itself.
#: Both are easy to miss and are exactly the sort of omission that turns a
#: "thermally ungauged" arm into a gauged one.
_WATER_TEMPERATURE_PREFIXES = ("WTEMP_",)
_WATER_TEMPERATURE_EXTRAS = ("clim_anom", "persistence")

#: Columns derived from the target site's discharge.
_FLOW_PREFIXES = ("FLOW_",)

#: Target-site long-term statistics.  These are not dropped at ``L1``; they are
#: recomputed from pooled training stations, which the runner does.  They are
#: named here so the contract is explicit about what "no target-site long-term
#: statistic" means.
POOLED_AT_L1 = ("clim_t", "clim_target")


@dataclass(frozen=True)
class InformationLevel:
    """One rung of the local-information ladder."""

    name: str
    water_temperature_visible: bool
    flow_visible: bool
    target_site_climatology: str      # "per_station" | "training_pooled"
    anchor: str
    description: str

    @property
    def pooled_climatology(self) -> bool:
        return self.target_site_climatology == "training_pooled"


LEVELS: dict[str, InformationLevel] = {
    "L0": InformationLevel(
        name="L0",
        water_temperature_visible=True,
        flow_visible=True,
        target_site_climatology="per_station",
        anchor="per_station_damped_persistence",
        description="history-rich gauged site",
    ),
    "L1": InformationLevel(
        name="L1",
        water_temperature_visible=True,
        flow_visible=True,
        target_site_climatology="training_pooled",
        anchor="training_pooled_damped_persistence",
        description="gauged cold-start; no target-site long-term statistic",
    ),
    "L2": InformationLevel(
        name="L2",
        water_temperature_visible=False,
        flow_visible=True,
        target_site_climatology="training_pooled",
        anchor="training_pooled_climatology_only",
        description="thermally ungauged, hydrology observed",
    ),
    "L2_U2": InformationLevel(
        name="L2_U2",
        water_temperature_visible=False,
        flow_visible=False,
        target_site_climatology="training_pooled",
        anchor="training_pooled_climatology_only",
        description="fully local-observation-free",
    ),
}


def level(name: str) -> InformationLevel:
    try:
        return LEVELS[name]
    except KeyError:
        raise ValueError(
            f"unknown information level {name!r}; expected one of {sorted(LEVELS)}"
        ) from None


def _matches(column: str, prefixes: Sequence[str], extras: Sequence[str]) -> bool:
    return column.startswith(tuple(prefixes)) or column in extras


def prohibited_columns(name: str, columns: Iterable[str]) -> tuple[str, ...]:
    """Columns this level forbids, in the order they appear.

    Written as an explicit list of (hidden variable, matcher) pairs rather than
    a boolean chain: which variable a rung hides is the thing a reviewer needs
    to check, and it should be readable without resolving operator precedence.
    """
    rung = level(name)
    hidden: list[tuple[Sequence[str], Sequence[str]]] = []
    if not rung.water_temperature_visible:
        hidden.append((_WATER_TEMPERATURE_PREFIXES, _WATER_TEMPERATURE_EXTRAS))
    if not rung.flow_visible:
        hidden.append((_FLOW_PREFIXES, ()))
    return tuple(
        column
        for column in columns
        if any(_matches(column, prefixes, extras) for prefixes, extras in hidden)
    )


def admissible_columns(name: str, columns: Iterable[str]) -> tuple[str, ...]:
    """Columns this level permits, preserving the caller's order.

    Dropping rather than filling is the whole point: a column absent from the
    design matrix cannot influence a prediction, whatever is done to the panel
    it would have come from.
    """
    ordered = tuple(columns)
    forbidden = set(prohibited_columns(name, ordered))
    kept = tuple(column for column in ordered if column not in forbidden)
    if not kept:
        raise ValueError(f"level {name} admits no feature column")
    return kept

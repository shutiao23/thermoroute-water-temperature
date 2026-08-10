#!/usr/bin/env python3
"""Corrected local-information ladder, whole-region geometry (Phase 3).

Replaces the 432-cell ladder that DLOG-025 withdrew.  That one was destroyed by
a feature-path change nobody checked against the actual training rows; this one
is built so the equivalent change is checkable before any model is fitted.

Design, per fold and information level:

* the held-out fold is a set of whole HUC2 regions, so no gauge from a held
  region appears in training;
* every preprocessing statistic a level forbids is refitted on the in-fold
  stations only -- imputer, climatology, damped rate -- rather than reused;
* every feature column a level forbids is *dropped* from the design matrix, so
  a prohibited input cannot reach the model by any route
  (:mod:`thermoroute.information_levels`);
* before fitting, a mask-invariance proof perturbs the held-out stations' own
  water temperature (and discharge at ``L2_U2``) in the raw panel, rebuilds the
  whole feature path, and requires the resulting design matrix to be bit
  identical.  That is the check the previous ladder did not have.

Levels and their level-legal anchor:

  L0      per-station climatology and damped anchor
  L1      pooled climatology and pooled damped anchor
  L2      pooled climatology only; no damped anchor exists without local WTEMP
  L2_U2   as L2, and discharge is gone as well

``ResidualLightGBM`` fits the residual to whichever anchor the level permits,
which is the damped anchor at L0/L1 and the pooled climatology at L2/L2_U2.
Fitting it against an anchor the level forbids would smuggle the forbidden
information back in through the target.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.final.run_forcing_ladder_v5_observed as V5
import scripts.final.run_forcing_placebo_v5a_shuffle as P
from thermoroute import config as C
from thermoroute import data as D
from thermoroute import features as F
from thermoroute.information_levels import admissible_columns, level

LEVEL_NAMES = ("L0", "L1", "L2", "L2_U2")
GEOMETRY = "whole_region"
N_FOLDS = 4
OUTPUT_DIR = V5.FINAL_OUTPUT_ROOT / "information_ladder_v6_observed"
SHARD_DIRNAME = "ladder_shards_v6_observed"
MANIFEST_FILENAME = "ladder_lineage_manifest_v6_observed.json"
STATUS = "POST_OUTCOME_OBSERVED_LINEAGE_LADDER"
PERTURBATION_SEED = 20260810


class LadderError(RuntimeError):
    """A ladder precondition or invariance check failed."""


@dataclass(frozen=True)
class Fold:
    index: int
    in_fold: tuple[str, ...]
    held: tuple[str, ...]
    regions_held: tuple[str, ...]


# ---------------------------------------------------------------- folds


def whole_region_folds(registry: pd.DataFrame) -> list[Fold]:
    """Pack whole HUC2 regions into four balanced folds, deterministically."""
    by_region: dict[str, list[str]] = {}
    for row in registry.itertuples(index=False):
        by_region.setdefault(str(row.huc2).zfill(2), []).append(
            str(row.site_no).zfill(8)
        )
    buckets: list[list[str]] = [[] for _ in range(N_FOLDS)]
    regions: list[list[str]] = [[] for _ in range(N_FOLDS)]
    loads = [0] * N_FOLDS
    # largest region first, always into the lightest bucket; ties broken by the
    # region code so the packing is a pure function of the registry
    for region, sites in sorted(by_region.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        target = int(np.argmin(loads))
        buckets[target].extend(sites)
        regions[target].append(region)
        loads[target] += len(sites)
    everyone = {str(s).zfill(8) for s in registry["site_no"]}
    folds = []
    for index, (held, held_regions) in enumerate(zip(buckets, regions, strict=True)):
        folds.append(Fold(
            index=index,
            in_fold=tuple(sorted(everyone - set(held))),
            held=tuple(sorted(held)),
            regions_held=tuple(sorted(held_regions)),
        ))
    covered = sorted({s for fold in folds for s in fold.held})
    if covered != sorted(everyone):
        raise LadderError("whole-region folds do not partition the cohort")
    return folds


# ------------------------------------------------------- level preprocessing


@dataclass(frozen=True)
class LevelPreprocessing:
    imputer: Any
    climatology: Any
    damped: Any | None
    level_name: str
    fold_index: int


def fit_level_preprocessing(
    raw_frame: pd.DataFrame, fold: Fold, level_name: str
) -> LevelPreprocessing:
    """Refit every statistic the level constrains, on the stations it allows."""
    rung = level(level_name)
    training, _validation = V5.period_masks(raw_frame)
    # L0 lets the target site contribute its own state; every other rung does
    # not, so the fit set is the in-fold stations.
    fit_stations = None if level_name == "L0" else fold.in_fold

    # A per-station imputer has no fill for a station it never saw, and the
    # target site's own imputer state is precisely what L1 and above forbid.
    # Pooling is therefore not a convenience here, it is the level's contract.
    imputer = D.Imputer.fit(
        raw_frame,
        training,
        fit_stations=fit_stations,
        pooled=fit_stations is not None,
    )
    climatology = F.HarmonicClimatology.fit(
        raw_frame,
        training,
        fit_stations=fit_stations,
        pooled=rung.pooled_climatology,
    )
    damped = None
    if rung.water_temperature_visible:
        damped = F.DampedPersistenceAnchor.fit(
            raw_frame,
            training,
            climatology,
            fit_stations=fit_stations,
            pooled=rung.pooled_climatology,
        )
    return LevelPreprocessing(
        imputer=imputer, climatology=climatology, damped=damped,
        level_name=level_name, fold_index=fold.index,
    )


def level_anchor_values(
    rows: pd.DataFrame, preprocessing: LevelPreprocessing, horizon: int
) -> np.ndarray:
    """The anchor the level permits, used for the residual target and prediction.

    At L0/L1 that is the damped anchor.  At L2 there is no local water
    temperature, so the only legal anchor is the pooled climatology at the
    target date -- using the damped anchor there would reintroduce exactly the
    information the level removes.
    """
    if preprocessing.damped is not None:
        return V5._damped_values(rows, preprocessing.damped, horizon)
    return _climatology_at_target(rows, preprocessing)


def _climatology_at_target(
    rows: pd.DataFrame, preprocessing: LevelPreprocessing
) -> np.ndarray:
    out = np.empty(len(rows), dtype=float)
    for station, block in rows.groupby("site_id", sort=False):
        out[block.index.to_numpy()] = preprocessing.climatology.predict_dates(
            station, block["target_date"]
        )
    return out


# ------------------------------------------------------ mask-invariance proof


def perturb_prohibited_inputs(
    raw_frame: pd.DataFrame, fold: Fold, level_name: str, *, seed: int
) -> pd.DataFrame:
    """Replace the held-out stations' prohibited observations with noise.

    A level that genuinely hides an input must produce an identical design
    matrix after this.  Perturbing the *held* stations specifically is what
    catches the three failures that matter: a surviving feature column, a
    climatology that quietly used held-station data, and an anchor that did.
    """
    rung = level(level_name)
    variables: list[str] = []
    if not rung.water_temperature_visible:
        variables.append("WTEMP")
    if not rung.flow_visible:
        variables.append("FLOW")
    if not variables:
        return raw_frame

    rng = np.random.default_rng(seed)
    out = raw_frame.copy(deep=True)
    held = out["site_id"].isin(fold.held).to_numpy()
    for variable in variables:
        values = out[variable].to_numpy(float).copy()
        observed = np.isfinite(values) & held
        # keep the missingness pattern; move every observed held value
        values[observed] = rng.normal(50.0, 25.0, size=int(observed.sum()))
        out[variable] = values
        out[f"{variable}_observed"] = np.isfinite(values)
    return out


def build_design(
    raw_frame: pd.DataFrame, fold: Fold, level_name: str, horizon: int
) -> tuple[pd.DataFrame, tuple[str, ...], LevelPreprocessing]:
    """Full feature path for one (fold, level, lead), prohibited columns dropped."""
    source = hashlib.sha256(
        V5._canonical_frame_sha256(raw_frame, V5.RAW_PANEL_COLUMNS).encode()
    ).hexdigest()
    panel = V5.make_raw_label_panel(raw_frame, source_sha256=source)
    preprocessing = fit_level_preprocessing(raw_frame, fold, level_name)
    imputed = V5.impute_feature_panel(panel, preprocessing.imputer)
    table, base_columns = V5.build_observed_feature_table(
        panel, imputed, preprocessing.climatology, horizon
    )
    columns = admissible_columns(level_name, base_columns)
    return table, columns, preprocessing


def prove_mask_invariance(
    raw_frame: pd.DataFrame, fold: Fold, level_name: str, horizon: int
) -> dict[str, Any]:
    """Require a bit-identical design matrix after perturbing hidden inputs."""
    rung = level(level_name)
    if rung.water_temperature_visible and rung.flow_visible:
        return {"applicable": False, "reason": "level hides no local observation"}

    table, columns, _ = build_design(raw_frame, fold, level_name, horizon)
    perturbed_frame = perturb_prohibited_inputs(
        raw_frame, fold, level_name, seed=PERTURBATION_SEED
    )
    other, other_columns, _ = build_design(
        perturbed_frame, fold, level_name, horizon
    )
    if columns != other_columns:
        raise LadderError(f"{level_name} column set changed under perturbation")

    held = table["site_id"].isin(fold.held)
    left = table.loc[held, list(columns)].to_numpy(float)
    right = other.loc[held.values, list(columns)].to_numpy(float)
    if left.shape != right.shape:
        raise LadderError(f"{level_name} held-station row count changed")
    if not np.array_equal(np.isnan(left), np.isnan(right)):
        raise LadderError(f"{level_name} missingness changed under perturbation")
    finite = ~np.isnan(left)
    worst = float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0
    if worst != 0.0:
        raise LadderError(
            f"{level_name} design matrix moved by {worst} under perturbation of "
            "inputs the level forbids; the mask is not complete"
        )
    return {
        "applicable": True,
        "perturbed_variables": (
            ["WTEMP"] if rung.flow_visible else ["WTEMP", "FLOW"]
        ),
        "held_rows_checked": int(held.sum()),
        "columns_checked": len(columns),
        "max_abs_difference": worst,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--prove-only", action="store_true",
                        help="run the mask-invariance proof over every fold and stop")
    parser.add_argument("--folds", nargs="*", type=int, default=None,
                        help="restrict to these fold indices (explicit, never implicit)")
    parser.add_argument("--levels", nargs="*", default=list(LEVEL_NAMES))
    parser.add_argument("--horizons", nargs="*", type=int, default=list(V5.HORIZONS))
    return parser


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bounds = P.capture_placebo_inputs()
    raw, stations, registry = V5._load_raw_panel_from_bounds(bounds)
    C.STATIONS = stations
    reference = V5.validate_reference_registry(
        pd.read_parquet(io.BytesIO(bounds["primary_key_registry"].payload)),
        require_production_counts=True,
    )
    return V5._validated_raw_frame(raw), registry, reference


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw_frame, registry, _reference = load_inputs()
    folds = whole_region_folds(registry)
    if args.folds is not None:
        folds = [f for f in folds if f.index in set(args.folds)]
        if not folds:
            raise LadderError(f"no fold matches {args.folds}")

    if args.prove_only or not args.execute:
        report = []
        # --prove-only means "run the proofs and stop", not "only fold 0";
        # restricting the fold set silently would make a passing proof cover a
        # quarter of the cohort while reading as if it covered all of it.
        for fold in folds:
            for level_name in args.levels:
                for horizon in args.horizons:
                    proof = prove_mask_invariance(
                        raw_frame, fold, level_name, horizon
                    )
                    report.append({
                        "fold": fold.index, "level": level_name,
                        "horizon": horizon, **proof,
                    })
                    print(
                        f"  fold{fold.index} {level_name} h{horizon}: {proof}",
                        flush=True,
                    )
        print(json.dumps({"mask_invariance": report}, sort_keys=True, indent=1))
        return 0

    raise SystemExit("execution path is added in the next step")


if __name__ == "__main__":
    raise SystemExit(main())

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
from thermoroute.baselines import _lgb_fit
from thermoroute.information_levels import admissible_columns, level

LEVEL_NAMES = ("L0", "L1", "L2", "L2_U2")
GEOMETRIES = ("whole_region", "random_site")
GEOMETRY = "whole_region"          # default; --geometry selects
N_FOLDS = 4
#: Random-site geometry is averaged over seeds, so one seed is never a result.
#: The protocol pairs the station effect inside each seed and averages the
#: paired contrasts, never the risks.
RANDOM_SEEDS = (0, 1, 2, 3, 4)
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


def random_site_folds(registry: pd.DataFrame, seed: int) -> list[Fold]:
    """Four balanced random-site folds, the interpolation-like comparator.

    Held sites are scattered, so a held gauge's neighbours normally remain in
    training.  Contrasting this with the whole-region packing at the same
    information level is what isolates spatial geometry from local information.
    """
    sites = sorted(str(s).zfill(8) for s in registry["site_no"])
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sites))
    folds = []
    for index in range(N_FOLDS):
        held = sorted(sites[j] for j in order[index::N_FOLDS])
        folds.append(Fold(
            index=index,
            in_fold=tuple(sorted(set(sites) - set(held))),
            held=tuple(held),
            regions_held=(f"random_seed{seed}",),
        ))
    covered = sorted({s for fold in folds for s in fold.held})
    if covered != sites:
        raise LadderError("random-site folds do not partition the cohort")
    return folds


def folds_for(registry: pd.DataFrame, geometry: str, seed: int | None) -> list[Fold]:
    if geometry == "whole_region":
        return whole_region_folds(registry)
    if geometry == "random_site":
        if seed is None:
            raise LadderError("random-site geometry requires a seed")
        return random_site_folds(registry, seed)
    raise LadderError(f"unknown geometry {geometry!r}")


# ------------------------------------------------------- level preprocessing


@dataclass(frozen=True)
class LevelPreprocessing:
    imputer: Any
    climatology: Any
    damped: Any | None
    meteorology_climatologies: Any
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
    # F3 needs these to fill a missing future value.  They are target-site
    # statistics like any other, so a level that pools the water climatology
    # pools these as well; leaving them per-station would reintroduce the
    # station's own record through the substitution path.
    meteorology = {
        variable: F.HarmonicClimatology.fit(
            raw_frame, training, target=variable,
            fit_stations=fit_stations, pooled=rung.pooled_climatology,
        )
        for variable in V5.METEOROLOGY_VARIABLES
    }
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
        meteorology_climatologies=meteorology,
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
    raw_frame: pd.DataFrame, fold: Fold, level_name: str, horizon: int,
    forcing: str = "F0",
) -> tuple[pd.DataFrame, tuple[str, ...], LevelPreprocessing]:
    """Full feature path for one (fold, level, forcing, lead).

    The two axes are orthogonal by construction: the information level decides
    which *local* observations the model may see, the forcing level decides
    whether it may see future meteorology.  An L2/F3 cell is therefore a real
    and interesting one -- thermally ungauged but with perfect future weather --
    rather than a contradiction, and it is what makes the interaction between
    local information and forcing measurable at all.
    """
    source = hashlib.sha256(
        V5._canonical_frame_sha256(raw_frame, V5.RAW_PANEL_COLUMNS).encode()
    ).hexdigest()
    panel = V5.make_raw_label_panel(raw_frame, source_sha256=source)
    preprocessing = fit_level_preprocessing(raw_frame, fold, level_name)
    imputed = V5.impute_feature_panel(panel, preprocessing.imputer)
    table, base_columns = V5.build_observed_feature_table(
        panel, imputed, preprocessing.climatology, horizon
    )
    if forcing == "F3_full":
        future = V5.build_raw_future_registry(
            panel, table[["site_id", "issue_date"]], horizon
        )
        table, base_columns = V5.materialize_forcing_features(
            table, base_columns, arm="F3_full", horizon=horizon,
            raw_future_registry=future,
            meteorology_climatologies=preprocessing.meteorology_climatologies,
        )
    elif forcing != "F0":
        raise LadderError(f"unknown forcing level {forcing!r}")
    columns = admissible_columns(level_name, base_columns)
    return table, columns, preprocessing


def prove_mask_invariance(
    raw_frame: pd.DataFrame, fold: Fold, level_name: str, horizon: int,
    forcing: str = "F0",
) -> dict[str, Any]:
    """Require a bit-identical design matrix after perturbing hidden inputs."""
    rung = level(level_name)
    if rung.water_temperature_visible and rung.flow_visible:
        return {"applicable": False, "reason": "level hides no local observation"}

    table, columns, _ = build_design(raw_frame, fold, level_name, horizon, forcing)
    perturbed_frame = perturb_prohibited_inputs(
        raw_frame, fold, level_name, seed=PERTURBATION_SEED
    )
    other, other_columns, _ = build_design(
        perturbed_frame, fold, level_name, horizon, forcing
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


# ---------------------------------------------------------------- execution


def fit_ladder_cell(
    table: pd.DataFrame,
    evaluation: pd.DataFrame,
    columns: Sequence[str],
    preprocessing: LevelPreprocessing,
    fold: Fold,
    model: str,
    horizon: int,
    geometry: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit on in-fold stations only and predict the held region's stations."""
    in_fold = set(fold.in_fold)
    train = table[table["split"].eq("train") & table["site_id"].isin(in_fold)].copy()
    validation = table[table["split"].eq("val") & table["site_id"].isin(in_fold)].copy()
    if train.empty or validation.empty:
        raise LadderError(f"fold {fold.index} has an empty train or validation set")

    # the two guarantees the old ladder lost
    for rows, label in ((train, "train"), (validation, "validation")):
        if not rows["issue_wtemp_observed"].to_numpy(bool).all():
            raise LadderError(f"{label} row has an imputed issue label")
        if not rows["target_wtemp_observed"].to_numpy(bool).all():
            raise LadderError(f"{label} row has an imputed target label")
    identity = ["site_id", "issue_date", "target_date"]
    if set(train[identity].itertuples(index=False, name=None)) & set(
        validation[identity].itertuples(index=False, name=None)
    ):
        raise LadderError("train and validation identities overlap")
    if set(train["site_id"]) & set(fold.held):
        raise LadderError("a held-region station reached the training set")

    held = evaluation[evaluation["site_id"].isin(set(fold.held))].copy()
    if held.empty:
        raise LadderError(f"fold {fold.index} scored no held station")
    held = held.reset_index(drop=True)

    train_anchor = level_anchor_values(train.reset_index(drop=True), preprocessing, horizon)
    val_anchor = level_anchor_values(
        validation.reset_index(drop=True), preprocessing, horizon
    )
    held_anchor = level_anchor_values(held, preprocessing, horizon)

    train_outcome = train["y"].to_numpy(float)
    val_outcome = validation["y"].to_numpy(float)
    if model == "ResidualLightGBM":
        train_outcome = train_outcome - train_anchor
        val_outcome = val_outcome - val_anchor

    fitted = _lgb_fit(
        train[list(columns)], train_outcome,
        validation[list(columns)], val_outcome,
        "regression",
        n_est=V5.BEST_ITER_UPPER_BOUND[horizon],
        params_override=dict(V5.FROZEN_PARAMS[horizon]),
    )
    prediction = np.asarray(
        fitted.predict(held[list(columns)], num_threads=1), dtype=np.float64
    )
    if model == "ResidualLightGBM":
        prediction = prediction + held_anchor
    if len(prediction) != len(held) or not np.isfinite(prediction).all():
        raise LadderError("ladder prediction is non-finite or misaligned")

    shard = pd.DataFrame({
        "key_id": held["key_id"].to_numpy(),
        "site_id": held["site_id"].to_numpy(),
        "issue_date": held["issue_date"].to_numpy(),
        "target_date": held["target_date"].to_numpy(),
        "horizon": np.full(len(held), horizon, dtype=np.int16),
        "level": preprocessing.level_name,
        "geometry": geometry,
        "fold": np.full(len(held), fold.index, dtype=np.int16),
        "model": model,
        "analysis_status": STATUS,
        "y_true": held["y_true"].to_numpy(float),
        "y_pred": prediction,
        "y_anchor": held_anchor,
    })
    evidence = {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "held_rows": len(held),
        "features": len(columns),
        "best_iteration": int(
            getattr(fitted, "best_iteration_", 0) or V5.BEST_ITER_UPPER_BOUND[horizon]
        ),
        "target_kind": "level_anchor_residual" if model == "ResidualLightGBM" else "raw_y",
        "anchor": level(preprocessing.level_name).anchor,
    }
    return shard, evidence


def execute(args: argparse.Namespace) -> int:
    raw_frame, registry, reference = load_inputs()
    geometry = args.geometry
    seeds: list[int | None] = (
        [None] if geometry == "whole_region" else list(args.seeds)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shard_dir = OUTPUT_DIR / SHARD_DIRNAME
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    for seed in seeds:
      folds = folds_for(registry, geometry, seed)
      if args.folds is not None:
          folds = [f for f in folds if f.index in set(args.folds)]
      tag = "" if seed is None else f"_seed{seed}"
      for fold in folds:
        for level_name in args.levels:
            for horizon in args.horizons:
                forcing = args.forcing
                proof = prove_mask_invariance(
                    raw_frame, fold, level_name, horizon, forcing
                )
                table, columns, preprocessing = build_design(
                    raw_frame, fold, level_name, horizon, forcing
                )
                # Bind against the full frozen namespace: that validator is
                # what enforces two-sided equality with the formal key
                # registry, and it only recognises the complete F0 column set.
                # The level's column subset is applied when the model is fitted.
                bind_columns = (
                    V5.FROZEN_BASE_FEATURE_COLUMNS if forcing == "F0"
                    else (*V5.FROZEN_BASE_FEATURE_COLUMNS,
                          *V5._future_feature_columns(horizon))
                )
                evaluation = V5.bind_exact_evaluation_rows(
                    table, reference, horizon, bind_columns
                )
                for model in V5.MODELS:
                    prefix = "" if forcing == "F0" else f"{forcing}_"
                    name = (
                        f"{prefix}{level_name}_{geometry}{tag}_fold{fold.index}"
                        f"_{model}_h{horizon}"
                    )
                    path = shard_dir / f"{name}.parquet"
                    if path.exists():
                        print(f"  skip {name} (already written)", flush=True)
                        continue
                    shard, evidence = fit_ladder_cell(
                        table, evaluation, columns, preprocessing,
                        fold, model, horizon, geometry,
                    )
                    shard.to_parquet(path, index=False)
                    manifest.append({
                        "cell": name, "level": level_name, "geometry": geometry,
                        "forcing": forcing, "seed": seed,
                        "fold": fold.index, "regions_held": list(fold.regions_held),
                        "model": model, "horizon": horizon,
                        "mask_invariance": proof, "fit": evidence,
                        "shard": path.name,
                    })
                    print(f"  {name}: {len(shard)} rows, {evidence['features']} features",
                          flush=True)

    manifest_name = (
        MANIFEST_FILENAME if not args.manifest_tag
        else MANIFEST_FILENAME.replace(".json", f"_{args.manifest_tag}.json")
    )
    (OUTPUT_DIR / manifest_name).write_text(
        json.dumps({
            "format": "thermoroute.information-ladder-v6-observed.v1",
            "status": STATUS,
            "geometry": geometry,
            "seeds": [s for s in seeds],
            "levels": list(args.levels),
            "cells": manifest,
        }, sort_keys=True, indent=1, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "WRITTEN", "cells": len(manifest)},
                     sort_keys=True, indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--prove-only", action="store_true",
                        help="run the mask-invariance proof over every fold and stop")
    parser.add_argument("--folds", nargs="*", type=int, default=None,
                        help="restrict to these fold indices (explicit, never implicit)")
    parser.add_argument("--levels", nargs="*", default=list(LEVEL_NAMES))
    parser.add_argument("--horizons", nargs="*", type=int, default=list(V5.HORIZONS))
    parser.add_argument("--geometry", choices=GEOMETRIES, default="whole_region")
    parser.add_argument("--forcing", choices=("F0", "F3_full"), default="F0")
    parser.add_argument("--manifest-tag", default="",
                        help="suffix for this worker's manifest, so parallel "
                             "workers on disjoint cells do not overwrite each "
                             "other's lineage record")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(RANDOM_SEEDS),
                        help="random-site split seeds; ignored for whole_region")
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

    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())

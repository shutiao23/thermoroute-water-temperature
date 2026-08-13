#!/usr/bin/env python3
"""Plain causal TCN, information-matched to the tree arms (architecture axis).

The crossed matrix needs a second model class that sees *exactly* what the trees
see, so that a difference between them is a difference of model class and not of
information.  ``run_neural_information_regimes.py`` was supposed to supply it:
4,602 lines that import no deep-learning library and fit nothing.  This is a
working replacement, deliberately small.

Information matching is structural here rather than promised.  The trees consume
a frozen tabular namespace that is already a lag window -- ``{VAR}_lag0 …
{VAR}_lag14`` per variable -- plus rolling statistics, deltas, calendar
harmonics, the climatology and the anchor.  This model reshapes the *same*
columns: lag columns become the sequence axis of a causal convolution, and every
remaining column enters as a static feature at the head.  Nothing the trees see
is withheld and nothing they cannot see is added.

Causality is by construction.  The convolution is left-padded so position *k*
draws only on positions at or before it, with the lag axis ordered oldest to
newest, and only the newest position is read out.  On this feature layout a
non-causal kernel would in fact be harmless -- every lag is already in the past
of the issue time -- but the architecture should be what its name says, so the
property is enforced and unit-tested rather than assumed.  The dilations are
sized so the receptive field spans the whole lag grid: a stack that reached back
only seven of the eight positions would be causal *and* blind to the oldest
observation the trees can split on, which is an information mismatch dressed up
as an architecture.

Two matching concessions are recorded rather than hidden.  LightGBM routes NaN
down a learned branch; a dense network cannot, so missing cells are set to the
training mean after standardisation.  That gives the network *less* to work with
than the trees, so it cannot inflate the architecture contrast in the network's
favour, and the affected fraction is written into the manifest.  Second, a
variable whose lag set is only partly admissible at some level would give the
convolution a ragged history, so its surviving columns are demoted to static
features -- still visible, just not convolved.

Level masking reuses :mod:`thermoroute.information_levels` unchanged, the
residual is added to the level-legal anchor, and the bounded variant applies the
same +/-1 degC algebraic limit as the manuscript's constrained model, so
"bounded" means the same thing across model classes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.final.run_information_ladder_v6_observed as LAD
import scripts.final.run_forcing_ladder_v5_observed as V5

OUTPUT_DIR = V5.FINAL_OUTPUT_ROOT / "plain_tcn_arm_v1"
SHARD_DIRNAME = "tcn_shards_v1"
MANIFEST_FILENAME = "tcn_lineage_manifest_v1.json"
STATUS = "POST_OUTCOME_OBSERVED_LINEAGE_ARCHITECTURE_ARM"

LAG_PATTERN = re.compile(r"^(?P<var>[A-Za-z0-9]+)_lag(?P<lag>\d+)$")
FIT_SEEDS = (0, 1, 2)
BOUND_DEGC = 1.0
EPOCHS = 40
#: Stop when the in-fold validation loss has not improved for this many epochs.
#: A fixed 40-epoch budget converged at L0 (median best epoch 9-23) but bound at
#: L2, where the median was 37 and a third of cells were still improving at the
#: cap -- so the L2 architecture penalty partly measured a training budget
#: rather than a model class.  Patience makes the stopping rule a property of
#: convergence instead of the clock, at a cost only in the cells that need it.
PATIENCE = 20
BATCH = 4096
LEARNING_RATE = 3e-3
CHANNELS = 32
KERNEL = 3
DILATIONS = (1, 2, 4)


class TcnError(RuntimeError):
    """A precondition or invariance check failed."""


# ------------------------------------------------------------------- layout


def sequence_layout(
    columns: Sequence[str],
) -> tuple[list[str], list[int], list[str]]:
    """Split an admissible namespace into a lag grid and static features.

    Returns ``(variables, lags, static)``.  ``lags`` is the intersection of the
    lag offsets present for every candidate variable, ordered oldest first so
    the sequence axis runs forward in time; a variable that does not carry the
    full grid stays out of the sequence branch and its columns fall through to
    ``static``, where the head still sees them.
    """
    grid: dict[str, set[int]] = {}
    for column in columns:
        match = LAG_PATTERN.match(column)
        if match:
            grid.setdefault(match["var"], set()).add(int(match["lag"]))
    if not grid:
        return [], [], list(columns)
    lags = sorted(set.intersection(*grid.values()), reverse=True)   # oldest first
    variables = sorted(v for v, present in grid.items() if set(lags) <= present)
    used = {f"{v}_lag{lag}" for v in variables for lag in lags}
    static = [column for column in columns if column not in used]
    return variables, lags, static


class PlainCausalTCN(nn.Module):
    """Dilated causal convolutions over the lag axis, then a static head."""

    def __init__(self, n_variables: int, n_static: int, *, bounded: bool) -> None:
        super().__init__()
        self.bounded = bounded
        self.n_variables = n_variables
        self.pads = nn.ModuleList()
        self.convs = nn.ModuleList()
        in_channels = n_variables
        for dilation in DILATIONS if n_variables else ():
            self.pads.append(nn.ConstantPad1d(((KERNEL - 1) * dilation, 0), 0.0))
            self.convs.append(
                nn.Conv1d(in_channels, CHANNELS, KERNEL, dilation=dilation)
            )
            in_channels = CHANNELS
        pooled = CHANNELS if n_variables else 0
        self.head = nn.Sequential(
            nn.Linear(pooled + n_static, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def convolve(self, sequence: torch.Tensor) -> torch.Tensor:
        hidden = sequence
        for pad, conv in zip(self.pads, self.convs, strict=True):
            hidden = torch.relu(conv(pad(hidden)))
        return hidden

    def forward(self, sequence: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.n_variables:
            parts.append(self.convolve(sequence)[:, :, -1])   # newest position only
        parts.append(static)
        out = self.head(torch.cat(parts, dim=1)).squeeze(1)
        return BOUND_DEGC * torch.tanh(out) if self.bounded else out


# ------------------------------------------------------------------ tensors


def to_tensors(
    frame: pd.DataFrame,
    variables: Sequence[str],
    lags: Sequence[int],
    static: Sequence[str],
    stats: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], float]:
    """Standardise with training-set statistics; report the imputed fraction."""
    rows = len(frame)
    if variables:
        stacked = np.stack(
            [
                np.stack(
                    [frame[f"{v}_lag{lag}"].to_numpy(np.float32) for lag in lags],
                    axis=1,
                )
                for v in variables
            ],
            axis=1,
        )                                                    # (rows, vars, lags)
    else:
        stacked = np.zeros((rows, 0, max(len(lags), 1)), dtype=np.float32)
    flat = (
        frame[list(static)].to_numpy(np.float32)
        if static
        else np.zeros((rows, 0), dtype=np.float32)
    )

    missing = int(np.isnan(stacked).sum() + np.isnan(flat).sum())
    cells = max(stacked.size + flat.size, 1)

    if stats is None:
        stats = {
            "seq_mean": (
                np.nan_to_num(np.nanmean(stacked, axis=(0, 2), keepdims=True))
                if variables else None
            ),
            "seq_std": (
                np.nan_to_num(np.nanstd(stacked, axis=(0, 2), keepdims=True)) + 1e-6
                if variables else None
            ),
            "flat_mean": (
                np.nan_to_num(np.nanmean(flat, axis=0, keepdims=True))
                if static else np.zeros((1, 0), np.float32)
            ),
            "flat_std": (
                np.nan_to_num(np.nanstd(flat, axis=0, keepdims=True)) + 1e-6
                if static else np.ones((1, 0), np.float32)
            ),
        }
    if variables:
        stacked = (stacked - stats["seq_mean"]) / stats["seq_std"]
    if static:
        flat = (flat - stats["flat_mean"]) / stats["flat_std"]
    # after centring, zero *is* the training mean: NaN -> mean imputation
    stacked = np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    return (
        torch.from_numpy(np.ascontiguousarray(stacked, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(flat, dtype=np.float32)),
        stats,
        missing / cells,
    )


# ---------------------------------------------------------------------- fit


def fit_cell(
    table: pd.DataFrame,
    evaluation: pd.DataFrame,
    columns: Sequence[str],
    preprocessing: Any,
    fold: Any,
    horizon: int,
    seed: int,
    bounded: bool,
    epochs: int = EPOCHS,
    patience: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Fit on in-fold stations only and predict the held region's stations."""
    variables, lags, static = sequence_layout(columns)
    in_fold = set(fold.in_fold)
    train = table[table["split"].eq("train") & table["site_id"].isin(in_fold)]
    validation = table[table["split"].eq("val") & table["site_id"].isin(in_fold)]
    if train.empty or validation.empty:
        raise TcnError(f"fold {fold.index} has an empty train or validation set")

    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    # the same two guarantees the tree ladder enforces (DLOG-025)
    for rows, label in ((train, "train"), (validation, "validation")):
        if not rows["issue_wtemp_observed"].to_numpy(bool).all():
            raise TcnError(f"{label} row has an imputed issue label")
        if not rows["target_wtemp_observed"].to_numpy(bool).all():
            raise TcnError(f"{label} row has an imputed target label")
    if set(train["site_id"]) & set(fold.held):
        raise TcnError("a held-region station reached the training set")

    held = evaluation[evaluation["site_id"].isin(set(fold.held))].reset_index(drop=True)
    if held.empty:
        raise TcnError(f"fold {fold.index} scored no held station")

    train_anchor = LAD.level_anchor_values(train, preprocessing, horizon)
    val_anchor = LAD.level_anchor_values(validation, preprocessing, horizon)
    held_anchor = LAD.level_anchor_values(held, preprocessing, horizon)

    seq_tr, flat_tr, stats, missing = to_tensors(train, variables, lags, static)
    seq_va, flat_va, _, _ = to_tensors(validation, variables, lags, static, stats)
    seq_he, flat_he, _, _ = to_tensors(held, variables, lags, static, stats)
    y_tr = torch.from_numpy(
        (train["y"].to_numpy(float) - train_anchor).astype(np.float32)
    )
    y_va = torch.from_numpy(
        (validation["y"].to_numpy(float) - val_anchor).astype(np.float32)
    )

    torch.manual_seed(seed)
    model = PlainCausalTCN(len(variables), flat_tr.shape[1], bounded=bounded)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)

    best_state: dict[str, torch.Tensor] | None = None
    best_loss, best_epoch = float("inf"), -1
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(y_tr), generator=generator)
        for start in range(0, len(order), BATCH):
            index = order[start : start + BATCH]
            optimiser.zero_grad()
            loss = loss_fn(model(seq_tr[index], flat_tr[index]), y_tr[index])
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(model(seq_va, flat_va), y_va))
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif patience and epoch - best_epoch >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        residual = model(seq_he, flat_he).numpy().astype(np.float64)
    prediction = residual + held_anchor
    if len(prediction) != len(held) or not np.isfinite(prediction).all():
        raise TcnError("TCN prediction is non-finite or misaligned")
    if bounded and float(np.max(np.abs(residual))) > BOUND_DEGC + 1e-6:
        raise TcnError("bounded variant exceeded its algebraic limit")

    evidence = {
        "sequence_variables": variables,
        "lags_oldest_first": lags,
        "static_features": len(static),
        "features_total": len(columns),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "held_rows": len(held),
        "best_epoch": best_epoch,
        "epoch_budget": epochs,
        "patience": patience,
        "stopped_at_budget": bool(best_epoch >= epochs - 1),
        "best_validation_mse": best_loss,
        "bounded": bounded,
        "max_abs_residual": float(np.max(np.abs(residual))),
        "train_cells_mean_imputed_fraction": round(missing, 6),
        "target_kind": "level_anchor_residual",
        "anchor": LAD.level(preprocessing.level_name).anchor,
    }
    return held, prediction, evidence


# ---------------------------------------------------------------- execution


def execute(args: argparse.Namespace) -> int:
    torch.set_num_threads(1)
    raw_frame, registry, reference = LAD.load_inputs()
    # Random-site geometry carries its own folds per split seed, so the split
    # seed is part of a cell's identity and appears in the shard name.  Reusing
    # the whole-region name for a random-site cell would silently overwrite it.
    split_seeds: list[int | None] = (
        [None] if args.geometry == "whole_region" else list(args.split_seeds)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shard_dir = OUTPUT_DIR / SHARD_DIRNAME
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    suffix = "bounded" if args.bounded else "unbounded"

    for split_seed in split_seeds:
      folds = LAD.folds_for(registry, args.geometry, split_seed)
      if args.folds is not None:
          folds = [f for f in folds if f.index in set(args.folds)]
      split_tag = "" if split_seed is None else f"_seed{split_seed}"
      for fold in folds:
        for level_name in args.levels:
            for horizon in args.horizons:
                table, columns, preprocessing = LAD.build_design(
                    raw_frame, fold, level_name, horizon, args.forcing
                )
                bind_columns = (
                    V5.FROZEN_BASE_FEATURE_COLUMNS
                    if args.forcing == "F0"
                    else (
                        *V5.FROZEN_BASE_FEATURE_COLUMNS,
                        *V5._future_feature_columns(horizon),
                    )
                )
                evaluation = V5.bind_exact_evaluation_rows(
                    table, reference, horizon, bind_columns
                )
                for seed in args.seeds:
                    prefix = "" if args.forcing == "F0" else f"{args.forcing}_"
                    name = (
                        f"{prefix}{level_name}_{args.geometry}{split_tag}"
                        f"_fold{fold.index}"
                        f"_PlainTCN_{suffix}_seed{seed}_h{horizon}"
                        f"{args.shard_suffix}"
                    )
                    path = shard_dir / f"{name}.parquet"
                    if path.exists():
                        print(f"  skip {name} (already written)", flush=True)
                        continue
                    held, prediction, evidence = fit_cell(
                        table, evaluation, columns, preprocessing,
                        fold, horizon, seed, args.bounded,
                        epochs=args.epochs, patience=args.patience,
                    )
                    pd.DataFrame({
                        "key_id": held["key_id"].to_numpy(),
                        "site_id": held["site_id"].to_numpy(),
                        "issue_date": held["issue_date"].to_numpy(),
                        "target_date": held["target_date"].to_numpy(),
                        "horizon": np.full(len(held), horizon, dtype=np.int16),
                        "level": level_name,
                        "forcing": args.forcing,
                        "geometry": args.geometry,
                        "fold": np.full(len(held), fold.index, dtype=np.int16),
                        "model": f"PlainTCN_{suffix}",
                        "seed": np.full(len(held), seed, dtype=np.int16),
                        "analysis_status": STATUS,
                        "y_true": held["y_true"].to_numpy(float),
                        "y_pred": prediction,
                    }).to_parquet(path, index=False)
                    manifest.append({
                        "cell": name, "level": level_name, "forcing": args.forcing,
                        "geometry": args.geometry, "split_seed": split_seed,
                        "fold": fold.index,
                        "regions_held": list(fold.regions_held),
                        "seed": seed, "horizon": horizon,
                        "model": f"PlainTCN_{suffix}", "shard": path.name,
                        "fit": evidence,
                    })
                    print(
                        f"  {name}: {len(held)} rows, "
                        f"{len(evidence['sequence_variables'])} sequence variables x "
                        f"{len(evidence['lags_oldest_first'])} lags, "
                        f"{evidence['static_features']} static",
                        flush=True,
                    )

    tag = f"_{args.manifest_tag}" if args.manifest_tag else ""
    (OUTPUT_DIR / MANIFEST_FILENAME.replace(".json", f"{tag}.json")).write_text(
        json.dumps({
            "format": "thermoroute.plain-tcn-arm.v1",
            "status": STATUS,
            "architecture": {
                "kind": "plain causal TCN over the lag axis",
                "channels": CHANNELS, "kernel": KERNEL,
                "dilations": list(DILATIONS),
                "receptive_field_positions": 1 + (KERNEL - 1) * sum(DILATIONS),
                "epochs": EPOCHS, "batch": BATCH,
                "learning_rate": LEARNING_RATE,
                "bound_degC": BOUND_DEGC,
                "model_selection": "lowest in-fold validation MSE across epochs",
            },
            "information_matching": (
                "the same frozen tabular namespace the tree arms consume, under "
                "the same level mask; lag columns reshaped into the sequence "
                "axis, all remaining columns entering as static features. "
                "Missing cells are set to the training mean, which gives the "
                "network strictly less than LightGBM's learned NaN branch."
            ),
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
    parser.add_argument("--levels", nargs="*", default=["L0", "L2"])
    parser.add_argument("--horizons", nargs="*", type=int, default=list(V5.HORIZONS))
    parser.add_argument("--forcing", choices=("F0", "F3_full"), default="F0")
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.add_argument("--geometry", choices=LAD.GEOMETRIES, default="whole_region")
    parser.add_argument("--split-seeds", nargs="*", type=int,
                        default=list(LAD.RANDOM_SEEDS),
                        help="random-site split seeds; ignored for whole_region")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(FIT_SEEDS),
                        help="fit seeds, which are a different axis from the "
                             "split seeds above")
    parser.add_argument("--bounded", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=0,
                        help="stop after this many epochs without improvement; "
                             "0 keeps the fixed-budget behaviour")
    parser.add_argument("--shard-suffix", default="",
                        help="suffix for shard names, so a sensitivity run "
                             "cannot overwrite the primary arm's cells")
    parser.add_argument("--manifest-tag", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        print(json.dumps({
            "levels": args.levels, "horizons": args.horizons,
            "seeds": args.seeds, "bounded": args.bounded,
            "geometry": args.geometry,
            "cells": len(args.levels) * len(args.horizons) * len(args.seeds)
                     * (len(args.folds) if args.folds else LAD.N_FOLDS)
                     * (1 if args.geometry == "whole_region"
                        else len(args.split_seeds)),
        }, indent=1))
        return 0
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())

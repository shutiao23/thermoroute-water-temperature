#!/usr/bin/env python3
"""Run the +/-7-day time-shifted forcing arms (protocol v5a, P4 timing control).

The within-month shuffle established that the forcing value is event-scale
rather than seasonal-phase information, but it cannot say whether the model is
using the *exact* days that followed an issue or merely the sub-monthly weather
window around them: a donor drawn from the same station-month can land within a
few days of the true valid time.  These arms answer that.  The realized future
is displaced by a whole week in each direction, which preserves station
climate, near-seasonal phase, the joint marginals and local weather persistence
while destroying exact event timing.

Construction follows the shuffle arm: the displacement is applied upstream to
the meteorology columns of the raw panel the future lookup reads, so the
registry is built through the ordinary public factory and every v5 validator
runs unchanged.  A panel row dated ``D`` is made to carry the meteorology of
``D + 7`` for the plus arm and ``D - 7`` for the minus arm, which is exactly
what ``build_raw_future_registry`` needs for a key's valid time ``t + h`` to
receive the weather of ``t + h +/- 7``.

Boundary.  Displacement leaves seven days at one end of each station's record
without a donor.  Those cells become missing, the F3 materialization
substitutes climatology for them, and the substitution is counted per key.  The
sealed boundary rule requires the affected keys to be dropped from *both* shift
arms and from the true arm within the shift contrast, so the scoring authority
forms one common shift-registry from the recorded substitution counts.  That
registry is named separately and is never mixed with the primary 358,765-key
registry.

The minus arm is the weaker control of the pair and is labelled so: displacing
backwards moves the future window toward information the model already has at
issue time.  The plus arm is the primary timing control that decision rule P4
is stated in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.final.run_forcing_ladder_v5_observed as V5
import scripts.final.run_forcing_placebo_v5a_shuffle as SHUF
from thermoroute import config as C

SHIFT_ARMS = {"F3_shift_plus7": 7, "F3_shift_minus7": -7}
FEATURE_ARM = "F3_full"

OUTPUT_DIR = V5.FINAL_OUTPUT_ROOT / "forcing_placebo_v5a_shift"
SHARD_DIRNAME = "placebo_shards_v5a_shift"
MANIFEST_FILENAME = "placebo_lineage_manifest_v5a_shift.json"


class ShiftError(RuntimeError):
    """A shift precondition or invariant failed; nothing was written."""


def shift_meteorology(frame: pd.DataFrame, *, days: int) -> pd.DataFrame:
    """Make each station-day carry the meteorology of ``days`` later.

    A positive ``days`` implements the +7 arm: the row dated ``D`` receives the
    values observed on ``D + days``, so a lookup at valid time ``t + h`` returns
    the weather of ``t + h + days``.  Rows with no donor inside the station's
    record become missing and their observedness masks follow, which keeps the
    raw-panel invariant ``*_observed == isfinite(value)`` true by construction.
    """
    if days == 0:
        raise ShiftError("a zero-day shift is not a control")
    out = frame.copy(deep=True)
    variables = list(V5.METEOROLOGY_VARIABLES)
    values = out[variables].to_numpy(copy=True)
    shifted = np.full_like(values, np.nan)

    order = np.lexsort((out["DATE"].to_numpy(), out["site_id"].to_numpy()))
    sites = out["site_id"].to_numpy()[order]
    dates = out["DATE"].to_numpy(dtype="datetime64[D]")[order]
    donor_rows = 0
    orphan_rows = 0

    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and sites[stop] == sites[start]:
            stop += 1
        block = order[start:stop]
        block_dates = dates[start:stop]
        # the record is a strict daily calendar per station, so the donor of
        # row i is row i+days; verify rather than assume
        gaps = np.diff(block_dates)
        if len(gaps) and not np.all(gaps == np.timedelta64(1, "D")):
            raise ShiftError(
                f"station {sites[start]!r} is not a strict daily calendar; "
                "a positional shift would silently mis-date the donor"
            )
        size = len(block)
        if days > 0:
            take, put = block[days:], block[: size - days]
        else:
            take, put = block[: size + days], block[-days:]
        shifted[put] = values[take]
        donor_rows += len(put)
        orphan_rows += size - len(put)
        start = stop

    out[variables] = shifted
    for variable in variables:
        out[f"{variable}_observed"] = np.isfinite(out[variable].to_numpy())
    out.attrs["shift_ledger"] = {
        "shift_days": days,
        "rows_with_a_donor": donor_rows,
        "rows_left_without_a_donor": orphan_rows,
    }
    return out


def build_shifted_panel(raw_panel: Any, *, days: int) -> tuple[Any, dict[str, Any]]:
    """Return a factory-issued raw panel whose future meteorology is displaced."""
    true_frame = V5._validated_raw_frame(raw_panel)
    shifted = shift_meteorology(true_frame, days=days)
    ledger = dict(shifted.attrs.get("shift_ledger", {}))

    content = V5._canonical_frame_sha256(shifted, V5.RAW_PANEL_COLUMNS)
    source = hashlib.sha256(
        f"F3_shift|days={days}|from={raw_panel.source_sha256}|content={content}".encode()
    ).hexdigest()
    panel = V5.make_raw_label_panel(shifted, source_sha256=source)

    rebuilt = V5._validated_raw_frame(panel)
    for column in ("site_id", "DATE"):
        if not np.array_equal(true_frame[column].to_numpy(), rebuilt[column].to_numpy()):
            raise ShiftError(f"shift altered {column}, which must be held fixed")
    for column in ("WTEMP", "FLOW"):
        left, right = true_frame[column].to_numpy(), rebuilt[column].to_numpy()
        if not np.array_equal(np.isnan(left), np.isnan(right)) or not np.array_equal(
            left[~np.isnan(left)], right[~np.isnan(right)]
        ):
            raise ShiftError(f"shift altered {column}, which must be held fixed")
    ledger["shifted_panel_content_sha256"] = content
    ledger["shifted_panel_source_sha256"] = source
    return panel, ledger


def shard_filename(arm: str, model: str, horizon: int) -> str:
    return f"{arm}_{model}_h{horizon}_v5a.parquet"


def execute(*, require_clean_tree: bool = True) -> int:
    authority = SHUF.build_execution_authority(require_clean_tree=require_clean_tree)
    authority["arm"] = sorted(SHIFT_ARMS)
    authority["runner_sha256"] = SHUF._sha256_file(Path(__file__).resolve())
    authority["shuffle_runner_sha256"] = SHUF._sha256_file(Path(SHUF.__file__).resolve())

    snapshot = SHUF.capture_placebo_inputs()
    raw, stations, _ = V5._load_raw_panel_from_bounds(snapshot)
    reference = V5.validate_reference_registry(
        pd.read_parquet(__import__("io").BytesIO(snapshot["primary_key_registry"].payload)),
        require_production_counts=True,
    )
    C.STATIONS = stations
    preprocessing = V5.fit_observed_preprocessing(raw, stations)
    preprocessing_record = V5.preprocessing_content_record(preprocessing)
    imputed = V5.impute_feature_panel(raw, preprocessing.imputer)

    expected_files: dict[str, dict[str, object]] = {}
    cell_manifest: list[dict[str, object]] = []

    with V5._BundleTransaction(
        OUTPUT_DIR,
        allowed_root=V5.FINAL_OUTPUT_ROOT,
        expected_destination_name=OUTPUT_DIR.name,
    ) as transaction:
        for horizon in V5.HORIZONS:
            base_table, base_columns = V5.build_observed_feature_table(
                raw, imputed, preprocessing.water_climatology, horizon,
            )
            for arm, days in sorted(SHIFT_ARMS.items()):
                panel, ledger = build_shifted_panel(raw, days=days)
                registry = V5.build_raw_future_registry(
                    panel, base_table[["site_id", "issue_date"]], horizon,
                )
                arm_table, model_columns = V5.materialize_forcing_features(
                    base_table, base_columns, arm=FEATURE_ARM, horizon=horizon,
                    raw_future_registry=registry,
                    meteorology_climatologies=preprocessing.meteorology_climatologies,
                )
                V5.assert_fit_identities_unchanged(base_table, arm_table)
                evaluation = V5.bind_exact_evaluation_rows(
                    arm_table, reference, horizon, model_columns,
                )
                for model in V5.MODELS:
                    cell = V5.Cell(FEATURE_ARM, model, horizon)
                    frame, fit_evidence = V5.fit_tree_cell(
                        arm_table, evaluation, model_columns, preprocessing, cell,
                    )
                    frame = V5.validate_v5_shard(frame, cell, reference)
                    relative = f"{SHARD_DIRNAME}/{shard_filename(arm, model, horizon)}"
                    binding = transaction.write_parquet(relative, frame)
                    expected_files[relative] = dict(binding)
                    substituted = int((frame["forcing_substitution_count"] > 0).sum())
                    cell_manifest.append({
                        "placebo_arm": arm,
                        "shift_days": days,
                        "feature_arm": FEATURE_ARM,
                        "model": model,
                        "horizon": horizon,
                        "shard": {"path": relative, "rows": len(frame), **binding},
                        "keys_with_a_boundary_substitution": substituted,
                        "shift_ledger": ledger,
                        "future_registry_identity_sha256": registry.identity_sha256,
                        "fit": fit_evidence,
                    })
                    print(f"  {arm} {model} h{horizon}: {len(frame)} rows, "
                          f"{substituted} boundary-substituted", flush=True)

        manifest = {
            "format": "thermoroute.forcing-placebo-v5a-shift-lineage.v1",
            "status": "SEALED_PRE_OUTCOME_PLACEBO_ARM",
            "execution_authority": authority,
            "primary_timing_control": "F3_shift_plus7",
            "weaker_control_note": (
                "F3_shift_minus7 displaces the future window toward information "
                "already available at issue time and is the weaker of the pair"
            ),
            "cells": cell_manifest,
        }
        payload = json.dumps(manifest, sort_keys=True, indent=1, allow_nan=False).encode()
        expected_files[MANIFEST_FILENAME] = dict(
            transaction.write_bytes(MANIFEST_FILENAME, payload)
        )

        def precommit() -> None:
            expected = len(V5.HORIZONS) * len(SHIFT_ARMS) * len(V5.MODELS)
            written = [n for n in expected_files if n.endswith(".parquet")]
            if len(written) != expected or len(cell_manifest) != expected:
                raise ShiftError(f"staged {len(written)} shards, expected {expected}")
            V5._validated_raw_frame(raw)
            V5._validated_imputed_frame(imputed)
            if V5.preprocessing_content_record(preprocessing) != preprocessing_record:
                raise ShiftError("preprocessing content changed before commit")
            if SHUF.SEALER.verify_seal()["drifted_inputs"]:
                raise ShiftError("sealed placebo inputs drifted during the run")

        transaction.commit(expected_files, precommit_check=precommit)

    print(json.dumps({"status": "WRITTEN", "out_dir": str(OUTPUT_DIR),
                      "shards": len(cell_manifest)}, sort_keys=True, indent=1))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-dirty-tree", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(
            SHUF.build_execution_authority(require_clean_tree=not args.allow_dirty_tree),
            sort_keys=True, indent=1))
        return 0
    return execute(require_clean_tree=not args.allow_dirty_tree)


if __name__ == "__main__":
    raise SystemExit(main())

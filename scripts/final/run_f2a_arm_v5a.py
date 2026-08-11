#!/usr/bin/env python3
"""The F2a arm: what a real fixed-lead temperature forecast recovers.

The component arms showed the whole forcing value is future *air temperature*:
at seven days it alone recovers 95% of the oracle gain, and withholding it
leaves 21%.  That is what makes this arm worth running.  F2a is a
temperature-only product, so if the value had been spread across five variables
a temperature forecast could only ever have recovered a fraction of it; because
it is not, F2a can in principle recover almost all of it, and the recovery
fraction becomes a real measurement rather than a formality.

Construction mirrors ``only_air_temperature`` exactly, with one substitution:
air temperature at each valid time is the *forecast* value rather than the
realized one, and every other meteorological variable is climatological.  The
comparator is therefore ``only_air_temperature`` itself, which is
``F3_temperature_only`` under another name -- realized air temperature, all
else climatological -- so the two arms differ in exactly one thing.

Training, which is where the obvious implementation is wrong.  The forecast
archive covers 2021-2023 only, so a model *trained* on F2a features sees a
future-temperature column that is climatology on every training row: it learns
the column is uninformative and ignores it, and substituting real forecasts at
evaluation then changes nothing.  A first run of this arm did exactly that and
reported a recovery of 1.8%, which measured the mistake rather than the
product.  The arm therefore trains on *realized* air temperature -- the
F3_temperature_only features -- and substitutes the forecast only when
predicting, which is also what an operational system does: learn the
relationship from reanalysis, apply it to a forecast.

Fixed-lead semantics, stated because the alternative reading is forbidden.  For
horizon ``h`` this uses the lead-``h`` forecast series at every valid time in
``t+1 … t+h``.  That is a *fixed-lead composite*: each value is what a run
issued ``h`` days earlier said about that day.  It is not a coherent trajectory
from one initialization, and the sealed protocol forbids calling it one or
calling it an as-issued operational archive.  The recovery fraction it supports
is named ``retrospective_temperature_composite_recovery`` and its denominator
is ``F3_temperature_only``, never the full five-variable oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import io
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

import scripts.final.run_forcing_components_v5a as CP
import scripts.final.run_forcing_ladder_v5_observed as V5
import scripts.final.run_forcing_placebo_v5a_shuffle as SHUF
from thermoroute import config as C

F2A_PANEL = V5.FINAL_OUTPUT_ROOT / "f2a_forecast_panel_v1.parquet"
OUTPUT_DIR = V5.FINAL_OUTPUT_ROOT / "forcing_f2a_v5a"
SHARD_DIRNAME = "f2a_shards_v5a"
MANIFEST_FILENAME = "f2a_lineage_manifest_v5a.json"
ARM = "F2a_temperature_only"
FEATURE_ARM = "F3_full"


class F2aArmError(RuntimeError):
    """An F2a precondition failed."""


def forecast_panel(
    raw_frame: pd.DataFrame,
    climatologies: dict[str, Any],
    forecasts: pd.DataFrame,
    horizon: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Air temperature becomes the lead-``horizon`` forecast; the rest climatology.

    A station-day with no forecast keeps a missing value rather than falling
    back to the realized observation, because a silent fallback would let the
    oracle leak into the arm that is supposed to be worse than it.  The
    substitution machinery then fills those cells with climatology and counts
    them, exactly as it does for any other gap.
    """
    out = CP.climatological_panel(raw_frame, climatologies, ("TEMP",))
    lead = forecasts[forecasts["lead_days"] == horizon]
    lookup = lead.set_index(["site_id", "target_date"])["f2a_temp_c"]
    index = pd.MultiIndex.from_arrays(
        [out["site_id"].to_numpy(), out["DATE"].to_numpy()]
    )
    values = lookup.reindex(index).to_numpy(float)
    covered = int(np.isfinite(values).sum())
    out["TEMP"] = values
    out["TEMP_observed"] = np.isfinite(values)
    return out, {
        "horizon": horizon,
        "station_days_with_a_forecast": covered,
        "station_days_without": int(len(out) - covered),
        "fallback_to_realized": False,
    }


def execute(*, require_clean_tree: bool = True) -> int:
    if not F2A_PANEL.exists():
        raise F2aArmError(
            f"forecast panel missing: {F2A_PANEL}; run build_f2a_forecast_panel.py"
        )
    authority = SHUF.build_execution_authority(require_clean_tree=require_clean_tree)
    authority["arm"] = ARM
    authority["runner_sha256"] = SHUF._sha256_file(Path(__file__).resolve())
    authority["forecast_panel_sha256"] = SHUF._sha256_file(F2A_PANEL)

    forecasts = pd.read_parquet(F2A_PANEL)
    snapshot = SHUF.capture_placebo_inputs()
    raw, stations, _ = V5._load_raw_panel_from_bounds(snapshot)
    reference = V5.validate_reference_registry(
        pd.read_parquet(io.BytesIO(snapshot["primary_key_registry"].payload)),
        require_production_counts=True,
    )
    C.STATIONS = stations
    preprocessing = V5.fit_observed_preprocessing(raw, stations)
    imputed = V5.impute_feature_panel(raw, preprocessing.imputer)
    true_frame = V5._validated_raw_frame(raw)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shard_dir = OUTPUT_DIR / SHARD_DIRNAME
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    for horizon in V5.HORIZONS:
        base_table, base_columns = V5.build_observed_feature_table(
            raw, imputed, preprocessing.water_climatology, horizon,
        )
        # training design: realized air temperature, everything else climatology
        train_frame = CP.climatological_panel(
            true_frame, preprocessing.meteorology_climatologies, ("TEMP",)
        )
        train_panel = V5.make_raw_label_panel(
            train_frame,
            source_sha256=hashlib.sha256(
                f"f2a-train|h{horizon}|"
                f"{V5._canonical_frame_sha256(train_frame, V5.RAW_PANEL_COLUMNS)}".encode()
            ).hexdigest(),
        )
        train_registry = V5.build_raw_future_registry(
            train_panel, base_table[["site_id", "issue_date"]], horizon
        )
        train_table, train_columns = V5.materialize_forcing_features(
            base_table, base_columns, arm=FEATURE_ARM, horizon=horizon,
            raw_future_registry=train_registry,
            meteorology_climatologies=preprocessing.meteorology_climatologies,
        )

        # evaluation design: the forecast substituted for realized temperature
        frame, coverage = forecast_panel(
            true_frame, preprocessing.meteorology_climatologies, forecasts, horizon
        )
        content = V5._canonical_frame_sha256(frame, V5.RAW_PANEL_COLUMNS)
        panel = V5.make_raw_label_panel(
            frame,
            source_sha256=hashlib.sha256(f"f2a|h{horizon}|{content}".encode()).hexdigest(),
        )
        registry = V5.build_raw_future_registry(
            panel, base_table[["site_id", "issue_date"]], horizon,
        )
        arm_table, model_columns = V5.materialize_forcing_features(
            base_table, base_columns, arm=FEATURE_ARM, horizon=horizon,
            raw_future_registry=registry,
            meteorology_climatologies=preprocessing.meteorology_climatologies,
        )
        V5.assert_fit_identities_unchanged(base_table, arm_table)
        if model_columns != train_columns:
            raise F2aArmError("training and evaluation feature namespaces differ")
        evaluation = V5.bind_exact_evaluation_rows(
            arm_table, reference, horizon, model_columns,
        )
        for model in V5.MODELS:
            name = f"{ARM}_{model}_h{horizon}"
            path = shard_dir / f"{name}.parquet"
            if path.exists():
                print(f"  skip {name}", flush=True)
                continue
            cell = V5.Cell(FEATURE_ARM, model, horizon)
            # train on realized temperature, predict on the forecast
            shard, fit_evidence = V5.fit_tree_cell(
                train_table, evaluation, model_columns, preprocessing, cell,
            )
            shard = V5.validate_v5_shard(shard, cell, reference)
            shard.to_parquet(path, index=False)
            manifest.append({
                "arm": ARM, "model": model, "horizon": horizon,
                "shard": path.name, "rows": len(shard),
                "forecast_coverage": coverage,
                "trained_on": "F3_temperature_only (realized)",
                "predicted_on": "F2a fixed-lead forecast",
                "fit": fit_evidence,
            })
            print(f"  {name}: {len(shard)} rows", flush=True)

    (OUTPUT_DIR / MANIFEST_FILENAME).write_text(
        json.dumps({
            "format": "thermoroute.forcing-f2a-v5a.v1",
            "status": "SEALED_PRE_OUTCOME_F2A_ARM",
            "semantics": {
                "fixed_lead_composite": True,
                "coherent_single_initialization": False,
                "as_issued_operational_archive": False,
                "only_legal_denominator": "F3_temperature_only "
                                          "(= the only_air_temperature arm)",
                "forbidden": ["operational recovery fraction",
                              "recovery against F3_full"],
            },
            "execution_authority": authority,
            "cells": manifest,
        }, sort_keys=True, indent=1, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "WRITTEN", "cells": len(manifest)},
                     sort_keys=True, indent=1))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-dirty-tree", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"arm": ARM, "cells": len(V5.MODELS) * len(V5.HORIZONS),
                          "denominator": "only_air_temperature"}, indent=1))
        return 0
    return execute(require_clean_tree=not args.allow_dirty_tree)


if __name__ == "__main__":
    raise SystemExit(main())

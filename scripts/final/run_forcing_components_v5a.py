#!/usr/bin/env python3
"""Which meteorological variable carries the forcing value (protocol v5a).

The shuffle and shift arms established that the F0/F3 gain is event-scale
weather information tied to exact dates.  Neither says *which* weather.  These
arms answer that, under the same seal.

Definition, and it needs stating because "single-component" is ambiguous.  A
component arm gives the selected variable its realized future values and gives
every other meteorological variable its *training climatology* at the same
valid times.  So exactly one variable carries event-scale future information
and the rest carry only the seasonal cycle they would have contributed anyway.
That is the protocol's F1 negative control applied to the complement of the
selected set, and it keeps the feature namespace and the substitution
accounting identical across arms -- a component arm that simply dropped the
other variables would confound "which variable matters" with "how many columns
the model has".

Two families, because neither alone is honest about correlated predictors:

* ``only_X``       -- X realized, everything else climatological
* ``without_X``    -- X climatological, everything else realized

Single-component values do not sum to the full-forcing value and their
differences are not a variance decomposition; air temperature, radiation and
humidity move together.  The two families are reported side by side and the gap
between them is the thing to read.
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

import scripts.final.run_forcing_ladder_v5_observed as V5
import scripts.final.run_forcing_placebo_v5a_shuffle as SHUF
from thermoroute import config as C

VARIABLES = V5.METEOROLOGY_VARIABLES          # TEMP, PRCP, DH, RHMEAN, WDSP
FEATURE_ARM = "F3_full"
OUTPUT_DIR = V5.FINAL_OUTPUT_ROOT / "forcing_components_v5a"
SHARD_DIRNAME = "component_shards_v5a"
MANIFEST_FILENAME = "component_lineage_manifest_v5a.json"

#: Humidity and wind are declared as one group in the sealed protocol, so they
#: are held together rather than split into two arms.
GROUPS: dict[str, tuple[str, ...]] = {
    "air_temperature": ("TEMP",),
    "radiation": ("DH",),
    "precipitation": ("PRCP",),
    "humidity_wind": ("RHMEAN", "WDSP"),
}


class ComponentError(RuntimeError):
    """A component precondition failed."""


def arms() -> list[tuple[str, tuple[str, ...]]]:
    """(arm name, variables that receive realized future values)."""
    out: list[tuple[str, tuple[str, ...]]] = []
    for name, members in GROUPS.items():
        out.append((f"only_{name}", members))
    for name, members in GROUPS.items():
        realized = tuple(v for v in VARIABLES if v not in members)
        out.append((f"without_{name}", realized))
    return out


def climatological_panel(
    raw_frame: pd.DataFrame,
    climatologies: dict[str, Any],
    realized: Sequence[str],
) -> pd.DataFrame:
    """Replace the non-selected variables with their training climatology.

    Only the meteorology moves.  Water temperature, discharge and every
    identity column are untouched, and the observedness masks follow the values
    so the raw-panel invariant holds by construction.
    """
    keep = set(realized)
    missing = keep - set(VARIABLES)
    if missing:
        raise ComponentError(f"unknown meteorological variable(s): {sorted(missing)}")
    out = raw_frame.copy(deep=True)
    for variable in VARIABLES:
        if variable in keep:
            continue
        values = np.empty(len(out), dtype=float)
        for station, block in out.groupby("site_id", sort=False):
            values[block.index.to_numpy()] = climatologies[variable].predict_dates(
                station, block["DATE"]
            )
        out[variable] = values
        out[f"{variable}_observed"] = np.isfinite(values)
    return out


def execute(*, require_clean_tree: bool = True) -> int:
    authority = SHUF.build_execution_authority(require_clean_tree=require_clean_tree)
    authority["arm"] = [name for name, _ in arms()]
    authority["runner_sha256"] = SHUF._sha256_file(Path(__file__).resolve())

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
        for arm_name, realized in arms():
            frame = climatological_panel(
                true_frame, preprocessing.meteorology_climatologies, realized
            )
            content = V5._canonical_frame_sha256(frame, V5.RAW_PANEL_COLUMNS)
            source = hashlib.sha256(
                f"component|{arm_name}|{content}".encode()
            ).hexdigest()
            panel = V5.make_raw_label_panel(frame, source_sha256=source)
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
                name = f"{arm_name}_{model}_h{horizon}"
                path = shard_dir / f"{name}.parquet"
                if path.exists():
                    print(f"  skip {name}", flush=True)
                    continue
                cell = V5.Cell(FEATURE_ARM, model, horizon)
                shard, fit_evidence = V5.fit_tree_cell(
                    arm_table, evaluation, model_columns, preprocessing, cell,
                )
                shard = V5.validate_v5_shard(shard, cell, reference)
                shard.to_parquet(path, index=False)
                manifest.append({
                    "arm": arm_name,
                    "realized_variables": list(realized),
                    "climatological_variables": [
                        v for v in VARIABLES if v not in set(realized)
                    ],
                    "model": model, "horizon": horizon,
                    "shard": path.name, "rows": len(shard),
                    "fit": fit_evidence,
                })
                print(f"  {name}: {len(shard)} rows", flush=True)

    (OUTPUT_DIR / MANIFEST_FILENAME).write_text(
        json.dumps({
            "format": "thermoroute.forcing-components-v5a.v1",
            "status": "SEALED_PRE_OUTCOME_COMPONENT_ARM",
            "definition": (
                "the selected variable receives realized future values; every "
                "other meteorological variable receives its training "
                "climatology at the same valid times"
            ),
            "non_additivity": (
                "single-component values do not sum to the full-forcing value "
                "and their differences are not a variance decomposition"
            ),
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
        print(json.dumps({
            "arms": [name for name, _ in arms()],
            "cells": len(arms()) * len(V5.MODELS) * len(V5.HORIZONS),
        }, sort_keys=True, indent=1))
        return 0
    return execute(require_clean_tree=not args.allow_dirty_tree)


if __name__ == "__main__":
    raise SystemExit(main())

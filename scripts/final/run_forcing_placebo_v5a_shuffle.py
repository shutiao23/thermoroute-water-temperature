#!/usr/bin/env python3
"""Run the within-station-month shuffled-forcing placebo (protocol v5a, arm F3_shuffle_month).

The v5 result says realized future meteorology is worth 0.13 degC at one day and
0.54-0.63 degC at three and seven days.  That number is only an *event-scale*
information value if it disappears when the correspondence between a forecast
key and the weather that actually followed it is destroyed while everything
else about the weather is held fixed.  This arm destroys exactly that.

How the placebo is built, and why this way.  ``RawFutureRegistry`` exists to
mean "the exact realized future", so it is the wrong object to fill with
permuted values.  Instead the permutation is applied upstream, to the
meteorology columns of the raw panel that the future lookup reads, and the
registry is then built through the ordinary public factory.  Every validator in
the v5 path -- observedness equals isfinite, one row per station-day, exact
future identity, substitution accounting, no imputed training label, disjoint
train/validation -- therefore runs unchanged on the placebo, and no private
constructor is touched.

The base feature table, the preprocessing, the imputer, the water climatology,
the damped anchor and the meteorology climatologies all come from the *true*
panel, so issue-time information is bit-identical to the reference run and the
future forcing is the only thing that moves.

Stratum.  The sealed protocol says ``[site_id, target_month]``.  That admits two
readings: month-of-year pooled across years, or a specific year-month.  This
runner uses the specific year-month, which is the stricter control: donors come
from the same month of the same year, so the placebo keeps not only the station
climate and the seasonal phase but the realized conditions of that particular
month, and only the day-to-day correspondence is destroyed.  Pooling across
years would scramble more and make the finding look better.

What this arm can and cannot show.  A within-month derangement preserves each
station-month's mean exactly, so the seasonal cycle survives essentially
untouched -- on a seasonally dominated series the raw lag-1 autocorrelation
barely moves, while the anomaly about the stratum mean decorrelates completely
(see ``tests/final/test_forcing_placebo_v5a_shuffle.py``).  The quantity this
arm removes is therefore the *within-month, day-to-day* correspondence, which
is exactly the part of future weather a forecast key can be said to correspond
to.  It follows that a placebo value near the true value would mean the F3 gain
is seasonal-phase information, and a placebo value near zero would mean it is
day-to-day weather information.  It does not, on its own, separate day-to-day
weather from sub-monthly synoptic persistence; the +/-7-day shift arm is the
control for that.

Chronology.  Protocol v5a was sealed at
``protocols/wrr_forcing_placebo_protocol_v5a_seal.json`` before any placebo
outcome existed; this runner verifies that seal and refuses to start if the
protocol bytes or the decision rules have moved.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import subprocess
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
import scripts.final.seal_forcing_placebo_protocol_v5a as SEALER
from thermoroute import config as C

PLACEBO_ARM = "F3_shuffle_month"
SHUFFLE_SEEDS = (0, 1, 2, 3, 4)
FEATURE_ARM = "F3_full"  # the placebo shares F3's feature namespace exactly

OUTPUT_DIR = V5.FINAL_OUTPUT_ROOT / "forcing_placebo_v5a"
SHARD_DIRNAME = "placebo_shards_v5a_shuffle"
MANIFEST_FILENAME = "placebo_lineage_manifest_v5a_shuffle.json"

RUNTIME_PACKAGES = ("numpy", "pandas", "pyarrow", "lightgbm", "scikit-learn", "scipy")


class PlaceboError(RuntimeError):
    """A placebo precondition or invariant failed; nothing was written."""


# ---------------------------------------------------------------------------
# execution authority: bind the runner, the runtime and a clean commit
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    out = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def _runtime_identity() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for name in RUNTIME_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:  # pragma: no cover
            raise PlaceboError(f"required runtime package absent: {name}") from exc
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "packages": versions,
    }


def build_execution_authority(*, require_clean_tree: bool) -> dict[str, Any]:
    """Bind everything protocol v5a requires before the first placebo fit."""
    seal_state = SEALER.verify_seal()
    if seal_state["drifted_inputs"]:
        raise PlaceboError(
            f"sealed placebo inputs drifted: {sorted(seal_state['drifted_inputs'])}"
        )
    sealed = json.loads(SEALER.SEAL.read_text(encoding="utf-8"))
    if sealed["execution_authorized"] is not False:
        raise PlaceboError("the specification seal must not carry execution authority")

    dirty = _git("status", "--porcelain")
    if dirty and require_clean_tree:
        raise PlaceboError(
            "refusing to run a sealed arm from a dirty tree; commit first so the "
            f"design commit is bindable:\n{dirty}"
        )
    return {
        "authority_format": "thermoroute.forcing-placebo-v5a-shuffle-execution.v1",
        "specification_seal_sha256": _sha256_file(SEALER.SEAL),
        "protocol_sha256": sealed["protocol_sha256"],
        "sealed_decision_rules_sha256": sealed["sealed_decision_rules_sha256"],
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "v5_runner_sha256": _sha256_file(
            ROOT / "scripts" / "final" / "run_forcing_ladder_v5_observed.py"
        ),
        "design_commit": _git("rev-parse", "HEAD"),
        "design_tree_clean": dirty == "",
        "runtime": _runtime_identity(),
        "arm": PLACEBO_ARM,
        "shuffle_seeds": list(SHUFFLE_SEEDS),
        "pinned_input_sha256": dict(V5.PINNED_INPUT_SHA256),
        "stratum": "site_id x year-month of the valid date (stricter of the two sealed readings)",
    }


# ---------------------------------------------------------------------------
# the permutation
# ---------------------------------------------------------------------------


def permute_meteorology_within_station_month(
    frame: pd.DataFrame, *, seed: int
) -> pd.DataFrame:
    """Derange the meteorology vector across dates inside each station-month.

    The five meteorology variables move together as one vector, so within-day
    meteorological coherence survives and only the date a vector is attached to
    changes.  Observedness masks are recomputed from the permuted values, which
    is equivalent to permuting them alongside and keeps the raw-panel invariant
    ``*_observed == isfinite(value)`` true by construction.

    Strata of size one cannot be deranged and are left in place; they are
    counted and returned to the caller's ledger rather than silently kept.
    """
    out = frame.copy(deep=True)
    variables = list(V5.METEOROLOGY_VARIABLES)
    values = out[variables].to_numpy(copy=True)

    months = out["DATE"].dt.year.to_numpy() * 100 + out["DATE"].dt.month.to_numpy()
    sites = out["site_id"].to_numpy()
    order = np.lexsort((out["DATE"].to_numpy(), months, sites))

    rng = np.random.default_rng(seed)
    permuted = values.copy()
    singleton_rows = 0
    fixed_points = 0
    n_strata = 0

    start = 0
    while start < len(order):
        stop = start + 1
        key = (sites[order[start]], months[order[start]])
        while stop < len(order) and (sites[order[stop]], months[order[stop]]) == key:
            stop += 1
        block = order[start:stop]
        n_strata += 1
        size = len(block)
        if size == 1:
            singleton_rows += 1
            start = stop
            continue
        # derangement: resample until no row keeps its own vector
        for _ in range(64):
            candidate = rng.permutation(size)
            if not np.any(candidate == np.arange(size)):
                break
        else:  # pragma: no cover - astronomically unlikely for size >= 2
            candidate = np.roll(np.arange(size), 1)
        fixed_points += int(np.sum(candidate == np.arange(size)))
        permuted[block] = values[block[candidate]]
        start = stop

    out[variables] = permuted
    for variable in variables:
        out[f"{variable}_observed"] = np.isfinite(out[variable].to_numpy())
    out.attrs["placebo_ledger"] = {
        "strata": n_strata,
        "singleton_rows_left_in_place": singleton_rows,
        "residual_fixed_points": fixed_points,
    }
    return out


def build_shuffled_panel(raw_panel: Any, *, seed: int) -> tuple[Any, dict[str, Any]]:
    """Return a factory-issued raw panel whose future meteorology is deranged."""
    true_frame = V5._validated_raw_frame(raw_panel)
    permuted = permute_meteorology_within_station_month(true_frame, seed=seed)
    ledger = dict(permuted.attrs.get("placebo_ledger", {}))

    # honest provenance: the permuted panel is a different panel
    content = V5._canonical_frame_sha256(permuted, V5.RAW_PANEL_COLUMNS)
    source = hashlib.sha256(
        f"{PLACEBO_ARM}|seed={seed}|from={raw_panel.source_sha256}|content={content}".encode()
    ).hexdigest()
    panel = V5.make_raw_label_panel(permuted, source_sha256=source)

    # the permutation must not have touched identity, water temperature or flow
    for column in ("site_id", "DATE", "WTEMP", "FLOW"):
        left = true_frame[column].to_numpy()
        right = V5._validated_raw_frame(panel)[column].to_numpy()
        same = (
            np.array_equal(left, right)
            if column in ("site_id", "DATE")
            else np.array_equal(np.isnan(left), np.isnan(right))
            and np.allclose(left[~np.isnan(left)], right[~np.isnan(right)], rtol=0, atol=0)
        )
        if not same:
            raise PlaceboError(f"shuffle altered {column}, which must be held fixed")
    ledger["permuted_panel_content_sha256"] = content
    ledger["permuted_panel_source_sha256"] = source
    return panel, ledger


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------


def capture_placebo_inputs() -> dict[str, Any]:
    """Bind the six pinned data inputs under this protocol's own authority.

    The placebo deliberately does not call ``V5.capture_execution_inputs``.
    That function enforces the *v5* score-execution authority, which authorizes
    twelve specific v5 cells and, because the sealed v5 protocol document embeds
    the digests of the append-only governance documents, can no longer be
    reproduced once a DLOG entry is appended (see the DLOG-027 addendum).
    Borrowing it would also mean running one protocol's arm under another
    protocol's authorization.

    What is load-bearing for a placebo is that the *data* is the same data the
    reference arm used.  That is exactly what this checks: each pinned input is
    read through the v5 stable-file reader and its digest must equal the v5
    literal pin, so the placebo and the reference cannot silently diverge on
    inputs.
    """
    captured: dict[str, Any] = {}
    for role, path in V5.PINNED_INPUT_PATHS.items():
        bound = V5._read_stable_regular(path, root=ROOT, label=f"placebo input {role}")
        expected = V5.PINNED_INPUT_SHA256[role]
        if bound.sha256 != expected:
            raise PlaceboError(
                f"placebo input {role} differs from the reference run: "
                f"expected {expected}, observed {bound.sha256}"
            )
        captured[role] = bound
    return captured


def shard_filename(model: str, horizon: int, seed: int) -> str:
    return f"{PLACEBO_ARM}_{model}_h{horizon}_seed{seed}_v5a.parquet"


def execute(*, require_clean_tree: bool = True) -> int:
    authority = build_execution_authority(require_clean_tree=require_clean_tree)

    snapshot = capture_placebo_inputs()
    raw, stations, _registry = V5._load_raw_panel_from_bounds(snapshot)
    reference = V5.validate_reference_registry(
        pd.read_parquet(io.BytesIO(snapshot["primary_key_registry"].payload)),
        require_production_counts=True,
    )
    C.STATIONS = stations
    # preprocessing, imputation and every issue-time feature come from the TRUE
    # panel: the placebo moves future forcing only.
    preprocessing = V5.fit_observed_preprocessing(raw, stations)
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
            for seed in SHUFFLE_SEEDS:
                panel, ledger = build_shuffled_panel(raw, seed=seed)
                registry = V5.build_raw_future_registry(
                    panel, base_table[["site_id", "issue_date"]], horizon,
                )
                arm_table, model_columns = V5.materialize_forcing_features(
                    base_table,
                    base_columns,
                    arm=FEATURE_ARM,
                    horizon=horizon,
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
                    relative = f"{SHARD_DIRNAME}/{shard_filename(model, horizon, seed)}"
                    binding = transaction.write_parquet(relative, frame)
                    expected_files[relative] = dict(binding)
                    cell_manifest.append({
                        "placebo_arm": PLACEBO_ARM,
                        "feature_arm": FEATURE_ARM,
                        "model": model,
                        "horizon": horizon,
                        "shuffle_seed": seed,
                        "shard": {"path": relative, "rows": len(frame), **binding},
                        "permutation_ledger": ledger,
                        "future_registry_identity_sha256": registry.identity_sha256,
                        "future_registry_content_sha256": registry.content_sha256,
                        "fit": fit_evidence,
                    })
                    print(
                        f"  {PLACEBO_ARM} {model} h{horizon} seed{seed}: "
                        f"{len(frame)} rows",
                        flush=True,
                    )

        manifest = {
            "format": "thermoroute.forcing-placebo-v5a-shuffle-lineage.v1",
            "status": "SEALED_PRE_OUTCOME_PLACEBO_ARM",
            "execution_authority": authority,
            "reference_run": V5.DEFAULT_OUTPUT_DIR.name,
            "cells": cell_manifest,
        }
        payload = json.dumps(manifest, sort_keys=True, indent=1, allow_nan=False).encode()
        expected_files[MANIFEST_FILENAME] = dict(
            transaction.write_bytes(MANIFEST_FILENAME, payload)
        )
        transaction.commit(expected_files)

    print(json.dumps({
        "status": "WRITTEN",
        "out_dir": str(OUTPUT_DIR),
        "shards": len(cell_manifest),
    }, sort_keys=True, indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="fit and write the arm")
    parser.add_argument(
        "--allow-dirty-tree", action="store_true",
        help="bind a dirty design commit (development only; recorded in the manifest)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        print(json.dumps(
            build_execution_authority(require_clean_tree=not args.allow_dirty_tree),
            sort_keys=True, indent=1,
        ))
        return 0
    return execute(require_clean_tree=not args.allow_dirty_tree)


if __name__ == "__main__":
    raise SystemExit(main())

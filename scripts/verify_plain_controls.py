#!/usr/bin/env python
"""G15: verify the plain-control preprocessing borrow on 2019-2020 dev test.

The two plain neural controls (``PlainMLP-7var`` / ``PlainCausalTCN-7var``) have
no inference bundle, so the conventional holdout scorer borrows the frozen
stage-09 preprocessing from the same-variable ThermoRoute bundle
(``FI.build_frozen_confirmation_windows``).  This script proves that borrow is
faithful: it rebuilds the 2019-2020 confirmation windows from the frozen
ThermoRoute transforms, loads each plain-control checkpoint ``best_model_state``,
runs inference, and compares the per-seed predictions to the stored
``arm_predictions/<arm>/seed*.parquet`` **test** rows (the already-inspected
2019-2020 development evaluation).

Comparison is per ``(site_id, horizon, issue_date, target_date)`` forecast key
on ``y_pred / q05 / q50 / q95 / p_exceed`` at ``atol = 1e-2``.  An arm passes
only if every dev-test key is reproduced (``n_common == n_dev_test``) and every
metric's ``max_abs_diff <= atol``.  On failure the arm is recorded as
``EXCLUDED_PREPROCESSING_MISMATCH`` (the holdout scorer must then exclude it)
rather than aborting, but the report states the conclusion unambiguously.

Environment: ``/home/lzq/anaconda3/envs/route-a/bin/python`` with
``PYTHONPATH=src`` and ``THERMOROUTE_FORMAL_THREADS=8`` (plus the OMP/MKL/
OPENBLAS/NUMEXPR caps).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from thermoroute import config as C  # noqa: E402
from thermoroute import checkpoint as CKPT  # noqa: E402
from thermoroute import conventional_score as CS  # noqa: E402
from thermoroute import frozen_inference as FI  # noqa: E402
from thermoroute.plain_controls import (  # noqa: E402
    CONTROL_SEEDS,
    PLAIN_CONTROL_ARMS,
)

DEFAULT_RUN_ID = "e87f141ad92c33ce1d3f"
THERMOROUTE_BUNDLE = "thermoroute_usgs_bundle_665afcb4d674161943ae"
DEV_INTERVAL = ("2019-01-01", "2020-12-31")
COMPARE_METRICS = ("y_pred", "q05", "q50", "q95", "p_exceed")
MERGE_KEYS = ["site_id", "horizon", "issue_date", "target_date"]


def log(msg: str) -> None:
    print(f"[g15] {msg}", flush=True)


def load_registry(registry_path: Path) -> pd.DataFrame:
    reg = pd.read_csv(registry_path, dtype={"site_no": str, "legacy_site_id": str})
    reg = reg[["site_no", "legacy_site_id", "lat", "lon"]].copy()
    reg["site_no"] = reg["site_no"].str.strip()
    return reg


def load_dev_panel_mapped(panel_path: Path, registry_path: Path) -> pd.DataFrame:
    panel = pd.read_parquet(panel_path)
    reg = load_registry(registry_path)
    mapping = dict(zip(reg["legacy_site_id"], reg["site_no"]))
    panel["site_id"] = panel["site_id"].map(mapping).astype(str)
    panel["DATE"] = pd.to_datetime(panel["DATE"])
    return panel


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--multicore", default=str(REPO.parent / "thermoroute-water-temperature-multicore"),
                   help="sibling multicore worktree root (holds outputs/models and outputs/runs)")
    p.add_argument("--run-id", default=DEFAULT_RUN_ID,
                   help="Stage-09b run id holding the plain-control checkpoints + arm_predictions")
    p.add_argument("--panel", default=str(REPO / "data_usgs" / "panel_usgs_120v2.parquet"))
    p.add_argument("--registry", default=str(REPO / "data_usgs" / "station_registry_v1.csv"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--atol", type=float, default=1e-2)
    p.add_argument("--out", default=str(REPO / "outputs" / "conventional" / "plain_controls_g15.json"))
    p.add_argument("--arms", nargs="*", default=list(PLAIN_CONTROL_ARMS))
    args = p.parse_args(argv)

    multicore = Path(args.multicore)
    run_dir = multicore / "outputs" / "runs" / "09b_development_controls" / args.run_id
    checkpoint_dir = run_dir / "checkpoints"
    arm_pred_dir = run_dir / "arm_predictions"
    tr_bundle = multicore / "outputs" / "models" / THERMOROUTE_BUNDLE
    for label, path in (("checkpoint_dir", checkpoint_dir),
                        ("arm_predictions", arm_pred_dir),
                        ("thermoroute_bundle", tr_bundle)):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    log("loading dev panel + registry ...")
    panel = load_dev_panel_mapped(Path(args.panel), Path(args.registry))
    station_ids = sorted(panel["site_id"].astype(str).unique())

    log(f"building frozen confirmation windows over {DEV_INTERVAL[0]}..{DEV_INTERVAL[1]} "
        f"(preprocessing borrow from {THERMOROUTE_BUNDLE}) ...")
    _weights, tr_meta = CKPT.load_inference_bundle(tr_bundle)
    if list(tr_meta.get("feature_order", [])) != [
        "WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP",
    ]:
        raise RuntimeError(
            "ThermoRoute bundle feature_order is not the 7 plain-control variables; "
            "the preprocessing borrow premise is invalid"
        )
    wd, _transforms, _imputed = FI.build_frozen_confirmation_windows(
        panel, tr_meta, station_ids, interval=DEV_INTERVAL, external=False)
    station_names = tuple(C.STATIONS)
    log(f"  built {len(wd.X)} confirmation windows; {len(station_names)} stations")

    report: dict = {
        "gate": "G15_plain_control_preprocessing_borrow",
        "run_id": args.run_id,
        "interval": f"{DEV_INTERVAL[0]}..{DEV_INTERVAL[1]}",
        "borrowed_bundle": THERMOROUTE_BUNDLE,
        "atol": args.atol,
        "n_stations": len(station_names),
        "arms": {},
    }
    overall_pass = True
    for arm in args.arms:
        log(f"=== {arm} ===")
        ens_frame, member_frames, meta = CS.plain_control_ensemble(
            checkpoint_dir, wd, station_names, arm,
            device=args.device, split="confirm", scope="development_only_2006_2020",
            feature_set="all_7_variables", n_stations=len(station_names),
            seeds=CONTROL_SEEDS, expected_run_id=args.run_id,
        )
        log(f"  trainable_parameters={meta.get('trainable_parameters')} "
            f"n_members={meta.get('member_count')}")
        # Load the stored dev-test reference (test split) for every seed.
        ref_parts = []
        for seed in CONTROL_SEEDS:
            ref = pd.read_parquet(arm_pred_dir / arm / f"seed{seed}.parquet")
            ref_parts.append(ref[ref["split"] == "test"].copy())
        reference = pd.concat(ref_parts, ignore_index=True)
        diag = CS.compare_predictions_to_reference(
            member_frames, reference, atol=args.atol,
            metrics=COMPARE_METRICS, require_all_keys=True)
        arm_pass = bool(diag["match"])
        arm_report: dict = {
            "trainable_parameters": meta.get("trainable_parameters"),
            "n_members": meta.get("member_count"),
            "seeds": {},
            "status": "PASS" if arm_pass else "EXCLUDED_PREPROCESSING_MISMATCH",
        }
        for entry in diag["per_seed"]:
            seed = int(entry["seed"])
            seed_ok = bool(entry.get("within_atol") and entry.get("all_keys_reproduced"))
            entry["status"] = "PASS" if seed_ok else "EXCLUDED_PREPROCESSING_MISMATCH"
            arm_report["seeds"][str(seed)] = entry
            log(f"  seed{seed}: n_common={entry.get('n_common')}/{entry.get('n_dev')} "
                f"overall_max_abs_diff={entry.get('overall_max_abs_diff', float('nan')):.6g} "
                f"-> {entry['status']}")
        if not arm_pass:
            overall_pass = False
        report["arms"][arm] = arm_report

    report["overall_status"] = "PASS" if overall_pass else "EXCLUDED_PREPROCESSING_MISMATCH"
    report["conclusion"] = (
        "The stage-09 preprocessing borrow (frozen ThermoRoute transforms) reproduces "
        "every plain-control dev-test prediction within tolerance; plain controls are "
        "admitted to the conventional holdout."
        if overall_pass else
        "The preprocessing borrow does NOT reproduce the stored plain-control dev-test "
        "predictions; the failing arm(s) are EXCLUDED from the conventional holdout "
        "(EXCLUDED_PREPROCESSING_MISMATCH) and must not be published as verified."
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log(f"wrote {out_path}")
    log(f"OVERALL: {report['overall_status']}")
    return 0 if overall_pass else 2


if __name__ == "__main__":
    sys.exit(main())

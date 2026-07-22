#!/usr/bin/env python3
"""Stage 23 — predeclared Route-A input-stress and OOD stratified evaluation.

This is a development-period stress test, not a second blind test and not a
counterfactual climate-impact model.  It evaluates the frozen ThermoRoute
ensemble under input defects/shifts while keeping every outcome and forecast key
fixed.  Positive ``delta_rmse`` means degradation relative to clean inputs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from thermoroute import config as C
from thermoroute import data as D
from thermoroute import datasets as DS
from thermoroute import features as F
from thermoroute.checkpoint import instantiate_inference_ensemble
from thermoroute.repro import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_parquet,
    cache_is_valid,
    initialise_run_directory,
    resolve_run_identity,
    seal_artifact,
    sha256_file,
    sidecar_path,
)
from thermoroute.robustness import (
    ROBUSTNESS_SCHEMA_VERSION,
    PerturbationSpec,
    build_outcome_strata,
    enforce_common_robustness_keys,
    predict_perturbation,
    route_a_perturbation_ladder,
    summarise_degradation,
)
from thermoroute.spatial import huc2_cluster_map, load_station_registry
from thermoroute.thermoroute import ThermoRoute


USGS_VARS = ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP")
DEFAULT_PANEL = ROOT / "data_usgs" / "panel_usgs_120v2.parquet"
DEFAULT_REGISTRY = ROOT / "data_usgs" / "station_registry_v1.csv"
DEFAULT_BUNDLE_POINTER = C.MODELS / "thermoroute_usgs_bundle.json"
DEFAULT_CANONICAL_PREDICTIONS = C.PREDICTIONS / "usgs_predictions_stage9_v2.parquet"
PREDICTION_CONTENT_SCHEMA = f"{ROBUSTNESS_SCHEMA_VERSION}.predictions"
_START = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - _START:7.1f}s] {message}", flush=True)


def _load_bundle_pointer(path: Path) -> tuple[Path, dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"inference bundle pointer not found: {path}. Run Stage 9 first.")
    pointer = json.loads(path.read_text())
    required = {"run_id", "bundle_path", "member_count", "metadata_sha256",
                "weights_sha256"}
    missing = sorted(required - set(pointer))
    if missing:
        raise ValueError(f"bundle pointer missing fields: {missing}")
    directory = ROOT / pointer["bundle_path"]
    if not directory.is_dir():
        raise FileNotFoundError(f"bundle directory not found: {directory}")
    if sha256_file(directory / "metadata.json") != pointer["metadata_sha256"]:
        raise ValueError("bundle metadata does not match its frozen pointer")
    if sha256_file(directory / "weights.pt") != pointer["weights_sha256"]:
        raise ValueError("bundle weights do not match their frozen pointer")
    return directory, pointer


def _registry_maps(path: Path, stations: tuple[str, ...]) -> dict[str, str]:
    registry = load_station_registry(path)
    expected = set(stations)
    actual = set(registry.site_no)
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(f"station registry is missing canonical sites: {missing[:5]}")
    clusters = huc2_cluster_map(registry)
    return {site: clusters[site] for site in stations}


def _compare_float_map(actual: dict, expected: dict, label: str,
                       *, atol: float = 1e-7) -> None:
    if set(actual) != set(expected):
        raise ValueError(f"bundle {label} keys do not match reconstructed preprocessing")
    for key in actual:
        if not np.isclose(float(actual[key]), float(expected[key]), atol=atol, rtol=0):
            raise ValueError(f"bundle {label} mismatch at {key}")


def _verify_bundle_contract(metadata: dict, *, panel_path: Path,
                            registry_path: Path, wd, clim) -> None:
    """Bind weights to exact data, source, identifiers and train-fit transforms."""
    identity = {
        "panel_sha256": sha256_file(panel_path),
        "registry_sha256": sha256_file(registry_path),
    }
    for key, value in identity.items():
        if metadata.get(key) != value:
            raise ValueError(f"bundle {key} is not bound to the requested Route-A input")
    station_to_index = {station: i for i, station in enumerate(C.STATIONS)}
    if metadata.get("station_to_index") != station_to_index:
        raise ValueError("bundle station_to_index does not match stable site_no order")
    if metadata.get("feature_order") != list(wd.var_names):
        raise ValueError("bundle feature order does not match reconstructed windows")
    if metadata.get("horizons") != list(wd.horizons):
        raise ValueError("bundle horizons do not match Route-A horizons")
    preprocessing = metadata.get("preprocessing", {})
    input_schema = preprocessing.get("input_schema", {})
    expected_schema = {
        "variables": list(wd.var_names),
        "physics_forcings": list(wd.phys_vars),
        "context_length": int(wd.X.shape[1]),
        "transforms": {
            variable: ("signed_log1p" if variable == "FLOW" else "log1p_nonnegative")
            for variable in C.LOG1P_VARS if variable in set(wd.var_names)
        },
        "missingness_mask": True,
    }
    if input_schema != expected_schema:
        raise ValueError("bundle input schema does not match reconstructed preprocessing")
    scaler = preprocessing.get("scaler", {})
    expected_mean = {
        f"{station}|{variable}": value
        for (station, variable), value in wd.scaler.mean.items()
        if variable in set(wd.var_names)
    }
    expected_std = {
        f"{station}|{variable}": value
        for (station, variable), value in wd.scaler.std.items()
        if variable in set(wd.var_names)
    }
    _compare_float_map(scaler.get("mean", {}), expected_mean, "scaler mean")
    _compare_float_map(scaler.get("std", {}), expected_std, "scaler std")
    _compare_float_map(
        preprocessing.get("damped_anchor", {}).get("phi", {}),
        wd.damped_anchor.phi, "damped-anchor phi")
    stored_clim = preprocessing.get("climatology", {}).get("coefficients", {})
    if set(stored_clim) != set(clim.coef):
        raise ValueError("bundle climatology station registry mismatch")
    for station, coefficients in clim.coef.items():
        if not np.allclose(np.asarray(stored_clim[station], float), coefficients,
                           atol=1e-7, rtol=0):
            raise ValueError(f"bundle climatology mismatch at {station}")


def _model_factory(_member_name: str, metadata: dict) -> ThermoRoute:
    architecture = metadata.get("architecture", {})
    if architecture.get("class") != "thermoroute.thermoroute.ThermoRoute":
        raise ValueError("unsupported bundle architecture")
    kwargs = dict(architecture.get("kwargs", {}))
    cfg = C.TrainConfig(**architecture.get("train_config", {}))
    return ThermoRoute(
        horizons=tuple(int(value) for value in metadata["horizons"]),
        cfg=cfg,
        **kwargs,
    )


def _ensemble_condition(models: dict[str, torch.nn.Module], wd, indices: np.ndarray,
                        spec: PerturbationSpec, *, device: str, batch_size: int,
                        base_seed: int) -> pd.DataFrame:
    members = []
    for ordinal, (member_name, model) in enumerate(sorted(models.items())):
        try:
            member_seed = int(member_name.removeprefix("seed"))
        except ValueError:
            member_seed = ordinal
        members.append(predict_perturbation(
            model, wd, indices, spec, device=device, batch_size=batch_size,
            base_seed=base_seed, model_name="ThermoRoute", model_seed=member_seed,
        ))
    raw = pd.concat(members, ignore_index=True)
    keys = ["condition_id", "scenario", "severity", "severity_unit",
            "site_id", "horizon", "issue_date", "target_date"]
    truth = raw.groupby(keys).y_true.agg(["min", "max"])
    if not np.allclose(truth["min"], truth["max"], atol=1e-8, rtol=0):
        raise AssertionError("ensemble members disagree on y_true")
    ensemble = raw.groupby(keys, as_index=False).agg(
        y_true=("y_true", "first"), y_pred=("y_pred", "mean"),
        member_count=("seed", "nunique"),
    )
    if not ensemble.member_count.eq(len(models)).all():
        raise AssertionError("ensemble condition is missing a member on some keys")
    ensemble["model"] = "ThermoRoute-ensemble"
    ensemble["seed"] = -1
    return ensemble


def _verify_clean_prediction_parity(clean: pd.DataFrame, canonical_path: Path, *,
                                    bundle_run_id: str, member_count: int,
                                    atol: float = 5e-5) -> None:
    """Prove that bundle inference reproduces the frozen Stage-9 ensemble mean."""
    if not canonical_path.is_file() or not sidecar_path(canonical_path).is_file():
        raise FileNotFoundError(
            "canonical Stage-9 predictions and lineage sidecar are required for parity")
    lineage = json.loads(sidecar_path(canonical_path).read_text())
    if lineage.get("artifact_sha256") != sha256_file(canonical_path):
        raise ValueError("canonical predictions no longer match their lineage checksum")
    if lineage.get("artifact_bytes") != canonical_path.stat().st_size:
        raise ValueError("canonical predictions no longer match their lineage byte count")
    if lineage.get("run", {}).get("run_id") != bundle_run_id:
        raise ValueError("canonical prediction run_id differs from the inference bundle")
    raw = pd.read_parquet(canonical_path)
    raw = raw[(raw.model == "ThermoRoute") & (raw.split == "test")].copy()
    if raw.empty:
        raise ValueError("canonical predictions contain no ThermoRoute test rows")
    keys = ["site_id", "horizon", "issue_date", "target_date"]
    for column in ("issue_date", "target_date"):
        raw[column] = pd.to_datetime(raw[column])
    truth = raw.groupby(keys).y_true.agg(["min", "max"])
    if not np.allclose(truth["min"], truth["max"], atol=1e-8, rtol=0):
        raise ValueError("canonical ensemble members disagree on y_true")
    canonical = raw.groupby(keys, as_index=False).agg(
        y_true=("y_true", "first"), y_pred=("y_pred", "mean"),
        member_count=("seed", "nunique"),
    )
    if not canonical.member_count.eq(member_count).all():
        raise ValueError("canonical predictions do not contain every bundle member")
    candidate = clean.copy()
    for column in ("issue_date", "target_date"):
        candidate[column] = pd.to_datetime(candidate[column])
    paired = canonical.merge(
        candidate[keys + ["y_true", "y_pred"]], on=keys, how="outer",
        suffixes=("_canonical", "_bundle"), indicator=True, validate="one_to_one")
    if not paired._merge.eq("both").all():
        raise ValueError("bundle clean inference does not share canonical forecast keys")
    if not np.allclose(paired.y_true_canonical, paired.y_true_bundle,
                       atol=1e-8, rtol=0):
        raise ValueError("bundle clean inference disagrees with canonical y_true")
    difference = np.abs(paired.y_pred_canonical - paired.y_pred_bundle)
    if not np.isfinite(difference).all() or float(difference.max()) > atol:
        raise ValueError(
            f"bundle clean prediction parity failed (max |difference|="
            f"{float(difference.max()):.3g}, tolerance={atol:g})")


def _training_thresholds(panel: pd.DataFrame, train_mask: np.ndarray
                         ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    train = panel.loc[train_mask]
    heat, low, high = {}, {}, {}
    for station in C.STATIONS:
        group = train[train.site_id == station]
        water = pd.to_numeric(group.WTEMP, errors="coerce").dropna()
        flow = pd.to_numeric(group.FLOW, errors="coerce").dropna()
        if water.empty or flow.empty:
            raise ValueError(f"insufficient training observations for strata at {station}")
        heat[station] = float(water.quantile(C.EXCEEDANCE_QUANTILE))
        low[station] = float(flow.quantile(0.10))
        high[station] = float(flow.quantile(0.90))
    return heat, low, high


def _write_report(summary: pd.DataFrame, path: Path, *, member_count: int,
                  n_keys: int, n_stations: int, run_id: str) -> None:
    primary = summary[(summary.stratum == "all") &
                      (summary.cluster_level == "huc2")].sort_values(
                          ["scenario", "severity", "horizon"])
    lines = [
        "# Route-A input robustness and OOD stratification",
        "",
        f"Run `{run_id}`; {member_count}-member ensemble mean; {n_stations} stable "
        f"USGS site numbers; {n_keys:,} common forecast keys per condition.",
        "",
        "This is a previously inspected 2019–2020 development evaluation. Positive "
        "ΔRMSE means worse performance under stress. Perturbations modify only "
        "issue-time/history inputs; y and forecast keys are fixed. Air/flow shifts "
        "are sensitivity probes, not causal climate projections.",
        "Missingness targets optional forcing channels and preserves the mandatory "
        "issue-time WTEMP safety anchor. Sensor noise is Gaussian in each feature's "
        "frozen train-standardised space (after the declared log or signed-log transform); noisy issue "
        "WTEMP is propagated into the damped anchor rather than leaving inconsistent "
        "side inputs. TEMP shifts are additive train-SD offsets, while FLOW shifts "
        "are multipliers in original signed physical flow units (legitimate "
        "reverse-flow observations remain negative).",
        "",
        "Primary aggregation is the median paired station RMSE difference. Confidence "
        "intervals below resample complete HUC2 groups; the companion CSV also reports "
        "station-cluster intervals. Sites without verified HUC2 are separate "
        "`UNMAPPED:<site_no>` clusters. These intervals quantify between-site/basin "
        "sampling uncertainty; they are not daily time-series confidence intervals.",
        "",
        "| scenario | severity | h | clean RMSE | stressed RMSE | ΔRMSE [95% CI] | "
        "relative Δ | sites | HUC2 clusters |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary.itertuples(index=False):
        lines.append(
            f"| {row.scenario} | {row.severity:g} {row.severity_unit} | "
            f"{int(row.horizon)} | {row.clean_rmse:.3f} | {row.stressed_rmse:.3f} | "
            f"{row.delta_rmse:+.3f} [{row.ci_low:+.3f}, {row.ci_high:+.3f}] | "
            f"{100 * row.relative_delta:+.1f}% | {int(row.n_stations)} | "
            f"{int(row.n_clusters)} |")
    lines.extend([
        "",
        "Conditional `heat_event`, `low_flow`, and `high_flow` results are retained "
        "in `usgs_robustness_summary_v1.csv`. Heat events use target WTEMP above the "
        "station's training q90 only for scoring; low/high flow use clean issue-time "
        "FLOW below/above station training q10/q90. These subsets are neither model "
        "inputs nor severity-selection criteria.",
    ])
    atomic_write_bytes(path, ("\n".join(lines) + "\n").encode())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path,
                        default=Path(os.environ.get("USGS_PANEL", DEFAULT_PANEL)))
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--bundle-pointer", type=Path, default=DEFAULT_BUNDLE_POINTER)
    parser.add_argument("--canonical-predictions", type=Path,
                        default=DEFAULT_CANONICAL_PREDICTIONS)
    parser.add_argument("--device", default="cpu",
                        help="torch device; CPU is the deterministic default")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=23001)
    args = parser.parse_args()
    if args.batch_size < 1 or args.n_bootstrap < 100:
        parser.error("batch-size must be positive and n-bootstrap must be >=100")
    panel_path = args.panel.resolve()
    registry_path = args.registry.resolve()
    canonical_predictions_path = args.canonical_predictions.resolve()
    if not canonical_predictions_path.is_file():
        raise FileNotFoundError(
            f"canonical predictions not found: {canonical_predictions_path}")
    bundle_directory, pointer = _load_bundle_pointer(args.bundle_pointer.resolve())

    prepared = D.prepare_dataset_from_panel(str(panel_path), stable_site_ids=True)
    panel = prepared["panel_raw"]
    panel_imputed = prepared["panel"]
    masks = prepared["masks"]
    stations = tuple(prepared["stations"])
    # USGS identifiers are stable digit strings, not uniformly eight digits.
    # The frozen cohort contains a legitimate 15-digit site number; registry
    # validation below is the authoritative identity contract.
    huc2 = _registry_maps(registry_path, stations)
    climatology = F.HarmonicClimatology.fit(panel, masks.train)
    wd = DS.build_windows(
        panel_imputed, masks, climatology, variables=USGS_VARS,
        require_observed_target=True,
    )

    models, metadata = instantiate_inference_ensemble(
        bundle_directory, model_factory=_model_factory,
        expected_member_count=int(pointer["member_count"]), device=args.device,
    )
    if metadata.get("run_id") != pointer["run_id"]:
        raise ValueError("bundle run_id does not match its frozen pointer")
    _verify_bundle_contract(
        metadata, panel_path=panel_path, registry_path=registry_path, wd=wd,
        clim=climatology,
    )

    ladder = route_a_perturbation_ladder()
    resolved_config = {
        "stage": "route_a_robustness",
        "schema": ROBUSTNESS_SCHEMA_VERSION,
        "development_split": C.SPLIT.test,
        "ensemble_bundle_run_id": metadata["run_id"],
        "ensemble_weights_sha256": metadata["weights_sha256"],
        "canonical_predictions_sha256": sha256_file(canonical_predictions_path),
        "member_count": len(models),
        "variables": USGS_VARS,
        "horizons": C.HORIZONS,
        "perturbations": [asdict(spec) for spec in ladder],
        "severity_definitions": {
            "missing_rate": (
                "Bernoulli probability per optional forcing sensor cell, keyed by "
                "site/observation-date/variable; issue WTEMP anchor preserved"),
            "missing_block": (
                "most-recent consecutive issue/history days removed for all optional "
                "forcing channels; issue WTEMP anchor preserved"),
            "sensor_noise": (
                "zero-mean Gaussian SD in frozen train-standardised feature space; "
                "FLOW uses signed-log1p and PRCP uses non-negative log1p"),
            "air_temperature_shift": "additive TEMP offset in train SD units",
            "flow_shift": "multiplicative FLOW factor in original signed physical units",
        },
        "strata": {
            "heat_event": "target WTEMP > station train q90",
            "low_flow": "clean issue-time FLOW <= station train q10",
            "high_flow": "clean issue-time FLOW >= station train q90",
        },
        "effect": "median station RMSE(stress)-RMSE(clean)",
        "cluster_bootstrap": ["station", "HUC2"],
        "uncertainty_scope": "between-station/basin; no daily time resampling",
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
    }
    run_identity = resolve_run_identity(
        root=ROOT, panel=panel_path, registry=registry_path, config=resolved_config)
    if metadata.get("source_sha256") != run_identity.source_sha256:
        raise ValueError(
            "inference bundle source hash differs from the robustness code tree; "
            "rerun Stage 9 before this formal evaluation")
    run_dir = initialise_run_directory(
        C.OUTPUTS / "runs", run_identity, resolved_config,
        provenance={
            "bundle_path": bundle_directory.relative_to(ROOT).as_posix(),
            "bundle_metadata_sha256": sha256_file(bundle_directory / "metadata.json"),
            "development_not_confirmatory": True,
            "labels_modified": False,
        },
    )
    atomic_write_json(run_dir / "robustness_protocol.json", resolved_config)

    test_index = wd.idx("test")
    log(f"verified bundle {metadata['run_id']} ({len(models)} members); "
        f"{len(test_index):,} windows, {len(ladder)} fixed conditions")
    condition_frames: list[pd.DataFrame] = []
    for position, spec in enumerate(ladder, start=1):
        cache = run_dir / "conditions" / f"{spec.condition_id.replace(':', '_')}.parquet"
        if cache_is_valid(cache, run_identity, schema=PREDICTION_CONTENT_SCHEMA):
            condition = pd.read_parquet(cache)
            if set(condition.condition_id) != {spec.condition_id}:
                raise ValueError(f"condition cache has wrong identity: {cache}")
            log(f"[{position:02d}/{len(ladder)}] verified cache {spec.scenario} "
                f"{spec.severity:g} {spec.severity_unit}")
        else:
            condition = _ensemble_condition(
                models, wd, test_index, spec, device=args.device,
                batch_size=args.batch_size, base_seed=args.seed,
            )
            atomic_write_parquet(condition, cache, index=False)
            seal_artifact(
                cache, run_identity, kind="robustness_condition_predictions",
                schema=PREDICTION_CONTENT_SCHEMA,
                extra={"condition": asdict(spec), "member_count": len(models)},
            )
            log(f"[{position:02d}/{len(ladder)}] completed {spec.scenario} "
                f"{spec.severity:g} {spec.severity_unit}")
        if spec.scenario == "clean":
            _verify_clean_prediction_parity(
                condition, canonical_predictions_path,
                bundle_run_id=metadata["run_id"], member_count=len(models))
            log("clean bundle inference matches canonical Stage-9 ensemble predictions")
        condition_frames.append(condition)

    predictions = pd.concat(condition_frames, ignore_index=True)
    audit = enforce_common_robustness_keys(predictions)
    heat, low_flow, high_flow = _training_thresholds(panel, masks.train)
    strata = build_outcome_strata(
        wd, test_index, heat_thresholds=heat, low_flow_thresholds=low_flow,
        high_flow_thresholds=high_flow,
    )
    summary, station_effects = summarise_degradation(
        predictions, strata, huc2_by_site=huc2, n_boot=args.n_bootstrap,
        seed=args.seed,
    )

    run_predictions = run_dir / "robustness_predictions.parquet"
    atomic_write_parquet(predictions, run_predictions, index=False)
    seal_artifact(
        run_predictions, run_identity, kind="robustness_predictions",
        schema=PREDICTION_CONTENT_SCHEMA,
        extra={"common_keys_per_condition": audit.n_common,
               "conditions": len(audit.conditions)},
    )
    alias_predictions = C.PREDICTIONS / "usgs_robustness_v1.parquet"
    summary_path = C.TABLES / "usgs_robustness_summary_v1.csv"
    station_effects_path = C.TABLES / "usgs_robustness_station_effects_v1.csv"
    protocol_path = C.REPORTS / "usgs_robustness_protocol_v1.json"
    report_path = C.REPORTS / "usgs_robustness_v1.md"
    atomic_write_parquet(predictions, alias_predictions, index=False)
    atomic_write_bytes(
        summary_path,
        summary.to_csv(index=False).encode(),
    )
    atomic_write_bytes(
        station_effects_path,
        station_effects.to_csv(index=False).encode(),
    )
    protocol_record = {
        **resolved_config,
        "run_identity": run_identity.as_dict(),
        "forecast_key": ["site_id", "horizon", "issue_date", "target_date"],
        "n_common_keys_per_condition": audit.n_common,
        "n_conditions": len(audit.conditions),
        "n_stations": len(stations),
        "huc2_verified_or_unmapped_clusters": len(set(huc2.values())),
        "artifact_run_directory": run_dir.relative_to(ROOT).as_posix(),
    }
    atomic_write_json(protocol_path, protocol_record)
    _write_report(
        summary, report_path,
        member_count=len(models), n_keys=audit.n_common, n_stations=len(stations),
        run_id=run_identity.run_id,
    )
    for artifact, kind, schema in (
        (alias_predictions, "robustness_predictions_alias", PREDICTION_CONTENT_SCHEMA),
        (summary_path, "robustness_summary", f"{ROBUSTNESS_SCHEMA_VERSION}.summary"),
        (station_effects_path, "robustness_station_effects",
         f"{ROBUSTNESS_SCHEMA_VERSION}.station-effects"),
        (protocol_path, "robustness_protocol", ROBUSTNESS_SCHEMA_VERSION),
        (report_path, "robustness_report", ROBUSTNESS_SCHEMA_VERSION),
    ):
        seal_artifact(artifact, run_identity, kind=kind, schema=schema)
    log(f"done: {audit.n_common:,} common keys/condition; outputs are labelled "
        "development stress evidence, not confirmation")


if __name__ == "__main__":
    main()

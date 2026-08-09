"""Synthetic tests for the forcing-regime whole-HUC2 inference authority."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import scripts.final.build_forcing_regime_inference_v4 as inference

PROTOCOL_SHA256 = "a" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sites() -> tuple[str, ...]:
    return tuple(f"{index:08d}" for index in range(1, inference.EXPECTED_STATIONS + 1))


def _synthetic_paired() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_index, model in enumerate(inference.MODELS):
        for horizon in inference.HORIZONS:
            for site_index, site in enumerate(_sites(), start=1):
                requested = 0.1 + model_index / 100.0 + horizon / 1000.0 + site_index / 100_000.0
                rmse_reference = 1.0
                rmse_arm = rmse_reference - requested
                delta = rmse_arm - rmse_reference
                forcing_value = -delta
                rows.append(
                    {
                        "authority_schema": inference.UPSTREAM_SCHEMA,
                        "authority_status": inference.AUTHORITY_STATUS,
                        "scope_status": inference.SCOPE_STATUS,
                        "prospective": False,
                        "confirmatory": False,
                        "protocol_sha256": PROTOCOL_SHA256,
                        "arm": "F3_full",
                        "protocol_arm": "F3_full",
                        "reference_arm": "F0",
                        "reference_protocol_arm": "F0",
                        "model": model,
                        "contrast_role": "primary_tree_F3_full_minus_F0",
                        "model_status": "frozen_tree_model",
                        "site_id": site,
                        "horizon": horizon,
                        "n_common_keys": 100,
                        "rmse_arm": rmse_arm,
                        "rmse_reference": rmse_reference,
                        "delta_rmse": delta,
                        "reportable": True,
                        "forcing_value_rmse": forcing_value,
                        "delta_estimand": inference.DELTA_ESTIMAND,
                        "forcing_value_estimand": inference.STATION_FORCING_VALUE,
                        "forcing_value_sign_rule": inference.SIGN_RULE,
                    }
                )
    return pd.DataFrame(rows, columns=inference.PAIRED_COLUMNS)


def _write_valid_inputs(tmp_path: Path) -> tuple[Path, Path, Path, pd.DataFrame]:
    upstream = tmp_path / "forcing_regime_v4_authority"
    upstream.mkdir(parents=True)
    paired_path = upstream / "forcing_regime_v4_paired_contrasts.parquet"
    paired = _synthetic_paired()
    paired.to_parquet(paired_path, index=False)

    registry_path = tmp_path / "station_registry_v1.csv"
    registry_rows = []
    for index, site in enumerate(_sites(), start=1):
        huc2 = (index - 1) % inference.EXPECTED_HUC2_CLUSTERS + 1
        registry_rows.append(
            {
                "site_no": site,
                "huc_cd": f"{huc2:02d}000001",
                "huc2": str(huc2),
                "huc_metadata_status": "USGS_SNAPSHOT_SITE_NO_MATCH",
            }
        )
    pd.DataFrame(registry_rows).to_csv(registry_path, index=False)

    cells = [
        {"arm": arm, "model": model, "horizon": horizon, "protocol_arm": arm}
        for arm in inference.ARMS
        for model in inference.MODELS
        for horizon in inference.HORIZONS
    ]
    sites = _sites()
    manifest = {
        "authority_schema": inference.UPSTREAM_SCHEMA,
        "authority_status": inference.AUTHORITY_STATUS,
        "scope_status": inference.SCOPE_STATUS,
        "governance": {
            "post_outcome_normalization": True,
            "completed_viewed_domain_subset": True,
            "prospective": False,
            "confirmatory": False,
            "full_v4_protocol_authority": False,
            "automatic_P5_joint_verdict": False,
        },
        "scope": {
            "arms": list(inference.ARMS),
            "protocol_arms": list(inference.ARMS),
            "models": list(inference.MODELS),
            "horizons": list(inference.HORIZONS),
            "exact_cell_count": inference.EXPECTED_UPSTREAM_CELLS,
            "cells": cells,
        },
        "bindings": {
            "v4_draft_protocol": {
                "path": "/frozen/wrr_information_regimes_protocol_v4.yaml",
                "bytes": 123,
                "sha256": PROTOCOL_SHA256,
                "exact_bytes_bound": True,
            },
            "data": {
                "station_registry": {
                    "path": "/frozen/station_registry_v1.csv",
                    "bytes": registry_path.stat().st_size,
                    "sha256": _sha256(registry_path),
                }
            },
        },
        "audits": {
            "exact_canonical_cell_count": inference.EXPECTED_UPSTREAM_CELLS,
            "protocol_sha256_computed_from_current_exact_bytes": PROTOCOL_SHA256,
            "common_reportable_station_set_across_all_12_cells": True,
            "station_first_estimand_only": True,
            "sign_relation_verified_station_by_station": True,
            "protocol_forcing_value_sign_contract": (
                inference.UPSTREAM_PROTOCOL_FORCING_VALUE_SIGN_CONTRACT
            ),
            "common_reportable_station_count": inference.EXPECTED_STATIONS,
            "common_reportable_station_ids": list(sites),
            "common_reportable_station_ids_sha256": inference._site_list_sha256(sites),
        },
        "outputs": [
            {
                "name": paired_path.name,
                "rows": len(paired),
                "bytes": paired_path.stat().st_size,
                "sha256": _sha256(paired_path),
            }
        ],
        "row_counts": {"authority_contrast_rows": len(paired)},
    }
    manifest_path = upstream / "forcing_regime_v4_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path, paired_path, registry_path, paired


def test_cluster_bootstrap_explicitly_resamples_complete_huc2_groups() -> None:
    effects = np.array([0.0, 10.0, 100.0, 101.0, 102.0])
    clusters = np.array(["01", "01", "02", "02", "02"])
    draws = 25
    seed = 31991
    actual = inference.whole_huc2_cluster_bootstrap_distribution(
        effects, clusters, draws=draws, seed=seed, batch_size=draws
    )

    rng = np.random.default_rng(seed)
    multiplicities = rng.multinomial(2, [0.5, 0.5], size=draws)
    complete_groups = [effects[clusters == "01"], effects[clusters == "02"]]
    expected = []
    for counts in multiplicities:
        replicated = [
            group
            for count, group in zip(counts, complete_groups, strict=True)
            for _ in range(int(count))
        ]
        expected.append(np.median(np.concatenate(replicated)))
    assert np.array_equal(actual, np.asarray(expected))


def test_exact_sign_flip_three_tail_formulas_match_enumeration() -> None:
    effects = np.array([-0.25, 0.125, 0.5])
    clusters = np.array(["01", "02", "03"])
    result = inference.exact_whole_huc2_sign_flip(effects, clusters)

    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=3)))
    simulated = np.median(signs * effects[None, :], axis=1)
    observed = float(np.median(effects))
    assert result["exact"] is True
    assert result["configurations"] == 2**3
    assert result["seed"] is None
    assert result["p_less_or_equal"] == pytest.approx(np.mean(simulated <= observed))
    assert result["p_greater_or_equal"] == pytest.approx(np.mean(simulated >= observed))
    assert result["p_two_sided_absolute"] == pytest.approx(
        np.mean(np.abs(simulated) >= abs(observed))
    )


def test_forcing_value_direction_and_sign_contract_are_fail_closed(tmp_path: Path) -> None:
    manifest, paired_path, registry, paired = _write_valid_inputs(tmp_path)
    validated, huc2_by_site, _ = inference.load_validated_inputs(
        manifest_path=manifest, paired_path=paired_path, registry_path=registry
    )
    assert (validated["forcing_value_rmse"] > 0.0).all()
    summary, per_huc, loco, _ = inference.build_cluster_sensitivities(validated, huc2_by_site)
    assert len(summary) == inference.EXPECTED_ANALYSIS_CELLS
    assert len(per_huc) == inference.EXPECTED_ANALYSIS_CELLS * 15
    assert len(loco) == inference.EXPECTED_ANALYSIS_CELLS * 15
    assert set(summary["equal_station_direction_vs_zero"]) == {"POSITIVE_FAVORS_F3_FULL"}
    assert set(summary["primary_weighting"]) == {"EQUAL_STATION"}
    assert set(summary["cluster_bootstrap_target_estimand"]) == {"EQUAL_STATION_MEDIAN"}

    tampered = paired.copy()
    tampered.loc[0, "forcing_value_rmse"] *= -1.0
    with pytest.raises(inference.InferenceAuthorityError, match="sign contract"):
        inference.validate_paired_contrasts(
            tampered,
            common_stations=_sites(),
            protocol_sha256=PROTOCOL_SHA256,
        )


def test_hash_tamper_is_rejected_before_paired_use(tmp_path: Path) -> None:
    manifest, paired, registry, _ = _write_valid_inputs(tmp_path)
    inference.load_validated_inputs(
        manifest_path=manifest, paired_path=paired, registry_path=registry
    )
    paired.write_bytes(paired.read_bytes() + b"tamper")
    with pytest.raises(inference.InferenceAuthorityError, match="hash/binding mismatch"):
        inference.load_validated_inputs(
            manifest_path=manifest, paired_path=paired, registry_path=registry
        )


def test_exact_upstream_and_six_cell_inventory_are_enforced(tmp_path: Path) -> None:
    manifest_path, paired_path, registry, paired = _write_valid_inputs(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scope"]["cells"] = manifest["scope"]["cells"][:-1]
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(inference.InferenceAuthorityError, match="exactly 12"):
        inference.load_validated_inputs(
            manifest_path=manifest_path,
            paired_path=paired_path,
            registry_path=registry,
        )

    missing_station = paired.iloc[:-1].copy()
    with pytest.raises(inference.InferenceAuthorityError, match="expected 696"):
        inference.validate_paired_contrasts(
            missing_station,
            common_stations=_sites(),
            protocol_sha256=PROTOCOL_SHA256,
        )


def test_atomic_publication_is_create_only_and_hashes_code_inputs_outputs(
    tmp_path: Path,
) -> None:
    manifest, paired, registry, _ = _write_valid_inputs(tmp_path)
    destination = tmp_path / "review" / "forcing_regime_inference_v4"
    built = inference.build_authority(
        manifest_path=manifest,
        paired_path=paired,
        registry_path=registry,
        output_dir=destination,
    )
    assert built == destination.absolute()
    authority_manifest_path = destination / f"{inference.OUTPUT_PREFIX}_manifest.json"
    authority_manifest = json.loads(authority_manifest_path.read_text(encoding="utf-8"))
    assert authority_manifest["authority_status"] == inference.AUTHORITY_STATUS
    assert authority_manifest["scope_status"] == inference.SCOPE_STATUS
    assert authority_manifest["status_permanent"] is True
    assert authority_manifest["status_contract"]["permanent"] is True
    assert authority_manifest["prospective"] is False
    assert authority_manifest["confirmatory"] is False
    assert authority_manifest["scope"]["claim_verdict_emitted"] is False
    assert authority_manifest["scope"]["multiplicity_adjustment_computed"] is False
    assert authority_manifest["scope"]["architecture_comparison_in_scope"] is False
    assert authority_manifest["row_counts"] == {
        "upstream_paired_rows": 696,
        "summary_rows": 6,
        "per_huc2_rows": 90,
        "leave_one_huc2_rows": 90,
    }
    assert authority_manifest["software"]["builder"]["sha256"] == _sha256(Path(inference.__file__))
    for output in authority_manifest["outputs"]:
        path = destination / output["name"]
        assert path.stat().st_size == output["bytes"]
        assert _sha256(path) == output["sha256"]
    assert authority_manifest["output_set_sha256"] == inference._canonical_json_sha256(
        authority_manifest["outputs"]
    )

    with pytest.raises(inference.InferenceAuthorityError, match="refusing overwrite"):
        inference.build_authority(
            manifest_path=manifest,
            paired_path=paired,
            registry_path=registry,
            output_dir=destination,
        )

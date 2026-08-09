from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "final"
    / "build_information_regime_inference_v4.py"
)
SPEC = importlib.util.spec_from_file_location("information_regime_inference_v4_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
inference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inference)

PROTOCOL_SHA256 = "a" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sites() -> tuple[str, ...]:
    return tuple(f"{index:08d}" for index in range(1, 16))


def _common() -> dict[int, tuple[str, ...]]:
    return {horizon: _sites() for horizon in inference.HORIZONS}


def _synthetic_paired() -> pd.DataFrame:
    rows = []
    for layout_index, layout in enumerate(inference.CONTRAST_LAYOUTS):
        family, contrast, definition, geometry, level, favors = layout
        for model_index, model in enumerate(inference.MODELS):
            for horizon in inference.HORIZONS:
                for site_index, site in enumerate(_sites(), start=1):
                    delta = (
                        (layout_index + 1) / 100.0
                        + model_index / 1000.0
                        + horizon / 10000.0
                        + site_index / 100000.0
                    )
                    rows.append(
                        {
                            "model": model,
                            "horizon": horizon,
                            "site_id": site,
                            "authority_schema": inference.UPSTREAM_SCHEMA,
                            "authority_status": inference.AUTHORITY_STATUS,
                            "protocol_sha256": PROTOCOL_SHA256,
                            "contrast_family": family,
                            "contrast_id": contrast,
                            "definition": definition,
                            "conditioning_geometry": geometry,
                            "conditioning_level": level,
                            # A zero reference makes the exact subtraction
                            # identity deterministic in the synthetic fixture.
                            "candidate_rmse": delta,
                            "reference_rmse": 0.0,
                            "delta_rmse": delta,
                            "n_common_keys": 100,
                            "negative_delta_favors": favors,
                        }
                    )
    return pd.DataFrame(rows, columns=inference.PAIRED_COLUMNS)


def _write_valid_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    paired_path = tmp_path / "information_regime_v4_paired_contrasts.parquet"
    paired = _synthetic_paired()
    paired.to_parquet(paired_path, index=False)

    registry_path = tmp_path / "station_registry_v1.csv"
    registry = pd.DataFrame(
        {
            "site_no": _sites(),
            "huc_cd": [f"{index:02d}000000" for index in range(1, 16)],
            "huc2": [f"{index:02d}" for index in range(1, 16)],
            "huc_metadata_status": ["USGS_SNAPSHOT_SITE_NO_MATCH"] * 15,
        }
    )
    registry.to_csv(registry_path, index=False)

    common = {
        str(horizon): {
            "n": len(_sites()),
            "site_ids": list(_sites()),
            "site_ids_sha256": inference._site_list_sha256(_sites()),
        }
        for horizon in inference.HORIZONS
    }
    manifest = {
        "authority_schema": inference.UPSTREAM_SCHEMA,
        "authority_status": inference.AUTHORITY_STATUS,
        "scope": {
            "full_protocol_v2_authority": False,
            "models": list(inference.MODELS),
            "horizons": list(inference.HORIZONS),
            "levels": ["L0", "L1", "L2"],
            "geometries": ["random", "region"],
        },
        "protocol_binding": {
            "path": "/frozen/protocol-v2.yaml",
            "bytes": 123,
            "sha256": PROTOCOL_SHA256,
            "exact_bytes_bound": True,
        },
        "outputs": [
            {
                "name": paired_path.name,
                "rows": len(paired),
                "bytes": paired_path.stat().st_size,
                "sha256": _sha256(paired_path),
            }
        ],
        "row_counts": {"paired_contrast_rows": len(paired)},
        "common_reportable_stations": common,
    }
    manifest_path = tmp_path / "information_regime_v4_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path, paired_path, registry_path


def test_cluster_bootstrap_resamples_whole_huc2_groups() -> None:
    effects = np.array([0.0, 10.0, 100.0, 101.0, 102.0])
    clusters = np.array(["01", "01", "02", "02", "02"])
    draws = 25
    seed = 31991
    actual = inference.whole_huc2_cluster_bootstrap_distribution(
        effects, clusters, draws=draws, seed=seed, batch_size=draws
    )

    rng = np.random.default_rng(seed)
    multiplicities = rng.multinomial(2, [0.5, 0.5], size=draws)
    expected = []
    groups = [effects[clusters == "01"], effects[clusters == "02"]]
    for counts in multiplicities:
        complete_groups = [
            group for count, group in zip(counts, groups, strict=True) for _ in range(int(count))
        ]
        expected.append(np.median(np.concatenate(complete_groups)))
    assert np.array_equal(actual, np.asarray(expected))


def test_point_is_equal_station_not_equal_huc_and_duplicate_station_fails() -> None:
    effects = np.array([0.0, 0.0, 0.0, 100.0])
    clusters = np.array(["01", "01", "01", "02"])
    per_huc, equal_huc = inference.equal_huc2_sensitivity(effects, clusters)

    assert np.median(effects) == 0.0
    assert equal_huc == 50.0
    assert [row["n_stations"] for row in per_huc] == [3, 1]

    paired = _synthetic_paired()
    duplicated = pd.concat([paired, paired.iloc[[0]]], ignore_index=True)
    with pytest.raises(inference.InferenceAuthorityError, match="station-level"):
        inference.validate_paired_contrasts(
            duplicated,
            common_stations=_common(),
            protocol_sha256=PROTOCOL_SHA256,
        )


def test_leave_one_huc2_direction_gate_is_strict() -> None:
    _, stable = inference.leave_one_huc2_sensitivity(
        np.array([0.1, 0.2, 0.3]), np.array(["01", "02", "03"])
    )
    assert stable["loco_min_delta_rmse"] > 0.0
    assert stable["loco_range_direction_vs_zero"] == "ALL_POSITIVE"
    assert stable["loco_direction_gate_pass"] is True

    rows, crossing = inference.leave_one_huc2_sensitivity(
        np.array([-2.0, 0.1, 0.2]), np.array(["01", "02", "03"])
    )
    assert crossing["loco_min_delta_rmse"] < 0.0
    assert crossing["loco_max_delta_rmse"] > 0.0
    assert crossing["loco_range_direction_vs_zero"] == "CROSSES_OR_TOUCHES_ZERO"
    assert crossing["loco_direction_gate_pass"] is False
    assert len(rows) == 3


def test_exact_sign_flip_enumerates_all_fifteen_clusters() -> None:
    result = inference.exact_whole_huc2_sign_flip(
        np.linspace(-0.2, 0.3, 15), np.array([f"{index:02d}" for index in range(15)])
    )
    assert result["exact"] is True
    assert result["configurations"] == 2**15
    assert result["seed"] is None
    for key in (
        "p_less_or_equal",
        "p_greater_or_equal",
        "p_two_sided_absolute",
    ):
        assert 0.0 <= result[key] <= 1.0


def test_exact_sign_flip_tail_formulas_match_explicit_enumeration() -> None:
    effects = np.array([-0.2, 0.1, 0.4])
    clusters = np.array(["01", "02", "03"])
    result = inference.exact_whole_huc2_sign_flip(effects, clusters)

    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=3)))
    simulated = np.median(signs * effects[None, :], axis=1)
    observed = float(np.median(effects))
    assert result["p_less_or_equal"] == pytest.approx(np.mean(simulated <= observed))
    assert result["p_greater_or_equal"] == pytest.approx(np.mean(simulated >= observed))
    assert result["p_two_sided_absolute"] == pytest.approx(
        np.mean(np.abs(simulated) >= abs(observed))
    )


def test_upstream_paired_hash_tamper_fails_before_use(tmp_path: Path) -> None:
    manifest, paired, registry = _write_valid_inputs(tmp_path)
    validated, huc2_by_site, bindings = inference.load_validated_inputs(
        manifest_path=manifest,
        paired_path=paired,
        registry_path=registry,
    )
    assert len(validated) == 60 * 15
    assert len(set(huc2_by_site.values())) == 15
    assert bindings["upstream_paired_contrasts"]["hash_validated_against_upstream_manifest"] is True

    paired.write_bytes(paired.read_bytes() + b"tamper")

    with pytest.raises(inference.InferenceAuthorityError, match="hash/binding mismatch"):
        inference.load_validated_inputs(
            manifest_path=manifest,
            paired_path=paired,
            registry_path=registry,
        )


def test_v2_family_audit_refuses_holm_for_missing_and_ambiguous_family() -> None:
    registry = []
    for test_id in ("N01", "N02", "N03", "N04", "N05", "N06", "N10"):
        registry.append(
            {
                "analysis_id": f"analysis-{test_id}",
                "model": "LightGBM",
                "conditioning_geometry": "random",
                "v2_test_id_candidate": test_id,
            }
        )
    audit = inference.v2_family_audit(registry)

    assert audit["source_contrast_absent_test_ids"] == ["N07", "N08", "N09"]
    assert audit["holm_computed"] is False
    assert audit["confirmatory_holm_verdict_emitted"] is False
    assert any("model presentation" in reason for reason in audit["blocking_reasons"])


def test_publication_is_create_only(tmp_path: Path) -> None:
    destination = tmp_path / "inference-authority"
    destination.mkdir()
    sentinel = destination / "sentinel"
    sentinel.write_text("retain", encoding="utf-8")

    with pytest.raises(inference.InferenceAuthorityError, match="refusing overwrite"):
        inference.publish_authority(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            output_dir=destination,
            manifest_base={},
        )
    assert sentinel.read_text(encoding="utf-8") == "retain"

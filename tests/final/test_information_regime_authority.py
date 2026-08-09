from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "final"
    / "build_information_regime_authority.py"
)
SPEC = importlib.util.spec_from_file_location("information_regime_authority_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
authority = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(authority)

PROTOCOL_SHA256 = "a" * 64


def _synthetic_effects() -> pd.DataFrame:
    rows = []
    sites = ("00000001", "00000002")
    level_value = {"L0": 1.0, "L1": 2.0, "L2": 4.0}
    model_value = {"LightGBM": 0.0, "ResidualLightGBM": 0.25}
    for geometry in authority.GEOMETRIES:
        seeds = (
            authority.RANDOM_SEEDS
            if geometry == "random"
            else authority.REGION_SEEDS
        )
        geometry_value = 0.0 if geometry == "random" else 10.0
        for level in authority.LEVELS:
            for model in authority.MODELS:
                for seed in seeds:
                    for horizon in authority.HORIZONS:
                        for site in sites:
                            n = 100
                            # One failed raw cell removes site 2 from the common
                            # h1 set everywhere, while leaving h3/h7 untouched.
                            if (
                                geometry == "random"
                                and level == "L2"
                                and model == "LightGBM"
                                and seed == 4
                                and horizon == 1
                                and site == "00000002"
                            ):
                                n = 99
                            seed_value = float(seed) if geometry == "random" else 0.0
                            value = (
                                geometry_value
                                + level_value[level]
                                + model_value[model]
                                + horizon / 100.0
                                + (0.1 if site == "00000002" else 0.0)
                                + seed_value
                            )
                            rows.append(
                                {
                                    "geometry": geometry,
                                    "level": level,
                                    "model": model,
                                    "seed": seed,
                                    "site_id": site,
                                    "horizon": horizon,
                                    "n": n,
                                    "rmse": value,
                                    "rmse_damped": value + 0.5,
                                    "reportable": n >= 100,
                                }
                            )
    return pd.DataFrame(rows)


def test_missing_shard_cell_or_seed_fails_closed() -> None:
    expected = authority.expected_shard_cells()
    assert len(expected) == 432
    authority.validate_shard_cell_set(expected)

    missing = set(expected)
    missing.remove(("random", "L2", "LightGBM", 4, 3, 7))
    with pytest.raises(authority.AuthorityError, match="missing"):
        authority.validate_shard_cell_set(missing)

    unexpected_seed = set(expected)
    unexpected_seed.remove(("random", "L2", "LightGBM", 4, 3, 7))
    unexpected_seed.add(("random", "L2", "LightGBM", 5, 3, 7))
    with pytest.raises(authority.AuthorityError, match="unexpected"):
        authority.validate_shard_cell_set(unexpected_seed)

    effects = _synthetic_effects()
    missing_effect_cell = effects.loc[
        ~(
            (effects["geometry"] == "random")
            & (effects["level"] == "L2")
            & (effects["model"] == "LightGBM")
            & (effects["seed"] == 4)
            & (effects["horizon"] == 7)
        )
    ]
    with pytest.raises(authority.AuthorityError, match="missing cells"):
        authority.validate_effects_subset(missing_effect_cell)

    unexpected_effect_seed = effects.copy()
    mask = (
        (unexpected_effect_seed["geometry"] == "random")
        & (unexpected_effect_seed["level"] == "L2")
        & (unexpected_effect_seed["model"] == "LightGBM")
        & (unexpected_effect_seed["seed"] == 4)
        & (unexpected_effect_seed["horizon"] == 7)
    )
    unexpected_effect_seed.loc[mask, "seed"] = 5
    with pytest.raises(authority.AuthorityError, match="unexpected cells"):
        authority.validate_effects_subset(unexpected_effect_seed)


def test_random_is_seed_first_and_common_reportable_is_horizon_wide() -> None:
    effects = _synthetic_effects()
    absolute, common = authority.build_absolute_station_rows(
        effects, protocol_sha256=PROTOCOL_SHA256
    )

    assert common[1] == ("00000001",)
    assert common[3] == ("00000001", "00000002")
    assert common[7] == ("00000001", "00000002")
    assert set(absolute.loc[absolute["horizon"] == 1, "site_id"]) == {"00000001"}

    row = absolute.loc[
        (absolute["geometry"] == "random")
        & (absolute["level"] == "L0")
        & (absolute["model"] == "LightGBM")
        & (absolute["horizon"] == 1)
        & (absolute["site_id"] == "00000001")
    ].iloc[0]
    # Per-seed station RMSE values are 1.01 + [0,1,2,3,4].
    assert row["rmse"] == pytest.approx(3.01)
    assert row["seed_count"] == 5
    assert row["aggregation_order"] == "station_then_seed_mean"

    contrasts = authority.build_paired_contrasts(absolute)
    assert set(contrasts["contrast_id"]) == {
        "L1-L0",
        "L2-L1",
        "L2-L0",
        "G@L0",
        "G@L1",
        "G@L2",
        "LxG",
    }
    assert len(contrasts) == 100  # 10 contrasts x 2 models x (1+2+2) sites
    assert contrasts.loc[contrasts["contrast_id"] == "LxG", "delta_rmse"].abs().max() < 1e-12


def test_summary_uses_paired_deltas_not_difference_of_medians() -> None:
    candidate = [50.0, 101.0, 202.0]
    reference = [0.0, 100.0, 200.0]
    contrasts = pd.DataFrame(
        {
            "authority_schema": authority.AUTHORITY_SCHEMA,
            "authority_status": authority.AUTHORITY_STATUS,
            "protocol_sha256": PROTOCOL_SHA256,
            "contrast_family": "geometry",
            "contrast_id": "G@L2",
            "definition": "RMSE(region,L2) - RMSE(random,L2)",
            "conditioning_geometry": "",
            "conditioning_level": "L2",
            "model": "LightGBM",
            "horizon": 7,
            "site_id": ["00000001", "00000002", "00000003"],
            "candidate_rmse": candidate,
            "reference_rmse": reference,
            "delta_rmse": [left - right for left, right in zip(candidate, reference)],
        }
    )

    summary = authority.summarize_paired(contrasts).iloc[0]
    difference_of_medians = pd.Series(candidate).median() - pd.Series(reference).median()
    assert difference_of_medians == 1.0
    assert summary["median_delta_rmse"] == 2.0
    assert summary["median_delta_rmse"] != difference_of_medians
    assert summary["q25_delta_rmse"] == 1.5
    assert summary["q75_delta_rmse"] == 26.0
    assert summary["iqr_delta_rmse"] == 24.5
    assert summary["win_fraction"] == 0.0
    assert summary["n"] == 3
    assert summary["summary_source"] == "paired_station_delta_rows_only"


def test_create_only_publication_refuses_existing_directory(tmp_path) -> None:
    destination = tmp_path / "information_regime_v4"
    destination.mkdir()
    sentinel = destination / "sentinel"
    sentinel.write_text("do not replace", encoding="utf-8")

    with pytest.raises(authority.AuthorityError, match="refusing overwrite"):
        authority.publish_authority(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            output_dir=destination,
            manifest_base={},
        )

    assert sentinel.read_text(encoding="utf-8") == "do not replace"

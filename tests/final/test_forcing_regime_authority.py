"""Synthetic-only tests for the forcing-regime v4 authority builder."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import scripts.final.build_forcing_regime_authority as authority
import scripts.final.run_forcing_ladder_v4 as runner

ROOT = Path(__file__).resolve().parents[2]
SITES = ("00000001", "00000002", "00000003")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol_text() -> str:
    return f"""\
protocol_id: {authority.EXPECTED_PROTOCOL_ID}
version: {authority.EXPECTED_PROTOCOL_VERSION}
status: {authority.EXPECTED_PROTOCOL_STATUS}
execution_authorized: false
seal_path: null
chronology_and_evidence_status:
  already_viewed_domains:
    - existing F0 tree result
    - existing F3_full tree result
  normalization_only_domains:
    - station-paired normalization of F3_full effects
common_key_and_reportability_contract:
  reportability:
    minimum_paired_targets_per_station_lead: 100
    station_set_rule: intersection_across_every_arm_in_the_declared_contrast
  target_numeric_reconstruction_contract: |
    Only an absolute tolerance of 2e-6 degrees C with zero relative tolerance
    is allowed. The observed maximum was 1.5258789076710855e-6. Every emitted
    row copies the registry y_true value.
station_first_estimands:
  forcing_value: median_i_of_R_F0_minus_R_Fk
"""


def _initialise_git_repository(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "synthetic@example.invalid"],
        cwd=path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Synthetic Authority"],
        cwd=path,
        check=True,
    )
    tracked = path / "tracked.txt"
    tracked.write_text("initial\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "synthetic"], cwd=path, check=True)
    tracked.write_text("dirty\n", encoding="utf-8")


def _make_fixture(tmp_path: Path) -> dict[str, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    shards = tmp_path / "forcing_shards_v4"
    shards.mkdir()
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(_protocol_text(), encoding="utf-8")
    station_registry = tmp_path / "station_registry.csv"
    pd.DataFrame({"site_no": SITES}).to_csv(station_registry, index=False)
    train_panel = tmp_path / "train.parquet"
    test_panel = tmp_path / "test.parquet"
    pd.DataFrame(
        {"site_id": SITES, "DATE": pd.to_datetime(["2015-01-01"] * 3), "WTEMP": 1.0}
    ).to_parquet(train_panel, index=False)
    pd.DataFrame(
        {"site_id": SITES, "DATE": pd.to_datetime(["2021-01-01"] * 3), "WTEMP": 1.0}
    ).to_parquet(test_panel, index=False)

    registry_rows: list[dict[str, Any]] = []
    for horizon in authority.HORIZONS:
        for site in SITES:
            issue_dates = pd.date_range("2021-01-01", periods=100, freq="D")
            for index, issue_date in enumerate(issue_dates):
                registry_rows.append(
                    {
                        "key_id": f"{site}-h{horizon}-k{index:03d}",
                        "site_id": site,
                        "horizon": horizon,
                        "issue_date": issue_date,
                        "target_date": issue_date + pd.Timedelta(days=horizon),
                        "y_true": 0.0,
                    }
                )
    registry = pd.DataFrame(registry_rows)
    forecast_registry = tmp_path / "forecast_keys.parquet"
    registry.to_parquet(forecast_registry, index=False)

    f0_rmse = dict(zip(SITES, (0.0, 100.0, 200.0), strict=True))
    f3_rmse = dict(zip(SITES, (100.0, 200.0, 100.0), strict=True))
    damped_rmse = dict(zip(SITES, (50.0, 150.0, 150.0), strict=True))
    all_shards: list[pd.DataFrame] = []
    for cell in authority.expected_cells():
        subset = registry.loc[registry["horizon"] == cell.horizon].copy()
        subset["arm"] = cell.arm
        subset["protocol_arm"] = cell.arm
        subset["model"] = cell.model
        predictions = f0_rmse if cell.arm == "F0" else f3_rmse
        subset["y_pred"] = subset["site_id"].map(predictions).astype(float)
        subset["y_damped"] = subset["site_id"].map(damped_rmse).astype(float)
        for column in authority.SUBSTITUTION_COLUMNS:
            subset[column] = 0
        if cell.arm == "F3_full":
            first_key_per_site = subset.groupby("site_id", sort=False).cumcount().eq(0)
            subset.loc[first_key_per_site, "forcing_substitutions_PRCP"] = 1
        subset["forcing_substitution_count"] = subset[list(authority.SUBSTITUTION_COLUMNS)].sum(
            axis=1
        )
        shard = subset.loc[:, authority.KEY_LEVEL_COLUMNS].copy()
        shard.to_parquet(shards / cell.filename, index=False)
        all_shards.append(shard)
    predictions = pd.concat(all_shards, ignore_index=True)
    effects_frame = runner.station_metrics_from_predictions(predictions)
    contrasts_frame = runner.paired_arm_contrasts(predictions)

    effects = tmp_path / "forcing_effects_v4.parquet"
    contrasts = tmp_path / "forcing_contrasts_v4.parquet"
    summary = tmp_path / "forcing_summary_v4.json"
    effects_frame.to_parquet(effects, index=False)
    contrasts_frame.to_parquet(contrasts, index=False)
    governance = runner.validate_execution_governance(protocol, normalization_only=True)
    governance = {
        **governance,
        "execution_resources": {
            "air2stream_worker_budget": 1,
            "THERMOROUTE_FORMAL_THREADS": 2,
        },
    }
    summary_document = runner.build_summary(
        predictions,
        effects_frame,
        contrasts_frame,
        [runner.Cell(cell.arm, cell.model, cell.horizon) for cell in authority.expected_cells()],
        governance=governance,
        legacy_f3_alias_used=False,
        air2stream_worker_budget=1,
    )
    summary.write_text(json.dumps(summary_document, indent=2), encoding="utf-8")

    dependency = tmp_path / "dependency.py"
    dependency.write_text("VALUE = 'synthetic'\n", encoding="utf-8")
    repository = tmp_path / "repository"
    _initialise_git_repository(repository)
    output = tmp_path / "forcing_regime_v4_authority"
    return {
        "shards": shards,
        "effects": effects,
        "contrasts": contrasts,
        "summary": summary,
        "forecast_registry": forecast_registry,
        "protocol": protocol,
        "station_registry": station_registry,
        "train_panel": train_panel,
        "test_panel": test_panel,
        "runner": ROOT / "scripts" / "final" / "run_forcing_ladder_v4.py",
        "sources": (dependency,),
        "repository": repository,
        "output": output,
    }


def _build(fixture: dict[str, Any]) -> Path:
    return authority.build_authority(
        shards_path=fixture["shards"],
        effects_path=fixture["effects"],
        contrasts_path=fixture["contrasts"],
        summary_path=fixture["summary"],
        forecast_registry_path=fixture["forecast_registry"],
        protocol_path=fixture["protocol"],
        station_registry_path=fixture["station_registry"],
        train_panel_path=fixture["train_panel"],
        test_panel_path=fixture["test_panel"],
        runner_path=fixture["runner"],
        source_paths=fixture["sources"],
        output_dir=fixture["output"],
        repository_root=fixture["repository"],
    )


def test_exact_twelve_cell_inventory_and_canonical_protocol_arm(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    inventory = authority.validate_shard_inventory(fixture["shards"])
    assert set(inventory) == set(authority.expected_cells())

    missing = inventory[authority.expected_cells()[0]]
    missing.unlink()
    with pytest.raises(authority.AuthorityError, match="exact 12 canonical cells"):
        authority.validate_shard_inventory(fixture["shards"])

    fixture = _make_fixture(tmp_path / "second")
    path = fixture["shards"] / authority.expected_cells()[-1].filename
    frame = pd.read_parquet(path)
    frame["protocol_arm"] = "F3_temperature_only"
    with pytest.raises(authority.AuthorityError, match="identity differs"):
        authority.validate_shard_frame(
            frame,
            cell=authority.expected_cells()[-1],
            label=path.name,
        )


def test_registry_identity_is_exact_but_y_true_has_absolute_only_tolerance(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    cell = authority.Cell("F0", "LightGBM", 1)
    path = fixture["shards"] / cell.filename
    frame = pd.read_parquet(path)
    frame.loc[0, "y_true"] = authority.TARGET_ABSOLUTE_TOLERANCE - 1e-7
    checked = authority.validate_shard_frame(frame, cell=cell, label=path.name)
    registry = authority.validate_forecast_registry(pd.read_parquet(fixture["forecast_registry"]))
    _joined, maximum, mismatches = authority._registry_join(
        checked, registry, cell=cell, label=path.name
    )
    assert maximum == pytest.approx(1.9e-6)
    assert mismatches == 1

    outside = checked.copy()
    outside.loc[0, "y_true"] = authority.TARGET_ABSOLUTE_TOLERANCE + 1e-7
    with pytest.raises(authority.AuthorityError, match="beyond absolute tolerance"):
        authority._registry_join(outside, registry, cell=cell, label=path.name)

    wrong_target = checked.copy()
    wrong_target.loc[0, "target_date"] += pd.Timedelta(days=1)
    with pytest.raises(authority.AuthorityError, match="target_date"):
        authority._registry_join(wrong_target, registry, cell=cell, label=path.name)


def test_substitution_accounting_and_common_n_gate_fail_closed(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    cell = authority.Cell("F3_full", "LightGBM", 3)
    frame = pd.read_parquet(fixture["shards"] / cell.filename)
    frame.loc[0, "forcing_substitution_count"] += 1
    with pytest.raises(authority.AuthorityError, match="per-variable sum"):
        authority.validate_shard_frame(frame, cell=cell, label=cell.filename)

    predictions = pd.concat(
        [pd.read_parquet(path) for path in sorted(fixture["shards"].glob("*.parquet"))],
        ignore_index=True,
    )
    effects = authority.station_metrics_from_predictions(predictions)
    mask = (
        effects["arm"].eq("F0")
        & effects["model"].eq("LightGBM")
        & effects["horizon"].eq(1)
        & effects["site_id"].eq(SITES[0])
    )
    effects.loc[mask, ["n", "reportable"]] = [99, False]
    with pytest.raises(authority.AuthorityError, match="station set"):
        authority.common_reportable_sites(effects)


def test_station_paired_estimand_is_not_difference_of_marginal_medians(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    predictions = pd.concat(
        [pd.read_parquet(path) for path in sorted(fixture["shards"].glob("*.parquet"))],
        ignore_index=True,
    )
    effects = authority.station_metrics_from_predictions(predictions)
    contrasts = authority.paired_contrasts_from_predictions(predictions)
    subset = effects.loc[effects["model"].eq("LightGBM") & effects["horizon"].eq(1)]
    marginal_difference = (
        subset.loc[subset["arm"].eq("F3_full"), "rmse"].median()
        - subset.loc[subset["arm"].eq("F0"), "rmse"].median()
    )
    station_delta = contrasts.loc[
        contrasts["model"].eq("LightGBM") & contrasts["horizon"].eq(1),
        "delta_rmse",
    ]
    assert marginal_difference == 0.0
    assert station_delta.median() == 100.0
    assert np.median(-station_delta.to_numpy(float)) == -100.0


@pytest.mark.parametrize(
    ("artifact", "column", "message"),
    [
        ("effects", "rmse", "runner effects.rmse differs"),
        ("contrasts", "delta_rmse", "runner contrasts.delta_rmse differs"),
    ],
)
def test_runner_derived_parquet_tampering_fails_itemwise(
    tmp_path: Path,
    artifact: str,
    column: str,
    message: str,
) -> None:
    fixture = _make_fixture(tmp_path)
    frame = pd.read_parquet(fixture[artifact])
    frame.loc[0, column] += 0.01
    frame.to_parquet(fixture[artifact], index=False)
    with pytest.raises(authority.AuthorityError, match=message):
        _build(fixture)


def test_runner_summary_tampering_fails_itemwise(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    document = json.loads(fixture["summary"].read_text(encoding="utf-8"))
    key = "LightGBM/h1/F3_full_minus_F0"
    document["paired_forcing_effects"][key]["median_station_delta_rmse"] += 0.01
    fixture["summary"].write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(authority.AuthorityError, match="median_station_delta_rmse differs"):
        _build(fixture)


def test_atomic_create_only_build_binds_every_input_and_emits_no_verdict(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    protected_hashes = {
        name: _sha256(fixture[name]) for name in ("effects", "contrasts", "summary")
    }
    destination = _build(fixture)
    assert destination == fixture["output"]
    manifest_path = destination / "forcing_regime_v4_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["authority_status"] == "POST_OUTCOME_NORMALIZATION_ONLY"
    assert manifest["scope_status"] == "COMPLETED_VIEWED_DOMAIN_SUBSET"
    assert manifest["governance"]["prospective"] is False
    assert manifest["governance"]["confirmatory"] is False
    assert manifest["bindings"]["v4_draft_protocol"]["sha256"] == _sha256(fixture["protocol"])
    assert manifest["bindings"]["shards"]["count"] == 12
    assert len(manifest["bindings"]["shards"]["files"]) == 12
    assert len(manifest["bindings"]["source_code"]) == 3
    assert manifest["bindings"]["git_state"]["worktree_dirty"] is True
    assert set(manifest["bindings"]["runner_derived_artifacts"]) == {
        "effects",
        "contrasts",
        "summary",
    }
    assert set(manifest["bindings"]["data"]) == {
        "station_registry",
        "training_panel",
        "test_panel",
        "forecast_registry",
    }
    assert manifest["audits"]["runner_summary_equal_independent_reconstruction_item_by_item"]
    assert manifest["audits"]["P5_joint_verdict"] == ("NOT_ADJUDICATED_BY_AUTHORITY_BUILDER")
    summary = pd.read_parquet(destination / "forcing_regime_v4_summary.parquet")
    assert (
        not summary.astype(str)
        .apply(lambda column: column.str.contains("CONFIRMED", regex=False).any())
        .any()
    )
    assert set(summary.loc[summary["horizon"].eq(7), "P5_applicable"]) == {True}
    assert {
        name: _sha256(fixture[name]) for name in ("effects", "contrasts", "summary")
    } == protected_hashes

    with pytest.raises(authority.AuthorityError, match="refusing overwrite"):
        _build(fixture)

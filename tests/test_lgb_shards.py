from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from thermoroute.lgb_shards import (
    LIGHTGBM_HEADS,
    LightGBMShardError,
    LightGBMShardLineage,
    finalize_shard_set,
    lightgbm_design_key_digest,
    save_lightgbm_shard,
    shard_manifest_path,
    try_load_lightgbm_shard,
    validate_shard_set,
)
from thermoroute.repro import RunIdentity


@pytest.fixture
def identity() -> RunIdentity:
    return RunIdentity(
        run_id="a" * 20,
        panel_sha256="1" * 64,
        registry_sha256="2" * 64,
        config_sha256="3" * 64,
        source_sha256="4" * 64,
        runtime_sha256="5" * 64,
    )


@pytest.fixture
def design() -> pd.DataFrame:
    return pd.DataFrame({
        "temperature": np.linspace(-1.0, 1.0, 32),
        "flow": np.linspace(0.2, 4.0, 32),
        "station_code": pd.Categorical(
            ["A", "B"] * 16, categories=["A", "B"], ordered=False
        ),
    })


@pytest.fixture
def booster(design: pd.DataFrame) -> lgb.Booster:
    target = (
        0.7 * design["temperature"].to_numpy(float)
        + 0.1 * design["flow"].to_numpy(float)
    )
    return lgb.train(
        {
            "objective": "regression",
            "verbosity": -1,
            "num_threads": 1,
            "deterministic": True,
            "force_col_wise": True,
            "min_data_in_leaf": 2,
        },
        lgb.Dataset(design, label=target, categorical_feature=["station_code"]),
        num_boost_round=5,
    )


def _lineage(
    identity: RunIdentity,
    head: str,
    *,
    horizon: int = 1,
    config_tag: str = "frozen",
) -> LightGBMShardLineage:
    return LightGBMShardLineage.from_run_identity(
        identity,
        cohort="temporal_stage9",
        seed=0,
        horizon=horizon,
        head=head,
        design_key_sha256="d" * 64,
        head_config={"tag": config_tag, "head": head},
    )


def _save_all_heads(
    root: Path,
    identity: RunIdentity,
    booster: lgb.Booster,
    probe: pd.DataFrame,
) -> list[LightGBMShardLineage]:
    lineages = [_lineage(identity, head) for head in LIGHTGBM_HEADS]
    for lineage in lineages:
        save_lightgbm_shard(
            root, lineage=lineage, model=booster, parity_input=probe
        )
    return lineages


def test_partial_run_resumes_valid_shards_and_finalizes_only_when_complete(
    tmp_path, identity, booster, design,
):
    probe = design.iloc[:11].copy()
    lineages = [_lineage(identity, head) for head in LIGHTGBM_HEADS]
    point = lineages[0]
    save_lightgbm_shard(
        tmp_path, lineage=point, model=booster, parity_input=probe
    )
    point_manifest = shard_manifest_path(tmp_path, point)
    point_mtime = point_manifest.stat().st_mtime_ns
    assert try_load_lightgbm_shard(
        tmp_path, lineage=point, parity_input=probe
    ) is not None
    assert try_load_lightgbm_shard(
        tmp_path, lineage=lineages[1], parity_input=probe
    ) is None
    with pytest.raises(LightGBMShardError, match="incomplete"):
        finalize_shard_set(
            tmp_path, lineages=lineages, parity_inputs={1: probe}
        )

    for lineage in lineages[1:]:
        save_lightgbm_shard(
            tmp_path, lineage=lineage, model=booster, parity_input=probe
        )
    complete = finalize_shard_set(
        tmp_path, lineages=lineages, parity_inputs={1: probe}
    )
    complete_mtime = complete.stat().st_mtime_ns
    assert validate_shard_set(
        complete, cache_root=tmp_path, expected_lineages=lineages
    )["shard_count"] == len(LIGHTGBM_HEADS)

    # An exact retry is a read-only hit for both shard and complete-set paths.
    save_lightgbm_shard(
        tmp_path, lineage=point, model=booster, parity_input=probe
    )
    assert point_manifest.stat().st_mtime_ns == point_mtime
    assert finalize_shard_set(
        tmp_path, lineages=lineages, parity_inputs={1: probe}
    ).stat().st_mtime_ns == complete_mtime


def test_present_corrupt_shard_fails_closed_and_is_never_retrained_over(
    tmp_path, identity, booster, design,
):
    probe = design.iloc[:9].copy()
    lineage = _lineage(identity, "point")
    save_lightgbm_shard(
        tmp_path, lineage=lineage, model=booster, parity_input=probe
    )
    manifest_path = shard_manifest_path(tmp_path, lineage)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["unexpected"] = "must be rejected"
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    corrupt_bytes = manifest_path.read_bytes()

    with pytest.raises(LightGBMShardError, match="schema is not exact"):
        try_load_lightgbm_shard(
            tmp_path, lineage=lineage, parity_input=probe
        )
    with pytest.raises(LightGBMShardError, match="schema is not exact"):
        save_lightgbm_shard(
            tmp_path, lineage=lineage, model=booster, parity_input=probe
        )
    assert manifest_path.read_bytes() == corrupt_bytes


def test_stale_lineage_at_expected_path_fails_closed(
    tmp_path, identity, booster, design,
):
    probe = design.iloc[:7].copy()
    old = _lineage(identity, "q05", config_tag="old")
    current = _lineage(identity, "q05", config_tag="current")
    save_lightgbm_shard(
        tmp_path, lineage=old, model=booster, parity_input=probe
    )
    stale_payload = shard_manifest_path(tmp_path, old).read_bytes()
    current_path = shard_manifest_path(tmp_path, current)
    current_path.parent.mkdir(parents=True, exist_ok=True)
    current_path.write_bytes(stale_payload)

    with pytest.raises(LightGBMShardError, match="stale lineage"):
        try_load_lightgbm_shard(
            tmp_path, lineage=current, parity_input=probe
        )
    with pytest.raises(LightGBMShardError, match="stale lineage"):
        save_lightgbm_shard(
            tmp_path, lineage=current, model=booster, parity_input=probe
        )
    assert current_path.read_bytes() == stale_payload


def test_native_text_object_corruption_and_probe_drift_are_rejected(
    tmp_path, identity, booster, design,
):
    probe = design.iloc[:13].copy()
    lineage = _lineage(identity, "event")
    save_lightgbm_shard(
        tmp_path, lineage=lineage, model=booster, parity_input=probe
    )
    manifest = json.loads(
        shard_manifest_path(tmp_path, lineage).read_text(encoding="utf-8")
    )
    object_path = tmp_path / manifest["model"]["path"]
    assert object_path.suffix == ".txt"
    assert not list(tmp_path.rglob("*.pkl"))
    assert not list(tmp_path.rglob("*.joblib"))

    changed_probe = probe.copy()
    changed_probe.loc[changed_probe.index[0], "temperature"] += 0.01
    with pytest.raises(LightGBMShardError, match="parity record is invalid"):
        try_load_lightgbm_shard(
            tmp_path, lineage=lineage, parity_input=changed_probe
        )

    original = object_path.read_bytes()
    object_path.write_bytes(original + b"corruption")
    corrupt = object_path.read_bytes()
    with pytest.raises(LightGBMShardError, match="native-text object is corrupt"):
        try_load_lightgbm_shard(
            tmp_path, lineage=lineage, parity_input=probe
        )
    with pytest.raises(LightGBMShardError, match="native-text object is corrupt"):
        save_lightgbm_shard(
            tmp_path, lineage=lineage, model=booster, parity_input=probe
        )
    assert object_path.read_bytes() == corrupt


def test_valid_shard_cannot_be_replaced_by_a_different_booster(
    tmp_path, identity, booster, design,
):
    probe = design.iloc[:10].copy()
    lineage = _lineage(identity, "q50")
    save_lightgbm_shard(
        tmp_path, lineage=lineage, model=booster, parity_input=probe
    )
    manifest_path = shard_manifest_path(tmp_path, lineage)
    manifest_before = manifest_path.read_bytes()
    objects_before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in (tmp_path / "objects").iterdir()
    }
    other_target = np.square(design["temperature"].to_numpy(float))
    other = lgb.train(
        {
            "objective": "regression",
            "verbosity": -1,
            "num_threads": 1,
            "deterministic": True,
            "force_col_wise": True,
            "min_data_in_leaf": 2,
        },
        lgb.Dataset(
            design, label=other_target, categorical_feature=["station_code"]
        ),
        num_boost_round=7,
    )
    with pytest.raises(LightGBMShardError, match="different model content"):
        save_lightgbm_shard(
            tmp_path, lineage=lineage, model=other, parity_input=probe
        )
    assert manifest_path.read_bytes() == manifest_before
    assert {
        path.relative_to(tmp_path): path.read_bytes()
        for path in (tmp_path / "objects").iterdir()
    } == objects_before


def test_cache_child_symlink_cannot_redirect_native_model_publication(
    tmp_path, identity, booster, design,
):
    cache = tmp_path / "cache"
    outside = tmp_path / "outside"
    cache.mkdir()
    outside.mkdir()
    (cache / "objects").symlink_to(outside, target_is_directory=True)
    with pytest.raises(LightGBMShardError, match="non-directory or symlink"):
        save_lightgbm_shard(
            cache,
            lineage=_lineage(identity, "point"),
            model=booster,
            parity_input=design.iloc[:8].copy(),
        )
    assert not list(outside.iterdir())


def test_complete_set_rejects_missing_head_and_noncartesian_registry(
    tmp_path, identity, booster, design,
):
    probe = design.iloc[:5].copy()
    incomplete = [_lineage(identity, head) for head in LIGHTGBM_HEADS[:-1]]
    with pytest.raises(LightGBMShardError, match="missing a probabilistic head"):
        finalize_shard_set(
            tmp_path, lineages=incomplete, parity_inputs={1: probe}
        )

    lineages = _save_all_heads(tmp_path, identity, booster, probe)
    lineages.extend(
        _lineage(identity, head, horizon=2)
        for head in LIGHTGBM_HEADS[:-1]
    )
    with pytest.raises(LightGBMShardError, match="missing a probabilistic head"):
        finalize_shard_set(
            tmp_path, lineages=lineages, parity_inputs={1: probe, 2: probe}
        )


def test_design_key_digest_binds_partitions_keys_and_feature_order():
    frame = pd.DataFrame({
        "site_id": ["A", "B"],
        "split": ["train", "train"],
        "issue_date": pd.to_datetime(["2010-01-01", "2010-01-02"]),
        "target_date": pd.to_datetime(["2010-01-02", "2010-01-03"]),
    })
    original = lightgbm_design_key_digest(
        {"train": frame}, feature_order=["WTEMP", "FLOW"]
    )
    assert original == lightgbm_design_key_digest(
        {"train": frame.iloc[::-1]}, feature_order=["WTEMP", "FLOW"]
    )
    changed = frame.copy()
    changed.loc[0, "target_date"] = pd.Timestamp("2010-01-04")
    assert original != lightgbm_design_key_digest(
        {"train": changed}, feature_order=["WTEMP", "FLOW"]
    )
    assert original != lightgbm_design_key_digest(
        {"train": frame}, feature_order=["FLOW", "WTEMP"]
    )

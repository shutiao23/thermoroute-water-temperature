"""Adversarial governance tests for forcing-v5 point-result authority."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.final.build_forcing_v5_result_authority as A  # noqa: E402


def test_dry_run_does_not_touch_outputs() -> None:
    """Dry run must print the fixed non-side-effect plan."""
    plan = A.dry_run_plan()  # returns a mapping, not a JSON string
    assert plan["result_input"] == "outputs/final/forcing_regime_v5_observed"
    assert plan["point_authority_output"] == "outputs/final/forcing_regime_v5_observed_point_authority_v1"
    assert plan["inference_authority_output"] == "outputs/final/forcing_regime_v5_observed_inference_authority_v1"
    assert plan["verification_performed"] is False
    assert plan["publication_performed"] is False
    assert plan["inference_authority_published_by_this_builder"] is False
    assert plan["scope"]["cell_count"] == 12


def test_main_verify_requires_manifest_pin() -> None:
    assert A.main(["--verify"]) == 2


def test_main_publish_requires_manifest_pin() -> None:
    assert A.main(["--publish"]) == 2


def test_parser_default_prints_plan(capsys: pytest.CaptureFixture[str]) -> None:
    assert A.main([]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["format"] == A.AUTHORITY_FORMAT
    assert plan["mode"] == "DRY_RUN_NO_BYTES_READ_NO_MODEL_FIT_NO_WRITE"
    assert plan["point_authority_output"] == "outputs/final/forcing_regime_v5_observed_point_authority_v1"


def test_expected_cells_covers_twelve_cells() -> None:
    cells = A.expected_cells()
    assert len(cells) == 12
    assert cells[0].arm == "F0"
    assert cells[-1] == A.V5.Cell("F3_full", "ResidualLightGBM", 7)


def test_shard_binding_uses_result_relative_path() -> None:
    payload = b"minimal-shard-mock"
    bound = A.BoundFile(
        path=Path("/tmp/forcing_shards_v5_observed/F0_LightGBM_h1_v5_observed.parquet"),
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        stat_signature=(0, 0, 0, len(payload), 0, 0),
    )
    expected = {
        "path": "forcing_shards_v5_observed/F0_LightGBM_h1_v5_observed.parquet",
        "sha256": bound.sha256,
        "bytes": len(payload),
    }
    A._assert_shard_binding(
        expected,
        bound,
        label="runner shard Cell(arm='F0', model='LightGBM', horizon=1)",
        expected_path="forcing_shards_v5_observed/F0_LightGBM_h1_v5_observed.parquet",
    )
    mismatch = dict(expected)
    mismatch["path"] = "outputs/final/forcing_regime_v5_observed/" + expected["path"]
    with pytest.raises(A.AuthorityError):
        A._assert_shard_binding(
            mismatch,
            bound,
            label="runner shard Cell(arm='F0', model='LightGBM', horizon=1)",
            expected_path="forcing_shards_v5_observed/F0_LightGBM_h1_v5_observed.parquet",
        )


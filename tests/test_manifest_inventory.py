from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "manifest_inventory_test", ROOT / "scripts" / "14_manifest.py"
)
assert SPEC is not None and SPEC.loader is not None
MANIFEST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MANIFEST)


def test_development_replay_receipts_are_manifest_artifacts(tmp_path):
    replay = (
        tmp_path
        / "outputs"
        / "model_replay"
        / "route_a_development_replay_v1.json"
    )
    nested = replay.parent / "archive" / "replay.json"
    nested.parent.mkdir(parents=True)
    replay.write_text('{"status":"PASS"}\n', encoding="utf-8")
    nested.write_text('{"status":"PASS"}\n', encoding="utf-8")

    artifacts = MANIFEST.inventory(tmp_path, MANIFEST.ARTIFACT_PATTERNS)

    assert replay.relative_to(tmp_path).as_posix() in artifacts
    assert nested.relative_to(tmp_path).as_posix() in artifacts


def test_manifest_truth_audit_uses_exact_float32_model_semantics():
    key = ["site_id", "horizon", "issue_date", "target_date"]
    truth64 = np.asarray([32.1], dtype=np.float64)
    truth32 = truth64.astype(np.float32)
    frame = pd.DataFrame(
        {
            "site_id": ["1", "1"],
            "horizon": [1, 1],
            "issue_date": ["2020-01-01", "2020-01-01"],
            "target_date": ["2020-01-02", "2020-01-02"],
            "y_true": [truth64[0], truth32[0]],
        }
    )
    assert MANIFEST._truth_matches_at_model_precision(frame, key=key)
    frame.loc[1, "y_true"] = np.nextafter(
        truth32[0], np.float32(np.inf), dtype=np.float32
    )
    assert not MANIFEST._truth_matches_at_model_precision(frame, key=key)

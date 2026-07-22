from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
FORMAL_ENTRYPOINTS = (
    ("scripts/09_usgs_experiment.py", "--_thermoroute-stage09-worker"),
    ("scripts/16_lstm_baseline.py", "--_thermoroute-stage16-worker"),
    ("scripts/24_freeze_model_suite.py", "--_thermoroute-stage24-worker"),
    ("scripts/25_train_external_pooled_suite.py", "--_thermoroute-stage25-worker"),
)


def test_stage09_formal_publication_gate_is_cpu_only():
    path = ROOT / "scripts" / "09_usgs_experiment.py"
    spec = importlib.util.spec_from_file_location("stage09_cpu_gate_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    canonical = ROOT / "data_usgs" / "panel_usgs_120v2.parquet"
    assert module.formal_publication_candidate(
        panel_path=canonical, training_device="cpu", exploratory=False
    )
    for device in ("auto", "mps", "cuda"):
        assert not module.formal_publication_candidate(
            panel_path=canonical, training_device=device, exploratory=False
        )
    assert not module.formal_publication_candidate(
        panel_path=canonical, training_device="cpu", exploratory=True
    )


def _artifact_snapshot() -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    for relative in (
        "outputs/runs", "outputs/models", "outputs/predictions", "outputs/tables",
    ):
        base = ROOT / relative
        if not base.exists():
            continue
        base_stat = base.stat()
        snapshot[relative] = (-1, int(base_stat.st_mtime_ns))
        for path in base.rglob("*"):
            stat = path.stat()
            if path.is_dir():
                snapshot[path.relative_to(ROOT).as_posix()] = (
                    -1, int(stat.st_mtime_ns),
                )
            elif path.is_file():
                snapshot[path.relative_to(ROOT).as_posix()] = (
                    int(stat.st_size), int(stat.st_mtime_ns),
                )
    return snapshot


@pytest.mark.parametrize(("relative", "_worker_argument"), FORMAL_ENTRYPOINTS)
def test_formal_entrypoint_help_is_zero_training_and_zero_artifact_output(
    relative, _worker_argument,
):
    before = _artifact_snapshot()
    result = subprocess.run(
        [sys.executable, str(ROOT / relative), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert _artifact_snapshot() == before


@pytest.mark.parametrize(("relative", "worker_argument"), FORMAL_ENTRYPOINTS)
def test_formal_worker_cannot_bypass_controller_handshake(
    tmp_path, relative, worker_argument,
):
    cache = tmp_path / "untrusted-cache"
    cache.mkdir()
    result = subprocess.run(
        [
            sys.executable, "-I", "-X", f"pycache_prefix={cache}",
            str(ROOT / relative), worker_argument, "--help",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=os.environ.copy(),
    )
    assert result.returncode != 0
    assert "worker handshake is incomplete" in result.stderr

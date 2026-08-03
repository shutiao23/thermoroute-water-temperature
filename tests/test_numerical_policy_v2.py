"""Regression tests for the unified numerical-policy cap (amendment v2).

The Stage-09 RunIdentity embeds the numerical runtime contract (which embeds
the process thread cap), and worker processes recompute it and require exact
equality with the parent.  These tests pin the parent/child identity
consistency under a shared cap (positive) and the divergence under a
different cap (negative), plus the frozen policy document and role-cap
enforcement.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import thermoroute.repro as repro_module

from thermoroute.repro import (
    FORMAL_THREAD_LIMIT,
    assert_role_thread_cap,
    formal_policy_document,
    numerical_policy_role_cap,
    numerical_runtime_contract,
    sha256_json,
)

ROOT = Path(__file__).resolve().parents[1]

_CHILD_CODE = """
import os, sys
cap = sys.argv[1]
os.environ["THERMOROUTE_FORMAL_THREADS"] = cap
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = cap
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.path.insert(0, sys.argv[2])
# Mirror the real worker's loaded-library stack: the runtime contract
# content-binds every loaded BLAS/OpenMP library, so the child must load the
# same native libraries as the parent script processes do.
import numpy, pandas, scipy, torch, lightgbm  # noqa: F401
from thermoroute.repro import numerical_runtime_contract, sha256_json
print(sha256_json(numerical_runtime_contract()))
"""


def _contract_sha(cap: int) -> str:
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_CODE, str(cap), str(ROOT / "src")],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_runtime_contract_is_stable_under_shared_cap():
    """Two processes with the same script config and cap share one contract.

    This is the invariant the Stage-09 control closure relies on: worker
    processes recompute the runtime contract and require exact equality with
    the parent identity, so identical configuration must yield identical
    contracts (imports and environment included, as the real workers do).
    """
    first = _contract_sha(8)
    second = _contract_sha(8)
    assert first == second


def test_runtime_contract_diverges_under_different_cap():
    """A worker with a different declared cap can never match the parent."""
    parent = _contract_sha(8)
    other = 4
    child = _contract_sha(other)
    assert child != parent


def test_policy_document_caps_and_tolerances_are_frozen():
    doc = formal_policy_document(ROOT)
    assert doc["role_thread_caps"] == {
        "stage09": 8,
        "stage09b": 8,
        "stage16": 8,
        "stage25": 8,
    }
    assert doc["replay_tolerances"] == {
        "lightgbm": 1e-12,
        "thermoroute": 1e-5,
        "lstm": 1e-5,
        "probability": 1e-6,
    }
    assert doc["rules"]["cap_derived_from_this_document_not_ambient_choice"] is True


def test_role_cap_must_match_policy_document(monkeypatch):
    assert numerical_policy_role_cap(ROOT, "stage09") == 8
    monkeypatch.setattr(repro_module, "FORMAL_THREAD_LIMIT", 4)
    with pytest.raises(RuntimeError, match="frozen policy cap"):
        assert_role_thread_cap(ROOT, "stage09")


def test_amendment_v2_supersedes_v1_and_declares_defects():
    v2 = formal_policy_document(ROOT)
    path = ROOT / "protocols" / "route_a_numerical_policy_amendment_v2.json"
    doc = json_load(path)
    assert doc["supersedes"]["amendment"] == (
        "protocols/route_a_numerical_policy_amendment_v1.json"
    )
    assert len(doc["supersedes"]["v1_defects"]) >= 3
    assert doc["numerical_policy"]["amended"]["role_caps"] == v2["role_thread_caps"]
    assert doc["numerical_policy"]["replay_tolerances"] == v2["replay_tolerances"]


def json_load(path: Path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))

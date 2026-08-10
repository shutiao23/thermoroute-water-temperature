"""Tests for the forcing-placebo v5a specification seal.

The seal's only job is to make one claim falsifiable: that the decision rules
were fixed before any placebo outcome existed.  These tests check that the
claim cannot be made when it is untrue, and that the seal binds the protocol
bytes -- the property the v2 and v3 seals lack.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import scripts.final.seal_forcing_placebo_protocol_v5a as S


def test_protocol_declares_the_four_decision_rules_and_a_stopping_rule() -> None:
    document = S.load_protocol()
    assert set(document["decision_rules"]) >= {
        "P1_event_scale_supported",
        "P2_partly_seasonal",
        "P3_finding_withdrawn",
        "P4_timing_specificity",
        "stopping_rule",
    }
    # P3 is the rule that lets the forcing finding lose; a placebo design
    # without an admissible negative outcome is not a control.
    assert "withdraw" in document["decision_rules"]["P3_finding_withdrawn"].lower()


def test_protocol_is_not_execution_authorized() -> None:
    assert S.load_protocol()["execution_authorized"] is False


def test_seal_refuses_a_protocol_that_claims_execution_authority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    doctored = tmp_path / "protocol.yaml"
    text = S.PROTOCOL.read_text(encoding="utf-8").replace(
        "execution_authorized: false", "execution_authorized: true", 1
    )
    doctored.write_text(text, encoding="utf-8")
    monkeypatch.setattr(S, "PROTOCOL", doctored)
    with pytest.raises(S.SealError, match="execution_authorized"):
        S.load_protocol()


def test_seal_refuses_when_a_placebo_outcome_already_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    existing = tmp_path / "outputs" / "final" / "forcing_placebo_v5a"
    existing.mkdir(parents=True)
    monkeypatch.setattr(S, "ROOT", tmp_path)
    with pytest.raises(S.SealError, match="already exist"):
        S.assert_no_placebo_outcome_exists()


def test_the_guard_refuses_once_an_outcome_exists() -> None:
    """The property that outlives the run.

    Before the shuffle arm ran, this asserted that no placebo outcome was
    present.  That is a fact about a moment, and executing the arm falsified it
    -- the same trap that left three v5 tests permanently red.  What must remain
    true is the guard itself: with an outcome on disk the sealer must refuse to
    mint a seal, so any seal in the repository necessarily predates its outcome.
    """
    outcome_present = any(
        (S.ROOT / rel).exists()
        for rel in (
            "outputs/final/forcing_placebo_v5a",
            "outputs/final/forcing_placebo_v5a_authority_v1",
        )
    )
    if outcome_present:
        with pytest.raises(S.SealError, match="already exist"):
            S.assert_no_placebo_outcome_exists()
    else:
        assert S.assert_no_placebo_outcome_exists()


def test_the_existing_seal_records_that_it_preceded_the_outcome() -> None:
    if not S.SEAL.exists():
        pytest.skip("protocol has not been sealed")
    sealed = json.loads(S.SEAL.read_text(encoding="utf-8"))
    assertion = sealed["pre_outcome_assertion"]
    assert "outputs/final/forcing_placebo_v5a" in assertion["checked_absent_paths"]
    assert sealed["repository"]["tree_clean_at_seal_time"] is True


@pytest.mark.skipif(not S.SEAL.exists(), reason="protocol has not been sealed yet")
def test_seal_binds_the_exact_protocol_bytes() -> None:
    sealed = json.loads(S.SEAL.read_text(encoding="utf-8"))
    assert sealed["protocol_sha256"] == S._sha256_file(S.PROTOCOL)
    assert sealed["execution_authorized"] is False
    assert sealed["status"] == S.SEAL_STATUS


@pytest.mark.skipif(not S.SEAL.exists(), reason="protocol has not been sealed yet")
def test_verify_detects_a_post_seal_protocol_edit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    edited = tmp_path / "protocol.yaml"
    edited.write_text(
        S.PROTOCOL.read_text(encoding="utf-8") + "\n# post-seal edit\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(S, "PROTOCOL", edited)
    with pytest.raises(S.SealError, match="protocol bytes changed"):
        S.verify_seal()


@pytest.mark.skipif(not S.SEAL.exists(), reason="protocol has not been sealed yet")
def test_seal_is_immutable() -> None:
    """A published seal must never be overwritten.

    ``build_seal`` now refuses earlier, because an outcome exists, so the
    immutability guard is exercised directly rather than through it.
    """
    with pytest.raises(S.SealError, match="immutable"):
        S.write_seal({"format": S.SEAL_FORMAT, "status": "REPLACEMENT_ATTEMPT"})


@pytest.mark.skipif(not S.SEAL.exists(), reason="protocol has not been sealed yet")
def test_live_verification_passes() -> None:
    result = S.verify_seal()
    assert result["seal_verified"] is True

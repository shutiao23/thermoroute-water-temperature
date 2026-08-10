#!/usr/bin/env python3
"""Seal the forcing-placebo protocol v5a before any placebo outcome exists.

This is a *specification* seal, not an execution authorization.  It binds the
exact bytes of ``protocols/wrr_forcing_placebo_protocol_v5a.yaml`` together
with the inputs the placebo arms must consume and the repository state at
sealing time, so that the decision rules P1-P4 -- including P3, under which the
forcing finding is withdrawn -- are provably fixed before any shuffled,
shifted, or single-component outcome is computed.

Why this exists separately from the v2/v3 seals: those seal files do not bind
the bytes of the YAML protocol they claim to seal, which is why
``docs/SCIENTIFIC_EVIDENCE_STATUS.md`` records both as
``PROVISIONAL_NOT_YET_COMPLETE``.  This one binds the protocol digest, verifies
it on every subsequent read, and refuses to overwrite an existing seal.

Execution still requires a second, separate authority that binds the placebo
runner source, the resolved runtime, and a clean design commit.  That authority
does not exist yet and this seal does not stand in for it.

Usage::

    python scripts/final/seal_forcing_placebo_protocol_v5a.py            # dry run
    python scripts/final/seal_forcing_placebo_protocol_v5a.py --seal     # write
    python scripts/final/seal_forcing_placebo_protocol_v5a.py --verify   # re-check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]

PROTOCOL = ROOT / "protocols" / "wrr_forcing_placebo_protocol_v5a.yaml"
SEAL = ROOT / "protocols" / "wrr_forcing_placebo_protocol_v5a_seal.json"

SEAL_FORMAT = "thermoroute.forcing-placebo-v5a-specification-seal.v1"
SEAL_STATUS = "SPECIFICATION_SEALED_EXECUTION_NOT_AUTHORIZED"

#: The inputs every placebo arm must consume, bound here so that an arm cannot
#: later be scored against a different key registry or a different reference
#: result and still call itself the sealed design.
BOUND_INPUTS = {
    "point_authority_summary": (
        "outputs/final/forcing_regime_v5_observed_point_authority_v1/"
        "forcing_v5_point_summary.json"
    ),
    "point_authority_manifest": (
        "outputs/final/forcing_regime_v5_observed_point_authority_v1/"
        "forcing_v5_point_result_authority_manifest.json"
    ),
    "inference_authority_manifest": (
        "outputs/final/forcing_regime_v5_observed_inference_authority_v1/"
        "forcing_v5_inference_authority_manifest.json"
    ),
    "reference_lineage_manifest": (
        "outputs/final/forcing_regime_v5_observed/"
        "forcing_lineage_manifest_v5_observed.json"
    ),
    "station_registry": "data_usgs/station_registry_v1.csv",
    "parent_protocol": "protocols/wrr_information_regimes_protocol_v4.yaml",
}

#: Fields whose values the seal restates, so that a later edit to the YAML that
#: changed a decision threshold would be visible in the seal diff and not only
#: in an opaque digest change.
SEALED_DECISION_FIELDS = (
    "decision_rules",
    "estimands",
    "arms",
    "held_fixed",
)


class SealError(RuntimeError):
    """A seal precondition failed; nothing was written."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_default(value: Any) -> str:
    """Render YAML scalars JSON cannot hold, without silently dropping them.

    PyYAML resolves unquoted ISO dates to ``datetime.date``.  Stringifying is
    safe here because the digest only has to be stable and injective for the
    document as written; anything this does not recognise raises.
    """
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"cannot canonicalise {type(value).__name__} in the protocol")


def _canonical_json_sha256(payload: Any) -> str:
    text = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        default=_canonical_default,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover
        raise SealError(f"git {' '.join(args)} failed: {exc}") from exc
    return out.stdout.strip()


def assert_no_placebo_outcome_exists() -> list[str]:
    """Refuse to seal if any placebo outcome is already on disk.

    A specification seal is worthless if the outcomes it claims to precede have
    already been computed.  This is the check that makes the pre-outcome claim
    falsifiable rather than merely asserted.
    """
    forbidden = [
        "outputs/final/forcing_placebo_v5a",
        "outputs/final/forcing_placebo_v5a_authority_v1",
        "outputs/final/forcing_placebo_v5a_inference_authority_v1",
        "outputs/final/forcing_component_v5a",
    ]
    present = [rel for rel in forbidden if (ROOT / rel).exists()]
    if present:
        raise SealError(
            "placebo outcomes already exist; this seal cannot claim to precede "
            f"them: {present}"
        )
    return forbidden


def load_protocol() -> dict[str, Any]:
    if not PROTOCOL.exists():
        raise SealError(f"protocol missing: {PROTOCOL}")
    document = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if document.get("protocol_id") != "thermoroute_wrr_forcing_placebo_v5a":
        raise SealError("protocol_id is not the v5a placebo protocol")
    if document.get("execution_authorized") is not False:
        raise SealError(
            "protocol declares execution_authorized true; a specification seal "
            "must not carry execution authority"
        )
    for field in SEALED_DECISION_FIELDS:
        if field not in document:
            raise SealError(f"protocol lacks the sealed field {field!r}")
    rules = document["decision_rules"]
    for required in ("P1_event_scale_supported", "P2_partly_seasonal",
                     "P3_finding_withdrawn", "P4_timing_specificity",
                     "stopping_rule"):
        if required not in rules:
            raise SealError(f"decision rules lack {required!r}")
    return document


def build_seal() -> dict[str, Any]:
    document = load_protocol()
    forbidden = assert_no_placebo_outcome_exists()

    inputs: dict[str, str] = {}
    for role, rel in sorted(BOUND_INPUTS.items()):
        path = ROOT / rel
        if not path.exists():
            raise SealError(f"bound input missing for {role}: {rel}")
        inputs[role] = _sha256_file(path)

    return {
        "format": SEAL_FORMAT,
        "status": SEAL_STATUS,
        "protocol_path": PROTOCOL.relative_to(ROOT).as_posix(),
        "protocol_sha256": _sha256_file(PROTOCOL),
        "protocol_canonical_document_sha256": _canonical_json_sha256(document),
        "protocol_id": document["protocol_id"],
        "protocol_version": document["version"],
        "execution_authorized": False,
        "sealed_decision_rules_sha256": _canonical_json_sha256(
            {field: document[field] for field in SEALED_DECISION_FIELDS}
        ),
        "sealed_decision_rule_names": sorted(document["decision_rules"]),
        "bound_input_sha256": inputs,
        "bound_input_paths": {role: rel for role, rel in sorted(BOUND_INPUTS.items())},
        "repository": {
            "commit": _git("rev-parse", "HEAD"),
            "tree_clean_at_seal_time": _git("status", "--porcelain") == "",
        },
        "pre_outcome_assertion": {
            "checked_absent_paths": forbidden,
            "meaning": (
                "No shuffled, shifted, or single-component forcing outcome "
                "existed when this seal was written."
            ),
        },
        "what_this_seal_does_not_do": [
            "authorize execution of any placebo or component arm",
            "bind a placebo runner source hash or resolved runtime",
            "confer prospective status on the already-inspected F0/F3 result",
            "permit any operational or deployable claim about F3",
        ],
        "next_required_authority": (
            "a separate execution authority binding the placebo runner source, "
            "resolved runtime, frozen key registry and a clean design commit"
        ),
        "sealer_sha256": _sha256_file(Path(__file__).resolve()),
    }


def write_seal(payload: dict[str, Any]) -> None:
    if SEAL.exists():
        raise SealError(
            f"seal already exists and is immutable: {SEAL}. A changed design "
            "requires a new protocol version, not a rewritten seal."
        )
    SEAL.write_text(
        json.dumps(payload, sort_keys=True, indent=1, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def verify_seal() -> dict[str, Any]:
    if not SEAL.exists():
        raise SealError(f"no seal at {SEAL}")
    sealed = json.loads(SEAL.read_text(encoding="utf-8"))
    actual = _sha256_file(PROTOCOL)
    if sealed["protocol_sha256"] != actual:
        raise SealError(
            "protocol bytes changed after sealing: "
            f"sealed={sealed['protocol_sha256']} actual={actual}"
        )
    document = load_protocol()
    rules_now = _canonical_json_sha256(
        {field: document[field] for field in SEALED_DECISION_FIELDS}
    )
    if sealed["sealed_decision_rules_sha256"] != rules_now:
        raise SealError("sealed decision fields changed after sealing")
    drifted = {
        role: rel for role, rel in sealed["bound_input_paths"].items()
        if _sha256_file(ROOT / rel) != sealed["bound_input_sha256"][role]
    }
    return {"seal_verified": True, "drifted_inputs": drifted}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--seal", action="store_true", help="write the seal file")
    group.add_argument("--verify", action="store_true", help="verify an existing seal")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify:
        print(json.dumps(verify_seal(), sort_keys=True, indent=1))
        return 0
    payload = build_seal()
    if not args.seal:
        print(json.dumps({"dry_run": True, **payload}, sort_keys=True, indent=1))
        return 0
    write_seal(payload)
    print(json.dumps({
        "status": "SEALED",
        "seal_path": SEAL.relative_to(ROOT).as_posix(),
        "seal_sha256": _sha256_file(SEAL),
        "protocol_sha256": payload["protocol_sha256"],
    }, sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

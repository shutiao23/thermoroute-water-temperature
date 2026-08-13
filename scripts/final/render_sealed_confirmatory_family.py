#!/usr/bin/env python3
"""Render the five sealed confirmatory tests at the margins they were sealed at.

`protocols/route_a_confirmatory_v1.json` registers five tests, and two of them
are *non-inferiority* tests against a +0.05 degC margin fixed on 2026-07-22,
before any post-2020 outcome was requested (`route_a_protocol_seal_v1.json`,
`prelabel_attestation.post_2020_wtemp_requested_or_inspected: false`).  The
manuscript was reporting all five against a margin of zero, which renders H2 as
a superiority test it was never registered as and discards the only
pre-specified negligibility statement the study owns.  The protocol is explicit
that this is not allowed: `all_five_tests_must_be_rendered_exactly_once: true`.

Getting this wrong in the conservative direction still got it wrong.  A
superiority p-value of 1.0 on a row registered as non-inferiority reads to a
referee as a failed test, when the sealed decision contract for that row was
satisfied.  The manuscript then reached for the minimum detectable effect to say
what the margin already said properly, which is how the "evidence of absence"
sentence in Section 4.10 came to exist.

Three things this deliberately does not do.  It does not invent a margin for the
three H1 rows, which were sealed at 0.00 degC and are superiority tests.  It
does not upgrade the +0.05 degC margin into an ecological or regulatory
statement -- the protocol's own rationale field disclaims that, and its
`noninferiority_wording_limit` forbids the words "equivalent", "parity" and
"equally good".  And it does not report a non-inferiority result for any
comparison outside the sealed family; a margin chosen after the outcome is not
evidence, and Section 4.10's architecture contrasts have no sealed margin, so
they stay underpowered rather than becoming negligible.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from thermoroute.significance import (
    cluster_bootstrap_paired_effect,
    cluster_sign_flip_pvalue,
    holm_adjust,
    noninferiority_decision,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "protocols" / "route_a_confirmatory_v1.json"
SEAL = ROOT / "protocols" / "route_a_protocol_seal_v1.json"
PAIRED = ROOT / "outputs" / "final" / "paired_effects.parquet"
OUT = ROOT / "outputs" / "final" / "sealed_confirmatory_family_v1"


def render(paired: pd.DataFrame, family: Sequence[dict]) -> pd.DataFrame:
    rows = []
    for test in family:
        g = paired[
            (paired.candidate == test["candidate"])
            & (paired.reference == test["reference"])
            & (paired.horizon == int(test["horizon"]))
        ].dropna(subset=["delta_rmse"])
        effects = g.delta_rmse.to_numpy(float)
        clusters = g.huc2.to_numpy()
        margin = float(test["margin_c"])

        boot = cluster_bootstrap_paired_effect(
            effects, clusters, n_boot=10000, seed=int(test["bootstrap_seed"])
        )
        # The margin enters the randomisation test, not just the interval: the
        # null is sign symmetry *around the margin*, which is what makes the
        # H2 rows non-inferiority tests rather than superiority tests.
        p = cluster_sign_flip_pvalue(effects, clusters, null_margin=margin)
        rows.append(
            {
                "test_id": test["test_id"],
                "candidate": test["candidate"],
                "reference": test["reference"],
                "horizon": int(test["horizon"]),
                "kind": "superiority" if margin == 0.0 else "non-inferiority",
                "margin_c": margin,
                "effect_c": boot["effect"],
                "ci_low_c": boot["ci_low"],
                "ci_high_c": boot["ci_high"],
                "p_sign_flip_at_margin": p,
                "n_stations": boot["n_stations"],
                "n_clusters": boot["n_clusters"],
            }
        )
    out = pd.DataFrame(rows)
    out["holm_p"] = holm_adjust(out.p_sign_flip_at_margin.to_numpy(float))
    # The sealed contract is a conjunction, and it is a conjunction on purpose:
    # a Holm-significant p with an interval straddling the margin, or the
    # reverse, is EVIDENCE_CONFLICT and not a supported claim.
    out["ci_below_margin"] = [
        noninferiority_decision(ci, m) if m > 0 else bool(ci < m)
        for ci, m in zip(out.ci_high_c, out.margin_c)
    ]
    out["holm_significant"] = out.holm_p <= 0.05
    # "SUPPORTED" reads as "the candidate won", which is not what a
    # non-inferiority row establishes -- row 4 meets its margin at a lead where
    # the tree is the better model. The wording states what was actually met.
    out["decision"] = np.where(
        out.holm_significant & out.ci_below_margin,
        "Meets margin",
        np.where(
            out.holm_significant | out.ci_below_margin,
            "Evidence conflict",
            "Does not meet margin",
        ),
    )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    family = protocol["primary_inference_contract"]["confirmatory_family"]
    paired = pd.read_parquet(PAIRED)

    table = render(paired, family)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out / "sealed_confirmatory_family.parquet", index=False)
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "protocol_id": protocol["protocol_id"],
                "seal_status": seal["status"],
                "seal_recorded_date": seal["recorded_date"],
                "prelabel_attestation": seal["prelabel_attestation"],
                "margin_rationale": protocol["primary_inference_contract"][
                    "noninferiority_margin_rationale"
                ],
                "wording_limit": protocol["primary_inference_contract"][
                    "confirmatory_claim_decision_contract"
                ]["noninferiority_wording_limit"],
                "source": str(PAIRED.relative_to(ROOT)),
                "n_tests": int(len(table)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    pd.set_option("display.width", 200)
    print(
        table[
            [
                "test_id", "kind", "margin_c", "effect_c", "ci_low_c", "ci_high_c",
                "p_sign_flip_at_margin", "holm_p", "decision",
            ]
        ].to_string(index=False)
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

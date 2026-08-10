#!/usr/bin/env python3
"""Minimum detectable effect for the frozen five-test family.

Section 3.6 says the clustered procedures are "approximate descriptive
sensitivities, not decision evidence", and Section 5.2 repeats it.  The
Abstract then says the seven-day ThermoRoute-versus-LightGBM difference "is not
distinguishable from zero at the whole-HUC2 cluster level".  That is an
equivalence statement drawn from a non-significant result, from a procedure the
paper says cannot support one, with no statement of what the design could have
resolved.  It is the one place where the manuscript's own epistemic standard is
violated, and an external review flagged it.

This builder supplies the missing quantity.  For each row of the frozen
five-test family it computes the smallest constant effect the cohort's
dependence structure can resolve at alpha = 0.05, by translating the observed
station-level effect vector toward zero until the exact whole-HUC2 sign-flip
tail crosses alpha.  With fifteen clusters the tail can never fall below
1/2^15, so the floor is reported beside it.

The point is not to rescue the equivalence claim.  It is to let the manuscript
replace "not distinguishable from zero" with the honest pair of numbers: the
observed effect with its interval, and the smallest effect this design could
have detected.  Where the second is larger than the first, the correct sentence
is that the comparison is underpowered, not that the models are equivalent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.final.build_forcing_v5_inference_authority as INF
from thermoroute.significance import (
    cluster_bootstrap_paired_effect,
    cluster_sign_flip_pvalue,
)

FINAL = ROOT / "outputs" / "final"
PAIRED = FINAL / "paired_effects.parquet"
DEFAULT_OUT = FINAL / "family_power.parquet"

FORMAT = "thermoroute.five-test-family-power.v1"
ALPHA = 0.05
N_BOOT = 10_000
BOOT_SEED = 0

#: The frozen five-test family of Section 3.6, in its published order.
FAMILY = (
    ("ThermoRoute", "DampedPersistence", 1),
    ("ThermoRoute", "DampedPersistence", 3),
    ("ThermoRoute", "DampedPersistence", 7),
    ("ThermoRoute", "LightGBM", 3),
    ("ThermoRoute", "LightGBM", 7),
)


class PowerError(RuntimeError):
    """A family row could not be resolved."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_rows() -> list[dict[str, Any]]:
    frame = pd.read_parquet(PAIRED)
    rows: list[dict[str, Any]] = []
    for candidate, reference, horizon in FAMILY:
        block = frame[
            (frame["candidate"] == candidate)
            & (frame["reference"] == reference)
            & (frame["horizon"] == horizon)
        ].sort_values("site_id")
        if block.empty:
            raise PowerError(f"family row absent: {candidate} vs {reference} h{horizon}")
        deltas = block["delta_rmse"].to_numpy(float)
        clusters = block["huc2"].astype(str).to_numpy()

        boot = cluster_bootstrap_paired_effect(
            deltas, clusters, statistic="median", n_boot=N_BOOT, seed=BOOT_SEED,
        )
        tail = cluster_sign_flip_pvalue(deltas, clusters, statistic="median")
        mde = INF.minimum_detectable_effect(deltas, clusters, alpha=ALPHA)

        observed = float(np.median(deltas))
        detectable = mde["minimum_detectable_effect_degC"]
        # "underpowered" means the design could not have resolved an effect the
        # size of the one observed, so a null result carries no information
        # about equivalence.
        if not mde["resolvable"]:
            if observed >= 0.0:
                # the family's test is one-sided for "candidate better"; an
                # effect favouring the reference is outside what it can speak to
                limitation = "one_sided_test_favours_reference"
            else:
                limitation = "underpowered_at_the_observed_effect_size"
            underpowered = True
        elif abs(observed) < detectable:
            limitation = "underpowered_at_the_observed_effect_size"
            underpowered = True
        else:
            limitation = "adequately_powered_at_the_observed_effect_size"
            underpowered = False
        rows.append({
            "candidate": candidate,
            "reference": reference,
            "horizon": int(horizon),
            "median_delta_rmse": observed,
            "ci_low": boot["ci_low"],
            "ci_high": boot["ci_high"],
            "sign_flip_p_one_sided": tail,
            "smallest_attainable_p_value": mde["smallest_attainable_p_value"],
            "minimum_detectable_effect_degC": detectable,
            "mde_resolvable": bool(mde["resolvable"]),
            "mde_reason": mde["reason"],
            "underpowered_for_an_effect_of_the_observed_size": underpowered,
            "limitation": limitation,
            "interval_excludes_zero": bool(
                boot["ci_low"] > 0.0 or boot["ci_high"] < 0.0
            ),
            "n_stations": len(deltas),
            "n_clusters": len(np.unique(clusters)),
            "recommended_wording": (
                "report the effect and its interval; the design cannot support "
                "an equivalence statement here"
                if underpowered else
                "the effect exceeds the smallest this design can resolve"
            ),
        })
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)

    rows = build_rows()
    table = pd.DataFrame(rows)
    if args.print_only:
        print(table[[
            "candidate", "reference", "horizon", "median_delta_rmse",
            "ci_low", "ci_high", "minimum_detectable_effect_degC",
            "underpowered_for_an_effect_of_the_observed_size",
        ]].to_string(index=False))
        return 0

    table.to_parquet(args.out, index=False)
    manifest = {
        "format": FORMAT,
        "alpha": ALPHA,
        "source_sha256": _sha256_file(PAIRED),
        "builder_sha256": _sha256_file(Path(__file__).resolve()),
        "output_sha256": _sha256_file(args.out),
        "note": (
            "Power for an existing frozen family; no model was refitted and no "
            "new comparison was added to the family."
        ),
    }
    (args.out.parent / "family_power_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=1) + "\n", encoding="utf-8",
    )
    print(json.dumps({"status": "WRITTEN", "rows": len(table)},
                     sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

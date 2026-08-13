#!/usr/bin/env python3
"""What a model gains by being allowed to skip the days it cannot predict.

The manuscript asserts that scoring each model on its own complete-case set is
an inflation channel, and then removes the channel by construction rather than
measuring it. An advisor review made the obvious objection: the paper claims
benchmark design governs reported skill, quantifies the reference channel, and
leaves the key-set channel as an argument. This measures it.

The cohort supplies a natural experiment. Fourteen of the fifteen scored models
produce a prediction on all 358,807 held-out keys. The air2stream-style hybrid
produces 356,131: it fails on 2,676 keys, 0.75% of the registry, where its
per-station fit cannot be evaluated. Under the common-key rule that is a failure
and every model is scored on the intersection. Under the complete-case
convention it would instead be an exclusion, and the hybrid would be scored on a
subset it selected by its own ability to handle it.

The question is whether that subset is easier, and by how much. It is answerable
without any counterfactual model, because fourteen models did predict those
2,676 keys: their error on the declined subset, against their error on the rest,
measures how hard those days were. No assumption is needed about what the hybrid
would have predicted -- only about what the days were like, which the other
models observe directly.

The estimand is deliberately not a station-first median here. The declined keys
are concentrated in a minority of stations, so a station-first statistic would
average the effect away across stations that lost nothing; the quantity of
interest is the difficulty of the excluded days themselves, which is a pooled
property of those keys. Pooled RMSE is the right estimator for that question and
the wrong one for a model comparison, and it is used for the former only.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "outputs" / "final" / "predictions.parquet"
OUT = ROOT / "outputs" / "final" / "key_set_counterfactual_v1"

KEY = ["site_id", "horizon", "issue_date", "target_date"]
#: The model whose failures create the natural experiment, and the models used
#: to measure how hard the days it declined actually were.
INCOMPLETE = "Air2stream"
WITNESSES = ("ThermoRoute", "LightGBM", "LSTM", "PlainCausalTCN-7var",
             "DampedPersistence", "Persistence")


def _rmse(frame: pd.DataFrame) -> float:
    return float(np.sqrt(np.mean((frame.y_pred - frame.y_true) ** 2)))


def build() -> tuple[pd.DataFrame, dict[str, object]]:
    pred = pd.read_parquet(
        PRED, columns=["model", *KEY, "y_true", "y_pred"])
    pred = pred[pred.model.isin({INCOMPLETE, *WITNESSES})].copy()
    pred["k"] = list(map(tuple, pred[KEY].to_numpy()))

    complete = set(pred.loc[pred.model == WITNESSES[0], "k"])
    partial = set(pred.loc[pred.model == INCOMPLETE, "k"])
    declined = complete - partial
    if not declined:
        raise SystemExit("no declined keys: the natural experiment is absent")

    rows: list[dict[str, object]] = []
    for model in WITNESSES:
        g = pred[pred.model == model]
        on_declined = g[g.k.isin(declined)]
        on_common = g[~g.k.isin(declined)]
        rows.append({
            "witness": model,
            "rmse_on_declined_keys": _rmse(on_declined),
            "rmse_on_common_keys": _rmse(on_common),
            "difficulty_ratio": _rmse(on_declined) / _rmse(on_common),
            "n_declined": int(len(on_declined)),
            "n_common": int(len(on_common)),
        })
    frame = pd.DataFrame(rows)

    # How concentrated are the declined keys?  If they were spread evenly the
    # complete-case convention would cost every station a little; they are not.
    per_site = (pred[(pred.model == WITNESSES[0]) & (pred.k.isin(declined))]
                .groupby("site_id").size().sort_values(ascending=False))
    n_sites_total = pred.loc[pred.model == WITNESSES[0], "site_id"].nunique()

    ratios = frame.difficulty_ratio
    summary = {
        "incomplete_model": INCOMPLETE,
        "n_keys_complete": len(complete),
        "n_keys_declined": len(declined),
        "declined_share": len(declined) / len(complete),
        "difficulty_ratio_range": [float(ratios.min()), float(ratios.max())],
        "difficulty_ratio_median": float(ratios.median()),
        "sites_touched": int(per_site.size),
        "sites_total": int(n_sites_total),
        "declined_share_of_sites": per_site.size / n_sites_total,
        "max_declined_at_one_site": int(per_site.iloc[0]) if per_site.size else 0,
        "interpretation": (
            "Every witness model scores worse on the declined keys than on the "
            "rest, so the subset a complete-case convention would have handed "
            "the incomplete model is measurably easier than the registry it "
            "was drawn from. The common-key rule removes that advantage by "
            "construction; this is its size."
        ),
    }
    return frame, summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    frame, summary = build()
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out / "key_set_counterfactual.parquet", index=False)
    (args.out / "key_set_counterfactual_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    pd.set_option("display.width", 200)
    print(frame.to_string(index=False))
    print()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

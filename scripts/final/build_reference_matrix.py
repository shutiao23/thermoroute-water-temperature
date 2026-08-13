#!/usr/bin/env python3
"""Every model's reported skill against every reference, on one key set.

The manuscript's headline -- that a seven-day skill of +0.250 becomes +0.038 when
the reference changes -- was demonstrated on ThermoRoute alone, which leaves a
reader free to read it as a fact about that model. It is not. On these keys the
collapse happens to every model in the suite, and this is what makes the claim a
statement about benchmark construction rather than about one architecture.

Two things fall out that the single-model version could not show.

The weak reference *compresses* the suite. Against persistence the six learned
and hybrid models sit inside a 0.03-wide band and are, for practical reporting
purposes, indistinguishable. Against damped persistence the same six spread
across a range several times wider. A reader comparing two papers that both
quote skill against persistence is comparing numbers that were never able to
separate the models in the first place.

And the reference can flip a sign, not merely shrink a number. The
air2stream-style hybrid reports a clearly positive skill against persistence and
a negative one against damped persistence: the same fitted object, the same
days, described as beating the baseline by a fifth or as losing to it, depending
only on the denominator.

Intervals are whole-HUC2 cluster bootstrap, the same procedure and the same
resampling unit as the primary comparisons, and they carry the same limitation:
15 non-probability clusters with an effective count near nine, so they are
descriptive sensitivities rather than decision evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute.significance import cluster_bootstrap_paired_effect  # noqa: E402

PAIRED = ROOT / "outputs" / "final" / "paired_effects.parquet"
OUT = ROOT / "outputs" / "final" / "reference_matrix_v1"

#: Suite order for the table: the learned models, then the process-style hybrid.
MODELS = (
    "ThermoRoute",
    "LightGBM",
    "LSTM",
    "PlainCausalTCN-7var",
    "PlainMLP-7var",
    "Air2stream",
)
REFERENCES = ("Persistence", "DampedPersistence")
#: Seeded per (model, reference, horizon) so a re-run reproduces the interval.
#: Derived, not stored, because these rows are descriptive and are not part of
#: the sealed five-test family, whose seeds are fixed in the protocol.
def _seed(model: str, reference: str, horizon: int) -> int:
    return 7000 + 13 * MODELS.index(model) + 101 * REFERENCES.index(reference) + horizon


def build(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for reference in REFERENCES:
            for horizon in (1, 3, 7):
                g = paired[
                    (paired.candidate == model)
                    & (paired.reference == reference)
                    & (paired.horizon == horizon)
                ].dropna(subset=["station_skill", "delta_rmse"])
                if g.empty:
                    continue
                skill = g.station_skill.to_numpy(float)
                delta = g.delta_rmse.to_numpy(float)
                clusters = g.huc2.to_numpy()
                boot = cluster_bootstrap_paired_effect(
                    skill, clusters, n_boot=10000,
                    seed=_seed(model, reference, horizon))
                rows.append({
                    "model": model,
                    "reference": reference,
                    "horizon": horizon,
                    "median_station_skill": boot["effect"],
                    "skill_ci_low": boot["ci_low"],
                    "skill_ci_high": boot["ci_high"],
                    "median_delta_rmse": float(np.median(delta)),
                    "n_stations": boot["n_stations"],
                    "n_clusters": boot["n_clusters"],
                })
    return pd.DataFrame(rows)


def summarise(frame: pd.DataFrame, horizon: int = 7) -> dict[str, object]:
    """The two properties the single-model version of this claim could not show."""
    at = frame[frame.horizon == horizon]
    wide = at.pivot(index="model", columns="reference",
                    values="median_station_skill").reindex(MODELS)
    spread_pers = float(wide["Persistence"].max() - wide["Persistence"].min())
    spread_damp = float(wide["DampedPersistence"].max()
                        - wide["DampedPersistence"].min())
    flips = [m for m in wide.index
             if wide.loc[m, "Persistence"] > 0 > wide.loc[m, "DampedPersistence"]]
    return {
        "horizon": horizon,
        "skill_vs_persistence_range": [float(wide["Persistence"].min()),
                                       float(wide["Persistence"].max())],
        "skill_vs_damped_range": [float(wide["DampedPersistence"].min()),
                                  float(wide["DampedPersistence"].max())],
        "spread_vs_persistence": spread_pers,
        "spread_vs_damped": spread_damp,
        "spread_ratio_damped_over_persistence": spread_damp / spread_pers,
        "models_changing_sign": flips,
        "n_models": int(len(wide)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    paired = pd.read_parquet(PAIRED)
    frame = build(paired)
    summary = summarise(frame)

    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out / "reference_matrix.parquet", index=False)
    (args.out / "reference_matrix_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    pd.set_option("display.width", 200)
    print(frame[frame.horizon == 7].to_string(index=False))
    print()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

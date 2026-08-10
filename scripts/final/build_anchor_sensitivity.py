#!/usr/bin/env python3
"""How much of the headline depends on how the damped anchor was built.

The paper's thesis is that the reference model governs the reported gain.  The
reference it uses is a fitted object: a per-station harmonic climatology plus a
per-station no-intercept AR(1) anomaly coefficient, fitted on 2006-2015,
clipped to [0, 0.999], and extrapolated to lead ``h`` as ``phi^h``.  None of
those choices was varied anywhere in the manuscript, which leaves the paper
open to the obvious reply: the authors built one strong baseline, never tested
how strong it could have been, and then concluded that baselines matter.

This builder rebuilds the anchor under a predeclared set of variants, scores
each on the frozen 2021-2023 evaluation keys, and reports how far the headline
seven-day skill against damped persistence moves.  Nothing is refitted on the
learned side: ThermoRoute's predictions are the published ones, so every
difference here is attributable to the reference alone.

Variants, and what each one probes:

* ``clim_k1`` / ``clim_k5``      -- is three harmonics the right seasonal flexibility?
* ``clim_doy_window``           -- would a nonparametric day-of-year mean do better?
* ``phi_train_through_2018``    -- does a longer fitting window sharpen the anchor?
* ``phi_pooled``                -- how much of the anchor's strength is per-station?
* ``phi_direct_lead``           -- is ``phi^h`` extrapolation costing the reference?
* ``phi_unclipped``             -- does the [0, 0.999] clip bind anywhere that matters?

``phi_direct_lead`` is the variant most likely to strengthen the reference: it
fits the lead-h anomaly decay directly rather than raising the one-day
coefficient to the h-th power, which is only exact for a pure AR(1).

The baseline reconstruction is checked against the published
``DampedPersistence`` station RMSEs before any variant is reported; if the
rebuild does not reproduce the frozen artifact the whole comparison is void and
the builder fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.final.run_forcing_ladder_v5_observed as V5
import scripts.final.run_forcing_placebo_v5a_shuffle as P
from thermoroute import config as C
from thermoroute import features as F
from thermoroute.significance import cluster_bootstrap_paired_effect
from thermoroute.spatial import huc2_cluster_map, load_station_registry

FINAL = ROOT / "outputs" / "final"
PREDICTIONS = FINAL / "predictions.parquet"
STATION_METRICS = FINAL / "station_metrics.parquet"
DEFAULT_OUT = FINAL / "anchor_sensitivity.parquet"

FORMAT = "thermoroute.anchor-sensitivity.v1"
CANDIDATE = "ThermoRoute"
EVAL_START, EVAL_END = pd.Timestamp("2021-01-01"), pd.Timestamp("2023-12-31")
TRAIN_END_BASELINE = pd.Timestamp("2015-12-31")
TRAIN_END_EXTENDED = pd.Timestamp("2018-12-31")
#: The rebuilt baseline must reproduce the published anchor this closely.
BASELINE_TOLERANCE_DEGC = 2e-3
N_BOOT = 10_000
BOOT_SEED = 0


class AnchorError(RuntimeError):
    """A reconstruction or coverage check failed."""


@dataclass(frozen=True)
class Variant:
    name: str
    harmonics: int = C.SEASONAL_HARMONICS
    doy_window: int | None = None      # nonparametric day-of-year mean instead
    train_end: pd.Timestamp = TRAIN_END_BASELINE
    pooled_phi: bool = False
    direct_lead_phi: bool = False      # fit phi_h on lag-h pairs, not phi**h
    upper_bound: float = 0.999
    note: str = ""


VARIANTS = (
    Variant("baseline", note="as published: k=3, 2006-2015, per-station, phi^h"),
    Variant("clim_k1", harmonics=1, note="one harmonic"),
    Variant("clim_k5", harmonics=5, note="five harmonics"),
    Variant("clim_doy_window", doy_window=7, note="day-of-year mean, +/-7 day window"),
    Variant("phi_train_through_2018", train_end=TRAIN_END_EXTENDED,
            note="anchor fitted through the calibration year"),
    Variant("phi_pooled", pooled_phi=True, note="one station-balanced phi for all"),
    Variant("phi_direct_lead", direct_lead_phi=True,
            note="phi_h fitted on lag-h anomaly pairs"),
    Variant("phi_unclipped", upper_bound=1.0, note="clip relaxed to [0, 1.0]"),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------- inputs


def load_panel() -> pd.DataFrame:
    """The exact raw panel the reference run used, with observedness intact."""
    bounds = P.capture_placebo_inputs()
    raw, stations, _ = V5._load_raw_panel_from_bounds(bounds)
    C.STATIONS = stations
    return V5._validated_raw_frame(raw)


def load_evaluation() -> pd.DataFrame:
    """Published ThermoRoute and DampedPersistence rows on the test window."""
    frame = pd.read_parquet(
        PREDICTIONS,
        columns=["model", "site_id", "horizon", "issue_date", "target_date",
                 "y_true", "y_pred"],
        filters=[("model", "in", [CANDIDATE, "DampedPersistence"])],
    )
    frame = frame[
        (frame["target_date"] >= EVAL_START) & (frame["target_date"] <= EVAL_END)
    ]
    wide = frame.pivot_table(
        index=["site_id", "horizon", "issue_date", "target_date", "y_true"],
        columns="model", values="y_pred",
    ).reset_index()
    missing = {CANDIDATE, "DampedPersistence"} - set(wide.columns)
    if missing:
        raise AnchorError(f"predictions lack {sorted(missing)}")
    return wide.dropna(subset=[CANDIDATE, "DampedPersistence"])


def reportable_stations() -> dict[int, set[str]]:
    metrics = pd.read_parquet(STATION_METRICS)
    metrics = metrics[metrics["model"] == CANDIDATE]
    return {
        int(h): set(block["site_id"])
        for h, block in metrics.groupby("horizon")
    }


# ------------------------------------------------------------------- anchors


def _doy_window_climatology(
    panel: pd.DataFrame, train_mask: np.ndarray, window: int
) -> dict[str, np.ndarray]:
    """Per-station day-of-year mean over a wrapped +/- ``window`` day band."""
    table: dict[str, np.ndarray] = {}
    train = panel.loc[train_mask]
    for station, block in train.groupby("site_id", sort=True):
        doy = pd.to_datetime(block["DATE"]).dt.dayofyear.to_numpy()
        values = block["WTEMP"].to_numpy(float)
        finite = np.isfinite(values)
        doy, values = doy[finite], values[finite]
        curve = np.full(367, np.nan)
        for day in range(1, 367):
            delta = np.abs(doy - day)
            near = np.minimum(delta, 365 - delta) <= window
            if near.any():
                curve[day] = float(np.mean(values[near]))
        overall = float(np.nanmean(values)) if len(values) else np.nan
        curve = np.where(np.isfinite(curve), curve, overall)
        table[station] = curve
    return table


def _anomaly(panel: pd.DataFrame, station: str, clim: Any, doy_table: Any) -> np.ndarray:
    values = panel["WTEMP"].to_numpy(float)
    if doy_table is not None:
        doy = pd.to_datetime(panel["DATE"]).dt.dayofyear.to_numpy()
        baseline = doy_table[station][doy]
    else:
        baseline = clim.predict_dates(station, panel["DATE"])
    return values - baseline


def fit_variant(panel: pd.DataFrame, variant: Variant) -> tuple[Any, Any, dict]:
    """Return (climatology, doy_table, phi) for one variant."""
    train_mask = (
        pd.to_datetime(panel["DATE"]) <= variant.train_end
    ).to_numpy() & (pd.to_datetime(panel["DATE"]) >= pd.Timestamp("2006-01-01")).to_numpy()

    doy_table = None
    clim = None
    if variant.doy_window is not None:
        doy_table = _doy_window_climatology(panel, train_mask, variant.doy_window)
    else:
        clim = F.HarmonicClimatology.fit(
            panel, train_mask, target="WTEMP", k=variant.harmonics,
        )

    leads = (1, 3, 7)
    phi: dict[tuple[str, int], float] = {}
    pooled_pairs: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {h: [] for h in leads}

    for station, block in panel.loc[train_mask].groupby("site_id", sort=True):
        block = block.sort_values("DATE")
        anomaly = _anomaly(block, station, clim, doy_table)
        observed = block["WTEMP_observed"].to_numpy(bool)
        dates = pd.to_datetime(block["DATE"]).to_numpy(dtype="datetime64[D]")
        for lead in leads:
            step = lead if variant.direct_lead_phi else 1
            gap = (dates[step:] - dates[:-step]) == np.timedelta64(step, "D")
            valid = (
                observed[step:] & observed[:-step]
                & np.isfinite(anomaly[step:]) & np.isfinite(anomaly[:-step]) & gap
            )
            x, y = anomaly[:-step][valid], anomaly[step:][valid]
            pooled_pairs[lead].append((x, y))
            if len(x) >= 30 and float(np.mean(x * x)) > 1e-8:
                raw = float(np.sum(x * y) / np.sum(x * x))
            else:
                raw = 0.9 ** step
            phi[(station, lead)] = float(np.clip(raw, 0.0, variant.upper_bound))
            if not variant.direct_lead_phi:
                break  # one coefficient, raised to the power h at prediction time

    if variant.pooled_phi:
        for lead in leads:
            xs = np.concatenate([x for x, _ in pooled_pairs[lead]])
            ys = np.concatenate([y for _, y in pooled_pairs[lead]])
            value = float(np.clip(
                np.sum(xs * ys) / np.sum(xs * xs), 0.0, variant.upper_bound
            ))
            for station in panel["site_id"].unique():
                phi[(station, lead)] = value
            if not variant.direct_lead_phi:
                break
    return clim, doy_table, phi


def score_variant(
    panel: pd.DataFrame, evaluation: pd.DataFrame, variant: Variant,
) -> pd.DataFrame:
    """Per-station RMSE of the variant anchor and of ThermoRoute on shared keys."""
    clim, doy_table, phi = fit_variant(panel, variant)

    issue = panel[["site_id", "DATE", "WTEMP", "WTEMP_observed"]].rename(
        columns={"DATE": "issue_date", "WTEMP": "issue_wtemp"}
    )
    merged = evaluation.merge(issue, on=["site_id", "issue_date"], how="left")
    merged = merged[merged["WTEMP_observed"].fillna(False).to_numpy(bool)]
    if merged.empty:
        raise AnchorError("no evaluation row retained an observed issue temperature")

    baselines = []
    for station, block in merged.groupby("site_id", sort=False):
        if doy_table is not None:
            issue_doy = pd.to_datetime(block["issue_date"]).dt.dayofyear.to_numpy()
            target_doy = pd.to_datetime(block["target_date"]).dt.dayofyear.to_numpy()
            c_issue = doy_table[station][issue_doy]
            c_target = doy_table[station][target_doy]
        else:
            c_issue = clim.predict_dates(station, block["issue_date"])
            c_target = clim.predict_dates(station, block["target_date"])
        horizon = block["horizon"].to_numpy(int)
        if variant.direct_lead_phi:
            decay = np.array([phi[(station, int(h))] for h in horizon])
        else:
            decay = np.array([phi[(station, 1)] ** int(h) for h in horizon])
        baselines.append(
            pd.DataFrame({
                "site_id": station,
                "horizon": horizon,
                "y_true": block["y_true"].to_numpy(),
                "anchor": c_target + decay * (block["issue_wtemp"].to_numpy() - c_issue),
                CANDIDATE: block[CANDIDATE].to_numpy(),
            })
        )
    scored = pd.concat(baselines, ignore_index=True)
    scored["e_anchor"] = (scored["anchor"] - scored["y_true"]) ** 2
    scored["e_model"] = (scored[CANDIDATE] - scored["y_true"]) ** 2
    grouped = scored.groupby(["site_id", "horizon"], sort=True)[
        ["e_anchor", "e_model"]
    ].mean()
    out = np.sqrt(grouped).reset_index()
    out.columns = ["site_id", "horizon", "rmse_anchor", "rmse_model"]
    out["variant"] = variant.name
    return out


# ---------------------------------------------------------------- aggregation


def summarise(scored: pd.DataFrame, clusters: dict[str, str],
              keep: dict[int, set[str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (variant, horizon), block in scored.groupby(["variant", "horizon"], sort=True):
        block = block[block["site_id"].isin(keep.get(int(horizon), set()))]
        skill = 1.0 - block["rmse_model"].to_numpy() / block["rmse_anchor"].to_numpy()
        delta = block["rmse_model"].to_numpy() - block["rmse_anchor"].to_numpy()
        groups = block["site_id"].map(clusters).to_numpy()
        boot = cluster_bootstrap_paired_effect(
            skill, groups, statistic="median", n_boot=N_BOOT, seed=BOOT_SEED,
        )
        rows.append({
            "variant": variant,
            "horizon": int(horizon),
            "median_station_skill_vs_anchor": boot["effect"],
            "skill_ci_low": boot["ci_low"],
            "skill_ci_high": boot["ci_high"],
            "median_station_delta_rmse": float(np.median(delta)),
            "median_anchor_rmse": float(np.median(block["rmse_anchor"])),
            "median_model_rmse": float(np.median(block["rmse_model"])),
            "n_stations": len(block),
        })
    return rows


def check_baseline(rows: list[dict[str, Any]]) -> dict[str, float]:
    """The rebuilt baseline must reproduce the published DampedPersistence."""
    published = pd.read_parquet(STATION_METRICS)
    published = published[published["model"] == "DampedPersistence"]
    drift: dict[str, float] = {}
    for row in rows:
        if row["variant"] != "baseline":
            continue
        reference = float(
            published[published["horizon"] == row["horizon"]]["rmse"].median()
        )
        gap = abs(reference - row["median_anchor_rmse"])
        drift[f"h{row['horizon']}"] = gap
        if gap > BASELINE_TOLERANCE_DEGC:
            raise AnchorError(
                f"rebuilt baseline anchor differs from the published "
                f"DampedPersistence at h{row['horizon']} by {gap:.4f} degC "
                f"(tolerance {BASELINE_TOLERANCE_DEGC}); the sensitivity "
                "comparison would be meaningless"
            )
    return drift


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)

    panel = load_panel()
    evaluation = load_evaluation()
    keep = reportable_stations()
    clusters = huc2_cluster_map(
        load_station_registry(ROOT / "data_usgs" / "station_registry_v1.csv")
    )

    scored = pd.concat(
        [score_variant(panel, evaluation, variant) for variant in VARIANTS],
        ignore_index=True,
    )
    rows = summarise(scored, clusters, keep)
    drift = check_baseline(rows)

    table = pd.DataFrame(rows)
    seven = table[table["horizon"] == 7].sort_values("median_station_skill_vs_anchor")
    if args.print_only:
        print(f"baseline reconstruction drift (degC): {drift}\n")
        print(seven[[
            "variant", "median_anchor_rmse", "median_station_skill_vs_anchor",
            "skill_ci_low", "skill_ci_high", "n_stations",
        ]].to_string(index=False))
        return 0

    table.to_parquet(args.out, index=False)
    baseline = float(
        table[(table["variant"] == "baseline") & (table["horizon"] == 7)]
        ["median_station_skill_vs_anchor"].iloc[0]
    )
    manifest = {
        "format": FORMAT,
        "builder_sha256": _sha256_file(Path(__file__).resolve()),
        "output_sha256": _sha256_file(args.out),
        "baseline_reconstruction_drift_degC": drift,
        "baseline_seven_day_skill": baseline,
        "strongest_variant_seven_day_skill": float(
            seven["median_station_skill_vs_anchor"].min()
        ),
        "variants": {v.name: v.note for v in VARIANTS},
        "note": (
            "Only the reference model varies; ThermoRoute predictions are the "
            "published ones, so every difference is attributable to the anchor."
        ),
    }
    (args.out.parent / "anchor_sensitivity_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=1) + "\n", encoding="utf-8",
    )
    print(json.dumps({"status": "WRITTEN", "rows": len(table),
                      **{k: v for k, v in manifest.items() if "skill" in k}},
                     sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Stage 17 — audit the strict bounded-deviation model contract.

For the Route-A architecture the named reference is the frozen, train-fitted
damped-persistence anchor.  The learned dynamic prior is only a proposal.  With
finite ``delta`` the point forecast obeys, for every row,

    |forecast - damped_anchor| <= delta.

Consequently ``RMSE(forecast) <= RMSE(anchor) + delta`` by Minkowski.  This is
an algebraic robustness constraint, not a claim of ecological safety, superior
skill, calibration, or operational value.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from thermoroute import config as C
from thermoroute.checkpoint import load_inference_bundle
from thermoroute.repro import sha256_file
from thermoroute.thermoroute import ThermoRoute


_spec = importlib.util.spec_from_file_location(
    "region13c", ROOT / "scripts" / "13c_region_transfer.py")
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load the shared canonical-panel preparation")
R13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R13)

COLOUR = {1: "#185FA5", 3: "#2E8B57", 7: "#993C1D"}


def rmse(y_true: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - prediction) ** 2)))


def main() -> None:
    _, _, _, _, _, windows, stations = R13.prep()
    pointer_path = C.MODELS / "thermoroute_usgs_bundle.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    bundle_path = ROOT / pointer["bundle_path"]
    if sha256_file(bundle_path / "metadata.json") != pointer["metadata_sha256"]:
        raise ValueError("inference bundle metadata does not match its current pointer")
    if sha256_file(bundle_path / "weights.pt") != pointer["weights_sha256"]:
        raise ValueError("inference bundle weights do not match their current pointer")
    members, metadata = load_inference_bundle(
        bundle_path, expected_member_count=int(pointer["member_count"])
    )
    if metadata["run_id"] != pointer["run_id"] or "seed0" not in members:
        raise ValueError("current inference bundle does not contain Route-A seed0")
    architecture = metadata.get("architecture", {})
    if architecture.get("class") != "thermoroute.thermoroute.ThermoRoute":
        raise ValueError("current inference bundle is not a ThermoRoute architecture")
    kwargs = dict(architecture.get("kwargs", {}))
    delta = kwargs.get("delta_scale")
    if kwargs.get("safety_anchor") != "damped" or delta is None:
        raise ValueError("bounded-deviation audit requires a finite damped-anchor bundle")
    if (
        kwargs.get("n_vars") != len(windows.var_names)
        or kwargs.get("n_stations") != len(stations)
        or kwargs.get("n_phys") != windows.n_phys
        or metadata.get("feature_order") != list(windows.var_names)
        or metadata.get("horizons") != list(windows.horizons)
    ):
        raise ValueError("bundle architecture does not match reconstructed canonical windows")
    model = ThermoRoute(
        horizons=tuple(metadata["horizons"]),
        cfg=C.TrainConfig(**architecture.get("train_config", {})),
        **kwargs,
    )
    model.load_state_dict(members["seed0"], strict=True)
    model.eval()
    index = windows.idx("test")
    predictions, anchors = [], []
    with torch.no_grad():
        for start in range(0, len(index), 4096):
            output = model(windows.batch(index[start:start + 4096]))
            predictions.append(output.point.numpy())
            anchors.append(output.prior.numpy())
    prediction = np.concatenate(predictions)
    anchor = np.concatenate(anchors)
    observed = windows.y[index]
    sites = np.array([C.STATIONS[i] for i in windows.station[index]])
    correction = prediction - anchor

    tolerance = 1e-6
    max_abs_correction = float(np.max(np.abs(correction)))
    contract_fraction = float((np.abs(correction) <= delta + tolerance).mean())
    error_bound_fraction = float((
        np.abs(prediction - observed) <= np.abs(anchor - observed) + delta + tolerance
    ).mean())
    if contract_fraction < 1.0 or error_bound_fraction < 1.0:
        raise AssertionError("the saved model violates its strict damped-anchor contract")

    rows: list[dict[str, float | str | int]] = []
    for hi, horizon in enumerate(windows.horizons):
        for site in stations:
            selected = sites == site
            if selected.sum() < 10:
                continue
            anchor_rmse = rmse(observed[selected, hi], anchor[selected, hi])
            model_rmse = rmse(observed[selected, hi], prediction[selected, hi])
            rows.append({
                "site_id": site,
                "horizon": horizon,
                "anchor_rmse": anchor_rmse,
                "model_rmse": model_rmse,
                "rmse_upper_bound": anchor_rmse + delta,
            })
    audit = pd.DataFrame(rows)
    audit["rmse_contract_holds"] = (
        audit.model_rmse <= audit.rmse_upper_bound + tolerance
    )
    if not audit.rmse_contract_holds.all():
        raise AssertionError("a station-level RMSE bound failed")

    C.TABLES.mkdir(parents=True, exist_ok=True)
    audit.to_csv(C.TABLES / "bounded_deviation_audit.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    extent = float(max(audit.anchor_rmse.max(), audit.model_rmse.max()) * 1.05)
    axes[0].plot([0, extent], [0, extent], color="#888", lw=1, label="equal RMSE")
    axes[0].plot(
        [0, extent], [delta, extent + delta], color="#993C1D", ls="--", lw=1.5,
        label=f"anchor RMSE + δ; δ={delta:g} °C",
    )
    for horizon in windows.horizons:
        subset = audit[audit.horizon == horizon]
        axes[0].scatter(
            subset.anchor_rmse, subset.model_rmse, s=22, alpha=0.7,
            color=COLOUR[horizon], label=f"h={horizon} d",
        )
    axes[0].set_xlabel("damped-anchor RMSE (°C)")
    axes[0].set_ylabel("ThermoRoute RMSE (°C)")
    axes[0].set_title("Algebraic RMSE upper bound")
    axes[0].set_xlim(0, extent)
    axes[0].set_ylim(0, extent)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].hist(correction.ravel(), bins=60, color="#185FA5", alpha=0.85)
    axes[1].axvline(-delta, color="#993C1D", ls="--", lw=1.5)
    axes[1].axvline(delta, color="#993C1D", ls="--", lw=1.5)
    axes[1].set_xlabel("forecast − frozen damped anchor (°C)")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"max |correction| = {max_abs_correction:.3f} °C")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / "fig_prop1_binding.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    report = [
        "# Strict bounded-deviation contract audit\n",
        "The saved Route-A seed-0 model was evaluated on the already-inspected "
        "2019–2020 development partition. Its named reference is the frozen, "
        "train-fitted damped-persistence anchor; the learned physics component "
        "is not the reference.\n",
        f"- Pointwise anchor contract: {contract_fraction * 100:.2f}%.",
        f"- Derived pointwise error inequality: {error_bound_fraction * 100:.2f}%.",
        f"- Maximum absolute correction: {max_abs_correction:.4f} °C "
        f"(configured δ={delta:g} °C).",
        f"- Station-by-horizon RMSE inequality: "
        f"{audit.rmse_contract_holds.mean() * 100:.2f}% ({len(audit)} cells).\n",
        "Interpretation: this check proves only that the implementation cannot "
        "move farther than δ from its named anchor. It does not prove better "
        "accuracy, hydrologic extrapolation, calibrated uncertainty, ecological "
        "safety, or economic value. Those require separate empirical tests.",
        "\n![Bounded-deviation audit](../figures/fig_prop1_binding.png)",
    ]
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    (C.REPORTS / "prop1_binding.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()

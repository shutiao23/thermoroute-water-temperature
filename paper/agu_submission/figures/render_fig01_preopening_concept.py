#!/usr/bin/env python3
# TODO(conventional reframe -- later re-render stage): this renderer still uses the
# pre-registration apparatus vocabulary of the prior sealed-confirmatory design.
# When it is reworked to render result figures from the 2021-2023 conventional
# holdout numbers (outputs/conventional/holdout_metrics_2021_2023.csv), reframe the
# following to match paper/FIGURE_REDRAW_SPEC.md (conventional comparative holdout):
#   one-time opening / the opening / sealed opening  -> 2021-2023 holdout scoring
#   trusted scorer (inside the one-time opening)     -> conventional holdout scorer (src/thermoroute/conventional_score.py)
#   verified/opening receipt, outputs/confirmatory/.../opening_receipt_v1.json -> holdout scoring receipt (outputs/conventional/...)
#   opening.py:NNNN-NNNN references                  -> the conventional holdout scorer
#   confirmatory protocol                            -> evaluation protocol; confirmatory ... period -> holdout period
#   claim registry                                   -> comparison registry
#   claim gate / inference gate / DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED / PRE_FROZEN_INFERENCE_GATE / gate verdicts -> descriptive-only scope
#   one-shot boundary                                -> holdout boundary
#   POST/PRE figure-state tokens (POST_TEMPLATE_ONLY, POST gate, POST skeletons) -> holdout-period / structural labels
#   Stage-19 "withheld script" framing               -> the conventional holdout scorer computes the target-period probabilistic family
# No result numbers are filled in this pass; <<...>> / pending cells wait on the metrics CSV.
"""Render the three-panel ThermoRoute opening figure (restructured 2026-08-06).

Panel assignment under the benchmark restructure:
  (a) station map -- 120 retained gauges on a CONUS coordinate scatter, coloured
      by HUC2 with redundant marker encoding and sized by retained 2006--2015
      observed WTEMP day count, with a nearest-neighbour distance inset.
  (b) cluster geometry against the claim gate -- HUC2 counts, three frozen gate
      gauges, and the HUC2/4/6/8 ladder.
  (c) the persistence challenge -- station median |d_h T| by horizon from
      observed exact-day pairs in the 2006--2015 training interval.

The bounded-correction tanh schematic that previously occupied panel (b) has been
relocated to Figure S3(c).  The renderer keeps every frozen-input integrity check
and reuses the cohort geometry, gate, and persistence derivations of the prior
render; only the panel assignment and the ladder extension are new.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


WIDTH_MM = 140
HEIGHT_MM = 92
DPI = 300
MIN_VISIBLE_STROKE_PT = 0.6
# Persistence (panel c) publication scale, declared independent of observed extrema.
PERSIST_Y_MIN_C = 0.0
PERSIST_Y_MAX_C = 2.5
PERSIST_Y_TICK_INTERVAL_C = 0.5
# Station-map (panel a) declared CONUS coordinate limits.
MAP_X_MIN = -126.0
MAP_X_MAX = -69.0
MAP_Y_MIN = 24.0
MAP_Y_MAX = 50.5
WHOLE_REGION_HOLDOUT_MEAN_KM = 289.0
NN_THRESHOLD_KM = 10.0
TRAIN_START = pd.Timestamp("2006-01-01")
TRAIN_END = pd.Timestamp("2015-12-31")
HORIZONS = (1, 3, 7)
FIGURE_ID = "fig01_preopening_concept"
FIGURE_SCHEMA_VERSION = "2.0.0"
EXPECTED_PANEL_SHA256 = "0427a07ea4514ba29ce7d0cf89594e6c35c7f9134cc4d1d96fdc90daeaf5ba69"
EXPECTED_REGISTRY_SHA256 = "090e7c0daf39ac38ceefeb1af8a12c178283e18347905d8e72ada969ad5460c9"
EXPECTED_CONFIG_SHA256 = "7661e82df4a6017dcc351f9f4f07e3b94afd5d2426689f775b83e12b78c41c1f"
EXPECTED_INFERENCE_AMENDMENT_SHA256 = (
    "c936dac301e7f90b05e40bfcb40f86e3a2cd88e692e422fc21fa51fee785bcf8"
)
EXPECTED_CONFIRMATORY_PROTOCOL_SHA256 = (
    "93c32e9dbfe976eaa7ef31cc5181ae1f4a415ad2b2e30674a5df64c46b968c05"
)
EXPECTED_AUDIT_SHA256 = "c221bc67bd988da3bf5f2fcbe34b0e9e38646d7bc239bb130b46e4584f800883"
EXPECTED_HUC_COUNTS = {
    "01": 5,
    "02": 13,
    "03": 14,
    "04": 10,
    "05": 10,
    "06": 2,
    "07": 8,
    "09": 2,
    "10": 7,
    "11": 4,
    "12": 5,
    "14": 8,
    "16": 3,
    "17": 26,
    "18": 3,
}

# HUC2/4/6/8 cluster ladder, verified to every digit in
# docs/OPTION_A_DESCRIPTIVE_BENCHMARK_SCOPE.md section 1 (gate's own geometry).
CLUSTER_LADDER = (
    # (unit, n_clusters, effective_count, effective_fraction, largest_share, passes_arithmetic)
    ("HUC2", 15, 9.536, 0.6358, 0.2167, False),
    ("HUC4", 64, 32.432, 0.5068, 0.1000, False),
    ("HUC6", 75, 36.364, 0.4848, 0.1000, False),
    ("HUC8", 95, 72.000, 0.7579, 0.0417, True),
)

# Okabe--Ito hues, plus neutral ink. Patterns and symbols duplicate color.
BLUE = "#0072B2"
VERMILION = "#D55E00"
TEAL = "#008C7A"
INK = "#202020"
MID = "#60656B"
LIGHT = "#D0D0D0"
PALE_BLUE = "#DCEAF4"
PALE_VERMILION = "#F9E3D6"
PALE_TEAL = "#DCEFEA"
WHITE = "#FFFFFF"
# Full Okabe--Ito set for the 15-way HUC2 map encoding (mirrors Figure S1).
OI_HUC_COLORS = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7")
OI_HUC_MARKERS = ("o", "s", "^", "D", "v", "<", ">", "p")

CAPTION = (
    "Figure 1. Cohort, geometry against the claim gate, and the persistence "
    "challenge. (a) The 120 retained gauges on a CONUS coordinate scatter (no "
    "basemap), coloured by HUC2 region with redundant marker encoding and sized "
    "by retained 2006--2015 observed WTEMP day count; inset: zero-based histogram "
    "of nearest-neighbour distance between retained stations, annotating the "
    "10 km mark (19 stations) and the 289 km whole-region-holdout mean "
    "nearest-training-gauge distance. (b) The 657,480-row panel collapses to 120 "
    "sites across 15 pre-attrition HUC2 groups; three frozen gate checks "
    "(>=30 groups, >=0.75 effective fraction, <25% largest share) are shown as "
    "separate gauges, and a HUC2/4/6/8 ladder (15/64/75/95 clusters; "
    "0.636/0.507/0.485/0.758 effective fractions) marks HUC8 as passing the "
    "arithmetic while adjacent units on one river are not independent. Target "
    "WTEMP at t+h and horizon-specific future weather are excluded predictor "
    "inputs. Two of three gate criteria fail, limiting interpretation to "
    "fixed-cohort descriptive evidence. (c) Across 120 equally weighted "
    "stations, distributions of the station median absolute observed "
    "water-temperature change at 1-, 3-, and 7-day horizons, using exact-day "
    "pairs whose two endpoints fall in the 2006--2015 training interval; this is "
    "the motivation quantity, not a model score."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str, role: str) -> None:
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(
            f"Frozen {role} SHA-256 drifted: expected {expected}, observed {observed}"
        )


def load_frozen_contracts(
    config_path: Path,
    inference_amendment_path: Path,
    confirmatory_protocol_path: Path,
) -> tuple[float, dict[str, object], dict[str, object]]:
    config_tree = ast.parse(config_path.read_text(encoding="utf-8"), filename=str(config_path))
    delta_values: list[object] = []
    for node in config_tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "DELTA_SCALE" and node.value is not None:
                delta_values.append(ast.literal_eval(node.value))
        elif isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "DELTA_SCALE" for target in node.targets):
                delta_values.append(ast.literal_eval(node.value))
    if delta_values != [1.0]:
        raise RuntimeError(
            f"Frozen config must define DELTA_SCALE exactly once as 1.0; observed {delta_values!r}"
        )

    amendment = json.loads(inference_amendment_path.read_text(encoding="utf-8"))
    rule = amendment.get("inference_scope", {}).get("small_cluster_rule")
    verdict = amendment.get("decision_overlay", {}).get("gate_failure_verdict")
    if not isinstance(rule, str):
        raise RuntimeError("Inference amendment lacks small_cluster_rule text")
    patterns = {
        "minimum_reportable_groups": r"n_clusters\s*>=\s*([0-9]+)",
        "minimum_effective_fraction": r"effective_cluster_fraction\s*>=\s*([0-9.]+)",
        "maximum_largest_group_share": r"largest_cluster_share\s*<\s*([0-9.]+)",
    }
    parsed: dict[str, object] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, rule)
        if match is None:
            raise RuntimeError(f"Could not parse {key} from frozen small_cluster_rule")
        parsed[key] = int(match.group(1)) if key == "minimum_reportable_groups" else float(match.group(1))
    parsed["gate_failure_verdict"] = verdict
    expected = {
        "minimum_reportable_groups": 30,
        "minimum_effective_fraction": 0.75,
        "maximum_largest_group_share": 0.25,
        "gate_failure_verdict": "DESCRIPTIVE_ONLY_INFERENCE_GATE_FAILED",
    }
    if parsed != expected:
        raise RuntimeError(f"Frozen inference gate semantics drifted: {parsed!r}")

    protocol = json.loads(confirmatory_protocol_path.read_text(encoding="utf-8"))
    historical = protocol.get("primary_historical_input_contract")
    expected_historical = {
        "information_cutoff": "issue_date_end; no target-date or post-issue values",
        "horizon_specific_future_nwp_consumed": False,
        "operational_replay_claim_allowed": False,
    }
    if not isinstance(historical, dict):
        raise RuntimeError("Confirmatory protocol lacks primary_historical_input_contract")
    parsed_historical = {
        key: historical.get(key) for key in expected_historical
    }
    if parsed_historical != expected_historical:
        raise RuntimeError(
            f"Frozen historical-input semantics drifted: {parsed_historical!r}"
        )
    return float(delta_values[0]), parsed, parsed_historical


def persistence_data(
    panel_path: Path,
) -> tuple[dict[int, np.ndarray], dict[int, int], pd.DataFrame, int, tuple[str, ...]]:
    """Panel-(c) persistence: station median |d_h T| from observed exact-day pairs."""
    panel = pd.read_parquet(panel_path, columns=["DATE", "site_id", "WTEMP"])
    panel_row_count = int(len(panel))
    if panel_row_count != 657_480:
        raise RuntimeError("Figure-1 unit chain requires exactly 657,480 panel rows")
    panel_site_ids = tuple(sorted(panel["site_id"].astype(str).unique()))
    panel["DATE"] = pd.to_datetime(panel["DATE"])
    train = panel.loc[
        panel["DATE"].between(TRAIN_START, TRAIN_END),
        ["DATE", "site_id", "WTEMP"],
    ]
    if train["site_id"].nunique() != 120:
        raise RuntimeError("Panel-c contract requires exactly 120 training stations")
    if train.duplicated(["site_id", "DATE"]).any():
        raise RuntimeError("Panel-c contract requires unique station/date keys")

    observed = train.dropna(subset=["WTEMP"]).copy()
    right = observed.rename(columns={"DATE": "target_date", "WTEMP": "target_wtemp"})
    values: dict[int, np.ndarray] = {}
    pair_counts: dict[int, int] = {}
    mark_rows: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        left = observed.rename(columns={"WTEMP": "issue_wtemp"}).copy()
        left["target_date"] = left["DATE"] + pd.Timedelta(days=horizon)
        pairs = left.merge(
            right[["site_id", "target_date", "target_wtemp"]],
            on=["site_id", "target_date"],
            how="inner",
            validate="one_to_one",
        )
        pairs["absolute_change_c"] = (
            pairs["target_wtemp"] - pairs["issue_wtemp"]
        ).abs()
        by_station = pairs.groupby("site_id", sort=True)["absolute_change_c"].agg(
            station_median_abs_change_c="median", n_pairs="size"
        )
        if len(by_station) != 120:
            raise RuntimeError(f"Horizon {horizon} d does not retain all 120 stations")
        by_station = by_station.reset_index()
        by_station.insert(1, "horizon", horizon)
        mark_rows.append(
            by_station[
                ["site_id", "horizon", "n_pairs", "station_median_abs_change_c"]
            ]
        )
        values[horizon] = by_station["station_median_abs_change_c"].to_numpy(dtype=float)
        pair_counts[horizon] = int(len(pairs))
    projection = pd.concat(mark_rows, ignore_index=True)
    projection = projection.sort_values(["horizon", "site_id"], kind="stable").reset_index(drop=True)
    projection.insert(
        0,
        "mark_id",
        projection.apply(
            lambda row: f"c.station.{row['site_id']}.h{int(row['horizon']):02d}", axis=1
        ),
    )
    projection.insert(
        2,
        "site_value_id",
        projection["site_id"].map(lambda site: f"fig01.c.site.{site}"),
    )
    projection.insert(
        4,
        "x_value_id",
        projection["horizon"].map(lambda h: f"route_a.horizon_{int(h)}_days"),
    )
    projection.insert(
        6,
        "n_pairs_value_id",
        projection.apply(
            lambda row: f"fig01.c.station.{row['site_id']}.h{int(row['horizon']):02d}.n_pairs",
            axis=1,
        ),
    )
    projection["y_value_id"] = projection.apply(
        lambda row: f"fig01.c.station.{row['site_id']}.h{int(row['horizon']):02d}.median_abs_change_c",
        axis=1,
    )
    return values, pair_counts, projection, panel_row_count, panel_site_ids


def observed_wtemp_day_counts(panel_path: Path) -> dict[str, int]:
    """Per-station finite observed WTEMP day count over the 2006--2015 training interval."""
    panel = pd.read_parquet(panel_path, columns=["DATE", "site_id", "WTEMP"])
    panel["DATE"] = pd.to_datetime(panel["DATE"])
    train = panel.loc[
        panel["DATE"].between(TRAIN_START, TRAIN_END), ["site_id", "WTEMP"]
    ].dropna(subset=["WTEMP"])
    counts = train.groupby("site_id").size()
    counts.index = counts.index.astype(str)
    result = {str(site): int(count) for site, count in counts.items()}
    if len(result) != 120:
        raise RuntimeError("Panel-a observed-day-count contract requires exactly 120 stations")
    return result


def _haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    radius = 6371.0088
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius * np.arcsin(np.sqrt(a))


def nearest_neighbor_distances(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    n = len(lat)
    dists = np.full(n, np.inf)
    for i in range(n):
        d = _haversine_km(float(lat[i]), float(lon[i]), lat, lon)
        d[i] = np.inf
        dists[i] = float(np.min(d))
    return dists


def station_map(
    registry_path: Path,
    audit_path: Path,
    observed_counts: dict[str, int],
) -> dict[str, object]:
    registry = pd.read_csv(
        registry_path, dtype={"huc2": "string", "legacy_site_id": "string"}
    )
    if len(registry) != 120:
        raise RuntimeError("Panel-a contract requires exactly 120 registry rows")
    labels = registry["huc2"].str.zfill(2)
    counts = labels.value_counts().sort_index()
    observed = {str(key): int(value) for key, value in counts.items()}
    if observed != EXPECTED_HUC_COUNTS:
        raise RuntimeError(
            "Frozen HUC2 geometry drifted: "
            f"expected {EXPECTED_HUC_COUNTS}, observed {observed}"
        )
    site = registry["site_no"].astype(str).to_numpy()
    legacy = registry["legacy_site_id"].astype(str).to_numpy()
    lat = registry["lat"].astype(float).to_numpy()
    lon = registry["lon"].astype(float).to_numpy()
    huc2 = labels.to_numpy()
    nn = nearest_neighbor_distances(lat, lon)

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("post_2020_values_read") is not False:
        raise RuntimeError("Environmental audit is not outcome-free for panel-a binding")
    nearest = audit["geography"]["nearest_station_distance_km"]
    nn_median = float(np.median(nn))
    nn_min = float(np.min(nn))
    within_10km = int(np.sum(nn < NN_THRESHOLD_KM))
    if not (
        abs(nn_median - nearest["median"]) < 1e-9
        and abs(nn_min - nearest["minimum"]) < 1e-9
        and within_10km == nearest["stations_with_neighbor_within_10km"]
    ):
        raise RuntimeError(
            "Panel-a nearest-neighbour diagnostics drift vs frozen audit: "
            f"median={nn_median} (audit {nearest['median']}), "
            f"min={nn_min} (audit {nearest['minimum']}), "
            f"within_10km={within_10km} (audit {nearest['stations_with_neighbor_within_10km']})"
        )
    day_counts = np.array([int(observed_counts[str(alias)]) for alias in legacy], dtype=int)
    return {
        "site": site,
        "legacy": legacy,
        "lat": lat,
        "lon": lon,
        "huc2": huc2,
        "day_counts": day_counts,
        "nn": nn,
        "nn_median": nn_median,
        "nn_min": nn_min,
        "within_10km": within_10km,
    }


def registry_geometry(registry_path: Path) -> tuple[list[str], np.ndarray, tuple[str, ...]]:
    registry = pd.read_csv(
        registry_path, dtype={"huc2": "string", "legacy_site_id": "string"}
    )
    if len(registry) != 120:
        raise RuntimeError("Panel-b contract requires exactly 120 registry rows")
    labels = registry["huc2"].str.zfill(2)
    counts = labels.value_counts().sort_index()
    observed = {str(key): int(value) for key, value in counts.items()}
    if observed != EXPECTED_HUC_COUNTS:
        raise RuntimeError(
            "Frozen HUC2 geometry drifted: "
            f"expected {EXPECTED_HUC_COUNTS}, observed {observed}"
        )
    aliases = tuple(sorted(registry["legacy_site_id"].astype(str)))
    if len(set(aliases)) != 120:
        raise RuntimeError("Registry legacy_site_id aliases must be unique")
    return list(EXPECTED_HUC_COUNTS), np.array(list(EXPECTED_HUC_COUNTS.values())), aliases


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "thermoroute-figure-1-v3",
            "hatch.linewidth": MIN_VISIBLE_STROKE_PT,
            "savefig.facecolor": WHITE,
            "figure.facecolor": WHITE,
        }
    )


def prepare_panel(ax: mpl.axes.Axes) -> None:
    ax.set_axis_off()
    frame = Rectangle(
        (0.0, 0.0),
        1.0,
        1.0,
        transform=ax.transAxes,
        facecolor=WHITE,
        edgecolor=LIGHT,
        linewidth=0.85,
        clip_on=False,
    )
    ax.add_patch(frame)


def panel_heading(ax: mpl.axes.Axes, letter: str, title: str, subtitle: str) -> None:
    ax.text(
        0.030,
        0.918,
        f"({letter})  {title}",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="center",
        color=INK,
    )
    ax.text(
        0.100,
        0.835,
        subtitle,
        transform=ax.transAxes,
        fontsize=7.5,
        ha="left",
        va="center",
        color=MID,
    )


def _huc_encoding(huc_codes: list[str]) -> list[dict[str, object]]:
    """15-way unique (marker, fill) encoding mirroring Figure S1."""
    encoding: list[dict[str, object]] = []
    for index in range(len(huc_codes)):
        marker = OI_HUC_MARKERS[index % len(OI_HUC_MARKERS)]
        filled = index < len(OI_HUC_MARKERS)
        encoding.append(
            {
                "huc2": huc_codes[index],
                "marker": marker,
                "fill": "filled" if filled else "open",
                "color": OI_HUC_COLORS[index % len(OI_HUC_COLORS)],
            }
        )
    if len({(e["marker"], e["fill"]) for e in encoding}) != len(huc_codes):
        raise RuntimeError("Panel-a HUC2 shape-by-fill encoding is not unique")
    return encoding


def draw_panel_a(
    panel: mpl.axes.Axes,
    map_data: dict[str, object],
) -> None:
    prepare_panel(panel)
    panel_heading(
        panel,
        "a",
        "Station map",
        "120 retained gauges \u2022 coloured by HUC2 \u2022 no basemap",
    )
    ax = panel.inset_axes([0.105, 0.150, 0.820, 0.620])
    huc_codes = sorted(EXPECTED_HUC_COUNTS, key=int)
    huc_codes = [f"{int(c):02d}" for c in huc_codes]
    encoding = _huc_encoding(huc_codes)
    enc_by_code = {e["huc2"]: e for e in encoding}
    day_counts = map_data["day_counts"]
    dc_min = float(day_counts.min())
    dc_max = float(day_counts.max())
    # Map observed-day count to marker area in [18, 78] pt^2.
    def size_for(count: int) -> float:
        if dc_max <= dc_min:
            return 38.0
        return 18.0 + 60.0 * (count - dc_min) / (dc_max - dc_min)

    legend_handles: list[Line2D] = []
    for code in huc_codes:
        e = enc_by_code[code]
        mask = map_data["huc2"] == code
        face = e["color"] if e["fill"] == "filled" else WHITE
        sizes = np.array([size_for(int(c)) for c in day_counts[mask]])
        ax.scatter(
            map_data["lon"][mask],
            map_data["lat"][mask],
            s=sizes,
            marker=e["marker"],
            facecolor=face,
            edgecolor=INK,
            linewidth=0.6,
            alpha=0.92,
            zorder=3,
        )
        legend_handles.append(
            Line2D(
                [0], [0], marker=e["marker"], color="none",
                markerfacecolor=face, markeredgecolor=INK, markeredgewidth=0.6,
                markersize=4.2, label=code,
            )
        )
    ax.set_xlim(MAP_X_MIN, MAP_X_MAX)
    ax.set_ylim(MAP_Y_MIN, MAP_Y_MAX)
    ax.set_xticks([-120, -100, -80, -70])
    ax.set_yticks([25, 35, 45, 50])
    ax.set_xlabel("Longitude (\u00b0E)", labelpad=2)
    ax.set_ylabel("Latitude (\u00b0N)", labelpad=2)
    ax.grid(True, color=LIGHT, linewidth=MIN_VISIBLE_STROKE_PT, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, width=0.65, color=MID)
    ax.text(
        0.025, 0.035, "coordinate scatter \u2022 no basemap",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, color=MID,
        bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 1.0},
    )
    ax.legend(
        handles=legend_handles, title="HUC2", ncol=8, loc="upper center",
        bbox_to_anchor=(0.5, -0.165), frameon=False, columnspacing=0.5,
        handletextpad=0.1, borderaxespad=0.0, title_fontsize=7.5,
    )

    # Inset: zero-based histogram of nearest-neighbour distance.
    nn = map_data["nn"]
    inset = panel.inset_axes([0.085, 0.535, 0.370, 0.300])
    bins = np.linspace(0.0, max(float(nn.max()), WHOLE_REGION_HOLDOUT_MEAN_KM) * 1.02, 16)
    inset.hist(nn, bins=bins, color=PALE_BLUE, edgecolor=BLUE, linewidth=0.7)
    inset.axvline(
        NN_THRESHOLD_KM, color=VERMILION, linestyle=(0, (4, 2)), linewidth=0.9
    )
    inset.axvline(
        WHOLE_REGION_HOLDOUT_MEAN_KM, color=TEAL, linestyle=(0, (1, 2)), linewidth=0.9
    )
    inset.set_title("Nearest-neighbour distance (km)", fontsize=7.5, pad=2)
    inset.set_xticks([0, 100, 200, 300])
    inset.set_yticks([])
    inset.tick_params(length=2.0, width=0.6, color=MID, labelsize=7.5)
    inset.spines[["top", "right"]].set_visible(False)
    label_bbox = {"facecolor": WHITE, "edgecolor": "none", "pad": 0.8, "alpha": 0.92}
    inset.text(
        0.015, 0.94, "10 km\n19 stns", transform=inset.transAxes,
        fontsize=7.5, color=VERMILION, ha="left", va="top", bbox=label_bbox,
    )
    inset.text(
        0.985, 0.94, "289 km\nholdout mean", transform=inset.transAxes,
        fontsize=7.5, color=TEAL, ha="right", va="top", bbox=label_bbox,
    )


def gate_row(
    ax: mpl.axes.Axes,
    y: float,
    text: str,
    passed: bool,
    height: float = 0.105,
    guard_id: str | None = None,
) -> None:
    color = TEAL if passed else VERMILION
    fill = PALE_TEAL if passed else PALE_VERMILION
    symbol = "\u2713" if passed else "\u00d7"
    box = Rectangle(
        (0.585, y - height / 2),
        0.380,
        height,
        transform=ax.transAxes,
        facecolor=fill,
        edgecolor=color,
        linewidth=0.85,
    )
    if guard_id is not None:
        box.set_gid(f"guard.{guard_id}.box")
    ax.add_patch(box)
    ax.text(
        0.610,
        y,
        symbol,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color=color,
        ha="center",
        va="center",
    )
    gate_text = ax.text(
        0.642,
        y,
        text,
        transform=ax.transAxes,
        fontsize=7.5,
        color=INK,
        ha="left",
        va="center",
    )
    if guard_id is not None:
        gate_text.set_gid(f"guard.{guard_id}.text")


def draw_panel_b(
    ax: mpl.axes.Axes,
    huc_labels: list[str],
    counts: np.ndarray,
    gate_contract: dict[str, object],
    historical_input_contract: dict[str, object],
) -> None:
    prepare_panel(ax)
    panel_heading(
        ax,
        "b",
        "Cluster geometry vs the claim gate",
        "657,480 rows \u2192 120 stations \u2192 15 pre-attrition HUC2 groups",
    )

    bars_ax = ax.inset_axes([0.060, 0.355, 0.470, 0.200])
    bars_ax.patch.set_gid("guard.bar_axes.patch")
    x = np.arange(len(huc_labels))
    facecolors = [PALE_BLUE] * len(huc_labels)
    edgecolors = [BLUE] * len(huc_labels)
    hatches = ["//"] * len(huc_labels)
    largest = int(np.argmax(counts))
    facecolors[largest] = "#C6DFEC"
    edgecolors[largest] = "#004F7C"
    hatches[largest] = "xx"
    bars = bars_ax.bar(
        x,
        counts,
        width=0.70,
        color=facecolors,
        edgecolor=edgecolors,
        linewidth=0.85,
    )
    for bar, hatch, count, huc_label in zip(bars, hatches, counts, huc_labels):
        bar.set_hatch(hatch)
        bar_label = bars_ax.text(
            bar.get_x() + bar.get_width() / 2,
            float(count) + 0.75,
            str(int(count)),
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=INK,
        )
        bar_label.set_gid(f"guard.bar_label.{huc_label}")
    bars_ax.set_ylim(0, 30)
    bars_ax.set_xticks(x, huc_labels)
    bars_ax.set_yticks([0, 10, 20, 30])
    bars_ax.set_ylabel("Stations")
    bars_ax.grid(axis="y", color=LIGHT, linewidth=MIN_VISIBLE_STROKE_PT)
    bars_ax.set_axisbelow(True)
    bars_ax.spines[["top", "right"]].set_visible(False)
    bars_ax.tick_params(length=2.3, width=0.65, color=MID, pad=1.5)

    if (
        historical_input_contract["information_cutoff"]
        != "issue_date_end; no target-date or post-issue values"
        or historical_input_contract["horizon_specific_future_nwp_consumed"] is not False
    ):
        raise RuntimeError("Excluded-input annotation lacks its frozen contract semantics")
    excluded_box = Rectangle(
        (0.025, 0.630),
        0.550,
        0.200,
        transform=ax.transAxes,
        facecolor=PALE_VERMILION,
        edgecolor=VERMILION,
        linewidth=0.8,
        hatch="//",
        zorder=5,
    )
    excluded_box.set_gid("guard.excluded_inputs.box")
    ax.add_patch(excluded_box)
    excluded_status = ax.text(
        0.300,
        0.760,
        "EXCLUDED FROM INPUT",
        transform=ax.transAxes,
        fontsize=7.5,
        fontweight="bold",
        color=VERMILION,
        ha="center",
        va="center",
        zorder=6,
    )
    excluded_status.set_gid("guard.excluded_inputs.status")
    excluded_text = ax.text(
        0.300,
        0.700,
        "target WTEMP at t+h \u2022 horizon-specific future weather",
        transform=ax.transAxes,
        fontsize=7.5,
        fontstretch="condensed",
        color=VERMILION,
        ha="center",
        va="center",
        zorder=6,
    )
    excluded_text.set_gid("guard.excluded_inputs.text")

    ax.text(
        0.585,
        0.815,
        "Frozen gate",
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        color=INK,
        ha="left",
        va="bottom",
    )
    effective_count = float(1.0 / np.sum((counts / counts.sum()) ** 2))
    effective_fraction = effective_count / len(counts)
    largest_share = float(np.max(counts) / counts.sum())
    required_groups = int(gate_contract["minimum_reportable_groups"])
    minimum_effective = float(gate_contract["minimum_effective_fraction"])
    maximum_share = float(gate_contract["maximum_largest_group_share"])
    gate_row(
        ax,
        0.720,
        f"\u2264{len(counts)} pre-attrition groups\n<{required_groups} required for reporting",
        passed=len(counts) >= required_groups,
        height=0.160,
        guard_id="gate_groups",
    )
    gate_row(
        ax,
        0.575,
        f"{effective_count:.2f}/{len(counts)}={effective_fraction:.3f}<{minimum_effective:.2f}",
        passed=effective_fraction >= minimum_effective,
        height=0.110,
    )
    gate_row(
        ax,
        0.445,
        f"{largest_share:.1%}<{maximum_share:.0%}",
        passed=largest_share < maximum_share,
        height=0.110,
    )
    ax.text(
        0.775,
        0.365,
        "FIXED-COHORT DESCRIPTIVE ONLY",
        transform=ax.transAxes,
        fontsize=7.5,
        fontweight="bold",
        color=VERMILION,
        ha="center",
        va="center",
    )

    # HUC2/4/6/8 cluster ladder: full-width 3-line cells, all >=7.5 pt.
    ladder_y = 0.165
    ladder_h = 0.165
    cell_w = 0.216
    cell_gap = 0.012
    start_x = 0.060
    for index, (unit, n_clust, eff_count, eff_frac, lshare, passes) in enumerate(CLUSTER_LADDER):
        cx = start_x + index * (cell_w + cell_gap)
        edge = TEAL if passes else MID
        face = PALE_TEAL if passes else "#EFEFEF"
        cell = Rectangle(
            (cx, ladder_y), cell_w, ladder_h,
            transform=ax.transAxes, facecolor=face, edgecolor=edge, linewidth=0.85,
        )
        ax.add_patch(cell)
        symbol = "\u2713" if passes else "\u00d7"
        ax.text(
            cx + cell_w / 2, ladder_y + ladder_h - 0.022, f"{symbol} {unit}",
            transform=ax.transAxes, fontsize=7.5, fontweight="bold",
            color=INK if passes else edge, ha="center", va="top",
        )
        ax.text(
            cx + cell_w / 2, ladder_y + ladder_h / 2, f"{n_clust} clusters",
            transform=ax.transAxes, fontsize=7.5, color=INK,
            ha="center", va="center",
        )
        eff_line = f"eff {eff_frac:.3f}"
        if unit == "HUC8":
            eff_line = f"eff {eff_frac:.3f} \u00b7 not indep."
        ax.text(
            cx + cell_w / 2, ladder_y + 0.028, eff_line,
            transform=ax.transAxes, fontsize=7.5,
            color=INK if passes else MID, ha="center", va="center",
        )

    node_labels = (
        "dated inputs\n\u2264 t",
        "exact\nkeys",
        "station\neffects",
        "HUC\nsensitivity",
        "claim\ngate",
    )
    node_x = np.linspace(0.115, 0.885, len(node_labels))
    for index, (node, label) in enumerate(zip(node_x, node_labels)):
        box = Rectangle(
            (node - 0.073, 0.030),
            0.146,
            0.115,
            transform=ax.transAxes,
            facecolor=PALE_TEAL,
            edgecolor=TEAL,
            linewidth=0.8,
        )
        ax.add_patch(box)
        ax.text(
            node,
            0.087,
            label,
            transform=ax.transAxes,
            fontsize=7.5,
            color=INK,
            ha="center",
            va="center",
            linespacing=0.95,
        )
        if index < len(node_labels) - 1:
            ax.text(
                (node + node_x[index + 1]) / 2,
                0.087,
                "\u2192",
                transform=ax.transAxes,
                fontsize=9,
                color=TEAL,
                fontweight="bold",
                ha="center",
                va="center",
            )


def draw_panel_c(panel: mpl.axes.Axes, values: dict[int, np.ndarray]) -> None:
    prepare_panel(panel)
    panel_heading(
        panel,
        "c",
        "Persistence challenge",
        "2006--2015 exact pairs \u2022 120 stations \u2022 motivation quantity",
    )
    ax = panel.inset_axes([0.185, 0.135, 0.770, 0.615])
    colors = (BLUE, "#3D86B8", "#005A8D")
    pale = ("#E7F1F7", PALE_BLUE, "#D2E5F0")
    markers = ("o", "s", "^")
    hatches = ("//", "xx", "..")
    distributions = [values[h] for h in HORIZONS]
    observed = np.concatenate(distributions)
    if (
        not np.all(np.isfinite(observed))
        or float(np.min(observed)) < PERSIST_Y_MIN_C
        or float(np.max(observed)) > PERSIST_Y_MAX_C
    ):
        raise RuntimeError(
            "Panel-c station points fall outside the fixed 0--2.5 degrees C publication scale"
        )
    boxes = ax.boxplot(
        distributions,
        positions=np.arange(1, 4),
        widths=0.48,
        whis=(5, 95),
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "linewidth": 1.5},
        whiskerprops={"color": MID, "linewidth": 0.9},
        capprops={"color": MID, "linewidth": 0.9},
    )
    rng = np.random.default_rng(240731)
    for index, (vals, color, fill, marker, hatch) in enumerate(
        zip(distributions, colors, pale, markers, hatches), start=1
    ):
        patch = boxes["boxes"][index - 1]
        patch.set(facecolor=fill, edgecolor=color, linewidth=1.1, hatch=hatch)
        jitter = rng.uniform(-0.16, 0.16, size=len(vals))
        ax.scatter(
            index + jitter,
            vals,
            s=8,
            marker=marker,
            facecolor=WHITE,
            edgecolor=color,
            linewidth=MIN_VISIBLE_STROKE_PT,
            alpha=0.62,
            zorder=2,
        )
        median = float(np.median(vals))
        ax.text(
            index,
            2.28,
            f"med. {median:.2f}",
            color=color,
            fontsize=7.5,
            fontweight="bold",
            ha="center",
            va="bottom",
        )

    ax.set_xlim(0.52, 3.48)
    ax.set_ylim(PERSIST_Y_MIN_C, PERSIST_Y_MAX_C)
    ax.set_xticks([1, 2, 3], ["h = 1 d", "h = 3 d", "h = 7 d"])
    ax.set_yticks(
        np.arange(
            PERSIST_Y_MIN_C,
            PERSIST_Y_MAX_C + PERSIST_Y_TICK_INTERVAL_C / 2,
            PERSIST_Y_TICK_INTERVAL_C,
        )
    )
    ax.set_ylabel("Station median  |\u0394\u2095T|  (\u00b0C)", labelpad=2)
    ax.grid(axis="y", color=LIGHT, linewidth=MIN_VISIBLE_STROKE_PT)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, width=0.65, color=MID)


def build_figure(
    map_data: dict[str, object],
    values: dict[int, np.ndarray],
    labels: list[str],
    counts: np.ndarray,
    gate_contract: dict[str, object],
    historical_input_contract: dict[str, object],
) -> plt.Figure:
    configure_matplotlib()
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=DPI)
    ax_a = fig.add_axes([0.035, 0.615, 0.450, 0.350])  # (a) station map, top-left
    ax_c = fig.add_axes([0.515, 0.615, 0.450, 0.350])  # (c) persistence, top-right
    ax_b = fig.add_axes([0.035, 0.045, 0.930, 0.540])  # (b) geometry/gate, bottom full
    draw_panel_a(ax_a, map_data)
    draw_panel_b(ax_b, labels, counts, gate_contract, historical_input_contract)
    draw_panel_c(ax_c, values)
    validate_visual_bounds(fig)
    return fig


def validate_visual_bounds(fig: plt.Figure) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    artists = {artist.get_gid(): artist for artist in fig.findobj() if artist.get_gid()}
    for guard_id, minimum_padding_mm, label_suffixes in (
        ("gate_groups", 0.5, ("text",)),
        ("excluded_inputs", 2.0, ("status", "text")),
    ):
        box = artists.get(f"guard.{guard_id}.box")
        labels = [artists.get(f"guard.{guard_id}.{suffix}") for suffix in label_suffixes]
        if box is None or any(label is None for label in labels):
            raise RuntimeError(f"Missing visual bbox guard artists for {guard_id}")
        box_bbox = box.get_window_extent(renderer)
        padding_px = fig.dpi * minimum_padding_mm / 25.4
        for label in labels:
            label_bbox = label.get_window_extent(renderer)
            if not (
                label_bbox.x0 >= box_bbox.x0 + padding_px
                and label_bbox.x1 <= box_bbox.x1 - padding_px
                and label_bbox.y0 >= box_bbox.y0 + padding_px
                and label_bbox.y1 <= box_bbox.y1 - padding_px
            ):
                raise RuntimeError(
                    f"Visual bbox overflow for {guard_id}: label={label_bbox.bounds}, "
                    f"box={box_bbox.bounds}, required_padding_mm={minimum_padding_mm}"
                )

    excluded_bbox = artists["guard.excluded_inputs.box"].get_window_extent(renderer)
    bar_axes_bbox = artists["guard.bar_axes.patch"].get_window_extent(renderer)
    excluded_separation_px = fig.dpi * 2.0 / 25.4
    if excluded_bbox.y0 - bar_axes_bbox.y1 < excluded_separation_px:
        raise RuntimeError(
            "Excluded-input band is too close to the bar axes: "
            f"box={excluded_bbox.bounds}, bars={bar_axes_bbox.bounds}, "
            "required_separation_mm=2.0"
        )
    bar_labels = [
        artist for gid, artist in artists.items() if gid.startswith("guard.bar_label.")
    ]
    if len(bar_labels) != len(EXPECTED_HUC_COUNTS):
        raise RuntimeError("Excluded-input guard requires all 15 HUC2 bar labels")
    for bar_label in bar_labels:
        bar_label_bbox = bar_label.get_window_extent(renderer)
        if excluded_bbox.y0 - bar_label_bbox.y1 < excluded_separation_px:
            raise RuntimeError(
                "Excluded-input band overlaps a HUC2 bar label: "
                f"box={excluded_bbox.bounds}, label={bar_label_bbox.bounds}, "
                "required_separation_mm=2.0"
            )


def make_svg_accessible(svg_path: Path) -> None:
    content = svg_path.read_text(encoding="utf-8")
    content = re.sub(
        r'(<svg\b[^>]*?)width="[^"]+" height="[^"]+"',
        rf'\g<1>width="{WIDTH_MM}mm" height="{HEIGHT_MM}mm"',
        content,
        count=1,
    )
    content = re.sub(r"\n <title>.*?</title>", "", content, count=1, flags=re.DOTALL)
    root_match = re.search(r"<svg\b[^>]*>", content)
    if root_match is None:
        raise RuntimeError("Generated SVG has no root element")
    root = root_match.group(0)
    accessible_root = root[:-1] + ' role="img" aria-labelledby="fig-title fig-desc">'
    accessible_text = (
        "\n <title id=\"fig-title\">Station map, cluster geometry against the claim gate, "
        "and the persistence challenge</title>"
        "\n <desc id=\"fig-desc\">Three panels show the 120-station cohort map with a "
        "nearest-neighbour distance inset, the registry-derived HUC2 counts with three "
        "frozen inference-gate gauges and a HUC2/4/6/8 cluster ladder, and observed "
        "training-period water-temperature persistence by horizon.</desc>"
    )
    content = content[: root_match.start()] + accessible_root + accessible_text + content[root_match.end() :]
    content = "\n".join(line.rstrip() for line in content.splitlines()) + "\n"
    if len(re.findall(r"<title(?:\s|>)", content)) != 1:
        raise RuntimeError("Accessible SVG must contain exactly one unprefixed title element")
    svg_path.write_text(content, encoding="utf-8", newline="\n")


def validate_svg_visible_strokes(svg_path: Path) -> float:
    root = ElementTree.parse(svg_path).getroot()
    graphical_tags = {
        "circle",
        "ellipse",
        "line",
        "path",
        "polygon",
        "polyline",
        "rect",
        "text",
        "use",
    }
    failures: list[str] = []
    visible_widths: list[float] = []

    def parse_style(style: str) -> dict[str, str]:
        declarations: dict[str, str] = {}
        for item in style.split(";"):
            if ":" not in item:
                continue
            key, value = item.split(":", 1)
            declarations[key.strip()] = value.strip()
        return declarations

    def numeric(value: str, default: float) -> float:
        match = re.match(r"^[ ]*([0-9]+(?:\.[0-9]+)?)", value)
        return float(match.group(1)) if match else default

    def visit(element: ElementTree.Element, inherited: dict[str, str]) -> None:
        properties = dict(inherited)
        properties.update(parse_style(element.attrib.get("style", "")))
        for key in (
            "display",
            "opacity",
            "stroke",
            "stroke-opacity",
            "stroke-width",
            "visibility",
        ):
            if key in element.attrib:
                properties[key] = element.attrib[key]
        hidden = (
            properties.get("display") == "none"
            or properties.get("visibility") == "hidden"
            or numeric(properties.get("opacity", "1"), 1.0) == 0
        )
        tag = element.tag.rsplit("}", 1)[-1]
        stroke = properties.get("stroke")
        stroke_visible = (
            not hidden
            and tag in graphical_tags
            and stroke is not None
            and stroke.lower() not in {"none", "transparent"}
            and numeric(properties.get("stroke-opacity", "1"), 1.0) > 0
        )
        if stroke_visible:
            width = numeric(properties.get("stroke-width", "1"), 1.0)
            visible_widths.append(width)
            if width + 1e-12 < MIN_VISIBLE_STROKE_PT:
                failures.append(
                    f"tag={tag} id={element.attrib.get('id', '<none>')} stroke-width={width}"
                )
        for child in element:
            visit(child, properties)

    visit(root, {})
    if failures:
        raise RuntimeError(
            "SVG contains visible strokes below "
            f"{MIN_VISIBLE_STROKE_PT:.1f} pt: " + "; ".join(failures[:12])
        )
    if not visible_widths:
        raise RuntimeError("SVG visible-stroke guard found no stroked graphical elements")
    return min(visible_widths)


def summary(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum_c": round(float(np.min(values)), 6),
        "q25_c": round(float(np.quantile(values, 0.25)), 6),
        "median_c": round(float(np.median(values)), 6),
        "q75_c": round(float(np.quantile(values, 0.75)), 6),
        "maximum_c": round(float(np.max(values)), 6),
        "mean_c": round(float(np.mean(values)), 6),
    }


def boxplot_summary(values: np.ndarray) -> dict[str, float]:
    stats = mpl.cbook.boxplot_stats(values, whis=(5, 95))[0]
    return {
        "whisker_low_c": float(stats["whislo"]),
        "q25_c": float(stats["q1"]),
        "median_c": float(stats["med"]),
        "q75_c": float(stats["q3"]),
        "whisker_high_c": float(stats["whishi"]),
    }


def bound_value(
    value: object,
    unit: str,
    evidence_role: str,
    source_pointer: str,
    derivation: str,
    rounding: str,
) -> dict[str, object]:
    return {
        "value": value,
        "unit": unit,
        "evidence_role": evidence_role,
        "source_pointer": source_pointer,
        "derivation": derivation,
        "rounding": rounding,
    }


EXPECTED_MARK_COUNTS = {
    ("a", "text_annotation"): 1,
    ("a", "map_axis"): 1,
    ("a", "station_marker"): 120,
    ("a", "nearest_neighbor_histogram"): 1,
    ("a", "scope_status"): 1,
    ("b", "unit_chain_annotation"): 1,
    ("b", "zero_based_count_bar"): 15,
    ("b", "excluded_predictor_inputs"): 1,
    ("b", "gate_check"): 3,
    ("b", "cluster_ladder_entry"): 4,
    ("b", "cluster_ladder_note"): 1,
    ("b", "scope_status"): 1,
    ("b", "evidence_spine_node"): 5,
    ("c", "text_annotation"): 1,
    ("c", "axis_scale"): 1,
    ("c", "station_point"): 360,
    ("c", "box_and_median_summary"): 3,
}
EXPECTED_VALUE_COUNT = 1673
EXPECTED_MARK_COUNT = 520


def validate_binder_document(document: dict[str, object], mark_data: pd.DataFrame) -> None:
    required_value_fields = {
        "value",
        "unit",
        "evidence_role",
        "source_pointer",
        "derivation",
        "rounding",
    }
    registry = document["values"]
    if not isinstance(registry, dict) or not registry:
        raise RuntimeError("Figure binder values registry is empty")
    for value_id, record in registry.items():
        if not isinstance(record, dict) or not required_value_fields.issubset(record):
            raise RuntimeError(f"Incomplete value object: {value_id}")
    required_shared_ids = {
        "route_a.panel.row_count",
        "route_a.registry.station_count",
        "route_a.registry.huc2_group_count",
        *(f"route_a.registry.huc2_{code}.station_count" for code in EXPECTED_HUC_COUNTS),
    }
    if not required_shared_ids.issubset(registry):
        raise RuntimeError("Missing canonical cross-figure cohort value_id")

    figure = document["figures"][FIGURE_ID]
    registered_marks = document["mark_registry"]
    mark_ids = [entry["mark_id"] for entry in registered_marks]
    if (
        figure["panel_order"] != document["panel_order"]
        or figure["mark_registry"] != mark_ids
        or figure["caption_value_ids"] != document["caption_value_ids"]
        or figure["scope_status_value_id"] != document["scope_status_value_id"]
        or figure["render_profile"] != document["render_profile"]
    ):
        raise RuntimeError("Nested figure binder does not mirror top-level contract fields")
    if len(mark_ids) != len(set(mark_ids)):
        raise RuntimeError("Duplicate mark_id in mark_registry")
    observed_mark_counts: dict[tuple[str, str], int] = {}
    for entry in registered_marks:
        key = (entry["panel_id"], entry["mark_type"])
        observed_mark_counts[key] = observed_mark_counts.get(key, 0) + 1
    if observed_mark_counts != EXPECTED_MARK_COUNTS:
        raise RuntimeError(
            f"Figure-1 mark coverage drifted: {observed_mark_counts!r}"
        )
    for panel_id in document["panel_order"]:
        marks = figure["panels"][panel_id]["marks"]
        for mark_id, mark in marks.items():
            if mark_id not in mark_ids:
                raise RuntimeError(f"Unregistered mark: {mark_id}")
            for cell_name, cell in mark["cells"].items():
                value_id = cell.get("value_id")
                if value_id not in registry:
                    raise RuntimeError(
                        f"Missing value_id {value_id!r} for {mark_id}.{cell_name}"
                    )
    if set(mark_ids) != {
        mark_id
        for panel in figure["panels"].values()
        for mark_id in panel["marks"]
    }:
        raise RuntimeError("mark_registry and panel mark dictionaries disagree")
    if any(value_id not in registry for value_id in document["caption_value_ids"]):
        raise RuntimeError("caption_value_ids contains an unknown value_id")
    if document["scope_status_value_id"] not in registry:
        raise RuntimeError("scope_status_value_id is not registered")
    if len(document["source_bindings"]) != 10:
        raise RuntimeError("Figure-1 binder requires exactly ten source bindings")
    if len(registry) != EXPECTED_VALUE_COUNT or len(registered_marks) != EXPECTED_MARK_COUNT:
        raise RuntimeError(
            f"Figure-1 binder coverage drifted: observed {len(registry)} values / "
            f"{len(registered_marks)} marks; expected {EXPECTED_VALUE_COUNT} / {EXPECTED_MARK_COUNT}"
        )

    required_csv_ids = {
        "site_value_id",
        "x_value_id",
        "y_value_id",
        "n_pairs_value_id",
    }
    if not required_csv_ids.issubset(mark_data.columns):
        raise RuntimeError("Panel-c projection is missing value_id columns")
    if len(mark_data) != 360 or mark_data["mark_id"].nunique() != 360:
        raise RuntimeError("Panel-c binder requires exactly 360 unique station marks")
    for row in mark_data.itertuples(index=False):
        mark = figure["panels"]["c"]["marks"][row.mark_id]
        expected = {
            "site": row.site_value_id,
            "x": row.x_value_id,
            "y": row.y_value_id,
            "n_pairs": row.n_pairs_value_id,
        }
        observed = {name: mark["cells"][name]["value_id"] for name in expected}
        if observed != expected:
            raise RuntimeError(f"CSV/JSON binder disagreement for {row.mark_id}")


def write_sidecar(
    sidecar_path: Path,
    repo_root: Path,
    renderer_path: Path,
    panel_path: Path,
    registry_path: Path,
    audit_path: Path,
    option_a_path: Path,
    mark_data_path: Path,
    panel_row_count: int,
    gate_contract: dict[str, object],
    historical_input_contract: dict[str, object],
    artifact_paths: list[Path],
    values: dict[int, np.ndarray],
    pair_counts: dict[int, int],
    mark_data: pd.DataFrame,
    huc_labels: list[str],
    huc_counts: np.ndarray,
    map_data: dict[str, object],
    observed_counts: dict[str, int],
) -> None:
    shares = huc_counts / huc_counts.sum()
    effective_count = float(1.0 / np.sum(shares**2))
    effective_fraction = effective_count / len(huc_counts)
    largest_share = float(np.max(shares))
    minimum_reportable_groups = int(gate_contract["minimum_reportable_groups"])
    minimum_effective_fraction = float(gate_contract["minimum_effective_fraction"])
    maximum_largest_share = float(gate_contract["maximum_largest_group_share"])
    gate_failure_verdict = str(gate_contract["gate_failure_verdict"])
    if historical_input_contract != {
        "information_cutoff": "issue_date_end; no target-date or post-issue values",
        "horizon_specific_future_nwp_consumed": False,
        "operational_replay_claim_allowed": False,
    }:
        raise RuntimeError("Historical input contract drifted before binder construction")
    group_count_passed = len(huc_counts) >= minimum_reportable_groups
    effective_fraction_passed = effective_fraction >= minimum_effective_fraction
    largest_share_passed = largest_share < maximum_largest_share
    source_specs = [
        (
            "route_a_panel",
            panel_path,
            "parquet",
            "frozen development panel; training-only mismatch, persistence pairs, and full row count",
        ),
        (
            "route_a_registry",
            registry_path,
            "csv",
            "frozen station aliases, coordinates, HUC2 geometry, and day counts",
        ),
        (
            "development_environmental_audit",
            audit_path,
            "json",
            "PRE audit of panel/cohort geometry and nearest-neighbour diagnostics",
        ),
        (
            "thermoroute_config",
            repo_root / "src" / "thermoroute" / "config.py",
            "python",
            "frozen delta_scale configuration contract",
        ),
        (
            "inference_amendment",
            repo_root / "protocols" / "route_a_inference_amendment_v2.json",
            "json",
            "frozen gate thresholds and failed-gate disposition",
        ),
        (
            "confirmatory_protocol",
            repo_root / "protocols" / "route_a_confirmatory_v1.json",
            "json",
            "frozen historical predictor-input and information-cutoff contract",
        ),
        (
            "claim_registry",
            repo_root / "protocols" / "route_a_claim_registry_v1.json",
            "json",
            "permanent claim and scope limitations",
        ),
        (
            "figure_redraw_spec",
            repo_root / "paper" / "FIGURE_REDRAW_SPEC.md",
            "markdown",
            "authoritative Figure 1 redraw and binder contract",
        ),
        (
            "option_a_scope",
            option_a_path,
            "markdown",
            "verified HUC2/4/6/8 cluster ladder geometry",
        ),
        (
            "fig01_renderer",
            renderer_path,
            "python",
            "deterministic figure projection and rendering",
        ),
    ]
    source_bindings = [
        {
            "source_id": source_id,
            "path": path.relative_to(repo_root).as_posix(),
            "sha256": sha256(path),
            "format": source_format,
            "semantic_role": semantic_role,
        }
        for source_id, path, source_format, semantic_role in source_specs
    ]

    value_registry: dict[str, dict[str, object]] = {}
    panel_marks: dict[str, dict[str, dict[str, object]]] = {
        "a": {},
        "b": {},
        "c": {},
    }
    mark_registry: list[dict[str, object]] = []

    def add_value(value_id: str, record: dict[str, object]) -> str:
        if value_id in value_registry and value_registry[value_id] != record:
            raise RuntimeError(f"Conflicting definition for value_id {value_id}")
        value_registry[value_id] = record
        return value_id

    def add_mark(
        panel_id: str,
        mark_id: str,
        mark_type: str,
        cells: dict[str, str],
    ) -> None:
        if mark_id in panel_marks[panel_id]:
            raise RuntimeError(f"Duplicate mark_id {mark_id}")
        panel_marks[panel_id][mark_id] = {
            "mark_type": mark_type,
            "cells": {name: {"value_id": value_id} for name, value_id in cells.items()},
        }
        mark_registry.append(
            {
                "mark_id": mark_id,
                "panel_id": panel_id,
                "mark_type": mark_type,
                "cell_fields": list(cells),
            }
        )

    shared_row_count = add_value(
        "route_a.panel.row_count",
        bound_value(
            panel_row_count,
            "daily panel rows",
            "PRE_FROZEN_PANEL_GEOMETRY",
            "source:route_a_panel#row_count",
            "Count all rows in the frozen 2006-01-01/2020-12-31 panel.",
            "integer; no rounding",
        ),
    )
    shared_station_count = add_value(
        "route_a.registry.station_count",
        bound_value(
            int(huc_counts.sum()),
            "stations",
            "PRE_FROZEN_COHORT_GEOMETRY",
            "source:route_a_registry#legacy_site_id",
            "Count unique frozen legacy_site_id aliases; validated equal to panel site_id.",
            "integer; no rounding",
        ),
    )
    shared_huc_count = add_value(
        "route_a.registry.huc2_group_count",
        bound_value(
            len(huc_counts),
            "pre-attrition HUC2 groups",
            "PRE_FROZEN_COHORT_GEOMETRY",
            "source:route_a_registry#huc2",
            "Zero-pad HUC2 labels to two digits and count unique pre-attrition groups.",
            "integer; no rounding",
        ),
    )
    train_start_id = add_value(
        "route_a.training.start_date",
        bound_value(
            str(TRAIN_START.date()),
            "calendar date",
            "PRE_TRAINING_ONLY_DESCRIPTIVE",
            "source:route_a_panel#DATE",
            "Frozen inclusive lower bound of the training-only pair interval.",
            "ISO 8601 day; no rounding",
        ),
    )
    train_end_id = add_value(
        "route_a.training.end_date",
        bound_value(
            str(TRAIN_END.date()),
            "calendar date",
            "PRE_TRAINING_ONLY_DESCRIPTIVE",
            "source:route_a_panel#DATE",
            "Frozen inclusive upper bound of the training-only pair interval.",
            "ISO 8601 day; no rounding",
        ),
    )

    # ---- Panel (a) station map ------------------------------------------------
    add_mark(
        "a",
        "a.subtitle.cohort",
        "text_annotation",
        {"start_date": train_start_id, "end_date": train_end_id, "site_count": shared_station_count},
    )
    map_axis_ids = {
        "x_min": add_value(
            "fig01.a.axis.x_min_lon",
            bound_value(
                MAP_X_MIN, "degrees east", "PRE_DECLARED_PUBLICATION_SCALE",
                "source:figure_redraw_spec#Figure-1-panel-a",
                "Fixed CONUS longitude publication extent declared independent of data.",
                "exact axis endpoint; one decimal place",
            ),
        ),
        "x_max": add_value(
            "fig01.a.axis.x_max_lon",
            bound_value(
                MAP_X_MAX, "degrees east", "PRE_DECLARED_PUBLICATION_SCALE",
                "source:figure_redraw_spec#Figure-1-panel-a",
                "Fixed CONUS longitude publication extent declared independent of data.",
                "exact axis endpoint; one decimal place",
            ),
        ),
        "y_min": add_value(
            "fig01.a.axis.y_min_lat",
            bound_value(
                MAP_Y_MIN, "degrees north", "PRE_DECLARED_PUBLICATION_SCALE",
                "source:figure_redraw_spec#Figure-1-panel-a",
                "Fixed CONUS latitude publication extent declared independent of data.",
                "exact axis endpoint; one decimal place",
            ),
        ),
        "y_max": add_value(
            "fig01.a.axis.y_max_lat",
            bound_value(
                MAP_Y_MAX, "degrees north", "PRE_DECLARED_PUBLICATION_SCALE",
                "source:figure_redraw_spec#Figure-1-panel-a",
                "Fixed CONUS latitude publication extent declared independent of data.",
                "exact axis endpoint; one decimal place",
            ),
        ),
    }
    add_mark("a", "a.axis.map_scale", "map_axis", map_axis_ids)

    nn_median_id = add_value(
        "fig01.a.nn_distance.median_km",
        bound_value(
            round(float(map_data["nn_median"]), 12), "km",
            "PRE_FROZEN_COHORT_GEOMETRY",
            "source:route_a_registry#lat,lon;source:development_environmental_audit#geography.nearest_station_distance_km.median",
            "Great-circle (haversine) nearest-neighbour distance per station; median over 120 stations, validated equal to the frozen audit median.",
            "display: 0.001 km; registry: 0.000000000001",
        ),
    )
    nn_min_id = add_value(
        "fig01.a.nn_distance.minimum_km",
        bound_value(
            round(float(map_data["nn_min"]), 12), "km",
            "PRE_FROZEN_COHORT_GEOMETRY",
            "source:route_a_registry#lat,lon;source:development_environmental_audit#geography.nearest_station_distance_km.minimum",
            "Minimum great-circle nearest-neighbour distance, validated equal to the frozen audit minimum.",
            "display: 0.001 km; registry: 0.000000000001",
        ),
    )
    within_10km_id = add_value(
        "fig01.a.nn_distance.within_10km_count",
        bound_value(
            int(map_data["within_10km"]), "stations",
            "PRE_FROZEN_COHORT_GEOMETRY",
            "source:route_a_registry#lat,lon;source:development_environmental_audit#geography.nearest_station_distance_km.stations_with_neighbor_within_10km",
            "Count stations whose nearest-neighbour distance is below the 10 km threshold, validated equal to the frozen audit count.",
            "integer; no rounding",
        ),
    )
    threshold_10km_id = add_value(
        "fig01.a.nn_distance.threshold_km",
        bound_value(
            NN_THRESHOLD_KM, "km", "PRE_DECLARED_PUBLICATION_SCALE",
            "source:figure_redraw_spec#Figure-1-panel-a",
            "Declared nearest-neighbour distance threshold annotated on the inset histogram.",
            "exact; one decimal place",
        ),
    )
    holdout_mean_id = add_value(
        "fig01.a.whole_region_holdout.mean_nearest_training_gauge_km",
        bound_value(
            WHOLE_REGION_HOLDOUT_MEAN_KM, "km",
            "PRE_STRUCTURAL_GEOMETRY",
            "source:figure_redraw_spec#Figure-1-panel-a;source:option_a_scope#region-transfer-fold-geometry",
            "Stage-13c whole-region-holdout mean nearest-training-gauge distance, bound as a structural geometry quantity and never as a score.",
            "integer km; no rounding",
        ),
    )
    add_mark(
        "a",
        "a.nn_distance_histogram",
        "nearest_neighbor_histogram",
        {
            "median": nn_median_id,
            "minimum": nn_min_id,
            "within_10km_count": within_10km_id,
            "threshold_10km": threshold_10km_id,
            "whole_region_holdout_mean": holdout_mean_id,
        },
    )
    basemap_status_id = add_value(
        "fig01.a.basemap_status",
        bound_value(
            "NO_BASEMAP", "status", "PRE_SCOPE",
            "source:figure_redraw_spec#Figure-1-panel-a",
            "Panel (a) renders a coordinate scatter; no basemap or HUC boundary source is bound.",
            "categorical status; no rounding",
        ),
    )
    add_mark("a", "a.basemap_status", "scope_status", {"status": basemap_status_id})

    for index in range(len(map_data["site"])):
        site = str(map_data["legacy"][index])
        site_no = str(map_data["site"][index])
        add_value(
            f"fig01.a.site.{site}",
            bound_value(
                site_no, "NWIS site number", "PRE_FROZEN_SITE_IDENTITY",
                f"source:route_a_registry#site_no={site_no}",
                "Frozen NWIS site number for the retained station.",
                "identifier; no rounding",
            ),
        )
        add_value(
            f"fig01.a.site.{site}.latitude",
            bound_value(
                round(float(map_data["lat"][index]), 8), "degrees north",
                "PRE_FROZEN_COHORT_GEOMETRY",
                f"source:route_a_registry#lat={site_no}",
                "Station latitude from the frozen registry.",
                "display: 0.0001; registry: 0.00000001",
            ),
        )
        add_value(
            f"fig01.a.site.{site}.longitude",
            bound_value(
                round(float(map_data["lon"][index]), 8), "degrees east",
                "PRE_FROZEN_COHORT_GEOMETRY",
                f"source:route_a_registry#lon={site_no}",
                "Station longitude from the frozen registry.",
                "display: 0.0001; registry: 0.00000001",
            ),
        )
        add_value(
            f"fig01.a.site.{site}.huc2",
            bound_value(
                str(map_data["huc2"][index]), "HUC2 code",
                "PRE_FROZEN_COHORT_GEOMETRY",
                f"source:route_a_registry#huc2={site_no}",
                "Zero-padded two-digit HUC2 region label.",
                "two-character category; no rounding",
            ),
        )
        add_value(
            f"fig01.a.site.{site}.observed_wtemp_day_count",
            bound_value(
                int(map_data["day_counts"][index]), "observed WTEMP days",
                "PRE_TRAINING_ONLY_DESCRIPTIVE",
                f"source:route_a_panel#site_id={site};horizon=none;period=2006-2015",
                "Count finite observed WTEMP rows for the station over the 2006-2015 training interval; drives marker size.",
                "integer; no rounding",
            ),
        )
        add_value(
            f"fig01.a.site.{site}.nearest_neighbor_km",
            bound_value(
                round(float(map_data["nn"][index]), 12), "km",
                "PRE_FROZEN_COHORT_GEOMETRY",
                f"source:route_a_registry#lat,lon={site_no}",
                "Great-circle nearest-neighbour distance from the station to any other retained station.",
                "display: 0.001 km; registry: 0.000000000001",
            ),
        )
        add_mark(
            "a",
            f"a.station.{site}",
            "station_marker",
            {
                "site": f"fig01.a.site.{site}",
                "latitude": f"fig01.a.site.{site}.latitude",
                "longitude": f"fig01.a.site.{site}.longitude",
                "huc2": f"fig01.a.site.{site}.huc2",
                "observed_wtemp_day_count": f"fig01.a.site.{site}.observed_wtemp_day_count",
                "nearest_neighbor_km": f"fig01.a.site.{site}.nearest_neighbor_km",
            },
        )

    # ---- Panel (b) cluster geometry / gate / ladder --------------------------
    target_wtemp_excluded_id = add_value(
        "route_a.input_boundary.target_wtemp_t_plus_h.status",
        bound_value(
            "TARGET_WTEMP_AT_T_PLUS_H_EXCLUDED", "predictor input status",
            "PRE_FROZEN_INFORMATION_BOUNDARY",
            "source:confirmatory_protocol#primary_historical_input_contract.information_cutoff",
            (
                "The exact cutoff is 'issue_date_end; no target-date or post-issue values'; "
                "therefore target WTEMP at t+h is outside predictor input."
            ),
            "categorical status; no rounding",
        ),
    )
    future_weather_excluded_id = add_value(
        "route_a.input_boundary.horizon_specific_future_weather.status",
        bound_value(
            "HORIZON_SPECIFIC_FUTURE_WEATHER_EXCLUDED", "predictor input status",
            "PRE_FROZEN_INFORMATION_BOUNDARY",
            (
                "source:confirmatory_protocol#primary_historical_input_contract."
                "horizon_specific_future_nwp_consumed"
            ),
            "Bind the exact false contract field; no horizon-specific future NWP is consumed.",
            "categorical status; no rounding",
        ),
    )
    add_mark(
        "b",
        "b.excluded_predictor_inputs",
        "excluded_predictor_inputs",
        {
            "target_wtemp_t_plus_h": target_wtemp_excluded_id,
            "horizon_specific_future_weather": future_weather_excluded_id,
        },
    )
    add_mark(
        "b",
        "b.unit_chain",
        "unit_chain_annotation",
        {
            "panel_rows": shared_row_count,
            "stations": shared_station_count,
            "pre_attrition_huc2_groups": shared_huc_count,
        },
    )
    for code, count in zip(huc_labels, huc_counts):
        code_id = add_value(
            f"route_a.registry.huc2_{code}.code",
            bound_value(
                code, "HUC2 code", "PRE_FROZEN_COHORT_GEOMETRY",
                f"source:route_a_registry#huc2={code}",
                "Zero-pad the registry HUC2 label to two digits.",
                "two-character category; no rounding",
            ),
        )
        count_id = add_value(
            f"route_a.registry.huc2_{code}.station_count",
            bound_value(
                int(count), "stations", "PRE_FROZEN_COHORT_GEOMETRY",
                f"source:route_a_registry#huc2={code}",
                "Count frozen registry rows in the pre-attrition HUC2 group.",
                "integer; no rounding",
            ),
        )
        add_mark(
            "b",
            f"b.huc2.{code}",
            "zero_based_count_bar",
            {"x": code_id, "y": count_id, "label": count_id},
        )

    effective_count_id = add_value(
        "route_a.registry.huc2_effective_count",
        bound_value(
            round(effective_count, 6), "effective HUC2 groups",
            "PRE_FROZEN_COHORT_GEOMETRY",
            "source:route_a_registry#huc2",
            "Compute 1 / sum_k (n_k/120)^2 from the 15 bound HUC2 station counts.",
            "display: 0.01; registry: 0.000001",
        ),
    )
    effective_fraction_id = add_value(
        "route_a.registry.huc2_effective_fraction",
        bound_value(
            round(effective_fraction, 6), "fraction of pre-attrition HUC2 groups",
            "PRE_FROZEN_COHORT_GEOMETRY",
            "source:route_a_registry#huc2",
            "Divide inverse-Herfindahl effective count by 15 pre-attrition groups.",
            "display: 0.001; registry: 0.000001",
        ),
    )
    largest_share_id = add_value(
        "route_a.registry.largest_huc2_station_share",
        bound_value(
            round(largest_share, 6), "fraction of stations",
            "PRE_FROZEN_COHORT_GEOMETRY",
            "source:route_a_registry#huc2",
            "Divide the largest bound HUC2 station count (26) by 120 stations.",
            "display: 0.1 percentage point; registry: 0.000001",
        ),
    )
    min_groups_id = add_value(
        "route_a.inference_gate.minimum_reportable_group_count",
        bound_value(
            minimum_reportable_groups, "required reportable groups",
            "PRE_FROZEN_INFERENCE_GATE",
            "source:inference_amendment#inference_scope.small_cluster_rule",
            "Use the frozen minimum reportable-cluster threshold.",
            "integer; no rounding",
        ),
    )
    min_effective_fraction_id = add_value(
        "route_a.inference_gate.minimum_effective_fraction",
        bound_value(
            minimum_effective_fraction, "fraction",
            "PRE_FROZEN_INFERENCE_GATE",
            "source:inference_amendment#inference_scope.small_cluster_rule",
            "Use the frozen minimum effective-cluster fraction threshold.",
            "display: 0.01",
        ),
    )
    max_largest_share_id = add_value(
        "route_a.inference_gate.maximum_largest_group_share",
        bound_value(
            maximum_largest_share, "fraction of stations",
            "PRE_FROZEN_INFERENCE_GATE",
            "source:inference_amendment#inference_scope.small_cluster_rule",
            "Use the frozen strict upper threshold for largest-group share.",
            "display: whole percentage",
        ),
    )
    gate_status_ids = {}
    for gate_name, status, derivation in (
        (
            "reportable_group_count",
            "PASS" if group_count_passed else "FAIL",
            (
                f"At most {len(huc_counts)} pre-attrition groups "
                f"cannot meet {minimum_reportable_groups} required reportable groups."
            ),
        ),
        (
            "effective_fraction",
            "PASS" if effective_fraction_passed else "FAIL",
            (
                f"The registry-derived effective fraction {effective_fraction:.6f} "
                f"is below {minimum_effective_fraction:.2f}."
            ),
        ),
        (
            "largest_group_share",
            "PASS" if largest_share_passed else "FAIL",
            (
                f"The registry-derived largest share {largest_share:.6f} is strictly "
                f"below {maximum_largest_share:.2f}."
            ),
        ),
    ):
        gate_status_ids[gate_name] = add_value(
            f"route_a.inference_gate.{gate_name}.status",
            bound_value(
                status, "gate status", "PRE_FROZEN_INFERENCE_GATE",
                "source:inference_amendment#inference_scope.small_cluster_rule",
                derivation, "categorical status; no rounding",
            ),
        )
    add_mark(
        "b",
        "b.gate.reportable_group_count",
        "gate_check",
        {
            "pre_attrition_upper_bound": shared_huc_count,
            "required_reportable_minimum": min_groups_id,
            "status": gate_status_ids["reportable_group_count"],
        },
    )
    add_mark(
        "b",
        "b.gate.effective_fraction",
        "gate_check",
        {
            "effective_count": effective_count_id,
            "pre_attrition_group_count": shared_huc_count,
            "value": effective_fraction_id,
            "minimum": min_effective_fraction_id,
            "status": gate_status_ids["effective_fraction"],
        },
    )
    add_mark(
        "b",
        "b.gate.largest_group_share",
        "gate_check",
        {
            "value": largest_share_id,
            "maximum_strict": max_largest_share_id,
            "status": gate_status_ids["largest_group_share"],
        },
    )
    failed_count_id = add_value(
        "route_a.inference_gate.failed_component_count",
        bound_value(
            int(sum(not passed for passed in (group_count_passed, effective_fraction_passed, largest_share_passed))),
            "failed gate components", "PRE_FROZEN_INFERENCE_GATE",
            "source:inference_amendment#inference_scope.small_cluster_rule",
            "Count FAIL across the three bound gate checks.",
            "integer; no rounding",
        ),
    )
    scope_status_id = add_value(
        "route_a.scope.fixed_cohort_descriptive_status",
        bound_value(
            gate_failure_verdict, "claim scope status", "PRE_SCOPE_LIMITATION",
            "source:inference_amendment#decision_overlay.gate_failure_verdict",
            "Apply the frozen failed-gate verdict because not all three gate components pass.",
            "categorical status; no rounding",
        ),
    )
    add_mark(
        "b",
        "b.scope_status",
        "scope_status",
        {"status": scope_status_id, "failed_gate_components": failed_count_id},
    )

    for unit, n_clust, eff_count, eff_frac, lshare, passes in CLUSTER_LADDER:
        unit_id = add_value(
            f"fig01.b.ladder.{unit}.unit",
            bound_value(
                unit, "clustering unit", "PRE_FROZEN_COHORT_GEOMETRY",
                "source:option_a_scope#section-1-cluster-geometry",
                "Registered clustering unit in the HUC2/4/6/8 ladder.",
                "categorical unit; no rounding",
            ),
        )
        n_clust_id = add_value(
            f"fig01.b.ladder.{unit}.cluster_count",
            bound_value(
                int(n_clust), "clusters", "PRE_FROZEN_COHORT_GEOMETRY",
                "source:option_a_scope#section-1-cluster-geometry",
                f"Number of {unit} clusters from the gate's own cluster_geometry over the frozen registry.",
                "integer; no rounding",
            ),
        )
        eff_count_id = add_value(
            f"fig01.b.ladder.{unit}.effective_count",
            bound_value(
                round(float(eff_count), 3), "effective clusters",
                "PRE_FROZEN_COHORT_GEOMETRY",
                "source:option_a_scope#section-1-cluster-geometry",
                f"Inverse-Herfindahl effective cluster count at {unit}.",
                "display: 0.001; registry: 0.001",
            ),
        )
        eff_frac_id = add_value(
            f"fig01.b.ladder.{unit}.effective_fraction",
            bound_value(
                round(float(eff_frac), 4), "fraction of clusters",
                "PRE_FROZEN_COHORT_GEOMETRY",
                "source:option_a_scope#section-1-cluster-geometry",
                f"Effective cluster count divided by cluster count at {unit}.",
                "display: 0.001; registry: 0.0001",
            ),
        )
        lshare_id = add_value(
            f"fig01.b.ladder.{unit}.largest_share",
            bound_value(
                round(float(lshare), 4), "fraction of stations",
                "PRE_FROZEN_COHORT_GEOMETRY",
                "source:option_a_scope#section-1-cluster-geometry",
                f"Largest-cluster station share at {unit}.",
                "display: 0.1 percentage point; registry: 0.0001",
            ),
        )
        passes_id = add_value(
            f"fig01.b.ladder.{unit}.passes_arithmetic",
            bound_value(
                bool(passes), "boolean", "PRE_FROZEN_COHORT_GEOMETRY",
                "source:option_a_scope#section-1-cluster-geometry",
                f"Whether {unit} passes all three gate criteria arithmetically.",
                "boolean; no rounding",
            ),
        )
        add_mark(
            "b",
            f"b.ladder.{unit}",
            "cluster_ladder_entry",
            {
                "unit": unit_id,
                "cluster_count": n_clust_id,
                "effective_count": eff_count_id,
                "effective_fraction": eff_frac_id,
                "largest_share": lshare_id,
                "passes_arithmetic": passes_id,
            },
        )
    huc8_note_id = add_value(
        "fig01.b.ladder.huc8_independence_status",
        bound_value(
            "PASSES_ARITHMETIC_NOT_INDEPENDENT", "status", "PRE_SCOPE_LIMITATION",
            "source:option_a_scope#section-1-cluster-geometry;source:figure_redraw_spec#Figure-1-panel-b",
            "HUC8 passes the cluster-gate arithmetic, but adjacent HUC8 units on one river are not independent, so it does not establish reportable clustering.",
            "categorical status; no rounding",
        ),
    )
    add_mark("b", "b.ladder.huc8_note", "cluster_ladder_note", {"status": huc8_note_id})

    spine_specs = (
        ("dated_inputs", "dated inputs <= t", "date-indexed historical-information contract"),
        ("exact_keys", "exact keys", "identical station/date/horizon key contract"),
        ("station_effects", "station effects", "station-balanced aggregation role"),
        ("huc_sensitivity", "HUC sensitivity", "whole-HUC2 sensitivity role"),
        ("claim_gate", "claim gate", "fail-closed claim/QC gate role"),
    )
    for index, (node_id, label, derivation) in enumerate(spine_specs, start=1):
        value_id = add_value(
            f"fig01.b.evidence_spine.{node_id}",
            bound_value(
                label, "workflow stage", "PRE_DESIGN_WORKFLOW",
                "source:figure_redraw_spec#Figure-1-panel-b-evidence-spine",
                derivation, "categorical stage; no rounding",
            ),
        )
        add_mark(
            "b",
            f"b.spine.{index:02d}.{node_id}",
            "evidence_spine_node",
            {"stage": value_id},
        )

    retrospective_status_id = add_value(
        "route_a.scope.not_operational_replay_status",
        bound_value(
            "LATEST_RETROSPECTIVE_NOT_AS_ISSUED", "information scope status",
            "PRE_SCOPE_LIMITATION",
            (
                "source:confirmatory_protocol#primary_historical_input_contract."
                "operational_replay_claim_allowed; "
                "source:claim_registry#permanent_constraints[P03_NOT_OPERATIONAL_REPLAY]"
            ),
            "Bind dated covariates to latest-available retrospective products, not archived as-issued vintages.",
            "categorical status; no rounding",
        ),
    )

    # ---- Panel (c) persistence ------------------------------------------------
    add_mark(
        "c",
        "c.subtitle.training_cohort",
        "text_annotation",
        {"start_date": train_start_id, "end_date": train_end_id, "site_count": shared_station_count},
    )
    panel_c_axis_ids = {
        "y_min": add_value(
            "fig01.c.axis.y_min_c",
            bound_value(
                PERSIST_Y_MIN_C, "degrees C", "PRE_DECLARED_PUBLICATION_SCALE",
                "source:figure_redraw_spec#Figure-1-panel-c",
                "fixed publication scale declared independent of observed extrema",
                "exact axis endpoint; one decimal place",
            ),
        ),
        "y_max": add_value(
            "fig01.c.axis.y_max_c",
            bound_value(
                PERSIST_Y_MAX_C, "degrees C", "PRE_DECLARED_PUBLICATION_SCALE",
                "source:figure_redraw_spec#Figure-1-panel-c",
                "fixed publication scale declared independent of observed extrema",
                "exact axis endpoint; one decimal place",
            ),
        ),
        "y_tick_interval": add_value(
            "fig01.c.axis.y_tick_interval_c",
            bound_value(
                PERSIST_Y_TICK_INTERVAL_C, "degrees C", "PRE_DECLARED_PUBLICATION_SCALE",
                "source:figure_redraw_spec#Figure-1-panel-c",
                "fixed publication scale declared independent of observed extrema",
                "exact tick interval; one decimal place",
            ),
        ),
    }
    add_mark("c", "c.axis.y_scale", "axis_scale", panel_c_axis_ids)

    horizon_value_ids: dict[int, str] = {}
    for horizon in HORIZONS:
        horizon_value_ids[horizon] = add_value(
            f"route_a.horizon_{horizon}_days",
            bound_value(
                horizon, "days", "PRE_FROZEN_HORIZON",
                "source:figure_redraw_spec#Figure-1-panel-c",
                "Use the registered Figure 1 horizon category.",
                "integer day; no rounding",
            ),
        )
        add_value(
            f"fig01.c.h{horizon:02d}.total_pair_count",
            bound_value(
                pair_counts[horizon], "observed exact-day pairs",
                "PRE_TRAINING_ONLY_DESCRIPTIVE",
                f"source:route_a_panel#derived_exact_pairs[horizon_days={horizon}]",
                "Count same-station pairs with DATE_th=DATE_t+h, finite WTEMP at both endpoints, and both dates inside training.",
                "integer; no rounding",
            ),
        )

    for site_id in sorted(mark_data["site_id"].unique()):
        add_value(
            f"fig01.c.site.{site_id}",
            bound_value(
                site_id, "legacy station alias", "PRE_FROZEN_SITE_IDENTITY",
                f"source:route_a_registry#legacy_site_id={site_id}",
                "Use the frozen registry alias after exact equality validation against panel site_id.",
                "identifier; no rounding",
            ),
        )

    for row in mark_data.itertuples(index=False):
        add_value(
            row.y_value_id,
            bound_value(
                round(float(row.station_median_abs_change_c), 6), "degrees C",
                "PRE_TRAINING_ONLY_DESCRIPTIVE",
                f"source:route_a_panel#site_id={row.site_id};horizon_days={row.horizon}",
                "Median of abs(WTEMP[s,t+h]-WTEMP[s,t]) over that station's observed exact-day training pairs.",
                "CSV: 0.000001 °C; plotted coordinate uses the same value",
            ),
        )
        add_value(
            row.n_pairs_value_id,
            bound_value(
                int(row.n_pairs), "observed exact-day pairs",
                "PRE_TRAINING_ONLY_DESCRIPTIVE_DENOMINATOR",
                f"source:route_a_panel#site_id={row.site_id};horizon_days={row.horizon}",
                "Count finite same-station exact-day WTEMP pairs contributing to the station median.",
                "integer; no rounding",
            ),
        )
        add_mark(
            "c",
            row.mark_id,
            "station_point",
            {
                "site": row.site_value_id,
                "x": row.x_value_id,
                "y": row.y_value_id,
                "n_pairs": row.n_pairs_value_id,
            },
        )

    summary_value_ids: dict[int, dict[str, str]] = {}
    for horizon in HORIZONS:
        stats = boxplot_summary(values[horizon])
        ids: dict[str, str] = {}
        for field, number in stats.items():
            value_id = f"fig01.c.h{horizon:02d}.box.{field}"
            rounding = (
                "annotation: 0.01 °C; box median coordinate uses the same value"
                if field == "median_c"
                else "coordinate: unrounded; JSON: 0.000001 °C"
            )
            ids[field] = add_value(
                value_id,
                bound_value(
                    round(float(number), 6), "degrees C",
                    "PRE_TRAINING_ONLY_STATION_DISTRIBUTION_SUMMARY",
                    f"source:fig01_renderer#boxplot_stats[horizon_days={horizon},whis=(5,95)]",
                    (
                        "Apply Matplotlib 3.8 boxplot_stats with whis=(5,95) to the 120 "
                        "bound station-median values; no row weighting."
                    ),
                    rounding,
                ),
            )
        summary_value_ids[horizon] = ids
        add_mark(
            "c",
            f"c.summary.h{horizon:02d}",
            "box_and_median_summary",
            {
                "x": horizon_value_ids[horizon],
                "whisker_low": ids["whisker_low_c"],
                "q25": ids["q25_c"],
                "median": ids["median_c"],
                "q75": ids["q75_c"],
                "whisker_high": ids["whisker_high_c"],
                "median_annotation": ids["median_c"],
            },
        )

    caption_value_ids = [
        train_start_id,
        train_end_id,
        shared_station_count,
        shared_huc_count,
        shared_row_count,
        nn_median_id,
        within_10km_id,
        threshold_10km_id,
        holdout_mean_id,
        basemap_status_id,
        *(horizon_value_ids[h] for h in HORIZONS),
        effective_count_id,
        effective_fraction_id,
        largest_share_id,
        failed_count_id,
        scope_status_id,
        retrospective_status_id,
        target_wtemp_excluded_id,
        future_weather_excluded_id,
        huc8_note_id,
    ]
    render_profile = {
        "dimensions_mm": {"width": WIDTH_MM, "height": HEIGHT_MM},
        "png_dpi": DPI,
        "minimum_body_font_pt": 7.5,
        "minimum_visible_stroke_pt": MIN_VISIBLE_STROKE_PT,
        "panel_order": ["a", "b", "c"],
        "axis_rules": {
            "a": {
                "x_min_lon": MAP_X_MIN,
                "x_max_lon": MAP_X_MAX,
                "y_min_lat": MAP_Y_MIN,
                "y_max_lat": MAP_Y_MAX,
                "encoding": "HUC2 colour + marker shape + filled/open; size by observed-day count",
                "basemap": "NO_BASEMAP",
            },
            "b": {
                "bar_y_origin": 0,
                "bar_y_limit": 30,
                "bar_y_ticks": [0, 10, 20, 30],
                "gate_shared_numeric_axis": False,
                "cluster_ladder": "HUC2/4/6/8: 15/64/75/95 clusters; 0.636/0.507/0.485/0.758",
            },
            "c": {
                "x": "categorical registered horizons [1,3,7] days; jitter is non-empirical",
                "y_min_c_value_id": panel_c_axis_ids["y_min"],
                "y_max_c_value_id": panel_c_axis_ids["y_max"],
                "y_tick_interval_c_value_id": panel_c_axis_ids["y_tick_interval"],
                "y_limits_c": [PERSIST_Y_MIN_C, PERSIST_Y_MAX_C],
                "y_ticks_c": [
                    float(value)
                    for value in np.arange(
                        PERSIST_Y_MIN_C,
                        PERSIST_Y_MAX_C + PERSIST_Y_TICK_INTERVAL_C / 2,
                        PERSIST_Y_TICK_INTERVAL_C,
                    )
                ],
                "box_whiskers_percentile_rule": [5, 95],
                "jitter_seed": 240731,
                "jitter_uniform_range": [-0.16, 0.16],
            },
        },
        "axis_tick_policy": (
            "Decorative/reference axis ticks are fixed render rules and do not receive "
            "scientific value_ids."
        ),
        "bbox_guards_mm": {
            "first_gate_internal_padding": 0.5,
            "excluded_input_internal_padding": 2.0,
            "excluded_input_to_bar_axes_and_labels_separation": 2.0,
        },
        "semantic_palette": {
            "TR_BLUE": BLUE,
            "TR_BLUE_LIGHT": PALE_BLUE,
            "ALLOWED_TEAL": TEAL,
            "ALLOWED_TEAL_LIGHT": PALE_TEAL,
            "WARNING_VERMILION": VERMILION,
            "WARNING_LIGHT": PALE_VERMILION,
            "NEUTRAL_INK": INK,
            "NEUTRAL_GRID": LIGHT,
        },
        "redundant_encodings": {
            "panel_a": "HUC2 colour plus marker shape plus filled/open; marker size by day count",
            "panel_b": "bar hatch; gate symbol plus text; ladder symbol plus text",
            "panel_c": "circle/square/triangle plus hatch",
        },
        "svg_hashsalt": mpl.rcParams["svg.hashsalt"],
        "metadata_date": "2026-08-06T00:00:00Z",
    }

    document = {
        "format": "thermoroute.figure-binder.v2",
        "figure_id": FIGURE_ID,
        "figure_schema_version": FIGURE_SCHEMA_VERSION,
        "source_bindings": source_bindings,
        "panel_order": ["a", "b", "c"],
        "mark_registry": mark_registry,
        "caption_value_ids": caption_value_ids,
        "scope_status_value_id": scope_status_id,
        "render_profile": render_profile,
        "values": value_registry,
        "figures": {
            FIGURE_ID: {
                "figure_id": FIGURE_ID,
                "figure_schema_version": FIGURE_SCHEMA_VERSION,
                "source_binding_ids": [
                    binding["source_id"] for binding in source_bindings
                ],
                "panel_order": ["a", "b", "c"],
                "mark_registry": [entry["mark_id"] for entry in mark_registry],
                "caption_value_ids": caption_value_ids,
                "scope_status_value_id": scope_status_id,
                "render_profile": render_profile,
                "panels": {
                    panel_id: {"marks": marks} for panel_id, marks in panel_marks.items()
                },
            }
        },
        "figure": {
            "id": FIGURE_ID,
            "dimensions_mm": {"width": WIDTH_MM, "height": HEIGHT_MM},
            "narrative": [
                "cohort map",
                "cluster geometry against the claim gate",
                "persistence challenge",
            ],
            "minimum_body_font_pt": 7.5,
            "caption": CAPTION,
        },
        "inputs": {
            "development_panel": {
                "path": "data_usgs/panel_usgs_120v2.parquet",
                "sha256": sha256(panel_path),
            },
            "station_registry": {
                "path": "data_usgs/station_registry_v1.csv",
                "sha256": sha256(registry_path),
            },
            "environmental_audit": {
                "path": "data_usgs/development_environmental_audit_v1.json",
                "sha256": sha256(audit_path),
            },
            "renderer": {
                "path": "paper/agu_submission/figures/render_fig01_preopening_concept.py",
                "sha256": sha256(renderer_path),
            },
        },
        "panel_a": {
            "period": {"start": str(TRAIN_START.date()), "end": str(TRAIN_END.date())},
            "plot_type": "CONUS coordinate scatter coloured by HUC2 with NN-distance inset",
            "station_count": int(len(map_data["site"])),
            "station_count_value_id": shared_station_count,
            "axis_scale_value_ids": map_axis_ids,
            "nearest_neighbor": {
                "median_km_value_id": nn_median_id,
                "minimum_km_value_id": nn_min_id,
                "within_10km_count_value_id": within_10km_id,
                "threshold_km_value_id": threshold_10km_id,
            },
            "whole_region_holdout_mean_km_value_id": holdout_mean_id,
            "basemap_status_value_id": basemap_status_id,
            "marker_size": "retained 2006-2015 observed WTEMP day count",
            "huc2_encoding": "15-way unique (marker, fill) mirroring Figure S1",
        },
        "panel_b": {
            "panel_row_count": panel_row_count,
            "panel_row_count_value_id": shared_row_count,
            "unit_chain": [
                "657,480 daily panel rows",
                "120 stations",
                "15 pre-attrition HUC2 groups",
            ],
            "huc2_station_counts": {
                label: int(count) for label, count in zip(huc_labels, huc_counts)
            },
            "station_count": int(huc_counts.sum()),
            "station_count_value_id": shared_station_count,
            "cluster_count": int(len(huc_counts)),
            "pre_attrition_huc2_group_count_value_id": shared_huc_count,
            "inverse_herfindahl_effective_count": round(effective_count, 6),
            "effective_fraction": round(effective_fraction, 6),
            "largest_cluster_share": round(largest_share, 6),
            "gate": [
                {
                    "criterion": f"reportable_group_count >= {minimum_reportable_groups}",
                    "value": len(huc_counts),
                    "value_role": "pre_attrition_upper_bound",
                    "passed": group_count_passed,
                },
                {
                    "criterion": (
                        "effective_count / pre_attrition_group_count >= "
                        f"{minimum_effective_fraction:.2f}"
                    ),
                    "value": round(effective_fraction, 6),
                    "display": f"{effective_count:.2f} / {len(huc_counts)} = {effective_fraction:.3f}",
                    "passed": effective_fraction_passed,
                },
                {
                    "criterion": f"largest_cluster_share < {maximum_largest_share:.2f}",
                    "value": round(largest_share, 6),
                    "display": f"{largest_share:.1%}",
                    "passed": largest_share_passed,
                },
            ],
            "verdict": gate_failure_verdict,
            "scope_status_value_id": scope_status_id,
            "claim_boundary": "fixed-cohort descriptive only",
            "cluster_ladder": [
                {
                    "unit": unit,
                    "cluster_count": int(n_clust),
                    "effective_count": round(float(eff_count), 3),
                    "effective_fraction": round(float(eff_frac), 4),
                    "largest_share": round(float(lshare), 4),
                    "passes_arithmetic": bool(passes),
                }
                for unit, n_clust, eff_count, eff_frac, lshare, passes in CLUSTER_LADDER
            ],
            "huc8_independence_status_value_id": huc8_note_id,
            "evidence_spine": [
                "dated inputs <= t",
                "exact keys",
                "station effects",
                "HUC sensitivity",
                "claim gate",
            ],
            "excluded_predictor_inputs": {
                "target_wtemp_t_plus_h_value_id": target_wtemp_excluded_id,
                "horizon_specific_future_weather_value_id": future_weather_excluded_id,
            },
        },
        "panel_c": {
            "period": {"start": str(TRAIN_START.date()), "end": str(TRAIN_END.date())},
            "quantity": "station median absolute observed water-temperature change",
            "formula": "m_s,h = median_t |WTEMP_s,t+h - WTEMP_s,t|",
            "axis_scale_value_ids": panel_c_axis_ids,
            "pair_rule": (
                "same station; target DATE equals issue DATE plus h calendar days; "
                "both WTEMP endpoints observed; both dates inside 2006-01-01/2015-12-31"
            ),
            "aggregation": "one median per station and horizon; 120 stations equally weighted",
            "mark_data": {
                "path": "paper/agu_submission/figures/fig01_preopening_concept_data.csv",
                "sha256": sha256(mark_data_path),
                "row_count": 360,
                "primary_key": ["site_id", "horizon"],
                "columns": [
                    "mark_id",
                    "site_id",
                    "site_value_id",
                    "horizon",
                    "x_value_id",
                    "n_pairs",
                    "n_pairs_value_id",
                    "station_median_abs_change_c",
                    "y_value_id",
                ],
            },
            "horizons": {
                str(horizon): {
                    "pair_count": pair_counts[horizon],
                    "pair_count_value_id": f"fig01.c.h{horizon:02d}.total_pair_count",
                    "site_count": int(len(values[horizon])),
                    "site_count_value_id": shared_station_count,
                    "station_median_distribution": summary(values[horizon]),
                    "box_value_ids": summary_value_ids[horizon],
                }
                for horizon in HORIZONS
            },
        },
        "scope": {
            "retrospective_data_qualifier": (
                "Dated covariates are latest-available retrospective products, "
                "not as-issued operational vintages."
            ),
            "temporal_audit_qualifier": (
                "Eligibility uses dated inputs no later than issue date t and exact keys; "
                "this is an auditable temporal constraint, not an absolute no-leakage guarantee."
            ),
            "population_limit": (
                "The availability-enriched 120-station cohort is not a probability sample "
                "and does not support a U.S.-river superpopulation claim."
            ),
            "panel_reassignment_note": (
                "2026-08-06 restructure: panel (a)=station map (new), "
                "panel (b)=cluster geometry/gate extended with the HUC2/4/6/8 ladder "
                "(was old panel c), panel (c)=persistence challenge (was old panel a). "
                "The bounded-correction schematic (old panel b) relocated to Figure S3(c)."
            ),
        },
        "artifacts": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in artifact_paths
        },
        "render": {
            "matplotlib_version": mpl.__version__,
            "dpi_png": DPI,
            "svg_hashsalt": mpl.rcParams["svg.hashsalt"],
            "metadata_date": "2026-08-06T00:00:00Z",
        },
    }
    validate_binder_document(document, mark_data)
    sidecar_path.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    renderer_path = Path(__file__).resolve()
    figure_dir = renderer_path.parent
    repo_root = figure_dir.parents[2]
    panel_path = repo_root / "data_usgs" / "panel_usgs_120v2.parquet"
    registry_path = repo_root / "data_usgs" / "station_registry_v1.csv"
    audit_path = repo_root / "data_usgs" / "development_environmental_audit_v1.json"
    option_a_path = repo_root / "docs" / "OPTION_A_DESCRIPTIVE_BENCHMARK_SCOPE.md"
    config_path = repo_root / "src" / "thermoroute" / "config.py"
    inference_amendment_path = (
        repo_root / "protocols" / "route_a_inference_amendment_v2.json"
    )
    confirmatory_protocol_path = (
        repo_root / "protocols" / "route_a_confirmatory_v1.json"
    )
    require_sha256(panel_path, EXPECTED_PANEL_SHA256, "development panel")
    require_sha256(registry_path, EXPECTED_REGISTRY_SHA256, "station registry")
    require_sha256(audit_path, EXPECTED_AUDIT_SHA256, "environmental audit")
    require_sha256(config_path, EXPECTED_CONFIG_SHA256, "canonical model config")
    require_sha256(
        inference_amendment_path,
        EXPECTED_INFERENCE_AMENDMENT_SHA256,
        "inference amendment",
    )
    require_sha256(
        confirmatory_protocol_path,
        EXPECTED_CONFIRMATORY_PROTOCOL_SHA256,
        "confirmatory protocol",
    )
    _, gate_contract, historical_input_contract = load_frozen_contracts(
        config_path, inference_amendment_path, confirmatory_protocol_path
    )
    values, pair_counts, mark_data, panel_row_count, panel_site_ids = persistence_data(
        panel_path
    )
    observed_counts = observed_wtemp_day_counts(panel_path)
    map_data = station_map(registry_path, audit_path, observed_counts)
    huc_labels, huc_counts, registry_aliases = registry_geometry(registry_path)
    if panel_site_ids != registry_aliases:
        raise RuntimeError(
            "Frozen panel site_id values do not exactly equal registry legacy_site_id aliases"
        )
    mark_data_path = figure_dir / "fig01_preopening_concept_data.csv"
    mark_data.to_csv(
        mark_data_path,
        index=False,
        float_format="%.6f",
        lineterminator="\n",
    )
    figure = build_figure(
        map_data,
        values,
        huc_labels,
        huc_counts,
        gate_contract,
        historical_input_contract,
    )

    stem = figure_dir / "fig01_preopening_concept"
    svg_path = stem.with_suffix(".svg")
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    fixed_time = datetime(2026, 8, 6, tzinfo=timezone.utc)
    figure.savefig(
        svg_path,
        format="svg",
        metadata={
            "Title": "Station map, cluster geometry against the claim gate, and the persistence challenge",
            "Creator": "ThermoRoute deterministic Figure 1 renderer",
            "Description": CAPTION,
            "Date": "2026-08-06T00:00:00Z",
        },
    )
    make_svg_accessible(svg_path)
    validate_svg_visible_strokes(svg_path)
    figure.savefig(
        pdf_path,
        format="pdf",
        metadata={
            "Title": "Station map, cluster geometry against the claim gate, and the persistence challenge",
            "Author": "ThermoRoute authors",
            "Subject": CAPTION,
            "Creator": "ThermoRoute deterministic Figure 1 renderer",
            "Producer": "Matplotlib 3.8",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        },
    )
    figure.savefig(
        png_path,
        format="png",
        dpi=DPI,
        metadata={
            "Title": "Station map, cluster geometry against the claim gate, and the persistence challenge",
            "Author": "ThermoRoute authors",
            "Description": CAPTION,
            "Creation Time": "2026-08-06T00:00:00Z",
            "Software": "Matplotlib 3.8",
        },
    )
    plt.close(figure)

    write_sidecar(
        stem.with_suffix(".json"),
        repo_root,
        renderer_path,
        panel_path,
        registry_path,
        audit_path,
        option_a_path,
        mark_data_path,
        panel_row_count,
        gate_contract,
        historical_input_contract,
        [svg_path, pdf_path, png_path, mark_data_path],
        values,
        pair_counts,
        mark_data,
        huc_labels,
        huc_counts,
        map_data,
        observed_counts,
    )


if __name__ == "__main__":
    main()

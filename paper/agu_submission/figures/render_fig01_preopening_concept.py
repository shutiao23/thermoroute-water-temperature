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
"""Render the three-panel ThermoRoute opening figure (relaid out 2026-08-07).

Panel assignment under the benchmark restructure:
  (a) station map -- 120 retained gauges on a CONUS coordinate scatter, coloured
      by HUC2 with redundant marker encoding and sized by retained 2006--2015
      observed WTEMP day count, beside the nearest-neighbour distance histogram.
  (b) cluster geometry against the claim gate -- HUC2 counts, three frozen gate
      gauges, and the HUC2/4/6/8 ladder.
  (c) the persistence challenge -- station median |dT| by horizon from
      observed exact-day pairs in the 2006--2015 training interval.

The bounded-correction tanh schematic that previously occupied panel (b) has been
relocated to Figure S3(c).  The renderer keeps every frozen-input integrity check
and reuses the cohort geometry, gate, and persistence derivations of the prior
render.

Layout (2026-08-07 redraw).  The previous revision positioned panels, insets,
gate boxes, ladder cells and flow nodes in figure/axes fractions by hand, and at
least six pairs of elements collided in the rendered PDF: the (a) heading under
the inset axis label, the inset over the map, four inset annotations over each
other, the (c) heading cut by the (a) axes, the (c) subtitle clipped at the page
edge, the (b) hatched band over the bar labels and the y-axis label, the ladder
under the scope verdict, and the HUC2 legend spilling outside the figure.  Every
one of those was a consequence of guessing coordinates.  Nothing is positioned in
figure fractions any more: one figure-level ``GridSpec`` under constrained layout
owns the geometry, each element claims a cell, and ``figstyle.check_overlaps``
fails the render if any two text artists still touch.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# paper/figstyle.py is the single style contract for every submission figure:
# Nimbus Sans (metric Helvetica), 8 pt body / 7.5 pt ticks / 9 pt bold panel
# labels, Wong (2011) colour-blind-safe palette, pdf.fonttype 42, 600 dpi,
# constrained layout, and the text-collision guard.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import figstyle  # noqa: E402  (path is set immediately above)


WIDTH_MM = figstyle.FULL_MM      # the AGU class \textwidth, from the style module
HEIGHT_MM = 186.0                # < AGU's 228 mm ceiling; see LAYOUT_MM below
DPI = 600
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
# The one gate criterion the ladder can move, drawn once as a threshold line and
# checked against the frozen contract before it is used.
MIN_EFFECTIVE_FRACTION_DISPLAY = 0.75
# Where each gate gauge puts its own threshold, as a fraction of its own axis.
# The strip to the right of it is reserved for the PASS/FAIL word.
GATE_THRESHOLD_AXIS_FRACTION = 0.62
EVIDENCE_SPINE_LABELS = (
    "dated inputs\n≤ t",
    "exact\nkeys",
    "station\neffects",
    "HUC\nsensitivity",
    "claim\ngate",
)

# Wong (2011) hues via the shared style module, plus neutral ink.  Patterns and
# words duplicate colour everywhere colour carries meaning.
BLUE = figstyle.WONG["blue"]
VERMILION = figstyle.WONG["vermillion"]
TEAL = figstyle.WONG["green"]
INK = "#202020"
MID = figstyle.MUTED
LIGHT = figstyle.GRID
PALE_BLUE = "#DCEAF4"
PALE_VERMILION = "#F9E3D6"
PALE_TEAL = "#DCEFEA"
PALE_GREY = "#EDEDED"
WHITE = "#FFFFFF"
# Full Okabe--Ito/Wong set for the 15-way HUC2 map encoding (mirrors Figure S1).
OI_HUC_COLORS = (
    figstyle.WONG["blue"],
    figstyle.WONG["orange"],
    figstyle.WONG["green"],
    figstyle.WONG["vermillion"],
    figstyle.WONG["purple"],
)
OI_HUC_MARKERS = ("o", "s", "^", "D", "v", "<", ">", "p")

# Nimbus Sans has no U+2713 CHECK MARK and no U+2095 SUBSCRIPT SMALL H; both
# render as tofu once the figure leaves DejaVu.  Verdicts are therefore words
# (which are also the non-colour channel), and the horizon is on the x axis
# rather than in a subscript.
PASS_WORD = "PASS"
FAIL_WORD = "FAIL"

# Panel block heights in mm.  These are GridSpec height ratios, not positions:
# constrained layout distributes the padding, so the numbers only set the
# relative vertical budget of the three panel blocks.
LAYOUT_MM = {"a": 47.0, "b": 86.0, "c": 32.0}
# Height of each panel-heading row, which lives in the outer grid.
HEAD_MM = 6.5
# Declared nearest-neighbour histogram extent (panel a), fixed independently of
# the observed maximum and validated against it at render time.
NN_HIST_MAX_KM = 400.0
NN_HIST_BIN_KM = 20.0

CAPTION = (
    "Figure 1. Cohort, geometry against the claim gate, and the persistence "
    "challenge. (a) The 120 retained gauges on a CONUS coordinate scatter (no "
    "basemap), coloured by HUC2 region with redundant marker encoding and sized "
    "by retained 2006--2015 observed WTEMP day count; beside it, a zero-based "
    "histogram of nearest-neighbour distance between retained stations, marking "
    "the 10 km distance (19 stations) and the 289 km whole-region-holdout mean "
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
    """Apply the shared submission style, then this figure's render guards."""
    figstyle.use()
    mpl.rcParams.update(
        {
            # The SVG guard below refuses any visible stroke under 0.6 pt, so
            # the grid and hatch widths are lifted to that floor.
            "grid.linewidth": MIN_VISIBLE_STROKE_PT,
            "hatch.linewidth": MIN_VISIBLE_STROKE_PT,
            "svg.hashsalt": "thermoroute-figure-1-v4",
            "savefig.dpi": DPI,
            "savefig.facecolor": WHITE,
            "figure.facecolor": WHITE,
        }
    )
    resolved = figstyle.resolved_font()
    if resolved == "DejaVu Sans":
        raise RuntimeError(
            "Figure 1 resolved to DejaVu Sans: no Helvetica-metric sans-serif is "
            "installed, and AGU line art must not ship the silent fallback"
        )


def blank_axes(ax: mpl.axes.Axes) -> mpl.axes.Axes:
    """Turn a grid cell into a bare drawing surface with unit coordinates.

    ``set_axis_off`` leaves invisible tick labels behind, and those still carry
    text and a window extent, so the collision guard reports phantom hits
    between two annotation cells.  Emptying the locators removes them.
    """
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.patch.set_visible(False)
    return ax


def free_text(ax: mpl.axes.Axes, x: float, y: float, text: str, **kwargs) -> mpl.text.Text:
    """Annotation text that draws but does not push the layout around.

    Constrained layout sizes every cell from its artists' tight bounding box,
    so a multi-line note in an annotation cell would steal width and height
    from the plot beside it -- panel (c) collapsed to 8 mm that way.  These
    strings are laid out inside a cell that is already the right size, so they
    are excluded from the margin calculation and policed by the collision
    guard instead.
    """
    artist = ax.text(x, y, text, **kwargs)
    artist.set_in_layout(False)
    return artist


def panel_heading(ax: mpl.axes.Axes, letter: str, title: str, subtitle: str) -> None:
    """Panel label, title and subtitle in a cell of their own.

    The heading used to be text at a guessed axes fraction inside the panel, so
    it sat on whatever the panel drew there.  It now owns a grid row, and the
    rule marks where the panel block starts.
    """
    blank_axes(ax)
    ax.axhline(0.80, color=LIGHT, linewidth=MIN_VISIBLE_STROKE_PT, zorder=0)
    free_text(
        ax, 0.0, 0.12, f"({letter})  {title}",
        transform=ax.transAxes, fontsize=9.0, fontweight="bold",
        ha="left", va="baseline", color=INK,
    )
    free_text(
        ax, 1.0, 0.12, subtitle,
        transform=ax.transAxes, fontsize=7.5,
        ha="right", va="baseline", color=MID,
    )


def sub_title(ax: mpl.axes.Axes, text: str) -> None:
    """A within-panel axes title, one step below the panel heading."""
    ax.set_title(text, loc="left", fontsize=8.0, fontweight="normal", color=INK, pad=3.0)


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
    axes: dict[str, mpl.axes.Axes],
    map_data: dict[str, object],
) -> None:
    """(a) Station map, nearest-neighbour distances, and the HUC2 key.

    Three cells: the coordinate scatter, the distance histogram beside it (it
    was an inset lying across the map), and a legend strip beneath both (the
    15-entry HUC2 key used to overflow the left figure margin).
    """
    panel_heading(
        axes["head"],
        "a",
        "Station map",
        "120 retained gauges • marker size: 2006–2015 observed days",
    )

    ax = axes["map"]
    huc_codes = [f"{int(code):02d}" for code in sorted(EXPECTED_HUC_COUNTS, key=int)]
    encoding = _huc_encoding(huc_codes)
    enc_by_code = {e["huc2"]: e for e in encoding}
    day_counts = map_data["day_counts"]
    dc_min = float(day_counts.min())
    dc_max = float(day_counts.max())

    def size_for(count: int) -> float:
        # Observed-day count -> marker area in [8, 38] pt^2.
        if dc_max <= dc_min:
            return 23.0
        return 8.0 + 30.0 * (count - dc_min) / (dc_max - dc_min)

    legend_handles: list[Line2D] = []
    for code in huc_codes:
        e = enc_by_code[code]
        mask = map_data["huc2"] == code
        filled = e["fill"] == "filled"
        face = e["color"] if filled else WHITE
        # Open markers carry their hue on the edge; drawing them white-on-black
        # threw colour away for seven of the fifteen regions.
        edge = INK if filled else e["color"]
        sizes = np.array([size_for(int(c)) for c in day_counts[mask]])
        ax.scatter(
            map_data["lon"][mask],
            map_data["lat"][mask],
            s=sizes,
            marker=e["marker"],
            facecolor=face,
            edgecolor=edge,
            linewidth=0.7 if filled else 0.8,
            alpha=0.92,
            zorder=3,
        )
        legend_handles.append(
            Line2D(
                [0], [0], marker=e["marker"], color="none",
                markerfacecolor=face, markeredgecolor=edge,
                markeredgewidth=0.7 if filled else 0.8, markersize=3.6, label=code,
            )
        )
    ax.set_xlim(MAP_X_MIN, MAP_X_MAX)
    ax.set_ylim(MAP_Y_MIN, MAP_Y_MAX)
    # Degree-suffixed tick labels carry the axis meaning, so the map needs no
    # axis labels and gives the row back to the data area.
    ax.set_xticks(
        [-120, -110, -100, -90, -80, -70],
        ["120°W", "110°W", "100°W", "90°W", "80°W", "70°W"],
    )
    ax.set_yticks([25, 30, 35, 40, 45, 50], ["25°N", "30°N", "35°N", "40°N", "45°N", "50°N"])
    ax.grid(True, color=LIGHT, linewidth=MIN_VISIBLE_STROKE_PT, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=MIN_VISIBLE_STROKE_PT, color=MID)
    free_text(
        ax, 0.015, 0.03, "coordinate scatter • no basemap",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, color=MID,
        bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 1.0}, zorder=4,
    )

    # Nearest-neighbour distance: its own cell, with the two marked distances
    # as legend entries rather than four text boxes stacked on the bars.
    nn = np.asarray(map_data["nn"], dtype=float)
    if float(nn.max()) > NN_HIST_MAX_KM or WHOLE_REGION_HOLDOUT_MEAN_KM > NN_HIST_MAX_KM:
        raise RuntimeError(
            "Panel-a nearest-neighbour histogram extent "
            f"{NN_HIST_MAX_KM:.0f} km does not cover the bound distances"
        )
    hist = axes["nn"]
    bins = np.arange(0.0, NN_HIST_MAX_KM + NN_HIST_BIN_KM / 2.0, NN_HIST_BIN_KM)
    hist.hist(nn, bins=bins, color=PALE_BLUE, edgecolor=BLUE, linewidth=0.7, zorder=2)
    hist.axvline(
        NN_THRESHOLD_KM, color=VERMILION, linestyle=(0, (3, 2)), linewidth=0.9,
        zorder=3,
        label=f"{NN_THRESHOLD_KM:.0f} km · {int(map_data['within_10km'])} stations",
    )
    hist.axvline(
        WHOLE_REGION_HOLDOUT_MEAN_KM, color=TEAL, linestyle=(0, (1, 1.6)), linewidth=0.9,
        zorder=3,
        label=f"{WHOLE_REGION_HOLDOUT_MEAN_KM:.0f} km · whole-region\nholdout mean",
    )
    # Fix the count axis to a round multiple of ten.  An autoscaled axis leaves
    # a locator tick outside the view; the tick is never drawn but it still
    # carries a label with a window extent, and the collision guard sees it.
    tallest_bin = int(np.histogram(nn, bins=bins)[0].max())
    hist_top = int(np.ceil(tallest_bin / 10.0) * 10)
    hist.set_ylim(0, hist_top)
    hist.set_yticks(list(range(0, hist_top + 1, 10)))
    hist.set_xlim(0.0, NN_HIST_MAX_KM)
    hist.set_xticks([0, 100, 200, 300])
    hist.set_xlabel("Nearest-neighbour distance (km)", labelpad=1.5)
    hist.set_ylabel("Stations", labelpad=1.5)
    hist.grid(axis="y", color=LIGHT, linewidth=MIN_VISIBLE_STROKE_PT)
    hist.set_axisbelow(True)
    hist.tick_params(length=2.5, width=MIN_VISIBLE_STROKE_PT, color=MID)
    hist.legend(
        loc="upper right", fontsize=7.5, frameon=False, handlelength=1.3,
        handletextpad=0.4, labelspacing=0.45, borderaxespad=0.2,
    )

    key = blank_axes(axes["legend"])
    key_legend = key.legend(
        handles=legend_handles, ncol=len(legend_handles), loc="center",
        title="HUC2 region", title_fontsize=7.5, fontsize=7.5, frameon=False,
        handlelength=0.9, handletextpad=0.35, columnspacing=0.7,
        borderpad=0.0, borderaxespad=0.0,
    )
    key_legend.set_in_layout(False)


def gate_gauge(
    ax: mpl.axes.Axes,
    criterion: str,
    value: float,
    threshold: float,
    value_text: str,
    threshold_text: str,
    passed: bool,
    guard_id: str | None = None,
) -> None:
    """One frozen balance check as a single-bar gauge on its own scale.

    The three checks compare a count, a fraction and a share; the spec forbids
    putting them on one shared numeric axis, so each gauge keeps its own limits
    and shows only two reference ticks -- zero and its own threshold.  Each
    scale is set so the threshold falls at the same fraction of every gauge,
    which is what makes three incommensurable checks readable side by side
    without pretending they share an axis: bar past the dashed line = pass.
    """
    color = TEAL if passed else VERMILION
    fill = PALE_TEAL if passed else PALE_VERMILION
    span = max(value, threshold)
    bar_y = -0.36
    bar = ax.barh(
        [bar_y], [value], height=0.80, facecolor=fill, edgecolor=color,
        linewidth=0.8, hatch=None if passed else "//", zorder=2,
    )[0]
    if guard_id is not None:
        bar.set_gid(f"guard.{guard_id}.bar")
    # The threshold line stops below its own label rather than striking through it.
    ax.axvline(
        threshold, ymin=0.02, ymax=0.58, color=INK, linewidth=0.8,
        linestyle=(0, (3, 2)), zorder=3,
    )
    ax.set_xlim(0.0, span / GATE_THRESHOLD_AXIS_FRACTION)
    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks([bar_y], [criterion])
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0.0, pad=2.0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # A gauge, not a plot: the requirement is printed on its own threshold line
    # rather than as a tick, so the three incommensurable checks carry no axis
    # to be mistaken for a shared one.
    free_text(
        ax, threshold, 0.16, threshold_text, ha="center", va="bottom",
        fontsize=7.5, color=MID, zorder=4,
    )
    # The measured value rides inside its own bar; the verdict word is the
    # non-colour channel and sits in the strip reserved past the threshold.
    free_text(
        ax, value - 0.07 * span, bar_y, value_text, ha="right", va="center",
        fontsize=7.5, color=INK, zorder=4,
        bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 0.6, "alpha": 0.85},
    )
    verdict = free_text(
        ax, 0.995, bar_y, PASS_WORD if passed else FAIL_WORD,
        transform=ax.get_yaxis_transform(), ha="right", va="center",
        fontsize=7.5, fontweight="bold", color=color,
    )
    if guard_id is not None:
        verdict.set_gid(f"guard.{guard_id}.text")


def draw_excluded_inputs(
    ax: mpl.axes.Axes, historical_input_contract: dict[str, object]
) -> None:
    """The information-boundary statement as a full-width band of its own."""
    if (
        historical_input_contract["information_cutoff"]
        != "issue_date_end; no target-date or post-issue values"
        or historical_input_contract["horizon_specific_future_nwp_consumed"] is not False
    ):
        raise RuntimeError("Excluded-input annotation lacks its frozen contract semantics")
    blank_axes(ax)
    band = Rectangle(
        (0.0, 0.06), 1.0, 0.88, transform=ax.transAxes,
        facecolor=PALE_VERMILION, edgecolor=VERMILION, linewidth=0.8, zorder=2,
    )
    band.set_gid("guard.excluded_inputs.box")
    ax.add_patch(band)
    # A hatched end-stripe keeps the warning legible without laying a hatch
    # under the text, which is what made the old full-width hatched box unreadable.
    stripe = Rectangle(
        (0.0, 0.06), 0.022, 0.88, transform=ax.transAxes,
        facecolor=PALE_VERMILION, edgecolor=VERMILION, linewidth=0.8,
        hatch="///", zorder=3,
    )
    ax.add_patch(stripe)
    status = free_text(
        ax, 0.038, 0.50, "EXCLUDED FROM PREDICTOR INPUT",
        transform=ax.transAxes, fontsize=7.5, fontweight="bold", color=VERMILION,
        ha="left", va="center", zorder=4,
    )
    status.set_gid("guard.excluded_inputs.status")
    detail = free_text(
        ax, 0.978, 0.50,
        "target WTEMP at t+h  •  horizon-specific future weather",
        transform=ax.transAxes, fontsize=7.5, color=VERMILION,
        ha="right", va="center", zorder=4,
    )
    detail.set_gid("guard.excluded_inputs.text")


def draw_huc_bars(ax: mpl.axes.Axes, huc_labels: list[str], counts: np.ndarray) -> None:
    """Zero-based station counts for the 15 pre-attrition HUC2 groups."""
    x = np.arange(len(huc_labels))
    largest = int(np.argmax(counts))
    facecolors = [PALE_BLUE] * len(huc_labels)
    edgecolors = [BLUE] * len(huc_labels)
    hatches = ["//"] * len(huc_labels)
    facecolors[largest] = "#C6DFEC"
    edgecolors[largest] = "#004F7C"
    hatches[largest] = "xx"
    bars = ax.bar(x, counts, width=0.72, color=facecolors, edgecolor=edgecolors, linewidth=0.8)
    for bar, hatch, count, huc_label in zip(bars, hatches, counts, huc_labels):
        bar.set_hatch(hatch)
        label = free_text(
            ax, bar.get_x() + bar.get_width() / 2.0, float(count) + 0.8, str(int(count)),
            ha="center", va="bottom", fontsize=7.5, color=INK,
        )
        label.set_gid(f"guard.bar_label.{huc_label}")
    ax.set_ylim(0, 30)
    ax.set_xlim(-0.75, len(huc_labels) - 0.25)
    ax.set_xticks(x, huc_labels)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_ylabel("Stations", labelpad=1.5)
    ax.grid(axis="y", color=LIGHT, linewidth=MIN_VISIBLE_STROKE_PT)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.3, width=MIN_VISIBLE_STROKE_PT, color=MID, pad=1.5)
    sub_title(ax, "Stations per HUC2 region")


def draw_cluster_ladder(ax: mpl.axes.Axes) -> None:
    """HUC2/4/6/8 ladder against the one gate criterion it can move.

    Four stacked text cells became four bars on a real effective-fraction axis
    with the frozen 0.75 threshold drawn once.  Cluster counts sit inside their
    own bar, so no label can drift onto a neighbour.
    """
    rows = list(CLUSTER_LADDER)
    positions = np.arange(len(rows))[::-1]
    tick_labels: list[str] = []
    for position, (unit, n_clusters, _eff_count, eff_fraction, _share, passes) in zip(
        positions, rows
    ):
        ax.barh(
            [float(position)], [float(eff_fraction)], height=0.62,
            facecolor=PALE_TEAL if passes else PALE_GREY,
            edgecolor=TEAL if passes else MID,
            hatch=None if passes else "//",
            linewidth=0.8, zorder=2,
        )
        free_text(
            ax, 0.018, float(position), f"{int(n_clusters)} clusters",
            ha="left", va="center", fontsize=7.5, color=INK, zorder=4,
            bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 0.8, "alpha": 0.82},
        )
        free_text(
            ax, float(eff_fraction) + 0.018, float(position), f"{eff_fraction:.3f}",
            ha="left", va="center", fontsize=7.5, fontweight="bold",
            color=TEAL if passes else MID, zorder=4,
        )
        tick_labels.append(f"{unit} *" if unit == "HUC8" else unit)
    ax.axvline(
        MIN_EFFECTIVE_FRACTION_DISPLAY, color=INK, linewidth=0.8,
        linestyle=(0, (3, 2)), zorder=3,
    )
    ax.set_yticks(positions, tick_labels)
    ax.set_ylim(-0.65, len(rows) - 0.35)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.50", "0.75", "1.0"])
    ax.set_xlabel("Effective cluster fraction (dashed: 0.75 gate)", labelpad=1.5)
    ax.grid(axis="x", color=LIGHT, linewidth=MIN_VISIBLE_STROKE_PT)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.3, width=MIN_VISIBLE_STROKE_PT, color=MID, pad=1.5)
    sub_title(ax, "Cluster ladder: HUC2 → HUC8")


def draw_ladder_note(
    ax: mpl.axes.Axes, failed_component_count: int, gate_failure_verdict: str
) -> None:
    """The HUC8 refusal and the scope verdict, in the cell beside the ladder."""
    blank_axes(ax)
    free_text(
        ax, 0.0, 1.0,
        "* HUC8 clears all three checks,\n"
        "but adjacent HUC8 units on one\n"
        "river are not independent.",
        transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
        color=INK, linespacing=1.18,
    )
    box = Rectangle(
        (0.0, 0.0), 1.0, 0.53, transform=ax.transAxes,
        facecolor=PALE_VERMILION, edgecolor=VERMILION, linewidth=0.8, zorder=2,
    )
    box.set_gid("guard.scope_status.box")
    ax.add_patch(box)
    verdict = free_text(
        ax, 0.5, 0.265,
        f"{failed_component_count} of 3 checks fail:\n"
        "FIXED-COHORT\nDESCRIPTIVE ONLY",
        transform=ax.transAxes, ha="center", va="center", fontsize=7.5,
        fontweight="bold", color=VERMILION, linespacing=1.18, zorder=3,
    )
    verdict.set_gid("guard.scope_status.text")


def draw_evidence_spine(ax: mpl.axes.Axes, node_labels: tuple[str, ...]) -> None:
    """The left-to-right evidence spine, one node per unit of a 0..n axis."""
    blank_axes(ax)
    ax.set_xlim(0.0, float(len(node_labels)))
    for index, label in enumerate(node_labels):
        ax.add_patch(
            Rectangle(
                (index + 0.04, 0.10), 0.74, 0.80,
                facecolor=PALE_TEAL, edgecolor=TEAL, linewidth=0.8, zorder=2,
            )
        )
        free_text(
            ax, index + 0.41, 0.50, label, ha="center", va="center", fontsize=7.5,
            color=INK, linespacing=1.15, zorder=3,
        )
        if index < len(node_labels) - 1:
            # An empty annotation: a drawn arrow, not a glyph the typeface may
            # not carry, and nothing for the collision guard to trip over.
            ax.annotate(
                "", xy=(index + 0.97, 0.50), xytext=(index + 0.82, 0.50),
                arrowprops={
                    "arrowstyle": "-|>", "color": TEAL, "linewidth": 0.8,
                    "shrinkA": 0.0, "shrinkB": 0.0, "mutation_scale": 5.0,
                },
            )


def draw_panel_b(
    axes: dict[str, mpl.axes.Axes],
    huc_labels: list[str],
    counts: np.ndarray,
    gate_contract: dict[str, object],
    historical_input_contract: dict[str, object],
) -> None:
    """(b) Cohort geometry against the frozen claim gate."""
    panel_heading(
        axes["head"],
        "b",
        "Cluster geometry vs the claim gate",
        "657,480 rows → 120 stations → 15 HUC2 groups",
    )
    draw_excluded_inputs(axes["excluded"], historical_input_contract)
    draw_huc_bars(axes["bars"], huc_labels, counts)

    effective_count = float(1.0 / np.sum((counts / counts.sum()) ** 2))
    effective_fraction = effective_count / len(counts)
    largest_share = float(np.max(counts) / counts.sum())
    required_groups = int(gate_contract["minimum_reportable_groups"])
    minimum_effective = float(gate_contract["minimum_effective_fraction"])
    maximum_share = float(gate_contract["maximum_largest_group_share"])
    if abs(minimum_effective - MIN_EFFECTIVE_FRACTION_DISPLAY) > 1e-12:
        raise RuntimeError(
            "Ladder threshold line and the frozen effective-fraction gate disagree"
        )
    # The HUC2 row of the ladder and the gate computed from the bound counts are
    # the same geometry; refuse to draw them side by side unless they agree.
    ladder_huc2 = {row[0]: row for row in CLUSTER_LADDER}["HUC2"]
    if (
        int(ladder_huc2[1]) != len(counts)
        or abs(float(ladder_huc2[3]) - effective_fraction) > 5e-4
        or abs(float(ladder_huc2[4]) - largest_share) > 5e-4
    ):
        raise RuntimeError(
            "Panel-b ladder HUC2 row disagrees with the registry-derived gate values: "
            f"ladder={ladder_huc2!r}, registry=({len(counts)}, {effective_fraction:.6f}, "
            f"{largest_share:.6f})"
        )

    sub_title(axes["gate"][0], "Frozen claim gate")
    gate_gauge(
        axes["gate"][0],
        "pre-attrition\ngroups",
        float(len(counts)),
        float(required_groups),
        f"≤{len(counts)}",
        f"{required_groups} required",
        passed=len(counts) >= required_groups,
        guard_id="gate_groups",
    )
    gate_gauge(
        axes["gate"][1],
        f"eff. fraction\n({effective_count:.2f} of {len(counts)})",
        effective_fraction,
        minimum_effective,
        f"{effective_fraction:.3f}",
        f"{minimum_effective:.2f} required",
        passed=effective_fraction >= minimum_effective,
        guard_id="gate_effective_fraction",
    )
    gate_gauge(
        axes["gate"][2],
        "largest\ngroup share",
        largest_share,
        maximum_share,
        f"{largest_share:.1%}",
        f"{maximum_share:.0%} limit",
        passed=largest_share < maximum_share,
        guard_id="gate_largest_share",
    )
    draw_cluster_ladder(axes["ladder"])
    failed_component_count = int(
        sum(
            not passed
            for passed in (
                len(counts) >= required_groups,
                effective_fraction >= minimum_effective,
                largest_share < maximum_share,
            )
        )
    )
    draw_ladder_note(
        axes["note"], failed_component_count, str(gate_contract["gate_failure_verdict"])
    )
    draw_evidence_spine(axes["spine"], EVIDENCE_SPINE_LABELS)


def draw_panel_c(
    axes: dict[str, mpl.axes.Axes], values: dict[int, np.ndarray]
) -> None:
    """(c) The persistence challenge, with its read-me note in its own cell."""
    panel_heading(
        axes["head"],
        "c",
        "Persistence challenge",
        "2006–2015 exact-day pairs • 120 stations • motivation quantity",
    )
    ax = axes["plot"]
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
        widths=0.46,
        whis=(5, 95),
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "linewidth": 1.4},
        whiskerprops={"color": MID, "linewidth": 0.9},
        capprops={"color": MID, "linewidth": 0.9},
    )
    rng = np.random.default_rng(240731)
    for index, (vals, color, fill, marker, hatch) in enumerate(
        zip(distributions, colors, pale, markers, hatches), start=1
    ):
        patch = boxes["boxes"][index - 1]
        patch.set(facecolor=fill, edgecolor=color, linewidth=1.0, hatch=hatch)
        jitter = rng.uniform(-0.16, 0.16, size=len(vals))
        ax.scatter(
            index + jitter,
            vals,
            s=7,
            marker=marker,
            facecolor=WHITE,
            edgecolor=color,
            linewidth=MIN_VISIBLE_STROKE_PT,
            alpha=0.62,
            zorder=2,
        )
        free_text(
            ax,
            index,
            PERSIST_Y_MAX_C - 0.18,
            f"med. {float(np.median(vals)):.2f}",
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
    # Nimbus Sans carries no subscript h, so the horizon lives on the x axis
    # instead of inside the symbol.
    ax.set_ylabel("Station median\n|ΔT|  (°C)", labelpad=1.5, linespacing=1.2)
    ax.grid(axis="y", color=LIGHT, linewidth=MIN_VISIBLE_STROKE_PT)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=MIN_VISIBLE_STROKE_PT, color=MID)

    note = blank_axes(axes["note"])
    free_text(
        note, 0.0, 1.0,
        "Points: one median per station and\n"
        "horizon, of |WTEMP(t+h) − WTEMP(t)|\n"
        "over its observed exact-day pairs.\n"
        "Box: quartiles; line: median.\n"
        "Whiskers: 5th–95th percentile of\n"
        "the 120 station medians.",
        transform=note.transAxes, ha="left", va="top", fontsize=7.5,
        color=INK, linespacing=1.28,
    )


def build_layout(fig: plt.Figure) -> dict[str, dict[str, object]]:
    """One figure-level GridSpec; every element below gets its own cell.

    The three panel headings are rows of the *outer* grid rather than of their
    own block, so all three panel labels share one left edge whatever the
    blocks below them reserve for tick labels and axis titles.
    """
    outer = fig.add_gridspec(
        6, 1,
        height_ratios=[
            HEAD_MM, LAYOUT_MM["a"], HEAD_MM, LAYOUT_MM["b"], HEAD_MM, LAYOUT_MM["c"],
        ],
        hspace=0.03,
    )

    gs_a = outer[1].subgridspec(
        2, 2, height_ratios=[40.0, 7.0], width_ratios=[1.22, 1.0],
        hspace=0.035, wspace=0.07,
    )
    panel_a = {
        "head": fig.add_subplot(outer[0]),
        "map": fig.add_subplot(gs_a[0, 0]),
        "nn": fig.add_subplot(gs_a[0, 1]),
        "legend": fig.add_subplot(gs_a[1, :]),
    }

    gs_b = outer[3].subgridspec(
        4, 2, height_ratios=[7.5, 36.0, 31.0, 9.5], width_ratios=[1.35, 1.0],
        hspace=0.055, wspace=0.06,
    )
    # The first gauge carries the block title, so it needs a taller cell.
    gs_gate = gs_b[1, 1].subgridspec(3, 1, height_ratios=[1.45, 1.0, 1.0], hspace=0.02)
    panel_b = {
        "head": fig.add_subplot(outer[2]),
        "excluded": fig.add_subplot(gs_b[0, :]),
        "bars": fig.add_subplot(gs_b[1, 0]),
        "gate": [fig.add_subplot(gs_gate[row, 0]) for row in range(3)],
        "ladder": fig.add_subplot(gs_b[2, 0]),
        "note": fig.add_subplot(gs_b[2, 1]),
        "spine": fig.add_subplot(gs_b[3, :]),
    }

    gs_c = outer[5].subgridspec(
        1, 2, width_ratios=[1.45, 1.0], wspace=0.06,
    )
    panel_c = {
        "head": fig.add_subplot(outer[4]),
        "plot": fig.add_subplot(gs_c[0, 0]),
        "note": fig.add_subplot(gs_c[0, 1]),
    }
    return {"a": panel_a, "b": panel_b, "c": panel_c}


def build_figure(
    map_data: dict[str, object],
    values: dict[int, np.ndarray],
    labels: list[str],
    counts: np.ndarray,
    gate_contract: dict[str, object],
    historical_input_contract: dict[str, object],
) -> plt.Figure:
    configure_matplotlib()
    fig = plt.figure(figsize=figstyle.figsize(WIDTH_MM, HEIGHT_MM), layout="constrained")
    panels = build_layout(fig)
    draw_panel_a(panels["a"], map_data)
    draw_panel_b(panels["b"], labels, counts, gate_contract, historical_input_contract)
    draw_panel_c(panels["c"], values)
    validate_visual_bounds(fig)
    return fig


def validate_visual_bounds(fig: plt.Figure) -> None:
    """Refuse any render whose text escapes its box or touches other text."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    artists = {artist.get_gid(): artist for artist in fig.findobj() if artist.get_gid()}
    for guard_id, minimum_padding_mm, label_suffixes in (
        ("excluded_inputs", 0.6, ("status", "text")),
        ("scope_status", 0.6, ("text",)),
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

    bar_labels = [
        artist for gid, artist in artists.items() if gid.startswith("guard.bar_label.")
    ]
    if len(bar_labels) != len(EXPECTED_HUC_COUNTS):
        raise RuntimeError("Panel-b guard requires all 15 HUC2 bar labels")
    for guard_id in ("gate_groups", "gate_effective_fraction", "gate_largest_share"):
        for suffix in ("bar", "text"):
            if f"guard.{guard_id}.{suffix}" not in artists:
                raise RuntimeError(f"Missing gate gauge artist guard.{guard_id}.{suffix}")

    # The layout guard proper: no two visible text artists in the whole figure
    # may overlap.  Every collision the manual layout produced was text on text.
    collisions = figstyle.check_overlaps(fig)
    if collisions:
        detail = "; ".join(f"{a!r} x {b!r}" for a, b in collisions[:8])
        raise RuntimeError(
            f"Figure 1 has {len(collisions)} overlapping text pairs -> {detail}"
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
                "nearest_neighbour_histogram_max_km": NN_HIST_MAX_KM,
                "nearest_neighbour_histogram_bin_km": NN_HIST_BIN_KM,
            },
            "b": {
                "bar_y_origin": 0,
                "bar_y_limit": 30,
                "bar_y_ticks": [0, 10, 20, 30],
                "gate_shared_numeric_axis": False,
                "gate_gauge_rule": (
                    "one axis per check, each scaled so its own threshold falls at "
                    f"{GATE_THRESHOLD_AXIS_FRACTION:.2f} of that gauge; no tick is shared"
                ),
                "cluster_ladder": "HUC2/4/6/8: 15/64/75/95 clusters; 0.636/0.507/0.485/0.758",
                "cluster_ladder_x_limits": [0.0, 1.0],
                "cluster_ladder_threshold": MIN_EFFECTIVE_FRACTION_DISPLAY,
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
        "layout": {
            "engine": "matplotlib constrained_layout over one figure-level GridSpec",
            "style_module": "paper/figstyle.py",
            "font_family": figstyle.resolved_font(),
            "outer_rows_mm": {
                "heading": HEAD_MM,
                "panel_a": LAYOUT_MM["a"],
                "panel_b": LAYOUT_MM["b"],
                "panel_c": LAYOUT_MM["c"],
            },
            "cells": {
                "a": ["heading", "map", "nearest_neighbour_histogram", "huc2_key"],
                "b": [
                    "heading", "excluded_inputs_band", "huc2_count_bars",
                    "gate_gauge_x3", "cluster_ladder", "huc8_note_and_scope_verdict",
                    "evidence_spine",
                ],
                "c": ["heading", "persistence_distributions", "reading_note"],
            },
            "manual_figure_fraction_placement": False,
            "text_collision_guard": "figstyle.check_overlaps; zero pairs required",
        },
        "bbox_guards_mm": {
            "excluded_input_internal_padding": 0.6,
            "scope_status_internal_padding": 0.6,
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
            "panel_a": (
                "HUC2 hue plus marker shape plus filled/open outline; marker size by "
                "observed-day count"
            ),
            "panel_b": (
                "bar hatch; gate gauge colour plus hatch plus the PASS/FAIL word; "
                "ladder colour plus hatch plus position against the drawn threshold"
            ),
            "panel_c": "circle/square/triangle plus hatch",
        },
        "glyph_policy": (
            "No U+2713 CHECK MARK and no U+2095 SUBSCRIPT SMALL H: Nimbus Sans carries "
            "neither, so verdicts are words and the horizon is an axis category."
        ),
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
            "plot_type": (
                "CONUS coordinate scatter coloured by HUC2, with the "
                "nearest-neighbour distance histogram in its own cell beside it"
            ),
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
            "layout_note": (
                "2026-08-07 redraw: the manual figure-fraction layout was replaced by "
                "one constrained-layout GridSpec.  Panel content is unchanged except "
                "that the nearest-neighbour histogram left panel (a)'s map as an inset "
                "and became its own cell, the three gate checks became single-bar "
                "gauges on their own scales, and the four cluster-ladder text cells "
                "became bars on one effective-cluster-fraction axis against the frozen "
                "0.75 threshold.  Figure height rose from 92 mm to 186 mm to keep every "
                "string at or above 7.5 pt."
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
            "font_family": figstyle.resolved_font(),
            "style_module": "paper/figstyle.py",
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
    # The ladder's source document was moved under docs/archive/ on 2026-08-07
    # with its bytes unchanged; bind whichever copy this tree carries.
    option_a_path = repo_root / "docs" / "archive" / "OPTION_A_DESCRIPTIVE_BENCHMARK_SCOPE.md"
    if not option_a_path.is_file():
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
    # The shared writer is the gate: it refuses to emit anything if two text
    # artists overlap.  The three files are then rewritten in place with the
    # deterministic metadata the submission package needs.
    figstyle.save(figure, FIGURE_ID, figure_dir, strict=True)
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

#!/usr/bin/env python3
"""
Editable ThermoRoute paper diagrams drawn with Python + SVG.

Why SVG?
--------
- Text and boxes remain editable in Inkscape/Illustrator.
- The same source exports vector PDF for LaTeX.
- Coordinates are explicit and easy to modify.

Outputs
-------
fig_main_model_concept.{svg,pdf,png}
fig_information_regime_framework.{svg,pdf,png}
figS_full_thermoroute_architecture.{svg,pdf,png}
"""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import svgwrite
import cairosvg


# ---------------------------------------------------------------------
# 1. Global style: edit here first
# ---------------------------------------------------------------------
C = {
    "ink": "#172033",
    "muted": "#5E6878",
    "line": "#9AA4B2",
    "bg": "#FFFFFF",
    "panel": "#F7F9FC",
    "panel_edge": "#D6DCE5",

    # Fixed semantic colors used in all diagrams
    "history": "#2F80ED",
    "history_light": "#EAF3FF",
    "forcing": "#F2994A",
    "forcing_light": "#FFF1E5",
    "geometry": "#219653",
    "geometry_light": "#E9F8F0",
    "model": "#7B61FF",
    "model_light": "#F0EDFF",
    "anchor": "#667085",
    "anchor_light": "#EEF1F5",
    "output": "#00A7A0",
    "output_light": "#E7FAF8",
    "warning": "#C2410C",
    "warning_light": "#FFF7ED",
}

FONT = "DejaVu Sans, Arial, Helvetica, sans-serif"


# ---------------------------------------------------------------------
# 2. SVG helpers
# ---------------------------------------------------------------------
def add_arrow_marker(dwg: svgwrite.Drawing, color: str, marker_id: str):
    marker = dwg.marker(
        insert=(10, 5),
        size=(10, 10),
        orient="auto",
        id=marker_id,
        markerUnits="strokeWidth",
    )
    marker.add(dwg.path(d="M 0 0 L 10 5 L 0 10 z", fill=color))
    dwg.defs.add(marker)
    return marker


def rounded_rect(
    dwg,
    x,
    y,
    w,
    h,
    fill,
    stroke,
    *,
    rx=16,
    sw=2,
    dash=None,
):
    kwargs = {}
    if dash:
        kwargs["stroke_dasharray"] = dash
    r = dwg.rect(
        insert=(x, y),
        size=(w, h),
        rx=rx,
        ry=rx,
        fill=fill,
        stroke=stroke,
        stroke_width=sw,
        **kwargs,
    )
    dwg.add(r)
    return r


def line(
    dwg,
    x1,
    y1,
    x2,
    y2,
    *,
    color=None,
    sw=3,
    marker=None,
    dash=None,
):
    kwargs = {}
    if marker:
        kwargs["marker_end"] = marker.get_funciri()
    if dash:
        kwargs["stroke_dasharray"] = dash
    obj = dwg.line(
        start=(x1, y1),
        end=(x2, y2),
        stroke=color or C["line"],
        stroke_width=sw,
        stroke_linecap="round",
        **kwargs,
    )
    dwg.add(obj)
    return obj


def path(
    dwg,
    d,
    *,
    color=None,
    sw=3,
    marker=None,
    dash=None,
    fill="none",
):
    kwargs = {}
    if marker:
        kwargs["marker_end"] = marker.get_funciri()
    if dash:
        kwargs["stroke_dasharray"] = dash
    obj = dwg.path(
        d=d,
        stroke=color or C["line"],
        stroke_width=sw,
        stroke_linecap="round",
        stroke_linejoin="round",
        fill=fill,
        **kwargs,
    )
    dwg.add(obj)
    return obj


def text(
    dwg,
    s,
    x,
    y,
    *,
    size=24,
    weight="normal",
    color=None,
    anchor="start",
    italic=False,
):
    obj = dwg.text(
        s,
        insert=(x, y),
        font_family=FONT,
        font_size=size,
        font_weight=weight,
        fill=color or C["ink"],
        text_anchor=anchor,
        font_style="italic" if italic else "normal",
    )
    dwg.add(obj)
    return obj


def multiline(
    dwg,
    lines,
    x,
    y,
    *,
    size=22,
    weight="normal",
    color=None,
    anchor="start",
    leading=1.28,
):
    if isinstance(lines, str):
        lines = lines.split("\n")
    obj = dwg.text(
        "",
        insert=(x, y),
        font_family=FONT,
        font_size=size,
        font_weight=weight,
        fill=color or C["ink"],
        text_anchor=anchor,
    )
    for i, ln in enumerate(lines):
        obj.add(dwg.tspan(ln, x=[x], dy=[0 if i == 0 else size * leading]))
    dwg.add(obj)
    return obj


def chip(
    dwg,
    x,
    y,
    w,
    h,
    title,
    subtitle,
    color,
    light,
    *,
    dashed=False,
):
    rounded_rect(
        dwg,
        x,
        y,
        w,
        h,
        light,
        color,
        rx=14,
        sw=2.2,
        dash="10,7" if dashed else None,
    )
    text(dwg, title, x + 18, y + 30, size=22, weight="bold", color=color)
    multiline(
        dwg,
        subtitle,
        x + 18,
        y + 60,
        size=17,
        color=C["muted"],
        leading=1.18,
    )


def export_all(dwg, outdir: Path, stem: str, width: int, height: int):
    outdir.mkdir(parents=True, exist_ok=True)
    svg_path = outdir / f"{stem}.svg"
    pdf_path = outdir / f"{stem}.pdf"
    png_path = outdir / f"{stem}.png"

    dwg.saveas(svg_path)
    cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
    cairosvg.svg2png(
        url=str(svg_path),
        write_to=str(png_path),
        output_width=width,
        output_height=height,
    )


# ---------------------------------------------------------------------
# 3. Main-text minimal model concept
# ---------------------------------------------------------------------
def draw_main_model_concept(outdir: Path):
    W, H = 1800, 760
    dwg = svgwrite.Drawing(size=(W, H), viewBox=f"0 0 {W} {H}")
    dwg.add(dwg.rect(insert=(0, 0), size=(W, H), fill=C["bg"]))

    arrow_gray = add_arrow_marker(dwg, C["line"], "concept_gray")
    arrow_blue = add_arrow_marker(dwg, C["history"], "concept_blue")
    arrow_orange = add_arrow_marker(dwg, C["forcing"], "concept_orange")
    arrow_model = add_arrow_marker(dwg, C["model"], "concept_model")

    text(
        dwg,
        "Common anchor-residual formulation under matched information",
        70,
        55,
        size=31,
        weight="bold",
    )
    text(
        dwg,
        "Main-text concept figure - scientific abstraction, not a layer-by-layer network diagram",
        70,
        90,
        size=18,
        color=C["muted"],
    )

    # Panel backgrounds
    rounded_rect(dwg, 60, 135, 430, 430, C["panel"], C["panel_edge"], rx=20)
    rounded_rect(dwg, 560, 135, 660, 430, C["panel"], C["panel_edge"], rx=20)
    rounded_rect(dwg, 1290, 135, 450, 430, C["panel"], C["panel_edge"], rx=20)

    text(dwg, "(a) Information inputs", 90, 180, size=24, weight="bold")
    text(dwg, "(b) Common prediction decomposition", 590, 180, size=24, weight="bold")
    text(dwg, "(c) Forecast", 1320, 180, size=24, weight="bold")

    chip(
        dwg,
        90,
        210,
        370,
        100,
        "Local thermal state Hᵢ,ₜ",
        ["WTEMP history - FLOW", "observedness masks"],
        C["history"],
        C["history_light"],
    )
    chip(
        dwg,
        90,
        330,
        370,
        108,
        "Meteorological history Xᵢ,≤ₜ",
        ["TEMP - PRCP - RHMEAN", "shortwave radiation - wind"],
        C["history"],
        C["history_light"],
    )
    chip(
        dwg,
        90,
        458,
        370,
        82,
        "Optional future forcing",
        ["F0: absent   |   F3: realized oracle"],
        C["forcing"],
        C["forcing_light"],
        dashed=True,
    )

    rounded_rect(dwg, 610, 215, 560, 118, C["anchor_light"], C["anchor"], rx=16, sw=2.5)
    text(
        dwg,
        "Damped-persistence anchor",
        890,
        252,
        size=25,
        weight="bold",
        color=C["anchor"],
        anchor="middle",
    )
    text(
        dwg,
        "Aᵢ,ₜ₊ₕ = cᵢ,ₜ₊ₕ + φᵢʰ ( yᵢ,ₜ - cᵢ,ₜ )",
        890,
        296,
        size=24,
        anchor="middle",
    )

    rounded_rect(dwg, 610, 372, 560, 135, C["model_light"], C["model"], rx=16, sw=2.5)
    text(
        dwg,
        "Residual learner ĝθ",
        890,
        410,
        size=25,
        weight="bold",
        color=C["model"],
        anchor="middle",
    )
    text(
        dwg,
        "predicts only what the anchor does not explain",
        890,
        446,
        size=19,
        color=C["muted"],
        anchor="middle",
    )
    text(
        dwg,
        "LightGBM - plain causal TCN - ThermoRoute",
        890,
        482,
        size=20,
        weight="bold",
        anchor="middle",
    )

    rounded_rect(dwg, 1335, 225, 360, 175, C["output_light"], C["output"], rx=18, sw=2.6)
    text(
        dwg,
        "Final point forecast",
        1515,
        270,
        size=25,
        weight="bold",
        color=C["output"],
        anchor="middle",
    )
    text(
        dwg,
        "ŷᵢ,ₜ₊ₕ = Aᵢ,ₜ₊ₕ + r̂ᵢ,ₜ₊ₕ",
        1515,
        318,
        size=22,
        weight="bold",
        anchor="middle",
    )
    multiline(
        dwg,
        [
            "ThermoRoute may apply a declared bounded residual;",
            "matched controls remain explicit.",
        ],
        1515,
        355,
        size=15,
        color=C["muted"],
        anchor="middle",
        leading=1.15,
    )

    # Information flow
    path(
        dwg,
        "M 460 260 C 520 260, 535 250, 610 250",
        color=C["history"],
        sw=3.2,
        marker=arrow_blue,
    )
    path(
        dwg,
        "M 460 385 C 520 385, 535 425, 610 425",
        color=C["history"],
        sw=3.2,
        marker=arrow_blue,
    )
    path(
        dwg,
        "M 460 500 C 525 500, 540 470, 610 470",
        color=C["forcing"],
        sw=3.2,
        marker=arrow_orange,
        dash="10,7",
    )
    path(
        dwg,
        "M 1170 275 C 1230 275, 1245 285, 1335 285",
        color=C["anchor"],
        sw=3.2,
        marker=arrow_gray,
    )
    path(
        dwg,
        "M 1170 438 C 1235 438, 1250 330, 1335 330",
        color=C["model"],
        sw=3.2,
        marker=arrow_model,
    )

    # Evaluation strip
    rounded_rect(dwg, 60, 615, 1680, 95, "#FAFBFD", "#C8D0DB", rx=16, sw=2)
    text(dwg, "(d) Evaluation contract", 90, 652, size=21, weight="bold")

    steps = [
        ("Identical station/date/horizon keys", 410, 315),
        ("RMSE within each station", 790, 290),
        ("Paired station contrast", 1110, 290),
        ("Median across stations", 1450, 290),
    ]
    for i, (label, center_x, box_w) in enumerate(steps):
        rounded_rect(dwg, center_x - box_w / 2, 635, box_w, 48, "#FFFFFF", "#B9C2CF", rx=12, sw=1.8)
        text(dwg, label, center_x, 666, size=17, weight="bold", anchor="middle")
        if i < len(steps) - 1:
            next_center, next_w = steps[i + 1][1], steps[i + 1][2]
            line(
                dwg,
                center_x + box_w / 2 + 8,
                659,
                next_center - next_w / 2 - 8,
                659,
                sw=2.5,
                marker=arrow_gray,
            )

    export_all(dwg, outdir, "fig_main_model_concept", W, H)


# ---------------------------------------------------------------------
# 4. Main-text F x L x G x A experimental design
# ---------------------------------------------------------------------
def draw_information_regime_framework(outdir: Path):
    W, H = 2100, 1380
    dwg = svgwrite.Drawing(size=(W, H), viewBox=f"0 0 {W} {H}")
    dwg.add(dwg.rect(insert=(0, 0), size=(W, H), fill=C["bg"]))

    arrow_gray = add_arrow_marker(dwg, C["line"], "framework_gray")

    text(
        dwg,
        "Experimental identification of river-temperature information regimes",
        65,
        58,
        size=34,
        weight="bold",
    )
    text(
        dwg,
        "Core design: 2 forcing levels x 2 local-information levels x 2 geometries x 2 model classes x 3 leads = 48 matched cells",
        65,
        96,
        size=19,
        color=C["muted"],
    )

    rounded_rect(dwg, 45, 135, 470, 930, C["panel"], C["panel_edge"], rx=20)
    rounded_rect(dwg, 540, 135, 1030, 930, C["panel"], C["panel_edge"], rx=20)
    rounded_rect(dwg, 1595, 135, 460, 930, C["panel"], C["panel_edge"], rx=20)

    text(dwg, "(a) Four experimental axes", 75, 180, size=25, weight="bold")
    text(dwg, "(b) Core matched response surface", 570, 180, size=25, weight="bold")
    text(dwg, "(c) Identified contrasts", 1625, 180, size=25, weight="bold")

    chip(
        dwg,
        75,
        215,
        410,
        118,
        "F - Future forcing",
        ["F0  no future meteorology", "F3  realized future meteorology", "     retrospective oracle"],
        C["forcing"],
        C["forcing_light"],
    )
    chip(
        dwg,
        75,
        355,
        410,
        122,
        "L - Local information",
        ["L0  history-rich gauged site", "L2  no target-site WT history", "L1 / L2-U2: diagnostics"],
        C["history"],
        C["history_light"],
    )
    chip(
        dwg,
        75,
        500,
        410,
        105,
        "G - Spatial geometry",
        ["Random-site", "Whole-region"],
        C["geometry"],
        C["geometry_light"],
    )
    chip(
        dwg,
        75,
        630,
        410,
        130,
        "A - Model class",
        ["LightGBM", "Plain causal TCN", "ThermoRoute: selected extension"],
        C["model"],
        C["model_light"],
    )

    rounded_rect(dwg, 75, 795, 410, 220, "#FFFFFF", "#C9D1DC", rx=14, sw=1.8)
    text(dwg, "Forecast leads", 100, 835, size=21, weight="bold")
    for i, label in enumerate(["1 day", "3 days", "7 days"]):
        cx = 145 + i * 115
        dwg.add(dwg.circle(center=(cx, 890), r=34, fill="#FFFFFF", stroke=C["output"], stroke_width=3))
        text(dwg, label, cx, 897, size=18, weight="bold", color=C["output"], anchor="middle")
    multiline(
        dwg,
        ["Every factor is defined by an information", "contract, not by a model-specific feature set."],
        100,
        960,
        size=17,
        color=C["muted"],
    )

    # Forcing headers
    x0s = [660, 1100]
    facet_w = 410
    forcing_titles = [
        ("F0 - no future forcing", C["history"]),
        ("F3 - realized oracle", C["forcing"]),
    ]
    for j, (title_s, color) in enumerate(forcing_titles):
        x0 = x0s[j]
        rounded_rect(dwg, x0, 205, facet_w, 42, "#FFFFFF", color, rx=11, sw=2.2)
        text(dwg, title_s, x0 + facet_w / 2, 233, size=19, weight="bold", color=color, anchor="middle")

    model_rows = [
        (255, "LightGBM", C["model"]),
        (630, "Plain causal TCN", "#B547CE"),
    ]

    def mini_matrix(x0, y0):
        text(dwg, "Random-site", x0 + 140, y0 + 24, size=16, weight="bold", color=C["geometry"], anchor="middle")
        text(dwg, "Whole-region", x0 + 315, y0 + 24, size=16, weight="bold", color=C["geometry"], anchor="middle")
        text(dwg, "L0", x0 + 38, y0 + 105, size=17, weight="bold", color=C["history"], anchor="middle")
        text(dwg, "L2", x0 + 38, y0 + 225, size=17, weight="bold", color=C["forcing"], anchor="middle")

        cell_x = [x0 + 70, x0 + 245]
        cell_y = [y0 + 55, y0 + 175]
        for r in range(2):
            for c in range(2):
                fill = C["history_light"] if r == 0 else C["forcing_light"]
                stroke = "#AEB7C4" if c == 0 else C["geometry"]
                dash = None if c == 0 else "8,5"
                rounded_rect(dwg, cell_x[c], cell_y[r], 145, 92, fill, stroke, rx=12, sw=2.2, dash=dash)
                text(dwg, "Rᵢ,ₕ", cell_x[c] + 72.5, cell_y[r] + 39, size=24, weight="bold", anchor="middle")
                text(dwg, "same keys", cell_x[c] + 72.5, cell_y[r] + 67, size=15, color=C["muted"], anchor="middle")

    for y0, model_name, model_color in model_rows:
        rounded_rect(dwg, 570, y0 + 45, 72, 220, "#FFFFFF", model_color, rx=12, sw=2.2)
        label = dwg.text(
            model_name,
            insert=(606, y0 + 155),
            font_family=FONT,
            font_size=20,
            font_weight="bold",
            fill=model_color,
            text_anchor="middle",
            transform=f"rotate(-90 606 {y0 + 155})",
        )
        dwg.add(label)
        mini_matrix(660, y0)
        mini_matrix(1100, y0)

    rounded_rect(dwg, 600, 990, 910, 48, "#FFFFFF", "#C8D0DB", rx=12, sw=1.7)
    text(
        dwg,
        "Four 2 x 2 matrices are repeated independently at h = 1, 3, and 7 days",
        1055,
        1021,
        size=17,
        weight="bold",
        anchor="middle",
    )

    # Main effects
    contrast_boxes = [
        ("Future-forcing value", "V_F = R(F0) - R(F3)", C["forcing"], C["forcing_light"]),
        ("Local-state value", "V_L = R(L2) - R(L0)", C["history"], C["history_light"]),
        ("Geometry penalty", "P_G = R(region) - R(random)", C["geometry"], C["geometry_light"]),
        ("Architecture value", "V_A = R(LGBM) - R(TCN)", C["model"], C["model_light"]),
    ]
    yy = 220
    for title_s, formula, color, light in contrast_boxes:
        rounded_rect(dwg, 1625, yy, 400, 104, light, color, rx=14, sw=2.1)
        text(dwg, title_s, 1648, yy + 34, size=19, weight="bold", color=color)
        text(dwg, formula, 1648, yy + 72, size=19, weight="bold")
        yy += 122

    rounded_rect(dwg, 1625, 720, 400, 286, "#FFFFFF", "#BFC7D3", rx=14, sw=2)
    text(dwg, "Double differences", 1648, 758, size=21, weight="bold")
    interactions = [
        ("F x L", "Does forcing value change without local WT?"),
        ("F x A", "Does a sequence model use future forcing better?"),
        ("L x G", "Does local WT mask spatial-transfer difficulty?"),
        ("L x A", "Does architecture matter in cold-start transfer?"),
    ]
    yy2 = 800
    for tag, desc in interactions:
        rounded_rect(dwg, 1648, yy2 - 24, 68, 34, C["model_light"], C["model"], rx=10, sw=1.6)
        text(dwg, tag, 1682, yy2, size=16, weight="bold", color=C["model"], anchor="middle")
        multiline(
            dwg,
            textwrap.wrap(desc, width=34),
            1730,
            yy2 - 3,
            size=15,
            color=C["muted"],
            leading=1.15,
        )
        yy2 += 54

    # Evaluation contract
    rounded_rect(dwg, 45, 1100, 2010, 225, "#FAFBFD", "#C8D0DB", rx=20, sw=2)
    text(dwg, "(d) One scoring contract for every cell and contrast", 75, 1145, size=24, weight="bold")
    pipeline = [
        ("Common registry", "same station / issue / target / lead keys"),
        ("Station metric", "RMSE computed within each station"),
        ("Paired contrast", "only matched station cells enter"),
        ("Across stations", "unweighted median is primary"),
        ("Sensitivity", "HUC - year - season - split seed"),
    ]
    px = [230, 600, 970, 1340, 1740]
    for i, ((title_s, subtitle), center_x) in enumerate(zip(pipeline, px)):
        rounded_rect(dwg, center_x - 145, 1175, 290, 105, "#FFFFFF", "#B9C2CF", rx=14, sw=1.8)
        text(dwg, title_s, center_x, 1210, size=18, weight="bold", anchor="middle")
        multiline(
            dwg,
            textwrap.wrap(subtitle, width=31),
            center_x,
            1242,
            size=14,
            color=C["muted"],
            anchor="middle",
            leading=1.12,
        )
        if i < len(pipeline) - 1:
            line(
                dwg,
                center_x + 150,
                1228,
                px[i + 1] - 155,
                1228,
                sw=2.5,
                marker=arrow_gray,
            )

    text(
        dwg,
        "F3 is a retrospective realized-meteorology oracle; it is not an operational forecast.",
        1050,
        1305,
        size=16,
        weight="bold",
        color=C["warning"],
        anchor="middle",
    )

    export_all(dwg, outdir, "fig_information_regime_framework", W, H)


# ---------------------------------------------------------------------
# 5. Supporting-Information full architecture
# ---------------------------------------------------------------------
def draw_full_architecture(outdir: Path):
    W, H = 2300, 1380
    dwg = svgwrite.Drawing(size=(W, H), viewBox=f"0 0 {W} {H}")
    dwg.add(dwg.rect(insert=(0, 0), size=(W, H), fill=C["bg"]))

    arrow_gray = add_arrow_marker(dwg, C["line"], "arch_gray")
    arrow_blue = add_arrow_marker(dwg, C["history"], "arch_blue")
    arrow_orange = add_arrow_marker(dwg, C["forcing"], "arch_orange")
    arrow_model = add_arrow_marker(dwg, C["model"], "arch_model")
    arrow_output = add_arrow_marker(dwg, C["output"], "arch_output")

    text(dwg, "Full ThermoRoute architecture and calibration dataflow", 65, 58, size=34, weight="bold")
    text(
        dwg,
        "Supporting Information figure - full scientific modules, simplified computational operators",
        65,
        96,
        size=19,
        color=C["muted"],
    )

    columns = [
        (45, 140, 470, 1070, "(a) Inputs and masks"),
        (540, 140, 545, 1070, "(b) Anchor and proposal"),
        (1110, 140, 520, 1070, "(c) Residual representation"),
        (1655, 140, 600, 1070, "(d) Heads and calibration"),
    ]
    for x, y, w, h, title_s in columns:
        rounded_rect(dwg, x, y, w, h, C["panel"], C["panel_edge"], rx=20, sw=2)
        text(dwg, title_s, x + 30, y + 46, size=24, weight="bold")

    # Inputs
    chip(
        dwg,
        80,
        215,
        400,
        145,
        "32-day construction buffer",
        ["WTEMP - FLOW - TEMP - PRCP", "RHMEAN - DH - WDSP", "values + observedness masks"],
        C["history"],
        C["history_light"],
    )
    chip(
        dwg,
        80,
        390,
        400,
        108,
        "Calendar and horizon",
        ["season encoding", "h in {1, 3, 7 days}"],
        C["history"],
        C["history_light"],
    )
    chip(
        dwg,
        80,
        530,
        400,
        125,
        "Optional forcing-regime extension",
        ["future TEMP / PRCP / RH / DH / wind", "F3: realized gridded oracle"],
        C["forcing"],
        C["forcing_light"],
        dashed=True,
    )
    rounded_rect(dwg, 80, 700, 400, 92, "#F3F4F6", "#9CA3AF", rx=14, sw=1.8, dash="8,6")
    text(dwg, "WLEVEL excluded", 280, 740, size=21, weight="bold", color="#6B7280", anchor="middle")
    text(dwg, "not available to any model head", 280, 770, size=16, color=C["muted"], anchor="middle")
    line(dwg, 105, 716, 138, 750, color="#9CA3AF", sw=3)
    line(dwg, 138, 716, 105, 750, color="#9CA3AF", sw=3)

    rounded_rect(dwg, 80, 840, 400, 245, "#FFFFFF", "#C5CDD8", rx=14, sw=1.8)
    text(dwg, "Information-boundary notes", 105, 880, size=20, weight="bold")
    multiline(
        dwg,
        [
            "- Main F0 model sees dates <= issue time t.",
            "- Future channels enter only in declared",
            "  forcing-regime extensions.",
            "- Target WTEMP and target observedness",
            "  remain labels, never predictors.",
        ],
        105,
        920,
        size=16,
        color=C["muted"],
        leading=1.35,
    )

    # Anchor
    rounded_rect(dwg, 585, 215, 455, 255, C["anchor_light"], C["anchor"], rx=18, sw=2.5)
    text(dwg, "Fixed damped-persistence branch", 812, 255, size=23, weight="bold", color=C["anchor"], anchor="middle")
    rounded_rect(dwg, 625, 285, 150, 80, "#FFFFFF", C["anchor"], rx=12, sw=1.8)
    text(dwg, "Climatology", 700, 320, size=17, weight="bold", anchor="middle")
    text(dwg, "cᵢ,ₜ₊ₕ", 700, 349, size=19, weight="bold", color=C["anchor"], anchor="middle")
    rounded_rect(dwg, 850, 285, 150, 80, "#FFFFFF", C["anchor"], rx=12, sw=1.8)
    text(dwg, "Decay", 925, 320, size=17, weight="bold", anchor="middle")
    text(dwg, "φᵢʰ", 925, 349, size=20, weight="bold", color=C["anchor"], anchor="middle")
    line(dwg, 775, 325, 850, 325, color=C["anchor"], sw=2.6, marker=arrow_gray)
    rounded_rect(dwg, 665, 390, 310, 58, "#FFFFFF", C["anchor"], rx=12, sw=1.8)
    text(dwg, "Anchor Aᵢ,ₜ₊ₕ", 820, 427, size=21, weight="bold", color=C["anchor"], anchor="middle")

    # Proposal
    rounded_rect(dwg, 585, 520, 455, 360, C["model_light"], C["model"], rx=18, sw=2.5)
    text(dwg, "Learned relaxation proposal", 812, 560, size=23, weight="bold", color=C["model"], anchor="middle")
    rounded_rect(dwg, 625, 605, 150, 78, "#FFFFFF", C["model"], rx=12, sw=1.8)
    text(dwg, "Equilibrium", 700, 637, size=16, weight="bold", anchor="middle")
    text(dwg, "eᵢ,ₜ", 700, 666, size=19, weight="bold", color=C["model"], anchor="middle")
    rounded_rect(dwg, 850, 605, 150, 78, "#FFFFFF", C["model"], rx=12, sw=1.8)
    text(dwg, "Relaxation", 925, 637, size=16, weight="bold", anchor="middle")
    text(dwg, "κᵢ,ₜ", 925, 666, size=19, weight="bold", color=C["model"], anchor="middle")
    rounded_rect(dwg, 665, 735, 310, 70, "#FFFFFF", C["model"], rx=12, sw=1.8)
    text(dwg, "Proposal Pᵢ,ₜ₊ₕ", 820, 778, size=21, weight="bold", color=C["model"], anchor="middle")
    path(dwg, "M 700 683 C 700 715, 760 735, 780 735", color=C["model"], sw=2.4, marker=arrow_model)
    path(dwg, "M 925 683 C 925 715, 875 735, 855 735", color=C["model"], sw=2.4, marker=arrow_model)
    multiline(
        dwg,
        ["κ is a statistical allocation parameter,", "not a heat-transfer coefficient."],
        812,
        835,
        size=15,
        color=C["muted"],
        anchor="middle",
    )

    # Inputs to branches
    path(dwg, "M 480 290 C 535 290, 535 330, 625 330", color=C["history"], sw=3, marker=arrow_blue)
    path(dwg, "M 480 455 C 535 455, 535 640, 625 640", color=C["history"], sw=3, marker=arrow_blue)
    path(dwg, "M 480 590 C 545 590, 550 680, 850 640", color=C["forcing"], sw=3, marker=arrow_orange, dash="10,7")

    # Residual representation
    rounded_rect(dwg, 1150, 215, 440, 145, C["model_light"], C["model"], rx=18, sw=2.5)
    text(dwg, "Variable-lag selector", 1370, 258, size=23, weight="bold", color=C["model"], anchor="middle")
    text(dwg, "7 variables x lags 0-14", 1370, 297, size=18, weight="bold", anchor="middle")
    text(dwg, "allocation over input history", 1370, 330, size=16, color=C["muted"], anchor="middle")

    rounded_rect(dwg, 1150, 420, 440, 160, C["model_light"], C["model"], rx=18, sw=2.5)
    text(dwg, "Left-looking temporal encoder", 1370, 462, size=23, weight="bold", color=C["model"], anchor="middle")
    text(dwg, "TCN: 2 residual blocks", 1370, 502, size=18, weight="bold", anchor="middle")
    text(dwg, "kernel 3 - dilations 1 and 2", 1370, 535, size=17, color=C["muted"], anchor="middle")

    rounded_rect(dwg, 1150, 650, 440, 150, C["model_light"], C["model"], rx=18, sw=2.5)
    text(dwg, "Three-expert mixture", 1370, 694, size=23, weight="bold", color=C["model"], anchor="middle")
    text(dwg, "soft gate combines expert states", 1370, 737, size=18, color=C["muted"], anchor="middle")
    text(dwg, "residual rᵢ,ₜ₊ₕ", 1370, 775, size=20, weight="bold", color=C["model"], anchor="middle")

    line(dwg, 1370, 360, 1370, 420, color=C["model"], sw=3, marker=arrow_model)
    line(dwg, 1370, 580, 1370, 650, color=C["model"], sw=3, marker=arrow_model)
    path(dwg, "M 480 300 C 700 300, 930 250, 1150 280", color=C["history"], sw=3, marker=arrow_blue)
    path(dwg, "M 480 590 C 720 590, 930 525, 1150 500", color=C["forcing"], sw=3, marker=arrow_orange, dash="10,7")

    rounded_rect(dwg, 1150, 875, 440, 195, "#FFFFFF", "#C5CDD8", rx=14, sw=1.8)
    text(dwg, "Interpretation boundary", 1178, 915, size=20, weight="bold")
    multiline(
        dwg,
        [
            "- Selector allocates variable-lag weight.",
            "- It is not river-network routing.",
            "- No verified topology or travel time enters.",
        ],
        1178,
        955,
        size=16,
        color=C["muted"],
        leading=1.35,
    )

    # Output heads
    rounded_rect(dwg, 1700, 215, 510, 135, "#FFFFFF", "#BFC7D3", rx=16, sw=2)
    text(dwg, "Combine proposal and residual", 1955, 255, size=22, weight="bold", anchor="middle")
    text(dwg, "z = P - A + r", 1955, 305, size=27, weight="bold", color=C["model"], anchor="middle")

    path(dwg, "M 975 420 C 1300 420, 1500 270, 1700 270", color=C["anchor"], sw=3, marker=arrow_gray)
    path(dwg, "M 975 770 C 1300 770, 1480 315, 1700 315", color=C["model"], sw=3, marker=arrow_model)
    path(dwg, "M 1590 735 C 1640 735, 1655 315, 1700 315", color=C["model"], sw=3, marker=arrow_model)

    rounded_rect(dwg, 1700, 400, 510, 180, C["output_light"], C["output"], rx=18, sw=2.6)
    text(dwg, "Point head", 1955, 443, size=23, weight="bold", color=C["output"], anchor="middle")
    text(dwg, "ŷ = A + δ tanh(z / δ)", 1955, 495, size=25, weight="bold", anchor="middle")
    text(dwg, "δ = 1.0 °C relative to the anchor", 1955, 532, size=17, color=C["muted"], anchor="middle")
    rounded_rect(dwg, 1770, 548, 370, 24, "#FFFFFF", C["warning"], rx=10, sw=1.5, dash="7,5")
    text(dwg, "unbounded control in hard-regime sensitivity", 1955, 566, size=13, weight="bold", color=C["warning"], anchor="middle")

    rounded_rect(dwg, 1700, 640, 240, 165, "#FFFFFF", C["output"], rx=16, sw=2)
    text(dwg, "Quantile heads", 1820, 680, size=20, weight="bold", color=C["output"], anchor="middle")
    text(dwg, "q05 - q50 - q95", 1820, 724, size=20, weight="bold", anchor="middle")
    text(dwg, "member average", 1820, 766, size=16, color=C["muted"], anchor="middle")

    rounded_rect(dwg, 1970, 640, 240, 165, "#FFFFFF", C["output"], rx=16, sw=2)
    text(dwg, "Event head", 2090, 680, size=20, weight="bold", color=C["output"], anchor="middle")
    text(dwg, "q90 exceedance", 2090, 724, size=19, weight="bold", anchor="middle")
    text(dwg, "station-specific", 2090, 766, size=16, color=C["muted"], anchor="middle")

    rounded_rect(dwg, 1700, 860, 510, 220, "#FFFFFF", "#BFC7D3", rx=16, sw=2)
    text(dwg, "Calibration fitted on 2018 only", 1955, 902, size=22, weight="bold", anchor="middle")
    rounded_rect(dwg, 1740, 940, 195, 90, C["output_light"], C["output"], rx=12, sw=1.8)
    text(dwg, "Platt map", 1837, 978, size=18, weight="bold", color=C["output"], anchor="middle")
    text(dwg, "event probability", 1837, 1008, size=15, color=C["muted"], anchor="middle")
    rounded_rect(dwg, 1975, 940, 195, 90, C["output_light"], C["output"], rx=12, sw=1.8)
    text(dwg, "CQR offset", 2072, 978, size=18, weight="bold", color=C["output"], anchor="middle")
    text(dwg, "prediction interval", 2072, 1008, size=15, color=C["muted"], anchor="middle")

    line(dwg, 1955, 350, 1955, 400, color=C["output"], sw=3, marker=arrow_output)
    path(dwg, "M 1955 580 C 1955 610, 1820 620, 1820 640", color=C["output"], sw=2.6, marker=arrow_output)
    path(dwg, "M 1955 580 C 1955 610, 2090 620, 2090 640", color=C["output"], sw=2.6, marker=arrow_output)
    path(dwg, "M 1820 805 C 1820 835, 2072 830, 2072 940", color=C["output"], sw=2.4, marker=arrow_output)
    path(dwg, "M 2090 805 C 2090 835, 1837 830, 1837 940", color=C["output"], sw=2.4, marker=arrow_output)

    # Bottom interpretive note
    rounded_rect(dwg, 45, 1240, 2210, 95, C["warning_light"], "#E5B38A", rx=16, sw=1.8)
    text(dwg, "Interpretive limits:", 75, 1277, size=18, weight="bold", color=C["warning"])
    text(dwg, "selector is not physical river routing", 270, 1277, size=17, weight="bold", color=C["warning"])
    text(dwg, "-", 605, 1277, size=17, weight="bold", color=C["muted"])
    text(dwg, "+/- 1 C bound is not an error or safety guarantee", 635, 1277, size=17, weight="bold", color=C["warning"])
    text(dwg, "-", 1100, 1277, size=17, weight="bold", color=C["muted"])
    text(dwg, "F3 is a retrospective oracle, not an operational forecast", 1130, 1277, size=17, weight="bold", color=C["warning"])
    text(
        dwg,
        "Full fitting, loss, calibration, and ablation details remain in SI text and tables.",
        75,
        1312,
        size=15,
        color=C["muted"],
    )

    export_all(dwg, outdir, "figS_full_thermoroute_architecture", W, H)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figure",
        choices=["all", "concept", "framework", "architecture"],
        default="all",
    )
    parser.add_argument("--outdir", default="generated_figures")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    if args.figure in ("all", "concept"):
        draw_main_model_concept(outdir)
    if args.figure in ("all", "framework"):
        draw_information_regime_framework(outdir)
    if args.figure in ("all", "architecture"):
        draw_full_architecture(outdir)

    print(f"Figures written to {outdir.resolve()}")


if __name__ == "__main__":
    main()

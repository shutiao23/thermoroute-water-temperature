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
"""POST skeleton for main-text Figures 2--4.

This module encodes the panel, value-ID, dependency, render-profile, and
prohibited-semantics contracts of ``paper/FIGURE_REDRAW_SPEC.md`` (Figure 2--4
sections) in the fail-closed style of ``render_fig01_preopening_concept.py``.
It is a skeleton: before the POST gate it renders nothing -- no SVG/PDF/PNG, no
empty axes, no ``PENDING`` boxes, no zero-filled values.  ``--status`` is the
only read-only inspection mode.

Gate order (all required before any panel builder may run):

1. ``paper/FIGURE_REDRAW_SPEC.md`` no longer lists the figure in a
   ``*TEMPLATE_ONLY*`` state;
2. the verified opening receipt exists under
   ``outputs/confirmatory/route_a_*/opening_receipt_v1.json``;
3. every dependency declared ``Severity.REQUIRED`` resolves;
4. every required ``value_id`` is declared through :class:`ValueBinder` and
   every visible mark routes its cells through ``ValueBinder.mark``.

Dependencies declared ``Severity.QUALIFIER`` never block.  They are resolved,
recorded, and reported so the render receipt carries an accurate provenance
statement.

Stage-19 disposition (2026-08-05)
---------------------------------
Stage-19 (``scripts/19_probabilistic.py``) will not be produced for this
submission.  Its frozen contract check uses ``q05 >= q95`` -- equality
included -- and the protocol forbids evaluation-time repair; relaxing the check
means editing ``scripts/``, which invalidates all four training receipts.
Stage-10 (``scripts/10_usgs_analysis.py``) validates the Stage-19 receipt before
reading any table and therefore cascades.

The measured cause is a **zero-width nominal interval (degenerate quantile
head)**, not quantile crossing.  A full scan of
``outputs/predictions/usgs_predictions_with_perstation_v2.parquet``
(26,993,675 member-level rows with complete quantile heads) found **0** strict
ordering violations and **135** rows where ``q05 == q50 == q95`` bit-identically
(0.0005%; LightGBM 123, LightGBM-perstation 12; maximum monotonicity violation
exactly 0.000 degrees C).  Split-CQR would have widened every one of them, so no
delivered interval is degenerate.  Any wording in this module, in the spec, or in
a caption that says "quantile crossing" is factually wrong.

Consequently ``blocked_by_stage19`` is retired as a hard refusal.

The target-period probability family is NOT affected
----------------------------------------------------
Determination of 2026-08-05: the withheld Stage-19 script is a *development*
tool.  The target-period probabilistic family is computed independently by the
trusted scorer inside the one-time opening.  ``opening.py:8932-8948`` emits, per
cohort x model x horizon, ``coverage_90``, ``mean_interval_width_c``,
``pinball_q05/q50/q95_c``, ``equal_weight_three_quantile_pinball_mean_c``,
``brier_score``, ``frozen_reference_brier_score``,
``brier_skill_frozen_seasonal``, ``log_loss``, ``auroc``, ``auprc``,
``ece_10_equal_width``, ``calibration_intercept``, ``calibration_slope`` and
``event_rate``, plus station-balanced reliability bins (``opening.py:8920-8926``)
against a validated frozen seasonal event reference.

Benchmark restructure (2026-08-06)
----------------------------------
``docs/PAPER_BENCHMARK_RESTRUCTURE.md`` turned the manuscript into an evaluation
benchmark and needs four main figures whose jobs did not match Figures 1--4 as
specified.  Applied here, with ``paper/FIGURE_REDRAW_SPEC.md`` sections 4 and 6.4
as the authority:

* **fig02** gains panel (a), the *target-period* reference ladder, and its
  former panels shift to (b)--(e).
* **fig03** is now the spatial-partition figure: temporal versus random
  held-site versus whole-region holdout.  It is **development-period**, because
  the one-time opening emits no held-region artifact at all
  (``docs/R13_POSTOPEN_TABLE_RENDERER.md`` section 6), and it therefore carries
  ``EvidencePeriod.DEVELOPMENT``, a mandatory in-panel scope band, and an
  explicit prohibition on comparison with any target-period figure.  It is still
  gated on the verified opening receipt like every other POST figure.
* **fig04** is now regional/seasonal heterogeneity plus the coverage--width
  plane.  It absorbs the former fig03 panel (a) and the former fig04 panel (c).
* The former fig03 event-score and reliability panels move to figS5; the former
  fig04 architecture panel moves to the new figS10, its attrition waterfall to
  figS6, and its external arm to figS8.  Nothing is dropped.

Two proposals in the restructure brief are **refused** here because one figure
never mixes two evidence periods: a development-period reference ladder inside
target-period fig02, and a development-period conformal panel inside
target-period fig04.  Their content lives in fig03(a) and figS9 respectively.
``PanelSpec.evidence_period`` plus ``validate_manifest`` make that refusal
machine-enforced rather than editorial: a panel that declares a period different
from its figure raises ``ManifestError`` before any gate is even read.

Pre-opening guard
-----------------
``opening.py:7093-7094`` applies a **strict** ``q05 < q95`` to the
member-averaged nominal heads and raises ``OpeningContractError`` -- aborting the
whole one-time opening -- on violation.  This is the same degeneracy trap that
stopped Stage-19, one layer up.  Measured on the development panel: member
averaging clears every affected LightGBM key (5 members); the only 12 survivors
are single-member ``LightGBM-perstation`` keys, and that model is absent from
``PRIMARY_MODELS`` and from the confirmatory protocol.  Re-run the check against
the target-period predictions **before** executing the one-time opening.

Evidence-period discipline
--------------------------
Stage-22 (``scripts/22_adaptive_conformal.py``) is **development-period**
evidence: ``outputs/tables/aci_coverage.csv`` spans 2019-01-01 to 2020-12-24 and
``protocols/route_a_confirmatory_v1.json`` sets
``time_holdout.primary_target_start = 2021-01-01``.  Under spec section 2.1 a
development-only source may support a clearly labelled development diagnostic but
may never fill a target-period interval, probability, or verdict coordinate.  It
is confined to supporting Figure S9.  Any figure declaring
``evidence_period = EvidencePeriod.DEVELOPMENT`` must render an in-panel scope
band; :func:`assert_evidence_period_labelled` refuses otherwise.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

FIGURE_SCHEMA_VERSION = "4.0.0-post-skeleton-stage19-independent"

EVIDENCE_ROOT_ENV = "THERMOROUTE_FIGURE_EVIDENCE_ROOT"

# --------------------------------------------------------------------------
# AGU / WRR render profile.
#
# Authority is paper/FIGURE_REDRAW_SPEC.md section 3.2.  The committed AGU class
# (agujournal2019.cls) sets \textwidth 5.5in = 139.7 mm, so 140 mm is the
# governing full-width target.  AGU's published single-column range is 50--85 mm;
# authoring at 85 mm avoids a production rescale that would push 7.5 pt ticks
# below the declared floor.  190 mm and 95 mm are NOT AGU sizes -- 190 mm exceeds
# the 170 mm double-column maximum and 95 mm exceeds the 85 mm single-column
# maximum.  Do not reintroduce them.
# --------------------------------------------------------------------------
# Full width is 139.7 mm, not 140: the AGU class sets \textwidth to 5.5 in =
# 397.48 TeX pt, so a 140 mm figure overshoots by 0.30 mm and makes pdflatex
# emit an Overfull \hbox of 0.85 pt for every figure, burying real warnings.
# These duplicate paper/figstyle.py because the shared module is imported far
# below (it needs sys.path set up first); the values are asserted equal there.
SINGLE_COLUMN_MM = 85.0
FULL_WIDTH_MM = 139.7
MAX_HEIGHT_MM = 228.0
MIN_BODY_PT = 8.0
MIN_ABSOLUTE_PT = 7.5
PANEL_LABEL_PT_RANGE = (9.0, 10.0)
MIN_STROKE_PT = 0.6
VECTOR_FORMATS = ("pdf", "svg")
RASTER_MIN_DPI = 300
LINE_ART_MIN_DPI = 600

# Okabe--Ito derived semantic palette (spec section 3.1).  Colour never carries a
# result on its own: every series also varies marker shape, dash pattern, or
# hatch, and every panel must survive a grayscale check.
PALETTE = {
    "TR_BLUE": "#0072B2",
    "TR_BLUE_LIGHT": "#DCEAF4",
    "DAMPED_ORANGE": "#E69F00",
    "PERSIST_GRAY": "#777777",
    "CLIM_GRAY": "#B8B8B8",
    "LGBM_PURPLE": "#CC79A7",
    "LSTM_GREEN": "#009E73",
    "ALLOWED_TEAL": "#008C7A",
    "WARNING_VERMILION": "#D55E00",
    "GATE_RED": "#B2182B",
    "NEUTRAL_INK": "#202020",
    "NEUTRAL_GRID": "#D0D0D0",
    "NA_FILL": "#F2F2F2",
}

# --------------------------------------------------------------------------
# Source-controlled paths (resolved against the paper repo root).
# Mirrors src/thermoroute/model_suite.py and development_controls_gate.py.
#   STAGE9_COMPLETION_RECEIPT_PATH   = outputs/models/route_a_stage09_completion.json
#   STAGE16_COMPLETION_RECEIPT_PATH  = outputs/models/route_a_stage16_completion.json
#   STAGE25_COMPLETION_RECEIPT_PATH  = outputs/models/route_a_stage25_completion.json
#   STAGE09B_COMPLETION_RECEIPT_PATH = outputs/models/route_a_stage09b_completion.json
# (the last from development_controls_gate.py); verified 2026-08-05.
# --------------------------------------------------------------------------
OPENING_RECEIPT_GLOB = "outputs/confirmatory/route_a_*/opening_receipt_v1.json"

# ---------------------------------------------------------------------------
# Target-period artifacts written by the trusted scorer inside the one-time
# opening.  Names are exact, taken from the state-path table in
# src/thermoroute/opening.py:1226-1281 (verified 2026-08-05); the only unknown
# is the run namespace, which the leading glob absorbs.  These are the
# fail-closed guards on target-period evidence: they resolve only after the
# opening runs.  Do not delete them to "unblock" a figure.
# ---------------------------------------------------------------------------
_TRUSTED = "outputs/confirmatory/route_a_*/trusted"
POST_TEMPORAL_PREDICTIONS = f"{_TRUSTED}/temporal_predictions_v1.parquet"
POST_EXTERNAL_PREDICTIONS = f"{_TRUSTED}/external_predictions_v1.parquet"
POST_AVAILABILITY_REGISTRY = f"{_TRUSTED}/availability_registry_v1.csv"
POST_STATISTICS = f"{_TRUSTED}/statistics_v1.json"
POST_PROBABILISTIC_EVALUATION = f"{_TRUSTED}/probabilistic_evaluation_v2.json"
POST_SPATIAL_SENSITIVITY = f"{_TRUSTED}/spatial_sensitivity_v1.json"
POST_OUTCOME_QC_GATE = f"{_TRUSTED}/outcome_qc_gate_v1.json"
POST_TEMPORAL_COVERAGE_AUDIT = f"{_TRUSTED}/temporal_coverage_audit_v1.json"
STAGE09_RECEIPT = "outputs/models/route_a_stage09_completion.json"
STAGE09B_RECEIPT = "outputs/models/route_a_stage09b_completion.json"
STAGE16_RECEIPT = "outputs/models/route_a_stage16_completion.json"
STAGE25_RECEIPT = "outputs/models/route_a_stage25_completion.json"

# Stage-22 adaptive conformal (scripts/22_adaptive_conformal.py).
STAGE22_ROW_TABLE = "outputs/tables/aci_coverage.csv"
STAGE22_REPORT = "outputs/reports/adaptive_conformal.md"

# Stage-13b/13c spatial-partition evidence.  These are DEVELOPMENT-period
# artifacts (2019-2020) and are the only source of whole-region-holdout results
# that will ever exist for this submission: the one-time opening emits no
# held-region artifact.  They may bind Figure 3 and nothing else.
STAGE13C_TABLE = "outputs/tables/region_transfer.csv"
STAGE13C_REPORT = "outputs/reports/region_transfer.md"
TRANSFER_ARMS_REPORT = "outputs/reports/tuurt.md"
LSTM_BASELINE_REPORT = "outputs/reports/lstm_baseline.md"

STATION_REGISTRY = "data_usgs/station_registry_v1.csv"

SI06_RECEIPT = "paper/si/SI06_formal_five_rows_RECEIPT.md"
SI07_RECEIPT = "paper/si/SI07_all_model_scores_RECEIPT.md"
SI08_RECEIPT = "paper/si/SI08_probability_metrics_RECEIPT.md"
SI10_RECEIPT = "paper/si/SI10_temporal_coverage_RECEIPT.md"
SI11_RECEIPT = "paper/si/SI11_spatial_sensitivity_RECEIPT.md"
SI14_RECEIPT = "paper/si/SI14_missingness_failures_RECEIPT.md"
ERRATUM_CONTRACT = "protocols/route_a_probability_metric_erratum_v1.json"
CONFIRMATORY_PROTOCOL = "protocols/route_a_confirmatory_v1.json"

PRIMARY_MODELS = (
    "persistence",
    "damped_persistence",
    "climatology",
    "lightgbm",
    "global_lstm",
    "thermoroute",
)
HORIZONS = (1, 3, 7)

# Stage-22 conformal methods, frozen order as emitted by the Stage-22 report.
CONFORMAL_METHODS = (
    "split_cqr",
    "block_max_cqr_7_retained_rows",
    "aci_gamma_0p005",
    "aci_gamma_0p02",
    "aci_gamma_0p05",
)
CONFORMAL_SLICES = ("overall", "warm_train_q90_tail", "lead_1d", "lead_3d", "lead_7d")

# Development-period boundary; anything at or after this date is target period.
PRIMARY_TARGET_START = "2021-01-01"
STAGE22_OBSERVED_SPAN = ("2019-01-01", "2020-12-24")
# The 2019-2020 development-evaluation partition, which is what the Stage-13b/13c
# transfer arms score on.
DEVELOPMENT_EVALUATION_SPAN = ("2019-01-01", "2020-12-31")

# Spatial arms of Figure 3, in the frozen order the transfer-arms report emits.
TRANSFER_ARMS = (
    "temporal_development",
    "random_held_site_warm_start",
    "held_region_gauged_transfer",
)
REGION_TRANSFER_FOLD_SIZES = (30, 30, 31, 29)

TEMPLATE_ONLY_MARKER = "TEMPLATE_ONLY"


class Severity:
    """Whether a dependency blocks rendering or only annotates provenance."""

    REQUIRED = "required"
    QUALIFIER = "qualifier"


class EvidencePeriod:
    DEVELOPMENT = "development_2019_2020"
    TARGET = "target_2021_onward"
    STRUCTURAL = "pre_structural"


class Stage19Role:
    NONE = "none"
    SUBSTITUTED = "substituted_stage22"
    DROPPED = "dropped_no_substitute"


class PostGateNotPassed(RuntimeError):
    """Raised before any figure object or artifact may be created."""


class PanelBuilderNotImplemented(RuntimeError):
    """Raised when a POST-authorized panel lacks its concrete builder."""


class ManifestError(RuntimeError):
    """Raised when the in-module figure manifest is internally inconsistent."""


@dataclass(frozen=True)
class Roots:
    """Two independent roots.

    ``repo`` holds source-controlled inputs (the spec, ``protocols/``, the SI
    receipt documents).  ``evidence`` holds computed artifacts (``outputs/``).
    They differ whenever the compute chain runs in a sibling worktree, which is
    the case for this submission: the Stage-09/16/25 receipts and the Stage-22
    tables live in ``thermoroute-water-temperature-multicore``, not here.
    """

    repo: Path
    evidence: Path

    def for_kind(self, root_id: str) -> Path:
        return self.repo if root_id == "repo" else self.evidence


def resolve_roots(explicit: str | None = None) -> Roots:
    # HERE = paper/agu_submission/figures; parents[2] of that directory is the
    # repository root.
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    raw = explicit or os.environ.get(EVIDENCE_ROOT_ENV)
    evidence = Path(raw).expanduser().resolve() if raw else repo
    return Roots(repo=repo, evidence=evidence)


@dataclass(frozen=True)
class Dependency:
    authority_id: str
    kind: str  # "glob" | "file"
    path: str
    root: str = "evidence"  # "evidence" | "repo"
    severity: str = Severity.REQUIRED
    note: str = ""

    def resolve(self, roots: Roots) -> tuple[bool, str]:
        base = roots.for_kind(self.root)
        if self.kind == "glob":
            matches = sorted(base.glob(self.path))
            if len(matches) == 1:
                return True, str(matches[0].relative_to(base))
            return False, f"{len(matches)} matches for {self.path}"
        if self.kind == "file":
            return (base / self.path).is_file(), self.path
        raise ManifestError(f"unknown dependency kind: {self.kind!r}")


@dataclass(frozen=True)
class PanelSpec:
    panel_id: str
    plot_type: str
    contract: str
    stage19_role: str = Stage19Role.NONE
    substitution_note: str = ""
    # Optional, and normally left empty so the panel inherits its figure's
    # period.  Declaring one is how a panel author states an intention
    # explicitly; validate_manifest then refuses it if it disagrees with the
    # figure.  This is the machine enforcement of "one figure never mixes two
    # evidence periods" at panel granularity.
    evidence_period: str = ""


@dataclass(frozen=True)
class DroppedPanel:
    """A panel removed from the figure set, with the reason on the record."""

    panel_id: str
    former_content: str
    reason: str
    searched_substitutes: str


@dataclass(frozen=True)
class RenderProfile:
    width_mm: float
    max_height_mm: float = MAX_HEIGHT_MM
    formats: tuple[str, ...] = VECTOR_FORMATS
    min_pt: float = MIN_ABSOLUTE_PT
    body_pt: float = MIN_BODY_PT
    embed_fonts: bool = True
    colour_redundant_encoding: bool = True

    def validate(self) -> None:
        if self.width_mm not in (SINGLE_COLUMN_MM, FULL_WIDTH_MM):
            raise ManifestError(
                f"width {self.width_mm} mm is neither the AGU single-column "
                f"{SINGLE_COLUMN_MM} mm nor the full-width {FULL_WIDTH_MM} mm target"
            )
        if self.max_height_mm > MAX_HEIGHT_MM:
            raise ManifestError(f"height {self.max_height_mm} mm exceeds AGU maximum")
        if self.min_pt < MIN_ABSOLUTE_PT:
            raise ManifestError(f"minimum type size {self.min_pt} pt is below floor")
        if not set(self.formats) & set(VECTOR_FORMATS):
            raise ManifestError("line art must ship at least one vector format")
        if not self.embed_fonts:
            raise ManifestError("fonts must be embedded")
        if not self.colour_redundant_encoding:
            raise ManifestError("no result may be encoded by colour alone")


@dataclass(frozen=True)
class FigureSpec:
    figure_id: str
    stem: str
    spec_anchor: str
    panels: tuple[PanelSpec, ...]
    required_value_ids: tuple[str, ...]
    dependencies: tuple[Dependency, ...]
    prohibited_semantics: tuple[str, ...]
    render_profile: RenderProfile
    evidence_period: str = EvidencePeriod.TARGET
    shares_value_ids_with: tuple[str, ...] = ()
    dropped_panels: tuple[DroppedPanel, ...] = ()
    provenance_qualifiers: tuple[str, ...] = ()
    scope_band_value_id: str = ""
    # Reported by --status for any development-period figure so the render
    # receipt records which span the evidence actually covers.  The defaults
    # reproduce the Stage-22 wording that figS9 has always emitted.
    evidence_span: tuple[str, str] = STAGE22_OBSERVED_SPAN
    evidence_span_label: str = "Stage-22 span"


# Namespaces a figure may legitimately share value IDs with, each with the
# evidence period its values carry.  Enforced by validate_manifest so
# shares_value_ids_with cannot silently rot and, more importantly, so a
# target-period figure cannot quietly reuse a development-period value ID.
NAMESPACE_EVIDENCE_PERIOD: dict[str, str] = {
    "fig01": EvidencePeriod.STRUCTURAL,
    "fig02": EvidencePeriod.TARGET,
    "fig03": EvidencePeriod.DEVELOPMENT,
    "fig04": EvidencePeriod.TARGET,
    "figS1": EvidencePeriod.STRUCTURAL,
    "figS2": EvidencePeriod.STRUCTURAL,
    "figS3": EvidencePeriod.STRUCTURAL,
    "figS4": EvidencePeriod.TARGET,
    "figS5": EvidencePeriod.TARGET,
    "figS6": EvidencePeriod.TARGET,
    "figS7": EvidencePeriod.TARGET,
    "figS8": EvidencePeriod.TARGET,
    "figS9": EvidencePeriod.DEVELOPMENT,
    "figS10": EvidencePeriod.TARGET,
    "table_t2": EvidencePeriod.TARGET,
    "si06": EvidencePeriod.TARGET,
    "si07": EvidencePeriod.TARGET,
    "si08": EvidencePeriod.TARGET,
    "si09": EvidencePeriod.DEVELOPMENT,
    "si10": EvidencePeriod.TARGET,
    "si11": EvidencePeriod.TARGET,
    "si12": EvidencePeriod.TARGET,
    "si13": EvidencePeriod.TARGET,
    "si14": EvidencePeriod.TARGET,
    "stage22_conformal": EvidencePeriod.DEVELOPMENT,
}
KNOWN_VALUE_ID_NAMESPACES = frozenset(NAMESPACE_EVIDENCE_PERIOD)


FIGURES: tuple[FigureSpec, ...] = (
    FigureSpec(
        figure_id="fig02",
        stem="fig02_point_performance",
        spec_anchor="FIGURE_REDRAW_SPEC.md#figure-2",
        evidence_period=EvidencePeriod.TARGET,
        render_profile=RenderProfile(width_mm=FULL_WIDTH_MM, max_height_mm=156.0),
        panels=(
            PanelSpec("a", "reference ladder",
                      "the SAME ThermoRoute predictions scored against every "
                      "reference the trusted scorer emits (persistence, damped "
                      "persistence, climatology, global LightGBM, global LSTM); "
                      "one dimensionless skill axis per lead, labelled 'positive "
                      "favours the candidate'; no ΔRMSE value on this axis; the "
                      "Stage-09b plain causal TCN is development-only and is NOT "
                      "a rung",
                      evidence_period=EvidencePeriod.TARGET),
            PanelSpec("b", "small-multiple station RMSE points/ECDF, h=1 d",
                      "six primary models on one declared all-model exact-common-key set; retained station counts visible"),
            PanelSpec("c", "small-multiple station RMSE points/ECDF, h=3 d",
                      "same key set and denominators as (b)"),
            PanelSpec("d", "small-multiple station RMSE points/ECDF, h=7 d",
                      "same key set and denominators as (b)"),
            PanelSpec("e", "registered five-row forest",
                      "unweighted median station-level TR-minus-ref RMSE + whole-HUC2 bootstrap CI; 0.00 C line for damped rows; +0.05 C ceiling for LightGBM row; status/station/cluster counts beside rows; no stars"),
        ),
        dropped_panels=(
            DroppedPanel(
                panel_id="a_development_ladder",
                former_content="the reference ladder as proposed by "
                               "docs/PAPER_BENCHMARK_RESTRUCTURE.md section 7: a "
                               "DEVELOPMENT-period panel (+0.251 vs persistence, "
                               "+0.038 vs damped at 7 d) carrying a scope band "
                               "inside this otherwise target-period figure",
                reason="refused: one figure never mixes two evidence periods. A "
                       "reader comparing a rung of panel (a) with a station in "
                       "panel (b) would be comparing 2019-2020 with 2021-2023",
                searched_substitutes="the ladder is rebuilt at target period from "
                                     "trusted/temporal_predictions_v1.parquet over "
                                     "the scorer's own reference models; the "
                                     "development-period ladder is fig03 panel (a)",
            ),
        ),
        required_value_ids=(
            "ladder_mark.reference_model_id", "ladder_mark.horizon",
            "ladder_mark.skill", "ladder_mark.candidate_station_median_rmse",
            "ladder_mark.reference_station_median_rmse",
            "ladder_mark.common_key_digest", "ladder_mark.station_count",
            "ladder_mark.sign_convention",
            "all_model_mark.model_id", "all_model_mark.horizon",
            "all_model_mark.site_id", "all_model_mark.common_key_digest",
            "all_model_mark.paired_key_count", "all_model_mark.station_rmse",
            "all_model_mark.reportability",
            "forest_mark.test_id", "forest_mark.candidate_ref_ids",
            "forest_mark.horizon", "forest_mark.margin", "forest_mark.status",
            "forest_mark.effect", "forest_mark.ci_low", "forest_mark.ci_high",
            "forest_mark.station_count", "forest_mark.cluster_count",
            "forest_mark.gate_verdict",
            "t2_companion.win_rate", "t2_companion.raw_p",
            "t2_companion.holm_p", "t2_companion.margin_checks",
        ),
        dependencies=(
            Dependency("opening_receipt", "glob", OPENING_RECEIPT_GLOB),
            Dependency("stage09_receipt", "file", STAGE09_RECEIPT),
            Dependency("si06_formal_rows", "file", SI06_RECEIPT, root="repo"),
            Dependency("si07_all_model_scores", "file", SI07_RECEIPT, root="repo"),
            Dependency("confirmatory_protocol", "file", CONFIRMATORY_PROTOCOL,
                       root="repo"),
            # Fail-closed until the one-time opening materializes them.
            Dependency("post_temporal_predictions", "glob",
                       POST_TEMPORAL_PREDICTIONS),
            Dependency("post_availability_registry", "glob",
                       POST_AVAILABILITY_REGISTRY,
                       note="exact-key / reportability denominators"),
            Dependency("post_statistics", "glob", POST_STATISTICS,
                       note="the five registered formal rows; shared with T2"),
        ),
        prohibited_semantics=(
            "significance coloring or stars", "wins/beats/non-inferior/equivalent/parity",
            "national claim", "row-wise pooled uncertainty as station uncertainty",
            "hidden non-estimable rows", "development-cache substitution",
            "a development-period rung on the target-period ladder",
            "a dimensionless skill value and a degrees-C ΔRMSE on one axis",
            "axis limits chosen after inspecting favourable values",
        ),
        shares_value_ids_with=("table_t2",),
        scope_band_value_id="fig02.scope.fixed_cohort_descriptive",
    ),
    # ------------------------------------------------------------------
    # Figure 3 -- REASSIGNED 2026-08-06 to the spatial-partition contrast.
    #
    # This slot previously held the marginal-interval / event-probability
    # figure.  The benchmark restructure needs a main figure for "random
    # held-site versus whole-region holdout", and that content had no main-text
    # slot at all.  The interval and probability panels are not lost: the
    # coverage-width plane becomes fig04 panel (c) and the event-score and
    # reliability panels become figS5 panels (b) and (c).
    #
    # This figure is DEVELOPMENT-period, and permanently so.  The one-time
    # opening produces no held-region artifact -- the confirmatory protocol
    # registers a temporal cohort and a site-ID-disjoint external cohort, and
    # docs/R13_POSTOPEN_TABLE_RENDERER.md section 6 records that Table 4.6's
    # held-region fragment renders as NOT_EMITTED_BY_THE_ONE_TIME_OPENING.
    # There will never be a target-period counterpart to bind, so the figure
    # declares EvidencePeriod.DEVELOPMENT, renders a mandatory in-panel scope
    # band, and forbids numerical comparison with any target-period figure.
    #
    # It is nevertheless gated on the verified opening receipt exactly like
    # every other POST figure.  That gate is not what supplies its numbers; it
    # is what proves the submission is past the one-shot boundary, so that a
    # development display cannot be published as a stand-in for a target-period
    # result nobody attempted.
    # ------------------------------------------------------------------
    FigureSpec(
        figure_id="fig03",
        stem="fig03_spatial_partition_transfer",
        spec_anchor="FIGURE_REDRAW_SPEC.md#figure-3",
        evidence_period=EvidencePeriod.DEVELOPMENT,
        evidence_span=DEVELOPMENT_EVALUATION_SPAN,
        evidence_span_label="Stage-13c development-evaluation span",
        render_profile=RenderProfile(width_mm=FULL_WIDTH_MM, max_height_mm=182.0),
        panels=(
            PanelSpec("a", "three-arm skill slope plot",
                      "median station skill vs persistence for the temporal, "
                      "random held-site warm-start, and held-region gauged-transfer "
                      "arms at each lead, with the vs-damped values on a paired "
                      "secondary panel; dimensionless axes labelled 'positive "
                      "favours the candidate'; the three arms are three different "
                      "key sets and are never pooled into one mark",
                      evidence_period=EvidencePeriod.DEVELOPMENT),
            PanelSpec("b", "fold geometry map",
                      "the 15 HUC2 groups packed into four folds of "
                      f"{list(REGION_TRANSFER_FOLD_SIZES)} stations on the same "
                      "base as Figure 1(a); one colour AND one hatch per fold; "
                      "held-out stations outlined; reconciles to 120 stations",
                      evidence_period=EvidencePeriod.DEVELOPMENT),
            PanelSpec("c", "distance-association scatter",
                      "per-station held-region skill against distance to the "
                      "nearest training gauge, 289 km mean marked, random "
                      "held-site arm overplotted muted at its own distances; "
                      "association only -- no fitted line, no correlation "
                      "coefficient, no causal verb; the statement is in-panel",
                      evidence_period=EvidencePeriod.DEVELOPMENT),
            PanelSpec("d", "held-region ranking and paired effects",
                      "held-region station-median RMSE for ThermoRoute, global "
                      "LightGBM and the global LSTM by lead, with paired ΔRMSE and "
                      "whole-HUC2 intervals beneath, in degrees C labelled "
                      "'negative favours the candidate'",
                      evidence_period=EvidencePeriod.DEVELOPMENT),
        ),
        required_value_ids=(
            "arm.arm_id", "arm.reference_model_id", "arm.horizon",
            "arm.median_station_skill", "arm.station_count",
            "arm.cluster_count", "arm.exact_key_digest", "arm.sign_convention",
            "fold.fold_id", "fold.huc2_unit", "fold.site_id",
            "fold.coordinates", "fold.held_out_flag", "fold.station_count",
            "distance.site_id", "distance.nearest_training_gauge_km",
            "distance.held_region_skill", "distance.arm_id",
            "distance.mean_reference_km", "distance.association_only_status",
            "ranking.model_id", "ranking.horizon",
            "ranking.station_median_rmse_c", "ranking.paired_delta_rmse_c",
            "ranking.ci_low_c", "ranking.ci_high_c", "ranking.win_rate",
            "ranking.station_count",
            "provenance.evidence_period", "provenance.target_start",
            "provenance.source_digest",
        ),
        dependencies=(
            # Still fail-closed on the opening receipt: a development-period main
            # figure may only be published once the one-shot boundary is past.
            Dependency("opening_receipt", "glob", OPENING_RECEIPT_GLOB),
            Dependency("stage13c_region_transfer_table", "file", STAGE13C_TABLE,
                       note="fold geometry and per-fold rows"),
            Dependency("stage13c_region_transfer_report", "file", STAGE13C_REPORT,
                       note="held-region arm results and the 289 km mean"),
            Dependency("transfer_arms_report", "file", TRANSFER_ARMS_REPORT,
                       note="temporal / random-held-site / held-region arms "
                            "against both references"),
            Dependency("station_registry", "file", STATION_REGISTRY,
                       root="repo",
                       note="fold map base; shared with Figure 1(a)"),
            Dependency("confirmatory_protocol", "file", CONFIRMATORY_PROTOCOL,
                       root="repo"),
            Dependency("lstm_baseline_report", "file", LSTM_BASELINE_REPORT,
                       severity=Severity.QUALIFIER,
                       note="global-LSTM held-region rows for panel (d)"),
        ),
        prohibited_semantics=(
            "any target-period reading of this figure",
            "ungauged prediction",
            "river-network or hydraulic transfer",
            "HUC2 as an independent river-network component",
            "national inference",
            "a causal reading of the distance panel",
            "superiority/non-inferiority/equivalence/parity",
            "numerical comparison against Figure 2, Figure 4, or any "
            "target-period SI figure as if the two were one cohort",
            "a held-region arm described as an evaluation-period result",
        ),
        # fig01 is PRE-structural: registry coordinates and fold geometry are
        # period-neutral facts, so sharing those IDs is legitimate.  No
        # target-period namespace appears here.
        shares_value_ids_with=("fig01",),
        provenance_qualifiers=(
            "Panels bind Stage-13b/13c development-period evidence "
            "(2019-01-01..2020-12-31) on the 120-site cohort, 15 HUC2 groups, "
            "four folds of [30, 30, 31, 29] stations, leads {1,3,7}. The "
            "confirmatory target period starts 2021-01-01; no panel in this "
            "figure is a target-period result and no value here may be compared "
            "numerically with Figure 2, Figure 4, or Figures S4-S8 and S10.",
            "The one-time opening emits NO held-region artifact: the "
            "leave-one-HUC2-region-out arm is not in the confirmatory model "
            "registry and Table 4.6 renders it as "
            "NOT_EMITTED_BY_THE_ONE_TIME_OPENING "
            "(docs/R13_POSTOPEN_TABLE_RENDERER.md section 6). This figure is "
            "therefore permanently development-period; running a target-period "
            "regional holdout would be a protocol amendment, not a figure "
            "change.",
        ),
        scope_band_value_id="fig03.scope.development_period_not_confirmation",
    ),
    # ------------------------------------------------------------------
    # Figure 4 -- REASSIGNED 2026-08-06 to regional/seasonal heterogeneity plus
    # the price of a calibrated interval.
    #
    # Its former panels are all placed, none dropped: the architecture-control
    # matrix becomes figS10, the attrition waterfall becomes figS6 panel (a),
    # and the external history-dependent arm becomes figS8 panel (b).  Its
    # temporal-sensitivity panel stays here as panel (b), joined by the
    # aggregate per-HUC2 heterogeneity that figS7 expands and by the
    # coverage-width plane of the former fig03 panel (a).
    #
    # The restructure brief proposed a fourth panel, "what calibration costs",
    # carrying the Stage-22 split-CQR / block-max / delayed-ACI contrast with a
    # scope band.  That is 2019-2020 evidence and is refused here for the same
    # reason fig02 refuses a development rung; it lives in figS9, which already
    # declares EvidencePeriod.DEVELOPMENT and renders the band.
    # ------------------------------------------------------------------
    FigureSpec(
        figure_id="fig04",
        stem="fig04_heterogeneity_and_interval_cost",
        spec_anchor="FIGURE_REDRAW_SPEC.md#figure-4",
        evidence_period=EvidencePeriod.TARGET,
        render_profile=RenderProfile(width_mm=FULL_WIDTH_MM, max_height_mm=182.0),
        panels=(
            PanelSpec("a", "per-HUC2 heterogeneity dot plot",
                      "per-HUC2 median skill against persistence AND against "
                      "damped persistence at each lead, ordered by region, with "
                      "the pooled median and the region-weighted mean as named "
                      "reference lines and station count encoded by marker size; "
                      "per_huc[].huc2 is a cluster label ('HUC2:01' / "
                      "'UNMAPPED:<site_no>'), never a bare two-digit code, and "
                      "UNMAPPED units render as themselves",
                      evidence_period=EvidencePeriod.TARGET),
            PanelSpec("b", "temporal sensitivity dot plot",
                      "eight frozen candidates (12 year-by-season equal weight, 3 leave-one-year, 4 leave-one-season); deterministic worst marked; formal effect retained as distinct reference"),
            PanelSpec("c", "coverage-width plane",
                      "station-balanced empirical 90% marginal coverage vs mean "
                      "interval width per eligible learned model x horizon; 0.90 "
                      "nominal reference named, not a formal coverage test; "
                      "point-only models bind NOT_AVAILABLE, never invented heads; "
                      "coverage is never shown without the width that buys it",
                      evidence_period=EvidencePeriod.TARGET),
        ),
        dropped_panels=(
            DroppedPanel(
                panel_id="d_calibration_cost",
                former_content="'what calibration costs' as proposed by "
                               "docs/PAPER_BENCHMARK_RESTRUCTURE.md section 7: the "
                               "Stage-22 split-CQR, block-maximum and delayed-ACI "
                               "variants on the same coverage-width plane, "
                               "DEVELOPMENT-period, with a scope band, inside this "
                               "otherwise target-period figure",
                reason="refused: one figure never mixes two evidence periods. A "
                       "reader would compare a 2019-2020 coverage number with a "
                       "2021-2023 coverage number on one plane",
                searched_substitutes="figS9 panels (a) and (b) already carry "
                                     "exactly this evidence with "
                                     "EvidencePeriod.DEVELOPMENT and a mandatory "
                                     "in-panel scope band",
            ),
        ),
        required_value_ids=(
            "per_huc.comparison_id", "per_huc.cluster_label",
            "per_huc.reference_model_id", "per_huc.horizon",
            "per_huc.effect", "per_huc.station_count",
            "per_huc.interval_status", "per_huc.pooled_median",
            "per_huc.region_weighted_mean", "per_huc.registry_binding",
            "temporal.test_id", "temporal.sensitivity_id_order",
            "temporal.candidate_definition", "temporal.effect",
            "temporal.support", "temporal.formal_effect_reference",
            "temporal.deterministic_worst_flag",
            "cohort", "model", "horizon", "coverage_90",
            "mean_interval_width_c", "station_weight_audit",
            "forecast_count_pre_reportability",
            "forecast_count_post_reportability", "site_count", "min_targets",
            "undefined_reason", "probability_pipeline_source",
        ),
        dependencies=(
            Dependency("opening_receipt", "glob", OPENING_RECEIPT_GLOB),
            Dependency("stage09_receipt", "file", STAGE09_RECEIPT),
            Dependency("si08_probability_metrics", "file", SI08_RECEIPT,
                       root="repo"),
            Dependency("si10_temporal_coverage", "file", SI10_RECEIPT, root="repo"),
            Dependency("si11_spatial_sensitivity", "file", SI11_RECEIPT,
                       root="repo"),
            Dependency("erratum_contract", "file", ERRATUM_CONTRACT, root="repo"),
            Dependency("confirmatory_protocol", "file", CONFIRMATORY_PROTOCOL,
                       root="repo"),
            Dependency("post_spatial_sensitivity", "glob",
                       POST_SPATIAL_SENSITIVITY,
                       note="panel (a) per-HUC2 effects"),
            Dependency("post_temporal_coverage_audit", "glob",
                       POST_TEMPORAL_COVERAGE_AUDIT,
                       note="panel (b) eight frozen candidates"),
            Dependency("post_probabilistic_evaluation", "glob",
                       POST_PROBABILISTIC_EVALUATION,
                       note="panel (c) trusted-scorer target-period coverage and "
                            "width; independent of the withheld Stage-19 script"),
            Dependency("post_temporal_predictions", "glob",
                       POST_TEMPORAL_PREDICTIONS),
            Dependency("post_availability_registry", "glob",
                       POST_AVAILABILITY_REGISTRY,
                       note="reportability denominators"),
            # Development-only by construction; it may never fill a panel here.
            Dependency("stage09b_receipt", "file", STAGE09B_RECEIPT,
                       severity=Severity.QUALIFIER,
                       note="development-only controls; SI09 tabulation only, "
                            "never substituted into any panel of this figure"),
        ),
        prohibited_semantics=(
            "conditional coverage", "distribution-free target-period guarantee",
            "CRPS", "operational forecast reliability", "economic value",
            "coverage without width", "all calendar days",
            "year/season stability", "missing-at-random",
            "HUC2 as an independent river-network component", "national claim",
            "rescue of failed/unfavourable formal row",
            "quantile crossing as the Stage-19 cause",
            "development-period conformal numbers shown as target-period results",
            "development control substituted for a POST target sensitivity",
        ),
        shares_value_ids_with=("figS5", "figS6", "figS7", "table_t2"),
        provenance_qualifiers=(
            "The Stage-19 development-period probabilistic script is withheld for "
            "this submission (zero-width nominal intervals; 0 strict quantile "
            "crossings in 26,993,675 member-level rows). Panel (c) does NOT "
            "depend on it: the target-period metrics are computed by the trusted "
            "scorer inside the one-time opening "
            "(src/thermoroute/opening.py:8932-8948).",
            "Pre-opening guard: opening.py:7093-7094 enforces a strict q05 < q95 "
            "on member-averaged nominal heads and aborts the whole opening on "
            "violation. On the development panel, member averaging clears every "
            "affected LightGBM key (5 members); the only survivors are 12 "
            "single-member LightGBM-perstation keys, and that model is not in "
            "PRIMARY_MODELS or the confirmatory protocol. Re-run the check on the "
            "target-period predictions before executing the one-time opening.",
        ),
        scope_band_value_id="fig04.scope.fixed_cohort_descriptive",
    ),
)


def validate_manifest(figures: tuple[FigureSpec, ...] = FIGURES) -> None:
    """Fail loudly if the in-module contract is internally inconsistent.

    This is what makes ``shares_value_ids_with``, ``render_profile``, and
    ``dropped_panels`` load-bearing rather than decorative.
    """
    seen_ids: set[str] = set()
    for figure in figures:
        if figure.figure_id in seen_ids:
            raise ManifestError(f"duplicate figure_id {figure.figure_id}")
        seen_ids.add(figure.figure_id)

        figure.render_profile.validate()

        panel_ids = [p.panel_id for p in figure.panels]
        if len(panel_ids) != len(set(panel_ids)):
            raise ManifestError(f"{figure.figure_id}: duplicate panel_id")
        if len(figure.required_value_ids) != len(set(figure.required_value_ids)):
            raise ManifestError(f"{figure.figure_id}: duplicate required value_id")

        dropped_ids = {d.panel_id for d in figure.dropped_panels}
        if dropped_ids & set(panel_ids):
            raise ManifestError(
                f"{figure.figure_id}: panel is both live and dropped")

        for peer in figure.shares_value_ids_with:
            if peer not in KNOWN_VALUE_ID_NAMESPACES:
                raise ManifestError(
                    f"{figure.figure_id}: shares_value_ids_with unknown namespace "
                    f"{peer!r}")
            if peer == figure.figure_id:
                raise ManifestError(f"{figure.figure_id}: shares with itself")
            # A shared value ID means the same number, unit and rounding in two
            # places.  Two evidence periods can never satisfy that, so a
            # cross-period share is a manifest error rather than a render-time
            # surprise.  PRE-structural namespaces are period-neutral by
            # construction (spec section 2.1 layer 1) and are always allowed.
            peer_period = NAMESPACE_EVIDENCE_PERIOD[peer]
            if (
                peer_period != figure.evidence_period
                and peer_period != EvidencePeriod.STRUCTURAL
            ):
                raise ManifestError(
                    f"{figure.figure_id} ({figure.evidence_period}) shares value "
                    f"IDs with {peer!r} ({peer_period}); one figure never mixes "
                    "two evidence periods, and neither does one value ID")

        # A figure bound to development evidence must carry an in-panel scope
        # band and must say so in its qualifiers.
        if figure.evidence_period == EvidencePeriod.DEVELOPMENT:
            if not figure.scope_band_value_id:
                raise ManifestError(
                    f"{figure.figure_id}: development-period figure needs a "
                    "scope_band_value_id")
            if not figure.provenance_qualifiers:
                raise ManifestError(
                    f"{figure.figure_id}: development-period figure needs "
                    "provenance qualifiers")

        for panel in figure.panels:
            # A substituted panel must explain itself.
            if panel.stage19_role == Stage19Role.SUBSTITUTED and not panel.substitution_note:
                raise ManifestError(
                    f"{figure.figure_id}.{panel.panel_id}: substituted panel "
                    "needs a substitution_note")
            # One figure never mixes two evidence periods.  A panel may leave
            # its period empty and inherit the figure's; declaring a different
            # one is the error this check exists to make unrepresentable.
            if panel.evidence_period and panel.evidence_period != figure.evidence_period:
                raise ManifestError(
                    f"{figure.figure_id}.{panel.panel_id}: panel declares "
                    f"{panel.evidence_period} inside a "
                    f"{figure.evidence_period} figure; one figure never mixes "
                    "two evidence periods")


def spec_state(figure_id: str, roots: Roots) -> str:
    """Read the figure's current state from the authoritative spec."""
    spec_path = roots.repo / "paper" / "FIGURE_REDRAW_SPEC.md"
    text = spec_path.read_text(encoding="utf-8")
    number = figure_id[len("fig"):].lstrip("0") or "0"
    pattern = re.compile(
        rf"^## Figure {re.escape(number)} —.*?^\*\*State:\*\* `([A-Z0-9_]+)`",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    if match is None:
        raise PostGateNotPassed(
            f"Could not locate a state line for Figure {number} in {spec_path.name}"
        )
    return match.group(1)


def assert_evidence_period_labelled(figure: FigureSpec) -> None:
    """A development-period figure may never present itself as a POST result."""
    if figure.evidence_period != EvidencePeriod.DEVELOPMENT:
        return
    if not figure.scope_band_value_id:
        raise PostGateNotPassed(
            f"{figure.figure_id}: development-period evidence requires a bound "
            "in-panel scope band; refusing to render an unlabelled figure."
        )


@dataclass
class GateRow:
    authority_id: str
    ok: bool
    detail: str
    severity: str = Severity.REQUIRED

    @property
    def blocking(self) -> bool:
        return self.severity == Severity.REQUIRED and not self.ok


def gate_report(figure: FigureSpec, roots: Roots) -> list[GateRow]:
    rows: list[GateRow] = []

    state = spec_state(figure.figure_id, roots)
    rows.append(GateRow(
        "spec_state",
        TEMPLATE_ONLY_MARKER not in state,
        f"spec state = {state}",
    ))

    for dep in figure.dependencies:
        ok, detail = dep.resolve(roots)
        if dep.severity == Severity.QUALIFIER and dep.note:
            detail = f"{detail} [{dep.note}]"
        rows.append(GateRow(dep.authority_id, ok, detail, dep.severity))

    # Recorded provenance, never blocking.
    for qualifier in figure.provenance_qualifiers:
        rows.append(GateRow("provenance_qualifier", True, qualifier,
                            Severity.QUALIFIER))
    for dropped in figure.dropped_panels:
        rows.append(GateRow(
            "dropped_panel", True,
            f"{dropped.panel_id}: {dropped.reason}", Severity.QUALIFIER))

    if figure.evidence_period == EvidencePeriod.DEVELOPMENT:
        rows.append(GateRow(
            "evidence_period", True,
            f"{figure.evidence_period} ({figure.evidence_span_label} "
            f"{figure.evidence_span[0]}..{figure.evidence_span[1]}; confirmatory "
            f"target starts {PRIMARY_TARGET_START}); in-panel scope band "
            f"{figure.scope_band_value_id} is mandatory",
            Severity.QUALIFIER))

    return rows


def assert_post_authority(figure: FigureSpec, roots: Roots) -> None:
    blocking = [row for row in gate_report(figure, roots) if row.blocking]
    if blocking:
        rendered = "; ".join(f"{r.authority_id}: {r.detail}" for r in blocking)
        raise PostGateNotPassed(
            f"{figure.figure_id} POST gate not passed ({rendered}). "
            "No artifact, axes, or placeholder may be rendered."
        )
    assert_evidence_period_labelled(figure)


class ValueBinder:
    """Mark-level binder mirroring render_pre_supporting_figures.py semantics.

    ``value`` and ``cell`` reproduce the PRE closure exactly.  ``mark`` is the
    enforcement point: routing every visible cell through ``cell`` is what makes
    "every visible mark binds a declared value_id" true rather than aspirational.
    """

    def __init__(self, figure: FigureSpec) -> None:
        self.figure = figure
        self.values: dict[str, dict[str, object]] = {}
        self.marks: list[dict[str, object]] = []

    def value(
        self,
        value_id: str,
        raw_value: object,
        unit: str,
        evidence_role: str,
        source_pointer: str,
        derivation: str,
        rounding: str,
    ) -> str:
        record = {
            "value": raw_value,
            "unit": unit,
            "evidence_role": evidence_role,
            "source_pointer": source_pointer,
            "derivation": derivation,
            "rounding": rounding,
        }
        if value_id in self.values and self.values[value_id] != record:
            raise RuntimeError(f"Conflicting binder value: {value_id}")
        self.values[value_id] = record
        return value_id

    def cell(self, value_id: str) -> dict[str, str]:
        if value_id not in self.values:
            raise RuntimeError(f"Mark references undeclared value: {value_id}")
        return {"value_id": value_id}

    def mark(self, mark_id: str, panel_id: str, **cells: str) -> dict[str, object]:
        known = {p.panel_id for p in self.figure.panels}
        if panel_id not in known:
            raise RuntimeError(
                f"{self.figure.figure_id}: mark {mark_id} names unknown panel "
                f"{panel_id}")
        record = {
            "mark_id": mark_id,
            "panel_id": panel_id,
            "cells": {f: self.cell(v) for f, v in cells.items()},
        }
        self.marks.append(record)
        return record

    def require_contract_ids(self) -> None:
        """Every required value ID must be declared before render authority."""
        undeclared = [vid for vid in self.figure.required_value_ids
                      if not any(vid == v or v.startswith(vid + ".")
                                 for v in self.values)]
        if undeclared:
            raise RuntimeError(
                f"{self.figure.figure_id}: {len(undeclared)} required value IDs "
                f"undeclared, e.g. {undeclared[:5]}"
            )

    def require_scope_band(self) -> None:
        band = self.figure.scope_band_value_id
        if band and band not in self.values:
            raise RuntimeError(
                f"{self.figure.figure_id}: scope band {band} is not bound; a "
                "development-period or scope-limited figure may not render "
                "without its in-panel boundary statement."
            )

    def cross_check_shared_value_ids(
        self, peer_values: dict[str, dict[str, object]]
    ) -> None:
        """Shared IDs must agree in value AND rounding across figures/tables."""
        for value_id, record in self.values.items():
            peer = peer_values.get(value_id)
            if peer is None:
                continue
            if peer != record:
                raise RuntimeError(
                    f"{self.figure.figure_id}: shared value {value_id} disagrees "
                    "with its peer binding (value, unit, or rounding differs)"
                )


PANEL_BUILDERS: dict[str, object] = {}


def render_figure(figure: FigureSpec, roots: Roots) -> Path:
    assert_post_authority(figure, roots)
    # matplotlib is imported only after the gate passes; the skeleton never
    # creates a Figure object pre-POST.
    binder = ValueBinder(figure)
    for panel in figure.panels:  # pragma: no cover - POST work
        builder = PANEL_BUILDERS.get(f"{figure.figure_id}.{panel.panel_id}")
        if builder is None:
            raise PanelBuilderNotImplemented(
                f"{figure.figure_id} panel ({panel.panel_id}) builder not "
                "implemented; POST work must bind evidence and draw it."
            )
        builder(binder, panel, roots)  # type: ignore[operator]
    binder.require_contract_ids()
    binder.require_scope_band()
    raise PanelBuilderNotImplemented(
        f"{figure.figure_id}: skeleton carries contracts only"
    )


def print_status(roots: Roots) -> int:
    print(f"# evidence root : {roots.evidence}")
    print(f"# repo root     : {roots.repo}")
    print(f"# schema        : {FIGURE_SCHEMA_VERSION}")
    print(f"{'figure':8} {'authority':28} {'sev':10} {'ok':4} detail")
    overall = 0
    for figure in FIGURES:
        for row in gate_report(figure, roots):
            flag = "yes" if row.ok else "no"
            print(f"{figure.figure_id:8} {row.authority_id:28} "
                  f"{row.severity:10} {flag:4} {row.detail}")
            if row.blocking:
                overall = 2
    print("# exit 2 = at least one REQUIRED gate is unmet (expected pre-POST); "
          "qualifier rows never block")
    return overall


# ===========================================================================
# Conventional 2021-2023 render path.
#
# The sealed-confirmatory POST gate above (opening receipt under
# outputs/confirmatory/route_a_*/, trusted/* artifacts, Stage receipts) never
# resolves for the conventional paper: that apparatus was not built and the
# conventional holdout scorer (src/thermoroute/conventional_score.py) writes a
# different, pooled long-form table instead.  This block is the surgical
# conventional reframe requested for the feat/conventional branch: it reads the
# real holdout numbers from outputs/conventional/holdout_metrics_2021_2023.csv
# and holdout_summary_2021_2023.json and renders fig02-04 + S4-S9 into this
# directory, reusing the AGU render profile and the Okabe-Ito semantic palette
# defined above.  No sealed-confirmatory artifact is required and none is read.
#
# Data reality (matches paper section 4.6 and docs/B3_NUMBER_PLACEHOLDER_MAP.md):
# the CSV is POOLED per model x horizon (RMSE/MAE/BIAS/SKILL_PERSISTENCE/
# SKILL_CLIMATOLOGY/N_SKILL) with a string "pooled" horizon; there is no
# station-level, no HUC2, and no probability/interval dimension.  Figures whose
# required source does not exist (fig03/fig04/S5/S7/S9) are rendered as explicit
# "not reported" notices that cite the paper disposition, never as empty axes or
# invented coordinates.
# ===========================================================================
import json as _json
import textwrap as _tw

import numpy as _np
import pandas as _pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as _plt
from matplotlib.patches import Patch as _Patch
from matplotlib.patches import Rectangle as _Rectangle
from matplotlib.text import Text as _Text
from matplotlib.transforms import Bbox as _Bbox

# ---------------------------------------------------------------------------
# Shared WRR style.  paper/figstyle.py is the single authority for typeface,
# type sizes, palette, figure widths, colorbar placement, and the text-overlap
# gate that every save must pass.  It lives one level above this package, so the
# paper directory goes on sys.path before the import.
# ---------------------------------------------------------------------------
_PAPER_DIR = str(Path(__file__).resolve().parents[2])
if _PAPER_DIR not in sys.path:
    sys.path.insert(0, _PAPER_DIR)
import figstyle  # noqa: E402

# The width constants above are declared before this import can run, so assert
# they still agree with the shared authority rather than letting them drift.
if (SINGLE_COLUMN_MM, FULL_WIDTH_MM, MAX_HEIGHT_MM) != (
    figstyle.SINGLE_MM, figstyle.FULL_MM, figstyle.MAX_HEIGHT_MM
):
    raise ValueError(
        "figure widths disagree with paper/figstyle.py: "
        f"{(SINGLE_COLUMN_MM, FULL_WIDTH_MM, MAX_HEIGHT_MM)} vs "
        f"{(figstyle.SINGLE_MM, figstyle.FULL_MM, figstyle.MAX_HEIGHT_MM)}"
    )

CONV_CSV = "outputs/conventional/holdout_metrics_2021_2023.csv"
CONV_JSON = "outputs/conventional/holdout_summary_2021_2023.json"
CONV_HORIZONS = (1, 3, 7)

PRIMARY_MODELS_CONV = (
    "Persistence", "DampedPersistence", "Climatology",
    "LightGBM", "LSTM", "ThermoRoute",
)
LADDER_REFERENCES = (
    "Persistence", "DampedPersistence", "Climatology", "LightGBM", "LSTM",
)
ABLATION_MODELS_CONV = (
    "TR-noTCN", "TR-noMoE", "TR-noRouter",
    "TR-noDynamicPrior", "TR-fixedKappa", "TR-unbounded",
)
EXT_MODELS_CONV = ("LightGBM-ext", "LSTM-ext", "ThermoRoute-ext")
BASELINE_MODELS_CONV = (
    "Persistence", "DampedPersistence", "DampedPriorOnly", "Climatology",
)
ALL_MODELS_CONV = BASELINE_MODELS_CONV + ("LightGBM", "LSTM", "ThermoRoute") \
    + ABLATION_MODELS_CONV + EXT_MODELS_CONV

# (colour, marker, linestyle).  Colour comes from figstyle.SERIES so a model
# keeps one identity across every figure in the submission; the ablation and
# external variants borrow from the same Wong palette.  Every series also varies
# marker shape and/or dash, so no result is encoded by colour alone and each
# panel survives a grayscale check.
MODEL_STYLE_CONV = {
    "ThermoRoute":       (figstyle.SERIES["ThermoRoute"], "o", "-"),
    "LightGBM":          (figstyle.SERIES["LightGBM"], "s", "-"),
    "LSTM":              (figstyle.SERIES["LSTM"], "^", "-"),
    "Persistence":       (figstyle.SERIES["Persistence"], "D", (0, (4, 2))),
    "DampedPersistence": (figstyle.SERIES["DampedPersistence"], "v", (0, (4, 2))),
    "Climatology":       (figstyle.SERIES["Climatology"], "P", (0, (1, 2))),
    "DampedPriorOnly":   (figstyle.SERIES["DampedPersistence"], "v", (0, (1, 2))),
    "TR-noTCN":          (figstyle.WONG["green"], "o", (0, (5, 2))),
    "TR-noMoE":          (figstyle.WONG["sky"], "s", (0, (5, 2))),
    "TR-noRouter":       (figstyle.WONG["blue"], "^", (0, (5, 2))),
    "TR-noDynamicPrior": (figstyle.WONG["purple"], "D", (0, (5, 2))),
    "TR-fixedKappa":     (figstyle.WONG["orange"], "v", (0, (5, 2))),
    "TR-unbounded":      (figstyle.WONG["vermillion"], "P", (0, (5, 2))),
    "ThermoRoute-ext":   (figstyle.SERIES["ThermoRoute"], "o", (0, (2, 2))),
    "LSTM-ext":          (figstyle.SERIES["LSTM"], "^", (0, (2, 2))),
    "LightGBM-ext":      (figstyle.SERIES["LightGBM"], "s", (0, (2, 2))),
}
MODEL_LABEL_CONV = {
    "ThermoRoute": "ThermoRoute", "LightGBM": "LightGBM", "LSTM": "global LSTM",
    "Persistence": "Persistence", "DampedPersistence": "Damped persistence",
    "Climatology": "Climatology", "DampedPriorOnly": "Damped anchor only",
    "TR-noTCN": "TR-noTCN", "TR-noMoE": "TR-noMoE", "TR-noRouter": "TR-noRouter",
    "TR-noDynamicPrior": "TR-noDynamicPrior", "TR-fixedKappa": "TR-fixedKappa",
    "TR-unbounded": "TR-unbounded", "ThermoRoute-ext": "ThermoRoute-ext",
    "LSTM-ext": "LSTM-ext", "LightGBM-ext": "LightGBM-ext",
}
# Short forms used for direct end-of-line labelling, where a legend would
# otherwise have to sit on top of the data.  Expanded in the figure footnote.
MODEL_TAG_CONV = {
    "ThermoRoute": "ThermoRoute", "LightGBM": "LightGBM", "LSTM": "LSTM",
    "Persistence": "Persistence", "DampedPersistence": "Damped",
    "Climatology": "Climatology",
}


def _conventional_rcparams() -> None:
    """Apply the shared submission style.

    Everything this used to set by hand -- typeface chain, the 8 pt / 7.5 pt
    ladder, ``pdf.fonttype 42``, the spines -- now comes from ``paper/figstyle``,
    which also turns constrained layout on.  Constrained layout is the actual
    fix for most of the collisions this renderer used to produce: it reserves
    space for titles, colorbars and figure-level text instead of leaving them to
    hand-tuned ``left/right/top/bottom`` fractions that drift whenever a label
    changes length.
    """
    figstyle.use()


def load_conventional(roots: Roots):
    """Return (metrics, summary) where metrics[model][horizon][metric]=float.

    Horizon keys are python ints 1/3/7 and the string "pooled".  Baseline skill
    rows (Persistence/DampedPersistence/Climatology) are derived from pooled
    RMSE exactly as paper section 4.6 and B3 section 5.2 specify, because the
    CSV emits no SKILL_* rows for the three pure baselines.
    """
    csv_path = roots.repo / CONV_CSV
    json_path = roots.repo / CONV_JSON
    if not csv_path.is_file():
        raise FileNotFoundError(f"conventional metrics CSV not found: {csv_path}")
    if not json_path.is_file():
        raise FileNotFoundError(f"conventional summary JSON not found: {json_path}")
    df = _pd.read_csv(csv_path)
    summary = _json.loads(json_path.read_text(encoding="utf-8"))
    metrics: dict[str, dict] = {}
    for _, row in df.iterrows():
        raw_h = row["horizon"]
        horizon = "pooled" if str(raw_h) == "pooled" else int(raw_h)
        model = str(row["model"])
        metrics.setdefault(model, {}).setdefault(horizon, {})[row["metric"]] = \
            float(row["value"])
        metrics[model][horizon]["n"] = int(row["n"])
    for h in CONV_HORIZONS:
        rmse = {m: metrics[m][h]["RMSE"] for m in
                ("Persistence", "DampedPersistence", "Climatology")}
        metrics["Persistence"][h]["SKILL_PERSISTENCE"] = 0.0
        metrics["Persistence"][h]["SKILL_CLIMATOLOGY"] = \
            1.0 - rmse["Persistence"] / rmse["Climatology"]
        metrics["DampedPersistence"][h]["SKILL_PERSISTENCE"] = \
            1.0 - rmse["DampedPersistence"] / rmse["Persistence"]
        metrics["DampedPersistence"][h]["SKILL_CLIMATOLOGY"] = \
            1.0 - rmse["DampedPersistence"] / rmse["Climatology"]
        metrics["Climatology"][h]["SKILL_PERSISTENCE"] = \
            1.0 - rmse["Climatology"] / rmse["Persistence"]
        metrics["Climatology"][h]["SKILL_CLIMATOLOGY"] = 0.0
    station_metrics = _pd.read_csv(roots.repo / "outputs/conventional/station_metrics_2021_2023.csv")
    sm: dict[str, dict] = {}
    for (model, h), g in station_metrics.groupby(["model", "horizon"]):
        sm.setdefault(str(model), {})[int(h)] = {
            "RMSE": float(g["rmse"].median()),
            "MAE": float(g["mae"].median()),
            "n_stations": int(g["site_id"].nunique()),
        }
    skill_table = _pd.read_csv(roots.repo / "outputs/conventional/skill_table_2021_2023.csv")
    for _, row in skill_table.iterrows():
        if row["baseline"] == "Persistence":
            sm.setdefault(str(row["model"]), {})[int(row["horizon"])][
                "SKILL_PERSISTENCE"] = float(row["skill"])
    metrics["station_median"] = sm
    return metrics, summary


def _skill_vs(metrics, candidate, reference, horizon):
    sm = metrics["station_median"]
    if candidate in sm and reference in sm and horizon in sm[candidate] \
            and horizon in sm[reference]:
        return 1.0 - sm[candidate][horizon]["RMSE"] / \
            sm[reference][horizon]["RMSE"]
    return 1.0 - metrics[candidate][horizon]["RMSE"] / \
        metrics[reference][horizon]["RMSE"]


def _sm_rmse(metrics, model, horizon):
    sm = metrics["station_median"]
    return sm.get(model, {}).get(horizon, {}).get("RMSE", _np.nan)


RENDER_LOG: list[dict[str, object]] = []


def _min_type_size(fig, floor_pt: float = MIN_ABSOLUTE_PT) -> float:
    """Return the smallest type size actually drawn, refusing below the floor.

    ``figstyle.check_overlaps`` only sees text that belongs to an axes or to the
    figure; this walks every ``Text`` in the tree, so legend entries and the
    ruled pseudo-tables are covered too.  Nothing may be smaller than 7.5 pt at
    final size, and final size is authored size because every figure is saved at
    its exact millimetre width with no tight bounding box.
    """
    fig.canvas.draw()
    sizes = []
    offenders = []
    for artist in fig.findobj(_Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        size = float(artist.get_fontsize())
        sizes.append(size)
        if size < floor_pt - 1e-6:
            offenders.append((artist.get_text()[:32], round(size, 2)))
    if offenders:
        raise ValueError(
            f"type below the {floor_pt} pt floor: {sorted(set(offenders))[:8]}")
    return min(sizes) if sizes else float("nan")


def _assert_on_page(fig, tolerance_px: float = 1.0) -> None:
    """Refuse a figure whose text runs off the page.

    ``figstyle.check_overlaps`` catches text landing on text; it cannot catch
    text landing on nothing because it left the canvas.  A ``supxlabel`` is the
    usual culprit -- constrained layout reserves a strip for it but never
    shortens it -- and a note that runs out of both margins reads as a broken
    figure, not a busy one.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    page = fig.bbox
    spills = []
    # Deliberately not ``findobj``: matplotlib keeps tick-label artists for
    # locator ticks that fall outside the view limits, parked off-canvas and
    # never drawn.  They are not a spill.  The artists that can genuinely leave
    # the page are the figure-level and axes-level ones placed by hand.
    artists = list(fig.texts)
    for ax in fig.axes:
        artists.extend(ax.texts)
        artists.extend([ax.title, ax.xaxis.label, ax.yaxis.label])
        legend = ax.get_legend()
        if legend is not None:
            artists.extend(legend.get_texts())
    for artist in artists:
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        box = artist.get_window_extent(renderer)
        if (box.x0 < page.x0 - tolerance_px or box.x1 > page.x1 + tolerance_px
                or box.y0 < page.y0 - tolerance_px
                or box.y1 > page.y1 + tolerance_px):
            spills.append(artist.get_text()[:32])
    if spills:
        raise ValueError(f"text runs off the page: {sorted(set(spills))[:6]}")


def _save_conventional(fig, stem, out_dir: Path) -> None:
    """Save at the exact authored size, refusing on collisions or small type.

    No ``bbox_inches='tight'``: a tight bbox grows the figure past the AGU
    target whenever a label spills, and the production rescale back to the
    column width then drops 7.5 pt type below the floor.  Constrained layout has
    already fitted everything inside the authored box, so the box is what ships.
    ``figstyle.save`` raises if any two text artists still overlap.
    """
    width_mm, height_mm = (v * 25.4 for v in fig.get_size_inches())
    smallest = _min_type_size(fig)
    _assert_on_page(fig)
    figstyle.save(fig, stem, out_dir)
    _plt.close(fig)
    RENDER_LOG.append({"stem": stem, "width_mm": round(width_mm, 1),
                       "height_mm": round(height_mm, 1),
                       "min_pt": round(smallest, 2)})
    print(f"wrote {stem}.pdf/.png/.svg  "
          f"{width_mm:.0f} x {height_mm:.0f} mm  min type {smallest:.1f} pt",
          flush=True)


def _style(model):
    return MODEL_STYLE_CONV.get(model, (PALETTE["NEUTRAL_INK"], "o", "-"))


def _panel_label(ax, text):
    """Bold panel label and short title in one string, left-aligned above the axes.

    Kept short on purpose: a long ``loc='left'`` title overruns its own axes and
    lands on the neighbouring panel's title, which is not something constrained
    layout can undo.  Anything longer belongs in the axis label or the caption.
    """
    ax.set_title(text, loc="left", fontweight="bold", fontsize=PANEL_LABEL_PT_RANGE[0])


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=figstyle.GRID, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)


def _spread(values, height, lo, hi):
    """Push 1-D label centres apart so no two boxes of ``height`` can overlap."""
    out = list(values)
    prev = None
    for i in sorted(range(len(out)), key=lambda k: out[k]):
        out[i] = out[i] if prev is None else max(out[i], prev + height)
        prev = out[i]
    overflow = (max(out) + height / 2.0) - hi
    if overflow > 0:
        out = [v - overflow for v in out]
    return out


def _direct_labels(ax, entries, x_end, *, fontsize=MIN_ABSOLUTE_PT,
                   pad_pt=4.0, gap_pt=1.5):
    """Label each series at its right-hand end instead of drawing a legend.

    A legend inside a small panel either covers data or is squeezed below the
    type floor; both were true of the previous draft.  Direct labels remove the
    choice.  Room for them is taken out of the x range after measuring the
    widest label, and the label centres are pushed apart in display space, so
    coincident series (LightGBM and the LSTM are equal to three decimals at
    7 days) still read as two labels rather than one smear.

    ``entries`` is ``[(y_value, text, colour), ...]`` at ``x_end``.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [ax.text(x_end, y, label, color=colour, fontsize=fontsize,
                     va="center", ha="left", clip_on=False)
             for y, label, colour in entries]
    fig.canvas.draw()
    boxes = [t.get_window_extent(renderer) for t in texts]
    axis_box = ax.get_window_extent(renderer)
    pad_px = pad_pt * fig.dpi / 72.0
    needed = max(b.width for b in boxes) + pad_px
    if axis_box.width - needed <= 0.35 * axis_box.width:
        raise ValueError("direct labels would take more than the panel can give")
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x0 + (x1 - x0) * axis_box.width / (axis_box.width - needed))
    fig.canvas.draw()

    axis_box = ax.get_window_extent(renderer)
    step = max(b.height for b in boxes) + gap_pt * fig.dpi / 72.0
    centres = _spread([ax.transData.transform((x_end, y))[1] for y, _, _ in entries],
                      step, axis_box.y0, axis_box.y1)
    inverse = ax.transData.inverted()
    span = ax.get_xlim()[1] - ax.get_xlim()[0]
    pad_data = pad_px * span / axis_box.width
    for text, centre, (y_true, _, colour) in zip(texts, centres, entries):
        y_label = inverse.transform((0, centre))[1]
        text.set_position((x_end + pad_data, y_label))
        # A label pushed clear of its own series needs a leader, or the reader
        # attributes it to whichever line it drifted next to.
        if abs(centre - ax.transData.transform((x_end, y_true))[1]) > 2.0:
            ax.plot([x_end + 0.18 * pad_data, x_end + 0.82 * pad_data],
                    [y_true, y_label], color=colour, lw=0.4, clip_on=False,
                    solid_capstyle="butt", zorder=1)
    return texts


def _text_table(ax, headers, rows, col_x, *, fontsize=MIN_ABSOLUTE_PT,
                top=0.95, bottom=0.06, stripe=True):
    """Draw a small ruled table out of ordinary text artists.

    ``matplotlib.table.Table`` neither wraps nor reports its cell text to
    ``figstyle.check_overlaps``, so a cell that outgrows its column overflows
    silently.  Building the table from text artists puts every string under the
    same collision gate as the rest of the figure, and lets a cell carry
    pre-wrapped multi-line content whose height the row honours.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # Clear the ticks before switching the axis off.  ``axis("off")`` only stops
    # the axis being drawn; the tick-label artists survive with visible=True and
    # are still counted by the overlap gate, which then reports a collision
    # against invisible "0.0 ... 1.0" labels nobody can see.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    row_lines = [max(cell.count("\n") + 1 for cell in row) for row in rows]
    units = 1.0 + 0.45 + sum(row_lines) + 0.35 * max(len(rows) - 1, 0)
    step = (top - bottom) / units
    cursor = top
    for x, head in zip(col_x, headers):
        ax.text(x, cursor, head, transform=ax.transAxes, va="top", ha="left",
                fontsize=fontsize, fontweight="bold", color=PALETTE["NEUTRAL_INK"])
    cursor -= step
    ax.axhline(cursor + 0.22 * step, xmin=col_x[0], xmax=0.99,
               color=figstyle.RULE, linewidth=0.6)
    cursor -= 0.45 * step
    for index, (row, lines) in enumerate(zip(rows, row_lines)):
        if stripe and index % 2 == 0:
            ax.add_patch(_Rectangle(
                (col_x[0] - 0.02, cursor - lines * step), 1.01 - col_x[0],
                lines * step, transform=ax.transAxes, zorder=0,
                facecolor=PALETTE["NA_FILL"], edgecolor="none"))
        for x, cell in zip(col_x, row):
            ax.text(x, cursor - 0.12 * step, cell, transform=ax.transAxes,
                    va="top", ha="left", fontsize=fontsize,
                    color=PALETTE["NEUTRAL_INK"], linespacing=1.35)
        cursor -= lines * step + 0.35 * step
    return cursor


def _fig_note(fig, text, *, fontsize=MIN_ABSOLUTE_PT, start=96):
    """Attach a figure-level note, wrapped to the figure's own width.

    ``fig.supxlabel`` neither wraps nor is shrunk by constrained layout, so a
    long note simply runs off both edges of the page -- which is what the first
    draft of this renderer shipped.  Wrap, measure, and narrow until it fits.
    """
    words = " ".join(text.split())
    note = None
    for columns in range(start, 39, -6):
        note = fig.supxlabel("\n".join(_tw.wrap(words, columns)),
                             fontsize=fontsize, color=figstyle.MUTED,
                             linespacing=1.35)
        fig.canvas.draw()
        width = note.get_window_extent(fig.canvas.get_renderer()).width
        if width <= fig.bbox.width - 0.02 * fig.bbox.width:
            return note
    return note


def _short_reason(raw, width):
    """Verbatim failure reason, wrapped, with the URL query truncated.

    The NWIS failures carry the whole request URL.  Keeping the human-readable
    prefix verbatim and marking the cut with an ellipsis says exactly as much as
    the record supports without pretending the string was shorter than it is.
    """
    text = (raw or "—").split(": http")[0].strip()
    if text != (raw or "—").strip():
        text = text + " …"
    return "\n".join(_tw.wrap(text, width=width)) or "—"


def render_fig02(metrics, summary, out_dir):
    """fig02 -- baseline choice sets the reported gain (target, pooled).

    Layout: (a) and (b) share the top row, (c) takes the whole second row so its
    five row labels fit on one line each.  Both line panels are labelled at the
    right-hand end of each series rather than by a legend, which is what used to
    sit over the data in (a) and (b).
    """
    fig = _plt.figure(figsize=figstyle.figsize(FULL_WIDTH_MM, 112.0))
    # Two subfigures, not one 2x2 gridspec.  In a shared gridspec the wide row
    # labels of panel (c) set the left margin of the whole first column, which
    # left panel (a) with about half the plotting width of panel (b).  A
    # subfigure gets its own margins.
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.0, 0.82])
    gs = top.add_gridspec(1, 2)

    ax_a = top.add_subplot(gs[0, 0])
    ladder = []
    for ref in LADDER_REFERENCES:
        colour, marker, ls = _style(ref)
        ys = [_skill_vs(metrics, "ThermoRoute", ref, h) for h in CONV_HORIZONS]
        ax_a.plot(CONV_HORIZONS, ys, ls=ls, marker=marker, color=colour,
                  ms=4.0, lw=1.2, markeredgecolor="white", markeredgewidth=0.4)
        ladder.append((float(ys[-1]), MODEL_TAG_CONV[ref], colour))
    ax_a.axhline(0.0, color=PALETTE["NEUTRAL_INK"], lw=0.7)
    ax_a.set_xticks(CONV_HORIZONS)
    ax_a.set_xlim(0.5, 7.5)
    ax_a.set_ylim(-0.16, 0.74)
    ax_a.set_xlabel("forecast horizon (d)")
    ax_a.set_ylabel("skill vs reference\n(+ favours ThermoRoute)")
    _panel_label(ax_a, "(a) Reference ladder")
    _grid(ax_a)

    ax_b = top.add_subplot(gs[0, 1])
    primary = []
    for model in PRIMARY_MODELS_CONV:
        colour, marker, ls = _style(model)
        ys = [_sm_rmse(metrics, model, h) for h in CONV_HORIZONS]
        ax_b.plot(CONV_HORIZONS, ys, ls=ls, marker=marker, color=colour,
                  ms=4.0, lw=1.2, markeredgecolor="white", markeredgewidth=0.4)
        primary.append((float(ys[-1]), MODEL_TAG_CONV[model], colour))
    ax_b.set_xticks(CONV_HORIZONS)
    ax_b.set_xlim(0.5, 7.5)
    ax_b.set_ylim(0.45, 2.42)
    ax_b.set_xlabel("forecast horizon (d)")
    ax_b.set_ylabel("station-median RMSE (\u00b0C)")
    _panel_label(ax_b, "(b) Six primary models")
    _grid(ax_b)

    ax_c = bottom.subplots()
    rows = [
        ("Damped persistence", 1, "DampedPersistence"),
        ("Damped persistence", 3, "DampedPersistence"),
        ("Damped persistence", 7, "DampedPersistence"),
        ("LightGBM", 3, "LightGBM"),
        ("LightGBM", 7, "LightGBM"),
    ]
    values = _np.array([_sm_rmse(metrics, "ThermoRoute", h) - _sm_rmse(metrics, ref, h)
                        for _, h, ref in rows])
    colours = [_style(ref)[0] for _, _, ref in rows]
    y = _np.arange(len(rows))[::-1]
    ax_c.hlines(y, 0, values, color=colours, lw=1.6)
    ax_c.scatter(values, y, color=colours, s=26, zorder=3,
                 edgecolor="white", linewidth=0.5)
    ax_c.axvline(0.0, color=PALETTE["NEUTRAL_INK"], lw=0.7)
    ax_c.axvline(0.05, color=figstyle.WONG["vermillion"], lw=0.7, ls=(0, (3, 2)))
    ax_c.text(0.05, 0.5, "+0.05 \u00b0C ceiling", transform=ax_c.get_xaxis_transform(),
              color=figstyle.WONG["vermillion"], fontsize=MIN_ABSOLUTE_PT,
              rotation=90, ha="right", va="center")
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([f"vs {name}, {h} d" for name, h, _ in rows])
    ax_c.set_ylim(-0.7, len(rows) - 0.3)
    ax_c.set_xlim(-0.20, 0.12)
    ax_c.set_xlabel("\u0394RMSE (\u00b0C; \u2212 favours ThermoRoute)")
    _panel_label(ax_c, "(c) Paired \u0394RMSE, ThermoRoute minus reference")
    _grid(ax_c, axis="x")

    fig.suptitle("Held-out 2021\u20132023 \u00b7 station-median metrics over the common forecast keys")
    _fig_note(fig,
              "In (a) and (b) \u201cDamped\u201d is damped persistence. \u0394RMSE in (c) is "
              "the unweighted station median of paired per-station differences "
              "over the 116 reportable stations; clustered intervals, win "
              "rates, and Holm-adjusted p-values are in Table 4.6.")

    # Direct labels last, and only once the suptitle and the two-line figure
    # footnote are in place: both change how much height constrained layout
    # gives the axes, and these labels are positioned in display space.
    _direct_labels(ax_a, ladder, CONV_HORIZONS[-1])
    _direct_labels(ax_b, primary, CONV_HORIZONS[-1])
    _save_conventional(fig, "fig02_point_performance", out_dir)


def render_figS4(metrics, summary, out_dir):
    """figS4 -- point-performance heterogeneity (all models x horizons,
    unweighted station medians over the 116 reportable stations).

    The collision this figure used to ship was structural, not cosmetic:
    ``fig.colorbar(im0, ax=axes[0])`` puts the bar immediately right of the left
    heatmap, which is exactly where the right heatmap's row labels are, so the
    bar's tick numbers landed on "Damped persistence", "Climatology",
    "ThermoRoute", "TR-noDynamicPrior" and "TR-fixedKappa".  Routing both bars
    through ``figstyle.colorbar`` makes them part of the constrained layout,
    which reserves the column instead of overprinting it.  The two heatmaps also
    share one y axis now: sixteen model names printed twice cost more width than
    the panels could spare.
    """
    sm = metrics["station_median"]
    models = list(ALL_MODELS_CONV)
    rmse = _np.array([[sm.get(m, {}).get(h, {}).get("RMSE", _np.nan)
                       for h in CONV_HORIZONS] for m in models])
    skill = _np.array([[sm.get(m, {}).get(h, {}).get(
        "SKILL_PERSISTENCE", _np.nan) for h in CONV_HORIZONS] for m in models])

    fig, axes = _plt.subplots(
        1, 2, sharey=True, figsize=figstyle.figsize(FULL_WIDTH_MM, 118.0))

    im0 = axes[0].imshow(rmse, cmap=figstyle.SEQUENTIAL, aspect="auto")
    vlim = max(abs(_np.nanmin(skill)), abs(_np.nanmax(skill)))
    im1 = axes[1].imshow(skill, cmap=figstyle.DIVERGING, aspect="auto",
                         vmin=-vlim, vmax=vlim)
    _panel_label(axes[0], "(a) Pooled RMSE (\u00b0C)")
    _panel_label(axes[1], "(b) Skill vs persistence")

    axes[0].set_yticks(range(len(models)))
    axes[0].set_yticklabels([MODEL_LABEL_CONV[m] for m in models])
    for ax, mat, fmt, span in ((axes[0], rmse, "{:.2f}",
                                (float(rmse.min()), float(rmse.max()))),
                               (axes[1], skill, "{:+.2f}", (-vlim, vlim))):
        ax.set_xticks(range(len(CONV_HORIZONS)))
        ax.set_xticklabels([f"{h} d" for h in CONV_HORIZONS])
        ax.set_xlabel("forecast horizon")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        low, high = span
        for (i, j), v in _ndenumerate(mat):
            shade = (v - low) / (high - low) if high > low else 0.5
            dark = shade > 0.66 or (span[0] < 0 and shade < 0.2)
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    fontsize=MIN_ABSOLUTE_PT,
                    color="white" if dark else PALETTE["NEUTRAL_INK"])
    for image, ax, label in ((im0, axes[0], "pooled RMSE (\u00b0C)"),
                             (im1, axes[1], "skill (+ favours the model)")):
        bar = figstyle.colorbar(fig, image, ax, label=label)
        bar.ax.tick_params(labelsize=MIN_ABSOLUTE_PT)
        bar.set_label(label, fontsize=MIN_ABSOLUTE_PT)

    fig.suptitle("Sixteen models \u00d7 horizon, unweighted station medians over the held-out 2021\u20132023 window")
    _fig_note(fig, "Station-median RMSE and skill versus persistence over the "
                   "116 reportable stations on the common forecast keys; the "
                   "external-history variants (ext) use the pooled-training "
                   "cohort.")
    _save_conventional(fig, "figS4_point_heterogeneity", out_dir)


def _ndenumerate(arr):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            yield (i, j), arr[i, j]


def render_figS6(metrics, summary, out_dir):
    """figS6 -- temporal opportunity, missingness, and attrition.

    Panel (b) was a 6.3 pt monospace block whose reason strings were cut at 29
    characters mid-word; it is now a ruled two-column list at the 7.5 pt floor
    with the reasons wrapped rather than clipped.
    """
    failures = summary.get("station_failures", [])
    n_panel = summary.get("n_stations_panel", 120)
    n_scored = n_panel - len(failures)
    sm = metrics.get("station_median", {})
    n_reportable = sm.get("ThermoRoute", {}).get(1, {}).get("n_stations", n_scored)
    n_win = summary.get("n_windows_temporal")

    fig = _plt.figure(figsize=figstyle.figsize(FULL_WIDTH_MM, 88.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.12])

    ax_a = fig.add_subplot(gs[0, 0])
    stages = ["Panel sites", "Scored sites", "Reportable sites",
              "h = 1 d keys", "h = 3 d keys", "h = 7 d keys"]
    counts = [n_panel, n_scored, n_reportable,
              metrics["ThermoRoute"][1]["n"],
              metrics["ThermoRoute"][3]["n"],
              metrics["ThermoRoute"][7]["n"]]
    colours = [figstyle.WONG["blue"], figstyle.WONG["green"],
               figstyle.WONG["sky"], figstyle.WONG["orange"],
               figstyle.WONG["vermillion"], figstyle.WONG["purple"]]
    bars = ax_a.barh(range(len(stages)), counts, color=colours,
                     edgecolor="white", linewidth=0.5, zorder=2)
    ax_a.set_yticks(range(len(stages)))
    ax_a.set_yticklabels(stages)
    ax_a.invert_yaxis()
    ax_a.set_xscale("log")
    ax_a.set_xlim(50, max(counts) * 9)
    ax_a.set_xlabel("count (log scale)")
    _panel_label(ax_a, f"(a) Attrition: {n_panel} \u2192 {n_scored} \u2192 {n_reportable} sites")
    for bar, count in zip(bars, counts):
        ax_a.text(count * 1.15, bar.get_y() + bar.get_height() / 2,
                  f"{count:,}", va="center", ha="left",
                  fontsize=MIN_ABSOLUTE_PT)
    _grid(ax_a, axis="x")

    ax_b = fig.add_subplot(gs[0, 1])
    _panel_label(ax_b, f"(b) Fetch failures ({len(failures)} of {n_panel})")
    rows = [[str(f.get("site_no", "?")), _short_reason(f.get("nwis"), 30)]
            for f in failures]
    tail = _text_table(ax_b, ["site", "reason for exclusion"], rows,
                       col_x=(0.03, 0.32), top=0.95, bottom=0.14)
    if n_win is not None:
        ax_b.text(0.03, max(tail - 0.03, 0.0),
                  f"temporal windows scored: {n_win:,}",
                  transform=ax_b.transAxes, va="top", ha="left",
                  fontsize=MIN_ABSOLUTE_PT, color=figstyle.MUTED)

    fig.suptitle("Missingness and attrition on the held-out 2021\u20132023 panel")
    _save_conventional(fig, "figS6_attrition_missingness", out_dir)


def render_figS8(metrics, summary, out_dir):
    """figS8 -- external history-dependent arm and failure disposition.

    Two restructures rather than restyles.  The six-entry legend ("LightGBM
    (temporal)", "LightGBM-ext (external)", ...) could not be set at 7.5 pt in a
    half-width panel, so the encoding is factored into its two real dimensions:
    colour names the model, hatch names the cohort, and the key is five short
    entries in reserved headroom.  And the panels are stacked instead of side by
    side, which gives the failure list the width to print each reason on one
    line.
    """
    pairs = [("LightGBM", "LightGBM-ext"), ("LSTM", "LSTM-ext"),
             ("ThermoRoute", "ThermoRoute-ext")]
    failures = summary.get("station_failures", [])

    fig = _plt.figure(figsize=figstyle.figsize(FULL_WIDTH_MM, 108.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.72])

    ax_a = fig.add_subplot(gs[0, 0])
    width = 0.13
    x = _np.arange(len(CONV_HORIZONS))
    for k, (temporal, external) in enumerate(pairs):
        colour = _style(temporal)[0]
        offset = (k - 1) * 2.2 * width
        ax_a.bar(x + offset - width / 2,
                 [metrics[temporal][h]["RMSE"] for h in CONV_HORIZONS],
                 width, color=colour, edgecolor="white", linewidth=0.5, zorder=2)
        ax_a.bar(x + offset + width / 2,
                 [metrics[external][h]["RMSE"] for h in CONV_HORIZONS],
                 width, facecolor="white", hatch="////", edgecolor=colour,
                 linewidth=0.6, zorder=2)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"{h} d" for h in CONV_HORIZONS])
    ax_a.set_xlabel("forecast horizon")
    ax_a.set_ylabel("pooled RMSE (\u00b0C)")
    ax_a.set_ylim(0, 2.75)
    _panel_label(ax_a, "(a) External (site-ID-disjoint) cohort against the temporal cohort")
    _grid(ax_a)
    handles = [_Patch(facecolor=_style(m)[0], edgecolor="white",
                      label=MODEL_LABEL_CONV[m]) for m, _ in pairs]
    handles += [_Patch(facecolor=figstyle.MUTED, edgecolor="white",
                       label="temporal cohort"),
                _Patch(facecolor="white", hatch="////", edgecolor=figstyle.MUTED,
                       label="external cohort")]
    ax_a.legend(handles=handles, loc="upper left", ncol=5,
                fontsize=MIN_ABSOLUTE_PT, columnspacing=1.0, handlelength=1.4,
                handletextpad=0.5)

    ax_b = fig.add_subplot(gs[1, 0])
    _panel_label(ax_b, f"(b) Failure disposition ({len(failures)} sites excluded)")
    rows = [[str(f.get("site_no", "?")), "excluded",
             _short_reason(f.get("nwis"), 70)] for f in failures] \
        or [["\u2014", "\u2014", "no failures recorded"]]
    _text_table(ax_b, ["site", "status", "reason"], rows,
                col_x=(0.01, 0.12, 0.25), top=0.94, bottom=0.05)

    fig.suptitle("External arm and failure disposition, held-out 2021\u20132023")
    _fig_note(fig,
              "The external cohort is site-ID disjoint and history-dependent; "
              "it is not an ungauged-basin arm. The outcome-QC waterfall "
              "(Table 4.10) is not reported for the held-out window.")
    _save_conventional(fig, "figS8_external_arm_failures", out_dir)


NOTICE_WRAP_COLUMNS = 76
NOTICE_HEADLINE_PT = 8.5
NOTICE_FLAG_PT = 8.0


def render_notice(label, stem, title, body_text, out_dir,
                  width_mm=FULL_WIDTH_MM, flag=None):
    """Render an explicit 'not reported' notice; no axes, no invented data.

    These panels are legitimate -- the held-out window really does emit no
    held-region arm and no probability family -- so they have to look
    deliberate.  The previous draft fixed an 85 x 62 mm square, poured
    hand-broken lines into it, and drew the border at a fixed fraction of the
    figure: the prose ran out of the box on both sides and broke mid-sentence,
    inside a mostly empty square.

    Here the order is inverted.  The prose is one string, wrapped to a measured
    column; the figure height is then set from what the wrapped block actually
    measures; and the border is drawn last, around the text's own extent,
    centred on the page.  The box fits the text, never the other way round.
    """
    lines = _tw.wrap(" ".join(body_text.split()), width=NOTICE_WRAP_COLUMNS)
    flag = flag or "NOT REPORTED for the 2021\u20132023 held-out window"
    outer_mm, inset_mm = 3.5, 6.0
    gap_head_mm, gap_flag_mm = 1.6, 3.4

    fig = _plt.figure(figsize=figstyle.figsize(width_mm, 60.0))
    fig.set_layout_engine("none")
    head = fig.text(0.5, 0.9, f"{label} \u2014 {title}", ha="center", va="top",
                    fontsize=NOTICE_HEADLINE_PT, fontweight="bold",
                    color=PALETTE["NEUTRAL_INK"])
    banner = fig.text(0.5, 0.6, flag, ha="center", va="top",
                      fontsize=NOTICE_FLAG_PT, fontweight="bold",
                      color=PALETTE["WARNING_VERMILION"])
    body = fig.text(0.5, 0.3, "\n".join(lines), ha="center", va="top",
                    fontsize=MIN_ABSOLUTE_PT, linespacing=1.5,
                    color=PALETTE["NEUTRAL_INK"])

    # Measure what the three blocks really occupy, then size the page to them.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def mm_of(artist, attribute):
        box = artist.get_window_extent(renderer)
        return getattr(box, attribute) / fig.dpi * 25.4

    head_mm, flag_mm, body_mm = (mm_of(a, "height") for a in (head, banner, body))
    height_mm = (2 * outer_mm + 2 * inset_mm + head_mm + gap_head_mm
                 + flag_mm + gap_flag_mm + body_mm)
    fig.set_size_inches(*figstyle.figsize(width_mm, height_mm))

    cursor = outer_mm + inset_mm
    for artist, block in ((head, head_mm), (banner, flag_mm), (body, body_mm)):
        artist.set_position((0.5, 1.0 - cursor / height_mm))
        cursor += block + (gap_head_mm if artist is head else gap_flag_mm)

    fig.canvas.draw()
    extent = _Bbox.union([a.get_window_extent(renderer)
                          for a in (head, banner, body)])
    extent = extent.transformed(fig.transFigure.inverted())
    pad_x, pad_y = inset_mm / width_mm, inset_mm / height_mm
    half = extent.width / 2.0 + pad_x          # centred on the page, not on the
    fig.add_artist(_Rectangle(                  # widest line's own midpoint
        (0.5 - half, extent.y0 - pad_y), 2 * half, extent.height + 2 * pad_y,
        transform=fig.transFigure, zorder=0, facecolor=PALETTE["NA_FILL"],
        edgecolor=PALETTE["WARNING_VERMILION"], linewidth=1.0))
    _save_conventional(fig, stem, out_dir)


def render_fig03_notice(metrics, summary, out_dir):
    render_notice(
        "Figure 3", "fig03_spatial_partition_transfer",
        "Spatial partition and whole-region transfer",
        "Development-period figure (Stage-13c region-transfer evidence, "
        "2019-01-01 to 2020-12-31). The 2021\u20132023 conventional holdout emits no "
        "held-region arm, so this figure is not rendered from the conventional "
        "metrics CSV. See paper section 4.4 for the development-period "
        "three-arm values.", out_dir,
        flag="NO HELD-REGION ARM EXISTS for the 2021\u20132023 held-out window")


def render_fig04_notice(metrics, summary, out_dir):
    render_notice(
        "Figure 4", "fig04_heterogeneity_and_interval_cost",
        "Regional and seasonal heterogeneity, and interval cost",
        "Panels require a per-HUC2 regional breakdown, the eight "
        "temporal-coverage candidates, and the 90% coverage\u2013width plane. None "
        "was computed for 2021\u20132023: the conventional holdout CSV is pooled "
        "(no HUC2 dimension) and the probability pipeline was not re-run "
        "(paper Table 4.9: not reported). No value is invented.", out_dir)


def render_figS5_notice(metrics, summary, out_dir):
    render_notice(
        "Figure S5", "figS5_probability_reliability",
        "Event score, reliability, and probabilistic diagnostics",
        "Requires the SI08 probability family (coverage, pinball, Brier, log "
        "loss, AUROC/AUPRC, ECE, calibration slope and intercept, "
        "station-balanced reliability bins). Not computed for 2021\u20132023 (paper "
        "Table 4.9: not reported). The development-period probability "
        "diagnostics of paper section 4.5 stand as the only probability "
        "evidence in this paper.", out_dir)


def render_figS7_notice(metrics, summary, out_dir):
    render_notice(
        "Figure S7", "figS7_spatial_leave_huc2",
        "Spatial and leave-HUC2 influence",
        "Requires per-HUC2 effects and leave-one-HUC2 omissions. The "
        "conventional holdout CSV is pooled with no HUC2 dimension, so this "
        "figure is not reported for the held-out window. No value is invented.",
        out_dir)


def render_figS9_notice(metrics, summary, out_dir):
    render_notice(
        "Figure S9", "figS9_conformal_calibration",
        "Development-period conformal calibration sensitivity",
        "Development-period figure (Stage-22 adaptive conformal, 2019-01-01 to "
        "2020-12-24). Not a 2021\u20132023 result and not rendered from the "
        "conventional metrics CSV. See paper section 4.5 for the "
        "development-period split-CQR, block-maximum and delayed-ACI values.",
        out_dir,
        flag="DEVELOPMENT-PERIOD EVIDENCE ONLY \u00b7 not a 2021\u20132023 result")



def render_conventional(roots: Roots, only: str | None = None) -> None:
    _conventional_rcparams()
    out_dir = Path(__file__).resolve().parent
    metrics, summary = load_conventional(roots)
    rendered, skipped = [], []
    for figure_id, builder in CONVENTIONAL_DISPATCH:
        if only is not None and only != figure_id:
            continue
        try:
            builder(metrics, summary, out_dir)
            rendered.append(figure_id)
        except Exception as exc:
            skipped.append((figure_id, repr(exc)))
            print(f"ERROR rendering {figure_id}: {exc!r}", file=sys.stderr)
    print(f"rendered: {rendered}", flush=True)
    print(f"resolved typeface: {figstyle.resolved_font()}", flush=True)
    for entry in RENDER_LOG:
        print(f"  {entry['stem']:38} {entry['width_mm']:>5.1f} x "
              f"{entry['height_mm']:>5.1f} mm   min type {entry['min_pt']:.1f} pt",
              flush=True)
    if skipped:
        print(f"failed: {skipped}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conventional 2021-2023 figure renderer (fig02-04, S4-S9). "
                    "The legacy sealed-confirmatory POST gate is retained for "
                    "--status inspection only.")
    parser.add_argument("--status", action="store_true",
                        help="read-only sealed-confirmatory gate report "
                             "(apparatus legacy; renders nothing)")
    parser.add_argument("--figure", default=None,
                        help="render one figure id (fig02, fig03, fig04, "
                             "figS4..figS9); default renders all nine")
    parser.add_argument("--evidence-root", default=None,
                        help="root holding outputs/ (defaults to "
                             f"${EVIDENCE_ROOT_ENV} or this worktree)")
    args = parser.parse_args()

    roots = resolve_roots(args.evidence_root)

    if args.status:
        validate_manifest()
        sys.exit(print_status(roots))

    render_conventional(roots, only=args.figure)




def render_fig03(metrics, summary, out_dir):
    """fig03 -- matched spatial-transfer experiment on the held-out window.

    (a) station-median RMSE per lead for the random and whole-region arms of
    the matched spatial experiment (station-agnostic LightGBM, 2021-2023 keys)
    with the damped-persistence reference; (b) per-station transfer penalty
    versus nearest-training-gauge distance, with median bins per arm.
    """
    import pandas as _pd
    rt = _pd.read_csv(Path(__file__).resolve().parents[3]
                      / "outputs/conventional/region_transfer_metrics_2021_2023.csv")
    fig = _plt.figure(figsize=figstyle.figsize(FULL_WIDTH_MM, 108.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.18])

    ax_a = fig.add_subplot(gs[0, 0])
    rows = []
    for h in CONV_HORIZONS:
        for arm, label in (("random", "Random held-site"),
                           ("region", "Whole-region")):
            g = rt[(rt.arm == arm) & (rt.horizon == h)]
            rows.append((h, label, float(g.rmse.median()), arm))
    y = _np.arange(len(rows))[::-1]
    for (h, label, v, arm), yi in zip(rows, y):
        colour = figstyle.WONG["blue"] if arm == "random" else figstyle.WONG["orange"]
        ax_a.plot([0.45, v], [yi, yi], color=colour, lw=1.6)
        ax_a.scatter(v, yi, color=colour, s=26, zorder=3,
                     edgecolor="white", linewidth=0.5)
    damped = rt.groupby("horizon")["rmse_damped"].median()
    for h, yi in zip(CONV_HORIZONS, y[::2]):
        ax_a.axvline(damped[h], color=PALETTE["NEUTRAL_INK"], lw=0.6,
                     ls=(0, (2, 2)))
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([f"{label}, {h} d" for h, label, _, _ in rows])
    ax_a.set_xlim(0.45, 1.9)
    ax_a.set_xlabel("station-median RMSE (\u00b0C)")
    _panel_label(ax_a, "(a) Spatial arms, 2021\u20132023")
    _grid(ax_a, axis="x")
    ax_a.text(1.78, len(rows) - 0.4, "damped reference",
              fontsize=MIN_ABSOLUTE_PT, ha="right",
              color=PALETTE["NEUTRAL_INK"])

    ax_b = fig.add_subplot(gs[0, 1])
    for arm, colour, marker in (("random", figstyle.WONG["blue"], "o"),
                                ("region", figstyle.WONG["orange"], "s")):
        g = rt[(rt.arm == arm) & (rt.horizon == 3)]
        g = g.dropna(subset=["nearest_km"])
        ax_b.scatter(g.nearest_km, g.rmse - g.rmse_damped, s=7, color=colour,
                     marker=marker, alpha=0.55, edgecolor="none", label=arm)
        g = g.sort_values("nearest_km")
        b = g.groupby(_pd.qcut(g.nearest_km, 5), group_keys=False).apply(
            lambda x: _pd.Series({"x": x.nearest_km.median(),
                                  "y": (x.rmse - x.rmse_damped).median()}),
            include_groups=False)
        ax_b.plot(b.x, b.y, color=colour, lw=1.8, ls="-",
                  marker="D", ms=3.5, markeredgecolor="white")
    ax_b.axhline(0.0, color=PALETTE["NEUTRAL_INK"], lw=0.7)
    ax_b.set_xscale("log")
    ax_b.set_xticks([10, 30, 100, 300, 1000])
    ax_b.set_xticklabels(["10", "30", "100", "300", "1000"])
    ax_b.set_xlabel("nearest-training-gauge distance (km)")
    ax_b.set_ylabel("\u0394RMSE vs damped persistence, 3 d (\u00b0C)")
    _panel_label(ax_b, "(b) Transfer penalty by distance")
    _grid(ax_b, axis="x")
    ax_b.legend(fontsize=MIN_ABSOLUTE_PT, frameon=False, loc="upper left")

    fig.suptitle("Matched spatial transfer, independent 2021\u20132023 window")
    _fig_note(fig,
              "Station-agnostic LightGBM with frozen per-lead hyperparameters "
              "(one seed); four deterministic whole-HUC2-region folds versus "
              "four balanced random folds; identical preprocessing in both "
              "arms; unweighted station medians over the 118 scored sites. "
              "Median nearest-training-gauge distance: 268 km (region) versus "
              "60 km (random).")
    _save_conventional(fig, "fig03_spatial_partition_transfer", out_dir)


def render_fig04(metrics, summary, out_dir):
    """fig04 -- regional heterogeneity and interval cost (held-out window).

    (a) per-HUC2 seven-day skill against persistence from station-median RMSE
    joined to the stable registry; (b) empirical 90% coverage versus mean
    interval width per model and lead (Table 4.9).
    """
    import pandas as _pd
    root = Path(__file__).resolve().parents[3]
    sm = _pd.read_csv(root / "outputs/conventional/station_metrics_2021_2023.csv")
    reg = _pd.read_csv(root / "data_usgs/station_registry_v1.csv")
    reg["site_no8"] = reg["site_no"].astype(str).str.zfill(8)
    sm["site_id"] = sm["site_id"].astype(str).str.zfill(8)
    sm = sm.merge(reg[["site_no8", "huc2"]], left_on="site_id", right_on="site_no8",
                  how="left")
    prob = _pd.read_csv(root / "outputs/conventional/probability_metrics_2021_2023.csv")

    fig = _plt.figure(figsize=figstyle.figsize(FULL_WIDTH_MM, 118.0))
    gs = fig.add_gridspec(1, 2)

    ax_a = fig.add_subplot(gs[0, 0])
    per_huc = (sm[sm.model == "ThermoRoute"].groupby(["huc2", "horizon"])["rmse"]
               .median().unstack())
    damped = (sm[sm.model == "DampedPersistence"].groupby(["huc2", "horizon"])["rmse"]
              .median().unstack())
    hucs = sorted(per_huc.index, key=lambda h: -per_huc.loc[h, 7])
    skill7 = 1.0 - per_huc.loc[hucs, 7] / damped.loc[hucs, 7]
    y = _np.arange(len(hucs))
    ax_a.barh(y, skill7, color=figstyle.WONG["sky"], edgecolor="white",
              linewidth=0.5, height=0.72, zorder=2)
    ax_a.axvline(0.0, color=PALETTE["NEUTRAL_INK"], lw=0.7)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([f"HUC2:{h}" for h in hucs], fontsize=MIN_ABSOLUTE_PT)
    ax_a.set_xlim(-0.03, 0.31)
    ax_a.set_xlabel("7 d skill vs persistence (station median)")
    _panel_label(ax_a, "(a) Regional heterogeneity, 15 HUC2 groups")
    _grid(ax_a, axis="x")

    ax_b = fig.add_subplot(gs[0, 1])
    for model, colour, marker in (("LightGBM", figstyle.SERIES["LightGBM"], "s"),
                                  ("LSTM", figstyle.SERIES["LSTM"], "^"),
                                  ("ThermoRoute", figstyle.SERIES["ThermoRoute"], "o")):
        g = prob[prob.model == model]
        ax_b.scatter(g.interval_width, g.interval_coverage, s=30, color=colour,
                     marker=marker, zorder=3, edgecolor="white", linewidth=0.5,
                     label=model)

    ax_b.axhline(0.90, color=PALETTE["NEUTRAL_INK"], lw=0.8, ls=(0, (3, 2)))
    ax_b.text(1.0, 0.884, "nominal 90%", fontsize=MIN_ABSOLUTE_PT,
              color=PALETTE["NEUTRAL_INK"], va="top")
    ax_b.set_xlabel("mean interval width (\u00b0C)")
    ax_b.set_ylabel("empirical coverage")
    ax_b.set_ylim(0.88, 0.935)
    _panel_label(ax_b, "(b) Coverage bought with width")
    _grid(ax_b)
    ax_b.legend(fontsize=MIN_ABSOLUTE_PT, frameon=False, loc="lower right")

    fig.suptitle("Regional heterogeneity and interval cost, 2021\u20132023",
                 y=0.99)
    fig.subplots_adjust(top=0.86)
    _fig_note(fig,
              "Coverage at the nominal 90% level from the frozen CQR + Platt "
              "calibration applied identically to the held-out predictions "
              "(Table 4.9); extended scoring in SI08.")
    _save_conventional(fig, "fig04_heterogeneity_and_interval_cost", out_dir)


def render_figS5(metrics, summary, out_dir):
    """figS5 -- probability diagnostics on the held-out window.

    (a) Brier score by model and lead; (b) reliability bins at 1 day with the
    calibration diagonal.
    """
    import pandas as _pd
    root = Path(__file__).resolve().parents[3]
    prob = _pd.read_csv(root / "outputs/conventional/probability_metrics_2021_2023.csv")
    rel = _pd.read_csv(root / "outputs/conventional/reliability_bins_2021_2023.csv")

    fig = _plt.figure(figsize=figstyle.figsize(FULL_WIDTH_MM, 96.0))
    gs = fig.add_gridspec(1, 2)

    ax_a = fig.add_subplot(gs[0, 0])
    models = ["LightGBM", "LSTM", "ThermoRoute"]
    x = _np.arange(len(models))
    width = 0.26
    for j, h in enumerate(CONV_HORIZONS):
        vals = [float(prob[(prob.model == m) & (prob.horizon == h)].brier.iloc[0])
                for m in models]
        ax_a.bar(x + (j - 1) * width, vals, width * 0.92, color=(
            figstyle.WONG["sky"], figstyle.WONG["orange"], figstyle.WONG["vermillion"])[j],
            edgecolor="white", linewidth=0.4, label=f"{h} d")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(models, fontsize=MIN_ABSOLUTE_PT)
    ax_a.set_ylabel("Brier score")
    _panel_label(ax_a, "(a) Brier by horizon")
    _grid(ax_a, axis="y")
    ax_a.legend(fontsize=MIN_ABSOLUTE_PT, frameon=False)

    ax_b = fig.add_subplot(gs[0, 1])
    for model, colour, marker in (("LightGBM", figstyle.SERIES["LightGBM"], "s"),
                                  ("LSTM", figstyle.SERIES["LSTM"], "^"),
                                  ("ThermoRoute", figstyle.SERIES["ThermoRoute"], "o")):
        g = rel[(rel.model == model) & (rel.horizon == 1)]
        ax_b.plot(g.mean_forecast, g.observed_rate, color=colour, lw=1.2,
                  marker=marker, ms=3.5, markeredgecolor="white")
    ax_b.plot([0, 1], [0, 1], color=PALETTE["NEUTRAL_INK"], lw=0.7, ls=(0, (2, 2)))
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    ax_b.set_xlabel("mean forecast probability")
    ax_b.set_ylabel("observed rate")
    _panel_label(ax_b, "(b) Reliability, 1 d")
    _grid(ax_b)

    fig.suptitle("Probability diagnostics, held-out 2021\u20132023 keys")
    _fig_note(fig, "Frozen CQR + Platt calibration applied identically to the "
                   "held-out predictions; empirical diagnostics, not a "
                   "finite-sample guarantee.")
    _save_conventional(fig, "figS5_probability_reliability", out_dir)


def render_figS7(metrics, summary, out_dir):
    """figS7 -- leave-one-HUC2 influence on the paired station effect.

    For each HUC2 group, the seven-day station-median DeltaRMSE
    (ThermoRoute minus damped persistence) recomputed without that group.
    """
    import pandas as _pd
    root = Path(__file__).resolve().parents[3]
    sm = _pd.read_csv(root / "outputs/conventional/station_metrics_2021_2023.csv")
    reg = _pd.read_csv(root / "data_usgs/station_registry_v1.csv")
    reg["site_no8"] = reg["site_no"].astype(str).str.zfill(8)
    sm["site_id"] = sm["site_id"].astype(str).str.zfill(8)
    sm = sm.merge(reg[["site_no8", "huc2"]], left_on="site_id", right_on="site_no8",
                  how="left")
    wide = sm[sm.horizon == 7].pivot_table(index="site_id", columns="model",
                                           values="rmse")
    d = (wide["ThermoRoute"] - wide["DampedPersistence"]).dropna()
    huc_of = dict(zip(sm.site_id, sm.huc2))
    d = d.rename(index=huc_of)
    hucs = sorted(set(huc_of.values()))
    base = float(d.median())
    rows = []
    for h in hucs:
        sub = d[d.index != h]
        if len(sub) > 10:
            rows.append((h, float(sub.median())))
    rows.sort(key=lambda kv: -kv[1])
    fig, ax = _plt.subplots(figsize=figstyle.figsize(FULL_WIDTH_MM, 88.0))
    y = _np.arange(len(rows))
    ax.scatter([r[1] for r in rows], y, s=22, color=figstyle.WONG["sky"],
               edgecolor="white", linewidth=0.4, zorder=3)
    ax.axvline(base, color=figstyle.WONG["orange"], lw=1.4, zorder=2)
    ax.text(base, len(rows) - 0.4, f"all groups ({base:+.3f})",
            fontsize=MIN_ABSOLUTE_PT, color=figstyle.WONG["orange"], va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([f"exclude HUC2:{h}" for h, _ in rows],
                       fontsize=MIN_ABSOLUTE_PT)
    ax.set_xlabel("7 d station-median \u0394RMSE, ThermoRoute \u2212 damped (\u00b0C)")
    _panel_label(ax, "(a) Leave-one-HUC2 sensitivity, 7 d")
    _grid(ax, axis="x")
    fig.suptitle("Leave-one-HUC2 influence, held-out 2021\u20132023")
    _fig_note(fig, "Negative values favor ThermoRoute; the excluded-group "
                   "median varies within \u00b10.01 \u00b0C of the all-groups value, "
                   "so no single region carries the paired effect.")
    _save_conventional(fig, "figS7_spatial_leave_huc2", out_dir)


def render_fig05(metrics, summary, out_dir):
    """fig05 -- hydrologic conditions governing incremental skill.

    (a) thermal half-life distribution; (b) one-day learned gain versus
    half-life; (c) seven-day stratified DeltaRMSE forest plot; (d) memory /
    learned decomposition of the seven-day error budget.
    """
    import json as _json
    root = Path(__file__).resolve().parents[3]
    st = _pd.read_csv(root / "outputs/conventional/mechanism_station_level_2021_2023.csv")
    mech = _json.loads((root / "outputs/conventional/mechanism_2021_2023.json")
                       .read_text(encoding="utf-8"))

    fig = _plt.figure(figsize=figstyle.figsize(FULL_WIDTH_MM, 176.0))
    gs = fig.add_gridspec(2, 2)

    ax_a = fig.add_subplot(gs[0, 0])
    hl = st[st.horizon == 1].dropna(subset=["half_life"])
    ax_a.hist(hl.half_life, bins=22, color=figstyle.WONG["sky"],
              edgecolor="white", linewidth=0.4, zorder=2)
    med = float(hl.half_life.median())
    ax_a.axvline(med, color=figstyle.WONG["orange"], lw=1.4)
    ymax = ax_a.get_ylim()[1]
    ax_a.set_ylim(0, ymax * 1.25)
    ax_a.text(0.02, 0.96, f"median {med:.1f} d", transform=ax_a.transAxes,
              fontsize=MIN_ABSOLUTE_PT, color=figstyle.WONG["orange"], va="top")
    ax_a.set_xlabel("thermal half-life (d)")
    ax_a.set_ylabel("stations")
    _panel_label(ax_a, "(a) Thermal memory")
    _grid(ax_a, axis="y")

    ax_b = fig.add_subplot(gs[0, 1])
    g = st[st.horizon == 1].dropna(subset=["half_life", "G_learned"])
    ax_b.scatter(g.half_life, g.G_learned, s=9, color=figstyle.WONG["blue"],
                 alpha=0.6, edgecolor="none")
    m, b = _np.polyfit(g.half_life, g.G_learned, 1)
    xs = _np.linspace(g.half_life.min(), g.half_life.max(), 40)
    ax_b.plot(xs, m * xs + b, color=figstyle.WONG["orange"], lw=1.4)
    corr = float(_np.corrcoef(g.half_life, g.G_learned)[0, 1])
    ax_b.text(0.97, 0.05, f"r = {corr:+.2f}", transform=ax_b.transAxes,
              fontsize=MIN_ABSOLUTE_PT, ha="right")
    ax_b.set_xlabel("thermal half-life (d)")
    ax_b.set_ylabel("1 d learned gain over damped (\u00b0C)")
    _panel_label(ax_b, "(b) Longer memory, less to learn")
    _grid(ax_b)

    ax_c = fig.add_subplot(gs[1, 0])
    strata = mech["stratified_delta_rmse"]["7"]
    order = ["all", "rapid_change10", "warmest10", "high_flow10", "low_flow10",
             "high_disequilibrium10", "coldest10"]
    labels = ["All keys", "Fastest warming 10%", "Warmest 10%", "High flow 10%",
              "Low flow 10%", "High air\u2013water gap 10%", "Coldest 10%"]
    vals = [strata[k]["delta_rmse"] for k in order]
    y = _np.arange(len(vals))[::-1]
    ax_c.hlines(y, 0, vals, color=PALETTE["NEUTRAL_INK"], lw=1.4)
    ax_c.scatter(vals, y, s=26, color=figstyle.WONG["sky"], zorder=3,
                 edgecolor="white", linewidth=0.5)
    ax_c.axvline(0.0, color=PALETTE["NEUTRAL_INK"], lw=0.8)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(labels)
    ax_c.set_xlim(-0.38, 0.05)
    ax_c.set_xlabel("7 d \u0394RMSE, ThermoRoute \u2212 damped (\u00b0C)")
    _panel_label(ax_c, "(c) Learned gain by hydrologic state")
    _grid(ax_c, axis="x")

    ax_d = fig.add_subplot(gs[1, 1])
    h7 = mech["memory_learned_median"]["7"]
    stages = ["Persistence", "Damped", "ThermoRoute"]
    errors = [0.0, h7["G_memory"], h7["G_learned"]]
    starts = [0.0, errors[1], errors[1] + errors[2]]
    x = _np.arange(len(stages))
    ax_d.bar(x, errors, bottom=starts, color=(figstyle.WONG["blue"],
                                              figstyle.WONG["sky"],
                                              figstyle.WONG["orange"]),
             width=0.5, edgecolor="white", linewidth=0.5)
    for xi, (s0, e) in enumerate(zip(starts, errors)):
        ax_d.text(xi, s0 + e + 0.01, f"{e:.2f}", ha="center",
                  fontsize=MIN_ABSOLUTE_PT)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(["Persistence\n(2.20 \u00b0C)", "Damped\n(1.77 \u00b0C)",
                          "ThermoRoute\n(1.69 \u00b0C)"], fontsize=MIN_ABSOLUTE_PT)
    ax_d.set_ylabel("RMSE contribution (\u00b0C)")
    ax_d.set_ylim(0, 0.6)
    _panel_label(ax_d, "(d) 7 d error-budget decomposition")
    _grid(ax_d, axis="y")

    fig.suptitle("Hydrologic conditions governing incremental skill, 2021\u20132023")
    _fig_note(fig,
              "Half-life from a training-period AR(1) on damped-seasonal "
              "anomalies; learned gain is the station-median RMSE difference "
              "damped minus ThermoRoute; stratified differences are pooled "
              "over the held-out keys within each state (Section 4.8).")
    _save_conventional(fig, "fig05_hydrologic_mechanism", out_dir)
CONVENTIONAL_DISPATCH = (
    ("fig02", render_fig02),
    ("fig03", render_fig03),
    ("fig04", render_fig04),
    ("fig05", render_fig05),
    ("figS4", render_figS4),
    ("figS5", render_figS5),
    ("figS6", render_figS6),
    ("figS7", render_figS7),
    ("figS8", render_figS8),
    ("figS9", render_figS9_notice),
)


if __name__ == "__main__":
    main()

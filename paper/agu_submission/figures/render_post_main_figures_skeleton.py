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
SINGLE_COLUMN_MM = 85.0
FULL_WIDTH_MM = 140.0
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

import numpy as _np
import pandas as _pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as _plt
from matplotlib.patches import Rectangle as _Rectangle

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

# (colour, marker, linestyle).  Colour is Okabe-Ito derived (colourblind-safe);
# every series also varies marker shape and/or dash so each panel survives a
# grayscale check.  No result is encoded by colour alone.
MODEL_STYLE_CONV = {
    "ThermoRoute":       ("#0072B2", "o", "-"),
    "LightGBM":          ("#CC79A7", "s", "-"),
    "LSTM":              ("#009E73", "^", "-"),
    "Persistence":       ("#777777", "D", (0, (4, 2))),
    "DampedPersistence": ("#E69F00", "v", (0, (4, 2))),
    "Climatology":       ("#B8B8B8", "P", (0, (1, 2))),
    "DampedPriorOnly":   ("#E69F00", "v", (0, (1, 2))),
    "TR-noTCN":          ("#008C7A", "o", (0, (5, 2))),
    "TR-noMoE":          ("#56B4E9", "s", (0, (5, 2))),
    "TR-noRouter":       ("#0072B2", "^", (0, (5, 2))),
    "TR-noDynamicPrior": ("#009E73", "D", (0, (5, 2))),
    "TR-fixedKappa":     ("#CC79A7", "v", (0, (5, 2))),
    "TR-unbounded":      ("#D55E00", "P", (0, (5, 2))),
    "ThermoRoute-ext":   ("#0072B2", "o", (0, (2, 2))),
    "LSTM-ext":          ("#009E73", "^", (0, (2, 2))),
    "LightGBM-ext":      ("#CC79A7", "s", (0, (2, 2))),
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


def _mm(value: float) -> float:
    return value / 25.4


def _conventional_rcparams() -> None:
    _plt.rcParams.update({
        "font.size": 8.0, "axes.titlesize": 8.5, "axes.titleweight": "bold",
        "axes.labelsize": 8.0, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.dpi": 300, "figure.dpi": 120, "axes.grid": False,
        "grid.alpha": 0.25, "lines.linewidth": 1.3,
    })


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
    return metrics, summary


def _skill_vs(metrics, candidate, reference, horizon):
    return 1.0 - metrics[candidate][horizon]["RMSE"] / \
        metrics[reference][horizon]["RMSE"]


def _save_conventional(fig, stem, out_dir: Path) -> None:
    # Save at the exact authored figure size (no bbox_inches="tight"): tight
    # bbox expands beyond the AGU target when labels/annotations spill, which
    # would scale type below the 7.5 pt floor at placed size.  Margins are set
    # per figure so labels stay inside the box without clipping.
    for ext in ("pdf", "png", "svg"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300)
    _plt.close(fig)
    print(f"wrote {stem}.pdf/.png/.svg", flush=True)


def _style(model):
    return MODEL_STYLE_CONV.get(model, ("#202020", "o", "-"))


def _panel_label(ax, text):
    ax.set_title(text, loc="left", fontweight="bold", fontsize=9.0)


def render_fig02(metrics, summary, out_dir):
    """fig02 -- baseline choice sets the reported gain (target, pooled)."""
    fig = _plt.figure(figsize=(_mm(140), _mm(95)))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0],
                          hspace=0.62, wspace=0.30,
                          left=0.135, right=0.955, top=0.89, bottom=0.135)
    ax_a = fig.add_subplot(gs[0, :])
    for ref in LADDER_REFERENCES:
        colour, marker, ls = _style(ref)
        ys = [_skill_vs(metrics, "ThermoRoute", ref, h) for h in CONV_HORIZONS]
        ax_a.plot(CONV_HORIZONS, ys, ls=ls, marker=marker, color=colour,
                  ms=5.5, lw=1.4, label=MODEL_LABEL_CONV[ref],
                  markeredgecolor="k", markeredgewidth=0.3)
        ax_a.annotate(f"{ys[-1]:+.2f}", (CONV_HORIZONS[-1], ys[-1]),
                      textcoords="offset points", xytext=(0, 7),
                      fontsize=6.6, color=colour, ha="center")
    ax_a.axhline(0.0, color="#202020", lw=0.8)
    ax_a.set_xticks(CONV_HORIZONS)
    ax_a.set_xlabel("forecast horizon (days)")
    ax_a.set_ylabel("Skill vs reference\n(+ favours candidate)")
    _panel_label(ax_a, "(a) Reference ladder: ThermoRoute skill against each reference")
    ax_a.legend(loc="upper left", ncol=3, frameon=False, fontsize=6.8,
                columnspacing=1.2, handlelength=2.2)
    ax_a.grid(axis="y", alpha=0.25)
    ax_a.set_xlim(0.6, 7.6)

    ax_b = fig.add_subplot(gs[1, 0])
    for m in PRIMARY_MODELS_CONV:
        colour, marker, ls = _style(m)
        ys = [metrics[m][h]["RMSE"] for h in CONV_HORIZONS]
        ax_b.plot(CONV_HORIZONS, ys, ls=ls, marker=marker, color=colour,
                  ms=5, lw=1.3, label=MODEL_LABEL_CONV[m],
                  markeredgecolor="k", markeredgewidth=0.3)
    ax_b.set_xticks(CONV_HORIZONS)
    ax_b.set_xlabel("forecast horizon (days)")
    ax_b.set_ylabel("pooled RMSE (\u00b0C)")
    _panel_label(ax_b, "(b) Six primary models, pooled RMSE")
    ax_b.legend(loc="upper left", ncol=2, frameon=False, fontsize=6.5,
                columnspacing=1.0, handlelength=2.0)
    ax_b.grid(axis="y", alpha=0.25)

    ax_c = fig.add_subplot(gs[1, 1])
    rows = [
        ("ThermoRoute \u2212 Damped persistence", 1,
         metrics["ThermoRoute"][1]["RMSE"] - metrics["DampedPersistence"][1]["RMSE"]),
        ("ThermoRoute \u2212 Damped persistence", 3,
         metrics["ThermoRoute"][3]["RMSE"] - metrics["DampedPersistence"][3]["RMSE"]),
        ("ThermoRoute \u2212 Damped persistence", 7,
         metrics["ThermoRoute"][7]["RMSE"] - metrics["DampedPersistence"][7]["RMSE"]),
        ("ThermoRoute \u2212 LightGBM", 3,
         metrics["ThermoRoute"][3]["RMSE"] - metrics["LightGBM"][3]["RMSE"]),
        ("ThermoRoute \u2212 LightGBM", 7,
         metrics["ThermoRoute"][7]["RMSE"] - metrics["LightGBM"][7]["RMSE"]),
    ]
    y = _np.arange(len(rows))[::-1]
    vals = _np.array([r[2] for r in rows])
    colours = ["#E69F00", "#E69F00", "#E69F00", "#CC79A7", "#CC79A7"]
    ax_c.hlines(y, 0, vals, color=colours, lw=2.0, alpha=0.85)
    ax_c.scatter(vals, y, color=colours, s=42, zorder=3,
                 edgecolor="k", linewidth=0.4)
    ax_c.axvline(0.0, color="#202020", lw=0.9)
    ax_c.axvline(0.05, color="#D55E00", lw=0.9, ls=(0, (3, 2)))
    ax_c.text(0.045, 0.99, "+0.05 ceiling",
              transform=ax_c.get_xaxis_transform(), color="#D55E00",
              fontsize=6.2, ha="right", va="top")
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([f"{r[0]}\n{r[1]} d" for r in rows], fontsize=6.8)
    ax_c.set_xlabel("\u0394RMSE (\u00b0C; \u2212 favours ThermoRoute)", fontsize=7.5)
    _panel_label(ax_c, "(c) Paired \u0394RMSE (pooled)")
    ax_c.set_xlim(-0.20, 0.12)
    ax_c.grid(axis="x", alpha=0.25)
    fig.text(0.5, 0.012,
             "CI and win rate: not reported (\u2014) for the held-out window "
             "(paper Table 4.6).\n\u0394RMSE is pooled over common held-out keys; "
             "station-level clustered procedure not re-run.",
             ha="center", va="bottom", fontsize=6.0, color="#444444",
             linespacing=1.3)
    fig.suptitle("Held-out 2021\u20132023 \u00b7 pooled metrics over common forecast keys",
                 y=0.965, fontsize=8.5, fontweight="bold")
    _save_conventional(fig, "fig02_point_performance", out_dir)


def render_figS4(metrics, summary, out_dir):
    """figS4 -- point-performance heterogeneity (all models x horizons, pooled)."""
    fig = _plt.figure(figsize=(_mm(140), _mm(102)))
    gs = fig.add_gridspec(1, 2, left=0.20, right=0.86, top=0.88,
                          bottom=0.09, wspace=0.50)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    models = list(ALL_MODELS_CONV)
    rmse = _np.array([[metrics[m][h]["RMSE"] for h in CONV_HORIZONS]
                      for m in models])
    skill = _np.array([[metrics[m][h]["SKILL_PERSISTENCE"] for h in CONV_HORIZONS]
                       for m in models])

    im0 = axes[0].imshow(rmse, cmap="YlOrRd", aspect="auto")
    axes[0].set_title("(a) Pooled RMSE (\u00b0C)", loc="left", fontweight="bold")
    vlim = max(abs(_np.nanmin(skill)), abs(_np.nanmax(skill)))
    im1 = axes[1].imshow(skill, cmap="RdBu_r", aspect="auto",
                         vmin=-vlim, vmax=vlim)
    axes[1].set_title("(b) Skill vs persistence (+ favours)",
                      loc="left", fontweight="bold", fontsize=8.0)
    for ax, mat, fmt in ((axes[0], rmse, "{:.2f}"),
                         (axes[1], skill, "{:+.2f}")):
        ax.set_xticks(range(len(CONV_HORIZONS)))
        ax.set_xticklabels([f"{h} d" for h in CONV_HORIZONS])
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([MODEL_LABEL_CONV[m] for m in models], fontsize=6.0)
        ax.set_xlabel("forecast horizon")
        vmax = _np.nanmax(_np.abs(mat))
        for (i, j), v in _ndenumerate(mat):
            color = "white" if abs(v) > 0.55 * vmax else "black"
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    fontsize=6.0, color=color)
    fig.colorbar(im0, ax=axes[0], shrink=0.82)
    fig.colorbar(im1, ax=axes[1], shrink=0.82)
    fig.suptitle("figS4 \u00b7 16 models \u00d7 horizon, pooled 2021\u20132023 "
                 "(station-level detail not in CSV)",
                 fontsize=8.0, fontweight="bold")
    _save_conventional(fig, "figS4_point_heterogeneity", out_dir)


def _ndenumerate(arr):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            yield (i, j), arr[i, j]


def render_figS6(metrics, summary, out_dir):
    """figS6 -- temporal opportunity, missingness, and attrition."""
    failures = summary.get("station_failures", [])
    n_panel = summary.get("n_stations_panel", 120)
    n_scored = n_panel - len(failures)
    n_win = summary.get("n_windows_temporal")
    fig = _plt.figure(figsize=(_mm(140), _mm(90)))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.26,
                          left=0.125, right=0.975, top=0.88, bottom=0.13)
    ax_a = fig.add_subplot(gs[0, 0])
    stages = ["Panel sites", "Scored sites", "h=1 keys", "h=3 keys", "h=7 keys"]
    counts = [n_panel, n_scored,
              metrics["ThermoRoute"][1]["n"],
              metrics["ThermoRoute"][3]["n"],
              metrics["ThermoRoute"][7]["n"]]
    colours = ["#0072B2", "#009E73", "#CC79A7", "#E69F00", "#D55E00"]
    bars = ax_a.barh(range(len(stages)), counts, color=colours, alpha=0.9,
                     edgecolor="k", linewidth=0.4)
    ax_a.set_yticks(range(len(stages)))
    ax_a.set_yticklabels(stages, fontsize=6.8)
    ax_a.invert_yaxis()
    ax_a.set_xscale("log")
    ax_a.set_xlabel("count (log scale)")
    _panel_label(ax_a, "(a) Attrition: 120 sites \u2192 115 scored \u2192 keys")
    for bar, c in zip(bars, counts):
        ax_a.text(c * 1.03, bar.get_y() + bar.get_height() / 2,
                  f"{c:,}", va="center", fontsize=6.8)
    ax_a.set_xlim(50, max(counts) * 4)
    ax_a.grid(axis="x", alpha=0.25, which="both")

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.axis("off")
    _panel_label(ax_b, f"(b) Fetch failures ({len(failures)} of {n_panel} excluded)")
    lines = ["site_no    reason", "-" * 24]
    for f in failures:
        reason = (f.get("nwis") or "\u2014")
        reason = (reason[:28] + "\u2026") if len(reason) > 29 else reason
        lines.append(f"{str(f.get('site_no', '?')):<10} {reason}")
    if n_win is not None:
        lines.append("")
        lines.append(f"temporal windows scored: {n_win:,}")
    ax_b.text(0.02, 0.93, "\n".join(lines), transform=ax_b.transAxes,
              va="top", ha="left", fontsize=6.3,
              family="DejaVu Sans Mono",
              bbox=dict(boxstyle="round,pad=0.35", fc=PALETTE["NA_FILL"],
                        ec=PALETTE["WARNING_VERMILION"], lw=0.8))
    fig.suptitle("figS6 \u00b7 missingness and attrition on the held-out 2021\u20132023 panel",
                 y=0.965, fontsize=8.0, fontweight="bold")
    _save_conventional(fig, "figS6_attrition_missingness", out_dir)


def render_figS8(metrics, summary, out_dir):
    """figS8 -- external history-dependent arm and failure disposition."""
    pairs = [("LightGBM", "LightGBM-ext"), ("LSTM", "LSTM-ext"),
             ("ThermoRoute", "ThermoRoute-ext")]
    failures = summary.get("station_failures", [])
    fig = _plt.figure(figsize=(_mm(140), _mm(95)))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.28,
                          left=0.095, right=0.975, top=0.88, bottom=0.16)
    ax_a = fig.add_subplot(gs[0, 0])
    width = 0.38
    x = _np.arange(len(CONV_HORIZONS))
    for k, (temp, ext) in enumerate(pairs):
        colour, marker, _ = _style(temp)
        y_temp = [metrics[temp][h]["RMSE"] for h in CONV_HORIZONS]
        y_ext = [metrics[ext][h]["RMSE"] for h in CONV_HORIZONS]
        off = (k - 1) * width
        ax_a.bar(x + off - width / 2, y_temp, width, color=colour, alpha=0.9,
                 edgecolor="k", linewidth=0.4,
                 label=f"{MODEL_LABEL_CONV[temp]} (temporal)")
        ax_a.bar(x + off + width / 2, y_ext, width, color=colour, alpha=0.4,
                 hatch="///", edgecolor="k", linewidth=0.4,
                 label=f"{MODEL_LABEL_CONV[ext]} (external)")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"{h} d" for h in CONV_HORIZONS])
    ax_a.set_xlabel("forecast horizon")
    ax_a.set_ylabel("pooled RMSE (\u00b0C)")
    _panel_label(ax_a, "(a) External (site-ID-disjoint) vs temporal cohort")
    ax_a.legend(loc="upper left", ncol=2, frameon=False, fontsize=5.8,
                columnspacing=0.8, handlelength=1.6)
    ax_a.grid(axis="y", alpha=0.25)

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.axis("off")
    _panel_label(ax_b, f"(b) Failures ({len(failures)} excluded)")
    rows = [["site_no", "status", "reason"]]
    for f in failures:
        reason = (f.get("nwis") or "\u2014")
        reason = (reason[:16] + "\u2026") if len(reason) > 17 else reason
        rows.append([str(f.get("site_no", "?")), "excluded", reason])
    celltext = rows[1:] if len(rows) > 1 else [["\u2014", "\u2014", "\u2014"]]
    table = ax_b.table(cellText=celltext, colLabels=rows[0],
                       loc="upper center", cellLoc="left",
                       colWidths=[0.15, 0.14, 0.46])
    table.auto_set_font_size(False)
    table.set_fontsize(5.6)
    table.scale(1.0, 1.25)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D0D0")
        if r == 0:
            cell.set_facecolor("#DCEAF4")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#FFFFFF" if r % 2 else "#F7F7F7")
    ax_b.text(0.5, 0.04,
              "External cohort: site-ID disjoint,\nhistory-dependent; not ungauged.\n"
              "Outcome-QC waterfall (Table 4.10)\nnot reported for the held-out window.",
              transform=ax_b.transAxes, ha="center", va="bottom",
              fontsize=5.8, color="#444444", linespacing=1.3)
    fig.suptitle("figS8 \u00b7 external arm and failure disposition (2021\u20132023)",
                 y=0.965, fontsize=8.0, fontweight="bold")
    _save_conventional(fig, "figS8_external_arm_failures", out_dir)


def render_notice(figure_id, stem, title, reason_lines, out_dir,
                  width_mm=85.0, height_mm=62.0):
    """Render an explicit 'not reported' notice; no axes, no invented data."""
    fig, ax = _plt.subplots(figsize=(_mm(width_mm), _mm(height_mm)))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(_Rectangle((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
                            facecolor=PALETTE["NA_FILL"],
                            edgecolor=PALETTE["WARNING_VERMILION"], lw=1.6))
    body = "\n".join(reason_lines)
    text = f"{figure_id} \u2014 {title}\n\nNOT REPORTED for the 2021\u20132023 held-out window\n\n{body}"
    ax.text(0.5, 0.5, text, ha="center", va="center",
            transform=ax.transAxes, fontsize=7.6, color=PALETTE["NEUTRAL_INK"],
            linespacing=1.45, wrap=True)
    _save_conventional(fig, stem, out_dir)


def render_fig03_notice(metrics, summary, out_dir):
    render_notice(
        "fig03", "fig03_spatial_partition_transfer",
        "Spatial partition / whole-region transfer",
        ["Development-period figure (Stage-13c region-transfer evidence,",
         "2019-01-01 to 2020-12-31).  The 2021-2023 conventional holdout",
         "emits no held-region arm, so this figure is not rendered from the",
         "conventional metrics CSV.  See paper section 4.4 for the",
         "development-period three-arm values."], out_dir)


def render_fig04_notice(metrics, summary, out_dir):
    render_notice(
        "fig04", "fig04_heterogeneity_and_interval_cost",
        "Regional/seasonal heterogeneity and interval cost",
        ["Panels require a per-HUC2 regional breakdown, the eight temporal-",
         "coverage candidates, and the 90% coverage-width plane.  None was",
         "computed for 2021-2023: the conventional holdout CSV is pooled (no",
         "HUC2 dimension) and the probability pipeline was not re-run (paper",
         "Table 4.9: not reported).  No value is invented."], out_dir)


def render_figS5_notice(metrics, summary, out_dir):
    render_notice(
        "figS5", "figS5_probability_reliability",
        "Event score, reliability, and probabilistic diagnostics",
        ["Requires the SI08 probability family (coverage, pinball, Brier,",
         "log loss, AUROC/AUPRC, ECE, calibration slope/intercept,",
         "station-balanced reliability bins).  Not computed for 2021-2023",
         "(paper Table 4.9: not reported).  The development-period",
         "probability diagnostics (paper section 4.5) stand as the only",
         "probability evidence in this paper."], out_dir)


def render_figS7_notice(metrics, summary, out_dir):
    render_notice(
        "figS7", "figS7_spatial_leave_huc2",
        "Spatial and leave-HUC2 influence",
        ["Requires per-HUC2 effects and leave-one-HUC2 omissions.  The",
         "conventional holdout CSV is pooled with no HUC2 dimension, so this",
         "figure is not reported for the held-out window.  No value is",
         "invented."], out_dir)


def render_figS9_notice(metrics, summary, out_dir):
    render_notice(
        "figS9", "figS9_conformal_calibration",
        "Development-period conformal calibration sensitivity",
        ["Development-period figure (Stage-22 adaptive conformal,",
         "2019-01-01 to 2020-12-24).  Not a 2021-2023 result and not rendered",
         "from the conventional metrics CSV.  See paper section 4.5 for the",
         "development-period split-CQR / block-max / delayed-ACI values."],
        out_dir)


CONVENTIONAL_DISPATCH = (
    ("fig02", render_fig02),
    ("fig03", render_fig03_notice),
    ("fig04", render_fig04_notice),
    ("figS4", render_figS4),
    ("figS5", render_figS5_notice),
    ("figS6", render_figS6),
    ("figS7", render_figS7_notice),
    ("figS8", render_figS8),
    ("figS9", render_figS9_notice),
)


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
    if skipped:
        print(f"failed: {skipped}", file=sys.stderr)


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


if __name__ == "__main__":
    main()

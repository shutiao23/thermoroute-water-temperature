#!/usr/bin/env python3
"""POST skeleton for Supporting Figures S4--S10.

Encodes the panel, value-ID, dependency, render-profile, and prohibited-semantics
contracts of ``paper/FIGURE_REDRAW_SPEC.md`` (Figure S4--S10 sections) in the
fail-closed style of ``render_fig01_preopening_concept.py`` and the main-figure
skeleton ``paper/agu_submission/figures/render_post_main_figures_skeleton.py``.
Before the POST gate this module renders nothing -- no SVG/PDF/PNG, no empty
axes, no ``PENDING`` boxes, no zero-filled values.  ``--status`` is read-only.

Gate order (all required before any panel builder may run):

1. the spec no longer lists the figure in a ``*TEMPLATE_ONLY*`` state;
2. the verified opening receipt exists under
   ``outputs/confirmatory/route_a_*/opening_receipt_v1.json``;
3. every dependency declared ``Severity.REQUIRED`` resolves;
4. every required ``value_id`` is declared through :class:`ValueBinder` and every
   visible mark routes its cells through ``ValueBinder.mark``.

Dependencies declared ``Severity.QUALIFIER`` never block; they are resolved,
recorded, and reported so the render receipt carries accurate provenance.

SI figures expand the main figures and never duplicate them; every retained/NA
station must reconcile with Figure 2 and Table T2 value IDs.

Stage-19 disposition (2026-08-05)
---------------------------------
The Stage-19 *development* script will not be produced for this submission (see
the main-figure skeleton for the full statement).  The measured cause is a
**zero-width nominal interval (degenerate quantile head)**, not quantile
crossing: across 26,993,675 member-level rows there are **0** strict ordering
violations and **135** rows with ``q05 == q50 == q95`` bit-identically (0.0005%),
maximum monotonicity violation exactly 0.000 degrees C.

**Figure S5 is not affected.**  The SI08 metric family is produced at target
period by the trusted scorer inside the one-time opening
(``src/thermoroute/opening.py:8932-8948``), per cohort x model x horizon with
station-balanced reliability bins.  S5 therefore keeps its original expanded
design, bound to
``outputs/confirmatory/route_a_*/trusted/probabilistic_evaluation_v2.json``.

Benchmark restructure (2026-08-06)
----------------------------------
``docs/PAPER_BENCHMARK_RESTRUCTURE.md`` reassigned the four main figures; see
``paper/FIGURE_REDRAW_SPEC.md`` section 6.4 and the main-figure skeleton.  The SI
consequences, all applied here:

* **figS5** receives the event-score panel and the three per-horizon reliability
  panels of the former main Figure 3, whose main-text slot became the
  spatial-partition figure.  Only the coverage-width plane stays in the main
  text, as fig04 panel (c), so S5 now expands **fig04**, not fig03.
* **figS6** receives the attrition waterfall of the former fig04 panel (b).  It
  already carried the same denominator-preserving construction, so panel (a)
  simply gains the reportable-cluster stage.
* **figS7** expands fig04 panel (a) rather than standing alone: fig04 carries the
  aggregate per-HUC2 medians, S7 carries every unit and every leave-one omission.
* **figS8** receives the external history-dependent arm of the former fig04
  panel (d) into panel (b), which already owned that cohort's scope statement.
* **figS9** is confirmed as the only home of the Stage-22 conformal contrast; the
  restructure's proposed development-period panel inside target-period fig04 is
  refused, because one figure never mixes two evidence periods.
* **figS10** is new and receives the seven registered architecture interventions
  demoted from the former fig04 panel (a).  It is target-period: the protocol's
  ``mandatory_exploratory_architecture_controls`` are resolved into the temporal
  cohort's required model set beside the six primary models, so the trusted
  scorer emits a row for every control on the same exact common keys.  The
  Stage-09b information-matched controls are development-period and stay a
  non-blocking qualifier plus an SI09 table; they never fill a panel.

``PanelSpec.evidence_period`` plus ``validate_manifest`` make the one-period rule
machine-enforced at panel granularity, and ``NAMESPACE_EVIDENCE_PERIOD`` extends
it to ``shares_value_ids_with``: a target-period figure can no longer declare that
it shares value IDs with a development-period namespace.

Figure S9 (new)
---------------
Stage-22 (``scripts/22_adaptive_conformal.py``) is **development-period**
evidence: ``outputs/tables/aci_coverage.csv`` spans 2019-01-01..2020-12-24 while
the confirmatory target period starts 2021-01-01.  It is not a substitute for any
target-period figure.  It is carried by Figure S9 as a development-period
sensitivity to the choice of conformal calibration method, with
``EvidencePeriod.DEVELOPMENT`` and a mandatory in-panel scope band.  S9 is a
separate figure, never a panel of S5, because one figure never mixes two
evidence periods.  S9 is optional: dropping it weakens no registered claim.
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

# AGU / WRR render profile -- authority is FIGURE_REDRAW_SPEC.md section 3.2.
# 140 mm full width matches agujournal2019.cls \textwidth 5.5in = 139.7 mm;
# 85 mm is the AGU single-column maximum (published range 50--85 mm).  95 mm and
# 190 mm are not AGU sizes and must not be reintroduced.
SINGLE_COLUMN_MM = 85.0
FULL_WIDTH_MM = 140.0
MAX_HEIGHT_MM = 228.0
MIN_BODY_PT = 8.0
MIN_ABSOLUTE_PT = 7.5
MIN_STROKE_PT = 0.6
VECTOR_FORMATS = ("pdf", "svg")
RASTER_MIN_DPI = 300
LINE_ART_MIN_DPI = 600

# Source-controlled paths mirror src/thermoroute/model_suite.py and
# src/thermoroute/development_controls_gate.py; verified 2026-08-05.
OPENING_RECEIPT_GLOB = "outputs/confirmatory/route_a_*/opening_receipt_v1.json"

# Target-period artifacts written by the trusted scorer inside the one-time
# opening.  Exact names from the state-path table in
# src/thermoroute/opening.py:1226-1281 (verified 2026-08-05); the leading glob
# absorbs the run namespace.  Fail-closed guards -- never delete to unblock.
_TRUSTED = "outputs/confirmatory/route_a_*/trusted"
POST_TEMPORAL_PREDICTIONS = f"{_TRUSTED}/temporal_predictions_v1.parquet"
POST_EXTERNAL_PREDICTIONS = f"{_TRUSTED}/external_predictions_v1.parquet"
POST_AVAILABILITY_REGISTRY = f"{_TRUSTED}/availability_registry_v1.csv"
POST_STATISTICS = f"{_TRUSTED}/statistics_v1.json"
POST_PROBABILISTIC_EVALUATION = f"{_TRUSTED}/probabilistic_evaluation_v2.json"
POST_SPATIAL_SENSITIVITY = f"{_TRUSTED}/spatial_sensitivity_v1.json"
POST_OUTCOME_QC_GATE = f"{_TRUSTED}/outcome_qc_gate_v1.json"
POST_OUTCOME_QUALITY_AUDIT = f"{_TRUSTED}/outcome_quality_audit_v1.json"
POST_TEMPORAL_COVERAGE_AUDIT = f"{_TRUSTED}/temporal_coverage_audit_v1.json"
STAGE09_RECEIPT = "outputs/models/route_a_stage09_completion.json"
STAGE09B_RECEIPT = "outputs/models/route_a_stage09b_completion.json"
STAGE16_RECEIPT = "outputs/models/route_a_stage16_completion.json"
STAGE25_RECEIPT = "outputs/models/route_a_stage25_completion.json"

# Stage-22 adaptive conformal (scripts/22_adaptive_conformal.py).
STAGE22_ROW_TABLE = "outputs/tables/aci_coverage.csv"
STAGE22_REPORT = "outputs/reports/adaptive_conformal.md"
# Stage-13c region transfer (scripts/13c_region_transfer.py).
STAGE13C_TABLE = "outputs/tables/region_transfer.csv"
STAGE13C_REPORT = "outputs/reports/region_transfer.md"

SI06_RECEIPT = "paper/si/SI06_formal_five_rows_RECEIPT.md"
SI07_RECEIPT = "paper/si/SI07_all_model_scores_RECEIPT.md"
SI08_RECEIPT = "paper/si/SI08_probability_metrics_RECEIPT.md"
SI09_RECEIPT = "paper/si/SI09_development_controls_RECEIPT.md"
SI10_RECEIPT = "paper/si/SI10_temporal_coverage_RECEIPT.md"
SI11_RECEIPT = "paper/si/SI11_spatial_sensitivity_RECEIPT.md"
SI12_RECEIPT = "paper/si/SI12_qc_qualifiers_RECEIPT.md"
SI13_RECEIPT = "paper/si/SI13_external_history_arm_RECEIPT.md"
SI14_RECEIPT = "paper/si/SI14_missingness_failures_RECEIPT.md"
ERRATUM_CONTRACT = "protocols/route_a_probability_metric_erratum_v1.json"
CONFIRMATORY_PROTOCOL = "protocols/route_a_confirmatory_v1.json"

CONFORMAL_METHODS = (
    "split_cqr",
    "block_max_cqr_7_retained_rows",
    "aci_gamma_0p005",
    "aci_gamma_0p02",
    "aci_gamma_0p05",
)
CONFORMAL_SLICES = ("overall", "warm_train_q90_tail", "lead_1d", "lead_3d", "lead_7d")

PRIMARY_TARGET_START = "2021-01-01"
STAGE22_OBSERVED_SPAN = ("2019-01-01", "2020-12-24")

TEMPLATE_ONLY_MARKER = "TEMPLATE_ONLY"


class Severity:
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
    """Source-controlled root vs computed-artifact root.

    They differ for this submission: the Stage-09/16/25 receipts and Stage-22
    tables live in the sibling ``thermoroute-water-temperature-multicore``
    worktree, not in the paper tree.
    """

    repo: Path
    evidence: Path

    def for_kind(self, root_id: str) -> Path:
        return self.repo if root_id == "repo" else self.evidence


def resolve_roots(explicit: str | None = None) -> Roots:
    # HERE = paper/si/figures; parents[2] of that directory is the repository
    # root.
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
    # Empty means "inherit the figure's period".  Declaring a different one is
    # refused by validate_manifest: one figure never mixes two evidence periods.
    evidence_period: str = ""


@dataclass(frozen=True)
class DroppedPanel:
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
    # Reported by --status for any development-period figure.  The defaults
    # reproduce the Stage-22 wording figS9 has always emitted.
    evidence_span: tuple[str, str] = STAGE22_OBSERVED_SPAN
    evidence_span_label: str = "Stage-22 span"


# Every namespace a figure may share value IDs with, and the evidence period
# those values carry.  A shared value ID means the same number, unit and
# rounding in two places, which two evidence periods can never satisfy, so
# validate_manifest refuses a cross-period share.  PRE-structural namespaces are
# period-neutral by construction (spec section 2.1 layer 1).
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
        figure_id="figS4",
        stem="figS4_point_heterogeneity",
        spec_anchor="FIGURE_REDRAW_SPEC.md#figure-s4",
        evidence_period=EvidencePeriod.TARGET,
        render_profile=RenderProfile(width_mm=FULL_WIDTH_MM, max_height_mm=182.0),
        panels=(
            PanelSpec("a", "HUC2-sorted site-by-horizon heatmap",
                      "paired TR-minus-reference RMSE; one shared declared scale; explicit NA cells"),
            PanelSpec("b", "station-effect ECDFs",
                      "same paired differences; denominators labelled"),
            PanelSpec("c", "optional HUC2 small multiples",
                      "shared declared scale only; no per-panel adaptive scales"),
        ),
        required_value_ids=(
            "pair.site_id", "pair.huc2", "pair.horizon", "pair.model_ids",
            "pair.exact_key_digest", "pair.station_rmse_candidate",
            "pair.station_rmse_reference", "pair.paired_difference",
            "pair.paired_target_count", "pair.reportability_na_reason",
            "sorting_registry", "shared_scale_limits",
        ),
        dependencies=(
            Dependency("opening_receipt", "glob", OPENING_RECEIPT_GLOB),
            Dependency("stage09_receipt", "file", STAGE09_RECEIPT),
            Dependency("si06_formal_rows", "file", SI06_RECEIPT, root="repo"),
            Dependency("si07_all_model_scores", "file", SI07_RECEIPT, root="repo"),
            Dependency("confirmatory_protocol", "file", CONFIRMATORY_PROTOCOL,
                       root="repo"),
            Dependency("post_temporal_predictions", "glob",
                       POST_TEMPORAL_PREDICTIONS),
            Dependency("post_availability_registry", "glob",
                       POST_AVAILABILITY_REGISTRY,
                       note="exact-key / reportability denominators"),
            Dependency("post_statistics", "glob", POST_STATISTICS,
                       note="reconciliation against Figure 2 / T2"),
        ),
        prohibited_semantics=(
            "selective sites", "per-panel adaptive color scales",
            "national map interpolation", "hydrologic connectivity",
            "causal spatial explanation", "omitted NA",
        ),
        shares_value_ids_with=("fig02", "table_t2"),
        scope_band_value_id="figS4.scope.fixed_cohort_descriptive",
    ),
    # ------------------------------------------------------------------
    # Figure S5 -- RESTORED to target period 2026-08-05 (supersedes the
    # development-period rebinding drafted earlier the same day), then
    # EXTENDED 2026-08-06 by the benchmark restructure.
    #
    # Determination: the SI08 metric family IS available at target period.  It
    # is produced by the trusted scorer inside the one-time opening
    # (src/thermoroute/opening.py:8932-8948), not by the withheld Stage-19
    # development script, and it is emitted per cohort x model x horizon with
    # station-balanced reliability bins.  The original expanded design is
    # therefore restored and bound to trusted/probabilistic_evaluation_v2.json.
    #
    # 2026-08-06: the restructure keeps only the coverage-width plane in the
    # main text, as fig04 panel (c).  The event-score panel and the three
    # per-horizon reliability panels of the former main Figure 3 move HERE and
    # are now this figure's primary content.  Nothing is dropped, and this
    # figure now expands fig04 rather than fig03 -- fig03 is the
    # development-period spatial-partition figure and shares no value ID with
    # anything here.
    # ------------------------------------------------------------------
    FigureSpec(
        figure_id="figS5",
        stem="figS5_probability_diagnostics",
        spec_anchor="FIGURE_REDRAW_SPEC.md#figure-s5",
        evidence_period=EvidencePeriod.TARGET,
        render_profile=RenderProfile(width_mm=FULL_WIDTH_MM, max_height_mm=182.0),
        panels=(
            PanelSpec("a", "model-by-horizon coverage/width dot matrix",
                      "full grid; every cell bound or explicit NA"),
            PanelSpec("b", "event score and score matrix",
                      "Brier skill vs the frozen seasonal reference by model x "
                      "horizon, with the zero-skill line and the bound reference "
                      "identity drawn and confirmation-period event prevalence "
                      "never used as the reference (relocated from the former "
                      "main Figure 3 panel (b)); shown with the pinball, "
                      "interval, Brier, and log score matrix under an explicit "
                      "scoring-stage legend (nominal pre-CQR heads vs deployed CQR "
                      "interval vs post-Platt probability); the equal-weight "
                      "three-quantile pinball summary is never labelled CRPS"),
            PanelSpec("c", "reliability panels, one per horizon",
                      "observed station-balanced event frequency vs mean forecast "
                      "probability with the identity line and point area = bound "
                      "bin denominator / station-balanced effective weight "
                      "(relocated from the former main Figure 3 panels (c)-(e)); "
                      "every registered bin and count; empty bins explicit; "
                      "station-balanced bin weights reconcile to one"),
            PanelSpec("d", "calibration and discrimination diagnostics",
                      "calibration slope/intercept, AUROC, AUPRC, ECE with bound "
                      "NA reasons for single-class or fit-failure cases"),
        ),
        required_value_ids=(
            "si08_metric_fields", "probability_source_stage", "model_count",
            "horizon_count", "bin.boundary", "bin.statistic",
            "bin.denominator", "bin.station_balanced_weight",
            "bin.undefined_reason", "reference_identity",
            "calibration_fit_status", "event_reference_binding",
            "threshold_scope", "undefined_reason",
        ),
        dependencies=(
            Dependency("opening_receipt", "glob", OPENING_RECEIPT_GLOB),
            Dependency("post_probabilistic_evaluation", "glob",
                       POST_PROBABILISTIC_EVALUATION,
                       note="trusted-scorer target-period probability metrics; "
                            "independent of the withheld Stage-19 script"),
            Dependency("post_temporal_predictions", "glob",
                       POST_TEMPORAL_PREDICTIONS),
            Dependency("post_external_predictions", "glob",
                       POST_EXTERNAL_PREDICTIONS,
                       note="external cohort rows of the same grid"),
            Dependency("post_availability_registry", "glob",
                       POST_AVAILABILITY_REGISTRY,
                       note="reportability denominators"),
            Dependency("erratum_contract", "file", ERRATUM_CONTRACT, root="repo"),
            Dependency("si08_probability_metrics", "file", SI08_RECEIPT,
                       root="repo"),
            Dependency("confirmatory_protocol", "file", CONFIRMATORY_PROTOCOL,
                       root="repo"),
        ),
        prohibited_semantics=(
            "three-quantile score labelled CRPS", "silent metric substitution",
            "empty-bin merging", "conditional-coverage language",
            "model rows with invented heads",
            "distribution-free target-period guarantee",
            "confirmation event prevalence as the Brier reference",
            "quantile crossing as the Stage-19 cause",
            "development-period conformal numbers shown as target-period results",
        ),
        shares_value_ids_with=("fig04", "si08"),
        provenance_qualifiers=(
            "The Stage-19 development-period probabilistic script is withheld for "
            "this submission (zero-width nominal intervals; 0 strict quantile "
            "crossings in 26,993,675 member-level rows). This figure does NOT "
            "depend on it: the target-period metrics come from the trusted scorer "
            "inside the one-time opening (src/thermoroute/opening.py:8932-8948).",
            "Event-score and reliability panels relocated here on 2026-08-06 from "
            "the former main Figure 3, whose main-text slot became the "
            "development-period spatial-partition figure. Only the coverage-width "
            "plane remains in the main text, as Figure 4 panel (c); this figure "
            "expands that panel and duplicates none of it.",
        ),
        scope_band_value_id="figS5.scope.fixed_cohort_descriptive",
    ),
    FigureSpec(
        figure_id="figS6",
        stem="figS6_temporal_attrition",
        spec_anchor="FIGURE_REDRAW_SPEC.md#figure-s6",
        evidence_period=EvidencePeriod.TARGET,
        render_profile=RenderProfile(width_mm=FULL_WIDTH_MM, max_height_mm=182.0),
        panels=(
            PanelSpec("a", "denominator-preserving Sankey/waterfall",
                      "calendar opportunity -> eligible issue -> observed issue "
                      "WTEMP -> observed target WTEMP -> exact paired keys -> "
                      "retained stations -> reportable clusters; every stage a "
                      "distinct named denominator with its exclusion reason. "
                      "Absorbs the former main Figure 4 panel (b), which used the "
                      "same construction; the reportable-cluster stage is what it "
                      "added"),
            PanelSpec("b", "horizon-by-year/season opportunity heatmap",
                      "observable keys only; no all-calendar framing"),
            PanelSpec("c", "eight-candidate sensitivity dot plot",
                      "frozen order; formal effect retained as distinct reference"),
            PanelSpec("d", "retained-row block/feedback sensitivity",
                      "exact block semantics; feedback-date proxy role named"),
        ),
        required_value_ids=(
            "stage.calendar_opportunity", "stage.eligible_issue",
            "stage.observed_issue_wtemp", "stage.observed_target_wtemp",
            "stage.exact_paired_key", "stage.retained_station",
            "stage.completeness", "stage.missingness_reason",
            "stage.reportable_cluster", "stage.exclusion_reason",
            "sensitivity.year_season_block_id", "sensitivity.effect_score",
            "sensitivity.support", "sensitivity.formal_effect_reference",
            "sensitivity.feedback_date_proxy_role",
        ),
        dependencies=(
            Dependency("opening_receipt", "glob", OPENING_RECEIPT_GLOB),
            Dependency("si10_temporal_coverage", "file", SI10_RECEIPT, root="repo"),
            Dependency("si14_missingness_failures", "file", SI14_RECEIPT, root="repo"),
            Dependency("post_temporal_coverage_audit", "glob",
                       POST_TEMPORAL_COVERAGE_AUDIT,
                       note="denominator-preserving attrition stages"),
            Dependency("post_availability_registry", "glob",
                       POST_AVAILABILITY_REGISTRY),
        ),
        prohibited_semantics=(
            "seven retained rows called seven calendar days",
            "all-calendar performance", "real publication-latency replay",
            "favourable sensitivity rescuing the formal effect",
            "imputed unavailable outcomes",
        ),
        shares_value_ids_with=("fig04",),
        scope_band_value_id="figS6.scope.observable_keys_not_all_calendar",
    ),
    FigureSpec(
        figure_id="figS7",
        stem="figS7_spatial_leave_huc2",
        spec_anchor="FIGURE_REDRAW_SPEC.md#figure-s7",
        evidence_period=EvidencePeriod.TARGET,
        render_profile=RenderProfile(width_mm=FULL_WIDTH_MM, max_height_mm=156.0),
        panels=(
            PanelSpec("a", "per-HUC2 effect/count dot plot",
                      "all HUC2 groups appear or bind NA"),
            PanelSpec("b", "leave-one-HUC2 effect plot",
                      "frozen omission order; stability is not robustness proof"),
            PanelSpec("c", "cluster-share/effective-count diagnostics",
                      "gate thresholds and actual inputs visually distinct"),
            PanelSpec("d", "permanent claim-gate status box",
                      "scope warning visible inside the panel"),
        ),
        required_value_ids=(
            "cluster.definition_version", "unit.huc2_or_omitted",
            "counts.station", "counts.cluster", "largest_share",
            "effective_count", "effective_fraction", "per_huc.effect",
            "leave_one.effect", "interval_status", "gate.inputs",
            "gate.verdict", "registry_uq_binding",
        ),
        dependencies=(
            Dependency("opening_receipt", "glob", OPENING_RECEIPT_GLOB),
            Dependency("si11_spatial_sensitivity", "file", SI11_RECEIPT, root="repo"),
            Dependency("post_spatial_sensitivity", "glob", POST_SPATIAL_SENSITIVITY,
                       note="target-period per-HUC2 and leave-one-HUC2 effects"),
            Dependency("stage13c_region_transfer_table", "file", STAGE13C_TABLE),
            Dependency("stage13c_region_transfer_report", "file", STAGE13C_REPORT,
                       severity=Severity.QUALIFIER,
                       note="narrative companion to the region-transfer table"),
        ),
        prohibited_semantics=(
            "HUC2 as independent network component", "national inference",
            "superiority/non-inferiority", "leave-one stability as robustness proof",
            "post-outcome alternative clustering",
        ),
        # Expands fig04 panel (a): fig04 carries the aggregate per-HUC2 medians
        # and the two reference lines, S7 carries every unit and every
        # leave-one-HUC2 omission.  Shared per-HUC effects must agree in value,
        # unit and rounding.
        shares_value_ids_with=("fig04", "si11"),
        scope_band_value_id="figS7.scope.coarse_group_not_network",
    ),
    FigureSpec(
        figure_id="figS8",
        stem="figS8_qc_external_failures",
        spec_anchor="FIGURE_REDRAW_SPEC.md#figure-s8",
        evidence_period=EvidencePeriod.TARGET,
        render_profile=RenderProfile(width_mm=FULL_WIDTH_MM, max_height_mm=182.0),
        panels=(
            PanelSpec("a", "QC waterfall",
                      "raw response -> normalized series -> exact-A retained with qualifier/conflict categories; every count reconciles"),
            PanelSpec("b", "external-arm scope diagram and results",
                      "site-ID disjointness and history dependence, with the "
                      "in-panel clause 'site-ID disjoint, history-dependent; not "
                      "ungauged'; carries the six primary models x horizon on the "
                      "exact external-arm key set, absorbed from the former main "
                      "Figure 4 panel (d); paired effects or station-RMSE "
                      "distributions, never a river-network transfer map"),
            PanelSpec("c", "failure/attrition matrix",
                      "declared reason by disposition; adverse and non-estimable rows remain; exact SI12-SI14 tables preferred if illegible as one figure"),
        ),
        required_value_ids=(
            "qc.raw_request_response_series_ids", "qc.statistic_unit",
            "qc.qualifier_method_subset", "qc.raw_count", "qc.retained_count",
            "qc.excluded_count", "qc.conflict_count", "qc.exact_a_binding",
            "external.site_registry", "external.disjointness",
            "external.history_fields", "external.exact_keys",
            "external.scores_effects_status", "external.model_horizon",
            "external.site_count",
            "failure.reason", "failure.before_after_counts",
            "failure.disposition", "failure.source_pointers",
        ),
        dependencies=(
            Dependency("opening_receipt", "glob", OPENING_RECEIPT_GLOB),
            Dependency("stage25_receipt", "file", STAGE25_RECEIPT),
            Dependency("si12_qc_qualifiers", "file", SI12_RECEIPT, root="repo"),
            Dependency("si13_external_history_arm", "file", SI13_RECEIPT, root="repo"),
            Dependency("si14_missingness_failures", "file", SI14_RECEIPT, root="repo"),
            Dependency("post_outcome_qc_gate", "glob", POST_OUTCOME_QC_GATE,
                       note="panel (a) QC waterfall verdicts"),
            Dependency("post_outcome_quality_audit", "glob",
                       POST_OUTCOME_QUALITY_AUDIT,
                       note="qualifier/conflict counts"),
            Dependency("post_external_predictions", "glob",
                       POST_EXTERNAL_PREDICTIONS,
                       note="panel (b) external-arm results"),
        ),
        prohibited_semantics=(
            "site replacement", "post hoc threshold tuning",
            "suppressed conflicts or failed cases", "qualifier cherry-picking",
            "ungauged or river-network transfer", "national inference",
            "regulatory compliance", "external metadata as outcome authority",
        ),
        shares_value_ids_with=("si12", "si13", "si14"),
        scope_band_value_id="figS8.scope.external_not_ungauged",
    ),
    # ------------------------------------------------------------------
    # Figure S9 -- NEW 2026-08-05.
    #
    # Carries the Stage-22 adaptive-conformal work that the interim
    # Stage-19-independent plan built.  Once the trusted scorer was confirmed to
    # supply target-period interval metrics, this evidence stopped being a
    # substitute for Figure 3 and became what it always actually was: a
    # development-period sensitivity to the choice of conformal calibration
    # method.  It keeps that role, in SI, with a mandatory scope band.
    #
    # It is deliberately a SEPARATE figure rather than a panel of S5: S5 is
    # target-period (2021-2023) and this is development-period (2019-2020), and
    # one figure never mixes two evidence periods.
    #
    # This figure is optional.  If page budget is tight it can be dropped
    # without weakening any registered claim -- nothing in the confirmatory
    # five-test family depends on it.
    #
    # 2026-08-06: confirmed as the SOLE home of this evidence.  The benchmark
    # restructure proposed a "what calibration costs" panel inside target-period
    # fig04, carrying the split-CQR / block-maximum / delayed-ACI contrast.  That
    # is exactly panels (a) and (b) below, it is 2019-2020 evidence, and the
    # proposal is refused: one figure never mixes two evidence periods.  fig04
    # records the refusal as a DroppedPanel pointing here.
    # ------------------------------------------------------------------
    FigureSpec(
        figure_id="figS9",
        stem="figS9_conformal_sensitivity_development",
        spec_anchor="FIGURE_REDRAW_SPEC.md#figure-s9",
        evidence_period=EvidencePeriod.DEVELOPMENT,
        render_profile=RenderProfile(width_mm=FULL_WIDTH_MM, max_height_mm=156.0),
        panels=(
            PanelSpec(
                "a", "method-by-slice coverage matrix",
                "all five frozen conformal methods x all five slices (overall, "
                "warm train-q90 tail, leads 1/3/7 d); every cell bound or explicit "
                "NA with key counts; 0.90 nominal target named as a target",
                stage19_role=Stage19Role.NONE,
            ),
            PanelSpec(
                "b", "width and interval-score matrix with non-finite audit",
                "same grid for mean width and interval score; cells whose width is "
                "non-finite render as bound NA with their row count "
                "(44 / 782 / 4,068 at ACI gamma 0.005 / 0.02 / 0.05), never as a "
                "clipped or imputed number",
                stage19_role=Stage19Role.NONE,
            ),
            PanelSpec(
                "c", "per-HUC2 dispersion and weighting contrast",
                "coverage across all 15 HUC2 groups under equal-station weighting "
                "against unweighted row rates; the two are labelled as different "
                "quantities and never merged into one mark",
                stage19_role=Stage19Role.NONE,
            ),
            PanelSpec(
                "d", "ACI adaptation and idealized-feedback scope box",
                "feedback-update counts by gamma with the in-panel statement that "
                "target_date is an idealized feedback proxy and that no verified "
                "publication timestamp, revision, or latency record exists",
                stage19_role=Stage19Role.NONE,
            ),
        ),
        required_value_ids=(
            "conformal.method_id", "conformal.method_definition",
            "conformal.slice_id", "conformal.lead_days",
            "conformal.key_count", "conformal.site_count",
            "conformal.coverage_marginal", "conformal.mean_width_c",
            "conformal.interval_score_c", "conformal.nominal_target",
            "conformal.nonfinite_width_count", "conformal.nonfinite_reason",
            "conformal.aci_feedback_count", "conformal.feedback_proxy_role",
            "conformal.block_semantics_note",
            "weighting.station_balanced_rate", "weighting.row_count_rate",
            "weighting.station_weight_audit", "weighting.denominator_role",
            "huc2.unit_id", "huc2.station_count", "huc2.key_count",
            "huc2.coverage_marginal",
            "provenance.evidence_period", "provenance.target_start",
            "provenance.source_digest",
        ),
        dependencies=(
            Dependency("opening_receipt", "glob", OPENING_RECEIPT_GLOB),
            Dependency("stage22_row_table", "file", STAGE22_ROW_TABLE),
            Dependency("stage22_report", "file", STAGE22_REPORT),
            Dependency("confirmatory_protocol", "file", CONFIRMATORY_PROTOCOL,
                       root="repo"),
        ),
        prohibited_semantics=(
            "target-period coverage or any confirmatory interval claim",
            "conditional coverage",
            "distribution-free finite-sample guarantee",
            "adaptive conformal presented as cost-free",
            "silently dropped or clipped non-finite widths",
            "seven-calendar-day blocks (the block method uses 7 retained rows)",
            "real feedback latency or publication-vintage replay",
            "station-balanced and row-count rates merged into one mark",
            "comparison against the target-period intervals in Figure 4(c) or S5 as "
            "if the two were the same cohort",
        ),
        shares_value_ids_with=("stage22_conformal",),
        provenance_qualifiers=(
            "Panels bind Stage-22 development-period evidence "
            "(2019-01-01..2020-12-24) over 249,072 exact keys, 120 sites, 15 HUC2 "
            "groups, leads {1,3,7}. The confirmatory target period starts "
            "2021-01-01; no panel in this figure is a target-period result and no "
            "value here may be compared numerically with Figure 4(c) or S5.",
        ),
        scope_band_value_id="figS9.scope.development_period_not_confirmation",
    ),
    # ------------------------------------------------------------------
    # Figure S10 -- NEW 2026-08-06.
    #
    # Receives the architecture-intervention panel demoted from the former main
    # Figure 4(a) when the benchmark restructure made the architecture the
    # object under test rather than the contribution.  Demotion in prominence
    # only: the evidence, its period, and its gate are unchanged.
    #
    # It is TARGET-period, and that is not a judgement call.  The confirmatory
    # protocol's mandatory_exploratory_architecture_controls list is resolved
    # into the temporal cohort's required model set beside the six primary
    # models (opening._required_models), so the trusted scorer emits a row for
    # every control on the same exact common keys and Table 4.2 of the
    # manuscript transcribes them.  This figure is the graphical reading of
    # those rows.
    #
    # The Stage-09b information-matched controls (plain causal TCN, plain MLP)
    # are DEVELOPMENT-period.  They are a non-blocking qualifier here and a
    # table in SI09; they never fill a panel, and this figure declares no shared
    # value ID with the si09 namespace, which validate_manifest would refuse as
    # a cross-period share.
    # ------------------------------------------------------------------
    FigureSpec(
        figure_id="figS10",
        stem="figS10_architecture_interventions",
        spec_anchor="FIGURE_REDRAW_SPEC.md#figure-s10",
        evidence_period=EvidencePeriod.TARGET,
        render_profile=RenderProfile(width_mm=FULL_WIDTH_MM, max_height_mm=156.0),
        panels=(
            PanelSpec("a", "horizon-by-control dot matrix",
                      "all seven registered one-factor controls (prior-only, no "
                      "dynamic prior, fixed relaxation, no router, no mixture, no "
                      "TCN, unbounded residual); paired station-level RMSE "
                      "difference from full ThermoRoute in degrees C labelled "
                      "'negative favours the candidate'; seed and member "
                      "completeness per cell; no control selected after its value "
                      "was seen; no significance stars",
                      evidence_period=EvidencePeriod.TARGET),
            PanelSpec("b", "bounded-deviation audit",
                      "pointwise violation count and rate and the maximum "
                      "absolute correction against the configured delta over "
                      "every station-by-lead cell, with the denominator named and "
                      "the in-panel statement that the bound is relative to the "
                      "named anchor and is not a truth-error, ecological, "
                      "regulatory, or deployment-safety bound",
                      evidence_period=EvidencePeriod.TARGET),
        ),
        required_value_ids=(
            "control.model_id", "control.exact_intervention",
            "control.seed_member_registry", "control.key_digest",
            "control.horizon", "control.site_horizon_effect",
            "control.paired_target_count", "control.reportability_na_reason",
            "control.suite_receipt_lineage",
            "audit.bound_violation_count", "audit.bound_violation_rate",
            "audit.max_absolute_correction_c", "audit.configured_delta_c",
            "audit.station_by_lead_cell_count", "audit.denominator_role",
            "audit.non_safety_statement",
        ),
        dependencies=(
            Dependency("opening_receipt", "glob", OPENING_RECEIPT_GLOB),
            Dependency("stage09_receipt", "file", STAGE09_RECEIPT),
            Dependency("stage16_receipt", "file", STAGE16_RECEIPT),
            Dependency("si09_development_controls", "file", SI09_RECEIPT,
                       root="repo",
                       note="development-only companion table"),
            Dependency("confirmatory_protocol", "file", CONFIRMATORY_PROTOCOL,
                       root="repo"),
            Dependency("post_temporal_predictions", "glob",
                       POST_TEMPORAL_PREDICTIONS,
                       note="the seven target control rows on the exact common "
                            "keys"),
            Dependency("post_availability_registry", "glob",
                       POST_AVAILABILITY_REGISTRY,
                       note="exact-key / reportability denominators"),
            # Development-only by construction; it may never fill a panel.
            Dependency("stage09b_receipt", "file", STAGE09B_RECEIPT,
                       severity=Severity.QUALIFIER,
                       note="Stage-09b information-matched controls are "
                            "development-period; SI09 tabulation only, never a "
                            "coordinate in this figure"),
        ),
        prohibited_semantics=(
            "component necessity", "causal attribution",
            "capacity-matched claim unless the exact comparison proves it",
            "post hoc control selection",
            "a development-period control row on a target-period axis",
            "the deviation bound described as an error, safety, ecological, or "
            "regulatory bound",
            "rescue of a failed or unfavourable formal row",
            "significance coloring or stars",
        ),
        shares_value_ids_with=("table_t2",),
        provenance_qualifiers=(
            "Panel content demoted from the former main Figure 4(a) by the "
            "2026-08-06 benchmark restructure. The seven controls are part of the "
            "frozen temporal-cohort model registry, not an addition to it, so "
            "they are scored by the trusted scorer on the same exact common keys "
            "as the six primary models and are transcribed into manuscript Table "
            "4.2.",
        ),
        scope_band_value_id="figS10.scope.exploratory_not_component_necessity",
    ),
)


def validate_manifest(figures: tuple[FigureSpec, ...] = FIGURES) -> None:
    """Fail loudly if the in-module contract is internally inconsistent."""
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
            peer_period = NAMESPACE_EVIDENCE_PERIOD[peer]
            if (
                peer_period != figure.evidence_period
                and peer_period != EvidencePeriod.STRUCTURAL
            ):
                raise ManifestError(
                    f"{figure.figure_id} ({figure.evidence_period}) shares value "
                    f"IDs with {peer!r} ({peer_period}); one figure never mixes "
                    "two evidence periods, and neither does one value ID")

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
            if panel.stage19_role == Stage19Role.SUBSTITUTED and not panel.substitution_note:
                raise ManifestError(
                    f"{figure.figure_id}.{panel.panel_id}: substituted panel "
                    "needs a substitution_note")
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
    number = figure_id[len("fig"):]
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
    """Mark-level binder mirroring the PRE renderers' semantics.

    ``value`` and ``cell`` reproduce the closure in
    ``render_pre_supporting_figures.py`` exactly; ``mark`` is the enforcement
    point that routes every visible cell through ``cell``.
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
    print(f"{'figure':8} {'authority':32} {'sev':10} {'ok':4} detail")
    overall = 0
    for figure in FIGURES:
        for row in gate_report(figure, roots):
            flag = "yes" if row.ok else "no"
            print(f"{figure.figure_id:8} {row.authority_id:32} "
                  f"{row.severity:10} {flag:4} {row.detail}")
            if row.blocking:
                overall = 2
    print("# exit 2 = at least one REQUIRED gate is unmet (expected pre-POST); "
          "qualifier rows never block")
    return overall


def main() -> None:
    parser = argparse.ArgumentParser(description="POST skeleton for Figures S4-S10")
    parser.add_argument("--status", action="store_true",
                        help="read-only gate report; renders nothing")
    parser.add_argument("--figure", choices=[f.figure_id for f in FIGURES],
                        help="render one figure (POST only)")
    parser.add_argument("--evidence-root", default=None,
                        help="root holding outputs/ (defaults to "
                             f"${EVIDENCE_ROOT_ENV} or this worktree)")
    args = parser.parse_args()

    validate_manifest()
    roots = resolve_roots(args.evidence_root)

    if args.status:
        sys.exit(print_status(roots))

    targets = FIGURES if args.figure is None else tuple(
        f for f in FIGURES if f.figure_id == args.figure
    )
    for figure in targets:
        try:
            render_figure(figure, roots)
        except PostGateNotPassed as exc:
            print(f"REFUSED {figure.figure_id}: {exc}", file=sys.stderr)
            sys.exit(2)
        except PanelBuilderNotImplemented as exc:
            print(f"REFUSED {figure.figure_id}: {exc}", file=sys.stderr)
            sys.exit(2)
    print("All requested figures rendered.", file=sys.stderr)


if __name__ == "__main__":
    main()

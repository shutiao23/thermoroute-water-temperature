#!/usr/bin/env python3
"""Render the outcome-free PRE supporting figures FigS1--FigS3.

The renderer is deliberately fail-closed. FigS1 requires the frozen cohort
sources and outcome-free environmental audit; FigS2 requires the complete
outcome-free predictor-bridge binding chain; FigS3 asserts the protocol,
configuration, variable registry, and implementation invariants. No figure reads
target outcomes or a development-result cache.

Compatible with Matplotlib 3.8. SVG IDs, metadata, ordering, and text layout are
fixed so repeated renders from the bound inputs are byte-stable.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple, Sequence
from xml.etree import ElementTree

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = REPO / "data_usgs"

# One appearance for the whole submission: paper/figstyle.py owns the typeface
# chain, sizes, palette, widths and the constrained-layout defaults.  Importing
# it before any figure exists is deliberate -- rcParams set afterwards would not
# reach artists already created.
if str(REPO / "paper") not in sys.path:
    sys.path.insert(0, str(REPO / "paper"))
import figstyle  # noqa: E402  (must follow the sys.path bootstrap)

# Figure sizes come from figstyle.figsize(width_mm, height_mm); no local
# millimetre conversion is needed any more.
PNG_DPI = 600

# Okabe--Ito palette, taken from figstyle so one module owns the hexes.
# Neutral ink/grey come from the redraw spec's semantic tokens and are used only
# for structure and caveats.
OI = {
    "orange": figstyle.WONG["orange"],
    "sky": figstyle.WONG["sky"],
    "green": figstyle.WONG["green"],
    "yellow": figstyle.WONG["yellow"],
    "blue": figstyle.WONG["blue"],
    "vermillion": figstyle.WONG["vermillion"],
    "purple": figstyle.WONG["purple"],
    "ink": "#202020",
    "mid": "#666666",
}
SEMANTIC_TOKENS = {
    "TR_BLUE": "#0072B2",
    "TR_BLUE_LIGHT": "#DCEAF4",
    "ALLOWED_TEAL": "#008C7A",
    "ALLOWED_TEAL_LIGHT": "#DCEFEA",
    "WARNING_VERMILION": "#D55E00",
    "WARNING_LIGHT": "#F9E3D6",
    "NEUTRAL_INK": "#202020",
    "NEUTRAL_GRID": "#D0D0D0",
}

EXPECTED_SHA256 = {
    "station_registry_v1.csv":
        "090e7c0daf39ac38ceefeb1af8a12c178283e18347905d8e72ada969ad5460c9",
    "panel_usgs_120v2.parquet":
        "0427a07ea4514ba29ce7d0cf89594e6c35c7f9134cc4d1d96fdc90daeaf5ba69",
    "rejected_sites_120v2.csv":
        "bca746a03c41b4ca1112d02260c43f60faf82cf567aa4c5da31b5d98fddae6d6",
}

EXPECTED_REASONS = {
    "no NWIS WTEMP+FLOW": 950,
    "low full-period coverage": 376,
    "low blind-test-period coverage": 19,
}

EXPECTED_AUDIT_SHA256 = "c221bc67bd988da3bf5f2fcbe34b0e9e38646d7bc239bb130b46e4584f800883"
BRIDGE_MANIFEST = DATA / "development_predictor_bridge_v1.json"
BRIDGE_REPORT = DATA / "development_predictor_bridge_v1/bridge_report_v1.json"
BRIDGE_REQUEST_MAP = DATA / "development_predictor_bridge_v1/source_request_map_v1.json"
FIG1_BINDER = REPO / "paper/agu_submission/figures/fig01_preopening_concept.json"
EXPECTED_BRIDGE_MANIFEST_SHA256 = "4a8abbc92996ba933d68db1600a121eae85f0dbb7486532ff4fa265acfd8280f"
EXPECTED_BRIDGE_SOURCE_TREE_SHA256 = "d71197ab0a19059ba231e6109cdc4c84d065c2c45127a75e7a1237c840f6f761"
ROUTE_A_VARIABLES = ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP")
BRIDGE_FIELDS = ("TEMP", "PRCP", "RHMEAN", "DH", "WDSP")
HUC_MARKERS = ("o", "s", "^", "D", "v", "<", ">", "p")

# The shared style first, then only the two things it deliberately leaves to the
# caller: the SVG hash salt that makes repeated renders byte-stable, and the
# white background these SI figures are specified on.  Nothing here changes an
# appearance decision that figstyle owns.  The renderer previously declared
# ``font.family: DejaVu Sans`` outright, which is exactly the silent fallback
# figstyle exists to prevent; the font is now Nimbus Sans (Helvetica metrics).
figstyle.use()
mpl.rcParams.update(
    {
        "svg.hashsalt": "thermoroute-pre-figures-v1",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def verify_frozen_inputs() -> tuple[list[dict[str, str]], Counter[str]]:
    """Return registry rows and rejection counts after strict source checks."""

    for name, expected in EXPECTED_SHA256.items():
        actual = sha256(DATA / name)
        if actual != expected:
            raise RuntimeError(f"Refusing redraw: unexpected SHA-256 for {name}: {actual}")

    registry = csv_rows(DATA / "station_registry_v1.csv")
    rejected = csv_rows(DATA / "rejected_sites_120v2.csv")
    if len(registry) != 120 or len(rejected) != 1345:
        raise RuntimeError("Refusing redraw: expected 120 retained and 1,345 rejected rows")

    retained_ids = [row["site_no"] for row in registry]
    rejected_ids = [row["site"] for row in rejected]
    if len(set(retained_ids)) != 120 or len(set(rejected_ids)) != 1345:
        raise RuntimeError("Refusing redraw: duplicate site ID in a cohort ledger")
    if set(retained_ids) & set(rejected_ids):
        raise RuntimeError("Refusing redraw: retained/rejected site ledgers overlap")

    reasons = Counter(row["reason"] for row in rejected)
    if reasons != Counter(EXPECTED_REASONS):
        raise RuntimeError(f"Refusing redraw: unexpected rejection reasons: {dict(reasons)}")

    huc_counts = Counter(row["huc2"] for row in registry)
    if len(huc_counts) != 15 or sum(huc_counts.values()) != 120:
        raise RuntimeError("Refusing redraw: registry is not the frozen 120-site/15-HUC2 geometry")

    # The panel contributes no plotted value; its station identity binding is
    # nevertheless checked so the displayed retained cohort is panel-consistent.
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency failure is explicit
        raise RuntimeError("PyArrow is required to verify the frozen panel binding") from exc
    panel_path = DATA / "panel_usgs_120v2.parquet"
    panel_file = pq.ParquetFile(panel_path)
    if panel_file.metadata.num_rows != 657480:
        raise RuntimeError("Refusing redraw: unexpected panel row count")
    panel_ids = set(
        pc.unique(pq.read_table(panel_path, columns=["site_id"])["site_id"]).to_pylist()
    )
    registry_aliases = {row["legacy_site_id"] for row in registry}
    if panel_ids != registry_aliases or len(panel_ids) != 120:
        raise RuntimeError("Refusing redraw: registry/panel station identity mismatch")

    return registry, reasons


def verify_environmental_audit(
    registry: Sequence[dict[str, str]], reasons: Counter[str]
) -> dict[str, object]:
    path = DATA / "development_environmental_audit_v1.json"
    if sha256(path) != EXPECTED_AUDIT_SHA256:
        raise RuntimeError("Refusing redraw: environmental-audit SHA-256 drift")
    audit = json.loads(path.read_text(encoding="utf-8"))
    geography = audit["geography"]
    selection = audit["selection"]
    if audit["post_2020_values_read"] is not False:
        raise RuntimeError("Refusing redraw: environmental audit is not outcome-free")
    if geography["huc2_station_counts"] != {
        huc: count for huc, count in Counter(row["huc2"] for row in registry).items()
    }:
        raise RuntimeError("Refusing redraw: environmental-audit HUC2 counts disagree")
    corrected = selection["recorded_rejection_reason_counts_corrected"]
    expected_corrected = {
        "no NWIS WTEMP+FLOW": reasons["no NWIS WTEMP+FLOW"],
        "low full-period coverage": reasons["low full-period coverage"],
        "low 2019-2020 development-evaluation-period coverage":
            reasons["low blind-test-period coverage"],
    }
    if corrected != expected_corrected or selection["recorded_rejection_count"] != 1345:
        raise RuntimeError("Refusing redraw: environmental-audit selection mismatch")
    expected_sampling_interpretation = (
        "Availability-enriched convenience cohort, not a probability sample or a "
        "nationally representative sample of U.S. rivers."
    )
    if selection["interpretation"] != expected_sampling_interpretation:
        raise RuntimeError("Refusing redraw: environmental-audit sampling scope drift")
    expected_geo = {
        "unique_huc_code_count": 97,
        "stations_in_repeated_huc_code_groups": 38,
    }
    if any(geography[key] != value for key, value in expected_geo.items()):
        raise RuntimeError("Refusing redraw: environmental-audit geography drift")
    nearest = geography["nearest_station_distance_km"]
    if not (
        abs(nearest["median"] - 53.85601723110851) < 1e-12
        and abs(nearest["minimum"] - 0.7518401199511872) < 1e-12
        and nearest["stations_with_neighbor_within_10km"] == 19
    ):
        raise RuntimeError("Refusing redraw: nearest-station diagnostics drift")
    return audit


def verify_predictor_bridge() -> tuple[
    dict[str, object], dict[str, object], dict[str, object], dict[str, object]
]:
    if sha256(BRIDGE_MANIFEST) != EXPECTED_BRIDGE_MANIFEST_SHA256:
        raise RuntimeError("Refusing redraw: bridge-manifest SHA-256 drift")
    manifest = json.loads(BRIDGE_MANIFEST.read_text(encoding="utf-8"))
    report = json.loads(BRIDGE_REPORT.read_text(encoding="utf-8"))
    request_map = json.loads(BRIDGE_REQUEST_MAP.read_text(encoding="utf-8"))
    if set(manifest) != {
        "format", "gate", "normalized", "outcome_values_requested_or_read", "panel",
        "raw_snapshot_indexes", "registry", "report", "request_map",
        "source_tree_sha256", "status",
    }:
        raise RuntimeError("Refusing redraw: bridge-manifest schema drift")
    if manifest["source_tree_sha256"] != EXPECTED_BRIDGE_SOURCE_TREE_SHA256:
        raise RuntimeError("Refusing redraw: bridge source-tree binding drift")
    if set(manifest["normalized"]) != {"frozen", "refreshed"} or set(
        manifest["raw_snapshot_indexes"]
    ) != {"daymet", "gridmet", "gridmet_schema"}:
        raise RuntimeError("Refusing redraw: bridge nested binding registry drift")

    nested_bindings = {
        "panel": manifest["panel"],
        "registry": manifest["registry"],
        "report": manifest["report"],
        "request_map": manifest["request_map"],
        "normalized/frozen": manifest["normalized"]["frozen"],
        "normalized/refreshed": manifest["normalized"]["refreshed"],
        **{
            f"raw_snapshot_indexes/{key}": binding
            for key, binding in manifest["raw_snapshot_indexes"].items()
        },
    }
    for label, source_binding in nested_bindings.items():
        if set(source_binding) != {"path", "sha256"}:
            raise RuntimeError(f"Refusing redraw: malformed bridge binding {label}")
        bound_path = (REPO / source_binding["path"]).resolve()
        if REPO.resolve() not in bound_path.parents or not bound_path.is_file():
            raise RuntimeError(f"Refusing redraw: missing/escaping bridge binding {label}")
        if sha256(bound_path) != source_binding["sha256"]:
            raise RuntimeError(f"Refusing redraw: bridge binding digest mismatch {label}")
    if manifest["report"]["path"] != str(BRIDGE_REPORT.relative_to(REPO)):
        raise RuntimeError("Refusing redraw: bridge report path drift")
    if manifest["request_map"]["path"] != str(BRIDGE_REQUEST_MAP.relative_to(REPO)):
        raise RuntimeError("Refusing redraw: bridge request-map path drift")
    if {manifest["status"], report["status"]} != {"PASS_EXACT_PRODUCT_BRIDGE"}:
        raise RuntimeError("Refusing redraw: predictor bridge does not PASS")
    if any(obj["outcome_values_requested_or_read"] is not False
           for obj in (manifest, report, request_map)):
        raise RuntimeError("Refusing redraw: bridge evidence is not outcome-free")
    if report["interval"] != request_map["interval"] or report["interval"] != {
        "start": "2018-01-01", "end": "2020-12-31"
    }:
        raise RuntimeError("Refusing redraw: bridge interval drift")
    if (report["site_count"], report["row_count"], request_map["request_count"]) != (
        120, 131520, 120
    ):
        raise RuntimeError("Refusing redraw: bridge geometry drift")
    if request_map["request_count"] != len(request_map["requests"]) \
            or len({item["site_no"] for item in request_map["requests"]}) != 120:
        raise RuntimeError("Refusing redraw: bridge request-map count/identity drift")
    if tuple(report["fields"]) != tuple(sorted(BRIDGE_FIELDS)):
        raise RuntimeError("Refusing redraw: bridge report does not contain exactly five fields")
    if report["failures"] != [] or not isinstance(report["interpretation_limit"], str) \
            or not report["interpretation_limit"].strip():
        raise RuntimeError("Refusing redraw: bridge failure/interpretation contract drift")
    expected_limit = (
        "This gate tests product/parser compatibility and +/-1-day value alignment. "
        "It does not prove subdaily local-day equivalence with NWIS or as-issued availability."
    )
    if report["interpretation_limit"] != expected_limit:
        raise RuntimeError("Refusing redraw: bridge interpretation limit drift")
    for field, facts in report["fields"].items():
        if not (facts["exact_product_compatibility"] and facts["missing_pattern_exact"]
                and facts["zero_day_alignment_best_or_tied"]):
            raise RuntimeError(f"Refusing redraw: bridge field {field} failed")

    expected_index_counts = {"daymet": 120, "gridmet": 120, "gridmet_schema": 1}
    required_record_fields = {
        "byte_count", "metadata_byte_count", "metadata_path", "metadata_sha256",
        "provider", "request", "request_sha256", "response_path", "response_sha256",
        "retrieved_at_utc",
    }
    seen_request_hashes: set[str] = set()
    index_request_bindings: dict[str, dict[str, tuple[str, int]]] = {}
    index_summaries: dict[str, object] = {}
    for index_name, expected_count in expected_index_counts.items():
        index_binding = manifest["raw_snapshot_indexes"][index_name]
        index_path = (REPO / index_binding["path"]).resolve()
        index_parent = index_path.parent.resolve()
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if set(index) != {"schema_version", "snapshot_count", "records"}:
            raise RuntimeError(
                f"Refusing redraw: raw snapshot index schema drift at {index_name}"
            )
        records = index["records"]
        if index["schema_version"] != 2 or not isinstance(records, list) \
                or index["snapshot_count"] != len(records) \
                or len(records) != expected_count:
            raise RuntimeError(
                f"Refusing redraw: raw snapshot index count/version drift at {index_name}"
            )
        request_bindings: dict[str, tuple[str, int]] = {}
        for record_number, record in enumerate(records):
            if not isinstance(record, dict) or set(record) != required_record_fields:
                raise RuntimeError(
                    f"Refusing redraw: raw snapshot record schema drift at "
                    f"{index_name}[{record_number}]"
                )
            request_hash = record["request_sha256"]
            if not isinstance(request_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", request_hash):
                raise RuntimeError(
                    f"Refusing redraw: malformed request digest at {index_name}[{record_number}]"
                )
            if request_hash in seen_request_hashes:
                raise RuntimeError(
                    f"Refusing redraw: duplicate raw request digest at {index_name}[{record_number}]"
                )
            seen_request_hashes.add(request_hash)
            request_bindings[request_hash] = (
                record["response_sha256"], record["byte_count"]
            )
            if not isinstance(record["request"], dict) or not record["request"]:
                raise RuntimeError(
                    f"Refusing redraw: missing raw request object at {index_name}[{record_number}]"
                )
            for leaf_kind, path_key, digest_key, bytes_key in (
                ("response", "response_path", "response_sha256", "byte_count"),
                ("metadata", "metadata_path", "metadata_sha256", "metadata_byte_count"),
            ):
                relative_path = record[path_key]
                expected_digest = record[digest_key]
                expected_bytes = record[bytes_key]
                if not isinstance(relative_path, str) or not relative_path \
                        or not isinstance(expected_digest, str) \
                        or not re.fullmatch(r"[0-9a-f]{64}", expected_digest) \
                        or not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool) \
                        or expected_bytes < 0:
                    raise RuntimeError(
                        f"Refusing redraw: malformed {leaf_kind} binding at "
                        f"{index_name}[{record_number}]"
                    )
                leaf_path = (index_parent / relative_path).resolve()
                if index_parent not in leaf_path.parents or not leaf_path.is_file():
                    raise RuntimeError(
                        f"Refusing redraw: missing/escaping {leaf_kind} leaf at "
                        f"{index_name}[{record_number}]"
                    )
                if sha256(leaf_path) != expected_digest:
                    raise RuntimeError(
                        f"Refusing redraw: {leaf_kind} digest mismatch at "
                        f"{index_name}[{record_number}]"
                    )
                if leaf_path.stat().st_size != expected_bytes:
                    raise RuntimeError(
                        f"Refusing redraw: {leaf_kind} byte-count mismatch at "
                        f"{index_name}[{record_number}]"
                    )
        index_summaries[index_name] = {
            "schema_version": index["schema_version"],
            "declared_snapshot_count": index["snapshot_count"],
            "record_count": len(records),
            "leaf_file_count": 2 * len(records),
            "all_paths_contained_and_present": True,
            "all_sha256_digests_match": True,
            "all_byte_counts_match": True,
        }
        index_request_bindings[index_name] = request_bindings
    total_records = sum(item["record_count"] for item in index_summaries.values())
    raw_leaf_summary = {
        "status": "PASS_ALL_RAW_LEAVES_VERIFIED",
        "index_count": len(index_summaries),
        "record_count": total_records,
        "unique_request_sha256_count": len(seen_request_hashes),
        "all_request_sha256_unique": True,
        "leaf_file_count": 2 * total_records,
        "indexes": index_summaries,
    }
    if (total_records, len(seen_request_hashes), 2 * total_records) != (241, 241, 482):
        raise RuntimeError("Refusing redraw: aggregate raw snapshot leaf geometry drift")
    for provider_name in ("daymet", "gridmet"):
        mapped = {
            item[provider_name]["request_sha256"]: (
                item[provider_name]["response_sha256"], item[provider_name]["byte_count"]
            )
            for item in request_map["requests"]
        }
        if mapped != index_request_bindings[provider_name]:
            raise RuntimeError(
                f"Refusing redraw: request-map/raw-index chain drift at {provider_name}"
            )
    schema_contract = request_map["gridmet_provider_contract"]
    schema_binding = {
        schema_contract["request_sha256"]: (
            schema_contract["response_sha256"], schema_contract["byte_count"]
        )
    }
    if schema_binding != index_request_bindings["gridmet_schema"]:
        raise RuntimeError("Refusing redraw: gridMET schema request/raw-index chain drift")
    raw_leaf_summary["request_map_to_raw_indexes_match"] = True
    return manifest, report, request_map, raw_leaf_summary


def _literal_assignment(tree: ast.AST, name: str) -> object:
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.target.id == name:
            return ast.literal_eval(node.value)
    raise RuntimeError(f"Missing structural assignment: {name}")


def _class_literal_defaults(tree: ast.AST, class_name: str) -> dict[str, object]:
    class_node = next(
        (node for node in getattr(tree, "body", [])
         if isinstance(node, ast.ClassDef) and node.name == class_name),
        None,
    )
    if class_node is None:
        raise RuntimeError(f"Missing structural class: {class_name}")
    return {
        node.target.id: ast.literal_eval(node.value)
        for node in class_node.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        and node.value is not None
    }


def _display_year_interval(interval: Sequence[str]) -> str:
    start_year, end_year = str(interval[0])[:4], str(interval[1])[:4]
    return start_year if start_year == end_year else f"{start_year}–{end_year}"


def verify_temporal_source_contract(
    request_map: dict[str, object], bridge_report: dict[str, object]
) -> dict[str, object]:
    """Assert and project every source fact used by the FigS2 design."""

    config_path = REPO / "src/thermoroute/config.py"
    protocol_path = REPO / "protocols/route_a_confirmatory_v1.json"
    config_tree = ast.parse(config_path.read_text(encoding="utf-8"))
    split = _class_literal_defaults(config_tree, "TimeSplit")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    development = protocol["development_evidence"]
    holdout = protocol["time_holdout"]
    historical = protocol["primary_historical_input_contract"]
    daily = protocol["daily_outcome_quality_contract"]

    facts: dict[str, object] = {
        "horizons": tuple(_literal_assignment(config_tree, "HORIZONS")),
        "splits": {key: tuple(split[key]) for key in ("train", "val", "calib", "test")},
        "development_role": development["existing_2019_2020_role"],
        "development_reproducibility_scope": development["reproducibility_scope"],
        "target_start": holdout["start"],
        "primary_target_start": holdout["primary_target_start"],
        "target_end": holdout["end"],
        "issue_and_target_inside": holdout["issue_and_all_target_dates_must_remain_inside"],
        "evaluation_type": historical["evaluation_type"],
        "information_cutoff": historical["information_cutoff"],
        "horizon_specific_future_nwp_consumed":
            historical["horizon_specific_future_nwp_consumed"],
        "operational_replay_claim_allowed": historical["operational_replay_claim_allowed"],
        "autoregressive_inputs": tuple(historical["autoregressive_inputs"]),
        "meteorological_inputs": tuple(historical["retrospective_meteorological_inputs"]),
        "meteorological_time_support": historical["meteorological_time_support"],
        "provisional_vintage_limitation": historical["provisional_vintage_limitation"],
        "nwis_provider": daily["provider"],
    }
    expected = {
        "horizons": (1, 3, 7),
        "splits": {
            "train": ("2006-01-01", "2015-12-31"),
            "val": ("2016-01-01", "2017-12-31"),
            "calib": ("2018-01-01", "2018-12-31"),
            "test": ("2019-01-01", "2020-12-31"),
        },
        "development_role": "EXPLORATORY_NOT_BLIND",
        "development_reproducibility_scope": (
            "conditional on the frozen canonical panel bytes; the original WTEMP/FLOW/Daymet/"
            "gridMET HTTP responses, requests, and retrieval timestamps used to create that "
            "legacy panel were not retained and cannot be represented as source-level replay "
            "evidence"
        ),
        "target_start": "2021-01-01",
        "primary_target_start": "2021-01-01",
        "target_end": "2023-12-31",
        "issue_and_target_inside": True,
        "evaluation_type": "ONE_SHOT_RETROSPECTIVE_HISTORICAL_INFORMATION",
        "information_cutoff": "issue_date_end; no target-date or post-issue values",
        "horizon_specific_future_nwp_consumed": False,
        "operational_replay_claim_allowed": False,
        "autoregressive_inputs": ("WTEMP", "FLOW"),
        "meteorological_inputs": ("TEMP", "PRCP", "RHMEAN", "DH", "WDSP"),
        "meteorological_time_support": "issue_date and earlier only",
        "nwis_provider": "USGS NWIS Daily Values service",
    }
    for key, expected_value in expected.items():
        if facts[key] != expected_value:
            raise RuntimeError(f"Refusing redraw: temporal source contract drift at {key}")
    expected_vintage_limit = (
        "The as-issued provisional vintage that would have existed on each historical issue "
        "date cannot be reconstructed from the retrospective gridded products. Frozen retrieval "
        "responses prove the one-shot retrospective dataset, not an operationally available vintage."
    )
    if facts["provisional_vintage_limitation"] != expected_vintage_limit:
        raise RuntimeError("Refusing redraw: retrospective vintage limitation drift")

    daymet_fields = tuple(
        bridge_report["daymet_calendar_attestation"]
        ["refreshed_missing_count_on_those_dates"].keys()
    )
    if set(daymet_fields) != {"TEMP", "PRCP", "RHMEAN", "DH"}:
        raise RuntimeError("Refusing redraw: Daymet bridge field registry drift")
    gridmet_fields = tuple(field for field in BRIDGE_FIELDS if field not in daymet_fields)
    if gridmet_fields != ("WDSP",):
        raise RuntimeError("Refusing redraw: gridMET bridge field registry drift")
    if set(request_map["gridmet_provider_contract"]) != {
        "add_offset", "byte_count", "request_sha256", "response_sha256",
        "retrieved_at_utc", "scale_factor", "units",
    } or request_map["gridmet_provider_contract"]["units"] != "m/s":
        raise RuntimeError("Refusing redraw: gridMET provider contract drift")
    for request in request_map["requests"]:
        if not ({"daymet", "gridmet"} <= set(request)):
            raise RuntimeError("Refusing redraw: provider request-map entry drift")
        if request["contains_outcome"] or request["contains_outcome_labels"]:
            raise RuntimeError("Refusing redraw: provider request map contains outcomes")

    provider_by_variable = {
        variable: (
            "NWIS" if variable in facts["autoregressive_inputs"]
            else "Daymet" if variable in daymet_fields
            else "gridMET" if variable in gridmet_fields
            else "UNBOUND"
        )
        for variable in ROUTE_A_VARIABLES
    }
    if "UNBOUND" in provider_by_variable.values():
        raise RuntimeError("Refusing redraw: derived provider matrix is incomplete")
    facts["provider_by_variable"] = provider_by_variable
    facts["issue_date_symbol"] = "t"
    facts["target_date_expression"] = "t+h"
    facts["source_date_display"] = "date ≤ t"
    facts["retrieval_vintage_display"] = "latest values served at retrieval"
    facts["chronology"] = (
        {"key": "train", "interval": facts["splits"]["train"], "role": "TRAIN"},
        {"key": "validation", "interval": facts["splits"]["val"], "role": "VALIDATION"},
        {"key": "calibration", "interval": facts["splits"]["calib"], "role": "CALIBRATION"},
        {"key": "exploratory", "interval": facts["splits"]["test"],
         "role": "INSPECTED_EXPLORATORY"},
        {"key": "target", "interval": (facts["target_start"], facts["target_end"]),
         "role": "ONE_TIME_TARGET"},
    )
    return facts


def verify_architecture_sources() -> dict[str, object]:
    config_path = REPO / "src/thermoroute/config.py"
    suite_path = REPO / "src/thermoroute/model_suite.py"
    model_path = REPO / "src/thermoroute/thermoroute.py"
    features_path = REPO / "src/thermoroute/features.py"
    training_path = REPO / "src/thermoroute/train.py"
    protocol_path = REPO / "protocols/route_a_confirmatory_v1.json"
    config_tree = ast.parse(config_path.read_text(encoding="utf-8"))
    suite_tree = ast.parse(suite_path.read_text(encoding="utf-8"))
    model_tree = ast.parse(model_path.read_text(encoding="utf-8"))
    features_tree = ast.parse(features_path.read_text(encoding="utf-8"))
    training_tree = ast.parse(training_path.read_text(encoding="utf-8"))
    defaults = _class_literal_defaults(config_tree, "TrainConfig")
    time_split = _class_literal_defaults(config_tree, "TimeSplit")
    erratum_path = REPO / "protocols/route_a_calibration_inclusion_erratum_v1.json"
    erratum = json.loads(erratum_path.read_text(encoding="utf-8"))
    facts = {
        "variables": tuple(_literal_assignment(suite_tree, "STAGE9_USGS_VARIABLES")),
        "context_length": _literal_assignment(config_tree, "CONTEXT_LENGTH"),
        "max_router_lag": _literal_assignment(config_tree, "MAX_ROUTER_LAG"),
        "delta_scale": _literal_assignment(config_tree, "DELTA_SCALE"),
        "tcn_blocks": defaults["encoder_blocks"],
        "tcn_kernel": defaults["kernel_size"],
        "quantiles": tuple(_literal_assignment(config_tree, "QUANTILES")),
        "calibration_interval": tuple(time_split["calib"]),
    }
    facts["tcn_receptive_field"] = 1 + (facts["tcn_kernel"] - 1) * sum(
        2 ** index for index in range(facts["tcn_blocks"])
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    contract = protocol["primary_inference_contract"]
    expected = {
        "variables": ROUTE_A_VARIABLES,
        "context_length": 32,
        "max_router_lag": 14,
        "delta_scale": 1.0,
        "tcn_blocks": 2,
        "tcn_kernel": 3,
        "tcn_receptive_field": 7,
        "quantiles": (0.05, 0.50, 0.95),
        "calibration_interval": ("2018-01-01", "2018-12-31"),
    }
    if facts != expected:
        raise RuntimeError(f"Refusing redraw: architecture configuration drift: {facts}")
    if tuple(contract["feature_order"]) != ROUTE_A_VARIABLES or contract["wlevel_consumed"]:
        raise RuntimeError("Refusing redraw: protocol feature/WLEVEL contract drift")
    primary_estimand = contract["primary_estimand"]
    probability = contract["probabilistic_event_contract"]
    if "retain missingness masks" not in primary_estimand["admissible_issue"]:
        raise RuntimeError("Refusing redraw: missing-mask protocol contract drift")
    expected_probability = {
        "event_calibration": (
            "one frozen Platt calibrator per horizon fitted only on 2018 calibration forecasts; "
            "no confirmation label or confirmation event rate is used"
        ),
        "interval_calibration": (
            "frozen 2018 CQR offset by station and horizon for the temporal cohort and pooled by "
            "horizon for the external cohort"
        ),
    }
    if any(probability[key] != expected_value for key, expected_value in expected_probability.items()):
        raise RuntimeError("Refusing redraw: CQR/Platt protocol contract drift")
    if (
        erratum["status"] != "DRAFT_PRELABEL_OUTCOME_FREE"
        or erratum["prelabel_attestation"]["confirmation_outcomes_requested_or_inspected"]
        or "C.SPLIT.calib=(2018-01-01, 2018-12-31)"
            not in erratum["correction"]["new_development_rule"]
        or "CQR_purge_of_target_dates_after_2018-12-31"
            not in erratum["correction"]["unchanged"]
        or set(erratum["defect"]["affected_artifacts"][:2]) != {
            "frozen_2018_CQR_offsets", "frozen_2018_horizon_Platt_calibrators"
        }
    ):
        raise RuntimeError("Refusing redraw: calibration erratum contract drift")
    def assigned_values(tree: ast.AST, target_text: str) -> list[ast.AST]:
        values: list[ast.AST] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(ast.unparse(target) == target_text for target in targets) \
                        and node.value is not None:
                    values.append(node.value)
        return values

    def expression_matches(node: ast.AST, expression: str) -> bool:
        expected_node = ast.parse(expression, mode="eval").body
        return ast.dump(node, include_attributes=False) == ast.dump(
            expected_node, include_attributes=False
        )

    model_classes = {
        node.name for node in model_tree.body if isinstance(node, ast.ClassDef)
    }
    moe_values = assigned_values(model_tree, "self.moe")
    learned_prior_values = assigned_values(model_tree, "self.prior")
    prior_values = assigned_values(model_tree, "(internal_prior, kappa, teq)")
    anchor_values = assigned_values(model_tree, "anchor")
    topology_assertions = {
        "regime_moe_class_and_self_moe": (
            "RegimeMoE" in model_classes
            and any(
                isinstance(child, ast.Call) and ast.unparse(child.func) == "RegimeMoE"
                for value_node in moe_values for child in ast.walk(value_node)
            )
        ),
        "learned_prior_class_and_self_prior": (
            "DynamicThermalRelaxationPrior" in model_classes
            and any(
                isinstance(child, ast.Call)
                and ast.unparse(child.func) == "DynamicThermalRelaxationPrior"
                for value_node in learned_prior_values for child in ast.walk(value_node)
            )
        ),
        "router_and_left_looking_tcn_classes": (
            {"DynamicLagRouter", "CausalTCN"} <= model_classes
        ),
        "missingness_mask_is_consumed": (
            bool(assigned_values(model_tree, "self.missing_token"))
            and any(
                isinstance(node, ast.Subscript)
                and expression_matches(node, "batch['Mask']")
                for node in ast.walk(model_tree)
            )
        ),
        "learned_prior_outputs_internal_prior_kappa": any(
            expression_matches(value_node, "self.prior(batch)") for value_node in prior_values
        ),
        "frozen_damped_prior_is_anchor": any(
            expression_matches(value_node, "batch['damped_prior']")
            for value_node in anchor_values
        ),
        "distinct_point_q50_event_heads": all(
            assigned_values(model_tree, target)
            and any(
                isinstance(child, ast.Call) and ast.unparse(child.func) == "nn.Linear"
                for value_node in assigned_values(model_tree, target)
                for child in ast.walk(value_node)
            )
            for target in ("self.head_delta", "self.head_q50", "self.head_evt")
        ),
        "point_anchor_correction_identity": any(
            expression_matches(value_node, "anchor + point_correction")
            for value_node in assigned_values(model_tree, "point")
        ),
        "q50_anchor_correction_identity": any(
            expression_matches(value_node, "anchor + q50_correction")
            for value_node in assigned_values(model_tree, "q50")
        ),
        "q05_q95_positive_width_construction": (
            any(expression_matches(value_node, "q50 - lo")
                for value_node in assigned_values(model_tree, "q05"))
            and any(expression_matches(value_node, "q50 + hi")
                    for value_node in assigned_values(model_tree, "q95"))
            and bool(assigned_values(model_tree, "lo"))
            and bool(assigned_values(model_tree, "hi"))
        ),
        "event_head_emits_score": any(
            any(isinstance(child, ast.Attribute) and child.attr == "head_evt"
                for child in ast.walk(value_node))
            for value_node in assigned_values(model_tree, "evt")
        ),
        "point_head_mse_contract": any(
            expression_matches(
                value_node, "torch.mean((y - out.point) ** 2) / scale_sq_t"
            )
            for value_node in assigned_values(training_tree, "point")
        ),
    }

    damped_class = next(
        (node for node in features_tree.body
         if isinstance(node, ast.ClassDef) and node.name == "DampedPersistenceAnchor"),
        None,
    )
    damped_predict = next(
        (node for node in damped_class.body
         if isinstance(node, ast.FunctionDef) and node.name == "predict"),
        None,
    ) if damped_class is not None else None
    damped_return = next(
        (node.value for node in ast.walk(damped_predict) if isinstance(node, ast.Return)),
        None,
    ) if damped_predict is not None else None
    topology_assertions["damped_anchor_fitted_blend_formula"] = (
        damped_return is not None
        and expression_matches(
            damped_return,
            "np.asarray(clim_tgt, dtype=float) + self.phi[station] ** h * "
            "(wtemp_t - clim_t)",
        )
    )
    if not all(topology_assertions.values()):
        failed = sorted(key for key, passed in topology_assertions.items() if not passed)
        raise RuntimeError(
            "Refusing redraw: model topology/source assertions failed: " + ", ".join(failed)
        )
    facts["missingness_mask"] = True
    facts["calibration_display"] = _display_year_interval(facts["calibration_interval"])
    facts["interval_calibration_contract"] = probability["interval_calibration"]
    facts["event_calibration_contract"] = probability["event_calibration"]
    facts["calibration_erratum_status"] = erratum["status"]
    facts["topology_assertions"] = topology_assertions
    facts["anchor_identity"] = "ANCHOR A"
    facts["anchor_method"] = "frozen damped-persistence anchor"
    facts["anchor_composition"] = "fitted blend of last y and climatology"
    return facts


class Rect(NamedTuple):
    """An axes-fraction rectangle.  Every module in these figures is one."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return 0.5 * (self.x0 + self.x1)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y0 + self.y1)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(x, y, width, height), the form :func:`guarded_text` registers."""
        return (self.x0, self.y0, self.width, self.height)

    def inset(self, dx: float, dy: float | None = None) -> "Rect":
        dy = dx if dy is None else dy
        return Rect(self.x0 + dx, self.y0 + dy, self.x1 - dx, self.y1 - dy)

    def contains_point(self, x: float, y: float, *, slack: float = 0.0) -> bool:
        return (self.x0 - slack <= x <= self.x1 + slack
                and self.y0 - slack <= y <= self.y1 + slack)


def split_rows(rect: Rect, count: int, *, top_pad: float = 0.0,
               bottom_pad: float = 0.0) -> list[float]:
    """Centres of ``count`` equal rows inside ``rect``, top row first.

    Every stack of lines in these figures is derived from this rather than from
    a hand-tuned ``y = 0.66 - i * 0.15``.  When a panel changes height the rows
    follow, so a stack cannot walk out of its module or into its neighbour.
    """

    if count <= 0:
        return []
    top = rect.y1 - top_pad
    bottom = rect.y0 + bottom_pad
    step = (top - bottom) / count
    return [top - step * (index + 0.5) for index in range(count)]


def split_columns(x0: float, x1: float, count: int, *, gutter: float) -> list[Rect]:
    """``count`` equal columns spanning ``[x0, x1]`` separated by ``gutter``."""

    width = (x1 - x0 - gutter * (count - 1)) / count
    return [Rect(x0 + index * (width + gutter), 0.0,
                 x0 + index * (width + gutter) + width, 1.0)
            for index in range(count)]


def clean_axis(ax: mpl.axes.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel_heading(ax: mpl.axes.Axes, letter: str, title: str) -> None:
    """Bold ``(x)`` from the shared style, then the panel title beside it.

    The label goes through :func:`figstyle.panel_label` so every figure in the
    submission places it identically; the title is a real Axes title, so
    constrained layout reserves its height instead of the caller guessing.
    """

    figstyle.panel_label(ax, f"({letter})")
    ax.set_title(title, loc="left", x=0.085, pad=4.0)


def box(
    ax: mpl.axes.Axes,
    rect: Rect,
    *,
    facecolor: str = "white",
    edgecolor: str = OI["ink"],
    radius: float = 0.004,
    linewidth: float = 1.0,
    hatch: str | None = None,
    flag: bool = False,
    zorder: float = 1.0,
) -> FancyBboxPatch:
    """Draw a module rectangle.

    ``flag`` replaces a whole-box hatch with a narrow hatched strip inside the
    left edge.  The redundancy that makes the category readable without colour
    is kept, but the label no longer sits on top of hatch lines -- the single
    worst legibility fault in the previous draft.
    """

    patch = FancyBboxPatch(
        (rect.x0, rect.y0),
        rect.width,
        rect.height,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        hatch=None if flag else hatch,
        clip_on=False,
        zorder=zorder,
    )
    ax.add_patch(patch)
    if flag and hatch:
        strip = min(0.030, rect.width * 0.16)
        ax.add_patch(Rectangle(
            (rect.x0, rect.y0), strip, rect.height, transform=ax.transAxes,
            facecolor="none", edgecolor=edgecolor, hatch=hatch, linewidth=0.0,
            clip_on=False, zorder=zorder + 0.1,
        ))
        ax.plot([rect.x0 + strip, rect.x0 + strip], [rect.y0, rect.y1],
                transform=ax.transAxes, color=edgecolor, linewidth=0.6,
                zorder=zorder + 0.1, clip_on=False)
    return patch


def guarded_text(
    ax: mpl.axes.Axes,
    bounds: "Rect | tuple[float, float, float, float]",
    label: str,
    x: float,
    y: float,
    text_value: str,
    dimensions: str = "xy",
    **kwargs: object,
) -> mpl.text.Text:
    """Add text whose final bbox must stay at least 2 mm inside a module."""

    if isinstance(bounds, Rect):
        bounds = bounds.bounds
    artist = ax.text(x, y, text_value, transform=ax.transAxes, **kwargs)
    guards = getattr(ax.figure, "_thermoroute_text_guards", [])
    guards.append((artist, ax, bounds, label, 2.0, dimensions))
    ax.figure._thermoroute_text_guards = guards
    return artist


def validate_text_layout(fig: mpl.figure.Figure) -> None:
    """Fail closed on canvas clipping or registered module-boundary overflow."""

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_bbox = fig.bbox
    failures: list[str] = []
    for artist in fig.findobj(mpl.text.Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        bbox = artist.get_window_extent(renderer=renderer)
        if (bbox.x0 < figure_bbox.x0 - 0.5 or bbox.y0 < figure_bbox.y0 - 0.5
                or bbox.x1 > figure_bbox.x1 + 0.5 or bbox.y1 > figure_bbox.y1 + 0.5):
            failures.append(f"canvas:{artist.get_text()!r}")
    margin_px = fig.dpi * 2.0 / 25.4
    for artist, ax, (x, y, width, height), label, _, dimensions in getattr(
        fig, "_thermoroute_text_guards", []
    ):
        (x0, y0), (x1, y1) = ax.transAxes.transform([(x, y), (x + width, y + height)])
        container = mpl.transforms.Bbox.from_extents(
            min(x0, x1) + margin_px,
            min(y0, y1) + margin_px,
            max(x0, x1) - margin_px,
            max(y0, y1) - margin_px,
        )
        bbox = artist.get_window_extent(renderer=renderer)
        x_ok = bbox.x0 >= container.x0 and bbox.x1 <= container.x1
        y_ok = bbox.y0 >= container.y0 and bbox.y1 <= container.y1
        if not ((x_ok or "x" not in dimensions) and (y_ok or "y" not in dimensions)):
            failures.append(
                f"module:{label}:{artist.get_text()!r}:"
                f"text={tuple(round(v, 2) for v in bbox.extents)}:"
                f"safe={tuple(round(v, 2) for v in container.extents)}"
            )
    if failures:
        raise RuntimeError("Text bbox QA failed:\n" + "\n".join(failures))


def arrow_axes(
    ax: mpl.axes.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = OI["mid"],
    linewidth: float = 1.0,
    mutation_scale: float = 8,
    waypoints: Sequence[tuple[float, float]] = (),
    register: bool = True,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Draw a straight or orthogonally routed connector in axes coordinates.

    ``waypoints`` turns the connector into a polyline: the plain segments are
    drawn as a line and only the final leg carries the head, so a route that
    has to leave a column and come back does not become a curve sliding under
    the modules it passes.  Every segment is registered on the figure so
    :func:`validate_diagram_geometry` can prove it never enters a module.
    """

    points = [tuple(start), *[tuple(point) for point in waypoints], tuple(end)]
    if len(points) > 2:
        ax.plot([point[0] for point in points[:-1]],
                [point[1] for point in points[:-1]],
                transform=ax.transAxes, color=color, linewidth=linewidth,
                solid_capstyle="round", solid_joinstyle="round",
                zorder=3, clip_on=False)
    ax.add_patch(
        FancyArrowPatch(
            points[-2],
            points[-1],
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            color=color,
            shrinkA=0,
            shrinkB=0,
            clip_on=False,
            zorder=3,
        )
    )
    segments = list(zip(points[:-1], points[1:]))
    if register:
        routes = getattr(ax.figure, "_thermoroute_routes", [])
        routes.extend((ax, a, b) for a, b in segments)
        ax.figure._thermoroute_routes = routes
    return segments


def _segment_enters(rect: Rect, a: tuple[float, float], b: tuple[float, float],
                    *, slack: float) -> bool:
    """True when segment ``a``--``b`` has a point strictly inside ``rect``.

    Sampling is enough here and is far easier to trust than a clipping
    routine: every connector in these figures is either axis-aligned or a short
    straight run inside a gutter, so a point on 200 evenly spaced stations
    cannot miss an incursion that matters at 140 mm.
    """

    inner = rect.inset(slack)
    if inner.width <= 0 or inner.height <= 0:
        return False
    steps = 200
    for index in range(steps + 1):
        t = index / steps
        x = a[0] + (b[0] - a[0]) * t
        y = a[1] + (b[1] - a[1]) * t
        if inner.x0 < x < inner.x1 and inner.y0 < y < inner.y1:
            return True
    return False


def register_module(ax: mpl.axes.Axes, name: str, rect: Rect) -> Rect:
    """Record a module rectangle for the diagram geometry check.

    Registration is per-Axes because the rectangles are in axes fractions:
    panel (a)'s 0.5 and panel (d)'s 0.5 are different places, so comparing
    them would invent collisions.
    """

    modules = getattr(ax.figure, "_thermoroute_modules", [])
    modules.append((ax, name, rect))
    ax.figure._thermoroute_modules = modules
    return rect


def validate_diagram_geometry(fig: mpl.figure.Figure) -> None:
    """Fail closed when modules overlap or a connector runs through one.

    Text collisions are only half of what goes wrong in a hand-placed
    schematic.  The other half -- a box landing on a box, and an arrow taking a
    short cut across a module it has nothing to do with -- is invisible to a
    text-bbox check, so it is asserted here on the declared geometry.
    """

    modules = getattr(fig, "_thermoroute_modules", [])
    failures: list[str] = []
    for index, (ax_a, name_a, rect_a) in enumerate(modules):
        for ax_b, name_b, rect_b in modules[index + 1:]:
            if ax_a is not ax_b:
                continue
            dx = min(rect_a.x1, rect_b.x1) - max(rect_a.x0, rect_b.x0)
            dy = min(rect_a.y1, rect_b.y1) - max(rect_a.y0, rect_b.y0)
            if dx > 1e-9 and dy > 1e-9:
                failures.append(f"module-overlap:{name_a}|{name_b}")
    # A connector legitimately touches the two modules it joins, so an
    # endpoint on a module edge is allowed; anything reaching into the body of
    # a module is not.
    for route_ax, a, b in getattr(fig, "_thermoroute_routes", []):
        for module_ax, name, rect in modules:
            if module_ax is not route_ax:
                continue
            if _segment_enters(rect, a, b, slack=0.004):
                failures.append(
                    f"connector-through-module:{name}:"
                    f"{tuple(round(v, 3) for v in a)}->{tuple(round(v, 3) for v in b)}"
                )
    if failures:
        raise RuntimeError("Diagram geometry QA failed:\n" + "\n".join(sorted(set(failures))))


def save_figure(fig: mpl.figure.Figure, path: Path, title: str, description: str) -> None:
    """Save deterministic SVG/PDF/PNG deliverables and SVG accessibility text."""

    # Three independent gates, cheapest first.  figstyle.save with no formats
    # runs the shared submission-wide text-collision check and raises on any
    # overlapping pair; validate_text_layout adds this renderer's stricter
    # module-inset and canvas-containment rules; validate_diagram_geometry
    # covers what neither can see -- boxes and connectors.
    figstyle.save(fig, path.stem, path.parent, formats=())
    validate_text_layout(fig)
    validate_diagram_geometry(fig)
    fig.savefig(path, format="svg",
                metadata={"Creator": "ThermoRoute PRE supporting-figure renderer", "Date": None},
                facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), format="pdf",
                metadata={"Creator": "ThermoRoute PRE supporting-figure renderer",
                          "CreationDate": None, "ModDate": None},
                facecolor="white")
    fig.savefig(path.with_suffix(".png"), format="png", dpi=PNG_DPI,
                metadata={"Software": "ThermoRoute PRE supporting-figure renderer"},
                facecolor="white")
    plt.close(fig)
    svg = path.read_text(encoding="utf-8")
    start = svg.index("<svg ")
    svg = svg[:start + 4] + ' role="img" aria-labelledby="svg-title svg-desc"' + svg[start + 4:]
    start = svg.index("<svg ")
    root_end = svg.index(">", start) + 1
    accessible = (f"\n <title id=\"svg-title\">{title}</title>"
                  f"\n <desc id=\"svg-desc\">{description}</desc>")
    svg = svg[:root_end] + accessible + svg[root_end:]
    path.write_text(svg, encoding="utf-8", newline="\n")
    stroke_widths = [
        float(value)
        for value in re.findall(
            r"stroke-width\s*(?::|=)\s*[\"']?([0-9]*\.?[0-9]+)", svg
        )
    ]
    subminimum = [width for width in stroke_widths if 0 < width < 0.6 - 1e-12]
    if subminimum:
        raise RuntimeError(
            f"SVG stroke-width QA failed for {path.name}: minimum {min(subminimum)} pt"
        )


def render_fig_s1(
    registry: Sequence[dict[str, str]], reasons: Counter[str], audit: dict[str, object]
) -> Path:
    fig = plt.figure(figsize=figstyle.figsize(figstyle.FULL_MM, 152.0))
    # Constrained layout owns the margins.  The previous draft set left/right/
    # top/bottom and an hspace of 0.88 by hand to open room for panel headings
    # and panel (b)'s legend; the legend now has a cell of its own, so the
    # layout engine reserves exactly the space each panel needs.
    gs = fig.add_gridspec(2, 2, height_ratios=[1.00, 0.86], hspace=0.16, wspace=0.20)
    map_gs = gs[0, 1].subgridspec(2, 1, height_ratios=[1.0, 0.36], hspace=0.05)

    ax = fig.add_subplot(gs[0, 0])
    clean_axis(ax)
    panel_heading(ax, "a", "Cohort ledger")
    initial_rect = register_module(ax, "S1a initial", Rect(0.02, 0.855, 0.98, 0.975))
    box(ax, initial_rect, facecolor=SEMANTIC_TOKENS["TR_BLUE_LIGHT"],
        edgecolor=OI["blue"])
    guarded_text(ax, initial_rect, "S1a initial", initial_rect.cx, initial_rect.cy,
                 "Initial candidates   n = 1,465", ha="center", va="center",
                 fontsize=8.0, fontweight="bold", color=OI["blue"])

    rejected_rect = register_module(ax, "S1a rejected", Rect(0.02, 0.455, 0.98, 0.795))
    box(ax, rejected_rect, facecolor=SEMANTIC_TOKENS["WARNING_LIGHT"],
        edgecolor=OI["vermillion"], hatch="///", flag=True)
    arrow_axes(ax, (initial_rect.cx, initial_rect.y0), (rejected_rect.cx, rejected_rect.y1),
               color=OI["mid"], mutation_scale=7)

    reason_lines = [
        ("X", reasons["no NWIS WTEMP+FLOW"], "no NWIS WTEMP + FLOW"),
        ("^", reasons["low full-period coverage"], "low full-period coverage"),
        ("s", reasons["low blind-test-period coverage"], "low 2019–20 coverage"),
    ]
    reason_rows = split_rows(rejected_rect.inset(0.02, 0.025), 1 + len(reason_lines))
    guarded_text(ax, rejected_rect, "S1a rejected total", rejected_rect.cx, reason_rows[0],
                 "Rejected total   n = 1,345", ha="center", va="center",
                 fontsize=8.0, fontweight="bold", color=OI["vermillion"])
    for (marker, count, label), y in zip(reason_lines, reason_rows[1:]):
        # Marker shape carries the category as well as colour does.
        ax.scatter([0.115], [y], transform=ax.transAxes, marker=marker, s=17,
                   facecolor=OI["vermillion"], edgecolor=OI["ink"], linewidth=0.6,
                   zorder=4)
        # The guard is the module the text must stay inside; value-versus-label
        # crowding inside the module is caught by the text-collision check.
        guarded_text(ax, rejected_rect, "S1a reason count",
                     0.30, y, f"{count:,}", dimensions="x", ha="right", va="center",
                     fontsize=7.5, fontweight="bold")
        guarded_text(ax, rejected_rect, "S1a reason label",
                     0.345, y, label, dimensions="x", ha="left", va="center",
                     fontsize=7.5)

    retained_rect = register_module(ax, "S1a retained", Rect(0.02, 0.235, 0.98, 0.395))
    box(ax, retained_rect, facecolor=SEMANTIC_TOKENS["ALLOWED_TEAL_LIGHT"],
        edgecolor=SEMANTIC_TOKENS["ALLOWED_TEAL"])
    arrow_axes(ax, (rejected_rect.cx, rejected_rect.y0), (retained_rect.cx, retained_rect.y1),
               color=OI["mid"], mutation_scale=7)
    guarded_text(ax, retained_rect, "S1a retained", retained_rect.cx, retained_rect.cy,
                 "Retained fixed cohort\nn = 120", ha="center", va="center",
                 fontsize=8.0, fontweight="bold",
                 color=SEMANTIC_TOKENS["ALLOWED_TEAL"], linespacing=1.15)
    ax.text(0.50, 0.155, "1,345 + 120 = 1,465", transform=ax.transAxes,
            fontsize=7.5, ha="center", va="center", color=OI["mid"])

    ax = fig.add_subplot(map_gs[0])
    ax_legend = fig.add_subplot(map_gs[1])
    clean_axis(ax_legend)
    panel_heading(ax, "b", "Coordinates by HUC2")
    counts = Counter(row["huc2"] for row in registry)
    hucs = sorted(counts, key=int)
    huc_shape_fill = [
        (HUC_MARKERS[index % len(HUC_MARKERS)],
         "filled" if index < len(HUC_MARKERS) else "open")
        for index in range(len(hucs))
    ]
    if len(hucs) != 15 or len(set(huc_shape_fill)) != len(hucs):
        raise RuntimeError("Refusing redraw: HUC2 shape-by-fill encoding is not 15-way unique")
    colors = [OI["blue"], OI["orange"], OI["green"], OI["vermillion"], OI["purple"]]
    markers = HUC_MARKERS
    legend_handles: list[Line2D] = []
    for index, huc in enumerate(hucs):
        rows = [row for row in registry if row["huc2"] == huc]
        color = colors[index % len(colors)]
        marker = markers[index % len(markers)]
        filled = index < len(markers)
        marker_face = color if filled else "white"
        ax.scatter([float(row["lon"]) for row in rows], [float(row["lat"]) for row in rows],
                   s=22, marker=marker, facecolor=marker_face, edgecolor=OI["ink"],
                   linewidth=0.6, alpha=0.92, zorder=3)
        legend_handles.append(Line2D([0], [0], marker=marker, color="none",
                                     markerfacecolor=marker_face, markeredgecolor=OI["ink"],
                                     markeredgewidth=0.6,
                                     markersize=4.7, label=f"{int(huc):02d}"))
    ax.set_xlabel("Longitude (°E)", labelpad=2)
    ax.set_ylabel("Latitude (°N)", labelpad=2)
    ax.set_xlim(-126, -69)
    ax.set_ylim(24, 50.5)
    ax.set_xticks([-120, -100, -80, -70])
    ax.set_yticks([25, 35, 45, 50])
    ax.grid(True, color=SEMANTIC_TOKENS["NEUTRAL_GRID"], linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.03, 0.035, "coordinate scatter • no basemap", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7.5, color=OI["mid"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})
    # The legend lives in its own cell.  Hanging it off the map axes with
    # bbox_to_anchor put it outside anything the layout engine measures, which
    # is how it ended up crowding panel (d)'s heading.
    ax_legend.legend(handles=legend_handles, title="HUC2", ncol=5, loc="upper center",
                     bbox_to_anchor=(0.5, 1.0), frameon=False, columnspacing=0.7,
                     handletextpad=0.2, borderaxespad=0.0, labelspacing=0.35,
                     title_fontsize=7.5)

    ax = fig.add_subplot(gs[1, 0])
    panel_heading(ax, "c", "Stations per HUC2")
    values = [counts[huc] for huc in hucs]
    xpos = list(range(len(hucs)))
    ax.bar(xpos, values, width=0.70, color=SEMANTIC_TOKENS["TR_BLUE_LIGHT"],
           edgecolor=OI["blue"],
           linewidth=0.7, zorder=2)
    ax.scatter(xpos, values, marker="D", s=13, facecolor=OI["orange"],
               edgecolor=OI["ink"], linewidth=0.6, zorder=3)
    for x, value in zip(xpos, values):
        ax.text(x, value + 0.9, str(value), va="bottom", ha="center", fontsize=7.5)
    # Two-digit codes at 7.5 pt fit upright across 15 categories; the 45-degree
    # rotation the previous draft used made every neighbouring pair of labels
    # overlap (14 collisions) for no gain in fit.
    ax.set_xticks(xpos, [f"{int(huc):02d}" for huc in hucs])
    ax.set_xlim(-0.7, len(hucs) - 0.3)
    ax.set_ylim(0, 30)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_xlabel("HUC2", labelpad=2)
    ax.set_ylabel("Retained stations", labelpad=2)
    ax.tick_params(axis="x", pad=1.5)
    ax.grid(axis="y", color=SEMANTIC_TOKENS["NEUTRAL_GRID"], linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[1, 1])
    clean_axis(ax)
    panel_heading(ax, "d", "Registry diagnostics")
    geography = audit["geography"]
    nearest = geography["nearest_station_distance_km"]
    diagnostics = [
        ("97", "unique HUC codes", OI["blue"], "o"),
        ("38", "stations in repeated HUC", OI["purple"], "s"),
        ("53.856", "median nearest (km)", OI["green"], "D"),
        ("0.752", "minimum nearest (km)", OI["orange"], "^"),
        ("19", "stations <10 km", OI["vermillion"], "X"),
    ]
    # Marker | value | label are three fixed columns.  The value column used to
    # end at 0.27 with the marker at 0.055, so "53.856" ran back over its own
    # marker -- a collision no text-versus-text check can see.
    marker_x, value_right, label_left = 0.045, 0.265, 0.305
    diagnostic_rows = split_rows(Rect(0.02, 0.155, 0.98, 0.955), len(diagnostics))
    row_height = (0.955 - 0.155) / len(diagnostics)
    for index, ((value, label, color, marker), y) in enumerate(zip(diagnostics, diagnostic_rows)):
        row_rect = register_module(
            ax, f"S1d row {index}",
            Rect(0.02, y - row_height * 0.44, 0.98, y + row_height * 0.44),
        )
        ax.add_patch(Rectangle((row_rect.x0, row_rect.y0), row_rect.width, row_rect.height,
                               transform=ax.transAxes,
                               facecolor=mpl.colors.to_rgba(color, 0.09),
                               edgecolor=color, linewidth=0.75))
        ax.scatter([marker_x], [y], transform=ax.transAxes, marker=marker, s=20,
                   facecolor=color, edgecolor=OI["ink"], linewidth=0.6, zorder=4)
        guarded_text(ax, row_rect, f"S1 diagnostic value {index}",
                     value_right, y, value, dimensions="x", ha="right", va="center",
                     fontsize=8.0, fontweight="bold", color=OI["ink"])
        guarded_text(ax, row_rect, f"S1 diagnostic label {index}",
                     label_left, y, label, dimensions="x", ha="left", va="center",
                     fontsize=7.5, color=OI["ink"])
    ax.text(0.50, 0.065, "HUC overlap / proximity\n≠ hydraulic connectivity",
            transform=ax.transAxes, ha="center", va="center", fontsize=7.5,
            fontweight="bold", color=OI["vermillion"], linespacing=1.15)

    path = HERE / "figS1_cohort_registry.svg"
    save_figure(
        fig, path, "Fig. S1 Frozen cohort selection and registry geometry",
        "Four panels reconcile the 1,465-candidate ledger, encode all 120 retained coordinates by 15 HUC2 categories without a basemap, show HUC2 counts, and report outcome-free registry proximity and repeated-HUC diagnostics.",
    )
    return path


def render_fig_s2(
    temporal: dict[str, object], bridge_report: dict[str, object], request_map: dict[str, object]
) -> Path:
    fig = plt.figure(figsize=figstyle.figsize(figstyle.FULL_MM, 170.0))
    # Ratios follow the line counts each panel has to hold, not a guess: (d)
    # carries eight lines on one side and seven on the other, so it gets the
    # tallest row.
    gs = fig.add_gridspec(4, 1, height_ratios=[0.70, 0.92, 1.08, 1.30], hspace=0.22)

    ax = fig.add_subplot(gs[0, :])
    clean_axis(ax)
    panel_heading(ax, "a", "Frozen role chronology")
    stage_styles = [
        (OI["blue"], "o", None), (OI["sky"], "^", "///"),
        (OI["green"], "D", None), (OI["orange"], "s", "xxx"),
        (OI["purple"], "*", None),
    ]
    role_display = {
        "TRAIN": "TRAIN", "VALIDATION": "VALIDATION", "CALIBRATION": "CALIBRATION",
        "INSPECTED_EXPLORATORY": "INSPECTED\nEXPLORATORY",
        "ONE_TIME_TARGET": "ONE-TIME\nTARGET",
    }
    stages = [
        (_display_year_interval(stage["interval"]), role_display[stage["role"]], *style)
        for stage, style in zip(temporal["chronology"], stage_styles)
    ]
    # Five equal stages derived from the count, not five hand-typed x values.
    stage_columns = split_columns(0.0, 1.0, len(stages), gutter=0.022)
    stage_rects = [Rect(column.x0, 0.14, column.x1, 0.94) for column in stage_columns]
    for index, ((years, role, color, marker, hatch), rect) in enumerate(zip(stages, stage_rects)):
        register_module(ax, f"S2a stage {index}", rect)
        box(ax, rect, facecolor=mpl.colors.to_rgba(color, 0.14),
            edgecolor=color, hatch=hatch, flag=True, linewidth=1.0)
        # A two-line role gets two slots, so the longest stage label cannot
        # push its second line through the bottom of its own box.
        role_lines = role.count("\n") + 1
        rows = split_rows(rect.inset(0.010, 0.100), 2 + role_lines)
        ax.scatter([rect.cx], [rows[0]], transform=ax.transAxes, marker=marker,
                   s=30 if marker != "*" else 50, facecolor=color,
                   edgecolor=OI["ink"], linewidth=0.6, zorder=4)
        guarded_text(ax, rect, f"S2a stage years {index}", rect.cx, rows[1], years,
                     ha="center", va="center", fontsize=8.0, fontweight="bold")
        # Regular weight, not bold: "EXPLORATORY" set bold is wider than a
        # fifth of the text block, and the years above it already carry the
        # emphasis.
        guarded_text(ax, rect, f"S2a stage role {index}", rect.cx,
                     sum(rows[2:]) / role_lines, role,
                     ha="center", va="center", fontsize=7.5,
                     color=OI["ink"], linespacing=1.15)
        if index < len(stages) - 1:
            arrow_axes(ax, (rect.x1, rect.cy), (stage_rects[index + 1].x0, rect.cy),
                       color=OI["mid"], linewidth=1.0, mutation_scale=7)
    # "time ->" sat on the last stage box; it now has the strip below them.
    ax.text(0.995, 0.005, "time →", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color=OI["mid"])

    ax = fig.add_subplot(gs[1, 0])
    clean_axis(ax)
    panel_heading(ax, "b", "Issue-time boundary")
    allowed_rect = register_module(ax, "S2b allowed", Rect(0.02, 0.16, 0.465, 0.86))
    excluded_rect = register_module(ax, "S2b excluded", Rect(0.535, 0.16, 0.98, 0.86))
    box(ax, allowed_rect, facecolor=SEMANTIC_TOKENS["ALLOWED_TEAL_LIGHT"],
        edgecolor=SEMANTIC_TOKENS["ALLOWED_TEAL"], linewidth=1.0)
    box(ax, excluded_rect, facecolor=SEMANTIC_TOKENS["WARNING_LIGHT"],
        edgecolor=OI["vermillion"], linewidth=1.0, hatch="///", flag=True)
    ax.plot([0.5, 0.5], [0.16, 0.90], transform=ax.transAxes, color=OI["ink"],
            linewidth=1.4, zorder=5, solid_capstyle="butt")
    ax.text(0.5, 0.915, "ISSUE  t", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.0, fontweight="bold")

    allowed = [
        ("o", "Observed WTEMP history"),
        ("^", "FLOW + dated meteorology"),
        ("D", "Frozen reference inputs"),
    ]
    excluded = [
        ("X", f"Target WTEMP at {temporal['target_date_expression']}"),
        ("X", "Future weather"),
        ("X", "Future vintages"),
    ]
    # The admissible/inadmissible status is carried by a drawn marker rather
    # than a dingbat: Nimbus Sans has no U+2713/U+2715, and a glyph the chosen
    # typeface lacks is exactly the kind of thing that ships as a tofu box.
    for rect, title, title_color, status_marker, items, marker_color, label in (
        (allowed_rect, f"ALLOWED: {temporal['source_date_display']}",
         SEMANTIC_TOKENS["ALLOWED_TEAL"], "P", allowed, OI["blue"], "allowed"),
        (excluded_rect, "EXCLUDED AS INPUT", OI["vermillion"], "X", excluded,
         OI["vermillion"], "excluded"),
    ):
        rows = split_rows(rect.inset(0.012, 0.075), 1 + len(items))
        ax.scatter([rect.x0 + 0.055], [rows[0]], transform=ax.transAxes,
                   marker=status_marker, s=26, facecolor=title_color,
                   edgecolor=OI["ink"], linewidth=0.6, zorder=5)
        guarded_text(ax, rect, f"S2b {label} title", rect.x0 + 0.085, rows[0], title,
                     dimensions="x", ha="left", va="center", fontsize=7.5,
                     fontweight="bold", color=title_color)
        for (marker, item), y in zip(items, rows[1:]):
            ax.scatter([rect.x0 + 0.055], [y], transform=ax.transAxes, marker=marker,
                       s=20, facecolor=marker_color, edgecolor=OI["ink"],
                       linewidth=0.6, zorder=5)
            guarded_text(ax, rect, f"S2b {label} item", rect.x0 + 0.085, y, item,
                         dimensions="x", ha="left", va="center", fontsize=7.5)
    # The horizon set used to be a floating inset inside the excluded box, where
    # it landed on "Future vintages" -- the one text collision FigS2 reported.
    # It is now part of the panel footer, outside every module.
    horizons = ", ".join(str(value) for value in temporal["horizons"])
    ax.text(0.50, 0.045,
            f"date-indexed retrospective hindcast   •   h = {horizons} d",
            transform=ax.transAxes, ha="center", va="center", fontsize=7.5,
            fontweight="bold", color=OI["ink"])

    # Variable/provider/source-date matrix.
    ax = fig.add_subplot(gs[2, 0])
    clean_axis(ax)
    panel_heading(ax, "c", "Predictor source rules")
    matrix = [
        (variable, temporal["provider_by_variable"][variable], temporal["source_date_display"])
        for variable in ROUTE_A_VARIABLES
    ]
    header_y = 0.945
    rule_y = 0.895
    headers = [(0.115, "VARIABLE"), (0.40, "PROVIDER"), (0.66, "SOURCE DATE")]
    for x, label in headers:
        ax.text(x, header_y, label, transform=ax.transAxes, fontsize=7.5,
                fontweight="bold", ha="left", va="center", color=OI["mid"])
    ax.plot([0.04, 0.96], [rule_y, rule_y], transform=ax.transAxes,
            color=OI["mid"], linewidth=0.8)
    table_rect = Rect(0.04, 0.155, 0.96, rule_y - 0.02)
    row_ys = split_rows(table_rect, len(matrix))
    row_height = table_rect.height / len(matrix)
    for index, ((variable, provider, source_date), y) in enumerate(zip(matrix, row_ys)):
        if index % 2 == 0:
            ax.add_patch(Rectangle((table_rect.x0, y - row_height * 0.46),
                                   table_rect.width, row_height * 0.92,
                                   transform=ax.transAxes,
                                   facecolor="#F4F4F4", edgecolor="none"))
        marker = ["o", "s", "^", "D"][index % 4]
        ax.scatter([0.075], [y], transform=ax.transAxes, s=16, marker=marker,
                   facecolor=OI["sky"], edgecolor=OI["ink"], linewidth=0.6, zorder=4)
        ax.text(0.115, y, variable, transform=ax.transAxes, fontsize=7.5,
                fontweight="bold", ha="left", va="center")
        ax.text(0.40, y, provider, transform=ax.transAxes, fontsize=7.5,
                ha="left", va="center")
        ax.text(0.66, y, source_date, transform=ax.transAxes, fontsize=7.5,
                ha="left", va="center")
    ax.text(0.50, 0.055, f"vintage: {temporal['retrieval_vintage_display']}",
            transform=ax.transAxes, fontsize=7.5, fontweight="bold",
            ha="center", va="center", color=OI["blue"])

    # Equal-weight capability / limitation bridge panel.
    ax = fig.add_subplot(gs[3, 0])
    clean_axis(ax)
    panel_heading(ax, "d", "Outcome-free predictor bridge: capability and limit")
    left_rect = register_module(ax, "S2d capability", Rect(0.02, 0.155, 0.485, 0.93))
    right_rect = register_module(ax, "S2d limitation", Rect(0.515, 0.155, 0.98, 0.93))
    box(ax, left_rect, facecolor=SEMANTIC_TOKENS["ALLOWED_TEAL_LIGHT"],
        edgecolor=SEMANTIC_TOKENS["ALLOWED_TEAL"], linewidth=1.0)
    box(ax, right_rect, facecolor=SEMANTIC_TOKENS["WARNING_LIGHT"],
        edgecolor=OI["vermillion"], linewidth=1.0, hatch="///", flag=True)
    left_lines = [
        f"{_display_year_interval((bridge_report['interval']['start'], bridge_report['interval']['end']))} only",
        f"{bridge_report['site_count']} sites • {bridge_report['row_count']:,} rows",
        f"{len(BRIDGE_FIELDS)} meteorological fields only:",
        " • ".join(BRIDGE_FIELDS),
        "product / parser compatibility",
        "missing-pattern agreement",
        "±1-day value-alignment check",
    ]
    right_lines = [
        "not as-issued availability",
        "not NWIS/provider\nlocal-day equivalence",
        "not target-period input availability",
        "not an operational replay",
        "outcomes not requested or read",
    ]
    # Rows are counted in rendered lines, so the two-line limitation gets two
    # slots and cannot crowd the entry under it.
    for rect, title, title_color, status_marker, lines, label in (
        (left_rect, "PASS_EXACT_PRODUCT_BRIDGE", SEMANTIC_TOKENS["ALLOWED_TEAL"],
         "P", left_lines, "capability"),
        (right_rect, "LIMITS — NOT ESTABLISHED", OI["vermillion"],
         "X", right_lines, "limitation"),
    ):
        line_counts = [1] + [text.count("\n") + 1 for text in lines]
        rows = split_rows(rect.inset(0.012, 0.055), sum(line_counts))
        ax.scatter([rect.x0 + 0.055], [rows[0]], transform=ax.transAxes,
                   marker=status_marker, s=26, facecolor=title_color,
                   edgecolor=OI["ink"], linewidth=0.6, zorder=5)
        guarded_text(ax, rect, f"S2d {label} 0", rect.x0 + 0.085, rows[0], title,
                     dimensions="x", ha="left", va="center", fontsize=7.5,
                     fontweight="bold", color=title_color)
        cursor = 1
        for index, (text_value, span) in enumerate(zip(lines, line_counts[1:]), start=1):
            centre = sum(rows[cursor:cursor + span]) / span
            cursor += span
            guarded_text(ax, rect, f"S2d {label} {index}", rect.cx, centre, text_value,
                         ha="center", va="center", fontsize=7.5,
                         fontweight="bold" if index <= 2 else "normal",
                         color=OI["ink"], linespacing=1.15)
    ax.text(0.50, 0.055, "bound report + 120-request source map • latest retrospective retrieval",
            transform=ax.transAxes, ha="center", va="center", fontsize=7.5,
            color=OI["mid"], fontweight="bold")

    path = HERE / "figS2_information_boundary.svg"
    save_figure(
        fig, path, "Fig. S2 Temporal roles and issue-time information boundary",
        "Four panels separate temporal roles, the issue-time predictor boundary, all seven provider/source-date rules, and the equally weighted capabilities and limitations of the outcome-free 2018–2020 predictor bridge.",
    )
    return path


def stack_entries(
    ax: mpl.axes.Axes,
    rect: Rect,
    name: str,
    entries: Sequence[tuple[str, dict]],
    *,
    inset_x: float = 0.010,
    inset_y: float = 0.020,
    max_step: float | None = None,
) -> None:
    """Centre a stack of text entries inside a module, one slot per line.

    A two-line entry occupies two slots, so adding a line to any module pushes
    the stack apart instead of letting one entry sit on the next.  ``max_step``
    caps the slot height so a tall module holding few lines reads as a block of
    text rather than as widely scattered lines.
    """

    line_counts = [text.count("\n") + 1 for text, _ in entries]
    inner = rect.inset(inset_x, inset_y)
    total = sum(line_counts)
    if max_step is not None and total * max_step < inner.height:
        half = 0.5 * total * max_step
        inner = Rect(inner.x0, inner.cy - half, inner.x1, inner.cy + half)
    rows = split_rows(inner, total)
    cursor = 0
    for index, ((text_value, style), span) in enumerate(zip(entries, line_counts)):
        centre = sum(rows[cursor:cursor + span]) / span
        cursor += span
        guarded_text(ax, rect, f"{name} line {index}", rect.cx, centre, text_value,
                     ha="center", va="center", linespacing=1.15, **style)


def render_fig_s3(architecture: dict[str, object]) -> Path:
    """Draw the model/calibration schematic on a declared grid.

    The previous draft placed every box, inset and connector by hand in figure
    coordinates.  Three things went wrong that no text-collision check could
    see: module titles ran across the border of the neighbouring box, the
    tanh identity box overlapped the head boxes below it, and two long curved
    connectors were pushed to ``zorder=0.5`` so they could pass *underneath*
    three unrelated modules.

    Here the figure is a grid of four columns and four rows plus a full-width
    identity band.  Modules occupy whole cells, connectors are only ever
    allowed to travel in the gutters between cells, and
    :func:`validate_diagram_geometry` refuses the render if a module overlaps a
    module or a connector reaches into one.
    """

    fig = plt.figure(figsize=figstyle.figsize(figstyle.FULL_MM, 148.0))
    # One exact full-figure axes: the schematic is geometry, not data, so the
    # layout engine has nothing useful to solve here.
    fig.set_layout_engine("none")
    ax = fig.add_axes([0.012, 0.010, 0.976, 0.975])
    clean_axis(ax)

    # ---- the grid -------------------------------------------------------
    # Column widths follow the longest string each column has to hold at
    # 7.5 pt; the gutters are wide enough for an arrowhead, and the one
    # between representation and heads is wider because three connectors fan
    # out through it.
    columns = {
        "inputs": (0.005, 0.229),
        "representation": (0.259, 0.507),
        "heads": (0.557, 0.767),
        "calibration": (0.797, 0.995),
    }
    rows = {
        1: (0.815, 0.950),
        2: (0.658, 0.793),
        3: (0.501, 0.636),
        4: (0.344, 0.479),
    }
    band = Rect(0.259, 0.075, 0.995, 0.322)
    footer = Rect(0.005, 0.000, 0.995, 0.052)

    def cell(column: str, first: int, last: int | None = None) -> Rect:
        x0, x1 = columns[column]
        y0 = rows[last or first][0]
        y1 = rows[first][1]
        return Rect(x0, y0, x1, y1)

    def row_centre(index: int) -> float:
        return 0.5 * sum(rows[index])

    gutter_ab = 0.5 * (columns["inputs"][1] + columns["representation"][0])

    for column, label in (("inputs", "(a) INPUTS"),
                          ("representation", "(b) REPRESENTATION"),
                          ("heads", "(c) HEADS"),
                          ("calibration", "(d) CALIBRATION")):
        ax.text(columns[column][0], 0.978, label, transform=ax.transAxes,
                ha="left", va="center", fontsize=7.5, fontweight="bold",
                color=OI["mid"])

    quantile_labels = tuple(
        f"q{int(round(float(probability) * 100)):02d}"
        for probability in architecture["quantiles"]
    )
    variables = tuple(architecture["variables"])
    # Two per line, then the odd one: the widest line is "RHMEAN • DH", which
    # fits the inputs column with room to spare.  The list is projected from
    # the asserted registry rather than typed out, so it cannot drift from it.
    variable_lines = "\n".join(
        " • ".join(variables[start:stop])
        for start, stop in ((0, 2), (2, 4), (4, 6), (6, len(variables)))
        if variables[start:stop]
    )
    teal = SEMANTIC_TOKENS["ALLOWED_TEAL"]
    teal_light = SEMANTIC_TOKENS["ALLOWED_TEAL_LIGHT"]
    blue_light = SEMANTIC_TOKENS["TR_BLUE_LIGHT"]
    purple_light = "#F9EAF3"
    calibration_period = architecture["calibration_display"]

    title = {"fontsize": 8.0, "fontweight": "bold"}
    subtitle = {"fontsize": 7.5, "fontweight": "bold"}
    detail = {"fontsize": 7.5, "color": OI["ink"]}

    # ---- modules --------------------------------------------------------
    # (rect, key, edge, fill, hatch, marker, entries)
    modules = [
        (cell("inputs", 1, 2), "S3 inputs", OI["blue"], blue_light, None, None, [
            (f"{len(variables)} VARIABLES\n"
             f"+ {'MASK' if architecture['missingness_mask'] else 'NO MASK'}",
             {**title, "color": OI["blue"]}),
            (variable_lines, detail),
            (f"{architecture['context_length']}-day buffer\nconstruction only\n"
             "≠ effective\nmemory",
             {**subtitle, "color": OI["vermillion"]}),
        ]),
        (cell("inputs", 3), "S3 wlevel", OI["vermillion"],
         SEMANTIC_TOKENS["WARNING_LIGHT"], "///", ("X", OI["vermillion"]), [
            ("WLEVEL", {**subtitle, "color": OI["vermillion"]}),
            ("excluded", detail),
        ]),
        (Rect(columns["inputs"][0], band.y0, columns["inputs"][1], rows[4][1]),
         "S3 anchor", OI["orange"], "#FFF7D1", None, None, [
            (str(architecture["anchor_identity"]), {**title, "color": "#986900"}),
            ("frozen damped-\npersistence anchor", detail),
            ("fitted blend of\nlast y and\nclimatology", detail),
        ]),
        (cell("representation", 1), "S3 router", teal, teal_light, None, ("D", teal), [
            ("SPARSE ROUTER", {**subtitle, "color": teal}),
            (f"{len(variables)} variables\nlags 0–{architecture['max_router_lag']}",
             detail),
        ]),
        (cell("representation", 2), "S3 tcn", OI["blue"], blue_light, None,
         ("^", OI["blue"]), [
            ("LEFT-LOOKING TCN", {**subtitle, "color": OI["blue"]}),
            (f"{architecture['tcn_blocks']} blocks • kernel {architecture['tcn_kernel']}",
             detail),
            (f"receptive field {architecture['tcn_receptive_field']}", detail),
        ]),
        (cell("representation", 3), "S3 mixture", OI["purple"], purple_light, None,
         ("s", OI["purple"]), [
            ("MIXTURE OF\nEXPERTS", {**subtitle, "color": OI["purple"]}),
            ("combined\nrepresentation", detail),
        ]),
        (cell("representation", 4), "S3 proposal", teal, teal_light, None, None, [
            ("PROPOSAL P", {**subtitle, "color": teal}),
            ("learned relaxation κ", detail),
        ]),
        (cell("heads", 1), "S3 event head", teal, teal_light, None, ("^", teal), [
            ("EVENT HEAD", {**subtitle, "color": teal}),
            ("event score", detail),
        ]),
        (cell("heads", 2, 3), "S3 quantile heads", OI["purple"], purple_light, None,
         ("D", OI["purple"]), [
            ("Q HEADS", {**subtitle, "color": OI["purple"]}),
            (f"{quantile_labels[1]}: separate", detail),
            ("anchor-bounded", detail),
            (f"{quantile_labels[0]}/{quantile_labels[2]}: ± widths", detail),
        ]),
        (cell("heads", 4), "S3 point head", OI["blue"], blue_light, None,
         ("o", OI["blue"]), [
            ("POINT HEAD", {**subtitle, "color": OI["blue"]}),
            ("MSE residual r", detail),
        ]),
        (cell("calibration", 1), "S3 platt", teal, teal_light, None, None, [
            ("PLATT", {**title, "color": teal}),
            (f"fit {calibration_period} only", detail),
        ]),
        (cell("calibration", 2), "S3 probability", teal, "white", None, None, [
            ("CALIBRATED\nPROBABILITY", {**subtitle, "color": teal}),
        ]),
        (cell("calibration", 3), "S3 cqr", OI["purple"], purple_light, None, None, [
            ("CQR", {**title, "color": OI["purple"]}),
            ("member average", detail),
            (f"fit {calibration_period} only", detail),
        ]),
        (cell("calibration", 4), "S3 interval", OI["purple"], "white", None, None, [
            ("CALIBRATED\nINTERVAL", {**subtitle, "color": OI["purple"]}),
        ]),
    ]
    for rect, key, edge, fill, hatch, marker, entries in modules:
        register_module(ax, key, rect)
        box(ax, rect, facecolor=fill, edgecolor=edge, hatch=hatch,
            flag=hatch is not None, linewidth=1.0)
        if marker is not None:
            shape, marker_color = marker
            ax.scatter([rect.x0 + (0.050 if hatch else 0.022)], [rect.y1 - 0.024],
                       transform=ax.transAxes, marker=shape, s=20,
                       facecolor=marker_color, edgecolor=OI["ink"], linewidth=0.6,
                       zorder=5)
        # 0.036 of the axes is about 5 mm: comfortable leading, and it stops the
        # tall anchor and output modules from scattering three lines over 55 mm.
        stack_entries(ax, rect, key, entries, max_step=0.036)

    # ---- the anchor-bounded point identity ------------------------------
    # Relocated from Figure 1(b) on 2026-08-06 and kept here: the damped anchor
    # line A, the shaded A+/-delta envelope, and the in-panel non-safety
    # warning.  It now has a band of its own instead of a box overlapping the
    # head modules.
    register_module(ax, "S3 identity band", band)
    box(ax, band, facecolor="white", edgecolor=OI["blue"], linewidth=1.0)
    ax.text(band.x0 + 0.016, band.y1 - 0.026, "ANCHOR-BOUNDED POINT IDENTITY",
            transform=ax.transAxes, ha="left", va="center", fontsize=7.5,
            fontweight="bold", color=OI["blue"])

    tanh_ax = ax.inset_axes([0.330, 0.108, 0.150, 0.150])
    tz = np.linspace(-3.0, 3.0, 301)
    tanh_ax.axhspan(-1, 1, facecolor=blue_light, alpha=0.55,
                    hatch="//", edgecolor=SEMANTIC_TOKENS["TR_BLUE"])
    tanh_ax.axhline(1, color=SEMANTIC_TOKENS["TR_BLUE"], linestyle=(0, (4, 2)), linewidth=0.9)
    tanh_ax.axhline(-1, color=SEMANTIC_TOKENS["TR_BLUE"], linestyle=(0, (4, 2)), linewidth=0.9)
    tanh_ax.axhline(0, color=OI["mid"], linewidth=0.65)
    tanh_ax.plot(tz, np.tanh(tz), color=teal, linewidth=1.8)
    tanh_ax.scatter([0], [0], s=14, marker="D", facecolor="white",
                    edgecolor=teal, linewidth=0.6, zorder=4)
    tanh_ax.set_xlim(-3, 3)
    tanh_ax.set_ylim(-1.24, 1.24)
    tanh_ax.set_xticks([-2, 0, 2])
    tanh_ax.set_yticks([-1, 0, 1], ["A−δ", "A", "A+δ"])
    tanh_ax.set_xlabel("z/δ", labelpad=1.0, fontsize=7.5, color=OI["mid"])
    tanh_ax.spines[["top", "right"]].set_visible(False)
    tanh_ax.tick_params(length=2.0, width=0.6, color=OI["mid"], labelsize=7.5)

    identity_rect = Rect(0.500, band.y0 + 0.020, band.x1 - 0.014, band.y1 - 0.050)
    stack_entries(ax, identity_rect, "S3 identity", [
        ("ŷ = A + δ tanh(z/δ)", {"fontsize": 7.5}),
        (f"z = P − A + r  •  δ = {architecture['delta_scale']:.1f} °C",
         {"fontsize": 7.5}),
        ("Deviation from anchor;\nnot an error or safety bound",
         {**subtitle, "color": OI["vermillion"]}),
    ], inset_x=0.004, inset_y=0.004)

    # ---- connectors -----------------------------------------------------
    # Every route below either joins two facing edges or travels inside a
    # gutter.  The single crossing -- the router's descent past the input-to-TCN
    # feed -- is unavoidable for a parallel merge and crosses a line, not a box.
    input_rect = cell("inputs", 1, 2)
    router_rect = cell("representation", 1)
    tcn_rect = cell("representation", 2)
    mixture_rect = cell("representation", 3)
    proposal_rect = cell("representation", 4)
    event_rect = cell("heads", 1)
    quantile_rect = cell("heads", 2, 3)
    point_rect = cell("heads", 4)
    platt_rect = cell("calibration", 1)
    probability_rect = cell("calibration", 2)
    cqr_rect = cell("calibration", 3)
    interval_rect = cell("calibration", 4)
    anchor_rect = Rect(columns["inputs"][0], band.y0, columns["inputs"][1], rows[4][1])

    arrow_axes(ax, (input_rect.x1, row_centre(1)), (router_rect.x0, row_centre(1)),
               color=OI["blue"])
    arrow_axes(ax, (input_rect.x1, row_centre(2)), (tcn_rect.x0, row_centre(2)),
               color=OI["blue"])
    # Router and TCN are parallel branches; the router reaches the mixture down
    # the inputs/representation gutter rather than through the TCN box.
    arrow_axes(ax, (router_rect.x0, router_rect.y0 + 0.030),
               (mixture_rect.x0, mixture_rect.cy), color=teal,
               waypoints=[(gutter_ab, router_rect.y0 + 0.030),
                          (gutter_ab, mixture_rect.cy)])
    arrow_axes(ax, (tcn_rect.cx, tcn_rect.y0), (mixture_rect.cx, mixture_rect.y1),
               color=OI["blue"])
    arrow_axes(ax, (mixture_rect.cx, mixture_rect.y0), (proposal_rect.cx, proposal_rect.y1),
               color=OI["purple"])
    for target, color in ((event_rect, teal), (quantile_rect, OI["purple"]),
                          (point_rect, OI["blue"])):
        arrow_axes(ax, (mixture_rect.x1, mixture_rect.cy), (target.x0, target.cy),
                   color=color)
    arrow_axes(ax, (event_rect.x1, event_rect.cy), (platt_rect.x0, platt_rect.cy),
               color=teal)
    arrow_axes(ax, (platt_rect.cx, platt_rect.y0), (probability_rect.cx, probability_rect.y1),
               color=teal)
    arrow_axes(ax, (quantile_rect.x1, cqr_rect.cy), (cqr_rect.x0, cqr_rect.cy),
               color=OI["purple"])
    arrow_axes(ax, (cqr_rect.cx, cqr_rect.y0), (interval_rect.cx, interval_rect.y1),
               color=OI["purple"])
    # Anchor A, proposal P and the point residual r are the three terms of the
    # identity; each enters the band from the module directly beside or above it.
    arrow_axes(ax, (anchor_rect.x1, band.cy), (band.x0, band.cy), color=OI["orange"])
    arrow_axes(ax, (proposal_rect.cx, proposal_rect.y0), (proposal_rect.cx, band.y1),
               color=teal)
    arrow_axes(ax, (point_rect.cx, point_rect.y0), (point_rect.cx, band.y1),
               color=OI["blue"])

    box(ax, footer, facecolor="#F3F3F3", edgecolor=OI["mid"], radius=0.006,
        linewidth=0.7)
    ax.text(footer.cx, footer.cy, "κ / router: internal allocations ≠ physical routing",
            transform=ax.transAxes, ha="center", va="center", fontsize=7.5,
            fontweight="bold", color=OI["ink"])

    path = HERE / "figS3_model_architecture.svg"
    save_figure(
        fig, path, "Fig. S3 ThermoRoute PRE model and calibration architecture",
        "A four-column schematic shows the strictly checked seven-variable input contract, WLEVEL exclusion, construction-buffer/router/TCN distinctions, learned proposal and mixture, separate point/quantile/event heads, the anchor-bound bounded-correction tanh schematic (relocated from Figure 1b) with its A+/-delta envelope and non-safety warning, and the final CQR interval and Platt probability outputs.",
    )
    return path


def write_s1_marker_projection(registry: Sequence[dict[str, str]]) -> Path:
    """Write the exact data projection underlying FigS1's 120 scatter markers."""

    path = HERE / "figS1_cohort_registry_data.csv"
    fields = [
        "site_no", "state", "lat", "lon", "huc2",
        "site_value_id", "state_value_id", "x_value_id", "y_value_id",
        "huc2_value_id",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in sorted(registry, key=lambda item: item["site_no"]):
            site = row["site_no"]
            writer.writerow({
                "site_no": site,
                "state": row["state"],
                "lat": row["lat"],
                "lon": row["lon"],
                "huc2": row["huc2"],
                "site_value_id": f"figs1.station.{site}.site_id",
                "state_value_id": f"figs1.station.{site}.state",
                "x_value_id": f"figs1.station.{site}.longitude",
                "y_value_id": f"figs1.station.{site}.latitude",
                "huc2_value_id": f"figs1.station.{site}.huc2",
            })
    return path


def write_manifest(
    registry: Sequence[dict[str, str]],
    reasons: Counter[str],
    audit: dict[str, object],
    bridge_manifest: dict[str, object],
    bridge_report: dict[str, object],
    request_map: dict[str, object],
    raw_leaf_summary: dict[str, object],
    temporal: dict[str, object],
    architecture: dict[str, object],
) -> Path:
    """Bind sources, renderer, visible structural values, and vector outputs."""

    renderer = Path(__file__).resolve()
    captions = HERE / "FigS1-S3_captions.md"
    marker_data = HERE / "figS1_cohort_registry_data.csv"
    huc_counts = Counter(row["huc2"] for row in registry)
    huc_encoding_categories = [
        {
            "huc2": f"{int(huc):02d}",
            "marker": HUC_MARKERS[index % len(HUC_MARKERS)],
            "fill": "filled" if index < len(HUC_MARKERS) else "open",
            "color": [OI["blue"], OI["orange"], OI["green"],
                      OI["vermillion"], OI["purple"]][index % 5],
        }
        for index, huc in enumerate(sorted(huc_counts, key=int))
    ]
    if len(huc_encoding_categories) != 15 or len({
        (item["marker"], item["fill"]) for item in huc_encoding_categories
    }) != 15:
        raise RuntimeError("Refusing manifest: HUC2 grayscale encoding is not unique")

    source_paths = {
        "registry": DATA / "station_registry_v1.csv",
        "panel": DATA / "panel_usgs_120v2.parquet",
        "rejection_ledger": DATA / "rejected_sites_120v2.csv",
        "environmental_audit": DATA / "development_environmental_audit_v1.json",
        "bridge_manifest": BRIDGE_MANIFEST,
        "bridge_report": BRIDGE_REPORT,
        "bridge_request_map": BRIDGE_REQUEST_MAP,
        "bridge_panel": REPO / bridge_manifest["panel"]["path"],
        "bridge_registry": REPO / bridge_manifest["registry"]["path"],
        "bridge_raw_daymet_index": REPO / bridge_manifest["raw_snapshot_indexes"]["daymet"]["path"],
        "bridge_raw_gridmet_index": REPO / bridge_manifest["raw_snapshot_indexes"]["gridmet"]["path"],
        "bridge_raw_gridmet_schema_index":
            REPO / bridge_manifest["raw_snapshot_indexes"]["gridmet_schema"]["path"],
        "bridge_normalized_frozen": REPO / bridge_manifest["normalized"]["frozen"]["path"],
        "bridge_normalized_refreshed": REPO / bridge_manifest["normalized"]["refreshed"]["path"],
        "protocol": REPO / "protocols/route_a_confirmatory_v1.json",
        "calibration_erratum": REPO / "protocols/route_a_calibration_inclusion_erratum_v1.json",
        "configuration_source": REPO / "src/thermoroute/config.py",
        "model_suite_source": REPO / "src/thermoroute/model_suite.py",
        "model_source": REPO / "src/thermoroute/thermoroute.py",
        "anchor_source": REPO / "src/thermoroute/features.py",
        "training_source": REPO / "src/thermoroute/train.py",
        "redraw_specification": REPO / "paper/FIGURE_REDRAW_SPEC.md",
        "marker_projection": marker_data,
        "main_figure_1_binder": FIG1_BINDER,
    }
    formats = {
        ".csv": "CSV", ".json": "JSON", ".parquet": "Parquet",
        ".py": "Python", ".md": "Markdown",
    }
    semantic_roles = {
        "registry": "frozen retained-site identity, coordinates, and HUC labels",
        "panel": "frozen row-count and retained-station identity reconciliation",
        "rejection_ledger": "recorded site exclusions and recorded reasons",
        "environmental_audit": "outcome-free registry and proximity diagnostics",
        "bridge_manifest": "predictor-bridge path/hash/status authority",
        "bridge_report": "predictor product/parser and alignment bridge evidence",
        "bridge_request_map": "outcome-free site request inventory",
        "bridge_panel": "panel binding independently rechecked for predictor bridge",
        "bridge_registry": "registry binding independently rechecked for predictor bridge",
        "bridge_raw_daymet_index": "Daymet immutable raw-snapshot index",
        "bridge_raw_gridmet_index": "gridMET immutable raw-snapshot index",
        "bridge_raw_gridmet_schema_index": "gridMET schema-response snapshot index",
        "bridge_normalized_frozen": "normalized frozen predictor bridge slice",
        "bridge_normalized_refreshed": "normalized refreshed predictor bridge slice",
        "protocol": "frozen chronology, input, WLEVEL, and calibration contract",
        "calibration_erratum": "2018 calibration population and deployment scope",
        "configuration_source": "context, horizon, router, and TCN constants",
        "model_suite_source": "ordered Route-A seven-variable registry",
        "model_source": "model topology, heads, and anchor-bound implementation",
        "anchor_source": "fitted damped-persistence anchor identity and composition",
        "training_source": "point-head conditional-mean MSE objective",
        "redraw_specification": "authoritative PRE figure and binder contract",
        "marker_projection": "one-row-per-visible-coordinate projection with value IDs",
        "main_figure_1_binder": "cross-figure canonical value/unit consistency authority",
    }

    def binding(key: str) -> dict[str, object]:
        source = source_paths[key]
        return {
            "path": str(source.relative_to(REPO)),
            "sha256": sha256(source),
            "format": formats[source.suffix],
            "semantic_role": semantic_roles[key],
        }

    sources = {key: binding(key) for key in source_paths}

    values: dict[str, dict[str, object]] = {}

    def value(
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
        if value_id in values and values[value_id] != record:
            raise RuntimeError(f"Conflicting binder value: {value_id}")
        values[value_id] = record
        return value_id

    def cell(value_id: str) -> dict[str, str]:
        if value_id not in values:
            raise RuntimeError(f"Mark references undeclared value: {value_id}")
        return {"value_id": value_id}

    def mark(mark_id: str, panel_id: str, **cells: str) -> dict[str, object]:
        return {
            "mark_id": mark_id,
            "panel_id": panel_id,
            "cells": {field: cell(value_id) for field, value_id in cells.items()},
        }

    registry_path = "data_usgs/station_registry_v1.csv"
    rejection_path = "data_usgs/rejected_sites_120v2.csv"
    audit_path = "data_usgs/development_environmental_audit_v1.json"
    bridge_report_path = "data_usgs/development_predictor_bridge_v1/bridge_report_v1.json"
    bridge_map_path = "data_usgs/development_predictor_bridge_v1/source_request_map_v1.json"
    config_path = "src/thermoroute/config.py"
    suite_path = "src/thermoroute/model_suite.py"
    model_path = "src/thermoroute/thermoroute.py"
    features_path = "src/thermoroute/features.py"
    training_path = "src/thermoroute/train.py"
    protocol_path = "protocols/route_a_confirmatory_v1.json"

    # Canonical cross-figure structural identities.
    value("route_a.registry.station_count", 120, "stations", "PRE_STRUCTURAL",
          f"{registry_path}#row_count", "count registry rows", "integer_exact")
    value("route_a.panel.row_count", 657480, "daily panel rows", "PRE_STRUCTURAL",
          "data_usgs/panel_usgs_120v2.parquet#metadata.num_rows",
          "read Parquet metadata row count", "integer_exact")
    value("route_a.registry.huc2_group_count", 15, "pre-attrition HUC2 groups", "PRE_STRUCTURAL",
          f"{registry_path}#huc2", "count distinct nonmissing HUC2 values", "integer_exact")
    value("route_a.cohort.initial_candidate_count", 1465, "stations", "PRE_STRUCTURAL",
          f"{registry_path} + {rejection_path}",
          "retained row count plus rejected row count", "integer_exact")
    value("route_a.cohort.rejected_station_count", 1345, "stations", "PRE_STRUCTURAL",
          f"{rejection_path}#row_count", "count rejection-ledger rows", "integer_exact")
    reason_ids = {
        "no NWIS WTEMP+FLOW": "route_a.cohort.rejected.no_joint_nwis_count",
        "low full-period coverage": "route_a.cohort.rejected.low_full_period_count",
        "low blind-test-period coverage": "route_a.cohort.rejected.low_2019_2020_count",
    }
    reason_label_ids = {
        "no NWIS WTEMP+FLOW": "figs1.rejection.no_joint_nwis.reason_label",
        "low full-period coverage": "figs1.rejection.low_full_period.reason_label",
        "low blind-test-period coverage": "figs1.rejection.low_2019_2020.reason_label",
    }
    displayed_reason_labels = {
        "no NWIS WTEMP+FLOW": "no NWIS WTEMP + FLOW",
        "low full-period coverage": "low full-period coverage",
        "low blind-test-period coverage": "low 2019–20 coverage",
    }
    for reason, value_id in reason_ids.items():
        value(value_id, reasons[reason], "stations", "PRE_STRUCTURAL",
              f"{rejection_path}#reason={reason}", "count rows with recorded reason", "integer_exact")
        value(reason_label_ids[reason], displayed_reason_labels[reason], "reason label",
              "PRE_STRUCTURAL", f"{rejection_path}#reason={reason}",
              "format recorded reason as the visible compact category label", "string_exact")

    for raw_huc in sorted(huc_counts, key=int):
        huc = f"{int(raw_huc):02d}"
        value(f"route_a.registry.huc2_{huc}.code", huc, "HUC2 code", "PRE_STRUCTURAL",
              f"{registry_path}#huc2={raw_huc}", "normalize code to two digits", "string_exact")
        value(f"route_a.registry.huc2_{huc}.station_count", huc_counts[raw_huc],
              "stations", "PRE_STRUCTURAL", f"{registry_path}#huc2={raw_huc}",
              "count registry rows in HUC2", "integer_exact")

    station_marks: list[dict[str, object]] = []
    for row in sorted(registry, key=lambda item: item["site_no"]):
        site = row["site_no"]
        site_id = value(
            f"figs1.station.{site}.site_id", site, "USGS site number",
            "PRE_STRUCTURAL", f"{registry_path}#site_no={site}/site_no",
            "direct registry projection", "string_exact",
        )
        state_id = value(
            f"figs1.station.{site}.state", row["state"], "US state abbreviation",
            "PRE_STRUCTURAL", f"{registry_path}#site_no={site}/state",
            "direct registry projection", "string_exact",
        )
        lon_id = value(
            f"figs1.station.{site}.longitude", float(row["lon"]), "degrees_east",
            "PRE_STRUCTURAL", f"{registry_path}#site_no={site}/lon",
            "direct registry projection", "source_decimal_precision",
        )
        lat_id = value(
            f"figs1.station.{site}.latitude", float(row["lat"]), "degrees_north",
            "PRE_STRUCTURAL", f"{registry_path}#site_no={site}/lat",
            "direct registry projection", "source_decimal_precision",
        )
        huc_id = value(
            f"figs1.station.{site}.huc2", f"{int(row['huc2']):02d}", "HUC2 code",
            "PRE_STRUCTURAL", f"{registry_path}#site_no={site}/huc2",
            "normalize registry code to two digits", "string_exact",
        )
        station_marks.append(mark(f"figs1.station.{site}", "b", site=site_id,
                                  state=state_id, x=lon_id, y=lat_id,
                                  category=huc_id))

    geography = audit["geography"]
    nearest = geography["nearest_station_distance_km"]
    diagnostics = [
        ("figs1.registry.unique_huc_code_count", geography["unique_huc_code_count"],
         "HUC codes", "#/geography/unique_huc_code_count", "integer_exact"),
        ("figs1.registry.repeated_huc_station_count",
         geography["stations_in_repeated_huc_code_groups"], "stations",
         "#/geography/stations_in_repeated_huc_code_groups", "integer_exact"),
        ("figs1.registry.nearest_distance_median_km", nearest["median"], "km",
         "#/geography/nearest_station_distance_km/median", "3_decimal_places"),
        ("figs1.registry.nearest_distance_minimum_km", nearest["minimum"], "km",
         "#/geography/nearest_station_distance_km/minimum", "3_decimal_places"),
        ("figs1.registry.neighbor_within_10km_count",
         nearest["stations_with_neighbor_within_10km"], "stations",
         "#/geography/nearest_station_distance_km/stations_with_neighbor_within_10km",
         "integer_exact"),
    ]
    for value_id, raw_value, unit, pointer, rounding in diagnostics:
        value(value_id, raw_value, unit, "PRE_DIAGNOSTIC", audit_path + pointer,
              "direct environmental-audit projection", rounding)
    value("figs1.registry.neighbor_threshold_km", 10, "km", "PRE_DESIGN_THRESHOLD",
          audit_path + "#/geography/nearest_station_distance_km",
          "threshold named by environmental-audit diagnostic", "integer_exact")
    value("figs1.projection_reconciliation_status", "PASS_EXACT_120_ROWS", "status",
          "PRE_RENDER_GATE", f"paper/si/figures/{marker_data.name}",
          "one marker-projection row per sorted retained registry row", "string_exact")
    value("figs1.basemap_status", "NO_BASEMAP", "status", "PRE_SCOPE",
          "paper/FIGURE_REDRAW_SPEC.md#Figure-S1",
          "coordinate scatter intentionally contains no map layer", "string_exact")
    value("figs1.map_source_rights_status", "NOT_APPLICABLE_NO_MAP_SOURCE", "status",
          "PRE_SCOPE", "paper/FIGURE_REDRAW_SPEC.md#Figure-S1",
          "no external basemap or boundary source is rendered", "string_exact")
    value("figs1.hydraulic_connectivity_status", "NOT_ESTABLISHED", "status",
          "PRE_SCOPE", "paper/FIGURE_REDRAW_SPEC.md#Figure-S1/Forbidden",
          "registry overlap and coordinate proximity are descriptive only", "string_exact")
    value("figs1.discovery_source_replayability", "NOT_SOURCE_REPLAYABLE", "status",
          "PRE_SCOPE", f"{protocol_path}#/development_evidence/reproducibility_scope",
          "original discovery responses, requests, timestamps, command, and full run "
          "configuration were not retained; committed ledgers support audit but not "
          "source-level replay", "string_exact")
    value("figs1.cohort_sampling_scope",
          "AVAILABILITY_ENRICHED_NOT_PROBABILITY_SAMPLE", "status", "PRE_SCOPE",
          f"{audit_path}#/selection/interpretation",
          "classify the committed selection interpretation without a population-sampling "
          "or national-representativeness claim", "string_exact")
    value("figs1.scope_status", "MATERIALIZED_PRE_STRUCTURAL_ONLY", "status",
          "PRE_SCOPE", "paper/FIGURE_REDRAW_SPEC.md#Figure-S1/State",
          "current redraw state", "string_exact")

    s1_marks = [
        mark("figs1.waterfall.initial", "a", count="route_a.cohort.initial_candidate_count"),
        mark("figs1.waterfall.rejected", "a", count="route_a.cohort.rejected_station_count"),
        mark("figs1.waterfall.reason.no_joint_nwis", "a",
             count=reason_ids["no NWIS WTEMP+FLOW"],
             reason=reason_label_ids["no NWIS WTEMP+FLOW"]),
        mark("figs1.waterfall.reason.low_full_period", "a",
             count=reason_ids["low full-period coverage"],
             reason=reason_label_ids["low full-period coverage"]),
        mark("figs1.waterfall.reason.low_2019_2020", "a",
             count=reason_ids["low blind-test-period coverage"],
             reason=reason_label_ids["low blind-test-period coverage"]),
        mark("figs1.waterfall.retained", "a", count="route_a.registry.station_count"),
        mark("figs1.waterfall.reconciliation", "a",
             rejected="route_a.cohort.rejected_station_count",
             retained="route_a.registry.station_count",
             initial="route_a.cohort.initial_candidate_count",
             status="figs1.projection_reconciliation_status"),
    ] + station_marks
    for raw_huc in sorted(huc_counts, key=int):
        huc = f"{int(raw_huc):02d}"
        s1_marks.extend([
            mark(f"figs1.legend.huc2_{huc}", "b",
                 category=f"route_a.registry.huc2_{huc}.code",
                 count=f"route_a.registry.huc2_{huc}.station_count"),
            mark(f"figs1.bar.huc2_{huc}", "c",
                 category=f"route_a.registry.huc2_{huc}.code",
                 height=f"route_a.registry.huc2_{huc}.station_count"),
        ])
    for value_id, *_ in diagnostics:
        cells = {"value": value_id}
        if value_id == "figs1.registry.neighbor_within_10km_count":
            cells["threshold"] = "figs1.registry.neighbor_threshold_km"
        s1_marks.append(mark(value_id.replace("registry", "diagnostic"), "d", **cells))
    s1_marks.extend([
        mark("figs1.annotation.no_basemap", "b", status="figs1.basemap_status"),
        mark("figs1.annotation.no_connectivity", "d",
             status="figs1.hydraulic_connectivity_status"),
    ])

    # FigS2 chronology, issue boundary, variable matrix, and bridge evidence.
    chronology_pointers = {
        "train": f"{config_path}#TimeSplit.train",
        "validation": f"{config_path}#TimeSplit.val",
        "calibration": f"{config_path}#TimeSplit.calib",
        "exploratory": f"{protocol_path}#/development_evidence/existing_2019_2020_role",
        "target": f"{protocol_path}#/time_holdout",
    }
    chronology = [
        (stage["key"], _display_year_interval(stage["interval"]), stage["role"],
         chronology_pointers[stage["key"]])
        for stage in temporal["chronology"]
    ]
    s2_marks: list[dict[str, object]] = []
    for key, interval, role, pointer in chronology:
        interval_id = value(f"figs2.chronology.{key}.interval", interval, "calendar interval",
                            "PRE_DESIGN", pointer, "format bound interval for display",
                            "year_precision")
        role_id = value(f"figs2.chronology.{key}.role", role, "role", "PRE_DESIGN",
                        pointer, "direct contract role", "string_exact")
        s2_marks.append(mark(f"figs2.chronology.{key}", "a", interval=interval_id,
                             role=role_id))
    value("figs2.issue_date", temporal["issue_date_symbol"], "symbolic date", "PRE_DESIGN",
          f"{protocol_path}#/primary_historical_input_contract/information_cutoff",
          "symbolic issue date", "symbol_exact")
    value("figs2.target_date", temporal["target_date_expression"], "symbolic date",
          "PRE_DESIGN", f"{protocol_path}#/time_holdout/"
          "issue_and_all_target_dates_must_remain_inside",
          "express each target date as issue date plus registered horizon", "symbol_exact")
    for horizon in temporal["horizons"]:
        value(f"route_a.horizon_{horizon}_days", horizon, "days", "PRE_DESIGN",
              f"{config_path}#HORIZONS", "select registered horizon", "integer_exact")
        s2_marks.append(mark(f"figs2.horizon.{horizon}", "b",
                             issue_date="figs2.issue_date",
                             target_date="figs2.target_date",
                             horizon=f"route_a.horizon_{horizon}_days"))
    boundary_rules = [
        ("allowed_observed_history", "ALLOWED", "Observed WTEMP history", "date ≤ t"),
        ("allowed_flow_weather", "ALLOWED", "FLOW + dated meteorology", "date ≤ t"),
        ("allowed_reference", "ALLOWED", "Frozen reference inputs", "date ≤ t"),
        ("excluded_target", "EXCLUDED", "Target WTEMP at t+h", "date > t"),
        ("excluded_future_weather", "EXCLUDED", "Future weather", "date > t"),
        ("excluded_future_vintage", "EXCLUDED", "Future vintages", "date > t"),
    ]
    for key, admissibility, item, date_rule in boundary_rules:
        item_id = value(f"figs2.boundary.{key}.item", item, "input class", "PRE_DESIGN",
                        f"{protocol_path}#/primary_historical_input_contract", "direct contract summary",
                        "string_exact")
        status_id = value(f"figs2.boundary.{key}.admissibility", admissibility, "status",
                          "PRE_DESIGN", f"{protocol_path}#/primary_historical_input_contract",
                          "classify by issue-date contract", "string_exact")
        date_id = value(f"figs2.boundary.{key}.date_rule", date_rule, "date rule",
                        "PRE_DESIGN", f"{protocol_path}#/primary_historical_input_contract",
                        "express relative to issue date", "symbol_exact")
        s2_marks.append(mark(f"figs2.boundary.{key}", "b", item=item_id,
                             admissibility=status_id, source_date=date_id))
    for boundary_mark in s2_marks:
        if boundary_mark["mark_id"] == "figs2.boundary.excluded_target":
            boundary_mark["cells"]["target_date"] = cell("figs2.target_date")

    provider_by_variable = temporal["provider_by_variable"]
    for variable_name in ROUTE_A_VARIABLES:
        key = variable_name.lower()
        variable_id = value(f"route_a.input.{key}.variable", variable_name, "variable ID",
                            "PRE_DESIGN", f"{protocol_path}#/primary_inference_contract/feature_order",
                            "select ordered variable", "string_exact")
        provider = provider_by_variable[variable_name]
        provider_id = value(f"route_a.input.{key}.provider", provider, "provider/product",
                            "PRE_DESIGN", f"{protocol_path}#/primary_historical_input_contract",
                            "map variable to frozen source product", "string_exact")
        date_id = value(f"route_a.input.{key}.source_date_rule",
                        temporal["source_date_display"], "date rule",
                        "PRE_DESIGN", f"{protocol_path}#/primary_historical_input_contract/information_cutoff",
                        "apply issue-date cutoff", "symbol_exact")
        retrieval_id = value(f"route_a.input.{key}.retrieval_vintage",
                             temporal["retrieval_vintage_display"], "vintage rule", "PRE_SCOPE",
                             f"{protocol_path}#/primary_historical_input_contract/provisional_vintage_limitation",
                             "retrospective product limitation", "string_exact")
        s2_marks.append(mark(f"figs2.matrix.{key}", "c", variable=variable_id,
                             provider=provider_id, source_date=date_id,
                             retrieval_vintage=retrieval_id))

    bridge_values = [
        ("figs2.bridge.status", bridge_report["status"], "status", "PRE_BRIDGE",
         f"{bridge_report_path}#/status", "direct report status", "string_exact"),
        ("figs2.bridge.start_date", bridge_report["interval"]["start"], "date", "PRE_BRIDGE",
         f"{bridge_report_path}#/interval/start", "direct report interval endpoint", "day_precision"),
        ("figs2.bridge.end_date", bridge_report["interval"]["end"], "date", "PRE_BRIDGE",
         f"{bridge_report_path}#/interval/end", "direct report interval endpoint", "day_precision"),
        ("figs2.bridge.site_count", bridge_report["site_count"], "sites", "PRE_BRIDGE",
         f"{bridge_report_path}#/site_count", "direct report count", "integer_exact"),
        ("figs2.bridge.row_count", bridge_report["row_count"], "rows", "PRE_BRIDGE",
         f"{bridge_report_path}#/row_count", "direct report count", "integer_exact"),
        ("figs2.bridge.request_count", request_map["request_count"], "requests", "PRE_BRIDGE",
         f"{bridge_map_path}#/request_count", "direct request-map count", "integer_exact"),
        ("figs2.bridge.alignment_window", 1, "days each direction", "PRE_BRIDGE",
         f"{bridge_report_path}#/fields/*/date_shift_sensitivity",
         "maximum absolute tested date shift", "integer_exact"),
        ("figs2.bridge.field_count", len(BRIDGE_FIELDS), "meteorological fields",
         "PRE_BRIDGE", f"{bridge_report_path}#/fields", "count exact report field keys",
         "integer_exact"),
        ("figs2.bridge.field_registry", list(BRIDGE_FIELDS), "ordered variable IDs",
         "PRE_BRIDGE", f"{bridge_report_path}#/fields",
         "canonical display order of the five bridge-tested fields", "string_list_exact"),
        ("figs2.bridge.product_parser_compatibility", True, "boolean", "PRE_BRIDGE",
         f"{bridge_report_path}#/fields", "all fields exact_product_compatibility=true", "boolean_exact"),
        ("figs2.bridge.missing_pattern_agreement", True, "boolean", "PRE_BRIDGE",
         f"{bridge_report_path}#/fields", "all fields missing_pattern_exact=true", "boolean_exact"),
        ("figs2.bridge.zero_day_alignment_best_or_tied", True, "boolean", "PRE_BRIDGE",
         f"{bridge_report_path}#/fields", "all fields zero-day alignment best or tied", "boolean_exact"),
        ("figs2.bridge.as_issued_availability", "NOT_ESTABLISHED", "status", "PRE_SCOPE",
         f"{bridge_report_path}#/interpretation_limit", "direct interpretation limit", "string_exact"),
        ("figs2.bridge.local_day_equivalence", "NOT_ESTABLISHED", "status", "PRE_SCOPE",
         f"{bridge_report_path}#/interpretation_limit", "direct interpretation limit", "string_exact"),
        ("figs2.bridge.target_period_availability", "NOT_ESTABLISHED", "status", "PRE_SCOPE",
         f"{protocol_path}#/primary_historical_input_contract/required_evidence_before_label_opening",
         "development bridge cannot establish target-period availability", "string_exact"),
        ("figs2.bridge.operational_replay", "NOT_ESTABLISHED", "status", "PRE_SCOPE",
         f"{protocol_path}#/primary_historical_input_contract/operational_replay_claim_allowed",
         "protocol explicitly disallows operational-replay claim", "string_exact"),
        ("figs2.bridge.outcomes_read", False, "boolean", "PRE_SCOPE",
         f"{bridge_report_path}#/outcome_values_requested_or_read",
         "direct outcome-access flag", "boolean_exact"),
        ("figs2.scope_status", "MATERIALIZED_PRE_DESIGN_ONLY", "status", "PRE_SCOPE",
         "paper/FIGURE_REDRAW_SPEC.md#Figure-S2/State", "current redraw state", "string_exact"),
    ]
    for args in bridge_values:
        value(*args)
    s2_marks.extend([
        mark("figs2.bridge.capability", "d", status="figs2.bridge.status",
             start="figs2.bridge.start_date", end="figs2.bridge.end_date",
             sites="figs2.bridge.site_count", rows="figs2.bridge.row_count",
             requests="figs2.bridge.request_count",
             field_count="figs2.bridge.field_count",
             fields="figs2.bridge.field_registry",
             product_parser="figs2.bridge.product_parser_compatibility",
             missing_pattern="figs2.bridge.missing_pattern_agreement",
             alignment="figs2.bridge.zero_day_alignment_best_or_tied",
             alignment_window="figs2.bridge.alignment_window"),
        mark("figs2.bridge.limitations", "d", as_issued="figs2.bridge.as_issued_availability",
             local_day="figs2.bridge.local_day_equivalence",
             target_period="figs2.bridge.target_period_availability",
             operational_replay="figs2.bridge.operational_replay",
             outcomes_read="figs2.bridge.outcomes_read"),
    ])

    # FigS3 implementation/configuration assertions and visible output dataflow.
    value("figs3.input.variable_count", len(architecture["variables"]), "variables",
          "PRE_IMPLEMENTATION", f"{suite_path}#STAGE9_USGS_VARIABLES",
          "length of asserted ordered registry", "integer_exact")
    for order, variable_name in enumerate(architecture["variables"], start=1):
        value(f"figs3.input.variable_{order}.name", variable_name, "variable ID",
              "PRE_IMPLEMENTATION", f"{suite_path}#STAGE9_USGS_VARIABLES[{order - 1}]",
              "direct ordered source projection", "string_exact")
        value(f"figs3.input.variable_{order}.order", order, "ordinal",
              "PRE_IMPLEMENTATION", f"{suite_path}#STAGE9_USGS_VARIABLES",
              "one-based list position", "integer_exact")
    architecture_values = [
        ("figs3.input.missingness_mask", architecture["missingness_mask"], "boolean", "PRE_IMPLEMENTATION",
         f"{protocol_path}#/primary_inference_contract/primary_estimand/admissible_issue",
         "explicit mask retained by input contract", "boolean_exact"),
        ("figs3.input.wlevel_consumed", False, "boolean", "PRE_IMPLEMENTATION",
         f"{protocol_path}#/primary_inference_contract/wlevel_consumed",
         "direct protocol flag plus model default assertion", "boolean_exact"),
        ("figs3.input.context_buffer_days", architecture["context_length"], "days",
         "PRE_IMPLEMENTATION", f"{config_path}#CONTEXT_LENGTH", "literal source constant", "integer_exact"),
        ("figs3.anchor.identity", architecture["anchor_identity"], "module label",
         "PRE_IMPLEMENTATION", f"{features_path}#DampedPersistenceAnchor",
         "visible identity after class/formula assertion", "string_exact"),
        ("figs3.anchor.method", architecture["anchor_method"], "method identity",
         "PRE_IMPLEMENTATION", f"{features_path}#DampedPersistenceAnchor.fit",
         "source-projected frozen fitted-anchor description", "string_exact"),
        ("figs3.anchor.composition", architecture["anchor_composition"],
         "composition label", "PRE_IMPLEMENTATION",
         f"{features_path}#DampedPersistenceAnchor.predict + "
         f"{model_path}#ThermoRoute.forward/anchor",
         "describe asserted fitted blend without exposing fitted coefficients", "string_exact"),
        ("figs3.proposal.identity", "PROPOSAL P", "module label", "PRE_DESIGN",
         f"{model_path}#DynamicThermalRelaxationPrior",
         "visible identity for asserted learned-proposal module", "string_exact"),
        ("figs3.proposal.operation", "learned relaxation κ", "operation label",
         "PRE_IMPLEMENTATION", f"{model_path}#DynamicThermalRelaxationPrior.forward",
         "source-projected learned kappa operation", "string_exact"),
        ("figs3.mixture.identity", "MIXTURE OF EXPERTS", "module label", "PRE_DESIGN",
         f"{model_path}#RegimeMoE",
         "visible identity for asserted mixture module", "string_exact"),
        ("figs3.point_head.identity", "POINT HEAD", "head label", "PRE_DESIGN",
         f"{model_path}#ThermoRoute.head_delta",
         "visible identity for asserted distinct point head", "string_exact"),
        ("figs3.point_head.objective", "MSE residual r", "objective label",
         "PRE_IMPLEMENTATION", f"{training_path}#composite_loss_terms/point",
         "compact label for the asserted conditional-mean MSE point objective", "string_exact"),
        ("figs3.event_head.identity", "EVENT HEAD", "head label", "PRE_DESIGN",
         f"{model_path}#ThermoRoute.head_evt",
         "visible identity for asserted distinct event head", "string_exact"),
        ("figs3.event_head.output", "event score", "output label", "PRE_IMPLEMENTATION",
         f"{model_path}#ThermoRoute.forward/evt",
         "compact identity for the asserted pre-Platt event-head score", "string_exact"),
        ("figs3.calibration.cqr_aggregation", "member average", "operation label",
         "PRE_DESIGN",
         f"{protocol_path}#/primary_inference_contract/probabilistic_event_contract/"
         "ensemble_quantiles", "compact projection of member-wise averaging", "string_exact"),
        ("figs3.router.minimum_lag_days", 0, "days", "PRE_IMPLEMENTATION",
         f"{model_path}#DynamicLagRouter", "inclusive lag-range lower endpoint", "integer_exact"),
        ("figs3.router.maximum_lag_days", architecture["max_router_lag"], "days",
         "PRE_IMPLEMENTATION", f"{config_path}#MAX_ROUTER_LAG", "literal source constant", "integer_exact"),
        ("figs3.tcn.block_count", architecture["tcn_blocks"], "blocks", "PRE_IMPLEMENTATION",
         f"{config_path}#TrainConfig.encoder_blocks", "literal dataclass default", "integer_exact"),
        ("figs3.tcn.kernel_size", architecture["tcn_kernel"], "steps", "PRE_IMPLEMENTATION",
         f"{config_path}#TrainConfig.kernel_size", "literal dataclass default", "integer_exact"),
        ("figs3.tcn.receptive_field_steps", architecture["tcn_receptive_field"], "steps",
         "PRE_IMPLEMENTATION", f"{config_path}#TrainConfig.encoder_blocks+kernel_size",
         "1 + (kernel-1) × sum(2^block_index)", "integer_exact"),
        ("figs3.anchor.delta_celsius", architecture["delta_scale"], "degrees_C",
         "PRE_IMPLEMENTATION", f"{config_path}#DELTA_SCALE", "literal source constant", "1_decimal_place"),
        ("figs3.quantile.q05", architecture["quantiles"][0], "quantile probability", "PRE_IMPLEMENTATION",
         f"{config_path}#QUANTILES", "select nominal lower head", "2_decimal_places"),
        ("figs3.quantile.q50", architecture["quantiles"][1], "quantile probability", "PRE_IMPLEMENTATION",
         f"{config_path}#QUANTILES", "select nominal median head", "2_decimal_places"),
        ("figs3.quantile.q95", architecture["quantiles"][2], "quantile probability", "PRE_IMPLEMENTATION",
         f"{config_path}#QUANTILES", "select nominal upper head", "2_decimal_places"),
        ("figs3.calibration.cqr_fit_period", architecture["calibration_display"], "calendar year", "PRE_DESIGN",
         f"{config_path}#TimeSplit.calib", "format calibration interval as year", "year_precision"),
        ("figs3.calibration.platt_fit_period", architecture["calibration_display"], "calendar year", "PRE_DESIGN",
         f"{config_path}#TimeSplit.calib", "format calibration interval as year", "year_precision"),
        ("figs3.output.final_interval", "DEPLOYMENT_OUTPUT", "status", "PRE_DESIGN",
         f"{protocol_path}#/primary_inference_contract/probabilistic_event_contract/interval_calibration",
         "CQR-adjusted member-averaged interval output", "string_exact"),
        ("figs3.output.final_probability", "DEPLOYMENT_OUTPUT", "status", "PRE_DESIGN",
         f"{protocol_path}#/primary_inference_contract/probabilistic_event_contract/event_calibration",
         "Platt-calibrated event probability output", "string_exact"),
        ("figs3.calibration.cqr_contract", architecture["interval_calibration_contract"],
         "contract", "PRE_DESIGN",
         f"{protocol_path}#/primary_inference_contract/probabilistic_event_contract/interval_calibration",
         "direct frozen interval-calibration contract", "string_exact"),
        ("figs3.calibration.platt_contract", architecture["event_calibration_contract"],
         "contract", "PRE_DESIGN",
         f"{protocol_path}#/primary_inference_contract/probabilistic_event_contract/event_calibration",
         "direct frozen event-calibration contract", "string_exact"),
        ("figs3.calibration.erratum_status", architecture["calibration_erratum_status"],
         "status", "PRE_DESIGN",
         "protocols/route_a_calibration_inclusion_erratum_v1.json#/status",
         "direct outcome-free calibration-erratum status", "string_exact"),
        ("figs3.anchor.truth_error_bound", "NOT_ESTABLISHED", "status", "PRE_SCOPE",
         "paper/FIGURE_REDRAW_SPEC.md#Figure-S3/Forbidden",
         "algebraic deviation bound is not truth-error bound", "string_exact"),
        ("figs3.anchor.safety_bound", "NOT_ESTABLISHED", "status", "PRE_SCOPE",
         "paper/FIGURE_REDRAW_SPEC.md#Figure-S3/Forbidden",
         "algebraic deviation bound is not safety bound", "string_exact"),
        ("figs3.router.physical_routing", "NOT_ESTABLISHED", "status", "PRE_SCOPE",
         "paper/FIGURE_REDRAW_SPEC.md#Figure-S3/Forbidden",
         "latent statistical allocation has no physical-routing interpretation", "string_exact"),
        ("figs3.scope_status", "MATERIALIZED_PRE_DESIGN_ONLY_SUITE_GATED_FINAL", "status",
         "PRE_SCOPE", "paper/FIGURE_REDRAW_SPEC.md#Figure-S3/State",
         "current redraw state", "string_exact"),
        ("figs3.bounded_correction.proposal_equation", "z = P - A + r_theta",
         "equation identity", "PRE_IMPLEMENTATION",
         "paper/FIGURE_REDRAW_SPEC.md#Figure-S3/panel-c",
         "unrestricted learned displacement relative to the frozen anchor (relocated from Figure 1b)",
         "string_exact"),
        ("figs3.bounded_correction.correction_equation", "y_hat = A + delta * tanh(z / delta)",
         "equation identity", "PRE_IMPLEMENTATION",
         "paper/FIGURE_REDRAW_SPEC.md#Figure-S3/panel-c",
         "damped anchor plus delta-bounded tanh displacement (relocated from Figure 1b)",
         "string_exact"),
        ("figs3.bounded_correction.deviation_bound", "abs(y_hat - A) < delta",
         "algebraic bound identity", "PRE_IMPLEMENTATION",
         "paper/FIGURE_REDRAW_SPEC.md#Figure-S3/panel-c",
         "abs(tanh(u))<1 for finite u and positive finite delta",
         "string_exact"),
        ("figs3.bounded_correction.warning_status",
         "DEVIATION_FROM_ANCHOR_NOT_ERROR_OR_SAFETY_BOUND", "scope status", "PRE_SCOPE",
         "paper/FIGURE_REDRAW_SPEC.md#Figure-S3/panel-c",
         "in-panel warning carried by the relocated A+/-delta envelope",
         "string_exact"),
    ]
    for args in architecture_values:
        value(*args)

    fig1_shared_ids: list[str] = []
    if FIG1_BINDER.is_file():
        fig1_manifest = json.loads(FIG1_BINDER.read_text(encoding="utf-8"))
        fig1_values = fig1_manifest.get("values")
        if not isinstance(fig1_values, dict):
            raise RuntimeError("Refusing redraw: Figure 1 binder lacks a values registry")
        fig1_shared_ids = sorted(set(values) & set(fig1_values))
        for value_id in fig1_shared_ids:
            local_pair = (values[value_id]["value"], values[value_id]["unit"])
            fig1_pair = (fig1_values[value_id].get("value"), fig1_values[value_id].get("unit"))
            if local_pair != fig1_pair:
                raise RuntimeError(
                    f"Refusing redraw: cross-figure value/unit conflict for {value_id}: "
                    f"SI={local_pair!r}, Fig1={fig1_pair!r}"
                )
    s3_marks: list[dict[str, object]] = []
    for order in range(1, len(architecture["variables"]) + 1):
        s3_marks.append(mark(f"figs3.input.variable_{order}", "a",
                             name=f"figs3.input.variable_{order}.name",
                             order=f"figs3.input.variable_{order}.order"))
    s3_marks.extend([
        mark("figs3.input.contract", "a", count="figs3.input.variable_count",
             mask="figs3.input.missingness_mask", wlevel="figs3.input.wlevel_consumed",
             buffer="figs3.input.context_buffer_days"),
        mark("figs3.router", "b", variable_count="figs3.input.variable_count",
             lag_min="figs3.router.minimum_lag_days", lag_max="figs3.router.maximum_lag_days"),
        mark("figs3.tcn", "b", blocks="figs3.tcn.block_count",
             kernel="figs3.tcn.kernel_size", receptive_field="figs3.tcn.receptive_field_steps"),
        mark("figs3.anchor_module", "a", identity="figs3.anchor.identity",
             method="figs3.anchor.method", composition="figs3.anchor.composition"),
        mark("figs3.proposal", "b", identity="figs3.proposal.identity",
             operation="figs3.proposal.operation"),
        mark("figs3.mixture", "b", identity="figs3.mixture.identity"),
        mark("figs3.point_head", "c", identity="figs3.point_head.identity",
             objective="figs3.point_head.objective"),
        mark("figs3.event_head", "c", identity="figs3.event_head.identity",
             output="figs3.event_head.output"),
        mark("figs3.quantile_heads", "c", q05="figs3.quantile.q05",
             q50="figs3.quantile.q50", q95="figs3.quantile.q95"),
        mark("figs3.anchor_bound", "c", delta="figs3.anchor.delta_celsius",
             truth_error="figs3.anchor.truth_error_bound", safety="figs3.anchor.safety_bound"),
        mark("figs3.bounded_correction_schematic", "c",
             anchor="figs3.anchor.identity",
             proposal_equation="figs3.bounded_correction.proposal_equation",
             correction_equation="figs3.bounded_correction.correction_equation",
             delta="figs3.anchor.delta_celsius",
             deviation_bound="figs3.bounded_correction.deviation_bound",
             warning="figs3.bounded_correction.warning_status"),
        mark("figs3.cqr", "d", fit_period="figs3.calibration.cqr_fit_period",
             aggregation="figs3.calibration.cqr_aggregation",
             contract="figs3.calibration.cqr_contract",
             erratum_status="figs3.calibration.erratum_status",
             output="figs3.output.final_interval"),
        mark("figs3.platt", "d", fit_period="figs3.calibration.platt_fit_period",
             contract="figs3.calibration.platt_contract",
             erratum_status="figs3.calibration.erratum_status",
             output="figs3.output.final_probability"),
        mark("figs3.no_physical_routing", "b", status="figs3.router.physical_routing"),
    ])

    figure_specs = {
        "FigS1": {
            "path": HERE / "figS1_cohort_registry.svg", "height_mm": 152,
            "description": "four-panel cohort ledger, coordinate/HUC2 display, bars, and audit",
            "source_keys": ["registry", "panel", "rejection_ledger", "environmental_audit",
                            "marker_projection", "redraw_specification",
                            "main_figure_1_binder"],
            "panel_order": ["a", "b", "c", "d"], "marks": s1_marks,
            "caption_value_ids": [
                "route_a.cohort.initial_candidate_count", "route_a.cohort.rejected_station_count",
                *reason_ids.values(), *reason_label_ids.values(), "route_a.registry.station_count",
                "route_a.registry.huc2_group_count", "figs1.registry.unique_huc_code_count",
                "figs1.registry.repeated_huc_station_count",
                "figs1.registry.nearest_distance_median_km",
                "figs1.registry.nearest_distance_minimum_km",
                "figs1.registry.neighbor_within_10km_count",
                "figs1.registry.neighbor_threshold_km", "figs1.basemap_status",
                "figs1.map_source_rights_status", "figs1.projection_reconciliation_status",
                "figs1.hydraulic_connectivity_status",
                "figs1.discovery_source_replayability", "figs1.cohort_sampling_scope",
            ],
            "scope_status_value_id": "figs1.scope_status",
        },
        "FigS2": {
            "path": HERE / "figS2_information_boundary.svg", "height_mm": 170,
            "description": "four-panel chronology, issue boundary, source matrix, and bridge limits",
            "source_keys": ["protocol", "configuration_source", "bridge_manifest",
                            "bridge_report", "bridge_request_map", "bridge_panel",
                            "bridge_registry", "bridge_raw_daymet_index",
                            "bridge_raw_gridmet_index", "bridge_raw_gridmet_schema_index",
                            "bridge_normalized_frozen", "bridge_normalized_refreshed",
                            "redraw_specification", "main_figure_1_binder"],
            "panel_order": ["a", "b", "c", "d"], "marks": s2_marks,
            "caption_value_ids": [
                *[f"figs2.chronology.{key}.interval" for key, *_ in chronology],
                *[f"figs2.chronology.{key}.role" for key, *_ in chronology],
                "figs2.issue_date", "figs2.target_date",
                "route_a.horizon_1_days", "route_a.horizon_3_days", "route_a.horizon_7_days",
                *[
                    f"figs2.boundary.{key}.{field}"
                    for key, *_ in boundary_rules
                    for field in ("item", "admissibility", "date_rule")
                ],
                *[
                    f"route_a.input.{variable_name.lower()}.{field}"
                    for variable_name in ROUTE_A_VARIABLES
                    for field in ("variable", "provider", "source_date_rule",
                                  "retrieval_vintage")
                ],
                "figs2.bridge.status", "figs2.bridge.start_date", "figs2.bridge.end_date",
                "figs2.bridge.site_count", "figs2.bridge.row_count",
                "figs2.bridge.request_count", "figs2.bridge.field_count",
                "figs2.bridge.field_registry", "figs2.bridge.alignment_window",
                "figs2.bridge.product_parser_compatibility",
                "figs2.bridge.missing_pattern_agreement",
                "figs2.bridge.zero_day_alignment_best_or_tied",
                "figs2.bridge.as_issued_availability", "figs2.bridge.local_day_equivalence",
                "figs2.bridge.target_period_availability",
                "figs2.bridge.operational_replay", "figs2.bridge.outcomes_read",
            ],
            "scope_status_value_id": "figs2.scope_status",
        },
        "FigS3": {
            "path": HERE / "figS3_model_architecture.svg", "height_mm": 148,
            "description": "four-column PRE model, bound, heads, and calibration dataflow",
            "source_keys": ["protocol", "calibration_erratum", "configuration_source",
                            "model_suite_source", "model_source", "anchor_source",
                            "training_source", "redraw_specification",
                            "main_figure_1_binder"],
            "panel_order": ["a", "b", "c", "d"], "marks": s3_marks,
            "caption_value_ids": [
                "figs3.input.variable_count", "figs3.input.wlevel_consumed",
                "figs3.input.missingness_mask",
                *[
                    f"figs3.input.variable_{order}.{field}"
                    for order in range(1, len(architecture["variables"]) + 1)
                    for field in ("name", "order")
                ],
                "figs3.input.context_buffer_days", "figs3.router.minimum_lag_days",
                "figs3.router.maximum_lag_days", "figs3.tcn.block_count",
                "figs3.tcn.kernel_size", "figs3.tcn.receptive_field_steps",
                "figs3.anchor.delta_celsius", "figs3.quantile.q05", "figs3.quantile.q50",
                "figs3.quantile.q95", "figs3.calibration.cqr_fit_period",
                "figs3.calibration.platt_fit_period", "figs3.calibration.cqr_contract",
                "figs3.calibration.platt_contract", "figs3.calibration.erratum_status",
                "figs3.anchor.identity", "figs3.anchor.method", "figs3.anchor.composition",
                "figs3.proposal.identity", "figs3.proposal.operation",
                "figs3.mixture.identity", "figs3.point_head.identity",
                "figs3.point_head.objective", "figs3.event_head.identity",
                "figs3.event_head.output", "figs3.calibration.cqr_aggregation",
                "figs3.output.final_interval", "figs3.output.final_probability",
                "figs3.anchor.truth_error_bound",
                "figs3.anchor.safety_bound", "figs3.router.physical_routing",
                "figs3.bounded_correction.proposal_equation",
                "figs3.bounded_correction.correction_equation",
                "figs3.bounded_correction.deviation_bound",
                "figs3.bounded_correction.warning_status",
            ],
            "scope_status_value_id": "figs3.scope_status",
        },
    }

    figures: dict[str, object] = {}
    expected_figure_status = {
        "FigS1": "MATERIALIZED_PRE_STRUCTURAL_ONLY",
        "FigS2": "MATERIALIZED_PRE_DESIGN_ONLY",
        "FigS3": "MATERIALIZED_PRE_DESIGN_ONLY_SUITE_GATED_FINAL",
    }
    for figure_id, spec in figure_specs.items():
        path = spec["path"]
        ElementTree.parse(path)
        svg = path.read_text(encoding="utf-8")
        font_sizes = [float(value) for value in re.findall(r"font:\s*(?:700\s+)?([0-9.]+)px", svg)]
        artifacts = {}
        for suffix, artifact_format in [(".svg", "SVG"), (".pdf", "PDF"), (".png", "PNG")]:
            artifact_path = path.with_suffix(suffix)
            artifacts[artifact_format] = {
                "path": str(artifact_path.relative_to(REPO)),
                "sha256": sha256(artifact_path),
                "byte_count": artifact_path.stat().st_size,
            }
        artifacts["SVG"]["vector"] = True
        artifacts["PDF"]["vector"] = True
        artifacts["PDF"]["font_embedding"] = "TrueType (Matplotlib pdf.fonttype=42)"
        artifacts["PNG"]["dpi"] = PNG_DPI
        marks = spec["marks"]
        panels = {
            panel_id: {"marks": [item for item in marks if item["panel_id"] == panel_id]}
            for panel_id in spec["panel_order"]
        }
        scope_status = values[spec["scope_status_value_id"]]["value"]
        if scope_status != expected_figure_status[figure_id]:
            raise RuntimeError(f"Refusing manifest: figure/scope status drift for {figure_id}")
        figures[figure_id] = {
            "figure_id": figure_id,
            "figure_schema_version": "thermoroute.figure_binder.v1",
            "source_bindings": [binding(key) for key in spec["source_keys"]],
            "panel_order": spec["panel_order"],
            "panels": panels,
            "mark_registry": [
                item["mark_id"]
                for panel_id in spec["panel_order"]
                for item in panels[panel_id]["marks"]
            ],
            "caption_value_ids": spec["caption_value_ids"],
            "scope_status_value_id": spec["scope_status_value_id"],
            "render_profile": {
                "placed_width_mm": round(figstyle.FULL_MM, 1),
                "placed_height_mm": spec["height_mm"],
                "background": "white",
                "minimum_nominal_text_pt": min(font_sizes),
                "minimum_module_text_inset_mm": 2.0,
                "semantic_palette_binding":
                    "paper/FIGURE_REDRAW_SPEC.md#3.1-Semantic-palette",
                "svg_accessibility": "role=img; aria-labelledby=svg-title svg-desc",
                "pdf_fonttype": 42,
                "png_dpi": PNG_DPI,
            },
            "artifacts": artifacts,
            "width_mm": round(figstyle.FULL_MM, 1),
            "height_mm": spec["height_mm"],
            "white_background": True,
            "svg_xml_well_formed": True,
            "minimum_nominal_text_pt": min(font_sizes),
            "description": spec["description"],
            "status": scope_status,
        }

    manifest = {
        "format": "thermoroute.pre_supporting_figures.v2",
        "status": "PRE_MATERIALIZED_NO_TARGET_RESULTS",
        "contains_target_outcomes": False,
        "contains_performance_or_effect_coordinates": False,
        "values": values,
        "renderer": {
            "path": str(renderer.relative_to(REPO)),
            "sha256": sha256(renderer),
            "command": "python paper/si/figures/render_pre_supporting_figures.py",
            "matplotlib_version": mpl.__version__,
            "svg_hashsalt": mpl.rcParams["svg.hashsalt"],
        },
        "captions": {"path": str(captions.relative_to(REPO)), "sha256": sha256(captions)},
        "sources": sources,
        "cross_figure_canonical_check": {
            "binder_path": str(FIG1_BINDER.relative_to(REPO)),
            "binder_sha256": sha256(FIG1_BINDER),
            "compared_fields": ["value", "unit"],
            "shared_value_ids": fig1_shared_ids,
            "all_shared_values_match": True,
        },
        "figures": figures,
        "semantic_palette": {
            "status": "PASS_EXACT_SPEC_TOKENS",
            "source_binding": "paper/FIGURE_REDRAW_SPEC.md#3.1-Semantic-palette",
            "tokens_used": SEMANTIC_TOKENS,
            "categorical_note": (
                "Okabe–Ito colors remain auxiliary for HUC2 and chronology categories; "
                "allowed/verified semantics use ALLOWED_TEAL"
            ),
        },
        "figS1_marker_projection": {
            "path": str(marker_data.relative_to(REPO)),
            "sha256": sha256(marker_data),
            "columns": ["site_no", "state", "lat", "lon", "huc2", "site_value_id",
                        "state_value_id", "x_value_id", "y_value_id", "huc2_value_id"],
            "row_count": 120,
            "one_row_per_visible_coordinate_marker": True,
            "site_state_coordinate_category_cells_have_distinct_value_ids": True,
            "reconciliation_status_value_id": "figs1.projection_reconciliation_status",
        },
        "figS1_huc2_encoding": {
            "rule": "eight shapes crossed with filled/open state; color is auxiliary",
            "grayscale_redundancy": "unique shape-by-fill combination for every HUC2",
            "all_combinations_unique": True,
            "categories": huc_encoding_categories,
        },
        "source_ledger_summary": {
            "initial_candidate_count": 1465,
            "retained_station_count": 120,
            "rejected_station_count": 1345,
            "rejection_reason_counts_as_recorded": dict(sorted(reasons.items())),
            "panel_row_count": 657480,
        },
        "source_assertions": {
            "temporal_information_contract": {
                "status": "PASS_EXACT_SOURCE_PROJECTION",
                "config_path": config_path,
                "config_sha256": sha256(REPO / config_path),
                "protocol_path": protocol_path,
                "protocol_sha256": sha256(REPO / protocol_path),
                "horizons": list(temporal["horizons"]),
                "splits": {key: list(interval) for key, interval in temporal["splits"].items()},
                "chronology": [
                    {**stage, "interval": list(stage["interval"])}
                    for stage in temporal["chronology"]
                ],
                "development_role": temporal["development_role"],
                "development_reproducibility_scope":
                    temporal["development_reproducibility_scope"],
                "target_start": temporal["target_start"],
                "primary_target_start": temporal["primary_target_start"],
                "target_end": temporal["target_end"],
                "information_cutoff": temporal["information_cutoff"],
                "horizon_specific_future_nwp_consumed":
                    temporal["horizon_specific_future_nwp_consumed"],
                "operational_replay_claim_allowed":
                    temporal["operational_replay_claim_allowed"],
                "provisional_vintage_limitation": temporal["provisional_vintage_limitation"],
                "provider_by_variable": temporal["provider_by_variable"],
                "source_date_display": temporal["source_date_display"],
                "issue_date_symbol": temporal["issue_date_symbol"],
                "target_date_expression": temporal["target_date_expression"],
                "request_map_path": bridge_map_path,
                "request_map_sha256": sha256(REPO / bridge_map_path),
                "bridge_report_path": bridge_report_path,
                "bridge_report_sha256": sha256(REPO / bridge_report_path),
            },
            "architecture_probability_contract": {
                "status": "PASS_EXACT_SOURCE_PROJECTION",
                "model_path": model_path,
                "model_sha256": sha256(REPO / model_path),
                "anchor_source_path": features_path,
                "anchor_source_sha256": sha256(REPO / features_path),
                "training_source_path": training_path,
                "training_source_sha256": sha256(REPO / training_path),
                "quantiles": list(architecture["quantiles"]),
                "missingness_mask": architecture["missingness_mask"],
                "calibration_interval": list(architecture["calibration_interval"]),
                "calibration_display": architecture["calibration_display"],
                "interval_calibration_contract":
                    architecture["interval_calibration_contract"],
                "event_calibration_contract": architecture["event_calibration_contract"],
                "calibration_erratum_status": architecture["calibration_erratum_status"],
                "topology_status": "PASS_EXACT_AST_SOURCE_ASSERTIONS",
                "topology_assertions": architecture["topology_assertions"],
            },
        },
        "bridge_assertions": {
            "raw_snapshot_leaves": raw_leaf_summary,
            "manifest": {
                "path": str(BRIDGE_MANIFEST.relative_to(REPO)),
                "sha256": sha256(BRIDGE_MANIFEST),
                "expected_frozen_sha256": EXPECTED_BRIDGE_MANIFEST_SHA256,
                "status": bridge_manifest["status"],
                "source_tree_sha256": bridge_manifest["source_tree_sha256"],
                "source_tree_validation": (
                    "equals the frozen implementation-provenance expectation; it is not a "
                    "raw-leaf tree digest and is not recomputed against later working-tree state"
                ),
                "panel": bridge_manifest["panel"],
                "registry": bridge_manifest["registry"],
                "report": bridge_manifest["report"],
                "request_map": bridge_manifest["request_map"],
                "raw_snapshot_indexes": bridge_manifest["raw_snapshot_indexes"],
                "normalized": bridge_manifest["normalized"],
                "all_nested_paths_exist_and_sha256_match": True,
                "outcome_values_requested_or_read":
                    bridge_manifest["outcome_values_requested_or_read"],
            },
            "report": {
                "status": bridge_report["status"],
                "interval": bridge_report["interval"],
                "site_count": bridge_report["site_count"],
                "row_count": bridge_report["row_count"],
                "fields": {
                    field: {
                        "exact_product_compatibility":
                            bridge_report["fields"][field]["exact_product_compatibility"],
                        "missing_pattern_exact":
                            bridge_report["fields"][field]["missing_pattern_exact"],
                        "zero_day_alignment_best_or_tied":
                            bridge_report["fields"][field]["zero_day_alignment_best_or_tied"],
                    }
                    for field in BRIDGE_FIELDS
                },
                "failures": bridge_report["failures"],
                "interpretation_limit": bridge_report["interpretation_limit"],
                "outcome_values_requested_or_read":
                    bridge_report["outcome_values_requested_or_read"],
            },
            "request_map": {
                "interval": request_map["interval"],
                "request_count": request_map["request_count"],
                "outcome_values_requested_or_read":
                    request_map["outcome_values_requested_or_read"],
            },
        },
        "qa_contract": {
            "text_bbox_canvas_containment": "PASS_REQUIRED_AT_RENDER",
            "registered_module_text_inset_mm": 2.0,
            "minimum_visible_svg_stroke_width_pt": 0.6,
            "figS1_huc2_grayscale_shape_fill_uniqueness": "PASS_REQUIRED",
            "svg_xml_parse": "PASS_REQUIRED",
            "svg_root_role_img": "PASS_REQUIRED",
            "pdf_embedded_fonts": "PASS_REQUIRED",
            "double_render_byte_identity": "PASS_REQUIRED",
        },
    }
    path = HERE / "pre_supporting_figures_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=("all", "S1", "S2", "S3"), default="all",
                        help="figure to render (default: all)")
    args = parser.parse_args()

    outputs: list[Path] = []
    if args.figure in {"all", "S1"}:
        registry, reasons = verify_frozen_inputs()
        audit = verify_environmental_audit(registry, reasons)
        svg = render_fig_s1(registry, reasons, audit)
        outputs.extend([svg, svg.with_suffix(".pdf"), svg.with_suffix(".png")])
        outputs.append(write_s1_marker_projection(registry))
    if args.figure in {"all", "S2"}:
        bridge_manifest, bridge_report, request_map, raw_leaf_summary = verify_predictor_bridge()
        temporal = verify_temporal_source_contract(request_map, bridge_report)
        svg = render_fig_s2(temporal, bridge_report, request_map)
        outputs.extend([svg, svg.with_suffix(".pdf"), svg.with_suffix(".png")])
    if args.figure in {"all", "S3"}:
        architecture = verify_architecture_sources()
        svg = render_fig_s3(architecture)
        outputs.extend([svg, svg.with_suffix(".pdf"), svg.with_suffix(".png")])
    if args.figure == "all":
        outputs.append(write_manifest(registry, reasons, audit, bridge_manifest,
                                      bridge_report, request_map, raw_leaf_summary,
                                      temporal, architecture))
    for output in outputs:
        print(output.relative_to(REPO))


if __name__ == "__main__":
    main()

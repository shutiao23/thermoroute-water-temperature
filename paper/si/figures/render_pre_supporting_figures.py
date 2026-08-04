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
from collections import Counter
from pathlib import Path
from typing import Sequence
from xml.etree import ElementTree

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = REPO / "data_usgs"

MM = 1 / 25.4
FULL_WIDTH = 140 * MM

# Okabe--Ito palette. Neutral greys are used only for structure and caveats.
OI = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
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

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "text.color": SEMANTIC_TOKENS["NEUTRAL_INK"],
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.0,
        "axes.edgecolor": SEMANTIC_TOKENS["NEUTRAL_INK"],
        "axes.labelcolor": SEMANTIC_TOKENS["NEUTRAL_INK"],
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.color": SEMANTIC_TOKENS["NEUTRAL_INK"],
        "ytick.color": SEMANTIC_TOKENS["NEUTRAL_INK"],
        "legend.fontsize": 7.5,
        "figure.titlesize": 10.5,
        "svg.fonttype": "none",
        "svg.hashsalt": "thermoroute-pre-figures-v1",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.75,
        "lines.linewidth": 1.2,
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


def clean_axis(ax: mpl.axes.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel_heading(ax: mpl.axes.Axes, letter: str, title: str) -> None:
    ax.text(0.0, 1.02, f"({letter})", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, fontweight="bold", color=OI["ink"])
    ax.text(0.14, 1.02, title, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, fontweight="bold", color=OI["ink"])


def box(
    ax: mpl.axes.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str = "white",
    edgecolor: str = OI["ink"],
    radius: float = 0.004,
    linewidth: float = 1.0,
    hatch: str | None = None,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.007,rounding_size={radius}",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        hatch=hatch,
        clip_on=False,
        zorder=1,
    )
    ax.add_patch(patch)
    return patch


def guarded_text(
    ax: mpl.axes.Axes,
    bounds: tuple[float, float, float, float],
    label: str,
    x: float,
    y: float,
    text_value: str,
    dimensions: str = "xy",
    **kwargs: object,
) -> mpl.text.Text:
    """Add text whose final bbox must stay at least 2 mm inside a module."""

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
    linewidth: float = 1.2,
    mutation_scale: float = 9,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
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


def add_figure_header(fig: mpl.figure.Figure, title: str, subtitle: str) -> None:
    fig.text(0.03, 0.965, title, ha="left", va="top", fontsize=10.5,
             fontweight="bold", color=OI["ink"])
    fig.text(0.03, 0.925, subtitle, ha="left", va="top", fontsize=7.7,
             color=OI["mid"])


def save_figure(fig: mpl.figure.Figure, path: Path, title: str, description: str) -> None:
    """Save deterministic SVG/PDF/PNG deliverables and SVG accessibility text."""

    validate_text_layout(fig)
    fig.savefig(path, format="svg",
                metadata={"Creator": "ThermoRoute PRE supporting-figure renderer", "Date": None},
                facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), format="pdf",
                metadata={"Creator": "ThermoRoute PRE supporting-figure renderer",
                          "CreationDate": None, "ModDate": None},
                facecolor="white")
    fig.savefig(path.with_suffix(".png"), format="png", dpi=300,
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
    fig = plt.figure(figsize=(FULL_WIDTH, 156 * MM))
    gs = fig.add_gridspec(2, 2, left=0.09, right=0.98, bottom=0.085, top=0.95,
                          height_ratios=[1.02, 0.80], hspace=0.88, wspace=0.22)

    ax = fig.add_subplot(gs[0, 0])
    clean_axis(ax)
    panel_heading(ax, "a", "Cohort ledger")
    box(ax, (0.04, 0.81), 0.92, 0.12,
        facecolor=SEMANTIC_TOKENS["TR_BLUE_LIGHT"], edgecolor=OI["blue"])
    ax.text(0.50, 0.87, "Initial candidates   n = 1,465", transform=ax.transAxes,
            fontsize=8.2, fontweight="bold", ha="center", va="center", color=OI["blue"])
    arrow_axes(ax, (0.50, 0.80), (0.50, 0.74), mutation_scale=7)

    box(ax, (0.04, 0.37), 0.92, 0.36, facecolor=SEMANTIC_TOKENS["WARNING_LIGHT"],
        edgecolor=OI["vermillion"], hatch="///")
    ax.text(0.50, 0.665, "Rejected total   n = 1,345", transform=ax.transAxes,
            fontsize=8.2, fontweight="bold", ha="center", va="center",
            color=OI["vermillion"])
    reason_lines = [
        ("×", reasons["no NWIS WTEMP+FLOW"], "no NWIS WTEMP + FLOW"),
        ("△", reasons["low full-period coverage"], "low full-period coverage"),
        ("□", reasons["low blind-test-period coverage"], "low 2019–20 coverage"),
    ]
    for index, (symbol, count, label) in enumerate(reason_lines):
        y = 0.57 - index * 0.085
        ax.text(0.10, y, symbol, transform=ax.transAxes, fontsize=8.5,
                fontweight="bold", va="center", color=OI["vermillion"])
        ax.text(0.18, y, f"{count:,}", transform=ax.transAxes, fontsize=7.7,
                fontweight="bold", va="center")
        ax.text(0.38, y, label, transform=ax.transAxes, fontsize=7.5, va="center")
    arrow_axes(ax, (0.50, 0.36), (0.50, 0.30), mutation_scale=7)

    box(ax, (0.04, 0.13), 0.92, 0.16,
        facecolor=SEMANTIC_TOKENS["ALLOWED_TEAL_LIGHT"],
        edgecolor=SEMANTIC_TOKENS["ALLOWED_TEAL"])
    ax.text(0.50, 0.21, "Retained fixed cohort\nn = 120", transform=ax.transAxes,
            fontsize=8.2, fontweight="bold", ha="center", va="center",
            color=SEMANTIC_TOKENS["ALLOWED_TEAL"], linespacing=1.02)
    ax.text(0.50, 0.055, "1,345 + 120 = 1,465", transform=ax.transAxes,
            fontsize=7.5, ha="center", va="center", color=OI["mid"])

    ax = fig.add_subplot(gs[0, 1])
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
    ax.text(0.03, 0.04, "coordinate scatter • no basemap", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7.5, color=OI["mid"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})
    ax.legend(handles=legend_handles, title="HUC2", ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.17), frameon=False, columnspacing=0.55,
              handletextpad=0.15, borderaxespad=0.0, title_fontsize=7.5)

    ax = fig.add_subplot(gs[1, 0])
    panel_heading(ax, "c", "Stations per HUC2")
    values = [counts[huc] for huc in hucs]
    xpos = list(range(len(hucs)))
    ax.bar(xpos, values, width=0.68, color=SEMANTIC_TOKENS["TR_BLUE_LIGHT"],
           edgecolor=OI["blue"],
           linewidth=0.7, zorder=2)
    ax.scatter(xpos, values, marker="D", s=14, facecolor=OI["orange"],
               edgecolor=OI["ink"], linewidth=0.6, zorder=3)
    for x, value in zip(xpos, values):
        ax.text(x, value + 0.7, str(value), va="bottom", ha="center", fontsize=7.5)
    ax.set_xticks(xpos, [f"{int(huc):02d}" for huc in hucs], rotation=45, ha="right")
    ax.set_ylim(0, 30)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_xlabel("HUC2", labelpad=1)
    ax.set_ylabel("Retained stations", labelpad=2)
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
    for index, (value, label, color, marker) in enumerate(diagnostics):
        y = 0.84 - index * 0.145
        row_bounds = (0.03, y - 0.065, 0.94, 0.13)
        ax.add_patch(Rectangle(row_bounds[:2], row_bounds[2], row_bounds[3], transform=ax.transAxes,
                               facecolor=mpl.colors.to_rgba(color, 0.09),
                               edgecolor=color, linewidth=0.75))
        ax.scatter([0.055], [y], transform=ax.transAxes, marker=marker, s=22,
                   facecolor=color, edgecolor=OI["ink"], linewidth=0.6, zorder=4)
        guarded_text(ax, (0.02, y - 0.065, 0.30, 0.13), f"S1 diagnostic value {index}",
                     0.27, y, value, dimensions="x", ha="right", va="center",
                     fontsize=8.0, fontweight="bold", color=OI["ink"])
        guarded_text(ax, (0.30, y - 0.065, 0.67, 0.13), f"S1 diagnostic label {index}",
                     0.342, y, label, dimensions="x", ha="left", va="center",
                     fontsize=7.5, color=OI["ink"])
    ax.text(0.50, 0.055, "HUC overlap / proximity\n≠ hydraulic connectivity",
            transform=ax.transAxes, ha="center", va="center", fontsize=7.5,
            fontweight="bold", color=OI["vermillion"], linespacing=0.95)

    path = HERE / "figS1_cohort_registry.svg"
    save_figure(
        fig, path, "Fig. S1 Frozen cohort selection and registry geometry",
        "Four panels reconcile the 1,465-candidate ledger, encode all 120 retained coordinates by 15 HUC2 categories without a basemap, show HUC2 counts, and report outcome-free registry proximity and repeated-HUC diagnostics.",
    )
    return path


def render_fig_s2(
    temporal: dict[str, object], bridge_report: dict[str, object], request_map: dict[str, object]
) -> Path:
    fig = plt.figure(figsize=(FULL_WIDTH, 182 * MM))
    gs = fig.add_gridspec(4, 1, left=0.065, right=0.975, bottom=0.055, top=0.96,
                          height_ratios=[0.70, 1.02, 1.12, 1.24], hspace=0.52)

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
    xs = [0.015, 0.215, 0.415, 0.615, 0.815]
    width = 0.17
    for idx, ((years, role, color, marker, hatch), x) in enumerate(zip(stages, xs)):
        box(ax, (x, 0.12), width, 0.68, facecolor=mpl.colors.to_rgba(color, 0.14),
            edgecolor=color, hatch=hatch, linewidth=1.1)
        ax.scatter([x + 0.085], [0.72], transform=ax.transAxes, marker=marker,
                   s=42 if marker != "*" else 64, facecolor=color,
                   edgecolor=OI["ink"], linewidth=0.6, zorder=4)
        ax.text(x + 0.085, 0.56, years, transform=ax.transAxes, ha="center", va="center",
                fontsize=8.4, fontweight="bold")
        role_lines = role.split("\n")
        role_positions = [0.30] if len(role_lines) == 1 else [0.37, 0.22]
        for role_line, role_y in zip(role_lines, role_positions):
            ax.text(x + 0.085, role_y, role_line, transform=ax.transAxes,
                    ha="center", va="center", fontsize=7.5, fontweight="bold",
                    color=OI["ink"])
        if idx < len(stages) - 1:
            arrow_axes(ax, (x + width + 0.006, 0.50), (xs[idx + 1] - 0.008, 0.50),
                       linewidth=1.0, mutation_scale=8)
    ax.text(0.985, 0.04, "time →", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color=OI["mid"])

    ax = fig.add_subplot(gs[1, 0])
    clean_axis(ax)
    panel_heading(ax, "b", "Issue-time boundary")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(Rectangle((0.02, 0.20), 0.44, 0.70, transform=ax.transAxes,
                           facecolor=SEMANTIC_TOKENS["ALLOWED_TEAL_LIGHT"],
                           edgecolor=SEMANTIC_TOKENS["ALLOWED_TEAL"], linewidth=1.0))
    ax.add_patch(Rectangle((0.54, 0.20), 0.44, 0.70, transform=ax.transAxes,
                           facecolor=SEMANTIC_TOKENS["WARNING_LIGHT"],
                           edgecolor=OI["vermillion"], linewidth=1.0,
                           hatch="///"))
    ax.axvline(0.5, ymin=0.22, ymax=0.94, color=OI["ink"], linewidth=1.5, zorder=5)
    ax.text(0.5, 0.95, "ISSUE  t", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.8, fontweight="bold")
    ax.text(0.24, 0.82, f"✓  ALLOWED: {temporal['source_date_display']}", transform=ax.transAxes,
            fontsize=7.8, fontweight="bold", color=SEMANTIC_TOKENS["ALLOWED_TEAL"],
            va="center", ha="center")
    allowed = [
        ("○", "Observed WTEMP history"),
        ("△", "FLOW + dated meteorology"),
        ("◇", "Frozen reference inputs"),
    ]
    for i, (symbol, label) in enumerate(allowed):
        y = 0.66 - i * 0.15
        ax.text(0.08, y, symbol, transform=ax.transAxes, fontsize=9.0,
                color=OI["blue"], va="center", fontweight="bold")
        ax.text(0.14, y, label, transform=ax.transAxes, fontsize=7.5, va="center")

    ax.text(0.76, 0.82, "✕  EXCLUDED AS INPUT", transform=ax.transAxes,
            fontsize=7.8, fontweight="bold", color=OI["vermillion"], va="center", ha="center")
    excluded = [
        ("X", f"Target WTEMP at {temporal['target_date_expression']}"),
        ("X", "Future weather"),
        ("X", "Future vintages"),
    ]
    for i, (symbol, label) in enumerate(excluded):
        y = 0.66 - i * 0.15
        ax.scatter([0.60], [y], transform=ax.transAxes, marker=symbol, s=25,
                   facecolor=OI["vermillion"], edgecolor=OI["ink"], linewidth=0.6, zorder=5)
        ax.text(0.66, y, label, transform=ax.transAxes, fontsize=7.5, va="center")
    guarded_text(ax, (0.54, 0.20, 0.44, 0.70), "S2 horizon inset", 0.76, 0.315,
                 "h = " + ", ".join(str(value) for value in temporal["horizons"]) + " d",
                 ha="center", va="center", fontsize=7.5,
                 fontweight="bold", color=OI["vermillion"])

    ax.text(0.50, 0.07, "date-indexed retrospective hindcast", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.5, fontweight="bold", color=OI["ink"])

    # Variable/provider/source-date matrix.
    ax = fig.add_subplot(gs[2, 0])
    clean_axis(ax)
    panel_heading(ax, "c", "Predictor source rules")
    matrix = [
        (variable, temporal["provider_by_variable"][variable], temporal["source_date_display"])
        for variable in ROUTE_A_VARIABLES
    ]
    headers = [(0.07, "VARIABLE"), (0.38, "PROVIDER"), (0.66, "SOURCE DATE")]
    for x, label in headers:
        ax.text(x, 0.88, label, transform=ax.transAxes, fontsize=7.5,
                fontweight="bold", ha="left", va="center", color=OI["mid"])
    ax.plot([0.04, 0.96], [0.835, 0.835], transform=ax.transAxes,
            color=OI["mid"], linewidth=0.8)
    for index, (variable, provider, source_date) in enumerate(matrix):
        y = 0.77 - index * 0.087
        if index % 2 == 0:
            ax.add_patch(Rectangle((0.04, y - 0.038), 0.92, 0.076, transform=ax.transAxes,
                                   facecolor="#F4F4F4", edgecolor="none"))
        marker = ["o", "s", "^", "D"][index % 4]
        ax.scatter([0.075], [y], transform=ax.transAxes, s=16, marker=marker,
                   facecolor=OI["sky"], edgecolor=OI["ink"], linewidth=0.6, zorder=4)
        ax.text(0.115, y, variable, transform=ax.transAxes, fontsize=7.5,
                fontweight="bold", ha="left", va="center")
        ax.text(0.38, y, provider, transform=ax.transAxes, fontsize=7.5,
                ha="left", va="center")
        ax.text(0.66, y, source_date, transform=ax.transAxes, fontsize=7.5,
                ha="left", va="center")
    ax.text(0.50, 0.08, f"vintage: {temporal['retrieval_vintage_display']}",
            transform=ax.transAxes, fontsize=7.5, fontweight="bold",
            ha="center", va="center", color=OI["blue"])

    # Equal-weight capability / limitation bridge panel.
    ax = fig.add_subplot(gs[3, 0])
    clean_axis(ax)
    panel_heading(ax, "d", "Outcome-free predictor bridge: capability and limit")
    left_bounds = (0.02, 0.12, 0.46, 0.76)
    right_bounds = (0.52, 0.12, 0.46, 0.76)
    ax.add_patch(Rectangle(left_bounds[:2], left_bounds[2], left_bounds[3],
                           transform=ax.transAxes,
                           facecolor=SEMANTIC_TOKENS["ALLOWED_TEAL_LIGHT"],
                           edgecolor=SEMANTIC_TOKENS["ALLOWED_TEAL"], linewidth=1.1))
    ax.add_patch(Rectangle(right_bounds[:2], right_bounds[2], right_bounds[3],
                           transform=ax.transAxes, facecolor=SEMANTIC_TOKENS["WARNING_LIGHT"],
                           edgecolor=OI["vermillion"], linewidth=1.1, hatch="///"))
    guarded_text(ax, left_bounds, "S2 bridge capability title", 0.25, 0.785,
                 "✓  PASS_EXACT_PRODUCT_BRIDGE", ha="center", va="center",
                 fontsize=7.5, fontweight="bold", color=SEMANTIC_TOKENS["ALLOWED_TEAL"])
    guarded_text(ax, right_bounds, "S2 bridge limitation title", 0.75, 0.785,
                 "!  LIMITS — NOT ESTABLISHED", ha="center", va="center",
                 fontsize=8.0, fontweight="bold", color=OI["vermillion"])
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
    for index, text_value in enumerate(left_lines):
        guarded_text(ax, left_bounds, f"S2 capability {index}", 0.25, 0.69 - index * 0.072,
                     text_value, ha="center", va="center", fontsize=7.5,
                     fontweight="bold" if index < 2 else "normal", color=OI["ink"])
    for index, text_value in enumerate(right_lines):
        guarded_text(ax, right_bounds, f"S2 limitation {index}", 0.75, 0.67 - index * 0.105,
                     text_value, ha="center", va="center", fontsize=7.5,
                     fontweight="bold" if index < 2 else "normal", color=OI["ink"])
    ax.text(0.50, 0.045, "bound report + 120-request source map • latest retrospective retrieval",
            transform=ax.transAxes, ha="center", va="center", fontsize=7.5,
            color=OI["mid"], fontweight="bold")

    path = HERE / "figS2_information_boundary.svg"
    save_figure(
        fig, path, "Fig. S2 Temporal roles and issue-time information boundary",
        "Four panels separate temporal roles, the issue-time predictor boundary, all seven provider/source-date rules, and the equally weighted capabilities and limitations of the outcome-free 2018–2020 predictor bridge.",
    )
    return path


def render_fig_s3(architecture: dict[str, object]) -> Path:
    fig = plt.figure(figsize=(FULL_WIDTH, 122 * MM))
    ax = fig.add_axes([0.025, 0.04, 0.95, 0.92])
    clean_axis(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for x, label in [(0.115, "(a) INPUTS"), (0.375, "(b) REPRESENTATION"),
                     (0.655, "(c) HEADS"), (0.89, "(d) CALIBRATION")]:
        ax.text(x, 0.975, label, transform=ax.transAxes, ha="center", va="center",
                fontsize=7.5, fontweight="bold", color=OI["mid"])

    quantile_labels = tuple(
        f"q{int(round(float(probability) * 100)):02d}"
        for probability in architecture["quantiles"]
    )

    # Input contract.
    input_bounds = (0.002, 0.54, 0.245, 0.35)
    box(ax, input_bounds[:2], input_bounds[2], input_bounds[3],
        facecolor=SEMANTIC_TOKENS["TR_BLUE_LIGHT"], edgecolor=OI["blue"])
    guarded_text(ax, input_bounds, "S3 input title", 0.1245, 0.845,
                 f"{len(architecture['variables'])} VARIABLES\n"
                 f"+ {'MASK' if architecture['missingness_mask'] else 'NO MASK'}",
                 ha="center", va="center", fontsize=8.0,
                 fontweight="bold", color=OI["blue"], linespacing=1.05)
    variable_lines = "WTEMP • FLOW\nTEMP • PRCP\nRHMEAN • DH\nWDSP"
    guarded_text(ax, input_bounds, "S3 variable registry", 0.1245, 0.745, variable_lines,
                 ha="center", va="center", fontsize=7.5, linespacing=1.22)
    guarded_text(ax, input_bounds, "S3 construction buffer", 0.1245, 0.605,
                 "32-day buffer\nconstruction only\n≠ effective\nmemory",
                 ha="center", va="center", fontsize=7.5, fontweight="bold",
                 color=OI["vermillion"], linespacing=0.98)

    wlevel_bounds = (0.002, 0.40, 0.245, 0.09)
    box(ax, wlevel_bounds[:2], wlevel_bounds[2], wlevel_bounds[3],
        facecolor=SEMANTIC_TOKENS["WARNING_LIGHT"], edgecolor=OI["vermillion"],
        hatch="///", radius=0.01)
    ax.scatter([0.045], [0.445], transform=ax.transAxes, marker="X", s=25,
               facecolor=OI["vermillion"], edgecolor=OI["ink"], linewidth=0.6, zorder=5)
    guarded_text(ax, wlevel_bounds, "S3 WLEVEL module", 0.13, 0.46, "WLEVEL",
                 ha="center", va="center", fontsize=7.5, fontweight="bold",
                 color=OI["vermillion"])
    guarded_text(ax, wlevel_bounds, "S3 WLEVEL module", 0.13, 0.43, "excluded",
                 ha="center", va="center", fontsize=7.5, color=OI["ink"])

    anchor_bounds = (0.002, 0.11, 0.245, 0.22)
    box(ax, anchor_bounds[:2], anchor_bounds[2], anchor_bounds[3],
        facecolor="#FFF7D1", edgecolor=OI["orange"])
    guarded_text(ax, anchor_bounds, "S3 anchor module", 0.1245, 0.292,
                 architecture["anchor_identity"],
                 ha="center", va="center", fontsize=8.0, fontweight="bold",
                 color="#986900")
    guarded_text(ax, anchor_bounds, "S3 anchor module", 0.1245, 0.238,
                 str(architecture["anchor_method"]).replace(
                     "frozen damped-persistence anchor",
                     "frozen damped-\npersistence anchor",
                 ),
                 ha="center", va="center", fontsize=7.5, linespacing=1.0)
    guarded_text(ax, anchor_bounds, "S3 anchor module", 0.1245, 0.175,
                 str(architecture["anchor_composition"]).replace(
                     "fitted blend of last y and climatology",
                     "fitted blend of\nlast y and\nclimatology",
                 ),
                 ha="center", va="center", fontsize=7.5, linespacing=1.0)

    # Parallel representation branches and their combination.
    box(ax, (0.27, 0.70), 0.21, 0.16,
        facecolor=SEMANTIC_TOKENS["ALLOWED_TEAL_LIGHT"],
        edgecolor=SEMANTIC_TOKENS["ALLOWED_TEAL"])
    ax.scatter([0.30], [0.81], transform=ax.transAxes, marker="D", s=23,
               facecolor=SEMANTIC_TOKENS["ALLOWED_TEAL"], edgecolor=OI["ink"],
               linewidth=0.6, zorder=5)
    ax.text(0.33, 0.81, "SPARSE ROUTER", transform=ax.transAxes, ha="left", va="center",
            fontsize=7.8, fontweight="bold", color=SEMANTIC_TOKENS["ALLOWED_TEAL"])
    ax.text(0.375, 0.75,
            f"{len(architecture['variables'])} variables • lags 0–{architecture['max_router_lag']}",
            transform=ax.transAxes,
            ha="center", va="center", fontsize=7.5)

    tcn_bounds = (0.26, 0.50, 0.245, 0.16)
    box(ax, tcn_bounds[:2], tcn_bounds[2], tcn_bounds[3],
        facecolor=SEMANTIC_TOKENS["TR_BLUE_LIGHT"], edgecolor=OI["blue"])
    ax.scatter([0.29], [0.61], transform=ax.transAxes, marker="^", s=25,
               facecolor=OI["blue"], edgecolor=OI["ink"], linewidth=0.6, zorder=5)
    guarded_text(ax, tcn_bounds, "S3 TCN module", 0.3825, 0.62, "LEFT-LOOKING TCN",
                 ha="center", va="center", fontsize=7.5, fontweight="bold",
                 color=OI["blue"])
    guarded_text(ax, tcn_bounds, "S3 TCN module", 0.3825, 0.575,
                 f"{architecture['tcn_blocks']} blocks • kernel {architecture['tcn_kernel']}",
                 ha="center", va="center", fontsize=7.5)
    guarded_text(ax, tcn_bounds, "S3 TCN module", 0.3825, 0.535,
                 f"receptive field {architecture['tcn_receptive_field']}",
                 ha="center", va="center", fontsize=7.5)

    box(ax, (0.27, 0.31), 0.21, 0.13, facecolor="#F9EAF3", edgecolor=OI["purple"])
    ax.scatter([0.30], [0.39], transform=ax.transAxes, marker="s", s=21,
               facecolor=OI["purple"], edgecolor=OI["ink"], linewidth=0.6, zorder=5)
    ax.text(0.375, 0.405, "MIXTURE OF", transform=ax.transAxes, ha="center",
            va="center", fontsize=7.5, fontweight="bold", color=OI["purple"])
    ax.text(0.375, 0.37, "EXPERTS", transform=ax.transAxes, ha="center",
            va="center", fontsize=7.5, fontweight="bold", color=OI["purple"])
    ax.text(0.375, 0.335, "combined\nrepresentation", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.5, linespacing=1.0)

    box(ax, (0.27, 0.14), 0.21, 0.12,
        facecolor=SEMANTIC_TOKENS["ALLOWED_TEAL_LIGHT"],
        edgecolor=SEMANTIC_TOKENS["ALLOWED_TEAL"])
    ax.text(0.375, 0.22, "PROPOSAL P", transform=ax.transAxes, ha="center", va="center",
            fontsize=7.8, fontweight="bold", color=SEMANTIC_TOKENS["ALLOWED_TEAL"])
    ax.text(0.375, 0.17, "learned relaxation κ", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.5)

    arrow_axes(ax, (0.21, 0.77), (0.27, 0.78),
               color=SEMANTIC_TOKENS["ALLOWED_TEAL"], linewidth=1.0)
    arrow_axes(ax, (0.21, 0.68), (0.27, 0.58), color=OI["blue"], linewidth=1.0)
    arrow_axes(ax, (0.375, 0.70), (0.375, 0.45),
               color=SEMANTIC_TOKENS["ALLOWED_TEAL"], linewidth=1.0)
    arrow_axes(ax, (0.43, 0.50), (0.43, 0.45), color=OI["blue"], linewidth=1.0)
    arrow_axes(ax, (0.21, 0.60), (0.27, 0.20),
               color=SEMANTIC_TOKENS["ALLOWED_TEAL"], linewidth=1.0)

    # Three explicitly separate heads.
    head_specs = [
        (0.62, OI["blue"], SEMANTIC_TOKENS["TR_BLUE_LIGHT"],
         "o", "POINT HEAD", "MSE residual r"),
        (0.16, SEMANTIC_TOKENS["ALLOWED_TEAL"],
         SEMANTIC_TOKENS["ALLOWED_TEAL_LIGHT"], "^", "EVENT HEAD", "event score"),
    ]
    for y, color, face, marker, title, detail in head_specs:
        box(ax, (0.56, y), 0.19, 0.14, facecolor=face, edgecolor=color)
        ax.scatter([0.59], [y + 0.095], transform=ax.transAxes, marker=marker, s=22,
                   facecolor=color, edgecolor=OI["ink"], linewidth=0.6, zorder=5)
        ax.text(0.62, y + 0.095, title, transform=ax.transAxes, ha="left", va="center",
                fontsize=7.5, fontweight="bold", color=color)
        ax.text(0.655, y + 0.045, detail, transform=ax.transAxes, ha="center", va="center",
                fontsize=7.5)
    quantile_bounds = (0.545, 0.35, 0.225, 0.22)
    box(ax, quantile_bounds[:2], quantile_bounds[2], quantile_bounds[3],
        facecolor="#F9EAF3", edgecolor=OI["purple"])
    ax.scatter([0.575], [0.53], transform=ax.transAxes, marker="D", s=22,
               facecolor=OI["purple"], edgecolor=OI["ink"], linewidth=0.6, zorder=5)
    ax.text(0.605, 0.53, "Q HEADS", transform=ax.transAxes, ha="left", va="center",
            fontsize=7.5, fontweight="bold", color=OI["purple"])
    guarded_text(ax, quantile_bounds, "S3 quantile module", 0.6575, 0.485,
                 f"{quantile_labels[1]}: separate", ha="center", va="center", fontsize=7.5)
    guarded_text(ax, quantile_bounds, "S3 quantile module", 0.6575, 0.44,
                 "anchor-bounded", ha="center", va="center", fontsize=7.5)
    guarded_text(ax, quantile_bounds, "S3 quantile module", 0.6575, 0.395,
                 f"{quantile_labels[0]}/{quantile_labels[2]}: ± widths",
                 ha="center", va="center", fontsize=7.5)
    arrow_axes(ax, (0.48, 0.39), (0.56, 0.69), color=OI["blue"], linewidth=1.0)
    arrow_axes(ax, (0.48, 0.38), (0.56, 0.47), color=OI["purple"], linewidth=1.0)
    arrow_axes(ax, (0.48, 0.37), (0.56, 0.26),
               color=SEMANTIC_TOKENS["ALLOWED_TEAL"], linewidth=1.0)

    # Wide, plain-text point identity avoids cramped mathtext and makes the
    # anchor/proposal/residual merge explicit.
    formula_bounds = (0.54, 0.77, 0.44, 0.17)
    box(ax, formula_bounds[:2], formula_bounds[2], formula_bounds[3],
        facecolor="white", edgecolor=OI["blue"],
        linewidth=1.0)
    guarded_text(ax, formula_bounds, "S3 anchor-bound formula", 0.76, 0.91,
                 "ANCHOR-BOUND POINT", ha="center", va="center", fontsize=7.8,
                 fontweight="bold", color=OI["blue"])
    guarded_text(ax, formula_bounds, "S3 anchor-bound formula", 0.76, 0.88,
                 "ŷ = A + δ tanh(z / δ)", ha="center", va="center", fontsize=7.5)
    guarded_text(ax, formula_bounds, "S3 anchor-bound formula", 0.76, 0.85,
                 f"z = P − A + r  •  δ = {architecture['delta_scale']:.1f} °C",
                 ha="center", va="center", fontsize=7.5)
    guarded_text(ax, formula_bounds, "S3 non-safety bound", 0.76, 0.81,
                 "anchor deviation ≠ truth error\n≠ safety bound",
                 ha="center", va="center", fontsize=7.5, fontweight="bold",
                 color=OI["vermillion"], linespacing=0.95)
    arrow_axes(ax, (0.655, 0.76), (0.655, 0.77), color=OI["blue"], linewidth=1.0)

    # Anchor and learned proposal lanes merge into the point identity. Curved
    # connectors sit behind modules so no line crosses text.
    for start, color, rad in [((0.21, 0.20), OI["orange"], -0.25),
                              ((0.48, 0.20), SEMANTIC_TOKENS["ALLOWED_TEAL"], -0.15)]:
        ax.add_patch(FancyArrowPatch(start, (0.58, 0.77), transform=ax.transAxes,
                                     arrowstyle="-|>", mutation_scale=8,
                                     connectionstyle=f"arc3,rad={rad}", linewidth=1.0,
                                     color=color, clip_on=False, zorder=0.5))

    box(ax, (0.80, 0.40), 0.18, 0.17, facecolor="#F9EAF3", edgecolor=OI["purple"])
    ax.text(0.89, 0.525, "CQR", transform=ax.transAxes, ha="center", va="center",
            fontsize=8.0, fontweight="bold", color=OI["purple"])
    ax.text(0.89, 0.48, "member average", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.5)
    ax.text(0.89, 0.435, f"fit {architecture['calibration_display']} only", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.5)
    arrow_axes(ax, (0.77, 0.47), (0.80, 0.47), color=OI["purple"], linewidth=1.0)
    interval_bounds = (0.80, 0.30, 0.18, 0.085)
    ax.add_patch(Rectangle(interval_bounds[:2], interval_bounds[2], interval_bounds[3],
                           transform=ax.transAxes, facecolor="white",
                           edgecolor=OI["purple"], linewidth=0.9))
    guarded_text(ax, interval_bounds, "S3 final interval", 0.89, 0.343,
                 "CALIBRATED\nINTERVAL", ha="center", va="center", fontsize=7.5,
                 fontweight="bold", color=OI["purple"], linespacing=0.95)
    arrow_axes(ax, (0.89, 0.40), (0.89, 0.375), color=OI["purple"], linewidth=1.0)

    box(ax, (0.80, 0.17), 0.18, 0.12,
        facecolor=SEMANTIC_TOKENS["ALLOWED_TEAL_LIGHT"],
        edgecolor=SEMANTIC_TOKENS["ALLOWED_TEAL"])
    ax.text(0.89, 0.255, "PLATT", transform=ax.transAxes, ha="center", va="center",
            fontsize=8.0, fontweight="bold", color=SEMANTIC_TOKENS["ALLOWED_TEAL"])
    ax.text(0.89, 0.205, f"fit {architecture['calibration_display']} only", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.5)
    arrow_axes(ax, (0.75, 0.225), (0.80, 0.225),
               color=SEMANTIC_TOKENS["ALLOWED_TEAL"], linewidth=1.0)
    probability_bounds = (0.80, 0.07, 0.18, 0.085)
    ax.add_patch(Rectangle(probability_bounds[:2], probability_bounds[2], probability_bounds[3],
                           transform=ax.transAxes, facecolor="white",
                           edgecolor=SEMANTIC_TOKENS["ALLOWED_TEAL"], linewidth=0.9))
    guarded_text(ax, probability_bounds, "S3 final probability", 0.89, 0.113,
                 "CALIBRATED\nPROBABILITY", ha="center", va="center", fontsize=7.5,
                 fontweight="bold", color=SEMANTIC_TOKENS["ALLOWED_TEAL"], linespacing=0.95)
    arrow_axes(ax, (0.89, 0.17), (0.89, 0.145),
               color=SEMANTIC_TOKENS["ALLOWED_TEAL"], linewidth=1.0)

    box(ax, (0.02, 0.008), 0.72, 0.055, facecolor="#F3F3F3", edgecolor=OI["mid"],
        radius=0.008, linewidth=0.7)
    ax.text(0.38, 0.035, "κ / router: internal allocations ≠ physical routing",
            transform=ax.transAxes, ha="center", va="center", fontsize=7.5,
            fontweight="bold", color=OI["ink"])

    path = HERE / "figS3_model_architecture.svg"
    save_figure(
        fig, path, "Fig. S3 ThermoRoute PRE model and calibration architecture",
        "A four-column schematic shows the strictly checked seven-variable input contract, WLEVEL exclusion, construction-buffer/router/TCN distinctions, learned proposal and mixture, separate point/quantile/event heads, anchor-only numerical bound, and final CQR interval and Platt probability outputs.",
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
            "path": HERE / "figS1_cohort_registry.svg", "height_mm": 156,
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
            "path": HERE / "figS2_information_boundary.svg", "height_mm": 182,
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
            "path": HERE / "figS3_model_architecture.svg", "height_mm": 122,
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
        artifacts["PNG"]["dpi"] = 300
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
                "placed_width_mm": 140,
                "placed_height_mm": spec["height_mm"],
                "background": "white",
                "minimum_nominal_text_pt": min(font_sizes),
                "minimum_module_text_inset_mm": 2.0,
                "semantic_palette_binding":
                    "paper/FIGURE_REDRAW_SPEC.md#3.1-Semantic-palette",
                "svg_accessibility": "role=img; aria-labelledby=svg-title svg-desc",
                "pdf_fonttype": 42,
                "png_dpi": 300,
            },
            "artifacts": artifacts,
            "width_mm": 140,
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

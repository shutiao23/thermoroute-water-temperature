"""Frozen panel and confirmatory-holdout evidence contracts.

This module distinguishes three concepts that were previously conflated:

* a legacy panel's internal alias (``n00`` ...),
* the stable USGS ``site_no`` scientific primary key, and
* a genuinely untouched confirmatory evaluation that has not yet been opened.

The 2019--2020 results in this repository are development/exploratory evidence.
They are intentionally not relabelled as untouched by this code.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .provenance import canonical_json_bytes, sha256_file


class EvidenceError(RuntimeError):
    """Raised when a frozen evidence contract is incomplete or violated."""


DEFAULT_FROZEN_PANEL_SPEC = (
    Path(__file__).resolve().parents[2] / "data_usgs" / "frozen_panel_v1.json"
)


@dataclass(frozen=True)
class FrozenPanelSpec:
    spec_path: Path
    document: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path = DEFAULT_FROZEN_PANEL_SPEC) -> "FrozenPanelSpec":
        spec_path = Path(path).resolve()
        try:
            document = json.loads(spec_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise EvidenceError(f"cannot read frozen panel spec: {spec_path}") from exc
        if document.get("schema_version") != 1:
            raise EvidenceError("unsupported frozen panel spec schema")
        return cls(spec_path=spec_path, document=document)

    @property
    def root(self) -> Path:
        return self.spec_path.parent

    @property
    def panel_path(self) -> Path:
        return (self.root / str(self.document["panel"]["path"])).resolve()

    @property
    def registry_path(self) -> Path:
        return (self.root / str(self.document["station_registry"]["path"])).resolve()

    @property
    def source_metadata_path(self) -> Path:
        return (
            self.root
            / str(self.document["station_registry"]["source_metadata_path"])
        ).resolve()

    @property
    def spec_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.document)).hexdigest()

    def _verify_artifact(self, artifact: Mapping[str, Any], path: Path) -> None:
        if not path.is_file():
            raise EvidenceError(f"frozen artifact is missing: {path}")
        expected = str(artifact["sha256"])
        actual = sha256_file(path)
        if actual != expected:
            raise EvidenceError(
                f"frozen artifact checksum mismatch for {path.name}: "
                f"expected {expected}, got {actual}")

    def load_registry(self) -> pd.DataFrame:
        artifact = self.document["station_registry"]
        self._verify_artifact(artifact, self.registry_path)
        self._verify_artifact(
            {"sha256": artifact["source_metadata_sha256"]},
            self.source_metadata_path,
        )
        huc = artifact.get("huc_metadata", {})
        if huc.get("runtime_dependency"):
            huc_source = (self.root / str(huc["source_path"])).resolve()
            self._verify_artifact({"sha256": huc["source_sha256"]}, huc_source)
            if huc.get("provenance_path"):
                huc_provenance = (self.root / str(huc["provenance_path"])).resolve()
                self._verify_artifact(
                    {"sha256": huc["provenance_sha256"]}, huc_provenance)
        registry = pd.read_csv(
            self.registry_path,
            dtype={"site_no": "string", "legacy_site_id": "string"},
            keep_default_na=False,
        )
        required = {"site_no", "legacy_site_id", "station_nm", "lat", "lon", "state"}
        missing = required - set(registry.columns)
        if missing:
            raise EvidenceError(f"station registry missing columns: {sorted(missing)}")
        for key in ("site_no", "legacy_site_id"):
            registry[key] = registry[key].astype("string").str.strip()
            if registry[key].eq("").any() or registry[key].duplicated().any():
                raise EvidenceError(f"station registry key {key!r} is empty or non-unique")
        expected_n = int(artifact["station_count"])
        if len(registry) != expected_n:
            raise EvidenceError(
                f"station registry has {len(registry)} rows, expected {expected_n}")
        if huc.get("status") in {
            "PARTIAL_LEGACY_SOURCE", "COMPLETE_USGS_RAW_SNAPSHOT"
        }:
            if "huc_metadata_status" not in registry:
                raise EvidenceError("partial HUC contract lacks per-station status")
            counts = registry["huc_metadata_status"].value_counts().to_dict()
            joined_status = huc["joined_status"]
            if counts.get(joined_status, 0) != huc["site_no_joined_count"]:
                raise EvidenceError("HUC joined count differs from frozen contract")
            if counts.get("UNVERIFIED_MISSING", 0) != huc["missing_count"]:
                raise EvidenceError("HUC missing count differs from frozen contract")
        return registry

    def load_panel(self, *, stable_site_ids: bool = True) -> pd.DataFrame:
        artifact = self.document["panel"]
        self._verify_artifact(artifact, self.panel_path)
        panel = pd.read_parquet(self.panel_path)
        required = set(artifact["required_columns"])
        missing = required - set(panel.columns)
        if missing:
            raise EvidenceError(f"panel missing columns: {sorted(missing)}")
        panel["DATE"] = pd.to_datetime(panel["DATE"])
        expected_rows = int(artifact["row_count"])
        expected_sites = int(artifact["station_count"])
        if len(panel) != expected_rows or panel["site_id"].nunique() != expected_sites:
            raise EvidenceError(
                f"panel dimensions changed: rows={len(panel)}, "
                f"stations={panel['site_id'].nunique()}")
        if panel["site_id"].isna().any():
            raise EvidenceError("panel contains missing legacy station identifiers")
        if panel.duplicated(["site_id", "DATE"]).any():
            raise EvidenceError("panel contains duplicate (site_id, DATE) rows")
        if str(panel["DATE"].min().date()) != artifact["date_start"]:
            raise EvidenceError("panel start date does not match frozen spec")
        if str(panel["DATE"].max().date()) != artifact["date_end"]:
            raise EvidenceError("panel end date does not match frozen spec")

        registry = self.load_registry()
        legacy = set(panel["site_id"].astype(str).unique())
        registered = set(registry["legacy_site_id"].astype(str))
        if legacy != registered:
            raise EvidenceError(
                "panel/registry legacy-key mismatch: "
                f"panel_only={sorted(legacy - registered)[:5]}, "
                f"registry_only={sorted(registered - legacy)[:5]}")
        if stable_site_ids:
            mapping = dict(zip(registry["legacy_site_id"], registry["site_no"]))
            panel = panel.copy()
            panel.insert(2, "legacy_site_id", panel["site_id"].astype("string"))
            panel["site_id"] = panel["legacy_site_id"].map(mapping).astype("string")
            if panel["site_id"].isna().any() or panel["site_id"].nunique() != expected_sites:
                raise EvidenceError("failed to map every panel station to stable USGS site_no")
        return panel

    def verify(self) -> dict[str, object]:
        registry = self.load_registry()
        panel = self.load_panel(stable_site_ids=True)
        role = str(self.document.get("evidence_role", ""))
        if role != "development_exploratory":
            raise EvidenceError(
                "legacy 2006--2020 panel must remain development_exploratory")
        return {
            "panel_id": self.document["panel_id"],
            "panel_sha256": self.document["panel"]["sha256"],
            "registry_sha256": self.document["station_registry"]["sha256"],
            "row_count": len(panel),
            "station_count": len(registry),
            "site_primary_key": "site_no",
            "evidence_role": role,
        }


FORBIDDEN_CONFIRMATORY_COLUMNS = (
    "wtemp", "water_temperature", "target", "label", "outcome",
    "event", "exceed", "test_coverage", "holdout_coverage",
)


def select_confirmatory_sites(
    candidates: pd.DataFrame,
    development_site_nos: set[str],
    *,
    n_sites: int,
    selection_seed: str,
) -> pd.DataFrame:
    """Select new sites from metadata only, without looking at holdout outcomes.

    The candidate registry itself must be frozen from a discovery-only response
    before holdout labels are requested.  Outcome/coverage fields are rejected,
    and development stations are excluded before deterministic hash ranking.
    """
    lowered = {str(c).strip().lower() for c in candidates.columns}
    forbidden = sorted(
        c for c in lowered
        if any(token in c for token in FORBIDDEN_CONFIRMATORY_COLUMNS)
    )
    if forbidden:
        raise EvidenceError(
            "confirmatory site selection cannot inspect outcome/coverage columns: "
            f"{forbidden}")
    if "site_no" not in candidates.columns:
        raise EvidenceError("confirmatory candidate registry requires site_no")
    out = candidates.copy()
    out["site_no"] = out["site_no"].astype("string").str.strip()
    if out["site_no"].eq("").any() or out["site_no"].duplicated().any():
        raise EvidenceError("confirmatory candidate site_no must be non-empty and unique")
    out = out[~out["site_no"].isin({str(x) for x in development_site_nos})].copy()
    if len(out) < n_sites:
        raise EvidenceError(
            f"only {len(out)} new candidate sites remain; {n_sites} requested")
    out["selection_rank_sha256"] = out["site_no"].map(
        lambda s: hashlib.sha256(f"{selection_seed}:{s}".encode()).hexdigest())
    out = out.sort_values(["selection_rank_sha256", "site_no"]).head(n_sites)
    return out.reset_index(drop=True)


def load_confirmatory_protocol(path: str | Path) -> Mapping[str, Any]:
    try:
        protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read confirmatory protocol: {path}") from exc
    if protocol.get("schema_version") != 1:
        raise EvidenceError("unsupported confirmatory protocol schema")
    # Crucial honesty guard: the checked-in protocol is a plan until a candidate
    # registry is frozen and labels are independently sealed/acquired.
    if protocol.get("status") not in {
        "PLANNED_NOT_ACQUIRED", "FROZEN_NOT_ACQUIRED",
        "REGISTRY_FROZEN_LABELS_SEALED", "OPENED_ONCE"
    }:
        raise EvidenceError("invalid confirmatory protocol status")
    return protocol

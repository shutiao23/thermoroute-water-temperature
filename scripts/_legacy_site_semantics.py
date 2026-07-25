"""Shared guards for the withdrawn legacy three-site interpretation.

The claim ledger owns the regex bytes.  This dependency-free module compiles
them once, masks only exact canonical correction sentences, then scans whole
paragraphs after minimal Markdown/TeX normalization.  It also retires misleading
generated filenames. These finite English/Chinese publication checks are defense
in depth;
frozen document hashes and the structured semantic binding are authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import html
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping


LINT_PREFIX = "LINT_LEGACY_THREE_SITE_"
ALLOW_PREFIX = "ALLOW_LEGACY_THREE_SITE_"
REQUIRED_LINT_IDS = (
    "LINT_LEGACY_THREE_SITE_CASCADE_DESCRIPTOR",
    "LINT_LEGACY_THREE_SITE_CHINESE_OVERCLAIM",
    "LINT_LEGACY_THREE_SITE_DIRECTED_ARROW",
    "LINT_LEGACY_THREE_SITE_FLOW_PATH_SYNONYM",
    "LINT_LEGACY_THREE_SITE_FLOW_KAPPA_CAUSAL",
    "LINT_LEGACY_THREE_SITE_DIRECT_SEMANTIC_SYNONYM",
    "LINT_LEGACY_THREE_SITE_HYDRAULIC_RELATION",
    "LINT_LEGACY_THREE_SITE_KAPPA_PHYSICAL_TIME",
    "LINT_LEGACY_THREE_SITE_LATENT_PHYSICAL_SYNONYM",
    "LINT_LEGACY_THREE_SITE_MULTI_ROLE",
    "LINT_LEGACY_THREE_SITE_ORDERING",
    "LINT_LEGACY_THREE_SITE_RESERVOIR_ROLE_SYNONYM",
    "LINT_LEGACY_THREE_SITE_ROUTER_CAUSAL_DRIVER",
    "LINT_LEGACY_THREE_SITE_RELAXATION_PHYSICAL_MODEL",
    "LINT_LEGACY_THREE_SITE_ROLE_SHORTHAND",
    "LINT_LEGACY_THREE_SITE_SINGLE_ROLE",
    "LINT_LEGACY_THREE_SITE_TOPOLOGY_SYNONYM",
    "LINT_LEGACY_THREE_SITE_CHINESE_ORDER_PATH",
)
REQUIRED_ALLOW_IDS = (
    "ALLOW_LEGACY_THREE_SITE_EXPLICIT_DENIAL_PREFIX",
    "ALLOW_LEGACY_THREE_SITE_EPISTEMIC_TOPOLOGY_LIMIT",
    "ALLOW_LEGACY_THREE_SITE_HISTORICAL_CORRECTION",
    "ALLOW_LEGACY_THREE_SITE_ORDINARY_MONITORING_CORRECTION",
    "ALLOW_LEGACY_THREE_SITE_STRUCTURED_NEGATED_PREDICATE",
)
LEGACY_REPORT_FILENAME = "mechanism_summary.md"
LEGACY_DATA_AUDIT_FILENAME = "data_audit.md"
LEGACY_FIGURE_BASENAMES = (
    "fig1_study_area",
    "fig5_blindtest_trajectory",
    "fig7_lag_importance",
    "fig8_dynamic_kappa",
    "fig9_loso",
    "fig10_flow_lagmaps",
)
ORDINARY_MONITORING_SENTENCE = (
    "b1, s2, and p3 are ordinary monitoring stations, not reservoirs."
)
EPISTEMIC_TOPOLOGY_SENTENCE = (
    "No verified metadata establish any upstream/downstream ordering, hydraulic "
    "connectivity, regulation status, or travel time among b1, s2, and p3."
)
LEGACY_REPORT_TOMBSTONE = (
    "# Withdrawn legacy filename\n\n"
    f"{ORDINARY_MONITORING_SENTENCE} {EPISTEMIC_TOPOLOGY_SENTENCE}\n\n"
    "This path is retained only as a tombstone so stale physical-mechanism claims "
    "cannot survive a failed or partial rerun. The current single-seed descriptive "
    "artifact, when completed, is `latent_component_diagnostics.md`; neither "
    "artifact identifies physical or causal mechanisms.\n"
)
LEGACY_DATA_AUDIT_TOMBSTONE = (
    "# Withdrawn legacy data-audit bytes\n\n"
    f"{ORDINARY_MONITORING_SENTENCE} {EPISTEMIC_TOPOLOGY_SENTENCE}\n\n"
    "A current data-QC report has not completed. This tombstone replaces older "
    "topology and travel-time text before any data operation, so a failed or "
    "partial rerun cannot leave that withdrawn interpretation active.\n"
)

_PARAGRAPH = re.compile(r"\S(?:.*?)(?=\n\s*\n|\Z)", flags=re.DOTALL)
_SENTENCE = re.compile(r".+?(?:[.!?](?=\s|\Z)|\Z)", flags=re.DOTALL)
_ABBREVIATION = re.compile(
    r"\b(?:e\.g|i\.e|fig|eq|sec|u\.s|u\.k|vs|dr|prof)\.",
    flags=re.IGNORECASE,
)
_TEX_TEXT_WRAPPER = re.compile(
    r"\\(?:text|texttt|textbf|textit|textrm|textsf|textnormal|emph|underline|"
    r"mbox|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}",
    flags=re.IGNORECASE,
)
_TEX_HREF_WRAPPER = re.compile(
    r"\\href\s*\{[^{}]*\}\s*\{([^{}]*)\}",
    flags=re.IGNORECASE,
)
_TEX_HYPERREF_WRAPPER = re.compile(
    r"\\hyperref\s*\[[^\]]*\]\s*\{([^{}]*)\}",
    flags=re.IGNORECASE,
)
_MARKDOWN_LINK = re.compile(r"!?\[([^\]]+)\]\([^)]*\)", flags=re.DOTALL)
_HTML_COMMENT = re.compile(r"<!--.*?-->", flags=re.DOTALL)
_HTML_TAG = re.compile(r"<[^>]+>", flags=re.DOTALL)
_HTML_SEMANTIC_ATTRIBUTE = re.compile(
    r'''(?<![\w:-])(?:alt|title|aria-label)\s*=\s*'''
    r'''(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))''',
    flags=re.IGNORECASE,
)
_TEX_DIRECTED_ARROW = re.compile(
    r"\\(?:to|longto|rightarrow|longrightarrow|Rightarrow|"
    r"xrightarrow(?:\s*\{[^{}]{0,40}\})?)",
    flags=re.IGNORECASE,
)
_MARKUP_TRANSLATION = str.maketrans("", "", "\x60*_{}$")


class LegacySemanticGuardError(ValueError):
    """The three-site semantic contract is missing, malformed, or unsafe."""


def _preserve_html_semantic_attributes(match: re.Match[str]) -> str:
    """Keep user-visible accessibility/tooltip text while removing HTML tags."""
    values: list[str] = []
    for attribute in _HTML_SEMANTIC_ATTRIBUTE.finditer(match.group()):
        value = next(
            (candidate for candidate in attribute.groups() if candidate is not None),
            "",
        )
        if value:
            values.append(value)
    return f" {' '.join(values)} " if values else ""


@dataclass(frozen=True)
class LegacySemanticViolation:
    """One positive legacy interpretation outside a correction context."""

    lint_id: str
    start: int
    excerpt: str


@dataclass(frozen=True)
class LegacySemanticPolicy:
    """Compiled forbidden interpretations plus exact correction contexts."""

    forbidden: Mapping[str, re.Pattern[str]]
    allowed: Mapping[str, re.Pattern[str]]


_CANONICAL_SENTENCES = {
    "ALLOW_LEGACY_THREE_SITE_ORDINARY_MONITORING_CORRECTION": (
        ORDINARY_MONITORING_SENTENCE
    ),
    "ALLOW_LEGACY_THREE_SITE_EPISTEMIC_TOPOLOGY_LIMIT": (
        EPISTEMIC_TOPOLOGY_SENTENCE
    ),
}
_SEMANTIC_BINDING_EXACT = {
    "format": "thermoroute.legacy-three-site-semantics-binding.v1",
    "constraint_id": "P09_LEGACY_THREE_SITE_NO_UNVERIFIED_TOPOLOGY",
    "notice_path": "protocols/legacy_three_site_semantics_notice_v1.md",
    "site_ids": ["b1", "s2", "p3"],
    "site_classification": "ORDINARY_MONITORING_STATIONS_NOT_RESERVOIRS",
    "verified_network_metadata": False,
    "topology_inference_allowed": False,
    "hydraulic_connectivity_inference_allowed": False,
    "regulation_inference_allowed": False,
    "travel_time_inference_allowed": False,
    "route_a_evidence": False,
    "source_provenance_status": "UNVERIFIED",
    "measurement_dictionary_status": "UNVERIFIED",
    "public_redistribution_authorization_status": "UNVERIFIED_BLOCKED",
}


def _validate_semantic_binding(registry: Mapping[str, object]) -> Mapping[str, object]:
    binding = registry.get("legacy_three_site_semantics_binding")
    if not isinstance(binding, Mapping):
        raise LegacySemanticGuardError(
            "claim ledger legacy three-site semantic binding is absent"
        )
    expected_keys = set(_SEMANTIC_BINDING_EXACT) | {"notice_sha256"}
    if set(binding) != expected_keys or any(
        binding.get(key) != value for key, value in _SEMANTIC_BINDING_EXACT.items()
    ):
        raise LegacySemanticGuardError(
            "claim ledger legacy three-site semantic binding changed"
        )
    digest = binding.get("notice_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise LegacySemanticGuardError(
            "claim ledger legacy three-site notice digest is malformed"
        )
    return binding


def _compile_named_regex_entries(
    entries: object,
    *,
    id_key: str,
    regex_key: str,
    prefix: str,
    required_ids: tuple[str, ...],
    label: str,
) -> Mapping[str, re.Pattern[str]]:
    if not isinstance(entries, list):
        raise LegacySemanticGuardError(f"claim ledger {label} is not a list")
    selected = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and isinstance(entry.get(id_key), str)
        and entry[id_key].startswith(prefix)
    ]
    identifiers = [str(entry[id_key]) for entry in selected]
    if len(identifiers) != len(set(identifiers)) or set(identifiers) != set(
        required_ids
    ):
        raise LegacySemanticGuardError(
            f"claim ledger {label} is incomplete or duplicated"
        )
    compiled: dict[str, re.Pattern[str]] = {}
    for entry in selected:
        if set(entry) != {id_key, regex_key} or not isinstance(
            entry.get(regex_key), str
        ):
            raise LegacySemanticGuardError(f"claim ledger {label} entry is malformed")
        identifier = str(entry[id_key])
        try:
            compiled[identifier] = re.compile(
                str(entry[regex_key]), flags=re.IGNORECASE | re.DOTALL
            )
        except re.error as exc:
            raise LegacySemanticGuardError(
                f"claim ledger {label} regex {identifier} is invalid"
            ) from exc
    return {identifier: compiled[identifier] for identifier in required_ids}


def compile_legacy_semantic_policy(
    registry: Mapping[str, object],
) -> LegacySemanticPolicy:
    """Compile the exact three-site policy from an already-loaded ledger."""
    _validate_semantic_binding(registry)
    forbidden = _compile_named_regex_entries(
        registry.get("free_text_lints"),
        id_key="lint_id",
        regex_key="regex",
        prefix=LINT_PREFIX,
        required_ids=REQUIRED_LINT_IDS,
        label="legacy three-site lint family",
    )
    allowed = _compile_named_regex_entries(
        registry.get("legacy_semantics_sentence_allowlist"),
        id_key="allow_id",
        regex_key="regex",
        prefix=ALLOW_PREFIX,
        required_ids=REQUIRED_ALLOW_IDS,
        label="legacy three-site sentence allowlist",
    )
    for allow_id, sentence in _CANONICAL_SENTENCES.items():
        if allowed[allow_id].fullmatch(sentence) is None:
            raise LegacySemanticGuardError(
                f"claim ledger {allow_id} does not admit its canonical sentence"
            )
    return LegacySemanticPolicy(forbidden=forbidden, allowed=allowed)


def load_legacy_semantic_policy(root: Path) -> LegacySemanticPolicy:
    """Load and compile the three-site policy from the repository ledger."""
    registry_path = root / "protocols/route_a_claim_registry_v1.json"
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LegacySemanticGuardError(
            f"cannot load legacy semantic lints from {registry_path}"
        ) from exc
    if not isinstance(registry, Mapping):
        raise LegacySemanticGuardError("claim ledger is not a JSON object")
    policy = compile_legacy_semantic_policy(registry)
    binding = _validate_semantic_binding(registry)
    notice_path = root / str(binding["notice_path"])
    try:
        actual = hashlib.sha256(notice_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise LegacySemanticGuardError(
            f"cannot read legacy semantic notice {notice_path}"
        ) from exc
    if actual != binding["notice_sha256"]:
        raise LegacySemanticGuardError(
            "legacy semantic notice differs from its structured binding"
        )
    return policy


def find_legacy_semantic_violations(
    text: str,
    policy: LegacySemanticPolicy,
) -> tuple[LegacySemanticViolation, ...]:
    """Find forbidden interpretations in abbreviation-aware prose sentences.

    Known scholarly abbreviations are protected before sentence splitting, and
    only an exact allowlisted sentence is skipped.  The returned offset is
    conservative after markup normalization and is intended for diagnostics,
    not byte slicing.
    """
    violations: list[LegacySemanticViolation] = []
    for paragraph_match in _PARAGRAPH.finditer(text):
        paragraph = _HTML_COMMENT.sub("", paragraph_match.group())
        paragraph = html.unescape(paragraph)
        paragraph = _HTML_TAG.sub(_preserve_html_semantic_attributes, paragraph)
        for _ in range(16):
            unwrapped = _TEX_HREF_WRAPPER.sub(r"\1", paragraph)
            unwrapped = _TEX_HYPERREF_WRAPPER.sub(r"\1", unwrapped)
            unwrapped = _TEX_TEXT_WRAPPER.sub(r"\1", unwrapped)
            unwrapped = _MARKDOWN_LINK.sub(r"\1", unwrapped)
            if unwrapped == paragraph:
                break
            paragraph = unwrapped
        paragraph = _TEX_DIRECTED_ARROW.sub(" → ", paragraph)
        paragraph = paragraph.translate(_MARKUP_TRANSLATION)
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        protected = _ABBREVIATION.sub(
            lambda match: match.group().replace(".", "\u2024"),
            paragraph,
        )
        for sentence_match in _SENTENCE.finditer(protected):
            protected_sentence = sentence_match.group().strip()
            sentence = protected_sentence.replace("\u2024", ".")
            if not sentence or any(
                pattern.fullmatch(sentence) is not None
                for pattern in policy.allowed.values()
            ):
                continue
            for lint_id, pattern in policy.forbidden.items():
                match = pattern.search(protected_sentence)
                if match is not None:
                    excerpt_start = max(0, match.start() - 40)
                    violations.append(
                        LegacySemanticViolation(
                            lint_id=lint_id,
                            start=(
                                paragraph_match.start()
                                + sentence_match.start()
                                + match.start()
                            ),
                            excerpt=sentence[
                                excerpt_start : match.end() + 80
                            ][:240],
                        )
                    )
    return tuple(violations)


def write_atomic_text(destination: Path, payload: str) -> Path:
    """Atomically replace one UTF-8 text artifact and fsync its directory."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        os.chmod(destination, 0o644)
        directory_fd = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def retire_legacy_report_output(report_dir: Path) -> Path:
    """Atomically replace the misleading old report name before model work."""
    return write_atomic_text(
        report_dir / LEGACY_REPORT_FILENAME,
        LEGACY_REPORT_TOMBSTONE,
    )


def retire_legacy_data_audit_output(report_dir: Path) -> Path:
    """Retire old topology text before any Stage-01 data operation."""
    return write_atomic_text(
        report_dir / LEGACY_DATA_AUDIT_FILENAME,
        LEGACY_DATA_AUDIT_TOMBSTONE,
    )


def retire_legacy_figure_outputs(figure_dir: Path) -> tuple[Path, ...]:
    """Remove only exact misleading PNG/PDF names from old reruns."""
    removed: list[Path] = []
    for basename in LEGACY_FIGURE_BASENAMES:
        for suffix in (".png", ".pdf"):
            candidate = figure_dir / f"{basename}{suffix}"
            try:
                candidate.unlink()
            except FileNotFoundError:
                continue
            removed.append(candidate)
    return tuple(removed)

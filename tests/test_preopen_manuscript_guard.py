from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
GUARD_SCRIPT = ROOT / "scripts" / "_preopen_manuscript_guard.py"
RENDER_SCRIPT = ROOT / "scripts" / "29_render_preopen_manuscripts.py"
GUARD_SPEC = importlib.util.spec_from_file_location(
    "preopen_manuscript_guard_test", GUARD_SCRIPT
)
assert GUARD_SPEC is not None and GUARD_SPEC.loader is not None
GUARD = importlib.util.module_from_spec(GUARD_SPEC)
GUARD_SPEC.loader.exec_module(GUARD)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _preopen_fixture(root: Path) -> None:
    contents = {
        "paper/ThermoRoute_paper.md": b"frozen main\n",
        "paper/cover_letter.md": b"frozen cover\n",
        "paper/highlights.md": b"frozen highlights\n",
    }
    for relative, payload in contents.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    required = [
        "README.md",
        *contents,
        "paper/agu_submission/ThermoRoute_WRR.tex",
    ]
    frozen_hashes = {relative: "0" * 64 for relative in required}
    frozen_hashes.update({
        relative: _sha256(payload) for relative, payload in contents.items()
    })
    registry = {
        "format": "thermoroute.route-a-claim-ledger.v2",
        "required_documents": required,
        "preopen_document_sha256": frozen_hashes,
        "phase_resolver": dict(GUARD.EXPECTED_PHASE_RESOLVER),
    }
    registry_path = root / GUARD.CLAIM_REGISTRY_RELATIVE
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(json.dumps(registry), encoding="utf-8")


def test_preopen_manuscript_guard_accepts_exact_frozen_sources(tmp_path):
    _preopen_fixture(tmp_path)
    actual = GUARD.assert_preopen_manuscript_render_allowed(tmp_path)
    assert set(actual) == set(GUARD.PREOPEN_MANUSCRIPT_SOURCES)


@pytest.mark.parametrize("relative", GUARD.PREOPEN_MANUSCRIPT_SOURCES)
def test_preopen_manuscript_guard_rejects_drift_in_each_source(
    tmp_path, relative,
):
    _preopen_fixture(tmp_path)
    (tmp_path / relative).write_bytes(b"post-opening or otherwise changed\n")
    with pytest.raises(
        GUARD.PreopenManuscriptGuardError,
        match="differs from its frozen SHA-256",
    ):
        GUARD.assert_preopen_manuscript_render_allowed(tmp_path)


def test_preopen_manuscript_guard_rejects_opening_authorization(tmp_path):
    _preopen_fixture(tmp_path)
    authorization = tmp_path / GUARD.EXPECTED_PHASE_RESOLVER["canonical_authorization"]
    authorization.parent.mkdir(parents=True)
    authorization.write_text("{}\n", encoding="utf-8")
    with pytest.raises(GUARD.PreopenManuscriptGuardError, match="authorization exists"):
        GUARD.assert_preopen_manuscript_render_allowed(tmp_path)


def test_preopen_manuscript_guard_rejects_even_empty_confirmation_namespace(
    tmp_path,
):
    _preopen_fixture(tmp_path)
    (tmp_path / "outputs" / "confirmatory").mkdir(parents=True)
    with pytest.raises(
        GUARD.PreopenManuscriptGuardError,
        match="confirmation output namespace exists",
    ):
        GUARD.assert_preopen_manuscript_render_allowed(tmp_path)


def test_preopen_manuscript_guard_rejects_phase_resolver_drift(tmp_path):
    _preopen_fixture(tmp_path)
    registry_path = tmp_path / GUARD.CLAIM_REGISTRY_RELATIVE
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["phase_resolver"]["namespace_glob"] = "outputs/not-confirmation/*"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(GUARD.PreopenManuscriptGuardError, match="phase resolver changed"):
        GUARD.assert_preopen_manuscript_render_allowed(tmp_path)


def test_preopen_guard_static_entry_needs_no_site_packages(tmp_path):
    _preopen_fixture(tmp_path)
    completed = subprocess.run(
        [sys.executable, "-S", str(GUARD_SCRIPT), "--root", str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "PASS frozen PRE-OPEN manuscript sources" in completed.stdout


@pytest.mark.parametrize(
    ("forbidden_state", "error_fragment"),
    (
        ("authorization", "authorization exists"),
        ("confirmation_namespace", "confirmation output namespace exists"),
        (
            "paper/ThermoRoute_paper.md",
            "differs from its frozen SHA-256: paper/ThermoRoute_paper.md",
        ),
        (
            "paper/cover_letter.md",
            "differs from its frozen SHA-256: paper/cover_letter.md",
        ),
        (
            "paper/highlights.md",
            "differs from its frozen SHA-256: paper/highlights.md",
        ),
    ),
)
def test_render_subprocess_fails_before_docx_import_and_preserves_targets(
    tmp_path, forbidden_state, error_fragment,
):
    _preopen_fixture(tmp_path)
    sentinels = {
        "paper/ThermoRoute_paper.docx": b"sentinel-main-docx\n",
        "paper/cover_letter.docx": b"sentinel-cover-docx\n",
        "paper/highlights.docx": b"sentinel-highlights-docx\n",
    }
    for relative, payload in sentinels.items():
        (tmp_path / relative).write_bytes(payload)

    if forbidden_state == "authorization":
        forbidden = (
            tmp_path / GUARD.EXPECTED_PHASE_RESOLVER["canonical_authorization"]
        )
        forbidden.parent.mkdir(parents=True)
        forbidden.write_text("{}\n", encoding="utf-8")
    elif forbidden_state == "confirmation_namespace":
        forbidden = tmp_path / "outputs/confirmatory/arbitrary_namespace/marker.txt"
        forbidden.parent.mkdir(parents=True)
        forbidden.write_text("must refuse\n", encoding="utf-8")
    else:
        (tmp_path / forbidden_state).write_bytes(b"changed after PRE freeze\n")

    completed = subprocess.run(
        [sys.executable, "-S", str(RENDER_SCRIPT), "--root", str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert error_fragment in completed.stderr
    for relative, payload in sentinels.items():
        assert (tmp_path / relative).read_bytes() == payload


def test_preopen_guard_runs_before_python_docx_import():
    source = RENDER_SCRIPT.read_text(encoding="utf-8")
    guard_call = source.index(
        "_EARLY_GUARDED_ROOT = _guard_before_docx_import(sys.argv[1:])"
    )
    docx_import = source.index("from docx import Document")
    assert guard_call < docx_import

"""Shared pytest fixtures for Route-A ownership-sensitive publication tests.

WSL2/Linux developer umask is often ``0o002``, so ``Path.mkdir`` trees become
``0o775`` (group-writable). Publication gates reject parents with group/other
write (``st_mode & 0o022``). Use session umask ``0o022`` so fixtures land as
``0o755``/``0o644`` — owner-controlled without looking like mid-recovery
``0o700`` directories (umask ``0o077`` falsely triggers permission-recovery
resume phases). Production owner-controlled checks stay unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest


# ``pytest`` console entry points do not consistently retain the checkout root
# on ``sys.path`` once the configured ``src`` path is applied.  Several audited
# tests import reusable helpers from ``scripts.*``; make that repository-local
# namespace explicit for collection in both console and ``python -m pytest``
# invocations.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def _owner_controlled_umask() -> None:
    previous = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(previous)

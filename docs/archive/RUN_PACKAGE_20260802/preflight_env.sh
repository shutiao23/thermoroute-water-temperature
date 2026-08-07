#!/usr/bin/env bash
# ThermoRoute run-package environment preflight (target machine, 2026-08-02).
# Verifies Python, dependency versions, source identity, and frozen data.
# Usage: bash docs/RUN_PACKAGE_20260802/preflight_env.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

EXPECTED_SOURCE="0e932f19975033ef0749d2589b60c6aefe057adbef1ee1d9e2f75bef8180920f"
if [[ -x "$REPO_ROOT/.venv-route-a/bin/python" ]]; then
  PY="$REPO_ROOT/.venv-route-a/bin/python"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PY="$REPO_ROOT/.venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi
echo "== Using interpreter: $PY =="

echo "== Python =="
"$PY" --version

echo "== Core dependency versions =="
"$PY" - <<'EOF'
import sys
mods = {
    "torch": None, "pandas": None, "numpy": None, "lightgbm": None,
    "pyarrow": None, "scipy": None, "statsmodels": None, "sklearn": None,
}
for name in mods:
    try:
        m = __import__(name)
        mods[name] = getattr(m, "__version__", "?")
    except Exception as exc:  # noqa: BLE001
        print(f"  MISSING {name}: {exc}")
        sys.exit(1)
for name, version in mods.items():
    print(f"  {name:12s} {version}")
EOF

echo "== Source identity =="
PYTHONPATH=src "$PY" - <<EOF
from thermoroute.repro import source_tree_hash
h = source_tree_hash(".")
print("  source_tree_hash:", h)
assert h == "$EXPECTED_SOURCE", "source mismatch with run package"
print("  OK: matches run package (0e932f19…)")
EOF

echo "== Frozen data artifacts =="
for f in data_usgs/panel_usgs_120v2.parquet \
         data_usgs/station_registry_v1.csv \
         data_usgs/frozen_panel_v1.json \
         data_usgs/development_predictor_bridge_v1.json; do
  if [[ -f "$f" ]]; then echo "  OK $f"; else echo "  MISSING $f"; exit 1; fi
done

echo "== No live processes =="
pgrep -f '09_usgs_experiment|09b_development_controls|phase2_watch' && {
  echo "  WARNING: live process found (see above)"; exit 1
} || echo "  clean"

echo "== Focused regression (T1-T5) =="
"$PY" -m pytest -q tests/test_development_controls.py tests/test_stage09b_precompute.py

echo "PREFLIGHT OK"

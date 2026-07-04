#!/usr/bin/env bash
# Build the Zenodo release archive for ThermoRoute.
#
# Includes: all source, scripts, tests, the manuscript + submission package,
# input station panels (the actual model inputs, ~30 MB), and the small derived
# artifacts (tables, reports, the sha256 manifest). EXCLUDES the multi-hundred-MB
# prediction parquets and model checkpoints under outputs/{predictions,models}
# and the .git history — those are regenerable from this archive via
# scripts/run_all.sh, and the sha256 manifest certifies the exact bytes.
#
# Usage:  bash scripts/make_release_archive.sh
# Output: dist/thermoroute_release_v<version>.zip  (+ printed sha256)
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION=$(python3 -c "import json;print(json.load(open('.zenodo.json'))['version'])")
OUT="dist/thermoroute_release_v${VERSION}.zip"
mkdir -p dist

# refresh the artifact manifest so the archive carries a current evidence chain
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python3 scripts/14_manifest.py >/dev/null 2>&1 || \
  echo "[warn] manifest refresh skipped (14_manifest.py failed)"

rm -f "$OUT"
zip -r -q "$OUT" \
  src scripts tests paper \
  README.md LICENSE .zenodo.json requirements.txt requirements-lock.txt \
  .github \
  data_usgs/panel_usgs_100.parquet data_usgs/panel_usgs_wind.parquet \
  outputs/tables outputs/reports outputs/figures \
  -x '*.pyc' -x '*__pycache__*' -x '*/.DS_Store' \
  -x 'paper/**/*.aux' -x 'paper/**/*.log' -x 'paper/**/*.out' \
  -x 'paper/**/*.bbl' -x 'paper/**/*.blg' -x 'paper/**/*.synctex.gz'

SHA=$(shasum -a 256 "$OUT" | awk '{print $1}')
SIZE=$(ls -lh "$OUT" | awk '{print $5}')
echo "built  $OUT  ($SIZE)"
echo "sha256 $SHA"
echo ""
echo "Next: upload $OUT to https://zenodo.org (New upload -> reserve DOI ->"
echo "fill authors from .zenodo.json -> publish), then paste the minted DOI into"
echo "outputs/reports/data_availability.md and the manuscript Open Research section."

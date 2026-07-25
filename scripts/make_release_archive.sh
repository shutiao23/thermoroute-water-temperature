#!/usr/bin/env bash
# Build one of the two explicit Route-A local evidence profiles.
#
# PREOPEN_NOT_COMPLETE (default) is local development evidence only. It contains no
# old active outputs, confirmation namespace or labels and cannot support a
# Route-A confirmatory conclusion.  The separately verified full-history bundle
# intentionally retains reachable deleted Git objects for chronology; this is not
# a byte-level purge and does not make those objects current evidence. The
# archive contains material with unresolved redistribution rights and MUST NOT
# be transferred to a third party or published.
#
# ROUTE_A_OPENED_COMPLETE is accepted only when a production authorization and
# its canonical one-shot namespace close over every model, pre-label input, raw
# response, outcome, prediction, statistic, report, receipt and attestation.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT_DIR="$PWD"
readonly THERMOROUTE_PYTHON="${THERMOROUTE_PYTHON:-python}"
if ! command -v "$THERMOROUTE_PYTHON" >/dev/null 2>&1; then
  echo "release refused: cannot find THERMOROUTE_PYTHON=$THERMOROUTE_PYTHON" >&2
  exit 2
fi
"$THERMOROUTE_PYTHON" -c \
  'import sys; v=sys.version_info[:2]; sys.exit(f"Route A requires Python 3.12, got {v[0]}.{v[1]}") if v != (3, 12) else None'
PROFILE="PREOPEN_NOT_COMPLETE"
AUTHORIZATION=""
DISTRIBUTION=""

usage() {
  echo "usage: bash scripts/make_release_archive.sh [--profile PREOPEN_NOT_COMPLETE|ROUTE_A_OPENED_COMPLETE] [--authorization PATH] [--distribution LOCAL_EVIDENCE_ONLY|PUBLIC]" >&2
}

while (( $# > 0 )); do
  case "$1" in
    --profile)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      PROFILE="$2"
      shift 2
      ;;
    --authorization)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      AUTHORIZATION="$2"
      shift 2
      ;;
    --distribution)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      DISTRIBUTION="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$DISTRIBUTION" ]]; then
  echo "archive refused: explicit --distribution is required" >&2
  echo "use LOCAL_EVIDENCE_ONLY only on the owner-controlled machine" >&2
  exit 2
fi

case "$DISTRIBUTION" in
  LOCAL_EVIDENCE_ONLY)
    ;;
  PUBLIC)
    echo "archive refused: PUBLIC distribution is blocked pending a byte-bound rights review" >&2
    echo "the local evidence profile contains unverified redistribution material" >&2
    exit 2
    ;;
  *)
    echo "archive refused: unknown distribution mode: $DISTRIBUTION" >&2
    usage
    exit 2
    ;;
esac

case "$PROFILE" in
  PREOPEN_NOT_COMPLETE)
    if [[ -n "$AUTHORIZATION" ]]; then
      echo "release refused: PREOPEN_NOT_COMPLETE must not accept an authorization" >&2
      exit 2
    fi
    ;;
  ROUTE_A_OPENED_COMPLETE)
    if [[ -z "$AUTHORIZATION" ]]; then
      echo "release refused: ROUTE_A_OPENED_COMPLETE requires --authorization" >&2
      exit 2
    fi
    if [[ ! -f "$AUTHORIZATION" ]]; then
      echo "release refused: authorization is absent: $AUTHORIZATION" >&2
      exit 2
    fi
    if [[ ! -f "outputs/prelabel/route_a_prelabel_chronology_v1.json" ]]; then
      echo "release refused: prelabel chronology receipt is absent" >&2
      exit 2
    fi
    ;;
  *)
    echo "release refused: unknown profile: $PROFILE" >&2
    usage
    exit 2
    ;;
esac

if [[ "$PROFILE" == "ROUTE_A_OPENED_COMPLETE" ]]; then
  # This production state may contain exactly the one create-only authorization
  # and its canonical opening namespace.  The local dirty override is never
  # allowed to weaken an archive that claims confirmatory completeness.
  "$THERMOROUTE_PYTHON" -I -B scripts/verify_release.py \
    --check-postopen-dirt --source-root "$ROOT_DIR" \
    --authorization "$AUTHORIZATION"
else
  DIRTY="$(git status --porcelain --untracked-files=all)"
  if [[ -n "$DIRTY" ]]; then
    echo "release refused: formal pre-opening release requires a clean Git worktree" >&2
    echo "$DIRTY" >&2
    exit 2
  fi
fi

required=(
  pyproject.toml requirements.txt requirements-lock.txt requirements-lock-py312-hashed.txt README.md LICENSE
  data/b1.csv data/s2.csv data/p3.csv
  data_usgs/panel_usgs_120v2.parquet
  data_usgs/station_registry_v1.csv
  data_usgs/stations_meta_120v2.csv
  data_usgs/frozen_panel_v1.json
  data_usgs/huc_metadata_usgs_v1.csv
  data_usgs/huc_metadata_usgs_v1.provenance.json
  data_usgs/raw_snapshots/huc-v1/snapshot_index.json
  protocols/route_a_confirmatory_v1.json
  protocols/route_a_confirmatory_protocol.md
  protocols/route_a_protocol_seal_v1.json
  protocols/route_a_inference_amendment_v2.json
  protocols/route_a_inference_amendment_seal_v2.json
  protocols/route_a_probability_metric_erratum_v1.json
  protocols/route_a_probability_metric_erratum_seal_v1.json
  protocols/route_a_model_matrix_amendment_v1.json
  protocols/route_a_model_matrix_amendment_seal_v1.json
  protocols/route_a_native_thread_enforcement_notice_v1.md
  protocols/route_a_native_artifact_publication_notice_v1.md
  protocols/legacy_three_site_semantics_notice_v1.md
  protocols/route_a_claim_registry_v1.json
  scripts/26_validate_claims.py
  scripts/deterministic_zip.py scripts/verify_release.py
)
for path in "${required[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "release refused: required file missing: $path" >&2
    exit 2
  fi
done
for path in src scripts tests .github protocols; do
  if [[ ! -d "$path" ]]; then
    echo "release refused: required directory missing: $path" >&2
    exit 2
  fi
done

# The manuscript source boundary is explicit.  Rendered PDFs/DOCX files and
# historical figure binaries are not evidence unless they are regenerated and
# bound by the current release contract, so never copy paper/ wholesale.
paper_paths=(
  paper/ThermoRoute_paper.md
  paper/highlights.md
  paper/cover_letter.md
  paper/references.bib
  paper/agu_submission/README.md
  paper/agu_submission/ThermoRoute_WRR.tex
  paper/agu_submission/agujournal2019.cls
  paper/agu_submission/build_agu.py
)
for path in "${paper_paths[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "release refused: registered manuscript source missing: $path" >&2
    exit 2
  fi
done

VERSION="$("$THERMOROUTE_PYTHON" -c 'import pathlib, tomllib; print(tomllib.loads(pathlib.Path("pyproject.toml").read_text(encoding="utf-8"))["project"]["version"])')"
DIST_DIR="$ROOT_DIR/dist"
OUT="$DIST_DIR/thermoroute_LOCAL_EVIDENCE_DO_NOT_DISTRIBUTE_v${VERSION}_${PROFILE}.zip"
SHA_FILE="${OUT}.sha256"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/thermoroute-release.XXXXXX")"
STAGE="$TMP_ROOT/thermoroute"
TMP_ZIP="$TMP_ROOT/thermoroute_release.zip"

cleanup() {
  local status=$?
  trap - EXIT
  rm -rf "$TMP_ROOT" || true
  exit "$status"
}
trap cleanup EXIT

SOURCE_GIT_COMMIT="$(git rev-parse HEAD)"
SOURCE_GIT_TREE="$(git rev-parse 'HEAD^{tree}')"

mkdir -p "$STAGE"

copy_path() {
  local source="$1"
  local destination="$STAGE/$source"
  mkdir -p "$(dirname "$destination")"
  cp -R "$source" "$destination"
}

copy_tracked_tree() {
  local source="$1"
  local tracked=""
  while IFS= read -r -d '' tracked; do
    copy_path "$tracked"
  done < <(git ls-files -z -- "$source")
}

# Common source/material is profile-independent.  Scientific result directories
# are never copied wholesale; the opened profile materializer selects only the
# authorization-derived current namespace.
for path in src scripts tests .github protocols; do
  copy_tracked_tree "$path"
done
for path in "${paper_paths[@]}"; do
  copy_path "$path"
done
for path in README.md LICENSE .gitignore pyproject.toml requirements.txt \
            requirements-lock*.txt; do
  copy_path "$path"
done
mkdir -p "$STAGE/data"
cp data/b1.csv data/s2.csv data/p3.csv "$STAGE/data/"

PROFILE_ARGS=(
  --materialize-profile "$STAGE"
  --source-root "$ROOT_DIR"
  --profile "$PROFILE"
  --distribution "$DISTRIBUTION"
)
if [[ -n "$AUTHORIZATION" ]]; then
  PROFILE_ARGS+=(--authorization "$AUTHORIZATION")
fi
"$THERMOROUTE_PYTHON" -I -B scripts/verify_release.py "${PROFILE_ARGS[@]}"
# Establish and independently replay the archive-to-bundle protected-source
# binding before any Python copied into the stage is allowed to execute.
"$THERMOROUTE_PYTHON" -I -B scripts/verify_release.py \
  --materialize-git-history "$STAGE" --source-root "$ROOT_DIR" \
  --profile "$PROFILE" --distribution "$DISTRIBUTION"
"$THERMOROUTE_PYTHON" -I -B scripts/verify_release.py \
  --materialize-claim-audit "$STAGE" --profile "$PROFILE" \
  --distribution "$DISTRIBUTION"

# A pre-opening archive must have exactly one outputs artifact: its provenance
# manifest.  A post-opening archive receives only the canonical namespace files
# copied by the profile materializer above.  No stale cohort output is copied.
mkdir -p "$STAGE/outputs"
MANIFEST_ARGS=(
  --root "$STAGE"
  --manifest "$STAGE/outputs/manifest.json"
  --no-git
  --source-git-commit "$SOURCE_GIT_COMMIT"
  --source-git-tree "$SOURCE_GIT_TREE"
)
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  MANIFEST_ARGS+=(--source-git-dirty)
fi
PYTHONDONTWRITEBYTECODE=1 "$THERMOROUTE_PYTHON" "$STAGE/scripts/14_manifest.py" \
  "${MANIFEST_ARGS[@]}"
PYTHONDONTWRITEBYTECODE=1 "$THERMOROUTE_PYTHON" "$STAGE/scripts/14_manifest.py" \
  --root "$STAGE" --manifest "$STAGE/outputs/manifest.json" --check --no-git

mkdir -p "$DIST_DIR"
PYTHONDONTWRITEBYTECODE=1 "$THERMOROUTE_PYTHON" scripts/deterministic_zip.py \
  "$STAGE" "$TMP_ZIP" --archive-root thermoroute
mv "$TMP_ZIP" "$OUT"

SHA="$(shasum -a 256 "$OUT" | awk '{print $1}')"
printf '%s  %s\n' "$SHA" "$(basename "$OUT")" > "$SHA_FILE"
SIZE="$(ls -lh "$OUT" | awk '{print $5}')"

# Production verification always invokes the fixed trusted replay interface for
# ROUTE_A_OPENED_COMPLETE.  PREOPEN_NOT_COMPLETE never touches outcome code/data.
"$THERMOROUTE_PYTHON" -I -B scripts/verify_release.py "$OUT" \
  --distribution "$DISTRIBUTION"

echo "profile $PROFILE"
echo "scope   LOCAL_OWNER_EVIDENCE_ONLY — DO NOT DISTRIBUTE"
echo "built  $OUT  ($SIZE)"
echo "sha256 $SHA"
echo "checksum $SHA_FILE"

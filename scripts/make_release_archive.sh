#!/usr/bin/env bash
# Build one of the two explicit Route-A release profiles.
#
# PREOPEN_NOT_COMPLETE (default) is development evidence only.  It contains no
# old outputs, confirmation namespace or labels and cannot support a Route-A
# confirmatory conclusion.
#
# ROUTE_A_OPENED_COMPLETE is accepted only when a production authorization and
# its canonical one-shot namespace close over every model, pre-label input, raw
# response, outcome, prediction, statistic, report, receipt and attestation.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT_DIR="$PWD"
PROFILE="PREOPEN_NOT_COMPLETE"
AUTHORIZATION=""

usage() {
  echo "usage: bash scripts/make_release_archive.sh [--profile PREOPEN_NOT_COMPLETE|ROUTE_A_OPENED_COMPLETE] [--authorization PATH]" >&2
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
  PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_release.py \
    --check-postopen-dirt --source-root "$ROOT_DIR" \
    --authorization "$AUTHORIZATION"
else
  if [[ "${ALLOW_DIRTY_RELEASE:-0}" != "1" ]]; then
    DIRTY="$(git status --porcelain --untracked-files=all)"
    if [[ -n "$DIRTY" ]]; then
      echo "release refused: Git worktree is dirty" >&2
      echo "$DIRTY" >&2
      echo "Commit/stash changes, or set ALLOW_DIRTY_RELEASE=1 for a local non-release test." >&2
      exit 2
    fi
  fi
fi

required=(
  pyproject.toml requirements.txt requirements-lock.txt requirements-lock-py312-hashed.txt README.md LICENSE .zenodo.json
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
for path in src scripts tests paper .github protocols; do
  if [[ ! -d "$path" ]]; then
    echo "release refused: required directory missing: $path" >&2
    exit 2
  fi
done

VERSION="$(python3 -c "import json; print(json.load(open('.zenodo.json'))['version'])")"
DIST_DIR="$ROOT_DIR/dist"
OUT="$DIST_DIR/thermoroute_release_v${VERSION}_${PROFILE}.zip"
SHA_FILE="${OUT}.sha256"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/thermoroute-release.XXXXXX")"
STAGE="$TMP_ROOT/thermoroute"
TMP_ZIP="$TMP_ROOT/thermoroute_release.zip"

cleanup() {
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

SOURCE_GIT_COMMIT="$(git rev-parse HEAD)"
SOURCE_GIT_TREE="$(git rev-parse 'HEAD^{tree}')"
SOURCE_GIT_DIRTY=()
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  SOURCE_GIT_DIRTY=(--source-git-dirty)
fi

mkdir -p "$STAGE"

copy_path() {
  local source="$1"
  local destination="$STAGE/$source"
  mkdir -p "$(dirname "$destination")"
  cp -R "$source" "$destination"
}

# Common source/material is profile-independent.  Scientific result directories
# are never copied wholesale; the opened profile materializer selects only the
# authorization-derived current namespace.
for path in src scripts tests paper .github protocols; do
  copy_path "$path"
done
for path in README.md LICENSE .zenodo.json .gitignore pyproject.toml \
            requirements.txt requirements-lock.txt; do
  copy_path "$path"
done
mkdir -p "$STAGE/data"
cp data/b1.csv data/s2.csv data/p3.csv "$STAGE/data/"

PROFILE_ARGS=(
  --materialize-profile "$STAGE"
  --source-root "$ROOT_DIR"
  --profile "$PROFILE"
)
if [[ -n "$AUTHORIZATION" ]]; then
  PROFILE_ARGS+=(--authorization "$AUTHORIZATION")
fi
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_release.py "${PROFILE_ARGS[@]}"
# Establish and independently replay the archive-to-bundle protected-source
# binding before any Python copied into the stage is allowed to execute.
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_release.py \
  --materialize-git-history "$STAGE" --source-root "$ROOT_DIR" \
  --profile "$PROFILE"
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_release.py \
  --materialize-claim-audit "$STAGE" --profile "$PROFILE"

# A pre-opening archive must have exactly one outputs artifact: its provenance
# manifest.  A post-opening archive receives only the canonical namespace files
# copied by the profile materializer above.  No stale cohort output is copied.
mkdir -p "$STAGE/outputs"
PYTHONDONTWRITEBYTECODE=1 python3 "$STAGE/scripts/14_manifest.py" \
  --root "$STAGE" --manifest "$STAGE/outputs/manifest.json" --no-git \
  --source-git-commit "$SOURCE_GIT_COMMIT" --source-git-tree "$SOURCE_GIT_TREE" \
  "${SOURCE_GIT_DIRTY[@]}"
PYTHONDONTWRITEBYTECODE=1 python3 "$STAGE/scripts/14_manifest.py" \
  --root "$STAGE" --manifest "$STAGE/outputs/manifest.json" --check --no-git

mkdir -p "$DIST_DIR"
PYTHONDONTWRITEBYTECODE=1 python3 scripts/deterministic_zip.py \
  "$STAGE" "$TMP_ZIP" --archive-root thermoroute
mv "$TMP_ZIP" "$OUT"

SHA="$(shasum -a 256 "$OUT" | awk '{print $1}')"
printf '%s  %s\n' "$SHA" "$(basename "$OUT")" > "$SHA_FILE"
SIZE="$(ls -lh "$OUT" | awk '{print $5}')"

# Production verification always invokes the fixed trusted replay interface for
# ROUTE_A_OPENED_COMPLETE.  PREOPEN_NOT_COMPLETE never touches outcome code/data.
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_release.py "$OUT"

echo "profile $PROFILE"
echo "built  $OUT  ($SIZE)"
echo "sha256 $SHA"
echo "checksum $SHA_FILE"

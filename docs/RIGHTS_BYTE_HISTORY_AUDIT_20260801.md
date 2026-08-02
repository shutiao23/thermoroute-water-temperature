# Rights exposure audit — Git trees and history

| Field | Value |
| --- | --- |
| Snapshot date | 2026-08-01 |
| HEAD | `bc6d4eaca8b5c327da66b52e815f8ac4dd21a105` |
| Method | Read-only Git ref/tree/history/object-store inventory |
| Scope | Initial 10-ref tree/history pass plus later dynamic local-ref and public-remote snapshots; no tags observed |
| Safety | No ref, remote, worktree, output or Stage-09 mutation; no confirmation target-label inspection |
| Verdict | **EXPOSURE FACTS COLLECTED / BYTE SHA INVENTORY AND RIGHTS DECISIONS INCOMPLETE** |

This record separates three questions that must not be conflated:

1. Is a byte object present or retrievable from a local/public Git lineage?
2. What provider, author or third party does it appear to come from?
3. Has redistribution/relicensing been affirmatively reviewed?

This audit advances question 1 and identifies owners for question 2. It does not
answer question 3 and is not a FAIR or PUBLIC-release approval.

## 1. Direct ref exposure

### Legacy three-site data

Every inspected ref exposes the same `data/` tree containing:

- `data/b1.csv`
- `data/p3.csv`
- `data/s2.csv`

Their source and redistribution authority remain undocumented.

### `data_usgs` trees

The refs expose two principal direct tree variants.

The Route-A/feature variant contains the frozen/development panel closure,
including:

- environmental and predictor-bridge audits;
- frozen and refreshed 2018–2020 predictor Parquet files;
- request maps and bridge reports;
- frozen-panel declaration;
- HUC metadata and provenance;
- the station registry.

The main/fix variant contains the legacy panel family and station metadata,
including:

- `panel_usgs.parquet`, `panel_usgs_100.parquet`,
  `panel_usgs_120v2.parquet`, and `panel_usgs_wind.parquet`;
- rejected-site and station-metadata CSV files.

Both variants directly expose the same `data_usgs/raw_snapshots` tree.

### Raw service responses

The raw-snapshot tree includes request metadata and `response.bin` objects for:

- USGS NWIS site metadata / HUC evidence;
- ORNL Daymet single-pixel predictor bridge requests;
- gridMET OPeNDAP schema and NCSS requests;
- snapshot indexes and their later schema versions.

These are the largest direct provider-byte exposure category. “Provider service
was publicly reachable” is not, by itself, a redistribution or relicensing
decision for the stored responses.

### Other direct third-party or release-relevant paths

- `paper/agu_submission/agujournal2019.cls`
- dependency declaration and lock files
- the repository `LICENSE`

The software license must not be projected onto the data, raw responses, or AGU
class without separate evidence.

## 2. Reachable history-only exposure

The following categories are absent from the current tip at those paths but are
still reachable from commit-ref history.

### Work-in-progress data and extracted publication material

- `_archive_wip/panel_usgs_100_wip.parquet`
- `_archive_wip/usgs_predictions_wip.parquet`
- `_archive_wip/usgs_scores_wip.csv`
- `_archive_wip/usgs_experiment_wip.md`
- `tmp/pdfs/topp2023_page-10.png` through `topp2023_page-15.png`
- `tmp/pdfs/topp2023_stream_temp.txt`

The extracted paper pages/text require publisher/author/source review; deleting
their tip paths did not remove the history exposure.

### Historical outputs and submission packages

Route-A branch history retains deleted:

- `outputs/figures/*.{png,pdf}`;
- `outputs/manifest.json`;
- `outputs/reports/*`;
- `outputs/tables/*.{csv,md,npz}`;
- manuscript, cover-letter and highlights DOCX/PDF files;
- AGU BBL/PDF/template/figure/bibliography/`trackchanges.sty` files.

### Historical release claims

`.zenodo.json` is history-only on the relevant feature/yiqu lineages. Any earlier
license or openness statement in that file must not be reused as current
authority without a fresh byte-scoped review.

### Earlier versions of current paths

At least one prior `development_predictor_bridge_v1.json` blob is history-only.
An earlier `stations_meta.csv` blob is not history-only at the byte level because
the same blob remains directly exposed under another current path.

## 3. Local unreachable objects

`git fsck --full --no-reflogs --unreachable` reports unreachable blobs and trees,
but no unreachable commits. They were deliberately not expanded or read during
this audit. They remain a separate `local_unreachable/unclassified` inventory
class until byte/object review determines what they contain and whether they are
within any release or incident-remediation scope.

A later metadata-only recount on the same date found 270 unreachable objects:
185 blobs and 85 trees, with no unreachable commit reported. The command emitted
object IDs/types only; payloads and tree contents were not opened. Counts are a
volatile local-object-store snapshot and are not a byte classification.

The local object store currently has two packs, about 30,335 packed objects and
approximately 1.17 GiB of packed data. No tracked `.gitattributes` or
`.lfsconfig` was found and the Git LFS client is not installed; those facts do not
prove that no remote lineage ever used LFS.

## 4. Required byte-object inventory schema

The final inventory must deduplicate content while retaining every path/ref
observation. At minimum it should contain:

```text
schema_version
generated_at_utc
repo_id
head_oid
ref_name
ref_oid
ref_type
reachability                  # direct_ref | reachable_history | local_unreachable
git_object_oid
git_object_type
git_mode
byte_size
sha256
path
first_seen_commit
last_seen_commit
exposure_class                # raw response | metadata | panel | output | manuscript | third party | dependency | unknown
provider_hint
provenance_locator
rights_evidence_locator
assessment_status             # unreviewed | evidence_collected | externally_resolved
```

The inventory must cover every inspected ref tree, all commit trees reachable by
`git rev-list --all`, and separately enumerated unreachable blob/tree roots. A
worktree-only hash list is insufficient. The full content pass should run in a
low-I/O stationary window and must not share resources with Stage-09 validation
or Phase-2 materialization.

## 5. External decisions still required

Official-source policy evidence collected after this history pass is recorded in
`docs/RIGHTS_PROVIDER_EVIDENCE_20260801.md`. It narrows but does not close the
decisions below; exact-byte mapping and qualified review are still required.

| Byte class | Required external evidence or owner decision |
| --- | --- |
| USGS/NWIS responses and derived station/panel data | applicable USGS terms, attribution, series-specific constraints and redistribution conclusion |
| ORNL Daymet responses/metadata | service/data terms, attribution and repackaging conclusion |
| gridMET responses/metadata | actual provider/product terms, attribution and repackaging conclusion |
| `data/*.csv` | source, collector/rightsholder and redistribution authority |
| `tmp/pdfs/topp*` | article/publisher/database rights and remediation decision |
| AGU class/template and `trackchanges.sty` | third-party license/notice or exclusion decision |
| dependency wheels/binaries | license notices if redistributed in an image/archive |
| manuscript/output history | author/coauthor/employer/funder release and incident-remediation decision where applicable |

## 6. Current release consequence

The PUBLIC builder must remain fail-closed. The safe fallback remains:

- source code whose license has been reviewed;
- synthetic fixtures;
- legal retrieval scripts and metadata;
- no legacy/raw/provider bytes whose redistribution decision is unresolved.

No remote visibility change, history rewrite, object deletion, archive build,
deposit or DOI action is authorized by this audit.

## 7. Dynamic-ref follow-up

A later `git for-each-ref` snapshot found 12 local refs: four Codex tree refs and
eight commit refs, resolving to eight unique object IDs. The previously discussed
`origin/pull/1/head` ref was not present locally. Therefore the final generator
must enumerate live refs at execution time and record the resulting ref set; it
must not hard-code the earlier ref list or assume that ref count is stationary.

## 8. Public-remote visibility follow-up

A read-only GitHub/remote snapshot later on 2026-08-01 returned:

- repository visibility `PUBLIC`, default branch `main`, zero forks and last
  recorded push `2026-07-28T17:44:35Z`;
- four remote branch heads: `feat/route-a-completion`, `feat/yiqu-upgrade`,
  `fix/delta-leakage-lightgbm-parity-p0p1` and `main`;
- one advertised pull ref, `refs/pull/1/head`, at the same object as the fix
  branch.

The pull ref exists on the remote even though no matching local remote-tracking
ref was present. A still later local snapshot contained 11 refs (four Codex tree
refs and seven commit refs) resolving to eight unique object IDs. The change from
the earlier local count illustrates why the final manifest must separately
capture `git for-each-ref` and `git ls-remote --refs origin` at execution time.
Neither public visibility nor remote reachability supplies redistribution rights.

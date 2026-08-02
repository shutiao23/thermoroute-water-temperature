# FAIR / rights acceptance matrix — P0-F

| Field | Value |
| --- | --- |
| Date | 2026-08-02 |
| Blueprint item | P0-F: FAIR, rights, DOI and a reproducible public release |
| Current status | **`EVIDENCE_COLLECTED + BLOCKED_EXTERNAL`** |
| Scope | Static, docs-only acceptance map. No archive, deposit, DOI, remote, history or visibility action is authorized here. |
| Decision boundary | This is not legal advice, a rights decision, a release receipt, or a statement that any byte may be redistributed. |

## 1. P0-F completion rule

[`FAIR_SUBMISSION_READINESS_AND_TEMPLATES.md`](FAIR_SUBMISSION_READINESS_AND_TEMPLATES.md) defines P0-F as complete only when every public byte has an inventory identity and rights decision, release metadata agree, a public archive validates on a clean host, final author declarations are signed, and the DOI resolves to the reviewed archive. This matrix decomposes that gate. Any row below marked `BLOCKED_EXTERNAL`, `MISSING`, `EXCLUDE_PUBLIC`, or `UNRESOLVED_EXCLUDE` keeps P0-F open.

The existing release contract is a separate local-evidence system. Its mechanics are useful evidence for a future public design, but a local evidence receipt cannot satisfy a P0-F public-release criterion.

## 2. Acceptance matrix

| Acceptance requirement | Current evidence (not a rights conclusion) | Missing external input / responsible role | Automated evidence or future receipt | Required close-out evidence |
| --- | --- | --- | --- | --- |
| **A. Inventory identity for every exposed public byte and proposed public member.** | [`RIGHTS_BYTE_HISTORY_AUDIT_20260801.md`](RIGHTS_BYTE_HISTORY_AUDIT_20260801.md) records direct-ref, reachable-history and local-unreachable exposure classes and gives the required final schema. [`RIGHTS_INVENTORY_DRAFT.md`](RIGHTS_INVENTORY_DRAFT.md) provides procedures but says they are not a completed decision. Local and remote ref sets are dynamic. | Owner preserves an audit copy and runs a full low-I/O inventory over live local refs, `git ls-remote --refs origin`, every reachable commit tree and separately enumerated unreachable roots. | Run the §3 procedures in `RIGHTS_INVENTORY_DRAFT.md` only in a stationary window. Future self-hashed inventory receipt must bind generator version, ref snapshots, object/path observations and SHA-256 tables. | A content-deduplicated SHA-256 table retaining every `ref/path/blob/reachability` observation; no exposed object or candidate member remains unclassified. |
| **B. Exact-byte include/exclude decision.** | [`RIGHTS_PROVIDER_EVIDENCE_20260801.md`](RIGHTS_PROVIDER_EVIDENCE_20260801.md) defines object-level fields and only `INCLUDE_PUBLIC`, `EXCLUDE_PUBLIC`, `UNRESOLVED_EXCLUDE`; it expressly rejects a provider-wide fact as a byte licence. | Qualified rights/data reviewer supplies product/version/retrieval mapping, policy snapshot hash, attribution and scope analysis for every object. | Future signed/attested byte-decision manifest using provider-evidence §2 schema. Its validator rejects missing evidence, duplicate/ambiguous objects and any non-`INCLUDE_PUBLIC` archive member. | Every exposed/candidate object has one SHA-256-bound disposition; every public member also has reviewed `INCLUDE_PUBLIC`. No evidence means `UNRESOLVED_EXCLUDE`, never default inclusion. |
| **C. Derived data have their own provenance-rights closure.** | MIT applies only to software. Existing audits identify USGS, Daymet and gridMET components and state that derived panels cannot inherit the most permissive upstream status automatically. | Component/provenance map from each derived object to source product/version/transformation, plus qualified decisions on repackaging, relicensing and attribution. | Future provenance-to-rights closure receipt joins each derived SHA-256 to component decisions and fails if any component is unresolved/excluded. | Every derived byte has a complete component closure and a separately scoped data/archive licence. |
| **D. Third-party TeX/template bytes are file-scoped.** | `agujournal2019.cls` has recorded SHA-256 `deca12479ddeeeae31ed5873a7ebd21251858aaa1dea1b403e91a229d92b83d7`; current evidence supports submission use, not archive redistribution. `THIRD_PARTY_NOTICES` is absent. | AGU/Wiley file-level permission/licence, or an implemented exclusion plan. Review every vendored `.cls`, `.sty`, `.bst`, macro and logo actually placed in an archive. | Future allowlist test rejects such bytes unless their SHA-256 appears in a reviewed notice/decision manifest. Future third-party-notice validation receipt binds the final member list. | Each third-party byte is reviewed `INCLUDE_PUBLIC` with notice, or excluded. A bundle requiring an excluded class is not a public compilable bundle unless a legal user-supplied dependency build is independently verified. |
| **E. Legacy CSV, manuscript/output and dependency bytes remain distinct classes.** | `data/b1.csv`, `data/p3.csv`, `data/s2.csv` lack authoritative source records. History contains WIP panels/predictions, extracted paper material, legacy outputs and manuscript packages. Dependency notices matter only for binaries/images actually redistributed. | Rightsholder/source/collection terms for legacy data; author/coauthor/employer/funder or publisher decisions for manuscript/output objects; per-byte dependency notices as applicable. | Future byte-class decision receipt, with reviewer and evidence locator. Validator must reject classification by path prefix or repository MIT header alone. | No legacy/third-party byte enters an archive without a separate SHA-256 decision. |
| **F. Git history is remediated or excluded, not silently released.** | The audit distinguishes tip bytes, reachable-history-only bytes and 270 locally unreachable objects whose payloads were not read. The local evidence archive deliberately carries a Git bundle that may contain unreviewed current and historical bytes. | Owner and qualified governance/rights reviewer determine incident/remediation scope, preserve audit evidence, classify objects and choose the hosting-provider/cached-content process where needed. | Future history-scope/remediation decision receipt binds complete inventory and before/after ref snapshots. Deleting paths or rewriting a ref does not create this receipt. | A public archive excludes all unreviewed history bytes (including any Git bundle), or has reviewed authority for each included historical object. “Tip deleted” and “history rewritten” are insufficient evidence. |
| **G. Public archive is allowlisted, hashed and fail-closed.** | `scripts/make_release_archive.sh` accepts only `LOCAL_EVIDENCE_ONLY`; its local marker declares unverified `data_usgs/**`, Git bundle and AGU class scopes. It is not a public package. | A reviewed public-member specification and new public builder/verifier after A–F; final member list, archive hash and no-unreviewed-byte proof. | Present guard: `python -m pytest -q tests/test_manifest_release.py -k public_distribution_mode_is_unconditionally_blocked`. Future public-member inventory + verifier receipt bind the exact final archive. | Archive contains only `INCLUDE_PUBLIC` objects, complete member SHA-256 inventory and consistent scope metadata; no unreviewed history/raw/provider/cache/checkpoint bytes. |
| **H. Independent public clean-room reproduction.** | Synthetic smoke and same-host local release-mechanics acceptance are intentionally limited. There is no public archive, fixed image, independent replay or POST renderer/PDF-SI E2E proof. | Independent Linux host/operator, fixed image/environment evidence and final build inputs where claims require them. | Future clean-host archive/reproducer receipt, deterministic synthetic-workflow receipt, and separate receipt-to-paper/PDF-SI receipt. | Independent operator runs the documented top-level command against reviewed bytes, without provider/target bytes or network where promised, and reproduces claimed artifacts within declared tolerance. |
| **I. Signed author administration and mutually consistent metadata.** | Author/affiliation/ORCID/CRediT/funding/COI are intentionally absent. `CITATION.cff`, `codemeta.json`, `.zenodo.json` and `THIRD_PARTY_NOTICES` are absent; historical `.zenodo.json` is non-authoritative. | Signed author intake, corresponding-author confirmation, funding/COI declarations, repository/deposit custodian, final scope/version/licence and rights-reviewer approval. | Intake schemas are in `FAIR_SUBMISSION_READINESS_AND_TEMPLATES.md` §§2–4. Future cross-file validator checks creators, version, date, licence, scope and DOI. | Signed declarations and a machine-validated metadata set consistent with the reviewed archive; placeholders and historic Zenodo claims fail. |
| **J. DOI resolves to the reviewed immutable archive.** | No DOI, citable release or authorized deposit exists. | Repository/deposit service, authorized depositor, final archive hash and persistent DOI record. | Future deposit-resolution receipt records DOI, landing page/version, member manifest/hash and access time. | DOI resolves to the same reviewed bytes/version used in G–I; a package version or local checksum alone is insufficient. |

## 3. Provider and third-party disposition branches

These branches specify a conservative workflow for a future qualified reviewer. They make no rights determination.

| Byte family | Current branch | What must precede consideration of `INCLUDE_PUBLIC` | Default now |
| --- | --- | --- | --- |
| USGS/NWIS responses, station data and USGS-derived objects | Official USGS evidence is useful but not self-applying. | Classify each exact response as USGS-produced rather than partner/third-party material; bind series/provider provenance, policy snapshot, attribution and qualified review. | `UNRESOLVED_EXCLUDE` |
| Daymet V4/V4R1 raw grids, subsets, extracts, aggregates/features, weights or predictions materially encoding Daymet values | Product DOI/version/citation are known; no explicit product-level redistribution licence is recorded, and upstream caveats remain. | Written ORNL/Daymet scope-specific confirmation for the proposed material/use, then exact-object mapping and qualified review. | `UNRESOLVED_EXCLUDE`; public fallback may contain retrieval instructions, version/DOI, hashes and citation only. |
| gridMET responses and derived objects | Existing policy evidence lowers risk but is not a blanket byte decision. | Bind product/version/response bytes and access-date policy snapshot; preserve attribution/provenance; review wrappers/ancillary content. | `UNRESOLVED_EXCLUDE` |
| Mixed derived panel, registry, model or output artifacts | Cannot inherit one upstream source’s permissive status. | Complete component/provenance closure plus derived-object decision. | `UNRESOLVED_EXCLUDE` |
| Legacy three-site CSV | No source/rightsholder record is in the repository. | Identify source, collector/rightsholder, collection terms and redistribution authority for exact bytes. | `UNRESOLVED_EXCLUDE` |
| AGU 2019/2025 class, macros and vendored TeX bytes | Submission availability is distinct from redistribution. | File-level licence/permission and required notice, or exclude and validate a legal external-dependency build. | `EXCLUDE_PUBLIC` |
| History-only WIP data, old outputs, withdrawn `.zenodo.json`, extracted publication material | Exposure/provenance only; old openness claims cannot be reused. | Object classification and remediation/rightsholder decision; repeat A–E if inclusion is ever proposed. | `EXCLUDE_PUBLIC` |
| Local unreachable objects | Metadata-only count; payloads are unclassified. | Scoped classification/remediation decision without treating unread objects as harmless. | Not an archive member; `local_unreachable/unclassified` |
| Source code and synthetic fixtures | Potential safe fallback, still subject to exact-member review. | Confirm licence scope and absence of provider, target, model-cache or unreviewed empirical bytes. | Exclude until member check passes. |

## 4. PUBLIC remains fail-closed

The current mechanical guard is an acceptance test, not a rights conclusion:

```bash
python -m pytest -q tests/test_manifest_release.py \
  -k public_distribution_mode_is_unconditionally_blocked
```

It asserts that the verifier refuses `PUBLIC` before accepting an archive path, the shell builder returns status 2 before staging, and omitted distribution also fails. Equivalent negative checks are:

```bash
bash scripts/make_release_archive.sh --distribution PUBLIC
python scripts/verify_release.py --distribution PUBLIC
```

Both are expected to fail today. A pass would be a release-blocking regression, not progress.

Read-only verification on 2026-08-02 used the project environment and disabled
the pytest cache:

```bash
env PYTHONDONTWRITEBYTECODE=1 .venv-route-a/bin/python -m pytest -q \
  -p no:cacheprovider tests/test_manifest_release.py \
  -k public_distribution_mode_is_unconditionally_blocked
```

Result: `1 passed`. This proves only that the current PUBLIC guard fails closed;
it is not a rights decision, public-builder implementation or P0-F receipt.

`PUBLIC` may not be enabled merely because documentation, a provider page, a visible GitHub repository, an inventory command, a local evidence receipt or a DOI draft exists. A future public profile needs independent evidence for all of the following:

1. Every exact public member is `INCLUDE_PUBLIC` with required attribution/notice; builder and verifier exclude all others.
2. The public list excludes the local evidence Git bundle unless every included historical object independently satisfies item 1.
3. Exact member SHA-256 inventory, metadata-consistency result and clean-host reproduction receipt bind one final archive.
4. Signed author/rightsholder/reviewer declarations and resolved DOI bind that same version and scope.
5. Negative regression tests prove that an unreviewed byte, absent decision, hash mismatch, third-party byte, history bundle or metadata mismatch fails before publication.

Until an implemented, independently reviewed public profile proves all five, `LOCAL_EVIDENCE_ONLY` is the sole enabled distribution state. It is not permission to transfer a local archive to another person or service.

## 5. Evidence / non-evidence ledger

| Existing item | Establishes | Does not establish |
| --- | --- | --- |
| `RIGHTS_INVENTORY_DRAFT.md` | Exposure categories, high-risk objects and inventory procedures. | Procedure execution, byte-level closure or redistribution permission. |
| `RIGHTS_BYTE_HISTORY_AUDIT_20260801.md` | Direct/history/unreachable exposure facts and final inventory schema. | Per-object decisions, remediation or rights approval. |
| `RIGHTS_PROVIDER_EVIDENCE_20260801.md` | Official-source facts, product IDs and conservative defaults. | Automatic licence for stored responses, derived panels or templates. |
| `RIGHTS_CRITICAL_PATH.md` | P1–P5 false-release chains and prohibited inferences. | That public visibility, a tip deletion or a completed Stage closes rights. |
| `FAIR_SUBMISSION_READINESS_AND_TEMPLATES.md` | Intake schemas and the P0-F definition. | Signed administration, metadata, DOI, archive or clean-room proof. |
| Release scripts and `tests/test_manifest_release.py` | A tested present-day local-only/PUBLIC-blocked guard. | A future public archive’s legality, scientific completeness, FAIR completion or external reproducibility. |

## 6. Future P0-F close-out packet

The sealed P0-F protocol must bind, at minimum:

1. complete content-deduplicated inventory with live/remote ref snapshots and reachability class;
2. qualified exact-byte rights-decision manifest and third-party notice register;
3. history-scope/remediation decision record;
4. self/hash-verified public archive and member inventory;
5. independent clean-host reproducer receipt and, if paper claims are included, paper/SI reproduction receipt;
6. signed author, CRediT, funding, COI and rights-reviewer declarations;
7. mutually validated `CITATION.cff`, `codemeta.json`, deposit metadata and data-licence records; and
8. DOI-resolution record binding the deposited bytes to items 1–7.

Any absent, mismatched, unresolved or non-`INCLUDE_PUBLIC` item keeps P0-F open. The safe fallback is a reviewed code-and-synthetic-fixture package with retrieval instructions and provenance metadata, not a repackaged empirical-data or history bundle.

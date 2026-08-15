# R10 — AGU 2025 class assets: rights determination and release handling

| Field | Value |
| --- | --- |
| Date | 2026-08-05 |
| Queue item | R10 |
| Trigger | Commit `6ee35e5` migrated the WRR build to `agujournal2025.cls` and vendored five third-party files into `paper/agu_submission/` |
| Outcome | **`EXCLUDE_PUBLIC`, implemented as a machine-enforced exclusion.** The archive omits the class assets; the archived package README carries the upstream source and a SHA-256 for each file |
| Decision boundary | This records and implements a disposition **already written** in the rights documents. It is not a new rights decision, not legal advice, and does not authorize any redistribution |

---

## 1. The rights determination, and the document that establishes it

The determination did not need to be invented. It was already recorded, in two
places, before the 2025 migration happened.

**Primary: `docs/RIGHTS_PROVIDER_EVIDENCE_20260801.md` §5 ("AGU template
file-level qualification"), disposition list.** Verbatim:

> - `agujournal2019.cls`: `EXCLUDE_PUBLIC`;
> - an otherwise author-owned TeX source bundle that requires that class:
>   `EXCLUDE_AS_COMPILABLE_BUNDLE` unless the class is omitted/replaced or
>   permission is obtained;
> - **AGU 2025 class/macros/logos: `EXCLUDE_PUBLIC` unless a file-level licence
>   or written permission appears;**
> - third-party TeX dependencies: `NOT_PRESENT / REVIEW_IF_VENDORED`.

The same section's reasoning: the 2025 class and `wiley-macros.tex` "remain
unlicensed for downstream redistribution", and "commented example language about
a published article's CC BY-NC-ND status is not a template licence."

**Corroborating: `docs/FAIR_RIGHTS_ACCEPTANCE_MATRIX.md`.** §3 lists the byte
family "AGU 2019/2025 class, macros and vendored TeX bytes" with default
`EXCLUDE_PUBLIC`, closing note "Submission availability is distinct from
redistribution", and the required precondition "File-level licence/permission and
required notice, **or exclude and validate a legal external-dependency build**."
§2 row D requires that each third-party byte be "reviewed `INCLUDE_PUBLIC` with
notice, or excluded".

**Verified against the bytes, not just the documents.** Inspected on 2026-08-05:

| File | Licence text found in the file | Effect |
|---|---|---|
| `agujournal2025.cls` | none; only a commented example CC BY-NC-ND line describing a *published article* | no grant |
| `wiley-macros.tex` | same commented example line | no grant |
| `agu-logo-small.pdf`, `agu-logo-large.pdf` | none (AGU branding) | no grant |
| `tweaklist-git-moderncv-fixed.sty` | **LPPL 1.3c**, "This work may be distributed and/or modified under the conditions of the LaTeX Project Public License version 1.3c" | redistributable, but see below |

`tweaklist-git-moderncv-fixed.sty` is the one separable case, and
`RIGHTS_PROVIDER_EVIDENCE_20260801.md` §5 anticipated it ("declares LPPL 1.3c,
but it is not present in this project"). It is now present. It is nonetheless
excluded with the set: it is a `published`-branch-only dependency, the draft
submission build never loads it, and shipping it alone is useless without the
unlicensed class. Excluding it costs nothing and keeps one rule for one family.

**Determination: `EXCLUDE_PUBLIC`.** These are AGU/Wiley (and, for the `.sty`,
moderncv) bytes, not this project's. Both rights documents say so, and the file
headers confirm that no file-level grant exists to override them. Nothing here
overturns that; journals do not ask authors to redistribute the class, and
`FAIR_RIGHTS_ACCEPTANCE_MATRIX.md` §3's alternative branch — "exclude and
validate a legal external-dependency build" — is the branch taken, with the
build validated in §6.

### 1.1 The exclusion is a named closure condition, not a workaround

`docs/RIGHTS_CRITICAL_PATH.md` path P1 ("AGU class 进 archive") states the
closure condition as: a citable third-party notice **or** exclusion from any
public code/data archive. `docs/RIGHTS_INVENTORY_DRAFT.md` §4 item 3 repeats it
— "可引用条款写入 notice **或** 正式排除决定（两者皆缺）" — a citable notice or a
**formal exclusion decision**, and records that both were missing.

This document supplies the second branch for the five 2025 assets, and
`scripts/verify_release.py` makes it enforceable rather than aspirational.

`docs/FAIR_EXTERNAL_CLOSEOUT_REQUEST_PACK.md` §3 independently enumerates the
four actions an AGU/Wiley permission request would have to separate, and its
action **(iv)** is "an archive that omits the file and asks users to supply a
lawful external dependency". That is precisely what was implemented, so the
AGU permission request is now *optional* for these five files — needed only if
the owner later wants them inside an archive — and remains required for
`agujournal2019.cls`.

P1 is **not** thereby closed. It remains open for `agujournal2019.cls` (§5) and
for repository/history exposure (§6).

---

## 2. The dilemma, and why exclusion resolves it without weakening anything

Before this change:

- `ALLOWED_PAPER_MEMBERS` is an exact-path allowlist and
  `REQUIRED_PAPER_MEMBERS = ALLOWED_PAPER_MEMBERS`, so allowlisting a file also
  makes it mandatory.
- `validate_members` rejects any `paper/` member outside that allowlist.
- `FORBIDDEN_UNREGISTERED_RENDERED_SUFFIXES` rejects every `.pdf`/`.png` member.

An archive containing the assets fails; an archive without them will not compile.

**The `.pdf` half of the dilemma has no lawful mechanical answer.** The constant
is named `FORBIDDEN_UNREGISTERED_RENDERED_SUFFIXES`, so it is natural to assume
there is a *registration* path that would let the logos in. **There is not.** The
constant is consumed at exactly one site, `_validate_release_member_names`, with
no exemption set and no escape hatch; the archive carries zero rendered binaries
by design. Registering the logos would have meant *creating* an exemption to a
rule that currently has none — i.e. weakening the check. That was not done, and
the "unregistered" in the name should be read as "not evidence", not as "not yet
registered".

Exclusion resolves both halves and strengthens the verifier:

1. **Rights**: the archive stays clean of unlicensed third-party bytes, which is
   what the rights documents already required.
2. **Compilation**: reproducibility moves from "bytes we may not ship" to
   "instructions and hashes we may ship". `paper/agu_submission/README.md` is
   itself an archive member, so the archive tells its own reader where to get the
   class and how to confirm identical bytes.
3. **No rule was relaxed.** The forbidden-suffix rule, the exact-allowlist rule
   and the rights gating are untouched. `ALLOWED_PAPER_MEMBERS` is unchanged, so
   `scripts/make_release_archive.sh` and the test that pins the two lists to each
   other stay consistent.

---

## 3. What was implemented

The change spans two branches, which is why the two halves are not visible from
one checkout: `scripts/verify_release.py` was edited on `feat/route-a-remediation`
and everything under `paper/`/`docs/` on `feat/route-a-completion` (which is where
`6ee35e5` vendored the assets). Until they merge, the verifier's
`EXCLUDED_THIRD_PARTY_CLASS_ASSETS` and the README it binds to live on different
branches, and the README-binding check is therefore unsatisfied on
`feat/route-a-remediation` alone — correctly so, since that branch's manuscript
also loads `agujournal2025` without shipping it.

### 3.1 `scripts/verify_release.py` (remediation worktree)

The outcome is stated by name, so a later contributor who re-adds these files
gets the decision instead of a generic rejection.

- **`EXCLUDED_THIRD_PARTY_CLASS_ASSETS`** — the five paths mapped to their
  SHA-256, plus `EXCLUDED_THIRD_PARTY_CLASS_ASSET_SOURCE` (the upstream repo) and
  `EXCLUDED_THIRD_PARTY_CLASS_ASSET_PREFIXES` (the removed upstream tree). The
  block cites `RIGHTS_PROVIDER_EVIDENCE_20260801.md` §5 and
  `FAIR_RIGHTS_ACCEPTANCE_MATRIX.md` §3 inline, and explains why the 2019 class
  is deliberately *not* in the set.
- **Import-time invariant** — if any excluded path ever appears in
  `ALLOWED_PAPER_MEMBERS`, the module raises immediately, naming the licence or
  written permission that would have to exist first. Allowlisting them can no
  longer be done quietly.
- **`_reject_excluded_third_party_class_assets(members)`**, called as the *first*
  statement of `validate_members`, ahead of the suffix and allowlist rules so the
  specific message wins for all five — including the two logos, which would
  otherwise have produced "unregistered rendered/binary manuscript artifact". The
  error names the paths, the rights basis, the upstream source, and says not to
  allowlist them to make it pass.
- **`_verify_excluded_class_asset_retrieval_instructions(root)`**, called on the
  extracted archive root. If the archived `ThermoRoute_WRR.tex` loads
  `agujournal2025`, then `paper/agu_submission/README.md` **must** contain the
  upstream source and every excluded asset's SHA-256, or verification fails. This
  makes the reproducer instructions a *binding archive property* rather than a
  documentation convention. It is scoped to the real dependency: a manuscript on
  a class the archive already carries needs no retrieval note, so the check is a
  no-op there.

**Deliberately not done:** no key was added to
`LOCAL_EVIDENCE_DISTRIBUTION_FIELDS`. `_verify_git_history_evidence` reconstructs
the manifest's `release_evidence.distribution` from *every* key in that mapping
and compares it to what `scripts/deterministic_zip.py` writes from a **hardcoded
15-name tuple**. Adding a key there without editing `deterministic_zip.py` would
have broken the manifest binding on every archive. `known_minimum_unverified_
redistribution_scopes` is also pinned by exact-set assertion in
`tests/test_manifest_release.py`. Both files are outside this task's edit scope,
so the exclusion was expressed as its own registry instead — which is the better
model anyway: that field lists bytes the archive *carries* under a warning, and
these bytes are not carried at all.

### 3.2 `paper/` and `docs/` (manuscript worktree)

- `paper/agu_submission/README.md` — rewritten rights section citing §5 and §3 by
  name, plus a new section **"Obtaining the AGU class assets (required to
  compile this package)"**: upstream URL, the five files with SHA-256, the
  `sha256sum` command to verify them, why they go beside the `.tex` rather than
  on `TEXINPUTS`, and that only `agujournal2025.cls` is needed for the draft
  build.
- `paper/agujournal2025-latex-template-main/` — **removed** (§4).
- `docs/AGU2025_TEMPLATE_MIGRATION.md` — §4.1 rewritten from "owner-owned, open"
  to the resolution; new §4.2 records the removed tree's provenance and manifest.
- `docs/RIGHTS_PROVIDER_EVIDENCE_20260801.md` — dated-snapshot addendum (the
  snapshot body is not rewritten).
- `docs/RIGHTS_INVENTORY_DRAFT.md`, `docs/WRR_SUBMISSION_CHECKLIST.md` — updated.

---

## 4. The pristine upstream tree

`paper/agujournal2025-latex-template-main/` was removed, as recommended. It was a
committed "Download ZIP" of the upstream template. Five of its seven files were
byte-identical to the vendored copies (verified by SHA-256, §5); it therefore
added a second unreferenced copy of the same unlicensed bytes, plus a 141 KB
illustration PNG and `agujournaltemplate.tex` that nothing consumes. Nothing in
the repository referenced the directory path.

Leaving it was not an option: it sits under `paper/`, so any future builder that
enumerates `paper/**` rather than the explicit staged list would have swept
unlicensed bytes — including a `.png` — into an archive. Its prefix is now in
`EXCLUDED_THIRD_PARTY_CLASS_ASSET_PREFIXES`, so restoring it cannot reach an
archive silently either.

Provenance and the full seven-file manifest are recorded in
`docs/AGU2025_TEMPLATE_MIGRATION.md` §4.2, so the deletion is reversible from
upstream.

---

## 5. SHA-256 manifest of the class assets

Upstream source: **`https://github.com/AGU-Publications/agujournal2025-latex-template`**
(AGU-Publications/agujournal2025-latex-template — the repository AGU links from
its author pages). `docs/RIGHTS_PROVIDER_EVIDENCE_20260801.md` records snapshot
commit `355052226d872cf6b9211c12b73b2ed2da133a7d`; the byte correspondence
between the downloaded ZIP and that commit was **not** independently re-verified
here, so the SHA-256 values below are the authoritative bind, not the commit id.

Vendored in `paper/agu_submission/` — **excluded from every archive**:

| File | SHA-256 | Draft build |
|---|---|---|
| `agujournal2025.cls` | `a6645a79392906eb31fd63a4d3f28a2322464c0552f93c4a35da9f4a85067fab` | required |
| `tweaklist-git-moderncv-fixed.sty` | `85a31337d69412566a227209ec908f3a3242ecf673819ca86564d9dc23bfbef1` | not used (`published` only) |
| `wiley-macros.tex` | `7ea18648b7bd2c632065d4303830a54192626450139d76db39760516a16ef71c` | not used at all |
| `agu-logo-small.pdf` | `c5c18836bc66fff737254bb0a2d321afc538e90028a364e63c0de564efc5abdf` | not used (`published` only) |
| `agu-logo-large.pdf` | `cf6382d62086d149d5538dc65c2eeac79c9e33bcc2730f70cb36c7feb324b1b5` | not used (`published` only) |

Additionally present only in the removed upstream tree:

| File | SHA-256 |
|---|---|
| `agujournaltemplate.tex` | `ad077031c061593ef712fc7fb62427ccf31218239e1afdb1ce379cc863f6ec5c` |
| `agujournal2025 illustration 1 (1).png` | `e117c680d082f70eb102913d78c0abf3e7e487754b5509c69bb8cfb79c872cae` |

Retained and still an archive member under a declared unverified-redistribution
scope: `agujournal2019.cls`,
`deca12479ddeeeae31ed5873a7ebd21251858aaa1dea1b403e91a229d92b83d7`.

These values were recomputed from the working tree on 2026-08-05 and match the
table already in `docs/AGU2025_TEMPLATE_MIGRATION.md` §4 exactly.

---

## 6. What a reproducer must do to build the PDF

From an extracted archive:

1. `cd paper/agu_submission`.
2. Fetch the class from `https://github.com/AGU-Publications/agujournal2025-latex-template`.
3. Copy `agujournal2025.cls` into that same directory — beside
   `ThermoRoute_WRR.tex`, **not** onto `TEXINPUTS`. The class loads
   `./agu-logo-small.pdf` and `./agu-logo-large.pdf` by an explicit relative
   path that kpathsea cannot resolve, so `TEXINPUTS` cannot substitute for a
   local copy. For the same reason, add the other four files too if you intend to
   exercise the `published` branch; the `draft` submission build does not read
   them.
4. Verify the bytes: `sha256sum agujournal2025.cls` must print
   `a6645a79392906eb31fd63a4d3f28a2322464c0552f93c4a35da9f4a85067fab`. A
   different hash means different bytes — record it, do not ignore it.
5. Build with `build_agu.py`; `ThermoRoute_WRR.tex` is generated from
   `paper/ThermoRoute_paper.md` and must not be hand-edited.

The archived `.tex` is unchanged by this decision — step 3 is the only added
step, and it is the step AGU's own workflow already assumes.

**This path was executed, not assumed** (2026-08-05, TeX Live `pdflatex`):

| Directory state | `pdflatex ThermoRoute_WRR.tex` | Result |
|---|---|---|
| `paper/agu_submission/` as committed, after the upstream tree was deleted | exit 0 | 44-page PDF — confirms the removed tree was genuinely unused |
| same, with all five class assets removed (i.e. what an archive contains) | **exit 1** | confirms the archive alone is not compilable, which is why the README instructions are required and verifier-enforced |
| same, with **only** `agujournal2025.cls` restored (hash re-checked as `a6645a79…5067fab`) | exit 0 | 44-page PDF — confirms step 3's claim that the class alone suffices for the `draft` build |

### 6.1 What is still open (not closed by R10)

- **`agujournal2019.cls` remains an archive member.** It carries the same
  `EXCLUDE_PUBLIC` disposition, but it is pinned by `ALLOWED_PAPER_MEMBERS`,
  `scripts/make_release_archive.sh` and an exact-set assertion in
  `tests/test_manifest_release.py`, and it is declared in
  `known_minimum_unverified_redistribution_scopes` — i.e. carried *with* a
  standing warning inside a `LOCAL_EVIDENCE_ONLY` package that may not be
  transferred to anyone. Removing it touches two files outside this task's edit
  scope. It is also now **dead weight**: nothing in the build reads it. Recommend
  a follow-up that drops it from all three lists together.
- **Repository exposure is a separate question.** The five files remain committed
  in `paper/agu_submission/` on a GitHub-hosted repository, because the author
  needs them to build the submission. Whether hosting unlicensed third-party TeX
  in a readable repository is itself a redistribution problem is the pre-existing
  P1/history question in `RIGHTS_CRITICAL_PATH.md` and
  `RIGHTS_BYTE_HISTORY_AUDIT_20260801.md`. R10 closes the **archive** question
  only. Removing the directory from Git history is not closed either.
- **No third-party notice register exists.** `THIRD_PARTY_NOTICES` is still
  absent (`FAIR_RIGHTS_ACCEPTANCE_MATRIX.md` §2 row D). Exclusion is the branch
  taken precisely because the notice branch is not available.
- **PUBLIC distribution stays fail-closed.** Nothing here enables it.

---

## 7. Test results

Environment: `/home/lzq/anaconda3/envs/route-a/bin/python` (3.12), run in
`thermoroute-remediation` with `PYTHONDONTWRITEBYTECODE=1 -p no:cacheprovider`.

```bash
python -m pytest -q -p no:cacheprovider --tb=no -rf \
  tests/test_release_acceptance.py tests/test_manifest_release.py \
  tests/test_manifest_inventory.py tests/test_repro.py \
  tests/test_stage09b_completion.py tests/test_thread_cap_contract.py \
  tests/test_formal_training_entrypoints.py
```

The last three files were added because they are the remaining tests that load
`verify_release.py`.

**395 tests: 378 passed, 17 failed, 0 skipped, 0 errors.**

**All 17 failures are pre-existing and none are attributable to this change.**
Evidence:

1. All 17 are in `tests/test_manifest_release.py`, in post-opening / transport /
   checkpoint / lineage fixtures, and fail during an unrelated in-flight
   `RunIdentity` v2→v3 migration — representative traceback
   `ValueError: Git stage09_completion RunIdentity v3 is malformed`, raised
   around `verify_release.py:1393`, far from anything added here. That change is
   not fixed or worked around here.
2. A run started **before** any edit in this task produced a byte-identical
   17-name failure list.
3. The tests that actually cover the changed code paths all pass:
   `test_release_boundary_requires_contract_and_rejects_traversal` (which
   exercises `validate_members` and pins `scripts/make_release_archive.sh`'s
   paper list to `ALLOWED_PAPER_MEMBERS`),
   `test_exact_archive_layout_rejects_unregistered_files_even_with_new_manifest`,
   `test_exact_archive_layout_rejects_unregistered_empty_directory`, and
   `public_distribution_mode_is_unconditionally_blocked` — 7 selected, 7 passed.

Note: pytest exits without printing its final summary line in this environment,
on both the full run and a 7-test run, before and after this change. The counts
above were taken from the per-test progress characters, which reach `[100%]` and
whose 17 `F` marks match the 17 `FAILED` lines exactly.

Targeted behavioural checks for this change (all passing):

- each of the five excluded paths, and a path under the removed upstream tree,
  is rejected by `validate_members` with the rights message;
- the pre-existing messages are unchanged — `paper/figures/stale-result.png`
  still raises "rendered/binary manuscript artifact" and
  `paper/unregistered_notes.md` still raises "unregistered manuscript artifact";
- the real `paper/agu_submission/README.md` satisfies the retrieval-instruction
  binding, a README without the hashes fails it, and a manuscript on
  `agujournal2019` makes the check a no-op.
</content>

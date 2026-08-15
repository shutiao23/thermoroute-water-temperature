# FAIR external close-out request pack — drafts and intake only

| Field | Value |
| --- | --- |
| Date | 2026-08-02 |
| Blueprint item | P0-F — FAIR, rights, DOI and reproducible public release |
| State | **`DRAFT_ONLY / NOT_SENT / NO_EXTERNAL_ACTION_AUTHORIZED`** |
| Purpose | Narrow future correspondence and intake templates for the external evidence that P0-F still needs. |
| Non-purpose | This file is not legal advice, a rights determination, permission, a release plan, author approval, deposit authorization, DOI record, or reproduction receipt. |
| Safety boundary | No message was sent, no recipient was contacted, no archive was built, no DOI was reserved/deposited, and no remote/history/visibility state was changed while making this document. |

The present release state remains `LOCAL_EVIDENCE_ONLY`; `PUBLIC` remains
fail-closed.  A provider reply, a template email, an author form, or a proposed
reviewer assignment is evidence to be assessed later, not an automatic
`INCLUDE_PUBLIC` decision.  Absence of the specified evidence below means
`UNRESOLVED_EXCLUDE` for the affected byte(s), never inferred permission.

## 1. Reference boundary and common handling rules

This pack derives its scope from these local documents.  The hashes make the
draft's factual starting point reviewable; they do **not** license the contents
of those documents or any byte described by them.

| Local source | SHA-256 at drafting | What it contributes |
| --- | --- | --- |
| [`FAIR_RIGHTS_ACCEPTANCE_MATRIX.md`](FAIR_RIGHTS_ACCEPTANCE_MATRIX.md) | `b2dbfdab8600de338fa99118d1cee39597e1ef3171cd995d55977b22fe3106db` | P0-F acceptance rules and fail-closed branches |
| [`FAIR_SUBMISSION_READINESS_AND_TEMPLATES.md`](FAIR_SUBMISSION_READINESS_AND_TEMPLATES.md) | `b1e07b6320ed1ccaef25fa2bee3022d980259975d6c93f1310cf40baeac8965e` | author, metadata and clean-room intake requirements |
| [`RIGHTS_PROVIDER_EVIDENCE_20260801.md`](RIGHTS_PROVIDER_EVIDENCE_20260801.md) | `a5913859de0849aa14d56f4911f0a8b8e11075265f59662d80f3bef8123a1a7d` | the limited official-source facts and unresolved Daymet/AGU branches |
| [`CLEANROOM_REPRODUCTION_DESIGN.md`](CLEANROOM_REPRODUCTION_DESIGN.md) | `f176c4bd370e02ab39f88018eb43fcc7e30989bd4fef8725e0f1d0e02d7e1d61` | independent Linux reproduction design and layer boundaries |
| [`RIGHTS_BYTE_HISTORY_AUDIT_20260801.md`](RIGHTS_BYTE_HISTORY_AUDIT_20260801.md) | `5dc360390bc03d89f49af4f06d76228df00399f467d0b93ca8bf1f56671a48c5` | complete-inventory and ref/history coverage requirements |

### 1.1 Preconditions for any future use

An authorized human sender must create a dated request packet.  It must record
the intended recipient, authorized sender, recipient role/address, request ID,
draft SHA-256, every attachment's SHA-256, and the exact scope.  Do not attach
provider payloads, target outcomes, credentials, personal identity documents,
or an unreviewed Git bundle merely to make a request more persuasive.

At the time of a future send, keep a local request/response evidence register
with at least:

```text
request_id
request_draft_sha256
authorized_sender_and_role
intended_recipient_and_role
sent_at_utc                         # empty until actually sent
attachment_logical_name
attachment_sha256
attachment_byte_size
attachment_scope
response_raw_file_sha256            # empty until an answer is retained
response_received_at_utc            # empty until received
scope_match_verdict                 # pending | match | partial | no_match
next_disposition                    # include candidate | exclude | blocked
```

This register is an evidence index, not a permission manifest.  A later
qualified byte-rights reviewer must still bind a response to an exact object
and issue a byte-level disposition.

### 1.2 Anti-overbreadth rule

No draft below may be widened into a question such as “is Daymet open?”, “can
we release all data/models/predictions?”, “is the AGU template MIT?”, or “may
we publish the whole repository/history?”  Public readability, a provider-wide
statement, a DOI landing page, or silence are not a substitute for scope.

Every request must name, as applicable:

1. the exact product, version, and retrieval/service context;
2. each proposed byte class and its content SHA-256 (or, before bytes exist, a
   manifest schema and an explicit statement that no inclusion is sought yet);
3. the intended action — e.g. retain locally, publish a hash/citation,
   redistribute an exact copy, distribute a derived aggregate, or include an
   artifact that materially encodes a source value;
4. the proposed archive's immutable member-manifest SHA-256 once one exists;
5. the requested attribution/notice and every condition stated by the source;
   and
6. an explicit non-inference rule: a response outside the stated scope does not
   cover another version, object, transformation, archive, or future release.

Requests must not ask a provider, publisher, author, reviewer, depositor, or
reproducer to bless unrelated repository history, decide the status of another
rightsholder's bytes, supply an overall legal opinion, or approve a release
before its exact member inventory is known.

### 1.3 General acceptance and silence rule

For any external response to be considered later, the retained raw response
must have a SHA-256 and show a dated, identifiable source/role.  It must point
to an authoritative term or give written, scope-specific permission/exclusion
guidance that matches the exact listed product/version/bytes/use.  Conditions,
attribution, term URL/version, and limitations must be recorded verbatim or
faithfully summarized within the applicable evidence record.  A reply that is
general, conditional without the condition being met, ambiguous, or does not
match the manifest is `partial` or `no_match`, not approval.

No reply by the stated follow-up deadline, a bounced request, or uncertainty
has the same safety consequence as missing evidence: affected bytes remain
`UNRESOLVED_EXCLUDE`; the relevant close-out requirement stays `BLOCKED`.
Deadlines are administrative follow-up points, not a mechanism for deemed
consent.

## 2. ORNL DAAC / Daymet scope-specific clarification draft — do not send

| Field | Required future value |
| --- | --- |
| Request ID | `ORNL-DAYMET-<UTC-date>-<counter>` |
| State | **`DRAFT_ONLY — DO NOT SEND WITHOUT OWNER AUTHORIZATION`** |
| Intended recipient | An ORNL DAAC/Daymet contact authorized to point to governing product terms or issue scope-specific written clarification |
| Product boundary | Daymet V4 DOI `10.3334/ORNLDAAC/1840` and/or V4R1 DOI `10.3334/ORNLDAAC/2129`; choose the actually used version(s), never “Daymet generally” |
| Default pending disposition | Raw and materialized Daymet-derived bytes: `UNRESOLVED_EXCLUDE`; citation, product/version identifiers, retrieval instructions, and hashes are separate non-payload candidates pending review |

### Required attachment and hash checklist

1. A one-page factual scope cover sheet: product/version, service/retrieval
   context, desired use, request ID, and its SHA-256.
2. A Daymet object/derivative manifest.  One row per proposed object or
   homogeneous object class must give `payload_sha256`, byte size, product,
   version, retrieval date/service, path/object selector, transformation ID,
   derivative status, and proposed archive-member-manifest SHA-256.  The
   manifest itself must be SHA-256 hashed.
3. A transformation appendix, also hashed, that distinguishes: original grid
   / NetCDF copy; spatial or temporal subset; station/point extract; aggregate
   / feature table; model weight; prediction or other artifact materially
   encoding Daymet values; and non-payload citation/retrieval metadata.
4. A policy-evidence appendix containing the precise term/guide URLs and local
   snapshot SHA-256s used for the question.  It must not imply that the cited
   pages already grant the requested right.
5. A citation/attribution draft.  It must be labelled proposed and include no
   invented licence language.

The request must not include the underlying Daymet payload merely for review.
If a recipient needs a representative object to identify the scope, obtain a
separate authorization decision before transferring it; a hash, schema, and
non-payload description are the default attachments.

### Draft text

```text
Subject: Scope-specific clarification request for listed Daymet [V4/V4R1] objects — not a general licence request

Dear [authorized ORNL DAAC / Daymet contact],

We are preparing a possible future public research artifact.  We are not
asking whether Daymet is generally open, and we will not treat public access,
silence, or a general NASA/DOE statement as permission to redistribute data.

The attached, hashed scope sheet and object/derivative manifest identify only
the Daymet [product/version] objects and transformations at issue.  For each
listed category, could you please identify the authoritative terms or written
scope-specific permission, if any, that addresses:

  (a) public redistribution of the listed exact copies, if any;
  (b) public distribution of the listed subsets, point extracts, aggregates or
      feature tables, if any;
  (c) artifacts such as model weights or predictions when they materially
      encode Daymet values, if any;
  (d) applicable attribution, notice, version, upstream-source, commercial or
      other conditions; and
  (e) whether a stated condition applies to the listed product/version and
      archive use.

Please identify any limitation rather than assuming it has been satisfied.
This request does not seek an opinion about other providers, repository history,
unlisted versions, unlisted data, or a future artifact whose member manifest
does not match the attached hash.  If no applicable written term or
scope-specific permission can be identified, we will exclude the affected
materialized bytes and retain only non-payload retrieval/citation information
where separately appropriate.

Thank you,
[authorized human sender; affiliation; reply contact]
```

### Evidence accepted later; default if unanswered

An answer can advance only the listed Daymet rows when it identifies an
authoritative term or comes from a role authorized to give written,
version-specific clarification; names which categories/actions it covers; and
states conditions/attribution.  It still requires exact-byte mapping and an
independent qualified review before `INCLUDE_PUBLIC` is possible.  No response
or no scope match by the recorded follow-up date leaves all affected Daymet
payload/derivative objects `UNRESOLVED_EXCLUDE`; a proposed archive requiring
one remains `BLOCKED`.

## 3. AGU / Wiley TeX file-level permission-or-exclusion draft — do not send

| Field | Required future value |
| --- | --- |
| Request ID | `AGU-TEX-<UTC-date>-<counter>` |
| State | **`DRAFT_ONLY — DO NOT SEND WITHOUT OWNER AUTHORIZATION`** |
| Intended recipient | An AGU/Wiley contact able to identify a file-level licence or written permission for the exact named TeX asset |
| Known local asset | `paper/agu_submission/agujournal2019.cls`, SHA-256 `deca12479ddeeeae31ed5873a7ebd21251858aaa1dea1b403e91a229d92b83d7` |
| Known local assets (added 2026-08-05 by `6ee35e5`) | `agujournal2025.cls` `a6645a79392906eb31fd63a4d3f28a2322464c0552f93c4a35da9f4a85067fab`; `wiley-macros.tex` `7ea18648b7bd2c632065d4303830a54192626450139d76db39760516a16ef71c`; `tweaklist-git-moderncv-fixed.sty` `85a31337d69412566a227209ec908f3a3242ecf673819ca86564d9dc23bfbef1` (declares LPPL 1.3c — do **not** include it in a request that presumes no grant exists); `agu-logo-small.pdf` `c5c18836bc66fff737254bb0a2d321afc538e90028a364e63c0de564efc5abdf`; `agu-logo-large.pdf` `cf6382d62086d149d5538dc65c2eeac79c9e33bcc2730f70cb36c7feb324b1b5`. All under `paper/agu_submission/`; upstream `AGU-Publications/agujournal2025-latex-template` |
| Default pending disposition | `EXCLUDE_PUBLIC` for the class and any unreviewed vendored third-party TeX asset |
| Disposition already applied to the 2025 assets (R10, 2026-08-05) | Scope-sheet action **(iv)** — the archive omits these files and directs archive consumers to obtain them from AGU. Implemented and enforced in `scripts/verify_release.py`; the reproducer path was compile-tested. This request therefore remains **optional** for the 2025 assets: it is needed only if the owner later wants them *inside* an archive. See `R10_AGU2025_RELEASE_ASSETS.md`. It stays **required** for `agujournal2019.cls`, which is still an archive member |

### Required attachment and hash checklist

1. A TeX asset manifest, SHA-256 hashed, listing every vendored `.cls`, `.sty`,
   `.bst`, macro, logo, font, and other third-party TeX byte proposed for the
   archive.  For each row record exact path, content SHA-256, byte size,
   upstream/version evidence, intended archive action, and required notice.
   `agujournal2019.cls` must appear with the hash above.
2. A narrow use-scope sheet, SHA-256 hashed, separating (i) local manuscript
   preparation/submission use, (ii) redistribution of the exact file in a
   source/reproducibility archive, (iii) modification, and (iv) an archive
   that omits the file and asks users to supply a lawful external dependency.
3. The proposed public-member manifest SHA-256, if one exists; otherwise a
   statement that no public archive is yet proposed.  Do not attach the entire
   repository, a Git bundle, paper PDF, or unrelated source history.
4. A proposed `THIRD_PARTY_NOTICES` row with a SHA-256 of the draft notice.  It
   is explicitly a placeholder until a file-level evidence match is reviewed.

Do not phrase the request as whether “the AGU template” or “all Wiley files”
may be redistributed.  Each file/version/hash and each action needs its own
answer.  The fact that a class permits manuscript submission does not answer an
archive-redistribution question.

### Draft text

```text
Subject: File-level terms or permission request for one identified TeX class — possible source-archive exclusion if unavailable

Dear [authorized AGU/Wiley contact],

We use the exact local file listed below for manuscript preparation.  We are
not asking for a general statement about AGU/Wiley templates, articles, or
other files.

  path: paper/agu_submission/agujournal2019.cls
  SHA-256: deca12479ddeeeae31ed5873a7ebd21251858aaa1dea1b403e91a229d92b83d7
  local class date: 2019-04-16

For the file and actions precisely described in the attached hashed scope
sheet, please identify the applicable public licence or written permission, if
any, for: (a) redistributing this exact file in a source/reproducibility
archive; (b) any permitted modification; and (c) required copyright/attribution
or notice text.  If a term or permission is not available for this file/action,
we will exclude the file from the public archive rather than infer a right from
submission use or public readability.

The request does not cover other AGU/Wiley assets, the manuscript's publication
rights, repository history, or any byte not listed in the attached manifest.
An archive that excludes this file may instead document a user-supplied lawful
dependency, subject to separate independent build verification.

Thank you,
[authorized human sender; affiliation; reply contact]
```

### Evidence accepted later; default if unanswered

Accept only a file-level licence/permission or authoritative terms that match
the file hash/version and stated archive action, with required notices and
conditions.  A response about article copyright, template availability, or a
different file is not a match.  If no answer/term is received by the recorded
follow-up date, `agujournal2019.cls` and each unanswered TeX byte remain
`EXCLUDE_PUBLIC`.  A public package that requires an excluded class to compile
is `BLOCKED` until an independently verified, lawful user-supplied dependency
route is available.

## 4. Author / ORCID / CRediT / funding / COI intake — do not solicit yet

| Field | Required future value |
| --- | --- |
| Intake ID | `AUTHOR-ADMIN-<release-candidate-id>` |
| State | **`INTAKE_TEMPLATE_ONLY — NO AUTHOR DATA COLLECTED`** |
| Owner | Corresponding-author candidate or another explicitly authorized project administrator |
| Default pending disposition | No verified author metadata; submission/citable-release metadata remains `BLOCKED` |

Use one access-controlled, signed record per author.  Do not invent names,
affiliations, contributions, ORCIDs, funding, or conflicts from Git history,
email aliases, manuscript placeholders, or public profiles.  Do not collect
identity documents; an ORCID is included only when the author verifies it for
this use.

### Required attachments and hash binding

Before requesting attestation, provide each author only the material needed to
attest to their own record: the exact manuscript/release-candidate identifier,
the relevant draft SHA-256, and a controlled CRediT taxonomy form.  The
administrator must retain, with restricted access:

```text
author_intake_form_sha256
attested_author_record_sha256
manuscript_or_release_candidate_sha256
proposed_archive_member_manifest_sha256       # if a release is in scope
final_author_list_manifest_sha256              # only after all records reconcile
attestation_timestamp_and_locator
```

Each form must collect the fields in
[`FAIR_SUBMISSION_READINESS_AND_TEMPLATES.md`](FAIR_SUBMISSION_READINESS_AND_TEMPLATES.md)
§2: publishing/citation name, affiliation and country, email, corresponding
author status, verified ORCID if supplied, each applicable CRediT role,
funding/award IDs, competing-interest declaration, employer/copyright
constraints, attestation time, and signature/attestation locator.  Project
administration additionally requires the funding statement, COI statement,
acknowledgments, repository owner, intended DOI depositor, data-rights reviewer,
and corresponding-author approval of final manuscript/release bytes.

### Narrow attestation text

```text
I attest only to the author-administration fields that I completed for the
identified manuscript/release candidate [identifier and SHA-256].  I have
confirmed my name, affiliation, correspondence status, supplied ORCID (if any),
CRediT roles, funding information, competing-interest statement, and disclosed
employer/copyright constraints.  I have not been asked to attest for another
author, to grant rights in third-party or provider data, or to approve an
archive whose final member manifest differs from the one identified here.
```

The form must not ask an author to attest that “all data are open,” that all
coauthors agree, or that an unfinalized archive/publication is ready.  A blank
or unsigned form, a non-verified ORCID, a conflict/funding item awaiting
clarification, or a mismatch to the final SHA keeps the author record pending.
No response by the agreed administrative deadline means no author metadata is
used for submission/deposit; affected author administration and P0-F remain
`BLOCKED`.

## 5. Qualified byte-rights reviewer assignment form — do not assign yet

| Field | Required future value |
| --- | --- |
| Assignment ID | `BYTE-RIGHTS-REVIEW-<inventory-id>` |
| State | **`ASSIGNMENT_TEMPLATE_ONLY — NO REVIEWER APPOINTED`** |
| Assigning authority | Project owner or other explicitly authorized custodian |
| Scope unit | One content-deduplicated SHA-256 object plus every observed ref/path/reachability occurrence |
| Default pending disposition | `UNRESOLVED_EXCLUDE` for every object without a completed evidence-backed review |

The assignment must identify the reviewer, organizational role, relevant
qualifications for the byte classes under review, independence/conflict
disclosure, and authority to stop inclusion.  It must not ask a reviewer to
give a blanket provider-wide or repository-wide approval, remediate history, or
make an unsupported legal conclusion.

### Required assignment packet and hashes

1. Complete inventory and ref snapshots, generated in a stationary low-I/O
   window, with a SHA-256 for the inventory, `git for-each-ref` snapshot, and
   `git ls-remote --refs origin` snapshot.  It must cover direct refs,
   reachable history, and separately enumerated local-unreachable roots as
   required by the history audit.
2. Proposed public-member manifest SHA-256 and a negative list showing that no
   non-`INCLUDE_PUBLIC` object is a member.  No review may treat a path prefix
   or repository MIT header as an object decision.
3. Provider/third-party evidence packet SHA-256s, correspondence response
   SHA-256s if any, required-attribution/notice drafts, and derived-data
   component/provenance closure manifest SHA-256.
4. The assignment form SHA-256, reviewer qualification/conflict attestation
   SHA-256, and a decision-manifest schema/version SHA-256.

The reviewer must return a signed scope acceptance and an evidence-backed
decision manifest.  Each object record must preserve at least provider/product/
version/retrieval context, payload SHA-256, every Git object/path observation,
policy/correspondence evidence and its SHA-256, attribution, reviewer, date,
decision scope, and exactly one disposition:
`INCLUDE_PUBLIC`, `EXCLUDE_PUBLIC`, or `UNRESOLVED_EXCLUDE`.

### Acceptance and silence rule

The assignment is accepted only when qualifications and conflicts are recorded,
the reviewer accepts the exact inventory/version, and the completed manifest is
internally consistent, content-deduplicated, evidence-linked, and hash-bound to
the candidate member list.  The reviewer’s record still does not turn a
nonmatching provider response into permission.  If no qualified reviewer
accepts/finishes by the agreed date, or any object is missing/ambiguous,
unreviewed objects remain `UNRESOLVED_EXCLUDE`; the public archive and P0-F stay
`BLOCKED`.

## 6. DOI depositor and immutable-release intake — do not initiate a deposit

| Field | Required future value |
| --- | --- |
| Intake ID | `DOI-DEPOSITOR-<release-candidate-id>` |
| State | **`INTAKE_TEMPLATE_ONLY — NO DOI RESERVED OR DEPOSIT INITIATED`** |
| Owner | A named, authorized repository/deposit custodian, after author and rights gates close |
| Default pending disposition | No citable release or DOI; P0-F remains `BLOCKED` |

This intake chooses an authorized custodian and a deposit workflow only.  It
does not ask a service or an individual to make an unreviewed archive public,
does not create metadata with placeholders, and does not reserve a DOI from an
unfinalized member list.

### Required intake attachments and hashes

All of the following must exist and agree before the depositor may accept the
intake:

1. final public archive SHA-256 and byte size, plus complete member manifest
   SHA-256;
2. reviewed byte-rights decision manifest, third-party-notice register, and
   derived-data provenance-rights closure receipt, each SHA-256 bound to that
   member manifest;
3. signed/reconciled author-administration manifest SHA-256, including final
   corresponding-author confirmation;
4. final `CITATION.cff`, `codemeta.json`, and deposit-metadata file SHA-256s,
   agreeing on creators, version, date, scope, licences, keywords and DOI state;
5. independent clean-host reproduction receipt SHA-256 and the relevant fixed
   Linux image/source/dependency identities; and
6. custodian authorization/role attestation SHA-256, with the selected service,
   account/organizational authority, versioning policy, and post-deposit
   verification owner.  Do not record service credentials in this packet.

### Narrow custodian attestation

```text
I am authorized to deposit only the release candidate identified by the listed
archive and member-manifest SHA-256 values.  I will not substitute, add, remove,
or make public any byte after the final review without a new review and hashes.
I will record the service receipt, DOI/landing-page/version identifiers, access
time, and resolved deposited-archive hash for independent verification.
```

The depositor must not be asked to attest that all repository history is clean,
that a provider granted rights it did not state, or that a DOI itself proves
archive review.  If no authorized custodian accepts, any prerequisite hash is
absent/mismatched, or deposit verification does not bind the DOI to the reviewed
archive, no deposit occurs and P0-F remains `BLOCKED`.  Silence is never
authorization to deposit.

## 7. Independent Linux reproducer intake — do not commission a run yet

| Field | Required future value |
| --- | --- |
| Intake ID | `INDEPENDENT-LINUX-REPRO-<release-candidate-id>` |
| State | **`INTAKE_TEMPLATE_ONLY — NO INDEPENDENT RUN COMMISSIONED`** |
| Independence | Operator and host must be genuinely independent of the build host/custodian; document the relationship and conflicts |
| Default pending disposition | No independent clean-room receipt; reproduction and P0-F remain `BLOCKED` |

The intake must select one declared layer: (A) public synthetic smoke, (B)
validated local-evidence replay, or (C) receipt-driven paper/PDF/SI rendering.
It must not call a synthetic run an empirical Route-A/Route-B replay, and it
must not give an operator provider/target data, live experiment outputs, or
credentials merely to make a run succeed.

### Required intake attachments and hashes

1. exact reviewed source/public archive SHA-256 and complete member manifest
   SHA-256;
2. fixed Linux base-image immutable digest, CPU architecture, OS/libc, Python
   3.12 patch version, BLAS/OpenMP/Torch build identities, lockfile SHA-256, and
   declared wheel/source policy;
3. top-level command and documentation SHA-256, declared layer, input manifest
   SHA-256, output schema, expected artifact hashes or numerical tolerances, and
   an explicit no-network-after-staging statement where promised;
4. clean build-root rules: an explicit empty non-symlinked root, no writes to
   the repository’s live `outputs/**`, and no `--system-site-packages` shortcut;
5. a reproducer task form SHA-256 recording independent operator/host, conflict
   disclosure, scheduled window, and a safe channel for returning logs/receipt
   hashes; and
6. for paper rendering only, validated input receipt SHA-256s and PDF/SI QA
   acceptance criteria.  Do not substitute draft prose or placeholder results.

### Required return evidence and default if no run

An acceptable independent receipt records the input and environment identities,
commands, exit codes, network condition, `pip check`, installation/test result,
output hashes or declared-tolerance comparison, and any deviation/failure.  The
operator must retain the raw log/receipt with its SHA-256.  A same-host run,
an environment with undeclared dependencies, a run that reads live experiment
namespaces, or a result outside the declared layer is not independent clean-room
evidence.

If no qualified independent operator accepts or completes the run by the
recorded follow-up date, if the result is not reproducible within its declared
tolerance, or if required receipts are absent, there is no clean-room proof and
P0-F remains `BLOCKED`.  The fallback does not relax the public archive guard.

## 8. Close-out tracker

All rows are intentionally open.  This pack creates no external evidence.

| Workstream | Current state | Evidence still required before it can close | Silence / timeout default |
| --- | --- | --- | --- |
| ORNL/Daymet | `NOT_SENT` | scope-matching term/clarification; exact-object mapping; qualified decision manifest | affected Daymet bytes `UNRESOLVED_EXCLUDE`; package requiring them `BLOCKED` |
| AGU/Wiley TeX | `NOT_SENT` | file-level terms/permission or exclusion plus independently verified external-dependency path where needed | exact asset `EXCLUDE_PUBLIC`; dependent public compilable bundle `BLOCKED` |
| Authors/ORCID/CRediT/funding/COI | `NOT_COLLECTED` | signed, reconciled individual records bound to final candidate hashes | submission/citable metadata `BLOCKED` |
| Byte-rights reviewer | `NOT_ASSIGNED` | qualified/conflict-cleared reviewer and complete hash-bound object manifest | unreviewed objects `UNRESOLVED_EXCLUDE`; PUBLIC `BLOCKED` |
| DOI depositor | `NOT_IDENTIFIED` | authorized custodian and all final archive/metadata/reproduction inputs | no reservation/deposit; P0-F `BLOCKED` |
| Independent Linux reproducer | `NOT_COMMISSIONED` | independent operator receipt matching declared layer and identities | clean-room/P0-F `BLOCKED` |

Nothing in this document authorizes a send, a public archive, a deposit, a DOI,
a release metadata file, a remote change, or an inference that unresolved data
or TeX bytes may be redistributed.

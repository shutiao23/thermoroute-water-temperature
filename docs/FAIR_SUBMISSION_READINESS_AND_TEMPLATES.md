# FAIR and submission administration readiness

| Field | Value |
| --- | --- |
| Date | 2026-08-01 |
| Public release | **BLOCKED** |
| Allowed distribution state | `LOCAL_EVIDENCE_ONLY` under the existing fail-closed contract |
| Status | Readiness record and intake templates; not release metadata or rights approval |

## 1. Current readiness matrix

| Item | State | Consequence |
| --- | --- | --- |
| Software license | MIT exists for software | Does not license data, service responses, manuscripts or AGU class |
| Data/code availability prose | PRE incomplete statement exists | DOI, public URL, data license and final scope remain absent |
| Verified authors/affiliations | Missing | Submission and citable metadata blocked |
| Corresponding author / ORCID | Missing | Verified corresponding-author details are required for the AGU package; ORCID requirements follow the selected journal, while repository/deposit ORCIDs may be optional but must be verified if supplied |
| CRediT contributions | Missing | Contribution statement blocked |
| Funding / COI | Missing | Journal declarations blocked |
| `CITATION.cff` | Missing intentionally | Do not create with placeholder creators/DOI |
| `codemeta.json` | Missing intentionally | Do not create with unverified scope/license/creators |
| `.zenodo.json` | Current-tree file absent; an older version remains history-reachable | Existing governance records say its old openness/license claims must not be reused; this audit is not an external withdrawal notice |
| `THIRD_PARTY_NOTICES` | Missing | AGU class/template and other bundled third-party bytes unresolved |
| Machine-readable data dictionary | Missing | Parquet schema and prose do not close units/provenance metadata |
| DOI/citable release | Missing | Package version `1.0.0` is not a deposited release |
| PUBLIC archive | Correctly fails closed | Keep blocked until byte-scoped rights decisions pass |
| Minimal reproducer | Synthetic smoke exists, public package does not | Requires fixed image, archive contract and independent validation |
| POST renderer / PDF-SI E2E | Missing | Final paper is not reproducible from receipts |

## 2. Author administration intake

One signed row is required per author. Unverified fields remain empty; do not use
invented placeholders in a submission build.

```text
legal_or_publishing_name
preferred_citation_name
affiliation_name
affiliation_address
country
orcid
email
corresponding_author              # true | false
CRediT_conceptualization
CRediT_methodology
CRediT_software
CRediT_validation
CRediT_formal_analysis
CRediT_investigation
CRediT_data_curation
CRediT_writing_original_draft
CRediT_writing_review_editing
CRediT_visualization
CRediT_supervision
CRediT_project_administration
CRediT_funding_acquisition
funding_award_ids
competing_interest_declaration
copyright_or_employer_constraints
verified_by_author_at
signature_or_attestation_locator
```

Project-level intake must also record funding statement, COI statement, acknowledgments,
repository owner, DOI depositor, data-rights reviewer and corresponding-author
approval of the final manuscript/release bytes.

## 3. Data-license matrix template

Initial status for every class is `UNREVIEWED`. Public inclusion is false until an
identified reviewer records an evidence-backed decision.

```text
inventory_key
byte_class
path_or_object_selector
sha256
provider_or_rightsholder
source_terms_url
terms_snapshot_sha256
required_attribution
derivative_status
repackaging_allowed              # yes | no | conditional | unknown
relicensing_allowed              # yes | no | conditional | unknown
privacy_or_sensitive_content
reviewer
reviewed_at
decision                         # include | exclude | retrieval-script-only | unresolved
public_archive_inclusion         # false until approved
evidence_locator
```

Minimum classes include USGS/NWIS response bytes/metadata, Daymet, gridMET,
derived panel/registries, three legacy CSV files, history-only paper extracts,
model outputs, manuscript assets, AGU files and redistributed dependency binaries.

## 4. Third-party notice register

```text
component
exact_paths
sha256s
version
upstream_project
license_or_terms_evidence
copyright_notice
required_notice_text
archive_decision                 # include | exclude | unresolved
reviewer
reviewed_at
```

AGU class/template and `trackchanges.sty` require an evidence-backed include or
exclude decision. Absence from the eventual public archive is an acceptable
closure if the paper can be built through a legal user-supplied dependency.

## 5. Public minimal-reproducer contract

A public minimal reproducer must:

- contain only reviewed software and synthetic fixtures;
- perform no network call and acquire no provider/target bytes;
- write only to an explicit clean build root;
- install and validate inside the fixed Linux image;
- run tests, a tiny deterministic train/predict path and schema/receipt checks;
- state prominently that it reproduces pipeline mechanics, not Route-A/Route-B
  empirical results;
- contain no model checkpoint or cache derived from unreviewed data;
- generate a self-hashed reproducer receipt and a complete archive-member SHA
  inventory.

The local evidence reproducer is a separate non-public layer and must validate
rights/scope before accepting a bundle.

## 6. Release-metadata readiness gates

Create real `CITATION.cff`, `codemeta.json` and deposit metadata only after:

1. verified creators and affiliations are supplied; journal-required ORCIDs are
   present, and any optional repository/deposit ORCID is verified before use;
2. software/data/third-party license scopes are decided;
3. the exact public archive member list and SHA-256 inventory pass review;
4. release version, title, description, keywords and repository URL are final;
5. DOI reservation/deposit workflow and custodian are chosen;
6. every metadata file agrees on creators, version, date, license, DOI and scope;
7. the public archive independently reproduces the claimed synthetic/minimal
   workflow;
8. final author and rights-reviewer approvals are recorded.

`CITATION.cff` should cite software; a separately deposited dataset receives its
own DOI and data license. Do not imply that one MIT software citation licenses all
empirical bytes.

## 7. P0-F completion gate

P0-F is complete only when every public byte has an inventory identity and rights
decision, release metadata are mutually consistent, the public archive validates
on a clean host, final author declarations are signed, and the DOI resolves to the
reviewed archive. Until then, PUBLIC generation remains fail-closed and no remote,
history, deposit or visibility mutation is authorized by this document.

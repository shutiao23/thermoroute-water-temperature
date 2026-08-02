# Provider rights evidence — official-source snapshot

| Field | Value |
| --- | --- |
| Snapshot | 2026-08-01 |
| Status | **EVIDENCE COLLECTED / NOT LEGAL ADVICE / PUBLIC RELEASE STILL BLOCKED** |
| Scope | USGS/NWIS, Daymet, gridMET and AGU template bytes present or represented in repository history |
| Safety | Official pages and local file headers inspected read-only; no remote, history, data, output or Stage-09 mutation |

This note narrows provider-specific questions. It does not substitute a qualified
rights review, and a general provider policy is not automatically an exact-byte
license for every response stored in this repository.

## 1. Evidence matrix

| Byte class | Official evidence | What the evidence supports | Remaining release gate |
| --- | --- | --- | --- |
| USGS-authored NWIS data and information | [USGS copyright FAQ](https://www.usgs.gov/faqs/are-usgs-reportspublications-copyrighted); [USGS data licensing guidance](https://www.usgs.gov/data-management/data-licensing) | USGS states that USGS-authored or produced data/information are public domain in the United States and asks users to credit USGS. Its data guidance explains CC0 use and foreign-copyright considerations. | Classify each stored response as USGS-produced rather than partner/copyrighted material, preserve series/provider provenance and attribution, and record the exact policy snapshot used. USGS itself warns that some hosted material is third-party. |
| Daymet V4/V4R1 products and ORNL subset responses | [Daymet V4 guide](https://daac.ornl.gov/DAYMET/guides/Daymet_Daily_V4.html); [Daymet V4R1 guide](https://daac.ornl.gov/DAYMET/guides/Daymet_Daily_V4R1.html); [V4R1 NASA CMR record](https://cmr.earthdata.nasa.gov/search/concepts/C2532426483-ORNL_CLOUD.umm_json); [NASA Earthdata data-use guidance](https://www.earthdata.nasa.gov/engage/open-data-services-software/data-use-policy) | The guides fix V4 DOI `10.3334/ORNLDAAC/1840`, V4R1 DOI `10.3334/ORNLDAAC/2129`, citations, versions and ORNL DAAC distribution. NASA distinguishes NASA-led mission data from non-NASA data subject to the sponsoring organization's terms. | Neither guide nor the inspected CMR metadata supplies an explicit product-level redistribution licence. Do not infer `CC0` from NASA distribution. Until ORNL/Daymet gives written scope-specific confirmation, exclude original NetCDF, subsets, aggregates and tables that materially encode Daymet values from a PUBLIC archive; retain only retrieval instructions, version/DOI, hashes and required citation. |
| gridMET data and subset responses | [Climatology Lab gridMET page](https://www.climatologylab.org/gridmet.html) | The publisher states that John Abatzoglou waived, to the extent possible, copyright and related rights in gridMET and describes it as free of known copyright restrictions. The same page gives citation, acquisition and revision context. | Bind the statement and access date to the exact gridMET product/version and response bytes. Preserve attribution and upstream provenance; separately review whether stored service wrappers or non-gridMET ancillary content are included. This lowers risk but is not a blanket approval for every raw snapshot. |
| `agujournal2019.cls` | [AGU manuscript-preparation page](https://www.agu.org/Publications/Authors/Journals); [official 2019 LaTeX guide](https://www.agu.org/Publications/Authors/-/media/E070CC3AFBE249C680098DCDEBFDCD6A.ashx) | AGU distributes and recommends the template for manuscript preparation and identifies `agujournal2019.cls` as its primary class. The local byte has SHA-256 `deca12479ddeeeae31ed5873a7ebd21251858aaa1dea1b403e91a229d92b83d7`. | The official pages and local class header supply no public-archive redistribution/modification licence. Submission use and redistribution are distinct. Exclude the class and any bundle requiring it from a PUBLIC archive unless AGU/Wiley provides a file-scoped licence or written permission. |
| Current AGU 2025 template | [AGU's linked template repository](https://github.com/AGU-Publications/agujournal2025-latex-template) at snapshot commit `355052226d872cf6b9211c12b73b2ed2da133a7d` | Confirms AGU now points authors to a newer official template and that the repository is publicly readable. | No root `LICENSE`, tag or release was observed; `agujournal2025.cls` and `wiley-macros.tex` contain no source-licence grant. Public readability and accepting pull requests are not redistribution rights. Updating templates would also not cure the 2019-byte history exposure. |
| Legacy `data/b1.csv`, `data/p3.csv`, `data/s2.csv` | No authoritative provider record found in the repository | Nothing beyond local existence and Git exposure. | Owner must identify source, collector/rightsholder, collection terms and redistribution authority. Until then these bytes remain excluded from every PUBLIC package. |

## 2. Exact-byte decision rule

For each content-deduplicated SHA-256 object, the future rights manifest must
record:

```text
provider
product
product_version
retrieval_service
retrieved_at_utc
payload_sha256
all_git_blob_ids_and_paths
policy_url
policy_snapshot_sha256
required_attribution
third_party_content_possible
reviewer
decision
decision_scope
decision_date_utc
```

Allowed decisions are `INCLUDE_PUBLIC`, `EXCLUDE_PUBLIC`, and
`UNRESOLVED_EXCLUDE`. Absence of evidence maps to `UNRESOLVED_EXCLUDE`; it never
maps to inclusion. A policy must be evaluated against the actual product and
retrieval date, not merely the provider's name.

## 3. Current disposition

- USGS and gridMET now have useful official evidence, but exact-object mapping and
  qualified review remain outstanding.
- Daymet needs a dataset-specific redistribution statement or written ORNL DAAC
  confirmation before repackaged provider responses are included.
- The AGU class and legacy three-site CSV files remain `UNRESOLVED_EXCLUDE`.
- Derived panels combine multiple sources and cannot inherit the most permissive
  upstream status automatically.
- The PUBLIC builder must remain fail-closed. A safe public fallback contains
  reviewed source code, synthetic fixtures, retrieval instructions and metadata,
  while excluding unresolved provider and history bytes.

No archive creation, deposit, DOI issuance, remote visibility change or history
rewrite is authorized by this evidence snapshot.

## 4. Daymet upstream and scope qualification

The Daymet guides identify GHCN-Daily as a core station input. The
[GHCN-Daily documentation](https://www.ncei.noaa.gov/pub/data/cdo/documentation/GHCND_documentation.pdf)
contains source-specific restrictions for some non-U.S./WMO material. This does
not prove that those restrictions attach to every Daymet gridded derivative, but
it makes a blanket public-domain inference unsafe and creates a question for the
Daymet rightsholder. V4R1 also uses ECCC/CCCS inputs for its 2024--2025 update;
the [ECCC open-data licence](https://eccc-msc.github.io/open-data/licence/readme_en/)
has attribution, non-endorsement and third-party-rights qualifications and does
not itself license the combined Daymet product.

The current fail-closed decision is therefore
`OWNER_LEGAL_CONFIRMATION_REQUIRED / EXCLUDE_FROM_PUBLIC_ARCHIVE`. A written
ORNL/Daymet answer should separately cover original grids, spatial/temporal
subsets, station extracts, aggregates/features, model weights and predictions,
commercial/non-commercial use and whether any upstream source condition carries
through. Provider access controls and the absence of an explicit restriction are
not affirmative redistribution grants. DOE's
[digital research data guidance](https://www.energy.gov/datamanagement/doe-requirements-and-guidance-digital-research-data-management)
likewise distinguishes public access from the ownership and licence terms that
must be documented for a specific research-data product.

## 5. AGU template file-level qualification

The local 2019 class header identifies an AGU class dated 2019-04-16 but contains
no `license`, `permission` or redistribution grant. The 2019 guide establishes
the intended manuscript-preparation/submission use only. AGU/Wiley open-access
and copyright pages concern accepted/published articles and do not retroactively
license template source code.

At the inspected 2025 repository commit, the class and `wiley-macros.tex` remain
unlicensed for downstream redistribution. Commented example language about a
published article's CC BY-NC-ND status is not a template licence. One separate
repository file, `tweaklist-git-moderncv-fixed.sty`, declares LPPL 1.3c, but it is
not present in this project and its licence cannot be projected onto other
files. The current project contains no vendored `.sty` or `.bst` file at the
top level of `paper/agu_submission/`; any future TeX dependency must be reviewed
per actual byte.

Current dispositions are:

- `agujournal2019.cls`: `EXCLUDE_PUBLIC`;
- an otherwise author-owned TeX source bundle that requires that class:
  `EXCLUDE_AS_COMPILABLE_BUNDLE` unless the class is omitted/replaced or
  permission is obtained;
- AGU 2025 class/macros/logos: `EXCLUDE_PUBLIC` unless a file-level licence or
  written permission appears;
- third-party TeX dependencies: `NOT_PRESENT / REVIEW_IF_VENDORED`.

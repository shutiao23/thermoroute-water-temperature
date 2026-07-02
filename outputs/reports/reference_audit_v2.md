# Reference audit report — v2 (web-verified, adversarial)

> **STATUS 2026-07-01: ALL 9 must-fix + 4 minor APPLIED** to `paper/ThermoRoute_paper.md` and re-rendered to PDF/DOCX. Line 572 resolved to Weierbach et al. 2022 (Water 14, 1032). This report is retained as the evidence trail.

> **UPDATE 2026-07-01: reference count is now 41.** Three foundational hydrological-uncertainty references were added after the audit — Beven & Binley 1992 (GLUE) and Kavetski et al. 2006 Theory + Application (BATEA) — all web-verified against Crossref. A machine-readable `paper/references.bib` (41 entries, parses cleanly in pandoc/biblatex) is now the single source of truth for citations.


Date: 2026-07-01
Auditor: 6 parallel verification agents, 102 web calls (Crossref REST API,
DataCite REST API, publisher landing pages, arXiv, DBLP, official USGS/ORNL
citation pages), letter-by-letter comparison of authors/order, year, title,
venue, volume, pages, DOI string.

## Scope

**References live in exactly one source file: `paper/ThermoRoute_paper.md`,
`## References`, lines 554–632 (38 entries).** No `.bib`/`.ris`/`.enw` file
exists. `paper/ThermoRoute_paper.pdf` and `.docx` are rendered from the
markdown, so they inherit every error until the markdown is fixed and
re-rendered. `cover_letter.md`, `highlights.md`, `README.md` contain no
references.

This audit also adjudicates the author's first-round audit
(`outputs/reports/reference_audit.md`, dated 2026-07-01, using older line
numbers 516–600 that predate subsequent manuscript edits).

## Verdict summary

| verdict | count | lines |
|---|---|---|
| **must-fix** | **9** | 558, 562, 572, 584, 614, 618, 620, 624, 632 |
| minor (recommended) | 4 | 586, 602, 622, 628 |
| ok (verified perfect) | 25 | all others |

## Adjudication of the author's first-round audit

* All **7 must-fix findings confirmed** independently (their L526→558, L530→562,
  L540→572, L582→614, L588→620, L592→624, L600→632). Every corrected author
  list / DOI they proposed matched the registered metadata exactly.
* **2 additional must-fix found that the first round missed or misjudged:**
  * **Line 584 (Romano CQR)** — first round said "OK, NeurIPS metadata confirms
    pages 3543–3553". Wrong: the official proceedings.neurips.cc page carries
    **no pagination**, and DBLP (derived from the printed Curran volume) gives
    **3538–3548**. The current page range is at best unverifiable, at worst
    wrong.
  * **Line 618 (dataRetrieval R)** — first round graded it "recommended";
    upgraded to **must-fix** because the official v2.7.17 `inst/CITATION` file
    and DataCite both order authors **De Cicco, Hirsch, Lorenz, Watkins,
    Johnson** (manuscript swaps Hirsch/Lorenz), the initials should be
    "Watkins, W. D.", and the version string "v.2.7.17" is part of the official
    citation.
* **1 minor the first round missed:** line 628 (PyTorch) — pages 8026–8037 are
  the ACM-DL variant; DBLP/printed proceedings give 8024–8035 and the official
  NeurIPS BibTeX has no pages.

## MUST-FIX entries (9)

### 1. Line 558 — Zwart et al. 2023, Frontiers in Water [F3]
**Error.** Author list materially wrong: manuscript lists 8 authors, two of
whom (**"Diaz-Gonzalez, M." and "Bertassello, L. E."**) are **not authors of
this paper**; 7 real authors missing; order wrong. Title/venue/volume/DOI
correct.
**Corrected:**
> Zwart, J. A., Diaz, J., Hamshaw, S., Oliver, S., Ross, J. C., Sleckman, M., Appling, A. P., Corson-Dosch, H., Jia, X., Read, J., Sadler, J., Thompson, T., Watkins, D., White, E., 2023. Evaluating deep learning architecture and data assimilation for improving water temperature forecasts at unmonitored locations. Frontiers in Water 5, 1184992. https://doi.org/10.3389/frwa.2023.1184992.

Evidence: https://api.crossref.org/works/10.3389/frwa.2023.1184992

### 2. Line 562 — Luo et al. 2025, ACM SIGSPATIAL [F5]
**Error.** Authors wrong: real list is **Luo, Yu, Chen, Fan, Xie, Li, Jia**
(7 authors). Manuscript drops Fan/Xie/Li, misplaces Jia, and **adds "Kumar,
V." who is not an author**. Pages 124–136 missing. Official proceedings title
omits "SIGSPATIAL" (cosmetic).
**Corrected:**
> Luo, S., Yu, R., Chen, S., Fan, Y., Xie, Y., Li, Y., Jia, X., 2025. Geo-aware models for stream temperature prediction across different spatial regions and scales, in: Proceedings of the 33rd ACM International Conference on Advances in Geographic Information Systems (SIGSPATIAL '25). ACM, pp. 124–136. https://doi.org/10.1145/3748636.3762716 (preprint: arXiv:2510.09500).

Evidence: https://api.crossref.org/works/10.1145/3748636.3762716 ; https://arxiv.org/abs/2510.09500

### 3. Line 572 — "Topp et al. 2023" [F11] — TWO PAPERS CONFLATED
**Error.** The DOI (10.1029/2022WR033880) belongs to **Topp et al. 2023,
"Stream Temperature Prediction in a Shifting Environment: Explaining the
Influence of Deep Learning Architecture", WRR 59** (authors: Topp, Barclay,
Diaz, Sun, Jia, Lu, Sadler, Appling). The **title** printed in the entry
belongs to **Weierbach et al. 2022, Water 14, 1032** (DOI 10.3390/w14071032).
The author list printed matches **neither** paper. The [F11] label is not
cited in the body text, so either replacement is safe.
**Option A (keep the DOI's paper):**
> Topp, S. N., Barclay, J., Diaz, J., Sun, A. Y., Jia, X., Lu, D., Sadler, J. M., Appling, A. P., 2023. Stream temperature prediction in a shifting environment: explaining the influence of deep learning architecture. Water Resources Research 59, e2022WR033880. https://doi.org/10.1029/2022WR033880.

**Option B (keep the title's paper):**
> Weierbach, H., Lima, A. R., Willard, J. D., Hendrix, V. C., Christianson, D. S., Lubich, M., Varadharajan, C., 2022. Stream temperature predictions for river basin management in the Pacific Northwest and Mid-Atlantic regions using machine learning. Water 14, 1032. https://doi.org/10.3390/w14071032.

Evidence: https://api.crossref.org/works/10.1029/2022WR033880 ; https://api.crossref.org/works/10.3390/w14071032

### 4. Line 584 — Romano et al. 2019 (CQR) — pages unverifiable/wrong
**Error.** Cited pages 3543–3553 are not confirmed by any official source: the
NeurIPS proceedings page has no pagination; DBLP (printed Curran volume) gives
**3538–3548**. Recommended: use DBLP's range or drop pages (official NeurIPS
BibTeX carries none). Authors/title/venue/year correct.
**Corrected (safest):**
> Romano, Y., Patterson, E., Candès, E. J., 2019. Conformalized quantile regression, in: Advances in Neural Information Processing Systems 32 (NeurIPS 2019), pp. 3538–3548.

Evidence: https://dblp.org/rec/conf/nips/RomanoPC19.html

### 5. Line 614 — Gupta et al. 2009 — DEAD DOI (typo)
**Error.** `10.1016/j.jhydro.2009.08.003` returns 404 on Crossref ("jhydro" is
a typo for "jhydrol"). All other fields correct.
**Corrected DOI only:** `https://doi.org/10.1016/j.jhydrol.2009.08.003`
Evidence: https://api.crossref.org/works/10.1016/j.jhydrol.2009.08.003 (resolves); the typo'd DOI 404s.

### 6. Line 618 — dataRetrieval (R) [D1] — author order + version
**Error.** Official v2.7.17 CITATION and DataCite order authors **De Cicco,
Hirsch, Lorenz, Watkins, Johnson** (manuscript swaps Hirsch/Lorenz); initials
"Watkins, W. D."; version "v.2.7.17" is part of the official citation.
**Corrected:**
> De Cicco, L. A., Hirsch, R. M., Lorenz, D., Watkins, W. D., Johnson, M., 2024. dataRetrieval: R packages for discovering and retrieving water data available from federal hydrologic web services, v.2.7.17. U.S. Geological Survey. https://doi.org/10.5066/P9X4L3GE.

Evidence: https://raw.githubusercontent.com/DOI-USGS/dataRetrieval/v2.7.17/inst/CITATION ; https://api.datacite.org/dois/10.5066/P9X4L3GE

### 7. Line 620 — dataretrieval (Python) — FABRICATED CO-AUTHOR
**Error.** "Decker, J. K." **does not exist on this work**. Official USGS
citation: **Hodson, Hariharan, Black, Horsburgh, 2023**.
**Corrected:**
> Hodson, T. O., Hariharan, J. A., Black, S., Horsburgh, J. S., 2023. dataretrieval (Python): a Python package for discovering and retrieving water data available from U.S. federal hydrologic web services. U.S. Geological Survey software release. https://doi.org/10.5066/P94I5TX3.

Evidence: https://api.datacite.org/dois/10.5066/P94I5TX3 ; https://github.com/DOI-USGS/dataretrieval-python

### 8. Line 624 — Daymet V4 R1 [D2] — wrong author list/order
**Error.** DataCite lists creators exactly: **Thornton, M. M.; Shrestha, R.;
Wei, Y.; Thornton, P. E.; Kao, S.-C.** Manuscript puts P. E. Thornton first,
demotes M. M. Thornton, misplaces Wei, and **adds "Wilson, B. E." who is not
in the official citation**.
**Corrected:**
> Thornton, M. M., Shrestha, R., Wei, Y., Thornton, P. E., Kao, S.-C., 2022. Daymet: daily surface weather data on a 1-km grid for North America, version 4 R1. ORNL DAAC, Oak Ridge, Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/2129.

Evidence: https://api.datacite.org/dois/10.3334/ORNLDAAC/2129

### 9. Line 632 — SciPy — DEAD DOI
**Error.** `10.1038/s41592-019-0686-5` returns 404. Correct DOI:
**`10.1038/s41592-019-0686-2`**. Author list otherwise verified correct.
**Corrected DOI only:** `https://doi.org/10.1038/s41592-019-0686-2`
Evidence: https://api.crossref.org/works/10.1038/s41592-019-0686-2

## MINOR entries (4, recommended)

| line | entry | recommendation |
|---|---|---|
| 586 | Vovk et al. 2005 book | add DOI `https://doi.org/10.1007/b106715` for consistency |
| 602 | Wilks 2011 | fine as-is ("Oxford" defensible; ISBN 978-0-12-385022-5 confirmed); optionally add "International Geophysics Series, vol. 100" and move the explanatory parenthetical to a footnote |
| 622 | USGS NWIS | add dataset DOI `https://doi.org/10.5066/F7P55KJN` + access date (USGS-recommended form; Elsevier data-citation style expects it) |
| 628 | PyTorch NeurIPS | pages 8026–8037 are the ACM-DL variant; DBLP gives 8024–8035; official BibTeX has none — recommend 8024–8035 or drop pages |

## Verified OK (25)

Lines 554, 556, 560, 564, 566, 568, 570, 574, 576, 578, 580, 588, 590, 592,
594, 596, 598, 600, 604, 606, 608, 610, 612, 626, 630 — every field
(authors/order, year, title, venue, volume, pages, DOI) matches the registered
metadata. Notables re-confirmed: Jia et al. SDM published title uses "Model"
(arXiv uses "Networks") — manuscript correctly follows the published version;
Rahmani ERL 2021 year follows IOP's own recommended citation; the
parenthesised DOIs (Harvey 1997, Nash–Sutcliffe 1970) resolve correctly.

## Pattern note (for the record)

The three worst errors (fabricated co-authors "Diaz-Gonzalez/Bertassello" on
F3, "Kumar" on F5, "Decker" on the Python package, "Wilson" on Daymet) are
name-transplants from *adjacent, related* works — a classic
LLM-style/memory-style confusion between co-author networks. Every reference
in future revisions should be generated from the registered metadata (Crossref
/DataCite lookup), never from memory.

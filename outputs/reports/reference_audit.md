# Reference audit report

Date: 2026-07-01

## Scope

Reference source found in `paper/ThermoRoute_paper.md`, under `## References`, lines 516-600. No standalone `.bib`, `.bbl`, `.ris`, or `.enw` bibliography file was found in the project.

Total references audited: 38.

## Method

Adversarial checks used:

- DOI resolution and registered metadata via Crossref / DataCite / OpenAlex.
- Publisher or authoritative landing pages where available.
- arXiv pages for preprints.
- Official software/data citation pages for USGS, ORNL DAAC/NASA, NeurIPS, JMLR, PMLR, Springer, and Nature.
- Manual comparison of title, author list, year, venue, volume, pages/article number, DOI, and citation type.

Important note: I did not edit the manuscript references. This report records what should be fixed.

## Executive Summary

Most references are real and traceable. However, 7 references need correction before submission because they contain materially wrong authors, a wrong DOI, a dead DOI, or a dataset/software citation mismatch:

- L526 / F3: Frontiers in Water paper has a materially wrong/incomplete author list.
- L530 / F5: ACM SIGSPATIAL paper has a materially wrong/incomplete author list and omitted pages.
- L540 / F11: two different papers are conflated; the DOI points to a Topp WRR paper, while the title belongs to a Weierbach Water paper.
- L582 / Gupta 2009: DOI has a typo: `j.jhydro` should be `j.jhydrol`.
- L588 / dataretrieval Python: author list is wrong; official USGS citation lists Hariharan, Black, and Horsburgh, not Decker.
- L592 / Daymet V4 R1: official citation author order/list differs; current reference includes Wilson and wrong first author.
- L600 / SciPy: DOI is wrong/dead; correct DOI ends in `-2`, not `-5`.

Recommended but less severe:

- L586 / R `dataRetrieval`: author order and version should follow the exact USGS package citation used for the installed version.
- L590 / NWIS: add the recommended DOI `10.5066/F7P55KJN` and an access date or specific query citation.
- Several conference/arXiv/software references have no DOI/URL; they are real, but adding canonical URLs would improve traceability.

## Full Audit Table

| # | Line | Verdict | Finding |
|---:|---:|---|---|
| 1 | 522 | OK | Corona & Hogue 2025: DOI, title, venue, volume, pages, authors match Crossref/HESS. |
| 2 | 524 | OK | Feigl et al. 2021: DOI, title, venue, volume, pages, authors match Crossref/HESS. |
| 3 | 526 | Must fix | DOI/title are real, but author list is materially wrong/incomplete. Actual authors include Zwart, Diaz, Hamshaw, Oliver, Ross, Sleckman, Appling, Corson-Dosch, Jia, Read, Sadler, Thompson, Watkins, White. Current reference omits many and includes a nonmatching Bertassello/Diaz-Gonzalez-style author string. |
| 4 | 528 | OK | Jia et al. 2021 SDM: DOI, conference, pages, and arXiv preprint are real. Published title uses "Model"; arXiv title uses "Networks", acceptable because the citation cites the published version. |
| 5 | 530 | Must fix | DOI/title are real, but author list is wrong. Actual ACM/arXiv authors are Luo, Yu, Chen, Fan, Xie, Li, Jia. Current reference omits Fan/Xie/Li, adds Kumar, and omits pages 124-136. |
| 6 | 532 | OK | Rahmani et al. ERL: DOI and metadata match. Crossref online date can appear as 2020, but volume 16/article 024025 is conventionally cited as 2021. |
| 7 | 534 | OK | Rahmani et al. Hydrological Processes: DOI, title, article number, authors match. |
| 8 | 536 | OK | Rahmani et al. WRR 2023: DOI, title, article number, authors match. |
| 9 | 538 | OK | Sadler et al. WRR 2022: DOI, title, article number, authors match. |
| 10 | 540 | Must fix | Conflated citation. DOI `10.1029/2022WR033880` is Topp et al. "Stream Temperature Prediction in a Shifting Environment"; current title is Weierbach et al. 2022 in Water, DOI `10.3390/w14071032`. |
| 11 | 542 | OK | Toffolon & Piccolroaz 2015: DOI/title/venue/article number match. |
| 12 | 544 | OK | Piccolroaz et al. 2016: DOI/title/venue/pages/authors match. |
| 13 | 546 | OK | Mohseni et al. 1998: DOI/title/venue/pages/authors match. |
| 14 | 548 | OK | Caissie 2006: DOI/title/venue/pages match. |
| 15 | 552 | OK | Romano, Patterson & Candes 2019: real NeurIPS 2019 paper. Official NeurIPS metadata confirms pages 3543-3553. |
| 16 | 554 | OK / optional | Vovk, Gammerman & Shafer 2005 is real. Optional: add Springer DOI `10.1007/b106715`. |
| 17 | 556 | OK | Martins & Astudillo 2016: PMLR 48:1614-1623 confirmed. |
| 18 | 558 | OK / optional | Bai, Kolter & Koltun 2018 arXiv is real. Optional: cite `https://doi.org/10.48550/arXiv.1803.01271`. |
| 19 | 560 | OK / optional | Shazeer et al. 2017 ICLR/arXiv is real. Optional: add arXiv `1701.06538` or OpenReview URL. |
| 20 | 562 | OK | Ke et al. LightGBM 2017 NeurIPS: title/authors/pages confirmed. |
| 21 | 564 | OK | Diebold & Mariano 1995: DOI/title/venue/pages/authors match. |
| 22 | 566 | OK | Harvey, Leybourne & Newbold 1997: DOI with parentheses is valid; title/venue/pages match. |
| 23 | 568 | OK | Kunsch/Kuensch 1989: DOI/title/venue/pages match. |
| 24 | 570 | OK | Wilks 2011 book citation is real. |
| 25 | 572 | OK | Richardson 2000: DOI/title/venue/pages match. |
| 26 | 574 | OK | Gneiting & Raftery 2007: DOI/title/venue/pages match. |
| 27 | 576 | OK / optional | Kingma & Ba 2015 ICLR is real. Optional: add arXiv `1412.6980`. |
| 28 | 578 | OK / optional | Loshchilov & Hutter 2019 ICLR is real. Optional: add OpenReview URL or arXiv `1711.05101`. |
| 29 | 580 | OK | Nash & Sutcliffe 1970: DOI with parentheses is valid; title/venue/pages match. |
| 30 | 582 | Must fix | DOI typo. Current `10.1016/j.jhydro.2009.08.003` does not resolve; correct DOI is `10.1016/j.jhydrol.2009.08.003`. |
| 31 | 586 | Recommended fix | DOI is real, but the exact package citation should match the USGS package version. Official 2024 citation orders authors as De Cicco, Hirsch, Lorenz, Watkins, Johnson and includes version `v.2.7.17`; current reference swaps Hirsch/Lorenz and omits version. |
| 32 | 588 | Must fix | Official USGS citation is Hodson, Hariharan, Black, Horsburgh, 2023. Current "Hodson, Decker" is wrong. |
| 33 | 590 | Recommended fix | NWIS source is real, but recommended citation should include the USGS Water Data/NWIS DOI `10.5066/F7P55KJN`, access date, and/or exact query URL. |
| 34 | 592 | Must fix | Official Daymet V4 R1 citation is Thornton, M. M., Shrestha, R., Wei, Y., Thornton, P. E., & Kao, S.-C., 2022. Current author order/list is wrong and includes Wilson. |
| 35 | 594 | OK | Abatzoglou 2013: DOI/title/venue/pages match. |
| 36 | 596 | OK | PyTorch NeurIPS 2019 is real; official NeurIPS metadata confirms pages 8026-8037, so current pages are acceptable. |
| 37 | 598 | OK | Scikit-learn JMLR 2011: title/authors/volume/pages match official JMLR page. |
| 38 | 600 | Must fix | SciPy DOI is wrong/dead. Correct DOI is `10.1038/s41592-019-0686-2`; current `...0686-5` does not resolve. |

## Recommended Corrections

### L526 / F3

Replace current author list with the publisher/Crossref author list:

Zwart, J. A., Diaz, J., Hamshaw, S., Oliver, S., Ross, J. C., Sleckman, M., Appling, A. P., Corson-Dosch, H., Jia, X., Read, J., Sadler, J., Thompson, T., Watkins, D., White, E., 2023. Evaluating deep learning architecture and data assimilation for improving water temperature forecasts at unmonitored locations. Frontiers in Water 5, 1184992. https://doi.org/10.3389/frwa.2023.1184992.

Evidence: Frontiers landing page and DOI metadata.

### L530 / F5

Replace author list and add pages:

Luo, S., Yu, R., Chen, S., Fan, Y., Xie, Y., Li, Y., Jia, X., 2025. Geo-aware models for stream temperature prediction across different spatial regions and scales, in: Proceedings of the 33rd ACM International Conference on Advances in Geographic Information Systems. ACM, pp. 124-136. https://doi.org/10.1145/3748636.3762716 (preprint: arXiv:2510.09500).

Evidence: ACM DOI metadata and arXiv page.

### L540 / F11

Decide which paper was intended.

If the intended paper is the current title, replace the whole entry with:

Weierbach, H., Lima, A. R., Willard, J. D., Hendrix, V. C., Christianson, D. S., Lubich, M., Varadharajan, C., 2022. Stream temperature predictions for river basin management in the Pacific Northwest and Mid-Atlantic regions using machine learning. Water 14, 1032. https://doi.org/10.3390/w14071032.

If the intended paper is DOI `10.1029/2022WR033880`, replace with:

Topp, S. N., Barclay, J. R., Diaz, J. A., Sun, A. Y., Jia, X., Lu, D., Sadler, J. M., Appling, A. P., 2023. Stream temperature prediction in a shifting environment: explaining the influence of deep learning architecture. Water Resources Research 59, e2022WR033880. https://doi.org/10.1029/2022WR033880.

Evidence: MDPI Water citation page for Weierbach et al.; USGS/AGU citation page for Topp et al.

### L582 / Gupta 2009

Change only the DOI:

`https://doi.org/10.1016/j.jhydrol.2009.08.003`

### L586 / dataRetrieval R

Use the exact citation for the package version used. For the 2024 USGS package citation page:

De Cicco, L. A., Hirsch, R. M., Lorenz, D., Watkins, W. D., Johnson, M., 2024. dataRetrieval: R packages for discovering and retrieving water data available from Federal hydrologic web services, v.2.7.17. U.S. Geological Survey. https://doi.org/10.5066/P9X4L3GE.

If the manuscript used a later installed version, regenerate with `citation(package = "dataRetrieval")` and cite that exact version.

### L588 / dataretrieval Python

Replace with official USGS citation:

Hodson, T. O., Hariharan, J. A., Black, S., Horsburgh, J. S., 2023. dataretrieval (Python): a Python package for discovering and retrieving water data available from U.S. federal hydrologic web services. U.S. Geological Survey software release. https://doi.org/10.5066/P94I5TX3.

### L590 / NWIS

Recommended form:

U.S. Geological Survey, 2024. USGS Water Data for the Nation: U.S. Geological Survey National Water Information System database. https://doi.org/10.5066/F7P55KJN. Accessed [insert exact access date and/or query URL].

### L592 / Daymet V4 R1

Replace with official NASA/ORNL DAAC citation:

Thornton, M. M., Shrestha, R., Wei, Y., Thornton, P. E., Kao, S.-C., 2022. Daymet: daily surface weather data on a 1-km grid for North America, version 4 R1 (Version 4.1). ORNL Distributed Active Archive Center. https://doi.org/10.3334/ORNLDAAC/2129.

### L600 / SciPy

Change DOI to:

`https://doi.org/10.1038/s41592-019-0686-2`

The rest of the entry is consistent with Nature Methods metadata.

## Key Evidence Links

- Frontiers F3 article: https://www.frontiersin.org/journals/water/articles/10.3389/frwa.2023.1184992/full
- ACM / arXiv F5: https://doi.org/10.1145/3748636.3762716 and https://arxiv.org/abs/2510.09500
- Weierbach Water article: https://www.mdpi.com/2073-4441/14/7/1032
- Topp WRR article via USGS: https://pubs.usgs.gov/publication/70250381
- Gupta 2009 DOI evidence: https://experts.arizona.edu/en/publications/decomposition-of-the-mean-squared-error-and-nse-performance-crite/
- R `dataRetrieval` citation: https://rconnect.usgs.gov/dataRetrieval/authors.html and https://water.code-pages.usgs.gov/dataRetrieval/
- Python `dataretrieval` citation: https://water.usgs.gov/catalog/tools/0388b0a4-66ec-47ad-9ba9-07ac621ddd06/
- Daymet V4 R1 official citation: https://www.earthdata.nasa.gov/data/catalog/ornl-cloud-daymet-daily-v4r1-2129-4.1
- SciPy Nature Methods article: https://www.nature.com/articles/s41592-019-0686-2
- NeurIPS CQR metadata: https://papers.nips.cc/paper_files/paper/2019/file/5103c3584b063c431bd1268e9b5e76fb-Metadata.json
- NeurIPS PyTorch metadata: https://papers.nips.cc/paper_files/paper/2019/file/bdbca288fee7f92f2bfa9f7012727740-Metadata.json
- JMLR scikit-learn page: https://www.jmlr.org/papers/v12/pedregosa11a.html
- PMLR sparsemax page: https://proceedings.mlr.press/v48/martins16.html
- Springer ALRW book: https://link.springer.com/book/10.1007/b106715

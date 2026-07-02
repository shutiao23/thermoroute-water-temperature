#!/usr/bin/env python3
"""Assemble the AGU/WRR LaTeX submission from the Markdown manuscript.

Front matter (title/authors/keypoints/abstract) is written in AGU macros; the
body is converted with pandoc; the single equation block becomes real LaTeX
math; the bibliography is apacite over references.bib. Compile with:
  pdflatex -> bibtex -> pdflatex -> pdflatex
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MD = ROOT / "paper" / "ThermoRoute_paper.md"
PANDOC = subprocess.check_output(
    [sys.executable, "-c", "import pypandoc; print(pypandoc.get_pandoc_path())"]
).decode().strip()

md = MD.read_text()
body_all = md.split("## References")[0]  # drop the manual reference list

# ---- inline citation labels -> AGU \citeA / \cite (raw latex passthrough) ----
body_all = body_all.replace("[F1]", r"\citeA{corona2025ml}")
body_all = body_all.replace("[G1]", r"\citeA{beven1992future}")
body_all = body_all.replace("[G2, G3]",
                            r"\cite{kavetski2006theory,kavetski2006application}")

# ---- the one equation block -> real LaTeX math (raw passthrough) ------------
eq_src = """```
e_t = g(weather_t) + b_s                                  (equilibrium anomaly)
κ   = σ(β_s + c_q·z(logFLOW) + c_l·z(WLEVEL) + w·season)  (daily relaxation rate)
â_h = e_t + (1−κ)^h (a_t − e_t)
Ŵ_{t+h}^prior = C_{t+h} + â_h
```"""
eq_tex = r"""
\begin{linenomath*}
\begin{align}
e_t &= g(\mathrm{weather}_t) + b_s
     && \text{(equilibrium anomaly)} \nonumber\\
\kappa &= \sigma\!\big(\beta_s + c_q\, z(\log \mathrm{FLOW})
        + c_l\, z(\mathrm{WLEVEL}) + w\,\mathrm{season}\big)
     && \text{(daily relaxation rate)} \nonumber\\
\hat a_h &= e_t + (1-\kappa)^h (a_t - e_t) \nonumber\\
\widehat W_{t+h}^{\mathrm{prior}} &= C_{t+h} + \hat a_h \nonumber
\end{align}
\end{linenomath*}
"""
if eq_src not in body_all:
    print("WARNING: equation block not found verbatim; check whitespace")
body_all = body_all.replace(eq_src, eq_tex)

# ---- split into title / abstract / body / open-research ---------------------
title = re.search(r"^# (.+)$", body_all, re.M).group(1).strip()

abstract = re.search(r"## Abstract\s*(.+?)\n## ", body_all, re.S).group(1).strip()

# body = from "## 1. Introduction" up to "## Data availability"
body = re.search(r"(## 1\. Introduction.*?)\n## Data availability",
                 body_all, re.S).group(1)
# open research = the two availability sections
openres = re.search(r"(## Data availability.*)$", body_all, re.S).group(1)

def pandoc(md_text, shift=0):
    args = [PANDOC, "-f", "markdown", "-t", "latex", "--wrap=preserve"]
    if shift:
        args.append(f"--shift-heading-level-by={shift}")
    p = subprocess.run(args, input=md_text.encode(), capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode())
    return p.stdout.decode()

def strip_heading_numbers(md_text):
    # "## 4. Results" -> "## Results"; "### 4.2 Foo" -> "### Foo"
    return re.sub(r"(?m)^(#+)\s+\d+(?:\.\d+)*\.?\s+", r"\1 ", md_text)

abstract_tex = pandoc(abstract)
# strip manual "**Figure N.** " from image captions (AGU auto-numbers figures)
body = re.sub(r"!\[\*\*Figure \d+\.\*\*\s*", "![", body)
# body: strip manual section numbers, promote H2->section, H3->subsection
body_tex = pandoc(strip_heading_numbers(body), shift=-1)
# open research: ## -> subsection, then make unnumbered under \section*{Open Research}
openres_tex = pandoc(openres)
openres_tex = openres_tex.replace(r"\subsection{", r"\subsection*{")

# demote markdown "## n. Title" -> AGU \section (pandoc already made \section);
# figures: outputs/figures/X.png -> figures/X.png ; strip pandoc figure floats' width to textwidth
for t in ("body_tex", "openres_tex"):
    pass
body_tex = body_tex.replace("outputs/figures/", "figures/")
# pandoc emits \includegraphics[width=...]{...}; make figures span textwidth and float
body_tex = re.sub(r"\\includegraphics\[[^\]]*\]", r"\\includegraphics[width=\\textwidth]", body_tex)

# ---- copy figures locally ---------------------------------------------------
figdir = HERE / "figures"; figdir.mkdir(exist_ok=True)
for f in ["fig3_results_heatmap.png", "fig_usgs_perstation.png",
          "fig_usgs_calibration.png", "fig_usgs_kappa.png"]:
    src = ROOT / "outputs" / "figures" / f
    if src.exists():
        shutil.copy(src, figdir / f)

# ---- unicode declarations for pdflatex --------------------------------------
UNI = r"""
\usepackage{newunicodechar}
\newunicodechar{§}{\S}
\newunicodechar{°}{\ensuremath{^\circ}}
\newunicodechar{±}{\ensuremath{\pm}}
\newunicodechar{²}{\textsuperscript{2}}
\newunicodechar{³}{\textsuperscript{3}}
\newunicodechar{¹}{\textsuperscript{1}}
\newunicodechar{·}{\ensuremath{\cdot}}
\newunicodechar{×}{\ensuremath{\times}}
\newunicodechar{â}{\^a}
\newunicodechar{Ŵ}{\^W}
\newunicodechar{Δ}{\ensuremath{\Delta}}
\newunicodechar{α}{\ensuremath{\alpha}}
\newunicodechar{β}{\ensuremath{\beta}}
\newunicodechar{θ}{\ensuremath{\theta}}
\newunicodechar{κ}{\ensuremath{\kappa}}
\newunicodechar{σ}{\ensuremath{\sigma}}
\newunicodechar{φ}{\ensuremath{\varphi}}
\newunicodechar{–}{--}
\newunicodechar{—}{---}
\newunicodechar{…}{\ldots}
\newunicodechar{⁰}{\textsuperscript{0}}
\newunicodechar{⁴}{\textsuperscript{4}}
\newunicodechar{⁵}{\textsuperscript{5}}
\newunicodechar{⁶}{\textsuperscript{6}}
\newunicodechar{⁸}{\textsuperscript{8}}
\newunicodechar{⁹}{\textsuperscript{9}}
\newunicodechar{⁻}{\textsuperscript{$-$}}
\newunicodechar{→}{\ensuremath{\rightarrow}}
\newunicodechar{∈}{\ensuremath{\in}}
\newunicodechar{−}{\ensuremath{-}}
\newunicodechar{∓}{\ensuremath{\mp}}
\newunicodechar{≈}{\ensuremath{\approx}}
\newunicodechar{≤}{\ensuremath{\leq}}
\newunicodechar{≥}{\ensuremath{\geq}}
\newunicodechar{⌈}{\ensuremath{\lceil}}
\newunicodechar{⌉}{\ensuremath{\rceil}}
"""

PREAMBLE = r"""\documentclass{agujournal2019}
\usepackage{amsmath}
\usepackage{url}
\usepackage{graphicx}
\graphicspath{{figures/}}
\usepackage{longtable,booktabs,array}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
""" + UNI + r"""
\journalname{Water Resources Research}
\begin{document}

\title{%s}

\authors{[Author One]\affil{1}\thanks{Corresponding author},
[Author Two]\affil{1}, [Author Three]\affil{2}}

\affiliation{1}{[Department / Laboratory, Institution, City, Country]}
\affiliation{2}{[Department / Laboratory, Institution, City, Country]}

\correspondingauthor{[Author One]}{[corresponding.author@institution.edu]}

\begin{keypoints}
\item A learnable relaxation prior reduces to damped persistence so a bounded neural residual adds skill only where forecast headroom exists
\item On 120 USGS stations ThermoRoute beats persistence and damped persistence and transfers to unseen basins in leave group out testing
\item Conformal calibration yields near nominal per station coverage without the explicit error model that the GLUE and BATEA lineage requires
\end{keypoints}

\begin{abstract}
%s
\end{abstract}

""" % (title, abstract_tex)

OPEN = r"""

\section*{Open Research}
%s
""" % openres_tex

TAIL = r"""

\nocite{*}
\bibliography{references}

\end{document}
"""

out = PREAMBLE + body_tex + OPEN + TAIL
(HERE / "ThermoRoute_WRR.tex").write_text(out)
print(f"wrote ThermoRoute_WRR.tex ({len(out)} chars); title len={len(title)}")
# keypoint length check
for kp in re.findall(r"\\item (.+)", PREAMBLE):
    print(f"  keypoint {len(kp)} chars: {kp[:60]}...")

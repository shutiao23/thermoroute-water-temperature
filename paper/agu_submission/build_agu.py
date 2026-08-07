#!/usr/bin/env python3
"""Build the AGU manuscript LaTeX from the canonical Markdown.

The Markdown manuscript is the only prose source.  This generator deliberately
contains no empirical result sentence: front matter is status-safe, the body is
converted from the current Markdown, and known withdrawn claims are rejected
before any TeX is written.  ``--check`` is read-only and fails when the checked-in
TeX is not exactly what this source would generate.

The manuscript is a conventional comparative 2021--2023 holdout study; the
held-out metric cells are ``<<...>>`` placeholders filled from
``outputs/conventional/holdout_metrics_2021_2023.csv``.  This builder only
converts the canonical Markdown to AGU LaTeX; it contains no empirical result
sentence and no figure or result renderer.  A future submission view that fills
the placeholders from the metrics CSV will use a separate renderer.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MARKDOWN = ROOT / "paper" / "ThermoRoute_paper.md"
OUTPUT = HERE / "ThermoRoute_WRR.tex"

WITHDRAWN_PATTERNS = {
    "legacy 40-site cohort": re.compile(r"\b40\s+(?:public\s+)?USGS stations\b", re.I),
    "legacy reportable N=114": re.compile(r"\b(?:n\s*=\s*)?114\s+(?:blind-test\s+)?stations\b", re.I),
    "legacy skill triplet": re.compile(r"\+0\.13\s*/\s*\+0\.14\s*/\s*\+0\.23"),
    "legacy RMSE triplet": re.compile(r"0\.554\s*/\s*1\.175\s*/\s*1\.490"),
    "legacy air2stream triplet": re.compile(r"0\.630\s*/\s*1\.289\s*/\s*1\.658"),
    "legacy coverage range": re.compile(r"89\s*[-–—]\s*97\s*%"),
    "legacy transfer distance": re.compile(r"\b358\s*km\b", re.I),
    "legacy superiority": re.compile(r"\bsignificantly beats\b", re.I),
    "legacy unseen-basin claim": re.compile(r"\btransfers? to unseen basins\b", re.I),
    "legacy coverage claim": re.compile(r"\bnear[- ]nominal per[- ]station coverage\b", re.I),
    "legacy safety claim": re.compile(r"\bbounded-degradation guarantee\b", re.I),
}

# The status block must (a) name the study design and (b) promise that anything
# not computed is flagged rather than silently omitted.  The second phrase used
# to be "clearly marked", from "the 2021-2023 metric cells are clearly marked
# <<...>> placeholders pending computation".  Section 4.6 is now filled from the
# holdout CSV, so that sentence was removed as untrue and replaced by "quantities
# whose pipelines were not re-run for 2021-2023 are explicitly marked as not
# reported".  The guard tracks the promise, not the old wording.
REQUIRED_STATUS_TEXT = (
    "conventional comparative holdout",
    "explicitly marked as not reported",
)

# AGU allows at most three Key Points of at most 140 characters each.  They are
# authored once, in the canonical Markdown, and mirrored in
# ``paper/highlights.md``; this generator reads them rather than restating them.
#
# ``agujournal2025.cls`` takes the Key Points as a three-argument macro
# (``\keypoints{}{}{}``) rather than the ``keypoints`` environment of
# ``agujournal2019.cls``.  The arity is fixed by the class, so KEYPOINT_COUNT is
# not free: ``_render`` re-asserts it before formatting the macro call.
KEYPOINT_LIMIT = 140
KEYPOINT_COUNT = 3
KEYPOINTS_MACRO_ARITY = 3

# Held-out 2021--2023 metric cells are ``<<...>>`` placeholders filled from
# ``outputs/conventional/holdout_metrics_2021_2023.csv``.  The marker is the
# opening ``<<`` of such a placeholder; the generator only checks that at least
# one placeholder survives conversion, so an empty cell can never be presented
# as a computed result.  The exact count is not fixed, because the metric tables
# carry one placeholder per (model, horizon, metric) cell.
RESULT_SLOT_MARKER = "<<"


def _result_slot_pattern() -> re.Pattern[str]:
    """Match one result-slot placeholder in *converted* LaTeX, not in Markdown.

    The canonical Markdown writes each slot inside a code span as ``<<...>>``.
    Pandoc escapes the angle brackets, so the literal ``<<`` is not present in
    the converted LaTeX.  This pattern accepts the escaped forms Pandoc emits
    (``\textless{}{}`` and ``\textless\textless``) as well as the raw marker, so
    the "no slot may be lost in conversion" guarantee holds regardless of how
    the brackets are rendered.
    """
    # Match the opening ``<<`` of a ``<<...>>`` placeholder, accepting both the
    # raw characters and the escaped forms Pandoc emits (``\textless{}\textless``
    # and ``\textless\textless``).  The body of the placeholder is irrelevant to
    # the survival check; only the opening pair is matched.
    less = r"(?:\\textless(?:\{\})?)+"
    return re.compile(rf"(?:<<|(?:{less}){{2}}|{less}(?:\\ )?{less})")


RESULT_SLOT_LATEX = _result_slot_pattern()

# The Stage-19 disposition bans this phrase: zero strict ordering violations were
# observed, and the affected rows are zero-width (degenerate) nominal intervals.
BANNED_PHRASES = ("quantile crossing",)

# agujournal2025.cls is a pdfTeX class and loads neither inputenc nor a Unicode
# font encoding, so every non-ASCII codepoint the manuscript uses must be mapped
# explicitly.  This table is the single source of truth: it emits the
# ``\DeclareUnicodeCharacter`` preamble *and* backs a pre-write check, so an
# unmapped character fails the generator with a named character instead of
# failing pdflatex dozens of pages into the run.  Do not delete entries that are
# currently unused; they are cheap and they keep prose edits from breaking the
# build.
#
# The 2026-08-06 benchmark restructure added equations (1)-(10) to the
# manuscript body, which brought in the Greek and mathematical codepoints below.
# Adding a mapping is the remedy this table is designed for; the alternative the
# error message offers -- rewording the Markdown -- was used only where no
# faithful mapping exists.  Two combining marks, U+0302 and U+0303, are
# deliberately absent: a ``\DeclareUnicodeCharacter`` mapping receives a
# combining mark *after* its base letter, and LaTeX accent commands are
# prefixes, so no honest mapping exists.  The manuscript was reworded instead
# (``q̃`` became ``q`` and ``q̂_τ`` became ``Q_τ``, both defined in place).  Do
# not add a mapping that silently drops or misplaces an accent.
UNICODE_DECLARATIONS: dict[int, str] = {
    0x00A7: r"\S{}",
    0x00B0: r"\ensuremath{^\circ}",
    0x00B1: r"\ensuremath{\pm}",
    0x00B2: r"\textsuperscript{2}",
    0x00B3: r"\textsuperscript{3}",
    0x00B7: r"\ensuremath{\cdot}",
    0x00D7: r"\ensuremath{\times}",
    0x00FC: r"\"u",
    0x0177: r"\^y",
    0x0394: r"\ensuremath{\Delta}",
    0x03A3: r"\ensuremath{\Sigma}",
    0x03B2: r"\ensuremath{\beta}",
    0x03B3: r"\ensuremath{\gamma}",
    0x03B4: r"\ensuremath{\delta}",
    0x03BA: r"\ensuremath{\kappa}",
    0x03BB: r"\ensuremath{\lambda}",
    0x03C0: r"\ensuremath{\pi}",
    0x03C1: r"\ensuremath{\rho}",
    0x03C3: r"\ensuremath{\sigma}",
    0x03C4: r"\ensuremath{\tau}",
    0x03C6: r"\ensuremath{\varphi}",
    0x2013: "--",
    0x2014: "---",
    0x2016: r"\ensuremath{\|}",
    0x2026: r"\ldots{}",
    0x2074: r"\textsuperscript{4}",
    0x2081: r"\ensuremath{_1}",
    0x2113: r"\ensuremath{\ell}",
    0x2192: r"\ensuremath{\rightarrow}",
    0x207B: r"\textsuperscript{-}",
    0x2208: r"\ensuremath{\in}",
    0x2212: r"\ensuremath{-}",
    0x221A: r"\ensuremath{\surd}",
    0x2248: r"\ensuremath{\approx}",
    0x2264: r"\ensuremath{\leq}",
    0x2265: r"\ensuremath{\geq}",
    0x27E8: r"\ensuremath{\langle}",
    0x27E9: r"\ensuremath{\rangle}",
}


def _unicode_declarations() -> str:
    return "\n".join(
        rf"\DeclareUnicodeCharacter{{{codepoint:04X}}}{{{replacement}}}"
        for codepoint, replacement in sorted(UNICODE_DECLARATIONS.items())
    )


def _assert_unicode_is_declared(latex: str) -> None:
    """Refuse to emit TeX carrying a codepoint pdflatex has no mapping for."""
    unmapped = sorted(
        {
            character
            for character in latex
            if ord(character) > 127 and ord(character) not in UNICODE_DECLARATIONS
        }
    )
    if unmapped:
        detail = ", ".join(
            f"U+{ord(character):04X} ({unicodedata.name(character, 'unnamed')})"
            for character in unmapped
        )
        raise ValueError(
            "generated TeX uses undeclared non-ASCII characters; add them to "
            f"UNICODE_DECLARATIONS or reword the Markdown: {detail}"
        )


def _pandoc_path() -> str:
    candidates: list[Path] = []
    environment_candidate = os.environ.get("PANDOC_PATH")
    if environment_candidate:
        candidates.append(Path(environment_candidate))
    executable = shutil.which("pandoc")
    if executable:
        candidates.append(Path(executable))
    # pypandoc commonly vendors its executable without placing it on PATH.
    # Search only known Python-environment roots, never the repository or an
    # unbounded filesystem tree.
    for library_root in (
        Path(sys.prefix) / "lib",
        Path("/opt/anaconda3/lib"),
        Path("/opt/homebrew/lib"),
        Path("/usr/local/lib"),
    ):
        if library_root.is_dir():
            candidates.extend(sorted(
                library_root.glob("python*/site-packages/pypandoc/files/pandoc")
            ))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pypandoc; print(pypandoc.get_pandoc_path())",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    candidate = process.stdout.strip()
    if process.returncode != 0 or not candidate or not Path(candidate).is_file():
        raise RuntimeError("pandoc is required to build the AGU manuscript")
    return candidate


def _convert(markdown: str, *, shift_headings: int | None = None) -> str:
    command = [
        _pandoc_path(),
        "--from=gfm-raw_html+implicit_figures",
        "--to=latex",
        "--wrap=preserve",
    ]
    if shift_headings is not None:
        command.append(f"--shift-heading-level-by={shift_headings}")
    process = subprocess.run(
        command,
        input=markdown,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"pandoc conversion failed: {process.stderr.strip()}")
    return process.stdout.strip()


def _extract(markdown: str, pattern: str, *, label: str) -> str:
    match = re.search(pattern, markdown, flags=re.MULTILINE | re.DOTALL)
    if match is None:
        raise ValueError(f"canonical Markdown lacks one unambiguous {label}")
    return match.group(1).strip()


def _extract_keypoints(markdown: str) -> tuple[str, ...]:
    """Read the Key Points block from the canonical Markdown and check AGU limits."""
    block = _extract(
        markdown,
        r"^## Key Points\s*\n(.*?)(?=^## Abstract)",
        label="Key Points block",
    )
    items: list[str] = []
    for raw in re.split(r"(?m)^- ", block):
        candidate = " ".join(raw.split())
        if candidate:
            items.append(candidate)
    if len(items) != KEYPOINT_COUNT:
        raise ValueError(
            f"canonical Markdown must contain exactly {KEYPOINT_COUNT} Key Points, "
            f"found {len(items)}"
        )
    for item in items:
        if len(item) > KEYPOINT_LIMIT:
            raise ValueError(
                f"Key Point exceeds the {KEYPOINT_LIMIT}-character AGU limit "
                f"({len(item)}): {item}"
            )
        if not item.endswith("."):
            raise ValueError(f"Key Point is not a complete sentence: {item}")
    return tuple(items)


def _validate_markdown(markdown: str) -> None:
    folded = markdown.casefold()
    for phrase in BANNED_PHRASES:
        if phrase in folded:
            raise ValueError(f"canonical Markdown contains a banned phrase: {phrase}")
    # The invariant is that an empty result cell can never be presented as a
    # computed value.  While the held-out cells were unfilled that meant "at
    # least one ``<<...>>`` must survive"; now that Section 4.6 is computed from
    # holdout_metrics_2021_2023.csv it means the opposite -- no unfilled slot may
    # remain.  Anything genuinely unavailable is written out as "not reported",
    # which is prose a reader can act on, not a token that looks like a value.
    if "<<placeholder>>" in folded:
        raise ValueError("canonical Markdown contains a literal <<placeholder>> stub")
    if RESULT_SLOT_MARKER in markdown:
        surviving = markdown.count(RESULT_SLOT_MARKER)
        raise ValueError(
            f"canonical Markdown still carries {surviving} unfilled "
            f"'{RESULT_SLOT_MARKER}...' result slot(s); fill them from the holdout "
            "CSV or state the quantity as not reported"
        )
    _extract_keypoints(markdown)
    violations = [
        label for label, pattern in WITHDRAWN_PATTERNS.items()
        if pattern.search(markdown)
    ]
    if violations:
        raise ValueError(f"withdrawn claims remain in canonical Markdown: {violations}")


def _strip_machine_comments(markdown: str) -> str:
    return re.sub(r"<!--.*?-->", "", markdown, flags=re.DOTALL)


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _make_code_spans_breakable(latex: str) -> str:
    """Render Pandoc code spans as breakable, non-linking URL-style text.

    The payload pattern admits one level of nested braces.  A flat ``[^{}]+``
    silently skipped every code span Pandoc had brace-protected -- which is
    exactly the widest ones, because Pandoc writes ``[`` and ``]`` as ``{[}`` and
    ``{]}``.  Those spans were then the unbreakable cells that pushed the
    five-comparison table past the text block.
    """
    pattern = re.compile(r"\\texttt\{((?:[^{}]|\{[^{}]*\})*)\}")

    def replace(match: re.Match[str]) -> str:
        payload = match.group(1)
        payload = payload.replace(r"\_", r"\_\allowbreak{}")
        payload = payload.replace("/", r"/\allowbreak{}")
        # Pandoc escapes spaces inside code spans as ``\ ``; without an explicit
        # break opportunity a multi-word code span is one unbreakable box.
        payload = payload.replace("\\ ", "\\ \\allowbreak{}")
        return rf"\texttt{{{payload}}}"

    rendered = pattern.sub(replace, latex)
    return rendered.replace(
        "station/date/horizon", r"station/\allowbreak date/\allowbreak horizon"
    )


# A column whose widest cell reaches this many characters cannot be trusted to a
# natural-width ``l`` column: several such columns in one table overrun the text
# block no matter how many break opportunities the cell content carries, because
# ``l`` never wraps.  Such columns are promoted to bounded, wrapping ``X``.
WRAPPING_COLUMN_CHARACTERS = 24

_UNESCAPED_AMPERSAND = re.compile(r"(?<!\\)&")

# A promoted ``X`` column keeps the alignment Pandoc inferred from the Markdown.
# The last column is always promoted (tabularx needs at least one ``X``), so a
# single raggedright constant would silently left-align the final numeric column
# of every numeric table while its siblings stayed right-aligned.
_X_COLUMNS = {
    "l": r">{\raggedright\arraybackslash}X",
    "c": r">{\centering\arraybackslash}X",
    "r": r">{\raggedleft\arraybackslash}X",
}
_X_COLUMN = _X_COLUMNS["l"]


def _column_content_widths(columns: str, content: str) -> list[int]:
    """Widest cell per column, measured over the table's data and header rows."""
    widths = [0] * len(columns)
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("\\"):
            continue
        cells = _UNESCAPED_AMPERSAND.split(stripped.removesuffix(r"\\"))
        if len(cells) != len(columns):
            continue
        for index, cell in enumerate(cells):
            widths[index] = max(widths[index], len(cell.strip()))
    return widths


def _bounded_column_spec(columns: str, content: str) -> str:
    """Bound a Pandoc ``lcr`` spec, wrapping every column that needs to wrap.

    The last column is always bounded, which is what the 2019-era conversion did
    and what tabularx requires (at least one ``X``).  Any other column whose
    content is wide is bounded as well; leaving them natural-width is what let
    the formal-comparison table run 155pt past the margin.
    """
    widths = _column_content_widths(columns, content)
    last = len(columns) - 1
    return "".join(
        _X_COLUMNS[letter]
        if index == last or widths[index] >= WRAPPING_COLUMN_CHARACTERS
        else letter
        for index, letter in enumerate(columns)
    )


def _fit_longtables(latex: str) -> str:
    """Convert Pandoc's natural-width tables to bounded tabularx tables."""
    pattern = re.compile(
        r"\{\\def\\LTcaptype\{none\}[^\n]*\n"
        r"\\begin\{longtable\}\[\]\{@\{\}([lcr]+)@\{\}\}\n"
        r"(.*?)"
        r"\\end\{longtable\}\n\}",
        flags=re.DOTALL,
    )

    def replace(match: re.Match[str]) -> str:
        columns = match.group(1)
        if not columns:
            raise ValueError("Pandoc longtable has no columns")
        bounded_columns = _bounded_column_spec(columns, match.group(2))
        content = match.group(2)
        content = content.replace("\\endhead\n", "")
        content = content.replace(
            "\\bottomrule\\noalign{}\n\\endlastfoot\n", ""
        ).rstrip()
        return (
            "{\\small\n"
            "\\setlength{\\tabcolsep}{3pt}\n"
            "\\renewcommand{\\arraystretch}{1.12}\n"
            "\\noindent\n"
            "\\begin{tabularx}{\\dimexpr\\linewidth-24pt\\relax}"
            f"{{@{{}}{bounded_columns}@{{}}}}\n"
            f"{content}\n"
            "\\bottomrule\\noalign{}\n"
            "\\end{tabularx}\n"
            "}"
        )

    fitted = pattern.sub(replace, latex)
    # The 2019 preamble carried a ``c@none`` counter shim so that a longtable
    # Pandoc had tagged ``\LTcaptype{none}`` could still be typeset.  Every table
    # is now converted to tabularx, so that shim was dead code and has been
    # dropped.  Assert the precondition instead of silently depending on it: a
    # table this converter does not recognise must fail the build, not reach
    # pdflatex and error there.
    if "\\begin{longtable}" in fitted or "\\LTcaptype" in fitted:
        raise ValueError(
            "a Pandoc longtable survived tabularx conversion; either extend "
            "_fit_longtables or restore the c@none counter shim in the preamble"
        )
    return fitted


def _agu_back_matter(latex: str) -> str:
    """Route AGU's fixed back-matter headings to the class's own constructs.

    AGU numbers the narrative sections itself but expects Open Research,
    Acknowledgments, and Supporting Information to stand outside that numbering,
    with Acknowledgments using the class's ``\\acknowledgments`` construct.

    ``agujournaltemplate.tex`` names the data-availability heading "Open Research
    Statement"; the canonical Markdown heading is "Open Research".  The heading
    text is AGU's, not a claim, so it is normalised here rather than in the
    hash-frozen Markdown.  The template also places that section immediately
    before the bibliography, which the current Markdown ordering does not do --
    see docs/AGU2025_TEMPLATE_MIGRATION.md; reordering narrative sections is an
    author decision and is deliberately not done by this generator.
    """
    latex, replaced = re.subn(
        r"\\section\{Open Research\}\\label\{[^}]*\}",
        r"\\section*{Open Research Statement}",
        latex,
    )
    if replaced != 1:
        raise ValueError("converted body lacks exactly one Open Research section")
    latex = re.sub(
        r"\\section\{Supporting Information\}\\label\{[^}]*\}",
        r"\\section*{Supporting Information}",
        latex,
    )
    latex, replaced = re.subn(
        r"\\section\{Acknowledgments\}\\label\{[^}]*\}",
        r"\\acknowledgments",
        latex,
    )
    if replaced != 1:
        raise ValueError("converted body lacks exactly one Acknowledgments section")
    return latex


def _render(markdown: str) -> str:
    _validate_markdown(markdown)
    title = _extract(markdown, r"^#\s+(.+?)$", label="title")
    keypoint_items = _extract_keypoints(markdown)
    abstract = _extract(
        markdown,
        r"^## Abstract\s*\n(.*?)(?=^\*\*Plain Language Summary\.\*\*)",
        label="abstract",
    )
    plain_language = _extract(
        markdown,
        r"^\*\*Plain Language Summary\.\*\*\s*(.*?)(?=^\*\*Keywords:\*\*)",
        label="Plain Language Summary",
    )
    keywords = _extract(
        markdown,
        r"^\*\*Keywords:\*\*\s*(.*?)(?=^---\s*$)",
        label="keyword list",
    )
    body = _extract(markdown, r"(^## 1\..*)\Z", label="numbered body")
    body = _strip_machine_comments(body)
    # AGU numbers sections itself.  Preserve subsection hierarchy while
    # removing only the manual numeric prefixes from Markdown headings.
    body = re.sub(
        r"(?m)^(#{2,6})\s+\d+(?:\.\d+)*\.?\s+",
        r"\1 ",
        body,
    )
    abstract_tex = _make_code_spans_breakable(
        _convert(_strip_machine_comments(abstract))
    )
    plain_language_tex = _make_code_spans_breakable(_convert(plain_language))
    keywords_tex = _make_code_spans_breakable(_convert(keywords))
    body_tex = _agu_back_matter(
        _fit_longtables(
            _make_code_spans_breakable(_convert(body, shift_headings=-1))
        )
    )
    # ``\allowbreak{}`` is a break hint this generator injects, never manuscript
    # content, so it is normalised away before the slot markers are counted.
    surviving = len(
        RESULT_SLOT_LATEX.findall(body_tex.replace(r"\allowbreak{}", ""))
    )
    # Counterpart to the Markdown-side check in _validate_markdown: while the
    # held-out cells were unfilled this guarded against Pandoc eating the
    # placeholders, so it required at least one to survive.  Section 4.6 is now
    # computed, the Markdown carries no slots, and the invariant inverts -- a
    # surviving slot here would mean the converter manufactured one.
    if surviving:
        raise ValueError(
            f"conversion produced {surviving} '<<...>>' result slot(s) from a "
            "Markdown source that carries none"
        )

    if len(keypoint_items) != KEYPOINTS_MACRO_ARITY:
        raise ValueError(
            f"\\keypoints in agujournal2025.cls takes exactly "
            f"{KEYPOINTS_MACRO_ARITY} arguments, got {len(keypoint_items)}"
        )
    keypoints = "\n".join(
        f"    {{{_latex_escape(item)}}}" for item in keypoint_items
    )
    unicode_declarations = _unicode_declarations()
    rendered = rf"""\documentclass[draft,linenumbers]{{agujournal2025}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs,longtable,array,tabularx}}
\usepackage{{rotating}}
\usepackage{{url,xurl}}
\usepackage{{hyperref}}
\providecommand{{\tightlist}}{{\setlength{{\itemsep}}{{0pt}}\setlength{{\parskip}}{{0pt}}}}
\providecommand{{\ph}}[1]{{\texttt{{\textless{{}}\textless{{}}#1\textgreater{{}}\textgreater{{}}}}}}
\providecommand{{\pandocbounded}}[1]{{#1}}
\setlength{{\emergencystretch}}{{3em}}
% Float policy (reviewer-layout pass): keep figures on the same page as
% text; a figure is never alone on a page unless it fills most of it.
\setcounter{{topnumber}}{{2}}
\setcounter{{bottomnumber}}{{1}}
\setcounter{{totalnumber}}{{3}}
\renewcommand{{\topfraction}}{{0.90}}
\renewcommand{{\bottomfraction}}{{0.75}}
\renewcommand{{\textfraction}}{{0.08}}
\renewcommand{{\floatpagefraction}}{{0.88}}
\setlength{{\textfloatsep}}{{10pt plus 2pt minus 2pt}}
\setlength{{\floatsep}}{{8pt plus 2pt minus 2pt}}
\setlength{{\intextsep}}{{10pt plus 2pt minus 2pt}}
\makeatletter
\setlength{{\@fptop}}{{0pt}}
\setlength{{\@fpsep}}{{12pt}}
\setlength{{\@fpbot}}{{0pt plus 1fil}}
\makeatother
% Still required under agujournal2025.cls: removing \sloppy reintroduces four
% overfull \hbox warnings in the body text.
\sloppy
{unicode_declarations}

\begin{{document}}

% agujournal2025.cls requires \journalname after \begin{{document}}; the 2019
% class took it in the preamble.  In the submission (non-``published'') branch it
% feeds the "manuscript submitted to ..." running head.
\journalname{{Water Resources Research}}

\title{{{_latex_escape(title)}}}

% AUTHOR BLOCK TO BE COMPLETED.  Names, affiliations, ORCIDs, the author count,
% and the corresponding author are deliberately unfilled and are not invented by
% this generator.  Replace with one \authors entry carrying the final agreed
% order in the 2025 form --- ``Name\affil{{1}}\thanks{{funding}}, Name\affil{{2}}''
% --- one \affiliation per distinct affiliation, and a verified institutional
% address.  The signed intake schema is
% docs/FAIR_SUBMISSION_READINESS_AND_TEMPLATES.md section 2.
\authors{{[AUTHOR LIST TO BE COMPLETED]\affil{{1}}}}
\affiliation{{1}}{{[AFFILIATION 1 TO BE COMPLETED --- department or laboratory, institution, city, postcode, country]}}
\affiliation{{2}}{{[AFFILIATION 2 TO BE COMPLETED --- add or delete affiliation lines to match the final author list]}}

% \authoraddr is the 2025 corresponding-author interface and replaces the 2019
% \correspondingauthor{{name}}{{email}} pair.  Both are emitted on purpose: in the
% ``published'' branch \authoraddr is typeset in the first-page margin and
% \correspondingauthor does not exist, while in the submission branch used here
% \authoraddr is defined as a no-op and only \correspondingauthor reaches the
% page.  Keep the two in agreement when the placeholders are filled.
\authoraddr{{[CORRESPONDING AUTHOR TO BE COMPLETED --- full name, department,
institution, street, city, state, postcode, country
([INSTITUTIONAL E-MAIL TO BE COMPLETED])]}}
\correspondingauthor{{[CORRESPONDING AUTHOR TO BE COMPLETED]}}{{[INSTITUTIONAL E-MAIL TO BE COMPLETED]}}

% Present because agujournaltemplate.tex carries them.  The class defines both as
% no-ops in either branch; AGU sets the running heads itself.
\authorrunninghead{{}}
\titlerunninghead{{}}

% Three-argument macro in agujournal2025.cls, not the 2019 ``keypoints''
% environment.  All three slots are always supplied; ``{{}}'' would mark an
% unused one, but the canonical Markdown is required to carry exactly three.
\keypoints%
{keypoints}

% Required by agujournaltemplate.tex after the front matter.  It is a no-op in
% the submission branch and typesets the Wiley title page under ``published''.
\maketitle

\begin{{abstract}}
{abstract_tex}

% agujournal2025.cls provides a native Plain Language Summary environment, nested
% inside the abstract.  This replaces the \section*{{Plain Language Summary}}
% workaround the 2019 class required.  No abstract body may follow it.
\begin{{plainlanguagesummary}}
{plain_language_tex}
\end{{plainlanguagesummary}}
\end{{abstract}}

\noindent\textbf{{Keywords:}} {keywords_tex}

{body_tex}

% BIBLIOGRAPHY.  agujournal2025.cls loads apacite and sets
% \bibliographystyle{{apacite}} itself, so no \bibliographystyle is emitted here
% and no separate .bst has to be vendored.
%
% TODO(AUTHORS, docs/WRR_SUBMISSION_CHECKLIST.md item 6.4): the canonical
% Markdown still carries its citations as author-year prose, so this TeX contains
% no \cite or \citeA key.  \nocite{{*}} is a deliberate, temporary bridge: it
% emits the full reference list from ../references.bib, which checklist items
% 6.1 and 6.2 verified to be reconciled one-to-one with the in-text citation
% list, so every printed entry is in fact cited in the prose.  It is not a
% substitute for the citation-convention conversion, and it must be deleted in
% the same change that introduces real \cite/\citeA keys.
\nocite{{*}}
\bibliography{{../references}}

\end{{document}}
"""
    # Reviewer-layout pass: cap main-text figures so they can share a page
    # with prose (0.46 textheight) and keep their natural aspect ratio.
    rendered = re.sub(
        r"(\\pandocbounded\{\\includegraphics\[)(keepaspectratio,alt=)",
        r"\1width=\\linewidth,height=0.46\\textheight,\2",
        rendered,
    )
    rendered = rendered.replace("\\begin{figure}", "\\begin{figure}[!t]")
    _assert_unicode_is_declared(rendered)
    return rendered


def _write_create_or_replace(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o644)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify checked-in TeX bytes without modifying the repository",
    )
    args = parser.parse_args()
    # The manuscript is a conventional comparative 2021--2023 holdout study and
    # is edited normally.  This builder only converts the canonical Markdown; a
    # future submission view that fills the ``<<...>>`` placeholders from
    # ``outputs/conventional/holdout_metrics_2021_2023.csv`` will use a separate
    # renderer.
    rendered = _render(MARKDOWN.read_text(encoding="utf-8"))
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != rendered:
            raise SystemExit("ThermoRoute_WRR.tex is stale; run build_agu.py")
        print("AGU TeX is current and contains no withdrawn claim")
        return
    _write_create_or_replace(OUTPUT, rendered)
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({len(rendered.encode('utf-8'))} bytes)")


if __name__ == "__main__":
    main()

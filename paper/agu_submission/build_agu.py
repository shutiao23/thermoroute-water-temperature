#!/usr/bin/env python3
"""Build the pre-opening AGU manuscript from the canonical Markdown.

The Markdown manuscript is the only prose source.  This generator deliberately
contains no empirical result sentence: front matter is status-safe, the body is
converted from the current Markdown, and known withdrawn claims are rejected
before any TeX is written.  ``--check`` is read-only and fails when the checked-in
TeX is not exactly what this source would generate.
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

REQUIRED_STATUS_TEXT = (
    "no current performance result",
    "no empirical performance conclusion",
)

KEYPOINTS = (
    "A frozen protocol separates development, model freeze, predictor evidence, and one-time outcome opening",
    "Primary comparisons use identical station/date/horizon keys and whole-HUC2 clustered inference",
    "This pre-opening draft reports methods and limitations; current empirical results are intentionally absent",
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
        "--from=gfm-raw_html",
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


def _validate_markdown(markdown: str) -> None:
    if markdown.count("<!-- ROUTE_A_CLAIM ") != 8:
        raise ValueError("canonical Markdown must contain exactly eight scope claims")
    folded = markdown.casefold()
    if not all(text in folded for text in REQUIRED_STATUS_TEXT):
        raise ValueError("canonical Markdown does not state its pre-opening result status")
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
    """Render Pandoc code spans as breakable, non-linking URL-style text."""
    pattern = re.compile(r"\\texttt\{([^{}]+)\}")

    def replace(match: re.Match[str]) -> str:
        payload = match.group(1)
        payload = payload.replace(r"\_", r"\_\allowbreak{}")
        payload = payload.replace("/", r"/\allowbreak{}")
        return rf"\texttt{{{payload}}}"

    rendered = pattern.sub(replace, latex)
    return rendered.replace(
        "station/date/horizon", r"station/\allowbreak date/\allowbreak horizon"
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
        bounded_columns = columns[:-1] + r">{\raggedright\arraybackslash}X"
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

    return pattern.sub(replace, latex)


def _render(markdown: str) -> str:
    _validate_markdown(markdown)
    title = _extract(markdown, r"^#\s+(.+?)$", label="title")
    abstract = _extract(
        markdown,
        r"^## Abstract\s*\n(.*?)(?=^## 1\.)",
        label="abstract",
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
    body_tex = _fit_longtables(
        _make_code_spans_breakable(_convert(body, shift_headings=-1))
    )

    keypoints = "\n".join(f"\\item {_latex_escape(item)}" for item in KEYPOINTS)
    return rf"""\documentclass[draft]{{agujournal2019}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs,longtable,array,tabularx}}
\usepackage{{url,xurl}}
\usepackage{{hyperref}}
\providecommand{{\tightlist}}{{\setlength{{\itemsep}}{{0pt}}\setlength{{\parskip}}{{0pt}}}}
\setlength{{\emergencystretch}}{{3em}}
\sloppy
% Pandoc's longtable wrapper asks for a caption type even when the Markdown
% table has no caption; AGU leaves that type as ``none``.  Define only the
% otherwise-missing counter so the official class can typeset the table.
\makeatletter
\@ifundefined{{c@none}}{{\newcounter{{none}}}}{{}}
\providecommand{{\thenone}}{{\arabic{{none}}}}
\providecommand{{\fnum@none}}{{Table~\thenone}}
\makeatother
\DeclareUnicodeCharacter{{00B0}}{{\ensuremath{{^\circ}}}}
\DeclareUnicodeCharacter{{00B1}}{{\ensuremath{{\pm}}}}
\DeclareUnicodeCharacter{{00B2}}{{\textsuperscript{{2}}}}
\DeclareUnicodeCharacter{{00B3}}{{\textsuperscript{{3}}}}
\DeclareUnicodeCharacter{{00B7}}{{\ensuremath{{\cdot}}}}
\DeclareUnicodeCharacter{{00D7}}{{\ensuremath{{\times}}}}
\DeclareUnicodeCharacter{{2013}}{{--}}
\DeclareUnicodeCharacter{{2014}}{{---}}
\DeclareUnicodeCharacter{{207B}}{{\textsuperscript{{-}}}}
\DeclareUnicodeCharacter{{2212}}{{\ensuremath{{-}}}}
\DeclareUnicodeCharacter{{2264}}{{\ensuremath{{\leq}}}}
\DeclareUnicodeCharacter{{2265}}{{\ensuremath{{\geq}}}}

\journalname{{Water Resources Research}}

\begin{{document}}

\title{{{_latex_escape(title)}}}

\authors{{[Verified authors and affiliations required before submission]}}
\affiliation{{1}}{{[Department / Laboratory, Institution, City, Country]}}
\correspondingauthor{{[Verified corresponding author]}}{{[verified.email@example.org]}}

\begin{{keypoints}}
{keypoints}
\end{{keypoints}}

\begin{{abstract}}
{abstract_tex}
\end{{abstract}}

{body_tex}

\end{{document}}
"""


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

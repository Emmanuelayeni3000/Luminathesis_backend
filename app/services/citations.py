"""
Citation grounding checks.

LLMs invent plausible-looking citations. This module finds the citations in a
piece of generated text and verifies each one against what was actually
retrieved, so fabricated ones can be flagged (fed back to the editor) or
stripped before the user sees them.

Two citation styles are understood:
  * Author-year:   (Smith, 2020)  [Smith et al., 2020]  (Smith & Jones, 2019; Lee, 2021)
                   Supported if the surname appears near that year in the reference text.
  * Source-style:  (Source: report.docx)
                   Supported if it matches a known source name.
"""
import re
from typing import Iterable, Optional

# Any parenthesised/bracketed span; contents are classified in the callback.
_GROUP = re.compile(r"[\(\[]([^()\[\]]{3,200})[\)\]]")

_SURNAME = r"[A-Z][A-Za-z'’\-]+"
_AUTHOR_YEAR = re.compile(
    rf"""^\s*(?:(?:see|e\.g\.,?|cf\.)\s+)?
        (?P<surname>{_SURNAME})
        (?:\s+et\s+al\.?|(?:\s*(?:&|and)\s*{_SURNAME})+)?
        ,?\s+(?P<year>(?:19|20)\d{{2}})[a-z]?
        (?:,?\s*pp?\.\s*\d+(?:\s*[-–]\s*\d+)?)?\s*$""",
    re.VERBOSE,
)
_SOURCE = re.compile(r"^\s*Source:\s*(?P<name>.+?)\s*$", re.IGNORECASE)

# How close (in characters) the year must be to the surname in the reference text.
_PROXIMITY = 160


def _author_year_supported(surname: str, year: str, reference: str) -> bool:
    for m in re.finditer(rf"\b{re.escape(surname)}\b", reference, re.IGNORECASE):
        window = reference[max(0, m.start() - _PROXIMITY): m.end() + _PROXIMITY]
        if year in window:
            return True
    return False


def _source_supported(name: str, known_sources: Iterable[str]) -> bool:
    n = name.strip().lower()
    for s in known_sources:
        s = s.strip().lower()
        if n == s or n in s or s in n:
            return True
    return False


def _process(
    text: str,
    reference_text: str,
    known_sources: Optional[Iterable[str]],
    strip: bool,
) -> tuple[str, list[str]]:
    unsupported: list[str] = []
    sources = list(known_sources) if known_sources is not None else None

    def handle(match: re.Match) -> str:
        inner = match.group(1)

        src = _SOURCE.match(inner)
        if src:
            # Only verifiable when we were told which sources exist.
            if sources is None or _source_supported(src.group("name"), sources):
                return match.group(0)
            unsupported.append(match.group(0))
            return "" if strip else match.group(0)

        parts = inner.split(";")
        kept: list[str] = []
        bad = False
        for part in parts:
            ay = _AUTHOR_YEAR.match(part)
            if not ay:
                kept.append(part)  # not a citation (e.g. "see Figure 2") — leave alone
                continue
            if not reference_text or _author_year_supported(ay.group("surname"), ay.group("year"), reference_text):
                kept.append(part)
            else:
                bad = True
                unsupported.append(part.strip())
        if not bad:
            return match.group(0)
        if not strip:
            return match.group(0)
        if not kept:
            return ""
        open_c, close_c = match.group(0)[0], match.group(0)[-1]
        return f"{open_c}{';'.join(kept).strip()}{close_c}"

    result = _GROUP.sub(handle, text)
    if strip and unsupported:
        # Tidy whitespace/punctuation left behind by removed citations.
        result = re.sub(r"[ \t]+([.,;:])", r"\1", result)
        result = re.sub(r"[ \t]{2,}", " ", result)
    return result, unsupported


def find_unsupported_citations(
    text: str,
    reference_text: str = "",
    known_sources: Optional[Iterable[str]] = None,
) -> list[str]:
    """Citations in `text` that are not backed by `reference_text` / `known_sources`.

    With an empty `reference_text`, author-year citations are not judged
    (there is nothing to check them against).
    """
    return _process(text, reference_text, known_sources, strip=False)[1]


def strip_unsupported_citations(
    text: str,
    reference_text: str = "",
    known_sources: Optional[Iterable[str]] = None,
) -> tuple[str, list[str]]:
    """Return (cleaned_text, removed_citations)."""
    return _process(text, reference_text, known_sources, strip=True)

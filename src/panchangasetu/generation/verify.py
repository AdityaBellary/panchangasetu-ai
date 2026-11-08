"""Post-generation checks: the answer may only contain facts that are in the cited sources.

Checked mechanically (no second LLM call):
  - citations refer to sources that were actually provided;
  - every clock time and calendar date in the answer appears in the cited sources (times are
    compared as minutes-of-day, so "10:29 AM", "ఉదయం 10:29", "ఉ.10-29" and "10:29" all match).
"""

from __future__ import annotations

import contextlib
import re
import unicodedata
from datetime import date

_TIME_RE = re.compile(r"(?<!\d)(\d{1,2})[:.\-](\d{2})(?!\d)\s*(am|pm|a\.m\.|p\.m\.)?", re.I)
_ISO_DATE_RE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
_DMY_RE = re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](20\d{2})\b")
_MONTHS = {
    m: i
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1
    )
}
_TE_MONTHS = {
    m: i
    for i, m in enumerate(
        ["జనవరి", "ఫిబ్రవరి", "మార్చి", "ఏప్రిల్", "మే", "జూన్", "జూలై", "ఆగస్టు", "సెప్టెంబర్", "అక్టోబర్", "నవంబర్", "డిసెంబర్"], 1
    )
}
_EN_DAY_MONTH = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?(?:\s+(20\d{2}))?",
    re.I,
)
_EN_MONTH_DAY = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(20\d{2}))?",
    re.I,
)
_TE_MONTH_ALT = "|".join(_TE_MONTHS)
_TE_DAY_MONTH = re.compile(r"(\d{1,2})\s*(" + _TE_MONTH_ALT + r")")
# Month first, as the book writes it: "2026 జూన్ 1 వరకు", "డిసెంబర్ 5వ తేదీ", "జూన్2 నుండి".
_TE_MONTH_DAY = re.compile(r"(?<![\u0C00-\u0C7F])(" + _TE_MONTH_ALT + r")\s*(\d{1,2})(?!\d)")


def _minutes_variants(h: int, m: int, ampm: str | None) -> set[int]:
    """A 12-hour clock reading is ambiguous without am/pm (sources print "ప.3-13")."""
    if ampm:
        pm = ampm.lower().startswith("p")
        h24 = (h % 12) + (12 if pm else 0)
        return {h24 * 60 + m}
    return {(h % 12) * 60 + m, ((h % 12) + 12) * 60 + m}


def times_in(text: str) -> list[tuple[str, set[int]]]:
    out = []
    for m in _TIME_RE.finditer(text):
        h, mi = int(m.group(1)), int(m.group(2))
        if h > 24 or mi > 59:
            continue
        out.append((m.group(0).strip(), _minutes_variants(h, mi, m.group(3))))
    return out


def dates_in(text: str, default_year: int) -> set[date]:
    found: set[date] = set()

    def add(y: int, mo: int, d: int) -> None:
        with contextlib.suppress(ValueError):  # e.g. "31 Feb" in free text
            found.add(date(y, mo, d))

    for m in _ISO_DATE_RE.finditer(text):
        add(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    for m in _DMY_RE.finditer(text):
        add(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    for m in _EN_DAY_MONTH.finditer(text):
        add(int(m.group(3) or default_year), _MONTHS[m.group(2).lower()[:3]], int(m.group(1)))
    for m in _EN_MONTH_DAY.finditer(text):
        add(int(m.group(3) or default_year), _MONTHS[m.group(1).lower()[:3]], int(m.group(2)))
    for m in _TE_DAY_MONTH.finditer(text):
        add(default_year, _TE_MONTHS[m.group(2)], int(m.group(1)))
    for m in _TE_MONTH_DAY.finditer(text):
        add(default_year, _TE_MONTHS[m.group(1)], int(m.group(2)))
    return found


_NUMBER_RE = re.compile(r"(?<![\d:.\-])(\d{1,3})(?:st|nd|rd|th)?(?![\d:.\-])")


def numbers_in(text: str) -> set[int]:
    """Standalone numbers and ordinals ("11th", "2", "12") - not parts of times or dates."""
    return {int(m.group(1)) for m in _NUMBER_RE.finditer(text)}


_TE_ORDINALS = {
    "ప్రథమ": 1, "మొదటి": 1, "ద్వితీయ": 2, "రెండవ": 2, "రెండో": 2, "తృతీయ": 3, "మూడవ": 3, "చతుర్థ": 4,
    "నాల్గవ": 4, "నాలుగవ": 4, "పంచమ": 5, "ఐదవ": 5, "షష్ఠ": 6, "షష్ట": 6, "ఆరవ": 6, "సప్తమ": 7, "ఏడవ": 7,
    "అష్టమ": 8, "ఎనిమిదవ": 8, "నవమ": 9, "తొమ్మిదవ": 9, "దశమ": 10, "పదవ": 10, "ఏకాదశ": 11, "పదకొండవ": 11,
    "ద్వాదశ": 12, "పన్నెండవ": 12, "పండ్రెండవ": 12,
}  # fmt: skip


# Large round numbers are usually written as words in the book ("వర్ష శతమేవ" = 100 years).
_TE_CARDINALS = {"శత": 100, "వంద": 100, "నూరు": 100, "సహస్ర": 1000, "వెయ్యి": 1000, "లక్ష": 100_000, "కోటి": 10**7}


def telugu_ordinals_in(text: str) -> set[int]:
    return {n for word, n in (_TE_ORDINALS | _TE_CARDINALS).items() if word in text}


def _without_dates(text: str) -> str:
    """Dates removed, so "2027-01-15" or "15-01-2027" can't be read as the time 01:15."""
    return _DMY_RE.sub(" ", _ISO_DATE_RE.sub(" ", text))


def check_answer(answer: str, cited_texts: list[str], *, default_year: int) -> list[str]:
    """Problems found (empty = passed)."""
    problems: list[str] = []
    answer = strip_citation_markers(answer)
    source = "\n".join(cited_texts)
    # Every number in the answer must occur in the sources (catches invented counts, houses, padas).
    # Sources often spell positions as Telugu ordinal words (దశమ = 10th), which count too.
    source_numbers = {int(n) for n in re.findall(r"\d+", source)} | telugu_ordinals_in(source)
    for n in sorted(numbers_in(_TIME_RE.sub(" ", answer)) - source_numbers):
        if n > 1:  # "1" is too common in prose ("one of", list markers) to be a useful signal
            problems.append(f"number {n} is not in the cited sources")
    source_minutes: set[int] = set()
    for _, variants in times_in(_without_dates(source)):
        source_minutes |= variants
    for raw, variants in times_in(_without_dates(answer)):
        if not variants & source_minutes:
            problems.append(f"time '{raw}' is not in the cited sources")
    source_dates = dates_in(source, default_year) | dates_in(source, default_year + 1)
    for d in dates_in(answer, default_year):
        if d not in source_dates and d.replace(year=d.year + 1) not in source_dates:
            problems.append(f"date {d.isoformat()} is not in the cited sources")
    return problems


def _squash(text: str) -> str:
    """Comparison form for quotes: NFC, no joiners, and no spaces or punctuation at all. Telugu word
    boundaries vary between OCR runs ("ముక్కుమీద" / "ముక్కు మీద"), so spacing carries no signal here."""
    text = unicodedata.normalize("NFC", text).replace("\u200c", "").replace("\u200d", "")
    return re.sub(r"[\s\u00a0|*#\[\]()\"'“”‘’.,;:!?-]+", "", text).lower()


CITATION_MARKER_RE = re.compile(r"\s*[\[(]?\bS\d{1,2}\b[\])]?")
# "According to [S1], …" loses its object once the marker goes; drop the lead-in with it.
_LEAD_IN_RE = re.compile(
    r"\b(?:according to|as per|as stated in|as mentioned in)\s*(?:the\s+)?[\[(]?S\d{1,2}[\])]?,?\s*", re.I
)


def strip_citation_markers(text: str) -> str:
    """Models sometimes write "(S3)" inline; citations are returned separately, so drop them
    (and never let "S2" be read as the number 2)."""
    text = _LEAD_IN_RE.sub("", text)
    text = re.sub(r"\s+([.,;:])", r"\1", CITATION_MARKER_RE.sub("", text)).strip()
    return text[:1].upper() + text[1:] if text[:1].isascii() else text


def _grams(text: str, n: int = 4) -> set[str]:
    return {text[i : i + n] for i in range(max(1, len(text) - n + 1))}


def quote_support(quote: str, source: str, *, threshold: float = 0.85) -> bool:
    """Verbatim, or near-verbatim (>= 85% of the quote's character 4-grams occur in the source).
    Tolerates OCR spelling variants and small normalisations; rejects invented sentences."""
    q, s = _squash(quote), _squash(source)
    if q in s:
        return True
    grams = _grams(q)
    return len(grams & _grams(s)) / len(grams) >= threshold


def check_quotes(
    quotes: list[tuple[str, str]], sources: dict[str, str]
) -> tuple[list[str], dict[str, list[str]]]:
    """Every quote must be supported by some provided source (models often cite the wrong id).
    Returns (problems, {source_id: [quotes it supports]})."""
    if not quotes:
        return ["no supporting quotes"], {}
    problems: list[str] = []
    supported: dict[str, list[str]] = {}
    for sid, quote in quotes:
        if len(_squash(quote)) < 8:
            problems.append(f"quote is too short to verify: {quote!r}")
            continue
        order = [sid, *(k for k in sources if k != sid)] if sid in sources else list(sources)
        match = next((k for k in order if quote_support(quote, sources[k])), None)
        if match is None:
            problems.append(f"quote is not in any source: {quote[:60]!r}")
        else:
            supported.setdefault(match, []).append(quote)
    return problems, supported

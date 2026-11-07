"""Telugu-aware text utilities: script statistics, normalisation, grapheme-safe splitting.

Kept dependency-light and pure so it can be reused by ingestion, query-time language
detection and evaluation.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import regex

TELUGU_BLOCK = (0x0C00, 0x0C7F)
PUA_RANGES = ((0xE000, 0xF8FF), (0xF0000, 0xFFFFD), (0x100000, 0x10FFFD))

_TELUGU_DIGITS = {ord(c): str(i) for i, c in enumerate("౦౧౨౩౪౫౬౭౮౯")}

# Zero-width / formatting characters that OCR engines and PDFs emit spuriously.
# ZWJ (U+200D) and ZWNJ (U+200C) are deliberately kept: they change Telugu conjunct rendering.
_STRIP_CHARS = dict.fromkeys(map(ord, "﻿­​⁠᠎"), None)

_WS_RE = regex.compile(r"[ \t  -   　]+")
_MULTI_NL_RE = regex.compile(r"\n{3,}")
_GRAPHEME_RE = regex.compile(r"\X")

# Sentence terminators: Latin punctuation, danda/double danda (used in slokas), and '॥'.
_SENTENCE_RE = regex.compile(r"(?<=[.!?।॥])\s+|(?<=\n)\s*\n+")


@dataclass(frozen=True, slots=True)
class ScriptStats:
    telugu: int
    latin: int
    digits: int
    pua: int  # private-use code points: signature of legacy (non-Unicode) font encodings
    other_letters: int

    @property
    def letters(self) -> int:
        return self.telugu + self.latin + self.pua + self.other_letters

    @property
    def telugu_ratio(self) -> float:
        known = self.telugu + self.latin
        return self.telugu / known if known else 0.0

    @property
    def valid_ratio(self) -> float:
        """Share of letter-like code points that are real Unicode Telugu or Latin."""
        return (self.telugu + self.latin) / self.letters if self.letters else 0.0


def _is_pua(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in PUA_RANGES)


def script_stats(text: str) -> ScriptStats:
    telugu = latin = digits = pua = other = 0
    for ch in text:
        cp = ord(ch)
        if TELUGU_BLOCK[0] <= cp <= TELUGU_BLOCK[1]:
            if ch.isdigit():
                digits += 1
            else:
                telugu += 1
        elif _is_pua(cp):
            pua += 1
        elif ch.isascii():
            if ch.isalpha():
                latin += 1
            elif ch.isdigit():
                digits += 1
        elif ch.isalpha():
            other += 1
    return ScriptStats(telugu=telugu, latin=latin, digits=digits, pua=pua, other_letters=other)


def normalize_telugu(text: str, *, ascii_digits: bool = True) -> str:
    """Canonical form used for indexing and matching.

    - NFC (OCR engines differ in whether they emit precomposed vowel signs, e.g. ై)
    - Telugu digits -> ASCII digits (so "౧౫" and "15" match)
    - drop spurious zero-width characters, collapse horizontal whitespace
    - trim trailing spaces per line and runs of blank lines
    """
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_STRIP_CHARS)
    if ascii_digits:
        text = text.translate(_TELUGU_DIGITS)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


_VIRAMA = "్"


def _is_telugu_consonant(ch: str) -> bool:
    return "క" <= ch <= "హ" or "ౘ" <= ch <= "ౚ"


def graphemes(text: str) -> list[str]:
    """Telugu-safe user-perceived characters (aksharas).

    Starts from extended grapheme clusters, then joins a cluster ending in virama (్) with the
    following consonant cluster. The `regex` module predates Unicode 15.1's conjunct rule
    (GB9c), so plain `\\X` would split "స్వా" into "స్" + "వా", cutting a conjunct in half.
    """
    out: list[str] = []
    for cluster in _GRAPHEME_RE.findall(text):
        if out and out[-1].endswith(_VIRAMA) and _is_telugu_consonant(cluster[0]):
            out[-1] += cluster
        else:
            out.append(cluster)
    return out


def split_sentences(text: str) -> list[str]:
    parts = (p.strip() for p in _SENTENCE_RE.split(text))
    return [p for p in parts if p]


def truncate_graphemes(text: str, max_graphemes: int) -> str:
    gs = graphemes(text)
    return text if len(gs) <= max_graphemes else "".join(gs[:max_graphemes])

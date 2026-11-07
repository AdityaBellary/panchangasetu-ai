"""Question language detection: Telugu script, English, mixed, and romanised Telugu.

Answer-language policy (requirement §5): the dominant language of the question.
  - Telugu script                     -> answer in Telugu
  - English (even with Telugu terms
    such as "nakshatram", "tithi")    -> answer in English
  - Romanised Telugu ("ee roju tithi enti?") -> answer in Telugu
Detection is local and free; no provider call is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import regex

from panchangasetu.telugu.text import script_stats

# Function words and common verbs of Telugu as typed in Latin script. Domain terms (tithi,
# nakshatram) are deliberately absent: English speakers use them too.
_ROMANISED_TELUGU = frozenset(
    """
    ee aa roju repu ninna ellundi monna enti emiti emi eppudu ela elaa enduku evaru ekkada entha
    enni unda undi unnayi vunda vundi ledu kadu avunu cheppandi cheppu chepandi chudandi naaku naku
    maa mana meeru miru nenu memu vachhe vachche vache vasthundi vastundi ayindi avutundi cheyali
    cheyachha cheyavacha cheyocha manchidi manchi chedu nela varam vaaram samvatsaram lo ki ni tho
    nunchi varaku gurinchi ante antha kuda kooda inka mari
    """.split()
)
_WORD_RE = regex.compile(r"[a-z]+")


@dataclass(frozen=True, slots=True)
class Language:
    code: str  # "te" | "en": the language to answer in
    script: str  # "Telu" | "Latn" | "mixed"
    romanised_telugu: bool
    telugu_ratio: float  # share of letters that are Telugu script


def detect_language(text: str, *, romanised_threshold: float = 0.25) -> Language:
    st = script_stats(text)
    ratio = st.telugu_ratio
    if ratio >= 0.5:
        return Language("te", "Telu" if ratio > 0.9 else "mixed", False, ratio)
    words = _WORD_RE.findall(text.lower())
    romanised = sum(w in _ROMANISED_TELUGU for w in words)
    if words and romanised / len(words) >= romanised_threshold and romanised >= 2:
        return Language("te", "Latn", True, ratio)
    return Language("en", "Latn" if ratio < 0.1 else "mixed", False, ratio)

"""Cross-language query expansion (free, deterministic).

English and romanised-Telugu questions get the Telugu terms the documents actually use appended:
from the curated glossary (config/query_glossary.yaml) and from the panchanga lexicon (tithi and
nakshatra names). Telugu-script questions pass through unchanged. An LLM rewriter can replace this
behind the same function signature later (Phase 6 adds conversation-aware rewriting).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from panchangasetu.query.language import Language
from panchangasetu.query.slots import Slots


@dataclass(frozen=True, slots=True)
class Glossary:
    entries: dict[str, list[str]]

    @classmethod
    def load(cls, path: Path | None) -> Glossary:
        if path is None or not path.is_file():
            return cls({})
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls({str(k).lower(): [str(v) for v in vals] for k, vals in raw.items()})

    def lookup(self, text: str) -> list[str]:
        low = text.lower()
        hits: list[str] = []
        # Longest phrases first so "shukla paksha" wins over "paksha".
        for phrase in sorted(self.entries, key=len, reverse=True):
            if re.search(rf"(?<![a-z]){re.escape(phrase)}s?(?![a-z])", low):
                hits += self.entries[phrase]
                low = low.replace(phrase, " ")
        return hits


def expand_query(text: str, language: Language, slots: Slots, glossary: Glossary) -> str:
    if language.script == "Telu":
        return text
    terms: list[str] = glossary.lookup(text)
    for name in (slots.tithi, slots.nakshatra):
        if name:
            terms.append(name)
    unique = list(dict.fromkeys(t for t in terms if t not in text))
    return f"{text} {' '.join(unique)}" if unique else text

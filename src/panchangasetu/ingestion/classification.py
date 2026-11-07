"""Assign each page to a knowledge collection (and choose its OCR engine).

Precedence:
  1. Explicit rules from `config/collections.yaml` (document glob + page ranges). These encode
     curator knowledge about a publication's structure and always win.
  2. A heuristic on the extracted text: density of panchanga vocabulary and time stamps.

Classification is per page, not per document: one TTD panchangam book contains both daily
calendar tables (Panchanga) and stotras / rasi phalalu / muhurtha rules (General).
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from panchangasetu.config import OCREngineName
from panchangasetu.core.types import Classification, Collection
from panchangasetu.telugu.lexicon import calendar_signal


def parse_page_spec(spec: str | int) -> set[int]:
    """'1-3, 7, 10-12' -> {1, 2, 3, 7, 10, 11, 12}."""
    pages: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            if lo > hi:
                raise ValueError(f"bad page range {part!r}")
            pages.update(range(lo, hi + 1))
        else:
            pages.add(int(part))
    return pages


@dataclass(frozen=True, slots=True)
class PageRule:
    pages: frozenset[int] | None  # None = whole document
    collection: Collection | None
    section_type: str | None
    ocr_engine: OCREngineName | None
    columns: int | None = None  # 2 = OCR the two columns separately (two-column lists/tables)


@dataclass(slots=True)
class DocumentRules:
    match: str
    rules: list[PageRule] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class PagePlan:
    """What the rules say about one page before extraction."""

    collection: Collection | None
    section_type: str | None
    ocr_engine: OCREngineName | None
    columns: int | None = None  # 2 = OCR the two columns separately (two-column lists/tables)


class CollectionRules:
    def __init__(self, documents: list[DocumentRules]) -> None:
        self._documents = documents

    @classmethod
    def load(cls, path: Path | None) -> CollectionRules:
        if path is None or not path.is_file():
            return cls([])
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        docs: list[DocumentRules] = []
        for entry in raw.get("documents", []):
            doc = DocumentRules(match=str(entry["match"]))
            for r in [entry.get("defaults"), *list(entry.get("ranges", []))]:
                if not r:
                    continue
                collection = r.get("collection")
                doc.rules.append(
                    PageRule(
                        pages=frozenset(parse_page_spec(r["pages"])) if "pages" in r else None,
                        collection=Collection(collection) if collection else None,
                        section_type=r.get("section_type"),
                        ocr_engine=r.get("ocr_engine"),
                        columns=int(r["columns"]) if r.get("columns") else None,
                    )
                )
            docs.append(doc)
        return cls(docs)

    def plan(self, source_path: str, page_number: int) -> PagePlan:
        """Later, more specific rules override earlier ones field by field."""
        collection: Collection | None = None
        section_type: str | None = None
        engine: OCREngineName | None = None
        columns: int | None = None
        name = Path(source_path).name
        for doc in self._documents:
            if not (fnmatch.fnmatch(source_path, doc.match) or fnmatch.fnmatch(name, doc.match)):
                continue
            for rule in doc.rules:
                if rule.pages is not None and page_number not in rule.pages:
                    continue
                collection = rule.collection or collection
                section_type = rule.section_type or section_type
                engine = rule.ocr_engine or engine
                columns = rule.columns or columns
        return PagePlan(collection=collection, section_type=section_type, ocr_engine=engine, columns=columns)


# Thresholds tuned on the 176-page TTD panchangam (first full run, 2026-11-07): explanatory
# pages reach 15-35 terms/k but ~0 times/k and dates/k; muhurtham lists, transit tables and
# daily tables have >= 5 times/k; dated schedules (utsavam lists) have >= 10 dates/k.
# Curator rules (config/collections.yaml) take precedence; this is the fallback for new documents.
_TERMS_PER_K = 12.0
_TIMES_PER_K = 4.0
_DATES_PER_K = 3.0
_MIN_DATED_FACTS = 3
_MIN_TEXT_CHARS = 40


def classify_page(text: str, plan: PagePlan) -> Classification:
    if plan.collection is not None:
        return Classification(
            collection=plan.collection,
            section_type=plan.section_type or plan.collection.value,
            confidence=1.0,
            reason="rule",
        )
    stripped = text.strip()
    if len(stripped) < _MIN_TEXT_CHARS:
        return Classification(Collection.SKIP, "empty", 0.9, f"too little text ({len(stripped)} chars)")
    sig = calendar_signal(stripped)
    dated = max(sig["times_per_k"] / _TIMES_PER_K, sig["dates_per_k"] / _DATES_PER_K)
    # Dated facts are required (dense AND at least a few in absolute terms, so a short snippet
    # mentioning one date does not qualify); vocabulary density only adds confidence.
    calendarish = dated >= 1.0 and sig["dated_facts"] >= _MIN_DATED_FACTS
    score = min(1.0, 0.7 * min(dated, 1.0) + 0.3 * min(sig["terms_per_k"] / _TERMS_PER_K, 1.0))
    reason = (
        f"heuristic terms/k={sig['terms_per_k']:.1f} times/k={sig['times_per_k']:.1f} "
        f"dates/k={sig['dates_per_k']:.1f}"
    )
    if calendarish:
        return Classification(
            Collection.PANCHANGA, plan.section_type or "calendar_text", round(score, 3), reason
        )
    return Classification(
        Collection.GENERAL, plan.section_type or "prose", round(1.0 - min(score, 0.99), 3), reason
    )

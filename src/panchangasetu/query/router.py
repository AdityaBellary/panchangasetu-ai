"""Route a question to the right knowledge: structured panchanga facts, panchanga search,
general knowledge, or both.

Two stages, both free:
  1. Rules over extracted slots and dates decide every case they can decide with certainty
     (a date + a panchanga attribute is a lookup; "when/next" + a named tithi is a find).
  2. A semantic router (nearest example questions, local embeddings) decides the rest.
     Below `min_confidence` the route becomes HYBRID: querying both collections is cheaper
     than answering from the wrong one.
An LLM classifier can be slotted in behind `SemanticClassifier` later without changing callers.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import numpy as np
import yaml

from panchangasetu.core.interfaces import Embedder
from panchangasetu.query.dates import DateSpec, parse_dates
from panchangasetu.query.language import Language, detect_language
from panchangasetu.query.slots import DAY_ATTRIBUTES, EVENT_ATTRIBUTES, Slots, extract_slots


class Intent(StrEnum):
    PANCHANGA = "panchanga"  # calendar facts / dated items
    GENERAL = "general"  # explanations, predictions, rules, stotras
    HYBRID = "hybrid"  # both, e.g. "what is Ekadashi and when is the next one?"


class PanchangaMode(StrEnum):
    DAY = "day"  # facts for specific date(s): structured lookup
    FIND = "find"  # next occurrence of a named tithi/nakshatra: structured search
    SEARCH = "search"  # festivals, muhurthams, events: vector search over dated items
    NONE = "none"


@dataclass(slots=True)
class Route:
    intent: Intent
    mode: PanchangaMode
    language: Language
    dates: list[DateSpec]
    slots: Slots
    confidence: float
    reasons: list[str] = field(default_factory=list)

    @property
    def wants_panchanga(self) -> bool:
        return self.intent in (Intent.PANCHANGA, Intent.HYBRID)

    @property
    def wants_general(self) -> bool:
        return self.intent in (Intent.GENERAL, Intent.HYBRID)


class SemanticClassifier(Protocol):
    def classify(self, text: str) -> tuple[str, float]:
        """(label in {"panchanga", "general"}, confidence 0..1)."""
        ...


class EmbeddingRouter:
    """k-nearest example questions (config/router_examples.yaml), weighted by similarity."""

    def __init__(self, embedder: Embedder, examples: dict[str, list[str]], k: int = 7) -> None:
        self.k = k
        self.labels: list[str] = []
        texts: list[str] = []
        for label, items in examples.items():
            self.labels += [label] * len(items)
            texts += items
        self.embedder = embedder
        self.matrix = np.asarray(embedder.embed_documents([f"{t}" for t in texts]), dtype=np.float32)

    @classmethod
    def from_yaml(cls, embedder: Embedder, path: Path, k: int = 7) -> EmbeddingRouter:
        data: dict[str, list[str]] = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(embedder, data, k=k)

    def classify(self, text: str) -> tuple[str, float]:
        q = np.asarray(self.embedder.embed_query(text), dtype=np.float32)
        sims = self.matrix @ q
        top = np.argsort(-sims)[: self.k]
        votes: dict[str, float] = defaultdict(float)
        for i in top:
            votes[self.labels[i]] += float(np.exp(sims[i] * 20))  # sharpen: nearest examples dominate
        label, weight = max(votes.items(), key=lambda kv: kv[1])
        return label, weight / sum(votes.values())


class QueryRouter:
    def __init__(self, semantic: SemanticClassifier | None = None, *, min_confidence: float = 0.65) -> None:
        self.semantic = semantic
        self.min_confidence = min_confidence

    def route(self, text: str, today: date) -> Route:
        lang = detect_language(text)
        dates = parse_dates(text, today)
        slots = extract_slots(text)
        day_attrs = [a for a in slots.attributes if a in DAY_ATTRIBUTES]
        named = slots.tithi or slots.nakshatra
        reasons: list[str] = []

        def route(intent: Intent, mode: PanchangaMode, conf: float, why: str) -> Route:
            reasons.append(why)
            return Route(intent, mode, lang, dates, slots, conf, reasons)

        event_attrs = [a for a in slots.attributes if a in EVENT_ATTRIBUTES]
        festival_named = bool(named and slots.residual)  # "Sri Rama Navami", "రథసప్తమి"

        if dates and event_attrs and not slots.concept:
            return route(Intent.PANCHANGA, PanchangaMode.SEARCH, 0.9, "date/range + festivals/muhurthams")
        if dates and (day_attrs or not slots.attributes) and not (slots.concept or slots.general_topic):
            return route(Intent.PANCHANGA, PanchangaMode.DAY, 0.95, "date + day attribute")
        if named and slots.when and (slots.concept or (slots.weak_concept and slots.multi_clause)):
            return route(Intent.HYBRID, PanchangaMode.FIND, 0.9, "named tithi/nakshatra: meaning + when")
        if named and slots.when and not festival_named and not event_attrs:
            return route(Intent.PANCHANGA, PanchangaMode.FIND, 0.9, "when/next + named tithi/nakshatra")
        if slots.when and (event_attrs or festival_named):
            return route(Intent.PANCHANGA, PanchangaMode.SEARCH, 0.85, "when + festival/muhurtham/karthe")
        if dates and slots.concept and not slots.general_topic:
            return route(Intent.HYBRID, PanchangaMode.DAY, 0.8, "date + significance question")
        if not dates and not slots.when and (slots.concept or slots.weak_concept or slots.general_topic):
            return route(Intent.GENERAL, PanchangaMode.NONE, 0.85, "conceptual question, no date/when")
        if slots.general_topic and not slots.when:
            return route(Intent.GENERAL, PanchangaMode.NONE, 0.8, "general topic (predictions/rules/omens)")

        if not dates and re.search(r"which months?|what months?|ఏ మాస|ఏ నెల|ye masam|ye nela", text, re.I):
            # "Which months are good for X" asks for a rule (General) that dated lists only illustrate.
            return route(
                Intent.HYBRID, PanchangaMode.SEARCH, 0.75, "which-months question: rules + dated lists"
            )

        if self.semantic is None:
            return route(Intent.HYBRID, PanchangaMode.SEARCH, 0.5, "no rule matched; no semantic router")
        label, conf = self.semantic.classify(text)
        reasons.append(f"semantic: {label} ({conf:.2f})")
        if conf < self.min_confidence:
            return route(Intent.HYBRID, PanchangaMode.SEARCH, conf, "low confidence -> both collections")
        if label == "panchanga":
            mode = PanchangaMode.FIND if named and not festival_named else PanchangaMode.SEARCH
            return route(Intent.PANCHANGA, mode, conf, "semantic router")
        return route(Intent.GENERAL, PanchangaMode.NONE, conf, "semantic router")


def route_many(router: QueryRouter, texts: Sequence[str], today: date) -> list[Route]:
    return [router.route(t, today) for t in texts]

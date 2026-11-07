"""Question -> route -> evidence. No generation happens here (that is Phase 5).

PANCHANGA / DAY     exact day facts for each requested date (+ General search for attributes the
                    daily table does not carry, e.g. rahukalam rules)
PANCHANGA / FIND    next occurrences of a named tithi/nakshatra (+ dated festival items)
PANCHANGA / SEARCH  vector search over dated items (+ exact events when a date range is given)
GENERAL             vector search over the General collection
HYBRID              both sides; the generator decides what the evidence supports
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from qdrant_client import models

from panchangasetu.query.dates import Clock
from panchangasetu.query.language import Language
from panchangasetu.query.rewrite import Glossary, expand_query
from panchangasetu.query.router import PanchangaMode, QueryRouter, Route
from panchangasetu.retrieval.evidence import Evidence
from panchangasetu.retrieval.structured import StructuredRetriever
from panchangasetu.retrieval.vector_store import VectorIndex
from panchangasetu.structured.weekday_periods import WeekdayPeriods

# Day attributes the book's daily table does not print; their rules live in General text.
_NOT_IN_DAILY_TABLE = {"rahukalam", "yamagandam", "gulika"}
_ATTRIBUTE_TE = {"rahukalam": "రాహుకాలము రాహుకాలః", "yamagandam": "యమగండము యమకంటక", "gulika": "గుళిక కాలము"}
_WEEKDAY_TE = ("సోమవారము", "మంగళవారము", "బుధవారము", "గురువారము", "శుక్రవారము", "శనివారము", "ఆదివారము")


@dataclass(slots=True)
class RetrievalResult:
    question: str
    route: Route
    search_query: str
    evidence: list[Evidence] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # e.g. "no data for 2028-01-01"
    timings_ms: dict[str, float] = field(default_factory=dict)


class Retriever:
    def __init__(
        self,
        router: QueryRouter,
        vectors: VectorIndex,
        structured: StructuredRetriever,
        clock: Clock,
        glossary: Glossary,
        *,
        top_k: int = 6,
        candidates: int = 30,
        hybrid_min_telugu_ratio: float = 0.5,
        weekday_periods: WeekdayPeriods | None = None,
    ) -> None:
        self.router = router
        self.vectors = vectors
        self.structured = structured
        self.clock = clock
        self.glossary = glossary
        self.top_k = top_k
        self.candidates = candidates
        self.hybrid_min_telugu_ratio = hybrid_min_telugu_ratio
        self.weekday_periods = weekday_periods

    def retrieve(
        self, question: str, *, now: datetime | None = None, language: Language | None = None
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        now = now or self.clock()
        route = self.router.route(question, now.date(), language=language)
        query = expand_query(question, route.language, route.slots, self.glossary)
        result = RetrievalResult(question, route, query)
        result.timings_ms["route"] = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        if route.wants_panchanga:
            self._panchanga(result, now)
        result.timings_ms["panchanga"] = (time.perf_counter() - t1) * 1000
        t2 = time.perf_counter()
        attrs = set(route.slots.attributes)
        missing = attrs & _NOT_IN_DAILY_TABLE
        tabled = missing & (self.weekday_periods.kinds if self.weekday_periods else frozenset())
        if route.mode == PanchangaMode.DAY and tabled and self.weekday_periods is not None:
            # Fixed by weekday: answered from the book's printed weekday table, not by searching.
            for spec in route.dates:
                for n in range(min((spec.end - spec.start).days + 1, 7)):
                    day = spec.start + timedelta(days=n)
                    result.evidence += [self.weekday_periods.evidence(k, day) for k in sorted(tabled)]
            missing = missing - tabled
        if route.mode == PanchangaMode.DAY and missing:
            # The daily table does not print these; their weekday rules live in General text.
            # Search for the rule itself, not the whole question (dates would only add noise).
            weekdays = sorted({_WEEKDAY_TE[s.start.weekday()] for s in route.dates if s.is_day})
            targeted = " ".join([*(_ATTRIBUTE_TE[a] for a in sorted(missing)), *weekdays, *sorted(missing)])
            result.evidence += self._search("general", targeted, limit=3)
        if route.wants_general:
            if route.slots.rasi:
                # Rasi predictions of neighbouring signs share pages and vocabulary; facts from the
                # wrong sign must never reach the answer. Search only the predictions section, with
                # the sign's own section title in the query, then keep only chunks headed by that sign.
                section: list[models.Condition] = [
                    models.FieldCondition(key="section_type", match=models.MatchValue(value="rasi_phalalu"))
                ]
                targeted = f"{route.slots.rasi}రాశి ఫలితములు {query}"
                hits = self._search("general", targeted, limit=40, must=section)
                result.evidence += _only_rasi(hits, route.slots.rasi)[: self.top_k]
            else:
                result.evidence += self._search("general", query, limit=self.top_k)
        result.timings_ms["general"] = (time.perf_counter() - t2) * 1000
        result.evidence = _dedupe(result.evidence)
        result.timings_ms["total"] = (time.perf_counter() - t0) * 1000
        return result

    def _panchanga(self, result: RetrievalResult, now: datetime) -> None:
        route = result.route
        if route.mode == PanchangaMode.DAY:
            for spec in route.dates or []:
                days = self.structured.days(spec.start, spec.end)
                if not days:
                    result.notes.append(f"no panchanga data for {spec.start}..{spec.end}")
                elif spec.kind == "range" and spec.end != days[-1].date:
                    result.notes.append(f"range truncated at {days[-1].date}")
                result.evidence += days
            if not route.dates:
                result.notes.append("no date understood in the question")
        elif route.mode == PanchangaMode.FIND:
            for kind, name in (("tithi", route.slots.tithi), ("nakshatra", route.slots.nakshatra)):
                if name:
                    found = self.structured.next_occurrences(kind, name, now)
                    if not found:
                        result.notes.append(f"no upcoming {kind} {name} within the available data")
                    result.evidence += found
            result.evidence += self._search(
                "panchanga", result.search_query, limit=3, from_date=now.date().isoformat()
            )
        else:  # SEARCH (or HYBRID with a search side)
            for spec in route.dates:
                if spec.kind == "range" or not spec.is_day:
                    result.evidence += self.structured.events(spec.start, spec.end)
            result.evidence += self._search(
                "panchanga", result.search_query, limit=self.top_k, date_range=_range(route)
            )

    def _search(
        self,
        collection: str,
        query: str,
        *,
        limit: int,
        from_date: str | None = None,
        date_range: tuple[int, int] | None = None,
        must: list[models.Condition] | None = None,
    ) -> list[Evidence]:
        must = list(must or [])
        if date_range:
            must.append(
                models.FieldCondition(
                    key="date_int", range=models.Range(gte=date_range[0], lte=date_range[1])
                )
            )
        elif from_date:
            # Upcoming dated items, plus undated chunks (muhurtham lists, transit tables).
            must.append(
                models.Filter(
                    should=[
                        models.FieldCondition(
                            key="date_int", range=models.Range(gte=int(from_date.replace("-", "")))
                        ),
                        models.IsEmptyCondition(is_empty=models.PayloadField(key="date_int")),
                    ]
                )
            )
        hits = self.vectors.search(
            collection,
            query,
            top_k=limit,
            candidates=self.candidates,
            hybrid_min_telugu_ratio=self.hybrid_min_telugu_ratio,
            must=must,
        )
        out = []
        for h in hits:
            p = h.payload or {}
            out.append(
                Evidence(
                    id=str(h.id),
                    kind=str(p.get("kind", "chunk")),
                    collection=collection,
                    text=_with_heading(p),
                    document=str(p.get("document", "")),
                    pages=list(p.get("pages", [])),
                    date=p.get("date"),
                    verified=None if p.get("kind") == "chunk" else True,
                    score=h.score,
                    source_kind=str(p.get("section_type", "")),
                    data={"heading_path": p.get("heading_path", [])},
                )
            )
        return out


def _range(route: Route) -> tuple[int, int] | None:
    ranges = [s for s in route.dates if s.kind == "range"]
    if not ranges:
        return None
    s = ranges[0]
    return int(s.start.strftime("%Y%m%d")), int(s.end.strftime("%Y%m%d"))


def _dedupe(items: list[Evidence]) -> list[Evidence]:
    seen: set[str] = set()
    out = []
    for e in items:
        if e.id not in seen:
            seen.add(e.id)
            out.append(e)
    return out


def _with_heading(payload: dict[str, object]) -> str:
    """Chunk text as shown to the answer model, prefixed with its section heading. A chunk from
    the middle of "కన్యారాశి ఫలితములు" does not repeat the rasi name; without the heading the model
    cannot tell which rasi it is about."""
    text = str(payload.get("text", ""))
    headings = payload.get("heading_path") or []
    if isinstance(headings, list) and headings:
        return f"[Section: {' > '.join(str(h) for h in headings)}]\n{text}"
    return text


def _only_rasi(items: list[Evidence], rasi: str) -> list[Evidence]:
    """Drop rasi-prediction chunks whose section heading names a different sign."""
    from panchangasetu.telugu.lexicon import RASIS

    forms = [rasi, *(a for a in RASIS[rasi] if not a.isascii())]
    kept = []
    for e in items:
        if e.source_kind != "rasi_phalalu":
            continue  # the question is about a sign: other sections only add noise
        heading = " ".join(str(h) for h in e.data.get("heading_path", []))
        if any(f in heading for f in forms):
            kept.append(e)
    return kept

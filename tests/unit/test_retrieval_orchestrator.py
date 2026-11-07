"""Structured retrieval and the orchestrator on SQLite + in-memory Qdrant (no network, no models)."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from panchangasetu.providers.embeddings.telugu_sparse import TeluguSparseEncoder
from panchangasetu.query.rewrite import Glossary
from panchangasetu.query.router import QueryRouter
from panchangasetu.retrieval.orchestrator import Retriever
from panchangasetu.retrieval.structured import StructuredRetriever
from panchangasetu.retrieval.vector_store import VectorIndex
from panchangasetu.storage.db import make_engine, make_session_factory
from panchangasetu.storage.models import (
    Base,
    ChunkRecord,
    Document,
    DocumentVersion,
    PanchangaDay,
    PanchangaEvent,
)
from tests.unit.test_vector_store import FakeEmbedder

NOW = datetime(2026, 11, 7, 9, 0)


def tithi(name: str, ends: str | None, *, verified: bool = True, masa: str = "కార్తీకం") -> dict:
    return {
        "kind": "tithi",
        "name": name,
        "ends_at": ends,
        "printed_end": "x",
        "verified": verified,
        "masa": masa,
        "paksha": "శుద్ధ",
        "adhika": False,
    }


@pytest.fixture
def env(tmp_path: Path):  # type: ignore[no-untyped-def]
    engine = make_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    Base.metadata.create_all(engine)
    sessions = make_session_factory(engine)
    with sessions.begin() as s:
        book = Document(source_path="book.pdf", filename="book.pdf")
        cal = Document(source_path="cal.pdf", filename="cal.pdf")
        s.add_all([book, cal])
        s.flush()
        bv = DocumentVersion(
            document_id=book.id, sha256="b" * 64, size_bytes=1, page_count=200, object_key="b"
        )
        cv = DocumentVersion(document_id=cal.id, sha256="c" * 64, size_bytes=1, page_count=12, object_key="c")
        s.add_all([bv, cv])
        s.flush()

        def day(v, d, kind, elements, page=131):  # type: ignore[no-untyped-def]
            s.add(
                PanchangaDay(
                    day=d,
                    version_id=v.id,
                    source_kind=kind,
                    page_number=page,
                    row_index=0,
                    bbox=[],
                    weekday_printed=None,
                    weekday_check="ok",
                    date_check="ok",
                    sunrise="6-13",
                    sunset="5-41",
                    tamil_day=None,
                    raw_text="",
                    elements=elements,
                    windows=[],
                    notes="",
                    warnings=[],
                    element_count=len(elements),
                    verified_count=sum(e["verified"] for e in elements),
                    extractor_version="t",
                )
            )

        day(bv, date(2026, 11, 7), "book_daily_table", [tithi("త్రయోదశి", "2026-11-07T10:29")])
        day(cv, date(2026, 11, 7), "wall_calendar", [tithi("చతుర్దశి", None, verified=False)])  # loses to book
        day(bv, date(2026, 11, 20), "book_daily_table", [tithi("ఏకాదశి", "2026-11-21T02:04")], page=132)
        day(bv, date(2026, 11, 21), "book_daily_table", [tithi("ఏకాదశి", "2026-11-21T02:04")], page=132)
        day(bv, date(2026, 12, 4), "book_daily_table", [tithi("ఏకాదశి", "2026-12-05T01:36")], page=134)
        day(
            cv, date(2027, 6, 1), "wall_calendar", [tithi("పంచమి", None, verified=False)]
        )  # calendar-only date
        s.add(
            PanchangaEvent(
                day=date(2026, 11, 8),
                version_id=bv.id,
                source_kind="festival_table",
                page_number=159,
                title="",
                details="నరకచతుర్దశి, దీపావళి పండుగ",
                tamil_date=None,
                weekday_printed="ఆది",
                weekday_check="ok",
                raw="",
                extractor_version="e",
            )
        )
        s.add(
            ChunkRecord(
                version_id=bv.id,
                chunker_version="c1",
                chunk_index=0,
                collection="general",
                section_type="panchanga_vignanam",
                heading_path=["రాహుకాలము"],
                page_start=83,
                page_end=83,
                text="రాహుకాలము శుక్రవారము ఉ.10-30 నుండి 12-00",
                token_count=10,
                content_hash="1",
            )
        )
    index = VectorIndex(QdrantClient(":memory:"), sessions, FakeEmbedder(), TeluguSparseEncoder(), prefix="t")
    index.build(chunker_version="c1", gloss=True)
    structured = StructuredRetriever(sessions)
    retriever = Retriever(QueryRouter(None), index, structured, lambda: NOW, Glossary({}))
    return structured, retriever


def test_book_is_preferred_over_wall_calendar(env) -> None:  # type: ignore[no-untyped-def]
    structured, _ = env
    (ev,) = structured.days(date(2026, 11, 7), date(2026, 11, 7))
    assert ev.source_kind == "book_daily_table" and ev.verified is True and "త్రయోదశి" in ev.text


def test_calendar_only_dates_are_flagged_unverified(env) -> None:  # type: ignore[no-untyped-def]
    structured, _ = env
    (ev,) = structured.days(date(2027, 6, 1), date(2027, 6, 1))
    assert ev.verified is False and "unverified" in ev.text.lower()


def test_next_occurrence_skips_past_and_merges_a_tithi_spanning_two_sunrises(env) -> None:  # type: ignore[no-untyped-def]
    structured, _ = env
    found = structured.next_occurrences("tithi", "ఏకాదశి", NOW)
    assert [e.date for e in found] == ["2026-11-20", "2026-12-04"]
    assert found[0].kind == "occurrence" and "2026-11-21 02:04" in found[0].text


def test_day_question_returns_day_facts(env) -> None:  # type: ignore[no-untyped-def]
    _, retriever = env
    r = retriever.retrieve("What is today's tithi?", now=NOW)
    assert r.evidence[0].kind == "day" and r.evidence[0].date == "2026-11-07"


def test_out_of_coverage_date_yields_a_note_and_no_day_evidence(env) -> None:  # type: ignore[no-untyped-def]
    _, retriever = env
    r = retriever.retrieve("What is the tithi on 15 August 2030?", now=NOW)
    assert not [e for e in r.evidence if e.kind == "day"]
    assert any("no panchanga data" in n for n in r.notes)


def test_find_question_returns_next_occurrences(env) -> None:  # type: ignore[no-untyped-def]
    _, retriever = env
    r = retriever.retrieve("వచ్చే ఏకాదశి ఎప్పుడు?", now=NOW)
    assert [e.date for e in r.evidence if e.kind == "occurrence"] == ["2026-11-20", "2026-12-04"]


def test_rahukalam_question_also_fetches_the_weekday_rule_text(env) -> None:  # type: ignore[no-untyped-def]
    _, retriever = env
    r = retriever.retrieve("రాహుకాలము శుక్రవారము", now=NOW)
    kinds = {e.kind for e in r.evidence}
    assert "chunk" in kinds and any(83 in e.pages for e in r.evidence)


def test_festival_range_question_returns_exact_events(env) -> None:  # type: ignore[no-untyped-def]
    _, retriever = env
    r = retriever.retrieve("Festivals this week", now=NOW)
    assert any(e.kind == "event" and e.date == "2026-11-08" for e in r.evidence)

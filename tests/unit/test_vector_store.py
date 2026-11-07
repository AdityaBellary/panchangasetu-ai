"""VectorIndex lifecycle against Qdrant's in-memory mode with a deterministic fake embedder."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from qdrant_client import QdrantClient
from sqlalchemy import select

from panchangasetu.providers.embeddings.telugu_sparse import TeluguSparseEncoder
from panchangasetu.retrieval.vector_store import VectorIndex, alias_name
from panchangasetu.storage.db import make_engine, make_session_factory
from panchangasetu.storage.models import (
    Base,
    ChunkRecord,
    Document,
    DocumentVersion,
    IndexBuild,
    PanchangaEvent,
)


class FakeEmbedder:
    """Bag-of-words hashed into 64 dims: similar texts get similar vectors, deterministically."""

    name = "fake-embedder"
    dim = 64

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dim)
        for tok in text.split():
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(v)
        return (v / n).tolist() if n else v.tolist()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture
def setup(tmp_path: Path):  # type: ignore[no-untyped-def]
    engine = make_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    Base.metadata.create_all(engine)
    sessions = make_session_factory(engine)
    with sessions.begin() as s:
        doc = Document(source_path="book.pdf", filename="book.pdf")
        s.add(doc)
        s.flush()
        v = DocumentVersion(document_id=doc.id, sha256="a" * 64, size_bytes=1, page_count=200, object_key="k")
        s.add(v)
        s.flush()
        rows = [
            ("general", "rasi_phalalu", 31, "కన్యారాశి ఫలితములు ఈ సంవత్సరం శుభం"),
            ("general", "panchanga_vignanam", 92, "గృహప్రవేశము మాఘ వైశాఖ మాసములు ప్రశస్తము"),
            ("panchanga", "shubha_muhurthams", 44, "ఏప్రిల్ 12 వివాహ ముహూర్తం"),
            ("panchanga", "daily_panchangam_table", 120, "noisy table OCR that must not be indexed"),
        ]
        for i, (coll, section, page, text) in enumerate(rows):
            s.add(
                ChunkRecord(
                    version_id=v.id,
                    chunker_version="c1",
                    chunk_index=i,
                    collection=coll,
                    section_type=section,
                    heading_path=[],
                    page_start=page,
                    page_end=page,
                    text=text,
                    token_count=10,
                    content_hash=str(i),
                )
            )
        s.add(
            PanchangaEvent(
                day=date(2026, 12, 20),
                version_id=v.id,
                source_kind="festival_table",
                page_number=161,
                title="",
                details="వైకుంఠ ఏకాదశి ముక్కోటి ఏకాదశి",
                tamil_date=None,
                weekday_printed="ఆది",
                weekday_check="ok",
                raw="",
                extractor_version="e",
            )
        )
    client = QdrantClient(":memory:")
    index = VectorIndex(client, sessions, FakeEmbedder(), TeluguSparseEncoder(), prefix="t", keep_previous=1)
    return index, sessions, client


def test_build_creates_aliased_collections_and_excludes_structured_only_sections(setup) -> None:  # type: ignore[no-untyped-def]
    index, sessions, client = setup
    report = index.build(chunker_version="c1", gloss=True)
    assert report.built == {"t_general": 2, "t_panchanga": 2}  # muhurtham chunk + event; table excluded
    assert {a.alias_name for a in client.get_aliases().aliases} == {"t_general", "t_panchanga"}
    with sessions() as s:
        assert {b.status for b in s.scalars(select(IndexBuild))} == {"active"}


def test_unchanged_corpus_is_skipped_and_force_swaps_alias_atomically(setup) -> None:  # type: ignore[no-untyped-def]
    index, _sessions, client = setup
    index.build(chunker_version="c1", gloss=True)
    first = {a.alias_name: a.collection_name for a in client.get_aliases().aliases}
    assert index.build(chunker_version="c1", gloss=True).skipped == ["t_general", "t_panchanga"]
    index.build(chunker_version="c1", gloss=True, force=True)
    second = {a.alias_name: a.collection_name for a in client.get_aliases().aliases}
    assert first["t_general"] != second["t_general"]
    assert client.collection_exists(first["t_general"])  # kept for rollback (keep_previous=1)
    index.build(chunker_version="c1", gloss=True, force=True)
    assert not client.collection_exists(first["t_general"])  # older than keep_previous: dropped


def test_telugu_query_uses_hybrid_and_finds_the_right_chunk(setup) -> None:  # type: ignore[no-untyped-def]
    index, _, _ = setup
    index.build(chunker_version="c1", gloss=True)
    hits = index.search("general", "కన్యారాశి ఫలితములు", top_k=1)
    assert hits[0].payload["page_start"] == 31 and hits[0].payload["kind"] == "chunk"


def test_event_documents_carry_their_exact_date(setup) -> None:  # type: ignore[no-untyped-def]
    index, _, _ = setup
    index.build(chunker_version="c1", gloss=True)
    hits = index.search("panchanga", "వైకుంఠ ఏకాదశి", top_k=2)
    event = next(h for h in hits if h.payload["kind"] == "event")
    assert event.payload["date"] == "2026-12-20" and event.payload["pages"] == [161]
    assert uuid.UUID(str(event.id))


def test_search_filters_by_payload(setup) -> None:  # type: ignore[no-untyped-def]
    from qdrant_client import models

    index, _, _ = setup
    index.build(chunker_version="c1", gloss=True)
    must = [models.FieldCondition(key="section_type", match=models.MatchValue(value="panchanga_vignanam"))]
    hits = index.search("general", "కన్యారాశి", top_k=5, must=must)
    assert hits and all(h.payload["section_type"] == "panchanga_vignanam" for h in hits)


def test_alias_name() -> None:
    assert alias_name("psetu", "general") == "psetu_general"

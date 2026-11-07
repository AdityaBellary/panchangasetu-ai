"""Pipeline lifecycle tests on SQLite with a fake OCR engine (no network, no Postgres)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from panchangasetu.config import Settings
from panchangasetu.ingestion.classification import CollectionRules
from panchangasetu.ingestion.pipeline import IngestionPipeline
from panchangasetu.storage.db import make_engine, make_session_factory
from panchangasetu.storage.models import Base, ChunkRecord, Document, DocumentVersion, IngestionRun, Page
from panchangasetu.storage.object_store import LocalObjectStore
from tests.conftest import LEGACY_TEXT, FakeOCREngine, make_pdf

ENGLISH = "Unicode text layer page about festivals and their significance. " * 3


@pytest.fixture
def env(tmp_path: Path) -> dict[str, object]:
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    rules = tmp_path / "collections.yaml"
    rules.write_text(
        'documents:\n  - match: "book.pdf"\n'
        '    ranges:\n      - {pages: "1", collection: skip, section_type: cover}\n',
        encoding="utf-8",
    )
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        database_url=f"sqlite:///{tmp_path / 'db.sqlite'}",
        input_dir=input_dir,
        object_store_dir=tmp_path / "objects",
        collections_rules_path=rules,
        ocr_engine="sarvam",
        ocr_fallback_engine="tesseract",
        ocr_batch_pages=2,
        render_dpi=72,
        min_image_short_side_px=0,
    )
    engine = make_engine(settings.database_url)
    Base.metadata.create_all(engine)
    return {"settings": settings, "sessions": make_session_factory(engine), "input": input_dir}


def build(
    env: dict[str, object], primary: FakeOCREngine, fallback: FakeOCREngine | None = None
) -> IngestionPipeline:
    settings: Settings = env["settings"]  # type: ignore[assignment]
    engines = {"sarvam": primary, "tesseract": fallback or FakeOCREngine("tesseract")}
    return IngestionPipeline(
        settings,
        env["sessions"],  # type: ignore[arg-type]
        LocalObjectStore(settings.object_store_dir),
        engine_factory=lambda name: engines[name],  # type: ignore[return-value,arg-type]
        rules=CollectionRules.load(settings.collections_rules_path),
    )


def book(input_dir: Path, name: str = "book.pdf") -> Path:
    # p1 cover (skipped by rule), p2-p3 legacy font (OCR), p4 real Unicode text layer, p5 image-only (OCR)
    return make_pdf(
        input_dir / name,
        [{"image": True}, {"text": LEGACY_TEXT}, {"text": LEGACY_TEXT}, {"text": ENGLISH}, {"image": True}],
    )


def count(sessions: sessionmaker[Session], model: type, *where: object) -> int:
    with sessions() as s:
        return int(s.scalar(select(func.count()).select_from(model).where(*where)) or 0)


def test_first_run_extracts_classifies_and_chunks(env: dict[str, object]) -> None:
    book(env["input"])  # type: ignore[arg-type]
    ocr = FakeOCREngine()
    _, stats = build(env, ocr).run()
    assert stats.new_versions == 1 and stats.versions_processed == 1
    assert stats.pages_skipped == 1 and stats.pages_text_layer == 1 and stats.pages_ocr == 3
    assert ocr.calls == [[2, 3], [5]]  # batched by ocr_batch_pages=2, cover never sent
    sessions = env["sessions"]
    with sessions() as s:  # type: ignore[operator]
        v = s.scalar(select(DocumentVersion))
        assert v.status == "completed" and v.page_count == 5
        pages = {p.page_number: p for p in s.scalars(select(Page))}
        assert pages[1].collection == "skip"
        assert pages[2].page_kind == "born_digital_legacy_font" and pages[2].ocr_engine == "sarvam"
        assert pages[4].extraction_method == "text_layer" and "festivals" in (pages[4].text or "")
        assert pages[2].ocr_payload_key is not None
        chunks = list(s.scalars(select(ChunkRecord)))
        assert chunks and all(c.collection != "skip" for c in chunks)
        assert s.scalar(select(IngestionRun)).status == "completed"


def test_second_run_skips_unchanged_documents(env: dict[str, object]) -> None:
    book(env["input"])  # type: ignore[arg-type]
    build(env, FakeOCREngine()).run()
    ocr = FakeOCREngine()
    _, stats = build(env, ocr).run()
    assert stats.unchanged == 1 and stats.versions_processed == 0 and ocr.calls == []


def test_modified_file_creates_new_current_version(env: dict[str, object]) -> None:
    path = book(env["input"])  # type: ignore[arg-type]
    build(env, FakeOCREngine()).run()
    make_pdf(path, [{"text": ENGLISH}, {"text": LEGACY_TEXT}])
    _, stats = build(env, FakeOCREngine()).run()
    assert stats.new_versions == 1
    sessions = env["sessions"]
    assert count(sessions, DocumentVersion) == 2  # type: ignore[arg-type]
    assert count(sessions, DocumentVersion, DocumentVersion.is_current.is_(True)) == 1  # type: ignore[arg-type]


def test_identical_copy_is_recorded_as_duplicate_not_reprocessed(env: dict[str, object]) -> None:
    src = book(env["input"])  # type: ignore[arg-type]
    shutil.copy(src, env["input"] / "copy-of-book.pdf")  # type: ignore[operator]
    ocr = FakeOCREngine()
    _, stats = build(env, ocr).run()
    assert stats.new_versions == 1 and stats.duplicates == 1
    assert len(ocr.calls) == 2  # only one document was OCR'd


def test_deleted_file_is_tombstoned(env: dict[str, object]) -> None:
    path = book(env["input"])  # type: ignore[arg-type]
    build(env, FakeOCREngine()).run()
    path.unlink()
    _, stats = build(env, FakeOCREngine()).run()
    assert stats.deleted == 1
    with env["sessions"]() as s:  # type: ignore[operator]
        doc = s.scalar(select(Document))
        assert doc.status == "deleted" and doc.deleted_at is not None
        assert not any(v.is_current for v in doc.versions)


def test_primary_failure_falls_back_to_secondary_engine(env: dict[str, object]) -> None:
    book(env["input"])  # type: ignore[arg-type]
    fallback = FakeOCREngine("tesseract")
    _, stats = build(env, FakeOCREngine(raise_error=True), fallback).run()
    assert stats.pages_ocr == 3 and stats.pages_failed == 0
    with env["sessions"]() as s:  # type: ignore[operator]
        assert {p.ocr_engine for p in s.scalars(select(Page).where(Page.extraction_method == "ocr"))} == {
            "tesseract"
        }


def test_partial_failure_is_retried_on_next_run_without_redoing_good_pages(env: dict[str, object]) -> None:
    book(env["input"])  # type: ignore[arg-type]
    failing = FakeOCREngine(fail_pages={3})
    _, stats = build(env, failing, FakeOCREngine("tesseract", fail_pages={3})).run()
    assert stats.pages_failed == 1
    with env["sessions"]() as s:  # type: ignore[operator]
        assert s.scalar(select(DocumentVersion)).status == "partial"

    healthy = FakeOCREngine()
    _, stats = build(env, healthy).run()
    assert healthy.calls == [[3]]  # only the failed page is re-OCR'd
    assert stats.pages_reused == 4
    with env["sessions"]() as s:  # type: ignore[operator]
        assert s.scalar(select(DocumentVersion)).status == "completed"


def test_rechunk_reuses_cached_page_text(env: dict[str, object]) -> None:
    book(env["input"])  # type: ignore[arg-type]
    build(env, FakeOCREngine()).run()
    ocr = FakeOCREngine()
    _, stats = build(env, ocr).run(rechunk=True)
    assert ocr.calls == [] and stats.versions_processed == 1 and stats.chunks_written > 0
    # Rechunking replaces, never duplicates, chunks for the same chunker version.
    assert count(env["sessions"], ChunkRecord) == stats.chunks_written  # type: ignore[arg-type]

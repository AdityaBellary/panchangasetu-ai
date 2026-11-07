"""Pipeline against real Postgres (docker compose) and real Tesseract.

Run: docker compose up -d postgres && pytest -m integration
Uses a throwaway schema so it never touches ingested data.
"""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker

from panchangasetu.config import Settings
from panchangasetu.ingestion.classification import CollectionRules
from panchangasetu.ingestion.pipeline import IngestionBusyError, IngestionPipeline
from panchangasetu.providers.ocr.tesseract import TesseractOCREngine
from panchangasetu.storage.models import Base, Page
from panchangasetu.storage.object_store import LocalObjectStore
from tests.conftest import FakeOCREngine, make_pdf

pytestmark = pytest.mark.integration

PG_URL = os.environ.get("PSETU_TEST_DATABASE_URL", "postgresql+psycopg://psetu:psetu@localhost:55432/psetu")
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def pg_sessions() -> Iterator[sessionmaker]:  # type: ignore[type-arg]
    schema = f"test_{uuid.uuid4().hex[:8]}"
    admin = create_engine(PG_URL)
    try:
        with admin.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Postgres not reachable: {exc}")
    engine = create_engine(PG_URL, connect_args={"options": f"-csearch_path={schema}"})
    Base.metadata.create_all(engine)
    try:
        yield sessionmaker(engine, expire_on_commit=False)
    finally:
        engine.dispose()
        with admin.begin() as conn:
            conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


def _settings(tmp_path: Path) -> Settings:
    (tmp_path / "raw").mkdir()
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        database_url=PG_URL,
        input_dir=tmp_path / "raw",
        object_store_dir=tmp_path / "objects",
        collections_rules_path=tmp_path / "none.yaml",
        render_dpi=72,
        ingestion_lock_key=uuid.uuid4().int % (2**62),  # independent of any live ingestion
    )


def test_pipeline_runs_on_postgres(tmp_path: Path, pg_sessions: sessionmaker) -> None:  # type: ignore[type-arg]
    settings = _settings(tmp_path)
    make_pdf(settings.input_dir / "a.pdf", [{"image": True}, {"image": True}])
    pipe = IngestionPipeline(
        settings,
        pg_sessions,
        LocalObjectStore(settings.object_store_dir),
        engine_factory=lambda _: FakeOCREngine(),
        rules=CollectionRules([]),
    )
    _, stats = pipe.run()
    assert stats.pages_ocr == 2 and stats.chunks_written >= 1
    with pg_sessions() as s:
        assert all(p.text_stats["chars"] > 0 for p in s.scalars(select(Page)))


def test_concurrent_runs_are_rejected_by_advisory_lock(tmp_path: Path, pg_sessions: sessionmaker) -> None:  # type: ignore[type-arg]
    settings = _settings(tmp_path)
    pipe = IngestionPipeline(
        settings,
        pg_sessions,
        LocalObjectStore(settings.object_store_dir),
        engine_factory=lambda _: FakeOCREngine(),
        rules=CollectionRules([]),
    )
    with pipe._exclusive(), pytest.raises(IngestionBusyError):
        pipe.run()


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_tesseract_reads_telugu_from_real_panchangam_page() -> None:
    from panchangasetu.ingestion.pdf_pages import PdfReader, inspect_page, render_upright

    pdf = REPO / "data/raw/TTD_panchangam.pdf"
    tessdata = REPO / "data/tessdata"
    if not pdf.exists() or not (tessdata / "tel.traineddata").exists():
        pytest.skip("sample PDF / Telugu traineddata not present")
    with PdfReader(pdf) as r:
        page = next(r.pages([100]))
        info = inspect_page(page, min_chars=40, min_valid_ratio=0.9)
        image = render_upright(page, info, dpi=300, min_short_side_px=0)
    result = TesseractOCREngine(tessdata_dir=tessdata).ocr([image])[0]
    # Day row for 19 March 2026 (Ugadi): Amavasya ends 7-06 morning, Shukla Padyami begins.
    assert "అమావాస్య" in result.text and "7-06" in result.text
    assert result.confidence is not None and result.confidence > 0.6

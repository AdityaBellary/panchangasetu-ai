"""StructuredExtractor end to end: real book page, real Tesseract, SQLite (no Postgres needed)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from sqlalchemy import select

from panchangasetu.providers.ocr.tesseract import TesseractOCREngine
from panchangasetu.storage.db import make_engine, make_session_factory
from panchangasetu.storage.models import Base, Document, DocumentVersion, PanchangaDay
from panchangasetu.storage.object_store import LocalObjectStore, raw_pdf_key
from panchangasetu.structured.service import StructuredExtractor

pytestmark = pytest.mark.integration
REPO = Path(__file__).resolve().parents[2]
PDF = REPO / "data/raw/TTD_panchangam.pdf"
TESSDATA = REPO / "data/tessdata"


@pytest.mark.skipif(
    not PDF.exists() or shutil.which("tesseract") is None or not (TESSDATA / "tel.traineddata").exists(),
    reason="needs the sample PDF and Tesseract with Telugu data",
)
def test_daily_table_page_is_extracted_verified_and_idempotent(tmp_path: Path) -> None:
    engine = make_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    Base.metadata.create_all(engine)
    sessions = make_session_factory(engine)
    store = LocalObjectStore(tmp_path / "objects")
    sha = "0" * 64
    store.put_file(raw_pdf_key(sha), PDF)
    with sessions.begin() as s:
        doc = Document(source_path="TTD_panchangam.pdf", filename="TTD_panchangam.pdf")
        s.add(doc)
        s.flush()
        s.add(
            DocumentVersion(
                document_id=doc.id,
                sha256=sha,
                size_bytes=1,
                page_count=176,
                object_key=raw_pdf_key(sha),
                status="completed",
            )
        )
    sources = tmp_path / "sources.yaml"
    sources.write_text(
        'sources:\n  - match: "TTD_panchangam.pdf"\n    daily_table:\n      pages: "100"\n'
        "      first_date: 2026-03-19\n      initial_masa: ఫాల్గుణం\n      initial_paksha: బహుళ\n",
        encoding="utf-8",
    )

    def row_engine(psm: int) -> TesseractOCREngine:
        return TesseractOCREngine(langs="tel", psm=psm, tessdata_dir=TESSDATA)

    extractor = StructuredExtractor(sessions, store, sources, row_engine)
    stats = extractor.run()
    assert stats.days_written == 11  # p100 = 19..29 March 2026
    with sessions() as s:
        ugadi = s.scalar(
            select(PanchangaDay).where(PanchangaDay.day == __import__("datetime").date(2026, 3, 19))
        )
        assert ugadi is not None and ugadi.sunrise == "6-22" and ugadi.date_check in ("ok", "unreadable")
        tithis = [e for e in ugadi.elements if e["kind"] == "tithi"]
        assert tithis[0]["name"] == "అమావాస్య" and tithis[0]["printed_end"] == "ఉ.7-06" and tithis[0]["verified"]
        assert "ఉగాది" in ugadi.notes
    assert extractor.run().sources_skipped == 1  # up to date: nothing re-extracted

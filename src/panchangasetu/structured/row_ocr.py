"""OCR one cropped day row at a time (free, local Tesseract) and trim crop bleed.

A day's text always starts with its weekday token and ends with the day-length marker
"భో.x-xx అహః yy-yy". Anything outside that span is a sliver of a neighbouring row that the
rectangular crop caught, and is dropped.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import pymupdf

from panchangasetu.core.interfaces import OCREngine
from panchangasetu.core.types import PageImage
from panchangasetu.structured.book_daily import DayRow
from panchangasetu.telugu.text import normalize_telugu

WEEKDAY_TOKENS = ("ఆది", "సోమ", "మంగళ", "బుధ", "గురు", "శుక్ర", "శని")
_START_RE = re.compile(r"(?:^|\n)\s*(" + "|".join(WEEKDAY_TOKENS) + r")\s")
_END_RE = re.compile(r"అహః\s*\d{1,2}-\d{2}")


def trim_row_text(text: str) -> tuple[str, str | None]:
    """Cut OCR text to [weekday ... అహః NN-NN]. Returns (text, weekday_token or None)."""
    start = _START_RE.search(text)
    weekday = start.group(1) if start else None
    body = text[start.start() :].lstrip("\n") if start else text
    end = _END_RE.search(body)
    if end:
        body = body[: end.end()]
    return body.strip(), weekday


def render_row(page: pymupdf.Page, row: DayRow, dpi: int) -> PageImage:
    original = page.rotation
    page.set_rotation(0)  # row boxes are in unrotated page space
    try:
        b = row.main_box
        pix = page.get_pixmap(dpi=dpi, clip=pymupdf.Rect(b.x0, b.y0, b.x1, b.y1), alpha=False)
    finally:
        page.set_rotation(original)
    return PageImage(
        page_number=row.page_number,
        png=pix.tobytes("png"),
        width=pix.width,
        height=pix.height,
        rotation_applied=0,
    )


def ocr_rows(
    doc: pymupdf.Document, rows: Sequence[DayRow], engine: OCREngine, *, dpi: int = 300, workers: int = 4
) -> list[tuple[str, str | None]]:
    """OCR rows in parallel (Tesseract is a subprocess, so threads parallelise fine).
    Rendering stays on the calling thread: PyMuPDF documents are not thread-safe."""
    images = [render_row(doc[r.page_number - 1], r, dpi) for r in rows]

    def one(img: PageImage) -> str:
        return engine.ocr([img])[0].text

    with ThreadPoolExecutor(max_workers=workers) as pool:
        texts = list(pool.map(one, images))
    return [trim_row_text(normalize_telugu(t)) for t in texts]

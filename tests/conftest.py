from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pymupdf
import pytest

from panchangasetu.core.interfaces import OCRError
from panchangasetu.core.types import OCRPage, PageImage

# Stand-in for a legacy-font text layer: a text layer that is neither Telugu nor Latin Unicode.
# Real legacy fonts emit Private Use Area codes (covered in test_telugu_text), but PyMuPDF's
# built-in fonts cannot embed PUA glyphs, so fixtures use CJK via the built-in "china-s" font,
# which exercises the same "untrusted text layer -> OCR" path.
LEGACY_TEXT = "中文字体编码测试页面内容 " * 12


def make_pdf(path: Path, pages: Sequence[dict[str, object]]) -> Path:
    """Build a small PDF. Each page dict may have: text (str), rotate_text (int), image (bool),
    page_rotation (int)."""
    doc = pymupdf.open()
    for spec in pages:
        page = doc.new_page(width=400, height=600)
        if spec.get("image"):
            pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 300), False)
            pix.clear_with(200)
            page.insert_image(page.rect, pixmap=pix)
        if text := spec.get("text"):
            # Unrotated text line direction controls the expected upright rotation.
            fontname = "china-s" if any(ord(c) > 0x2E80 for c in str(text)) else "helv"
            rotate = int(spec.get("rotate_text", 0))
            # Start where the run of text stays on the page for its direction.
            origin = {0: (10, 300), 90: (200, 590), 180: (390, 300), 270: (200, 10)}[rotate]
            page.insert_text(origin, str(text), fontsize=6, fontname=fontname, rotate=rotate)
        if rot := spec.get("page_rotation"):
            page.set_rotation(int(rot))
    doc.save(path)
    doc.close()
    return path


class FakeOCREngine:
    """Deterministic OCR stand-in. Records calls; can be told to fail specific pages."""

    def __init__(
        self, name: str = "sarvam", fail_pages: set[int] | None = None, raise_error: bool = False
    ) -> None:
        self.name = name
        self.calls: list[list[int]] = []
        self.fail_pages = fail_pages or set()
        self.raise_error = raise_error

    @property
    def version(self) -> str:
        return f"fake-{self.name}-1"

    def ocr(self, pages: Sequence[PageImage]) -> list[OCRPage]:
        self.calls.append([p.page_number for p in pages])
        if self.raise_error:
            raise OCRError("boom", retryable=True)
        return [
            OCRPage(
                page_number=p.page_number,
                text=f"## శీర్షిక {p.page_number}\n\nఈ రాశివారికి గురుడు సామాన్యఫలితములను ఇచ్చును. పేజీ {p.page_number}.",
                text_format="markdown",
                engine=self.name,
                engine_version=self.version,
                payload={"page": p.page_number},
            )
            for p in pages
            if p.page_number not in self.fail_pages
        ]


@pytest.fixture
def legacy_text() -> str:
    return LEGACY_TEXT

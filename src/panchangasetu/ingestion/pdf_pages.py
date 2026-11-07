"""Per-page inspection and preparation of upright page images for OCR.

Pages are processed lazily, one at a time, so large PDFs never sit fully in memory.

Orientation: the `/Rotate` flag in these PDFs is unreliable (the TTD panchangam's daily tables
carry `/Rotate 90` yet render sideways). For pages with a text layer, the dominant line direction
(reported by MuPDF in unrotated page space) tells us exactly which display rotation makes the
text read left-to-right, even when the glyph encoding itself is a legacy font. Image-only pages
are rendered with their declared rotation.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from panchangasetu.core.types import PageImage, PageKind
from panchangasetu.telugu.text import ScriptStats, script_stats

# Unrotated line direction -> clockwise display rotation that makes it read left-to-right.
_DIR_TO_ROTATION: dict[tuple[int, int], int] = {(1, 0): 0, (0, -1): 90, (-1, 0): 180, (0, 1): 270}


@dataclass(frozen=True, slots=True)
class PageInfo:
    page_number: int  # 1-based
    kind: PageKind
    declared_rotation: int
    upright_rotation: int
    text_layer: str
    stats: ScriptStats
    image_count: int


def dominant_text_rotation(page: pymupdf.Page) -> int | None:
    weights: Counter[tuple[int, int]] = Counter()
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            key = (round(line["dir"][0]), round(line["dir"][1]))
            weights[key] += sum(len(s["text"].strip()) for s in line["spans"])
    if not weights:
        return None
    direction, count = weights.most_common(1)[0]
    if count == 0:
        return None
    return _DIR_TO_ROTATION.get(direction)


def inspect_page(page: pymupdf.Page, *, min_chars: int, min_valid_ratio: float) -> PageInfo:
    text = page.get_text("text", sort=True)
    stats = script_stats(text)
    images = len(page.get_images(full=False))
    if stats.letters >= min_chars:
        kind = (
            PageKind.BORN_DIGITAL_UNICODE
            if stats.valid_ratio >= min_valid_ratio
            else PageKind.BORN_DIGITAL_LEGACY_FONT
        )
    elif images:
        kind = PageKind.IMAGE_ONLY
    else:
        kind = PageKind.EMPTY
    rotation = dominant_text_rotation(page) if stats.letters >= min_chars else None
    return PageInfo(
        page_number=page.number + 1,
        kind=kind,
        declared_rotation=page.rotation,
        upright_rotation=page.rotation if rotation is None else rotation,
        text_layer=text,
        stats=stats,
        image_count=images,
    )


def _image_only_dpi(page: pymupdf.Page, max_dpi: int, min_short_side_px: int) -> int:
    """Render image-only pages at roughly native resolution (no point oversampling a 900px
    scan to 300 DPI on a 2646pt page), but upscale small scans for OCR legibility."""
    native_dpi = 0.0
    for img in page.get_images(full=True):
        width_px = img[2]
        bbox = page.get_image_bbox(img)
        if bbox.width > 0:
            native_dpi = max(native_dpi, 72.0 * width_px / bbox.width)
    short_pts = min(page.rect.width, page.rect.height)
    needed = 72.0 * min_short_side_px / short_pts if short_pts else 0.0
    return int(min(max_dpi, max(native_dpi, needed, 72.0)))


def render_upright(page: pymupdf.Page, info: PageInfo, *, dpi: int, min_short_side_px: int) -> PageImage:
    original = page.rotation
    effective_dpi = _image_only_dpi(page, dpi, min_short_side_px) if info.kind == PageKind.IMAGE_ONLY else dpi
    try:
        if info.upright_rotation != original:
            page.set_rotation(info.upright_rotation)  # in-memory only; the source file is never saved
        pix = page.get_pixmap(dpi=effective_dpi, alpha=False, colorspace=pymupdf.csRGB)
    finally:
        if page.rotation != original:
            page.set_rotation(original)
    return PageImage(
        page_number=info.page_number,
        png=pix.tobytes("png"),
        width=pix.width,
        height=pix.height,
        rotation_applied=info.upright_rotation,
    )


class PdfReader:
    """Context-managed, lazy page access to one PDF."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._doc: pymupdf.Document | None = None

    def __enter__(self) -> PdfReader:
        self._doc = pymupdf.open(self.path)
        if self._doc.needs_pass:
            self._doc.close()
            raise ValueError(f"encrypted PDF is not supported: {self.path.name}")
        return self

    def __exit__(self, *exc: object) -> None:
        if self._doc is not None:
            self._doc.close()

    @property
    def doc(self) -> pymupdf.Document:
        if self._doc is None:
            raise RuntimeError("PdfReader used outside of a with-block")
        return self._doc

    @property
    def page_count(self) -> int:
        return int(self.doc.page_count)

    def metadata(self) -> dict[str, object]:
        meta = {k: v for k, v in (self.doc.metadata or {}).items() if v}
        meta["page_count"] = self.page_count
        return meta

    def pages(self, numbers: list[int] | None = None) -> Iterator[pymupdf.Page]:
        wanted = numbers if numbers is not None else range(1, self.page_count + 1)
        for n in wanted:
            yield self.doc[n - 1]


def find_gutter(page: pymupdf.Page, *, search: tuple[float, float] = (0.35, 0.65)) -> float | None:
    """x of the emptiest vertical band in the middle of the page, from exact glyph boxes in the
    text layer (reliable even when the glyph *encoding* is a legacy font). None if no clear gutter."""
    width = int(page.rect.width)
    cover = [0] * (width + 1)
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                if span["text"].strip():
                    for x in range(max(0, int(span["bbox"][0])), min(width, int(span["bbox"][2]))):
                        cover[x] += 1
    lo, hi = int(width * search[0]), int(width * search[1])
    x = min(range(lo, hi), key=lambda i: cover[i])
    busiest = max(cover[lo:hi]) or 1
    return float(x) if cover[x] <= 0.05 * busiest else None


def _full_width_band_bottom(page: pymupdf.Page, gutter: float, *, within: float = 0.3) -> float:
    """Bottom y of the title band: text-layer lines near the top that cross the gutter (a table
    title spans both columns and would otherwise be cut in half)."""
    limit = page.rect.height * within
    bottom = 0.0
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            x0, y0, x1, y1 = line["bbox"]
            if (
                y0 < limit
                and x0 < gutter - 5 < gutter + 5 < x1
                and any(s["text"].strip() for s in line["spans"])
            ):
                bottom = max(bottom, y1)
    return bottom


def render_columns(page: pymupdf.Page, info: PageInfo, *, dpi: int) -> list[PageImage] | None:
    """[title band (optional), left column, right column] images of an upright two-column page,
    or None when there is no clear gutter."""
    original = page.rotation
    try:
        if info.upright_rotation != original:
            page.set_rotation(info.upright_rotation)
        gutter = find_gutter(page)
        if gutter is None:
            return None
        top = _full_width_band_bottom(page, gutter)
        rects = [pymupdf.Rect(0, 0, page.rect.width, top + 2)] if top > 0 else []
        rects += [
            pymupdf.Rect(0, top, gutter, page.rect.height),
            pymupdf.Rect(gutter, top, page.rect.width, page.rect.height),
        ]
        images = []
        for rect in rects:
            pix = page.get_pixmap(dpi=dpi, clip=rect, alpha=False, colorspace=pymupdf.csRGB)
            images.append(
                PageImage(info.page_number, pix.tobytes("png"), pix.width, pix.height, info.upright_rotation)
            )
        return images
    finally:
        if page.rotation != original:
            page.set_rotation(original)

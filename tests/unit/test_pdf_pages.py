from pathlib import Path

import pymupdf
import pytest

from panchangasetu.core.types import PageKind
from panchangasetu.ingestion.pdf_pages import PdfReader, inspect_page, render_upright
from tests.conftest import LEGACY_TEXT, make_pdf

ENGLISH = "The quick brown fox jumps over the lazy dog. " * 4


def _info(path: Path, n: int = 1):  # type: ignore[no-untyped-def]
    with PdfReader(path) as r:
        page = next(r.pages([n]))
        return inspect_page(page, min_chars=40, min_valid_ratio=0.9)


def test_unicode_text_layer_is_trusted(tmp_path: Path) -> None:
    info = _info(make_pdf(tmp_path / "a.pdf", [{"text": ENGLISH}]))
    assert info.kind == PageKind.BORN_DIGITAL_UNICODE
    assert info.upright_rotation == 0


def test_untrusted_text_layer_is_routed_to_ocr(tmp_path: Path) -> None:
    info = _info(make_pdf(tmp_path / "b.pdf", [{"text": LEGACY_TEXT}]))
    assert info.kind == PageKind.BORN_DIGITAL_LEGACY_FONT


def test_image_only_and_empty_pages(tmp_path: Path) -> None:
    path = make_pdf(tmp_path / "c.pdf", [{"image": True}, {}])
    assert _info(path, 1).kind == PageKind.IMAGE_ONLY
    assert _info(path, 2).kind == PageKind.EMPTY


@pytest.mark.parametrize(("rotate_text", "expected"), [(0, 0), (90, 90), (180, 180), (270, 270)])
def test_upright_rotation_follows_text_direction_not_rotate_flag(
    tmp_path: Path, rotate_text: int, expected: int
) -> None:
    # Declare a misleading /Rotate flag, as the TTD panchangam does.
    path = make_pdf(tmp_path / "r.pdf", [{"text": ENGLISH, "rotate_text": rotate_text, "page_rotation": 90}])
    info = _info(path)
    assert info.declared_rotation == 90
    assert info.upright_rotation == expected


def test_rendered_image_is_upright_and_source_untouched(tmp_path: Path) -> None:
    path = make_pdf(tmp_path / "r.pdf", [{"text": ENGLISH, "rotate_text": 90}])
    with PdfReader(path) as r:
        page = next(r.pages([1]))
        info = inspect_page(page, min_chars=40, min_valid_ratio=0.9)
        img = render_upright(page, info, dpi=72, min_short_side_px=0)
        assert page.rotation == 0  # restored after rendering
    assert img.width > img.height  # portrait page turned so the vertical text reads horizontally
    assert img.png.startswith(b"\x89PNG")
    with pymupdf.open(path) as doc:
        assert doc[0].rotation == 0


def test_image_only_pages_are_upscaled_to_min_short_side(tmp_path: Path) -> None:
    path = make_pdf(tmp_path / "i.pdf", [{"image": True}])
    with PdfReader(path) as r:
        page = next(r.pages([1]))
        info = inspect_page(page, min_chars=40, min_valid_ratio=0.9)
        img = render_upright(page, info, dpi=300, min_short_side_px=800)
    assert min(img.width, img.height) >= 795


def test_encrypted_pdf_is_rejected(tmp_path: Path) -> None:
    doc = pymupdf.open()
    doc.new_page()
    path = tmp_path / "enc.pdf"
    doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
    with pytest.raises(ValueError, match="encrypted"), PdfReader(path):
        pass

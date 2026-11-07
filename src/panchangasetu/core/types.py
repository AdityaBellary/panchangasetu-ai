"""Domain types shared across ingestion, retrieval and providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Collection(StrEnum):
    PANCHANGA = "panchanga"
    GENERAL = "general"
    SKIP = "skip"  # covers, blank pages, decorative pages


class ExtractionMethod(StrEnum):
    TEXT_LAYER = "text_layer"  # trusted Unicode text layer, no OCR needed
    OCR = "ocr"


class PageKind(StrEnum):
    BORN_DIGITAL_UNICODE = "born_digital_unicode"
    BORN_DIGITAL_LEGACY_FONT = "born_digital_legacy_font"  # text layer present but not Unicode
    IMAGE_ONLY = "image_only"
    EMPTY = "empty"


@dataclass(frozen=True, slots=True)
class PageImage:
    """A page prepared for OCR: upright, PNG-encoded."""

    page_number: int  # 1-based, as in the source PDF
    png: bytes
    width: int
    height: int
    rotation_applied: int  # degrees clockwise applied to make text upright


@dataclass(frozen=True, slots=True)
class OCRPage:
    page_number: int
    text: str
    text_format: str  # "markdown" | "plain"
    engine: str
    engine_version: str
    confidence: float | None = None
    # Raw engine output (blocks, bboxes) kept for audit and future citation highlighting.
    payload: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class Classification:
    collection: Collection
    section_type: str
    confidence: float
    reason: str


@dataclass(slots=True)
class Chunk:
    text: str
    collection: Collection
    section_type: str
    heading_path: list[str]
    page_start: int
    page_end: int
    chunk_index: int
    token_count: int
    metadata: dict[str, object] = field(default_factory=dict)

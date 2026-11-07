"""Provider interfaces. Concrete implementations live in `panchangasetu.providers` and are
selected through settings, so swapping a vendor never touches pipeline code."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from panchangasetu.core.types import OCRPage, PageImage


class OCRError(RuntimeError):
    """OCR failed for a batch. `retryable` tells the caller whether a later retry may help."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


@runtime_checkable
class OCREngine(Protocol):
    name: str

    @property
    def version(self) -> str:
        """Identifies model/config so cached OCR can be invalidated when it changes."""
        ...

    def ocr(self, pages: Sequence[PageImage]) -> list[OCRPage]:
        """OCR a batch of upright page images. Returns one result per page that succeeded;
        pages missing from the result are treated as failed by the caller."""
        ...


class ObjectStore(Protocol):
    def put(self, key: str, data: bytes) -> None: ...

    def get(self, key: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...

    def delete(self, key: str) -> None: ...

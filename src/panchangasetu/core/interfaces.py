"""Provider interfaces. Concrete implementations live in `panchangasetu.providers` and are
selected through settings, so swapping a vendor never touches pipeline code."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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


class Embedder(Protocol):
    """Dense text embedder. Query and document encodings may differ (e.g. e5 prefixes)."""

    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


SparseVector = tuple[list[int], list[float]]  # (indices, values)


class SparseEncoder(Protocol):
    """Lexical (keyword) encoder producing sparse vectors for hybrid search."""

    name: str

    def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]: ...

    def encode_query(self, text: str) -> SparseVector: ...


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str  # system | user | assistant
    content: str


@dataclass(frozen=True, slots=True)
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class LLMError(RuntimeError):
    """A provider call failed. `retryable` = a later retry (or another provider) may succeed."""

    def __init__(self, message: str, *, retryable: bool, quota: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.quota = quota  # out of credits: needs billing action, fall back to another provider


class LLM(Protocol):
    name: str

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 600,
        temperature: float = 0.1,
        json: bool = False,
    ) -> LLMResult: ...

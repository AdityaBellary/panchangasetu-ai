"""Sarvam Document Intelligence (Sarvam Vision) OCR engine.

Contract, verified empirically on 2026-11-07 (SDK `sarvamai` 0.1.43 + live API):
  POST /doc-ai/v1/job/parse   (SDK `doc_ai.parse`)       -> {job_id, status}
  GET  /doc-ai/v1/job/{id}/status (SDK `doc_ai.get_status`) -> {status, usage}
  GET  /doc-ai/v1/job/{id}/results?format=json  -> raw HTTP; the SDK cannot decode it
       (it returns `type: "digitise"`, which the SDK's tagged union rejects).
       Shape: documents[0].pages[] = {page_num, status, image_width, image_height,
               blocks[] = {block_id, text, layout_tag, reading_order, coordinates, bbox_norm}}
       Table blocks carry HTML in `text`.

We always send rendered page images packed into a fresh image-only PDF, never the source PDF:
the source may carry a misleading legacy-font text layer and wrong /Rotate flags.

Blocks tagged `image` hold a *model-generated description* of a picture, not text read from
the page (observed inventing dates). They are excluded from page text to protect grounding.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from html.parser import HTMLParser
from typing import Any

import httpx
import pymupdf
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from panchangasetu.core.interfaces import OCRError
from panchangasetu.core.types import OCRPage, PageImage
from panchangasetu.observability.logging import get_logger

log = get_logger(__name__)

API_BASE = "https://api.sarvam.ai"
TERMINAL_OK = {"completed", "partially_completed"}
TERMINAL_FAIL = {"failed", "rejected"}

# Layout tags whose text is not reliable page content for retrieval.
EXCLUDED_TAGS = frozenset({"image", "page-number", "page_number", "folio"})
# Tags observed on the TTD corpus (2026-11-07): section-title (165 blocks), headline, header.
HEADING_TAGS = frozenset(
    {
        "header",
        "title",
        "section-title",
        "section_title",
        "section-header",
        "section_header",
        "heading",
        "headline",
    }
)
# Model-written descriptions of pictures can appear under any tag ("This image contains no
# discernible text", "ఈ చిత్రంలో మొసలి …"). They are not page text and must never be indexed.
_GENERATED_DESCRIPTION = re.compile(r"^\s*(?:this|the)\s+(?:image|picture|page|figure)\b|^\s*ఈ\s*చిత్ర", re.I)


def _status_code(exc: BaseException) -> int | None:
    code = getattr(exc, "status_code", None)
    if code is None and isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
    return code if isinstance(code, int) else None


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError | httpx.TimeoutException):
        return True
    code = _status_code(exc)
    return code is not None and (code == 429 or code >= 500)


def images_to_pdf(pages: Sequence[PageImage]) -> bytes:
    """Pack PNG page images into a PDF whose pages are exactly the image size (1px = 1pt)."""
    out = pymupdf.open()
    try:
        for img in pages:
            page = out.new_page(width=img.width, height=img.height)
            page.insert_image(page.rect, stream=img.png)
        return bytes(out.tobytes(deflate=True, garbage=3))
    finally:
        out.close()


class _TableText(HTMLParser):
    """Flatten an HTML table to text: one line per row, cells separated by ' | '.
    Nested tables are flattened into the enclosing cell."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._depth += 1
        elif tag == "tr" and self._depth == 1:
            self._row = []
        elif tag in ("td", "th") and self._depth == 1:
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "table":
            self._depth -= 1
        elif tag in ("td", "th") and self._depth == 1 and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._depth == 1 and self._row is not None:
            if any(self._row):
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def table_html_to_text(html: str) -> str:
    parser = _TableText()
    parser.feed(html)
    parser.close()
    return "\n".join(" | ".join(c for c in row if c) for row in parser.rows)


def blocks_to_markdown(blocks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for block in sorted(blocks, key=lambda b: b.get("reading_order") or 0):
        tag = (block.get("layout_tag") or "").lower()
        text = (block.get("text") or "").strip()
        if not text or tag in EXCLUDED_TAGS or _GENERATED_DESCRIPTION.match(text):
            continue
        if tag == "table" or text.lstrip().startswith("<table"):
            text = table_html_to_text(text)
        elif tag in HEADING_TAGS:
            text = "\n".join(f"## {line.strip()}" for line in text.splitlines() if line.strip())
        if text:
            parts.append(text)
    return "\n\n".join(parts)


class SarvamOCREngine:
    name = "sarvam"

    def __init__(
        self,
        client: Any,
        http: httpx.Client,
        *,
        language: str = "te-IN",
        content_type: str = "printed",
        poll_interval_s: float = 3.0,
        job_timeout_s: float = 900.0,
        max_retries: int = 4,
        model: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._http = http
        self._language = language
        self._content_type = content_type
        self._poll_interval_s = poll_interval_s
        self._job_timeout_s = job_timeout_s
        self._model = model
        self._sleep = sleep
        self._retrying = Retrying(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(max_retries + 1),
            wait=wait_exponential_jitter(initial=2, max=60),
            reraise=True,
            sleep=sleep,
        )

    @classmethod
    def from_api_key(
        cls, api_key: str, *, request_timeout_s: float = 120.0, base_url: str = API_BASE, **kwargs: Any
    ) -> SarvamOCREngine:
        from sarvamai import SarvamAI

        client = SarvamAI(api_subscription_key=api_key, timeout=request_timeout_s)
        http = httpx.Client(
            base_url=base_url, headers={"api-subscription-key": api_key}, timeout=request_timeout_s
        )
        return cls(client, http, **kwargs)

    @property
    def version(self) -> str:
        return f"doc-ai-parse:{self._model or 'default'}:{self._language}:{self._content_type}"

    def _call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return self._retrying(fn, *args, **kwargs)
        except Exception as exc:
            code = _status_code(exc)
            # 402 = out of credits, 401/403 = bad key: retrying won't help without action.
            raise OCRError(
                f"sarvam doc-ai call failed (status={code}): {type(exc).__name__}",
                retryable=_is_retryable(exc),
            ) from exc

    def _fetch_results(self, job_id: str) -> dict[str, Any]:
        resp = self._http.get(f"/doc-ai/v1/job/{job_id}/results", params={"format": "json"})
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return data

    def ocr(self, pages: Sequence[PageImage]) -> list[OCRPage]:
        if not pages:
            return []
        kwargs: dict[str, Any] = {
            "file": [("batch.pdf", images_to_pdf(pages), "application/pdf")],
            "language": self._language,
            "content_type": self._content_type,
        }
        if self._model:
            kwargs["model"] = self._model
        job = self._call(self._client.doc_ai.parse, **kwargs)
        job_id = job.job_id
        bound = log.bind(job_id=job_id, pages=[p.page_number for p in pages])
        bound.info("sarvam.parse.submitted")

        deadline = time.monotonic() + self._job_timeout_s
        status = getattr(job, "status", "")
        while status not in TERMINAL_OK | TERMINAL_FAIL:
            if time.monotonic() > deadline:
                raise OCRError(f"sarvam job {job_id} timed out after {self._job_timeout_s}s", retryable=True)
            self._sleep(self._poll_interval_s)
            status = self._call(self._client.doc_ai.get_status, job_id).status
        if status in TERMINAL_FAIL:
            raise OCRError(f"sarvam job {job_id} ended with status={status}", retryable=False)

        results = self._call(self._fetch_results, job_id)
        by_batch_index: dict[int, dict[str, Any]] = {}
        for document in results.get("documents") or []:
            for item in document.get("pages") or []:
                num = item.get("page_num", item.get("page_number"))
                if num is not None and item.get("status", "completed") == "completed":
                    by_batch_index[int(num)] = item

        out: list[OCRPage] = []
        for idx, page in enumerate(pages, start=1):
            item = by_batch_index.get(idx)
            if item is None:
                continue  # missing/failed page -> caller records failure / tries fallback
            blocks = item.get("blocks") or []
            out.append(
                OCRPage(
                    page_number=page.page_number,
                    text=blocks_to_markdown(blocks),
                    text_format="markdown",
                    engine=self.name,
                    engine_version=self.version,
                    payload={"job_id": job_id, "page": item},
                )
            )
        bound.info("sarvam.parse.done", status=status, returned=len(out))
        return out

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from panchangasetu.core.interfaces import OCRError
from panchangasetu.core.types import PageImage
from panchangasetu.providers.ocr.sarvam import SarvamOCREngine, blocks_to_markdown, table_html_to_text

PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108020000009077"
    "53de0000000c4944415408d763f8ffff3f0005fe02fea7d6a4c40000000049454e44ae426082"
)


def img(n: int) -> PageImage:
    return PageImage(page_number=n, png=PNG_1PX, width=1, height=1, rotation_applied=0)


def block(text: str, tag: str, order: int) -> dict[str, Any]:
    return {"text": text, "layout_tag": tag, "reading_order": order}


def test_image_descriptions_and_page_numbers_are_excluded() -> None:
    md = blocks_to_markdown(
        [
            block("ఈ చిత్రం 12-01-2025 ఉత్సవం గురించి", "image", 1),  # model-generated description
            block("కన్యారాశి ఫలితములు", "header", 2),
            block("ఈ రాశివారికి గురుడు", "paragraph", 3),
            block("੧", "page-number", 4),
        ]
    )
    assert md == "## కన్యారాశి ఫలితములు\n\nఈ రాశివారికి గురుడు"


def test_blocks_are_ordered_by_reading_order() -> None:
    assert blocks_to_markdown([block("రెండు", "paragraph", 2), block("ఒకటి", "paragraph", 1)]) == "ఒకటి\n\nరెండు"


def test_html_tables_flatten_to_rows_including_nested_tables() -> None:
    html = (
        "<table><thead><tr><th>సూ.ఉ.</th><th>తిథి</th></tr></thead><tbody>"
        "<tr><td>6-22</td><td>అమావాస్య<br/>ఉ.7-06</td><td><table><tr><td>ర</td></tr></table>x</td></tr>"
        "</tbody></table>"
    )
    assert table_html_to_text(html) == "సూ.ఉ. | తిథి\n6-22 | అమావాస్య ఉ.7-06 | రx"


class FakeDocAI:
    def __init__(self, statuses: list[str], parse_error: Exception | None = None) -> None:
        self.statuses = statuses
        self.parse_error = parse_error
        self.parse_kwargs: dict[str, Any] = {}

    def parse(self, **kwargs: Any) -> Any:
        if self.parse_error:
            raise self.parse_error
        self.parse_kwargs = kwargs
        return SimpleNamespace(job_id="job-1", status="pending")

    def get_status(self, job_id: str) -> Any:
        return SimpleNamespace(status=self.statuses.pop(0))


def make_engine(
    results: dict[str, Any] | None, statuses: list[str], **kw: Any
) -> tuple[SarvamOCREngine, FakeDocAI]:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/doc-ai/v1/job/job-1/results"
        assert request.url.params["format"] == "json"
        return httpx.Response(200, json=results)

    docai = FakeDocAI(statuses, **kw)
    http = httpx.Client(base_url="https://api.test", transport=httpx.MockTransport(handler))
    engine = SarvamOCREngine(SimpleNamespace(doc_ai=docai), http, sleep=lambda _: None, max_retries=2)
    return engine, docai


def test_ocr_maps_batch_positions_back_to_source_page_numbers() -> None:
    results = {
        "documents": [
            {
                "pages": [
                    {"page_num": 1, "status": "completed", "blocks": [block("మొదటి", "paragraph", 1)]},
                    {"page_num": 2, "status": "completed", "blocks": [block("రెండవ", "paragraph", 1)]},
                ]
            }
        ]
    }
    engine, docai = make_engine(results, ["running", "completed"])
    out = engine.ocr([img(31), img(100)])
    assert [(p.page_number, p.text) for p in out] == [(31, "మొదటి"), (100, "రెండవ")]
    assert docai.parse_kwargs["language"] == "te-IN"
    _filename, payload, mime = docai.parse_kwargs["file"][0]
    assert mime == "application/pdf" and payload.startswith(b"%PDF")
    assert out[0].payload is not None and out[0].payload["job_id"] == "job-1"


def test_failed_pages_in_partial_job_are_omitted() -> None:
    results = {
        "documents": [
            {
                "pages": [
                    {"page_num": 1, "status": "completed", "blocks": [block("ok", "paragraph", 1)]},
                    {"page_num": 2, "status": "failed", "blocks": []},
                ]
            }
        ]
    }
    engine, _ = make_engine(results, ["partially_completed"])
    assert [p.page_number for p in engine.ocr([img(5), img(6)])] == [5]


def test_failed_job_raises_non_retryable_error() -> None:
    engine, _ = make_engine(None, ["failed"])
    with pytest.raises(OCRError) as exc:
        engine.ocr([img(1)])
    assert exc.value.retryable is False


class _ApiError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def test_payment_required_is_not_retried() -> None:
    engine, _ = make_engine(None, [], parse_error=_ApiError(402))
    with pytest.raises(OCRError) as exc:
        engine.ocr([img(1)])
    assert exc.value.retryable is False


def test_rate_limit_is_retried_then_reported_retryable() -> None:
    engine, _ = make_engine(None, [], parse_error=_ApiError(429))
    with pytest.raises(OCRError) as exc:
        engine.ocr([img(1)])
    assert exc.value.retryable is True

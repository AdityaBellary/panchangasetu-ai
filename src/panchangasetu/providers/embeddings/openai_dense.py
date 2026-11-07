"""OpenAI embeddings (hosted). Plain HTTP via httpx; the key stays server-side.

Pricing at time of writing (third-party aggregators; verify on OpenAI's page): text-embedding-3-small
$0.02 / 1M tokens, text-embedding-3-large $0.13 / 1M tokens. Embedding the whole current
corpus costs cents; per-query cost is negligible.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
import numpy as np
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from panchangasetu.observability.logging import get_logger

log = get_logger(__name__)
API_BASE = "https://api.openai.com/v1"


class ProviderQuotaError(RuntimeError):
    """The account has no credits/quota left. Not retryable: it needs billing action."""


def _error_code(resp: httpx.Response) -> str:
    try:
        err = resp.json().get("error") or {}
    except ValueError:
        return ""
    return str(err.get("code") or err.get("type") or "")


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError | httpx.TimeoutException):
        return True
    if not isinstance(exc, httpx.HTTPStatusError):
        return False
    status = exc.response.status_code
    if status == 429:
        # 429 is overloaded: rate limits are transient, an empty credit balance is not.
        return _error_code(exc.response) not in ("insufficient_quota", "credit_balance_exhausted")
    return status >= 500


class OpenAIEmbedder:
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-large",
        *,
        dimensions: int | None = None,
        batch_size: int = 64,
        timeout_s: float = 60.0,
        max_retries: int = 4,
        base_url: str = API_BASE,
        http: httpx.Client | None = None,
    ) -> None:
        self.name = f"openai:{model}" + (f"@{dimensions}" if dimensions else "")
        self._model = model
        self._dimensions = dimensions
        self._batch = batch_size
        self._http = http or httpx.Client(
            base_url=base_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout_s
        )
        self._retrying = Retrying(
            retry=retry_if_exception(_retryable),
            stop=stop_after_attempt(max_retries + 1),
            wait=wait_exponential_jitter(initial=1, max=30),
            reraise=True,
        )
        self.tokens_used = 0
        self.dim = len(self.embed_query("dimension probe"))

    def _embed(self, texts: list[str]) -> list[list[float]]:
        body: dict[str, object] = {"model": self._model, "input": texts}
        if self._dimensions:
            body["dimensions"] = self._dimensions

        def call() -> dict[str, Any]:
            resp = self._http.post("/embeddings", json=body)
            if resp.status_code == 429 and _error_code(resp) in (
                "insufficient_quota",
                "credit_balance_exhausted",
            ):
                raise ProviderQuotaError("OpenAI account has no credits left (add billing to continue)")
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            return data

        data = self._retrying(call)
        self.tokens_used += int((data.get("usage") or {}).get("total_tokens", 0))
        rows: list[dict[str, Any]] = sorted(data["data"], key=lambda r: int(r["index"]))
        out = []
        for row in rows:
            v = np.asarray(row["embedding"], dtype=np.float32)
            out.append((v / (np.linalg.norm(v) or 1.0)).tolist())
        return out

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            out.extend(self._embed(list(texts[i : i + self._batch])))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

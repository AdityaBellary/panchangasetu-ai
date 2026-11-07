"""Chat completions over HTTP for OpenAI-compatible APIs: Sarvam and OpenAI.

Verified 2026-11-07:
  Sarvam  POST https://api.sarvam.ai/v1/chat/completions  header `api-subscription-key`
          `sarvam-105b-conversations`: no reasoning, ~0.3 s, JSON mode works.
          `sarvam-105b`: always reasons (reasoning_effort low|medium|high, cannot be disabled);
          reasoning tokens consume max_tokens and are billed as output.
          402 = credits exhausted.
  OpenAI  POST https://api.openai.com/v1/chat/completions  header `Authorization: Bearer`
          429 with code insufficient_quota / credit_balance_exhausted = credits exhausted.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import httpx
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from panchangasetu.core.interfaces import ChatMessage, LLMError, LLMResult

_QUOTA_CODES = ("insufficient_quota", "credit_balance_exhausted")


def _code(resp: httpx.Response) -> str:
    try:
        err = resp.json().get("error") or {}
    except ValueError:
        return ""
    return str(err.get("code") or err.get("type") or "")


def _classify(resp: httpx.Response) -> LLMError | None:
    s = resp.status_code
    if s < 400:
        return None
    if s == 402 or (s == 429 and _code(resp) in _QUOTA_CODES):
        return LLMError(f"provider out of credits (HTTP {s})", retryable=False, quota=True)
    if s == 429 or s >= 500:
        return LLMError(f"provider busy/unavailable (HTTP {s})", retryable=True)
    return LLMError(f"provider rejected the request (HTTP {s}): {resp.text[:200]}", retryable=False)


class OpenAICompatibleChat:
    def __init__(
        self,
        *,
        name: str,
        model: str,
        base_url: str,
        headers: dict[str, str],
        timeout_s: float = 60.0,
        max_retries: int = 2,
        extra_body: dict[str, Any] | None = None,
        http: httpx.Client | None = None,
    ) -> None:
        self.name = f"{name}:{model}"
        self.model = model
        self._extra = extra_body or {}
        self._http = http or httpx.Client(base_url=base_url, headers=headers, timeout=timeout_s)
        self._retrying = Retrying(
            retry=retry_if_exception(lambda e: isinstance(e, LLMError) and e.retryable),
            stop=stop_after_attempt(max_retries + 1),
            wait=wait_exponential_jitter(initial=1, max=15),
            reraise=True,
        )

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 600,
        temperature: float = 0.1,
        json: bool = False,
    ) -> LLMResult:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **self._extra,
        }
        if json:
            body["response_format"] = {"type": "json_object"}

        def call() -> LLMResult:
            t0 = time.perf_counter()
            try:
                resp = self._http.post("/chat/completions", json=body)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                raise LLMError(f"network error: {type(exc).__name__}", retryable=True) from exc
            if err := _classify(resp):
                raise err
            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            text = (choice.get("message") or {}).get("content") or ""
            if not text.strip():
                reason = choice.get("finish_reason")
                raise LLMError(f"empty completion (finish_reason={reason})", retryable=reason != "length")
            usage = data.get("usage") or {}
            return LLMResult(
                text=text,
                model=str(data.get("model") or self.model),
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        return self._retrying(call)


def sarvam_chat(api_key: str, model: str = "sarvam-105b-conversations", **kw: Any) -> OpenAICompatibleChat:
    extra = {"reasoning_effort": "low"} if model == "sarvam-105b" else None
    return OpenAICompatibleChat(
        name="sarvam",
        model=model,
        base_url="https://api.sarvam.ai/v1",
        headers={"api-subscription-key": api_key},
        extra_body=extra,
        **kw,
    )


def openai_chat(api_key: str, model: str = "gpt-4.1-mini", **kw: Any) -> OpenAICompatibleChat:
    return OpenAICompatibleChat(
        name="openai",
        model=model,
        base_url="https://api.openai.com/v1",
        headers={"Authorization": f"Bearer {api_key}"},
        **kw,
    )


class FallbackLLM:
    """Try providers in order. Quota exhaustion or an outage moves on to the next provider;
    a bad request (our bug) does not."""

    def __init__(self, providers: Sequence[OpenAICompatibleChat]) -> None:
        if not providers:
            raise ValueError("at least one LLM provider is required")
        self.providers = list(providers)
        self.name = " -> ".join(p.name for p in self.providers)

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 600,
        temperature: float = 0.1,
        json: bool = False,
    ) -> LLMResult:
        last: LLMError | None = None
        for provider in self.providers:
            try:
                return provider.complete(messages, max_tokens=max_tokens, temperature=temperature, json=json)
            except LLMError as exc:
                last = exc
                if not (exc.quota or exc.retryable):
                    raise
        assert last is not None
        raise last

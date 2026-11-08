"""Chat completions over HTTP for OpenAI-compatible APIs: Sarvam, Gemini (AI Studio) and OpenAI.

Verified 2026-11-07:
  Sarvam  POST https://api.sarvam.ai/v1/chat/completions  header `api-subscription-key`
          `sarvam-105b-conversations`: no reasoning, ~0.3 s, JSON mode works.
          `sarvam-105b`: always reasons (reasoning_effort low|medium|high, cannot be disabled);
          reasoning tokens consume max_tokens and are billed as output.
          402 = credits exhausted.
  OpenAI  POST https://api.openai.com/v1/chat/completions  header `Authorization: Bearer`
          429 with code insufficient_quota / credit_balance_exhausted = credits exhausted.
  Gemini  POST https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
          header `Authorization: Bearer`
          (docs checked 2026-11-08). Gemini 3 models always think (`reasoning_effort` minimal|low|medium|high
          lowers it, "none" only works on 2.5); thinking tokens bill as output. The docs show schema-based
          structured output only, so JSON is asked for in the prompt instead of `response_format`.

Every HTTP attempt (including retries) is recorded with its tokens, ₹ cost, latency and outcome
(observability/calls.py).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import httpx
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from panchangasetu.core.interfaces import ChatMessage, LLMError, LLMResult
from panchangasetu.observability.calls import LlmCall, PriceTable, current_purpose, record_call

_QUOTA_CODES = ("insufficient_quota", "credit_balance_exhausted")


def _error(resp: httpx.Response) -> dict[str, Any]:
    """The error object; Gemini wraps it in a list (`[{"error": {...}}]`)."""
    try:
        body = resp.json()
    except ValueError:
        return {}
    if isinstance(body, list):
        body = body[0] if body else {}
    err = body.get("error") if isinstance(body, dict) else None
    return err if isinstance(err, dict) else {}


def _code(resp: httpx.Response) -> str:
    err = _error(resp)
    return str(err.get("code") or err.get("type") or "")


def _classify(resp: httpx.Response) -> LLMError | None:
    s = resp.status_code
    if s < 400:
        return None
    err = _error(resp)
    # Gemini: 429 RESOURCE_EXHAUSTED "You exceeded your current quota, please check your plan and billing
    # details" (seen 2026-11-08) is a quota, not a burst: retrying wastes time, fall through instead.
    quota_text = "quota" in str(err.get("message", "")).lower() and "check your plan" in str(
        err.get("message", "")
    )
    if s == 402 or (s == 429 and (_code(resp) in _QUOTA_CODES or quota_text)):
        return LLMError(f"provider out of credits or quota (HTTP {s})", retryable=False, quota=True)
    if s == 429 or s >= 500:
        detail = str(_error(resp).get("message") or "")[:120]
        return LLMError(
            f"provider busy/unavailable (HTTP {s}){': ' + detail if detail else ''}", retryable=True
        )
    return LLMError(f"provider rejected the request (HTTP {s}): {resp.text[:200]}", retryable=False)


def _billed(usage: dict[str, Any]) -> tuple[int, int]:
    """(prompt, output) tokens as billed. Gemini leaves thinking tokens out of `completion_tokens`
    but counts them in `total_tokens` (2026-11-08: a one-word reply showed 1 completion, 66 total),
    and bills them as output, so output is whatever the total has beyond the prompt."""
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    total = int(usage.get("total_tokens") or 0)
    return prompt, max(completion, total - prompt)


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
        pricing: PriceTable | None = None,
        json_mode: bool = True,
        thinking_headroom: int = 0,
    ) -> None:
        self.name = f"{name}:{model}"
        self.provider = name
        self.model = model
        self.pricing = pricing or PriceTable({})
        self.json_mode = json_mode
        # Thinking models spend part of max_tokens before writing; extra allowance keeps the reply from
        # being cut off. Only tokens actually generated are billed.
        self.thinking_headroom = thinking_headroom
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
            "max_tokens": max_tokens + self.thinking_headroom,
            **self._extra,
        }
        if json and self.json_mode:
            body["response_format"] = {"type": "json_object"}

        def call() -> LLMResult:
            t0 = time.perf_counter()
            usage: dict[str, Any] = {}
            model = self.model
            try:
                try:
                    resp = self._http.post("/chat/completions", json=body)
                except (httpx.TransportError, httpx.TimeoutException) as exc:
                    raise LLMError(f"network error: {type(exc).__name__}", retryable=True) from exc
                if err := _classify(resp):
                    raise err
                data = resp.json()
                model = str(data.get("model") or self.model)
                usage = data.get("usage") or {}  # billed even when the reply turns out empty
                choice = (data.get("choices") or [{}])[0]
                text = (choice.get("message") or {}).get("content") or ""
                if not text.strip():
                    reason = choice.get("finish_reason")
                    raise LLMError(f"empty completion (finish_reason={reason})", retryable=reason != "length")
            except LLMError as exc:
                self._record(model, usage, t0, "quota" if exc.quota else "error", str(exc)[:200])
                raise
            prompt_tokens, completion_tokens = _billed(usage)
            result = LLMResult(
                text=text,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
            self._record(model, usage, t0, "ok")
            return result

        return self._retrying(call)

    def _record(self, model: str, usage: dict[str, Any], t0: float, status: str, error: str = "") -> None:
        p, c = _billed(usage)
        record_call(
            LlmCall(
                purpose=current_purpose(),
                provider=self.provider,
                model=model,
                prompt_tokens=p,
                completion_tokens=c,
                cost_inr=self.pricing.cost(model, p, c),
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                status=status,
                error=error,
            )
        )


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


def gemini_chat(
    api_key: str, model: str = "gemini-3.8-flash", *, reasoning_effort: str = "low", **kw: Any
) -> OpenAICompatibleChat:
    return OpenAICompatibleChat(
        name="gemini",
        model=model,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        headers={"Authorization": f"Bearer {api_key}"},
        extra_body={"reasoning_effort": reasoning_effort} if reasoning_effort else None,
        json_mode=False,
        thinking_headroom=2048,
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

"""Every LLM call, as it happens: purpose, provider, model, tokens, ₹ cost, latency and outcome.

Providers record each HTTP attempt (retries and fallbacks included) into whatever collectors are
active for the current request; the answer and conversation layers open collectors to attach the
calls to the answer, the stored turn and the metrics. Context variables keep concurrent requests
(threads in the API's pool) apart.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml


@dataclass(slots=True)
class LlmCall:
    purpose: str  # answer | answer_retry | escalation | followup_rewrite | other
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_inr: float
    latency_ms: float
    status: str  # ok | error | quota
    error: str = ""
    at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class PriceTable:
    """₹ per 1M tokens by model (config/llm_pricing.yaml)."""

    prices: dict[str, tuple[float, float]]
    default: tuple[float, float] = (15.0, 60.0)

    @classmethod
    def load(cls, path: Path | None) -> PriceTable:
        if path is None or not path.is_file():
            return cls({})
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        fx = float(raw.get("usd_to_inr", 88.0))

        def inr(spec: dict[str, float]) -> tuple[float, float]:
            if "input_inr" in spec:
                return float(spec["input_inr"]), float(spec["output_inr"])
            return float(spec["input_usd"]) * fx, float(spec["output_usd"]) * fx

        default = inr(raw["default"]) if raw.get("default") else (15.0, 60.0)
        return cls({str(k): inr(v) for k, v in (raw.get("models") or {}).items()}, default)

    def cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        # Providers may report a dated variant ("gemini-3.8-flash-001") or Google's resource name
        # ("models/gemini-3.8-flash"): match the longest known prefix of the bare name.
        name = model.removeprefix("models/")
        key = max((m for m in self.prices if name.startswith(m)), key=len, default=None)
        p_in, p_out = self.prices[key] if key else self.default
        return round((prompt_tokens * p_in + completion_tokens * p_out) / 1_000_000, 6)


_collectors: ContextVar[tuple[list[LlmCall], ...]] = ContextVar("llm_call_collectors", default=())
_purpose: ContextVar[str] = ContextVar("llm_call_purpose", default="other")


@contextmanager
def collect_calls() -> Iterator[list[LlmCall]]:
    """Collect the LLM calls made inside the block (nested collectors all see them)."""
    calls: list[LlmCall] = []
    token = _collectors.set((*_collectors.get(), calls))
    try:
        yield calls
    finally:
        _collectors.reset(token)


@contextmanager
def call_purpose(name: str) -> Iterator[None]:
    token = _purpose.set(name)
    try:
        yield
    finally:
        _purpose.reset(token)


def current_purpose() -> str:
    return _purpose.get()


def record_call(call: LlmCall) -> None:
    for calls in _collectors.get():
        calls.append(call)

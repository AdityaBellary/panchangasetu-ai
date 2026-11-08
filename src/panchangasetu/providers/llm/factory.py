"""Build the configured LLM chain from settings (only providers with a key are included)."""

from __future__ import annotations

from panchangasetu.config import Settings
from panchangasetu.observability.calls import PriceTable
from panchangasetu.providers.llm.openai_compatible import (
    FallbackLLM,
    OpenAICompatibleChat,
    gemini_chat,
    sarvam_chat,
)


def _key(secret: object) -> str:
    value = secret.get_secret_value() if secret is not None else ""  # type: ignore[attr-defined]
    return str(value or "")


def _chain(name: str, settings: Settings, **common: object) -> list[OpenAICompatibleChat]:
    if name == "sarvam" and (key := _key(settings.sarvam_api_key)):
        return [sarvam_chat(key, settings.sarvam_chat_model, **common)]
    if name == "gemini" and (key := _key(settings.gemini_api_key)):
        # Newer Flash models hit "high demand" 503s; one quick retry, then the next Gemini model.
        return [
            gemini_chat(
                key, model, reasoning_effort=settings.gemini_reasoning_effort, max_retries=1, **common
            )
            for model in dict.fromkeys([settings.gemini_chat_model, *settings.gemini_fallback_models])
        ]
    return []


def build_llm(settings: Settings) -> FallbackLLM | None:
    pricing = PriceTable.load(settings.llm_pricing_path)
    chain = [
        p
        for name in settings.llm_providers
        for p in _chain(name, settings, timeout_s=settings.llm_timeout_s, pricing=pricing)
    ]
    return FallbackLLM(chain) if chain else None


def build_escalation_llm(settings: Settings) -> FallbackLLM | None:
    """The stronger model tried once when the primary cannot produce a verified answer."""
    name = settings.escalation_provider
    if not name:
        return None
    pricing = PriceTable.load(settings.llm_pricing_path)
    if name == "sarvam":
        key = _key(settings.sarvam_api_key)
        timeout = max(settings.llm_timeout_s, 180)  # sarvam-105b reasons for ~20 s
        chain = (
            [sarvam_chat(key, settings.escalation_model, timeout_s=timeout, pricing=pricing)] if key else []
        )
    else:
        chain = _chain(name, settings, timeout_s=settings.llm_timeout_s, pricing=pricing)
    return FallbackLLM(chain) if chain else None

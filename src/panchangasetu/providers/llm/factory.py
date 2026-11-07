"""Build the configured LLM chain from settings (only providers with a key are included)."""

from __future__ import annotations

from panchangasetu.config import Settings
from panchangasetu.providers.llm.openai_compatible import (
    FallbackLLM,
    OpenAICompatibleChat,
    openai_chat,
    sarvam_chat,
)


def build_llm(settings: Settings) -> FallbackLLM | None:
    chain: list[OpenAICompatibleChat] = []
    for name in settings.llm_providers:
        if name == "sarvam" and settings.sarvam_api_key and settings.sarvam_api_key.get_secret_value():
            chain.append(
                sarvam_chat(
                    settings.sarvam_api_key.get_secret_value(),
                    settings.sarvam_chat_model,
                    timeout_s=settings.llm_timeout_s,
                )
            )
        elif name == "openai" and settings.openai_api_key and settings.openai_api_key.get_secret_value():
            chain.append(
                openai_chat(
                    settings.openai_api_key.get_secret_value(),
                    settings.openai_chat_model,
                    timeout_s=settings.llm_timeout_s,
                )
            )
    return FallbackLLM(chain) if chain else None

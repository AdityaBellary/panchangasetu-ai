"""Build the configured embedder. Swapping providers is a settings change, not a code change."""

from __future__ import annotations

from panchangasetu.config import Settings
from panchangasetu.core.interfaces import Embedder


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "openai":
        from panchangasetu.providers.embeddings.openai_dense import OpenAIEmbedder

        if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value():
            raise RuntimeError("OPENAI_API_KEY is not set (add it to .env)")
        return OpenAIEmbedder(
            settings.openai_api_key.get_secret_value(),
            settings.embedding_model,
            dimensions=settings.openai_embedding_dimensions,
        )
    from panchangasetu.providers.embeddings.fastembed_dense import FastEmbedDense

    return FastEmbedDense(settings.embedding_model, cache_dir=settings.model_cache_dir)

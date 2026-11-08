"""Build the configured embedder (local, no API key: multilingual-e5-large by default)."""

from __future__ import annotations

from panchangasetu.config import Settings
from panchangasetu.core.interfaces import Embedder


def build_embedder(settings: Settings) -> Embedder:
    from panchangasetu.providers.embeddings.fastembed_dense import FastEmbedDense

    return FastEmbedDense(settings.embedding_model, cache_dir=settings.model_cache_dir)

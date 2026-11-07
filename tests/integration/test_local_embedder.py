"""Local embedder smoke test against the cached e5 model (skipped when the model is not present)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from panchangasetu.config import Settings
from panchangasetu.providers.embeddings.factory import build_embedder

pytestmark = pytest.mark.integration
CACHE = Path(__file__).resolve().parents[2] / "data/models"


@pytest.mark.skipif(not any(CACHE.glob("*multilingual-e5-large*")), reason="e5 model not cached locally")
def test_e5_vectors_are_unit_length_and_cross_lingual() -> None:
    emb = build_embedder(Settings(_env_file=None, model_cache_dir=CACHE))  # type: ignore[call-arg]
    assert emb.name == "intfloat/multilingual-e5-large" and emb.dim == 1024
    docs = emb.embed_documents(["వైకుంఠ ఏకాదశి తిరుమల శ్రీవారి ఉత్సవం", "గృహప్రవేశము మాఘ వైశాఖ మాసములు"])
    q = np.array(emb.embed_query("When is Vaikuntha Ekadasi?"))
    assert all(abs(np.linalg.norm(d) - 1) < 1e-4 for d in docs) and abs(np.linalg.norm(q) - 1) < 1e-4
    assert q @ np.array(docs[0]) > q @ np.array(docs[1])  # English query lands on the Telugu festival text


def test_openai_provider_requires_a_key() -> None:
    settings = Settings(_env_file=None, embedding_provider="openai", embedding_model="text-embedding-3-large")  # type: ignore[call-arg]
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        build_embedder(settings)

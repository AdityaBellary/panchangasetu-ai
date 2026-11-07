import httpx
import pytest

from panchangasetu.providers.embeddings.openai_dense import OpenAIEmbedder, ProviderQuotaError
from panchangasetu.providers.embeddings.telugu_sparse import TeluguSparseEncoder, features


def test_sparse_features_include_words_and_akshara_bigrams() -> None:
    f = features("కన్యారాశి 2026")
    assert "w:కన్యారాశి" in f and "w:2026" in f
    # Aksharas of కన్యారాశి are క | న్యా | రా | శి: bigrams never split the conjunct న్యా.
    assert [x for x in f if x.startswith("b:")] == ["b:కన్యా", "b:న్యారా", "b:రాశి"]


def test_sparse_matching_tolerates_spacing_variants() -> None:
    enc = TeluguSparseEncoder()
    ((doc_idx, _),) = enc.encode_documents(["కన్యా రాశి వారికి ఈ సంవత్సరం"])
    q_idx, q_vals = enc.encode_query("కన్యారాశి")
    assert set(q_idx) & set(doc_idx)  # shared akshara bigrams despite different word splits
    assert all(v == 1.0 for v in q_vals)


def test_sparse_document_weights_saturate_with_repetition() -> None:
    enc = TeluguSparseEncoder()
    ((_, once),) = enc.encode_documents(["ఏకాదశి"])
    ((_, many),) = enc.encode_documents(["ఏకాదశి " * 50])
    assert max(many) < 6 * max(once)  # 1 + ln(tf), not raw counts


def _client(handler) -> httpx.Client:  # type: ignore[no-untyped-def]
    return httpx.Client(base_url="https://api.test/v1", transport=httpx.MockTransport(handler))


def test_openai_embedder_normalises_and_orders_by_index() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        n = len(__import__("json").loads(request.content)["input"])
        data = [{"index": i, "embedding": [3.0, 4.0]} for i in reversed(range(n))]
        return httpx.Response(200, json={"data": data, "usage": {"total_tokens": 7}})

    emb = OpenAIEmbedder("k", http=_client(handler), max_retries=0)
    vecs = emb.embed_documents(["a", "b", "c"])
    assert emb.dim == 2 and len(vecs) == 3
    assert vecs[0] == pytest.approx([0.6, 0.8])
    assert emb.tokens_used == 14  # probe + one batch


def test_openai_empty_credit_balance_fails_fast_without_retries() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            429, json={"error": {"type": "insufficient_quota", "code": "credit_balance_exhausted"}}
        )

    with pytest.raises(ProviderQuotaError):
        OpenAIEmbedder("k", http=_client(handler), max_retries=3)
    assert calls == 1


def test_openai_rate_limit_is_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, json={"error": {"type": "requests", "code": "rate_limit_exceeded"}})
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}], "usage": {}})

    emb = OpenAIEmbedder("k", http=_client(handler), max_retries=2)
    assert emb.dim == 1 and calls == 2

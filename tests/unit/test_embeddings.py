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

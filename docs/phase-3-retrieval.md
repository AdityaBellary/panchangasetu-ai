# Phase 3: Vector Database & Embeddings

Status: **implemented, awaiting approval**  ·  Date: 2026-11-07  ·  Provider spend: **₹0 / $0**

## 1. Result

| Production retrieval (Qdrant, real index, 64 questions) | Value |
|---|---|
| recall@5, all | **0.922** |
| English questions → Telugu documents | **0.875** |
| Telugu questions | **0.969** |
| Panchanga collection | **1.000** |
| MRR@10 | 0.744 |
| Search latency p50 / p95 (query embedding + Qdrant) | 20 ms / 24 ms |

Embedding model: **`intfloat/multilingual-e5-large`**, running locally (ONNX via FastEmbed, MIT licence,
no API cost). Index: 172 General + 999 Panchanga documents.

## 2. Bake-off (decision record)

The dataset is `eval/datasets/retrieval_v1.jsonl`: 32 intents asked in English **and** Telugu with
page-level gold labels, plus 6 negative questions. The same corpus and questions were used for every
candidate. Everything ran locally.

| Model (with English section descriptions) | recall@5 | EN | TE | MRR@10 | Notes |
|---|---|---|---|---|---|
| **multilingual-e5-large**, language-aware hybrid | **0.859** | **0.781** | **0.938** | 0.677 | **chosen** |
| BGE-M3, language-aware hybrid | 0.766 | 0.656 | 0.875 | 0.631 | |
| paraphrase-multilingual-mpnet | 0.20–0.36 | | | | too weak for Telugu |
| Telugu sparse (keyword) only | 0.48 | 0.13 | 0.84 | 0.43 | useless for English, strong for Telugu |
| OpenAI text-embedding-3-large / small | n/a | | | | **not run: the account has no credits** (see §6) |

On the production path (real Qdrant, `chunk-3`, event and day-note documents) e5 beat BGE-M3
0.922 vs 0.859 recall@5 and built the index in 1.5 vs 4.3 minutes.

### Findings that changed the design
- **FastEmbed returned unnormalised vectors for e5**, which silently turned cosine search into
  dot-product search. In the first run e5 *looked* worst (0.09). Every embedder now L2-normalises its
  output, and a test guards it.
- **Equal-weight hybrid fusion hurts English queries** (BGE-M3 English recall 0.625 → 0.344), because
  Telugu keyword vectors can't match English words. Hybrid search is therefore **language-aware**:
  sparse + dense RRF only when a query is ≥ 50% Telugu script, otherwise dense only (`hybrid_min_telugu_ratio`).
- **English section descriptions** (a free, one-line gloss per curated section, e.g. "Rasi phalalu:
  yearly and monthly horoscope predictions…") added 3–6 points for English queries.
- **Reranker (`bge-reranker-v2-m3`, int8, top 10):** MRR 0.68 → 0.82 and Telugu recall 0.94, but
  **4–6 s per query on CPU**, which breaks the ≤ 6 s end-to-end budget once the LLM is added. **Off by
  default.** Worth revisiting with a GPU or a hosted reranker.
- **Similarity scores can't decide abstention.** For e5, relevant top-1 cosines range 0.77–0.87
  while irrelevant questions reach 0.78. "I don't have enough information" must come from the evidence
  check in Phase 5, not from a score threshold.

## 3. What gets indexed
- **general:** all General chunks (stotras, samvatsara and rasi phalalu, panchanga vignanam).
- **panchanga:** Panchanga chunks (muhurtham lists, transits, karthelu, masa dates, festival and utsavam
  pages), **plus** 585 festival/utsavam events and 242 book day-notes, **each carrying its exact date**,
  so "when is X?" lands on a dated item.
- **Not indexed:** daily-table and wall-calendar page OCR. Those facts come from the verified
  `panchanga_days` table (Phase 2b), and noisy OCR text must not compete with them.

Each point carries the document, pages, section, heading path, and the date for events, plus a dense
vector and a Telugu sparse vector (word + akshara-bigram features; Qdrant applies IDF).

## 4. Production mechanics
- **Aliases with zero-downtime rebuilds.** Queries hit `psetu_general` / `psetu_panchanga`. A build writes a
  new physical collection (`<alias>__<model>__<chunker>__<timestamp>_<rand>`), then swaps the alias atomically.
  The previous build is kept for instant rollback (`keep_previous_index_builds`).
- **Postgres stays the source of truth.** `index_builds` records every build (model, chunker, fingerprint,
  status). Qdrant can be deleted and rebuilt at any time (this was done once, see §5).
- **Change detection:** the fingerprint covers document ids, rendered text, models and settings, so an unchanged
  corpus is skipped. A changed chunker or model rebuilds automatically.
- **Swapping providers is a settings change:** `PSETU_EMBEDDING_PROVIDER=fastembed|openai`,
  `PSETU_EMBEDDING_MODEL=…`. Switching from BGE-M3 to e5 was exactly that, plus `psetu index`.
- CLI: `psetu index [--force]`, `psetu search <general|panchanga> "<query>"`.
- Eval: `eval/retrieval_bakeoff.py` (model comparison) and `eval/retrieval_eval.py` (production path,
  latency, misses). Reports are in `eval/reports/`.

## 5. Other fixes made in this phase
- **Headings leaked across sections:** "వివాహ ప్రకరణము" from p66 was attached to **267 chunks**, carried
  through every later page without headings. Headings now reset when the curated section changes and
  expire after 2 heading-less pages (`chunk-3`). Running-header spelling variants (సిద్ధాంత / సిద్ధాన్త)
  are matched within 2 edits. Re-chunking was free (cached OCR).
- An older chunking test passed **vacuously** (it asserted over an empty list). Fixed, and the test data
  now uses distinct per-page content.
- **Qdrant client/server mismatch** (client 1.20 vs server 1.15): the server is now `v1.20.0`,
  pinned in step with `qdrant-client>=1.20,<1.21`, and the index was rebuilt from Postgres.
- **OpenAI 429 handling:** "no credits" is now a non-retryable `ProviderQuotaError`. Real rate limits
  are still retried.
- Physical collection names gained a random suffix (rapid or concurrent builds could collide).

## 6. Outstanding
1. **OpenAI comparison is pending credits.** The key works but the account has no balance. Running both
   OpenAI models over the corpus costs about $0.05:
   `uv run python eval/retrieval_bakeoff.py --gloss-only --models openai:text-embedding-3-large openai:text-embedding-3-small`.
2. **Remaining misses (5 of 64)** are mostly English questions phrased far from the Telugu text ("lizard
   falls on the head" → బల్లిపాటు, "Shukla/Krishna paksha"). Phase 4's query rewriting (lexicon
   expansion and Telugu translation of English queries) targets exactly these.
3. One truncated running-header variant (`ల తిరుపతి దేవస్థానములు 74`) is not yet stripped.
4. The evaluation set is small (±5–8 points of noise). It grows in Phase 9.

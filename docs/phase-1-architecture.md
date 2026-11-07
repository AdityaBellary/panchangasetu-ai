# PanchangaSetu AI — Phase 1: Data Analysis & Architecture Proposal

Status: **proposal, awaiting approval**  ·  Date: 2026-11-07

---

## 1. What is actually in the PDFs

| File | Pages | SHA-256 (prefix) | Text layer | Content |
|---|---|---|---|---|
| `TTD_panchangam.pdf` | 176 | `8e7aefcfa99cb195` | Yes, but **not Unicode** (see 1.1) | TTD Siddhanti panchangam for *Sri Parabhava nama samvatsara* (≈ Mar 2026 → Mar 2027) |
| `TTD-Calendar-2026.pdf` | 12 | `7118f58cf7099bf3` | **None** (1 raster image/page, 902×1315 px) | TTD wall calendar 2026, one month per page |
| `TTD-Calendar-PDF-2027-...pdf` | 12 | `ce9a971c2f93dff2` | **None** (1 raster image/page, 1080×1504 px) | TTD wall calendar 2027, one month per page |

Source location today: `~/Downloads/`. The project directory was empty apart from a stray directory named `${GOOGLE_WORKSPACE_MCP_CREDENTIALS_DIR}`, which can be deleted.

### 1.1 Finding 1: The panchangam's text layer is unusable as-is

PyMuPDF extracts about 513K characters but **0 Telugu Unicode code points**. The book is typeset in legacy Telugu DTP fonts: `Priyaanka`, `PallaviBold`, `Kranthi`, `Brahma`, `GowthamiBold/Medium`, `TeluguNumbersThin/Bold`. They map glyphs to Private Use Area code points (U+F020–U+F0FF), for example `…`. Indexing that text would give garbage embeddings that look valid.

Options considered:

| Approach | Verdict |
|---|---|
| Write per-font legacy→Unicode converters | ❌ Needs 9 font-specific glyph tables, plus reordering of vowel signs and conjuncts (vattulu). Brittle, with no maintained public mappings for these fonts. Kept only as a possible later optimisation. |
| **Render each page at 300 DPI (de-rotated) → OCR** | ✅ The pages are born-digital vector text, so renders are perfectly clean and OCR accuracy should be very high. The same pipeline also handles the image-only calendars. |

### 1.2 Finding 2: One PDF holds both knowledge categories

The panchangam book is not purely "calendar". Page-level layout analysis:

| Pages | Layout | Content (sampled) | Target collection |
|---|---|---|---|
| 1–3 | Cover / empty | — | skip |
| 4–~30 | Prose | Nitya dhyana slokas, stotras (Venkatesa Vajrakavacham, Lakshmi Ashtottaram…) | **General** |
| ~31–~60 | Prose | Rasi phalalu (yearly and monthly predictions per rasi) | **General** |
| ~61–96 | Prose | Muhurtha rules (vastu, griharambham, devalaya pratishta…) | **General** |
| 97, 100–151 | **Rotated 90°** dense tables, Telugu numerals | Day-by-day panchangam (tithi, nakshatra, yoga, karana, varjyam, durmuhurtam, festivals, lagna/sunrise tables) | **Panchanga** |
| 152–176 | Prose / tables | To be classified in Phase 2 | per section |

**Consequence:** collection assignment has to happen **per page or section**, not per document. The ingestion pipeline classifies each section, and that classification is stored in metadata so it can be overridden through config.

### 1.3 Finding 3: The calendars are dense, abbreviated and noisy

Each day cell (see the Jan 2026 and Jan 2027 pages) contains the following:

```
 17  [Gregorian]                                    11 [Telugu month day]
          31
   నవమి ఉ.9-43          ← tithi + END time (ఉ=morning, ప=day, సా=evening, రా=night, తె=pre-dawn)
   విశాఖ ఉ.7-14         ← nakshatra + END time
   ప.వ.11-31 ల 1-14     ← varjyam window   (దు. = durmuhurtam)
```

Each row has a fixed weekday Rahukalam (e.g. Sunday సా.4-30–6-00). There is a festivals column and Telugu month and samvatsara headers. The footer states that the times are end times, on an udaya (sunrise) basis.

Problems found:
- The 2026 file is downsampled (902 px wide for a 2646 pt page), so glyphs are about 12 px tall.
- A "LaxmiTeluguTech" watermark sits on top of several 2026 day cells.
- Abbreviations (ప.వ., ఉ.శే.వ.) need a domain lexicon for parsing.

### 1.4 Finding 4: Calendar questions are lookups, not semantic search

"What is today's nakshatram?" has to resolve *today* (IST) to a date and then look up a field. Vector similarity cannot do that reliably, and it is where RAG systems hallucinate dates. **Calendar facts must be extracted into a structured, validated table during ingestion.** Vector search stays for festival descriptions and prose.

---

## 2. Sarvam AI capability check (verified against docs.sarvam.ai, 2026-11-07)

| Need | Sarvam offering | Use it? |
|---|---|---|
| Answer generation (Te/En) | `sarvam-105b`: 128K context, reasoning (default `low`), 10 Indic languages + English, ₹15 per 1M input / ₹60 per 1M output tokens | ✅ **Primary LLM** |
| Multi-turn chat variant | `sarvam-105b-conversations`: 32K context, no reasoning | Optional for query rewriting |
| Query classification | `sarvam-prism`: decision model with probabilities, ₹4 per 1M input, ₹0 output | ✅ Candidate. Cheap, and the probabilities give a confidence score. Validate in Phase 4. |
| OCR / layout / tables | Document Intelligence (Sarvam Vision 2.1): Parse → MD/HTML/JSON with tables; Extract → schema-guided JSON. 22 Indic languages + English, **Telugu included**. ₹0.50/page parse, ₹1/page extract. 20 pages per parse request, 50 MB per file. | ✅ **Primary OCR and calendar extraction** |
| Language ID | Mayura LID: ₹0.35 per 1K characters, short text only | ⚠️ Only as a fallback for romanised Telugu ("ee roju nakshatram enti"). Script detection is free and deterministic. |
| Translation | Mayura (`te-IN` supported) | Not needed in the hot path (see 3.4) |
| **Embeddings** | **None.** No embeddings endpoint, and no official embedding model on Hugging Face. | ❌ Alternative needed |
| **Reranking** | **None** | ❌ Alternative needed |
| Deprecated | `sarvam-m`, `sarvam-30b` | Do not use |

**Embeddings recommendation:** use **BGE-M3** (MIT licence, 100+ languages including Telugu, 8K tokens), self-hosted through Hugging Face Text-Embeddings-Inference (TEI). One model produces **dense and sparse** vectors, which gives hybrid search natively in Qdrant without a separate BM25 stack. As you asked, this is **not locked in yet**. Phase 3 starts with a cross-lingual bake-off: BGE-M3 vs `multilingual-e5-large-instruct` vs one hosted API (e.g. Gemini/Cohere multilingual), scored as recall@5 on English→Telugu and Telugu→Telugu queries.

**Reranker:** `bge-reranker-v2-m3` (multilingual cross-encoder), served from the same TEI image.

**Rough cost:** ingesting about 200 pages costs ≈ ₹100 for parse plus ≈ ₹24 for calendar extract (one-off). A query (~3K input + 300 output tokens on 105B) costs ≈ ₹0.06.

---

## 3. Architecture

### 3.1 Component view

```
                ┌──────────────────────── API tier (stateless, N replicas) ──────────────────────┐
 Consumer ──►   │ FastAPI /v1  ─ auth(API key) ─ rate-limit ─ validation ─ tracing             │
 (API key)      │   │                                                                            │
                │   ▼  Query Orchestrator                                                        │
                │   1 language detect (script ratio; Mayura LID fallback for romanised)          │
                │   2 load session history ──► 3 contextual rewrite (resolve "the next one")     │
                │   4 router → {PANCHANGA_LOOKUP | PANCHANGA_SEARCH | GENERAL | HYBRID | OOS}   │
                │        + slots {date/range, attribute, festival, tithi…}                       │
                │   5 retrieve ── structured SQL (Postgres)  and/or  hybrid vector (Qdrant)      │
                │   6 rerank → dedupe → evidence gate (score thresholds) ──► abstain if weak     │
                │   7 grounded generation (sarvam-105b, cite-by-ID) → 8 citation/fact verifier   │
                └────────┬───────────────┬───────────────┬──────────────┬─────────────────────────┘
                         │               │               │              │
                   Postgres          Qdrant          Redis          Sarvam API / TEI (embed+rerank)
              (source of truth)  (derived index)  (cache, rate
                                                   limits, broker)
                         ▲               ▲
                ┌────────┴───────────────┴─── Ingestion workers (Celery, scale separately) ───┐
                │ discover → hash/dedupe → store raw (object store) → render pages (PyMuPDF,   │
                │ de-rotate, 300 DPI) → OCR (Sarvam Parse | Tesseract fallback) → normalise   │
                │ Telugu (NFC, numerals ౧→1) → section-classify → ① chunk+embed → Qdrant      │
                │                                                  ② calendar extract →       │
                │                       validate (cyclic tithi/nakshatra order, weekday,      │
                │                       cross-source agreement) → panchanga_day table / review│
                └─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 The two collections, and their physical form

| Logical collection | Physical stores | Answers |
|---|---|---|
| **A: Panchanga** | `panchanga_day` (Postgres, one row per date per source) **and** Qdrant collection `panchanga` (chunks of day cells, festival lists, month headers) | "Today's nakshatram", "next Ekadashi", "when is Vaikuntha Ekadasi 2027", Rahukalam on Tuesday |
| **B: General** | Qdrant collection `general` (chunks of stotras, rasi phalalu, muhurtha rules…) | "Significance of Ekadashi", "Kanya rasi phalalu", "rules for griharambham" |

`panchanga_day` columns: date, weekday, source_doc_id, page, bbox, samvatsara, ayana, ritu, masam, paksham, tithi[{name, end_time, end_qualifier}], nakshatra[…], yoga, karana, varjyam[], durmuhurtam[], rahukalam, festivals[], raw_text, extraction_confidence, review_status. Raw OCR text is always kept next to the parsed fields, so an answer can quote the source verbatim.

**HYBRID routing** (e.g. "What is Ekadashi and when is the next one?") runs both retrievers in parallel and merges the results under one evidence budget. The router chooses HYBRID only when it detects two intents, so we avoid querying both collections unnecessarily.

### 3.3 Routing

Two stages, for both cost and reliability:
1. **Deterministic lexicon pass**: a Telugu, English and romanised term lexicon (తిథి/thithi/tithi, నక్షత్రం, ఏకాదశి/ekadashi, రాహుకాలం, dates, "today/tomorrow/next"). This catches most calendar queries with zero LLM cost and extracts the slots.
2. **LLM classifier** (Prism or 105B with a JSON schema) for the remaining ambiguous cases. It returns intent, slots and a confidence score. Below the threshold, the system falls back to HYBRID.

The router is measured against a labelled set of 300+ Telugu, English and mixed questions (target ≥ 95% accuracy).

### 3.4 Cross-language retrieval

- Telugu chunks are indexed as-is with multilingual embeddings, which is the main mechanism.
- **Ingestion-time enrichment** adds an English gloss field to each chunk (e.g. a "Kanya rasi yearly predictions 2026" title, festival names in English transliteration) as a second sparse/dense field. Adding it at ingestion costs nothing at query time and helps English→Telugu recall.
- The query rewriter adds Telugu terms to English queries, e.g. "nakshatram" → నక్షత్రం.
- The **answer language** follows the detected question language. If the question is mixed, the dominant script by character count decides.

### 3.5 Hallucination controls

1. **Evidence gate:** below the reranker threshold, or with no `panchanga_day` row for the date (e.g. 2028), the system returns the fixed abstention message and makes no LLM call.
2. **Closed-book prompt:** context is numbered `[S1]…[Sn]`, and the model must cite IDs and answer only from them.
3. **Post-generation verifier:** every cited ID must exist, and every time, date or number in the answer (e.g. `6-08`) must appear in the cited evidence. If either check fails, the system regenerates once and then abstains.
4. **Structured answers for lookups:** for PANCHANGA_LOOKUP, the facts come from SQL. The LLM only *phrases* them, so it cannot alter a tithi or time.
5. **Source conflicts:** the calendar and the book may disagree for the same date. The system shows both with sources and never silently picks one.

The response includes `sources: [{document, page, snippet, collection}]` and `confidence`.

### 3.6 Sessions

Sessions are stored in Postgres (`sessions`, `messages`, linked to the API key that owns them, so one consumer can't read another's sessions). Redis caches the hot tail of the last N turns. History is used **only for query rewriting**, never as evidence. Each session has a TTL, and a Celery-beat job deletes expired sessions.

### 3.7 Storage and source of truth (your recommendation, adopted)

```
object store (local volume → S3/MinIO): raw PDFs (immutable, by sha256), page renders
Postgres: documents, document_versions, pages(ocr_text, ocr_engine, confidence),
          sections, chunks(text, chunker_version), panchanga_day, ingestion_jobs,
          embedding_runs(model, dim, chunker_version), sessions, messages,
          api_keys(hashed), usage_events
Qdrant:   derived only. Fully rebuildable from Postgres chunks via a re-index job.
```

Incremental ingestion compares content hashes, so unchanged files are skipped, modified files create a new version, and deleted files are tombstoned and their points removed. Changing the embedding model or chunker creates a new Qdrant collection (`general_v2`), which is then swapped in through an alias with zero downtime.

### 3.8 Chunking (Telugu-aware)

- Chunks follow structure: **section heading → paragraph → sentence**. Telugu sentences are split on `।`, `.` and `?`, and grapheme clusters are never split, so vowel signs and virama conjuncts stay intact. Sizes are counted in tokens of the embedder's tokenizer, not in characters.
- Each chunk carries its heading path (e.g. `కన్యారాశి ఫలితములు > నెలవారీ ఫలితములు > మే`) as a contextual prefix.
- Each calendar day is one atomic chunk, and each monthly festival list is one chunk.
- Each stotra is one chunk, or split per verse group if it is long.

---

## 4. Technology stack, compared with your starting table

| Component | Your proposal | Recommendation | Why |
|---|---|---|---|
| Backend | FastAPI | **FastAPI** ✅ | Async, OpenAPI out of the box |
| LLM | Sarvam | **`sarvam-105b`** ✅ (+ `sarvam-prism` for classification) | Verified current; 30B and M are deprecated |
| Embeddings | Sarvam | **BGE-M3 via TEI** (pending bake-off) | Sarvam has no embedding API |
| Vector DB | Qdrant | **Qdrant** ✅ | Named dense+sparse vectors, payload filters, aliases for zero-downtime re-index. pgvector is viable at this corpus size but has weaker hybrid search; Milvus is too much to operate for this scale. |
| PDF extraction | PyMuPDF + Docling | **PyMuPDF only** | Docling adds a heavy torch dependency and still can't decode the legacy fonts or the image-only pages. Sarvam Parse covers layout and tables. |
| Telugu OCR | Tesseract | **Sarvam Vision primary, Tesseract `tel` fallback** | Behind an `OCREngine` interface. Tesseract also serves as the offline baseline in the evaluation. |
| Reranking | Multilingual reranker | **bge-reranker-v2-m3 (TEI)** ✅ | |
| Sessions | Postgres + Redis | ✅ | |
| Jobs | Celery | **Celery + Redis broker** ✅ | |
| Monitoring | Prom + Grafana + OTel | ✅ + **structlog** JSON logs | |
| Deploy | Docker | Docker + Compose (api, worker, beat, postgres, qdrant, redis, tei) | |

Every provider sits behind a Protocol interface (`LLMProvider`, `Embedder`, `Reranker`, `OCREngine`, `CalendarExtractor`, `LanguageDetector`) and is chosen through settings.

---

## 5. Proposed directory structure

```
panchangasetu-ai/
├── pyproject.toml  (uv)        ├── docker/ (Dockerfile.api, Dockerfile.worker)
├── docker-compose.yml          ├── .env.example
├── src/panchangasetu/
│   ├── config.py               # pydantic-settings, all tunables
│   ├── api/v1/                 # routers: query, sessions, documents, health, admin_keys
│   ├── api/deps.py, errors.py, middleware.py (auth, rate-limit, tracing)
│   ├── core/                   # domain models, interfaces (Protocols)
│   ├── providers/              # sarvam/, tei/, tesseract/  (implement interfaces)
│   ├── ingestion/              # discover, hashing, render, ocr, normalize, sectionize,
│   │                           # chunking, calendar_extract, validate, pipeline, tasks
│   ├── telugu/                 # numerals, abbreviations lexicon, tithi/nakshatra tables, script detect
│   ├── retrieval/              # router, rewriter, structured (SQL), vector (Qdrant), rerank, evidence
│   ├── generation/             # prompts, grounded answerer, verifier
│   ├── sessions/               # store, memory, cleanup
│   ├── storage/                # db (SQLAlchemy + Alembic), object store, qdrant client
│   └── observability/          # logging, metrics, tracing
├── migrations/                 # alembic
├── sdk/python/panchangasetu_client/
├── eval/                       # datasets (jsonl), runners, metrics, reports
├── tests/{unit,integration,e2e}/
├── scripts/                    # inspect_pdfs.py, reindex.py, create_api_key.py
└── docs/
```

---

## 6. Production-readiness targets (proposed)

| Metric | Target |
|---|---|
| Router accuracy (labelled set) | ≥ 95% |
| Retrieval recall@5, Te→Te / En→Te | ≥ 0.90 / ≥ 0.85 |
| Calendar field accuracy vs hand-verified gold (≥ 60 days) | ≥ 99% after validation and review |
| Hallucination rate on negative set | **0%** fabricated facts; ≥ 95% correct abstentions |
| Citation accuracy (cited page actually supports the claim) | ≥ 95% |
| Latency p95: structured lookup / RAG answer | ≤ 2.5 s / ≤ 6 s (LLM-bound); cache hit ≤ 150 ms |
| Query-time PDF access | 0 (enforced: API image has no PDF deps or volume) |

---

## 7. Risks and open items

1. **Calendar OCR quality (2026 file):** the low resolution and watermark are the main accuracy risk. Mitigations: schema-guided extraction, validation rules (tithi and nakshatra advance cyclically; weekday Rahukalam is constant), cross-checking against the book's daily tables (same Siddhanti), and a review queue for low-confidence cells. Getting a higher-resolution 2026 source would help a lot.
2. **Coverage gaps:** the book covers about Mar 2026 → Mar 2027 and the calendars cover Jan 2026 → Dec 2027. Questions outside that range must abstain.
3. **"Today" semantics:** the system will default to **Asia/Kolkata**, configurable per request. Panchangam times are as printed (TTD/Tirupati basis), and we won't recompute for other locations.
4. **Sarvam Parse on rotated pages:** not documented, so we de-rotate ourselves before sending pages.
5. **Romanised Telugu** ("ee roju tithi enti") needs lexicon coverage and the LID fallback. It will be included in the evaluation set.

## 8. Decisions needed before Phase 2

1. **Sarvam API key**: is one available? It is needed to spike Sarvam Parse on ~10 sample pages, which is the first task of Phase 2.
2. **Embeddings**: is a self-hosted BGE-M3 (CPU container) acceptable, with a hosted API evaluated only as a bake-off contender?
3. **Data location**: OK to copy the PDFs into `./data/raw/` (gitignored) as the configured input directory?
4. **Human review**: is a review step for low-confidence calendar cells acceptable, or must ingestion be fully unattended (accepting lower accuracy)?
5. **Docling dropped** and **structured `panchanga_day` table added**: do you agree with both?

# Phase 2: PDF Extraction & Ingestion

Status: **implemented, awaiting approval**  ·  Date: 2026-11-07

## 1. What was built

```
discover ─► hash (streaming SHA-256) ─► reconcile with Postgres
   new / modified / unchanged / duplicate / deleted
        │
        ▼  per version, page by page (lazy, bounded memory)
inspect page ─► kind: unicode text layer | legacy-font text layer | image-only | empty
        │         + upright rotation from text direction (the /Rotate flag is unreliable)
        ├─ unicode text layer ─► use directly (no OCR)
        ├─ skip rule / empty  ─► record, no OCR
        └─ otherwise ─► render upright PNG ─► object store ─► batch ─► OCR engine (per rules)
                                                   └─ on failure / missing pages ─► fallback engine
        ▼
normalise Telugu (NFC, ౦-౯→0-9, zero-width cleanup) ─► classify page → collection
        ▼
strip running headers ─► structure-aware Telugu chunking ─► chunks (Postgres)
```

| Area | Files |
|---|---|
| Settings (all tunables, env-driven) | `src/panchangasetu/config.py`, `.env.example` |
| Interfaces / domain types | `core/interfaces.py` (`OCREngine`, `ObjectStore`), `core/types.py` |
| Discovery + fingerprinting | `ingestion/discovery.py` |
| Page inspection, orientation, rendering | `ingestion/pdf_pages.py` |
| OCR providers | `providers/ocr/sarvam.py`, `providers/ocr/tesseract.py`, `providers/ocr/factory.py` |
| Telugu text utilities | `telugu/text.py` (normalisation, script stats, akshara segmentation, sentences) |
| Panchanga lexicon | `telugu/lexicon.py` (tithis, nakshatras, masas, varas, terms; reused in Phase 4 routing) |
| Page → collection classification | `ingestion/classification.py` + curator rules `config/collections.yaml` |
| Chunking | `ingestion/chunking.py` |
| Orchestration | `ingestion/pipeline.py` |
| Storage (source of truth) | `storage/models.py`, `storage/db.py`, `storage/object_store.py`, `migrations/` |
| Operator CLI | `cli.py`: `psetu db-upgrade`, `psetu ingest [--rechunk/--reprocess/--only/--engine]`, `psetu status`, `psetu show-page` |
| Local infra | `docker-compose.yml` (Postgres 17 on 127.0.0.1:55432) |

## 2. Key decisions (and the evidence behind them)

1. **Rasterise before OCR, always.** The panchangam's text layer is legacy-font PUA codes. We
   render pages ourselves and send *image-only* PDFs to Sarvam, so no engine can be fooled
   by the broken text layer.
2. **Orientation from text direction, not `/Rotate`.** Pages 100–151 declare `/Rotate 90` yet
   render sideways. MuPDF reports glyph line direction in unrotated space even for legacy fonts,
   which gives the exact upright rotation. This was verified visually on pages 22, 97 and 100.
3. **OCR engine per page range (config, not code).** The spike on 3 pages showed:
   - Sarvam Parse gives clean Telugu prose with layout blocks and bboxes, but on the book's daily
     tables it **drops most of each row** and garbles the custom Telugu-numeral glyphs.
   - Tesseract (`tessdata_best` tel) reads those table lines nearly verbatim (`గురు అమావాస్య 1-50 ఉ.7-06 …`).
   → `config/collections.yaml` routes p97 and p100–151 to Tesseract and everything else to Sarvam.
4. **Sarvam `image` blocks are excluded.** They contain a *model-written description* of a picture,
   not page text. In the spike, one invented the date "12-01-2025". Indexing them would let a
   hallucination masquerade as evidence.
5. **SDK workaround.** `sarvamai` 0.1.43 cannot decode the live results payload (`type: "digitise"`),
   so results are fetched with a raw HTTP call. Submit and status still go through the SDK.
6. **Raw OCR payloads are kept** (`data/objects/ocr/...json`: blocks, layout tags, bboxes). That
   enables re-deriving text without paying again, and highlighting cited regions later.
7. **Akshara-safe splitting.** The `regex` module's `\X` splits Telugu conjuncts at the virama
   (`స్` | `వా`), because it predates Unicode 15.1's GB9c rule. `graphemes()` re-joins them, and a
   test guards this.
8. **Per-page, not per-document, collections.** Running headers are auto-detected (lines recurring
   on ≥30% of a document's pages) and stripped before chunking.

## 3. Incremental behaviour (all covered by tests)

| Situation | Behaviour |
|---|---|
| Unchanged file | skipped (no OCR, no rechunk) |
| Modified file (same path, new bytes) | new `document_versions` row becomes current; old kept for audit |
| Byte-identical copy under another name | recorded as `duplicate`, never reprocessed |
| File removed | document tombstoned (`deleted`), versions no longer current |
| OCR batch fails | fallback engine tries the missing pages; still-failed pages → version `partial`, retried next run (only those pages) |
| Chunking logic changes | bump `PSETU_CHUNKER_VERSION` or run `--rechunk`: rechunks from cached text, **zero OCR cost** |
| OCR/normalisation changes | bump `PSETU_EXTRACTION_VERSION` or run `--reprocess` |
| Two workers start at once | Postgres advisory lock; the second gets `IngestionBusyError` |

## 4. Tests

- `tests/unit` (51): Telugu normalisation, akshara segmentation, script detection, rules and heuristic
  classification, chunker invariants (max tokens, no syllable cut, no collection mixing, page
  spans, header stripping, overlap), page kinds and orientation (all 4 directions with misleading
  `/Rotate`), Sarvam engine against a mocked HTTP API (page mapping, partial jobs, 402 not retried,
  429 retried), and the full pipeline lifecycle on SQLite with a fake OCR engine.
- `tests/integration` (3): pipeline on real Postgres (isolated schema), advisory-lock exclusion,
  real Tesseract on the real p100 Ugadi row.
- `ruff` clean, `mypy --strict` clean.

## 5. Results on the real corpus

| | TTD_panchangam.pdf | TTD-Calendar-2026 | TTD-Calendar-2027 |
|---|---|---|---|
| Pages | 176 (3 cover skipped) | 12 | 12 |
| Page kinds | 173 legacy-font, 3 image | 12 image-only | 12 image-only |
| OCR engine | Sarvam 120 · Tesseract 53 | Sarvam 12 | Sarvam 12 |
| Failed pages | 0 | 0 | 0 |
| Pages → collection | general 84 · panchanga 89 | panchanga 12 | panchanga 12 |
| Chunks (`chunk-2`) | general 174 · panchanga 227 | 20 | 13 |

- **Cost:** about 147 Sarvam pages × ₹0.50 ≈ **₹74** (3-page spike + full run). Re-runs and re-chunks cost ₹0,
  because OCR is cached. Verify against the Sarvam dashboard.
- **Time:** about 20 min for the first full run (Sarvam ≈ 1 min per 10-page job; Tesseract ≈ 10 s/page). Incremental
  no-op run: a few seconds.
- **Chunks:** 434 total, 46–508 approx. tokens (target 350, max 512). No running headers remain.
- **Classification:** curator rules map every page of the book (sections listed in `config/collections.yaml`).
  The text heuristic alone, scored against those rules on all 197 content pages: **accuracy 0.909,
  Panchanga precision 0.970, recall 0.867**. Its misses are the festival table, where dates are bare
  day numbers. That's good enough as the fallback for new documents, and is why the rules exist.
- **Spot checks against page images:** prose (p31) and the TTD festival table (p155) are near-verbatim,
  including table rows. Daily tables (Tesseract) keep each day's line intact.

### Issues found and fixed during the run
1. The heuristic routed concept-explanation pages (paksham types, nakshatra padas) to Panchanga
   because they use the vocabulary heavily → it now requires **dated facts** (times, full dates).
2. Numeric lists like `2-3-5-7` counted as dates → numeric dates now require a 4-digit year.
3. Recto/verso running headers (each on ~26% of pages) weren't stripped at a 30% threshold → threshold
   lowered to 15%, but only lines with ≥ 10 letters qualify, so recurring **data** such as `6-20` or
   `భో.3-34 అహః 29-55` is never stripped.
4. The overlap carry could push a chunk past `max_tokens` (522 > 512) → re-checked. Tiny title/fragment
   chunks are merged into their section neighbour.

## 6. Outstanding issues and next steps

1. **Wall calendars need cell-level extraction (proposed Phase 2b).** Whole-page Sarvam OCR reads the
   text but loses which day cell it belongs to, so those chunks are **not trustworthy for date facts**.
   Plan: detect the 7×N grid, crop each day cell, OCR or extract per cell, validate (weekday matches date;
   tithi/nakshatra sequences advance cyclically), and cross-check against the book.
2. **Structured `panchanga_day` / festival tables (Phase 2b).** The book's daily tables (p100–151) and
   festival table (p152–163) are now cleanly extracted and parseable into rows. Promising technique: the
   legacy-font text layer is useless as text, but its **glyph positions are exact**, so it can segment
   table rows and cells precisely for per-cell OCR.
3. The 2026 calendar scan is low-resolution (902 px) with a watermark. A better source would help.
4. Token counts are approximate (graphemes/3) until the embedder's tokenizer is fixed in Phase 3.
5. Superseded `chunk-1` rows are kept for audit; a retention/cleanup job comes with the Phase 3 indexer.
6. `sarvamai` SDK lags the live API (results decoding); our HTTP path is covered by mocked tests. A
   paid contract test (`-m sarvam`) can be added to catch future API drift.
7. Object store is local disk; S3/MinIO implementation of the same protocol is a deployment task (Phase 8).

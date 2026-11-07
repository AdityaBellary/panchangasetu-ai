# PanchangaSetu AI

Telugu document intelligence and grounded RAG platform. It answers Telugu and English questions
about TTD panchangam and calendar PDFs, with citations, and abstains when the documents
don't support an answer.

Status: Phase 6 (conversations and sessions) complete. See `docs/phase-1-architecture.md` through
`docs/phase-6-conversations.md`.

## Quick start (local)

Prerequisites: Python 3.12, [uv](https://docs.astral.sh/uv/), Docker, Tesseract (`brew install tesseract`).

```bash
uv sync
cp .env.example .env        # then set SARVAM_API_KEY
docker compose up -d postgres qdrant
uv run psetu db-upgrade
```

Telugu Tesseract model (fallback OCR engine):

```bash
mkdir -p data/tessdata && cp "$(brew --prefix)/share/tessdata/"{eng,osd}.traineddata data/tessdata/
curl -L -o data/tessdata/tel.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/tel.traineddata
```

Ingest the PDFs placed in `data/raw/`:

```bash
uv run psetu ingest           # incremental: only new/changed files are processed
uv run psetu status
uv run psetu show-page TTD_panchangam.pdf 31
uv run psetu extract-structured   # panchanga_days / panchanga_events (also run by ingest)
uv run psetu index                # build/refresh the Qdrant index (local e5 embeddings)
uv run psetu search general "కుజదోషం అంటే ఏమిటి?"
uv run psetu retrieve "ఈరోజు నక్షత్రం ఏమిటి?" "When is the next Ekadashi?"   # route + evidence
uv run psetu ask -v "What is Kuja dosha?"   # grounded answer with citations (uses Sarvam chat credits)
uv run psetu chat -v                        # terminal chat with follow-ups ("and tomorrow?"); /new resets
```

## Tests

```bash
uv run pytest                 # unit tests (no network, no services)
uv run pytest -m integration  # needs docker compose Postgres + Tesseract
uv run ruff check src tests && uv run mypy src
```

Tests marked `sarvam` call the real Sarvam API and are excluded by default.

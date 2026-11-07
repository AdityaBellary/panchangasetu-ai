# PanchangaSetu AI

Telugu document intelligence and grounded RAG platform. It answers Telugu and English questions
about TTD panchangam and calendar PDFs, with citations, and abstains when the documents
don't support an answer.

Status: Phase 2b (structured calendar facts) complete. See `docs/phase-1-architecture.md`,
`docs/phase-2-ingestion.md` and `docs/phase-2b-structured-calendar.md`.

## Quick start (local)

Prerequisites: Python 3.12, [uv](https://docs.astral.sh/uv/), Docker, Tesseract (`brew install tesseract`).

```bash
uv sync
cp .env.example .env        # then set SARVAM_API_KEY
docker compose up -d postgres
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
```

## Tests

```bash
uv run pytest                 # unit tests (no network, no services)
uv run pytest -m integration  # needs docker compose Postgres + Tesseract
uv run ruff check src tests && uv run mypy src
```

Tests marked `sarvam` call the real Sarvam API and are excluded by default.

# PanchangaSetu API + web chat (also runs ingestion / indexing commands).
# Data (PDFs, page images, OCR cache), the embedding-model cache and secrets are mounted at
# runtime; the image holds only code and configuration.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Tesseract is the fallback OCR engine for ingestion; the API itself does not need it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-tel \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv

WORKDIR /app
# Dependencies first so code changes don't invalidate the (large) dependency layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY config ./config
COPY migrations ./migrations
COPY alembic.ini ./
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 10001 psetu && mkdir -p /app/data && chown psetu /app/data
USER psetu

ENV PATH=/app/.venv/bin:$PATH \
    PSETU_API_HOST=0.0.0.0 \
    PSETU_API_PORT=8000 \
    PSETU_LOG_JSON=true
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"

# Apply migrations, then serve. One process: rate limits and metrics are per process.
CMD ["sh", "-c", "psetu db-upgrade && exec psetu serve"]

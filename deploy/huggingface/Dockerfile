# PanchangaSetu on a Hugging Face Docker Space (free CPU tier: 2 vCPU, 16 GB).
# The Space repository is assembled by deploy_space.py from a commit on main; this file is its Dockerfile.
# Postgres (Neon) and Qdrant (Qdrant Cloud) live outside the Space because Space storage is wiped on
# restart. The embedding model is baked into the image so a cold start doesn't download 2 GB.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv

# Spaces run the container as uid 1000.
RUN useradd --create-home --uid 1000 user && mkdir -p /app/data && chown -R user /app
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY config ./config
COPY migrations ./migrations
COPY alembic.ini start.sh fetch_model.py ./
RUN uv sync --frozen --no-dev && chmod +x start.sh

ENV PATH=/app/.venv/bin:$PATH \
    PSETU_API_HOST=0.0.0.0 \
    PSETU_API_PORT=7860 \
    PSETU_MODEL_CACHE_DIR=/app/data/models \
    PSETU_OBJECT_STORE_DIR=/app/data/objects \
    PSETU_LOG_JSON=true

USER user
RUN python fetch_model.py

EXPOSE 7860
CMD ["./start.sh"]

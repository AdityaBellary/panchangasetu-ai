"""One-time: copy this laptop's processed data to the hosted services, so nothing is OCR'd again.

    uv run python deploy/huggingface/seed_data.py database   # local Postgres -> Neon
    uv run python deploy/huggingface/seed_data.py index      # local chunks -> Qdrant Cloud (local embeddings)
    uv run python deploy/huggingface/seed_data.py pages      # data/objects/pages -> private HF dataset

Reads .env.deploy (HF_TOKEN, HF_USERNAME, NEON_DATABASE_URL, QDRANT_URL, QDRANT_API_KEY). Conversations,
LLM-call records and API keys are not copied: production starts with its own keys and history.
Needs the docker compose Postgres running (`docker compose up -d postgres`).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PG_CONTAINER = "panchangasetu-ai-postgres-1"
PG_CLIENT = "postgres:17.6-alpine"  # same major as the local server; restores fine into newer Neon
SKIP_DATA = ["chat_sessions", "chat_turns", "llm_calls", "api_keys"]
PAGES_DATASET = "panchangasetu-pages"


def deploy_env() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (ROOT / ".env.deploy").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("'\"")
    return values


def direct(url: str) -> str:
    """Neon's pooled endpoint (host "...-pooler...") keeps session settings such as the dump's empty
    search_path on connections it then hands to others; the direct endpoint doesn't."""
    return url.replace("-pooler.", ".", 1)


def database(cfg: dict[str, str]) -> None:
    # The URL holds the password: it goes in through the environment, never on a command line that an
    # error message would print.
    os.environ["PGURL"] = direct(cfg["NEON_DATABASE_URL"])
    shell = ["sh", "-c", 'exec psql "$PGURL" -v ON_ERROR_STOP=1 -q "$@"', "psql"]
    psql = ["docker", "run", "--rm", "-i", "-e", "PGURL", PG_CLIENT, *shell]
    existing = subprocess.run(
        [*psql, "-Atc", "select count(*) from information_schema.tables where table_schema = 'public'"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()  # fmt: skip
    if existing != "0" and "--force" not in sys.argv:
        sys.exit(f"Neon already has {existing} tables; refusing to overwrite (--force drops and recopies)")
    if existing != "0":
        subprocess.run([*psql, "-c", "drop schema public cascade; create schema public;"], check=True)
    dump = ["docker", "exec", PG_CONTAINER, "pg_dump", "-U", "psetu", "-d", "psetu"]
    dump += ["--no-owner", "--no-privileges"]
    dump += [f"--exclude-table-data={t}" for t in SKIP_DATA]
    with subprocess.Popen(dump, stdout=subprocess.PIPE) as producer:
        subprocess.run(psql, stdin=producer.stdout, check=True)
    if producer.returncode:
        sys.exit("pg_dump failed")
    counts = subprocess.run(
        [*psql, "-Atc", "select count(*) from chunks", "-c", "select count(*) from panchanga_days"],
        check=True, capture_output=True, text=True,
    ).stdout.split()  # fmt: skip
    print(f"copied to Neon: {counts[0]} chunks, {counts[1]} panchanga days")


def index(cfg: dict[str, str]) -> None:
    env = dict(os.environ, PSETU_QDRANT_URL=cfg["QDRANT_URL"], PSETU_QDRANT_API_KEY=cfg["QDRANT_API_KEY"])
    subprocess.run(["uv", "run", "psetu", "index"], check=True, env=env, cwd=ROOT)


def pages(cfg: dict[str, str]) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=cfg["HF_TOKEN"])
    repo = f"{cfg['HF_USERNAME']}/{PAGES_DATASET}"
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path=ROOT / "data/objects/pages")
    print(f"page images uploaded to the private dataset {repo}")


STEPS = {"database": database, "index": index, "pages": pages}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STEPS:
        sys.exit(__doc__)
    try:
        STEPS[sys.argv[1]](deploy_env())
    except subprocess.CalledProcessError as exc:  # its message would include the command line
        sys.exit(f"{sys.argv[1]} failed (exit {exc.returncode}); see the output above")

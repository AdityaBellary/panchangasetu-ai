"""Deploy the checked-out commit to the Hugging Face Space (run by .github/workflows/deploy.yml).

1. Creates the Space if it doesn't exist (public, Docker SDK).
2. Sets the Space's secrets and variables from this process's environment.
3. Uploads the app (src, config, migrations, lockfile) plus this folder's Dockerfile, start.sh and README as
   the Space repository, replacing whatever was there. The Space then rebuilds.
4. Waits for the Space to come up and checks /healthz, so a failed build fails the workflow.

Environment: HF_TOKEN (write; used here only), HF_SPACE (owner/name), PAGES_DATASET (owner/name), GIT_SHA,
the app secrets listed in SECRETS below, and optionally HF_PAGES_READ_TOKEN. Values are never printed.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
APP_FILES = ["src", "config", "migrations", "alembic.ini", "pyproject.toml", "uv.lock"]
SPACE_FILES = ["Dockerfile", "start.sh", "fetch_model.py", "README.md"]

# Space secret name -> environment variable that holds it here.
SECRETS = {
    "SARVAM_API_KEY": "SARVAM_API_KEY",
    "GEMINI_API_KEY": "GEMINI_API_KEY",
    "PSETU_DATABASE_URL": "NEON_DATABASE_URL",
    "PSETU_QDRANT_URL": "QDRANT_URL",
    "PSETU_QDRANT_API_KEY": "QDRANT_API_KEY",
    "PSETU_API_METRICS_TOKEN": "PSETU_API_METRICS_TOKEN",
}
# The Space is a public web app, so it never gets the write token that deploys it: page images are read
# with a separate read-only token scoped to the dataset. Without one, the app runs without page images.
PAGES_TOKEN = "HF_PAGES_READ_TOKEN"
BUILD_TIMEOUT_S = 40 * 60


def env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        sys.exit(f"missing environment variable {name}")
    return value


def sqlalchemy_url(url: str) -> str:
    """Neon hands out postgresql://...; the app's driver is psycopg 3. The direct endpoint is used, not
    the pooler ("-pooler" host), which can hand one client's session settings to another."""
    url = url.replace("-pooler.", ".", 1)
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix) :]
    return url


def assemble(target: Path) -> None:
    for name in APP_FILES:
        src = ROOT / name
        if src.is_dir():
            shutil.copytree(src, target / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(src, target / name)
    for name in SPACE_FILES:
        shutil.copy2(HERE / name, target / name)


def wait_until_up(api: HfApi, space: str, sha: str) -> None:
    host = space.replace("/", "-").replace("_", "-").replace(".", "-").lower()
    # Right after the upload the previous build may still answer; wait for the rebuild to start first.
    started = time.time()
    while time.time() - started < 180 and api.get_space_runtime(space).stage == "RUNNING":
        time.sleep(10)
    deadline = time.time() + BUILD_TIMEOUT_S
    stage = ""
    while time.time() < deadline:
        runtime = api.get_space_runtime(space)
        if runtime.stage != stage:
            stage = runtime.stage
            print(f"space stage: {stage}", flush=True)
        if stage in ("BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"):
            sys.exit(f"space failed: {stage}; see the Space's logs on huggingface.co")
        if stage == "RUNNING":
            try:
                with urllib.request.urlopen(f"https://{host}.hf.space/healthz", timeout=20) as r:
                    if r.status == 200:
                        print(f"deployed {sha[:7]}: https://{host}.hf.space", flush=True)
                        return
            except Exception:
                pass  # still starting: the model and migrations take a minute
        time.sleep(20)
    sys.exit("timed out waiting for the Space to come up")


def main() -> None:
    token, space, sha = env("HF_TOKEN"), env("HF_SPACE"), env("GIT_SHA")
    api = HfApi(token=token)
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True, private=False)

    for name, source in SECRETS.items():
        value = env(source)
        api.add_space_secret(space, name, sqlalchemy_url(value) if name == "PSETU_DATABASE_URL" else value)
    if os.environ.get(PAGES_TOKEN, "").strip():
        api.add_space_secret(space, "PSETU_HF_TOKEN", env(PAGES_TOKEN))
        api.add_space_variable(space, "PSETU_PAGES_DATASET", env("PAGES_DATASET"))
    else:
        print(f"{PAGES_TOKEN} not set: the Space runs without page images", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        assemble(folder)
        api.upload_folder(
            repo_id=space,
            repo_type="space",
            folder_path=folder,
            delete_patterns="*",
            commit_message=f"Deploy {sha[:7]}",
        )
    print(f"uploaded {sha[:7]} to {space}; waiting for the build", flush=True)
    wait_until_up(api, space, sha)


if __name__ == "__main__":
    main()

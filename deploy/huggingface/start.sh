#!/bin/sh
# Space entrypoint: fetch page images in the background, apply migrations, serve.
set -e

if [ -n "$PSETU_HF_TOKEN" ] && [ -n "$PSETU_PAGES_DATASET" ]; then
  # Page images (the "view source" pictures) live in a private dataset, not in the public Space.
  # The app answers while they download; a source page opens once its image has arrived.
  python - <<'EOF' &
import os
from huggingface_hub import snapshot_download

try:
    snapshot_download(
        repo_id=os.environ["PSETU_PAGES_DATASET"],
        repo_type="dataset",
        local_dir=os.path.join(os.environ["PSETU_OBJECT_STORE_DIR"], "pages"),
        token=os.environ["PSETU_HF_TOKEN"],
    )
    print("page images ready", flush=True)
except Exception as exc:  # the app still answers without them
    print(f"page images unavailable: {exc}", flush=True)
EOF
fi

psetu db-upgrade
exec psetu serve

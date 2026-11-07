"""Content-addressed blob storage for raw PDFs and page renders.

Local filesystem implementation; an S3/MinIO implementation can satisfy the same
`ObjectStore` protocol without changes elsewhere.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


class LocalObjectStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"object key escapes store root: {key!r}")
        return path

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: never leave a half-written blob behind a valid key.
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def put_file(self, key: str, source: Path) -> None:
        """Copy a (possibly large) file in without reading it fully into memory."""
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as out, source.open("rb") as src:
                while chunk := src.read(1 << 20):
                    out.write(chunk)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def local_path(self, key: str) -> Path:
        return self._path(key)

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)


def raw_pdf_key(sha256: str) -> str:
    return f"raw/{sha256[:2]}/{sha256}.pdf"


def page_image_key(sha256: str, page_number: int, rotation: int) -> str:
    return f"pages/{sha256[:2]}/{sha256}/p{page_number:04d}_r{rotation}.png"


def ocr_payload_key(sha256: str, page_number: int, engine: str) -> str:
    return f"ocr/{sha256[:2]}/{sha256}/p{page_number:04d}_{engine}.json"

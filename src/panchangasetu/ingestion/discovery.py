"""Discover source PDFs and fingerprint them (streaming SHA-256, never whole-file reads)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

_PDF_MAGIC = b"%PDF-"


@dataclass(frozen=True, slots=True)
class DiscoveredFile:
    path: Path  # absolute
    source_path: str  # POSIX path relative to the input dir: the document's stable identity
    sha256: str
    size_bytes: int


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def is_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            # The header may be preceded by junk bytes; the spec allows it within 1 KiB.
            return _PDF_MAGIC in fh.read(1024)
    except OSError:
        return False


def discover_pdfs(input_dir: Path) -> list[DiscoveredFile]:
    root = input_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {root}")
    found: list[DiscoveredFile] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".pdf" or path.name.startswith("."):
            continue
        if not is_pdf(path):
            continue
        found.append(
            DiscoveredFile(
                path=path,
                source_path=path.relative_to(root).as_posix(),
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
            )
        )
    return found

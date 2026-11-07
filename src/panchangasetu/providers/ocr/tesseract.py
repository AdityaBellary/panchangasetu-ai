"""Tesseract OCR engine (offline fallback and evaluation baseline).

Invoked as a subprocess (no Python wrapper dependency). Uses `tessdata_best` Telugu models.
Per-page mean word confidence is computed from TSV output so low-quality pages can be flagged.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from functools import cached_property
from pathlib import Path

from panchangasetu.core.interfaces import OCRError
from panchangasetu.core.types import OCRPage, PageImage


class TesseractOCREngine:
    name = "tesseract"

    def __init__(
        self,
        *,
        cmd: str = "tesseract",
        langs: str = "tel+eng",
        tessdata_dir: Path | None = None,
        timeout_s: float = 300.0,
        psm: int = 3,
    ) -> None:
        self._cmd = cmd
        self._langs = langs
        self._tessdata_dir = tessdata_dir
        self._timeout_s = timeout_s
        self._psm = psm

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self._tessdata_dir is not None:
            env["TESSDATA_PREFIX"] = str(self._tessdata_dir.resolve())
        env.setdefault("OMP_THREAD_LIMIT", "1")  # we parallelise at the page level, not inside
        return env

    @cached_property
    def version(self) -> str:
        try:
            out = subprocess.run(
                [self._cmd, "--version"], capture_output=True, text=True, timeout=30, env=self._env()
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise OCRError(f"tesseract not available: {exc}", retryable=False) from exc
        first = (out.stdout or out.stderr).splitlines()[0] if (out.stdout or out.stderr) else "unknown"
        return f"{first.strip()}:{self._langs}:psm{self._psm}"

    def _run(self, png: bytes, fmt: str) -> str:
        args = [self._cmd, "stdin", "stdout", "-l", self._langs, "--psm", str(self._psm)]
        if fmt == "tsv":
            # Use the variable directly: the `tsv` config file lives in tessdata/configs,
            # which a custom TESSDATA_PREFIX need not contain.
            args += ["-c", "tessedit_create_tsv=1"]
        try:
            proc = subprocess.run(
                args, input=png, capture_output=True, timeout=self._timeout_s, env=self._env()
            )
        except subprocess.TimeoutExpired as exc:
            raise OCRError("tesseract timed out", retryable=True) from exc
        except OSError as exc:
            raise OCRError(f"tesseract not available: {exc}", retryable=False) from exc
        if proc.returncode != 0:
            raise OCRError(f"tesseract failed: {proc.stderr.decode(errors='replace')[:300]}", retryable=False)
        return proc.stdout.decode("utf-8", errors="replace")

    def raw_tsv(self, png: bytes) -> str:
        """Word-level TSV (boxes, confidences) for layout work such as calendar grid fitting."""
        return self._run(png, "tsv")

    def raw_text(self, png: bytes) -> str:
        from panchangasetu.telugu.text import normalize_telugu

        return normalize_telugu(self._tsv_to_text_and_conf(self._run(png, "tsv"))[0])

    @staticmethod
    def _tsv_to_text_and_conf(tsv: str) -> tuple[str, float | None]:
        lines: dict[tuple[int, int, int], list[str]] = {}
        confs: list[float] = []
        for row in tsv.splitlines()[1:]:
            cols = row.split("\t")
            if len(cols) < 12 or cols[0] != "5":  # level 5 = word
                continue
            word = cols[11].strip()
            if not word:
                continue
            key = (int(cols[2]), int(cols[3]), int(cols[4]))  # block, paragraph, line
            lines.setdefault(key, []).append(word)
            conf = float(cols[10])
            if conf >= 0:
                confs.append(conf)
        text_lines: list[str] = []
        prev_par: tuple[int, int] | None = None
        for (block, par, _line), words in lines.items():
            if prev_par is not None and (block, par) != prev_par:
                text_lines.append("")  # paragraph break
            text_lines.append(" ".join(words))
            prev_par = (block, par)
        mean_conf = round(sum(confs) / len(confs) / 100.0, 4) if confs else None
        return "\n".join(text_lines), mean_conf

    def ocr(self, pages: Sequence[PageImage]) -> list[OCRPage]:
        results: list[OCRPage] = []
        for page in pages:
            text, conf = self._tsv_to_text_and_conf(self._run(page.png, "tsv"))
            results.append(
                OCRPage(
                    page_number=page.page_number,
                    text=text,
                    text_format="plain",
                    engine=self.name,
                    engine_version=self.version,
                    confidence=conf,
                )
            )
        return results

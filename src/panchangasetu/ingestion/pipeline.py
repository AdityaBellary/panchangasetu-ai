"""Ingestion pipeline: discover -> reconcile -> extract (text layer | OCR) -> classify -> chunk.

Guarantees:
- Idempotent and incremental: unchanged files (same path + SHA-256) are skipped; modified files
  get a new version; files that vanished are tombstoned; byte-identical copies under another
  name are recorded as duplicates and not reprocessed.
- OCR results are cached per page, keyed by extraction_version + engine, so re-runs and
  chunker changes never pay for OCR twice. Bump `extraction_version` to force re-OCR.
- Resumable: each OCR batch is committed as it completes; a crashed run resumes where it stopped.
- Bounded memory: pages are rendered lazily; at most one OCR batch of images is held at a time.
- Query-time code never touches PDFs; everything downstream reads `pages` / `chunks`.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from typing import Any

from sqlalchemy import delete, select, text
from sqlalchemy.orm import Session, sessionmaker

from panchangasetu.config import OCREngineName, Settings
from panchangasetu.core.interfaces import OCREngine, OCRError
from panchangasetu.core.types import Collection, ExtractionMethod, OCRPage, PageImage, PageKind
from panchangasetu.ingestion.chunking import PageText, TeluguChunker, approx_token_counter
from panchangasetu.ingestion.classification import CollectionRules, PagePlan, classify_page
from panchangasetu.ingestion.discovery import DiscoveredFile, discover_pdfs
from panchangasetu.ingestion.pdf_pages import PdfReader, inspect_page, render_upright
from panchangasetu.observability.logging import get_logger
from panchangasetu.storage.db import session_scope
from panchangasetu.storage.models import (
    ChunkRecord,
    Document,
    DocumentStatus,
    DocumentVersion,
    IngestionRun,
    Page,
    VersionStatus,
    utcnow,
)
from panchangasetu.storage.object_store import (
    LocalObjectStore,
    ocr_payload_key,
    page_image_key,
    raw_pdf_key,
)
from panchangasetu.telugu.text import normalize_telugu, script_stats

log = get_logger(__name__)


class IngestionBusyError(RuntimeError):
    """Another ingestion run holds the lock."""


@dataclass(slots=True)
class RunStats:
    discovered: int = 0
    new_versions: int = 0
    unchanged: int = 0
    duplicates: int = 0
    deleted: int = 0
    versions_processed: int = 0
    pages_text_layer: int = 0
    pages_ocr: int = 0
    pages_reused: int = 0
    pages_skipped: int = 0
    pages_failed: int = 0
    chunks_written: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "errors"}


@dataclass(slots=True)
class _Pending:
    image: PageImage
    page_id: uuid.UUID


EngineFactory = Callable[[OCREngineName], OCREngine]


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        session_factory: sessionmaker[Session],
        object_store: LocalObjectStore,
        engine_factory: EngineFactory,
        rules: CollectionRules | None = None,
    ) -> None:
        self.settings = settings
        self.sessions = session_factory
        self.store = object_store
        self._engine_factory = engine_factory
        self._engines: dict[OCREngineName, OCREngine] = {}
        self.rules = rules if rules is not None else CollectionRules.load(settings.collections_rules_path)
        self.chunker = TeluguChunker(
            target_tokens=settings.chunk_target_tokens,
            max_tokens=settings.chunk_max_tokens,
            overlap_units=settings.chunk_overlap_sentences,
            min_tokens=settings.chunk_min_tokens,
            count_tokens=approx_token_counter(settings.approx_chars_per_token),
        )

    # ------------------------------------------------------------------ public API
    def run(
        self,
        *,
        trigger: str = "cli",
        reprocess: bool = False,
        rechunk: bool = False,
        only: list[str] | None = None,
    ) -> tuple[uuid.UUID, RunStats]:
        stats = RunStats()
        with self._exclusive():
            with session_scope(self.sessions) as s:
                run = IngestionRun(
                    trigger=trigger, options={"reprocess": reprocess, "rechunk": rechunk, "only": only}
                )
                s.add(run)
                s.flush()
                run_id = run.id
            log.info("ingest.run.start", run_id=str(run_id), input_dir=str(self.settings.input_dir))
            status = "completed"
            try:
                files = discover_pdfs(self.settings.input_dir)
                if only:
                    files = [f for f in files if f.source_path in only or f.path.name in only]
                stats.discovered = len(files)
                queue = self._reconcile(
                    files, stats, reprocess=reprocess, rechunk=rechunk, full_scan=not only
                )
                for version_id in queue:
                    try:
                        self.process_version(version_id, stats, reprocess=reprocess)
                    except Exception as exc:  # one bad document must not stop the run
                        log.exception("ingest.version.failed", version_id=str(version_id))
                        stats.errors.append({"version_id": str(version_id), "error": repr(exc)[:500]})
                        self._set_version_status(version_id, VersionStatus.FAILED, error=repr(exc)[:2000])
                if stats.errors or stats.pages_failed:
                    status = "completed_with_errors"
            except Exception as exc:
                status = "failed"
                stats.errors.append({"error": repr(exc)[:500]})
                raise
            finally:
                with session_scope(self.sessions) as s:
                    finished = s.get(IngestionRun, run_id)
                    assert finished is not None
                    finished.status = status
                    finished.stats = stats.as_dict()
                    finished.errors = stats.errors
                    finished.finished_at = utcnow()
                log.info("ingest.run.done", run_id=str(run_id), status=status, **stats.as_dict())
        return run_id, stats

    # ------------------------------------------------------------------ concurrency
    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        """Postgres advisory lock so two workers never ingest concurrently."""
        session = self.sessions()
        is_pg = session.get_bind().dialect.name == "postgresql"
        try:
            if is_pg:
                got = session.execute(
                    text("SELECT pg_try_advisory_lock(:k)"), {"k": self.settings.ingestion_lock_key}
                ).scalar()
                if not got:
                    raise IngestionBusyError("another ingestion run is in progress")
            yield
        finally:
            if is_pg:
                session.execute(
                    text("SELECT pg_advisory_unlock(:k)"), {"k": self.settings.ingestion_lock_key}
                )
            session.close()

    # ------------------------------------------------------------------ reconcile
    def _reconcile(
        self, files: list[DiscoveredFile], stats: RunStats, *, reprocess: bool, rechunk: bool, full_scan: bool
    ) -> list[uuid.UUID]:
        queue: list[uuid.UUID] = []
        now = utcnow()
        with session_scope(self.sessions) as s:
            seen: set[str] = set()
            for f in files:
                seen.add(f.source_path)
                doc = s.scalar(select(Document).where(Document.source_path == f.source_path))
                if doc is None:
                    doc = Document(source_path=f.source_path, filename=f.path.name)
                    s.add(doc)
                    s.flush()
                doc.last_seen_at = now
                if doc.status == DocumentStatus.DELETED:
                    doc.status, doc.deleted_at = DocumentStatus.ACTIVE, None
                current = s.scalar(
                    select(DocumentVersion).where(
                        DocumentVersion.document_id == doc.id, DocumentVersion.is_current.is_(True)
                    )
                )
                if current is not None and current.sha256 == f.sha256:
                    if current.status == VersionStatus.DUPLICATE:
                        stats.duplicates += 1
                    elif reprocess or rechunk or self._needs_work(current):
                        queue.append(current.id)
                    else:
                        stats.unchanged += 1
                    continue

                if current is not None:
                    current.is_current = False
                original = s.scalar(
                    select(DocumentVersion).where(
                        DocumentVersion.sha256 == f.sha256,
                        DocumentVersion.is_current.is_(True),
                        DocumentVersion.document_id != doc.id,
                        DocumentVersion.status != VersionStatus.DUPLICATE,
                    )
                )
                key = raw_pdf_key(f.sha256)
                if not self.store.exists(key):
                    self.store.put_file(key, f.path)
                with PdfReader(self.store.local_path(key)) as reader:
                    page_count, meta = reader.page_count, reader.metadata()
                version = DocumentVersion(
                    document_id=doc.id,
                    sha256=f.sha256,
                    size_bytes=f.size_bytes,
                    page_count=page_count,
                    object_key=key,
                    pdf_metadata=meta,
                    status=VersionStatus.DUPLICATE if original else VersionStatus.PENDING,
                    duplicate_of_version_id=original.id if original else None,
                )
                s.add(version)
                s.flush()
                if original:
                    stats.duplicates += 1
                    log.info("ingest.duplicate", source_path=f.source_path, duplicate_of=str(original.id))
                else:
                    stats.new_versions += 1
                    queue.append(version.id)
                    log.info(
                        "ingest.version.new", source_path=f.source_path, sha256=f.sha256, pages=page_count
                    )

            if full_scan:
                gone = s.scalars(
                    select(Document).where(
                        Document.status == DocumentStatus.ACTIVE, Document.source_path.not_in(seen or {""})
                    )
                ).all()
                for doc in gone:
                    doc.status, doc.deleted_at = DocumentStatus.DELETED, now
                    for v in doc.versions:
                        v.is_current = False
                    stats.deleted += 1
                    log.info("ingest.document.deleted", source_path=doc.source_path)
        return queue

    def _needs_work(self, v: DocumentVersion) -> bool:
        return (
            v.status
            in (VersionStatus.PENDING, VersionStatus.PROCESSING, VersionStatus.PARTIAL, VersionStatus.FAILED)
            or v.extraction_version != self.settings.extraction_version
            or v.chunker_version != self.settings.chunker_version
        )

    def _set_version_status(self, version_id: uuid.UUID, status: str, *, error: str | None = None) -> None:
        with session_scope(self.sessions) as s:
            v = s.get(DocumentVersion, version_id)
            if v is not None:
                v.status, v.error = status, error

    # ------------------------------------------------------------------ extraction
    def _engine(self, name: OCREngineName) -> OCREngine:
        if name not in self._engines:
            self._engines[name] = self._engine_factory(name)
        return self._engines[name]

    def _page_is_fresh(self, page: Page, planned_engine: str | None) -> bool:
        if page.extraction_version != self.settings.extraction_version or page.error or page.text is None:
            return False
        # If the rules now route this page to a different engine (or the page was last read by
        # the fallback after the primary failed), re-OCR it with the planned engine.
        return not (
            page.extraction_method == ExtractionMethod.OCR
            and planned_engine
            and page.ocr_engine != planned_engine
        )

    def process_version(self, version_id: uuid.UUID, stats: RunStats, *, reprocess: bool = False) -> None:
        with session_scope(self.sessions) as s:
            v = s.get(DocumentVersion, version_id)
            assert v is not None
            v.status, v.error = VersionStatus.PROCESSING, None
            sha, key, source_path = v.sha256, v.object_key, v.document.source_path
            existing = {
                p.page_number: p for p in s.scalars(select(Page).where(Page.version_id == version_id))
            }
            existing_ids = {n: p.id for n, p in existing.items()}
            fresh = {
                n
                for n, p in existing.items()
                if not reprocess
                and self._page_is_fresh(
                    p, self.rules.plan(source_path, n).ocr_engine or self.settings.ocr_engine
                )
            }
        log.info(
            "ingest.version.process", version_id=str(version_id), source_path=source_path, cached=len(fresh)
        )

        batches: dict[OCREngineName, list[_Pending]] = defaultdict(list)
        with PdfReader(self.store.local_path(key)) as reader:
            for pdf_page in reader.pages():
                number = pdf_page.number + 1
                if number in fresh:
                    stats.pages_reused += 1
                    continue
                plan = self.rules.plan(source_path, number)
                info = inspect_page(
                    pdf_page,
                    min_chars=self.settings.text_layer_min_chars,
                    min_valid_ratio=self.settings.text_layer_min_valid_ratio,
                )
                with session_scope(self.sessions) as s:
                    page = s.get(Page, existing_ids[number]) if number in existing_ids else None
                    if page is None:
                        page = Page(version_id=version_id, page_number=number, page_kind=info.kind)
                        s.add(page)
                    page.page_kind = info.kind
                    page.rotation_applied = info.upright_rotation
                    page.text_stats = {"text_layer": _stats_dict(info.stats), "images": info.image_count}
                    page.error = None
                    if plan.collection == Collection.SKIP or info.kind == PageKind.EMPTY:
                        self._store_text(page, "", ExtractionMethod.TEXT_LAYER, engine=None)
                        stats.pages_skipped += 1
                        continue
                    if info.kind == PageKind.BORN_DIGITAL_UNICODE:
                        self._store_text(page, info.text_layer, ExtractionMethod.TEXT_LAYER, engine=None)
                        stats.pages_text_layer += 1
                        continue
                    image = render_upright(
                        pdf_page,
                        info,
                        dpi=self.settings.render_dpi,
                        min_short_side_px=self.settings.min_image_short_side_px,
                    )
                    img_key = page_image_key(sha, number, image.rotation_applied)
                    self.store.put(img_key, image.png)
                    page.image_key, page.image_width, page.image_height = img_key, image.width, image.height
                    s.flush()
                    page_id = page.id
                engine_name: OCREngineName = plan.ocr_engine or self.settings.ocr_engine
                batches[engine_name].append(_Pending(image=image, page_id=page_id))
                if len(batches[engine_name]) >= self.settings.ocr_batch_pages:
                    self._flush_batch(engine_name, batches.pop(engine_name), sha, stats)
        for engine_name, pending in list(batches.items()):
            self._flush_batch(engine_name, pending, sha, stats)

        self._classify_and_chunk(version_id, source_path, stats)

    def _flush_batch(
        self, engine_name: OCREngineName, pending: list[_Pending], sha: str, stats: RunStats
    ) -> None:
        results: dict[int, OCRPage] = {}
        error: str | None = None
        try:
            for r in self._engine(engine_name).ocr([p.image for p in pending]):
                results[r.page_number] = r
        except OCRError as exc:
            error = str(exc)
            log.warning("ingest.ocr.batch_failed", engine=engine_name, error=error, retryable=exc.retryable)
        missing = [p for p in pending if p.image.page_number not in results]
        fallback = self.settings.ocr_fallback_engine
        if missing and fallback and fallback != engine_name:
            log.info("ingest.ocr.fallback", engine=fallback, pages=[p.image.page_number for p in missing])
            try:
                for r in self._engine(fallback).ocr([p.image for p in missing]):
                    results[r.page_number] = r
            except OCRError as exc:
                error = f"{error}; fallback: {exc}" if error else f"fallback: {exc}"

        with session_scope(self.sessions) as s:
            for p in pending:
                page = s.get(Page, p.page_id)
                assert page is not None
                result = results.get(p.image.page_number)
                if result is None:
                    page.error = error or "page missing from OCR output"
                    page.text = None
                    stats.pages_failed += 1
                    stats.errors.append(
                        {"page_id": str(p.page_id), "page": p.image.page_number, "error": page.error}
                    )
                    continue
                if result.payload is not None:
                    pkey = ocr_payload_key(sha, p.image.page_number, result.engine)
                    self.store.put(pkey, json.dumps(result.payload, ensure_ascii=False).encode("utf-8"))
                    page.ocr_payload_key = pkey
                self._store_text(page, result.text, ExtractionMethod.OCR, engine=result)
                stats.pages_ocr += 1

    def _store_text(self, page: Page, raw: str, method: ExtractionMethod, *, engine: OCRPage | None) -> None:
        page.raw_text = raw
        page.text = normalize_telugu(raw)
        page.extraction_method = method
        page.extraction_version = self.settings.extraction_version
        page.ocr_engine = engine.engine if engine else None
        page.ocr_engine_version = engine.engine_version if engine else None
        page.ocr_confidence = engine.confidence if engine else None
        page.text_format = engine.text_format if engine else "plain"
        page.error = None
        st = script_stats(page.text)
        page.text_stats = {**(page.text_stats or {}), "normalized": _stats_dict(st), "chars": len(page.text)}

    # ------------------------------------------------------------------ classify + chunk
    def _classify_and_chunk(self, version_id: uuid.UUID, source_path: str, stats: RunStats) -> None:
        with session_scope(self.sessions) as s:
            v = s.get(DocumentVersion, version_id)
            assert v is not None
            pages = list(
                s.scalars(select(Page).where(Page.version_id == version_id).order_by(Page.page_number))
            )
            failed = [p.page_number for p in pages if p.error or p.text is None]
            page_texts: list[PageText] = []
            for p in pages:
                plan: PagePlan = self.rules.plan(source_path, p.page_number)
                c = classify_page(p.text or "", plan)
                p.collection, p.section_type = c.collection, c.section_type
                p.classification_confidence, p.classification_reason = c.confidence, c.reason[:500]
                if p.text and not p.error:
                    page_texts.append(PageText(p.page_number, p.text, c.collection, c.section_type))

            chunks = self.chunker.chunk(page_texts)
            s.execute(
                delete(ChunkRecord).where(
                    ChunkRecord.version_id == version_id,
                    ChunkRecord.chunker_version == self.settings.chunker_version,
                )
            )
            for ch in chunks:
                s.add(
                    ChunkRecord(
                        version_id=version_id,
                        chunker_version=self.settings.chunker_version,
                        chunk_index=ch.chunk_index,
                        collection=ch.collection,
                        section_type=ch.section_type,
                        heading_path=ch.heading_path,
                        page_start=ch.page_start,
                        page_end=ch.page_end,
                        text=ch.text,
                        token_count=ch.token_count,
                        content_hash=_sha256(ch.text),
                        chunk_metadata=ch.metadata,
                    )
                )
            stats.chunks_written += len(chunks)
            stats.versions_processed += 1
            v.extraction_version = self.settings.extraction_version
            v.chunker_version = self.settings.chunker_version
            v.status = VersionStatus.PARTIAL if failed else VersionStatus.COMPLETED
            v.error = f"pages failed: {failed}" if failed else None
            v.processed_at = utcnow()
            log.info(
                "ingest.version.done", version_id=str(version_id), chunks=len(chunks), failed_pages=failed
            )


def _stats_dict(st: Any) -> dict[str, Any]:
    return {
        "telugu": st.telugu,
        "latin": st.latin,
        "digits": st.digits,
        "pua": st.pua,
        "valid_ratio": round(st.valid_ratio, 4),
    }


def _sha256(text_: str) -> str:
    return hashlib.sha256(text_.encode("utf-8")).hexdigest()

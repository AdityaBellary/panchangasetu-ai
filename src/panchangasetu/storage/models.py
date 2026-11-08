"""Relational schema: the source of truth. Qdrant (Phase 3) is a derived, rebuildable index.

documents          one row per logical source file (by relative path)
document_versions  one row per distinct content (sha256) seen for a document
pages              per-version page extraction results (OCR text is cached here)
chunks             per-version chunks, tagged with chunker_version
ingestion_runs     audit trail + stats for each pipeline execution
chat_sessions      conversations (Phase 6)
chat_turns         questions, their standalone rewrites and answers
api_keys           API clients (hashed keys, Phase 7)
llm_spend          append-only LLM spend ledger (daily budget, Phase 8)
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from datetime import date as date_type
from typing import Any, ClassVar

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

JSONType = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    type_annotation_map: ClassVar[dict[Any, Any]] = {dict[str, Any]: JSONType, list[Any]: JSONType}


class DocumentStatus:
    ACTIVE = "active"
    DELETED = "deleted"  # source file disappeared; kept as tombstone for audit


class VersionStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL = "partial"  # some pages failed extraction; retried on next run
    FAILED = "failed"
    DUPLICATE = "duplicate"  # identical bytes already ingested under another document


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    source_path: Mapped[str] = mapped_column(String(1024), unique=True)  # relative to input dir
    filename: Mapped[str] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(32), default=DocumentStatus.ACTIVE)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    versions: Mapped[list[DocumentVersion]] = relationship(
        back_populates="document", order_by="DocumentVersion.created_at"
    )


class DocumentVersion(Base):
    __tablename__ = "document_versions"
    __table_args__ = (
        Index("ix_versions_sha256", "sha256"),
        Index("ix_versions_current", "document_id", "is_current"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    sha256: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    page_count: Mapped[int] = mapped_column(Integer)
    object_key: Mapped[str] = mapped_column(String(1024))
    pdf_metadata: Mapped[dict[str, Any]] = mapped_column(default=dict)
    status: Mapped[str] = mapped_column(String(32), default=VersionStatus.PENDING)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True)
    duplicate_of_version_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("document_versions.id", ondelete="SET NULL")
    )
    extraction_version: Mapped[str | None] = mapped_column(String(64))
    chunker_version: Mapped[str | None] = mapped_column(String(64))
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    document: Mapped[Document] = relationship(back_populates="versions")
    pages: Mapped[list[Page]] = relationship(
        back_populates="version", order_by="Page.page_number", cascade="all, delete-orphan"
    )


class Page(Base):
    __tablename__ = "pages"
    __table_args__ = (UniqueConstraint("version_id", "page_number"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("document_versions.id", ondelete="CASCADE"))
    page_number: Mapped[int] = mapped_column(Integer)  # 1-based
    page_kind: Mapped[str] = mapped_column(String(48))
    rotation_applied: Mapped[int] = mapped_column(Integer, default=0)
    image_key: Mapped[str | None] = mapped_column(String(1024))
    image_width: Mapped[int | None] = mapped_column(Integer)
    image_height: Mapped[int | None] = mapped_column(Integer)

    extraction_method: Mapped[str | None] = mapped_column(String(32))
    extraction_version: Mapped[str | None] = mapped_column(String(64))
    ocr_engine: Mapped[str | None] = mapped_column(String(64))
    ocr_engine_version: Mapped[str | None] = mapped_column(String(128))
    ocr_confidence: Mapped[float | None] = mapped_column(Float)
    text_format: Mapped[str | None] = mapped_column(String(16))
    ocr_payload_key: Mapped[str | None] = mapped_column(String(1024))  # raw engine output (blocks, bboxes)
    raw_text: Mapped[str | None] = mapped_column(Text)  # exactly as extracted
    text: Mapped[str | None] = mapped_column(Text)  # normalised (NFC, digits, whitespace)
    text_stats: Mapped[dict[str, Any]] = mapped_column(default=dict)

    collection: Mapped[str | None] = mapped_column(String(32))
    section_type: Mapped[str | None] = mapped_column(String(64))
    classification_confidence: Mapped[float | None] = mapped_column(Float)
    classification_reason: Mapped[str | None] = mapped_column(String(512))

    error: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    version: Mapped[DocumentVersion] = relationship(back_populates="pages")


class ChunkRecord(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("version_id", "chunker_version", "chunk_index"),
        Index("ix_chunks_collection", "collection"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("document_versions.id", ondelete="CASCADE"))
    chunker_version: Mapped[str] = mapped_column(String(64))
    chunk_index: Mapped[int] = mapped_column(Integer)
    collection: Mapped[str] = mapped_column(String(32))
    section_type: Mapped[str] = mapped_column(String(64))
    heading_path: Mapped[list[Any]] = mapped_column(default=list)
    page_start: Mapped[int] = mapped_column(Integer)
    page_end: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(64))
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    trigger: Mapped[str] = mapped_column(String(32))  # cli | api | schedule
    status: Mapped[str] = mapped_column(String(32), default="running")
    options: Mapped[dict[str, Any]] = mapped_column(default=dict)
    stats: Mapped[dict[str, Any]] = mapped_column(default=dict)
    errors: Mapped[list[Any]] = mapped_column(default=list)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class PanchangaDay(Base):
    """One calendar day as printed in a source, with every value's verification status.

    Query-time calendar answers come from here (exact date lookups), never from vector search.
    """

    __tablename__ = "panchanga_days"
    __table_args__ = (
        UniqueConstraint("version_id", "source_kind", "day"),
        Index("ix_panchanga_days_day", "day"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    day: Mapped[date_type] = mapped_column(Date)
    version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("document_versions.id", ondelete="CASCADE"))
    source_kind: Mapped[str] = mapped_column(String(32))  # book_daily_table | wall_calendar
    page_number: Mapped[int] = mapped_column(Integer)
    row_index: Mapped[int] = mapped_column(Integer)
    bbox: Mapped[list[Any]] = mapped_column(default=list)  # unrotated page points
    weekday_printed: Mapped[str | None] = mapped_column(String(16))
    weekday_check: Mapped[str] = mapped_column(String(16))  # ok | mismatch | missing
    date_check: Mapped[str] = mapped_column(String(16))  # ok | mismatch | unreadable
    sunrise: Mapped[str | None] = mapped_column(String(8))
    sunset: Mapped[str | None] = mapped_column(String(8))
    tamil_day: Mapped[str | None] = mapped_column(String(256))
    raw_text: Mapped[str] = mapped_column(Text)
    elements: Mapped[list[Any]] = mapped_column(default=list)
    windows: Mapped[list[Any]] = mapped_column(default=list)
    notes: Mapped[str] = mapped_column(Text, default="")
    warnings: Mapped[list[Any]] = mapped_column(default=list)
    element_count: Mapped[int] = mapped_column(Integer, default=0)
    verified_count: Mapped[int] = mapped_column(Integer, default=0)
    ocr_variant: Mapped[str | None] = mapped_column(String(32))
    extractor_version: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class PanchangaEvent(Base):
    """A dated festival / utsavam / observance as printed in a source."""

    __tablename__ = "panchanga_events"
    __table_args__ = (
        Index("ix_panchanga_events_day", "day"),
        Index("ix_panchanga_events_version", "version_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    day: Mapped[date_type] = mapped_column(Date)
    version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("document_versions.id", ondelete="CASCADE"))
    source_kind: Mapped[str] = mapped_column(String(32))  # festival_table | utsavam_schedule
    page_number: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(Text, default="")
    details: Mapped[str] = mapped_column(Text)
    tamil_date: Mapped[str | None] = mapped_column(String(64))
    weekday_printed: Mapped[str | None] = mapped_column(String(16))
    weekday_check: Mapped[str] = mapped_column(String(16))
    raw: Mapped[str] = mapped_column(Text)
    extractor_version: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class IndexBuild(Base):
    """One build of a vector-index collection. Postgres stays the source of truth; Qdrant
    collections are derived and can be rebuilt from `chunks` / `panchanga_*` at any time."""

    __tablename__ = "index_builds"
    __table_args__ = (Index("ix_index_builds_alias", "alias", "status"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    alias: Mapped[str] = mapped_column(String(128))  # logical name served to queries
    physical_name: Mapped[str] = mapped_column(String(256), unique=True)
    embed_model: Mapped[str] = mapped_column(String(128))
    sparse_model: Mapped[str] = mapped_column(String(64))
    chunker_version: Mapped[str] = mapped_column(String(64))
    fingerprint: Mapped[str] = mapped_column(String(64))  # hash of docs + models + render settings
    doc_count: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16))  # building | active | retired | failed
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ChatSession(Base):
    """A conversation. `owner` is the API client (Phase 7); the CLI leaves it empty."""

    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    owner: Mapped[str | None] = mapped_column(String(128), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ChatTurn(Base):
    """One question and its answer. `standalone_question` is what was actually answered after
    follow-up resolution; `asked_at` is the clock the question was answered against ("today")."""

    __tablename__ = "chat_turns"
    __table_args__ = (UniqueConstraint("session_id", "seq", name="uq_chat_turn_seq"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True
    )
    seq: Mapped[int] = mapped_column(Integer)
    question: Mapped[str] = mapped_column(Text)
    standalone_question: Mapped[str] = mapped_column(Text)
    rewrite_method: Mapped[str] = mapped_column(String(16))  # none | rule | llm
    language: Mapped[str] = mapped_column(String(8))
    answer: Mapped[str] = mapped_column(Text)
    abstained: Mapped[bool] = mapped_column(Boolean)
    method: Mapped[str] = mapped_column(String(16))
    route: Mapped[str] = mapped_column(String(32))
    citations: Mapped[list[Any]] = mapped_column(default=list)
    model: Mapped[str | None] = mapped_column(String(128))
    cost_inr: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    asked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ApiKey(Base):
    """An API client. Only a SHA-256 of the key is stored; the key itself is shown once at creation."""

    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(128))
    prefix: Mapped[str] = mapped_column(String(16), index=True)  # first characters, to identify a key
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    rate_limit_per_minute: Mapped[int | None] = mapped_column(Integer)  # None: the server default
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class LlmSpend(Base):
    """Append-only LLM spend ledger for the daily budget. Kept apart from chat_turns so that deleting
    or pruning conversations can never lower the day's recorded spend."""

    __tablename__ = "llm_spend"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    spent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    cost_inr: Mapped[float] = mapped_column(Float)
    owner: Mapped[str | None] = mapped_column(String(128))

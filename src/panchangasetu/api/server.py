"""Production wiring for the API: real stores, page images from the object store, readiness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from sqlalchemy import select, text
from sqlalchemy.orm import Session, sessionmaker

from panchangasetu.api.app import ApiServices, create_app
from panchangasetu.api.keys import ApiKeyStore
from panchangasetu.api.today import today_facts
from panchangasetu.config import Settings, get_settings
from panchangasetu.observability.ops import OpsQueries
from panchangasetu.storage.db import make_engine, make_session_factory, session_scope
from panchangasetu.storage.models import Document, DocumentVersion, Page
from panchangasetu.storage.object_store import LocalObjectStore


class StoredPageImages:
    """Page images rendered at ingestion time; nothing is rendered from a PDF per request."""

    def __init__(self, factory: sessionmaker[Session], objects: LocalObjectStore) -> None:
        self.factory = factory
        self.objects = objects

    def path(self, document: str, page: int) -> Path | None:
        with session_scope(self.factory) as s:
            key = s.scalar(
                select(Page.image_key)
                .join(DocumentVersion, DocumentVersion.id == Page.version_id)
                .join(Document, Document.id == DocumentVersion.document_id)
                .where(
                    Document.filename == document,
                    DocumentVersion.is_current.is_(True),
                    Page.page_number == page,
                )
                .limit(1)
            )
        return self.objects.local_path(key) if key and self.objects.exists(key) else None


def build_services(settings: Settings) -> ApiServices:
    from panchangasetu.cli import build_conversation_service

    factory = make_session_factory(make_engine(settings.database_url))
    conversations = build_conversation_service(settings)

    def readiness() -> dict[str, bool]:
        checks: dict[str, bool] = {}
        try:
            with session_scope(factory) as s:
                s.execute(text("select 1"))
            checks["database"] = True
        except Exception:
            checks["database"] = False
        try:
            conversations.answers.retriever.vectors.client.get_collections()
            checks["vector_index"] = True
        except Exception:
            checks["vector_index"] = False
        checks["llm"] = conversations.answers.llm is not None
        return checks

    retriever = conversations.answers.retriever

    def today() -> dict[str, Any]:
        return today_facts(retriever.structured, retriever.weekday_periods, retriever.clock())

    return ApiServices(
        conversations=conversations,
        keys=ApiKeyStore(factory),
        pages=StoredPageImages(factory, LocalObjectStore(settings.object_store_dir)),
        readiness=readiness,
        today=today,
        ops=OpsQueries(factory, settings.timezone, settings.llm_daily_budget_inr),
        rate_limit_per_minute=settings.api_rate_limit_per_minute,
        max_question_chars=settings.api_max_question_chars,
        metrics_token=settings.api_metrics_token.get_secret_value() if settings.api_metrics_token else None,
    )


def build_app() -> FastAPI:
    """`uvicorn panchangasetu.api.server:build_app --factory`"""
    return create_app(build_services(get_settings()))

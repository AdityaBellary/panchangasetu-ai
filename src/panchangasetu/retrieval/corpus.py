"""Assemble index documents from Postgres (the source of truth).

general   : chunks classified General
panchanga : Panchanga chunks except structured-only sections, plus
            - one document per dated festival/utsavam event (book tables), and
            - one document per book day with observance notes,
            so semantic questions ("when is Garuda Seva?") land on items that carry an exact date.
Wall-calendar data is unverified and is not indexed.
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import select
from sqlalchemy.orm import Session

from panchangasetu.retrieval.documents import STRUCTURED_ONLY_SECTIONS, IndexDocument, render_for_embedding
from panchangasetu.storage.models import ChunkRecord, Document, DocumentVersion, PanchangaDay, PanchangaEvent

WEEKDAY_TE = ("సోమవారం", "మంగళవారం", "బుధవారం", "గురువారం", "శుక్రవారం", "శనివారం", "ఆదివారం")
WEEKDAY_EN = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def iter_documents(s: Session, *, chunker_version: str, gloss: bool) -> Iterator[IndexDocument]:
    current = (
        select(DocumentVersion.id, Document.filename)
        .join(Document, Document.id == DocumentVersion.document_id)
        .where(DocumentVersion.is_current.is_(True), Document.status == "active")
        .subquery()
    )
    chunks = s.execute(
        select(ChunkRecord, current.c.filename)
        .join(current, current.c.id == ChunkRecord.version_id)
        .where(
            ChunkRecord.chunker_version == chunker_version,
            ChunkRecord.section_type.not_in(STRUCTURED_ONLY_SECTIONS),
        )
        .order_by(ChunkRecord.version_id, ChunkRecord.chunk_index)
    ).all()
    for c, filename in chunks:
        yield IndexDocument(
            id=str(c.id),
            collection=c.collection,
            text=render_for_embedding(c.section_type, c.heading_path, c.text, gloss=gloss),
            display_text=c.text,
            payload={
                "kind": "chunk",
                "chunk_id": str(c.id),
                "version_id": str(c.version_id),
                "document": filename,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "pages": list(range(c.page_start, c.page_end + 1)),
                "section_type": c.section_type,
                "heading_path": c.heading_path,
                "content_hash": c.content_hash,
            },
        )

    events = s.execute(
        select(PanchangaEvent, current.c.filename).join(current, current.c.id == PanchangaEvent.version_id)
    ).all()
    for e, filename in events:
        day = e.day
        label = "TTD utsavam schedule" if e.source_kind == "utsavam_schedule" else "TTD festival calendar"
        header = f"{day.isoformat()} {WEEKDAY_EN[day.weekday()]} {WEEKDAY_TE[day.weekday()]}"
        body = f"{e.title}: {e.details}" if e.title else e.details
        yield IndexDocument(
            id=str(e.id),
            collection="panchanga",
            text=f"{label} on {header}\n{body}" if gloss else f"{header}\n{body}",
            display_text=f"{header}: {body}",
            payload={
                "kind": "event",
                "event_id": str(e.id),
                "version_id": str(e.version_id),
                "document": filename,
                "date": day.isoformat(),
                "page_start": e.page_number,
                "page_end": e.page_number,
                "pages": [e.page_number],
                "section_type": e.source_kind,
                "title": e.title,
            },
        )

    days = s.execute(
        select(PanchangaDay, current.c.filename)
        .join(current, current.c.id == PanchangaDay.version_id)
        .where(PanchangaDay.source_kind == "book_daily_table", PanchangaDay.notes != "")
    ).all()
    for d, filename in days:
        header = f"{d.day.isoformat()} {WEEKDAY_EN[d.day.weekday()]} {WEEKDAY_TE[d.day.weekday()]}"
        yield IndexDocument(
            id=str(d.id),
            collection="panchanga",
            text=(
                f"Daily panchangam observances on {header}\n{d.notes}" if gloss else f"{header}\n{d.notes}"
            ),
            display_text=f"{header}: {d.notes}",
            payload={
                "kind": "day_note",
                "day_id": str(d.id),
                "version_id": str(d.version_id),
                "document": filename,
                "date": d.day.isoformat(),
                "page_start": d.page_number,
                "page_end": d.page_number,
                "pages": [d.page_number],
                "section_type": "daily_panchangam_notes",
            },
        )

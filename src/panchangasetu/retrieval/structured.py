"""Exact calendar lookups over panchanga_days / panchanga_events (no vector search, no LLM).

Source preference per date: the book's verified daily table, then the wall calendar (whose values
are marked unverified). Dates outside every source return nothing, so callers can abstain.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from panchangasetu.retrieval.evidence import Evidence
from panchangasetu.storage.models import Document, DocumentVersion, PanchangaDay, PanchangaEvent
from panchangasetu.telugu.lexicon import KARANAS, MASAS, NAKSHATRAS, TITHIS, YOGAS, english_name

SOURCE_PREFERENCE = ("book_daily_table", "wall_calendar")
WEEKDAY_EN = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
WEEKDAY_TE = ("సోమవారం", "మంగళవారం", "బుధవారం", "గురువారం", "శుక్రవారం", "శనివారం", "ఆదివారం")
PAKSHA_EN = {"శుద్ధ": "Shukla", "బహుళ": "Krishna (Bahula)"}
_TABLES = {"tithi": TITHIS, "nakshatra": NAKSHATRAS, "yoga": YOGAS, "karana": KARANAS}
_MAX_RANGE_DAYS = 31


@dataclass(frozen=True, slots=True)
class Coverage:
    verified: tuple[date, date] | None
    any_source: tuple[date, date] | None


def _hm(iso: str | None) -> str:
    if not iso:
        return "beyond the next sunrise"
    dt = datetime.fromisoformat(iso)
    return dt.strftime("%Y-%m-%d %H:%M")


def _name(kind: str, telugu: str) -> str:
    table = _TABLES.get(kind)
    en = english_name(table, telugu) if table else telugu
    return f"{telugu} ({en})" if en != telugu else telugu


def render_day(row: PanchangaDay, document: str, following: PanchangaDay | None = None) -> str:
    d = row.day
    head = f"{d.isoformat()} ({WEEKDAY_EN[d.weekday()]} / {WEEKDAY_TE[d.weekday()]})"
    src = "TTD book daily panchangam" if row.source_kind == "book_daily_table" else "TTD wall calendar (OCR)"
    lines = [f"[Panchanga for {head}] Source: {document} p.{row.page_number}, {src}."]
    if row.source_kind == "wall_calendar":
        lines.append("NOTE: wall-calendar values are unverified OCR; state this if you use them.")
    if row.sunrise or row.sunset:
        lines.append(f"Sunrise {row.sunrise or '?'}, sunset {row.sunset or '?'} (as printed, local time).")
    for kind in ("tithi", "nakshatra", "yoga", "karana"):
        items = [e for e in row.elements if e.get("kind") == kind]
        for e in items:
            extra = ""
            if kind == "tithi" and e.get("masa"):
                masa_en = english_name(MASAS, e["masa"])
                adhika = "Adhika " if e.get("adhika") else ""
                paksha = e.get("paksha", "")
                extra = f", {adhika}{e['masa']} {paksha} ({adhika}{masa_en} {PAKSHA_EN.get(paksha, '')})"
            status = "verified" if e.get("verified") else "UNVERIFIED"
            printed = e.get("printed_end") or ""
            lines.append(
                f"{kind.capitalize()}: {_name(kind, e['name'])}{extra}; ends {_hm(e.get('ends_at'))} "
                f"(printed {printed}) [{status}]"
            )
    if following is not None:
        for kind in ("tithi", "nakshatra"):
            nxt = next((e for e in following.elements if e.get("kind") == kind), None)
            if nxt:
                lines.append(
                    f"Next {kind} after that: {_name(kind, nxt['name'])}, ends {_hm(nxt.get('ends_at'))}."
                )
    labels = {
        "varjyam": "Varjyam",
        "durmuhurtam": "Durmuhurtam",
        "amrutham": "Amrutha kalam",
        "punaramrutham": "Amrutha kalam (second)",
    }
    for w in row.windows:
        lines.append(f"{labels.get(w['kind'], w['kind'])}: {_hm(w['starts_at'])} to {_hm(w['ends_at'])}.")
    if row.notes:
        lines.append(f"Observances / notes (as printed): {row.notes}")
    return "\n".join(lines)


def best_rows(rows: Iterable[tuple[PanchangaDay, str]]) -> dict[date, tuple[PanchangaDay, str]]:
    """One row per day: the preferred source, except that a book row which yielded no elements
    (festival rows can be unreadable) loses to a readable wall-calendar row. An answer marked
    unverified beats a day that silently disappears from "next Purnima" searches."""

    def rank(row: PanchangaDay) -> tuple[bool, int]:
        return (not row.elements, SOURCE_PREFERENCE.index(row.source_kind))

    best: dict[date, tuple[PanchangaDay, str]] = {}
    for row, filename in rows:
        current = best.get(row.day)
        if current is None or rank(row) < rank(current[0]):
            best[row.day] = (row, filename)
    return best


class StructuredRetriever:
    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self.sessions = sessions

    def _rows(self, s: Session, start: date, end: date) -> dict[date, tuple[PanchangaDay, str]]:
        rows = s.execute(
            select(PanchangaDay, Document.filename)
            .join(DocumentVersion, DocumentVersion.id == PanchangaDay.version_id)
            .join(Document, Document.id == DocumentVersion.document_id)
            .where(DocumentVersion.is_current.is_(True), PanchangaDay.day.between(start, end))
        ).all()
        return best_rows(rows)

    def coverage(self) -> Coverage:
        with self.sessions() as s:

            def span(kinds: tuple[str, ...]) -> tuple[date, date] | None:
                days = s.scalars(select(PanchangaDay.day).where(PanchangaDay.source_kind.in_(kinds))).all()
                return (min(days), max(days)) if days else None

            return Coverage(span(("book_daily_table",)), span(SOURCE_PREFERENCE))

    def days(self, start: date, end: date) -> list[Evidence]:
        end = min(end, start + timedelta(days=_MAX_RANGE_DAYS - 1))
        with self.sessions() as s:
            rows = self._rows(s, start, end + timedelta(days=1))
        out = []
        d = start
        while d <= end:
            if d in rows:
                row, doc = rows[d]
                following = rows.get(d + timedelta(days=1), (None, ""))[0]
                out.append(_day_evidence(row, doc, following))
            d += timedelta(days=1)
        return out

    def next_occurrences(self, kind: str, name: str, after: datetime, *, limit: int = 2) -> list[Evidence]:
        """Next days on which `name` (a tithi or nakshatra) is in force, from `after` onward."""
        with self.sessions() as s:
            rows = self._rows(s, after.date() - timedelta(days=1), after.date() + timedelta(days=400))
        out: list[Evidence] = []
        last_day: date | None = None
        for d in sorted(rows):
            row, doc = rows[d]
            for e in row.elements:
                if e.get("kind") != kind or e.get("name") != name:
                    continue
                ends = datetime.fromisoformat(e["ends_at"]) if e.get("ends_at") else None
                if ends is not None and ends < after.replace(tzinfo=None):
                    continue
                if last_day is not None and (d - last_day).days <= 1:
                    continue  # same occurrence spanning two sunrises
                following = rows.get(d + timedelta(days=1), (None, ""))[0]
                ev = _day_evidence(row, doc, following)
                ev.kind = "occurrence"
                ev.data["match"] = e
                ev.text = (
                    f"[Occurrence of {kind} {_name(kind, name)}: ends {_hm(e.get('ends_at'))}]\n" + ev.text
                )
                out.append(ev)
                last_day = d
                break
            if len(out) >= limit:
                break
        return out

    def events(self, start: date, end: date, *, limit: int = 60) -> list[Evidence]:
        with self.sessions() as s:
            rows = s.execute(
                select(PanchangaEvent, Document.filename)
                .join(DocumentVersion, DocumentVersion.id == PanchangaEvent.version_id)
                .join(Document, Document.id == DocumentVersion.document_id)
                .where(DocumentVersion.is_current.is_(True), PanchangaEvent.day.between(start, end))
                .order_by(PanchangaEvent.day)
                .limit(limit)
            ).all()
        return [_event_evidence(e, doc) for e, doc in rows]


def _day_evidence(row: PanchangaDay, document: str, following: PanchangaDay | None) -> Evidence:
    data: dict[str, Any] = {
        "elements": row.elements,
        "windows": row.windows,
        "notes": row.notes,
        "sunrise": row.sunrise,
        "sunset": row.sunset,
        "following": following.elements if following is not None else [],
    }
    return Evidence(
        id=str(row.id),
        kind="day",
        collection="panchanga",
        text=render_day(row, document, following),
        document=document,
        pages=[row.page_number] if row.source_kind != "wall_calendar" else [],
        date=row.day.isoformat(),
        verified=row.source_kind == "book_daily_table" and row.verified_count > 0,
        source_kind=row.source_kind,
        data=data,
    )


def _event_evidence(e: PanchangaEvent, document: str) -> Evidence:
    d = e.day
    body = f"{e.title}: {e.details}" if e.title else e.details
    label = "TTD utsavam schedule" if e.source_kind == "utsavam_schedule" else "TTD festival calendar"
    return Evidence(
        id=str(e.id),
        kind="event",
        collection="panchanga",
        text=f"[{label}, {d.isoformat()} {WEEKDAY_EN[d.weekday()]}] {body} "
        f"(Source: {document} p.{e.page_number})",
        document=document,
        pages=[e.page_number],
        date=d.isoformat(),
        verified=e.weekday_check in ("ok", "corrected"),
        source_kind=e.source_kind,
    )

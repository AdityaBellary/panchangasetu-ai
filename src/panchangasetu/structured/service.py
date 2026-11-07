"""Structured extraction stage: dated panchanga facts -> panchanga_days / panchanga_events.

Runs after page extraction. Uses only free, local components (PDF geometry + Tesseract) and
the already-cached page OCR, so re-running it never spends provider credits. Each source is
re-extracted only when its document version or the extractor version changes.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pymupdf
import yaml
from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from panchangasetu.core.interfaces import OCREngine
from panchangasetu.ingestion.classification import parse_page_spec
from panchangasetu.observability.logging import get_logger
from panchangasetu.storage.db import session_scope
from panchangasetu.storage.models import Document, DocumentVersion, Page, PanchangaDay, PanchangaEvent
from panchangasetu.storage.object_store import LocalObjectStore
from panchangasetu.structured.book_daily import DayRow, assign_dates, iter_daily_rows
from panchangasetu.structured.festivals import (
    EVENTS_PARSER_VERSION,
    parse_festival_table,
    parse_utsavam_schedules,
)
from panchangasetu.structured.panchanga_parse import PARSER_VERSION, ParsedDay, parse_day, resolve_sequences
from panchangasetu.structured.row_ocr import WEEKDAY_TOKENS, ocr_rows
from panchangasetu.structured.wall_calendar import (
    CAL_PARSER_VERSION,
    CalendarCell,
    TsvRunner,
    detect_grid,
    median_grid,
    read_month,
)

log = get_logger(__name__)

DAILY_EXTRACTOR_VERSION = f"{PARSER_VERSION}+rows-1"
_WEEKDAY_OF = ("సోమ", "మంగళ", "బుధ", "గురు", "శుక్ర", "శని", "ఆది")
assert set(_WEEKDAY_OF) == set(WEEKDAY_TOKENS)

# OCR retries for rows whose first reading parsed poorly, best first (measured on the 2026-27
# book: 24 of 34 weak rows improved, mostly by 400 dpi).
RETRY_VARIANTS: tuple[tuple[int, int], ...] = ((400, 6), (250, 6), (300, 4), (400, 4))

RowEngineFactory = Callable[[int], OCREngine]  # psm -> engine


@dataclass(slots=True)
class StructuredStats:
    days_written: int = 0
    events_written: int = 0
    sources_skipped: int = 0
    problems: list[str] = field(default_factory=list)


def load_sources(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(raw.get("sources", []))


def _day_score(day: ParsedDay) -> int:
    kinds = {e.kind for e in day.elements}
    return sum(e.verified for e in day.elements) + 3 * ("tithi" in kinds) + 3 * ("nakshatra" in kinds)


def _parse_row(row: DayRow, text: str, weekday: str | None) -> ParsedDay:
    assert row.date is not None
    return parse_day(
        text,
        row.date,
        sunrise=row.sunrise,
        sunset=row.sunset,
        weekday_printed=weekday,
        exact_numbers=row.numbers_in_row,
    )


def extract_daily_table(
    doc: pymupdf.Document, cfg: dict[str, Any], row_engine: RowEngineFactory, *, workers: int = 6
) -> tuple[list[DayRow], list[ParsedDay], list[str]]:
    pages = sorted(parse_page_spec(cfg["pages"]))
    rows = list(iter_daily_rows(doc, pages))
    first = (
        cfg["first_date"]
        if isinstance(cfg["first_date"], date)
        else date.fromisoformat(str(cfg["first_date"]))
    )
    assign_dates(rows, first)
    problems = [
        f"p{r.page_number} row {r.row_index}: printed day {r.printed_day} != {r.date}"
        for r in rows
        if r.date_check == "mismatch"
    ]

    texts = ocr_rows(doc, rows, row_engine(6), dpi=300, workers=workers)
    parsed = [_parse_row(r, t, w) for r, (t, w) in zip(rows, texts, strict=True)]
    variants: list[str | None] = [None] * len(rows)

    weak = [i for i, d in enumerate(parsed) if not {"tithi", "nakshatra"} <= {e.kind for e in d.elements}]
    for dpi, psm in RETRY_VARIANTS:
        if not weak:
            break
        retry = ocr_rows(doc, [rows[i] for i in weak], row_engine(psm), dpi=dpi, workers=workers)
        for i, (text, weekday) in zip(weak, retry, strict=True):
            candidate = _parse_row(rows[i], text, weekday or parsed[i].weekday_printed)
            if _day_score(candidate) > _day_score(parsed[i]):
                parsed[i], variants[i] = candidate, f"{dpi}dpi/psm{psm}"
        weak = [i for i in weak if not {"tithi", "nakshatra"} <= {e.kind for e in parsed[i].elements}]

    resolve_sequences(parsed, initial_masa=cfg["initial_masa"], initial_paksha=cfg["initial_paksha"])
    for d, v in zip(parsed, variants, strict=True):
        if v:
            d.warnings.append(f"ocr retry: {v}")
    return rows, parsed, problems


class StructuredExtractor:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        object_store: LocalObjectStore,
        sources_path: Path,
        row_engine: RowEngineFactory,
        tesseract_runner: TsvRunner | None = None,
    ) -> None:
        self.sessions = session_factory
        self.store = object_store
        self.sources = load_sources(sources_path)
        self.row_engine = row_engine
        self.tesseract_runner = tesseract_runner

    def run(self, *, force: bool = False) -> StructuredStats:
        stats = StructuredStats()
        with self.sessions() as s:
            targets = s.execute(
                select(Document.filename, DocumentVersion.id, DocumentVersion.object_key)
                .join(DocumentVersion, DocumentVersion.document_id == Document.id)
                .where(DocumentVersion.is_current.is_(True), Document.status == "active")
            ).all()
        for filename, version_id, key in targets:
            for src in self.sources:
                if not Path(filename).match(src["match"]):
                    continue
                if "daily_table" in src:
                    self._daily(version_id, key, src["daily_table"], stats, force=force)
                if "festival_table" in src or "utsavam_schedules" in src:
                    self._events(version_id, src, stats, force=force)
                if "wall_calendar" in src and self.tesseract_runner is not None:
                    self._wall_calendar(version_id, key, src["wall_calendar"], stats, force=force)
        log.info(
            "structured.done",
            days=stats.days_written,
            events=stats.events_written,
            skipped=stats.sources_skipped,
            problems=len(stats.problems),
        )
        return stats

    def _up_to_date(
        self,
        model: type[PanchangaDay] | type[PanchangaEvent],
        version_id: uuid.UUID,
        kind: str,
        extractor_version: str,
    ) -> bool:
        with self.sessions() as s:
            return (
                s.scalar(
                    select(model.id)
                    .where(
                        model.version_id == version_id,
                        model.source_kind == kind,
                        model.extractor_version == extractor_version,
                    )
                    .limit(1)
                )
                is not None
            )

    def _daily(
        self, version_id: uuid.UUID, key: str, cfg: dict[str, Any], stats: StructuredStats, *, force: bool
    ) -> None:
        kind = "book_daily_table"
        if not force and self._up_to_date(PanchangaDay, version_id, kind, DAILY_EXTRACTOR_VERSION):
            stats.sources_skipped += 1
            return
        with pymupdf.open(self.store.local_path(key)) as doc:
            rows, parsed, problems = extract_daily_table(doc, cfg, self.row_engine)
        stats.problems += problems
        with session_scope(self.sessions) as s:
            s.execute(
                delete(PanchangaDay).where(
                    PanchangaDay.version_id == version_id, PanchangaDay.source_kind == kind
                )
            )
            for row, day in zip(rows, parsed, strict=True):
                assert row.date is not None
                printed = day.weekday_printed
                s.add(
                    PanchangaDay(
                        day=row.date,
                        version_id=version_id,
                        source_kind=kind,
                        page_number=row.page_number,
                        row_index=row.row_index,
                        bbox=row.main_box.as_list(),
                        weekday_printed=printed,
                        weekday_check="missing"
                        if printed is None
                        else ("ok" if printed == _WEEKDAY_OF[row.date.weekday()] else "mismatch"),
                        date_check=row.date_check,
                        sunrise=row.sunrise,
                        sunset=row.sunset,
                        tamil_day=row.tamil_day,
                        raw_text=day.source_text,
                        elements=[_element_dict(e) for e in day.elements],
                        windows=[_window_dict(w) for w in day.windows],
                        notes=day.notes,
                        warnings=day.warnings,
                        element_count=len(day.elements),
                        verified_count=sum(e.verified for e in day.elements),
                        ocr_variant=next(
                            (w.split(": ", 1)[1] for w in day.warnings if w.startswith("ocr retry")), None
                        ),
                        extractor_version=DAILY_EXTRACTOR_VERSION,
                    )
                )
        stats.days_written += len(parsed)

    def _events(
        self, version_id: uuid.UUID, src: dict[str, Any], stats: StructuredStats, *, force: bool
    ) -> None:
        if not force and self._up_to_date(
            PanchangaEvent, version_id, "festival_table", EVENTS_PARSER_VERSION
        ):
            stats.sources_skipped += 1
            return
        with self.sessions() as s:
            pages = dict(
                s.execute(select(Page.page_number, Page.text).where(Page.version_id == version_id)).all()
            )
        events = []
        if fcfg := src.get("festival_table"):
            nums = sorted(parse_page_spec(fcfg["pages"]))
            start = fcfg["start_after"]
            start = start if isinstance(start, date) else date.fromisoformat(str(start))
            ev, problems = parse_festival_table([(n, pages.get(n) or "") for n in nums], start=start)
            events += [("festival_table", e) for e in ev]
            stats.problems += problems
        if ucfg := src.get("utsavam_schedules"):
            nums = sorted(parse_page_spec(ucfg["pages"]))
            ev, problems = parse_utsavam_schedules([(n, pages.get(n) or "") for n in nums])
            events += [("utsavam_schedule", e) for e in ev]
            stats.problems += problems
        with session_scope(self.sessions) as s:
            s.execute(
                delete(PanchangaEvent).where(
                    PanchangaEvent.version_id == version_id,
                    PanchangaEvent.source_kind.in_(["festival_table", "utsavam_schedule"]),
                )
            )
            for kind, e in events:
                s.add(
                    PanchangaEvent(
                        day=date.fromisoformat(e.date),
                        version_id=version_id,
                        source_kind=kind,
                        page_number=e.page_number,
                        title=e.title,
                        details=e.details,
                        tamil_date=e.tamil_date,
                        weekday_printed=e.weekday_printed,
                        weekday_check=e.weekday_check,
                        raw=e.raw,
                        extractor_version=EVENTS_PARSER_VERSION,
                    )
                )
        stats.events_written += len(events)

    def _wall_calendar(
        self, version_id: uuid.UUID, key: str, cfg: dict[str, Any], stats: StructuredStats, *, force: bool
    ) -> None:
        kind = "wall_calendar"
        if not force and self._up_to_date(PanchangaDay, version_id, kind, CAL_PARSER_VERSION):
            stats.sources_skipped += 1
            return
        assert self.tesseract_runner is not None
        run = self.tesseract_runner
        year, first_month = int(cfg["year"]), int(cfg.get("first_month", 1))
        months: list[tuple[int, int]] = []
        with pymupdf.open(self.store.local_path(key)) as doc:
            for index in range(doc.page_count):
                month = first_month + index
                if month > 12:
                    break
                months.append((index, month))
            grids = {m: detect_grid(doc, i, year, m, run) for i, m in months}
            good = [g for g, _ in grids.values() if g is not None]
            if not good:
                stats.problems.append(f"{key}: no calendar grid could be fitted on any page")
                return
            fallback = median_grid(good)
            cells: list[CalendarCell] = []
            for index, month in months:
                grid, _agree = grids[month]
                if grid is None:
                    stats.problems.append(
                        f"{year}-{month:02d}: grid fit failed, using the calendar's median grid"
                    )
                cells += read_month(doc, index, year, month, grid or fallback, run)
        with self.sessions() as s:
            book = {
                d: els
                for d, els in s.execute(
                    select(PanchangaDay.day, PanchangaDay.elements).where(
                        PanchangaDay.source_kind == "book_daily_table",
                        PanchangaDay.day.in_([c.date for c in cells]),
                    )
                ).all()
            }
        with session_scope(self.sessions) as s:
            s.execute(
                delete(PanchangaDay).where(
                    PanchangaDay.version_id == version_id, PanchangaDay.source_kind == kind
                )
            )
            for c in cells:
                elements = _calendar_elements(c, book.get(c.date))
                s.add(
                    PanchangaDay(
                        day=c.date,
                        version_id=version_id,
                        source_kind=kind,
                        page_number=c.date.month,
                        row_index=0,
                        bbox=[],
                        weekday_printed=None,
                        weekday_check="missing",
                        date_check="grid",
                        sunrise=None,
                        sunset=None,
                        tamil_day=None,
                        raw_text=c.text,
                        elements=elements,
                        windows=[],
                        notes="",
                        warnings=c.warnings,
                        element_count=len(elements),
                        verified_count=sum(e["verified"] for e in elements),
                        ocr_variant="tesseract-cell",
                        extractor_version=CAL_PARSER_VERSION,
                    )
                )
        stats.days_written += len(cells)


def _calendar_elements(cell: CalendarCell, book: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Calendar values are verified only when the book (independently extracted and checked)
    prints the same name and end time for that date."""
    out = []
    for kind, name, end in (
        ("tithi", cell.tithi, cell.tithi_end),
        ("nakshatra", cell.nakshatra, cell.nakshatra_end),
    ):
        if name is None:
            continue
        printed = (end or "").replace(",", ".")
        match = next((e for e in book or [] if e["kind"] == kind), None)
        cross = (
            "n/a"
            if match is None
            else ("agree" if match["name"] == name and match["printed_end"] == printed else "disagree")
        )
        out.append(
            {
                "kind": kind,
                "name": name,
                "printed_end": printed,
                "ends_at": None,
                "name_source": "ocr",
                "cross_check": cross,
                "verified": cross == "agree",
            }
        )
    return out


def _element_dict(e: Any) -> dict[str, Any]:
    d = {f: getattr(e, f) for f in e.__dataclass_fields__}
    d["verified"] = e.verified
    return d


def _window_dict(w: Any) -> dict[str, Any]:
    return {f: getattr(w, f) for f in w.__dataclass_fields__}

"""Parse the TTD festival table (ఉత్సవాదుల పట్టిక) and utsavam schedules into dated events.

Festival table rows (Sarvam OCR, one row per line):
    "<[month] day> | <[tamil month] tamil day> | <weekday> | <events>"
The month name only appears on a month's first row, and OCR sometimes drops it, so each row's
date is resolved as the next date after the previous row whose day-of-month AND weekday both
match what is printed. A misread can therefore never silently move an event to a wrong date:
an unresolvable row is reported, not guessed.

Utsavam schedules carry full dates:
    <title line>
    "తేదీలు | వారము | [పగలు ఉత్సవములు | రాత్రి ఉత్సవములు | ఉత్సవములు]"
    "dd-mm-yyyy | <weekday> | <day events> | <night events>"
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Any

from panchangasetu.telugu.lexicon import GREGORIAN_MONTH_NUMBER, GREGORIAN_MONTHS_TE

EVENTS_PARSER_VERSION = "ttd-events-1"

WEEKDAYS = ("సోమ", "మంగళ", "బుధ", "గురు", "శుక్ర", "శని", "ఆది")  # date.weekday() order
_MONTH_RE = re.compile("(" + "|".join(GREGORIAN_MONTHS_TE) + r")\s*(\d{1,2})")
_DAY_RE = re.compile(r"(?<!\d)(\d{1,2})\s*$")
_YEAR_ONLY_RE = re.compile(r"^\s*(19|20)\d{2}\s*$")
# Running headers that must never become a schedule title.
_RUNNING_HEADER_RE = re.compile(r"(సిద్ధా\S*\s+పంచాంగం|దేవస్థానములు\s+\d+)")
_FULL_DATE_RE = re.compile(r"^(\d{1,2})-(\d{1,2})-(\d{4})$")
_MAX_GAP_DAYS = 45


@dataclass(slots=True)
class Event:
    date: str
    weekday_printed: str | None
    weekday_check: str  # ok | corrected | mismatch | missing
    kind: str  # festival | utsavam
    title: str  # utsavam: the schedule title; festival: "" (the details are the events)
    details: str
    tamil_date: str | None
    page_number: int
    raw: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _weekday_token(cell: str) -> str | None:
    cell = cell.strip()
    for w in WEEKDAYS:
        if cell.startswith(w):
            return w
    return None


def _resolve(
    prev: date, day: int | None, weekday: str | None, month: int | None, year_hint: int
) -> date | None:
    """Next date after `prev` matching day-of-month (and month/weekday when known). With no
    legible day number, only the next matching weekday within a week is accepted."""
    if day == prev.day and (weekday is None or WEEKDAYS[prev.weekday()] == weekday):
        return prev  # a second row for the same date (long event lists wrap onto a new row)
    d = prev
    for _ in range(_MAX_GAP_DAYS if day is not None else 7):
        d += timedelta(days=1)
        if (day is not None and d.day != day) or (month is not None and d.month != month):
            continue
        if weekday is not None and WEEKDAYS[d.weekday()] != weekday:
            continue
        return d
    if month is not None and day is not None:  # first row of the table: nothing to walk from
        try:
            d = date(year_hint, month, day)
        except ValueError:
            return None
        if weekday is None or WEEKDAYS[d.weekday()] == weekday:
            return d
    return None


def parse_festival_table(pages: Iterable[tuple[int, str]], *, start: date) -> tuple[list[Event], list[str]]:
    """`start` is the day before the table's first row (the table opens in March 2026)."""
    events: list[Event] = []
    problems: list[str] = []
    prev = start
    for page_number, text in pages:
        for line in text.splitlines():
            cells = [c.strip() for c in line.split(" | ")]
            if len(cells) < 3 or cells[0].startswith("ఇంగ్లీష్"):
                continue  # headers, month banners ("తులామాసము") and other prose
            # Columns: english [tamil] weekday events. Locate the weekday column.
            wi = next((i for i, c in enumerate(cells[:3]) if _weekday_token(c)), None)
            if wi is None or wi + 1 >= len(cells):
                problems.append(f"p{page_number}: no weekday column: {line[:80]!r}")
                continue
            weekday = _weekday_token(cells[wi])
            english = cells[0]
            tamil = cells[1] if wi == 2 else None
            month_m = _MONTH_RE.search(english)
            day_m = month_m or _DAY_RE.search(english)
            if _YEAR_ONLY_RE.match(english):  # the "month day" part was lost; weekday decides
                day: int | None = None
            elif day_m is None:
                problems.append(f"p{page_number}: no day number: {line[:80]!r}")
                continue
            else:
                day = int(day_m.group(2) if month_m else day_m.group(1))
            month = GREGORIAN_MONTH_NUMBER[month_m.group(1)] if month_m else None
            resolved = _resolve(prev, day, weekday, month, prev.year)
            if resolved is None:
                problems.append(
                    f"p{page_number}: cannot place day {day} ({weekday}) after {prev}: {line[:80]!r}"
                )
                continue
            prev = resolved
            events.append(
                Event(
                    date=resolved.isoformat(),
                    weekday_printed=weekday,
                    weekday_check="ok",
                    kind="festival",
                    title="",
                    details=" | ".join(cells[wi + 1 :]),
                    tamil_date=tamil,
                    page_number=page_number,
                    raw=line,
                )
            )
    return events, problems


def parse_utsavam_schedules(pages: Iterable[tuple[int, str]]) -> tuple[list[Event], list[str]]:
    events: list[Event] = []
    problems: list[str] = []
    title = ""
    for page_number, text in pages:
        last_prose = ""
        for line in text.splitlines():
            line = line.strip().removeprefix("## ")
            if not line:
                continue
            cells = [c.strip() for c in line.split(" | ")]
            if cells[0].startswith("తేదీలు"):
                title = last_prose or title  # the line just above a header names the schedule
                continue
            m = _FULL_DATE_RE.match(cells[0])
            if not m:
                if (
                    " | " not in line
                    and not re.match(r"^\d+\s", line)
                    and not _RUNNING_HEADER_RE.search(line)
                ):
                    last_prose = line
                continue
            try:
                d = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            except ValueError:
                problems.append(f"p{page_number}: invalid date {cells[0]!r}")
                continue
            weekday = _weekday_token(cells[1]) if len(cells) > 1 else None
            check = (
                "missing" if weekday is None else ("ok" if WEEKDAYS[d.weekday()] == weekday else "mismatch")
            )
            if check == "mismatch":
                # Source typos happen ("31-08-2026" between 30-07 and 01-08). If the day after
                # the previous row of the same schedule carries the printed weekday, use it.
                prev_ev = events[-1] if events and events[-1].title == title else None
                candidate = date.fromisoformat(prev_ev.date) + timedelta(days=1) if prev_ev else None
                if (
                    candidate is not None
                    and WEEKDAYS[candidate.weekday()] == weekday
                    and candidate.day == d.day
                ):
                    problems.append(
                        f"p{page_number}: printed {cells[0]} corrected to {candidate} by continuity"
                    )
                    d, check = candidate, "corrected"
                else:
                    problems.append(f"p{page_number}: weekday {weekday} does not match {d}: {line[:80]!r}")
            parts = [c for c in cells[2:] if c and c != "--"]
            events.append(
                Event(
                    date=d.isoformat(),
                    weekday_printed=weekday,
                    weekday_check=check,
                    kind="utsavam",
                    title=title,
                    details=" | ".join(parts),
                    tamil_date=None,
                    page_number=page_number,
                    raw=line,
                )
            )
    return events, problems

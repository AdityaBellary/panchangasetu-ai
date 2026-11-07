"""Segment the TTD panchangam's day-by-day tables into one record per calendar day.

Layout (verified on pp. 100-151 of the 2026-27 edition, in unrotated page space ~689x518 pt):

    | sunrise | sunset | English date | Tamil date | graha-gati | main text (tithi, nakshatra, ...) |
      x<28      28-52     52-85          85-117       117-149      149 -> right edge / chart box

Even pages carry ~10-12 day rows; odd pages carry 3-5 day rows above a lagna table. Day rows
are not separated by rules, so rows are anchored on the exact sunrise/date numbers recovered
from the legacy-font text layer (see `legacy_text`). Every text-layer line of the main column is
assigned to the nearest anchor above it, giving tight per-day crops for OCR.

Dates are assigned by continuity from the first row (19 Mar 2026, Ugadi) and *validated*
against the printed day number, so a misread digit can never silently shift a date.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date, timedelta

import pymupdf

from panchangasetu.structured.legacy_text import decode_legacy_ascii

# Column x-boundaries (pt, unrotated). Detected per page from vertical rules when available.
_DEFAULT_COLUMNS = (4.0, 28.0, 52.0, 85.0, 117.0, 149.0)
_TIME_RE = re.compile(r"^(\d{1,2})-(\d{2})")
_DAY_RE = re.compile(r"(?:^|\D)(\d{1,2})\s*$")
_ROW_MERGE_PT = 8.0


@dataclass(frozen=True, slots=True)
class Box:
    x0: float
    y0: float
    x1: float
    y1: float

    def as_list(self) -> list[float]:
        return [round(self.x0, 1), round(self.y0, 1), round(self.x1, 1), round(self.y1, 1)]


@dataclass(slots=True)
class DayRow:
    page_number: int
    row_index: int  # 0-based within the page
    printed_day: int | None  # English day-of-month as printed (None if not recoverable)
    sunrise: str | None  # "6-22" as printed
    sunset: str | None
    tamil_day: str | None
    main_box: Box
    numbers_in_row: list[str] = field(default_factory=list)  # exact times from the text layer
    date: date | None = None
    date_check: str = "unchecked"  # ok | mismatch | unreadable


@dataclass(slots=True)
class _Line:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str  # decoded


@dataclass(slots=True)
class _RowLines:
    main: list[_Line] = field(default_factory=list)
    sun: list[_Line] = field(default_factory=list)
    date: list[_Line] = field(default_factory=list)
    tamil: list[_Line] = field(default_factory=list)
    nums: list[str] = field(default_factory=list)


def _lines(page: pymupdf.Page) -> list[_Line]:
    out: list[_Line] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            text = decode_legacy_ascii("".join(s["text"] for s in line["spans"]))
            if text.strip():
                x0, y0, x1, y1 = line["bbox"]
                out.append(_Line(x0, y0, x1, y1, text))
    return out


def _rules(page: pymupdf.Page) -> tuple[list[float], list[tuple[float, float, float]]]:
    """Full-width horizontal rule y's and vertical rule segments (x, y0, y1).

    A "full-width" rule may be drawn as several collinear segments (the lagna-table separator is
    two halves), so horizontal segments are grouped by y and their x-coverage is summed.
    """
    segments: list[tuple[float, float, float]] = []  # (y, x0, x1)
    verticals: list[tuple[float, float, float]] = []
    width = page.rect.width
    for path in page.get_drawings():
        for item in path["items"]:
            if item[0] == "l":
                a, b = item[1], item[2]
                if abs(a.y - b.y) < 0.6 and abs(a.x - b.x) > 20:
                    segments.append((a.y, min(a.x, b.x), max(a.x, b.x)))
                elif abs(a.x - b.x) < 0.6 and abs(a.y - b.y) > 40:
                    verticals.append((a.x, min(a.y, b.y), max(a.y, b.y)))
            elif item[0] == "re":
                r = item[1]
                if r.height < 1.5 and r.width > 20:
                    segments.append((r.y0, r.x0, r.x1))
                elif r.width < 1.5 and r.height > 40:
                    verticals.append((r.x0, r.y0, r.y1))
    horizontals: list[float] = []
    for y in sorted({round(s[0]) for s in segments}):
        spans = sorted((x0, x1) for sy, x0, x1 in segments if abs(sy - y) <= 1.0)
        covered, end = 0.0, -1.0
        for x0, x1 in spans:  # union of intervals
            if x0 > end:
                covered += x1 - x0
                end = x1
            elif x1 > end:
                covered += x1 - end
                end = x1
        if covered > 0.6 * width:
            horizontals.append(float(y))
    return horizontals, verticals


def _row_region(page: pymupdf.Page, horizontals: list[float], anchors: list[float]) -> tuple[float, float]:
    """Day rows live between the header rule above the first anchor and the next full-width
    rule below the last anchor (the lagna table on odd pages, or the page border)."""
    top = max((h for h in horizontals if h < anchors[0] - 2), default=0.0)
    bottom = min((h for h in horizontals if h > anchors[-1] + 2), default=page.rect.height)
    return top, bottom


def _cluster(values: list[float]) -> list[float]:
    out: list[float] = []
    for v in sorted(values):
        if not out or v - out[-1] > _ROW_MERGE_PT:
            out.append(v)
    return out


def segment_page(page: pymupdf.Page) -> list[DayRow]:
    """Return day rows for one daily-table page. The page's rotation is set to 0 (unrotated
    space) for the duration of the call and restored afterwards."""
    original = page.rotation
    page.set_rotation(0)
    try:
        return _segment(page)
    finally:
        page.set_rotation(original)


def _segment(page: pymupdf.Page) -> list[DayRow]:
    _, c_set, c_date, c_tamil, c_graha, c_main = _DEFAULT_COLUMNS
    horizontals, verticals = _rules(page)
    lines = _lines(page)

    # Anchors: a sunrise time in column 1, or a day number in the date column.
    anchor_ys: list[float] = []
    for ln in lines:
        if (ln.x0 < c_set and _TIME_RE.match(ln.text.strip())) or (
            c_date - 4 <= ln.x0 < c_tamil and _DAY_RE.search(ln.text) and "-" not in ln.text
        ):
            anchor_ys.append(ln.y0)
    anchors = _cluster(anchor_ys)
    if not anchors:
        return []
    top, bottom = _row_region(page, horizontals, anchors)
    anchors = [a for a in anchors if top < a < bottom]
    if not anchors:
        return []

    def row_of(y_center: float) -> int | None:
        # Lines are assigned by their vertical centre: the main text is set larger than the
        # date column, so a day's first line *starts* a few points above its date anchor.
        idx = None
        for i, a in enumerate(anchors):
            if y_center >= a - 1:
                idx = i
        return idx

    rows = [_RowLines() for _ in anchors]
    for ln in lines:
        center = (ln.y0 + ln.y1) / 2
        if not top < center < bottom:
            continue
        i = row_of(center)
        if i is None:
            continue
        row = rows[i]
        # MuPDF sometimes merges a day's first line with the graha-gati column into one line,
        # so anything that reaches well into the main column counts as main text.
        if ln.x0 >= c_main - 3 or ln.x1 > c_main + 20:
            row.main.append(ln)
            row.nums.extend(re.findall(r"\d{1,2}-\d{2}", ln.text))
        elif ln.x0 < c_date - 4:
            row.sun.append(ln)
        elif ln.x0 < c_tamil:
            row.date.append(ln)
        elif ln.x0 < c_graha:
            row.tamil.append(ln)

    # Right edge of the main column: page border, or a chart box drawn inside the column.
    right_border = max((x for x, _, _ in verticals), default=page.rect.width - 4)
    out: list[DayRow] = []
    for i, (anchor, row) in enumerate(zip(anchors, rows, strict=True)):
        main = row.main
        if main:
            y0 = min(ln.y0 for ln in main)
            y1 = max(ln.y1 for ln in main)
        else:
            y0, y1 = anchor, (anchors[i + 1] if i + 1 < len(anchors) else bottom)
        x1 = right_border
        for vx, vy0, vy1 in verticals:
            if c_main + 50 < vx < right_border - 5 and vy0 < y1 and vy1 > y0:
                x1 = min(x1, vx)  # chart box overlapping this row
        sun_text = " ".join(ln.text for ln in row.sun)
        times = re.findall(r"\d{1,2}-\d{2}", sun_text.replace(" ", ""))
        date_text = " ".join(ln.text for ln in row.date)
        m = _DAY_RE.search(date_text.replace("2026", "").replace("2027", "").strip())
        if m is None:  # first row of a month: the day number is merged into the sunrise span
            m = re.search(r"\d-\d{2}\d-\d{2}\D+(\d{1,2})\b", sun_text.replace(" ", ""))
        tamil = " ".join(ln.text for ln in row.tamil).strip() or None
        out.append(
            DayRow(
                page_number=page.number + 1,
                row_index=i,
                printed_day=int(m.group(1)) if m else None,
                sunrise=times[0] if times else None,
                sunset=times[1] if len(times) > 1 else None,
                tamil_day=tamil,
                main_box=Box(c_main, y0 - 1.5, x1 - 1, y1 + 1.5),
                numbers_in_row=list(row.nums),
            )
        )
    return out


def assign_dates(rows: list[DayRow], first_date: date) -> list[DayRow]:
    """Assign consecutive dates and check them against printed day numbers."""
    for offset, row in enumerate(rows):
        row.date = first_date + timedelta(days=offset)
        if row.printed_day is None:
            row.date_check = "unreadable"
        else:
            row.date_check = "ok" if row.printed_day == row.date.day else "mismatch"
    return rows


def iter_daily_rows(doc: pymupdf.Document, pages: list[int]) -> Iterator[DayRow]:
    for number in pages:
        yield from segment_page(doc[number - 1])

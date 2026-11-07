"""Read TTD wall-calendar month pages (image-only) cell by cell.

Layout: rows are weekdays (Sunday first), columns are weeks; dates that fall in a sixth week
wrap into the first column (e.g. 31 Jan 2027 sits above 3 Jan). Each cell holds a big date
number, then "<tithi> <part-of-day>.<h-mm>", "<nakshatra> <part-of-day>.<h-mm>|పూర్తి", then
varjyam / durmuhurtam windows.

Because the layout is a regular grid, a cell's date follows from calendar arithmetic. OCR of
the big numbers is only used to *fit and validate* the grid (it misses many coloured numbers,
which is fine). Cell text is OCR'd per cell, so content can never land on the wrong date.

Times read from these low-resolution scans cannot be verified the way book rows are (no
ghatis, no text layer), so every value carries `time_verified=False` unless a second source
agrees (see `cross_check`).
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta

import pymupdf

from panchangasetu.structured.panchanga_parse import match_name

CAL_PARSER_VERSION = "wall-cal-1"
_UPSCALE = 3
_CELL_RE = re.compile(
    r"(?P<name>[^\s\d.,()|:;'\"!?]+)\s*(?:(?P<p>ఉ|ప|మ|సా|రా|తె)[.,]\s?(?P<t>\d{1,2}[-.=]\d{0,2})|(?P<full>పూర్తి))"
)

TsvRunner = Callable[[bytes, str, int], str]  # (png, langs, psm) -> TSV/text


@dataclass(slots=True)
class CalendarCell:
    date: date
    text: str
    tithi: str | None = None
    tithi_end: str | None = None  # as read, e.g. "సా.6-08" (unverified)
    nakshatra: str | None = None
    nakshatra_end: str | None = None
    warnings: list[str] = field(default_factory=list)


def month_position(d: date) -> tuple[int, int]:
    """(column, row) of a date: rows are weekdays Sunday..Saturday, columns are weeks."""
    first = d.replace(day=1)
    lead = (first.weekday() + 1) % 7  # Sunday = 0
    p = lead + d.day - 1
    col, row = divmod(p, 7)
    if col >= 5:  # sixth week wraps into the first column
        col -= 5
    return col, row


def _big_numbers(tsv: str, min_height: int) -> list[tuple[int, int, int, int, int]]:
    """(value, x_center, y_top, height, width) of large standalone numbers."""
    out = []
    for line in tsv.splitlines()[1:]:
        c = line.split("\t")
        if len(c) >= 12 and c[0] == "5" and re.fullmatch(r"\d{1,2}", c[11].strip()):
            left, top, width, height = (int(v) for v in c[6:10])
            if height >= min_height:
                out.append((int(c[11]), left + width // 2, top, height, width))
    return out


@dataclass(frozen=True, slots=True)
class Grid:
    x0: float  # x centre of column 0
    dx: float
    y0: float  # top of the big number in row 0
    dy: float
    number_height: float

    def cell(self, col: int, row: int) -> pymupdf.IRect:
        cx = self.x0 + col * self.dx
        top = self.y0 + row * self.dy + self.number_height * 1.05
        return pymupdf.IRect(
            int(cx - self.dx * 0.5),
            int(top),
            int(cx + self.dx * 0.5),
            int(top + self.dy - self.number_height * 1.2),
        )


def fit_grid(numbers: list[tuple[int, int, int, int, int]], year: int, month: int) -> tuple[Grid, int]:
    """Robust linear fit of column x / row y from detected numbers whose value is a valid day of
    the month. Returns the grid and how many detections agreed with it."""
    obs = []
    for value, x, y, h, _w in numbers:
        try:
            d = date(year, month, value)
        except ValueError:
            continue
        col, row = month_position(d)
        obs.append((col, row, x, y, h))

    def fit(points: list[tuple[int, float]]) -> tuple[float, float]:
        # Median of pairwise slopes (Theil-Sen) resists misread numbers.
        slopes = [
            (b[1] - a[1]) / (b[0] - a[0])
            for i, a in enumerate(points)
            for b in points[i + 1 :]
            if b[0] != a[0]
        ]
        slope = statistics.median(slopes)
        intercept = statistics.median(p[1] - slope * p[0] for p in points)
        return intercept, slope

    x0, dx = fit([(c, x) for c, _, x, _, _ in obs])
    y0, dy = fit([(r, y) for _, r, _, y, _ in obs])
    agree = sum(
        abs(x0 + c * dx - x) < dx * 0.15 and abs(y0 + r * dy - y) < dy * 0.15 for c, r, x, y, _ in obs
    )
    number_height = float(statistics.median(o[4] for o in obs))
    return Grid(x0, dx, y0, dy, number_height), agree


def grid_is_plausible(grid: Grid, agree: int, width: int, height: int) -> bool:
    """Reject degenerate fits (too few agreeing numbers, or spacings no TTD calendar has)."""
    return agree >= 4 and 0.08 * width < grid.dx < 0.25 * width and 0.04 * height < grid.dy < 0.12 * height


def median_grid(grids: list[Grid]) -> Grid:
    return Grid(
        *(statistics.median(getattr(g, f) for g in grids) for f in ("x0", "dx", "y0", "dy", "number_height"))
    )


def parse_cell(text: str, day: date) -> CalendarCell:
    cell = CalendarCell(date=day, text=text)
    for m in _CELL_RE.finditer(text):
        found = match_name(m.group("name"), ["tithi", "nakshatra"])
        if found is None:
            continue
        kind, name, _score = found
        end = "పూర్తి" if m.group("full") else f"{m.group('p')}.{m.group('t')}"
        if kind == "tithi" and cell.tithi is None:
            cell.tithi, cell.tithi_end = name, end
        elif kind == "nakshatra" and cell.nakshatra is None and cell.tithi is not None:
            cell.nakshatra, cell.nakshatra_end = name, end
    if cell.tithi is None:
        cell.warnings.append("no tithi read")
    if cell.nakshatra is None:
        cell.warnings.append("no nakshatra read")
    return cell


def page_image(doc: pymupdf.Document, page_index: int) -> pymupdf.Pixmap:
    """The page's single embedded scan at native resolution, upscaled for OCR."""
    xref = doc[page_index].get_images()[0][0]
    pix = pymupdf.Pixmap(doc, xref)
    if pix.alpha or pix.colorspace.n != 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    return pymupdf.Pixmap(pix, pix.width * _UPSCALE, pix.height * _UPSCALE, None)


def detect_grid(
    doc: pymupdf.Document, page_index: int, year: int, month: int, run: TsvRunner
) -> tuple[Grid | None, int]:
    img = page_image(doc, page_index)
    tsv = run(img.tobytes("png"), "eng", 11)
    numbers = _big_numbers(tsv, min_height=int(img.height * 0.015))
    try:
        grid, agree = fit_grid(numbers, year, month)
    except statistics.StatisticsError:
        return None, 0
    return (grid if grid_is_plausible(grid, agree, img.width, img.height) else None), agree


def read_month(
    doc: pymupdf.Document, page_index: int, year: int, month: int, grid: Grid, run: TsvRunner
) -> list[CalendarCell]:
    img = page_image(doc, page_index)
    bounds = pymupdf.IRect(0, 0, img.width, img.height)
    cells: list[CalendarCell] = []
    d = date(year, month, 1)
    while d.month == month:
        col, row = month_position(d)
        rect = grid.cell(col, row) & bounds
        if rect.is_empty or rect.width < 20 or rect.height < 20:
            cells.append(CalendarCell(date=d, text="", warnings=["cell outside image"]))
        else:
            crop = pymupdf.Pixmap(pymupdf.csRGB, rect, False)
            crop.copy(img, rect)
            cells.append(parse_cell(run(crop.tobytes("png"), "tel", 6), d))
        d += timedelta(days=1)
    return cells

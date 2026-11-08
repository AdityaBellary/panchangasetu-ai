"""Resolve date expressions in Telugu, English and romanised Telugu to concrete dates.

"Today" is evaluated in the configured timezone (default Asia/Kolkata), never the server's.
Supported: today/tomorrow/yesterday/day-after (ఈరోజు, రేపు, నిన్న, ఎల్లుండి, ee roju, repu …),
ISO and Indian day-first numeric dates, "15 December 2026" / "December 15" / "డిసెంబర్ 15",
weekdays ("next Friday", "శుక్రవారం"), and ranges ("this week", "next month", "ఈ నెల",
"in December 2026"). A date without a year resolves to its nearest sensible occurrence.
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from panchangasetu.telugu.lexicon import GREGORIAN_MONTH_NUMBER

Clock = Callable[[], datetime]


def system_clock(tz: str = "Asia/Kolkata") -> Clock:
    zone = ZoneInfo(tz)
    return lambda: datetime.now(zone)


@dataclass(frozen=True, slots=True)
class DateSpec:
    start: date
    end: date  # inclusive; == start for a single day
    matched: str  # the text that produced it
    kind: str  # day | range

    @property
    def is_day(self) -> bool:
        return self.start == self.end


_EN_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m} | {
    m.lower(): i for i, m in enumerate(calendar.month_abbr) if m
}
_EN_MONTHS["sept"] = 9
_MONTH_ALT = "|".join(sorted((re.escape(m) for m in _EN_MONTHS), key=len, reverse=True))
_TE_MONTH_ALT = "|".join(sorted(GREGORIAN_MONTH_NUMBER, key=len, reverse=True))
# Telugu month names need boundaries: "మే" (May) must not match inside "మేషరాశి". Case suffixes
# that attach directly ("మేలో", "డిసెంబర్‌లో", "జూన్ నెల") are allowed.
_TE_B = r"(?<![\u0C00-\u0C7F])"
_TE_E = r"(?:లో|లోని|న|\u200c?లో)?(?![\u0C00-\u0C7F])"

_RELATIVE: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"day after tomorrow|ఎల్లుండి|\bellundi\b", re.I), 2),
    (re.compile(r"\btomorrow\b|రేపు|\brepu\b", re.I), 1),
    (re.compile(r"\byesterday\b|నిన్న|\bninna\b", re.I), -1),
    (re.compile(r"\btoday\b|\btonight\b|ఈ\s?రోజు|నేడు|ఇవాళ|ఈనాడు|\bee\s?roju\b|\bivala\b|\bivvala\b", re.I), 0),
]
_ISO = re.compile(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b")
_NUMERIC = re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](20\d{2}|\d{2})\b")  # day-first (Indian usage)
_DAY_MONTH = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s*(?:of\s+)?({_MONTH_ALT})\b\.?,?\s*(20\d{{2}})?", re.I
)
_MONTH_DAY = re.compile(rf"\b({_MONTH_ALT})\b\.?\s*(\d{{1,2}})(?:st|nd|rd|th)?\b,?\s*(20\d{{2}})?", re.I)
# "(?!\d)": in "27 మార్చి 2027" the year must not be read as "మార్చి 20" (held-out eval, Phase 9).
_TE_MONTH_DAY = re.compile(rf"{_TE_B}({_TE_MONTH_ALT})\s*(\d{{1,2}})(?!\d)(?:\s*,?\s*(20\d{{2}}))?")
_TE_DAY_MONTH = re.compile(rf"(\d{{1,2}})\s*{_TE_B}({_TE_MONTH_ALT}){_TE_E}(?:\s*,?\s*(20\d{{2}}))?")
_MONTH_ONLY = re.compile(rf"\b(?:in|during|for)\s+({_MONTH_ALT})\b\.?\s*(20\d{{2}})?", re.I)
_TE_MONTH_ONLY = re.compile(rf"{_TE_B}({_TE_MONTH_ALT}){_TE_E}\s*(?:నెల|మాసం)?\s*(20\d{{2}})?")

_WEEKDAYS_EN = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_WEEKDAYS_TE = ["సోమవారం", "మంగళవారం", "బుధవారం", "గురువారం", "శుక్రవారం", "శనివారం", "ఆదివారం"]
_WEEKDAY_RE = re.compile(r"\b(next|this|coming|last)?\s*(" + "|".join(_WEEKDAYS_EN) + r")\b", re.I)

_RANGES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bnext week\b|వచ్చే వారం|\bvachche varam\b", re.I), "next_week"),
    (re.compile(r"\bthis week\b|ఈ వారం|\bee varam\b", re.I), "this_week"),
    (re.compile(r"\bnext month\b|వచ్చే నెల|\bvachche nela\b", re.I), "next_month"),
    (re.compile(r"\bthis month\b|ఈ నెల|\bee nela\b", re.I), "this_month"),
]


def _infer_year(month: int, day: int, today: date) -> date | None:
    """No year given: this year's date unless it is well in the past, then next year's."""
    for year in (today.year, today.year + 1):
        try:
            d = date(year, month, day)
        except ValueError:
            continue
        if d >= today - timedelta(days=60) or year > today.year:
            return d
    return None


def _ymd(year: int | None, month: int, day: int, today: date) -> date | None:
    if year is None:
        return _infer_year(month, day, today)
    try:
        return date(year if year > 99 else 2000 + year, month, day)
    except ValueError:
        return None


def _month_range(year: int, month: int) -> tuple[date, date]:
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def parse_dates(text: str, today: date) -> list[DateSpec]:
    """All date expressions in `text`, in order of appearance."""
    found: list[tuple[int, DateSpec]] = []
    taken: list[tuple[int, int]] = []

    def add(m: re.Match[str], spec: DateSpec | None) -> None:
        if spec is None or any(not (m.end() <= a or m.start() >= b) for a, b in taken):
            return
        taken.append((m.start(), m.end()))
        found.append((m.start(), spec))

    for rx, offset in _RELATIVE:
        for m in rx.finditer(text):
            d = today + timedelta(days=offset)
            add(m, DateSpec(d, d, m.group(0), "day"))
    for m in _ISO.finditer(text):
        parsed = _ymd(int(m.group(1)), int(m.group(2)), int(m.group(3)), today)
        add(m, DateSpec(parsed, parsed, m.group(0), "day") if parsed else None)
    for m in _NUMERIC.finditer(text):
        parsed = _ymd(int(m.group(3)), int(m.group(2)), int(m.group(1)), today)
        add(m, DateSpec(parsed, parsed, m.group(0), "day") if parsed else None)
    for m in _DAY_MONTH.finditer(text):
        y = int(m.group(3)) if m.group(3) else None
        parsed = _ymd(y, _EN_MONTHS[m.group(2).lower()], int(m.group(1)), today)
        add(m, DateSpec(parsed, parsed, m.group(0).strip(), "day") if parsed else None)
    for m in _MONTH_DAY.finditer(text):
        y = int(m.group(3)) if m.group(3) else None
        parsed = _ymd(y, _EN_MONTHS[m.group(1).lower()], int(m.group(2)), today)
        add(m, DateSpec(parsed, parsed, m.group(0).strip(), "day") if parsed else None)
    for rx, mi, di in ((_TE_MONTH_DAY, 1, 2), (_TE_DAY_MONTH, 2, 1)):
        for m in rx.finditer(text):
            y = int(m.group(3)) if m.group(3) else None
            parsed = _ymd(y, GREGORIAN_MONTH_NUMBER[m.group(mi)], int(m.group(di)), today)
            add(m, DateSpec(parsed, parsed, m.group(0).strip(), "day") if parsed else None)
    for m in _WEEKDAY_RE.finditer(text):
        target = _WEEKDAYS_EN.index(m.group(2).lower())
        ahead = (target - today.weekday()) % 7
        qualifier = (m.group(1) or "").lower()
        if qualifier == "next" and ahead == 0:
            ahead = 7
        if qualifier == "last":
            ahead = ahead - 7 if ahead else -7
        d = today + timedelta(days=ahead)
        add(m, DateSpec(d, d, m.group(0).strip(), "day"))
    for i, name in enumerate(_WEEKDAYS_TE):
        for m in re.finditer(name, text):
            d = today + timedelta(days=(i - today.weekday()) % 7)
            add(m, DateSpec(d, d, m.group(0), "day"))
    for rx, kind in _RANGES:
        for m in rx.finditer(text):
            if kind == "this_week":
                start = today - timedelta(days=today.weekday())
                add(m, DateSpec(today, start + timedelta(days=6), m.group(0), "range"))
            elif kind == "next_week":
                start = today - timedelta(days=today.weekday()) + timedelta(days=7)
                add(m, DateSpec(start, start + timedelta(days=6), m.group(0), "range"))
            else:
                y, mo = today.year, today.month
                if kind == "next_month":
                    y, mo = (y + 1, 1) if mo == 12 else (y, mo + 1)
                start, end = _month_range(y, mo)
                add(
                    m,
                    DateSpec(max(start, today) if kind == "this_month" else start, end, m.group(0), "range"),
                )
    for m in _MONTH_ONLY.finditer(text):
        mo = _EN_MONTHS[m.group(1).lower()]
        y = int(m.group(2)) if m.group(2) else (today.year if mo >= today.month else today.year + 1)
        add(m, DateSpec(*_month_range(y, mo), m.group(0).strip(), "range"))
    for m in _TE_MONTH_ONLY.finditer(text):
        mo = GREGORIAN_MONTH_NUMBER[m.group(1)]
        y = int(m.group(2)) if m.group(2) else (today.year if mo >= today.month else today.year + 1)
        add(m, DateSpec(*_month_range(y, mo), m.group(0).strip(), "range"))
    return [spec for _, spec in sorted(found, key=lambda t: t[0])]

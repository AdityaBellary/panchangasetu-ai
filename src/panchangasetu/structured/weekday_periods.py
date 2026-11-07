"""Rahukalam, yamagandam and gulika: fixed by weekday, so answered from the book's printed table
(config/weekday_periods.yaml, transcribed from p. 83 and checked against the page image) rather
than by an LLM reading a three-column OCR table."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from panchangasetu.retrieval.evidence import Evidence

_DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_WEEKDAY_EN = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_LABEL = {"rahukalam": "Rahukalam", "yamagandam": "Yamagandam", "gulika": "Gulika kalam"}


@dataclass(frozen=True, slots=True)
class WeekdayPeriods:
    document: str
    pages: list[int]
    table: dict[str, dict[str, list[str]]]  # kind -> weekday -> [start, end, printed]

    @classmethod
    def load(cls, path: Path | None) -> WeekdayPeriods | None:
        if path is None or not path.is_file():
            return None
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        table = {k: {d: [str(x) for x in v[d]] for d in _DAYS} for k, v in raw["periods"].items()}
        return cls(str(raw["source"]["document"]), [int(p) for p in raw["source"]["pages"]], table)

    @property
    def kinds(self) -> frozenset[str]:
        return frozenset(self.table)

    def evidence(self, kind: str, day: date) -> Evidence:
        start, end, printed = self.table[kind][_DAYS[day.weekday()]]
        weekday = _WEEKDAY_EN[day.weekday()]
        text = (
            f"[{_LABEL[kind]} for {day.isoformat()} ({weekday}), from the weekday table] "
            f"{_LABEL[kind]} on {weekday}s: {start} to {end} (printed {printed}; counted from 6 AM, "
            "shifts with actual sunrise and sunset) [verified]"
        )
        return Evidence(
            f"period:{kind}:{day.isoformat()}",
            "weekday_period",
            "general",
            text,
            self.document,
            list(self.pages[:1]),
            date=day.isoformat(),
            verified=True,
            source_kind="weekday_table",
            data={"period": kind, "start": start, "end": end, "printed": printed},
        )

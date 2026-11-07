"""Evidence: the only thing answer generation may use.

Every item carries its provenance (document, page, date) and whether its values were verified,
so answers can cite sources and never present unverified OCR as fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Evidence:
    id: str
    kind: str  # day | occurrence | event | day_note | chunk
    collection: str  # panchanga | general
    text: str  # rendered for the LLM / for display
    document: str
    pages: list[int]
    date: str | None = None
    verified: bool | None = None  # None = not applicable (prose)
    score: float | None = None
    source_kind: str = ""  # book_daily_table | wall_calendar | festival_table | chunk section ...
    data: dict[str, Any] = field(default_factory=dict)  # structured payload (elements, windows ...)

    def citation(self) -> str:
        pages = ", ".join(str(p) for p in self.pages[:3])
        return f"{self.document}, p. {pages}" + (f" ({self.date})" if self.date else "")

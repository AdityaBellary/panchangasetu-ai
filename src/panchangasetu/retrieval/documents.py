"""What goes into the vector index, and how each item is rendered for embedding.

Indexed: chunks of the current chunker version, except sections whose facts are served from
the structured tables (daily panchangam rows, wall-calendar pages) - their OCR text is noisy
and a vector hit on it could compete with the verified structured answer.

Each document is rendered as: [section description] + heading path + text. The section
description is a short English gloss of the curator's section label; it is free (no LLM) and
gives English queries a lexical/semantic bridge into Telugu content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Sections answered from panchanga_days / panchanga_events instead of vector search.
STRUCTURED_ONLY_SECTIONS = frozenset({"daily_panchangam_table", "monthly_calendar"})

SECTION_GLOSS: dict[str, str] = {
    "stotras": "Stotras and daily prayer slokas (nitya dhyana slokas) to Sri Venkateswara and deities",
    "samvatsara_phalalu": "Parabhava samvatsara year predictions: ruling planets, kartari, prices, rain",
    "rasi_phalalu": "Rasi phalalu: yearly and monthly horoscope predictions for each zodiac sign (moon sign)",
    "panchanga_vignanam": "Panchanga knowledge and muhurtha rules: marriage, birth, house, vastu, omens",
    "shubha_muhurthams": "Auspicious muhurtham dates: marriage, upanayanam, griha pravesham, annaprasana",
    "graha_rasi_pravesham": "Planetary transit dates: when the Sun and planets enter each zodiac sign",
    "karthelu": "Karthelu (solar nakshatra periods) start dates and rainfall predictions",
    "masa_dates": "Lunar month (masa) start and end dates and month lords for 2026-27",
    "ttd_festival_calendar": "TTD festival calendar: dates of festivals, utsavams and observances 2026-27",
    "ttd_utsavam_schedules": "TTD utsavam and brahmotsavam schedules with vahana seva dates",
}


@dataclass(slots=True)
class IndexDocument:
    id: str  # stable: chunk id
    collection: str
    text: str  # what is embedded
    display_text: str  # what is shown / passed to the LLM
    payload: dict[str, Any] = field(default_factory=dict)


def render_for_embedding(section_type: str, heading_path: list[str], text: str, *, gloss: bool = True) -> str:
    parts = []
    if gloss and section_type in SECTION_GLOSS:
        parts.append(SECTION_GLOSS[section_type])
    if heading_path:
        parts.append(" > ".join(heading_path))
    parts.append(text)
    return "\n".join(parts)

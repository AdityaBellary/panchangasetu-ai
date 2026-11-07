"""Today at a glance for the UI header: tithi (with moon phase), nakshatra, rahukalam, sunrise.

Read from the same structured tables the answers use; no LLM, no search."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from panchangasetu.retrieval.structured import StructuredRetriever
from panchangasetu.structured.weekday_periods import WeekdayPeriods
from panchangasetu.telugu.lexicon import MASAS, NAKSHATRAS, TITHIS, english_name

_TITHI_ORDER = list(TITHIS)


def tithi_number(name: str, paksha: str | None) -> int | None:
    """1-15 waxing (శుద్ధ), 16-30 waning (బహుళ); 15 = full moon, 30 = new moon."""
    if name == "పూర్ణిమ":
        return 15
    if name == "అమావాస్య":
        return 30
    if name not in _TITHI_ORDER:
        return None
    n = _TITHI_ORDER.index(name) + 1
    return n + 15 if paksha == "బహుళ" else n


def _current(
    elements: list[dict[str, Any]], following: list[dict[str, Any]], kind: str, now: datetime
) -> dict[str, Any] | None:
    """The element of `kind` in force at `now`: the day's own, or the next one once it has ended."""
    items = [e for e in elements if e.get("kind") == kind]
    for e in items:
        ends = e.get("ends_at")
        if not ends or datetime.fromisoformat(ends) > now.replace(tzinfo=None):
            return e
    nxt = [e for e in following if e.get("kind") == kind]
    return nxt[0] if nxt else (items[-1] if items else None)


def today_facts(
    structured: StructuredRetriever, periods: WeekdayPeriods | None, now: datetime
) -> dict[str, Any]:
    day = now.date()
    out: dict[str, Any] = {"date": day.isoformat(), "weekday": day.strftime("%A"), "available": False}
    found = structured.days(day, day)
    if periods is not None:
        rk = periods.evidence("rahukalam", day).data
        out["rahukalam"] = {"start": rk["start"], "end": rk["end"]}
    if not found:
        return out
    ev = found[0]
    elements = list(ev.data.get("elements", []))
    following = list(ev.data.get("following", []))
    out |= {
        "available": True,
        "verified": bool(ev.verified),
        "sunrise": ev.data.get("sunrise"),
        "sunset": ev.data.get("sunset"),
        "source": {"document": ev.document, "pages": ev.pages},
    }
    t = _current(elements, following, "tithi", now)
    if t:
        paksha = t.get("paksha")
        masa = t.get("masa")
        out["tithi"] = {
            "name": t["name"],
            "name_en": english_name(TITHIS, t["name"]),
            "ends_at": t.get("ends_at"),
            "paksha": paksha,
            "paksha_en": {"శుద్ధ": "Shukla", "బహుళ": "Krishna"}.get(paksha or "", ""),
            "masa": masa,
            "masa_en": english_name(MASAS, masa) if masa else None,
            "number": tithi_number(t["name"], paksha),
        }
    n = _current(elements, following, "nakshatra", now)
    if n:
        out["nakshatra"] = {
            "name": n["name"],
            "name_en": english_name(NAKSHATRAS, n["name"]),
            "ends_at": n.get("ends_at"),
        }
    return out

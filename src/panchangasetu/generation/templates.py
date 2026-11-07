"""Deterministic answers for verified day lookups: no LLM, no cost, nothing to hallucinate.

Used when the route is a day lookup, the evidence comes from the book's verified daily table, and
every requested attribute is a printed field. Anything else goes to the LLM path.
"""

from __future__ import annotations

from datetime import date, datetime

from panchangasetu.retrieval.evidence import Evidence
from panchangasetu.telugu.lexicon import KARANAS, MASAS, NAKSHATRAS, TITHIS, YOGAS, english_name

TEMPLATE_ATTRIBUTES = frozenset(
    {"tithi", "nakshatra", "yoga", "karana", "varjyam", "durmuhurtam", "amrutham", "sunrise", "sunset",
     "masa", "paksha", "panchangam"}
)  # fmt: skip
_TABLES = {"tithi": TITHIS, "nakshatra": NAKSHATRAS, "yoga": YOGAS, "karana": KARANAS}
_LABEL_TE = {
    "tithi": "తిథి",
    "nakshatra": "నక్షత్రం",
    "yoga": "యోగం",
    "karana": "కరణం",
    "varjyam": "వర్జ్యం",
    "durmuhurtam": "దుర్ముహూర్తం",
    "amrutham": "అమృతకాలం",
}
_LABEL_EN = {
    "tithi": "Tithi",
    "nakshatra": "Nakshatram",
    "yoga": "Yogam",
    "karana": "Karanam",
    "varjyam": "Varjyam",
    "durmuhurtam": "Durmuhurtam",
    "amrutham": "Amrutha kalam",
}
_MONTHS_TE = ["జనవరి", "ఫిబ్రవరి", "మార్చి", "ఏప్రిల్", "మే", "జూన్", "జూలై", "ఆగస్టు", "సెప్టెంబర్", "అక్టోబర్", "నవంబర్", "డిసెంబర్"]
_WEEKDAY_TE = ("సోమవారం", "మంగళవారం", "బుధవారం", "గురువారం", "శుక్రవారం", "శనివారం", "ఆదివారం")
_PAKSHA_EN = {"శుద్ధ": "Shukla paksha", "బహుళ": "Krishna paksha"}


def _en_time(iso: str | None, day: date) -> str:
    if not iso:
        return "beyond the next sunrise"
    dt = datetime.fromisoformat(iso)
    clock = dt.strftime("%I:%M %p").lstrip("0")
    return clock if dt.date() == day else f"{clock} on {dt.strftime('%d %b')}"


def _te_time(iso: str | None, day: date) -> str:
    if not iso:
        return "మరుసటి సూర్యోదయం తరువాత వరకు"
    dt = datetime.fromisoformat(iso)
    part = ("తెల్లవారుజాము" if dt.hour < 6 else "ఉదయం" if dt.hour < 12 else "మధ్యాహ్నం" if dt.hour < 16
            else "సాయంత్రం" if dt.hour < 19 else "రాత్రి")  # fmt: skip
    clock = f"{dt.hour % 12 or 12}:{dt.minute:02d}"
    prefix = "" if dt.date() == day else f"{dt.day} {_MONTHS_TE[dt.month - 1]} "
    return f"{prefix}{part} {clock}"


def _en_name(kind: str, telugu: str) -> str:
    table = _TABLES.get(kind)
    return english_name(table, telugu) if table else telugu


def can_template(evidence: list[Evidence], attributes: list[str]) -> bool:
    days = [e for e in evidence if e.kind == "day"]
    return (
        bool(days)
        and all(e.verified for e in days)
        and bool(attributes)
        and set(attributes) <= TEMPLATE_ATTRIBUTES
    )


def render_day_answer(ev: Evidence, attributes: list[str], lang: str) -> str:
    d = date.fromisoformat(ev.date or "")
    elements: list[dict] = ev.data.get("elements", [])  # type: ignore[type-arg]
    windows: list[dict] = ev.data.get("windows", [])  # type: ignore[type-arg]
    following: list[dict] = ev.data.get("following", [])  # type: ignore[type-arg]
    wanted = set(attributes)
    if "panchangam" in wanted:
        wanted |= {"tithi", "nakshatra", "yoga", "karana", "varjyam", "durmuhurtam", "amrutham", "sunrise"}
    if "masa" in wanted or "paksha" in wanted:
        wanted.add("tithi")
    lines: list[str] = []
    for kind in ("tithi", "nakshatra", "yoga", "karana"):
        if kind not in wanted:
            continue
        items = [e for e in elements if e.get("kind") == kind]
        if not items:
            lines.append(
                f"{_LABEL_TE[kind]}: పత్రంలో లభ్యం కాలేదు."
                if lang == "te"
                else f"{_LABEL_EN[kind]}: not legible in the source."
            )
            continue
        parts = []
        for e in items:
            if lang == "te":
                parts.append(f"{e['name']} {_te_time(e.get('ends_at'), d)} వరకు")
            else:
                parts.append(f"{_en_name(kind, e['name'])} until {_en_time(e.get('ends_at'), d)}")
        nxt = next((e for e in following if e.get("kind") == kind), None)
        if nxt and items[-1].get("ends_at"):
            parts.append(f"తరువాత {nxt['name']}" if lang == "te" else f"then {_en_name(kind, nxt['name'])}")
        label = _LABEL_TE[kind] if lang == "te" else _LABEL_EN[kind]
        sep = ", " if lang == "te" else "; "
        lines.append(f"{label}: {sep.join(parts)}.")
        if kind == "tithi" and ({"masa", "paksha"} & wanted) and items[0].get("masa"):
            first = items[0]
            adhika = first.get("adhika")
            if lang == "te":
                lines.append(
                    f"మాసం/పక్షం: {'అధిక ' if adhika else ''}{first['masa']}, {first.get('paksha')} పక్షం."
                )
            else:
                masa = ("Adhika " if adhika else "") + english_name(MASAS, first["masa"])
                lines.append(f"Masam / paksham: {masa}, {_PAKSHA_EN.get(first.get('paksha', ''), '')}.")
    for kind in ("varjyam", "durmuhurtam", "amrutham"):
        if kind not in wanted:
            continue
        ws = [
            w
            for w in windows
            if w.get("kind") == kind or (kind == "amrutham" and w.get("kind") == "punaramrutham")
        ]
        label = _LABEL_TE[kind] if lang == "te" else _LABEL_EN[kind]
        if not ws:
            lines.append(
                f"{label}: లేదు / పత్రంలో లేదు." if lang == "te" else f"{label}: none printed for this day."
            )
            continue
        if lang == "te":
            spans = [f"{_te_time(w['starts_at'], d)} నుండి {_te_time(w['ends_at'], d)} వరకు" for w in ws]
        else:
            spans = [f"{_en_time(w['starts_at'], d)} to {_en_time(w['ends_at'], d)}" for w in ws]
        lines.append(f"{label}: {'; '.join(spans)}.")
    if "sunrise" in wanted or "sunset" in wanted:
        sr, ss = ev.data.get("sunrise"), ev.data.get("sunset")
        lines.append(
            f"సూర్యోదయం {sr}, సూర్యాస్తమయం {ss} (పత్రంలో ముద్రించినట్లు)."
            if lang == "te"
            else f"Sunrise {sr} AM, sunset {ss} PM (as printed)."
        )
    head = (
        f"{d.day} {_MONTHS_TE[d.month - 1]} {d.year} ({_WEEKDAY_TE[d.weekday()]})"
        if lang == "te"
        else d.strftime("%A, %d %B %Y")
    )
    return f"{head}\n" + "\n".join(lines)


def can_template_occurrences(evidence: list[Evidence]) -> bool:
    occ = [e for e in evidence if e.kind == "occurrence"]
    return bool(occ) and all(e.verified for e in occ)


def render_occurrence_answer(evidence: list[Evidence], lang: str) -> str:
    """'When is the next Ekadashi?' -> the next occurrence(s), exactly as printed in the book."""
    occ = [e for e in evidence if e.kind == "occurrence"]
    lines = []
    for i, ev in enumerate(occ):
        d = date.fromisoformat(ev.date or "")
        m = ev.data.get("match", {})
        kind = str(m.get("kind", ""))
        masa, paksha = m.get("masa"), m.get("paksha")
        if lang == "te":
            head = "తదుపరి" if i == 0 else "ఆ తరువాత"
            ctx = f" ({'అధిక ' if m.get('adhika') else ''}{masa} {paksha} {m['name']})" if masa else ""
            when = f"{d.day} {_MONTHS_TE[d.month - 1]} {d.year} ({_WEEKDAY_TE[d.weekday()]})"
            lines.append(f"{head} {m['name']}: {when}{ctx}, {_te_time(m.get('ends_at'), d)} వరకు.")
        else:
            head = "Next" if i == 0 else "After that"
            name = _en_name(kind, m["name"])
            ctx = ""
            if masa:
                adhika = "Adhika " if m.get("adhika") else ""
                ctx = f" ({adhika}{english_name(MASAS, masa)} {_PAKSHA_EN.get(paksha or '', '')} {name})"
            lines.append(
                f"{head} {name}: {d.strftime('%A, %d %B %Y')}{ctx}, until {_en_time(m.get('ends_at'), d)}."
            )
    return "\n".join(lines)

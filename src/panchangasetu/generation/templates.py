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
# Fixed by weekday; answered from the book's weekday table (structured/weekday_periods.py).
PERIOD_ATTRIBUTES = frozenset({"rahukalam", "yamagandam", "gulika"})
_PERIOD_TE = {"rahukalam": "రాహుకాలం", "yamagandam": "యమగండం", "gulika": "గుళిక కాలం"}
_PERIOD_EN = {"rahukalam": "Rahukalam", "yamagandam": "Yamagandam", "gulika": "Gulika kalam"}
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
    wanted = set(attributes)
    periods = wanted & PERIOD_ATTRIBUTES
    rest = wanted - PERIOD_ATTRIBUTES
    if not wanted or not rest <= TEMPLATE_ATTRIBUTES:
        return False
    days = [e for e in evidence if e.kind == "day"]
    if rest and not (days and all(e.verified for e in days)):
        return False
    found = {e.data.get("period") for e in evidence if e.kind == "weekday_period"}
    return periods <= found


def _head(d: date, lang: str) -> str:
    if lang == "te":
        return f"{d.day} {_MONTHS_TE[d.month - 1]} {d.year} ({_WEEKDAY_TE[d.weekday()]})"
    return d.strftime("%A, %d %B %Y")


def render_day_answers(evidence: list[Evidence], attributes: list[str], lang: str) -> str:
    """Day facts plus weekday-table periods (rahukalam, yamagandam, gulika), one block per date."""
    days: dict[str | None, Evidence] = {e.date: e for e in evidence if e.kind == "day"}
    periods = sorted((e for e in evidence if e.kind == "weekday_period"), key=lambda e: str(e.data["start"]))
    day_attrs = [a for a in attributes if a not in PERIOD_ATTRIBUTES]
    dates = sorted({d for d in days if d and day_attrs} | {p.date for p in periods if p.date})
    blocks = []
    for iso in dates:
        d = date.fromisoformat(iso or "")
        ev = days.get(iso)
        block = render_day_answer(ev, day_attrs, lang) if ev is not None and day_attrs else _head(d, lang)
        for p in periods:
            if p.date == iso:
                start, end = f"{iso}T{p.data['start']}", f"{iso}T{p.data['end']}"
                kind = str(p.data["period"])
                if lang == "te":
                    block += f"\n{_PERIOD_TE[kind]}: {_te_time(start, d)} నుండి {_te_time(end, d)} వరకు."
                else:
                    block += f"\n{_PERIOD_EN[kind]}: {_en_time(start, d)} to {_en_time(end, d)}."
        blocks.append(block)
    if periods:
        page = periods[0].pages[0] if periods[0].pages else ""
        blocks.append(
            f"(పుస్తకంలోని వారపు పట్టిక, పుట {page}: ఉదయం 6 గంటల నుండి లెక్కించినవి; సూర్యోదయ, సూర్యాస్తమయాలను బట్టి కొద్దిగా మారుతాయి.)"
            if lang == "te"
            else f"(From the book's weekday table, p. {page}: counted from 6 AM; the book notes they "
            "shift slightly with the actual sunrise and sunset.)"
        )
    return "\n\n".join(blocks)


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
        # Printed as "6-14"; shown as a clock time.
        sr, ss = (str(ev.data.get(k) or "").replace("-", ":") for k in ("sunrise", "sunset"))
        lines.append(
            f"సూర్యోదయం {sr}, సూర్యాస్తమయం {ss} (పత్రంలో ముద్రించినట్లు)."
            if lang == "te"
            else f"Sunrise {sr} AM, sunset {ss} PM (as printed)."
        )
    return f"{_head(d, lang)}\n" + "\n".join(lines)


def can_template_occurrences(evidence: list[Evidence]) -> bool:
    # Unverified occurrences (wall calendar only) are templated too, with an explicit note: an LLM
    # handed a mix of verified and unverified days has skipped the unverified one (Phase 6 eval).
    return any(e.kind == "occurrence" for e in evidence)


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
            note = " (గోడ క్యాలెండర్ ప్రకారం; పుస్తకంతో ధృవీకరించబడలేదు)" if not ev.verified else ""
            lines.append(f"{head} {m['name']}: {when}{ctx}, {_te_time(m.get('ends_at'), d)} వరకు{note}.")
        else:
            head = "Next" if i == 0 else "After that"
            name = _en_name(kind, m["name"])
            ctx = ""
            if masa:
                adhika = "Adhika " if m.get("adhika") else ""
                ctx = f" ({adhika}{english_name(MASAS, masa)} {_PAKSHA_EN.get(paksha or '', '')} {name})"
            note = " (wall calendar only; not verified against the book)" if not ev.verified else ""
            until = _en_time(m.get("ends_at"), d)
            lines.append(f"{head} {name}: {d.strftime('%A, %d %B %Y')}{ctx}, until {until}{note}.")
    return "\n".join(lines)

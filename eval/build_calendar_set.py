"""Build the held-out calendar set from the database: random verified book days, varied phrasings
in English, Telugu and romanised Telugu, with expected answers taken from the extracted rows.

Usage: uv run python eval/build_calendar_set.py   (writes eval/datasets/heldout_calendar_v1.jsonl)

What it tests: date parsing, routing, retrieval of the right day, and the answer template. It
cannot test extraction accuracy (expected values come from the same extracted rows; extraction is
checked by the printed-digit verification in Phase 2b). Expected next occurrences are computed
here by a plain scan of the rows, independently of the retriever's search code.
"""

from __future__ import annotations

import json
import random
from datetime import date, datetime, timedelta
from pathlib import Path

from sqlalchemy import select

from panchangasetu.config import get_settings
from panchangasetu.retrieval.structured import best_rows
from panchangasetu.storage.db import make_engine, make_session_factory
from panchangasetu.storage.models import Document, DocumentVersion, PanchangaDay
from panchangasetu.structured.weekday_periods import WeekdayPeriods
from panchangasetu.telugu.lexicon import GREGORIAN_MONTH_NUMBER, NAKSHATRAS, TITHIS, YOGAS, english_name

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026_11_07
NOW = "2026-11-07T09:00"
TE_MONTH = {n: m for m, n in reversed(list(GREGORIAN_MONTH_NUMBER.items()))}
TE_WEEKDAY = ("సోమవారం", "మంగళవారం", "బుధవారం", "గురువారం", "శుక్రవారం", "శనివారం", "ఆదివారం")


def hm(iso: str) -> str:
    """'2026-12-15T16:25' -> '4:25' (both the English and Telugu templates print h:mm)."""
    t = datetime.fromisoformat(iso)
    return f"{t.hour % 12 or 12}:{t.minute:02d}"


def en_date(d: date, style: int) -> str:
    return [
        f"{d.day} {d:%B} {d.year}",
        f"{d:%B} {d.day}, {d.year}",
        f"{d.day}/{d.month}/{d.year}",
        f"{d.day}-{d.month:02d}-{d.year}",
    ][style % 4]


def te_date(d: date, style: int) -> str:
    return [f"{d.day} {TE_MONTH[d.month]} {d.year}", f"{d.day}-{d.month:02d}-{d.year}"][style % 2]


def main() -> None:
    rng = random.Random(SEED)
    settings = get_settings()
    factory = make_session_factory(make_engine(settings.database_url))
    with factory() as s:
        raw = s.execute(
            select(PanchangaDay, Document.filename)
            .join(DocumentVersion, DocumentVersion.id == PanchangaDay.version_id)
            .join(Document, Document.id == DocumentVersion.document_id)
            .where(DocumentVersion.is_current.is_(True))
        ).all()
    rows = best_rows(raw)
    book = sorted(
        d
        for d, (r, _) in rows.items()
        if r.source_kind == "book_daily_table" and r.element_count and r.verified_count == r.element_count
    )
    periods = WeekdayPeriods.load(settings.weekday_periods_path)
    assert periods is not None
    out: list[dict[str, object]] = []

    def add(
        kind: str, q: str, lang: str, must: list[list[str]], expect: str = "answer", now: str = NOW
    ) -> None:
        out.append(
            {
                "id": f"cal{len(out) + 1:03d}",
                "kind": kind,
                "lang": lang,
                "q": q,
                "now": now,
                "expect": expect,
                "must": must,
            }
        )

    def first(r: PanchangaDay, kind: str) -> dict | None:  # type: ignore[type-arg]
        return next((e for e in r.elements if e.get("kind") == kind), None)

    # --- day lookups: tithi, nakshatra, yoga
    specs = [("tithi", 40, TITHIS), ("nakshatra", 30, NAKSHATRAS), ("yoga", 15, YOGAS)]
    en_q = {
        "tithi": ["What is the tithi on {d}?", "tithi on {d}", "Which tithi is it on {d}?"],
        "nakshatra": [
            "What is the nakshatram on {d}?",
            "Which star (nakshatra) is on {d}?",
            "nakshatram {d}",
        ],
        "yoga": ["What is the yogam on {d}?", "yoga on {d}"],
    }
    te_q = {
        "tithi": ["{d} న తిథి ఏమిటి?", "{d} తిథి ఏది?"],
        "nakshatra": ["{d} న నక్షత్రం ఏమిటి?", "{d} నక్షత్రం ఏది?"],
        "yoga": ["{d} న యోగం ఏమిటి?"],
    }
    roman_q = {"tithi": "{d} tithi enti?", "nakshatra": "{d} nakshatram enti cheppandi"}
    for kind, n, table in specs:
        for i, d in enumerate(rng.sample(book, n)):
            r = rows[d][0]
            e = first(r, kind)
            if e is None or not e.get("ends_at"):
                continue
            lang = ["en", "en", "te", "te", "roman"][i % 5] if kind in roman_q else ["en", "te"][i % 2]
            if lang == "en":
                q = rng.choice(en_q[kind]).format(d=en_date(d, rng.randrange(4)))
                name = english_name(table, e["name"]) if table else e["name"]
                names = [name, e["name"]]
            elif lang == "te":
                q = rng.choice(te_q[kind]).format(d=te_date(d, rng.randrange(2)))
                names = [e["name"]]
            else:
                q = roman_q[kind].format(d=f"{d.day}/{d.month}/{d.year}")
                names = [e["name"], english_name(table, e["name"]) if table else e["name"]]
            add(f"day_{kind}", q, lang, [names, [hm(e["ends_at"])]])

    # --- varjyam (printed windows)
    with_varjyam = [d for d in book if any(w.get("kind") == "varjyam" for w in rows[d][0].windows)]
    for i, d in enumerate(rng.sample(with_varjyam, 15)):
        w = next(w for w in rows[d][0].windows if w.get("kind") == "varjyam")
        if i % 2:
            add("day_varjyam", f"{te_date(d, i)} న వర్జ్యం ఎప్పుడు?", "te", [[hm(w["starts_at"])]])
        else:
            add("day_varjyam", f"When is varjyam on {en_date(d, i)}?", "en", [[hm(w["starts_at"])]])

    # --- rahukalam / yamagandam (weekday table)
    for i, d in enumerate(rng.sample(book, 15)):
        kind = "rahukalam" if i % 3 else "yamagandam"
        start = periods.evidence(kind, d).data["start"]
        h, m = (int(x) for x in str(start).split(":"))
        expected = f"{h % 12 or 12}:{m:02d}"
        if i % 2:
            word = "రాహుకాలం" if kind == "rahukalam" else "యమగండం"
            add(f"day_{kind}", f"{te_date(d, i)} {word} ఎప్పుడు?", "te", [[expected]])
        else:
            add(f"day_{kind}", f"{kind} on {en_date(d, i)}?", "en", [[expected]])

    # --- next occurrence: independent scan of the rows from a random "now"
    def next_day(kind: str, name: str, after: datetime) -> date | None:
        last: date | None = None
        for d in sorted(rows):
            if d < after.date() - timedelta(days=1):
                continue
            for e in rows[d][0].elements:
                if e.get("kind") == kind and e.get("name") == name:
                    ends = e.get("ends_at")
                    if ends and datetime.fromisoformat(ends) < after:
                        continue
                    if last is None:
                        return d
        return last

    finds = [("tithi", n) for n in ("ఏకాదశి", "పూర్ణిమ", "అమావాస్య", "చవితి", "అష్టమి", "ద్వాదశి", "పంచమి")]
    finds += [("nakshatra", n) for n in ("రోహిణి", "శ్రవణం", "పుష్యమి", "అశ్విని", "మృగశిర")]
    candidates = [d for d in book if d <= book[-1] - timedelta(days=60)]
    for i in range(35):
        kind, name = finds[i % len(finds)]
        now = datetime.combine(rng.choice(candidates), datetime.min.time()).replace(hour=9)
        d = next_day(kind, name, now)
        if d is None:
            continue
        table = TITHIS if kind == "tithi" else NAKSHATRAS
        aliases = [a for a in table[name] if a.isascii()]
        if i % 2 and aliases:
            q = rng.choice([f"When is the next {aliases[0].title()}?", f"next {aliases[0]} date?"])
            add(
                f"find_{kind}",
                q,
                "en",
                [[f"{d.day} {d:%B} {d.year}", f"{d.day:02d} {d:%B} {d.year}"]],
                now=now.isoformat(timespec="minutes"),
            )
        else:
            q = rng.choice([f"వచ్చే {name} ఎప్పుడు?", f"తదుపరి {name} ఏ రోజు?"])
            months = [m for m, n in GREGORIAN_MONTH_NUMBER.items() if n == d.month]  # ఆగస్టు / ఆగష్టు
            add(
                f"find_{kind}",
                q,
                "te",
                [[f"{d.day} {m} {d.year}" for m in months]],
                now=now.isoformat(timespec="minutes"),
            )

    # --- outside the documents: must abstain
    for i, d in enumerate(
        [date(2025, 2, 3), date(2028, 6, 1), date(2029, 1, 14), date(2030, 8, 15), date(2024, 12, 25)] * 2
    ):
        if i % 2:
            add("out_of_range", f"{te_date(d, i)} న తిథి ఏమిటి?", "te", [], expect="abstain")
        else:
            add("out_of_range", f"What is the nakshatram on {en_date(d, i)}?", "en", [], expect="abstain")

    path = ROOT / "eval/datasets/heldout_calendar_v1.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    kinds: dict[str, int] = {}
    for r in out:
        kinds[str(r["kind"])] = kinds.get(str(r["kind"]), 0) + 1
    print(f"wrote {len(out)} questions to {path.relative_to(ROOT)}: {kinds}")


if __name__ == "__main__":
    main()

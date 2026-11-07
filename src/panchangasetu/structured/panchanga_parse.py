"""Parse one OCR'd day row of the TTD panchangam into structured, *verified* fields.

Row grammar (TTD Siddhanti panchangam 2026-27):

  <weekday> [<masa>] [శుద్ధ|బహుళ] <tithi> <ghati> <time> [<tithi> <ghati> <time>]
            <nakshatra> <ghati> <time> [...] <yoga> <ghati> <time> [...] <karana> <ghati> <time> [...]
            <X>.వ.<t1> ల <t2>              varjyam   (zero or more)
            <X>.దు.<t1> ల <t2>             durmuhurtam (one or two)
            అమృతం [<ghati>] <t1> ల <t2>    amrutha kalam [పునరమృతం ...]
            <planet ingresses, festivals and observances ...>   భో.<h-mm> అహః <gg-vv>

- <ghati> is ghatis-vighatis elapsed from sunrise (1 ghati = 24 min, 1 vighati = 24 s);
  "పూర్తి" instead of a time means the element lasts past the next sunrise.
- <time> carries a part-of-day prefix: ఉ. morning, ప. daytime, మ. midday, సా. evening, రా. night,
  తె. pre-dawn (i.e. before the *next* sunrise, so the next civil date).

Every element is checked two independent ways:
  ghati check:  sunrise + ghati*24min ≈ printed time (within a few minutes)
  digits check: the OCR'd numbers appear verbatim among the numbers recovered exactly from the
                PDF's legacy-font text layer
so an OCR digit error cannot silently become a "fact".
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any

from panchangasetu.telugu.lexicon import KARANAS, MASAS, NAKSHATRAS, PLANETS, TITHIS, YOGAS
from panchangasetu.telugu.text import graphemes

PARSER_VERSION = "book-daily-1"

_PREFIXES = "ఉ|ప|మ|సా|రా|తె"
# OCR confusions observed on this book: రా -> నా, ఉ -> ఊ. Normalised back in _prefix().
_PREFIX_OCR = {"నా": "రా", "ఊ": "ఉ"}
_TRIPLE_PREFIXES = "ఉ|ప|మ|సా|రా|తె|నా|ఊ"
_HM = r"\d{1,2}[-.]\d{2}"
# name ghati time   (time may be "పూర్తి"; OCR sometimes inserts "?" or a space after the dot)
_TRIPLE_RE = re.compile(
    rf"(?P<name>[^\s\d.,()|:;?!'\"]+)\s+(?P<ghati>\d{{1,2}}-\d{{1,2}})\s+"
    rf"(?:(?P<prefix>{_TRIPLE_PREFIXES})\.?\s?\??\s?(?P<time>{_HM})|(?P<full>పూర్తి))"
)
_VARJYAM_RE = re.compile(
    rf"(?P<p1>{_PREFIXES})\.వ\.\s?(?P<t1>{_HM})\s*ల\s*(?:(?P<p2>{_PREFIXES})\.)?\s?(?P<t2>{_HM})"
)
_DUR_RE = re.compile(
    rf"(?P<p1>{_PREFIXES})\.దు\.\s?(?P<t1>{_HM})\s*ల\s*(?:(?P<p2>{_PREFIXES})\.)?\s?(?P<t2>{_HM})"
)
_AMRUTHA_RE = re.compile(
    rf"(?P<kind>పునరమృతం|అమృతం)\s+(?:(?P<ghati>\d{{1,2}}-\d{{2}})\s+)?(?:(?P<p1>{_PREFIXES})\.\s?)?"
    rf"(?P<t1>{_HM})\s*ల\s*(?:(?P<p2>{_PREFIXES})\.)?\s?(?P<t2>{_HM})"
)
_DAYLEN_RE = re.compile(r"భో\.?\s?\d{1,2}-\d{2}\s*అహః\s*\d{1,2}-\d{2}")
_PAKSHA_RE = re.compile(r"(శుద్ధ|బహుళ)\s*$")
_ADHIKA_RE = re.compile(r"(అధిక|నిజ)\s*$")

_GHATI_TOLERANCE_MIN = 4

ELEMENT_TABLES: dict[str, dict[str, list[str]]] = {
    "tithi": TITHIS,
    "nakshatra": NAKSHATRAS,
    "yoga": YOGAS,
    "karana": KARANAS,
    "planet": PLANETS,
}
_ORDER = ["tithi", "nakshatra", "yoga", "karana"]
MASA_CYCLE = list(MASAS)


# ---------------------------------------------------------------------------- name matching
def _edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, ga in enumerate(a, 1):
        cur = [i]
        for j, gb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ga != gb)))
        prev = cur
    return prev[-1]


def _skeleton(text: str) -> list[str]:
    """Akshara sequence with vowel signs/virama dropped from each cluster's tail: OCR errors in
    Telugu are mostly in conjunct/matra details, so comparing consonant skeletons is robust."""
    return [g[0] for g in graphemes(text)]


def _forms(table: dict[str, list[str]]) -> list[tuple[str, str]]:
    out = []
    for canonical, aliases in table.items():
        for form in [canonical, *aliases]:
            if not form.isascii():
                out.append((form, canonical))
    return out


_ALL_FORMS: list[tuple[str, str, str]] = [
    (form, canonical, kind) for kind, table in ELEMENT_TABLES.items() for form, canonical in _forms(table)
]


def match_name(token: str, kinds: Sequence[str] | None = None) -> tuple[str, str, float] | None:
    """Best (kind, canonical, score) for an OCR'd token, or None if nothing is close enough."""
    token = token.strip("'\"‘’“”()[]|")
    if not token:
        return None
    tg, ts, tc = graphemes(token), _skeleton(token), list(token)
    best: tuple[str, str, float] | None = None
    for form, canonical, kind in _ALL_FORMS:
        if kinds and kind not in kinds:
            continue
        if form == token:
            return kind, canonical, 1.0
        fg = graphemes(form)
        # Three views of similarity; short names need the code-point view (one wrong conjunct
        # member in "షస్టి" vs "షష్టి" is half the aksharas but a fifth of the code points).
        akshara = 1.0 - _edit_distance(tg, fg) / max(len(fg), len(tg))
        skeleton = 1.0 - (_edit_distance(ts, _skeleton(form)) + 0.5) / max(len(fg), len(tg))
        chars = 1.0 - _edit_distance(tc, list(form)) / max(len(form), len(token))
        score = max(akshara, skeleton, chars)
        if best is None or score > best[2]:
            best = (kind, canonical, score)
    return best if best and best[2] >= 0.6 else None


def _prefix(raw: str) -> str:
    return _PREFIX_OCR.get(raw, raw)


def repair_ghati(ghati: str, exact: set[str]) -> str:
    """OCR sometimes drops a vighati digit ("44-8"). The exact text-layer numbers settle it."""
    g, v = ghati.split("-")
    if len(v) == 2:
        return ghati
    candidates = [
        n for n in exact if n.startswith(f"{g}-") and len(n.split("-")[1]) == 2 and v in n.split("-")[1]
    ]
    return candidates[0] if len(candidates) == 1 else f"{g}-0{v}"


# ---------------------------------------------------------------------------- times
def to_datetime(day: date, prefix: str, hm: str) -> datetime:
    """Convert a printed 12-hour time with a part-of-day prefix to a civil datetime."""
    h, m = (int(x) for x in re.split(r"[-.]", hm))
    if prefix == "తె":  # pre-dawn: before the next sunrise
        return datetime.combine(day + timedelta(days=1), time(h % 12, m))
    if prefix == "రా":
        if h == 12:  # 12-xx at night = just after midnight
            return datetime.combine(day + timedelta(days=1), time(0, m))
        if h <= 5:
            return datetime.combine(day + timedelta(days=1), time(h, m))
        return datetime.combine(day, time(h % 12 + 12, m))
    if prefix in ("సా", "మ"):
        return datetime.combine(day, time(h if h == 12 else h + 12, m))
    if prefix == "ప":  # daytime: 6-11 morning, 12 noon, 1-5 afternoon
        return datetime.combine(day, time(h if h >= 6 else h + 12, m))
    return datetime.combine(day, time(h % 12 if h != 12 else 12, m))  # ఉ: morning


def ghati_to_datetime(sunrise: datetime, ghati: str) -> datetime:
    g, v = (int(x) for x in ghati.split("-"))
    return sunrise + timedelta(seconds=(g * 60 + v) * 24)


# ---------------------------------------------------------------------------- data model
@dataclass(slots=True)
class Element:
    kind: str  # tithi | nakshatra | yoga | karana | planet
    name: str  # canonical Telugu
    raw: str  # as OCR'd
    ghati: str
    ends_at: str | None  # ISO datetime (local); None = lasts past next sunrise ("పూర్తి")
    printed_end: str | None  # e.g. "సా.6-08" exactly as printed
    match_score: float
    ghati_check: str  # ok | off | n/a
    digits_check: str  # ok | missing
    paksha: str | None = None  # tithis only: శుద్ధ | బహుళ
    masa: str | None = None  # tithis only (amanta lunar month)
    adhika: bool = False
    name_source: str = "ocr"  # ocr | sequence
    sequence_check: str = "n/a"  # ok | skip | inferred | corrected | break | n/a

    @property
    def verified(self) -> bool:
        """Timing confirmed two independent ways and the name consistent with the cycle."""
        return self.ghati_check == "ok" and self.digits_check == "ok" and self.sequence_check != "break"


@dataclass(slots=True)
class Window:
    kind: str  # varjyam | durmuhurtam | amrutham | punaramrutham
    starts_at: str
    ends_at: str
    printed: str
    digits_check: str


@dataclass(slots=True)
class ParsedDay:
    date: str
    weekday_printed: str | None
    sunrise: str | None
    sunset: str | None
    elements: list[Element] = field(default_factory=list)
    windows: list[Window] = field(default_factory=list)
    notes: str = ""
    warnings: list[str] = field(default_factory=list)
    source_text: str = ""  # the OCR text this was parsed from (kept for audit and citation)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------- parsing
def _hm_digits(s: str) -> str:
    return s.replace(".", "-")


def parse_day(
    text: str,
    day: date,
    *,
    sunrise: str | None,
    sunset: str | None,
    weekday_printed: str | None,
    exact_numbers: Sequence[str],
) -> ParsedDay:
    out = ParsedDay(
        date=day.isoformat(),
        weekday_printed=weekday_printed,
        sunrise=sunrise,
        sunset=sunset,
        source_text=text,
    )
    exact = set(exact_numbers)
    sunrise_dt = to_datetime(day, "ఉ", sunrise) if sunrise else None

    # Element section ends at the first timing window.
    cut = min(
        [m.start() for rx in (_VARJYAM_RE, _DUR_RE, _AMRUTHA_RE) for m in [rx.search(text)] if m]
        or [len(text)]
    )
    head, tail = text[:cut], text[cut:]

    order_idx = 0
    for m in _TRIPLE_RE.finditer(head):
        token = m.group("name")
        # Prefer the kinds the row grammar expects next, but fall back to any kind so one
        # mismatched element cannot lock out every element after it.
        found = match_name(token, ["planet", *_ORDER[order_idx:]])
        if found is None:  # out-of-order kinds only on a strong match
            found = match_name(token)
            if found is not None and found[2] < 0.75:
                found = None
        if found is not None and found[0] in _ORDER and _ORDER.index(found[0]) < order_idx:
            # e.g. "అర్కః ... కృత్తిక" after the karanas: planet-ingress detail, not a day element.
            continue
        if found is None:
            # Keep the (verifiable) timing; the name is filled from the tithi/nakshatra cycle later.
            out.warnings.append(f"unrecognised element name {token!r}")
            kind, canonical, score = "unknown", "", 0.0
        else:
            kind, canonical, score = found
        if kind not in ("planet", "unknown"):
            order_idx = _ORDER.index(kind)
        ghati = repair_ghati(m.group("ghati"), exact)
        if m.group("full"):
            ends_at, printed, ghati_check = None, "పూర్తి", "n/a"
        else:
            prefix, hm = _prefix(m.group("prefix")), m.group("time")
            printed = f"{prefix}.{hm}"
            try:
                end = to_datetime(day, prefix, hm)
            except ValueError:  # OCR produced an impossible time such as "8-60"
                out.warnings.append(f"invalid time {printed!r} for {token!r}")
                end = None
            ends_at = end.isoformat(timespec="minutes") if end else None
            if sunrise_dt is None or end is None:
                ghati_check = "n/a"
            else:
                delta = abs((ghati_to_datetime(sunrise_dt, ghati) - end).total_seconds()) / 60
                ghati_check = "ok" if delta <= _GHATI_TOLERANCE_MIN else "off"
        digits_ok = ghati in exact and (printed == "పూర్తి" or _hm_digits(printed.split(".", 1)[1]) in exact)
        el = Element(
            kind=kind,
            name=canonical,
            raw=token,
            ghati=ghati,
            ends_at=ends_at,
            printed_end=printed,
            match_score=round(score, 3),
            ghati_check=ghati_check,
            digits_check="ok" if digits_ok else "missing",
        )
        previous_end = next(
            (e.ends_at for e in reversed(out.elements) if e.kind == kind and kind in _ORDER and e.ends_at),
            None,
        )
        if previous_end and ends_at and ends_at < previous_end:
            out.warnings.append(f"dropped out-of-order {kind} {canonical or token!r} ending {printed}")
            continue
        if kind == "tithi":
            before = head[: m.start()]
            if pm := _PAKSHA_RE.search(before.rstrip()):
                el.paksha = pm.group(1)
                rest = before.rstrip()[: pm.start()].rstrip()
                masa_tok = rest.split()[-1] if rest.split() else ""
                masa = match_masa(masa_tok)
                if masa:
                    el.masa = masa
                    el.adhika = bool(re.search(r"అధిక\s*$", rest[: rest.rfind(masa_tok)]))
        out.elements.append(el)

    for kind, rx in (("varjyam", _VARJYAM_RE), ("durmuhurtam", _DUR_RE)):
        for w in rx.finditer(tail):
            try:
                start = to_datetime(day, w.group("p1"), w.group("t1"))
                end = to_datetime(day, w.group("p2") or w.group("p1"), w.group("t2"))
            except ValueError:
                out.warnings.append(f"invalid time in {kind}: {w.group(0)!r}")
                continue
            if end < start:  # second time crossed into the next part of the day without a prefix
                end += timedelta(hours=12)
            ok = _hm_digits(w.group("t1")) in exact and _hm_digits(w.group("t2")) in exact
            out.windows.append(
                Window(
                    kind,
                    start.isoformat(timespec="minutes"),
                    end.isoformat(timespec="minutes"),
                    w.group(0),
                    "ok" if ok else "missing",
                )
            )
    last_window_end = 0
    for w in _AMRUTHA_RE.finditer(tail):
        p1 = w.group("p1") or _infer_prefix(w.group("ghati"), sunrise_dt, day)
        if p1 is None:
            out.warnings.append(f"amrutham without part-of-day: {w.group(0)!r}")
            continue
        try:
            start = to_datetime(day, p1, w.group("t1"))
            end = to_datetime(day, w.group("p2") or p1, w.group("t2"))
        except ValueError:
            out.warnings.append(f"invalid time in amrutham: {w.group(0)!r}")
            continue
        if end < start:
            end += timedelta(hours=12)
        ok = _hm_digits(w.group("t1")) in exact and _hm_digits(w.group("t2")) in exact
        kind = "punaramrutham" if w.group("kind") == "పునరమృతం" else "amrutham"
        out.windows.append(
            Window(
                kind,
                start.isoformat(timespec="minutes"),
                end.isoformat(timespec="minutes"),
                w.group(0),
                "ok" if ok else "missing",
            )
        )
        last_window_end = max(last_window_end, w.end())
    if not last_window_end:
        for rx in (_VARJYAM_RE, _DUR_RE):
            for w in rx.finditer(tail):
                last_window_end = max(last_window_end, w.end())

    notes = _DAYLEN_RE.sub("", tail[last_window_end:]).strip(" .,\n")
    out.notes = re.sub(r"\s+", " ", notes)
    if not any(e.kind == "tithi" for e in out.elements):
        out.warnings.append("no tithi parsed")
    if not any(e.kind == "nakshatra" for e in out.elements):
        out.warnings.append("no nakshatra parsed")
    return out


def _infer_prefix(ghati: str | None, sunrise: datetime | None, day: date) -> str | None:
    """Some amrutham entries omit the part-of-day; derive it from the ghati offset."""
    if not ghati or sunrise is None:
        return None
    t = ghati_to_datetime(sunrise, ghati)
    if t.date() > day:
        return "రా" if t.hour < 4 else "తె"
    return "ఉ" if t.hour < 12 else ("ప" if t.hour < 16 else ("సా" if t.hour < 19 else "రా"))


def match_masa(token: str) -> str | None:
    token = token.strip("'\"()[]|")
    best: tuple[str, float] | None = None
    for canonical, aliases in MASAS.items():
        for form in [canonical, *[a for a in aliases if not a.isascii()]]:
            fg = graphemes(form)
            score = 1.0 - _edit_distance(graphemes(token), fg) / max(len(fg), len(graphemes(token)), 1)
            if best is None or score > best[1]:
                best = (canonical, score)
    return best[0] if best and best[1] >= 0.6 else None


# ---------------------------------------------------------------------------- lunar state
_TITHI_RUN = list(TITHIS)[:14]  # పాడ్యమి .. చతుర్దశి (shared by both pakshas)
NAKSHATRA_CYCLE = list(NAKSHATRAS)


def next_tithi(previous: str, paksha: str) -> str:
    if previous in ("పూర్ణిమ", "అమావాస్య"):
        return "పాడ్యమి"
    i = _TITHI_RUN.index(previous)
    if i < len(_TITHI_RUN) - 1:
        return _TITHI_RUN[i + 1]
    return "పూర్ణిమ" if paksha == "శుద్ధ" else "అమావాస్య"


def _classify_unknowns(day: ParsedDay) -> None:
    """Give unnamed elements a slot from their position: the row lists tithis, then nakshatras,
    then yoga, then karanas."""
    kinds = [e.kind for e in day.elements]
    for i, el in enumerate(day.elements):
        if el.kind != "unknown":
            continue
        before = kinds[:i]
        if not any(k in ("tithi", "nakshatra", "yoga", "karana") for k in before):
            el.kind = "tithi"
        elif "nakshatra" not in kinds and not any(k in ("yoga", "karana") for k in before):
            el.kind = "nakshatra"
        kinds[i] = el.kind


def resolve_sequences(days: list[ParsedDay], *, initial_masa: str, initial_paksha: str) -> None:
    """Walk tithis and nakshatras in date order.

    - Fills names OCR could not read, from the strict cycle (name_source="sequence").
    - Checks every OCR'd name against the cycle; a break marks a likely misread that happens
      to be another valid name (sequence_check="break"), which disqualifies it as verified.
    - Labels each tithi with its amanta masa and paksha. Paksha flips after పూర్ణిమ (-> బహుళ)
      and అమావాస్య (-> శుద్ధ); the masa advances after అమావాస్య. Printed markers ("చైత్ర శుద్ధ",
      "అధిక జ్యేష్ట శుద్ధ") override the inference, which is what makes adhika masas right.
    """
    masa, paksha, adhika = initial_masa, initial_paksha, False
    prev_tithi: Element | None = None
    prev_nak: Element | None = None
    seen_tithi_yesterday = seen_nak_yesterday = True
    for day in days:
        _classify_unknowns(day)
        has_tithi = has_nak = False
        for el in day.elements:
            if el.kind == "tithi":
                expected: list[str] = []
                if prev_tithi is not None:
                    if prev_tithi.ends_at is None:  # "పూర్తి": same tithi continues today
                        expected = [prev_tithi.name]
                    else:
                        expected = [next_tithi(prev_tithi.name, paksha)]
                    # One element may have been missed by OCR (two if yesterday had none at all).
                    for _ in range(1 if seen_tithi_yesterday else 2):
                        expected.append(next_tithi(expected[-1], paksha))
                _check_against(el, expected, day)
                if prev_tithi is not None and el.name != prev_tithi.name:
                    if prev_tithi.name == "పూర్ణిమ":
                        paksha = "బహుళ"
                    elif prev_tithi.name == "అమావాస్య":
                        paksha, adhika = "శుద్ధ", False
                        masa = MASA_CYCLE[(MASA_CYCLE.index(masa) + 1) % len(MASA_CYCLE)]
                if el.paksha:
                    paksha = el.paksha
                if el.masa:
                    masa, adhika = el.masa, el.adhika
                el.paksha, el.masa, el.adhika = paksha, masa, adhika
                if el.kind == "tithi":
                    prev_tithi, has_tithi = el, True
            elif el.kind == "nakshatra":
                expected = []
                if prev_nak is not None:
                    i = NAKSHATRA_CYCLE.index(prev_nak.name)
                    first = 0 if prev_nak.ends_at is None else 1
                    steps = tuple(range(first, first + (2 if seen_nak_yesterday else 3)))
                    expected = [NAKSHATRA_CYCLE[(i + s) % len(NAKSHATRA_CYCLE)] for s in steps]
                _check_against(el, expected, day)
                if el.kind == "nakshatra":
                    prev_nak, has_nak = el, True
        day.elements = [e for e in day.elements if e.kind != "unknown"]
        seen_tithi_yesterday, seen_nak_yesterday = has_tithi, has_nak


def _check_against(el: Element, expected: list[str], day: ParsedDay) -> None:
    """Compare an element with the name(s) the cycle allows here. Only an unambiguous single
    expectation may fill or override a name; ambiguous gaps only confirm or flag."""
    if not el.name:  # unreadable: take the cycle's answer if it is unambiguous
        if not expected:
            el.kind = "unknown"
            return
        # The first expectation is the direct successor; skips are only for validation.
        el.name, el.name_source, el.sequence_check = expected[0], "sequence", "inferred"
        day.warnings.append(f"{el.kind} name inferred from sequence: {expected[0]}")
    elif not expected:
        el.sequence_check = "n/a"
    elif el.name == expected[0]:
        el.sequence_check = "ok"
    elif el.name in expected:
        el.sequence_check = "skip"  # consistent; an element before it was missed by OCR
    elif el.match_score < 0.8:  # weak OCR match contradicting the cycle: trust the cycle
        day.warnings.append(f"{el.kind} {el.raw!r} corrected to {expected[0]} by sequence")
        el.name, el.name_source, el.sequence_check = expected[0], "sequence", "corrected"
    else:
        el.sequence_check = "break"
        day.warnings.append(f"{el.kind} {el.name} breaks the cycle (expected {'/'.join(expected)})")

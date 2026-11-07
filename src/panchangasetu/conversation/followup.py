"""Follow-up questions -> standalone questions, so retrieval and answering stay single-turn.

"And tomorrow?" after "What is today's tithi?" becomes "What is tomorrow's tithi?". Rules handle
the calendar-shaped follow-ups for free, by editing the previous standalone question (its "frame"):

  dates only         "and tomorrow?" / "రేపు?"          -> previous frame, dates swapped
  attributes only    "and the nakshatram?"              -> previous frame, attribute swapped
  entity only        "and Purnima?" / "కన్యా రాశికి?"   -> previous frame, tithi/rasi/festival swapped
  "it" / "దాని"      "when is the next one?"            -> pronoun replaced by the previous entity
  "that day"         "ఆ రోజు నక్షత్రం?"                  -> replaced by the previous (or answered) date

Anything the rules cannot place (e.g. "what are its remedies?" after a long prose question) goes to
a small LLM rewrite when one is configured; otherwise the question is used as typed. A question
that names its own subject is never rewritten.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime

from panchangasetu.conversation.store import TurnRecord
from panchangasetu.core.interfaces import LLM, ChatMessage, LLMError
from panchangasetu.observability.logging import get_logger
from panchangasetu.query.dates import DateSpec, parse_dates
from panchangasetu.query.language import detect_language
from panchangasetu.query.slots import ATTRIBUTES, Slots, extract_slots
from panchangasetu.telugu.lexicon import GREGORIAN_MONTH_NUMBER, NAKSHATRAS, RASIS, TITHIS

log = get_logger(__name__)

Span = tuple[int, int]

# Pronouns that stand for the previous subject. Telugu case forms keep their suffix.
_ENTITY_PRONOUNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:that|this) (?:festival|day's festival|tithi|nakshatram|one)\b", re.I), "{}"),
    (re.compile(r"(?<=\bnext )one\b|(?<=\bthe one )after\b", re.I), "{}"),
    (re.compile(r"\bits\b", re.I), "{}'s"),
    (re.compile(r"\bit\b", re.I), "{}"),
    (re.compile(r"దానికి"), "{}కి"),
    (re.compile(r"దాన్ని"), "{}ని"),
    (re.compile(r"దాని|దీని|అదే|అది|ఇది|ఆ పండుగ"), "{}"),
    (re.compile(r"\b(?:adi|daani|dani)\b", re.I), "{}"),
]
_DAY_PRONOUNS = re.compile(
    r"\b(?:that|the same) (?:day|date)\b|\bthen\b|అదే రోజు(?:న)?|ఆ ?రోజు(?:న)?|అప్పుడు|\baa ?roju\b", re.I
)
_LEAD = re.compile(r"^\s*(?:and|also|what about|how about|mari|inka|మరి|ఇంకా)\b", re.I)
# Words that carry no subject of their own in a follow-up.
_FILLER = frozenset(
    "one it its that this same then about also how they them mari inka adi daani dani "
    "మరి ఇంకా అది దాని దానికి దాన్ని దీని ఇది అదే".split()
)
_TE_MONTHS = {n: name for name, n in reversed(list(GREGORIAN_MONTH_NUMBER.items()))}


@dataclass(frozen=True, slots=True)
class Resolution:
    question: str  # the standalone question to answer
    method: str  # none | rule | llm
    reason: str = ""


@dataclass(slots=True)
class _Parts:
    text: str
    dates: list[DateSpec]
    date_spans: list[Span]
    slots: Slots
    attr_spans: list[Span]
    entity: tuple[str, Span] | None  # (kind, span) kind: rasi | tithi | nakshatra | festival | topic
    topic: list[str]  # content words outside dates/entity/fillers
    entity_pronouns: list[tuple[Span, str]] = field(default_factory=list)
    day_pronouns: list[Span] = field(default_factory=list)
    lead: bool = False

    def surface(self, span: Span) -> str:
        return self.text[span[0] : span[1]]

    @property
    def entity_text(self) -> str | None:
        return self.surface(self.entity[1]) if self.entity else None

    @property
    def has_subject(self) -> bool:
        return bool(self.dates or self.slots.attributes or self.entity or self.topic)


def _find(text: str, forms: Sequence[str]) -> Span | None:
    low = text.lower()
    for form in sorted({f.lower() for f in forms if len(f) >= 3}, key=len, reverse=True):
        if form.isascii():
            m = re.search(rf"(?<![a-z]){re.escape(form)}(?:s|'s)?(?![a-z])", low)
        else:
            m = re.search(re.escape(form), low)
        if m:
            return m.start(), m.end()
    return None


def _tokens(text: str) -> list[tuple[str, Span]]:
    return [(m.group(0).lower(), m.span()) for m in re.finditer(r"[A-Za-z]+|[ఀ-౿‌‍]+", text)]


def _overlaps(span: Span, spans: Sequence[Span]) -> bool:
    return any(span[0] < b and a < span[1] for a, b in spans)


def _date_spans(text: str, dates: list[DateSpec]) -> list[Span]:
    spans = []
    for d in dates:
        i = text.find(d.matched)
        if i >= 0:
            spans.append((i, i + len(d.matched)))
    return spans


def _entity(text: str, slots: Slots, residual: list[tuple[str, Span]]) -> tuple[str, Span] | None:
    if slots.rasi:
        forms = [slots.rasi, *RASIS[slots.rasi]]
        te = re.search(rf"{re.escape(slots.rasi)}\s*రాశి", text)
        if te:
            return "rasi", te.span()
        en = _find(text, [f"{f} rasi" for f in forms] + [f"{f} rashi" for f in forms] + forms)
        if en:
            return "rasi", en
    named = slots.tithi or slots.nakshatra
    if named:
        table = TITHIS if slots.tithi else NAKSHATRAS
        span = _find(text, [named, *table[named]])
        if span:
            if residual and len(residual) <= 3:  # "Sri Rama Navami", "రథసప్తమి": a festival
                spans = [span, *(s for _, s in residual)]
                return "festival", (min(a for a, _ in spans), max(b for _, b in spans))
            return ("tithi" if slots.tithi else "nakshatra"), span
    if residual and len(residual) <= 3:
        spans = [s for _, s in residual]
        kind = "festival" if (slots.when or "festival" in slots.attributes) else "topic"
        return kind, (min(a for a, _ in spans), max(b for _, b in spans))
    return None


def analyse(text: str, today: date) -> _Parts:
    slots = extract_slots(text)
    dates = parse_dates(text, today)
    date_spans = _date_spans(text, dates)
    attr_spans = [s for a in slots.attributes if (s := _find(text, ATTRIBUTES[a]))]
    pronouns = [(m.span(), fmt) for rx, fmt in _ENTITY_PRONOUNS for m in rx.finditer(text)]
    day_pronouns = [m.span() for m in _DAY_PRONOUNS.finditer(text)]
    taken = [*date_spans, *attr_spans, *(s for s, _ in pronouns), *day_pronouns]
    residual_set = set(slots.residual)
    residual = [
        (tok, span)
        for tok, span in _tokens(text)
        if tok in residual_set and tok not in _FILLER and not _overlaps(span, taken)
    ]
    entity = _entity(text, slots, residual)
    if pronouns and entity and entity[0] in ("festival", "topic"):
        entity = None  # "why is it important?": "important" is not a subject, "it" stands for one
    topic = [tok for tok, span in residual if not (entity and _overlaps(span, [entity[1]]))]
    if entity and entity[0] in ("festival", "topic"):
        topic = []
    return _Parts(
        text=text,
        dates=dates,
        date_spans=date_spans,
        slots=slots,
        attr_spans=attr_spans,
        entity=entity,
        topic=topic,
        entity_pronouns=_dedupe_spans(pronouns),
        day_pronouns=day_pronouns,
        lead=bool(_LEAD.search(text)),
    )


def _dedupe_spans(items: list[tuple[Span, str]]) -> list[tuple[Span, str]]:
    out: list[tuple[Span, str]] = []
    for span, fmt in sorted(items, key=lambda x: (x[0][0], -(x[0][1] - x[0][0]))):
        if not _overlaps(span, [s for s, _ in out]):
            out.append((span, fmt))
    return out


def _replace(text: str, edits: list[tuple[Span, str]]) -> str:
    for (a, b), new in sorted(edits, key=lambda e: e[0][0], reverse=True):
        text = text[:a] + new + text[b:]
    return re.sub(r"\s{2,}", " ", text).strip()


def _strip_lead(text: str) -> str:
    return _LEAD.sub("", text).strip(" ,?")


def format_date(d: date, lang: str) -> str:
    return f"{d.day} {_TE_MONTHS[d.month]} {d.year}" if lang == "te" else f"{d.day} {d:%B} {d.year}"


def _previous_day(prev: _Parts, record: TurnRecord, lang: str) -> str | None:
    """The day the previous turn was about: its own date, else the date its answer cited."""
    if prev.dates and prev.date_spans:
        spec = prev.dates[0]
        return format_date(spec.start, lang) if spec.is_day else prev.surface(prev.date_spans[0])
    for iso in record.cited_dates:
        try:
            return format_date(date.fromisoformat(iso[:10]), lang)
        except ValueError:
            continue
    return None


def resolve_by_rules(question: str, history: Sequence[TurnRecord], today: date) -> Resolution:
    if not history:
        return Resolution(question, "none")
    record = history[-1]
    cur = analyse(question, today)
    prev = analyse(record.standalone_question, record.asked_at.date())
    lang = detect_language(question).code

    edits: list[tuple[Span, str]] = []
    reasons: list[str] = []
    if cur.entity_pronouns and prev.entity and not cur.entity:
        assert prev.entity_text is not None
        edits += [(span, fmt.format(prev.entity_text)) for span, fmt in cur.entity_pronouns]
        reasons.append(f"pronoun -> {prev.entity_text}")
    if cur.day_pronouns and not cur.dates:
        day = _previous_day(prev, record, lang)
        if day:
            for a, b in cur.day_pronouns:
                # English needs a preposition: "nakshatram that day" -> "nakshatram on 20 November 2026".
                needs_on = lang == "en" and not re.search(r"\b(?:on|of|for)\s*$", question[:a], re.I)
                edits.append(((a, b), f"on {day}" if needs_on else day))
            reasons.append(f"that day -> {day}")
    if edits:
        return Resolution(_replace(question, edits), "rule", "; ".join(reasons))

    only_dates = cur.dates and not (cur.slots.attributes or cur.entity or cur.topic)
    only_attrs = cur.slots.attributes and not (cur.dates or cur.entity or cur.topic or cur.slots.concept)
    only_entity = (
        cur.entity and cur.entity[0] != "topic" and not (cur.dates or cur.slots.attributes or cur.topic)
    )
    asks_itself = cur.slots.when or cur.slots.concept or cur.slots.weak_concept

    if only_dates and (prev.slots.attributes or prev.entity):
        new = " ".join(cur.surface(s) for s in cur.date_spans) or _strip_lead(question)
        if prev.date_spans:
            text = _replace(prev.text, [(prev.date_spans[0], new), *((s, "") for s in prev.date_spans[1:])])
        else:
            text = f"{prev.text.rstrip(' ?')} {new}?"
        return Resolution(text, "rule", f"dates -> {new}")

    if only_attrs and (prev.dates or record.cited_dates):
        new = (
            cur.surface((cur.attr_spans[0][0], cur.attr_spans[-1][1]))
            if cur.attr_spans
            else _strip_lead(question)
        )
        if prev.dates and prev.attr_spans:
            text = _replace(prev.text, [(prev.attr_spans[0], new), *((s, "") for s in prev.attr_spans[1:])])
        else:
            day = _previous_day(prev, record, lang)
            if day is None:
                return Resolution(question, "none")
            text = f"{_strip_lead(question)} {day}"
        return Resolution(text, "rule", f"attribute -> {new}")

    if only_entity and (cur.lead or not asks_itself) and prev.entity and prev.entity[0] != "topic":
        assert cur.entity is not None
        new = cur.surface(cur.entity[1])
        return Resolution(_replace(prev.text, [(prev.entity[1], new)]), "rule", f"subject -> {new}")

    if not cur.has_subject and prev.entity:
        # "why?", "ఎప్పుడు?", "explain more": ask the same about the previous subject.
        assert prev.entity_text is not None
        return Resolution(f"{prev.entity_text} {_strip_lead(question)}".strip(), "rule", "bare follow-up")
    return Resolution(question, "none")


def needs_context(question: str, today: date) -> bool:
    """True when the question cannot stand alone: pronouns, "and …?" or no subject at all."""
    cur = analyse(question, today)
    return bool(cur.entity_pronouns or cur.day_pronouns or cur.lead or not cur.has_subject)


_SYSTEM = """You rewrite a user's follow-up message into one standalone question.
Use the earlier questions only to fill in what the follow-up leaves out (the subject, date, rasi
or topic it refers to). Do not answer it. Do not add facts, dates or names that are not in the
conversation. Keep the follow-up's language and script exactly (Telugu stays Telugu, English stays
English). If the message already stands alone, return it unchanged.
Reply as JSON: {"question": "<standalone question>"}"""


class FollowUpResolver:
    """Rules first; an LLM rewrite only for follow-ups the rules cannot place."""

    def __init__(self, llm: LLM | None = None, *, max_tokens: int = 200) -> None:
        self.llm = llm
        self.max_tokens = max_tokens
        self.last_cost_tokens: tuple[int, int] = (0, 0)

    def resolve(self, question: str, history: Sequence[TurnRecord], now: datetime) -> Resolution:
        self.last_cost_tokens = (0, 0)
        res = resolve_by_rules(question, history, now.date())
        if res.method != "none" or not history or self.llm is None or not needs_context(question, now.date()):
            return res
        return self._llm(question, history)

    def _llm(self, question: str, history: Sequence[TurnRecord]) -> Resolution:
        assert self.llm is not None
        convo = "\n".join(f"Q{i + 1}: {t.standalone_question}" for i, t in enumerate(history))
        messages = [
            ChatMessage("system", _SYSTEM),
            ChatMessage("user", f"Earlier questions:\n{convo}\n\nFollow-up: {question}"),
        ]
        try:
            res = self.llm.complete(messages, max_tokens=self.max_tokens, temperature=0.0, json=True)
        except LLMError as exc:
            log.warning("followup.llm_failed", error=str(exc))
            return Resolution(question, "none", "rewrite unavailable")
        self.last_cost_tokens = (res.prompt_tokens, res.completion_tokens)
        m = re.search(r"\{.*\}", res.text, re.S)
        try:
            data = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            data = {}
        text = str(data.get("question", "")).strip() if isinstance(data, dict) else ""
        if not text or len(text) > 400 or detect_language(text).code != detect_language(question).code:
            return Resolution(question, "none", "rewrite rejected")
        if text == question:
            return Resolution(question, "none")
        return Resolution(text, "llm", "llm rewrite")

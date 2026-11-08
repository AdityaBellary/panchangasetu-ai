"""Slot extraction from a question: which panchanga attribute is asked, which named tithi /
nakshatra is meant, and whether it is a "when/next" or a "what/why" question.

Terms are matched in Telugu script, English and romanised Telugu.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from panchangasetu.telugu.lexicon import NAKSHATRAS, TITHIS
from panchangasetu.telugu.text import normalize_telugu

ATTRIBUTES: dict[str, list[str]] = {
    "tithi": ["tithi", "thithi", "thidhi", "tithis", "తిథి", "తిధి", "తిథులు"],
    "nakshatra": ["nakshatram", "nakshatra", "nakshathram", "star", "నక్షత్రం", "నక్షత్ర"],
    "yoga": ["yogam", "yoga", "యోగం"],
    "karana": ["karanam", "karana", "కరణం"],
    "masa": ["masam", "masa", "lunar month", "మాసం", "మాసము"],
    "paksha": ["paksham", "paksha", "fortnight", "పక్షం", "పక్షము"],
    "varjyam": ["varjyam", "varjam", "వర్జ్యం"],
    "durmuhurtam": ["durmuhurtam", "durmuhurtham", "dur muhurtham", "దుర్ముహూర్తం"],
    "amrutham": ["amrutha kalam", "amrita kalam", "amritakalam", "amrutham", "అమృతకాలం", "అమృత కాలం", "అమృతం"],
    "rahukalam": ["rahu kalam", "rahukalam", "rahu kaal", "rahukaal", "rahu kalamu", "రాహుకాలం", "రాహు కాలం"],
    "yamagandam": [
        "yamagandam",
        "yama gandam",
        "yamaganda",
        "yamakantakam",
        "యమగండం",
        "యమగండము",
        "యమకంటకం",
        "యమకంటక",
    ],
    "gulika": ["gulika kalam", "gulikai", "gulika", "kuligai", "గుళిక కాలం", "గుళికకాలం", "గుళిక"],
    "sunrise": ["sunrise", "sun rise", "సూర్యోదయం"],
    "sunset": ["sunset", "sun set", "సూర్యాస్తమయం"],
    "festival": [
        "festival",
        "festivals",
        "panduga",
        "pandugalu",
        "utsavam",
        "utsavams",
        "brahmotsavam",
        "jayanti",
        "vratam",
        "పండుగ",
        "పండుగలు",
        "ఉత్సవం",
        "ఉత్సవాలు",
        "బ్రహ్మోత్సవం",
        "జయంతి",
        "వ్రతం",
    ],
    "muhurtham": [
        "muhurtham",
        "muhurtam",
        "muhurat",
        "muhurtham",
        "auspicious time",
        "auspicious date",
        "ముహూర్తం",
        "ముహూర్తాలు",
        "ముహూర్తము",
        "శుభ ముహూర్తం",
    ],
    "panchangam": ["panchangam", "panchang", "panchanga", "పంచాంగం"],
    "karthe": ["karthe", "karte", "kaarthe", "karthi", "కార్తె", "కార్తెలు"],
    "transit": ["transit", "enters", "enter", "sankramanam", "sankranti", "ప్రవేశం", "సంక్రమణం", "సంచారం"],
}
# Dated items answered by searching events/muhurtham lists rather than one day's facts.
EVENT_ATTRIBUTES = frozenset({"festival", "muhurtham", "karthe", "transit"})

# Topics that live in the General collection (predictions, rules, omens, prayers).
GENERAL_TOPICS = [
    "rasi", "rashi", "phalalu", "prediction", "predictions", "horoscope", "dosha", "dosham", "vastu",
    "stotram", "stotra", "sloka", "slokas", "prayer", "dream", "dreams", "lizard", "omen", "tarpanam",
    "remedy", "remedies", "gemstone", "matching", "compatibility", "after marriage", "king", "minister",
    "ఫలితాలు", "ఫలితములు", "రాశి", "దోషం", "దోష", "వాస్తు", "స్తోత్రం", "శ్లోకం", "శ్లోకములు", "కల",
    "బల్లి", "శకునం", "తర్పణం", "పొంతన", "రాజు", "మంత్రి", "నివారణ",
    # The year's forecasts (samvatsara phalalu) and fixed weekday tables live in General text too.
    "rain", "rains", "rainfall", "weather", "price", "prices", "forecast", "gowri", "gauri",
    "వర్షం", "వర్షాలు", "వర్షములు", "వర్ష", "ధర", "ధరలు", "గౌరీ", "ఆదాయం", "వ్యయం", "కందాయ",
]  # fmt: skip
_CLAUSE_SPLIT = re.compile(r"\s(?:and|also)\s|,|;|మరియు|\?(?=.)")
_STOPWORDS = frozenset(
    """
    the a an is are was of for on in at to and or what which when how date day next upcoming coming
    this year 2026 2027 eppudu vachche ye roju ee enti tell me please sri srii ఏ ఈ ఎప్పుడు వచ్చే
    తదుపరి తర్వాతి తరువాతి రోజు తేదీ ఏమిటి శ్రీ
    """.split()
)
DAY_ATTRIBUTES = frozenset(
    {"tithi", "nakshatra", "yoga", "karana", "masa", "paksha", "varjyam", "durmuhurtam", "amrutham",
     "rahukalam", "yamagandam", "gulika", "sunrise", "sunset", "panchangam"}
)  # fmt: skip

WHEN_MARKERS = [
    "when", "which date", "what date", "which day", "date of", "next", "upcoming", "coming",
    "eppudu", "yeppudu", "vachche", "ఎప్పుడు", "ఏ తేదీ", "ఏ రోజు", "వచ్చే", "తర్వాతి", "తదుపరి", "తరువాతి",
]  # fmt: skip
# Strong markers ask about meaning/reasons; they make a question (partly) General even when it
# names a date. Weak markers ("what is", "అంటే") also open plain lookups ("what is today's tithi").
STRONG_CONCEPT_MARKERS = [
    "significance", "meaning", "importance", "why", "explain", "rules", "story", "benefits", "how to",
    "enduku", "ప్రాముఖ్యత", "విశిష్టత", "ఎందుకు", "నియమాలు", "కథ", "ఫలితం", "ఫలితాలు", "ఫలితములు",
]  # fmt: skip
WEAK_CONCEPT_MARKERS = [
    "what is",
    "what are",
    "how is",
    "ante enti",
    "ante emiti",
    "ela",
    "అంటే",
    "ఎలా",
    "గురించి",
]


@dataclass(slots=True)
class Slots:
    attributes: list[str] = field(default_factory=list)
    rasi: str | None = None  # canonical Telugu stem ("కన్యా") when the question names a zodiac sign
    tithi: str | None = None  # canonical Telugu name
    nakshatra: str | None = None
    when: bool = False  # "when / next / ఎప్పుడు"
    concept: bool = False  # strong: "significance / why / ఎందుకు"
    weak_concept: bool = False  # "what is / అంటే"
    next_occurrence: bool = False
    general_topic: bool = False
    multi_clause: bool = False
    # Content words left after removing the named tithi/nakshatra, markers and stopwords. A tithi
    # with extra content ("Sri Rama Navami", "రథసప్తమి") names a festival, not just the tithi.
    residual: list[str] = field(default_factory=list)


def _norm(text: str) -> str:
    return " " + re.sub(r"\s+", " ", normalize_telugu(text).lower()) + " "


def _contains(haystack: str, term: str) -> bool:
    if term.isascii():  # allow simple English plurals ("muhurthams", "festivals")
        return re.search(rf"(?<![a-z]){re.escape(term)}s?(?![a-z])", haystack) is not None
    return term in haystack  # Telugu: suffixes attach directly ("ఏకాదశికి"), so substring match


def _name(haystack: str, table: dict[str, list[str]]) -> str | None:
    best: tuple[int, str] | None = None
    for canonical, aliases in table.items():
        for form in (canonical, *aliases):
            # Longest match wins (ఉత్తరాషాఢ over ఉత్తర).
            if len(form) >= 3 and _contains(haystack, form.lower()) and (best is None or len(form) > best[0]):
                best = (len(form), canonical)
    return best[1] if best else None


def extract_slots(text: str) -> Slots:
    h = _norm(text)
    slots = Slots()
    for attr, terms in ATTRIBUTES.items():
        if any(_contains(h, t.lower()) for t in terms):
            slots.attributes.append(attr)
    # "దుర్ముహూర్తం" / "durmuhurtham" contain "ముహూర్తం" / "muhurtham" but mean the opposite.
    if "durmuhurtam" in slots.attributes and "muhurtham" in slots.attributes:
        rest = h.replace("దుర్ముహూర్తం", " ").replace("durmuhurtham", " ").replace("durmuhurtam", " ")
        if not any(_contains(rest, t.lower()) for t in ATTRIBUTES["muhurtham"]):
            slots.attributes.remove("muhurtham")
    slots.rasi = _rasi(h)
    # "Makara Sankranti" / "మకర సంక్రాంతి" names a festival (the Sun entering the sign), not a question
    # about Makara rasi predictions.
    if slots.rasi and re.search(r"(?:sankranti|sankranthi|sankramanam|సంక్రాంతి|సంక్రమణ)", h):
        slots.rasi = None
    slots.tithi = _name(h, TITHIS)
    slots.nakshatra = _name(h, NAKSHATRAS)
    # "Magha masam" / "మాఘ మాసం" names a lunar month; the nakshatra Magha/మఖ is only an alias clash.
    if slots.nakshatra and re.search(r"(?:magha|maagha|మాఘ)\s*(?:masam|masa|month|మాస)", h):
        slots.nakshatra = None
    # A bare nakshatra name (e.g. "hasta") inside other words is common in English text;
    # only trust it when the question is about panchanga or names a nakshatra explicitly.
    if slots.nakshatra and slots.nakshatra.isascii():
        slots.nakshatra = None
    slots.when = any(_contains(h, m) for m in WHEN_MARKERS)
    slots.concept = any(_contains(h, m) for m in STRONG_CONCEPT_MARKERS)
    slots.weak_concept = any(_contains(h, m) for m in WEAK_CONCEPT_MARKERS)
    slots.general_topic = any(_contains(h, t.lower()) for t in GENERAL_TOPICS)
    slots.multi_clause = len([c for c in _CLAUSE_SPLIT.split(text) if c.strip()]) > 1
    slots.residual = _residual(h, slots)
    slots.next_occurrence = any(
        _contains(h, m) for m in ("next", "upcoming", "coming", "vachche", "వచ్చే", "తర్వాతి", "తదుపరి", "తరువాతి")
    )
    return slots


def _residual(h: str, slots: Slots) -> list[str]:
    from panchangasetu.telugu.lexicon import NAKSHATRAS as _N
    from panchangasetu.telugu.lexicon import TITHIS as _T

    drop: set[str] = set(_STOPWORDS)
    for terms in ATTRIBUTES.values():
        drop.update(w for t in terms for w in t.lower().split())
    for m in WHEN_MARKERS + STRONG_CONCEPT_MARKERS + WEAK_CONCEPT_MARKERS:
        drop.update(m.lower().split())
    names: set[str] = set()
    for table, chosen in ((_T, slots.tithi), (_N, slots.nakshatra)):
        if chosen:
            names.update(f.lower() for f in (chosen, *table[chosen]))
    out = []
    for tok in re.findall(r"[a-z]+|[\u0C00-\u0C7F]+", h):
        if tok in drop or tok in names or tok.isdigit():
            continue
        # Telugu: a token that merely inflects the name ("ఏకాదశికి") is not extra content,
        # but a compound that contains it ("రథసప్తమి") is.
        if any(tok.startswith(n) for n in names if not n.isascii()):
            continue
        out.append(tok)
    return out


def _rasi(h: str) -> str | None:
    from panchangasetu.telugu.lexicon import RASIS

    for stem, aliases in RASIS.items():
        for form in (stem, *aliases):
            if form.isascii():
                if re.search(rf"(?<![a-z]){form}(?:\s*ras[hi]+)?(?![a-z])", h):
                    return stem
            elif re.search(rf"{form}\s*రాశి|{form}రాశి", h):  # Telugu needs "రాశి": "కన్య" alone is "girl"
                return stem
    return None

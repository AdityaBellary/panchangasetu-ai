from datetime import date

import pytest

from panchangasetu.query.dates import parse_dates
from panchangasetu.query.language import detect_language
from panchangasetu.query.rewrite import Glossary, expand_query
from panchangasetu.query.router import Intent, PanchangaMode, QueryRouter
from panchangasetu.query.slots import extract_slots

TODAY = date(2026, 11, 7)  # a Saturday


@pytest.mark.parametrize(
    ("text", "code", "romanised"),
    [
        ("What is today's nakshatram?", "en", False),  # Telugu terms in English stay English
        ("ఈరోజు నక్షత్రం ఏమిటి?", "te", False),
        ("ee roju tithi enti?", "te", True),
        ("vachche pournami eppudu vastundi?", "te", True),
        ("Ekadashi significance", "en", False),
    ],
)
def test_answer_language_follows_the_question(text: str, code: str, romanised: bool) -> None:
    lang = detect_language(text)
    assert (lang.code, lang.romanised_telugu) == (code, romanised)


@pytest.mark.parametrize(
    ("text", "start", "end"),
    [
        ("today's tithi", "2026-11-07", "2026-11-07"),
        ("రేపు తిథి", "2026-11-08", "2026-11-08"),
        ("repu amrutha kalam", "2026-11-08", "2026-11-08"),
        ("ఎల్లుండి", "2026-11-09", "2026-11-09"),
        ("on 15/12/2026", "2026-12-15", "2026-12-15"),
        ("2027-01-14 panchangam", "2027-01-14", "2027-01-14"),
        ("15 December", "2026-12-15", "2026-12-15"),
        ("Jan 3", "2027-01-03", "2027-01-03"),  # no year: nearest sensible occurrence
        ("డిసెంబర్ 20 న", "2026-12-20", "2026-12-20"),
        ("next Friday", "2026-11-13", "2026-11-13"),
        ("శుక్రవారం", "2026-11-13", "2026-11-13"),
        ("festivals this week", "2026-11-07", "2026-11-08"),
        ("వచ్చే నెల", "2026-12-01", "2026-12-31"),
        ("in April 2027", "2027-04-01", "2027-04-30"),
        ("మేలో పండుగలు", "2027-05-01", "2027-05-31"),  # suffix allowed after a Telugu month
    ],
)
def test_dates_resolve(text: str, start: str, end: str) -> None:
    specs = parse_dates(text, TODAY)
    assert specs and (specs[0].start.isoformat(), specs[0].end.isoformat()) == (start, end)


def test_telugu_month_name_inside_a_word_is_not_a_date() -> None:
    assert parse_dates("మేషరాశి ఫలితాలు", TODAY) == []  # "మే" (May) inside "మేషరాశి"


def test_slots_distinguish_durmuhurtam_from_muhurtham() -> None:
    assert "muhurtham" not in extract_slots("డిసెంబర్ 20 న దుర్ముహూర్తం").attributes
    assert "muhurtham" in extract_slots("marriage muhurthams in April").attributes


def test_festival_named_after_a_tithi_has_residual_content() -> None:
    assert extract_slots("Sri Rama Navami date").residual == ["rama"]
    assert extract_slots("వచ్చే ఏకాదశి ఎప్పుడు?").residual == []
    assert extract_slots("తిరుమల రథసప్తమి ఎప్పుడు?").residual  # compound + place name


ROUTER = QueryRouter(None)  # rules only


@pytest.mark.parametrize(
    ("text", "intent", "mode"),
    [
        ("What is today's nakshatram?", Intent.PANCHANGA, PanchangaMode.DAY),
        ("ఈరోజు నక్షత్రం ఏమిటి?", Intent.PANCHANGA, PanchangaMode.DAY),
        ("When is the next Ekadashi?", Intent.PANCHANGA, PanchangaMode.FIND),
        ("వచ్చే ఏకాదశి ఎప్పుడు?", Intent.PANCHANGA, PanchangaMode.FIND),
        ("Sri Rama Navami 2026 date", Intent.HYBRID, PanchangaMode.SEARCH),  # needs semantic; safe fallback
        ("Festivals this week", Intent.PANCHANGA, PanchangaMode.SEARCH),
        ("డిసెంబర్ 20 న దుర్ముహూర్తం ఎప్పుడు?", Intent.PANCHANGA, PanchangaMode.DAY),
        ("What is Kuja dosha?", Intent.GENERAL, PanchangaMode.NONE),
        ("How is Rahu kalam calculated?", Intent.GENERAL, PanchangaMode.NONE),
        ("మేషరాశి వారికి ఈ ఏడాది ఫలితాలు", Intent.GENERAL, PanchangaMode.NONE),
        ("What is Ekadashi and when is the next one?", Intent.HYBRID, PanchangaMode.FIND),
    ],
)
def test_rule_routing(text: str, intent: Intent, mode: PanchangaMode) -> None:
    r = ROUTER.route(text, TODAY)
    assert (r.intent, r.mode) == (intent, mode)


class FakeSemantic:
    def __init__(self, label: str, conf: float) -> None:
        self.result = (label, conf)

    def classify(self, text: str) -> tuple[str, float]:
        return self.result


def test_low_semantic_confidence_queries_both_collections() -> None:
    r = QueryRouter(FakeSemantic("general", 0.55)).route("krishnashtami eppudu?", TODAY)
    assert r.intent == Intent.HYBRID


def test_confident_semantic_decision_is_used() -> None:
    r = QueryRouter(FakeSemantic("panchanga", 0.95)).route("Garuda seva at Tirumala", TODAY)
    assert (r.intent, r.mode) == (Intent.PANCHANGA, PanchangaMode.SEARCH)


def test_query_expansion_adds_telugu_terms_for_english_only() -> None:
    glossary = Glossary({"lizard": ["బల్లిపాటు"], "shukla paksha": ["శుక్లపక్షము"], "paksha": ["పక్షము"]})
    en = "What if a lizard falls? Shukla paksha meaning"
    out = expand_query(en, detect_language(en), extract_slots(en), glossary)
    assert "బల్లిపాటు" in out and "శుక్లపక్షము" in out and "పక్షము" not in out.replace("శుక్లపక్షము", "")
    te = "బల్లి పడితే?"
    assert expand_query(te, detect_language(te), extract_slots(te), glossary) == te
    nxt = "When is the next Ekadashi?"
    assert "ఏకాదశి" in expand_query(nxt, detect_language(nxt), extract_slots(nxt), glossary)


def test_telugu_day_month_year_is_not_misread_as_month_day() -> None:
    from datetime import date

    from panchangasetu.query.dates import parse_dates

    for text, expected in [
        ("27 మార్చి 2027 తిథి ఏది?", date(2027, 3, 27)),
        ("2 జూలై 2026 న తిథి ఏమిటి?", date(2026, 7, 2)),
        ("డిసెంబర్ 5, 2026 తిథి", date(2026, 12, 5)),
    ]:
        assert [s.start for s in parse_dates(text, date(2026, 11, 7))] == [expected]


def test_month_and_festival_names_are_not_mistaken_for_nakshatra_or_rasi() -> None:
    assert extract_slots("When will it rain in Magha masam?").nakshatra is None
    assert extract_slots("Why is Makara Sankranti on 15 January?").rasi is None
    assert extract_slots("మకర సంక్రాంతి ఎప్పుడు?").rasi is None
    assert extract_slots("What are the predictions for Makara rasi?").rasi == "మకర"

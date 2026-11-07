from pathlib import Path

import pytest

from panchangasetu.core.types import Collection
from panchangasetu.ingestion.classification import CollectionRules, PagePlan, classify_page, parse_page_spec

RULES_YAML = """
documents:
  - match: "TTD-Calendar-*.pdf"
    defaults: {collection: panchanga, section_type: monthly_calendar}
  - match: "book.pdf"
    defaults: {ocr_engine: sarvam}
    ranges:
      - {pages: "1-2", collection: skip, section_type: cover}
      - {pages: "10, 12-13", collection: panchanga, section_type: daily_table, ocr_engine: tesseract}
"""


@pytest.fixture
def rules(tmp_path: Path) -> CollectionRules:
    p = tmp_path / "collections.yaml"
    p.write_text(RULES_YAML, encoding="utf-8")
    return CollectionRules.load(p)


def test_parse_page_spec_supports_ranges_and_lists() -> None:
    assert parse_page_spec("1-3, 7,10-11") == {1, 2, 3, 7, 10, 11}
    assert parse_page_spec(5) == {5}
    with pytest.raises(ValueError):
        parse_page_spec("5-2")


def test_rules_apply_defaults_then_more_specific_ranges(rules: CollectionRules) -> None:
    assert rules.plan("book.pdf", 1) == PagePlan(Collection.SKIP, "cover", "sarvam")
    assert rules.plan("book.pdf", 12) == PagePlan(Collection.PANCHANGA, "daily_table", "tesseract")
    assert rules.plan("book.pdf", 11) == PagePlan(None, None, "sarvam")


def test_rules_match_by_filename_glob_in_subdirectories(rules: CollectionRules) -> None:
    plan = rules.plan("2027/TTD-Calendar-2027.pdf", 4)
    assert plan.collection == Collection.PANCHANGA and plan.section_type == "monthly_calendar"


def test_missing_rules_file_means_no_rules(tmp_path: Path) -> None:
    assert CollectionRules.load(tmp_path / "nope.yaml").plan("x.pdf", 1) == PagePlan(None, None, None)


def test_rule_wins_over_heuristic() -> None:
    c = classify_page("ఈ రాశివారికి గురుడు సామాన్య ఫలితములను ఇచ్చును.", PagePlan(Collection.PANCHANGA, "x", None))
    assert c.collection == Collection.PANCHANGA and c.reason == "rule" and c.confidence == 1.0


def test_heuristic_routes_dense_calendar_text_to_panchanga() -> None:
    text = (
        "నవమి ఉ.9-43 విశాఖ ఉ.7-14 ప.వ.11-31 ల 1-14\n"
        "ఏకాదశి సా.6-08 విశాఖ రా.11-49 తె.వ.4-07 ల 5-50\n"
        "తదియ పూర్తి శ్రవణం ప.1-49 సా.వ.6-09 ల 7-53"
    )
    c = classify_page(text, PagePlan(None, None, None))
    assert c.collection == Collection.PANCHANGA


def test_heuristic_routes_prose_to_general() -> None:
    text = (
        "ఈ రాశివారికి గురుడు 2026 జూన్ 1 వరకు దశమ స్థానమందు తామ్రమూర్తి సామాన్యఫలితములను, "
        "తదుపరి జూన్ 2 నుండి ఏకాదశ స్థానమందు లోహమూర్తిగా ధననాశమును కలుగజేయును. జీవితభాగస్వామి "
        "ద్వారా ఆర్థిక లాభాలకు, తల్లి లేదా పెద్ద తోబుట్టువుల నుండి గణనీయమైన లాభాలు."
    )
    c = classify_page(text, PagePlan(None, None, None))
    assert c.collection == Collection.GENERAL


def test_near_empty_page_is_skipped() -> None:
    assert classify_page(" 12 ", PagePlan(None, None, None)).collection == Collection.SKIP


def test_explanatory_text_dense_with_panchanga_vocabulary_is_general() -> None:
    # Explains paksham/tithi concepts (answers "what is a paksham?") but carries no dated facts.
    text = (
        "పక్షము - రెండు విధములు : శుక్లపక్షము - కృష్ణపక్షము. శుక్లపక్షమున పాడ్యమి, విదియ, తదియ, "
        "చవితి, పంచమి తిథులు వచ్చును. పూర్ణిమ తరువాత బహుళ పక్షము ప్రారంభమగును. అమావాస్య వరకు "
        "తిథులు క్షీణించును. నక్షత్రము, యోగం, కరణం పంచాంగములోని అంగములు."
    )
    c = classify_page(text, PagePlan(None, None, None))
    assert c.collection == Collection.GENERAL


def test_dated_muhurtham_list_is_panchanga() -> None:
    text = (
        "ఏప్రిల్ 12. ఆది. దశమి శ్రవణం ఉపనయన వివాహ గృహారంభములు మేషం ఉ.7-20\n"
        "ఏప్రిల్ 15. బుధ. త్రయోదశి పుబ్బ వివాహములు వృషభం ఉ.9-05\n"
        "ఏప్రిల్ 20. సోమ. తదియ రోహిణి గృహప్రవేశం మిథునం ప.12-10"
    )
    assert classify_page(text, PagePlan(None, None, None)).collection == Collection.PANCHANGA


def test_number_lists_are_not_mistaken_for_dates() -> None:
    from panchangasetu.telugu.lexicon import DATE_RE

    assert DATE_RE.findall("తిథులు 2-3-5-7-10 శుభము") == []
    assert DATE_RE.findall("13-8-2026 నుండి 11-9-2026 వరకు, ఏప్రిల్ 12") == ["13-8-2026", "11-9-2026"]


def test_real_rules_file_maps_book_sections() -> None:
    from pathlib import Path

    rules = CollectionRules.load(Path(__file__).resolve().parents[2] / "config/collections.yaml")
    assert rules.plan("TTD_panchangam.pdf", 31).collection == Collection.GENERAL
    assert rules.plan("TTD_panchangam.pdf", 120) == PagePlan(
        Collection.PANCHANGA, "daily_panchangam_table", "tesseract"
    )
    assert rules.plan("TTD_panchangam.pdf", 155).section_type == "ttd_festival_calendar"
    assert rules.plan("TTD_panchangam.pdf", 2).collection == Collection.SKIP

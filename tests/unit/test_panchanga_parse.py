from datetime import date

import pytest

from panchangasetu.structured.legacy_text import decode_legacy_ascii, digits_only
from panchangasetu.structured.panchanga_parse import (
    ParsedDay,
    match_name,
    next_tithi,
    parse_day,
    repair_ghati,
    resolve_sequences,
    to_datetime,
)

# Real OCR'd rows from the TTD 2026-27 book (pp. 100 and 131), with the numbers recovered
# exactly from the PDF's legacy-font text layer for those rows.
UGADI = (
    "గురు అమావాస్య 1-50 ఉ.7-06 చైత్ర శుద్ధ పాడ్యమి 58-33 తె.5-47 ఉత్తరాభాద్ర 56-45 తె.5-04 శుక్రం 49-42 రా.2-15\n"
    "నాగవం 1-50 ఉ.7-06 కింస్తుఘ్నం 30-11 సా.6-27 బవ 58-33 తె.5-47 ప.వ.2-58 ల 4-32 ఉ.దు.10-21 ల\n"
    "11-09 ప.దు. 3-08 ల 3-56 అమృతం 45-00 రా.12-22 ల 1-56. శ్రీ పరాభవ నామ సంవత్సర ఉగాది, చాంద్రవత్సరాది,\n"
    "భో3-34 అహః 29-55"
)
UGADI_NUMS = [
    *["1-50", "7-06", "30-11", "6-27", "58-33", "5-47", "56-45", "5-04", "49-42", "2-15", "2-58"],
    *["4-32", "10-21", "11-09", "3-08", "3-56", "45-00", "12-22", "1-56", "3-34", "29-55"],
]
NOV7 = (
    "శని త్రయోదశి 10-40 ఉ.10-29 చిత్ర 59-57 తె.6-12 ప్రీతి 56-05 తె.4-39 వణి 10-40 ఉ.10-29 భద్ర 40-46 "
    "రా.10-32 ప.వ.1-47 ల 3-26\nఉ.దు.6-13 ల 7-44 అమృతం 43-33 రా.11-38 ల 1-16 విశాఖ 1 అర్క! 8-13 ఉ.9-21. "
    "ధన్వంతరి జయంతి, మాసశివరాత్రి. భో.1-47 అహః 28-40"
)
NOV7_NUMS = [
    *["10-40", "10-29", "59-57", "6-12", "56-05", "4-39", "40-46", "10-32", "1-47"],
    *["3-26", "6-13", "7-44", "43-33", "11-38", "1-16", "8-13", "9-21", "28-40"],
]


def parse(text: str, day: date, sunrise: str, nums: list[str]) -> ParsedDay:
    return parse_day(text, day, sunrise=sunrise, sunset=None, weekday_printed=None, exact_numbers=nums)


def test_legacy_font_digits_are_recovered_exactly() -> None:
    assert decode_legacy_ascii("") == "6-22·"
    assert digits_only("") == "19"
    assert decode_legacy_ascii("తె.4-11") == "తె.4-11"  # real Unicode passes through


@pytest.mark.parametrize(
    ("prefix", "hm", "expected"),
    [
        ("ఉ", "7-06", "2026-03-19T07:06"),
        ("ప", "9-58", "2026-03-19T09:58"),  # daytime 6-11 is morning
        ("ప", "3-13", "2026-03-19T15:13"),
        ("సా", "6-27", "2026-03-19T18:27"),
        ("రా", "9-21", "2026-03-19T21:21"),
        ("రా", "12-22", "2026-03-20T00:22"),  # just after midnight -> next civil date
        ("రా", "2-15", "2026-03-20T02:15"),
        ("తె", "5-47", "2026-03-20T05:47"),  # pre-dawn belongs to the next civil date
    ],
)
def test_part_of_day_times_map_to_civil_datetimes(prefix: str, hm: str, expected: str) -> None:
    assert to_datetime(date(2026, 3, 19), prefix, hm).isoformat(timespec="minutes") == expected


def test_ugadi_row_parses_and_every_element_is_verified_two_ways() -> None:
    d = parse(UGADI, date(2026, 3, 19), "6-22", UGADI_NUMS)
    tithis = [e for e in d.elements if e.kind == "tithi"]
    assert [(t.name, t.printed_end, t.paksha) for t in tithis] == [
        ("అమావాస్య", "ఉ.7-06", None),
        ("పాడ్యమి", "తె.5-47", "శుద్ధ"),
    ]
    assert tithis[1].masa == "చైత్రం"
    assert [e.name for e in d.elements if e.kind == "nakshatra"] == ["ఉత్తరాభాద్ర"]
    assert all(e.verified for e in d.elements)
    kinds = [(w.kind, w.starts_at[11:], w.ends_at[11:]) for w in d.windows]
    assert ("varjyam", "14:58", "16:32") in kinds
    assert ("durmuhurtam", "10:21", "11:09") in kinds
    assert ("amrutham", "00:22", "01:56") in kinds
    assert "ఉగాది" in d.notes


def test_misread_time_is_caught_by_ghati_and_text_layer_checks() -> None:
    tampered = NOV7.replace("ఉ.10-29 చిత్ర", "ఉ.10-59 చిత్ర")  # OCR turns 2 into 5
    d = parse(tampered, date(2026, 11, 7), "6-13", NOV7_NUMS)
    tithi = next(e for e in d.elements if e.kind == "tithi")
    assert tithi.ghati_check == "off" and tithi.digits_check == "missing" and not tithi.verified


def test_dropped_vighati_digit_is_repaired_from_exact_numbers() -> None:
    assert repair_ghati("44-8", {"44-08", "11-58"}) == "44-08"
    assert repair_ghati("44-8", {"44-18", "44-28"}) == "44-08"  # ambiguous -> conservative default


@pytest.mark.parametrize(
    ("token", "kind", "name"),
    [
        ("షస్టి", "tithi", "షష్ఠి"),
        ("చతుర్రశి", "tithi", "చతుర్దశి"),
        ("చిత్ర", "nakshatra", "చిత్త"),
        ("జ్యేష్ట", "nakshatra", "జ్యేష్ఠ"),
        ("బాలువ", "karana", "బాలవ"),
        ("విస్త", "karana", "విష్టి"),  # must not become the nakshatra హస్త
        ("బ్రాహ్మం", "yoga", "బ్రహ్మం"),
    ],
)
def test_ocr_name_variants_resolve_to_the_right_element(token: str, kind: str, name: str) -> None:
    found = match_name(token)
    assert found is not None and found[:2] == (kind, name)


def test_next_tithi_handles_paksha_ends() -> None:
    assert next_tithi("చతుర్దశి", "శుద్ధ") == "పూర్ణిమ"
    assert next_tithi("చతుర్దశి", "బహుళ") == "అమావాస్య"
    assert next_tithi("అమావాస్య", "బహుళ") == "పాడ్యమి"


def _day(d: date, text: str, sunrise: str = "6-00") -> ParsedDay:
    nums = [n for n in __import__("re").findall(r"\d{1,2}-\d{2}", text)]
    return parse(text, d, sunrise, nums)


def test_sequence_fills_unreadable_names_and_tracks_masa_paksha_and_adhika() -> None:
    days = [
        _day(date(2026, 5, 15), "శుక్ర చతుర్దశి 30-00 సా.6-00 భరణి 30-00 సా.6-00"),
        _day(date(2026, 5, 16), "శని అమావాస్య 32-30 సా.7-00 కృత్తిక 32-30 సా.7-00"),
        _day(date(2026, 5, 17), "ఆది అధిక జ్యేష్ట శుద్ధ పాడ్యమి 35-00 రా.8-00 రోహిణి 35-00 రా.8-00"),
        _day(date(2026, 5, 18), "సోమ ఙఞక 37-30 రా.9-00 మృగశిర 37-30 రా.9-00"),  # garbled tithi name
    ]
    resolve_sequences(days, initial_masa="వైశాఖం", initial_paksha="బహుళ")
    t = [next(e for e in d.elements if e.kind == "tithi") for d in days]
    assert (t[1].name, t[1].masa, t[1].paksha) == ("అమావాస్య", "వైశాఖం", "బహుళ")
    assert (t[2].name, t[2].masa, t[2].paksha, t[2].adhika) == ("పాడ్యమి", "జ్యేష్ఠం", "శుద్ధ", True)
    assert (t[3].name, t[3].name_source, t[3].sequence_check) == ("విదియ", "sequence", "inferred")
    assert t[3].adhika is True and t[3].ghati_check == "ok"  # timing still verified


def test_confident_name_that_contradicts_the_cycle_is_flagged_not_trusted() -> None:
    days = [
        _day(date(2026, 6, 1), "సోమ పంచమి 30-00 సా.6-00 హస్త 30-00 సా.6-00"),
        _day(date(2026, 6, 2), "మంగళ దశమి 30-00 సా.6-00 చిత్త 30-00 సా.6-00"),  # expected షష్ఠి
    ]
    resolve_sequences(days, initial_masa="జ్యేష్ఠం", initial_paksha="శుద్ధ")
    tithi = next(e for e in days[1].elements if e.kind == "tithi")
    assert tithi.sequence_check == "break" and not tithi.verified


def test_planet_ingress_mentions_after_karanas_are_not_day_elements() -> None:
    text = "శని త్రయోదశి 10-40 ఉ.10-29 చిత్ర 59-57 తె.6-12 ప్రీతి 56-05 తె.4-39 వణి 10-40 ఉ.10-29 కృత్తిక 12-00 ప.11-00"
    d = _day(date(2026, 11, 7), text, "6-13")
    assert [e.name for e in d.elements if e.kind == "nakshatra"] == ["చిత్త"]

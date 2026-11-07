from datetime import date

from panchangasetu.structured.festivals import parse_festival_table, parse_utsavam_schedules
from panchangasetu.structured.wall_calendar import Grid, fit_grid, month_position, parse_cell

FESTIVAL_PAGE = """## 152
ఇంగ్లీష్ | తమిళం | వారం | ఉత్సవాది విశేషములు
2026 | పంగుని 2 | సోమ | తిరుపతి శ్రీ కోదండరామస్వామివారి బ్రహ్మోత్సవ అంకురార్పణ
19 | 4 | గురు | శ్రీ పరాభవనామ సంవత్సర ఉగాది
30 | 16 | సోమ | శ్రీరామ నవమి
ఆశ్వయుజ మాసము
2 | 19 | గురు | ఏప్రిల్ రెండవ వరుస (no month printed)
2 | 19 | గురు | same date, wrapped second row
6 | సోమ | tamil column missing on this row
"""


def test_festival_rows_resolve_by_day_and_weekday_including_month_changes() -> None:
    events, problems = parse_festival_table([(152, FESTIVAL_PAGE)], start=date(2026, 3, 15))
    dates = [e.date for e in events]
    assert dates[:3] == ["2026-03-16", "2026-03-19", "2026-03-30"]  # year-only cell resolved by weekday
    assert dates[3] == dates[4] == "2026-04-02"  # month rolled over; wrapped row keeps the date
    assert dates[5] == "2026-04-06"  # tamil column absent; weekday column still found
    assert events[1].details.endswith("ఉగాది") and events[1].tamil_date == "4"
    assert problems == []


def test_festival_row_that_cannot_match_its_weekday_is_reported_not_guessed() -> None:
    page = "ఇంగ్లీష్ | తమిళం | వారం | విశేషములు\n19 | 4 | ఆది | wrong weekday for 19 March\n"
    events, problems = parse_festival_table([(152, page)], start=date(2026, 3, 15))
    assert events == [] or events[0].date != "2026-03-19"
    assert problems or events[0].weekday_printed == "ఆది"


UTSAVAM_PAGE = """167 శ్రీ పరాభవనామ సంవత్సర సిద్ధాన్త పంచాంగం
తిరుపతి శ్రీ గోవిందరాజస్వామివారి పవిత్రోత్సవం
తేదీలు | వారము | ఉత్సవములు
30-07-2026 | గురు | అంకురార్పణ
31-08-2026 | శుక్ర | పవిత్రప్రతిష్ఠ
01-08-2026 | శని | పూర్ణాహుతి
"""


def test_utsavam_source_typo_is_corrected_by_continuity_and_reported() -> None:
    events, problems = parse_utsavam_schedules([(167, UTSAVAM_PAGE)])
    assert [e.date for e in events] == ["2026-07-30", "2026-07-31", "2026-08-01"]
    assert events[1].weekday_check == "corrected"
    assert all(e.title == "తిరుపతి శ్రీ గోవిందరాజస్వామివారి పవిత్రోత్సవం" for e in events)  # not the running header
    assert any("corrected" in p for p in problems)


def test_month_grid_positions_wrap_the_sixth_week_into_the_first_column() -> None:
    assert month_position(date(2027, 1, 1)) == (0, 5)  # Friday of week 0
    assert month_position(date(2027, 1, 3)) == (1, 0)  # Sunday of week 1
    assert month_position(date(2027, 1, 31)) == (0, 0)  # sixth week wraps above 3 Jan


def test_grid_fit_ignores_misread_numbers() -> None:
    # (value, x_center, y_top, height, width): true grid x0=890 dx=420, y0=1890 dy=325.
    good = [(v, 890 + c * 420, 1890 + r * 325, 100, 60) for v in (3, 10, 17, 24, 13, 22) for (c, r) in
            [month_position(date(2027, 1, v))]]  # fmt: skip
    noisy = [*good, (4, 900, 3513, 100, 60)]  # a "1" misread as "4"
    grid, agree = fit_grid(noisy, 2027, 1)
    assert abs(grid.x0 - 890) < 5 and abs(grid.dx - 420) < 5 and abs(grid.dy - 325) < 5
    assert agree == len(good)
    assert isinstance(grid, Grid)


def test_calendar_cell_text_parses_tithi_and_nakshatra() -> None:
    cell = parse_cell("' ఏకాదశి సా.6-08\nవిశాఖ రా.11-49\n1 తె,వ,4-07 ల 5-50", date(2027, 1, 3))
    assert (cell.tithi, cell.tithi_end, cell.nakshatra, cell.nakshatra_end) == (
        "ఏకాదశి",
        "సా.6-08",
        "విశాఖ",
        "రా.11-49",
    )
    full = parse_cell("అష్టమి ఉ.8-57\nవిశాఖ పూర్తి", date(2027, 1, 30))
    assert full.nakshatra_end == "పూర్తి"

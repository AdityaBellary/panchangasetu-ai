"""Rahukalam / yamagandam / gulika from the book's weekday table (p. 83), answered by template."""

from __future__ import annotations

from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path

import pytest

from panchangasetu.generation.answerer import AnswerService
from panchangasetu.generation.templates import can_template, render_day_answers
from panchangasetu.query.router import Intent, PanchangaMode
from panchangasetu.query.slots import extract_slots
from panchangasetu.structured.weekday_periods import WeekdayPeriods
from tests.unit.test_generation import NOW, FakeLLM, FakeRetriever

TABLE = WeekdayPeriods.load(Path(__file__).resolve().parents[2] / "config/weekday_periods.yaml")
TUESDAY = date(2026, 12, 15)


def periods() -> WeekdayPeriods:
    assert TABLE is not None
    return TABLE


def test_table_matches_the_printed_page() -> None:
    t = periods()
    assert t.kinds == {"rahukalam", "yamagandam", "gulika"}
    assert t.table["rahukalam"]["tue"][:2] == ["15:00", "16:30"]  # మంగళ 3 – 4½
    assert t.table["yamagandam"]["fri"][:2] == ["15:00", "16:30"]  # శుక్ర 3 – 4½
    assert t.table["gulika"]["sat"][:2] == ["06:00", "07:30"]  # శని 6 – 7½


def test_the_three_periods_never_overlap_on_any_weekday() -> None:
    t = periods()
    for n in range(7):
        day = TUESDAY + timedelta(days=n)
        spans = sorted((e.data["start"], e.data["end"]) for e in (t.evidence(k, day) for k in t.kinds))
        assert all(a[1] <= b[0] for a, b in pairwise(spans))


def test_weekday_period_answer_in_both_languages() -> None:
    ev = [periods().evidence("rahukalam", TUESDAY)]
    assert can_template(ev, ["rahukalam"])
    en = render_day_answers(ev, ["rahukalam"], "en")
    assert "Tuesday, 15 December 2026" in en and "Rahukalam: 3:00 PM to 4:30 PM." in en and "p. 83" in en
    te = render_day_answers(ev, ["rahukalam"], "te")
    assert "రాహుకాలం: మధ్యాహ్నం 3:00 నుండి సాయంత్రం 4:30 వరకు." in te


def test_missing_period_evidence_is_not_templated() -> None:
    assert not can_template([periods().evidence("rahukalam", TUESDAY)], ["rahukalam", "gulika"])


@pytest.mark.parametrize(
    ("question", "attr"),
    [
        ("gulika kalam today", "gulika"),
        ("ఈరోజు యమకంటకం ఎప్పుడు?", "yamagandam"),
        ("rahu kalam on Friday", "rahukalam"),
    ],
)
def test_period_names_are_recognised(question: str, attr: str) -> None:
    assert attr in extract_slots(question).attributes


def test_rahukalam_question_is_answered_without_the_llm() -> None:
    llm = FakeLLM(["{}"])
    retriever = FakeRetriever(
        [periods().evidence("rahukalam", NOW.date())], PanchangaMode.DAY, Intent.PANCHANGA, ["rahukalam"]
    )
    answer = AnswerService(retriever, llm).ask("What is today's rahukalam?", now=NOW)  # type: ignore[arg-type]
    assert answer.method == "template" and llm.calls == 0
    assert answer.citations[0].pages == [83]

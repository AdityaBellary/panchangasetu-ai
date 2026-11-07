"""Conversations: follow-up resolution, session storage and the multi-turn service."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from panchangasetu.conversation.followup import FollowUpResolver, needs_context, resolve_by_rules
from panchangasetu.conversation.service import ConversationService
from panchangasetu.conversation.store import (
    InMemorySessionStore,
    SessionNotFound,
    SqlSessionStore,
    TurnRecord,
)
from panchangasetu.core.interfaces import LLMError
from panchangasetu.generation.answerer import AnswerService
from panchangasetu.query.router import Intent, PanchangaMode
from panchangasetu.storage.models import Base
from tests.unit.test_generation import PROSE, FakeLLM, FakeRetriever, reply

NOW = datetime(2026, 11, 7, 9, 0)


def turn(q: str, cited: list[str] | None = None, at: datetime = NOW) -> TurnRecord:
    return TurnRecord(q, q, "none", "en", "", False, "llm", "r", at, [{"date": d} for d in cited or []])


def resolve(prev: str, q: str, cited: list[str] | None = None) -> str:
    return resolve_by_rules(q, [turn(prev, cited)], NOW.date()).question


# ------------------------------------------------------------------ rules
@pytest.mark.parametrize(
    ("prev", "follow_up", "expected"),
    [
        ("What is today's tithi?", "and tomorrow?", "What is tomorrow's tithi?"),
        ("ఈరోజు తిథి ఏమిటి?", "రేపు?", "రేపు తిథి ఏమిటి?"),
        ("What is today's tithi?", "and the nakshatram?", "What is today's nakshatram?"),
        (
            "What is the varjyam on 15 December 2026?",
            "and rahukalam?",
            "What is the rahukalam on 15 December 2026?",
        ),
        ("When is the next Ekadashi?", "and Purnima?", "When is the next Purnima?"),
        ("వచ్చే ఏకాదశి ఎప్పుడు?", "దాని ప్రాముఖ్యత ఏమిటి?", "ఏకాదశి ప్రాముఖ్యత ఏమిటి?"),
        ("When is Deepavali?", "why is it celebrated?", "why is Deepavali celebrated?"),
        (
            "మకర రాశి వారికి ఉద్యోగ ఫలితాలు ఎలా ఉన్నాయి?",
            "కన్యా రాశికి?",
            "కన్యా రాశి వారికి ఉద్యోగ ఫలితాలు ఎలా ఉన్నాయి?",
        ),
        (
            "What are the yearly predictions for Kanya rasi?",
            "and Tula?",
            "What are the yearly predictions for Tula?",
        ),
        ("When is Deepavali?", "ఎప్పుడు?", "Deepavali ఎప్పుడు"),
    ],
)
def test_rules_rewrite_follow_ups_from_the_previous_frame(prev: str, follow_up: str, expected: str) -> None:
    assert resolve(prev, follow_up) == expected


def test_that_day_becomes_the_date_the_previous_answer_cited() -> None:
    out = resolve("When is the next Ekadashi?", "what is the nakshatram that day?", cited=["2026-11-20"])
    assert out == "what is the nakshatram on 20 November 2026?"
    te = resolve("వచ్చే ఏకాదశి ఎప్పుడు?", "ఆ రోజు నక్షత్రం ఏమిటి?", cited=["2026-11-20"])
    assert te == "20 నవంబర్ 2026 నక్షత్రం ఏమిటి?"


@pytest.mark.parametrize(
    "question",
    [
        "What is Kuja dosha?",
        "When is the next Ekadashi?",
        "What is the tithi on 15 December 2026?",
        "దీపావళి ఎప్పుడు?",
    ],
)
def test_questions_with_their_own_subject_are_never_rewritten(question: str) -> None:
    res = resolve_by_rules(question, [turn("What is today's tithi?")], NOW.date())
    assert res.method == "none" and res.question == question


def test_short_prose_topic_is_carried_by_rules() -> None:
    assert resolve("What is Kuja dosha?", "what are its remedies?") == "what are Kuja dosha's remedies?"


def test_no_history_means_no_rewrite() -> None:
    assert resolve_by_rules("and tomorrow?", [], NOW.date()).method == "none"


def test_needs_context_detects_pronouns_and_bare_follow_ups() -> None:
    assert needs_context("what are its remedies?", NOW.date())
    assert needs_context("and?", NOW.date())
    assert not needs_context("What is Kuja dosha?", NOW.date())


# ------------------------------------------------------------------ LLM fallback
def test_llm_rewrites_what_rules_cannot_place() -> None:
    llm = FakeLLM([json.dumps({"question": "What are the remedies for Kuja dosha?"})])
    history = [turn("What is Kuja dosha and when is it not a problem for marriage?")]
    res = FollowUpResolver(llm).resolve("what are its remedies?", history, NOW)
    assert (res.method, res.question) == ("llm", "What are the remedies for Kuja dosha?")


def test_llm_is_not_called_when_rules_suffice_or_question_stands_alone() -> None:
    llm = FakeLLM([json.dumps({"question": "x"})])
    resolver = FollowUpResolver(llm)
    resolver.resolve("and tomorrow?", [turn("What is today's tithi?")], NOW)
    resolver.resolve("What is Kuja dosha?", [turn("What is today's tithi?")], NOW)
    assert llm.calls == 0


def test_llm_rewrite_in_another_language_is_rejected() -> None:
    llm = FakeLLM([json.dumps({"question": "What are its remedies for Kuja dosha?"})])
    history = [turn("What is Kuja dosha and when is it not a problem for marriage?")]
    res = FollowUpResolver(llm).resolve("దాని నివారణలు ఏమిటి?", history, NOW)
    assert res.method == "none" and res.question == "దాని నివారణలు ఏమిటి?"


def test_llm_failure_falls_back_to_the_question_as_typed() -> None:
    llm = FakeLLM([LLMError("out of credits", retryable=False, quota=True)])
    history = [turn("What is Kuja dosha and when is it not a problem for marriage?")]
    res = FollowUpResolver(llm).resolve("what are its remedies?", history, NOW)
    assert res.method == "none"


# ------------------------------------------------------------------ stores
def _sql_store() -> SqlSessionStore:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return SqlSessionStore(sessionmaker(engine, expire_on_commit=False))


@pytest.mark.parametrize("make", [InMemorySessionStore, _sql_store])
def test_store_keeps_turns_in_order_per_session(make: type) -> None:
    store = make()
    a, b = store.create("client-a"), store.create()
    for i in range(4):
        store.add_turn(a, turn(f"q{i}"))
    store.add_turn(b, turn("other"))
    assert [t.question for t in store.recent_turns(a, 2)] == ["q2", "q3"]
    assert [t.question for t in store.recent_turns(b, 5)] == ["other"]
    assert store.exists(a, "client-a") and not store.exists(a, "client-b")
    assert not store.exists("not-a-uuid")
    with pytest.raises(SessionNotFound):
        store.recent_turns("00000000-0000-0000-0000-000000000000", 3)
    store.create("client-a")  # empty sessions are not listed
    listed = store.list_sessions("client-a")
    assert [(s.id, s.title, s.turns) for s in listed] == [(a, "q0", 4)]
    assert [s.id for s in store.list_sessions(None)] == [b]
    assert not store.delete(a, "client-b") and store.delete(a, "client-a")
    assert not store.exists(a) and store.list_sessions("client-a") == []


# ------------------------------------------------------------------ service
def _service(llm: FakeLLM, retriever: FakeRetriever) -> ConversationService:
    return ConversationService(AnswerService(retriever, llm), InMemorySessionStore(), FollowUpResolver(llm))  # type: ignore[arg-type]


def test_service_answers_the_standalone_question_in_the_follow_ups_language() -> None:
    retriever = FakeRetriever([PROSE], PanchangaMode.NONE, Intent.GENERAL, [])
    llm = FakeLLM([reply("Kuja dosha does not apply when Mars is with Jupiter.")])
    svc = _service(llm, retriever)
    first = svc.ask("When is Deepavali?", now=NOW)
    assert first.session_id and first.standalone_question == "When is Deepavali?"
    second = svc.ask("why is it celebrated?", session_id=first.session_id, now=NOW)
    assert retriever.last_question == "why is Deepavali celebrated?"
    assert (
        second.question == "why is it celebrated?"
        and second.standalone_question == "why is Deepavali celebrated?"
    )
    history = svc.store.recent_turns(first.session_id, 5)
    assert [t.rewrite_method for t in history] == ["none", "rule"]


def test_service_ignores_turns_after_the_idle_window() -> None:
    retriever = FakeRetriever([PROSE], PanchangaMode.NONE, Intent.GENERAL, [])
    svc = _service(FakeLLM([reply("x")]), retriever)
    sid = svc.ask("When is Deepavali?", now=NOW).session_id
    svc.ask("why is it celebrated?", session_id=sid, now=NOW + timedelta(hours=3))
    assert retriever.last_question == "why is it celebrated?"


def test_service_rejects_unknown_or_foreign_sessions() -> None:
    svc = _service(FakeLLM([reply("x")]), FakeRetriever([PROSE], PanchangaMode.NONE, Intent.GENERAL, []))
    with pytest.raises(SessionNotFound):
        svc.ask("hi", session_id="nope", now=NOW)
    sid = svc.start(owner="client-a")
    with pytest.raises(SessionNotFound):
        svc.ask("hi", session_id=sid, owner="client-b", now=NOW)

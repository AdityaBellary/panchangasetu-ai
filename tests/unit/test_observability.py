"""Observability: per-call recording and pricing, Gemini request shape, stored traces, ops queries,
and admin-only monitoring endpoints."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from panchangasetu.api.app import ApiServices, create_app
from panchangasetu.api.keys import Client
from panchangasetu.conversation.followup import FollowUpResolver
from panchangasetu.conversation.service import ConversationService
from panchangasetu.conversation.store import SqlSessionStore
from panchangasetu.core.interfaces import ChatMessage, LLMResult
from panchangasetu.generation.answerer import AnswerService
from panchangasetu.observability.calls import PriceTable, call_purpose, collect_calls
from panchangasetu.observability.ops import OpsQueries, percentile
from panchangasetu.providers.llm.openai_compatible import OpenAICompatibleChat, gemini_chat
from panchangasetu.query.router import Intent, PanchangaMode
from panchangasetu.storage.models import Base, LlmCallRecord
from tests.unit.test_generation import PROSE, FakeLLM, FakeRetriever, reply

REPO = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 11, 7, 9, 0, tzinfo=UTC)
MSG = [ChatMessage("user", "hi")]


# ------------------------------------------------------------------ pricing and recording
def test_price_table_converts_usd_and_matches_model_variants() -> None:
    prices = PriceTable.load(REPO / "config/llm_pricing.yaml")
    assert prices.cost("sarvam-105b-conversations", 1_000_000, 0) == 15.0
    assert prices.cost("gemini-3.8-flash", 1_000_000, 1_000_000) == pytest.approx((0.75 + 3.75) * 88.0)
    assert prices.cost("gemini-3.8-flash-001", 1000, 0) == prices.cost("gemini-3.8-flash", 1000, 0)
    assert prices.cost("models/gemini-3.8-flash", 1000, 0) == prices.cost("gemini-3.8-flash", 1000, 0)
    assert prices.cost("never-heard-of-it", 1_000_000, 0) == 15.0  # default: costed like Sarvam


def _chat(handler: httpx.MockTransport, **kw: object) -> OpenAICompatibleChat:
    return OpenAICompatibleChat(
        name="sarvam",
        model="sarvam-105b-conversations",
        base_url="https://x",
        headers={},
        http=httpx.Client(base_url="https://x", transport=handler),
        pricing=PriceTable({"sarvam-105b-conversations": (15.0, 60.0)}),
        **kw,  # type: ignore[arg-type]
    )


def test_every_attempt_is_recorded_with_purpose_tokens_and_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda _s: None)
    replies = iter([httpx.Response(503), httpx.Response(200, json={
        "choices": [{"message": {"content": "{}"}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
        "model": "sarvam-105b-conversations",
    })])  # fmt: skip
    chat = _chat(httpx.MockTransport(lambda _r: next(replies)))
    with collect_calls() as calls, call_purpose("answer"):
        chat.complete(MSG)
    assert [(c.status, c.purpose) for c in calls] == [("error", "answer"), ("ok", "answer")]
    assert calls[1].prompt_tokens == 1000 and calls[1].cost_inr == pytest.approx(0.021)


def test_gemini_asks_for_low_reasoning_and_no_json_mode() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}], "usage": {}})

    chat = gemini_chat("k", http=httpx.Client(base_url="https://g", transport=httpx.MockTransport(handler)))
    chat.complete(MSG, json=True)
    assert seen["reasoning_effort"] == "low" and "response_format" not in seen
    assert seen["model"] == "gemini-3.8-flash" and chat.provider == "gemini"


def test_percentile_interpolates() -> None:
    assert percentile([100, 200, 300, 400], 0.5) == 250.0
    assert percentile([], 0.5) is None


# ------------------------------------------------------------------ stored traces and ops queries
def _stack(llm: FakeLLM) -> tuple[ConversationService, OpsQueries, sessionmaker]:  # type: ignore[type-arg]
    # One shared connection: the API test client calls from other threads.
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    retriever = FakeRetriever([PROSE], PanchangaMode.NONE, Intent.GENERAL, [])
    svc = ConversationService(AnswerService(retriever, llm), SqlSessionStore(factory), FollowUpResolver(None))  # type: ignore[arg-type]
    return svc, OpsQueries(factory, "UTC", daily_budget_inr=25), factory


def test_retry_is_recorded_as_its_own_call_and_traced() -> None:
    bad = json.dumps(
        {"sufficient": True, "answer": "Mars in the 11th house.", "citations": ["S1"], "quotes": []}
    )
    llm = FakeLLM([bad, reply("Kuja dosha does not apply when Mars is with Jupiter.")])
    svc, ops, _ = _stack(llm)
    answer = svc.ask("What is Kuja dosha?", owner="client-a", now=NOW)
    assert [c.purpose for c in answer.llm_calls] == ["answer", "answer_retry"]
    assert answer.cost_inr == pytest.approx(0.042) and answer.prompt_tokens == 2000
    assert {"retrieve", "rewrite", "total", "llm_1", "llm_2"} <= set(answer.latency_ms)

    [row] = ops.requests()
    assert (row["method"], row["llm_calls"], row["cost_inr"]) == ("llm", 2, pytest.approx(0.042))
    trace = ops.trace(row["id"])
    assert trace is not None
    assert [c["purpose"] for c in trace["llm_calls"]] == ["answer", "answer_retry"]
    assert trace["sources"][0]["pages"] == [74] and "llm_2" in trace["timings"]

    o = ops.overview(hours=24, now=NOW + timedelta(minutes=5))
    assert o["kpis"]["questions"] == 1 and o["kpis"]["llm_calls"] == 2
    assert o["models"][0]["purposes"] == {"answer": 1, "answer_retry": 1}
    assert o["budget"]["spent_today_inr"] == pytest.approx(0.042)


def test_llm_calls_survive_deleting_the_conversation() -> None:
    svc, ops, factory = _stack(FakeLLM([reply("Kuja dosha does not apply when Mars is with Jupiter.")]))
    a = svc.ask("What is Kuja dosha?", now=NOW)
    assert svc.store.delete(a.session_id or "")
    with factory() as s:
        [call] = s.scalars(select(LlmCallRecord)).all()
        assert call.turn_id is None and call.cost_inr == pytest.approx(0.021)
    assert ops.overview(hours=24, now=NOW)["budget"]["spent_today_inr"] == pytest.approx(0.021)


# ------------------------------------------------------------------ admin endpoints
class Keys:
    def authenticate(self, key: str) -> Client | None:
        return {"psk_admin": Client("a", "ops", None, True), "psk_user": Client("u", "web", None, False)}.get(
            key
        )


def test_monitoring_endpoints_need_an_admin_key() -> None:
    svc, ops, _ = _stack(FakeLLM([reply("x")]))
    api = TestClient(create_app(ApiServices(conversations=svc, keys=Keys(), ops=ops)))
    assert api.get("/v1/admin/overview", headers={"X-API-Key": "psk_user"}).status_code == 403
    assert api.get("/v1/whoami", headers={"X-API-Key": "psk_user"}).json() == {
        "name": "web",
        "is_admin": False,
    }
    ok = api.get("/v1/admin/overview?hours=24", headers={"X-API-Key": "psk_admin"})
    assert ok.status_code == 200 and "kpis" in ok.json()
    assert api.get("/v1/admin/requests/not-a-uuid", headers={"X-API-Key": "psk_admin"}).status_code == 404
    assert api.get("/v1/admin/calls", headers={"X-API-Key": "psk_admin"}).json() == []
    assert api.get("/dashboard.html").status_code == 200


def test_gemini_thinking_tokens_are_billed_as_output() -> None:
    from panchangasetu.providers.llm.openai_compatible import _billed

    assert _billed({"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 66}) == (7, 59)
    assert _billed({"prompt_tokens": 10, "completion_tokens": 5}) == (10, 5)  # no total reported


def test_gemini_quota_errors_fall_through_without_retrying() -> None:
    message = "You exceeded your current quota, please check your plan and billing details."
    body = [{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": message}}]
    hits = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(1)
        return httpx.Response(429, json=body)

    chat = _chat(httpx.MockTransport(handler), max_retries=2)
    with collect_calls() as calls, pytest.raises(Exception) as exc:
        chat.complete(MSG)
    assert getattr(exc.value, "quota", False) and len(hits) == 1
    assert calls[0].status == "quota"


# ------------------------------------------------------------------ escalation to a stronger model
class EscalationLLM(FakeLLM):
    """A second model; its calls are recorded under its own model name."""

    def __init__(self, replies: list[str | Exception], model: str = "gemini-3.6-flash") -> None:
        super().__init__(replies)
        self.model = model


def test_refusal_from_the_primary_is_escalated_and_traced() -> None:
    escalation = EscalationLLM([reply("Kuja dosha does not apply when Mars is with Jupiter.")])
    retriever = FakeRetriever([PROSE], PanchangaMode.NONE, Intent.GENERAL, [])
    svc = AnswerService(retriever, FakeLLM([reply("", sufficient=False)]), escalation=escalation)  # type: ignore[arg-type]
    answer = svc.ask("What is Kuja dosha?", now=NOW)
    assert not answer.abstained and escalation.calls == 1
    assert [c.purpose for c in answer.llm_calls] == ["answer", "escalation"]


def test_no_second_try_when_the_primary_chain_already_used_the_escalation_model() -> None:
    # Sarvam out of credits -> the primary chain answered with Gemini; asking Gemini again is waste.
    escalation = EscalationLLM([reply("x")], model="fake-model")
    retriever = FakeRetriever([PROSE], PanchangaMode.NONE, Intent.GENERAL, [])
    svc = AnswerService(retriever, FakeLLM([reply("", sufficient=False)]), escalation=escalation)  # type: ignore[arg-type]
    assert svc.ask("What is Kuja dosha?", now=NOW).abstained and escalation.calls == 0


def test_escalation_uses_the_gemini_chain_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from panchangasetu.config import Settings
    from panchangasetu.providers.llm.factory import build_escalation_llm

    for var in ("SARVAM_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    llm = build_escalation_llm(settings)
    assert llm is not None
    assert [p.model for p in llm.providers] == [settings.gemini_chat_model, *settings.gemini_fallback_models]
    assert build_escalation_llm(Settings(_env_file=None, escalation_provider="")) is None  # type: ignore[call-arg]
    monkeypatch.delenv("GEMINI_API_KEY")
    assert build_escalation_llm(Settings(_env_file=None)) is None  # type: ignore[call-arg]  # no key


# ------------------------------------------------------------------ spend is never lost
def test_deleting_the_session_mid_answer_still_counts_the_spend() -> None:
    from panchangasetu.conversation.store import SessionNotFound

    class DeletesSession(FakeLLM):
        def complete(self, *a: object, **kw: object) -> LLMResult:  # type: ignore[override]
            svc.store.delete(sid)  # the client deletes the conversation while it is being answered
            return super().complete(*a, **kw)  # type: ignore[arg-type]

    svc, ops, _ = _stack(DeletesSession([reply("Kuja dosha does not apply when Mars is with Jupiter.")]))
    sid = svc.start("client-a")
    with pytest.raises(SessionNotFound):
        svc.ask("What is Kuja dosha?", session_id=sid, owner="client-a", now=NOW)
    assert svc.spent_today(NOW) == pytest.approx(0.021)
    [call] = ops.calls()
    assert call["turn_id"] is None and call["purpose"] == "answer"


def test_a_failure_after_a_paid_call_still_counts_the_spend() -> None:
    class Breaks(FakeLLM):
        def complete(self, *a: object, **kw: object) -> LLMResult:  # type: ignore[override]
            super().complete(*a, **kw)  # type: ignore[arg-type]
            raise RuntimeError("bug after the call was paid for")

    svc, _, _ = _stack(Breaks([reply("x")]))
    with pytest.raises(RuntimeError):
        svc.ask("What is Kuja dosha?", now=NOW)
    assert svc.spent_today(NOW) == pytest.approx(0.021)

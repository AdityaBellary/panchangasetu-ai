"""HTTP API: auth, session ownership, rate limits, validation, page images, UI serving."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from panchangasetu.api.app import ApiServices, RateLimiter, create_app
from panchangasetu.api.keys import ApiKeyStore, Client
from panchangasetu.api.today import tithi_number
from panchangasetu.conversation.followup import FollowUpResolver
from panchangasetu.conversation.service import ConversationService
from panchangasetu.conversation.store import InMemorySessionStore
from panchangasetu.generation.answerer import AnswerService
from panchangasetu.query.router import Intent, PanchangaMode
from panchangasetu.storage.models import ApiKey, Base
from tests.unit.test_generation import NOW, PROSE, FakeLLM, FakeRetriever, reply

KEYS = {
    "psk_alice": Client("alice", "alice", None),
    "psk_bob": Client("bob", "bob", None),
    "psk_carol": Client("carol", "carol", 2),
}


class FakeKeys:
    def authenticate(self, key: str) -> Client | None:
        return KEYS.get(key)


class FakePages:
    def __init__(self, path: Path) -> None:
        self.file = path

    def path(self, document: str, page: int) -> Path | None:
        return self.file if (document, page) == ("book.pdf", 74) else None


@pytest.fixture
def api(tmp_path: Path) -> TestClient:
    png = tmp_path / "p74.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    retriever = FakeRetriever([PROSE], PanchangaMode.NONE, Intent.GENERAL, [])
    retriever.clock = lambda: NOW
    llm = FakeLLM([reply("Kuja dosha does not apply when Mars is with Jupiter.")])
    conversations = ConversationService(
        AnswerService(retriever, llm),  # type: ignore[arg-type]
        InMemorySessionStore(),
        FollowUpResolver(None),
    )
    services = ApiServices(
        conversations=conversations,
        keys=FakeKeys(),
        pages=FakePages(png),
        readiness=lambda: {"database": True, "vector_index": False},
        today=lambda: {"date": "2026-11-07", "available": False},
        rate_limit_per_minute=100,
        max_question_chars=50,
    )
    return TestClient(create_app(services))


ALICE = {"Authorization": "Bearer psk_alice"}
BOB = {"X-API-Key": "psk_bob"}


def test_v1_requires_a_valid_key(api: TestClient) -> None:
    assert api.post("/v1/chat", json={"question": "hi"}).status_code == 401
    assert (
        api.post("/v1/chat", json={"question": "hi"}, headers={"Authorization": "Bearer nope"}).status_code
        == 401
    )
    assert api.get("/v1/today", headers=BOB).status_code == 200  # X-API-Key works too


def test_chat_continues_a_session_and_returns_citations(api: TestClient) -> None:
    first = api.post("/v1/chat", json={"question": "When is Deepavali?"}, headers=ALICE).json()
    assert first["session_id"] and not first["abstained"] and first["method"] == "llm"
    assert first["citations"][0]["page_image_urls"] == ["/v1/pages/book.pdf/74"]
    second = api.post(
        "/v1/chat",
        json={"question": "why is it celebrated?", "session_id": first["session_id"]},
        headers=ALICE,
    ).json()
    assert second["session_id"] == first["session_id"]
    assert second["standalone_question"] == "why is Deepavali celebrated?"
    detail = api.get(f"/v1/sessions/{first['session_id']}", headers=ALICE).json()
    assert [t["question"] for t in detail["turns"]] == ["When is Deepavali?", "why is it celebrated?"]


def test_sessions_belong_to_the_key_that_created_them(api: TestClient) -> None:
    sid = api.post("/v1/chat", json={"question": "When is Deepavali?"}, headers=ALICE).json()["session_id"]
    assert api.get(f"/v1/sessions/{sid}", headers=BOB).status_code == 404
    assert api.post("/v1/chat", json={"question": "and?", "session_id": sid}, headers=BOB).status_code == 404
    assert api.delete(f"/v1/sessions/{sid}", headers=BOB).status_code == 404
    assert api.get("/v1/sessions", headers=BOB).json() == []
    listed = api.get("/v1/sessions", headers=ALICE).json()
    assert [(s["id"], s["title"], s["turns"]) for s in listed] == [(sid, "When is Deepavali?", 1)]
    assert api.delete(f"/v1/sessions/{sid}", headers=ALICE).status_code == 204
    assert api.get("/v1/sessions", headers=ALICE).json() == []


def test_questions_are_validated(api: TestClient) -> None:
    assert api.post("/v1/chat", json={"question": "   "}, headers=ALICE).status_code == 422
    assert api.post("/v1/chat", json={"question": "x" * 51}, headers=ALICE).status_code == 422
    assert api.post("/v1/chat", json={}, headers=ALICE).status_code == 422


def test_per_key_rate_limit(api: TestClient) -> None:
    carol = {"X-API-Key": "psk_carol"}
    codes = [api.get("/v1/today", headers=carol).status_code for _ in range(3)]  # carol's key: 2 per minute
    assert codes == [200, 200, 429]
    assert api.get("/v1/today", headers=ALICE).status_code == 200  # limits are per key


def test_rate_limiter_window_slides() -> None:
    rl = RateLimiter()
    assert [rl.allow("k", 2, now=t) for t in (0, 1, 2)] == [True, True, False]
    assert rl.allow("k", 2, now=61)


def test_page_images_are_served_only_when_stored(api: TestClient) -> None:
    ok = api.get("/v1/pages/book.pdf/74", headers=ALICE)
    assert ok.status_code == 200 and ok.headers["content-type"] == "image/png"
    assert api.get("/v1/pages/book.pdf/75", headers=ALICE).status_code == 404
    assert api.get("/v1/pages/book.pdf/74").status_code == 401


def test_health_readiness_and_ui(api: TestClient) -> None:
    assert api.get("/healthz").json() == {"status": "ok"}
    ready = api.get("/readyz")
    assert ready.status_code == 503 and ready.json()["checks"]["vector_index"] is False
    ui = api.get("/")
    assert ui.status_code == 200 and "PanchangaSetu" in ui.text
    assert "default-src 'self'" in ui.headers["content-security-policy"]
    assert api.get("/app.js").status_code == 200


def test_api_keys_are_stored_hashed_and_can_be_revoked() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    store = ApiKeyStore(factory)
    key, info = store.create("web")
    assert key.startswith("psk_") and len(key) > 40
    with factory() as s:
        stored = s.scalars(select(ApiKey)).one()
        assert key not in (stored.key_hash, stored.prefix) and stored.key_hash != key
    client = store.authenticate(key)
    assert client is not None and client.name == "web"
    assert store.authenticate(key + "x") is None and store.authenticate("not-a-key") is None
    assert store.revoke(info.prefix) == 1
    assert store.authenticate(key) is None


@pytest.mark.parametrize(
    ("name", "paksha", "number"),
    [("పాడ్యమి", "శుద్ధ", 1), ("పూర్ణిమ", None, 15), ("చతుర్దశి", "బహుళ", 29), ("అమావాస్య", "బహుళ", 30)],
)
def test_tithi_numbers_drive_the_moon_phase(name: str, paksha: str | None, number: int) -> None:
    assert tithi_number(name, paksha) == number

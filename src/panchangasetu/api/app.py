"""HTTP API and web chat UI.

  GET  /                      web chat UI (static; it asks for an API key and keeps it in the browser)
  GET  /healthz               liveness (no auth)
  GET  /readyz                database and vector index reachable (no auth)
  GET  /metrics               Prometheus metrics (bearer token when PSETU_API_METRICS_TOKEN is set)
  POST /v1/chat               ask a question, optionally continuing a session
  GET  /v1/sessions           this client's conversations, most recent first
  GET  /v1/sessions/{id}      one conversation's turns
  DELETE /v1/sessions/{id}
  GET  /v1/pages/{document}/{page}   the source page image a citation points to
  GET  /v1/today              today's tithi (with moon phase), nakshatra and rahukalam
  GET  /v1/admin/...          monitoring (admin keys): overview, request log, traces, LLM calls
  GET  /dashboard.html        monitoring dashboard (static; uses the admin endpoints)

Every /v1 call needs `Authorization: Bearer <key>` (or `X-API-Key`). Sessions belong to the key
that created them; another key gets 404, never someone else's conversation. Answering is
synchronous (local embeddings + one LLM call), so endpoints are plain `def` and run in FastAPI's
thread pool.
"""

import hmac
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, Protocol
from urllib.parse import quote

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from panchangasetu.api.keys import Client
from panchangasetu.conversation.service import ConversationService
from panchangasetu.conversation.store import SessionNotFound
from panchangasetu.generation.answerer import Answer
from panchangasetu.observability.context import request_id_var
from panchangasetu.observability.logging import get_logger
from panchangasetu.observability.metrics import HTTP_REQUESTS, HTTP_SECONDS, REGISTRY
from panchangasetu.observability.ops import OpsQueries

log = get_logger(__name__)
STATIC_DIR = Path(__file__).parent / "static"
_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})


class Authenticator(Protocol):
    def authenticate(self, key: str) -> Client | None: ...


class PageImages(Protocol):
    def path(self, document: str, page: int) -> Path | None: ...


@dataclass
class ApiServices:
    conversations: ConversationService
    keys: Authenticator
    pages: PageImages | None = None
    readiness: Callable[[], dict[str, bool]] | None = None
    today: Callable[[], dict[str, Any]] | None = None
    ops: OpsQueries | None = None  # monitoring queries (admin keys only)
    rate_limit_per_minute: int = 30
    max_question_chars: int = 500
    metrics_token: str | None = None


# ------------------------------------------------------------------ schemas
class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str | None = None


class CitationOut(BaseModel):
    document: str
    pages: list[int]
    date: str | None
    verified: bool | None
    snippet: str
    quotes: list[str]
    page_image_urls: list[str]


class ChatResponse(BaseModel):
    session_id: str
    question: str
    standalone_question: str | None
    answer: str
    language: str
    abstained: bool
    method: str
    route: str
    citations: list[CitationOut]
    model: str | None
    cost_inr: float
    latency_ms: float


class TurnOut(BaseModel):
    question: str
    standalone_question: str
    answer: str
    language: str
    abstained: bool
    method: str
    citations: list[CitationOut]
    asked_at: str


class SessionOut(BaseModel):
    id: str
    title: str
    turns: int
    created_at: str
    updated_at: str


class SessionDetail(BaseModel):
    id: str
    turns: list[TurnOut]


# ------------------------------------------------------------------ helpers
@dataclass
class RateLimiter:
    """Sliding one-minute window per client, in process memory (one API process per deployment;
    a shared store is a Phase 8 concern if the API is scaled out)."""

    hits: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(deque))
    lock: Lock = field(default_factory=Lock)

    def allow(self, client: str, limit: int, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self.lock:
            q = self.hits[client]
            while q and now - q[0] >= 60:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            return True


def _page_urls(document: str, pages: list[int]) -> list[str]:
    return [f"/v1/pages/{quote(document)}/{p}" for p in pages[:3]]


def _citations(raw: list[dict[str, Any]]) -> list[CitationOut]:
    return [
        CitationOut(
            document=str(c.get("document", "")),
            pages=list(c.get("pages") or []),
            date=c.get("date"),
            verified=c.get("verified"),
            snippet=str(c.get("snippet", "")),
            quotes=list(c.get("quotes") or []),
            page_image_urls=_page_urls(str(c.get("document", "")), list(c.get("pages") or [])),
        )
        for c in raw
    ]


def _chat_response(a: Answer) -> ChatResponse:
    from dataclasses import asdict

    return ChatResponse(
        session_id=a.session_id or "",
        question=a.question,
        standalone_question=a.standalone_question,
        answer=a.answer,
        language=a.language,
        abstained=a.abstained,
        method=a.method,
        route=a.route,
        citations=_citations([asdict(c) for c in a.citations]),
        model=a.model,
        cost_inr=a.cost_inr,
        latency_ms=round(a.latency_ms.get("total", 0.0), 1),
    )


# ------------------------------------------------------------------ app
def create_app(services: ApiServices) -> FastAPI:
    app = FastAPI(
        title="PanchangaSetu AI",
        version="0.7.0",
        description="Grounded Telugu/English answers from TTD panchangam documents, with citations.",
    )
    limiter = RateLimiter()

    @app.middleware("http")
    async def request_context(request: Request, call_next: Callable[[Request], Any]) -> Response:
        rid = (request.headers.get("x-request-id") or uuid.uuid4().hex)[:32]
        request_id_var.set(rid)  # links the stored turn and its LLM calls to this request
        t0 = time.perf_counter()
        try:
            response: Response = await call_next(request)
        except Exception:  # pragma: no cover - last-resort guard; details stay in the log
            log.exception("api.unhandled", request_id=rid, path=request.url.path)
            response = JSONResponse({"detail": "internal error", "request_id": rid}, status_code=500)
        response.headers["x-request-id"] = rid
        response.headers["x-content-type-options"] = "nosniff"
        response.headers["referrer-policy"] = "no-referrer"
        if not request.url.path.startswith("/v1/"):
            response.headers["content-security-policy"] = (
                "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; "
                "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
            )
        elapsed = time.perf_counter() - t0
        if request.url.path.startswith("/v1/"):
            # Route template, not the raw path: session ids and document names must not become labels.
            route = getattr(request.scope.get("route"), "path", "unmatched")
            HTTP_REQUESTS.labels(request.method, route, str(response.status_code)).inc()
            HTTP_SECONDS.labels(route).observe(elapsed)
            log.info(
                "api.request",
                request_id=rid,
                method=request.method,
                path=route,
                status=response.status_code,
                ms=round(elapsed * 1000, 1),
            )
        return response

    def client(
        authorization: Annotated[str | None, Header()] = None,
        x_api_key: Annotated[str | None, Header()] = None,
    ) -> Client:
        key = x_api_key or (
            authorization[7:].strip() if authorization and authorization[:7].lower() == "bearer " else None
        )
        found = services.keys.authenticate(key) if key else None
        if found is None:
            raise HTTPException(401, "missing or invalid API key", headers={"WWW-Authenticate": "Bearer"})
        if not limiter.allow(found.id, found.rate_limit_per_minute or services.rate_limit_per_minute):
            raise HTTPException(
                429, "rate limit exceeded; try again in a minute", headers={"Retry-After": "60"}
            )
        return found

    Authed = Annotated[Client, Depends(client)]

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics", include_in_schema=False)
    def metrics(request: Request, authorization: Annotated[str | None, Header()] = None) -> Response:
        # Prometheus scrape endpoint. With a token configured, scrapers must send it as a bearer token;
        # without one, only clients on this machine (loopback) may read it.
        if services.metrics_token:
            expected = f"Bearer {services.metrics_token}".encode()
            if not hmac.compare_digest((authorization or "").encode(), expected):
                raise HTTPException(401, "metrics token required", headers={"WWW-Authenticate": "Bearer"})
        elif (request.client.host if request.client else "") not in _LOOPBACK:
            raise HTTPException(403, "set PSETU_API_METRICS_TOKEN to scrape metrics from another host")
        return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

    @app.get("/readyz", include_in_schema=False)
    def readyz() -> JSONResponse:
        checks = services.readiness() if services.readiness else {}
        ok = all(checks.values())
        return JSONResponse({"ready": ok, "checks": checks}, status_code=200 if ok else 503)

    def admin(who: Authed) -> Client:
        if not who.is_admin:
            raise HTTPException(403, "monitoring needs an admin key (psetu api-keys create NAME --admin)")
        if services.ops is None:
            raise HTTPException(404, "monitoring is not configured")
        return who

    Admin = Annotated[Client, Depends(admin)]

    @app.get("/v1/whoami")
    def whoami(who: Authed) -> dict[str, Any]:
        return {"name": who.name, "is_admin": who.is_admin}

    @app.get("/v1/admin/overview")
    def admin_overview(_: Admin, hours: int = 24) -> dict[str, Any]:
        assert services.ops is not None
        return services.ops.overview(min(max(hours, 1), 24 * 90))

    @app.get("/v1/admin/requests")
    def admin_requests(
        _: Admin, limit: int = 50, offset: int = 0, method: str | None = None
    ) -> list[dict[str, Any]]:
        assert services.ops is not None
        return services.ops.requests(min(max(limit, 1), 500), max(offset, 0), method)

    @app.get("/v1/admin/requests/{turn_id}")
    def admin_trace(turn_id: str, _: Admin) -> dict[str, Any]:
        assert services.ops is not None
        trace = services.ops.trace(turn_id)
        if trace is None:
            raise HTTPException(404, "request not found")
        return trace

    @app.get("/v1/admin/calls")
    def admin_calls(_: Admin, limit: int = 100, errors: bool = False) -> list[dict[str, Any]]:
        assert services.ops is not None
        return services.ops.calls(min(max(limit, 1), 1000), errors)

    @app.get("/v1/today")
    def today(who: Authed) -> dict[str, Any]:
        if services.today is None:
            raise HTTPException(404, "not available")
        return services.today()

    @app.post("/v1/chat", response_model=ChatResponse)
    def chat(body: ChatRequest, who: Authed) -> ChatResponse:
        question = body.question.strip()
        if not question:
            raise HTTPException(422, "question is empty")
        if len(question) > services.max_question_chars:
            raise HTTPException(422, f"question is longer than {services.max_question_chars} characters")
        try:
            answer = services.conversations.ask(question, session_id=body.session_id, owner=who.id)
        except SessionNotFound as exc:
            raise HTTPException(404, "session not found") from exc
        return _chat_response(answer)

    @app.get("/v1/sessions", response_model=list[SessionOut])
    def sessions(who: Authed, limit: int = 50) -> list[SessionOut]:
        rows = services.conversations.store.list_sessions(who.id, min(max(limit, 1), 200))
        return [
            SessionOut(
                id=s.id,
                title=s.title,
                turns=s.turns,
                created_at=s.created_at.isoformat(),
                updated_at=s.updated_at.isoformat(),
            )
            for s in rows
        ]

    @app.get("/v1/sessions/{session_id}", response_model=SessionDetail)
    def session(session_id: str, who: Authed) -> SessionDetail:
        store = services.conversations.store
        if not store.exists(session_id, who.id):
            raise HTTPException(404, "session not found")
        turns = store.recent_turns(session_id, 500)
        return SessionDetail(
            id=session_id,
            turns=[
                TurnOut(
                    question=t.question,
                    standalone_question=t.standalone_question,
                    answer=t.answer,
                    language=t.language,
                    abstained=t.abstained,
                    method=t.method,
                    citations=_citations(t.citations),
                    asked_at=t.asked_at.isoformat(),
                )
                for t in turns
            ],
        )

    @app.delete("/v1/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str, who: Authed) -> Response:
        if not services.conversations.store.delete(session_id, who.id):
            raise HTTPException(404, "session not found")
        return Response(status_code=204)

    @app.get("/v1/pages/{document}/{page}")
    def page_image(document: str, page: int, who: Authed) -> FileResponse:
        path = services.pages.path(document, page) if services.pages else None
        if path is None or not path.is_file():
            raise HTTPException(404, "page image not available")
        return FileResponse(path, media_type="image/png", headers={"cache-control": "private, max-age=86400"})

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
    return app

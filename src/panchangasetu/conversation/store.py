"""Session history storage: Postgres in production, in-memory for tests and throwaway use."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from panchangasetu.observability.calls import LlmCall
from panchangasetu.storage.db import session_scope
from panchangasetu.storage.models import ChatSession, ChatTurn, LlmCallRecord, utcnow


@dataclass(slots=True)
class TurnRecord:
    question: str
    standalone_question: str
    rewrite_method: str  # none | rule | llm
    language: str
    answer: str
    abstained: bool
    method: str
    route: str
    asked_at: datetime
    citations: list[dict[str, Any]] = field(default_factory=list)
    model: str | None = None
    cost_inr: float = 0.0
    latency_ms: float = 0.0
    # Observability (trace view)
    request_id: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reason: str = ""
    timings: dict[str, float] = field(default_factory=dict)
    sources: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: list[LlmCall] = field(default_factory=list)

    @property
    def cited_dates(self) -> list[str]:
        return [str(c["date"]) for c in self.citations if c.get("date")]


class SessionNotFound(KeyError):
    pass


@dataclass(slots=True)
class SessionSummary:
    id: str
    title: str  # the first question
    turns: int
    created_at: datetime
    updated_at: datetime


class SessionStore(Protocol):
    def create(self, owner: str | None = None) -> str: ...

    def exists(self, session_id: str, owner: str | None = None) -> bool: ...

    def recent_turns(self, session_id: str, limit: int) -> list[TurnRecord]:
        """The last `limit` turns, oldest first."""
        ...

    def add_turn(self, session_id: str, turn: TurnRecord) -> None: ...

    def add_calls(
        self, calls: list[LlmCall], *, session_id: str | None, owner: str | None, request_id: str | None
    ) -> None:
        """Record LLM calls that belong to no saved turn, so their spend still counts."""
        ...

    def list_sessions(self, owner: str | None, limit: int = 50) -> list[SessionSummary]:
        """Most recently active first; empty sessions are left out."""
        ...

    def delete(self, session_id: str, owner: str | None = None) -> bool: ...

    def spent_since(self, since: datetime) -> float:
        """LLM spend (INR) since `since`, across all sessions, from the append-only record of LLM
        calls written with each turn. Deleting or pruning conversations does not lower it."""
        ...

    def prune(self, inactive_before: datetime) -> int:
        """Delete sessions with no activity since `inactive_before`; returns how many."""
        ...


class InMemorySessionStore:
    def __init__(self) -> None:
        self._turns: dict[str, list[TurnRecord]] = {}
        self._calls: list[LlmCall] = []
        self._owners: dict[str, str | None] = {}

    def create(self, owner: str | None = None) -> str:
        sid = str(uuid.uuid4())
        self._turns[sid] = []
        self._owners[sid] = owner
        return sid

    def exists(self, session_id: str, owner: str | None = None) -> bool:
        return session_id in self._turns and (owner is None or self._owners[session_id] == owner)

    def recent_turns(self, session_id: str, limit: int) -> list[TurnRecord]:
        if session_id not in self._turns:
            raise SessionNotFound(session_id)
        return self._turns[session_id][-limit:] if limit > 0 else []

    def add_turn(self, session_id: str, turn: TurnRecord) -> None:
        if session_id not in self._turns:
            raise SessionNotFound(session_id)
        self._turns[session_id].append(turn)
        self._calls.extend(turn.llm_calls)

    def add_calls(
        self, calls: list[LlmCall], *, session_id: str | None, owner: str | None, request_id: str | None
    ) -> None:
        self._calls.extend(calls)

    def list_sessions(self, owner: str | None, limit: int = 50) -> list[SessionSummary]:
        out = [
            SessionSummary(sid, turns[0].question, len(turns), turns[0].asked_at, turns[-1].asked_at)
            for sid, turns in self._turns.items()
            if turns and self._owners[sid] == owner
        ]
        return sorted(out, key=lambda s: s.updated_at, reverse=True)[:limit]

    def delete(self, session_id: str, owner: str | None = None) -> bool:
        if not self.exists(session_id, owner):
            return False
        del self._turns[session_id], self._owners[session_id]
        return True

    def spent_since(self, since: datetime) -> float:
        return sum(c.cost_inr for c in self._calls if c.at >= since)

    def prune(self, inactive_before: datetime) -> int:
        stale = [sid for sid, turns in self._turns.items() if turns and turns[-1].asked_at < inactive_before]
        for sid in stale:
            del self._turns[sid], self._owners[sid]
        return len(stale)


def _uuid(session_id: str) -> uuid.UUID:
    try:
        return uuid.UUID(session_id)
    except ValueError as exc:
        raise SessionNotFound(session_id) from exc


class SqlSessionStore:
    def __init__(self, factory: sessionmaker[Session]) -> None:
        self.factory = factory

    def create(self, owner: str | None = None) -> str:
        with session_scope(self.factory) as s:
            row = ChatSession(owner=owner)
            s.add(row)
            s.flush()
            return str(row.id)

    def exists(self, session_id: str, owner: str | None = None) -> bool:
        try:
            sid = _uuid(session_id)
        except SessionNotFound:
            return False
        with session_scope(self.factory) as s:
            row = s.get(ChatSession, sid)
            return row is not None and (owner is None or row.owner == owner)

    def recent_turns(self, session_id: str, limit: int) -> list[TurnRecord]:
        sid = _uuid(session_id)
        with session_scope(self.factory) as s:
            if s.get(ChatSession, sid) is None:
                raise SessionNotFound(session_id)
            if limit <= 0:
                return []
            rows = s.scalars(
                select(ChatTurn).where(ChatTurn.session_id == sid).order_by(ChatTurn.seq.desc()).limit(limit)
            ).all()
            return [_record(r) for r in reversed(rows)]

    def add_turn(self, session_id: str, turn: TurnRecord) -> None:
        sid = _uuid(session_id)
        with session_scope(self.factory) as s:
            chat = s.get(ChatSession, sid, with_for_update=True)
            if chat is None:
                raise SessionNotFound(session_id)
            seq = (s.scalar(select(func.max(ChatTurn.seq)).where(ChatTurn.session_id == sid)) or 0) + 1
            row = ChatTurn(
                session_id=sid,
                seq=seq,
                question=turn.question,
                standalone_question=turn.standalone_question,
                rewrite_method=turn.rewrite_method,
                language=turn.language,
                answer=turn.answer,
                abstained=turn.abstained,
                method=turn.method,
                route=turn.route,
                citations=turn.citations,
                model=turn.model,
                cost_inr=turn.cost_inr,
                latency_ms=turn.latency_ms,
                asked_at=turn.asked_at,
                request_id=turn.request_id,
                prompt_tokens=turn.prompt_tokens,
                completion_tokens=turn.completion_tokens,
                reason=turn.reason,
                timings=turn.timings,
                sources=turn.sources,
            )
            s.add(row)
            chat.updated_at = utcnow()
            s.flush()  # the turn's id links its LLM calls
            for c in turn.llm_calls:  # same transaction as the turn: never one without the other
                s.add(_call_row(c, row.id, sid, chat.owner, turn.request_id))

    def add_calls(
        self, calls: list[LlmCall], *, session_id: str | None, owner: str | None, request_id: str | None
    ) -> None:
        sid = _uuid(session_id) if session_id else None
        with session_scope(self.factory) as s:
            s.add_all(_call_row(c, None, sid, owner, request_id) for c in calls)

    def list_sessions(self, owner: str | None, limit: int = 50) -> list[SessionSummary]:
        with session_scope(self.factory) as s:
            first = (
                select(ChatTurn.question)
                .where(ChatTurn.session_id == ChatSession.id)
                .order_by(ChatTurn.seq)
                .limit(1)
                .scalar_subquery()
            )
            count = (
                select(func.count(ChatTurn.id)).where(ChatTurn.session_id == ChatSession.id).scalar_subquery()
            )
            owner_filter = ChatSession.owner.is_(None) if owner is None else ChatSession.owner == owner
            rows = s.execute(
                select(ChatSession, first, count)
                .where(owner_filter, count > 0)
                .order_by(ChatSession.updated_at.desc())
                .limit(limit)
            ).all()
            return [SessionSummary(str(c.id), str(q), int(n), c.created_at, c.updated_at) for c, q, n in rows]

    def delete(self, session_id: str, owner: str | None = None) -> bool:
        if not self.exists(session_id, owner):
            return False
        with session_scope(self.factory) as s:
            # LLM calls outlive the conversation (spend history, daily budget); unlink them first so
            # this holds on every database, not only where ON DELETE SET NULL is enforced.
            turns = select(ChatTurn.id).where(ChatTurn.session_id == _uuid(session_id))
            s.query(LlmCallRecord).filter(LlmCallRecord.turn_id.in_(turns)).update(
                {LlmCallRecord.turn_id: None}, synchronize_session=False
            )
            s.query(ChatTurn).filter(ChatTurn.session_id == _uuid(session_id)).delete()
            s.query(ChatSession).filter(ChatSession.id == _uuid(session_id)).delete()
        return True

    def spent_since(self, since: datetime) -> float:
        with session_scope(self.factory) as s:
            total = s.scalar(
                select(func.coalesce(func.sum(LlmCallRecord.cost_inr), 0.0)).where(LlmCallRecord.at >= since)
            )
            return float(total or 0.0)

    def prune(self, inactive_before: datetime) -> int:
        with session_scope(self.factory) as s:
            ids = list(s.scalars(select(ChatSession.id).where(ChatSession.updated_at < inactive_before)))
            if ids:
                turns = select(ChatTurn.id).where(ChatTurn.session_id.in_(ids))
                s.query(LlmCallRecord).filter(LlmCallRecord.turn_id.in_(turns)).update(
                    {LlmCallRecord.turn_id: None}, synchronize_session=False
                )
                s.query(ChatTurn).filter(ChatTurn.session_id.in_(ids)).delete(synchronize_session=False)
                s.query(ChatSession).filter(ChatSession.id.in_(ids)).delete(synchronize_session=False)
            return len(ids)


def _record(r: ChatTurn) -> TurnRecord:
    return TurnRecord(
        question=r.question,
        standalone_question=r.standalone_question,
        rewrite_method=r.rewrite_method,
        language=r.language,
        answer=r.answer,
        abstained=r.abstained,
        method=r.method,
        route=r.route,
        asked_at=r.asked_at,
        citations=list(r.citations or []),
        model=r.model,
        cost_inr=r.cost_inr,
        latency_ms=r.latency_ms,
        request_id=r.request_id,
        prompt_tokens=r.prompt_tokens,
        completion_tokens=r.completion_tokens,
        reason=r.reason,
        timings=dict(r.timings or {}),
        sources=list(r.sources or []),
    )


def _call_row(
    c: LlmCall,
    turn_id: uuid.UUID | None,
    session_id: uuid.UUID | None,
    owner: str | None,
    request_id: str | None,
) -> LlmCallRecord:
    return LlmCallRecord(
        at=c.at,
        turn_id=turn_id,
        session_id=session_id,
        owner=owner,
        request_id=request_id,
        purpose=c.purpose,
        provider=c.provider,
        model=c.model,
        prompt_tokens=c.prompt_tokens,
        completion_tokens=c.completion_tokens,
        cost_inr=c.cost_inr,
        latency_ms=c.latency_ms,
        status=c.status,
        error=c.error,
    )

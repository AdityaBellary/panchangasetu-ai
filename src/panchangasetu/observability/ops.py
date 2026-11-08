"""Monitoring queries behind the admin dashboard: usage, cost, tokens, latency per stage and per
model, why answers were refused, and full per-request traces with every LLM call.

Computed from `chat_turns` and `llm_calls` for the requested window. Rows are aggregated in
Python: at this scale it is fast and keeps percentiles exact and database-agnostic.
"""

from __future__ import annotations

import math
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from panchangasetu.observability.metrics import abstention_reason
from panchangasetu.storage.db import session_scope
from panchangasetu.storage.models import ApiKey, ChatSession, ChatTurn, LlmCallRecord

_MAX_ROWS = 50_000


def percentile(values: Iterable[float], q: float) -> float | None:
    data = sorted(v for v in values if v is not None)
    if not data:
        return None
    k = (len(data) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return round(data[lo] + (data[hi] - data[lo]) * (k - lo), 1)


def _utc(dt: datetime) -> datetime:
    """Timestamps are stored in UTC; some drivers (SQLite) hand them back without a zone."""
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _iso(dt: datetime | None) -> str | None:
    return _utc(dt).isoformat() if dt else None


@dataclass
class OpsQueries:
    factory: sessionmaker[Session]
    timezone: str = "Asia/Kolkata"
    daily_budget_inr: float = 0.0

    # ------------------------------------------------------------------ overview
    def overview(self, hours: int = 24, now: datetime | None = None) -> dict[str, Any]:
        tz = ZoneInfo(self.timezone)
        now = (now or datetime.now(UTC)).astimezone(tz)
        since = now - timedelta(hours=hours)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        with session_scope(self.factory) as s:
            turns = s.execute(
                select(ChatTurn, ChatSession.owner)
                .join(ChatSession, ChatSession.id == ChatTurn.session_id)
                .where(ChatTurn.asked_at >= since)
                .order_by(ChatTurn.asked_at)
                .limit(_MAX_ROWS)
            ).all()
            calls = list(
                s.scalars(
                    select(LlmCallRecord).where(LlmCallRecord.at >= min(since, midnight)).limit(_MAX_ROWS)
                )
            )
            names = {str(k.id): k.name for k in s.scalars(select(ApiKey))}

        window_calls = [c for c in calls if _utc(c.at) >= since]
        spent_today = sum(c.cost_inr for c in calls if _utc(c.at) >= midnight)
        rows = [t for t, _ in turns]
        answered = [t for t in rows if not t.abstained]
        llm_answers = [t for t in rows if t.method == "llm"]
        cost = sum(c.cost_inr for c in window_calls)

        hourly = hours <= 48
        step = timedelta(hours=1) if hourly else timedelta(days=1)

        def bucket(dt: datetime) -> datetime:
            local = _utc(dt).astimezone(tz)
            return (
                local.replace(minute=0, second=0, microsecond=0)
                if hourly
                else local.replace(hour=0, minute=0, second=0, microsecond=0)
            )

        start = bucket(since)
        keys: list[datetime] = []
        k = start
        while k <= now:
            keys.append(k)
            k += step
        series: dict[datetime, dict[str, Any]] = {
            k: {"t": k.isoformat(), "template": 0, "llm": 0, "abstain": 0, "cost_inr": 0.0, "lat": []}
            for k in keys
        }
        for t in rows:
            b = series.get(bucket(t.asked_at))
            if b is not None:
                b[t.method if t.method in ("template", "llm", "abstain") else "abstain"] += 1
                b["lat"].append(t.latency_ms)
        for c in window_calls:
            b = series.get(bucket(c.at))
            if b is not None:
                b["cost_inr"] += c.cost_inr
        points = []
        for b in series.values():
            lat = b.pop("lat")
            points.append(
                b
                | {
                    "cost_inr": round(b["cost_inr"], 4),
                    "p50_ms": percentile(lat, 0.5),
                    "p95_ms": percentile(lat, 0.95),
                }
            )

        stages: dict[str, list[float]] = defaultdict(list)
        for t in rows:
            timings = t.timings or {}
            for name in ("rewrite", "retrieve", "total"):
                if name in timings:
                    stages[name].append(float(timings[name]))
            llm_ms = sum(float(v) for key, v in timings.items() if key.startswith("llm_"))
            if llm_ms:
                stages["llm"].append(llm_ms)

        models: dict[tuple[str, str], list[LlmCallRecord]] = defaultdict(list)
        for c in window_calls:
            models[(c.provider, c.model)].append(c)

        owners: dict[str, dict[str, Any]] = defaultdict(lambda: {"questions": 0, "cost_inr": 0.0})
        for _turn, owner in turns:
            o = owners[names.get(owner or "", owner or "terminal")]
            o["questions"] += 1
        for c in window_calls:
            owners[names.get(c.owner or "", c.owner or "terminal")]["cost_inr"] += c.cost_inr

        return {
            "window_hours": hours,
            "generated_at": now.isoformat(),
            "kpis": {
                "questions": len(rows),
                "answered": len(answered),
                "abstained": len(rows) - len(answered),
                "answer_rate": round(len(answered) / len(rows), 3) if rows else None,
                "template_answers": sum(t.method == "template" for t in rows),
                "llm_answers": len(llm_answers),
                "cost_inr": round(cost, 4),
                "avg_cost_per_llm_answer_inr": round(cost / len(llm_answers), 4) if llm_answers else None,
                "llm_calls": len(window_calls),
                "llm_errors": sum(c.status != "ok" for c in window_calls),
                "prompt_tokens": sum(c.prompt_tokens for c in window_calls),
                "completion_tokens": sum(c.completion_tokens for c in window_calls),
                "latency_p50_ms": percentile((t.latency_ms for t in rows), 0.5),
                "latency_p95_ms": percentile((t.latency_ms for t in rows), 0.95),
            },
            "budget": {"limit_inr": self.daily_budget_inr, "spent_today_inr": round(spent_today, 4)},
            "series": points,
            "stages": {
                name: {"p50_ms": percentile(v, 0.5), "p95_ms": percentile(v, 0.95), "n": len(v)}
                for name, v in stages.items()
            },
            "methods": dict(Counter(t.method for t in rows)),
            "abstention_reasons": dict(
                Counter(
                    abstention_reason(t.reason) if t.reason else "unrecorded" for t in rows if t.abstained
                )
            ),
            "languages": dict(Counter(t.language for t in rows)),
            "routes": dict(Counter(t.route for t in rows).most_common(8)),
            "rewrites": dict(Counter(t.rewrite_method for t in rows)),
            "models": [
                {
                    "provider": p,
                    "model": m,
                    "calls": len(cs),
                    "errors": sum(c.status != "ok" for c in cs),
                    "purposes": dict(Counter(c.purpose for c in cs)),
                    "prompt_tokens": sum(c.prompt_tokens for c in cs),
                    "completion_tokens": sum(c.completion_tokens for c in cs),
                    "cost_inr": round(sum(c.cost_inr for c in cs), 4),
                    "avg_cost_inr": round(sum(c.cost_inr for c in cs) / len(cs), 5),
                    "p50_ms": percentile((c.latency_ms for c in cs), 0.5),
                    "p95_ms": percentile((c.latency_ms for c in cs), 0.95),
                }
                for (p, m), cs in sorted(models.items(), key=lambda kv: -len(kv[1]))
            ],
            "clients": [
                {"name": n, "questions": v["questions"], "cost_inr": round(v["cost_inr"], 4)}
                for n, v in sorted(owners.items(), key=lambda kv: -kv[1]["questions"])
            ],
        }

    # ------------------------------------------------------------------ request log and traces
    def requests(self, limit: int = 50, offset: int = 0, method: str | None = None) -> list[dict[str, Any]]:
        with session_scope(self.factory) as s:
            q = (
                select(ChatTurn, ChatSession.owner)
                .join(ChatSession, ChatSession.id == ChatTurn.session_id)
                .order_by(ChatTurn.asked_at.desc())
                .limit(limit)
                .offset(offset)
            )
            if method:
                q = q.where(ChatTurn.method == method)
            rows = s.execute(q).all()
            names = {str(k.id): k.name for k in s.scalars(select(ApiKey))}
            n_calls = Counter(
                str(turn_id)
                for turn_id in s.scalars(
                    select(LlmCallRecord.turn_id).where(LlmCallRecord.turn_id.in_([t.id for t, _ in rows]))
                )
            )
        return [
            {
                "id": str(t.id),
                "asked_at": _iso(t.asked_at),
                "client": names.get(owner or "", owner or "terminal"),
                "question": t.question,
                "standalone_question": t.standalone_question if t.standalone_question != t.question else None,
                "language": t.language,
                "method": t.method,
                "route": t.route,
                "abstained": t.abstained,
                "reason": (abstention_reason(t.reason) if t.reason else "unrecorded")
                if t.abstained
                else None,
                "model": t.model,
                "llm_calls": n_calls.get(str(t.id), 0),
                "prompt_tokens": t.prompt_tokens,
                "completion_tokens": t.completion_tokens,
                "cost_inr": t.cost_inr,
                "latency_ms": round(t.latency_ms, 1),
            }
            for t, owner in rows
        ]

    def trace(self, turn_id: str) -> dict[str, Any] | None:
        try:
            tid = uuid.UUID(turn_id)
        except ValueError:
            return None
        with session_scope(self.factory) as s:
            found = s.execute(
                select(ChatTurn, ChatSession.owner)
                .join(ChatSession, ChatSession.id == ChatTurn.session_id)
                .where(ChatTurn.id == tid)
            ).first()
            if found is None:
                return None
            t, owner = found
            calls = list(
                s.scalars(
                    select(LlmCallRecord).where(LlmCallRecord.turn_id == tid).order_by(LlmCallRecord.at)
                )
            )
            key = s.get(ApiKey, uuid.UUID(owner)) if owner and _is_uuid(owner) else None
        return {
            "id": str(t.id),
            "session_id": str(t.session_id),
            "request_id": t.request_id,
            "asked_at": _iso(t.asked_at),
            "client": key.name if key else (owner or "terminal"),
            "question": t.question,
            "standalone_question": t.standalone_question,
            "rewrite_method": t.rewrite_method,
            "language": t.language,
            "route": t.route,
            "method": t.method,
            "abstained": t.abstained,
            "reason": t.reason,
            "answer": t.answer,
            "model": t.model,
            "prompt_tokens": t.prompt_tokens,
            "completion_tokens": t.completion_tokens,
            "cost_inr": t.cost_inr,
            "latency_ms": round(t.latency_ms, 1),
            "timings": t.timings or {},
            "sources": t.sources or [],
            "citations": t.citations or [],
            "llm_calls": [_call(c) for c in calls],
        }

    def calls(self, limit: int = 100, errors_only: bool = False) -> list[dict[str, Any]]:
        with session_scope(self.factory) as s:
            q = select(LlmCallRecord).order_by(LlmCallRecord.at.desc()).limit(limit)
            if errors_only:
                q = q.where(LlmCallRecord.status != "ok")
            return [_call(c) for c in s.scalars(q)]


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


def _call(c: LlmCallRecord) -> dict[str, Any]:
    return {
        "at": _iso(c.at),
        "turn_id": str(c.turn_id) if c.turn_id else None,
        "request_id": c.request_id,
        "purpose": c.purpose,
        "provider": c.provider,
        "model": c.model,
        "prompt_tokens": c.prompt_tokens,
        "completion_tokens": c.completion_tokens,
        "cost_inr": c.cost_inr,
        "latency_ms": c.latency_ms,
        "status": c.status,
        "error": c.error,
    }

"""Multi-turn question answering: resolve the follow-up, answer it standalone, record the turn.

Only the previous questions (their standalone forms) and the dates their answers cited are used as
context. Earlier answers are never fed back to the answering model, so a wrong earlier answer
cannot leak into a later one: every answer is still grounded only in retrieved sources.
"""

from __future__ import annotations

import time
from dataclasses import asdict, replace
from datetime import datetime, timedelta

from panchangasetu.conversation.followup import FollowUpResolver
from panchangasetu.conversation.store import SessionNotFound, SessionStore, TurnRecord
from panchangasetu.generation.answerer import Answer, AnswerService, apply_calls
from panchangasetu.observability.calls import call_purpose, collect_calls
from panchangasetu.observability.context import request_id_var
from panchangasetu.observability.logging import get_logger
from panchangasetu.observability.metrics import record_answer, record_budget
from panchangasetu.query.language import detect_language

log = get_logger(__name__)


class ConversationService:
    def __init__(
        self,
        answers: AnswerService,
        store: SessionStore,
        resolver: FollowUpResolver | None = None,
        *,
        history_turns: int = 3,
        idle_minutes: float = 60.0,
        daily_budget_inr: float = 0.0,
    ) -> None:
        self.answers = answers
        self.store = store
        self.resolver = resolver or FollowUpResolver()
        self.history_turns = history_turns
        self.idle = timedelta(minutes=idle_minutes)
        # LLM spend allowed per local day across all sessions; 0 = no limit. Template answers
        # (calendar lookups) cost nothing and keep working once the budget is used up.
        self.daily_budget_inr = daily_budget_inr

    def spent_today(self, now: datetime) -> float:
        return self.store.spent_since(now.replace(hour=0, minute=0, second=0, microsecond=0))

    def llm_allowed(self, now: datetime) -> bool:
        spent = self.spent_today(now)
        record_budget(self.daily_budget_inr, spent)
        return self.daily_budget_inr <= 0 or spent < self.daily_budget_inr

    def start(self, owner: str | None = None) -> str:
        return self.store.create(owner)

    def ask(
        self,
        question: str,
        *,
        session_id: str | None = None,
        owner: str | None = None,
        now: datetime | None = None,
    ) -> Answer:
        now = now or self.answers.retriever.clock()
        if session_id is None:
            session_id = self.store.create(owner)
        elif not self.store.exists(session_id, owner):
            raise SessionNotFound(session_id)
        # A conversation that went quiet is not context any more: "and tomorrow?" an hour later
        # is more likely a slip than a reference to a question the user has forgotten.
        history = [
            # Stored times come back in UTC; "today" of an earlier turn is a local-time question.
            replace(t, asked_at=t.asked_at.astimezone(now.tzinfo)) if now.tzinfo and t.asked_at.tzinfo else t
            for t in self.store.recent_turns(session_id, self.history_turns)
            if now - t.asked_at <= self.idle
        ]
        allow_llm = self.llm_allowed(now)
        t0 = time.perf_counter()
        with collect_calls() as calls:
            with call_purpose("followup_rewrite"):
                res = self.resolver.resolve(question, history, now, allow_llm=allow_llm)
            rewrite_ms = (time.perf_counter() - t0) * 1000
            language = detect_language(question) if res.method != "none" else None
            answer = self.answers.ask(res.question, now=now, language=language, allow_llm=allow_llm)
        for c in calls:  # spend is booked at the turn's time, the clock the daily budget uses
            c.at = now
        apply_calls(answer, calls)  # the rewrite's call counts towards this answer's cost
        answer.latency_ms["rewrite"] = round(rewrite_ms, 1)
        answer.latency_ms["total"] = round((time.perf_counter() - t0) * 1000, 1)
        answer.question = question
        answer.standalone_question = res.question
        answer.session_id = session_id
        record_answer(answer, res.method)
        if answer.cost_inr:
            record_budget(self.daily_budget_inr, self.spent_today(now) + answer.cost_inr)
        # One line per answer for dashboards and debugging; the question itself is not logged.
        log.info(
            "answer.done",
            session_id=session_id,
            method=answer.method,
            route=answer.route,
            language=answer.language,
            abstained=answer.abstained,
            reason=answer.reason[:120] or None,
            rewrite=res.method,
            model=answer.model,
            llm_calls=len(answer.llm_calls),
            prompt_tokens=answer.prompt_tokens,
            completion_tokens=answer.completion_tokens,
            cost_inr=answer.cost_inr,
            latency_ms=round(answer.latency_ms.get("total", 0.0), 1),
            question_chars=len(question),
        )
        self.store.add_turn(
            session_id,
            TurnRecord(
                question=question,
                standalone_question=res.question,
                rewrite_method=res.method,
                language=answer.language,
                answer=answer.answer,
                abstained=answer.abstained,
                method=answer.method,
                route=answer.route,
                asked_at=now,
                citations=[asdict(c) for c in answer.citations],
                model=answer.model,
                cost_inr=answer.cost_inr,
                latency_ms=answer.latency_ms.get("total", 0.0),
                request_id=request_id_var.get(),
                prompt_tokens=answer.prompt_tokens,
                completion_tokens=answer.completion_tokens,
                reason=answer.reason,
                timings={k: round(v, 1) for k, v in answer.latency_ms.items()},
                sources=answer.sources,
                llm_calls=answer.llm_calls,
            ),
        )
        return answer

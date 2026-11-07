"""Multi-turn question answering: resolve the follow-up, answer it standalone, record the turn.

Only the previous questions (their standalone forms) and the dates their answers cited are used as
context. Earlier answers are never fed back to the answering model, so a wrong earlier answer
cannot leak into a later one: every answer is still grounded only in retrieved sources.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timedelta

from panchangasetu.conversation.followup import FollowUpResolver
from panchangasetu.conversation.store import SessionNotFound, SessionStore, TurnRecord
from panchangasetu.generation.answerer import Answer, AnswerService
from panchangasetu.observability.logging import get_logger
from panchangasetu.observability.metrics import record_answer
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

    def llm_allowed(self, now: datetime) -> bool:
        if self.daily_budget_inr <= 0:
            return True
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return self.store.spent_since(midnight) < self.daily_budget_inr

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
        res = self.resolver.resolve(question, history, now, allow_llm=allow_llm)
        language = detect_language(question) if res.method != "none" else None
        answer = self.answers.ask(res.question, now=now, language=language, allow_llm=allow_llm)
        rewrite_in, rewrite_out = self.resolver.last_cost_tokens
        if rewrite_in or rewrite_out:
            answer.prompt_tokens += rewrite_in
            answer.completion_tokens += rewrite_out
            answer.cost_inr = round(answer.cost_inr + self.answers.pricing.cost(rewrite_in, rewrite_out), 5)
        answer.question = question
        answer.standalone_question = res.question
        answer.session_id = session_id
        record_answer(answer, res.method)
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
            ),
        )
        return answer

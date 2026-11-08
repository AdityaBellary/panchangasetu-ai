"""Question -> grounded answer with citations, or an explicit abstention.

Decision order (cheapest and safest first):
  1. No usable evidence (or the date is outside every source)      -> abstain, no LLM call.
  2. Verified day lookup whose attributes are printed fields        -> template answer, no LLM call.
  3. Otherwise                                                      -> LLM over numbered sources,
     then mechanical verification (citations exist; every time/date is in the cited sources);
     one corrective retry, then abstain.
"""

from __future__ import annotations

import json
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import date, datetime
from itertools import zip_longest
from typing import Any

from panchangasetu.core.interfaces import LLM, LLMError
from panchangasetu.generation.prompts import (
    ABSTAIN_EN,
    ABSTAIN_TE,
    BUDGET_EN,
    BUDGET_TE,
    DAY_INTRO_EN,
    DAY_INTRO_TE,
    UNAVAILABLE_EN,
    UNAVAILABLE_TE,
    build_messages,
    retry_feedback,
)
from panchangasetu.generation.templates import (
    can_template,
    can_template_occurrences,
    render_day_answers,
    render_occurrence_answer,
)
from panchangasetu.generation.verify import check_answer, check_quotes, strip_citation_markers
from panchangasetu.observability.calls import LlmCall, call_purpose, collect_calls
from panchangasetu.observability.logging import get_logger
from panchangasetu.query.language import Language
from panchangasetu.query.router import PanchangaMode
from panchangasetu.retrieval.evidence import Evidence
from panchangasetu.retrieval.orchestrator import RetrievalResult, Retriever

log = get_logger(__name__)
# Full chunks (Telugu runs ~1.5-2k characters per chunk) beat many truncated ones: cutting a chunk
# can drop exactly the sentence that answers the question.
_MAX_EVIDENCE = 5
_MAX_REFUSAL_DAYS = 3  # "is next week good for…?" shows at most three days
# Answers to the question as asked (a day's facts, the next occurrence, a weekday period). Dated
# events and daily notes are search results like passages: for "rain in Chaitra masam" they must not
# crowd out the General text that answers it.
_EXACT_KINDS = frozenset({"day", "occurrence", "weekday_period"})


@dataclass(slots=True)
class Citation:
    document: str
    pages: list[int]
    date: str | None
    verified: bool | None
    snippet: str
    quotes: list[str] = field(default_factory=list)  # verbatim supporting passages (verified)


@dataclass(slots=True)
class Answer:
    question: str
    answer: str
    language: str
    abstained: bool
    method: str  # template | llm | abstain
    citations: list[Citation] = field(default_factory=list)
    route: str = ""
    reason: str = ""  # why it abstained / notes
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_inr: float = 0.0
    latency_ms: dict[str, float] = field(default_factory=dict)
    # Set by the conversation layer (Phase 6): the follow-up rewritten as a standalone question.
    standalone_question: str | None = None
    session_id: str | None = None
    # Observability: every LLM call made for this answer (cost and tokens above are their sums), the
    # sources shown to the model, and the retrieval stage timings.
    llm_calls: list[LlmCall] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)


def apply_calls(answer: Answer, calls: list[LlmCall]) -> None:
    """Cost, tokens and model of an answer come from the calls actually made for it."""
    answer.llm_calls = list(calls)
    answer.prompt_tokens = sum(c.prompt_tokens for c in calls)
    answer.completion_tokens = sum(c.completion_tokens for c in calls)
    answer.cost_inr = round(sum(c.cost_inr for c in calls), 6)
    answered_by = [c for c in calls if c.status == "ok" and c.purpose != "followup_rewrite"]
    if answered_by:
        answer.model = answered_by[-1].model


def select_evidence(evidence: list[Evidence], limit: int) -> list[Evidence]:
    """Exact calendar facts first, then searched passages alternating between collections.

    A HYBRID question ("which months are good for griha pravesham?") retrieves dated lists and
    the rule that explains them; taking the first `limit` in retrieval order let the lists crowd
    the rule out entirely."""
    exact = [e for e in evidence if e.kind in _EXACT_KINDS]
    searched = [e for e in evidence if e.kind not in _EXACT_KINDS]  # passages and dated events
    groups = [[e for e in searched if e.collection == c] for c in ("general", "panchanga")]
    mixed = [e for pair in zip_longest(*groups) for e in pair if e is not None]
    return (exact + mixed)[:limit]


def _source(e: Evidence) -> dict[str, Any]:
    """What the trace view shows for one source given to the model."""
    return {
        "kind": e.kind,
        "collection": e.collection,
        "document": e.document,
        "pages": e.pages[:3],
        "date": e.date,
        "score": round(e.score, 3) if e.score is not None else None,
        "section": e.source_kind,
        "heading": " > ".join(str(h) for h in e.data.get("heading_path", []))[:160],
    }


def _event_name(e: Evidence) -> str:
    """The festival text of an event, without its "[TTD festival calendar, date]" label and source."""
    body = e.text.split("] ", 1)[-1]
    return re.sub(r"\s*\(Source: [^)]*\)\s*$", "", body).strip()


def _citation(e: Evidence) -> Citation:
    return Citation(e.document, e.pages, e.date, e.verified, e.text[:240])


def _abstain_text(lang: str) -> str:
    return ABSTAIN_TE if lang == "te" else ABSTAIN_EN


def _parse(text: str) -> dict[str, object] | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _models(llm: LLM) -> set[str]:
    providers = getattr(llm, "providers", [llm])
    return {str(getattr(p, "model", "")) for p in providers}


class AnswerService:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLM | None,
        *,
        escalation: LLM | None = None,
        escalation_max_tokens: int = 4000,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        # A stronger (slower, pricier) model tried once when the primary cannot produce a verified
        # answer although evidence exists. Only hard questions pay for it.
        self.escalation = escalation
        self.escalation_max_tokens = escalation_max_tokens

    def ask(
        self,
        question: str,
        *,
        now: datetime | None = None,
        language: Language | None = None,
        allow_llm: bool = True,
    ) -> Answer:
        t0 = time.perf_counter()
        now = now or self.retriever.clock()
        with collect_calls() as calls:
            r = self.retriever.retrieve(question, now=now, language=language)
            answer = self._respond(question, r, now, t0, allow_llm)
        if answer.abstained:
            self._helpful_refusal(answer, r)
        apply_calls(answer, calls)
        answer.sources = [_source(e) for e in select_evidence(r.evidence, _MAX_EVIDENCE)]
        answer.latency_ms |= {f"retrieve_{k}": round(v, 1) for k, v in r.timings_ms.items() if k != "total"}
        return answer

    def _helpful_refusal(self, answer: Answer, r: RetrievalResult) -> None:
        """A refusal still helps: for a question about a day, show that day's panchangam (with
        rahukalam and the festivals printed for it) instead of a bare "I don't know"."""
        te = answer.language == "te"
        if "budget" in answer.reason:
            intro = BUDGET_TE if te else BUDGET_EN
        elif "unavailable" in answer.reason:
            intro = UNAVAILABLE_TE if te else UNAVAILABLE_EN
        else:
            intro = DAY_INTRO_TE if te else DAY_INTRO_EN
        days = [e for e in r.evidence if e.kind == "day"][:_MAX_REFUSAL_DAYS]
        if r.route.mode != PanchangaMode.DAY or not days:
            if "unavailable" in answer.reason:
                answer.answer = intro  # the budget and not-found texts are already set
            return
        periods = getattr(self.retriever, "weekday_periods", None)
        structured = getattr(self.retriever, "structured", None)
        extra: list[Evidence] = []
        events: list[Evidence] = []
        for e in days:
            d = date.fromisoformat(e.date or "")
            if periods is not None:
                extra += [periods.evidence(k, d) for k in sorted(periods.kinds)]
            if structured is not None:
                events += structured.events(d, d)
        attrs = ["panchangam", *sorted(periods.kinds if periods is not None else ())]
        blocks = [intro, render_day_answers(days + extra, attrs, answer.language)]
        if events:
            names = list(dict.fromkeys(_event_name(e) for e in events))
            label = "ఆ రోజు పండుగలు / విశేషాలు" if te else "Festivals and observances that day"
            blocks.append(f"{label}: " + "; ".join(names))
        answer.answer = "\n\n".join(blocks)
        answer.citations = [_citation(e) for e in days + events]

    def _respond(
        self, question: str, r: RetrievalResult, now: datetime, t0: float, allow_llm: bool
    ) -> Answer:
        lang = r.route.language.code
        route = f"{r.route.intent.value}/{r.route.mode.value}"
        base = {"question": question, "language": lang, "route": route}
        timings = {"retrieve": r.timings_ms.get("total", 0.0)}

        evidence = select_evidence(r.evidence, _MAX_EVIDENCE)
        if not evidence:
            return self._done(
                base,
                timings,
                t0,
                None,
                0,
                0,
                answer=_abstain_text(lang),
                abstained=True,
                method="abstain",
                reason="; ".join(r.notes) or "no relevant evidence retrieved",
            )

        attrs = [a for a in r.route.slots.attributes if a]
        if r.route.mode == PanchangaMode.DAY and not r.route.wants_general and can_template(evidence, attrs):
            days = [e for e in evidence if e.kind in ("day", "weekday_period")]
            text = render_day_answers(evidence, attrs, lang)
            return self._done(
                base,
                timings,
                t0,
                None,
                0,
                0,
                answer=text,
                abstained=False,
                method="template",
                citations=[_citation(e) for e in days],
            )

        if (
            r.route.mode == PanchangaMode.FIND
            and not r.route.wants_general
            and can_template_occurrences(evidence)
        ):
            occ = [e for e in evidence if e.kind == "occurrence"]
            return self._done(
                base,
                timings,
                t0,
                None,
                0,
                0,
                answer=render_occurrence_answer(occ, lang),
                abstained=False,
                method="template",
                citations=[_citation(e) for e in occ],
            )

        if self.llm is None or not allow_llm:
            return self._done(
                base,
                timings,
                t0,
                None,
                0,
                0,
                answer=_abstain_text(lang)
                if self.llm is None
                else (BUDGET_TE if lang == "te" else BUDGET_EN),
                abstained=True,
                method="abstain",
                reason="no LLM configured" if self.llm is None else "daily LLM budget reached",
                citations=[_citation(e) for e in evidence],
            )
        with call_purpose("answer"):
            first = self._llm_answer(question, r, evidence, now, base, timings, t0, self.llm, 700)
        if (
            first.abstained
            and self.escalation is not None
            and "unavailable" not in first.reason
            # The primary chain already fell through to the escalation model (e.g. Sarvam out of credits).
            and first.model not in _models(self.escalation)
            # Only the day's table to go on ("is tomorrow good for a new business?"): a stronger model
            # can't find a rule that isn't there. The refusal shows the day's panchangam instead.
            and not (r.route.mode == PanchangaMode.DAY and not r.route.wants_general)
        ):
            log.info("answer.escalate", reason=first.reason)
            with call_purpose("escalation"):
                second = self._llm_answer(
                    question,
                    r,
                    evidence,
                    now,
                    base,
                    dict(timings),
                    t0,
                    self.escalation,
                    self.escalation_max_tokens,
                    attempts=1,
                )
            if second.abstained:
                second.reason = f"{first.reason}; escalated: {second.reason}"
            return second
        return first

    def _llm_answer(
        self,
        question: str,
        r: RetrievalResult,
        evidence: list[Evidence],
        now: datetime,
        base: dict[str, str],
        timings: dict[str, float],
        t0: float,
        llm: LLM,
        max_tokens: int,
        *,
        attempts: int = 2,
    ) -> Answer:
        lang = base["language"]
        hints = self.retriever.glossary.lookup_pairs(question) if r.route.language.script != "Telu" else []
        messages = build_messages(question, evidence, language=lang, now=now, notes=r.notes, term_hints=hints)
        prompt_tokens = completion_tokens = 0
        model: str | None = None
        problems: list[str] = []
        for attempt in range(attempts):
            try:
                with call_purpose("answer_retry") if attempt else nullcontext():
                    res = llm.complete(messages, max_tokens=max_tokens, temperature=0.1, json=True)
            except LLMError as exc:
                log.warning("answer.llm_failed", error=str(exc), quota=exc.quota)
                return self._done(
                    base,
                    timings,
                    t0,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    answer=_abstain_text(lang),
                    abstained=True,
                    method="abstain",
                    reason=f"answer service unavailable: {exc}",
                )
            model = res.model
            prompt_tokens += res.prompt_tokens
            completion_tokens += res.completion_tokens
            timings[f"llm_{attempt + 1}"] = res.latency_ms
            data = _parse(res.text)
            if data is None:
                problems = ["the reply was not valid JSON"]
            elif not data.get("sufficient", False):
                return self._done(
                    base,
                    timings,
                    t0,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    answer=_abstain_text(lang),
                    abstained=True,
                    method="abstain",
                    reason="model judged the sources insufficient",
                )
            else:
                raw_ids = data.get("citations") or []
                ids = [str(c) for c in raw_ids] if isinstance(raw_ids, list) else []
                idx = [
                    int(c[1:]) - 1
                    for c in ids
                    if re.fullmatch(r"S\d+", c) and 0 < int(c[1:]) <= len(evidence)
                ]
                answer = str(data.get("answer", "")).strip()
                problems = []
                if not idx:
                    problems.append("no valid citations")
                if not answer:
                    problems.append("empty answer")
                cited = [evidence[i] for i in dict.fromkeys(idx)]
                problems += check_answer(answer, [e.text for e in cited], default_year=now.year)
                raw_quotes = data.get("quotes") or []
                quotes = [
                    (str(q.get("source", "")), str(q.get("text", "")))
                    for q in (raw_quotes if isinstance(raw_quotes, list) else [])
                    if isinstance(q, dict)
                ]
                quote_problems, supported = check_quotes(
                    quotes, {f"S{i + 1}": e.text for i, e in enumerate(evidence)}
                )
                problems += quote_problems
                # Sources that support a verified quote are cited even if the model forgot the id.
                for sid in supported:
                    i = int(sid[1:]) - 1
                    if evidence[i] not in cited:
                        cited.append(evidence[i])
                answer = strip_citation_markers(answer)
                if not problems:
                    return self._done(
                        base,
                        timings,
                        t0,
                        model,
                        prompt_tokens,
                        completion_tokens,
                        answer=answer,
                        abstained=False,
                        method="llm",
                        citations=_with_quotes(cited, evidence, supported),
                    )
            log.info("answer.verification_failed", attempt=attempt + 1, problems=problems)
            messages = [*messages, retry_feedback(problems)]
        return self._done(
            base,
            timings,
            t0,
            model,
            prompt_tokens,
            completion_tokens,
            answer=_abstain_text(lang),
            abstained=True,
            method="abstain",
            reason="answer failed verification: " + "; ".join(problems),
        )

    def _done(
        self,
        base: dict[str, str],
        timings: dict[str, float],
        t0: float,
        model: str | None,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        answer: str,
        abstained: bool,
        method: str,
        reason: str = "",
        citations: list[Citation] | None = None,
    ) -> Answer:
        timings["total"] = (time.perf_counter() - t0) * 1000
        return Answer(
            question=base["question"],
            language=base["language"],
            route=base["route"],
            answer=answer,
            abstained=abstained,
            method=method,
            reason=reason,
            citations=citations or [],
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_inr=0.0,  # filled from the recorded calls (apply_calls)
            latency_ms=timings,
        )


def _with_quotes(
    cited: list[Evidence], evidence: list[Evidence], supported: dict[str, list[str]]
) -> list[Citation]:
    out = []
    for e in cited:
        c = _citation(e)
        c.quotes = supported.get(f"S{evidence.index(e) + 1}", [])
        out.append(c)
    return out

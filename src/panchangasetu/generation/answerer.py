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
from dataclasses import dataclass, field
from datetime import datetime
from itertools import zip_longest

from panchangasetu.core.interfaces import LLM, LLMError
from panchangasetu.generation.prompts import ABSTAIN_EN, ABSTAIN_TE, build_messages, retry_feedback
from panchangasetu.generation.templates import (
    can_template,
    can_template_occurrences,
    render_day_answers,
    render_occurrence_answer,
)
from panchangasetu.generation.verify import check_answer, check_quotes, strip_citation_markers
from panchangasetu.observability.logging import get_logger
from panchangasetu.query.language import Language
from panchangasetu.query.router import PanchangaMode
from panchangasetu.retrieval.evidence import Evidence
from panchangasetu.retrieval.orchestrator import RetrievalResult, Retriever

log = get_logger(__name__)
# Full chunks (Telugu runs ~1.5-2k characters per chunk) beat many truncated ones: cutting a chunk
# can drop exactly the sentence that answers the question.
_MAX_EVIDENCE = 5


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


@dataclass(frozen=True, slots=True)
class Pricing:
    """Per 1M tokens, in INR (Sarvam pricing page, 2026-11-07)."""

    input_inr: float = 15.0
    output_inr: float = 60.0

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return round((prompt_tokens * self.input_inr + completion_tokens * self.output_inr) / 1_000_000, 5)


def select_evidence(evidence: list[Evidence], limit: int) -> list[Evidence]:
    """Exact calendar facts first, then searched passages alternating between collections.

    A HYBRID question ("which months are good for griha pravesham?") retrieves dated lists and
    the rule that explains them; taking the first `limit` in retrieval order let the lists crowd
    the rule out entirely."""
    exact = [e for e in evidence if e.kind != "chunk"]
    groups = [
        [e for e in evidence if e.kind == "chunk" and e.collection == c] for c in ("general", "panchanga")
    ]
    mixed = [e for pair in zip_longest(*groups) for e in pair if e is not None]
    return (exact + mixed)[:limit]


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


class AnswerService:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLM | None,
        *,
        pricing: Pricing | None = None,
        escalation: LLM | None = None,
        escalation_max_tokens: int = 4000,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.pricing = pricing or Pricing()
        # A stronger (slower, pricier) model tried once when the primary cannot produce a verified
        # answer although evidence exists. Only hard questions pay for it.
        self.escalation = escalation
        self.escalation_max_tokens = escalation_max_tokens

    def ask(self, question: str, *, now: datetime | None = None, language: Language | None = None) -> Answer:
        t0 = time.perf_counter()
        now = now or self.retriever.clock()
        r = self.retriever.retrieve(question, now=now, language=language)
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

        if self.llm is None:
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
                reason="no LLM configured",
                citations=[_citation(e) for e in evidence],
            )
        first = self._llm_answer(question, r, evidence, now, base, timings, t0, self.llm, 700)
        if first.abstained and self.escalation is not None and "unavailable" not in first.reason:
            log.info("answer.escalate", reason=first.reason)
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
            second.prompt_tokens += first.prompt_tokens
            second.completion_tokens += first.completion_tokens
            second.cost_inr = round(second.cost_inr + first.cost_inr, 5)
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
            cost_inr=self.pricing.cost(prompt_tokens, completion_tokens),
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

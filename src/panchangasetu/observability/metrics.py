"""Prometheus metrics. Labels are low-cardinality by construction: route templates (never raw
paths or session ids), answer methods, languages. Question text is never a label or a log field."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from panchangasetu.generation.answerer import Answer

REGISTRY = CollectorRegistry(auto_describe=True)

HTTP_REQUESTS = Counter(
    "psetu_http_requests_total", "HTTP requests", ["method", "route", "status"], registry=REGISTRY
)
HTTP_SECONDS = Histogram(
    "psetu_http_request_seconds",
    "HTTP request latency",
    ["route"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16),
    registry=REGISTRY,
)
ANSWERS = Counter(
    "psetu_answers_total",
    "Answers by method (template/llm/abstain)",
    ["method", "language"],
    registry=REGISTRY,
)
ANSWER_SECONDS = Histogram(
    "psetu_answer_seconds",
    "Time to answer",
    ["method"],
    buckets=(0.005, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16),
    registry=REGISTRY,
)
ABSTENTIONS = Counter(
    "psetu_abstentions_total", "Questions the documents could not answer", ["reason"], registry=REGISTRY
)
LLM_CALLS = Counter(
    "psetu_llm_calls_total",
    "LLM HTTP calls (retries and fallbacks included)",
    ["provider", "model", "purpose", "status"],
    registry=REGISTRY,
)
LLM_CALL_SECONDS = Histogram(
    "psetu_llm_call_seconds",
    "LLM call latency",
    ["provider", "model"],
    buckets=(0.25, 0.5, 1, 2, 4, 8, 16, 32, 64),
    registry=REGISTRY,
)
LLM_TOKENS = Counter("psetu_llm_tokens_total", "LLM tokens", ["provider", "model", "kind"], registry=REGISTRY)
LLM_COST = Counter("psetu_llm_cost_inr_total", "LLM spend in INR", ["provider", "model"], registry=REGISTRY)
STAGE_SECONDS = Histogram(
    "psetu_stage_seconds",
    "Time per answering stage (rewrite, retrieve, llm, total)",
    ["stage"],
    buckets=(0.005, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16),
    registry=REGISTRY,
)
BUDGET_LIMIT = Gauge("psetu_llm_budget_inr", "Daily LLM budget (0 = unlimited)", registry=REGISTRY)
BUDGET_SPENT = Gauge("psetu_llm_spent_today_inr", "LLM spend since local midnight", registry=REGISTRY)
REWRITES = Counter("psetu_followup_rewrites_total", "Follow-up rewrites", ["method"], registry=REGISTRY)


def abstention_reason(text: str) -> str:
    """Bucket free-text abstention reasons into a few stable labels."""
    for key, label in (
        ("budget", "budget"),
        ("unavailable", "llm_unavailable"),
        ("verification", "verification_failed"),
        ("insufficient", "insufficient_sources"),
        ("no panchanga data", "date_not_covered"),
        ("no relevant", "no_evidence"),
        ("no LLM", "no_llm"),
    ):
        if key in text:
            return label
    return "other"


def record_answer(answer: Answer, rewrite_method: str) -> None:
    ANSWERS.labels(answer.method, answer.language).inc()
    ANSWER_SECONDS.labels(answer.method).observe(answer.latency_ms.get("total", 0.0) / 1000)
    if answer.abstained:
        ABSTENTIONS.labels(abstention_reason(answer.reason)).inc()
    for c in answer.llm_calls:
        LLM_CALLS.labels(c.provider, c.model, c.purpose, c.status).inc()
        LLM_CALL_SECONDS.labels(c.provider, c.model).observe(c.latency_ms / 1000)
        LLM_TOKENS.labels(c.provider, c.model, "prompt").inc(c.prompt_tokens)
        LLM_TOKENS.labels(c.provider, c.model, "completion").inc(c.completion_tokens)
        LLM_COST.labels(c.provider, c.model).inc(c.cost_inr)
    t = answer.latency_ms
    llm_ms = sum(c.latency_ms for c in answer.llm_calls)
    for stage, ms in (
        ("rewrite", t.get("rewrite")),
        ("retrieve", t.get("retrieve")),
        ("total", t.get("total")),
    ):
        if ms is not None:
            STAGE_SECONDS.labels(stage).observe(ms / 1000)
    if answer.llm_calls:
        STAGE_SECONDS.labels("llm").observe(llm_ms / 1000)
    REWRITES.labels(rewrite_method).inc()


def record_budget(limit_inr: float, spent_inr: float) -> None:
    BUDGET_LIMIT.set(limit_inr)
    BUDGET_SPENT.set(spent_inr)

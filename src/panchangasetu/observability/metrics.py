"""Prometheus metrics. Labels are low-cardinality by construction: route templates (never raw
paths or session ids), answer methods, languages. Question text is never a label or a log field."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram

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
LLM_TOKENS = Counter("psetu_llm_tokens_total", "LLM tokens", ["kind"], registry=REGISTRY)
LLM_COST = Counter("psetu_llm_cost_inr_total", "LLM spend in INR", registry=REGISTRY)
REWRITES = Counter("psetu_followup_rewrites_total", "Follow-up rewrites", ["method"], registry=REGISTRY)


def _reason(text: str) -> str:
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
        ABSTENTIONS.labels(_reason(answer.reason)).inc()
    if answer.prompt_tokens or answer.completion_tokens:
        LLM_TOKENS.labels("prompt").inc(answer.prompt_tokens)
        LLM_TOKENS.labels("completion").inc(answer.completion_tokens)
    if answer.cost_inr:
        LLM_COST.inc(answer.cost_inr)
    REWRITES.labels(rewrite_method).inc()

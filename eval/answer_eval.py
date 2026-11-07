"""End-to-end answer evaluation: correct facts, correct abstentions, language, cost, latency.

Usage: uv run python eval/answer_eval.py [--out eval/reports/answer_eval.json]
Costs provider credits for LLM-path questions (~₹0.05 each with sarvam-105b-conversations).

Scoring per question:
  expect=answer  : answered (not abstained) AND every "must" group has at least one alternative
                   present in the answer  -> correct
  expect=abstain : abstained                -> correct (answering = a potential hallucination)
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from panchangasetu.cli import build_answer_service
from panchangasetu.config import get_settings
from panchangasetu.query.language import detect_language

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "eval/reports/answer_eval.json")
    args = ap.parse_args()
    settings = get_settings()
    service = build_answer_service(settings)
    tz = ZoneInfo(settings.timezone)
    rows = [json.loads(x) for x in (ROOT / "eval/datasets/answers_v1.jsonl").read_text().splitlines() if x.strip()]
    results = []
    for row in rows:
        now = datetime.fromisoformat(row["now"]).replace(tzinfo=tz)
        a = service.ask(row["q"], now=now)
        text = a.answer
        if row["expect"] == "abstain":
            correct = a.abstained
            missing: list[list[str]] = []
        else:
            missing = [g for g in row["must"] if not any(alt.lower() in text.lower() for alt in g)]
            correct = not a.abstained and not missing
        lang_ok = a.abstained or detect_language(text).code == detect_language(row["q"]).code
        results.append({
            "id": row["id"], "q": row["q"], "expect": row["expect"], "correct": correct, "language_ok": lang_ok,
            "method": a.method, "abstained": a.abstained, "reason": a.reason, "missing": missing,
            "cost_inr": a.cost_inr, "latency_ms": round(a.latency_ms.get("total", 0)), "answer": text,
        })
        mark = "OK " if correct else "BAD"
        print(f"{mark} {row['id']} [{a.method:8}] {row['q'][:55]:55} ₹{a.cost_inr:.3f} {a.latency_ms.get('total', 0):6.0f}ms"
              + ("" if correct else f"  missing={missing} reason={a.reason[:80]!r}"))

    answerable = [r for r in results if r["expect"] == "answer"]
    negatives = [r for r in results if r["expect"] == "abstain"]
    summary = {
        "answer_accuracy": round(sum(r["correct"] for r in answerable) / len(answerable), 3),
        "abstention_accuracy": round(sum(r["correct"] for r in negatives) / len(negatives), 3),
        "hallucinated_answers_on_negatives": sum(not r["abstained"] for r in negatives),
        "false_abstentions": sum(r["abstained"] for r in answerable),
        "language_match": round(sum(r["language_ok"] for r in results) / len(results), 3),
        "methods": {m: sum(r["method"] == m for r in results) for m in ("template", "llm", "abstain")},
        "total_cost_inr": round(sum(r["cost_inr"] for r in results), 3),
        "median_latency_ms": statistics.median(r["latency_ms"] for r in results),
        "p95_latency_ms": sorted(r["latency_ms"] for r in results)[int(0.95 * len(results)) - 1],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

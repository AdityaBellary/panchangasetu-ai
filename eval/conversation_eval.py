"""Multi-turn evaluation: are follow-ups rewritten correctly, and are they answered correctly?

Usage:
  uv run python eval/conversation_eval.py            # rewrite check only: free, no LLM, no database
  uv run python eval/conversation_eval.py --answer   # also answer every turn (Sarvam credits, ~₹1)

Scoring per follow-up turn:
  rewrite : every "standalone" substring appears in the standalone question (case-insensitive)
  answer  : like eval/answer_eval.py: not abstained and every "must" group present
            (or abstained when "expect" is "abstain")
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from panchangasetu.config import get_settings
from panchangasetu.conversation.followup import FollowUpResolver
from panchangasetu.conversation.store import TurnRecord

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answer", action="store_true", help="answer every turn (uses LLM credits)")
    ap.add_argument("--out", type=Path, default=ROOT / "eval/reports/conversation_eval.json")
    args = ap.parse_args()
    settings = get_settings()
    tz = ZoneInfo(settings.timezone)
    convs = [
        json.loads(x)
        for x in (ROOT / "eval/datasets/conversations_v1.jsonl").read_text().splitlines()
        if x.strip()
    ]

    service = None
    if args.answer:
        from panchangasetu.cli import build_answer_service
        from panchangasetu.conversation.service import ConversationService
        from panchangasetu.conversation.store import InMemorySessionStore

        answers = build_answer_service(settings)
        service = ConversationService(answers, InMemorySessionStore(), FollowUpResolver(answers.llm))

    resolver = FollowUpResolver(None)  # rewrite check: rules only, free
    results = []
    for conv in convs:
        now = datetime.fromisoformat(conv["now"]).replace(tzinfo=tz)
        history: list[TurnRecord] = []
        session_id = service.start() if service else None
        for i, t in enumerate(conv["turns"]):
            row: dict[str, object] = {"id": f"{conv['id']}.{i + 1}", "q": t["q"], "follow_up": i > 0}
            if service is not None:
                a = service.ask(t["q"], session_id=session_id, now=now)
                standalone = a.standalone_question or t["q"]
                text = a.answer
                if t.get("expect") == "abstain":
                    ok = a.abstained
                else:
                    ok = not a.abstained and all(any(x.lower() in text.lower() for x in g) for g in t["must"])
                row |= {"answer_ok": ok, "answer": text, "method": a.method, "cost_inr": a.cost_inr}
            else:
                standalone = resolver.resolve(t["q"], history, now).question
                cited = [{"date": d} for d in t.get("cited", [])]
                history.append(TurnRecord(t["q"], standalone, "", "", "", False, "", "", now, cited))
            want = t.get("standalone", [])
            row |= {
                "standalone": standalone,
                "rewrite_ok": all(w.lower() in standalone.lower() for w in want),
            }
            results.append(row)
            mark = "OK " if row["rewrite_ok"] and row.get("answer_ok", True) else "BAD"
            print(f"{mark} {row['id']:6} {t['q'][:40]:40} -> {standalone[:70]}")
            if service is not None and not row["answer_ok"]:
                print(f"       answer: {str(row['answer'])[:160]!r}")

    follow = [r for r in results if r["follow_up"]]
    summary: dict[str, object] = {
        "follow_ups": len(follow),
        "rewrite_accuracy": round(sum(bool(r["rewrite_ok"]) for r in follow) / len(follow), 3),
    }
    if service is not None:
        summary |= {
            "answer_accuracy_all_turns": round(sum(bool(r["answer_ok"]) for r in results) / len(results), 3),
            "answer_accuracy_follow_ups": round(sum(bool(r["answer_ok"]) for r in follow) / len(follow), 3),
            "total_cost_inr": round(sum(float(r["cost_inr"]) for r in results), 3),  # type: ignore[arg-type]
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

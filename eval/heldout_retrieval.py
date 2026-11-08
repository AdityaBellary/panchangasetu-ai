"""Free retrieval check on the held-out prose set: does the passage a question was written from
reach the sources the answering model sees?

Usage: uv run python eval/heldout_retrieval.py [--dataset eval/datasets/heldout_prose_v1.jsonl]

hit@5   the gold page is among the 5 sources given to the LLM (what decides answer quality)
hit@all the gold page was retrieved at all (before the 5-source cut)
MRR     1/rank of the first gold source among the 5
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from panchangasetu.cli import build_retriever
from panchangasetu.config import get_settings
from panchangasetu.generation.answerer import _MAX_EVIDENCE, select_evidence

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=ROOT / "eval/datasets/heldout_prose_v1.jsonl")
    args = ap.parse_args()
    settings = get_settings()
    retriever = build_retriever(settings)
    tz = ZoneInfo(settings.timezone)
    rows = [json.loads(x) for x in args.dataset.read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r.get("gold_pages")]
    hits5 = hits_all = 0
    rr = 0.0
    for row in rows:
        gold = set(row["gold_pages"])
        r = retriever.retrieve(row["q"], now=datetime.fromisoformat(row["now"]).replace(tzinfo=tz))
        shown = select_evidence(r.evidence, _MAX_EVIDENCE)
        rank = next((i + 1 for i, e in enumerate(shown) if gold & set(e.pages)), None)
        found = any(gold & set(e.pages) for e in r.evidence)
        hits5 += rank is not None
        hits_all += found
        rr += 1 / rank if rank else 0.0
        mark = "OK " if rank else ("CUT" if found else "MISS")
        pages = [e.pages[0] for e in shown if e.pages]
        route = f"{r.route.intent.value}/{r.route.mode.value}"
        where = f"gold={sorted(gold)} got={pages} {route}"
        print(f"{mark:4} {row['id']} rank={rank or '-':2} {where}  {row['q'][:60]}")
    n = len(rows)
    print(
        json.dumps(
            {
                "questions": n,
                "hit@5": round(hits5 / n, 3),
                "hit@all": round(hits_all / n, 3),
                "mrr@5": round(rr / n, 3),
            }
        )
    )


if __name__ == "__main__":
    main()

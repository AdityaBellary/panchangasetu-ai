"""Evaluate retrieval through the production path (Qdrant aliases + VectorIndex.search).

Usage:  uv run python eval/retrieval_eval.py [--k 5] [--out eval/reports/retrieval_eval.json]

Queries whose `collection` is general/panchanga are searched in that collection (Phase 4 adds
automatic routing; this isolates retrieval quality). Negative queries are searched in both
collections to report how high irrelevant scores go.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from panchangasetu.cli import build_vector_index
from panchangasetu.config import get_settings
from panchangasetu.query.language import detect_language
from panchangasetu.query.rewrite import Glossary, expand_query
from panchangasetu.query.slots import extract_slots

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", type=Path, default=ROOT / "eval/reports/retrieval_eval.json")
    ap.add_argument("--expand", action="store_true", help="apply Phase 4 query expansion (glossary)")
    args = ap.parse_args()
    settings = get_settings()
    index = build_vector_index(settings)
    glossary = Glossary.load(settings.query_glossary_path)
    queries = [
        json.loads(line)
        for line in (ROOT / "eval/datasets/retrieval_v1.jsonl").read_text().splitlines()
        if line.strip()
    ]

    metrics: dict[str, list[float]] = defaultdict(list)
    latencies: list[float] = []
    misses: list[dict[str, object]] = []
    neg_scores: list[float] = []
    for q in queries:
        collections = [q["collection"]] if q["gold_pages"] else ["general", "panchanga"]
        for collection in collections:
            t0 = time.perf_counter()
            text = q["query"]
            if args.expand:
                text = expand_query(text, detect_language(text), extract_slots(text), glossary)
            hits = index.search(
                collection,
                text,
                top_k=10,
                candidates=settings.retrieval_candidates,
                hybrid_min_telugu_ratio=settings.hybrid_min_telugu_ratio,
            )
            latencies.append((time.perf_counter() - t0) * 1000)
            if not q["gold_pages"]:
                if hits:
                    neg_scores.append(hits[0].score)
                continue
            gold = set(q["gold_pages"])
            found = [bool(gold & set((h.payload or {}).get("pages", []))) for h in hits]
            r = float(any(found[: args.k]))
            mrr = next((1.0 / (i + 1) for i, f in enumerate(found) if f), 0.0)
            for key in ("all", q["lang"], q["collection"]):
                metrics[f"{key}:recall@{args.k}"].append(r)
                metrics[f"{key}:mrr@10"].append(mrr)
            if not r:
                top = [(h.payload or {}).get("pages", [None])[0] for h in hits[: args.k]]
                misses.append(
                    {
                        "id": q["id"],
                        "lang": q["lang"],
                        "query": q["query"],
                        "gold": q["gold_pages"],
                        "top_pages": top,
                    }
                )

    latencies.sort()
    report = {
        "embedding_model": settings.embedding_model,
        "query_expansion": args.expand,
        "chunker_version": settings.chunker_version,
        **{k: round(statistics.mean(v), 3) for k, v in sorted(metrics.items())},
        "latency_ms_p50": round(latencies[len(latencies) // 2], 1),
        "latency_ms_p95": round(latencies[int(len(latencies) * 0.95) - 1], 1),
        "negative_top1_scores": [round(s, 3) for s in neg_scores],
        "misses": misses,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "misses"}, ensure_ascii=False, indent=2))
    print(f"misses: {len(misses)}")
    for m in misses:
        print(f"  {m['id']} [{m['lang']}] {m['query']}  gold={m['gold']} got={m['top_pages']}")


if __name__ == "__main__":
    main()

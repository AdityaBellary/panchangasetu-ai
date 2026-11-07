"""Routing accuracy on eval/datasets/routing_v1.jsonl (intent and panchanga mode), fully local.

Usage: uv run python eval/routing_eval.py [--rules-only]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

from panchangasetu.config import get_settings
from panchangasetu.providers.embeddings.factory import build_embedder
from panchangasetu.query.router import EmbeddingRouter, QueryRouter

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules-only", action="store_true")
    ap.add_argument("--today", default="2026-11-07")
    args = ap.parse_args()
    settings = get_settings()
    semantic = (
        None
        if args.rules_only
        else EmbeddingRouter.from_yaml(build_embedder(settings), ROOT / "config/router_examples.yaml")
    )
    router = QueryRouter(semantic)
    rows = [
        json.loads(line)
        for line in (ROOT / "eval/datasets/routing_v1.jsonl").read_text().splitlines()
        if line
    ]
    today = date.fromisoformat(args.today)
    intent_ok = mode_ok = 0
    confusion: Counter[tuple[str, str]] = Counter()
    for row in rows:
        r = router.route(row["q"], today)
        ok_i = r.intent.value == row["intent"]
        ok_m = ok_i and (row["intent"] == "general" or r.mode.value == row["mode"])
        intent_ok += ok_i
        mode_ok += ok_m
        confusion[(row["intent"], r.intent.value)] += 1
        if not ok_m:
            print(
                f"  MISS {row['q']!r}: expected {row['intent']}/{row['mode']}, got "
                f"{r.intent.value}/{r.mode.value} via {r.reasons}"
            )
    n = len(rows)
    print(f"intent accuracy {intent_ok / n:.3f}  intent+mode accuracy {mode_ok / n:.3f}  (n={n})")
    print("confusion (expected -> got):", dict(confusion))


if __name__ == "__main__":
    main()

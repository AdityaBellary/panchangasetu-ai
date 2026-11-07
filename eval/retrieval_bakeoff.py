"""Embedding bake-off: which dense model (alone or hybrid with Telugu sparse) retrieves best?

Usage:  uv run python eval/retrieval_bakeoff.py [--models ...] [--out eval/reports/bakeoff.json]

Corpus: current indexable chunks from Postgres. Ground truth: eval/datasets/retrieval_v1.jsonl
(page-level: a hit is a retrieved chunk whose page span intersects the gold pages).
Everything runs locally; no provider credits are used.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sqlalchemy import select

from panchangasetu.config import get_settings
from panchangasetu.providers.embeddings.fastembed_dense import FastEmbedDense
from panchangasetu.providers.embeddings.openai_dense import OpenAIEmbedder
from panchangasetu.providers.embeddings.telugu_sparse import TeluguSparseEncoder
from panchangasetu.retrieval.documents import STRUCTURED_ONLY_SECTIONS, render_for_embedding
from panchangasetu.storage.db import make_engine, make_session_factory
from panchangasetu.storage.models import ChunkRecord, DocumentVersion

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = [
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-large",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
]


def load_corpus() -> list[dict]:
    settings = get_settings()
    sessions = make_session_factory(make_engine(settings.database_url))
    with sessions() as s:
        rows = s.scalars(
            select(ChunkRecord)
            .join(DocumentVersion, DocumentVersion.id == ChunkRecord.version_id)
            .where(
                DocumentVersion.is_current.is_(True),
                ChunkRecord.chunker_version == settings.chunker_version,
                ChunkRecord.section_type.not_in(STRUCTURED_ONLY_SECTIONS),
            )
        ).all()
        return [
            {
                "id": str(c.id),
                "section": c.section_type,
                "headings": c.heading_path,
                "text": c.text,
                "pages": set(range(c.page_start, c.page_end + 1)),
                "collection": c.collection,
            }
            for c in rows
        ]


def sparse_scores(query_vec, doc_vecs, idf) -> np.ndarray:
    qi = dict(zip(*query_vec, strict=True))
    out = np.zeros(len(doc_vecs))
    for d, (idx, vals) in enumerate(doc_vecs):
        out[d] = sum(v * idf.get(i, 0.0) for i, v in zip(idx, vals, strict=True) if i in qi)
    return out


def rrf(*rankings: np.ndarray, k: int = 60) -> np.ndarray:
    score = defaultdict(float)
    for ranking in rankings:
        for pos, doc in enumerate(ranking[:100]):
            score[int(doc)] += 1.0 / (k + pos + 1)
    return np.array(sorted(score, key=score.__getitem__, reverse=True))


def evaluate(rankings: dict[int, np.ndarray], queries: list[dict], corpus: list[dict]) -> dict:
    res: dict[str, list[float]] = defaultdict(list)
    for qi, q in enumerate(queries):
        if not q["gold_pages"]:
            continue
        gold = set(q["gold_pages"])
        ranked = rankings[qi]
        hits = [bool(corpus[d]["pages"] & gold) for d in ranked[:10]]
        r5 = float(any(hits[:5]))
        mrr = next((1.0 / (i + 1) for i, h in enumerate(hits) if h), 0.0)
        for key in ("all", q["lang"]):
            res[f"{key}:recall@5"].append(r5)
            res[f"{key}:mrr@10"].append(mrr)
    return {k: round(float(np.mean(v)), 3) for k, v in sorted(res.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--out", type=Path, default=ROOT / "eval/reports/bakeoff_v1.json")
    ap.add_argument("--gloss-only", action="store_true", help="skip the no-gloss variant")
    args = ap.parse_args()

    corpus = load_corpus()
    queries = [
        json.loads(line)
        for line in (ROOT / "eval/datasets/retrieval_v1.jsonl").read_text().splitlines()
        if line
    ]
    print(f"corpus={len(corpus)} chunks, queries={len(queries)}")

    sparse = TeluguSparseEncoder()
    report: dict[str, dict] = {}
    for gloss in (True,) if args.gloss_only else (False, True):
        docs = [render_for_embedding(c["section"], c["headings"], c["text"], gloss=gloss) for c in corpus]
        dvecs = sparse.encode_documents(docs)
        df: dict[int, int] = defaultdict(int)
        for idx, _ in dvecs:
            for i in idx:
                df[i] += 1
        idf = {i: float(np.log(1 + (len(docs) - n + 0.5) / (n + 0.5))) for i, n in df.items()}
        sparse_rank = {
            qi: np.argsort(-sparse_scores(sparse.encode_query(q["query"]), dvecs, idf))
            for qi, q in enumerate(queries)
        }
        report[f"sparse{'+gloss' if gloss else ''}"] = evaluate(sparse_rank, queries, corpus)

        for model in args.models:
            if model.startswith("openai:"):
                key = get_settings().openai_api_key
                assert key is not None, "OPENAI_API_KEY is not set"
                emb = OpenAIEmbedder(key.get_secret_value(), model.split(":", 1)[1])
            else:
                emb = FastEmbedDense(model, cache_dir=ROOT / "data/models")
            t0 = time.time()
            D = np.array(emb.embed_documents(docs))
            index_s = time.time() - t0
            t0 = time.time()
            Q = np.array([emb.embed_query(q["query"]) for q in queries])
            query_ms = (time.time() - t0) / len(queries) * 1000
            S = Q @ D.T
            dense_rank = {qi: np.argsort(-S[qi]) for qi in range(len(queries))}
            hybrid_rank = {qi: rrf(dense_rank[qi], sparse_rank[qi]) for qi in range(len(queries))}
            tag = f"{model.split('/')[-1]}{'+gloss' if gloss else ''}"
            pos_top = [float(S[qi].max()) for qi, q in enumerate(queries) if q["gold_pages"]]
            neg_top = [float(S[qi].max()) for qi, q in enumerate(queries) if not q["gold_pages"]]
            report[tag] = {
                **evaluate(dense_rank, queries, corpus),
                "index_s": round(index_s, 1),
                "query_ms": round(query_ms, 1),
                "top1_score_pos_min/median": [round(min(pos_top), 3), round(float(np.median(pos_top)), 3)],
                "top1_score_neg_max": round(max(neg_top), 3),
            }
            report[f"{tag} + sparse (RRF)"] = evaluate(hybrid_rank, queries, corpus)
            if isinstance(emb, OpenAIEmbedder):
                report[tag]["openai_tokens"] = emb.tokens_used
            print(tag, report[tag], "\n  hybrid", report[f"{tag} + sparse (RRF)"], flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("\n=== summary (all:recall@5 / en / te / all:mrr@10)")
    for name, r in report.items():
        print(
            f"{name:62} {r['all:recall@5']:.3f}  en={r['en:recall@5']:.3f}  te={r['te:recall@5']:.3f}  "
            f"mrr={r['all:mrr@10']:.3f}"
        )


if __name__ == "__main__":
    main()

"""Operator CLI: `psetu ingest`, `psetu status`, `psetu show-page`, `psetu db-upgrade`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sqlalchemy import func, select

from panchangasetu.config import Settings, get_settings
from panchangasetu.generation.answerer import Answer, AnswerService
from panchangasetu.ingestion.pipeline import IngestionPipeline
from panchangasetu.observability.logging import configure_logging
from panchangasetu.providers.ocr.factory import build_ocr_engine
from panchangasetu.providers.ocr.tesseract import TesseractOCREngine
from panchangasetu.retrieval.orchestrator import Retriever
from panchangasetu.retrieval.vector_store import VectorIndex, describe
from panchangasetu.storage.db import make_engine, make_session_factory
from panchangasetu.storage.models import ChunkRecord, Document, DocumentVersion, Page
from panchangasetu.storage.object_store import LocalObjectStore
from panchangasetu.structured.service import StructuredExtractor


def build_pipeline(settings: Settings) -> IngestionPipeline:
    sessions = make_session_factory(make_engine(settings.database_url))
    return IngestionPipeline(
        settings,
        sessions,
        LocalObjectStore(settings.object_store_dir),
        engine_factory=lambda name: build_ocr_engine(name, settings),
    )


def build_structured(settings: Settings) -> StructuredExtractor:
    sessions = make_session_factory(make_engine(settings.database_url))

    def row_engine(psm: int) -> TesseractOCREngine:
        return TesseractOCREngine(
            cmd=settings.tesseract_cmd,
            langs=settings.structured_row_langs,
            tessdata_dir=settings.tesseract_data_dir,
            timeout_s=settings.tesseract_timeout_s,
            psm=psm,
        )

    cell_engines = {
        (langs, psm): TesseractOCREngine(
            cmd=settings.tesseract_cmd,
            langs=langs,
            tessdata_dir=settings.tesseract_data_dir,
            timeout_s=settings.tesseract_timeout_s,
            psm=psm,
        )
        for langs, psm in (("eng", 11), ("tel", 6))
    }

    def tesseract_runner(png: bytes, langs: str, psm: int) -> str:
        engine = cell_engines[(langs, psm)]
        return engine.raw_tsv(png) if langs == "eng" else engine.raw_text(png)

    return StructuredExtractor(
        sessions,
        LocalObjectStore(settings.object_store_dir),
        settings.structured_sources_path,
        row_engine,
        tesseract_runner,
    )


def cmd_extract_structured(settings: Settings, args: argparse.Namespace) -> int:
    stats = build_structured(settings).run(force=args.force)
    print(
        json.dumps(
            {
                "days_written": stats.days_written,
                "events_written": stats.events_written,
                "sources_skipped": stats.sources_skipped,
                "problems": stats.problems[:30],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_vector_index(settings: Settings) -> VectorIndex:
    from qdrant_client import QdrantClient

    from panchangasetu.providers.embeddings.factory import build_embedder
    from panchangasetu.providers.embeddings.telugu_sparse import TeluguSparseEncoder

    key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    return VectorIndex(
        QdrantClient(url=settings.qdrant_url, api_key=key),
        make_session_factory(make_engine(settings.database_url)),
        build_embedder(settings),
        TeluguSparseEncoder(),
        prefix=settings.index_prefix,
        keep_previous=settings.keep_previous_index_builds,
    )


def cmd_index(settings: Settings, args: argparse.Namespace) -> int:
    index = build_vector_index(settings)
    report = index.build(
        chunker_version=settings.chunker_version, gloss=settings.index_section_gloss, force=args.force
    )
    print(
        json.dumps(
            {
                "built": report.built,
                "skipped": report.skipped,
                "retired": report.retired,
                "collections": describe(index),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_search(settings: Settings, args: argparse.Namespace) -> int:
    index = build_vector_index(settings)
    hits = index.search(
        args.collection,
        args.query,
        top_k=args.k,
        candidates=settings.retrieval_candidates,
        hybrid_min_telugu_ratio=settings.hybrid_min_telugu_ratio,
    )
    for h in hits:
        p = h.payload or {}
        where = f"{p.get('document')} p{p.get('page_start')}" + (f" {p.get('date')}" if p.get("date") else "")
        tag = f"[{p.get('kind')}/{p.get('section_type')}]"
        print(f"{h.score:.3f}  {tag} {where}")
        print(f"       {str(p.get('text'))[:160]!r}")
    return 0


def build_retriever(settings: Settings) -> Retriever:
    from panchangasetu.query.dates import system_clock
    from panchangasetu.query.rewrite import Glossary
    from panchangasetu.query.router import EmbeddingRouter, QueryRouter
    from panchangasetu.retrieval.structured import StructuredRetriever

    vectors = build_vector_index(settings)
    semantic = EmbeddingRouter.from_yaml(vectors.embedder, settings.router_examples_path)
    return Retriever(
        QueryRouter(semantic, min_confidence=settings.router_min_confidence),
        vectors,
        StructuredRetriever(make_session_factory(make_engine(settings.database_url))),
        system_clock(settings.timezone),
        Glossary.load(settings.query_glossary_path),
        top_k=settings.retrieval_top_k,
        candidates=settings.retrieval_candidates,
        hybrid_min_telugu_ratio=settings.hybrid_min_telugu_ratio,
    )


def cmd_retrieve(settings: Settings, args: argparse.Namespace) -> int:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    retriever = build_retriever(settings)
    now = datetime.fromisoformat(args.now).replace(tzinfo=ZoneInfo(settings.timezone)) if args.now else None
    for question in args.questions:
        r = retriever.retrieve(question, now=now)
        print(f"\n=== {question}")
        print(
            f"route: {r.route.intent.value}/{r.route.mode.value} lang={r.route.language.code} "
            f"conf={r.route.confidence:.2f} {r.route.reasons}"
        )
        print(
            f"dates: {[(d.start.isoformat(), d.end.isoformat()) for d in r.route.dates]}  "
            f"query: {r.search_query!r}  timings: { {k: round(v) for k, v in r.timings_ms.items()} }"
        )
        for note in r.notes:
            print(f"note: {note}")
        for e in r.evidence[: args.k]:
            flag = {True: "verified", False: "UNVERIFIED", None: "prose"}[e.verified]
            print(
                f"- [{e.kind}/{e.source_kind}] {e.citation()} {flag}"
                + (f" score={e.score:.3f}" if e.score is not None else "")
            )
            print("    " + e.text[:300].replace("\n", "\n    "))
    return 0


def build_answer_service(settings: Settings) -> AnswerService:
    from panchangasetu.providers.llm.factory import build_llm
    from panchangasetu.providers.llm.openai_compatible import sarvam_chat

    escalation = None
    key = settings.sarvam_api_key.get_secret_value() if settings.sarvam_api_key else ""
    if settings.escalation_model and key:
        escalation = sarvam_chat(key, settings.escalation_model, timeout_s=max(settings.llm_timeout_s, 180))
    return AnswerService(build_retriever(settings), build_llm(settings), escalation=escalation)


def _print_answer(a: AnswerService | None, ans: Answer, verbose: bool) -> None:
    print(ans.answer)
    if ans.citations and not ans.abstained:
        print("\nSources:")
        for c in ans.citations:
            pages = ", ".join(str(p) for p in c.pages[:3])
            flag = {True: "", False: " (unverified)", None: ""}[c.verified]
            print(
                f"  - {c.document}"
                + (f", p. {pages}" if pages else "")
                + (f", {c.date}" if c.date else "")
                + flag
            )
    if verbose:
        cost = f", cost ₹{ans.cost_inr:.4f}" if ans.cost_inr else ""
        print(
            f"\n[{ans.method} | route {ans.route} | {ans.model or 'no LLM'}{cost} | "
            f"{ans.latency_ms.get('total', 0):.0f} ms" + (f" | {ans.reason}" if ans.reason else "") + "]"
        )


def cmd_ask(settings: Settings, args: argparse.Namespace) -> int:
    service = build_answer_service(settings)
    for q in args.questions:
        print(f"\n> {q}")
        _print_answer(service, service.ask(q), args.verbose)
    return 0


def cmd_chat(settings: Settings, args: argparse.Namespace) -> int:
    service = build_answer_service(settings)
    print("PanchangaSetu chat. Ask in Telugu or English. Empty line or Ctrl-D to exit.")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if not q:
            break
        _print_answer(service, service.ask(q), args.verbose)
    return 0


def cmd_db_upgrade(settings: Settings, _: argparse.Namespace) -> int:
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(Path(__file__).resolve().parents[2] / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", settings.database_url)
    command.upgrade(cfg, "head")
    return 0


def cmd_ingest(settings: Settings, args: argparse.Namespace) -> int:
    if args.engine:
        settings = settings.model_copy(update={"ocr_engine": args.engine})
    pipeline = build_pipeline(settings)
    if args.rederive:
        print(f"re-derived text for {pipeline.rederive_from_payloads()} pages from stored OCR payloads")
    run_id, stats = pipeline.run(
        trigger="cli", reprocess=args.reprocess, rechunk=args.rechunk or args.rederive, only=args.only
    )
    # Structured facts (free, local) are refreshed after every ingest; up-to-date sources skip.
    structured = build_structured(settings).run(force=args.reprocess)
    print(
        json.dumps(
            {
                "run_id": str(run_id),
                **stats.as_dict(),
                "errors": stats.errors[:20],
                "structured": {
                    "days_written": structured.days_written,
                    "events_written": structured.events_written,
                    "sources_skipped": structured.sources_skipped,
                    "problems": len(structured.problems),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if stats.errors else 0


def cmd_status(settings: Settings, _: argparse.Namespace) -> int:
    sessions = make_session_factory(make_engine(settings.database_url))
    with sessions() as s:
        rows = s.execute(
            select(Document, DocumentVersion)
            .join(DocumentVersion, DocumentVersion.document_id == Document.id)
            .where(DocumentVersion.is_current.is_(True))
            .order_by(Document.source_path)
        ).all()
        for doc, v in rows:
            by_coll = dict(
                s.execute(
                    select(Page.collection, func.count())
                    .where(Page.version_id == v.id)
                    .group_by(Page.collection)
                ).all()
            )
            engines = dict(
                s.execute(
                    select(Page.ocr_engine, func.count())
                    .where(Page.version_id == v.id)
                    .group_by(Page.ocr_engine)
                ).all()
            )
            failed = s.scalar(select(func.count()).where(Page.version_id == v.id, Page.error.is_not(None)))
            chunks = dict(
                s.execute(
                    select(ChunkRecord.collection, func.count())
                    .where(
                        ChunkRecord.version_id == v.id,
                        ChunkRecord.chunker_version == settings.chunker_version,
                    )
                    .group_by(ChunkRecord.collection)
                ).all()
            )
            print(
                f"{doc.source_path}\n  status={v.status} pages={v.page_count} sha={v.sha256[:12]} "
                f"failed_pages={failed}\n  pages_by_collection={by_coll} engines={engines}\n  chunks={chunks}"
            )
    return 0


def cmd_show_page(settings: Settings, args: argparse.Namespace) -> int:
    sessions = make_session_factory(make_engine(settings.database_url))
    with sessions() as s:
        page = s.scalar(
            select(Page)
            .join(DocumentVersion, DocumentVersion.id == Page.version_id)
            .join(Document, Document.id == DocumentVersion.document_id)
            .where(
                Document.filename == args.document,
                DocumentVersion.is_current.is_(True),
                Page.page_number == args.page,
            )
        )
        if page is None:
            print("not found", file=sys.stderr)
            return 1
        print(
            f"# kind={page.page_kind} rot={page.rotation_applied} engine={page.ocr_engine} "
            f"conf={page.ocr_confidence} collection={page.collection}/{page.section_type} "
            f"({page.classification_reason})"
        )
        print(page.text or f"<no text> error={page.error}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="psetu", description="PanchangaSetu AI operator CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("db-upgrade", help="apply database migrations")
    ing = sub.add_parser("ingest", help="discover and ingest PDFs from the input directory")
    ing.add_argument(
        "--reprocess", action="store_true", help="re-extract every page (re-runs OCR; costs credits)"
    )
    ing.add_argument("--rechunk", action="store_true", help="re-classify and re-chunk from cached page text")
    ing.add_argument("--only", nargs="+", help="limit to these file names / relative paths")
    ing.add_argument(
        "--rederive",
        action="store_true",
        help="rebuild page text from stored OCR payloads first (free); implies --rechunk",
    )
    ing.add_argument("--engine", choices=["sarvam", "tesseract"], help="override the default OCR engine")
    es = sub.add_parser(
        "extract-structured", help="(re)build panchanga_days / panchanga_events (free, local)"
    )
    es.add_argument("--force", action="store_true", help="re-extract even if up to date")
    ix = sub.add_parser("index", help="build/refresh the vector index (alias swap, zero downtime)")
    ix.add_argument("--force", action="store_true", help="rebuild even if the corpus is unchanged")
    se = sub.add_parser("search", help="debug retrieval against the vector index")
    se.add_argument("collection", choices=["general", "panchanga"])
    se.add_argument("query")
    se.add_argument("-k", type=int, default=5)
    rt = sub.add_parser("retrieve", help="debug: route a question and show the evidence it gets")
    rt.add_argument("questions", nargs="+")
    rt.add_argument("-k", type=int, default=4, help="evidence items to show")
    rt.add_argument("--now", help="pretend the current local time is this ISO datetime")
    ak = sub.add_parser("ask", help="answer one or more questions (grounded, with sources)")
    ak.add_argument("questions", nargs="+")
    ak.add_argument("-v", "--verbose", action="store_true", help="show method, route, model, cost, latency")
    ch = sub.add_parser("chat", help="interactive question answering in the terminal")
    ch.add_argument("-v", "--verbose", action="store_true")
    sub.add_parser("status", help="show ingestion status per document")
    sp = sub.add_parser("show-page", help="print the extracted text of one page")
    sp.add_argument("document")
    sp.add_argument("page", type=int)
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level, json=settings.log_json)
    handlers = {
        "db-upgrade": cmd_db_upgrade,
        "ingest": cmd_ingest,
        "extract-structured": cmd_extract_structured,
        "index": cmd_index,
        "search": cmd_search,
        "retrieve": cmd_retrieve,
        "ask": cmd_ask,
        "chat": cmd_chat,
        "status": cmd_status,
        "show-page": cmd_show_page,
    }
    return handlers[args.cmd](settings, args)


if __name__ == "__main__":
    raise SystemExit(main())

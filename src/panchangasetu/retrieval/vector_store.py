"""Qdrant integration: aliased, versioned collections and hybrid search.

Each logical collection (general, panchanga) is served through an alias
`<prefix>_<collection>`. A build writes a brand-new physical collection
`<prefix>_<collection>__<model-slug>__<chunker>__<timestamp>`, then swaps the alias atomically,
so re-indexing or changing the embedding model has zero downtime and can be rolled back.

Vectors per point:
  dense   - normalised embedding (cosine)
  lexical - Telugu sparse features, IDF applied by Qdrant (Modifier.IDF)
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from qdrant_client import QdrantClient, models
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from panchangasetu.core.interfaces import Embedder, SparseEncoder
from panchangasetu.observability.logging import get_logger
from panchangasetu.retrieval.corpus import iter_documents
from panchangasetu.retrieval.documents import IndexDocument
from panchangasetu.storage.db import session_scope
from panchangasetu.storage.models import IndexBuild, utcnow
from panchangasetu.telugu.text import script_stats

log = get_logger(__name__)

COLLECTIONS = ("general", "panchanga")
DENSE, LEXICAL = "dense", "lexical"
_UPSERT_BATCH = 64


def alias_name(prefix: str, collection: str) -> str:
    return f"{prefix}_{collection}"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40]


def fingerprint(docs: Sequence[IndexDocument], embed_name: str, sparse_name: str) -> str:
    h = hashlib.sha256(f"{embed_name}|{sparse_name}".encode())
    for d in sorted(docs, key=lambda d: d.id):
        h.update(d.id.encode())
        h.update(hashlib.sha256(d.text.encode()).digest())
    return h.hexdigest()


@dataclass(slots=True)
class BuildReport:
    built: dict[str, int] = field(default_factory=dict)  # alias -> doc count
    skipped: list[str] = field(default_factory=list)
    retired: list[str] = field(default_factory=list)


class VectorIndex:
    def __init__(
        self,
        client: QdrantClient,
        sessions: sessionmaker[Session],
        embedder: Embedder,
        sparse: SparseEncoder,
        *,
        prefix: str = "psetu",
        keep_previous: int = 1,
    ) -> None:
        self.client = client
        self.sessions = sessions
        self.embedder = embedder
        self.sparse = sparse
        self.prefix = prefix
        self.keep_previous = keep_previous

    # ------------------------------------------------------------------ build
    def build(self, *, chunker_version: str, gloss: bool, force: bool = False) -> BuildReport:
        report = BuildReport()
        with self.sessions() as s:
            docs = list(iter_documents(s, chunker_version=chunker_version, gloss=gloss))
        for collection in COLLECTIONS:
            subset = [d for d in docs if d.collection == collection]
            alias = alias_name(self.prefix, collection)
            fp = fingerprint(subset, self.embedder.name, self.sparse.name) + ("g" if gloss else "")
            fp = hashlib.sha256(fp.encode()).hexdigest()
            if not force and self._active_fingerprint(alias) == fp and self.client.collection_exists(alias):
                report.skipped.append(alias)
                continue
            self._build_one(alias, collection, subset, fp, chunker_version)
            report.built[alias] = len(subset)
            report.retired += self._retire_old(alias)
        return report

    def _active_fingerprint(self, alias: str) -> str | None:
        with self.sessions() as s:
            return s.scalar(
                select(IndexBuild.fingerprint)
                .where(IndexBuild.alias == alias, IndexBuild.status == "active")
                .order_by(IndexBuild.activated_at.desc())
                .limit(1)
            )

    def _build_one(
        self, alias: str, collection: str, docs: list[IndexDocument], fp: str, chunker_version: str
    ) -> None:
        # Timestamp for humans, random suffix so concurrent/rapid builds never collide.
        stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        physical = (
            f"{alias}__{_slug(self.embedder.name)}__{_slug(chunker_version)}__{stamp}_{uuid.uuid4().hex[:6]}"
        )
        with session_scope(self.sessions) as s:
            build = IndexBuild(
                alias=alias,
                physical_name=physical,
                embed_model=self.embedder.name,
                sparse_model=self.sparse.name,
                chunker_version=chunker_version,
                fingerprint=fp,
                doc_count=len(docs),
                status="building",
            )
            s.add(build)
            s.flush()
            build_id = build.id
        log.info("index.build.start", alias=alias, physical=physical, docs=len(docs))
        try:
            self.client.create_collection(
                physical,
                vectors_config={
                    DENSE: models.VectorParams(size=self.embedder.dim, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={LEXICAL: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )
            for name, schema in (
                ("kind", models.PayloadSchemaType.KEYWORD),
                ("section_type", models.PayloadSchemaType.KEYWORD),
                ("document", models.PayloadSchemaType.KEYWORD),
                ("date", models.PayloadSchemaType.KEYWORD),
                ("pages", models.PayloadSchemaType.INTEGER),
            ):
                self.client.create_payload_index(physical, name, field_schema=schema)
            for i in range(0, len(docs), _UPSERT_BATCH):
                batch = docs[i : i + _UPSERT_BATCH]
                dense = self.embedder.embed_documents([d.text for d in batch])
                sparse = self.sparse.encode_documents([d.text for d in batch])
                self.client.upsert(
                    physical,
                    points=[
                        models.PointStruct(
                            id=str(uuid.UUID(d.id)),
                            vector={DENSE: dv, LEXICAL: models.SparseVector(indices=sv[0], values=sv[1])},
                            payload={**d.payload, "collection": collection, "text": d.display_text},
                        )
                        for d, dv, sv in zip(batch, dense, sparse, strict=True)
                    ],
                    wait=True,
                )
            self._swap_alias(alias, physical)
        except Exception:
            with session_scope(self.sessions) as s:
                failed = s.get(IndexBuild, build_id)
                if failed:
                    failed.status = "failed"
            if self.client.collection_exists(physical):
                self.client.delete_collection(physical)
            raise
        with session_scope(self.sessions) as s:
            for old in s.scalars(
                select(IndexBuild).where(IndexBuild.alias == alias, IndexBuild.status == "active")
            ):
                old.status = "retired"
            done = s.get(IndexBuild, build_id)
            assert done is not None
            done.status, done.activated_at = "active", utcnow()
        log.info("index.build.done", alias=alias, physical=physical)

    def _swap_alias(self, alias: str, physical: str) -> None:
        ops: list[Any] = []
        if alias in {a.alias_name for a in self.client.get_aliases().aliases}:
            ops.append(models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)))
        ops.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(collection_name=physical, alias_name=alias)
            )
        )
        self.client.update_collection_aliases(change_aliases_operations=ops)  # atomic

    def _retire_old(self, alias: str) -> list[str]:
        """Keep the active build plus `keep_previous` retired ones (for instant rollback)."""
        with session_scope(self.sessions) as s:
            retired = list(
                s.scalars(
                    select(IndexBuild)
                    .where(IndexBuild.alias == alias, IndexBuild.status == "retired")
                    .order_by(IndexBuild.created_at.desc())
                )
            )
            dropped = []
            for build in retired[self.keep_previous :]:
                if self.client.collection_exists(build.physical_name):
                    self.client.delete_collection(build.physical_name)
                build.status = "deleted"
                dropped.append(build.physical_name)
            return dropped

    # ------------------------------------------------------------------ search
    def search(
        self,
        collection: str,
        query: str,
        *,
        top_k: int = 8,
        candidates: int = 30,
        hybrid_min_telugu_ratio: float = 0.5,
        must: Sequence[models.Condition] = (),
        score_threshold: float | None = None,
    ) -> list[models.ScoredPoint]:
        alias = alias_name(self.prefix, collection)
        flt = models.Filter(must=list(must)) if must else None
        dense_q = self.embedder.embed_query(query)
        use_lexical = script_stats(query).telugu_ratio >= hybrid_min_telugu_ratio
        if not use_lexical:
            return self.client.query_points(
                alias,
                query=dense_q,
                using=DENSE,
                limit=top_k,
                query_filter=flt,
                score_threshold=score_threshold,
                with_payload=True,
            ).points
        idx, vals = self.sparse.encode_query(query)
        return self.client.query_points(
            alias,
            prefetch=[
                models.Prefetch(
                    query=dense_q, using=DENSE, limit=candidates, filter=flt, score_threshold=score_threshold
                ),
                models.Prefetch(
                    query=models.SparseVector(indices=idx, values=vals),
                    using=LEXICAL,
                    limit=candidates,
                    filter=flt,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        ).points


def describe(index: VectorIndex) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for collection in COLLECTIONS:
        alias = alias_name(index.prefix, collection)
        if index.client.collection_exists(alias):
            info = index.client.get_collection(alias)
            out[alias] = {"points": info.points_count, "status": str(info.status)}
    result: dict[str, Any] = json.loads(json.dumps(out, default=str))
    return result

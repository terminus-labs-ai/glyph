from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import asyncpg

from glyph.domain.models import Chunk, ChunkType, DocType, Document, Source

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    source_type TEXT NOT NULL,
    origin TEXT NOT NULL,
    config JSONB DEFAULT '{{}}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (name, version)
);

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    title TEXT NOT NULL,
    doc_type TEXT NOT NULL,
    raw_content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (source_id, path)
);

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    source_name TEXT NOT NULL,
    source_version TEXT NOT NULL,
    chunk_type TEXT NOT NULL,
    qualified_name TEXT NOT NULL,
    parent_name TEXT NOT NULL,
    heading TEXT NOT NULL,
    summary TEXT DEFAULT '',
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{{}}',
    embedding VECTOR({dimensions}),
    token_count INTEGER DEFAULT 0,
    chunk_index INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    fts tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(qualified_name, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(heading, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(summary, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(content, '')), 'C')
    ) STORED
);

CREATE INDEX IF NOT EXISTS idx_chunks_parent_type ON chunks (parent_name, chunk_type);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks (source_name, source_version);
CREATE INDEX IF NOT EXISTS idx_chunks_qualified ON chunks (qualified_name);
"""

VECTOR_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = {lists});
"""


class PostgresStore:
    def __init__(self, dsn: str, dimensions: int = 512):
        self._dsn = dsn
        self._dimensions = dimensions
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)
        logger.info("Connected to PostgreSQL")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    async def init_schema(self) -> None:
        sql = SCHEMA_SQL.format(dimensions=self._dimensions)
        async with self._pool.acquire() as conn:
            await conn.execute(sql)
        logger.info("Schema initialized")

    async def create_vector_index(self, lists: int = 100) -> None:
        sql = VECTOR_INDEX_SQL.format(lists=lists)
        async with self._pool.acquire() as conn:
            await conn.execute(sql)
        logger.info("Vector index created")

    async def upgrade_schema(self) -> None:
        """Add FTS column and index. Idempotent -- safe to run multiple times."""
        async with self._pool.acquire() as conn:
            await conn.execute("""
                ALTER TABLE chunks ADD COLUMN IF NOT EXISTS fts tsvector
                    GENERATED ALWAYS AS (
                        setweight(to_tsvector('english', coalesce(qualified_name, '')), 'A') ||
                        setweight(to_tsvector('english', coalesce(heading, '')), 'A') ||
                        setweight(to_tsvector('english', coalesce(summary, '')), 'B') ||
                        setweight(to_tsvector('english', coalesce(content, '')), 'C')
                    ) STORED
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_fts ON chunks USING GIN (fts)
            """)
        logger.info("Schema upgraded: FTS column and GIN index applied")

    async def upsert_source(self, source: Source) -> uuid.UUID:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO sources (id, name, version, source_type, origin, config)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (name, version) DO UPDATE SET
                    source_type = EXCLUDED.source_type,
                    origin = EXCLUDED.origin,
                    config = EXCLUDED.config,
                    updated_at = NOW()
                RETURNING id
                """,
                source.id, source.name, source.version,
                source.source_type, source.origin,
                json.dumps(source.config),
            )
            return row["id"]

    async def upsert_document(self, doc: Document) -> tuple[uuid.UUID, bool]:
        """Returns (id, changed). changed=True if content was new or updated."""
        async with self._pool.acquire() as conn:
            existing = await conn.fetchrow(
                "SELECT id, content_hash FROM documents WHERE source_id = $1 AND path = $2",
                doc.source_id, doc.path,
            )
            if existing and existing["content_hash"] == doc.content_hash:
                return existing["id"], False

            row = await conn.fetchrow(
                """
                INSERT INTO documents (id, source_id, path, title, doc_type, raw_content, content_hash)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (source_id, path) DO UPDATE SET
                    title = EXCLUDED.title,
                    doc_type = EXCLUDED.doc_type,
                    raw_content = EXCLUDED.raw_content,
                    content_hash = EXCLUDED.content_hash,
                    updated_at = NOW()
                RETURNING id
                """,
                doc.id, doc.source_id, doc.path, doc.title,
                doc.doc_type.value, doc.raw_content, doc.content_hash,
            )
            return row["id"], True

    async def delete_chunks_for_document(self, document_id: uuid.UUID) -> int:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM chunks WHERE document_id = $1", document_id,
            )
            count = int(result.split()[-1])
            return count

    async def insert_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO chunks (
                    id, document_id, source_name, source_version,
                    chunk_type, qualified_name, parent_name, heading,
                    summary, content, metadata, embedding,
                    token_count, chunk_index
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
                )
                """,
                [
                    (
                        c.id, c.document_id, c.source_name, c.source_version,
                        c.chunk_type.value, c.qualified_name, c.parent_name, c.heading,
                        c.summary, c.content, json.dumps(c.metadata),
                        _embedding_str(c.embedding, self._dimensions),
                        c.token_count, c.chunk_index,
                    )
                    for c in chunks
                ],
            )
            return len(chunks)

    async def search(
        self,
        embedding: list[float],
        *,
        source_name: str | None = None,
        source_version: str | None = None,
        chunk_types: list[ChunkType] | None = None,
        parent_name: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        conditions = []
        params: list[Any] = [str(embedding)]
        idx = 2

        if source_name:
            conditions.append(f"source_name = ${idx}")
            params.append(source_name)
            idx += 1
        if source_version:
            conditions.append(f"source_version = ${idx}")
            params.append(source_version)
            idx += 1
        if chunk_types:
            conditions.append(f"chunk_type = ANY(${idx}::text[])")
            params.append([ct.value for ct in chunk_types])
            idx += 1
        if parent_name:
            conditions.append(f"parent_name = ${idx}")
            params.append(parent_name)
            idx += 1

        where = (" AND " + " AND ".join(conditions)) if conditions else ""

        params.append(limit)
        sql = f"""
            SELECT id, qualified_name, heading, summary, content, metadata,
                   chunk_type, parent_name, source_name, source_version,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM chunks
            WHERE embedding IS NOT NULL{where}
            ORDER BY embedding <=> $1::vector
            LIMIT ${idx}
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
            return [dict(r) for r in rows]

    async def text_search(
        self,
        query: str,
        *,
        source_name: str | None = None,
        source_version: str | None = None,
        chunk_types: list[str] | None = None,
        parent_name: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        conditions = []
        params: list[Any] = [query]
        idx = 2

        if source_name:
            conditions.append(f"source_name = ${idx}")
            params.append(source_name)
            idx += 1
        if source_version:
            conditions.append(f"source_version = ${idx}")
            params.append(source_version)
            idx += 1
        if chunk_types:
            conditions.append(f"chunk_type = ANY(${idx}::text[])")
            params.append(chunk_types)
            idx += 1
        if parent_name:
            conditions.append(f"parent_name = ${idx}")
            params.append(parent_name)
            idx += 1

        where = (" AND " + " AND ".join(conditions)) if conditions else ""

        params.append(limit)
        sql = f"""
            SELECT id, qualified_name, heading, summary, content, metadata,
                   chunk_type, parent_name, source_name, source_version,
                   ts_rank_cd(fts, websearch_to_tsquery('english', $1)) AS rank
            FROM chunks
            WHERE fts @@ websearch_to_tsquery('english', $1){where}
            ORDER BY rank DESC
            LIMIT ${idx}
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
            return [dict(r) for r in rows]

    async def hybrid_search(
        self,
        query: str,
        embedding: list[float] | None,
        *,
        source_name: str | None = None,
        source_version: str | None = None,
        chunk_types: list[str] | None = None,
        parent_name: str | None = None,
        limit: int = 10,
        rrf_k: int = 60,
        fetch_multiplier: int = 3,
    ) -> list[dict[str, Any]]:
        fetch_limit = limit * fetch_multiplier

        # Always run FTS
        fts_results = await self.text_search(
            query,
            source_name=source_name,
            source_version=source_version,
            chunk_types=chunk_types,
            parent_name=parent_name,
            limit=fetch_limit,
        )

        # Conditionally run vector search
        vec_results: list[dict[str, Any]] = []
        if embedding is not None:
            chunk_type_enums = (
                [ChunkType(ct) for ct in chunk_types] if chunk_types else None
            )
            vec_results = await self.search(
                embedding,
                source_name=source_name,
                source_version=source_version,
                chunk_types=chunk_type_enums,
                parent_name=parent_name,
                limit=fetch_limit,
            )

        # RRF fusion
        fusion: dict[str, dict[str, Any]] = {}

        for rank, doc in enumerate(fts_results):
            key = str(doc["id"])
            fusion[key] = {
                "doc": doc,
                "score": 1.0 / (rrf_k + rank + 1),
                "in_fts": True,
                "in_vec": False,
            }

        for rank, doc in enumerate(vec_results):
            key = str(doc["id"])
            if key in fusion:
                fusion[key]["score"] += 1.0 / (rrf_k + rank + 1)
                fusion[key]["in_vec"] = True
            else:
                fusion[key] = {
                    "doc": doc,
                    "score": 1.0 / (rrf_k + rank + 1),
                    "in_fts": False,
                    "in_vec": True,
                }

        # Build output
        for entry in fusion.values():
            if entry["in_fts"] and entry["in_vec"]:
                entry["doc"]["retrieval"] = "hybrid"
            elif entry["in_fts"]:
                entry["doc"]["retrieval"] = "keyword"
            else:
                entry["doc"]["retrieval"] = "semantic"
            entry["doc"]["score"] = entry["score"]

        ranked = sorted(fusion.values(), key=lambda e: e["score"], reverse=True)
        return [entry["doc"] for entry in ranked[:limit]]

    async def get_by_qualified_name(self, qualified_name: str) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, document_id, qualified_name, parent_name, heading,
                       summary, content, metadata, chunk_type, chunk_index,
                       source_name, source_version, token_count
                FROM chunks
                WHERE qualified_name = $1
                """,
                qualified_name,
            )
            return dict(row) if row else None

    async def get_by_parent(
        self,
        parent_name: str,
        source_name: str | None = None,
        source_version: str | None = None,
    ) -> list[dict[str, Any]]:
        conditions = ["parent_name = $1"]
        params: list[Any] = [parent_name]
        idx = 2

        if source_name:
            conditions.append(f"source_name = ${idx}")
            params.append(source_name)
            idx += 1
        if source_version:
            conditions.append(f"source_version = ${idx}")
            params.append(source_version)
            idx += 1

        where = " AND ".join(conditions)
        sql = f"""
            SELECT id, document_id, qualified_name, parent_name, heading,
                   summary, content, metadata, chunk_type, chunk_index,
                   source_name, source_version, token_count
            FROM chunks
            WHERE {where}
            ORDER BY chunk_type, chunk_index
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
            return [dict(r) for r in rows]

    async def get_sources_with_counts(self) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT s.name, s.version, s.source_type, s.origin,
                       COUNT(DISTINCT d.id) AS document_count,
                       COUNT(DISTINCT c.id) AS chunk_count
                FROM sources s
                LEFT JOIN documents d ON d.source_id = s.id
                LEFT JOIN chunks c ON c.source_name = s.name
                    AND c.source_version = s.version
                GROUP BY s.id, s.name, s.version, s.source_type, s.origin
                ORDER BY s.name, s.version
                """
            )
            return [dict(r) for r in rows]

    async def get_all_chunks(
        self,
        source_name: str,
        source_version: str,
    ) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, qualified_name, parent_name, heading, summary,
                       content, metadata, chunk_type, chunk_index
                FROM chunks
                WHERE source_name = $1 AND source_version = $2
                ORDER BY parent_name, chunk_index
                """,
                source_name, source_version,
            )
            return [dict(r) for r in rows]

    async def get_stats(self) -> dict[str, Any]:
        async with self._pool.acquire() as conn:
            sources = await conn.fetchval("SELECT COUNT(*) FROM sources")
            docs = await conn.fetchval("SELECT COUNT(*) FROM documents")
            chunks = await conn.fetchval("SELECT COUNT(*) FROM chunks")
            by_type = await conn.fetch(
                "SELECT chunk_type, COUNT(*) as cnt FROM chunks GROUP BY chunk_type ORDER BY cnt DESC"
            )
            return {
                "sources": sources,
                "documents": docs,
                "chunks": chunks,
                "by_type": {r["chunk_type"]: r["cnt"] for r in by_type},
            }


def _embedding_str(embedding: list[float] | None, dimensions: int) -> str | None:
    if embedding is None:
        return None
    vec = embedding[:dimensions]
    if len(vec) < dimensions:
        vec.extend([0.0] * (dimensions - len(vec)))
    return "[" + ",".join(str(v) for v in vec) + "]"

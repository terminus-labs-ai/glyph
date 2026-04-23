from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from mcp.server.fastmcp import FastMCP

from glyph.server import GlyphServer, _format_search_results


# --- Helpers ---

def _make_chunk(
    qualified_name: str = "Node2D.get_position",
    parent_name: str = "Node2D",
    chunk_type: str = "method",
    heading: str = "get_position",
    summary: str = "Returns the node's position.",
    content: str = "func get_position() -> Vector2",
    source_name: str = "godot",
    source_version: str = "4.6.1",
    metadata: dict | None = None,
    chunk_id: uuid.UUID | None = None,
    **kwargs,
) -> dict:
    result = {
        "id": chunk_id or uuid.uuid4(),
        "document_id": uuid.uuid4(),
        "qualified_name": qualified_name,
        "parent_name": parent_name,
        "chunk_type": chunk_type,
        "heading": heading,
        "summary": summary,
        "content": content,
        "source_name": source_name,
        "source_version": source_version,
        "metadata": metadata or {},
        "chunk_index": 0,
        "token_count": 10,
    }
    result.update(kwargs)
    return result


def _build_test_server(mock_store, mock_embedder, mock_reranker=None) -> GlyphServer:
    with patch("glyph.server.load_config"):
        srv = GlyphServer.__new__(GlyphServer)
        srv._config_path = "test.yaml"
        srv._store = mock_store
        srv._embedder = mock_embedder
        srv._reranker = mock_reranker
        srv.mcp = FastMCP("test")
        srv._register_tools()
        srv._register_resources()
    return srv


@pytest.fixture
def mock_store():
    store = AsyncMock()
    store.connect = AsyncMock()
    store.close = AsyncMock()
    return store


@pytest.fixture
def mock_embedder():
    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=[[0.1] * 512])
    return embedder


@pytest.fixture
def mock_pool():
    from unittest.mock import MagicMock

    pool = MagicMock()
    conn = AsyncMock()
    cm = AsyncMock()
    cm.__aenter__.return_value = conn
    pool.acquire.return_value = cm
    return pool, conn


# --- Group 1: text_search() store method ---

class TestTextSearch:
    async def test_text_search_returns_results(self, mock_pool):
        from glyph.store.postgres import PostgresStore
        pool, conn = mock_pool
        row1 = _make_chunk(qualified_name="Node2D.get_position", rank=0.12)
        row2 = _make_chunk(qualified_name="Node3D.get_position", rank=0.08)
        conn.fetch = AsyncMock(return_value=[row1, row2])

        store = PostgresStore.__new__(PostgresStore)
        store._pool = pool

        results = await store.text_search("get position")
        assert len(results) == 2
        assert results[0]["qualified_name"] == "Node2D.get_position"
        assert "rank" in results[0]

    async def test_text_search_with_filters(self, mock_pool):
        from glyph.store.postgres import PostgresStore
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])

        store = PostgresStore.__new__(PostgresStore)
        store._pool = pool

        await store.text_search(
            "position",
            source_name="godot",
            source_version="4.6.1",
            parent_name="Node2D",
            chunk_types=["method"],
        )
        conn.fetch.assert_called_once()
        sql = conn.fetch.call_args[0][0]
        assert "source_name" in sql
        assert "source_version" in sql
        assert "parent_name" in sql
        assert "chunk_type = ANY" in sql

    async def test_text_search_no_matches(self, mock_pool):
        from glyph.store.postgres import PostgresStore
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])

        store = PostgresStore.__new__(PostgresStore)
        store._pool = pool

        results = await store.text_search("nonexistent_query_xyz")
        assert results == []

    async def test_text_search_passes_limit(self, mock_pool):
        from glyph.store.postgres import PostgresStore
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])

        store = PostgresStore.__new__(PostgresStore)
        store._pool = pool

        await store.text_search("test", limit=25)
        args = conn.fetch.call_args[0]
        # limit should be the last positional param
        assert args[-1] == 25


# --- Group 2: RRF scoring via hybrid_search() ---

class TestRRFScoring:
    """Test RRF fusion logic through hybrid_search with mocked sub-methods."""

    async def test_chunk_in_both_sets_ranks_highest(self):
        from glyph.store.postgres import PostgresStore
        store = PostgresStore.__new__(PostgresStore)

        shared_id = uuid.uuid4()
        only_fts_id = uuid.uuid4()
        only_vec_id = uuid.uuid4()

        store.text_search = AsyncMock(return_value=[
            _make_chunk(qualified_name="shared", chunk_id=shared_id, rank=0.12),
            _make_chunk(qualified_name="fts_only", chunk_id=only_fts_id, rank=0.08),
        ])
        store.search = AsyncMock(return_value=[
            _make_chunk(qualified_name="shared", chunk_id=shared_id, similarity=0.95),
            _make_chunk(qualified_name="vec_only", chunk_id=only_vec_id, similarity=0.88),
        ])

        results = await store.hybrid_search("test", [0.1] * 512, limit=10)

        # Shared chunk should rank highest (summed RRF scores)
        assert results[0]["qualified_name"] == "shared"
        assert results[0]["retrieval"] == "hybrid"
        assert results[0]["score"] > results[1]["score"]

    async def test_disjoint_sets(self):
        from glyph.store.postgres import PostgresStore
        store = PostgresStore.__new__(PostgresStore)

        id_a, id_b = uuid.uuid4(), uuid.uuid4()
        id_c, id_d = uuid.uuid4(), uuid.uuid4()

        store.text_search = AsyncMock(return_value=[
            _make_chunk(qualified_name="A", chunk_id=id_a, rank=0.12),
            _make_chunk(qualified_name="B", chunk_id=id_b, rank=0.08),
        ])
        store.search = AsyncMock(return_value=[
            _make_chunk(qualified_name="C", chunk_id=id_c, similarity=0.95),
            _make_chunk(qualified_name="D", chunk_id=id_d, similarity=0.88),
        ])

        results = await store.hybrid_search("test", [0.1] * 512, limit=10)
        names = {r["qualified_name"] for r in results}
        assert names == {"A", "B", "C", "D"}

    async def test_embedding_none_skips_vector(self):
        from glyph.store.postgres import PostgresStore
        store = PostgresStore.__new__(PostgresStore)

        store.text_search = AsyncMock(return_value=[
            _make_chunk(qualified_name="fts_result", rank=0.12),
        ])
        store.search = AsyncMock()

        results = await store.hybrid_search("test", None, limit=10)
        store.search.assert_not_called()
        assert len(results) == 1
        assert results[0]["retrieval"] == "keyword"

    async def test_both_empty(self):
        from glyph.store.postgres import PostgresStore
        store = PostgresStore.__new__(PostgresStore)

        store.text_search = AsyncMock(return_value=[])
        store.search = AsyncMock(return_value=[])

        results = await store.hybrid_search("test", [0.1] * 512, limit=10)
        assert results == []


# --- Group 3: hybrid_search() method ---

class TestHybridSearch:
    async def test_passes_filters_through(self):
        from glyph.store.postgres import PostgresStore
        store = PostgresStore.__new__(PostgresStore)

        store.text_search = AsyncMock(return_value=[])
        store.search = AsyncMock(return_value=[])

        await store.hybrid_search(
            "test", [0.1] * 512,
            source_name="godot",
            source_version="4.6.1",
            chunk_types=["method"],
            parent_name="Node2D",
            limit=5,
        )

        # text_search gets raw string chunk_types
        ts_kwargs = store.text_search.call_args.kwargs
        assert ts_kwargs["source_name"] == "godot"
        assert ts_kwargs["source_version"] == "4.6.1"
        assert ts_kwargs["chunk_types"] == ["method"]
        assert ts_kwargs["parent_name"] == "Node2D"

        # search gets ChunkType enum values
        s_kwargs = store.search.call_args.kwargs
        assert s_kwargs["source_name"] == "godot"

    async def test_fetch_multiplier(self):
        from glyph.store.postgres import PostgresStore
        store = PostgresStore.__new__(PostgresStore)

        store.text_search = AsyncMock(return_value=[])
        store.search = AsyncMock(return_value=[])

        await store.hybrid_search("test", [0.1] * 512, limit=5, fetch_multiplier=3)

        ts_kwargs = store.text_search.call_args.kwargs
        assert ts_kwargs["limit"] == 15  # 5 * 3

    async def test_limit_respected(self):
        from glyph.store.postgres import PostgresStore
        store = PostgresStore.__new__(PostgresStore)

        # Return more results than limit
        fts_results = [
            _make_chunk(qualified_name=f"fts_{i}", chunk_id=uuid.uuid4(), rank=0.1)
            for i in range(10)
        ]
        store.text_search = AsyncMock(return_value=fts_results)
        store.search = AsyncMock(return_value=[])

        results = await store.hybrid_search("test", None, limit=3)
        assert len(results) == 3

    async def test_result_has_retrieval_field(self):
        from glyph.store.postgres import PostgresStore
        store = PostgresStore.__new__(PostgresStore)

        store.text_search = AsyncMock(return_value=[
            _make_chunk(qualified_name="kw_only", chunk_id=uuid.uuid4(), rank=0.1),
        ])
        store.search = AsyncMock(return_value=[
            _make_chunk(qualified_name="sem_only", chunk_id=uuid.uuid4(), similarity=0.9),
        ])

        results = await store.hybrid_search("test", [0.1] * 512, limit=10)
        for r in results:
            assert "retrieval" in r
            assert r["retrieval"] in ("hybrid", "keyword", "semantic")
            assert "score" in r


# --- Group 4: upgrade_schema() ---

class TestUpgradeSchema:
    async def test_executes_migration_sql(self, mock_pool):
        from glyph.store.postgres import PostgresStore
        pool, conn = mock_pool

        store = PostgresStore.__new__(PostgresStore)
        store._pool = pool

        await store.upgrade_schema()
        assert conn.execute.call_count == 2
        calls = [str(c) for c in conn.execute.call_args_list]
        sql_combined = " ".join(calls)
        assert "fts tsvector" in sql_combined
        assert "idx_chunks_fts" in sql_combined

    async def test_idempotent(self, mock_pool):
        from glyph.store.postgres import PostgresStore
        pool, conn = mock_pool

        store = PostgresStore.__new__(PostgresStore)
        store._pool = pool

        await store.upgrade_schema()
        first_count = conn.execute.call_count

        await store.upgrade_schema()
        assert conn.execute.call_count == first_count * 2  # Same calls both times


# --- Group 5: MCP search tool integration ---

class TestSearchToolHybrid:
    async def test_uses_hybrid_search(self, mock_store, mock_embedder):
        mock_store.hybrid_search = AsyncMock(return_value=[
            _make_chunk(score=0.045, retrieval="hybrid"),
        ])
        srv = _build_test_server(mock_store, mock_embedder)

        result = await srv.mcp._tool_manager.call_tool("search", {"query": "position"})
        mock_store.hybrid_search.assert_called_once()
        assert "Node2D.get_position" in result

    async def test_embedder_failure_falls_back(self, mock_store, mock_embedder):
        mock_embedder.embed = AsyncMock(side_effect=RuntimeError("connection refused"))
        mock_store.hybrid_search = AsyncMock(return_value=[
            _make_chunk(score=0.016, retrieval="keyword"),
        ])
        srv = _build_test_server(mock_store, mock_embedder)

        result = await srv.mcp._tool_manager.call_tool("search", {"query": "position"})
        # Should NOT return an error -- should return results from FTS fallback
        assert "Node2D.get_position" in result
        assert "Error" not in result
        # embedding should be None
        call_args = mock_store.hybrid_search.call_args
        assert call_args[0][1] is None  # second positional arg is embedding

    async def test_passes_filters(self, mock_store, mock_embedder):
        mock_store.hybrid_search = AsyncMock(return_value=[
            _make_chunk(score=0.045, retrieval="hybrid"),
        ])
        srv = _build_test_server(mock_store, mock_embedder)

        await srv.mcp._tool_manager.call_tool("search", {
            "query": "position",
            "source": "godot",
            "version": "4.6.1",
            "chunk_types": ["method"],
            "parent": "Node2D",
            "limit": 5,
        })

        kwargs = mock_store.hybrid_search.call_args.kwargs
        assert kwargs["source_name"] == "godot"
        assert kwargs["source_version"] == "4.6.1"
        assert kwargs["chunk_types"] == ["method"]
        assert kwargs["parent_name"] == "Node2D"
        assert kwargs["limit"] == 5

    async def test_no_results(self, mock_store, mock_embedder):
        mock_store.hybrid_search = AsyncMock(return_value=[])
        srv = _build_test_server(mock_store, mock_embedder)

        result = await srv.mcp._tool_manager.call_tool("search", {"query": "nonexistent"})
        assert "No results found" in result


# --- Group 6: Retrieval tags in formatting ---

class TestRetrievalTags:
    def test_hybrid_tag(self):
        results = [_make_chunk(score=0.045, retrieval="hybrid")]
        output = _format_search_results(results)
        assert "[hybrid]" in output
        assert "0.045" in output

    def test_keyword_tag(self):
        results = [_make_chunk(score=0.016, retrieval="keyword")]
        output = _format_search_results(results)
        assert "[keyword]" in output

    def test_semantic_tag(self):
        results = [_make_chunk(score=0.016, retrieval="semantic")]
        output = _format_search_results(results)
        assert "[semantic]" in output

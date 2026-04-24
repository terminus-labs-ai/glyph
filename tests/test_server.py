from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from mcp.server.fastmcp import FastMCP

from glyph.rerankers import LlamaReranker
from glyph.server import (
    GlyphServer,
    _format_chunk_detail,
    _format_context,
    _format_search_results,
    _format_sources,
)


# --- Fixtures ---

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
    similarity: float | None = None,
    **kwargs,
) -> dict:
    result = {
        "id": uuid.uuid4(),
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
    if similarity is not None:
        result["similarity"] = similarity
    result.update(kwargs)
    return result


def _make_source(
    name: str = "godot",
    version: str = "4.6.1",
    source_type: str = "godot_xml",
    origin: str = "/path/to/godot",
    document_count: int = 150,
    chunk_count: int = 3200,
) -> dict:
    return {
        "name": name,
        "version": version,
        "source_type": source_type,
        "origin": origin,
        "document_count": document_count,
        "chunk_count": chunk_count,
    }


def _build_test_server(mock_store, mock_embedder, mock_reranker=None) -> GlyphServer:
    """Create a GlyphServer with mocked dependencies, bypassing lifespan."""
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
def mock_reranker():
    reranker = AsyncMock()
    reranker.rerank = AsyncMock(
        return_value=[0.9, 0.7, 0.5, 0.3, 0.1],
    )
    return reranker


# --- Formatter tests ---


class TestFormatSearchResults:
    def test_basic_result(self):
        results = [_make_chunk(score=0.045, retrieval="hybrid")]
        output = _format_search_results(results)
        assert "### Node2D.get_position" in output
        assert "[hybrid]" in output
        assert "0.045" in output
        assert "godot 4.6.1" in output
        assert "method" in output

    def test_multiple_results(self):
        results = [
            _make_chunk(qualified_name="Node2D.get_position", score=0.045, retrieval="hybrid"),
            _make_chunk(
                qualified_name="Node2D.set_position",
                heading="set_position",
                score=0.032, retrieval="keyword",
            ),
        ]
        output = _format_search_results(results)
        assert "Node2D.get_position" in output
        assert "Node2D.set_position" in output
        assert output.count("---") == 2


class TestFormatChunkDetail:
    def test_basic_detail(self):
        chunk = _make_chunk()
        output = _format_chunk_detail(chunk)
        assert "# Node2D.get_position" in output
        assert "**Type:** method" in output
        assert "**Parent:** Node2D" in output
        assert "func get_position() -> Vector2" in output

    def test_with_inherits_metadata(self):
        chunk = _make_chunk(
            chunk_type="class_overview",
            metadata={"inherits": "CanvasItem"},
        )
        output = _format_chunk_detail(chunk)
        assert "**Inherits:** CanvasItem" in output


class TestFormatContext:
    def test_groups_by_type(self):
        chunks = [
            _make_chunk(
                qualified_name="Node2D",
                chunk_type="class_overview",
                heading="Node2D",
                summary="2D game object.",
                content="Node2D is the base for all 2D nodes.",
            ),
            _make_chunk(
                qualified_name="Node2D.position",
                chunk_type="property",
                heading="position",
                summary="The node position.",
                content="var position: Vector2",
            ),
            _make_chunk(
                qualified_name="Node2D.get_position",
                chunk_type="method",
                heading="get_position",
                summary="Returns position.",
                content="func get_position() -> Vector2",
            ),
        ]
        output = _format_context("Node2D", chunks)
        assert "# Node2D" in output
        assert "## Properties" in output
        assert "## Methods" in output
        assert "### position" in output
        assert "### get_position" in output

    def test_no_overview(self):
        chunks = [_make_chunk(chunk_type="method")]
        output = _format_context("Node2D", chunks)
        assert "# Node2D" in output
        assert "## Methods" in output


class TestFormatSources:
    def test_table_format(self):
        sources = [_make_source(), _make_source(name="stripe", version="2024-01")]
        output = _format_sources(sources)
        assert "| godot |" in output
        assert "| stripe |" in output
        assert "| 150 |" in output


# --- Tool tests (via FastMCP tool_manager) ---


class TestSearchTool:
    async def test_search_returns_formatted_results(self, mock_store, mock_embedder):
        mock_store.hybrid_search = AsyncMock(return_value=[
            _make_chunk(score=0.045, retrieval="hybrid"),
            _make_chunk(qualified_name="Node2D.set_position", heading="set_position", score=0.032, retrieval="keyword"),
        ])
        srv = _build_test_server(mock_store, mock_embedder)

        result = await srv.mcp._tool_manager.call_tool("search", {"query": "position"})
        assert "Node2D.get_position" in result
        assert "Node2D.set_position" in result
        assert "0.045" in result

    async def test_search_no_results(self, mock_store, mock_embedder):
        mock_store.hybrid_search = AsyncMock(return_value=[])
        srv = _build_test_server(mock_store, mock_embedder)

        result = await srv.mcp._tool_manager.call_tool("search", {"query": "nonexistent"})
        assert "No results found" in result

    async def test_search_passes_filters_to_store(self, mock_store, mock_embedder):
        mock_store.hybrid_search = AsyncMock(return_value=[_make_chunk(score=0.045, retrieval="hybrid")])
        srv = _build_test_server(mock_store, mock_embedder)

        await srv.mcp._tool_manager.call_tool("search", {
            "query": "position",
            "source": "godot",
            "version": "4.6.1",
            "chunk_types": ["method"],
            "parent": "Node2D",
            "limit": 5,
        })

        mock_store.hybrid_search.assert_called_once()
        kwargs = mock_store.hybrid_search.call_args.kwargs
        assert kwargs["source_name"] == "godot"
        assert kwargs["source_version"] == "4.6.1"
        assert kwargs["parent_name"] == "Node2D"
        assert kwargs["limit"] == 5

    async def test_search_clamps_limit(self, mock_store, mock_embedder):
        mock_store.hybrid_search = AsyncMock(return_value=[])
        srv = _build_test_server(mock_store, mock_embedder)

        await srv.mcp._tool_manager.call_tool("search", {"query": "test", "limit": 999})
        kwargs = mock_store.hybrid_search.call_args.kwargs
        assert kwargs["limit"] == 50

    async def test_search_embedder_failure(self, mock_store, mock_embedder):
        mock_embedder.embed = AsyncMock(side_effect=RuntimeError("connection refused"))
        mock_store.hybrid_search = AsyncMock(return_value=[
            _make_chunk(score=0.016, retrieval="keyword"),
        ])
        srv = _build_test_server(mock_store, mock_embedder)

        result = await srv.mcp._tool_manager.call_tool("search", {"query": "test"})
        # Embedding failure falls back to keyword-only search, not an error
        assert "Node2D.get_position" in result
        assert "Error" not in result
        # embedding arg should be None
        call_args = mock_store.hybrid_search.call_args
        assert call_args[0][1] is None


class TestLookupTool:
    async def test_lookup_found(self, mock_store, mock_embedder):
        mock_store.get_by_qualified_name = AsyncMock(return_value=_make_chunk())
        srv = _build_test_server(mock_store, mock_embedder)

        result = await srv.mcp._tool_manager.call_tool("lookup", {"qualified_name": "Node2D.get_position"})
        assert "Node2D.get_position" in result
        assert "method" in result

    async def test_lookup_not_found(self, mock_store, mock_embedder):
        mock_store.get_by_qualified_name = AsyncMock(return_value=None)
        srv = _build_test_server(mock_store, mock_embedder)

        result = await srv.mcp._tool_manager.call_tool("lookup", {"qualified_name": "Nope.nope"})
        assert "No chunk found" in result


class TestGetContextTool:
    async def test_groups_by_type(self, mock_store, mock_embedder):
        mock_store.get_by_parent = AsyncMock(return_value=[
            _make_chunk(chunk_type="class_overview", qualified_name="Node2D", heading="Node2D"),
            _make_chunk(chunk_type="property", qualified_name="Node2D.position", heading="position"),
            _make_chunk(chunk_type="method", qualified_name="Node2D.get_position", heading="get_position"),
        ])
        srv = _build_test_server(mock_store, mock_embedder)

        result = await srv.mcp._tool_manager.call_tool("get_context", {"parent_name": "Node2D"})
        assert "# Node2D" in result
        assert "## Properties" in result
        assert "## Methods" in result

    async def test_not_found(self, mock_store, mock_embedder):
        mock_store.get_by_parent = AsyncMock(return_value=[])
        srv = _build_test_server(mock_store, mock_embedder)

        result = await srv.mcp._tool_manager.call_tool("get_context", {"parent_name": "Nonexistent"})
        assert "No chunks found" in result

    async def test_passes_source_filter(self, mock_store, mock_embedder):
        mock_store.get_by_parent = AsyncMock(return_value=[_make_chunk(chunk_type="method")])
        srv = _build_test_server(mock_store, mock_embedder)

        await srv.mcp._tool_manager.call_tool("get_context", {
            "parent_name": "Node2D",
            "source": "godot",
            "version": "4.6.1",
        })
        mock_store.get_by_parent.assert_called_once_with(
            "Node2D", source_name="godot", source_version="4.6.1",
        )


class TestListSourcesTool:
    async def test_list_sources(self, mock_store, mock_embedder):
        mock_store.get_sources_with_counts = AsyncMock(return_value=[
            _make_source(),
            _make_source(name="stripe", version="2024-01"),
        ])
        srv = _build_test_server(mock_store, mock_embedder)

        result = await srv.mcp._tool_manager.call_tool("list_sources", {})
        assert "godot" in result
        assert "stripe" in result

    async def test_list_sources_empty(self, mock_store, mock_embedder):
        mock_store.get_sources_with_counts = AsyncMock(return_value=[])
        srv = _build_test_server(mock_store, mock_embedder)

        result = await srv.mcp._tool_manager.call_tool("list_sources", {})
        assert "No sources indexed" in result


# --- Reranker tool tests ---


class TestSearchRerankTool:
    """Tests for two-stage retrieval with reranker."""

    async def test_rerank_path_takes_vector_then_rerank(
        self, mock_store, mock_embedder, mock_reranker
    ):
        mock_store.search = AsyncMock(return_value=[
            _make_chunk(qualified_name="Node2D.get_position", heading="get_position",
                        content="Returns position", similarity=0.3),
            _make_chunk(qualified_name="Node2D.set_position", heading="set_position",
                        content="Sets position", similarity=0.6),
            _make_chunk(qualified_name="Node2D.position", heading="position",
                        content="The position property", similarity=0.9),
        ])
        mock_reranker.rerank = AsyncMock(
            return_value=[0.1, 0.95, 0.5],  # reorders: set_position first
        )
        srv = _build_test_server(mock_store, mock_embedder, mock_reranker)

        result = await srv.mcp._tool_manager.call_tool("search", {
            "query": "position", "rerank": True, "limit": 5, "candidates": 10,
        })

        # Should call store.search (vector) not hybrid_search
        mock_store.search.assert_called_once()
        assert mock_reranker.rerank.called
        # Re-ordered by rerank_score: set_position (0.95) should be first
        assert "Node2D.set_position" in result
        assert "[reranked]" in result

    async def test_rerank_false_uses_hybrid(
        self, mock_store, mock_embedder, mock_reranker
    ):
        mock_store.hybrid_search = AsyncMock(return_value=[
            _make_chunk(score=0.045, retrieval="hybrid"),
        ])
        srv = _build_test_server(mock_store, mock_embedder, mock_reranker)

        result = await srv.mcp._tool_manager.call_tool("search", {
            "query": "test", "rerank": False, "limit": 5,
        })

        mock_store.hybrid_search.assert_called_once()
        assert "[hybrid]" in result
        assert "[reranked]" not in result

    async def test_no_reranker_ignores_rerank_param(
        self, mock_store, mock_embedder
    ):
        mock_store.hybrid_search = AsyncMock(return_value=[
            _make_chunk(score=0.045, retrieval="hybrid"),
        ])
        srv = _build_test_server(mock_store, mock_embedder, mock_reranker=None)

        # rerank=True but no reranker configured → falls back to hybrid
        result = await srv.mcp._tool_manager.call_tool("search", {
            "query": "test", "rerank": True, "limit": 5,
        })

        mock_store.hybrid_search.assert_called_once()
        mock_store.search.assert_not_called()

    async def test_no_reranker_surfaces_fallback_note_in_response(
        self, mock_store, mock_embedder
    ):
        """When rerank=True but no reranker configured, the response text should
        include a note informing the MCP client about the fallback."""
        mock_store.hybrid_search = AsyncMock(return_value=[
            _make_chunk(score=0.045, retrieval="hybrid"),
        ])
        srv = _build_test_server(mock_store, mock_embedder, mock_reranker=None)

        result = await srv.mcp._tool_manager.call_tool("search", {
            "query": "test", "rerank": True, "limit": 5,
        })

        assert "no reranker configured" in result.lower(), (
            f"Expected a note about missing reranker in response, got: {result[:200]}"
        )

    async def test_rerank_failure_falls_back(self, mock_store, mock_embedder, mock_reranker):
        mock_store.search = AsyncMock(return_value=[
            _make_chunk(qualified_name="Node2D.get_position", heading="get_position",
                        content="Returns position", similarity=0.3),
            _make_chunk(qualified_name="Node2D.set_position", heading="set_position",
                        content="Sets position", similarity=0.6),
        ])
        mock_reranker.rerank = AsyncMock(side_effect=ConnectionError("timeout"))
        srv = _build_test_server(mock_store, mock_embedder, mock_reranker)

        result = await srv.mcp._tool_manager.call_tool("search", {
            "query": "test", "rerank": True, "limit": 5,
        })

        # Should still return results, just not reranked
        assert "Node2D.get_position" in result
        assert "Node2D.set_position" in result
        assert "Error" not in result

    async def test_rerank_empty_candidates(self, mock_store, mock_embedder, mock_reranker):
        mock_store.search = AsyncMock(return_value=[])
        srv = _build_test_server(mock_store, mock_embedder, mock_reranker)

        result = await srv.mcp._tool_manager.call_tool("search", {
            "query": "test", "rerank": True, "limit": 5,
        })

        assert "No results found" in result
        mock_reranker.rerank.assert_not_called()

    async def test_rerank_single_candidate(self, mock_store, mock_embedder, mock_reranker):
        mock_store.search = AsyncMock(return_value=[
            _make_chunk(qualified_name="Node2D.get_position", heading="get_position",
                        content="Returns position", similarity=0.5),
        ])
        srv = _build_test_server(mock_store, mock_embedder, mock_reranker)

        result = await srv.mcp._tool_manager.call_tool("search", {
            "query": "test", "rerank": True, "limit": 5,
        })

        mock_reranker.rerank.assert_not_called()
        assert "Node2D.get_position" in result

    async def test_rerank_limit_clamping(self, mock_store, mock_embedder, mock_reranker):
        mock_store.search = AsyncMock(return_value=[])
        srv = _build_test_server(mock_store, mock_embedder, mock_reranker)

        await srv.mcp._tool_manager.call_tool("search", {
            "query": "test", "limit": 999,
        })

        # limit should be clamped to 50
        # Since reranker is configured, it calls search, not hybrid_search
        kwargs = mock_store.search.call_args.kwargs
        assert kwargs["limit"] == 50

    async def test_rerank_passes_chunk_types(self, mock_store, mock_embedder, mock_reranker):
        mock_store.search = AsyncMock(return_value=[
            _make_chunk(qualified_name="Node2D.get_position", heading="get_position",
                        content="Returns position", similarity=0.5),
            _make_chunk(qualified_name="Node2D.set_position", heading="set_position",
                        content="Sets position", similarity=0.4),
        ])
        mock_reranker.rerank = AsyncMock(return_value=[0.9, 0.3])
        srv = _build_test_server(mock_store, mock_embedder, mock_reranker)

        result = await srv.mcp._tool_manager.call_tool("search", {
            "query": "position", "rerank": True, "chunk_types": ["method"],
        })

        # chunk_types should be passed through to store.search
        kwargs = mock_store.search.call_args.kwargs
        assert kwargs["chunk_types"] == ["method"]
        assert "Node2D.get_position" in result

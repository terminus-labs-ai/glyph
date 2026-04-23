from __future__ import annotations

import json
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import FastMCP

from glyph.config import load_config, load_global_config, resolve_config_for_repo
from glyph.embedders.llama import LlamaEmbedder
from glyph.pipeline import run_export, run_ingest, run_search
from glyph.rerankers import LlamaReranker
from glyph.store import PostgresStore

logger = logging.getLogger(__name__)


class GlyphServer:
    """MCP server exposing the Glyph knowledge base."""

    def __init__(self, config_path: str):
        self._config_path = config_path
        self._store: PostgresStore | None = None
        self._embedder: LlamaEmbedder | None = None
        self._reranker: LlamaReranker | None = None
        self.mcp = FastMCP(
            "glyph",
            instructions="Glyph knowledge base server. Search API docs, look up classes/functions, and browse indexed sources.",
            lifespan=self._lifespan,
        )
        self._register_tools()
        self._register_resources()

    @asynccontextmanager
    async def _lifespan(self, server: FastMCP):
        cfg = load_config(self._config_path)
        self._store = PostgresStore(cfg.database.url, cfg.embedder.dimensions)
        await self._store.connect()
        self._embedder = LlamaEmbedder(
            cfg.embedder.url,
            cfg.embedder.model,
            cfg.embedder.dimensions,
            cfg.embedder.batch_size,
        )
        if cfg.reranker:
            self._reranker = LlamaReranker(
                cfg.reranker.url,
                cfg.reranker.model,
                cfg.reranker.batch_size,
                cfg.reranker.timeout,
            )
        logger.info("Glyph MCP server started")
        try:
            yield {}
        finally:
            await self._store.close()
            logger.info("Glyph MCP server stopped")

    def _register_tools(self) -> None:
        @self.mcp.tool()
        async def search(
            query: str,
            source: str | None = None,
            version: str | None = None,
            chunk_types: list[str] | None = None,
            parent: str | None = None,
            limit: int = 10,
            rerank: bool = True,
            candidates: int = 50,
        ) -> str:
            """Search the Glyph knowledge base using hybrid semantic + keyword search.

            Args:
                query: Natural language search query
                source: Filter to a specific source name (e.g., "godot", "stripe-api")
                version: Filter to a specific version
                chunk_types: Filter by chunk type (e.g., ["method", "property"])
                parent: Filter to chunks under a specific parent (e.g., "Node2D")
                limit: Number of results to return (default 10, max 50)
                rerank: Re-rank results using a cross-encoder reranker (default True,
                        ignored if no reranker is configured)
                candidates: Number of candidate chunks to fetch from pgvector before
                            reranking (default 50, only used when rerank=True;
                            must be >= limit)
            """
            if not self._store or not self._embedder:
                return "Error: Server not initialized"

            limit = max(1, min(limit, 50))
            if candidates < limit:
                candidates = limit

            # Determine retrieval strategy
            if rerank and self._reranker is None:
                logger.info("rerank=True but no reranker configured, using hybrid search")

            results = await run_search(
                self._store,
                self._embedder,
                self._reranker,
                query,
                source_name=source,
                source_version=version,
                chunk_types=chunk_types,
                parent_name=parent,
                limit=limit,
                rerank=rerank,
                n_candidates=candidates,
            )

            if not results:
                filters = _describe_filters(source=source, version=version, parent=parent, chunk_types=chunk_types)
                return f"No results found for \"{query}\"{filters}"

            return _format_search_results(results)

        @self.mcp.tool()
        async def lookup(qualified_name: str) -> str:
            """Look up a chunk by its exact qualified name.

            Args:
                qualified_name: e.g., "Node2D.get_position", "Users.createUser"
            """
            if not self._store:
                return "Error: Server not initialized"

            result = await self._store.get_by_qualified_name(qualified_name)
            if not result:
                return f"No chunk found with qualified_name \"{qualified_name}\""

            return _format_chunk_detail(result)

        @self.mcp.tool()
        async def get_context(
            parent_name: str,
            source: str | None = None,
            version: str | None = None,
        ) -> str:
            """Retrieve all chunks for a parent (class, module, etc.), structured as an overview.

            Args:
                parent_name: e.g., "Node2D", "Users", "player.Player"
                source: Disambiguate if same parent exists in multiple sources
                version: Filter to a specific version
            """
            if not self._store:
                return "Error: Server not initialized"

            chunks = await self._store.get_by_parent(
                parent_name,
                source_name=source,
                source_version=version,
            )

            if not chunks:
                filters = _describe_filters(source=source, version=version)
                return f"No chunks found for parent \"{parent_name}\"{filters}"

            return _format_context(parent_name, chunks)

        @self.mcp.tool()
        async def list_sources() -> str:
            """List all indexed sources in the Glyph knowledge base."""
            if not self._store:
                return "Error: Server not initialized"

            sources = await self._store.get_sources_with_counts()
            if not sources:
                return "No sources indexed."

            return _format_sources(sources)

        @self.mcp.tool()
        async def ingest_repo(
            path: str,
            name: str | None = None,
            version: str | None = None,
            skip_embeddings: bool = False,
            strict: bool = False,
        ) -> str:
            """Ingest a repository into the Glyph knowledge base.

            Indexes source code and documentation from a local directory.
            Uses .glyph.yaml from the repo root if present, otherwise
            auto-discovers languages and structure.

            Args:
                path: Absolute path to the repository directory
                name: Override the source name (default: directory name)
                version: Override the version (default: git tag/branch or "latest")
                skip_embeddings: Skip embedding generation for faster indexing
                strict: Fail hard if embedding endpoints are unreachable
            """
            try:
                global_cfg = load_config(self._config_path)
            except Exception:
                global_cfg = load_global_config()

            if global_cfg is None:
                return "Error: No global config found. Configure database and embedder settings."

            try:
                config, source_cfg = resolve_config_for_repo(
                    path,
                    global_config=global_cfg,
                    name_override=name,
                    version_override=version,
                )
            except ValueError as e:
                return f"Error: {e}"
            except FileNotFoundError:
                return f"Error: Directory not found: {path}"

            try:
                summary = await run_ingest(
                    config,
                    source_configs=[source_cfg],
                    skip_embeddings=skip_embeddings,
                    strict=strict,
                )
                sources = summary["sources"]
                source_info = sources[0] if sources else {"name": source_cfg.name, "version": source_cfg.version}
                return (
                    f"Indexed **{source_info['name']}** v{source_info['version']}: "
                    f"{summary['total_documents']} documents, {summary['total_chunks']} chunks"
                )
            except Exception as e:
                logger.exception("Ingest failed")
                return f"Error during ingest: {e}"

        @self.mcp.tool()
        async def export_source(
            source: str,
            version: str,
        ) -> str:
            """Export an indexed source as tiered markdown files.

            Generates three tiers of markdown output:
            - Tier 1: Index with class/module names and one-line summaries
            - Tier 2: Class summaries with member lists
            - Tier 3: Full detail for each class/module

            Args:
                source: Source name (e.g., "godot", "my-project")
                version: Source version (e.g., "4.4", "main")
            """
            try:
                config = load_config(self._config_path)
            except Exception:
                config = load_global_config()

            if config is None:
                return "Error: No config found."

            try:
                output_path = await run_export(config, source, version)
                return f"Exported **{source}** v{version} to `{output_path}`"
            except Exception as e:
                logger.exception("Export failed")
                return f"Error during export: {e}"

        @self.mcp.tool()
        async def reindex(
            path: str,
            files: list[str] | None = None,
            skip_embeddings: bool = False,
        ) -> str:
            """Reindex a repository or specific files within it.

            For incremental updates after code changes. If specific files
            are provided, only those files are re-processed. Otherwise,
            runs a full re-ingest (content hashing skips unchanged files).

            Args:
                path: Absolute path to the repository directory
                files: Specific file paths to reindex (for incremental updates)
                skip_embeddings: Skip embedding generation
            """
            try:
                global_cfg = load_config(self._config_path)
            except Exception:
                global_cfg = load_global_config()

            if global_cfg is None:
                return "Error: No config found."

            try:
                config, source_cfg = resolve_config_for_repo(path, global_config=global_cfg)
            except ValueError as e:
                return f"Error: {e}"

            try:
                summary = await run_ingest(
                    config,
                    source_configs=[source_cfg],
                    skip_embeddings=skip_embeddings,
                    file_filter=files,
                )
                file_msg = f" ({len(files)} files)" if files else ""
                return (
                    f"Reindexed **{source_cfg.name}** v{source_cfg.version}{file_msg}: "
                    f"{summary['total_documents']} documents, {summary['total_chunks']} chunks"
                )
            except Exception as e:
                logger.exception("Reindex failed")
                return f"Error during reindex: {e}"

    def _register_resources(self) -> None:
        @self.mcp.resource("glyph://sources")
        async def sources_resource() -> str:
            """List all available sources."""
            if not self._store:
                return "[]"
            sources = await self._store.get_sources_with_counts()
            return json.dumps(
                [{"name": s["name"], "version": s["version"]} for s in sources],
                indent=2,
            )

        @self.mcp.resource("glyph://sources/{source_name}/{version}/index")
        async def source_index(source_name: str, version: str) -> str:
            """Tier 1 index for a source — class names with one-liners."""
            if not self._store:
                return "Error: Server not initialized"

            chunks = await self._store.get_all_chunks(source_name, version)
            if not chunks:
                return f"No data found for {source_name} v{version}"

            by_parent: dict[str, list[dict]] = defaultdict(list)
            for chunk in chunks:
                by_parent[chunk["parent_name"]].append(chunk)

            lines = [f"# {source_name} v{version} — Index\n"]
            for name in sorted(by_parent):
                overview = _find_overview(by_parent[name])
                summary = overview.get("summary", "") if overview else ""
                lines.append(f"- **{name}** — {summary}")

            return "\n".join(lines)

        @self.mcp.resource("glyph://sources/{source_name}/{version}/classes/{class_name}")
        async def class_detail(source_name: str, version: str, class_name: str) -> str:
            """Full Tier 3 class/module detail."""
            if not self._store:
                return "Error: Server not initialized"

            chunks = await self._store.get_by_parent(
                class_name,
                source_name=source_name,
                source_version=version,
            )

            if not chunks:
                return f"No data found for {class_name} in {source_name} v{version}"

            return _format_context(class_name, chunks)

    def run(self, transport: str = "stdio", host: str = "127.0.0.1", port: int = 8420) -> None:
        if transport == "sse":
            self.mcp._host = host
            self.mcp._port = port
            self.mcp.run("sse")
        elif transport == "streamable-http":
            self.mcp._host = host
            self.mcp._port = port
            self.mcp.run("streamable-http")
        else:
            self.mcp.run("stdio")


# --- Formatting helpers ---


def _parse_metadata(metadata: Any) -> dict:
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            return json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _find_overview(chunks: list[dict]) -> dict | None:
    for c in chunks:
        if c["chunk_type"] == "class_overview":
            return c
    return None


def _describe_filters(**kwargs: Any) -> str:
    parts = []
    for key, val in kwargs.items():
        if val is not None:
            parts.append(f"{key}={val}")
    return f" (filters: {', '.join(parts)})" if parts else ""


def _format_search_results(results: list[dict[str, Any]]) -> str:
    lines = []
    for r in results:
        score = r.get("score", 0)
        retrieval = r.get("retrieval", "hybrid")
        rerank_score = r.get("rerank_score")
        tag = f"[{retrieval}]" if not rerank_score else f"[reranked]"
        lines.append(f"### {r['qualified_name']} {tag}")
        lines.append(
            f"**Type:** {r['chunk_type']} | "
            f"**Source:** {r['source_name']} {r['source_version']} | "
            f"**Score:** {score:.3f}"
        )
        if rerank_score is not None:
            lines.append(f"**Rerank Score:** {rerank_score:.3f}")
        if r.get("parent_name"):
            lines.append(f"**Parent:** {r['parent_name']}")
        lines.append("")
        if r.get("summary"):
            lines.append(r["summary"])
            lines.append("")
        lines.append(r.get("content", ""))
        lines.append("\n---\n")
    return "\n".join(lines)


def _format_chunk_detail(chunk: dict[str, Any]) -> str:
    lines = [
        f"# {chunk['qualified_name']}",
        "",
        f"**Type:** {chunk['chunk_type']} | "
        f"**Source:** {chunk['source_name']} {chunk['source_version']}",
        f"**Parent:** {chunk['parent_name']}",
    ]
    metadata = _parse_metadata(chunk.get("metadata", {}))
    if metadata.get("inherits"):
        lines.append(f"**Inherits:** {metadata['inherits']}")
    lines.append("")
    if chunk.get("summary"):
        lines.append(f"*{chunk['summary']}*")
        lines.append("")
    lines.append(chunk.get("content", ""))
    return "\n".join(lines)


def _format_context(parent_name: str, chunks: list[dict[str, Any]]) -> str:
    lines = [f"# {parent_name}\n"]

    overview = _find_overview(chunks)
    if overview:
        metadata = _parse_metadata(overview.get("metadata", {}))
        if metadata.get("inherits"):
            lines.append(f"**Inherits:** {metadata['inherits']}\n")
        src = f"{overview.get('source_name', '')} {overview.get('source_version', '')}"
        lines.append(f"**Source:** {src.strip()}\n")
        lines.append(overview.get("content", ""))
        lines.append("")

    by_type: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        if c["chunk_type"] != "class_overview":
            by_type[c["chunk_type"]].append(c)

    type_headings = {
        "property": "Properties",
        "method": "Methods",
        "signal": "Signals",
        "constant": "Constants",
        "enum": "Enumerations",
        "annotation": "Annotations",
        "tutorial_section": "Sections",
        "code_example": "Code Examples",
    }

    for chunk_type, heading in type_headings.items():
        members = by_type.get(chunk_type, [])
        if not members:
            continue
        lines.append(f"\n## {heading}\n")
        for m in members:
            lines.append(f"### {m['heading']}\n")
            if m.get("summary"):
                lines.append(f"*{m['summary']}*\n")
            lines.append(m.get("content", ""))
            lines.append("")

    return "\n".join(lines)


def _format_sources(sources: list[dict[str, Any]]) -> str:
    lines = ["# Indexed Sources\n"]
    lines.append("| Source | Version | Type | Documents | Chunks |")
    lines.append("|--------|---------|------|-----------|--------|")
    for s in sources:
        lines.append(
            f"| {s['name']} | {s['version']} | {s['source_type']} | "
            f"{s['document_count']} | {s['chunk_count']} |"
        )
    return "\n".join(lines)

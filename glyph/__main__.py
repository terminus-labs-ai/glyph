from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click

from glyph.config import load_config

logger = logging.getLogger("glyph")


@click.group()
@click.option("--config", "-c", default="glyph.yaml", help="Config file path")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
@click.pass_context
def cli(ctx: click.Context, config: str, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


@cli.command()
@click.pass_context
def init_db(ctx: click.Context) -> None:
    """Initialize database schema."""
    asyncio.run(_init_db(ctx.obj["config_path"]))


async def _init_db(config_path: str) -> None:
    from glyph.store import PostgresStore

    cfg = load_config(config_path)
    store = PostgresStore(cfg.database.url, cfg.embedder.dimensions)
    await _connect_store(store)
    await store.init_schema()
    await store.upgrade_schema()
    await store.close()
    logger.info("Database schema initialized")


async def _connect_store(store) -> None:
    try:
        await store.connect()
    except Exception as e:
        logger.error(f"Could not connect to database: {e}")
        logger.error("Check your glyph.yaml database.url setting")
        raise SystemExit(1)


@cli.command()
@click.option("--source", "-s", help="Only ingest a specific source by name")
@click.option("--skip-embeddings", is_flag=True, help="Skip embedding generation")
@click.option("--strict", is_flag=True, help="Fail hard if embedding endpoints are unreachable")
@click.pass_context
def ingest(ctx: click.Context, source: str | None, skip_embeddings: bool, strict: bool) -> None:
    """Ingest documentation from configured sources."""
    asyncio.run(_ingest(ctx.obj["config_path"], source, skip_embeddings, strict))


async def _ingest(config_path: str, source_filter: str | None, skip_embeddings: bool, strict: bool = False) -> None:
    from glyph.pipeline import run_ingest

    cfg = load_config(config_path)
    summary = await run_ingest(
        cfg,
        source_filter=source_filter,
        skip_embeddings=skip_embeddings,
        strict=strict,
    )
    logger.info(
        f"Ingest complete: {summary['total_documents']} documents, "
        f"{summary['total_chunks']} chunks across {len(summary['sources'])} sources"
    )


@cli.command()
@click.option("--source", "-s", required=True, help="Source name")
@click.option("--version", "-V", required=True, help="Source version")
@click.pass_context
def export(ctx: click.Context, source: str, version: str) -> None:
    """Export chunks as tiered markdown files."""
    asyncio.run(_export(ctx.obj["config_path"], source, version))


async def _export(config_path: str, source_name: str, source_version: str) -> None:
    from glyph.pipeline import run_export

    cfg = load_config(config_path)
    output_path = await run_export(cfg, source_name, source_version)
    logger.info(f"Export complete: {output_path}")


@cli.command()
@click.option("--path", "-p", required=True, type=click.Path(exists=True), help="Repository path to reindex")
@click.option("--files", "-f", multiple=True, help="Specific files to reindex (can be repeated)")
@click.option("--name", "-n", help="Override source name")
@click.option("--version", "-V", help="Override source version")
@click.option("--skip-embeddings", is_flag=True, help="Skip embedding generation")
@click.option("--strict", is_flag=True, help="Fail hard if embedding endpoints are unreachable")
@click.pass_context
def reindex(
    ctx: click.Context,
    path: str,
    files: tuple[str, ...],
    name: str | None,
    version: str | None,
    skip_embeddings: bool,
    strict: bool,
) -> None:
    """Reindex a repository (or specific files within it).

    Uses .glyph.yaml from the repo if present, otherwise auto-discovers.
    Database and embedder settings come from the global config or -c flag.
    """
    asyncio.run(_reindex(
        ctx.obj["config_path"], path, list(files) or None,
        name, version, skip_embeddings, strict,
    ))


async def _reindex(
    config_path: str,
    repo_path: str,
    files: list[str] | None,
    name_override: str | None,
    version_override: str | None,
    skip_embeddings: bool,
    strict: bool,
) -> None:
    from glyph.config import load_global_config, resolve_config_for_repo
    from glyph.pipeline import run_ingest

    # Try loading as full config first (backwards compat), fall back to global
    config_file = Path(config_path)
    if config_file.exists():
        try:
            global_cfg = load_config(config_path)
        except (KeyError, TypeError):
            global_cfg = load_global_config(config_path)
    else:
        global_cfg = load_global_config()

    global_cfg, source_cfg = resolve_config_for_repo(
        repo_path,
        global_config=global_cfg,
        name_override=name_override,
        version_override=version_override,
    )

    summary = await run_ingest(
        global_cfg,
        source_configs=[source_cfg],
        skip_embeddings=skip_embeddings,
        strict=strict,
        file_filter=files,
    )

    logger.info(
        f"Reindex complete: {summary['total_documents']} documents, "
        f"{summary['total_chunks']} chunks"
    )


@cli.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show database statistics."""
    asyncio.run(_stats(ctx.obj["config_path"]))


async def _stats(config_path: str) -> None:
    from glyph.store import PostgresStore

    cfg = load_config(config_path)
    store = PostgresStore(cfg.database.url, cfg.embedder.dimensions)
    await _connect_store(store)
    result = await store.get_stats()
    await store.close()

    click.echo(f"Sources:   {result['sources']}")
    click.echo(f"Documents: {result['documents']}")
    click.echo(f"Chunks:    {result['chunks']}")
    if result["by_type"]:
        click.echo("By type:")
        for chunk_type, count in result["by_type"].items():
            click.echo(f"  {chunk_type:20s} {count}")


@cli.command()
@click.argument("query")
@click.option("--source", "-s", help="Filter by source name")
@click.option("--version", "-V", help="Filter by source version")
@click.option("--type", "chunk_types", help="Comma-separated chunk types (e.g. method,property)")
@click.option("--parent", help="Filter to chunks under a specific parent (e.g. Node2D)")
@click.option("--limit", default=10, show_default=True, help="Number of results")
@click.option("--rerank/--no-rerank", default=None, help="Enable/disable reranking (default: auto)")
@click.option("--candidates", default=50, show_default=True, help="Candidates for pgvector before rerank")
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    source: str | None,
    version: str | None,
    chunk_types: str | None,
    parent: str | None,
    limit: int,
    rerank: bool | None,
    candidates: int,
) -> None:
    """Search the knowledge base using hybrid semantic + keyword search."""
    asyncio.run(_search(ctx.obj["config_path"], query, source, version, chunk_types, parent, limit, rerank, candidates))


async def _search(
    config_path: str,
    query: str,
    source: str | None,
    version: str | None,
    chunk_types_str: str | None,
    parent: str | None,
    limit: int,
    rerank: bool | None,
    n_candidates: int,
) -> None:
    from glyph.embedders.llama import LlamaEmbedder
    from glyph.pipeline import run_search
    from glyph.rerankers import LlamaReranker
    from glyph.store import PostgresStore

    cfg = load_config(config_path)
    store = PostgresStore(cfg.database.url, cfg.embedder.dimensions)
    await _connect_store(store)

    chunk_types = [t.strip() for t in chunk_types_str.split(",")] if chunk_types_str else None

    # Determine if reranking is available and desired
    reranker_available = cfg.reranker is not None
    use_rerank = rerank if rerank is not None else reranker_available

    if rerank and not reranker_available:
        logger.info("rerank=True but no reranker configured, using hybrid search")

    # Build reranker instance when needed
    reranker = None
    if reranker_available:
        reranker = LlamaReranker(
            cfg.reranker.url,
            cfg.reranker.model,
            cfg.reranker.batch_size,
            cfg.reranker.timeout,
        )

    embedder = LlamaEmbedder(
        cfg.embedder.url,
        cfg.embedder.model,
        cfg.embedder.dimensions,
        cfg.embedder.batch_size,
    )

    results = await run_search(
        store,
        embedder,
        reranker,
        query,
        source_name=source,
        source_version=version,
        chunk_types=chunk_types,
        parent_name=parent,
        limit=limit,
        rerank=use_rerank,
        n_candidates=n_candidates,
    )

    await store.close()

    if not results:
        click.echo(f'No results for "{query}"')
        return

    for i, r in enumerate(results, 1):
        rerank_score = r.get("rerank_score")
        retrieval = r.get("retrieval", "hybrid")
        tag = f"[reranked]" if rerank_score is not None else (r.get("retrieval_tag", "") or f"[{retrieval}]")
        score = r.get("score", 0.0)
        qualified_name = r.get("qualified_name") or r.get("heading") or "(unnamed)"
        chunk_type = r.get("chunk_type", "")
        src = f"{r.get('source_name', '')} {r.get('source_version', '')}".strip()
        summary = (r.get("summary") or "").strip()
        if len(summary) > 120:
            summary = summary[:117] + "..."

        header = click.style(f"{i}. {qualified_name}", bold=True)
        meta = click.style(f"  [{chunk_type}] {src} {tag} score={score:.4f}", fg="cyan")
        if rerank_score is not None:
            meta = click.style(f"  [{chunk_type}] {src} {tag} rerank={rerank_score:.4f}", fg="cyan")
        click.echo(header)
        click.echo(meta)
        if summary:
            click.echo(f"  {summary}")
        click.echo()


@cli.command()
@click.option(
    "--transport", "-t",
    type=click.Choice(["stdio", "sse", "streamable-http"]),
    default="stdio",
    help="MCP transport (default: stdio)",
)
@click.option("--host", "-H", default="127.0.0.1", help="Host for SSE/HTTP transport")
@click.option("--port", "-p", default=8420, type=int, help="Port for SSE/HTTP transport")
@click.pass_context
def serve(ctx: click.Context, transport: str, host: str, port: int) -> None:
    """Start the MCP server for Glyph knowledge base queries."""
    from glyph.server import GlyphServer

    server = GlyphServer(ctx.obj["config_path"])
    server.run(transport=transport, host=host, port=port)


if __name__ == "__main__":
    cli()

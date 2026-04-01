from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click

from ragify.config import load_config
from ragify.domain.models import Source

logger = logging.getLogger("ragify")


@click.group()
@click.option("--config", "-c", default="ragify.yaml", help="Config file path")
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
    from ragify.store import PostgresStore

    cfg = load_config(config_path)
    store = PostgresStore(cfg.database.url, cfg.embedder.dimensions)
    await store.connect()
    await store.init_schema()
    await store.close()
    logger.info("Database schema initialized")


@cli.command()
@click.option("--source", "-s", help="Only ingest a specific source by name")
@click.option("--skip-embeddings", is_flag=True, help="Skip embedding generation")
@click.pass_context
def ingest(ctx: click.Context, source: str | None, skip_embeddings: bool) -> None:
    """Ingest documentation from configured sources."""
    asyncio.run(_ingest(ctx.obj["config_path"], source, skip_embeddings))


async def _ingest(config_path: str, source_filter: str | None, skip_embeddings: bool) -> None:
    from ragify.chunkers.api_chunker import APIChunker
    from ragify.chunkers.text_chunker import TextChunker
    from ragify.domain.models import DocType
    from ragify.embedders.llama import LlamaEmbedder
    from ragify.ingestors.godot_xml import GodotXMLIngestor
    from ragify.ingestors.html import HTMLIngestor
    from ragify.store import PostgresStore

    cfg = load_config(config_path)
    store = PostgresStore(cfg.database.url, cfg.embedder.dimensions)
    await store.connect()
    await store.init_schema()

    embedder = None
    if not skip_embeddings:
        embedder = LlamaEmbedder(
            cfg.embedder.url,
            cfg.embedder.model,
            cfg.embedder.dimensions,
            cfg.embedder.batch_size,
        )

    for src_cfg in cfg.sources:
        if source_filter and src_cfg.name != source_filter:
            continue

        logger.info(f"Processing source: {src_cfg.name} v{src_cfg.version}")

        for ing_cfg in src_cfg.ingestors:
            # Create source record
            source_type = ing_cfg.type
            origin = ing_cfg.settings.get("path") or ing_cfg.settings.get("base_url", "")

            source_obj = Source(
                name=src_cfg.name,
                version=src_cfg.version,
                source_type=source_type,
                origin=origin,
            )
            source_id = await store.upsert_source(source_obj)
            source_obj.id = source_id

            # Build ingestor
            ingestor = _build_ingestor(ing_cfg.type, ing_cfg.settings, source_id)
            if not ingestor:
                logger.warning(f"Unknown ingestor type: {ing_cfg.type}")
                continue

            # Ingest documents
            documents = await ingestor.ingest()
            logger.info(f"Ingested {len(documents)} documents from {ing_cfg.type}")

            # Build chunkers
            api_chunker = APIChunker(src_cfg.name, src_cfg.version)
            text_chunker = TextChunker(src_cfg.name, src_cfg.version)

            source_code_chunker = None
            if ing_cfg.type == "source_code":
                from ragify.chunkers.source_code_chunker import SourceCodeChunker
                include_bodies = ing_cfg.settings.get("include_bodies", False)
                source_code_chunker = SourceCodeChunker(
                    src_cfg.name, src_cfg.version,
                    include_bodies=include_bodies,
                )

            total_chunks = 0
            for doc in documents:
                doc.source_id = source_id
                doc_id, changed = await store.upsert_document(doc)
                doc.id = doc_id

                if not changed:
                    logger.debug(f"Skipping unchanged: {doc.title}")
                    continue

                # Delete old chunks for this document
                await store.delete_chunks_for_document(doc_id)

                # Chunk based on ingestor type and doc type
                if source_code_chunker:
                    chunks = source_code_chunker.chunk(doc)
                elif doc.doc_type == DocType.CLASS_REF:
                    chunks = api_chunker.chunk(doc)
                else:
                    chunks = text_chunker.chunk(doc)

                # Set document_id on all chunks
                for chunk in chunks:
                    chunk.document_id = doc_id

                # Generate embeddings
                if embedder and chunks:
                    texts = [c.content for c in chunks]
                    logger.info(f"Generating embeddings for {len(texts)} chunks from {doc.title}")
                    embeds = await embedder.embed(texts)
                    for chunk, emb in zip(chunks, embeds):
                        chunk.embedding = emb

                inserted = await store.insert_chunks(chunks)
                total_chunks += inserted

            logger.info(f"Stored {total_chunks} chunks from {ing_cfg.type}")

    stats = await store.get_stats()
    logger.info(f"Database stats: {stats}")
    await store.close()


def _build_ingestor(ingestor_type: str, settings: dict, source_id):
    from ragify.ingestors.godot_xml import GodotXMLIngestor
    from ragify.ingestors.html import HTMLIngestor
    from ragify.ingestors.source_code import SourceCodeIngestor

    if ingestor_type == "godot_xml":
        return GodotXMLIngestor(settings["path"], source_id)
    elif ingestor_type == "html":
        return HTMLIngestor(
            settings["base_url"],
            source_id,
            max_pages=settings.get("max_pages", 500),
            delay=settings.get("delay", 0.2),
            include_patterns=settings.get("include_patterns"),
            exclude_patterns=settings.get("exclude_patterns"),
            max_concurrent=settings.get("max_concurrent", 10),
        )
    elif ingestor_type == "source_code":
        return SourceCodeIngestor(
            settings["path"],
            source_id,
            extensions=settings.get("extensions"),
            exclude_dirs=settings.get("exclude_dirs"),
            exclude_patterns=settings.get("exclude_patterns"),
        )
    return None


@cli.command()
@click.option("--source", "-s", required=True, help="Source name")
@click.option("--version", "-V", required=True, help="Source version")
@click.pass_context
def export(ctx: click.Context, source: str, version: str) -> None:
    """Export chunks as tiered markdown files."""
    asyncio.run(_export(ctx.obj["config_path"], source, version))


async def _export(config_path: str, source_name: str, source_version: str) -> None:
    from ragify.exporters.markdown import MarkdownExporter
    from ragify.store import PostgresStore

    cfg = load_config(config_path)
    store = PostgresStore(cfg.database.url, cfg.embedder.dimensions)
    await store.connect()

    chunks = await store.get_all_chunks(source_name, source_version)
    logger.info(f"Exporting {len(chunks)} chunks for {source_name} v{source_version}")

    exporter = MarkdownExporter(cfg.output.directory)
    output_path = exporter.export(chunks, source_name, source_version)
    logger.info(f"Export complete: {output_path}")

    await store.close()


@cli.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show database statistics."""
    asyncio.run(_stats(ctx.obj["config_path"]))


async def _stats(config_path: str) -> None:
    from ragify.store import PostgresStore

    cfg = load_config(config_path)
    store = PostgresStore(cfg.database.url, cfg.embedder.dimensions)
    await store.connect()
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
    """Start the MCP server for RAGify knowledge base queries."""
    from ragify.server import RagifyServer

    server = RagifyServer(ctx.obj["config_path"])
    server.run(transport=transport, host=host, port=port)


if __name__ == "__main__":
    cli()

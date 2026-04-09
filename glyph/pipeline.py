"""Reusable ingest/export/reindex pipeline.

Decoupled from the CLI — can be called from MCP tools, scripts, or tests.
"""

from __future__ import annotations

import logging
from pathlib import Path

from glyph.config import Config, SourceConfig
from glyph.domain.models import DocType, Source

logger = logging.getLogger(__name__)


async def run_ingest(
    config: Config,
    source_configs: list[SourceConfig] | None = None,
    source_filter: str | None = None,
    skip_embeddings: bool = False,
    strict: bool = False,
    file_filter: list[str] | None = None,
) -> dict:
    """Run the ingest pipeline.

    Args:
        config: Full config with database/embedder settings.
        source_configs: Source configs to ingest. If None, uses config.sources.
        source_filter: Only ingest sources matching this name (applies to config.sources).
        skip_embeddings: Skip embedding generation.
        strict: Fail hard if embedding endpoints are unreachable.
        file_filter: Only process documents matching these paths (for incremental reindex).

    Returns:
        Summary dict with total_documents, total_chunks, sources processed.
    """
    from glyph.embedders.llama import LlamaEmbedder
    from glyph.store import PostgresStore

    store = PostgresStore(config.database.url, config.embedder.dimensions)
    await store.connect()
    await store.init_schema()

    embedder = None
    if not skip_embeddings:
        embedder = LlamaEmbedder(
            config.embedder.url,
            config.embedder.model,
            config.embedder.dimensions,
            config.embedder.batch_size,
            strict=strict,
        )

    sources_to_process = source_configs or config.sources
    if source_filter and not source_configs:
        sources_to_process = [s for s in sources_to_process if s.name == source_filter]

    summary = {
        "sources": [],
        "total_documents": 0,
        "total_chunks": 0,
    }

    try:
        for src_cfg in sources_to_process:
            source_summary = await _ingest_source(
                store, embedder, src_cfg, file_filter=file_filter,
            )
            summary["sources"].append(source_summary)
            summary["total_documents"] += source_summary["documents"]
            summary["total_chunks"] += source_summary["chunks"]

    finally:
        await store.close()

    return summary


async def run_export(
    config: Config,
    source_name: str,
    source_version: str,
) -> str:
    """Run the export pipeline.

    Returns:
        The output directory path.
    """
    from glyph.exporters.markdown import MarkdownExporter
    from glyph.store import PostgresStore

    store = PostgresStore(config.database.url, config.embedder.dimensions)
    await store.connect()

    try:
        chunks = await store.get_all_chunks(source_name, source_version)
        logger.info(f"Exporting {len(chunks)} chunks for {source_name} v{source_version}")

        exporter = MarkdownExporter(config.output.directory)
        output_path = exporter.export(chunks, source_name, source_version)
        logger.info(f"Export complete: {output_path}")
        return str(output_path)

    finally:
        await store.close()


async def _ingest_source(
    store,
    embedder,
    src_cfg: SourceConfig,
    file_filter: list[str] | None = None,
) -> dict:
    """Ingest a single source config. Returns summary dict."""
    from glyph.chunkers.api_chunker import APIChunker
    from glyph.chunkers.text_chunker import TextChunker

    source_summary = {
        "name": src_cfg.name,
        "version": src_cfg.version,
        "documents": 0,
        "chunks": 0,
    }

    logger.info(f"Processing source: {src_cfg.name} v{src_cfg.version}")

    for ing_cfg in src_cfg.ingestors:
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

        # Filter to specific files if requested (for incremental reindex)
        if file_filter:
            source_root = Path(ing_cfg.settings.get("path", ".")).resolve()
            normalized_filter = {str(Path(f).resolve()) for f in file_filter}
            documents = [
                d for d in documents
                if str((source_root / d.path).resolve()) in normalized_filter
            ]
            logger.info(f"Filtered to {len(documents)} documents matching file filter")

        # Build chunkers
        api_chunker = APIChunker(src_cfg.name, src_cfg.version)
        text_chunker = TextChunker(src_cfg.name, src_cfg.version)

        source_code_chunker = None
        if ing_cfg.type == "source_code":
            from glyph.chunkers.source_code_chunker import SourceCodeChunker
            include_bodies = ing_cfg.settings.get("include_bodies", False)
            source_code_chunker = SourceCodeChunker(
                src_cfg.name, src_cfg.version,
                include_bodies=include_bodies,
            )

        unreal_doc_chunker = None
        if ing_cfg.type == "unreal_doc":
            from glyph.chunkers.unreal_doc_chunker import UnrealDocChunker
            unreal_doc_chunker = UnrealDocChunker(
                src_cfg.name, src_cfg.version,
                json_path=ing_cfg.settings["path"],
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
            elif unreal_doc_chunker:
                chunks = unreal_doc_chunker.chunk(doc)
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

        source_summary["documents"] += len(documents)
        source_summary["chunks"] += total_chunks
        logger.info(f"Stored {total_chunks} chunks from {ing_cfg.type}")

    return source_summary


def _build_ingestor(ingestor_type: str, settings: dict, source_id):
    """Build an ingestor instance from type and settings."""
    from glyph.ingestors.godot_xml import GodotXMLIngestor
    from glyph.ingestors.html import HTMLIngestor
    from glyph.ingestors.source_code import SourceCodeIngestor

    if ingestor_type == "godot_xml":
        return GodotXMLIngestor(
            settings["path"],
            source_id,
            include_patterns=settings.get("include_patterns"),
            exclude_patterns=settings.get("exclude_patterns"),
        )
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
    elif ingestor_type == "unreal_doc":
        from glyph.ingestors.unreal_doc import UnrealDocIngestor
        return UnrealDocIngestor(settings["path"], source_id)
    elif ingestor_type == "docs":
        from glyph.ingestors.docs import DocsIngestor
        return DocsIngestor(
            settings["path"],
            source_id,
            extensions=settings.get("extensions"),
            include_patterns=settings.get("include_patterns"),
            exclude_patterns=settings.get("exclude_patterns"),
            exclude_dirs=settings.get("exclude_dirs"),
        )
    return None

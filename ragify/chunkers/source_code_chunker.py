from __future__ import annotations

import logging
from pathlib import Path

from ragify.domain.models import Chunk, ChunkType, Document
from ragify.chunkers._parsers import get_parser

logger = logging.getLogger(__name__)

# Map file extensions to language identifiers
EXTENSION_MAP = {
    ".py": "python",
    ".gd": "gdscript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".rs": "rust",
    ".go": "go",
}


class SourceCodeChunker:
    """Chunks source code files into class/function/method chunks.

    Uses tree-sitter for supported languages, regex-based fallback otherwise.
    """

    def __init__(
        self,
        source_name: str,
        source_version: str,
        *,
        include_bodies: bool = False,
    ):
        self._source_name = source_name
        self._source_version = source_version
        self._include_bodies = include_bodies

    def chunk(self, document: Document) -> list[Chunk]:
        ext = Path(document.path).suffix
        lang = EXTENSION_MAP.get(ext)
        if not lang:
            return []

        parser = get_parser(lang)
        if not parser:
            logger.warning(f"No parser for language: {lang}")
            return []

        symbols = parser.parse(document.raw_content, include_bodies=self._include_bodies)

        chunks: list[Chunk] = []
        file_stem = Path(document.path).stem

        for idx, sym in enumerate(symbols):
            # Build qualified name
            if sym.parent:
                qualified = f"{file_stem}.{sym.parent}.{sym.name}"
                parent = f"{file_stem}.{sym.parent}"
            else:
                qualified = f"{file_stem}.{sym.name}"
                parent = file_stem

            chunks.append(Chunk(
                document_id=document.id,
                source_name=self._source_name,
                source_version=self._source_version,
                chunk_type=sym.chunk_type,
                qualified_name=qualified,
                parent_name=parent,
                heading=sym.name,
                summary=sym.summary,
                content=sym.content,
                metadata=sym.metadata,
                token_count=_estimate_tokens(sym.content),
                chunk_index=idx,
            ))

        return chunks


def _estimate_tokens(text: str) -> int:
    return len(text.split()) * 4 // 3

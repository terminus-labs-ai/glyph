from __future__ import annotations

import re

from ragify.domain.models import Chunk, ChunkType, DocType, Document


class TextChunker:
    """Chunks unstructured text documents (tutorials, guides) by headings."""

    def __init__(
        self,
        source_name: str,
        source_version: str,
        *,
        max_chunk_size: int = 2000,
    ):
        self._source_name = source_name
        self._source_version = source_version
        self._max_chunk_size = max_chunk_size

    def chunk(self, document: Document) -> list[Chunk]:
        if document.doc_type == DocType.CLASS_REF:
            return []

        sections = self._split_by_headings(document.raw_content)
        chunks: list[Chunk] = []

        for idx, (heading, content) in enumerate(sections):
            if not content.strip():
                continue

            # Split oversized sections
            sub_parts = self._split_large(content)
            for sub_idx, part in enumerate(sub_parts):
                qualified = f"{document.title}.{heading}" if heading else document.title
                if len(sub_parts) > 1:
                    qualified += f".{sub_idx}"

                chunks.append(Chunk(
                    document_id=document.id,
                    source_name=self._source_name,
                    source_version=self._source_version,
                    chunk_type=ChunkType.TUTORIAL_SECTION,
                    qualified_name=qualified,
                    parent_name=document.title,
                    heading=heading or document.title,
                    summary=_first_sentence(part),
                    content=part,
                    token_count=_estimate_tokens(part),
                    chunk_index=idx * 100 + sub_idx,
                ))

        return chunks

    def _split_by_headings(self, text: str) -> list[tuple[str, str]]:
        # Split on markdown-style headings or lines that look like section headers
        heading_re = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
        matches = list(heading_re.finditer(text))

        if not matches:
            return [("", text)]

        sections = []
        # Text before first heading
        pre = text[: matches[0].start()].strip()
        if pre:
            sections.append(("", pre))

        for i, match in enumerate(matches):
            heading = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            sections.append((heading, content))

        return sections

    def _split_large(self, text: str) -> list[str]:
        if len(text) <= self._max_chunk_size:
            return [text]

        parts = []
        paragraphs = text.split("\n\n")
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 2 > self._max_chunk_size and current:
                parts.append(current.strip())
                current = para
            else:
                current = current + "\n\n" + para if current else para

        if current.strip():
            parts.append(current.strip())

        return parts or [text]


def _first_sentence(text: str) -> str:
    if not text:
        return ""
    for sep in (". ", ".\n"):
        pos = text.find(sep)
        if pos > 0:
            return text[: pos + 1]
    return text[:200]


def _estimate_tokens(text: str) -> int:
    return len(text.split()) * 4 // 3

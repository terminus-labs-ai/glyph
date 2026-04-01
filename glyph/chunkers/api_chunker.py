from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path
from xml.etree import ElementTree as ET

from glyph.domain.models import Chunk, ChunkType, DocType, Document

logger = logging.getLogger(__name__)


class APIChunker:
    """Chunks structured API documents into semantic pieces.

    For Godot XML class references, parses the original XML to produce
    one chunk per class overview, method, property, signal, and constant.
    """

    def __init__(self, source_name: str, source_version: str):
        self._source_name = source_name
        self._source_version = source_version

    def chunk(self, document: Document) -> list[Chunk]:
        if document.doc_type != DocType.CLASS_REF:
            return []

        path = Path(document.path)

        # If it's an XML file, parse structured chunks from XML
        if path.suffix == ".xml" and path.exists():
            return self._chunk_xml(document, path)

        # Otherwise chunk from raw content
        return self._chunk_from_text(document)

    def _chunk_xml(self, document: Document, xml_path: Path) -> list[Chunk]:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            logger.error(f"XML parse error for {xml_path}: {e}")
            return []

        class_name = root.get("name", document.title)
        inherits = root.get("inherits", "")
        chunks: list[Chunk] = []
        idx = 0

        # Class overview chunk
        brief = root.findtext("brief_description", "").strip()
        desc = root.findtext("description", "").strip()
        overview_parts = []
        if inherits:
            overview_parts.append(f"**Inherits:** {inherits}")
        if brief:
            overview_parts.append(brief)
        if desc:
            overview_parts.append(desc)

        if overview_parts:
            content = "\n\n".join(overview_parts)
            chunks.append(Chunk(
                document_id=document.id,
                source_name=self._source_name,
                source_version=self._source_version,
                chunk_type=ChunkType.CLASS_OVERVIEW,
                qualified_name=class_name,
                parent_name=class_name,
                heading=class_name,
                summary=brief or _first_sentence(desc),
                content=content,
                metadata={"inherits": inherits} if inherits else {},
                token_count=_estimate_tokens(content),
                chunk_index=idx,
            ))
            idx += 1

        # Methods
        for method_el in root.iter("method"):
            chunk = self._parse_method(document, class_name, method_el, idx)
            if chunk:
                chunks.append(chunk)
                idx += 1

        # Properties / members
        for member_el in root.iter("member"):
            chunk = self._parse_member(document, class_name, member_el, idx)
            if chunk:
                chunks.append(chunk)
                idx += 1

        # Signals
        for signal_el in root.iter("signal"):
            chunk = self._parse_signal(document, class_name, signal_el, idx)
            if chunk:
                chunks.append(chunk)
                idx += 1

        # Constants
        for const_el in root.iter("constant"):
            chunk = self._parse_constant(document, class_name, const_el, idx)
            if chunk:
                chunks.append(chunk)
                idx += 1

        return chunks

    def _parse_method(
        self, doc: Document, class_name: str, el: ET.Element, idx: int
    ) -> Chunk | None:
        name = el.get("name", "")
        if not name:
            return None

        ret_el = el.find("return")
        ret_type = ret_el.get("type", "void") if ret_el is not None else "void"

        params = []
        param_dicts = []
        for p in el.findall("param"):
            p_name = p.get("name", "")
            p_type = p.get("type", "Variant")
            p_default = p.get("default", "")
            sig = f"{p_name}: {p_type}"
            if p_default:
                sig += f" = {p_default}"
            params.append(sig)
            param_dicts.append({
                "name": p_name, "type": p_type, "default": p_default or None,
            })

        signature = f"{name}({', '.join(params)}) -> {ret_type}"
        description = el.findtext("description", "").strip()

        content = f"```gdscript\n{signature}\n```"
        if description:
            content += f"\n\n{description}"

        qualifiers = el.get("qualifiers", "")

        return Chunk(
            document_id=doc.id,
            source_name=self._source_name,
            source_version=self._source_version,
            chunk_type=ChunkType.METHOD,
            qualified_name=f"{class_name}.{name}",
            parent_name=class_name,
            heading=name,
            summary=_first_sentence(description) or signature,
            content=content,
            metadata={
                "return_type": ret_type,
                "params": param_dicts,
                "qualifiers": qualifiers or None,
            },
            token_count=_estimate_tokens(content),
            chunk_index=idx,
        )

    def _parse_member(
        self, doc: Document, class_name: str, el: ET.Element, idx: int
    ) -> Chunk | None:
        name = el.get("name", "")
        if not name:
            return None

        m_type = el.get("type", "Variant")
        default = el.get("default", "")
        description = (el.text or "").strip()

        sig = f"{name}: {m_type}"
        if default:
            sig += f" = {default}"

        content = f"`{sig}`"
        if description:
            content += f"\n\n{description}"

        return Chunk(
            document_id=doc.id,
            source_name=self._source_name,
            source_version=self._source_version,
            chunk_type=ChunkType.PROPERTY,
            qualified_name=f"{class_name}.{name}",
            parent_name=class_name,
            heading=name,
            summary=_first_sentence(description) or sig,
            content=content,
            metadata={
                "type": m_type,
                "default": default or None,
                "setter": el.get("setter", None),
                "getter": el.get("getter", None),
            },
            token_count=_estimate_tokens(content),
            chunk_index=idx,
        )

    def _parse_signal(
        self, doc: Document, class_name: str, el: ET.Element, idx: int
    ) -> Chunk | None:
        name = el.get("name", "")
        if not name:
            return None

        params = []
        for p in el.findall("param"):
            p_name = p.get("name", "")
            p_type = p.get("type", "")
            params.append(f"{p_name}: {p_type}")

        description = el.findtext("description", "").strip()
        sig = f"{name}({', '.join(params)})" if params else name
        content = f"`signal {sig}`"
        if description:
            content += f"\n\n{description}"

        return Chunk(
            document_id=doc.id,
            source_name=self._source_name,
            source_version=self._source_version,
            chunk_type=ChunkType.SIGNAL,
            qualified_name=f"{class_name}.{name}",
            parent_name=class_name,
            heading=name,
            summary=_first_sentence(description) or f"Signal: {sig}",
            content=content,
            metadata={"params": params} if params else {},
            token_count=_estimate_tokens(content),
            chunk_index=idx,
        )

    def _parse_constant(
        self, doc: Document, class_name: str, el: ET.Element, idx: int
    ) -> Chunk | None:
        name = el.get("name", "")
        if not name:
            return None

        value = el.get("value", "")
        enum_name = el.get("enum", "")
        description = (el.text or "").strip()

        content = f"`{name} = {value}`"
        if description:
            content += f"\n\n{description}"

        qualified = f"{class_name}.{name}"
        if enum_name:
            qualified = f"{class_name}.{enum_name}.{name}"

        return Chunk(
            document_id=doc.id,
            source_name=self._source_name,
            source_version=self._source_version,
            chunk_type=ChunkType.ENUM if enum_name else ChunkType.CONSTANT,
            qualified_name=qualified,
            parent_name=class_name,
            heading=name,
            summary=_first_sentence(description) or f"{name} = {value}",
            content=content,
            metadata={
                "value": value,
                "enum": enum_name or None,
            },
            token_count=_estimate_tokens(content),
            chunk_index=idx,
        )

    def _chunk_from_text(self, document: Document) -> list[Chunk]:
        """Fallback: produce a single class_overview chunk from raw text."""
        return [Chunk(
            document_id=document.id,
            source_name=self._source_name,
            source_version=self._source_version,
            chunk_type=ChunkType.CLASS_OVERVIEW,
            qualified_name=document.title,
            parent_name=document.title,
            heading=document.title,
            summary=_first_sentence(document.raw_content),
            content=document.raw_content,
            token_count=_estimate_tokens(document.raw_content),
            chunk_index=0,
        )]


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

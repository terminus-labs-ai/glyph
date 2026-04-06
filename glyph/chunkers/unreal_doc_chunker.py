from __future__ import annotations

import json
import logging
from pathlib import Path

from glyph.domain.models import Chunk, ChunkType, DocType, Document

logger = logging.getLogger(__name__)


class UnrealDocChunker:
    """Produces fine-grained chunks from unreal-doc JSON output.

    Reads the source documentation.json once, then produces per-method,
    per-property, and per-enum chunks for each document.
    """

    def __init__(self, source_name: str, source_version: str, json_path: str):
        self._source_name = source_name
        self._source_version = source_version
        self._data: dict | None = None
        self._json_path = Path(json_path)
        self._index: dict[str, dict] = {}

    def _ensure_loaded(self) -> None:
        if self._data is not None:
            return
        with open(self._json_path, encoding="utf-8") as f:
            self._data = json.load(f)
        # Build lookup index by name
        for cls in self._data.get("classes", []):
            if cls.get("name"):
                self._index[cls["name"]] = cls
        for struct in self._data.get("structs", []):
            if struct.get("name"):
                self._index[struct["name"]] = struct
        for enum in self._data.get("enums", []):
            if enum.get("name"):
                self._index[enum["name"]] = enum

    def chunk(self, document: Document) -> list[Chunk]:
        if document.doc_type == DocType.API_OVERVIEW:
            return self._chunk_global_functions(document)
        if document.doc_type != DocType.CLASS_REF:
            return []

        self._ensure_loaded()
        item = self._index.get(document.title)
        if not item:
            return [self._fallback_chunk(document)]

        # Enum vs struct/class
        if "variants" in item:
            return self._chunk_enum(document, item)
        return self._chunk_struct_class(document, item)

    def _chunk_struct_class(self, document: Document, item: dict) -> list[Chunk]:
        name = item["name"]
        chunks: list[Chunk] = []
        idx = 0

        # Class/struct overview
        mode = item.get("mode", "Struct")
        overview_parts = [f"**Type:** {mode.lower()}"]
        inherits = item.get("inherits", [])
        if inherits:
            parents = [pair[1] for pair in inherits if len(pair) == 2]
            if parents:
                overview_parts.append(f"**Inherits:** {', '.join(parents)}")
        if item.get("template"):
            overview_parts.append(f"**Template:** `{item['template']}`")
        if item.get("api"):
            overview_parts.append(f"**API:** {item['api']}")
        if item.get("doc_comments"):
            overview_parts.append(item["doc_comments"])

        content = "\n\n".join(overview_parts)
        chunks.append(Chunk(
            document_id=document.id,
            source_name=self._source_name,
            source_version=self._source_version,
            chunk_type=ChunkType.CLASS_OVERVIEW,
            qualified_name=name,
            parent_name=name,
            heading=name,
            summary=_first_sentence(item.get("doc_comments", "")) or f"{mode} {name}",
            content=content,
            metadata={
                "mode": mode.lower(),
                "inherits": [p[1] for p in inherits if len(p) == 2] or None,
                "api": item.get("api"),
            },
            token_count=_estimate_tokens(content),
            chunk_index=idx,
        ))
        idx += 1

        # Methods
        for method in item.get("methods", []):
            chunk = self._chunk_method(document, name, method, idx)
            if chunk:
                chunks.append(chunk)
                idx += 1

        # Properties
        for prop in item.get("properties", []):
            chunk = self._chunk_property(document, name, prop, idx)
            if chunk:
                chunks.append(chunk)
                idx += 1

        return chunks

    def _chunk_method(
        self, doc: Document, class_name: str, method: dict, idx: int
    ) -> Chunk | None:
        name = method.get("name", "")
        if not name:
            return None

        ret = method.get("return_type") or "void"
        args = []
        arg_dicts = []
        for arg in method.get("arguments", []):
            vtype = arg.get("value_type", "")
            aname = arg.get("name", "")
            default = arg.get("default_value")
            sig = f"{vtype} {aname}" if aname else vtype
            if default:
                sig += f" = {default}"
            args.append(sig)
            arg_dicts.append({
                "name": aname or None,
                "type": vtype,
                "default": default,
            })

        signature = f"{ret} {name}({', '.join(args)})"
        qualifiers = []
        if method.get("is_static"):
            qualifiers.append("static")
        if method.get("is_virtual"):
            qualifiers.append("virtual")
        if method.get("is_const_this"):
            qualifiers.append("const")
        if method.get("is_override"):
            qualifiers.append("override")

        description = method.get("doc_comments", "") or ""
        content = f"```cpp\n{signature}\n```"
        if qualifiers:
            content += f"\n\n**Qualifiers:** {', '.join(qualifiers)}"
        if description:
            content += f"\n\n{description}"

        return Chunk(
            document_id=doc.id,
            source_name=self._source_name,
            source_version=self._source_version,
            chunk_type=ChunkType.METHOD,
            qualified_name=f"{class_name}::{name}",
            parent_name=class_name,
            heading=name,
            summary=_first_sentence(description) or signature,
            content=content,
            metadata={
                "return_type": ret,
                "params": arg_dicts,
                "visibility": method.get("visibility", "Public").lower(),
                "static": method.get("is_static", False) or None,
                "virtual": method.get("is_virtual", False) or None,
            },
            token_count=_estimate_tokens(content),
            chunk_index=idx,
        )

    def _chunk_property(
        self, doc: Document, class_name: str, prop: dict, idx: int
    ) -> Chunk | None:
        name = prop.get("name", "")
        if not name:
            return None

        vtype = prop.get("value_type", "")
        sig = f"{vtype} {name}"
        array = prop.get("array", "None")
        if array == "Unsized":
            sig += "[]"
        elif isinstance(array, dict) and "Sized" in array:
            sig += f"[{array['Sized']}]"

        description = prop.get("doc_comments", "") or ""
        content = f"`{sig}`"
        if prop.get("default_value"):
            content += f"\n\n**Default:** `{prop['default_value']}`"
        if description:
            content += f"\n\n{description}"

        return Chunk(
            document_id=doc.id,
            source_name=self._source_name,
            source_version=self._source_version,
            chunk_type=ChunkType.PROPERTY,
            qualified_name=f"{class_name}::{name}",
            parent_name=class_name,
            heading=name,
            summary=_first_sentence(description) or sig,
            content=content,
            metadata={
                "type": vtype,
                "default": prop.get("default_value"),
                "visibility": prop.get("visibility", "Public").lower(),
                "static": prop.get("is_static", False) or None,
            },
            token_count=_estimate_tokens(content),
            chunk_index=idx,
        )

    def _chunk_enum(self, document: Document, item: dict) -> list[Chunk]:
        name = item["name"]
        description = item.get("doc_comments", "") or ""
        variants = item.get("variants", [])

        variant_lines = "\n".join(f"  {v}" for v in variants)
        content = f"```cpp\nenum class {name} : uint8 {{\n{variant_lines}\n}};\n```"
        if description:
            content += f"\n\n{description}"

        return [Chunk(
            document_id=document.id,
            source_name=self._source_name,
            source_version=self._source_version,
            chunk_type=ChunkType.ENUM,
            qualified_name=name,
            parent_name=name,
            heading=name,
            summary=_first_sentence(description) or f"Enum {name}",
            content=content,
            metadata={"variants": variants},
            token_count=_estimate_tokens(content),
            chunk_index=0,
        )]

    def _chunk_global_functions(self, document: Document) -> list[Chunk]:
        self._ensure_loaded()
        chunks: list[Chunk] = []
        idx = 0

        for func in (self._data or {}).get("functions", []):
            chunk = self._chunk_method(document, "Global", func, idx)
            if chunk:
                chunk.qualified_name = func.get("name", "")
                chunk.parent_name = "Global"
                chunks.append(chunk)
                idx += 1

        if not chunks:
            return [self._fallback_chunk(document)]
        return chunks

    def _fallback_chunk(self, document: Document) -> Chunk:
        return Chunk(
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
        )


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

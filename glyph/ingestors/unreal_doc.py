from __future__ import annotations

import hashlib
import json
import logging
import uuid
from pathlib import Path

from glyph.domain.models import DocType, Document

logger = logging.getLogger(__name__)


class UnrealDocIngestor:
    """Reads documentation.json produced by the unreal-doc tool.

    See https://github.com/PsichiX/unreal-doc for the tool that generates
    structured JSON from Unreal Engine C++ headers.
    """

    def __init__(self, path: str, source_id: uuid.UUID):
        self._path = Path(path)
        self._source_id = source_id

    async def ingest(self) -> list[Document]:
        if not self._path.is_file():
            raise FileNotFoundError(f"unreal-doc JSON not found: {self._path}")

        with open(self._path, encoding="utf-8") as f:
            data = json.load(f)

        documents: list[Document] = []

        for cls in data.get("classes", []):
            doc = self._build_struct_class_doc(cls)
            if doc:
                documents.append(doc)

        for struct in data.get("structs", []):
            doc = self._build_struct_class_doc(struct)
            if doc:
                documents.append(doc)

        for enum in data.get("enums", []):
            doc = self._build_enum_doc(enum)
            if doc:
                documents.append(doc)

        # Group top-level functions into a single document
        functions = data.get("functions", [])
        if functions:
            doc = self._build_functions_doc(functions)
            if doc:
                documents.append(doc)

        logger.info(f"Parsed {len(documents)} documents from {self._path}")
        return documents

    def _build_struct_class_doc(self, item: dict) -> Document | None:
        name = item.get("name", "")
        if not name:
            return None

        mode = item.get("mode", "Struct")
        parts = [f"{'Class' if mode == 'Class' else 'Struct'}: {name}"]

        inherits = item.get("inherits", [])
        if inherits:
            parents = [pair[1] for pair in inherits if len(pair) == 2]
            if parents:
                parts.append(f"Inherits: {', '.join(parents)}")

        if item.get("template"):
            parts.append(f"Template: {item['template']}")

        if item.get("api"):
            parts.append(f"API: {item['api']}")

        doc_comments = item.get("doc_comments")
        if doc_comments:
            parts.append(f"Description: {doc_comments}")

        for method in item.get("methods", []):
            sig = self._format_function_sig(method)
            parts.append(f"Method: {sig}")
            if method.get("doc_comments"):
                parts.append(f"  {method['doc_comments']}")

        for prop in item.get("properties", []):
            sig = self._format_property_sig(prop)
            parts.append(f"Property: {sig}")
            if prop.get("doc_comments"):
                parts.append(f"  {prop['doc_comments']}")

        raw_content = "\n".join(parts)
        content_hash = hashlib.md5(raw_content.encode()).hexdigest()

        return Document(
            source_id=self._source_id,
            path=f"{self._path}#{name}",
            title=name,
            doc_type=DocType.CLASS_REF,
            raw_content=raw_content,
            content_hash=content_hash,
        )

    def _build_enum_doc(self, item: dict) -> Document | None:
        name = item.get("name", "")
        if not name:
            return None

        parts = [f"Enum: {name}"]

        doc_comments = item.get("doc_comments")
        if doc_comments:
            parts.append(f"Description: {doc_comments}")

        for variant in item.get("variants", []):
            parts.append(f"  {variant}")

        raw_content = "\n".join(parts)
        content_hash = hashlib.md5(raw_content.encode()).hexdigest()

        return Document(
            source_id=self._source_id,
            path=f"{self._path}#{name}",
            title=name,
            doc_type=DocType.CLASS_REF,
            raw_content=raw_content,
            content_hash=content_hash,
        )

    def _build_functions_doc(self, functions: list[dict]) -> Document | None:
        parts = ["Global Functions"]
        for func in functions:
            sig = self._format_function_sig(func)
            parts.append(f"Function: {sig}")
            if func.get("doc_comments"):
                parts.append(f"  {func['doc_comments']}")

        raw_content = "\n".join(parts)
        content_hash = hashlib.md5(raw_content.encode()).hexdigest()

        return Document(
            source_id=self._source_id,
            path=f"{self._path}#GlobalFunctions",
            title="Global Functions",
            doc_type=DocType.API_OVERVIEW,
            raw_content=raw_content,
            content_hash=content_hash,
        )

    @staticmethod
    def _format_function_sig(func: dict) -> str:
        name = func.get("name", "")
        ret = func.get("return_type") or "void"
        args = []
        for arg in func.get("arguments", []):
            a = arg.get("value_type", "")
            if arg.get("name"):
                a += f" {arg['name']}"
            if arg.get("default_value"):
                a += f" = {arg['default_value']}"
            args.append(a)
        sig = f"{name}({', '.join(args)}) -> {ret}"
        qualifiers = []
        if func.get("is_static"):
            qualifiers.append("static")
        if func.get("is_virtual"):
            qualifiers.append("virtual")
        if func.get("is_const_this"):
            qualifiers.append("const")
        if func.get("is_override"):
            qualifiers.append("override")
        if qualifiers:
            sig += f" [{', '.join(qualifiers)}]"
        return sig

    @staticmethod
    def _format_property_sig(prop: dict) -> str:
        name = prop.get("name", "")
        vtype = prop.get("value_type", "")
        sig = f"{name}: {vtype}"
        array = prop.get("array", "None")
        if array == "Unsized":
            sig += "[]"
        elif isinstance(array, dict) and "Sized" in array:
            sig += f"[{array['Sized']}]"
        if prop.get("default_value"):
            sig += f" = {prop['default_value']}"
        return sig

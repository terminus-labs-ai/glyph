from __future__ import annotations

import logging

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

from glyph.chunkers._parsers import Symbol
from glyph.domain.models import ChunkType

logger = logging.getLogger(__name__)

PY_LANGUAGE = Language(tspython.language())


class PythonParser:
    """Extracts classes, methods, and functions from Python source using tree-sitter."""

    def parse(self, source: str, *, include_bodies: bool = False) -> list[Symbol]:
        parser = Parser(PY_LANGUAGE)
        src = source.encode("utf-8")
        tree = parser.parse(src)
        symbols: list[Symbol] = []

        for node in tree.root_node.children:
            if node.type == "class_definition":
                symbols.extend(self._parse_class(src, node, include_bodies))
            elif node.type == "function_definition":
                sym = self._parse_function(src, node, None, include_bodies)
                if sym:
                    symbols.append(sym)

        return symbols

    def _parse_class(
        self, src: bytes, node, include_bodies: bool
    ) -> list[Symbol]:
        symbols = []
        class_name = _child_text(src, node, "name")
        if not class_name:
            return symbols

        # Superclasses
        bases = []
        arg_list = _child_by_type(node, "argument_list")
        if arg_list:
            bases = [_slice(src, c) for c in arg_list.children
                     if c.type not in ("(", ")", ",")]

        # Class docstring
        body = _child_by_type(node, "block")
        docstring = _extract_docstring(src, body) if body else ""

        # Build class overview content
        sig = f"class {class_name}"
        if bases:
            sig += f"({', '.join(bases)})"

        content = f"```python\n{sig}\n```"
        if docstring:
            content += f"\n\n{docstring}"

        symbols.append(Symbol(
            name=class_name,
            chunk_type=ChunkType.CLASS_OVERVIEW,
            content=content,
            summary=_first_line(docstring) or f"Class {class_name}",
            metadata={"bases": bases} if bases else {},
        ))

        # Methods
        if body:
            for child in body.children:
                if child.type == "function_definition":
                    sym = self._parse_function(src, child, class_name, include_bodies)
                    if sym:
                        symbols.append(sym)
                elif child.type == "decorated_definition":
                    func = _child_by_type(child, "function_definition")
                    if func:
                        decorators = [_slice(src, d)
                                      for d in child.children if d.type == "decorator"]
                        sym = self._parse_function(src, func, class_name, include_bodies, decorators)
                        if sym:
                            symbols.append(sym)

        return symbols

    def _parse_function(
        self,
        src: bytes,
        node,
        parent_class: str | None,
        include_bodies: bool,
        decorators: list[str] | None = None,
    ) -> Symbol | None:
        name = _child_text(src, node, "name")
        if not name:
            return None

        # Parameters
        params_node = _child_by_type(node, "parameters")
        params_text = _slice(src, params_node) if params_node else "()"

        # Return type
        ret_type = None
        ret_node = _child_by_type(node, "type")
        if ret_node:
            ret_type = _slice(src, ret_node)

        # Docstring
        body = _child_by_type(node, "block")
        docstring = _extract_docstring(src, body) if body else ""

        # Build signature
        sig = f"def {name}{params_text}"
        if ret_type:
            sig += f" -> {ret_type}"

        # Build content
        if include_bodies and body:
            full_text = _slice(src, node)
            if decorators:
                full_text = "\n".join(decorators) + "\n" + full_text
            content = f"```python\n{full_text}\n```"
        else:
            prefix = "\n".join(decorators) + "\n" if decorators else ""
            content = f"```python\n{prefix}{sig}:\n```"
            if docstring:
                content += f"\n\n{docstring}"

        chunk_type = ChunkType.METHOD

        metadata = {}
        if ret_type:
            metadata["return_type"] = ret_type
        if decorators:
            metadata["decorators"] = decorators

        return Symbol(
            name=name,
            chunk_type=chunk_type,
            content=content,
            summary=_first_line(docstring) or sig,
            parent=parent_class,
            metadata=metadata,
        )


def _child_by_type(node, type_name: str):
    for child in node.children:
        if child.type == type_name:
            return child
    return None


def _slice(src: bytes, node) -> str:
    """Decode a node's byte span from the UTF-8 source."""
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _child_text(src: bytes, node, field_name: str) -> str | None:
    child = node.child_by_field_name(field_name)
    if child:
        return _slice(src, child)
    return None


def _extract_docstring(src: bytes, body_node) -> str:
    if not body_node or not body_node.children:
        return ""
    first_stmt = body_node.children[0]
    if first_stmt.type == "expression_statement":
        expr = first_stmt.children[0] if first_stmt.children else None
        if expr and expr.type in ("string", "concatenated_string"):
            raw = _slice(src, expr)
            # Strip triple quotes
            for q in ('"""', "'''"):
                if raw.startswith(q) and raw.endswith(q):
                    return raw[3:-3].strip()
            # Strip single quotes
            for q in ('"', "'"):
                if raw.startswith(q) and raw.endswith(q):
                    return raw[1:-1].strip()
            return raw
    return ""


def _first_line(text: str) -> str:
    if not text:
        return ""
    line = text.split("\n")[0].strip()
    return line[:200]

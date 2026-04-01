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
        tree = parser.parse(source.encode())
        symbols: list[Symbol] = []

        for node in tree.root_node.children:
            if node.type == "class_definition":
                symbols.extend(self._parse_class(source, node, include_bodies))
            elif node.type == "function_definition":
                sym = self._parse_function(source, node, None, include_bodies)
                if sym:
                    symbols.append(sym)

        return symbols

    def _parse_class(
        self, source: str, node, include_bodies: bool
    ) -> list[Symbol]:
        symbols = []
        class_name = _child_text(source, node, "name")
        if not class_name:
            return symbols

        # Superclasses
        bases = []
        arg_list = _child_by_type(node, "argument_list")
        if arg_list:
            bases = [source[c.start_byte:c.end_byte] for c in arg_list.children
                     if c.type not in ("(", ")", ",")]

        # Class docstring
        body = _child_by_type(node, "block")
        docstring = _extract_docstring(source, body) if body else ""

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
                    sym = self._parse_function(source, child, class_name, include_bodies)
                    if sym:
                        symbols.append(sym)
                elif child.type == "decorated_definition":
                    func = _child_by_type(child, "function_definition")
                    if func:
                        decorators = [source[d.start_byte:d.end_byte]
                                      for d in child.children if d.type == "decorator"]
                        sym = self._parse_function(source, func, class_name, include_bodies, decorators)
                        if sym:
                            symbols.append(sym)

        return symbols

    def _parse_function(
        self,
        source: str,
        node,
        parent_class: str | None,
        include_bodies: bool,
        decorators: list[str] | None = None,
    ) -> Symbol | None:
        name = _child_text(source, node, "name")
        if not name:
            return None

        # Parameters
        params_node = _child_by_type(node, "parameters")
        params_text = source[params_node.start_byte:params_node.end_byte] if params_node else "()"

        # Return type
        ret_type = None
        ret_node = _child_by_type(node, "type")
        if ret_node:
            ret_type = source[ret_node.start_byte:ret_node.end_byte]

        # Docstring
        body = _child_by_type(node, "block")
        docstring = _extract_docstring(source, body) if body else ""

        # Build signature
        sig = f"def {name}{params_text}"
        if ret_type:
            sig += f" -> {ret_type}"

        # Build content
        if include_bodies and body:
            full_text = source[node.start_byte:node.end_byte]
            if decorators:
                full_text = "\n".join(decorators) + "\n" + full_text
            content = f"```python\n{full_text}\n```"
        else:
            prefix = "\n".join(decorators) + "\n" if decorators else ""
            content = f"```python\n{prefix}{sig}:\n```"
            if docstring:
                content += f"\n\n{docstring}"

        # Determine chunk type
        chunk_type = ChunkType.METHOD if parent_class else ChunkType.METHOD

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


def _child_text(source: str, node, field_name: str) -> str | None:
    child = node.child_by_field_name(field_name)
    if child:
        return source[child.start_byte:child.end_byte]
    return None


def _extract_docstring(source: str, body_node) -> str:
    if not body_node or not body_node.children:
        return ""
    first_stmt = body_node.children[0]
    if first_stmt.type == "expression_statement":
        expr = first_stmt.children[0] if first_stmt.children else None
        if expr and expr.type in ("string", "concatenated_string"):
            raw = source[expr.start_byte:expr.end_byte]
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

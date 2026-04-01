from __future__ import annotations

import logging
import re

import tree_sitter_typescript as tstypescript
from tree_sitter import Language, Parser, Node

from glyph.chunkers._parsers import Symbol
from glyph.domain.models import ChunkType

logger = logging.getLogger(__name__)

TS_LANGUAGE = Language(tstypescript.language_typescript())
TSX_LANGUAGE = Language(tstypescript.language_tsx())


class TypeScriptParser:
    """Extracts classes, functions, interfaces, enums, and type aliases from TypeScript using tree-sitter."""

    def __init__(self, *, tsx: bool = False) -> None:
        self._language = TSX_LANGUAGE if tsx else TS_LANGUAGE

    def parse(self, source: str, *, include_bodies: bool = False) -> list[Symbol]:
        parser = Parser(self._language)
        tree = parser.parse(source.encode())
        symbols: list[Symbol] = []

        nodes = tree.root_node.children
        for i, node in enumerate(nodes):
            prev_comment = _get_preceding_comment(source, nodes, i)
            self._process_node(source, node, symbols, include_bodies, prev_comment)

        return symbols

    def _process_node(
        self,
        source: str,
        node: Node,
        symbols: list[Symbol],
        include_bodies: bool,
        prev_comment: str,
        is_exported: bool = False,
        is_default: bool = False,
    ) -> None:
        if node.type == "export_statement":
            exported = True
            default = any(c.type == "default" for c in node.children)
            for child in node.children:
                if child.type in (
                    "class_declaration", "abstract_class_declaration",
                    "function_declaration", "lexical_declaration",
                    "interface_declaration", "enum_declaration",
                    "type_alias_declaration",
                ):
                    inner_comment = _get_jsdoc_from_export(source, node) or prev_comment
                    self._process_node(source, child, symbols, include_bodies, inner_comment, exported, default)
            return

        if node.type in ("class_declaration", "abstract_class_declaration"):
            symbols.extend(self._parse_class(source, node, include_bodies, prev_comment, is_exported, is_default))
        elif node.type == "function_declaration":
            sym = self._parse_function(source, node, None, include_bodies, prev_comment, is_exported)
            if sym:
                symbols.append(sym)
        elif node.type == "lexical_declaration":
            sym = self._parse_arrow_function(source, node, include_bodies, prev_comment, is_exported)
            if sym:
                symbols.append(sym)
        elif node.type == "interface_declaration":
            sym = self._parse_interface(source, node, include_bodies, prev_comment, is_exported)
            if sym:
                symbols.append(sym)
        elif node.type == "enum_declaration":
            sym = self._parse_enum(source, node, include_bodies, prev_comment, is_exported)
            if sym:
                symbols.append(sym)
        elif node.type == "type_alias_declaration":
            sym = self._parse_type_alias(source, node, prev_comment, is_exported)
            if sym:
                symbols.append(sym)

    # ── Class ──────────────────────────────────────────────────────────

    def _parse_class(
        self, source: str, node: Node, include_bodies: bool,
        docstring: str, is_exported: bool, is_default: bool,
    ) -> list[Symbol]:
        symbols: list[Symbol] = []
        name = _child_text(source, node, "name")
        if not name:
            return symbols

        is_abstract = node.type == "abstract_class_declaration"
        bases: list[str] = []
        implements: list[str] = []

        heritage = _child_by_type(node, "class_heritage")
        if heritage:
            for clause in heritage.children:
                if clause.type == "extends_clause":
                    for c in clause.children:
                        if c.type in ("identifier", "type_identifier", "generic_type"):
                            bases.append(source[c.start_byte:c.end_byte])
                elif clause.type == "implements_clause":
                    for c in clause.children:
                        if c.type in ("identifier", "type_identifier", "generic_type"):
                            implements.append(source[c.start_byte:c.end_byte])

        type_params = _extract_type_parameters(source, node)

        sig = f"class {name}"
        if type_params:
            sig += f"<{type_params}>"
        if bases:
            sig += f" extends {', '.join(bases)}"
        if implements:
            sig += f" implements {', '.join(implements)}"
        if is_abstract:
            sig = "abstract " + sig

        doc_text = _parse_jsdoc_text(docstring)
        content = f"```typescript\n{sig}\n```"
        if doc_text:
            content += f"\n\n{doc_text}"

        metadata: dict = {}
        if bases:
            metadata["bases"] = bases
        if implements:
            metadata["implements"] = implements
        if is_exported:
            metadata["is_exported"] = True
        if is_default:
            metadata["is_default"] = True
        if is_abstract:
            metadata["is_abstract"] = True
        if type_params:
            metadata["type_parameters"] = type_params

        symbols.append(Symbol(
            name=name,
            chunk_type=ChunkType.CLASS_OVERVIEW,
            content=content,
            summary=_first_line(doc_text) or f"Class {name}",
            metadata=metadata,
        ))

        body = _child_by_type(node, "class_body")
        if body:
            children = body.children
            for i, child in enumerate(children):
                comment = _get_preceding_comment(source, children, i)
                if child.type == "method_definition":
                    sym = self._parse_method(source, child, name, include_bodies, comment)
                    if sym:
                        symbols.append(sym)
                elif child.type == "abstract_method_signature":
                    sym = self._parse_abstract_method(source, child, name, comment)
                    if sym:
                        symbols.append(sym)

        return symbols

    def _parse_method(
        self, source: str, node: Node, class_name: str,
        include_bodies: bool, docstring: str,
    ) -> Symbol | None:
        name = _child_text(source, node, "name")
        if not name:
            return None

        visibility = None
        is_static = False
        is_async = False
        decorators: list[str] = []

        for child in node.children:
            if child.type == "accessibility_modifier":
                visibility = source[child.start_byte:child.end_byte]
            elif child.type == "static":
                is_static = True
            elif child.type == "async":
                is_async = True
            elif child.type == "decorator":
                decorators.append(source[child.start_byte:child.end_byte])

        params_node = _child_by_type(node, "formal_parameters")
        params_text = source[params_node.start_byte:params_node.end_byte] if params_node else "()"

        ret_type = _extract_return_type(source, node)
        type_params = _extract_type_parameters(source, node)

        sig_parts: list[str] = []
        if visibility:
            sig_parts.append(visibility)
        if is_static:
            sig_parts.append("static")
        if is_async:
            sig_parts.append("async")
        sig_parts.append(name)
        if type_params:
            sig_parts[-1] += f"<{type_params}>"
        sig = " ".join(sig_parts) + params_text
        if ret_type:
            sig += f": {ret_type}"

        doc_text = _parse_jsdoc_text(docstring)
        jsdoc_meta = _parse_jsdoc_tags(docstring)

        if include_bodies:
            full_text = source[node.start_byte:node.end_byte]
            content = f"```typescript\n{full_text}\n```"
        else:
            content = f"```typescript\n{sig}\n```"
            if doc_text:
                content += f"\n\n{doc_text}"

        metadata: dict = {}
        if ret_type:
            metadata["return_type"] = ret_type
        if visibility:
            metadata["visibility"] = visibility
        if is_static:
            metadata["is_static"] = True
        if is_async:
            metadata["is_async"] = True
        if decorators:
            metadata["decorators"] = decorators
        if type_params:
            metadata["type_parameters"] = type_params
        metadata.update(jsdoc_meta)

        return Symbol(
            name=name,
            chunk_type=ChunkType.METHOD,
            content=content,
            summary=_first_line(doc_text) or sig,
            parent=class_name,
            metadata=metadata,
        )

    def _parse_abstract_method(
        self, source: str, node: Node, class_name: str, docstring: str,
    ) -> Symbol | None:
        name = _child_text(source, node, "name")
        if not name:
            return None

        visibility = None
        for child in node.children:
            if child.type == "accessibility_modifier":
                visibility = source[child.start_byte:child.end_byte]

        params_node = _child_by_type(node, "formal_parameters")
        params_text = source[params_node.start_byte:params_node.end_byte] if params_node else "()"
        ret_type = _extract_return_type(source, node)

        sig_parts = ["abstract"]
        if visibility:
            sig_parts.append(visibility)
        sig_parts.append(f"{name}{params_text}")
        if ret_type:
            sig_parts[-1] += f": {ret_type}"
        sig = " ".join(sig_parts)

        doc_text = _parse_jsdoc_text(docstring)
        content = f"```typescript\n{sig}\n```"
        if doc_text:
            content += f"\n\n{doc_text}"

        metadata: dict = {"is_abstract": True}
        if visibility:
            metadata["visibility"] = visibility
        if ret_type:
            metadata["return_type"] = ret_type

        return Symbol(
            name=name,
            chunk_type=ChunkType.METHOD,
            content=content,
            summary=_first_line(doc_text) or sig,
            parent=class_name,
            metadata=metadata,
        )

    # ── Standalone function ────────────────────────────────────────────

    def _parse_function(
        self, source: str, node: Node, parent: str | None,
        include_bodies: bool, docstring: str, is_exported: bool = False,
    ) -> Symbol | None:
        name = _child_text(source, node, "name")
        if not name:
            return None

        is_async = any(c.type == "async" for c in node.children)
        params_node = _child_by_type(node, "formal_parameters")
        params_text = source[params_node.start_byte:params_node.end_byte] if params_node else "()"
        ret_type = _extract_return_type(source, node)
        type_params = _extract_type_parameters(source, node)

        prefix = "async " if is_async else ""
        sig = f"{prefix}function {name}"
        if type_params:
            sig += f"<{type_params}>"
        sig += params_text
        if ret_type:
            sig += f": {ret_type}"

        doc_text = _parse_jsdoc_text(docstring)
        jsdoc_meta = _parse_jsdoc_tags(docstring)

        if include_bodies:
            full_text = source[node.start_byte:node.end_byte]
            content = f"```typescript\n{full_text}\n```"
        else:
            content = f"```typescript\n{sig}\n```"
            if doc_text:
                content += f"\n\n{doc_text}"

        metadata: dict = {}
        if ret_type:
            metadata["return_type"] = ret_type
        if is_async:
            metadata["is_async"] = True
        if is_exported:
            metadata["is_exported"] = True
        if type_params:
            metadata["type_parameters"] = type_params
        metadata.update(jsdoc_meta)

        return Symbol(
            name=name,
            chunk_type=ChunkType.METHOD,
            content=content,
            summary=_first_line(doc_text) or sig,
            parent=parent,
            metadata=metadata,
        )

    # ── Arrow function (const x = (...) => ...) ───────────────────────

    def _parse_arrow_function(
        self, source: str, node: Node, include_bodies: bool,
        docstring: str, is_exported: bool = False,
    ) -> Symbol | None:
        for child in node.children:
            if child.type != "variable_declarator":
                continue
            name = _child_text(source, child, "name")
            arrow = _child_by_type(child, "arrow_function")
            if not name or not arrow:
                continue

            is_async = any(c.type == "async" for c in arrow.children)
            params_node = _child_by_type(arrow, "formal_parameters")
            params_text = source[params_node.start_byte:params_node.end_byte] if params_node else "()"
            ret_type = _extract_return_type(source, arrow)
            type_params = _extract_type_parameters(source, arrow)

            prefix = "async " if is_async else ""
            sig = f"const {name} = {prefix}"
            if type_params:
                sig += f"<{type_params}>"
            sig += params_text
            if ret_type:
                sig += f": {ret_type}"
            sig += " => ..."

            doc_text = _parse_jsdoc_text(docstring)
            jsdoc_meta = _parse_jsdoc_tags(docstring)

            if include_bodies:
                full_text = source[node.start_byte:node.end_byte]
                content = f"```typescript\n{full_text}\n```"
            else:
                content = f"```typescript\n{sig}\n```"
                if doc_text:
                    content += f"\n\n{doc_text}"

            metadata: dict = {}
            if ret_type:
                metadata["return_type"] = ret_type
            if is_async:
                metadata["is_async"] = True
            if is_exported:
                metadata["is_exported"] = True
            if type_params:
                metadata["type_parameters"] = type_params
            metadata.update(jsdoc_meta)

            return Symbol(
                name=name,
                chunk_type=ChunkType.METHOD,
                content=content,
                summary=_first_line(doc_text) or sig,
                metadata=metadata,
            )
        return None

    # ── Interface ──────────────────────────────────────────────────────

    def _parse_interface(
        self, source: str, node: Node, include_bodies: bool,
        docstring: str, is_exported: bool = False,
    ) -> Symbol | None:
        name = _child_text(source, node, "name")
        if not name:
            return None

        type_params = _extract_type_parameters(source, node)
        extends: list[str] = []
        heritage = _child_by_type(node, "extends_type_clause")
        if heritage:
            for c in heritage.children:
                if c.type in ("type_identifier", "generic_type"):
                    extends.append(source[c.start_byte:c.end_byte])

        sig = f"interface {name}"
        if type_params:
            sig += f"<{type_params}>"
        if extends:
            sig += f" extends {', '.join(extends)}"

        doc_text = _parse_jsdoc_text(docstring)

        if include_bodies:
            full_text = source[node.start_byte:node.end_byte]
            content = f"```typescript\n{full_text}\n```"
        else:
            content = f"```typescript\n{sig}\n```"
            if doc_text:
                content += f"\n\n{doc_text}"

        metadata: dict = {}
        if extends:
            metadata["bases"] = extends
        if is_exported:
            metadata["is_exported"] = True
        if type_params:
            metadata["type_parameters"] = type_params

        return Symbol(
            name=name,
            chunk_type=ChunkType.CLASS_OVERVIEW,
            content=content,
            summary=_first_line(doc_text) or f"Interface {name}",
            metadata=metadata,
        )

    # ── Enum ───────────────────────────────────────────────────────────

    def _parse_enum(
        self, source: str, node: Node, include_bodies: bool,
        docstring: str, is_exported: bool = False,
    ) -> Symbol | None:
        name = _child_text(source, node, "name")
        if not name:
            return None

        members: list[str] = []
        body = _child_by_type(node, "enum_body")
        if body:
            for child in body.children:
                if child.type in ("enum_assignment", "property_identifier"):
                    member_name = _child_text(source, child, "name")
                    if not member_name:
                        member_name = source[child.start_byte:child.end_byte].split("=")[0].strip()
                    members.append(member_name)

        doc_text = _parse_jsdoc_text(docstring)

        if include_bodies:
            full_text = source[node.start_byte:node.end_byte]
            content = f"```typescript\n{full_text}\n```"
        else:
            sig = f"enum {name}"
            content = f"```typescript\n{sig}\n```"
            if members:
                content += f"\n\nMembers: {', '.join(members)}"
            if doc_text:
                content += f"\n\n{doc_text}"

        metadata: dict = {}
        if members:
            metadata["members"] = members
        if is_exported:
            metadata["is_exported"] = True

        return Symbol(
            name=name,
            chunk_type=ChunkType.ENUM,
            content=content,
            summary=_first_line(doc_text) or f"Enum {name}",
            metadata=metadata,
        )

    # ── Type alias ─────────────────────────────────────────────────────

    def _parse_type_alias(
        self, source: str, node: Node, docstring: str, is_exported: bool = False,
    ) -> Symbol | None:
        name = _child_text(source, node, "name")
        if not name:
            return None

        type_params = _extract_type_parameters(source, node)
        full_text = source[node.start_byte:node.end_byte]
        doc_text = _parse_jsdoc_text(docstring)

        content = f"```typescript\n{full_text}\n```"
        if doc_text:
            content += f"\n\n{doc_text}"

        metadata: dict = {}
        if is_exported:
            metadata["is_exported"] = True
        if type_params:
            metadata["type_parameters"] = type_params

        return Symbol(
            name=name,
            chunk_type=ChunkType.CONSTANT,
            content=content,
            summary=_first_line(doc_text) or f"Type {name}",
            metadata=metadata,
        )


# ── Helpers ────────────────────────────────────────────────────────────


def _child_by_type(node: Node, type_name: str) -> Node | None:
    for child in node.children:
        if child.type == type_name:
            return child
    return None


def _child_text(source: str, node: Node, field_name: str) -> str | None:
    child = node.child_by_field_name(field_name)
    if child:
        return source[child.start_byte:child.end_byte]
    return None


def _extract_return_type(source: str, node: Node) -> str | None:
    ann = _child_by_type(node, "type_annotation")
    if ann and len(ann.children) >= 2:
        type_node = ann.children[-1]
        return source[type_node.start_byte:type_node.end_byte]
    return None


def _extract_type_parameters(source: str, node: Node) -> str:
    tp = _child_by_type(node, "type_parameters")
    if tp:
        text = source[tp.start_byte:tp.end_byte]
        return text.strip("<>")
    return ""


def _get_preceding_comment(source: str, siblings: list, index: int) -> str:
    if index <= 0:
        return ""
    prev = siblings[index - 1]
    if prev.type == "comment":
        text = source[prev.start_byte:prev.end_byte]
        if text.strip().startswith("/**"):
            return text
    return ""


def _get_jsdoc_from_export(source: str, export_node: Node) -> str:
    for child in export_node.children:
        if child.type == "comment":
            text = source[child.start_byte:child.end_byte]
            if text.strip().startswith("/**"):
                return text
    return ""


def _parse_jsdoc_text(raw: str) -> str:
    """Extract the description text from a JSDoc comment, stripping tags."""
    if not raw:
        return ""
    # Remove /** and */
    text = raw.strip()
    if text.startswith("/**"):
        text = text[3:]
    if text.endswith("*/"):
        text = text[:-2]
    # Remove leading * on each line
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("* "):
            line = line[2:]
        elif line == "*":
            line = ""
        elif line.startswith("*"):
            line = line[1:]
        # Stop at first @tag
        if line.strip().startswith("@"):
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _parse_jsdoc_tags(raw: str) -> dict:
    """Extract @param, @returns, @deprecated from JSDoc into metadata dict."""
    if not raw:
        return {}
    metadata: dict = {}
    tag_pattern = re.compile(r"@(\w+)\s*(.*)")
    text = raw.strip()
    if text.startswith("/**"):
        text = text[3:]
    if text.endswith("*/"):
        text = text[:-2]
    for line in text.split("\n"):
        line = line.strip().lstrip("* ")
        m = tag_pattern.match(line)
        if m:
            tag, value = m.group(1), m.group(2).strip()
            if tag == "deprecated":
                metadata["is_deprecated"] = True
                if value:
                    metadata["deprecated_message"] = value
            elif tag == "returns" or tag == "return":
                metadata["jsdoc_returns"] = value
            elif tag == "param":
                metadata.setdefault("jsdoc_params", []).append(value)
    return metadata


def _first_line(text: str) -> str:
    if not text:
        return ""
    line = text.split("\n")[0].strip()
    return line[:200]

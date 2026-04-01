from __future__ import annotations

import logging

import tree_sitter_go as tsgo
from tree_sitter import Language, Parser, Node

from ragify.chunkers._parsers import Symbol
from ragify.domain.models import ChunkType

logger = logging.getLogger(__name__)

GO_LANGUAGE = Language(tsgo.language())


class GoParser:
    """Extracts structs, interfaces, functions, methods, and constants from Go using tree-sitter."""

    def parse(self, source: str, *, include_bodies: bool = False) -> list[Symbol]:
        parser = Parser(GO_LANGUAGE)
        tree = parser.parse(source.encode())
        symbols: list[Symbol] = []

        nodes = tree.root_node.children
        for i, node in enumerate(nodes):
            doc = _collect_doc_comments(source, nodes, i)
            self._process_node(source, node, symbols, include_bodies, doc)

        return symbols

    def _process_node(
        self, source: str, node: Node, symbols: list[Symbol],
        include_bodies: bool, doc: str,
    ) -> None:
        if node.type == "type_declaration":
            symbols.extend(self._parse_type_declaration(source, node, include_bodies, doc))
        elif node.type == "function_declaration":
            sym = self._parse_function(source, node, include_bodies, doc)
            if sym:
                symbols.append(sym)
        elif node.type == "method_declaration":
            sym = self._parse_method(source, node, include_bodies, doc)
            if sym:
                symbols.append(sym)
        elif node.type == "const_declaration":
            symbols.extend(self._parse_const_block(source, node, doc))
        elif node.type == "var_declaration":
            symbols.extend(self._parse_var_block(source, node, doc))

    # ── Type declarations ──────────────────────────────────────────────

    def _parse_type_declaration(
        self, source: str, node: Node, include_bodies: bool, doc: str,
    ) -> list[Symbol]:
        symbols: list[Symbol] = []

        for child in node.children:
            if child.type == "type_spec":
                sym = self._parse_type_spec(source, child, include_bodies, doc)
                if sym:
                    symbols.append(sym)
            elif child.type == "type_alias":
                sym = self._parse_type_alias(source, child, doc)
                if sym:
                    symbols.append(sym)

        return symbols

    def _parse_type_spec(
        self, source: str, node: Node, include_bodies: bool, doc: str,
    ) -> Symbol | None:
        name = _child_text(source, node, "name")
        if not name:
            return None

        is_exported = name[0].isupper()

        # Determine what kind of type this is
        type_node = _child_by_type(node, "struct_type")
        if type_node:
            return self._parse_struct(source, node, type_node, name, is_exported, include_bodies, doc)

        iface_node = _child_by_type(node, "interface_type")
        if iface_node:
            return self._parse_interface(source, node, iface_node, name, is_exported, include_bodies, doc)

        # Named type (type X Y)
        type_params = _extract_type_parameters(source, node)
        full = source[node.start_byte:node.end_byte]
        content = f"```go\ntype {full}\n```"
        if doc:
            content += f"\n\n{doc}"

        metadata: dict = {"is_exported": is_exported}
        if type_params:
            metadata["type_parameters"] = type_params

        return Symbol(
            name=name,
            chunk_type=ChunkType.CONSTANT,
            content=content,
            summary=_first_line(doc) or f"Type {name}",
            metadata=metadata,
        )

    def _parse_struct(
        self, source: str, spec_node: Node, struct_node: Node,
        name: str, is_exported: bool, include_bodies: bool, doc: str,
    ) -> Symbol:
        fields: list[str] = []
        field_list = _child_by_type(struct_node, "field_declaration_list")
        if field_list:
            for f in field_list.children:
                if f.type == "field_declaration":
                    fname = _child_text(source, f, "name")
                    ftype = _child_text(source, f, "type")
                    if fname and ftype:
                        fields.append(f"{fname} {ftype}")
                    elif ftype:
                        # Embedded type
                        fields.append(ftype)

        type_params = _extract_type_parameters(source, spec_node)

        sig = f"type {name}"
        if type_params:
            sig += type_params
        sig += " struct"

        if include_bodies:
            full = source[spec_node.start_byte:spec_node.end_byte]
            content = f"```go\ntype {full}\n```"
        else:
            content = f"```go\n{sig}\n```"
            if fields:
                content += f"\n\nFields: {', '.join(fields)}"
        if doc:
            content += f"\n\n{doc}"

        metadata: dict = {"is_exported": is_exported}
        if fields:
            metadata["fields"] = fields
        if type_params:
            metadata["type_parameters"] = type_params

        return Symbol(
            name=name,
            chunk_type=ChunkType.CLASS_OVERVIEW,
            content=content,
            summary=_first_line(doc) or f"Struct {name}",
            metadata=metadata,
        )

    def _parse_interface(
        self, source: str, spec_node: Node, iface_node: Node,
        name: str, is_exported: bool, include_bodies: bool, doc: str,
    ) -> Symbol:
        methods: list[str] = []
        for child in iface_node.children:
            if child.type == "method_elem":
                mname = _child_text(source, child, "name")
                if mname:
                    methods.append(mname)

        type_params = _extract_type_parameters(source, spec_node)

        sig = f"type {name}"
        if type_params:
            sig += type_params
        sig += " interface"

        if include_bodies:
            full = source[spec_node.start_byte:spec_node.end_byte]
            content = f"```go\ntype {full}\n```"
        else:
            content = f"```go\n{sig}\n```"
            if methods:
                content += f"\n\nMethods: {', '.join(methods)}"
        if doc:
            content += f"\n\n{doc}"

        metadata: dict = {"is_exported": is_exported}
        if methods:
            metadata["methods"] = methods
        if type_params:
            metadata["type_parameters"] = type_params

        return Symbol(
            name=name,
            chunk_type=ChunkType.CLASS_OVERVIEW,
            content=content,
            summary=_first_line(doc) or f"Interface {name}",
            metadata=metadata,
        )

    def _parse_type_alias(
        self, source: str, node: Node, doc: str,
    ) -> Symbol | None:
        name = _child_text(source, node, "name")
        if not name:
            return None

        is_exported = name[0].isupper()
        full = source[node.start_byte:node.end_byte]

        content = f"```go\ntype {full}\n```"
        if doc:
            content += f"\n\n{doc}"

        return Symbol(
            name=name,
            chunk_type=ChunkType.CONSTANT,
            content=content,
            summary=_first_line(doc) or f"Type alias {name}",
            metadata={"is_exported": is_exported},
        )

    # ── Function ───────────────────────────────────────────────────────

    def _parse_function(
        self, source: str, node: Node, include_bodies: bool, doc: str,
    ) -> Symbol | None:
        name = _child_text(source, node, "name")
        if not name:
            return None

        is_exported = name[0].isupper()
        params = _child_by_type(node, "parameter_list")
        params_text = source[params.start_byte:params.end_byte] if params else "()"
        ret_type = _extract_return_type(source, node)
        type_params = _extract_type_parameters(source, node)

        sig = f"func {name}"
        if type_params:
            sig += type_params
        sig += params_text
        if ret_type:
            sig += f" {ret_type}"

        if include_bodies:
            full = source[node.start_byte:node.end_byte]
            content = f"```go\n{full}\n```"
        else:
            content = f"```go\n{sig}\n```"
            if doc:
                content += f"\n\n{doc}"

        metadata: dict = {"is_exported": is_exported}
        if ret_type:
            metadata["return_type"] = ret_type
        if type_params:
            metadata["type_parameters"] = type_params

        return Symbol(
            name=name,
            chunk_type=ChunkType.METHOD,
            content=content,
            summary=_first_line(doc) or sig,
            metadata=metadata,
        )

    # ── Method (with receiver) ─────────────────────────────────────────

    def _parse_method(
        self, source: str, node: Node, include_bodies: bool, doc: str,
    ) -> Symbol | None:
        name = _child_text(source, node, "name")
        if not name:
            return None

        is_exported = name[0].isupper()

        # Extract receiver
        receiver_type = ""
        is_pointer_receiver = False
        receiver_params = node.children
        for child in receiver_params:
            if child.type == "parameter_list":
                # First parameter_list is the receiver
                for param in child.children:
                    if param.type == "parameter_declaration":
                        for pchild in param.children:
                            if pchild.type == "pointer_type":
                                is_pointer_receiver = True
                                receiver_type = source[pchild.start_byte:pchild.end_byte].lstrip("*")
                            elif pchild.type == "type_identifier":
                                if not receiver_type:
                                    receiver_type = source[pchild.start_byte:pchild.end_byte]
                break  # Only process first parameter_list (the receiver)

        parent = receiver_type or None

        # Get the actual parameter list (second one)
        param_lists = [c for c in node.children if c.type == "parameter_list"]
        params_text = "()"
        if len(param_lists) >= 2:
            params_text = source[param_lists[1].start_byte:param_lists[1].end_byte]

        ret_type = _extract_return_type(source, node, skip_params=2)

        recv = f"*{receiver_type}" if is_pointer_receiver else receiver_type
        sig = f"func ({recv}) {name}{params_text}"
        if ret_type:
            sig += f" {ret_type}"

        if include_bodies:
            full = source[node.start_byte:node.end_byte]
            content = f"```go\n{full}\n```"
        else:
            content = f"```go\n{sig}\n```"
            if doc:
                content += f"\n\n{doc}"

        metadata: dict = {
            "is_exported": is_exported,
            "receiver_type": receiver_type,
            "is_pointer_receiver": is_pointer_receiver,
        }
        if ret_type:
            metadata["return_type"] = ret_type

        return Symbol(
            name=name,
            chunk_type=ChunkType.METHOD,
            content=content,
            summary=_first_line(doc) or sig,
            parent=parent,
            metadata=metadata,
        )

    # ── Const block ────────────────────────────────────────────────────

    def _parse_const_block(
        self, source: str, node: Node, doc: str,
    ) -> list[Symbol]:
        symbols: list[Symbol] = []
        for child in node.children:
            if child.type == "const_spec":
                sym = self._parse_const_spec(source, child, doc)
                if sym:
                    symbols.append(sym)
                    doc = ""  # Only first const gets the block doc
        return symbols

    def _parse_const_spec(self, source: str, node: Node, doc: str) -> Symbol | None:
        name = _child_text(source, node, "name")
        if not name:
            # Try to find identifier directly
            for child in node.children:
                if child.type == "identifier":
                    name = source[child.start_byte:child.end_byte]
                    break
        if not name:
            return None

        is_exported = name[0].isupper()
        full = source[node.start_byte:node.end_byte]

        content = f"```go\nconst {full}\n```"
        if doc:
            content += f"\n\n{doc}"

        return Symbol(
            name=name,
            chunk_type=ChunkType.CONSTANT,
            content=content,
            summary=_first_line(doc) or f"Constant {name}",
            metadata={"is_exported": is_exported},
        )

    # ── Var block ──────────────────────────────────────────────────────

    def _parse_var_block(
        self, source: str, node: Node, doc: str,
    ) -> list[Symbol]:
        symbols: list[Symbol] = []
        for child in node.children:
            if child.type == "var_spec":
                sym = self._parse_var_spec(source, child, doc)
                if sym:
                    symbols.append(sym)
                    doc = ""
            elif child.type == "var_spec_list":
                for spec in child.children:
                    if spec.type == "var_spec":
                        sym = self._parse_var_spec(source, spec, doc)
                        if sym:
                            symbols.append(sym)
                            doc = ""
        return symbols

    def _parse_var_spec(self, source: str, node: Node, doc: str) -> Symbol | None:
        name = _child_text(source, node, "name")
        if not name:
            for child in node.children:
                if child.type == "identifier":
                    name = source[child.start_byte:child.end_byte]
                    break
        if not name:
            return None

        is_exported = name[0].isupper()
        full = source[node.start_byte:node.end_byte]

        content = f"```go\nvar {full}\n```"
        if doc:
            content += f"\n\n{doc}"

        return Symbol(
            name=name,
            chunk_type=ChunkType.PROPERTY,
            content=content,
            summary=_first_line(doc) or f"Variable {name}",
            metadata={"is_exported": is_exported},
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


def _extract_type_parameters(source: str, node: Node) -> str:
    tp = _child_by_type(node, "type_parameter_list")
    if tp:
        return source[tp.start_byte:tp.end_byte]
    return ""


def _extract_return_type(source: str, node: Node, *, skip_params: int = 1) -> str | None:
    """Extract return type(s) from a Go function/method.

    For functions: skip the first parameter_list (params).
    For methods: skip the first two parameter_lists (receiver + params).
    The next parameter_list or type_identifier is the return type.
    """
    param_count = 0
    for child in node.children:
        if child.type == "parameter_list":
            param_count += 1
            if param_count > skip_params:
                # This is the return type tuple
                return source[child.start_byte:child.end_byte]
        elif child.type in ("type_identifier", "pointer_type", "array_type",
                            "slice_type", "map_type", "qualified_type",
                            "generic_type", "interface_type"):
            if param_count >= skip_params:
                return source[child.start_byte:child.end_byte]
    return None


def _collect_doc_comments(source: str, siblings: list, index: int) -> str:
    """Collect consecutive // comments preceding a node."""
    lines: list[str] = []
    i = index - 1
    while i >= 0:
        prev = siblings[i]
        if prev.type == "comment":
            text = source[prev.start_byte:prev.end_byte]
            if text.startswith("//"):
                lines.append(text[2:].strip())
                i -= 1
                continue
        break
    if not lines:
        return ""
    lines.reverse()
    return "\n".join(lines)


def _first_line(text: str) -> str:
    if not text:
        return ""
    line = text.split("\n")[0].strip()
    return line[:200]

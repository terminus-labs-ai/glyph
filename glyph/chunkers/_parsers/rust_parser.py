from __future__ import annotations

import logging

import tree_sitter_rust as tsrust
from tree_sitter import Language, Parser, Node

from glyph.chunkers._parsers import Symbol
from glyph.domain.models import ChunkType

logger = logging.getLogger(__name__)

RUST_LANGUAGE = Language(tsrust.language())


class RustParser:
    """Extracts structs, enums, traits, functions, and impl methods from Rust using tree-sitter."""

    def parse(self, source: str, *, include_bodies: bool = False) -> list[Symbol]:
        parser = Parser(RUST_LANGUAGE)
        tree = parser.parse(source.encode())
        symbols: list[Symbol] = []

        nodes = tree.root_node.children
        for i, node in enumerate(nodes):
            doc = _collect_doc_comments(source, nodes, i)
            attrs = _collect_attributes(source, nodes, i)
            self._process_node(source, node, symbols, include_bodies, doc, attrs)

        return symbols

    def _process_node(
        self, source: str, node: Node, symbols: list[Symbol],
        include_bodies: bool, doc: str, attrs: list[str],
    ) -> None:
        if node.type == "struct_item":
            symbols.append(self._parse_struct(source, node, include_bodies, doc, attrs))
        elif node.type == "enum_item":
            symbols.append(self._parse_enum(source, node, include_bodies, doc, attrs))
        elif node.type == "impl_item":
            symbols.extend(self._parse_impl(source, node, include_bodies))
        elif node.type == "trait_item":
            symbols.extend(self._parse_trait(source, node, include_bodies, doc, attrs))
        elif node.type == "function_item":
            sym = self._parse_function(source, node, None, include_bodies, doc, attrs)
            if sym:
                symbols.append(sym)
        elif node.type == "const_item":
            sym = self._parse_const(source, node, doc)
            if sym:
                symbols.append(sym)
        elif node.type == "static_item":
            sym = self._parse_const(source, node, doc)
            if sym:
                symbols.append(sym)
        elif node.type == "type_item":
            sym = self._parse_type_alias(source, node, doc)
            if sym:
                symbols.append(sym)
        elif node.type == "mod_item":
            self._parse_mod(source, node, symbols, include_bodies)

    # ── Struct ─────────────────────────────────────────────────────────

    def _parse_struct(
        self, source: str, node: Node, include_bodies: bool,
        doc: str, attrs: list[str],
    ) -> Symbol:
        name = _child_text(source, node, "name") or "Unknown"
        vis = _extract_visibility(source, node)
        generics = _extract_generics(source, node)

        fields: list[str] = []
        field_list = _child_by_type(node, "field_declaration_list")
        if field_list:
            for f in field_list.children:
                if f.type == "field_declaration":
                    fname = _child_text(source, f, "name")
                    if fname:
                        fvis = _extract_visibility(source, f)
                        prefix = f"{fvis} " if fvis else ""
                        ftype = _child_by_type(f, "type_identifier") or _child_by_last_type(f)
                        ftype_text = source[ftype.start_byte:ftype.end_byte] if ftype else ""
                        fields.append(f"{prefix}{fname}: {ftype_text}")

        derives = _extract_derives(attrs)

        sig = f"{'pub ' if vis else ''}struct {name}"
        if generics:
            sig += generics

        if include_bodies:
            full = source[node.start_byte:node.end_byte]
            content = f"```rust\n{full}\n```"
        else:
            content = f"```rust\n{sig}\n```"
            if fields:
                content += f"\n\nFields: {', '.join(fields)}"
        if doc:
            content += f"\n\n{doc}"

        metadata: dict = {}
        if vis:
            metadata["visibility"] = vis
        if fields:
            metadata["fields"] = fields
        if derives:
            metadata["derives"] = derives
        if attrs:
            metadata["attributes"] = attrs
        if generics:
            metadata["generics"] = generics

        return Symbol(
            name=name,
            chunk_type=ChunkType.CLASS_OVERVIEW,
            content=content,
            summary=_first_line(doc) or f"Struct {name}",
            metadata=metadata,
        )

    # ── Enum ───────────────────────────────────────────────────────────

    def _parse_enum(
        self, source: str, node: Node, include_bodies: bool,
        doc: str, attrs: list[str],
    ) -> Symbol:
        name = _child_text(source, node, "name") or "Unknown"
        vis = _extract_visibility(source, node)
        generics = _extract_generics(source, node)

        variants: list[str] = []
        variant_list = _child_by_type(node, "enum_variant_list")
        if variant_list:
            for v in variant_list.children:
                if v.type == "enum_variant":
                    vname = _child_text(source, v, "name")
                    if vname:
                        variants.append(vname)

        derives = _extract_derives(attrs)

        sig = f"{'pub ' if vis else ''}enum {name}"
        if generics:
            sig += generics

        if include_bodies:
            full = source[node.start_byte:node.end_byte]
            content = f"```rust\n{full}\n```"
        else:
            content = f"```rust\n{sig}\n```"
            if variants:
                content += f"\n\nVariants: {', '.join(variants)}"
        if doc:
            content += f"\n\n{doc}"

        metadata: dict = {}
        if vis:
            metadata["visibility"] = vis
        if variants:
            metadata["variants"] = variants
        if derives:
            metadata["derives"] = derives
        if attrs:
            metadata["attributes"] = attrs

        return Symbol(
            name=name,
            chunk_type=ChunkType.ENUM,
            content=content,
            summary=_first_line(doc) or f"Enum {name}",
            metadata=metadata,
        )

    # ── Impl ───────────────────────────────────────────────────────────

    def _parse_impl(
        self, source: str, node: Node, include_bodies: bool,
    ) -> list[Symbol]:
        symbols: list[Symbol] = []

        # Determine if this is a trait impl (has "for" keyword)
        type_name = None
        trait_name = None
        has_for = False

        for child in node.children:
            if child.type == "for":
                has_for = True

        if has_for:
            # impl Trait for Type
            type_ids = [c for c in node.children
                        if c.type in ("type_identifier", "generic_type", "scoped_type_identifier")]
            if len(type_ids) >= 2:
                trait_name = source[type_ids[0].start_byte:type_ids[0].end_byte]
                type_name = source[type_ids[1].start_byte:type_ids[1].end_byte]
            elif len(type_ids) == 1:
                type_name = source[type_ids[0].start_byte:type_ids[0].end_byte]
        else:
            # impl Type
            type_ids = [c for c in node.children
                        if c.type in ("type_identifier", "generic_type", "scoped_type_identifier")]
            if type_ids:
                type_name = source[type_ids[0].start_byte:type_ids[0].end_byte]

        if not type_name:
            return symbols

        # Strip generics from parent name for cleaner qualified names
        parent = type_name.split("<")[0] if type_name else type_name

        decl_list = _child_by_type(node, "declaration_list")
        if not decl_list:
            return symbols

        children = decl_list.children
        for i, child in enumerate(children):
            if child.type == "function_item":
                doc = _collect_doc_comments(source, children, i)
                attrs = _collect_attributes(source, children, i)
                sym = self._parse_function(source, child, parent, include_bodies, doc, attrs)
                if sym:
                    if trait_name:
                        sym.metadata["trait_impl"] = trait_name
                    symbols.append(sym)

        return symbols

    # ── Trait ──────────────────────────────────────────────────────────

    def _parse_trait(
        self, source: str, node: Node, include_bodies: bool,
        doc: str, attrs: list[str],
    ) -> list[Symbol]:
        symbols: list[Symbol] = []
        name = _child_text(source, node, "name") or "Unknown"
        vis = _extract_visibility(source, node)
        generics = _extract_generics(source, node)

        sig = f"{'pub ' if vis else ''}trait {name}"
        if generics:
            sig += generics

        content = f"```rust\n{sig}\n```"
        if doc:
            content += f"\n\n{doc}"

        metadata: dict = {}
        if vis:
            metadata["visibility"] = vis
        if generics:
            metadata["generics"] = generics

        symbols.append(Symbol(
            name=name,
            chunk_type=ChunkType.CLASS_OVERVIEW,
            content=content,
            summary=_first_line(doc) or f"Trait {name}",
            metadata=metadata,
        ))

        # Parse trait methods
        decl_list = _child_by_type(node, "declaration_list")
        if decl_list:
            children = decl_list.children
            for i, child in enumerate(children):
                method_doc = _collect_doc_comments(source, children, i)
                if child.type == "function_signature_item":
                    sym = self._parse_function_signature(source, child, name, method_doc)
                    if sym:
                        symbols.append(sym)
                elif child.type == "function_item":
                    method_attrs = _collect_attributes(source, children, i)
                    sym = self._parse_function(source, child, name, include_bodies, method_doc, method_attrs)
                    if sym:
                        symbols.append(sym)

        return symbols

    def _parse_function_signature(
        self, source: str, node: Node, parent: str, doc: str,
    ) -> Symbol | None:
        name = _child_text(source, node, "name") or _extract_identifier(source, node)
        if not name:
            return None

        full = source[node.start_byte:node.end_byte]
        content = f"```rust\n{full}\n```"
        if doc:
            content += f"\n\n{doc}"

        ret_type = _extract_rust_return_type(source, node)
        metadata: dict = {}
        if ret_type:
            metadata["return_type"] = ret_type

        return Symbol(
            name=name,
            chunk_type=ChunkType.METHOD,
            content=content,
            summary=_first_line(doc) or full.rstrip(";").strip(),
            parent=parent,
            metadata=metadata,
        )

    # ── Function ───────────────────────────────────────────────────────

    def _parse_function(
        self, source: str, node: Node, parent: str | None,
        include_bodies: bool, doc: str, attrs: list[str],
    ) -> Symbol | None:
        name = _child_text(source, node, "name") or _extract_identifier(source, node)
        if not name:
            return None

        vis = _extract_visibility(source, node)
        generics = _extract_generics(source, node)
        ret_type = _extract_rust_return_type(source, node)

        is_async = False
        is_unsafe = False
        modifiers = _child_by_type(node, "function_modifiers")
        if modifiers:
            mod_text = source[modifiers.start_byte:modifiers.end_byte]
            is_async = "async" in mod_text
            is_unsafe = "unsafe" in mod_text

        params_node = _child_by_type(node, "parameters")
        params_text = source[params_node.start_byte:params_node.end_byte] if params_node else "()"

        sig_parts: list[str] = []
        if vis:
            sig_parts.append(vis)
        if is_async:
            sig_parts.append("async")
        if is_unsafe:
            sig_parts.append("unsafe")
        sig_parts.append("fn")
        fn_name = name
        if generics:
            fn_name += generics
        sig_parts.append(fn_name + params_text)
        if ret_type:
            sig_parts[-1] += f" -> {ret_type}"
        sig = " ".join(sig_parts)

        if include_bodies:
            full = source[node.start_byte:node.end_byte]
            content = f"```rust\n{full}\n```"
        else:
            content = f"```rust\n{sig}\n```"
            if doc:
                content += f"\n\n{doc}"

        metadata: dict = {}
        if vis:
            metadata["visibility"] = vis
        if ret_type:
            metadata["return_type"] = ret_type
        if is_async:
            metadata["is_async"] = True
        if is_unsafe:
            metadata["is_unsafe"] = True
        if generics:
            metadata["generics"] = generics
        if attrs:
            metadata["attributes"] = attrs

        return Symbol(
            name=name,
            chunk_type=ChunkType.METHOD,
            content=content,
            summary=_first_line(doc) or sig,
            parent=parent,
            metadata=metadata,
        )

    # ── Const / Static ─────────────────────────────────────────────────

    def _parse_const(self, source: str, node: Node, doc: str) -> Symbol | None:
        name = _child_text(source, node, "name") or _extract_identifier(source, node)
        if not name:
            return None

        vis = _extract_visibility(source, node)
        full = source[node.start_byte:node.end_byte]

        content = f"```rust\n{full}\n```"
        if doc:
            content += f"\n\n{doc}"

        metadata: dict = {}
        if vis:
            metadata["visibility"] = vis

        return Symbol(
            name=name,
            chunk_type=ChunkType.CONSTANT,
            content=content,
            summary=_first_line(doc) or full.rstrip(";").strip(),
            metadata=metadata,
        )

    # ── Type alias ─────────────────────────────────────────────────────

    def _parse_type_alias(self, source: str, node: Node, doc: str) -> Symbol | None:
        name = _child_text(source, node, "name") or _extract_identifier(source, node)
        if not name:
            return None

        vis = _extract_visibility(source, node)
        full = source[node.start_byte:node.end_byte]

        content = f"```rust\n{full}\n```"
        if doc:
            content += f"\n\n{doc}"

        metadata: dict = {}
        if vis:
            metadata["visibility"] = vis

        return Symbol(
            name=name,
            chunk_type=ChunkType.CONSTANT,
            content=content,
            summary=_first_line(doc) or full.rstrip(";").strip(),
            metadata=metadata,
        )

    # ── Module ─────────────────────────────────────────────────────────

    def _parse_mod(
        self, source: str, node: Node, symbols: list[Symbol],
        include_bodies: bool,
    ) -> None:
        decl_list = _child_by_type(node, "declaration_list")
        if not decl_list:
            return

        children = decl_list.children
        for i, child in enumerate(children):
            doc = _collect_doc_comments(source, children, i)
            attrs = _collect_attributes(source, children, i)
            self._process_node(source, child, symbols, include_bodies, doc, attrs)


# ── Helpers ────────────────────────────────────────────────────────────


def _child_by_type(node: Node, type_name: str) -> Node | None:
    for child in node.children:
        if child.type == type_name:
            return child
    return None


def _child_by_last_type(node: Node) -> Node | None:
    """Return the last non-punctuation child (useful for getting type annotations)."""
    for child in reversed(node.children):
        if child.type not in (",", ";", "{", "}", "(", ")"):
            return child
    return None


def _child_text(source: str, node: Node, field_name: str) -> str | None:
    child = node.child_by_field_name(field_name)
    if child:
        return source[child.start_byte:child.end_byte]
    return None


def _extract_identifier(source: str, node: Node) -> str | None:
    """Fall back to finding an identifier child node."""
    for child in node.children:
        if child.type == "identifier":
            return source[child.start_byte:child.end_byte]
    return None


def _extract_visibility(source: str, node: Node) -> str:
    vis = _child_by_type(node, "visibility_modifier")
    if vis:
        return source[vis.start_byte:vis.end_byte]
    return ""


def _extract_generics(source: str, node: Node) -> str:
    tp = _child_by_type(node, "type_parameters")
    if tp:
        return source[tp.start_byte:tp.end_byte]
    return ""


def _extract_rust_return_type(source: str, node: Node) -> str | None:
    """Extract the return type after -> in a function signature."""
    found_arrow = False
    for child in node.children:
        if found_arrow and child.type not in ("block", "{", "}"):
            return source[child.start_byte:child.end_byte]
        if child.type == "->":
            found_arrow = True
    return None


def _collect_doc_comments(source: str, siblings: list, index: int) -> str:
    """Collect consecutive /// doc comments preceding a node, skipping attributes."""
    lines: list[str] = []
    i = index - 1
    # Skip past any attribute items first
    while i >= 0 and siblings[i].type == "attribute_item":
        i -= 1
    while i >= 0:
        prev = siblings[i]
        if prev.type == "line_comment":
            text = source[prev.start_byte:prev.end_byte]
            if text.startswith("///"):
                lines.append(text[3:].strip())
                i -= 1
                continue
        break
    if not lines:
        return ""
    lines.reverse()
    return "\n".join(lines)


def _collect_attributes(source: str, siblings: list, index: int) -> list[str]:
    """Collect #[...] attribute items preceding a node."""
    attrs: list[str] = []
    i = index - 1
    while i >= 0:
        prev = siblings[i]
        if prev.type == "attribute_item":
            attrs.append(source[prev.start_byte:prev.end_byte])
            i -= 1
            continue
        elif prev.type == "line_comment":
            # Skip past doc comments to find attributes before them
            i -= 1
            continue
        break
    attrs.reverse()
    return attrs


def _extract_derives(attrs: list[str]) -> list[str]:
    """Extract derive names from #[derive(...)] attributes."""
    derives: list[str] = []
    for attr in attrs:
        if "derive(" in attr:
            inner = attr.split("derive(", 1)[1].rstrip(")]")
            derives.extend(d.strip() for d in inner.split(","))
    return derives


def _first_line(text: str) -> str:
    if not text:
        return ""
    line = text.split("\n")[0].strip()
    return line[:200]

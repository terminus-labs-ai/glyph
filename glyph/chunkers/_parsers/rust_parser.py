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
        src = source.encode("utf-8")
        tree = parser.parse(src)
        symbols: list[Symbol] = []

        nodes = tree.root_node.children
        for i, node in enumerate(nodes):
            doc = _collect_doc_comments(src, nodes, i)
            attrs = _collect_attributes(src, nodes, i)
            self._process_node(src, node, symbols, include_bodies, doc, attrs)

        return symbols

    def _process_node(
        self, src: bytes, node: Node, symbols: list[Symbol],
        include_bodies: bool, doc: str, attrs: list[str],
    ) -> None:
        if node.type == "struct_item":
            symbols.append(self._parse_struct(src, node, include_bodies, doc, attrs))
        elif node.type == "enum_item":
            symbols.append(self._parse_enum(src, node, include_bodies, doc, attrs))
        elif node.type == "impl_item":
            symbols.extend(self._parse_impl(src, node, include_bodies))
        elif node.type == "trait_item":
            symbols.extend(self._parse_trait(src, node, include_bodies, doc, attrs))
        elif node.type == "function_item":
            sym = self._parse_function(src, node, None, include_bodies, doc, attrs)
            if sym:
                symbols.append(sym)
        elif node.type == "const_item":
            sym = self._parse_const(src, node, doc)
            if sym:
                symbols.append(sym)
        elif node.type == "static_item":
            sym = self._parse_const(src, node, doc)
            if sym:
                symbols.append(sym)
        elif node.type == "type_item":
            sym = self._parse_type_alias(src, node, doc)
            if sym:
                symbols.append(sym)
        elif node.type == "mod_item":
            self._parse_mod(src, node, symbols, include_bodies)

    # ── Struct ─────────────────────────────────────────────────────────

    def _parse_struct(
        self, src: bytes, node: Node, include_bodies: bool,
        doc: str, attrs: list[str],
    ) -> Symbol:
        name = _child_text(src, node, "name") or "Unknown"
        vis = _extract_visibility(src, node)
        generics = _extract_generics(src, node)

        fields: list[str] = []
        field_list = _child_by_type(node, "field_declaration_list")
        if field_list:
            for f in field_list.children:
                if f.type == "field_declaration":
                    fname = _child_text(src, f, "name")
                    if fname:
                        fvis = _extract_visibility(src, f)
                        prefix = f"{fvis} " if fvis else ""
                        ftype = _child_by_type(f, "type_identifier") or _child_by_last_type(f)
                        ftype_text = _slice(src, ftype) if ftype else ""
                        fields.append(f"{prefix}{fname}: {ftype_text}")

        derives = _extract_derives(attrs)

        sig = f"{'pub ' if vis else ''}struct {name}"
        if generics:
            sig += generics

        if include_bodies:
            content = f"```rust\n{_slice(src, node)}\n```"
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
        self, src: bytes, node: Node, include_bodies: bool,
        doc: str, attrs: list[str],
    ) -> Symbol:
        name = _child_text(src, node, "name") or "Unknown"
        vis = _extract_visibility(src, node)
        generics = _extract_generics(src, node)

        variants: list[str] = []
        variant_list = _child_by_type(node, "enum_variant_list")
        if variant_list:
            for v in variant_list.children:
                if v.type == "enum_variant":
                    vname = _child_text(src, v, "name")
                    if vname:
                        variants.append(vname)

        derives = _extract_derives(attrs)

        sig = f"{'pub ' if vis else ''}enum {name}"
        if generics:
            sig += generics

        if include_bodies:
            content = f"```rust\n{_slice(src, node)}\n```"
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
        self, src: bytes, node: Node, include_bodies: bool,
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
                trait_name = _slice(src, type_ids[0])
                type_name = _slice(src, type_ids[1])
            elif len(type_ids) == 1:
                type_name = _slice(src, type_ids[0])
        else:
            # impl Type
            type_ids = [c for c in node.children
                        if c.type in ("type_identifier", "generic_type", "scoped_type_identifier")]
            if type_ids:
                type_name = _slice(src, type_ids[0])

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
                doc = _collect_doc_comments(src, children, i)
                attrs = _collect_attributes(src, children, i)
                sym = self._parse_function(src, child, parent, include_bodies, doc, attrs)
                if sym:
                    if trait_name:
                        sym.metadata["trait_impl"] = trait_name
                    symbols.append(sym)

        return symbols

    # ── Trait ──────────────────────────────────────────────────────────

    def _parse_trait(
        self, src: bytes, node: Node, include_bodies: bool,
        doc: str, attrs: list[str],
    ) -> list[Symbol]:
        symbols: list[Symbol] = []
        name = _child_text(src, node, "name") or "Unknown"
        vis = _extract_visibility(src, node)
        generics = _extract_generics(src, node)

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
                method_doc = _collect_doc_comments(src, children, i)
                if child.type == "function_signature_item":
                    sym = self._parse_function_signature(src, child, name, method_doc)
                    if sym:
                        symbols.append(sym)
                elif child.type == "function_item":
                    method_attrs = _collect_attributes(src, children, i)
                    sym = self._parse_function(src, child, name, include_bodies, method_doc, method_attrs)
                    if sym:
                        symbols.append(sym)

        return symbols

    def _parse_function_signature(
        self, src: bytes, node: Node, parent: str, doc: str,
    ) -> Symbol | None:
        name = _child_text(src, node, "name") or _extract_identifier(src, node)
        if not name:
            return None

        full = _slice(src, node)
        content = f"```rust\n{full}\n```"
        if doc:
            content += f"\n\n{doc}"

        ret_type = _extract_rust_return_type(src, node)
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
        self, src: bytes, node: Node, parent: str | None,
        include_bodies: bool, doc: str, attrs: list[str],
    ) -> Symbol | None:
        name = _child_text(src, node, "name") or _extract_identifier(src, node)
        if not name:
            return None

        vis = _extract_visibility(src, node)
        generics = _extract_generics(src, node)
        ret_type = _extract_rust_return_type(src, node)

        is_async = False
        is_unsafe = False
        modifiers = _child_by_type(node, "function_modifiers")
        if modifiers:
            mod_text = _slice(src, modifiers)
            is_async = "async" in mod_text
            is_unsafe = "unsafe" in mod_text

        params_node = _child_by_type(node, "parameters")
        params_text = _slice(src, params_node) if params_node else "()"

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
            content = f"```rust\n{_slice(src, node)}\n```"
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

    def _parse_const(self, src: bytes, node: Node, doc: str) -> Symbol | None:
        name = _child_text(src, node, "name") or _extract_identifier(src, node)
        if not name:
            return None

        vis = _extract_visibility(src, node)
        full = _slice(src, node)

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

    def _parse_type_alias(self, src: bytes, node: Node, doc: str) -> Symbol | None:
        name = _child_text(src, node, "name") or _extract_identifier(src, node)
        if not name:
            return None

        vis = _extract_visibility(src, node)
        full = _slice(src, node)

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
        self, src: bytes, node: Node, symbols: list[Symbol],
        include_bodies: bool,
    ) -> None:
        decl_list = _child_by_type(node, "declaration_list")
        if not decl_list:
            return

        children = decl_list.children
        for i, child in enumerate(children):
            doc = _collect_doc_comments(src, children, i)
            attrs = _collect_attributes(src, children, i)
            self._process_node(src, child, symbols, include_bodies, doc, attrs)


# ── Helpers ────────────────────────────────────────────────────────────


def _slice(src: bytes, node: Node) -> str:
    """Decode a node's byte span from the UTF-8 source."""
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


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


def _child_text(src: bytes, node: Node, field_name: str) -> str | None:
    child = node.child_by_field_name(field_name)
    if child:
        return _slice(src, child)
    return None


def _extract_identifier(src: bytes, node: Node) -> str | None:
    """Fall back to finding an identifier child node."""
    for child in node.children:
        if child.type == "identifier":
            return _slice(src, child)
    return None


def _extract_visibility(src: bytes, node: Node) -> str:
    vis = _child_by_type(node, "visibility_modifier")
    if vis:
        return _slice(src, vis)
    return ""


def _extract_generics(src: bytes, node: Node) -> str:
    tp = _child_by_type(node, "type_parameters")
    if tp:
        return _slice(src, tp)
    return ""


def _extract_rust_return_type(src: bytes, node: Node) -> str | None:
    """Extract the return type after -> in a function signature."""
    found_arrow = False
    for child in node.children:
        if found_arrow and child.type not in ("block", "{", "}"):
            return _slice(src, child)
        if child.type == "->":
            found_arrow = True
    return None


def _collect_doc_comments(src: bytes, siblings: list, index: int) -> str:
    """Collect consecutive /// doc comments preceding a node, skipping attributes."""
    lines: list[str] = []
    i = index - 1
    # Skip past any attribute items first
    while i >= 0 and siblings[i].type == "attribute_item":
        i -= 1
    while i >= 0:
        prev = siblings[i]
        if prev.type == "line_comment":
            text = _slice(src, prev)
            if text.startswith("///"):
                lines.append(text[3:].strip())
                i -= 1
                continue
        break
    if not lines:
        return ""
    lines.reverse()
    return "\n".join(lines)


def _collect_attributes(src: bytes, siblings: list, index: int) -> list[str]:
    """Collect #[...] attribute items preceding a node."""
    attrs: list[str] = []
    i = index - 1
    while i >= 0:
        prev = siblings[i]
        if prev.type == "attribute_item":
            attrs.append(_slice(src, prev))
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

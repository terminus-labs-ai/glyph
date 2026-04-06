from __future__ import annotations

import logging
import re

import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser, Node

from glyph.chunkers._parsers import Symbol
from glyph.domain.models import ChunkType

logger = logging.getLogger(__name__)

CPP_LANGUAGE = Language(tscpp.language())

# UE macro patterns — used for stripping (simple versions for preprocess)
_UE_TYPE_MACRO_STRIP_RE = re.compile(
    r"^\s*(?:UCLASS|USTRUCT|UENUM)\s*\(.*$", re.MULTILINE
)
_UE_MEMBER_MACRO_STRIP_RE = re.compile(
    r"^\s*(?:UPROPERTY|UFUNCTION)\s*\(.*$", re.MULTILINE
)
_GENERATED_BODY_RE = re.compile(
    r"^\s*GENERATED_(?:USTRUCT_)?BODY\(\)\s*$", re.MULTILINE
)
# API export macros like ENGINE_API, COREUOBJECT_API, etc.
_API_MACRO_RE = re.compile(r"\b[A-Z][A-Z0-9_]*_API\b")
# UE_DEPRECATED(version, "message")
_UE_DEPRECATED_RE = re.compile(r"\bUE_DEPRECATED\s*\([^)]*\)\s*")
# UMETA(...) in enum variants
_UMETA_RE = re.compile(r"\s*UMETA\s*\([^)]*\)")
# ENUM_CLASS_FLAGS(...)
_ENUM_FLAGS_RE = re.compile(r"^\s*ENUM_CLASS_FLAGS\s*\([^)]*\)\s*$", re.MULTILINE)
# PRAGMA macros
_PRAGMA_RE = re.compile(r"^\s*PRAGMA_\w+\s*$", re.MULTILINE)
# DECLARE_*DELEGATE* macros
_DELEGATE_RE = re.compile(r"^\s*DECLARE_\w+\s*\([^;]*\)\s*;?\s*$", re.MULTILINE)
# Preprocessor directives (#if, #endif, #ifdef, #define, #include, etc.)
_PREPROCESSOR_RE = re.compile(r"^\s*#\s*(?:if|ifdef|ifndef|else|elif|endif|define|include|pragma|error|warning)\b.*$", re.MULTILINE)


class CppParser:
    """Extracts classes, structs, enums, methods, and properties from C++ using tree-sitter.

    Handles Unreal Engine macros (UCLASS, UPROPERTY, UFUNCTION, etc.) by
    stripping them before parsing and re-associating their metadata afterward.
    """

    def parse(self, source: str, *, include_bodies: bool = False) -> list[Symbol]:
        # Phase 1: Extract UE macro metadata keyed by source line number
        type_macros = _extract_macros(source, ["UCLASS", "USTRUCT", "UENUM"])
        member_macros = _extract_macros(source, ["UPROPERTY", "UFUNCTION"])

        # Phase 2: Clean source for tree-sitter (blank-line replacement preserves line numbers)
        cleaned = _preprocess(source)

        # Phase 3: Parse with tree-sitter
        parser = Parser(CPP_LANGUAGE)
        src = cleaned.encode("utf-8")
        tree = parser.parse(src)

        # Phase 4: Extract symbols from AST
        symbols: list[Symbol] = []
        nodes = tree.root_node.children
        for i, node in enumerate(nodes):
            doc = _collect_doc_comment(src, nodes, i)
            line = node.start_point[0]

            if node.type == "class_specifier":
                if not _child_by_type(node, "field_declaration_list"):
                    continue  # Skip forward declarations
                macro_args = _find_nearest_macro(type_macros, line)
                symbols.extend(self._parse_class(
                    src, node, include_bodies, doc, macro_args,
                    member_macros,
                ))
            elif node.type == "struct_specifier":
                if not _child_by_type(node, "field_declaration_list"):
                    continue
                macro_args = _find_nearest_macro(type_macros, line)
                symbols.extend(self._parse_class(
                    src, node, include_bodies, doc, macro_args,
                    member_macros, is_struct=True,
                ))
            elif node.type == "enum_specifier":
                if not _child_by_type(node, "enumerator_list"):
                    continue  # Skip forward declarations
                macro_args = _find_nearest_macro(type_macros, line)
                symbols.append(self._parse_enum(src, node, doc, macro_args))
            elif node.type == "function_definition":
                macro_args = _find_nearest_macro(member_macros, line)
                sym = self._parse_free_function(src, node, include_bodies, doc, macro_args)
                if sym:
                    symbols.append(sym)
            elif node.type == "declaration":
                if _has_function_declarator(node):
                    macro_args = _find_nearest_macro(member_macros, line)
                    sym = self._parse_free_function(src, node, include_bodies, doc, macro_args)
                    if sym:
                        symbols.append(sym)

        return symbols

    # ── Class / Struct ──────────────────────────────────────────────────

    def _parse_class(
        self,
        src: bytes,
        node: Node,
        include_bodies: bool,
        doc: str,
        macro_args: str | None,
        member_macros: dict[int, tuple[str, str]],
        *,
        is_struct: bool = False,
    ) -> list[Symbol]:
        name = _child_type_id(src, node)
        if not name:
            return []

        bases = _extract_bases(src, node)
        kind = "struct" if is_struct else "class"
        specifiers = _parse_macro_args(macro_args) if macro_args else {}

        # Build class signature
        sig_parts = [f"{kind} {name}"]
        if bases:
            sig_parts.append(f" : {', '.join(bases)}")
        signature = "".join(sig_parts)

        summary = _first_sentence(doc) or signature
        content = f"```cpp\n{signature}\n```"
        if doc:
            content += f"\n\n{doc}"

        symbols: list[Symbol] = []
        symbols.append(Symbol(
            name=name,
            chunk_type=ChunkType.CLASS_OVERVIEW,
            content=content,
            summary=summary,
            metadata={
                "kind": kind,
                "bases": bases or None,
                "ue_specifiers": specifiers or None,
            },
        ))

        # Parse members
        field_list = _child_by_type(node, "field_declaration_list")
        if field_list:
            visibility = "public" if is_struct else "private"
            members = field_list.children
            for j, child in enumerate(members):
                if child.type == "access_specifier":
                    visibility = _slice(src, child).strip().rstrip(":")
                    continue
                if child.type in ("field_declaration", "declaration", "function_definition"):
                    child_doc = _collect_doc_comment(src, members, j)
                    m_args = _find_nearest_macro(member_macros, child.start_point[0])
                    m_specifiers = _parse_macro_args(m_args) if m_args else {}

                    if _has_function_declarator(child):
                        sym = self._parse_method(
                            src, child, name, include_bodies, child_doc,
                            visibility, m_specifiers,
                        )
                    else:
                        sym = self._parse_field(
                            src, child, name, child_doc,
                            visibility, m_specifiers,
                        )
                    if sym:
                        symbols.append(sym)

        return symbols

    def _parse_method(
        self,
        src: bytes,
        node: Node,
        class_name: str,
        include_bodies: bool,
        doc: str,
        visibility: str,
        ue_specifiers: dict,
    ) -> Symbol | None:
        # Get the full declaration text for signature
        decl_text = _slice(src, node).strip().rstrip(";").strip()
        # Remove inline body if present
        body_node = _child_by_type(node, "compound_statement")
        if body_node and not include_bodies:
            decl_text = _slice(src, node, end=body_node.start_byte).strip()

        # Extract function name from declarator
        func_decl = _find_function_declarator(node)
        if not func_decl:
            return None
        func_name = _child_text(src, func_decl, "declarator") or _child_text(src, func_decl, "identifier")
        if not func_name:
            # Try nested declarator
            for c in func_decl.children:
                if c.type in ("identifier", "field_identifier", "destructor_name"):
                    func_name = _slice(src, c)
                    break
                elif c.type == "qualified_identifier":
                    func_name = _slice(src, c)
                    break
        if not func_name:
            return None

        # Detect qualifiers
        qualifiers = []
        text = _slice(src, node)
        if "virtual" in text.split("(")[0]:
            qualifiers.append("virtual")
        if "static" in text.split("(")[0]:
            qualifiers.append("static")
        if node.type == "function_definition" and "const" in text.split(")")[-1]:
            qualifiers.append("const")
        elif "const" in text.rsplit(")", 1)[-1] if ")" in text else "":
            qualifiers.append("const")
        if "override" in text:
            qualifiers.append("override")

        # Extract return type
        ret_type = _extract_return_type(src, node)

        # Extract parameters
        params = _extract_parameters(src, func_decl)

        summary = _first_sentence(doc) or decl_text[:100]
        content = f"```cpp\n{decl_text}\n```"
        if qualifiers:
            content += f"\n\n**Qualifiers:** {', '.join(qualifiers)}"
        if doc:
            content += f"\n\n{doc}"

        return Symbol(
            name=func_name,
            chunk_type=ChunkType.METHOD,
            content=content,
            summary=summary,
            parent=class_name,
            metadata={
                "return_type": ret_type,
                "params": params or None,
                "visibility": visibility,
                "qualifiers": qualifiers or None,
                "ue_specifiers": ue_specifiers or None,
            },
        )

    def _parse_field(
        self,
        src: bytes,
        node: Node,
        class_name: str,
        doc: str,
        visibility: str,
        ue_specifiers: dict,
    ) -> Symbol | None:
        decl_text = _slice(src, node).strip().rstrip(";").strip()

        # Extract field name — last identifier-like token before ; or = or :
        field_name = None
        for child in reversed(node.children):
            if child.type in ("field_identifier", "identifier"):
                field_name = _slice(src, child)
                break

        if not field_name:
            return None

        # Extract type (everything before the field name, minus qualifiers)
        field_type = _extract_field_type(src, node, field_name)

        summary = _first_sentence(doc) or decl_text[:100]
        content = f"`{decl_text}`"
        if doc:
            content += f"\n\n{doc}"

        return Symbol(
            name=field_name,
            chunk_type=ChunkType.PROPERTY,
            content=content,
            summary=summary,
            parent=class_name,
            metadata={
                "type": field_type,
                "visibility": visibility,
                "ue_specifiers": ue_specifiers or None,
            },
        )

    # ── Enum ────────────────────────────────────────────────────────────

    def _parse_enum(
        self, src: bytes, node: Node, doc: str, macro_args: str | None,
    ) -> Symbol:
        name = _child_type_id(src, node) or "AnonymousEnum"
        specifiers = _parse_macro_args(macro_args) if macro_args else {}

        variants: list[str] = []
        enum_list = _child_by_type(node, "enumerator_list")
        if enum_list:
            for child in enum_list.children:
                if child.type == "enumerator":
                    v = _slice(src, child).strip()
                    # Clean UMETA from variant text
                    v = _UMETA_RE.sub("", v).strip().rstrip(",")
                    if v:
                        variants.append(v)

        variant_text = "\n".join(f"  {v}," for v in variants)
        is_class = any(c.type == "class" for c in node.children)
        base_type = ""
        for c in node.children:
            if c.type == ":" and c.next_sibling:
                base_type = f" : {_slice(src, c.next_sibling)}"
                break

        sig = f"enum {'class ' if is_class else ''}{name}{base_type}"
        content = f"```cpp\n{sig} {{\n{variant_text}\n}};\n```"
        if doc:
            content += f"\n\n{doc}"

        return Symbol(
            name=name,
            chunk_type=ChunkType.ENUM,
            content=content,
            summary=_first_sentence(doc) or sig,
            metadata={
                "variants": variants,
                "ue_specifiers": specifiers or None,
            },
        )

    # ── Free function ───────────────────────────────────────────────────

    def _parse_free_function(
        self,
        src: bytes,
        node: Node,
        include_bodies: bool,
        doc: str,
        macro_args: str | None,
    ) -> Symbol | None:
        func_decl = _find_function_declarator(node)
        if not func_decl:
            return None
        func_name = None
        for c in func_decl.children:
            if c.type in ("identifier", "field_identifier", "qualified_identifier"):
                func_name = _slice(src, c)
                break
        if not func_name:
            return None

        decl_text = _slice(src, node).strip().rstrip(";").strip()
        body_node = _child_by_type(node, "compound_statement")
        if body_node and not include_bodies:
            decl_text = _slice(src, node, end=body_node.start_byte).strip()

        ret_type = _extract_return_type(src, node)
        params = _extract_parameters(src, func_decl)
        specifiers = _parse_macro_args(macro_args) if macro_args else {}

        summary = _first_sentence(doc) or decl_text[:100]
        content = f"```cpp\n{decl_text}\n```"
        if doc:
            content += f"\n\n{doc}"

        return Symbol(
            name=func_name,
            chunk_type=ChunkType.METHOD,
            content=content,
            summary=summary,
            metadata={
                "return_type": ret_type,
                "params": params or None,
                "ue_specifiers": specifiers or None,
            },
        )


# ── Helpers ─────────────────────────────────────────────────────────────


def _slice(src: bytes, node: Node, *, end: int | None = None) -> str:
    return src[node.start_byte : end or node.end_byte].decode("utf-8", errors="replace")


def _child_by_type(node: Node, type_name: str) -> Node | None:
    for c in node.children:
        if c.type == type_name:
            return c
    return None


def _child_text(src: bytes, node: Node, field: str) -> str | None:
    child = node.child_by_field_name(field)
    if child:
        return _slice(src, child)
    return None


def _child_type_id(src: bytes, node: Node) -> str | None:
    """Get the type_identifier child (class/struct/enum name)."""
    for c in node.children:
        if c.type == "type_identifier":
            return _slice(src, c)
    return None


def _has_function_declarator(node: Node) -> bool:
    """Check if a node contains a function declarator anywhere in its tree."""
    for c in node.children:
        if c.type == "function_declarator":
            return True
        if c.type in ("parenthesized_declarator", "pointer_declarator", "reference_declarator"):
            if _has_function_declarator(c):
                return True
    return False


def _find_function_declarator(node: Node) -> Node | None:
    """Find the function_declarator node in a declaration."""
    for c in node.children:
        if c.type == "function_declarator":
            return c
        if c.type in ("parenthesized_declarator", "pointer_declarator", "reference_declarator"):
            result = _find_function_declarator(c)
            if result:
                return result
    return None


def _extract_bases(src: bytes, node: Node) -> list[str]:
    """Extract base classes from a class/struct specifier."""
    clause = _child_by_type(node, "base_class_clause")
    if not clause:
        return []
    bases = []
    for c in clause.children:
        if c.type == "type_identifier":
            bases.append(_slice(src, c))
        elif c.type == "qualified_identifier":
            bases.append(_slice(src, c))
    return bases


def _extract_return_type(src: bytes, node: Node) -> str:
    """Extract the return type from a function declaration."""
    parts = []
    for c in node.children:
        if c.type in ("function_declarator", "pointer_declarator", "reference_declarator"):
            break
        if c.type in ("identifier", "field_identifier"):
            break
        if c.type in ("primitive_type", "type_identifier", "qualified_identifier",
                       "sized_type_specifier", "template_type"):
            parts.append(_slice(src, c))
        elif c.type in ("type_qualifier",):
            parts.append(_slice(src, c))
    return " ".join(parts) or "void"


def _extract_field_type(src: bytes, node: Node, field_name: str) -> str:
    """Extract the type portion of a field declaration."""
    parts = []
    for c in node.children:
        if c.type in ("field_identifier", "identifier"):
            name = _slice(src, c)
            if name == field_name:
                break
        if c.type in ("primitive_type", "type_identifier", "qualified_identifier",
                       "sized_type_specifier", "template_type", "type_qualifier"):
            parts.append(_slice(src, c))
    return " ".join(parts) or ""


def _extract_parameters(src: bytes, func_decl: Node) -> list[dict]:
    """Extract parameters from a function declarator."""
    params = []
    param_list = _child_by_type(func_decl, "parameter_list")
    if not param_list:
        return params
    for c in param_list.children:
        if c.type == "parameter_declaration":
            p_text = _slice(src, c).strip()
            params.append({"text": p_text})
        elif c.type == "optional_parameter_declaration":
            p_text = _slice(src, c).strip()
            params.append({"text": p_text})
    return params


def _collect_doc_comment(src: bytes, siblings: list[Node], index: int) -> str:
    """Collect /** */ or /// doc comments preceding a node."""
    comments = []
    i = index - 1
    while i >= 0:
        sib = siblings[i]
        if sib.type == "comment":
            text = _slice(src, sib).strip()
            if text.startswith("/**") or text.startswith("///"):
                comments.insert(0, _clean_doc_comment(text))
            else:
                break
        elif sib.type in ("expression_statement",):
            # Skip leftover macro stubs between comment and declaration
            i -= 1
            continue
        else:
            break
        i -= 1
    return "\n".join(comments).strip()


def _clean_doc_comment(text: str) -> str:
    """Strip comment delimiters from a doc comment."""
    if text.startswith("/**") and text.endswith("*/"):
        text = text[3:-2]
    elif text.startswith("///"):
        text = text[3:]
    # Strip leading * on each line
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("* "):
            line = line[2:]
        elif line.startswith("*"):
            line = line[1:]
        lines.append(line)
    return "\n".join(lines).strip()


def _extract_macros(source: str, macro_names: list[str]) -> dict[int, tuple[str, str]]:
    """Extract UE macros with balanced-parentheses scanning.

    Returns {line_number: (macro_name, args_string)}.
    Handles nested parentheses like meta=(AllowPrivateAccess="true").
    """
    macros = {}
    for name in macro_names:
        pattern = re.compile(rf"^\s*({name})\s*\(", re.MULTILINE)
        for m in pattern.finditer(source):
            line = source[:m.start()].count("\n")
            # Scan forward from the opening paren to find balanced close
            start = m.end() - 1  # position of '('
            args = _extract_balanced_parens(source, start)
            if args is not None:
                macros[line] = (name, args)
    return macros


def _extract_balanced_parens(source: str, start: int) -> str | None:
    """Extract content between balanced parentheses starting at source[start]='('."""
    if start >= len(source) or source[start] != "(":
        return None
    depth = 0
    i = start
    while i < len(source):
        if source[i] == "(":
            depth += 1
        elif source[i] == ")":
            depth -= 1
            if depth == 0:
                return source[start + 1 : i].strip()
        i += 1
    return None


def _find_nearest_macro(macros: dict[int, tuple[str, str]], target_line: int) -> str | None:
    """Find the macro args string for the nearest preceding macro within 3 lines."""
    best = None
    best_dist = 4  # max distance
    for line, (name, args) in macros.items():
        dist = target_line - line
        if 0 < dist < best_dist:
            best = args
            best_dist = dist
    return best


def _parse_macro_args(args_str: str) -> dict:
    """Parse UE macro args like 'BlueprintCallable, Category=Movement, meta=(...)' into a dict."""
    if not args_str:
        return {}
    result: dict[str, str | bool] = {}
    # Handle nested parentheses in meta=(...)
    depth = 0
    current = ""
    for ch in args_str:
        if ch == "(":
            depth += 1
            current += ch
        elif ch == ")":
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            _add_macro_arg(result, current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        _add_macro_arg(result, current.strip())
    return result


def _add_macro_arg(result: dict, arg: str) -> None:
    """Add a single macro argument to the result dict."""
    if "=" in arg:
        key, _, value = arg.partition("=")
        result[key.strip()] = value.strip().strip('"')
    else:
        result[arg] = True


def _preprocess(source: str) -> str:
    """Strip UE macros and preprocessor directives for clean tree-sitter parsing.

    Works line-by-line to guarantee line numbers are preserved exactly.
    Each stripped line becomes empty, keeping total line count identical.
    """
    lines = source.split("\n")
    result: list[str] = []

    # Patterns that cause an entire line to be blanked
    _line_strip_patterns = [
        re.compile(r"^\s*GENERATED_(?:USTRUCT_)?BODY\(\)\s*$"),
        re.compile(r"^\s*(?:UCLASS|USTRUCT|UENUM)\s*\("),
        re.compile(r"^\s*(?:UPROPERTY|UFUNCTION)\s*\("),
        re.compile(r"^\s*ENUM_CLASS_FLAGS\s*\("),
        re.compile(r"^\s*PRAGMA_\w+\s*$"),
        re.compile(r"^\s*DECLARE_\w+\s*\("),
        re.compile(r"^\s*#\s*(?:if|ifdef|ifndef|else|elif|endif|define|include|pragma|error|warning)\b"),
    ]

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = False

        for pat in _line_strip_patterns:
            if pat.match(line):
                # For macros with parentheses, blank all continuation lines too
                if "(" in line:
                    # Count unbalanced parens to handle multi-line macros
                    depth = line.count("(") - line.count(")")
                    result.append("")
                    while depth > 0 and i + 1 < len(lines):
                        i += 1
                        depth += lines[i].count("(") - lines[i].count(")")
                        result.append("")
                else:
                    result.append("")
                stripped = True
                break

        if not stripped:
            # Inline replacements (don't change line count)
            cleaned_line = line
            cleaned_line = _API_MACRO_RE.sub("", cleaned_line)
            cleaned_line = _UE_DEPRECATED_RE.sub("", cleaned_line)
            cleaned_line = _UMETA_RE.sub("", cleaned_line)
            result.append(cleaned_line)

        i += 1

    return "\n".join(result)


def _first_sentence(text: str) -> str:
    if not text:
        return ""
    for sep in (". ", ".\n"):
        pos = text.find(sep)
        if pos > 0:
            return text[: pos + 1]
    return text[:200]

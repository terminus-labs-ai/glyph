from __future__ import annotations

import re
import logging
from glyph.chunkers._parsers import Symbol
from glyph.domain.models import ChunkType

logger = logging.getLogger(__name__)

# #define NAME value (with optional \ continuations)
_DEFINE_RE = re.compile(r"^#define\s+(\w+)\s+((?:[^\n\\]|\\\n)*)", re.MULTILINE)

# cbuffer Name : register(bN) {
_CBUFFER_RE = re.compile(r"^cbuffer\s+(\w+)\s*(?::\s*register\((\w+)\))?\s*\{", re.MULTILINE)

# struct Name {
_STRUCT_RE = re.compile(r"^struct\s+(\w+)\s*\{", re.MULTILINE)

# Resource types: Texture2D<...>, SamplerState, etc.
_RESOURCE_RE = re.compile(
    r"^((?:RW)?(?:Texture(?:2D(?:Array)?|3D|Cube|2DMS|2DArray)|StructuredBuffer|ByteAddressBuffer|"
    r"AppendStructuredBuffer|ConsumeStructuredBuffer)(?:<[^>]*>)?|SamplerState|SamplerComparisonState|Buffer(?:<[^>]*>)?)\s+"
    r"(\w+)\s*(?::\s*register\((\w+)\))?\s*;",
    re.MULTILINE,
)

# Function: ReturnType Name(params) [: SEMANTIC] {
_FUNCTION_RE = re.compile(
    r"^([\w][\w<>, ]*?)\s+(\w+)\s*\(([^)]*)\)\s*(?::\s*(\w+))?\s*\{",
    re.MULTILINE,
)

# [numthreads(X, Y, Z)]
_NUMTHREADS_RE = re.compile(r"\[numthreads\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\]")

# struct/cbuffer member line: Type Name [: SEMANTIC];
_MEMBER_RE = re.compile(r"^\s+([\w][\w<>]*(?:\s*<[^>]*>)?)\s+(\w+)\s*(?::\s*(\w+))?\s*;", re.MULTILINE)

_SV_ENTRY_SEMANTICS = frozenset({
    "SV_Target", "SV_Target0", "SV_Target1", "SV_Target2", "SV_Target3",
    "SV_Target4", "SV_Target5", "SV_Target6", "SV_Target7",
    "SV_Position", "SV_POSITION",
    "SV_DispatchThreadID", "SV_GroupID", "SV_GroupIndex", "SV_GroupThreadID",
    "SV_Depth", "SV_Coverage",
})


def _extract_brace_block(source: str, open_pos: int) -> tuple[str, int]:
    """Returns (content_between_braces, position_after_closing_brace).
    Returns ("", -1) if unbalanced.
    open_pos should be the position of the opening '{'.
    """
    depth = 0
    i = open_pos
    start = -1
    while i < len(source):
        if source[i] == '{':
            depth += 1
            if depth == 1:
                start = i + 1
        elif source[i] == '}':
            depth -= 1
            if depth == 0:
                return source[start:i], i + 1
        i += 1
    return "", -1  # unbalanced


def _first_line(text: str) -> str:
    if not text:
        return ""
    return text.split("\n")[0].strip()[:200]


class HLSLParser:
    """Regex-based HLSL parser for extracting shaders, structs, resources, and functions."""

    def parse(self, source: str, *, include_bodies: bool = False) -> list[Symbol]:
        symbols: list[Symbol] = []
        try:
            occupied: list[tuple[int, int]] = []  # (start, end) of struct/cbuffer bodies
            lines = source.split("\n")
            self._parse_defines(source, symbols, lines)
            self._parse_cbuffers(source, symbols, occupied, lines)
            self._parse_resources(source, symbols, occupied, lines)
            self._parse_structs(source, symbols, occupied, lines)
            self._parse_functions(source, symbols, occupied, lines, include_bodies)
        except Exception as e:
            logger.warning(f"HLSL parse error (partial results returned): {e}")
        return symbols

    def _in_occupied(self, pos: int, occupied: list[tuple[int, int]]) -> bool:
        return any(start <= pos < end for start, end in occupied)

    def _find_doc_comment(self, lines: list[str], source: str, byte_offset: int) -> str:
        """Find /// or /** */ doc comments immediately preceding a declaration."""
        line_num = source[:byte_offset].count("\n")
        doc_lines: list[str] = []
        i = line_num - 1
        in_block = False
        while i >= 0:
            stripped = lines[i].strip()
            # Handle block comment end (*/)
            if stripped.endswith("*/"):
                in_block = True
                # Strip trailing */
                content = stripped[:-2].strip()
                if content.startswith("/**"):
                    content = content[3:].strip()
                elif content.startswith("/*"):
                    content = content[2:].strip()
                if content:
                    doc_lines.insert(0, content)
                i -= 1
                continue
            if in_block:
                if stripped.startswith("/**") or stripped.startswith("/*"):
                    content = stripped.lstrip("/**").lstrip("/*").strip()
                    if content:
                        doc_lines.insert(0, content)
                    in_block = False
                elif stripped.startswith("*"):
                    content = stripped.lstrip("*").strip()
                    if content:
                        doc_lines.insert(0, content)
                else:
                    doc_lines.insert(0, stripped)
                i -= 1
                continue
            if stripped.startswith("///"):
                doc_lines.insert(0, stripped[3:].strip())
                i -= 1
            elif not stripped:
                i -= 1  # skip blank lines between comment and declaration
            else:
                break
        return "\n".join(doc_lines)

    def _parse_defines(self, source: str, symbols: list[Symbol], lines: list[str]) -> None:
        for match in _DEFINE_RE.finditer(source):
            name = match.group(1)
            raw_value = match.group(2)
            # Join continuation lines (strip trailing backslashes)
            value = re.sub(r"\\\n\s*", " ", raw_value).strip()

            doc = self._find_doc_comment(lines, source, match.start())
            define_line = f"#define {name} {value}"
            content = f"`{define_line}`"
            if doc:
                content += f"\n\n{doc}"

            symbols.append(Symbol(
                name=name,
                chunk_type=ChunkType.CONSTANT,
                content=content,
                summary=_first_line(doc) or define_line,
                metadata={"value": value},
            ))

    def _parse_cbuffers(
        self,
        source: str,
        symbols: list[Symbol],
        occupied: list[tuple[int, int]],
        lines: list[str],
    ) -> None:
        for match in _CBUFFER_RE.finditer(source):
            name = match.group(1)
            register = match.group(2)  # e.g. "b0" or None

            # Find the opening brace
            open_brace = source.index("{", match.start())
            body, end_pos = _extract_brace_block(source, open_brace)
            if end_pos == -1:
                # Unbalanced — still emit a partial symbol but skip members
                occupied.append((match.start(), len(source)))
                body = ""
                end_pos = len(source)
            else:
                occupied.append((open_brace, end_pos))

            doc = self._find_doc_comment(lines, source, match.start())
            cbuffer_line = f"cbuffer {name}"
            if register:
                cbuffer_line += f" : register({register})"

            symbols.append(Symbol(
                name=name,
                chunk_type=ChunkType.SHADER_UNIFORM_BLOCK,
                content=f"`{cbuffer_line}`",
                summary=_first_line(doc) or cbuffer_line,
                metadata={"register": register},
            ))

            # Emit PROPERTY for each member inside the block
            for member in _MEMBER_RE.finditer(body):
                mtype = member.group(1).strip()
                mname = member.group(2).strip()
                semantic = member.group(3)
                meta: dict = {"type": mtype}
                if semantic:
                    meta["semantic"] = semantic
                symbols.append(Symbol(
                    name=mname,
                    chunk_type=ChunkType.PROPERTY,
                    content=f"`{mtype} {mname}`",
                    summary=f"{mtype} {mname}",
                    parent=name,
                    metadata=meta,
                ))

    def _parse_resources(
        self,
        source: str,
        symbols: list[Symbol],
        occupied: list[tuple[int, int]],
        lines: list[str],
    ) -> None:
        for match in _RESOURCE_RE.finditer(source):
            if self._in_occupied(match.start(), occupied):
                continue
            rtype = match.group(1).strip()
            rname = match.group(2).strip()
            register = match.group(3)  # e.g. "t0" or None

            doc = self._find_doc_comment(lines, source, match.start())
            decl = f"{rtype} {rname}"
            if register:
                decl += f" : register({register})"
            content = f"`{decl}`"
            if doc:
                content += f"\n\n{doc}"

            meta: dict = {"type": rtype}
            if register:
                meta["register"] = register

            symbols.append(Symbol(
                name=rname,
                chunk_type=ChunkType.SHADER_RESOURCE,
                content=content,
                summary=_first_line(doc) or decl,
                metadata=meta,
            ))

    def _parse_structs(
        self,
        source: str,
        symbols: list[Symbol],
        occupied: list[tuple[int, int]],
        lines: list[str],
    ) -> None:
        for match in _STRUCT_RE.finditer(source):
            name = match.group(1)

            open_brace = source.index("{", match.start())
            body, end_pos = _extract_brace_block(source, open_brace)
            if end_pos == -1:
                occupied.append((match.start(), len(source)))
                body = ""
            else:
                occupied.append((open_brace, end_pos))

            doc = self._find_doc_comment(lines, source, match.start())

            # Build single-line struct summary for content
            struct_sig = f"struct {name} {{ ... }}"
            content = f"```hlsl\n{struct_sig}\n```"
            if doc:
                content += f"\n\n{doc}"

            symbols.append(Symbol(
                name=name,
                chunk_type=ChunkType.CLASS_OVERVIEW,
                content=content,
                summary=_first_line(doc) or struct_sig,
            ))

            # Emit PROPERTY for each member
            for member in _MEMBER_RE.finditer(body):
                mtype = member.group(1).strip()
                mname = member.group(2).strip()
                semantic = member.group(3)
                meta: dict = {"type": mtype}
                if semantic:
                    meta["semantic"] = semantic
                symbols.append(Symbol(
                    name=mname,
                    chunk_type=ChunkType.PROPERTY,
                    content=f"`{mtype} {mname}`",
                    summary=f"{mtype} {mname}",
                    parent=name,
                    metadata=meta,
                ))

    def _parse_functions(
        self,
        source: str,
        symbols: list[Symbol],
        occupied: list[tuple[int, int]],
        lines: list[str],
        include_bodies: bool,
    ) -> None:
        for match in _FUNCTION_RE.finditer(source):
            if self._in_occupied(match.start(), occupied):
                continue

            ret_type = match.group(1).strip()
            name = match.group(2).strip()
            params_str = match.group(3).strip()
            trailing_semantic = match.group(4)  # e.g. "SV_Target" or None

            # Skip keywords that look like functions but aren't
            if ret_type in ("if", "for", "while", "switch", "else", "do"):
                continue

            # Check for [numthreads(...)] attribute before this match
            prefix = source[:match.start()]
            numthreads_match = _NUMTHREADS_RE.search(prefix)
            # Only use numthreads if it appears close to this function (after last })
            numthreads = None
            if numthreads_match:
                last_close = prefix.rfind("}")
                if numthreads_match.start() > last_close:
                    x = int(numthreads_match.group(1))
                    y = int(numthreads_match.group(2))
                    z = int(numthreads_match.group(3))
                    numthreads = (x, y, z)

            # Determine entry point
            is_entry = (
                name == "main"
                or numthreads is not None
                or (trailing_semantic is not None and trailing_semantic in _SV_ENTRY_SEMANTICS)
            )

            # Build signature string
            sig = f"{ret_type} {name}({params_str})"
            if trailing_semantic:
                sig += f" : {trailing_semantic}"

            doc = self._find_doc_comment(lines, source, match.start())

            # Build content
            if include_bodies:
                # Extract full function body
                open_brace_pos = match.end() - 1
                # match ends just past '{', so go back one char
                # Actually _FUNCTION_RE ends with '\{' so match.end() points after '{'
                # We need to find the '{' at or just before match.end()
                brace_search_start = match.start()
                open_brace_pos = source.rindex("{", brace_search_start, match.end())
                body_content, _ = _extract_brace_block(source, open_brace_pos)
                full_text = source[match.start():match.end()] + body_content + "}"
                content = f"```hlsl\n{full_text}\n```"
            else:
                content = f"```hlsl\n{sig}\n```"
                if doc:
                    content += f"\n\n{doc}"

            # Parse params into list
            param_list = []
            if params_str:
                for p in params_str.split(","):
                    p = p.strip()
                    if not p:
                        continue
                    # e.g. "float3 pos" or "VSOutput input" or "uint3 id : SV_DispatchThreadID"
                    # Strip semantic if present
                    p_no_semantic = p.split(":")[0].strip()
                    parts = p_no_semantic.rsplit(None, 1)
                    if len(parts) == 2:
                        param_list.append({"name": parts[1], "type": parts[0]})
                    else:
                        param_list.append({"name": p_no_semantic, "type": None})

            meta: dict = {"return_type": ret_type}
            if param_list:
                meta["params"] = param_list
            if is_entry:
                if numthreads:
                    meta["numthreads"] = numthreads
                if trailing_semantic:
                    meta["semantic"] = trailing_semantic
            else:
                if trailing_semantic:
                    meta["semantic"] = trailing_semantic

            chunk_type = ChunkType.SHADER_ENTRY_POINT if is_entry else ChunkType.METHOD

            symbols.append(Symbol(
                name=name,
                chunk_type=chunk_type,
                content=content,
                summary=_first_line(doc) or sig,
                metadata=meta,
            ))

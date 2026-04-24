from __future__ import annotations

import logging
import re
from typing import Any

from glyph.chunkers._parsers import Symbol
from glyph.domain.models import ChunkType

logger = logging.getLogger(__name__)

_GODOT_ENTRY_POINTS = frozenset({"main", "vertex", "fragment", "light", "start", "process"})

_SAMPLER_IMAGE_TYPES = frozenset({
    "sampler1D", "sampler2D", "sampler3D", "samplerCube",
    "sampler2DArray", "samplerCubeArray", "sampler2DShadow", "sampler2DMS",
    "image1D", "image2D", "image3D", "imageCube",
    "isampler2D", "usampler2D", "iimage2D", "uimage2D",
    "sampler2DArrayShadow", "samplerCubeShadow",
})

# #version 450 [core]
_VERSION_RE = re.compile(r"^#version\s+(\d+)(?:\s+(\w+))?", re.MULTILINE)

# shader_type canvas_item;
_SHADER_TYPE_RE = re.compile(r"^shader_type\s+(\w+)\s*;", re.MULTILINE)

# render_mode blend_mix, unshaded;
_RENDER_MODE_RE = re.compile(r"^render_mode\s+([^;]+);", re.MULTILINE)

# layout(std140, binding = 0) uniform BlockName {
# Must be followed by an identifier then '{', not a type+name (standalone uniform)
_UNIFORM_BLOCK_RE = re.compile(
    r"^(?:layout\s*\(([^)]*)\)\s+)?uniform\s+(\w+)\s*\{",
    re.MULTILINE,
)

# [layout(...)] uniform [precision] Type name [array] [: godot_hint] [= default];
_STANDALONE_UNIFORM_RE = re.compile(
    r"^(?:layout\s*\(([^)]*)\)\s+)?uniform\s+"
    r"(?:(lowp|mediump|highp)\s+)?"
    r"(\w+)\s+(\w+)"
    r"(?:\s*\[([^\]]*)\])?"
    r"(?:\s*:\s*(\w+(?:\([^)]*\))?))?"
    r"(?:\s*=\s*([^;]+))?\s*;",
    re.MULTILINE,
)

# [layout(...)] in/out Type name;
_IN_OUT_RE = re.compile(
    r"^(?:layout\s*\(([^)]*)\)\s+)?(in|out)\s+(\w+)\s+(\w+)\s*;",
    re.MULTILINE,
)

# struct Name {
_STRUCT_RE = re.compile(r"^struct\s+(\w+)\s*\{", re.MULTILINE)

# ReturnType name(params) {
_FUNCTION_RE = re.compile(
    r"^(\w+)\s+(\w+)\s*\(([^)]*)\)\s*\{",
    re.MULTILINE,
)

# layout(local_size_x = X [, local_size_y = Y] [, local_size_z = Z]) in;
_LOCAL_SIZE_RE = re.compile(
    r"layout\s*\([^)]*?"
    r"local_size_x\s*=\s*(\d+)"
    r"(?:[^)]*?local_size_y\s*=\s*(\d+))?"
    r"(?:[^)]*?local_size_z\s*=\s*(\d+))?"
    r"[^)]*\)\s*in\s*;",
    re.MULTILINE | re.DOTALL,
)

# struct/uniform block member: [precision] Type name [array];
# TODO: does not handle multiple declarators (float a, b, c;)
_MEMBER_RE = re.compile(
    r"^\s+"
    r"(?:(?:lowp|mediump|highp)\s+)?"
    r"(\w[\w<>]*(?:\s*<[^>]*>)?)\s+"
    r"(\w+)"
    r"(?:\s*\[([^\]]*)\])?"
    r"\s*;",
    re.MULTILINE,
)


def _find_top_level_brace_ranges(source: str) -> list[tuple[int, int]]:
    """Return (open_brace_pos, close_brace_pos) for all depth-0 brace blocks."""
    # TODO: does not handle #define macros whose body contains literal { or }.
    # Multi-line #define FOO(x) \ ... { ... } will corrupt depth tracking.
    # Rare in GLSL, occasional in UE USF. If this causes false positives
    # during ingestion, add #-directive skipping with line-continuation support.
    ranges = []
    depth = 0
    open_pos = -1
    i = 0
    in_line_comment = False
    in_block_comment = False
    in_string = False
    while i < len(source):
        ch = source[i]
        nxt = source[i + 1] if i + 1 < len(source) else ""
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
        elif in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 1
        elif in_string:
            if ch == "\\" and nxt:
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == "/" and nxt == "/":
            in_line_comment = True
            i += 1
        elif ch == "/" and nxt == "*":
            in_block_comment = True
            i += 1
        elif ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                open_pos = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and open_pos >= 0:
                ranges.append((open_pos, i))
                open_pos = -1
        i += 1
    return ranges


def _extract_brace_block(source: str, open_pos: int) -> tuple[str, int]:
    """Returns (content_between_braces, position_after_closing_brace).
    Returns ("", -1) if unbalanced.
    """
    depth = 0
    i = open_pos
    start = -1
    while i < len(source):
        if source[i] == "{":
            depth += 1
            if depth == 1:
                start = i + 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start:i], i + 1
        i += 1
    return "", -1


def _first_line(text: str) -> str:
    if not text:
        return ""
    return text.split("\n")[0].strip()[:200]


class GLSLParser:
    """Regex-based GLSL/Godot shader parser."""

    def parse(self, source: str, *, include_bodies: bool = False) -> list[Symbol]:
        symbols: list[Symbol] = []
        file_metadata: dict[str, Any] = {}
        try:
            self._extract_file_metadata(source, file_metadata)
            local_size: tuple[int, int, int] | None = None
            ls_m = _LOCAL_SIZE_RE.search(source)
            if ls_m:
                x = int(ls_m.group(1))
                y = int(ls_m.group(2)) if ls_m.group(2) else 1
                z = int(ls_m.group(3)) if ls_m.group(3) else 1
                local_size = (x, y, z)
            top_level_ranges = _find_top_level_brace_ranges(source)
            self._parse_structs(source, symbols, top_level_ranges)
            self._parse_uniform_blocks(source, symbols, top_level_ranges)
            self._parse_standalone_uniforms(source, symbols, top_level_ranges)
            self._parse_in_out(source, symbols, top_level_ranges)
            self._parse_functions(source, symbols, top_level_ranges, include_bodies, local_size)
        except Exception as e:
            logger.warning(f"GLSL parse error (partial results returned): {e}")
        # file-level context attached to every symbol for stable filterability
        if file_metadata:
            for sym in symbols:
                sym.metadata["file"] = dict(file_metadata)
        return symbols

    def _extract_file_metadata(self, source: str, meta: dict[str, Any]) -> None:
        m = _VERSION_RE.search(source)
        if m:
            meta["version"] = m.group(1)
        m = _SHADER_TYPE_RE.search(source)
        if m:
            meta["shader_type"] = m.group(1)
        m = _RENDER_MODE_RE.search(source)
        if m:
            meta["render_mode"] = m.group(1).strip()

    def _find_doc_comment(self, lines: list[str], source: str, byte_offset: int) -> str:
        line_num = source[:byte_offset].count("\n")
        doc_lines: list[str] = []
        i = line_num - 1
        while i >= 0:
            stripped = lines[i].strip()
            if stripped.startswith("///"):
                doc_lines.insert(0, stripped[3:].strip())
                i -= 1
            elif not stripped:
                i -= 1
            else:
                break
        return "\n".join(doc_lines)

    def _parse_structs(
        self,
        source: str,
        symbols: list[Symbol],
        top_level_ranges: list[tuple[int, int]],
    ) -> None:
        lines = source.split("\n")
        for m in _STRUCT_RE.finditer(source):
            if any(s < m.start() < e for s, e in top_level_ranges):
                continue
            name = m.group(1)
            brace_pos = source.index("{", m.start())
            body, end_pos = _extract_brace_block(source, brace_pos)
            if end_pos == -1:
                continue
            sig = f"struct {name} {{ ... }}"
            doc = self._find_doc_comment(lines, source, m.start())
            symbols.append(Symbol(
                name=name,
                chunk_type=ChunkType.CLASS_OVERVIEW,
                content=f"```glsl\n{sig}\n```" + (f"\n\n{doc}" if doc else ""),
                summary=_first_line(doc) or f"Struct {name}",
            ))
            for mm in _MEMBER_RE.finditer(body):
                mtype, mname = mm.group(1), mm.group(2)
                array_size = mm.group(3)
                meta: dict[str, Any] = {"type": mtype}
                if array_size is not None:
                    meta["array_size"] = array_size.strip()
                symbols.append(Symbol(
                    name=mname,
                    chunk_type=ChunkType.PROPERTY,
                    content=f"`{mtype} {mname}`",
                    summary=f"{mtype} {mname}",
                    parent=name,
                    metadata=meta,
                ))

    def _parse_uniform_blocks(
        self,
        source: str,
        symbols: list[Symbol],
        top_level_ranges: list[tuple[int, int]],
    ) -> None:
        lines = source.split("\n")
        for m in _UNIFORM_BLOCK_RE.finditer(source):
            if any(s < m.start() < e for s, e in top_level_ranges):
                continue
            layout_str = (m.group(1) or "").strip()
            name = m.group(2)
            brace_pos = source.index("{", m.start())
            body, end_pos = _extract_brace_block(source, brace_pos)
            if end_pos == -1:
                continue
            doc = self._find_doc_comment(lines, source, m.start())
            symbols.append(Symbol(
                name=name,
                chunk_type=ChunkType.SHADER_UNIFORM_BLOCK,
                content=f"`uniform {name} {{ ... }}`",
                summary=_first_line(doc) or f"Uniform block {name}",
                metadata={"layout": layout_str},
            ))
            for mm in _MEMBER_RE.finditer(body):
                mtype, mname = mm.group(1), mm.group(2)
                array_size = mm.group(3)
                meta: dict[str, Any] = {"type": mtype}
                if array_size is not None:
                    meta["array_size"] = array_size.strip()
                symbols.append(Symbol(
                    name=mname,
                    chunk_type=ChunkType.PROPERTY,
                    content=f"`{mtype} {mname}`",
                    summary=f"{mtype} {mname}",
                    parent=name,
                    metadata=meta,
                ))

    def _parse_standalone_uniforms(
        self,
        source: str,
        symbols: list[Symbol],
        top_level_ranges: list[tuple[int, int]],
    ) -> None:
        for m in _STANDALONE_UNIFORM_RE.finditer(source):
            if any(s < m.start() < e for s, e in top_level_ranges):
                continue
            layout_str = (m.group(1) or "").strip()
            precision = m.group(2)
            utype = m.group(3)
            uname = m.group(4)
            array_size = m.group(5)
            godot_hint = m.group(6)
            default_val = (m.group(7) or "").strip() or None

            meta: dict[str, Any] = {"type": utype}
            if precision:
                meta["precision"] = precision
            if layout_str:
                meta["layout"] = layout_str
            if array_size is not None:
                meta["array_size"] = array_size.strip()
            if godot_hint:
                meta["godot_hint"] = godot_hint.strip()
            if default_val:
                meta["default"] = default_val

            is_resource = utype in _SAMPLER_IMAGE_TYPES
            chunk_type = ChunkType.SHADER_RESOURCE if is_resource else ChunkType.PROPERTY
            sig = f"uniform {utype} {uname}"
            symbols.append(Symbol(
                name=uname,
                chunk_type=chunk_type,
                content=f"`{sig}`",
                summary=sig,
                metadata=meta,
            ))

    def _parse_in_out(
        self,
        source: str,
        symbols: list[Symbol],
        top_level_ranges: list[tuple[int, int]],
    ) -> None:
        for m in _IN_OUT_RE.finditer(source):
            if any(s < m.start() < e for s, e in top_level_ranges):
                continue
            layout_str = (m.group(1) or "").strip()
            qualifier = m.group(2)
            vtype = m.group(3)
            vname = m.group(4)
            meta: dict[str, Any] = {"qualifier": qualifier, "type": vtype}
            if layout_str:
                meta["layout"] = layout_str
            symbols.append(Symbol(
                name=vname,
                chunk_type=ChunkType.PROPERTY,
                content=f"`{qualifier} {vtype} {vname}`",
                summary=f"{qualifier} {vtype} {vname}",
                metadata=meta,
            ))

    def _parse_functions(
        self,
        source: str,
        symbols: list[Symbol],
        top_level_ranges: list[tuple[int, int]],
        include_bodies: bool,
        local_size: tuple[int, int, int] | None,
    ) -> None:
        lines = source.split("\n")
        for m in _FUNCTION_RE.finditer(source):
            if any(s < m.start() < e for s, e in top_level_ranges):
                continue
            ret = m.group(1)
            name = m.group(2)
            params = m.group(3)

            if ret in ("if", "for", "while", "switch", "else", "do"):
                continue

            is_entry = name in _GODOT_ENTRY_POINTS or (local_size is not None and name == "main")
            chunk_type = ChunkType.SHADER_ENTRY_POINT if is_entry else ChunkType.METHOD

            sig = f"{ret} {name}({params})"
            doc = self._find_doc_comment(lines, source, m.start())

            brace_pos = source.index("{", m.start())
            _, end_pos = _extract_brace_block(source, brace_pos)

            if include_bodies:
                full_src = source[m.start():end_pos] if end_pos != -1 else sig
                content = f"```glsl\n{full_src}\n```"
            else:
                content = f"```glsl\n{sig}\n```"
                if doc:
                    content += f"\n\n{doc}"

            meta: dict[str, Any] = {"return_type": ret}
            if local_size is not None and name == "main":
                meta["local_size"] = list(local_size)
                meta["is_compute"] = True

            symbols.append(Symbol(
                name=name,
                chunk_type=chunk_type,
                content=content,
                summary=_first_line(doc) or sig,
                metadata=meta,
            ))

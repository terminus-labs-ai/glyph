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

# [layout(...)] uniform Type name [: godot_hint] [= default];
# The key distinction from uniform blocks: has TWO identifiers after 'uniform' (type + name)
_STANDALONE_UNIFORM_RE = re.compile(
    r"^(?:layout\s*\(([^)]*)\)\s+)?uniform\s+(\w+)\s+(\w+)"
    r"(?:\s*:\s*([\w]+(?:\([^)]*\))?))?(?:\s*=\s*([^;]+))?\s*;",
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

# layout(local_size_x = X, ...) in;
_LOCAL_SIZE_RE = re.compile(
    r"layout\s*\([^)]*local_size_x\s*=\s*(\d+)[^)]*\)\s*in\s*;",
    re.MULTILINE,
)

# struct/uniform block member: Type name;
_MEMBER_RE = re.compile(r"^\s+(\w[\w<>]*(?:\s*<[^>]*>)?)\s+(\w+)\s*;", re.MULTILINE)


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
            is_compute = bool(_LOCAL_SIZE_RE.search(source))
            occupied: list[tuple[int, int]] = []
            self._parse_structs(source, symbols, occupied)
            self._parse_uniform_blocks(source, symbols, occupied)
            self._parse_standalone_uniforms(source, symbols, occupied)
            self._parse_in_out(source, symbols, occupied)
            self._parse_functions(source, symbols, occupied, include_bodies, is_compute)
        except Exception as e:
            logger.warning(f"GLSL parse error (partial results returned): {e}")
        # Attach file-level metadata to first symbol
        if symbols and file_metadata:
            symbols[0].metadata.update(file_metadata)
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
        occupied: list[tuple[int, int]],
    ) -> None:
        lines = source.split("\n")
        for m in _STRUCT_RE.finditer(source):
            name = m.group(1)
            brace_pos = source.index("{", m.start())
            body, end_pos = _extract_brace_block(source, brace_pos)
            if end_pos == -1:
                continue
            occupied.append((m.start(), end_pos))
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
                symbols.append(Symbol(
                    name=mname,
                    chunk_type=ChunkType.PROPERTY,
                    content=f"`{mtype} {mname}`",
                    summary=f"{mtype} {mname}",
                    parent=name,
                    metadata={"type": mtype},
                ))

    def _parse_uniform_blocks(
        self,
        source: str,
        symbols: list[Symbol],
        occupied: list[tuple[int, int]],
    ) -> None:
        lines = source.split("\n")
        for m in _UNIFORM_BLOCK_RE.finditer(source):
            if any(s <= m.start() <= e for s, e in occupied):
                continue
            layout_str = (m.group(1) or "").strip()
            name = m.group(2)
            brace_pos = source.index("{", m.start())
            body, end_pos = _extract_brace_block(source, brace_pos)
            if end_pos == -1:
                continue
            occupied.append((m.start(), end_pos))
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
                symbols.append(Symbol(
                    name=mname,
                    chunk_type=ChunkType.PROPERTY,
                    content=f"`{mtype} {mname}`",
                    summary=f"{mtype} {mname}",
                    parent=name,
                    metadata={"type": mtype},
                ))

    def _parse_standalone_uniforms(
        self,
        source: str,
        symbols: list[Symbol],
        occupied: list[tuple[int, int]],
    ) -> None:
        for m in _STANDALONE_UNIFORM_RE.finditer(source):
            if any(s <= m.start() <= e for s, e in occupied):
                continue
            layout_str = (m.group(1) or "").strip()
            utype = m.group(2)
            uname = m.group(3)
            godot_hint = m.group(4)
            default_val = (m.group(5) or "").strip() or None

            meta: dict[str, Any] = {"type": utype}
            if layout_str:
                meta["layout"] = layout_str
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
        occupied: list[tuple[int, int]],
    ) -> None:
        for m in _IN_OUT_RE.finditer(source):
            if any(s <= m.start() <= e for s, e in occupied):
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
        occupied: list[tuple[int, int]],
        include_bodies: bool,
        is_compute: bool,
    ) -> None:
        lines = source.split("\n")
        for m in _FUNCTION_RE.finditer(source):
            if any(s <= m.start() <= e for s, e in occupied):
                continue
            ret = m.group(1)
            name = m.group(2)
            params = m.group(3)

            # Skip control flow keywords that match the pattern
            if ret in ("if", "for", "while", "switch", "else", "do"):
                continue

            is_entry = name in _GODOT_ENTRY_POINTS or (is_compute and name == "main")
            chunk_type = ChunkType.SHADER_ENTRY_POINT if is_entry else ChunkType.METHOD

            sig = f"{ret} {name}({params})"
            doc = self._find_doc_comment(lines, source, m.start())

            brace_pos = source.index("{", m.start())
            body_content, end_pos = _extract_brace_block(source, brace_pos)
            if end_pos != -1:
                occupied.append((m.start(), end_pos))

            if include_bodies:
                full_src = source[m.start():end_pos] if end_pos != -1 else sig
                content = f"```glsl\n{full_src}\n```"
            else:
                content = f"```glsl\n{sig}\n```"
                if doc:
                    content += f"\n\n{doc}"

            symbols.append(Symbol(
                name=name,
                chunk_type=chunk_type,
                content=content,
                summary=_first_line(doc) or sig,
                metadata={"return_type": ret},
            ))

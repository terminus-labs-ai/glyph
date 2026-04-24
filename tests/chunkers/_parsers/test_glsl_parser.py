from __future__ import annotations

import pytest

from glyph.chunkers._parsers.glsl_parser import GLSLParser
from glyph.domain.models import ChunkType


def _find(symbols, name, parent=None):
    for s in symbols:
        if s.name == name and s.parent == parent:
            return s
    names = [(s.name, s.parent) for s in symbols]
    raise AssertionError(f"Symbol {name!r} (parent={parent!r}) not found in {names}")


def _parse(source: str) -> list:
    return GLSLParser().parse(source)


# ── False positives inside function bodies ────────────────────────────


def test_function_not_matched_in_body():
    """A nested function-like pattern at column 0 inside braces must not
    produce a second function symbol.  `vec4 helper(float x) {` inside main's
    body matches _FUNCTION_RE but should be excluded by brace-depth tracking."""
    source = (
        "void main() {\n"
        "vec4 helper(float x) {\n"
        "    return vec4(x);\n"
        "}\n"
        "}\n"
    )
    symbols = _parse(source)
    func_symbols = [
        s for s in symbols
        if s.chunk_type in (ChunkType.METHOD, ChunkType.SHADER_ENTRY_POINT)
    ]
    assert len(func_symbols) == 1, (
        f"Expected exactly 1 function symbol (main), got {len(func_symbols)}: "
        f"{[(s.name, s.chunk_type) for s in func_symbols]}"
    )
    assert func_symbols[0].name == "main"


# ── Compute shader local_size ─────────────────────────────────────────


def test_compute_shader_local_size():
    """layout(local_size_x=...) should populate metadata on the entry point."""
    source = (
        "layout(local_size_x=8, local_size_y=4, local_size_z=1) in;\n"
        "void main() { }\n"
    )
    symbols = _parse(source)
    main = _find(symbols, "main")
    assert main.chunk_type == ChunkType.SHADER_ENTRY_POINT
    assert main.metadata.get("is_compute") is True
    assert main.metadata.get("local_size") == [8, 4, 1]


# ── Array uniforms ────────────────────────────────────────────────────


def test_array_uniform():
    """Array uniforms should capture the array size expression in metadata."""
    source = "uniform vec4 positions[MAX_LIGHTS];\n"
    symbols = _parse(source)
    pos = _find(symbols, "positions")
    assert pos.chunk_type == ChunkType.PROPERTY
    assert pos.metadata.get("array_size") == "MAX_LIGHTS"


# ── Uniform block array member ────────────────────────────────────────


def test_uniform_block_array_member():
    """Array members inside a uniform block should have array_size metadata."""
    source = (
        "layout(std140) uniform LightBlock {\n"
        "    vec4 colors[4];\n"
        "};\n"
    )
    symbols = _parse(source)
    colors = _find(symbols, "colors", parent="LightBlock")
    assert colors.chunk_type == ChunkType.PROPERTY
    assert colors.metadata.get("array_size") == "4"


# ── File metadata propagation ─────────────────────────────────────────


def test_file_metadata_on_all_symbols():
    """File-level metadata (shader_type, etc.) must be present on EVERY symbol,
    not just the first one."""
    source = (
        "shader_type canvas_item;\n"
        "uniform float speed = 1.0;\n"
        "void fragment() { }\n"
    )
    symbols = _parse(source)
    assert len(symbols) >= 2, "Expected at least 2 symbols (uniform + function)"
    for s in symbols:
        file_meta = s.metadata.get("file", {})
        assert file_meta.get("shader_type") == "canvas_item", (
            f"Symbol {s.name!r} missing file metadata: {s.metadata}"
        )


# ── Precision-qualified uniforms ──────────────────────────────────────


def test_precision_qualified_uniform():
    """Precision qualifiers (lowp/mediump/highp) before the type should be
    captured separately from the base type."""
    source = "uniform highp sampler2D tex;\n"
    symbols = _parse(source)
    tex = _find(symbols, "tex")
    assert tex.chunk_type == ChunkType.SHADER_RESOURCE
    assert tex.metadata.get("type") == "sampler2D"
    assert tex.metadata.get("precision") == "highp"


# ── Nested function edge case ────────────────────────────────────────


def test_nested_function_like_code_not_matched():
    """The brace-range fix should reject function-like patterns inside other
    function bodies. This source is not valid GLSL but tests that the parser
    doesn't produce a spurious symbol for the inner block."""
    source = '''
void outer() {
    vec4 foo(vec2 uv) {
        return vec4(uv, 0.0, 1.0);
    }
}
'''
    parser = GLSLParser()
    symbols = parser.parse(source)
    function_symbols = [
        s for s in symbols
        if s.chunk_type in (ChunkType.METHOD, ChunkType.SHADER_ENTRY_POINT)
    ]
    assert len(function_symbols) == 1
    assert function_symbols[0].name == "outer"

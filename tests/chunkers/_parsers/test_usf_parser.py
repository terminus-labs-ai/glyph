from __future__ import annotations

import logging

import pytest

from glyph.chunkers._parsers.usf_parser import USFParser
from glyph.domain.models import ChunkType


def _parse(source: str) -> list:
    return USFParser().parse(source)


def _find(symbols, name, parent=None):
    for s in symbols:
        if s.name == name and s.parent == parent:
            return s
    names = [(s.name, s.parent) for s in symbols]
    raise AssertionError(f"Symbol {name!r} (parent={parent!r}) not found in {names}")


def test_shader_parameter_array_symbolic_count():
    """SHADER_PARAMETER_ARRAY with a #define count like [MAX_CASCADES] should
    store the raw string, not require a numeric literal."""
    source = (
        "BEGIN_SHADER_PARAMETER_STRUCT(FTest, )\n"
        "    SHADER_PARAMETER_ARRAY(float, Foo, [MAX_CASCADES])\n"
        "END_SHADER_PARAMETER_STRUCT()\n"
    )
    symbols = _parse(source)
    foo = _find(symbols, "Foo", parent="FTest")
    assert foo.chunk_type == ChunkType.PROPERTY
    assert foo.metadata.get("array_count") == "MAX_CASCADES"
    assert "array_count_int" not in foo.metadata


def test_unterminated_parameter_struct_logs_warning(caplog):
    """An unterminated BEGIN_SHADER_PARAMETER_STRUCT should emit a WARNING log
    and still produce a partial symbol with no parameter children."""
    caplog.set_level(logging.WARNING, logger="glyph.chunkers._parsers.usf_parser")
    source = (
        "BEGIN_SHADER_PARAMETER_STRUCT(FBroken, )\n"
        "    SHADER_PARAMETER(float, X)\n"
    )
    symbols = _parse(source)

    # Partial symbol emitted
    broken = _find(symbols, "FBroken")
    assert broken.chunk_type == ChunkType.SHADER_UNIFORM_BLOCK

    # No parameter children extracted
    children = [s for s in symbols if s.parent == "FBroken"]
    assert len(children) == 0

    # Warning was logged
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("Unterminated" in r.message and "FBroken" in r.message for r in warnings)

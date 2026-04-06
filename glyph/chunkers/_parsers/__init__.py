from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from glyph.domain.models import ChunkType


@dataclass
class Symbol:
    """A parsed code symbol (class, function, method, etc.)."""
    name: str
    chunk_type: ChunkType
    content: str  # formatted representation
    summary: str  # one-liner
    parent: str | None = None  # enclosing class name, if any
    metadata: dict[str, Any] = field(default_factory=dict)


class LanguageParser(Protocol):
    def parse(self, source: str, *, include_bodies: bool = False) -> list[Symbol]:
        ...


def get_parser(language: str) -> LanguageParser | None:
    if language == "python":
        from glyph.chunkers._parsers.python_parser import PythonParser
        return PythonParser()
    elif language == "gdscript":
        from glyph.chunkers._parsers.gdscript_parser import GDScriptParser
        return GDScriptParser()
    elif language == "typescript":
        from glyph.chunkers._parsers.typescript_parser import TypeScriptParser
        return TypeScriptParser()
    elif language == "tsx":
        from glyph.chunkers._parsers.typescript_parser import TypeScriptParser
        return TypeScriptParser(tsx=True)
    elif language == "rust":
        from glyph.chunkers._parsers.rust_parser import RustParser
        return RustParser()
    elif language == "go":
        from glyph.chunkers._parsers.go_parser import GoParser
        return GoParser()
    elif language == "cpp":
        from glyph.chunkers._parsers.cpp_parser import CppParser
        return CppParser()
    return None

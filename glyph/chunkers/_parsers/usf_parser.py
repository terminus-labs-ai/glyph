from __future__ import annotations

import logging
import re
from typing import Any

from glyph.chunkers._parsers import Symbol
from glyph.chunkers._parsers.hlsl_parser import HLSLParser
from glyph.domain.models import ChunkType

logger = logging.getLogger(__name__)

# BEGIN_SHADER_PARAMETER_STRUCT(Name, API)
_PARAM_STRUCT_BEGIN_RE = re.compile(
    r"BEGIN_SHADER_PARAMETER_STRUCT\(\s*(\w+)\s*,\s*([^)]*)\)",
    re.MULTILINE,
)

# END_SHADER_PARAMETER_STRUCT()
_PARAM_STRUCT_END_RE = re.compile(r"END_SHADER_PARAMETER_STRUCT\(\)", re.MULTILINE)

# SHADER_PARAMETER(Type, Name)
# SHADER_PARAMETER_TEXTURE(TextureType, Name)
# SHADER_PARAMETER_SAMPLER(SamplerType, Name)
# SHADER_PARAMETER_ARRAY(Type, Name, [Count])
# etc.
_PARAM_MACRO_RE = re.compile(
    r"\b(SHADER_PARAMETER(?:_ARRAY|_STRUCT_REF|_TEXTURE|_SAMPLER"
    r"|_RDG_TEXTURE|_RDG_BUFFER_SRV)?)"
    r"\(\s*([\w<>]+)\s*,\s*(\w+)(?:\s*,\s*\[(\d+)\])?\s*\)",
    re.MULTILINE,
)


class USFParser:
    """
    USF (Unreal Engine Shader) parser.

    Composes with HLSLParser. Extracts UE-specific
    BEGIN/END_SHADER_PARAMETER_STRUCT regions first, then delegates the
    cleaned source to HLSLParser for defines, cbuffers, resources, and functions.
    """

    def __init__(self) -> None:
        self._hlsl = HLSLParser()

    def parse(self, source: str, *, include_bodies: bool = False) -> list[Symbol]:
        symbols: list[Symbol] = []
        try:
            cleaned = self._extract_parameter_structs(source, symbols)
            hlsl_symbols = self._hlsl.parse(cleaned, include_bodies=include_bodies)
            symbols.extend(hlsl_symbols)
        except Exception as e:
            logger.warning(f"USF parse error (partial results returned): {e}")
        return symbols

    def _extract_parameter_structs(self, source: str, symbols: list[Symbol]) -> str:
        """
        Find BEGIN/END_SHADER_PARAMETER_STRUCT regions, emit symbols, replace
        those regions with blank lines (preserving line count for HLSL offsets).
        Returns cleaned source string.
        """
        # Collect regions to blank out: list of (start, end) char offsets
        regions: list[tuple[int, int]] = []

        for begin_m in _PARAM_STRUCT_BEGIN_RE.finditer(source):
            struct_name = begin_m.group(1)

            # Find matching END after the BEGIN
            end_m = _PARAM_STRUCT_END_RE.search(source, begin_m.end())
            if not end_m:
                # Unterminated block — emit partial symbol, skip blanking
                symbols.append(Symbol(
                    name=struct_name,
                    chunk_type=ChunkType.SHADER_UNIFORM_BLOCK,
                    content=f"`BEGIN_SHADER_PARAMETER_STRUCT({struct_name}, ...)`",
                    summary=f"UE shader parameter struct: {struct_name}",
                    metadata={"ue_parameter_struct": True},
                ))
                continue

            region_start = begin_m.start()
            region_end = end_m.end()
            region = source[region_start:region_end]

            # Emit the struct block symbol
            symbols.append(Symbol(
                name=struct_name,
                chunk_type=ChunkType.SHADER_UNIFORM_BLOCK,
                content=f"`BEGIN_SHADER_PARAMETER_STRUCT({struct_name}, ...)`",
                summary=f"UE shader parameter struct: {struct_name}",
                metadata={"ue_parameter_struct": True},
            ))

            # Emit PROPERTY for each SHADER_PARAMETER* macro inside the region
            for param_m in _PARAM_MACRO_RE.finditer(region):
                macro_kind = param_m.group(1)
                param_type = param_m.group(2)
                param_name = param_m.group(3)
                array_count = param_m.group(4)

                meta: dict[str, Any] = {
                    "ue_parameter_kind": macro_kind,
                    "type": param_type,
                }
                if array_count:
                    meta["array_count"] = int(array_count)

                symbols.append(Symbol(
                    name=param_name,
                    chunk_type=ChunkType.PROPERTY,
                    content=f"`{macro_kind}({param_type}, {param_name})`",
                    summary=f"{macro_kind}: {param_type} {param_name}",
                    parent=struct_name,
                    metadata=meta,
                ))

            regions.append((region_start, region_end))

        # Replace each region with blank lines (same newline count → same line numbers)
        if not regions:
            return source

        # Build cleaned source by slicing around each region
        parts: list[str] = []
        prev = 0
        for start, end in regions:
            parts.append(source[prev:start])
            # Count newlines in the region and replace with same number of blank lines
            newline_count = source[start:end].count("\n")
            parts.append("\n" * newline_count)
            prev = end
        parts.append(source[prev:])
        return "".join(parts)

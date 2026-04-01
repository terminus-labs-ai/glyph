from __future__ import annotations

import re
from dataclasses import dataclass

from glyph.chunkers._parsers import Symbol
from glyph.domain.models import ChunkType

# GDScript patterns
CLASS_NAME_RE = re.compile(r"^class_name\s+(\w+)", re.MULTILINE)
EXTENDS_RE = re.compile(r"^extends\s+(\w+)", re.MULTILINE)
INNER_CLASS_RE = re.compile(r"^class\s+(\w+)(?:\s+extends\s+(\w+))?:", re.MULTILINE)
FUNC_RE = re.compile(
    r"^(?P<indent>\t*)func\s+(?P<name>\w+)\s*\((?P<params>[^)]*)\)"
    r"(?:\s*->\s*(?P<return>\w+))?\s*:",
    re.MULTILINE,
)
SIGNAL_RE = re.compile(r"^signal\s+(\w+)(?:\(([^)]*)\))?", re.MULTILINE)
VAR_RE = re.compile(
    r"^(?:@export[^\n]*\n)?(?P<indent>\t*)var\s+(?P<name>\w+)"
    r"(?:\s*:\s*(?P<type>\w+))?"
    r"(?:\s*=\s*(?P<default>[^\n]+))?",
    re.MULTILINE,
)
CONST_RE = re.compile(
    r"^const\s+(\w+)(?:\s*:\s*(\w+))?\s*=\s*(.+)",
    re.MULTILINE,
)
ENUM_RE = re.compile(r"^enum\s+(\w+)\s*\{([^}]*)\}", re.MULTILINE)
ANNOTATION_RE = re.compile(r"^(@\w+(?:\([^)]*\))?)", re.MULTILINE)
COMMENT_BLOCK_RE = re.compile(r"^##\s*(.*)", re.MULTILINE)


class GDScriptParser:
    """Regex-based GDScript parser for extracting classes, functions, signals, etc."""

    def parse(self, source: str, *, include_bodies: bool = False) -> list[Symbol]:
        symbols: list[Symbol] = []
        lines = source.split("\n")

        # File-level class info
        class_name_match = CLASS_NAME_RE.search(source)
        extends_match = EXTENDS_RE.search(source)
        file_class = class_name_match.group(1) if class_name_match else None
        extends = extends_match.group(1) if extends_match else None

        # Class overview from top-level doc comments
        top_doc = self._extract_top_doc(lines)

        if file_class or extends:
            sig = f"class_name {file_class}" if file_class else ""
            if extends:
                sig += f"\nextends {extends}" if sig else f"extends {extends}"

            content = f"```gdscript\n{sig}\n```"
            if top_doc:
                content += f"\n\n{top_doc}"

            symbols.append(Symbol(
                name=file_class or "(unnamed)",
                chunk_type=ChunkType.CLASS_OVERVIEW,
                content=content,
                summary=_first_line(top_doc) or f"GDScript class extending {extends or 'Object'}",
                metadata={"extends": extends} if extends else {},
            ))

        # Enums
        for match in ENUM_RE.finditer(source):
            name = match.group(1)
            body = match.group(2).strip()
            content = f"```gdscript\nenum {name} {{{body}}}\n```"
            symbols.append(Symbol(
                name=name,
                chunk_type=ChunkType.ENUM,
                content=content,
                summary=f"Enum {name}",
                parent=file_class,
                metadata={"values": [v.strip().split("=")[0].strip()
                                     for v in body.split(",") if v.strip()]},
            ))

        # Constants
        for match in CONST_RE.finditer(source):
            name, type_hint, value = match.group(1), match.group(2), match.group(3).strip()
            sig = f"const {name}"
            if type_hint:
                sig += f": {type_hint}"
            sig += f" = {value}"
            doc = self._find_doc_comment(lines, match.start())
            content = f"`{sig}`"
            if doc:
                content += f"\n\n{doc}"
            symbols.append(Symbol(
                name=name,
                chunk_type=ChunkType.CONSTANT,
                content=content,
                summary=doc or sig,
                parent=file_class,
                metadata={"value": value, "type": type_hint},
            ))

        # Signals
        for match in SIGNAL_RE.finditer(source):
            name = match.group(1)
            params = match.group(2) or ""
            sig = f"signal {name}({params})" if params else f"signal {name}"
            doc = self._find_doc_comment(lines, match.start())
            content = f"`{sig}`"
            if doc:
                content += f"\n\n{doc}"
            symbols.append(Symbol(
                name=name,
                chunk_type=ChunkType.SIGNAL,
                content=content,
                summary=doc or f"Signal: {name}",
                parent=file_class,
            ))

        # Variables / properties (top-level only, indent=0 or 1 tab)
        for match in VAR_RE.finditer(source):
            indent = match.group("indent")
            if len(indent) > 0:
                continue  # skip local vars
            name = match.group("name")
            type_hint = match.group("type") or ""
            default = match.group("default") or ""
            sig = f"var {name}"
            if type_hint:
                sig += f": {type_hint}"
            if default:
                sig += f" = {default.strip()}"

            doc = self._find_doc_comment(lines, match.start())
            # Check for @export annotation on preceding line
            line_num = source[:match.start()].count("\n")
            export = ""
            if line_num > 0:
                prev_line = lines[line_num - 1].strip()
                if prev_line.startswith("@export"):
                    export = prev_line

            content = f"`{sig}`"
            if export:
                content = f"`{export}`\n{content}"
            if doc:
                content += f"\n\n{doc}"

            symbols.append(Symbol(
                name=name,
                chunk_type=ChunkType.PROPERTY,
                content=content,
                summary=doc or sig,
                parent=file_class,
                metadata={"type": type_hint, "default": default.strip() or None,
                           "export": export or None},
            ))

        # Functions
        for match in FUNC_RE.finditer(source):
            indent = match.group("indent")
            name = match.group("name")
            params = match.group("params")
            ret = match.group("return")

            # Determine parent (top-level or inner class method)
            parent = file_class
            if len(indent) >= 1:
                # Could be method of inner class — for now attribute to file class
                pass

            sig = f"func {name}({params})"
            if ret:
                sig += f" -> {ret}"

            doc = self._find_doc_comment(lines, match.start())

            if include_bodies:
                body = self._extract_body(lines, match.start(), source)
                content = f"```gdscript\n{body}\n```"
            else:
                content = f"```gdscript\n{sig}:\n```"
                if doc:
                    content += f"\n\n{doc}"

            # Parse params into metadata
            param_list = []
            if params.strip():
                for p in params.split(","):
                    p = p.strip()
                    parts = p.split(":")
                    p_name = parts[0].strip()
                    p_type = parts[1].strip().split("=")[0].strip() if len(parts) > 1 else None
                    p_default = None
                    if "=" in p:
                        p_default = p.split("=", 1)[1].strip()
                    param_list.append({
                        "name": p_name,
                        "type": p_type,
                        "default": p_default,
                    })

            metadata = {}
            if ret:
                metadata["return_type"] = ret
            if param_list:
                metadata["params"] = param_list

            symbols.append(Symbol(
                name=name,
                chunk_type=ChunkType.METHOD,
                content=content,
                summary=_first_line(doc) or sig,
                parent=parent,
                metadata=metadata,
            ))

        return symbols

    def _extract_top_doc(self, lines: list[str]) -> str:
        """Extract doc comments (##) from the top of the file."""
        doc_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("##"):
                doc_lines.append(stripped[2:].strip())
            elif stripped.startswith("#") or not stripped:
                continue  # skip regular comments and blank lines at top
            else:
                break
        return "\n".join(doc_lines)

    def _find_doc_comment(self, lines: list[str], byte_offset: int) -> str:
        """Find ## doc comments immediately preceding a declaration."""
        # Count lines to get line number
        line_num = sum(1 for i, c in enumerate(("\n".join(lines))[:byte_offset]) if c == "\n")
        doc_lines = []
        i = line_num - 1
        while i >= 0:
            stripped = lines[i].strip()
            if stripped.startswith("##"):
                doc_lines.insert(0, stripped[2:].strip())
                i -= 1
            elif stripped.startswith("@"):
                # Annotation, skip and keep looking
                i -= 1
            else:
                break
        return "\n".join(doc_lines)

    def _extract_body(self, lines: list[str], byte_offset: int, source: str) -> str:
        """Extract a function body by indentation."""
        line_num = source[:byte_offset].count("\n")
        func_line = lines[line_num]
        base_indent = len(func_line) - len(func_line.lstrip("\t"))

        body_lines = [func_line]
        for i in range(line_num + 1, len(lines)):
            line = lines[i]
            if not line.strip():
                body_lines.append("")
                continue
            indent = len(line) - len(line.lstrip("\t"))
            if indent <= base_indent:
                break
            body_lines.append(line)

        return "\n".join(body_lines)


def _first_line(text: str) -> str:
    if not text:
        return ""
    return text.split("\n")[0].strip()[:200]

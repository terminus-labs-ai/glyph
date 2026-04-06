from __future__ import annotations

import hashlib
import logging
import re
import uuid
from pathlib import Path

from glyph.domain.models import DocType, Document

logger = logging.getLogger(__name__)

# RST heading underline characters, ordered by conventional precedence
RST_HEADING_CHARS = "=-~^"

# RST directives to strip entirely (directive + indented body)
_RST_DIRECTIVE_RE = re.compile(
    r"^\.\. (?:image|figure|toctree|contents|raw)::.*?(?:\n(?:[ \t]+.+|\s*$))*",
    re.MULTILINE,
)

# RST inline markup: :ref:`text <target>` → text, :doc:`path` → path
_RST_REF_RE = re.compile(r":(?:ref|doc|class):`([^<`]+?)(?:\s*<[^>]+>)?`")

# RST admonition directives (note, warning, tip, etc.) → keep content as blockquote
_RST_ADMONITION_RE = re.compile(
    r"^\.\. (note|warning|tip|important|caution|danger|seealso|hint|attention)::\s*\n"
    r"((?:[ \t]+.+\n?)*)",
    re.MULTILINE,
)

# Default extensions to ingest
DEFAULT_EXTENSIONS = [".md", ".rst", ".txt"]

# Default directories to skip
DEFAULT_EXCLUDE_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", "_build", "_static", "_templates"}


class DocsIngestor:
    """Walks a directory tree for documentation files (.md, .rst, .txt).

    Converts RST headings and inline markup to markdown so the TextChunker
    can split on headings. Produces TUTORIAL documents.
    """

    def __init__(
        self,
        path: str,
        source_id: uuid.UUID,
        *,
        extensions: list[str] | None = None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        exclude_dirs: list[str] | None = None,
    ):
        self._path = Path(path)
        self._source_id = source_id
        self._extensions = set(extensions or DEFAULT_EXTENSIONS)
        self._include = [re.compile(p) for p in (include_patterns or [])]
        self._exclude = [re.compile(p) for p in (exclude_patterns or [])]
        self._exclude_dirs = DEFAULT_EXCLUDE_DIRS | set(exclude_dirs or [])

    async def ingest(self) -> list[Document]:
        if not self._path.is_dir():
            raise FileNotFoundError(f"Docs directory not found: {self._path}")

        documents: list[Document] = []
        for file_path in sorted(self._path.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix not in self._extensions:
                continue
            if any(part in self._exclude_dirs for part in file_path.parts):
                continue
            if not self._should_include(file_path):
                continue

            doc = self._read_doc(file_path)
            if doc:
                documents.append(doc)

        logger.info(f"Ingested {len(documents)} docs from {self._path}")
        return documents

    def _should_include(self, path: Path) -> bool:
        path_str = str(path)
        if self._exclude:
            for pat in self._exclude:
                if pat.search(path_str):
                    return False
        if self._include:
            return any(pat.search(path_str) for pat in self._include)
        return True

    def _read_doc(self, file_path: Path) -> Document | None:
        try:
            raw = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return None

        if not raw.strip():
            return None

        # Skip index files that are just toctrees
        if file_path.name == "index.rst" and ".. toctree::" in raw and raw.count("\n") < 30:
            return None

        if file_path.suffix == ".rst":
            content = self._convert_rst(raw)
        else:
            content = raw

        if len(content.strip()) < 50:
            return None

        # Extract title from first heading or filename
        title = self._extract_title(content) or file_path.stem.replace("_", " ").title()

        # Build a relative path from the base directory for a cleaner document path
        try:
            rel_path = file_path.relative_to(self._path)
        except ValueError:
            rel_path = file_path

        content_hash = hashlib.md5(content.encode()).hexdigest()

        return Document(
            source_id=self._source_id,
            path=str(rel_path),
            title=title,
            doc_type=DocType.TUTORIAL,
            raw_content=content,
            content_hash=content_hash,
        )

    @staticmethod
    def _convert_rst(text: str) -> str:
        """Convert RST to markdown-ish text for heading-based chunking."""
        # Strip RST labels (.. _doc_label:)
        text = re.sub(r"^\.\. _[\w-]+:\s*\n", "", text, flags=re.MULTILINE)

        # Convert admonitions: keep content, prefix with bold label
        def _admonition_repl(m: re.Match) -> str:
            label = m.group(1).capitalize()
            body = re.sub(r"^[ \t]+", "", m.group(2), flags=re.MULTILINE).strip()
            return f"**{label}:** {body}\n"

        text = _RST_ADMONITION_RE.sub(_admonition_repl, text)

        # Strip image/toctree/figure directives
        text = _RST_DIRECTIVE_RE.sub("", text)

        # Convert RST inline refs to plain text
        text = _RST_REF_RE.sub(r"\1", text)

        # Strip remaining simple directives (.. code-block::, etc.) but keep indented content
        text = re.sub(r"^\.\. [\w-]+::\s*.*$", "", text, flags=re.MULTILINE)

        # Convert RST headings (underlined) to markdown headings
        lines = text.split("\n")
        result: list[str] = []
        i = 0
        while i < len(lines):
            # Check for underlined heading: current line is text, next line is all same char
            if (
                i + 1 < len(lines)
                and lines[i].strip()
                and not lines[i].startswith(" ")
                and len(lines[i + 1].strip()) >= 3
                and _is_rst_underline(lines[i + 1])
            ):
                heading_text = lines[i].strip()
                underline_char = lines[i + 1].strip()[0]
                level = RST_HEADING_CHARS.index(underline_char) + 1 if underline_char in RST_HEADING_CHARS else 3
                result.append(f"\n{'#' * level} {heading_text}\n")
                i += 2
            # Check for overlined heading: underline, text, underline
            elif (
                i + 2 < len(lines)
                and _is_rst_underline(lines[i])
                and lines[i + 1].strip()
                and _is_rst_underline(lines[i + 2])
                and lines[i].strip()[0] == lines[i + 2].strip()[0]
            ):
                heading_text = lines[i + 1].strip()
                underline_char = lines[i].strip()[0]
                level = RST_HEADING_CHARS.index(underline_char) + 1 if underline_char in RST_HEADING_CHARS else 1
                result.append(f"\n{'#' * level} {heading_text}\n")
                i += 3
            else:
                result.append(lines[i])
                i += 1

        text = "\n".join(result)

        # Strip double-backtick inline code to single backtick
        text = text.replace("``", "`")

        # Clean up excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    @staticmethod
    def _extract_title(content: str) -> str | None:
        """Extract the first markdown heading as the document title."""
        m = re.search(r"^#{1,3}\s+(.+)$", content, re.MULTILINE)
        if m:
            return m.group(1).strip()
        # Fallback: first non-empty line
        for line in content.split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                return line[:100]
        return None


def _is_rst_underline(line: str) -> bool:
    """Check if a line is an RST heading underline (all same punctuation char)."""
    stripped = line.strip()
    if not stripped or len(stripped) < 3:
        return False
    char = stripped[0]
    if char.isalnum() or char.isspace():
        return False
    return all(c == char for c in stripped)

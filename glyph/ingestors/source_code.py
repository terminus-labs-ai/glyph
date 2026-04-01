from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path

from glyph.domain.models import DocType, Document

logger = logging.getLogger(__name__)

# Map file extensions to language identifiers
EXTENSION_MAP = {
    ".py": "python",
    ".gd": "gdscript",
}

# Files/dirs to always skip
SKIP_DIRS = {
    "__pycache__", ".git", ".venv", "venv", "node_modules",
    ".mypy_cache", ".pytest_cache", "build", "dist", ".eggs",
}

SKIP_FILES = {"__init__.py"}


class SourceCodeIngestor:
    """Walks a source tree and produces one Document per source file."""

    def __init__(
        self,
        path: str,
        source_id: uuid.UUID,
        *,
        extensions: list[str] | None = None,
        exclude_dirs: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ):
        self._path = Path(path)
        self._source_id = source_id
        self._extensions = set(extensions or EXTENSION_MAP.keys())
        self._exclude_dirs = SKIP_DIRS | set(exclude_dirs or [])
        self._exclude_patterns = exclude_patterns or []

    async def ingest(self) -> list[Document]:
        if not self._path.is_dir():
            raise FileNotFoundError(f"Source directory not found: {self._path}")

        documents = []
        for file_path in self._walk():
            doc = self._read_file(file_path)
            if doc:
                documents.append(doc)

        logger.info(f"Ingested {len(documents)} source files from {self._path}")
        return documents

    def _walk(self):
        for file_path in sorted(self._path.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix not in self._extensions:
                continue
            if file_path.name in SKIP_FILES:
                continue
            if any(d in file_path.parts for d in self._exclude_dirs):
                continue
            if any(p in str(file_path) for p in self._exclude_patterns):
                continue
            yield file_path

    def _read_file(self, file_path: Path) -> Document | None:
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            if not content.strip():
                return None

            rel_path = file_path.relative_to(self._path)
            lang = EXTENSION_MAP.get(file_path.suffix, "unknown")

            return Document(
                source_id=self._source_id,
                path=str(rel_path),
                title=str(rel_path),
                doc_type=DocType.CLASS_REF,
                raw_content=content,
                content_hash=hashlib.md5(content.encode()).hexdigest(),
            )
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return None

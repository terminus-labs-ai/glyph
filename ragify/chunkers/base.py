from __future__ import annotations

from typing import Protocol

from ragify.domain.models import Chunk, Document


class Chunker(Protocol):
    def chunk(self, document: Document) -> list[Chunk]:
        """Break a document into chunks."""
        ...

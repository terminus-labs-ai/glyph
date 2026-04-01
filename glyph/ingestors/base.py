from __future__ import annotations

from typing import Protocol

from glyph.domain.models import Document


class Ingestor(Protocol):
    async def ingest(self) -> list[Document]:
        """Ingest documents from the source and return them."""
        ...

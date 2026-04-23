from __future__ import annotations

from typing import Protocol


class Reranker(Protocol):
    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Return one relevance score per document, same order as input.

        Args:
            query: Search query string.
            documents: List of document texts to score.

        Returns:
            List of float scores, one per input document, in the same
            order as the input documents list.
        """
        ...

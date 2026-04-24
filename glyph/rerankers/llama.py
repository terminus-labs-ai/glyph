from __future__ import annotations

import logging

import aiohttp

logger = logging.getLogger(__name__)


class LlamaReranker:
    """Rerank documents via a local llama-server /v1/rerank endpoint."""

    def __init__(self, url: str, model: str, batch_size: int = 32, timeout: int = 30):
        self._url = url.rstrip("/")
        self._model = model
        self._batch_size = batch_size
        self._timeout = timeout

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Score documents against the query.

        Returns one score per document in the same order as ``documents``.
        """
        if not documents:
            return []

        scores: list[float] = [0.0] * len(documents)

        for i in range(0, len(documents), self._batch_size):
            batch = documents[i : i + self._batch_size]
            batch_scores = await self._rerank_batch(query, batch)
            # batch_scores is sorted by the API (by score desc)
            # Map back to input order
            for item in batch_scores:
                idx = item.get("index")
                score = item.get("relevance_score")
                if idx is None or score is None:
                    logger.warning("Skipping malformed rerank result (missing key): %s", item)
                    continue
                original_idx = i + idx
                scores[original_idx] = score

        return scores

    async def _rerank_batch(self, query: str, documents: list[str]) -> list[dict]:
        payload = {
            "model": self._model,
            "query": query,
            "documents": documents,
        }

        timeout = aiohttp.ClientTimeout(total=self._timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self._url}/v1/rerank",
                json=payload,
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"rerank endpoint returned {resp.status}")
                data = await resp.json()

        # Response format: {"results": [{"index": int, "relevance_score": float}, ...]}
        results = data.get("results", [])
        return results

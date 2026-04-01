from __future__ import annotations

import logging

import aiohttp

logger = logging.getLogger(__name__)


class LlamaEmbedder:
    """Generate embeddings via a local llama-server HTTP API."""

    def __init__(self, url: str, model: str, dims: int, batch_size: int = 5):
        self._url = url.rstrip("/")
        self._model = model
        self._dims = dims
        self._batch_size = batch_size

    @property
    def dimensions(self) -> int:
        return self._dims

    async def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            batch_embeddings = await self._embed_batch(batch)
            embeddings.extend(batch_embeddings)

        return embeddings

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Try common embedding API endpoints
        endpoints = [
            (f"{self._url}/v1/embeddings", self._payload_openai),
            (f"{self._url}/api/embeddings", self._payload_ollama),
            (f"{self._url}/embedding", self._payload_llama_cpp),
        ]

        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for endpoint, payload_fn in endpoints:
                try:
                    payload = payload_fn(texts)
                    async with session.post(endpoint, json=payload) as resp:
                        if resp.status != 200:
                            continue
                        data = await resp.json()
                        result = self._extract_embeddings(data, len(texts))
                        if result:
                            return result
                except Exception as e:
                    logger.debug(f"Endpoint {endpoint} failed: {e}")
                    continue

        logger.warning(f"All embedding endpoints failed, returning zero vectors for {len(texts)} texts")
        return [[0.0] * self._dims for _ in texts]

    def _payload_openai(self, texts: list[str]) -> dict:
        return {"model": self._model, "input": texts}

    def _payload_ollama(self, texts: list[str]) -> dict:
        return {"model": self._model, "prompt": texts[0] if len(texts) == 1 else " ".join(texts)}

    def _payload_llama_cpp(self, texts: list[str]) -> dict:
        return {"content": texts}

    def _extract_embeddings(self, data: dict, expected: int) -> list[list[float]] | None:
        # OpenAI format: {"data": [{"embedding": [...]}]}
        if "data" in data and isinstance(data["data"], list):
            embeds = []
            for item in data["data"]:
                if isinstance(item, dict) and "embedding" in item:
                    embeds.append(item["embedding"])
            if embeds:
                return embeds

        # Ollama format: {"embedding": [...]}
        if "embedding" in data:
            return [data["embedding"]] * expected

        # llama.cpp: {"results": [{"embedding": [...]}]}
        if "results" in data and isinstance(data["results"], list):
            embeds = []
            for item in data["results"]:
                if isinstance(item, dict) and "embedding" in item:
                    embeds.append(item["embedding"])
            if embeds:
                return embeds

        return None

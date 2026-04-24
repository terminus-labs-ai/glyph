"""Tests for LlamaEmbedder — focused on the Ollama batch embedding bug fix.

The bug: _payload_ollama() joins multiple texts into a single string via
" ".join(texts), and _extract_embeddings() duplicates that single embedding
for all texts in the batch. Every text gets the SAME embedding.

Required behavior: Ollama endpoint should send one request per text (since
/api/embeddings only supports a single prompt), producing distinct embeddings.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from glyph.embedders.llama import LlamaEmbedder


# ---------------------------------------------------------------------------
# Mock helpers (same pattern as tests/rerankers/test_llama_reranker.py)
# ---------------------------------------------------------------------------


class _AsyncResp:
    """Minimal async context manager mimicking aiohttp response."""

    def __init__(self, status: int = 200, json_data: dict | None = None):
        self.status = status
        self._json_data = json_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def json(self):
        return self._json_data


class _AsyncSession:
    """Minimal async context manager mimicking aiohttp.ClientSession."""

    def __init__(self, *, status: int = 200, json_data: dict | None = None, **_kwargs):
        self._resp = _AsyncResp(status=status, json_data=json_data)

    def __call__(self, *args, **kwargs):
        return self

    def post(self, *args, **_kwargs):
        return self._resp

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DIMS = 4  # small dimension for test brevity


def _make_embedding(seed: int) -> list[float]:
    """Return a deterministic embedding that is unique per seed."""
    return [float(seed + i) for i in range(DIMS)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOllamaBatchEmbedding:
    """Tests for the Ollama endpoint batch embedding bug fix."""

    async def test_batch_of_three_produces_distinct_embeddings(self):
        """When 3 texts are sent via the Ollama endpoint, each must get a
        DISTINCT embedding — not the same one duplicated 3 times."""
        embedder = LlamaEmbedder(
            url="http://localhost:11434",
            model="test-model",
            dims=DIMS,
            batch_size=3,
        )

        texts = ["alpha text", "beta text", "gamma text"]

        # Track calls and return a unique embedding per call
        call_count = [0]

        class _PerTextSession:
            def __init__(self, **kwargs):
                pass

            def __call__(self, *args, **kwargs):
                return self

            def post(self, url: str, **kwargs):
                # Only the Ollama endpoint should succeed
                if "/api/embeddings" in url:
                    call_count[0] += 1
                    emb = _make_embedding(call_count[0])
                    return _AsyncResp(
                        status=200, json_data={"embedding": emb}
                    )
                # All other endpoints fail
                return _AsyncResp(status=404, json_data={})

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with patch(
            "glyph.embedders.llama.aiohttp.ClientSession",
            _PerTextSession(),
        ):
            result = await embedder.embed(texts)

        assert len(result) == 3
        # Each embedding must be different
        assert result[0] != result[1], "Texts 0 and 1 got the same embedding"
        assert result[1] != result[2], "Texts 1 and 2 got the same embedding"
        assert result[0] != result[2], "Texts 0 and 2 got the same embedding"

    async def test_ollama_batch_sends_one_request_per_text(self):
        """The Ollama endpoint must be called once PER text in the batch,
        since /api/embeddings only supports a single prompt."""
        embedder = LlamaEmbedder(
            url="http://localhost:11434",
            model="test-model",
            dims=DIMS,
            batch_size=3,
        )

        texts = ["text A", "text B", "text C"]
        captured_payloads: list[dict] = []

        class _TrackingSession:
            def __init__(self, **kwargs):
                pass

            def __call__(self, *args, **kwargs):
                return self

            def post(self, url: str, **kwargs):
                if "/api/embeddings" in url:
                    payload = kwargs.get("json", {})
                    captured_payloads.append(payload)
                    idx = len(captured_payloads)
                    return _AsyncResp(
                        status=200,
                        json_data={"embedding": _make_embedding(idx)},
                    )
                return _AsyncResp(status=404, json_data={})

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with patch(
            "glyph.embedders.llama.aiohttp.ClientSession",
            _TrackingSession(),
        ):
            await embedder.embed(texts)

        # Must be 3 separate calls, one per text
        assert len(captured_payloads) == 3, (
            f"Expected 3 Ollama API calls (one per text), got {len(captured_payloads)}"
        )
        # Each payload should contain a single text, not a concatenation
        for i, payload in enumerate(captured_payloads):
            prompt = payload.get("prompt", "")
            assert prompt == texts[i], (
                f"Payload {i} prompt should be {texts[i]!r}, got {prompt!r}"
            )

    async def test_ollama_single_text_still_works(self):
        """A batch of 1 text via Ollama should still work correctly."""
        embedder = LlamaEmbedder(
            url="http://localhost:11434",
            model="test-model",
            dims=DIMS,
            batch_size=1,
        )

        expected_embedding = _make_embedding(42)

        class _SingleSession:
            def __init__(self, **kwargs):
                pass

            def __call__(self, *args, **kwargs):
                return self

            def post(self, url: str, **kwargs):
                if "/api/embeddings" in url:
                    payload = kwargs.get("json", {})
                    assert payload.get("prompt") == "single text"
                    return _AsyncResp(
                        status=200,
                        json_data={"embedding": expected_embedding},
                    )
                return _AsyncResp(status=404, json_data={})

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with patch(
            "glyph.embedders.llama.aiohttp.ClientSession",
            _SingleSession(),
        ):
            result = await embedder.embed(["single text"])

        assert len(result) == 1
        assert result[0] == expected_embedding


class TestOpenAIBatchEmbeddingRegression:
    """Regression tests: OpenAI endpoint already handles batches correctly.
    Make sure the fix doesn't break it."""

    async def test_openai_batch_returns_distinct_embeddings(self):
        """OpenAI endpoint sends all texts in one request and returns
        per-text embeddings. This must continue to work."""
        embedder = LlamaEmbedder(
            url="http://localhost:11434",
            model="test-model",
            dims=DIMS,
            batch_size=3,
        )

        texts = ["first", "second", "third"]
        emb_a = _make_embedding(10)
        emb_b = _make_embedding(20)
        emb_c = _make_embedding(30)

        openai_response = {
            "data": [
                {"embedding": emb_a},
                {"embedding": emb_b},
                {"embedding": emb_c},
            ]
        }

        class _OpenAISession:
            def __init__(self, **kwargs):
                pass

            def __call__(self, *args, **kwargs):
                return self

            def post(self, url: str, **kwargs):
                if "/v1/embeddings" in url:
                    return _AsyncResp(status=200, json_data=openai_response)
                return _AsyncResp(status=404, json_data={})

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with patch(
            "glyph.embedders.llama.aiohttp.ClientSession",
            _OpenAISession(),
        ):
            result = await embedder.embed(texts)

        assert len(result) == 3
        assert result[0] == emb_a
        assert result[1] == emb_b
        assert result[2] == emb_c

    async def test_openai_batch_sends_single_request(self):
        """OpenAI endpoint should send ONE request with all texts."""
        embedder = LlamaEmbedder(
            url="http://localhost:11434",
            model="test-model",
            dims=DIMS,
            batch_size=3,
        )

        texts = ["first", "second", "third"]
        call_count = [0]

        openai_response = {
            "data": [
                {"embedding": _make_embedding(i)} for i in range(3)
            ]
        }

        class _CountingSession:
            def __init__(self, **kwargs):
                pass

            def __call__(self, *args, **kwargs):
                return self

            def post(self, url: str, **kwargs):
                if "/v1/embeddings" in url:
                    call_count[0] += 1
                    payload = kwargs.get("json", {})
                    assert payload.get("input") == texts
                    return _AsyncResp(status=200, json_data=openai_response)
                return _AsyncResp(status=404, json_data={})

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with patch(
            "glyph.embedders.llama.aiohttp.ClientSession",
            _CountingSession(),
        ):
            await embedder.embed(texts)

        assert call_count[0] == 1, (
            f"OpenAI endpoint should send 1 batch request, sent {call_count[0]}"
        )

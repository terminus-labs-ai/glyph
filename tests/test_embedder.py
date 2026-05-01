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


class TestRetryAndBackoff:
    """Tests for exponential backoff retries on transient failures."""

    async def test_retry_on_500_server_error(self):
        """A 500 response should trigger a retry with exponential backoff."""
        embedder = LlamaEmbedder(
            url="http://localhost:11434",
            model="test-model",
            dims=DIMS,
            batch_size=3,
            max_retries=3,
            retry_base_delay=0.01,  # fast for tests
        )

        call_count = [0]

        class _RetrySession:
            def __init__(self, **kwargs):
                pass

            def __call__(self, *args, **kwargs):
                return self

            def post(self, url: str, **kwargs):
                if "/v1/embeddings" in url:
                    call_count[0] += 1
                    if call_count[0] <= 2:
                        return _AsyncResp(status=500, json_data={"error": "busy"})
                    return _AsyncResp(
                        status=200,
                        json_data={"data": [{"embedding": _make_embedding(42)}]},
                    )
                return _AsyncResp(status=404, json_data={})

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with patch(
            "glyph.embedders.llama.aiohttp.ClientSession",
            _RetrySession(),
        ):
            result = await embedder.embed(["text A"])

        assert len(result) == 1
        assert call_count[0] == 3  # 2 failures + 1 success

    async def test_gives_up_after_max_retries(self):
        """After max_retries attempts, should return zero vectors."""
        embedder = LlamaEmbedder(
            url="http://localhost:11434",
            model="test-model",
            dims=DIMS,
            batch_size=3,
            max_retries=2,
            retry_base_delay=0.01,
        )

        call_count = [0]

        class _AlwaysFailSession:
            def __init__(self, **kwargs):
                pass

            def __call__(self, *args, **kwargs):
                return self

            def post(self, url: str, **kwargs):
                call_count[0] += 1
                return _AsyncResp(status=500, json_data={"error": "busy"})

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with patch(
            "glyph.embedders.llama.aiohttp.ClientSession",
            _AlwaysFailSession(),
        ):
            result = await embedder.embed(["text A"])

        assert len(result) == 1
        assert result[0] == [0.0] * DIMS  # zero vector fallback
        assert call_count[0] >= 2  # at least max_retries attempts


class TestBatchDelay:
    """Tests for rate limiting between batches."""

    async def test_no_delay_when_batch_delay_is_zero(self):
        """Default behavior: no delay between batches."""
        import time

        embedder = LlamaEmbedder(
            url="http://localhost:11434",
            model="test-model",
            dims=DIMS,
            batch_size=2,
            batch_delay=0.0,
        )

        texts = ["a", "b", "c", "d", "e"]  # 3 batches of size 2

        class _FastSession:
            def __init__(self, **kwargs):
                pass

            def __call__(self, *args, **kwargs):
                return self

            def post(self, url: str, **kwargs):
                return _AsyncResp(
                    status=200,
                    json_data={"data": [{"embedding": _make_embedding(1)}]},
                )

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with patch(
            "glyph.embedders.llama.aiohttp.ClientSession",
            _FastSession(),
        ):
            start = time.monotonic()
            result = await embedder.embed(texts)
            elapsed = time.monotonic() - start

        assert len(result) == 5
        # With no delay and instant mocks, should be very fast
        assert elapsed < 0.1, f"Should complete quickly without delay, took {elapsed:.3f}s"


class TestPersistentSession:
    """Tests for the persistent ClientSession."""

    async def test_close_cleans_up_session(self):
        """Calling close() should close the underlying session."""
        embedder = LlamaEmbedder(
            url="http://localhost:11434",
            model="test-model",
            dims=DIMS,
            batch_size=1,
        )

        session_created = []

        class _TrackingSession:
            def __init__(self, **kwargs):
                session_created.append(self)

            def __call__(self, *args, **kwargs):
                return self

            def post(self, url: str, **kwargs):
                return _AsyncResp(
                    status=200,
                    json_data={"data": [{"embedding": _make_embedding(1)}]},
                )

            @property
            def closed(self):
                return getattr(self, "_closed", False)

            async def close(self):
                self._closed = True

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with patch(
            "glyph.embedders.llama.aiohttp.ClientSession",
            _TrackingSession,
        ):
            await embedder.embed(["test"])
            assert len(session_created) == 1  # session created
            assert not session_created[0].closed

            await embedder.close()
            assert session_created[0].closed  # session closed

    async def test_session_reused_across_batches(self):
        """The same session should be reused for multiple embed() calls."""
        session_instances = []

        class _TrackingSession:
            def __init__(self, **kwargs):
                session_instances.append(id(self))

            def __call__(self, *args, **kwargs):
                return self

            def post(self, url: str, **kwargs):
                return _AsyncResp(
                    status=200,
                    json_data={"data": [{"embedding": _make_embedding(1)}]},
                )

            @property
            def closed(self):
                return False

            async def close(self):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        embedder = LlamaEmbedder(
            url="http://localhost:11434",
            model="test-model",
            dims=DIMS,
            batch_size=2,
            batch_delay=0.0,
        )

        with patch(
            "glyph.embedders.llama.aiohttp.ClientSession",
            _TrackingSession,
        ):
            # Two batches (4 texts, batch_size=2)
            await embedder.embed(["a", "b", "c", "d"])
            # Should have created exactly one session
            assert len(session_instances) == 1, (
                f"Expected 1 session, got {len(session_instances)} "
                f"(session was recreated between batches)"
            )

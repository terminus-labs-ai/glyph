from __future__ import annotations

from unittest.mock import patch

import pytest

from glyph.rerankers.llama import LlamaReranker

SAMPLE_RESPONSE = {
    "results": [
        {"index": 3, "relevance_score": 0.95},
        {"index": 1, "relevance_score": 0.82},
        {"index": 0, "relevance_score": 0.70},
        {"index": 2, "relevance_score": 0.45},
    ]
}


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
    """Minimal async context manager mimicking aiohttp.ClientSession.

    Callable so it can be used as a patch target that replaces the class.
    """

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


class TestLlamaReranker:
    async def test_basic_rerank(self):
        reranker = LlamaReranker(
            url="http://localhost:11434",
            model="qwen3-reranker",
            batch_size=32,
        )
        docs = [f"Document {i}" for i in range(4)]

        with patch(
            "glyph.rerankers.llama.aiohttp.ClientSession",
            _AsyncSession(json_data=SAMPLE_RESPONSE),
        ):
            scores = await reranker.rerank("test query", docs)

        # Scores should be re-sorted to input order
        assert scores == [0.70, 0.82, 0.45, 0.95]

    async def test_request_body_shape(self):
        reranker = LlamaReranker(
            url="http://localhost:11441",
            model="qwen3-reranker",
        )
        docs = ["First doc", "Second doc"]

        expected_payload = {
            "model": "qwen3-reranker",
            "query": "my query",
            "documents": docs,
        }

        captured = {"payload": None}
        batch_response = {
            "results": [
                {"index": 1, "relevance_score": 0.82},
                {"index": 0, "relevance_score": 0.70},
            ]
        }

        class _TrackingSession(_AsyncSession):
            def __init__(self, **kw):
                super().__init__(**kw, json_data=batch_response)
                captured["payload"] = None

            def post(self, *args, **kwargs):
                captured["payload"] = kwargs.get("json")
                return self._resp

        with patch(
            "glyph.rerankers.llama.aiohttp.ClientSession",
            _TrackingSession(),
        ):
            await reranker.rerank("my query", docs)

        assert captured["payload"] == expected_payload

    async def test_empty_documents(self):
        reranker = LlamaReranker(
            url="http://localhost:11434",
            model="qwen3-reranker",
        )
        scores = await reranker.rerank("query", [])
        assert scores == []

    async def test_single_document(self):
        reranker = LlamaReranker(
            url="http://localhost:11434",
            model="qwen3-reranker",
        )
        scores = await reranker.rerank("query", ["single doc"])
        assert scores == [1.0]

    async def test_re_sort_to_input_order(self):
        """API returns results sorted by score — must be re-ordered to match input."""
        reranker = LlamaReranker(
            url="http://localhost:11434",
            model="qwen3-reranker",
        )

        custom_response = {
            "results": [
                {"index": 2, "relevance_score": 0.90},
                {"index": 0, "relevance_score": 0.70},
                {"index": 1, "relevance_score": 0.30},
            ]
        }

        with patch(
            "glyph.rerankers.llama.aiohttp.ClientSession",
            _AsyncSession(json_data=custom_response),
        ):
            scores = await reranker.rerank("find me", ["doc A", "doc B", "doc C"])

        # Should match input order: [doc A=0.70, doc B=0.30, doc C=0.90]
        assert scores == [0.70, 0.30, 0.90]

    async def test_batch_splitting(self):
        """More docs than batch_size should split into multiple API calls."""
        reranker = LlamaReranker(
            url="http://localhost:11434",
            model="qwen3-reranker",
            batch_size=2,
        )
        docs = [f"Doc {i}" for i in range(5)]  # 5 docs, batch_size=2 → 3 batches

        call_count = [0]
        batch_responses = [
            {"results": [{"index": 1, "relevance_score": 0.95}, {"index": 0, "relevance_score": 0.85}]},
            {"results": [{"index": 1, "relevance_score": 0.75}, {"index": 0, "relevance_score": 0.65}]},
            {"results": [{"index": 0, "relevance_score": 0.55}]},
        ]

        class _CountingSession:
            def __init__(self, **kwargs):
                call_count[0] += 1
                idx = min(call_count[0] - 1, len(batch_responses) - 1)
                self._resp = _AsyncResp(status=200, json_data=batch_responses[idx])

            def __call__(self, *args, **kwargs):
                return self

            def post(self, *args, **_kwargs):
                return self._resp

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with patch(
            "glyph.rerankers.llama.aiohttp.ClientSession",
            _CountingSession,
        ):
            scores = await reranker.rerank("query", docs)

        assert call_count[0] == 3  # 2 + 2 + 1
        assert len(scores) == 5

    async def test_failure_fallback(self):
        """Connection error should raise, but caller catches it."""
        reranker = LlamaReranker(
            url="http://localhost:9999",
            model="qwen3-reranker",
            timeout=1,
        )

        class _FailingSession:
            def __init__(self, **kwargs):
                pass

            def __call__(self, *args, **kwargs):
                return self

            def post(self, *args, **kwargs):
                raise ConnectionRefusedError("connection refused")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with patch(
            "glyph.rerankers.llama.aiohttp.ClientSession",
            _FailingSession(),
        ):
            with pytest.raises(ConnectionRefusedError):
                await reranker.rerank("query", ["doc1", "doc2"])


class TestLlamaRerankerIntegration:
    """Integration tests that hit a live reranker service. Skipped unless
    --reranker-url is passed."""

    async def test_integration_rerank(self, request):
        from glyph.rerankers import LlamaReranker

        url = request.config.getoption("--reranker-url", None)
        if url is None:
            pytest.skip("--reranker-url not set")

        reranker = LlamaReranker(url=url, model="qwen3-reranker", timeout=10)

        docs = [
            "Python is a programming language.",
            "The capital of France is Paris.",
            "Bananas are yellow fruits.",
        ]

        scores = await reranker.rerank("What is Python?", docs)
        assert len(scores) == len(docs)
        # Scores should be in input order
        # Python doc should score higher than unrelated docs
        assert scores[0] > scores[1], "Python doc should score higher than France doc"

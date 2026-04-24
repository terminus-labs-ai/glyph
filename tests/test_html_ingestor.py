"""Tests for HTMLIngestor concurrent fetching behavior.

Verifies that the HTML crawler uses asyncio.Semaphore-based concurrency
rather than sequential fetching, while respecting max_concurrent limits
and per-request politeness delays.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from glyph.ingestors.html import HTMLIngestor


# --- Helpers ---

SOURCE_ID = uuid.uuid4()
BASE_URL = "https://docs.example.com"


def _html_page(title: str, body: str, links: list[str] | None = None) -> str:
    """Build a minimal HTML page with optional links."""
    link_html = ""
    if links:
        link_html = "\n".join(f'<a href="{u}">{u}</a>' for u in links)
    return f"""<!DOCTYPE html>
<html>
<head><title>{title}</title></head>
<body>
<main>
<h1>{title}</h1>
<p>{body}</p>
{link_html}
</main>
</body>
</html>"""


# Content must be >= MIN_CONTENT_LENGTH (100 chars)
FILLER = "x" * 120


def _make_response(html: str, url: str, status: int = 200) -> AsyncMock:
    """Create a mock aiohttp response context manager."""
    resp = AsyncMock()
    resp.status = status
    resp.text = AsyncMock(return_value=html)
    resp.url = url  # URL after redirects

    # Make it work as an async context manager
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


class FakeSession:
    """Mock aiohttp.ClientSession that records fetch timing.

    Tracks when each URL fetch starts/ends and how many are in-flight
    simultaneously. Supports configurable per-URL delay to simulate
    network latency.
    """

    def __init__(
        self,
        responses: dict[str, str],
        fetch_delay: float = 0.05,
    ):
        """
        Args:
            responses: URL -> HTML content mapping
            fetch_delay: simulated network latency per fetch (seconds)
        """
        self._responses = responses
        self._fetch_delay = fetch_delay

        # Concurrency tracking
        self.fetch_times: list[tuple[str, float, float]] = []  # (url, start, end)
        self._active_count = 0
        self.max_concurrent_observed = 0
        self._lock = asyncio.Lock()

    def get(self, url: str):
        """Return an async context manager that simulates fetching the URL."""
        session = self

        class _FetchCtx:
            async def __aenter__(ctx_self):
                async with session._lock:
                    session._active_count += 1
                    session.max_concurrent_observed = max(
                        session.max_concurrent_observed, session._active_count
                    )
                start = time.monotonic()

                # Simulate network delay
                await asyncio.sleep(session._fetch_delay)

                async with session._lock:
                    session._active_count -= 1
                end = time.monotonic()
                session.fetch_times.append((url, start, end))

                html = session._responses.get(url, "")
                resp = MagicMock()
                resp.status = 200 if html else 404
                resp.text = AsyncMock(return_value=html)
                resp.url = url
                return resp

            async def __aexit__(ctx_self, *args):
                pass

        return _FetchCtx()


# --- Tests ---


class TestConcurrentFetching:
    """Verify that HTMLIngestor fetches multiple URLs concurrently."""

    @pytest.mark.asyncio
    async def test_concurrent_fetches_happen_in_parallel(self):
        """With max_concurrent=3 and 6 discoverable URLs, fetches should
        overlap in time rather than running strictly sequentially.

        We verify this by checking total wall-clock time is significantly
        less than sequential would take (6 * fetch_delay + 6 * politeness_delay).
        """
        fetch_delay = 0.05
        politeness_delay = 0.01

        # Page 1 (seed) links to pages 2-6
        child_urls = [f"{BASE_URL}/page{i}" for i in range(2, 7)]
        responses = {
            BASE_URL: _html_page(
                "Home", FILLER, links=child_urls
            ),
        }
        for i, url in enumerate(child_urls):
            responses[url] = _html_page(f"Page {i+2}", FILLER)

        fake_session = FakeSession(responses, fetch_delay=fetch_delay)

        ingestor = HTMLIngestor(
            BASE_URL,
            SOURCE_ID,
            max_pages=100,
            delay=politeness_delay,
            max_concurrent=3,
        )

        # Patch aiohttp.ClientSession to return our fake session
        with patch("glyph.ingestors.html.aiohttp.ClientSession") as mock_cs:
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=fake_session)
            ctx.__aexit__ = AsyncMock(return_value=False)
            mock_cs.return_value = ctx

            start = time.monotonic()
            docs = await ingestor.ingest()
            elapsed = time.monotonic() - start

        # We should have fetched 6 pages total (seed + 5 children)
        assert len(docs) == 6

        # Sequential time would be ~6 * (fetch_delay + politeness_delay) = 0.36s
        # With concurrency=3, it should be roughly half or less.
        sequential_time = 6 * (fetch_delay + politeness_delay)
        assert elapsed < sequential_time * 0.75, (
            f"Fetching took {elapsed:.3f}s, expected less than "
            f"{sequential_time * 0.75:.3f}s (75% of sequential {sequential_time:.3f}s). "
            f"Fetches are likely still sequential."
        )

    @pytest.mark.asyncio
    async def test_semaphore_limits_max_concurrent(self):
        """No more than max_concurrent fetches should be in-flight at once,
        but concurrent fetches SHOULD actually happen (> 1 at a time)."""
        max_concurrent = 2
        fetch_delay = 0.05

        # Seed links to 5 child pages — all discovered at once from seed
        child_urls = [f"{BASE_URL}/page{i}" for i in range(2, 7)]
        responses = {
            BASE_URL: _html_page("Home", FILLER, links=child_urls),
        }
        for i, url in enumerate(child_urls):
            responses[url] = _html_page(f"Page {i+2}", FILLER)

        fake_session = FakeSession(responses, fetch_delay=fetch_delay)

        ingestor = HTMLIngestor(
            BASE_URL,
            SOURCE_ID,
            max_pages=100,
            delay=0.0,
            max_concurrent=max_concurrent,
        )

        with patch("glyph.ingestors.html.aiohttp.ClientSession") as mock_cs:
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=fake_session)
            ctx.__aexit__ = AsyncMock(return_value=False)
            mock_cs.return_value = ctx

            await ingestor.ingest()

        # Must not exceed the semaphore limit
        assert fake_session.max_concurrent_observed <= max_concurrent, (
            f"Observed {fake_session.max_concurrent_observed} concurrent fetches, "
            f"but max_concurrent={max_concurrent}. Semaphore not enforced."
        )

        # Must actually use concurrency (> 1 simultaneous fetch).
        # After the seed is fetched, 5 child URLs are known — they should
        # be dispatched concurrently up to the semaphore limit.
        assert fake_session.max_concurrent_observed > 1, (
            f"Max concurrent observed was {fake_session.max_concurrent_observed}. "
            f"Expected > 1 — fetches are still sequential."
        )

    @pytest.mark.asyncio
    async def test_politeness_delay_applied_per_request(self):
        """Each fetch should incur the politeness delay, verifiable via
        total elapsed time being at least N * delay.
        """
        politeness_delay = 0.03
        num_pages = 4

        child_urls = [f"{BASE_URL}/page{i}" for i in range(2, num_pages + 1)]
        responses = {
            BASE_URL: _html_page("Home", FILLER, links=child_urls),
        }
        for i, url in enumerate(child_urls):
            responses[url] = _html_page(f"Page {i+2}", FILLER)

        fake_session = FakeSession(responses, fetch_delay=0.0)

        ingestor = HTMLIngestor(
            BASE_URL,
            SOURCE_ID,
            max_pages=100,
            delay=politeness_delay,
            max_concurrent=10,
        )

        with patch("glyph.ingestors.html.aiohttp.ClientSession") as mock_cs:
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=fake_session)
            ctx.__aexit__ = AsyncMock(return_value=False)
            mock_cs.return_value = ctx

            start = time.monotonic()
            docs = await ingestor.ingest()
            elapsed = time.monotonic() - start

        assert len(docs) == num_pages
        # With concurrent fetches, the delay still accumulates per-request.
        # Even with full concurrency, we need at least (num_pages * delay) total
        # sleep time distributed across workers. The minimum wall-clock time is
        # ceil(num_pages / max_concurrent) * delay, but each request should
        # individually sleep. We check a lower bound: total elapsed >= num_pages * delay * 0.5
        # (generous margin for scheduling jitter while still catching "no delay at all").
        min_expected = num_pages * politeness_delay * 0.5
        assert elapsed >= min_expected, (
            f"Total time {elapsed:.3f}s is too short. "
            f"Expected at least {min_expected:.3f}s from {num_pages} requests "
            f"with {politeness_delay}s delay each. Delay may not be applied."
        )


class TestBFSBehavior:
    """Verify BFS link discovery still works correctly with concurrency."""

    @pytest.mark.asyncio
    async def test_discovered_links_are_fetched(self):
        """Links found in page A should be added to the queue and fetched."""
        # Seed -> page2 -> page3 (chain of discovery)
        responses = {
            BASE_URL: _html_page(
                "Home", FILLER, links=[f"{BASE_URL}/page2"]
            ),
            f"{BASE_URL}/page2": _html_page(
                "Page 2", FILLER, links=[f"{BASE_URL}/page3"]
            ),
            f"{BASE_URL}/page3": _html_page("Page 3", FILLER),
        }

        fake_session = FakeSession(responses, fetch_delay=0.0)

        ingestor = HTMLIngestor(
            BASE_URL,
            SOURCE_ID,
            max_pages=100,
            delay=0.0,
            max_concurrent=5,
        )

        with patch("glyph.ingestors.html.aiohttp.ClientSession") as mock_cs:
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=fake_session)
            ctx.__aexit__ = AsyncMock(return_value=False)
            mock_cs.return_value = ctx

            docs = await ingestor.ingest()

        fetched_urls = {url for url, _, _ in fake_session.fetch_times}
        assert BASE_URL in fetched_urls
        assert f"{BASE_URL}/page2" in fetched_urls
        assert f"{BASE_URL}/page3" in fetched_urls
        assert len(docs) == 3

    @pytest.mark.asyncio
    async def test_visited_urls_not_refetched(self):
        """URLs already visited should not be fetched again, even if
        discovered by multiple pages."""
        # Both seed and page2 link to page3
        responses = {
            BASE_URL: _html_page(
                "Home",
                FILLER,
                links=[f"{BASE_URL}/page2", f"{BASE_URL}/page3"],
            ),
            f"{BASE_URL}/page2": _html_page(
                "Page 2", FILLER, links=[f"{BASE_URL}/page3"]
            ),
            f"{BASE_URL}/page3": _html_page("Page 3", FILLER),
        }

        fake_session = FakeSession(responses, fetch_delay=0.0)

        ingestor = HTMLIngestor(
            BASE_URL,
            SOURCE_ID,
            max_pages=100,
            delay=0.0,
            max_concurrent=5,
        )

        with patch("glyph.ingestors.html.aiohttp.ClientSession") as mock_cs:
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=fake_session)
            ctx.__aexit__ = AsyncMock(return_value=False)
            mock_cs.return_value = ctx

            docs = await ingestor.ingest()

        # page3 should appear only once in fetch_times
        page3_fetches = [
            url for url, _, _ in fake_session.fetch_times
            if url == f"{BASE_URL}/page3"
        ]
        assert len(page3_fetches) == 1, (
            f"page3 was fetched {len(page3_fetches)} times, expected 1"
        )

    @pytest.mark.asyncio
    async def test_max_pages_respected(self):
        """Ingestor should stop fetching once max_pages documents are collected."""
        # Seed links to 10 pages, but max_pages=3
        child_urls = [f"{BASE_URL}/page{i}" for i in range(2, 12)]
        responses = {
            BASE_URL: _html_page("Home", FILLER, links=child_urls),
        }
        for i, url in enumerate(child_urls):
            responses[url] = _html_page(f"Page {i+2}", FILLER)

        fake_session = FakeSession(responses, fetch_delay=0.0)

        ingestor = HTMLIngestor(
            BASE_URL,
            SOURCE_ID,
            max_pages=3,
            delay=0.0,
            max_concurrent=5,
        )

        with patch("glyph.ingestors.html.aiohttp.ClientSession") as mock_cs:
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=fake_session)
            ctx.__aexit__ = AsyncMock(return_value=False)
            mock_cs.return_value = ctx

            docs = await ingestor.ingest()

        assert len(docs) <= 3

    @pytest.mark.asyncio
    async def test_cross_domain_links_ignored(self):
        """Links to other domains should not be followed."""
        responses = {
            BASE_URL: _html_page(
                "Home",
                FILLER,
                links=[
                    f"{BASE_URL}/page2",
                    "https://evil.example.com/steal",
                ],
            ),
            f"{BASE_URL}/page2": _html_page("Page 2", FILLER),
        }

        fake_session = FakeSession(responses, fetch_delay=0.0)

        ingestor = HTMLIngestor(
            BASE_URL,
            SOURCE_ID,
            max_pages=100,
            delay=0.0,
            max_concurrent=5,
        )

        with patch("glyph.ingestors.html.aiohttp.ClientSession") as mock_cs:
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=fake_session)
            ctx.__aexit__ = AsyncMock(return_value=False)
            mock_cs.return_value = ctx

            docs = await ingestor.ingest()

        fetched_urls = {url for url, _, _ in fake_session.fetch_times}
        assert "https://evil.example.com/steal" not in fetched_urls
        assert len(docs) == 2


class TestConcurrencyWithBFS:
    """Verify that concurrency and BFS interact correctly."""

    @pytest.mark.asyncio
    async def test_concurrent_fetches_with_multi_level_bfs(self):
        """Multi-level BFS should still discover and fetch all pages
        when using concurrent fetching.

        Level 0: seed -> [A, B]
        Level 1: A -> [C], B -> [D]
        Level 2: C -> [E], D -> []
        Total: 6 pages
        """
        responses = {
            BASE_URL: _html_page(
                "Home",
                FILLER,
                links=[f"{BASE_URL}/a", f"{BASE_URL}/b"],
            ),
            f"{BASE_URL}/a": _html_page(
                "Page A", FILLER, links=[f"{BASE_URL}/c"]
            ),
            f"{BASE_URL}/b": _html_page(
                "Page B", FILLER, links=[f"{BASE_URL}/d"]
            ),
            f"{BASE_URL}/c": _html_page(
                "Page C", FILLER, links=[f"{BASE_URL}/e"]
            ),
            f"{BASE_URL}/d": _html_page("Page D", FILLER),
            f"{BASE_URL}/e": _html_page("Page E", FILLER),
        }

        fake_session = FakeSession(responses, fetch_delay=0.01)

        ingestor = HTMLIngestor(
            BASE_URL,
            SOURCE_ID,
            max_pages=100,
            delay=0.0,
            max_concurrent=3,
        )

        with patch("glyph.ingestors.html.aiohttp.ClientSession") as mock_cs:
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=fake_session)
            ctx.__aexit__ = AsyncMock(return_value=False)
            mock_cs.return_value = ctx

            docs = await ingestor.ingest()

        fetched_urls = {url for url, _, _ in fake_session.fetch_times}
        expected = {
            BASE_URL,
            f"{BASE_URL}/a",
            f"{BASE_URL}/b",
            f"{BASE_URL}/c",
            f"{BASE_URL}/d",
            f"{BASE_URL}/e",
        }
        assert fetched_urls == expected, (
            f"Missing URLs: {expected - fetched_urls}"
        )
        assert len(docs) == 6

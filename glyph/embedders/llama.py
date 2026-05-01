from __future__ import annotations

import asyncio
import logging
import time

import aiohttp
import click

logger = logging.getLogger(__name__)


class LlamaEmbedder:
    """Generate embeddings via a local llama-server HTTP API.

    Key design: strictly sequential requests with configurable timeouts.
    Only one request is ever in-flight at a time. The client waits for
    the full response before sending the next request, preventing the
    server's HTTP backlog from building up.

    Supports:
    - Persistent sessions (reuses TCP connections)
    - Configurable batch delay between batches
    - Configurable request timeout (default 300s for long queues)
    - Exponential backoff on transient errors (429, 5xx, timeout)
    - Zero-vector fallback when server is unreachable (unless strict mode)
    """

    def __init__(
        self,
        url: str,
        model: str,
        dims: int,
        batch_size: int = 5,
        strict: bool = False,
        batch_delay: float = 0.0,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        request_timeout: float = 300.0,
        max_input_chars: int | None = None,
    ):
        self._url = url.rstrip("/")
        self._model = model
        self._dims = dims
        self._batch_size = batch_size
        self._strict = strict
        self._batch_delay = batch_delay
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._request_timeout = request_timeout
        self._max_input_chars = max_input_chars
        self._session: aiohttp.ClientSession | None = None

        # Stats for logging
        self._batches_sent = 0
        self._batches_retried = 0
        self._total_request_time = 0.0

    @property
    def dimensions(self) -> int:
        return self._dims

    async def _get_session(self) -> aiohttp.ClientSession:
        """Lazy-init a persistent ClientSession to reuse TCP connections."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._request_timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close the persistent session. Call before shutdown."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        if self._batches_sent > 0:
            avg_time = self._total_request_time / self._batches_sent
            logger.info(
                f"Embedding stats: {self._batches_sent} batches, "
                f"avg {avg_time:.1f}s/batch, "
                f"{self._batches_retried} required retries"
            )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, sending them in batches.

        Strictly sequential: waits for each batch response before sending
        the next. This prevents the server's HTTP backlog from building up.
        """
        embeddings: list[list[float]] = []
        total_batches = (len(texts) + self._batch_size - 1) // self._batch_size

        for i in range(0, len(texts), self._batch_size):
            batch_num = i // self._batch_size + 1
            batch = texts[i : i + self._batch_size]

            # Rate limit: pause between batches (skip before first batch)
            if self._batch_delay > 0 and batch_num > 1:
                await asyncio.sleep(self._batch_delay)

            batch_embeddings = await self._embed_batch(batch)
            embeddings.extend(batch_embeddings)
            self._batches_sent += 1

        return embeddings

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Send a single batch with retry logic and exponential backoff.

        Only one request is in-flight at a time. We wait for the full
        HTTP response before attempting the next request.
        """
        session = await self._get_session()

        if self._max_input_chars is not None:
            truncated = [t[: self._max_input_chars] for t in texts]
            n_truncated = sum(1 for o, t in zip(texts, truncated) if len(o) > self._max_input_chars)
            if n_truncated:
                logger.warning(
                    f"Truncated {n_truncated}/{len(texts)} texts to {self._max_input_chars} chars "
                    f"to fit model context window"
                )
            texts = truncated

        batch_endpoints = [
            (f"{self._url}/v1/embeddings", self._payload_openai),
            (f"{self._url}/embedding", self._payload_llama_cpp),
        ]
        ollama_endpoint = f"{self._url}/api/embeddings"

        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            batch_start = time.monotonic()
            got_response = False

            # Try each batch-capable endpoint
            for endpoint, payload_fn in batch_endpoints:
                try:
                    payload = payload_fn(texts)
                    async with session.post(endpoint, json=payload) as resp:
                        if resp.status == 429:
                            wait = self._retry_base_delay * (2**attempt)
                            logger.warning(
                                f"Rate limited (429) by {endpoint}, "
                                f"waiting {wait:.1f}s (attempt {attempt + 1}/{self._max_retries})"
                            )
                            await asyncio.sleep(wait)
                            last_error = RuntimeError("Rate limited: 429")
                            break  # go to next retry attempt

                        if resp.status >= 500:
                            elapsed = time.monotonic() - batch_start
                            logger.warning(
                                f"Server error {resp.status} from {endpoint} "
                                f"(after {elapsed:.1f}s), will retry"
                            )
                            last_error = RuntimeError(f"Server error: {resp.status}")
                            continue  # try next endpoint

                        if resp.status != 200:
                            logger.debug(
                                f"Client error {resp.status} from {endpoint}, trying next"
                            )
                            continue  # try next endpoint

                        # Success
                        elapsed = time.monotonic() - batch_start
                        self._total_request_time += elapsed
                        got_response = True

                        if elapsed > 30:
                            logger.info(
                                f"Batch response took {elapsed:.1f}s from {endpoint}"
                            )

                        data = await resp.json()
                        result = self._extract_embeddings(data, len(texts))
                        if result:
                            if attempt > 0:
                                logger.info(
                                    f"Batch succeeded on retry {attempt + 1} "
                                    f"after {elapsed:.1f}s"
                                )
                            return result
                        else:
                            logger.warning(
                                f"Could not parse embedding response from {endpoint}"
                            )

                except asyncio.TimeoutError:
                    elapsed = time.monotonic() - batch_start
                    logger.warning(
                        f"Request to {endpoint} timed out after {elapsed:.1f}s "
                        f"(limit: {self._request_timeout:.0f}s). "
                        f"Server may be overloaded. "
                        f"Waiting {self._retry_base_delay * (2**attempt):.1f}s before retry."
                    )
                    last_error = asyncio.TimeoutError(
                        f"Timeout after {elapsed:.1f}s (limit {self._request_timeout:.0f}s)"
                    )
                    await asyncio.sleep(self._retry_base_delay * (2**attempt))
                    break  # go to next retry attempt

                except ConnectionError as e:
                    logger.warning(
                        f"Connection error to {endpoint}: {e}. "
                        f"Will retry with backoff."
                    )
                    last_error = e
                    await asyncio.sleep(self._retry_base_delay * (2**attempt))
                    break  # connection error means all endpoints will fail

                except Exception as e:
                    logger.debug(f"Unexpected error from {endpoint}: {e}")
                    continue

            if got_response:
                continue

            # Try Ollama fallback if batch endpoints didn't work
            if not got_response and attempt < self._max_retries:
                try:
                    ollama_embeddings: list[list[float]] = []
                    ollama_failed = False

                    for text in texts:
                        payload = self._payload_ollama([text])
                        async with session.post(
                            ollama_endpoint, json=payload
                        ) as resp:
                            if resp.status == 429:
                                wait = self._retry_base_delay * (2**attempt)
                                logger.warning(
                                    f"Rate limited by {ollama_endpoint}, "
                                    f"waiting {wait:.1f}s"
                                )
                                await asyncio.sleep(wait)
                                ollama_failed = True
                                break

                            if resp.status >= 500:
                                ollama_failed = True
                                last_error = RuntimeError(
                                    f"Ollama server error: {resp.status}"
                                )
                                break

                            if resp.status != 200:
                                ollama_failed = True
                                break

                            data = await resp.json()
                            result = self._extract_embeddings(data, 1)
                            if not result:
                                ollama_failed = True
                                break
                            ollama_embeddings.append(result[0])

                    elapsed = time.monotonic() - batch_start
                    self._total_request_time += elapsed

                    if not ollama_failed and len(ollama_embeddings) == len(texts):
                        if attempt > 0:
                            logger.info(
                                f"Ollama batch succeeded on retry {attempt + 1} "
                                f"after {elapsed:.1f}s"
                            )
                        return ollama_embeddings

                except asyncio.TimeoutError:
                    elapsed = time.monotonic() - batch_start
                    logger.warning(
                        f"Ollama request timed out after {elapsed:.1f}s. "
                        f"Waiting before retry."
                    )
                    last_error = asyncio.TimeoutError()
                    await asyncio.sleep(self._retry_base_delay * (2**attempt))

                except ConnectionError as e:
                    logger.warning(f"Ollama connection error: {e}. Will retry.")
                    last_error = e
                    await asyncio.sleep(self._retry_base_delay * (2**attempt))

                except Exception as e:
                    logger.debug(f"Ollama error: {e}")

            # Wait before next retry attempt
            if attempt < self._max_retries - 1:
                self._batches_retried += 1
                wait = self._retry_base_delay * (2**attempt)
                logger.info(
                    f"All endpoints failed for batch ({len(texts)} texts). "
                    f"Retry {attempt + 2}/{self._max_retries} in {wait:.1f}s. "
                    f"{' (last error: ' + str(last_error) + ')'}" if last_error else ""
                )
                await asyncio.sleep(wait)

        # All retries exhausted
        msg = (
            f"Embedding failed for {len(texts)} texts after "
            f"{self._max_retries} attempts. "
            f"Server at {self._url} may be overloaded or unreachable."
        )
        if last_error:
            msg += f" Last error: {last_error}"

        logger.error(msg)
        click.echo(click.style(f"ERROR: {msg}", fg="red", bold=True), err=True)

        if self._strict:
            raise RuntimeError(
                f"Embedding failed (strict mode): all endpoints at "
                f"{self._url} unreachable after {self._max_retries} attempts"
            )

        logger.warning(
            "Returning zero vectors. Search quality will be degraded."
        )
        return [[0.0] * self._dims for _ in texts]

    def _payload_openai(self, texts: list[str]) -> dict:
        return {"model": self._model, "input": texts}

    def _payload_ollama(self, texts: list[str]) -> dict:
        return {
            "model": self._model,
            "prompt": texts[0] if len(texts) == 1 else " ".join(texts),
        }

    def _payload_llama_cpp(self, texts: list[str]) -> dict:
        return {"content": texts}

    def _extract_embeddings(
        self, data: dict, expected: int
    ) -> list[list[float]] | None:
        """Extract embeddings from various API response formats."""
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

        # llama.cpp format: {"results": [{"embedding": [...]}]}
        if "results" in data and isinstance(data["results"], list):
            embeds = []
            for item in data["results"]:
                if isinstance(item, dict) and "embedding" in item:
                    embeds.append(item["embedding"])
            if embeds:
                return embeds

        return None

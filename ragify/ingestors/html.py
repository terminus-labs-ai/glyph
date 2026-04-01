from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import uuid
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup

from ragify.domain.models import DocType, Document

logger = logging.getLogger(__name__)

MIN_CONTENT_LENGTH = 100


class HTMLIngestor:
    """Generic async HTML crawler for documentation sites."""

    def __init__(
        self,
        base_url: str,
        source_id: uuid.UUID,
        *,
        max_pages: int = 500,
        delay: float = 0.2,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        max_concurrent: int = 10,
    ):
        self._base_url = base_url.rstrip("/")
        self._source_id = source_id
        self._max_pages = max_pages
        self._delay = delay
        self._include = [re.compile(p) for p in (include_patterns or [])]
        self._exclude = [re.compile(p) for p in (exclude_patterns or [])]
        self._max_concurrent = max_concurrent

    async def ingest(self) -> list[Document]:
        visited: set[str] = set()
        documents: list[Document] = []
        queue = [self._base_url]

        connector = aiohttp.TCPConnector(limit=self._max_concurrent)
        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            while queue and len(documents) < self._max_pages:
                url = queue.pop(0)
                normalized = self._normalize_url(url)

                if normalized in visited:
                    continue
                visited.add(normalized)

                if not self._should_include(normalized):
                    continue

                logger.info(f"Fetching: {normalized} ({len(documents)}/{self._max_pages})")

                doc, links = await self._fetch_page(session, normalized)
                if doc:
                    documents.append(doc)

                for link in links:
                    link_norm = self._normalize_url(link)
                    if link_norm not in visited and self._is_same_domain(link_norm):
                        queue.append(link_norm)

                await asyncio.sleep(self._delay)

        logger.info(f"Scraped {len(documents)} pages from {self._base_url}")
        return documents

    async def _fetch_page(
        self, session: aiohttp.ClientSession, url: str
    ) -> tuple[Document | None, list[str]]:
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None, []

                html = await resp.text()
                soup = BeautifulSoup(html, "lxml")

                # Extract title
                title_tag = soup.find("title")
                title = title_tag.get_text().strip() if title_tag else url.split("/")[-1]

                # Extract links before removing nav elements
                # Use response URL (after redirects) for correct relative resolution
                resolved_base = str(resp.url)
                links = []
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    full = urljoin(resolved_base, href)
                    # Strip fragment
                    full = full.split("#")[0]
                    if full and self._is_same_domain(full):
                        links.append(full)

                # Remove noise
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()

                # Try to find main content area
                main = soup.find("main") or soup.find("article") or soup.find("div", class_="document")
                content_root = main or soup.body or soup

                text = content_root.get_text(separator="\n", strip=True)
                if len(text) < MIN_CONTENT_LENGTH:
                    return None, links

                content_hash = hashlib.md5(text.encode()).hexdigest()

                # Determine doc type from URL
                doc_type = self._classify_url(url)

                doc = Document(
                    source_id=self._source_id,
                    path=url,
                    title=title,
                    doc_type=doc_type,
                    raw_content=text,
                    content_hash=content_hash,
                )

                return doc, links

        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None, []

    def _normalize_url(self, url: str) -> str:
        url = url.split("#")[0]
        url = url.rstrip("/")
        return url

    def _is_same_domain(self, url: str) -> bool:
        try:
            base = urlparse(self._base_url)
            target = urlparse(url)
            return base.netloc == target.netloc
        except Exception:
            return False

    def _should_include(self, url: str) -> bool:
        if self._exclude:
            for pat in self._exclude:
                if pat.search(url):
                    return False
        if self._include:
            return any(pat.search(url) for pat in self._include)
        return True

    def _classify_url(self, url: str) -> DocType:
        path = urlparse(url).path.lower()
        if "/classes/" in path or "/class_" in path:
            return DocType.CLASS_REF
        if "/tutorial" in path or "/getting_started" in path:
            return DocType.TUTORIAL
        return DocType.GUIDE

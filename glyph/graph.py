"""Cross-source graph utilities for extracting code references from markdown content."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Fenced code block: ```optional_lang\n...\n```
_FENCED_BLOCK_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)

# Inline backtick: `content` (not empty, not double-backtick)
_INLINE_BACKTICK_RE = re.compile(r"(?<!`)(`([^`]+)`)(?!`)")

# Qualified identifiers: dot-separated or :: separated, with optional trailing parens
# Matches things like Node2D.rotate, AActor::BeginPlay, player.Player.move
_QUALIFIED_DOT_RE = re.compile(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)(?:\([^)]*\))?")
_QUALIFIED_CPP_RE = re.compile(r"\b([A-Za-z_]\w*(?:::[A-Za-z_]\w*)+)(?:\([^)]*\))?")

# Single PascalCase identifier (starts with uppercase, has at least one lowercase after)
# or any identifier 3+ chars from backticks
_SINGLE_IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")

# String literals (single or double quoted)
_STRING_LITERAL_RE = re.compile(r"""(["']).*?\1""")


def _is_noise(token: str) -> bool:
    """Return True if the token should be filtered out as noise."""
    # Too short
    if len(token) <= 2:
        return True
    # File paths
    if "/" in token:
        return True
    # URLs
    if "http" in token.lower() or "://" in token:
        return True
    # Pure numeric
    if re.match(r"^\d+(\.\d+)?$", token):
        return True
    return False


def _extract_from_code_text(text: str) -> list[str]:
    """Extract qualified names from a piece of code text."""
    # Strip string literals to avoid extracting paths/URLs inside quotes
    text = _STRING_LITERAL_RE.sub("", text)

    refs = []

    # Dot-separated identifiers
    for m in _QUALIFIED_DOT_RE.finditer(text):
        candidate = m.group(1)
        # Filter out numeric literals like 100.5
        if re.match(r"^\d+(\.\d+)?$", candidate):
            continue
        # Filter out parts that look like numbers (e.g., "1.0")
        if re.match(r"^\d", candidate):
            continue
        refs.append(candidate)

    # C++ :: identifiers
    for m in _QUALIFIED_CPP_RE.finditer(text):
        refs.append(m.group(1))

    return refs


def _extract_from_backtick(text: str) -> list[str]:
    """Extract identifiers from inline backtick content."""
    text = text.strip()
    if not text:
        return []

    if _is_noise(text):
        return []

    # Strip trailing parenthesized expression
    text = re.sub(r"\([^)]*\)$", "", text)

    if not text or _is_noise(text):
        return []

    # Try qualified names first
    refs = _extract_from_code_text(text)
    if refs:
        return refs

    # Single identifier — must be PascalCase or 3+ alphanumeric chars
    if _SINGLE_IDENT_RE.match(text) and len(text) >= 3:
        # Check it's not purely lowercase short keyword
        # Accept PascalCase (has at least one uppercase letter)
        if any(c.isupper() for c in text):
            return [text]

    return []


def extract_code_references(content: str, *, include_inline: bool = True) -> list[str]:
    """Extract potential qualified names from code blocks and optionally inline backticks.

    Args:
        content: Markdown text to scan.
        include_inline: Whether to extract from inline backtick spans. Set False
            for non-markdown content where backtick parsing would be unreliable.

    Returns:
        Deduplicated list of qualified name strings found in code contexts.
    """
    if not content:
        return []

    seen: set[str] = set()
    result: list[str] = []

    def _add(ref: str) -> None:
        if ref not in seen:
            seen.add(ref)
            result.append(ref)

    # 1. Find all fenced code blocks and extract from them
    # Also track their spans so we can exclude them from inline backtick search
    fenced_spans: list[tuple[int, int]] = []
    for m in _FENCED_BLOCK_RE.finditer(content):
        fenced_spans.append((m.start(), m.end()))
        block_content = m.group(1)
        for ref in _extract_from_code_text(block_content):
            if not _is_noise(ref):
                _add(ref)

    # 2. Find inline backticks (outside fenced blocks)
    if not include_inline:
        return result

    for m in _INLINE_BACKTICK_RE.finditer(content):
        # Check this backtick is not inside a fenced block
        pos = m.start()
        in_fenced = any(start <= pos < end for start, end in fenced_spans)
        if in_fenced:
            continue

        backtick_content = m.group(2)
        for ref in _extract_from_backtick(backtick_content):
            if not _is_noise(ref):
                _add(ref)

    return result


_DOC_CHUNK_TYPES = {"tutorial_section", "code_example"}


async def build_edges_for_group(store: Any, group: str, config: Any) -> int:
    """Build cross-reference edges for all sources in a group.

    Scans doc-type chunks (tutorial_section, code_example) for references to
    API-type chunks (everything else) across all sources in the group.

    Args:
        store: PostgresStore instance.
        group: Group name to build edges for.
        config: Config object with ``sources`` list.

    Returns:
        Number of edges inserted.
    """
    # 1. Filter sources to those in this group
    group_sources = [s for s in config.sources if s.group == group]
    if not group_sources:
        return 0

    # 2. Collect all chunks from all sources in the group
    doc_chunks: list[dict] = []
    api_chunks: list[dict] = []

    for src in group_sources:
        chunks = await store.get_all_chunks(src.name, src.version)
        for chunk in chunks:
            if chunk["chunk_type"] in _DOC_CHUNK_TYPES:
                doc_chunks.append(chunk)
            else:
                api_chunks.append(chunk)

    # 3. Build lookup dicts from API chunks
    qname_to_id: dict[str, Any] = {}
    parent_to_ids: dict[str, set] = {}

    for chunk in api_chunks:
        qname = chunk["qualified_name"]
        if qname:
            qname_to_id[qname] = chunk["id"]
        pname = chunk["parent_name"]
        if pname:
            parent_to_ids.setdefault(pname, set()).add(chunk["id"])

    # 4. Delete old edges for ALL sources in the group before inserting
    for src in group_sources:
        await store.delete_edges_for_source(src.name, src.version)

    # 5. Scan doc chunks for references and build edges
    seen_edges: set[tuple] = set()
    all_edges: list[tuple] = []

    for doc_chunk in doc_chunks:
        refs = extract_code_references(doc_chunk["content"])
        for ref in refs:
            matched_ids: set = set()

            # Match against qualified_name
            if ref in qname_to_id:
                matched_ids.add(qname_to_id[ref])

            # Match against parent_name
            if ref in parent_to_ids:
                matched_ids.update(parent_to_ids[ref])

            for api_id in matched_ids:
                edge_key = (doc_chunk["id"], api_id)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    all_edges.append((doc_chunk["id"], api_id, "references"))

    # 6. Insert edges
    if not all_edges:
        return 0

    return await store.insert_edges(all_edges)

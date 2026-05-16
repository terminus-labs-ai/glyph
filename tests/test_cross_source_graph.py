"""Tests for cross-source graph linking — Steps 1, 2, 3 & 4."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from mcp.server.fastmcp import FastMCP

from glyph.config import SourceConfig, load_config, load_repo_config
from glyph.domain.models import Source
from glyph.graph import build_edges_for_group, extract_code_references
from glyph.server import GlyphServer
from glyph.store.postgres import PostgresStore


# --- Fixtures ---


@pytest.fixture
def mock_pool():
    pool = MagicMock()
    conn = AsyncMock()
    cm = AsyncMock()
    cm.__aenter__.return_value = conn
    pool.acquire.return_value = cm
    return pool, conn


# --- Group 1: SourceConfig.group field ---


class TestSourceConfigGroup:
    def test_group_field_exists_on_source_config(self):
        """SourceConfig has a group field."""
        cfg = SourceConfig(name="test", version="1.0", ingestors=[], group="mygroup")
        assert cfg.group == "mygroup"

    def test_group_defaults_to_none(self):
        """SourceConfig.group defaults to None when not specified."""
        cfg = SourceConfig(name="test", version="1.0", ingestors=[])
        assert cfg.group is None


# --- Group 2: Config parsing reads group from YAML ---


class TestConfigParsingGroup:
    def test_full_config_parses_group(self, tmp_path):
        """group field in sources YAML is parsed into SourceConfig."""
        cfg_data = {
            "database": {"url": "postgresql://test@localhost/test"},
            "sources": [
                {
                    "name": "godot-api",
                    "version": "4.4",
                    "group": "godot",
                    "ingestors": [{"type": "source_code", "path": "/src"}],
                },
            ],
        }
        cfg_file = tmp_path / "glyph.yaml"
        cfg_file.write_text(yaml.dump(cfg_data))

        cfg = load_config(cfg_file)
        assert cfg.sources[0].group == "godot"

    def test_full_config_group_absent_is_none(self, tmp_path):
        """Sources without group in YAML get group=None."""
        cfg_data = {
            "database": {"url": "postgresql://test@localhost/test"},
            "sources": [
                {
                    "name": "my-lib",
                    "version": "1.0",
                    "ingestors": [{"type": "source_code", "path": "/src"}],
                },
            ],
        }
        cfg_file = tmp_path / "glyph.yaml"
        cfg_file.write_text(yaml.dump(cfg_data))

        cfg = load_config(cfg_file)
        assert cfg.sources[0].group is None

    def test_multiple_sources_different_groups(self, tmp_path):
        """Multiple sources can have different group values."""
        cfg_data = {
            "database": {"url": "postgresql://test@localhost/test"},
            "sources": [
                {
                    "name": "godot-api",
                    "version": "4.4",
                    "group": "godot",
                    "ingestors": [{"type": "source_code", "path": "/a"}],
                },
                {
                    "name": "godot-tutorials",
                    "version": "4.4",
                    "group": "godot",
                    "ingestors": [{"type": "docs", "path": "/b"}],
                },
                {
                    "name": "unrelated",
                    "version": "1.0",
                    "ingestors": [{"type": "source_code", "path": "/c"}],
                },
            ],
        }
        cfg_file = tmp_path / "glyph.yaml"
        cfg_file.write_text(yaml.dump(cfg_data))

        cfg = load_config(cfg_file)
        assert cfg.sources[0].group == "godot"
        assert cfg.sources[1].group == "godot"
        assert cfg.sources[2].group is None

    def test_repo_config_parses_group(self, tmp_path):
        """Per-repo .glyph.yaml also supports the group field."""
        repo = tmp_path / "my-project"
        repo.mkdir()
        glyph_cfg = {
            "name": "my-project",
            "version": "1.0",
            "group": "my-ecosystem",
            "ingestors": [{"type": "source_code", "path": "."}],
        }
        (repo / ".glyph.yaml").write_text(yaml.dump(glyph_cfg))

        src_cfg = load_repo_config(repo)
        assert src_cfg.group == "my-ecosystem"

    def test_repo_config_group_absent_is_none(self, tmp_path):
        """Per-repo config without group gets group=None."""
        repo = tmp_path / "my-project"
        repo.mkdir()
        glyph_cfg = {
            "name": "my-project",
            "version": "1.0",
            "ingestors": [{"type": "source_code", "path": "."}],
        }
        (repo / ".glyph.yaml").write_text(yaml.dump(glyph_cfg))

        src_cfg = load_repo_config(repo)
        assert src_cfg.group is None


# --- Group 3: Source domain model ---


class TestSourceModelGroup:
    def test_group_field_exists(self):
        """Source domain model has a group field."""
        src = Source(
            name="test",
            version="1.0",
            source_type="source_code",
            origin="/src",
            dimensions=512,
            group="godot",
        )
        assert src.group == "godot"

    def test_group_defaults_to_none(self):
        """Source.group defaults to None when not specified."""
        src = Source(
            name="test",
            version="1.0",
            source_type="source_code",
            origin="/src",
            dimensions=512,
        )
        assert src.group is None


# --- Group 4: upsert_source() persists group ---


class TestUpsertSourceGroup:
    async def test_upsert_source_includes_group_value(self, mock_pool):
        """upsert_source() SQL includes the group value in INSERT."""
        from glyph.store.postgres import PostgresStore

        pool, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value={"id": uuid.uuid4()})
        store = PostgresStore.__new__(PostgresStore)
        store._pool = pool

        source = Source(
            name="godot-api",
            version="4.4",
            source_type="source_code",
            origin="/src",
            dimensions=512,
            group="godot",
        )
        await store.upsert_source(source)

        conn.fetchrow.assert_called_once()
        sql = conn.fetchrow.call_args[0][0]
        # SQL must reference the group column
        assert "group" in sql
        # The group value must be passed as a parameter
        args = conn.fetchrow.call_args[0]
        assert "godot" in args, (
            f"Expected 'godot' group value in positional args, got: {args[1:]}"
        )

    async def test_upsert_source_includes_group_in_upsert_clause(self, mock_pool):
        """upsert_source() ON CONFLICT updates group along with other fields."""
        from glyph.store.postgres import PostgresStore

        pool, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value={"id": uuid.uuid4()})
        store = PostgresStore.__new__(PostgresStore)
        store._pool = pool

        source = Source(
            name="godot-api",
            version="4.4",
            source_type="source_code",
            origin="/src",
            dimensions=512,
            group="godot",
        )
        await store.upsert_source(source)

        sql = conn.fetchrow.call_args[0][0]
        # The ON CONFLICT clause should update group
        # Split on ON CONFLICT to check the DO UPDATE part
        assert "ON CONFLICT" in sql
        conflict_part = sql.split("ON CONFLICT")[1]
        assert "group" in conflict_part

    async def test_upsert_source_handles_none_group(self, mock_pool):
        """upsert_source() passes None when source has no group."""
        from glyph.store.postgres import PostgresStore

        pool, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value={"id": uuid.uuid4()})
        store = PostgresStore.__new__(PostgresStore)
        store._pool = pool

        source = Source(
            name="my-lib",
            version="1.0",
            source_type="source_code",
            origin="/src",
            dimensions=512,
            # no group — defaults to None
        )
        await store.upsert_source(source)

        conn.fetchrow.assert_called_once()
        sql = conn.fetchrow.call_args[0][0]
        assert "group" in sql
        # None should be passed for group parameter
        args = conn.fetchrow.call_args[0]
        # source.group is None, so None must appear in positional args
        assert None in args[1:], (
            f"Expected None for group in positional args, got: {args[1:]}"
        )


# --- Group 5: extract_code_references ---


class TestExtractCodeReferences:
    """Tests for extract_code_references() — Step 2: reference extraction from markdown."""

    # -- Fenced code blocks --

    def test_fenced_block_dot_separated_identifier(self):
        """Extracts dot-separated identifiers from fenced code blocks."""
        content = """Some text.

```
var pos = Node2D.get_position()
```
"""
        refs = extract_code_references(content)
        assert "Node2D.get_position" in refs

    def test_fenced_block_multiple_identifiers(self):
        """Extracts multiple dot-separated identifiers from a single fenced block."""
        content = """```
var body = CharacterBody2D.new()
body.move_and_slide()
Node2D.rotate(1.0)
```
"""
        refs = extract_code_references(content)
        assert "CharacterBody2D.new" in refs
        assert "Node2D.rotate" in refs

    def test_fenced_block_with_language_tag(self):
        """Extracts identifiers from language-tagged fenced blocks."""
        content = """```gdscript
var sprite = Sprite2D.new()
sprite.set_texture(load("res://icon.png"))
```
"""
        refs = extract_code_references(content)
        assert "Sprite2D.new" in refs

    def test_fenced_block_python_language_tag(self):
        """Extracts identifiers from python-tagged fenced blocks."""
        content = """```python
result = player.Player.move(direction)
```
"""
        refs = extract_code_references(content)
        assert "player.Player.move" in refs

    def test_multiple_fenced_blocks(self):
        """Extracts from all fenced code blocks in the document."""
        content = """First example:

```
Node2D.rotate(0.5)
```

Second example:

```gdscript
CharacterBody2D.move_and_slide()
```
"""
        refs = extract_code_references(content)
        assert "Node2D.rotate" in refs
        assert "CharacterBody2D.move_and_slide" in refs

    # -- Inline backticks --

    def test_inline_backtick_dot_separated(self):
        """Extracts dot-separated names from inline backticks."""
        content = "Call `Node2D.get_position` to get the current position."
        refs = extract_code_references(content)
        assert "Node2D.get_position" in refs

    def test_inline_backtick_single_class_name(self):
        """Extracts standalone class names from inline backticks."""
        content = "The `CharacterBody2D` node handles physics movement."
        refs = extract_code_references(content)
        assert "CharacterBody2D" in refs

    def test_inline_backtick_with_parens_stripped(self):
        """Strips parentheses from method calls in inline backticks."""
        content = "Use `Node2D.rotate()` to rotate the node."
        refs = extract_code_references(content)
        assert "Node2D.rotate" in refs
        # Should NOT have the version with parens
        assert "Node2D.rotate()" not in refs

    def test_inline_backtick_with_args_stripped(self):
        """Strips parenthesized arguments from method calls."""
        content = "Call `Node2D.rotate(0.5)` with a float."
        refs = extract_code_references(content)
        assert "Node2D.rotate" in refs

    # -- C++ qualified names --

    def test_cpp_double_colon_identifier(self):
        """Extracts C++-style :: qualified names."""
        content = """```cpp
AActor::BeginPlay();
```
"""
        refs = extract_code_references(content)
        assert "AActor::BeginPlay" in refs

    def test_cpp_inline_backtick(self):
        """Extracts C++-style :: names from inline backticks."""
        content = "Override `UObject::GetName` to customize the name."
        refs = extract_code_references(content)
        assert "UObject::GetName" in refs

    def test_cpp_with_parens_stripped(self):
        """Strips parentheses from C++ method calls."""
        content = "Call `AActor::BeginPlay()` during initialization."
        refs = extract_code_references(content)
        assert "AActor::BeginPlay" in refs

    # -- Deduplication --

    def test_deduplication_across_blocks(self):
        """Same reference in multiple places returns only once."""
        content = """Use `Node2D.rotate` for rotation.

```
Node2D.rotate(angle)
```

Remember that `Node2D.rotate` takes radians.
"""
        refs = extract_code_references(content)
        assert refs.count("Node2D.rotate") == 1

    def test_deduplication_within_block(self):
        """Same reference repeated in one code block returns only once."""
        content = """```
Node2D.rotate(0.5)
Node2D.rotate(1.0)
Node2D.rotate(1.5)
```
"""
        refs = extract_code_references(content)
        assert refs.count("Node2D.rotate") == 1

    # -- Prose is ignored --

    def test_prose_not_extracted(self):
        """Text outside code blocks and backticks is NOT extracted."""
        content = "The Node2D.rotate method is useful for rotation."
        refs = extract_code_references(content)
        assert "Node2D.rotate" not in refs

    def test_prose_with_dotted_words_ignored(self):
        """Dot-separated words in plain prose are not extracted."""
        content = """CharacterBody2D.move_and_slide is the primary movement method.
It handles collision response automatically."""
        refs = extract_code_references(content)
        assert "CharacterBody2D.move_and_slide" not in refs

    def test_prose_ignored_but_code_extracted(self):
        """Only code references extracted, prose references ignored, from same doc."""
        content = """Node2D.rotate is great. Use `Node2D.get_position` instead.

Also Sprite2D.set_texture in prose is ignored.
"""
        refs = extract_code_references(content)
        assert "Node2D.get_position" in refs
        assert "Node2D.rotate" not in refs
        assert "Sprite2D.set_texture" not in refs

    # -- Noise filtering --

    def test_file_paths_not_extracted(self):
        """File paths like src/main.py are not extracted as identifiers."""
        content = "Edit the file `src/main.py` to add your code."
        refs = extract_code_references(content)
        assert "src/main.py" not in refs
        # Also shouldn't extract partial path segments as qualified names
        for ref in refs:
            assert "/" not in ref

    def test_urls_not_extracted(self):
        """URLs are not extracted as identifiers."""
        content = "See `http://example.com/docs` for more info."
        refs = extract_code_references(content)
        for ref in refs:
            assert "http" not in ref
            assert "example.com" not in ref

    def test_short_tokens_filtered(self):
        """Very short tokens (1-2 chars) are filtered out."""
        content = """```
x = 5
```
"""
        refs = extract_code_references(content)
        assert "x" not in refs

    def test_plain_numbers_not_extracted(self):
        """Numeric literals are not extracted."""
        content = """```
var speed = 100.5
var count = 42
```
"""
        refs = extract_code_references(content)
        assert "100.5" not in refs
        assert "42" not in refs
        assert "100" not in refs

    def test_string_literals_not_extracted(self):
        """Quoted strings inside code blocks are not extracted as identifiers."""
        content = """```
var path = "res://scenes/main.tscn"
```
"""
        refs = extract_code_references(content)
        for ref in refs:
            assert "res://" not in ref
            assert "main.tscn" not in ref

    # -- Method calls with parentheses --

    def test_parens_stripped_in_fenced_block(self):
        """Parentheses stripped from calls in fenced code blocks."""
        content = """```
Node2D.rotate()
CharacterBody2D.move_and_slide()
```
"""
        refs = extract_code_references(content)
        assert "Node2D.rotate" in refs
        assert "CharacterBody2D.move_and_slide" in refs
        # No entries with parens
        for ref in refs:
            assert "(" not in ref
            assert ")" not in ref

    # -- Mixed content --

    def test_mixed_fenced_and_inline(self):
        """Document with both fenced blocks and inline code extracts from both."""
        content = """# Tutorial

Use `Sprite2D.set_texture` to set the texture.

```gdscript
var node = Node2D.new()
Node2D.rotate(1.0)
```

Then call `CharacterBody2D.move_and_slide` for movement.
"""
        refs = extract_code_references(content)
        assert "Sprite2D.set_texture" in refs
        assert "Node2D.new" in refs
        assert "Node2D.rotate" in refs
        assert "CharacterBody2D.move_and_slide" in refs

    # -- Empty / no-code content --

    def test_empty_string_returns_empty(self):
        """Empty string returns empty list."""
        refs = extract_code_references("")
        assert refs == []

    def test_no_code_blocks_returns_empty(self):
        """Content with no code blocks or backticks returns empty list."""
        content = "This is a plain text document with no code at all."
        refs = extract_code_references(content)
        assert refs == []

    def test_empty_code_block_returns_empty(self):
        """Empty fenced code block returns empty list."""
        content = """```
```
"""
        refs = extract_code_references(content)
        assert refs == []

    def test_empty_inline_backtick_returns_empty(self):
        """Empty inline backticks return empty list."""
        content = "This has `` empty backticks."
        refs = extract_code_references(content)
        assert refs == []

    # -- Return type --

    def test_returns_list(self):
        """Return type is a list."""
        result = extract_code_references("no code here")
        assert isinstance(result, list)

    def test_returns_list_of_strings(self):
        """All items in the returned list are strings."""
        content = "Use `Node2D.rotate` and `Sprite2D.new` here."
        refs = extract_code_references(content)
        assert len(refs) > 0
        for ref in refs:
            assert isinstance(ref, str)

    # -- Edge cases --

    def test_nested_backticks_in_fenced_block(self):
        """Inline backticks inside fenced blocks still extract identifiers."""
        content = """```
# Call Node2D.rotate to rotate
var pos = Sprite2D.get_position()
```
"""
        refs = extract_code_references(content)
        assert "Node2D.rotate" in refs
        assert "Sprite2D.get_position" in refs

    def test_triple_dot_chain(self):
        """Three-segment qualified names are extracted."""
        content = "`player.Player.move` handles character movement."
        refs = extract_code_references(content)
        assert "player.Player.move" in refs

    def test_deeply_nested_qualified_name(self):
        """Four-segment qualified names are extracted."""
        content = "`engine.physics.Body2D.apply_force` applies a force."
        refs = extract_code_references(content)
        assert "engine.physics.Body2D.apply_force" in refs

    def test_underscore_identifiers(self):
        """Identifiers with underscores are extracted."""
        content = "`my_module.MyClass.my_method` does things."
        refs = extract_code_references(content)
        assert "my_module.MyClass.my_method" in refs

    def test_single_class_name_uppercase(self):
        """Single PascalCase class name extracted from backticks."""
        content = "The `Node2D` class is the base for 2D nodes."
        refs = extract_code_references(content)
        assert "Node2D" in refs

    def test_mixed_dot_and_cpp_notation(self):
        """Extracts both dot and :: notation from the same content."""
        content = """```
Node2D.rotate(0.5)
AActor::BeginPlay()
```
"""
        refs = extract_code_references(content)
        assert "Node2D.rotate" in refs
        assert "AActor::BeginPlay" in refs

    def test_keywords_and_builtins_not_matched(self):
        """Common language keywords in backticks are not useful references.

        Short keywords like `var`, `if`, `for` should be filtered by the
        minimum-length or identifier-pattern rules.
        """
        content = "Use `var` to declare, `if` to branch, `for` to loop."
        refs = extract_code_references(content)
        assert "var" not in refs
        assert "if" not in refs
        assert "for" not in refs


# --- Group 6: Edge store methods (Step 3) ---


class TestEdgeStoreMethods:
    """Tests for PostgresStore edge CRUD: insert_edges, delete_edges_for_source,
    get_related, get_edge_summary."""

    # -- insert_edges --

    async def test_insert_edges_basic(self, mock_pool):
        """insert_edges() calls executemany with correct SQL and returns count."""
        pool, conn = mock_pool
        conn.executemany = AsyncMock()
        store = PostgresStore.__new__(PostgresStore)
        store._pool = pool

        id1, id2, id3 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        edges = [
            (id1, id2, "references"),
            (id1, id3, "inherits"),
        ]
        result = await store.insert_edges(edges)

        conn.executemany.assert_called_once()
        sql = conn.executemany.call_args[0][0]
        args_list = conn.executemany.call_args[0][1]

        # SQL must insert into edges table
        assert "INSERT INTO edges" in sql
        # SQL must use ON CONFLICT DO NOTHING for idempotency
        assert "ON CONFLICT" in sql
        assert "DO NOTHING" in sql
        # Must reference source_chunk_id, target_chunk_id, edge_type columns
        assert "source_chunk_id" in sql
        assert "target_chunk_id" in sql
        assert "edge_type" in sql
        # Batch args match input
        assert len(args_list) == 2
        assert result == 2

    async def test_insert_edges_empty_list(self, mock_pool):
        """insert_edges() with empty list returns 0 without touching DB."""
        pool, conn = mock_pool
        conn.executemany = AsyncMock()
        store = PostgresStore.__new__(PostgresStore)
        store._pool = pool

        result = await store.insert_edges([])

        assert result == 0
        conn.executemany.assert_not_called()

    async def test_insert_edges_returns_count(self, mock_pool):
        """insert_edges() returns the number of edges in the input list."""
        pool, conn = mock_pool
        conn.executemany = AsyncMock()
        store = PostgresStore.__new__(PostgresStore)
        store._pool = pool

        edges = [(uuid.uuid4(), uuid.uuid4(), "references") for _ in range(5)]
        result = await store.insert_edges(edges)

        assert result == 5

    async def test_insert_edges_passes_correct_tuples(self, mock_pool):
        """insert_edges() passes (source_chunk_id, target_chunk_id, edge_type) tuples."""
        pool, conn = mock_pool
        conn.executemany = AsyncMock()
        store = PostgresStore.__new__(PostgresStore)
        store._pool = pool

        id_a, id_b = uuid.uuid4(), uuid.uuid4()
        edges = [(id_a, id_b, "references")]
        await store.insert_edges(edges)

        args_list = conn.executemany.call_args[0][1]
        row = args_list[0]
        # The tuple should contain the source, target, and edge_type values
        assert id_a in row
        assert id_b in row
        assert "references" in row

    # -- delete_edges_for_source --

    async def test_delete_edges_for_source_returns_count(self, mock_pool):
        """delete_edges_for_source() returns the number of deleted edges."""
        pool, conn = mock_pool
        conn.execute = AsyncMock(return_value="DELETE 7")
        store = PostgresStore.__new__(PostgresStore)
        store._pool = pool

        result = await store.delete_edges_for_source("godot-api", "4.4")

        assert result == 7

    async def test_delete_edges_for_source_sql_references_both_endpoints(self, mock_pool):
        """delete_edges_for_source() SQL targets edges where EITHER endpoint belongs to source."""
        pool, conn = mock_pool
        conn.execute = AsyncMock(return_value="DELETE 0")
        store = PostgresStore.__new__(PostgresStore)
        store._pool = pool

        await store.delete_edges_for_source("godot-api", "4.4")

        conn.execute.assert_called_once()
        sql = conn.execute.call_args[0][0]
        # SQL must delete from edges
        assert "DELETE" in sql.upper()
        assert "edges" in sql
        # Must reference both source_chunk_id and target_chunk_id in the WHERE
        assert "source_chunk_id" in sql
        assert "target_chunk_id" in sql
        # Must use a subquery to find chunks by source_name/source_version
        assert "source_name" in sql
        assert "source_version" in sql

    async def test_delete_edges_for_source_passes_name_and_version(self, mock_pool):
        """delete_edges_for_source() passes source name and version as parameters."""
        pool, conn = mock_pool
        conn.execute = AsyncMock(return_value="DELETE 0")
        store = PostgresStore.__new__(PostgresStore)
        store._pool = pool

        await store.delete_edges_for_source("my-lib", "2.0")

        args = conn.execute.call_args[0]
        # source_name and source_version must appear in the positional args
        assert "my-lib" in args
        assert "2.0" in args

    async def test_delete_edges_for_source_zero_deleted(self, mock_pool):
        """delete_edges_for_source() returns 0 when no edges match."""
        pool, conn = mock_pool
        conn.execute = AsyncMock(return_value="DELETE 0")
        store = PostgresStore.__new__(PostgresStore)
        store._pool = pool

        result = await store.delete_edges_for_source("nonexistent", "0.0")

        assert result == 0

    # -- get_related --

    async def test_get_related_returns_list_of_dicts(self, mock_pool):
        """get_related() returns a list of dicts with expected keys."""
        pool, conn = mock_pool
        mock_row = {
            "qualified_name": "Node2D.get_position",
            "heading": "get_position",
            "source_name": "godot-api",
            "source_version": "4.4",
            "chunk_type": "method",
            "edge_type": "references",
            "summary": "Returns the position.",
        }
        conn.fetch = AsyncMock(return_value=[mock_row])
        store = PostgresStore.__new__(PostgresStore)
        store._pool = pool

        result = await store.get_related(uuid.uuid4())

        assert isinstance(result, list)
        assert len(result) == 1
        item = result[0]
        assert item["qualified_name"] == "Node2D.get_position"
        assert item["heading"] == "get_position"
        assert item["source_name"] == "godot-api"
        assert item["source_version"] == "4.4"
        assert item["chunk_type"] == "method"
        assert item["edge_type"] == "references"
        assert item["summary"] == "Returns the position."

    async def test_get_related_direction_both(self, mock_pool):
        """get_related(direction='both') queries both source_chunk_id and target_chunk_id."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])
        store = PostgresStore.__new__(PostgresStore)
        store._pool = pool

        chunk_id = uuid.uuid4()
        await store.get_related(chunk_id, direction="both")

        sql = conn.fetch.call_args[0][0]
        # "both" direction means the SQL should match edges where this chunk
        # is either source or target
        assert "source_chunk_id" in sql
        assert "target_chunk_id" in sql

    async def test_get_related_direction_outgoing(self, mock_pool):
        """get_related(direction='outgoing') only finds edges where chunk is source."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])
        store = PostgresStore.__new__(PostgresStore)
        store._pool = pool

        chunk_id = uuid.uuid4()
        await store.get_related(chunk_id, direction="outgoing")

        sql = conn.fetch.call_args[0][0]
        # For outgoing, the SQL should filter on source_chunk_id = chunk_id
        assert "source_chunk_id" in sql
        # The chunk_id should be passed as a parameter
        args = conn.fetch.call_args[0]
        assert chunk_id in args

    async def test_get_related_direction_incoming(self, mock_pool):
        """get_related(direction='incoming') only finds edges where chunk is target."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])
        store = PostgresStore.__new__(PostgresStore)
        store._pool = pool

        chunk_id = uuid.uuid4()
        await store.get_related(chunk_id, direction="incoming")

        sql = conn.fetch.call_args[0][0]
        # For incoming, the SQL should filter on target_chunk_id = chunk_id
        assert "target_chunk_id" in sql
        args = conn.fetch.call_args[0]
        assert chunk_id in args

    async def test_get_related_no_edge_type_filter_by_default(self, mock_pool):
        """get_related() without edge_type does not constrain by edge type."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])
        store = PostgresStore.__new__(PostgresStore)
        store._pool = pool

        chunk_id = uuid.uuid4()
        await store.get_related(chunk_id)

        args = conn.fetch.call_args[0]
        # When edge_type is None, no edge_type string should be in the query params
        param_values = args[1:]
        for v in param_values:
            assert v not in ("references", "inherits"), (
                f"edge_type value {v!r} should not be in params when edge_type=None"
            )

    async def test_get_related_edge_type_filter(self, mock_pool):
        """get_related() with edge_type filters to that type."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])
        store = PostgresStore.__new__(PostgresStore)
        store._pool = pool

        chunk_id = uuid.uuid4()
        await store.get_related(chunk_id, edge_type="inherits")

        sql = conn.fetch.call_args[0][0]
        assert "edge_type" in sql
        args = conn.fetch.call_args[0]
        assert "inherits" in args

    async def test_get_related_limit_applied(self, mock_pool):
        """get_related() passes the limit parameter to the query."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])
        store = PostgresStore.__new__(PostgresStore)
        store._pool = pool

        chunk_id = uuid.uuid4()
        await store.get_related(chunk_id, limit=25)

        sql = conn.fetch.call_args[0][0]
        assert "LIMIT" in sql.upper()
        args = conn.fetch.call_args[0]
        assert 25 in args

    async def test_get_related_limit_clamped_min(self, mock_pool):
        """get_related() clamps limit to minimum of 1."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])
        store = PostgresStore.__new__(PostgresStore)
        store._pool = pool

        chunk_id = uuid.uuid4()
        await store.get_related(chunk_id, limit=0)

        args = conn.fetch.call_args[0]
        # Limit should be clamped to 1, not 0
        assert 0 not in args
        assert 1 in args

    async def test_get_related_limit_clamped_max(self, mock_pool):
        """get_related() clamps limit to maximum of 50."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])
        store = PostgresStore.__new__(PostgresStore)
        store._pool = pool

        chunk_id = uuid.uuid4()
        await store.get_related(chunk_id, limit=100)

        args = conn.fetch.call_args[0]
        # Limit should be clamped to 50, not 100
        assert 100 not in args
        assert 50 in args

    async def test_get_related_default_limit(self, mock_pool):
        """get_related() uses default limit of 10."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])
        store = PostgresStore.__new__(PostgresStore)
        store._pool = pool

        await store.get_related(uuid.uuid4())

        args = conn.fetch.call_args[0]
        assert 10 in args

    # -- get_edge_summary --

    async def test_get_edge_summary_returns_dict(self, mock_pool):
        """get_edge_summary() returns a dict mapping chunk_ids to related items."""
        pool, conn = mock_pool
        chunk_id = uuid.uuid4()
        mock_row = {
            "chunk_id": chunk_id,
            "qualified_name": "Node2D.rotate",
            "source_name": "godot-api",
            "edge_type": "references",
            "direction": "outgoing",
        }
        conn.fetch = AsyncMock(return_value=[mock_row])
        store = PostgresStore.__new__(PostgresStore)
        store._pool = pool

        result = await store.get_edge_summary([chunk_id])

        assert isinstance(result, dict)
        assert chunk_id in result
        items = result[chunk_id]
        assert isinstance(items, list)
        assert len(items) == 1
        item = items[0]
        assert item["qualified_name"] == "Node2D.rotate"
        assert item["source_name"] == "godot-api"
        assert item["edge_type"] == "references"
        assert item["direction"] == "outgoing"

    async def test_get_edge_summary_empty_input(self, mock_pool):
        """get_edge_summary() with empty list returns empty dict without querying."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])
        store = PostgresStore.__new__(PostgresStore)
        store._pool = pool

        result = await store.get_edge_summary([])

        assert result == {}
        conn.fetch.assert_not_called()

    async def test_get_edge_summary_multiple_chunks(self, mock_pool):
        """get_edge_summary() groups results by chunk_id."""
        pool, conn = mock_pool
        id_a, id_b = uuid.uuid4(), uuid.uuid4()
        mock_rows = [
            {
                "chunk_id": id_a,
                "qualified_name": "Node2D.rotate",
                "source_name": "godot-api",
                "edge_type": "references",
                "direction": "outgoing",
            },
            {
                "chunk_id": id_a,
                "qualified_name": "Sprite2D.set_texture",
                "source_name": "godot-api",
                "edge_type": "references",
                "direction": "outgoing",
            },
            {
                "chunk_id": id_b,
                "qualified_name": "AActor.BeginPlay",
                "source_name": "unreal-api",
                "edge_type": "inherits",
                "direction": "incoming",
            },
        ]
        conn.fetch = AsyncMock(return_value=mock_rows)
        store = PostgresStore.__new__(PostgresStore)
        store._pool = pool

        result = await store.get_edge_summary([id_a, id_b])

        assert len(result[id_a]) == 2
        assert len(result[id_b]) == 1

    async def test_get_edge_summary_chunk_without_edges_absent(self, mock_pool):
        """Chunks with no edges are absent from the returned dict."""
        pool, conn = mock_pool
        id_with_edges = uuid.uuid4()
        id_no_edges = uuid.uuid4()
        mock_rows = [
            {
                "chunk_id": id_with_edges,
                "qualified_name": "Node2D.rotate",
                "source_name": "godot-api",
                "edge_type": "references",
                "direction": "outgoing",
            },
        ]
        conn.fetch = AsyncMock(return_value=mock_rows)
        store = PostgresStore.__new__(PostgresStore)
        store._pool = pool

        result = await store.get_edge_summary([id_with_edges, id_no_edges])

        assert id_with_edges in result
        assert id_no_edges not in result

    async def test_get_edge_summary_batch_query(self, mock_pool):
        """get_edge_summary() passes all chunk_ids in a single query (batch operation)."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])
        store = PostgresStore.__new__(PostgresStore)
        store._pool = pool

        ids = [uuid.uuid4() for _ in range(5)]
        await store.get_edge_summary(ids)

        # Should make exactly one fetch call (batch, not N+1)
        assert conn.fetch.call_count == 1
        # The chunk IDs should be passed as a parameter (likely as a list/array)
        args = conn.fetch.call_args[0]
        sql = args[0]
        # SQL should reference the edges table and join to chunks
        assert "edges" in sql
        assert "chunks" in sql


# --- Group 7: Edge building orchestrator + post-ingest hook (Step 4) ---


def _make_chunk_dict(
    *,
    chunk_id: uuid.UUID | None = None,
    source_name: str = "godot-api",
    source_version: str = "4.4",
    chunk_type: str = "method",
    qualified_name: str = "Node2D.rotate",
    parent_name: str = "Node2D",
    heading: str = "rotate",
    summary: str = "Rotates the node.",
    content: str = "Rotates the node by the given angle.",
) -> dict:
    """Helper to build a chunk dict matching the shape returned by store.get_all_chunks()."""
    return {
        "id": chunk_id or uuid.uuid4(),
        "source_name": source_name,
        "source_version": source_version,
        "chunk_type": chunk_type,
        "qualified_name": qualified_name,
        "parent_name": parent_name,
        "heading": heading,
        "summary": summary,
        "content": content,
        "metadata": {},
        "chunk_index": 0,
    }


def _make_source_config(
    name: str = "godot-api",
    version: str = "4.4",
    group: str | None = "godot",
) -> SourceConfig:
    """Helper to build a SourceConfig with minimal fields."""
    return SourceConfig(
        name=name,
        version=version,
        ingestors=[],
        group=group,
    )


def _make_config_with_sources(source_configs: list[SourceConfig]):
    """Build a minimal Config object containing the given sources."""
    from glyph.config import Config, DatabaseConfig, EmbedderConfig, OutputConfig

    return Config(
        database=DatabaseConfig(url="postgresql://test@localhost/test"),
        embedder=EmbedderConfig(),
        sources=source_configs,
        output=OutputConfig(),
    )


class TestEdgeBuilding:
    """Tests for build_edges_for_group() and the post-ingest hook in run_ingest."""

    # -- Basic edge building --

    async def test_basic_edge_building(self):
        """Doc chunk content with references matching API chunk qualified_names produces edges."""
        api_chunk = _make_chunk_dict(
            source_name="godot-api",
            chunk_type="method",
            qualified_name="Node2D.rotate",
            parent_name="Node2D",
        )
        doc_chunk = _make_chunk_dict(
            source_name="godot-tutorials",
            source_version="4.4",
            chunk_type="tutorial_section",
            qualified_name="getting_started.movement",
            parent_name="getting_started",
            content="Use `Node2D.rotate` to rotate the node.",
        )

        store = AsyncMock()
        store.get_all_chunks = AsyncMock(side_effect=lambda name, version: {
            ("godot-api", "4.4"): [api_chunk],
            ("godot-tutorials", "4.4"): [doc_chunk],
        }[(name, version)])
        store.delete_edges_for_source = AsyncMock(return_value=0)
        store.insert_edges = AsyncMock(return_value=1)

        src_cfgs = [
            _make_source_config(name="godot-api", group="godot"),
            _make_source_config(name="godot-tutorials", group="godot"),
        ]
        config = _make_config_with_sources(src_cfgs)

        result = await build_edges_for_group(store, "godot", config)

        store.insert_edges.assert_called_once()
        edges = store.insert_edges.call_args[0][0]
        assert len(edges) >= 1
        # Each edge is a tuple (source_chunk_id, target_chunk_id, edge_type)
        edge = edges[0]
        assert edge[0] == doc_chunk["id"]  # from the doc chunk
        assert edge[1] == api_chunk["id"]  # to the API chunk
        assert edge[2] == "references"
        assert result >= 1

    # -- Parent name matching --

    async def test_parent_name_matching(self):
        """References matching parent_name (not just qualified_name) also produce edges."""
        api_chunk = _make_chunk_dict(
            source_name="godot-api",
            chunk_type="class_overview",
            qualified_name="CharacterBody2D",
            parent_name="CharacterBody2D",
        )
        doc_chunk = _make_chunk_dict(
            source_name="godot-tutorials",
            chunk_type="tutorial_section",
            content="The `CharacterBody2D` node handles physics.",
        )

        store = AsyncMock()
        store.get_all_chunks = AsyncMock(side_effect=lambda name, version: {
            ("godot-api", "4.4"): [api_chunk],
            ("godot-tutorials", "4.4"): [doc_chunk],
        }[(name, version)])
        store.delete_edges_for_source = AsyncMock(return_value=0)
        store.insert_edges = AsyncMock(return_value=1)

        src_cfgs = [
            _make_source_config(name="godot-api", group="godot"),
            _make_source_config(name="godot-tutorials", group="godot"),
        ]
        config = _make_config_with_sources(src_cfgs)

        result = await build_edges_for_group(store, "godot", config)

        store.insert_edges.assert_called_once()
        edges = store.insert_edges.call_args[0][0]
        # Should have matched via parent_name "CharacterBody2D"
        target_ids = [e[1] for e in edges]
        assert api_chunk["id"] in target_ids
        assert result >= 1

    # -- Cross-source only --

    async def test_cross_source_matching(self):
        """Doc chunks from source A match API chunks from source B (both in same group)."""
        api_chunk_a = _make_chunk_dict(
            source_name="godot-api",
            chunk_type="method",
            qualified_name="Node2D.rotate",
            parent_name="Node2D",
        )
        doc_chunk_b = _make_chunk_dict(
            source_name="godot-tutorials",
            chunk_type="tutorial_section",
            content="Call `Node2D.rotate` to rotate.",
        )

        store = AsyncMock()
        store.get_all_chunks = AsyncMock(side_effect=lambda name, version: {
            ("godot-api", "4.4"): [api_chunk_a],
            ("godot-tutorials", "4.4"): [doc_chunk_b],
        }[(name, version)])
        store.delete_edges_for_source = AsyncMock(return_value=0)
        store.insert_edges = AsyncMock(return_value=1)

        src_cfgs = [
            _make_source_config(name="godot-api", group="godot"),
            _make_source_config(name="godot-tutorials", group="godot"),
        ]
        config = _make_config_with_sources(src_cfgs)

        await build_edges_for_group(store, "godot", config)

        edges = store.insert_edges.call_args[0][0]
        assert len(edges) >= 1
        # Doc chunk is from godot-tutorials, API chunk is from godot-api
        edge = edges[0]
        assert edge[0] == doc_chunk_b["id"]
        assert edge[1] == api_chunk_a["id"]

    # -- No group = no edges --

    async def test_no_sources_in_group_returns_zero(self):
        """If no sources have this group, returns 0 and inserts no edges."""
        store = AsyncMock()
        store.get_all_chunks = AsyncMock()
        store.insert_edges = AsyncMock()
        store.delete_edges_for_source = AsyncMock()

        # Config with sources but none matching the requested group
        src_cfgs = [
            _make_source_config(name="unrelated", group="other"),
        ]
        config = _make_config_with_sources(src_cfgs)

        result = await build_edges_for_group(store, "nonexistent-group", config)

        assert result == 0
        store.insert_edges.assert_not_called()

    # -- Deletes old edges first --

    async def test_deletes_old_edges_before_inserting(self):
        """Calls delete_edges_for_source for each source in the group before inserting new ones."""
        api_chunk = _make_chunk_dict(source_name="godot-api", chunk_type="method")
        doc_chunk = _make_chunk_dict(
            source_name="godot-tutorials",
            chunk_type="tutorial_section",
            content="Use `Node2D.rotate` here.",
        )

        call_order = []

        store = AsyncMock()
        store.get_all_chunks = AsyncMock(side_effect=lambda name, version: {
            ("godot-api", "4.4"): [api_chunk],
            ("godot-tutorials", "4.4"): [doc_chunk],
        }[(name, version)])

        async def track_delete(*args, **kwargs):
            call_order.append("delete")
            return 0

        async def track_insert(*args, **kwargs):
            call_order.append("insert")
            return 1

        store.delete_edges_for_source = AsyncMock(side_effect=track_delete)
        store.insert_edges = AsyncMock(side_effect=track_insert)

        src_cfgs = [
            _make_source_config(name="godot-api", group="godot"),
            _make_source_config(name="godot-tutorials", group="godot"),
        ]
        config = _make_config_with_sources(src_cfgs)

        await build_edges_for_group(store, "godot", config)

        # delete_edges_for_source called for each source in the group
        assert store.delete_edges_for_source.call_count == 2
        delete_calls = [
            (c.args[0], c.args[1])
            for c in store.delete_edges_for_source.call_args_list
        ]
        assert ("godot-api", "4.4") in delete_calls
        assert ("godot-tutorials", "4.4") in delete_calls

        # All deletes happen before any insert
        first_insert_idx = call_order.index("insert") if "insert" in call_order else len(call_order)
        for i, action in enumerate(call_order):
            if action == "delete":
                assert i < first_insert_idx, "delete must happen before insert"

    # -- Deduplication --

    async def test_deduplication_same_ref_multiple_times(self):
        """Same qualified_name referenced multiple times in one doc produces only one edge."""
        api_chunk = _make_chunk_dict(
            source_name="godot-api",
            chunk_type="method",
            qualified_name="Node2D.rotate",
            parent_name="Node2D",
        )
        doc_chunk = _make_chunk_dict(
            source_name="godot-tutorials",
            chunk_type="tutorial_section",
            content="""Use `Node2D.rotate` to rotate.

```gdscript
Node2D.rotate(0.5)
Node2D.rotate(1.0)
```

Remember `Node2D.rotate` takes radians.
""",
        )

        store = AsyncMock()
        store.get_all_chunks = AsyncMock(side_effect=lambda name, version: {
            ("godot-api", "4.4"): [api_chunk],
            ("godot-tutorials", "4.4"): [doc_chunk],
        }[(name, version)])
        store.delete_edges_for_source = AsyncMock(return_value=0)
        store.insert_edges = AsyncMock(return_value=1)

        src_cfgs = [
            _make_source_config(name="godot-api", group="godot"),
            _make_source_config(name="godot-tutorials", group="godot"),
        ]
        config = _make_config_with_sources(src_cfgs)

        await build_edges_for_group(store, "godot", config)

        edges = store.insert_edges.call_args[0][0]
        # Count edges from doc_chunk to api_chunk
        matching_edges = [
            e for e in edges
            if e[0] == doc_chunk["id"] and e[1] == api_chunk["id"]
        ]
        assert len(matching_edges) == 1, (
            f"Expected exactly 1 edge for repeated reference, got {len(matching_edges)}"
        )

    # -- Returns edge count --

    async def test_returns_edge_count(self):
        """Returns the total number of edges inserted."""
        api_chunks = [
            _make_chunk_dict(
                source_name="godot-api",
                chunk_type="method",
                qualified_name="Node2D.rotate",
                parent_name="Node2D",
            ),
            _make_chunk_dict(
                source_name="godot-api",
                chunk_type="method",
                qualified_name="Sprite2D.set_texture",
                parent_name="Sprite2D",
            ),
        ]
        doc_chunk = _make_chunk_dict(
            source_name="godot-tutorials",
            chunk_type="tutorial_section",
            content="Use `Node2D.rotate` and `Sprite2D.set_texture` here.",
        )

        store = AsyncMock()
        store.get_all_chunks = AsyncMock(side_effect=lambda name, version: {
            ("godot-api", "4.4"): api_chunks,
            ("godot-tutorials", "4.4"): [doc_chunk],
        }[(name, version)])
        store.delete_edges_for_source = AsyncMock(return_value=0)
        store.insert_edges = AsyncMock(return_value=2)

        src_cfgs = [
            _make_source_config(name="godot-api", group="godot"),
            _make_source_config(name="godot-tutorials", group="godot"),
        ]
        config = _make_config_with_sources(src_cfgs)

        result = await build_edges_for_group(store, "godot", config)

        # insert_edges was called with 2 edges
        edges = store.insert_edges.call_args[0][0]
        assert len(edges) == 2
        # Return value matches what insert_edges returned
        assert result == 2

    # -- Only doc-type chunks are scanned for references --

    async def test_only_doc_chunks_scanned(self):
        """Only tutorial_section and code_example chunks are scanned for references, not API chunks."""
        api_chunk_1 = _make_chunk_dict(
            source_name="godot-api",
            chunk_type="method",
            qualified_name="Node2D.rotate",
            parent_name="Node2D",
            # This API chunk mentions Sprite2D.set_texture but it should NOT be scanned
            content="See also `Sprite2D.set_texture` for textures.",
        )
        api_chunk_2 = _make_chunk_dict(
            source_name="godot-api",
            chunk_type="method",
            qualified_name="Sprite2D.set_texture",
            parent_name="Sprite2D",
        )
        # No doc chunks exist — only API chunks
        store = AsyncMock()
        store.get_all_chunks = AsyncMock(side_effect=lambda name, version: {
            ("godot-api", "4.4"): [api_chunk_1, api_chunk_2],
        }[(name, version)])
        store.delete_edges_for_source = AsyncMock(return_value=0)
        store.insert_edges = AsyncMock(return_value=0)

        src_cfgs = [
            _make_source_config(name="godot-api", group="godot"),
        ]
        config = _make_config_with_sources(src_cfgs)

        result = await build_edges_for_group(store, "godot", config)

        # No doc chunks → no references extracted → no edges
        assert result == 0
        # insert_edges either not called or called with empty list
        if store.insert_edges.called:
            edges = store.insert_edges.call_args[0][0]
            assert len(edges) == 0

    # -- code_example chunks also scanned --

    async def test_code_example_chunks_scanned(self):
        """code_example chunks are also scanned for references, not just tutorial_section."""
        api_chunk = _make_chunk_dict(
            source_name="godot-api",
            chunk_type="method",
            qualified_name="Node2D.rotate",
            parent_name="Node2D",
        )
        doc_chunk = _make_chunk_dict(
            source_name="godot-tutorials",
            chunk_type="code_example",
            content="```gdscript\nNode2D.rotate(0.5)\n```",
        )

        store = AsyncMock()
        store.get_all_chunks = AsyncMock(side_effect=lambda name, version: {
            ("godot-api", "4.4"): [api_chunk],
            ("godot-tutorials", "4.4"): [doc_chunk],
        }[(name, version)])
        store.delete_edges_for_source = AsyncMock(return_value=0)
        store.insert_edges = AsyncMock(return_value=1)

        src_cfgs = [
            _make_source_config(name="godot-api", group="godot"),
            _make_source_config(name="godot-tutorials", group="godot"),
        ]
        config = _make_config_with_sources(src_cfgs)

        result = await build_edges_for_group(store, "godot", config)

        store.insert_edges.assert_called_once()
        edges = store.insert_edges.call_args[0][0]
        assert len(edges) >= 1
        assert result >= 1

    # -- No matches produces no edges --

    async def test_no_matching_references_produces_no_edges(self):
        """Doc chunk with references that don't match any API chunks produces no edges."""
        api_chunk = _make_chunk_dict(
            source_name="godot-api",
            chunk_type="method",
            qualified_name="Node2D.rotate",
            parent_name="Node2D",
        )
        doc_chunk = _make_chunk_dict(
            source_name="godot-tutorials",
            chunk_type="tutorial_section",
            content="Use `CompletelyUnknownClass.mystery_method` here.",
        )

        store = AsyncMock()
        store.get_all_chunks = AsyncMock(side_effect=lambda name, version: {
            ("godot-api", "4.4"): [api_chunk],
            ("godot-tutorials", "4.4"): [doc_chunk],
        }[(name, version)])
        store.delete_edges_for_source = AsyncMock(return_value=0)
        store.insert_edges = AsyncMock(return_value=0)

        src_cfgs = [
            _make_source_config(name="godot-api", group="godot"),
            _make_source_config(name="godot-tutorials", group="godot"),
        ]
        config = _make_config_with_sources(src_cfgs)

        result = await build_edges_for_group(store, "godot", config)

        assert result == 0

    # -- Post-ingest hook --

    async def test_post_ingest_hook_calls_build_edges(self):
        """When run_ingest processes a source with a group, build_edges_for_group is called."""
        from glyph.config import Config, DatabaseConfig, EmbedderConfig, IngestorConfig, OutputConfig

        src_cfg = SourceConfig(
            name="godot-tutorials",
            version="4.4",
            ingestors=[IngestorConfig(type="docs", settings={"path": "/fake/docs"})],
            group="godot",
        )
        config = Config(
            database=DatabaseConfig(url="postgresql://test@localhost/test"),
            embedder=EmbedderConfig(),
            sources=[src_cfg],
            output=OutputConfig(),
        )

        with (
            patch("glyph.pipeline._ingest_source", new_callable=AsyncMock) as mock_ingest,
            patch("glyph.pipeline.build_edges_for_group", new_callable=AsyncMock) as mock_build_edges,
            patch("glyph.store.PostgresStore") as mock_store_cls,
            patch("glyph.embedders.llama.LlamaEmbedder") as mock_embedder_cls,
        ):
            mock_ingest.return_value = {"name": "godot-tutorials", "version": "4.4", "documents": 1, "chunks": 5}
            mock_build_edges.return_value = 3

            mock_store = AsyncMock()
            mock_store_cls.return_value = mock_store

            mock_embedder = AsyncMock()
            mock_embedder.dimensions = 512
            mock_embedder_cls.return_value = mock_embedder

            from glyph.pipeline import run_ingest

            await run_ingest(config)

            mock_build_edges.assert_called_once()
            call_args = mock_build_edges.call_args
            # Should be called with (store, group_name, config)
            assert call_args[0][1] == "godot"

    async def test_post_ingest_hook_not_called_without_group(self):
        """When run_ingest processes a source without a group, build_edges_for_group is NOT called."""
        from glyph.config import Config, DatabaseConfig, EmbedderConfig, IngestorConfig, OutputConfig

        src_cfg = SourceConfig(
            name="standalone-lib",
            version="1.0",
            ingestors=[IngestorConfig(type="source_code", settings={"path": "/fake/src"})],
            # No group
        )
        config = Config(
            database=DatabaseConfig(url="postgresql://test@localhost/test"),
            embedder=EmbedderConfig(),
            sources=[src_cfg],
            output=OutputConfig(),
        )

        with (
            patch("glyph.pipeline._ingest_source", new_callable=AsyncMock) as mock_ingest,
            patch("glyph.pipeline.build_edges_for_group", new_callable=AsyncMock) as mock_build_edges,
            patch("glyph.store.PostgresStore") as mock_store_cls,
            patch("glyph.embedders.llama.LlamaEmbedder") as mock_embedder_cls,
        ):
            mock_ingest.return_value = {"name": "standalone-lib", "version": "1.0", "documents": 1, "chunks": 5}

            mock_store = AsyncMock()
            mock_store_cls.return_value = mock_store

            mock_embedder = AsyncMock()
            mock_embedder.dimensions = 512
            mock_embedder_cls.return_value = mock_embedder

            from glyph.pipeline import run_ingest

            await run_ingest(config)

            mock_build_edges.assert_not_called()

    async def test_post_ingest_hook_called_once_per_group(self):
        """When multiple sources share the same group, build_edges_for_group is called
        for each source ingest (the function itself handles the full group rebuild)."""
        from glyph.config import Config, DatabaseConfig, EmbedderConfig, IngestorConfig, OutputConfig

        src_a = SourceConfig(
            name="godot-api",
            version="4.4",
            ingestors=[IngestorConfig(type="source_code", settings={"path": "/fake/a"})],
            group="godot",
        )
        src_b = SourceConfig(
            name="godot-tutorials",
            version="4.4",
            ingestors=[IngestorConfig(type="docs", settings={"path": "/fake/b"})],
            group="godot",
        )
        config = Config(
            database=DatabaseConfig(url="postgresql://test@localhost/test"),
            embedder=EmbedderConfig(),
            sources=[src_a, src_b],
            output=OutputConfig(),
        )

        with (
            patch("glyph.pipeline._ingest_source", new_callable=AsyncMock) as mock_ingest,
            patch("glyph.pipeline.build_edges_for_group", new_callable=AsyncMock) as mock_build_edges,
            patch("glyph.store.PostgresStore") as mock_store_cls,
            patch("glyph.embedders.llama.LlamaEmbedder") as mock_embedder_cls,
        ):
            mock_ingest.side_effect = [
                {"name": "godot-api", "version": "4.4", "documents": 1, "chunks": 5},
                {"name": "godot-tutorials", "version": "4.4", "documents": 1, "chunks": 3},
            ]
            mock_build_edges.return_value = 2

            mock_store = AsyncMock()
            mock_store_cls.return_value = mock_store

            mock_embedder = AsyncMock()
            mock_embedder.dimensions = 512
            mock_embedder_cls.return_value = mock_embedder

            from glyph.pipeline import run_ingest

            await run_ingest(config)

            # Called once for each source that has a group
            assert mock_build_edges.call_count == 2
            # All calls use the same group name
            for call in mock_build_edges.call_args_list:
                assert call[0][1] == "godot"


# --- Group 8: CLI link command (Step 5) ---


class TestLinkCLI:
    """Tests for the `glyph link` CLI command."""

    def test_link_group_calls_build_edges(self):
        """glyph link --group godot calls build_edges_for_group with the correct group."""
        from click.testing import CliRunner
        from glyph.__main__ import cli

        with (
            patch("glyph.config.load_config") as mock_load_config,
            patch("glyph.graph.build_edges_for_group", new_callable=AsyncMock) as mock_build,
            patch("glyph.store.PostgresStore") as mock_store_cls,
        ):
            mock_config = _make_config_with_sources([
                _make_source_config(name="godot-api", group="godot"),
            ])
            mock_load_config.return_value = mock_config
            mock_build.return_value = 5

            mock_store = AsyncMock()
            mock_store_cls.return_value = mock_store

            runner = CliRunner()
            result = runner.invoke(cli, ["link", "--group", "godot"])

            assert result.exit_code == 0, f"CLI failed: {result.output}"
            mock_build.assert_called_once()
            # Second positional arg (after store) should be the group name
            call_args = mock_build.call_args[0]
            assert call_args[1] == "godot"

    def test_link_all_calls_build_edges_for_each_group(self):
        """glyph link --all calls build_edges_for_group for each unique group in config."""
        from click.testing import CliRunner
        from glyph.__main__ import cli

        with (
            patch("glyph.config.load_config") as mock_load_config,
            patch("glyph.graph.build_edges_for_group", new_callable=AsyncMock) as mock_build,
            patch("glyph.store.PostgresStore") as mock_store_cls,
        ):
            mock_config = _make_config_with_sources([
                _make_source_config(name="godot-api", group="godot"),
                _make_source_config(name="godot-tutorials", group="godot"),
                _make_source_config(name="unreal-api", group="unreal"),
                _make_source_config(name="standalone", group=None),
            ])
            mock_load_config.return_value = mock_config
            mock_build.return_value = 3

            mock_store = AsyncMock()
            mock_store_cls.return_value = mock_store

            runner = CliRunner()
            result = runner.invoke(cli, ["link", "--all"])

            assert result.exit_code == 0, f"CLI failed: {result.output}"
            # Should be called once per unique non-None group: "godot" and "unreal"
            assert mock_build.call_count == 2
            group_names = {c[0][1] for c in mock_build.call_args_list}
            assert group_names == {"godot", "unreal"}

    def test_link_without_group_or_all_shows_error(self):
        """glyph link without --group or --all shows an error or help message."""
        from click.testing import CliRunner
        from glyph.__main__ import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["link"])

        # Should fail or show usage guidance, not succeed silently
        assert result.exit_code != 0 or "error" in result.output.lower() or "usage" in result.output.lower() or "--group" in result.output.lower()

    def test_link_output_includes_edge_count(self):
        """glyph link --group godot output includes the edge count summary."""
        from click.testing import CliRunner
        from glyph.__main__ import cli

        with (
            patch("glyph.config.load_config") as mock_load_config,
            patch("glyph.graph.build_edges_for_group", new_callable=AsyncMock) as mock_build,
            patch("glyph.store.PostgresStore") as mock_store_cls,
        ):
            mock_config = _make_config_with_sources([
                _make_source_config(name="godot-api", group="godot"),
            ])
            mock_load_config.return_value = mock_config
            mock_build.return_value = 42

            mock_store = AsyncMock()
            mock_store_cls.return_value = mock_store

            runner = CliRunner()
            result = runner.invoke(cli, ["link", "--group", "godot"])

            assert result.exit_code == 0, f"CLI failed: {result.output}"
            # Output should mention the edge count
            assert "42" in result.output

    def test_link_group_and_all_mutually_exclusive(self):
        """glyph link --group X --all is rejected as conflicting options."""
        from click.testing import CliRunner
        from glyph.__main__ import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["link", "--group", "godot", "--all"])

        # Should fail — can't specify both
        assert result.exit_code != 0


# --- Group 9: MCP graph tools (Step 5) ---


class TestMCPGraphTools:
    """Tests for the get_related and link_sources MCP tools."""

    def _build_graph_test_server(self, mock_store, mock_embedder, mock_config=None):
        """Create a GlyphServer with mocked deps + config for graph tools."""
        with patch("glyph.server.load_config"):
            srv = GlyphServer.__new__(GlyphServer)
            srv._config_path = "test.yaml"
            srv._store = mock_store
            srv._embedder = mock_embedder
            srv._reranker = None
            srv._config = mock_config
            srv.mcp = FastMCP("test")
            srv._register_tools()
            srv._register_resources()
        return srv

    # -- get_related --

    async def test_get_related_valid_name_returns_formatted_results(self):
        """get_related with a valid qualified_name returns formatted linked items."""
        chunk_id = uuid.uuid4()
        mock_store = AsyncMock()
        mock_store.get_by_qualified_name = AsyncMock(return_value={
            "id": chunk_id,
            "qualified_name": "Node2D.rotate",
            "chunk_type": "method",
            "parent_name": "Node2D",
            "source_name": "godot-api",
            "source_version": "4.4",
        })
        mock_store.get_related = AsyncMock(return_value=[
            {
                "qualified_name": "getting_started.movement",
                "heading": "Movement Tutorial",
                "source_name": "godot-tutorials",
                "source_version": "4.4",
                "chunk_type": "tutorial_section",
                "edge_type": "references",
                "summary": "How to move nodes.",
            },
        ])
        mock_embedder = AsyncMock()

        srv = self._build_graph_test_server(mock_store, mock_embedder)
        result = await srv.mcp._tool_manager.call_tool("get_related", {
            "qualified_name": "Node2D.rotate",
        })

        assert "getting_started.movement" in result
        assert "references" in result
        assert "godot-tutorials" in result

    async def test_get_related_unknown_name_returns_error(self):
        """get_related with an unknown qualified_name returns an error message."""
        mock_store = AsyncMock()
        mock_store.get_by_qualified_name = AsyncMock(return_value=None)
        mock_embedder = AsyncMock()

        srv = self._build_graph_test_server(mock_store, mock_embedder)
        result = await srv.mcp._tool_manager.call_tool("get_related", {
            "qualified_name": "Nonexistent.nothing",
        })

        # Should not call get_related on store
        mock_store.get_related.assert_not_called()
        # Should return an error/not-found message
        assert "not found" in result.lower() or "no chunk" in result.lower() or "error" in result.lower()

    async def test_get_related_no_edges_returns_no_related_message(self):
        """get_related when chunk exists but has no edges returns a 'no related' message."""
        chunk_id = uuid.uuid4()
        mock_store = AsyncMock()
        mock_store.get_by_qualified_name = AsyncMock(return_value={
            "id": chunk_id,
            "qualified_name": "Node2D.rotate",
            "chunk_type": "method",
            "parent_name": "Node2D",
            "source_name": "godot-api",
            "source_version": "4.4",
        })
        mock_store.get_related = AsyncMock(return_value=[])
        mock_embedder = AsyncMock()

        srv = self._build_graph_test_server(mock_store, mock_embedder)
        result = await srv.mcp._tool_manager.call_tool("get_related", {
            "qualified_name": "Node2D.rotate",
        })

        assert "no related" in result.lower() or "no linked" in result.lower() or "none" in result.lower()

    async def test_get_related_passes_edge_type_filter(self):
        """get_related passes edge_type filter to the store."""
        chunk_id = uuid.uuid4()
        mock_store = AsyncMock()
        mock_store.get_by_qualified_name = AsyncMock(return_value={
            "id": chunk_id,
            "qualified_name": "Node2D.rotate",
            "chunk_type": "method",
            "parent_name": "Node2D",
            "source_name": "godot-api",
            "source_version": "4.4",
        })
        mock_store.get_related = AsyncMock(return_value=[])
        mock_embedder = AsyncMock()

        srv = self._build_graph_test_server(mock_store, mock_embedder)
        await srv.mcp._tool_manager.call_tool("get_related", {
            "qualified_name": "Node2D.rotate",
            "edge_type": "inherits",
        })

        mock_store.get_related.assert_called_once()
        call_kwargs = mock_store.get_related.call_args
        # edge_type should be passed through
        args_and_kwargs = str(call_kwargs)
        assert "inherits" in args_and_kwargs

    async def test_get_related_passes_limit(self):
        """get_related passes limit to the store, clamped to 1-50."""
        chunk_id = uuid.uuid4()
        mock_store = AsyncMock()
        mock_store.get_by_qualified_name = AsyncMock(return_value={
            "id": chunk_id,
            "qualified_name": "Node2D.rotate",
            "chunk_type": "method",
            "parent_name": "Node2D",
            "source_name": "godot-api",
            "source_version": "4.4",
        })
        mock_store.get_related = AsyncMock(return_value=[])
        mock_embedder = AsyncMock()

        srv = self._build_graph_test_server(mock_store, mock_embedder)
        await srv.mcp._tool_manager.call_tool("get_related", {
            "qualified_name": "Node2D.rotate",
            "limit": 25,
        })

        mock_store.get_related.assert_called_once()
        call_args = mock_store.get_related.call_args
        # Limit should be passed through
        args_and_kwargs = str(call_args)
        assert "25" in args_and_kwargs

    async def test_get_related_clamps_limit(self):
        """get_related clamps limit to max 50."""
        chunk_id = uuid.uuid4()
        mock_store = AsyncMock()
        mock_store.get_by_qualified_name = AsyncMock(return_value={
            "id": chunk_id,
            "qualified_name": "Node2D.rotate",
            "chunk_type": "method",
            "parent_name": "Node2D",
            "source_name": "godot-api",
            "source_version": "4.4",
        })
        mock_store.get_related = AsyncMock(return_value=[])
        mock_embedder = AsyncMock()

        srv = self._build_graph_test_server(mock_store, mock_embedder)
        await srv.mcp._tool_manager.call_tool("get_related", {
            "qualified_name": "Node2D.rotate",
            "limit": 999,
        })

        mock_store.get_related.assert_called_once()
        call_args = mock_store.get_related.call_args
        # 999 should be clamped to 50
        args_and_kwargs = str(call_args)
        assert "999" not in args_and_kwargs

    # -- link_sources --

    async def test_link_sources_specific_group(self):
        """link_sources with a specific group calls build_edges_for_group for that group."""
        mock_store = AsyncMock()
        mock_embedder = AsyncMock()
        mock_config = _make_config_with_sources([
            _make_source_config(name="godot-api", group="godot"),
            _make_source_config(name="godot-tutorials", group="godot"),
        ])

        with patch("glyph.server.build_edges_for_group", new_callable=AsyncMock) as mock_build:
            mock_build.return_value = 7

            srv = self._build_graph_test_server(mock_store, mock_embedder, mock_config)
            result = await srv.mcp._tool_manager.call_tool("link_sources", {
                "group": "godot",
            })

            mock_build.assert_called_once()
            call_args = mock_build.call_args[0]
            assert call_args[1] == "godot"
            assert "7" in result

    async def test_link_sources_all_groups(self):
        """link_sources with group=None builds edges for all unique groups."""
        mock_store = AsyncMock()
        mock_embedder = AsyncMock()
        mock_config = _make_config_with_sources([
            _make_source_config(name="godot-api", group="godot"),
            _make_source_config(name="godot-tutorials", group="godot"),
            _make_source_config(name="unreal-api", group="unreal"),
            _make_source_config(name="standalone", group=None),
        ])

        with patch("glyph.server.build_edges_for_group", new_callable=AsyncMock) as mock_build:
            mock_build.return_value = 5

            srv = self._build_graph_test_server(mock_store, mock_embedder, mock_config)
            result = await srv.mcp._tool_manager.call_tool("link_sources", {})

            # Should be called once per unique non-None group
            assert mock_build.call_count == 2
            group_names = {c[0][1] for c in mock_build.call_args_list}
            assert group_names == {"godot", "unreal"}

    async def test_link_sources_returns_summary_with_count(self):
        """link_sources returns a summary string including edge count."""
        mock_store = AsyncMock()
        mock_embedder = AsyncMock()
        mock_config = _make_config_with_sources([
            _make_source_config(name="godot-api", group="godot"),
        ])

        with patch("glyph.server.build_edges_for_group", new_callable=AsyncMock) as mock_build:
            mock_build.return_value = 15

            srv = self._build_graph_test_server(mock_store, mock_embedder, mock_config)
            result = await srv.mcp._tool_manager.call_tool("link_sources", {
                "group": "godot",
            })

            assert "15" in result
            assert "godot" in result.lower()


# --- Group 10: Inline edge summaries in search/lookup/get_context (Step 6) ---


class TestInlineSummaries:
    """Tests for edge summary decorations in search, lookup, and get_context tools.

    When chunks returned by these tools have edges in the graph, the response
    should include a lightweight summary of related items (source names, counts).
    When no edges exist, the response should be unchanged.
    """

    def _build_inline_test_server(self, mock_store, mock_embedder):
        """Create a GlyphServer with mocked deps for inline summary tests."""
        with patch("glyph.server.load_config"):
            srv = GlyphServer.__new__(GlyphServer)
            srv._config_path = "test.yaml"
            srv._store = mock_store
            srv._embedder = mock_embedder
            srv._reranker = None
            srv._config = None
            srv.mcp = FastMCP("test")
            srv._register_tools()
            srv._register_resources()
        return srv

    def _make_search_chunk(
        self,
        *,
        chunk_id=None,
        qualified_name="Node2D.get_position",
        parent_name="Node2D",
        chunk_type="method",
        heading="get_position",
        summary="Returns the node's position.",
        content="func get_position() -> Vector2",
        source_name="godot",
        source_version="4.4",
        score=0.045,
        retrieval="hybrid",
    ):
        """Build a chunk dict matching hybrid_search return shape."""
        return {
            "id": chunk_id or uuid.uuid4(),
            "document_id": uuid.uuid4(),
            "qualified_name": qualified_name,
            "parent_name": parent_name,
            "chunk_type": chunk_type,
            "heading": heading,
            "summary": summary,
            "content": content,
            "source_name": source_name,
            "source_version": source_version,
            "metadata": {},
            "chunk_index": 0,
            "token_count": 10,
            "score": score,
            "retrieval": retrieval,
        }

    # -- search --

    async def test_search_with_edges_includes_related_info(self):
        """Search results with edges include related item info in the response."""
        chunk_id = uuid.uuid4()
        chunk = self._make_search_chunk(chunk_id=chunk_id)

        mock_store = AsyncMock()
        mock_store.hybrid_search = AsyncMock(return_value=[chunk])
        mock_store.get_edge_summary = AsyncMock(return_value={
            chunk_id: [
                {
                    "qualified_name": "getting_started.movement",
                    "source_name": "godot-tutorials",
                    "edge_type": "references",
                    "direction": "incoming",
                },
                {
                    "qualified_name": "getting_started.rotation",
                    "source_name": "godot-tutorials",
                    "edge_type": "references",
                    "direction": "incoming",
                },
            ],
        })
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[[0.1] * 512])

        srv = self._build_inline_test_server(mock_store, mock_embedder)
        result = await srv.mcp._tool_manager.call_tool("search", {"query": "position"})

        # Response should mention the related source
        assert "godot-tutorials" in result
        # Should indicate there are related/linked items
        result_lower = result.lower()
        assert "related" in result_lower or "linked" in result_lower

    async def test_search_without_edges_no_related_info(self):
        """Search results without edges have no spurious related info."""
        chunk_id = uuid.uuid4()
        chunk = self._make_search_chunk(chunk_id=chunk_id)

        mock_store = AsyncMock()
        mock_store.hybrid_search = AsyncMock(return_value=[chunk])
        mock_store.get_edge_summary = AsyncMock(return_value={})
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[[0.1] * 512])

        srv = self._build_inline_test_server(mock_store, mock_embedder)
        result = await srv.mcp._tool_manager.call_tool("search", {"query": "position"})

        # get_edge_summary should still be called with the chunk IDs
        mock_store.get_edge_summary.assert_called_once()
        call_args = mock_store.get_edge_summary.call_args[0][0]
        assert chunk_id in call_args

        # Response should NOT contain "related" or "linked" as decorations
        # (the word "related" could appear in other contexts, so check the
        # portion after the main result content for edge decoration text)
        result_lower = result.lower()
        assert "godot-tutorials" not in result
        # No edge summary lines should be present
        assert "linked" not in result_lower

    # -- lookup --

    async def test_lookup_with_edges_includes_related_info(self):
        """Lookup result with edges includes related item info in the response."""
        chunk_id = uuid.uuid4()
        chunk = {
            "id": chunk_id,
            "qualified_name": "Node2D.rotate",
            "parent_name": "Node2D",
            "chunk_type": "method",
            "heading": "rotate",
            "summary": "Rotates the node.",
            "content": "func rotate(angle: float) -> void",
            "source_name": "godot",
            "source_version": "4.4",
            "metadata": {},
            "chunk_index": 0,
            "token_count": 10,
        }

        mock_store = AsyncMock()
        mock_store.get_by_qualified_name = AsyncMock(return_value=chunk)
        mock_store.get_edge_summary = AsyncMock(return_value={
            chunk_id: [
                {
                    "qualified_name": "getting_started.movement",
                    "source_name": "godot-tutorials",
                    "edge_type": "references",
                    "direction": "incoming",
                },
            ],
        })
        mock_embedder = AsyncMock()

        srv = self._build_inline_test_server(mock_store, mock_embedder)
        result = await srv.mcp._tool_manager.call_tool("lookup", {
            "qualified_name": "Node2D.rotate",
        })

        # Should include the main chunk info
        assert "Node2D.rotate" in result
        # Should include related item source
        assert "godot-tutorials" in result
        result_lower = result.lower()
        assert "related" in result_lower or "linked" in result_lower

    async def test_lookup_without_edges_no_related_info(self):
        """Lookup result without edges has no spurious related info."""
        chunk_id = uuid.uuid4()
        chunk = {
            "id": chunk_id,
            "qualified_name": "Node2D.rotate",
            "parent_name": "Node2D",
            "chunk_type": "method",
            "heading": "rotate",
            "summary": "Rotates the node.",
            "content": "func rotate(angle: float) -> void",
            "source_name": "godot",
            "source_version": "4.4",
            "metadata": {},
            "chunk_index": 0,
            "token_count": 10,
        }

        mock_store = AsyncMock()
        mock_store.get_by_qualified_name = AsyncMock(return_value=chunk)
        mock_store.get_edge_summary = AsyncMock(return_value={})
        mock_embedder = AsyncMock()

        srv = self._build_inline_test_server(mock_store, mock_embedder)
        result = await srv.mcp._tool_manager.call_tool("lookup", {
            "qualified_name": "Node2D.rotate",
        })

        # get_edge_summary should be called with the chunk ID
        mock_store.get_edge_summary.assert_called_once()
        call_args = mock_store.get_edge_summary.call_args[0][0]
        assert chunk_id in call_args

        # Main result still present
        assert "Node2D.rotate" in result
        # No edge decoration
        assert "godot-tutorials" not in result
        assert "linked" not in result.lower()

    # -- get_context --

    async def test_get_context_with_edges_includes_related_info(self):
        """get_context results with edges include related item info."""
        chunk_id_1 = uuid.uuid4()
        chunk_id_2 = uuid.uuid4()
        chunks = [
            {
                "id": chunk_id_1,
                "qualified_name": "Node2D",
                "parent_name": "Node2D",
                "chunk_type": "class_overview",
                "heading": "Node2D",
                "summary": "2D game object.",
                "content": "Node2D is the base for all 2D nodes.",
                "source_name": "godot",
                "source_version": "4.4",
                "metadata": {},
                "chunk_index": 0,
                "token_count": 20,
            },
            {
                "id": chunk_id_2,
                "qualified_name": "Node2D.rotate",
                "parent_name": "Node2D",
                "chunk_type": "method",
                "heading": "rotate",
                "summary": "Rotates the node.",
                "content": "func rotate(angle: float) -> void",
                "source_name": "godot",
                "source_version": "4.4",
                "metadata": {},
                "chunk_index": 1,
                "token_count": 10,
            },
        ]

        mock_store = AsyncMock()
        mock_store.get_by_parent = AsyncMock(return_value=chunks)
        mock_store.get_edge_summary = AsyncMock(return_value={
            chunk_id_2: [
                {
                    "qualified_name": "getting_started.movement",
                    "source_name": "godot-tutorials",
                    "edge_type": "references",
                    "direction": "incoming",
                },
            ],
        })
        mock_embedder = AsyncMock()

        srv = self._build_inline_test_server(mock_store, mock_embedder)
        result = await srv.mcp._tool_manager.call_tool("get_context", {
            "parent_name": "Node2D",
        })

        # Main context present
        assert "# Node2D" in result
        assert "rotate" in result
        # Edge info present
        assert "godot-tutorials" in result
        result_lower = result.lower()
        assert "related" in result_lower or "linked" in result_lower

    async def test_get_context_without_edges_no_related_info(self):
        """get_context results without edges have no spurious related info."""
        chunk_id_1 = uuid.uuid4()
        chunk_id_2 = uuid.uuid4()
        chunks = [
            {
                "id": chunk_id_1,
                "qualified_name": "Node2D",
                "parent_name": "Node2D",
                "chunk_type": "class_overview",
                "heading": "Node2D",
                "summary": "2D game object.",
                "content": "Node2D is the base for all 2D nodes.",
                "source_name": "godot",
                "source_version": "4.4",
                "metadata": {},
                "chunk_index": 0,
                "token_count": 20,
            },
            {
                "id": chunk_id_2,
                "qualified_name": "Node2D.rotate",
                "parent_name": "Node2D",
                "chunk_type": "method",
                "heading": "rotate",
                "summary": "Rotates the node.",
                "content": "func rotate(angle: float) -> void",
                "source_name": "godot",
                "source_version": "4.4",
                "metadata": {},
                "chunk_index": 1,
                "token_count": 10,
            },
        ]

        mock_store = AsyncMock()
        mock_store.get_by_parent = AsyncMock(return_value=chunks)
        mock_store.get_edge_summary = AsyncMock(return_value={})
        mock_embedder = AsyncMock()

        srv = self._build_inline_test_server(mock_store, mock_embedder)
        result = await srv.mcp._tool_manager.call_tool("get_context", {
            "parent_name": "Node2D",
        })

        # get_edge_summary should be called with both chunk IDs
        mock_store.get_edge_summary.assert_called_once()
        call_args = mock_store.get_edge_summary.call_args[0][0]
        assert chunk_id_1 in call_args
        assert chunk_id_2 in call_args

        # Main context still present
        assert "# Node2D" in result
        assert "rotate" in result
        # No edge decoration
        assert "godot-tutorials" not in result
        assert "linked" not in result.lower()

    # -- Edge summary shows count --

    async def test_search_edge_summary_includes_count(self):
        """When multiple edges exist, the response indicates the count."""
        chunk_id = uuid.uuid4()
        chunk = self._make_search_chunk(chunk_id=chunk_id)

        mock_store = AsyncMock()
        mock_store.hybrid_search = AsyncMock(return_value=[chunk])
        mock_store.get_edge_summary = AsyncMock(return_value={
            chunk_id: [
                {
                    "qualified_name": f"tutorial.section{i}",
                    "source_name": "godot-tutorials",
                    "edge_type": "references",
                    "direction": "incoming",
                }
                for i in range(5)
            ],
        })
        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[[0.1] * 512])

        srv = self._build_inline_test_server(mock_store, mock_embedder)
        result = await srv.mcp._tool_manager.call_tool("search", {"query": "position"})

        # Should indicate the count of related items
        assert "5" in result
        assert "godot-tutorials" in result

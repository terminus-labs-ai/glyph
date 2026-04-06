# Glyph

CLI tool for converting API documentation and source code into structured RAG knowledge bases.

## Quick Start

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e .
cp glyph.example.yaml glyph.yaml  # edit with your settings
glyph init-db
glyph ingest
glyph export -s godot -V 4.6.1
```

## Architecture

```
glyph/
├── __main__.py               # CLI (click): init-db, ingest, export, stats, search, serve
├── config.py                 # YAML → dataclass config loader
├── server.py                 # MCP server (FastMCP): search, lookup, get_context, list_sources
├── domain/
│   └── models.py             # Source, Document, Chunk + ChunkType/DocType enums
├── ingestors/
│   ├── base.py               # Ingestor Protocol
│   ├── godot_xml.py          # Parses Godot doc/classes/*.xml
│   ├── html.py               # Async BFS web crawler with URL filtering
│   └── source_code.py        # Directory walker, one Document per file
├── chunkers/
│   ├── base.py               # Chunker Protocol
│   ├── api_chunker.py        # XML → per-element chunks (method, property, signal...)
│   ├── text_chunker.py       # Heading-based splitting for tutorials/guides
│   ├── source_code_chunker.py # Delegates to language parsers
│   └── _parsers/
│       ├── __init__.py        # Symbol dataclass, LanguageParser Protocol, registry
│       ├── python_parser.py   # tree-sitter AST extraction
│       ├── gdscript_parser.py # Regex-based extraction
│       ├── typescript_parser.py # tree-sitter AST (TS + TSX)
│       ├── rust_parser.py     # tree-sitter AST
│       └── go_parser.py       # tree-sitter AST
├── embedders/
│   ├── base.py               # Embedder Protocol
│   └── llama.py              # HTTP client, tries OpenAI/Ollama/llama.cpp endpoints
├── store/
│   └── postgres.py           # asyncpg, schema DDL, upsert/search/export queries
└── exporters/
    └── markdown.py           # 3-tier markdown file generator
```

## Data Flow

```
Source (XML/HTML/code)
  → Ingestor (produces Documents)
    → Chunker (produces Chunks with qualified names, summaries, metadata)
      → Embedder (generates vectors)
        → PostgresStore (pgvector)
        → MarkdownExporter (tiered files)
```

## Key Patterns

- **Protocols** (`typing.Protocol`) for all extension points -- no ABC inheritance
- **Async throughout**: aiohttp for HTTP, asyncpg for DB
- **YAML config** (`glyph.yaml`) for all settings, git-ignored
- **Content hashing**: MD5 of raw content for incremental updates (skip unchanged docs)
- **Denormalized** `source_name`/`source_version` on chunks for direct queries without joins
- **Chunker selection** in `__main__.py:_ingest()`:
  - `source_code` ingestor → `SourceCodeChunker`
  - `unreal_doc` ingestor → `UnrealDocChunker`
  - `CLASS_REF` doc_type → `APIChunker`
  - Everything else → `TextChunker`

## Database

PostgreSQL with `uuid-ossp` + `vector` extensions. Three tables:

- **sources**: keyed on `(name, version)` unique -- e.g. "godot" / "4.6.1"
- **documents**: keyed on `(source_id, path)` unique -- tracks `content_hash` for change detection
- **chunks**: the RAG units with `qualified_name`, `summary`, `content`, `metadata` JSONB, `embedding` VECTOR, denormalized `source_name`/`source_version`

Indexes: `(parent_name, chunk_type)`, `(source_name, source_version)`, `(qualified_name)`, IVFFlat on embedding, GIN on `fts` tsvector.

The `chunks` table has an `fts` tsvector generated column (weighted: A=qualified_name/heading, B=summary, C=content) for full-text search. `upgrade_schema()` adds it idempotently to existing databases. `idx_chunks_fts` (GIN on `fts`) lives only in `upgrade_schema()`, not in `SCHEMA_SQL` — the index must be created after the column exists on pre-existing tables.

Schema DDL lives in `store/postgres.py` as `SCHEMA_SQL` string (uses `str.format()` -- literal braces must be escaped as `{{}}`).

## Ingestors

| Type | Source | Produces |
|------|--------|----------|
| `godot_xml` | `doc/classes/*.xml` from Godot engine repo | CLASS_REF docs with structured text |
| `html` | Async BFS crawl from base_url | CLASS_REF/TUTORIAL/GUIDE docs |
| `source_code` | Directory walk, one Document per file | CLASS_REF docs with raw source |
| `unreal_doc` | `documentation.json` from [unreal-doc](https://github.com/PsichiX/unreal-doc) tool | CLASS_REF/API_OVERVIEW docs from UE C++ headers |
| `docs` | Directory walk for `.md`/`.rst`/`.txt` files | TUTORIAL docs with RST→markdown heading conversion |

Registered in `__main__.py:_build_ingestor()`. Each type reads its settings from `IngestorConfig.settings` dict. All file-based ingestors (`godot_xml`, `source_code`, `docs`) support `include_patterns` and `exclude_patterns` for regex filtering on file paths.

## Chunkers

| Chunker | Input | Output |
|---------|-------|--------|
| `APIChunker` | CLASS_REF docs (re-parses XML if available) | Per-element chunks: class_overview, method, property, signal, constant, enum |
| `TextChunker` | TUTORIAL/GUIDE docs | Heading-based sections, splits >2000 chars on paragraph boundaries |
| `SourceCodeChunker` | Source files | Delegates to language parser, respects `include_bodies` flag |
| `UnrealDocChunker` | unreal-doc JSON output | Per-method/property/enum chunks with C++ signatures, uses `::` qualified names |

## Language Parsers (`chunkers/_parsers/`)

| Language | Parser | Method | Extracts |
|----------|--------|--------|----------|
| Python | `PythonParser` | tree-sitter AST | classes, methods, functions, decorators, docstrings |
| GDScript | `GDScriptParser` | compiled regex | class_name/extends, funcs, signals, vars, enums, constants, `##` doc comments, @export |
| TypeScript/TSX | `TypeScriptParser` | tree-sitter AST | classes, methods, functions, arrow functions, interfaces, enums, type aliases, JSDoc (`@param`, `@returns`, `@deprecated`), abstract classes, export/default detection |
| Rust | `RustParser` | tree-sitter AST | structs, enums, impl methods, traits, trait impls, standalone fns, const/static, type aliases, `///` doc comments, `#[derive()]` attributes, visibility (`pub`/`pub(crate)`/`pub(super)`), async/unsafe |
| Go | `GoParser` | tree-sitter AST | structs, interfaces, functions, receiver methods (pointer/value), const blocks, var declarations, type aliases, `//` doc comments, exported detection (uppercase) |
| C++ | `CppParser` | preprocess + tree-sitter AST | classes, structs, enums, methods, fields, `/** */` doc comments, **Unreal Engine macros** (UCLASS, UPROPERTY, UFUNCTION, UENUM, USTRUCT — specifiers extracted as metadata) |

Add new languages: implement `LanguageParser` protocol, register in `get_parser()`, add to `EXTENSION_MAP`.

**UTF-8 offset correctness:** tree-sitter byte offsets index into the UTF-8 encoded bytes, not Python `str` character indices. All parsers must encode source to `bytes` before parsing and slice `src[node.start_byte:node.end_byte]` (then decode). Do NOT slice the original `str` — files with non-ASCII characters (em-dashes, curly quotes, unicode in docstrings) will produce mangled names for everything after the first non-ASCII byte. `PythonParser` is correctly using `src: bytes` throughout; verify new parsers do the same.

## Embedder

`LlamaEmbedder` tries three HTTP endpoint formats in order:
1. `/v1/embeddings` (OpenAI-compatible)
2. `/api/embeddings` (Ollama)
3. `/embedding` (llama.cpp)

Batches texts, 60s timeout. On failure: emits a loud red CLI warning and returns zero vectors (default). Pass `strict=True` to raise `RuntimeError` instead — wired via `glyph ingest --strict`.

## Output

Tiered markdown files exported to `{output.directory}/{source}/{version}/`:
- **Tier 1**: `index.md` -- class names + one-liners (minimal tokens)
- **Tier 2**: `classes/_index.md` -- summaries with member lists (moderate tokens)
- **Tier 3**: `classes/ClassName.md` -- full detail with all members (full tokens)

Tutorials: `tutorials/_index.md` (tier 2) + `tutorials/name.md` (tier 3).

## Extension Points

| Extension | Protocol | Register in |
|-----------|----------|-------------|
| New ingestor | `Ingestor.ingest() → list[Document]` | `__main__.py:_build_ingestor()` |
| New language parser | `LanguageParser.parse(str) → list[Symbol]` | `_parsers/__init__.py:get_parser()` + EXTENSION_MAP |
| New embedder | `Embedder.embed(list[str]) → list[list[float]]` | `__main__.py:_ingest()` |
| New exporter | Match MarkdownExporter interface | `__main__.py:_export()` |

## MCP Server

`glyph serve` starts an MCP server exposing the knowledge base for runtime queries.

- **Transport:** stdio (default, for Claude Code/Desktop), SSE (`-t sse`), streamable-http (`-t streamable-http`)
- **Tools:** `search` (hybrid semantic + keyword via RRF), `lookup` (exact qualified_name), `get_context` (full parent overview), `list_sources`
- **Resources:** `glyph://sources`, `glyph://sources/{name}/{version}/index`, `glyph://sources/{name}/{version}/classes/{class}`
- **Implementation:** `server.py` uses FastMCP with lifespan for store/embedder init. Formatting helpers produce markdown output matching the tiered export style.
- **Hybrid search:** `hybrid_search()` combines `text_search()` (FTS via `websearch_to_tsquery`) and `search()` (pgvector) using Reciprocal Rank Fusion (RRF, k=60). Gracefully degrades to keyword-only when embeddings unavailable. Results tagged `[hybrid]`, `[keyword]`, or `[semantic]`.
- **Store methods added for MCP:** `get_by_qualified_name()`, `get_by_parent()`, `get_sources_with_counts()`, `text_search()`, `hybrid_search()`, `upgrade_schema()`.
- **Tests:** `tests/test_server.py` — mocked store, covers all 4 tools + formatters. `tests/test_hybrid_search.py` — 21 tests for FTS, RRF, hybrid search, upgrade_schema, MCP integration, retrieval tags.

## Dependencies

Python 3.11+, aiohttp, asyncpg, beautifulsoup4, lxml, pyyaml, click, mcp[cli], tree-sitter, tree-sitter-python, tree-sitter-typescript, tree-sitter-rust, tree-sitter-go

## GitHub Action

Glyph is a reusable composite action (`action.yml` at repo root). Callers reference it as `uses: owner/glyph@main`.

**Inputs:** `config` (required, full glyph.yaml content as string for secret interpolation), `commands` (space-separated: init-db/ingest/export/stats), `source`, `version`, `python-version`, `extra-args`.

**How it works:** Sets up Python + uv, installs glyph from the action's source, writes config to a temp file, runs commands in sequence, cleans up config.

**Caller example** (in another repo's workflow):
```yaml
- uses: owner/glyph@main
  with:
    config: |
      database:
        url: "${{ secrets.GLYPH_DB_URL }}"
      sources:
        - name: my-project
          version: "${{ github.sha }}"
          ingestors:
            - type: source_code
              path: "."
              extensions: [".py"]
    commands: "init-db ingest"
```

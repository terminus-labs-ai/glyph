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
├── __main__.py               # CLI (click): init-db, ingest, export, stats, serve
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
  - `CLASS_REF` doc_type → `APIChunker`
  - Everything else → `TextChunker`

## Database

PostgreSQL with `uuid-ossp` + `vector` extensions. Three tables:

- **sources**: keyed on `(name, version)` unique -- e.g. "godot" / "4.6.1"
- **documents**: keyed on `(source_id, path)` unique -- tracks `content_hash` for change detection
- **chunks**: the RAG units with `qualified_name`, `summary`, `content`, `metadata` JSONB, `embedding` VECTOR, denormalized `source_name`/`source_version`

Indexes: `(parent_name, chunk_type)`, `(source_name, source_version)`, `(qualified_name)`, IVFFlat on embedding.

Schema DDL lives in `store/postgres.py` as `SCHEMA_SQL` string (uses `str.format()` -- literal braces must be escaped as `{{}}`).

## Ingestors

| Type | Source | Produces |
|------|--------|----------|
| `godot_xml` | `doc/classes/*.xml` from Godot engine repo | CLASS_REF docs with structured text |
| `html` | Async BFS crawl from base_url | CLASS_REF/TUTORIAL/GUIDE docs |
| `source_code` | Directory walk, one Document per file | CLASS_REF docs with raw source |

Registered in `__main__.py:_build_ingestor()`. Each type reads its settings from `IngestorConfig.settings` dict.

## Chunkers

| Chunker | Input | Output |
|---------|-------|--------|
| `APIChunker` | CLASS_REF docs (re-parses XML if available) | Per-element chunks: class_overview, method, property, signal, constant, enum |
| `TextChunker` | TUTORIAL/GUIDE docs | Heading-based sections, splits >2000 chars on paragraph boundaries |
| `SourceCodeChunker` | Source files | Delegates to language parser, respects `include_bodies` flag |

## Language Parsers (`chunkers/_parsers/`)

| Language | Parser | Method | Extracts |
|----------|--------|--------|----------|
| Python | `PythonParser` | tree-sitter AST | classes, methods, functions, decorators, docstrings |
| GDScript | `GDScriptParser` | compiled regex | class_name/extends, funcs, signals, vars, enums, constants, `##` doc comments, @export |
| TypeScript/TSX | `TypeScriptParser` | tree-sitter AST | classes, methods, functions, arrow functions, interfaces, enums, type aliases, JSDoc (`@param`, `@returns`, `@deprecated`), abstract classes, export/default detection |
| Rust | `RustParser` | tree-sitter AST | structs, enums, impl methods, traits, trait impls, standalone fns, const/static, type aliases, `///` doc comments, `#[derive()]` attributes, visibility (`pub`/`pub(crate)`/`pub(super)`), async/unsafe |
| Go | `GoParser` | tree-sitter AST | structs, interfaces, functions, receiver methods (pointer/value), const blocks, var declarations, type aliases, `//` doc comments, exported detection (uppercase) |

Add new languages: implement `LanguageParser` protocol, register in `get_parser()`, add to `EXTENSION_MAP`.

## Embedder

`LlamaEmbedder` tries three HTTP endpoint formats in order:
1. `/v1/embeddings` (OpenAI-compatible)
2. `/api/embeddings` (Ollama)
3. `/embedding` (llama.cpp)

Batches texts, 60s timeout, returns zero vectors on failure.

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
- **Tools:** `search` (semantic), `lookup` (exact qualified_name), `get_context` (full parent overview), `list_sources`
- **Resources:** `glyph://sources`, `glyph://sources/{name}/{version}/index`, `glyph://sources/{name}/{version}/classes/{class}`
- **Implementation:** `server.py` uses FastMCP with lifespan for store/embedder init. Formatting helpers produce markdown output matching the tiered export style.
- **Store methods added for MCP:** `get_by_qualified_name()`, `get_by_parent()`, `get_sources_with_counts()`. `search()` extended with `source_version` and `parent_name` filters.
- **Tests:** `tests/test_server.py` — mocked store, covers all 4 tools + formatters

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

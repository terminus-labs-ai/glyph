# Glyph

Convert API documentation and source code into structured RAG knowledge bases. Ingest from multiple source types, chunk intelligently by API element, generate embeddings, store in PostgreSQL with pgvector, and export as tiered markdown files for LLM context injection.

## Features

- **Multiple ingestors** -- Godot XML class reference, Unreal Engine C++ headers (via unreal-doc), local docs (Markdown/RST), HTML web scraping, source code analysis
- **Structured chunking** -- one chunk per method, property, signal, class, etc. (not arbitrary text splits)
- **Source code parsing** -- tree-sitter (Python, TypeScript/TSX, Rust, Go) and regex (GDScript) extract classes, functions, and docstrings
- **Tiered markdown export** -- index (minimal tokens) / summary (moderate) / detail (full) for context-efficient LLM use
- **Incremental updates** -- content hashing skips unchanged documents on re-ingest
- **Configurable embeddings** -- pluggable embedding backend (ships with llama-server support)
- **Pluggable architecture** -- add new languages, doc sources, or export formats by implementing a Protocol

## Install

```bash
git clone https://github.com/terminus-labs-ai/glyph.git
cd glyph
uv venv .venv && source .venv/bin/activate
uv pip install -e .
```

**Requirements:** Python 3.11+, PostgreSQL with [pgvector](https://github.com/pgvector/pgvector) extension

## Quick Start

```bash
# 1. Configure
cp glyph.example.yaml glyph.yaml
# Edit glyph.yaml with your database URL, embedding server, and sources

# 2. Initialize database
uv run glyph init-db

# 3. Ingest documentation
uv run glyph ingest

# 4. Export as tiered markdown
uv run glyph export -s godot -V 4.4

# 5. Check stats
uv run glyph stats
```

## CLI Reference

All commands accept `-c <path>` to specify a config file (default: `glyph.yaml`) and `-v` for verbose logging.

| Command | Description |
|---------|-------------|
| `glyph init-db` | Create database tables and pgvector extension |
| `glyph ingest` | Ingest docs from all configured sources |
| `glyph ingest -s godot` | Ingest only the named source |
| `glyph ingest --skip-embeddings` | Ingest and chunk without generating embeddings |
| `glyph export -s NAME -V VERSION` | Export chunks as tiered markdown |
| `glyph stats` | Show source/document/chunk counts |
| `glyph serve` | Start MCP server (default: stdio transport) |
| `glyph serve -t sse -H 0.0.0.0 -p 8420` | Start MCP server with SSE transport |

## Configuration

Full example in [`glyph.example.yaml`](glyph.example.yaml). Key sections:

### Database

```yaml
database:
  url: "postgresql://user:password@host:5432/dbname"
```

### Embedder

```yaml
embedder:
  type: "llama"                      # Embedding backend
  url: "http://localhost:11434"      # llama-server / Ollama URL
  model: "nomic-embed-text"          # Model name
  dimensions: 512                    # Vector dimensions
  batch_size: 5                      # Texts per request
```

The llama embedder auto-detects OpenAI-compatible, Ollama, and llama.cpp API formats.

### Sources

Each source has a `name`, `version`, and one or more `ingestors`:

```yaml
sources:
  - name: godot
    version: "4.4"
    ingestors:
      # Godot XML class reference (from engine source repo)
      - type: godot_xml
        path: "/path/to/godot/doc/classes"

      # HTML web scraping
      - type: html
        base_url: "https://docs.godotengine.org/en/stable/"
        max_pages: 500
        delay: 0.2
        include_patterns: ["/classes/", "/tutorials/"]
        exclude_patterns: ["/contributing/"]

      # Source code analysis
      - type: source_code
        path: "/path/to/project/src"
        extensions: [".py", ".gd", ".ts", ".tsx", ".rs", ".go"]
        include_bodies: false        # true = full function bodies
        exclude_dirs: ["tests"]
```

### Output

```yaml
output:
  directory: "/path/to/knowledge"
  formats: ["markdown"]
```

## Ingestor Types

### `godot_xml`

Parses Godot engine's XML class reference files (`doc/classes/*.xml`). Extracts class hierarchy, methods with full signatures, properties, signals, constants, and enums.

| Setting | Required | Default | Description |
|---------|----------|---------|-------------|
| `path` | yes | -- | Path to directory containing XML class files |
| `include_patterns` | no | all | Regex patterns to filter file paths |
| `exclude_patterns` | no | none | Regex patterns to exclude file paths |

### `docs`

Walks a directory tree for documentation files (`.md`, `.rst`, `.txt`). Converts RST headings and inline markup to markdown for heading-based chunking. Produces TUTORIAL documents.

| Setting | Required | Default | Description |
|---------|----------|---------|-------------|
| `path` | yes | -- | Root directory to scan |
| `extensions` | no | `.md`, `.rst`, `.txt` | File extensions to include |
| `include_patterns` | no | all | Regex patterns to filter file paths |
| `exclude_patterns` | no | none | Regex patterns to exclude file paths |
| `exclude_dirs` | no | see below | Directory names to skip |

Default excluded directories: `.git`, `__pycache__`, `.venv`, `venv`, `node_modules`, `_build`, `_static`, `_templates`

**RST conversion:** Headings (underlined with `=`, `-`, `~`, `^`) are converted to markdown `#` headings. Admonitions (`.. note::`, `.. warning::`) are converted to bold labels. Image/toctree directives are stripped. Inline refs (`:ref:`, `:doc:`, `:class:`) are converted to plain text.

```yaml
sources:
  - name: godot-docs
    version: "4.7"
    ingestors:
      - type: docs
        path: "/path/to/godot-docs/tutorials"
        extensions: [".rst"]
        exclude_dirs: ["img"]
```

### `html`

Async web crawler with filtering. Follows same-domain links, respects rate limits.

| Setting | Required | Default | Description |
|---------|----------|---------|-------------|
| `base_url` | yes | -- | Starting URL |
| `max_pages` | no | 500 | Crawl limit |
| `delay` | no | 0.2 | Seconds between requests |
| `include_patterns` | no | all | Regex URL patterns to include |
| `exclude_patterns` | no | none | Regex URL patterns to exclude |
| `max_concurrent` | no | 10 | Parallel connections |


### `source_code`

Walks a directory tree and parses source files into structured chunks.

| Setting | Required | Default | Description |
|---------|----------|---------|-------------|
| `path` | yes | -- | Root directory to scan |
| `extensions` | no | `.py`, `.gd`, `.ts`, `.tsx`, `.rs`, `.go` | File extensions to include |
| `include_bodies` | no | `false` | Include full function bodies in chunks |
| `exclude_dirs` | no | see below | Additional directories to skip |
| `exclude_patterns` | no | none | Path substrings to exclude |

Default excluded directories: `__pycache__`, `.git`, `.venv`, `venv`, `node_modules`, `.mypy_cache`, `.pytest_cache`, `build`, `dist`, `.eggs`

**Supported languages:**

| Language | Parser | Extracts |
|----------|--------|----------|
| Python | tree-sitter AST | classes, methods, functions, decorators, docstrings |
| GDScript | regex | class_name/extends, funcs, signals, vars, enums, constants, `##` doc comments |
| TypeScript/TSX | tree-sitter AST | classes, interfaces, enums, type aliases, methods, arrow functions, JSDoc tags, export/abstract detection |
| Rust | tree-sitter AST | structs, enums, impl methods, traits, trait impls, const/static, type aliases, `///` doc comments, `#[derive()]`, visibility |
| Go | tree-sitter AST | structs, interfaces, receiver methods, functions, const/var blocks, type aliases, `//` doc comments, exported detection |

**Language-specific configuration examples:**

```yaml
sources:
  # TypeScript project
  - name: my-ts-app
    version: "1.0"
    ingestors:
      - type: source_code
        path: "/path/to/ts-project/src"
        extensions: [".ts", ".tsx"]
        include_bodies: false
        exclude_dirs: ["node_modules", "dist", "__tests__"]

  # Rust crate
  - name: my-rust-crate
    version: "0.5.0"
    ingestors:
      - type: source_code
        path: "/path/to/rust-project/src"
        extensions: [".rs"]
        include_bodies: true
        exclude_dirs: ["target"]

  # Go module
  - name: my-go-module
    version: "1.2.0"
    ingestors:
      - type: source_code
        path: "/path/to/go-project"
        extensions: [".go"]
        include_bodies: false
        exclude_dirs: ["vendor", "testdata"]
        exclude_patterns: ["_test.go"]
```

## Tiered Output

The markdown exporter produces three tiers of detail, so an LLM can load only what it needs:

```
knowledge/godot/4.4/
├── index.md              # Tier 1: class names + one-line summaries
├── classes/
│   ├── _index.md         # Tier 2: class summaries with member lists
│   ├── Node2D.md         # Tier 3: full detail (methods, properties, signals...)
│   └── Sprite2D.md
└── tutorials/
    ├── _index.md         # Tier 2: tutorial summaries
    └── getting_started.md
```

- **Tier 1** -- load the index to find what's relevant (minimal tokens)
- **Tier 2** -- load a class summary for quick reference (moderate tokens)
- **Tier 3** -- load full detail when working with specific APIs (full tokens)

## Database Schema

Three tables, all with UUID primary keys:

**`sources`** -- registered documentation sources (unique on `name` + `version`)

**`documents`** -- individual pages/files, tracked by content hash for incremental updates

**`chunks`** -- the actual RAG units, with:
- `qualified_name` -- e.g. `Node2D.get_position`
- `chunk_type` -- `class_overview`, `method`, `property`, `signal`, `constant`, `enum`, `tutorial_section`
- `summary` -- one-liner for tier 1/2 output
- `content` -- full formatted content for tier 3
- `metadata` -- JSONB with type-specific fields (params, return_type, default values, etc.)
- `embedding` -- pgvector column for similarity search
- `source_name` / `source_version` -- denormalized for direct queries without joins

## Extending Glyph

All extension points use `typing.Protocol` -- implement the interface and plug in.

### Add a new language parser

Create `glyph/chunkers/_parsers/your_language_parser.py`:

```python
from glyph.chunkers._parsers import Symbol, LanguageParser
from glyph.domain.models import ChunkType

class YourLanguageParser:
    def parse(self, source: str, *, include_bodies: bool = False) -> list[Symbol]:
        # Extract classes, functions, etc.
        # Return a list of Symbol dataclasses
        ...
```

Register it in `glyph/chunkers/_parsers/__init__.py`:

```python
def get_parser(language: str) -> LanguageParser | None:
    ...
    elif language == "your_language":
        from glyph.chunkers._parsers.your_language_parser import YourLanguageParser
        return YourLanguageParser()
```

Add the extension mapping in `glyph/chunkers/source_code_chunker.py`:

```python
EXTENSION_MAP = {
    ...,
    ".ext": "your_language",
}
```

See the existing parsers (`python_parser.py`, `typescript_parser.py`, `rust_parser.py`, `go_parser.py`) for reference implementations using tree-sitter.

### Add a new ingestor

Implement the `Ingestor` protocol:

```python
class MyIngestor:
    async def ingest(self) -> list[Document]:
        ...
```

Register it in `__main__.py:_build_ingestor()`.

### Add a new embedder

Implement the `Embedder` protocol:

```python
class MyEmbedder:
    @property
    def dimensions(self) -> int:
        return 1536

    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...
```

## MCP Integration

Glyph includes an MCP (Model Context Protocol) server, enabling any MCP-compatible client to query the knowledge base at runtime.

### Tools

| Tool | Description |
|------|-------------|
| `search` | Semantic similarity search with optional source/version/type/parent filters |
| `lookup` | Exact match by `qualified_name` (e.g., `Node2D.get_position`) |
| `get_context` | Full class/module overview — all members grouped by type |
| `list_sources` | List all indexed sources with document/chunk counts |

### Resources

| URI | Description |
|-----|-------------|
| `glyph://sources` | JSON list of all source name/version pairs |
| `glyph://sources/{name}/{version}/index` | Tier 1 index markdown for a source |
| `glyph://sources/{name}/{version}/classes/{class}` | Tier 3 full class detail |

### Client Configuration

**Claude Code** (`~/.claude/claude_code_config.json` or project `.claude/settings.json`):

```json
{
  "mcpServers": {
    "glyph": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/glyph", "glyph", "serve"]
    }
  }
}
```

**Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "glyph": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/glyph", "glyph", "serve"]
    }
  }
}
```

**Remote/SSE client** (first start the server with `glyph serve -t sse -H 0.0.0.0 -p 8420`):

```json
{
  "mcpServers": {
    "glyph": {
      "url": "http://localhost:8420/sse"
    }
  }
}
```

### Testing with MCP Inspector

```bash
npx @modelcontextprotocol/inspector uv run glyph serve
```

## Architecture

```
glyph/
├── domain/models.py          # Source, Document, Chunk dataclasses + enums
├── config.py                 # YAML config loader
├── server.py                 # MCP server (FastMCP, 4 tools + resources)
├── ingestors/
│   ├── base.py               # Ingestor Protocol
│   ├── docs.py               # Local .md/.rst/.txt file walker
│   ├── godot_xml.py          # Godot XML class reference parser
│   ├── html.py               # Async HTML crawler
│   ├── source_code.py        # Source tree walker
│   └── unreal_doc.py         # unreal-doc JSON parser
├── chunkers/
│   ├── base.py               # Chunker Protocol
│   ├── api_chunker.py        # Structured API chunks (from XML)
│   ├── text_chunker.py       # Heading-based text splitting
│   ├── source_code_chunker.py # Code -> chunks via language parsers
│   ├── unreal_doc_chunker.py # UE class/method/property chunks
│   └── _parsers/             # Language-specific parsers
│       ├── python_parser.py  # tree-sitter AST
│       ├── gdscript_parser.py # regex-based
│       ├── typescript_parser.py # tree-sitter AST (TS + TSX)
│       ├── rust_parser.py    # tree-sitter AST
│       └── go_parser.py      # tree-sitter AST
├── embedders/
│   ├── base.py               # Embedder Protocol
│   └── llama.py              # llama-server / Ollama client
├── store/
│   └── postgres.py           # PostgreSQL + pgvector
├── exporters/
│   └── markdown.py           # Tiered markdown export
└── __main__.py               # CLI (click)
```

## License

MIT

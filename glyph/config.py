from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Well-known config locations, checked in order
GLOBAL_CONFIG_PATHS = [
    Path.home() / ".config" / "glyph" / "config.yaml",
    Path.home() / ".config" / "glyph" / "config.yml",
    Path.home() / ".glyph" / "config.yaml",
]

REPO_CONFIG_NAMES = [".glyph.yaml", ".glyph.yml"]

# File extensions mapped to language names for auto-discovery
LANGUAGE_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".gd": "gdscript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".rs": "rust",
    ".go": "go",
    ".cpp": "cpp",
    ".h": "cpp",
    ".hpp": "cpp",
    ".hlsl": "hlsl",
    ".hlsli": "hlsl",
    ".fx": "hlsl",
    ".fxh": "hlsl",
    ".usf": "usf",
    ".ush": "usf",
    ".glsl": "glsl",
    ".shader": "glsl",
    ".gdshader": "glsl",
    ".vert": "glsl",
    ".frag": "glsl",
    ".comp": "glsl",
    ".geom": "glsl",
    ".tesc": "glsl",
    ".tese": "glsl",
    ".vs": "glsl",
    ".fs": "glsl",
}

DEFAULT_EXCLUDE_DIRS = [
    ".git", ".github", ".vscode", ".idea",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".venv", "venv", ".tox", ".nox", ".eggs",
    "node_modules", ".next",
    "build", "dist", "target",
    ".godot", ".import",
]

DEFAULT_EXCLUDE_PATTERNS = [
    "_test.", ".test.", ".spec.", "_mock.", "conftest.py",
]


@dataclass
class DatabaseConfig:
    url: str


@dataclass
class EmbedderConfig:
    type: str = "llama"
    url: str = "http://localhost:11434"
    model: str = "nomic-embed-text"
    dimensions: int = 512
    batch_size: int = 5


@dataclass
class RerankerConfig:
    type: str = "llama"
    url: str = "http://localhost:11434"
    model: str = "qwen3-reranker"
    batch_size: int = 32
    timeout: int = 30


@dataclass
class IngestorConfig:
    type: str
    # Flexible per-ingestor settings
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> str | None:
        return self.settings.get("path")

    @property
    def base_url(self) -> str | None:
        return self.settings.get("base_url")


@dataclass
class SourceConfig:
    name: str
    version: str
    ingestors: list[IngestorConfig]


@dataclass
class DefaultsConfig:
    include_bodies: bool = False
    exclude_dirs: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE_DIRS))
    exclude_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE_PATTERNS))


@dataclass
class OutputConfig:
    directory: str = "./output"
    formats: list[str] = field(default_factory=lambda: ["markdown"])


@dataclass
class Config:
    database: DatabaseConfig
    embedder: EmbedderConfig
    sources: list[SourceConfig]
    output: OutputConfig
    reranker: RerankerConfig | None = None
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)


def load_config(path: str | Path) -> Config:
    """Load a full glyph.yaml config file (original behavior, backwards-compatible)."""
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)

    return _parse_full_config(raw)


def load_global_config(path: str | Path | None = None) -> Config | None:
    """Load the global config from a specific path or well-known locations.

    Returns None if no global config is found.
    The global config provides database, embedder, output, and defaults —
    but typically has no sources (those come from per-repo configs).
    """
    if path:
        p = Path(path)
        if p.exists():
            with p.open() as f:
                raw = yaml.safe_load(f)
            return _parse_full_config(raw)
        return None

    for candidate in GLOBAL_CONFIG_PATHS:
        if candidate.exists():
            logger.debug(f"Found global config at {candidate}")
            with candidate.open() as f:
                raw = yaml.safe_load(f)
            return _parse_full_config(raw)

    return None


def load_repo_config(repo_path: str | Path) -> SourceConfig | None:
    """Load a per-repo .glyph.yaml from a repository root.

    Returns None if no .glyph.yaml exists. The per-repo config only
    defines source identity (name, version) and ingestor preferences.
    """
    repo_path = Path(repo_path)
    for name in REPO_CONFIG_NAMES:
        config_file = repo_path / name
        if config_file.exists():
            logger.debug(f"Found repo config at {config_file}")
            with config_file.open() as f:
                raw = yaml.safe_load(f)
            return _parse_repo_config(raw, repo_path)

    return None


def discover_source(repo_path: str | Path, defaults: DefaultsConfig | None = None) -> SourceConfig:
    """Auto-discover source config from a repository directory.

    Scans for known file extensions, derives name from directory name,
    and resolves version from git.
    """
    repo_path = Path(repo_path).resolve()
    if defaults is None:
        defaults = DefaultsConfig()

    name = repo_path.name
    version = _resolve_version_auto(repo_path)

    # Scan for file extensions (only top 3 levels to keep it fast)
    found_extensions: set[str] = set()
    for depth_pattern in ["*", "*/*", "*/*/*"]:
        for f in repo_path.glob(depth_pattern):
            if f.is_file() and f.suffix in LANGUAGE_EXTENSIONS:
                # Skip excluded dirs
                parts = f.relative_to(repo_path).parts
                if not any(d in defaults.exclude_dirs for d in parts[:-1]):
                    found_extensions.add(f.suffix)

    ingestors: list[IngestorConfig] = []

    if found_extensions:
        extensions = sorted(found_extensions)
        ingestors.append(IngestorConfig(
            type="source_code",
            settings={
                "path": str(repo_path),
                "extensions": extensions,
                "include_bodies": defaults.include_bodies,
                "exclude_dirs": defaults.exclude_dirs,
                "exclude_patterns": defaults.exclude_patterns,
            },
        ))

    # Check for docs directory
    docs_dir = repo_path / "docs"
    if docs_dir.is_dir():
        doc_extensions = []
        for ext in [".md", ".rst", ".txt"]:
            if any(docs_dir.rglob(f"*{ext}")):
                doc_extensions.append(ext)
        if doc_extensions:
            ingestors.append(IngestorConfig(
                type="docs",
                settings={
                    "path": str(docs_dir),
                    "extensions": doc_extensions,
                    "exclude_dirs": defaults.exclude_dirs,
                },
            ))

    if not ingestors:
        logger.warning(f"No recognized source files found in {repo_path}")
        # Return a source_code ingestor with common defaults anyway
        ingestors.append(IngestorConfig(
            type="source_code",
            settings={
                "path": str(repo_path),
                "extensions": [".py"],
                "exclude_dirs": defaults.exclude_dirs,
                "exclude_patterns": defaults.exclude_patterns,
            },
        ))

    return SourceConfig(name=name, version=version, ingestors=ingestors)


def resolve_config_for_repo(
    repo_path: str | Path,
    global_config: Config | None = None,
    name_override: str | None = None,
    version_override: str | None = None,
) -> tuple[Config, SourceConfig]:
    """Resolve the full config + source config for a given repo path.

    Priority:
    1. Per-repo .glyph.yaml (if exists)
    2. Auto-discovery fallback
    3. Merged with global config for database/embedder/output

    Returns (global_config, source_config) tuple.
    Raises ValueError if no global config available (need DB/embedder settings).
    """
    repo_path = Path(repo_path).resolve()

    # Load global config if not provided
    if global_config is None:
        global_config = load_global_config()
        if global_config is None:
            raise ValueError(
                "No global config found. Create ~/.config/glyph/config.yaml "
                "with database and embedder settings, or pass a config file with -c."
            )

    # Try per-repo config first, then auto-discover
    source_cfg = load_repo_config(repo_path)
    if source_cfg is None:
        source_cfg = discover_source(repo_path, global_config.defaults)
        logger.info(f"Auto-discovered source: {source_cfg.name} v{source_cfg.version}")
    else:
        logger.info(f"Loaded repo config: {source_cfg.name} v{source_cfg.version}")

    # Apply overrides
    if name_override:
        source_cfg.name = name_override
    if version_override:
        source_cfg.version = version_override

    return global_config, source_cfg


# --- Internal helpers ---


def _parse_full_config(raw: dict) -> Config:
    """Parse a full glyph.yaml dict into a Config object."""
    database = DatabaseConfig(url=raw["database"]["url"])

    emb = raw.get("embedder", {})
    embedder = EmbedderConfig(
        type=emb.get("type", "llama"),
        url=emb.get("url", "http://localhost:11434"),
        model=emb.get("model", "nomic-embed-text"),
        dimensions=emb.get("dimensions", 512),
        batch_size=emb.get("batch_size", 5),
    )

    rer = raw.get("reranker")
    reranker = RerankerConfig(
        type=rer.get("type", "llama"),
        url=rer.get("url", "http://localhost:11434"),
        model=rer.get("model", "qwen3-reranker"),
        batch_size=rer.get("batch_size", 32),
        timeout=rer.get("timeout", 30),
    ) if rer else None

    sources = []
    for src in raw.get("sources", []):
        ingestors = []
        for ing in src.get("ingestors", []):
            ing_type = ing["type"]
            settings = {k: v for k, v in ing.items() if k != "type"}
            ingestors.append(IngestorConfig(type=ing_type, settings=settings))
        sources.append(SourceConfig(
            name=src["name"],
            version=src["version"],
            ingestors=ingestors,
        ))

    out = raw.get("output", {})
    output = OutputConfig(
        directory=out.get("directory", "./output"),
        formats=out.get("formats", ["markdown"]),
    )

    defs = raw.get("defaults", {})
    defaults = DefaultsConfig(
        include_bodies=defs.get("include_bodies", False),
        exclude_dirs=defs.get("exclude_dirs", list(DEFAULT_EXCLUDE_DIRS)),
        exclude_patterns=defs.get("exclude_patterns", list(DEFAULT_EXCLUDE_PATTERNS)),
    )

    return Config(
        database=database,
        embedder=embedder,
        reranker=reranker,
        sources=sources,
        output=output,
        defaults=defaults,
    )


def _parse_repo_config(raw: dict, repo_path: Path) -> SourceConfig:
    """Parse a per-repo .glyph.yaml dict into a SourceConfig."""
    name = raw.get("name", repo_path.name)

    version_raw = raw.get("version", "auto")
    if version_raw == "auto":
        version = _resolve_version_auto(repo_path)
    else:
        version = str(version_raw)

    ingestors = []
    for ing in raw.get("ingestors", []):
        ing_type = ing["type"]
        settings = {k: v for k, v in ing.items() if k != "type"}

        # Resolve relative paths against repo root
        if "path" in settings:
            p = Path(settings["path"])
            if not p.is_absolute():
                settings["path"] = str(repo_path / p)

        ingestors.append(IngestorConfig(type=ing_type, settings=settings))

    return SourceConfig(name=name, version=version, ingestors=ingestors)


def _resolve_version_auto(repo_path: Path) -> str:
    """Resolve version from git: tag > branch > 'latest'."""
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return "latest"

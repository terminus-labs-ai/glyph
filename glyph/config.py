from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


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
class OutputConfig:
    directory: str = "./output"
    formats: list[str] = field(default_factory=lambda: ["markdown"])


@dataclass
class Config:
    database: DatabaseConfig
    embedder: EmbedderConfig
    sources: list[SourceConfig]
    output: OutputConfig


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)

    database = DatabaseConfig(url=raw["database"]["url"])

    emb = raw.get("embedder", {})
    embedder = EmbedderConfig(
        type=emb.get("type", "llama"),
        url=emb.get("url", "http://localhost:11434"),
        model=emb.get("model", "nomic-embed-text"),
        dimensions=emb.get("dimensions", 512),
        batch_size=emb.get("batch_size", 5),
    )

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

    return Config(
        database=database,
        embedder=embedder,
        sources=sources,
        output=output,
    )

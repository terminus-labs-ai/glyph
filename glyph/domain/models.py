from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


class ChunkType(str, enum.Enum):
    CLASS_OVERVIEW = "class_overview"
    METHOD = "method"
    PROPERTY = "property"
    SIGNAL = "signal"
    CONSTANT = "constant"
    ENUM = "enum"
    ANNOTATION = "annotation"
    TUTORIAL_SECTION = "tutorial_section"
    CODE_EXAMPLE = "code_example"
    SHADER_ENTRY_POINT = "shader_entry_point"
    SHADER_RESOURCE = "shader_resource"
    SHADER_UNIFORM_BLOCK = "shader_uniform_block"


class DocType(str, enum.Enum):
    CLASS_REF = "class_ref"
    TUTORIAL = "tutorial"
    GUIDE = "guide"
    API_OVERVIEW = "api_overview"


@dataclass
class Source:
    name: str
    version: str
    source_type: str  # "xml", "html", "rst"
    origin: str  # URL or filesystem path
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    config: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Document:
    source_id: uuid.UUID
    path: str  # URL or file path
    title: str
    doc_type: DocType
    raw_content: str
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    content_hash: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Chunk:
    document_id: uuid.UUID
    source_name: str
    source_version: str
    chunk_type: ChunkType
    qualified_name: str  # e.g. "Node2D.get_position"
    parent_name: str  # e.g. "Node2D"
    heading: str
    content: str
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    token_count: int = 0
    chunk_index: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

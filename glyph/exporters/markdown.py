from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MarkdownExporter:
    """Export chunks as tiered markdown files.

    Tier 1 (index.md): class names + one-liners
    Tier 2 (classes/_index.md): class summaries with member lists
    Tier 3 (classes/ClassName.md): full detail per class
    """

    def __init__(self, output_dir: str):
        self._output_dir = Path(output_dir)

    def export(
        self,
        chunks: list[dict[str, Any]],
        source_name: str,
        source_version: str,
    ) -> Path:
        base = self._output_dir / source_name / source_version
        classes_dir = base / "classes"
        tutorials_dir = base / "tutorials"
        classes_dir.mkdir(parents=True, exist_ok=True)
        tutorials_dir.mkdir(parents=True, exist_ok=True)

        # Group chunks by parent
        by_parent: dict[str, list[dict]] = defaultdict(list)
        for chunk in chunks:
            by_parent[chunk["parent_name"]].append(chunk)

        # Separate classes from tutorials
        class_parents = {}
        tutorial_parents = {}

        for parent, parent_chunks in by_parent.items():
            chunk_types = {c["chunk_type"] for c in parent_chunks}
            if "class_overview" in chunk_types or "method" in chunk_types or "property" in chunk_types:
                class_parents[parent] = parent_chunks
            else:
                tutorial_parents[parent] = parent_chunks

        # Tier 1: index.md
        self._write_index(base / "index.md", class_parents, tutorial_parents)

        # Tier 2 + 3: classes
        self._write_class_index(classes_dir / "_index.md", class_parents)
        for class_name, class_chunks in sorted(class_parents.items()):
            self._write_class_detail(classes_dir / f"{class_name}.md", class_name, class_chunks)

        # Tier 2 + 3: tutorials
        if tutorial_parents:
            self._write_tutorial_index(tutorials_dir / "_index.md", tutorial_parents)
            for title, tut_chunks in sorted(tutorial_parents.items()):
                safe_name = _safe_filename(title)
                self._write_tutorial_detail(tutorials_dir / f"{safe_name}.md", title, tut_chunks)

        logger.info(f"Exported to {base}")
        return base

    def _write_index(
        self,
        path: Path,
        class_parents: dict[str, list[dict]],
        tutorial_parents: dict[str, list[dict]],
    ) -> None:
        lines = ["# API Reference Index\n"]

        if class_parents:
            lines.append("## Classes\n")
            for name in sorted(class_parents):
                overview = _find_overview(class_parents[name])
                summary = overview.get("summary", "") if overview else ""
                lines.append(f"- **{name}** — {summary}")

        if tutorial_parents:
            lines.append("\n## Tutorials\n")
            for title in sorted(tutorial_parents):
                first = tutorial_parents[title][0]
                summary = first.get("summary", "")
                lines.append(f"- **{title}** — {summary}")

        path.write_text("\n".join(lines) + "\n")

    def _write_class_index(self, path: Path, class_parents: dict[str, list[dict]]) -> None:
        lines = ["# Class Summaries\n"]

        for name in sorted(class_parents):
            chunks = class_parents[name]
            overview = _find_overview(chunks)

            lines.append(f"## {name}\n")
            if overview:
                meta = _parse_metadata(overview.get("metadata", "{}"))
                if meta.get("inherits"):
                    lines.append(f"**Inherits:** {meta['inherits']}\n")
                lines.append(f"{overview.get('summary', '')}\n")

            # List members
            methods = [c for c in chunks if c["chunk_type"] == "method"]
            props = [c for c in chunks if c["chunk_type"] == "property"]
            signals = [c for c in chunks if c["chunk_type"] == "signal"]

            if props:
                lines.append(f"**Properties:** {', '.join(c['heading'] for c in props)}")
            if methods:
                lines.append(f"**Methods:** {', '.join(c['heading'] for c in methods)}")
            if signals:
                lines.append(f"**Signals:** {', '.join(c['heading'] for c in signals)}")
            lines.append("")

        path.write_text("\n".join(lines) + "\n")

    def _write_class_detail(self, path: Path, class_name: str, chunks: list[dict]) -> None:
        lines = [f"# {class_name}\n"]

        overview = _find_overview(chunks)
        if overview:
            lines.append(overview.get("content", ""))
            lines.append("")

        # Group by type
        by_type: dict[str, list[dict]] = defaultdict(list)
        for c in chunks:
            if c["chunk_type"] != "class_overview":
                by_type[c["chunk_type"]].append(c)

        type_headings = {
            "property": "Properties",
            "method": "Methods",
            "signal": "Signals",
            "constant": "Constants",
            "enum": "Enumerations",
            "annotation": "Annotations",
        }

        for chunk_type, heading in type_headings.items():
            members = by_type.get(chunk_type, [])
            if not members:
                continue
            lines.append(f"\n## {heading}\n")
            for m in members:
                lines.append(f"### {m['heading']}\n")
                lines.append(m.get("content", ""))
                lines.append("")

        path.write_text("\n".join(lines) + "\n")

    def _write_tutorial_index(self, path: Path, tutorial_parents: dict[str, list[dict]]) -> None:
        lines = ["# Tutorials\n"]
        for title in sorted(tutorial_parents):
            first = tutorial_parents[title][0]
            lines.append(f"- **{title}** — {first.get('summary', '')}")
        path.write_text("\n".join(lines) + "\n")

    def _write_tutorial_detail(self, path: Path, title: str, chunks: list[dict]) -> None:
        lines = [f"# {title}\n"]
        for c in sorted(chunks, key=lambda x: x.get("chunk_index", 0)):
            if c["heading"] and c["heading"] != title:
                lines.append(f"\n## {c['heading']}\n")
            lines.append(c.get("content", ""))
            lines.append("")
        path.write_text("\n".join(lines) + "\n")


def _find_overview(chunks: list[dict]) -> dict | None:
    for c in chunks:
        if c["chunk_type"] == "class_overview":
            return c
    return None


def _parse_metadata(metadata: Any) -> dict:
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            return json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)[:100]

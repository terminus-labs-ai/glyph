from __future__ import annotations

import hashlib
import logging
import re
import uuid
from pathlib import Path
from xml.etree import ElementTree as ET

from glyph.domain.models import DocType, Document

logger = logging.getLogger(__name__)


class GodotXMLIngestor:
    """Parses Godot's XML class reference files from doc/classes/."""

    def __init__(
        self,
        path: str,
        source_id: uuid.UUID,
        *,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ):
        self._path = Path(path)
        self._source_id = source_id
        self._include = [re.compile(p) for p in (include_patterns or [])]
        self._exclude = [re.compile(p) for p in (exclude_patterns or [])]

    def _should_include(self, path: Path) -> bool:
        path_str = str(path)
        if self._exclude:
            for pat in self._exclude:
                if pat.search(path_str):
                    return False
        if self._include:
            return any(pat.search(path_str) for pat in self._include)
        return True

    async def ingest(self) -> list[Document]:
        if not self._path.is_dir():
            raise FileNotFoundError(f"Godot XML class directory not found: {self._path}")

        xml_files = sorted(f for f in self._path.rglob("*.xml") if self._should_include(f))
        logger.info(f"Found {len(xml_files)} XML class files in {self._path}")

        documents = []
        for xml_file in xml_files:
            doc = self._parse_class_file(xml_file)
            if doc:
                documents.append(doc)

        logger.info(f"Parsed {len(documents)} class documents")
        return documents

    def _parse_class_file(self, xml_file: Path) -> Document | None:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            if root.tag != "class":
                return None

            class_name = root.get("name", xml_file.stem)
            inherits = root.get("inherits", "")

            # Build raw content from all text in the XML
            parts = [f"Class: {class_name}"]
            if inherits:
                parts.append(f"Inherits: {inherits}")

            brief = root.findtext("brief_description", "").strip()
            if brief:
                parts.append(f"Brief: {brief}")

            desc = root.findtext("description", "").strip()
            if desc:
                parts.append(f"Description: {desc}")

            # Methods
            for method in root.iter("method"):
                name = method.get("name", "")
                method_desc = method.findtext("description", "").strip()
                ret = method.find("return")
                ret_type = ret.get("type", "void") if ret is not None else "void"
                params = []
                for param in method.findall("param"):
                    p_name = param.get("name", "")
                    p_type = param.get("type", "")
                    p_default = param.get("default", "")
                    param_str = f"{p_name}: {p_type}"
                    if p_default:
                        param_str += f" = {p_default}"
                    params.append(param_str)
                sig = f"{name}({', '.join(params)}) -> {ret_type}"
                parts.append(f"Method: {sig}")
                if method_desc:
                    parts.append(f"  {method_desc}")

            # Members / properties
            for member in root.iter("member"):
                name = member.get("name", "")
                m_type = member.get("type", "")
                default = member.get("default", "")
                member_desc = (member.text or "").strip()
                parts.append(f"Property: {name}: {m_type} = {default}")
                if member_desc:
                    parts.append(f"  {member_desc}")

            # Signals
            for signal in root.iter("signal"):
                name = signal.get("name", "")
                sig_desc = signal.findtext("description", "").strip()
                parts.append(f"Signal: {name}")
                if sig_desc:
                    parts.append(f"  {sig_desc}")

            # Constants
            for constant in root.iter("constant"):
                name = constant.get("name", "")
                value = constant.get("value", "")
                const_desc = (constant.text or "").strip()
                parts.append(f"Constant: {name} = {value}")
                if const_desc:
                    parts.append(f"  {const_desc}")

            raw_content = "\n".join(parts)
            content_hash = hashlib.md5(raw_content.encode()).hexdigest()

            return Document(
                source_id=self._source_id,
                path=str(xml_file),
                title=class_name,
                doc_type=DocType.CLASS_REF,
                raw_content=raw_content,
                content_hash=content_hash,
            )

        except ET.ParseError as e:
            logger.error(f"Failed to parse {xml_file}: {e}")
            return None

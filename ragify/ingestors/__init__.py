from .base import Ingestor
from .godot_xml import GodotXMLIngestor
from .html import HTMLIngestor
from .source_code import SourceCodeIngestor

__all__ = ["Ingestor", "GodotXMLIngestor", "HTMLIngestor", "SourceCodeIngestor"]

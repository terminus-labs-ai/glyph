"""Sample Python file for parser tests."""
from __future__ import annotations


class BaseProcessor:
    """Base class for all processors."""

    pass


class DataProcessor(BaseProcessor):
    """Processes data records.

    Handles validation, transformation, and output
    for structured data pipelines.
    """

    default_limit: int = 100

    def __init__(self, name: str, limit: int = 100) -> None:
        """Initialize the processor with a name and limit."""
        self.name = name
        self.limit = limit

    def process(self, data: list[dict]) -> list[dict]:
        """Process a list of data records.

        Validates each record and applies transformation.
        """
        return [self._transform(r) for r in data[: self.limit]]

    def _transform(self, record: dict) -> dict:
        """Apply transformation to a single record."""
        return record

    @staticmethod
    def validate(record: dict) -> bool:
        """Return True if the record is valid."""
        return bool(record)

    @classmethod
    def from_config(cls, config: dict) -> "DataProcessor":
        """Create a DataProcessor from a config dict."""
        return cls(name=config["name"], limit=config.get("limit", 100))

    @property
    def display_name(self) -> str:
        """Human-readable processor name."""
        return self.name.replace("_", " ").title()


def run_pipeline(processors: list[DataProcessor], data: list[dict]) -> list[dict]:
    """Run data through a sequence of processors and return results."""
    result = data
    for p in processors:
        result = p.process(result)
    return result

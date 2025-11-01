from abc import ABC, abstractmethod
from typing import Any

class IIngestor(ABC):
    """Interface for all ingestion components."""

    @abstractmethod
    def ingest(self, source_path: str) -> Any:
        """Convert one document into structured JSON chunks and metadata."""
        pass

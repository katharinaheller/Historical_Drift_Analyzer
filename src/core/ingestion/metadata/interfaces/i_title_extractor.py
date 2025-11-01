from abc import ABC, abstractmethod
from typing import Any, Dict

class ITitleExtractor(ABC):
    """Interface for extracting the main document title."""

    @abstractmethod
    def extract(self, pdf_path: str, parsed_document: Dict[str, Any]) -> str | None:
        """Extract the document title."""
        raise NotImplementedError

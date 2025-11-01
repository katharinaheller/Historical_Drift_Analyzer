from abc import ABC, abstractmethod
from typing import Any, Dict

class IYearExtractor(ABC):
    """Interface for extracting publication year."""

    @abstractmethod
    def extract(self, pdf_path: str, parsed_document: Dict[str, Any]) -> str | None:
        """Extract the publication year as string (e.g. '2023')."""
        raise NotImplementedError

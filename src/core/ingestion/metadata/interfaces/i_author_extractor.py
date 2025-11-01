from abc import ABC, abstractmethod
from typing import Any, Dict, List

class IAuthorExtractor(ABC):
    """Interface for extracting document authors."""

    @abstractmethod
    def extract(self, pdf_path: str, parsed_document: Dict[str, Any]) -> List[str]:
        """Extract author names as a list of strings."""
        raise NotImplementedError

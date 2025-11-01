from abc import ABC, abstractmethod
from typing import Any, Dict

class IFileSizeExtractor(ABC):
    """Interface for extracting the file size of a PDF."""

    @abstractmethod
    def extract(self, pdf_path: str) -> int | None:
        """Return file size in bytes."""
        raise NotImplementedError

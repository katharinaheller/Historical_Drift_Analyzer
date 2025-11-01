from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any


class IPdfParser(ABC):
    """Interface for all local PDF parsers."""

    @abstractmethod
    def parse(self, pdf_path: str) -> Dict[str, Any]:
        """Parse a PDF file and return structured output with text and metadata."""
        raise NotImplementedError

from __future__ import annotations
from typing import Any, Dict, List
from abc import ABC, abstractmethod


class TocExtractorInterface(ABC):
    """Interface for table-of-contents extractors."""

    @abstractmethod
    def extract(self, pdf_path: str, parsed_document: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        """
        Extracts the table of contents (TOC) from a given PDF file.

        Returns:
            A list of dictionaries with keys:
                - level: int
                - title: str
                - page: int
        """
        pass

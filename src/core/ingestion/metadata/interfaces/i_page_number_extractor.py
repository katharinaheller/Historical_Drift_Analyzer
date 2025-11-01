from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional


class IPageNumberExtractor(ABC):
    """Interface for extracting page numbers from a document."""

    @abstractmethod
    def extract_page_number(self, pdf_path: str) -> Optional[int]:
        """Extract the page number from the given PDF path.
        
        :param pdf_path: Path to the PDF file
        :return: The page number (if available), otherwise None.
        """
        pass

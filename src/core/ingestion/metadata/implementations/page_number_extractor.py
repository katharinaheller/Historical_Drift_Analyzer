from __future__ import annotations
from pathlib import Path
from typing import Optional
import fitz  # PyMuPDF
from src.core.ingestion.metadata.interfaces.i_page_number_extractor import IPageNumberExtractor


class PageNumberExtractor(IPageNumberExtractor):
    """Extracts the page number from the first page of the PDF document."""
    
    def __init__(self, base_dir: Path | str | None = None):
        self.base_dir = Path(base_dir).resolve() if base_dir else None

    def extract_page_number(self, pdf_path: str) -> Optional[int]:
        """Extract the page number from the first page of the PDF.
        
        :param pdf_path: Path to the PDF file.
        :return: The page number (if available), otherwise None.
        """
        pdf_file = Path(pdf_path)

        try:
            with fitz.open(pdf_file) as doc:
                # Getting the total page count
                total_pages = len(doc)
                if total_pages > 0:
                    # For this example, we're assuming the page number is extracted from the first page
                    return 1  # For example, we return the first page number; adapt as needed
        except Exception as e:
            print(f"Error extracting page number from {pdf_file}: {e}")
        
        return None

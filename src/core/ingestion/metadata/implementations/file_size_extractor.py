from __future__ import annotations
import os
from pathlib import Path


class FileSizeExtractor:
    """Extracts the exact file size (in bytes) directly from the PDF file."""

    def __init__(self, base_dir: Path | str | None = None):
        self.base_dir = Path(base_dir).resolve() if base_dir else None

    # ------------------------------------------------------------------
    def extract(self, pdf_path: str) -> int | None:
        """Return the file size of the given PDF file in bytes."""
        pdf_file = Path(pdf_path)
        try:
            if not pdf_file.is_file():
                return None
            return pdf_file.stat().st_size
        except Exception:
            return None

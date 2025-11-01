# src/core/ingestion/ingestor.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional

from docling.document_converter import DocumentConverter
from src.core.ingestion.interfaces.i_ingestor import IIngestor


class Ingestor(IIngestor):
    # Handles PDF ingestion using Docling (no chunking)
    def __init__(self, enable_ocr: bool = True):
        self.enable_ocr = enable_ocr
        self.converter = DocumentConverter()

    def ingest(self, source_path: str) -> Dict[str, Any]:
        pdf_path = Path(source_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"File not found: {pdf_path}")

        # Perform robust conversion using Docling
        result = self.converter.convert(str(pdf_path))
        document = getattr(result, "document", None)

        # Extract plain text (fallback if needed)
        text = ""
        if document:
            if hasattr(document, "export_to_markdown"):
                text = document.export_to_markdown()
            elif hasattr(document, "text"):
                text = document.text

        # Build metadata
        metadata = {
            "source_file": pdf_path.name,
            "source_path": str(pdf_path),
            "page_count": self._get_page_count(document),
            "has_ocr": self._has_ocr(document),
            "toc": self._extract_toc(document),
        }

        # Single-chunk representation (no splitting yet)
        chunks = [{"text": text, "page": None, "bbox": None}] if text else []

        return {"metadata": metadata, "chunks": chunks}

    # -------------------------------------------------------
    def _get_page_count(self, document: Any) -> Optional[int]:
        if document is None:
            return None
        return len(getattr(document, "pages", []))

    def _has_ocr(self, document: Any) -> bool:
        if document is None:
            return False
        return bool(getattr(document, "ocr_content", None))

    def _extract_toc(self, document: Any) -> List[Dict[str, Any]]:
        toc: List[Dict[str, Any]] = []
        if document and hasattr(document, "outline_items") and document.outline_items:
            for item in document.outline_items:
                toc.append({
                    "title": getattr(item, "title", ""),
                    "page": getattr(item, "page_number", None),
                })
        return toc

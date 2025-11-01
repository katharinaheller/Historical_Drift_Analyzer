from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import fitz  # PyMuPDF
import re
from src.core.ingestion.parser.interfaces.i_pdf_parser import IPdfParser


class PyMuPDFParser(IPdfParser):
    """Robust PDF parser using PyMuPDF with layout-based filtering of non-body text."""

    def __init__(self, exclude_toc: bool = True, max_pages: int | None = None):
        self.exclude_toc = exclude_toc
        self.max_pages = max_pages

    # ------------------------------------------------------------------
    def parse(self, pdf_path: str) -> Dict[str, Any]:
        """Extract clean body text (excluding title, abstract, headers) and minimal metadata."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"File not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        text_blocks: List[str] = []
        toc_titles = self._extract_toc_titles(doc) if self.exclude_toc else []

        for page_index, page in enumerate(doc):
            if self.max_pages and page_index >= self.max_pages:
                break
            blocks = page.get_text("blocks")
            page_body = self._extract_body_from_blocks(blocks)
            if not page_body:
                continue
            # Skip ToC-like pages entirely
            if self.exclude_toc and self._looks_like_toc_page(page_body, toc_titles):
                continue
            text_blocks.append(page_body)

        clean_text = "\n".join(t for t in text_blocks if t)
        clean_text = self._remove_residual_metadata(clean_text)

        metadata = {
            "source_file": str(pdf_path.name),
            "page_count": len(doc),
            "has_toc": bool(doc.get_toc()),
        }

        doc.close()
        return {"text": clean_text.strip(), "metadata": metadata}

    # ------------------------------------------------------------------
    def _extract_toc_titles(self, doc) -> List[str]:
        """Extract TOC titles to later filter out from main text."""
        toc = doc.get_toc()
        return [entry[1].strip() for entry in toc if len(entry) >= 2]

    # ------------------------------------------------------------------
    def _looks_like_toc_page(self, text: str, toc_titles: List[str]) -> bool:
        """Heuristic to detect table of contents pages."""
        if not text or len(text) < 100:
            return False
        if re.search(r"(?i)\btable\s+of\s+contents\b|\binhaltsverzeichnis\b", text):
            return True
        match_count = sum(1 for t in toc_titles if t and t in text)
        return match_count > 5

    # ------------------------------------------------------------------
    def _extract_body_from_blocks(self, blocks: list) -> str:
        """Select only text blocks likely belonging to main body based on layout and heuristics."""
        # Determine median vertical position → ignore top 15–20%
        y_positions = [b[1] for b in blocks if len(b) >= 5]
        if not y_positions:
            return ""
        page_top_cutoff = sorted(y_positions)[int(len(y_positions) * 0.15)]
        candidate_blocks = []

        for (x0, y0, x1, y1, text, *_ ) in blocks:
            if not text or len(text.strip()) < 30:
                continue
            # Exclude upper-page fragments (titles, authors, abstract)
            if y1 < page_top_cutoff + 100:
                if re.search(r"(?i)(abstract|title|author|doi|keywords|arxiv|university|faculty|institute|version)", text):
                    continue
                if len(text.strip().split()) < 20:
                    continue
            candidate_blocks.append(text.strip())

        return "\n".join(candidate_blocks)

    # ------------------------------------------------------------------
    def _remove_residual_metadata(self, text: str) -> str:
        """Remove residual header/footer and metadata-like fragments."""
        patterns = [
            r"(?im)^table\s+of\s+contents.*$",
            r"(?im)^inhaltsverzeichnis.*$",
            r"(?im)^\s*(abstract|zusammenfassung)\b.*?(?=\n[A-Z][a-z]|$)",
            r"(?im)^title\s*:.*$",
            r"(?im)^author\s*:.*$",
            r"(?im)^\s*(version|revision|doi|arxiv).*?$",
            r"(?im)\bpage\s+\d+\b",
            r"(?m)^\s*\d+\s*$",
        ]
        for p in patterns:
            text = re.sub(p, "", text, flags=re.DOTALL)
        text = re.sub(r"\n{2,}", "\n\n", text)
        return text.strip()

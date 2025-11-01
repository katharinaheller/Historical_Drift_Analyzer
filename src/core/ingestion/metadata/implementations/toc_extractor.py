from __future__ import annotations
from typing import List, Dict, Any
import fitz
import re
from pathlib import Path


class TocExtractor:
    """Extracts table of contents (TOC) entries directly from a PDF using PyMuPDF."""

    def __init__(self, base_dir: Path | str | None = None):
        self.base_dir = Path(base_dir).resolve() if base_dir else None

    # ------------------------------------------------------------------
    def extract(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Return structured TOC entries, purely PDF-based."""
        pdf_file = Path(pdf_path)
        toc_entries: List[Dict[str, Any]] = []
        try:
            with fitz.open(pdf_file) as doc:
                toc = doc.get_toc(simple=True)
                if toc:
                    for level, title, page in toc:
                        title_clean = re.sub(r"\s+", " ", title.strip())
                        if title_clean:
                            toc_entries.append(
                                {"level": int(level), "title": title_clean, "page": int(page)}
                            )
                    return toc_entries

                # fallback to heuristic textual TOC
                toc_entries = self._extract_textual_toc(doc)
                return toc_entries

        except Exception as e:
            print(f"[WARN] Failed to extract TOC from {pdf_file}: {e}")
            return []

    # ------------------------------------------------------------------
    def _extract_textual_toc(self, doc: fitz.Document) -> List[Dict[str, Any]]:
        """Detect textual TOCs on the first pages when outline metadata is missing."""
        toc_entries: List[Dict[str, Any]] = []
        try:
            max_pages = min(5, len(doc))
            pattern = re.compile(r"^\s*(\d{0,2}\.?)+\s*[A-ZÄÖÜa-zäöü].{3,}\s+(\d{1,3})\s*$")

            for i in range(max_pages):
                text = doc.load_page(i).get_text("text")
                if not re.search(r"(Inhalt|Contents|Table of Contents)", text, re.I):
                    continue
                for line in text.splitlines():
                    if pattern.match(line.strip()):
                        parts = re.split(r"\s{2,}", line.strip())
                        if len(parts) >= 2:
                            title = re.sub(r"^\d+(\.\d+)*\s*", "", parts[0])
                            page = re.findall(r"\d+", parts[-1])
                            page_num = int(page[0]) if page else None
                            toc_entries.append(
                                {"level": 1, "title": title.strip(), "page": page_num}
                            )
            return toc_entries

        except Exception:
            return []

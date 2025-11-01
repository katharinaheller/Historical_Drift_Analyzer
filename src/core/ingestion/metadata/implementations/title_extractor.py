from __future__ import annotations
from typing import Any
import fitz
import re
from pathlib import Path
from lxml import etree


class TitleExtractor:
    """Extracts the main document title directly from the PDF or GROBID XML."""

    def __init__(self, base_dir: Path | str | None = None):
        self.base_dir = Path(base_dir).resolve() if base_dir else None

    # ------------------------------------------------------------------
    def extract(self, pdf_path: str) -> str | None:
        pdf_file = Path(pdf_path)
        title = self._extract_from_pdf_metadata(pdf_file)
        if title:
            return title

        # Try GROBID XML if available
        xml_path = self._find_grobid_xml(pdf_file)
        if xml_path and xml_path.exists():
            grobid_title = self._extract_from_grobid(xml_path)
            if grobid_title:
                return grobid_title

        # Fallback: filename heuristic
        return pdf_file.stem.replace("_", " ").strip()

    # ------------------------------------------------------------------
    def _extract_from_pdf_metadata(self, pdf_file: Path) -> str | None:
        """Extract title directly from PDF metadata using PyMuPDF."""
        try:
            with fitz.open(pdf_file) as doc:
                meta = doc.metadata or {}
                title = meta.get("title") or meta.get("Title")
                if title and len(title.strip()) > 2:
                    return title.strip()
        except Exception:
            return None
        return None

    # ------------------------------------------------------------------
    def _find_grobid_xml(self, pdf_file: Path) -> Path | None:
        """Find associated GROBID XML next to the PDF (same basename)."""
        xml_candidate = pdf_file.with_suffix(".tei.xml")
        if xml_candidate.exists():
            return xml_candidate
        if self.base_dir:
            alt = self.base_dir / "grobid_xml" / f"{pdf_file.stem}.tei.xml"
            if alt.exists():
                return alt
        return None

    # ------------------------------------------------------------------
    def _extract_from_grobid(self, xml_path: Path) -> str | None:
        """Extract title from GROBID TEI XML."""
        try:
            with open(xml_path, "r", encoding="utf-8") as f:
                xml = f.read()
            root = etree.fromstring(xml.encode("utf-8"))
            ns = {"tei": "http://www.tei-c.org/ns/1.0"}
            paths = [
                "//tei:analytic/tei:title[@type='main']",
                "//tei:monogr/tei:title[@type='main']",
                "//tei:titleStmt/tei:title[@type='main']",
                "//tei:titleStmt/tei:title",
            ]
            for p in paths:
                t = root.xpath(f"string({p})", namespaces=ns)
                if t and len(t.strip()) > 2:
                    return re.sub(r"\s+", " ", t.strip())
        except Exception:
            return None
        return None

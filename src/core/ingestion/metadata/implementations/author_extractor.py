from __future__ import annotations
from typing import List
from pathlib import Path
import fitz
import re
from lxml import etree


class AuthorsExtractor:
    """Extracts author names directly from PDF metadata or GROBID TEI XML."""

    def __init__(self, base_dir: Path | str | None = None):
        self.base_dir = Path(base_dir).resolve() if base_dir else None

    # ------------------------------------------------------------------
    def extract(self, pdf_path: str) -> List[str]:
        pdf_file = Path(pdf_path)

        # 1. Try GROBID XML if available
        xml_path = self._find_grobid_xml(pdf_file)
        if xml_path and xml_path.exists():
            authors = self._extract_from_grobid(xml_path)
            if authors:
                return authors

        # 2. Try from PDF metadata
        authors = self._extract_from_pdf_metadata(pdf_file)
        if authors:
            return authors

        return []

    # ------------------------------------------------------------------
    def _extract_from_pdf_metadata(self, pdf_file: Path) -> List[str]:
        """Read author field from XMP metadata."""
        try:
            with fitz.open(pdf_file) as doc:
                meta = doc.metadata or {}
                author_field = meta.get("author") or meta.get("Author") or meta.get("authors")
                if isinstance(author_field, str) and len(author_field.strip()) > 1:
                    parts = re.split(r"[;,/&]", author_field)
                    authors = [p.strip() for p in parts if len(p.strip()) > 1]
                    return authors
        except Exception:
            return []
        return []

    # ------------------------------------------------------------------
    def _find_grobid_xml(self, pdf_file: Path) -> Path | None:
        xml_candidate = pdf_file.with_suffix(".tei.xml")
        if xml_candidate.exists():
            return xml_candidate
        if self.base_dir:
            alt = self.base_dir / "grobid_xml" / f"{pdf_file.stem}.tei.xml"
            if alt.exists():
                return alt
        return None

    # ------------------------------------------------------------------
    def _extract_from_grobid(self, xml_path: Path) -> List[str]:
        """Parse TEI XML to extract author names."""
        ns = {"tei": "http://www.tei-c.org/ns/1.0"}
        try:
            with open(xml_path, "r", encoding="utf-8") as f:
                xml = f.read()
            root = etree.fromstring(xml.encode("utf-8"))
            authors: List[str] = []
            for node in root.xpath("//tei:author", namespaces=ns):
                first = node.xpath("string(.//tei:forename)", namespaces=ns).strip()
                last = node.xpath("string(.//tei:surname)", namespaces=ns).strip()
                if first or last:
                    authors.append(" ".join(x for x in [first, last] if x))
            seen = set()
            return [a for a in authors if not (a in seen or seen.add(a))]
        except Exception:
            return []

from __future__ import annotations
import re
import datetime
from typing import Any
from pathlib import Path
import fitz
from lxml import etree

CURRENT_YEAR = datetime.datetime.now().year


class YearExtractor:
    """Extracts publication year strictly from PDF or GROBID XML."""

    def __init__(self, base_dir: Path | str | None = None):
        self.base_dir = Path(base_dir).resolve() if base_dir else None

    # ------------------------------------------------------------------
    def extract(self, pdf_path: str) -> str | None:
        pdf_file = Path(pdf_path)

        # 1. Try GROBID XML
        xml_path = self._find_grobid_xml(pdf_file)
        if xml_path and xml_path.exists():
            year = self._extract_from_grobid(xml_path)
            if year:
                return year

        # 2. Try PDF metadata
        year = self._extract_from_pdf_metadata(pdf_file)
        if year:
            return year

        # 3. Try from filename
        m = re.search(r"(19|20)\d{2}", pdf_file.name)
        if m:
            y = int(m.group(0))
            if 1900 <= y <= CURRENT_YEAR:
                return str(y)

        return None

    # ------------------------------------------------------------------
    def _extract_from_pdf_metadata(self, pdf_file: Path) -> str | None:
        """Extract year directly from PDF XMP metadata."""
        try:
            with fitz.open(pdf_file) as doc:
                meta = doc.metadata or {}
                for key in ("creationDate", "modDate", "CreationDate", "date"):
                    val = meta.get(key)
                    if not val:
                        continue
                    m = re.search(r"(19|20)\d{2}", val)
                    if m:
                        y = int(m.group(0))
                        if 1900 <= y <= CURRENT_YEAR:
                            return str(y)
        except Exception:
            return None
        return None

    # ------------------------------------------------------------------
    def _find_grobid_xml(self, pdf_file: Path) -> Path | None:
        """Locate optional GROBID TEI XML next to the PDF or in grobid_xml/."""
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
        """Parse GROBID TEI XML for publication date."""
        try:
            with open(xml_path, "r", encoding="utf-8") as f:
                xml = f.read()
            root = etree.fromstring(xml.encode("utf-8"))
            ns = {"tei": "http://www.tei-c.org/ns/1.0"}
            years = root.xpath("//tei:sourceDesc//tei:date/@when", namespaces=ns)
            for y in years:
                y = y.strip()
                if len(y) >= 4 and y[:4].isdigit():
                    year = int(y[:4])
                    if 1900 <= year <= CURRENT_YEAR:
                        return str(year)
        except Exception:
            return None
        return None

from __future__ import annotations
import re
import datetime
import time
from typing import Iterable, Tuple, Optional, Dict, List
from pathlib import Path
import fitz  # PyMuPDF
from lxml import etree

# optional normalizer
try:
    from unidecode import unidecode
except Exception:
    unidecode = None

# optional Crossref client
try:
    from habanero import Crossref
except Exception:
    Crossref = None  # type: ignore

CURRENT_YEAR = datetime.datetime.now().year
FUTURE_GRACE = 1  # allow slight future offset for clock drift


class YearExtractor:
    """Highly robust publication year extractor using multi-source heuristics, TEI, and Crossref."""

    def __init__(self, base_dir: Path | str | None = None, max_text_pages: int = 3, enable_crossref: bool = True):
        self.base_dir = Path(base_dir).resolve() if base_dir else None
        self.max_text_pages = max(1, int(max_text_pages))
        self.crossref = None
        if enable_crossref and Crossref is not None:
            try:
                self.crossref = Crossref(mailto="contact@example.com")
            except Exception:
                self.crossref = None

    # ------------------------------------------------------------------
    def extract(self, pdf_path: str) -> Optional[str]:
        """Main orchestrator for multi-source year extraction."""
        pdf_file = Path(pdf_path)
        candidates: List[Tuple[int, int, str]] = []  # (priority, score, year_str)

        # 1) TEI XML (most reliable)
        xml_path = self._find_grobid_xml(pdf_file)
        if xml_path and xml_path.exists():
            year = self._extract_from_grobid(xml_path)
            if year:
                candidates.append((1, 100, year))

        # 2) PDF metadata (check & refine)
        year_meta, meta_score = self._extract_from_pdf_metadata(pdf_file)
        if year_meta:
            refined = self._refine_with_text_consistency(pdf_file, int(year_meta))
            if refined:
                candidates.append((2, meta_score, refined))

        # 3) Page text (explicit © or Published year)
        year_text, text_score = self._extract_from_page_text(pdf_file, self.max_text_pages)
        if year_text:
            candidates.append((3, text_score, year_text))

        # 4) Filename patterns / arXiv ID
        year_fn, fn_score = self._extract_from_filename(pdf_file.name)
        if year_fn:
            candidates.append((4, fn_score, year_fn))

        # 5) Optional DOI / Crossref fallback
        if not candidates:
            title, doi, arxiv = self._extract_title_doi_arxiv(pdf_file)
            if arxiv:
                y = self._year_from_arxiv_id(arxiv)
                if y:
                    candidates.append((5, 60, str(y)))
            if (doi or title) and self.crossref:
                y = self._lookup_via_crossref(title, doi)
                if y:
                    candidates.append((6, 70, y))

        if not candidates:
            return None

        # final prioritization
        candidates.sort(key=lambda t: (t[0], -t[1], -int(t[2])))
        best_year = candidates[0][2]
        return best_year

    # ------------------------------------------------------------------
    def _lookup_via_crossref(self, title: Optional[str], doi: Optional[str]) -> Optional[str]:
        """Crossref lookup (last resort)."""
        if not self.crossref:
            return None
        for attempt in range(2):
            try:
                if doi:
                    result = self.crossref.works(ids=doi, timeout=3)
                    msg = result.get("message", {})
                elif title:
                    result = self.crossref.works(query=title, limit=1, timeout=3)
                    msg = result.get("message", {}).get("items", [{}])[0]
                else:
                    return None

                for key in ("published-print", "published-online", "issued"):
                    info = msg.get(key)
                    if info and "date-parts" in info:
                        y = info["date-parts"][0][0]
                        if self._valid_year(y):
                            return str(y)
            except Exception:
                time.sleep(1)
        return None

    # ------------------------------------------------------------------
    def _extract_title_doi_arxiv(self, pdf_file: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract DOI, arXiv ID, and title from early pages."""
        doi_re = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
        arxiv_re = re.compile(r"\b(?:arxiv[:/ ]?)?(\d{4}\.\d{4,5}|[a-z\-]+/\d{7}|[0-9]{7,8})(v\d+)?\b", re.I)
        title = doi = arxiv = None

        try:
            with fitz.open(pdf_file) as doc:
                meta_title = (doc.metadata or {}).get("title")
                if meta_title:
                    title = meta_title.strip()
                text = ""
                for i in range(min(3, len(doc))):
                    text += (doc.load_page(i).get_text("text") or "") + "\n"
        except Exception:
            text = ""

        if unidecode and text:
            text = unidecode(text)

        m = doi_re.search(text)
        if m:
            doi = m.group(0).strip().rstrip(".,)")

        m = arxiv_re.search(text)
        if m:
            arxiv = m.group(1).strip()

        if not title and text:
            for line in text.splitlines():
                s = line.strip()
                if len(s) > 10 and not re.search(r"(abstract|introduction|contents)", s, re.I):
                    title = s
                    break

        return title, doi, arxiv

    # ------------------------------------------------------------------
    def _year_from_arxiv_id(self, arxiv_id: str) -> Optional[int]:
        """Infer year from arXiv ID pattern."""
        try:
            m = re.match(r"^(\d{2})(\d{2})\.\d{4,5}$", arxiv_id)
            if m:
                yy = int(m.group(1))
                year = 2000 + yy if yy < 25 else 1900 + yy
                return year if self._valid_year(year) else None
            m = re.match(r"^[a-z\-]+/(\d{2})(\d{2})\d{3,4}$", arxiv_id, re.I)
            if m:
                yy = int(m.group(1))
                year = 2000 + yy if yy < 25 else 1900 + yy
                return year if self._valid_year(year) else None
            m = re.match(r"^(\d{2})(\d{2})\d{3,4}$", arxiv_id)
            if m:
                yy = int(m.group(1))
                year = 2000 + yy if yy < 25 else 1900 + yy
                return year if self._valid_year(year) else None
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    def _valid_year(self, y: int) -> bool:
        """Check plausible range (1900–current+grace)."""
        return isinstance(y, int) and 1900 <= y <= (CURRENT_YEAR + FUTURE_GRACE)

    # ------------------------------------------------------------------
    def _find_grobid_xml(self, pdf_file: Path) -> Optional[Path]:
        """Find possible TEI XML companion file."""
        xml_candidate = pdf_file.with_suffix(".tei.xml")
        if xml_candidate.exists():
            return xml_candidate
        if self.base_dir:
            alt = self.base_dir / "grobid_xml" / f"{pdf_file.stem}.tei.xml"
            if alt.exists():
                return alt
        return None

    # ------------------------------------------------------------------
    def _extract_from_grobid(self, xml_path: Path) -> Optional[str]:
        """Parse TEI XML for earliest valid date."""
        try:
            xml_bytes = xml_path.read_bytes()
            root = etree.fromstring(xml_bytes)
            ns = {"tei": "http://www.tei-c.org/ns/1.0"}
            xpaths = [
                "//tei:sourceDesc//tei:imprint/tei:date",
                "//tei:profileDesc//tei:creation/tei:date",
                "//tei:biblStruct//tei:imprint/tei:date",
            ]
            for xp in xpaths:
                for node in root.xpath(xp, namespaces=ns):
                    for key in ("when", "when-iso", "notBefore", "notAfter"):
                        val = node.get(key)
                        if val:
                            y = self._year_from_date_string(val)
                            if y and self._valid_year(y):
                                return str(y)
                    text_val = (node.text or "").strip()
                    if text_val:
                        for y in self._years_from_string(text_val):
                            if self._valid_year(y):
                                return str(y)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    def _extract_from_pdf_metadata(self, pdf_file: Path) -> Tuple[Optional[str], int]:
        """Parse PDF metadata and filter out nonsense years (1970, 1980, 1600)."""
        try:
            with fitz.open(pdf_file) as doc:
                meta = doc.metadata or {}
                kv: Dict[str, str] = {k.lower(): v for k, v in meta.items() if isinstance(v, str) and v.strip()}
                best: Tuple[Optional[int], int] = (None, -1)
                for key, score in [("creationdate", 80), ("moddate", 60), ("date", 50)]:
                    val = kv.get(key)
                    if not val:
                        continue
                    y = self._year_from_date_string(val)
                    if not y or y in {0, 1, 1600, 1970, 1980}:
                        continue
                    if self._valid_year(y) and score > best[1]:
                        best = (y, score)
                if best[0]:
                    return str(best[0]), best[1]
        except Exception:
            pass
        return None, -1

    # ------------------------------------------------------------------
    def _refine_with_text_consistency(self, pdf_file: Path, year_meta: Optional[int]) -> Optional[str]:
        """Cross-check metadata with visible text for explicit © years."""
        try:
            with fitz.open(pdf_file) as doc:
                text_parts = []
                for i in range(min(3, len(doc))):
                    text_parts.append(doc.load_page(i).get_text("text") or "")
                for i in range(max(0, len(doc) - 2), len(doc)):
                    text_parts.append(doc.load_page(i).get_text("text") or "")
                text = "\n".join(text_parts)
        except Exception:
            return str(year_meta) if year_meta else None

        if unidecode and text:
            text = unidecode(text)

        # check for explicit © or "Published in"
        explicit_match = re.findall(r"(?:©|Copyright|Published\s+in|Reprinted\s+in)\s*(19|20)\d{2}", text, re.I)
        if explicit_match:
            ys = [int(y[-4:]) for y in re.findall(r"(19|20)\d{2}", text)]
            ys = [y for y in ys if self._valid_year(y)]
            if ys:
                y_max = max(ys)
                return str(y_max)

        # fallback: most recent plausible number
        years = [int(m.group()) for m in re.finditer(r"(19|20)\d{2}", text)]
        years = [y for y in years if self._valid_year(y)]
        if not years:
            return str(year_meta) if year_meta else None
        y_max = max(years)

        # adjust if metadata is suspiciously old
        if year_meta and (year_meta < 1950 or abs(year_meta - y_max) >= 10):
            return str(y_max)
        return str(year_meta if year_meta else y_max)

    # ------------------------------------------------------------------
    def _extract_from_page_text(self, pdf_file: Path, pages: int = 2) -> Tuple[Optional[str], int]:
        """Scan early pages for explicit publication indicators."""
        try:
            with fitz.open(pdf_file) as doc:
                best: Tuple[Optional[int], int] = (None, -1)
                for i in range(min(len(doc), pages)):
                    text = doc.load_page(i).get_text("text") or ""
                    if unidecode and text:
                        text = unidecode(text)
                    for pattern in [
                        r"(?:©|Copyright|Published\s+in|Reprinted\s+in)\s*(19|20)\d{2}",
                        r"(19|20)\d{2}",
                    ]:
                        years = [int(y[-4:]) for y in re.findall(pattern, text)]
                        years = [y for y in years if self._valid_year(y)]
                        if years:
                            y_pick = max(years)
                            score = 45 + (pages - i)
                            if score > best[1]:
                                best = (y_pick, score)
                if best[0]:
                    return str(best[0]), best[1]
        except Exception:
            pass
        return None, -1

    # ------------------------------------------------------------------
    def _extract_from_filename(self, filename: str) -> Tuple[Optional[str], int]:
        """Infer from filename (e.g., arXiv IDs, embedded years)."""
        base = unidecode(filename) if unidecode else filename
        base = re.sub(r"v\d+\b", "", base, flags=re.I)

        m = re.search(r"\b(\d{4}\.\d{4,5})\b", base)
        if m:
            y = self._year_from_arxiv_id(m.group(1))
            if y:
                return str(y), 50

        m = re.search(r"\b([a-z\-]+/\d{7,8}|\d{7,8})\b", base, re.I)
        if m:
            y = self._year_from_arxiv_id(m.group(1))
            if y:
                return str(y), 45

        years = [y for y in self._years_from_string(base) if self._valid_year(y) and y != 1970]
        if years:
            return str(max(years)), 30
        return None, -1

    # ------------------------------------------------------------------
    def _year_from_date_string(self, s: str) -> Optional[int]:
        """Parse ISO/XMP date strings like D:YYYYMMDD or YYYY-MM-DD."""
        if not s:
            return None
        s = s.strip()
        m = re.match(r"^D:(\d{4})", s)
        if m:
            y = int(m.group(1))
            return y if self._valid_year(y) else None
        m = re.match(r"^(\d{4})(?:[-/]\d{2}(?:[-/]\d{2})?)?$", s)
        if m:
            y = int(m.group(1))
            return y if self._valid_year(y) else None
        for y in self._years_from_string(s):
            if self._valid_year(y):
                return y
        return None

    # ------------------------------------------------------------------
    def _years_from_string(self, s: str) -> Iterable[int]:
        """Yield distinct plausible years in order."""
        seen = set()
        for m in re.finditer(r"(?<!\d)(19|20)\d{2}(?!\d)", s):
            y = int(m.group(0))
            if y not in seen:
                seen.add(y)
                yield y

from __future__ import annotations
from pathlib import Path
from typing import Optional
import re
import fitz

try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


class DetectedLanguageExtractor:
    """Detects the dominant document language from PDF text or XMP metadata."""

    def __init__(self, base_dir: Path | str | None = None):
        self.base_dir = Path(base_dir).resolve() if base_dir else None

    # ------------------------------------------------------------------
    def extract(self, pdf_path: str) -> Optional[str]:
        """Identify document language via PDF content or metadata."""
        pdf_file = Path(pdf_path)

        # 1. Try PDF XMP metadata
        lang = self._extract_from_metadata(pdf_file)
        if lang:
            return lang

        # 2. Try text-based detection (without parser)
        lang = self._detect_from_text(pdf_file)
        if lang:
            return lang

        return None

    # ------------------------------------------------------------------
    def _extract_from_metadata(self, pdf_file: Path) -> Optional[str]:
        """Check embedded language fields in PDF metadata."""
        try:
            with fitz.open(pdf_file) as doc:
                meta = doc.metadata or {}
                for key in ("language", "Language", "lang", "Lang"):
                    val = meta.get(key)
                    if val and isinstance(val, str):
                        val_norm = val.lower().strip()
                        if val_norm.startswith("en"):
                            return "en"
                        if val_norm.startswith("de"):
                            return "de"
        except Exception:
            return None
        return None

    # ------------------------------------------------------------------
    def _detect_from_text(self, pdf_file: Path) -> Optional[str]:
        """Extract small text sample from first pages and apply language detection."""
        sample_text = ""
        try:
            with fitz.open(pdf_file) as doc:
                max_pages = min(3, len(doc))
                for i in range(max_pages):
                    sample_text += doc.load_page(i).get_text("text") + "\n"
        except Exception:
            return None

        if not sample_text or len(sample_text.strip()) < 30:
            return None

        # Use langdetect if installed
        if LANGDETECT_AVAILABLE:
            try:
                lang = detect(sample_text)
                if lang in ("en", "de"):
                    return lang
            except Exception:
                pass

        # Simple regex-based fallback
        text_lower = sample_text.lower()
        if re.search(r"\b(und|nicht|sein|dass|werden|ein|eine|mit|auf|für|dem|den|das|ist|sind|der|die|das)\b", text_lower):
            return "de"
        if re.search(r"\b(the|and|of|for|with|from|that|this|by|is|are|we|deep|learning|language|model)\b", text_lower):
            return "en"
        return None

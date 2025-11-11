from __future__ import annotations
import re
from typing import List
from src.core.ingestion.cleaner.base_cleaner import BaseTextCleaner
from src.core.ingestion.cleaner import rules


class HTMLCleaner(BaseTextCleaner):
    """Removes HTML tags and entities like &nbsp;."""

    def _clean_impl(self, text: str) -> str:
        # Remove all HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove HTML entities like &nbsp;
        text = re.sub(r'&[a-zA-Z]+;', '', text)
        return text


class ScientificNotationCleaner(BaseTextCleaner):
    """Removes scientific notations and references like 'Eq. 1', 'Theorem 3'."""

    def _clean_impl(self, text: str) -> str:
        # Remove scientific expressions such as Eq. 1 or Lemma 3
        text = re.sub(r'\b(Eq|Theorem|Lemma)\s+\d+\b', '', text)
        return text


class UnicodeNormalizer(BaseTextCleaner):
    """Normalize unicode spaces and zero-width chars."""

    def _clean_impl(self, text: str) -> str:
        # Replace unicode spaces with a normal space
        text = rules.UNICODE_SPACES_RE.sub(" ", text)
        return text


class SoftHyphenCleaner(BaseTextCleaner):
    """Remove soft hyphens and join line-broken words."""

    def _clean_impl(self, text: str) -> str:
        # Remove soft hyphen character
        text = text.replace(rules.SOFT_HYPHEN, "")
        # Join words split by hyphen and linebreak
        text = rules.HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)
        return text


class LayoutLineJoinCleaner(BaseTextCleaner):
    """
    Join lines that were broken by the PDF layout but belong together.
    """

    def _clean_impl(self, text: str) -> str:
        # Join inline linebreaks like "probabilistic\nreasoning" -> "probabilistic reasoning"
        text = rules.INLINE_LINEBREAK_RE.sub(r"\1 \2", text)
        # Collapse too many blank lines
        text = rules.MULTI_NEWLINE_RE.sub("\n\n", text)
        return text.strip()


class HeaderFooterCleaner(BaseTextCleaner):
    """Remove obvious header/footer/single-line noise."""

    def _clean_impl(self, text: str) -> str:
        cleaned_lines: List[str] = []
        for line in text.splitlines():
            raw = line.rstrip()

            # Skip headers
            if rules.match_any(rules.SINGLE_LINE_HEADER_PATTERNS, raw):
                continue

            # Skip short footers
            if rules.match_any(rules.FOOTER_PATTERNS, raw) and rules.is_short_line(raw):
                continue

            # Skip short funding or preprint notices
            if rules.match_any(rules.FUNDING_PATTERNS, raw) and rules.is_short_line(raw):
                continue

            cleaned_lines.append(raw)

        return "\n".join(cleaned_lines).strip()


class TrailingWhitespaceCleaner(BaseTextCleaner):
    """Normalize whitespace globally."""

    def _clean_impl(self, text: str) -> str:
        # Strip trailing spaces per line
        lines = [ln.rstrip() for ln in text.splitlines()]
        text = "\n".join(lines)
        # Collapse 3+ newlines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


# ----------------------------------------------------------------------
# NEW CLEANER: removes APA / IEEE / arXiv / ACM / MLA style in-text citations
# ----------------------------------------------------------------------
class ReferencePatternCleaner(BaseTextCleaner):
    """
    Removes inline reference patterns such as (Smith, 2020), [12], [1–5], (Smith et al., 2021),
    DOIs, arXiv IDs, and conference citation snippets (Proc. IEEE, JMLR, NeurIPS, etc.).
    """

    def _clean_impl(self, text: str) -> str:
        # APA/Harvard-style: (Smith, 2020), (Doe & Roe, 2021)
        text = re.sub(
            r"\([A-Z][A-Za-z\-]+(?:,?\s(?:[A-Z]\.)+)*(?:\s&\s[A-Z][A-Za-z\-]+)*,\s?\d{4}[a-z]?\)",
            "",
            text,
        )

        # et al. style: (Smith et al., 2020)
        text = re.sub(
            r"\([A-Z][A-Za-z\-]+\s+et\s+al\.,\s*\d{4}[a-z]?\)",
            "",
            text,
        )

        # IEEE numeric: [1], [12], [1,2], [1–5], [1-3]
        text = re.sub(
            r"\[\s?\d+(?:[\-,–]\s?\d+)*(?:\s*,\s*\d+)*\s?\]",
            "",
            text,
        )

        # Year-only: (2020), (1999a)
        text = re.sub(
            r"\(\s?\d{4}[a-z]?\s?\)",
            "",
            text,
        )

        # Inline DOIs and arXiv references
        text = re.sub(r"\bdoi:\s*\S+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\barXiv:\s*\S+", "", text, flags=re.IGNORECASE)

        # Conference/journal abbreviations with year numbers
        text = re.sub(
            r"\b(Proc\.|Proceedings|Conf\.|Conference|JMLR|ICML|NeurIPS|NIPS|AAAI|IJCAI|ACL|EMNLP|COLING|IEEE|ACM)\b.*?\d{4}",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # Author-name style in brackets: [Smith 2020]
        text = re.sub(
            r"\[[A-Z][A-Za-z\-]+(?:\s+et\s+al\.)?,?\s*\d{4}[a-z]?\]",
            "",
            text,
        )

        # “In Proceedings of …” phrases
        text = re.sub(
            r"(?i)\bin\s+proceedings\s+of\b.*?(?=[\.\n])",
            "",
            text,
        )

        # Normalize multiple spaces and punctuation spacing
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"\s+([\.,;:])", r"\1", text)

        return text.strip()


class ReferencesCleaner(BaseTextCleaner):
    """Removes everything starting from 'References', 'Bibliography', or 'Literaturverzeichnis'."""

    def _clean_impl(self, text: str) -> str:
        pattern = re.compile(
            r"(?im)^\s*(references|bibliography|literaturverzeichnis)\s*$"
        )
        match = pattern.search(text)
        if match:
            cutoff_index = match.start()
            # Only cut if reference section is beyond 20% of text
            if cutoff_index > len(text) * 0.2:
                text = text[:cutoff_index]
        return text.strip()

from __future__ import annotations
import re
from typing import List
from src.core.ingestion.cleaner.base_cleaner import BaseTextCleaner
from src.core.ingestion.cleaner import rules


class UnicodeNormalizer(BaseTextCleaner):
    """Normalize unicode spaces and zero-width chars."""

    def _clean_impl(self, text: str) -> str:
        # Replace unicode spaces by normal space
        text = rules.UNICODE_SPACES_RE.sub(" ", text)
        return text


class SoftHyphenCleaner(BaseTextCleaner):
    """Remove soft hyphens and join line-broken words."""

    def _clean_impl(self, text: str) -> str:
        # Remove soft hyphen character
        text = text.replace(rules.SOFT_HYPHEN, "")
        # Join hyphenated line breaks
        text = rules.HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)
        return text


class LayoutLineJoinCleaner(BaseTextCleaner):
    """
    Join lines that were broken by the PDF layout but belong together.
    We do this conservatively: only if both sides are non-space.
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

            # Drop header-like lines
            if rules.match_any(rules.SINGLE_LINE_HEADER_PATTERNS, raw):
                continue

            # Drop footer-like things only if short
            if rules.match_any(rules.FOOTER_PATTERNS, raw) and rules.is_short_line(raw):
                continue

            # Drop funding/preprint if short
            if rules.match_any(rules.FUNDING_PATTERNS, raw) and rules.is_short_line(raw):
                continue

            cleaned_lines.append(raw)

        return "\n".join(cleaned_lines).strip()


class TrailingWhitespaceCleaner(BaseTextCleaner):
    """Normalize whitespace globally."""

    def _clean_impl(self, text: str) -> str:
        # Strip each line
        lines = [ln.rstrip() for ln in text.splitlines()]
        text = "\n".join(lines)
        # Final collapse of multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

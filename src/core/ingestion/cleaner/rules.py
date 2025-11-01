# src/core/ingestion/cleaner/rules.py
from __future__ import annotations
import re

# Common residual headings we want to drop only if they appear alone in a line
SINGLE_LINE_HEADER_PATTERNS = [
    r"(?i)^\s*table\s+of\s+contents\s*$",
    r"(?i)^\s*inhaltsverzeichnis\s*$",
    r"(?i)^\s*contents\s*$",
]

# Footer / page indicator
FOOTER_PATTERNS = [
    r"(?i)^\s*page\s+\d+\s*$",
    r"(?i)^\s*p\.?\s*\d+\s*$",
    r"^\s*\d+\s*$",
]

# Funding and preprint noise – but only if line is short
FUNDING_PATTERNS = [
    r"(?i)^\s*funded\s+by.*$",
    r"(?i)^\s*supported\s+by.*$",
    r"(?i)^\s*this\s+preprint\s+.*$",
    r"(?i)^\s*arxiv:\s*\d{4}\.\d{4,5}.*$",
    r"(?i)^\s*doi:\s*10\.\d{4,9}/[-._;()/:A-Z0-9]+$",
]

# Lines we consider layout trash if they are too short
MAX_LEN_FOR_STRICT_DROP = 80

SOFT_HYPHEN = "\xad"
SOFT_HYPHEN_RE = re.compile(SOFT_HYPHEN)

# Hyphenated line break like "prob-\nabilistic"
HYPHEN_LINEBREAK_RE = re.compile(r"(\w+)-\n(\w+)")

# Linebreak in the middle of sentence like "proba-\nbilistic"
INLINE_LINEBREAK_RE = re.compile(r"(\S)\n(\S)")

# Multiple blank lines
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

# Unicode spaces
UNICODE_SPACES_RE = re.compile(r"[\u00a0\u2000-\u200b]+")


def is_short_line(line: str) -> bool:
    # Heuristic: many layout lines are short
    return len(line.strip()) <= MAX_LEN_FOR_STRICT_DROP


def match_any(patterns: list[str], line: str) -> bool:
    for pat in patterns:
        if re.match(pat, line):
            return True
    return False

# src/core/ingestion/cleaner/rag_text_cleaner.py
from __future__ import annotations
from typing import List
from src.core.ingestion.cleaner.i_text_cleaner import ITextCleaner
from src.core.ingestion.cleaner.simple_cleaners import (
    UnicodeNormalizer,
    SoftHyphenCleaner,
    HeaderFooterCleaner,
    LayoutLineJoinCleaner,
    TrailingWhitespaceCleaner,
)


class RagTextCleaner(ITextCleaner):
    """
    Composite text cleaner for RAG ingestion.
    Runs several deterministic cleaners in a fixed order.
    """

    def __init__(self, cleaners: List[ITextCleaner]):
        self.cleaners = cleaners

    @classmethod
    def default(cls) -> "RagTextCleaner":
        # Order is important
        cleaners: List[ITextCleaner] = [
            UnicodeNormalizer(),       # normalize spaces and zero-width
            SoftHyphenCleaner(),       # remove soft hyphen and join "foo-\nbar"
            HeaderFooterCleaner(),     # drop obvious non-flow lines
            LayoutLineJoinCleaner(),   # fix line breaks
            TrailingWhitespaceCleaner()  # final formatting
        ]
        return cls(cleaners)

    def clean(self, text: str) -> str:
        # Run all cleaners consecutively
        for cleaner in self.cleaners:
            text = cleaner.clean(text)
        return text

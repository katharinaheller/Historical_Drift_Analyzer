from __future__ import annotations
from typing import final
from src.core.ingestion.cleaner.i_text_cleaner import ITextCleaner


class BaseTextCleaner(ITextCleaner):
    """Base class providing a stable clean(...) interface."""

    @final
    def clean(self, text: str) -> str:
        # Guarantee non-null string
        if not text:
            return ""
        return self._clean_impl(text)

    def _clean_impl(self, text: str) -> str:
        # Must be implemented by subclasses
        raise NotImplementedError

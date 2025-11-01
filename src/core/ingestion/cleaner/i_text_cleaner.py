from __future__ import annotations
from abc import ABC, abstractmethod


class ITextCleaner(ABC):
    """Interface for all text cleaners in the ingestion pipeline."""

    @abstractmethod
    def clean(self, text: str) -> str:
        """Return a cleaned version of the given text."""
        raise NotImplementedError

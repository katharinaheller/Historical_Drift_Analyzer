from abc import ABC, abstractmethod
from typing import Any, Dict

class ILanguageDetector(ABC):
    """Interface for detecting the main document language."""

    @abstractmethod
    def detect(self, text: str, metadata: Dict[str, Any] | None = None) -> str | None:
        """Detect the dominant language code (e.g. 'en', 'de')."""
        raise NotImplementedError

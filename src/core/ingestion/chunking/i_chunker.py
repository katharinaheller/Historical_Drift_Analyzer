from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class IChunker(ABC):
    """Interface for all chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Split cleaned text into small units.
        Each returned item should at least have: {"text": ..., "metadata": {...}}
        """
        raise NotImplementedError

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class IRetriever(ABC):
    """Interface for all retriever implementations."""
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # Return top-k relevant documents/chunks for a query
        pass

    @abstractmethod
    def close(self) -> None:
        # Release resources if necessary
        pass

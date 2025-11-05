from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class IReranker(ABC):
    """Interface for reranker strategies (temporal, semantic, hybrid, etc.)."""
    @abstractmethod
    def rerank(self, results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        # Rerank a list of retrieval results based on a custom strategy
        pass

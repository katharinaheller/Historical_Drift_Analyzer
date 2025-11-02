from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict


class ILLM(ABC):
    """Interface for any local or remote LLM backend."""

    @abstractmethod
    def generate(self, prompt: str, context: List[Dict[str, str]]) -> str:
        """Generate an answer given a prompt and retrieved context chunks."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Gracefully close model connection or session."""
        raise NotImplementedError

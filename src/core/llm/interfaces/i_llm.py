from __future__ import annotations
from abc import ABC, abstractmethod


class ILLM(ABC):
    """Interface for any local or remote LLM backend."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate an answer given a COMPLETE prompt.

        Der Prompt enthält:
        - Systeminstruktion,
        - User-Query,
        - eingebettete Kontextsnippets (falls RAG).
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Gracefully close model connection or session."""
        raise NotImplementedError

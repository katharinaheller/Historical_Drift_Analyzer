from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class IChunker(ABC):
    """Interface for all chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split the cleaned text into smaller chunks.
        
        Each returned item should be a dictionary with:
        - "text": the chunked text as a string.
        - "metadata": additional information (e.g., document info, chunking context).
        
        Args:
            text: The cleaned text to be chunked.
            metadata: Optional metadata related to the chunking process. Default is an empty dictionary.
        
        Returns:
            A list of dictionaries, each containing:
            - "text": A chunk of the input text.
            - "metadata": The metadata associated with the chunk.
        """
        if metadata is None:
            metadata = {}
        raise NotImplementedError

# src/core/ingestion/chunking/static_chunker.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from src.core.ingestion.chunking.i_chunker import IChunker
import spacy

class StaticChunker(IChunker):
    """Chunker that uses fixed chunk size and overlap for chunking."""

    def __init__(self, 
                 chunk_size: int,  # Configuration-based chunk size
                 overlap: int, 
                 min_chunk_length: int):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_length = min_chunk_length

        # Initialize spaCy NLP model for sentence segmentation
        self.nlp = spacy.load("en_core_web_sm")

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Split text into overlapping fixed-size chunks with sentence boundaries."""
        chunks = []
        current_chunk = ""

        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                current_chunk += ". " + sentence
            else:
                chunks.append({
                    "text": current_chunk.strip(),
                    "chunk_size": len(current_chunk.strip()),  # Display actual chunk length
                    "configured_chunk_size": self.chunk_size,   # Config value for comparison
                    "overlap": self.overlap                    # Configured overlap
                })
                current_chunk = sentence

            # Merge small chunks if necessary
            if len(current_chunk) < self.min_chunk_length and chunks:
                last_chunk = chunks[-1]
                last_chunk["text"] += " " + current_chunk
                last_chunk["chunk_size"] = len(last_chunk["text"])
                current_chunk = ""

        if current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "chunk_size": len(current_chunk.strip()),
                "configured_chunk_size": self.chunk_size,
                "overlap": self.overlap
            })

        return chunks

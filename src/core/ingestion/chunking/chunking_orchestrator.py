from __future__ import annotations
import logging
from typing import Any, Dict, Optional
from src.core.ingestion.chunking.i_chunker import IChunker
from src.core.ingestion.chunking.adaptive_chunker import AdaptiveChunker
from src.core.ingestion.chunking.static_chunker import StaticChunker
from src.core.ingestion.metadata.implementations.page_number_extractor import PageNumberExtractor


class ChunkingOrchestrator:
    """Handles the selection and execution of chunking strategies based on YAML config."""

    def __init__(self, config: Dict[str, Any], pdf_path: Optional[str] = None):
        """Initialize with YAML configuration and an optional PDF path."""
        self.config = config
        self.pdf_path = pdf_path
        
        # Überprüfen, ob der 'chunking' Abschnitt in der Konfiguration vorhanden ist
        if "chunking" not in self.config:
            raise KeyError("The 'chunking' section is missing in the configuration file")

        # Debugging: Gibt den 'chunking' Abschnitt aus
        logging.debug(f"Chunking config: {self.config['chunking']}")

        # Wählt die passende Chunking-Strategie aus der Konfiguration
        self.chunker = self.select_chunker()  

    def select_chunker(self) -> IChunker:
        """Select chunking strategy based on the configuration."""
        chunking_config = self.config["chunking"]  # Zugriff auf den 'chunking'-Abschnitt der Konfiguration
        chunking_mode = chunking_config["mode"]
        chunk_size = chunking_config["chunk_size"]  # Get the chunk size from config
        overlap = chunking_config["overlap"]
        enable_overlap = chunking_config["enable_overlap"]
        min_chunk_length = chunking_config["min_chunk_length"]
        sentence_boundary_detection = chunking_config["sentence_boundary_detection"]
        merge_short_chunks = chunking_config["merge_short_chunks"]

        # Erstelle die Chunker-Instanz basierend auf dem gewählten Modus
        if chunking_mode == "adaptive":
            # Instanziiere AdaptiveChunker mit den relevanten Konfigurationswerten
            page_number_extractor = PageNumberExtractor() if self.pdf_path else None
            return AdaptiveChunker(
                chunk_size=chunk_size,
                overlap=overlap if enable_overlap else 0,
                min_chunk_length=min_chunk_length,
                pdf_path=self.pdf_path,
                page_number_extractor=page_number_extractor,
            )
        elif chunking_mode == "static":
            # Instanziiere StaticChunker mit den relevanten Konfigurationswerten
            return StaticChunker(
                chunk_size=chunk_size,  # Pass the chunk_size from config
                overlap=overlap if enable_overlap else 0,
                min_chunk_length=min_chunk_length,
            )
        else:
            raise ValueError(f"Unknown chunking strategy: {chunking_mode}")

    def process(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Process the text and return chunks with metadata using the selected chunking strategy."""
        return self.chunker.chunk(text, metadata)

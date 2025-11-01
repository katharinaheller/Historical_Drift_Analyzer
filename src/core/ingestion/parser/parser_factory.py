from __future__ import annotations
from typing import Dict, Any, Optional
import logging
import multiprocessing

from src.core.ingestion.parser.interfaces.i_pdf_parser import IPdfParser
from src.core.ingestion.parser.pymupdf_parser import PyMuPDFParser


class ParserFactory:
    """
    Factory for creating local PDF parsers and optional parallel orchestrators.
    Handles YAML parameters like 'parallelism' and parser configuration.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        opts = config.get("options", {})

        self.parser_mode = opts.get("pdf_parser", "auto").lower()
        self.parallelism = opts.get("parallelism", "auto")
        self.language = opts.get("language", "auto")
        self.exclude_toc = True  # enforced globally
        self.max_pages = opts.get("max_pages", None)

        # determine optimal CPU usage
        if isinstance(self.parallelism, int) and self.parallelism > 0:
            self.num_workers = self.parallelism
        else:
            self.num_workers = max(1, multiprocessing.cpu_count() - 1)

        self.logger.info(
            f"ParserFactory initialized | mode={self.parser_mode}, "
            f"workers={self.num_workers}, exclude_toc={self.exclude_toc}"
        )

    # ------------------------------------------------------------------
    def create_parser(self) -> IPdfParser:
        """Return configured parser instance (currently only PyMuPDF)."""
        if self.parser_mode in ("fitz", "auto"):
            return PyMuPDFParser(exclude_toc=self.exclude_toc, max_pages=self.max_pages)
        raise ValueError(f"Unsupported parser mode: {self.parser_mode}")

    # ------------------------------------------------------------------
    def create_parallel_parser(self):
        """Return a parallel orchestrator that distributes parsing tasks."""
        from src.core.ingestion.parser.parallel_pdf_parser import ParallelPdfParser
        return ParallelPdfParser(self.config, logger=self.logger)

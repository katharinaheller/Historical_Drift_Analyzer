from __future__ import annotations
from typing import Any, Dict, List
import importlib
import logging
from pathlib import Path


class MetadataExtractorFactory:
    """
    Dynamically loads and executes PDF-based metadata extractors.
    Each extractor implementation must expose a class named <Name>Extractor in
    src.core.ingestion.metadata.implementations.<name>_extractor.
    """

    def __init__(self, active_fields: List[str], pdf_root: str | Path):
        self.logger = logging.getLogger(__name__)
        self.active_fields = active_fields
        self.pdf_root = Path(pdf_root).resolve()
        self.extractors: Dict[str, Any] = {}
        self._load_extractors()

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> MetadataExtractorFactory:
        """Initialize factory from ingestion.yaml configuration."""
        opts = cfg.get("options", {})
        active = opts.get("metadata_fields", [])
        pdf_root = cfg.get("paths", {}).get("raw_pdfs", "./data/raw_pdfs")
        return cls(active_fields=active, pdf_root=pdf_root)

    # ------------------------------------------------------------------
    def _load_extractors(self) -> None:
        """Dynamically import extractor implementations for requested fields."""
        base_pkg = "src.core.ingestion.metadata.implementations"

        mapping = {
            "title": "title_extractor",
            "authors": "author_extractor",
            "year": "year_extractor",
            # "abstract": "abstract_extractor",  # Deactivated abstract extraction
            "detected_language": "language_detector",
            "file_size": "file_size_extractor",
            "toc": "toc_extractor",
            "page_number": "page_number_extractor",  # Page number extractor
        }

        for field in self.active_fields:
            module_name = mapping.get(field)
            if not module_name:
                self.logger.warning(f"No extractor mapping found for field '{field}' → skipped.")
                continue

            try:
                # Dynamically import the module
                module = importlib.import_module(f"{base_pkg}.{module_name}")
                # Construct the extractor class name from the field (e.g., "PageNumberExtractor" from "page_number")
                class_name = "".join([p.capitalize() for p in field.split("_")]) + "Extractor"
                # Get the extractor class from the module
                extractor_class = getattr(module, class_name)
                # Instantiate the extractor class and store it in the dictionary
                self.extractors[field] = extractor_class(base_dir=self.pdf_root)
                self.logger.debug(f"Loaded extractor: {extractor_class.__name__} for '{field}'")
            except ModuleNotFoundError:
                self.logger.warning(f"Implementation missing for '{field}' ({module_name}.py not found).")
            except Exception as e:
                self.logger.error(f"Failed to load extractor for '{field}': {e}")

    # ------------------------------------------------------------------
    def extract_all(self, pdf_path: str) -> Dict[str, Any]:
        """Run all configured extractors directly on a given PDF file path."""
        pdf_path = str(Path(pdf_path).resolve())
        results: Dict[str, Any] = {}

        for field, extractor in self.extractors.items():
            try:
                value = extractor.extract(pdf_path)
                results[field] = value
            except Exception as e:
                self.logger.warning(f"Extractor '{field}' failed for {pdf_path}: {e}")
                results[field] = None

        return results
